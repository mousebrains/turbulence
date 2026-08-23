# Aug-2026, Claude and Pat Welch, pat@mousebrains.com
"""Synthetic glider deployments with known answers.

There is no glider ``.p`` file, no ``hotel.nc`` and no ``.ebd`` in the repo
(plan A11), and "the coefficients came out plausible" is not a test.  This
module builds a deployment from *known* Steinhart-Hart coefficients, a *known*
clock offset, a *known* CTD response and an optional *known* ``t_0`` drift, so
the estimators can be checked against truth rather than against themselves.

It also models the thing that motivates the whole design: the CTD is enabled on
only every n-th yo, so the reference is genuinely absent over most of the
record --- not sparse, not noisy, absent.

The two lags are modelled separately on purpose, because the plan claims they
are separable (A9 / R8):

``clock_offset``
    The glider computer's clock against the instrument's.  Shifts the CTD's
    temperature *and* pressure timestamps equally.
``ctd_delay`` + ``ctd_tau``
    Thermal and plumbing response.  Affects temperature only.

So ``estimate_clock_offset`` should recover ``clock_offset`` alone, while the
temperature lag recovers roughly ``clock_offset + ctd_delay`` plus the single
pole's group delay.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from odas_tpw.fp07cal.logr import BridgeParams
from odas_tpw.fp07cal.pairs import _single_pole
from odas_tpw.fp07cal.series import ProbeSeries, ReferenceSeries, sanitize_reference

# A plausible FP07 half-bridge, in the shape PFile hands us.
DEFAULT_BRIDGE = BridgeParams(a=0.0, b=1.0, g=6.0, e_b=0.68, adc_fs=4.096, adc_bits=16)

# Order-1 Steinhart-Hart truth: t_0 = 288.0 K, beta_1 = 3100 K.
DEFAULT_COEFFS = np.array([1.0 / 288.0, 1.0 / 3100.0])


@dataclass
class SynthConfig:
    n_yos: int = 24
    yo_seconds: float = 1200.0
    fs: float = 16.0
    P_max: float = 180.0
    T_surface: float = 21.0
    T_deep: float = 13.0
    thermocline_dbar: float = 60.0
    files_per_deployment: int = 6

    ct_every_n: int = 3
    """CT enabled on every n-th yo.  The whole point of the exercise."""

    ref_rate: float = 1.0
    clock_offset: float = 0.0
    ctd_delay: float = 0.0
    ctd_tau: float = 0.5

    probe_noise_K: float = 0.002
    ref_noise_K: float = 0.002
    drift_K_per_day: float = 0.0
    """Applied as a slow ``t_0`` walk: a pure offset, no slope change."""

    seed: int = 12345
    start_epoch: float = 1.7e9


def _water_temperature(P: np.ndarray, cfg: SynthConfig) -> np.ndarray:
    """A monotone thermocline — temperature is a function of depth alone."""
    span = cfg.T_surface - cfg.T_deep
    return cfg.T_deep + span * (1.0 - np.tanh(P / cfg.thermocline_dbar))


def _pressure_track(
    cfg: SynthConfig,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]], np.ndarray]:
    """Triangle-wave yos.  Returns (t_rel, P, profile spans, yo index per sample)."""
    dt = 1.0 / cfg.fs
    n_per = round(cfg.yo_seconds / dt)
    half = n_per // 2
    t_rel: list[np.ndarray] = []
    P: list[np.ndarray] = []
    spans: list[tuple[int, int]] = []
    yo_of: list[np.ndarray] = []
    cursor = 0
    for k in range(cfg.n_yos):
        down = np.linspace(0.0, cfg.P_max, half, endpoint=False)
        up = np.linspace(cfg.P_max, 0.0, n_per - half, endpoint=False)
        p = np.concatenate([down, up])
        t = (cursor + np.arange(p.size)) * dt
        spans.append((cursor, cursor + half - 1))          # dive
        spans.append((cursor + half, cursor + p.size - 1))  # climb
        yo_of.append(np.full(p.size, k, dtype=np.int32))
        t_rel.append(t)
        P.append(p)
        cursor += p.size
    return (
        np.concatenate(t_rel),
        np.concatenate(P),
        spans,
        np.concatenate(yo_of),
    )


def _invert_to_L(T_C: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    """Solve ``sum a_i L^i = 1/T_K`` for L (Newton, seeded from the linear term)."""
    target = 1.0 / (np.asarray(T_C, dtype=np.float64) + 273.15)
    L = np.asarray((target - coeffs[0]) / coeffs[1], dtype=np.float64)
    if coeffs.size == 2:
        return L
    for _ in range(40):
        f = np.zeros_like(L)
        df = np.zeros_like(L)
        for i, a in enumerate(coeffs):
            f = f + a * L**i
            if i >= 1:
                df = df + i * a * L ** (i - 1)
        step = (f - target) / np.where(df == 0, np.nan, df)
        L = L - step
        if np.nanmax(np.abs(step)) < 1e-14:
            break
    return np.asarray(L, dtype=np.float64)


def _L_to_counts(L: np.ndarray, bp: BridgeParams) -> np.ndarray:
    """Exact inverse of :func:`odas_tpw.fp07cal.logr.log_r`."""
    e = np.exp(L)
    Z = (1.0 - e) / (1.0 + e)
    return bp.a + bp.b * Z * (bp.g * bp.e_b) / (2.0 * (bp.adc_fs / 2.0**bp.adc_bits))


def make_deployment(
    cfg: SynthConfig | None = None,
    *,
    coeffs: np.ndarray | None = None,
    channels: tuple[str, ...] = ("T1", "T2"),
    t2_offset_K: float = 0.0,
    t2_drift_K_per_day: float = 0.0,
) -> tuple[list[ProbeSeries], ReferenceSeries, dict]:
    """Build ``(probes, reference, truth)``.

    ``t2_offset_K`` / ``t2_drift_K_per_day`` perturb the second probe only, so
    a test can create a deployment where ``T1 - T2`` does (or pointedly does
    not) corroborate the blocked drift.
    """
    cfg = cfg or SynthConfig()
    coeffs = DEFAULT_COEFFS if coeffs is None else np.asarray(coeffs, dtype=np.float64)
    rng = np.random.default_rng(cfg.seed)

    t_rel, P, spans, yo_of = _pressure_track(cfg)
    t_abs = cfg.start_epoch + t_rel
    T_water = _water_temperature(P, cfg)
    days = (t_abs - t_abs[0]) / 86400.0

    counts: dict[str, np.ndarray] = {}
    factory: dict[str, np.ndarray] = {}
    bridge: dict[str, BridgeParams] = {}
    for ch in channels:
        drift = cfg.drift_K_per_day * days
        if ch == "T2":
            drift = drift + t2_offset_K + t2_drift_K_per_day * days
        T_probe = T_water + drift + rng.normal(0.0, cfg.probe_noise_K, T_water.size)
        counts[ch] = _L_to_counts(_invert_to_L(T_probe, coeffs), DEFAULT_BRIDGE)
        factory[ch] = coeffs.copy()
        bridge[ch] = DEFAULT_BRIDGE

    # Split into files on yo boundaries.
    n_files = max(1, cfg.files_per_deployment)
    yos_per_file = int(np.ceil(cfg.n_yos / n_files))
    probes: list[ProbeSeries] = []
    for f in range(n_files):
        lo_yo, hi_yo = f * yos_per_file, min((f + 1) * yos_per_file, cfg.n_yos)
        if lo_yo >= hi_yo:
            break
        m = (yo_of >= lo_yo) & (yo_of < hi_yo)
        idx = np.flatnonzero(m)
        i0 = int(idx[0])
        local = [
            (s - i0, e - i0)
            for (s, e) in spans
            if s >= i0 and e <= int(idx[-1])
        ]
        probes.append(
            ProbeSeries(
                label=f"synth_{f:04d}.p",
                time=t_abs[m],
                counts={ch: counts[ch][m] for ch in channels},
                pressure=P[m],
                bridge=dict(bridge),
                factory={ch: factory[ch].copy() for ch in channels},
                beta_key={ch: "beta_1" for ch in channels},
                profiles=local,
                instrument_sn="SYNTH-01",
                probe_sn={ch: "T" for ch in channels},  # placeholder, as real configs often are
                speed=np.full(int(m.sum()), 0.35),
            )
        )

    # ---- the reference: only on every n-th yo -------------------------------
    T_ctd_full = _single_pole(T_water, 1.0 / cfg.fs, cfg.ctd_tau)
    ref_dt = 1.0 / cfg.ref_rate
    ref_t: list[float] = []
    ref_T: list[float] = []
    ref_P: list[float] = []
    for k in range(0, cfg.n_yos, max(1, cfg.ct_every_n)):
        sel = np.flatnonzero(yo_of == k)
        if sel.size == 0:
            continue
        t0, t1 = t_abs[sel[0]], t_abs[sel[-1]]
        stamps = np.arange(t0 + cfg.ctd_delay + cfg.clock_offset, t1, ref_dt)
        # Value sampled at (stamp - clock_offset - ctd_delay) for T, and at
        # (stamp - clock_offset) for P: the clock shifts both, the response
        # only the temperature.
        tT = stamps - cfg.clock_offset - cfg.ctd_delay
        tP = stamps - cfg.clock_offset
        ref_t.extend(stamps.tolist())
        ref_T.extend(np.interp(tT, t_abs, T_ctd_full).tolist())
        ref_P.extend(np.interp(tP, t_abs, P).tolist())

    ref_T_arr = np.array(ref_T) + rng.normal(0.0, cfg.ref_noise_K, len(ref_T))
    reference = sanitize_reference(
        np.array(ref_t), ref_T_arr, pressure=np.array(ref_P), source="synthetic:sci_water_temp"
    )

    truth = {
        "coeffs": coeffs,
        "t_0": 1.0 / coeffs[0],
        "beta_1": 1.0 / coeffs[1],
        "clock_offset": cfg.clock_offset,
        "ctd_delay": cfg.ctd_delay,
        "ctd_tau": cfg.ctd_tau,
        "drift_K_per_day": cfg.drift_K_per_day,
        "t2_drift_K_per_day": t2_drift_K_per_day,
        "bridge": DEFAULT_BRIDGE,
        "n_yos": cfg.n_yos,
        "n_yos_with_ct": len(range(0, cfg.n_yos, max(1, cfg.ct_every_n))),
        "T_range": (float(np.min(T_water)), float(np.max(T_water))),
    }
    return probes, reference, truth

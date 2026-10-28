# Oct-2026, Claude and Pat Welch, pat@mousebrains.com
"""Single-pole-equivalent FP07 time constants of a bead pair, from their spectral ratio.

Two FP07 beads on one probe head see the same temperature microstructure, so
the median over windows of the log ratio of their gradient spectra measures
their responses with no turbulence model (:mod:`odas_tpw.chi.fp07_pair`, which
uses the same ratio to equalize the pair)::

    ln R(x) = ln G + ln|H2(x)|^2 - ln|H1(x)|^2

on the reduced-frequency axis x (frequency at the pair's median speed W_ref).
Fitting a single pole to each bead, ``|H_i|^2 = 1/(1 + (2 pi x tau_i)^2)``,
over 1-25 Hz gives each bead's SINGLE-POLE-EQUIVALENT time constant at W_ref.
What that number is, and is not:

* **The pole form is not determined.** A double pole fits the ratio as well
  and puts the 3-dB point 30-40% lower (``docs/fp07_tau_prefactor_PLAN.md``).
* **The absolute level is family-, band- and estimator-dependent** (on
  ARCTERX-2022 SN194 the same products give 17/25 ms over 1-25 Hz, 11/17 ms
  over 1-60 Hz, and 13/20 ms from spectral sums), and no external reference
  anchors it: the in-situ anchor failed its synthetic check.
* **The ratio is the robust part.** It is recorded model-free, as the power
  ratio at 10 and 20 Hz relative to the reference band.
* **Matched beads carry no information about tau**: their ratio is flat and
  the fit returns nothing it can defend. :data:`RULE` decides, and a pair it
  does not identify gets no tau, only its ratios.
* **A failing bead loses amplitude, not bandwidth.** A flat gain beyond ln 2
  (the :mod:`fp07_pair` amplitude signature) is never a tau measurement.

The tau of the electronics is not separated: the anti-alias filters sit near
110 Hz, far above the band, so a bead's tau here is the bead's (the PI's
reasoning, 2026-10-28). The time span is explicit, so one measurement
describes one bead pair on one instrument over one stretch of time; the two
time halves are fitted too, because a bead can slow within a deployment
(SN194's both did, by 25-50%).

``fp07-pair tau`` writes one JSON record per bead pair, which
``fp07-cal registry response`` records in the FP07 calibration sheet.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from odas_tpw.chi import fp07_pair as P

FORMAT = "fp07-bead-tau/1"
METHOD = "pair-ratio-sp"
"""Single pole per bead fitted to the median over windows of ln(S2/S1), 1-25 Hz."""

FIT_BAND = (1.0, 25.0)  # Hz of reduced frequency
# The fit is made at fixed points FIT_STEP apart, read off the pooled curve, so
# its weighting does not depend on the products' FFT resolution or on the speed
# spread (which scatters FFT bins over the log grid's bins). On ARCTERX-2022
# SN194 the log-grid bins as filled gave taus ~10% higher than these points;
# these points match the analysis-era fit behind the fleet table (1 Hz FFT
# bins, docs/fp07_tau_prefactor_PLAN.md) to 4%.
FIT_STEP = 1.0  # Hz
RATIO_F = (10.0, 20.0)  # Hz: the model-free power ratio is recorded here
TAU_BOUNDS = (5e-4, 0.2)  # s: the fit's search bounds
_STARTS = (0.002, 0.006, 0.015, 0.04)  # s: multi-start grid for each tau

# The identifiability rule, set on synthetics (scripts/fp07_tau/bead_tau_synth.py;
# docs/fp07pair/runbook.md section 6 has the validation and its history).
# Chi files (profiles). On synthetics the 16-84% file-bootstrap interval covered
# the truth 50-56% of the time at 8 files (nominal 68%) and 65-95% from 16 on.
MIN_FILES = 16
MIN_FIT_BINS = 12  # measured fit points (FIT_STEP apart) inside FIT_BAND
MIN_TOP_HZ = 20.0  # the fitted band must reach this: the 20 Hz ratio is measured
DIFFER_SIGMAS = 3.0  # |ln R'(20 Hz)| >= this many bootstrap half-widths ...
# ... and at least this: the beads differ by 16% in power at 20 Hz. Synthetic
# 11/10 ms pairs (|ln R'(20 Hz)| ~ 0.13) were still identified now and then, with
# intervals that missed the truth by 35-50%: near-matched beads put the absolute
# level on a ridge the file bootstrap does not see. 12/10 ms (0.26) is recovered.
DIFFER_MIN = 0.15
CI_MAX_FACTOR = 1.5  # each tau's 16-84% interval spans at most this factor
BOUND_MARGIN = 1.05  # ... and stays this far inside TAU_BOUNDS
BOOT_OK_FRAC = 0.9  # fraction of bootstrap replicates that must fit
# Drift (a flag, not part of the rule): |ln(tau_2nd / tau_1st)| beyond this many
# combined half-widths (in ln) of the halves' 16-84% intervals. Disjoint 16-84%
# intervals false-alarmed on up to 4 of 20 stationary synthetic pairs; this, 1 of 19.
DRIFT_SIGMAS = 2.0

# Noise-model check (a diagnostic, not part of the rule): in the quietest
# windows the observed spectrum over this band should be the noise model.
NOISE_BAND = (60.0, 100.0)  # Hz, raw frequency
NOISE_QUIET = 0.1  # the quietest decile, by S/N over 1-10 Hz
NOISE_TOL = 1.5  # obs/model outside [1/NOISE_TOL, NOISE_TOL] is flagged

RULE = (
    f"A bead pair's taus are identified when (1) at least {MIN_FILES} chi files and "
    f"{MIN_FIT_BINS} measured fit points (every {FIT_STEP:g} Hz of reduced frequency) in "
    f"{FIT_BAND[0]:g}-{FIT_BAND[1]:g} Hz reaching at least {MIN_TOP_HZ:g} Hz; (2) the flat "
    f"gain |ln G| <= ln 2 (beyond it is a failing bead's amplitude loss, not a time "
    f"constant); (3) the beads differ beyond bootstrap noise: |ln R'(20 Hz)| >= "
    f"{DIFFER_SIGMAS:g} half-widths of its 16-84% file-bootstrap interval and >= "
    f"{DIFFER_MIN:g}; (4) each tau and its 16-84% interval lie inside the fit bounds "
    f"{TAU_BOUNDS[0] * 1e3:g}-{TAU_BOUNDS[1] * 1e3:g} ms with a "
    f"{BOUND_MARGIN:g}x margin, and each interval spans at most a factor {CI_MAX_FACTOR:g}; "
    f"(5) at least {BOOT_OK_FRAC:.0%} of the bootstrap replicates fit. Otherwise tau is null "
    f"and only the ratios are recorded."
)

FLAGS = {
    "amplitude": "flat gain |ln G| > ln 2: an amplitude-loss signature (a failing bead), "
                 "never a tau measurement",
    "outlier": f"|ln R'(20 Hz)| > {P.OUTLIER_LNR20:g}, beyond the fleet spread: possibly a "
               "failing bead",
    "noise-model": f"a probe's observed spectrum over {NOISE_BAND[0]:g}-{NOISE_BAND[1]:g} Hz in "
                   f"its quietest windows is more than {NOISE_TOL:g}x off its noise model, "
                   "which biases tau",
    "drift": f"a bead's two time halves differ by more than {DRIFT_SIGMAS:g}x their combined "
             "1-sigma (16-84%) uncertainty",
}


# ---------------------------------------------------------------- the fit
def h2(f: np.ndarray, tau: float) -> np.ndarray:
    """Single-pole power response ``1/(1 + (2 pi f tau)^2)``."""
    return 1.0 / (1.0 + (2.0 * np.pi * np.asarray(f) * tau) ** 2)


def fit_single_pole(x: np.ndarray, lnR: np.ndarray) -> tuple[float, float, float, float]:
    """``(ln G, tau1, tau2, rms)`` for ``ln R = ln G + ln|H2|^2 - ln|H1|^2``.

    Least squares in ln R with ln G and both ln tau free, from a 4 x 4 grid
    of starting taus (the surface has a ridge when the beads nearly match);
    taus bounded to :data:`TAU_BOUNDS`.
    """
    from scipy.optimize import least_squares

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(lnR, dtype=np.float64)
    lo, hi = np.log(TAU_BOUNDS[0]), np.log(TAU_BOUNDS[1])
    w = (2.0 * np.pi * x) ** 2

    def resid(p: np.ndarray) -> np.ndarray:
        lg, lt1, lt2 = p
        model = lg + np.log1p(w * np.exp(2 * lt1)) - np.log1p(w * np.exp(2 * lt2))
        return np.asarray(y - model, dtype=np.float64)

    best = None
    for t1 in _STARTS:
        for t2 in _STARTS:
            r = least_squares(resid, [0.0, np.log(t1), np.log(t2)],
                              bounds=([-3.0, lo, lo], [3.0, hi, hi]))
            if best is None or r.cost < best.cost:
                best = r
    assert best is not None
    lg, lt1, lt2 = best.x
    return float(lg), float(np.exp(lt1)), float(np.exp(lt2)), float(np.sqrt(2 * best.cost / x.size))


def _smoothed(files: list[P._FileRatio], grid: np.ndarray, min_windows: int) -> np.ndarray:
    """The pooled ln-ratio curve exactly as ``fp07-pair fit`` builds it."""
    curve, _total = P._curve(files, grid, min_windows)
    return P._running_median(curve, 1)


def _value_at(curve: np.ndarray, grid: np.ndarray, f0: float) -> float:
    """*curve* at *f0* (log-linear between finite bins); NaN if not measured there."""
    ok = np.isfinite(curve)
    if not ok.any():
        return np.nan
    lx, l0 = np.log10(grid[ok]), math.log10(f0)
    if l0 < lx[0] or l0 > lx[-1] or np.min(np.abs(lx - l0)) > 2 * P.LOG_STEP + 1e-9:
        return np.nan
    return float(np.interp(l0, lx, curve[ok]))


@dataclass
class _One:
    """One curve's fit and ratios (the point estimate or a bootstrap replicate)."""

    ok: bool
    ln_G: float = np.nan
    tau: tuple[float, float] = (np.nan, np.nan)
    rms: float = np.nan
    n_bins: int = 0
    band: tuple[float, float] = (np.nan, np.nan)
    ln_ratio: tuple[float, ...] = ()


def fit_points(band: tuple[float, float]) -> np.ndarray:
    """The fixed frequencies [Hz] the fit is made at: every FIT_STEP across *band*."""
    return np.arange(band[0], band[1] + 1e-9, FIT_STEP)


def _curve_at(curve: np.ndarray, grid: np.ndarray, f: np.ndarray) -> np.ndarray:
    """*curve* at *f*, log-linear between its finite bins; NaN outside them (no extrapolation)."""
    ok = np.isfinite(curve)
    out = np.full(f.shape, np.nan)
    if ok.sum() < 2:
        return out
    lx, lf = np.log10(grid[ok]), np.log10(f)
    inside = (lf >= lx[0]) & (lf <= lx[-1])
    out[inside] = np.interp(lf[inside], lx, curve[ok])
    return out


def _fit_curve(curve: np.ndarray, grid: np.ndarray, ref_bins: np.ndarray,
               band: tuple[float, float]) -> _One:
    ref = float(np.nanmean(curve[ref_bins])) if ref_bins.size else np.nan
    ratios = tuple(_value_at(curve, grid, f0) - ref for f0 in RATIO_F)
    fx = fit_points(band)
    y = _curve_at(curve, grid, fx)
    sel = np.isfinite(y)
    n = int(sel.sum())
    if n < MIN_FIT_BINS:
        return _One(False, n_bins=n, ln_ratio=ratios)
    lg, t1, t2, rms = fit_single_pole(fx[sel], y[sel])
    return _One(True, lg, (t1, t2), rms, n, (float(fx[sel][0]), float(fx[sel][-1])), ratios)


def _interval(v: list[float]) -> tuple[float, float]:
    a = np.asarray(v, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return (np.nan, np.nan)
    lo, hi = np.percentile(a, [16, 84])
    return (float(lo), float(hi))


@dataclass
class TauFit:
    """One stretch of files: both taus at W_ref, intervals, ratios, and the rule's verdict."""

    first: float  # epoch s: first and last file start_time
    last: float
    n_files: int
    n_windows: int
    identified: bool
    reasons: tuple[str, ...]
    tau: tuple[float, float] | None  # s, in channel order; None when not identified
    tau_fit: tuple[float, float]  # s, the fit's values whatever the verdict
    tau_ci: tuple[tuple[float, float], tuple[float, float]]
    ln_G_fit: float  # the fit's free gain
    ln_G: float  # the reference-band level, as fp07-pair fit reports it
    rms: float
    fit_band: tuple[float, float]
    n_bins: int
    ln_ratio: dict[float, float]  # f -> ln R'(f) = ln (S2/S1)(f) - reference level
    ln_ratio_ci: dict[float, tuple[float, float]]
    n_boot: int
    n_boot_ok: int

    def to_dict(self) -> dict[str, Any]:
        def num(x: float) -> float | None:
            return None if not np.isfinite(x) else float(x)

        return {
            "first": _iso(self.first), "last": _iso(self.last),
            "n_files": self.n_files, "n_windows": self.n_windows,
            "identified": self.identified, "reasons": list(self.reasons),
            "tau_s": None if self.tau is None else [float(t) for t in self.tau],
            "tau_fit_s": [num(t) for t in self.tau_fit],
            "tau_16_84_s": [[num(a), num(b)] for a, b in self.tau_ci],
            "ln_G_fit": num(self.ln_G_fit), "ln_G": num(self.ln_G), "rms": num(self.rms),
            "fit_band_Hz": [num(x) for x in self.fit_band], "n_bins": self.n_bins,
            "ln_ratio": {f"{f:g}": num(v) for f, v in self.ln_ratio.items()},
            "ln_ratio_16_84": {f"{f:g}": [num(a), num(b)]
                               for f, (a, b) in self.ln_ratio_ci.items()},
            "n_boot": self.n_boot, "n_boot_ok": self.n_boot_ok,
        }


def fit_files(files: list[P._FileRatio], grid: np.ndarray, *, band: tuple[float, float] = FIT_BAND,
              n_boot: int = 200, min_windows: int = 200, rng: np.random.Generator | None = None,
              min_files: int = MIN_FILES) -> TauFit:
    """Fit one stretch of per-file ratios and apply :data:`RULE`."""
    rng = rng or np.random.default_rng(0)
    n = len(files)
    t = [f.t0 for f in files]
    n_win = int(sum(f.n_windows for f in files))
    curve = _smoothed(files, grid, min_windows)
    ok = np.isfinite(curve)
    ref_bins = P.reference_bins(ok, grid)
    point = _fit_curve(curve, grid, ref_bins, band)
    ln_G = float(np.mean(curve[ref_bins])) if ref_bins.size else np.nan
    taus: list[list[float]] = [[], []]
    rat: list[list[float]] = [[] for _ in RATIO_F]
    n_ok = 0
    for _ in range(n_boot if n >= 2 else 0):
        pick = [files[i] for i in rng.integers(0, n, n)]
        rep = _fit_curve(_smoothed(pick, grid, min_windows), grid, ref_bins, band)
        for k, v in enumerate(rep.ln_ratio):
            rat[k].append(v)
        if rep.ok:
            n_ok += 1
            taus[0].append(rep.tau[0])
            taus[1].append(rep.tau[1])
    tau_ci = (_interval(taus[0]), _interval(taus[1]))
    ratio_ci = {f0: _interval(rat[k]) for k, f0 in enumerate(RATIO_F)}
    ln_ratio = dict(zip(RATIO_F, point.ln_ratio or (np.nan,) * len(RATIO_F), strict=True))

    reasons: list[str] = []
    if n < min_files:
        reasons.append(f"{n} chi files < {min_files}")
    if not point.ok:
        reasons.append(f"{point.n_bins} measured fit points in {band[0]:g}-{band[1]:g} Hz "
                       f"< {MIN_FIT_BINS}")
    elif point.band[1] < MIN_TOP_HZ:
        reasons.append(f"the usable band ends at {point.band[1]:.1f} Hz < {MIN_TOP_HZ:g} Hz")
    if not np.isfinite(ln_G):
        reasons.append(f"no usable bins below {P.REF_BAND_MAX:g} Hz for the reference level")
    elif abs(ln_G) > P.LN_G_MAX:
        reasons.append(f"flat gain ln G = {ln_G:+.2f} beyond ln 2: an amplitude-loss "
                       f"signature (a failing bead), not a time constant")
    r20 = ln_ratio.get(20.0, np.nan)
    lo20, hi20 = ratio_ci.get(20.0, (np.nan, np.nan))
    hw = 0.5 * (hi20 - lo20)
    if not (np.isfinite(r20) and np.isfinite(hw)):
        reasons.append("no ratio at 20 Hz")
    elif abs(r20) < max(DIFFER_SIGMAS * hw, DIFFER_MIN):
        reasons.append(f"beads not distinguishable: |ln R'(20 Hz)| = {abs(r20):.3f} < "
                       f"max({DIFFER_SIGMAS:g} x {hw:.3f}, {DIFFER_MIN:g})")
    if n_boot and n_ok < BOOT_OK_FRAC * n_boot:
        reasons.append(f"{n_ok} of {n_boot} bootstrap replicates fitted")
    lo_b, hi_b = TAU_BOUNDS[0] * BOUND_MARGIN, TAU_BOUNDS[1] / BOUND_MARGIN
    for k in (0, 1):
        a, b = tau_ci[k]
        v = point.tau[k]
        if not all(np.isfinite([a, b, v])):
            reasons.append(f"tau{k + 1}: no interval")
        elif not (lo_b < min(a, v) and max(b, v) < hi_b):
            reasons.append(f"tau{k + 1} = {v * 1e3:.1f} ms [{a * 1e3:.1f}-{b * 1e3:.1f}] at or "
                           f"near a fit bound")
        elif b / a > CI_MAX_FACTOR:
            reasons.append(f"tau{k + 1} interval {a * 1e3:.1f}-{b * 1e3:.1f} ms spans "
                           f"x{b / a:.2f} > x{CI_MAX_FACTOR:g}")
    identified = not reasons
    return TauFit(
        first=float(min(t)) if t else np.nan, last=float(max(t)) if t else np.nan,
        n_files=n, n_windows=n_win, identified=identified, reasons=tuple(reasons),
        tau=point.tau if identified else None, tau_fit=point.tau, tau_ci=tau_ci,
        ln_G_fit=point.ln_G, ln_G=ln_G, rms=point.rms, fit_band=point.band,
        n_bins=point.n_bins, ln_ratio=ln_ratio, ln_ratio_ci=ratio_ci,
        n_boot=n_boot, n_boot_ok=n_ok,
    )


def drifted(a: TauFit, b: TauFit, c: int) -> bool:
    """Did bead *c* change between halves *a* and *b*? (:data:`DRIFT_SIGMAS` in ln tau)."""
    if a.tau is None or b.tau is None:
        return False
    s = [0.5 * math.log(h.tau_ci[c][1] / h.tau_ci[c][0]) for h in (a, b)]
    return abs(math.log(b.tau[c] / a.tau[c])) > DRIFT_SIGMAS * math.hypot(*s)


def split_halves(t: Sequence[float]) -> int | None:
    """Index splitting time-sorted *t* into two halves, between DIFFERENT times.

    Every profile cut from one .p file carries that file's start_time, so a
    split inside such a group would put one recording in both halves. Times
    are compared in whole seconds, the resolution the record and the sheet
    keep, so the halves never share a printed time. The split nearest the
    middle by file count wins; None when all times are equal.
    """
    t = [math.floor(x) for x in t]
    cands = [k for k in range(1, len(t)) if t[k] > t[k - 1]]
    if not cands:
        return None
    return min(cands, key=lambda k: (abs(k - len(t) / 2), k))


# ------------------------------------------------------------ the products
def _peek(path: str) -> dict:
    """What grouping needs from a chi product, without its spectra."""
    import netCDF4

    with netCDF4.Dataset(path) as ds:
        if "spec_gradT" not in ds.variables or ds["spec_gradT"].shape[0] < 2:
            raise ValueError(f"{path}: needs spectra from two thermistors")
        probes = None
        if "probe" in ds.variables:
            pv = ds["probe"][:]
            probes = [str(x) for x in (netCDF4.chartostring(pv) if pv.dtype.kind == "S" else pv)]
        return {
            "W": np.asarray(ds["speed"][:], dtype=np.float64),
            "f_max": float(np.nanmax(np.asarray(ds["F"][:], dtype=np.float64))),
            "channels": tuple((probes or ["T1_dT1", "T2_dT2"])[:2]),
            "source": str(getattr(ds, "source", "")),
            "start": str(getattr(ds, "start_time", "")),
            "fp07_model": str(getattr(ds, "fp07_model", "single_pole")),
        }


def _noise_stats(prod: dict) -> tuple[np.ndarray, np.ndarray]:
    """Per (probe, window): median S/N over 1-10 Hz, and over :data:`NOISE_BAND`."""
    S, N, F = prod["S"], prod["N"], prod["F"]
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(N > 0, S / np.where(N > 0, N, 1.0), np.nan)  # (probe, freq, window)
    quiet = np.full((2, r.shape[2]), np.nan)
    high = np.full((2, r.shape[2]), np.nan)
    for j in range(r.shape[2]):
        lo = (F[:, j] >= 1.0) & (F[:, j] <= 10.0)
        hb = (F[:, j] >= NOISE_BAND[0]) & (F[:, j] <= NOISE_BAND[1])
        for p in (0, 1):
            if lo.any():
                quiet[p, j] = np.nanmedian(r[p, lo, j])
            if hb.any():
                high[p, j] = np.nanmedian(r[p, hb, j])
    return quiet, high


def _noise_check(quiet: np.ndarray, high: np.ndarray) -> list[float | None]:
    out: list[float | None] = []
    for p in (0, 1):
        m = np.isfinite(quiet[p]) & np.isfinite(high[p])
        if m.sum() < 10:
            out.append(None)
            continue
        cut = np.quantile(quiet[p][m], NOISE_QUIET)
        out.append(float(np.median(high[p][m & (quiet[p] <= cut)])))
    return out


@dataclass
class PairTau:
    """One bead pair's measurement over one explicit span on one instrument."""

    instrument: str
    channels: tuple[str, str]
    serials: tuple[str, str]
    tau_model: str  # the speed scaling the reduced frequency follows
    W_ref: float  # m/s
    band: tuple[float, float]
    requested: tuple[float, float]  # epoch s: the span asked for
    all: TauFit
    halves: tuple[TauFit, TauFit] | None
    noise: list[float | None]  # per channel: quiet-window obs/model over NOISE_BAND
    flags: tuple[str, ...]
    drift: tuple[bool, bool]  # per channel: halves' intervals disjoint
    files: tuple[str, str]  # first and last chi file (basenames)
    software_version: str
    created: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": FORMAT,
            "method": METHOD,
            "instrument": self.instrument,
            "channels": list(self.channels),
            "serials": list(self.serials),
            "speed_scaling": self.tau_model,
            "W_ref_m_s": self.W_ref,
            "band_Hz": list(self.band),
            "requested": [_iso(self.requested[0]), _iso(self.requested[1])],
            "all": self.all.to_dict(),
            "halves": None if self.halves is None else [h.to_dict() for h in self.halves],
            "noise_obs_over_model": self.noise,
            "flags": list(self.flags),
            "drift": list(self.drift),
            "first_file": self.files[0],
            "last_file": self.files[1],
            "rule": RULE,
            "software_version": self.software_version,
            "created": self.created,
        }


def _iso(t: float) -> str:
    return datetime.fromtimestamp(float(t), UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_time(s: str) -> float:
    """Epoch seconds from an ISO-8601 date-time (``Z`` or an offset; naive is UTC)."""
    dt = datetime.fromisoformat(s.strip().replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.timestamp()


@dataclass
class MeasureResult:
    pairs: list[PairTau]
    report: list[str]


def measure(paths: Sequence[str], *, instrument: str, first: float, last: float,
            beads: Sequence[str] | None = None, band: tuple[float, float] = FIT_BAND,
            n_boot: int = 200, min_windows: int = 200, min_files: int = MIN_FILES,
            rng_seed: int = 0) -> MeasureResult:
    """Each bead pair's taus on *instrument* over files starting in [first, last].

    *paths* are chi product NetCDFs, as for ``fp07-pair fit``; a file belongs
    to the span when its ``start_time`` does. *instrument* is a serial
    (``194``, ``SN194``); the instrument and the bead serials are read from
    each file's source configuration (:func:`fp07_pair.identity_from_source`),
    never from the file name. *beads*, when given, keeps only pairs holding
    every serial listed. Files are read twice (speeds, then spectra one file
    at a time), so memory does not grow with the deployment.
    """
    from odas_tpw._version_info import software_version
    from odas_tpw.chi.fp07 import default_tau_model

    want = "SN" + str(int(str(instrument).strip().upper().removeprefix("SN").strip()))
    report: list[str] = []
    groups: dict[tuple, list[tuple[str, dict]]] = {}
    n_skip = n_out = n_other = 0
    # Every profile cut from one .p file carries that file's start_time and its
    # configuration, so the identity is read once per (source directory, start,
    # channels). Reading it per profile over a network share took ~0.5 s a file.
    identities: dict[tuple, tuple[str, list[str]]] = {}
    for p in sorted(paths):
        try:
            meta = _peek(p)
            t0 = P._epoch(meta["start"])
            if not np.isfinite(t0):
                raise ValueError("no start_time")
            if not first <= t0 <= last:
                n_out += 1
                continue
            ikey = (os.path.dirname(meta["source"]), meta["start"], meta["channels"])
            if ikey not in identities:
                identities[ikey] = P.identity_from_source(meta["source"], meta["channels"],
                                                          product=p)
            inst, ser = identities[ikey]
        except (OSError, ValueError, KeyError) as exc:
            n_skip += 1
            report.append(f"SKIPPED {os.path.basename(p)}: {exc}")
            continue
        if inst != want:
            n_other += 1
            continue
        meta["t0"] = t0
        key = (inst, meta["channels"], tuple(ser), default_tau_model(meta["fp07_model"]))
        groups.setdefault(key, []).append((p, meta))
    if beads:
        groups = {k: v for k, v in groups.items() if all(b in k[2] for b in beads)}
    report.insert(0, (
        f"{len(paths)} chi files: {n_out} outside {_iso(first)} .. {_iso(last)}, {n_other} on "
        f"other instruments, {n_skip} skipped; {len(groups)} bead pair(s) on {want}"
        + (f" holding {', '.join(beads)}" if beads else "")
    ))
    created = datetime.now(UTC).isoformat(timespec="seconds")
    out: list[PairTau] = []
    for (inst, chans, sers, tau_model), members in sorted(groups.items()):
        members.sort(key=lambda m: (m[1]["t0"], m[0]))
        W_all = np.concatenate([m[1]["W"] for m in members])
        W_ref = float(np.median(W_all[np.isfinite(W_all) & (W_all > 0)]))
        grid = P._grid(min(min(m[1]["f_max"] for m in members), 100.0))
        files: list[P._FileRatio] = []
        quiet, high = [], []
        for p, _meta in members:
            prod = P._read_product(p)
            files.append(P._file_ratio(prod, W_ref, tau_model, grid, p))
            q, h = _noise_stats(prod)
            quiet.append(q)
            high.append(h)
        rng = np.random.default_rng(rng_seed)
        fit_all = fit_files(files, grid, band=band, n_boot=n_boot, min_windows=min_windows,
                            rng=rng, min_files=min_files)
        k = split_halves([f.t0 for f in files])
        halves = None
        if k is not None:
            halves = (
                fit_files(files[:k], grid, band=band, n_boot=n_boot, min_windows=min_windows,
                          rng=rng, min_files=min_files),
                fit_files(files[k:], grid, band=band, n_boot=n_boot, min_windows=min_windows,
                          rng=rng, min_files=min_files),
            )
        noise = _noise_check(np.concatenate(quiet, axis=1), np.concatenate(high, axis=1))
        flags = []
        if np.isfinite(fit_all.ln_G) and abs(fit_all.ln_G) > P.LN_G_MAX:
            flags.append("amplitude")
        r20 = fit_all.ln_ratio.get(20.0, np.nan)
        if np.isfinite(r20) and abs(r20) > P.OUTLIER_LNR20:
            flags.append("outlier")
        if any(v is not None and not 1 / NOISE_TOL <= v <= NOISE_TOL for v in noise):
            flags.append("noise-model")
        drift = [False, False]
        if halves is not None and all(h.identified for h in halves):
            for c in (0, 1):
                drift[c] = drifted(halves[0], halves[1], c)
        if any(drift):
            flags.append("drift")
        pt = PairTau(
            instrument=inst, channels=chans, serials=sers, tau_model=tau_model, W_ref=W_ref,
            band=band, requested=(first, last), all=fit_all, halves=halves, noise=noise,
            flags=tuple(flags), drift=(drift[0], drift[1]),
            files=(os.path.basename(members[0][0]), os.path.basename(members[-1][0])),
            software_version=software_version(), created=created,
        )
        out.append(pt)
        report.extend(format_pair(pt))
    return MeasureResult(out, report)


def _ms(v: float) -> str:
    return f"{v * 1e3:.1f}" if np.isfinite(v) else "-"


def _fit_line(name: str, f: TauFit, chans: tuple[str, str], sers: tuple[str, str]) -> list[str]:
    r = [f"{np.exp(f.ln_ratio[q]):.3f}" if np.isfinite(f.ln_ratio[q]) else "-" for q in RATIO_F]
    ci = [f"[{np.exp(a):.3f}-{np.exp(b):.3f}]" if np.isfinite(a) else "" for a, b in
          (f.ln_ratio_ci[q] for q in RATIO_F)]
    head = (f"  {name}: {_iso(f.first)} .. {_iso(f.last)}, {f.n_files} files, {f.n_windows} "
            f"windows; S2/S1 power ratio vs the reference band: "
            + ", ".join(f"{q:g} Hz {a} {b}" for q, a, b in zip(RATIO_F, r, ci, strict=True))
            + f"; ln G {f.ln_G:+.3f}")
    lines = [head]
    verdict = "identified" if f.identified else "NOT identified"
    tau = ", ".join(
        f"{c}={s} {_ms(f.tau_fit[k])} [{_ms(f.tau_ci[k][0])}-{_ms(f.tau_ci[k][1])}] ms"
        for k, (c, s) in enumerate(zip(chans, sers, strict=True)))
    lines.append(f"    single-pole tau at W_ref ({verdict}): {tau}; fit "
                 f"{f.fit_band[0]:.1f}-{f.fit_band[1]:.1f} Hz, {f.n_bins} points, rms {f.rms:.3f}, "
                 f"{f.n_boot_ok}/{f.n_boot} replicates")
    lines += [f"    not identified: {why}" for why in f.reasons]
    return lines


def format_pair(pt: PairTau) -> list[str]:
    """The text report for one pair."""
    c, s = pt.channels, pt.serials
    lines = [
        f"{pt.instrument} {c[0]}={s[0]} {c[1]}={s[1]}: W_ref {pt.W_ref:.3f} m/s "
        f"({pt.tau_model} speed scaling), band {pt.band[0]:g}-{pt.band[1]:g} Hz; files "
        f"{pt.files[0]} .. {pt.files[1]}",
    ]
    lines += _fit_line("all", pt.all, c, s)
    if pt.halves is None:
        lines.append("  halves: none (every file shares one start_time)")
    else:
        for name, h in zip(("first half", "second half"), pt.halves, strict=True):
            lines += _fit_line(name, h, c, s)
    noise = ", ".join(f"{ch} {'-' if v is None else f'{v:.2f}'}" for ch, v in zip(c, pt.noise,
                                                                              strict=True))
    lines.append(f"  noise check (obs/model, {NOISE_BAND[0]:g}-{NOISE_BAND[1]:g} Hz, quietest "
                 f"{NOISE_QUIET:.0%} of windows): {noise}")
    lines.append(f"  flags: {', '.join(pt.flags) or '-'}")
    return lines


def output_stem(pt: PairTau) -> str:
    """``fp07tau_<instrument>_<serial1>-<serial2>_<first file start>``."""
    import re

    def clean(x: str) -> str:
        return re.sub(r"[^A-Za-z0-9]+", "", x) or "x"

    d0 = datetime.fromtimestamp(pt.all.first, UTC).strftime("%Y%m%dT%H%MZ")
    return f"fp07tau_{clean(pt.instrument)}_{clean(pt.serials[0])}-{clean(pt.serials[1])}_{d0}"


def write_record(pt: PairTau, out_dir: str | Path) -> tuple[Path, Path]:
    """``<stem>.json`` (the record ``fp07-cal registry response`` reads) and ``<stem>.txt``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = output_stem(pt)
    js, txt = out / f"{stem}.json", out / f"{stem}.txt"
    js.write_text(json.dumps(pt.to_dict(), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    txt.write_text("\n".join(format_pair(pt)) + "\n", encoding="utf-8")
    return js, txt

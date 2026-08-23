# Aug-2026, Claude and Pat Welch, pat@mousebrains.com
"""Building ``(L, T_ref)`` calibration pairs on the CTD's own sample times.

The inversion that makes sparse references work
-----------------------------------------------
The obvious approach --- interpolate the 1 Hz CTD up onto the instrument's
64 Hz grid and regress there --- is wrong twice over.  It invents reference
values wherever the CTD was not sampling (plan A1), and it leaves the regressor
carrying ~20 Hz of bandwidth the reference cannot see, which attenuates the
fitted slope toward zero by classic errors-in-variables (plan A2).  That slope
IS ``beta_1``.

So we go the other way: **decimate the thermistor down onto real CTD samples.**
For each genuine CTD sample at ``t_k`` we form

    L_k = < L(t) >  over the CTD's sampling kernel, centred at t_k - lag

which fixes both problems at once and makes the degrees of freedom honest ---
``N`` is the number of real CTD samples, not an interpolation-inflated count.
Sparsity then needs no special handling anywhere: no CTD sample, no pair.

The kernel is the CTD's integration window (a boxcar of its sample interval)
applied to a thermistor that has first been slowed to the CTD's response with a
causal single pole.  Both are configurable and both are things V4 tests the
sensitivity to.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace

import numpy as np

from odas_tpw.fp07cal.logr import log_r
from odas_tpw.fp07cal.series import ProbeSeries, ReferenceSeries


@dataclass
class PairConfig:
    """Gates and kernel shape.  Every rejection is counted by reason."""

    max_gap: float = 30.0
    """[s] Largest CTD sample spacing still considered continuous coverage.

    Not a smoothing parameter --- a definition of where the reference exists.
    """

    kernel_width: float | None = None
    """[s] Boxcar width.  ``None`` -> the CTD's median sample interval."""

    kernel_tau: float = 0.5
    """[s] Single-pole time constant approximating the reference's response."""

    min_speed: float = 0.05
    """[m/s] Below this the thermistor is not flushing.  Skipped if no speed."""

    min_kernel_samples: int = 2
    """Minimum thermistor samples inside the kernel window."""

    require_profile: bool = True
    """Only accept samples inside a detected profile (excludes apogee/surface)."""

    min_corr: float = 0.7
    """Minimum |Pearson r| between L and 1/T_ref for a lag to be believed (R9)."""


@dataclass
class PairSet:
    """Calibration pairs, plus the accounting for everything rejected."""

    time: np.ndarray = field(default_factory=lambda: np.empty(0))
    T_ref: np.ndarray = field(default_factory=lambda: np.empty(0))
    L: np.ndarray = field(default_factory=lambda: np.empty(0))
    pressure: np.ndarray = field(default_factory=lambda: np.empty(0))
    w: np.ndarray = field(default_factory=lambda: np.empty(0))
    """Vertical speed dP/dt [dbar/s] at the kernel centre.

    Carried so that a residual proportional to dT/dz can be split into the
    geometric sensor offset (dz) and a leftover timing error (tau*w) --- the two
    are otherwise degenerate.  See :mod:`odas_tpw.fp07cal.geometry`.
    """

    direction: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int8))
    profile_uid: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=object))
    file_label: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=object))
    channel: str = "?"
    lag: float = 0.0
    corr: float = float("nan")
    rejected: Counter = field(default_factory=Counter)
    per_file: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.time.size)

    @property
    def T_range(self) -> tuple[float, float]:
        if not len(self):
            return (float("nan"), float("nan"))
        return (float(np.min(self.T_ref)), float(np.max(self.T_ref)))

    def n_profiles(self) -> int:
        return int(np.unique(self.profile_uid).size) if len(self) else 0

    def concat(self, other: PairSet) -> PairSet:
        if other.channel != self.channel and len(self) and len(other):
            raise ValueError(f"cannot concatenate channels {self.channel!r} and {other.channel!r}")
        rej = Counter(self.rejected)
        rej.update(other.rejected)
        merged = dict(self.per_file)
        merged.update(other.per_file)
        return PairSet(
            time=np.concatenate([self.time, other.time]),
            T_ref=np.concatenate([self.T_ref, other.T_ref]),
            L=np.concatenate([self.L, other.L]),
            pressure=np.concatenate([self.pressure, other.pressure]),
            w=np.concatenate([self.w, other.w]),
            direction=np.concatenate([self.direction, other.direction]),
            profile_uid=np.concatenate([self.profile_uid, other.profile_uid]),
            file_label=np.concatenate([self.file_label, other.file_label]),
            channel=self.channel or other.channel,
            lag=self.lag,
            corr=self.corr,
            rejected=rej,
            per_file=merged,
        )


def _single_pole(x: np.ndarray, dt: float, tau: float) -> np.ndarray:
    """Causal one-pole low-pass --- the reference's response, applied to the probe.

    Causal on purpose: the CTD genuinely lags the water, so slowing the
    thermistor the same way is the physically honest match.  Whatever bulk
    delay this introduces is absorbed by the lag search, which runs on the
    already-filtered signal.

    Initialised to steady state at ``x[0]`` rather than to zero.  A zero
    initial condition makes the filter ramp up from 0 over the first few tau,
    which on ``L`` (order -0.1) is a large excursion --- it would put a
    fabricated cold spike at the head of every file and drag any pair built
    there.  The transient is silent: the values are finite and smooth.
    """
    x = np.asarray(x, dtype=np.float64)
    if tau <= 0 or not np.isfinite(tau) or x.size == 0:
        return x
    alpha = dt / (tau + dt)
    from scipy.signal import lfilter

    b, a = [alpha], [1.0, -(1.0 - alpha)]
    seed = x[0] if np.isfinite(x[0]) else np.nanmean(x)
    if not np.isfinite(seed):
        seed = 0.0
    zi = np.array([(1.0 - alpha) * seed])
    y, _ = lfilter(b, a, x, zi=zi)
    return np.asarray(y)


def _boxcar_at(
    t: np.ndarray, x: np.ndarray, centers: np.ndarray, width: float
) -> tuple[np.ndarray, np.ndarray]:
    """Mean of *x* over ``[c - width/2, c + width/2]`` for each centre.

    NaN-aware: non-finite samples are excluded from both sum and count, so the
    returned count is the number of GOOD samples actually averaged and a window
    that is mostly holes is visible as a low count rather than a quiet NaN.
    """
    good = np.isfinite(x)
    xz = np.where(good, x, 0.0)
    csum = np.concatenate(([0.0], np.cumsum(xz)))
    ccnt = np.concatenate(([0.0], np.cumsum(good.astype(np.float64))))
    lo = np.searchsorted(t, centers - width / 2.0, side="left")
    hi = np.searchsorted(t, centers + width / 2.0, side="right")
    n = ccnt[hi] - ccnt[lo]
    s = csum[hi] - csum[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(n > 0, s / np.maximum(n, 1.0), np.nan)
    return mean, n


def _reference_indices_in_span(ref: ReferenceSeries, cfg: PairConfig, t0: float, t1: float):
    """Reference sample indices inside valid spans that overlap ``[t0, t1]``."""
    idx: list[np.ndarray] = []
    for s, e in ref.valid_spans(cfg.max_gap):
        if ref.time[e] < t0 or ref.time[s] > t1:
            continue
        k = np.arange(s, e + 1)
        k = k[(ref.time[k] >= t0) & (ref.time[k] <= t1)]
        if k.size:
            idx.append(k)
    return np.concatenate(idx) if idx else np.empty(0, dtype=int)


@dataclass
class PreparedProbe:
    """Per-probe arrays that do not depend on the lag.

    The lag search evaluates ~160 trial lags, and everything here --- the
    bridge conversion, the single-pole filter over ~200k samples, the
    per-sample profile ids, the vertical speed --- is identical at every one of
    them.  Only the boxcar centres move.  Recomputing them per lag made the
    lag search roughly an order of magnitude slower than it needs to be on a
    real deployment.
    """

    channel: str
    L_slow: np.ndarray
    clipped: np.ndarray
    w: np.ndarray
    pid: np.ndarray
    dt: float


def prepare_probe(probe: ProbeSeries, channel: str, cfg: PairConfig) -> PreparedProbe | None:
    """Precompute the lag-independent half of :func:`build_pairs`."""
    if channel not in probe.counts or probe.time.size < 2:
        return None
    dt = 1.0 / probe.fs
    L_raw, clipped = log_r(probe.counts[channel], probe.bridge[channel])
    return PreparedProbe(
        channel=channel,
        L_slow=_single_pole(L_raw, dt, cfg.kernel_tau),
        clipped=clipped,
        w=np.gradient(probe.pressure, probe.time),
        pid=probe.profile_id(),
        dt=dt,
    )


def build_pairs(
    probe: ProbeSeries,
    ref: ReferenceSeries,
    channel: str,
    *,
    lag: float = 0.0,
    cfg: PairConfig | None = None,
    prepared: PreparedProbe | None = None,
) -> PairSet:
    """Pairs from one file and one channel.  Zero pairs is a normal outcome.

    Pass *prepared* (from :func:`prepare_probe`) when sweeping many lags over
    the same probe; it skips the lag-independent work.
    """
    cfg = cfg or PairConfig()
    if prepared is not None and prepared.channel != channel:
        raise ValueError(
            f"prepared probe is for {prepared.channel!r}, not {channel!r}"
        )
    rejected: Counter = Counter()
    empty = PairSet(channel=channel, lag=lag, rejected=rejected)

    if channel not in probe.counts:
        rejected["channel_absent"] += 1
        return replace(empty, per_file={probe.label: 0})
    if probe.time.size < 2:
        rejected["probe_too_short"] += 1
        return replace(empty, per_file={probe.label: 0})

    if prepared is None:
        prepared = prepare_probe(probe, channel, cfg)
        if prepared is None:
            rejected["probe_too_short"] += 1
            return replace(empty, per_file={probe.label: 0})
    dt = prepared.dt
    width = cfg.kernel_width if cfg.kernel_width else ref.median_interval()
    if not np.isfinite(width) or width <= 0:
        width = max(dt, 1.0)

    # The kernel needs the whole window inside the file, plus the lag shift.
    pad = width / 2.0 + abs(lag)
    k = _reference_indices_in_span(
        ref, cfg, float(probe.time[0]) + pad, float(probe.time[-1]) - pad
    )
    n_span = int(k.size)
    if n_span == 0:
        rejected["no_reference_coverage"] += 1
        return replace(empty, per_file={probe.label: 0})

    w_probe = prepared.w
    clipped = prepared.clipped
    L_slow = prepared.L_slow

    centers = ref.time[k] - lag
    L_k, n_k = _boxcar_at(probe.time, L_slow, centers, width)
    # A clipped sample anywhere in the window poisons the average (A7).
    clip_frac, _ = _boxcar_at(probe.time, clipped.astype(np.float64), centers, width)

    keep = np.isfinite(L_k)
    rejected["kernel_all_nan"] += int(np.sum(~keep))

    thin = n_k < cfg.min_kernel_samples
    rejected["kernel_too_few_samples"] += int(np.sum(thin & keep))
    keep &= ~thin

    clipbad = np.nan_to_num(clip_frac, nan=1.0) > 0.0
    rejected["clipped_counts"] += int(np.sum(clipbad & keep))
    keep &= ~clipbad

    # Locate each kernel centre on the probe grid for the per-sample gates.
    j = np.clip(np.searchsorted(probe.time, centers), 0, probe.time.size - 1)

    if cfg.require_profile:
        pid = prepared.pid[j]
        outside = pid < 0
        rejected["outside_profile"] += int(np.sum(outside & keep))
        keep &= ~outside
    else:
        pid = prepared.pid[j]

    if probe.speed is not None and cfg.min_speed > 0:
        slow = ~(probe.speed[j] >= cfg.min_speed)
        rejected["below_min_speed"] += int(np.sum(slow & keep))
        keep &= ~slow

    T_ref_k = ref.value[k]
    bad_ref = ~np.isfinite(T_ref_k)
    rejected["reference_not_finite"] += int(np.sum(bad_ref & keep))
    keep &= ~bad_ref

    if not np.any(keep):
        return replace(empty, per_file={probe.label: 0})

    kk = np.flatnonzero(keep)
    dirs = np.array(
        [probe.profile_direction(probe.profiles[p]) if 0 <= p < len(probe.profiles) else 0
         for p in pid[kk]],
        dtype=np.int8,
    )
    uid = np.array([f"{probe.label}#{p}" for p in pid[kk]], dtype=object)

    return PairSet(
        time=ref.time[k][kk],
        T_ref=T_ref_k[kk],
        L=L_k[kk],
        pressure=probe.pressure[j][kk],
        w=w_probe[j][kk],
        direction=dirs,
        profile_uid=uid,
        file_label=np.array([probe.label] * kk.size, dtype=object),
        channel=channel,
        lag=lag,
        rejected=rejected,
        per_file={probe.label: int(kk.size)},
    )


def build_pairs_multi(
    probes: list[ProbeSeries],
    ref: ReferenceSeries,
    channel: str,
    *,
    lag: float = 0.0,
    cfg: PairConfig | None = None,
) -> PairSet:
    """Pool pairs across every file in the deployment.

    Pooling *pairs* rather than per-file *coefficients* is deliberate: files
    with partial coverage give badly-constrained fits, and a median over those
    is dominated by its noisiest members.
    """
    cfg = cfg or PairConfig()
    out = PairSet(channel=channel, lag=lag)
    for p in probes:
        out = out.concat(build_pairs(p, ref, channel, lag=lag, cfg=cfg))
    out.lag = lag
    if len(out):
        out.corr = pair_correlation(out)
    return out


def build_pairs_iter(
    probe_iter,
    ref: ReferenceSeries,
    channel: str,
    *,
    lag: float = 0.0,
    cfg: PairConfig | None = None,
) -> PairSet:
    """Streaming form of :func:`build_pairs_multi`.

    A real deployment is ~1200 ``.p`` files of ~200k slow samples each --- some
    8 GB if every ``ProbeSeries`` is held at once, against a pair set that is a
    few MB.  The iterator is consumed one file at a time so only one file's
    arrays are ever live.
    """
    cfg = cfg or PairConfig()
    out = PairSet(channel=channel, lag=lag)
    for probe in probe_iter:
        out = out.concat(build_pairs(probe, ref, channel, lag=lag, cfg=cfg))
    out.lag = lag
    if len(out):
        out.corr = pair_correlation(out)
    return out


def pair_correlation(pairs: PairSet) -> float:
    """Pearson r between ``L`` and ``1/T_K``.

    The quantity the fit actually regresses, so it is the honest measure of
    whether a lag is any good.  ``perturb/fp07_cal.py`` computes a correlation
    and then never thresholds it, which is why a fabricated mid-gap ramp
    survives there at r = 0.02 (plan A12).
    """
    if len(pairs) < 3:
        return float("nan")
    y = 1.0 / (pairs.T_ref + 273.15)
    x = pairs.L
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def estimate_lag(
    probes: list[ProbeSeries],
    ref: ReferenceSeries,
    channel: str,
    *,
    cfg: PairConfig | None = None,
    max_lag: float = 20.0,
    step: float = 0.25,
) -> tuple[float, float, PairSet]:
    """Choose the lag that maximises |r| between ``L`` and ``1/T_K``.

    Scored on exactly the quantity the fit regresses, through exactly the same
    kernel --- rather than on a differenced, separately-filtered proxy.  That
    keeps the lag honest under sparse coverage, where a differenced
    cross-correlation has almost nothing to hold onto.

    Returns ``(lag, r, pairs_at_that_lag)``.  Callers must apply the ``min_corr``
    gate; a returned r below it means the lag is noise (R9).
    """
    cfg = cfg or PairConfig()
    lags = np.arange(-max_lag, max_lag + step / 2, step)
    scores: list[tuple[float, float]] = []
    best: tuple[float, float, PairSet] | None = None
    for lag in lags:
        ps = build_pairs_multi(probes, ref, channel, lag=float(lag), cfg=cfg)
        if len(ps) < 3:
            continue
        r = pair_correlation(ps)
        if not np.isfinite(r):
            continue
        scores.append((float(lag), abs(r)))
        if best is None or abs(r) > abs(best[1]):
            best = (float(lag), r, ps)
    if best is None:
        return 0.0, float("nan"), PairSet(channel=channel)

    # Refine to sub-step precision by fitting a parabola to |r| across the peak.
    # Without this the lag is quantised to `step`, and on a thermocline a
    # half-second lag error is a systematic dive-vs-climb temperature split of
    # order 0.02 K — larger than the noise floor, and it would be silently
    # absorbed into t_0.  (The refined value still cannot be trusted below the
    # reference's own sample interval; it is a peak location, not a resolution
    # claim.)
    refined = _parabolic_peak(scores, best[0], step)
    if refined is not None and abs(refined - best[0]) < step:
        ps = build_pairs_multi(probes, ref, channel, lag=refined, cfg=cfg)
        r = pair_correlation(ps)
        if len(ps) >= 3 and np.isfinite(r) and abs(r) >= abs(best[1]):
            best = (refined, r, ps)
    return best


def _parabolic_peak(scores: list[tuple[float, float]], peak: float, step: float) -> float | None:
    """Vertex of the parabola through the peak sample and its two neighbours."""
    by_lag = dict(scores)
    y0 = by_lag.get(round(peak - step, 6))
    y1 = by_lag.get(peak)
    y2 = by_lag.get(round(peak + step, 6))
    if y0 is None or y1 is None or y2 is None:
        # Dict keys come from arange and may not round-trip; fall back to search.
        lags = np.array([s[0] for s in scores])
        vals = np.array([s[1] for s in scores])
        i = int(np.argmin(np.abs(lags - peak)))
        if i == 0 or i == lags.size - 1:
            return None
        y0, y1, y2 = float(vals[i - 1]), float(vals[i]), float(vals[i + 1])
        step = float(lags[i + 1] - lags[i])
    denom = y0 - 2.0 * y1 + y2
    if denom == 0 or not np.isfinite(denom):
        return None
    return float(peak + 0.5 * step * (y0 - y2) / denom)


def estimate_clock_offset(
    probes: list[ProbeSeries],
    ref: ReferenceSeries,
    *,
    max_lag: float = 60.0,
    step: float = 0.5,
    width: float | None = None,
) -> tuple[float, float]:
    """Instrument-vs-glider clock offset from CTD pressure against instrument pressure.

    The T-vs-T lag conflates two physically distinct things: the clock offset
    between the two computers, and the CTD's own thermal/plumbing response
    (plan A9).  Pressure carries the first with none of the second --- both
    sensors see the same depth at the same instant --- so this pins the clock
    independently and lets the residual in the temperature lag be read as the
    sensor response.

    Returns ``(offset_s, r)``; ``(nan, nan)`` when the hotel file carries no
    CTD pressure.
    """
    if ref.pressure is None or not np.any(np.isfinite(ref.pressure)):
        return float("nan"), float("nan")

    t_all = np.concatenate([p.time for p in probes])
    p_all = np.concatenate([p.pressure for p in probes])
    order = np.argsort(t_all, kind="stable")
    t_all, p_all = t_all[order], p_all[order]
    if t_all.size < 2:
        return float("nan"), float("nan")

    w = width if width else max(ref.median_interval(), 1.0)
    lags = np.arange(-max_lag, max_lag + step / 2, step)
    best = (float("nan"), float("nan"))
    for lag in lags:
        centers = ref.time - lag
        inside = (centers >= t_all[0] + w) & (centers <= t_all[-1] - w)
        if int(np.sum(inside)) < 10:
            continue
        m, n = _boxcar_at(t_all, p_all, centers[inside], w)
        ok = np.isfinite(m) & (n > 0) & np.isfinite(ref.pressure[inside])
        if int(np.sum(ok)) < 10:
            continue
        a, b = m[ok], ref.pressure[inside][ok]
        if np.std(a) == 0 or np.std(b) == 0:
            continue
        r = float(np.corrcoef(a, b)[0, 1])
        if not np.isfinite(best[1]) or abs(r) > abs(best[1]):
            best = (float(lag), r)
    return best

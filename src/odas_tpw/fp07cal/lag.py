# Aug-2026, Claude and Pat Welch, pat@mousebrains.com
"""Lag estimation that works on a monotonic glider dive.

Why the obvious approach fails
------------------------------
A glider dive is a near-linear pressure ramp, and temperature is a smooth
monotone function of depth.  Cross-correlating two such series is degenerate:
shifting a straight line in time gives back the same straight line plus a
constant, and every correlation coefficient removes means, so ``r`` is ~1 at
*every* lag.  Measured on real osu685 data over a +/-30 s search:

===========================  ==========  ==================
score                        r at peak   range across +/-30 s
===========================  ==========  ==================
raw pressure                 1.000000    0.00002
raw L vs 1/T_ref             0.999997    0.00023
high-passed (30 s) L vs 1/T  0.972995    0.97
===========================  ==========  ==================

The first two look magnificent and mean nothing --- the argmax is picking noise
off a plateau thousands of times flatter than the correlation itself.  The peak
is only 0.5 s wide once the ramp is removed, and 3.5-9 s wide before.

**Timing information lives in the curvature, not the ramp.**  So every score
here is computed on high-passed series, and the gate is on the SHARPNESS of the
peak, never on the correlation value.  A high ``r`` is not evidence; a peak that
stands well above its own surroundings is.

All estimators here were validated by injecting a known shift into the
reference timestamps and confirming the recovered peak moves by exactly that
amount.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from odas_tpw.fp07cal.pairs import PairConfig, PairSet, build_pairs, build_pairs_multi
from odas_tpw.fp07cal.series import ProbeSeries, ReferenceSeries


def highpass(t: np.ndarray, x: np.ndarray, window_s: float) -> np.ndarray:
    """``x`` minus its running mean over *window_s* --- keeps the wiggle, drops the ramp.

    Assumes near-uniform sampling, which holds inside a ``.p`` file and inside a
    CTD span.  Edges are extended with the end values rather than zero-padded so
    the filter does not manufacture a step at the boundary.
    """
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if x.size < 3:
        return np.zeros_like(x)
    dt = float(np.median(np.diff(t)))
    if not np.isfinite(dt) or dt <= 0:
        return x - np.mean(x)
    n = max(3, int(round(window_s / dt)) | 1)
    if n >= x.size:
        return x - np.mean(x)
    pad = n // 2
    xp = np.concatenate([np.full(pad, x[0]), x, np.full(pad, x[-1])])
    return x - np.convolve(xp, np.ones(n) / n, mode="valid")[: x.size]


@dataclass
class LagResult:
    """A lag with the evidence for it.

    ``score`` (the correlation) is deliberately NOT the acceptance criterion ---
    see the module docstring.  ``dynamic_range`` and ``width`` are.
    """

    lag: float = float("nan")
    score: float = float("nan")
    dynamic_range: float = float("nan")
    """Peak score minus the lowest score in the searched window."""

    width: float = float("nan")
    """Width [s] of the region within 1% of the peak's dynamic range."""

    at_boundary: bool = False
    """True when the peak sits at the edge of the searched range."""

    n_pairs: int = 0
    lags: np.ndarray = field(default_factory=lambda: np.empty(0))
    scores: np.ndarray = field(default_factory=lambda: np.empty(0))
    label: str = "?"

    def trustworthy(self, *, min_range: float = 0.10, max_width: float = 5.0) -> bool:
        """A resolved peak: sharp, well above its surroundings, and interior.

        The boundary test is not a nicety.  A score that is still climbing at
        the edge of the search has no maximum inside it, and ``argmax`` then
        returns the edge itself --- a number that looks like a measurement and
        is only a statement about where the search stopped.  Measured in 150
        dbar bands on osu685, unresolved bands piled up at exactly +/-25.00 s
        (the search bound) and would otherwise have been averaged in as data.
        """
        return bool(
            np.isfinite(self.lag)
            and not self.at_boundary
            and np.isfinite(self.dynamic_range)
            and self.dynamic_range >= min_range
            and np.isfinite(self.width)
            and self.width <= max_width
        )

    def summary(self) -> str:
        if self.trustworthy():
            verdict = "ok"
        elif self.at_boundary:
            verdict = "NOT TRUSTWORTHY (peak at search boundary)"
        else:
            verdict = "NOT TRUSTWORTHY (flat peak)"
        return (
            f"{self.label}: lag {self.lag:+.2f} s, r={self.score:.6f}, "
            f"range={self.dynamic_range:.3g}, width={self.width:.1f} s — {verdict}"
        )


def _summarize(lags: np.ndarray, scores: np.ndarray, label: str, n_pairs: int) -> LagResult:
    if not np.any(np.isfinite(scores)):
        return LagResult(label=label, lags=lags, scores=scores)
    i = int(np.nanargmax(scores))
    peak = float(scores[i])
    lo = float(np.nanmin(scores))
    rng = peak - lo
    best = float(lags[i])
    if 0 < i < scores.size - 1 and np.all(np.isfinite(scores[i - 1 : i + 2])):
        d = scores[i - 1] - 2 * scores[i] + scores[i + 1]
        if d != 0:
            step = float(lags[i + 1] - lags[i])
            best = float(lags[i] + 0.5 * step * (scores[i - 1] - scores[i + 1]) / d)
    if rng > 0:
        near = lags[scores >= peak - 0.01 * rng]
        width = float(near.max() - near.min()) if near.size else 0.0
    else:
        width = float(lags.max() - lags.min())
    finite = np.flatnonzero(np.isfinite(scores))
    at_edge = bool(finite.size and (i <= finite[0] or i >= finite[-1]))
    return LagResult(
        lag=best, score=peak, dynamic_range=rng, width=width,
        at_boundary=at_edge, n_pairs=n_pairs, lags=lags, scores=scores, label=label,
    )


def temperature_lag(
    probes: list[ProbeSeries],
    ref: ReferenceSeries,
    channel: str,
    *,
    cfg: PairConfig | None = None,
    max_lag: float = 20.0,
    step: float = 0.25,
    detrend_s: float = 30.0,
) -> tuple[LagResult, PairSet]:
    """Total probe-vs-reference lag, scored on high-passed ``L`` vs ``1/T_ref``.

    This is clock skew + geometric transit + sensor response together.  Subtract
    :func:`pressure_offset` to isolate the sensor response.

    Detrending is done **per file** and the detrended residuals are only then
    pooled.  High-passing the concatenated series instead would run the
    running-mean window across the hours-long gaps between files, which
    destroys the very structure the score depends on --- measured on osu685,
    pooling before detrending collapsed the dynamic range from 0.97 to 0.003
    and made an otherwise sharp peak untrustworthy.
    """
    cfg = cfg or PairConfig()
    lags = np.arange(-max_lag, max_lag + step / 2, step)
    scores = np.full(lags.size, np.nan)
    for j, lag in enumerate(lags):
        A: list[np.ndarray] = []
        B: list[np.ndarray] = []
        for probe in probes:
            ps = build_pairs(probe, ref, channel, lag=float(lag), cfg=cfg)
            if len(ps) < 50:
                continue
            o = np.argsort(ps.time)
            t = ps.time[o]
            a = highpass(t, ps.L[o], detrend_s)
            b = highpass(t, 1.0 / (ps.T_ref[o] + 273.15), detrend_s)
            g = np.isfinite(a) & np.isfinite(b)
            if g.sum() > 10:
                A.append(a[g])
                B.append(b[g])
        if not A:
            continue
        a = np.concatenate(A)
        b = np.concatenate(B)
        if np.std(a) > 0 and np.std(b) > 0:
            scores[j] = abs(float(np.corrcoef(a, b)[0, 1]))

    res = _summarize(lags, scores, f"{channel} temperature lag", 0)
    if not np.isfinite(res.lag):
        return res, PairSet(channel=channel)
    pairs = build_pairs_multi(probes, ref, channel, lag=res.lag, cfg=cfg)
    res.n_pairs = len(pairs)
    return res, pairs


def pressure_offset(
    probes: list[ProbeSeries],
    ref: ReferenceSeries,
    *,
    max_lag: float = 40.0,
    step: float = 0.25,
    detrend_s: float = 30.0,
) -> LagResult:
    """Instrument-vs-reference offset from pressure alone.

    Pressure carries clock skew and the geometric sensor separation with no
    thermal physics in it, so subtracting this from
    :func:`temperature_lag` leaves the CTD's own response.

    On this platform the FP07 sits ~1 m ahead of the CTD along the vehicle
    axis, which contributes a transit of ``(1 m x sin(theta)) / w = 1 / U`` ---
    independent of pitch, and ~2.3 s at 0.44 m/s.  Because that transit affects
    the CTD's temperature and pressure equally, it cancels in the subtraction;
    what does NOT cancel is that the total lag is speed-dependent, so a single
    scalar lag is an approximation whose error shows up in the dive/climb split.

    ``detrend_s`` matters and should be cross-checked, not trusted blind: on
    osu685, windows of 20/30/60 s agree on +4.2..+5.0 s and each track an
    injected shift exactly, while 10 s locks onto a wrong correlation lobe at
    -21.7 s with a deceptively narrow peak.  Agreement ACROSS windows is the
    real check; the width gate alone will pass a confidently wrong lobe.
    """
    if ref.pressure is None or not np.any(np.isfinite(ref.pressure)):
        return LagResult(label="pressure offset (no CTD pressure)")

    lags = np.arange(-max_lag, max_lag + step / 2, step)
    scores = np.full(lags.size, np.nan)
    # Per-file, for the same reason as temperature_lag: a running mean over the
    # concatenated record would average across the gaps between files.
    segs = []
    for probe in probes:
        t0, t1 = float(probe.time[0]), float(probe.time[-1])
        m = (ref.time >= t0 - max_lag) & (ref.time <= t1 + max_lag)
        if int(m.sum()) < 100 or probe.time.size < 100:
            continue
        segs.append((
            probe.time,
            highpass(probe.time, probe.pressure, detrend_s),
            ref.time[m],
            highpass(ref.time[m], ref.pressure[m], detrend_s),
        ))
    if not segs:
        return LagResult(label="pressure offset (no overlapping data)")

    n_used = 0
    for j, lag in enumerate(lags):
        A: list[np.ndarray] = []
        B: list[np.ndarray] = []
        for tm, pm_hp, tr, pr_hp in segs:
            c = tr - lag
            ok = (c >= tm[0]) & (c <= tm[-1])
            if int(ok.sum()) < 50:
                continue
            a = np.interp(c[ok], tm, pm_hp)
            b = pr_hp[ok]
            g = np.isfinite(a) & np.isfinite(b)
            if g.sum() > 10:
                A.append(a[g])
                B.append(b[g])
        if not A:
            continue
        a = np.concatenate(A)
        b = np.concatenate(B)
        n_used = max(n_used, a.size)
        if np.std(a) > 0 and np.std(b) > 0:
            scores[j] = abs(float(np.corrcoef(a, b)[0, 1]))
    return _summarize(lags, scores, "pressure offset", n_used)

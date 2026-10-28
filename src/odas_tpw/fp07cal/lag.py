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

A narrow search window
----------------------
The sharpness gate was tuned on gliders, where the lag is several seconds and
the search spans +/-20 s: a resolved peak there falls by ~0.9 across the
window, and asking it to fall by at least 0.10 refuses a ramp's plateau. A VMP
scored against its own JAC CT has a lag of ~0.1 s and is searched over
+/-1 s, where even a perfectly identified peak falls by only 0.002-0.10
because its scored signals decorrelate over seconds (issue #238). The fall
measures the window as much as the peak, so a peak that falls by less than
0.10 in a search narrower than +/-20 s is judged on two things that do not
depend on the window (:meth:`LagResult.trustworthy`); a search of +/-20 s or
more measures the fall itself and is judged on it alone, as before:

* **curvature**, from the peak's own width: a parabola that falls by ``R``
  across the window is ``w = 2 sqrt(0.01 R / c)`` wide at 1% of ``R``, so its
  curvature is ``c = 0.04 R / w^2``, and ``c * 20^2`` is the fall it would show
  across the +/-20 s the gate was tuned on. ``w`` is the CONTIGUOUS run of
  lags within 1% around the peak, widened by a lag step at each edge so never
  sharper than the grid shows, and it must be measured, not just bounded:
  at least two lag steps wide, the only top within 1% anywhere in the search
  (two lobes are an ambiguity, not a peak), and at most a quarter of the
  search (a flat top). The same 0.10 applies. Measured: 2.3-49 on the 18
  peaks of eight VMP deployments, 5-12 steps wide, and 1.4e-4 (pressure) and
  5.2e-4 (temperature) on a glider's mean-removed series (osu684, 2025), the
  plateau the gate exists for;
* **stability**: the peak located again on two interleaved halves of the
  profiles must land within two lag steps of each other. A fixed tolerance,
  not the peak's width, which grows with the very ambiguity it should catch.
  A peak picked from noise does not replicate: the same mean-removed glider
  pressure gave halves 21.8 s apart; the VMP halves agree to 0.1-8 ms, under
  half a 1/64 s step.

Width relative to the window cannot replace the curvature: at 1% of the fall,
a parabola is about a tenth of any window wide, which is what the VMP peaks
show (0.070-0.094 of the window) and also the glider's mean-removed
temperature (0.087).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from odas_tpw.fp07cal.pairs import (
    PairConfig,
    PairSet,
    build_pairs,
    build_pairs_multi,
    prepare_probe,
)
from odas_tpw.fp07cal.series import ProbeSeries, ReferenceSeries

MIN_RANGE = 0.10
"""The fall a trustworthy peak shows across a +/-20 s search (the glider gate)."""

MAX_WIDTH_S = 5.0
"""[s] Widest peak (at 1% of its fall) trusted at all."""

REFERENCE_HALF_WINDOW_S = 20.0
"""[s] The half-window :data:`MIN_RANGE` was tuned on (the glider lag search)."""

HALF_WINDOW_TOL_S = 1e-6
"""[s] A search whose half-window is within this of 20 s IS a +/-20 s search."""

HALVES_TOLERANCE_STEPS = 2
"""Lag steps the two halves' peaks may differ by (VMP: <= 0.5 step measured)."""

NARROW_MIN_PEAK_STEPS = 2
"""The peak's contiguous 1% width, in lag steps, below which its curvature is not measured."""

NARROW_MAX_PEAK_FRACTION = 0.25
"""The widest contiguous peak, as a fraction of the search span (VMP: 0.07-0.09)."""


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
        return np.asarray(x - np.mean(x), dtype=np.float64)
    n = max(3, round(window_s / dt) | 1)
    if n >= x.size:
        return np.asarray(x - np.mean(x), dtype=np.float64)
    pad = n // 2
    xp = np.concatenate([np.full(pad, x[0]), x, np.full(pad, x[-1])])
    smooth = np.convolve(xp, np.ones(n) / n, mode="valid")[: x.size]
    return np.asarray(x - smooth, dtype=np.float64)


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
    """Width [s] spanned by ALL lags within 1% of the peak's dynamic range.

    A second top within 1% of the peak, however far away, widens it: the
    +/-5 s gate on it refuses an ambiguous glider peak.
    """

    at_boundary: bool = False
    """True when the peak sits at the edge of the searched range."""

    n_pairs: int = 0
    lags: np.ndarray = field(default_factory=lambda: np.empty(0))
    scores: np.ndarray = field(default_factory=lambda: np.empty(0))
    label: str = "?"
    peak_width: float = float("nan")
    """Width [s] of the CONTIGUOUS run of lags within 1% of the fall, around the peak.

    The peak itself, for its curvature.  Equal to :attr:`width` unless another
    top lies within 1% of the peak somewhere else in the search.
    """

    max_lag: float = float("nan")
    """[s] The configured half-window of the search."""

    half_lags: tuple[float, float] = (float("nan"), float("nan"))
    """The peak located on each of two interleaved halves of the profiles."""

    half_interior: bool = False
    """True when neither half's peak sits at the edge of the search."""

    exact: bool = False
    """Known by construction, not searched: the reference is on the instrument's clock."""

    def _interior(self, max_width: float) -> bool:
        return bool(
            np.isfinite(self.lag)
            and not self.at_boundary
            and np.isfinite(self.dynamic_range)
            and np.isfinite(self.width)
            and self.width <= max_width
        )

    def _step(self) -> float:
        return float(np.median(np.diff(self.lags))) if self.lags.size > 1 else float("nan")

    def fall_at_reference(self) -> float:
        """The fall the peak's curvature implies across +/-20 s: ``0.04 R / w^2 * 20^2``.

        ``w`` is the CONTIGUOUS width (:attr:`peak_width`).  It is read on the
        lag grid, so the true width can be up to one step wider at each edge;
        the curvature is taken at its LOWER bound, with ``w + 2 * step``.
        """
        R, w, step = self.dynamic_range, self.peak_width, self._step()
        if not (np.isfinite(R) and np.isfinite(w) and np.isfinite(step)) or R <= 0:
            return float("nan")
        w_hi = w + 2.0 * step
        if w_hi <= 0:
            return float("nan")
        return float(0.04 * R / w_hi**2 * REFERENCE_HALF_WINDOW_S**2)

    def halves_agree(self) -> bool:
        """Both halves' peaks interior and within :data:`HALVES_TOLERANCE_STEPS` lag steps.

        A fixed tolerance, not the peak's width: a peak picked from a flat top or
        between two lobes is as wide as the ambiguity it should be refused for.
        """
        a, b = self.half_lags
        step = self._step()
        if not (self.half_interior and np.isfinite(a) and np.isfinite(b) and np.isfinite(step)):
            return False
        return bool(abs(a - b) <= HALVES_TOLERANCE_STEPS * step * (1.0 + 1e-9))

    def search_half_window(self) -> float:
        """[s] The configured half-window; from the lag grid for a result built without it."""
        if np.isfinite(self.max_lag):
            return float(self.max_lag)
        if self.lags.size < 2:
            return float("nan")
        return float(self.lags[-1] - self.lags[0]) / 2.0

    def narrow_window(self, *, min_range: float = MIN_RANGE) -> bool:
        """True when the narrow-window rule decides: a fall under *min_range* in a
        search narrower than +/-20 s.

        A search of +/-20 s or wider MEASURES the fall the rule would
        extrapolate, so there a fall under *min_range* stays refused: the glider
        gate is exactly what it was.  Decided on the configured ``max_lag`` with
        a tolerance: ``np.arange`` puts the last lag of a +/-20 s search at
        19.99999999999993 s for a 0.2 s step.
        """
        return bool(
            np.isfinite(self.dynamic_range) and self.dynamic_range < min_range
            and self.search_half_window() < REFERENCE_HALF_WINDOW_S - HALF_WINDOW_TOL_S
        )

    def narrow_problem(self, *, min_range: float = MIN_RANGE) -> str:
        """Why the narrow-window rule refuses this peak; ``""`` when it accepts.

        In order: the grid must resolve the peak (its contiguous 1% width at
        least :data:`NARROW_MIN_PEAK_STEPS` steps, so its curvature is measured
        and not merely bounded); it must be the only top (no other lag within
        1% of it); it must not be a flat top (contiguous width at most
        :data:`NARROW_MAX_PEAK_FRACTION` of the search); its curvature must
        imply a fall of *min_range* across +/-20 s; and the halves must agree.
        """
        step, w, W = self._step(), self.peak_width, self.width
        if not (np.isfinite(step) and np.isfinite(w) and np.isfinite(W)):
            return "no score curve to judge"
        if w < NARROW_MIN_PEAK_STEPS * step * (1.0 - 1e-9):
            return (f"the peak is {w / step:.0f} lag step(s) wide, so the grid does not "
                    f"resolve its curvature")
        if W - w > 1e-9 * step:
            return (f"more than one top: lags within 1% of the peak span {W:.3g} s, the "
                    f"peak itself {w:.3g} s")
        span = 2.0 * self.search_half_window()
        if w > NARROW_MAX_PEAK_FRACTION * span:
            return f"a flat top {w:.3g} s wide in a {span:.3g} s search"
        fall = self.fall_at_reference()
        if not fall >= min_range:
            return f"flat: its curvature implies a fall of only {fall:.3g} over +/-20 s"
        if not self.halves_agree():
            a, b = self.half_lags
            return f"the halves do not find the peak again: {a:+.3f} / {b:+.3f} s"
        return ""

    def trustworthy(
        self, *, min_range: float = MIN_RANGE, max_width: float = MAX_WIDTH_S
    ) -> bool:
        """A resolved peak: sharp, well above its surroundings, and interior.

        The boundary test is not a nicety.  A score that is still climbing at
        the edge of the search has no maximum inside it, and ``argmax`` then
        returns the edge itself --- a number that looks like a measurement and
        is only a statement about where the search stopped.  Measured in 150
        dbar bands on osu685, unresolved bands piled up at exactly +/-25.00 s
        (the search bound) and would otherwise have been averaged in as data.

        "Well above its surroundings" is a fall of *min_range* across the
        search.  A peak that falls by less in a search narrower than +/-20 s,
        too narrow to show that fall, is trusted only when
        :meth:`narrow_problem` finds nothing wrong: a single top the grid
        resolves, whose curvature implies that fall across +/-20 s, AND which
        two halves of the profiles find again (see the module docstring).
        Neither test alone will do: a mean-removed ramp replicates on both
        halves, and a peak picked from noise can be one step wide.  A search of
        +/-20 s or more is judged on the fall alone, as it always was.
        """
        if self.exact:
            return True
        if not self._interior(max_width):
            return False
        if self.dynamic_range >= min_range:
            return True
        return bool(self.narrow_window(min_range=min_range)
                    and not self.narrow_problem(min_range=min_range))

    def resolution(self) -> dict:
        """The narrow-window evidence, for the coefficient record."""
        return {
            "rule": "curvature and halves (fall under 0.10 across the search)",
            "peak_width_s": float(self.peak_width),
            "fall_at_20s": self.fall_at_reference(),
            "half_lags_s": [float(x) for x in self.half_lags],
            "halves_agree": self.halves_agree(),
            "search_s": (
                [float(self.lags[0]), float(self.lags[-1])] if self.lags.size else None
            ),
            "problem": self.narrow_problem(),
        }

    def summary(self) -> str:
        if self.exact:
            return f"{self.label}: {self.lag:+.2f} s"
        if self.trustworthy():
            verdict = "ok"
            if self.narrow_window():
                a, b = self.half_lags
                verdict = (f"ok (narrow window: curvature implies a fall of "
                           f"{self.fall_at_reference():.3g} over +/-20 s; halves "
                           f"{a:+.3f} / {b:+.3f} s)")
        elif self.at_boundary:
            verdict = "NOT TRUSTWORTHY (peak at search boundary)"
        elif self._interior(MAX_WIDTH_S) and self.narrow_window():
            verdict = f"NOT TRUSTWORTHY ({self.narrow_problem()})"
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
        inside = np.nan_to_num(scores, nan=-np.inf) >= peak - 0.01 * rng
        a = b = i
        while a > 0 and inside[a - 1]:
            a -= 1
        while b < scores.size - 1 and inside[b + 1]:
            b += 1
        peak_width = float(lags[b] - lags[a])
    else:
        width = peak_width = float(lags.max() - lags.min())
    finite = np.flatnonzero(np.isfinite(scores))
    at_edge = bool(finite.size and (i <= finite[0] or i >= finite[-1]))
    return LagResult(
        lag=best, score=peak, dynamic_range=rng, width=width,
        at_boundary=at_edge, n_pairs=n_pairs, lags=lags, scores=scores, label=label,
        peak_width=peak_width,
    )


def _profile_starts(probes: list[ProbeSeries]) -> list[tuple[np.ndarray, int]]:
    """Per probe: its profiles' start times and the global ordinal of its first."""
    out, offset = [], 0
    for probe in probes:
        starts = np.array([probe.time[s] for s, _e in probe.profiles], dtype=np.float64)
        out.append((starts, offset))
        offset += len(probe.profiles)
    return out


def _half(starts: np.ndarray, offset: int, t: np.ndarray) -> np.ndarray:
    """Which interleaved half (0/1) each sample's profile falls in.

    Consecutive PAIRS of profiles alternate, ``(ordinal // 2) % 2``, so a
    glider yo's dive and climb stay together and a VMP's descents alternate.
    A sample before the first profile counts with it.
    """
    pidx = np.maximum(np.searchsorted(starts, t, side="right") - 1, 0)
    return np.asarray(((offset + pidx) // 2) % 2, dtype=np.int8)


def _score(a: np.ndarray, b: np.ndarray) -> float:
    if a.size > 10 and np.std(a) > 0 and np.std(b) > 0:
        return abs(float(np.corrcoef(a, b)[0, 1]))
    return float("nan")


def _with_halves(res: LagResult, lags: np.ndarray, halves: np.ndarray) -> LagResult:
    ra = _summarize(lags, halves[0], res.label, 0)
    rb = _summarize(lags, halves[1], res.label, 0)
    res.half_lags = (ra.lag, rb.lag)
    res.half_interior = bool(
        np.isfinite(ra.lag) and np.isfinite(rb.lag) and not ra.at_boundary and not rb.at_boundary
    )
    return res


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
    halves = np.full((2, lags.size), np.nan)
    # The bridge conversion and the single-pole filter are identical at every
    # trial lag; only the boxcar centres move.  Preparing once per probe turns
    # an O(n_lags) sweep over 200k-sample arrays into O(1).
    starts = _profile_starts(probes)
    prep = [(probe, prepare_probe(probe, channel, cfg), st)
            for probe, st in zip(probes, starts)]
    prep = [(probe, pp, st) for probe, pp, st in prep if pp is not None]
    for j, lag in enumerate(lags):
        A: list[np.ndarray] = []
        B: list[np.ndarray] = []
        H: list[np.ndarray] = []
        for probe, pp, (st, off) in prep:
            ps = build_pairs(probe, ref, channel, lag=float(lag), cfg=cfg, prepared=pp)
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
                H.append(_half(st, off, t[g] - lag))
        if not A:
            continue
        a = np.concatenate(A)
        b = np.concatenate(B)
        if np.std(a) > 0 and np.std(b) > 0:
            scores[j] = abs(float(np.corrcoef(a, b)[0, 1]))
        h = np.concatenate(H)
        for k in (0, 1):
            halves[k, j] = _score(a[h == k], b[h == k])

    res = _with_halves(_summarize(lags, scores, f"{channel} temperature lag", 0), lags, halves)
    res.max_lag = float(max_lag)
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
    halves = np.full((2, lags.size), np.nan)
    # Per-file, for the same reason as temperature_lag: a running mean over the
    # concatenated record would average across the gaps between files.
    segs = []
    for probe, (st, off) in zip(probes, _profile_starts(probes)):
        t0, t1 = float(probe.time[0]), float(probe.time[-1])
        m = (ref.time >= t0 - max_lag) & (ref.time <= t1 + max_lag)
        if int(m.sum()) < 100 or probe.time.size < 100:
            continue
        segs.append((
            probe.time,
            highpass(probe.time, probe.pressure, detrend_s),
            ref.time[m],
            highpass(ref.time[m], ref.pressure[m], detrend_s),
            st, off,
        ))
    if not segs:
        return LagResult(label="pressure offset (no overlapping data)")

    n_used = 0
    for j, lag in enumerate(lags):
        A: list[np.ndarray] = []
        B: list[np.ndarray] = []
        H: list[np.ndarray] = []
        for tm, pm_hp, tr, pr_hp, st, off in segs:
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
                H.append(_half(st, off, c[ok][g]))
        if not A:
            continue
        a = np.concatenate(A)
        b = np.concatenate(B)
        n_used = max(n_used, a.size)
        if np.std(a) > 0 and np.std(b) > 0:
            scores[j] = abs(float(np.corrcoef(a, b)[0, 1]))
        h = np.concatenate(H)
        for k in (0, 1):
            halves[k, j] = _score(a[h == k], b[h == k])
    res = _with_halves(_summarize(lags, scores, "pressure offset", n_used), lags, halves)
    res.max_lag = float(max_lag)
    return res


SAME_CLOCK_TOL_S = 1e-6
"""[s] A reference timestamp this close to an instrument sample time IS that sample."""

SAME_PRESSURE_TOL_DBAR = 1e-3
"""[dbar] ...and its pressure this close to the instrument's there is the same sensor."""


def shares_clock(
    probes: list[ProbeSeries],
    ref: ReferenceSeries,
    *,
    min_samples: int = 100,
    min_fraction: float = 0.99,
) -> bool:
    """Is the reference the instrument's own record, and so on its own clock?

    True when, inside the files' spans, the reference's samples ARE the
    instrument's: each timestamp within :data:`SAME_CLOCK_TOL_S` of one of its
    sample times, and the reference's pressure there within
    :data:`SAME_PRESSURE_TOL_DBAR` of the instrument's ``P`` --- a VMP fitted
    against its own JAC CT, whose reference is a subset of the very slow
    samples the thermistor is on.  The clock offset is then zero by
    construction, and :func:`pressure_offset` (a +/-40 s search over every
    file) need not run (issue #238).

    Both halves are needed.  Matching timestamps alone are not proof: a CTD
    resampled onto the instrument's grid, or a synthetic one stamped in whole
    seconds on a 4 Hz grid (``fp07-cal demo``), lands on the sample times
    while its values sit on another clock --- and then its pressure, from a
    different sensor and a shifted time, does not match.  A separate CTD on its
    own clock puts a stamp within 1 us of a 64 Hz sample with probability
    ~1.3e-4, and its pressure sensor never agrees with the instrument's to
    1 mm at a hundred samples.  No reference pressure, no decision: the search
    runs (and returns at once).
    """
    if ref.pressure is None:
        return False
    n_in = n_hit = 0
    for probe in probes:
        t = np.asarray(probe.time, dtype=np.float64)
        if t.size < 2:
            continue
        lo = int(np.searchsorted(ref.time, t[0], side="left"))
        hi = int(np.searchsorted(ref.time, t[-1], side="right"))
        if hi <= lo:
            continue
        rt = ref.time[lo:hi]
        rp = np.asarray(ref.pressure[lo:hi], dtype=np.float64)
        i = np.clip(np.searchsorted(t, rt), 1, t.size - 1)
        j = np.where(np.abs(rt - t[i - 1]) <= np.abs(rt - t[i]), i - 1, i)
        same_t = np.abs(rt - t[j]) <= SAME_CLOCK_TOL_S
        with np.errstate(invalid="ignore"):
            same_p = np.abs(rp - np.asarray(probe.pressure, dtype=np.float64)[j]) \
                <= SAME_PRESSURE_TOL_DBAR
        n_in += int(rt.size)
        n_hit += int(np.count_nonzero(same_t & same_p))
    return bool(n_in >= min_samples and n_hit >= min_fraction * n_in)

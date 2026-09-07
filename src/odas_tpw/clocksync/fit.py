# Aug-2026, Claude and Pat Welch, pat@mousebrains.com
"""One clock model per file: an offset and a rate, weighted by real sigmas.

The MR clock does two things that a single number cannot describe.  It **jumps
between files** --- so the offset is a per-file quantity, never a per-deployment
constant --- and it **runs at the wrong rate**, so within a 3 h file the error
drifts.  The model here is therefore, for file *j*,

    correction(t) = offset_j + rate_j * (t - t0_j)

fitted from several independent lag estimates spread across the file.  ``rate``
is dimensionless (s/s); 1e-5 is 0.86 s/day.

Two things distinguish this from fitting a line to segment lags and stopping:

**The weights are real.**  Each lag arrives from
:func:`odas_tpw.clocksync.lag.estimate_lag` with a sigma derived from the
cross-spectral coherence, so a segment measured during a flat calm is weighted
down instead of counting the same as one measured under a good swell.  The
MATLAB had only the scatter of equally-weighted integer-sample lags, which
conflates estimator noise with genuine drift.

**The residual is diagnostic.**  If reduced chi-squared is ~1 the linear clock
model is adequate and the covariance means what it says.  If it is large the
clock did something else inside the file --- a step, a reset --- and the fitted
line is a summary of a thing that is not a line.  That is reported, not hidden
in a wider error bar.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from odas_tpw.clocksync.lag import LagResult, estimate_lag
from odas_tpw.clocksync.series import PressureSeries


@dataclass
class FileFit:
    """The clock model for one target file, with the evidence behind it."""

    name: str = ""
    source: str = ""
    t0: float = np.nan
    t1: float = np.nan
    offset: float = np.nan
    offset_sigma: float = np.nan
    rate: float = np.nan
    rate_sigma: float = np.nan
    covariance: float = np.nan
    chi2: float = np.nan
    n_windows: int = 0
    n_used: int = 0
    coherence: float = np.nan
    ok: bool = False
    why: str = ""
    lags: list[LagResult] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)

    def correction(self, t: np.ndarray) -> np.ndarray:
        """Seconds to add to the target's clock at epoch time *t*."""
        return self.offset + self.rate * (np.asarray(t, dtype=np.float64) - self.t0)

    @property
    def drift_per_day(self) -> float:
        return self.rate * 86400.0


def window_pairs(
    ref: PressureSeries,
    tgt: PressureSeries,
    fs: float,
    window: float,
    step: float | None = None,
    max_gap: float = 1.0,
    min_fill: float = 0.8,
):
    """Yield gap-free (t_mid, ref_chunk, tgt_chunk) pairs on a shared grid.

    Both series are put on the *same* uniform grid at *fs* --- built from the
    reference's own sample times, so the reference is never resampled onto the
    target's questionable clock.  Windows that are more than ``1 - min_fill``
    empty are skipped rather than filled: a lag fitted through a hole is a lag
    fitted through an assumption.
    """
    step = step or window
    t0 = max(ref.span[0], tgt.span[0])
    t1 = min(ref.span[1], tgt.span[1])
    if not np.isfinite(t0) or t1 - t0 < window:
        return
    r = ref.clip(t0, t1).uniform(fs, max_gap=max_gap)
    g = tgt.clip(t0 - max_gap, t1 + max_gap)
    # Target onto the reference's grid: decimate down, never interpolate the
    # reference up (the fp07cal convention -- match bandwidth to the thing you
    # trust, and invent nothing).
    tg = PressureSeries(
        r.t,
        np.interp(r.t, g.t, g.p, left=np.nan, right=np.nan),
        tgt.name,
        tgt.source,
        fs,
    )
    if g.t.size > 1:
        idx = np.searchsorted(g.t, r.t).clip(1, g.t.size - 1)
        near = np.minimum(np.abs(r.t - g.t[idx - 1]), np.abs(r.t - g.t[idx]))
        tg.p[near > max_gap] = np.nan

    n = round(window * fs)
    stride = max(1, round(step * fs))
    max_gap_samples = max(1, int(np.floor(max_gap * fs)))
    for i in range(0, max(0, r.t.size - n + 1), stride):
        a, b = r.p[i : i + n], tg.p[i : i + n]
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < min_fill * n:
            continue
        if not ok.all():
            # The aggregate min_fill test does NOT imply every remaining hole is
            # short: a single 100 s hole in a 900 s window is 11% missing, well
            # inside a 0.8 fill, and the old comment asserting "already under
            # the max_gap limit" was simply false -- the window was interpolated
            # straight across it and passed on for lag and uncertainty
            # estimation with a fabricated segment nothing identified
            # (issue #180 F17). Check the longest CONSECUTIVE run explicitly.
            if _longest_gap(ok) > max_gap_samples:
                continue
            # Short interior holes only; bridge them so the FFT has something
            # continuous, and only because the window is mostly real data.
            a = np.interp(np.arange(n), np.flatnonzero(ok), a[ok])
            b = np.interp(np.arange(n), np.flatnonzero(ok), b[ok])
        yield float(r.t[i] + 0.5 * window), a, b


def _longest_gap(ok: np.ndarray) -> int:
    """Longest run of consecutive False in *ok* [samples]."""
    if ok.all():
        return 0
    # Run-length over the inverted mask via the indices of the True entries.
    idx = np.flatnonzero(ok)
    if idx.size == 0:
        return int(ok.size)
    interior = int(np.max(np.diff(idx)) - 1) if idx.size > 1 else 0
    lead = int(idx[0])
    trail = int(ok.size - 1 - idx[-1])
    return max(interior, lead, trail)


def fit_file(
    ref: PressureSeries,
    tgt: PressureSeries,
    fs: float,
    *,
    window: float = 900.0,
    step: float | None = None,
    band: tuple[float, float] = (0.05, 0.35),
    max_lag: float = 900.0,   # wide by default: see the runbook, section 2
    min_windows: int = 3,
    max_gap: float = 1.0,
    fit_rate: bool = True,
    **lag_kw,
) -> FileFit:
    """Solve offset (and rate) for one target file against the reference."""
    out = FileFit(name=tgt.name, source=tgt.source, t0=tgt.span[0], t1=tgt.span[1])
    for t_mid, a, b in window_pairs(ref, tgt, fs, window, step, max_gap=max_gap):
        out.lags.append(
            estimate_lag(a, b, fs, band=band, max_lag=max_lag, t_mid=t_mid, **lag_kw)
        )
    out.n_windows = len(out.lags)
    good = [x for x in out.lags if x.ok and np.isfinite(x.sigma) and x.sigma > 0]
    out.n_used = len(good)
    if not good:
        why = [x.why for x in out.lags]
        # Summarise, do not enumerate: one reason per KIND, with a count.
        # Dumping every window's message produced 2000-character lines that
        # buried the one fact that mattered (the search window was too small).
        kinds: dict[str, int] = {}
        for reason in why:   # not `w`: that name is the weight vector below
            key = reason.split(":")[0].strip() if reason else "unknown"
            kinds[key] = kinds.get(key, 0) + 1
        detail = "; ".join(f"{k} (x{v})" if v > 1 else k
                           for k, v in sorted(kinds.items(), key=lambda kv: -kv[1]))
        out.why = f"no usable window of {out.n_windows}" + (f": {detail}" if detail else "")
        return out
    out.coherence = float(np.median([x.coherence for x in good]))

    t = np.array([x.t_mid for x in good]) - out.t0
    y = np.array([x.lag for x in good])
    w = 1.0 / np.array([x.sigma for x in good]) ** 2

    if len(good) < min_windows or not fit_rate:
        # Not enough lever for a rate: report the weighted mean offset and say
        # so, rather than fitting a slope through two points and pretending.
        out.offset = float(np.sum(w * y) / np.sum(w))
        out.offset_sigma = float(1.0 / np.sqrt(np.sum(w)))
        out.rate, out.rate_sigma, out.covariance = 0.0, np.nan, 0.0
        if len(good) > 1:
            resid = y - out.offset
            out.chi2 = float(np.sum(w * resid**2) / (len(good) - 1))
        if fit_rate and len(good) < min_windows:
            out.flags.append(f"only {len(good)} windows: rate not fitted, offset only")
        out.ok = True
        return out

    A = np.column_stack([np.ones_like(t), t])
    W = np.diag(w)
    cov = np.linalg.inv(A.T @ W @ A)
    beta = cov @ (A.T @ W @ y)
    resid = y - A @ beta
    dof = len(good) - 2
    out.offset, out.rate = float(beta[0]), float(beta[1])
    out.offset_sigma = float(np.sqrt(cov[0, 0]))
    out.rate_sigma = float(np.sqrt(cov[1, 1]))
    out.covariance = float(cov[0, 1])
    out.chi2 = float(np.sum(w * resid**2) / dof) if dof > 0 else np.nan
    if np.isfinite(out.chi2) and out.chi2 > 10.0:
        out.flags.append(
            f"reduced chi2 {out.chi2:.1f}: the clock did not drift linearly inside this file"
        )
    out.ok = True
    return out

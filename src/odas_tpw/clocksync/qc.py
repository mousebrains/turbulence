# Aug-2026, Claude and Pat Welch, pat@mousebrains.com
"""Cross-file checks --- the ones a per-file fit cannot make about itself.

A per-file solve is blind to its neighbours, and that is where the remaining
failure mode lives: a cycle slip, or a lag locked onto the wrong side lobe,
produces a perfectly self-consistent file fit with a small sigma that happens
to be one wave period away from the truth.  Nothing inside that file objects.

The check that catches it is physical: **a clock drifts, it does not teleport.**
Between two consecutive files the offset may step (the MR clock is known to
jump at file boundaries) but the steps should be small and the sequence should
have structure.  A single file sitting a whole wave period off its neighbours
is an estimator artefact, not a clock.

Everything here is robust --- median and MAD, never mean and sd.  A handful of
slipped files is exactly the heavy tail that would drag a mean estimate onto
the outliers it is supposed to find.
"""

from __future__ import annotations

import numpy as np

from odas_tpw.clocksync.fit import FileFit

# Absolute floor [s] on the continuity scale. A deployment whose offsets are
# identical to the last bit still cannot resolve a slip below the timing
# resolution the method claims, and dividing by a scale of zero would flag
# arithmetic noise. 1 ms is below the few-tens-of-ms this package reports.
_SCALE_FLOOR_S = 1e-3


def _mad(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def check_continuity(
    fits: list[FileFit],
    *,
    n_sigma: float = 5.0,
    window: int = 9,
    wave_period: float | None = None,
) -> list[FileFit]:
    """Flag files whose offset departs from the local trend of their neighbours.

    ``window`` is the number of files in the running median (odd).  The flag is
    advisory --- it is written into ``fit.flags`` and never silently drops a
    file, because on a real deployment a genuine clock reset looks exactly like
    an outlier and only a person can tell them apart.

    When *wave_period* is given, a departure within 25% of an integer number of
    wave periods is named as a probable **cycle slip**, which is actionable
    (rerun that file with a tighter ``max_lag``) rather than merely odd.
    """
    ok = [f for f in fits if f.ok and np.isfinite(f.offset)]
    if len(ok) < 5:
        return fits
    ok.sort(key=lambda f: f.t0)
    off = np.array([f.offset for f in ok])
    half = max(1, window // 2)
    trend = np.array(
        [np.median(off[max(0, i - half) : i + half + 1]) for i in range(off.size)]
    )
    resid = off - trend

    # The scale each residual is judged against EXCLUDES that residual, and is
    # floored by the fits' own measurement uncertainty.
    #
    # With ten otherwise-identical offsets and one slip, MAD is exactly zero and
    # the old fallback was np.std(resid) -- computed over the outlier itself.
    # For residuals [A, 0, ..., 0] that ratio is N/sqrt(N-1), which is 3.33 at
    # N=10 REGARDLESS OF A: a 10 s slip and a 10,000 s slip were both silently
    # unflagged under a 5-sigma threshold (issue #180 F18). An indeterminate
    # scale is not evidence of continuity.
    sigmas = np.array([f.offset_sigma for f in ok], dtype=np.float64)
    finite_sigma = sigmas[np.isfinite(sigmas) & (sigmas > 0)]
    sigma_floor = float(np.median(finite_sigma)) if finite_sigma.size else 0.0
    for i, (f, r) in enumerate(zip(ok, resid, strict=True)):
        loo = np.delete(resid, i)  # leave-one-out: the point cannot absolve itself
        scale = _mad(loo)
        if not np.isfinite(scale) or scale <= 0:
            scale = float(np.std(loo))
        scale = max(scale, sigma_floor, _SCALE_FLOOR_S)
        if abs(r) <= n_sigma * scale:
            continue
        msg = (
            f"offset {f.offset:+.3f} s departs from the local trend by "
            f"{r:+.3f} s ({abs(r) / scale:.1f} robust sigma)"
        )
        if wave_period:
            cycles = r / wave_period
            if abs(cycles) >= 0.75 and abs(cycles - round(cycles)) < 0.25:
                msg += f" -- {round(cycles):+d} wave periods: probable CYCLE SLIP"
        f.flags.append(msg)
    return fits


def summarize(fits: list[FileFit]) -> dict:
    """Deployment-level numbers, robust throughout."""
    ok = [f for f in fits if f.ok]
    out: dict = {
        "n_files": len(fits),
        "n_solved": len(ok),
        "n_flagged": sum(1 for f in fits if f.flags),
    }
    if not ok:
        return out
    off = np.array([f.offset for f in ok])
    sig = np.array([f.offset_sigma for f in ok])
    rate = np.array([f.rate for f in ok if np.isfinite(f.rate)])
    steps = np.diff(off[np.argsort([f.t0 for f in ok])])
    out.update(
        offset_median=float(np.median(off)),
        offset_mad=_mad(off),
        offset_min=float(off.min()),
        offset_max=float(off.max()),
        sigma_median=float(np.median(sig)),
        sigma_max=float(np.nanmax(sig)),
        step_median=float(np.median(np.abs(steps))) if steps.size else np.nan,
        step_max=float(np.max(np.abs(steps))) if steps.size else np.nan,
        coherence_median=float(np.median([f.coherence for f in ok])),
        windows_used=int(sum(f.n_used for f in ok)),
        windows_total=int(sum(f.n_windows for f in ok)),
    )
    if rate.size:
        out.update(
            rate_median=float(np.median(rate)),
            rate_mad=_mad(rate),
            drift_median_s_per_day=float(np.median(rate) * 86400.0),
        )
    return out

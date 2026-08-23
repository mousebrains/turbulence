# Aug-2026, Claude and Pat Welch, pat@mousebrains.com
"""Lag from dT/dz — depth-space matching, independent of the time-domain score.

Why this is the better-posed question
-------------------------------------
A timing error ``tau`` does not blur a profile, it *displaces* it: the sensor
appears to have been at depth ``z + w*tau`` when it reported.  Since the
vertical velocity ``w`` reverses between dive and climb, the same ``tau``
pushes the dive and climb ``T(z)`` curves in OPPOSITE directions, opening a
hysteresis loop of width ``2*w*tau``.  The correct lag is the one that closes
it.

Two things make this strictly better than correlating the time series:

* It works on a **monotonic** profile.  ``T(t)`` on a glider dive is smooth and
  monotone, so time-domain correlation is nearly degenerate (see
  :mod:`odas_tpw.fp07cal.lag`); but ``dT/dz`` has real structure --- the
  thermocline is a peak, not a ramp --- so there is genuine signal to align.
* It is **self-referencing**.  Closing the dive/climb loop needs no external
  clock at all, so it measures the lag without assuming the CTD's timestamps
  are right.

Two estimators are provided:

``hysteresis_lag``
    Uses one sensor against itself, dive versus climb.  Needs no reference.
``gradient_lag``
    Matches the probe's ``dT/dz`` against the reference's ``dT/dz``, which is
    the cross-sensor version and is what pins the probe to the CTD.

These are independent of the high-passed time-domain score, so agreement
between them is real evidence rather than one method's artifact repeated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from odas_tpw.fp07cal.lag import LagResult, _summarize
from odas_tpw.fp07cal.logr import log_r, temperature
from odas_tpw.fp07cal.pairs import PairConfig, build_pairs
from odas_tpw.fp07cal.series import ProbeSeries, ReferenceSeries


@dataclass
class GradientProfile:
    """``dT/dz`` on a uniform pressure grid, for one leg of one yo."""

    edges: np.ndarray
    dTdz: np.ndarray
    direction: int
    n: np.ndarray


def _grad_on_grid(
    P: np.ndarray, T: np.ndarray, edges: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Bin-average T onto a pressure grid, then difference — dT/dz [K/dbar]."""
    idx = np.digitize(P, edges) - 1
    nb = edges.size - 1
    ok = (idx >= 0) & (idx < nb) & np.isfinite(T) & np.isfinite(P)
    sums = np.zeros(nb)
    cnts = np.zeros(nb)
    np.add.at(sums, idx[ok], T[ok])
    np.add.at(cnts, idx[ok], 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        Tb = np.where(cnts > 0, sums / np.maximum(cnts, 1.0), np.nan)
    dz = float(np.median(np.diff(edges)))
    g = np.gradient(Tb, dz)
    return g, cnts


def hysteresis_lag(
    probes: list[ProbeSeries],
    channel: str,
    *,
    max_lag: float = 20.0,
    step: float = 0.25,
    bin_dbar: float = 2.0,
    min_w: float = 0.05,
) -> LagResult:
    """Lag that closes the dive/climb ``T(z)`` hysteresis loop.

    Reference-free: it asks only that the same sensor report the same water
    column going down as coming up.  A residual lag ``tau`` displaces the two
    legs by ``+w*tau`` and ``-w*tau``, so the score is the agreement between the
    dive and climb ``dT/dz`` profiles once each has been shifted by its own
    ``w*tau``.

    Because it needs no CTD, this is the one lag estimate that is available on
    every yo --- including the ones where the CT was switched off.
    """
    lags = np.arange(-max_lag, max_lag + step / 2, step)
    scores = np.full(lags.size, np.nan)

    legs: list[tuple[np.ndarray, np.ndarray, np.ndarray, int]] = []
    for probe in probes:
        if channel not in probe.counts:
            continue
        L, clipped = log_r(probe.counts[channel], probe.bridge[channel])
        T = temperature(L, probe.factory[channel])
        T = np.where(clipped, np.nan, T)
        dt = 1.0 / probe.fs
        w = np.gradient(probe.pressure, dt)
        for span in probe.profiles:
            s, e = span
            d = probe.profile_direction(span)
            if d == 0 or e - s < 100:
                continue
            sl = slice(s, e + 1)
            if np.nanmedian(np.abs(w[sl])) < min_w:
                continue
            legs.append((probe.pressure[sl], T[sl], w[sl], d))
    if len(legs) < 2:
        return LagResult(label=f"{channel} hysteresis lag (too few legs)")

    lo = max(1.0, min(np.nanmin(p) for p, _t, _w, _d in legs))
    hi = min(np.nanmax(p) for p, _t, _w, _d in legs)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 10 * bin_dbar:
        return LagResult(label=f"{channel} hysteresis lag (no common depth range)")
    edges = np.arange(lo, hi + bin_dbar, bin_dbar)

    for j, lag in enumerate(lags):
        down: list[np.ndarray] = []
        up: list[np.ndarray] = []
        for P, T, w, d in legs:
            # Correct each sample to the depth the sensor was ACTUALLY at when
            # the water it is reporting passed it.
            g, cnt = _grad_on_grid(P - w * lag, T, edges)
            g = np.where(cnt > 0, g, np.nan)
            (down if d > 0 else up).append(g)
        if not down or not up:
            continue
        a = np.nanmean(np.vstack(down), axis=0)
        b = np.nanmean(np.vstack(up), axis=0)
        g = np.isfinite(a) & np.isfinite(b)
        if g.sum() < 5:
            continue
        # Score = agreement between the two legs' gradient profiles.  1 means
        # the loop is closed.
        num = float(np.nanmean((a[g] - b[g]) ** 2))
        den = float(np.nanvar(np.concatenate([a[g], b[g]])))
        scores[j] = 1.0 - num / den if den > 0 else np.nan

    return _summarize(lags, scores, f"{channel} hysteresis lag (dT/dz, dive vs climb)", len(legs))


def gradient_lag(
    probes: list[ProbeSeries],
    ref: ReferenceSeries,
    channel: str,
    *,
    cfg: PairConfig | None = None,
    max_lag: float = 20.0,
    step: float = 0.25,
    bin_dbar: float = 2.0,
) -> LagResult:
    """Lag that best aligns the probe's ``dT/dz`` with the reference's.

    The cross-sensor form: matched in depth space, where a timing error is a
    displacement rather than a smear, and on the gradient, where the thermocline
    supplies structure that the monotone ``T(z)`` itself does not.
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
            P = ps.pressure
            good = np.isfinite(P)
            if good.sum() < 50:
                continue
            lo, hi = float(np.min(P[good])), float(np.max(P[good]))
            if hi - lo < 10 * bin_dbar:
                continue
            edges = np.arange(lo, hi + bin_dbar, bin_dbar)
            T_probe = temperature(ps.L, probe.factory[channel])
            ga, ca = _grad_on_grid(P, T_probe, edges)
            gb, cb = _grad_on_grid(P, ps.T_ref, edges)
            g = (ca > 0) & (cb > 0) & np.isfinite(ga) & np.isfinite(gb)
            if g.sum() >= 5:
                A.append(ga[g])
                B.append(gb[g])
        if not A:
            continue
        a = np.concatenate(A)
        b = np.concatenate(B)
        if np.std(a) > 0 and np.std(b) > 0:
            scores[j] = abs(float(np.corrcoef(a, b)[0, 1]))
    return _summarize(lags, scores, f"{channel} gradient lag (dT/dz vs reference)", 0)

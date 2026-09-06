#!/usr/bin/env python3
"""Will recording dives actually break the alpha / U_EM-scale degeneracy?

OSU are collecting dives on the current deployment: ~25 deg down-pitch on the
dive, ~35-40 deg up-pitch on the climb. This asks -- BEFORE the data exists --
whether that geometry is enough, so the answer is not "we found out afterwards".

PRE-REGISTERED. The dives have since flown, but the data had NOT arrived when
this was committed (2026-09-06), which is the whole point: the forecast in
docs/glider_dive_geometry_forecast.md is on the record before it can be
tuned to the answer. Do not edit the constants below to match whatever the
data turns out to say -- add a comparison instead.

The observable, per profile, is

    excess(theta) = U_EM * sin(theta) / |W| = b * sin(theta) / sin(theta + alpha)

Two unknowns. A single theta cannot separate them; two well-separated thetas
can, through the curvature of sin(). How WELL is the question, and that depends
on the separation, the per-profile scatter, and how many profiles of each.

Per-profile scatter is taken from osu685, where 721 climb profiles gave a
bootstrap SEM of 0.0005 on the median excess -> sigma ~ 1.3% per profile.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

SIGMA = 0.013          # per-profile scatter in `excess`, measured on osu685
ALPHA_TRUE = 3.0       # deg
B_TRUE = 1.10          # the EM scale we are trying to measure


def excess(theta_deg, alpha_deg, b):
    th = np.radians(theta_deg)
    return b * np.sin(th) / np.sin(th + np.radians(alpha_deg))


def fit_once(theta, obs, rng):
    def resid(p):
        return excess(theta, p[0], p[1]) - obs
    try:
        out = least_squares(resid, x0=[3.0, 1.05], bounds=([-20, 0.5], [30, 2.0]))
        return out.x
    except Exception:
        return np.array([np.nan, np.nan])


def trial(config, n_mc=400, seed=11):
    """config = list of (theta_deg, n_profiles)."""
    rng = np.random.default_rng(seed)
    theta = np.concatenate([np.full(n, t) for t, n in config])
    truth = excess(theta, ALPHA_TRUE, B_TRUE)
    A, B = [], []
    for _ in range(n_mc):
        obs = truth + rng.normal(0, SIGMA, theta.size)
        a, b = fit_once(theta, obs, rng)
        if np.isfinite(a):
            A.append(a)
            B.append(b)
    A, B = np.array(A), np.array(B)
    return A, B


CONFIGS = [
    ("osu685 as flown (climbs only, 40.4-48.0)",
     [(40.4, 90), (44.4, 540), (48.0, 90)]),
    ("osu684 post-failure (climbs only, 35.8-39.4)",
     [(35.8, 10), (37.3, 33), (39.4, 10)]),
    ("PLANNED: dives 25 + climbs 37.5, 10% dives",
     [(25.0, 72), (37.5, 648)]),
    ("PLANNED: dives 25 + climbs 37.5, 25% dives",
     [(25.0, 180), (37.5, 540)]),
    ("PLANNED: dives 25 + climbs 37.5, 50% dives",
     [(25.0, 360), (37.5, 360)]),
    ("if climbs were 45 instead of 37.5, 25% dives",
     [(25.0, 180), (45.0, 540)]),
]

print(__doc__)
print(f"truth: alpha = {ALPHA_TRUE} deg, b = {B_TRUE}; "
      f"per-profile sigma = {SIGMA} ({100*SIGMA:.1f}%)\n")
print(f"{'configuration':>46s} {'sep':>5s} {'alpha 16/50/84':>22s} "
      f"{'b 16/50/84':>22s} {'corr':>6s}")
print("-" * 108)
for name, cfg in CONFIGS:
    A, B = trial(cfg)
    th = [t for t, _ in cfg]
    sep = max(th) - min(th)
    qa = np.percentile(A, [16, 50, 84])
    qb = np.percentile(B, [16, 50, 84])
    corr = np.corrcoef(A, B)[0, 1]
    print(f"{name:>46s} {sep:5.1f} "
          f"{qa[0]:6.2f}/{qa[1]:5.2f}/{qa[2]:5.2f}      "
          f"{qb[0]:6.3f}/{qb[1]:5.3f}/{qb[2]:5.3f}   {corr:+.3f}")

print("\nRead the alpha column as the 68% interval you would recover.")
print("An interval of +/-1 deg on alpha is +/-2% on U and +/-8% on epsilon.")

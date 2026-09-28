#!/usr/bin/env python3
"""Angle of attack from the Aquadopp HR beams, anchored on MEASURED pressure.

The wrap branch is chosen with a prior built only from quantities the Aquadopp
and the glider measure directly -- the depth rate dh/dt and the pitch -- never
from a flight model:

    U_prior = |dh/dt| / sin(|pitch| + alpha),  alpha starting at 0

Predicted beam velocities from that prior pick the nearest branch
(b_true = b_meas + 2*Va*round((b_pred - b_meas)/(2*Va))); the DATA then set the
velocity vector within +-Va, and alpha is recomputed and iterated. The prior
only has to be right to better than Va/cos(25 deg) ~ 0.14 m/s, and it is
anchored to a measured vertical velocity, so a few degrees of alpha error moves
it by ~0.05 m/s.

What comes out -- alpha and U -- is then an INDEPENDENT check on the Merckelbach
flight model, which has no measured speed anywhere in its calibration.
"""
from __future__ import annotations

import numpy as np

from beams import enu_to_xyz
from read_prf import config, read

VA = 0.1285           # m/s ambiguity velocity (beam saturation)
CELLS = slice(4, 16)  # skip nearest cells (hull) and the noisy far end
MIN_CORR = 50


def solve(n_iter=4):
    c = config()
    d = read()
    T = c["transform"]
    Tinv = np.linalg.inv(T)
    t, P = d["time"], d["pressure_dbar"]
    pitch = d["pitch_deg"]
    xyz = enu_to_xyz(d["vel"], d["heading_deg"], d["pitch_deg"], d["roll_deg"])
    beam = np.einsum("ij,njc->nic", Tinv, xyz)
    b = np.where(d["corr"][:, :, CELLS] > MIN_CORR, beam[:, :, CELLS], np.nan)
    bm = np.nanmean(b, axis=2)                       # (n, 3)

    dPdt = np.gradient(P, t)
    w = np.convolve(dPdt, np.ones(9) / 9, mode="same")   # m/s, + = sinking
    glide = np.abs(w) > 0.05
    ok = glide & np.isfinite(bm).all(axis=1) & (P > 10)

    alpha = np.zeros(t.size)
    for _ in range(n_iter):
        gl = np.radians(np.abs(pitch) + alpha)            # glide angle
        U_prior = np.where(np.sin(gl) > 0.1, np.abs(w) / np.maximum(np.sin(gl), 0.1), np.nan)
        # platform-relative flow in the instrument frame: along -x, with alpha in x-z
        a = np.radians(alpha)
        v_inst = np.stack([-U_prior * np.cos(a), np.zeros_like(U_prior), -U_prior * np.sin(a) * np.sign(w)], axis=1)
        b_pred = np.einsum("ij,nj->ni", Tinv, v_inst)
        k = np.round((b_pred - bm) / (2 * VA))
        b_true = bm + 2 * VA * k
        v_meas = np.einsum("ij,nj->ni", T, b_true)
        U = np.linalg.norm(v_meas, axis=1)
        alpha_new = np.degrees(np.arctan2(-v_meas[:, 2] * np.sign(w), -v_meas[:, 0]))
        alpha = np.where(ok & np.isfinite(alpha_new), np.clip(alpha_new, -15, 15), 0.0)
    return dict(t=t, P=P, pitch=pitch, w=w, U=U, alpha=alpha, ok=ok, k=k, v=v_meas)


if __name__ == "__main__":
    r = solve()
    ok = r["ok"]
    dive = ok & (r["w"] > 0)
    climb = ok & (r["w"] < 0)
    print(f"usable 1-Hz samples: {ok.sum():,}")
    for lab, m in (("dive", dive), ("climb", climb)):
        U, a, p, w = (x[m] for x in (r["U"], r["alpha"], r["pitch"], r["w"]))
        print(f"{lab:5s} n={m.sum():7,d}  U {np.nanmedian(U):.3f} m/s (p5 {np.nanpercentile(U,5):.3f} p95 {np.nanpercentile(U,95):.3f})"
              f"  alpha {np.nanmedian(a):+.2f} deg (p5 {np.nanpercentile(a,5):+.2f} p95 {np.nanpercentile(a,95):+.2f})"
              f"  |pitch| {np.median(np.abs(p)):.1f}  |w| {np.median(np.abs(w)):.3f} m/s")
    # consistency: does U sin(pitch+alpha) reproduce the measured w?
    gl = np.radians(np.abs(r["pitch"][ok]) + r["alpha"][ok])
    w_pred = r["U"][ok] * np.sin(gl)
    w_obs = np.abs(r["w"][ok])
    resid = w_pred - w_obs
    print(f"closure |w|: median residual {np.nanmedian(resid)*100:+.2f} cm/s, MAD {1.4826*np.nanmedian(np.abs(resid-np.nanmedian(resid)))*100:.2f} cm/s")
    # alpha vs pitch, the relation a flight model predicts
    print(f"\n{'|pitch| bin':>12s} {'n':>7s} {'alpha median':>13s}")
    ap = np.abs(r["pitch"][ok]); al = r["alpha"][ok]
    for lo, hi in ((15, 20), (20, 23), (23, 26), (26, 30), (30, 40)):
        m = (ap >= lo) & (ap < hi)
        if m.sum() > 200:
            print(f"{lo:5d}-{hi:<6d} {m.sum():7,d} {np.nanmedian(al[m]):13.2f}")

"""Undo the Aquadopp's own ENU rotation -> XYZ -> beam, and look for HR wrapping.

The file stores velocities in ENU (user config CoordSystem = 0): the instrument
rotated them with its own compass and tilt. Every record carries the heading,
pitch and roll it used, so the rotation is exactly invertible:

    ENU = H(heading) . P(pitch, roll) . T . beam          (Nortek convention)

so  beam = T^-1 . (H.P)^T . ENU, since H and P are rotations.

Phase wrapping is a per-BEAM (radial) effect, so it has to be examined and
undone here, before any rotation back.
"""
from __future__ import annotations

import numpy as np

from read_prf import config, read


def enu_to_xyz(enu, heading_deg, pitch_deg, roll_deg):
    """enu: (n, 3, ncell) -> xyz in the instrument frame."""
    hh = np.radians(heading_deg - 90.0)
    pp = np.radians(pitch_deg)
    rr = np.radians(roll_deg)
    ch, sh, cp, sp, cr, sr = np.cos(hh), np.sin(hh), np.cos(pp), np.sin(pp), np.cos(rr), np.sin(rr)
    n = enu.shape[0]
    H = np.zeros((n, 3, 3))
    H[:, 0, 0] = ch; H[:, 0, 1] = sh
    H[:, 1, 0] = -sh; H[:, 1, 1] = ch
    H[:, 2, 2] = 1.0
    P = np.zeros((n, 3, 3))
    P[:, 0, 0] = cp; P[:, 0, 1] = -sp * sr; P[:, 0, 2] = -cr * sp
    P[:, 1, 1] = cr; P[:, 1, 2] = -sr
    P[:, 2, 0] = sp; P[:, 2, 1] = sr * cp; P[:, 2, 2] = cp * cr
    R = H @ P
    return np.einsum("nji,njc->nic", R, enu)      # R^T . enu


def main():
    c = config()
    d = read()
    T = c["transform"]
    Tinv = np.linalg.inv(T)
    xyz = enu_to_xyz(d["vel"], d["heading_deg"], d["pitch_deg"], d["roll_deg"])
    beam = np.einsum("ij,njc->nic", Tinv, xyz)
    np.save("beam_vel.npy", beam.astype(np.float32))
    np.save("xyz_vel.npy", xyz.astype(np.float32))
    inwater = d["pressure_dbar"] > 5
    print(f"in-water records {inwater.sum():,} of {inwater.size:,}")
    for b in range(3):
        v = beam[inwater, b, :].ravel()
        v = v[np.isfinite(v)]
        print(f"beam{b+1}: p1 {np.percentile(v,1):+.3f}  p50 {np.median(v):+.3f}  p99 {np.percentile(v,99):+.3f}  "
              f"min {v.min():+.3f} max {v.max():+.3f}  std {v.std():.3f}")
        h, e = np.histogram(v, bins=120, range=(-3, 3))
        peaks = e[:-1][np.argsort(h)[-4:]] + (e[1] - e[0]) / 2
        print(f"        4 biggest histogram peaks at {np.sort(peaks).round(2)}")
    # the ENU "up" component vs the glider's own depth rate is the wrap tell
    up = d["vel"][:, 2, :]
    print("\nENU up-component (cells 1-5) median:", np.nanmedian(up[inwater][:, :5]).round(3), "m/s")
    dP = np.gradient(d["pressure_dbar"], d["time"])
    print("aqd dP/dt median while diving:", round(float(np.median(dP[inwater & (dP > 0.05)])), 3),
          "| climbing:", round(float(np.median(dP[inwater & (dP < -0.05)])), 3), "dbar/s")


if __name__ == "__main__":
    main()

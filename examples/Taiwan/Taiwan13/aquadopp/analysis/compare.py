#!/usr/bin/env python3
"""Aquadopp-measured speed and angle of attack vs the Merckelbach flight model, glider jane.

The Aquadopp values come from `../aqd/aoa.py`: beam velocities unwrapped with a
prior built only from the measured depth rate and pitch. The flight model values
come from the same recipe used for the two Taiwan MicroRiders (gliderflight
1.2.1, per-day Cd0/Vg calibrated against the depth rate). The two share no
information beyond pressure and pitch, and the model never sees a speed.

Clock: glider_time = aqd_time + 38.0 s (measured, drifting 37 -> 39 s over the
record; pressure vs pressure, sharpness 117).
"""
from __future__ import annotations

import numpy as np
import xarray as xr

from aoa import solve

LAG0, LAG_DRIFT_PER_DAY = 38.0, 0.4    # glider_time = aqd_time + lag


def main():
    a = solve()
    ok = a["ok"]
    t_a = a["t"][ok]
    lag = LAG0 + LAG_DRIFT_PER_DAY * (t_a - t_a.min()) / 86400.0
    t_g = t_a + lag                                   # onto the glider clock

    m = xr.open_dataset("jane_flight_hotel.nc", decode_times=False)
    tm = m["time"].values
    U_m = np.interp(t_g, tm, m["speed"].values)
    al_m = np.interp(t_g, tm, m["angle_of_attack"].values)
    excl = np.interp(t_g, tm, m["calibration_excluded"].values) > 0.5

    U_a, al_a, pit_a, w = a["U"][ok], np.abs(a["alpha"][ok]), a["pitch"][ok], a["w"][ok]
    use = np.isfinite(U_m) & np.isfinite(U_a) & ~excl & (np.abs(pit_a) > 15)
    dive, climb = use & (w > 0), use & (w < 0)
    print(f"matched samples: {use.sum():,} (dive {dive.sum():,}, climb {climb.sum():,})\n")
    print(f"{'':6s} {'U aqd':>7s} {'U model':>8s} {'ratio':>6s} | {'|a| aqd':>8s} {'|a| model':>10s} {'diff':>6s}")
    for lab, s in (("dive", dive), ("climb", climb)):
        ua, um = np.nanmedian(U_a[s]), np.nanmedian(U_m[s])
        aa, am = np.nanmedian(al_a[s]), np.nanmedian(np.abs(al_m[s]))
        print(f"{lab:6s} {ua:7.3f} {um:8.3f} {um/ua:6.3f} | {aa:8.2f} {am:10.2f} {am-aa:+6.2f}")
    print(f"\n{'|pitch|':>9s} {'n':>7s} {'|a| aqd':>8s} {'|a| model':>10s} {'U aqd':>7s} {'U model':>8s}")
    ap = np.abs(pit_a)
    for lo, hi in ((18, 22), (22, 25), (25, 28), (28, 32)):
        s = use & (ap >= lo) & (ap < hi)
        if s.sum() > 500:
            print(f"{lo:4d}-{hi:<4d} {s.sum():7,d} {np.nanmedian(al_a[s]):8.2f} {np.nanmedian(np.abs(al_m[s])):10.2f} "
                  f"{np.nanmedian(U_a[s]):7.3f} {np.nanmedian(U_m[s]):8.3f}")
    # what a mis-specified alpha does to epsilon: eps ~ U^-3.9
    for lab, s in (("dive", dive), ("climb", climb)):
        aa, am = np.nanmedian(al_a[s]), np.nanmedian(np.abs(al_m[s]))
        th = np.nanmedian(np.abs(pit_a[s]))
        r = np.sin(np.radians(th + aa)) / np.sin(np.radians(th + am))
        print(f"{lab}: sin(theta+a_aqd)/sin(theta+a_model) = {r:.3f}  ->  U_model high by {100*(1/r-1):+.1f}%, "
              f"epsilon low by {100*(1-r**3.9):+.1f}%")


if __name__ == "__main__":
    main()

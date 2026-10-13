#!/usr/bin/env -S uv run --script
# /// script
# requires-python = "==3.12.*"
# dependencies = [
#   "gliderflight==1.2.1",
#   "numpy==2.5.3",
#   "scipy==1.18.1",
#   "xarray==2026.7.0",
#   "netCDF4==1.7.4",
#   "gsw==3.6.23",
# ]
# ///
"""Through-water speed for MicroRider MR-SE SN 240 on SeaExplorer SEA025 (Kuroshio, June 2017).

The SeaExplorer of Tanaka et al. (2022, JTECH, doi:10.1175/JTECH-D-21-0074.1),
the first published lift/drag coefficients for this glider type. STEADY-STATE
model only (the paper's choice): gliderflight's dynamic model carries Slocum
added-mass coefficients that nobody has measured for a SeaExplorer.

FIXED, from the paper (Sect. 3.2): a = 4.0 rad^-1 (sum ah + aw; ah = 0 here),
Cd1 = 5.0 rad^-2, S = 0.14 m^2 (2 wing plates a side), m = 63.45 kg.
FITTED, against the navigation depth rate, whole deployment: Cd0, Vg and the
hull compressibility epsilon. The paper's MEA10 iteration gave Cd0 0.19,
Vg 62.0 L, c 5.3e-10 Pa^-1; this fit is the check that they reproduce.

Inputs: nav.nc (make_nav.py, 20 s, nav clock), mr1hz/*.npz (mr_clock.py
extract, 1 Hz JAC_T/JAC_C/P on the MR clock) for density, mr_clock_model.json.
The paper's quasi-steady criteria are applied as the calibration mask: depth
50-650 m, |pitch| > 15 deg, oil volume settled (|pos - cmd| < 10 ml), and
angular-actuator rate inside its 1-99 % range (the paper's C7).

    uv run make_flight_speed.py calibrate   # -> calibration.json
    uv run make_flight_speed.py run         # -> sea025_flight_hotel.nc (MR clock)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import gsw
import numpy as np
import xarray as xr
from gliderflight import SteadyStateCalibrate, SteadyStateGliderModel

HERE = Path(__file__).resolve().parent
CAL = HERE / "calibration.json"
OUT = HERE / "sea025_flight_hotel.nc"

MG = 63.45                                   # kg, Tanaka et al. (2022) Sect. 2.3.1
FIXED = dict(ah=0.0, aw=4.0, Cd1=5.0, S=0.14)  # Sect. 3.2; only ah + aw enters
DEPTH_MIN, DEPTH_MAX = 50.0, 650.0
PITCH_MIN = np.radians(15)
OIL_TOL = 10.0                               # [ml]


def lag_model(t):
    m = json.loads((HERE / "mr_clock_model.json").read_text())["blocks"]
    L = np.full(np.shape(t), np.nan)
    for b in m.values():
        lo = -np.inf if b["sel_lo_epoch"] is None else b["sel_lo_epoch"]
        hi = np.inf if b["sel_hi_epoch"] is None else b["sel_hi_epoch"]
        sel = (t >= lo) & (t < hi)
        L[sel] = b["lag_at_ref_s"] + b["drift_s_per_day"] * (t[sel] - b["t_ref_epoch"]) / 86400.0
    return L


def mr_ct():
    """1 Hz JAC T, C, P from mr1hz on the MR clock, all files concatenated."""
    ts, Ts, Cs, Ps, U = [], [], [], [], []
    for f in sorted((HERE / "mr1hz").glob("*.npz")):
        d = np.load(f)
        if "JAC_T" not in d:
            raise SystemExit(f"{f.name} has no JAC_T: re-run mr_clock.py extract with the rsijac converters")
        v = d["valid"]
        ts.append(d["t"][v]); Ts.append(d["JAC_T"][v]); Cs.append(d["JAC_C"][v]); Ps.append(d["P"][v]); U.append(d["U_EMC"][v])
    t = np.concatenate(ts); o = np.argsort(t)
    return t[o], np.concatenate(Ts)[o], np.concatenate(Cs)[o], np.concatenate(Ps)[o], np.concatenate(U)[o]


def load():
    nav = xr.open_dataset(HERE / "nav.nc", decode_times=False)
    t = nav["time"].values.astype(float)
    depth = nav["depth"].values
    pitch = np.radians(nav["pitch"].values)
    ok = np.isfinite(t) & np.isfinite(depth) & np.isfinite(pitch)
    d = {k: nav[k].values[ok] for k in ("depth", "pitch", "roll", "ballast_pos", "ballast_cmd", "ang_pos", "lat", "lon", "nav_state")}
    d["t"] = t[ok]; d["pitch"] = np.radians(d["pitch"])
    # density from the MR's JAC CT, put onto nav time via MR_time = nav_time - L
    tm, Tm, Cm, Pm, Um = mr_ct()
    t_on_mr = d["t"] - lag_model(d["t"])
    good = np.isfinite(Tm) & np.isfinite(Cm) & (Cm > 20)
    inside = (t_on_mr >= tm[good][0]) & (t_on_mr <= tm[good][-1])
    T = np.interp(t_on_mr, tm[good], Tm[good]); C = np.interp(t_on_mr, tm[good], Cm[good])
    T[~inside] = np.nan; C[~inside] = np.nan
    lat = np.where(np.isfinite(d["lat"]), d["lat"], np.nanmedian(d["lat"]))
    lon = np.where(np.isfinite(d["lon"]), d["lon"], np.nanmedian(d["lon"]))
    p = gsw.p_from_z(-d["depth"], lat)
    SP = gsw.SP_from_C(C, T, p)
    SA = gsw.SA_from_SP(SP, p, lon, lat)
    d["rho"] = gsw.rho(SA, gsw.CT_from_t(SA, T, p), p)
    d["P_bar"] = p / 10.0
    d["T"], d["C"], d["SP"] = T, C, SP
    d["U_EMC"] = np.interp(t_on_mr, tm, Um)
    d["U_EMC"][~inside] = np.nan
    keep = np.isfinite(d["rho"])
    return {k: v[keep] for k, v in d.items()}


def calibration_mask(d, dhdt):
    excl = (d["depth"] < DEPTH_MIN) | (d["depth"] > DEPTH_MAX) | (np.abs(d["pitch"]) < PITCH_MIN)
    excl |= np.abs(d["ballast_pos"] - d["ballast_cmd"]) > OIL_TOL
    excl |= np.r_[np.diff(d["t"]) > 45.0, True] | ~np.isfinite(dhdt) | (np.abs(dhdt) > 0.6)
    rate = np.r_[np.diff(d["ang_pos"]) / np.diff(d["t"]), 0.0]
    lo, hi = np.nanpercentile(rate, [1, 99])
    excl |= (rate < lo) | (rate > hi)
    return excl


def data_dict(d):
    return dict(time=d["t"], pressure=d["P_bar"], pitch=d["pitch"], buoyancy_change=d["ballast_pos"], density=d["rho"])


def calibrate():
    d = load()
    rho0 = float(np.median(d["rho"]))
    ss = SteadyStateCalibrate(rho0=rho0)
    ss.define(mg=MG, Vg=MG / rho0, Cd0=0.19, epsilon=5e-10, **FIXED)
    ss.set_input_data(**data_dict(d))
    excl = calibration_mask(d, ss.input_data["dhdt"])
    ss.set_mask(excl)
    r = ss.calibrate("Cd0", "Vg", "epsilon")
    cal = {"steady_global": {k: float(v) for k, v in r.items()} | {"n_samples": int((~excl).sum())},
           "settings": dict(mg_kg=MG, **FIXED, depth_window_m=[DEPTH_MIN, DEPTH_MAX], pitch_min_deg=15.0,
                            oil_tolerance_ml=OIL_TOL, rho0=rho0, source="Tanaka et al. 2022 Sect. 3.2 a, Cd1, S, m"),
           "paper": dict(Cd0=0.19, Vg_L=62.0, epsilon=5.3e-10)}
    CAL.write_text(json.dumps(cal, indent=2) + "\n")
    print(json.dumps(cal["steady_global"], indent=1))


def run():
    d = load()
    cal = json.loads(CAL.read_text())
    g = cal["steady_global"]
    m = SteadyStateGliderModel(rho0=cal["settings"]["rho0"])
    m.define(mg=MG, Vg=g["Vg"], Cd0=g["Cd0"], epsilon=g["epsilon"], **FIXED)
    r = m.solve(data_dict(d))
    U, alpha, w_mod = r.U, r.alpha, r.w
    dhdt = m.compute_dhdt(d["t"], d["P_bar"])
    excl = calibration_mask(d, dhdt) | ~np.isfinite(U)
    L = lag_model(d["t"])
    t_mr = d["t"] - L
    sel = np.flatnonzero(np.isfinite(t_mr) & np.isfinite(U))
    sel = sel[np.argsort(t_mr[sel])]

    def var(x, units, long_name, **attrs):
        return ("time", np.asarray(x)[sel].astype("float64"), dict(units=units, long_name=long_name, **attrs))

    ds = xr.Dataset(
        {
            "speed": var(U, "m s-1", "through-water speed, steady-state flight model with Tanaka et al. (2022) SeaExplorer coefficients",
                         comment="Use with perturb speed.method: hotel, speed.hotel_var: speed"),
            "angle_of_attack": var(np.degrees(alpha), "degree", "angle of attack (negative on dives)"),
            "pitch": var(np.degrees(d["pitch"]), "degree", "navigation pitch (negative nose down)"),
            "w_glider": var(w_mod, "m s-1", "modeled vertical velocity relative to the water (positive up)"),
            "dhdt": var(dhdt, "m s-1", "depth rate from navigation depth (positive up)"),
            "w_water": var(dhdt - w_mod, "m s-1", "implied vertical water velocity = dhdt - w_glider"),
            "ballast_pos": var(d["ballast_pos"], "cm3", "measured oil volume"),
            "calibration_excluded": var(excl.astype(float), "1", "1 = outside the quasi-steady calibration mask"),
            "U_EMC": var(d["U_EMC"], "m s-1", "MicroRider EM speed (1 Hz mean, UNCORRECTED), for comparison only"),
            "lat": var(d["lat"], "degrees_north", "navigation latitude"),
            "lon": var(d["lon"], "degrees_east", "navigation longitude"),
            "nav_time": var(d["t"], "seconds since 1970-01-01T00:00:00+00:00", "SeaExplorer navigation clock"),
            "mr_clock_lag": var(L, "s", "L in nav_time = MR_time + L (mr_clock_model.json)"),
        },
        coords={"time": ("time", t_mr[sel], dict(units="seconds since 1970-01-01T00:00:00+00:00", standard_name="time", axis="T",
                                                   long_name="time on the MicroRider SN 240 CLOCK (nav time minus mr_clock_lag)"))},
        attrs=dict(
            title="SeaExplorer SEA025 flight-model hotel file for MicroRider SN 240, Kuroshio 2017",
            Conventions="CF-1.13",
            comment=("Time axis is ALREADY on the MicroRider clock: set perturb hotel.time_offset: 0. Steady-state "
                     "gliderflight 1.2.1 model, a = 4.0, Cd1 = 5.0, S = 0.14 m2, m = 63.45 kg from Tanaka et al. (2022); "
                     "Cd0, Vg, epsilon fitted to the navigation depth rate. 20 s cadence."),
            references="Tanaka et al. 2022 doi:10.1175/JTECH-D-21-0074.1; gliderflight doi:10.5281/zenodo.2222694",
            flight_model=json.dumps(g),
        ),
    )
    enc = {v: {"zlib": True, "complevel": 4, "_FillValue": np.nan} for v in ds.data_vars}
    ds.to_netcdf(OUT, encoding=enc)
    use = ~excl
    for lab, mm in (("dive", use & (d["pitch"] < 0)), ("climb", use & (d["pitch"] > 0))):
        ratio = U[mm] / d["U_EMC"][mm]
        print(f"{lab}: n {mm.sum()}  U {np.nanmedian(U[mm]):.3f} m/s  alpha {np.degrees(np.nanmedian(alpha[mm])):+.2f} deg  "
              f"pitch {np.degrees(np.nanmedian(d['pitch'][mm])):+.1f}  U_EMC {np.nanmedian(d['U_EMC'][mm]):.3f}  "
              f"U/U_EMC median {np.nanmedian(ratio):.3f} (IQR {np.nanpercentile(ratio, 25):.3f}-{np.nanpercentile(ratio, 75):.3f})  "
              f"w_water {np.nanmedian((dhdt - w_mod)[mm]):+.4f}")
    print(f"wrote {OUT.name}: {sel.size} samples")


if __name__ == "__main__":
    {"calibrate": calibrate, "run": run}[sys.argv[1]]()

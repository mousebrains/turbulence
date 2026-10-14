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
"""Through-water speed for MicroRider MR1000-LP SN 058 on Slocum unit211 (20151113_unit211) -- flight model.

One of Tanaka's pre-paper deployments (no EM, no measured lift). Adapted from
../../20170724_unit095/analysis/make_flight_speed.py; the differences:

  * LIFT AND INDUCED DRAG ARE gliderflight's DEFAULT PARAMETERISATION
    (ah = 3.8, aw from AR = 7 and Omega = 45 deg, Cd1 from aw), as for every
    other fleet glider without a direct measurement (CASPER, Taiwan). Tanaka
    et al. (2022) showed these misfit unit095's measured pitch-AOA relation by
    up to 1.3 deg; whether they misfit unit211 is UNTESTED. The 2015-11-13
    mission carried a DVL (post-recovery/glider/*.pd0) that could measure it.
  * Mass 60 kg NOMINAL (unknown for unit211); only mg - rho*Vg is identifiable.
  * One- to two-day missions: Cd0 and Vg per UTC day with the 30-min pad.

    uv run make_flight_speed.py calibrate   # -> calibration.json   (resumable)
    uv run make_flight_speed.py run         # -> unit211_flight_hotel.nc
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import gsw
import numpy as np
import xarray as xr
from gliderflight import DynamicCalibrate, DynamicGliderModel, SteadyStateCalibrate

HERE = Path(__file__).resolve().parent
CAL = HERE / "calibration.json"
OUT = HERE / "unit211_flight_hotel.nc"

MG = 60.0            # kg, NOMINAL
K1, K2 = 0.20, 0.92  # added mass, Merckelbach et al. (2019)
FIXED = dict(ah=3.8, AR=7.0, Omega=45 * np.pi / 180)  # gliderflight Slocum defaults
DEPTH_MIN, DEPTH_MAX = 50.0, 950.0  # [m] calibration window
PITCH_MIN = np.radians(15)
PUMP_TOL = 10.0                     # [cc] |measured - commanded| oil volume while pumping
TURN_GUARD = 120.0                  # [s] around a commanded buoyancy reversal
PAD = 1800.0                        # [s] dynamic-model warm-up either side of a day
T_FIRST = np.datetime64("2015-11-13T00:00:00", "s").astype(float)
T_LAST = np.datetime64("2015-11-15T00:00:00", "s").astype(float)


def epoch(v):
    v = np.asarray(v)
    return v.astype("datetime64[ns]").astype("int64") / 1e9 if np.issubdtype(v.dtype, np.datetime64) else v.astype(float)


def load():
    ds = xr.open_dataset(HERE / "flight_inputs.nc")
    t = epoch(ds["m_present_time"].values)
    P = ds["m_pressure"].values  # bar
    pitch, roll = ds["m_pitch"].values, ds["m_roll"].values
    bp, cbp = ds["m_de_oil_vol"].values, ds["c_de_oil_vol"].values
    T, C, Pc = ds["sci_water_temp"].values, ds["sci_water_cond"].values * 10.0, ds["sci_water_pressure"].values * 10.0
    lat, lon = ds["m_lat"].values, ds["m_lon"].values
    ok = np.isfinite(t) & np.isfinite(P) & np.isfinite(pitch) & np.isfinite(bp) & (t >= T_FIRST - 2 * PAD) & (t < T_LAST + 2 * PAD)
    t, P, pitch, roll, bp, cbp, T, C, Pc, lat, lon = (x[ok] for x in (t, P, pitch, roll, bp, cbp, T, C, Pc, lat, lon))
    good = np.isfinite(T) & np.isfinite(C) & (C > 20.0) & np.isfinite(Pc)
    Ti, Ci = np.interp(t, t[good], T[good]), np.interp(t, t[good], C[good])
    gp = np.isfinite(lat) & np.isfinite(lon)
    lat, lon = np.interp(t, t[gp], lat[gp]), np.interp(t, t[gp], lon[gp])
    p_dbar = P * 10.0
    SP = gsw.SP_from_C(Ci, Ti, p_dbar)
    SA = gsw.SA_from_SP(SP, p_dbar, lon, lat)
    rho = gsw.rho(SA, gsw.CT_from_t(SA, Ti, p_dbar), p_dbar)
    return dict(t=t, P=P, pitch=pitch, roll=roll, bp=bp, cbp=cbp, T=T, C=C, Pc=Pc, rho=rho, SP=SP, lat=lat, lon=lon)


def calibration_mask(d, dhdt):
    """True = EXCLUDED from calibration (gliderflight's convention)."""
    t = d["t"]
    depth = d["P"] * 10.0
    excl = (depth < DEPTH_MIN) | (depth > DEPTH_MAX) | (np.abs(d["pitch"]) < PITCH_MIN)
    excl |= np.abs(d["bp"] - d["cbp"]) > PUMP_TOL
    excl |= np.r_[np.diff(t) > 20.0, True] | ~np.isfinite(dhdt) | (np.abs(dhdt) > 0.6)
    for i in np.flatnonzero(np.diff(np.sign(d["cbp"])) != 0):
        excl |= np.abs(t - t[i]) < TURN_GUARD
    return excl


def data_dict(d, sel):
    return dict(time=d["t"][sel], pressure=d["P"][sel], pitch=d["pitch"][sel],
                buoyancy_change=d["bp"][sel], density=d["rho"][sel])


def calibrate():
    d = load()
    rho0 = float(np.median(d["rho"]))
    cal = json.loads(CAL.read_text()) if CAL.exists() else {}
    inwin = (d["t"] >= T_FIRST) & (d["t"] < T_LAST)

    if "steady_global" not in cal:
        ss = SteadyStateCalibrate(rho0=rho0)
        ss.define(mg=MG, Vg=MG / rho0, Cd0=0.18, epsilon=5e-10, **FIXED)
        ss.set_input_data(**data_dict(d, slice(None)))
        excl = calibration_mask(d, ss.input_data["dhdt"]) | ~inwin
        ss.set_mask(excl)
        r = ss.calibrate("Cd0", "Vg", "epsilon")
        cal["steady_global"] = {k: float(v) for k, v in r.items()} | {"n_samples": int((~excl).sum())}
        CAL.write_text(json.dumps(cal, indent=2) + "\n")
        print("steady_global", cal["steady_global"], flush=True)
    eps = cal["steady_global"]["epsilon"]

    days = np.unique(np.floor(d["t"][inwin] / 86400).astype(int))
    cal.setdefault("dynamic_by_day", {})
    cal.setdefault("steady_by_day", {})
    for day in days:
        key = str(np.datetime64(int(day) * 86400, "s").astype("datetime64[D]"))
        t0, t1 = day * 86400.0, (day + 1) * 86400.0
        core = (d["t"] >= t0) & (d["t"] < t1)
        win = (d["t"] >= t0 - PAD) & (d["t"] < t1 + PAD)
        if key not in cal["dynamic_by_day"] and core.sum() < 3000:
            cal["steady_by_day"][key] = {"skipped": f"{int(core.sum())} samples in the day < 3000"}
            cal["dynamic_by_day"][key] = cal["steady_by_day"][key]
            CAL.write_text(json.dumps(cal, indent=2) + "\n")
            continue
        dd = data_dict(d, win)
        if key not in cal["steady_by_day"]:
            ss = SteadyStateCalibrate(rho0=rho0)
            ss.define(mg=MG, Vg=cal["steady_global"]["Vg"], Cd0=cal["steady_global"]["Cd0"], epsilon=eps, **FIXED)
            ss.set_input_data(**dd)
            excl = calibration_mask({k: v[win] for k, v in d.items()}, ss.input_data["dhdt"]) | ~core[win]
            if (~excl).sum() < 2000:
                cal["steady_by_day"][key] = {"skipped": f"{int((~excl).sum())} calibration samples < 2000"}
                cal["dynamic_by_day"][key] = cal["steady_by_day"][key]
                CAL.write_text(json.dumps(cal, indent=2) + "\n")
                continue
            ss.set_mask(excl)
            r = ss.calibrate("Cd0", "Vg")
            cal["steady_by_day"][key] = {k: float(v) for k, v in r.items()} | {"n_samples": int((~excl).sum())}
            CAL.write_text(json.dumps(cal, indent=2) + "\n")
        if key in cal["dynamic_by_day"]:
            continue
        dm = DynamicCalibrate(rho0=rho0, k1=K1, k2=K2, dt=1.0)
        seed = cal["steady_by_day"][key]
        dm.define(mg=MG, Vg=seed["Vg"], Cd0=seed["Cd0"], epsilon=eps, **FIXED)
        dm.set_input_data(**dd)
        excl = calibration_mask({k: v[win] for k, v in d.items()}, dm.input_data["dhdt"]) | ~core[win]
        dm.set_mask(excl)
        r = dm.calibrate("Cd0", "Vg")
        cal["dynamic_by_day"][key] = {k: float(v) for k, v in r.items()} | {"n_samples": int((~excl).sum())}
        CAL.write_text(json.dumps(cal, indent=2) + "\n")
        print(key, cal["dynamic_by_day"][key], flush=True)
    cal["settings"] = dict(mg_kg=MG, k1=K1, k2=K2, **{k: float(v) for k, v in FIXED.items()}, depth_window_m=[DEPTH_MIN, DEPTH_MAX],
                           pitch_min_deg=float(np.degrees(PITCH_MIN)), pump_tolerance_cc=PUMP_TOL, turn_guard_s=TURN_GUARD,
                           dynamic_warmup_pad_s=PAD, rho0=rho0, source="gliderflight default lift/Cd1 parameterisation")
    CAL.write_text(json.dumps(cal, indent=2) + "\n")


def lag_model(t):
    m = json.loads((HERE / "mr_clock_model.json").read_text())["blocks"]
    L = np.full(t.shape, np.nan)
    for b in m.values():
        lo = -np.inf if b["sel_lo_epoch"] is None else b["sel_lo_epoch"]
        hi = np.inf if b["sel_hi_epoch"] is None else b["sel_hi_epoch"]
        sel = (t >= lo) & (t < hi)
        L[sel] = b["lag_at_ref_s"] + b["drift_s_per_day"] * (t[sel] - b["t_ref_epoch"]) / 86400.0
    return L


def run():
    d = load()
    cal = json.loads(CAL.read_text())
    eps = cal["steady_global"]["epsilon"]
    rho0 = cal["settings"]["rho0"]
    good = {k: v for k, v in cal["dynamic_by_day"].items() if "Cd0" in v}
    centers = np.array([np.datetime64(k, "s").astype(float) + 43200.0 for k in good])
    Cd0_t = np.interp(d["t"], centers, np.array([v["Cd0"] for v in good.values()]))
    Vg_t = np.interp(d["t"], centers, np.array([v["Vg"] for v in good.values()]))

    n = d["t"].size
    U, alpha, w_mod, dhdt = (np.full(n, np.nan) for _ in range(4))
    blocks = np.split(np.arange(n), np.flatnonzero(np.diff(d["t"]) > 6 * 3600) + 1)
    for idx in blocks:
        if idx.size < 600:
            print(f"skipping a {idx.size}-sample block at {np.datetime64(int(d['t'][idx[0]]), 's')}")
            continue
        dm = DynamicGliderModel(rho0=rho0, k1=K1, k2=K2, dt=1.0)
        dm.define(mg=MG, Vg=Vg_t[idx], Cd0=Cd0_t[idx], epsilon=eps, **FIXED)
        r = dm.solve(data_dict(d, idx))
        U[idx], alpha[idx], w_mod[idx] = r.U, r.alpha, r.w
        dhdt[idx] = dm.compute_dhdt(d["t"][idx], d["P"][idx])
    excl = calibration_mask(d, dhdt) | ~np.isfinite(U)

    t_mr = d["t"] - lag_model(d["t"])
    keep = np.isfinite(t_mr) & np.isfinite(U)
    order = np.argsort(t_mr[keep])
    sel = np.flatnonzero(keep)[order]
    if np.any(np.diff(t_mr[sel]) <= 0):
        raise SystemExit("MR-clock time is not strictly increasing; check mr_clock_model.json")

    def var(x, units, long_name, **attrs):
        return ("time", np.asarray(x)[sel].astype("float64"), dict(units=units, long_name=long_name, **attrs))

    ds = xr.Dataset(
        {
            "speed": var(U, "m s-1", "through-water speed, Merckelbach dynamic flight model with gliderflight default lift and Cd1",
                         comment="Use with perturb speed.method: hotel, speed.hotel_var: speed"),
            "angle_of_attack": var(np.degrees(alpha), "degree", "angle of attack (Slocum sign: negative on dives)"),
            "m_pitch": var(np.degrees(d["pitch"]), "degree", "glider pitch (m_pitch; negative nose down)"),
            "m_roll": var(np.degrees(d["roll"]), "degree", "glider roll (m_roll); NOT used by the model"),
            "w_glider": var(w_mod, "m s-1", "modeled glider vertical velocity relative to the water (positive up)"),
            "dhdt": var(dhdt, "m s-1", "depth rate from m_pressure (positive up)"),
            "w_water": var(dhdt - w_mod, "m s-1", "implied vertical water velocity = dhdt - w_glider"),
            "m_de_oil_vol": var(d["bp"], "cm3", "measured buoyancy-engine oil volume"),
            "Cd0": var(Cd0_t, "1", "parasite drag coefficient (daily dynamic calibration, linear between day centers)"),
            "Vg": var(Vg_t, "m3", f"glider volume at the nominal mass {MG} kg (daily dynamic calibration)"),
            "calibration_excluded": var(excl.astype(float), "1", "1 = outside the calibration mask (surface, turn, pumping, gap)"),
            "sci_water_temp": var(d["T"], "degree_Celsius", "glider CTD temperature (Sea-Bird, sci_ctd41cp_timestamp)"),
            "sci_water_cond": var(d["C"], "mS/cm", "glider CTD conductivity (S/m x 10, applied once, here)"),
            "sci_water_pressure": var(d["Pc"], "dbar", "glider CTD pressure (bar x 10, applied once, here)"),
            "lat": var(d["lat"], "degrees_north", "dead-reckoned latitude (m_lat, interpolated)"),
            "lon": var(d["lon"], "degrees_east", "dead-reckoned longitude (m_lon, interpolated)"),
            "glider_time": var(d["t"], "seconds since 1970-01-01T00:00:00+00:00", "unit211 flight-computer time (m_present_time)"),
            "mr_clock_lag": var(lag_model(d["t"]), "s", "L in glider_time = MR_time + L (mr_clock_model.json)"),
        },
        coords={"time": ("time", t_mr[sel], dict(units="seconds since 1970-01-01T00:00:00+00:00", calendar="standard",
                                                   standard_name="time", axis="T",
                                                   long_name="time on the MicroRider SN 058 CLOCK (glider time minus mr_clock_lag)"))},
        attrs=dict(
            title="Glider unit211 flight-model hotel file for MicroRider SN 058, 20151113_unit211",
            Conventions="CF-1.13",
            comment=("Time axis is ALREADY on the MicroRider clock: set perturb hotel.time_offset: 0. "
                     "Speed from gliderflight 1.2.1 DynamicGliderModel (Merckelbach et al. 2019), default lift and Cd1, "
                     "Cd0 and Vg calibrated per UTC day against the depth rate; see README.md."),
            references=("Tanaka et al. 2022 doi:10.1175/JTECH-D-21-0074.1; Merckelbach, Smeed & Griffiths 2010 "
                        "doi:10.1175/2009JTECHO710.1; Merckelbach et al. 2019 doi:10.1175/JTECH-D-18-0168.1; "
                        "gliderflight doi:10.5281/zenodo.2222694"),
            flight_model_settings=json.dumps(cal["settings"]),
            flight_model_epsilon=eps,
        ),
    )
    enc = {v: {"zlib": True, "complevel": 4, "_FillValue": np.nan} for v in ds.data_vars}
    ds.to_netcdf(OUT, encoding=enc)
    use = ~excl
    for lab, m in (("dive", use & (d["pitch"] < 0)), ("climb", use & (d["pitch"] > 0))):
        print(f"{lab}: U median {np.nanmedian(U[m]):.3f} m/s, alpha {np.degrees(np.nanmedian(alpha[m])):+.2f} deg, "
              f"pitch {np.degrees(np.nanmedian(d['pitch'][m])):+.1f} deg, w_water median {np.nanmedian((dhdt - w_mod)[m]):+.4f} m/s")
    print(f"wrote {OUT.name}: {sel.size} samples")


if __name__ == "__main__":
    {"calibrate": calibrate, "run": run}[sys.argv[1]]()

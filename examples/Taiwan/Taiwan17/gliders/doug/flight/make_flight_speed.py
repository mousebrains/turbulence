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
"""Through-water speed for MicroRider SN 134 on glider doug (Taiwan 2017) -- Merckelbach flight model.

Copied from Taiwan13/gliders/20130526_husker/flight/make_flight_speed.py (read its
README.md). Differences: output name, position (fixed, below), no drift-corrected
track. Everything else -- model, calibration, masks, gotchas -- is identical.

    uv run make_flight_speed.py calibrate   # -> calibration.json   (resumable, ~30 min)
    uv run make_flight_speed.py run         # -> doug_flight_hotel.nc (perturb hotel file)

Neither Taiwan MicroRider carried an EM flowmeter, so speed comes from a glider
flight model. This uses Lucas Merckelbach's own implementation, `gliderflight`
1.2.1 (MIT; doi:10.5281/zenodo.2222694), of

  * Merckelbach, Smeed & Griffiths (2010), JTECH 27, 547-563 -- steady state;
  * Merckelbach, Berger, Krahmann, Dengler & Carpenter (2019), JTECH 36,
    281-296 -- the dynamic model, calibrated in situ for microstructure.

See README.md for the choices, the evidence behind each, and the limitations.

INPUTS  flight_inputs.nc (dinkum-hotel, flight clock), mr_clock_model.json
        (mr_clock.py). Position for TEOS-10 is FIXED at the median of the 661
        GPS fixes in ../NH_201702160638_doug.mat (21.462 N, 119.420 E; fixes span
        21.27-22.12 N, 119.18-120.15 E) -- at most ~1e-3 kg m-3 in density.

GOTCHAS IN gliderflight 1.2.1, handled here
  * `pressure` is in BAR (the code multiplies by 1e5), although set_input_data's
    docstring says Pa. `buoyancy_change` is in cc.
  * DynamicCalibrate defaults to k1 = 0.02, but the paper (Merckelbach et al.
    2019, below eq. 11) gives m11 = 0.2 mg, m22 = 0.92 mg. k1 = 0.20 is set
    explicitly; DynamicGliderModel's own default is already 0.20.
  * A callable Cd0/Vg/mg reaches an undefined name (`ti`) in integrate(); time
    series are passed as ARRAYS instead, which broadcast correctly.
  * The dynamic model parallelises with a process pool, so everything runs
    under `if __name__ == "__main__"` (macOS spawn).
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
OUT = HERE / "doug_flight_hotel.nc"
LAT, LON = 21.462, 119.420  # median GPS fix, see INPUTS

MG = 52.0          # kg, NOMINAL. Only mg - rho*Vg is identifiable; Vg absorbs it.
K1, K2 = 0.20, 0.92  # added mass, Merckelbach et al. (2019)
FIXED = dict(ah=3.8, AR=7.0, Omega=45 * np.pi / 180)  # gliderflight Slocum defaults
DEPTH_MIN, DEPTH_MAX = 10.0, 175.0  # [m] calibration window (turns at ~5 and ~190 m)
PITCH_MIN = np.radians(15)          # below this the glider is turning, not gliding
PUMP_TOL = 10.0                     # [cc] |measured - commanded| ballast while pumping
TURN_GUARD = 90.0                   # [s] around a commanded ballast reversal
PAD = 1800.0                        # [s] dynamic-model warm-up either side of a day


def epoch(v):
    v = np.asarray(v)
    return v.astype("datetime64[ns]").astype("int64") / 1e9 if np.issubdtype(v.dtype, np.datetime64) else v.astype(float)


def load():
    ds = xr.open_dataset(HERE / "flight_inputs.nc")
    t = epoch(ds["m_present_time"].values)
    P = ds["m_pressure"].values  # bar
    pitch, roll = ds["m_pitch"].values, ds["m_roll"].values
    bp, cbp = ds["m_ballast_pumped"].values, ds["c_ballast_pumped"].values
    T, C, Pc = ds["sci_water_temp"].values, ds["sci_water_cond"].values * 10.0, ds["sci_water_pressure"].values * 10.0
    ok = np.isfinite(t) & np.isfinite(P) & np.isfinite(pitch) & np.isfinite(bp)
    t, P, pitch, roll, bp, cbp, T, C, Pc = (x[ok] for x in (t, P, pitch, roll, bp, cbp, T, C, Pc))
    # density from the glider CTD; gaps filled in time (CTD finite in ~95% of samples)
    good = np.isfinite(T) & np.isfinite(C) & (C > 10.0) & np.isfinite(Pc)
    Ti, Ci = np.interp(t, t[good], T[good]), np.interp(t, t[good], C[good])
    lat = np.full(t.shape, LAT)
    lon = np.full(t.shape, LON)
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

    # 1. whole deployment, steady state: Cd0, Vg AND compressibility epsilon.
    if "steady_global" not in cal:
        ss = SteadyStateCalibrate(rho0=rho0)
        ss.define(mg=MG, Vg=MG / rho0, Cd0=0.15, epsilon=5e-10, **FIXED)
        ss.set_input_data(**data_dict(d, slice(None)))
        excl = calibration_mask(d, ss.input_data["dhdt"])
        ss.set_mask(excl)
        r = ss.calibrate("Cd0", "Vg", "epsilon")
        cal["steady_global"] = {k: float(v) for k, v in r.items()} | {"n_samples": int((~excl).sum())}
        CAL.write_text(json.dumps(cal, indent=2) + "\n")
    eps = cal["steady_global"]["epsilon"]

    # 2. per UTC day, dynamic model: Cd0 and Vg, epsilon held at the global value.
    days = np.unique(np.floor(d["t"] / 86400).astype(int))
    cal.setdefault("dynamic_by_day", {})
    cal.setdefault("steady_by_day", {})
    for day in days:
        key = str(np.datetime64(int(day) * 86400, "s").astype("datetime64[D]"))
        t0, t1 = day * 86400.0, (day + 1) * 86400.0
        core = (d["t"] >= t0) & (d["t"] < t1)
        win = (d["t"] >= t0 - PAD) & (d["t"] < t1 + PAD)
        if key not in cal["dynamic_by_day"] and core.sum() < 3000:
            # too little to calibrate (and np.gradient needs >= 2 samples at all)
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
            if (~excl).sum() < 3000:
                cal["steady_by_day"][key] = {"skipped": f"{int((~excl).sum())} calibration samples < 3000"}
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
                           dynamic_warmup_pad_s=PAD, rho0=rho0)
    CAL.write_text(json.dumps(cal, indent=2) + "\n")


def lag_model(t):
    m = json.loads((HERE / "mr_clock_model.json").read_text())["blocks"]
    L = np.full(t.shape, np.nan)
    for b in m.values():
        sel = (t < b["split_epoch"]) if b["t_max_epoch"] < b["split_epoch"] else (t >= b["split_epoch"])
        L[sel] = b["lag_at_ref_s"] + b["drift_s_per_day"] * (t[sel] - b["t_ref_epoch"]) / 86400.0
    return L


def run():
    d = load()
    cal = json.loads(CAL.read_text())
    eps = cal["steady_global"]["epsilon"]
    rho0 = cal["settings"]["rho0"]
    good = {k: v for k, v in cal["dynamic_by_day"].items() if "Cd0" in v}
    centers = np.array([np.datetime64(k, "s").astype(float) + 43200.0 for k in good])
    Cd0_day = np.array([v["Cd0"] for v in good.values()])
    Vg_day = np.array([v["Vg"] for v in good.values()])
    # Parameters vary linearly between day centers and are held beyond the ends.
    Cd0_t = np.interp(d["t"], centers, Cd0_day)
    Vg_t = np.interp(d["t"], centers, Vg_day)

    n = d["t"].size
    U, alpha, w_mod, dhdt = (np.full(n, np.nan) for _ in range(4))
    blocks = np.split(np.arange(n), np.flatnonzero(np.diff(d["t"]) > 6 * 3600) + 1)  # the 2-week gap
    for idx in blocks:
        if idx.size < 600:  # e.g. a stray sample days before the record: nothing to integrate, left NaN
            print(f"skipping a {idx.size}-sample block at {np.datetime64(int(d['t'][idx[0]]), 's')}")
            continue
        dm = DynamicGliderModel(rho0=rho0, k1=K1, k2=K2, dt=1.0)
        dm.define(mg=MG, Vg=Vg_t[idx], Cd0=Cd0_t[idx], epsilon=eps, **FIXED)
        r = dm.solve(data_dict(d, idx))
        U[idx], alpha[idx], w_mod[idx] = r.U, r.alpha, r.w
        dhdt[idx] = dm.compute_dhdt(d["t"][idx], d["P"][idx])
    excl = calibration_mask(d, dhdt) | ~np.isfinite(U)

    t_mr = d["t"] - lag_model(d["t"])  # glider_time = MR_time + L  ->  MR_time = glider_time - L
    keep = np.isfinite(t_mr) & np.isfinite(U)
    order = np.argsort(t_mr[keep])
    sel = np.flatnonzero(keep)[order]
    if np.any(np.diff(t_mr[sel]) <= 0):
        raise SystemExit("MR-clock time is not strictly increasing; check mr_clock_model.json")

    def var(x, units, long_name, **attrs):
        return ("time", np.asarray(x)[sel].astype("float64"), dict(units=units, long_name=long_name, **attrs))

    ds = xr.Dataset(
        {
            "speed": var(U, "m s-1", "through-water speed, Merckelbach dynamic flight model",
                         comment="Use with perturb speed.method: hotel, speed.hotel_var: speed"),
            "angle_of_attack": var(np.degrees(alpha), "degree", "angle of attack (Slocum sign: negative on dives)"),
            "m_pitch": var(np.degrees(d["pitch"]), "degree", "glider pitch (m_pitch; negative nose down)"),
            "m_roll": var(np.degrees(d["roll"]), "degree", "glider roll (m_roll); NOT used by the model"),
            "w_glider": var(w_mod, "m s-1", "modeled glider vertical velocity relative to the water (positive up)"),
            "dhdt": var(dhdt, "m s-1", "depth rate from m_pressure (positive up)"),
            "w_water": var(dhdt - w_mod, "m s-1", "implied vertical water velocity = dhdt - w_glider"),
            "m_ballast_pumped": var(d["bp"], "cm3", "measured buoyancy change"),
            "Cd0": var(Cd0_t, "1", "parasite drag coefficient in force (daily dynamic calibration, linear between day centers)"),
            "Vg": var(Vg_t, "m3", f"glider volume at the nominal mass {MG} kg (daily dynamic calibration)"),
            "calibration_excluded": var(excl.astype(float), "1", "1 = outside the calibration mask (surface, turn, pumping, gap)"),
            "sci_water_temp": var(d["T"], "degree_Celsius", "glider CTD temperature (projected onto the flight clock)"),
            "sci_water_cond": var(d["C"], "mS/cm", "glider CTD conductivity (S/m x 10, applied once, here)"),
            "sci_water_pressure": var(d["Pc"], "dbar", "glider CTD pressure (bar x 10, applied once, here)"),
            "glider_time": var(d["t"], "seconds since 1970-01-01T00:00:00+00:00", "doug flight-computer time (m_present_time)"),
            "mr_clock_lag": var(lag_model(d["t"]), "s", "L in glider_time = MR_time + L (mr_clock_model.json)"),
        },
        coords={"time": ("time", t_mr[sel], dict(units="seconds since 1970-01-01T00:00:00+00:00", calendar="standard",
                                                   standard_name="time", axis="T",
                                                   long_name="time on the MicroRider SN 134 CLOCK (glider time minus mr_clock_lag)"))},
        attrs=dict(
            title="Glider doug flight-model hotel file for MicroRider SN 134, Taiwan 2017",
            Conventions="CF-1.13",
            comment=("Time axis is ALREADY on the MicroRider clock: set perturb hotel.time_offset: 0. "
                     "Speed from gliderflight 1.2.1 DynamicGliderModel (Merckelbach et al. 2019) with Cd0 and Vg "
                     "calibrated per UTC day against the depth rate; see README.md."),
            references=("Merckelbach, Smeed & Griffiths 2010 doi:10.1175/2009JTECHO710.1; "
                        "Merckelbach et al. 2019 doi:10.1175/JTECH-D-18-0168.1; gliderflight doi:10.5281/zenodo.2222694"),
            flight_model_settings=json.dumps(cal["settings"]),
            flight_model_epsilon=eps,
        ),
    )
    enc = {v: {"zlib": True, "complevel": 4, "_FillValue": np.nan} for v in ds.data_vars}
    ds.to_netcdf(OUT, encoding=enc)
    use = ~excl
    for lab, m in (("dive", use & (d["pitch"] < 0)), ("climb", use & (d["pitch"] > 0))):
        print(f"{lab}: U median {np.nanmedian(U[m]):.3f} m/s, alpha {np.degrees(np.nanmedian(alpha[m])):+.2f} deg, "
              f"w_water median {np.nanmedian((dhdt - w_mod)[m]):+.4f} m/s")
    print(f"wrote {OUT.name}: {sel.size} samples")


if __name__ == "__main__":
    {"calibrate": calibrate, "run": run}[sys.argv[1]]()

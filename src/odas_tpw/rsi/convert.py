# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""
NetCDF conversion for Rockland .p files -- ATOMIX L1_converted format.

Follows the ATOMIX shear-probe benchmark structure (Lueck et al., 2024,
doi:10.3389/fmars.2024.1334327, Figure 4).  The L1_converted group contains
time series converted to physical units, including speed-normalized shear
and temperature gradient.
"""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from odas_tpw.rsi.p_file import PFile

import numpy as np

# Minimum profiling speed to avoid singularity [m/s]
_SPEED_MIN = 0.05

# Butterworth smoothing time constant for speed computation [s]
_SPEED_TAU = 1.5


def _compute_speed(pf: "PFile") -> tuple[np.ndarray, np.ndarray]:
    """Compute profiling speed from pressure.

    Matches the ODAS odas_p2mat.m speed pipeline:
      1. W = gradient(P) filtered with Butterworth at 0.68/tau
      2. speed = abs(W), interpolated to fast rate
      3. Second Butterworth smoothing pass
      4. Clamped to _SPEED_MIN

    Returns (speed_fast, speed_slow) in m/s.
    """
    from scipy.signal import butter, filtfilt

    from odas_tpw.rsi.profile import _smooth_fall_rate

    P_slow = pf.channels.get("P_dP", pf.channels.get("P"))
    if P_slow is None:
        raise ValueError("No pressure channel (P or P_dP) found")

    W_slow = _smooth_fall_rate(P_slow, pf.fs_slow, tau=_SPEED_TAU)
    speed_slow = np.abs(W_slow)

    speed_fast = np.interp(pf.t_fast, pf.t_slow, speed_slow)

    f_c = 0.68 / _SPEED_TAU
    b_s, a_s = butter(1, f_c / (pf.fs_slow / 2.0))
    speed_slow = filtfilt(b_s, a_s, speed_slow)
    b_f, a_f = butter(1, f_c / (pf.fs_fast / 2.0))
    speed_fast = filtfilt(b_f, a_f, speed_fast)

    speed_slow = np.maximum(speed_slow, _SPEED_MIN)
    speed_fast = np.maximum(speed_fast, _SPEED_MIN)

    return speed_fast, speed_slow


def p_to_L1(
    p_filepath: str | Path,
    nc_filepath: str | Path | None = None,
) -> "tuple[PFile, Path]":
    """Convert a .p file to ATOMIX L1_converted NetCDF format.

    Produces a NetCDF4 file with root-level dimensions and an L1_converted
    group containing time series in physical units.  ATOMIX core variables
    (SHEAR, VIB, ACC, GRADT, TEMP, TEMP_CTD, COND_CTD, PRES, PITCH, ROLL,
    MAG) are mapped from PFile channels.  Biogeochemical sensors (CHLA,
    TURB, DOXY, DOXY_TEMP) get CF-compliant names.  Remaining channels
    are included as supplementary variables.

    Parameters
    ----------
    p_filepath : str or Path
        Path to the .p file.
    nc_filepath : str or Path, optional
        Output path. Defaults to same name with .nc extension.

    Returns
    -------
    tuple of (PFile, Path)
        The parsed data object and the output path.
    """
    import netCDF4 as nc

    from odas_tpw.rsi.p_file import PFile

    pf = PFile(p_filepath)
    if nc_filepath is None:
        nc_filepath = pf.filepath.with_suffix(".nc")
    nc_filepath = Path(nc_filepath)

    # ---- Profiling speed from pressure ----
    speed_fast, _speed_slow = _compute_speed(pf)

    # ---- Time vectors as decimal days since Jan 1 of start year ----
    ref_year = pf.start_time.year
    epoch = datetime(ref_year, 1, 1, tzinfo=UTC)
    st = pf.start_time
    if st.tzinfo is None:
        st = st.replace(tzinfo=UTC)
    t0_days = (st - epoch).total_seconds() / 86400.0
    time_fast = t0_days + pf.t_fast / 86400.0
    time_slow = t0_days + pf.t_slow / 86400.0
    time_units = f"Days since {ref_year}-01-01T00:00:00Z"

    # ---- Classify channels by ATOMIX role ----
    shear_names = sorted(
        n for n in pf.channels if pf.channel_info[n]["type"] == "shear"
    )
    vib_names = sorted(
        n for n in pf.channels if pf.channel_info[n]["type"] == "piezo"
    )
    acc_names = sorted(
        n for n in pf.channels if pf.channel_info[n]["type"] == "accel"
    )
    mag_names = sorted(
        n for n in pf.channels if pf.channel_info[n]["type"] == "mag"
    )
    # Fast thermistors -> TEMP (fast-rate) and GRADT
    gradt_names = sorted(
        n for n in pf.channels
        if pf.channel_info[n]["type"] == "therm" and pf.is_fast(n)
    )
    # CTD conductivity (slow JAC_C) -> COND_CTD
    cond_ctd_names = sorted(
        n for n in pf.channels if pf.channel_info[n]["type"] == "jac_c"
    )

    # Pressure (prefer deconvolved P_dP, else P)
    P_slow = pf.channels.get("P_dP", pf.channels.get("P"))
    P_fast = np.interp(pf.t_fast, pf.t_slow, P_slow)

    # CTD temperature (slow JAC_T) -> TEMP_CTD
    temp_ctd_name = None
    for n in pf.channels:
        if pf.channel_info[n]["type"] == "jac_t":
            temp_ctd_name = n
            break

    # Pitch/Roll from inclinometer
    pitch_name = roll_name = None
    for n in pf.channels:
        if pf.channel_info[n]["type"] == "inclxy":
            if "X" in n:
                pitch_name = n
            elif "Y" in n:
                roll_name = n

    # Named biogeochemical/auxiliary sensors
    chla_name = next(
        (n for n in pf.channels if n.lower().startswith("chloro")), None
    )
    turb_name = next(
        (n for n in pf.channels if n.lower().startswith("turb")), None
    )
    doxy_name = next(
        (n for n in pf.channels if pf.channel_info[n]["type"] == "aroft_o2"),
        None,
    )
    doxy_temp_name = next(
        (n for n in pf.channels if pf.channel_info[n]["type"] == "aroft_t"),
        None,
    )

    # Identify supplementary channels (not mapped to named variables)
    mapped = set(
        shear_names + vib_names + gradt_names + acc_names + mag_names
        + cond_ctd_names
    )
    for n in (
        temp_ctd_name, pitch_name, roll_name, "P", "P_dP",
        chla_name, turb_name, doxy_name, doxy_temp_name,
    ):
        if n is not None:
            mapped.add(n)
    # Exclude slow thermistors (base channels for deconvolved fast ones)
    for n in list(pf.channels):
        if pf.channel_info[n]["type"] == "therm" and not pf.is_fast(n):
            mapped.add(n)
    supplementary = sorted(n for n in pf.channels if n not in mapped)

    # ---- Create NetCDF file ----
    ds = nc.Dataset(str(nc_filepath), "w", format="NETCDF4")

    # Global attributes
    ds.Conventions = "CF-1.13"
    ds.title = f"VMP data from {pf.filepath.name}"
    ds.source = pf.config["instrument_info"].get("model", "")
    ds.platform_type = pf.config["instrument_info"].get("vehicle", "")
    ds.instrument_sn = pf.config["instrument_info"].get("sn", "")
    ds.operator = pf.config["cruise_info"].get("operator", "")
    ds.project = pf.config["cruise_info"].get("project", "")
    ds.start_time = pf.start_time.isoformat()
    ds.source_file = pf.filepath.name
    ds.history = (
        f"L1 converted from {pf.filepath.name} on "
        f"{datetime.now(UTC).isoformat()}"
    )
    ds.configuration_string = pf.config_str

    # Root dimensions (shared across groups, following ATOMIX convention)
    ds.createDimension("TIME", len(pf.t_fast))
    ds.createDimension("TIME_SLOW", len(pf.t_slow))
    if shear_names:
        ds.createDimension("N_SHEAR_SENSORS", len(shear_names))
    if vib_names:
        ds.createDimension("N_VIB_SENSORS", len(vib_names))
    if acc_names:
        ds.createDimension("N_ACC_SENSORS", len(acc_names))
    if gradt_names:
        ds.createDimension("N_GRADT_SENSORS", len(gradt_names))
        ds.createDimension("N_TEMP_SENSORS", len(gradt_names))
    if mag_names:
        ds.createDimension("N_MAG_SENSORS", len(mag_names))

    # ---- L1_converted group ----
    L1 = ds.createGroup("L1_converted")

    # TIME (fast)
    v = L1.createVariable("TIME", "f8", ("TIME",), zlib=True)
    v[:] = time_fast
    v.standard_name = "time"
    v.long_name = "Decimal day"
    v.units = time_units
    v.axis = "T"

    # TIME_SLOW
    v = L1.createVariable("TIME_SLOW", "f8", ("TIME_SLOW",), zlib=True)
    v[:] = time_slow
    v.standard_name = "time"
    v.long_name = "Decimal day"
    v.units = time_units
    v.axis = "T"

    # PRES (fast, interpolated from slow)
    v = L1.createVariable("PRES", "f8", ("TIME",), zlib=True)
    v[:] = P_fast
    v.standard_name = "sea_water_pressure"
    v.units = "decibar"
    v.long_name = "Sea water pressure, equals 0 at sea-level"

    # PRES_SLOW
    v = L1.createVariable("PRES_SLOW", "f8", ("TIME_SLOW",), zlib=True)
    v[:] = P_slow
    v.standard_name = "sea_water_pressure"
    v.units = "decibar"
    v.long_name = "Sea water pressure, equals 0 at sea-level"

    # SHEAR — velocity shear du/dz, normalized by speed^2
    # Our convert_shear gives E_dt / (2*sqrt(2)*G_D*S) which equals
    # W^2 * du/dz; dividing by speed^2 yields du/dz in s^-1.
    if shear_names:
        v = L1.createVariable(
            "SHEAR", "f8", ("N_SHEAR_SENSORS", "TIME"), zlib=True
        )
        for i, name in enumerate(shear_names):
            v[i, :] = pf.channels[name] / speed_fast**2
        v.standard_name = "sea_water_velocity_shear"
        v.units = "s-1"
        v.long_name = (
            "rate of change of cross axis sea water velocity along "
            "transect measured by shear probes"
        )
        v.sensor_names = ", ".join(shear_names)

    # VIB — piezo-accelerometer vibration
    if vib_names:
        v = L1.createVariable(
            "VIB", "f8", ("N_VIB_SENSORS", "TIME"), zlib=True
        )
        for i, name in enumerate(vib_names):
            v[i, :] = pf.channels[name]
        v.standard_name = "platform_vibration"
        v.units = "1"
        v.long_name = "platform vibration detected by piezo-accelerometers"
        v.sensor_names = ", ".join(vib_names)

    # ACC — linear accelerometer
    if acc_names:
        v = L1.createVariable(
            "ACC", "f8", ("N_ACC_SENSORS", "TIME"), zlib=True
        )
        for i, name in enumerate(acc_names):
            v[i, :] = pf.channels[name]
        v.standard_name = "platform_acceleration"
        v.units = "m s-2"
        v.long_name = "platform acceleration detected by accelerometers"
        v.sensor_names = ", ".join(acc_names)

    # GRADT — temperature gradient dT/dz from fast thermistors
    # First-difference approximation: dT/dz = fs * diff(T) / speed
    if gradt_names:
        v = L1.createVariable(
            "GRADT", "f8", ("N_GRADT_SENSORS", "TIME"), zlib=True
        )
        for i, name in enumerate(gradt_names):
            T_fast_ch = pf.channels[name]
            dTdt = np.diff(T_fast_ch) * pf.fs_fast
            dTdt = np.append(dTdt, dTdt[-1])
            v[i, :] = dTdt / speed_fast
        v.standard_name = "along_profile_gradient_of_seawater_temperature"
        v.units = "degrees_Celsius m-1"
        v.long_name = (
            "the gradient of seawater temperature along the path of "
            "the profiler measured by an FP07 thermistor"
        )
        v.sensor_names = ", ".join(gradt_names)

    # TEMP — fast-rate FP07 thermistor temperature
    if gradt_names:
        v = L1.createVariable(
            "TEMP", "f8", ("N_TEMP_SENSORS", "TIME"), zlib=True
        )
        for i, name in enumerate(gradt_names):
            v[i, :] = pf.channels[name]
        v.standard_name = "sea_water_temperature"
        v.units = "degree_Celsius"
        v.long_name = "sea water temperature in-situ ITS-90 scale"
        v.sensor_names = ", ".join(gradt_names)

    # TEMP_CTD — slow-rate CTD temperature (JAC_T)
    if temp_ctd_name:
        v = L1.createVariable("TEMP_CTD", "f8", ("TIME_SLOW",), zlib=True)
        v[:] = pf.channels[temp_ctd_name]
        v.standard_name = "sea_water_temperature"
        v.units = "degree_Celsius"
        v.long_name = "sea water temperature in-situ ITS-90 scale"

    # COND_CTD — slow-rate CTD conductivity (JAC_C)
    if cond_ctd_names:
        v = L1.createVariable("COND_CTD", "f8", ("TIME_SLOW",), zlib=True)
        v[:] = pf.channels[cond_ctd_names[0]]
        v.standard_name = "sea_water_electrical_conductivity"
        v.units = "mS cm-1"
        v.long_name = "sea water electrical conductivity"

    # PITCH
    if pitch_name:
        v = L1.createVariable("PITCH", "f8", ("TIME_SLOW",), zlib=True)
        v[:] = pf.channels[pitch_name]
        v.standard_name = "platform_pitch_angle_fore_down"
        v.units = "degrees"
        v.long_name = (
            "Positive pitch represents the front of the platform "
            "lowering as viewed by an observer on top of the platform "
            "facing forward"
        )

    # ROLL
    if roll_name:
        v = L1.createVariable("ROLL", "f8", ("TIME_SLOW",), zlib=True)
        v[:] = pf.channels[roll_name]
        v.standard_name = "platform_roll_angle_starboard_down"
        v.units = "degrees"
        v.long_name = (
            "Positive roll represents the right side of the platform "
            "falling as viewed by an observer on top of the platform "
            "facing forward"
        )

    # MAG — magnetometer
    if mag_names:
        v = L1.createVariable(
            "MAG", "f8", ("N_MAG_SENSORS", "TIME_SLOW"), zlib=True
        )
        for i, name in enumerate(mag_names):
            v[i, :] = pf.channels[name]
        v.standard_name = "magnetic_field"
        v.units = "micro_Tesla"
        v.long_name = "magnetic field from magnetometer"
        v.sensor_names = ", ".join(mag_names)

    # CHLA — chlorophyll-a fluorescence
    if chla_name:
        dim = ("TIME",) if pf.is_fast(chla_name) else ("TIME_SLOW",)
        v = L1.createVariable("CHLA", "f8", dim, zlib=True)
        v[:] = pf.channels[chla_name]
        v.standard_name = (
            "mass_concentration_of_chlorophyll_a_in_sea_water"
        )
        v.units = "ug L-1"
        v.long_name = "chlorophyll-a fluorescence"

    # TURB — turbidity
    if turb_name:
        dim = ("TIME",) if pf.is_fast(turb_name) else ("TIME_SLOW",)
        v = L1.createVariable("TURB", "f8", dim, zlib=True)
        v[:] = pf.channels[turb_name]
        v.standard_name = "sea_water_turbidity"
        v.units = "FTU"
        v.long_name = "sea water turbidity"

    # DOXY — dissolved oxygen concentration
    if doxy_name:
        dim = ("TIME",) if pf.is_fast(doxy_name) else ("TIME_SLOW",)
        v = L1.createVariable("DOXY", "f8", dim, zlib=True)
        v[:] = pf.channels[doxy_name]
        v.standard_name = (
            "mole_concentration_of_dissolved_molecular_oxygen_in_sea_water"
        )
        v.units = "umol L-1"
        v.long_name = "dissolved oxygen concentration"

    # DOXY_TEMP — oxygen optode sensor temperature
    if doxy_temp_name:
        dim = ("TIME",) if pf.is_fast(doxy_temp_name) else ("TIME_SLOW",)
        v = L1.createVariable("DOXY_TEMP", "f8", dim, zlib=True)
        v[:] = pf.channels[doxy_temp_name]
        v.units = "degree_Celsius"
        v.long_name = "oxygen optode sensor temperature"

    # Supplementary channels (V_Bat, Gnd, etc.)
    for name in supplementary:
        info = pf.channel_info[name]
        dim = ("TIME",) if pf.is_fast(name) else ("TIME_SLOW",)
        var_name = name.replace(" ", "_")
        v = L1.createVariable(var_name, "f4", dim, zlib=True)
        v[:] = pf.channels[name].astype(np.float32)
        v.units = info["units"]
        v.sensor_type = info["type"]
        v.long_name = name

    # Group attributes
    L1.time_reference_year = float(ref_year)
    L1.fs_fast = pf.fs_fast
    L1.fs_slow = pf.fs_slow
    L1.f_AA = 98.0
    L1.vehicle = pf.config["instrument_info"].get("vehicle", "").lower()

    ds.close()
    return pf, nc_filepath


# Backward-compatible alias
p_to_netcdf = p_to_L1


def _convert_one(args):
    """Worker function for parallel conversion. Must be top-level for pickling."""
    p_path, nc_path = args
    p_to_L1(p_path, nc_path)
    size_mb = nc_path.stat().st_size / 1e6
    return p_path.name, nc_path.name, size_mb


def convert_all(p_files: list[Path], output_dir: Path | None = None, jobs: int = 1) -> None:
    """Convert multiple .p files to ATOMIX L1_converted NetCDF.

    Parameters
    ----------
    p_files : list of Path
        .p files to convert.
    output_dir : Path, optional
        Output directory. If None, each .nc goes next to its .p file.
    jobs : int
        Number of parallel workers. 0 means use all CPU cores.
    """
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    work = []
    for pf_path in p_files:
        pf_path = Path(pf_path)
        if output_dir is not None:
            nc_path = output_dir / pf_path.with_suffix(".nc").name
        else:
            nc_path = pf_path.with_suffix(".nc")
        work.append((pf_path, nc_path))

    if jobs == 0:
        jobs = os.cpu_count() or 1

    if jobs == 1:
        for p_path, nc_path in work:
            print(f"{p_path.name} -> {nc_path.name} ... ", end="", flush=True)
            try:
                _name, _, size_mb = _convert_one((p_path, nc_path))
                print(f"{size_mb:.1f} MB")
            except (OSError, ValueError, RuntimeError) as e:
                print(f"ERROR: {e}")
    else:
        print(f"Converting {len(work)} files with {jobs} workers")
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = {pool.submit(_convert_one, w): w for w in work}
            for future in as_completed(futures):
                p_path, nc_path = futures[future]
                try:
                    _, _, size_mb = future.result()
                    print(f"  {p_path.name} -> {nc_path.name}  {size_mb:.1f} MB")
                except (OSError, ValueError, RuntimeError) as e:
                    print(f"  {p_path.name}  ERROR: {e}")

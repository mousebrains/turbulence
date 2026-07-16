# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""
NetCDF conversion for Rockland .p files -- ATOMIX L1_converted format.

Follows the ATOMIX shear-probe benchmark structure (Lueck et al., 2024,
doi:10.3389/fmars.2024.1334327, Figure 4).  The L1_converted group contains
time series converted to physical units, including speed-normalized shear
and temperature gradient.
"""

import logging
import os
import struct
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from odas_tpw.rsi.p_file import PFile

import numpy as np

logger = logging.getLogger(__name__)

# Minimum profiling speed to avoid singularity [m/s]
_SPEED_MIN = 0.05

# Butterworth smoothing time constant for speed computation [s]
_SPEED_TAU = 1.5


def _compute_speed(pf: "PFile") -> tuple[np.ndarray, np.ndarray]:
    """Compute profiling speed from pressure.

    Returns (speed_fast, speed_slow) in m/s.
    """
    from odas_tpw.scor160.profile import compute_speed_fast

    P_slow = pf.channels.get("P_dP", pf.channels.get("P"))
    if P_slow is None:
        raise ValueError("No pressure channel (P or P_dP) found")

    speed_fast, _W_slow = compute_speed_fast(
        P_slow,
        pf.t_fast,
        pf.t_slow,
        pf.fs_fast,
        pf.fs_slow,
        tau=_SPEED_TAU,
        speed_min=_SPEED_MIN,
    )
    return speed_fast, np.maximum(np.abs(_W_slow), _SPEED_MIN)


def _classify_channels(pf: "PFile") -> dict:
    """Classify PFile channels into ATOMIX roles.

    Returns a dict with keys: shear, vib, acc, mag, gradt, cond_ctd,
    temp_ctd, pitch, roll, chla, turb, doxy, doxy_temp, supplementary.
    """
    shear_names = sorted(n for n in pf.channels if pf.channel_info[n]["type"] == "shear")
    vib_names = sorted(n for n in pf.channels if pf.channel_info[n]["type"] == "piezo")
    acc_names = sorted(n for n in pf.channels if pf.channel_info[n]["type"] == "accel")
    mag_names = sorted(n for n in pf.channels if pf.channel_info[n]["type"] == "magn")
    gradt_names = sorted(
        n for n in pf.channels if pf.channel_info[n]["type"] == "therm" and pf.is_fast(n)
    )
    cond_ctd_names = sorted(n for n in pf.channels if pf.channel_info[n]["type"] == "jac_c")

    temp_ctd_name = next((n for n in pf.channels if pf.channel_info[n]["type"] == "jac_t"), None)
    pitch_name = roll_name = None
    for n in pf.channels:
        if pf.channel_info[n]["type"] == "inclxy":
            if "X" in n:
                pitch_name = n
            elif "Y" in n:
                roll_name = n

    chla_name = next((n for n in pf.channels if n.lower().startswith("chloro")), None)
    turb_name = next((n for n in pf.channels if n.lower().startswith("turb")), None)
    doxy_name = next((n for n in pf.channels if pf.channel_info[n]["type"] == "aroft_o2"), None)
    doxy_temp_name = next((n for n in pf.channels if pf.channel_info[n]["type"] == "aroft_t"), None)

    # Identify supplementary channels (not mapped to named variables)
    mapped = set(shear_names + vib_names + gradt_names + acc_names + mag_names + cond_ctd_names)
    for n in (
        temp_ctd_name,
        pitch_name,
        roll_name,
        "P",
        "P_dP",
        chla_name,
        turb_name,
        doxy_name,
        doxy_temp_name,
    ):
        if n is not None:
            mapped.add(n)
    for n in list(pf.channels):
        if pf.channel_info[n]["type"] == "therm" and not pf.is_fast(n):
            mapped.add(n)
    supplementary = sorted(n for n in pf.channels if n not in mapped)

    return {
        "shear": shear_names,
        "vib": vib_names,
        "acc": acc_names,
        "mag": mag_names,
        "gradt": gradt_names,
        "cond_ctd": cond_ctd_names,
        "temp_ctd": temp_ctd_name,
        "pitch": pitch_name,
        "roll": roll_name,
        "chla": chla_name,
        "turb": turb_name,
        "doxy": doxy_name,
        "doxy_temp": doxy_temp_name,
        "supplementary": supplementary,
    }


def _create_l1_variables(group, specs, complevel=4):
    """Write a list of variable specs to a NetCDF group.

    Each spec is (var_name, dtype, dims, data, attrs).  Specs with
    data=None are skipped (conditional variables).

    Parameters
    ----------
    complevel : int
        zlib compression level (0 = no compression, 1-9 = increasing).
    """
    use_zlib = complevel > 0
    for var_name, dtype, dims, data, attrs in specs:
        if data is None:
            continue
        v = group.createVariable(var_name, dtype, dims, zlib=use_zlib, complevel=complevel)  # type: ignore[call-overload]
        v[:] = data
        for key, val in attrs.items():
            setattr(v, key, val)


def _l1_variable_specs(
    pf, ch, time_fast, time_slow, time_units, P_fast, P_slow, speed_fast, gradt_data
):
    """Build the list of (name, dtype, dims, data, attrs) for L1 variables.

    Parameters
    ----------
    pf : PFile
        Parsed instrument file.
    ch : dict
        Channel classification from _classify_channels().
    time_fast, time_slow : ndarray
        Time vectors as decimal days.
    time_units : str
        CF time units string.
    P_fast, P_slow : ndarray
        Pressure arrays (fast interpolated, slow original).
    speed_fast : ndarray
        Profiling speed at fast rate.
    gradt_data : ndarray or None
        Pre-computed (N_GRADT_SENSORS, TIME) gradient array, or None.
    """
    shear_names = ch["shear"]
    vib_names = ch["vib"]
    acc_names = ch["acc"]
    mag_names = ch["mag"]
    gradt_names = ch["gradt"]
    cond_ctd_names = ch["cond_ctd"]
    temp_ctd_name = ch["temp_ctd"]
    pitch_name = ch["pitch"]
    roll_name = ch["roll"]
    chla_name = ch["chla"]
    turb_name = ch["turb"]
    doxy_name = ch["doxy"]
    doxy_temp_name = ch["doxy_temp"]

    # Multi-sensor stacking helper
    def _stack(names):
        return np.stack([pf.channels[n] for n in names], axis=0)

    # SHEAR — normalized by speed^2
    shear_data = None
    if shear_names:
        shear_data = np.stack(
            [pf.channels[n] / speed_fast**2 for n in shear_names],
            axis=0,
        )

    # TEMP (fast FP07)
    temp_data = _stack(gradt_names) if gradt_names else None

    specs = [
        # TIME vectors
        (
            "TIME",
            "f8",
            ("TIME",),
            time_fast,
            {"standard_name": "time", "long_name": "Decimal day", "units": time_units, "axis": "T"},
        ),
        (
            "TIME_SLOW",
            "f8",
            ("TIME_SLOW",),
            time_slow,
            {"standard_name": "time", "long_name": "Decimal day", "units": time_units, "axis": "T"},
        ),
        # Pressure
        (
            "PRES",
            "f8",
            ("TIME",),
            P_fast,
            {
                "standard_name": "sea_water_pressure",
                "units": "decibar",
                "long_name": "Sea water pressure, equals 0 at sea-level",
            },
        ),
        (
            "PRES_SLOW",
            "f8",
            ("TIME_SLOW",),
            P_slow,
            {
                "standard_name": "sea_water_pressure",
                "units": "decibar",
                "long_name": "Sea water pressure, equals 0 at sea-level",
            },
        ),
        # SHEAR
        (
            "SHEAR",
            "f8",
            ("N_SHEAR_SENSORS", "TIME"),
            shear_data,
            {
                "standard_name": "sea_water_velocity_shear",
                "units": "s-1",
                "long_name": "rate of change of cross axis sea water velocity "
                "along transect measured by shear probes",
                "sensor_names": ", ".join(shear_names),
            },
        ),
        # VIB
        (
            "VIB",
            "f8",
            ("N_VIB_SENSORS", "TIME"),
            _stack(vib_names) if vib_names else None,
            {
                "standard_name": "platform_vibration",
                "units": "1",
                "long_name": "platform vibration detected by piezo-accelerometers",
                "sensor_names": ", ".join(vib_names),
            },
        ),
        # ACC
        (
            "ACC",
            "f8",
            ("N_ACC_SENSORS", "TIME"),
            _stack(acc_names) if acc_names else None,
            {
                "standard_name": "platform_acceleration",
                "units": "m s-2",
                "long_name": "platform acceleration detected by accelerometers",
                "sensor_names": ", ".join(acc_names),
            },
        ),
        # GRADT
        (
            "GRADT",
            "f8",
            ("N_GRADT_SENSORS", "TIME"),
            gradt_data,
            {
                "standard_name": "along_profile_gradient_of_seawater_temperature",
                "units": "degrees_Celsius m-1",
                "long_name": "the gradient of seawater temperature along the path "
                "of the profiler measured by an FP07 thermistor",
                "sensor_names": ", ".join(gradt_names),
            },
        ),
        # TEMP (fast FP07)
        (
            "TEMP",
            "f8",
            ("N_TEMP_SENSORS", "TIME"),
            temp_data,
            {
                "standard_name": "sea_water_temperature",
                "units": "degree_Celsius",
                "long_name": "sea water temperature in-situ ITS-90 scale",
                "sensor_names": ", ".join(gradt_names),
            },
        ),
        # TEMP_CTD (slow)
        (
            "TEMP_CTD",
            "f8",
            ("TIME_SLOW",),
            pf.channels[temp_ctd_name] if temp_ctd_name else None,
            {
                "standard_name": "sea_water_temperature",
                "units": "degree_Celsius",
                "long_name": "sea water temperature in-situ ITS-90 scale",
            },
        ),
        # COND_CTD (slow)
        (
            "COND_CTD",
            "f8",
            ("TIME_SLOW",),
            pf.channels[cond_ctd_names[0]] if cond_ctd_names else None,
            {
                "standard_name": "sea_water_electrical_conductivity",
                "units": "mS cm-1",
                "long_name": "sea water electrical conductivity",
            },
        ),
        # PITCH
        (
            "PITCH",
            "f8",
            ("TIME_SLOW",),
            pf.channels[pitch_name] if pitch_name else None,
            {
                "standard_name": "platform_pitch_angle_fore_down",
                "units": "degrees",
                "long_name": "Positive pitch represents the front of the platform "
                "lowering as viewed by an observer on top of the "
                "platform facing forward",
            },
        ),
        # ROLL
        (
            "ROLL",
            "f8",
            ("TIME_SLOW",),
            pf.channels[roll_name] if roll_name else None,
            {
                "standard_name": "platform_roll_angle_starboard_down",
                "units": "degrees",
                "long_name": "Positive roll represents the right side of the "
                "platform falling as viewed by an observer on top of "
                "the platform facing forward",
            },
        ),
        # MAG
        (
            "MAG",
            "f8",
            ("N_MAG_SENSORS", "TIME_SLOW"),
            _stack(mag_names) if mag_names else None,
            {
                # 'micro_Tesla' is not UDUNITS-parseable; 'uT' is the canonical form
                # ('magnetic_field' is likewise not a CF standard name, but left as-is).
                "standard_name": "magnetic_field",
                "units": "uT",
                "long_name": "magnetic field from magnetometer",
                "sensor_names": ", ".join(mag_names),
            },
        ),
    ]

    # Biogeochemical sensors — dimension depends on sample rate
    for var_name, ch_name, attrs in [
        (
            "CHLA",
            chla_name,
            {
                "standard_name": "mass_concentration_of_chlorophyll_a_in_sea_water",
                "units": "ug L-1",
                "long_name": "chlorophyll-a fluorescence",
            },
        ),
        (
            "TURB",
            turb_name,
            {
                "standard_name": "sea_water_turbidity",
                "units": "FTU",
                "long_name": "sea water turbidity",
            },
        ),
        (
            "DOXY",
            doxy_name,
            {
                "standard_name": "mole_concentration_of_dissolved_molecular_oxygen_in_sea_water",
                "units": "umol L-1",
                "long_name": "dissolved oxygen concentration",
            },
        ),
        (
            "DOXY_TEMP",
            doxy_temp_name,
            {"units": "degree_Celsius", "long_name": "oxygen optode sensor temperature"},
        ),
    ]:
        if ch_name is not None:
            dim = ("TIME",) if pf.is_fast(ch_name) else ("TIME_SLOW",)
            specs.append((var_name, "f8", dim, pf.channels[ch_name], attrs))

    return specs


def p_to_L1(
    p_filepath: str | Path,
    nc_filepath: str | Path | None = None,
    complevel: int = 4,
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
    complevel : int
        zlib compression level (0 = no compression, 1-9 = increasing).
        Default: 4.

    Returns
    -------
    tuple of (PFile, Path)
        The parsed data object and the output path.
    """
    import netCDF4 as nc

    from odas_tpw.rsi.p_file import PFile, instrument_sn

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
    ch = _classify_channels(pf)
    shear_names = ch["shear"]
    vib_names = ch["vib"]
    acc_names = ch["acc"]
    mag_names = ch["mag"]
    gradt_names = ch["gradt"]
    supplementary = ch["supplementary"]

    # Pressure (prefer deconvolved P_dP, else P)
    P_slow = pf.channels.get("P_dP", pf.channels.get("P"))
    P_fast = np.interp(pf.t_fast, pf.t_slow, P_slow)

    # ---- Create NetCDF file ----
    ds = nc.Dataset(str(nc_filepath), "w", format="NETCDF4")

    # Global attributes
    ds.Conventions = "CF-1.13, ACDD-1.3"
    ds.title = f"VMP data from {pf.filepath.name}"
    ds.source = pf.config["instrument_info"].get("model", "")
    ds.platform_type = pf.config["instrument_info"].get("vehicle", "")
    ds.instrument_sn = instrument_sn(pf.config["instrument_info"])
    ds.operator = pf.config["cruise_info"].get("operator", "")
    ds.project = pf.config["cruise_info"].get("project", "")
    ds.start_time = pf.start_time.isoformat()
    ds.source_file = pf.filepath.name
    ds.history = f"L1 converted from {pf.filepath.name} on {datetime.now(UTC).isoformat()}"
    ds.configuration_string = pf.config_str
    # v1->v6 translation provenance (issue #141): complete key set (incl.
    # setup_file_md5 / sens_source, the sens^-2 audit trail) for both the
    # in-memory raw-v1 route and on-disk translated sources.
    for key, val in (getattr(pf, "v1_provenance", None) or {}).items():
        ds.setncattr(key, str(val))

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

    # Pre-compute GRADT arrays (need speed_fast before building specs)
    gradt_data = None
    if gradt_names:
        gradt_arrays = []
        for name in gradt_names:
            T_fast_ch = pf.channels[name]
            # GRADT uses the legacy ODAS 'first_difference' method
            # (fs*diff(T)/speed), NOT the ODAS default 'high_pass' on the
            # pre-emphasized signal, so this L1 GRADT differs sample-by-sample
            # from the ODAS reference product. (epsilon/chi use their own
            # spectral pre-emphasis path and do not consume this variable.)
            # np.diff on a single sample is empty; dTdt[-1] would IndexError.
            if T_fast_ch.size < 2:
                dTdt = np.zeros_like(T_fast_ch)
            else:
                dTdt = np.diff(T_fast_ch) * pf.fs_fast
                dTdt = np.append(dTdt, dTdt[-1])
            gradt_arrays.append(dTdt / speed_fast)
        gradt_data = np.stack(gradt_arrays, axis=0)

    # Build variable specs and write them
    specs = _l1_variable_specs(
        pf,
        ch,
        time_fast,
        time_slow,
        time_units,
        P_fast,
        P_slow,
        speed_fast,
        gradt_data,
    )
    _create_l1_variables(L1, specs, complevel=complevel)

    # Supplementary channels (V_Bat, Gnd, etc.) — dynamic names/types
    from odas_tpw.perturb.netcdf_schema import canonicalize_units

    use_zlib = complevel > 0
    for name in supplementary:
        info = pf.channel_info[name]
        dim = ("TIME",) if pf.is_fast(name) else ("TIME_SLOW",)
        var_name = name.replace(" ", "_")
        v = L1.createVariable(var_name, "f4", dim, zlib=use_zlib, complevel=complevel)  # type: ignore[call-overload]
        v[:] = pf.channels[name].astype(np.float32)
        # RSI emits non-UDUNITS unit strings (e.g. 'deg', 'mS_cm-1', 'deg_C');
        # canonicalize to match the per-profile writer (profile.py).
        v.units = canonicalize_units(info["units"])
        v.sensor_type = info["type"]
        v.long_name = name

    # Group attributes
    L1.time_reference_year = float(ref_year)
    L1.fs_fast = pf.fs_fast
    L1.fs_slow = pf.fs_slow
    # 98.0 Hz is the RSI/ODAS default anti-alias cutoff, hardcoded because the
    # AA cutoff is not parsed from the instrument config. Every writer and every
    # read-with-recovery (adapter.py, io.py) defaults to the same 98.0, so it is
    # self-consistent and does not alter published epsilon. Parsing the actual
    # AA cutoff from the config is a separate follow-up. (#104 U1-4.)
    L1.f_AA = 98.0
    L1.vehicle = pf.config["instrument_info"].get("vehicle", "").lower()

    ds.close()
    return pf, nc_filepath


# Backward-compatible alias
p_to_netcdf = p_to_L1


def _convert_one(args):
    """Worker function for parallel conversion. Must be top-level for pickling."""
    if len(args) == 3:
        p_path, nc_path, complevel = args
    else:
        p_path, nc_path = args
        complevel = 4
    p_to_L1(p_path, nc_path, complevel=complevel)
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
    # Detect output-name collisions: when output_dir is set we flatten to the
    # basename, so same-named .p files in different directories would map to one
    # .nc — silently overwriting (serial) or racing/corrupting (parallel).
    seen: dict[Path, Path] = {}
    for pf_path in p_files:
        pf_path = Path(pf_path)
        if output_dir is not None:
            nc_path = output_dir / pf_path.with_suffix(".nc").name
        else:
            nc_path = pf_path.with_suffix(".nc")
        prior = seen.get(nc_path)
        if prior is not None and prior != pf_path:
            raise ValueError(
                f"Output name collision: {prior} and {pf_path} both map to "
                f"{nc_path}. Use distinct filenames or per-source output dirs."
            )
        seen[nc_path] = pf_path
        work.append((pf_path, nc_path))

    if jobs == 0:
        jobs = os.cpu_count() or 1

    if jobs == 1:
        for p_path, nc_path in work:
            try:
                _name, _, size_mb = _convert_one((p_path, nc_path))
                logger.info(f"{p_path.name} -> {nc_path.name}  {size_mb:.1f} MB")
            # struct.error: truncated/short-header .p file must not abort the batch
            except (OSError, ValueError, RuntimeError, struct.error) as e:
                logger.error(f"{p_path.name}: {e}")
    else:
        logger.info(f"Converting {len(work)} files with {jobs} workers")
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = {pool.submit(_convert_one, w): w for w in work}
            for future in as_completed(futures):
                p_path, nc_path = futures[future]
                try:
                    _, _, size_mb = future.result()
                    logger.info(f"{p_path.name} -> {nc_path.name}  {size_mb:.1f} MB")
                # struct.error: truncated/short-header .p file must not abort the batch
                except (OSError, ValueError, RuntimeError, struct.error) as e:
                    logger.error(f"{p_path.name}: {e}")

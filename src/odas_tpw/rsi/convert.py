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
import re
import struct
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from odas_tpw.rsi.p_file import PFile

import numpy as np

from odas_tpw._version_info import provenance_attrs

logger = logging.getLogger(__name__)

# Minimum profiling speed to avoid singularity [m/s]
_SPEED_MIN = 0.05

# Butterworth smoothing time constant for speed computation [s]
_SPEED_TAU = 1.5

# Which channel p_to_L1 writes into which L1 variable, stated once for both
# directions: the writer fills each variable from a _classify_channels role,
# and l1_native_channels reads the same tables backwards to give a variable,
# or a row of one, its native channel name again.
#
# Stacked variables hold one row per channel, named in ``sensor_names``. Each
# role is chosen by sensor type (gradt: a FAST therm channel), so the type of
# every row is known from its variable. TEMP holds the FP07 channels
# themselves; GRADT a first difference of them, which no channel holds.
_L1_STACKED_ROLES: dict[str, str] = {
    "SHEAR": "shear",
    "VIB": "vib",
    "ACC": "acc",
    "MAG": "mag",
    "GRADT": "gradt",
    "TEMP": "gradt",
}
_ROLE_TYPES: dict[str, str] = {
    "shear": "shear",
    "vib": "piezo",
    "acc": "accel",
    "mag": "magn",
    "gradt": "therm",
}
# Single-channel variables and the role whose channel each holds (COND_CTD:
# the first of the role's sorted list). PRES_SLOW holds P_dP, or P without one,
# and PRES the same interpolated onto TIME.
_L1_SINGLE_ROLES: dict[str, str] = {
    "TEMP_CTD": "temp_ctd",
    "COND_CTD": "cond_ctd",
    "PITCH": "pitch",
    "ROLL": "roll",
    "CHLA": "chla",
    "TURB": "turb",
    "DOXY": "doxy",
    "DOXY_TEMP": "doxy_temp",
}


def _speed_from_pressure(
    P_slow: np.ndarray, t_fast: np.ndarray, t_slow: np.ndarray, fs_fast: float, fs_slow: float
) -> tuple[np.ndarray, np.ndarray]:
    """``(speed_fast, speed_slow)`` [m/s] from slow pressure: the speed whose
    square divides SHEAR in an L1 file."""
    from odas_tpw.scor160.profile import compute_speed_fast

    speed_fast, _W_slow = compute_speed_fast(
        P_slow,
        t_fast,
        t_slow,
        fs_fast,
        fs_slow,
        tau=_SPEED_TAU,
        speed_min=_SPEED_MIN,
    )
    return speed_fast, np.maximum(np.abs(_W_slow), _SPEED_MIN)


def _compute_speed(pf: "PFile") -> tuple[np.ndarray, np.ndarray]:
    """Compute profiling speed from pressure.

    Returns (speed_fast, speed_slow) in m/s.
    """
    P_slow = pf.channels.get("P_dP", pf.channels.get("P"))
    if P_slow is None:
        raise ValueError("No pressure channel (P or P_dP) found")
    return _speed_from_pressure(P_slow, pf.t_fast, pf.t_slow, pf.fs_fast, pf.fs_slow)


def _single_channel_roles(names: Sequence[str], types: Mapping[str, str]) -> dict[str, Any]:
    """The channels of the single-channel L1 variables' roles, chosen from *names*.

    *names* in channel order (the first match wins); ``cond_ctd`` is every
    conductivity channel, sorted, the others one name or None. p_to_L1 chooses
    from a PFile's channels and l1_native_channels from the configuration the
    file embeds, so both choose the same channel.
    """
    # jac_* is the JFE Advantech readout, rsijac_* RSI's own JAC-CT board; both
    # are the CTD's conductivity and temperature.
    cond_ctd_names = sorted(n for n in names if types[n] in ("jac_c", "rsijac_c"))
    temp_ctd_name = next((n for n in names if types[n] in ("jac_t", "rsijac_t")), None)
    pitch_name = roll_name = None
    for n in names:
        if types[n] == "inclxy":
            if "X" in n:
                pitch_name = n
            elif "Y" in n:
                roll_name = n
    return {
        "cond_ctd": cond_ctd_names,
        "temp_ctd": temp_ctd_name,
        "pitch": pitch_name,
        "roll": roll_name,
        "chla": next((n for n in names if n.lower().startswith("chloro")), None),
        "turb": next((n for n in names if n.lower().startswith("turb")), None),
        "doxy": next((n for n in names if types[n] == "aroft_o2"), None),
        "doxy_temp": next((n for n in names if types[n] == "aroft_t"), None),
    }


def _single_channel_sources(ch: Mapping[str, Any]) -> dict[str, str]:
    """ATOMIX name -> channel of each single-channel L1 variable *ch*'s roles fill."""
    sources: dict[str, str] = {}
    for var_name, role in _L1_SINGLE_ROLES.items():
        chosen = ch[role]
        if isinstance(chosen, list):
            chosen = chosen[0] if chosen else None
        if chosen is not None:
            sources[var_name] = chosen
    return sources


def _classify_channels(pf: "PFile") -> dict:
    """Classify PFile channels into ATOMIX roles.

    Returns a dict with keys: shear, vib, acc, mag, gradt, cond_ctd,
    temp_ctd, pitch, roll, chla, turb, doxy, doxy_temp, supplementary.
    """
    names = list(pf.channels)
    types = {n: pf.channel_info[n]["type"] for n in names}

    def _typed(role: str) -> list[str]:
        return sorted(n for n in names if types[n] == _ROLE_TYPES[role])

    shear_names = _typed("shear")
    vib_names = _typed("vib")
    acc_names = _typed("acc")
    mag_names = _typed("mag")
    gradt_names = [n for n in _typed("gradt") if pf.is_fast(n)]
    single = _single_channel_roles(names, types)
    cond_ctd_names = single["cond_ctd"]

    # Identify supplementary channels (not mapped to named variables)
    mapped = set(shear_names + vib_names + gradt_names + acc_names + mag_names + cond_ctd_names)
    for n in (
        single["temp_ctd"],
        single["pitch"],
        single["roll"],
        "P",
        "P_dP",
        single["chla"],
        single["turb"],
        single["doxy"],
        single["doxy_temp"],
    ):
        if n is not None:
            mapped.add(n)
    for n in names:
        if types[n] == "therm" and not pf.is_fast(n):
            mapped.add(n)
    supplementary = sorted(n for n in names if n not in mapped)

    return {
        "shear": shear_names,
        "vib": vib_names,
        "acc": acc_names,
        "mag": mag_names,
        "gradt": gradt_names,
        **single,
        "supplementary": supplementary,
    }


def _cal_attrs(pf: "PFile", names: list[str]) -> dict:
    """``cal_<key>``: the coefficients this variable's config supplied.

    A converted L1 file is routinely read years later, on a different machine,
    without the instrument config that produced it — and a shear record is
    meaningless without the ``sens`` and ``diff_gain`` that scaled it.  Writing
    the coefficients onto the variable makes the file self-describing, and lets
    an audit answer "which sensitivity is baked into this epsilon?" from the
    product rather than by re-deriving it from a ``.p`` file that may since have
    been re-patched.  (Concept from Jesse Cusack's pyturb.)

    For a stacked variable the value is an array parallel to ``sensor_names``,
    so probe 0's coefficient stays attached to probe 0.  A coefficient that
    only some probes in the stack carry is filled with NaN for the others
    rather than dropped: a short or re-ordered array would silently
    mis-attribute a coefficient to the wrong probe.

    A coefficient the config omitted — where the converter applied a documented
    default — is NOT listed; see
    :meth:`~odas_tpw.rsi.channels.CalRecorder.calibration`.
    """
    cals = [pf.channel_info[n].get("cal", {}) for n in names]
    if not cals:
        return {}
    keys = sorted(set().union(*(set(c) for c in cals)))
    attrs: dict = {}
    for key in keys:
        values = [c.get(key, np.nan) for c in cals]
        attrs[f"cal_{key}"] = (
            float(values[0]) if len(values) == 1 else np.array(values, dtype="f8")
        )
    return attrs


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
    single = _single_channel_sources(ch)

    # Multi-sensor stacking helper
    def _stack(names):
        return np.stack([pf.channels[n] for n in names], axis=0)

    def _single(var_name):
        name = single.get(var_name)
        return pf.channels[name] if name is not None else None

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
            _single("TEMP_CTD"),
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
            _single("COND_CTD"),
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
            _single("PITCH"),
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
            _single("ROLL"),
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
    for var_name, attrs in [
        (
            "CHLA",
            {
                "standard_name": "mass_concentration_of_chlorophyll_a_in_sea_water",
                "units": "ug L-1",
                "long_name": "chlorophyll-a fluorescence",
            },
        ),
        (
            "TURB",
            {
                "standard_name": "sea_water_turbidity",
                "units": "FTU",
                "long_name": "sea water turbidity",
            },
        ),
        (
            "DOXY",
            {
                "standard_name": "mole_concentration_of_dissolved_molecular_oxygen_in_sea_water",
                "units": "umol L-1",
                "long_name": "dissolved oxygen concentration",
            },
        ),
        (
            "DOXY_TEMP",
            {"units": "degree_Celsius", "long_name": "oxygen optode sensor temperature"},
        ),
    ]:
        ch_name = single.get(var_name)
        if ch_name is not None:
            dim = ("TIME",) if pf.is_fast(ch_name) else ("TIME_SLOW",)
            specs.append((var_name, "f8", dim, pf.channels[ch_name], attrs))

    # Calibration provenance, applied at one site rather than in each of the
    # fifteen specs above.  Stacked variables already declare their source
    # channels in ``sensor_names``; the single-channel ones are named by the
    # role tables above.
    sources: dict[str, list[str]] = {
        var_name: ch[role] for var_name, role in _L1_STACKED_ROLES.items()
    }
    sources.update({var_name: [name] for var_name, name in single.items()})
    if "P" in pf.channels:
        sources["PRES"] = sources["PRES_SLOW"] = ["P"]
    specs = [
        (name, dtype, dims, data, {**attrs, **_cal_attrs(pf, sources.get(name, []))})
        for name, dtype, dims, data, attrs in specs
    ]

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
    # The .p record framing, carried on to every profile rsi-tpw prof cuts from
    # this file (record-locked shear removal needs it; see rsi.record_locked).
    # The whole file is written, so its first fast sample opens a record.
    ds.samples_per_record_fast = int(pf.samples_per_record)
    ds.record_offset_fast = 0
    for key, val in provenance_attrs(f"L1 converted from {pf.filepath.name}").items():
        ds.setncattr(key, val)
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
        for cal_key, cal_val in info.get("cal", {}).items():
            v.setncattr(f"cal_{cal_key}", float(cal_val))

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
    # RDL bad-buffer dropouts of EVERY channel, the ones this file leaves out
    # (T1, P_dP) included, as the .p route's profiles carry them. Written even
    # when empty: its absence is what says a file cannot tell.
    from odas_tpw.rsi.bad_buffer import L1_SPANS_ATTR, channel_masks, encode_channel_spans

    L1.setncattr(L1_SPANS_ATTR, encode_channel_spans(channel_masks(pf)))

    ds.close()
    return pf, nc_filepath


# Backward-compatible alias
p_to_netcdf = p_to_L1


def _pfile_layout(config: Mapping[str, Any]) -> tuple[dict[str, str], dict[str, str]]:
    """``(types, pairs)``: the channels a PFile of *config* holds, and how.

    ``types`` maps each channel to its sensor type, in PFile's order; ``pairs``
    maps each pre-emphasized X_dX that PFile deconvolves to its base X.
    Mirrors PFile:

    - a channel is a section with a name, a type and an ``id`` (or
      ``id_even``/``id_odd``), a later section of a name replacing an earlier
      one; it is held when the address matrix samples every one of its ids;
    - ``_apply_deconvolution`` takes each held X_dX whose section has a
      ``diff_gain`` and is not shear, and builds X from it whether or not the
      matrix samples X: an unsampled X is held after the sampled channels, with
      its own section's type ("raw" without one). An X_dX is converted with a
      configured X's section, so it takes X's type.
    """
    in_matrix = {a for row in config.get("matrix", []) for a in row}
    sections: dict[str, dict[str, Any]] = {}
    held: dict[str, str] = {}
    for ch in config.get("channels", []):
        if "name" not in ch or "type" not in ch:
            continue
        if "id" in ch:
            id_str = ch["id"]
        elif "id_even" in ch and "id_odd" in ch:
            id_str = f"{ch['id_even']} {ch['id_odd']}"
        else:
            continue
        try:
            ids = [int(x) for x in str(id_str).replace(",", " ").split()]
        except ValueError:
            continue
        name = str(ch["name"]).strip()
        sections[name] = ch
        if 1 <= len(ids) <= 2 and all(i in in_matrix for i in ids):
            held[name] = str(ch["type"]).strip().lower()

    def _type(name: str) -> str:
        return str(sections[name]["type"]).strip().lower() if name in sections else "raw"

    pairs: dict[str, str] = {}
    for name in list(held):
        m = re.match(r"^(\w+)_d\1$", name)
        section = sections.get(name, {})
        if (
            m
            and "diff_gain" in section
            and str(section.get("type", "")).strip().lower() not in ("shear", "xmp_shear")
        ):
            base = m.group(1)
            pairs[name] = base
            held.setdefault(base, _type(base))
            if base in sections:
                held[name] = _type(base)
    return held, pairs


# The first release whose p_to_L1 divides SHEAR by the speed l1_native_channels
# recomputes: compute_speed_fast last changed in 579ce0f (#74) and abb1930
# (#54), both first released in v0.3.0, and _SPEED_TAU, _SPEED_MIN and the
# pressure it reads have not changed since. Raise it when any of them does.
_L1_SPEED_SINCE = (0, 3, 0)


def _l1_speed_is_current(software_version: object) -> bool:
    """Whether an L1 file's ``software_version`` shows today's SHEAR speed."""
    m = re.match(r"^\S+ (\d+)\.(\d+)\.(\d+)", str(software_version or ""))
    return m is not None and tuple(int(x) for x in m.groups()) >= _L1_SPEED_SINCE


def l1_native_channels(
    group: Any,
    config: Mapping[str, Any] | None,
    t_fast: np.ndarray,
    t_slow: np.ndarray,
    fs_fast: float,
    fs_slow: float,
    *,
    context: str = "",
    software_version: object = None,
) -> dict[str, tuple[np.ndarray, bool, dict[str, Any]]]:
    """Invert :func:`p_to_L1`: an ``L1_converted`` group's channels by native name.

    Returns ``{name: (data, is_fast, info)}`` in the order a PFile lists its
    channels, ``info`` shaped like ``PFile.channel_info`` (``units`` as the L1
    file states them, ``type``, and a shear channel's ``comment``), so a
    profile cut from the L1 file carries what one cut from the ``.p`` does:

    - the rows of a stacked variable take the names in its ``sensor_names``
      and the sensor type its role is chosen by. SHEAR rows are multiplied
      back by the square of the speed p_to_L1 divided them by, recomputed from
      PRES_SLOW (short NaN gaps repaired, as for profile detection) on the
      sample grid *t_fast*/*t_slow* (so pass the exact grid). That is today's
      speed: a WARNING says so when *software_version*, the file's own, does
      not show a release that divided by the same one;
    - a single-channel variable takes the name of the channel its role
      chooses from *config*, the configuration the file embeds; PRES_SLOW
      is ``P``;
    - channels p_to_L1 leaves out because PFile's deconvolution makes them
      repeat another are rebuilt as PFile builds them: the base X of a fast
      X_dX is every n-th sample of it (T1 of T1_dT1, sampled or not), a slow
      X_dX equals X (P_dP);
    - supplementary variables keep their names and their own ``sensor_type``.

    GRADT (a first difference of TEMP) and PRES (PRES_SLOW interpolated onto
    TIME) are derived and left out when their source is present. Anything
    else keeps its L1 name: an unknown variable, every single-channel
    variable when *config* is None, and each row of a stacked variable whose
    ``sensor_names`` does not name every row (``SHEAR_1``..., as stored, with a
    WARNING). A configured channel that ends up missing is logged at WARNING.

    Raises ValueError when two variables or rows map to one channel name.
    """
    from odas_tpw.rsi.p_file import SHEAR_INTERMEDIATE_COMMENT
    from odas_tpw.rsi.profile import _MAX_REPAIR_GAP_S, _nc_filled, _repair_nans

    prefix = f"{context}: " if context else ""
    held, pairs = _pfile_layout(config) if config is not None else ({}, {})
    single = _single_channel_sources(_single_channel_roles(list(held), held))
    variables = group.variables

    out: dict[str, tuple[np.ndarray, bool, dict[str, Any]]] = {}

    def _put(name: str, data: np.ndarray, fast: bool, info: dict[str, Any], src: str) -> None:
        if name in out:
            raise ValueError(f"{prefix}L1 {src} names channel {name!r} a second time")
        out[name] = (data, fast, info)

    def _shear_speed(pres_slow: Any) -> np.ndarray:
        if not _l1_speed_is_current(software_version):
            logger.warning(
                "%sSHEAR: software_version %r is not v%s or later, whose speed "
                "restores it here; if the file divided it by another speed, the "
                "shear channels come out scaled by (speed now / speed then)**2",
                prefix,
                software_version,
                ".".join(map(str, _L1_SPEED_SINCE)),
            )
        # One NaN would spread through the whole speed in filtfilt.
        P, _ = _repair_nans(
            _nc_filled(pres_slow),
            "PRES_SLOW",
            context,
            max(1, round(_MAX_REPAIR_GAP_S * fs_slow)),
        )
        return _speed_from_pressure(P, t_fast, t_slow, fs_fast, fs_slow)[0]

    for vname, var in variables.items():
        dims = var.dimensions
        if vname in ("TIME", "TIME_SLOW") or not dims or dims[-1] not in ("TIME", "TIME_SLOW"):
            continue
        fast = dims[-1] == "TIME"
        units = str(getattr(var, "units", ""))
        if len(dims) == 2:
            if vname == "GRADT" and "TEMP" in variables:
                continue
            rows = _nc_filled(var)
            role = _L1_STACKED_ROLES.get(vname)
            stype = _ROLE_TYPES[role] if role is not None else ""
            listed = getattr(var, "sensor_names", None)
            names = [s.strip() for s in str(listed or "").split(",")]
            names = [s for s in names if s]
            reason = None
            if role is None or vname == "GRADT":
                reason = "no channel holds it"
            elif len(names) != len(rows):
                reason = f"sensor_names {listed!r} does not name each of its {len(rows)} rows"
            elif vname == "SHEAR" and "PRES_SLOW" not in variables:
                reason = "without PRES_SLOW its speed normalization cannot be undone"
            if reason is not None:
                logger.warning(
                    "%s%s: %s; kept as stored, as %s_1..%s_%d",
                    prefix,
                    vname,
                    reason,
                    vname,
                    vname,
                    len(rows),
                )
                for i, row in enumerate(rows, 1):
                    comment = f"row {i} of the L1 variable {vname}, as stored there"
                    info = {"units": units, "type": stype, "comment": comment}
                    _put(f"{vname}_{i}", row, fast, info, vname)
                continue
            if vname == "SHEAR":
                rows = rows * _shear_speed(variables["PRES_SLOW"]) ** 2
            for name, row in zip(names, rows, strict=True):
                info = {"units": units, "type": stype}
                if stype == "shear":
                    info["comment"] = SHEAR_INTERMEDIATE_COMMENT
                _put(name, row, fast, info, vname)
        elif len(dims) == 1:
            if vname == "PRES" and "PRES_SLOW" in variables:
                continue
            data = _nc_filled(var)
            if vname == "PRES_SLOW":
                _put("P", data, fast, {"units": units, "type": held.get("P", "")}, vname)
            elif vname in single:
                name = single[vname]
                _put(name, data, fast, {"units": units, "type": held[name]}, vname)
            else:
                # A supplementary channel goes in under its own name (spaces as
                # underscores; long_name the name itself) with its sensor_type.
                stype = str(getattr(var, "sensor_type", ""))
                long_name = str(getattr(var, "long_name", ""))
                name = long_name if stype and long_name.replace(" ", "_") == vname else vname
                _put(name, data, fast, {"units": units, "type": stype}, vname)

    n_slow = len(t_slow)
    ratio = len(t_fast) // n_slow if n_slow else 1
    for dx_name, base in pairs.items():
        if dx_name in out and base not in out:
            data, fast, info = out[dx_name]
            base_data = data[::ratio][:n_slow] if fast else data
            _put(base, base_data, False, {**info, "type": held[base]}, dx_name)
        elif base in out and dx_name not in out and not out[base][1]:
            # p_to_L1 writes every fast channel, so an X_dX it left out is slow.
            data, fast, info = out[base]
            _put(dx_name, data, fast, {**info, "type": held[dx_name]}, base)

    missing = [n for n in held if n not in out]
    if missing:
        logger.warning(
            "%sconfigured channel(s) %s are not in the L1 file; profiles cut from it lack them",
            prefix,
            ", ".join(missing),
        )
    order = [n for n in held if n in out] + [n for n in out if n not in held]
    return {n: out[n] for n in order}


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

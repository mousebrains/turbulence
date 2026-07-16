# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Shared helpers for loading instrument data and preparing profiles.

These functions bridge PFile/NetCDF reading with spectral processing.
Originally part of dissipation.py, extracted so that both epsilon and
chi packages can use them without circular dependencies.
"""

from __future__ import annotations

import logging
import re
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    import xarray as xr

    from odas_tpw.rsi.p_file import PFile

logger = logging.getLogger(__name__)


class ChannelsDict(TypedDict, total=False):
    """Type for the dict returned by :func:`load_channels`."""

    shear: list[tuple[str, np.ndarray]]
    accel: list[tuple[str, np.ndarray]]
    P: np.ndarray
    T: np.ndarray
    C: np.ndarray
    JAC_T: np.ndarray
    U_EM: np.ndarray
    Incl_X: np.ndarray
    Incl_Y: np.ndarray
    speed_fast: np.ndarray
    W_slow: np.ndarray
    t_fast: np.ndarray
    t_slow: np.ndarray
    fs_fast: float
    fs_slow: float
    is_profile: bool
    metadata: dict[str, str]
    vehicle: str
    # Sensor type per loaded vibration/accelerometer channel name (.p:
    # pf.channel_info; nc: the sensor_type attr extract_profiles writes), so
    # _build_l1data_from_channels can label piezo-typed channels "VIB" like
    # the adapter (#131 W5-ii, W3 review F5).
    channel_types: dict[str, str]


# ---------------------------------------------------------------------------
# Channel name patterns for RSI instruments
# ---------------------------------------------------------------------------

SH_PATTERN = re.compile(r"^sh\d+$")
AC_PATTERN = re.compile(r"^A[xyz]$")
T_PATTERN = re.compile(r"^T\d+$")
DT_PATTERN = re.compile(r"^T\d+_dT\d+$")
# Reference-temperature candidates: numbered FP07s (T1, T2, ...) plus a bare
# slow "T" (some per-profile NC stores carry only that).
REF_T_PATTERN = re.compile(r"^T\d*$")

# ---------------------------------------------------------------------------
# Reference temperature/conductivity source selection + plausibility QC
# (issue #131 B1). ODAS divergence, deliberate: ODAS uses T1 (or the
# constant_temp override) with no plausibility check and silently substitutes
# 10 degC when temperature is missing; here "auto" walks a QC'd candidate
# chain (T1..Tn, bare T, JAC_T) and raises a per-file error instead of
# publishing products computed from an implausible temperature.
# ---------------------------------------------------------------------------

TEMP_QC_MIN = -3.0  # degC; below the freezing point of seawater with margin
# for ice-shelf-cavity ISW (in-situ T down to ~-2.6 degC)
TEMP_QC_MAX = 40.0  # degC; warmest plausible open-ocean surface water
TEMP_QC_MAX_BAD_FRAC = 0.10  # max fraction of finite samples outside range
TEMP_QC_MIN_FINITE_FRAC = 0.5  # min fraction of finite samples
IN_WATER_P_MIN = 0.5  # dbar; QC evaluates in-water samples only (deck/air
# readings must not fail a healthy sensor)


def reference_temperature_qc(
    T: npt.ArrayLike,
    n_min: int = 100,
    *,
    pressure: npt.ArrayLike | None = None,
) -> str | None:
    """Plausibility QC for a candidate reference-temperature channel.

    Returns ``None`` when the channel is plausible, else a short failure
    reason. When *pressure* (same length as *T*) is given, only in-water
    samples (P > ``IN_WATER_P_MIN``) are evaluated, falling back to the full
    channel when none qualify (e.g. a bench recording) — deck data in
    tropical sun or polar winter must not fail a healthy sensor.

    Checks, in order: mostly-non-finite; median outside the plausible ocean
    range (catches railed sensors, e.g. a corpus-wide 58.46 degC, or a dead
    -17.09 degC); too many samples outside the range (catches drift with an
    in-range median); exactly constant with at least *n_min* samples (a
    railed ADC — note a genuinely constant lab bath would also trip this).
    """
    T = np.asarray(T, dtype=np.float64)
    if pressure is not None:
        P = np.asarray(pressure, dtype=np.float64)
        if P.shape == T.shape:
            in_water = P > IN_WATER_P_MIN
            if np.any(in_water):
                T = T[in_water]
    if T.size == 0:
        return "no samples"
    finite = np.isfinite(T)
    finite_frac = finite.mean()
    if finite_frac < TEMP_QC_MIN_FINITE_FRAC:
        return f"mostly non-finite ({100 * finite_frac:.0f}% finite)"
    median = float(np.nanmedian(T))
    if not (TEMP_QC_MIN <= median <= TEMP_QC_MAX):
        return (
            f"median {median:.1f} degC outside plausible ocean range "
            f"[{TEMP_QC_MIN:g}, {TEMP_QC_MAX:g}] degC"
        )
    T_finite = T[finite]
    bad_frac = float(
        np.mean((T_finite < TEMP_QC_MIN) | (T_finite > TEMP_QC_MAX))
    )
    if bad_frac > TEMP_QC_MAX_BAD_FRAC:
        return (
            f"{100 * bad_frac:.0f}% of samples outside plausible ocean range "
            f"[{TEMP_QC_MIN:g}, {TEMP_QC_MAX:g}] degC"
        )
    if T.size >= n_min and np.nanmax(T) == np.nanmin(T):
        return f"constant at {median:.2f} degC (railed sensor?)"
    return None


def temperature_candidates(channels: Mapping[str, Any], n_slow: int) -> list[str]:
    """Slow reference-temperature candidates in ``auto`` QC order.

    Numbered FP07s first (T1, T2, ... — numeric order), then a bare slow
    ``T``, then ``JAC_T``. Only channels of slow length qualify.
    """

    def _is_slow(name: str) -> bool:
        try:
            return len(channels[name]) == n_slow
        except TypeError:
            return False

    numbered = sorted(
        (n for n in channels if REF_T_PATTERN.match(n) and n != "T" and _is_slow(n)),
        key=lambda n: int(n[1:]),
    )
    out = numbered
    if "T" in channels and _is_slow("T"):
        out.append("T")
    if "JAC_T" in channels and _is_slow("JAC_T"):
        out.append("JAC_T")
    return out


def resolve_temperature_channel(
    channels: Mapping[str, Any],
    n_slow: int,
    requested: str = "auto",
    *,
    pressure: npt.ArrayLike | None = None,
    context: str = "",
) -> tuple[str, str | None]:
    """Select the reference-temperature channel (viscosity for epsilon;
    viscosity and kappa_T for chi; the published ``T_mean``).

    Parameters
    ----------
    channels : mapping of name -> array
        Available channels (fast channels are filtered out by length).
    n_slow : int
        Slow-grid length; a reference temperature must be a slow channel.
    requested : str
        ``"auto"`` walks the candidate chain (T1..Tn, bare T, JAC_T) and
        returns the first that passes :func:`reference_temperature_qc`; each
        skipped candidate emits one warning naming the reason, and when none
        pass a ValueError lists every candidate + reason. An explicit channel
        name is honored even when it fails QC (warn loudly but proceed — the
        user's explicit choice); missing or non-slow names raise ValueError.
        A *numeric* temperature (constant reference, ODAS ``constant_temp``
        parity) is handled by the callers, not here.
    pressure : array, optional
        Slow pressure for in-water-only QC (see reference_temperature_qc).
    context : str
        Prefix for warnings/errors (e.g. the source file name).

    Returns
    -------
    (name, qc_reason_or_None)
    """
    prefix = f"{context}: " if context else ""
    if requested == "auto":
        candidates = temperature_candidates(channels, n_slow)
        if not candidates:
            raise ValueError(
                f"{prefix}no slow temperature channel found (looked for "
                "T1..Tn, T, JAC_T); pass --temperature / temperature: to "
                "select a channel explicitly, or a number for a constant "
                "reference temperature"
            )
        reasons: list[tuple[str, str]] = []
        for name in candidates:
            reason = reference_temperature_qc(channels[name], pressure=pressure)
            if reason is None:
                return name, None
            reasons.append((name, reason))
            warnings.warn(
                f"{prefix}skipping reference temperature channel {name}: {reason}",
                stacklevel=2,
            )
        detail = "; ".join(f"{n}: {r}" for n, r in reasons)
        raise ValueError(
            f"{prefix}no plausible reference temperature channel ({detail}); "
            "pass --temperature / temperature: to select one explicitly, or "
            "a number for a constant reference temperature"
        )

    if requested not in channels:
        raise ValueError(
            f"{prefix}temperature channel {requested!r} not found; slow "
            f"channels: {sorted(n for n in channels if _len_is(channels[n], n_slow))}"
        )
    if not _len_is(channels[requested], n_slow):
        raise ValueError(
            f"{prefix}temperature channel {requested!r} is not a slow channel "
            f"(length {len(channels[requested])} != {n_slow})"
        )
    reason = reference_temperature_qc(channels[requested], pressure=pressure)
    if reason is not None:
        warnings.warn(
            f"{prefix}explicitly selected temperature channel {requested} "
            f"fails plausibility QC ({reason}); proceeding with it anyway — "
            "pass a numeric temperature for a constant reference instead",
            stacklevel=2,
        )
    return requested, reason


def _len_is(arr: Any, n: int) -> bool:
    try:
        return len(arr) == n
    except TypeError:
        return False


def resolve_conductivity_channel(
    channels: Mapping[str, Any],
    n_slow: int,
    requested: str = "auto",
    *,
    context: str = "",
) -> str | None:
    """Select the conductivity channel for ``salinity="measured"``.

    ``"auto"`` returns ``JAC_C`` when present as a slow channel, else ``None``
    (no error — conductivity is optional; the salinity step falls back). An
    explicit name raises ValueError when missing or not a slow channel. No
    plausibility QC on conductivity in this pass (non-finite salinity samples
    are handled at the salinity step).
    """
    prefix = f"{context}: " if context else ""
    if requested == "auto":
        if "JAC_C" in channels and _len_is(channels["JAC_C"], n_slow):
            return "JAC_C"
        return None
    if requested not in channels:
        raise ValueError(f"{prefix}conductivity channel {requested!r} not found")
    if not _len_is(channels[requested], n_slow):
        raise ValueError(
            f"{prefix}conductivity channel {requested!r} is not a slow channel "
            f"(length {len(channels[requested])} != {n_slow})"
        )
    return requested


def resolve_measured_salinity(data: ChannelsDict | dict[str, Any]) -> np.ndarray | None:
    """Per-slow-sample practical salinity [PSU] for ``salinity="measured"``.

    Computes ``gsw.SP_from_C(C, T_pair, P)`` from the conductivity channel
    resolved by :func:`load_channels` (``data["C"]``). The pair temperature
    prefers the co-located ``JAC_T`` when the conductivity is ``JAC_C`` and
    JAC_T passes plausibility QC, falling back to the resolved reference
    temperature (with a warning); the pair used is recorded as
    ``metadata["salinity_pair_temperature"]``. Non-finite salinity samples
    are filled with the profile median (with a warning; perturb's
    ``_scrub_salinity`` policy). Returns ``None`` — fixed-S viscosity
    downstream — when there is no conductivity channel or the computed
    salinity is entirely non-finite.
    """
    metadata = data.get("metadata") or {}
    src = metadata.get("source", "")
    C = data.get("C")
    if C is None:
        warnings.warn(
            f"salinity='measured' but no conductivity channel was found in "
            f"{src}; falling back to fixed-salinity viscosity (visc35)",
            stacklevel=2,
        )
        return None
    C = np.asarray(C, dtype=np.float64)
    P = np.asarray(data["P"], dtype=np.float64)

    pair_T: np.ndarray | None = None
    pair_name = ""
    if metadata.get("conductivity_source") == "JAC_C":
        jac_t = data.get("JAC_T")
        if jac_t is not None:
            reason = reference_temperature_qc(jac_t, pressure=P)
            if reason is None:
                pair_T = np.asarray(jac_t, dtype=np.float64)
                pair_name = "JAC_T"
            else:
                warnings.warn(
                    f"measured salinity: JAC_T fails plausibility QC ({reason}); "
                    "pairing JAC_C with the reference temperature instead",
                    stacklevel=2,
                )
    if pair_T is None:
        pair_name = metadata.get("temperature_source", "T")
        pair_T = np.asarray(data["T"], dtype=np.float64)
        if metadata.get("conductivity_source") == "JAC_C" and "JAC_T" not in data:
            warnings.warn(
                "measured salinity: no JAC_T channel; pairing JAC_C with the "
                f"reference temperature ({pair_name})",
                stacklevel=2,
            )
    import gsw

    SP = np.asarray(gsw.SP_from_C(C, pair_T, P), dtype=np.float64)
    finite = np.isfinite(SP)
    if not finite.any():
        warnings.warn(
            f"measured salinity from {src} is entirely non-finite; falling "
            "back to fixed-salinity viscosity (visc35)",
            stacklevel=2,
        )
        return None
    metadata["salinity_pair_temperature"] = pair_name
    if not finite.all():
        fill = float(np.median(SP[finite]))
        warnings.warn(
            f"measured salinity has {int((~finite).sum())}/{SP.size} "
            f"non-finite sample(s); filling with the profile median "
            f"{fill:.3f} PSU",
            stacklevel=2,
        )
        SP = SP.copy()
        SP[~finite] = fill
    return SP


# ---------------------------------------------------------------------------
# Channel loading
# ---------------------------------------------------------------------------


def load_channels(
    source: PFile | str | Path,
    shear_pattern: str = r"^sh\d+$",
    accel_pattern: str = r"^A[xyz]$",
    pressure_name: str = "P",
    temperature_name: str | float = "auto",
    conductivity_name: str = "auto",
) -> ChannelsDict:
    """Load channel data from any supported source.

    Parameters
    ----------
    source : PFile, str, or Path
        A PFile object, .p file path, full-record .nc path,
        or per-profile .nc path.
    shear_pattern : str
        Regex pattern matching shear channel names.
    accel_pattern : str
        Regex pattern matching accelerometer channel names.
    pressure_name : str
        Name of the pressure channel.
    temperature_name : str or float
        Reference-temperature source: ``"auto"`` (first plausible of
        T1..Tn, T, JAC_T — see :func:`resolve_temperature_channel`), an
        explicit channel name (honored even when it fails plausibility QC,
        with a warning), or a number = constant reference temperature
        [degC] (ODAS ``constant_temp`` parity).
    conductivity_name : str
        ``"auto"`` (JAC_C when present; absent otherwise, no error) or an
        explicit channel name (missing/non-slow raises ValueError).

    Returns
    -------
    dict with keys:
        shear : list of (name, ndarray) — shear probe signals
        accel : list of (name, ndarray) — accelerometer signals
        channel_types : dict — sensor type per accel channel name (.p:
            pf.channel_info; nc: the sensor_type variable attr), so the
            L1 builder can label piezo-typed channels "VIB"
        P : ndarray — pressure (slow)
        T : ndarray — reference temperature (slow)
        C : ndarray — conductivity (slow; only when resolved)
        JAC_T : ndarray — CT temperature (slow; only when present, for
            the measured-salinity pairing)
        U_EM, Incl_X, Incl_Y : ndarray — slow channels consumed by the
            em/flight speed methods (only when present)
        t_fast : ndarray — fast time vector
        t_slow : ndarray — slow time vector
        fs_fast : float
        fs_slow : float
        is_profile : bool — whether source is a per-profile file
        metadata : dict — includes temperature_source / temperature_qc
            (and conductivity_source when conductivity was resolved)
    """
    from odas_tpw.rsi.p_file import PFile

    if isinstance(source, PFile):
        return _channels_from_pfile(
            source,
            shear_pattern,
            accel_pattern,
            pressure_name,
            temperature_name,
            conductivity_name,
        )

    source = Path(source)
    if source.suffix.lower() == ".p":
        pf = PFile(source)
        return _channels_from_pfile(
            pf,
            shear_pattern,
            accel_pattern,
            pressure_name,
            temperature_name,
            conductivity_name,
        )
    elif source.suffix.lower() == ".nc":
        return _channels_from_nc(
            source,
            shear_pattern,
            accel_pattern,
            pressure_name,
            temperature_name,
            conductivity_name,
        )
    else:
        raise ValueError(f"Unsupported file type: {source.suffix}")


def _resolve_reference_temperature(
    channels: Mapping[str, Any],
    n_slow: int,
    t_name: str | float,
    pressure: np.ndarray | None,
    context: str,
) -> tuple[np.ndarray, str, str]:
    """Shared T resolution: returns ``(T_array, source_label, qc_label)``.

    A numeric *t_name* is the constant-reference escape hatch (ODAS
    ``constant_temp`` parity — the vendor's own remedy for "measured
    temperature unreliable").
    """
    if isinstance(t_name, bool):
        raise ValueError(f"temperature={t_name!r} is not valid; use a channel name or a number")
    if isinstance(t_name, (int, float)):
        value = float(t_name)
        qc = "pass"
        if not np.isfinite(value) or not (TEMP_QC_MIN <= value <= TEMP_QC_MAX):
            qc = (
                f"constant {value:g} degC outside plausible ocean range "
                f"[{TEMP_QC_MIN:g}, {TEMP_QC_MAX:g}]"
            )
            warnings.warn(
                f"{context}: reference temperature {qc}; proceeding — "
                "viscosity/kappa_T will reflect this value",
                stacklevel=3,
            )
        return np.full(n_slow, value), f"constant:{value:g}", qc
    name, reason = resolve_temperature_channel(
        channels, n_slow, t_name, pressure=pressure, context=context
    )
    T = np.asarray(channels[name], dtype=np.float64)
    return T, name, "pass" if reason is None else reason


def _channels_from_pfile(
    pf: PFile,
    sh_pat: str,
    ac_pat: str,
    p_name: str,
    t_name: str | float,
    c_name: str = "auto",
) -> ChannelsDict:
    sh_re = re.compile(sh_pat) if sh_pat != SH_PATTERN.pattern else SH_PATTERN
    ac_re = re.compile(ac_pat) if ac_pat != AC_PATTERN.pattern else AC_PATTERN
    shear = sorted(
        [(n, pf.channels[n]) for n in pf._fast_channels if sh_re.match(n)],
        key=lambda x: x[0],
    )
    accel = sorted(
        [(n, pf.channels[n]) for n in pf._fast_channels if ac_re.match(n)],
        key=lambda x: x[0],
    )
    n_slow = len(pf.t_slow)
    P = pf.channels[p_name]
    T, t_source, t_qc = _resolve_reference_temperature(
        pf.channels, n_slow, t_name, P, pf.filepath.name
    )
    out: dict[str, Any] = {
        "shear": shear,
        "accel": accel,
        # Sensor types of the loaded vibration/accelerometer channels; lets
        # the L1 builder label piezo sensors "VIB" (see ChannelsDict).
        "channel_types": {
            n: str(pf.channel_info.get(n, {}).get("type", "")) for n, _ in accel
        },
        "P": P,
        "T": T,
        "t_fast": pf.t_fast,
        "t_slow": pf.t_slow,
        "fs_fast": pf.fs_fast,
        "fs_slow": pf.fs_slow,
        "is_profile": False,
        "vehicle": pf.config["instrument_info"].get("vehicle", "").lower(),
        "metadata": {
            "source": str(pf.filepath),
            "instrument": pf.config["instrument_info"].get("model", ""),
            "sn": pf.config["instrument_info"].get("sn", ""),
            "start_time": pf.start_time.isoformat(),
            "temperature_source": t_source,
            "temperature_qc": t_qc,
        },
    }
    c_found = resolve_conductivity_channel(pf.channels, n_slow, c_name, context=pf.filepath.name)
    if c_found is not None:
        out["C"] = pf.channels[c_found]
        out["metadata"]["conductivity_source"] = c_found
    if "JAC_T" in pf.channels and _len_is(pf.channels["JAC_T"], n_slow):
        out["JAC_T"] = pf.channels["JAC_T"]
    # Slow channels the speed methods (em/flight) need, when present.
    for name in ("U_EM", "Incl_X", "Incl_Y"):
        if name in pf.channels and _len_is(pf.channels[name], n_slow):
            out[name] = pf.channels[name]
    # Carry through the perturb-injected speed channel(s) if present.
    if "speed_fast" in pf.channels:
        out["speed_fast"] = pf.channels["speed_fast"]
    if "W_slow" in pf.channels:
        out["W_slow"] = pf.channels["W_slow"]
    return out


def _channels_from_nc(
    nc_path: Path,
    sh_pat: str,
    ac_pat: str,
    p_name: str,
    t_name: str | float,
    c_name: str = "auto",
) -> ChannelsDict:
    import netCDF4 as nc

    # Masked samples (_FillValue gaps) must become NaN, not the raw fill
    # buffer (~9.97e36): a single masked U_EM/Incl/P sample would otherwise
    # enter the speed estimation as a physical value and blow up epsilon
    # through the shear/speed**2 normalization. The NaN-safe interpolation
    # in the speed paths then repairs the gap. Applied to EVERY variable
    # this loader reads (no special-casing); a no-op for unmasked data.
    from odas_tpw.rsi.profile import _nc_filled

    ds = nc.Dataset(str(nc_path), "r")
    sh_re = re.compile(sh_pat) if sh_pat != SH_PATTERN.pattern else SH_PATTERN
    ac_re = re.compile(ac_pat) if ac_pat != AC_PATTERN.pattern else AC_PATTERN

    fs_fast = float(ds.fs_fast)
    fs_slow = float(ds.fs_slow)
    is_profile = hasattr(ds, "profile_number")

    t_fast = _nc_filled(ds.variables["t_fast"])
    t_slow = _nc_filled(ds.variables["t_slow"])
    n_slow = len(t_slow)
    P = _nc_filled(ds.variables[p_name])

    # Temperature/conductivity candidates: slow variables matching the
    # reference-T pattern plus the JAC CT pair, and any explicitly requested
    # name (read regardless of dims so a fast channel gets the clear
    # "not a slow channel" error from the resolver rather than "not found").
    resolve_map: dict[str, np.ndarray] = {}
    for vname in sorted(ds.variables.keys()):
        var = ds.variables[vname]
        wanted = (
            var.dimensions == ("time_slow",)
            or REF_T_PATTERN.match(vname)
            or vname in ("JAC_T", "JAC_C")
            or vname in (t_name, c_name)
        )
        if wanted and var.dimensions in (("time_slow",), ("time_fast",)):
            resolve_map[vname] = _nc_filled(var)

    shear = []
    accel = []
    channel_types: dict[str, str] = {}
    for vname in sorted(ds.variables.keys()):
        var = ds.variables[vname]
        if var.dimensions == ("time_fast",):
            if sh_re.match(vname):
                shear.append((vname, _nc_filled(var)))
            elif ac_re.match(vname):
                # Union of PR #137 (masked -> NaN) and PR #138 (type label).
                accel.append((vname, _nc_filled(var)))
                # sensor_type attr written by extract_profiles; lets the L1
                # builder label piezo sensors "VIB" (see ChannelsDict).
                channel_types[vname] = str(getattr(var, "sensor_type", ""))

    metadata = {"source": str(nc_path)}
    # speed_method / speed_source: per-profile speed provenance written by the
    # perturb pipeline (extract_profiles extra_attrs); prepare_profiles uses
    # them to enrich the precomputed-speed_fast provenance it stamps.
    for attr in (
        "instrument_model",
        "instrument_sn",
        "source_file",
        "start_time",
        "speed_method",
        "speed_source",
    ):
        if hasattr(ds, attr):
            metadata[attr] = getattr(ds, attr)

    from odas_tpw.rsi.vehicle import vehicle_from_nc_attrs

    vehicle = vehicle_from_nc_attrs(
        {
            k: getattr(ds, k)
            for k in ("vehicle", "platform_type", "instrument_model")
            if hasattr(ds, k)
        }
    )

    speed_fast = None
    W_slow = None
    if "speed_fast" in ds.variables:
        speed_fast = _nc_filled(ds.variables["speed_fast"])
    if "W_slow" in ds.variables:
        W_slow = _nc_filled(ds.variables["W_slow"])

    ds.close()

    # Resolve AFTER the dataset is closed (everything needed is in numpy by
    # now), so a QC ValueError cannot leak an open netCDF handle.
    T, t_source, t_qc = _resolve_reference_temperature(
        resolve_map, n_slow, t_name, P, nc_path.name
    )
    metadata["temperature_source"] = t_source
    metadata["temperature_qc"] = t_qc
    c_found = resolve_conductivity_channel(resolve_map, n_slow, c_name, context=nc_path.name)
    if c_found is not None:
        metadata["conductivity_source"] = c_found

    out: dict[str, Any] = {
        "shear": shear,
        "accel": accel,
        "channel_types": channel_types,
        "P": P,
        "T": T,
        "t_fast": t_fast,
        "t_slow": t_slow,
        "fs_fast": fs_fast,
        "fs_slow": fs_slow,
        "is_profile": is_profile,
        "vehicle": vehicle,
        "metadata": metadata,
    }
    if c_found is not None:
        out["C"] = resolve_map[c_found]
    if "JAC_T" in resolve_map and _len_is(resolve_map["JAC_T"], n_slow):
        out["JAC_T"] = resolve_map["JAC_T"]
    # Slow channels the speed methods (em/flight) need, when present.
    for name in ("U_EM", "Incl_X", "Incl_Y"):
        if name in resolve_map and _len_is(resolve_map[name], n_slow):
            out[name] = resolve_map[name]
    if speed_fast is not None:
        out["speed_fast"] = speed_fast
    if W_slow is not None:
        out["W_slow"] = W_slow
    return out


# ---------------------------------------------------------------------------
# Profile preparation
# ---------------------------------------------------------------------------


def prepare_profiles(
    data: ChannelsDict | dict[str, Any],
    speed: float | None,
    direction: str,
    salinity: npt.ArrayLike | None,
    tau: float | None = None,
    speed_cutout: float = 0.05,
    vehicle: str | None = None,
    W_min: float | None = None,
    speed_method: str | None = None,
    aoa_deg: float = 3.0,
) -> tuple | None:
    """Profile detection, speed computation, and salinity interpolation.

    Matches the ODAS odas_p2mat.m speed pipeline:
      1. W = gradient(P) filtered with Butterworth at 0.68/tau
      2. speed = abs(W)
      3. speed filtered again with Butterworth at 0.68/tau
      4. speed clamped to speed_cutout minimum

    ``tau=None`` (default) resolves the smoothing time constant from the
    vehicle table; pass an explicit value to override it.  ``W_min=None``
    (default) resolves the detection fall-rate floor from the resolved
    direction (0.3 dbar/s free-fall, 0.05 glide/horizontal); an explicit
    value passes through unchanged.

    Speed precedence (this function is the single owner): fixed ``speed`` >
    selected ``speed_method`` (``"em"``/``"flight"``, computed here via the
    shared speed module) > a precomputed ``speed_fast`` channel (perturb
    per-profile NetCDFs) > the historical |dP/dt| pressure path. An
    EXPLICIT ``speed_method="pressure"`` forces the pressure path even when
    a precomputed channel exists; ``None`` prefers the precomputed channel.
    A fixed ``speed`` together with an em/flight ``speed_method`` raises.

    Side effect: stamps ``data["metadata"]["speed_source"]`` (and keeps
    ``speed_method`` truthful) with the provenance of the speed actually
    used; the diss/chi builders merge the metadata into the product attrs.

    Returns (profiles_slow, speed_fast, P_fast, T_fast, sal_fast, fs_fast,
    fs_slow, ratio, t_fast).
    """
    from odas_tpw.rsi.vehicle import resolve_detection

    _validate_speed_selection(speed, speed_method)

    fs_fast = data["fs_fast"]
    fs_slow = data["fs_slow"]
    ratio = round(fs_fast / fs_slow)

    P_slow = data["P"]
    T_slow = data["T"]
    t_fast = data["t_fast"]
    t_slow = data["t_slow"]

    # Resolve vehicle, direction, tau, and the detection fall-rate floor
    if vehicle is None:
        vehicle = data.get("vehicle", "")
    direction, tau, W_min = resolve_detection(direction, vehicle, W_min=W_min, tau=tau)

    if data["is_profile"]:
        profiles_slow = [(0, len(P_slow) - 1)]
    else:
        from odas_tpw.rsi.profile import _smooth_fall_rate, get_profiles
        from odas_tpw.scor160.profile import explain_no_profiles

        logger.info(
            "profile detection: direction=%s W_min=%g dbar/s (vehicle=%r)",
            direction,
            W_min,
            vehicle,
        )
        W_slow = _smooth_fall_rate(P_slow, fs_slow, tau=tau)
        profiles_slow = get_profiles(
            P_slow, W_slow, fs_slow, W_min=W_min, direction=direction
        )
        if not profiles_slow:
            warnings.warn(
                explain_no_profiles(P_slow, W_slow, W_min=W_min, direction=direction),
                stacklevel=2,
            )
            return None

    metadata = data.get("metadata") if hasattr(data, "get") else None

    # Speed precedence point (see docstring); provenance is stamped here.
    # A stale upstream ``speed_method`` (copied from a perturb per-profile
    # NC) is cleared on the paths that do not use it, so the product attrs
    # can never pair e.g. speed_method="em" with a fixed-speed source.
    precomputed = data.get("speed_fast") if hasattr(data, "get") else None
    speed_method_out: str | None = None
    if speed is not None:
        speed_fast = np.full(len(t_fast), abs(speed))
        speed_source = f"fixed --speed {abs(speed):g}"
    elif speed_method is not None and speed_method != "pressure":
        speed_fast, speed_source = speed_from_method(data, speed_method, aoa_deg, vehicle)
        speed_method_out = speed_method
    elif speed_method is None and precomputed is not None and len(precomputed) == len(t_fast):
        speed_fast = np.asarray(precomputed, dtype=np.float64)
        upstream_method = (metadata or {}).get("speed_method")
        upstream_source = (metadata or {}).get("speed_source")
        speed_method_out = upstream_method
        if upstream_method:
            # Keep the upstream source string when it carries information
            # beyond the method name (e.g. "hotel:speed" names the hotel
            # channel, "constant:0.4" the value) so the diss/chi products
            # retain the full provenance, not just the method (#131 M10).
            if upstream_source and upstream_source != upstream_method:
                speed_source = (
                    f"precomputed speed_fast (perturb speed.method="
                    f"{upstream_method}, source={upstream_source})"
                )
            else:
                speed_source = f"precomputed speed_fast (perturb speed.method={upstream_method})"
        else:
            speed_source = "precomputed speed_fast channel"
    else:
        from odas_tpw.scor160.profile import compute_speed_fast

        if direction in ("glide", "horizontal") and speed_method is None:
            # ODAS uses a flight model (glide) or EM current meter /
            # hotel speed (horizontal) for these vehicles; |dP/dt|
            # underestimates the flow past the sensors, and epsilon has
            # roughly U^4 leverage on the speed through the shear
            # conversion and wavenumber transform. An EXPLICIT
            # speed_method="pressure" is the user's informed choice — no
            # warning then.
            fix = (
                "pass --speed-method em (U_EM present)"
                if "U_EM" in data
                else "provide an explicit speed or a precomputed speed_fast channel"
            )
            warnings.warn(
                f"Vehicle direction '{direction}' but speed is being computed "
                f"from |dP/dt|; {fix} for glider/horizontal platforms — "
                "epsilon scales as ~U^4 and will be strongly biased",
                stacklevel=2,
            )
        speed_fast, _W_slow = compute_speed_fast(
            P_slow,
            t_fast,
            t_slow,
            fs_fast,
            fs_slow,
            tau=tau,
            speed_min=speed_cutout,
        )
        speed_source = "pressure |dP/dt|"
    if metadata is not None:
        metadata["speed_source"] = speed_source
        if speed_method_out:
            metadata["speed_method"] = str(speed_method_out)
        else:
            metadata.pop("speed_method", None)

    # Floor (and NaN-scrub) speed once at this choke point so every consumer
    # gets a clean, positive speed: shear normalization (shear /= speed**2) and
    # the chi wavenumber / fp07 axis both blow up on a zero/NaN speed. The
    # fixed-speed and precomputed branches are otherwise unfloored;
    # compute_speed_fast already floors, so this is a no-op there.
    speed_fast = np.maximum(
        np.nan_to_num(np.asarray(speed_fast, dtype=np.float64), nan=speed_cutout),
        speed_cutout,
    )

    P_fast = np.interp(t_fast, t_slow, P_slow)
    T_fast = np.interp(t_fast, t_slow, T_slow)

    if salinity is not None:
        salinity = np.asarray(salinity, dtype=float)
        if salinity.ndim > 0:
            if len(salinity) == len(t_slow):
                sal_fast = np.interp(t_fast, t_slow, salinity)
            elif len(salinity) == len(t_fast):
                sal_fast = salinity
            else:
                raise ValueError(
                    f"salinity array length {len(salinity)} doesn't match "
                    f"slow ({len(t_slow)}) or fast ({len(t_fast)}) time series"
                )
        else:
            sal_fast = float(salinity)
    else:
        sal_fast = None

    return (profiles_slow, speed_fast, P_fast, T_fast, sal_fast, fs_fast, fs_slow, ratio, t_fast)


# ---------------------------------------------------------------------------
# Speed-method selection (em / flight) over a loaded channels dict
# ---------------------------------------------------------------------------

_SPEED_METHODS = (None, "pressure", "em", "flight")


def _validate_speed_selection(speed: float | None, speed_method: str | None) -> None:
    """Reject invalid ``speed``/``speed_method`` combinations up front.

    ``"constant"`` exists in ``compute_speed_for_pfile``'s vocabulary (the
    perturb ``speed:`` section carries its ``value``), but this layer has no
    value plumbing — the fixed ``speed`` parameter is the equivalent here.
    ``"hotel"`` is likewise perturb-only: hotel channels are merged into the
    instrument channels there, and this layer never sees them (load_channels
    carries only the named channels the em/flight methods need). A perturb
    per-profile NetCDF produced with ``speed.method: "hotel"`` arrives here
    as a precomputed ``speed_fast`` channel, which is preferred by default.
    """
    if speed_method not in _SPEED_METHODS:
        if speed_method == "constant":
            hint = " Use --speed for a fixed (constant) speed."
        elif speed_method == "hotel":
            hint = (
                " The hotel speed method is perturb-only (hotel channels are"
                " merged there): run the perturb pipeline with speed.method:"
                " 'hotel'; its per-profile NetCDFs carry the result as a"
                " precomputed speed_fast channel, which this layer uses"
                " automatically."
            )
        else:
            hint = ""
        raise ValueError(
            f"speed_method={speed_method!r} is not valid here; expected "
            f"pressure | em | flight.{hint}"
        )
    if speed is not None and speed_method is not None:
        raise ValueError(
            f"--speed (fixed speed {speed:g} m/s) and --speed-method "
            f"{speed_method!r} are mutually exclusive; pass one or the other"
        )


def speed_from_method(
    data: ChannelsDict | dict[str, Any],
    speed_method: str,
    aoa_deg: float = 3.0,
    vehicle: str | None = None,
) -> tuple[np.ndarray, str]:
    """Compute a through-water speed via :func:`compute_speed_for_pfile`.

    Builds a duck-typed shim over the loaded channels dict so the shared
    speed module (written against PFile) works for NetCDF-loaded sources
    too. Returns ``(speed_fast, speed_source)`` where ``speed_source`` is
    the human-readable provenance string for the product attrs (e.g.
    ``"em (U_EM)"``). Consumed by ``prepare_profiles``.
    """
    from types import SimpleNamespace

    from odas_tpw.rsi.speed import compute_speed_for_pfile

    channels: dict[str, Any] = {"P": data["P"]}
    for name in ("U_EM", "Incl_X", "Incl_Y"):
        if name in data:
            channels[name] = data[name]
    shim = SimpleNamespace(
        channels=channels,
        t_fast=data["t_fast"],
        t_slow=data["t_slow"],
        fs_fast=data["fs_fast"],
        fs_slow=data["fs_slow"],
    )
    if vehicle is None:
        vehicle = data.get("vehicle", "")
    speed_fast, _W_slow, source = compute_speed_for_pfile(
        shim, {"method": speed_method, "aoa_deg": aoa_deg}, vehicle
    )
    labels = {"em": "em (U_EM)", "flight": f"flight (aoa={aoa_deg:g})"}
    return speed_fast, labels.get(source, "pressure |dP/dt|")


# ---------------------------------------------------------------------------
# L1Data construction from load_channels output
# ---------------------------------------------------------------------------


def _build_l1data_from_channels(
    data: ChannelsDict | dict[str, Any],
    s_fast: int,
    e_fast: int,
    speed_fast: np.ndarray,
    P_fast: np.ndarray,
    T_fast: np.ndarray,
    direction: str = "down",
    *,
    therm_list: list[tuple[str, np.ndarray]] | None = None,
    diff_gains: list[float] | None = None,
) -> Any:
    """Build L1Data from load_channels output and interpolated profile arrays.

    Parameters
    ----------
    data : dict
        Output of load_channels().
    s_fast, e_fast : int
        Fast-rate slice indices for this profile.
    speed_fast : ndarray
        Full-length speed array (fast rate).
    P_fast, T_fast : ndarray
        Full-length pressure and temperature (interpolated to fast rate).
    direction : str
        Profile direction.
    therm_list : list of (name, array), optional
        Thermistor data for chi computation.
    diff_gains : list of float, optional
        Per-thermistor differentiator gains.
    """
    from odas_tpw.scor160.io import L1Data

    shear_arrays = [s[1] for s in data["shear"]]
    accel_arrays = [a[1] for a in data["accel"]]
    fs_fast = data["fs_fast"]
    n = e_fast - s_fast

    # Defensive floor (0.05 m/s cutout): callers normally pass a speed already
    # floored by prepare_profiles, but a zero/tiny speed here would blow up the
    # shear/speed**2 normalization. Matches adapter.pfile_to_l1data and speed.py.
    speed_prof = np.maximum(speed_fast[s_fast:e_fast], 0.05)

    # Shear: normalize by speed^2
    if shear_arrays:
        shear = np.stack(
            [arr[s_fast:e_fast] / speed_prof**2 for arr in shear_arrays],
            axis=0,
        )
    else:
        shear = np.zeros((0, n), dtype=np.float64)

    # Vibration/accelerometer
    if accel_arrays:
        vib = np.stack([arr[s_fast:e_fast] for arr in accel_arrays], axis=0)
        # Piezo-typed channels are uncalibrated vibration sensors, not linear
        # accelerometers: label them "VIB", mirroring the adapter's piezo ->
        # "VIB" convention (#131 W5-ii, W3 review F5). Types come from
        # load_channels (channel_types); dicts without them (older callers,
        # synthetic tests) keep the historical "ACC". Label-only: no numeric
        # consumer branches on vib_type (Goodman treats ACC and VIB alike).
        types = data.get("channel_types") or {}
        accel_names = [name for name, _ in data["accel"]]
        if accel_names and all(types.get(name) == "piezo" for name in accel_names):
            vib_type = "VIB"
        else:
            vib_type = "ACC"
    else:
        vib = np.zeros((0, n), dtype=np.float64)
        vib_type = "NONE"

    # Optional fast temperature
    if therm_list:
        tf = np.stack([arr[s_fast:e_fast] for _, arr in therm_list], axis=0)
    else:
        tf = np.zeros((0, 0), dtype=np.float64)

    return L1Data(
        time=data["t_fast"][s_fast:e_fast],
        pres=P_fast[s_fast:e_fast],
        shear=shear,
        vib=vib,
        vib_type=vib_type,
        fs_fast=fs_fast,
        f_AA=98.0,
        vehicle=data.get("vehicle", ""),
        profile_dir=direction,
        time_reference_year=2000,
        pspd_rel=speed_prof,
        temp=T_fast[s_fast:e_fast],
        temp_fast=tf,
        diff_gains=diff_gains or [],
    )


# ---------------------------------------------------------------------------
# Shared file-level output
# ---------------------------------------------------------------------------


def write_profile_results(
    results: list,
    source_path: Path,
    output_dir: Path,
    suffix: str,
) -> list[Path]:
    """Write per-profile xarray Datasets to NetCDF files.

    Parameters
    ----------
    results : list of xr.Dataset
        Per-profile results.
    source_path : Path
        Original source file (stem used for naming).
    output_dir : Path
        Output directory (created if needed).
    suffix : str
        File suffix, e.g. 'eps' or 'chi'.

    Returns
    -------
    list of Path
        Paths to output files written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []
    for i, ds in enumerate(results):
        if len(results) == 1:
            out_name = f"{source_path.stem}_{suffix}.nc"
        else:
            out_name = f"{source_path.stem}_prof{i + 1:03d}_{suffix}.nc"
        out_path = output_dir / out_name
        ds.to_netcdf(out_path)
        output_paths.append(out_path)
        logger.info(
            f"{out_path.name}: {ds.sizes['time']} estimates, "
            f"P={float(ds.P_mean.min()):.0f}-{float(ds.P_mean.max()):.0f} dbar"
        )
    return output_paths


# ---------------------------------------------------------------------------
# Shared dataset builder
# ---------------------------------------------------------------------------


def _build_result_dataset(
    variables: list[tuple[str, list[str], np.ndarray, dict]],
    probe_names: list[str],
    t_out: np.ndarray,
    probe_long_name: str,
    global_attrs: dict,
) -> xr.Dataset:
    """Build an xarray Dataset from a list of variable specs.

    Parameters
    ----------
    variables : list of (name, dims, data, attrs)
        Variable definitions.
    probe_names : list of str
        Probe/sensor coordinate labels.
    t_out : ndarray
        Time coordinate values.
    probe_long_name : str
        Long name for the probe coordinate.
    global_attrs : dict
        Dataset-level attributes.
    """
    import xarray as xr

    data_vars = {name: (dims, data, attrs) for name, dims, data, attrs in variables}
    ds = xr.Dataset(
        data_vars,
        coords={
            "probe": probe_names,
            "t": (["time"], t_out),
        },
        attrs=global_attrs,
    )
    ds.coords["probe"].attrs["long_name"] = probe_long_name
    return ds

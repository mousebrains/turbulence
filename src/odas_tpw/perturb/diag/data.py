# Jul-2026, Claude and Pat Welch, pat@mousebrains.com
"""Data access for ``perturb-diag``: the combo overview and per-cell drill-down.

The binned combo (``diss_combo_NN/combo.nc``) is scalar ``(bin, profile)`` and
drives the overview.  The per-cell spectra and per-profile diagnostics live in
the per-profile epsilon files (``diss_NN/*_prof*.nc``), which store the full
``spec_shear``/``spec_nasmyth``/``K`` arrays plus per-window scalars.  A clicked
overview cell ``(bin, profile)`` is mapped back to its per-profile file by
matching the combo's per-profile ``stime`` against each file's scalar ``stime``
(the same key the ``perturb-plot`` pseudo-var path uses), then to the window
whose depth is nearest the clicked bin.
"""

from __future__ import annotations

import glob
import os
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import xarray as xr


@dataclass
class OverviewData:
    """Scalar ``(bin, profile)`` fields for the pcolor overview panels.

    ``t`` is the per-profile start time (datetime64) for the cast-layout x-axis;
    ``stime_epoch`` is the same instant as float epoch seconds, the key used to
    locate each profile's per-profile file.  ``fields`` maps a variable name to
    its ``(bin, profile)`` array (already QC-masked when requested).
    """

    t: np.ndarray                 # (profile,) datetime64[ns]
    stime_epoch: np.ndarray       # (profile,) float epoch seconds
    bin: np.ndarray               # (bin,) depth centers, m (positive down)
    fields: dict[str, np.ndarray]  # name -> (bin, profile)


# HDF5 — and thus netCDF4 / xarray's default backend — is not thread-safe unless
# libhdf5 was built with --enable-threadsafe, which varies by wheel and platform.
# The inspector prewarms its stime indexes on a daemon thread (DiagInspector.
# _start_prewarm) while the main thread loads a clicked cell, so two file opens
# can overlap. Serialize every open+read in this module through one lock so that
# race cannot corrupt HDF5's global state ("NetCDF: HDF error", seen in CI on the
# non-threadsafe py3.12 wheel under `pytest -n auto`).
_NETCDF_LOCK = threading.Lock()


def load_overview(
    combo_path: str,
    field_names: tuple[str, ...],
    *,
    qc_var: str | None = "qc_drop_epsilon",
    apply_qc: bool = True,
) -> OverviewData:
    """Read *field_names* as ``(bin, profile)`` grids from a combo NetCDF.

    Opened with ``decode_times=False`` so ``stime`` is float epoch seconds
    (matching the per-profile files' scalar ``stime``); the datetime64 ``t`` for
    axis layout is derived from it.  When *apply_qc* and *qc_var* is present, its
    non-zero cells are NaN'd in every field (a single epsilon drop flag gates all
    epsilon variables together).
    """
    with _NETCDF_LOCK, xr.open_dataset(combo_path, decode_times=False) as ds:
        missing = [n for n in field_names if n not in ds.variables]
        if missing:
            raise SystemExit(
                f"{combo_path} is missing {missing}; is this a diss_combo?"
            )
        bin_depth = np.asarray(ds["bin"].values, dtype=float)
        stime_epoch = np.asarray(ds["stime"].values, dtype=float)
        fields = {
            n: np.asarray(ds[n].transpose("bin", "profile").values, dtype=float)
            for n in field_names
        }
        qc = None
        if apply_qc and qc_var is not None and qc_var in ds.variables:
            qc = np.asarray(
                ds[qc_var].transpose("bin", "profile").values, dtype=float
            )

    if qc is not None:
        mask = np.isfinite(qc) & (qc > 0)
        for n in fields:
            fields[n] = np.where(mask, np.nan, fields[n])

    # Float epoch seconds -> datetime64[ns] for the cast-layout time axis.
    t = (stime_epoch * 1e9).astype("int64").astype("datetime64[ns]")
    return OverviewData(t=t, stime_epoch=stime_epoch, bin=bin_depth, fields=fields)


def subset_profiles(ov: OverviewData, keep: np.ndarray) -> OverviewData:
    """Return *ov* restricted to the profiles selected by the boolean *keep*."""
    keep = np.asarray(keep, dtype=bool)
    return OverviewData(
        t=ov.t[keep],
        stime_epoch=ov.stime_epoch[keep],
        bin=ov.bin,
        fields={n: f[:, keep] for n, f in ov.fields.items()},
    )


def apply_sections(
    ov: OverviewData, sections_path: str | None, select: list[str] | None
) -> OverviewData:
    """Narrow *ov* to the casts within the selected section(s)' time window(s).

    Reuses the ``perturb-plot`` sections format (``sections.load_sections`` +
    ``select_sections``).  Only each section's UTC ``start``/``stop`` is used —
    the overview x-axis is always cast index, so a section's ``xaxis`` method is
    ignored.  The kept casts are the union over the selected sections (an open
    ``start``/``stop`` is unbounded on that side).  ``--select`` without
    ``--sections`` is an error, matching perturb-plot.
    """
    if not sections_path:
        if select:
            raise SystemExit("--select only applies together with --sections")
        return ov

    from odas_tpw.perturb.plot import sections as sec_mod

    secs = sec_mod.load_sections(sections_path)
    if select:
        secs = sec_mod.select_sections(secs, select)

    t = ov.t
    if t.size == 0:
        return ov
    tmin, tmax = t.min(), t.max()
    keep = np.zeros(t.shape, dtype=bool)
    for s in secs:
        lo = s.start if s.start is not None else tmin
        hi = s.stop if s.stop is not None else tmax
        keep |= (t >= lo) & (t <= hi)
    if not keep.any():
        raise SystemExit(
            "no casts fall within the selected section time window(s)"
        )
    return subset_profiles(ov, keep)


@dataclass
class ProfileFile:
    """Per-window arrays from one ``diss_NN/*_prof*.nc`` file.

    Spectra are ``(probe, freq, window)``; per-probe scalars are
    ``(probe, window)``; per-window scalars are ``(window,)``.  ``depth`` is
    ``P_mean`` converted to positive-down depth so it shares the overview's
    vertical coordinate.
    """

    path: str
    n_probe: int
    depth: np.ndarray          # (window,) m, positive down
    K: np.ndarray              # (freq, window) cpm
    spec_shear: np.ndarray     # (probe, freq, window)
    spec_nasmyth: np.ndarray   # (probe, freq, window)
    K_max: np.ndarray          # (probe, window) cpm
    method: np.ndarray         # (probe, window)
    FM: np.ndarray             # (probe, window)
    fom: np.ndarray            # (probe, window)
    epsilon: np.ndarray        # (probe, window) W/kg
    epsilon_mean: np.ndarray   # (window,)
    epsilon_ln_sigma: np.ndarray  # (window,)
    speed: np.ndarray          # (window,)
    nu: np.ndarray             # (window,) m^2/s
    T_mean: np.ndarray         # (window,) degC
    attrs: dict


def _p_to_depth(P: np.ndarray, lat: float) -> np.ndarray:
    """Positive-down depth from pressure via TEOS-10 (``-z_from_p``)."""
    import gsw

    lat_safe = 0.0 if not np.isfinite(lat) else float(lat)
    return -np.asarray(gsw.z_from_p(np.asarray(P, dtype=float), lat_safe))


# Raw diagnostic channels read from the per-profile *profiles* file for the
# drill-down's instrument-diagnostics row.  The microstructure channels (piezo
# shear / vibration / pre-emphasized temperature-gradient, ~512 Hz) get an
# HP+despike "processed" overlay; everything else is shown as recorded: speed,
# inclinometer, FP07 temperatures and fall rate.  Each is read on whichever
# clock its own dimension says (time_fast / time_slow), never assumed.
_HP_DIAG_CHANNELS = ("Ax", "Ay", "sh1", "sh2", "T1_dT1", "T2_dT2")
_DIAG_CHANNELS = (
    *_HP_DIAG_CHANNELS, "speed_fast", "Incl_X", "Incl_Y", "T1", "T2", "W_slow",
)

# The CT reference for the temperature, delta-T and conductivity panels, loaded
# under these role keys.  Its channel names depend on the platform, so each CT
# source is a (temperature, conductivity) pair, tried in order: the run's own
# ct.T_name/ct.C_name (see ct_reference_families), then these.
T_REF = "T_ref"
C_REF = "C_ref"
_CT_REF_DEFAULTS: tuple[tuple[str | None, str | None], ...] = (
    ("JAC_T", "JAC_C"),                    # JFE CT on a VMP or MicroRider
    ("sci_water_temp", "sci_water_cond"),  # a Slocum's CTD, via the hotel file
    ("SBT1", "SBC1"),                      # Sea-Bird CT (VMP-5500/6000 class)
)
# Never a CT reference, even when ct.T_name names one: delta-T measures the
# FP07s against the CT (T1 - T1 is no check; a CT-less MicroRider run may set
# ct.T_name: T1), and CT is TEOS-10 conservative temperature, derived from the
# in-situ reference rather than measured, which every profiles file carries.
_NOT_A_CT_REFERENCE = ("T1", "T2", "CT")

# Derived series: key -> (minuend, subtrahend), computed on the minuend's clock
# (the subtrahend interpolated onto it when the clocks differ).  FP07 minus CT,
# so an FP07 offset or lag against the CT shows at a glance.
_DERIVED_DIAG_SERIES: dict[str, tuple[str, str]] = {
    "dT1_ref": ("T1", T_REF),
    "dT2_ref": ("T2", T_REF),
}


def _usable_ct_name(name: object) -> str | None:
    """*name* if it can be a CT reference channel, else None."""
    if isinstance(name, str) and name and name not in _NOT_A_CT_REFERENCE:
        return name
    return None


def ct_reference_families(
    profiles_dir: str | None,
) -> tuple[tuple[str | None, str | None], ...]:
    """The CT sources to try, in order, as (temperature, conductivity) names.

    The profiles directory's ``.params_sha256_*`` signature records the
    ``ct.T_name``/``ct.C_name`` the run actually used (``sci_water_temp`` on the
    osu684 glider run, ``JAC_T`` on a VMP), so that pair comes first; the
    platform pairs follow, for an unsigned or unreadable directory or a name the
    file lacks.  A configured name in :data:`_NOT_A_CT_REFERENCE` is dropped
    (None), so a run with ``ct.T_name: T1`` still falls back to a JAC_T,
    sci_water_temp or SBT1 the file carries; the legend names whichever is used.
    """
    from pathlib import Path

    from odas_tpw.perturb.resolve import _read_signature

    try:
        sig = _read_signature(Path(profiles_dir)) if profiles_dir else None
    except OSError:  # e.g. an SMB listing error; the defaults still apply
        sig = None
    ct = (sig or {}).get("ct")
    ct = ct if isinstance(ct, dict) else {}
    configured = (_usable_ct_name(ct.get("T_name")), _usable_ct_name(ct.get("C_name")))
    pairs = (configured, *_CT_REF_DEFAULTS) if any(configured) else _CT_REF_DEFAULTS
    return tuple(dict.fromkeys(pairs))


@dataclass
class RawProfile:
    """Raw diagnostic channels for one profile, aligned to depth.

    ``raw`` holds each channel on its native clock.  For the microstructure
    channels, ``hp`` is the high-pass-filtered signal (spikes intact) and
    ``proc`` is that same signal after despiking — the epsilon path's cleanup.
    A drill-down overlays ``hp`` (faint) under ``proc`` (colored): both are
    zero-mean and in-band, so the panel scales to the microstructure signal (not
    a channel's DC offset / temperature ramp), and the gap between them is
    exactly what despike removed.  The other channels carry only ``raw``.  The
    CT reference is stored under the role keys :data:`T_REF`/:data:`C_REF`, with
    ``names`` giving the channel each one is, and the derived series (FP07 minus
    CT) under their own keys.  ``is_fast`` records each series' clock so a
    renderer can pick ``depth_fast`` vs ``depth_slow``.  All dicts are empty
    when the file lacks the pressure/time coordinates needed to place the
    signals in depth (the drill-down then shows "no raw data").
    """

    path: str
    depth_fast: np.ndarray     # (time_fast,) m, positive down
    depth_slow: np.ndarray     # (time_slow,) m, positive down
    raw: dict[str, np.ndarray]   # name -> signal on its native clock
    hp: dict[str, np.ndarray]    # name -> high-pass filtered (HP channels only)
    proc: dict[str, np.ndarray]  # name -> HP + despiked (HP channels only)
    is_fast: dict[str, bool]     # name -> True (fast clock) / False (slow)
    names: dict[str, str] = field(default_factory=dict)  # role -> channel name


def load_raw_profile(
    path: str,
    *,
    ct_families: tuple[tuple[str | None, str | None], ...] | None = None,
    hp_cut: float = 1.0,
    despike_thresh: float = 8.0,
    despike_smooth: float = 0.5,
) -> RawProfile:
    """Load the raw diagnostic channels and the CT reference from *path*.

    Each channel is placed on the clock its own dimension names, so a T1
    written at the fast rate is drawn on the fast depth, not dropped.  The
    microstructure channels on the fast clock are HP-filtered then despiked
    (``scor160`` helpers — the same preprocessing the ``perturb-plot`` variance
    diagnostics use) to give the ``proc`` overlay.  The CT temperature is the
    first of *ct_families* (default: the platform pairs) whose temperature the
    file holds; the conductivity comes from that same CT when the file has it,
    else from the first other pair that does.  Missing channels are skipped; a
    file without ``P``/``t_fast``/``t_slow`` (or one that fails to open) yields
    an empty :class:`RawProfile` rather than raising, so one bad file cannot
    break the inspector.
    """
    empty = RawProfile(path, np.array([]), np.array([]), {}, {}, {}, {})
    families = ct_families if ct_families is not None else _CT_REF_DEFAULTS
    try:
        with _NETCDF_LOCK, xr.open_dataset(path, decode_times=False) as ds:
            if not {"P", "t_fast", "t_slow"}.issubset(ds.variables):
                return empty
            lat = float(ds["lat"].values) if "lat" in ds.variables else np.nan
            t_fast = np.asarray(ds["t_fast"].values, dtype=float)
            t_slow = np.asarray(ds["t_slow"].values, dtype=float)
            P = np.asarray(ds["P"].values, dtype=float)
            fast_dims, slow_dims = ds["t_fast"].dims, ds["t_slow"].dims

            def series(ch: str | None) -> tuple[np.ndarray, bool] | None:
                """*ch* and whether it is on the fast clock, or None if absent.

                The clock is the variable's own dimension; only a variable on
                neither clock's dimension falls back to matching a length.
                """
                if ch is None or ch not in ds.variables:
                    return None
                var = ds[ch]
                sig = np.asarray(var.values, dtype=float).ravel()
                if var.dims == fast_dims:
                    fast = True
                elif var.dims == slow_dims:
                    fast = False
                elif sig.size in (t_fast.size, t_slow.size):
                    fast = sig.size == t_fast.size
                else:
                    return None
                n = t_fast.size if fast else t_slow.size
                return (sig, fast) if sig.size == n else None

            raw: dict[str, np.ndarray] = {}
            is_fast: dict[str, bool] = {}
            names: dict[str, str] = {}
            for ch in _DIAG_CHANNELS:
                got = series(ch)
                if got is not None:
                    raw[ch], is_fast[ch] = got

            def take(role: str, ch: str | None) -> bool:
                """Load *ch* as the CT *role*; False when the file lacks it."""
                got = series(ch)
                if got is None or ch is None:
                    return False
                raw[role], is_fast[role] = got
                names[role] = ch
                return True

            chosen = None
            for i, (t, _c) in enumerate(families):
                if take(T_REF, t):
                    chosen = i
                    break
            # Conductivity from the same CT as the temperature where possible.
            c_order = list(families)
            if chosen is not None:
                c_order.insert(0, c_order.pop(chosen))
            for _t, c in c_order:
                if take(C_REF, c):
                    break
    except (OSError, ValueError, RuntimeError):
        return empty

    for key, (a, b) in _DERIVED_DIAG_SERIES.items():
        if a in raw and b in raw:
            yb = raw[b]
            if is_fast[a] != is_fast[b]:
                t_a, t_b = (t_fast, t_slow) if is_fast[a] else (t_slow, t_fast)
                yb = np.interp(t_a, t_b, yb)
            raw[key] = raw[a] - yb
            is_fast[key] = is_fast[a]

    depth_slow = _p_to_depth(P, lat)
    depth_fast = _p_to_depth(np.interp(t_fast, t_slow, P), lat)

    hp: dict[str, np.ndarray] = {}
    proc: dict[str, np.ndarray] = {}
    dt = float(np.median(np.diff(t_fast))) if t_fast.size > 1 else np.nan
    if np.isfinite(dt) and dt > 0:
        from odas_tpw.scor160.despike import despike
        from odas_tpw.scor160.l2 import _hp_filter

        fs = 1.0 / dt
        for ch in _HP_DIAG_CHANNELS:
            sig = raw.get(ch)
            if sig is not None and is_fast[ch] and sig.size >= 16:
                hp_sig = _hp_filter(sig, fs, hp_cut)
                hp[ch] = hp_sig
                proc[ch] = despike(
                    hp_sig, fs, thresh=despike_thresh, smooth=despike_smooth,
                ).y
    return RawProfile(path, depth_fast, depth_slow, raw, hp, proc, is_fast, names)


def load_profile_file(path: str) -> ProfileFile:
    """Load the arrays a diagnostic drill-down needs from one diss file."""
    with _NETCDF_LOCK, xr.open_dataset(path, decode_times=False) as ds:
        lat = float(ds["lat"].values) if "lat" in ds.variables else np.nan
        P_mean = np.asarray(ds["P_mean"].values, dtype=float)

        def arr(name: str) -> np.ndarray:
            return np.asarray(ds[name].values, dtype=float)

        spec_shear = arr("spec_shear")
        return ProfileFile(
            path=path,
            n_probe=int(ds.sizes["probe"]),
            depth=_p_to_depth(P_mean, lat),
            K=arr("K"),
            spec_shear=spec_shear,
            spec_nasmyth=arr("spec_nasmyth"),
            K_max=arr("K_max"),
            method=arr("method"),
            FM=arr("FM"),
            fom=arr("fom"),
            epsilon=arr("epsilon"),
            epsilon_mean=arr("epsilonMean"),
            epsilon_ln_sigma=arr("epsilonLnSigma"),
            speed=arr("speed"),
            nu=arr("nu"),
            T_mean=arr("T_mean"),
            attrs=dict(ds.attrs),
        )


@dataclass
class ChiProfileFile:
    """Per-window arrays from one ``chi_NN/*_prof*.nc`` file (the chi drill-down).

    The chi analog of :class:`ProfileFile`: temperature-gradient spectra
    (``spec_gradT`` observed, ``spec_batch`` Batchelor/Kraichnan model,
    ``spec_noise`` floor) with the Batchelor wavenumber ``kB`` and upper limit
    ``K_max_T``.  ``depth`` is ``P_mean`` as positive-down depth, sharing the
    overview coordinate.
    """

    path: str
    n_probe: int
    depth: np.ndarray          # (window,) m, positive down
    K: np.ndarray              # (freq, window) cpm
    spec_gradT: np.ndarray     # (probe, freq, window) observed
    spec_batch: np.ndarray     # (probe, freq, window) Batchelor/Kraichnan model
    spec_noise: np.ndarray     # (probe, freq, window) noise floor
    kB: np.ndarray             # (probe, window) Batchelor wavenumber, cpm
    K_max_T: np.ndarray        # (probe, window) cpm
    chi: np.ndarray            # (probe, window) K^2/s
    epsilon_T: np.ndarray      # (probe, window) W/kg (from chi fit)
    fom: np.ndarray            # (probe, window)
    speed: np.ndarray          # (window,)
    nu: np.ndarray             # (window,) m^2/s
    T_mean: np.ndarray         # (window,) degC
    attrs: dict
    # Mixing quantities — present only when a mixing run (CTD) appended them to
    # the chi file (``pipeline._add_mixing_quantities``); ``None`` otherwise.
    # All (window,), on the chi grid.  Used by the mixing drill-down strip.
    N2: np.ndarray | None = None       # buoyancy frequency squared, s^-2
    dTdz: np.ndarray | None = None     # temperature gradient, K m^-1
    K_T: np.ndarray | None = None      # thermal diffusivity (Osborn-Cox), m^2/s
    K_rho: np.ndarray | None = None    # diapycnal diffusivity (Osborn), m^2/s
    Gamma: np.ndarray | None = None    # mixing coefficient (Oakey), dimensionless
    # The FP07 response the chi fit actually used, beyond the model: probe
    # channel names (to read the per-probe fp07_tau_scale_<name> attrs) and an
    # applied fp07-pair correction (f at W_ref, ln e per probe).
    probes: tuple[str, ...] = ()
    response_f: np.ndarray | None = None       # (n_f,) Hz at fp07_response_W_ref
    response_ln_e: np.ndarray | None = None    # (probe, n_f)


def load_chi_profile_file(path: str) -> ChiProfileFile:
    """Load the arrays the chi drill-down needs from one ``chi_NN`` file.

    The mixing quantities (N2/dTdz/K_T/K_rho/Gamma) are read when present — a
    mixing run with CTD appends them to the chi file — and left ``None``
    otherwise, so this same loader backs both the chi and mixing inspectors.
    """
    with _NETCDF_LOCK, xr.open_dataset(path, decode_times=False) as ds:
        lat = float(ds["lat"].values) if "lat" in ds.variables else np.nan
        P_mean = np.asarray(ds["P_mean"].values, dtype=float)

        def arr(name: str) -> np.ndarray:
            return np.asarray(ds[name].values, dtype=float)

        def opt(name: str) -> np.ndarray | None:
            return arr(name) if name in ds.variables else None

        return ChiProfileFile(
            path=path,
            n_probe=int(ds.sizes["probe"]),
            depth=_p_to_depth(P_mean, lat),
            K=arr("K"),
            spec_gradT=arr("spec_gradT"),
            spec_batch=arr("spec_batch"),
            spec_noise=arr("spec_noise"),
            kB=arr("kB"),
            K_max_T=arr("K_max_T"),
            chi=arr("chi"),
            epsilon_T=arr("epsilon_T"),
            fom=arr("fom"),
            speed=arr("speed"),
            nu=arr("nu"),
            T_mean=arr("T_mean"),
            attrs=dict(ds.attrs),
            N2=opt("N2"),
            dTdz=opt("dTdz"),
            K_T=opt("K_T"),
            K_rho=opt("K_rho"),
            Gamma=opt("Gamma"),
            probes=_probe_names(ds),
            response_f=opt("fp07_response_f"),
            response_ln_e=opt("fp07_response_ln_e"),
        )


def _probe_names(ds: xr.Dataset) -> tuple[str, ...]:
    if "probe" not in ds.variables:
        return ()
    vals = ds["probe"].values
    return tuple(v.decode() if isinstance(v, bytes) else str(v) for v in vals)


@dataclass
class Cell:
    """A resolved overview cell: a per-profile file and one window index.

    ``profile`` is the drill-down file for the active product — a
    :class:`ProfileFile` (epsilon), :class:`ChiProfileFile` (chi), or, for
    mixing, whichever the clicked variable calls for (see ``profiles``).
    ``profiles`` optionally holds several such files by key (``"eps"``/``"chi"``)
    when one cell draws on more than one product.  ``raw`` carries the matching
    profile's raw diagnostic channels when the source was given a
    ``profiles_dir``; it is ``None`` otherwise (or when no profiles file matched
    this cast).  ``field`` is the overview variable the selection came from, so a
    renderer can adapt (the mixing drill-down keys its spectra off it).

    A *no-product* cell has ``profile=None``: the cast has no usable file of the
    inspector's primary product, which is when its raw channels matter most.
    ``note`` then says why (e.g. "no diss_00 file for this cast") and ``depth``
    is the clicked bin depth, which stands in for the missing window's depth
    band.  For mixing, ``profiles["eps"]`` may still hold the cast's diss file.
    """

    profile: Any
    window: int
    raw: RawProfile | None = None
    profiles: dict[str, Any] | None = None
    field: str | None = None
    depth: float | None = None
    note: str | None = None


def _no_product_cell(
    raw: RawProfile | None, depth: float, note: str, *, eps: Any = None
) -> Cell | None:
    """A :class:`Cell` without its primary product file, or None with nothing.

    It carries the cast's raw channels and, for mixing, its diss file *eps*;
    None only when neither matched.  A profiles file that matched but holds no
    usable channels still makes a cell: its panels then say "no raw data",
    which beats a bare placeholder.
    """
    if raw is None and eps is None:
        return None
    profiles = {"eps": eps, "chi": None} if eps is not None else None
    return Cell(
        profile=None, window=-1, raw=raw, profiles=profiles,
        depth=float(depth), note=note,
    )


def _read_stime(path: str) -> float | None:
    """Read one file's scalar ``stime`` cheaply (netCDF4 single var, xr fallback).

    ``netCDF4.Dataset`` reads a single scalar ~1.6x faster than a full
    ``xarray.open_dataset`` (measured over the ARCTERX per-profile files); the
    xarray path is a fallback for environments without the netCDF4 module.
    """
    try:
        import netCDF4
    except ImportError:
        netCDF4 = None  # type: ignore[assignment]
    if netCDF4 is not None:
        try:
            with _NETCDF_LOCK, netCDF4.Dataset(path) as ds:
                if "stime" in ds.variables:
                    return float(ds.variables["stime"][...])
            return None
        except (OSError, RuntimeError, ValueError):
            return None
    try:
        with _NETCDF_LOCK, xr.open_dataset(path, decode_times=False) as ds:
            if "stime" in ds.variables:
                return float(ds["stime"].values)
    except (OSError, ValueError, RuntimeError):
        return None
    return None


def _scan_stime_map(directory: str) -> dict[float, str]:
    """``{round(stime, 3) -> path}`` over ``directory/*prof*.nc`` (one open each)."""
    m: dict[float, str] = {}
    for p in sorted(glob.glob(os.path.join(directory, "*prof*.nc"))):
        st = _read_stime(p)
        if st is not None:
            m[round(st, 3)] = p
    return m


class _StimeIndex:
    """Lazily-built ``{stime -> path}`` index over a per-profile directory.

    The scan (one open per file to read the scalar ``stime``) is deferred to the
    first lookup, so the inspector's overview — which needs only the combo —
    opens without paying for it.  ``perturb-diag``'s startup no longer scans the
    hundreds of per-profile files up front.

    The build is guarded by a lock so it is safe to call from a background
    thread: the inspector prewarms the index while the user reads the overview
    (see :meth:`EpsilonCellSource.prewarm`), and a click that arrives mid-scan
    simply waits on the same lock rather than racing a second scan.
    """

    def __init__(self, directory: str, tol: float) -> None:
        self._dir = directory
        self._tol = float(tol)
        self._keys: np.ndarray | None = None
        self._smap: dict[float, str] = {}
        self._lock = threading.Lock()

    def ensure(self) -> None:
        """Build the index if it isn't built yet (thread-safe, idempotent)."""
        if self._keys is not None:  # fast path: already built, no lock
            return
        with self._lock:
            if self._keys is None:
                smap = _scan_stime_map(self._dir)
                self._smap = smap
                # Publish _keys last: a reader that sees it non-None (via the
                # lock-free fast path) is then guaranteed a fully-built _smap.
                self._keys = np.array(sorted(smap), dtype=float)

    def path_for(self, stime_epoch: float) -> str | None:
        self.ensure()
        keys = self._keys
        if keys is None or not keys.size or not np.isfinite(stime_epoch):
            return None
        i = int(np.argmin(np.abs(keys - stime_epoch)))
        if abs(keys[i] - stime_epoch) <= self._tol:
            return self._smap[float(keys[i])]
        return None


def _stage_name(directory: str) -> str:
    """``diss_00`` from ``.../diss_00/`` — names the missing product in a note."""
    return os.path.basename(os.path.normpath(directory))


class _RawLookup:
    """The raw-profiles half of a cell source: ``stime`` index, cache, CT names.

    Shared by :class:`EpsilonCellSource` and :class:`MixingCellSource`.  The CT
    reference names are read from the profiles directory's signature on the
    first load (not at construction, which must not touch the directory).
    """

    _raw: _StimeIndex | None

    def _init_raw(self, profiles_dir: str | None, tol: float) -> None:
        self._raw = _StimeIndex(profiles_dir, tol) if profiles_dir else None
        self._raw_dir = profiles_dir
        self._raw_cache: dict[str, RawProfile] = {}
        self._ct_families: tuple[tuple[str | None, str | None], ...] | None = None

    def load_raw(self, path: str) -> RawProfile:
        rp = self._raw_cache.get(path)
        if rp is None:
            if self._ct_families is None:
                self._ct_families = ct_reference_families(self._raw_dir)
            rp = load_raw_profile(path, ct_families=self._ct_families)
            self._raw_cache[path] = rp
        return rp

    def raw_for(self, stime_epoch: float) -> RawProfile | None:
        """The cast's raw channels, or None (no profiles dir, or no match)."""
        if self._raw is None:
            return None
        rpath = self._raw.path_for(stime_epoch)
        return None if rpath is None else self.load_raw(rpath)


class EpsilonCellSource(_RawLookup):
    """Map overview ``(stime, bin)`` cells to a per-profile file + window.

    Files are located by matching the combo's per-profile ``stime`` (float epoch
    seconds) against each per-profile file's scalar ``stime`` within *tol*
    seconds, and cached once loaded.  The window is the one whose depth is
    nearest the clicked bin.  When *profiles_dir* is given, the same ``stime``
    match against the per-profile *profiles* files supplies each cell's raw
    diagnostic channels (CT temperature / conductivity, fall rate,
    inclinometer, accel, shear, temp-gradient) — and a raw-only cell when the
    cast has no usable per-profile product file.

    Both directory indexes are built lazily (see :class:`_StimeIndex`): opening
    the inspector no longer scans the per-profile directories; the first
    drill-down does.
    """

    def __init__(
        self, diss_dir: str, *, profiles_dir: str | None = None, tol: float = 1.0,
        loader: Callable[[str], Any] = load_profile_file,
    ) -> None:
        self._tol = float(tol)
        self._diss = _StimeIndex(diss_dir, self._tol)
        self._stage = _stage_name(diss_dir)
        self._init_raw(profiles_dir, self._tol)
        self._loader = loader
        self._cache: dict[str, Any] = {}

    def load(self, path: str) -> Any:
        pf = self._cache.get(path)
        if pf is None:
            pf = self._loader(path)
            self._cache[path] = pf
        return pf

    def prewarm(self) -> None:
        """Build the directory ``stime`` indexes now.

        Intended to run on a background thread once the overview is on screen
        (see :meth:`DiagInspector.show`), so the scan happens during the user's
        read time and the first drill-down click only pays the per-cell load.
        Best-effort and idempotent; the lazy path still works if it is skipped.
        """
        self._diss.ensure()
        if self._raw is not None:
            self._raw.ensure()

    def cell(self, stime_epoch: float, bin_depth: float) -> Cell | None:
        """Resolve one overview cell.

        Without a matching per-profile file (or one with no finite window
        depth) the cell is raw-only when the cast's profiles file matched, and
        None when it did not either.
        """
        path = self._diss.path_for(stime_epoch)
        if path is None:
            return _no_product_cell(
                self.raw_for(stime_epoch), bin_depth,
                f"no {self._stage} file for this cast",
            )
        pf = self.load(path)
        # Nearest window by depth; ignore NaN depths (failed windows).
        window = nearest_window(pf.depth, bin_depth)
        if window is None:
            return _no_product_cell(
                self.raw_for(stime_epoch), bin_depth,
                f"no finite window in {os.path.basename(path)}",
            )
        return Cell(
            profile=pf, window=window, raw=self.raw_for(stime_epoch),
            depth=float(bin_depth),
        )


def nearest_window(depth: np.ndarray, target: float) -> int | None:
    """Index of the finite ``depth`` entry nearest ``target``, or None if none."""
    d = np.asarray(depth, dtype=float).copy()
    finite = np.isfinite(d)
    if not finite.any():
        return None
    d[~finite] = np.inf
    return int(np.argmin(np.abs(d - target)))


class CellSource(Protocol):
    """Structural type for the inspector's cell provider (epsilon/chi/mixing)."""

    def cell(self, stime_epoch: float, bin_depth: float) -> Cell | None: ...
    def prewarm(self) -> None: ...


class MixingCellSource(_RawLookup):
    """Map overview ``(stime, bin)`` cells to a chi profile + a matching diss one.

    Mixing draws on two products: the chi file supplies the temperature-gradient
    spectra (K_T drill-down) and — from a mixing run — the window-scale
    stratification and coefficients (N2/dTdz/K_T/K_rho/Gamma) for the strip,
    while the diss file supplies the shear spectra (K_rho drill-down).  The chi
    file is primary: it fixes the depth grid and the selected window; the diss
    file is matched by ``stime`` and attached under ``cell.profiles['eps']`` so
    the adaptive spectra renderer can pick the window nearest the same depth.
    Without a usable chi file the cell is raw-only, as for
    :class:`EpsilonCellSource`.
    """

    def __init__(
        self, diss_dir: str, chi_dir: str, *,
        profiles_dir: str | None = None, tol: float = 1.0,
    ) -> None:
        self._tol = float(tol)
        self._chi = _StimeIndex(chi_dir, self._tol)
        self._chi_stage = _stage_name(chi_dir)
        self._diss = _StimeIndex(diss_dir, self._tol)
        self._init_raw(profiles_dir, self._tol)
        self._chi_cache: dict[str, ChiProfileFile] = {}
        self._diss_cache: dict[str, ProfileFile] = {}

    def load_chi(self, path: str) -> ChiProfileFile:
        cf = self._chi_cache.get(path)
        if cf is None:
            cf = load_chi_profile_file(path)
            self._chi_cache[path] = cf
        return cf

    def load_diss(self, path: str) -> ProfileFile:
        pf = self._diss_cache.get(path)
        if pf is None:
            pf = load_profile_file(path)
            self._diss_cache[path] = pf
        return pf

    def prewarm(self) -> None:
        """Build all directory ``stime`` indexes now (background-thread safe)."""
        self._chi.ensure()
        self._diss.ensure()
        if self._raw is not None:
            self._raw.ensure()

    def cell(self, stime_epoch: float, bin_depth: float) -> Cell | None:
        """Resolve a mixing cell (chi primary + diss + raw).

        Without a usable chi file the cell has no primary product but keeps the
        diss file (so a K_rho click still shows the shear spectrum) and the raw
        row; it is None only when neither of those matched either.
        """
        diss_pf = None
        diss_path = self._diss.path_for(stime_epoch)
        if diss_path is not None:
            diss_pf = self.load_diss(diss_path)

        chi_path = self._chi.path_for(stime_epoch)
        if chi_path is None:
            return _no_product_cell(
                self.raw_for(stime_epoch), bin_depth,
                f"no {self._chi_stage} file for this cast", eps=diss_pf,
            )
        chi_pf = self.load_chi(chi_path)
        window = nearest_window(chi_pf.depth, bin_depth)
        if window is None:
            return _no_product_cell(
                self.raw_for(stime_epoch), bin_depth,
                f"no finite window in {os.path.basename(chi_path)}", eps=diss_pf,
            )

        return Cell(
            profile=chi_pf, window=window, raw=self.raw_for(stime_epoch),
            profiles={"eps": diss_pf, "chi": chi_pf}, depth=float(bin_depth),
        )

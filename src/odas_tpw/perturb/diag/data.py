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
from dataclasses import dataclass
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
# drill-down's instrument-diagnostics row.  Fast channels (piezo shear /
# vibration / pre-emphasized temperature-gradient, ~512 Hz) get an HP+despike
# "processed" overlay; slow channels (inclinometer, ~64 Hz) are shown raw.
_FAST_DIAG_CHANNELS = ("Ax", "Ay", "sh1", "sh2", "T1_dT1", "T2_dT2")
_SLOW_DIAG_CHANNELS = ("Incl_X", "Incl_Y")


@dataclass
class RawProfile:
    """Raw diagnostic channels for one profile, aligned to depth.

    ``raw`` holds each channel on its native clock.  For the *fast* channels,
    ``hp`` is the high-pass-filtered signal (spikes intact) and ``proc`` is that
    same signal after despiking — the epsilon path's cleanup.  A drill-down
    overlays ``hp`` (faint) under ``proc`` (colored): both are zero-mean and
    in-band, so the panel scales to the microstructure signal (not a channel's
    DC offset / temperature ramp), and the gap between them is exactly what
    despike removed.  Slow channels (inclinometer) carry only ``raw``.
    ``is_fast`` records each channel's clock so a renderer can pick
    ``depth_fast`` vs ``depth_slow``.  All dicts are empty when the file lacks
    the pressure/time coordinates needed to place the signals in depth (the
    drill-down then shows "no raw data").
    """

    path: str
    depth_fast: np.ndarray     # (time_fast,) m, positive down
    depth_slow: np.ndarray     # (time_slow,) m, positive down
    raw: dict[str, np.ndarray]   # name -> signal on its native clock
    hp: dict[str, np.ndarray]    # name -> high-pass filtered (fast channels only)
    proc: dict[str, np.ndarray]  # name -> HP + despiked (fast channels only)
    is_fast: dict[str, bool]     # name -> True (fast clock) / False (slow)


def load_raw_profile(
    path: str,
    *,
    hp_cut: float = 1.0,
    despike_thresh: float = 8.0,
    despike_smooth: float = 0.5,
) -> RawProfile:
    """Load raw inclinometer/accel/shear/temp-gradient channels from *path*.

    Fast channels are HP-filtered then despiked (``scor160`` helpers — the same
    preprocessing the ``perturb-plot`` variance diagnostics use) to give the
    ``proc`` overlay.  Missing channels are skipped; a file without
    ``P``/``t_fast``/``t_slow`` (or one that fails to open) yields an empty
    :class:`RawProfile` rather than raising, so one bad file cannot break the
    inspector.
    """
    empty = RawProfile(path, np.array([]), np.array([]), {}, {}, {}, {})
    try:
        with _NETCDF_LOCK, xr.open_dataset(path, decode_times=False) as ds:
            if not {"P", "t_fast", "t_slow"}.issubset(ds.variables):
                return empty
            lat = float(ds["lat"].values) if "lat" in ds.variables else np.nan
            t_fast = np.asarray(ds["t_fast"].values, dtype=float)
            t_slow = np.asarray(ds["t_slow"].values, dtype=float)
            P = np.asarray(ds["P"].values, dtype=float)
            raw: dict[str, np.ndarray] = {}
            is_fast: dict[str, bool] = {}
            for ch in _FAST_DIAG_CHANNELS:
                if ch in ds.variables:
                    sig = np.asarray(ds[ch].values, dtype=float).ravel()
                    if sig.size == t_fast.size:
                        raw[ch] = sig
                        is_fast[ch] = True
            for ch in _SLOW_DIAG_CHANNELS:
                if ch in ds.variables:
                    sig = np.asarray(ds[ch].values, dtype=float).ravel()
                    if sig.size == t_slow.size:
                        raw[ch] = sig
                        is_fast[ch] = False
    except (OSError, ValueError, RuntimeError):
        return empty

    depth_slow = _p_to_depth(P, lat)
    depth_fast = _p_to_depth(np.interp(t_fast, t_slow, P), lat)

    hp: dict[str, np.ndarray] = {}
    proc: dict[str, np.ndarray] = {}
    dt = float(np.median(np.diff(t_fast))) if t_fast.size > 1 else np.nan
    if np.isfinite(dt) and dt > 0:
        from odas_tpw.scor160.despike import despike
        from odas_tpw.scor160.l2 import _hp_filter

        fs = 1.0 / dt
        for ch, sig in raw.items():
            if is_fast[ch] and sig.size >= 16:
                hp_sig = _hp_filter(sig, fs, hp_cut)
                hp[ch] = hp_sig
                proc[ch] = despike(
                    hp_sig, fs, thresh=despike_thresh, smooth=despike_smooth,
                ).y
    return RawProfile(path, depth_fast, depth_slow, raw, hp, proc, is_fast)


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
        )


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
    """

    profile: Any
    window: int
    raw: RawProfile | None = None
    profiles: dict[str, Any] | None = None
    field: str | None = None


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


class EpsilonCellSource:
    """Map overview ``(stime, bin)`` cells to a per-profile file + window.

    Files are located by matching the combo's per-profile ``stime`` (float epoch
    seconds) against each per-profile file's scalar ``stime`` within *tol*
    seconds, and cached once loaded.  The window is the one whose depth is
    nearest the clicked bin.  When *profiles_dir* is given, the same ``stime``
    match against the per-profile *profiles* files supplies each cell's raw
    diagnostic channels (inclinometer / accel / shear / temp-gradient).

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
        self._raw = _StimeIndex(profiles_dir, self._tol) if profiles_dir else None
        self._loader = loader
        self._cache: dict[str, Any] = {}
        self._raw_cache: dict[str, RawProfile] = {}

    def load(self, path: str) -> Any:
        pf = self._cache.get(path)
        if pf is None:
            pf = self._loader(path)
            self._cache[path] = pf
        return pf

    def load_raw(self, path: str) -> RawProfile:
        rp = self._raw_cache.get(path)
        if rp is None:
            rp = load_raw_profile(path)
            self._raw_cache[path] = rp
        return rp

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
        """Resolve one overview cell, or None if no per-profile file matches."""
        path = self._diss.path_for(stime_epoch)
        if path is None:
            return None
        pf = self.load(path)
        if pf.depth.size == 0:
            return None
        # Nearest window by depth; ignore NaN depths (failed windows).
        d = pf.depth.copy()
        finite = np.isfinite(d)
        if not finite.any():
            return None
        d[~finite] = np.inf
        window = int(np.argmin(np.abs(d - bin_depth)))
        raw = None
        if self._raw is not None:
            rpath = self._raw.path_for(stime_epoch)
            if rpath is not None:
                raw = self.load_raw(rpath)
        return Cell(profile=pf, window=window, raw=raw)


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


class MixingCellSource:
    """Map overview ``(stime, bin)`` cells to a chi profile + a matching diss one.

    Mixing draws on two products: the chi file supplies the temperature-gradient
    spectra (K_T drill-down) and — from a mixing run — the window-scale
    stratification and coefficients (N2/dTdz/K_T/K_rho/Gamma) for the strip,
    while the diss file supplies the shear spectra (K_rho drill-down).  The chi
    file is primary: it fixes the depth grid and the selected window; the diss
    file is matched by ``stime`` and attached under ``cell.profiles['eps']`` so
    the adaptive spectra renderer can pick the window nearest the same depth.
    """

    def __init__(
        self, diss_dir: str, chi_dir: str, *,
        profiles_dir: str | None = None, tol: float = 1.0,
    ) -> None:
        self._tol = float(tol)
        self._chi = _StimeIndex(chi_dir, self._tol)
        self._diss = _StimeIndex(diss_dir, self._tol)
        self._raw = _StimeIndex(profiles_dir, self._tol) if profiles_dir else None
        self._chi_cache: dict[str, ChiProfileFile] = {}
        self._diss_cache: dict[str, ProfileFile] = {}
        self._raw_cache: dict[str, RawProfile] = {}

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

    def load_raw(self, path: str) -> RawProfile:
        rp = self._raw_cache.get(path)
        if rp is None:
            rp = load_raw_profile(path)
            self._raw_cache[path] = rp
        return rp

    def prewarm(self) -> None:
        """Build all directory ``stime`` indexes now (background-thread safe)."""
        self._chi.ensure()
        self._diss.ensure()
        if self._raw is not None:
            self._raw.ensure()

    def cell(self, stime_epoch: float, bin_depth: float) -> Cell | None:
        """Resolve a mixing cell (chi primary + diss + raw), or None."""
        chi_path = self._chi.path_for(stime_epoch)
        if chi_path is None:
            return None
        chi_pf = self.load_chi(chi_path)
        window = nearest_window(chi_pf.depth, bin_depth)
        if window is None:
            return None

        diss_pf = None
        diss_path = self._diss.path_for(stime_epoch)
        if diss_path is not None:
            diss_pf = self.load_diss(diss_path)

        raw = None
        if self._raw is not None:
            rpath = self._raw.path_for(stime_epoch)
            if rpath is not None:
                raw = self.load_raw(rpath)

        return Cell(
            profile=chi_pf, window=window, raw=raw,
            profiles={"eps": diss_pf, "chi": chi_pf},
        )

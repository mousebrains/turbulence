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

from dataclasses import dataclass

import numpy as np
import xarray as xr

from odas_tpw.perturb.plot.diagnostics import _raw_stime_map


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
    with xr.open_dataset(combo_path, decode_times=False) as ds:
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


def load_profile_file(path: str) -> ProfileFile:
    """Load the arrays a diagnostic drill-down needs from one diss file."""
    with xr.open_dataset(path, decode_times=False) as ds:
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
class Cell:
    """A resolved overview cell: a per-profile file and one window index."""

    profile: ProfileFile
    window: int


class EpsilonCellSource:
    """Map overview ``(stime, bin)`` cells to a per-profile file + window.

    Files are located by matching the combo's per-profile ``stime`` (float epoch
    seconds) against each per-profile file's scalar ``stime`` within *tol*
    seconds, and cached once loaded.  The window is the one whose depth is
    nearest the clicked bin.
    """

    def __init__(self, diss_dir: str, *, tol: float = 1.0) -> None:
        self._tol = float(tol)
        # {round(stime, 3): path}; keys are per-profile file start times.
        self._smap = _raw_stime_map(diss_dir)
        self._keys = np.array(sorted(self._smap), dtype=float)
        self._cache: dict[str, ProfileFile] = {}

    def _file_for(self, stime_epoch: float) -> str | None:
        if not self._keys.size or not np.isfinite(stime_epoch):
            return None
        i = int(np.argmin(np.abs(self._keys - stime_epoch)))
        if abs(self._keys[i] - stime_epoch) <= self._tol:
            return self._smap[self._keys[i]]
        return None

    def load(self, path: str) -> ProfileFile:
        pf = self._cache.get(path)
        if pf is None:
            pf = load_profile_file(path)
            self._cache[path] = pf
        return pf

    def cell(self, stime_epoch: float, bin_depth: float) -> Cell | None:
        """Resolve one overview cell, or None if no per-profile file matches."""
        path = self._file_for(stime_epoch)
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
        return Cell(profile=pf, window=window)

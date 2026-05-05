# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Depth/time binning engine for profiles, dissipation, and chi.

Reference: Code/bin_by_real.m, Code/profile2binned.m, Code/diss2binned.m,
           Code/chi2binned.m
"""

import re
from pathlib import Path

import numpy as np
import xarray as xr

from odas_tpw.perturb.logging_setup import stage_log

# Per-profile NetCDFs are named ``<pfile_stem>_prof###.nc`` by
# :func:`odas_tpw.rsi.profile.extract_profiles`.  The binning step's
# per-input-file log scope groups them back by source .p file rather than
# splitting one log per profile, so we strip the suffix to recover the
# .p stem.  Inputs that don't match the pattern (third-party NetCDFs,
# already-binned outputs) keep their full stem.
_PROF_SUFFIX_RE = re.compile(r"_prof\d+$")


def _source_stem(profile_file: Path | str) -> str:
    """Return the source ``.p``-file stem for a per-profile NetCDF path."""
    return _PROF_SUFFIX_RE.sub("", Path(profile_file).stem)


def _bin_std(values: np.ndarray, coords: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    """NaN-aware std-per-bin via bincount (population std, ddof=0).

    Equivalent to a per-bin ``np.nanstd`` loop, but avoids the
    O(n_bins * n_samples) mask broadcast.
    """
    n_bins = len(bin_edges) - 1
    idx = np.clip(np.digitize(coords, bin_edges) - 1, 0, n_bins - 1)
    finite = np.isfinite(values)
    idx_f = idx[finite]
    vals_f = values[finite]
    counts = np.bincount(idx_f, minlength=n_bins)
    sums = np.bincount(idx_f, weights=vals_f, minlength=n_bins)
    sums_sq = np.bincount(idx_f, weights=vals_f * vals_f, minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)
        var = sums_sq / np.maximum(counts, 1) - means * means
        var = np.maximum(var, 0.0)  # cancellation safety
        std = np.where(counts > 1, np.sqrt(var), np.nan)
    return std


def _bin_array(
    values: np.ndarray,
    coords: np.ndarray,
    bin_edges: np.ndarray,
    agg_func,
) -> tuple[np.ndarray, np.ndarray]:
    """Core binning: assign values to bins and aggregate.

    Parameters
    ----------
    values : ndarray, 1D
    coords : ndarray, 1D (same length as values)
    bin_edges : ndarray, 1D (n_bins + 1)
    agg_func : callable (e.g. np.nanmean, np.nanmedian)

    Returns
    -------
    binned : ndarray, shape (n_bins,)
    counts : ndarray, shape (n_bins,) — int
    """
    n_bins = len(bin_edges) - 1
    idx = np.digitize(coords, bin_edges) - 1
    idx = np.clip(idx, 0, n_bins - 1)

    counts = np.bincount(idx, minlength=n_bins).astype(int)

    if agg_func is np.nanmean:
        # Vectorize the dominant case: NaN-aware mean via two bincounts.
        finite = np.isfinite(values)
        idx_f = idx[finite]
        vals_f = values[finite]
        sums = np.bincount(idx_f, weights=vals_f, minlength=n_bins)
        finite_counts = np.bincount(idx_f, minlength=n_bins)
        with np.errstate(invalid="ignore", divide="ignore"):
            binned = np.where(finite_counts > 0, sums / finite_counts, np.nan)
        return binned, counts

    # General fallback (e.g. nanmedian): sort once, then each bin is a
    # contiguous slice — avoids the per-bin O(n_samples) mask broadcast.
    order = np.argsort(idx, kind="stable")
    sorted_idx = idx[order]
    sorted_vals = values[order]
    splits = np.searchsorted(sorted_idx, np.arange(n_bins + 1))
    binned = np.full(n_bins, np.nan)
    for i in range(n_bins):
        sl = sorted_vals[splits[i] : splits[i + 1]]
        if sl.size > 0:
            binned[i] = agg_func(sl)
    return binned, counts


def _get_agg_func(aggregation: str):
    """Return the aggregation function for the given method name."""
    if aggregation == "median":
        return np.nanmedian
    return np.nanmean


def _depth_var_for(ds: xr.Dataset) -> str | None:
    """Return the canonical depth variable name in *ds*, or None."""
    for name in ("depth", "P", "P_mean"):
        if name in ds:
            return name
    return None


_SCALAR_NAMES = ("lat", "lon", "stime", "etime")


def _load_profile_snapshot(profile_file: Path) -> dict | None:
    """Read a profile NC once and return a snapshot for binning.

    Returned dict (or None when the file has no depth/P/P_mean variable):

    * ``depth`` — 1-D ndarray of the depth/P/P_mean coordinate
    * ``vars`` — ``{var_name: 1-D ndarray}`` for every numeric data var
      that aligns with depth
    * ``scalars`` — ``{lat|lon|stime|etime: float}`` (epoch seconds for time)
    * ``scalar_attrs`` — ``{name: attrs_dict}`` (units/calendar stripped — the
      combo's CF encoder re-emits them on write)

    One ``xr.open_dataset`` per call: callers cache snapshots so the
    second pass that needs ``bin_edges`` doesn't re-open. The dict is
    plain numpy / Python types so it pickles cheaply for worker pools.
    """
    with xr.open_dataset(profile_file) as ds:
        depth_var = _depth_var_for(ds)
        if depth_var is None:
            return None
        depth = ds[depth_var].values

        scalars: dict[str, float] = {}
        scalar_attrs: dict[str, dict] = {}
        for sname in _SCALAR_NAMES:
            if sname in ds.data_vars and ds[sname].ndim == 0:
                val = ds[sname].values
                if val.dtype.kind == "M":
                    val = val.astype("datetime64[ns]").astype(np.int64) / 1e9
                scalars[sname] = float(val)
                scalar_attrs[sname] = {
                    k: v
                    for k, v in ds[sname].attrs.items()
                    if k not in ("units", "calendar")
                }

        data: dict[str, np.ndarray] = {}
        for vname in ds.data_vars:
            if vname == depth_var or vname in _SCALAR_NAMES:
                continue
            arr = ds[vname].values
            if arr.ndim != 1 or len(arr) != len(depth):
                continue
            if arr.dtype.kind == "M":
                continue
            data[vname] = arr

    return {
        "depth": depth,
        "vars": data,
        "scalars": scalars,
        "scalar_attrs": scalar_attrs,
    }


def _bin_snapshot(snapshot: dict, bin_edges: np.ndarray, agg, diagnostics: bool) -> dict:
    """Bin a pre-loaded snapshot onto *bin_edges* — no file IO.

    Returns the same shape of dict that the previous ``_bin_one_profile``
    produced (``vars``, ``stds``, ``scalars``, ``scalar_attrs``).
    """
    out: dict = {"vars": {}, "stds": {}, "scalars": snapshot["scalars"],
                 "scalar_attrs": snapshot["scalar_attrs"]}
    depth = snapshot["depth"]
    for vname, arr in snapshot["vars"].items():
        binned, _ = _bin_array(arr, depth, bin_edges, agg)
        out["vars"][vname] = binned
        if diagnostics:
            out["stds"][vname] = _bin_std(arr, depth, bin_edges)
    return out


def _bin_one_profile(
    profile_file: Path,
    bin_edges: np.ndarray,
    agg,
    diagnostics: bool,
) -> dict:
    """Open + bin a single profile NetCDF onto *bin_edges*.

    Compatibility shim around :func:`_load_profile_snapshot` +
    :func:`_bin_snapshot` for callers that want the old "one open, one
    bin" behavior in a single function call.
    """
    snap = _load_profile_snapshot(profile_file)
    if snap is None:
        return {"vars": {}, "stds": {}, "scalars": {}, "scalar_attrs": {}}
    return _bin_snapshot(snap, bin_edges, agg, diagnostics)


def bin_by_depth(
    profile_files: list[Path],
    bin_width: float = 1.0,
    aggregation: str = "mean",
    diagnostics: bool = False,
    log_dir: Path | None = None,
) -> xr.Dataset:
    """Bin per-profile NetCDFs by depth into 2D (bin x profile).

    Parameters
    ----------
    profile_files : list of Path
        Paths to per-profile NetCDF files.
    bin_width : float
        Depth bin width [m].
    aggregation : str
        "mean" or "median".
    diagnostics : bool
        Include n_samples and *_std per bin.
    log_dir : Path, optional
        When supplied, the per-input-file binning loop is wrapped in
        :func:`~odas_tpw.perturb.logging_setup.stage_log` so each source
        file's records land in ``<log_dir>/<source_stem>.log`` (in
        addition to the worker/run logs).

    Returns
    -------
    xr.Dataset with dims (bin, profile).

    Notes
    -----
    Single open per profile NC: each file is read once into an in-memory
    snapshot (``_load_profile_snapshot``) holding its depth coord,
    aligned data vars and CF §9 scalars. The global bin grid is computed
    from the cached depths, then :func:`_bin_snapshot` bins each entry
    without further IO. The previous design re-opened every file in a
    second pass to bin it — that doubled the open count for what was
    really one file's worth of work.
    """
    agg = _get_agg_func(aggregation)

    snapshots: list[dict | None] = [_load_profile_snapshot(f) for f in profile_files]

    g_min = np.inf
    g_max = -np.inf
    saw_any = False
    for snap in snapshots:
        if snap is None:
            continue
        d = snap["depth"]
        d = d[np.isfinite(d)]
        if d.size == 0:
            continue
        saw_any = True
        g_min = min(g_min, float(d.min()))
        g_max = max(g_max, float(d.max()))
    if not saw_any:
        return xr.Dataset()

    d_min = np.floor(g_min / bin_width) * bin_width
    d_max = np.ceil(g_max / bin_width) * bin_width
    bin_edges = np.arange(d_min, d_max + bin_width, bin_width)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    n_bins = len(bin_centers)
    n_profiles = len(profile_files)

    result_vars: dict = {}
    profile_scalars = {
        "lat": np.full(n_profiles, np.nan),
        "lon": np.full(n_profiles, np.nan),
        "stime": np.full(n_profiles, np.nan),
        "etime": np.full(n_profiles, np.nan),
    }
    profile_scalar_attrs: dict[str, dict] = {}

    for pi, (pfile, snap) in enumerate(zip(profile_files, snapshots)):
        with stage_log(log_dir, _source_stem(pfile)):
            if snap is None:
                continue
            res = _bin_snapshot(snap, bin_edges, agg, diagnostics)
            # Drop the cached input data once the bin pass has consumed
            # it — keeps peak memory close to "one snapshot at a time"
            # for the binning pass even though we held them all through
            # the bin-edge computation.
            snapshots[pi] = None
            for sname, val in res["scalars"].items():
                profile_scalars[sname][pi] = val
                if sname not in profile_scalar_attrs:
                    profile_scalar_attrs[sname] = res["scalar_attrs"].get(sname, {})
            for vname, binned in res["vars"].items():
                if vname not in result_vars:
                    result_vars[vname] = np.full((n_bins, n_profiles), np.nan)
                    if diagnostics:
                        result_vars[f"{vname}_std"] = np.full(
                            (n_bins, n_profiles), np.nan
                        )
                result_vars[vname][:, pi] = binned
            if diagnostics:
                for vname, std_arr in res["stds"].items():
                    result_vars[f"{vname}_std"][:, pi] = std_arr

    if diagnostics:
        result_vars["n_samples"] = np.zeros((n_bins, n_profiles), dtype=float)

    data_vars: dict = {}
    for vname, arr in result_vars.items():
        data_vars[str(vname)] = (["bin", "profile"], arr)

    # Emit per-profile scalars as 1-D vars on the profile dim, only when at
    # least one profile actually carried them (preserves backward
    # compatibility with older per-profile NCs that lack lat/lon/stime/etime).
    for sname, sarr in profile_scalars.items():
        if np.any(np.isfinite(sarr)):
            data_vars[sname] = (["profile"], sarr, profile_scalar_attrs.get(sname, {}))

    return xr.Dataset(
        data_vars,
        coords={
            "bin": bin_centers,
            "profile": np.arange(n_profiles),
        },
    )


def bin_by_time(
    profile_files: list[Path],
    bin_width: float = 1.0,
    aggregation: str = "mean",
    diagnostics: bool = False,
    log_dir: Path | None = None,
) -> xr.Dataset:
    """Bin per-profile NetCDFs by time.

    Returns a concatenated dataset along a time dimension.

    See :func:`bin_by_depth` for the meaning of *log_dir*.
    """
    agg = _get_agg_func(aggregation)
    all_binned = []

    for f in profile_files:
        with stage_log(log_dir, _source_stem(f)):
            ds = xr.open_dataset(f)
            # Use the time coordinate
            for time_name in ("t_slow", "t_fast", "time"):
                if time_name in ds:
                    t = ds[time_name].values
                    break
            else:
                ds.close()
                continue

            t_min, t_max = np.nanmin(t), np.nanmax(t)
            bin_edges = np.arange(t_min, t_max + bin_width, bin_width)
            if len(bin_edges) < 2:
                bin_edges = np.array([t_min, t_min + bin_width])
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

            binned_data = {"time": bin_centers}
            for vname in ds.data_vars:
                arr = ds[vname].values
                if arr.ndim != 1 or len(arr) != len(t):
                    continue
                if arr.dtype.kind == "M":  # skip datetime64 variables
                    continue
                b, _ = _bin_array(arr, t, bin_edges, agg)
                binned_data[str(vname)] = b

            ds.close()
            if len(binned_data) > 1:
                bds = xr.Dataset(
                    {k: (["time"], v) for k, v in binned_data.items() if k != "time"},
                    coords={"time": binned_data["time"]},
                )
                all_binned.append(bds)

    if not all_binned:
        return xr.Dataset()
    return xr.concat(all_binned, dim="time")


def bin_diss(
    diss_files: list[Path],
    bin_width: float = 1.0,
    aggregation: str = "mean",
    method: str = "depth",
    diagnostics: bool = False,
    log_dir: Path | None = None,
) -> xr.Dataset:
    """Bin dissipation estimates by depth or time.

    Handles per-probe (e_1, e_2) and combined (epsilonMean) variables.
    """
    if method == "time":
        return bin_by_time(diss_files, bin_width, aggregation, diagnostics, log_dir=log_dir)
    return bin_by_depth(diss_files, bin_width, aggregation, diagnostics, log_dir=log_dir)


def bin_chi(
    chi_files: list[Path],
    bin_width: float = 1.0,
    aggregation: str = "mean",
    method: str = "depth",
    diagnostics: bool = False,
    log_dir: Path | None = None,
) -> xr.Dataset:
    """Bin chi estimates by depth or time."""
    if method == "time":
        return bin_by_time(chi_files, bin_width, aggregation, diagnostics, log_dir=log_dir)
    return bin_by_depth(chi_files, bin_width, aggregation, diagnostics, log_dir=log_dir)

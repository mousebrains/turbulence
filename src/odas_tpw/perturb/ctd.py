# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""CTD time-binning with GPS and seawater properties.

Bins slow (and optionally fast) scalar channels by time, computes
seawater properties per bin, interpolates GPS, and writes per-file
CTD NetCDF.

Uses BOTH up and down portions (full file, not profile-segmented).

Position is VMP-aware when down-casts are supplied: each descent is pinned to
its drop point (the ship fix at the cast start) rather than the ship's moving
position, and the reel-in gap walks the estimate from the drop point back onto
the ship track. See :func:`_assign_gps_with_casts`. Without casts (tow-yo, up-
only, or none detected) it falls back to the ship fix at each bin time.

Reference: Code/ctd2binned.m (the ``addGPS`` subfunction)
"""

from pathlib import Path

import numpy as np
import xarray as xr

from odas_tpw.perturb.binning import _bin_indices
from odas_tpw.perturb.gps import GPSProvider
from odas_tpw.perturb.seawater import add_seawater_properties


def _time_bin(
    t: np.ndarray,
    data: dict[str, np.ndarray],
    bin_width: float,
    method: str = "mean",
    diagnostics: bool = False,
    bin_edges: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Bin data arrays by time.

    Parameters
    ----------
    t : ndarray
        Time vector [s].
    data : dict of ndarray
        Data arrays to bin, keyed by name.
    bin_width : float
        Bin width [s].
    method : str
        "mean" or "median".
    diagnostics : bool
        If True, include per bin: ``n_samples`` (total samples landing in the
        bin, finite or not), per-channel ``{name}_std``, and per-channel
        ``{name}_n`` (the finite count actually averaged — n_samples overstates
        the effective N for channels with NaNs; mirrors the depth engine) (#42).
    bin_edges : ndarray, optional
        Pre-computed bin edges. If provided, *bin_width* is ignored and the
        same edges are used regardless of *t*'s exact range — required when
        binning slow and fast channels in a single output (their last
        samples differ by up to ``1/fs_slow - 1/fs_fast`` seconds, which
        can flip one extra bin into existence and produce mismatched
        time dimensions downstream).

    Returns
    -------
    dict with 'bin_centers' and binned data arrays (+ diagnostics if requested).
    """
    if bin_edges is None:
        t_min, t_max = np.nanmin(t), np.nanmax(t)
        bin_edges = np.arange(t_min, t_max + bin_width, bin_width)
        if len(bin_edges) < 2:
            bin_edges = np.array([t_min, t_min + bin_width])
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    n_bins = len(bin_centers)
    # Shared binner with an in-range mask: NaN / out-of-range times are EXCLUDED
    # rather than folded into the first/last bin (the np.clip this replaced
    # silently polluted the edge bins; mirrors the depth engine, binning.py).
    bin_idx, in_range = _bin_indices(t, bin_edges)

    result = {"bin_centers": bin_centers}

    if diagnostics:
        result["n_samples"] = np.bincount(bin_idx[in_range], minlength=n_bins).astype(int)

    if method == "median":
        # Sort once by bin index; each bin is then a contiguous slice. Avoids
        # the O(n_bins * n_samples) mask broadcasts of a naive ``bin_idx == i``
        # loop while still allowing per-bin nanmedian. Restrict to in-range
        # samples first so out-of-range times never enter a bin.
        idx_ir = bin_idx[in_range]
        order = np.argsort(idx_ir, kind="stable")
        sorted_idx = idx_ir[order]
        splits = np.searchsorted(sorted_idx, np.arange(n_bins + 1))
        for name, arr in data.items():
            sorted_arr = arr[in_range][order]
            binned = np.full(n_bins, np.nan)
            for i in range(n_bins):
                sl = sorted_arr[splits[i] : splits[i + 1]]
                # Guard on a finite value (not just size): channels like N2/dTdz
                # are NaN outside casts, so an all-NaN slice would make nanmedian
                # warn "All-NaN slice encountered" for every empty-of-data bin.
                if np.any(np.isfinite(sl)):
                    binned[i] = np.nanmedian(sl)
            result[name] = binned
            if diagnostics:
                std_arr = np.full(n_bins, np.nan)
                n_arr = np.zeros(n_bins, dtype=int)
                for i in range(n_bins):
                    sl = sorted_arr[splits[i] : splits[i + 1]]
                    n_finite = int(np.sum(np.isfinite(sl)))
                    n_arr[i] = n_finite
                    if n_finite > 1:
                        std_arr[i] = np.nanstd(sl)
                result[f"{name}_std"] = std_arr
                result[f"{name}_n"] = n_arr
        return result

    # Mean path — one O(N) bincount per channel, no per-bin Python loop.
    for name, arr in data.items():
        finite = np.isfinite(arr) & in_range
        idx_f = bin_idx[finite]
        arr_f = arr[finite]
        counts = np.bincount(idx_f, minlength=n_bins)
        sums = np.bincount(idx_f, weights=arr_f, minlength=n_bins)
        with np.errstate(invalid="ignore", divide="ignore"):
            binned = np.where(counts > 0, sums / counts, np.nan)
        result[name] = binned
        if diagnostics:
            # Per-channel finite count actually averaged (counts excludes NaNs),
            # distinct from the channel-agnostic n_samples (#42).
            result[f"{name}_n"] = counts.astype(int)
            sums_sq = np.bincount(idx_f, weights=arr_f * arr_f, minlength=n_bins)
            with np.errstate(invalid="ignore", divide="ignore"):
                # Population std (ddof=0) to match np.nanstd default.
                mean_per_bin = np.where(counts > 0, sums / counts, 0.0)
                var = sums_sq / np.maximum(counts, 1) - mean_per_bin * mean_per_bin
                # Numerical safety: clip tiny negatives from cancellation.
                var = np.maximum(var, 0.0)
                std_arr = np.where(counts > 1, np.sqrt(var), np.nan)
            result[f"{name}_std"] = std_arr

    return result


def _assign_gps_with_casts(
    bin_epoch: np.ndarray,
    cast_starts: np.ndarray,
    cast_ends: np.ndarray,
    gps: GPSProvider,
) -> tuple[np.ndarray, np.ndarray]:
    """VMP-aware lat/lon for CTD time bins (ports Matlab ``ctd2binned.addGPS``).

    A vertical profiler falls at essentially one position during a down-cast,
    then is reeled in while the ship steams on — so the GPS fix at the bin time
    (the ship's position) is *not* the instrument's position. This assigns,
    per bin-center time (epoch seconds):

    - **Before the first cast**: the ship fix at the bin time.
    - **During a down-cast** ``[start, end]``: the ship fix at the cast *start*,
      held fixed for the whole descent (the drop point).
    - **In the gap** ``(end_i, start_{i+1})`` (reel-in + steam to the next
      station): the ship-relative offset present at cast end decays linearly to
      zero by the next cast start, so the bin walks from the drop point onto the
      ship track:  ``pos(t) = gps(t) - (offset / dt_gap) * (t_next - t)``.

    *cast_starts* / *cast_ends* are the per-cast start/end times in epoch
    seconds, ordered.  Matches the reference exactly, including its edge: the
    final bin (at the end of the trailing gap) is left NaN.
    """
    bin_epoch = np.asarray(bin_epoch, dtype=np.float64)
    starts = np.asarray(cast_starts, dtype=np.float64)
    ends = np.asarray(cast_ends, dtype=np.float64)
    n = bin_epoch.size
    ncast = starts.size
    lat = np.full(n, np.nan)
    lon = np.full(n, np.nan)

    def _g(fn, t):
        return np.asarray(fn(np.atleast_1d(np.asarray(t, dtype=np.float64))), dtype=np.float64)

    # Before the first cast: ship fix at the bin time.
    pre = bin_epoch < starts[0] if ncast else np.ones(n, dtype=bool)
    if np.any(pre):
        lon[pre] = _g(gps.lon, bin_epoch[pre])
        lat[pre] = _g(gps.lat, bin_epoch[pre])

    # During each cast: ship fix at the cast start, held fixed for the descent.
    for i in range(ncast):
        during = (bin_epoch >= starts[i]) & (bin_epoch <= ends[i])
        if np.any(during):
            lon[during] = float(_g(gps.lon, starts[i])[0])
            lat[during] = float(_g(gps.lat, starts[i])[0])

    # Gaps (reel-in + steam): decay the cast-end ship-relative offset to zero.
    for i in range(ncast):
        t0 = ends[i]
        t1 = bin_epoch[-1] if i == ncast - 1 else starts[i + 1]
        gap = (bin_epoch > t0) & (bin_epoch < t1)
        if not np.any(gap):
            continue
        first = int(np.flatnonzero(gap)[0])
        # Drop-point fix = the bin just before the gap (the cast's last bin).
        # The ``first == 0`` guard deliberately departs from the reference: the
        # Matlab indexes ``ctd.lon(ii(1)-1)`` blindly and would error if a gap
        # begins at the very first bin (a cast ending before the first bin
        # center); here we degrade to the ship fix at t0 (zero offset -> plain
        # ship track) instead of crashing.
        lon0 = lon[first - 1] if first > 0 else float(_g(gps.lon, t0)[0])
        lat0 = lat[first - 1] if first > 0 else float(_g(gps.lat, t0)[0])
        dt_total = t1 - t0
        if dt_total <= 0:
            continue
        dlondt = (float(_g(gps.lon, t0)[0]) - lon0) / dt_total
        dlatdt = (float(_g(gps.lat, t0)[0]) - lat0) / dt_total
        t = bin_epoch[gap]
        dt = t1 - t
        lon[gap] = _g(gps.lon, t) - dlondt * dt
        lat[gap] = _g(gps.lat, t) - dlatdt * dt

    return lat, lon


def ctd_bin_file(
    pf,
    gps: GPSProvider,
    output_dir: str | Path,
    *,
    bin_width: float = 0.5,
    T_name: str = "JAC_T",
    C_name: str = "JAC_C",
    variables: list[str] | None = None,
    method: str = "mean",
    diagnostics: bool = False,
    output_stem: str | None = None,
    profiles: list[tuple[int, int]] | None = None,
    direction: str = "down",
) -> Path | None:
    """Time-bin CTD channels from a PFile and write to NetCDF.

    Parameters
    ----------
    pf : PFile
        Parsed .p file.
    gps : GPSProvider
        GPS provider for lat/lon interpolation.
    output_dir : str or Path
        Output directory.
    bin_width : float
        Time bin width [s].
    T_name, C_name : str
        Temperature and conductivity channel names.
    variables : list of str, optional
        Additional channels to bin (auto-detect if None).
    method : str
        Aggregation method ("mean" or "median").
    diagnostics : bool
        Write additional diagnostic variables.
    output_stem : str, optional
        Override the output filename stem. Defaults to ``pf.filepath.stem``.
    profiles : list of (int, int), optional
        Down-cast start/end indices into the slow time base (from
        ``get_profiles``). When given with ``direction="down"``, position is
        VMP-aware (descent pinned to the drop point, reel-in walked back onto
        the ship track). Otherwise the ship fix at each bin time is used.
    direction : str
        Profile direction. Cast-aware position applies only for ``"down"``.

    Returns
    -------
    Path or None
        Path to the output NetCDF, or None if insufficient data.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect channels to bin
    channels_to_bin = set()
    if T_name in pf.channels:
        channels_to_bin.add(T_name)
    if C_name in pf.channels:
        channels_to_bin.add(C_name)
    if "P" in pf.channels:
        channels_to_bin.add("P")
    # Background stratification injected by the pipeline (slow channels).
    for name in ("N2", "dTdz"):
        if name in pf.channels:
            channels_to_bin.add(name)

    if variables:
        for v in variables:
            if v in pf.channels:
                channels_to_bin.add(v)
    else:
        # Auto-detect scalar slow channels
        auto_names = ["DO", "Chlorophyll", "Turbidity", "Fluor"]
        for name in auto_names:
            if name in pf.channels:
                channels_to_bin.add(name)

    if not channels_to_bin:
        return None

    # Separate slow and fast channels
    slow_data = {}
    fast_data = {}
    for ch in channels_to_bin:
        if pf.is_fast(ch):
            fast_data[ch] = pf.channels[ch]
        else:
            slow_data[ch] = pf.channels[ch]

    # Compute a single set of bin edges spanning both time vectors so slow
    # and fast channels share one time dimension in the output dataset.
    t_lo = np.nanmin(pf.t_slow if slow_data else pf.t_fast)
    t_hi = max(
        np.nanmax(pf.t_slow) if slow_data else -np.inf,
        np.nanmax(pf.t_fast) if fast_data else -np.inf,
    )
    bin_edges = np.arange(t_lo, t_hi + bin_width, bin_width)
    if len(bin_edges) < 2:
        bin_edges = np.array([t_lo, t_lo + bin_width])

    binned: dict[str, np.ndarray] = {}
    bin_centers: np.ndarray | None = None
    if slow_data:
        result = _time_bin(
            pf.t_slow, slow_data, bin_width, method, diagnostics, bin_edges=bin_edges
        )
        bin_centers = result.pop("bin_centers")
        binned.update(result)

    if fast_data:
        result = _time_bin(
            pf.t_fast, fast_data, bin_width, method, diagnostics, bin_edges=bin_edges
        )
        if bin_centers is None:
            bin_centers = result.pop("bin_centers")
        else:
            result.pop("bin_centers")
        binned.update(result)

    if bin_centers is None or len(bin_centers) == 0:
        return None

    # GPS interpolation — query in epoch seconds so the GPS provider's
    # absolute time axis lines up with bin_centers (which live on the
    # file-relative pf.t_slow timeline).
    epoch_offset = pf.start_time.timestamp() if hasattr(pf, "start_time") else 0.0
    bin_epoch = np.asarray(bin_centers, dtype=np.float64) + epoch_offset
    if direction == "down" and profiles:
        # VMP-aware position: pin each descent to its drop point and walk the
        # reel-in gap from there back onto the ship track (Matlab addGPS). Cast
        # start/end times come from the slow time base, shifted to epoch.
        t_slow = np.asarray(pf.t_slow, dtype=np.float64) + epoch_offset
        cast_starts = np.array([t_slow[s] for s, _e in profiles], dtype=np.float64)
        cast_ends = np.array([t_slow[e] for _s, e in profiles], dtype=np.float64)
        lat, lon = _assign_gps_with_casts(bin_epoch, cast_starts, cast_ends, gps)
    else:
        # No down-casts (tow-yo, up-only, or none detected): the ship fix at the
        # bin time is the best estimate.
        lat = gps.lat(bin_epoch)
        lon = gps.lon(bin_epoch)
    binned["lat"] = lat
    binned["lon"] = lon

    # Seawater properties (if T, C, P available)
    if T_name in binned and C_name in binned and "P" in binned:
        sw = add_seawater_properties(binned[T_name], binned[C_name], binned["P"], lat, lon)
        binned.update(sw)

    # Promote bin_centers from file-relative seconds to epoch seconds so
    # concatenated CTD combos can be correctly ordered across files.  Add
    # CF time attrs so xarray's encoder serialises them as datetimes.
    time_epoch = np.asarray(bin_centers, dtype=np.float64) + epoch_offset

    # Build xarray Dataset
    ds = xr.Dataset(
        {name: (["time"], arr) for name, arr in binned.items()},
        coords={"time": time_epoch},
    )
    ds["time"].attrs["units"] = "seconds since 1970-01-01"
    ds["time"].attrs["calendar"] = "standard"
    ds["time"].attrs["standard_name"] = "time"
    ds["time"].attrs["long_name"] = "time bin center"
    ds["time"].attrs["axis"] = "T"
    ds["time"].attrs["units_metadata"] = "leap_seconds: utc"
    # Self-describing metadata for the injected background stratification, so
    # the CTD product's N2/dTdz is not confused with the dissipation/chi-window
    # N2/dTdz of the diss/chi products (different scale and method).
    strat_attrs = {
        "N2": {
            "units": "s-2",
            "long_name": "buoyancy frequency squared (background, Thorpe-sorted)",
            "comment": (
                "TEOS-10 N2 from the profile's own C/T/P over a background "
                "pressure window, Thorpe-sorted to a stable profile, then "
                "time-binned. Background (profile/CTD) scale — distinct from the "
                "dissipation/chi-window N2 in the diss/chi products."
            ),
        },
        "dTdz": {
            "units": "K m-1",
            "long_name": "background temperature gradient (positive down)",
            "comment": (
                "Least-squares slope of the Thorpe-sorted in-situ temperature "
                "vs depth over a background pressure window, then time-binned."
            ),
        },
    }
    for name, attrs in strat_attrs.items():
        if name in ds:
            ds[name].attrs.update(attrs)

    ds.attrs["bin_width"] = bin_width
    ds.attrs["method"] = method
    ds.attrs["source_file"] = pf.filepath.name
    ds.attrs["Conventions"] = "CF-1.13, ACDD-1.3"

    stem = output_stem or pf.filepath.stem
    out_path = output_dir / f"{stem}.nc"
    # Strip the auto-applied _FillValue from coordinate variables (notably the
    # float "time" coord) — CF-1.13 §2.5.1 forbids _FillValue on coordinates.
    # xarray emits one by default for floats; clear it via encoding, mirroring
    # make_combo. Without this the per-file ctd/*.nc products are non-compliant.
    encoding: dict = {cname: {"_FillValue": None} for cname in ds.coords}
    ds.to_netcdf(out_path, encoding=encoding)
    return out_path

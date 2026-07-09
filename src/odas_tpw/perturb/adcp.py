# Jul-2026, Claude and Pat Welch, pat@mousebrains.com
"""CODAS/UHDAS gridded shipboard-ADCP reader and finescale shear.

Reads the gridded NetCDF product a UHDAS installation writes under
``proc/<sonar>/contour/<sonar>.nc`` (e.g. ``wh300.nc``, ``os75nb.nc``) and
derives the finescale vertical shear squared

    ``S^2 = (du/dz)^2 + (dv/dz)^2``   [s^-2]

needed for gradient Richardson numbers ``Ri_g = N^2 / S^2`` on a
microstructure grid (Lewin et al. 2025, section 2b;
https://doi.org/10.1175/JPO-D-25-0012.1).

Contract and caveats
--------------------
- The CODAS *contour* product is already edit-masked: velocities are NaN
  wherever the editing flags rejected them, and finite cells carry
  ``pflag == 0``.  The reader still applies the ``pflag`` mask (belt and
  suspenders for files processed differently) and an optional
  percent-good floor.
- ``depth(time, depth_cell)`` is per-ensemble (instrument configuration
  can change mid-cruise); all depths are meters, positive down.
- Shear is first-differenced between *adjacent* cells only — a masked
  cell NaNs both differences that touch it, so a data gap never silently
  widens the differencing scale.  Ensembles within the query's time
  tolerance are averaged *before* differencing, which reduces (but does
  not remove) the single-ping noise floor ``~2*sigma_u^2/dz^2``; with
  cm/s ensemble noise over 2-m bins that floor is order 1e-4 s^-2,
  comparable to weak thermocline shear — treat small ``S^2`` (large
  ``Ri_g``) as an upper bound on shear, not a measurement.
- Different sonars (2-m wh300 vs 16-m os75) measure *different* shear:
  coarser bins systematically underestimate ``S^2``.  Callers must keep
  per-sonar results as separate labeled populations, never blend them.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np
import numpy.typing as npt

# Package-internal reuse of hotel.py's NaT-safe datetime64 conversion (a
# module-private name, but duplicating its NaT-sentinel handling is worse).
from odas_tpw.perturb.hotel import _dt64_to_epoch_s

# Ensembles within this many seconds of a query time contribute to its
# averaged velocity profile. wh300 writes ~120-s ensembles, so ±300 s
# averages ~5 ensembles over a VMP cast — comparable smoothing to the
# 20-30-s averages of Lewin et al. (2025) with their faster pings.
DEFAULT_TIME_TOLERANCE = 300.0

# A query depth is only interpolated when its bracketing finite shear
# mid-depths are within this factor of the sonar's median cell spacing —
# beyond that the "interpolation" would bridge a masked gap at a scale
# the differencing never measured.
DEFAULT_MAX_GAP_FACTOR = 2.5


class AdcpData(NamedTuple):
    """One CODAS gridded sonar product, masked and in epoch time."""

    name: str  # sonar label, e.g. "wh300" (defaults to the file stem)
    time: np.ndarray  # epoch seconds, shape (n_t,)
    depth: np.ndarray  # meters positive down, shape (n_t, n_cell)
    u: np.ndarray  # east velocity [m/s], NaN where rejected
    v: np.ndarray  # north velocity [m/s], NaN where rejected
    path: Path  # source file


class WindowShear(NamedTuple):
    """Per-query shear-squared estimates from :func:`window_shear`."""

    S2: np.ndarray  # shear squared [s^-2]; NaN where unresolvable
    n_ens: np.ndarray  # ensembles averaged per query (int)


def read_codas(
    path: str | Path,
    *,
    min_pg: float | None = None,
    name: str | None = None,
) -> AdcpData:
    """Read a CODAS gridded NetCDF (``contour/<sonar>.nc``).

    Parameters
    ----------
    path : str or Path
        The gridded product, e.g. ``.../proc/wh300/contour/wh300.nc``.
    min_pg : float or None
        Optional percent-good floor (``pg >= min_pg``).  The contour
        product is already edited (CODAS applies its own threshold), so
        the default is no additional cut.
    name : str or None
        Label for this sonar; defaults to the file stem (``wh300``).

    Returns
    -------
    AdcpData
        Time as epoch seconds, per-ensemble depths [m], and u/v [m/s]
        with rejected cells (non-finite, ``pflag != 0``, or below
        ``min_pg``) set to NaN.
    """
    import xarray as xr

    path = Path(path)
    with xr.open_dataset(path) as ds:
        time = _dt64_to_epoch_s(ds["time"].values)
        depth = np.asarray(ds["depth"].values, dtype=np.float64)
        u = np.asarray(ds["u"].values, dtype=np.float64)
        v = np.asarray(ds["v"].values, dtype=np.float64)
        bad = ~np.isfinite(u) | ~np.isfinite(v)
        if "pflag" in ds.variables:
            bad |= np.asarray(ds["pflag"].values) != 0
        if min_pg is not None:
            if "pg" not in ds.variables:
                # Silently skipping a requested QC floor would report
                # uncut data as if it passed the cut.
                raise ValueError(
                    f"{path}: min_pg={min_pg} requested but the file has "
                    "no 'pg' (percent-good) variable"
                )
            pg = np.asarray(ds["pg"].values, dtype=np.float64)
            bad |= ~(pg >= min_pg)
    u = np.where(bad, np.nan, u)
    v = np.where(bad, np.nan, v)
    # CODAS writes finite, strictly increasing time and shallow-to-deep
    # cell depths; violations mean a corrupt or concatenated file and
    # would silently break window_shear's searchsorted assumptions.
    if not (np.all(np.isfinite(time)) and np.all(np.diff(time) > 0)):
        raise ValueError(f"{path}: ADCP time axis is not finite and strictly increasing")
    if np.any(np.diff(depth, axis=-1) <= 0):  # NaN pairs compare False: skipped
        raise ValueError(f"{path}: ADCP cell depths are not shallow-to-deep ascending")
    return AdcpData(
        name=name if name is not None else path.stem,
        time=time,
        depth=depth,
        u=u,
        v=v,
        path=path,
    )


def shear_squared(
    u: npt.ArrayLike,
    v: npt.ArrayLike,
    depth: npt.ArrayLike,
) -> tuple[np.ndarray, np.ndarray]:
    """First-difference shear squared between adjacent depth cells.

    Works on a single profile ``(n_cell,)`` or a time-depth block
    ``(n_t, n_cell)``; differencing is along the last (cell) axis.

    Returns
    -------
    (S2, z_mid) : tuple of ndarray
        ``S2 = (du^2 + dv^2) / dz^2`` [s^-2] and the mid-cell depths
        [m], one element shorter than the input along the cell axis.
        NaN wherever either bounding cell is masked (no differencing
        across gaps) or the cell spacing is not positive.
    """
    u_arr = np.asarray(u, dtype=np.float64)
    v_arr = np.asarray(v, dtype=np.float64)
    z_arr = np.asarray(depth, dtype=np.float64)
    du = np.diff(u_arr, axis=-1)
    dv = np.diff(v_arr, axis=-1)
    dz = np.diff(z_arr, axis=-1)
    z_mid = 0.5 * (z_arr[..., 1:] + z_arr[..., :-1])
    with np.errstate(divide="ignore", invalid="ignore"):
        s2 = np.where(dz > 0, (du**2 + dv**2) / dz**2, np.nan)
    return s2, z_mid


def _nanmean_axis0(a: np.ndarray) -> np.ndarray:
    """NaN-aware column means without np.nanmean's empty-slice warning."""
    m = np.isfinite(a)
    cnt = m.sum(axis=0)
    tot = np.where(m, a, 0.0).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, tot / np.where(cnt > 0, cnt, 1), np.nan)


def window_shear(
    adcp: AdcpData,
    times: npt.ArrayLike,
    depths: npt.ArrayLike,
    *,
    time_tolerance: float = DEFAULT_TIME_TOLERANCE,
    max_gap: float | None = None,
) -> WindowShear:
    """Shear squared at (time, depth) query points (e.g. chi windows).

    For each query, ensembles within ``time_tolerance`` seconds are
    averaged per cell (NaN-aware) *before* differencing — averaging
    velocities first suppresses uncorrelated ensemble noise that would
    otherwise rectify into a spurious ``S^2`` floor — then ``S^2`` is
    linearly interpolated from the mid-cell depths to the query depth.

    Parameters
    ----------
    adcp : AdcpData
        A sonar product from :func:`read_codas`.
    times, depths : array_like, shape (n,)
        Query epoch times [s] and depths [m, positive down].
    time_tolerance : float
        Half-width [s] of the ensemble-selection window.
    max_gap : float or None
        Maximum separation [m] between the bracketing finite shear
        estimates for interpolation to proceed; ``None`` uses
        ``DEFAULT_MAX_GAP_FACTOR`` times the sonar's median cell
        spacing.  Queries outside the finite shear column, or spanned
        by a wider gap, return NaN.

    Returns
    -------
    WindowShear
        ``S2`` [s^-2] and the ensemble count per query.
    """
    q_times = np.atleast_1d(np.asarray(times, dtype=np.float64))
    q_depths = np.atleast_1d(np.asarray(depths, dtype=np.float64))
    if q_times.ndim > 1 or q_depths.ndim > 1:
        raise ValueError("times and depths must be 1-D (or scalar)")
    if q_times.shape != q_depths.shape:
        raise ValueError("times and depths must have the same shape")
    if not (np.isfinite(time_tolerance) and time_tolerance > 0):
        raise ValueError(f"time_tolerance must be positive, got {time_tolerance}")
    n = q_times.size
    s2_out = np.full(n, np.nan)
    n_ens = np.zeros(n, dtype=np.intp)

    if max_gap is None:
        cell_dz = np.diff(adcp.depth, axis=-1)
        med_dz = float(np.nanmedian(cell_dz)) if cell_dz.size else np.nan
        if not (np.isfinite(med_dz) and med_dz > 0):
            # < 2 cells (or no finite spacing): the sonar cannot produce a
            # first-difference shear at all — fail loudly, not with NaNs.
            raise ValueError(f"{adcp.name}: no usable cell spacing for shear differencing")
        max_gap = DEFAULT_MAX_GAP_FACTOR * med_dz

    t_adcp = adcp.time
    for i in range(n):
        if not (np.isfinite(q_times[i]) and np.isfinite(q_depths[i])):
            continue
        lo = int(np.searchsorted(t_adcp, q_times[i] - time_tolerance, "left"))
        hi = int(np.searchsorted(t_adcp, q_times[i] + time_tolerance, "right"))
        if hi <= lo:
            continue
        n_ens[i] = hi - lo
        # NaN-aware per-cell ensemble means. A cell present in only some
        # ensembles still contributes — the mask varies ping to ping.
        u_bar = _nanmean_axis0(adcp.u[lo:hi])
        v_bar = _nanmean_axis0(adcp.v[lo:hi])
        z_bar = _nanmean_axis0(adcp.depth[lo:hi])
        s2, z_mid = shear_squared(u_bar, v_bar, z_bar)
        good = np.isfinite(s2) & np.isfinite(z_mid)
        if not np.any(good):
            continue
        zg = z_mid[good]
        sg = s2[good]
        # zg inherits the cell ordering (shallow -> deep) from the file.
        k = int(np.searchsorted(zg, q_depths[i], side="right"))
        if k == 0:
            continue  # query above the finite shear column
        if k == zg.size:
            if q_depths[i] == zg[-1]:  # exactly on the deepest estimate
                s2_out[i] = sg[-1]
            continue  # else: below the finite shear column
        if zg[k] - zg[k - 1] > max_gap:
            continue  # bracketing estimates straddle a masked gap
        frac = (q_depths[i] - zg[k - 1]) / (zg[k] - zg[k - 1])
        s2_out[i] = sg[k - 1] + frac * (sg[k] - sg[k - 1])

    return WindowShear(S2=s2_out, n_ens=n_ens)

# Jun-2026, Claude and Pat Welch, pat@mousebrains.com
"""Plot-time diagnostic pseudo-variables for ``perturb-plot profiles``.

Shear / vibration / temperature-gradient *variance* is not stored in any binned
product — it lives in the raw fast channels of the per-profile files
(``profiles_NN/*_prof*.nc``).  This module computes, on demand, a
``(bin, profile)`` grid of the per-pressure-bin variance of a raw fast channel,
aligned to a combo's profiles by ``stime``.

Fidelity: the channel is preprocessed with the same two steps the epsilon path
uses — a first-order Butterworth high-pass (``_hp_filter``, removing platform
motion / the mean gradient swept through a bin) followed by the iterative
despike (removing spikes) — then the time-variance is taken in each pressure
bin.  It is applied over the *whole* cast record, NOT within the epsilon path's
speed/pressure-selected sections, so cast-start/turnaround transients are
included; this is a QC indicator, so the approximation is acceptable.  These
are **contamination/activity diagnostics**, not
turbulence quantities: ``Ax``/``Ay`` are raw piezo *counts* (so their variance
is instrument-relative), and the resolved-band variance of shear is related to
but not equal to epsilon.

Requested via ``--var <channel>_var`` (e.g. ``sh1_var``, ``Ax_var``,
``T1_dT1_var``).
"""

from __future__ import annotations

import glob
import os

import numpy as np

# pseudo-var -> (raw fast channel, colorbar label (mathtext units), cmocean cmap)
_DIAG_INFO: dict[str, tuple[str, str, str]] = {
    "sh1_var": ("sh1", r"sh1 HP/despiked variance (s$^{-2}$)", "amp"),
    "sh2_var": ("sh2", r"sh2 HP/despiked variance (s$^{-2}$)", "amp"),
    "Ax_var": ("Ax", r"Ax vibration variance (counts$^2$)", "amp"),
    "Ay_var": ("Ay", r"Ay vibration variance (counts$^2$)", "amp"),
    "T1_dT1_var": ("T1_dT1", r"T1 gradient-channel variance (K$^2$)", "amp"),
    "T2_dT2_var": ("T2_dT2", r"T2 gradient-channel variance (K$^2$)", "amp"),
}

# Cache of {round(stime,3) -> rawpath} per profiles dir (built once per run).
_STIME_CACHE: dict[str, dict[float, str]] = {}


def is_pseudo_var(name: str) -> bool:
    """True when *name* is a diagnostic pseudo-variable (computed, not stored)."""
    return name in _DIAG_INFO


def pseudo_names() -> list[str]:
    return list(_DIAG_INFO)


def pseudo_label(name: str) -> str:
    return _DIAG_INFO[name][1]


def pseudo_cmap(name: str) -> str:
    return _DIAG_INFO[name][2]


def _raw_stime_map(profiles_dir: str) -> dict[float, str]:
    """Map ``round(stime, 3) -> rawpath`` over ``profiles_dir/*_prof*.nc``.

    Built once per directory.  Reading a scalar ``stime`` is cheap (metadata
    only); matching is by value because the combo is stime-sorted across
    instruments, so filename/glob order does not correspond to combo order.
    """
    if profiles_dir in _STIME_CACHE:
        return _STIME_CACHE[profiles_dir]
    import xarray as xr

    m: dict[float, str] = {}
    for p in sorted(glob.glob(os.path.join(profiles_dir, "*prof*.nc"))):
        try:
            with xr.open_dataset(p, decode_times=False) as ds:
                if "stime" in ds.variables:
                    m[round(float(ds["stime"].values), 3)] = p
        except (OSError, ValueError):
            continue
    _STIME_CACHE[profiles_dir] = m
    return m


def _variance_by_bin(sig: np.ndarray, pressure: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    """Time-variance of *sig* within each pressure bin (NaN where < 3 samples)."""
    n = len(bin_edges) - 1
    out = np.full(n, np.nan)
    good = np.isfinite(sig) & np.isfinite(pressure)
    sig, pressure = sig[good], pressure[good]
    if sig.size == 0:
        return out
    idx = np.digitize(pressure, bin_edges) - 1
    idx[pressure == bin_edges[-1]] = n - 1  # keep the top edge in the last bin (cf. binning.py)
    for i in range(n):
        s = sig[idx == i]
        if s.size > 2:
            out[i] = float(np.var(s))
    return out


def _binned_channel_variance(
    rawpath: str, channel: str, bin_edges: np.ndarray,
    hp_cut: float, despike_thresh: float, despike_smooth: float,
) -> np.ndarray | None:
    """Per-pressure-bin variance of one HP-filtered, despiked fast channel."""
    import xarray as xr

    from odas_tpw.scor160.despike import despike
    from odas_tpw.scor160.l2 import _hp_filter

    try:
        with xr.open_dataset(rawpath, decode_times=False) as ds:
            if not all(v in ds.variables for v in (channel, "P", "t_fast", "t_slow")):
                return None
            sig = np.asarray(ds[channel].values, dtype=float).ravel()
            t_fast = np.asarray(ds["t_fast"].values, dtype=float)
            t_slow = np.asarray(ds["t_slow"].values, dtype=float)
            pres = np.asarray(ds["P"].values, dtype=float)
    except (OSError, ValueError, RuntimeError):  # one bad raw file shouldn't kill the figure
        return None

    if sig.size < 16 or sig.size != t_fast.size:
        return None
    dt = float(np.median(np.diff(t_fast)))
    if not np.isfinite(dt) or dt <= 0:
        return None
    fs = 1.0 / dt

    clean = despike(_hp_filter(sig, fs, hp_cut), fs,
                    thresh=despike_thresh, smooth=despike_smooth).y
    pressure_fast = np.interp(t_fast, t_slow, pres)
    return _variance_by_bin(clean, pressure_fast, bin_edges)


def compute_pseudo_grid(
    pseudo: str,
    combo_stime: np.ndarray,
    bin_centers: np.ndarray,
    profiles_dir: str,
    *,
    hp_cut: float = 1.0,
    despike_thresh: float = 8.0,
    despike_smooth: float = 0.5,
    tol: float = 1.0,
) -> np.ndarray:
    """``(n_bin, n_profile)`` grid of a pseudo-variable for the given profiles.

    Each combo profile (one per column of ``combo_stime``) is matched to its raw
    file by ``stime`` (within ``tol`` seconds) and its channel variance is binned
    on the combo's pressure grid (``bin_centers``, dbar).  Unmatched profiles are
    left NaN and reported.  Binning uses the combo's own bin edges so the panel
    aligns with the stored-variable panels.
    """
    from odas_tpw.perturb.plot import layout

    channel = _DIAG_INFO[pseudo][0]
    smap = _raw_stime_map(profiles_dir)
    keys = np.array(sorted(smap), dtype=float)
    bin_edges = layout.depth_edges(np.asarray(bin_centers, dtype=float))

    n_bin = len(bin_centers)
    combo_stime = np.asarray(combo_stime, dtype=float)
    grid = np.full((n_bin, combo_stime.size), np.nan)

    n_miss = 0
    for j, st in enumerate(combo_stime):
        path = None
        if keys.size and np.isfinite(st):
            i = int(np.argmin(np.abs(keys - st)))
            if abs(keys[i] - st) <= tol:
                path = smap[keys[i]]
        if path is None:
            n_miss += 1
            continue
        col = _binned_channel_variance(
            path, channel, bin_edges, hp_cut, despike_thresh, despike_smooth
        )
        if col is not None:
            grid[:, j] = col
    if n_miss:
        print(f"{pseudo}: {n_miss}/{combo_stime.size} profiles had no raw-file match")
    return grid

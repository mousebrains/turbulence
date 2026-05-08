# May-2026, Claude and Pat Welch, pat@mousebrains.com
"""Pcolor of log10(epsilon), log10(chi) and log10(chi/epsilon) vs depth and cast.

Reads ``diss_combo_NN/combo.nc`` for ``epsilonMean(bin, profile)`` and
``chi_combo_NN/combo.nc`` for ``chiMean(bin, profile)``. Falls back to
binning per-profile chi NetCDFs when ``chiMean`` isn't on the combo file
(i.e. perturb runs from before the chi_combine PR).
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import xarray as xr

from odas_tpw.perturb.plot import layout


def _load_epsilon(diss_combo_path: str):
    with xr.open_dataset(diss_combo_path) as ds:
        eps = ds["epsilonMean"].transpose("bin", "profile").values
        depth = ds["bin"].values
        times = ds["stime"].values
    return times, depth, eps


def _load_chi_from_combo(chi_combo_path: str, chi_attrs_dir: str):
    """Read chiMean(bin, profile) from a chi_combo NetCDF.

    The combo file does not carry chi fft / spectrum metadata, so read
    those from any per-profile NetCDF in *chi_attrs_dir* for the title.
    """
    with xr.open_dataset(chi_combo_path) as ds:
        chi = ds["chiMean"].transpose("bin", "profile").values
        depth = ds["bin"].values
        times = ds["stime"].values
    attrs: dict = {}
    sample = sorted(glob.glob(os.path.join(chi_attrs_dir, "*_prof*.nc")))
    if sample:
        with xr.open_dataset(sample[0]) as sds:
            attrs = dict(sds.attrs)
    return times, depth, chi, attrs


def _load_chi_legacy_per_profile(chi_dir: str, depth: np.ndarray):
    """Legacy fallback — bin per-profile chi onto the diss_combo depth grid.

    Used when chi_combo lacks ``chiMean`` (perturb runs predating the
    chi_combine landing). Computes the probe-mean of ``chi(probe, time)``
    and re-bins it at the diss depth resolution.
    """
    files = sorted(glob.glob(os.path.join(chi_dir, "*_prof*.nc")))
    if not files:
        raise SystemExit(f"No chi profiles in {chi_dir}")

    with xr.open_dataset(files[0]) as ds:
        sample_attrs = dict(ds.attrs)
    edges = layout.depth_edges(depth)
    chi = np.full((depth.size, len(files)), np.nan)
    times = np.empty(len(files), dtype="datetime64[ns]")

    for j, fn in enumerate(files):
        with xr.open_dataset(fn) as d:
            c = np.nanmean(d["chi"].values, axis=0)
            p = np.asarray(d["P_mean"].values)
            times[j] = d["stime"].values
        idx = np.digitize(p, edges) - 1
        ok = (idx >= 0) & (idx < depth.size) & np.isfinite(c)
        for i, v in zip(idx[ok], c[ok]):
            cur = chi[i, j]
            chi[i, j] = v if np.isnan(cur) else 0.5 * (cur + v)

    order = np.argsort(times)
    return times[order], chi[:, order], sample_attrs


def _per_profile_attrs(root: str, sibling_prefix: str) -> dict:
    """Pick attrs from any per-profile NetCDF in ``<root>/<sibling_prefix>_NN/``."""
    sib_dirs = sorted(glob.glob(os.path.join(root, f"{sibling_prefix}_[0-9][0-9]")))
    if not sib_dirs:
        return {}
    files = sorted(glob.glob(os.path.join(sib_dirs[-1], "*_prof*.nc")))
    if not files:
        return {}
    with xr.open_dataset(files[0]) as ds:
        return dict(ds.attrs)


def add_arguments(p: argparse.ArgumentParser) -> None:
    """Register CLI flags for the eps-chi subcommand on *p*."""
    p.add_argument("--root", required=True,
                   help="perturb output root (e.g. grg/processed/vmp)")
    p.add_argument("--out", default=None,
                   help="output figure path (default: <root>/eps_chi_pcolor.png)")
    p.add_argument("--title", default=None,
                   help="title prefix (default: basename of --root)")
    p.add_argument("--eps-vmin", type=float, default=None,
                   help="override 1%% quantile of epsilon")
    p.add_argument("--eps-vmax", type=float, default=None,
                   help="override 99%% quantile of epsilon")
    p.add_argument("--chi-vmin", type=float, default=None,
                   help="override 1%% quantile of chi")
    p.add_argument("--chi-vmax", type=float, default=None,
                   help="override 99%% quantile of chi")
    p.add_argument("--gam-vmin", type=float, default=None,
                   help="override 1%% quantile of chi/eps")
    p.add_argument("--gam-vmax", type=float, default=None,
                   help="override 99%% quantile of chi/eps")
    p.add_argument("--gap-seconds", type=float, default=600,
                   help="split casts when gap exceeds this (default 10 min; "
                        "gliders typically want a larger value, e.g. 14400)")


def run(args: argparse.Namespace) -> str:
    """Render the pcolor figure described by *args*. Returns the output path."""
    import cmocean
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    diss_dir = layout.latest_stage_dir(args.root, "diss_combo")
    if diss_dir is None:
        raise SystemExit(f"No diss_combo dir under {args.root}")
    diss_combo = os.path.join(diss_dir, "combo.nc")

    chi_combo_dir = layout.latest_stage_dir(args.root, "chi_combo")
    chi_dir = layout.latest_stage_dir(args.root, "chi")

    t_eps, depth, eps = _load_epsilon(diss_combo)

    if chi_combo_dir is not None:
        chi_combo_path = os.path.join(chi_combo_dir, "combo.nc")
        with xr.open_dataset(chi_combo_path) as peek:
            has_chi_mean = "chiMean" in peek.data_vars
        if has_chi_mean and chi_dir is not None:
            t_chi, _depth_chi, chi, chi_attrs = _load_chi_from_combo(
                chi_combo_path, chi_dir
            )
        elif chi_dir is not None:
            t_chi, chi, chi_attrs = _load_chi_legacy_per_profile(chi_dir, depth)
        else:
            raise SystemExit(f"No chi_NN dir under {args.root}")
    elif chi_dir is not None:
        t_chi, chi, chi_attrs = _load_chi_legacy_per_profile(chi_dir, depth)
    else:
        raise SystemExit(f"No chi data under {args.root}")

    if len(t_chi) != len(t_eps) or np.any(
        np.abs(
            t_chi.astype("datetime64[s]").astype(np.int64)
            - t_eps.astype("datetime64[s]").astype(np.int64)
        ) > 5
    ):
        # Realign chi onto the eps profile axis so gamma broadcasts.
        # Chi typically has *fewer* profiles than eps (fp07 calibration
        # rejects some), so we pad with NaN at eps profile slots that
        # have no chi match.
        eps_secs = t_eps.astype("datetime64[s]").astype(np.int64)
        chi_secs = t_chi.astype("datetime64[s]").astype(np.int64)
        aligned = np.full((chi.shape[0], len(t_eps)), np.nan)
        # Map each chi column to its nearest eps column (within 5 s).
        for j, s in enumerate(chi_secs):
            k = int(np.argmin(np.abs(eps_secs - s)))
            if abs(eps_secs[k] - s) <= 5:
                aligned[:, k] = chi[:, j]
        n_unmatched = int(np.sum(~np.isfinite(aligned).any(axis=0)))
        print(
            f"chi/eps profile counts differ ({len(t_chi)} chi vs "
            f"{len(t_eps)} eps); aligned chi onto eps axis "
            f"({n_unmatched} eps slots have no chi)"
        )
        chi = aligned
    t = t_eps

    cast_x, segments, centers, t_starts, _t_ends = layout.compute_layout(
        t, gap_seconds=args.gap_seconds
    )

    fig, (ax_e, ax_c, ax_g) = plt.subplots(
        3, 1, figsize=(11, 11), sharex=True, sharey=True,
        constrained_layout=True,
    )
    cmap = cmocean.cm.thermal

    with np.errstate(divide="ignore", invalid="ignore"):
        gamma = np.where((eps > 0) & np.isfinite(chi), chi / eps, np.nan)

    eps_vmin, eps_vmax = layout.quantile_limits(eps, args.eps_vmin, args.eps_vmax)
    chi_vmin, chi_vmax = layout.quantile_limits(chi, args.chi_vmin, args.chi_vmax)
    gam_vmin, gam_vmax = layout.quantile_limits(gamma, args.gam_vmin, args.gam_vmax)

    layout.plot_panel(
        ax_e, fig, cast_x, segments, depth, eps, cmap,
        LogNorm(vmin=eps_vmin, vmax=eps_vmax),
        r"$\varepsilon$  (W kg$^{-1}$)",
    )
    ax_e.set_ylabel("Depth (m)")

    layout.plot_panel(
        ax_c, fig, cast_x, segments, depth, chi, cmap,
        LogNorm(vmin=chi_vmin, vmax=chi_vmax),
        r"$\chi$  (K$^{2}$ s$^{-1}$)",
    )
    ax_c.set_ylabel("Depth (m)")

    layout.plot_panel(
        ax_g, fig, cast_x, segments, depth, gamma, cmap,
        LogNorm(vmin=gam_vmin, vmax=gam_vmax),
        r"$\chi / \varepsilon$  (K$^{2}$ s kg J$^{-1}$)",
    )
    ax_g.set_ylabel("Depth (m)")
    ax_g.set_xlabel("Cast number  (cluster start time, UTC)")

    eps_attrs = _per_profile_attrs(args.root, "diss")

    def _secs(attrs: dict, key: str) -> str:
        fs = attrs.get("fs_fast") or attrs.get("fs")
        if fs and key in attrs:
            return f"{attrs[key] / fs:.2g}"
        return "?"

    eps_fft, eps_diss = _secs(eps_attrs, "fft_length"), _secs(eps_attrs, "diss_length")
    chi_fft, chi_diss = _secs(chi_attrs, "fft_length"), _secs(chi_attrs, "diss_length")
    spectrum = chi_attrs.get("spectrum_model", "?")
    fp07 = chi_attrs.get("fp07_model", "?")
    if (eps_fft, eps_diss) == (chi_fft, chi_diss):
        lengths = rf"$\varepsilon$,$\chi$: fft {eps_fft} s, diss {eps_diss} s"
    else:
        lengths = (
            rf"$\varepsilon$: fft {eps_fft} s, diss {eps_diss} s   |   "
            rf"$\chi$: fft {chi_fft} s, diss {chi_diss} s"
        )
    title = args.title or os.path.basename(os.path.normpath(args.root))
    fig.suptitle(
        f"{title}   —   {lengths}   —   "
        rf"$\chi$: {spectrum} spectrum, $\varepsilon$-fixed $k_B$, FP07 {fp07}"
    )

    ax_e.invert_yaxis()

    def _fmt(dt: np.datetime64) -> str:
        return np.datetime_as_string(dt, unit="m").replace("T", " ")

    cluster_labels = []
    for k, (s, e) in enumerate(segments):
        n_label = f"{s + 1}" if e - s == 1 else f"{s + 1}-{e}"
        cluster_labels.append(f"{n_label}\n{_fmt(t_starts[k])}")
    ax_g.set_xticks(centers)
    ax_g.set_xticklabels(cluster_labels)
    ax_g.set_xticks(cast_x, minor=True)

    for label in ax_g.get_xticklabels():
        label.set_rotation(30)
        label.set_ha("right")

    out = args.out or os.path.join(args.root, "eps_chi_pcolor.png")
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}")
    return out

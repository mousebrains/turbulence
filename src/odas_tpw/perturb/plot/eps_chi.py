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
        # qc_drop_epsilon is float64 with NaN where no samples fell;
        # non-zero values flag bins to drop when --apply-qc is on.
        if "qc_drop_epsilon" in ds.data_vars:
            qc = ds["qc_drop_epsilon"].transpose("bin", "profile").values
        else:
            qc = None
    return times, depth, eps, qc


def _load_chi_from_combo(chi_combo_path: str, chi_attrs_dir: str | None):
    """Read chiMean(bin, profile) from a chi_combo NetCDF.

    The combo file does not carry chi fft / spectrum metadata, so read
    those from any per-profile NetCDF in *chi_attrs_dir* for the title.
    """
    with xr.open_dataset(chi_combo_path) as ds:
        chi = ds["chiMean"].transpose("bin", "profile").values
        depth = ds["bin"].values
        times = ds["stime"].values
        if "qc_drop_chi" in ds.data_vars:
            qc = ds["qc_drop_chi"].transpose("bin", "profile").values
        else:
            qc = None
    attrs: dict = {}
    # chi_attrs_dir only supplies the title attrs; it may be absent when the
    # combo carries chiMean but no per-profile chi_NN dir was kept.
    if chi_attrs_dir is not None:
        sample = sorted(glob.glob(os.path.join(chi_attrs_dir, "*_prof*.nc")))
        if sample:
            with xr.open_dataset(sample[0]) as sds:
                attrs = dict(sds.attrs)
    return times, depth, chi, attrs, qc


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
    # Legacy fallback has no aggregated qc bitfield.
    return times[order], chi[:, order], sample_attrs, None


def _reindex_rows_to_depth(
    arr: np.ndarray,
    src_depth: np.ndarray,
    dst_depth: np.ndarray,
    *,
    reduce: str = "mean",
) -> np.ndarray:
    """Re-bin a ``(bin, profile)`` array from *src_depth* rows onto *dst_depth*.

    The chi-combo carries its own ``bin`` (pressure) grid, built independently
    of the diss-combo grid: chi and diss profiles are top-trimmed / calibration-
    rejected differently, so even at identical bin width their centers routinely
    differ (and the widths are separately configurable). Plotting chi on eps's
    y-axis without this step silently misregisters chi in depth (and divides
    chi at one depth by eps at another in gamma). Mirrors the legacy per-profile
    path's ``depth_edges`` + ``digitize`` re-binning.

    ``reduce='mean'`` linearly averages source rows that fall in the same eps
    bin (consistent with the legacy path); ``reduce='max'`` ORs a drop bitfield
    so a flagged source bin flags its destination.
    """
    src_depth = np.asarray(src_depth, dtype=float)
    dst_depth = np.asarray(dst_depth, dtype=float)
    if src_depth.shape == dst_depth.shape and np.allclose(
        src_depth, dst_depth, atol=1e-6
    ):
        return arr  # already bin-for-bin aligned (the common matched-grid case)
    edges = layout.depth_edges(dst_depth)
    idx = np.digitize(src_depth, edges) - 1
    nprof = arr.shape[1]
    acc = np.zeros((dst_depth.size, nprof))
    cnt = np.zeros((dst_depth.size, nprof))
    for src, dst in enumerate(idx.tolist()):
        if 0 <= dst < dst_depth.size:
            row = arr[src]
            ok = np.isfinite(row)
            if reduce == "max":
                # True bitwise OR for the drop bitfield: np.maximum loses bits
                # (max(1,2)=2 but 1|2=3), contradicting the "ORs a drop
                # bitfield" contract. Flags are integral values carried in a
                # float array, so cast per-combine and store back (#52).
                acc[dst, ok] = np.bitwise_or(
                    acc[dst, ok].astype(np.int64), row[ok].astype(np.int64)
                )
                cnt[dst, ok] += 1
            else:
                acc[dst, ok] += row[ok]
                cnt[dst, ok] += 1
    with np.errstate(invalid="ignore"):
        return np.where(cnt > 0, acc if reduce == "max" else acc / cnt, np.nan)


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


def _fmt_minute(dt: np.datetime64) -> str:
    """Format a datetime64 as 'YYYY-MM-DD HH:MM'."""
    return str(np.datetime_as_string(dt, unit="m").replace("T", " "))


def _time_ticks(
    t: np.ndarray,
    cast_x: np.ndarray,
    segments: list[tuple[int, int]],
) -> tuple[np.ndarray, list[str]]:
    """Major-tick positions/labels for the cast x-axis.

    Inside each cluster, place labels at round-time intervals (1h, 2h,
    3h, 6h, 12h, 24h, 48h) chosen so the cluster gets ~4-8 ticks.
    Single-profile clusters get one label at the start time. The first
    tick in each cluster is forced to fall on a round multiple of the
    chosen interval so labels read as clean wall-clock times.
    """
    positions: list[float] = []
    labels: list[str] = []
    for k, (s, e) in enumerate(segments):
        cluster_t = t[s:e]
        cluster_x = cast_x[s:e]
        n = e - s
        if n == 1:
            positions.append(float(cluster_x[0]))
            labels.append(f"{s + 1}\n{_fmt_minute(cluster_t[0])}")
            continue
        span_sec = float(
            (cluster_t[-1] - cluster_t[0]) / np.timedelta64(1, "s")
        )
        span_hr = span_sec / 3600.0
        interval_hr = 48
        for cand in (1, 2, 3, 6, 12, 24, 48):
            if span_hr / cand <= 8:
                interval_hr = cand
                break
        # Round cluster start up to next multiple of interval_hr.
        t0_hr = (
            cluster_t[0].astype("datetime64[s]").astype(np.int64) // 3600
        )
        t0_aligned_hr = (
            (t0_hr + interval_hr - 1) // interval_hr
        ) * interval_hr
        t_aligned = np.datetime64(int(t0_aligned_hr * 3600), "s")
        # Annotate the first tick of the cluster with the cast range so
        # multi-cluster plots still convey what each block covers.
        first = True
        for j in range(64):
            tick_t = t_aligned + np.timedelta64(j * interval_hr, "h")
            if tick_t > cluster_t[-1]:
                break
            idx = int(np.argmin(np.abs(cluster_t - tick_t)))
            positions.append(float(cluster_x[idx]))
            label = _fmt_minute(tick_t)
            if first:
                label = f"{s + 1}-{e}\n{label}"
                first = False
            labels.append(label)
    return np.asarray(positions), labels


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
    p.add_argument("--apply-qc", dest="apply_qc", action="store_true",
                   default=True,
                   help="NaN bins where qc_drop_* is non-zero (default)")
    p.add_argument("--no-qc", dest="apply_qc", action="store_false",
                   help="ignore qc_drop_* and plot raw values "
                        "(only useful with drop_action: flag_only)")


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

    t_eps, depth, eps, eps_qc = _load_epsilon(diss_combo)

    if chi_combo_dir is not None:
        chi_combo_path = os.path.join(chi_combo_dir, "combo.nc")
        with xr.open_dataset(chi_combo_path) as peek:
            has_chi_mean = "chiMean" in peek.data_vars
        if has_chi_mean:
            # chiMean is on the combo: use it even if no chi_NN per-profile dir
            # exists (it only supplies the title attrs). Only fall back to the
            # legacy per-profile loader when chiMean is absent.
            t_chi, depth_chi, chi, chi_attrs, chi_qc = _load_chi_from_combo(
                chi_combo_path, chi_dir
            )
            # Re-bin chi onto eps's depth grid before any profile-axis work:
            # the chi-combo's bin centers are built independently and need not
            # match eps's, so plotting chi on `depth` without this would shift
            # it in depth (silent) or crash gamma=chi/eps on unequal bin counts.
            chi = _reindex_rows_to_depth(chi, depth_chi, depth)
            if chi_qc is not None:
                chi_qc = _reindex_rows_to_depth(
                    chi_qc, depth_chi, depth, reduce="max"
                )
        elif chi_dir is not None:
            t_chi, chi, chi_attrs, chi_qc = _load_chi_legacy_per_profile(
                chi_dir, depth
            )
        else:
            raise SystemExit(f"No chi_NN dir under {args.root}")
    elif chi_dir is not None:
        t_chi, chi, chi_attrs, chi_qc = _load_chi_legacy_per_profile(
            chi_dir, depth
        )
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
        aligned_qc = (
            np.full((chi.shape[0], len(t_eps)), np.nan) if chi_qc is not None else None
        )
        # Map each chi column to its nearest eps column (within 5 s).
        for j, s in enumerate(chi_secs):
            k = int(np.argmin(np.abs(eps_secs - s)))
            if abs(eps_secs[k] - s) <= 5:
                aligned[:, k] = chi[:, j]
                if aligned_qc is not None and chi_qc is not None:
                    aligned_qc[:, k] = chi_qc[:, j]
        n_unmatched = int(np.sum(~np.isfinite(aligned).any(axis=0)))
        print(
            f"chi/eps profile counts differ ({len(t_chi)} chi vs "
            f"{len(t_eps)} eps); aligned chi onto eps axis "
            f"({n_unmatched} eps slots have no chi)"
        )
        chi = aligned
        chi_qc = aligned_qc

    if args.apply_qc:
        n_eps_dropped = 0
        n_chi_dropped = 0
        if eps_qc is not None:
            mask = np.isfinite(eps_qc) & (eps_qc > 0)
            n_eps_dropped = int(mask.sum())
            eps = np.where(mask, np.nan, eps)
        if chi_qc is not None:
            mask = np.isfinite(chi_qc) & (chi_qc > 0)
            n_chi_dropped = int(mask.sum())
            chi = np.where(mask, np.nan, chi)
        if eps_qc is not None or chi_qc is not None:
            print(
                f"applied QC: NaN'd {n_eps_dropped} eps bins, "
                f"{n_chi_dropped} chi bins"
            )
    elif eps_qc is None and chi_qc is None:
        # User asked --no-qc but the dataset has no qc bitfield to skip.
        pass
    t = t_eps

    cast_x, segments, _centers, _t_starts, _t_ends = layout.compute_layout(
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

    def _safe_lognorm(vmn, vmx):
        # quantile_limits returns (None, None) for all-NaN / no-positive data;
        # LogNorm(None, None) then crashes on draw. Build a norm only for finite
        # positive limits, else None (-> "no data" placeholder).
        if vmn is None or vmx is None:
            return None
        if not (np.isfinite(vmn) and np.isfinite(vmx)) or vmn <= 0 or vmx <= 0:
            return None
        if vmn >= vmx:
            vmn = vmx / 10.0
        return LogNorm(vmin=vmn, vmax=vmx)

    panels = [
        (ax_e, eps, (eps_vmin, eps_vmax), r"$\varepsilon$  (W kg$^{-1}$)", r"$\varepsilon$"),
        (ax_c, chi, (chi_vmin, chi_vmax), r"$\chi$  (K$^{2}$ s$^{-1}$)", r"$\chi$"),
        (ax_g, gamma, (gam_vmin, gam_vmax),
         r"$\chi / \varepsilon$  (K$^{2}$ kg J$^{-1}$)", r"$\chi/\varepsilon$"),
    ]
    for ax, z, (vmn, vmx), cbar_label, short in panels:
        norm = _safe_lognorm(vmn, vmx)
        if norm is not None:
            layout.plot_panel(ax, fig, cast_x, segments, depth, z, cmap, norm, cbar_label)
        else:
            ax.text(0.5, 0.5, f"no finite {short} data", ha="center", va="center",
                    transform=ax.transAxes)
        ax.set_ylabel("Depth (m)")
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
    # chi_method tells us Method 1 (epsilon-driven) vs Method 2 (spectral fit).
    # Older chi NCs lack this attr -- treat as Method 1 for backwards compat.
    chi_method = chi_attrs.get("chi_method", "epsilon")
    if chi_method == "fit":
        fit_method = chi_attrs.get("fit_method", "iterative")
        method_phrase = rf"{fit_method}-fit $k_B+\chi$"
    else:
        method_phrase = r"$\varepsilon$-fixed $k_B$"
    qc_phrase = "QC applied" if args.apply_qc else "raw, no QC"
    title = args.title or os.path.basename(os.path.normpath(args.root))
    fig.suptitle(
        f"{title}   —   {lengths}   —   "
        rf"$\chi$: {spectrum} spectrum, {method_phrase}, FP07 {fp07}"
        f"   —   {qc_phrase}"
    )

    ax_e.invert_yaxis()

    # Lock x-limits to the cast layout so raw and QC plots are
    # directly comparable -- matplotlib otherwise auto-scales to the
    # extent of finite data, which differs between the two views.
    ax_e.set_xlim(cast_x[0] - 0.5, cast_x[-1] + 0.5)

    positions, labels = _time_ticks(t, cast_x, segments)
    ax_g.set_xticks(positions)
    ax_g.set_xticklabels(labels)
    ax_g.set_xticks(cast_x, minor=True)

    for label in ax_g.get_xticklabels():
        label.set_rotation(30)
        label.set_ha("right")

    out = args.out or os.path.join(args.root, "eps_chi_pcolor.png")
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}")
    return out

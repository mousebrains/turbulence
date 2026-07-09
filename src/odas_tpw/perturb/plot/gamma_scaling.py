# Jul-2026, Claude and Pat Welch, pat@mousebrains.com
"""Mixing-efficiency scaling figures (Lewin et al. 2025, Fig. 5).

``perturb-plot gamma-scaling`` sweeps a run's **per-window** products
(``chi_NN`` + ``diss_NN`` + ``profiles_NN``) and renders the flux
coefficient Gamma against the three dimensionless mixing parameters of
Lewin et al. (2025, https://doi.org/10.1175/JPO-D-25-0012.1):

- ``R_OT = L_O / L_T`` — Ozmidov over Thorpe scale (turbulence age proxy),
- ``Re_b = epsilon / (nu * N2_patch)`` — buoyancy Reynolds number,
- ``Ri_g = N2 / S2`` — gradient Richardson number (needs ``--adcp``).

Each panel: scatter of per-window values (colored by ``--sections``
windows when given — the analog of the paper's W1/W2), medians in
equispaced log-x bins with 5/95% whiskers, marginal pdfs, and the
literature slope guides (-1 and -4/3 on R_OT; -1/2 on Re_b; +1 on Ri_g).
A companion figure compares the two Thorpe-sort routes (JAC sigma0 vs
FP07 temperature) so the density route's ``L_T`` floor can be judged
against data rather than assumed.

Method notes (deliberate deviations from the paper are called out):

- **Gamma is recomputed from unmasked components** (``chiMean``, paired
  ``epsilonMean``, ``N2``, ``dTdz``) rather than read from the stored
  ``Gamma``: the pipeline's mixing product masks ``Gamma > 5`` and
  ``|dTdz| < 1e-4`` (sane for maps, but exactly the tails this figure
  exists to show — the paper's axis runs to 10 and its weak-gradient
  windows are *kept*, filtered instead by the Cox number). QC here is
  the paper's: ``Re_b > 20``, ``C_x > 50``, upper ``--min-depth`` cut,
  plus the pipeline's ``qc_drop_*`` hotel flags.
- **Thorpe scales** come from the slow (64 Hz) grid — both routes:
  ``sigma0`` (JAC CT; trusted salinity, so an upgrade on the paper's
  temperature-only sort) and the FP07 ``T1`` (the paper's proxy) — over
  a ``--sort-window`` (default 4 s, the paper's segment span) centered
  on each chi window.  ``N2_patch`` is the Smyth/Kaminski
  overturn-weighted form (see :mod:`odas_tpw.processing.thorpe`); when
  ``L_T`` is below the route's floor the window falls back to the
  chi-window background ``N2`` (the paper does the same below 0.05 m).
- **Our N2 is full TEOS-10** (temperature + JAC salinity); the paper's
  is temperature-only (their Eq. 4). Better physics, but medians can
  differ systematically where salinity stratifies or compensates.
- **Ri_g is a coarse-grained analog, not a clone**: shipboard-ADCP
  S2 over >= 2-m bins and ~120-s ensembles vs their 3-m pole ADCP at
  20-s cadence.  The numerator N2 is recomputed over ``--ri-n2-span``
  (default 4 m ~ 2 ADCP bins) so both Ri_g operands live at a matched
  vertical scale.  Multiple ``--adcp`` sonars are **never blended** —
  each gets its own panel (16-m os75 shear is not the same quantity as
  2-m wh300 shear).
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import xarray as xr

from odas_tpw.perturb import resolve
from odas_tpw.perturb.hotel import _dt64_to_epoch_s
from odas_tpw.perturb.plot import xaxis
from odas_tpw.perturb.plot.sections import (
    add_output_arguments,
    can_display,
    close_new_figs_on_error,
    closing_figs,
    fig_dpi,
    grouped,
    load_sections,
    save_or_show,
    select_sections,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

# Paper QC defaults (Lewin et al. 2025 section 2): discard Re_b <= 20 and
# Cox number <= 50; drop the ship-wake-contaminated upper water column.
DEFAULT_MIN_REB = 20.0
DEFAULT_MIN_COX = 50.0
DEFAULT_MIN_DEPTH = 10.0  # [dbar]

# diss->chi window pairing tolerance [s]. The pipeline writes both products
# on the same window grid (sub-ms offsets); 0.5 s tolerates a half-step
# offset without ever pairing across neighboring 1-s windows.
DEFAULT_MAX_DT = 0.5

# Vertical span [m] for the Ri_g numerator's N2 (~2 wh300 bins), so N2 and
# S2 live at a matched scale; 0 falls back to the chi-window N2.
DEFAULT_RI_N2_SPAN = 4.0

_GUIDES: dict[str, tuple[tuple[float, str], ...]] = {
    "R_OT": ((-1.0, "-1"), (-4.0 / 3.0, "-4/3")),
    "Re_b": ((-0.5, "-1/2"),),
    "Ri_g": ((1.0, "1"),),
}


@dataclass
class _Table:
    """Window-level columns concatenated over every profile."""

    cols: dict[str, np.ndarray] = field(default_factory=dict)
    s2: dict[str, np.ndarray] = field(default_factory=dict)  # per sonar
    rig: dict[str, np.ndarray] = field(default_factory=dict)  # per sonar
    n_profiles: int = 0

    def __getitem__(self, key: str) -> np.ndarray:
        return self.cols[key]

    @property
    def n(self) -> int:
        return self.cols["epoch"].size if self.cols else 0


def add_arguments(p: argparse.ArgumentParser) -> None:
    """Register the ``gamma-scaling`` flags on *p*."""
    p.add_argument(
        "--root",
        default=None,
        help="perturb output root (contains chi_NN/diss_NN/"
        "profiles_NN). Required unless --config is given.",
    )
    resolve.add_resolve_args(p)
    p.add_argument(
        "--sections",
        default=None,
        help="sections YAML; windows are colored by which section's "
        "start/stop window they fall in (only name/start/stop "
        "are used — the x-axis entries are ignored here)",
    )
    p.add_argument(
        "--select",
        action="append",
        default=None,
        metavar="NAME",
        help="use only the named section(s) for coloring (repeatable or comma-separated)",
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help="write PNGs here instead of showing on screen "
        "(default: display; headless falls back to --root)",
    )
    p.add_argument(
        "--adcp",
        action="append",
        default=None,
        metavar="FILE",
        help="CODAS gridded ADCP NetCDF (e.g. .../wh300/contour/"
        "wh300.nc) for the Ri_g panel; repeatable — each "
        "sonar gets its OWN panel (never blended)",
    )
    p.add_argument(
        "--min-pg",
        type=float,
        default=None,
        help="optional ADCP percent-good floor (default: none — "
        "the CODAS contour product is already edited)",
    )
    p.add_argument(
        "--time-tolerance",
        type=float,
        default=300.0,
        metavar="SEC",
        help="ADCP ensembles within this many seconds of a window "
        "are averaged for S2 (default 300)",
    )
    p.add_argument(
        "--sort-window",
        type=float,
        default=4.0,
        metavar="SEC",
        help="Thorpe sort-window span centered on each chi window "
        "(default 4 s — the paper's segment span; wider "
        "resolves larger overturns)",
    )
    p.add_argument(
        "--route",
        choices=("density", "temperature"),
        default="density",
        help="which Thorpe route feeds R_OT/Re_b: density "
        "(JAC sigma0; default) or temperature (FP07 T1, "
        "paper-style)",
    )
    p.add_argument(
        "--lt-floor-density",
        type=float,
        default=None,
        metavar="M",
        help="L_T floor for the sigma0 route (default 0.10 m — "
        "provisional; judge it from the route-comparison "
        "figure)",
    )
    p.add_argument(
        "--lt-floor-temperature",
        type=float,
        default=None,
        metavar="M",
        help="L_T floor for the T1 route (default 0.05 m, per the paper)",
    )
    p.add_argument(
        "--min-reb",
        type=float,
        default=DEFAULT_MIN_REB,
        help="discard windows with Re_b below this (default 20, per the paper; 0 disables)",
    )
    p.add_argument(
        "--min-cox",
        type=float,
        default=DEFAULT_MIN_COX,
        help="discard windows with Cox number below this (default 50, per the paper; 0 disables)",
    )
    p.add_argument(
        "--min-depth",
        type=float,
        default=DEFAULT_MIN_DEPTH,
        metavar="DBAR",
        help="discard windows shallower than this (default 10 — ship-wake contamination)",
    )
    p.add_argument(
        "--max-dt",
        type=float,
        default=DEFAULT_MAX_DT,
        metavar="SEC",
        help="diss->chi window pairing tolerance (default 0.5)",
    )
    p.add_argument(
        "--ri-n2-span",
        type=float,
        default=DEFAULT_RI_N2_SPAN,
        metavar="M",
        help="vertical span for the Ri_g numerator N2 (default "
        "4 m ~ 2 ADCP bins; 0 uses the chi-window N2)",
    )
    p.add_argument(
        "--min-run",
        type=float,
        default=10,
        metavar="N",
        help="Galbraith-Kelley coherence: an overturn counts as "
        "resolved only when >= N consecutive same-sign "
        "displacements occur in the window (default 10 slow "
        "samples ~ 10 cm at typical fall speed; noise flips "
        "sign every 1-2)",
    )
    p.add_argument(
        "--keep-truncated",
        action="store_true",
        help="keep window-edge-truncated overturns in R_OT "
        "(default: excluded — a clipped L_T is only a lower "
        "bound, biasing R_OT high)",
    )
    p.add_argument(
        "--max-profiles",
        type=int,
        default=0,
        metavar="N",
        help="sweep only the first N profiles (0 = all; for quick looks)",
    )
    p.add_argument(
        "--no-qc-flags",
        dest="qc_flags",
        action="store_false",
        default=True,
        help="ignore the pipeline's qc_drop_* hotel flags",
    )
    add_output_arguments(p)


# --------------------------------------------------------------------------- #
# sweep
# --------------------------------------------------------------------------- #


def _window_means(
    win_times: np.ndarray,
    half_width: float,
    t: np.ndarray,
    arrays: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """NaN-aware means of slow-grid *arrays* over each window (same time base)."""
    out = {k: np.full(win_times.size, np.nan) for k in arrays}
    for j, tau in enumerate(win_times):
        sel = np.abs(t - tau) <= half_width
        if not np.any(sel):
            continue
        for k, arr in arrays.items():
            w = arr[sel]
            w = w[np.isfinite(w)]
            if w.size:
                out[k][j] = float(np.mean(w))
    return out


def _pair_indices(src_t: np.ndarray, dst_t: np.ndarray, max_dt: float) -> np.ndarray:
    """Index of the nearest src window per dst window (-1 = unpaired).

    A single index pairing (rather than per-variable ``pair_nearest`` calls)
    guarantees every diss column — epsilon, its QC flag — comes from the SAME
    source window.
    """
    from odas_tpw.processing.mixing import pair_nearest

    idxf = pair_nearest(src_t, np.arange(src_t.size, dtype=float), dst_t, max_dt)
    idx = np.full(dst_t.size, -1, dtype=np.intp)
    ok = np.isfinite(idxf)
    idx[ok] = idxf[ok].astype(np.intp)
    return idx


def _gather(values: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """values[idx] with -1 mapped to NaN."""
    out = np.full(idx.size, np.nan)
    ok = idx >= 0
    out[ok] = np.asarray(values, dtype=float)[idx[ok]]
    return out


def _epoch(var: xr.DataArray) -> np.ndarray:
    """Decoded-time variable -> float epoch seconds (NaT -> NaN)."""
    return _dt64_to_epoch_s(np.atleast_1d(var.values))


def _sweep_profile(
    chi_path: str,
    diss_path: str,
    prof_path: str | None,
    args: argparse.Namespace,
    cols: dict[str, list],
    counters: dict[str, int],
) -> None:
    """Append one profile's window-level columns onto *cols*."""
    import gsw

    from odas_tpw.processing import mixing, thorpe

    with xr.open_dataset(chi_path) as c, xr.open_dataset(diss_path) as d:
        ct = _epoch(c["t"])
        n = ct.size
        lat = float(np.atleast_1d(c["lat"].values)[0])
        lon = float(np.atleast_1d(c["lon"].values)[0])
        chi = np.asarray(c["chiMean"].values, dtype=float)
        n2_win = np.asarray(c["N2"].values, dtype=float)
        dTdz = np.asarray(c["dTdz"].values, dtype=float)
        nu = np.asarray(c["nu"].values, dtype=float)
        P = np.asarray(c["P_mean"].values, dtype=float)
        T_mean = np.asarray(c["T_mean"].values, dtype=float)
        speed = np.asarray(c["speed"].values, dtype=float)
        qc_chi = (
            np.asarray(c["qc_drop_chi"].values, dtype=float)
            if "qc_drop_chi" in c.variables
            else np.zeros(n)
        )
        idx = _pair_indices(_epoch(d["t"]), ct, args.max_dt)
        eps = _gather(d["epsilonMean"].values, idx)
        qc_eps = (
            _gather(d["qc_drop_epsilon"].values, idx)
            if "qc_drop_epsilon" in d.variables
            else np.zeros(n)
        )

    half = args.sort_window / 2.0
    lt_s = np.full(n, np.nan)
    lt_t = np.full(n, np.nan)
    n2p_s = np.full(n, np.nan)
    n2p_t = np.full(n, np.nan)
    edge_s = np.zeros(n)
    edge_t = np.zeros(n)
    run_s = np.zeros(n)
    run_t = np.zeros(n)
    n2_ri = np.array(n2_win, dtype=float, copy=True)
    sp_mean = np.full(n, np.nan)
    if prof_path is not None:
        with xr.open_dataset(prof_path) as p:
            ts = _epoch(p["t_slow"])
            Ps = np.asarray(p["P"].values, dtype=float)
            have = {v for v in ("sigma0", "T1", "JAC_T", "SA", "CT", "SP") if v in p.variables}
            slow = {v: np.asarray(p[v].values, dtype=float) for v in have}

        means = _window_means(
            ct,
            half,
            ts,
            {k: slow[k] for k in ("SA", "CT", "SP") if k in slow},
        )
        sp_mean = means.get("SP", sp_mean)
        g = gsw.grav(lat, P)
        if "sigma0" in slow:
            res_s = thorpe.window_thorpe(
                ct, half, ts, Ps, slow["sigma0"], increasing_down=True, lat=lat
            )
            lt_s = res_s.L_T
            edge_s = res_s.edge_truncated.astype(float)
            run_s = res_s.max_run.astype(float)
            sig_means = _window_means(ct, half, ts, {"s0": slow["sigma0"]})
            rho0 = 1000.0 + sig_means["s0"]
            n2p_s = thorpe.patch_n2(res_s.rms_fluct, lt_s, g / rho0)
        if "T1" in slow:
            res_t = thorpe.window_thorpe(
                ct, half, ts, Ps, slow["T1"], increasing_down=False, lat=lat
            )
            lt_t = res_t.L_T
            edge_t = res_t.edge_truncated.astype(float)
            run_t = res_t.max_run.astype(float)
            if "SA" in means and "CT" in means:
                alpha = gsw.alpha(means["SA"], means["CT"], P)
                n2p_t = thorpe.patch_n2(res_t.rms_fluct, lt_t, alpha * g)

        # Matched-scale N2 for the Ri_g numerator: Thorpe-sorted background
        # over ~the ADCP differencing span, converted to a time half-width
        # via the cast's median fall speed.
        if args.adcp and args.ri_n2_span > 0 and "JAC_T" in slow:
            w_med = float(np.nanmedian(speed))
            if np.isfinite(w_med) and w_med > 0:
                strat = mixing.sorted_stratification(
                    ct,
                    (args.ri_n2_span / 2.0) / w_med,
                    ts,
                    Ps,
                    slow["JAC_T"],
                    S=slow.get("SP"),
                    lat=lat,
                    lon=lon,
                )
                n2_ri = strat.N2
    else:
        counters["no_profiles_file"] += 1

    # Gamma / Cox from UNMASKED components (see module docstring).
    from odas_tpw.scor160.ocean import kappa_T

    with np.errstate(divide="ignore", invalid="ignore"):
        ok_g = (
            np.isfinite(chi)
            & (chi > 0)
            & np.isfinite(eps)
            & (eps > 0)
            & np.isfinite(n2_win)
            & (n2_win > 0)
            & np.isfinite(dTdz)
            & (dTdz != 0)
        )
        gamma = np.where(
            ok_g,
            n2_win * chi / (2.0 * eps * np.where(ok_g, dTdz, 1.0) ** 2),
            np.nan,
        )
    sal = np.where(np.isfinite(sp_mean), sp_mean, 35.0)
    cox = thorpe.cox_number(chi, kappa_T(T_mean, sal, P), dTdz)

    cols["epoch"].append(ct)
    cols["P"].append(P)
    cols["lat"].append(np.full(n, lat))
    cols["lon"].append(np.full(n, lon))
    cols["eps"].append(eps)
    cols["nu"].append(nu)
    cols["chi"].append(chi)
    cols["N2_win"].append(n2_win)
    cols["N2_ri"].append(np.asarray(n2_ri, dtype=float))
    cols["dTdz"].append(dTdz)
    cols["Gamma"].append(gamma)
    cols["Cox"].append(cox)
    cols["LT_sigma"].append(np.asarray(lt_s, dtype=float))
    cols["LT_temp"].append(np.asarray(lt_t, dtype=float))
    cols["N2p_sigma"].append(np.asarray(n2p_s, dtype=float))
    cols["N2p_temp"].append(np.asarray(n2p_t, dtype=float))
    cols["edge_sigma"].append(edge_s)
    cols["edge_temp"].append(edge_t)
    cols["run_sigma"].append(run_s)
    cols["run_temp"].append(run_t)
    # Union of the two products' hotel-QC bitfields. Missing/unpaired flags
    # count as clean: an unpaired diss window already has eps=NaN, so its
    # Gamma never survives to the figure anyway.
    cols["qc"].append(
        np.where(np.isfinite(qc_chi), qc_chi, 0.0) + np.where(np.isfinite(qc_eps), qc_eps, 0.0)
    )


def sweep(args: argparse.Namespace) -> _Table:
    """Assemble the window-level table for the whole run (+ ADCP S2/Ri_g)."""
    import gsw

    from odas_tpw.perturb import adcp as adcp_mod
    from odas_tpw.processing import thorpe

    chi_dir = resolve.resolve_for_args(args, "chi")
    diss_dir = resolve.resolve_for_args(args, "diss")
    if chi_dir is None or diss_dir is None:
        raise SystemExit(
            "gamma-scaling needs per-profile chi_NN and diss_NN products "
            "(Gamma's ingredients); run the pipeline with chi enabled first"
        )
    prof_dir = resolve.resolve_for_args(args, "profiles", optional=True)
    if prof_dir is None:
        print(
            "no profiles_NN dir: Thorpe scales (R_OT) unavailable; Re_b uses the chi-window N2",
            file=sys.stderr,
        )

    files = sorted(os.path.basename(f) for f in glob.glob(os.path.join(chi_dir, "*.nc")))
    if args.max_profiles > 0:
        files = files[: args.max_profiles]
    if not files:
        raise SystemExit(f"no per-profile chi files under {chi_dir}")

    cols: dict[str, list] = {
        k: []
        for k in (
            "epoch",
            "P",
            "lat",
            "lon",
            "eps",
            "nu",
            "chi",
            "N2_win",
            "N2_ri",
            "dTdz",
            "Gamma",
            "Cox",
            "LT_sigma",
            "LT_temp",
            "N2p_sigma",
            "N2p_temp",
            "edge_sigma",
            "edge_temp",
            "run_sigma",
            "run_temp",
            "qc",
        )
    }
    counters = {"no_diss_file": 0, "no_profiles_file": 0}
    n_done = 0
    for f in files:
        diss_path = os.path.join(diss_dir, f)
        if not os.path.exists(diss_path):
            counters["no_diss_file"] += 1
            continue
        prof_path = (
            os.path.join(prof_dir, f)
            if prof_dir is not None and os.path.exists(os.path.join(prof_dir, f))
            else None
        )
        _sweep_profile(os.path.join(chi_dir, f), diss_path, prof_path, args, cols, counters)
        n_done += 1

    if n_done == 0:
        raise SystemExit("no chi/diss per-profile file pairs found")
    table = _Table(
        cols={k: np.concatenate(v) if v else np.empty(0) for k, v in cols.items()},
        n_profiles=n_done,
    )
    for name, count in counters.items():
        if count:
            print(f"note: {count} profile(s) skipped/degraded ({name})", file=sys.stderr)

    # Route selection + patch-N2 fallback (below the floor: background N2).
    floor_s = (
        args.lt_floor_density
        if args.lt_floor_density is not None
        else thorpe.DEFAULT_LT_FLOOR_DENSITY
    )
    floor_t = (
        args.lt_floor_temperature
        if args.lt_floor_temperature is not None
        else thorpe.DEFAULT_LT_FLOOR_TEMPERATURE
    )
    if args.route == "density":
        lt, n2p, floor = table["LT_sigma"], table["N2p_sigma"], floor_s
        edge, run = table["edge_sigma"], table["run_sigma"]
    else:
        lt, n2p, floor = table["LT_temp"], table["N2p_temp"], floor_t
        edge, run = table["edge_temp"], table["run_temp"]
    # A window's overturn is "resolved" (usable for L_T / N2_patch) when the
    # scale clears the route's noise floor, the displacement block is coherent
    # (Galbraith-Kelley run length — uncorrelated noise flips sign every 1-2
    # samples), and the overturn is NOT clipped by the sort window (a
    # truncated L_T is only a lower bound, which biases R_OT high; in weakly
    # stratified water saturation at ~the window span is the signature of
    # noise, not of a measured overturn).
    resolved = np.isfinite(lt) & (lt >= floor) & np.isfinite(n2p) & (n2p > 0)
    resolved &= run >= args.min_run
    if not args.keep_truncated:
        resolved &= edge == 0
    n2_bg = table["N2_win"]
    n2_scale = np.where(resolved, n2p, np.where(np.isfinite(n2_bg) & (n2_bg > 0), n2_bg, np.nan))
    table.cols["LT"] = np.where(resolved, lt, np.nan)
    table.cols["LT_floor_sigma"] = np.full(table.n, floor_s)
    table.cols["LT_floor_temp"] = np.full(table.n, floor_t)
    table.cols["resolved"] = resolved.astype(float)
    table.cols["N2_scale"] = n2_scale
    table.cols["Re_b"] = thorpe.reynolds_buoyancy(table["eps"], table["nu"], n2_scale)
    lo = thorpe.ozmidov(table["eps"], n2_scale)
    table.cols["R_OT"] = thorpe.r_ot(lo, table.cols["LT"])

    # Per-sonar S2 -> Ri_g (matched-scale numerator; sonars never blended).
    depth_m = np.asarray(-gsw.z_from_p(table["P"], table["lat"]), dtype=float)
    for path in args.adcp or []:
        src = adcp_mod.read_codas(path, min_pg=args.min_pg)
        ws = adcp_mod.window_shear(src, table["epoch"], depth_m, time_tolerance=args.time_tolerance)
        n2r = table["N2_ri"]
        with np.errstate(divide="ignore", invalid="ignore"):
            ok = np.isfinite(n2r) & (n2r > 0) & np.isfinite(ws.S2) & (ws.S2 > 0)
            table.rig[src.name] = np.where(ok, n2r / np.where(ok, ws.S2, 1.0), np.nan)
        table.s2[src.name] = ws.S2
    return table


# --------------------------------------------------------------------------- #
# QC + section coloring
# --------------------------------------------------------------------------- #


def qc_mask(table: _Table, args: argparse.Namespace) -> np.ndarray:
    """Paper-default QC: Re_b, Cox, depth, qc_drop flags, positive Gamma."""
    ok = np.isfinite(table["Gamma"]) & (table["Gamma"] > 0)
    ok &= table["P"] >= args.min_depth
    if args.min_reb > 0:
        ok &= np.isfinite(table["Re_b"]) & (table["Re_b"] > args.min_reb)
    if args.min_cox > 0:
        ok &= np.isfinite(table["Cox"]) & (table["Cox"] > args.min_cox)
    if args.qc_flags:
        ok &= table["qc"] == 0
    return np.asarray(ok)


def _section_colors(table: _Table, args: argparse.Namespace) -> tuple[np.ndarray, list[str]]:
    """(per-window section index, section names); index -1 = unsectioned."""
    idx = np.full(table.n, -1, dtype=int)
    if not args.sections:
        return idx, []
    sections = load_sections(args.sections)
    if args.select:
        sections = select_sections(sections, args.select)
    epoch = table["epoch"]
    for i, sec in enumerate(sections):
        m = np.ones(table.n, dtype=bool)
        if sec.start is not None:
            m &= epoch >= float(xaxis.to_epoch_seconds(np.array([sec.start]))[0])
        if sec.stop is not None:
            m &= epoch <= float(xaxis.to_epoch_seconds(np.array([sec.stop]))[0])
        idx[m & (idx < 0)] = i
    return idx, [s.name for s in sections]


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #


def _log_bins(x: np.ndarray, n_bins: int | None = None) -> np.ndarray:
    """Equispaced log bins across the robust (1-99%) range of *x*.

    The bin count adapts to the sample size (sqrt rule, clipped to 6..14) so
    sparse panels still accumulate the >= 5 points per bin the median/whisker
    overlay needs.
    """
    fin = x[np.isfinite(x) & (x > 0)]
    if n_bins is None:
        n_bins = int(np.clip(np.sqrt(fin.size), 6, 14)) if fin.size else 6
    lo, hi = np.percentile(fin, [1.0, 99.0]) if fin.size else (np.nan, np.nan)
    if not (np.isfinite(lo) and np.isfinite(hi)) or lo <= 0 or hi <= lo:
        lo, hi = 1e-1, 1e1
    return np.logspace(np.log10(lo), np.log10(hi), n_bins + 1)


def _binned_median(ax: Axes, x: np.ndarray, y: np.ndarray, bins: np.ndarray) -> None:
    """Gray squares: median y per log-x bin, 5/95% whiskers (paper style)."""
    for k in range(bins.size - 1):
        m = (x >= bins[k]) & (x < bins[k + 1]) & np.isfinite(y)
        if np.count_nonzero(m) < 5:
            continue
        xc = np.sqrt(bins[k] * bins[k + 1])
        med = float(np.median(y[m]))
        p5, p95 = np.percentile(y[m], [5.0, 95.0])
        ax.errorbar(
            xc,
            med,
            yerr=[[med - p5], [p95 - med]],
            fmt="s",
            color="0.25",
            mfc="0.85",
            mec="0.25",
            ms=5,
            elinewidth=0.9,
            capsize=2,
            zorder=5,
        )


def _marginal_pdf(
    ax: Axes, data: np.ndarray, bins: np.ndarray, color: Any, *, vertical: bool = False
) -> None:
    """Filled log-bin density on a marginal axis (paper-style pdfs)."""
    d = data[np.isfinite(data) & (data > 0)]
    if d.size < 5:
        return
    counts, edges = np.histogram(d, bins=bins)
    dens = counts / max(counts.sum(), 1) / np.diff(np.log10(edges))
    if vertical:
        ax.stairs(dens, edges, orientation="horizontal", fill=True, color=color, alpha=0.35)
    else:
        ax.stairs(dens, edges, fill=True, color=color, alpha=0.35)


def _slope_guides(
    ax: Axes, x: np.ndarray, y: np.ndarray, guides: tuple[tuple[float, str], ...]
) -> None:
    """Dashed slope-labeled reference lines anchored at the data median."""
    fin = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if np.count_nonzero(fin) < 5:
        return
    x0 = float(np.exp(np.median(np.log(x[fin]))))
    y0 = float(np.exp(np.median(np.log(y[fin]))))
    lo, hi = np.nanpercentile(x[fin], [1.0, 99.0])
    xs = np.logspace(np.log10(lo), np.log10(hi), 32)
    for slope, label in guides:
        ax.plot(xs, y0 * (xs / x0) ** slope, ls="--", lw=0.9, color="0.3", zorder=4)
        ax.annotate(
            label,
            (xs[-1], y0 * (xs[-1] / x0) ** slope),
            fontsize=8,
            color="0.3",
            textcoords="offset points",
            xytext=(3, 0),
        )


_PANEL_XLABEL = {
    "R_OT": r"$R_{OT} = L_O / L_T$",
    "Re_b": r"$\mathrm{Re}_b = \epsilon\,/\,(\nu N^2_{patch})$",
    "Ri_g": r"estimated local $\mathrm{Ri}_g = N^2 / S^2$",
}


def _fig_gamma(table: _Table, args: argparse.Namespace) -> Figure | None:
    """The Fig.-5 analog: Gamma vs R_OT / Re_b / Ri_g (one panel per sonar)."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    ok = qc_mask(table, args)
    gamma = np.where(ok, table["Gamma"], np.nan)
    panels: list[tuple[str, str, np.ndarray]] = [
        ("R_OT", _PANEL_XLABEL["R_OT"], np.where(ok, table["R_OT"], np.nan)),
        ("Re_b", _PANEL_XLABEL["Re_b"], np.where(ok, table["Re_b"], np.nan)),
    ]
    for name, rig in table.rig.items():
        panels.append(("Ri_g", f"{_PANEL_XLABEL['Ri_g']}  [{name}]", np.where(ok, rig, np.nan)))
    panels = [p for p in panels if np.count_nonzero(np.isfinite(p[2]) & np.isfinite(gamma)) >= 5]
    if not panels:
        print("gamma-scaling: no QC-passing windows to plot", file=sys.stderr)
        return None

    sec_idx, sec_names = _section_colors(table, args)
    cmap = plt.get_cmap("tab10")

    n_p = len(panels)
    figsize = args.figsize or (4.6 * n_p, 5.0)
    fig = plt.figure(figsize=figsize)
    outer = fig.add_gridspec(1, n_p, wspace=0.32, left=0.07, right=0.97, top=0.86, bottom=0.13)
    ylims = _gamma_limits(gamma)
    ybins = np.logspace(np.log10(ylims[0]), np.log10(ylims[1]), 25)

    for i, (kind, xlabel, x) in enumerate(panels):
        inner = outer[i].subgridspec(
            2, 2, width_ratios=(4, 1), height_ratios=(1, 4), hspace=0.06, wspace=0.06
        )
        ax = fig.add_subplot(inner[1, 0])
        ax_top = fig.add_subplot(inner[0, 0], sharex=ax)
        ax_right = fig.add_subplot(inner[1, 1], sharey=ax)
        fin = np.isfinite(x) & np.isfinite(gamma)
        if sec_names:
            un = sec_idx[fin & (sec_idx < 0)]
            if un.size:
                m = fin & (sec_idx < 0)
                ax.scatter(x[m], gamma[m], s=5, c="0.75", alpha=0.4, lw=0, rasterized=True)
            for si in range(len(sec_names)):
                m = fin & (sec_idx == si)
                if not m.any():
                    continue
                color = cmap(si % 10)
                ax.scatter(x[m], gamma[m], s=5, color=color, alpha=0.45, lw=0, rasterized=True)
                _marginal_pdf(ax_top, x[m], _log_bins(x[fin]), color)
                _marginal_pdf(ax_right, gamma[m], ybins, color, vertical=True)
        else:
            ax.scatter(x[fin], gamma[fin], s=5, color="tab:blue", alpha=0.35, lw=0, rasterized=True)
        _marginal_pdf(ax_top, x[fin], _log_bins(x[fin]), "0.5")
        _marginal_pdf(ax_right, gamma[fin], ybins, "0.5", vertical=True)
        bins = _log_bins(x[fin])
        _binned_median(ax, x, gamma, bins)
        _slope_guides(ax, x, gamma, _GUIDES[kind])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(*ylims)
        # Expand x to whole decades: a sub-decade log axis draws labeled
        # minor ticks that pile into an unreadable jumble.
        xf = x[fin & (x > 0)]
        if xf.size:
            ax.set_xlim(10 ** np.floor(np.log10(xf.min())), 10 ** np.ceil(np.log10(xf.max())))
        ax.set_xlabel(xlabel)
        if i == 0:
            ax.set_ylabel(r"$\Gamma = N^2\chi\,/\,(2\epsilon\,(dT/dz)^2)$")
        ax.grid(True, which="major", color="0.85", lw=0.5)
        ax.tick_params(labelsize=9)
        # sharex/sharey already propagate the log scales onto the marginals;
        # their own value axes are hidden entirely.
        ax_top.axis("off")
        ax_right.axis("off")

    if sec_names:
        handles = [
            Line2D([], [], marker="o", ls="", color=cmap(i % 10), label=name)
            for i, name in enumerate(sec_names)
        ]
        fig.legend(handles=handles, loc="upper right", fontsize=8, ncols=2, frameon=False)
    n_ok = int(np.count_nonzero(ok))
    title_id = os.path.basename(os.path.normpath(args.root))
    fig.suptitle(
        getattr(args, "title", None)
        or (
            f"{title_id} — mixing-efficiency scaling — {table.n_profiles} casts, "
            f"{grouped(n_ok)} QC-pass windows (of {grouped(table.n)}) — "
            f"Thorpe route: {args.route}"
        ),
        fontsize=11,
    )
    return fig


def _gamma_limits(gamma: np.ndarray) -> tuple[float, float]:
    """Paper axis (1e-3..1e1) stretched to cover the data quantiles."""
    fin = gamma[np.isfinite(gamma) & (gamma > 0)]
    lo, hi = 1e-3, 1e1
    if fin.size:
        lo = min(lo, 10 ** np.floor(np.log10(np.percentile(fin, 0.5))))
        hi = max(hi, 10 ** np.ceil(np.log10(np.percentile(fin, 99.5))))
    return lo, hi


def _fig_thorpe_compare(table: _Table, args: argparse.Namespace) -> Figure | None:
    """sigma0-route vs T1-route L_T: the floor-setting comparison figure."""
    import matplotlib.pyplot as plt

    lt_s = table["LT_sigma"]
    lt_t = table["LT_temp"]
    both = np.isfinite(lt_s) & (lt_s > 0) & np.isfinite(lt_t) & (lt_t > 0)
    if not both.any():
        print(
            "gamma-scaling: no windows with both Thorpe routes; route-comparison figure skipped",
            file=sys.stderr,
        )
        return None
    floor_s = float(table["LT_floor_sigma"][0])
    floor_t = float(table["LT_floor_temp"][0])

    fig, (ax, axh) = plt.subplots(1, 2, figsize=args.figsize or (10.0, 4.6))
    ax.scatter(lt_t[both], lt_s[both], s=4, alpha=0.3, lw=0, color="tab:blue", rasterized=True)
    lims = (1e-3, max(np.nanmax(lt_s[both]), np.nanmax(lt_t[both])) * 1.5)
    ax.plot(lims, lims, color="0.3", lw=0.8, ls="-", label="1:1")
    ax.axvline(floor_t, color="tab:red", ls="--", lw=0.9, label=f"T$_1$ floor {floor_t:g} m")
    ax.axhline(
        floor_s, color="tab:purple", ls="--", lw=0.9, label=rf"$\sigma_0$ floor {floor_s:g} m"
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*lims)
    ax.set_ylim(*lims)
    ax.set_xlabel(r"$L_T$ from FP07 $T_1$ (m)")
    ax.set_ylabel(r"$L_T$ from JAC $\sigma_0$ (m)")
    ax.grid(True, color="0.9", lw=0.5)
    ax.legend(fontsize=8, loc="upper left")

    bins = np.logspace(-3, np.log10(lims[1]), 40)
    for arr, floor, color, label in (
        (lt_s, floor_s, "tab:purple", r"$\sigma_0$ route"),
        (lt_t, floor_t, "tab:red", r"$T_1$ route"),
    ):
        fin = arr[np.isfinite(arr) & (arr > 0)]
        resolved = float(np.mean(fin >= floor)) if fin.size else 0.0
        axh.hist(
            fin,
            bins=bins,
            histtype="step",
            color=color,
            label=f"{label}: {resolved:.0%} of overturned windows $\\geq$ floor",
        )
        axh.axvline(floor, color=color, ls="--", lw=0.9)
    axh.set_xscale("log")
    axh.set_xlabel(r"$L_T$ (m)")
    axh.set_ylabel("windows")
    axh.legend(fontsize=8)
    axh.grid(True, color="0.9", lw=0.5)

    fallback = 1.0 - float(np.mean(table["resolved"]))
    title_id = os.path.basename(os.path.normpath(args.root))
    fig.suptitle(
        f"{title_id} — Thorpe-scale route comparison — route={args.route}: "
        f"{fallback:.0%} of windows fall back to background N$^2$ "
        f"(paper: 18%)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #


def build_figures(args: argparse.Namespace) -> Iterator[tuple[str, Any]]:
    """Yield the (stem, Figure) pairs: the scaling figure + route comparison."""
    args.root = resolve.require_root(args)
    if args.sort_window <= 0:
        raise SystemExit("--sort-window must be > 0")
    table = sweep(args)
    _report(table, args)
    for stem, builder in (
        ("gamma_scaling", _fig_gamma),
        ("gamma_thorpe_compare", _fig_thorpe_compare),
    ):
        with close_new_figs_on_error():
            fig = builder(table, args)
        if fig is not None:
            yield stem, fig


def _report(table: _Table, args: argparse.Namespace) -> None:
    """One-look sweep summary on stderr (coverage, QC attrition, fallback)."""
    ok = qc_mask(table, args)
    parts = [
        f"{table.n_profiles} casts, {grouped(table.n)} windows",
        f"QC pass {grouped(int(ok.sum()))}",
    ]
    for route, key in (("sigma0", "LT_sigma"), ("T1", "LT_temp")):
        lt = table[key]
        fin = np.isfinite(lt)
        if fin.any():
            parts.append(f"L_T[{route}] finite {float(fin.mean()):.0%}")
    edge_key = "edge_sigma" if args.route == "density" else "edge_temp"
    parts.append(
        f"patch-N2 resolved {float(np.mean(table['resolved'])):.0%} "
        f"(route {args.route}; {float(np.mean(table[edge_key] > 0)):.0%} "
        f"edge-truncated)"
    )
    for name, s2 in table.s2.items():
        cov = float(np.mean(np.isfinite(s2[ok]))) if ok.any() else 0.0
        parts.append(f"S2[{name}] covers {cov:.0%} of QC-pass windows")
    print("gamma-scaling: " + "; ".join(parts), file=sys.stderr)


def run(args: argparse.Namespace) -> str:
    """Render the scaling figures; show or write PNGs."""
    args.root = resolve.require_root(args)
    display = args.out_dir is None and can_display()
    if display:
        with closing_figs(build_figures(args)) as figs:
            shown = save_or_show(figs, None, fig_dpi(args))
        return f"displayed {shown} figure(s)"
    out_dir = args.out_dir or args.root
    os.makedirs(out_dir, exist_ok=True)
    with closing_figs(build_figures(args)) as figs:
        save_or_show(figs, out_dir, fig_dpi(args))
    return str(out_dir)

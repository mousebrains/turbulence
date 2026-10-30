# Jul-2026, Claude and Pat Welch, pat@mousebrains.com
"""Pure renderers for ``perturb-diag`` — each takes an Axes and stored arrays.

Kept free of interaction state so the same functions serve the interactive
inspector and headless snapshot rendering.  The overview mesh is drawn once;
the spectra and diagnostic-strip renderers clear and redraw their axes on every
cell selection.
"""

from __future__ import annotations

import numpy as np

from odas_tpw.perturb.diag.data import (
    C_REF,
    T_REF,
    Cell,
    ProfileFile,
    nearest_window,
)
from odas_tpw.perturb.plot import layout

# Per-probe colors, matching the rsi ProfileViewer palette.
_PROBE_COLORS = ("C0", "C1", "C4", "C5")


def draw_overview_mesh(ax, cast_x, segments, depth, z, cmap, norm):
    """Pcolor a depth-by-cast field on *ax*, one mesh per cluster (no colorbar).

    Mirrors :func:`layout.plot_panel` but returns the mesh without attaching a
    per-axis colorbar, so the inspector can share one colorbar across the three
    overview panels (they share a color scale).
    """
    z = layout.ffill_down(z)
    d_edges = layout.depth_edges(depth)
    pcm = None
    for s, e in segments:
        x_seg = cast_x[s:e]
        if len(x_seg) == 1:
            edges = np.array([x_seg[0] - 0.5, x_seg[0] + 0.5])
        else:
            mids = 0.5 * (x_seg[:-1] + x_seg[1:])
            edges = np.concatenate(([x_seg[0] - 0.5], mids, [x_seg[-1] + 0.5]))
        pcm = ax.pcolormesh(
            edges, d_edges, z[:, s:e], cmap=cmap, norm=norm, shading="flat"
        )
    return pcm


def band_extent(pf: ProfileFile, window: int) -> tuple[float, float]:
    """Depth interval [lo, hi] of the dissipation cell centered at *window*.

    Half-height is ``0.5 * (diss_length / fs_fast) * speed`` — the along-track
    length of the fft window converted to depth — matching the Matlab tool's
    grey band.  Falls back to +/-0.5 m when the length metadata is absent.
    """
    d0 = float(pf.depth[window])
    diss_len = pf.attrs.get("diss_length")
    fs = pf.attrs.get("fs_fast")
    spd = pf.speed[window] if window < pf.speed.size else np.nan
    if diss_len and fs and np.isfinite(spd):
        half = 0.5 * (float(diss_len) / float(fs)) * float(spd)
    else:
        half = 0.5
    return d0 - half, d0 + half


def draw_eps_spectra(ax, cell: Cell) -> None:
    """Shear wavenumber spectra with Nasmyth fits and K_max markers."""
    from odas_tpw.scor160.nasmyth import nasmyth

    ax.clear()
    pf, w = cell.profile, cell.window
    K = pf.K[:, w]
    kgood = np.isfinite(K) & (K > 0)
    if not kgood.any():
        ax.text(0.5, 0.5, "no spectrum", transform=ax.transAxes,
                ha="center", va="center")
        return

    for p in range(pf.n_probe):
        c = _PROBE_COLORS[p % len(_PROBE_COLORS)]
        shear = pf.spec_shear[p, :, w]
        nas = pf.spec_nasmyth[p, :, w]
        eps = pf.epsilon[p, w]
        kmax = pf.K_max[p, w]
        sgood = kgood & np.isfinite(shear) & (shear > 0)
        if sgood.any():
            ax.loglog(K[sgood], shear[sgood], "-", color=c, linewidth=1.5,
                      label=rf"$\epsilon_{p + 1}$={eps:.2g}")
        ngood = kgood & np.isfinite(nas) & (nas > 0)
        if ngood.any():
            ax.loglog(K[ngood], nas[ngood], "--", color=c, linewidth=0.8,
                      alpha=0.8)
        if np.isfinite(kmax) and sgood.any():
            ki = int(np.argmin(np.abs(K - kmax)))
            ax.loglog(kmax, shear[ki], marker="v", color=c, markersize=8,
                      zorder=5)

    # Faint reference Nasmyth spectra for orientation.
    nu = pf.nu[w]
    if np.isfinite(nu):
        for exp in (-10, -9, -8, -7, -6, -5):
            ax.loglog(K[kgood], nasmyth(10.0**exp, nu, K[kgood]),
                      color="0.85", linewidth=0.3, zorder=0)

    ax.set_xlabel("k (cpm)")
    ax.set_ylabel(r"$\Phi$(k)  (s$^{-2}$ cpm$^{-1}$)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=6, loc="lower left")

    em = pf.epsilon_mean[w]
    sig = pf.epsilon_ln_sigma[w]
    lo = f"{np.log10(em):.2f}" if np.isfinite(em) and em > 0 else "?"
    pm = f"{sig / np.log(10):.2f}" if np.isfinite(sig) else "?"
    meth = "/".join(
        "ISR" if pf.method[p, w] == 1 else "var" for p in range(pf.n_probe)
    )
    ax.set_title(
        rf"depth {pf.depth[w]:.1f} m   $\log_{{10}}\langle\epsilon\rangle$="
        rf"{lo}$\pm${pm}   method [{meth}]",
        fontsize=8,
    )


# Diagnostic strip: (label, per-probe 2-D array or None, per-window 1-D array,
# log-x flag).  Rendered left-to-right vs depth.
def draw_diss_strip(axes, cell: Cell) -> tuple[float, float]:
    """Per-profile dissipation diagnostics vs depth, with a cell depth band.

    Returns the (min, max) finite depth so the caller can set a shared,
    inverted y range.  Panels: epsilon (per probe + mean), FM, speed, nu, T.
    """
    pf, w = cell.profile, cell.window
    depth = pf.depth
    lo, hi = band_extent(pf, w)

    # Panel 0: per-probe epsilon (points) + mean (line), log-x.
    ax = axes[0]
    ax.clear()
    for p in range(pf.n_probe):
        c = _PROBE_COLORS[p % len(_PROBE_COLORS)]
        ax.plot(pf.epsilon[p], depth, ".", color=c, markersize=3,
                label=rf"$\epsilon_{p + 1}$")
    ax.plot(pf.epsilon_mean, depth, "-", color="k", linewidth=1.0,
            label=r"$\langle\epsilon\rangle$")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\epsilon$ (W kg$^{-1}$)")
    ax.set_ylabel("Depth (m)")
    ax.legend(fontsize=6, loc="best")

    # Remaining scalar panels.
    scalars = [
        ("FM", pf.FM, False),        # (probe, window)
        ("speed (m/s)", pf.speed, False),
        (r"$\nu$ (m$^2$/s)", pf.nu, False),
        ("T (°C)", pf.T_mean, False),
    ]
    for ax, (xlabel, arr, logx) in zip(axes[1:], scalars):
        ax.clear()
        if arr.ndim == 2:
            for p in range(pf.n_probe):
                c = _PROBE_COLORS[p % len(_PROBE_COLORS)]
                ax.plot(arr[p], depth, "-", color=c, linewidth=0.8)
        else:
            ax.plot(arr, depth, "-", color="C2", linewidth=0.8)
        if logx:
            ax.set_xscale("log")
        ax.set_xlabel(xlabel)

    # Shared grey depth band + grid on every panel.
    for ax in axes:
        ax.axhspan(lo, hi, color="0.5", alpha=0.25, zorder=0)
        ax.grid(True, alpha=0.3)

    finite = depth[np.isfinite(depth)]
    if finite.size:
        return float(finite.min()), float(finite.max())
    return 0.0, 1.0


# Raw instrument-diagnostics panels (vs depth): (xlabel, style, [(series,
# color, legend label), ...]).  A series is a key of RawProfile.raw: a channel,
# a CT-reference role (T_REF/C_REF: e.g. JAC_T/JAC_C or SBT1/SBC1 on a VMP, the
# hotel's sci_water_temp/cond on a glider) or a derived series (dT1_ref = T1 -
# T_REF).  "{T_ref}"/"{C_ref}" in a label become the channel the role resolved
# to, so the legend always names the reference actually plotted.
# Styles: "abs" plots the values as recorded; "delta" too, but on an x-range
# from the 0.5-99.5 percentiles (zero included) — on an ARCTERX-2023 VMP cast,
# 98% of T1 - JAC_T lay within -0.04..+0.006 K while a few surface samples
# autoscaled the axis to +/-0.2 K, flattening the offset the panel is for;
# "demean" subtracts the median (the inclinometer, so X and Y share a scale;
# median in the legend); "hp" overlays the HP+despiked signal on a faint HP
# trace (the microstructure channels).  Grouped as: water (T, FP07 - CT, C),
# flight (fall rate, attitude), and the epsilon path's sensors (vibration,
# shear, temp gradient).
_DIAG_PANELS: list[tuple[str, str, list[tuple[str, str, str]]]] = [
    ("T (°C)", "abs",
     [("T1", "C0", "T1"), ("T2", "C1", "T2"), (T_REF, "k", "{T_ref}")]),
    (r"$\delta$T (°C)", "delta",
     [("dT1_ref", "C0", "T1 - {T_ref}"), ("dT2_ref", "C1", "T2 - {T_ref}")]),
    # mS/cm: the unit the pipeline's salinity (gsw.SP_from_C) requires of C.
    (r"C (mS cm$^{-1}$)", "abs", [(C_REF, "C2", "{C_ref}")]),
    # W_slow is the smoothed dP/dt in dbar/s, signed (negative on a glider
    # climb, whatever its long_name's "|dP/dt|" says); speed_fast is in m/s.
    (r"W (dbar s$^{-1}$), speed (m s$^{-1}$)", "abs",
     [("W_slow", "C0", "W"), ("speed_fast", "C1", "speed")]),
    ("Incl (° dev.)", "demean", [("Incl_X", "C0", "X"), ("Incl_Y", "C1", "Y")]),
    ("Accel (counts)", "hp", [("Ax", "C0", "Ax"), ("Ay", "C1", "Ay")]),
    (r"Shear (s$^{-1}$)", "hp", [("sh1", "C0", "sh1"), ("sh2", "C1", "sh2")]),
    ("Temp grad", "hp", [("T1_dT1", "C0", "dT1"), ("T2_dT2", "C1", "dT2")]),
]

DIAG_PANEL_COUNT = len(_DIAG_PANELS)


def _stride(y: np.ndarray, depth: np.ndarray, maxn: int = 4000):
    """Uniformly stride a series to ~*maxn* points (fine for smooth signals)."""
    n = y.size
    if n <= maxn:
        return y, depth
    step = n // maxn + 1
    return y[::step], depth[::step]


def _envelope(y: np.ndarray, depth: np.ndarray, maxn: int = 4000):
    """Decimate to ~*maxn* points while preserving local min/max (keeps spikes).

    For each block of consecutive samples keep the block's minimum and maximum
    (in original order) so despiking-relevant extremes survive the reduction — a
    plain stride would drop the narrow spikes that the raw trace is meant to
    reveal.
    """
    n = y.size
    if n <= maxn:
        return y, depth
    step = int(np.ceil(n / (maxn / 2)))
    m = (n // step) * step
    if m < step:
        return y, depth
    yb = y[:m].reshape(-1, step)
    finite = np.isfinite(yb)
    has = finite.any(axis=1)
    rows = np.nonzero(has)[0]
    if rows.size == 0:
        return _stride(y, depth, maxn)
    safe = np.where(finite, yb, np.nan)
    base = rows * step
    amin = base + np.nanargmin(safe[rows], axis=1)
    amax = base + np.nanargmax(safe[rows], axis=1)
    idx = np.unique(np.concatenate([amin, amax, np.arange(m, n)]))
    return y[idx], depth[idx]


def _fill_label(label: str, names: dict[str, str]) -> str:
    """Replace each ``{role}`` in *label* with the channel that role resolved to."""
    for role, name in names.items():
        label = label.replace("{" + role + "}", name)
    return label


def draw_diag_strip(axes, cell: Cell) -> None:
    """Raw scalar and instrument diagnostics vs depth for the cell's cast.

    Panels (see :data:`_DIAG_PANELS`): temperature (T1/T2 and the CT), FP07
    minus CT, conductivity, fall rate and speed, inclinometer, accelerometer,
    shear and temperature gradient.  The microstructure channels overlay the
    high-pass-filtered signal (faint gray, min/max decimated so spikes survive)
    under the HP+despiked signal — the epsilon path's cleanup — in color; both
    are zero-mean and in-band, so the panel scales to the microstructure signal
    (not a channel's DC offset / temperature ramp) and the gray-to-color gap is
    exactly what despike removed.  The inclinometer is de-meaned so its two
    components share a scale (median tilt in the legend).  The scalar panels
    show values as recorded, delta-T on a robust x-range (its 0.5-99.5
    percentiles, zero included; a trace beyond it runs off the panel).  Every
    trace is clipped to the profile's window depth span, so the row lines up
    with the strip's shared inverted y-axis and drops surface/turnaround
    transients; a raw-only cell (no product file) shows the whole cast.  The
    gray band marks the selected window (a raw-only cell's clicked bin,
    +/-0.5 m).  A panel with none of its series reads "n/a".  Redraws all
    panels in place on each cell selection.
    """
    rp = cell.raw
    pf, w = cell.profile, cell.window
    clip_lo, clip_hi = -np.inf, np.inf
    if pf is not None:
        lo, hi = band_extent(pf, w)
        fin = pf.depth[np.isfinite(pf.depth)]
        if fin.size:
            d0, d1 = float(fin.min()), float(fin.max())
            margin = 0.05 * (d1 - d0) + 1.0
            clip_lo, clip_hi = d0 - margin, d1 + margin
    elif cell.depth is not None:
        lo, hi = cell.depth - 0.5, cell.depth + 0.5
    else:
        lo = hi = np.nan

    for ax, (xlabel, style, traces) in zip(axes, _DIAG_PANELS):
        ax.clear()
        drew = False
        shown: list[np.ndarray] = []  # "delta" values, for the robust x-range
        for ch, color, label in traces:
            if rp is None or ch not in rp.raw:
                continue
            fast = rp.is_fast[ch]
            depth = rp.depth_fast if fast else rp.depth_slow
            sel = np.isfinite(depth) & (depth >= clip_lo) & (depth <= clip_hi)
            if not sel.any() or not np.isfinite(rp.raw[ch][sel]).any():
                continue  # e.g. a hotel CT blanked over this whole cast
            depth_s = depth[sel]
            label = _fill_label(label, rp.names)
            if style == "hp" and ch in rp.proc:  # HP (faint) under HP+despike
                yy, dd = _envelope(rp.hp[ch][sel], depth_s)
                ax.plot(yy, dd, "-", color="0.75", lw=0.3, alpha=0.7, zorder=1)
                py, pdd = _stride(rp.proc[ch][sel], depth_s)
                ax.plot(py, pdd, "-", color=color, lw=0.7, label=label, zorder=2)
            elif style in ("abs", "delta"):  # as recorded; stride keeps NaN gaps
                yy, dd = _stride(rp.raw[ch][sel], depth_s)
                ax.plot(yy, dd, "-", color=color, lw=0.8, label=label, zorder=2)
                if style == "delta":
                    shown.append(rp.raw[ch][sel])
            else:  # inclinometer (or an HP channel too short to filter)
                raw_s = rp.raw[ch][sel]
                med = np.nanmedian(raw_s)
                raw_dm = raw_s - (med if np.isfinite(med) else 0.0)
                lab = f"{label} (μ={med:.1f})" if np.isfinite(med) else label
                yy, dd = _stride(raw_dm, depth_s)
                ax.plot(yy, dd, "-", color=color, lw=0.8, label=lab, zorder=2)
            drew = True
        if shown:
            q_lo, q_hi = np.nanpercentile(np.concatenate(shown), [0.5, 99.5])
            x0, x1 = min(float(q_lo), 0.0), max(float(q_hi), 0.0)
            pad = 0.1 * (x1 - x0) or 1e-3
            ax.set_xlim(x0 - pad, x1 + pad)
        if drew:
            ax.legend(fontsize=6, loc="best")
        else:
            if rp is None:
                msg = "no profiles file"
            elif not rp.raw:
                msg = "no raw data"
            else:
                msg = "n/a"  # the cast lacks this panel's channels
            ax.text(0.5, 0.5, msg, transform=ax.transAxes, ha="center",
                    va="center", fontsize=8, color="0.5")
            ax.set_xticks([])  # no data: the default 0-1 ticks would mislead
        ax.set_xlabel(xlabel)
        if np.isfinite(lo):
            ax.axhspan(lo, hi, color="0.5", alpha=0.25, zorder=0)
        ax.grid(True, alpha=0.3)
    if len(axes):
        axes[0].set_ylabel("Depth (m)")  # clear() dropped the one _build set


def draw_no_product(ax_spec, strip_axes, cell: Cell) -> None:
    """The spectra panel and product strip of a cell without its product file.

    The spectra panel says why there is nothing to fit (``cell.note``) and the
    product-strip panels read "n/a"; the raw row is drawn as usual by
    :func:`draw_diag_strip`.  A mixing cell that kept its diss file still shows
    the shear spectrum for a K_rho click (K_rho needs epsilon, not chi), at the
    window nearest the clicked depth, with the note above its title.
    """
    eps = (cell.profiles or {}).get("eps")
    w = (
        nearest_window(eps.depth, cell.depth)
        if eps is not None and cell.depth is not None else None
    )
    note = cell.note or "no product file"
    if w is not None and (cell.field or "").lower().startswith("k_rho"):
        draw_eps_spectra(ax_spec, Cell(profile=eps, window=w))
        ax_spec.set_title(f"{note}\n{ax_spec.get_title()}", fontsize=8)
    else:
        lines = [note]
        if cell.raw is not None:
            src = cell.raw.path.rsplit("/", 1)[-1]
            lines.append(f"raw channels below, from\n{src}")
        if w is not None:
            lines.append("click K_rho for the shear spectrum")
        _spectra_msg(ax_spec, "\n\n".join(lines))
    for ax in strip_axes:
        ax.clear()
        ax.text(0.5, 0.5, "n/a", transform=ax.transAxes, ha="center",
                va="center", fontsize=8, color="0.5")
        ax.set_xticks([])
        ax.grid(True, alpha=0.3)


def _geomean_over_probes(x: np.ndarray) -> np.ndarray:
    """Geometric mean across the probe axis (axis 0), ignoring non-positive/NaN."""
    with np.errstate(invalid="ignore", divide="ignore"):
        pos = np.where(x > 0, x, np.nan)
        return np.asarray(np.exp(np.nanmean(np.log(pos), axis=0)))


def _fp07_h2(
    K: np.ndarray, speed: float, attrs: dict, probe: str | None = None,
    response: tuple[np.ndarray, np.ndarray] | None = None,
    obs: np.ndarray | None = None, noise: np.ndarray | None = None,
) -> np.ndarray:
    """FP07 response |H|^2 on the wavenumber grid, or all-ones when unavailable.

    Reconstructs the response the chi fit divided out, for one probe: frequency
    ``f = K*speed`` (Hz), the speed-dependent ``tau`` paired with the stored
    ``fp07_model`` (single_pole -> Lueck, double_pole -> Goto; see
    :func:`odas_tpw.chi.fp07.default_tau_model`) times the probe's
    ``fp07_tau_scale_<probe>`` when the product records one, then the single-
    or double-pole ``|H|^2``, times an applied fp07-pair correction e
    (*response* = (f at W_ref, ln e) for this probe) when the product carries
    one. ``spec_batch`` is stored *without* this rolloff, so the model that
    matches the raw observed spectrum is ``spec_batch*|H|^2 + spec_noise``.
    (chi divided the raw spectrum and noise by e and fitted the default model;
    multiplying back gives this same curve against the stored raw data.) chi
    weights ln e by each bin's S/N (``fp07_pair.signal_weighted``); pass the
    window's *obs* and *noise* spectra to reproduce that exactly. A bead's slow
    response path (``fp07_slow_response_beta_<probe>`` / ``_tau_s_ms_<probe>``)
    multiplies in too, unweighted, as chi applied it.
    """
    from odas_tpw.chi.fp07 import (
        default_tau_model,
        fp07_double_pole,
        fp07_tau,
        fp07_transfer,
    )

    model = str(attrs.get("fp07_model", "")) if attrs else ""
    if not model or not np.isfinite(speed) or speed <= 0:
        return np.ones_like(K)
    tau_model = default_tau_model(model)
    scale = float(attrs.get(f"fp07_tau_scale_{probe}", 1.0)) if probe else 1.0
    tau0 = float(fp07_tau(float(speed), tau_model)) * scale
    f = np.asarray(K, dtype=float) * float(speed)
    tf = fp07_double_pole if model == "double_pole" else fp07_transfer
    h2 = np.asarray(tf(f, tau0), dtype=float)
    if response is not None and int(attrs.get("fp07_response_applied", 0)):
        from odas_tpw.chi.fp07_pair import reduced_frequency

        f_ref, ln_e = response
        w_ref = float(attrs.get("fp07_response_W_ref", speed))
        x = reduced_frequency(f, float(speed), w_ref, str(attrs.get("fp07_response_tau_model",
                                                                     tau_model)))
        lx = np.log10(np.clip(x, f_ref[0], f_ref[-1]))
        e = np.exp(np.interp(lx, np.log10(f_ref), ln_e))
        if obs is not None and noise is not None:
            from odas_tpw.chi.fp07_pair import signal_weighted

            # chi's S/N running mean sees only the bins from first_bin up (a
            # product without the attr used bin 1 and the whole grid).
            first_bin = int(attrs.get("first_bin", 1))
            usable = None
            if first_bin > 1:
                from odas_tpw.chi.chi import _usable_bins

                usable = _usable_bins(np.asarray(K, dtype=float), first_bin)
            e = signal_weighted(e, obs, noise, usable=usable)
        h2 = h2 * e
    # A bead's slow response path (chi.fp07_slow), recorded per probe only where
    # it was applied. chi divided raw S and N by it in every bin (unweighted),
    # so against the raw data the model's response carries it as a factor.
    beta = attrs.get(f"fp07_slow_response_beta_{probe}") if (attrs and probe) else None
    if beta is not None:
        from odas_tpw.chi.fp07_slow import slow_h2

        tau_ms = float(attrs.get(f"fp07_slow_response_tau_s_ms_{probe}", np.nan))
        h2 = h2 * slow_h2(f, float(beta), tau_ms / 1000.0)
    return h2


def draw_chi_spectra(ax, cell: Cell) -> None:
    """Temperature-gradient wavenumber spectra with the fitted model + noise floor.

    The chi analog of :func:`draw_eps_spectra`: per probe the observed gradient
    spectrum (solid) and, dashed, the fitted model that it is compared against —
    the Batchelor/Kraichnan spectrum rolled off by the FP07 response and lifted
    by the electronics-noise floor (``spec_batch*|H|^2 + spec_noise``), the same
    combination the fom integrates.  The noise floor alone is dotted, and the
    upper fit wavenumber ``K_max_T`` marked.  (Plotting the stored ``spec_batch``
    directly would omit the FP07 rolloff and the floor, so the "model" would sit
    well off the data — the fit only looks wrong without them.)
    """
    ax.clear()
    cf, w = cell.profile, cell.window
    K = cf.K[:, w]
    kgood = np.isfinite(K) & (K > 0)
    if not kgood.any():
        ax.text(0.5, 0.5, "no spectrum", transform=ax.transAxes,
                ha="center", va="center")
        return

    for p in range(cf.n_probe):
        c = _PROBE_COLORS[p % len(_PROBE_COLORS)]
        probe = cf.probes[p] if p < len(cf.probes) else None
        resp = (
            (cf.response_f, cf.response_ln_e[p])
            if cf.response_f is not None and cf.response_ln_e is not None
            else None
        )
        obs = cf.spec_gradT[p, :, w]
        noise = cf.spec_noise[p, :, w]
        h2 = _fp07_h2(K, float(cf.speed[w]), cf.attrs, probe=probe, response=resp,
                      obs=obs, noise=noise)
        model = cf.spec_batch[p, :, w] * h2 + noise
        og = kgood & np.isfinite(obs) & (obs > 0)
        if og.any():
            ax.loglog(K[og], obs[og], "-", color=c, linewidth=1.5,
                      label=rf"$\chi_{p + 1}$={cf.chi[p, w]:.2g}")
        mg = kgood & np.isfinite(model) & (model > 0)
        if mg.any():
            ax.loglog(K[mg], model[mg], "--", color=c, linewidth=1.1, alpha=0.9)
        ng = kgood & np.isfinite(noise) & (noise > 0)
        if ng.any():
            ax.loglog(K[ng], noise[ng], ":", color=c, linewidth=0.6, alpha=0.5)
        kmax = cf.K_max_T[p, w]
        if np.isfinite(kmax) and og.any():
            ki = int(np.argmin(np.abs(K - kmax)))
            ax.loglog(kmax, obs[ki], marker="v", color=c, markersize=8, zorder=5)

    # Focus the y-range on the observed spectrum within the fit band; the model's
    # tail beyond K_max can run orders of magnitude above the rolled-off data.
    kmax_all = cf.K_max_T[:, w]
    k_hi = float(np.nanmax(kmax_all)) if np.isfinite(kmax_all).any() else np.inf
    infit = kgood & np.less_equal(K, k_hi)
    vals = cf.spec_gradT[:, infit, w] if infit.any() else cf.spec_gradT[:, kgood, w]
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size:
        ax.set_ylim(float(vals.min()) * 0.3, float(vals.max()) * 3.0)

    ax.set_xlabel("k (cpm)")
    ax.set_ylabel(r"$\Psi$(k)  (K$^2$ m$^{-1}$)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=6, loc="lower left")

    cm = float(_geomean_over_probes(cf.chi[:, w]))
    lo = f"{np.log10(cm):.2f}" if np.isfinite(cm) and cm > 0 else "?"
    fom = "/".join(f"{cf.fom[p, w]:.2f}" for p in range(cf.n_probe))
    spectrum = str(cf.attrs.get("spectrum_model", "")).capitalize()
    method = str(cf.attrs.get("fit_method", ""))
    tag = " · ".join(t for t in (spectrum, f"{method} fit" if method else "") if t)
    title = (
        rf"depth {cf.depth[w]:.1f} m   $\log_{{10}}\langle\chi\rangle$={lo}"
        rf"   fom [{fom}]"
    )
    if tag:
        title += f"   {tag}"
    ax.set_title(title, fontsize=8)


def draw_chi_strip(axes, cell: Cell) -> tuple[float, float]:
    """Per-profile chi diagnostics vs depth: chi (per probe + geomean), fom,
    speed, nu, T.  Mirrors :func:`draw_diss_strip` (which see for the shared
    grey band / inverted-depth contract)."""
    cf, w = cell.profile, cell.window
    depth = cf.depth
    lo, hi = band_extent(cf, w)

    ax = axes[0]
    ax.clear()
    for p in range(cf.n_probe):
        c = _PROBE_COLORS[p % len(_PROBE_COLORS)]
        ax.plot(cf.chi[p], depth, ".", color=c, markersize=3,
                label=rf"$\chi_{p + 1}$")
    ax.plot(_geomean_over_probes(cf.chi), depth, "-", color="k", linewidth=1.0,
            label=r"$\langle\chi\rangle$")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\chi$ (K$^2$ s$^{-1}$)")
    ax.set_ylabel("Depth (m)")
    ax.legend(fontsize=6, loc="best")

    scalars = [
        ("fom", cf.fom, False),          # (probe, window)
        ("speed (m/s)", cf.speed, False),
        (r"$\nu$ (m$^2$/s)", cf.nu, False),
        ("T (°C)", cf.T_mean, False),
    ]
    for ax, (xlabel, a, logx) in zip(axes[1:], scalars):
        ax.clear()
        if a.ndim == 2:
            for p in range(cf.n_probe):
                c = _PROBE_COLORS[p % len(_PROBE_COLORS)]
                ax.plot(a[p], depth, "-", color=c, linewidth=0.8)
        else:
            ax.plot(a, depth, "-", color="C2", linewidth=0.8)
        if logx:
            ax.set_xscale("log")
        ax.set_xlabel(xlabel)

    for ax in axes:
        ax.axhspan(lo, hi, color="0.5", alpha=0.25, zorder=0)
        ax.grid(True, alpha=0.3)

    finite = depth[np.isfinite(depth)]
    if finite.size:
        return float(finite.min()), float(finite.max())
    return 0.0, 1.0


def _spectra_msg(ax, msg: str) -> None:
    ax.clear()
    ax.text(0.5, 0.5, msg, transform=ax.transAxes, ha="center", va="center",
            fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def draw_mixing_spectra(ax, cell: Cell) -> None:
    """Adaptive drill-down spectrum, keyed off the clicked overview field.

    Per the mixing decomposition, K_rho comes from shear (epsilon) and K_T from
    the temperature gradient (chi), so clicking a K_rho cell shows that cast's
    shear spectrum (from the diss file), a K_T cell its temperature-gradient
    Batchelor spectrum (from the chi file), and Gamma — which couples both — a
    note pointing at the two.  The chi profile is primary, so the window's depth
    picks the matching diss window.
    """
    field = (cell.field or "").lower()
    profs = cell.profiles or {}
    chi = cell.profile  # primary
    depth = (
        float(chi.depth[cell.window])
        if cell.window < chi.depth.size else float("nan")
    )

    if field.startswith("k_rho"):
        eps = profs.get("eps")
        if eps is None:
            _spectra_msg(ax, "no epsilon file matched this cast")
            return
        w = nearest_window(eps.depth, depth)
        if w is None:
            _spectra_msg(ax, "no finite epsilon window for this cast")
            return
        draw_eps_spectra(ax, Cell(profile=eps, window=w))
    elif field.startswith("k_t"):
        draw_chi_spectra(ax, Cell(profile=chi, window=cell.window))
    else:  # Gamma (or anything else): no single spectrum — it couples eps & chi
        _spectra_msg(
            ax,
            r"$\Gamma = N^2\,\chi\,/\,(2\,\epsilon\,(dT/dz)^2)$"
            "\n\nclick K_rho for the shear spectrum,\n"
            "K_T for the temperature-gradient spectrum",
        )


def _strip_panel(ax, x, depth, xlabel, lo, hi, *, logx=False, color="C2"):
    """One mixing-strip panel: *x* vs *depth*, or an 'n/a' note when absent."""
    ax.clear()
    xa = None if x is None else np.asarray(x, dtype=float)
    if xa is None or not np.isfinite(xa).any():
        ax.text(0.5, 0.5, "n/a", transform=ax.transAxes, ha="center",
                va="center", fontsize=8, color="0.5")
    else:
        ax.plot(xa, depth, "-", color=color, linewidth=0.9)
        if logx:
            ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.axhspan(lo, hi, color="0.5", alpha=0.25, zorder=0)
    ax.grid(True, alpha=0.3)


def draw_mixing_strip(axes, cell: Cell) -> tuple[float, float]:
    """Mixing diagnostics vs depth: epsilon, chi, N^2, dT/dz, K_T, K_rho, Gamma.

    Epsilon comes from the diss file (per-window mean, on its own grid); the rest
    come from the chi file, including the mixing quantities a mixing run appended
    (N2/dTdz/K_T/K_rho/Gamma) — those panels read 'n/a' when the run had no CTD.
    Mirrors :func:`draw_diss_strip`'s shared grey-band / inverted-depth contract.
    """
    chi = cell.profile
    eps = (cell.profiles or {}).get("eps")
    d_chi = chi.depth
    lo, hi = band_extent(chi, cell.window)

    axes[0].set_ylabel("Depth (m)")
    e_x = eps.epsilon_mean if eps is not None else None
    d_eps = eps.depth if eps is not None else d_chi
    _strip_panel(axes[0], e_x, d_eps, r"$\epsilon$ (W kg$^{-1}$)", lo, hi,
                 logx=True, color="C0")
    _strip_panel(axes[1], _geomean_over_probes(chi.chi), d_chi,
                 r"$\chi$ (K$^2$ s$^{-1}$)", lo, hi, logx=True, color="C1")
    _strip_panel(axes[2], chi.N2, d_chi, r"$N^2$ (s$^{-2}$)", lo, hi,
                 logx=True, color="C2")
    _strip_panel(axes[3], chi.dTdz, d_chi, r"dT/dz (K m$^{-1}$)", lo, hi,
                 color="C3")
    _strip_panel(axes[4], chi.K_T, d_chi, r"$K_T$ (m$^2$ s$^{-1}$)", lo, hi,
                 logx=True, color="C4")
    _strip_panel(axes[5], chi.K_rho, d_chi, r"$K_\rho$ (m$^2$ s$^{-1}$)", lo, hi,
                 logx=True, color="C5")
    _strip_panel(axes[6], chi.Gamma, d_chi, r"$\Gamma$", lo, hi, color="C6")

    spans = [d_chi[np.isfinite(d_chi)]]
    if eps is not None:
        spans.append(np.asarray(d_eps)[np.isfinite(d_eps)])
    alld = np.concatenate(spans)
    if alld.size:
        return float(alld.min()), float(alld.max())
    return 0.0, 1.0

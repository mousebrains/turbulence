# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Shared base class for interactive profile viewers (quick_look, diss_look).

ProfileViewer handles common setup (channel extraction, profile detection,
speed interpolation, shear conversion), navigation buttons, and shared
drawing methods.  Subclasses define their panel layout by overriding
``_setup_axes`` and ``_draw``.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button

from odas_tpw.rsi.helpers import AC_PATTERN, DT_PATTERN, SH_PATTERN, T_PATTERN
from odas_tpw.rsi.profile import _smooth_fall_rate, get_profiles
from odas_tpw.rsi.window import compute_chi_window, compute_eps_window
from odas_tpw.scor160.despike import despike
from odas_tpw.scor160.nasmyth import nasmyth
from odas_tpw.scor160.ocean import visc35

# ---------------------------------------------------------------------------
# Shared free functions
# ---------------------------------------------------------------------------


def select_mid_window(
    P_fast: np.ndarray, sel: slice, fft_length: int, diss_length: int | None = None
) -> slice:
    """Select the single diss_length window closest to the midpoint pressure.

    Returns a slice for the selected window, or *sel* unchanged if the
    segment is too short for even one window.
    """
    if diss_length is None:
        diss_length = 4 * fft_length
    n_fast = sel.stop - sel.start
    overlap = diss_length // 2
    step = diss_length - overlap
    n_windows = max(0, 1 + (n_fast - diss_length) // step)
    if n_windows == 0:
        return sel

    P_mid = float(np.mean(P_fast[sel]))
    best_idx = 0
    best_dist = float("inf")
    for widx in range(n_windows):
        s = sel.start + widx * step
        e = s + diss_length
        if e > sel.stop:
            break
        dist = abs(float(np.mean(P_fast[s:e])) - P_mid)
        if dist < best_dist:
            best_dist = dist
            best_idx = widx
    w_start = sel.start + best_idx * step
    w_end = min(w_start + diss_length, sel.stop)
    return slice(w_start, w_end)


def compute_depth_spectra(
    shear_data,
    accel_data,
    therm_data,
    diff_gains,
    P_fast,
    T_slow,
    speed_fast,
    fs_fast,
    sel,
    fft_length,
    f_AA,
    do_goodman,
    diss_length=None,
) -> dict:
    """Compute shear + chi spectra at one depth for spectral panels.

    Selects a single diss_length window closest to the midpoint pressure,
    matching MATLAB tpw_plot_spectra.

    Returns dict with shear spectra, Nasmyth fits, chi gradient spectra,
    and both Method 1/Method 2 model fits.
    """
    if diss_length is None:
        diss_length = 4 * fft_length
    n_shear = len(shear_data)
    n_therm = len(therm_data)
    n_freq = fft_length // 2 + 1

    # Select a single diss_length window closest to the midpoint pressure
    w_sel = select_mid_window(P_fast, sel, fft_length, diss_length)
    n_win = w_sel.stop - w_sel.start

    mean_T = float(np.mean(T_slow))

    result = {
        "K": np.array([]),
        "F": np.array([]),
        "W": 0.0,
        "nu": 0.0,
        "shear_specs": [],
        "nasmyth_specs": [],
        "epsilons": [],
        "methods": [],
        "K_maxes": [],
        "chi_obs_specs": [],
        "chi_m1_specs": [],
        "chi_m2_specs": [],
        "chi_m1_vals": [],
        "chi_m2_vals": [],
        "noise_K": None,
    }

    if n_win < diss_length:
        return result

    # --- Epsilon via shared compute_eps_window ---
    if n_shear > 0:
        sh_seg = np.column_stack([d[w_sel] for _, d in shear_data])
        ac_seg = np.column_stack([d[w_sel] for _, d in accel_data]) if accel_data else None
        er = compute_eps_window(
            sh_seg,
            ac_seg,
            speed_fast[w_sel],
            P_fast[w_sel],
            mean_T,
            fs_fast,
            fft_length,
            f_AA,
            do_goodman,
        )
        result["K"] = er.K
        result["F"] = er.F
        result["W"] = er.W
        result["nu"] = er.nu
        result["shear_specs"] = er.shear_specs
        result["nasmyth_specs"] = er.nasmyth_specs
        result["epsilons"] = list(er.epsilon)
        result["foms"] = list(er.fom)
        result["FMs"] = list(er.FM)
        result["methods"] = list(er.method)
        result["K_maxes"] = list(er.K_max)
        result["epsilons_isr"] = list(er.epsilon_isr)
        result["epsilons_var"] = list(er.epsilon_var)
    else:
        W = float(np.mean(np.abs(speed_fast[w_sel])))
        if W < 0.01:
            W = 0.01
        nu = float(visc35(mean_T))
        F = np.arange(n_freq) * fs_fast / fft_length
        result["K"] = F / W
        result["F"] = F
        result["W"] = W
        result["nu"] = nu

    # --- Chi via shared compute_chi_window (M1 + M2) ---
    f_AA_chi = 0.9 * f_AA
    W = result["W"]  # type: ignore[assignment]  # dict has mixed value types
    nu = result["nu"]  # type: ignore[assignment]

    if n_therm > 0:
        therm_segs = [therm_data[ci][1][w_sel] for ci in range(n_therm)]
        eps_arr = np.array(result["epsilons"]) if result["epsilons"] else None
        # The 1.15 QC limit is defined for the MAD-based Lueck FM
        # statistic, not the variance-ratio fom.
        fom_arr = np.array(result.get("FMs", [])) if result.get("FMs") else None

        # Method 1: chi from epsilon (kB fixed by FOM-filtered mean epsilon)
        cr_m1 = compute_chi_window(
            therm_segs,
            diff_gains,
            W,
            mean_T,
            nu,
            fs_fast,
            fft_length,
            f_AA_chi,
            spectrum_model="kraichnan",
            epsilon=eps_arr,
            fom=fom_arr,
            method=1,
        )
        result["chi_obs_specs"] = cr_m1.grad_specs
        result["chi_m1_specs"] = cr_m1.model_specs
        result["chi_m1_vals"] = list(cr_m1.chi)
        result["noise_K"] = cr_m1.noise_K

        # Method 2: iterative fit (kB fitted to observed spectrum)
        cr_m2 = compute_chi_window(
            therm_segs,
            diff_gains,
            W,
            mean_T,
            nu,
            fs_fast,
            fft_length,
            f_AA_chi,
            spectrum_model="kraichnan",
            method=2,
        )
        result["chi_m2_specs"] = cr_m2.model_specs
        result["chi_m2_vals"] = list(cr_m2.chi)

    return result


def compute_windowed_diss(
    shear_data,
    accel_data,
    therm_data,
    diff_gains,
    P_fast,
    T_slow,
    speed_fast,
    fs_fast,
    sel,
    fft_length,
    f_AA,
    do_goodman,
    diss_length=None,
) -> dict:
    """Compute windowed epsilon and chi (both Batchelor and Kraichnan) estimates.

    Returns dict with P_windows, epsilon, FM, fom, mad, K_max_ratio,
    chi_batchelor, chi_kraichnan arrays.
    """
    if diss_length is None:
        diss_length = 4 * fft_length
    overlap = diss_length // 2
    step = diss_length - overlap

    seg_start = sel.start
    seg_end = sel.stop
    N = seg_end - seg_start

    n_shear = len(shear_data)
    n_therm = len(therm_data)
    n_windows = max(0, 1 + (N - diss_length) // step)

    empty = {
        "P_windows": np.array([]),
        "epsilon": np.empty((n_shear, 0)),
        "FM": np.empty((n_shear, 0)),
        "fom": np.empty((n_shear, 0)),
        "mad": np.empty((n_shear, 0)),
        "K_max_ratio": np.empty((n_shear, 0)),
        "chi_batchelor": np.empty((n_therm, 0)),
        "chi_kraichnan": np.empty((n_therm, 0)),
        "chi_fom_batchelor": np.empty((n_therm, 0)),
        "chi_fom_kraichnan": np.empty((n_therm, 0)),
    }
    if n_windows == 0:
        return empty

    P_windows = np.full(n_windows, np.nan)
    eps_arr = np.full((n_shear, n_windows), np.nan)
    FM_arr = np.full((n_shear, n_windows), np.nan)
    fom_arr = np.full((n_shear, n_windows), np.nan)
    mad_arr = np.full((n_shear, n_windows), np.nan)
    Kmr_arr = np.full((n_shear, n_windows), np.nan)
    chi_batch_arr = np.full((n_therm, n_windows), np.nan)
    chi_kraich_arr = np.full((n_therm, n_windows), np.nan)
    chi_fom_batch_arr = np.full((n_therm, n_windows), np.nan)
    chi_fom_kraich_arr = np.full((n_therm, n_windows), np.nan)

    mean_T = float(np.mean(T_slow))
    f_AA_chi = 0.9 * f_AA

    for idx in range(n_windows):
        s = seg_start + idx * step
        e = s + diss_length
        if e > seg_end:
            break
        w_sel = slice(s, e)
        P_windows[idx] = float(np.mean(P_fast[w_sel]))

        # --- Epsilon ---
        if n_shear > 0:
            sh_seg = np.column_stack([d[w_sel] for _, d in shear_data])
            ac_seg = np.column_stack([d[w_sel] for _, d in accel_data]) if accel_data else None
            er = compute_eps_window(
                sh_seg,
                ac_seg,
                speed_fast[w_sel],
                P_fast[w_sel],
                mean_T,
                fs_fast,
                fft_length,
                f_AA,
                do_goodman,
            )
            eps_arr[:, idx] = er.epsilon
            FM_arr[:, idx] = er.FM
            fom_arr[:, idx] = er.fom
            mad_arr[:, idx] = er.mad
            Kmr_arr[:, idx] = er.K_max_ratio

        # --- Chi (both Batchelor and Kraichnan, per-probe epsilon) ---
        has_eps = n_shear > 0 and np.any(np.isfinite(er.epsilon) & (er.epsilon > 0))
        if n_therm > 0 and has_eps:
            therm_segs = [therm_data[ci][1][w_sel] for ci in range(n_therm)]
            for model, arr, fom_a in [
                ("batchelor", chi_batch_arr, chi_fom_batch_arr),
                ("kraichnan", chi_kraich_arr, chi_fom_kraich_arr),
            ]:
                cr = compute_chi_window(
                    therm_segs,
                    diff_gains,
                    er.W,
                    mean_T,
                    er.nu,
                    fs_fast,
                    fft_length,
                    f_AA_chi,
                    spectrum_model=model,
                    epsilon=er.epsilon,
                    fom=er.FM,
                    method=1,
                )
                arr[:, idx] = cr.chi
                fom_a[:, idx] = cr.fom

    return {
        "P_windows": P_windows,
        "epsilon": eps_arr,
        "FM": FM_arr,
        "fom": fom_arr,
        "mad": mad_arr,
        "K_max_ratio": Kmr_arr,
        "chi_batchelor": chi_batch_arr,
        "chi_kraichnan": chi_kraich_arr,
        "chi_fom_batchelor": chi_fom_batch_arr,
        "chi_fom_kraichnan": chi_fom_kraich_arr,
    }


def _hp_filter(x, fs, cutoff=1.0):
    """High-pass filter for shear display."""
    from scipy.signal import butter, filtfilt

    b, a = butter(1, cutoff / (fs / 2), btype="high")
    return filtfilt(b, a, x)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class ProfileViewer:
    """Base class for interactive profile viewers.

    Handles channel extraction, profile detection, interpolation, shear
    conversion, navigation buttons, and common drawing methods.
    Subclasses override ``_setup_axes`` and ``_draw`` for their specific layout.
    Subclasses may set ``_nrows``/``_ncols`` to request a larger panel grid.
    """

    # Panel grid; subclasses may override (e.g. mixing_look uses 3 rows).
    _nrows = 2
    _ncols = 4

    def __init__(
        self,
        pf,
        fft_length=1024,
        f_AA=98.0,
        goodman=True,
        direction="auto",
        P_min=0.5,
        W_min=0.3,
        min_duration=7.0,
        spec_P_range=None,
        diss_length=None,
        vehicle=None,
    ):
        from odas_tpw.rsi.vehicle import resolve_direction, resolve_tau

        self.pf = pf
        self.fft_length = fft_length
        self.diss_length = diss_length or 4 * fft_length
        self.f_AA = f_AA
        self.goodman = goodman
        self.spec_P_range = spec_P_range

        # Resolve vehicle, direction, and tau
        if vehicle is None:
            vehicle = pf.config["instrument_info"].get("vehicle", "").lower()
        direction = resolve_direction(direction, vehicle)
        tau = resolve_tau(vehicle)

        # Extract channel data
        self.shear = sorted(
            [(n, pf.channels[n]) for n in pf._fast_channels if SH_PATTERN.match(n)],
            key=lambda x: x[0],
        )
        self.accel = sorted(
            [(n, pf.channels[n]) for n in pf._fast_channels if AC_PATTERN.match(n)],
            key=lambda x: x[0],
        )
        self.therm_slow = sorted(
            [(n, pf.channels[n]) for n in pf._slow_channels if T_PATTERN.match(n)],
            key=lambda x: x[0],
        )
        self.therm_fast = sorted(
            [(n, pf.channels[n]) for n in pf._fast_channels if DT_PATTERN.match(n)],
            key=lambda x: x[0],
        )
        self.diff_gains = []
        for name, _ in self.therm_fast:
            ch_cfg: dict = next((ch for ch in pf.config["channels"] if ch.get("name") == name), {})
            self.diff_gains.append(float(ch_cfg.get("diff_gain", "0.94")))

        # All temperature channels for temperature panel (T1, T2, JAC_T)
        self.temp_channels = []
        for name in ["T1", "T2", "JAC_T"]:
            if name in pf.channels:
                self.temp_channels.append((name, pf.channels[name]))

        self.P = pf.channels["P"]
        self.T = pf.channels.get("T1", pf.channels.get("T", np.zeros_like(self.P)))
        self.JAC_T = pf.channels.get("JAC_T", None)
        self.t_fast = pf.t_fast
        self.t_slow = pf.t_slow
        self.fs_fast = pf.fs_fast
        self.fs_slow = pf.fs_slow

        # Detect profiles
        W = _smooth_fall_rate(self.P, self.fs_slow, tau=tau)
        self.W = W
        self.profiles = get_profiles(
            self.P,
            W,
            self.fs_slow,
            P_min=P_min,
            W_min=W_min,
            direction=direction,
            min_duration=min_duration,
        )
        if not self.profiles:
            raise ValueError("No profiles detected in this file")

        # Interpolate slow -> fast
        self.P_fast = np.interp(self.t_fast, self.t_slow, self.P)
        self.speed_fast = np.abs(np.interp(self.t_fast, self.t_slow, W))
        self.ratio = round(self.fs_fast / self.fs_slow)

        # Convert shear from piezo output to du/dz by dividing by speed²
        for i, (name, data) in enumerate(self.shear):
            self.shear[i] = (name, data / self.speed_fast**2)

        self.profile_idx = 0
        self.fig = None
        self._spec_bin_width = None
        # Per-draw caches populated by subclasses' _draw (used by the shared
        # spectra / FM / FOM drawing methods below).
        self._cached_diss: dict | None = None
        self._cached_spec: dict | None = None

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def _slow_to_fast_slice(self, s_slow, e_slow):
        s_fast = s_slow * self.ratio
        e_fast = min((e_slow + 1) * self.ratio, len(self.t_fast))
        return slice(s_fast, e_fast)

    def _spec_fast_slice(self, sel_fast):
        if self.spec_P_range is None:
            return sel_fast
        P_min, P_max = self.spec_P_range
        P_seg = self.P_fast[sel_fast]
        mask = (P_seg >= P_min) & (P_seg <= P_max)
        if not np.any(mask):
            return sel_fast
        indices = np.where(mask)[0]
        return slice(sel_fast.start + indices[0], sel_fast.start + indices[-1] + 1)

    def _spec_bin_dbar(self, s_slow, e_slow):
        sel = self._slow_to_fast_slice(s_slow, e_slow)
        mean_speed = float(np.mean(np.abs(self.speed_fast[sel])))
        if mean_speed < 0.01:
            mean_speed = 0.01
        return self.diss_length / self.fs_fast * mean_speed

    # ------------------------------------------------------------------
    # Shared drawing methods — take ax as first parameter
    # ------------------------------------------------------------------

    def _draw_overview(self, ax, s_slow, e_slow):
        """Profile overview: pressure vs time, current profile highlighted."""
        ax.plot(self.t_slow, self.P, "0.6", linewidth=0.5)
        ax.plot(
            self.t_slow[s_slow : e_slow + 1],
            self.P[s_slow : e_slow + 1],
            "C0",
            linewidth=1.5,
        )
        ax.set_ylabel("Pressure [dbar]")
        ax.set_xlabel("Time [s]")
        ax.set_title("Profile overview", fontsize=9)
        ax.grid(True, alpha=0.3)
        for i, (s, e) in enumerate(self.profiles):
            if i != self.profile_idx:
                ax.axvspan(self.t_slow[s], self.t_slow[e], alpha=0.05, color="C0")

    def _draw_eps_profile(self, ax, P_win, eps_arr):
        """Epsilon from each shear probe vs pressure."""
        if len(P_win) == 0:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
            return

        colors = ["C0", "C1", "C4", "C5"]
        has_data = False
        for i, (name, _) in enumerate(self.shear):
            c = colors[i % len(colors)]
            eps = eps_arr[i]
            valid = np.isfinite(eps) & (eps > 0)
            if np.any(valid):
                ax.plot(eps[valid], P_win[valid], f"{c}o-", markersize=3, linewidth=0.8, label=name)
                has_data = True

        if has_data:
            ax.set_xscale("log")
            ax.set_xlabel("ε [W/kg]")
            ax.set_ylabel("Pressure [dbar]")
            ax.legend(fontsize=6, loc="lower left")
            ax.set_title("ε profile", fontsize=9)
            ax.grid(True, alpha=0.3, which="both")
        else:
            ax.text(0.5, 0.5, "No valid ε", transform=ax.transAxes, ha="center", va="center")

    def _draw_shear(self, ax, sel_fast):
        """HP-filtered shear signals vs pressure."""
        P_prof = self.P_fast[sel_fast]

        if not self.shear:
            ax.text(0.5, 0.5, "No shear channels", transform=ax.transAxes, ha="center", va="center")
            return

        # Identify settled region: where fall rate is within 20% of median
        # (near terminal velocity, excluding initial acceleration transient)
        W_prof = self.speed_fast[sel_fast]
        median_W = np.median(W_prof)
        settled = W_prof >= 0.8 * median_W

        all_settled_vals = []
        for i, (name, data) in enumerate(self.shear):
            seg = data[sel_fast]
            if len(seg) > 2 * int(self.fs_fast):
                seg_hp = _hp_filter(seg, self.fs_fast, cutoff=1.0)
            else:
                seg_hp = seg - np.mean(seg)
            seg_hp, _, _, _ = despike(seg_hp, self.fs_fast, thresh=8, smooth=0.5)
            offset = i * 0.5
            ax.plot(seg_hp + offset, P_prof, linewidth=0.3, label=name)
            if np.any(settled):
                all_settled_vals.append(seg_hp[settled] + offset)

        # Zoom x-axis to 5x the settled-region shear range
        if all_settled_vals:
            combined = np.concatenate(all_settled_vals)
            x_lo = np.percentile(combined, 1)
            x_hi = np.percentile(combined, 99)
            half_range = (x_hi - x_lo) / 2
            center = (x_lo + x_hi) / 2
            half_range = max(half_range * 5, 0.05)
            ax.set_xlim(center - half_range, center + half_range)

        ax.set_xlabel("Shear [s⁻¹] (HP, offset)")
        ax.legend(fontsize=7, loc="lower right")
        ax.set_title("Shear probes", fontsize=9)
        ax.grid(True, alpha=0.3)

    def _draw_eps_spectra(self, ax, sel_spec, cached_spec):
        """Shear wavenumber spectra with Nasmyth fits + K_max markers."""
        r = cached_spec
        K = r["K"]
        if len(K) == 0:
            ax.text(0.5, 0.5, "No spectra", transform=ax.transAxes, ha="center", va="center")
            return

        colors = ["C0", "C1", "C4", "C5"]
        for i, (name, _) in enumerate(self.shear):
            if i >= len(r["shear_specs"]):
                break
            c = colors[i % len(colors)]
            spec = r["shear_specs"][i]
            ax.loglog(K, spec, c, linewidth=0.8, label=name)
            nas = r["nasmyth_specs"][i]
            eps_i = r["epsilons"][i]
            meth_i = r["methods"][i] if i < len(r["methods"]) else 0
            chosen = "ISR" if meth_i == 1 else "var"
            e_isr = r["epsilons_isr"][i] if i < len(r.get("epsilons_isr", [])) else np.nan
            e_var = r["epsilons_var"][i] if i < len(r.get("epsilons_var", [])) else np.nan
            if np.isfinite(eps_i):
                isr_mark = "*" if chosen == "ISR" else ""
                var_mark = "*" if chosen == "var" else ""
                label_parts = []
                if np.isfinite(e_isr):
                    label_parts.append(f"ISR={e_isr:.1e}{isr_mark}")
                if np.isfinite(e_var):
                    label_parts.append(f"var={e_var:.1e}{var_mark}")
                ax.loglog(
                    K,
                    nas,
                    c,
                    linewidth=0.7,
                    linestyle="--",
                    alpha=0.8,
                    label=f"Nas {', '.join(label_parts)}",
                )
            k_max_i = r["K_maxes"][i]
            if np.isfinite(k_max_i):
                k_idx = np.argmin(np.abs(K - k_max_i))
                ax.plot(k_max_i, spec[k_idx], marker="v", color=c, markersize=7, zorder=5)

        # Reference Nasmyth lines
        nu = r["nu"]
        for exp in [-9, -8, -7, -6, -5]:
            nas_ref = nasmyth(10.0**exp, nu, K)
            ax.loglog(K, nas_ref, "0.8", linewidth=0.3)

        # AA line
        K_AA = self.f_AA / r["W"]
        ax.axvline(K_AA, color="0.5", linestyle=":", linewidth=0.5, alpha=0.5)

        # Dynamic y-axis
        x_lo, x_hi = 0.5, 300
        in_range = (x_lo <= K) & (x_hi >= K)
        all_vals = []
        for spec in r["shear_specs"]:
            pos = spec[in_range & (spec > 0)]
            if len(pos):
                all_vals.append(pos)
        for nas in r["nasmyth_specs"]:
            pos = nas[in_range & (nas > 0)]
            if len(pos):
                all_vals.append(pos)
        if all_vals:
            combined = np.concatenate(all_vals)
            y_hi = 10 ** (np.ceil(np.log10(np.max(combined))) + 1)
            y_lo = 10 ** np.floor(np.log10(np.min(combined)))
        else:
            y_hi, y_lo = 1e0, 1e-9

        P_lo = float(self.P_fast[sel_spec.start])
        P_hi = float(self.P_fast[min(sel_spec.stop - 1, len(self.P_fast) - 1)])
        ax.set_xlabel("Wavenumber [cpm]")
        ax.set_ylabel("Φ_sh [s⁻² cpm⁻¹]")
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.legend(fontsize=5, loc="lower left")
        ax.set_title(
            f"ε spectra  speed={r['W']:.2f} m/s\nP={P_lo:.1f}–{P_hi:.1f} dbar",  # noqa: RUF001
            fontsize=9,
        )
        ax.grid(True, alpha=0.3, which="both")

    def _draw_temps(self, ax, s_slow, e_slow):
        """T1, T2, JAC_T vs pressure."""
        P_prof = self.P[s_slow : e_slow + 1]

        if not self.temp_channels:
            ax.text(
                0.5,
                0.5,
                "No temperature channels",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
            return

        colors = {"T1": "C3", "T2": "C1", "JAC_T": "C9"}
        styles = {"T1": "-", "T2": "-", "JAC_T": "--"}
        for name, data in self.temp_channels:
            c = colors.get(name, "C4")
            ls = styles.get(name, "-")
            ax.plot(data[s_slow : e_slow + 1], P_prof, c, linewidth=0.8, linestyle=ls, label=name)

        ax.set_xlabel("Temperature [°C]")
        ax.legend(fontsize=7, loc="lower left")
        ax.set_title("T1, T2, JAC_T", fontsize=9)
        ax.grid(True, alpha=0.3)

    def _draw_fall_rate(self, ax, s_slow, e_slow):
        """Fall rate vs pressure."""
        P_prof = self.P[s_slow : e_slow + 1]
        W_prof = self.W[s_slow : e_slow + 1]

        ax.plot(W_prof, P_prof, "C2", linewidth=0.8, label="dP/dt")
        ax.set_xlabel("Fall rate [dbar/s]")
        ax.set_ylabel("Pressure [dbar]")
        ax.legend(fontsize=7, loc="lower left")
        ax.set_title("Fall rate", fontsize=9)
        ax.grid(True, alpha=0.3)

    # ------------------------------------------------------------------
    # Shared FM / chi-spectra / chi-FOM panels (need self._cached_diss /
    # self._cached_spec, populated by the subclass _draw)
    # ------------------------------------------------------------------

    def _draw_fm_profile(self, ax):
        """Lueck (2022) FM figure of merit vs pressure."""
        d = self._cached_diss
        assert d is not None
        P = d["P_windows"]

        if len(P) == 0:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
            return

        colors = ["C0", "C1", "C4", "C5"]
        has_data = False
        pct1_lines = []
        pct115_lines = []
        for i, (name, _) in enumerate(self.shear):
            c = colors[i % len(colors)]
            fm = d["FM"][i]
            valid = np.isfinite(fm)
            if np.any(valid):
                fm_valid = fm[valid]
                pct1 = 100.0 * np.sum(fm_valid > 1.0) / len(fm_valid)
                pct115 = 100.0 * np.sum(fm_valid > 1.15) / len(fm_valid)
                pct1_lines.append(f"{name}:{pct1:.0f}%")
                pct115_lines.append(f"{name}:{pct115:.0f}%")
                ax.plot(fm_valid, P[valid], f"{c}o-", markersize=3, linewidth=0.8, label=name)
                has_data = True

        if has_data:
            ax.axvline(
                1.0,
                color="k",
                linestyle="-",
                linewidth=1.0,
                alpha=0.7,
                label=f"FM>1: {', '.join(pct1_lines)}",
            )
            ax.axvline(
                1.15,
                color="0.5",
                linestyle="--",
                linewidth=0.7,
                alpha=0.5,
                label=f"FM>1.15: {', '.join(pct115_lines)}",
            )
            xlim = ax.get_xlim()
            ax.axvspan(0, 1.0, color="green", alpha=0.05, zorder=0)
            ax.axvspan(1.0, max(xlim[1], 3), color="red", alpha=0.05, zorder=0)

            ax.set_xlabel("FM")
            ax.set_ylabel("Pressure [dbar]")
            ax.legend(fontsize=6, loc="lower right")
            ax.set_title("Lueck FM (2022)", fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, max(xlim[1], 2))
        else:
            ax.text(0.5, 0.5, "No valid FM", transform=ax.transAxes, ha="center", va="center")

    def _draw_chi_fom_profile(self, ax):
        """Chi figure of merit vs pressure (Batchelor & Kraichnan)."""
        d = self._cached_diss
        assert d is not None
        P = d["P_windows"]

        if len(P) == 0:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
            return

        colors = ["C3", "C1", "C4", "C5"]
        has_data = False
        for i, (name, _) in enumerate(self.therm_fast):
            c = colors[i % len(colors)]

            fom_b = d["chi_fom_batchelor"][i]
            valid_b = np.isfinite(fom_b)
            if np.any(valid_b):
                ax.plot(
                    fom_b[valid_b],
                    P[valid_b],
                    color=c,
                    linestyle="--",
                    marker="^",
                    markersize=2.5,
                    linewidth=0.8,
                    label=f"{name} Batch.",
                )
                has_data = True

            fom_k = d["chi_fom_kraichnan"][i]
            valid_k = np.isfinite(fom_k)
            if np.any(valid_k):
                ax.plot(
                    fom_k[valid_k],
                    P[valid_k],
                    color=c,
                    linestyle="-",
                    marker="o",
                    markersize=2.5,
                    linewidth=0.8,
                    label=f"{name} Kraich.",
                )
                has_data = True

        if has_data:
            ax.axvline(1.0, color="k", linestyle="-", linewidth=1.0, alpha=0.7, label="FOM = 1")
            ax.set_xlabel("χ FOM (obs/model)")
            ax.set_ylabel("Pressure [dbar]")
            ax.legend(fontsize=5, loc="lower right")
            ax.set_title("χ FOM (Batch. & Kraich.)", fontsize=9)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(
                0.5, 0.5, "No valid χ FOM", transform=ax.transAxes, ha="center", va="center"
            )

    def _draw_chi_m1m2_spectra(self, ax, sel_spec):
        """Chi spectra -- Method 1 (from epsilon) vs Method 2 (iterative fit)."""
        n_fast = sel_spec.stop - sel_spec.start
        if n_fast < self.diss_length:
            ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes, ha="center", va="center")
            return

        r = self._cached_spec
        assert r is not None
        K_chi = r["F"] / r["W"]

        if not r["chi_obs_specs"]:
            ax.text(0.5, 0.5, "No χ spectra", transform=ax.transAxes, ha="center", va="center")
            return

        colors_chi = ["C3", "C1"]
        for i, (name, _) in enumerate(self.therm_fast):
            if i >= len(r["chi_obs_specs"]):
                break
            c = colors_chi[i % len(colors_chi)]

            obs = r["chi_obs_specs"][i]
            valid = np.isfinite(obs) & (obs > 0) & (K_chi > 0)
            if np.any(valid):
                ax.loglog(K_chi[valid], obs[valid], c, linewidth=0.8, alpha=0.6, label=name)

            m1_spec = r["chi_m1_specs"][i]
            chi_m1 = r["chi_m1_vals"][i]
            valid_m1 = np.isfinite(m1_spec) & (m1_spec > 0) & (K_chi > 0)
            if np.any(valid_m1) and np.isfinite(chi_m1):
                ax.loglog(
                    K_chi[valid_m1],
                    m1_spec[valid_m1],
                    c,
                    linewidth=1.2,
                    linestyle="--",
                    alpha=0.9,
                    label=f"M1 χ={chi_m1:.1e}",
                )

            m2_spec = r["chi_m2_specs"][i]
            chi_m2 = r["chi_m2_vals"][i]
            valid_m2 = np.isfinite(m2_spec) & (m2_spec > 0) & (K_chi > 0)
            if np.any(valid_m2) and np.isfinite(chi_m2):
                ax.loglog(
                    K_chi[valid_m2],
                    m2_spec[valid_m2],
                    c,
                    linewidth=1.2,
                    linestyle="-.",
                    alpha=0.9,
                    label=f"M2 χ={chi_m2:.1e}",
                )

        noise = r["noise_K"]
        if noise is not None:
            valid_n = np.isfinite(noise) & (noise > 0) & (K_chi > 0)
            if np.any(valid_n):
                ax.loglog(
                    K_chi[valid_n],
                    noise[valid_n],
                    "0.5",
                    linewidth=0.6,
                    linestyle=":",
                    label="Noise",
                )

        K_AA = self.f_AA / r["W"]
        ax.axvline(K_AA, color="0.5", linestyle=":", linewidth=0.5, alpha=0.5)

        P_lo = float(self.P_fast[sel_spec.start])
        P_hi = float(self.P_fast[min(sel_spec.stop - 1, len(self.P_fast) - 1)])
        x_lo, x_hi = 0.5, 300
        in_range = (K_chi >= x_lo) & (K_chi <= x_hi)
        all_vals = []
        for spec_list in [r["chi_obs_specs"], r["chi_m1_specs"], r["chi_m2_specs"]]:
            for s in spec_list:
                pos = s[in_range & (s > 0) & np.isfinite(s)]
                if len(pos):
                    all_vals.append(pos)
        if all_vals:
            combined = np.concatenate(all_vals)
            y_hi = 10 ** (np.ceil(np.log10(np.max(combined))) + 1)
        else:
            y_hi = 1e-2
        if noise is not None:
            noise_in = noise[in_range & (noise > 0) & np.isfinite(noise)]
            y_lo = 10 ** (np.floor(np.log10(np.min(noise_in))) - 1) if len(noise_in) else 1e-11
        else:
            y_lo = 1e-11

        ax.set_xlabel("Wavenumber [cpm]")
        ax.set_ylabel("Φ_T [(K/m)² cpm⁻¹]")
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.legend(fontsize=5, loc="lower left")
        ax.set_title(
            f"χ spectra (Kraichnan)  P={P_lo:.1f}–{P_hi:.1f}\n"  # noqa: RUF001
            f"(-- M1 from ε, -· M2 iter)",
            fontsize=9,
        )
        ax.grid(True, alpha=0.3, which="both")

    # ------------------------------------------------------------------
    # Figure setup and navigation
    # ------------------------------------------------------------------

    def _setup_axes(self):
        """Configure axis linkage.  Override in subclass."""

    def _draw(self):
        """Draw all panels.  Override in subclass."""

    def _finish_draw(self, s_slow, e_slow, pressure_axes):
        """Shared title and green-band logic, called at end of _draw."""
        if self.spec_P_range is not None:
            sp_lo, sp_hi = self.spec_P_range
            for ax in pressure_axes:
                ax.axhspan(sp_lo, sp_hi, color="green", alpha=0.15, zorder=0)

        P_lo, P_hi = self.P[s_slow], self.P[e_slow]
        title = (
            f"{self.pf.filepath.name}  —  "
            f"Profile {self.profile_idx + 1}/{len(self.profiles)}  —  "
            f"P: {P_lo:.1f}–{P_hi:.1f} dbar"  # noqa: RUF001
        )
        if self.spec_P_range is not None:
            sp_lo, sp_hi = self.spec_P_range
            title += f"  [spectra: {sp_lo:.1f}–{sp_hi:.1f} dbar]"  # noqa: RUF001
        assert self.fig is not None
        self.fig.suptitle(title, fontsize=11, fontweight="bold")
        self.fig.canvas.draw_idle()

    def show(self) -> None:
        self.fig, self.axes = plt.subplots(
            self._nrows,
            self._ncols,
            figsize=(6.0 * self._ncols, 4.5 * self._nrows),
            gridspec_kw={"hspace": 0.35, "wspace": 0.32},
            squeeze=False,
        )
        left, right, bottom, top = 0.04, 0.98, 0.08, 0.92
        self.fig.subplots_adjust(bottom=bottom, top=top, left=left, right=right)

        self._setup_axes()
        self._add_nav_diamond(left, right, bottom, top)

        self._draw()
        plt.show()

    def _add_nav_diamond(self, left, right, bottom, top) -> None:
        """Place a textless 4-arrow navigation diamond at the figure centre.

        Left/right (◀ ▶) step the profile; up/down (▲ ▼) move the spectral
        depth window. The centre is the same self-relative figure location for
        any grid: horizontally between the middle two columns, vertically at the
        figure centre.
        """
        assert self.fig is not None
        cx = (left + right) / 2.0
        cy = (bottom + top) / 2.0
        bw, bh = 0.024, 0.032  # button size [figure fraction]
        s = 0.040  # diamond radius
        ax_up = self.fig.add_axes((cx - bw / 2, cy + s - bh / 2, bw, bh))
        ax_dn = self.fig.add_axes((cx - bw / 2, cy - s - bh / 2, bw, bh))
        ax_left = self.fig.add_axes((cx - s - bw / 2, cy - bh / 2, bw, bh))
        ax_right = self.fig.add_axes((cx + s - bw / 2, cy - bh / 2, bw, bh))
        self.btn_spec_up = Button(ax_up, "▲")
        self.btn_spec_dn = Button(ax_dn, "▼")
        self.btn_prev = Button(ax_left, "◀")
        self.btn_next = Button(ax_right, "▶")
        for b in (self.btn_spec_up, self.btn_spec_dn, self.btn_prev, self.btn_next):
            b.label.set_fontsize(13)
        self.btn_spec_up.on_clicked(self._on_spec_up)
        self.btn_spec_dn.on_clicked(self._on_spec_dn)
        self.btn_prev.on_clicked(self._on_prev)
        self.btn_next.on_clicked(self._on_next)

    def _on_prev(self, event):
        if self.profile_idx > 0:
            self.profile_idx -= 1
            self._draw()

    def _on_next(self, event):
        if self.profile_idx < len(self.profiles) - 1:
            self.profile_idx += 1
            self._draw()

    def _on_spec_up(self, event):
        s_slow, e_slow = self.profiles[self.profile_idx]
        P_top = float(min(self.P[s_slow], self.P[e_slow]))
        bw = self._spec_bin_width or 1.0
        if self.spec_P_range is None:
            self.spec_P_range = (P_top, P_top + bw)
        else:
            step = self.spec_P_range[1] - self.spec_P_range[0]
            new_lo = self.spec_P_range[0] - step
            new_hi = self.spec_P_range[1] - step
            if new_lo < P_top:
                new_lo = P_top
                new_hi = P_top + step
            self.spec_P_range = (new_lo, new_hi)
        self._draw()

    def _on_spec_dn(self, event):
        s_slow, e_slow = self.profiles[self.profile_idx]
        P_bot = float(max(self.P[s_slow], self.P[e_slow]))
        P_top = float(min(self.P[s_slow], self.P[e_slow]))
        bw = self._spec_bin_width or 1.0
        if self.spec_P_range is None:
            self.spec_P_range = (P_top, P_top + bw)
        else:
            step = self.spec_P_range[1] - self.spec_P_range[0]
            new_lo = self.spec_P_range[0] + step
            new_hi = self.spec_P_range[1] + step
            if new_hi > P_bot:
                new_hi = P_bot
                new_lo = P_bot - step
            self.spec_P_range = (new_lo, new_hi)
        self._draw()

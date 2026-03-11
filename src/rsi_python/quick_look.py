# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Interactive quick-look viewer for Rockland Scientific profiler data.

Port of quick_look.m from the ODAS MATLAB library.
Provides a multi-panel figure with Prev/Next buttons to step through profiles.

Layout (2 rows x 4 columns):
  Row 0 (profile panels, pressure y-axis linked):
    Overview | epsilon profile | chi profile | Shear probes
  Row 1 (spectra + detail):
    T1, T2, JAC_T | epsilon spectra | chi spectra | Fall rate
"""

import warnings

import numpy as np

from rsi_python.nasmyth import nasmyth
from rsi_python.ocean import visc35
from rsi_python.p_file import PFile
from rsi_python.viewer_base import (
    ProfileViewer,
    compute_depth_spectra,
    select_mid_window,
)
from rsi_python.window import compute_chi_window, compute_eps_window

# ---------------------------------------------------------------------------
# ql-specific free functions
# ---------------------------------------------------------------------------


def _compute_chi_spectra(
    therm_data,
    diff_gains,
    P_fast,
    T_slow,
    speed_fast,
    fs_fast,
    sel,
    fft_length,
    f_AA,
    epsilons,
    spectrum_model="kraichnan",
):
    """Compute temperature gradient spectra and all three chi methods for one profile segment.

    Uses shared ``compute_chi_window`` from ``window.py`` for Method 1 (chi
    from epsilon) and Method 2 Iterative (Peterson & Fer 2014).  Method 2 MLE
    uses ``_mle_fit_kB`` directly with the correct gradient spectrum from the
    shared function.

    Selects a single diss_length window closest to the midpoint pressure,
    matching diss_look and MATLAB plot_spectra.

    Returns (K, F, obs_spectra, methods_results, noise_K, mean_speed, nu).

    methods_results is a dict keyed by method name ("M1", "M2-MLE", "M2-Iter"),
    each mapping to a list of per-probe (batch_spec, chi_val, kB_val) tuples.
    """
    from rsi_python.batchelor import KAPPA_T
    from rsi_python.chi import _mle_fit_kB

    w_sel = select_mid_window(P_fast, sel, fft_length)

    mean_T = float(np.mean(T_slow))
    mean_speed = float(np.mean(np.abs(speed_fast[w_sel])))
    if mean_speed < 0.01:
        mean_speed = 0.01
    nu = float(visc35(mean_T))

    n_therm = len(therm_data)
    n_freq = fft_length // 2 + 1
    f_AA_chi = 0.9 * f_AA  # 10% margin, matching MATLAB get_chi

    therm_segs = [therm_data[ci][1][w_sel] for ci in range(n_therm)]
    eps_arr = np.array(epsilons) if epsilons else None

    # Method 1: chi from epsilon (also produces observed gradient spectra + noise)
    cr_m1 = compute_chi_window(
        therm_segs, diff_gains, mean_speed, mean_T, nu,
        fs_fast, fft_length, f_AA_chi,
        spectrum_model=spectrum_model, epsilon=eps_arr, method=1,
    )

    # Method 2 Iterative
    cr_m2_iter = compute_chi_window(
        therm_segs, diff_gains, mean_speed, mean_T, nu,
        fs_fast, fft_length, f_AA_chi,
        spectrum_model=spectrum_model, method=2,
    )

    K = cr_m1.K
    F = cr_m1.F
    H2 = cr_m1.H2
    obs_spectra = cr_m1.grad_specs
    noise_K = cr_m1.noise_K
    K_AA_chi = f_AA_chi / mean_speed

    nan_result = (np.full(n_freq, np.nan), np.nan, np.nan)
    methods_results = {"M1": [], "M2-MLE": [], "M2-Iter": []}

    for ci in range(n_therm):
        # M1
        if np.isfinite(cr_m1.chi[ci]) and cr_m1.chi[ci] > 0:
            methods_results["M1"].append(
                (cr_m1.model_specs[ci], float(cr_m1.chi[ci]), float(cr_m1.kB[ci]))
            )
        else:
            methods_results["M1"].append(nan_result)

        # M2-Iter
        if np.isfinite(cr_m2_iter.chi[ci]) and cr_m2_iter.chi[ci] > 0:
            methods_results["M2-Iter"].append(
                (cr_m2_iter.model_specs[ci], float(cr_m2_iter.chi[ci]),
                 float(cr_m2_iter.kB[ci]))
            )
        else:
            methods_results["M2-Iter"].append(nan_result)

        # M2-MLE: call _mle_fit_kB directly with gradient spectrum from shared code
        grad = obs_spectra[ci] if ci < len(obs_spectra) else np.full(n_freq, np.nan)
        dg_i = diff_gains[ci] if ci < len(diff_gains) else 0.94
        valid = np.isfinite(grad) & (grad > 0) & (K > 0) & (K <= K_AA_chi)
        if np.sum(valid) >= 3:
            chi_obs = max(6 * KAPPA_T * np.trapezoid(grad[valid], K[valid]), 1e-10)
            kB_best, chi_val, _, _, spec_raw, _, _ = _mle_fit_kB(
                grad, K, chi_obs, nu, mean_T, mean_speed, f_AA_chi,
                "single_pole", spectrum_model, fs_fast, dg_i, fft_length,
            )
            spec_mle = spec_raw * H2 if np.isfinite(chi_val) else np.full(n_freq, np.nan)
            methods_results["M2-MLE"].append((spec_mle, chi_val, kB_best))
        else:
            methods_results["M2-MLE"].append(nan_result)

    return K, F, obs_spectra, methods_results, noise_K, mean_speed, nu


def _compute_windowed_eps_chi(
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
    chi_method=1,
    spectrum_model="kraichnan",
):
    """Compute windowed epsilon and chi estimates for a profile segment.

    Returns (P_windows, eps_array, chi_array).
    eps_array shape: (n_shear, n_windows)
    chi_array shape: (n_therm, n_windows)

    Parameters
    ----------
    chi_method : int
        1 = chi from known epsilon (Method 1), 2 = iterative fit (Method 2).
    spectrum_model : str
        Theoretical spectrum: 'batchelor' or 'kraichnan'.
    """
    diss_length = 2 * fft_length
    overlap = diss_length // 2
    step = diss_length - overlap

    seg_start = sel.start
    seg_end = sel.stop
    N = seg_end - seg_start

    n_shear = len(shear_data)
    n_therm = len(therm_data)
    n_windows = max(0, 1 + (N - diss_length) // step)
    if n_windows == 0:
        return np.array([]), np.empty((n_shear, 0)), np.empty((n_therm, 0))

    P_windows = np.full(n_windows, np.nan)
    eps_arr = np.full((n_shear, n_windows), np.nan)
    chi_arr = np.full((n_therm, n_windows), np.nan)

    mean_T = float(np.mean(T_slow))
    f_AA_chi = 0.9 * f_AA  # 10% margin, matching MATLAB get_chi

    for idx in range(n_windows):
        s = seg_start + idx * step
        e = s + diss_length
        if e > seg_end:
            break
        w_sel = slice(s, e)
        P_windows[idx] = float(np.mean(P_fast[w_sel]))

        # --- Epsilon ---
        er = None
        if n_shear > 0:
            sh_seg = np.column_stack([d[w_sel] for _, d in shear_data])
            ac_seg = np.column_stack([d[w_sel] for _, d in accel_data]) if accel_data else None
            er = compute_eps_window(
                sh_seg, ac_seg, speed_fast[w_sel], P_fast[w_sel],
                mean_T, fs_fast, fft_length, f_AA, do_goodman,
            )
            eps_arr[:, idx] = er.epsilon

        # --- Chi ---
        if n_therm > 0 and er is not None:
            therm_segs = [therm_data[ci][1][w_sel] for ci in range(n_therm)]
            method = 1 if chi_method == 1 else 2
            eps_for_chi = er.epsilon if chi_method == 1 else None
            cr = compute_chi_window(
                therm_segs, diff_gains, er.W, mean_T, er.nu,
                fs_fast, fft_length, f_AA_chi,
                spectrum_model=spectrum_model, epsilon=eps_for_chi, method=method,
            )
            chi_arr[:, idx] = cr.chi

    return P_windows, eps_arr, chi_arr


# ---------------------------------------------------------------------------
# QuickLookViewer
# ---------------------------------------------------------------------------


class QuickLookViewer(ProfileViewer):
    """Interactive multi-panel profile viewer with Prev/Next navigation."""

    def __init__(self, pf, chi_method=1, spectrum_model="kraichnan", **kwargs):
        super().__init__(pf, **kwargs)
        self.chi_method = chi_method
        self.spectrum_model = spectrum_model

    def _setup_axes(self):
        # Link pressure y-axes: row 0 all cols + row 1 cols 0, 3
        p_ref = self.axes[0, 0]
        for col in range(1, 4):
            self.axes[0, col].sharey(p_ref)
        self.axes[1, 0].sharey(p_ref)
        self.axes[1, 3].sharey(p_ref)
        p_ref.invert_yaxis()

        # Link wavenumber x-axes: epsilon spectra (1,1) and chi spectra (1,2)
        self.axes[1, 2].sharex(self.axes[1, 1])

    def _draw(self):
        """Draw all panels for the current profile."""
        s_slow, e_slow = self.profiles[self.profile_idx]
        sel_fast = self._slow_to_fast_slice(s_slow, e_slow)
        self._spec_bin_width = self._spec_bin_dbar(s_slow, e_slow)
        sel_spec = self._spec_fast_slice(sel_fast)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            for ax in self.axes.flat:
                ax.clear()
        # Re-invert pressure y-axes (clear resets inversion)
        self.axes[0, 0].invert_yaxis()

        # Pre-compute depth spectra (dict format from viewer_base)
        n_spec = sel_spec.stop - sel_spec.start
        self._cached_spec = None
        if self.shear and n_spec >= 2 * self.fft_length:
            self._cached_spec = compute_depth_spectra(
                self.shear, self.accel, self.therm_fast, self.diff_gains,
                self.P_fast, self.T, self.speed_fast, self.fs_fast,
                sel_spec, self.fft_length, self.f_AA, self.goodman,
            )

        # Pre-compute windowed eps/chi for profile panels
        n_fast = sel_fast.stop - sel_fast.start
        self._cached_windowed = None
        if n_fast >= 2 * self.fft_length:
            P_win, eps_arr, chi_arr = _compute_windowed_eps_chi(
                self.shear, self.accel, self.therm_fast, self.diff_gains,
                self.P_fast, self.T, self.speed_fast, self.fs_fast,
                sel_fast, self.fft_length, self.f_AA, self.goodman,
                chi_method=self.chi_method, spectrum_model=self.spectrum_model,
            )
            if len(P_win) > 0:
                self._cached_windowed = (P_win, eps_arr, chi_arr)

        # Row 0: profile panels
        self._draw_overview(self.axes[0, 0], s_slow, e_slow)
        P_win = self._cached_windowed[0] if self._cached_windowed else np.array([])
        eps_arr = (
            self._cached_windowed[1]
            if self._cached_windowed
            else np.empty((len(self.shear), 0))
        )
        self._draw_eps_profile(self.axes[0, 1], P_win, eps_arr)
        self._draw_chi_profile()
        self._draw_shear(self.axes[0, 3], sel_fast)

        # Row 1: spectra and detail panels
        self._draw_temps(self.axes[1, 0], s_slow, e_slow)
        self._draw_spectra(sel_spec)
        self._draw_chi_spectra(sel_spec)
        self._draw_fall_rate(self.axes[1, 3], s_slow, e_slow)

        # Finish (green band + title)
        pressure_axes = (
            [self.axes[0, c] for c in range(4)]
            + [self.axes[1, 0], self.axes[1, 3]]
        )
        self._finish_draw(s_slow, e_slow, pressure_axes)

    # ------------------------------------------------------------------
    # ql-specific panels
    # ------------------------------------------------------------------

    def _draw_chi_profile(self):
        """Panel (0,2): Chi from each thermistor vs pressure."""
        ax = self.axes[0, 2]

        if self._cached_windowed is None:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center")
            return

        P_win, _, chi_arr = self._cached_windowed
        colors_chi = ["C3", "C4"]
        has_data = False
        for i, (name, _) in enumerate(self.therm_fast):
            c = colors_chi[i % len(colors_chi)]
            valid = np.isfinite(chi_arr[i]) & (chi_arr[i] > 0)
            if np.any(valid):
                ax.plot(
                    chi_arr[i, valid], P_win[valid],
                    f"{c}s-", markersize=3, linewidth=0.8, label=name,
                )
                has_data = True
        if has_data:
            ax.set_xscale("log")
            ax.set_xlabel("\u03c7 [K\u00b2/s]")
            ax.set_ylabel("Pressure [dbar]")
            ax.legend(fontsize=6, loc="lower left")
            selected = {1: "M1", 2: "M2-MLE"}.get(self.chi_method, "M1")
            ax.set_title(f"\u03c7 profile ({selected}, {self.spectrum_model})",
                         fontsize=9)
            ax.grid(True, alpha=0.3, which="both")
        else:
            ax.text(0.5, 0.5, "No valid \u03c7", transform=ax.transAxes,
                    ha="center", va="center")

    def _draw_spectra(self, sel_spec):
        """Panel (1,1): Epsilon spectra with Nasmyth + K_max."""
        ax = self.axes[1, 1]

        if self._cached_spec is None:
            ax.text(
                0.5, 0.5, "Insufficient data for spectra",
                transform=ax.transAxes, ha="center", va="center",
            )
            return

        r = self._cached_spec
        K = r["K"]
        if len(K) == 0:
            ax.text(0.5, 0.5, "No spectra", transform=ax.transAxes,
                    ha="center", va="center")
            return

        colors = ["C0", "C1", "C4", "C5"]
        for i, (name, _) in enumerate(self.shear):
            if i >= len(r["shear_specs"]):
                break
            c = colors[i % len(colors)]
            ax.loglog(K, r["shear_specs"][i], c, linewidth=0.8, label=name)
            eps_i = r["epsilons"][i]
            if np.isfinite(eps_i):
                ax.loglog(
                    K, r["nasmyth_specs"][i], c, linewidth=0.75,
                    linestyle="--", alpha=0.9,
                    label=f"Nasmyth (\u03b5={eps_i:.1e})",
                )
                k_max_i = r["K_maxes"][i]
                if np.isfinite(k_max_i):
                    ax.axvline(
                        k_max_i, color=c, linestyle="-.", linewidth=0.8,
                        alpha=0.6, label=f"K_max={k_max_i:.0f} cpm",
                    )
                    # Filled inverted triangle at K_max on the observed spectrum
                    k_idx = np.argmin(np.abs(K - k_max_i))
                    ax.plot(
                        k_max_i, r["shear_specs"][i][k_idx],
                        marker="v", color=c, markersize=8, zorder=5,
                    )

        nu = r["nu"]
        for exp in [-9, -8, -7, -6, -5]:
            eps_ref = 10.0**exp
            nas_ref = nasmyth(eps_ref, nu, K)
            ax.loglog(K, nas_ref, "0.8", linewidth=0.3)
            y_val = nas_ref[len(K) // 2]
            if np.isfinite(y_val) and y_val > 0:
                ax.text(K[len(K) // 2], y_val, f"{exp}",
                        fontsize=6, color="0.5", va="bottom")

        K_AA = self.f_AA / r["W"]
        ax.axvline(K_AA, color="0.5", linestyle=":", linewidth=0.5, alpha=0.5)

        P_lo = float(self.P_fast[sel_spec.start])
        P_hi = float(self.P_fast[min(sel_spec.stop - 1, len(self.P_fast) - 1)])
        ax.set_xlabel("Wavenumber [cpm]")
        ax.set_ylabel("\u03a6(k) [s\u207b\u00b2 cpm\u207b\u00b9]")
        ax.set_xlim(0.5, 300)
        ax.set_ylim(1e-9, 1e0)
        ax.legend(fontsize=5, loc="best", ncol=2)
        ax.set_title(
            f"\u03b5 spectra  speed={r['W']:.2f} m/s\n"
            f"P={P_lo:.1f}\u2013{P_hi:.1f} dbar",
            fontsize=9,
        )
        ax.grid(True, alpha=0.3, which="both")

    def _draw_chi_spectra(self, sel_spec):
        """Panel (1,2): Chi spectra + all three chi methods."""
        ax = self.axes[1, 2]
        self._cached_chi = None

        if not self.therm_fast:
            ax.text(
                0.5, 0.5, "No thermistor gradient channels",
                transform=ax.transAxes, ha="center", va="center",
            )
            return

        n_fast = sel_spec.stop - sel_spec.start
        if n_fast < 2 * self.fft_length:
            ax.text(
                0.5, 0.5, "Insufficient data for chi spectra",
                transform=ax.transAxes, ha="center", va="center",
            )
            return

        eps_for_chi = self._cached_spec["epsilons"] if self._cached_spec else []

        self._cached_chi = _compute_chi_spectra(
            self.therm_fast,
            self.diff_gains,
            self.P_fast,
            self.T,
            self.speed_fast,
            self.fs_fast,
            sel_spec,
            self.fft_length,
            self.f_AA,
            eps_for_chi,
            spectrum_model=self.spectrum_model,
        )
        K, F, obs_spectra, methods_results, noise_K, mean_speed, nu = self._cached_chi

        # Line styles for each method; selected method drawn thicker
        selected_method = {1: "M1", 2: "M2-MLE"}.get(self.chi_method, "M1")
        method_styles = [
            ("M1", "--", 0.9),
            ("M2-MLE", "-.", 0.9),
            ("M2-Iter", ":", 0.9),
        ]
        colors = ["C3", "C1", "C4", "C5"]
        n_probes = len(self.therm_fast)

        # Plot all lines (no labels yet -- legend order controlled manually)
        obs_lines = []
        method_lines = {m: [] for m, _, _ in method_styles}
        for i, (name, _) in enumerate(self.therm_fast):
            c = colors[i % len(colors)]
            valid = np.isfinite(obs_spectra[i]) & (obs_spectra[i] > 0)
            if np.any(valid):
                (line,) = ax.loglog(K[valid], obs_spectra[i][valid], c,
                                    linewidth=0.8)
                obs_lines.append((line, name))
            else:
                obs_lines.append(None)

            for method_name, ls, alpha in method_styles:
                batch_spec, chi_val, kB_val = methods_results[method_name][i]
                valid_b = np.isfinite(batch_spec) & (batch_spec > 0)
                lw = 1.5 if method_name == selected_method else 0.75
                if np.any(valid_b):
                    chi_str = f"{chi_val:.1e}" if np.isfinite(chi_val) else "N/A"
                    label = f"{method_name} \u03c7={chi_str}"
                    if method_name == selected_method:
                        label = f"*{label}"
                    (line,) = ax.loglog(
                        K[valid_b], batch_spec[valid_b], c,
                        linewidth=lw, linestyle=ls, alpha=alpha,
                    )
                    method_lines[method_name].append((line, label))
                else:
                    method_lines[method_name].append(None)

        valid_n = np.isfinite(noise_K) & (noise_K > 0) & (K > 0)
        noise_line = None
        if np.any(valid_n):
            (nl,) = ax.loglog(K[valid_n], noise_K[valid_n], "0.5",
                              linewidth=0.8, linestyle=":")
            noise_line = (nl, "Noise")

        K_AA = self.f_AA / mean_speed
        ax.axvline(K_AA, color="0.5", linestyle=":", linewidth=0.5, alpha=0.5)

        # Build legend: left column = probe 0, right column = probe 1.
        # matplotlib ncol fills column-major: entries 0..N/2-1 go in left col,
        # N/2..N-1 in right col.  So we build per-column lists then concatenate.
        rows = [obs_lines] + [method_lines[m] for m, _, _ in method_styles]
        columns = [[] for _ in range(n_probes)]
        for col in range(n_probes):
            for row in rows:
                if col < len(row) and row[col] is not None:
                    columns[col].append(row[col])
                else:
                    (ph,) = ax.plot([], [], alpha=0)
                    columns[col].append((ph, ""))
            # Noise in the last row of each column
            if col == 0 and noise_line is not None:
                columns[col].append(noise_line)
            else:
                (ph,) = ax.plot([], [], alpha=0)
                columns[col].append((ph, ""))

        # Pad columns to equal length
        max_len = max(len(c) for c in columns)
        for col in columns:
            while len(col) < max_len:
                (ph,) = ax.plot([], [], alpha=0)
                col.append((ph, ""))

        handles, labels = [], []
        for col in columns:
            for h, lbl in col:
                handles.append(h)
                labels.append(lbl)

        ax.set_xlabel("Wavenumber [cpm]")
        ax.set_ylabel("\u03a6_T(k) [(K/m)\u00b2 cpm\u207b\u00b9]")
        ax.set_xlim(0.5, 300)
        ax.set_ylim(1e-11, None)
        ax.legend(handles, labels, fontsize=5, loc="best", ncol=n_probes)
        ax.set_title(
            f"\u03c7 spectra  fit={selected_method}  model={self.spectrum_model}\n"
            f"speed={mean_speed:.2f} m/s",
            fontsize=9,
        )
        ax.grid(True, alpha=0.3, which="both")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def quick_look(
    source,
    fft_length=256,
    f_AA=98.0,
    goodman=True,
    direction="down",
    P_min=0.5,
    W_min=0.3,
    min_duration=7.0,
    spec_P_range=None,
    chi_method=1,
    spectrum_model="kraichnan",
):
    """Open an interactive quick-look viewer for a .p file.

    Parameters
    ----------
    source : str or Path
        Path to a Rockland .p file.
    fft_length : int
        FFT segment length for spectral estimation.
    f_AA : float
        Anti-aliasing filter cutoff [Hz].
    goodman : bool
        Apply Goodman coherent noise removal to shear spectra.
    direction : str
        Profile direction: 'up' or 'down'.
    P_min : float
        Minimum pressure for profile detection [dbar].
    W_min : float
        Minimum fall rate for profile detection [dbar/s].
    min_duration : float
        Minimum profile duration [s].
    spec_P_range : tuple of (float, float) or None
        Pressure range (P_min, P_max) in dbar for spectral calculations.
        If None, uses the full profile depth range.
    chi_method : int
        Chi method for windowed profile estimates: 1 = from known epsilon,
        2 = MLE Batchelor fit.  All three methods are always shown on the
        chi spectra panel.
    spectrum_model : str
        Theoretical spectrum model: 'batchelor' or 'kraichnan'.

    Returns
    -------
    QuickLookViewer
        The viewer instance (useful for scripting).
    """
    pf = PFile(source)
    viewer = QuickLookViewer(
        pf,
        fft_length=fft_length,
        f_AA=f_AA,
        goodman=goodman,
        direction=direction,
        P_min=P_min,
        W_min=W_min,
        min_duration=min_duration,
        spec_P_range=spec_P_range,
        chi_method=chi_method,
        spectrum_model=spectrum_model,
    )
    viewer.show()
    return viewer

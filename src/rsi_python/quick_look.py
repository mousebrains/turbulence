# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Interactive quick-look viewer for Rockland Scientific profiler data.

Port of quick_look.m from the ODAS MATLAB library.
Provides a multi-panel figure with Prev/Next buttons to step through profiles.

Layout (2 rows x 4 columns):
  Row 0 (pressure y-axis, all linked):
    Overview | Shear probes | Temperature & fall rate | T1, T2, JAC_T
  Row 1 (spectra + profiles):
    Shear spectra | Chi spectra | ε & χ vs pressure | (hidden)
"""

import re
import warnings

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button
from scipy.signal import butter, filtfilt

from rsi_python.batchelor import batchelor_grad, batchelor_kB
from rsi_python.despike import despike
from rsi_python.fp07 import fp07_tau, fp07_transfer, gradT_noise
from rsi_python.nasmyth import X_95, nasmyth
from rsi_python.ocean import visc35
from rsi_python.p_file import PFile
from rsi_python.profile import _smooth_fall_rate, get_profiles
from rsi_python.spectral import csd_odas


def _hp_filter(x, fs, cutoff=1.0):
    """High-pass filter for shear display."""
    b, a = butter(1, cutoff / (fs / 2), btype="high")
    return filtfilt(b, a, x)


def _compute_profile_spectra(
    shear_data, accel_data, P_fast, T_slow, speed_fast, fs_fast, sel, fft_length, f_AA, do_goodman
):
    """Compute shear spectra and Nasmyth references for one profile segment.

    Returns (K, F, spectra_k, nasmyth_specs, epsilons, K_maxes, mean_speed, nu).
    """
    from rsi_python.goodman import clean_shear_spec

    sh_seg = np.column_stack([s[sel] for _, s in shear_data])
    ac_seg = np.column_stack([a[sel] for _, a in accel_data]) if accel_data else None

    mean_T = float(np.mean(T_slow))
    mean_speed = float(np.mean(np.abs(speed_fast[sel])))
    if mean_speed < 0.01:
        mean_speed = 0.01
    nu = float(visc35(mean_T))

    if do_goodman and ac_seg is not None and sh_seg.shape[0] > 2 * fft_length:
        clean_UU, AA, UU, UA, F = clean_shear_spec(ac_seg, sh_seg, fft_length, fs_fast)
        n_sh = sh_seg.shape[1]
        spectra = []
        for i in range(n_sh):
            spectra.append(np.real(clean_UU[:, i, i]))
    else:
        F = None
        spectra = []
        for i in range(sh_seg.shape[1]):
            Pxx, F, _, _ = csd_odas(sh_seg[:, i], None, fft_length, fs_fast)
            spectra.append(Pxx)

    K = F / mean_speed
    spectra_k = [s * mean_speed for s in spectra]
    K_AA = f_AA / mean_speed

    nasmyth_specs = []
    epsilons = []
    K_maxes = []
    for spec_k in spectra_k:
        valid = (K > 0) & (K <= K_AA)
        if np.sum(valid) < 3:
            epsilons.append(np.nan)
            nasmyth_specs.append(np.full_like(K, np.nan))
            K_maxes.append(np.nan)
            continue
        dk = np.gradient(K)
        variance = np.sum(spec_k[valid] * dk[valid])
        eps_est = 7.5 * nu * variance
        for _ in range(5):
            nas = nasmyth(eps_est, nu, K[valid])
            nas_var = np.sum(nas * dk[valid])
            if nas_var > 0:
                eps_est *= variance / nas_var
        epsilons.append(eps_est)
        nasmyth_specs.append(nasmyth(eps_est, nu, K))
        K_95 = X_95 * (max(eps_est, 1e-15) / nu**3) ** 0.25
        K_maxes.append(min(K_AA, K_95))

    return K, F, spectra_k, nasmyth_specs, epsilons, K_maxes, mean_speed, nu


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
):
    """Compute temperature gradient spectra and all three chi methods for one profile segment.

    Returns (K, F, obs_spectra, methods_results, noise_K, mean_speed, nu).

    methods_results is a dict keyed by method name ("M1", "M2-MLE", "M2-Iter"),
    each mapping to a list of per-probe (batch_spec, chi_val, kB_val) tuples.
    """
    from rsi_python.batchelor import KAPPA_T
    from rsi_python.chi import _iterative_fit, _mle_fit_kB

    mean_T = float(np.mean(T_slow))
    mean_speed = float(np.mean(np.abs(speed_fast[sel])))
    if mean_speed < 0.01:
        mean_speed = 0.01
    nu = float(visc35(mean_T))

    eps_vals = [e for e in epsilons if np.isfinite(e) and e > 0]
    eps_mean = np.nanmean(eps_vals) if eps_vals else 1e-8

    n_freq = fft_length // 2 + 1
    F = np.arange(n_freq) * fs_fast / fft_length
    K = F / mean_speed

    obs_spectra = []
    methods_results = {"M1": [], "M2-MLE": [], "M2-Iter": []}

    dg = diff_gains[0] if diff_gains else 0.94
    noise_K, _ = gradT_noise(F, mean_T, mean_speed, fs=fs_fast, diff_gain=dg)

    nan_result = (np.full(n_freq, np.nan), np.nan, np.nan)

    for ci, (name, data) in enumerate(therm_data):
        seg = data[sel]
        if len(seg) < 2 * fft_length:
            obs_spectra.append(np.full(n_freq, np.nan))
            for m in methods_results:
                methods_results[m].append(nan_result)
            continue

        Pxx, F_spec = csd_odas(seg, None, fft_length, fs_fast, overlap=fft_length // 2)[:2]
        spec_obs = Pxx * mean_speed

        correction = np.ones(n_freq)
        with np.errstate(divide="ignore", invalid="ignore"):
            correction[1:] = (
                np.pi * F_spec[1:] / (fs_fast * np.sin(np.pi * F_spec[1:] / fs_fast))
            ) ** 2
        correction = np.where(np.isfinite(correction), correction, 1.0)
        spec_obs = spec_obs * correction
        obs_spectra.append(spec_obs)

        dg_i = diff_gains[ci] if ci < len(diff_gains) else 0.94

        # --- Method 1: chi from known epsilon ---
        kB = batchelor_kB(eps_mean, nu)
        if kB > 1:
            tau0 = fp07_tau(mean_speed)
            H2 = fp07_transfer(F, tau0)
            K_AA = f_AA / mean_speed
            valid = (K > 0) & (K <= K_AA)
            if np.sum(valid) >= 3:
                obs_var = np.trapezoid(spec_obs[valid], K[valid])
                chi_trial = max(6 * KAPPA_T * obs_var, 1e-12)
                batch_spec = batchelor_grad(K, kB, chi_trial) * H2
                model_var = np.trapezoid(batch_spec[valid], K[valid])
                if model_var > 0:
                    scale = obs_var / model_var
                    batch_spec = batch_spec * scale
                    chi_val = chi_trial * scale
                else:
                    chi_val = chi_trial
                methods_results["M1"].append((batch_spec, chi_val, kB))
            else:
                methods_results["M1"].append(nan_result)
        else:
            methods_results["M1"].append(nan_result)

        # --- Method 2 MLE ---
        K_AA = f_AA / mean_speed
        valid_m2 = (K > 0) & (K <= K_AA)
        if np.sum(valid_m2) >= 3:
            chi_obs = max(6 * KAPPA_T * np.trapezoid(spec_obs[valid_m2], K[valid_m2]), 1e-10)
        else:
            chi_obs = 1e-10
        kB_best, chi_val, _, _, spec_batch, _, _ = _mle_fit_kB(
            spec_obs,
            K,
            chi_obs,
            nu,
            mean_T,
            mean_speed,
            f_AA,
            "single_pole",
            "batchelor",
            fs_fast,
            dg_i,
            fft_length,
        )
        methods_results["M2-MLE"].append((spec_batch, chi_val, kB_best))

        # --- Method 2 Iterative ---
        kB_it, chi_it, _, _, spec_it, _, _ = _iterative_fit(
            spec_obs,
            K,
            nu,
            mean_T,
            mean_speed,
            f_AA,
            "single_pole",
            "batchelor",
            fs_fast,
            dg_i,
            fft_length,
        )
        methods_results["M2-Iter"].append((spec_it, chi_it, kB_it))

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
):
    """Compute windowed epsilon and chi estimates for a profile segment.

    Returns (P_windows, eps_array, chi_array, shear_names, therm_names).
    eps_array shape: (n_shear, n_windows)
    chi_array shape: (n_therm, n_windows)

    Parameters
    ----------
    chi_method : int
        1 = chi from known epsilon (Method 1), 2 = MLE Batchelor fit (Method 2).
    """
    from rsi_python.batchelor import KAPPA_T
    from rsi_python.chi import _mle_fit_kB
    from rsi_python.goodman import clean_shear_spec

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

    for idx in range(n_windows):
        s = seg_start + idx * step
        e = s + diss_length
        if e > seg_end:
            break
        w_sel = slice(s, e)

        W = float(np.mean(np.abs(speed_fast[w_sel])))
        if W < 0.01:
            W = 0.01
        mean_T = float(np.mean(T_slow))
        nu = float(visc35(mean_T))
        P_windows[idx] = float(np.mean(P_fast[w_sel]))

        K_AA = f_AA / W

        # Epsilon from shear probes
        eps_window = []
        if n_shear > 0:
            sh_seg = np.column_stack([d[w_sel] for _, d in shear_data])
            ac_seg = np.column_stack([d[w_sel] for _, d in accel_data]) if accel_data else None

            if do_goodman and ac_seg is not None and sh_seg.shape[0] > 2 * fft_length:
                clean_UU, _, _, _, F = clean_shear_spec(ac_seg, sh_seg, fft_length, fs_fast)
                for ci in range(n_shear):
                    spec_f = np.real(clean_UU[:, ci, ci])
                    K = F / W
                    spec_k = spec_f * W
                    valid = (K > 0) & (K <= K_AA)
                    if np.sum(valid) >= 3:
                        dk = np.gradient(K)
                        var = np.sum(spec_k[valid] * dk[valid])
                        eps_est = 7.5 * nu * var
                        for _ in range(3):
                            nas = nasmyth(eps_est, nu, K[valid])
                            nas_var = np.sum(nas * dk[valid])
                            if nas_var > 0:
                                eps_est *= var / nas_var
                        eps_arr[ci, idx] = eps_est
                        eps_window.append(eps_est)
            else:
                for ci in range(n_shear):
                    Pxx, F, _, _ = csd_odas(sh_seg[:, ci], None, fft_length, fs_fast)
                    K = F / W
                    spec_k = Pxx * W
                    valid = (K > 0) & (K <= K_AA)
                    if np.sum(valid) >= 3:
                        dk = np.gradient(K)
                        var = np.sum(spec_k[valid] * dk[valid])
                        eps_est = 7.5 * nu * var
                        for _ in range(3):
                            nas = nasmyth(eps_est, nu, K[valid])
                            nas_var = np.sum(nas * dk[valid])
                            if nas_var > 0:
                                eps_est *= var / nas_var
                        eps_arr[ci, idx] = eps_est
                        eps_window.append(eps_est)

        # Chi from thermistor gradient channels
        if n_therm > 0:
            n_freq = fft_length // 2 + 1
            F_chi = np.arange(n_freq) * fs_fast / fft_length
            K_chi = F_chi / W

            if chi_method == 2:
                # Method 2: MLE Batchelor fit per window
                for ci in range(n_therm):
                    seg = therm_data[ci][1][w_sel]
                    if len(seg) < 2 * fft_length:
                        continue
                    Pxx, _ = csd_odas(seg, None, fft_length, fs_fast, overlap=fft_length // 2)[:2]
                    spec_obs = Pxx * W
                    valid = (K_chi > 0) & (K_chi <= K_AA)
                    if np.sum(valid) < 3:
                        continue
                    chi_obs = max(
                        6 * KAPPA_T * np.trapezoid(spec_obs[valid], K_chi[valid]),
                        1e-10,
                    )
                    dg_i = diff_gains[ci] if ci < len(diff_gains) else 0.94
                    _, chi_val, _, _, _, _, _ = _mle_fit_kB(
                        spec_obs,
                        K_chi,
                        chi_obs,
                        nu,
                        mean_T,
                        W,
                        f_AA,
                        "single_pole",
                        "batchelor",
                        fs_fast,
                        dg_i,
                        fft_length,
                    )
                    if np.isfinite(chi_val) and chi_val > 0:
                        chi_arr[ci, idx] = chi_val
            else:
                # Method 1: chi from known epsilon
                eps_for_chi = [e for e in eps_window if np.isfinite(e) and e > 0]
                eps_mean = np.nanmean(eps_for_chi) if eps_for_chi else 1e-8
                kB = batchelor_kB(eps_mean, nu)

                for ci in range(n_therm):
                    seg = therm_data[ci][1][w_sel]
                    if len(seg) < 2 * fft_length:
                        continue
                    Pxx, _ = csd_odas(seg, None, fft_length, fs_fast, overlap=fft_length // 2)[:2]
                    spec_obs = Pxx * W
                    valid = (K_chi > 0) & (K_chi <= K_AA)
                    if np.sum(valid) >= 3 and kB > 1:
                        obs_var = np.trapezoid(spec_obs[valid], K_chi[valid])
                        chi_arr[ci, idx] = max(6 * KAPPA_T * obs_var, 1e-15)

    return P_windows, eps_arr, chi_arr


class QuickLookViewer:
    """Interactive multi-panel profile viewer with Prev/Next navigation."""

    def __init__(
        self,
        pf,
        fft_length=256,
        f_AA=98.0,
        goodman=True,
        direction="down",
        P_min=0.5,
        W_min=0.3,
        min_duration=7.0,
        spec_P_range=None,
        chi_method=1,
    ):
        self.pf = pf
        self.fft_length = fft_length
        self.f_AA = f_AA
        self.goodman = goodman
        self.spec_P_range = spec_P_range
        self.chi_method = chi_method

        # Extract channel data
        sh_re = re.compile(r"^sh\d+$")
        ac_re = re.compile(r"^A[xyz]$")
        T_re = re.compile(r"^T\d+$")
        dT_re = re.compile(r"^T\d+_dT\d+$")

        self.shear = sorted(
            [(n, pf.channels[n]) for n in pf._fast_channels if sh_re.match(n)],
            key=lambda x: x[0],
        )
        self.accel = sorted(
            [(n, pf.channels[n]) for n in pf._fast_channels if ac_re.match(n)],
            key=lambda x: x[0],
        )
        self.therm_slow = sorted(
            [(n, pf.channels[n]) for n in pf._slow_channels if T_re.match(n)],
            key=lambda x: x[0],
        )
        self.therm_fast = sorted(
            [(n, pf.channels[n]) for n in pf._fast_channels if dT_re.match(n)],
            key=lambda x: x[0],
        )
        self.diff_gains = []
        for name, _ in self.therm_fast:
            ch_cfg = next((ch for ch in pf.config["channels"] if ch.get("name") == name), {})
            self.diff_gains.append(float(ch_cfg.get("diff_gain", "0.94")))

        # All temperature channels for the temperature comparison panel
        # T1, T2 (slow FP07) + JAC_T (slow Sea-Bird)
        self.temp_channels = []
        for name in ["T1", "T2", "JAC_T"]:
            if name in pf.channels:
                self.temp_channels.append((name, pf.channels[name]))

        self.P = pf.channels["P"]
        self.T = pf.channels.get("T1", pf.channels.get("T", np.zeros_like(self.P)))
        self.t_fast = pf.t_fast
        self.t_slow = pf.t_slow
        self.fs_fast = pf.fs_fast
        self.fs_slow = pf.fs_slow

        # Detect profiles
        W = _smooth_fall_rate(self.P, self.fs_slow)
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

        # Interpolate slow → fast
        self.P_fast = np.interp(self.t_fast, self.t_slow, self.P)
        self.speed_fast = np.abs(np.interp(self.t_fast, self.t_slow, W))
        self.ratio = round(self.fs_fast / self.fs_slow)

        self.profile_idx = 0
        self.fig = None
        self._twin_ax = None
        self._spec_bin_width = None  # computed per profile in _draw()

    def _slow_to_fast_slice(self, s_slow, e_slow):
        """Convert slow-channel indices to fast-channel slice."""
        s_fast = s_slow * self.ratio
        e_fast = min((e_slow + 1) * self.ratio, len(self.t_fast))
        return slice(s_fast, e_fast)

    def _spec_fast_slice(self, sel_fast):
        """Restrict a fast slice to the spectral pressure range, if set."""
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
        """Compute the spectral bin width in dbar for the current profile."""
        sel = self._slow_to_fast_slice(s_slow, e_slow)
        mean_speed = float(np.mean(np.abs(self.speed_fast[sel])))
        if mean_speed < 0.01:
            mean_speed = 0.01
        diss_length = 2 * self.fft_length
        return diss_length / self.fs_fast * mean_speed

    def _draw(self):
        """Draw all panels for the current profile."""
        s_slow, e_slow = self.profiles[self.profile_idx]
        sel_fast = self._slow_to_fast_slice(s_slow, e_slow)

        # Compute bin width for stepping
        self._spec_bin_width = self._spec_bin_dbar(s_slow, e_slow)

        sel_spec = self._spec_fast_slice(sel_fast)

        # Remove twin axis from previous draw before clearing
        if self._twin_ax is not None:
            self._twin_ax.remove()
            self._twin_ax = None

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            for ax in self.axes.flat:
                ax.clear()

        # Re-invert pressure y-axes (clear resets inversion)
        self.axes[0, 0].invert_yaxis()

        # Pre-compute shear spectra (used by panels 4, 5, 6, 7)
        n_spec = sel_spec.stop - sel_spec.start
        self._cached_shear = None
        if self.shear and n_spec >= 2 * self.fft_length:
            self._cached_shear = _compute_profile_spectra(
                self.shear,
                self.accel,
                self.P_fast,
                self.T,
                self.speed_fast,
                self.fs_fast,
                sel_spec,
                self.fft_length,
                self.f_AA,
                self.goodman,
            )

        self._draw_overview(s_slow, e_slow)
        self._draw_shear(sel_fast)
        self._draw_temperature(s_slow, e_slow)
        self._draw_all_temps(s_slow, e_slow)
        self._draw_spectra(sel_spec)
        self._draw_chi_spectra(sel_spec)
        self._draw_eps_chi_profile(sel_fast)

        P_lo, P_hi = self.P[s_slow], self.P[e_slow]
        title = (
            f"{self.pf.filepath.name}  —  "
            f"Profile {self.profile_idx + 1}/{len(self.profiles)}  —  "
            f"P: {P_lo:.1f}–{P_hi:.1f} dbar"
        )
        if self.spec_P_range is not None:
            sp_lo, sp_hi = self.spec_P_range
            title += f"  [spectra: {sp_lo:.1f}–{sp_hi:.1f} dbar]"
        self.fig.suptitle(title, fontsize=11, fontweight="bold")
        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Row 0: Pressure y-axis panels
    # ------------------------------------------------------------------

    def _draw_overview(self, s_slow, e_slow):
        """Panel (0,0): Pressure vs time, current profile highlighted."""
        ax = self.axes[0, 0]
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

    def _draw_shear(self, sel_fast):
        """Panel (0,1): HP-filtered shear signals vs pressure."""
        ax = self.axes[0, 1]
        P_prof = self.P_fast[sel_fast]

        if not self.shear:
            ax.text(0.5, 0.5, "No shear channels", transform=ax.transAxes, ha="center", va="center")
            return

        for i, (name, data) in enumerate(self.shear):
            seg = data[sel_fast]
            if len(seg) > 2 * int(self.fs_fast):
                seg_hp = _hp_filter(seg, self.fs_fast, cutoff=1.0)
            else:
                seg_hp = seg - np.mean(seg)
            seg_hp, _, _, _ = despike(seg_hp, self.fs_fast, thresh=8, smooth=0.5)
            offset = i * 0.5
            ax.plot(seg_hp + offset, P_prof, linewidth=0.3, label=name)

        if self.spec_P_range is not None:
            ax.axhspan(*self.spec_P_range, alpha=0.08, color="C2", zorder=0)

        ax.set_xlabel("Shear [s⁻¹] (HP, offset)")
        ax.legend(fontsize=7, loc="lower right")
        ax.set_title("Shear probes", fontsize=9)
        ax.grid(True, alpha=0.3)

    def _draw_temperature(self, s_slow, e_slow):
        """Panel (0,2): Slow temperature + fall rate vs pressure."""
        ax = self.axes[0, 2]
        P_prof = self.P[s_slow : e_slow + 1]

        colors = ["C3", "C1", "C4", "C5"]
        for i, (name, data) in enumerate(self.therm_slow):
            c = colors[i % len(colors)]
            ax.plot(data[s_slow : e_slow + 1], P_prof, c, linewidth=1.0, label=name)

        # Fall rate on twin axis
        self._twin_ax = ax.twiny()
        W_prof = self.W[s_slow : e_slow + 1]
        self._twin_ax.plot(W_prof, P_prof, "C2--", linewidth=0.8, alpha=0.6, label="dP/dt")
        self._twin_ax.set_xlabel("Fall rate [dbar/s]", fontsize=8, color="C2")
        self._twin_ax.tick_params(axis="x", labelsize=7, colors="C2")

        if self.spec_P_range is not None:
            ax.axhspan(*self.spec_P_range, alpha=0.08, color="C2", zorder=0)

        ax.set_xlabel("Temperature [°C]")
        ax.legend(fontsize=7, loc="lower left")
        ax.set_title("Temperature & fall rate", fontsize=9)
        ax.grid(True, alpha=0.3)

    def _draw_all_temps(self, s_slow, e_slow):
        """Panel (0,3): T1, T2, JAC_T vs pressure."""
        ax = self.axes[0, 3]
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

        if self.spec_P_range is not None:
            ax.axhspan(*self.spec_P_range, alpha=0.08, color="C2", zorder=0)

        ax.set_xlabel("Temperature [°C]")
        ax.legend(fontsize=7, loc="lower left")
        ax.set_title("T1, T2, JAC_T", fontsize=9)
        ax.grid(True, alpha=0.3)

    # ------------------------------------------------------------------
    # Row 1: Spectral panels + eps/chi profile
    # ------------------------------------------------------------------

    def _draw_spectra(self, sel_spec):
        """Panel (1,0): Shear wavenumber spectra with Nasmyth + K_max."""
        ax = self.axes[1, 0]

        if self._cached_shear is None:
            ax.text(
                0.5,
                0.5,
                "Insufficient data for spectra",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
            return

        K, F, spectra_k, nasmyth_specs, epsilons, K_maxes, mean_speed, nu = self._cached_shear

        colors = ["C0", "C1", "C4", "C5"]
        for i, (name, _) in enumerate(self.shear):
            c = colors[i % len(colors)]
            ax.loglog(K, spectra_k[i], c, linewidth=0.8, label=name)
            if np.isfinite(epsilons[i]):
                ax.loglog(
                    K,
                    nasmyth_specs[i],
                    c,
                    linewidth=0.5,
                    linestyle="--",
                    alpha=0.7,
                    label=f"Nasmyth (ε={epsilons[i]:.1e})",
                )
                if np.isfinite(K_maxes[i]):
                    ax.axvline(
                        K_maxes[i],
                        color=c,
                        linestyle="-.",
                        linewidth=0.8,
                        alpha=0.6,
                        label=f"K_max={K_maxes[i]:.0f} cpm",
                    )
                    # Filled inverted triangle at K_max on the observed spectrum
                    k_idx = np.argmin(np.abs(K - K_maxes[i]))
                    ax.plot(
                        K_maxes[i],
                        spectra_k[i][k_idx],
                        marker="v",
                        color=c,
                        markersize=8,
                        zorder=5,
                    )

        for exp in [-9, -8, -7, -6, -5]:
            eps_ref = 10.0**exp
            nas_ref = nasmyth(eps_ref, nu, K)
            ax.loglog(K, nas_ref, "0.8", linewidth=0.3)
            y_val = nas_ref[len(K) // 2]
            if np.isfinite(y_val) and y_val > 0:
                ax.text(K[len(K) // 2], y_val, f"{exp}", fontsize=6, color="0.5", va="bottom")

        K_AA = self.f_AA / mean_speed
        ax.axvline(K_AA, color="0.5", linestyle=":", linewidth=0.5, alpha=0.5)

        P_lo = float(self.P_fast[sel_spec.start])
        P_hi = float(self.P_fast[min(sel_spec.stop - 1, len(self.P_fast) - 1)])
        ax.set_xlabel("Wavenumber [cpm]")
        ax.set_ylabel("Φ(k) [s⁻² cpm⁻¹]")
        ax.set_xlim(0.5, 300)
        ax.set_ylim(1e-9, 1e0)
        ax.legend(fontsize=5, loc="best", ncol=2)
        ax.set_title(
            f"Shear spectra  P={P_lo:.1f}–{P_hi:.1f}  speed={mean_speed:.2f} m/s",
            fontsize=9,
        )
        ax.grid(True, alpha=0.3, which="both")

    def _draw_chi_spectra(self, sel_spec):
        """Panel (1,1): Temperature gradient spectra + all three chi methods."""
        ax = self.axes[1, 1]
        self._cached_chi = None

        if not self.therm_fast:
            ax.text(
                0.5,
                0.5,
                "No thermistor gradient channels",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
            return

        n_fast = sel_spec.stop - sel_spec.start
        if n_fast < 2 * self.fft_length:
            ax.text(
                0.5,
                0.5,
                "Insufficient data for chi spectra",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
            return

        eps_for_chi = []
        if self._cached_shear is not None:
            _, _, _, _, epsilons, _, _, _ = self._cached_shear
            eps_for_chi = epsilons

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
        )
        K, F, obs_spectra, methods_results, noise_K, mean_speed, nu = self._cached_chi

        # Line styles for each method
        method_styles = {
            "M1": ("--", 0.7),
            "M2-MLE": ("-.", 0.7),
            "M2-Iter": (":", 0.7),
        }
        colors = ["C3", "C1", "C4", "C5"]
        for i, (name, _) in enumerate(self.therm_fast):
            c = colors[i % len(colors)]
            valid = np.isfinite(obs_spectra[i]) & (obs_spectra[i] > 0)
            if np.any(valid):
                ax.loglog(K[valid], obs_spectra[i][valid], c, linewidth=0.8, label=name)

            for method_name, (ls, alpha) in method_styles.items():
                batch_spec, chi_val, kB_val = methods_results[method_name][i]
                valid_b = np.isfinite(batch_spec) & (batch_spec > 0)
                if np.any(valid_b):
                    chi_str = f"{chi_val:.1e}" if np.isfinite(chi_val) else "N/A"
                    ax.loglog(
                        K[valid_b],
                        batch_spec[valid_b],
                        c,
                        linewidth=0.5,
                        linestyle=ls,
                        alpha=alpha,
                        label=f"{method_name} χ={chi_str}",
                    )

        valid_n = np.isfinite(noise_K) & (noise_K > 0) & (K > 0)
        if np.any(valid_n):
            ax.loglog(
                K[valid_n],
                noise_K[valid_n],
                "0.5",
                linewidth=0.8,
                linestyle=":",
                label="Noise",
            )

        K_AA = self.f_AA / mean_speed
        ax.axvline(K_AA, color="0.5", linestyle=":", linewidth=0.5, alpha=0.5)

        ax.set_xlabel("Wavenumber [cpm]")
        ax.set_ylabel("Φ_T(k) [(K/m)² cpm⁻¹]")
        ax.set_xlim(0.5, 300)
        ax.legend(fontsize=5, loc="best", ncol=2)
        ax.set_title(f"Temperature gradient spectra (speed={mean_speed:.2f} m/s)", fontsize=9)
        ax.grid(True, alpha=0.3, which="both")

    def _draw_eps_chi_profile(self, sel_spec):
        """Panel (1,2): Epsilon and chi vs pressure (windowed estimates)."""
        ax = self.axes[1, 2]

        n_spec = sel_spec.stop - sel_spec.start
        if n_spec < 2 * self.fft_length:
            ax.text(
                0.5,
                0.5,
                "Insufficient data",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
            return

        P_win, eps_arr, chi_arr = _compute_windowed_eps_chi(
            self.shear,
            self.accel,
            self.therm_fast,
            self.diff_gains,
            self.P_fast,
            self.T,
            self.speed_fast,
            self.fs_fast,
            sel_spec,
            self.fft_length,
            self.f_AA,
            self.goodman,
            chi_method=self.chi_method,
        )

        if len(P_win) == 0:
            ax.text(
                0.5,
                0.5,
                "No valid estimates",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
            return

        colors_eps = ["C0", "C1"]
        colors_chi = ["C3", "C4"]

        has_data = False
        for i, (name, _) in enumerate(self.shear):
            c = colors_eps[i % len(colors_eps)]
            valid = np.isfinite(eps_arr[i]) & (eps_arr[i] > 0)
            if np.any(valid):
                ax.plot(
                    eps_arr[i, valid],
                    P_win[valid],
                    f"{c}o-",
                    markersize=3,
                    linewidth=0.8,
                    label=f"ε ({name})",
                )
                has_data = True

        for i, (name, _) in enumerate(self.therm_fast):
            c = colors_chi[i % len(colors_chi)]
            valid = np.isfinite(chi_arr[i]) & (chi_arr[i] > 0)
            if np.any(valid):
                ax.plot(
                    chi_arr[i, valid],
                    P_win[valid],
                    f"{c}s-",
                    markersize=3,
                    linewidth=0.8,
                    label=f"χ ({name})",
                )
                has_data = True

        if has_data:
            ax.set_xscale("log")
            ax.set_xlabel("ε [W/kg] / χ [K²/s]")
            ax.set_ylabel("Pressure [dbar]")
            ax.legend(fontsize=6, loc="lower left")
            ax.set_title("ε & χ profiles", fontsize=9)
            ax.grid(True, alpha=0.3, which="both")
        else:
            ax.text(
                0.5,
                0.5,
                "No valid ε or χ estimates",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )

    # ------------------------------------------------------------------
    # Figure setup and navigation
    # ------------------------------------------------------------------

    def show(self):
        """Create the figure and display it."""
        self.fig, self.axes = plt.subplots(
            2,
            4,
            figsize=(24, 9),
            gridspec_kw={"hspace": 0.35, "wspace": 0.32},
        )
        self.fig.subplots_adjust(bottom=0.08, top=0.92, left=0.04, right=0.98)

        # Link all pressure y-axes: row 0 + eps/chi profile panel (1,2)
        p_ref = self.axes[0, 0]
        for col in range(1, 4):
            self.axes[0, col].sharey(p_ref)
        self.axes[1, 2].sharey(p_ref)
        # Invert once on the reference (shared propagates)
        p_ref.invert_yaxis()

        # Link wavenumber x-axes: shear spectra (1,0) and chi spectra (1,1)
        self.axes[1, 1].sharex(self.axes[1, 0])

        # Hide unused panel (1,3)
        self.axes[1, 3].set_visible(False)

        # Navigation buttons
        ax_prev = self.fig.add_axes([0.35, 0.01, 0.07, 0.035])
        ax_next = self.fig.add_axes([0.43, 0.01, 0.07, 0.035])
        self.btn_prev = Button(ax_prev, "◀ Prev")
        self.btn_next = Button(ax_next, "Next ▶")
        self.btn_prev.on_clicked(self._on_prev)
        self.btn_next.on_clicked(self._on_next)

        # Spectral depth range buttons
        ax_up = self.fig.add_axes([0.55, 0.01, 0.07, 0.035])
        ax_dn = self.fig.add_axes([0.63, 0.01, 0.07, 0.035])
        self.btn_spec_up = Button(ax_up, "▲ Spec")
        self.btn_spec_dn = Button(ax_dn, "▼ Spec")
        self.btn_spec_up.on_clicked(self._on_spec_up)
        self.btn_spec_dn.on_clicked(self._on_spec_dn)

        self._draw()
        plt.show()

    def _on_prev(self, event):
        if self.profile_idx > 0:
            self.profile_idx -= 1
            self._draw()

    def _on_next(self, event):
        if self.profile_idx < len(self.profiles) - 1:
            self.profile_idx += 1
            self._draw()

    def _on_spec_up(self, event):
        """Shift spectral pressure range upward (shallower) by one bin width."""
        s_slow, e_slow = self.profiles[self.profile_idx]
        P_top = float(min(self.P[s_slow], self.P[e_slow]))
        bw = self._spec_bin_width or 1.0
        if self.spec_P_range is None:
            # Initialize to first bin at top of profile
            self.spec_P_range = (P_top, P_top + bw)
        else:
            new_lo = self.spec_P_range[0] - bw
            new_hi = self.spec_P_range[1] - bw
            if new_lo < P_top:
                new_lo = P_top
                new_hi = P_top + bw
            self.spec_P_range = (new_lo, new_hi)
        self._draw()

    def _on_spec_dn(self, event):
        """Shift spectral pressure range downward (deeper) by one bin width."""
        s_slow, e_slow = self.profiles[self.profile_idx]
        P_bot = float(max(self.P[s_slow], self.P[e_slow]))
        P_top = float(min(self.P[s_slow], self.P[e_slow]))
        bw = self._spec_bin_width or 1.0
        if self.spec_P_range is None:
            # Initialize to first bin at top of profile
            self.spec_P_range = (P_top, P_top + bw)
        else:
            new_lo = self.spec_P_range[0] + bw
            new_hi = self.spec_P_range[1] + bw
            if new_hi > P_bot:
                new_hi = P_bot
                new_lo = P_bot - bw
            self.spec_P_range = (new_lo, new_hi)
        self._draw()


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
    )
    viewer.show()
    return viewer

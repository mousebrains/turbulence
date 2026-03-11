# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Dissipation quality viewer for Rockland Scientific profiler data.

Interactive multi-panel viewer focused on comparing epsilon, chi from
Batchelor vs Kraichnan models, and the Lueck (2022) figure of merit (FM).

Layout (2 rows x 4 columns):
  Row 0 (profile panels, pressure y-axis linked):
    Overview | ε profile | χ profile | FM profile
  Row 1 (spectra and comparison):
    ε spectra at depth | χ spectra at depth | χ Batch. vs Kraich. | Temperature
"""

import re
import warnings

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button

from rsi_python.chi import _bilinear_correction
from rsi_python.dissipation import _estimate_epsilon
from rsi_python.fp07 import fp07_tau, fp07_transfer, gradT_noise
from rsi_python.nasmyth import nasmyth
from rsi_python.ocean import visc35
from rsi_python.p_file import PFile
from rsi_python.profile import _smooth_fall_rate, get_profiles
from rsi_python.spectral import csd_odas

# ---------------------------------------------------------------------------
# Windowed computation: epsilon + FM + chi (both models)
# ---------------------------------------------------------------------------


def _compute_windowed_diss(
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
):
    """Compute windowed epsilon, FM, and chi (Batchelor + Kraichnan) estimates.

    Returns dict with keys:
        P_windows     (n_win,)
        epsilon       (n_shear, n_win)
        FM            (n_shear, n_win)
        fom           (n_shear, n_win)
        mad           (n_shear, n_win)
        K_max_ratio   (n_shear, n_win)
        chi_batchelor (n_therm, n_win)
        chi_kraichnan (n_therm, n_win)
    """
    from rsi_python.chi import _chi_from_epsilon
    from rsi_python.goodman import clean_shear_spec

    diss_length = 2 * fft_length
    overlap = diss_length // 2
    step = diss_length - overlap

    seg_start = sel.start
    seg_end = sel.stop
    N = seg_end - seg_start

    n_shear = len(shear_data)
    n_therm = len(therm_data)
    n_accel = len(accel_data)
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
    }
    if n_windows == 0:
        return empty

    # Pre-allocate
    P_windows = np.full(n_windows, np.nan)
    eps_arr = np.full((n_shear, n_windows), np.nan)
    FM_arr = np.full((n_shear, n_windows), np.nan)
    fom_arr = np.full((n_shear, n_windows), np.nan)
    mad_arr = np.full((n_shear, n_windows), np.nan)
    Kmr_arr = np.full((n_shear, n_windows), np.nan)
    chi_batch_arr = np.full((n_therm, n_windows), np.nan)
    chi_kraich_arr = np.full((n_therm, n_windows), np.nan)

    # Degrees of freedom for FM
    num_ffts = 2 * (diss_length // fft_length) - 1
    n_v = n_accel if do_goodman else 0

    f_AA_eff = 0.9 * f_AA

    for idx in range(n_windows):
        s = seg_start + idx * step
        e = s + diss_length
        if e > seg_end:
            break
        w_sel = slice(s, e)

        W = float(np.mean(np.abs(speed_fast[w_sel])))
        if W < 0.01:
            warnings.warn(
                f"Speed {W:.4f} m/s below minimum; clamped to 0.01 m/s",
                stacklevel=2,
            )
            W = 0.01  # minimum speed to avoid wavenumber singularity
        mean_T = float(np.mean(T_slow))
        nu = float(visc35(mean_T))
        P_windows[idx] = float(np.mean(P_fast[w_sel]))

        # --- Epsilon from shear probes ---
        eps_window = []
        if n_shear > 0:
            sh_seg = np.column_stack([d[w_sel] for _, d in shear_data])
            ac_seg = np.column_stack([d[w_sel] for _, d in accel_data]) if accel_data else None

            if do_goodman and ac_seg is not None and sh_seg.shape[0] > 2 * fft_length:
                clean_UU, _, _, _, F_g = clean_shear_spec(ac_seg, sh_seg, fft_length, fs_fast)
                K_g = F_g / W
                K_AA_g = f_AA_eff / W

                # Macoun-Lueck wavenumber correction
                correction = np.ones_like(K_g)
                mask = K_g <= 150
                correction[mask] = 1 + (K_g[mask] / 48) ** 2

                for ci in range(n_shear):
                    spec_k = np.real(clean_UU[:, ci, ci]) * W * correction
                    e4, k_max, mad, meth, nas, fom, Kmr, FM = _estimate_epsilon(
                        K_g,
                        spec_k,
                        nu,
                        K_AA_g,
                        3,
                        1.5e-5,
                        num_ffts=num_ffts,
                        n_v=n_v,
                    )
                    eps_arr[ci, idx] = e4
                    FM_arr[ci, idx] = FM
                    fom_arr[ci, idx] = fom
                    mad_arr[ci, idx] = mad
                    Kmr_arr[ci, idx] = Kmr
                    if np.isfinite(e4) and e4 > 0:
                        eps_window.append(e4)
            else:
                for ci in range(n_shear):
                    Pxx, F_s, _, _ = csd_odas(sh_seg[:, ci], None, fft_length, fs_fast)
                    K_s = F_s / W
                    correction = np.ones_like(K_s)
                    mask_c = K_s <= 150
                    correction[mask_c] = 1 + (K_s[mask_c] / 48) ** 2
                    spec_k = Pxx * W * correction
                    K_AA_s = f_AA_eff / W

                    e4, k_max, mad, meth, nas, fom, Kmr, FM = _estimate_epsilon(
                        K_s,
                        spec_k,
                        nu,
                        K_AA_s,
                        3,
                        1.5e-5,
                        num_ffts=num_ffts,
                        n_v=0,
                    )
                    eps_arr[ci, idx] = e4
                    FM_arr[ci, idx] = FM
                    fom_arr[ci, idx] = fom
                    mad_arr[ci, idx] = mad
                    Kmr_arr[ci, idx] = Kmr
                    if np.isfinite(e4) and e4 > 0:
                        eps_window.append(e4)

        # --- Chi from thermistors (both Batchelor and Kraichnan) ---
        if n_therm > 0 and eps_window:
            eps_mean = np.nanmean(eps_window)

            for ci in range(n_therm):
                seg = therm_data[ci][1][w_sel]
                if len(seg) < 2 * fft_length:
                    continue
                Pxx_t, F_t = csd_odas(seg, None, fft_length, fs_fast, overlap=fft_length // 2)[:2]
                spec_obs = Pxx_t * W
                K_t = F_t / W

                dg = diff_gains[ci] if ci < len(diff_gains) else 0.94

                # Batchelor model
                chi_b, _, _, _, _, _ = _chi_from_epsilon(
                    spec_obs,
                    K_t,
                    eps_mean,
                    nu,
                    mean_T,
                    W,
                    f_AA_eff,
                    "single_pole",
                    "batchelor",
                    fs_fast,
                    dg,
                    fft_length,
                )
                chi_batch_arr[ci, idx] = chi_b

                # Kraichnan model
                chi_k, _, _, _, _, _ = _chi_from_epsilon(
                    spec_obs,
                    K_t,
                    eps_mean,
                    nu,
                    mean_T,
                    W,
                    f_AA_eff,
                    "single_pole",
                    "kraichnan",
                    fs_fast,
                    dg,
                    fft_length,
                )
                chi_kraich_arr[ci, idx] = chi_k

    return {
        "P_windows": P_windows,
        "epsilon": eps_arr,
        "FM": FM_arr,
        "fom": fom_arr,
        "mad": mad_arr,
        "K_max_ratio": Kmr_arr,
        "chi_batchelor": chi_batch_arr,
        "chi_kraichnan": chi_kraich_arr,
    }


# ---------------------------------------------------------------------------
# Spectra at a single depth for the stepper panel
# ---------------------------------------------------------------------------


def _compute_depth_spectra(
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
):
    """Compute shear + chi spectra at one depth for the stepper panel.

    Returns dict with shear spectra, Nasmyth fits, chi gradient spectra,
    and both Batchelor/Kraichnan model fits.
    """
    from rsi_python.chi import _chi_from_epsilon
    from rsi_python.goodman import clean_shear_spec

    n_fast = sel.stop - sel.start
    n_shear = len(shear_data)
    n_therm = len(therm_data)
    n_freq = fft_length // 2 + 1
    f_AA_eff = 0.9 * f_AA

    W = float(np.mean(np.abs(speed_fast[sel])))
    if W < 0.01:
        warnings.warn(
            f"Speed {W:.4f} m/s below minimum; clamped to 0.01 m/s",
            stacklevel=2,
        )
        W = 0.01  # minimum speed to avoid wavenumber singularity
    mean_T = float(np.mean(T_slow))
    nu = float(visc35(mean_T))

    F = np.arange(n_freq) * fs_fast / fft_length
    K = F / W
    K_AA = f_AA_eff / W

    result = {
        "K": K,
        "F": F,
        "W": W,
        "nu": nu,
        "shear_specs": [],
        "nasmyth_specs": [],
        "epsilons": [],
        "methods": [],
        "K_maxes": [],
        "chi_obs_specs": [],
        "chi_batch_specs": [],
        "chi_kraich_specs": [],
        "chi_batch_vals": [],
        "chi_kraich_vals": [],
        "noise_K": None,
    }

    if n_fast < 2 * fft_length:
        return result

    # Shear spectra
    num_ffts = 2 * (n_fast // fft_length) - 1
    n_accel = len(accel_data)
    n_v = n_accel if do_goodman else 0

    epsilons = []
    if n_shear > 0:
        sh_seg = np.column_stack([d[sel] for _, d in shear_data])
        ac_seg = np.column_stack([d[sel] for _, d in accel_data]) if accel_data else None

        if do_goodman and ac_seg is not None:
            clean_UU, _, _, _, F_g = clean_shear_spec(ac_seg, sh_seg, fft_length, fs_fast)
            K_g = F_g / W

            correction = np.ones_like(K_g)
            mask = K_g <= 150
            correction[mask] = 1 + (K_g[mask] / 48) ** 2

            for ci in range(n_shear):
                spec_k = np.real(clean_UU[:, ci, ci]) * W * correction
                e4, k_max, _, meth, nas, _, _, _ = _estimate_epsilon(
                    K_g,
                    spec_k,
                    nu,
                    K_AA,
                    3,
                    1.5e-5,
                    num_ffts=num_ffts,
                    n_v=n_v,
                )
                result["shear_specs"].append(spec_k)
                result["nasmyth_specs"].append(nas)
                result["epsilons"].append(e4)
                result["methods"].append(meth)
                result["K_maxes"].append(k_max)
                epsilons.append(e4)
            result["K"] = K_g
            K = K_g
        else:
            for ci in range(n_shear):
                Pxx, F_s, _, _ = csd_odas(sh_seg[:, ci], None, fft_length, fs_fast)
                K_s = F_s / W
                correction = np.ones_like(K_s)
                mask_c = K_s <= 150
                correction[mask_c] = 1 + (K_s[mask_c] / 48) ** 2
                spec_k = Pxx * W * correction

                e4, k_max, _, meth, nas, _, _, _ = _estimate_epsilon(
                    K_s,
                    spec_k,
                    nu,
                    f_AA_eff / W,
                    3,
                    1.5e-5,
                    num_ffts=num_ffts,
                    n_v=0,
                )
                result["shear_specs"].append(spec_k)
                result["nasmyth_specs"].append(nas)
                result["epsilons"].append(e4)
                result["methods"].append(meth)
                result["K_maxes"].append(k_max)
                epsilons.append(e4)
            result["K"] = K_s if n_shear > 0 else K

    # Chi spectra — match quick_look approach:
    #   observed gets sinc² correction,
    #   model spectra include FP07 transfer function H2
    eps_valid = [e for e in epsilons if np.isfinite(e) and e > 0]
    eps_mean = np.nanmean(eps_valid) if eps_valid else np.nan

    K_chi = F / W
    if n_therm > 0:
        noise_K, _ = gradT_noise(F, mean_T, W, fs=fs_fast, diff_gain=diff_gains[0])
        result["noise_K"] = noise_K

        # Sinc² correction for spectral leakage
        sinc_corr = np.ones(n_freq)
        with np.errstate(divide="ignore", invalid="ignore"):
            sinc_corr[1:] = (np.pi * F[1:] / (fs_fast * np.sin(np.pi * F[1:] / fs_fast))) ** 2
        sinc_corr = np.where(np.isfinite(sinc_corr), sinc_corr, 1.0)

        # Bilinear correction: compensate analog-vs-digital deconvolution
        # filter mismatch (matches chi.py _compute_profile_chi)
        bl_corrections = [
            _bilinear_correction(F, dg, fs_fast)
            for dg in diff_gains
        ]

        # FP07 transfer function for model spectra
        tau0 = fp07_tau(W)
        H2 = fp07_transfer(F, tau0)

        for ci in range(n_therm):
            seg = therm_data[ci][1][sel]
            if len(seg) < 2 * fft_length:
                result["chi_obs_specs"].append(np.full(n_freq, np.nan))
                result["chi_batch_specs"].append(np.full(n_freq, np.nan))
                result["chi_kraich_specs"].append(np.full(n_freq, np.nan))
                result["chi_batch_vals"].append(np.nan)
                result["chi_kraich_vals"].append(np.nan)
                continue

            Pxx_t, _ = csd_odas(seg, None, fft_length, fs_fast, overlap=fft_length // 2)[:2]
            bl_ci = bl_corrections[ci] if ci < len(bl_corrections) else np.ones(n_freq)
            spec_obs = Pxx_t * W * sinc_corr * bl_ci
            result["chi_obs_specs"].append(spec_obs)

            dg = diff_gains[ci] if ci < len(diff_gains) else 0.94

            if np.isfinite(eps_mean) and eps_mean > 0:
                chi_b, kB_b, _, batch_spec_raw, _, _ = _chi_from_epsilon(
                    spec_obs,
                    K_chi,
                    eps_mean,
                    nu,
                    mean_T,
                    W,
                    f_AA_eff,
                    "single_pole",
                    "batchelor",
                    fs_fast,
                    dg,
                    fft_length,
                )
                chi_k, kB_k, _, kraich_spec_raw, _, _ = _chi_from_epsilon(
                    spec_obs,
                    K_chi,
                    eps_mean,
                    nu,
                    mean_T,
                    W,
                    f_AA_eff,
                    "single_pole",
                    "kraichnan",
                    fs_fast,
                    dg,
                    fft_length,
                )
                # Apply FP07 rolloff to model spectra so they match observed
                batch_spec = batch_spec_raw * H2
                kraich_spec = kraich_spec_raw * H2
            else:
                chi_b = chi_k = np.nan
                batch_spec = kraich_spec = np.zeros(n_freq)

            result["chi_batch_specs"].append(batch_spec)
            result["chi_kraich_specs"].append(kraich_spec)
            result["chi_batch_vals"].append(chi_b)
            result["chi_kraich_vals"].append(chi_k)

    return result


# ---------------------------------------------------------------------------
# DissLookViewer
# ---------------------------------------------------------------------------


class DissLookViewer:
    """Interactive dissipation quality viewer with Prev/Next navigation."""

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
    ):
        self.pf = pf
        self.fft_length = fft_length
        self.f_AA = f_AA
        self.goodman = goodman
        self.spec_P_range = spec_P_range

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

        self.P = pf.channels["P"]
        self.T = pf.channels.get("T1", pf.channels.get("T", np.zeros_like(self.P)))
        self.JAC_T = pf.channels.get("JAC_T", None)
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

        # Interpolate slow -> fast
        self.P_fast = np.interp(self.t_fast, self.t_slow, self.P)
        self.speed_fast = np.abs(np.interp(self.t_fast, self.t_slow, W))
        self.ratio = round(self.fs_fast / self.fs_slow)

        self.profile_idx = 0
        self.fig = None
        self._twin_ax = None
        self._spec_bin_width = None
        self._cached_diss = None

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
        diss_length = 2 * self.fft_length
        return diss_length / self.fs_fast * mean_speed

    # ------------------------------------------------------------------
    # Main draw
    # ------------------------------------------------------------------

    def _draw(self):
        s_slow, e_slow = self.profiles[self.profile_idx]
        sel_fast = self._slow_to_fast_slice(s_slow, e_slow)
        self._spec_bin_width = self._spec_bin_dbar(s_slow, e_slow)
        sel_spec = self._spec_fast_slice(sel_fast)

        if self._twin_ax is not None:
            self._twin_ax.remove()
            self._twin_ax = None

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            for ax in self.axes.flat:
                ax.clear()

        self.axes[0, 0].invert_yaxis()

        # Compute windowed dissipation for the full profile
        self._cached_diss = _compute_windowed_diss(
            self.shear,
            self.accel,
            self.therm_fast,
            self.diff_gains,
            self.P_fast,
            self.T,
            self.speed_fast,
            self.fs_fast,
            sel_fast,
            self.fft_length,
            self.f_AA,
            self.goodman,
        )

        # Compute spectra at depth (shared by chi and eps spectra panels)
        self._cached_spec = _compute_depth_spectra(
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
        )

        # Row 0: profile panels
        self._draw_overview(s_slow, e_slow)
        self._draw_eps_profile()
        self._draw_chi_profile()
        self._draw_fm_profile()

        # Row 1: spectra and comparison panels
        self._draw_eps_spectra(sel_spec)
        self._draw_chi_spectra(sel_spec)
        self._draw_chi_comparison()
        self._draw_temperature(s_slow, e_slow)

        # Green band for spectral depth range
        if self.spec_P_range is not None:
            sp_lo, sp_hi = self.spec_P_range
            for ax in [self.axes[0, c] for c in range(4)] + [self.axes[1, 2]]:
                ax.axhspan(sp_lo, sp_hi, color="green", alpha=0.15, zorder=0)

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
    # Row 0: Pressure y-axis panels (same as quick_look)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Row 0: Profile panels (pressure y-axis, all linked)
    # ------------------------------------------------------------------

    def _draw_overview(self, s_slow, e_slow):
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

    def _draw_chi_spectra(self, sel_spec):
        """Panel (1,1): Chi spectra at the stepper depth, both models."""
        ax = self.axes[1, 1]

        n_fast = sel_spec.stop - sel_spec.start
        if n_fast < 2 * self.fft_length:
            ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes, ha="center", va="center")
            return

        r = self._cached_spec
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

            # Batchelor M1 fit — dashed
            b_spec = r["chi_batch_specs"][i]
            chi_b = r["chi_batch_vals"][i]
            valid_b = np.isfinite(b_spec) & (b_spec > 0) & (K_chi > 0)
            if np.any(valid_b) and np.isfinite(chi_b):
                ax.loglog(
                    K_chi[valid_b],
                    b_spec[valid_b],
                    c,
                    linewidth=1.2,
                    linestyle="--",
                    alpha=0.9,
                    label=f"M1 Batch χ={chi_b:.1e}",
                )

            # Kraichnan M1 fit — dash-dot
            k_spec = r["chi_kraich_specs"][i]
            chi_k = r["chi_kraich_vals"][i]
            valid_k = np.isfinite(k_spec) & (k_spec > 0) & (K_chi > 0)
            if np.any(valid_k) and np.isfinite(chi_k):
                ax.loglog(
                    K_chi[valid_k],
                    k_spec[valid_k],
                    c,
                    linewidth=1.2,
                    linestyle="-.",
                    alpha=0.9,
                    label=f"M1 Kraich χ={chi_k:.1e}",
                )

        # Noise floor
        noise = r["noise_K"]
        if noise is not None:
            valid_n = np.isfinite(noise) & (noise > 0) & (K_chi > 0)
            if np.any(valid_n):
                ax.loglog(
                    K_chi[valid_n], noise[valid_n], "0.5",
                    linewidth=0.6, linestyle=":", label="Noise",
                )

        # AA line
        K_AA = self.f_AA / r["W"]
        ax.axvline(K_AA, color="0.5", linestyle=":", linewidth=0.5, alpha=0.5)

        P_lo = float(self.P_fast[sel_spec.start])
        P_hi = float(self.P_fast[min(sel_spec.stop - 1, len(self.P_fast) - 1)])
        ax.set_xlabel("Wavenumber [cpm]")
        ax.set_ylabel("Φ_T [(K/m)² cpm⁻¹]")
        ax.set_xlim(0.5, 300)
        ax.set_ylim(1e-11, None)
        ax.legend(fontsize=5, loc="lower left")
        ax.set_title(
            f"χ spectra  P={P_lo:.1f}–{P_hi:.1f}\n(-- M1 Batch, -· M1 Kraich)",
            fontsize=9,
        )
        ax.grid(True, alpha=0.3, which="both")

    # ------------------------------------------------------------------
    # Row 0 continued: Profile panels
    # ------------------------------------------------------------------

    def _draw_eps_profile(self):
        """Panel (0,1): Epsilon from each shear probe vs pressure."""
        ax = self.axes[0, 1]
        d = self._cached_diss
        P = d["P_windows"]

        if len(P) == 0:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
            return

        colors = ["C0", "C1", "C4", "C5"]
        has_data = False
        for i, (name, _) in enumerate(self.shear):
            c = colors[i % len(colors)]
            eps = d["epsilon"][i]
            valid = np.isfinite(eps) & (eps > 0)
            if np.any(valid):
                ax.plot(eps[valid], P[valid], f"{c}o-", markersize=3, linewidth=0.8, label=name)
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

    def _draw_chi_profile(self):
        """Panel (0,2): Chi profile vs pressure (mean of Batchelor & Kraichnan)."""
        ax = self.axes[0, 2]
        d = self._cached_diss
        P = d["P_windows"]

        if len(P) == 0:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
            return

        colors = ["C3", "C1", "C4", "C5"]
        has_data = False
        for i, (name, _) in enumerate(self.therm_fast):
            c = colors[i % len(colors)]
            chi_b = d["chi_batchelor"][i]
            chi_k = d["chi_kraichnan"][i]
            chi_mean = np.where(
                np.isfinite(chi_b) & np.isfinite(chi_k),
                (chi_b + chi_k) / 2,
                np.where(np.isfinite(chi_b), chi_b, chi_k),
            )
            valid = np.isfinite(chi_mean) & (chi_mean > 0)
            if np.any(valid):
                ax.plot(
                    chi_mean[valid],
                    P[valid],
                    f"{c}o-",
                    markersize=3,
                    linewidth=0.8,
                    label=name,
                )
                has_data = True

        if has_data:
            ax.set_xscale("log")
            ax.set_xlabel("χ [K²/s]")
            ax.set_ylabel("Pressure [dbar]")
            ax.legend(fontsize=6, loc="lower left")
            ax.set_title("χ profile (M1)", fontsize=9)
            ax.grid(True, alpha=0.3, which="both")
        else:
            ax.text(0.5, 0.5, "No valid χ", transform=ax.transAxes, ha="center", va="center")

    def _draw_chi_comparison(self):
        """Panel (1,2): Chi from Batchelor vs Kraichnan models vs pressure."""
        ax = self.axes[1, 2]
        d = self._cached_diss
        P = d["P_windows"]

        if len(P) == 0:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
            return

        colors = ["C3", "C1", "C4", "C5"]
        has_data = False
        for i, (name, _) in enumerate(self.therm_fast):
            c = colors[i % len(colors)]

            # Batchelor — dashed
            chi_b = d["chi_batchelor"][i]
            valid_b = np.isfinite(chi_b) & (chi_b > 0)
            if np.any(valid_b):
                ax.plot(
                    chi_b[valid_b],
                    P[valid_b],
                    color=c,
                    linestyle="--",
                    marker="^",
                    markersize=2.5,
                    linewidth=0.8,
                    label=f"{name} Batch.",
                )
                has_data = True

            # Kraichnan — solid
            chi_k = d["chi_kraichnan"][i]
            valid_k = np.isfinite(chi_k) & (chi_k > 0)
            if np.any(valid_k):
                ax.plot(
                    chi_k[valid_k],
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
            ax.set_xscale("log")
            ax.set_xlabel("χ [K²/s]")
            ax.set_ylabel("Pressure [dbar]")
            ax.legend(fontsize=5, loc="lower left")
            ax.set_title("χ: Batchelor vs Kraichnan (M1)", fontsize=9)
            ax.grid(True, alpha=0.3, which="both")
        else:
            ax.text(0.5, 0.5, "No valid χ", transform=ax.transAxes, ha="center", va="center")

    def _draw_fm_profile(self):
        """Panel (0,3): Lueck FM figure of merit vs pressure."""
        ax = self.axes[0, 3]
        d = self._cached_diss
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
            # Shade good region
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

    def _draw_eps_spectra(self, sel_spec):
        """Panel (1,0): Shear (epsilon) spectra at the stepper depth."""
        ax = self.axes[1, 0]

        n_fast = sel_spec.stop - sel_spec.start
        if n_fast < 2 * self.fft_length:
            ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes, ha="center", va="center")
            return

        r = self._cached_spec
        K = r["K"]
        if len(K) == 0:
            ax.text(0.5, 0.5, "No spectra", transform=ax.transAxes, ha="center", va="center")
            return

        # Shear spectra + Nasmyth
        colors_sh = ["C0", "C1"]
        for i, (name, _) in enumerate(self.shear):
            if i >= len(r["shear_specs"]):
                break
            c = colors_sh[i % len(colors_sh)]
            spec = r["shear_specs"][i]
            ax.loglog(K, spec, c, linewidth=0.8, label=name)
            nas = r["nasmyth_specs"][i]
            eps_i = r["epsilons"][i]
            meth_i = r["methods"][i] if i < len(r["methods"]) else 0
            meth_tag = "ISR" if meth_i == 1 else "var"
            if np.isfinite(eps_i):
                ax.loglog(
                    K,
                    nas,
                    c,
                    linewidth=0.7,
                    linestyle="--",
                    alpha=0.8,
                    label=f"Nas ε={eps_i:.1e} ({meth_tag})",
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

        P_lo = float(self.P_fast[sel_spec.start])
        P_hi = float(self.P_fast[min(sel_spec.stop - 1, len(self.P_fast) - 1)])
        ax.set_xlabel("Wavenumber [cpm]")
        ax.set_ylabel("Φ_sh [s⁻² cpm⁻¹]")
        ax.set_xlim(0.5, 300)
        ax.set_ylim(1e-9, 1e0)
        ax.legend(fontsize=5, loc="lower left")
        ax.set_title(f"ε spectra  P={P_lo:.1f}–{P_hi:.1f}", fontsize=9)
        ax.grid(True, alpha=0.3, which="both")

    def _draw_temperature(self, s_slow, e_slow):
        """Panel (1,3): Temperature channels (T1_dT1, T2_dT2, JAC_T) vs pressure."""
        ax = self.axes[1, 3]
        sel_fast = self._slow_to_fast_slice(s_slow, e_slow)
        P_prof_fast = self.P_fast[sel_fast]
        P_prof_slow = self.P[s_slow : e_slow + 1]

        colors = ["C3", "C1", "C4", "C5"]
        has_data = False

        for i, (name, data) in enumerate(self.therm_fast):
            c = colors[i % len(colors)]
            ax.plot(data[sel_fast], P_prof_fast, c, linewidth=0.5, alpha=0.7, label=name)
            has_data = True

        if self.JAC_T is not None:
            ax.plot(
                self.JAC_T[s_slow : e_slow + 1],
                P_prof_slow,
                "C2",
                linewidth=1.0,
                label="JAC_T",
            )
            has_data = True

        if has_data:
            ax.set_xlabel("Temperature gradient / T [°C]")
            ax.set_ylabel("Pressure [dbar]")
            ax.invert_yaxis()
            ax.legend(fontsize=6, loc="lower left")
            ax.set_title("Temperature", fontsize=9)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(
                0.5, 0.5, "No temperature channels",
                transform=ax.transAxes, ha="center", va="center",
            )

    # ------------------------------------------------------------------
    # Figure setup and navigation
    # ------------------------------------------------------------------

    def show(self):
        self.fig, self.axes = plt.subplots(
            2,
            4,
            figsize=(24, 9),
            gridspec_kw={"hspace": 0.35, "wspace": 0.32},
        )
        self.fig.subplots_adjust(bottom=0.08, top=0.92, left=0.04, right=0.98)

        # Link pressure y-axes: row 0 all cols (profile panels) + row 1 col 2 (chi comparison)
        p_ref = self.axes[0, 0]
        for col in range(1, 4):
            self.axes[0, col].sharey(p_ref)
        self.axes[1, 2].sharey(p_ref)
        p_ref.invert_yaxis()

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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def diss_look(
    source,
    fft_length=256,
    f_AA=98.0,
    goodman=True,
    direction="down",
    P_min=0.5,
    W_min=0.3,
    min_duration=7.0,
    spec_P_range=None,
):
    """Open an interactive dissipation quality viewer for a .p file.

    Displays epsilon, chi (Batchelor vs Kraichnan), and the Lueck (2022)
    figure of merit (FM) for spectral quality assessment.

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
        Pressure range (P_min, P_max) in dbar for spectral panel.

    Returns
    -------
    DissLookViewer
        The viewer instance.
    """
    pf = PFile(source)
    viewer = DissLookViewer(
        pf,
        fft_length=fft_length,
        f_AA=f_AA,
        goodman=goodman,
        direction=direction,
        P_min=P_min,
        W_min=W_min,
        min_duration=min_duration,
        spec_P_range=spec_P_range,
    )
    viewer.show()
    return viewer

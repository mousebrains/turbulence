# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Interactive quick-look viewer for Rockland Scientific profiler data.

Port of quick_look.m from the ODAS MATLAB library.
Provides a multi-panel figure with Prev/Next buttons to step through profiles.
"""

import re

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button
from scipy.signal import butter, filtfilt

from rsi_python.despike import despike
from rsi_python.nasmyth import nasmyth
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
    """Compute shear spectra and Nasmyth references for one profile segment."""
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
    # Convert frequency spectra to wavenumber spectra
    spectra_k = [s * mean_speed for s in spectra]

    # Fit Nasmyth to each probe
    nasmyth_specs = []
    epsilons = []
    for spec_k in spectra_k:
        # Simple variance-based epsilon estimate
        K_AA = f_AA / mean_speed
        valid = (K > 0) & (K <= K_AA)
        if np.sum(valid) < 3:
            epsilons.append(np.nan)
            nasmyth_specs.append(np.full_like(K, np.nan))
            continue
        dk = np.gradient(K)
        variance = np.sum(spec_k[valid] * dk[valid])
        # Iterate to find epsilon from Nasmyth
        eps_est = 7.5 * nu * variance
        for _ in range(5):
            nas = nasmyth(eps_est, nu, K[valid])
            nas_var = np.sum(nas * dk[valid])
            if nas_var > 0:
                eps_est *= variance / nas_var
        epsilons.append(eps_est)
        nasmyth_specs.append(nasmyth(eps_est, nu, K))

    return K, F, spectra_k, nasmyth_specs, epsilons, mean_speed, nu


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
    ):
        self.pf = pf
        self.fft_length = fft_length
        self.f_AA = f_AA
        self.goodman = goodman

        # Extract channel data
        sh_re = re.compile(r"^sh\d+$")
        ac_re = re.compile(r"^A[xyz]$")
        T_re = re.compile(r"^T\d+$")

        self.shear = sorted(
            [(n, pf.channels[n]) for n in pf._fast_channels if sh_re.match(n)],
            key=lambda x: x[0],
        )
        self.accel = sorted(
            [(n, pf.channels[n]) for n in pf._fast_channels if ac_re.match(n)],
            key=lambda x: x[0],
        )
        # Slow temperature channels for profile display
        self.therm_slow = sorted(
            [(n, pf.channels[n]) for n in pf._slow_channels if T_re.match(n)],
            key=lambda x: x[0],
        )

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

    def _slow_to_fast_slice(self, s_slow, e_slow):
        """Convert slow-channel indices to fast-channel slice."""
        s_fast = s_slow * self.ratio
        e_fast = min((e_slow + 1) * self.ratio, len(self.t_fast))
        return slice(s_fast, e_fast)

    def _draw(self):
        """Draw all panels for the current profile."""
        s_slow, e_slow = self.profiles[self.profile_idx]
        sel_fast = self._slow_to_fast_slice(s_slow, e_slow)

        for ax in self.axes.flat:
            ax.clear()

        self._draw_overview(s_slow, e_slow)
        self._draw_shear(sel_fast)
        self._draw_temperature(s_slow, e_slow, sel_fast)
        self._draw_spectra(sel_fast)

        title = (
            f"{self.pf.filepath.name}  —  "
            f"Profile {self.profile_idx + 1}/{len(self.profiles)}  —  "
            f"P: {self.P[s_slow]:.1f}–{self.P[e_slow]:.1f} dbar"
        )
        self.fig.suptitle(title, fontsize=11, fontweight="bold")
        self.fig.canvas.draw_idle()

    def _draw_overview(self, s_slow, e_slow):
        """Panel 1: Pressure vs time, current profile highlighted."""
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
        ax.invert_yaxis()
        ax.set_title("Profile overview", fontsize=9)
        ax.grid(True, alpha=0.3)

        # Mark all profiles lightly
        for i, (s, e) in enumerate(self.profiles):
            if i != self.profile_idx:
                ax.axvspan(
                    self.t_slow[s],
                    self.t_slow[e],
                    alpha=0.05,
                    color="C0",
                )

    def _draw_shear(self, sel_fast):
        """Panel 2: HP-filtered shear signals vs pressure."""
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

            # Despike for display
            seg_hp, _, _, _ = despike(seg_hp, self.fs_fast, thresh=8, smooth=0.5)

            offset = i * 0.5
            ax.plot(seg_hp + offset, P_prof, linewidth=0.3, label=name)

        ax.set_ylabel("Pressure [dbar]")
        ax.set_xlabel("Shear [s⁻¹] (HP, offset)")
        ax.invert_yaxis()
        ax.legend(fontsize=7, loc="lower right")
        ax.set_title("Shear probes", fontsize=9)
        ax.grid(True, alpha=0.3)

    def _draw_temperature(self, s_slow, e_slow, sel_fast):
        """Panel 3: Temperature vs pressure."""
        ax = self.axes[1, 0]
        P_slow_prof = self.P[s_slow : e_slow + 1]

        # Plot all slow temperature channels
        colors = ["C3", "C1", "C4", "C5"]
        for i, (name, data) in enumerate(self.therm_slow):
            c = colors[i % len(colors)]
            ax.plot(data[s_slow : e_slow + 1], P_slow_prof, c, linewidth=1.0, label=name)

        # Add fall rate on twin axis
        ax2 = ax.twiny()
        W_prof = self.W[s_slow : e_slow + 1]
        ax2.plot(W_prof, P_slow_prof, "C2--", linewidth=0.8, alpha=0.6, label="dP/dt")
        ax2.set_xlabel("Fall rate [dbar/s]", fontsize=8, color="C2")
        ax2.tick_params(axis="x", labelsize=7, colors="C2")

        ax.set_ylabel("Pressure [dbar]")
        ax.set_xlabel("Temperature [°C]")
        ax.invert_yaxis()
        ax.legend(fontsize=7, loc="lower left")
        ax.set_title("Temperature & fall rate", fontsize=9)
        ax.grid(True, alpha=0.3)

    def _draw_spectra(self, sel_fast):
        """Panel 4: Wavenumber spectra with Nasmyth references."""
        ax = self.axes[1, 1]

        if not self.shear or (sel_fast.stop - sel_fast.start) < 2 * self.fft_length:
            ax.text(
                0.5,
                0.5,
                "Insufficient data for spectra",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
            return

        K, F, spectra_k, nasmyth_specs, epsilons, mean_speed, nu = _compute_profile_spectra(
            self.shear,
            self.accel,
            self.P_fast,
            self.T,
            self.speed_fast,
            self.fs_fast,
            sel_fast,
            self.fft_length,
            self.f_AA,
            self.goodman,
        )

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

        # Reference Nasmyth curves
        for exp in [-9, -8, -7, -6, -5]:
            eps_ref = 10.0**exp
            nas_ref = nasmyth(eps_ref, nu, K)
            ax.loglog(K, nas_ref, "0.8", linewidth=0.3)
            # Label at right edge
            y_val = nas_ref[len(K) // 2]
            if np.isfinite(y_val) and y_val > 0:
                ax.text(K[len(K) // 2], y_val, f"{exp}", fontsize=6, color="0.5", va="bottom")

        # Mark f_AA
        K_AA = self.f_AA / mean_speed
        ax.axvline(K_AA, color="0.5", linestyle=":", linewidth=0.5, alpha=0.5)

        ax.set_xlabel("Wavenumber [cpm]")
        ax.set_ylabel("Φ(k) [s⁻² cpm⁻¹]")
        ax.set_xlim(0.5, 300)
        ax.set_ylim(1e-9, 1e0)
        ax.legend(fontsize=6, loc="upper right", ncol=2)
        ax.set_title(f"Shear spectra (speed={mean_speed:.2f} m/s)", fontsize=9)
        ax.grid(True, alpha=0.3, which="both")

    def show(self):
        """Create the figure and display it."""
        self.fig, self.axes = plt.subplots(2, 2, figsize=(14, 10))
        self.fig.subplots_adjust(bottom=0.08, top=0.92, hspace=0.35, wspace=0.30)

        # Navigation buttons
        ax_prev = self.fig.add_axes([0.35, 0.01, 0.1, 0.04])
        ax_next = self.fig.add_axes([0.55, 0.01, 0.1, 0.04])
        self.btn_prev = Button(ax_prev, "◀ Prev")
        self.btn_next = Button(ax_next, "Next ▶")
        self.btn_prev.on_clicked(self._on_prev)
        self.btn_next.on_clicked(self._on_next)

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


def quick_look(
    source,
    fft_length=256,
    f_AA=98.0,
    goodman=True,
    direction="down",
    P_min=0.5,
    W_min=0.3,
    min_duration=7.0,
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
    )
    viewer.show()
    return viewer

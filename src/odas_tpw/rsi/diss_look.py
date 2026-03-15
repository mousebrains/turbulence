# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Dissipation quality viewer for Rockland Scientific profiler data.

Interactive multi-panel viewer focused on comparing epsilon, chi from
Method 1 (from epsilon) vs Method 2 (iterative fit), and the Lueck (2022) figure of merit (FM).

Layout (2 rows x 4 columns):
  Row 0 (profile panels, pressure y-axis linked):
    Overview | epsilon profile | chi profile | Mixing efficiency
  Row 1 (spectra at depth):
    FM profile | epsilon spectra | chi spectra | chi FOM
"""

import warnings

import numpy as np

from odas_tpw.rsi.p_file import PFile
from odas_tpw.rsi.viewer_base import (
    ProfileViewer,
    compute_depth_spectra,
    compute_windowed_diss,
)


class DissLookViewer(ProfileViewer):
    """Interactive dissipation quality viewer with Prev/Next navigation."""

    def __init__(self, pf, **kwargs):
        super().__init__(pf, **kwargs)
        self._twin_ax = None
        self._cached_diss: dict | None = None
        self._cached_spec: dict | None = None

    def _setup_axes(self):
        p_ref = self.axes[0, 0]
        for col in range(1, 4):
            self.axes[0, col].sharey(p_ref)
        self.axes[1, 0].sharey(p_ref)
        self.axes[1, 3].sharey(p_ref)
        p_ref.invert_yaxis()

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

        # Pre-compute windowed dissipation for the full profile
        self._cached_diss = compute_windowed_diss(
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
            diss_length=self.diss_length,
        )

        # Pre-compute spectra at depth (shared by eps and chi spectra panels)
        self._cached_spec = compute_depth_spectra(
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
            diss_length=self.diss_length,
        )

        d = self._cached_diss
        assert d is not None

        # Row 0: profile panels
        self._draw_overview(self.axes[0, 0], s_slow, e_slow)
        self._draw_eps_profile(self.axes[0, 1], d["P_windows"], d["epsilon"])
        self._add_fm_filtered_eps(self.axes[0, 1], d)
        self._draw_chi_profile()
        self._draw_mixing_efficiency()

        # Row 1: spectra and comparison panels
        self._draw_fm_profile()
        self._draw_eps_spectra(self.axes[1, 1], sel_spec, self._cached_spec)
        self._draw_chi_spectra(sel_spec)
        self._draw_chi_fom_profile()

        # Finish (green band + title)
        pressure_axes = [self.axes[0, c] for c in range(4)] + [self.axes[1, 0], self.axes[1, 3]]
        self._finish_draw(s_slow, e_slow, pressure_axes)

    # ------------------------------------------------------------------
    # FM-filtered epsilon helper
    # ------------------------------------------------------------------

    def _fm_best_epsilon(self):
        """Select best epsilon per window using Lueck FM < 1.15 criterion.

        If both probes pass, average.  If one passes, use it.
        If neither passes, NaN.
        """
        d = self._cached_diss
        assert d is not None
        eps = d["epsilon"]  # (n_shear, n_windows)
        FM = d["FM"]  # (n_shear, n_windows)
        n_shear, n_win = eps.shape
        if n_shear == 0 or n_win == 0:
            return np.array([])
        best = np.full(n_win, np.nan)
        for j in range(n_win):
            good = []
            for i in range(n_shear):
                if (
                    np.isfinite(FM[i, j])
                    and FM[i, j] < 1.15
                    and np.isfinite(eps[i, j])
                    and eps[i, j] > 0
                ):
                    good.append(eps[i, j])
            if good:
                best[j] = np.mean(good)
        return best

    def _add_fm_filtered_eps(self, ax, d):
        """Overlay FM-filtered best epsilon as dashed black line."""
        P = d["P_windows"]
        eps_best = self._fm_best_epsilon()
        if len(eps_best) == 0:
            return
        valid = np.isfinite(eps_best) & (eps_best > 0)
        if np.any(valid):
            ax.plot(eps_best[valid], P[valid], "k--", linewidth=1.2, label="\u03b5 FM<1.15")
            ax.legend(fontsize=6, loc="lower left")

    # ------------------------------------------------------------------
    # Unique panels
    # ------------------------------------------------------------------

    def _draw_chi_profile(self):
        """Panel (0,2): Chi profile vs pressure (mean of Batchelor & Kraichnan)."""
        ax = self.axes[0, 2]
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
                    chi_mean[valid], P[valid], f"{c}o-", markersize=3, linewidth=0.8, label=name
                )
                has_data = True

        if has_data:
            ax.set_xscale("log")
            ax.set_xlabel("\u03c7 [K\u00b2/s]")
            ax.set_ylabel("Pressure [dbar]")
            ax.legend(fontsize=6, loc="lower left")
            ax.set_title("\u03c7 profile M1 (Batch. + Kraich.)", fontsize=9)
            ax.grid(True, alpha=0.3, which="both")
        else:
            ax.text(0.5, 0.5, "No valid \u03c7", transform=ax.transAxes, ha="center", va="center")

    def _draw_chi_fom_profile(self):
        """Panel (1,3): Chi figure of merit vs pressure (Batchelor & Kraichnan)."""
        ax = self.axes[1, 3]
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

            # Batchelor fom -- dashed
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

            # Kraichnan fom -- solid
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
            ax.set_xlabel("\u03c7 FOM (obs/model)")
            ax.set_ylabel("Pressure [dbar]")
            ax.legend(fontsize=5, loc="lower right")
            ax.set_title("\u03c7 FOM (Batch. & Kraich.)", fontsize=9)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(
                0.5, 0.5, "No valid \u03c7 FOM", transform=ax.transAxes, ha="center", va="center"
            )

    def _draw_fm_profile(self):
        """Panel (1,0): Lueck FM figure of merit vs pressure."""
        ax = self.axes[1, 0]
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

    def _draw_chi_spectra(self, sel_spec):
        """Panel (1,2): Chi spectra -- Method 1 (from epsilon) vs Method 2 (iter)."""
        ax = self.axes[1, 2]

        n_fast = sel_spec.stop - sel_spec.start
        if n_fast < self.diss_length:
            ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes, ha="center", va="center")
            return

        r = self._cached_spec
        assert r is not None
        K_chi = r["F"] / r["W"]

        if not r["chi_obs_specs"]:
            ax.text(0.5, 0.5, "No \u03c7 spectra", transform=ax.transAxes, ha="center", va="center")
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

            # Method 1 fit (kB from epsilon) -- dashed
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
                    label=f"M1 \u03c7={chi_m1:.1e}",
                )

            # Method 2 iterative fit (kB from spectral fit) -- dash-dot
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
                    label=f"M2 \u03c7={chi_m2:.1e}",
                )

        # Noise floor
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

        # AA line
        K_AA = self.f_AA / r["W"]
        ax.axvline(K_AA, color="0.5", linestyle=":", linewidth=0.5, alpha=0.5)

        P_lo = float(self.P_fast[sel_spec.start])
        P_hi = float(self.P_fast[min(sel_spec.stop - 1, len(self.P_fast) - 1)])
        # Dynamic y-axis: 1 order above max, 1 order below noise floor
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
        # Lower limit: 1 order below noise floor minimum
        if noise is not None:
            noise_in = noise[in_range & (noise > 0) & np.isfinite(noise)]
            y_lo = 10 ** (np.floor(np.log10(np.min(noise_in))) - 1) if len(noise_in) else 1e-11
        else:
            y_lo = 1e-11

        ax.set_xlabel("Wavenumber [cpm]")
        ax.set_ylabel("\u03a6_T [(K/m)\u00b2 cpm\u207b\u00b9]")
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.legend(fontsize=5, loc="lower left")
        ax.set_title(
            f"\u03c7 spectra (Kraichnan)  P={P_lo:.1f}\u2013{P_hi:.1f}\n"
            f"(-- M1 from \u03b5, -\u00b7 M2 iter)",
            fontsize=9,
        )
        ax.grid(True, alpha=0.3, which="both")

    def _draw_mixing_efficiency(self):
        """Panel (0,3): Mixing efficiency Lambda = chi / epsilon vs pressure."""
        ax = self.axes[0, 3]
        d = self._cached_diss
        assert d is not None
        P = d["P_windows"]

        if len(P) == 0:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
            return

        # Mean epsilon across shear probes
        eps_all = d["epsilon"]  # (n_shear, n_windows)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            eps_mean = np.nanmean(eps_all, axis=0)

        # FM-filtered best epsilon
        eps_fm = self._fm_best_epsilon()

        # Helper to compute mean chi per thermistor
        def _mean_chi(i):
            chi_b = d["chi_batchelor"][i]
            chi_k = d["chi_kraichnan"][i]
            return np.where(
                np.isfinite(chi_b) & np.isfinite(chi_k),
                (chi_b + chi_k) / 2,
                np.where(np.isfinite(chi_b), chi_b, chi_k),
            )

        # Per-thermistor lines (using mean epsilon)
        colors = ["C3", "C1", "C4", "C5"]
        has_data = False
        for i, (name, _) in enumerate(self.therm_fast):
            c = colors[i % len(colors)]
            chi_mean = _mean_chi(i)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                lam = chi_mean / eps_mean
            valid = np.isfinite(lam) & (lam > 0)
            if np.any(valid):
                ax.plot(lam[valid], P[valid], f"{c}o-", markersize=3, linewidth=0.8, label=name)
                has_data = True

        # FM-filtered line (mean chi across thermistors / FM-filtered epsilon)
        if len(eps_fm) > 0 and len(self.therm_fast) > 0:
            chi_all = np.array([_mean_chi(i) for i in range(len(self.therm_fast))])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                chi_avg = np.nanmean(chi_all, axis=0)
                lam_fm = chi_avg / eps_fm
            valid_fm = np.isfinite(lam_fm) & (lam_fm > 0)
            if np.any(valid_fm):
                ax.plot(lam_fm[valid_fm], P[valid_fm], "k--", linewidth=1.2, label="\u039b FM<1.15")
                has_data = True

        if has_data:
            ax.set_xscale("log")
            ax.axvline(
                0.2, color="k", linestyle=":", linewidth=1.0, alpha=0.7, label="\u039b = 0.2"
            )
            xlim = ax.get_xlim()
            ax.axvspan(xlim[0], 0.2, color="blue", alpha=0.04, zorder=0)
            ax.axvspan(0.2, xlim[1], color="red", alpha=0.04, zorder=0)
            ax.set_xlabel("\u039b = \u03c7 / \u03b5  [K\u00b2 s\u00b2 m\u207b\u00b2]")
            ax.set_ylabel("Pressure [dbar]")
            ax.legend(fontsize=6, loc="lower left")
            ax.set_title("Mixing efficiency \u039b", fontsize=9)
            ax.grid(True, alpha=0.3, which="both")
        else:
            ax.text(0.5, 0.5, "No valid \u039b", transform=ax.transAxes, ha="center", va="center")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def diss_look(
    source,
    fft_length=1024,
    f_AA=98.0,
    goodman=True,
    direction="down",
    P_min=0.5,
    W_min=0.3,
    min_duration=7.0,
    spec_P_range=None,
    diss_length=None,
) -> DissLookViewer:
    """Open an interactive dissipation quality viewer for a .p file.

    Displays epsilon, chi (Batchelor vs Kraichnan), and the Lueck (2022)
    figure of merit (FM) for spectral quality assessment.

    Parameters
    ----------
    source : str or Path
        Path to a Rockland .p file.
    fft_length : int
        FFT segment length for spectral estimation (default: 1024).
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
    diss_length : int or None
        Dissipation window length [samples]. Default: 4 * fft_length.

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
        diss_length=diss_length,
    )
    viewer.show()
    return viewer

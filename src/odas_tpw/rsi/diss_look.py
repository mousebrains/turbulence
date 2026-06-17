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
        # self._cached_diss / self._cached_spec are initialized by the base.

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
        self._draw_fm_profile(self.axes[1, 0])
        self._draw_eps_spectra(self.axes[1, 1], sel_spec, self._cached_spec)
        self._draw_chi_m1m2_spectra(self.axes[1, 2], sel_spec)
        self._draw_chi_fom_profile(self.axes[1, 3])

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

    def _draw_mixing_efficiency(self):
        """Panel (0,3): Mixing efficiency Gamma = chi / epsilon vs pressure."""
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
                ax.plot(
                    lam_fm[valid_fm], P[valid_fm], "k--", linewidth=1.2,
                    label="\u03c7/\u03b5 FM<1.15",
                )
                has_data = True

        # NB: this is the dimensional ratio chi/eps [K^2 s^2 m^-2], NOT the
        # dimensionless Osborn mixing coefficient Gamma (~0.2). The proper
        # Gamma = N^2*chi/(2*eps*(dT/dz)^2) is in the chi product's mixing vars.
        if has_data:
            ax.set_xscale("log")
            ax.set_xlabel("\u03c7 / \u03b5  [K\u00b2 s\u00b2 m\u207b\u00b2]")
            ax.set_ylabel("Pressure [dbar]")
            ax.legend(fontsize=6, loc="lower left")
            ax.set_title("\u03c7 / \u03b5 ratio (dimensional, not Osborn \u0393)", fontsize=9)
            ax.grid(True, alpha=0.3, which="both")
        else:
            ax.text(
                0.5, 0.5, "No valid \u03c7/\u03b5",
                transform=ax.transAxes, ha="center", va="center",
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def diss_look(
    source,
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
        Profile direction: 'auto', 'up', 'down', 'glide', or 'horizontal'.
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
    vehicle : str or None
        Vehicle type override (e.g. 'slocum_glider'). If None, read from
        instrument config.

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
        vehicle=vehicle,
    )
    viewer.show()
    return viewer

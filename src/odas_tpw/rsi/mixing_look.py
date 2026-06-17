# Jun-2026, Claude and Pat Welch, pat@mousebrains.com
"""Mixing viewer for Rockland Scientific profiler data.

Interactive multi-panel viewer focused on the diapycnal-mixing quantities the
pipeline derives from epsilon, chi, and the background stratification: the
buoyancy frequency N², the background temperature gradient dT/dz, the
Osborn-Cox heat diffusivity K_T, the measured Osborn mixing coefficient
Γ = N²χ / (2ε(dT/dz)²), and the Osborn diapycnal diffusivity K_rho.

Where ``diss_look`` shows only the dimensional χ/ε ratio, this viewer
computes the *proper* Γ by Thorpe-sorting each window to a stable profile
(removing overturns) for the background N² and dT/dz, exactly as the
``perturb``/``rsi`` pipelines do in their mixing step.

Layout (2 rows x 4 columns):
  Row 0 (profile panels, pressure y-axis linked):
    Overview | N² profile | K_T & K_rho profiles | Γ profile
  Row 1:
    dT/dz profile | ε profile | χ profile | K_T vs K_rho scatter
"""

import warnings

import numpy as np

from odas_tpw.processing.mixing import mixing_coefficients, sorted_stratification
from odas_tpw.rsi.p_file import PFile
from odas_tpw.rsi.viewer_base import (
    ProfileViewer,
    compute_depth_spectra,
    compute_windowed_diss,
)

# Color cycle: shear-probe (epsilon) and thermistor (chi) families, matching
# the conventions used by quick_look / diss_look.
_SHEAR_COLORS = ["C0", "C1", "C4", "C5"]
_THERM_COLORS = ["C3", "C1", "C4", "C5"]


class MixingLookViewer(ProfileViewer):
    """Interactive mixing viewer with Prev/Next navigation."""

    _nrows = 3  # extra row for the epsilon/chi spectra and FM/FOM panels

    def __init__(self, pf, salinity=None, **kwargs):
        super().__init__(pf, **kwargs)
        self._P_win = np.array([])
        self._strat = None
        self._mix = None
        self._eps_mix = np.array([])
        self._chi_mix = np.array([])

        # Slow-rate practical salinity for the background stratification:
        # an explicit override wins; otherwise the profile's own JAC C/T (when
        # both are present and aligned with the slow time base); otherwise None
        # (-> 35 PSU assumed by sorted_stratification). Mirrors the pipeline's
        # measured > supplied > 35 preference.
        self.salinity: float | np.ndarray | None
        if salinity is not None:
            self.salinity = float(salinity)
        else:
            self.salinity = self._measured_salinity(pf)

        # Colorbar for the pressure-colored K_T-vs-K_rho scatter; removed and
        # rebuilt each draw so its stolen space never compounds.
        self._scatter_cbar = None

    def _measured_salinity(self, pf) -> np.ndarray | None:
        """Slow-rate practical salinity from JAC C/T, or None when unavailable."""
        jac_c = pf.channels.get("JAC_C")
        jac_t = pf.channels.get("JAC_T")
        n_slow = len(self.t_slow)
        if (
            jac_c is None
            or jac_t is None
            or len(jac_c) != n_slow
            or len(jac_t) != n_slow
            or len(self.P) != n_slow
        ):
            return None
        import gsw

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            sp = gsw.SP_from_C(np.asarray(jac_c), np.asarray(jac_t), np.asarray(self.P))
        sp = np.asarray(sp, dtype=np.float64)
        return sp if np.isfinite(sp).any() else None

    # ------------------------------------------------------------------
    # Axis layout
    # ------------------------------------------------------------------

    def _setup_axes(self):
        p_ref = self.axes[0, 0]
        # Pressure-profile panels share the y-axis. Independent panels: the
        # K_T-vs-K_rho scatter (1, 3) and the two wavenumber spectra (2, 0/1).
        for col in range(1, 4):
            self.axes[0, col].sharey(p_ref)
        for col in range(3):  # (1,0) dT/dz, (1,1) eps, (1,2) chi
            self.axes[1, col].sharey(p_ref)
        self.axes[2, 2].sharey(p_ref)  # FM profile
        self.axes[2, 3].sharey(p_ref)  # chi FOM profile
        p_ref.invert_yaxis()

    # ------------------------------------------------------------------
    # Per-window combine helpers (match pipeline epsi_final / chi_final)
    # ------------------------------------------------------------------

    def _window_center_times(self, sel_fast) -> np.ndarray:
        """Window-center times aligned 1:1 with compute_windowed_diss windows.

        Mirrors the windowing in :func:`compute_windowed_diss` exactly (same
        overlap, step, count, and early-break) so each center time lines up with
        the corresponding ``P_windows`` / epsilon / chi column. The caller
        asserts the lengths match as a drift tripwire.
        """
        diss = self.diss_length
        overlap = diss // 2
        step = diss - overlap
        seg_start, seg_end = sel_fast.start, sel_fast.stop
        n_windows = max(0, 1 + (seg_end - seg_start - diss) // step)
        t_win = np.full(n_windows, np.nan)
        for idx in range(n_windows):
            s = seg_start + idx * step
            e = s + diss
            if e > seg_end:
                break
            t_win[idx] = float(np.mean(self.t_fast[s:e]))
        return t_win

    def _mixing_epsilon(self, d) -> np.ndarray:
        """Per-window epsilon for mixing: FM<1.15 geometric mean, nanmean fallback.

        Reproduces the pipeline's QC-then-combine: average the probes that pass
        the Lueck (2022) FM<1.15 quality cut (geometric mean, like
        ``_compute_epsi_final``); if no probe passes a window, fall back to the
        geometric mean of all finite positive probes so the window still yields a
        mixing estimate.
        """
        eps = d["epsilon"]  # (n_shear, n_win)
        FM = d["FM"]
        n_shear, n_win = eps.shape
        out = np.full(n_win, np.nan)
        if n_shear == 0 or n_win == 0:
            return out
        for j in range(n_win):
            col = eps[:, j]
            # Keep probes that pass the Lueck (2022) reject criterion FM <= 1.15
            # (reject FM > 1.15), matching the pipeline's fom_limit convention.
            passed = col[np.isfinite(FM[:, j]) & (FM[:, j] <= 1.15) & np.isfinite(col) & (col > 0)]
            pool = passed if passed.size else col[np.isfinite(col) & (col > 0)]
            if pool.size:
                out[j] = float(np.exp(np.mean(np.log(pool))))
        return out

    def _mixing_chi(self, d) -> np.ndarray:
        """Per-window chi for mixing: mean(Batchelor, Kraichnan) per thermistor,
        then geometric mean across thermistors (matching chi_final)."""
        chi_b = d["chi_batchelor"]  # (n_therm, n_win)
        chi_k = d["chi_kraichnan"]
        n_therm, n_win = chi_b.shape
        out = np.full(n_win, np.nan)
        if n_therm == 0 or n_win == 0:
            return out
        per = np.where(
            np.isfinite(chi_b) & np.isfinite(chi_k),
            (chi_b + chi_k) / 2.0,
            np.where(np.isfinite(chi_b), chi_b, chi_k),
        )
        for j in range(n_win):
            col = per[:, j]
            pool = col[np.isfinite(col) & (col > 0)]
            if pool.size:
                out[j] = float(np.exp(np.mean(np.log(pool))))
        return out

    # ------------------------------------------------------------------
    # Draw
    # ------------------------------------------------------------------

    def _draw(self):
        s_slow, e_slow = self.profiles[self.profile_idx]
        sel_fast = self._slow_to_fast_slice(s_slow, e_slow)
        self._spec_bin_width = self._spec_bin_dbar(s_slow, e_slow)
        sel_spec = self._spec_fast_slice(sel_fast)

        if self._scatter_cbar is not None:
            self._scatter_cbar.remove()
            self._scatter_cbar = None

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            for ax in self.axes.flat:
                ax.clear()
        self.axes[0, 0].invert_yaxis()

        d = compute_windowed_diss(
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
        self._cached_diss = d
        P = d["P_windows"]
        self._P_win = P

        # Spectra at the selected depth for the row-2 spectral panels.
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

        # Background stratification on the Thorpe-sorted slow profile, then the
        # mixing coefficients on the same window grid.
        t_win = self._window_center_times(sel_fast)
        assert len(t_win) == len(P), "window-time grid drifted from compute_windowed_diss"
        half_w = 0.5 * self.diss_length / self.fs_fast
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            self._strat = sorted_stratification(
                t_win, half_w, self.t_slow, self.P, self.T, S=self.salinity
            )
            self._eps_mix = self._mixing_epsilon(d)
            self._chi_mix = self._mixing_chi(d)
            self._mix = mixing_coefficients(
                self._eps_mix, self._chi_mix, self._strat.N2, self._strat.dTdz
            )

        # Row 0
        self._draw_overview(self.axes[0, 0], s_slow, e_slow)
        self._draw_n2(self.axes[0, 1])
        self._draw_diffusivities(self.axes[0, 2])
        self._draw_gamma(self.axes[0, 3])

        # Row 1
        self._draw_dtdz(self.axes[1, 0])
        self._draw_eps_profile(self.axes[1, 1], P, d["epsilon"])
        self._draw_chi_profile(self.axes[1, 2])
        self._draw_kt_krho_scatter(self.axes[1, 3])

        # Row 2: spectra at depth + Lueck FM / chi FOM (shared base panels)
        self._draw_eps_spectra(self.axes[2, 0], sel_spec, self._cached_spec)
        self._draw_chi_m1m2_spectra(self.axes[2, 1], sel_spec)
        self._draw_fm_profile(self.axes[2, 2])
        self._draw_chi_fom_profile(self.axes[2, 3])

        pressure_axes = (
            [self.axes[0, c] for c in range(4)]
            + [self.axes[1, c] for c in range(3)]
            + [self.axes[2, 2], self.axes[2, 3]]
        )
        self._finish_draw(s_slow, e_slow, pressure_axes)

    # ------------------------------------------------------------------
    # Panels
    # ------------------------------------------------------------------

    @staticmethod
    def _no_data(ax, msg):
        ax.text(0.5, 0.5, msg, transform=ax.transAxes, ha="center", va="center")

    def _draw_n2(self, ax):
        """Panel (0,1): background N² vs pressure (log; only N² > 0)."""
        P = self._P_win
        N2 = self._strat.N2 if self._strat is not None else np.array([])
        valid = np.isfinite(N2) & (N2 > 0) & np.isfinite(P)
        if not np.any(valid):
            self._no_data(ax, "No stratified N²")
            return
        ax.plot(N2[valid], P[valid], "C2o-", markersize=3, linewidth=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("N² [s⁻²]")
        ax.set_ylabel("Pressure [dbar]")
        ax.set_title("N² (Thorpe-sorted)", fontsize=9)
        ax.grid(True, alpha=0.3, which="both")

    def _draw_dtdz(self, ax):
        """Panel (1,0): background dT/dz vs pressure (linear; positive down)."""
        P = self._P_win
        dTdz = self._strat.dTdz if self._strat is not None else np.array([])
        valid = np.isfinite(dTdz) & np.isfinite(P)
        if not np.any(valid):
            self._no_data(ax, "No dT/dz")
            return
        ax.plot(dTdz[valid], P[valid], "C5o-", markersize=3, linewidth=0.8)
        ax.axvline(0.0, color="k", linewidth=0.7, alpha=0.5)
        ax.set_xlabel("dT/dz [K m⁻¹]")
        ax.set_ylabel("Pressure [dbar]")
        ax.set_title("Background dT/dz", fontsize=9)
        ax.grid(True, alpha=0.3)

    def _draw_diffusivities(self, ax):
        """Panel (0,2): K_T (Osborn-Cox) and K_rho (Osborn) vs pressure (log)."""
        P = self._P_win
        if self._mix is None:
            self._no_data(ax, "No data")
            return
        has = False
        for vals, color, label in (
            (self._mix.K_T, "C0", "K_T (Osborn-Cox)"),
            (self._mix.K_rho, "C3", "K_rho (Osborn)"),
        ):
            valid = np.isfinite(vals) & (vals > 0) & np.isfinite(P)
            if np.any(valid):
                ax.plot(vals[valid], P[valid], "o-", color=color, markersize=3,
                        linewidth=0.8, label=label)
                has = True
        if not has:
            self._no_data(ax, "No valid K")
            return
        ax.set_xscale("log")
        ax.set_xlabel("K [m² s⁻¹]")
        ax.set_ylabel("Pressure [dbar]")
        ax.legend(fontsize=6, loc="lower left")
        ax.set_title("Diffusivity: K_T vs K_rho", fontsize=9)
        ax.grid(True, alpha=0.3, which="both")

    def _draw_gamma(self, ax):
        """Panel (0,3): measured mixing coefficient Γ vs pressure (log; ref 0.2)."""
        P = self._P_win
        Gamma = self._mix.Gamma if self._mix is not None else np.array([])
        valid = np.isfinite(Gamma) & (Gamma > 0) & np.isfinite(P)
        if not np.any(valid):
            self._no_data(ax, "No valid Γ")
            return
        ax.plot(Gamma[valid], P[valid], "C4o-", markersize=3, linewidth=0.8)
        ax.axvline(0.2, color="k", linestyle="--", linewidth=0.9, alpha=0.7, label="Γ = 0.2")
        ax.set_xscale("log")
        ax.set_xlabel("Γ = N²χ / (2ε(dT/dz)²)")
        ax.set_ylabel("Pressure [dbar]")
        ax.legend(fontsize=6, loc="lower left")
        ax.set_title("Mixing coefficient Γ", fontsize=9)
        ax.grid(True, alpha=0.3, which="both")

    def _draw_chi_profile(self, ax):
        """Panel (1,2): chi per thermistor vs pressure (mean of Batchelor & Kraichnan)."""
        d = self._cached_diss
        P = self._P_win
        if d is None or len(P) == 0:
            self._no_data(ax, "No data")
            return
        has = False
        for i, (name, _) in enumerate(self.therm_fast):
            c = _THERM_COLORS[i % len(_THERM_COLORS)]
            chi_b = d["chi_batchelor"][i]
            chi_k = d["chi_kraichnan"][i]
            chi_mean = np.where(
                np.isfinite(chi_b) & np.isfinite(chi_k),
                (chi_b + chi_k) / 2.0,
                np.where(np.isfinite(chi_b), chi_b, chi_k),
            )
            valid = np.isfinite(chi_mean) & (chi_mean > 0)
            if np.any(valid):
                ax.plot(chi_mean[valid], P[valid], f"{c}o-", markersize=3,
                        linewidth=0.8, label=name)
                has = True
        if not has:
            self._no_data(ax, "No valid χ")
            return
        ax.set_xscale("log")
        ax.set_xlabel("χ [K²/s]")
        ax.set_ylabel("Pressure [dbar]")
        ax.legend(fontsize=6, loc="lower left")
        ax.set_title("χ profile (Batch. + Kraich.)", fontsize=9)
        ax.grid(True, alpha=0.3, which="both")

    def _draw_kt_krho_scatter(self, ax):
        """Panel (1,3): K_T vs K_rho (log-log, 1:1 line), colored by pressure.

        Points on the 1:1 line have measured Γ ≈ 0.2 (K_T ≈ K_rho); the pressure
        colour shows how the agreement varies with depth.
        """
        if self._mix is None:
            self._no_data(ax, "No data")
            return
        K_T = self._mix.K_T
        K_rho = self._mix.K_rho
        valid = (
            np.isfinite(K_T) & (K_T > 0) & np.isfinite(K_rho) & (K_rho > 0)
            & np.isfinite(self._P_win)
        )
        if not np.any(valid):
            self._no_data(ax, "No valid K_T/K_rho")
            return
        x = K_rho[valid]
        y = K_T[valid]
        pcol = self._P_win[valid]
        ax.set_xscale("log")
        ax.set_yscale("log")
        sc = ax.scatter(x, y, c=pcol, cmap="viridis", s=26, alpha=0.85, edgecolors="none")
        assert self.fig is not None
        cbar = self.fig.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label("Pressure [dbar]", fontsize=7)
        cbar.ax.tick_params(labelsize=6)
        self._scatter_cbar = cbar
        lo = float(min(x.min(), y.min()))
        hi = float(max(x.max(), y.max()))
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.9, alpha=0.7, label="1:1 (Γ = 0.2)")
        ax.set_xlabel("K_rho (Osborn) [m² s⁻¹]")
        ax.set_ylabel("K_T (Osborn-Cox) [m² s⁻¹]")
        ax.legend(fontsize=6, loc="upper left")
        ax.set_title("K_T vs K_rho (color = P)", fontsize=9)
        ax.grid(True, alpha=0.3, which="both")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def mixing_look(
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
    salinity=None,
) -> MixingLookViewer:
    """Open an interactive mixing viewer for a .p file.

    Displays the background stratification (N², dT/dz) and the derived
    diapycnal-mixing quantities (K_T, Γ, K_rho), computed exactly as the
    pipeline does: per-window epsilon and chi with the background N²/dT/dz
    from a Thorpe-sorted (overturn-removed) profile.

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
        Pressure range (P_min, P_max) in dbar to highlight on the profiles.
    diss_length : int or None
        Dissipation window length [samples]. Default: 4 * fft_length.
    vehicle : str or None
        Vehicle type override (e.g. 'slocum_glider'). If None, read from
        instrument config.
    salinity : float or None
        Fixed practical salinity [PSU] for the stratification. If None, the
        profile's own JAC C/T conductivity is used when available, else 35.

    Returns
    -------
    MixingLookViewer
        The viewer instance.
    """
    pf = PFile(source)
    viewer = MixingLookViewer(
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
        salinity=salinity,
    )
    viewer.show()
    return viewer

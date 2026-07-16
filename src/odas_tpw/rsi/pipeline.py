"""Full VMP data processing pipeline.

.P file → L1Data → L2 → L3 (shear spectra) → L4 (epsilon)
                       → L3_chi (temperature gradient spectra) → L4_chi
                       → L5 (depth binning) → L6 (combine profiles)
"""

from __future__ import annotations

import logging
import warnings
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import xarray as xr

from odas_tpw.chi.l2_chi import L2ChiParams, process_l2_chi
from odas_tpw.chi.l3_chi import process_l3_chi
from odas_tpw.chi.l4_chi import (
    _CHI_FOM_LIMIT,
    _CHI_K_MAX_RATIO_MIN,
    L4ChiData,
    _compute_chi_final,
    process_l4_chi_epsilon,
    process_l4_chi_fit,
)
from odas_tpw.rsi.adapter import pfile_to_l1data
from odas_tpw.rsi.binning import bin_by_depth
from odas_tpw.rsi.combine import combine_profiles
from odas_tpw.scor160.io import L1Data, L2Params, L3Data, L3Params, L4Data
from odas_tpw.scor160.l2 import process_l2
from odas_tpw.scor160.l3 import process_l3
from odas_tpw.scor160.l4 import process_l4

logger = logging.getLogger(__name__)

# Canonical chi (temperature-gradient) vibration high-pass cutoff [Hz]. Fixed
# across the modular chi path (chi_io.py) and perturb, independent of the chi
# FFT length, so the rsi pipeline pins it here too rather than reusing the
# epsilon shear high-pass (which scales with the epsilon FFT length).
_CHI_HP_CUT = 0.25

# QC thresholds for the per-probe chi used in the derived mixing quantities
# (K_T/Gamma) are the SINGLE source of truth in l4_chi.py (imported above), so
# the reported L4ChiData.chi_final and the mixing products here are guaranteed
# to filter with identical limits — they cannot drift out of sync. The band
# mirrors the chi-from-epsilon fom_limit in window.py and the K_max_ratio < 0.5
# "most variance extrapolated" convention (see CLAUDE.md).


def _qc_chi_final(
    chi: np.ndarray,
    fom: np.ndarray,
    k_max_ratio: np.ndarray,
    fom_limit: float = _CHI_FOM_LIMIT,
    k_max_ratio_min: float = _CHI_K_MAX_RATIO_MIN,
) -> np.ndarray:
    """Per-window geometric-mean chi over probes that pass spectral QC.

    Thin wrapper over :func:`odas_tpw.chi.l4_chi._compute_chi_final` with the QC
    arguments always supplied. As of the 2026-07-03 review (Gemini C),
    ``L4ChiData.chi_final`` is itself QC'd at the source with these same
    thresholds, so the mixing products and the reported chi are filtered
    identically; this wrapper is retained for the two-sided-band QC contract it
    documents and tests. All inputs are ``(n_probe, n_window)``; returns
    ``(n_window,)``.
    """
    return _compute_chi_final(chi, fom, k_max_ratio, fom_limit, k_max_ratio_min)


def _epsilon_hp_cut(fs_fast: float, fft_length: int, override: float | None) -> float:
    """Shear high-pass cutoff [Hz] for epsilon, matching ``_compute_epsilon``.

    Half the inverse FFT-segment duration (ODAS ``get_diss_odas.m`` guidance):
    ``0.5 * fs_fast / fft_length`` — 0.25 Hz for a 1024-sample FFT at 512 Hz,
    and it scales with the FFT length so a 256-sample window gives 1.0 Hz. An
    explicit ``override`` (if not None) wins, for callers that pin the cutoff.
    """
    if override is not None:
        return override
    return 0.5 * fs_fast / fft_length


def _resolve_salinity(
    l1: L1Data, salinity: float | np.ndarray | None
) -> tuple[float | np.ndarray | None, bool]:
    """Choose the salinity used for both chi viscosity and stratification.

    Preference order, so the two consumers never disagree:

    1. The per-sample practical salinity measured from the profile's own
       conductivity (VMP JAC C/T, computed on the L1 fast time base in
       ``pfile_to_l1data``). MicroRiders have no .p-level conductivity, so this
       is empty for them.
    2. A user-supplied ``salinity`` (scalar, or an array on the fast time base;
       a mismatched-length array is collapsed to its nanmean).
    3. ``None`` → 35 PSU assumed downstream (``visc35`` / temperature-only N2).

    Returns ``(salinity_value, measured)`` where ``measured`` is True only when
    the per-sample measured salinity is used.
    """
    if l1.salinity.size == l1.n_time and bool(np.isfinite(l1.salinity).any()):
        return l1.salinity, True
    if isinstance(salinity, np.ndarray) and len(salinity) != l1.n_time:
        return float(np.nanmean(salinity)), False
    return salinity, False


def _epsilon_window_salinity(
    salinity: float | np.ndarray | None,
    sample_time: np.ndarray,
    n_time: int,
    win_time: np.ndarray,
    n_spectra: int,
) -> float | np.ndarray | None:
    """Per-window salinity for L4 epsilon viscosity.

    ``process_l4`` accepts only per-window salinity (size ``n_spectra``); it
    collapses any other-sized array to its nanmean.  A *measured* per-sample
    series (size ``n_time``) would therefore drive epsilon viscosity with a
    constant profile mean while chi and N2 use the per-window measured series —
    so the three viscosities would not share a salinity basis.  Interpolate the
    per-sample series onto the window center times so they do.  Scalars,
    ``None``, and already-per-window arrays pass through unchanged.
    """
    if not (isinstance(salinity, np.ndarray) and salinity.size == n_time and n_spectra > 0):
        return salinity
    finite = np.isfinite(salinity)
    if not finite.any():
        return salinity
    return np.asarray(np.interp(win_time, sample_time[finite], salinity[finite]), dtype=np.float64)


def run_pipeline(
    p_files: list[Path],
    output_dir: Path,
    *,
    # Profile detection
    P_min: float = 0.5,
    W_min: float | None = None,
    direction: str = "auto",
    min_duration: float = 7.0,
    speed: float | None = None,
    speed_tau: float = 1.5,
    speed_method: str | None = None,
    aoa_deg: float = 3.0,
    vehicle: str | None = None,
    # L2 params
    HP_cut: float | None = None,
    despike_thresh: float = 8.0,
    # Epsilon spectral params
    fft_length: int = 1024,
    diss_length: int | None = None,
    overlap: int | None = None,
    f_AA: float = 98.0,
    fit_order: int = 3,
    salinity: float | None = None,
    temperature: str | float = "auto",
    conductivity: str = "auto",
    # Chi cleaning params
    despike_T_thresh: float = 10.0,
    # Chi spectral params
    chi_fft_length: int = 1024,
    chi_diss_length: int | None = None,
    chi_overlap: int | None = None,
    fp07_model: str = "single_pole",
    spectrum_model: str = "kraichnan",
    fit_method: str = "iterative",
    # Chain selection
    compute_chi_epsilon: bool = True,
    compute_chi_fit: bool = False,
    # Binning
    bin_size: float = 1.0,
    # Output
    goodman: bool = True,
) -> Path:
    """Run the full processing pipeline.

    Per .P file, per profile:
    1. pfile_to_l1data → L1Data
    2. process_l2 → L2Data
    3. process_l3 → L3Data (shear spectra)
    4. process_l4 → L4Data (epsilon)
    5. process_l3_chi → L3ChiData (temperature gradient spectra)
    6. process_l4_chi_epsilon or process_l4_chi_fit → L4ChiData
    7. bin_by_depth → L5
    8. combine_profiles → L6

    Parameters
    ----------
    p_files : list of Path
        Input .P files.
    output_dir : Path
        Base output directory.
    W_min : float or None
        Profile-detection fall-rate floor [dbar/s]. ``None`` (default)
        resolves per file from the vehicle direction — 0.3 for a
        free-falling profiler, 0.05 for glide/horizontal platforms — and
        the resolved value also feeds ``L2Params.profile_min_W``.
    temperature : str or float
        Reference-temperature source: "auto" (first plausible of T1..Tn, T,
        JAC_T — see helpers.resolve_temperature_channel), an explicit channel
        name (QC failure warns but proceeds), or a number = constant
        reference temperature [degC] (ODAS ``constant_temp`` parity). The
        resolved source lands in the L4 products' ``temperature_source`` /
        ``temperature_qc`` attrs.
    conductivity : str
        Conductivity channel for the measured practical salinity
        ("auto" = JAC_C when present).
    (all other params documented in plan)

    Returns
    -------
    Path
        Output directory.
    """
    from odas_tpw.rsi.helpers import _validate_speed_selection

    # Same speed/speed_method exclusivity contract as the eps/chi commands —
    # the adapter would otherwise silently prefer the fixed speed. Validate
    # the EXPLICIT combination first (None = unset), and only then resolve
    # the unset method to the historical pressure default: a "pressure"
    # default here would make a plain fixed speed always trip the guard.
    _validate_speed_selection(speed, speed_method)
    if speed_method is None:
        speed_method = "pressure"

    if diss_length is None:
        diss_length = 4 * fft_length
    if overlap is None:
        overlap = diss_length // 2
    if chi_diss_length is None:
        chi_diss_length = 4 * chi_fft_length
    if chi_overlap is None:
        chi_overlap = chi_diss_length // 2

    if diss_length != chi_diss_length:
        warnings.warn(
            f"Epsilon and chi use different dissipation lengths "
            f"(diss_length={diss_length}, chi_diss_length={chi_diss_length}). "
            f"Depth-binned estimates will have different vertical resolution.",
            stacklevel=2,
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for p_path in p_files:
        logger.info("=" * 60)
        logger.info(p_path.name)
        logger.info("=" * 60)

        # Per-file guard (mirrors the eps/chi CLI loops): one bad file — e.g.
        # unreadable, or no plausible reference temperature — must not kill a
        # multi-file pipeline run.
        try:
            _pipeline_one_file(
                p_path,
                output_dir,
                P_min=P_min,
                W_min=W_min,
                direction=direction,
                min_duration=min_duration,
                speed=speed,
                speed_tau=speed_tau,
                speed_method=speed_method,
                aoa_deg=aoa_deg,
                vehicle=vehicle,
                HP_cut=HP_cut,
                despike_thresh=despike_thresh,
                fft_length=fft_length,
                diss_length=diss_length,
                overlap=overlap,
                f_AA=f_AA,
                fit_order=fit_order,
                salinity=salinity,
                temperature=temperature,
                conductivity=conductivity,
                despike_T_thresh=despike_T_thresh,
                chi_fft_length=chi_fft_length,
                chi_diss_length=chi_diss_length,
                chi_overlap=chi_overlap,
                fp07_model=fp07_model,
                spectrum_model=spectrum_model,
                fit_method=fit_method,
                compute_chi_epsilon=compute_chi_epsilon,
                compute_chi_fit=compute_chi_fit,
                bin_size=bin_size,
                goodman=goodman,
            )
        except (OSError, ValueError, RuntimeError) as e:
            logger.error(f"{p_path.name}: ERROR: {e}")

    logger.info(f"Pipeline complete. Output: {output_dir}")
    return output_dir


def _pipeline_one_file(
    p_path: Path,
    output_dir: Path,
    *,
    P_min: float,
    W_min: float | None,
    direction: str,
    min_duration: float,
    speed: float | None,
    speed_tau: float,
    speed_method: str,
    aoa_deg: float,
    vehicle: str | None,
    HP_cut: float | None,
    despike_thresh: float,
    fft_length: int,
    diss_length: int,
    overlap: int,
    f_AA: float,
    fit_order: int,
    salinity: float | None,
    temperature: str | float,
    conductivity: str,
    despike_T_thresh: float,
    chi_fft_length: int,
    chi_diss_length: int,
    chi_overlap: int,
    fp07_model: str,
    spectrum_model: str,
    fit_method: str,
    compute_chi_epsilon: bool,
    compute_chi_fit: bool,
    bin_size: float,
    goodman: bool,
) -> None:
    """Process one .P file (run_pipeline's per-file body)."""
    from odas_tpw.rsi.helpers import (
        _resolve_reference_temperature,
        pfile_channel_types,
        temperature_candidates,
    )
    from odas_tpw.rsi.p_file import PFile
    from odas_tpw.rsi.profile import _smooth_fall_rate, get_profiles

    pf = PFile(p_path)
    pfile_dir = output_dir / p_path.stem
    pfile_dir.mkdir(parents=True, exist_ok=True)

    # Resolve vehicle, direction, tau, and the detection floor per file.
    # Resolved once so profile detection and L2Params.profile_min_W (the L2
    # section selector) apply the same floor.
    from odas_tpw.rsi.vehicle import resolve_detection

    file_vehicle = vehicle or pf.config["instrument_info"].get("vehicle", "").lower()
    file_direction, file_tau, file_W_min = resolve_detection(
        direction, file_vehicle, W_min=W_min
    )
    file_speed_tau = file_tau if speed_tau == 1.5 else speed_tau
    logger.info(
        "profile detection: direction=%s W_min=%g dbar/s (vehicle=%r)",
        file_direction,
        file_W_min,
        file_vehicle,
    )

    # Detect profiles on slow-rate pressure
    P_slow = pf.channels.get("P_dP", pf.channels.get("P"))
    assert P_slow is not None, "No pressure channel (P or P_dP) found"

    # Resolve the reference-temperature source ONCE per file (full-channel QC)
    # for provenance; the concrete selection is passed to the adapter so the
    # per-profile resolution cannot diverge from what the attrs report.
    if temperature == "auto" and not temperature_candidates(
        pf.channels, len(pf.t_slow), pfile_channel_types(pf)
    ):
        # Historical tolerance: an instrument with no slow temperature channel
        # still runs; the adapter warns and fills L1.temp with NaN (downstream
        # substitutes 10 degC loudly).
        temp_for_l1: str | float = "auto"
        temperature_source = "none"
        temperature_qc = "no slow temperature channel found"
    else:
        # Same shared resolver as the modular path (load_channels): handles
        # auto/explicit channels AND the numeric constant escape hatch, so an
        # implausible constant (99 degC, NaN) warns and records its QC reason
        # identically on both paths (and a bool is rejected identically).
        # types= carries channel types for the sbt CT candidate tail (#141).
        _T, temperature_source, temperature_qc = _resolve_reference_temperature(
            pf.channels,
            len(pf.t_slow),
            temperature,
            P_slow,
            p_path.name,
            types=pfile_channel_types(pf),
        )
        temp_for_l1 = (
            float(temperature) if isinstance(temperature, (int, float)) else temperature_source
        )

    W_slow = _smooth_fall_rate(P_slow, pf.fs_slow, tau=file_speed_tau)
    profiles = get_profiles(
        P_slow,
        W_slow,
        pf.fs_slow,
        P_min=P_min,
        W_min=file_W_min,
        direction=file_direction,
        min_duration=min_duration,
    )

    if not profiles:
        from odas_tpw.scor160.profile import explain_no_profiles

        logger.warning(
            explain_no_profiles(
                P_slow, W_slow, P_min=P_min, W_min=file_W_min, direction=file_direction
            )
        )
        return

    logger.info(f"{len(profiles)} profile(s) detected")

    # Epsilon shear high-pass cutoff, scaled with the epsilon FFT length
    # exactly as the modular _compute_epsilon / perturb path: 0.25 Hz at the
    # default 1024-sample FFT (512 Hz), 1.0 Hz at 256 samples. Chi keeps the
    # fixed canonical 0.25 Hz (see _CHI_HP_CUT) regardless of its FFT length.
    eps_hp_cut = _epsilon_hp_cut(pf.fs_fast, fft_length, HP_cut)

    all_binned = []
    profile_metadata = []

    for prof_idx, (s_slow, e_slow) in enumerate(profiles):
        prof_num = prof_idx + 1
        prof_dir = pfile_dir / f"profile_{prof_num:03d}"
        prof_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Profile {prof_num}: slow samples {s_slow}-{e_slow}")

        binned = _process_profile(
            pf,
            prof_slice=(s_slow, e_slow),
            prof_num=prof_num,
            prof_dir=prof_dir,
            p_path=p_path,
            speed=speed,
            direction=file_direction,
            speed_tau=file_speed_tau,
            speed_method=speed_method,
            aoa_deg=aoa_deg,
            l2_params=L2Params(
                HP_cut=eps_hp_cut,
                despike_sh=np.array([despike_thresh, 0.5, 0.04]),
                despike_A=np.array([np.inf, 0.5, 0.04]),
                profile_min_W=file_W_min,
                profile_min_P=P_min,
                profile_min_duration=min_duration,
                # Vehicle-resolved tau (matches L1 speed + profile detection);
                # the raw user default 1.5 would re-smooth glider speed at the
                # wrong 0.68/1.5 cutoff for non-VMP vehicles.
                speed_tau=file_speed_tau,
            ),
            l3_params=L3Params(
                fft_length=fft_length,
                diss_length=diss_length,
                overlap=overlap,
                HP_cut=eps_hp_cut,
                fs_fast=0,  # placeholder, overridden in _process_profile
                goodman=goodman,
            ),
            f_AA=f_AA,
            fit_order=fit_order,
            salinity=salinity,
            temperature=temp_for_l1,
            conductivity=conductivity,
            temperature_source=temperature_source,
            temperature_qc=temperature_qc,
            despike_T_thresh=despike_T_thresh,
            chi_fft_length=chi_fft_length,
            chi_diss_length=chi_diss_length,
            chi_overlap=chi_overlap,
            fp07_model=fp07_model,
            spectrum_model=spectrum_model,
            fit_method=fit_method,
            compute_chi_epsilon=compute_chi_epsilon,
            compute_chi_fit=compute_chi_fit,
            bin_size=bin_size,
        )

        if binned is not None:
            all_binned.append(binned)
            profile_metadata.append(
                {
                    "source_file": p_path.name,
                    "profile_number": prof_num,
                    "s_slow": s_slow,
                    "e_slow": e_slow,
                }
            )

    # Step 8: Combine profiles
    if all_binned:
        combined = combine_profiles(all_binned, profile_metadata)
        combined_path = pfile_dir / "L6_combined.nc"
        combined.to_netcdf(combined_path)
        logger.info(f"Combined: {combined_path}")


def _process_profile(
    pf,
    prof_slice: tuple[int, int],
    prof_num: int,
    prof_dir: Path,
    p_path: Path,
    *,
    speed: float | None,
    direction: str,
    speed_tau: float,
    speed_method: str = "pressure",
    aoa_deg: float = 3.0,
    l2_params: L2Params,
    l3_params: L3Params,
    f_AA: float,
    fit_order: int,
    salinity: float | None,
    temperature: str | float = "auto",
    conductivity: str = "auto",
    temperature_source: str | None = None,
    temperature_qc: str | None = None,
    despike_T_thresh: float,
    chi_fft_length: int,
    chi_diss_length: int,
    chi_overlap: int,
    fp07_model: str,
    spectrum_model: str,
    fit_method: str,
    compute_chi_epsilon: bool,
    compute_chi_fit: bool,
    bin_size: float,
) -> xr.Dataset | None:
    """Process a single profile: L1→L2→L3→L4→chi→binning→NetCDF."""
    s_slow, e_slow = prof_slice

    # Step 1: L1Data
    l1 = pfile_to_l1data(
        pf,
        profile_slice=(s_slow, e_slow),
        speed=speed,
        direction=direction,
        speed_tau=speed_tau,
        speed_method=speed_method,
        aoa_deg=aoa_deg,
        temperature=temperature,
        conductivity=conductivity,
    )

    # Step 2: L2
    l2 = process_l2(l1, l2_params)

    # Step 3: L3 shear spectra (update fs_fast from actual L1)
    l3_params = L3Params(
        fft_length=l3_params.fft_length,
        diss_length=l3_params.diss_length,
        overlap=l3_params.overlap,
        HP_cut=l3_params.HP_cut,
        fs_fast=l1.fs_fast,
        goodman=l3_params.goodman,
    )
    l3 = process_l3(l2, l1, l3_params)

    if l3.n_spectra == 0:
        logger.warning("No valid spectral windows")
        return None

    # Salinity resolved once (measured JAC C/T > user-supplied > 35 PSU) so the
    # epsilon, chi, and N2 viscosities never disagree within a run. This matches
    # perturb's chi.salinity:"measured" feeding both process_l3_chi and N2.
    chi_salinity, measured_sal = _resolve_salinity(l1, salinity)
    # chi (process_l3_chi) and N2 (sorted_stratification) consume the per-sample
    # series with timestamps; process_l4 only takes per-window salinity, so a
    # measured series is interpolated onto the L3 window times here — otherwise
    # epsilon would silently use the profile mean while chi/N2 use per-window.
    eps_salinity = _epsilon_window_salinity(
        chi_salinity, l1.time, l1.n_time, l3.time, l3.n_spectra
    )

    # Step 4: L4 epsilon
    l4 = process_l4(
        l3,
        temp=l3.temp,
        salinity=eps_salinity,
        f_AA=f_AA,
        fit_order=fit_order,
        num_ffts=2 * (l3_params.diss_length // l3_params.fft_length) - 1,
        n_v=l1.n_vib if l3_params.goodman else 0,
        diss_length_s=l3_params.diss_length / l1.fs_fast,
    )
    logger.info(f"Epsilon: {l4.n_spectra} estimates")

    # Step 5: L2_chi cleaning + chi spectra (if temperature data available)
    l3_chi = None
    l4_chi_eps = None
    l4_chi_fit_result = None

    if l1.has_temp_fast:
        l2_chi_params = L2ChiParams(
            HP_cut=_CHI_HP_CUT,
            despike_T=np.array([despike_T_thresh, 0.5, 0.04]),
        )
        l2_chi = process_l2_chi(l1, l2, l2_chi_params)

        l3_chi_params = L3Params(
            fft_length=chi_fft_length,
            diss_length=chi_diss_length,
            overlap=chi_overlap,
            HP_cut=_CHI_HP_CUT,
            fs_fast=l1.fs_fast,
            goodman=l3_params.goodman,
        )

        try:
            l3_chi = process_l3_chi(
                l2_chi,
                l3_chi_params,
                fp07_model=fp07_model,
                salinity=chi_salinity,
            )
        except (ValueError, RuntimeError) as e:
            logger.error(f"Chi spectra error: {e}")

        if l3_chi is not None and l3_chi.n_spectra > 0:
            # Step 6a: Chi from epsilon (Method 1)
            if compute_chi_epsilon and l4.n_spectra > 0:
                try:
                    l4_chi_eps = process_l4_chi_epsilon(
                        l3_chi,
                        l4,
                        spectrum_model=spectrum_model,
                        f_AA=f_AA,
                    )
                    n_valid = np.sum(np.isfinite(l4_chi_eps.chi_final))
                    logger.info(f"Chi (Method 1): {n_valid} valid estimates")
                except (ValueError, RuntimeError) as e:
                    logger.error(f"Chi epsilon error: {e}")

            # Step 6b: Chi from fit (Method 2)
            if compute_chi_fit:
                try:
                    l4_chi_fit_result = process_l4_chi_fit(
                        l3_chi,
                        spectrum_model=spectrum_model,
                        fit_method=fit_method,
                        f_AA=f_AA,
                    )
                    n_valid = np.sum(np.isfinite(l4_chi_fit_result.chi_final))
                    logger.info(f"Chi (Method 2): {n_valid} valid estimates")
                except (ValueError, RuntimeError) as e:
                    logger.error(f"Chi fit error: {e}")

    # Step 6c: Stratification + derived mixing quantities on the chi window
    # grid. N2/dTdz need only the CTD profile (no epsilon/chi) and K_T needs
    # only chi, so they are written whenever a chi result and temperature
    # exist; Gamma/K_rho additionally require shear epsilon. chi_primary is the
    # Method-1 result when present (chi-from-epsilon, the primary product) else
    # the Method-2 fit, so a Method-2-only run still gets these instead of
    # silently dropping the whole block (bug_003).
    chi_primary = l4_chi_eps if l4_chi_eps is not None else l4_chi_fit_result
    mixing_vars: dict[str, tuple[np.ndarray, dict]] | None = None
    if chi_primary is not None and chi_primary.n_spectra > 0 and l1.temp.size == l1.n_time:
        try:
            from odas_tpw.processing.mixing import (
                mixing_coefficients,
                pair_nearest,
                sorted_stratification,
            )

            # Same salinity as chi viscosity (resolved once, above): measured
            # JAC C/T > user-supplied > 35 PSU.
            half_w = 0.5 * chi_diss_length / l1.fs_fast
            strat = sorted_stratification(
                chi_primary.time,
                half_w,
                l1.time,
                l1.pres,
                l1.temp,
                S=chi_salinity,
            )
            # Pair epsilon from the nearest shear window when available; with no
            # shear epsilon K_T = chi/(2 dT/dz^2) stays valid while Gamma/K_rho
            # fall to NaN (bug_003). The else is defensive: run_pipeline returns
            # early when the shear L3 is empty (l4.n_spectra == l3.n_spectra),
            # so l4.n_spectra > 0 always holds here today — but mixing_coefficients
            # is kept robust to an all-NaN epsilon for reuse/future entry points.
            if l4.n_spectra > 0:
                eps_on_chi = pair_nearest(l4.time, l4.epsi_final, chi_primary.time)
            else:
                eps_on_chi = np.full(chi_primary.n_spectra, np.nan)
            # QC the per-probe chi before the mixing products (K_T/Gamma) so a
            # high-fom / poorly-resolved thermistor window cannot bias them. This
            # matches chi_primary.chi_final, which is now QC'd at the source with
            # the same thresholds; the raw per-probe chi_primary.chi array is what
            # the L4_chi output carries for traceability.
            chi_for_mixing = _qc_chi_final(
                chi_primary.chi, chi_primary.fom, chi_primary.K_max_ratio
            )
            mix = mixing_coefficients(eps_on_chi, chi_for_mixing, strat.N2, strat.dTdz)
            if measured_sal:
                sal_note = "practical salinity from JAC_C/JAC_T/P (TEOS-10)"
            elif salinity is None:
                sal_note = (
                    "salinity assumed 35 PSU (none provided); N2 reflects "
                    "temperature stratification only"
                )
            else:
                sal_note = "salinity from user-provided values"
            mixing_vars = {
                "N2": (
                    strat.N2,
                    {
                        "units": "s-2",
                        "long_name": "buoyancy frequency squared (window scale)",
                        "comment": (
                            "TEOS-10 (gsw.Nsquared) between the shallow- and "
                            "deep-half means of each chi window after Thorpe-"
                            f"sorting to a stable profile (overturns removed); {sal_note}."
                        ),
                    },
                ),
                "dTdz": (
                    strat.dTdz,
                    {
                        "units": "K m-1",
                        "long_name": "background temperature gradient (positive down)",
                        "comment": (
                            "Least-squares slope of the Thorpe-sorted in-situ "
                            "temperature vs depth over each chi window."
                        ),
                    },
                ),
                "K_T": (
                    mix.K_T,
                    {
                        "units": "m2 s-1",
                        "long_name": "Osborn-Cox eddy diffusivity of heat",
                        "comment": (
                            "K_T = chi / (2*(dT/dz)^2), Osborn & Cox (1972), "
                            "doi:10.1080/03091927208236085. NaN where "
                            "|dT/dz| < 1e-4 K/m (well-mixed)."
                        ),
                    },
                ),
                "Gamma": (
                    mix.Gamma,
                    {
                        "units": "1",
                        "long_name": "mixing coefficient (measured)",
                        "comment": (
                            "Gamma = N2*chi / (2*epsilon*(dT/dz)^2), Oakey "
                            "(1982), doi:10.1175/1520-0485(1982)012<0256:DOTROD>"
                            "2.0.CO;2. epsilon paired from the nearest shear-"
                            "probe window. NaN where N2 < 1e-9 s-2 or "
                            "|dT/dz| < 1e-4 K/m. Canonical value ~0.2."
                        ),
                    },
                ),
                "K_rho": (
                    mix.K_rho,
                    {
                        "units": "m2 s-1",
                        "long_name": "Osborn diapycnal diffusivity (Gamma_0 = 0.2)",
                        "comment": (
                            "K_rho = 0.2*epsilon/N2, Osborn (1980), "
                            "doi:10.1175/1520-0485(1980)010<0083:EOTLRO>2.0.CO;2. "
                            "Compare with K_T: agreement implies the measured "
                            "Gamma is near the canonical 0.2. NaN where "
                            "N2 < 1e-9 s-2 or K_rho > 10 m2 s-1 (physically "
                            "implausible diffusivity: the unbounded near-floor-N2 "
                            "artifact, or contaminated near-surface windows where "
                            "epsilon is itself spurious)."
                        ),
                    },
                ),
            }
            n_gamma = int(np.sum(np.isfinite(mix.Gamma)))
            logger.info(f"Mixing quantities: {n_gamma} valid Gamma estimates")
        except (ValueError, RuntimeError) as e:
            logger.error(f"Mixing quantities error: {e}")

    # Step 7: Depth binning
    binned_parts = []
    have_eps = l4.n_spectra > 0
    have_chi = chi_primary is not None and chi_primary.n_spectra > 0

    # Shared depth grid so eps and chi bin onto a bit-identical depth_bin coord:
    # independent per-array np.arange grids can differ by an ULP for a
    # non-binary-exact bin_size, and the join="outer" merge below would then
    # split one physical depth into two NaN-padded columns (M-16). With both
    # present, snap a single range from the union of pressures.
    shared_range: tuple[float, float] | None = None
    if have_eps and chi_primary is not None and have_chi:
        all_pres = np.concatenate(
            [np.asarray(l4.pres, dtype=float), np.asarray(chi_primary.pres, dtype=float)]
        )
        finite = all_pres[np.isfinite(all_pres)]
        if finite.size:
            shared_range = (
                float(np.floor(np.min(finite) / bin_size) * bin_size),
                float(np.ceil(np.max(finite) / bin_size) * bin_size),
            )

    if have_eps:
        eps_bin = bin_by_depth(
            l4.pres,
            {"epsilon": l4.epsi_final, "P_mean": l4.pres, "speed": l4.pspd_rel},
            bin_size=bin_size,
            pres_range=shared_range,
        )
        binned_parts.append(eps_bin)

    if chi_primary is not None and have_chi:
        chi_vals: dict[str, np.ndarray] = {"chi": chi_primary.chi_final}
        if mixing_vars is not None:
            chi_vals.update({name: arr for name, (arr, _a) in mixing_vars.items()})
        chi_bin = bin_by_depth(
            chi_primary.pres,
            chi_vals,
            bin_size=bin_size,
            pres_range=shared_range,
        )
        binned_parts.append(chi_bin)

    binned = None
    if binned_parts:
        binned = binned_parts[0]
        for extra in binned_parts[1:]:
            binned = binned.merge(extra, join="outer")
        binned.attrs["profile_number"] = prof_num
        binned.attrs["source_file"] = p_path.name
        binned.attrs["bin_size"] = bin_size

        binned.to_netcdf(prof_dir / "L5_binned.nc")

    # Write per-level NetCDF
    time_ref = pf.start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
    v1_provenance = getattr(pf, "v1_provenance", None) or None
    _write_l4_epsilon(
        l4,
        l3,
        prof_dir / "L4_epsilon.nc",
        pf,
        temperature_source=temperature_source,
        temperature_qc=temperature_qc,
        provenance=v1_provenance,
    )

    # Attach the mixing vars to whichever chi product they were computed on
    # (chi_primary): Method-1 when present, else the Method-2 fit (bug_003).
    if l4_chi_eps is not None:
        _write_l4_chi(
            l4_chi_eps,
            prof_dir / "L4_chi_epsilon.nc",
            time_ref,
            extra_vars=mixing_vars if chi_primary is l4_chi_eps else None,
            temperature_source=temperature_source,
            temperature_qc=temperature_qc,
            provenance=v1_provenance,
        )
    if l4_chi_fit_result is not None:
        _write_l4_chi(
            l4_chi_fit_result,
            prof_dir / "L4_chi_fit.nc",
            time_ref,
            extra_vars=mixing_vars if chi_primary is l4_chi_fit_result else None,
            temperature_source=temperature_source,
            temperature_qc=temperature_qc,
            provenance=v1_provenance,
        )

    return binned


def _write_l4_epsilon(
    l4: L4Data,
    l3: L3Data,
    path: Path,
    pf,
    temperature_source: str | None = None,
    temperature_qc: str | None = None,
    provenance: dict[str, str] | None = None,
) -> None:
    """Write L4 epsilon to NetCDF."""
    n_shear = l4.n_shear
    probe_names = [f"sh{i + 1}" for i in range(n_shear)]
    time_ref = pf.start_time.strftime("%Y-%m-%dT%H:%M:%SZ")

    ds = xr.Dataset(
        {
            "epsilon": (
                ["probe", "time"],
                l4.epsi,
                {
                    "units": "W kg-1",
                    "long_name": "TKE dissipation rate per shear probe",
                },
            ),
            "epsilon_final": (
                ["time"],
                l4.epsi_final,
                {
                    "units": "W kg-1",
                    "long_name": "TKE dissipation rate (best estimate)",
                },
            ),
            "fom": (
                ["probe", "time"],
                l4.fom,
                {
                    "units": "1",
                    "long_name": "figure of merit (observed/Nasmyth variance ratio)",
                    "comment": (
                        "Values near 1.0 indicate a good fit. NOT the MAD-based "
                        "ATOMIX/Rockland FM statistic; see the FM variable for that."
                    ),
                },
            ),
            "FM": (
                ["probe", "time"],
                l4.FM if l4.FM is not None else np.full_like(l4.fom, np.nan),
                {
                    "units": "1",
                    "long_name": "Lueck (2022) figure of merit",
                    "comment": (
                        "FM = MAD_ln / (T_M * sigma_ln) with sigma_ln = "
                        "sqrt(1.25 * N_eff**(-7/9)) and T_M = 0.8 + sqrt(1.56/N_s) "
                        "(Lueck 2022, doi:10.1175/JTECH-D-21-0051.1). A good fit "
                        "sits near its expected value ~0.7-0.8, NOT 0 (spectral "
                        "scatter floors this MAD statistic); ATOMIX recommends "
                        "rejecting FM > ~1.15."
                    ),
                },
            ),
            "mad": (
                ["probe", "time"],
                l4.mad,
                {
                    "units": "1",
                    "long_name": "mean absolute deviation ratio",
                },
            ),
            "kmax": (
                ["probe", "time"],
                l4.kmax,
                {
                    "units": "cpm",
                    "long_name": "upper integration wavenumber",
                },
            ),
            "var_resolved": (
                ["probe", "time"],
                l4.var_resolved,
                {
                    "units": "1",
                    "long_name": "fraction of variance resolved",
                },
            ),
            "sea_water_pressure": (
                ["time"],
                l4.pres,
                {
                    "units": "dbar",
                    "standard_name": "sea_water_pressure",
                    "long_name": "mean pressure per window",
                },
            ),
            "pspd_rel": (
                ["time"],
                l4.pspd_rel,
                {
                    "units": "m s-1",
                    "long_name": "profiling speed",
                },
            ),
        },
        coords={
            "probe": probe_names,
            "time": (
                ["time"],
                l4.time,
                {
                    "units": f"seconds since {time_ref}",
                    "standard_name": "time",
                    "calendar": "standard",
                },
            ),
        },
        attrs={
            "Conventions": "CF-1.13, ACDD-1.3",
            "source_file": str(pf.filepath.name),
            "history": f"Pipeline on {datetime.now(UTC).isoformat()}",
        },
    )
    if temperature_source is not None:
        ds.attrs["temperature_source"] = temperature_source
    if temperature_qc is not None:
        ds.attrs["temperature_qc"] = temperature_qc
    if provenance:
        # Complete v1->v6 translation provenance (issue #141): the
        # setup-file md5 + sens source audit the sens^-2 epsilon scaling.
        ds.attrs.update({k: str(v) for k, v in provenance.items()})
    ds.to_netcdf(path)


def _write_l4_chi(
    l4_chi: L4ChiData,
    path: Path,
    time_ref: str,
    extra_vars: dict[str, tuple[np.ndarray, dict]] | None = None,
    temperature_source: str | None = None,
    temperature_qc: str | None = None,
    provenance: dict[str, str] | None = None,
) -> None:
    """Write L4 chi to NetCDF.

    ``extra_vars`` maps variable name -> (1-D array on the time dim,
    attrs dict); used for the derived mixing quantities (N2, dTdz, K_T,
    Gamma, K_rho).
    """
    n_gradt = l4_chi.n_gradt
    probe_names = [f"T{i + 1}" for i in range(n_gradt)]

    ds = xr.Dataset(
        {
            "chi": (
                ["probe", "time"],
                l4_chi.chi,
                {
                    "units": "K2 s-1",
                    "long_name": "thermal variance dissipation rate per probe",
                },
            ),
            "chi_final": (
                ["time"],
                l4_chi.chi_final,
                {
                    "units": "K2 s-1",
                    "long_name": "thermal variance dissipation rate (best estimate)",
                },
            ),
            "epsilon_T": (
                ["probe", "time"],
                l4_chi.epsilon_T,
                {
                    "units": "W kg-1",
                    "long_name": "TKE dissipation rate from chi",
                },
            ),
            "kB": (
                ["probe", "time"],
                l4_chi.kB,
                {
                    "units": "cpm",
                    "long_name": "Batchelor wavenumber",
                },
            ),
            "K_max": (
                ["probe", "time"],
                l4_chi.K_max,
                {
                    "units": "cpm",
                    "long_name": "upper integration wavenumber",
                },
            ),
            "fom": (
                ["probe", "time"],
                l4_chi.fom,
                {
                    "units": "1",
                    "long_name": "figure of merit",
                },
            ),
            "K_max_ratio": (
                ["probe", "time"],
                l4_chi.K_max_ratio,
                {
                    "units": "1",
                    "long_name": "K_max to Batchelor wavenumber ratio",
                },
            ),
            "sea_water_pressure": (
                ["time"],
                l4_chi.pres,
                {
                    "units": "dbar",
                    "standard_name": "sea_water_pressure",
                    "long_name": "mean pressure per window",
                },
            ),
            "pspd_rel": (
                ["time"],
                l4_chi.pspd_rel,
                {
                    "units": "m s-1",
                    "long_name": "profiling speed",
                },
            ),
        },
        coords={
            "probe": probe_names,
            "time": (
                ["time"],
                l4_chi.time,
                {
                    "units": f"seconds since {time_ref}",
                    "standard_name": "time",
                    "calendar": "standard",
                },
            ),
        },
        attrs={
            "Conventions": "CF-1.13, ACDD-1.3",
            "method": l4_chi.method,
            "history": f"Pipeline on {datetime.now(UTC).isoformat()}",
        },
    )
    if temperature_source is not None:
        ds.attrs["temperature_source"] = temperature_source
    if temperature_qc is not None:
        ds.attrs["temperature_qc"] = temperature_qc
    if provenance:
        # Complete v1->v6 translation provenance (issue #141).
        ds.attrs.update({k: str(v) for k, v in provenance.items()})
    if extra_vars:
        for name, (arr, attrs) in extra_vars.items():
            ds[name] = xr.DataArray(arr, dims=["time"], attrs=attrs)
    ds.to_netcdf(path)

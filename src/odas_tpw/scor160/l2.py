"""L1 → L2: section selection, despiking, high-pass filtering, speed.

Implements Sec. 3.2 of Lueck et al. (2024).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt

from odas_tpw.scor160.despike import despike
from odas_tpw.scor160.io import L1Data, L2Data, L2Params


def _matlab_padlen(b: np.ndarray, a: np.ndarray) -> int:
    """Return the ``filtfilt`` *padlen* that matches MATLAB's convention.

    MATLAB pads with ``3*(nfilt-1)`` reflected samples where
    ``nfilt = max(len(a), len(b))``.  scipy defaults to ``3*nfilt``,
    which produces different edge transients.  For a 1st-order
    Butterworth (nfilt=2) this is padlen=3 vs scipy's 6 — enough to
    cause ~1 % vibration RMS error on records where the section spans
    the entire file (e.g. Nemo MR1000).
    """
    return 3 * (max(len(a), len(b)) - 1)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def process_l2(l1: L1Data, params: L2Params) -> L2Data:
    """Process L1 data into L2 following ATOMIX best practices.

    Steps (per Sec. 3.2):
      1. Compute profiling speed W from dP/dt (or use pre-computed if available).
      2. Select sections satisfying min speed, depth, duration criteria.
      3. High-pass filter shear and vibration.
      4. De-spike shear (and vibration) within selected sections.

    Parameters
    ----------
    l1 : L1Data
        Level-1 converted time series.
    params : L2Params
        Processing parameters (typically read from the benchmark file).

    Returns
    -------
    L2Data with cleaned shear, vibration, speed, and section labels.
    """
    fs = l1.fs_fast

    # 1. Profiling speed — smooth with ODAS-style LP filter (0.68/tau Hz)
    if l1.has_speed:
        pspd = l1.pspd_rel.copy()
    else:
        # ODAS smooths the fall rate exactly once, at 0.68/tau
        # (odas_p2mat.m:699-701).  When speed_tau > 0 that filter is
        # applied below, so no pre-smoothing happens here; previously the
        # SHEAR high-pass cutoff (params.HP_cut, an unrelated parameter)
        # was used as a pre-smoothing cutoff before the 0.68/tau pass.
        # The HP_cut fallback is kept only for speed_tau <= 0, where it
        # is the sole smoothing applied.
        pre_cut = None if params.speed_tau > 0 else params.HP_cut
        pspd = _compute_speed(l1.pres, fs, pre_cut, l1.profile_dir)

    tau = params.speed_tau
    if tau > 0:
        f_lp = 0.68 / tau
        b_lp, a_lp = butter(1, f_lp / (fs / 2))
        pspd = _filtfilt_nan(b_lp, a_lp, pspd)

    # 2. Section selection
    section_number = _select_sections(
        pspd,
        l1.pres,
        fs,
        min_speed=params.profile_min_W,
        min_pressure=params.profile_min_P,
        min_duration=params.profile_min_duration,
        direction=l1.profile_dir,
    )

    # 3 & 4. High-pass filter, then de-spike shear and vibration.
    #
    # ODAS applies the HP filter before despiking: the reference L2 data
    # contains exactly-constant replacement values within spike regions,
    # which is what our despike produces when operating on the HP-filtered
    # signal.  Reversing the order (despike→HP) would spread the constant
    # replacement through the filter, giving ~1-5 % extra error.
    shear_out = _hp_filter(l1.shear, fs, params.HP_cut)
    vib_out = _hp_filter(l1.vib, fs, params.HP_cut)

    # Despike parameters: [threshold, smooth_freq, half_width_fraction]
    sh_thresh, sh_smooth, sh_hw = params.despike_sh
    if l1.vib.shape[0] > 0:
        a_thresh, a_smooth, a_hw = params.despike_A

    # Despike bookkeeping for ATOMIX QC flags: which samples were
    # replaced (drives the per-window DESPIKE_FRACTION) and the pass
    # count per (channel, section) — the benchmark PASS_COUNT_SH layout
    # (drives the too-many-passes flag).
    section_ids = np.asarray(
        [s for s in np.unique(section_number) if s != 0], dtype=np.int64
    )
    n_sections = len(section_ids)
    despike_mask_sh = np.zeros_like(shear_out, dtype=bool)
    despike_passes_sh = np.zeros((l1.n_shear, n_sections), dtype=np.int64)
    despike_mask_A = np.zeros_like(vib_out, dtype=bool)
    despike_passes_A = np.zeros((l1.n_vib, n_sections), dtype=np.int64)

    # Despike within each section
    for si, sec_id in enumerate(section_ids):
        mask = section_number == sec_id

        # Despike shear probes
        if np.isfinite(sh_thresh):
            for i in range(l1.n_shear):
                N_sh = round(sh_hw * fs) if sh_hw < 1 else int(sh_hw)
                before = shear_out[i, mask]
                cleaned, _, n_passes, _ = despike(
                    before,
                    fs,
                    thresh=sh_thresh,
                    smooth=sh_smooth,
                    N=N_sh,
                )
                # NaN-safe change mask: NaN != NaN is True, so a raw `cleaned
                # != before` counts every unchanged NaN gap as "despiked",
                # inflating DESPIKE_FRACTION_SH and wrongly tripping QC bit 2 on
                # gappy data. Mirrors the guard in despike.py.
                despike_mask_sh[i, mask] = (cleaned != before) & ~(
                    np.isnan(cleaned) & np.isnan(before)
                )
                despike_passes_sh[i, si] = n_passes
                shear_out[i, mask] = cleaned

        # Despike vibration/accelerometer
        if l1.vib.shape[0] > 0 and np.isfinite(a_thresh):
            for i in range(l1.n_vib):
                N_a = round(a_hw * fs) if a_hw < 1 else int(a_hw)
                before = vib_out[i, mask]
                cleaned, _, n_passes, _ = despike(
                    before,
                    fs,
                    thresh=a_thresh,
                    smooth=a_smooth,
                    N=N_a,
                )
                despike_mask_A[i, mask] = (cleaned != before) & ~(
                    np.isnan(cleaned) & np.isnan(before)
                )
                despike_passes_A[i, si] = n_passes
                vib_out[i, mask] = cleaned

    return L2Data(
        time=l1.time.copy(),
        shear=shear_out,
        vib=vib_out,
        vib_type=l1.vib_type,
        pspd_rel=pspd,
        section_number=section_number,
        despike_mask_sh=despike_mask_sh,
        despike_passes_sh=despike_passes_sh,
        despike_mask_A=despike_mask_A,
        despike_passes_A=despike_passes_A,
        despike_section_ids=section_ids,
    )


# ---------------------------------------------------------------------------
# Speed from pressure
# ---------------------------------------------------------------------------


def _compute_speed(
    pres: np.ndarray,
    fs: float,
    lp_cut: float | None,
    direction: str,
) -> np.ndarray:
    """Compute profiling speed W from rate-of-change of pressure.

    Per Sec. 3.1 of the paper: W = |dP/dt| converted to m/s via the
    hydrostatic approximation (1 dbar ≈ 1 m). When *lp_cut* is given the
    result is smoothed with a low-pass Butterworth filter (forwards and
    backwards); ``None`` returns the raw gradient (the caller applies
    the single ODAS 0.68/tau smoothing pass itself).
    """
    # First-difference estimate of dP/dt
    dpdt = np.gradient(pres, 1.0 / fs)

    if lp_cut is not None:
        # Low-pass smooth to remove noise
        b, a = butter(1, lp_cut / (fs / 2))
        dpdt = filtfilt(b, a, dpdt, padlen=_matlab_padlen(b, a))

    # Sign convention: downward profiler → positive dP/dt is positive speed
    speed = dpdt.copy() if direction == "down" else -dpdt.copy()
    return speed


# ---------------------------------------------------------------------------
# Section selection
# ---------------------------------------------------------------------------


def _select_sections(
    speed: np.ndarray,
    pres: np.ndarray,
    fs: float,
    min_speed: float,
    min_pressure: float,
    min_duration: float,
    direction: str,
) -> np.ndarray:
    """Identify usable sections of the time series.

    Per Sec. 3.2.1: sections must satisfy minimum speed, depth, and duration.

    For horizontal profilers, pressure criteria check that the instrument
    is submerged but speed may not be derivable from pressure.
    """
    n = len(speed)
    section_number = np.zeros(n, dtype=np.float64)
    if n == 0:
        # Empty input: good[0]/good[-1] below would raise IndexError (#19).
        return section_number

    # Basic criteria: speed above threshold and pressure above minimum
    speed_ok = np.abs(speed) >= min_speed if direction == "horizontal" else speed >= min_speed
    good = speed_ok & (pres >= min_pressure)

    # Find contiguous runs of good data
    diff = np.diff(good.astype(np.int8))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1

    # Handle edge cases
    if good[0]:
        starts = np.concatenate([[0], starts])
    if good[-1]:
        ends = np.concatenate([ends, [n]])

    sec_id = 0
    for s, e in zip(starts, ends):
        duration = (e - s) / fs
        if duration >= min_duration:
            sec_id += 1
            section_number[s:e] = sec_id

    return section_number


# ---------------------------------------------------------------------------
# NaN-safe filtering
# ---------------------------------------------------------------------------


def _filtfilt_nan(
    b: np.ndarray,
    a: np.ndarray,
    x: np.ndarray,
) -> np.ndarray:
    """Apply ``filtfilt`` to a 1-D array, interpolating over NaN.

    NaN positions are linearly interpolated before filtering, then
    restored to NaN afterward.  If all values are NaN a copy is returned.

    Uses MATLAB-compatible ``padlen`` (see :func:`_matlab_padlen`).
    """
    padlen = _matlab_padlen(b, a)
    nans = np.isnan(x)
    if not nans.any():
        return filtfilt(b, a, x, padlen=padlen)
    if nans.all():
        return x.copy()
    filled = x.copy()
    good = np.where(~nans)[0]
    filled[nans] = np.interp(np.where(nans)[0], good, x[good])
    result = filtfilt(b, a, filled, padlen=padlen)
    result[nans] = np.nan
    return result


def _hp_filter(
    data: np.ndarray,
    fs: float,
    f_hp: float,
) -> np.ndarray:
    """First-order Butterworth HP filter, applied forwards and backwards.

    NaN values are linearly interpolated before filtering, then restored
    to NaN afterward.  This prevents ``filtfilt`` from propagating NaN
    to the entire output while preserving whole-record filter behaviour
    for files without NaN.
    """
    if data.size == 0 or f_hp <= 0:
        return data.copy()

    b, a = butter(1, f_hp / (fs / 2), btype="high")

    if data.ndim == 1:
        return _filtfilt_nan(b, a, data)

    out = np.empty_like(data)
    for i in range(data.shape[0]):
        out[i] = _filtfilt_nan(b, a, data[i])
    return out

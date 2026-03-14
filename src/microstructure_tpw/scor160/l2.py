"""L1 → L2: section selection, despiking, high-pass filtering, speed.

Implements Sec. 3.2 of Lueck et al. (2024).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt

from microstructure_tpw.scor160.despike import despike
from microstructure_tpw.scor160.io import L1Data, L2Data, L2Params

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def process_l2(l1: L1Data, params: L2Params) -> L2Data:
    """Process L1 data into L2 following ATOMIX best practices.

    Steps (per Sec. 3.2):
      1. Compute profiling speed W from dP/dt (or use pre-computed if available).
      2. Select sections satisfying min speed, depth, duration criteria.
      3. De-spike shear (and vibration) within selected sections.
      4. High-pass filter shear and vibration.

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
        pspd = _compute_speed(l1.pres, fs, params.HP_cut, l1.profile_dir)

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

    # 3 & 4. De-spike and high-pass filter shear and vibration
    shear_out = np.copy(l1.shear)
    vib_out = np.copy(l1.vib)

    # Despike parameters: [threshold, smooth_freq, half_width_fraction]
    sh_thresh, sh_smooth, sh_hw = params.despike_sh
    if l1.vib.shape[0] > 0:
        a_thresh, a_smooth, a_hw = params.despike_A

    # Despike within each section
    for sec_id in np.unique(section_number):
        if sec_id == 0:
            continue
        mask = section_number == sec_id

        # Despike shear probes
        if np.isfinite(sh_thresh):
            for i in range(l1.n_shear):
                N_sh = round(sh_hw * fs) if sh_hw < 1 else int(sh_hw)
                shear_out[i, mask], _, _, _ = despike(
                    l1.shear[i, mask],
                    fs,
                    thresh=sh_thresh,
                    smooth=sh_smooth,
                    N=N_sh,
                )

        # Despike vibration/accelerometer
        if l1.vib.shape[0] > 0 and np.isfinite(a_thresh):
            for i in range(l1.n_vib):
                N_a = round(a_hw * fs) if a_hw < 1 else int(a_hw)
                vib_out[i, mask], _, _, _ = despike(
                    l1.vib[i, mask],
                    fs,
                    thresh=a_thresh,
                    smooth=a_smooth,
                    N=N_a,
                )

    # HP-filter entire record, interpolating over NaN to avoid propagation
    shear_out = _hp_filter(shear_out, fs, params.HP_cut)
    vib_out = _hp_filter(vib_out, fs, params.HP_cut)

    return L2Data(
        time=l1.time.copy(),
        shear=shear_out,
        vib=vib_out,
        vib_type=l1.vib_type,
        pspd_rel=pspd,
        section_number=section_number,
    )


# ---------------------------------------------------------------------------
# Speed from pressure
# ---------------------------------------------------------------------------

def _compute_speed(
    pres: np.ndarray,
    fs: float,
    lp_cut: float,
    direction: str,
) -> np.ndarray:
    """Compute profiling speed W from rate-of-change of pressure.

    Per Sec. 3.1 of the paper: W = |dP/dt| converted to m/s via the
    hydrostatic approximation (1 dbar ≈ 1 m). The result is smoothed
    with a low-pass Butterworth filter (forwards and backwards).
    """
    # First-difference estimate of dP/dt
    dpdt = np.gradient(pres, 1.0 / fs)

    # Low-pass smooth to remove noise
    b, a = butter(1, lp_cut / (fs / 2))
    dpdt_smooth = filtfilt(b, a, dpdt)

    # Sign convention: downward profiler → positive dP/dt is positive speed
    speed = dpdt_smooth.copy() if direction == "down" else -dpdt_smooth.copy()
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
    """
    nans = np.isnan(x)
    if not nans.any():
        return filtfilt(b, a, x)
    if nans.all():
        return x.copy()
    filled = x.copy()
    good = np.where(~nans)[0]
    filled[nans] = np.interp(np.where(nans)[0], good, x[good])
    result = filtfilt(b, a, filled)
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

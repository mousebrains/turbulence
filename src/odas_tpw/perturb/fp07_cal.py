# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""FP07 in-situ calibration against a reference (slow) thermistor.

Steps:
  1. Convert FP07 to resistance ratio ln(R_T/R_0)
  2. Low-pass filter FP07 to match reference bandwidth
  3. Per-profile cross-correlate diff(filtered_FP07) vs diff(reference) -> lag
  4. Median lag across profiles
  5. Steinhart-Hart polynomial fit: 1/(T+273.15) = a0 + a1*ln(R) + ... + aN*ln(R)^order
  6. Apply calibration to slow and fast FP07 data

Reference: Code/fp07_calibration.m (354 lines)
"""

import re
import warnings

import numpy as np
from scipy.signal import butter, correlate, lfilter

from odas_tpw.rsi.p_file import PFile


def _find_fp07_channels(pf: PFile) -> list[str]:
    """Find FP07 thermistor channel names (T1, T2, ...)."""
    pattern = re.compile(r"^T\d+$")
    return sorted(
        name
        for name in pf.channels
        if pattern.match(name)
        and pf.channel_info.get(name, {}).get("type") in ("therm", "thermistor")
    )


def _get_channel_config(pf: PFile, ch_name: str) -> dict:
    """Extract calibration parameters for a channel from PFile config."""
    for ch in pf.config["channels"]:
        if ch.get("name", "").strip() == ch_name:
            return dict(ch)
    return {}


def _compute_RT_R0(
    counts: np.ndarray,
    ch_config: dict,
) -> np.ndarray:
    """Compute ln(R_T/R_0) from raw counts or physical values.

    For therm type: Z = factor * (counts - a) / b where factor = (adc_fs/2^adc_bits) * 2/(G*E_B)
    For other types: Z = (counts * factor + adc_zero - a) / b * 2/(G*E_B)
    RT_R0 = ln((1-Z)/(1+Z))
    """
    E_B = float(ch_config.get("e_b", 0))
    a = float(ch_config.get("a", 0))
    b = float(ch_config.get("b", 1))
    G = float(ch_config.get("g", 1))
    adc_fs = float(ch_config.get("adc_fs", 5))
    adc_bits = int(float(ch_config.get("adc_bits", 16)))
    adc_zero = float(ch_config.get("adc_zero", 0))
    ch_type = ch_config.get("type", "therm").strip().lower()

    factor = adc_fs / (2**adc_bits)

    if ch_type in ("therm", "thermistor"):
        factor = factor * 2.0 / (G * E_B) if (G * E_B) != 0 else factor
        Z = factor * (counts - a) / b if b != 0 else np.zeros_like(counts)
    else:
        Z = counts * factor + adc_zero
        Z = ((Z - a) / b) * 2.0 / (G * E_B) if (b * G * E_B) != 0 else np.zeros_like(counts)

    # Clip to avoid log domain errors
    ratio = (1.0 - Z) / (1.0 + Z)
    ratio = np.clip(ratio, 1e-20, None)
    return np.asarray(np.log(ratio))


def _lowpass_filter(
    fp07: np.ndarray,
    reference: str,
    fs: float,
    W: np.ndarray,
    profiles: list[tuple[int, int]],
) -> np.ndarray:
    """Low-pass filter FP07 to match reference sensor bandwidth.

    For JAC_T: fc = 0.73 * sqrt(mean_speed / 0.62)
    Otherwise: fc = fs/3
    """
    if reference.upper().startswith("JAC"):
        # Compute mean profiling speed across all profiles
        count = 0
        W_sum = 0.0
        for s, e in profiles:
            count += e - s + 1
            W_sum += np.sum(np.abs(W[s : e + 1]))
        W_mean = W_sum / count if count > 0 else 0.3
        fc = 0.73 * np.sqrt(W_mean / 0.62)
    else:
        fc = fs / 3.0

    # Ensure fc is below Nyquist
    fc = min(fc, fs / 2.0 * 0.99)
    b, a = butter(1, fc / (fs / 2.0))
    return np.asarray(lfilter(b, a, fp07))


def _calc_lag(
    T_ref: np.ndarray,
    T_fp07: np.ndarray,
    fs: float,
    max_lag_seconds: float = 10.0,
    must_be_negative: bool = True,
) -> tuple[float, float]:
    """Cross-correlate diff(T_ref) vs diff(T_fp07) to find lag.

    Returns (lag_seconds, max_correlation).
    """
    max_lag_samples = round(max_lag_seconds * fs)

    # 4 Hz Butterworth to suppress high-freq noise
    fc = min(4.0, fs / 2.0 * 0.99)
    bb, aa = butter(2, fc / (fs / 2.0))

    dx = lfilter(bb, aa, np.diff(T_fp07) - np.mean(np.diff(T_fp07)))
    dy = lfilter(bb, aa, np.diff(T_ref) - np.mean(np.diff(T_ref)))

    # Full cross-correlation
    corr = correlate(dx, dy, mode="full")
    # Normalize
    norm = np.sqrt(np.sum(dx**2) * np.sum(dy**2))
    if norm > 0:
        corr = corr / norm

    n = len(dx)
    lags = np.arange(-(n - 1), n)

    # Restrict to max_lag range
    mask = np.abs(lags) <= max_lag_samples
    corr = corr[mask]
    lags = lags[mask]

    if must_be_negative:
        neg_mask = lags <= 0
        corr_search = corr[neg_mask]
        lags_search = lags[neg_mask]
        if len(corr_search) == 0:
            return 0.0, 0.0
        idx = np.argmax(corr_search)
        return lags_search[idx] / fs, corr_search[idx]
    else:
        idx = np.argmax(corr)
        return lags[idx] / fs, corr[idx]


def fp07_calibrate(
    pf: PFile,
    profiles: list[tuple[int, int]],
    reference: str = "JAC_T",
    order: int = 2,
    max_lag_seconds: float = 10.0,
    must_be_negative: bool = True,
) -> dict:
    """Perform in-situ FP07 calibration.

    Parameters
    ----------
    pf : PFile
        Parsed .p file.
    profiles : list of (start, end) tuples
        Profile indices into slow-rate arrays.
    reference : str
        Reference temperature channel name (e.g. "JAC_T").
    order : int
        Steinhart-Hart polynomial order (1-3).
    max_lag_seconds : float
        Maximum cross-correlation lag [s].
    must_be_negative : bool
        If True, only search negative lags (FP07 leads reference).

    Returns
    -------
    dict with keys:
        channels : dict mapping channel name -> calibrated slow-rate array
        fast_channels : dict mapping channel name -> calibrated fast-rate array
        lags : dict mapping channel name -> lag in seconds
        coefficients : dict mapping channel name -> Steinhart-Hart coefficients
        info : dict of per-channel calibration stats
    """
    if reference not in pf.channels:
        warnings.warn(f"Reference channel {reference!r} not found")
        return {"channels": {}, "fast_channels": {}, "lags": {}, "coefficients": {}, "info": {}}

    T_ref = pf.channels[reference]
    fp07_names = _find_fp07_channels(pf)
    if not fp07_names:
        warnings.warn("No FP07 channels found")
        return {"channels": {}, "fast_channels": {}, "lags": {}, "coefficients": {}, "info": {}}

    # Get fall rate for low-pass filter cutoff
    from odas_tpw.rsi.profile import _smooth_fall_rate

    P = pf.channels.get("P")
    W = _smooth_fall_rate(P, pf.fs_slow) if P is not None else np.full(len(T_ref), 0.5)

    result: dict[str, dict] = {
        "channels": {},
        "fast_channels": {},
        "lags": {},
        "coefficients": {},
        "info": {},
    }

    for ch_name in fp07_names:
        ch_config = _get_channel_config(pf, ch_name)
        if not ch_config:
            continue

        # Get raw counts (channels_raw has pre-conversion data)
        if ch_name not in pf.channels_raw:
            continue

        raw_slow = pf.channels_raw[ch_name]
        # If fast rate, subsample for slow
        if pf.is_fast(ch_name):
            ratio = round(pf.fs_fast / pf.fs_slow)
            raw_slow = raw_slow[::ratio][: len(T_ref)]

        # Compute RT_R0
        RT_R0_slow = _compute_RT_R0(raw_slow, ch_config)

        # Low-pass filter for lag finding
        fp07_lp = _lowpass_filter(raw_slow, reference, pf.fs_slow, W, profiles)

        # Per-profile lag computation
        lags_list = []
        corrs_list = []
        for s, e in profiles:
            if e - s < 10:
                continue
            lag, corr = _calc_lag(
                T_ref[s : e + 1],
                fp07_lp[s : e + 1],
                pf.fs_slow,
                max_lag_seconds=max_lag_seconds,
                must_be_negative=must_be_negative,
            )
            lags_list.append(lag)
            corrs_list.append(corr)

        if not lags_list:
            continue

        median_lag = float(np.median(lags_list))
        i_shift = round(median_lag * pf.fs_slow)
        result["lags"][ch_name] = median_lag

        # Shift reference to align with FP07
        T_ref_shifted = np.roll(T_ref, i_shift)

        # Collect profile data for Steinhart-Hart fit
        all_RT_R0 = []
        all_T_ref = []
        for s, e in profiles:
            all_RT_R0.append(RT_R0_slow[s : e + 1])
            all_T_ref.append(T_ref_shifted[s : e + 1])

        RT_R0_fit = np.concatenate(all_RT_R0)
        T_ref_fit = np.concatenate(all_T_ref)

        # Remove NaN/inf
        valid = np.isfinite(RT_R0_fit) & np.isfinite(T_ref_fit)
        RT_R0_fit = RT_R0_fit[valid]
        T_ref_fit = T_ref_fit[valid]

        if len(RT_R0_fit) < order + 1:
            continue

        # Steinhart-Hart: 1/(T+273.15) = a0 + a1*RT_R0 + a2*RT_R0^2 + ...
        target = 1.0 / (T_ref_fit + 273.15)
        X = np.column_stack([RT_R0_fit**i for i in range(order + 1)])
        coeffs, _, _, _ = np.linalg.lstsq(X, target, rcond=None)
        result["coefficients"][ch_name] = coeffs

        # Apply calibration to slow data
        X_all = np.column_stack([RT_R0_slow**i for i in range(order + 1)])
        T_cal_slow = 1.0 / (X_all @ coeffs) - 273.15
        result["channels"][ch_name] = T_cal_slow

        # Apply calibration to fast data
        if ch_name in pf.channels_raw and pf.is_fast(ch_name):
            raw_fast = pf.channels_raw[ch_name]
            RT_R0_fast = _compute_RT_R0(raw_fast[: len(pf.t_fast)], ch_config)
            X_fast = np.column_stack([RT_R0_fast**i for i in range(order + 1)])
            T_cal_fast = 1.0 / (X_fast @ coeffs) - 273.15
            result["fast_channels"][ch_name] = T_cal_fast

        result["info"][ch_name] = {
            "median_lag": median_lag,
            "lag_std": float(np.std(lags_list)),
            "median_corr": float(np.median(corrs_list)),
            "n_profiles": len(lags_list),
            "coefficients": coeffs.tolist(),
        }

    return result

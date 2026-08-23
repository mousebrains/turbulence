# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""FP07 in-situ calibration against a reference (slow) thermistor, per ``.p`` file.

Steps:
  1. Convert FP07 to resistance ratio ``L = ln(R_T/R_0)``
  2. Low-pass filter the FP07 to the reference's own bandwidth
  3. Per-profile cross-correlate ``diff`` of each to find the lag
  4. Median lag across profiles, gated on correlation quality
  5. Steinhart-Hart fit ``1/(T+273.15) = a0 + a1*L + ... + aN*L^order``
  6. Apply the calibration to the slow and fast FP07 data

Reference: Code/fp07_calibration.m

Scope, and when to use something else
-------------------------------------
This is a **per-file** calibration: it needs a usable reference inside every
``.p`` file it is asked to calibrate. That is fine for a VMP with a hull-mounted
JAC sampling continuously alongside the FP07.

It is the wrong tool when the reference is **sparse** --- a glider whose CTD ran
on only some yos, say. Files with no reference silently keep the factory
coefficients while their neighbours get fitted ones, so adjacent profiles end up
on different calibrations. For that case use the deployment-scoped
``fp07-cal`` CLI (:mod:`odas_tpw.fp07cal`), which pools every real reference
sample across the whole deployment, fits once, and patches the result into the
``.p`` files before the pipeline runs.

The bridge algebra here is deliberately **not** reimplemented: it is imported
from :mod:`odas_tpw.fp07cal.logr`, which is the same code the reader
(``rsi/channels.py::convert_therm``) is verified against. Two copies of
``ln(R_T/R_0)`` that disagree on a default is exactly how this module used to
emit coefficients that did not reproduce.
"""

import re
import warnings

import numpy as np
from scipy.signal import butter, correlate, lfilter

from odas_tpw.fp07cal.logr import BridgeParams, log_r
from odas_tpw.processing.ct_align import shift_edge_hold
from odas_tpw.rsi.p_file import PFile

#: Below this |Pearson r| a lag is not established and the profile is dropped.
#: The old code computed a correlation, stored it, and never compared it to
#: anything --- so a reference fabricated by interpolation across a gap yielded
#: a confident-looking lag at r = 0.02 and was folded into the median.
MIN_LAG_CORR = 0.5

#: A fit needs at least this many profiles whose lag passed the gate.
MIN_LAG_PROFILES = 1

#: Refuse if more than this fraction of fit samples hit the bridge rail. A
#: clipped sample carries a wrong L and is indistinguishable downstream.
MAX_CLIPPED_FRACTION = 0.05


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


def _reference_interval(T_ref: np.ndarray, fs: float) -> float:
    """Effective sample interval [s] of a reference that has been interpolated up.

    A 1 Hz CTD merged **linearly** onto a 64 Hz grid is piecewise-linear with
    knots at its *original* samples, so the second difference is impulsive
    there and ~0 between. Counting the knots recovers the true interval.

    This is a **fallback** only: a pchip or nearest merge leaves no such
    structure, so the pipeline passes the reference's native interval
    explicitly (``reference_interval``, recorded by the hotel merge). The
    second difference is taken per contiguous finite run so a NaN hole does
    not splice two unrelated samples into one giant jump, and the threshold
    is a multiple of the median non-zero ``d2`` rather than a fraction of the
    max, so a single spike cannot swamp the real knots.

    Returns ``1/fs`` when no such structure is found (an already-fast
    reference), which reduces to "do not filter".
    """
    x = np.asarray(T_ref, dtype=np.float64)
    good = np.isfinite(x)
    if good.sum() < 8:
        return 1.0 / fs
    # Second difference per contiguous finite run, kept on the original index
    # so knot spacing is measured in grid samples.
    d2 = np.zeros(x.size)
    edges = np.flatnonzero(np.diff(good.astype(np.int8)))
    starts = np.concatenate(([0], edges + 1))
    stops = np.concatenate((edges + 1, [x.size]))
    for s0, s1 in zip(starts, stops, strict=True):
        if good[s0] and s1 - s0 >= 3:
            d2[s0 + 1 : s1 - 1] = np.abs(np.diff(x[s0:s1], n=2))
    nz = d2[d2 > 0]
    if nz.size == 0:
        return 1.0 / fs
    # Interior of a linear segment is ~0 up to round-off, so the median of the
    # non-zero d2 is either round-off (knots tower above it) or, for a fast
    # reference, the noise level (nothing towers above it).
    threshold = 20.0 * float(np.median(nz))
    knots = np.flatnonzero(d2 > threshold)
    if knots.size < 3:
        return 1.0 / fs
    step = float(np.median(np.diff(knots)))
    if not np.isfinite(step) or step < 1.0:
        return 1.0 / fs
    return step / fs


def _lowpass_filter(
    fp07: np.ndarray,
    reference: str,
    fs: float,
    W: np.ndarray,
    profiles: list[tuple[int, int]],
    T_ref: np.ndarray | None = None,
    reference_interval: float | None = None,
) -> np.ndarray:
    """Low-pass the FP07 to the reference's bandwidth.

    For a JAC: ``fc = 0.73*sqrt(mean_speed/0.62)``, the vendor relation.

    Otherwise the cutoff is the reference's own Nyquist, ``0.5 /
    reference_interval``. The interval comes from the caller when known (the
    hotel merge records each channel's native sample spacing); otherwise it is
    inferred from the merged array by :func:`_reference_interval`, which only
    works for a linear merge. The old code used ``fs/3`` here, which for a
    1 Hz CTD on a 64 Hz grid is ~21 Hz --- no filtering at all. Leaving that
    bandwidth in the regressor is textbook errors-in-variables: it attenuates
    the fitted slope toward zero, and that slope IS ``beta_1``.

    The filter is causal (``lfilter``), so the output lags the input by the
    filter's group delay (~0.125 s at fc = 0.5 Hz). Because the reference is
    shifted against this *same* filtered series before the regression, the
    calibration is unaffected; only the reported lag carries the bias (see
    :func:`_calc_lag`).
    """
    if reference.upper().startswith("JAC"):
        count = 0
        W_sum = 0.0
        for s, e in profiles:
            count += e - s + 1
            W_sum += np.sum(np.abs(W[s : e + 1]))
        W_mean = W_sum / count if count > 0 else 0.3
        fc = 0.73 * np.sqrt(W_mean / 0.62)
    else:
        if reference_interval is not None and np.isfinite(reference_interval) \
                and reference_interval > 0:
            interval = float(reference_interval)
        elif T_ref is not None:
            interval = _reference_interval(T_ref, fs)
            if interval <= 1.0 / fs:
                warnings.warn(
                    f"{reference}: native sample interval unknown and could not "
                    f"be inferred from the merged array (a pchip/nearest merge "
                    f"leaves nothing to infer from); the FP07 is NOT low-pass "
                    f"filtered to the reference bandwidth. Set "
                    f"fp07.reference_interval or merge the reference with "
                    f"interp: linear.",
                    stacklevel=3,
                )
        else:
            interval = 1.0 / fs
        fc = 0.5 / interval
        if interval <= 1.0 / fs:
            # Nothing to match: the reference is already at the grid rate.
            return np.asarray(fp07, dtype=np.float64)

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

    Returns ``(lag_seconds, max_correlation)``. The caller must gate on the
    correlation: this returns the best lag in the searched window whether or
    not anything actually correlates.

    Known bias: ``T_fp07`` arrives already passed through the causal
    :func:`_lowpass_filter` while ``T_ref`` is not, so the probe carries the
    filter's group delay (~0.125 s at fc = 0.5 Hz) and the lag is shifted
    positive by that much; with ``must_be_negative`` a true lag in
    (-0.125, 0] clamps to 0. The regression is immune (it aligns the
    reference against the same filtered series), so only the *reported* lag
    is affected. A zero-phase ``filtfilt`` would remove it but change every
    stored lag; left as is and documented.
    """
    max_lag_samples = round(max_lag_seconds * fs)

    # 4 Hz Butterworth to suppress high-freq noise
    fc = min(4.0, fs / 2.0 * 0.99)
    bb, aa = butter(2, fc / (fs / 2.0))

    dx = lfilter(bb, aa, np.diff(T_fp07) - np.mean(np.diff(T_fp07)))
    dy = lfilter(bb, aa, np.diff(T_ref) - np.mean(np.diff(T_ref)))

    # Full cross-correlation
    corr = correlate(dx, dy, mode="full")
    # Normalize. A flatlined segment (norm == 0) or one carrying a non-finite
    # sample (norm NaN) leaves corr un-normalized/NaN; argmax(|corr|) over the
    # negative-lag window would then return index 0 -> the most-negative
    # searched lag (-max_lag_seconds), silently poisoning the median lag that
    # aligns the Steinhart-Hart fit. Mirror ct_align and abstain instead.
    norm = np.sqrt(np.sum(dx**2) * np.sum(dy**2))
    if not np.isfinite(norm) or norm <= 0:
        return np.nan, np.nan
    corr = corr / norm

    n = len(dx)
    lags = np.arange(-(n - 1), n)

    # Restrict to max_lag range
    mask = np.abs(lags) <= max_lag_samples
    corr = corr[mask]
    lags = lags[mask]

    # Use |correlation| as ODAS cal_FP07_in_situ.m does: raw counts
    # correlate negatively with temperature when the bridge coefficient
    # b < 0, and a signed search would latch onto a wrong sidelobe.
    if must_be_negative:
        neg_mask = lags <= 0
        corr_search = corr[neg_mask]
        lags_search = lags[neg_mask]
        if len(corr_search) == 0:
            return 0.0, 0.0
        idx = int(np.argmax(np.abs(corr_search)))
        return lags_search[idx] / fs, corr_search[idx]
    idx = int(np.argmax(np.abs(corr)))
    return lags[idx] / fs, corr[idx]


def _polyfit_centered(
    L: np.ndarray, target: np.ndarray, order: int
) -> tuple[np.ndarray, float]:
    """Least-squares fit in a centered/scaled variable, returned in terms of ``L``.

    ``L`` occupies a short interval far from zero, so a raw Vandermonde in it is
    badly conditioned --- over a narrow temperature range the higher-order terms
    go numerically meaningless well before they go statistically meaningless.
    Fitting in ``u = (L - mean)/std`` and composing the polynomial back is
    exact, so the emitted coefficients still reproduce in the reader.

    Returns ``(coefficients_in_L, condition_number)``.
    """
    from numpy.polynomial import Polynomial

    center = float(np.mean(L))
    scale = float(np.std(L))
    if not np.isfinite(scale) or scale == 0:
        raise ValueError("L has zero spread; nothing to fit")
    u = (L - center) / scale
    X = np.column_stack([u**i for i in range(order + 1)])
    cond = float(np.linalg.cond(X))
    coeffs_u, *_ = np.linalg.lstsq(X, target, rcond=None)
    composed = Polynomial(coeffs_u)(Polynomial([-center / scale, 1.0 / scale]))
    return np.asarray(composed.coef, dtype=np.float64), cond


def fp07_calibrate(
    pf: PFile,
    profiles: list[tuple[int, int]],
    reference: str = "JAC_T",
    order: int = 2,
    max_lag_seconds: float = 10.0,
    must_be_negative: bool = True,
    min_corr: float = MIN_LAG_CORR,
    reference_interval: float | None = None,
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
    min_corr : float
        Minimum |Pearson r| for a profile's lag to count. A profile below this
        contributes nothing --- neither to the median lag nor to the
        coefficient fit; a channel with none above it is left on the factory
        coefficients rather than fitted against noise.
    reference_interval : float, optional
        Native sample interval [s] of a non-JAC reference, used to set the
        FP07 low-pass cutoff at the reference's Nyquist. The pipeline takes
        it from ``pf.hotel_native_dt`` (recorded by the hotel merge) or the
        ``fp07.reference_interval`` config key. ``None`` falls back to
        inferring it from the merged array, which only works for a linear
        merge and warns when it cannot.

    Returns
    -------
    dict with keys:
        channels : dict mapping channel name -> calibrated array at the
            channel's native rate (e.g. T1)
        fast_channels : dict mapping pre-emphasized channel name (e.g.
            T1_dT1, holding the deconvolved fast temperature) -> array
            recalibrated with the in-situ fit.  These feed the chi
            (temperature-gradient) pipeline; without them chi would
            silently keep the factory calibration.
        lags : dict mapping channel name -> lag in seconds
        coefficients : dict mapping channel name -> Steinhart-Hart coefficients
        info : dict of per-channel calibration stats
    """
    empty: dict = {
        "channels": {}, "fast_channels": {}, "lags": {}, "coefficients": {}, "info": {}
    }
    if reference not in pf.channels:
        warnings.warn(f"Reference channel {reference!r} not found", stacklevel=2)
        return empty

    T_ref = pf.channels[reference]
    fp07_names = _find_fp07_channels(pf)
    if not fp07_names:
        warnings.warn("No FP07 channels found", stacklevel=2)
        return empty

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
        if ch_name not in pf.channels_raw:
            continue

        # One shared implementation of L, and a hard error on a missing bridge
        # parameter rather than a substituted default: the fit's L must equal
        # the reader's L or the emitted coefficients do not reproduce.
        try:
            bridge = BridgeParams.from_channel_config(ch_config, ch_name)
        except ValueError as exc:
            warnings.warn(f"{ch_name}: {exc}", stacklevel=2)
            continue

        raw_slow = pf.channels_raw[ch_name]
        # If fast rate, subsample for slow
        if pf.is_fast(ch_name):
            ratio = round(pf.fs_fast / pf.fs_slow)
            raw_slow = raw_slow[::ratio][: len(T_ref)]

        # Low-pass to the reference's own bandwidth. Used both for lag finding
        # and for the Steinhart-Hart regression: ODAS cal_FP07_in_situ.m fits
        # on the filtered thermistor, since unfiltered high-frequency noise in
        # the regressor (errors-in-variables) attenuates the fitted slope.
        fp07_lp = _lowpass_filter(
            raw_slow, reference, pf.fs_slow, W, profiles, T_ref=T_ref,
            reference_interval=reference_interval,
        )
        L_fit_src, clipped = log_r(fp07_lp, bridge)

        # Per-profile lag computation. The profiles that pass are kept, because
        # they are also the only ones the regression may use: a profile whose
        # reference does not track the probe is not evidence about the
        # calibration, and letting it into the fit while excluding it from the
        # median lag would be the worst of both.
        lags_list = []
        corrs_list = []
        accepted: list[tuple[int, int]] = []
        n_rejected = 0
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
            # _calc_lag returns NaN for a flatlined / non-finite segment; drop
            # it so it cannot poison the median.
            if not np.isfinite(lag) or not np.isfinite(corr):
                continue
            # ...and drop a profile whose "best" lag correlates with nothing.
            # Without this a fabricated reference is fitted at r ~ 0.02.
            if abs(corr) < min_corr:
                n_rejected += 1
                continue
            lags_list.append(lag)
            corrs_list.append(corr)
            accepted.append((s, e))

        if len(lags_list) < MIN_LAG_PROFILES:
            warnings.warn(
                f"{ch_name}: no profile reached the lag correlation gate "
                f"(|r| >= {min_corr:g}; {n_rejected} rejected). Leaving the "
                f"factory calibration in place rather than fitting against a "
                f"reference that does not track it.",
                stacklevel=2,
            )
            continue

        median_lag = float(np.median(lags_list))
        i_shift = round(median_lag * pf.fs_slow)
        result["lags"][ch_name] = median_lag

        # Shift reference to align with FP07.  Edge-hold instead of
        # np.roll: wrapping would splice the start of the record into the
        # tail of the last profile (ODAS trims instead).
        T_ref_shifted = shift_edge_hold(T_ref, i_shift)

        # Collect profile data for the Steinhart-Hart fit -- from the ACCEPTED
        # profiles only. Iterating `profiles` here would let every profile the
        # correlation gate just rejected bias the coefficients (and the rail
        # fraction) anyway, so a single good profile would launder all the bad
        # ones. "Contributes nothing" has to mean nothing.
        all_L = []
        all_T_ref = []
        all_clip = []
        for s, e in accepted:
            all_L.append(L_fit_src[s : e + 1])
            all_T_ref.append(T_ref_shifted[s : e + 1])
            all_clip.append(clipped[s : e + 1])

        L_fit = np.concatenate(all_L)
        T_ref_fit = np.concatenate(all_T_ref)
        clip_fit = np.concatenate(all_clip)

        # A clipped sample sat on the bridge rail: its L is wrong and looks
        # perfectly ordinary. Exclude rather than regress it.
        n_clipped = int(np.count_nonzero(clip_fit))
        valid = np.isfinite(L_fit) & np.isfinite(T_ref_fit) & ~clip_fit
        if L_fit.size and n_clipped / L_fit.size > MAX_CLIPPED_FRACTION:
            warnings.warn(
                f"{ch_name}: {100.0 * n_clipped / L_fit.size:.1f}% of fit samples "
                f"hit the bridge rail (|Z| >= 0.6); the channel is outside its "
                f"range and the calibration is not trustworthy. Skipping.",
                stacklevel=2,
            )
            continue
        L_fit = L_fit[valid]
        T_ref_fit = T_ref_fit[valid]

        if len(L_fit) < order + 1:
            continue

        # ODAS warns when the in-situ temperature range is small: a
        # higher-order polynomial fit over a narrow range is poorly
        # constrained and extrapolates badly.
        T_range = float(np.max(T_ref_fit) - np.min(T_ref_fit))
        if order > 1 and T_range < 8.0:
            warnings.warn(
                f"{ch_name}: in-situ temperature range {T_range:.1f} degC < 8 degC; "
                f"order-{order} fit may be poorly constrained (consider order=1)",
                stacklevel=2,
            )

        # Steinhart-Hart: 1/(T+273.15) = a0 + a1*L + a2*L^2 + ...
        target = 1.0 / (T_ref_fit + 273.15)
        try:
            coeffs, cond = _polyfit_centered(L_fit, target, order)
        except ValueError as exc:
            warnings.warn(f"{ch_name}: {exc}", stacklevel=2)
            continue
        result["coefficients"][ch_name] = coeffs

        def _apply_cal(raw: np.ndarray, _bridge=bridge, _coeffs=coeffs) -> np.ndarray:
            rt, _ = log_r(raw, _bridge)
            Xm = np.column_stack([rt**i for i in range(len(_coeffs))])
            return np.asarray(1.0 / (Xm @ _coeffs) - 273.15)

        # Apply calibration to the channel at its NATIVE rate (the
        # previous code always stored the slow-subsampled array, which
        # mis-sized fast channels when assigned back to pf.channels).
        if pf.is_fast(ch_name):
            result["channels"][ch_name] = _apply_cal(
                pf.channels_raw[ch_name][: len(pf.t_fast)]
            )
        else:
            result["channels"][ch_name] = _apply_cal(raw_slow)

        # Recalibrate the deconvolved pre-emphasized variant (T1 ->
        # T1_dT1).  After PFile._apply_deconvolution, channels_raw holds
        # the DECONVOLVED counts for this channel, converted downstream
        # with the base channel's calibration — so applying the in-situ
        # polynomial here propagates the calibration into the fast
        # temperature used by the chi pipeline.
        dx_name = f"{ch_name}_d{ch_name}"
        if dx_name in pf.channels_raw:
            n_dx = len(pf.t_fast) if pf.is_fast(dx_name) else len(pf.t_slow)
            result["fast_channels"][dx_name] = _apply_cal(pf.channels_raw[dx_name][:n_dx])

        result["info"][ch_name] = {
            "median_lag": median_lag,
            "lag_std": float(np.std(lags_list)),
            "median_corr": float(np.median(corrs_list)),
            "n_profiles": len(lags_list),
            "n_profiles_rejected": n_rejected,
            "n_clipped": n_clipped,
            "condition_number": cond,
            "T_range": T_range,
            "coefficients": coeffs.tolist(),
        }

    return result

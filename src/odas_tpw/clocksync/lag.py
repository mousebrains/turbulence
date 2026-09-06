# Aug-2026, Claude and Pat Welch, pat@mousebrains.com
"""Lag between two pressure records, to a small fraction of a sample.

Sign convention, fixed by test and never inferred
-------------------------------------------------
``lag`` is **the number of seconds to ADD to the target's timestamps to line
them up with the reference**.  A positive lag means the target clock is slow.
``tests/test_clocksync.py`` injects a known shift and asserts the recovered
value, because a sign error here is invisible in every summary statistic and
fatal in the result.

Why not xcorr-and-argmax
------------------------
That is what the MATLAB did, and it caps out at one sample --- 62.5 ms at
16 Hz --- with three failure modes underneath:

1. **The tide is a ramp.**  With only the mean removed, a 15-minute window is
   dominated by a near-linear trend, and shifting a straight line gives back
   the same line plus a constant.  ``fp07cal/lag.py`` measured this on real
   data: raw pressure scored r = 1.000000 at *every* lag over +/-30 s.  The
   timing lives in the curvature, so everything here is band-passed first and
   gated on peak SHARPNESS, never on r.
2. **A swell is narrowband, so the correlation has side lobes** every wave
   period.  Picking the largest of 36 near-equal lobes is a coin flip.  The
   analytic envelope has no carrier, so its peak identifies the right lobe.
3. **A one-sample answer has no error bar.**  The scatter of segment lags then
   has to stand in for the uncertainty, which conflates real clock drift with
   estimator noise.

The estimator
-------------
1. band-pass both series, zero-phase (a causal filter would contribute its own
   group delay to the very quantity being measured);
2. cross-correlate by FFT, take ``|hilbert(r)|``, and pick the envelope
   maximum --- unambiguous to within a fraction of the envelope width;
3. refine by cross-spectral phase: for a pure delay the cross-spectrum has
   phase ``-2*pi*f*tau``, so a coherence-weighted straight line through
   ``phi(f)`` gives tau *and* a formal sigma.  Applied iteratively with a
   frequency-domain shift, so the residual stays far inside +/-pi and phase
   wrapping never arises.

Step 3 is where the precision comes from: with a 0.3 Hz band over 15 minutes
the phase slope is determined orders of magnitude better than the 1/16 s
sample spacing, and the coherence gives an honest error bar for the weighted
fit in :mod:`odas_tpw.clocksync.fit`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.fft import irfft, next_fast_len, rfft, rfftfreq
from scipy.signal import butter, coherence, csd, hilbert, sosfiltfilt, welch


@dataclass
class LagResult:
    """A lag with the evidence for it.

    ``coherence`` is the quality statistic worth trusting; the correlation
    value deliberately is not one of the gates (see the module docstring).
    """

    lag: float = np.nan
    sigma: float = np.nan
    coherence: float = np.nan
    dynamic_range: float = np.nan
    envelope_width: float = np.nan
    phase_chi2: float = np.nan
    amplitude: float = np.nan
    n: int = 0
    t_mid: float = np.nan
    ok: bool = False
    why: str = ""
    notes: list[str] = field(default_factory=list)


def bandpass(x: np.ndarray, fs: float, f_lo: float, f_hi: float, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth band-pass.

    ``sosfiltfilt`` not ``sosfilt``: a causal filter delays the signal by its
    own group delay, which is exactly the quantity under measurement.  Applying
    the same causal filter to both series would cancel only if both sat on
    identical grids with identical gaps, which is not guaranteed here.
    """
    nyq = 0.5 * fs
    lo = max(f_lo / nyq, 1e-6)
    hi = min(f_hi / nyq, 0.999999)
    if not lo < hi:
        raise ValueError(f"band {f_lo}-{f_hi} Hz is empty at fs = {fs} Hz")
    sos = butter(order, [lo, hi], btype="bandpass", output="sos")
    # padlen must fit: sosfiltfilt raises on short segments otherwise.
    padlen = min(3 * (2 * order + 1), x.size - 1)
    return np.asarray(sosfiltfilt(sos, x, padlen=padlen), dtype=np.float64)


def shift(x: np.ndarray, fs: float, tau: float) -> np.ndarray:
    """Delay *x* by *tau* seconds in the frequency domain.

    Sub-sample shifting by phase rotation rather than interpolation: a linear
    or spline resample would low-pass the signal by an amount that depends on
    the fractional part of the shift, biasing the next iteration's estimate.
    """
    n = x.size
    nfft = next_fast_len(n)
    X = rfft(x, nfft)
    f = rfftfreq(nfft, 1.0 / fs)
    return np.asarray(irfft(X * np.exp(-2j * np.pi * f * tau), nfft)[:n], dtype=np.float64)


def _envelope_peak(a: np.ndarray, b: np.ndarray, fs: float, max_lag: float):
    """Coarse lag from the peak of the cross-correlation *envelope*.

    Returns (lag_seconds, dynamic_range, width_seconds, at_edge).
    """
    n = min(a.size, b.size)
    a, b = a[:n], b[:n]
    nfft = next_fast_len(2 * n)
    r = irfft(rfft(a, nfft) * np.conj(rfft(b, nfft)), nfft)
    k = round(max_lag * fs)
    k = min(k, n - 1)
    # r[j] correlates a[i+j] with b[i]; positive j means b must move LATER.
    r = np.concatenate([r[-k:], r[: k + 1]])
    lags = np.arange(-k, k + 1, dtype=np.float64) / fs
    env = np.abs(hilbert(r))
    i = int(np.argmax(env))
    peak = env[i]
    med = float(np.median(env))
    dyn = peak / med if med > 0 else np.inf
    half = np.where(env >= 0.5 * peak)[0]
    width = (half[-1] - half[0] + 1) / fs if half.size else np.nan
    at_edge = i < 0.02 * env.size or i > 0.98 * env.size
    return float(lags[i]), float(dyn), float(width), bool(at_edge)


def _phase_slope(a: np.ndarray, b: np.ndarray, fs: float, band, nperseg: int, coh_min: float):
    """Delay from the slope of the cross-spectral phase, with its sigma.

    For b(t) = a(t - tau) the cross-spectrum S_ab(f) carries phase
    ``+2*pi*f*tau``; the returned value follows the package convention (add it
    to the target's clock).  The variance of a phase estimate at coherence
    ``g2`` over ``nd`` independent segments is ``(1 - g2) / (2 * nd * g2)``,
    which is what weights the fit and what makes ``sigma`` mean something.
    """
    nperseg = int(min(nperseg, a.size))
    f, Pab = csd(a, b, fs=fs, nperseg=nperseg, detrend="linear")
    _, g2 = coherence(a, b, fs=fs, nperseg=nperseg, detrend="linear")
    sel = (f >= band[0]) & (f <= band[1]) & np.isfinite(g2) & (g2 > coh_min)
    if sel.sum() < 4:
        return np.nan, np.nan, np.nan, float(np.nanmedian(g2[(f >= band[0]) & (f <= band[1])]))
    nd = max(1, 2 * a.size // nperseg - 1)
    g2s = np.clip(g2[sel], 1e-6, 1 - 1e-9)
    var = (1.0 - g2s) / (2.0 * nd * g2s)
    w = 1.0 / var
    ph = np.unwrap(np.angle(Pab[sel]))
    fs_ = f[sel]
    # Weighted straight line; the intercept is kept free on purpose -- a pure
    # delay has none, so a large one is a signal that the two records differ by
    # something other than a time shift.
    S = w.sum()
    Sx = (w * fs_).sum()
    Sy = (w * ph).sum()
    Sxx = (w * fs_ * fs_).sum()
    Sxy = (w * fs_ * ph).sum()
    det = S * Sxx - Sx * Sx
    if det <= 0:
        return np.nan, np.nan, np.nan, float(np.median(g2s))
    slope = (S * Sxy - Sx * Sy) / det
    var_slope = S / det
    resid = ph - ((Sy - slope * Sx) / S + slope * fs_)
    chi2 = float((w * resid**2).sum() / max(1, sel.sum() - 2))
    tau = slope / (2.0 * np.pi)
    sigma = np.sqrt(var_slope) / (2.0 * np.pi)
    return float(tau), float(sigma), chi2, float(np.median(g2s))


def estimate_lag(
    ref: np.ndarray,
    tgt: np.ndarray,
    fs: float,
    band: tuple[float, float] = (0.05, 0.35),
    max_lag: float = 60.0,
    *,
    t_mid: float = np.nan,
    coh_min: float = 0.3,
    min_amplitude: float = 0.005,
    min_dynamic_range: float = 3.0,
    min_coherence: float = 0.5,
    max_chi2: float = 25.0,
    nperseg: int = 2048,
    iterations: int = 4,
    tol: float = 1e-4,
) -> LagResult:
    """Seconds to add to the target's clock to align it with the reference.

    Both inputs must already sit on the same uniform grid at *fs* with no NaNs;
    :func:`odas_tpw.clocksync.fit.window_pairs` is what guarantees that.
    """
    out = LagResult(n=int(min(ref.size, tgt.size)), t_mid=t_mid)
    n = out.n
    if n < max(64, int(4 * fs / band[0])):
        out.why = f"too short: {n} samples < 4 cycles of {band[0]} Hz"
        return out
    a = bandpass(np.asarray(ref, float)[:n], fs, *band)
    b = bandpass(np.asarray(tgt, float)[:n], fs, *band)
    if not (np.std(a) > 0 and np.std(b) > 0):
        out.why = "a band-passed series is flat"
        return out

    # Is there actually a wave here?  Without this gate a window holding only
    # tide will still be accepted: the band-pass leaves numerical ringing, the
    # ringing is identical in both records, so coherence is 1.00 and the peak
    # looks sharp -- a confident lag derived from filter transients.  The gate
    # is in dbar because that is where the physics is: 5 mm of water is below
    # any real sea state and above any plausible ringing.
    out.amplitude = amp = float(min(np.std(a), np.std(b)))
    if amp < min_amplitude:
        out.why = f"no wave energy in band: {1000 * amp:.2f} mm < {1000 * min_amplitude:g} mm"
        return out

    coarse, dyn, width, at_edge = _envelope_peak(a, b, fs, max_lag)
    out.dynamic_range, out.envelope_width = dyn, width
    if at_edge:
        out.why = f"envelope peak at the +/-{max_lag:g} s search boundary"
        return out
    if dyn < min_dynamic_range:
        out.why = f"envelope peak too flat: {dyn:.2f} < {min_dynamic_range:g}"
        return out

    # Iterate: shift by the running estimate, measure what is left.  Each
    # residual is small, so the unwrapped phase never approaches +/-pi.
    tau = coarse
    sigma = chi2 = coh = np.nan
    for _ in range(iterations):
        d, sigma, chi2, coh = _phase_slope(a, shift(b, fs, tau), fs, band, nperseg, coh_min)
        if not np.isfinite(d):
            out.why = f"cross-spectral phase undefined (coherence {coh:.2f})"
            out.coherence = coh
            return out
        tau += d
        if abs(d) < tol:
            break
    else:
        out.notes.append(f"phase refinement still moving by {d:.4f} s at the last iteration")

    out.lag, out.sigma, out.coherence, out.phase_chi2 = tau, sigma, coh, chi2
    if abs(tau) > max_lag:
        out.why = f"refined lag {tau:.2f} s left the +/-{max_lag:g} s window"
        return out
    if coh < min_coherence:
        out.why = f"coherence {coh:.2f} < {min_coherence:g}"
        return out
    if chi2 > max_chi2:
        out.why = f"phase is not a straight line (reduced chi2 {chi2:.1f} > {max_chi2:g})"
        return out
    out.ok = True
    return out


def band_power_fraction(x: np.ndarray, fs: float, band: tuple[float, float]) -> float:
    """Fraction of variance inside *band* --- how much wave there is to work with."""
    f, P = welch(x, fs=fs, nperseg=int(min(2048, x.size)), detrend="linear")
    tot = float(np.trapezoid(P, f))
    if tot <= 0:
        return 0.0
    m = (f >= band[0]) & (f <= band[1])
    return float(np.trapezoid(P[m], f[m]) / tot)

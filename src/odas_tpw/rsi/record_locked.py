# Oct-2026, Claude and Pat Welch, pat@mousebrains.com
"""Remove shear interference that is locked to the ``.p`` data record.

An instrument can pick up electrical interference from its own data system at
a fixed position in every data record. A VMP-5500 from an outside user shows
it on both shear channels: two ~16 ms pulses in every record at the same
sample positions (6 and 81-82 of 512), each with an undershoot and a slow
recovery that lasts much of the record; its mean peaks at 0.4-10x (median 4x)
the median record rms of the high-passed shear. The FP07 gradients are clean
and the accelerometers carry it only weakly (peak 0.2-1.2x their own record
rms in 3 casts checked), so Goodman, which removes what the shear shares with
them, cannot remove it. Nor does the despiker:
it caught both pulses of every record (replacing > 8% of the shear) in only
21 of the 88 cast-channels, and replaced < 1% in 36. Turbulence is not
locked to the record, so the mean over records estimates the interference,
and subtracting it removes it. Only the shear channels are corrected.

The interference is NOT stationary within a cast. In several of those casts
its amplitude drifts several-fold (from >3x to ~0.4x its cast mean over the
descent), and its shape changes too. Measured on all 88 cast-channels as the
record-locked mean left in each eighth of a cast, in standard errors at the
corrected samples (median over the eighths, even records only): 4-126
uncorrected; 1.5-75 after subtracting one whole-profile mean, which
over-subtracts where the pulse is weak and under-subtracts where it is
strong; 2-6 after the local estimate below, fitted on the odd records alone.
(One fitted shape with a drifting amplitude, tried on 5 casts, left 1.4-9,
with maxima up to 41.)

So the subtraction works in three steps, per profile and channel:

1. **Where** (detection, over the whole profile): the mean of the high-passed
   shear over the profile's quiet complete records, and the samples where it
   departs from its median level by :data:`Z_SEED` standard errors, together
   with the contiguous samples above :data:`Z_GROW` around them (the flanks,
   undershoot and recovery) and :data:`DILATE` more on each side.
2. **What** (estimation, local in time): at those samples, each record gets
   the weighted mean of its neighbors within :data:`LOCAL_HALF_WINDOW`
   records, itself left out, relative to that mean's median over the record
   (the level the detection measured from). A neighbor is weighted by the
   inverse of its turbulence variance, but never more than a median record:
   that keeps a turbulent patch out without letting a few very quiet records
   decide the waveform (uncapped, they left up to 6 standard errors of it in
   the cast with the broadest waveform; unweighted, up to 6 in others).
3. **Again**, up to :data:`MAX_PASSES` times, while the corrected shear still
   has a significant sample. Records whose pulse was strong enough to make
   them loud are left out of the first detection and only become quiet once
   corrected; their weaker flanks then show (up to 10 standard errors in one
   channel of the 88).

Choices that keep it from doing harm:

* **Exact framing.** A sample's position in its record is its index modulo
  :attr:`PFile.samples_per_record
  <odas_tpw.rsi.p_file.PFile.samples_per_record>`, never its time modulo the
  record duration: ``fs_fast`` is not an integer (512.03 Hz on that VMP-5500),
  so time drifts ~70 ms against the records over a 20-minute cast.
* **Profiling records only, and quiet ones for detection.** Everything is
  estimated from the complete records of one profile, never from the deck and
  surface time of the file around it (on a multi-cast VMP-250 file, ARCTERX
  SN479_0005, the median record rms of the whole file is 22-26x that of its
  profiles). The detection uses only records whose rms is at most
  :data:`QUIET_RMS_FACTOR` times the median, which keeps its statistic
  honest: over 924 profile-channels of clean VMP-250 shear (SN479, ~200
  records each) the plain mean put 14 samples beyond 5 standard errors,
  because a few turbulent records dominate both the mean and its spread; the
  quiet mean put none.
* **Only where the waveform is significant.** Every other sample is left
  bit-identical, and on clean shear nothing is corrected at all.
* **Leave-one-out.** A record's own turbulence never enters the waveform
  subtracted from it. What the local mean costs instead is its noise: at
  least ``1/(2*LOCAL_HALF_WINDOW)`` of the shear variance, at the corrected
  samples only -- more where the weights leave fewer records effectively
  averaged.
* **Estimated as subtracted.** The waveform is subtracted from the raw shear,
  before the division by speed squared (the interference is electrical and
  lives in those units), and L2 then high-passes the result. So it is
  estimated with the high-pass at :data:`ESTIMATE_CUT_FRACTION` of the L2
  cut: estimated at the L2 cut itself it lacks what that filter removes, and
  H(1 - H) of the interference is left at its low harmonics. At an L2 cut of
  1 Hz (perturb ``fft_sec: 0.5``) that left the 2nd and 3rd harmonics of the
  record mean at up to 6.3 standard errors on 4 casts, 3.8 with the low cut,
  and it halved the median epsilon bias of 20x synthetic pulses (+0.031 to
  +0.014 dex). At 0.25 Hz the two hardly differ. The level of a high-passed
  mean is
  arbitrary, so its median over the record is taken as zero, in the
  detection and in every local estimate alike.

What is left, by design or by construction:

* **The insignificant part, which can add up.** A slow recovery spread over
  the whole record can sit below :data:`Z_GROW` at every sample and so stay
  uncorrected, yet still show at the record's fundamental: on 4 casts the
  record-mean fundamental went from 13-30 standard errors to 2-9, and the
  ~1 Hz spectral line from 3-12x its neighboring continuum to 1.5-2.2x.
* **What varies from record to record.** A mean removes only the part
  repeated in every record. After the correction the corrected samples kept
  more than 1.2x the power of the others in 9 of the 88 cast-channels over
  all records (up to 2.2x) and in 27 over the turbulent (loud) records alone,
  which the detection never sees (up to 4.8x), mostly where the pulse was
  strong enough for the despiker to have caught it in every record. Those
  windows' epsilon stays biased high. ``_phase_power_ratio`` /
  ``_phase_power_ratio_loud`` in the products measure it.
* **What depends on the profile's extent.** Every estimate is per profile, so
  the corrected samples follow the profile bounds: the same cast corrected
  365 (sh1) and 80 (sh2) samples per record on the rsi-tpw route, 258 and 120
  on perturb's top-trimmed profile.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple

import numpy as np
from scipy.stats import trim_mean

from odas_tpw.scor160.l2 import _hp_filter

logger = logging.getLogger(__name__)

#: Standard errors a sample's mean must depart from the waveform's median to
#: be corrected. Nominally P = 5.7e-7 per sample, ~3e-4 per 512-sample
#: channel; measured: no sample of 924 clean VMP-250 profile-channels.
Z_SEED = 5.0
#: A significant sample also takes the contiguous samples above this many
#: standard errors on either side of it (circularly, since a record's last
#: sample abuts the next record's first).
Z_GROW = 2.0
#: Then this many more samples on each side.
DILATE = 2
#: A record is quiet when its high-passed rms is at most this many times the
#: profile's median record rms.
QUIET_RMS_FACTOR = 3.0
#: Fewer quiet complete records than this and nothing is estimated: the
#: Student-t tails at 5 standard errors grow fast below ~60 degrees of freedom.
MIN_RECORDS = 60
#: Records on each side of a record that its local waveform averages: tracks
#: a drift on the scale of a minute on a VMP. 8, 16 and 32 left the same
#: out-of-sample residual on 5 of the casts above; 64 a little more (block
#: maxima up to 13 standard errors against 8).
LOCAL_HALF_WINDOW = 16
#: A local window with fewer usable records than this falls back to the
#: whole-profile waveform.
MIN_LOCAL_RECORDS = 8
#: Detect-and-subtract passes at most; each pass after the first runs only if
#: the one before corrected something.
MAX_PASSES = 3
#: The waveform that is subtracted is estimated with the high-pass cut at this
#: fraction of the L2 cut (detection keeps the L2 cut). It is subtracted from
#: the raw series, which L2 then high-passes: estimating at the L2 cut itself
#: would leave H(1 - H) of the waveform at its low harmonics.
ESTIMATE_CUT_FRACTION = 0.1


class RecordLockedResult(NamedTuple):
    """Outcome of :func:`remove_record_locked` for one channel of one profile."""

    signal: np.ndarray
    """The corrected series: a new array, the input untouched."""
    records: int
    """Quiet complete records the detection used."""
    samples: int
    """Samples per record corrected (0: nothing significant, or too few records)."""
    peak: float
    """Profile-mean waveform at its most significant sample, in the signal's
    units (NaN when too few records)."""
    snr: float
    """``|peak|`` in standard errors of the record mean (NaN when too few records)."""
    peak_range: tuple[float, float]
    """Smallest and largest local waveform subtracted at that sample, which
    shows how far the interference drifted (NaN when nothing was corrected)."""
    phase_power_ratio: float
    """High-passed power at the corrected samples over that at the others,
    after the correction, per record and averaged over records (NaN when
    nothing was corrected). ~1 when nothing is left."""
    phase_power_ratio_loud: float
    """The same over loud records only (rms > QUIET_RMS_FACTOR x the median),
    which the detection never sees (NaN when there are none). Noisy when the
    loud records or the corrected samples are few: values down to 0.5 occur,
    which no subtraction can produce."""


def _significant(z: np.ndarray) -> np.ndarray:
    """Samples to correct: seeds beyond Z_SEED, grown through Z_GROW, dilated."""
    absz = np.abs(z)
    mask = absz >= Z_SEED
    if not mask.any():
        return mask
    grow = absz >= Z_GROW
    while True:
        wider = mask | ((np.roll(mask, 1) | np.roll(mask, -1)) & grow)
        if np.array_equal(wider, mask):
            break
        mask = wider
    for _ in range(DILATE):
        mask = mask | np.roll(mask, 1) | np.roll(mask, -1)
    return mask


def _local_waveforms(
    stack: np.ndarray, usable: np.ndarray, mask: np.ndarray, fallback: np.ndarray
) -> np.ndarray:
    """Per-record waveform at the *mask* samples, zero elsewhere.

    Leave-one-out mean over +-LOCAL_HALF_WINDOW records, each weighted by the
    inverse of its variance over the uncorrected samples (turbulence, not the
    interference) floored at the median record's, referred to its own median
    over the record -- the level the detection measured significance from, so
    a corrected sample lands where an uncorrected one sits.
    """
    n_rec, spr = stack.shape
    data = np.where(usable[:, None], stack, 0.0)
    # Off-mask samples carry no significant interference, so their variance is
    # the record's turbulence. With (almost) every sample significant, use all.
    off = ~mask if (~mask).sum() >= 8 else np.ones(spr, dtype=bool)
    var = np.mean(data[:, off] ** 2, axis=1)
    floor = np.median(var[usable]) if usable.any() else 0.0
    var = np.maximum(var, floor)
    weight = np.where(usable & (var > 0), 1.0 / np.where(var > 0, var, 1.0), 0.0)
    weighted = weight[:, None] * data
    zero = np.zeros((1, spr))
    sums = np.vstack([zero, np.cumsum(weighted, axis=0)])
    wsum = np.concatenate([[0.0], np.cumsum(weight)])
    count = np.concatenate([[0], np.cumsum(usable.astype(np.int64))])
    r = np.arange(n_rec)
    lo = np.clip(r - LOCAL_HALF_WINDOW, 0, n_rec)
    hi = np.clip(r + LOCAL_HALF_WINDOW + 1, 0, n_rec)
    total = sums[hi] - sums[lo] - weighted  # leave the record itself out
    wtot = wsum[hi] - wsum[lo] - weight
    ntot = count[hi] - count[lo] - usable.astype(np.int64)
    local = total / np.where(wtot > 0, wtot, 1.0)[:, None]
    local -= np.median(local, axis=1, keepdims=True)
    ok = (ntot >= MIN_LOCAL_RECORDS) & (wtot > 0)
    local = np.where(ok[:, None], local, fallback[None, :])
    return np.where(mask[None, :], local, 0.0)


def remove_record_locked(
    x: np.ndarray,
    *,
    index0: int,
    samples_per_record: int,
    fs: float,
    hp_cut: float,
    bad: np.ndarray | None = None,
) -> RecordLockedResult:
    """Subtract the significant part of *x*'s record-locked waveform.

    Parameters
    ----------
    x : ndarray
        One profile of one shear channel, un-normalized (before ``/speed**2``).
    index0 : int
        Index of ``x[0]`` among the fast samples of the source ``.p`` file;
        ``x[k]`` is at position ``(index0 + k) % samples_per_record`` of its
        data record.
    samples_per_record : int
        Fast samples per data record of the source file.
    fs : float
        Fast sampling rate [Hz].
    hp_cut : float
        High-pass cutoff [Hz] applied before averaging (the L2 shear filter).
    bad : ndarray of bool, optional
        RDL bad-buffer samples of this channel, aligned with *x*; a record
        holding one (or a non-finite sample) is left out of every estimate,
        but is corrected like the others.
    """
    x = np.asarray(x, dtype=np.float64)
    spr = int(samples_per_record)
    if spr < 2:
        raise ValueError(f"samples_per_record must be >= 2, got {samples_per_record!r}")
    n = len(x)
    k0 = (-int(index0)) % spr  # first sample of the first complete record
    n_rec = max(0, (n - k0) // spr)
    stop = k0 + n_rec * spr
    none = (np.nan, np.nan)

    usable = np.zeros(n_rec, dtype=bool)
    if n_rec:
        usable = np.all(np.isfinite(x[k0:stop].reshape(n_rec, spr)), axis=1)
        if bad is not None:
            usable &= ~np.any(np.asarray(bad, dtype=bool)[k0:stop].reshape(n_rec, spr), axis=1)
    if usable.sum() < MIN_RECORDS:
        return RecordLockedResult(
            x.copy(), int(usable.sum()), 0, np.nan, np.nan, none, np.nan, np.nan
        )

    out = x.copy()
    corrected = np.zeros(spr, dtype=bool)
    n_quiet, peak, snr, span = 0, np.nan, np.nan, none
    for i_pass in range(MAX_PASSES):
        stack = _hp_filter(out, fs, hp_cut)[k0:stop].reshape(n_rec, spr)
        rms = np.full(n_rec, np.inf)
        rms[usable] = np.sqrt(np.mean(stack[usable] ** 2, axis=1))
        quiet_rows = usable & (rms <= QUIET_RMS_FACTOR * np.median(rms[usable]))
        quiet = stack[quiet_rows]
        if len(quiet) < MIN_RECORDS:
            break
        mean = quiet.mean(axis=0)
        se = quiet.std(axis=0, ddof=1) / np.sqrt(len(quiet))
        wave = mean - np.median(mean)
        # A channel that never varies (a dead or railed probe) has se == 0: no
        # evidence of anything, rather than an infinite z.
        z = np.divide(wave, se, out=np.zeros(spr), where=se > 0)
        mask = _significant(z)
        if i_pass == 0:
            # The uncorrected statistics are the ones reported.
            n_quiet = len(quiet)
            i_peak = int(np.argmax(np.abs(z)))
            peak, snr = float(wave[i_peak]), float(abs(z[i_peak]))
        if not mask.any():
            break
        # What is subtracted is estimated nearly unfiltered (see
        # ESTIMATE_CUT_FRACTION), where the detection used the L2 band.
        est = _hp_filter(out, fs, hp_cut * ESTIMATE_CUT_FRACTION)[k0:stop].reshape(n_rec, spr)
        est_mean = est[quiet_rows].mean(axis=0)
        local = _local_waveforms(est, usable, mask, est_mean - np.median(est_mean))
        out[k0:stop] -= local.ravel()
        # Partial records at the profile's ends take their nearest whole record's.
        out[:k0] -= local[0, spr - k0 :]
        out[stop:] -= local[-1, : n - stop]
        if i_pass == 0:
            span = (float(local[:, i_peak].min()), float(local[:, i_peak].max()))
        corrected |= mask
    if n_quiet == 0:  # too few quiet records to look at all
        return RecordLockedResult(x.copy(), len(quiet), 0, np.nan, np.nan, none, np.nan, np.nan)
    if not corrected.any():
        # Bit-identical to the input, not merely equal: nothing was subtracted.
        return RecordLockedResult(x.copy(), n_quiet, 0, peak, snr, none, np.nan, np.nan)
    ratio, ratio_loud = _phase_power_ratios(
        _hp_filter(out, fs, hp_cut)[k0:stop].reshape(n_rec, spr), usable, corrected
    )
    return RecordLockedResult(
        out, n_quiet, int(corrected.sum()), peak, snr, span, ratio, ratio_loud
    )


def _phase_power_ratios(
    stack: np.ndarray, usable: np.ndarray, corrected: np.ndarray
) -> tuple[float, float]:
    """Corrected- to uncorrected-sample power after the correction, per record.

    Averaged over every usable record, and over the loud ones alone. The
    record MEAN at the corrected samples is ~0 after the subtraction by
    construction; this is what is left that is locked to the record in
    position but not in shape, which no mean can remove. Near 1 means nothing
    is left. Each record's ratio is averaged (10% trimmed at each end, against
    a probe strike), not its median taken: with few corrected samples the
    ratio has few degrees of freedom and its median sits well below 1 on pure
    turbulence (0.79 on the test file).
    """
    if corrected.all():
        return np.nan, np.nan
    rows = stack[usable]
    ratio = np.mean(rows[:, corrected] ** 2, axis=1) / np.maximum(
        np.mean(rows[:, ~corrected] ** 2, axis=1), np.finfo(float).tiny
    )
    rms = np.sqrt(np.mean(rows**2, axis=1))
    loud = rms > QUIET_RMS_FACTOR * np.median(rms)
    return (
        float(trim_mean(ratio, 0.1)),
        float(trim_mean(ratio[loud], 0.1)) if loud.any() else np.nan,
    )


#: Written to every product made with the removal on; the per-channel
#: ``record_locked_<channel>_*`` attributes it describes sit beside it.
COMMENT = (
    "Shear interference locked to the .p data record was removed per profile, "
    "before the high-pass and despike. Detection: the mean of the high-passed "
    f"shear over quiet complete records (rms <= {QUIET_RMS_FACTOR:g}x the median, "
    f"at least {MIN_RECORDS}), significant where it departs from its median by "
    f">= {Z_SEED:g} standard errors, grown through >= {Z_GROW:g} and dilated "
    f"{DILATE} samples. Subtracted there only: each record's weighted mean over "
    f"the {LOCAL_HALF_WINDOW} records on either side, itself left out (weights: "
    "inverse turbulence variance, capped at the median record's; high-pass cut "
    f"{ESTIMATE_CUT_FRACTION:g}x the L2 one), relative to its median; repeated, up "
    f"to {MAX_PASSES} passes, while anything is still significant. Per channel: "
    "_records = quiet records used, _samples = samples per record corrected "
    "(0 = nothing significant), _peak = the profile-mean waveform at its most "
    "significant sample (shear channel units, before division by speed "
    "squared), _snr = that sample in standard errors, _peak_range = smallest and "
    "largest local waveform subtracted there, _phase_power_ratio and "
    "_phase_power_ratio_loud = the high-passed power at the corrected samples "
    "over that at the others, after, per record and averaged (10% trimmed) over "
    "all records and over loud records (rms > 3x median): ~1 means nothing is "
    "left; above 1, part of the "
    "interference varies in shape from record to record, which no mean removes, "
    "and those records' epsilon stays biased high. See odas_tpw.rsi.record_locked."
)


def remove_from_profile(
    data: Mapping[str, Any],
    target: Sequence[tuple[str, np.ndarray]],
    s_fast: int,
    e_fast: int,
    *,
    framing: tuple[int, int],
    fs: float,
    hp_cut: float,
    context: str,
) -> dict[str, Any]:
    """Correct one profile of every shear channel; return its product attrs.

    Estimates from ``data["shear"]`` (never modified) and writes the corrected
    ``[s_fast:e_fast]`` slice into the matching array of *target*, a copy the
    caller builds L1 from. *framing* is the source's ``(record_offset,
    samples_per_record)`` (``ChannelsDict.record_framing``).
    """
    offset0, spr = framing
    bad_masks = data.get("bad_masks") or {}
    attrs: dict[str, Any] = {
        "record_locked_samples_per_record": int(spr),
        "record_locked_comment": COMMENT,
    }
    for (name, arr), (_, out) in zip(data["shear"], target):
        bad = bad_masks.get(name)
        res = remove_record_locked(
            np.asarray(arr[s_fast:e_fast]),
            index0=offset0 + s_fast,
            samples_per_record=spr,
            fs=fs,
            hp_cut=hp_cut,
            bad=None if bad is None else bad[s_fast:e_fast],
        )
        out[s_fast:e_fast] = res.signal
        attrs[f"record_locked_{name}_records"] = res.records
        attrs[f"record_locked_{name}_samples"] = res.samples
        attrs[f"record_locked_{name}_peak"] = res.peak
        attrs[f"record_locked_{name}_snr"] = res.snr
        attrs[f"record_locked_{name}_peak_range"] = np.array(res.peak_range)
        attrs[f"record_locked_{name}_phase_power_ratio"] = res.phase_power_ratio
        attrs[f"record_locked_{name}_phase_power_ratio_loud"] = res.phase_power_ratio_loud
        if res.samples:
            logger.info(
                "%s %s: record-locked interference removed at %d of %d samples per "
                "record (%d quiet records; peak %.3g, SNR %.0f; local peak %.3g to %.3g; "
                "power left at those samples %.2fx the others, %.2fx in loud records)",
                context,
                name,
                res.samples,
                spr,
                res.records,
                res.peak,
                res.snr,
                *res.peak_range,
                res.phase_power_ratio,
                res.phase_power_ratio_loud,
            )
        elif np.isfinite(res.snr):
            logger.info(
                "%s %s: no significant record-locked interference (max SNR %.1f "
                "over %d quiet records); nothing corrected",
                context,
                name,
                res.snr,
                res.records,
            )
        else:
            logger.info(
                "%s %s: record-locked interference not estimated: %d quiet complete "
                "records, need %d; nothing corrected",
                context,
                name,
                res.records,
                MIN_RECORDS,
            )
    return attrs

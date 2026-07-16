# Jul-2026, Claude and Pat Welch, pat@mousebrains.com
"""Bench-test diagnostic for Rockland Scientific instruments.

Python port of ODAS ``quick_bench.m``, extended to automatically evaluate the
**Rockland Bench Test Review Checklist** (V3).

A *bench test* is a pre-deployment health check: the instrument is rested on
soft foam with dummy probes installed and a short (>=60 s) file is recorded.
The raw-count time series and counts^2/Hz spectra reveal corroded connections,
dead channels and excessive electronic noise before the instrument goes in the
water. See ``quick_look`` / ``diss_look`` for the equivalent check on real
ocean profiles.

Two figures are produced, faithful to ``quick_bench.m``:

* **Time series** — every channel in *raw counts* (inclinometers converted to
  physical units), with the fast thermistor gradient channels mean-subtracted
  and their offset shown in the legend.
* **Spectra** — a log-log overlay of the auto-spectra in *counts^2/Hz*, to be
  compared against the noise floor in the instrument's RSI calibration report.

An optional third **CT/CLTU** figure is drawn when JAC-T/C, turbidity or
chlorophyll channels are present.

Beyond ``quick_bench`` (which only plots), this module evaluates the numeric
criteria on the Rockland checklist and reports each as PASS / FAIL, marking the
genuinely subjective items ("similar to each other", "rising curve", "no
spikes") as REVIEW for a human.

Why raw counts require ``PFile(path, deconvolve=False)``: the checklist
thresholds for the pre-emphasized channels (``T1_dT1``, ``T2_dT2``, ``P_dP``)
are defined on the raw pre-emphasized signal (that is why their spectra
*rise*), not on the deconvolved high-resolution reconstruction that
``PFile`` produces by default. ``deconvolve=False`` keeps those channels as the
raw int16 counts the checklist — and the RSI calibration report — refer to.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from odas_tpw.rsi.p_file import PFile, instrument_sn
from odas_tpw.scor160.spectral import csd_odas

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Checklist thresholds (Rockland Bench Test Review Checklist V3)
# ---------------------------------------------------------------------------
# Time series, raw counts:
_AXY_RANGE = 500.0  # |Ax|, |Ay| deviation from mean
_TDT_RANGE = 40.0  # |T*_dT*| deviation from mean
_TDT_OFFSET = 100.0  # |mean| of T*_dT* (legend offset)
_SH_MEAN = 10.0  # |mean| of sh*
_SH_RANGE = 30.0  # sh* deviation from mean
_P_RANGE = 2.0  # P deviation from mean
_PDP_RANGE = 10.0  # P_dP deviation from mean
_C1_RANGE = 50.0  # C1_dC1 deviation from mean (if present)
_C1_OFFSET = 6000.0  # |mean| of C1_dC1 (if present)
# Spectra, counts^2/Hz:
_PDP_PSD_MAX = 10.0  # P_dP spectral density everywhere below this
_PDP_PSD_PEAK = 3.0  # P_dP spectral peak below this
_AXY_PSD_MAX = 100.0  # Ax, Ay spectral peaks below this
# CT/CLTU, raw counts (page 3):
_JAC_T_RANGE = 50.0
_TURB_RANGE = 50.0
_CHLA_RANGE = 400.0
# JAC_C's I (<= +/-5) and V (~1e4, <= +/-100) sub-channels are drawn in the
# CT/CLTU figure for visual review rather than range-checked here, because the
# conductivity board packs both into one 32-bit word (see _ctclu_panels).

# Expected spectral-density band value near 100 Hz (order of magnitude; the
# checklist wording is "rising curve of approximately 10^N counts^2/Hz near
# 10^2 Hz"). Reported alongside a REVIEW verdict, not auto-passed.
_PROBE_FREQ = 100.0  # Hz

# The checklist criteria read "typically within +/-N counts". That is a robust
# envelope of the bulk signal, NOT the single worst sample: a lone DAQ glitch or
# a bench bump (accelerometers register any nearby movement) must not fail an
# otherwise-clean channel. We take the 99.9th percentile of |x - mean| as the
# "typical" deviation. The true maximum is kept separately for the spike hint.
_TYP_PCT = 99.9

# Status vocabulary
PASS = "PASS"
FAIL = "FAIL"
REVIEW = "REVIEW"
NA = "N/A"

# ODAS quick_bench uses a 2 s FFT for the bench spectra.
DEFAULT_FFT_SEC = 2.0


# ---------------------------------------------------------------------------
# Per-channel statistics
# ---------------------------------------------------------------------------


@dataclass
class ChannelStat:
    """Time-series and (optionally) spectral statistics for one channel."""

    name: str
    ctype: str
    is_fast: bool
    unit: str
    n: int
    mean: float
    std: float
    typ_dev: float  # 99.9th-percentile |x - mean|  ("typically within +/-N")
    max_abs_dev: float  # true max|x - mean|  (spike hint only)
    # Spectrum (present only for the channels we transform):
    freq: np.ndarray | None = None
    psd: np.ndarray | None = None
    psd_max: float = float("nan")
    psd_at_probe: float = float("nan")  # PSD nearest _PROBE_FREQ
    peak_freq: float = float("nan")  # frequency of psd_max


@dataclass
class BenchStats:
    """All statistics needed to draw the figures and evaluate the checklist."""

    filename: str
    sn: str
    fs_fast: float
    fs_slow: float
    fft_sec: float
    start_time: str
    channels: dict[str, ChannelStat] = field(default_factory=dict)

    def get(self, name: str) -> ChannelStat | None:
        return self.channels.get(name)


@dataclass
class CheckItem:
    """One line of the evaluated checklist."""

    section: str  # "Time Series" | "Spectra" | "CT/CLTU"
    label: str
    measured: str
    threshold: str
    status: str  # PASS | FAIL | REVIEW | N/A


# ---------------------------------------------------------------------------
# Channel classification
# ---------------------------------------------------------------------------

# Which channels get an auto-spectrum, mirroring quick_bench.m: all fast
# vibration/shear/thermistor channels, plus the slow pre-emphasized pressure.
_VIB_TYPES = {"piezo", "accel"}


def _spectrum_channels(pf: PFile) -> list[str]:
    """Names to auto-spectrum: fast vibration/shear/therm channels + P_dP."""
    names = []
    for name in pf.channels_raw:
        ctype = pf.channel_info.get(name, {}).get("type", "")
        if pf.is_fast(name) and (ctype in _VIB_TYPES or ctype in {"shear", "therm"}):
            names.append(name)
    if "P_dP" in pf.channels_raw:
        names.append("P_dP")
    return names


# ---------------------------------------------------------------------------
# Spectra
# ---------------------------------------------------------------------------


def _psd(x: np.ndarray, nfft: int, fs: float) -> tuple[np.ndarray, np.ndarray] | None:
    """One-sided auto-spectrum in input-units^2/Hz via Welch/cosine window.

    Feeds the raw signal with linear per-segment detrending, exactly as ODAS
    ``quick_bench`` calls ``csd_odas(x, x, nfft, fs, [], nfft/2, 'linear')``.
    Returns ``(freq, psd)`` or ``None`` when the record is shorter than
    ``2*nfft`` (``csd_matrix`` requires at least two segments).
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if nfft < 2 or len(x) < 2 * nfft:
        return None
    res = csd_odas(x, None, nfft, fs, detrend="linear")
    return res.F, res.Cxy


def _nearest(freq: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(freq - target)))


def compute_bench_stats(pf: PFile, fft_sec: float = DEFAULT_FFT_SEC) -> BenchStats:
    """Compute the per-channel time-series and spectral statistics.

    ``pf`` should be constructed with ``deconvolve=False`` so the pre-emphasized
    channels carry raw counts.  Count-based channels use ``channels_raw``;
    inclinometers use the physical-unit ``channels`` (degrees / degC), matching
    the one conversion ``quick_bench`` performs.
    """
    nfft_fast = round(fft_sec * pf.fs_fast)
    nfft_slow = round(nfft_fast * pf.fs_slow / pf.fs_fast) if pf.fs_fast > 0 else 0

    spec_names = set(_spectrum_channels(pf))
    stats = BenchStats(
        filename=pf.filepath.name,
        sn=str(instrument_sn(pf.config.get("instrument_info", {}))),
        fs_fast=float(pf.fs_fast),
        fs_slow=float(pf.fs_slow),
        fft_sec=float(fft_sec),
        start_time=pf.start_time.isoformat(),
    )

    for name in pf.channels_raw:
        info = pf.channel_info.get(name, {})
        ctype = info.get("type", "")
        is_fast = pf.is_fast(name)
        # Inclinometers are reported in physical units; everything else in raw
        # counts.  channels_raw is int16 (deconvolve=False); cast to float for
        # the moment statistics.
        if ctype in {"inclxy", "inclt"}:
            arr = np.asarray(pf.channels[name], dtype=np.float64)
            unit = info.get("units", "")
        else:
            arr = np.asarray(pf.channels_raw[name], dtype=np.float64)
            unit = "counts"
        if arr.size == 0:
            continue
        mean = float(np.mean(arr))
        abs_dev = np.abs(arr - mean)
        cs = ChannelStat(
            name=name,
            ctype=ctype,
            is_fast=is_fast,
            unit=unit,
            n=int(arr.size),
            mean=mean,
            std=float(np.std(arr)),
            typ_dev=float(np.percentile(abs_dev, _TYP_PCT)),
            max_abs_dev=float(np.max(abs_dev)),
        )

        if name in spec_names:
            nfft = nfft_fast if is_fast else nfft_slow
            fs = pf.fs_fast if is_fast else pf.fs_slow
            spec = _psd(pf.channels_raw[name], nfft, fs)
            if spec is None:
                log.warning(
                    "%s: %s too short for a %d-point spectrum (need >= %d samples); "
                    "skipping its spectrum",
                    pf.filepath.name,
                    name,
                    nfft,
                    2 * nfft,
                )
            else:
                freq, psd = spec
                cs.freq = freq
                cs.psd = psd
                # Ignore the DC bin when locating the peak / maximum.
                body = slice(1, None)
                if psd[body].size:
                    imax = int(np.argmax(psd[body])) + 1
                    cs.psd_max = float(psd[imax])
                    cs.peak_freq = float(freq[imax])
                    cs.psd_at_probe = float(psd[_nearest(freq, _PROBE_FREQ)])
        stats.channels[name] = cs

    return stats


# ---------------------------------------------------------------------------
# Checklist evaluation
# ---------------------------------------------------------------------------


def _fmt(value: float, unit: str = "") -> str:
    if not np.isfinite(value):
        return "n/a"
    suffix = f" {unit}" if unit else ""
    if abs(value) >= 1e4 or (value != 0 and abs(value) < 1e-2):
        return f"{value:.2e}{suffix}"
    return f"{value:.3g}{suffix}"


def _add_range(
    items: list[CheckItem], section: str, label: str, cs: ChannelStat | None, limit: float
) -> None:
    """Append a "typically within +/-limit counts" PASS/FAIL check when present.

    Uses the robust 99.9th-percentile deviation (``typ_dev``), not the absolute
    maximum, matching the checklist's "typically within" wording. When the true
    maximum still exceeds the limit (out-of-range samples in the outer 0.1%),
    the max is surfaced in the measured value so the robust metric never
    silently masks genuine excursions.
    """
    if cs is None:
        return
    status = PASS if cs.typ_dev <= limit else FAIL
    measured = f"+/-{_fmt(cs.typ_dev)} counts"
    if cs.max_abs_dev > limit:
        measured += f" (max +/-{_fmt(cs.max_abs_dev)})"
    items.append(
        CheckItem(section, label, measured, f"<= +/-{limit:g} (p99.9)", status)
    )


def _sorted_by_type(stats: BenchStats, ctype: str) -> list[ChannelStat]:
    return [cs for cs in stats.channels.values() if cs.ctype == ctype]


def evaluate_checklist(stats: BenchStats) -> list[CheckItem]:
    """Evaluate the Rockland bench-test checklist against ``stats``.

    Quantitative criteria become PASS/FAIL with the measured value; genuinely
    subjective ones ("similar", "rising curve", "seemingly random") become
    REVIEW, carrying a helper number where one is cheap to compute.  Optional
    channels that are absent (e.g. micro-conductivity) are reported N/A so the
    operator sees they were considered.
    """
    items: list[CheckItem] = []
    ts = "Time Series"
    sp = "Spectra"

    # --- Vibration (Ax, Ay) ------------------------------------------------
    vib = [cs for cs in stats.channels.values() if cs.ctype in _VIB_TYPES]
    vib.sort(key=lambda c: c.name)
    for cs in vib:
        _add_range(items, ts, f"{cs.name} range", cs, _AXY_RANGE)
    if len(vib) >= 2:
        a, b = vib[0], vib[1]
        ratio = a.std / b.std if b.std else float("nan")
        items.append(
            CheckItem(
                ts,
                f"{a.name}~{b.name} similar, {a.name}>{b.name}",
                f"std ratio {_fmt(ratio)}",
                "similar; larger first",
                REVIEW,
            )
        )
    for cs in vib:
        # Whether a peak is a "large spike" vs cushioning vibration is a visual
        # call; report the peak-to-std ratio as a hint and defer to a human.
        ratio = cs.max_abs_dev / cs.std if cs.std else float("nan")
        items.append(
            CheckItem(
                ts, f"{cs.name} large spikes?", f"peak/std {_fmt(ratio)}", "no large spikes", REVIEW
            )
        )

    # --- Inclinometers -----------------------------------------------------
    for name in ("Incl_T", "Incl_X", "Incl_Y"):
        ics = stats.get(name)
        if ics is not None:
            items.append(
                CheckItem(
                    ts,
                    f"{name} reasonable & constant",
                    f"{_fmt(ics.mean, ics.unit)} (std {_fmt(ics.std)})",
                    "reasonable, constant",
                    REVIEW,
                )
            )

    # --- Fast thermistor gradient (T*_dT*) ---------------------------------
    tdt = sorted(
        (cs for cs in stats.channels.values() if cs.ctype == "therm" and cs.is_fast),
        key=lambda c: c.name,
    )
    for cs in tdt:
        _add_range(items, ts, f"{cs.name} range", cs, _TDT_RANGE)
        status = PASS if abs(cs.mean) < _TDT_OFFSET else FAIL
        items.append(
            CheckItem(
                ts, f"{cs.name} offset", f"{_fmt(cs.mean)} counts", f"< {_TDT_OFFSET:g}", status
            )
        )

    # --- Shear (sh*) -------------------------------------------------------
    shear = _sorted_by_type(stats, "shear")
    shear.sort(key=lambda c: c.name)
    for cs in shear:
        status = PASS if abs(cs.mean) < _SH_MEAN else FAIL
        items.append(
            CheckItem(ts, f"{cs.name} mean", f"{_fmt(cs.mean)} counts", f"< {_SH_MEAN:g}", status)
        )
        _add_range(items, ts, f"{cs.name} range", cs, _SH_RANGE)

    # --- Pressure ----------------------------------------------------------
    _add_range(items, ts, "P range", stats.get("P"), _P_RANGE)
    pdp = stats.get("P_dP")
    if pdp is not None:
        _add_range(items, ts, "P_dP range", pdp, _PDP_RANGE)
        items.append(
            CheckItem(ts, "P_dP seemingly random", "see figure", "no spikes/patterns", REVIEW)
        )

    # --- Micro-conductivity (optional; N/A when absent) --------------------
    c1 = next((cs for cs in stats.channels.values() if cs.ctype == "ucond"), None)
    if c1 is None:
        items.append(CheckItem(ts, "C1_dC1 range", "absent", f"<= +/-{_C1_RANGE:g}", NA))
        items.append(CheckItem(ts, "C1_dC1 offset", "absent", f"< {_C1_OFFSET:g}", NA))
    else:
        _add_range(items, ts, "C1_dC1 range", c1, _C1_RANGE)
        status = PASS if abs(c1.mean) < _C1_OFFSET else FAIL
        items.append(
            CheckItem(ts, "C1_dC1 offset", f"{_fmt(c1.mean)} counts", f"< {_C1_OFFSET:g}", status)
        )

    # --- Spectra -----------------------------------------------------------
    if pdp is not None and pdp.psd is not None:
        s1 = PASS if pdp.psd_max < _PDP_PSD_MAX else FAIL
        items.append(
            CheckItem(sp, "P_dP density max", _fmt(pdp.psd_max), f"< {_PDP_PSD_MAX:g}", s1)
        )
        s2 = PASS if pdp.psd_max < _PDP_PSD_PEAK else FAIL
        items.append(
            CheckItem(
                sp,
                "P_dP peak",
                f"{_fmt(pdp.psd_max)} at {_fmt(pdp.peak_freq)} Hz",
                f"< {_PDP_PSD_PEAK:g}, rolloff ~2 Hz",
                s2,
            )
        )
    for cs in vib:
        if cs.psd is None:
            continue
        status = PASS if cs.psd_max < _AXY_PSD_MAX else FAIL
        items.append(
            CheckItem(sp, f"{cs.name} peak", _fmt(cs.psd_max), f"< {_AXY_PSD_MAX:g}", status)
        )
    for cs in tdt:
        if cs.psd is None:
            continue
        items.append(
            CheckItem(
                sp,
                f"{cs.name} rising ~1e-1 near 100 Hz",
                f"{_fmt(cs.psd_at_probe)} at 100 Hz",
                "rising to ~1e-1",
                REVIEW,
            )
        )
    for cs in shear:
        if cs.psd is None:
            continue
        items.append(
            CheckItem(
                sp,
                f"{cs.name} rising ~1e-2 near 100 Hz",
                f"{_fmt(cs.psd_at_probe)} at 100 Hz",
                "rising to ~1e-2",
                REVIEW,
            )
        )

    # --- CT/CLTU (page 3) --------------------------------------------------
    items.extend(_evaluate_ctclu(stats))

    return items


def _evaluate_ctclu(stats: BenchStats) -> list[CheckItem]:
    sec = "CT/CLTU"
    items: list[CheckItem] = []
    jt = next((cs for cs in stats.channels.values() if cs.ctype == "jac_t"), None)
    _add_range(items, sec, "JAC_T range", jt, _JAC_T_RANGE)
    for name, limit in (("Turbidity", _TURB_RANGE), ("Chlorophyll", _CHLA_RANGE)):
        _add_range(items, sec, f"{name} range", stats.get(name), limit)
    if any(cs.ctype == "jac_c" for cs in stats.channels.values()):
        items.append(
            CheckItem(sec, "JAC_C I/V", "see figure", "I <= +/-5, V ~1e4 +/-100", REVIEW)
        )
    return items


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

_SYMBOL = {PASS: "[ ok ]", FAIL: "[FAIL]", REVIEW: "[rvw ]", NA: "[ -- ]"}


def render_checklist_text(items: list[CheckItem], header: str) -> str:
    """Format the evaluated checklist as an aligned, section-grouped report."""
    lines = [header, "=" * max((len(line) for line in header.splitlines()), default=0)]
    n_pass = sum(1 for it in items if it.status == PASS)
    n_fail = sum(1 for it in items if it.status == FAIL)
    n_rev = sum(1 for it in items if it.status == REVIEW)
    n_na = sum(1 for it in items if it.status == NA)
    lines.append(
        f"Summary: {n_pass} pass, {n_fail} fail, {n_rev} review, {n_na} n/a "
        f"({len(items)} checks)"
    )

    label_w = max((len(it.label) for it in items), default=10)
    meas_w = max((len(it.measured) for it in items), default=10)
    last_section = None
    for it in items:
        if it.section != last_section:
            lines.append("")
            lines.append(f"-- {it.section} --")
            last_section = it.section
        lines.append(
            f"  {_SYMBOL[it.status]} {it.label:<{label_w}}  "
            f"{it.measured:<{meas_w}}  ({it.threshold})"
        )
    lines.append("")
    lines.append(
        "REVIEW items need a human eye (compare spectra against the RSI "
        "calibration report). Narrow-band spikes at 50/60 Hz (AC) or 15 Hz (EM) "
        "are expected; broadband noise in one channel is not."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

# Time-series groups: (title, [channel names], subtract_mean, physical_units).
# Only groups with at least one present channel are drawn.


def _time_and_data(
    pf: PFile, name: str, physical: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Return (time, data) sliced to equal length for one channel."""
    src = pf.channels if physical else pf.channels_raw
    y = np.asarray(src[name], dtype=np.float64)
    t = pf.t_fast if pf.is_fast(name) else pf.t_slow
    n = min(len(t), len(y))
    return t[:n], y[:n]


def _timeseries_groups(pf: PFile) -> list[tuple[str, list[str], bool, bool]]:
    """Build the ordered list of time-series panels present in ``pf``."""

    def present(names: list[str]) -> list[str]:
        return [n for n in names if n in pf.channels_raw]

    def by_type(types: set[str], fast: bool | None = None) -> list[str]:
        out = []
        for n in sorted(pf.channels_raw):
            ct = pf.channel_info.get(n, {}).get("type", "")
            if ct in types and (fast is None or pf.is_fast(n) == fast):
                out.append(n)
        return out

    groups: list[tuple[str, list[str], bool, bool]] = []
    # (title, names, subtract_mean, physical)
    vib = by_type(_VIB_TYPES)
    if vib:
        groups.append(("Vibration [counts]", vib, False, False))
    shear = by_type({"shear"})
    if shear:
        groups.append(("Shear [counts]", shear, False, False))
    tdt = by_type({"therm"}, fast=True)
    if tdt:
        groups.append(("Thermistor gradient [counts]", tdt, True, False))
    ucond = by_type({"ucond"}, fast=True)
    if ucond:
        groups.append(("Micro-conductivity [counts]", ucond, True, False))
    mag = by_type({"magn"})
    if mag:
        groups.append(("Magnetometer [counts]", mag, False, False))
    pres = present(["P", "P_dP"])
    if pres:
        groups.append(("Pressure [counts]", pres, False, False))
    incl = present(["Incl_X", "Incl_Y", "Incl_T"])
    if incl:
        groups.append(("Inclinometer [physical]", incl, False, True))
    return groups


def _ts_range_limit(name: str, ctype: str) -> float | None:
    """The "typically within +/-N counts" limit for a time-series channel.

    Returns the +/- count limit whose band is shaded on the plot, or None for
    channels with no numeric criterion (inclinometers, magnetometer).
    """
    if ctype in _VIB_TYPES:
        return _AXY_RANGE
    if ctype == "therm":
        # Only the fast pre-emphasized gradient channels (T*_dT*) reach here:
        # _timeseries_groups builds the therm group with fast=True, so the slow
        # base T* (which would want a different criterion) is never shaded.
        return _TDT_RANGE
    if ctype == "shear":
        return _SH_RANGE
    if ctype == "ucond":
        return _C1_RANGE
    if ctype == "jac_t":
        return _JAC_T_RANGE
    return {
        "P": _P_RANGE,
        "P_dP": _PDP_RANGE,
        "Turbidity": _TURB_RANGE,
        "Chlorophyll": _CHLA_RANGE,
    }.get(name)


def build_timeseries_figure(pf: PFile, stats: BenchStats, title: str) -> Figure:
    """Stacked raw-count time series, one panel per channel group.

    Each channel's checklist "valid range" (mean +/- its threshold) is drawn as
    a shaded band in the channel's color, so an out-of-range excursion is
    visible as the trace poking outside its band.
    """
    groups = _timeseries_groups(pf)
    n = max(len(groups), 1)
    fig, axes = plt.subplots(n, 1, figsize=(11, 1.6 * n + 1.2), squeeze=False, sharex=True)
    ax_col = axes[:, 0]

    for ax, (glabel, names, sub_mean, physical) in zip(ax_col, groups, strict=False):
        for name in names:
            t, y = _time_and_data(pf, name, physical)
            cs = stats.get(name)
            center = cs.mean if cs is not None else float(np.mean(y))
            label = name
            if sub_mean:
                y = y - center
                label = f"{name}{-center:+.0f}"
                center = 0.0
            (line,) = ax.plot(t, y, linewidth=0.5, label=label)
            lim = _ts_range_limit(name, cs.ctype if cs is not None else "")
            if lim is not None:
                # Shaded "valid range" = mean +/- threshold, in the line's color.
                ax.axhspan(
                    center - lim, center + lim, color=line.get_color(), alpha=0.10, lw=0, zorder=0
                )
        ax.set_ylabel(glabel, fontsize=7)
        ax.legend(loc="upper right", fontsize=6, ncol=max(1, len(names)))
        ax.grid(True, alpha=0.3)
    for ax in ax_col[len(groups) :]:
        ax.set_visible(False)
    ax_col[min(len(groups), n) - 1].set_xlabel("t [s]")
    ax_col[0].set_title(title, fontsize=9)
    fig.tight_layout()
    return fig


def _spectra_valid_ranges(ax: Axes, stats: BenchStats) -> None:
    """Shade the checklist's spectral expectations onto the spectra axes.

    Green target boxes near 100 Hz mark the "rising to ~10^N counts^2/Hz"
    envelopes for the thermistor (~1e-1) and shear (~1e-2) channels; red dashed
    ceilings mark the accelerometer (<100) and P_dP (<10) upper bounds.
    """
    from matplotlib.patches import Rectangle

    types = {cs.ctype for cs in stats.channels.values()}
    # Target boxes: half a decade around the expected level, 50-150 Hz.
    targets = []
    if "therm" in types:
        targets.append((-1.0, "T target ~1e-1"))
    if "shear" in types:
        targets.append((-2.0, "sh target ~1e-2"))
    for exp, label in targets:
        lo, hi = 10.0 ** (exp - 0.5), 10.0 ** (exp + 0.5)
        ax.add_patch(
            Rectangle(
                (50.0, lo), 100.0, hi - lo, facecolor="tab:green", alpha=0.15, edgecolor="none",
                zorder=0,
            )
        )
        ax.text(150.0, hi, label, fontsize=6, color="tab:green", va="bottom", ha="right")
    # Ceilings: draw only the ones whose channels are present.
    ceilings = []
    if types & _VIB_TYPES:
        ceilings.append((_AXY_PSD_MAX, "Ax/Ay max"))
    if "P_dP" in stats.channels:
        ceilings.append((_PDP_PSD_MAX, "P_dP max"))
    x_right = 0.9 * stats.fs_fast / 2  # near Nyquist, clear of the top-left legend
    for yval, label in ceilings:
        ax.axhline(yval, color="tab:red", linestyle="--", linewidth=0.7, alpha=0.5, zorder=0)
        ax.text(
            x_right, yval, label, fontsize=6, color="tab:red", va="bottom", ha="right"
        )


def build_spectra_figure(pf: PFile, stats: BenchStats, title: str) -> Figure:
    """Log-log overlay of channel auto-spectra in counts^2/Hz.

    Overlays the checklist's spectral "valid ranges": green target boxes for the
    expected thermistor/shear noise envelopes near 100 Hz and red dashed ceilings
    for the accelerometer and pre-emphasized-pressure upper bounds.
    """
    fig, ax = plt.subplots(figsize=(11, 6))
    any_line = False
    ymax = 1e2
    for name in _spectrum_channels(pf):
        cs = stats.get(name)
        if cs is None or cs.psd is None or cs.freq is None:
            continue
        m = np.isfinite(cs.psd) & (cs.psd > 0) & (cs.freq > 0)
        if not np.any(m):
            continue
        ax.loglog(cs.freq[m], cs.psd[m], linewidth=0.7, label=name)
        any_line = True
        ymax = max(ymax, float(np.nanmax(cs.psd[m])))

    if any_line:
        _spectra_valid_ranges(ax, stats)
        ax.set_xlim(0.9 / stats.fft_sec, 1.1 * pf.fs_fast / 2)
        ax.set_ylim(1e-4, ymax * 1.5)
        ax.legend(loc="upper left", fontsize=7, ncol=2)
    else:
        ax.text(0.5, 0.5, "No spectra (record too short)", transform=ax.transAxes, ha="center")
    ax.set_xlabel("f [Hz]")
    ax.set_ylabel("counts$^2$ Hz$^{-1}$")
    ax.set_title(title, fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    return fig


def _ctclu_panels(pf: PFile) -> list[tuple[str | None, str, np.ndarray, np.ndarray]]:
    """(name, label, t, y) for each CT/CLTU channel present, JAC_C split into I/V.

    ``name`` is the stats key for a valid-range band, or None for the derived
    JAC_C I/V sub-channels (which have no single-channel threshold).
    """
    panels: list[tuple[str | None, str, np.ndarray, np.ndarray]] = []
    for name in ("Turbidity", "Chlorophyll", "JAC_T"):
        if name in pf.channels_raw:
            t, y = _time_and_data(pf, name, physical=False)
            panels.append((name, f"{name} [counts]", t, y))
    if "JAC_C" in pf.channels_raw:
        t = pf.t_slow
        raw = np.asarray(pf.channels_raw["JAC_C"], dtype=np.float64)
        n = min(len(t), len(raw))
        raw_i = np.floor(raw[:n] / 2**16)
        raw_v = np.mod(raw[:n], 2**16)
        panels.append((None, "JAC_C_I [counts]", t[:n], raw_i))
        panels.append((None, "JAC_C_V [counts]", t[:n], raw_v))
    return panels


def build_ctclu_figure(pf: PFile, stats: BenchStats, title: str) -> Figure | None:
    """Optional CT/CLTU time-series figure; None when no such channels exist.

    Draws each channel's checklist valid range (mean +/- threshold) as a shaded
    band, matching the main time-series figure.
    """
    panels = _ctclu_panels(pf)
    if not panels:
        return None
    n = len(panels)
    fig, axes = plt.subplots(n, 1, figsize=(11, 1.5 * n + 1.0), squeeze=False, sharex=True)
    ax_col = axes[:, 0]
    for ax, (name, label, t, y) in zip(ax_col, panels, strict=False):
        (line,) = ax.plot(t, y, linewidth=0.5)
        cs = stats.get(name) if name else None
        lim = _ts_range_limit(name, cs.ctype) if (name and cs is not None) else None
        if lim is not None and cs is not None:
            ax.axhspan(
                cs.mean - lim, cs.mean + lim, color=line.get_color(), alpha=0.10, lw=0, zorder=0
            )
        ax.set_ylabel(label, fontsize=7)
        ax.grid(True, alpha=0.3)
    ax_col[-1].set_xlabel("t [s]")
    ax_col[0].set_title(title, fontsize=9)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _checklist_text_figure(report: str) -> Figure:
    """Render the checklist report as a full-page monospace text figure.

    Used as the final page of the ``pdf-bundle`` output. Long lines (the trailing
    guidance note) are wrapped so they do not run off the page; the aligned
    table rows are all short and pass through untouched.
    """
    import textwrap

    lines: list[str] = []
    for ln in report.split("\n"):
        lines.extend(textwrap.wrap(ln, 108) if len(ln) > 108 else [ln])
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(
        0.04, 0.98, "\n".join(lines), family="monospace", fontsize=8, va="top", ha="left"
    )
    return fig


def _save_pdf_bundle(figs: list[Figure], path: Path, dpi: int) -> None:
    """Write all figures into one multi-page PDF, quick_bench-style.

    Written to a temp file and atomically renamed, so a failure mid-write never
    leaves a truncated PDF at ``path``.
    """
    from matplotlib.backends.backend_pdf import PdfPages

    tmp = path.with_name(path.name + ".tmp")
    try:
        with PdfPages(tmp) as pdf:
            for fig in figs:
                pdf.savefig(fig, dpi=dpi, bbox_inches="tight")
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()


def run_bench(
    path: str | Path,
    out_dir: str | Path | None = None,
    *,
    show: bool = False,
    sn: str | None = None,
    fft_sec: float = DEFAULT_FFT_SEC,
    dpi: int = 150,
    fmt: str = "png",
) -> tuple[BenchStats, list[CheckItem]]:
    """Run a bench test on one ``.p`` file.

    Reads with ``deconvolve=False`` (raw pre-emphasized counts), computes the
    statistics, evaluates the checklist (printed to stdout), and builds the
    figures.  When ``out_dir`` is given, writes ``QB_<SN>_<stem>_timeseries``,
    ``_spectra`` (and ``_ctclu`` when applicable) plus ``_checklist.txt``.
    ``fmt`` controls the figure files: ``png`` / ``pdf`` write one file per
    figure; ``both`` writes both; ``pdf-bundle`` writes a single multi-page
    ``QB_<SN>_<stem>.pdf`` with the checklist as a final page (quick_bench-style).
    When ``show`` is true, opens the figures interactively.

    Returns ``(stats, checklist_items)`` for scripting.
    """
    pf = PFile(path, deconvolve=False)
    stats = compute_bench_stats(pf, fft_sec=fft_sec)
    resolved_sn = sn or stats.sn or "___"
    # Sanitize for use in output file names (a serial number like "a/b" would
    # otherwise point the write at a nonexistent subdirectory).
    safe_sn = re.sub(r"[^\w.-]", "_", resolved_sn)

    stem = pf.filepath.stem
    header = (
        f"Bench test  {pf.filepath.name}\n"
        f"SN {resolved_sn}   {stats.start_time}   fs_fast={stats.fs_fast:.0f} Hz"
    )
    items = evaluate_checklist(stats)
    report = render_checklist_text(items, header)
    print(report)

    title_base = f"{pf.filepath.name}   SN {resolved_sn}"
    figs: dict[str, Figure] = {
        "timeseries": build_timeseries_figure(
            pf, stats, f"{title_base} — Time Series  (shaded = checklist valid range)"
        ),
        "spectra": build_spectra_figure(
            pf, stats, f"{title_base} — Spectra  (green = target band, red = ceiling)"
        ),
    }
    ctclu = build_ctclu_figure(pf, stats, f"{title_base} — CT/CLTU")
    if ctclu is not None:
        figs["ctclu"] = ctclu

    # try/finally so a save error (read-only dir, disk full, bad path) never
    # leaks the open figures across a batch run — the CLI catches per-file and
    # continues, so leaked figures would otherwise accumulate.
    try:
        if out_dir is not None:
            out = Path(out_dir)
            out.mkdir(parents=True, exist_ok=True)
            base = f"QB_{safe_sn}_{stem}"
            if fmt == "pdf-bundle":
                # Append the evaluated checklist as the final page of the bundle.
                checklist_fig = _checklist_text_figure(report)
                try:
                    _save_pdf_bundle([*figs.values(), checklist_fig], out / f"{base}.pdf", dpi)
                finally:
                    plt.close(checklist_fig)
            else:
                exts = ["png", "pdf"] if fmt == "both" else [fmt]
                for fig_name, fig in figs.items():
                    for ext in exts:
                        fig.savefig(
                            out / f"{base}_{fig_name}.{ext}", dpi=dpi, bbox_inches="tight"
                        )
            (out / f"{base}_checklist.txt").write_text(report)
            print(f"\nWrote figures and checklist to {out}/{base}*")

        if show:
            plt.show()  # pragma: no cover
    finally:
        if not show:
            for fig in figs.values():
                plt.close(fig)

    return stats, items

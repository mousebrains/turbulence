# Oct-2026, Claude and Pat Welch, pat@mousebrains.com
"""A bead's slow parallel response path, divided out of its chi (issue #252).

Bead T2005 (VMP SN194 T2 from 2021-06-29 to 2022-06-20) loses ~12% of its
amplitude through a path with a time constant of ~110 ms, which no pole model
produces. Measured from ln(S_bead/S_partner) between 0.125 and 12 Hz it fits

    H_slow(f) = (1 - beta) + beta / (1 + i 2 pi f tau_s)

    |H_slow(f)|^2 = (1 - beta)^2 + beta (2 - beta) / (1 + (2 pi f tau_s)^2)

with beta 0.124 and tau_s 110 ms common to its deployments (beta 0.122-0.128,
tau_s 105-112 ms per deployment with the bead's own fast pole fitted; beta
spans 0.09-0.13 across single-pole model choices). It is 1 at f -> 0 and falls
monotonically to (1 - beta)^2 above the corner at ~1-1.5 Hz. The chi spectra
(first bin 2 Hz at fft 256) sit almost entirely on that plateau, so in-band the
path is a nearly flat gain of 0.77-0.85 and the bead's chi reads ~20% low.

Applied like :mod:`odas_tpw.chi.fp07_pair`, by EQUALIZING THE DATA: ``l4_chi``
divides the probe's observed AND noise spectra by ``|H_slow|^2`` before chi is
computed; the model, ``H2`` and the variance correction stay on the default FP07
model. Unlike fp07_pair's factor it is applied in every bin, UNWEIGHTED: the
noise spectrum is unattenuated electronics noise, so ``S/e = H2*B + N/e`` is
exact, and here ``1/e <= (1 - beta)^-2`` (~1.3) and nearly flat, so the
large-``1/e``-at-the-noise-floor hazard ``fp07_pair.signal_weighted`` exists for
does not arise. Dividing S and N by one e keeps each bin's S/N, so the detection
gate, the band and K_max are unchanged; chi rises by ~1/e over the band.

What it rests on, and what it needs:

* **It is relative to the bead's calibrated DC level.** ``|H_slow(0)|^2 = 1``:
  an in-situ calibration (``fp07.calibrate: true``, or ``fp07-cal`` patched
  files) has leverage far below 0.1 Hz, so it fixes the DC gain and the path
  is what remains above the corner. Products made from the factory
  coefficients are not what the correction was measured on.
* **A1: the partner is the norm.** T2005 is corrected to the fleet norm: T1592
  reads like T1593 on the same port and every other pair checked is flat.
  Whether the norm itself has a slow path is untested (the JAC reference
  cannot test it: its own response time sits on tau_s). If it had one, the
  bead would be under-corrected, never over-corrected.
* **The factor is in frequency, not wavenumber.** tau_s was measured at VMP
  fall speeds and is held fixed in Hz; nothing scales it with speed.
* **Never with ``fp07_tau_scale`` or ``fp07_response``.** Both were fitted on
  spectra that already contain the path (ARCTERX-2022's tau scales 0.30/0.75;
  an fp07-pair correction's ``ln e`` holds a share of it), so stacking either
  on this corrects the bead twice.

A per-file guard (:func:`window_log_ratios`, :func:`guard_delta`) measures the
change of the median ln(S_bead/S_partner) from 0.25 to 2 Hz on the file's own
gradient time series, which the chi spectra cannot see, and compares it with
the configured path's prediction. It catches a probe swap without a config
update (the sign flips) and a component that has faded. It warns; it does not
refuse.
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from itertools import pairwise
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

#: Product attribute prefix.
ATTR = "fp07_slow_response"
#: An FP07 bead serial as .p configurations write it (``sn = T2005``).
BEAD_KEY = re.compile(r"^T\d+$")
#: Keys of one span. All four are required: a correction without a span would
#: follow the bead into deployments it was never measured on.
SPAN_KEYS = ("beta", "tau_s_ms", "first", "last")

# The guard statistic, as issue #252 measured it: Welch over 8-s Hann segments
# (50% overlap, mean removed) of the gradient, in windows of up to 32 s of a
# profile (a last part of >= 12 s is kept), the median over windows of
# ln(S_bead/S_partner) at each frequency, and the change from 0.25 to 2 Hz.
GUARD_F_LO = 0.25  # Hz
GUARD_F_HI = 2.0  # Hz
GUARD_SEGMENT_S = 8.0
GUARD_WINDOW_S = 32.0
GUARD_MIN_WINDOW_S = 12.0
GUARD_MIN_SEGMENTS = 2
#: |measured - predicted| above which the guard warns [ln of a power ratio].
#: Set from 85 healthy T2005 files (SUNRISE 2021 WaltonSmith 38, ARCTERX-2022
#: Interior 30 and Wake 2, SUNRISE 2022 PointSur before 06-20 10, SUNRISE 2021
#: Pelican after the swap 5): their per-file change minus the slow-only
#: prediction (-0.159 for beta 0.124, tau_s 110 ms) spans -0.08..+0.01 (5-95%),
#: at most 0.106 in magnitude except one ARCTERX-2022 Interior file at 0.26. The
#: measured change also holds the pair's ordinary FP07 pole difference (~-0.03 at
#: 2 Hz for T1592/T2005), which the prediction does not include. A pair without
#: the path (the pre-swap T1592/T1593) or a faded component sits ~+0.18 off.
SLOW_GUARD_MAX = 0.10
#: Windows a verdict needs. SLOW_GUARD_MAX was set on whole .p FILES (median
#: 78-148 windows per deployment). One window's ln ratio at 2 Hz scatters by
#: ~0.17 (1.4826 MAD over SUNRISE 2021 WaltonSmith SR1B_008's 83 windows; 0.014
#: at 0.25 Hz), so a median over 40 has a standard error of ~0.03 and the limit
#: sits ~2 of them beyond the healthy files' typical offset (-0.035). Random
#: 20-window subsets of that file strayed by more than 0.10 in 3% of draws,
#: 40-window subsets in 0.1%. A perturb profile is one to a few windows (a 20-s
#: SUNRISE cast is one), so perturb pools a file's profiles before judging.
SLOW_GUARD_MIN_WINDOWS = 40

MODEL_TEXT = (
    "|H_slow(f)|^2 = |(1 - beta) + beta/(1 + i 2 pi f tau_s)|^2, f in Hz (speed-independent), "
    "beta dimensionless, tau_s in ms in the attribute name; each listed probe's observed AND "
    "noise spectra were divided by it in every bin (unweighted) before chi. spec_gradT and "
    "spec_noise here are raw. Relative to the bead's calibrated DC level (|H_slow(0)| = 1)"
)
GUARD_TEXT = (
    f"delta = median over windows (<= {GUARD_WINDOW_S:g} s) of ln(S_bead/S_partner) at "
    f"{GUARD_F_HI:g} Hz minus at {GUARD_F_LO:g} Hz, Welch over {GUARD_SEGMENT_S:g}-s Hann "
    "segments (50% overlap) of the gradient; natural-log units. predicted = ln|H_slow|^2 at "
    f"{GUARD_F_HI:g} Hz minus at {GUARD_F_LO:g} Hz (bead minus partner when both are corrected)"
)


def slow_h2(f: Any, beta: float, tau_s: float) -> np.ndarray:
    """``|H_slow(f)|^2 = |(1 - beta) + beta / (1 + i 2 pi f tau_s)|^2``.

    Closed form ``(1 - beta)^2 + beta (2 - beta) / (1 + (2 pi f tau_s)^2)``:
    1 at ``f = 0``, ``(1 - beta)^2`` as ``f -> inf``, monotone between. *f* in
    Hz, *tau_s* in seconds.
    """
    x2 = (2.0 * np.pi * np.asarray(f, dtype=np.float64) * float(tau_s)) ** 2
    b = float(beta)
    return np.asarray((1.0 - b) ** 2 + b * (2.0 - b) / (1.0 + x2), dtype=np.float64)


def _iso(t: float) -> str:
    """UTC ISO-8601, to the microsecond when the time has a fraction of a second.

    Whole seconds print as seconds. A fraction must survive: :meth:`SlowSpan.config_form`
    is what workers parse back, and the hash keeps the fraction, so truncating
    it here would apply a span that differs from the one hashed.
    """
    t = float(t)
    spec = "seconds" if t == math.floor(t) else "microseconds"
    return datetime.fromtimestamp(t, UTC).isoformat(timespec=spec)


@dataclass(frozen=True)
class SlowSpan:
    """One bead's slow path over one stretch of time (epoch seconds, inclusive)."""

    bead: str
    beta: float
    tau_s_ms: float
    first: float
    last: float

    @property
    def tau_s(self) -> float:
        """The slow time constant [s]."""
        return self.tau_s_ms / 1000.0

    def factor(self, F: Any) -> np.ndarray:
        """``|H_slow(F)|^2`` at frequencies *F* [Hz]: what l4_chi divides by."""
        return slow_h2(F, self.beta, self.tau_s)

    def contains(self, t: float) -> bool:
        return self.first <= t <= self.last

    def ln_change(self) -> float:
        """ln|H_slow|^2 at GUARD_F_HI minus at GUARD_F_LO: the guard's prediction."""
        lo, hi = self.factor(np.array([GUARD_F_LO, GUARD_F_HI]))
        return float(np.log(hi) - np.log(lo))

    def span_text(self) -> str:
        return f"{_iso(self.first)}..{_iso(self.last)}"

    def hash_form(self) -> dict[str, float]:
        """What changes chi: the numbers and the span, the span as epoch seconds."""
        return {"beta": self.beta, "tau_s_ms": self.tau_s_ms,
                "first": self.first, "last": self.last}

    def config_form(self) -> dict[str, Any]:
        """A plain mapping that parses back to this span (ISO times, no YAML types)."""
        return {"beta": self.beta, "tau_s_ms": self.tau_s_ms,
                "first": _iso(self.first), "last": _iso(self.last)}


def _number(value: Any, where: str) -> float:
    # bool is an int subclass; `beta: yes` is a typo, not 1.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where} must be a number, got {value!r}")
    v = float(value)
    if not math.isfinite(v):
        raise ValueError(f"{where} must be finite, got {value!r}")
    return v


def parse_slow_response(
    cfg: Any, where: str = "fp07_slow_response"
) -> dict[str, list[SlowSpan]]:
    """Validate a ``{bead: span | [spans]}`` mapping; spans sorted by ``first``.

    Each span is ``{beta, tau_s_ms, first, last}``: ``0 <= beta < 1``,
    ``tau_s_ms`` finite and > 0, ``first < last`` as ISO-8601 times (quoted or
    unquoted; a time without a zone, or a bare date, is UTC midnight). A file is
    inside a span when ``first <= start_time <= last``. A bead's spans must not
    overlap. ``None`` or ``{}`` is no correction. Raises ValueError naming the
    offending key.
    """
    from odas_tpw.perturb.probe_exclusion import _parse_time

    if cfg is None:
        return {}
    if not isinstance(cfg, Mapping):
        raise ValueError(
            f"{where}: must be a mapping of bead serial -> {{beta, tau_s_ms, first, last}} "
            f"(or a list of them), got {cfg!r}"
        )
    out: dict[str, list[SlowSpan]] = {}
    for bead, raw in cfg.items():
        if not isinstance(bead, str) or not BEAD_KEY.match(bead.strip()):
            raise ValueError(
                f"{where}: key {bead!r} is not an FP07 bead serial (T followed by digits, "
                "e.g. T2005, as the .p configuration writes it)"
            )
        bead = bead.strip()
        entries = [raw] if isinstance(raw, Mapping) else raw
        if not isinstance(entries, list) or not entries:
            raise ValueError(
                f"{where}.{bead}: must be a mapping {{beta, tau_s_ms, first, last}} or a "
                f"non-empty list of them, got {raw!r}"
            )
        spans = []
        for n, entry in enumerate(entries):
            here = f"{where}.{bead}" + (f"[{n}]" if isinstance(raw, list) else "")
            if not isinstance(entry, Mapping):
                raise ValueError(f"{here}: must be a mapping, got {entry!r}")
            unknown = sorted(str(k) for k in entry if k not in SPAN_KEYS)
            if unknown:
                raise ValueError(f"{here}: unknown key(s) {unknown}; valid keys: {list(SPAN_KEYS)}")
            missing = [k for k in SPAN_KEYS if entry.get(k) is None]
            if missing:
                raise ValueError(
                    f"{here}: missing {missing}. The span (first, last) is mandatory: a "
                    "correction must not follow the bead into deployments it was not "
                    "measured on"
                )
            beta = _number(entry["beta"], f"{here}.beta")
            if not 0.0 <= beta < 1.0:
                raise ValueError(f"{here}.beta must be in [0, 1), got {entry['beta']!r}")
            tau = _number(entry["tau_s_ms"], f"{here}.tau_s_ms")
            if tau <= 0:
                raise ValueError(f"{here}.tau_s_ms must be > 0 (milliseconds), got {tau!r}")
            first = _parse_time(entry["first"], here, "first")
            last = _parse_time(entry["last"], here, "last")
            assert first is not None and last is not None  # checked above
            if not first < last:
                raise ValueError(
                    f"{here}: first ({_iso(first)}) must be before last ({_iso(last)})"
                )
            spans.append(SlowSpan(bead, beta, tau, first, last))
        spans.sort(key=lambda s: s.first)
        for a, b in pairwise(spans):
            if b.first <= a.last:
                raise ValueError(
                    f"{where}.{bead}: spans {a.span_text()} and {b.span_text()} overlap; a "
                    "file must select one at most"
                )
        if bead in out:
            raise ValueError(f"{where}: bead {bead} is listed twice")
        out[bead] = spans
    return out


def canonical_for_hash(cfg: Any, where: str = "fp07_slow_response") -> dict[str, list[dict]]:
    """Hash form: sorted beads, float values, spans as epoch seconds.

    A quoted and an unquoted time hash alike, and no YAML date object reaches
    json. An empty mapping is ``{}``; callers hash that as absent.
    """
    parsed = parse_slow_response(cfg, where)
    return {bead: [s.hash_form() for s in parsed[bead]] for bead in sorted(parsed)}


def config_form(cfg: Any, where: str = "fp07_slow_response") -> dict[str, list[dict]]:
    """Validated plain form (ISO strings, builtin types): safe to pickle and re-parse."""
    parsed = parse_slow_response(cfg, where)
    return {bead: [s.config_form() for s in parsed[bead]] for bead in sorted(parsed)}


# ---------------------------------------------------------------------------
# Matching a file
# ---------------------------------------------------------------------------


@dataclass
class SlowSelection:
    """What the configuration says about one file's thermistors."""

    applied: dict[int, SlowSpan] = field(default_factory=dict)  # channel index -> span
    outside: dict[int, str] = field(default_factory=dict)  # index -> bead outside every span
    t: float = math.nan  # the file's start time [epoch s]
    serials: list[str] = field(default_factory=list)  # every thermistor's bead, channel order


def select(
    spans: Mapping[str, Sequence[SlowSpan]], serials: Sequence[str], t: float
) -> SlowSelection:
    """Match configured beads against a file's bead *serials* at its start time *t*.

    *serials* are per thermistor channel, from :func:`odas_tpw.chi.fp07_pair.pair_identity`.
    A bead absent from the file leaves it untouched. A bead on the file outside
    every span is recorded in ``outside`` and not corrected. A bead on two
    channels INSIDE one of its spans is an error: which channel carries the
    path cannot be known. Outside every span nothing is applied, so it is only
    recorded (on every such channel): a configuration that writes one serial
    on both beads is a copy-paste placeholder (``fp07cal`` treats a shared
    serial as one), and refusing it there would cost the file its chi for
    nothing.
    """
    have = [str(s).strip() for s in serials]
    sel = SlowSelection(t=float(t), serials=have)
    for bead, bead_spans in spans.items():
        hits = [i for i, s in enumerate(have) if s == bead]
        if not hits:
            continue
        span = next((s for s in bead_spans if s.contains(t)), None)
        if span is None:
            for i in hits:
                sel.outside[i] = bead
            continue
        if len(hits) > 1:
            raise ValueError(
                f"fp07_slow_response: bead {bead} is on {len(hits)} thermistor channels of "
                f"this file, inside its span {span.span_text()}; its configuration names one "
                "bead twice, so which channel carries the slow path cannot be known"
            )
        sel.applied[hits[0]] = span
    return sel


def apply_to_l3(l3: Any, selection: SlowSelection, n_probes: int) -> None:
    """Attach each corrected probe's ``F -> |H_slow(F)|^2`` to ``l3.slow_factor``.

    ``l4_chi`` divides that probe's observed and noise spectra by it. Nothing
    stored in *l3* changes, so the product's spectra stay raw.
    """
    factors: list = [None] * n_probes
    for ci, span in selection.applied.items():
        factors[ci] = span.factor
    l3.slow_factor = factors


# ---------------------------------------------------------------------------
# The guard
# ---------------------------------------------------------------------------


def _runs(ok: np.ndarray, section: np.ndarray) -> list[tuple[int, int]]:
    """Half-open index ranges where *ok* holds and the section number is constant."""
    if ok.size == 0:
        return []
    key = np.where(ok, section, 0)
    edges = np.concatenate([[0], np.flatnonzero(np.diff(key) != 0) + 1, [ok.size]])
    return [(int(a), int(b)) for a, b in pairwise(edges) if ok[a]]


def window_log_ratios(
    x_bead: np.ndarray, x_partner: np.ndarray, section_number: np.ndarray, fs: float
) -> tuple[np.ndarray, np.ndarray]:
    """Per-window ln(S_bead/S_partner) at GUARD_F_LO and GUARD_F_HI.

    *x_bead*, *x_partner* are the two probes' gradient time series on one time
    base, *section_number* marks the profile (0 = excluded). Each run of one
    section with finite samples on both probes is cut into windows of
    GUARD_WINDOW_S (a last part of at least GUARD_MIN_WINDOW_S is kept), and
    each window's spectra are Welch sums over GUARD_SEGMENT_S Hann segments
    with 50% overlap, mean removed.
    """
    xb = np.asarray(x_bead, dtype=np.float64)
    xp = np.asarray(x_partner, dtype=np.float64)
    sec = np.asarray(section_number)
    nseg = round(GUARD_SEGMENT_S * fs)
    n_win = round(GUARD_WINDOW_S * fs)
    n_min = round(GUARD_MIN_WINDOW_S * fs)
    if nseg < 8 or xb.size != xp.size or xb.size != sec.size:
        return np.zeros(0), np.zeros(0)
    freq = np.fft.rfftfreq(nseg, 1.0 / fs)
    k_lo = int(np.argmin(np.abs(freq - GUARD_F_LO)))
    k_hi = int(np.argmin(np.abs(freq - GUARD_F_HI)))
    win = np.hanning(nseg)
    ok = (sec > 0) & np.isfinite(xb) & np.isfinite(xp)
    lo: list[float] = []
    hi: list[float] = []
    for a, b in _runs(ok, sec):
        for w0 in range(a, b, n_win):
            w1 = min(w0 + n_win, b)
            if w1 - w0 < n_min:
                continue
            starts = range(w0, w1 - nseg + 1, nseg // 2)
            if len(starts) < GUARD_MIN_SEGMENTS:
                continue
            pb = np.zeros(freq.size)
            pp = np.zeros(freq.size)
            for s in starts:
                yb = xb[s:s + nseg]
                yp = xp[s:s + nseg]
                pb += np.abs(np.fft.rfft((yb - yb.mean()) * win)) ** 2
                pp += np.abs(np.fft.rfft((yp - yp.mean()) * win)) ** 2
            if pb[k_lo] > 0 and pp[k_lo] > 0 and pb[k_hi] > 0 and pp[k_hi] > 0:
                lo.append(float(np.log(pb[k_lo] / pp[k_lo])))
                hi.append(float(np.log(pb[k_hi] / pp[k_hi])))
    return np.asarray(lo), np.asarray(hi)


def guard_delta(lo: np.ndarray, hi: np.ndarray) -> float:
    """Median over windows of the ratio at GUARD_F_HI minus at GUARD_F_LO; NaN if none."""
    if np.size(lo) == 0:
        return math.nan
    return float(np.median(hi) - np.median(lo))


def guard_verdict(measured: float, predicted: float, n_windows: int) -> str:
    """``"ok"``, ``"WARNING: ..."`` or ``"no verdict: ..."`` for one bead.

    A verdict needs SLOW_GUARD_MIN_WINDOWS windows; then the change measured
    on the data must lie within SLOW_GUARD_MAX of the configured path's.
    """
    if not np.isfinite(measured) or n_windows < SLOW_GUARD_MIN_WINDOWS:
        return (
            f"no verdict: {n_windows} window(s), fewer than {SLOW_GUARD_MIN_WINDOWS} "
            "(perturb pools a .p file's profiles)"
        )
    if abs(measured - predicted) > SLOW_GUARD_MAX:
        return (
            f"WARNING: the {GUARD_F_LO:g}->{GUARD_F_HI:g} Hz change {measured:+.3f} over "
            f"{n_windows} windows differs from the configured path's {predicted:+.3f} by more "
            f"than {SLOW_GUARD_MAX:g}: a swapped probe, a faded or changed path, or a "
            "different partner. Applied anyway"
        )
    return "ok"


def product_summary(attrs: Mapping[str, Any]) -> dict[str, Any] | None:
    """What one chi product's ``fp07_slow_response_*`` attributes say; None if absent.

    ``{"serials": [every thermistor's bead], "outside": [bead, ...], "applied":
    [{"bead", "channel", "measured", "predicted", "n_windows", "verdict", "lo",
    "hi"}, ...]}`` -- plain builtins, for a run summary (perturb, rsi-tpw) to
    pool. ``lo``/``hi`` are the guard's per-window ln ratios.
    """
    serials = attrs.get(f"{ATTR}_serials")
    if serials is None:
        return None
    out: dict[str, Any] = {"serials": [s for s in str(serials).split("/") if s],
                           "outside": [], "applied": []}
    prefix = f"{ATTR}_bead_"
    for key in sorted(attrs):
        if not key.startswith(prefix):
            continue
        ch, bead = key[len(prefix):], str(attrs[key])
        if f"{ATTR}_beta_{ch}" not in attrs:
            out["outside"].append(bead)
            continue

        def arr(name: str, _ch: str = ch) -> list[float]:
            v = attrs.get(f"{ATTR}_guard_lnr_{name}_{_ch}")
            return [] if v is None else [float(x) for x in np.atleast_1d(v)]

        out["applied"].append({
            "bead": bead, "channel": ch,
            "measured": float(attrs.get(f"{ATTR}_delta_measured_{ch}", math.nan)),
            "predicted": float(attrs.get(f"{ATTR}_delta_predicted_{ch}", math.nan)),
            "n_windows": int(attrs.get(f"{ATTR}_guard_windows_{ch}", 0)),
            "verdict": str(attrs.get(f"{ATTR}_guard_{ch}", "")),
            "lo": arr("lo"), "hi": arr("hi"),
        })
    return out

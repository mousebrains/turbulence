# Oct-2026, Claude and Pat Welch, pat@mousebrains.com
"""The FP07 calibration sheet: one hand-maintained YAML file, two generated views.

``microstructure_sensors/fp07_calibrations.yaml`` is the single source of truth
for the in-situ calibrations of individual FP07 beads.  Two views are
GENERATED from it and are never edited by hand:

* ``fp07_calibrations.md`` --- the human-readable sheet, one section per bead,
  with its deployments and its prior for a fresh deployment;
* ``fp07_calibrations.csv`` --- one row per fit, for spreadsheets.

``fp07-cal registry render`` writes both.  ``fp07-cal registry check`` validates
the YAML and fails when either view differs from what it renders, and a test
does the same, so the three cannot drift apart unnoticed.  Rendering is
deterministic (fixed ordering and number formats) for exactly that reason.

An entry is one bead x one deployment, fitted by ``fp07-cal fit`` pooled over
the deployment.  Its coefficients are in the ODAS/perturb config form that
``fp07-cal patch`` writes and ``rsi/channels.py::convert_therm`` reads::

    1/T_K = 1/t_0 + L/beta_1 + L^2/beta_2 + L^3/beta_3

with ``L = ln(R_T/R_0)`` defined by the channel's half-bridge constants (see
:mod:`odas_tpw.fp07cal.logr`), which therefore travel with every entry.

The loader is strict.  A record whose job is to be believed later has to be
refused when it is wrong now: an unknown or missing key, a bad serial, a first
date after the last, an order that disagrees with the number of beta terms, or a
second entry for the same bead and deployment each raise a named
:class:`RegistryError` subclass, with the entry it came from.

The prior (:meth:`Registry.prior_for`) is deliberately simple, because the
evidence behind it is a handful of beads (issue #189): the coefficients of the
bead's widest-range calibration not flagged ``short``, ``fast-descent`` or
``episode``, the sheet's measured transfer uncertainties, the bead's calibrated
temperature range, and how far its most recent calibration sits from the prior.
The loader also refuses coefficients that cannot describe an FP07 over the
stated range (:class:`PlausibilityError`).  It returns exactly what a ``t_0``-only refit
with the shape held fixed needs --- the fixed higher-order terms, the bridge
constants that define their ``L``, a starting ``t_0`` with its uncertainty, and
where the shape is constrained.  :mod:`odas_tpw.fp07cal.prior` does that refit
inside ``fp07-cal fit`` (``fit.prior``); an entry recorded from it has
``order_choice: prior`` and names its source in ``fit.shape_from``, and never
supplies a prior itself.

:func:`add_entry` (``fp07-cal registry add``) records a fit: it inserts the
entry as text in the sheet's own layout rather than re-dumping the YAML, so the
hand-written comments and folded prose survive, and validates the whole sheet
before writing.
"""

from __future__ import annotations

import contextlib
import csv
import io
import math
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from odas_tpw.fp07cal.logr import temperature

SCHEMA = "fp07-calibrations/1"
"""The ``schema`` value the sheet must carry."""

YAML_NAME = "fp07_calibrations.yaml"
RENDER_HINT = "run `fp07-cal registry render`"

BRIDGE_KEYS = ("a", "b", "g", "e_b", "adc_fs", "adc_bits")
BETA_KEYS = ("beta_1", "beta_2", "beta_3")

# A Rockland FP07 serial: T and a number without a leading zero (so T01592 and
# T1592 cannot be two beads).  The placeholder "T" (no digits), which configs
# carry when nobody filled the serial in, is NOT an identity and is refused.
_BEAD_RE = re.compile(r"T[1-9][0-9]*")
_INSTRUMENT_SERIAL_RE = re.compile(r"[0-9]+")
_REFERENCE_SERIAL_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_CHANNEL_RE = re.compile(r"T[0-9]")
_METHOD_RE = re.compile(r"[a-z0-9][a-z0-9-]*")
_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
_DATETIME_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")

# |beta| at or above this is the .p-config way of switching a term off
# (patch.NEUTRAL = 1e30).  In the sheet a term that is off is simply absent.
_NEUTRAL_BETA = 1e29

# Fewer blocks than this and a standard deviation of block offsets says nothing.
MIN_BLOCKS_FOR_SCATTER = 3

# Physical plausibility.  t_0 is a temperature in KELVIN near the bead's R_0
# point; a value outside this band is almost always degC typed for K.
T0_RANGE_K = (250.0, 320.0)
# convert_therm clips the bridge output Z to +/-0.6, so L = ln((1-Z)/(1+Z)) can
# never leave +/-ln(1.6/0.4): a fitted range that needs |L| beyond it is not a
# range this bead can report.
L_CLIP = math.log(1.6 / 0.4)

# Flags that disqualify a calibration as the source of a prior: its shape or
# its level is not a fair sample of the bead (too little data, the reference's
# own response in the curve, or a fault inside the fit).  The other flags
# describe how a sound fit was obtained and do not disqualify it.
DISQUALIFYING_FLAGS = ("short", "fast-descent", "episode")
FLAGS: dict[str, str] = {
    "lag-gate-overridden": (
        "fp07-cal's lag-trust gate, tuned for gliders, rejected the lag. It rejects "
        "every VMP lag (issue #238), so the fit was kept. `fp07-cal patch` would "
        "refuse the original record; use the coefficients here directly."
    ),
    "export-gate-overridden": (
        "fp07-cal's 1 K export gate refused the first fit, and the deployment was "
        "refitted with the gate raised. The notes say why."
    ),
    "reference-edited": (
        "Reference samples were blanked beyond fp07-cal's own QC. The notes say "
        "which and why."
    ),
    "screened": (
        "Part of the deployment was left out of the fit. The dates are those of "
        "the data actually used."
    ),
    "episode": (
        "The fitted data contain an offset episode, typically an electrical fault. "
        "It inflates the scatter; the notes say when and how large."
    ),
    "short": (
        "Too little data (files, profiles or days) for the block scatter, or a "
        "drift, to mean much."
    ),
    "fast-descent": (
        "Fast VMP descents through strong gradients, where the reference's own "
        "response biases the pooled curve. The block scatter is several times the "
        "quiet-water figure."
    ),
}

ORDER_CHOICES: dict[str, str] = {
    "auto": "least held-out error, split on temperature (fp07-cal `fit.order: auto`)",
    "fixed": "set by hand",
    "prior": (
        "the shape (beta terms) held fixed from the bead's calibration named in "
        "`fit.shape_from` and only t_0 fitted (fp07-cal `fit.prior`): kept over the full "
        "fit by held-out error within a tie band, a range narrower than the shape span, "
        "or no plausible full fit, or forced by `fit.prior.mode: offset`"
    ),
}

PRIOR_RULE = (
    "The coefficients, shape and t_0 alike, come from the bead's widest-range "
    "unflagged calibration: the widest fitted temperature range among calibrations "
    "not flagged `short`, `fast-descent` or `episode` (ties go to the more recent), "
    "counting only calibrations that ended by `as_of` when one is given. A calibration "
    "whose shape was itself held fixed from a prior (order `prior`) never supplies one: "
    "its shape was not fitted over its range. With a target "
    "instrument (serial; 'SN' and leading zeros are ignored) and optionally a channel "
    "(any case), the widest unflagged calibration on that instrument and channel is "
    "preferred. If there is none, the bead's widest unflagged calibration is used, and "
    "the cross-instrument term is added to t_0 in quadrature. If the bead has no "
    "unflagged calibration, its widest-range calibration is used (on the target when "
    "possible); the prior says so and adds that fit's held-out error to the shape term "
    "in quadrature. t_0 carries the sheet's carried-over offset uncertainty, and the "
    "shape the sheet's shape uncertainty. The prior reports the union of the bead's "
    "fitted ranges (outside the source's own range the shape is extrapolated) and how "
    "far the most recent calibration is from it, at the center of their common fitted "
    "range."
)


# --------------------------------------------------------------------- errors
class RegistryError(ValueError):
    """The calibration sheet cannot be trusted as written."""


class UnknownKeyError(RegistryError):
    """A key the schema does not define (a typo is the usual cause)."""


class MissingKeyError(RegistryError):
    """A required key is absent."""


class SerialPatternError(RegistryError):
    """A bead, instrument, channel or reference serial has the wrong form."""


class DateOrderError(RegistryError):
    """Dates out of order, or one bead (or one channel) in two places at once."""


class OrderMismatchError(RegistryError):
    """``fit.order`` disagrees with the coefficients or with how it was chosen."""


class DuplicateEntryError(RegistryError):
    """Two entries for the same bead and deployment."""


class PlausibilityError(RegistryError):
    """Coefficients that cannot describe an FP07 over the stated range."""


# ---------------------------------------------------------------- data model
@dataclass(frozen=True)
class Instrument:
    model: str
    serial: str
    channel: str
    platform: str | None = None

    def label(self) -> str:
        return f"{self.model} SN {self.serial} {self.channel}"

    def electronics(self) -> tuple[str, str]:
        """What defines the bridge: instrument serial and channel."""
        return (self.serial, self.channel)


@dataclass(frozen=True)
class Reference:
    kind: str
    serial: str | None
    cal_date: date | None
    note: str | None = None


@dataclass(frozen=True)
class FitSettings:
    method: str
    pooled: bool
    n_files: int
    n_profiles: int
    n_pairs: int
    order: int
    order_choice: str
    held_out_by_order_mK: dict[int, float] | None
    robust: bool
    geometry: dict[str, float] | None
    lag_s: float
    reference_pole_s: float
    software_version: str
    shape_from: str | None = None  # order_choice "prior": the entry the shape came from


@dataclass(frozen=True)
class Uncertainty:
    fit_rms_mK: float  # over the pairs the robust fit KEPT
    held_out_mK: float
    block_sd_mK: float | None
    n_blocks: int
    rms_all_mK: float | None = None  # over EVERY valid pair (fits made with fit.prior)


@dataclass(frozen=True)
class Calibration:
    """One bead x one deployment fit."""

    bead: str
    deployment: str
    instrument: Instrument
    first: datetime
    last: datetime
    reference: Reference
    fit: FitSettings
    coefficients: dict[str, float]
    bridge: dict[str, float]
    uncertainty: Uncertainty
    valid_T: tuple[float, float]
    flags: tuple[str, ...]
    notes: str

    @property
    def key(self) -> tuple[str, str]:
        return (self.bead, self.deployment)

    @property
    def poly(self) -> np.ndarray:
        """``[a0, a1, ...]`` with ``1/T_K = sum a_i L^i`` (the fp07-cal form)."""
        c = self.coefficients
        return np.array(
            [1.0 / c["t_0"]] + [1.0 / c[k] for k in BETA_KEYS if k in c], dtype=np.float64
        )

    @property
    def cal_age_years(self) -> float | None:
        """The reference's calibration age at the start of the deployment."""
        if self.reference.cal_date is None:
            return None
        return (self.first.date() - self.reference.cal_date).days / 365.25

    @property
    def duration_days(self) -> float:
        return (self.last - self.first).total_seconds() / 86400.0

    def temperature(self, L: np.ndarray) -> np.ndarray:
        """degC from ``L`` with this entry's coefficients."""
        return temperature(L, self.poly)

    def sort_key(self) -> tuple:
        return (_bead_key(self.bead), self.first, self.last, self.deployment)

    def to_dict(self) -> dict[str, Any]:
        """The entry as plain data, in the YAML's structure (for ``show --json``)."""
        out: dict[str, Any] = {
            "bead": self.bead,
            "deployment": self.deployment,
            "instrument": {
                "model": self.instrument.model,
                "serial": self.instrument.serial,
                "channel": self.instrument.channel,
            },
            "first": _iso(self.first),
            "last": _iso(self.last),
            "reference": {
                "kind": self.reference.kind,
                "serial": self.reference.serial,
                "cal_date": None if self.reference.cal_date is None
                else self.reference.cal_date.isoformat(),
                "cal_age_years": self.cal_age_years,
            },
            "fit": {
                "method": self.fit.method,
                "pooled": self.fit.pooled,
                "n_files": self.fit.n_files,
                "n_profiles": self.fit.n_profiles,
                "n_pairs": self.fit.n_pairs,
                "order": self.fit.order,
                "order_choice": self.fit.order_choice,
                "held_out_by_order_mK": None if self.fit.held_out_by_order_mK is None
                else {str(k): v for k, v in self.fit.held_out_by_order_mK.items()},
                "robust": self.fit.robust,
                "geometry": self.fit.geometry,
                "lag_s": self.fit.lag_s,
                "reference_pole_s": self.fit.reference_pole_s,
                "software_version": self.fit.software_version,
            },
            "coefficients": dict(self.coefficients),
            "bridge": dict(self.bridge),
            "uncertainty": {
                "fit_rms_mK": self.uncertainty.fit_rms_mK,
                "held_out_mK": self.uncertainty.held_out_mK,
                "block_sd_mK": self.uncertainty.block_sd_mK,
                "n_blocks": self.uncertainty.n_blocks,
                **({} if self.uncertainty.rms_all_mK is None
                   else {"rms_all_mK": self.uncertainty.rms_all_mK}),
            },
            "valid_T_degC": list(self.valid_T),
            "flags": list(self.flags),
            "notes": self.notes,
        }
        if self.instrument.platform is not None:
            out["instrument"]["platform"] = self.instrument.platform
        if self.reference.note is not None:
            out["reference"]["note"] = self.reference.note
        if self.fit.shape_from is not None:
            out["fit"]["shape_from"] = self.fit.shape_from
        return out


@dataclass(frozen=True)
class Transfer:
    """How far a calibration carries to another deployment (sheet-level)."""

    offset_mK: float
    shape_mK: float
    shape_span_K: tuple[float, float]
    cross_instrument_mK: float
    settling_mK: tuple[float, float]
    settling_days: tuple[float, float]
    source: str
    evidence: dict[str, str]
    reference_caveat: str


@dataclass(frozen=True)
class Prior:
    """A starting calibration for a fresh deployment of one bead.

    Everything a ``t_0``-only refit with the shape held fixed needs:
    :attr:`fixed_terms` (the shape), :attr:`bridge` (the ``L`` it belongs to),
    :attr:`coefficients` ``["t_0"]`` with :attr:`t_0_uncertainty_mK`, and
    :attr:`valid_T` (where the shape is constrained).
    """

    bead: str
    source: Calibration
    considered: tuple[Calibration, ...]
    target: Instrument | None
    cross_instrument: bool  # True when the cross-instrument term was applied
    fallback: bool  # True when no unflagged calibration existed (source is flagged)
    offset_mK: float  # the sheet's carried-over t_0 offset
    cross_instrument_mK: float  # the sheet's cross-instrument term (applied or not)
    t_0_uncertainty_mK: float  # what applies to this prior's t_0
    sheet_shape_mK: float  # the sheet's shape term
    shape_mK: float  # what applies to this prior's shape (widened on a fallback)
    shape_span_K: tuple[float, float]
    settling_mK: tuple[float, float]
    settling_days: tuple[float, float]
    union_T: tuple[tuple[float, float], ...]
    recent: Calibration  # the most recent calibration considered
    recent_offset_mK: float | None  # recent minus source at the same L, at recent_offset_T
    recent_offset_T: float | None  # center of their common fitted range [degC]
    notes: tuple[str, ...]
    rule: str = PRIOR_RULE

    @property
    def coefficients(self) -> dict[str, float]:
        return dict(self.source.coefficients)

    @property
    def fixed_terms(self) -> dict[str, float]:
        """The shape: every coefficient but ``t_0`` (held fixed in a ``t_0``-only refit)."""
        return {k: v for k, v in self.source.coefficients.items() if k != "t_0"}

    @property
    def order(self) -> int:
        return self.source.fit.order

    @property
    def bridge(self) -> dict[str, float]:
        return dict(self.source.bridge)

    @property
    def valid_T(self) -> tuple[float, float]:
        return self.source.valid_T

    @property
    def poly(self) -> np.ndarray:
        return self.source.poly

    def config_lines(self) -> list[str]:
        """``key = value`` lines for a ``.p`` channel section, at full precision."""
        return [f"{k} = {_full(v)}" for k, v in self.source.coefficients.items()]

    def to_dict(self) -> dict[str, Any]:
        return {
            "bead": self.bead,
            "rule": self.rule,
            "source": {
                "deployment": self.source.deployment,
                "instrument": self.source.instrument.label(),
                "first": _iso(self.source.first),
                "last": _iso(self.source.last),
            },
            "considered": [c.deployment for c in self.considered],
            "target": None if self.target is None else {
                "serial": self.target.serial,
                "channel": self.target.channel or None,
            },
            "cross_instrument": self.cross_instrument,
            "fallback": self.fallback,
            "most_recent": {
                "deployment": self.recent.deployment,
                "instrument": self.recent.instrument.label(),
                "offset_mK": self.recent_offset_mK,
                "at_T_degC": self.recent_offset_T,
            },
            "coefficients": self.coefficients,
            "poly": [float(x) for x in self.poly],
            "fixed_terms": self.fixed_terms,
            "order": self.order,
            "bridge": self.bridge,
            "uncertainty_mK": {
                "t_0": self.t_0_uncertainty_mK,
                "offset": self.offset_mK,
                "cross_instrument": self.cross_instrument_mK,
                "cross_instrument_applied": self.cross_instrument,
                "shape": self.shape_mK,
                "shape_sheet": self.sheet_shape_mK,
                "shape_span_K": list(self.shape_span_K),
                "settling": list(self.settling_mK),
                "settling_days": list(self.settling_days),
            },
            "valid_T_degC": list(self.valid_T),
            "union_T_degC": [list(r) for r in self.union_T],
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class Registry:
    schema: str
    transfer: Transfer
    methods: dict[str, str]
    calibrations: tuple[Calibration, ...]
    source: Path | None = None

    def beads(self) -> list[str]:
        return sorted({c.bead for c in self.calibrations}, key=_bead_key)

    def history(self, serial: str) -> list[Calibration]:
        """One bead's calibrations, oldest first."""
        out = sorted((c for c in self.calibrations if c.bead == serial), key=lambda c: c.sort_key())
        if not out:
            where = f" in {self.source}" if self.source else ""
            raise RegistryError(f"no calibration of bead {serial!r}{where}")
        return out

    def prior_for(
        self,
        serial: str,
        *,
        instrument: str | None = None,
        channel: str | None = None,
        as_of: date | datetime | str | None = None,
        excluding: tuple[datetime, datetime] | None = None,
    ) -> Prior:
        """A starting calibration for a fresh deployment of bead *serial*.

        *instrument* (a serial such as ``"194"``; ``"SN 194"`` and ``"0194"`` match
        too) and *channel* (``"T1"``, any case) describe where the bead is going;
        *as_of* restricts the evidence to calibrations that had ended by then.
        *excluding* ``(first, last)`` drops every calibration of the bead whose
        dates overlap that span: the deployment being fitted (a bead is in one
        place at a time, so an overlap is that deployment's own entry) must not
        be its own prior.

        The source, in order of preference (see :data:`PRIOR_RULE`):

        1. the widest-range unflagged calibration on the target instrument and
           channel (when a target is given);
        2. the bead's widest-range unflagged calibration, plus the
           cross-instrument term when a target was given and (1) found nothing;
        3. with no unflagged calibration at all, (1) and (2) over every
           calibration, with the source's held-out error added to the shape.

        "Unflagged" means free of :data:`DISQUALIFYING_FLAGS`.
        """
        history = self.history(serial)
        considered = history
        if as_of is not None:
            cutoff = _as_of(as_of)
            considered = [c for c in history if _ended_by(c, cutoff)]
            if not considered:
                raise RegistryError(
                    f"no calibration of bead {serial!r} had ended by {as_of}; "
                    f"the first ended {_iso(history[0].last)}"
                )
        if excluding is not None:
            ex_lo, ex_hi = (_datetime(x, "excluding") for x in excluding)
            considered = [c for c in considered if c.last < ex_lo or c.first > ex_hi]
            if not considered:
                raise RegistryError(
                    f"no calibration of bead {serial!r} outside {_iso(ex_lo)} .. {_iso(ex_hi)}, "
                    f"the span being fitted"
                )
        if channel is not None and instrument is None:
            raise RegistryError("a target channel needs a target instrument too")

        # A shape held fixed from a prior was not fitted over its own range, so
        # it cannot be the source of one (its source can).
        fitted = [c for c in considered if c.fit.order_choice != "prior"]
        if not fitted:
            raise RegistryError(
                f"every calibration of bead {serial!r} considered holds a borrowed shape "
                f"(order_choice prior); none fitted its own"
            )
        unflagged = [c for c in fitted if not set(c.flags) & set(DISQUALIFYING_FLAGS)]
        fallback = not unflagged
        pool = unflagged or fitted
        target: Instrument | None = None
        if instrument is not None:
            target = Instrument(model="", serial=_target_serial(instrument),
                                channel=_target_channel(channel))
            on_target = [
                c for c in pool
                if c.instrument.serial == target.serial
                and (not target.channel or c.instrument.channel == target.channel)
            ]
            source = _widest(on_target) if on_target else _widest(pool)
            cross = not on_target
        else:
            source, cross = _widest(pool), False
        recent = max(considered, key=_recency)

        t = self.transfer
        shape = math.hypot(t.shape_mK, source.uncertainty.held_out_mK) if fallback else t.shape_mK
        notes: list[str] = []
        if fallback:
            seen = [f for f in DISQUALIFYING_FLAGS if any(f in c.flags for c in considered)]
            notes.append(
                f"No unflagged calibration: every calibration of {serial} is flagged "
                f"{' or '.join(seen)}, so this prior comes from its widest-range calibration "
                f"({source.deployment}, flagged {', '.join(source.flags)}). Its shape term "
                f"adds that fit's {source.uncertainty.held_out_mK:.1f} mK held-out error to "
                f"the sheet's {t.shape_mK:g} mK in quadrature."
            )
        if target is None:
            notes.append(
                f"No target given, so this assumes the source's electronics "
                f"({source.instrument.label()})."
            )
        elif cross:
            where = f"SN {target.serial}" + (f" {target.channel}" if target.channel else "")
            notes.append(
                f"{serial} has no {'' if fallback else 'unflagged '}calibration on {where}. "
                f"The source is {source.instrument.label()}, so the bridge constants "
                f"differ and L, and with it the shape, is a different function of counts. "
                f"Compare the target channel's constants with the source's (`bridge`). "
                f"{t.cross_instrument_mK:g} mK, from one observation, is added to t_0 in "
                f"quadrature."
            )
        elif not target.channel:
            notes.append(
                f"No target channel given, so this assumes {source.instrument.channel}, "
                f"the source's; the other channel has other bridge constants."
            )
        recent_mK, recent_T = None, None
        if recent is source:
            notes.append("The source is also the most recent calibration.")
        else:
            lo = max(recent.valid_T[0], source.valid_T[0])
            hi = min(recent.valid_T[1], source.valid_T[1])
            other = (
                f" It is on other electronics ({recent.instrument.label()}), so this "
                f"includes the cross-instrument effect."
                if recent.instrument.electronics() != source.instrument.electronics() else ""
            )
            if lo < hi:
                recent_T = 0.5 * (lo + hi)
                recent_mK = curve_difference(recent, source, recent_T)
                notes.append(
                    f"The most recent calibration, {recent.deployment} "
                    f"({recent.last.date().isoformat()}), reads {recent_mK:+.1f} mK against "
                    f"this prior (most recent minus prior, at the same L) at "
                    f"{recent_T:.2f} degC, the center of their common fitted range."
                    + other
                )
            else:
                notes.append(
                    f"The most recent calibration, {recent.deployment}, has no fitted range "
                    f"in common with this prior, so the two are not compared." + other
                )
        union = _union([c.valid_T for c in considered])
        lo, hi = source.valid_T
        if len(union) > 1 or union[0][0] < lo or union[0][1] > hi:
            span = ", ".join(f"{a:.2f}-{b:.2f}" for a, b in union)
            msg = (
                f"The source was fitted over {lo:.2f}-{hi:.2f} degC; {serial} has been "
                f"calibrated over {span} degC. Outside {lo:.2f}-{hi:.2f} degC this "
                f"shape is extrapolated."
            )
            widest = _widest(pool)
            if widest is not source:
                msg += (
                    f" The widest-range {'' if fallback else 'unflagged '}calibration is "
                    f"{widest.deployment} ({widest.valid_T[0]:.2f}-{widest.valid_T[1]:.2f} "
                    f"degC)."
                )
            notes.append(msg)
        if source.flags and not fallback:
            notes.append(f"The source calibration is flagged: {', '.join(source.flags)}.")
        ref = source.reference
        if ref.cal_date is not None and source.cal_age_years is not None:
            notes.append(
                f"Every number here is relative to the source's reference "
                f"({_ref(source)}), which was {source.cal_age_years:.1f} years past its "
                f"{ref.cal_date.isoformat()} calibration. That reference's own error is "
                f"not included."
            )
        else:
            notes.append(
                f"Every number here is relative to the source's reference "
                f"({_ref(source)}), whose calibration date is not recorded. Its own "
                f"error is not included."
            )
        notes.append(
            f"Expect a further {_range(t.settling_mK)} mK of settling over the first "
            f"{_range(t.settling_days)} days of a long deployment; it is not in the "
            f"t_0 uncertainty."
        )
        return Prior(
            bead=serial,
            source=source,
            considered=tuple(considered),
            target=target,
            cross_instrument=cross,
            fallback=fallback,
            offset_mK=t.offset_mK,
            cross_instrument_mK=t.cross_instrument_mK,
            t_0_uncertainty_mK=math.hypot(t.offset_mK, t.cross_instrument_mK if cross else 0.0),
            sheet_shape_mK=t.shape_mK,
            shape_mK=shape,
            shape_span_K=t.shape_span_K,
            settling_mK=t.settling_mK,
            settling_days=t.settling_days,
            union_T=tuple(union),
            recent=recent,
            recent_offset_mK=recent_mK,
            recent_offset_T=recent_T,
            notes=tuple(notes),
        )


# ------------------------------------------------------------------ loading
def default_path() -> Path:
    """The repository's sheet when running from a checkout, else ``./microstructure_sensors/``."""
    repo = Path(__file__).resolve().parents[3] / "microstructure_sensors" / YAML_NAME
    if repo.exists():
        return repo
    return Path("microstructure_sensors") / YAML_NAME


def load(path: str | Path | None = None) -> Registry:
    """Read and validate the sheet.  Raises :class:`RegistryError` on any defect."""
    p = Path(path) if path is not None else default_path()
    if not p.exists():
        raise RegistryError(f"{p}: no such file")
    return loads(p.read_text(encoding="utf-8"), source=p)


def loads(text: str, *, source: Path | None = None) -> Registry:
    """Validate the sheet from its text (what :func:`load` reads from disk)."""
    from ruamel.yaml import YAML
    from ruamel.yaml.error import YAMLError

    try:
        data = YAML(typ="safe", pure=True).load(text)
    except YAMLError as exc:
        raise RegistryError(f"{source or 'registry'}: not valid YAML ({exc})") from exc
    return parse(data, source=source)


def prior_for(
    serial: str,
    *,
    instrument: str | None = None,
    channel: str | None = None,
    as_of: date | datetime | str | None = None,
    path: str | Path | None = None,
) -> Prior:
    """:meth:`Registry.prior_for` on the sheet at *path* (default: :func:`default_path`)."""
    return load(path).prior_for(serial, instrument=instrument, channel=channel, as_of=as_of)


def parse(data: object, *, source: Path | None = None) -> Registry:
    """Validate already-loaded YAML data and build the :class:`Registry`."""
    where = str(source) if source is not None else "registry"
    top = _mapping(data, where, ("schema", "transfer", "methods", "calibrations"))
    if top["schema"] != SCHEMA:
        raise RegistryError(f"{where}: schema is {top['schema']!r}, expected {SCHEMA!r}")
    transfer = _parse_transfer(top["transfer"])
    methods = _parse_methods(top["methods"])
    raw = top["calibrations"]
    if not isinstance(raw, list) or not raw:
        raise RegistryError("calibrations: expected a non-empty list")
    cals = [_parse_calibration(r, i, methods) for i, r in enumerate(raw)]
    _check_unique(cals)
    _check_occupancy(cals)
    _check_shape_sources(cals)
    return Registry(
        schema=SCHEMA,
        transfer=transfer,
        methods=methods,
        calibrations=tuple(sorted(cals, key=lambda c: c.sort_key())),
        source=source,
    )


def _parse_transfer(raw: object) -> Transfer:
    w = "transfer"
    m = _mapping(raw, w, ("offset_mK", "shape_mK", "shape_span_K", "cross_instrument_mK",
                          "settling_mK", "settling_days", "source", "evidence",
                          "reference_caveat"))
    ev = _mapping(m["evidence"], f"{w}.evidence",
                  ("offset", "shape", "cross_instrument", "settling"))
    return Transfer(
        offset_mK=_num(m["offset_mK"], f"{w}.offset_mK", positive=True),
        shape_mK=_num(m["shape_mK"], f"{w}.shape_mK", positive=True),
        shape_span_K=_pair(m["shape_span_K"], f"{w}.shape_span_K"),
        cross_instrument_mK=_num(m["cross_instrument_mK"], f"{w}.cross_instrument_mK",
                                 nonneg=True),
        settling_mK=_pair(m["settling_mK"], f"{w}.settling_mK"),
        settling_days=_pair(m["settling_days"], f"{w}.settling_days"),
        source=_str(m["source"], f"{w}.source"),
        # Whitespace-normalised: a `|` block's line breaks would split a table row.
        evidence={k: _prose(ev[k], f"{w}.evidence.{k}") for k in sorted(ev)},
        reference_caveat=_prose(m["reference_caveat"], f"{w}.reference_caveat"),
    )


def _parse_methods(raw: object) -> dict[str, str]:
    if not isinstance(raw, dict) or not raw:
        raise RegistryError("methods: expected a non-empty mapping of name -> description")
    out: dict[str, str] = {}
    for k, v in raw.items():
        if not isinstance(k, str) or not _METHOD_RE.fullmatch(k):
            raise RegistryError(f"methods: name {k!r} must be lower-case words joined by '-'")
        out[k] = _prose(v, f"methods.{k}")
    return out


_ENTRY_KEYS = ("bead", "deployment", "instrument", "first", "last", "reference", "fit",
               "coefficients", "bridge", "uncertainty", "valid_T_degC")
_ENTRY_OPTIONAL = ("flags", "notes")


def _parse_calibration(raw: object, i: int, methods: dict[str, str]) -> Calibration:
    w0 = f"calibrations[{i}]"
    m = _mapping(raw, w0, _ENTRY_KEYS, _ENTRY_OPTIONAL)
    bead = m["bead"]
    if not isinstance(bead, str) or not _BEAD_RE.fullmatch(bead):
        raise SerialPatternError(
            f"{w0}: bead serial {bead!r} is not of the form T<number> without a leading "
            f"zero (e.g. 'T1592'); the placeholder 'T' is not an identity"
        )
    deployment = _str(m["deployment"], f"{w0}.deployment")
    w = f"{w0} ({bead} / {deployment})"

    im = _mapping(m["instrument"], f"{w}.instrument", ("model", "serial", "channel"),
                  ("platform",))
    serial = im["serial"]
    if not isinstance(serial, str) or not _INSTRUMENT_SERIAL_RE.fullmatch(serial):
        raise SerialPatternError(
            f"{w}.instrument.serial: {serial!r} must be a quoted string of digits "
            f"(e.g. \"194\")"
        )
    chan = im["channel"]
    if not isinstance(chan, str) or not _CHANNEL_RE.fullmatch(chan):
        raise SerialPatternError(f"{w}.instrument.channel: {chan!r} is not of the form T<digit>")
    instrument = Instrument(
        model=_str(im["model"], f"{w}.instrument.model"),
        serial=_canonical_serial(serial),  # "0194" is SN 194
        channel=chan,
        platform=_opt_str(im.get("platform"), f"{w}.instrument.platform"),
    )

    first = _datetime(m["first"], f"{w}.first")
    last = _datetime(m["last"], f"{w}.last")
    if first > last:
        raise DateOrderError(f"{w}: first {_iso(first)} is after last {_iso(last)}")

    reference = _parse_reference(m["reference"], f"{w}.reference")
    if reference.cal_date is not None and reference.cal_date > first.date():
        raise DateOrderError(
            f"{w}.reference.cal_date {reference.cal_date.isoformat()} is after the "
            f"deployment began ({_iso(first)})"
        )

    fit = _parse_fit(m["fit"], f"{w}.fit", methods)
    coefficients = _parse_coefficients(m["coefficients"], f"{w}.coefficients", fit.order)

    bm = _mapping(m["bridge"], f"{w}.bridge", BRIDGE_KEYS)
    bridge = {k: _num(bm[k], f"{w}.bridge.{k}") for k in BRIDGE_KEYS}
    bridge["adc_bits"] = float(_int(bm["adc_bits"], f"{w}.bridge.adc_bits", minimum=1))
    if bridge["b"] == 0:
        raise RegistryError(f"{w}.bridge.b: must be non-zero")
    for k in ("g", "e_b", "adc_fs"):
        if bridge[k] <= 0:
            raise RegistryError(f"{w}.bridge.{k}: must be positive, got {bridge[k]}")

    um = _mapping(m["uncertainty"], f"{w}.uncertainty",
                  ("fit_rms_mK", "held_out_mK", "block_sd_mK", "n_blocks"), ("rms_all_mK",))
    n_blocks = _int(um["n_blocks"], f"{w}.uncertainty.n_blocks", minimum=0)
    block_sd = um["block_sd_mK"]
    if block_sd is not None:
        block_sd = _num(block_sd, f"{w}.uncertainty.block_sd_mK", nonneg=True)
        if n_blocks < MIN_BLOCKS_FOR_SCATTER:
            raise RegistryError(
                f"{w}.uncertainty: a block scatter from {n_blocks} block(s) says nothing; "
                f"set block_sd_mK: null below {MIN_BLOCKS_FOR_SCATTER} blocks"
            )
    elif n_blocks >= MIN_BLOCKS_FOR_SCATTER:
        raise MissingKeyError(
            f"{w}.uncertainty: block_sd_mK is null with {n_blocks} blocks; null is only "
            f"for fewer than {MIN_BLOCKS_FOR_SCATTER}"
        )
    uncertainty = Uncertainty(
        fit_rms_mK=_num(um["fit_rms_mK"], f"{w}.uncertainty.fit_rms_mK", nonneg=True),
        held_out_mK=_num(um["held_out_mK"], f"{w}.uncertainty.held_out_mK", nonneg=True),
        block_sd_mK=block_sd,
        n_blocks=n_blocks,
        rms_all_mK=(None if um.get("rms_all_mK") is None
                    else _num(um["rms_all_mK"], f"{w}.uncertainty.rms_all_mK", nonneg=True)),
    )
    _check_order_choice(fit, uncertainty, w)

    lo, hi = _pair(m["valid_T_degC"], f"{w}.valid_T_degC")
    if not lo < hi:
        raise RegistryError(f"{w}.valid_T_degC: [{lo}, {hi}] is empty")
    _check_curve(
        np.array([1.0 / coefficients["t_0"]]
                 + [1.0 / coefficients[k] for k in BETA_KEYS if k in coefficients]),
        (lo, hi), f"{w}.coefficients",
    )

    flags_raw = m.get("flags") or []
    if not isinstance(flags_raw, list):
        raise RegistryError(f"{w}.flags: expected a list")
    for f in flags_raw:
        if f not in FLAGS:
            raise RegistryError(f"{w}.flags: unknown flag {f!r}; known: {sorted(FLAGS)}")
    if len(set(flags_raw)) != len(flags_raw):
        raise RegistryError(f"{w}.flags: repeated flag")
    flags = tuple(f for f in FLAGS if f in flags_raw)  # vocabulary order: deterministic

    notes = m.get("notes")
    notes = "" if notes is None else _str(notes, f"{w}.notes")
    return Calibration(
        bead=bead,
        deployment=deployment,
        instrument=instrument,
        first=first,
        last=last,
        reference=reference,
        fit=fit,
        coefficients=coefficients,
        bridge=bridge,
        uncertainty=uncertainty,
        valid_T=(lo, hi),
        flags=flags,
        notes=" ".join(notes.split()),
    )


def _parse_reference(raw: object, w: str) -> Reference:
    m = _mapping(raw, w, ("kind", "serial", "cal_date"), ("note",))
    serial = m["serial"]
    if serial is not None and (
        not isinstance(serial, str) or not _REFERENCE_SERIAL_RE.fullmatch(serial)
    ):
        raise SerialPatternError(
            f"{w}.serial: {serial!r} must be a quoted string (e.g. \"0079\": unquoted, "
            f"YAML would read it as the number 79) or null"
        )
    cal_date = None if m["cal_date"] is None else _date(m["cal_date"], f"{w}.cal_date")
    note = _opt_str(m.get("note"), f"{w}.note")
    if (serial is None or cal_date is None) and not note:
        raise MissingKeyError(f"{w}: serial or cal_date is null; say why in `note`")
    return Reference(kind=_str(m["kind"], f"{w}.kind"), serial=serial, cal_date=cal_date,
                     note=None if note is None else " ".join(note.split()))


def _parse_fit(raw: object, w: str, methods: dict[str, str]) -> FitSettings:
    m = _mapping(raw, w, ("method", "pooled", "n_files", "n_profiles", "n_pairs", "order",
                          "order_choice", "robust", "geometry", "lag_s", "reference_pole_s",
                          "software_version"), ("held_out_by_order_mK", "shape_from"))
    method = _str(m["method"], f"{w}.method")
    if method not in methods:
        raise RegistryError(f"{w}.method: {method!r} is not defined under `methods`")
    order = _int(m["order"], f"{w}.order", minimum=1)
    if order > 3:
        raise OrderMismatchError(f"{w}.order: {order}; fp07-cal fits orders 1..3")
    choice = m["order_choice"]
    if choice not in ORDER_CHOICES:
        raise RegistryError(f"{w}.order_choice: {choice!r} is not one of {sorted(ORDER_CHOICES)}")
    held: dict[int, float] | None = None
    if m.get("held_out_by_order_mK") is not None:
        hm = m["held_out_by_order_mK"]
        if not isinstance(hm, dict) or not hm:
            raise RegistryError(f"{w}.held_out_by_order_mK: expected a mapping order -> mK")
        held = {}
        for k, v in hm.items():
            if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= 3:
                raise UnknownKeyError(f"{w}.held_out_by_order_mK: key {k!r} is not an order 1..3")
            held[k] = _num(v, f"{w}.held_out_by_order_mK.{k}", nonneg=True)
        held = dict(sorted(held.items()))
    elif choice == "auto":
        # "auto" is a claim about the held-out errors; without them it is uncheckable.
        raise MissingKeyError(
            f"{w}: order_choice auto needs held_out_by_order_mK (the record's "
            f"order_selection) so the choice can be checked; use `fixed` if it is unknown"
        )
    shape_from = _opt_str(m.get("shape_from"), f"{w}.shape_from")
    if choice == "prior" and shape_from is None:
        raise MissingKeyError(
            f"{w}: order_choice prior needs shape_from, the deployment of this bead's "
            f"calibration whose beta terms were held fixed"
        )
    if choice != "prior" and shape_from is not None:
        raise OrderMismatchError(
            f"{w}.shape_from is only for order_choice prior (the shape held fixed); "
            f"order_choice is {choice}"
        )
    geo_raw = m["geometry"]
    geometry: dict[str, float] | None
    if geo_raw is False:
        geometry = None
    elif isinstance(geo_raw, dict):
        gm = _mapping(geo_raw, f"{w}.geometry", ("dz_m", "tau_s"))
        geometry = {k: _num(gm[k], f"{w}.geometry.{k}") for k in ("dz_m", "tau_s")}
    else:
        raise RegistryError(
            f"{w}.geometry: `false` when the sensor-geometry term was not fitted, "
            f"else the fitted {{dz_m, tau_s}}; got {geo_raw!r}"
        )
    return FitSettings(
        method=method,
        pooled=_bool(m["pooled"], f"{w}.pooled"),
        n_files=_int(m["n_files"], f"{w}.n_files", minimum=1),
        n_profiles=_int(m["n_profiles"], f"{w}.n_profiles", minimum=1),
        n_pairs=_int(m["n_pairs"], f"{w}.n_pairs", minimum=1),
        order=order,
        order_choice=choice,
        held_out_by_order_mK=held,
        robust=_bool(m["robust"], f"{w}.robust"),
        geometry=geometry,
        lag_s=_num(m["lag_s"], f"{w}.lag_s"),
        reference_pole_s=_num(m["reference_pole_s"], f"{w}.reference_pole_s", nonneg=True),
        software_version=_str(m["software_version"], f"{w}.software_version"),
        shape_from=None if shape_from is None else " ".join(shape_from.split()),
    )


def _parse_coefficients(raw: object, w: str, order: int) -> dict[str, float]:
    m = _mapping(raw, w, ("t_0", "beta_1"), ("beta_2", "beta_3"))
    out = {"t_0": _num(m["t_0"], f"{w}.t_0", positive=True)}
    if not T0_RANGE_K[0] <= out["t_0"] <= T0_RANGE_K[1]:
        raise PlausibilityError(
            f"{w}.t_0 = {out['t_0']:g} is outside {T0_RANGE_K[0]:g}-{T0_RANGE_K[1]:g}; "
            f"t_0 is in KELVIN (degC + 273.15)"
        )
    if _num(m["beta_1"], f"{w}.beta_1", nonzero=True) <= 0:
        raise PlausibilityError(
            f"{w}.beta_1 = {m['beta_1']} must be positive: an FP07's resistance falls "
            f"as it warms"
        )
    present = [k for k in BETA_KEYS if k in m]
    if present != list(BETA_KEYS[: len(present)]):
        raise OrderMismatchError(f"{w}: beta terms {present} are not consecutive from beta_1")
    if len(present) != order:
        raise OrderMismatchError(
            f"{w}: fit.order is {order} but the coefficients carry {len(present)} beta "
            f"term(s) {present}; an order-n fit has exactly beta_1..beta_n"
        )
    for k in present:
        v = _num(m[k], f"{w}.{k}", nonzero=True)
        if abs(v) >= _NEUTRAL_BETA:
            raise OrderMismatchError(
                f"{w}.{k} = {v:g} switches the term off, which is how a .p config "
                f"neutralises it. In this sheet leave the key out and lower fit.order."
            )
        out[k] = v
    return out


def _check_order_choice(fit: FitSettings, unc: Uncertainty, w: str) -> None:
    held = fit.held_out_by_order_mK
    if held is None or fit.order_choice == "prior":
        # With the shape held fixed, held_out_mK is the offset-only fit's own
        # error, and held_out_by_order_mK (optional) the full fits' it was
        # weighed against; `fit.prior.mode: offset` may have forced the choice.
        return
    if fit.order not in held:
        raise OrderMismatchError(
            f"{w}.fit.held_out_by_order_mK has no entry for the chosen order {fit.order}"
        )
    if abs(held[fit.order] - unc.held_out_mK) > 0.005:
        raise OrderMismatchError(
            f"{w}: uncertainty.held_out_mK {unc.held_out_mK} is not the chosen order's "
            f"held-out error {held[fit.order]}"
        )
    if fit.order_choice == "auto" and held[fit.order] > min(held.values()):
        best = min(held, key=lambda k: held[k])
        raise OrderMismatchError(
            f"{w}: order_choice is auto, which picks the least held-out error, but "
            f"order {best} ({held[best]} mK) beats the chosen order {fit.order} "
            f"({held[fit.order]} mK)"
        )


def _check_unique(cals: list[Calibration]) -> None:
    seen: dict[tuple[str, str], int] = {}
    for i, c in enumerate(cals):
        key = (c.bead, _norm_label(c.deployment))
        if key in seen:
            raise DuplicateEntryError(
                f"calibrations[{seen[key]}] and calibrations[{i}] are both "
                f"{c.bead} / {c.deployment}; one entry per bead and deployment"
            )
        seen[key] = i


def _check_shape_sources(cals: list[Calibration]) -> None:
    """A borrowed shape must name its source, and be that source's shape exactly.

    The source is another calibration of the same bead that fitted its own
    shape (one hop, no chains), and its beta terms must equal this entry's: an
    entry claiming a prior's shape while carrying different betas would record
    a provenance that is not true.
    """
    by_key = {(c.bead, _norm_label(c.deployment)): c for c in cals}
    for c in cals:
        if c.fit.shape_from is None:
            continue
        w = f"{c.bead} / {c.deployment}: fit.shape_from"
        src = by_key.get((c.bead, _norm_label(c.fit.shape_from)))
        if src is None:
            raise MissingKeyError(
                f"{w} names {c.fit.shape_from!r}, which is not a calibration of {c.bead} "
                f"in this sheet"
            )
        if src is c:
            raise OrderMismatchError(f"{w} names the entry itself")
        if src.fit.order_choice == "prior":
            raise OrderMismatchError(
                f"{w} names {src.deployment}, whose shape is itself borrowed (from "
                f"{src.fit.shape_from}); name the calibration that fitted it"
            )
        mine = {k: v for k, v in c.coefficients.items() if k != "t_0"}
        theirs = {k: v for k, v in src.coefficients.items() if k != "t_0"}
        if mine.keys() != theirs.keys() or any(
            abs(mine[k] - theirs[k]) > 1e-12 * abs(theirs[k]) for k in mine
        ):
            raise OrderMismatchError(
                f"{w}: the beta terms {mine} are not those of {src.deployment} {theirs}"
            )


def _check_occupancy(cals: list[Calibration]) -> None:
    """A bead cannot be in two places at once, nor a channel carry two beads."""
    for label, key in (("bead", lambda c: c.bead),
                       ("channel", lambda c: c.instrument.electronics())):
        groups: dict[Any, list[Calibration]] = {}
        for c in cals:
            groups.setdefault(key(c), []).append(c)
        for group in groups.values():
            group.sort(key=lambda c: (c.first, c.last))
            for a, b in pairwise(group):
                if b.first < a.last:
                    what = (f"bead {a.bead}" if label == "bead"
                            else f"{a.instrument.label()}")
                    raise DateOrderError(
                        f"{what} is in two calibrations at once: {a.bead} / "
                        f"{a.deployment} ({_iso(a.first)} .. {_iso(a.last)}) overlaps "
                        f"{b.bead} / {b.deployment} ({_iso(b.first)} .. {_iso(b.last)})"
                    )


# ------------------------------------------------------------- value checks
def _mapping(obj: object, where: str, required: tuple[str, ...],
             optional: tuple[str, ...] = ()) -> dict:
    if not isinstance(obj, dict):
        raise RegistryError(f"{where}: expected a mapping, got {type(obj).__name__}")
    allowed = set(required) | set(optional)
    unknown = sorted(str(k) for k in obj if k not in allowed)
    if unknown:
        raise UnknownKeyError(f"{where}: unknown key(s) {unknown}; allowed: {sorted(allowed)}")
    missing = [k for k in required if k not in obj]
    if missing:
        raise MissingKeyError(f"{where}: missing required key(s) {missing}")
    return obj


def _str(v: object, where: str) -> str:
    if not isinstance(v, str) or not v.strip():
        raise RegistryError(f"{where}: expected a non-empty string, got {v!r}")
    return v.strip()


def _opt_str(v: object, where: str) -> str | None:
    return None if v is None else _str(v, where)


def _prose(v: object, where: str) -> str:
    """A string with every run of whitespace (line breaks included) made one space."""
    return " ".join(_str(v, where).split())


def _num(v: object, where: str, *, positive: bool = False, nonneg: bool = False,
         nonzero: bool = False) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise RegistryError(f"{where}: expected a number, got {v!r}")
    x = float(v)
    if not math.isfinite(x):
        raise RegistryError(f"{where}: must be finite, got {v!r}")
    if positive and x <= 0:
        raise RegistryError(f"{where}: must be positive, got {v!r}")
    if nonneg and x < 0:
        raise RegistryError(f"{where}: must be >= 0, got {v!r}")
    if nonzero and x == 0:
        raise RegistryError(
            f"{where}: must be non-zero (it is a reciprocal; zero would be an infinite term)"
        )
    return x


def _int(v: object, where: str, *, minimum: int) -> int:
    if isinstance(v, bool) or not isinstance(v, int):
        raise RegistryError(f"{where}: expected an integer, got {v!r}")
    if v < minimum:
        raise RegistryError(f"{where}: must be >= {minimum}, got {v}")
    return int(v)


def _bool(v: object, where: str) -> bool:
    if not isinstance(v, bool):
        raise RegistryError(f"{where}: expected true or false, got {v!r}")
    return v


def _pair(v: object, where: str) -> tuple[float, float]:
    if not isinstance(v, list) or len(v) != 2:
        raise RegistryError(f"{where}: expected [low, high], got {v!r}")
    lo, hi = (_num(x, where) for x in v)
    if lo > hi:
        raise RegistryError(f"{where}: low {lo} > high {hi}")
    return (lo, hi)


def _date(v: object, where: str) -> date:
    if isinstance(v, datetime):
        raise RegistryError(f"{where}: expected a date (YYYY-MM-DD), got a date-time")
    if isinstance(v, date):
        return v
    if isinstance(v, str) and _DATE_RE.fullmatch(v):
        try:
            return date.fromisoformat(v)
        except ValueError as exc:
            raise RegistryError(f"{where}: {v!r} is not a date ({exc})") from exc
    raise RegistryError(f"{where}: expected a date YYYY-MM-DD, got {v!r}")


def _datetime(v: object, where: str) -> datetime:
    if isinstance(v, datetime):
        dt = v
    elif isinstance(v, str) and _DATETIME_RE.fullmatch(v):
        try:
            dt = datetime.fromisoformat(v.replace("Z", "+00:00"))
        except ValueError as exc:
            raise RegistryError(f"{where}: {v!r} is not a date-time ({exc})") from exc
    else:
        raise RegistryError(f"{where}: expected a UTC date-time YYYY-MM-DDTHH:MM:SSZ, got {v!r}")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    if dt.utcoffset() != UTC.utcoffset(None):
        raise RegistryError(f"{where}: must be UTC, got {v!r}")
    return dt.astimezone(UTC)


# --------------------------------------------------------------- helpers
def _bead_key(serial: str) -> tuple[int, str]:
    digits = serial[1:]
    return (int(digits) if digits.isdigit() else -1, serial)


def _recency(c: Calibration) -> tuple:
    return (c.last, c.first, c.deployment)


def _widest(cals: list[Calibration]) -> Calibration:
    """The widest fitted range; ties go to the more recent."""
    return max(cals, key=lambda c: (round(c.valid_T[1] - c.valid_T[0], 6), _recency(c)))


def _canonical_serial(s: str) -> str:
    """An instrument serial as a number: "0194" and "194" are one instrument."""
    return str(int(s))


def _target_serial(v: object) -> str:
    s = str(v).strip()
    if s[:2].casefold() == "sn":
        s = s[2:].strip()
    return _canonical_serial(s) if s.isdigit() else s


def _target_channel(v: str | None) -> str:
    return "" if v is None else v.strip().upper()


def _norm_label(s: str) -> str:
    """Deployment labels compared without case or spacing differences."""
    return " ".join(s.split()).casefold()


def _L_at(poly: np.ndarray, T_degC: float) -> float | None:
    """The L at which ``1/T_K = sum a_i L^i`` reads *T_degC*, on the physical branch.

    The physical branch passes near L = 0 (T near t_0); the other roots of a
    quadratic or cubic lie at |L| of order beta_1/beta_2 or more, so the real
    root nearest zero is the one.  None when there is no real root.
    """
    p = np.array(poly, dtype=np.float64)
    p[0] -= 1.0 / (T_degC + 273.15)
    if p.size == 2:
        return float(-p[0] / p[1])
    roots = np.roots(p[::-1])
    real = roots[np.abs(roots.imag) <= 1e-9 * max(1.0, float(np.max(np.abs(roots))))].real
    if real.size == 0:
        return None
    return float(real[np.argmin(np.abs(real))])


def curve_difference(a: Calibration, b: Calibration, T_degC: float) -> float:
    """``a`` minus ``b`` [mK] at the L where ``b`` reads *T_degC*.

    Calibration curves are compared at a fixed L, never by ``t_0`` alone,
    because ``t_0`` and ``beta_1`` trade off.  Across instruments L is the same
    resistance ratio through different bridges, so the difference then
    includes the cross-instrument effect.
    """
    try:
        return poly_difference_mK(a.poly, b.poly, T_degC)
    except ValueError as exc:
        raise RegistryError(f"{b.bead} / {b.deployment} never reads {T_degC} degC") from exc


def poly_difference_mK(a: np.ndarray, b: np.ndarray, T_degC: float) -> float:
    """Curve *a* minus curve *b* [mK] at the L where *b* reads *T_degC*.

    Both are ``[a0, a1, ...]`` with ``1/T_K = sum a_i L^i``.  ``ValueError``
    when *b* never reads *T_degC* on its physical branch.
    """
    L = _L_at(b, T_degC)
    if L is None:
        raise ValueError(f"the curve never reads {T_degC} degC")
    La = np.array([L])
    return float((temperature(La, a)[0] - temperature(La, b)[0]) * 1e3)


def is_bead_serial(s: object) -> bool:
    """A Rockland FP07 serial (``T1592``), not the placeholder ``T`` or ``(no SN)``."""
    return isinstance(s, str) and _BEAD_RE.fullmatch(s) is not None


def _check_curve(poly: np.ndarray, valid_T: tuple[float, float], where: str) -> None:
    """The curve must reach both ends of its range inside the bridge's L, and fall with L."""
    Ls = []
    for T in valid_T:
        L = _L_at(poly, T)
        if L is None or abs(L) >= L_CLIP:
            got = "no real L" if L is None else f"L = {L:.3f}"
            raise PlausibilityError(
                f"{where}: the coefficients reach {T} degC only at {got}; the bridge "
                f"clips |Z| at 0.6, so |L| < ln(1.6/0.4) = {L_CLIP:.3f}. Check t_0 and "
                f"the beta terms against the fit's record."
            )
        Ls.append(L)
    grid = np.linspace(min(Ls), max(Ls), 201)
    inv = sum(a * grid**i for i, a in enumerate(poly))
    slope = sum(i * a * grid ** (i - 1) for i, a in enumerate(poly) if i)
    if np.any(inv <= 0) or np.any(slope <= 0):
        raise PlausibilityError(
            f"{where}: T must fall as L rises across {valid_T[0]}-{valid_T[1]} degC "
            f"(an NTC bead), but dT/dL >= 0 somewhere in L = {min(Ls):.3f}..{max(Ls):.3f}"
        )


def _as_of(v: date | datetime | str) -> date | datetime:
    if isinstance(v, (date, datetime)):
        return v if not isinstance(v, datetime) else _datetime(v, "as_of")
    if isinstance(v, str):
        if _DATE_RE.fullmatch(v):
            return _date(v, "as_of")
        return _datetime(v, "as_of")
    raise RegistryError(f"as_of: expected a date or a UTC date-time, got {v!r}")


def _ended_by(c: Calibration, cutoff: date | datetime) -> bool:
    if isinstance(cutoff, datetime):
        return c.last <= cutoff
    return c.last.date() <= cutoff


def _union(ranges: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for lo, hi in sorted(ranges):
        if out and lo <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], hi))
        else:
            out.append((lo, hi))
    return out


def _iso(dt: datetime) -> str:
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _full(x: float) -> str:
    """Shortest text that round-trips the float exactly."""
    return repr(float(x))


def _range(r: tuple[float, float]) -> str:
    return f"{r[0]:g}-{r[1]:g}" if r[0] != r[1] else f"{r[0]:g}"


# -------------------------------------------------------------- rendering
CSV_COLUMNS = (
    "bead", "deployment", "instrument_model", "instrument_serial", "channel", "platform",
    "first", "last", "duration_days",
    "reference_kind", "reference_serial", "reference_cal_date", "reference_cal_age_years",
    "method", "pooled", "n_files", "n_profiles", "n_pairs", "order", "order_choice",
    "shape_from", "held_out_order1_mK", "held_out_order2_mK", "held_out_order3_mK",
    "robust", "geometry_dz_m", "geometry_tau_s", "lag_s", "reference_pole_s",
    "t_0", "beta_1", "beta_2", "beta_3",
    "a", "b", "g", "e_b", "adc_fs", "adc_bits",
    "fit_rms_mK", "rms_all_mK", "held_out_mK", "block_sd_mK", "n_blocks",
    "T_min_degC", "T_max_degC", "flags", "software_version", "notes",
)


def _cell(v: object) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return _full(v)
    return str(v)


def render_csv(reg: Registry) -> str:
    """One row per fit; floats at full precision, so the CSV loses nothing."""
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(CSV_COLUMNS)
    for c in reg.calibrations:
        held = c.fit.held_out_by_order_mK or {}
        geo = c.fit.geometry or {}
        row = {
            "bead": c.bead,
            "deployment": c.deployment,
            "instrument_model": c.instrument.model,
            "instrument_serial": c.instrument.serial,
            "channel": c.instrument.channel,
            "platform": c.instrument.platform,
            "first": _iso(c.first),
            "last": _iso(c.last),
            "duration_days": round(c.duration_days, 2),
            "reference_kind": c.reference.kind,
            "reference_serial": c.reference.serial,
            "reference_cal_date": None if c.reference.cal_date is None
            else c.reference.cal_date.isoformat(),
            "reference_cal_age_years": None if c.cal_age_years is None
            else round(c.cal_age_years, 2),
            "method": c.fit.method,
            "pooled": c.fit.pooled,
            "n_files": c.fit.n_files,
            "n_profiles": c.fit.n_profiles,
            "n_pairs": c.fit.n_pairs,
            "order": c.fit.order,
            "order_choice": c.fit.order_choice,
            "shape_from": c.fit.shape_from,
            "held_out_order1_mK": held.get(1),
            "held_out_order2_mK": held.get(2),
            "held_out_order3_mK": held.get(3),
            "robust": c.fit.robust,
            "geometry_dz_m": geo.get("dz_m"),
            "geometry_tau_s": geo.get("tau_s"),
            "lag_s": c.fit.lag_s,
            "reference_pole_s": c.fit.reference_pole_s,
            "t_0": c.coefficients["t_0"],
            "beta_1": c.coefficients["beta_1"],
            "beta_2": c.coefficients.get("beta_2"),
            "beta_3": c.coefficients.get("beta_3"),
            **{k: c.bridge[k] for k in BRIDGE_KEYS if k != "adc_bits"},
            "adc_bits": int(c.bridge["adc_bits"]),
            "fit_rms_mK": c.uncertainty.fit_rms_mK,
            "rms_all_mK": c.uncertainty.rms_all_mK,
            "held_out_mK": c.uncertainty.held_out_mK,
            "block_sd_mK": c.uncertainty.block_sd_mK,
            "n_blocks": c.uncertainty.n_blocks,
            "T_min_degC": c.valid_T[0],
            "T_max_degC": c.valid_T[1],
            "flags": ";".join(c.flags),
            "software_version": c.fit.software_version,
            "notes": c.notes,
        }
        w.writerow([_cell(row[k]) for k in CSV_COLUMNS])
    return buf.getvalue()


def _md(text: str) -> str:
    """Escape the one character that breaks a Markdown table cell."""
    return text.replace("|", "\\|")


def _g(x: float) -> str:
    return f"{x:g}"


def _beta(c: Calibration, k: str) -> str:
    # Rounded far below 0.1 mK: beta_1 to 1e-4 K moves T by ~1e-6 K, and the
    # higher terms (|beta| >= 1e4 K, against L^2 <= ~1) by less.
    v = c.coefficients.get(k)
    if v is None:
        return "—"
    return f"{v:.4f}" if k == "beta_1" else f"{v:.1f}"


def _ref(c: Calibration) -> str:
    r = c.reference
    return r.kind + (f" SN {r.serial}" if r.serial else "")


def _mk(x: float | None) -> str:
    return "—" if x is None else f"{x:.1f}"


def _where(cals: list[Calibration]) -> str:
    seen: list[str] = []
    for c in cals:
        lab = c.instrument.label()
        if not seen or seen[-1] != lab:
            seen.append(lab)
    return " → ".join(seen)


def render_markdown(reg: Registry) -> str:
    """The human-readable sheet.  Deterministic: same YAML, same bytes."""
    t = reg.transfer
    cals = list(reg.calibrations)
    beads = reg.beads()
    first = min(c.first for c in cals)
    last = max(c.last for c in cals)
    out: list[str] = []
    P = out.append
    P("<!-- GENERATED by `fp07-cal registry render` from fp07_calibrations.yaml. "
      "Do not edit: edit the YAML and re-render. -->")
    P("")
    P("# FP07 in-situ calibration sheet")
    P("")
    P(f"**{len(cals)} calibrations of {len(beads)} FP07 beads**, "
      f"{first.date().isoformat()} to {last.date().isoformat()}.")
    P("")
    P("This page is generated from [`fp07_calibrations.yaml`](fp07_calibrations.yaml), the "
      "single source of truth. [`fp07_calibrations.csv`](fp07_calibrations.csv) is the "
      "same entries, one row per fit, for spreadsheets. Never edit either view by hand: "
      "edit the YAML, then run `fp07-cal registry render`. `fp07-cal registry check` "
      "(and the test suite) fails when a view is out of date.")
    P("")
    P("## How to read this sheet")
    P("")
    P("**An entry** is one FP07 bead (by its Rockland serial, e.g. T1592) in one "
      "deployment, calibrated in situ by `fp07-cal fit` against a reference thermometer "
      "on the same vehicle. All of a deployment's profiles are pooled into one fit. "
      "Beads move between channels and instruments, so a bead's history can span "
      "several instruments.")
    P("")
    P("**The equation.** The coefficients are in the form a Rockland `.p` configuration "
      "carries, which ODAS and this package (`rsi/channels.py::convert_therm`) read and "
      "`fp07-cal patch` writes:")
    P("")
    P("```")
    P("1/T_K = 1/t_0 + L/beta_1 + L^2/beta_2 + L^3/beta_3        T [degC] = T_K - 273.15")
    P("```")
    P("")
    P("`t_0` and every `beta` are in kelvin, and each `beta` is a *reciprocal*. A fit of "
      "order n has `beta_1` .. `beta_n`, and an absent term is no term. To switch a term "
      "off in a `.p` config that carries it, write `beta_2 = 1e30`, never `0` (zero would "
      "be an infinite term). `t_0` and `beta_1` trade off strongly against each other, so "
      "do not compare entries by `t_0` alone; compare the temperatures they imply at the "
      "same L.")
    P("")
    P("**What L is.** `L = ln(R_T/R_0)`, the log of the bead's resistance relative to the "
      "bridge's reference resistor. It is computed from raw counts through the channel's "
      "half-bridge constants:")
    P("")
    P("```")
    P("Z = ((counts - a)/b) * (adc_fs / 2^adc_bits) * 2 / (g * e_b)      "
      "L = ln((1 - Z)/(1 + Z))")
    P("```")
    P("")
    P(f"Z is clipped to ±0.6 (as ODAS does), so L never leaves ±ln(1.6/0.4) = "
      f"±{L_CLIP:.3f}; a sample on the clip is out of the bridge's range, not a "
      f"temperature. Each entry records the six constants (`a`, `b`, `g`, `e_b`, `adc_fs`, "
      "`adc_bits`) because the coefficients describe T as a function of *this* L. Copied "
      "to a channel with different constants (another instrument, or the other channel "
      "of the same one), the same coefficients give a different temperature.")
    P("")
    P("**How to use an entry.** Copy `t_0` and the `beta` terms into the channel's "
      "section of the `.p` configuration, provided its bridge constants are the ones "
      "listed. **Remove any legacy `beta` key from that section**: ODAS and "
      "`convert_therm` read `beta` before `beta_1`, so a pasted `beta_1` is silently "
      "ignored while `beta` is there. Then run perturb with `fp07.calibrate: false`. "
      "Full precision is in the "
      "YAML and the CSV; the tables here are rounded to well below 0.1 mK. Use an entry "
      "only inside its fitted temperature range, since a polynomial is not constrained "
      "outside it. Where the deployment has its own reference, its own pooled fit "
      "(`fp07-cal fit`, see `docs/fp07cal/runbook.md`) is better than any entry here. "
      "An entry's prior is for when a full fit is impossible, and the history is for QC.")
    P("")
    P("**What the uncertainties mean.** Each is a temperature error in mK, measured "
      "against that entry's reference:")
    P("")
    P("- *rms*: root-mean-square residual of the fit against the reference, in sample, over "
      "the pairs its robust 4-sigma clip kept. For fits made with `fit.prior` the notes also "
      "give the rms over every pair (`rms_all_mK`): a shape that does not fit clips the ends "
      "of the range and looks better over the pairs it kept.")
    P("- *held-out*: rms error predicting the cold half of the range from a fit to the "
      "warm half, and vice versa. It measures how well the shape extrapolates, and it "
      "chose the order where the order is `auto`. Where the order is `prior`, the shape "
      "was held fixed from the bead's calibration named in the notes and only `t_0` was "
      "fitted (`fp07-cal fit` with `fit.prior`); held-out is then that fit's.")
    P("- *block SD*: the deployment is cut into equal-time blocks, each block's `t_0` "
      "offset is refitted with the shape held fixed (fp07-cal's stability check), and "
      "this is the standard deviation of those offsets. It measures how far the "
      "calibration wandered within the deployment. It is left blank below "
      f"{MIN_BLOCKS_FOR_SCATTER} blocks.")
    P("")
    P("None of them includes the reference's own error. The *cal. age* column is how long "
      "the reference had gone without a calibration when the deployment began.")
    P("")
    P(f"The same limit applies between deployments. {_md(t.reference_caveat)}")
    P("")
    P("### Prior for a fresh deployment")
    P("")
    P("Each bead's section ends with a prior: a starting calibration for a deployment "
      "where a full in-situ fit is impossible (too little reference, or too narrow a "
      "temperature range to constrain the shape). The rule:")
    P("")
    P(f"> {PRIOR_RULE}")
    P("")
    P(f"The transfer uncertainties come from comparing the same beads across deployments "
      f"and years ([issue #189]({t.source})). They are typical observed magnitudes from a "
      f"handful of beads, not bounds and not formal standard errors; the evidence column "
      f"gives the largest seen:")
    P("")
    P("| term | typical size | evidence |")
    P("|---|---|---|")
    P(f"| t_0 carried over to a new deployment | ±{_g(t.offset_mK)} mK | "
      f"{_md(t.evidence['offset'])} |")
    P(f"| curve shape (`beta` terms) | ±{_g(t.shape_mK)} mK over "
      f"{_range(t.shape_span_K)} K | {_md(t.evidence['shape'])} |")
    P(f"| other instrument or channel | {_g(t.cross_instrument_mK)} mK, "
      f"added to t_0 in quadrature; one observation | "
      f"{_md(t.evidence['cross_instrument'])} |")
    P(f"| settling in a new deployment | {_range(t.settling_mK)} mK over the first "
      f"{_range(t.settling_days)} days | {_md(t.evidence['settling'])} |")
    P("")
    P("A `t_0`-only refit (the shape held fixed from the prior) removes the first term "
      "and needs almost no temperature range. `fp07-cal fit` does it when its config has "
      "a `fit.prior` block. It keeps that refit unless the full fit is clearly better on "
      "held-out error, and always over a range narrower than the shape span or when no "
      "full fit is a plausible curve for the bead. It flags a result outside the prior's "
      "uncertainty over their common range: a possible unrecorded bead swap, reference "
      "problem or failing bead (see `docs/fp07cal/runbook.md`). "
      "`fp07-cal registry show <serial>` prints "
      "a bead's prior, with `--instrument`/`--channel` for a different target and "
      "`--as-of` to use only calibrations that had ended by a date. `fp07-cal registry "
      "add` records a fit here from its `coefficients.json`.")
    P("")
    P("### Flags")
    P("")
    P("| flag | meaning |")
    P("|---|---|")
    for k, v in FLAGS.items():
        P(f"| `{k}` | {_md(v)} |")
    P("")
    P("### Methods")
    P("")
    P("| method | how the fits were made |")
    P("|---|---|")
    for k in sorted(reg.methods):
        P(f"| `{k}` | {_md(reg.methods[k])} |")
    P("")
    P("## Beads")
    P("")
    P("| bead | where | calibrations | first | last |")
    P("|---|---|---|---|---|")
    for b in beads:
        h = reg.history(b)
        P(f"| [{b}](#{b.lower()}) | {_md(_where(h))} | {len(h)} | "
          f"{h[0].first.date().isoformat()} | {h[-1].last.date().isoformat()} |")
    for b in beads:
        out.extend(_bead_section(reg, b))
    return "\n".join(out) + "\n"


def _bead_section(reg: Registry, bead: str) -> list[str]:
    h = reg.history(bead)
    out: list[str] = []
    P = out.append
    P("")
    P(f"## {bead}")
    P("")
    n = len(h)
    P(f"{_md(_where(h))}. {n} calibration{'s' if n != 1 else ''}, "
      f"{h[0].first.date().isoformat()} to {h[-1].last.date().isoformat()}.")
    P("")
    P("| # | deployment | instrument | first | last | reference | cal. age [y] | order "
      "| files / profiles | T range [°C] | rms [mK] | held-out [mK] | block SD [mK] | flags |")
    P("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for i, c in enumerate(h, start=1):
        ref = _ref(c)
        ref += f", cal. {c.reference.cal_date.isoformat()}" if c.reference.cal_date else ""
        age = "—" if c.cal_age_years is None else f"{c.cal_age_years:.1f}"
        block = _mk(c.uncertainty.block_sd_mK)
        block += f" ({c.uncertainty.n_blocks})" if c.uncertainty.block_sd_mK is not None else ""
        flags = ", ".join(f"`{f}`" for f in c.flags) or "—"
        inst = c.instrument.label() + (f" on {c.instrument.platform}"
                                       if c.instrument.platform else "")
        P(f"| {i} | {_md(c.deployment)} | {_md(inst)} | {c.first.date().isoformat()} | "
          f"{c.last.date().isoformat()} | {_md(ref)} | {age} | {c.fit.order} "
          f"({c.fit.order_choice}) | {c.fit.n_files} / {c.fit.n_profiles} | "
          f"{c.valid_T[0]:.2f} to {c.valid_T[1]:.2f} | {_mk(c.uncertainty.fit_rms_mK)} | "
          f"{_mk(c.uncertainty.held_out_mK)} | {block} | {flags} |")
    P("")
    P("Coefficients and the bridge constants that define their L:")
    P("")
    P("| # | t_0 [K] | beta_1 [K] | beta_2 [K] | beta_3 [K] | a | b | g | e_b | adc_fs [V] "
      "| adc_bits |")
    P("|---|---|---|---|---|---|---|---|---|---|---|")
    for i, c in enumerate(h, start=1):
        br = c.bridge
        P(f"| {i} | {c.coefficients['t_0']:.6f} | {_beta(c, 'beta_1')} | {_beta(c, 'beta_2')} | "
          f"{_beta(c, 'beta_3')} | {_g(br['a'])} | {_g(br['b'])} | {_g(br['g'])} | "
          f"{_g(br['e_b'])} | {_g(br['adc_fs'])} | {int(br['adc_bits'])} |")
    notes = [(i, c) for i, c in enumerate(h, start=1)
             if c.notes or c.reference.note or c.fit.geometry or c.fit.shape_from
             or c.uncertainty.rms_all_mK is not None]
    if notes:
        P("")
        P("Notes:")
        P("")
        for i, c in notes:
            parts = []
            if c.fit.shape_from:
                parts.append(f"Shape (beta terms) held fixed from {c.fit.shape_from}; only "
                             f"t_0 was fitted, and held-out is that fit's.")
            if c.uncertainty.rms_all_mK is not None:
                parts.append(f"rms over every pair {c.uncertainty.rms_all_mK:.1f} mK (the "
                             f"rms column is over the pairs the robust fit kept).")
            if c.fit.geometry:
                g = c.fit.geometry
                parts.append(f"Sensor geometry fitted jointly: dz = {g['dz_m']:.3f} m, "
                             f"tau = {g['tau_s']:.2f} s.")
            if c.notes:
                parts.append(c.notes)
            if c.reference.note:
                parts.append(f"Reference: {c.reference.note}")
            P(f"{i}. **{_md(c.deployment)}**: " + " ".join(parts))
    P("")
    out.extend(_prior_section(reg.prior_for(bead)))
    return out


def _prior_section(p: Prior) -> list[str]:
    s = p.source
    out: list[str] = []
    P = out.append
    P(f"### Prior for a fresh deployment of {p.bead}")
    P("")
    why = ("its widest-range calibration; **none is unflagged**" if p.fallback
           else "the widest-range unflagged calibration")
    P(f"From **{_md(s.deployment)}** ({_md(s.instrument.label())}, "
      f"{s.first.date().isoformat()} to {s.last.date().isoformat()}), {why}. "
      f"Order {p.order}.")
    P("")
    P("| t_0 [K] | beta_1 [K] | beta_2 [K] | beta_3 [K] |")
    P("|---|---|---|---|")
    P(f"| {s.coefficients['t_0']:.6f} | {_beta(s, 'beta_1')} | {_beta(s, 'beta_2')} | "
      f"{_beta(s, 'beta_3')} |")
    P("")
    other = math.hypot(p.offset_mK, p.cross_instrument_mK)
    P(f"- **t_0**: ±{p.t_0_uncertainty_mK:.0f} mK in temperature, the carried-over offset "
      f"(±{other:.0f} mK on another instrument or channel, a term that rests on one "
      f"observation).")
    if p.fallback:
        P(f"- **shape**: ±{p.shape_mK:.1f} mK over a {_range(p.shape_span_K)} K span: the "
          f"sheet's {_g(p.sheet_shape_mK)} mK and the source's "
          f"{s.uncertainty.held_out_mK:.1f} mK held-out error, in quadrature.")
    else:
        P(f"- **shape**: ±{_g(p.shape_mK)} mK over a {_range(p.shape_span_K)} K span.")
    P(f"- **fitted range**: {p.valid_T[0]:.2f} to {p.valid_T[1]:.2f} °C.")
    for i, note in enumerate(p.notes):
        P(f"- **{_md(note)}**" if p.fallback and i == 0 else f"- {_md(note)}")
    return out


# ------------------------------------------------------------- views on disk
def view_paths(path: str | Path) -> dict[str, Path]:
    """Where the generated views of the sheet at *path* live."""
    p = Path(path)
    return {"md": p.with_suffix(".md"), "csv": p.with_suffix(".csv")}


def rendered_views(path: str | Path | None = None) -> dict[Path, str]:
    """``{view path: text}`` rendered from the sheet at *path*."""
    p = Path(path) if path is not None else default_path()
    reg = load(p)
    v = view_paths(p)
    return {v["md"]: render_markdown(reg), v["csv"]: render_csv(reg)}


def _current(dst: Path, text: str) -> bool:
    # UTF-8 explicitly (the sheet carries ±, ° and →), and read in universal-
    # newline mode so a CRLF checkout does not read as stale.
    return dst.exists() and dst.read_text(encoding="utf-8") == text


def write_views(path: str | Path | None = None) -> list[Path]:
    """Render and write both views; returns the paths that changed."""
    changed = []
    for dst, text in rendered_views(path).items():
        if not _current(dst, text):
            dst.write_text(text, encoding="utf-8", newline="\n")
            changed.append(dst)
    return changed


def stale_views(path: str | Path | None = None) -> list[Path]:
    """Views whose committed text differs from what the sheet renders."""
    return [dst for dst, text in rendered_views(path).items() if not _current(dst, text)]


# ------------------------------------------------------------ adding a fit
class AddError(RegistryError):
    """A fit that cannot be recorded as given (a missing or contradictory field)."""


_BLOCK_ROW = re.compile(r"^\|\s*\d+\s*\|[^|]*\|\s*\d+\s*\|[^|]*\|\s*([+-]?\d+(?:\.\d+)?)\s*±")


def block_offsets_from_report(text: str) -> list[float]:
    """The block offsets [K] in a ``<ch>_report.md``'s temporal-stability table.

    ``coefficients.json`` keeps only the block count, so the block scatter the
    sheet records comes from the report ``fp07-cal fit`` wrote beside it.
    """
    out: list[float] = []
    in_table = False
    for line in text.splitlines():
        if line.startswith("| block | start | profiles | t_0 | offset [K] |"):
            in_table = True
            continue
        if in_table:
            if line.startswith("|---"):
                continue
            m = _BLOCK_ROW.match(line)
            if not m:
                break
            out.append(float(m.group(1)))
    return out


def _epoch_iso(t: object, where: str) -> str:
    if isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t):
        raise AddError(f"{where}: expected epoch seconds, got {t!r}")
    return _iso(datetime.fromtimestamp(float(t), UTC))


def entry_from_record(
    record: dict,
    channel: str,
    *,
    deployment: str,
    instrument_model: str,
    method: str,
    reference_kind: str,
    reference_serial: str | None,
    reference_cal_date: str | None,
    reference_pole_s: float,
    block_offsets_K: list[float],
    reference_note: str | None = None,
    bead: str | None = None,
    instrument_serial: str | None = None,
    platform: str | None = None,
    robust: bool = True,
    flags: list[str] | tuple[str, ...] = (),
    notes: str = "",
    allow_qc_flag: bool = False,
) -> dict[str, Any]:
    """One sheet entry, in the YAML's structure, from an ``fp07-cal fit`` record.

    *record* is a ``coefficients.json`` and *channel* one of its channels.
    Everything the record carries is taken from it; the keyword arguments
    supply the rest, which has no safe default (the instrument model, the
    reference's identity, the fit method, the reference pole, the block
    offsets from the report).  Refuses (:class:`AddError`) a channel the fit
    refused, a placeholder bead serial, a lag the trust gate rejected without
    the ``lag-gate-overridden`` flag, a fit its prior QC flagged without
    *allow_qc_flag*, and a fit with no held-out error for its chosen
    candidate.  The entry is validated in full by :func:`add_entry`, against
    the sheet it joins.
    """
    from odas_tpw.fp07cal.patch import validate_record

    try:
        validate_record(record)
    except ValueError as exc:
        raise AddError(str(exc)) from exc
    chans = record["channels"]
    if channel not in chans:
        raise AddError(f"{channel}: not in the record (it has {sorted(chans)})")
    e = chans[channel]
    if "config_equivalent" not in e:
        raise AddError(f"{channel}: the fit exported no coefficients ({e.get('error', '?')})")
    flags = list(flags)
    if not e.get("lag_trustworthy", True) and "lag-gate-overridden" not in flags:
        raise AddError(
            f"{channel}: fp07-cal's lag-trust gate rejected this fit's lag (lag_trustworthy "
            f"false). Record it only with --flag lag-gate-overridden, and say why in the notes."
        )
    prior = e.get("prior") or {}
    used = prior.get("status") == "used"
    if used and prior["qc"]["verdict"] == "flag" and not allow_qc_flag:
        raise AddError(
            f"{channel}: the fit's QC against the bead's prior flagged it: "
            f"{prior['qc']['message']} Resolve that first; --allow-qc-flag records it anyway."
        )

    fitted_bead = prior.get("bead") or None
    file_sn = str(e.get("probe_sn", "") or "")
    if bead is None:
        bead = fitted_bead or file_sn
        if not fitted_bead and not e.get("probe_sn_trusted", True):
            raise AddError(
                f"{channel}: the record's probe serial {file_sn!r} is shared between channels, "
                f"so it is a placeholder; name the bead with --bead"
            )
    elif fitted_bead and bead != fitted_bead:
        raise AddError(f"{channel}: --bead {bead} contradicts the bead the fit used for its "
                       f"prior, {fitted_bead}")
    if not is_bead_serial(bead):
        raise AddError(f"{channel}: bead serial {bead!r} is a placeholder, not an identity; "
                       f"name the bead with --bead T<number>")

    inst = _target_serial(
        instrument_serial if instrument_serial is not None else record["instrument_sn"]
    )
    if not inst.isdigit():
        raise AddError(f"instrument serial {inst!r} is not a number; give --instrument-serial")

    order = int(e["order"])
    held_full = {int(k): float(v["held_out_K"]) * 1e3
                 for k, v in (e.get("order_selection") or {}).items()}
    if used and not held_full:
        held_full = {int(c["order"]): float(c["held_out_K"]) * 1e3
                     for c in prior["candidates"].values()
                     if c["kind"] == "full" and c.get("held_out_K") is not None}
    shape_from = None
    held_out: float | None
    if used and prior["chosen_kind"] == "offset":
        choice = "prior"
        shape_from = str(prior["source"]["deployment"])
        h = prior["candidates"]["prior shape"].get("held_out_K")
        held_out = None if h is None else float(h) * 1e3
    elif e.get("order_selection") and held_full.get(order) == min(held_full.values()):
        choice, held_out = "auto", held_full.get(order)
    else:
        # Forced, or (with fit.prior) the held-out-best order was not a plausible
        # FP07 curve and the next plausible one was kept: not "auto" in the
        # sheet's sense, which the loader checks is the held-out minimum.
        choice, held_out = "fixed", held_full.get(order)
    if held_out is None:
        raise AddError(
            f"{channel}: the record has no held-out error for the chosen fit (order {order}, "
            f"{choice}); refit with fit.order: auto (or a fit.prior block), which scores it"
        )

    n_blocks = len(block_offsets_K)
    rec_blocks = (e.get("stability") or {}).get("n_blocks")
    if rec_blocks is not None and int(rec_blocks) != n_blocks:
        raise AddError(f"{channel}: the report's block table has {n_blocks} block(s) but the "
                       f"record says {rec_blocks}; is it the report from the same fit?")
    block_sd = (round(float(np.std(np.asarray(block_offsets_K) * 1e3, ddof=1)), 2)
                if n_blocks >= MIN_BLOCKS_FOR_SCATTER else None)

    geo = e.get("geometry")
    bridge = {k: float(e["bridge"][k]) for k in BRIDGE_KEYS}
    lo, hi = (float(x) for x in e["T_range"])
    fit: dict[str, Any] = {
        "method": method,
        "pooled": True,
        "n_files": int(record.get("n_fit_files") or 0),
        "n_profiles": int(e["n_profiles"]),
        "n_pairs": int(e["n_pairs"]),
        "order": order,
        "order_choice": choice,
    }
    if held_full:
        fit["held_out_by_order_mK"] = {k: round(v, 2) for k, v in sorted(held_full.items())}
    if shape_from is not None:
        fit["shape_from"] = shape_from
    fit.update({
        "robust": bool(robust),
        "geometry": False if not geo else {"dz_m": round(float(geo["dz_m"]), 3),
                                           "tau_s": round(float(geo["tau_s"]), 3)},
        "lag_s": round(float(e["lag_s"]), 3),
        "reference_pole_s": round(float(reference_pole_s), 3),
        "software_version": str(record.get("software_version") or "unknown"),
    })
    ref: dict[str, Any] = {"kind": reference_kind, "serial": reference_serial,
                           "cal_date": reference_cal_date}
    if reference_note:
        ref["note"] = " ".join(reference_note.split())
    inst_d: dict[str, Any] = {"model": instrument_model, "serial": inst, "channel": channel}
    if platform:
        inst_d["platform"] = platform
    entry: dict[str, Any] = {
        "bead": bead,
        "deployment": " ".join(deployment.split()),
        "instrument": inst_d,
        "first": _epoch_iso(record.get("time_start"), "time_start"),
        "last": _epoch_iso(record.get("time_end"), "time_end"),
        "reference": ref,
        "fit": fit,
        "coefficients": {k: float(v) for k, v in e["config_equivalent"].items()},
        "bridge": {**bridge, "adc_bits": int(bridge["adc_bits"])},
        "uncertainty": {"fit_rms_mK": round(float(e["rms_K"]) * 1e3, 2),
                        "held_out_mK": round(held_out, 2),
                        "block_sd_mK": block_sd, "n_blocks": n_blocks},
        "valid_T_degC": [round(lo, 2), round(hi, 2)],
    }
    rms_all = (prior["candidates"].get(prior["choice"], {}).get("rms_all_K") if used else None)
    if rms_all is not None:
        entry["uncertainty"]["rms_all_mK"] = round(float(rms_all) * 1e3, 2)
    if flags:
        entry["flags"] = flags
    if notes.strip():
        entry["notes"] = " ".join(notes.split())
    return entry


_PLAIN_RE = re.compile(r"[A-Za-z][A-Za-z0-9 ._()/+-]*[A-Za-z0-9.)]|[A-Za-z]")
_YAML_WORDS = {"true", "false", "yes", "no", "on", "off", "null", "y", "n"}


def _scalar(v: Any, *, quote: bool = False) -> str:
    """One YAML scalar in the sheet's style: plain where that is unambiguous."""
    import json

    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return _full(v)
    s = str(v)
    if (not quote and _PLAIN_RE.fullmatch(s) and s.lower() not in _YAML_WORDS
            and ": " not in s and " #" not in s):
        return s
    return json.dumps(s, ensure_ascii=False)


def _flow(d: dict) -> str:
    return "{" + ", ".join(f"{k}: {_scalar(v)}" for k, v in d.items()) + "}"


def _folded(key: str, text: str, indent: int) -> list[str]:
    import textwrap

    pad = " " * indent
    body = textwrap.wrap(" ".join(text.split()), width=88 - indent - 2,
                         break_long_words=False, break_on_hyphens=False)
    return [f"{pad}{key}: >-"] + [f"{pad}  {line}" for line in body]


def entry_yaml(entry: dict[str, Any], indent: int = 2) -> str:
    """*entry* as text in the sheet's own layout (block style, flow maps for short ones).

    *indent* is the column of the sequence's ``-`` (2 in the repository's sheet).
    """
    text = _entry_yaml2(entry)
    if indent == 2:
        return text
    shift = " " * indent
    return "".join(shift + line[2:] if line.strip() else line
                   for line in text.splitlines(keepends=True))


_TOP_KEY = re.compile(r"[A-Za-z_][\w-]*\s*:")
_CALS_KEY = re.compile(r"calibrations\s*:\s*(\[\s*\])?\s*(#.*)?")
_ITEM = re.compile(r"( *)- ")


def _insertion_point(lines: list[str]) -> tuple[int, int]:
    """``(line index, item indent)`` just after the last entry of ``calibrations:``.

    Blank lines and column-0 comments immediately before the next top-level
    key introduce that key, so the entry goes above them; an INDENTED comment
    after the last entry belongs to that entry and stays with it.  An empty
    ``calibrations: []`` is turned into block style in place (*lines* is
    edited), and its first entry is indented 2.
    """
    starts = [i for i, ln in enumerate(lines) if _CALS_KEY.fullmatch(ln.rstrip("\n"))]
    if len(starts) != 1:
        raise AddError("the sheet has no block-style `calibrations:` key at column 0 to add to")
    start = starts[0]
    if "[" in lines[start]:
        lines[start] = re.sub(r"\s*\[\s*\]", "", lines[start], count=1)
    end = len(lines)
    for i in range(start + 1, len(lines)):
        if _TOP_KEY.match(lines[i]):
            end = i
            break
    while end > start + 1 and (not lines[end - 1].strip() or lines[end - 1].startswith("#")):
        end -= 1
    items = [m.group(1) for ln in lines[start + 1:end] if (m := _ITEM.match(ln))]
    return end, (len(items[0]) if items else 2)


@contextlib.contextmanager
def _locked(path: Path):
    """An exclusive lock on a sidecar ``.<name>.lock`` (none where ``fcntl`` is missing)."""
    try:
        import fcntl
    except ImportError:  # pragma: no cover - not POSIX
        yield
        return
    with open(path.parent / f".{path.name}.lock", "a") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def _entry_yaml2(entry: dict[str, Any]) -> str:
    i, f, r, u = entry["instrument"], entry["fit"], entry["reference"], entry["uncertainty"]
    out = [f"  - bead: {_scalar(entry['bead'])}",
           f"    deployment: {_scalar(entry['deployment'])}",
           "    instrument:",
           f"      model: {_scalar(i['model'])}",
           f"      serial: {_scalar(i['serial'], quote=True)}",
           f"      channel: {_scalar(i['channel'])}"]
    if "platform" in i:
        out.append(f"      platform: {_scalar(i['platform'])}")
    out += [f"    first: {_scalar(entry['first'], quote=True)}",
            f"    last: {_scalar(entry['last'], quote=True)}",
            "    reference:",
            f"      kind: {_scalar(r['kind'])}",
            f"      serial: {_scalar(r['serial'], quote=True)}",
            f"      cal_date: {_scalar(r['cal_date'], quote=True)}"]
    if r.get("note"):
        out += _folded("note", r["note"], 6)
    out.append("    fit:")
    for k, v in f.items():
        if k == "held_out_by_order_mK" or (k == "geometry" and isinstance(v, dict)):
            out.append(f"      {k}: {_flow(v)}")
        else:
            out.append(f"      {k}: {_scalar(v, quote=k == 'software_version')}")
    out.append("    coefficients:")
    out += [f"      {k}: {_scalar(v)}" for k, v in entry["coefficients"].items()]
    out.append(f"    bridge: {_flow(entry['bridge'])}")
    out.append("    uncertainty:")
    out += [f"      {k}: {_scalar(u[k])}"
            for k in ("fit_rms_mK", "rms_all_mK", "held_out_mK", "block_sd_mK", "n_blocks")
            if k in u]
    lo, hi = entry["valid_T_degC"]
    out.append(f"    valid_T_degC: [{_scalar(lo)}, {_scalar(hi)}]")
    if entry.get("flags"):
        out.append("    flags: [" + ", ".join(entry["flags"]) + "]")
    if entry.get("notes"):
        out += _folded("notes", entry["notes"], 4)
    return "\n".join(out) + "\n"


def add_entry(path: str | Path, entry: dict[str, Any], *, dry_run: bool = False) -> str:
    """Append *entry* to the sheet at *path*, validated, and re-render its views.

    The sheet is hand-edited YAML with comments, so it is never re-dumped: a
    ruamel round trip of the sheet as committed (0.19.1, nothing changed)
    rewrote 2091 lines with default settings, re-wrapping the folded prose, and
    every ``null`` even when tuned.  The entry is INSERTED as text, in the
    sheet's own layout, after the last entry of ``calibrations:`` (the end of
    the file in the repository's sheet), and every existing byte is kept, with
    the file's line endings (LF or CRLF) and mode.  The loader sorts entries,
    so position carries no meaning.  Before anything is written the whole new
    text must pass the strict loader, and the inserted block must read back as
    exactly *entry*.  A second entry for the same bead and deployment is
    refused.  The read-modify-write holds an exclusive lock on a sidecar
    ``.<name>.lock`` (git-ignored in ``microstructure_sensors/``), so
    concurrent adds cannot lose one.  An empty ``calibrations:`` is accepted.
    Returns the inserted text; with *dry_run* nothing is written or locked.
    """
    p = Path(path)
    if dry_run:
        return _add(p, entry, dry_run=True)
    with _locked(p):  # read-modify-write: two concurrent adds must not lose one
        return _add(p, entry, dry_run=False)


def _add(p: Path, entry: dict[str, Any], *, dry_run: bool) -> str:
    import os
    import stat
    import tempfile

    from ruamel.yaml import YAML

    raw = p.read_bytes().decode("utf-8")
    crlf = "\r\n" in raw
    text = raw.replace("\r\n", "\n")
    try:
        existing: tuple[Calibration, ...] = loads(text, source=p).calibrations
    except RegistryError:
        data = YAML(typ="safe", pure=True).load(text)
        if not (isinstance(data, dict) and not data.get("calibrations")):
            raise
        existing = ()  # an empty sheet; the new text is validated in full below
    key = (entry["bead"], _norm_label(entry["deployment"]))
    for c in existing:
        if (c.bead, _norm_label(c.deployment)) == key:
            raise DuplicateEntryError(
                f"{p} already has {c.bead} / {c.deployment}: one entry per bead and "
                f"deployment. To replace it, edit the YAML by hand."
            )
    lines = text.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    at, indent = _insertion_point(lines)
    block = entry_yaml(entry, indent)
    back = YAML(typ="safe", pure=True).load(block)
    if back != [entry]:
        raise AddError(f"the entry does not read back as written (a quoting defect):\n{block}")
    head = "".join(lines[:at]).rstrip("\n") + "\n\n"
    tail = "".join(lines[at:])
    new = head + block + ("\n" + tail.lstrip("\n") if tail.strip() else "")
    loads(new, source=p)  # the strict loader, over the whole sheet with the entry in it
    if dry_run:
        return block
    if crlf:
        new = new.replace("\n", "\r\n")
    mode = stat.S_IMODE(os.stat(p).st_mode)
    fd, tmp = tempfile.mkstemp(dir=p.parent, prefix=f".{p.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
            fh.write(new)
        os.chmod(tmp, mode)  # mkstemp makes it 0600; keep the sheet's own mode
        os.replace(tmp, p)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    write_views(p)
    return block

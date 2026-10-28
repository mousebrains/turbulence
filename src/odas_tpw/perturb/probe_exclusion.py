# Oct-2026, Claude and Pat Welch, pat@mousebrains.com
"""Windowed probe exclusion: ``instruments.<key>.exclude`` (issue #190).

A shear probe can develop a pressure-related fault: bad inside a depth window,
fine above and below it. ``exclude_shear_probes`` can only drop a probe for the
whole deployment, which on CASPER-West SN 194 would have discarded two clean
weeks and everything below 65 m to remove a 30-70 dbar box, and it cannot name
a thermistor at all. The QC statistics do not catch such a fault (sh2's Lueck
FM peaked near 0.92 while ``e_2/e_1`` reached 15-18x), so it has to be
declared by the person who found it.

Each entry names ONE probe and the window over which its per-window estimates
are masked. A window is inside when EVERY bound the entry gives holds, tested
against what the product already stores for that window:

    P_min <= P_mean < P_max      (dbar)
    start <=   t    < stop       (UTC)

An omitted bound is unbounded, so an entry with no bounds is the whole
deployment. Masking is exactly what ``exclude_shear_probes`` does to its probe,
restricted to the window (see ``pipeline._apply_probe_exclusions``). A shear
entry is masked in the diss product and reaches chi through ``epsilonMean``; a
thermistor entry is masked in the chi product only.

Besides time, the only bounded quantity is pressure. A ``channel:`` key naming
another per-window variable would sit beside ``P_min``/``P_max`` without
changing what an existing entry means; it is not needed yet.

Kept free of numpy at import: ``perturb.config`` validates entries at load
time, and building the CLI parser must not pull numpy in.
"""

from __future__ import annotations

import datetime as dt
import math
import re
from dataclasses import dataclass
from typing import Any

#: Keys an ``exclude`` entry may carry.
ENTRY_KEYS = frozenset({"probe", "P_min", "P_max", "start", "stop", "reason"})

#: A shear probe, masked in the diss product (and so in ``epsilonMean``, which
#: chi Method 1 and the mixing quantities consume).
SHEAR_PROBE = re.compile(r"^sh\d+$")
#: A thermistor, masked in the chi product: the bead name (``T1``) or its
#: gradient channel (``T1_dT1``), the two spellings ``fp07_tau_scale`` takes.
#: The digits must agree, so the gradient spelling always names one bead.
THERMISTOR_PROBE = re.compile(r"^T(\d+)(_dT\1)?$")

#: Product-attribute prefix of each applied entry (``probe_exclusion_<n>_*``).
ATTR_PREFIX = "probe_exclusion"

#: ``probe_exclusion_<n>_comment`` on a chi file for a shear entry: it was
#: masked in that profile's diss product and changes the chi file only through
#: ``epsilonMean``.
INHERITED_COMMENT = (
    "masked upstream in this profile's diss product (n_windows counts diss "
    "windows); reaches this file through epsilonMean: chi Method 1 (kB), "
    "epsilon_paired, K_rho and Gamma"
)


def _iso(epoch: float) -> str:
    """Epoch seconds as an ISO-8601 UTC string ending in ``Z``."""
    return dt.datetime.fromtimestamp(epoch, dt.UTC).isoformat().replace("+00:00", "Z")


def _parse_time(value: Any, where: str, key: str) -> float | None:
    """An ISO-8601 time as epoch seconds; ``None`` passes through.

    YAML turns an UNQUOTED ``2017-10-09T00:00:00Z`` into a datetime and
    ``2017-10-09`` into a date before this sees it, so both are accepted as
    well as the string. A time without a zone is UTC: guessing local time would
    move the window by the operator's offset.
    """
    if value is None:
        return None
    if isinstance(value, dt.datetime):  # before date: datetime subclasses date
        parsed = value
    elif isinstance(value, dt.date):
        parsed = dt.datetime(value.year, value.month, value.day)
    elif isinstance(value, str):
        try:
            parsed = dt.datetime.fromisoformat(value.strip())
        except ValueError as exc:
            raise ValueError(
                f"{where}: {key}={value!r} is not an ISO-8601 time "
                f'(e.g. "2017-10-09T00:00:00Z")'
            ) from exc
    else:
        raise ValueError(
            f'{where}: {key} must be an ISO-8601 time string (e.g. "2017-10-09T00:00:00Z"), '
            f"got {value!r}"
        )
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.UTC)
    return parsed.timestamp()


def _parse_pressure(value: Any, where: str, key: str) -> float | None:
    """A finite pressure bound in dbar; ``None`` passes through."""
    if value is None:
        return None
    # bool is an int subclass; `P_min: yes` is a typo, not 1 dbar.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where}: {key} must be a number (dbar), got {value!r}")
    v = float(value)
    if not math.isfinite(v):
        raise ValueError(f"{where}: {key} must be finite, got {value!r}")
    return v


@dataclass(frozen=True)
class ProbeExclusion:
    """One validated ``exclude`` entry. Times are epoch seconds (UTC)."""

    instrument: str
    entry: int  # 1-based position in the instrument's `exclude` list
    probe: str
    reason: str
    P_min: float | None = None
    P_max: float | None = None
    start: float | None = None
    stop: float | None = None

    @property
    def label(self) -> str:
        """Where this entry lives in the config, for messages."""
        return f"instruments.{self.instrument}.exclude entry {self.entry} (probe {self.probe!r})"

    @property
    def product(self) -> str:
        """The per-profile product whose probes this entry masks."""
        return "diss" if SHEAR_PROBE.match(self.probe) else "chi"

    def canonical(self) -> dict[str, Any]:
        """The entry as plain values: bounds it gives only, times as ISO UTC.

        What the product records and the stage signature hashes, so ``30`` and
        ``30.0``, a quoted and an unquoted timestamp, or ``T1`` and ``T1_dT1``
        are one exclusion rather than two output directories. ``reason`` is
        part of it because it is written into the product.
        """
        out: dict[str, Any] = {"probe": self.probe, "reason": self.reason}
        for key in ("P_min", "P_max"):
            value = getattr(self, key)
            if value is not None:
                out[key] = value
        for key in ("start", "stop"):
            value = getattr(self, key)
            if value is not None:
                out[key] = _iso(value)
        return out

    def hash_form(self) -> dict[str, Any]:
        """:meth:`canonical` plus the entry's number, for a stage signature.

        A stage hashes only the entries that can change it, so the number is
        carried explicitly: it names the entry's attributes in the product,
        and removing an earlier entry renumbers the ones after it.
        """
        return {**self.canonical(), "entry": self.entry}

    def provenance_attrs(self, n_windows: int, *, inherited: bool = False) -> dict[str, Any]:
        """Product attributes recording this entry and what it masked.

        ``_masked_in`` names the product whose windows ``_n_windows`` counts.
        *inherited* marks a shear entry recorded on a chi file: masked in the
        diss product, it reaches chi through ``epsilonMean``. A bound the entry
        does not give is absent: unbounded on that side.
        """
        base = f"{ATTR_PREFIX}_{self.entry}"
        attrs = {f"{base}_{key}": value for key, value in self.canonical().items()}
        attrs[f"{base}_masked_in"] = self.product
        attrs[f"{base}_n_windows"] = int(n_windows)
        if inherited:
            attrs[f"{base}_comment"] = INHERITED_COMMENT
        return attrs

    def probe_index(self, probe_names: list[str], product: str) -> int:
        """Index of this entry's probe in *probe_names*; raise unless exactly one.

        ``T1`` matches ``T1_dT1``, as for ``fp07_tau_scale``. An unmatched name
        raises rather than being ignored: a typo would leave the faulty probe in
        ``epsilonMean`` while the config claims it was removed, and nothing
        downstream would show it.
        """
        hits = [
            i
            for i, name in enumerate(probe_names)
            if name == self.probe or name.startswith(f"{self.probe}_")
        ]
        if len(hits) != 1:
            raise ValueError(
                f"{self.label}: matched {len(hits)} of the {product} probes "
                f"{probe_names}; expected exactly one"
            )
        return hits[0]

    def window_mask(self, n_time: int, *, P: Any = None, t: Any = None, product: str = "") -> Any:
        """Boolean ``(n_time,)`` array: True where the window is inside every bound.

        *P* is the per-window ``P_mean`` [dbar] and *t* the per-window time in
        epoch seconds; either may be ``None`` when the entry does not bound it,
        and raises when it does. A window whose ``P_mean`` (or time) is NaN
        cannot be placed, so it is not inside.
        """
        import numpy as np

        inside = np.ones(n_time, dtype=bool)
        with np.errstate(invalid="ignore"):
            if self.P_min is not None or self.P_max is not None:
                if P is None:
                    raise ValueError(
                        f"{self.label}: bounds pressure, but the {product} product "
                        "has no per-window P_mean"
                    )
                pres = np.asarray(P, dtype=np.float64)
                if self.P_min is not None:
                    inside &= pres >= self.P_min
                if self.P_max is not None:
                    inside &= pres < self.P_max
            if self.start is not None or self.stop is not None:
                if t is None:
                    raise ValueError(
                        f"{self.label}: bounds time, but the {product} product's "
                        "window times carry no absolute reference"
                    )
                when = np.asarray(t, dtype=np.float64)
                if self.start is not None:
                    inside &= when >= self.start
                if self.stop is not None:
                    inside &= when < self.stop
        return inside


def parse_exclusions(instrument: Any, entries: Any) -> list[ProbeExclusion]:
    """Validate an ``instruments.<key>.exclude`` list into entries.

    Raises ``ValueError`` naming the instrument and the (1-based) entry for an
    entry that is not a mapping, carries an unknown key, names no recognizable
    probe, has no ``reason``, or gives an empty or unparseable window
    (``P_min >= P_max``, ``start >= stop``, a time that is not ISO-8601).
    ``None`` is no entries.
    """
    if entries is None:
        return []
    if not isinstance(entries, list):
        raise ValueError(
            f"instruments.{instrument}.exclude: must be a list of entries "
            f"(each a mapping with probe and reason), got {entries!r}"
        )
    out: list[ProbeExclusion] = []
    for n, raw in enumerate(entries, start=1):
        where = f"instruments.{instrument}.exclude entry {n}"
        if not isinstance(raw, dict):
            raise ValueError(f"{where}: must be a mapping, got {raw!r}")
        unknown = set(raw) - ENTRY_KEYS
        if unknown:
            raise ValueError(
                f"{where}: unknown key(s) {sorted(str(k) for k in unknown)}. "
                f"Valid keys: {sorted(ENTRY_KEYS)}"
            )
        probe = raw.get("probe")
        therm = THERMISTOR_PROBE.match(probe) if isinstance(probe, str) else None
        if not isinstance(probe, str) or not (SHEAR_PROBE.match(probe) or therm):
            raise ValueError(
                f"{where}: probe must name a shear probe (sh1, sh2, ...) or a "
                f"thermistor (T1, T2, ... or T1_dT1, ...), got {probe!r}"
            )
        where = f"{where} (probe {probe!r})"
        # One bead, one spelling: T1_dT1 is stored as T1, so the two spellings
        # match, hash and record identically.
        if therm:
            probe = f"T{therm.group(1)}"
        # Required: an exclusion deletes data a person judged bad, and the
        # product records why. Without it the mask is untraceable.
        reason = raw.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"{where}: reason is required (a non-empty string)")
        P_min = _parse_pressure(raw.get("P_min"), where, "P_min")
        P_max = _parse_pressure(raw.get("P_max"), where, "P_max")
        if P_min is not None and P_max is not None and P_min >= P_max:
            raise ValueError(
                f"{where}: P_min ({P_min:g}) must be less than P_max ({P_max:g}); "
                "the window [P_min, P_max) would be empty"
            )
        start = _parse_time(raw.get("start"), where, "start")
        stop = _parse_time(raw.get("stop"), where, "stop")
        if start is not None and stop is not None and start >= stop:
            raise ValueError(
                f"{where}: start ({_iso(start)}) must be before stop ({_iso(stop)}); "
                "the window [start, stop) would be empty"
            )
        out.append(
            ProbeExclusion(
                instrument=str(instrument),
                entry=n,
                probe=probe,
                reason=reason.strip(),
                P_min=P_min,
                P_max=P_max,
                start=start,
                stop=stop,
            )
        )
    return out

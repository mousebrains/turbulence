# Aug-2026, Claude and Pat Welch, pat@mousebrains.com
"""Inputs to the calibration: the probe side and the reference side.

Two containers, deliberately kept apart:

``ProbeSeries``
    One ``.p`` file's thermistor counts on the instrument clock, in epoch
    seconds, with the bridge constants and detected profiles.

``ReferenceSeries``
    The CTD's temperature on the CTD's OWN clock, in epoch seconds --- **the
    real samples only**.

The separation is the point.  ``perturb/hotel.py`` merges the CTD onto the
instrument's grid by interpolating across arbitrary gaps and edge-holding
outside coverage, so on a glider that samples CT every n-th yo the merged
``sci_water_temp`` channel is a fabricated ramp over most of the record --- and
a fit against it is a fit against fiction (plan findings A1, section 3.1).
``ReferenceSeries`` therefore never interpolates and never extrapolates.  Its
``valid_spans`` say where real samples sit closely enough together to be usable
at all; everywhere else there is simply no reference, and the calibration
contributes no data rather than inventing some.

Confirmed empirically: NaN-marking a gap --- what ``dinkum-hotel``'s
``projection.max_gap`` does --- produces byte-identical output from the perturb
loader, so builder-side gap control does not survive the merge.  Reading the
hotel file directly here is the only way to honour it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from odas_tpw.fp07cal.logr import BridgeParams, config_to_coeffs, live_beta_key


@dataclass
class ProbeSeries:
    """One ``.p`` file's thermistor channels on the instrument clock."""

    label: str
    time: np.ndarray  # epoch seconds, ascending
    counts: dict[str, np.ndarray]  # channel -> raw counts, same length as time
    pressure: np.ndarray  # dbar, same length as time
    bridge: dict[str, BridgeParams]  # channel -> bridge constants
    factory: dict[str, np.ndarray]  # channel -> factory Steinhart-Hart [a0, a1, ...]
    beta_key: dict[str, str] = field(default_factory=dict)  # channel -> "beta"|"beta_1"
    profiles: list[tuple[int, int]] = field(default_factory=list)  # inclusive index spans
    instrument_sn: str = "?"
    probe_sn: dict[str, str] = field(default_factory=dict)
    speed: np.ndarray | None = None  # m/s, for the flushing gate

    @property
    def fs(self) -> float:
        if self.time.size < 2:
            return float("nan")
        return 1.0 / float(np.median(np.diff(self.time)))

    def profile_direction(self, span: tuple[int, int]) -> int:
        """+1 descending (dive), -1 ascending (climb), 0 indeterminate."""
        s, e = span
        if e <= s:
            return 0
        dp = float(self.pressure[e] - self.pressure[s])
        return 1 if dp > 0 else (-1 if dp < 0 else 0)

    def in_profile_mask(self) -> np.ndarray:
        """True where the sample lies inside any detected profile."""
        mask = np.zeros(self.time.size, dtype=bool)
        for s, e in self.profiles:
            mask[s : e + 1] = True
        return mask

    def profile_id(self) -> np.ndarray:
        """Per-sample profile index, -1 outside every profile."""
        out = np.full(self.time.size, -1, dtype=np.int32)
        for i, (s, e) in enumerate(self.profiles):
            out[s : e + 1] = i
        return out


@dataclass
class ReferenceSeries:
    """CTD temperature on the CTD's own clock --- real samples only."""

    time: np.ndarray  # epoch seconds, strictly increasing
    value: np.ndarray  # degC
    source: str = "?"
    pressure: np.ndarray | None = None  # the CTD's OWN pressure, if carried

    def valid_spans(self, max_gap: float) -> list[tuple[int, int]]:
        """Inclusive index runs whose consecutive spacing never exceeds *max_gap*.

        A "span" is a stretch of genuinely continuous CTD sampling.  Between
        spans there is no reference at all --- not a wide one, not an
        interpolated one.  This is what makes every-n-th-yo sampling a
        non-event rather than a source of invented data.
        """
        if self.time.size == 0:
            return []
        if self.time.size == 1:
            return [(0, 0)]
        brk = np.flatnonzero(np.diff(self.time) > max_gap)
        starts = np.concatenate(([0], brk + 1))
        ends = np.concatenate((brk, [self.time.size - 1]))
        return [(int(s), int(e)) for s, e in zip(starts, ends)]

    def median_interval(self) -> float:
        if self.time.size < 2:
            return float("nan")
        return float(np.median(np.diff(self.time)))

    def coverage_report(self, max_gap: float) -> dict:
        spans = self.valid_spans(max_gap)
        durations = [float(self.time[e] - self.time[s]) for s, e in spans]
        total = float(self.time[-1] - self.time[0]) if self.time.size > 1 else 0.0
        return {
            "n_samples": int(self.time.size),
            "n_spans": len(spans),
            "median_interval_s": self.median_interval(),
            "covered_s": float(sum(durations)),
            "total_s": total,
            "duty_cycle": (sum(durations) / total) if total > 0 else float("nan"),
            "T_min": float(np.nanmin(self.value)) if self.value.size else float("nan"),
            "T_max": float(np.nanmax(self.value)) if self.value.size else float("nan"),
        }


def sanitize_reference(
    time: np.ndarray,
    value: np.ndarray,
    *,
    pressure: np.ndarray | None = None,
    valid_min: float = -5.0,
    valid_max: float = 45.0,
    time_min: float = 100.0,
    time_max: float | None = None,
    source: str = "?",
) -> ReferenceSeries:
    """Drop unusable reference samples and enforce a strictly increasing clock.

    Slocum repeats the last CTD timestamp on rows the CTD did not refresh and
    writes ``0.0`` where the field was never set, so duplicate and zero stamps
    are normal, not exceptional.  Duplicates are collapsed to their mean rather
    than dropped --- dropping would bias toward whichever row happened to sort
    first.
    """
    t = np.asarray(time, dtype=np.float64)
    v = np.asarray(value, dtype=np.float64)
    if t.shape != v.shape:
        raise ValueError(f"reference time {t.shape} and value {v.shape} differ in shape")
    p = None if pressure is None else np.asarray(pressure, dtype=np.float64)

    if time_max is None:
        time_max = 4.0e9  # ~2096; a stamp beyond this is a decode error, not a date
    keep = (
        np.isfinite(t)
        & np.isfinite(v)
        & (t >= time_min)
        & (t <= time_max)
        & (v >= valid_min)
        & (v <= valid_max)
    )
    t, v = t[keep], v[keep]
    if p is not None:
        p = p[keep]

    order = np.argsort(t, kind="stable")
    t, v = t[order], v[order]
    if p is not None:
        p = p[order]

    if t.size:
        uniq, inv = np.unique(t, return_inverse=True)
        if uniq.size != t.size:
            sums = np.zeros(uniq.size)
            cnts = np.zeros(uniq.size)
            np.add.at(sums, inv, v)
            np.add.at(cnts, inv, 1.0)
            v = sums / cnts
            if p is not None:
                psums = np.zeros(uniq.size)
                np.add.at(psums, inv, p)
                p = psums / cnts
            t = uniq

    return ReferenceSeries(time=t, value=v, source=source, pressure=p)


def load_hotel_reference(
    path: str | Path,
    *,
    time_var: str = "sci_ctd41cp_timestamp",
    value_var: str = "sci_water_temp",
    pressure_var: str | None = "sci_water_pressure",
    pressure_scale: float = 1.0,
    valid_min: float = -5.0,
    valid_max: float = 45.0,
) -> ReferenceSeries:
    """Read the CTD reference straight out of a hotel NetCDF.

    Deliberately bypasses ``perturb.hotel``: that path interpolates across gaps
    and edge-holds outside coverage (plan A1).  Here the file's own samples are
    all we take.

    ``pressure_scale`` exists because a Slocum reports ``sci_water_pressure`` in
    **bar**.  A hotel file built by ``dinkum-hotel`` has already applied the
    factor of 10 (see its ``sensors.sci_water_pressure.scale``), so the default
    of 1.0 is right for that path --- but reading a raw converted ``ebd.nc``
    directly, as the calibration tooling can, it has not, and the pressure
    arrives 10x small.  Lag estimates are correlation-based and so immune, but
    anything that compares the reference's pressure to the instrument's in
    physical units is not.  Pass 10.0 for raw Slocum units.
    """
    import xarray as xr

    with xr.open_dataset(path) as ds:
        for name in (time_var, value_var):
            if name not in ds.variables:
                raise KeyError(
                    f"{path}: variable {name!r} not found; available: {sorted(ds.variables)}"
                )
        t = np.asarray(ds[time_var].values, dtype=np.float64).ravel()
        v = np.asarray(ds[value_var].values, dtype=np.float64).ravel()
        p = None
        if pressure_var and pressure_var in ds.variables:
            p = np.asarray(ds[pressure_var].values, dtype=np.float64).ravel()
            if pressure_scale != 1.0:
                p = p * pressure_scale
    return sanitize_reference(
        t, v, pressure=p, valid_min=valid_min, valid_max=valid_max,
        source=f"{Path(path).name}:{value_var}",
    )


_THERM_TYPES = ("therm", "thermistor")

# Base thermistor channels only.  A .p config also carries the pre-emphasised
# variants (T1_dT1, T2_dT2) as type "therm", but those stanzas hold no bridge
# constants --- they inherit the base channel's --- so auto-selecting them makes
# BridgeParams.from_channel_config raise on every file.  They are the
# DERIVATIVE channel in any case, not a temperature to calibrate against a CTD.
_BASE_THERM = re.compile(r"^T\d+$")


def load_probe_series(
    path: str | Path,
    *,
    channels: list[str] | None = None,
    profiles: list[tuple[int, int]] | None = None,
    speed_var: str | None = "U_EM",
    W_min: float = 0.05,
    P_min: float = 0.5,
    min_duration: float = 60.0,
) -> ProbeSeries:
    """Build a :class:`ProbeSeries` from a ``.p`` file on the SLOW grid.

    The slow grid is the right one for calibration: the reference is ~1 Hz, so
    the fast channels carry three decades of bandwidth the CTD cannot see, and
    averaging them into the reference's sampling kernel (see ``pairs``) is what
    makes the regression bandwidth-matched instead of attenuated (plan A2).
    """
    from odas_tpw.rsi.p_file import PFile

    pf = PFile(str(path))
    t0 = pf.start_time.timestamp()
    t = t0 + np.asarray(pf.t_slow, dtype=np.float64)
    n = t.size

    by_name = {str(c.get("name", "")).strip(): dict(c) for c in pf.config["channels"]}
    wanted = channels or sorted(
        name
        for name, cfg in by_name.items()
        if str(cfg.get("type", "")).strip().lower() in _THERM_TYPES and _BASE_THERM.match(name)
    )

    counts: dict[str, np.ndarray] = {}
    bridge: dict[str, BridgeParams] = {}
    factory: dict[str, np.ndarray] = {}
    beta_key: dict[str, str] = {}
    probe_sn: dict[str, str] = {}
    ratio = max(1, round(float(pf.fs_fast) / float(pf.fs_slow)))
    for name in wanted:
        if name not in pf.channels_raw or name not in by_name:
            continue
        raw = np.asarray(pf.channels_raw[name], dtype=np.float64)
        if pf.is_fast(name):
            raw = raw[::ratio]
        counts[name] = raw[:n] if raw.size >= n else np.pad(raw, (0, n - raw.size), constant_values=np.nan)
        cfg = by_name[name]
        bridge[name] = BridgeParams.from_channel_config(cfg, name)
        factory[name] = config_to_coeffs(cfg)
        beta_key[name] = live_beta_key(cfg)
        probe_sn[name] = str(cfg.get("sn", "") or "").strip() or "(no SN)"

    P = np.asarray(pf.channels.get("P", np.full(n, np.nan)), dtype=np.float64)[:n]
    speed = None
    if speed_var and speed_var in pf.channels:
        speed = np.abs(np.asarray(pf.channels[speed_var], dtype=np.float64)[:n])

    if profiles is None:
        # A glider yo is BOTH legs: direction="glide" keeps dive and climb,
        # which the dive/climb residual split (the sharpest lag diagnostic)
        # depends on having both of.  W_min defaults to the glider value --- the
        # VMP-tuned 0.3 dbar/s rejects every glide.
        from odas_tpw.rsi.profile import _smooth_fall_rate
        from odas_tpw.scor160.profile import get_profiles

        try:
            W = _smooth_fall_rate(P, float(pf.fs_slow))
            profiles = [
                (int(a), int(b))
                for a, b in get_profiles(
                    P, W, float(pf.fs_slow), P_min=P_min, W_min=W_min,
                    direction="glide", min_duration=min_duration,
                )
            ]
        except Exception:
            profiles = []

    return ProbeSeries(
        label=Path(path).name,
        time=t,
        counts=counts,
        pressure=P,
        bridge=bridge,
        factory=factory,
        beta_key=beta_key,
        profiles=profiles,
        instrument_sn=str(pf.config.get("instrument_info", {}).get("sn", "?") or "?"),
        probe_sn=probe_sn,
        speed=speed,
    )

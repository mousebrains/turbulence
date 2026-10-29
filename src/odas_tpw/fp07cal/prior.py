# Oct-2026, Claude and Pat Welch, pat@mousebrains.com
"""Fitting a reused bead with its prior from the calibration sheet (``fit.prior``).

A bead that has been calibrated before carries a prior in
``microstructure_sensors/fp07_calibrations.yaml`` (:meth:`Registry.prior_for`):
the shape (``beta`` terms) of its widest-range calibration, and a ``t_0``.  Its
shape carries between deployments to ~5 mK, its ``t_0`` only to ~15 mK (issue
#189), so on a new deployment the prior is worth most where the deployment
itself cannot pin the shape: a narrow temperature range, where a full fit's
shape is noise and extrapolates badly.

So ``fp07-cal fit`` with a ``fit.prior`` block fits two ways:

* the **full fit**, orders 1..3, the order chosen by held-out error (today's
  fit, unchanged);
* an **offset-only fit**: the prior's shape held fixed and ``t_0`` alone
  fitted, with the same robust iteration, jointly with the sensor-geometry
  term when that is on (unknowns ``t_0``, ``dz``, ``tau``);

and chooses between them with the same temperature-split held-out error that
chooses the order, every candidate on the same folds
(:func:`odas_tpw.fp07cal.fit.held_out_errors`): the prior's shape is one more
candidate.  The full fit has to be CLEARLY better to win (``tie_fraction``,
default 5%), and it never wins when it is not a plausible curve for this bead
(:func:`plausibility`: NTC, and dT/dL within 10% of the prior's) or when the
deployment's range is narrower than the span over which the sheet knows a
bead's shape.  ``mode: offset`` and ``mode: full`` force the choice instead.

Whichever wins, the result is checked against the prior at the same L over the
overlap of the two fitted ranges: 2 sigma of ``t_0`` at its center, and
2 sqrt(sigma_t0^2 + sigma_shape^2) anywhere in it (:func:`_qc`).  Beyond either
it is flagged as a possible unrecorded bead swap, reference problem or failing
bead.

With no ``fit.prior`` block none of this runs, and ``fp07-cal fit`` writes what
it wrote before, byte for byte.
"""

from __future__ import annotations

import hashlib
import math
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from odas_tpw.fp07cal import registry as _reg
from odas_tpw.fp07cal.fit import (
    FitResult,
    fit_calibration,
    held_out_errors,
    shipped_fit,
    temperature_folds,
)
from odas_tpw.fp07cal.pairs import PairSet

MODES = ("auto", "offset", "full")
OFFSET = "prior shape"
"""Name of the offset-only candidate in the record."""

QC_SIGMAS = 2.0
"""A fit further than this many prior ``t_0`` uncertainties from the prior is flagged."""

_KEYS = ("sheet", "mode", "beads", "tie_fraction")
_NONE = ("none", "off")


class PriorError(ValueError):
    """A ``fit.prior`` block that cannot be honored as written."""


@dataclass(frozen=True)
class PriorConfig:
    """The validated ``fit.prior`` block."""

    sheet: Path
    mode: str
    beads: dict[str, str | None]  # channel -> "auto" | a bead serial | None (no prior)
    tie_fraction: float = 0.05  # see TIE_FRACTION


def parse_prior_config(
    fit_cfg: dict | None, channels: list[str] | None = None
) -> PriorConfig | None:
    """``fit.prior`` -> :class:`PriorConfig`, or ``None`` when there is no block.

    Strict, because a typo here silently means "no prior": an unknown key, a
    missing ``sheet``, an unknown mode, or a ``beads`` entry that is neither
    ``auto``, ``none`` nor a bead serial is a :class:`PriorError`.  A ``beads``
    key that is not a fitted channel is refused too (*channels*, when given).
    """
    raw = (fit_cfg or {}).get("prior")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise PriorError(f"fit.prior: expected a mapping with `sheet:`, got {raw!r}")
    unknown = sorted(str(k) for k in raw if k not in _KEYS)
    if unknown:
        raise PriorError(f"fit.prior: unknown key(s) {unknown}; allowed: {list(_KEYS)}")
    sheet = raw.get("sheet")
    if not isinstance(sheet, str) or not sheet.strip():
        raise PriorError(
            "fit.prior.sheet is required: the path to fp07_calibrations.yaml (it is not "
            "installed with the package; <CONFIG_DIR> works as elsewhere)"
        )
    mode = raw.get("mode", "auto")
    if mode not in MODES:
        raise PriorError(f"fit.prior.mode: {mode!r} is not one of {list(MODES)}")
    beads_raw = raw.get("beads") or {}
    if not isinstance(beads_raw, dict):
        raise PriorError(
            f"fit.prior.beads: expected a mapping channel -> serial, got {beads_raw!r}"
        )
    beads: dict[str, str | None] = {}
    for ch, v in beads_raw.items():
        if channels is not None and ch not in channels:
            raise PriorError(
                f"fit.prior.beads.{ch}: {ch!r} is not one of the fitted channels {channels}"
            )
        if v is None or (isinstance(v, str) and v.strip().lower() in _NONE):
            beads[str(ch)] = None
        elif isinstance(v, str) and v.strip().lower() == "auto":
            beads[str(ch)] = "auto"
        elif _reg.is_bead_serial(v):
            beads[str(ch)] = str(v)
        else:
            raise PriorError(
                f"fit.prior.beads.{ch}: {v!r} is not `auto`, `none` or a bead serial "
                f"T<number> (the placeholder `T` is not an identity)"
            )
    tie = raw.get("tie_fraction", TIE_FRACTION)
    if isinstance(tie, bool) or not isinstance(tie, (int, float)) or not 0 <= tie < 1:
        raise PriorError(f"fit.prior.tie_fraction: {tie!r} must be a number in [0, 1)")
    return PriorConfig(sheet=Path(sheet), mode=str(mode), beads=beads, tie_fraction=float(tie))


def load_sheet(pcfg: PriorConfig) -> tuple[_reg.Registry, str]:
    """The sheet and the SHA-256 of the exact bytes it was parsed from."""
    path = pcfg.sheet
    if not path.exists():
        raise PriorError(f"fit.prior.sheet {path} does not exist")
    raw = path.read_bytes()
    try:
        sheet = _reg.loads(raw.decode("utf-8"), source=path)
    except (_reg.RegistryError, UnicodeDecodeError) as exc:
        raise PriorError(f"fit.prior.sheet {path}: {exc}") from exc
    return sheet, hashlib.sha256(raw).hexdigest()


@dataclass
class ChannelPrior:
    """One channel's bead, its prior, and how this channel's electronics compare."""

    channel: str
    bead: str | None
    bead_from: str  # "files", "config", or "off"
    file_serials: list[str]
    sheet: Path
    sheet_sha256: str
    prior: _reg.Prior | None = None
    reason: str = ""  # why there is no prior
    bridge_this: dict[str, float] = field(default_factory=dict)
    bridge_prior: dict[str, float] = field(default_factory=dict)
    bridge_differs: list[str] = field(default_factory=list)
    bridge_varies_between_files: bool = False
    electronics_differ: bool = False
    sigma_mK: float = float("nan")


def _same(a: float, b: float) -> bool:
    # The tolerance `fp07-cal patch` uses to compare bridge constants.
    return abs(a - b) <= 1e-9 * max(1.0, abs(b))


def resolve_priors(
    pcfg: PriorConfig,
    probes: list,
    channels: list[str],
    *,
    span: tuple[float, float] | None,
) -> dict[str, ChannelPrior]:
    """Each channel's bead and prior, refusing anything ambiguous.

    The bead comes from the files' own configs unless ``fit.prior.beads``
    names it.  Files of one run that disagree on a channel's serial, or carry
    a placeholder (``T``, no serial), are refused: a prior keyed on the wrong
    bead is worse than none.  *span* (epoch seconds) is the time the fit
    covers: every calibration of the bead whose dates overlap it is excluded.
    """
    sheet, sha = load_sheet(pcfg)
    insts = sorted({str(p.instrument_sn).strip() for p in probes})
    if len(insts) > 1:
        raise PriorError(
            f"fit.prior: the files come from more than one instrument {insts}; a prior "
            f"needs one target"
        )
    inst = insts[0] if insts else "?"
    target = _reg._target_serial(inst) if inst not in ("", "?") else None
    if target is not None and not target.isdigit():
        target = None
    excluding = None
    if span is not None:
        excluding = (datetime.fromtimestamp(span[0], UTC), datetime.fromtimestamp(span[1], UTC))

    out: dict[str, ChannelPrior] = {}
    for ch in channels:
        have = [p for p in probes if ch in p.counts]
        if not have:
            continue
        serials = sorted({str(p.probe_sn.get(ch, "")).strip() for p in have})
        spec = pcfg.beads.get(ch, "auto")
        sheet_path = pcfg.sheet.resolve()
        if spec is None:
            out[ch] = ChannelPrior(channel=ch, bead=None, bead_from="off",
                                   file_serials=serials, sheet=sheet_path, sheet_sha256=sha,
                                   reason=f"fit.prior.beads.{ch} is none")
            continue
        if spec == "auto":
            if len(serials) != 1:
                raise PriorError(
                    f"{ch}: the files disagree on the bead serial ({', '.join(serials)}). A "
                    f"run is one bead per channel: split it at the change, or name the bead "
                    f"with fit.prior.beads.{ch}"
                )
            bead = serials[0]
            if not _reg.is_bead_serial(bead):
                raise PriorError(
                    f"{ch}: the files' bead serial {bead!r} is a placeholder, not an identity. "
                    f"Name the bead with fit.prior.beads.{ch}: T<number>, or `none` for no "
                    f"prior on this channel"
                )
            bead_from = "files"
        else:
            bead, bead_from = spec, "config"
            named = [s for s in serials if _reg.is_bead_serial(s) and s != bead]
            if named:
                print(f"  NOTE {ch}: fit.prior.beads names {bead}, but the files say "
                      f"{', '.join(named)}; the prior is {bead}'s")
        cp = ChannelPrior(channel=ch, bead=bead, bead_from=bead_from, file_serials=serials,
                          sheet=sheet_path, sheet_sha256=sha)
        try:
            cp.prior = sheet.prior_for(
                bead, instrument=target, channel=ch if target else None, excluding=excluding,
            )
        except _reg.RegistryError as exc:
            if pcfg.mode == "offset":
                raise PriorError(
                    f"{ch}: fit.prior.mode is offset but bead {bead} has no prior: {exc}"
                ) from exc
            cp.reason = str(exc)
            out[ch] = cp
            continue
        bridges = [p.bridge[ch].as_dict() for p in have if ch in p.bridge]
        cp.bridge_this = dict(bridges[0]) if bridges else {}
        cp.bridge_prior = cp.prior.bridge
        cp.bridge_varies_between_files = any(
            any(not _same(b[k], bridges[0][k]) for k in bridges[0]) for b in bridges[1:]
        )
        cp.bridge_differs = [
            k for k in _reg.BRIDGE_KEYS
            if k not in cp.bridge_this or not _same(cp.bridge_this[k], cp.bridge_prior[k])
        ]
        # Electronics differ when the sheet's serial/channel rule says so, when
        # the constants that define L differ, or when the instrument is unknown
        # (nothing then says they are the same).
        cp.electronics_differ = bool(
            cp.prior.cross_instrument or cp.bridge_differs or cp.bridge_varies_between_files
            or target is None
        )
        cp.sigma_mK = math.hypot(
            cp.prior.offset_mK, cp.prior.cross_instrument_mK if cp.electronics_differ else 0.0
        )
        out[ch] = cp
    return out


# ------------------------------------------------------------ the comparison
MAX_SLOPE_RATIO = 1.10
"""A full fit whose dT/dL leaves 1/1.10 .. 1.10 of the prior's, anywhere over its L range,
is not this bead's curve.

Across the sheet's 23 calibrations of 12 beads, dT/dL at the same temperature
differs by at most 1.65% between ANY two (0.53% within one bead; measured on
the sheet of PR #239), so +/-10% is ~6x the widest spread of real FP07s, while
fits to isothermal noise were off by 20x or more.
"""

TIE_FRACTION = 0.05
"""Default ``fit.prior.tie_fraction``: the full fit must beat the prior's held-out error by more.

On a wide (14-28.5 degC) synthetic, a prior shape wrong by the sheet's own
shape term (6.7 mK p-p) lost to the full fit by only +0.29 mK of 6.9 (4%, n=8,
MAD 0.23), and the seed-to-seed scatter of the margin was 1-4% of the error.
A margin under 5% is therefore within what the sheet already allows the
shape; a shape error of 14 mK still loses by +3.2 mK (47%).
"""

DROP_EXCESS = 0.02
"""Warn when the offset-only fit's robust loop drops this much more than the full fit's.

A fixed shape that does not fit the deployment leaves the ends of the range
outside the 4-sigma clip, and its rms over the kept pairs then looks better
than it is (beta_2 x 0.9 on a 14-28.5 degC synthetic: 7% dropped, 3.9 mK over
the kept pairs, 6.0 mK over all of them).
"""


def _fit_one(pairs: PairSet, *, order: int = 1, fixed: dict[str, float] | None = None,
             robust: bool, geometry: bool):
    """One candidate, as shipped; returns ``(fit, geo)`` (``geo`` None without geometry)."""
    if geometry:
        from odas_tpw.fp07cal.geometry import joint_fit

        return joint_fit(pairs, order=order, robust=robust, fixed_terms=fixed)
    return fit_calibration(pairs, order=order, robust=robust, fixed_terms=fixed), None


def _inv_and_dTdL(coeffs: np.ndarray, L: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    inv = sum(a * L**i for i, a in enumerate(coeffs))
    d_inv = sum(i * a * L ** (i - 1) for i, a in enumerate(coeffs) if i)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.asarray(inv), np.asarray(-d_inv / inv**2)


def is_ntc(fit: FitResult, n: int = 201) -> bool:
    """Does T fall as L rises (and ``1/T_K`` stay positive) over the fit's own L range?

    Necessary, not sufficient: a fit to isothermal noise can have the right sign
    and a slope 20x too steep or too shallow (see :func:`plausibility`).
    """
    lo, hi = fit.L_range
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return False
    inv, s = _inv_and_dTdL(fit.coeffs, np.linspace(lo, hi, n))
    return bool(np.all(inv > 0) and np.all(s < 0))


def plausibility(fit: FitResult, prior_poly: np.ndarray, n: int = 201
                 ) -> tuple[bool, str, list[float]]:
    """Is *fit* an FP07 curve like this bead's, over its own L range?

    It must be NTC (:func:`is_ntc`) and its dT/dL within :data:`MAX_SLOPE_RATIO`
    of the prior's at the same L everywhere in the range.  A deployment with no
    thermal signal (isothermal water) can produce a full fit that wins on
    held-out error --- a flat model predicts noise best --- with a slope far
    from any FP07's, and the curves still agree where the data are.  Returns
    ``(ok, why not, [min, max] slope ratio)``.
    """
    lo, hi = fit.L_range
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return False, "it has no L range", [float("nan")] * 2
    L = np.linspace(lo, hi, n)
    inv, s = _inv_and_dTdL(fit.coeffs, L)
    _inv_p, sp = _inv_and_dTdL(np.asarray(prior_poly, dtype=np.float64), L)
    r = s / sp
    rr = [float(np.nanmin(r)), float(np.nanmax(r))] if np.any(np.isfinite(r)) \
        else [float("nan")] * 2
    if not (np.all(inv > 0) and np.all(s < 0)):
        return False, "T does not fall as L rises over its range (not an FP07 curve)", rr
    if not (np.all(np.isfinite(r)) and rr[0] >= 1 / MAX_SLOPE_RATIO
            and rr[1] <= MAX_SLOPE_RATIO):
        return False, (f"its dT/dL is {rr[0]:.3g}-{rr[1]:.3g} x the prior's at the same L "
                       f"(an FP07's is within 1/{MAX_SLOPE_RATIO:g}-{MAX_SLOPE_RATIO:g})"), rr
    return True, "", rr


def _delta_mK(coeffs: np.ndarray, prior: _reg.Prior, T: float) -> float | None:
    try:
        return _reg.poly_difference_mK(np.asarray(coeffs, dtype=np.float64), prior.poly, T)
    except ValueError:
        return None


def _f(x: float | None, fmt: str = ".2f") -> str:
    return "—" if x is None or not np.isfinite(x) else format(x, fmt)


def _qc(fit: FitResult, cp: ChannelPrior, prior: _reg.Prior) -> dict[str, Any]:
    """The exported curve against the prior's, at the same L, over their common range.

    Two limits: at the center of the overlap the prior's ``t_0`` term alone
    (2 sigma_t0), and anywhere else in it the ``t_0`` and shape terms together
    (2 sqrt(sigma_t0^2 + sigma_shape^2)), since the shape is only known to
    sigma_shape away from where the curves are pinned.  With no overlap the
    comparison runs over this deployment's own range, where the prior is
    extrapolated, with the wider limit at every point, and says so.
    """
    T_lo, T_hi = fit.T_range
    p_lo, p_hi = prior.valid_T
    lo, hi = max(T_lo, p_lo), min(T_hi, p_hi)
    extrapolated = not lo < hi
    if extrapolated:
        lo, hi = T_lo, T_hi
    grid = np.linspace(lo, hi, 21) if hi > lo else np.array([lo])
    T_c = 0.5 * (lo + hi)
    sig_c = cp.sigma_mK
    sig_s = math.hypot(cp.sigma_mK, prior.shape_mK)
    lim_c = QC_SIGMAS * (sig_s if extrapolated else sig_c)
    lim_s = QC_SIGMAS * sig_s
    d_c = _delta_mK(fit.coeffs, prior, T_c)
    ds = [_delta_mK(fit.coeffs, prior, float(T)) for T in grid]
    out: dict[str, Any] = {
        "range_T_degC": [float(lo), float(hi)], "extrapolated": extrapolated,
        "center_T_degC": T_c, "delta_center_mK": d_c,
        "delta_ends_mK": [ds[0], ds[-1]],
        "limit_center_mK": lim_c, "limit_range_mK": lim_s,
        "sigma_t0_mK": sig_c, "sigma_shape_mK": float(prior.shape_mK),
    }
    if d_c is None or any(d is None for d in ds):
        out.update(verdict="flag", max_abs_delta_mK=None, at_T_degC=None, message=(
            f"the prior does not read every temperature in {lo:.2f}-{hi:.2f} degC on its "
            f"physical branch, so the fit cannot be compared with it"))
        return out
    vals = np.array(ds, dtype=np.float64)
    k = int(np.argmax(np.abs(vals)))
    out.update(max_abs_delta_mK=float(abs(vals[k])), at_T_degC=float(grid[k]))
    bad_c = abs(d_c) > lim_c
    bad_s = abs(vals[k]) > lim_s
    where = (f"over {lo:.2f}-{hi:.2f} degC, "
             + ("this deployment's range (the ranges do not overlap, so the prior is "
                "extrapolated there and the shape term applies everywhere)"
                if extrapolated else "the overlap of the two fitted ranges"))
    numbers = (f"{d_c:+.1f} mK at the center ({T_c:.2f} degC, limit {lim_c:.0f} mK), "
               f"{vals[0]:+.1f} / {vals[-1]:+.1f} mK at the ends, max |delta| "
               f"{abs(vals[k]):.1f} mK at {grid[k]:.2f} degC (limit {lim_s:.0f} mK)")
    if bad_c or bad_s:
        out.update(verdict="flag", message=(
            f"{numbers}, {where}: beyond the prior's uncertainty, so a possible unrecorded "
            f"bead swap, reference problem or failing bead (or a wrong serial in the "
            f"config). Check before trusting or recording this fit."))
    else:
        out.update(verdict="ok", message=f"{numbers}, {where}: within the prior's uncertainty")
    return out


def choose(
    pairs: PairSet,
    cp: ChannelPrior,
    *,
    mode: str,
    tie_fraction: float = TIE_FRACTION,
    full_fit: FitResult,
    full_geo,
    orders: tuple[int, ...],
    robust: bool,
    geometry: bool,
    exclude: np.ndarray | None = None,
):
    """The prior's shape against the full fit; returns ``(fit, geo, record)``.

    *exclude* is :func:`odas_tpw.fp07cal.fit.outlier_screen`'s mask, the same
    one the order was chosen with: those pairs are left out of the folds.

    *full_fit*/*full_geo* are what ``fp07-cal fit`` would have shipped without
    a prior; *orders* the full orders it chose among (``(1, 2, 3)`` for
    ``fit.order: auto``, else the forced one).  Every candidate is scored on
    the SAME temperature folds (:func:`held_out_errors`) and its in-sample rms
    is reported over the SAME pairs (all of them), beside the rms over the
    pairs its own robust loop kept.

    In ``auto`` the prior's shape is kept when no full order is a plausible
    FP07 curve for this bead (:func:`plausibility`), when the deployment's
    range is narrower than the sheet's shape span (held-out error over it
    cannot test a shape beyond the data), or when its held-out error is within
    *tie_fraction* of the full fit's: the full fit has to be clearly better.
    An implausible held-out-best order gives way to the next plausible one.
    ``fit`` is ``None`` when ``mode: offset`` and the offset-only fit failed;
    the record says why and the channel exports nothing.
    """
    prior = cp.prior
    assert prior is not None
    shape = prior.fixed_terms
    fits: dict[str, tuple[FitResult, Any]] = {}
    errors: dict[str, str] = {}
    for o in orders:
        name = f"order {o}"
        if o == full_fit.order:
            fits[name] = (full_fit, full_geo)
            continue
        try:
            fits[name] = _fit_one(pairs, order=o, robust=robust, geometry=geometry)
        except Exception as exc:
            errors[name] = f"{type(exc).__name__}: {exc}"
    try:
        fits[OFFSET] = _fit_one(pairs, fixed=shape, robust=robust, geometry=geometry)
    except Exception as exc:  # the candidate is unavailable, not the run
        errors[OFFSET] = f"{type(exc).__name__}: {exc}"

    fitters: dict[str, Any] = {
        f"order {o}": (lambda sub, o=o: shipped_fit(sub, order=o, robust=robust,
                                                    use_geometry=geometry))
        for o in orders if f"order {o}" in fits
    }
    if OFFSET in fits:
        fitters[OFFSET] = lambda sub: shipped_fit(sub, robust=robust, use_geometry=geometry,
                                                  fixed_terms=shape)
    held, n_folds = held_out_errors(
        pairs, fitters, min_train=max([o + 5 for o in orders] + [5]),
        folds=temperature_folds(pairs, exclude),
    )
    resid = {n: np.asarray(f.residual_K, dtype=np.float64) for n, (f, _g) in fits.items()}
    common = (np.logical_and.reduce([np.isfinite(r) for r in resid.values()])
              if resid else np.zeros(len(pairs), dtype=bool))

    candidates: dict[str, dict[str, Any]] = {}
    for name in [f"order {o}" for o in orders] + [OFFSET]:
        c: dict[str, Any] = {"kind": "offset" if name == OFFSET else "full",
                             "order": prior.order if name == OFFSET else int(name.split()[1])}
        if name in fits:
            f = fits[name][0]
            ok, why_not, ratio = plausibility(f, prior.poly)
            n_all = f.n + f.n_dropped
            c.update(
                held_out_K=held.get(name),
                rms_all_K=(float(np.sqrt(np.mean(resid[name][common] ** 2)))
                           if common.any() else None),
                in_sample_K=f.rms_K,
                dropped_fraction=f.n_dropped / n_all if n_all else None,
                plausible=ok, slope_ratio=ratio,
                **({"implausible": why_not} if not ok else {}),
            )
            if name == OFFSET:
                c["t_0"] = f.config_equivalent["t_0"]
        else:
            c.update(held_out_K=None, rms_all_K=None, in_sample_K=None, dropped_fraction=None,
                     plausible=None, error=errors.get(name, "not fitted"))
        candidates[name] = c

    # The full candidate: today's choice, unless it is not a plausible FP07
    # curve and another order is (auto only; mode full keeps today's fit).
    default = f"order {full_fit.order}"
    full_name, full_note = default, ""
    if mode != "full" and not candidates[default]["plausible"]:
        alts = sorted((n for n in fitters if n != OFFSET and candidates[n]["plausible"]),
                      key=lambda n: (held.get(n) is None, held.get(n) or 0.0))
        if alts:
            full_name = alts[0]
            full_note = (f"{default}, the held-out best, is not plausible "
                         f"({candidates[default]['implausible']}); {full_name} is the best "
                         f"plausible order")
    full_fit_c, full_geo_c = fits[full_name]
    full_ok = bool(candidates[full_name]["plausible"])
    h_off, h_full = held.get(OFFSET), held.get(full_name)
    T_lo, T_hi = full_fit_c.T_range
    span = float(T_hi - T_lo)
    narrow = span < prior.shape_span_K[0]
    margin = (None if h_off is None or h_full is None or h_full <= 0
              else h_off / h_full - 1.0)

    if mode == "full":
        win, why = "full", "fit.prior.mode is full: the full fit is kept; the prior is used for QC"
    elif mode == "offset":
        if OFFSET not in fits:
            rec = no_prior_record(cp, mode)
            rec.update(status="failed", reason=f"fit.prior.mode is offset but the offset-only "
                       f"fit failed ({errors[OFFSET]})", candidates=candidates)
            print(f"  {pairs.channel}: {rec['reason']}; nothing exported for this channel",
                  file=sys.stderr)
            return None, None, rec
        win, why = "offset", "fit.prior.mode is offset: the prior's shape is held fixed"
    elif OFFSET not in fits:
        win, why = "full", f"the offset-only fit failed ({errors[OFFSET]}), so the full fit is kept"
    elif not full_ok:
        win = "offset"
        why = (f"no full order is a plausible FP07 curve for this bead ({full_name}: "
               f"{candidates[full_name]['implausible']}; typically water with no temperature "
               f"range, where a fit to noise wins on held-out error), so the prior's shape is "
               f"used")
    elif narrow:
        win = "offset"
        why = (f"the deployment spans {span:.2f} K, narrower than the {prior.shape_span_K[0]:g} K "
               f"over which the sheet knows a bead's shape to {prior.shape_mK:g} mK: held-out "
               f"error over it cannot test the shape the full fit would extrapolate, so the "
               f"prior's shape is kept")
    elif h_off is None:
        win, why = "full", ("the offset-only fit has no held-out error (a temperature fold was "
                            "too small), so the full fit is kept")
    elif h_full is None:
        win, why = "offset", (f"the full fit ({full_name}) has no held-out error, so the prior's "
                              f"shape ({h_off * 1e3:.2f} mK held-out) is used")
    elif h_off <= h_full * (1.0 + tie_fraction):
        win = "offset"
        why = (f"held-out {h_off * 1e3:.2f} mK with the prior's shape against "
               f"{h_full * 1e3:.2f} mK for the full fit ({full_name}), a margin of "
               f"{100 * (margin or 0.0):+.1f}%: within the {100 * tie_fraction:g}% tie band, "
               f"so the prior's shape is kept (the full fit must be clearly better)")
    else:
        win = "full"
        why = (f"held-out {h_full * 1e3:.2f} mK for the full fit ({full_name}) against "
               f"{h_off * 1e3:.2f} mK with the prior's shape, a margin of "
               f"{100 * (margin or 0.0):+.1f}%: beyond the {100 * tie_fraction:g}% tie band, "
               f"so this deployment constrains its own shape")
    if full_note and win == "full":
        why += f" ({full_note})"

    fit, geo = fits[OFFSET] if win == "offset" else (full_fit_c, full_geo_c)
    qc = _qc(fit, cp, prior)
    if not candidates[OFFSET if win == "offset" else full_name].get("plausible", True):
        reason = candidates[OFFSET if win == "offset" else full_name].get("implausible", "")
        qc["verdict"] = "flag"
        qc["message"] += f" The exported fit is not a plausible FP07 curve: {reason}."
    warnings: list[str] = []
    if OFFSET in fits and full_name in fits:
        d_off = candidates[OFFSET]["dropped_fraction"] or 0.0
        d_full = candidates[full_name]["dropped_fraction"] or 0.0
        if d_off - d_full > DROP_EXCESS:
            warnings.append(
                f"the offset-only fit's robust loop dropped {100 * d_off:.1f}% of the pairs "
                f"against {100 * d_full:.1f}% for the full fit ({full_name}): a sign that the "
                f"prior's shape does not fit this deployment (rms over all pairs "
                f"{_f(_mk(candidates[OFFSET]['rms_all_K']))} against "
                f"{_f(_mk(candidates[full_name]['rms_all_K']))} mK)")

    record = {
        "status": "used",
        "mode": mode,
        "bead": cp.bead,
        "bead_from": cp.bead_from,
        "file_serials": cp.file_serials,
        "sheet": str(cp.sheet),
        "sheet_sha256": cp.sheet_sha256,
        "source": {
            "deployment": prior.source.deployment,
            "instrument": prior.source.instrument.label(),
            "first": _reg._iso(prior.source.first),
            "last": _reg._iso(prior.source.last),
            "order": prior.order,
            "coefficients": prior.coefficients,
            "valid_T_degC": list(prior.valid_T),
            "fallback": prior.fallback,
        },
        "considered": [c.deployment for c in prior.considered],
        "sheet_cross_instrument": prior.cross_instrument,
        "bridge": {
            "this": cp.bridge_this,
            "prior": cp.bridge_prior,
            "differs": cp.bridge_differs,
            "varies_between_files": cp.bridge_varies_between_files,
        },
        "electronics_differ": cp.electronics_differ,
        "t_0_sigma_mK": cp.sigma_mK,
        "shape_mK": prior.shape_mK,
        "candidates": candidates,
        "n_folds": n_folds,
        "full_candidate": full_name,
        "margin": {
            "held_out_prior_K": h_off, "held_out_full_K": h_full, "fraction": margin,
            "tie_fraction": tie_fraction, "range_K": span,
            "shape_span_K": list(prior.shape_span_K), "narrow_range": narrow,
        },
        "choice": OFFSET if win == "offset" else full_name,
        "chosen_kind": win,
        "why": why,
        "fits": {
            "full": {"order": full_fit_c.order,
                     "config_equivalent": full_fit_c.config_equivalent},
            "offset": None if OFFSET not in fits else {
                "order": fits[OFFSET][0].order,
                "config_equivalent": fits[OFFSET][0].config_equivalent,
            },
        },
        "delta_vs_prior_mK": qc["delta_center_mK"],
        "delta_at_T_degC": qc["center_T_degC"],
        "overlap_T_degC": None if qc["extrapolated"] else qc["range_T_degC"],
        "delta_full_mK": _delta_mK(full_fit_c.coeffs, prior, qc["center_T_degC"]),
        "delta_offset_mK": (None if OFFSET not in fits
                            else _delta_mK(fits[OFFSET][0].coeffs, prior, qc["center_T_degC"])),
        "qc": qc,
        "warnings": warnings,
        "prior_notes": list(prior.notes),
    }
    _print_summary(record)
    return fit, geo, record


def no_prior_record(cp: ChannelPrior, mode: str) -> dict:
    """The record for a channel that has a ``fit.prior`` block but no prior."""
    return {
        "status": "off" if cp.bead_from == "off" else "none",
        "mode": mode,
        "bead": cp.bead,
        "bead_from": cp.bead_from,
        "file_serials": cp.file_serials,
        "sheet": str(cp.sheet),
        "sheet_sha256": cp.sheet_sha256,
        "reason": cp.reason,
    }


def _pct(x: float | None) -> str:
    return "—" if x is None else f"{100 * x:.1f}"


def _print_summary(rec: dict) -> None:
    src = rec["source"]
    print(f"  prior: {rec['bead']} from {src['deployment']} ({src['instrument']}), order "
          f"{src['order']}, {src['valid_T_degC'][0]:.2f}-{src['valid_T_degC'][1]:.2f} degC; "
          f"t_0 +/-{rec['t_0_sigma_mK']:.0f} mK"
          + (" (electronics differ)" if rec["electronics_differ"] else ""))
    for name, c in rec["candidates"].items():
        extra = c.get("error") or c.get("implausible") or ""
        print(f"      {name}: held-out {_f(_mk(c['held_out_K']))} mK, rms all pairs "
              f"{_f(_mk(c['rms_all_K']))} mK (kept {_f(_mk(c['in_sample_K']))} mK, dropped "
              f"{_pct(c['dropped_fraction'])}%)" + (f" [{extra}]" if extra else ""))
    print(f"      chosen: {rec['choice']} — {rec['why']}")
    for w in rec["warnings"]:
        print(f"  WARNING {w}", file=sys.stderr)
    qc = rec["qc"]
    line = f"QC vs prior: {qc['message']}"
    if qc["verdict"] == "flag":
        print(f"  WARNING {line}", file=sys.stderr)
    else:
        print(f"      {line}")


def _mk(x: float | None) -> float | None:
    return None if x is None else 1e3 * x


def report_lines(rec: dict) -> list[str]:
    """The ``fit.prior`` section of ``<ch>_report.md``."""
    if rec.get("status") != "used":
        return ["", "## Prior (`fit.prior`)", "",
                f"No prior used ({rec.get('reason') or rec.get('status')}); the full fit "
                f"is reported. Sheet: `{rec['sheet']}` (sha256 `{rec['sheet_sha256'][:16]}…`)."]
    src = rec["source"]
    lines = [
        "",
        "## Prior (`fit.prior`)",
        "",
        f"- **bead**: {rec['bead']} (from the "
        + ("config; the files say " + ", ".join(rec["file_serials"])
           if rec["bead_from"] == "config" else "files") + ")",
        f"- **source**: {src['deployment']} ({src['instrument']}, {src['first'][:10]} to "
        f"{src['last'][:10]}), order {src['order']}, fitted over "
        f"{src['valid_T_degC'][0]:.2f}-{src['valid_T_degC'][1]:.2f} °C"
        + (" — **a fallback: none of the bead's calibrations is unflagged**"
           if src["fallback"] else ""),
        f"- **sheet**: `{rec['sheet']}`, sha256 `{rec['sheet_sha256']}`",
        f"- **mode**: {rec['mode']}; tie band {100 * rec['margin']['tie_fraction']:g}%",
        f"- **t_0 uncertainty**: ±{rec['t_0_sigma_mK']:.0f} mK"
        + (" (the electronics differ, so the cross-instrument term is included)"
           if rec["electronics_differ"] else "")
        + f"; shape ±{rec['shape_mK']:g} mK",
        "",
        "Bridge constants (they define L, which the prior's shape is a function of):",
        "",
        "| constant | this channel | prior's source | same? |",
        "|---|---|---|---|",
    ]
    br = rec["bridge"]
    for k in _reg.BRIDGE_KEYS:
        a, b = br["this"].get(k), br["prior"].get(k)
        lines.append(f"| `{k}` | {_g(a)} | {_g(b)} | {'no' if k in br['differs'] else 'yes'} |")
    if br["varies_between_files"]:
        lines.append("")
        lines.append("**This channel's bridge constants vary between the files.**")
    lines += [
        "",
        f"Candidates, scored on the same {rec['n_folds']} temperature fold(s) (the "
        f"warm-half/cold-half split that picks the order). *rms all* is over every pair, the "
        f"same set for each candidate; *rms kept* over the pairs its own robust loop kept.",
        "",
        "| candidate | held-out [mK] | rms all [mK] | rms kept [mK] | dropped [%] | plausible |",
        "|---|---|---|---|---|---|",
    ]
    for name, c in rec["candidates"].items():
        mark = " **(chosen)**" if name == rec["choice"] else ""
        note = c.get("error") or c.get("implausible") or ""
        plaus = "—" if c["plausible"] is None else ("yes" if c["plausible"] else "**no**")
        lines.append(f"| {name}{mark} | {_f(_mk(c['held_out_K']))} | "
                     f"{_f(_mk(c['rms_all_K']))} | {_f(_mk(c['in_sample_K']))} | "
                     f"{_pct(c['dropped_fraction'])} | {plaus}"
                     + (f" ({note})" if note else "") + " |")
    qc = rec["qc"]
    lines += [
        "",
        f"**Choice**: {rec['choice']} — {rec['why']}.",
        "",
        f"**Against the prior** (same L, center {rec['delta_at_T_degC']:.2f} °C): full fit "
        f"{_f(rec['delta_full_mK'], '+.1f')} mK, offset-only fit "
        f"{_f(rec['delta_offset_mK'], '+.1f')} mK.",
        "",
        f"**QC: {'FLAGGED' if qc['verdict'] == 'flag' else 'ok'}** — {qc['message']}",
    ]
    for w in rec["warnings"]:
        lines += ["", f"**Warning:** {w}."]
    if rec["prior_notes"]:
        lines += ["", "What the sheet says about this prior:", ""]
        lines += [f"- {n}" for n in rec["prior_notes"]]
    return lines


def _g(x: float | None) -> str:
    return "—" if x is None else f"{x:g}"


def warnings_for_report(rec: dict) -> list[str]:
    """Report-header warnings: a flagged QC and a suspicious robust drop."""
    if rec.get("status") != "used":
        return []
    out = [f"**{w[0].upper()}{w[1:]}.**" for w in rec["warnings"]]
    if rec["qc"]["verdict"] == "flag":
        out.insert(0, f"**Against the prior ({rec['bead']}): {rec['qc']['message']}**")
    return out

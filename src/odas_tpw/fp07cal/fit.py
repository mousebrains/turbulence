# Aug-2026, Claude and Pat Welch, pat@mousebrains.com
"""The Steinhart-Hart regression, and the diagnostics that decide whether to believe it.

The fit itself is three lines of least squares.  Everything else here exists
because a least-squares fit *always* returns numbers, and on this data most of
the ways it can be wrong leave the RMS residual looking fine:

* a badly conditioned Vandermonde in ``L`` over a 3 degC glider range (A3),
* a slope attenuated by noise in the regressor (A2),
* an unremoved lag, which shows up as a dive-vs-climb split and not in the RMS
  at all (D5.1),
* a handful of outliers dragging a low-order polynomial.

So ``fit_calibration`` reports the condition number, an errors-in-variables
bracket on ``beta_1``, and the residual decomposition, and the caller is
expected to look at them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.polynomial import Polynomial

from odas_tpw.fp07cal.logr import coeffs_to_config, temperature
from odas_tpw.fp07cal.pairs import PairSet


@dataclass
class FitResult:
    """Coefficients in ``L``, plus everything needed to distrust them."""

    coeffs: np.ndarray  # [a0, a1, ...] such that 1/T_K = sum a_i L^i
    order: int
    n: int
    n_dropped: int
    center: float
    scale: float
    condition: float
    rms_K: float
    residual_K: np.ndarray
    T_range: tuple[float, float]
    L_range: tuple[float, float]
    dive_climb_split_K: float = float("nan")
    beta1_bracket: tuple[float, float] = (float("nan"), float("nan"))
    channel: str = "?"
    lag: float = float("nan")
    corr: float = float("nan")
    kept: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=bool))
    outlier: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=bool))
    """Pairs the robust loop REJECTED: in the design, but beyond ``robust_sigma``.

    Not simply ``~kept``: the joint geometry fit also leaves out pairs with no
    reference gradient, which it never judged.  The export gate is evaluated
    over every pair but these (issue #238).
    """
    fixed_terms: dict[str, float] | None = None
    """The ``beta`` terms when the shape was HELD FIXED and only ``t_0`` fitted.

    ``None`` for an ordinary fit.  Kept in config form so that
    :attr:`config_equivalent` hands them back verbatim: ``1/(1/beta)`` need not
    round-trip in float64, and the fixed shape must be the prior's, bit for bit.
    """

    @property
    def config_equivalent(self) -> dict[str, float]:
        if self.fixed_terms is not None:
            return {"t_0": 1.0 / float(self.coeffs[0]), **self.fixed_terms}
        return coeffs_to_config(self.coeffs)

    def apply(self, L: np.ndarray) -> np.ndarray:
        return temperature(L, self.coeffs)


def _design(u: np.ndarray, order: int) -> np.ndarray:
    return np.column_stack([u**i for i in range(order + 1)])


_BETA_KEYS = ("beta_1", "beta_2", "beta_3")


def fixed_shape_poly(fixed_terms: dict[str, float]) -> np.ndarray:
    """``{beta_1, ...}`` -> ``[0, 1/beta_1, ...]``: the shape with a zero offset.

    ``1.0 / beta`` exactly as :attr:`registry.Calibration.poly` and the reader
    (``config_to_coeffs``) compute it, so a fit with this shape held fixed
    reproduces bit for bit once its ``t_0`` is patched next to these betas.
    """
    keys = [k for k in _BETA_KEYS if k in fixed_terms]
    if not keys or keys != list(_BETA_KEYS[: len(keys)]) or set(fixed_terms) - set(keys):
        raise ValueError(
            f"a fixed shape needs consecutive beta_1..beta_n and nothing else, got "
            f"{sorted(fixed_terms)}"
        )
    return np.array([0.0] + [1.0 / float(fixed_terms[k]) for k in keys], dtype=np.float64)


def higher_terms(L: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    """``sum_{i>=1} a_i L^i``: everything but the offset."""
    out = np.zeros_like(L, dtype=np.float64)
    for i, a in enumerate(coeffs):
        if i == 0:
            continue
        out = out + a * L**i
    return out


def _uncenter(c: np.ndarray, center: float, scale: float) -> np.ndarray:
    """Coefficients in ``u = (L - center)/scale`` -> coefficients in ``L``.

    Exact polynomial composition, not a refit: the centered fit and the
    returned coefficients describe the same curve, so the emitted ``t_0`` /
    ``beta_i`` reproduce the fit bit for bit in the reader.
    """
    composed = Polynomial(c)(Polynomial([-center / scale, 1.0 / scale]))
    return np.asarray(composed.coef, dtype=np.float64)


def fit_calibration(
    pairs: PairSet,
    *,
    order: int = 1,
    robust: bool = True,
    robust_sigma: float = 4.0,
    robust_iters: int = 3,
    fixed_terms: dict[str, float] | None = None,
) -> FitResult:
    """Least-squares Steinhart-Hart fit over pooled pairs.

    Fitted in a centered/scaled variable and transformed back exactly (A3).
    Over a glider's narrow temperature range the raw Vandermonde in ``L`` is
    badly conditioned well before the higher-order terms stop being
    *statistically* meaningful, and a fit can be numerically junk while looking
    entirely healthy.

    *fixed_terms* (``{beta_1, ...}`` in config form, e.g. a bead's prior from
    the calibration sheet) holds the shape fixed and fits ``t_0`` alone, with
    the same robust iteration.  *order* is then the shape's and is ignored.  An
    offset needs almost no temperature range, so neither does this fit.
    """
    shape = None if fixed_terms is None else fixed_shape_poly(fixed_terms)
    if shape is not None:
        order = shape.size - 1
    if order < 1 or order > 3:
        raise ValueError(f"order must be 1..3, got {order}")
    # Fitted polynomial coefficients: all of them, or only the offset.
    n_poly = order + 1 if shape is None else 1
    n_all = len(pairs)
    if n_all < n_poly + 1:
        raise ValueError(
            f"{pairs.channel}: {n_all} pairs is too few for an order-{order} fit"
        )

    L = np.asarray(pairs.L, dtype=np.float64)
    y = 1.0 / (np.asarray(pairs.T_ref, dtype=np.float64) + 273.15)
    if shape is not None:
        y = y - higher_terms(L, shape)

    center = float(np.mean(L))
    scale = float(np.std(L))
    if shape is not None:
        # The offset column is ones whatever the spread: an L that hardly
        # moves is no obstacle to fitting an offset.
        center, scale = 0.0, 1.0
    elif not np.isfinite(scale) or scale == 0:
        raise ValueError(f"{pairs.channel}: L has zero spread; nothing to fit")
    u = (L - center) / scale
    d_order = order if shape is None else 0

    keep = np.ones(n_all, dtype=bool)
    coeffs_u = np.zeros(n_poly)
    cond = float("nan")
    for _ in range(robust_iters if robust else 1):
        X = _design(u[keep], d_order)
        cond = float(np.linalg.cond(X))
        coeffs_u, *_ = np.linalg.lstsq(X, y[keep], rcond=None)
        if not robust:
            break
        resid = y - _design(u, d_order) @ coeffs_u
        mad = float(np.median(np.abs(resid[keep] - np.median(resid[keep]))))
        sigma = 1.4826 * mad
        if sigma <= 0 or not np.isfinite(sigma):
            break
        new_keep = np.abs(resid) <= robust_sigma * sigma
        if new_keep.sum() < n_poly + 1 or np.array_equal(new_keep, keep):
            break
        keep = new_keep
    else:
        # Loop exhausted with `keep` freshly updated but never refitted: the
        # returned coefficients would describe the PREVIOUS keep set while n /
        # rms_K describe the final one (N3). One last solve reconciles them.
        X = _design(u[keep], d_order)
        cond = float(np.linalg.cond(X))
        coeffs_u, *_ = np.linalg.lstsq(X, y[keep], rcond=None)

    if shape is None:
        coeffs = _uncenter(coeffs_u, center, scale)
    else:
        coeffs = shape.copy()
        coeffs[0] = float(coeffs_u[0])

    T_pred = temperature(L, coeffs)
    resid_K = np.asarray(pairs.T_ref, dtype=np.float64) - T_pred
    rms = float(np.sqrt(np.nanmean(resid_K[keep] ** 2)))

    split = float("nan")
    d = np.asarray(pairs.direction)
    if np.any(d[keep] > 0) and np.any(d[keep] < 0):
        split = float(
            np.nanmean(resid_K[keep & (d > 0)]) - np.nanmean(resid_K[keep & (d < 0)])
        )

    bracket = (float("nan"), float("nan"))
    if order == 1 and shape is None:
        bracket = _beta1_bracket(L[keep], y[keep])

    return FitResult(
        coeffs=coeffs,
        order=order,
        n=int(keep.sum()),
        n_dropped=int(n_all - keep.sum()),
        center=center,
        scale=scale,
        condition=cond,
        rms_K=rms,
        residual_K=resid_K,
        # The range the fit covers is that of the pairs it KEPT: a reference
        # glitch the robust loop threw out must not widen the "do not
        # extrapolate" range, the patch-time validity verdict, the sheet's
        # valid_T_degC or the NTC check (issue #238 review).
        T_range=(float(np.min(pairs.T_ref[keep])), float(np.max(pairs.T_ref[keep]))),
        L_range=(float(np.min(L[keep])), float(np.max(L[keep]))),
        dive_climb_split_K=split,
        beta1_bracket=bracket,
        channel=pairs.channel,
        lag=pairs.lag,
        corr=pairs.corr,
        kept=keep,
        outlier=~keep,
        fixed_terms=None if fixed_terms is None else {
            k: float(fixed_terms[k]) for k in _BETA_KEYS if k in fixed_terms
        },
    )


def _beta1_bracket(L: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Errors-in-variables bracket on ``beta_1``.

    Noise in the regressor attenuates an ordinary ``y|x`` slope toward zero;
    the reverse regression ``x|y`` over-corrects in the opposite direction.  The
    true slope lies between them, so the pair is an honest interval where a
    single OLS number would be a confident understatement.

    A wide bracket means the bandwidth match (D2) is not doing its job --- it is
    the direct readout of how much A2 is still biting.
    """
    if L.size < 3 or np.std(L) == 0 or np.std(y) == 0:
        return (float("nan"), float("nan"))
    a1_fwd = float(np.polyfit(L, y, 1)[0])
    b1 = float(np.polyfit(y, L, 1)[0])
    if a1_fwd == 0 or b1 == 0:
        return (float("nan"), float("nan"))
    a1_rev = 1.0 / b1
    betas = sorted((1.0 / a1_fwd, 1.0 / a1_rev))
    return (float(betas[0]), float(betas[1]))


def select_order(
    pairs: PairSet,
    *,
    candidates: tuple[int, ...] = (1, 2, 3),
    robust: bool = True,
    use_geometry: bool = False,
    exclude: np.ndarray | None = None,
) -> tuple[int, dict]:
    """Pick the polynomial order by **out-of-sample** error, not by in-sample fit.

    A higher order always fits the training data better, so an in-sample
    criterion can only ever say "more".  The question that matters is whether
    the extra term *extrapolates* better --- which is exactly what is being asked
    of it when a deployment-wide coefficient set is applied to profiles that
    went outside the temperature range it was fitted on.

    So the split is by TEMPERATURE, not at random: fit the warm side, predict
    the cold side, and vice versa.  A random split would leave both folds
    spanning the full range and would reward interpolation, which is not the
    failure mode of interest.

    The split point is the middle of the temperature RANGE, not the median.
    On a thermocline profile most samples sit in a fraction of a degree of
    deep water, so a median split leaves one "half" spanning well under 1 K
    --- and the held-out score is then dominated by extrapolating that sliver
    across the other several K, which punishes every order into the ground
    (a genuine cubic truth scored order 1 best under a median split).

    Scoring uses the SAME estimator that will be shipped: when the production
    fit is the joint geometry fit, so is each fold's (``use_geometry``),
    otherwise the order chosen can be optimal for a model nobody runs.

    *exclude* (:func:`outlier_screen`) leaves the deployment's glitches out of
    the folds, so they neither set the split nor dominate a held-out error.

    Measured on osu685 (24 degC of range, ~48k pairs):

    ======  ==========  ===========================
    order   in-sample   held-out (warm->cold)
    ======  ==========  ===========================
    1       22.3 mK     70.0 mK
    2       10.7 mK     31.0 mK
    3       10.7 mK     126.3 mK
    ======  ==========  ===========================

    ``beta_2`` is decisive --- it halves the in-sample residual and improves
    extrapolation two- to five-fold.  ``beta_3`` gains 0.013 mK in sample while
    making extrapolation four times worse, which is precisely the overfit this
    test exists to catch (its t-statistic of 10 would have called it
    "significant").
    """
    folds = temperature_folds(pairs, exclude)
    if folds is None:
        return min(candidates), {}
    scores: dict[int, dict] = {}

    def _fit(sub: PairSet, order: int):
        """The estimator actually shipped for this configuration."""
        return shipped_fit(sub, order=order, robust=robust, use_geometry=use_geometry)

    for order in candidates:
        try:
            in_sample = _fit(pairs, order).rms_K
        except Exception:
            continue
        held = held_out_error(
            pairs, lambda sub, o=order: _fit(sub, o), folds=folds, min_train=order + 5
        )
        if held is not None:
            scores[order] = {"held_out_K": held,
                             "in_sample_K": float(in_sample)}
    if not scores:
        return min(candidates), {}
    best = min(scores, key=lambda o: scores[o]["held_out_K"])
    return best, scores


def shipped_fit(
    pairs: PairSet,
    *,
    order: int = 1,
    robust: bool = True,
    use_geometry: bool = False,
    fixed_terms: dict[str, float] | None = None,
) -> FitResult:
    """The calibration part of the estimator ``fp07-cal fit`` ships.

    :func:`~odas_tpw.fp07cal.geometry.joint_fit` when the sensor-geometry term
    is on, else :func:`fit_calibration`; with *fixed_terms*, the same with the
    shape held fixed.  Held-out scores must come from this, or the choice they
    make can be optimal for a model nobody runs.
    """
    if use_geometry:
        from odas_tpw.fp07cal.geometry import joint_fit

        f, _geo = joint_fit(pairs, order=order, robust=robust, fixed_terms=fixed_terms)
        return f
    return fit_calibration(pairs, order=order, robust=robust, fixed_terms=fixed_terms)


def temperature_folds(pairs: PairSet, exclude: np.ndarray | None = None) -> tuple | None:
    """``((train, test), (train, test))``: the warm and cold halves of the range.

    Split at the middle of the temperature RANGE, not the median (see
    :func:`select_order`).  ``None`` when no reference temperature is finite.

    *exclude* (from :func:`outlier_screen`) leaves pairs out of both the split
    and every fold: a reference glitch at 2 degC would otherwise set the
    middle of a 13-21 degC deployment and, in a test half, dominate its error.
    """
    T = np.asarray(pairs.T_ref, dtype=np.float64)
    ok = np.isfinite(T)
    if exclude is not None and np.size(exclude) == T.size:
        ok = ok & ~np.asarray(exclude, dtype=bool)
    fin = T[ok]
    if fin.size == 0:
        return None
    mid = 0.5 * (float(np.min(fin)) + float(np.max(fin)))
    if exclude is None:
        return ((mid <= T, mid > T), (mid > T, mid <= T))
    return ((ok & (mid <= T), ok & (mid > T)), (ok & (mid > T), ok & (mid <= T)))


def outlier_screen(
    pairs: PairSet,
    *,
    orders: tuple[int, ...] = (1, 2, 3),
    robust: bool = True,
    use_geometry: bool = False,
) -> np.ndarray:
    """Pairs that the shipped fit at EVERY order in *orders* rejects as outliers.

    A glitch in the reference or the thermistor, as opposed to a shape one
    order cannot follow: order 1 over a glider's 24 degC rejects the ends of
    the range, order 3 does not, so only a pair all of them reject is left out
    of the held-out folds (:func:`temperature_folds`).  All False without the
    robust loop, or when no order can be fitted.
    """
    out: np.ndarray | None = None
    if robust:
        for o in orders:
            try:
                f = shipped_fit(pairs, order=o, robust=True, use_geometry=use_geometry)
            except Exception:
                continue
            m = (f.outlier if f.outlier.size == len(pairs)
                 else np.zeros(len(pairs), dtype=bool))
            out = m.copy() if out is None else out & m
    return out if out is not None else np.zeros(len(pairs), dtype=bool)


def held_out_error(
    pairs: PairSet,
    fitter,
    *,
    folds: tuple | None = None,
    min_train: int,
) -> float | None:
    """Mean over the temperature folds of the rms error predicting the held-out half [K].

    *fitter* maps a training :class:`PairSet` to a :class:`FitResult`.  A fold
    whose training half has fewer than *min_train* pairs, or whose test half
    has fewer than 10, is skipped, as is a fold whose fit fails; ``None`` when
    no fold is left.  This is the one held-out estimator: :func:`select_order`
    scores the polynomial orders with it, and the prior's fixed shape
    (``fit.prior``) is scored with it as one more candidate.
    """
    folds = temperature_folds(pairs) if folds is None else folds
    if folds is None:
        return None
    T = np.asarray(pairs.T_ref, dtype=np.float64)
    errs = []
    for train, test in folds:
        if int(train.sum()) < min_train or int(test.sum()) < 10:
            continue
        try:
            sub = _subset(pairs, train)
            f = fitter(sub)
            # Held-out prediction from the CALIBRATION part alone: the
            # geometry terms are per-sample corrections carried separately
            # and identically across candidates, so they cancel in the
            # comparison; what is being selected is the polynomial.
            pred = f.apply(np.asarray(pairs.L, dtype=np.float64)[test])
            errs.append(float(np.sqrt(np.nanmean((T[test] - pred) ** 2))))
        except Exception:
            continue
    return float(np.mean(errs)) if errs else None


def held_out_errors(
    pairs: PairSet,
    fitters: dict,
    *,
    min_train: int,
    folds: tuple | None = None,
) -> tuple[dict, int]:
    """:func:`held_out_error` for several candidates on the SAME folds.

    *fitters* maps a name to a fitter as in :func:`held_out_error`.  A fold is
    used only if every candidate can be fitted on it and predicts a finite
    error; otherwise it is dropped for all of them, so no candidate is scored
    on a fold another one was spared.  Returns ``({name: rms or None}, folds
    used)``.
    """
    folds = temperature_folds(pairs) if folds is None else folds
    names = list(fitters)
    if folds is None:
        return {n: None for n in names}, 0
    T = np.asarray(pairs.T_ref, dtype=np.float64)
    L = np.asarray(pairs.L, dtype=np.float64)
    errs: dict[str, list[float]] = {n: [] for n in names}
    used = 0
    for train, test in folds:
        if int(train.sum()) < min_train or int(test.sum()) < 10:
            continue
        sub = _subset(pairs, train)
        fold: dict[str, float] = {}
        for n in names:
            try:
                pred = fitters[n](sub).apply(L[test])
                e = float(np.sqrt(np.nanmean((T[test] - pred) ** 2)))
            except Exception:
                break
            if not np.isfinite(e):
                break
            fold[n] = e
        if len(fold) != len(names):
            continue
        used += 1
        for n, e in fold.items():
            errs[n].append(e)
    return {n: (float(np.mean(v)) if v else None) for n, v in errs.items()}, used


def _subset(pairs: PairSet, mask: np.ndarray) -> PairSet:
    from dataclasses import replace as _replace

    return _replace(
        pairs,
        time=pairs.time[mask], T_ref=pairs.T_ref[mask], L=pairs.L[mask],
        pressure=pairs.pressure[mask], w=pairs.w[mask],
        direction=pairs.direction[mask], profile_uid=pairs.profile_uid[mask],
        file_label=pairs.file_label[mask],
    )


def residual_breakdown(pairs: PairSet, fit: FitResult, n_bins: int = 12) -> dict:
    """Residual structure vs pressure and vs time --- D5.2 and D5.3.

    A calibration with a small RMS and a clear trend in either of these is not
    a good calibration; it is a bias waiting to be absorbed into ``t_0``.
    """
    keep = fit.kept
    out: dict = {}
    for name, axis in (("pressure", pairs.pressure), ("time", pairs.time)):
        a = np.asarray(axis, dtype=np.float64)[keep]
        r = fit.residual_K[keep]
        ok = np.isfinite(a) & np.isfinite(r)
        if int(ok.sum()) < n_bins:
            out[name] = None
            continue
        a, r = a[ok], r[ok]
        edges = np.linspace(a.min(), a.max(), n_bins + 1)
        idx = np.clip(np.digitize(a, edges) - 1, 0, n_bins - 1)
        centers, means, counts = [], [], []
        for b in range(n_bins):
            m = idx == b
            if not np.any(m):
                continue
            centers.append(0.5 * (edges[b] + edges[b + 1]))
            means.append(float(np.mean(r[m])))
            counts.append(int(m.sum()))
        slope = float(np.polyfit(a, r, 1)[0]) if a.size > 2 else float("nan")
        out[name] = {
            "centers": np.array(centers),
            "means": np.array(means),
            "counts": np.array(counts),
            "slope": slope,
        }
    return out

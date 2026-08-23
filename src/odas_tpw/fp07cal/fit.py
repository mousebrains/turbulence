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

    @property
    def config_equivalent(self) -> dict[str, float]:
        return coeffs_to_config(self.coeffs)

    def apply(self, L: np.ndarray) -> np.ndarray:
        return temperature(L, self.coeffs)


def _design(u: np.ndarray, order: int) -> np.ndarray:
    return np.column_stack([u**i for i in range(order + 1)])


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
) -> FitResult:
    """Least-squares Steinhart-Hart fit over pooled pairs.

    Fitted in a centered/scaled variable and transformed back exactly (A3).
    Over a glider's narrow temperature range the raw Vandermonde in ``L`` is
    badly conditioned well before the higher-order terms stop being
    *statistically* meaningful, and a fit can be numerically junk while looking
    entirely healthy.
    """
    if order < 1 or order > 3:
        raise ValueError(f"order must be 1..3, got {order}")
    n_all = len(pairs)
    if n_all < order + 2:
        raise ValueError(
            f"{pairs.channel}: {n_all} pairs is too few for an order-{order} fit"
        )

    L = np.asarray(pairs.L, dtype=np.float64)
    y = 1.0 / (np.asarray(pairs.T_ref, dtype=np.float64) + 273.15)

    center = float(np.mean(L))
    scale = float(np.std(L))
    if not np.isfinite(scale) or scale == 0:
        raise ValueError(f"{pairs.channel}: L has zero spread; nothing to fit")
    u = (L - center) / scale

    keep = np.ones(n_all, dtype=bool)
    coeffs_u = np.zeros(order + 1)
    cond = float("nan")
    for _ in range(robust_iters if robust else 1):
        X = _design(u[keep], order)
        cond = float(np.linalg.cond(X))
        coeffs_u, *_ = np.linalg.lstsq(X, y[keep], rcond=None)
        if not robust:
            break
        resid = y - _design(u, order) @ coeffs_u
        mad = float(np.median(np.abs(resid[keep] - np.median(resid[keep]))))
        sigma = 1.4826 * mad
        if sigma <= 0 or not np.isfinite(sigma):
            break
        new_keep = np.abs(resid) <= robust_sigma * sigma
        if new_keep.sum() < order + 2 or np.array_equal(new_keep, keep):
            break
        keep = new_keep

    coeffs = _uncenter(coeffs_u, center, scale)

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
    if order == 1:
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
        T_range=pairs.T_range,
        L_range=(float(np.min(L)), float(np.max(L))),
        dive_climb_split_K=split,
        beta1_bracket=bracket,
        channel=pairs.channel,
        lag=pairs.lag,
        corr=pairs.corr,
        kept=keep,
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

# Aug-2026, Claude and Pat Welch, pat@mousebrains.com
"""Separating sensor geometry from calibration error in the depth residual.

The problem this solves
-----------------------
On osu685 the FP07 sits ~15 cm above the vehicle centreline and the CTD ~2 cm
below it, so the two are ~17 cm apart PERPENDICULAR to the axis.  That is not a
lag: at any instant the two sensors are at different depths, sampling different
water.  It contributes

    T_ref - T_probe  ~=  dz * dT/dz          [dz ~ 0.17 * cos(theta) m]

which in a tropical thermocline at 0.2 K/m is ~0.03 K --- far above the glider CTD's
accuracy, and **strongly depth-dependent**, because dT/dz varies by a factor of
tens between the thermocline and the deep water.  Read naively, that is exactly
what "the calibration is depth-dependent" looks like.  It is geometry.

The degeneracy, and how it breaks
---------------------------------
An unremoved timing error ``tau`` produces a residual too, and it is
proportional to the SAME thing:

    T_ref - T_probe  ~=  tau * dT/dt  =  tau * w * dT/dz

So both terms scale with ``dT/dz`` and cannot be told apart from the depth
profile alone.  Divide through by ``dT/dz``:

    residual / (dT/dz)  =  dz  +  tau * w

and they separate cleanly: the intercept against vertical speed is the
geometric offset in metres, the slope is the residual lag in seconds.  Vertical
speed varies within a climb (slow out of apogee, faster mid-water), which is
what supplies the lever arm.

This is much better conditioned than trying to recover the LONGITUDINAL
separation from timing: there ``1/U`` spans only a factor of two and is 70-90%
collinear with elapsed time, and the fit on osu685 returned separations of
-18.6 m and +9.3 m depending on which nuisance term was included.  Here
``dT/dz`` spans a factor of tens, and it is measured rather than inferred.

Both quantities are platform-specific --- the longitudinal separation changes
with an extended energy bay, and the perpendicular offset with the mounting ---
so they belong in the per-deployment record, never in a shared default.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from odas_tpw.fp07cal.fit import FitResult
from odas_tpw.fp07cal.pairs import PairSet


@dataclass
class GeometryFit:
    """Decomposition of the ``dT/dz``-proportional part of the residual."""

    dz_m: float = float("nan")
    """Effective perpendicular sensor offset [m].  Positive = reference deeper."""

    dz_se_m: float = float("nan")
    tau_s: float = float("nan")
    """Residual timing error [s] left over after the applied lag.

    Sign convention: a positive ``tau_s`` means the applied lag was too LARGE
    by that much --- the corrected total lag is ``lag - tau_s``, not
    ``lag + tau_s``.
    """

    tau_se_s: float = float("nan")
    n: int = 0
    rms_K: float = float("nan")
    dTdz_range: tuple[float, float] = (float("nan"), float("nan"))
    w_range: tuple[float, float] = (float("nan"), float("nan"))
    collinearity: float = float("nan")
    """|corr(1, w)| proxy — how well the two terms are actually separated."""

    explained_K: float = float("nan")
    """RMS of the fitted geometry+lag term: how much of the depth structure it is."""

    dz_se_cluster_m: float = float("nan")
    """``dz`` standard error with PROFILES as the independent unit.

    ``dz_se_m`` above is the conditional IID-pair error: it divides by
    ``n_pairs - n_params`` and treats every sample within a profile as new
    information.  It is not an instrument uncertainty.  Repeating each
    observation 100 times while keeping the same six profile ids left dz at
    0.173288846 m but shrank ``dz_se_m`` 0.00579 -> 0.000577 m and
    ``tau_se_s`` 0.01196 -> 0.00119 s, on data carrying no new information at
    all (issue #180 F09).  These cluster-robust (sandwich) errors group the
    residual by ``profile_uid``, which is the unit that repeats.

    QUOTE THESE, not the IID pair errors -- but read ``n_clusters`` first: a
    sandwich estimator with a handful of clusters is itself noisy, and
    :attr:`cluster_warning` says when that applies.
    """

    tau_se_cluster_s: float = float("nan")
    """``tau`` standard error with profiles as the independent unit."""

    n_clusters: int = 0
    """Profiles contributing to the cluster-robust covariance."""

    cluster_warning: str = ""
    """Non-empty when ``n_clusters`` is too small for the sandwich to be quoted."""

    rank_deficient: bool = False
    """True when the FULL scaled design (calibration + geometry) is rank-deficient.

    The geometry ``collinearity`` below only asks whether dz and tau are
    separated FROM EACH OTHER; it says nothing about their degeneracy with the
    calibration polynomial.  A gradient of ``-1e-6*(T+273.15)**2`` makes the
    geometry regressor constant and therefore aliased with the intercept: the
    fit returned condition 3.8e30 with geometry correlation 4.7e-17 -- which
    passes a "separately resolved" test based on that correlation alone -- and
    a calibration error of 6.5e33 K with dz = 3448 +/- 0.68 m (issue #180 F20).
    """

    smallest_singular_value: float = float("nan")
    """Smallest singular value of the full scaled design (rank diagnostic)."""

    def separately_resolved(self, max_collinearity: float = 0.8) -> bool:
        """dz and tau resolved from each other AND from the calibration.

        The CLI predicate used to test only ``collinearity``, which a design
        aliased with the intercept passes trivially.
        """
        return (
            bool(np.isfinite(self.collinearity))
            and self.collinearity <= max_collinearity
            and not self.rank_deficient
        )

    def summary(self) -> str:
        se_dz = self.dz_se_cluster_m if np.isfinite(self.dz_se_cluster_m) else self.dz_se_m
        se_tau = self.tau_se_cluster_s if np.isfinite(self.tau_se_cluster_s) else self.tau_se_s
        note = f" [{self.cluster_warning}]" if self.cluster_warning else ""
        return (
            f"dz = {self.dz_m*100:+.1f} +/- {se_dz*100:.1f} cm, "
            f"residual lag = {self.tau_s:+.3f} +/- {se_tau:.3f} s "
            f"(n={self.n}, {self.n_clusters} profiles, "
            f"explains {self.explained_K:.4f} K rms){note}"
        )


# Below this many profiles a cluster-robust sandwich is itself too noisy to
# quote as an uncertainty: with a handful of clusters the meat matrix is a sum
# of a handful of outer products and the resulting SE has few degrees of freedom
# of its own. Reported anyway, with a warning attached (#180 F09 red team).
MIN_CLUSTERS_FOR_SANDWICH = 10


def _cluster_cov(X: np.ndarray, resid: np.ndarray, groups: np.ndarray) -> np.ndarray | None:
    """Cluster-robust (sandwich) covariance of the least-squares coefficients.

    ``(X'X)^-1 (sum_g X_g' u_g u_g' X_g) (X'X)^-1`` with a finite-sample
    correction, grouping by ``groups``.  Profiles are the unit that repeats;
    samples inside one are not independent replicates of the geometry.
    """
    n, k = X.shape
    uniq, inv = np.unique(groups, return_inverse=True)
    n_g = uniq.size
    if n_g < 2 or n <= k:
        return None
    xtx_inv = np.linalg.pinv(X.T @ X)
    meat = np.zeros((k, k))
    for gi in range(n_g):
        m = inv == gi
        sg = X[m].T @ resid[m]
        meat += np.outer(sg, sg)
    dfc = (n_g / (n_g - 1.0)) * ((n - 1.0) / (n - k))
    return np.asarray(dfc * (xtx_inv @ meat @ xtx_inv))


def local_dTdz(
    pairs: PairSet, *, bin_dbar: float = 10.0, min_count: int = 5
) -> np.ndarray:
    """``dT/dz`` [K/m] at each pair, from the REFERENCE profile.

    Taken from the reference rather than the probe on purpose: the probe's
    temperature is the thing under test, and using it would feed the
    calibration error back into the regressor.

    Computed per profile, since the water column changes over a 70-day
    deployment and a pooled T(z) curve would smear the thermocline.
    """
    out = np.full(len(pairs), np.nan)
    if not len(pairs):
        return out
    P = np.asarray(pairs.pressure, dtype=np.float64)
    T = np.asarray(pairs.T_ref, dtype=np.float64)
    uid = np.asarray(pairs.profile_uid, dtype=object)
    for u in np.unique(uid):
        m = np.flatnonzero(uid == u)
        p, t = P[m], T[m]
        good = np.isfinite(p) & np.isfinite(t)
        if good.sum() < 3 * min_count:
            continue
        lo, hi = float(np.min(p[good])), float(np.max(p[good]))
        if hi - lo < 3 * bin_dbar:
            continue
        edges = np.arange(lo, hi + bin_dbar, bin_dbar)
        idx = np.clip(np.digitize(p, edges) - 1, 0, edges.size - 2)
        nb = edges.size - 1
        sums = np.zeros(nb)
        cnts = np.zeros(nb)
        np.add.at(sums, idx[good], t[good])
        np.add.at(cnts, idx[good], 1.0)
        ok = cnts >= min_count
        if ok.sum() < 3:
            continue
        centers = 0.5 * (edges[:-1] + edges[1:])
        Tb = np.where(cnts > 0, sums / np.maximum(cnts, 1.0), np.nan)
        # dbar -> m is 1:1 to better than 2%, which is far inside the error bar.
        g = np.gradient(Tb[ok], centers[ok])
        out[m] = np.interp(p, centers[ok], g, left=np.nan, right=np.nan)
    return out


def geometry_fit(
    pairs: PairSet,
    fit: FitResult,
    *,
    dTdz: np.ndarray | None = None,
    min_abs_dTdz: float = 0.005,
    max_abs_resid_K: float = 0.5,
) -> GeometryFit:
    """Split the ``dT/dz``-proportional residual into offset and residual lag.

    Fits ``residual = dz * dTdz + tau * (w * dTdz)`` by least squares.  Pairs
    where ``dT/dz`` is too small to carry information are dropped --- dividing
    by a near-zero gradient is what would make this explode.
    """
    res = GeometryFit()
    if not len(pairs):
        return res
    g = local_dTdz(pairs) if dTdz is None else np.asarray(dTdz, dtype=np.float64)
    keep = fit.kept if fit.kept.size == len(pairs) else np.ones(len(pairs), dtype=bool)
    r = np.asarray(fit.residual_K, dtype=np.float64)
    w = np.asarray(pairs.w, dtype=np.float64)  # dbar/s ~ m/s

    ok = (
        keep
        & np.isfinite(g)
        & np.isfinite(r)
        & np.isfinite(w)
        & (np.abs(g) >= min_abs_dTdz)
        & (np.abs(r) <= max_abs_resid_K)
    )
    n = int(ok.sum())
    if n < 50:
        res.n = n
        return res

    g, r, w = g[ok], r[ok], w[ok]
    A = np.column_stack([g, w * g])
    coef, *_ = np.linalg.lstsq(A, r, rcond=None)
    pred = A @ coef
    resid = r - pred
    dof = max(1, n - 2)
    s2 = float(resid @ resid) / dof
    cov = s2 * np.linalg.pinv(A.T @ A)
    se = np.sqrt(np.diag(cov))

    res.dz_m = float(coef[0])
    res.tau_s = float(coef[1])
    res.dz_se_m = float(se[0])
    res.tau_se_s = float(se[1])
    res.n = n
    res.rms_K = float(np.sqrt(np.mean(resid**2)))
    res.dTdz_range = (float(np.min(g)), float(np.max(g)))
    res.w_range = (float(np.min(w)), float(np.max(w)))
    res.explained_K = float(np.sqrt(np.mean(pred**2)))
    c = np.corrcoef(A[:, 0], A[:, 1])[0, 1]
    res.collinearity = float(abs(c)) if np.isfinite(c) else float("nan")
    return res


def joint_fit(
    pairs: PairSet,
    *,
    order: int = 1,
    dTdz: np.ndarray | None = None,
    min_abs_dTdz: float = 0.005,
    robust: bool = True,
    robust_sigma: float = 4.0,
    robust_iters: int = 3,
) -> tuple[FitResult, GeometryFit]:
    """Fit the calibration AND the geometry together.

    Estimating the geometry from post-fit residuals systematically
    **underestimates** it: on a monotone profile ``dT/dz`` is correlated with
    ``T`` itself, so an ordinary calibration fit quietly absorbs most of the
    ``dz * dT/dz`` signal into ``t_0`` and leaves almost nothing in the
    residual.  Measured on synthetic data with a 25 cm offset injected, the
    two-step route recovered 0.4 cm --- it had been swallowed whole.

    Fitting jointly avoids that.  Linearising the geometry term about the
    reference temperature (``d(1/T_K)/dT = -1/T_K^2``, and the correction is
    ~0.03 K on ~290 K, so the linearisation error is negligible) gives a model
    that is linear in every parameter at once:

        1/T_K = sum_i a_i L^i  -  dz * (g / T_K^2)  -  tau * (w * g / T_K^2)

    (the sign is negative because ``d(1/T_K)/dT = -1/T_K^2``: a reference that
    reads WARM by ``dz*g`` reads LOW in reciprocal kelvin)

    so one least-squares solve returns the Steinhart-Hart coefficients, the
    perpendicular sensor offset, and any residual timing error, with none of
    them able to steal another's signal.
    """
    from odas_tpw.fp07cal.fit import _design, _uncenter
    from odas_tpw.fp07cal.logr import temperature as _temperature

    g_all = local_dTdz(pairs) if dTdz is None else np.asarray(dTdz, dtype=np.float64)
    L_all = np.asarray(pairs.L, dtype=np.float64)
    T_all = np.asarray(pairs.T_ref, dtype=np.float64)
    w_all = np.asarray(pairs.w, dtype=np.float64)

    usable = (
        np.isfinite(L_all) & np.isfinite(T_all) & np.isfinite(g_all) & np.isfinite(w_all)
    )
    if int(usable.sum()) < order + 4:
        raise ValueError(
            f"{pairs.channel}: {int(usable.sum())} usable pairs is too few for a "
            f"joint order-{order} + geometry fit"
        )
    # Pairs in near-isothermal water carry no geometry information; keep them
    # for the calibration but give them a zero geometry regressor rather than
    # dividing by a vanishing gradient.
    g_eff = np.where(np.abs(g_all) >= min_abs_dTdz, g_all, 0.0)

    y = 1.0 / (T_all + 273.15)
    T_K2 = (T_all + 273.15) ** 2
    center = float(np.mean(L_all[usable]))
    scale = float(np.std(L_all[usable]))
    if not np.isfinite(scale) or scale == 0:
        raise ValueError(f"{pairs.channel}: L has zero spread; nothing to fit")
    u = (L_all - center) / scale

    # Negated so the fitted coefficients are dz and tau directly (see the
    # docstring), and column-normalised because g/T_K^2 is O(1e-9) against an
    # O(1) Vandermonde -- unscaled, the joint design matrix has a condition
    # number of ~1e7 and the geometry terms are numerically swamped.
    extra_raw = np.column_stack([-g_eff / T_K2, -w_all * g_eff / T_K2])
    extra_scale = np.array([
        np.std(extra_raw[usable, k]) or 1.0 for k in range(extra_raw.shape[1])
    ])
    extra = extra_raw / extra_scale
    keep = usable.copy()
    coef = np.zeros(order + 3)
    cond = float("nan")
    for _ in range(robust_iters if robust else 1):
        X = np.column_stack([_design(u[keep], order), extra[keep]])
        cond = float(np.linalg.cond(X))
        coef, *_ = np.linalg.lstsq(X, y[keep], rcond=None)
        if not robust:
            break
        full = np.column_stack([_design(u, order), extra])
        resid = y - full @ coef
        mad = float(np.median(np.abs(resid[keep] - np.median(resid[keep]))))
        sigma = 1.4826 * mad
        if sigma <= 0 or not np.isfinite(sigma):
            break
        new_keep = usable & (np.abs(resid) <= robust_sigma * sigma)
        if new_keep.sum() < order + 4 or np.array_equal(new_keep, keep):
            break
        keep = new_keep
    else:
        # Loop exhausted with `keep` updated but not refitted (N3): one more
        # solve so the returned parameters describe the final keep set.
        X = np.column_stack([_design(u[keep], order), extra[keep]])
        cond = float(np.linalg.cond(X))
        coef, *_ = np.linalg.lstsq(X, y[keep], rcond=None)

    coeffs = _uncenter(coef[: order + 1], center, scale)
    dz = float(coef[order + 1] / extra_scale[0])
    tau = float(coef[order + 2] / extra_scale[1])

    T_pred = _temperature(L_all, coeffs) + dz * g_eff + tau * w_all * g_eff
    resid_K = T_all - T_pred
    rms = float(np.sqrt(np.nanmean(resid_K[keep] ** 2)))

    X = np.column_stack([_design(u[keep], order), extra[keep]])
    dof = max(1, int(keep.sum()) - X.shape[1])
    r = y[keep] - X @ coef
    cov = float(r @ r) / dof * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(cov))

    # Rank-revealing SVD on the FULL scaled design. `cond` above is recorded but
    # never gated, and the geometry `collinearity` only asks whether dz and tau
    # are separated from EACH OTHER -- a geometry column aliased with the
    # calibration intercept passes it at 4.7e-17 while the fit returns a 6.5e33 K
    # calibration error (issue #180 F20).
    sv = np.linalg.svd(X, compute_uv=False)
    smallest_sv = float(sv[-1]) if sv.size else float("nan")
    rank = int(np.linalg.matrix_rank(X))
    rank_deficient = rank < X.shape[1]

    # Cluster-robust covariance with PROFILES as the independent unit (#180 F09).
    uid_keep = np.asarray(pairs.profile_uid, dtype=object)[keep]
    cov_cl = _cluster_cov(X, r, uid_keep)
    n_clusters = int(np.unique(uid_keep).size)
    if cov_cl is not None:
        se_cl = np.sqrt(np.clip(np.diag(cov_cl), 0.0, None))
    else:
        se_cl = np.full(X.shape[1], np.nan)
    cluster_warning = ""
    if n_clusters < MIN_CLUSTERS_FOR_SANDWICH:
        cluster_warning = (
            f"only {n_clusters} profile(s): the cluster-robust SE is itself "
            f"noisy below {MIN_CLUSTERS_FOR_SANDWICH} clusters"
        )

    d = np.asarray(pairs.direction)
    split = float("nan")
    if np.any(d[keep] > 0) and np.any(d[keep] < 0):
        split = float(
            np.nanmean(resid_K[keep & (d > 0)]) - np.nanmean(resid_K[keep & (d < 0)])
        )

    fit = FitResult(
        coeffs=coeffs, order=order, n=int(keep.sum()),
        n_dropped=int(usable.sum() - keep.sum()), center=center, scale=scale,
        condition=cond, rms_K=rms, residual_K=resid_K,
        T_range=(float(np.min(T_all[usable])), float(np.max(T_all[usable]))),
        L_range=(float(np.min(L_all[usable])), float(np.max(L_all[usable]))),
        dive_climb_split_K=split, channel=pairs.channel, lag=pairs.lag,
        corr=pairs.corr, kept=keep,
    )

    gk = g_eff[keep]
    nz = np.abs(gk) > 0
    geo = GeometryFit(
        dz_m=dz, tau_s=tau,
        dz_se_m=float(se[order + 1] / extra_scale[0]),
        tau_se_s=float(se[order + 2] / extra_scale[1]),
        n=int(nz.sum()), rms_K=rms,
        dTdz_range=(float(np.min(gk[nz])), float(np.max(gk[nz]))) if nz.any()
        else (float("nan"), float("nan")),
        w_range=(float(np.min(w_all[keep])), float(np.max(w_all[keep]))),
        explained_K=float(np.sqrt(np.mean((dz * gk + tau * w_all[keep] * gk) ** 2))),
        dz_se_cluster_m=float(se_cl[order + 1] / extra_scale[0]),
        tau_se_cluster_s=float(se_cl[order + 2] / extra_scale[1]),
        n_clusters=n_clusters,
        cluster_warning=cluster_warning,
        rank_deficient=rank_deficient,
        smallest_singular_value=smallest_sv,
    )
    c = np.corrcoef(extra[keep][:, 0], extra[keep][:, 1])[0, 1]
    geo.collinearity = float(abs(c)) if np.isfinite(c) else float("nan")
    return fit, geo


def apply_geometry_correction(
    pairs: PairSet, geo: GeometryFit, *, dTdz: np.ndarray | None = None
) -> np.ndarray:
    """Reference temperature with the geometric offset removed.

    Use when the offset is established and you want the calibration fitted
    against what the FP07's own depth actually saw, rather than absorbing a
    fixed mounting difference into ``t_0``.
    """
    g = local_dTdz(pairs) if dTdz is None else np.asarray(dTdz, dtype=np.float64)
    corr = np.where(np.isfinite(g), geo.dz_m * g, 0.0)
    return np.asarray(pairs.T_ref, dtype=np.float64) - corr

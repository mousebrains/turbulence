# Aug-2026, Claude and Pat Welch, pat@mousebrains.com
"""Is the calibration stable over the deployment?

The estimator (plan D8)
-----------------------
``beta_1`` and ``t_0`` differ in both their expected stability and their
estimation requirements, and the two facts point in opposite directions:

* ``beta_1`` is the bead's material B-value.  Expected stable, and it is a
  *slope*, so it needs the widest temperature range available --- pool the
  whole deployment.
* ``t_0`` is probe ``R_0`` folded together with the bridge offset.  Expected to
  drift (aging, electronics, fouling), and it is an *offset*, so it needs
  almost no temperature range --- one block of yos will do.

So: **fit ``beta_1`` globally, then fit ``t_0`` per temporal block with the
higher-order terms held fixed.**  Refitting everything per block instead would
let each block's narrow range throw ``beta_1`` around and drag ``t_0`` with it
through their strong covariance --- the apparent drift would be mostly that
covariance.

Independence
------------
Samples within a profile are heavily autocorrelated, so treating pairs as
independent would understate the uncertainty by roughly the square root of the
samples-per-profile count and turn any wobble into a significant trend.  The
**profile** is the independent unit throughout.

Attribution
-----------
With one reference this measures the thermistor *relative to* the glider CTD, and
it also absorbs unremoved lag and the mounting separation between the two
sensors (plan section 8.3).  Charging the result to the FP07 is defensible
because SBE temperature stability is far better than plausible FP07 drift, but
it is an argument, not a measurement --- which is what :func:`t1_t2_series` is
for.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from odas_tpw.fp07cal.fit import FitResult
from odas_tpw.fp07cal.logr import log_r, temperature
from odas_tpw.fp07cal.pairs import PairSet
from odas_tpw.fp07cal.series import ProbeSeries

if TYPE_CHECKING:  # geometry imports stability's PairSet siblings; keep it lazy
    from odas_tpw.fp07cal.geometry import GeometryFit

SECONDS_PER_DAY = 86400.0


@dataclass
class Block:
    """One temporal block's offset estimate."""

    t_mid: float
    t_start: float
    t_end: float
    n_pairs: int
    n_profiles: int
    a0: float
    a0_se: float
    t_0: float
    dT_K: float
    dT_se_K: float


@dataclass
class StabilityResult:
    blocks: list[Block] = field(default_factory=list)
    drift_K_per_day: float = float("nan")
    """Rate of change of the CORRECTION (reference minus probe), K/day.

    Sign convention matters and is easy to get backwards: ``dT_K`` is what you
    would have to ADD to the probe to reach the reference, so a probe reading
    progressively warm gives a NEGATIVE ``drift_K_per_day``.  Use
    :attr:`probe_drift_K_per_day` when you want "how fast is the probe
    wandering", which is what ``T1 - T2`` measures.
    """

    drift_se_K_per_day: float = float("nan")
    permutation_p: float = float("nan")
    span_days: float = float("nan")
    total_drift_K: float = float("nan")
    significant: bool = False
    reason: str = ""
    channel: str = "?"

    @property
    def probe_drift_K_per_day(self) -> float:
        """How fast the probe itself is wandering (sign-flipped correction)."""
        return -self.drift_K_per_day

    def summary(self) -> str:
        if not self.blocks:
            return f"{self.channel}: no blocks — {self.reason}"
        verdict = "SIGNIFICANT" if self.significant else "not significant"
        return (
            f"{self.channel}: probe drift {self.probe_drift_K_per_day:+.2e} "
            f"± {self.drift_se_K_per_day:.1e} K/day over {self.span_days:.1f} d "
            f"({-self.total_drift_K:+.4f} K end-to-end), p={self.permutation_p:.3f} "
            f"— {verdict}"
        )


def _higher_terms(L: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    """``sum_{i>=1} a_i L^i`` — everything except the offset we are re-fitting."""
    out = np.zeros_like(L, dtype=np.float64)
    for i, a in enumerate(coeffs):
        if i == 0:
            continue
        out = out + a * L**i
    return out


def _block_edges(t: np.ndarray, n_blocks: int | None, block_days: float | None) -> np.ndarray:
    t0, t1 = float(np.min(t)), float(np.max(t))
    if block_days:
        n = max(1, int(np.ceil((t1 - t0) / (block_days * SECONDS_PER_DAY))))
    else:
        n = max(1, int(n_blocks or 6))
    return np.linspace(t0, t1, n + 1)


def blocked_offsets(
    pairs: PairSet,
    fit: FitResult,
    *,
    n_blocks: int | None = 6,
    block_days: float | None = None,
    min_profiles: int = 3,
    geo: GeometryFit | None = None,
    dTdz: np.ndarray | None = None,
) -> list[Block]:
    """Per-block ``t_0`` with the global higher-order terms held fixed.

    ``geo`` is the geometry fitted alongside the calibration by
    :func:`odas_tpw.fp07cal.geometry.joint_fit`.  PASS IT whenever the
    calibration was fitted with geometry enabled.  The joint fit removes
    ``dz*g + tau*w*g`` as a nuisance effect, but this stage used to re-expose
    the ORIGINAL reference temperature to a calibration polynomial that no
    longer contains it, so any change in gradient exposure across the
    deployment reappeared as a change in the fitted intercept -- i.e. as bead
    drift.  On a synthetic stable bead with a fixed dz = 0.17 m, ZERO true
    drift and a gradient that grows with date, the joint fit recovered
    dz = 0.1700186 m at 1.01e-5 K RMS while this stage then reported
    +3.46e-4 K/day with permutation p = 0.005 and ``significant = True``
    (issue #180 F08).

    ``dTdz`` is the same gradient array handed to the joint fit; ``None``
    recomputes it with :func:`geometry.local_dTdz`.
    """
    keep = fit.kept if fit.kept.size == len(pairs) else np.ones(len(pairs), dtype=bool)
    t = np.asarray(pairs.time, dtype=np.float64)[keep]
    L = np.asarray(pairs.L, dtype=np.float64)[keep]
    T = np.asarray(pairs.T_ref, dtype=np.float64)[keep]
    uid = np.asarray(pairs.profile_uid, dtype=object)[keep]
    if t.size == 0:
        return []

    if geo is not None and np.isfinite(geo.dz_m):
        # Subtract the fitted nuisance geometry from the reference BEFORE
        # blocking, so the blocks see the temperature the bead's own depth
        # actually sampled -- the same quantity the joint fit calibrated against.
        from odas_tpw.fp07cal.geometry import local_dTdz

        g_all = local_dTdz(pairs) if dTdz is None else np.asarray(dTdz, dtype=np.float64)
        g = np.asarray(g_all, dtype=np.float64)[keep]
        w = np.asarray(pairs.w, dtype=np.float64)[keep]
        tau = geo.tau_s if np.isfinite(geo.tau_s) else 0.0
        corr = np.where(np.isfinite(g) & np.isfinite(w), geo.dz_m * g + tau * w * g, 0.0)
        T = T - corr

    y = 1.0 / (T + 273.15)
    a0_pair = y - _higher_terms(L, fit.coeffs)

    edges = _block_edges(t, n_blocks, block_days)
    blocks: list[Block] = []
    for b in range(edges.size - 1):
        lo, hi = edges[b], edges[b + 1]
        m = (t >= lo) & (t <= hi) if b == edges.size - 2 else (t >= lo) & (t < hi)
        if not np.any(m):
            continue

        # Collapse to one value per profile FIRST — profiles are the
        # independent unit, pairs within a profile are not.
        u, inv = np.unique(uid[m], return_inverse=True)
        sums = np.zeros(u.size)
        cnts = np.zeros(u.size)
        np.add.at(sums, inv, a0_pair[m])
        np.add.at(cnts, inv, 1.0)
        per_prof = sums / cnts
        if per_prof.size < min_profiles:
            continue

        a0 = float(np.mean(per_prof))
        se = (
            float(np.std(per_prof, ddof=1) / np.sqrt(per_prof.size))
            if per_prof.size > 1 else float("nan")
        )

        # Express as a temperature offset at the block's own median L, which is
        # the only interpretable form: K, not reciprocal kelvin.
        L_ref = float(np.median(L[m]))
        h = float(_higher_terms(np.array([L_ref]), fit.coeffs)[0])
        T_block = 1.0 / (a0 + h) - 273.15
        T_glob = float(temperature(np.array([L_ref]), fit.coeffs)[0])
        T_K = T_block + 273.15
        blocks.append(
            Block(
                t_mid=float(np.mean(t[m])),
                t_start=float(lo),
                t_end=float(hi),
                n_pairs=int(m.sum()),
                n_profiles=int(per_prof.size),
                a0=a0,
                a0_se=se,
                t_0=1.0 / a0 if a0 != 0 else float("nan"),
                dT_K=float(T_block - T_glob),
                dT_se_K=float(T_K**2 * se) if np.isfinite(se) else float("nan"),
            )
        )
    return blocks


def drift_fit(
    blocks: list[Block],
    *,
    n_permutations: int = 5000,
    seed: int = 0,
    alpha: float = 0.05,
    min_blocks: int = 4,
) -> StabilityResult:
    """Weighted trend through the block offsets, with a permutation test.

    The permutation test is the guard against reading noise as drift: it
    reshuffles which offset belongs to which time and asks how often chance
    produces a trend this steep.  A formal standard error alone would not do,
    because the blocks are few and their errors are not identically
    distributed.

    Weights are ``1 / max(se_i, median(se))^2`` --- each block's standard
    error is FLOORED at the median SE before inverting, so one block with a
    luckily tiny (or zero) SE cannot dominate the trend.  This is deliberately
    not the textbook ``1/sigma^2``.
    """
    res = StabilityResult(blocks=blocks)
    if len(blocks) < min_blocks:
        res.reason = f"only {len(blocks)} usable block(s); need {min_blocks}"
        return res

    t = np.array([b.t_mid for b in blocks], dtype=np.float64)
    d = np.array([b.dT_K for b in blocks], dtype=np.float64)
    se = np.array([b.dT_se_K for b in blocks], dtype=np.float64)
    ok = np.isfinite(t) & np.isfinite(d)
    if int(ok.sum()) < min_blocks:
        res.reason = "too many non-finite block offsets"
        return res
    t, d, se = t[ok], d[ok], se[ok]

    days = (t - t.min()) / SECONDS_PER_DAY
    fallback = np.nanmedian(se[np.isfinite(se)]) if np.any(np.isfinite(se)) else 1.0
    w = 1.0 / np.maximum(se, fallback) ** 2
    if not np.all(np.isfinite(w)):
        w = np.ones_like(d)

    slope, _intercept, slope_se = _wls(days, d, w)
    res.drift_K_per_day = slope
    res.drift_se_K_per_day = slope_se
    res.span_days = float(days.max() - days.min())
    res.total_drift_K = slope * res.span_days

    rng = np.random.default_rng(seed)
    obs = abs(slope)
    count = 0
    for _ in range(n_permutations):
        perm = rng.permutation(d.size)
        s, _i, _e = _wls(days, d[perm], w[perm])
        if abs(s) >= obs:
            count += 1
    res.permutation_p = (count + 1) / (n_permutations + 1)

    res.significant = bool(
        res.permutation_p < alpha
        and np.isfinite(slope_se)
        and slope_se > 0
        and abs(slope) > 2.0 * slope_se
    )
    res.reason = "ok"
    return res


def _wls(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[float, float, float]:
    """Weighted least squares line; returns ``(slope, intercept, slope_se)``.

    With ``w = 1/sigma^2`` the slope variance is ``1/Sxx``.  That is scaled by
    the reduced chi-square when the blocks scatter by more than their stated
    errors --- which they will whenever anything systematic is left in the
    residual --- so the reported uncertainty reflects the observed spread
    rather than an optimistic error model.  Never scaled *down*: a
    better-than-expected chi-square is not evidence of a tighter slope.
    """
    W = np.sum(w)
    if W <= 0 or x.size < 3:
        return float("nan"), float("nan"), float("nan")
    xm = np.sum(w * x) / W
    ym = np.sum(w * y) / W
    sxx = float(np.sum(w * (x - xm) ** 2))
    if sxx <= 0:
        return float("nan"), float("nan"), float("nan")
    slope = float(np.sum(w * (x - xm) * (y - ym)) / sxx)
    intercept = float(ym - slope * xm)
    dof = x.size - 2
    if dof <= 0:
        return slope, intercept, float("nan")
    resid = y - (intercept + slope * x)
    chi2_red = float(np.sum(w * resid**2) / dof)
    return slope, intercept, float(np.sqrt(max(chi2_red, 1.0) / sxx))


def t1_t2_series(
    probes: list[ProbeSeries],
    *,
    channel_a: str = "T1",
    channel_b: str = "T2",
) -> dict:
    """Per-profile ``mean(T1 - T2)`` over the WHOLE deployment.

    Needs no reference, so it covers the yos where the CT was off --- the blind
    spot in the blocked estimator above.  It is also the discriminator: drift
    that shows up here *and* in the blocked offsets is probe-specific and real;
    drift in the blocked offsets but not here points instead at the bridge
    electronics, the reference, or selection bias in which yos carried CT
    (plan section 8.4).

    Factory coefficients are used deliberately --- this is a *differential*
    measurement, and the in-situ fit would inject exactly the signal under test.
    """
    times: list[float] = []
    values: list[float] = []
    counts: list[int] = []
    labels: list[str] = []
    for p in probes:
        if channel_a not in p.counts or channel_b not in p.counts:
            continue
        La, ca = log_r(p.counts[channel_a], p.bridge[channel_a])
        Lb, cb = log_r(p.counts[channel_b], p.bridge[channel_b])
        Ta = temperature(La, p.factory[channel_a])
        Tb = temperature(Lb, p.factory[channel_b])
        diff = np.where(ca | cb, np.nan, Ta - Tb)
        for i, (s, e) in enumerate(p.profiles):
            seg = diff[s : e + 1]
            ok = np.isfinite(seg)
            if int(ok.sum()) < 10:
                continue
            times.append(float(np.mean(p.time[s : e + 1])))
            values.append(float(np.mean(seg[ok])))
            counts.append(int(ok.sum()))
            labels.append(f"{p.label}#{i}")

    t = np.array(times)
    v = np.array(values)
    order = np.argsort(t)
    out = {
        "time": t[order],
        "value": v[order],
        "n": np.array(counts)[order] if counts else np.empty(0, dtype=int),
        "label": np.array(labels, dtype=object)[order] if labels else np.empty(0, dtype=object),
        "slope_K_per_day": float("nan"),
        "available": bool(t.size),
    }
    if t.size > 2:
        t_sorted = np.asarray(out["time"], dtype=np.float64)
        v_sorted = np.asarray(out["value"], dtype=np.float64)
        days = (t_sorted - t_sorted.min()) / SECONDS_PER_DAY
        out["slope_K_per_day"] = float(np.polyfit(days, v_sorted, 1)[0])
    return out


def corroborates(
    stab: StabilityResult,
    t1t2: dict,
    *,
    tol: float = 0.5,
    channel_a: str = "T1",
    channel_b: str = "T2",
) -> str | None:
    """Does the differential back up the blocked drift?  ``None`` when undecidable.

    Compared in **probe-drift space**, not correction space --- the differential
    measures how fast a probe is wandering, whereas
    :attr:`StabilityResult.drift_K_per_day` is the correction and carries the
    opposite sign.  Comparing the two raw would invert every verdict.

    CHANNEL-AWARE.  :func:`t1_t2_series` always forms ``channel_a - channel_b``
    (T1 - T2), which tracks a T1 drift POSITIVELY and a T2 drift NEGATIVELY.
    Comparing the same differential sign for either channel therefore inverted
    the verdict for the second probe: a T2 that really drifts +0.001 K/day
    against a fixed T1 gives a T1-T2 slope of -0.001, which is exactly the
    corroborating sign, and the function reported "opposes in sign ...
    unexplained -- do not apply a drift model" (issue #180 F19). The
    differential is negated for ``channel_b`` so both channels are read in their
    own probe-drift space.

    Note this identifies neither probe on its own: a differential says the two
    moved relative to each other, and attributing that to one of them assumes
    the other (and the reference) held still.

    The three outcomes are genuinely different problems:

    * agreement in sign --- probe-specific drift, the drift model is licensed;
    * the differential flat against a significant blocked drift --- both probes
      moved together, which is not a bead story: suspect the bridge, the
      reference, or selection bias in which yos carried CT;
    * opposition in sign --- unexplained, and no drift should be applied.
    """
    if not t1t2.get("available") or not np.isfinite(t1t2.get("slope_K_per_day", np.nan)):
        return None
    a = stab.probe_drift_K_per_day
    b = float(t1t2["slope_K_per_day"])
    if stab.channel == channel_b:
        b = -b  # the differential tracks channel_b's drift with the opposite sign
    elif stab.channel not in (channel_a, "", "?"):
        return (
            f"channel {stab.channel!r} is not part of the {channel_a}-{channel_b} "
            "differential: nothing to corroborate"
        )
    if not np.isfinite(a):
        return None
    if not stab.significant:
        return (
            f"blocked drift not significant; {channel_a}-{channel_b} slope "
            f"{b:+.2e} K/day (in {stab.channel or channel_a} probe-drift sign) "
            f"(nothing to corroborate)"
        )
    if abs(b) < tol * abs(a):
        return (
            f"{channel_a}-{channel_b} is flat ({b:+.2e} K/day) against a significant probe drift "
            f"({a:+.2e} K/day): both probes moved together — suspect the bridge, "
            f"the reference, or CT-coverage selection bias, NOT the bead"
        )
    if np.sign(a) == np.sign(b):
        return (
            f"{channel_a}-{channel_b} agrees in sign ({b:+.2e} vs probe drift {a:+.2e} K/day): "
            f"probe-specific drift"
        )
    return (
        f"{channel_a}-{channel_b} opposes in sign ({b:+.2e} vs probe drift {a:+.2e} K/day): "
        f"unexplained — do not apply a drift model"
    )

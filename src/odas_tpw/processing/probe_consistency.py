# Jul-2026, Claude and Pat Welch, pat@mousebrains.com
"""Cross-probe consistency diagnostics for multi-probe epsilon / chi (#131).

Per-window QC (fom, FM, the Lueck inter-probe CI test in mk_epsilon_mean /
mk_chi_mean) catches windows where probes disagree, but misses PERSISTENT
systematic disagreement: ARCTERX vmp142's probe pairs were mutually
inconsistent by 1.8x (epsilon) and 2.2x (chi) over whole deployments with
per-window QC content, and CAS_080.P has sh1 epsilon ~1000x sh2 with
fom ~ 1 on both probes.

This module computes an OBSERVATIONAL per-profile summary metric for every
probe pair — the median ratio over the windows where both probes are finite —
plus its statistical significance, and emits a two-tier ``logger.warning``.
Nothing is ever auto-dropped.

Tiers
-----
statistical
    ``z > Z_WARN`` (3) with ``n >= N_MIN_STATISTICAL`` (20) windows, where
    ``z = |median ln-ratio| / SE_med`` and
    ``SE_med = 1.2533 * sqrt(2) * mean(sigma_ln) / sqrt(0.7 * n)``.
    1.2533 = sqrt(pi/2) is the asymptotic standard-error inflation of the
    median of a normal sample relative to the mean; ``sqrt(2) * sigma_ln`` is
    the standard deviation of one window's ln-ratio (difference of two
    ln values, each with the Lueck (2022) single-probe ``sigma_ln``); and
    ``0.7 * n`` approximates the effective number of independent windows under
    the standard 50% window overlap.  This tier catches offsets that are small
    but statistically unambiguous (ARCTERX-B2-class 1.2-1.3x pathologies become
    detectable at deployment-scale window counts, not in a single short profile).
practical
    ``max(ratio, 1/ratio) > ratio_max`` (:data:`PROBE_RATIO_MAX`, default 1.8
    — the vmp142 epsilon scale) with ``n >= N_MIN_PRACTICAL`` (10) windows,
    regardless of formal significance.  This tier catches calibration-scale
    offsets even when N is too small for the statistical tier.

The floor catches calibration-scale offsets; z catches subtler-but-significant
ones.  A systematic offset at the 1.8x level across a whole profile is
calibration-scale, not turbulence (the per-window statistical bound at typical
sigma_ln is far wider, which is why per-window QC never fires on it).

Scope / lifetime of the attrs
-----------------------------
The attrs written by :func:`annotate_probe_consistency` are PER-PROFILE
diagnostics on the diss/chi dataset only.  Depth binning and the combo stage
rebuild global attrs from a schema (``rsi/binning.py``), so these attrs do NOT
survive into binned or combined products — read them from the per-profile
``eps_NN``/``chi_NN`` (or perturb ``diss_NN``) files, or from the run log.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import xarray as xr

logger = logging.getLogger(__name__)

# Statistical tier: significance threshold and minimum window count.
Z_WARN = 3.0
N_MIN_STATISTICAL = 20
# NOTE: n_eff = 0.7*N assumes mild window correlation; under heavy 50%-overlap
# correlation the z>3 false-positive rate can reach ~1% (still per-profile,
# observational logging only).

# Practical tier: calibration-scale median-ratio floor (vmp142's epsilon pair
# disagreed by 1.8x with clean per-window QC) and its minimum window count.
# The rsi path uses this module constant; the perturb pipeline exposes it as
# the qc.probe_ratio_max config key.
PROBE_RATIO_MAX = 1.8
N_MIN_PRACTICAL = 10

# sqrt(pi/2): asymptotic SE of the median relative to the mean (normal sample).
_MEDIAN_SE_FACTOR = 1.2533
# Effective fraction of independent windows under 50% window overlap.
_N_EFF_FRACTION = 0.7


@dataclass(frozen=True)
class PairStat:
    """Median-ratio consistency statistic for one probe pair.

    ``median_ratio`` is ``exp(median(ln(a/b)))`` over the ``n_windows`` windows
    where both probes are finite and positive; ``z`` is the significance of the
    median ln-ratio against zero (see the module docstring).
    """

    name_a: str
    name_b: str
    median_ratio: float
    n_windows: int
    z: float

    @property
    def pair(self) -> str:
        return f"{self.name_a}/{self.name_b}"


def lueck_ln_sigma(
    values: npt.ArrayLike,
    nu: npt.ArrayLike,
    speed: npt.ArrayLike,
    diss_length: float,
    fs: float,
    var_resolved: npt.ArrayLike | None = None,
) -> np.ndarray:
    """Per-window, per-probe Lueck (2022) ``sigma_ln`` for ``values(probe, time)``.

    The same variance model mk_epsilon_mean / mk_chi_mean use:
    ``L_K = (nu^3 / values)^(1/4)``, ``L = speed * diss_length / fs``,
    ``L_hat = L / L_K`` (times ``V_f**0.75`` when *var_resolved* is given —
    the eq (18) spectral-truncation correction), and
    ``var_ln = 5.5 / (1 + (L_hat/4)^(7/9))``.

    For chi, pass the per-probe ``epsilon_T`` as *values* (the Kolmogorov
    length needs a dissipation rate; mirrors ``mk_chi_mean``).
    """
    vals = np.asarray(values, dtype=np.float64)
    nu_arr = np.asarray(nu, dtype=np.float64)
    speed_arr = np.asarray(speed, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        L_K = (nu_arr[np.newaxis, :] ** 3 / vals) ** 0.25
        L_hat = (speed_arr * diss_length / fs)[np.newaxis, :] / L_K
        if var_resolved is not None:
            vf = np.asarray(var_resolved, dtype=np.float64)
            L_hat = L_hat * np.clip(vf, 1e-6, 1.0) ** 0.75
        var_ln = 5.5 / (1.0 + (L_hat / 4.0) ** (7.0 / 9.0))
    return np.asarray(np.sqrt(var_ln))


def probe_pair_stats(
    values: npt.ArrayLike,
    sigma_ln: npt.ArrayLike,
    probe_names: list[str],
) -> list[PairStat]:
    """Median-ratio consistency statistics for every probe pair.

    Parameters
    ----------
    values : array (n_probe, n_time)
        Per-probe epsilon (or chi) estimates; only windows where BOTH probes
        of a pair are finite and positive contribute to that pair.
    sigma_ln : array (n_probe, n_time) or (n_time,)
        Per-window Lueck (2022) ``sigma_ln`` (see :func:`lueck_ln_sigma`);
        a 2-D array is averaged over the pair on the common windows.
    probe_names : list of str
        Probe labels, aligned with the rows of *values*.

    Returns
    -------
    list of PairStat
        One entry per unordered pair, in row order.  Empty when fewer than
        two probes are given.
    """
    vals = np.atleast_2d(np.asarray(values, dtype=np.float64))
    sig = np.asarray(sigma_ln, dtype=np.float64)
    n_probe = vals.shape[0]
    stats: list[PairStat] = []
    for i in range(n_probe):
        for j in range(i + 1, n_probe):
            with np.errstate(invalid="ignore"):
                both = (
                    np.isfinite(vals[i])
                    & np.isfinite(vals[j])
                    & (vals[i] > 0)
                    & (vals[j] > 0)
                )
            n = int(np.count_nonzero(both))
            if n == 0:
                stats.append(PairStat(probe_names[i], probe_names[j], math.nan, 0, math.nan))
                continue
            med = float(np.median(np.log(vals[i][both] / vals[j][both])))
            # 2-D sigma: average the PAIR's sigma over the common windows.
            pair_sig = np.concatenate([sig[i][both], sig[j][both]]) if sig.ndim == 2 else sig[both]
            with np.errstate(invalid="ignore"):
                mean_sig = float(np.nanmean(pair_sig)) if pair_sig.size else math.nan
            if math.isfinite(mean_sig) and mean_sig > 0:
                se_med = (
                    _MEDIAN_SE_FACTOR
                    * math.sqrt(2.0)
                    * mean_sig
                    / math.sqrt(_N_EFF_FRACTION * n)
                )
                z = abs(med) / se_med
            else:
                z = math.nan
            stats.append(
                PairStat(probe_names[i], probe_names[j], float(math.exp(med)), n, float(z))
            )
    return stats


def _warn_tiers(stat: PairStat, ratio_max: float) -> str | None:
    """The tier description when *stat* warrants a warning, else ``None``."""
    tiers = []
    if (
        math.isfinite(stat.z)
        and stat.z > Z_WARN
        and stat.n_windows >= N_MIN_STATISTICAL
    ):
        tiers.append(f"z={stat.z:.1f} > {Z_WARN:g}")
    r = stat.median_ratio
    if (
        math.isfinite(r)
        and r > 0
        and max(r, 1.0 / r) > ratio_max
        and stat.n_windows >= N_MIN_PRACTICAL
    ):
        tiers.append(f"ratio beyond {ratio_max:g}x practical floor")
    return "; ".join(tiers) if tiers else None


def annotate_probe_consistency(
    ds: xr.Dataset,
    values: npt.ArrayLike,
    sigma_ln: npt.ArrayLike,
    probe_names: list[str],
    *,
    quantity: str = "epsilon",
    attr_prefix: str = "",
    context: str = "",
    ratio_max: float = PROBE_RATIO_MAX,
) -> list[PairStat]:
    """Write cross-probe consistency attrs on *ds* and warn on disagreement.

    Always writes (when >= 2 probes; prefixed by *attr_prefix*, e.g. ``chi_``):

    - ``probe_ratio_pairs``   — comma-joined pair labels, e.g. ``"sh1/sh2"``
    - ``probe_ratio_median``  — per-pair median ratio (first/second probe)
    - ``n_ratio_windows``     — per-pair contributing-window count
    - ``probe_ratio_z``       — per-pair significance of the median ln-ratio
    - ``probe_ratio_comment`` — definition + caveats

    Emits one ``logger.warning`` per pair that trips either tier (see the
    module docstring).  Returns the :class:`PairStat` list (empty for a
    single probe — no attrs are written then).

    Stage note: on the rsi path this runs at the diss/chi dataset build, over
    ALL finite windows — no fom/FM QC cut has been applied at that stage (the
    perturb pipeline computes the same metric after its FM cut and epsilon
    floor, so the two contexts can differ slightly for the same data).
    """
    stats = probe_pair_stats(values, sigma_ln, probe_names)
    if not stats:
        return stats
    p = attr_prefix
    ds.attrs[f"{p}probe_ratio_pairs"] = ", ".join(s.pair for s in stats)
    ds.attrs[f"{p}probe_ratio_median"] = [s.median_ratio for s in stats]
    ds.attrs[f"{p}n_ratio_windows"] = [s.n_windows for s in stats]
    ds.attrs[f"{p}probe_ratio_z"] = [s.z for s in stats]
    ds.attrs[f"{p}probe_ratio_comment"] = (
        f"Cross-probe {quantity} consistency diagnostic (issue #131): per pair "
        f"({p}probe_ratio_pairs), the median first/second-probe ratio over the "
        f"n windows where both are finite ({p}n_ratio_windows), with the "
        f"significance z of the median ln-ratio ({p}probe_ratio_z; "
        "SE_med = 1.2533*sqrt(2)*mean(sigma_ln)/sqrt(0.7*n), Lueck 2022 "
        "per-window sigma_ln). Observational only — no windows or probes are "
        "dropped. Computed at the per-profile dataset build over all finite "
        "windows (no fom/FM QC cut at this stage). NOTE: these attrs do not "
        "survive depth binning or the combo stage (attrs are rebuilt from a "
        "schema there); read them from the per-profile files."
    )
    for s in stats:
        tier = _warn_tiers(s, ratio_max)
        if tier is not None:
            where = f"{context}: " if context else ""
            logger.warning(
                "%spersistent inter-probe %s disagreement %s: median ratio %.3g "
                "over %d windows (%s); check probe calibration/damage — see "
                "issue #131 data-quality notes",
                where,
                quantity,
                s.pair,
                s.median_ratio,
                s.n_windows,
                tier,
            )
    return stats

# Jul-2026, Claude and Pat Welch, pat@mousebrains.com
"""ATOMIX L3->L4 epsilon regression gate (committed-fixture, runs in CI).

``scripts/compare_atomix.py`` validates the rsi-tpw epsilon estimator against
the full ATOMIX shear-probe benchmark files, but those files (~15 MB each) live
behind the gitignored ``AtomixData`` symlink and are absent in CI. This test
pins the same L3->L4 comparison against a tiny committed fixture
(``tests/data/atomix/VMP250_HaroStrait_L3L4.nc``, ~0.5 MB) so a regression in
``scor160.l4._estimate_epsilon`` fails a normal ``pytest`` run rather than only
showing up in a manual benchmark sweep.

The fixture is a *verbatim* subset of the reference L3_spectra / L4_dissipation
groups from ``VMP250_TidalChannel_024.nc`` (Haro Strait, British Columbia,
Oct 2016; VMP-250, ATOMIX benchmark, Fer et al. 2024,
doi:10.1038/s41597-024-03323-y). It carries the reference cleaned shear spectra
(``SH_SPEC_CLEAN``) and the reference dissipation (``EPSI`` / ``METHOD``), plus
the root global attributes (provenance, DOI, and ``f_AA``); L1/L2 time series
are dropped. Regenerate with ``tests/data/atomix/make_vmp250_l3l4_fixture.py``
when the source moves.

The comparison loop here mirrors ``compare_atomix.compare_dataset``: for each
reference spectrum/probe it feeds ``SH_SPEC_CLEAN`` + ``KCYC`` through
``_estimate_epsilon`` with the production epsilon configuration and compares the
result to the reference ``EPSI``. Like compare_atomix, the anti-alias wavenumber
is K_AA = 0.9 * f_AA / W, with ``f_AA`` read from the *root* global attributes
(this fixture carries ``f_AA = 98`` Hz, so the integration IS anti-alias
truncated — the same code path the production estimator takes). fit_order=3 and
E_ISR_THRESHOLD match the production path.

Tolerances bracket the values produced by the current estimator against this
fixture (median ratio 0.9996, log10 RMSD 0.0246, 100% within a factor of 2,
64/64 method agreement, 64 valid pairs) with margin, so an incidental numeric
shift passes but a real regression (a systematic bias, a broken fit branch, a
method-selection change) fails. The two "fraction" gates are set to tolerate a
single boundary spectrum flipping across the four CI platforms / Python versions
(a known source of tiny ``np.roots``/``np.polyfit`` differences) while still
catching a multi-spectrum regression. They are intentionally tighter than
compare_atomix's headline pass/fail thresholds (RMSD < 0.5, 90% within a
decade), which are set for the noisier field datasets. The tolerances are
empirical, pinned to the current estimator; re-pin here if the estimator
legitimately moves.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest
import xarray as xr  # hard dependency (pyproject: xarray>=2024.1) -> gate never skips

from odas_tpw.scor160.l4 import E_ISR_THRESHOLD, _estimate_epsilon
from odas_tpw.scor160.ocean import visc35

_FIXTURE = Path(__file__).parent / "data" / "atomix" / "VMP250_HaroStrait_L3L4.nc"

# Empirical tolerances, pinned to the current estimator against this fixture.
# Current values: median_ratio 0.9996, log10_rmsd 0.0246, factor-2 100%,
# method 64/64, n_valid 64. Bands are wide enough to absorb harmless numeric
# drift but tight enough to catch a real bias/branch/method regression.
_MEDIAN_RATIO_LO = 0.98
_MEDIAN_RATIO_HI = 1.02
_LOG10_RMSD_MAX = 0.04
# Fraction gates tolerate ONE of 64 pairs flipping (63/64 = 0.984 >= 0.98) so a
# single boundary spectrum shifting across CI platforms doesn't red the build,
# while 2+ flips (a real regression) still fail.
_FRAC_MIN = 0.98
_MIN_VALID_PAIRS = 60  # fixture yields 64; guards against a silent empty run.


def _orient3(var) -> np.ndarray:
    """Orient an L3 (probe, wavenumber, time)-ish var to (time, wn, probe)."""
    d = var.dims
    t = next(i for i, x in enumerate(d) if "TIME" in x)
    w = next(i for i, x in enumerate(d) if "WAVENUMBER" in x)
    p = next(i for i, x in enumerate(d) if "SHEAR" in x)
    return np.moveaxis(var.values, [t, w, p], [0, 1, 2])


def _orient2(var) -> np.ndarray:
    """Orient an L4 (probe, time)-ish var to (time, probe)."""
    d = var.dims
    if len(d) == 1:
        return var.values[:, None]
    t = next(i for i, x in enumerate(d) if "TIME" in x)
    p = next(i for i, x in enumerate(d) if "SHEAR" in x)
    return np.moveaxis(var.values, [t, p], [0, 1])


def _compare_fixture() -> dict[str, float]:
    """Run the fixture's reference L3 spectra through ``_estimate_epsilon``.

    Returns summary metrics (median ratio, log10 RMSD, factor-2 fraction,
    method-agreement fraction, valid-pair count), mirroring
    ``compare_atomix.compare_dataset`` on the committed subset.
    """
    # f_AA lives in the ROOT global attrs (compare_atomix reads it there via
    # read_atomix_attrs). This fixture carries f_AA=98 Hz -> K_AA truncates the
    # integration below the Nyquist, exercising the anti-alias code path. The
    # 1e4 fallback (= effectively no truncation) is only for benchmark files
    # that genuinely lack f_AA; HP_cut is NOT the anti-alias frequency.
    with xr.open_dataset(_FIXTURE) as root:
        f_AA = float(root.attrs.get("f_AA", 1.0e4))
    fit_order = 3

    with (
        xr.open_dataset(_FIXTURE, group="L3_spectra") as l3,
        xr.open_dataset(_FIXTURE, group="L4_dissipation") as l4,
    ):
        sh = _orient3(l3["SH_SPEC_CLEAN"])
        kcyc_var = l3["KCYC"]
        if kcyc_var.ndim == 2:
            dd = kcyc_var.dims
            kt = next(i for i, x in enumerate(dd) if "TIME" in x)
            kw = next(i for i, x in enumerate(dd) if "WAVENUMBER" in x)
            kcyc = np.moveaxis(kcyc_var.values, [kt, kw], [0, 1])
        else:
            kcyc = kcyc_var.values
        speed = l3["PSPD_REL"].values
        epsi = _orient2(l4["EPSI"])
        method = _orient2(l4["METHOD"])
        kvisc = l4["KVISC"].values if "KVISC" in l4 else None

        n_spec, _, n_probe = sh.shape
        rows: list[tuple[float, float, int, int]] = []
        for i in range(n_spec):
            K = kcyc[i] if kcyc.ndim == 2 else kcyc
            if np.all(np.isnan(K)) or len(K) < 3:
                continue
            W = float(speed[i])
            if W <= 0 or not np.isfinite(W):
                continue
            nu = (
                float(kvisc[i])
                if (kvisc is not None and np.isfinite(kvisc[i]))
                else visc35(10.0)
            )
            K_AA = 0.9 * f_AA / W
            for p in range(n_probe):
                spec = sh[i, :, p]
                if np.all(np.isnan(spec)) or np.all(spec <= 0):
                    continue
                ea = float(epsi[i, p]) if p < epsi.shape[1] else np.nan
                if not np.isfinite(ea) or ea <= 0:
                    continue
                spec_c = np.where(np.isfinite(spec) & (spec > 0), spec, 0.0)
                K_c = np.where(np.isfinite(K) & (K > 0), K, 1e-10)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    out = _estimate_epsilon(
                        K_c, spec_c, nu, K_AA, fit_order,
                        e_isr_threshold=E_ISR_THRESHOLD,
                    )
                er = float(out[0])
                mr = int(out[3])
                ma = int(method[i, p]) if p < method.shape[1] else -1
                rows.append((ea, er, ma, mr))

    arr = np.array(rows, dtype=float)
    ea, er, ma, mr = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    valid = (ea > 0) & (er > 0) & np.isfinite(ea) & np.isfinite(er)
    ratio = er[valid] / ea[valid]
    return {
        "n_valid": int(valid.sum()),
        "median_ratio": float(np.median(ratio)),
        "log10_rmsd": float(
            np.sqrt(np.mean((np.log10(er[valid]) - np.log10(ea[valid])) ** 2))
        ),
        "frac_within_factor2": float(np.mean(np.abs(np.log10(ratio)) < np.log10(2.0))),
        "method_agree": float(np.mean(mr[valid] == ma[valid])),
    }


@pytest.fixture(scope="module")
def metrics() -> dict[str, float]:
    if not _FIXTURE.exists():  # pragma: no cover - fixture is committed
        pytest.fail(
            f"ATOMIX L3/L4 fixture missing: {_FIXTURE}. "
            "Regenerate with tests/data/atomix/make_vmp250_l3l4_fixture.py."
        )
    return _compare_fixture()


def test_enough_valid_pairs(metrics: dict[str, float]) -> None:
    """The fixture must actually yield comparable spectra (no silent empty run)."""
    assert metrics["n_valid"] >= _MIN_VALID_PAIRS, metrics


def test_median_ratio_unbiased(metrics: dict[str, float]) -> None:
    """rsi-tpw epsilon tracks the reference with < ~2% median bias."""
    assert _MEDIAN_RATIO_LO <= metrics["median_ratio"] <= _MEDIAN_RATIO_HI, metrics


def test_log10_rmsd_small(metrics: dict[str, float]) -> None:
    """Per-spectrum scatter vs the reference stays well under a decade."""
    assert metrics["log10_rmsd"] < _LOG10_RMSD_MAX, metrics


def test_nearly_all_within_factor_of_two(metrics: dict[str, float]) -> None:
    """Essentially every reference spectrum reproduces within a factor of 2
    (allowing a single boundary spectrum to flip across CI platforms)."""
    assert metrics["frac_within_factor2"] >= _FRAC_MIN, metrics


def test_method_selection_matches(metrics: dict[str, float]) -> None:
    """rsi-tpw picks the same variance/ISR method as the reference (allowing a
    single boundary spectrum to flip across CI platforms)."""
    assert metrics["method_agree"] >= _FRAC_MIN, metrics

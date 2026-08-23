"""Is beta_2 statistically significant, and does it help out of sample?

Three questions, three tests:

1. **Significance.** Is the fitted a_2 = 1/beta_2 large compared with its own
   standard error?  Reported as a t-statistic on the coefficient.
2. **In-sample gain.** Does the residual actually drop, and by enough to matter
   against the ~4 mK per-profile scatter?
3. **Out-of-sample, the one that counts.** Fit on part of the temperature range
   and predict the rest.  A higher order always fits the training data better;
   the question is whether it EXTRAPOLATES better, which is exactly what
   happens when a coefficient set is applied to profiles that went outside the
   range it was fitted on.
"""
import glob
import sys

import numpy as np

sys.path.insert(0, "/Users/pat/Desktop/turbulence/.claude/worktrees/fp07-insitu-cal/src")
from odas_tpw.fp07cal import PairConfig, load_hotel_reference, load_probe_series
from odas_tpw.fp07cal.fit import _design, _uncenter, fit_calibration
from odas_tpw.fp07cal.lag import temperature_lag
from odas_tpw.fp07cal.logr import coeffs_to_config, temperature
from odas_tpw.fp07cal.pairs import PairSet

D = "/Volumes/SeaChest/ARCTERX/2025/Interior/Gliders/osu685"


def fit_with_errors(L, T, order):
    """Least squares in a centred variable, with coefficient standard errors."""
    y = 1.0 / (T + 273.15)
    c, s = float(np.mean(L)), float(np.std(L))
    u = (L - c) / s
    X = _design(u, order)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ coef
    dof = max(1, y.size - X.shape[1])
    cov = float(r @ r) / dof * np.linalg.pinv(X.T @ X)
    se_u = np.sqrt(np.diag(cov))
    return _uncenter(coef, c, s), coef, se_u, X


def subset(ps: PairSet, m) -> PairSet:
    return PairSet(time=ps.time[m], T_ref=ps.T_ref[m], L=ps.L[m],
                   pressure=ps.pressure[m], w=ps.w[m], direction=ps.direction[m],
                   profile_uid=ps.profile_uid[m], file_label=ps.file_label[m],
                   channel=ps.channel, lag=ps.lag)


def main() -> None:
    ref = load_hotel_reference(
        f"{D}/PASS0/ebd.nc", value_var="sci_water_temp",
        pressure_var="sci_water_pressure", pressure_scale=10.0, valid_min=1.0,
    )
    cands = [f for f in sorted(glob.glob(f"{D}/MR/*.p")) if int(f[-6:-2]) >= 3]
    probes = []
    for f in cands[:: max(1, len(cands) // 30)][:30]:
        try:
            p = load_probe_series(f)
        except Exception:
            continue
        if p.profiles:
            probes.append(p)
    cfg = PairConfig(max_gap=30.0, min_speed=0.0)
    print(f"{len(probes)} files\n")

    for ch in ("T1", "T2"):
        lr, ps = temperature_lag(probes, ref, ch, cfg=cfg, max_lag=20.0, step=0.5)
        L, T = ps.L, ps.T_ref
        print(f"=== {ch}: {len(ps)} pairs, T {T.min():.2f}..{T.max():.2f} degC")

        prev_rms = None
        for order in (1, 2, 3):
            coeffs, cu, se_u, _X = fit_with_errors(L, T, order)
            rms = float(np.sqrt(np.mean((T - temperature(L, coeffs)) ** 2)))
            ce = coeffs_to_config(coeffs)
            tstat = abs(cu[-1]) / se_u[-1] if se_u[-1] > 0 else np.inf
            gain = "" if prev_rms is None else f"  ({1e3*(prev_rms-rms):+.3f} mK vs order {order-1})"
            print(f"  order {order}: rms {rms*1e3:7.3f} mK{gain}")
            print(f"     {'  '.join(f'{k}={v:.6g}' for k, v in ce.items())}")
            print(f"     top coefficient t-stat = {tstat:.1f}"
                  f"  ({'significant' if tstat > 3 else 'NOT significant'})")
            prev_rms = rms

        # ---- out of sample: fit warm, predict cold, and vice versa ----------
        print("  out-of-sample (fit one half of the T range, predict the other):")
        mid = float(np.median(T))
        for name, tr, te in (("fit WARM -> predict COLD", T >= mid, T < mid),
                             ("fit COLD -> predict WARM", T < mid, T >= mid)):
            row = []
            for order in (1, 2, 3):
                try:
                    coeffs, *_ = fit_with_errors(L[tr], T[tr], order)
                except Exception:
                    row.append(np.nan); continue
                pred = temperature(L[te], coeffs)
                row.append(float(np.sqrt(np.nanmean((T[te] - pred) ** 2))))
            best = int(np.nanargmin(row)) + 1
            print(f"    {name}: " + "  ".join(
                f"order {o}: {v*1e3:8.2f} mK" for o, v in zip((1, 2, 3), row))
                + f"   -> best order {best}")
        print()


if __name__ == "__main__":
    main()

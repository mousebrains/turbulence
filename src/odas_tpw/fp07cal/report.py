# Aug-2026, Claude and Pat Welch, pat@mousebrains.com
"""Rendering the diagnostics a human has to look at before believing the fit.

A least-squares fit always returns numbers, and on this data the RMS residual
is the *least* informative of them.  The report leads with the things that
actually decide the question --- the dive/climb split, the errors-in-variables
bracket, the correlation the lag was chosen on, and the rejection accounting ---
and puts the coefficients at the end where they belong.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from odas_tpw.fp07cal.fit import FitResult, residual_breakdown
from odas_tpw.fp07cal.pairs import PairSet
from odas_tpw.fp07cal.stability import SECONDS_PER_DAY, StabilityResult, corroborates


def _iso(t: float) -> str:
    if not np.isfinite(t):
        return "?"
    return datetime.fromtimestamp(float(t), tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def coverage_text(ref_report: dict, per_file: dict, max_gap: float) -> str:
    """The Phase-0 question: what reference do we actually have?"""
    lines = [
        "# FP07 in-situ calibration — reference coverage",
        "",
        f"- reference samples: **{ref_report['n_samples']}**",
        f"- continuous spans (max_gap {max_gap:g} s): **{ref_report['n_spans']}**",
        f"- median sample interval: {ref_report['median_interval_s']:.3g} s",
        f"- covered {ref_report['covered_s'] / 3600:.1f} h of "
        f"{ref_report['total_s'] / 3600:.1f} h "
        f"(**duty cycle {100 * ref_report['duty_cycle']:.1f}%**)",
        f"- reference temperature range: {ref_report['T_min']:.3f} .. "
        f"{ref_report['T_max']:.3f} degC",
        "",
        "The duty cycle is the every-n-th-yo ratio stated plainly. It is not a",
        "problem to be fixed — the fit pools whatever is there and the patch step",
        "applies the result to every file. It does bound the temperature range the",
        "coefficients are valid over, which is what the fit report checks.",
        "",
        "## Pairs contributed per file",
        "",
        "| file | pairs |",
        "|---|---|",
    ]
    for name in sorted(per_file):
        lines.append(f"| `{name}` | {per_file[name]} |")
    zero = [n for n, v in per_file.items() if v == 0]
    lines += [
        "",
        f"{len(zero)} of {len(per_file)} files contributed no pairs. That is the "
        "expected outcome for a deployment that sampled CT on every n-th yo, not "
        "an error: they are still calibrated, from the pooled fit.",
    ]
    return "\n".join(lines)


def fit_text(
    pairs: PairSet,
    fit: FitResult,
    *,
    clock_offset: tuple[float, float] | None = None,
    stab: StabilityResult | None = None,
    t1t2: dict | None = None,
    min_corr: float = 0.7,
) -> str:
    cfgeq = fit.config_equivalent
    lo, hi = fit.beta1_bracket
    span = (
        (float(np.max(pairs.time)) - float(np.min(pairs.time))) / SECONDS_PER_DAY
        if len(pairs)
        else float("nan")
    )

    warn: list[str] = []
    if np.isfinite(fit.corr) and abs(fit.corr) < min_corr:
        warn.append(
            f"**Lag correlation |r| = {abs(fit.corr):.3f} is below the {min_corr:g} "
            f"gate.** The lag is not established; do not use these coefficients."
        )
    if np.isfinite(fit.dive_climb_split_K) and abs(fit.dive_climb_split_K) > 2 * fit.rms_K:
        warn.append(
            f"**Dive/climb split {fit.dive_climb_split_K:+.4f} K exceeds twice the "
            f"RMS residual ({fit.rms_K:.4f} K).** That is the signature of an "
            f"unremoved lag or a thermal-mass artifact, and it is currently being "
            f"absorbed into t_0. Fix the lag before trusting the offset."
        )
    if np.isfinite(lo) and np.isfinite(hi) and hi > 0:
        rel = (hi - lo) / abs(cfgeq.get("beta_1", np.nan))
        if np.isfinite(rel) and rel > 0.01:
            warn.append(
                f"**Errors-in-variables bracket on beta_1 spans {100 * rel:.1f}%.** "
                f"The bandwidth match is not doing its job; the point estimate "
                f"understates beta_1."
            )
    T_lo, T_hi = fit.T_range
    if np.isfinite(T_lo) and (T_hi - T_lo) < 8.0 and fit.order > 1:
        warn.append(
            f"**Order {fit.order} fitted over only {T_hi - T_lo:.1f} degC.** Poorly "
            f"constrained, and it extrapolates badly onto yos that went outside "
            f"this range. Prefer order 1."
        )

    lines = [
        f"# FP07 in-situ calibration — {fit.channel}",
        "",
        f"Generated {_iso(datetime.now(tz=UTC).timestamp())}",
        "",
    ]
    if warn:
        lines += ["## ⚠ Warnings", ""] + [f"- {w}" for w in warn] + [""]
    else:
        lines += ["All diagnostic gates passed.", ""]

    lines += [
        "## Diagnostics (read these first)",
        "",
        "| quantity | value | what it means |",
        "|---|---|---|",
        f"| lag correlation r | {fit.corr:.5f} | how well L tracks 1/T_ref; the lag was chosen to maximise this |",
        f"| dive/climb split | {fit.dive_climb_split_K:+.5f} K | should be ~0; nonzero means unremoved lag |",
        f"| RMS residual | {fit.rms_K:.5f} K | least informative number here |",
        f"| beta_1 EIV bracket | {lo:.2f} .. {hi:.2f} | true slope lies inside; wide = attenuated |",
        f"| condition number | {fit.condition:.3g} | of the centered design matrix |",
        f"| pairs used / dropped | {fit.n} / {fit.n_dropped} | robust rejection at 4 sigma |",
        f"| profiles | {pairs.n_profiles()} | the independent unit |",
        f"| fitted T range | {T_lo:.3f} .. {T_hi:.3f} degC | do NOT extrapolate beyond |",
        f"| deployment span | {span:.2f} d | |",
    ]
    if clock_offset and np.isfinite(clock_offset[0]):
        co, cr = clock_offset
        lines.append(
            f"| clock offset (P vs P) | {co:+.2f} s (r={cr:.4f}) | "
            f"instrument-vs-glider clock, measured without thermal physics |"
        )
        if np.isfinite(fit.lag):
            lines.append(
                f"| residual thermal lag | {fit.lag - co:+.2f} s | "
                f"total lag {fit.lag:+.2f} s minus the clock offset = sensor response |"
            )
    else:
        lines.append(
            f"| total lag | {fit.lag:+.2f} s | conflates clock offset and sensor "
            f"response — no CTD pressure available to separate them |"
        )

    lines += ["", "## Rejection accounting", "", "| reason | count |", "|---|---|"]
    for reason, count in sorted(pairs.rejected.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {reason} | {count} |")

    br = residual_breakdown(pairs, fit)
    lines += ["", "## Residual structure", ""]
    for name, unit in (("pressure", "K/dbar"), ("time", "K/s")):
        b = br.get(name)
        if b is None:
            lines.append(f"- vs {name}: too few points")
        else:
            lines.append(f"- vs {name}: slope {b['slope']:+.3e} {unit}")

    if stab is not None:
        lines += ["", "## Temporal stability", "", f"- {stab.summary()}"]
        if stab.blocks:
            lines += [
                "",
                "| block | start | profiles | t_0 | offset [K] |",
                "|---|---|---|---|---|",
            ]
            for i, b in enumerate(stab.blocks):
                lines.append(
                    f"| {i} | {_iso(b.t_start)} | {b.n_profiles} | "
                    f"{b.t_0:.4f} | {b.dT_K:+.5f} ± {b.dT_se_K:.5f} |"
                )
        if t1t2 is not None:
            verdict = corroborates(stab, t1t2)
            if verdict:
                lines += ["", f"- **T1−T2 check:** {verdict}"]
            elif not t1t2.get("available"):
                lines += [
                    "",
                    "- **T1−T2 check unavailable** (single thermistor). The blind "
                    "spot — drift on the yos with no CT — is uncovered, so treat "
                    "any drift conclusion as provisional.",
                ]

    lines += [
        "",
        "## Coefficients",
        "",
        "```",
        f"1/T_K = {' + '.join(f'{a:.10e}*L^{i}' for i, a in enumerate(fit.coeffs))}",
        "```",
        "",
        "| config key | value |",
        "|---|---|",
    ]
    for k, v in cfgeq.items():
        lines.append(f"| `{k}` | {v:.6f} |")
    lines += [
        "",
        "These are only valid alongside the bridge constants they were fitted "
        "against, and only over the temperature range above.",
    ]
    return "\n".join(lines)


def figure(
    pairs: PairSet,
    fit: FitResult,
    *,
    stab: StabilityResult | None = None,
    t1t2: dict | None = None,
    path: str | Path = "fp07cal.png",
) -> Path:
    """Four panels: the fit, the residual structure, and the stability record."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    keep = fit.kept
    d = np.asarray(pairs.direction)

    a = ax[0, 0]
    a.plot(pairs.L[keep], pairs.T_ref[keep], ".", ms=1, alpha=0.3, label="pairs")
    Lg = np.linspace(*fit.L_range, 200)
    a.plot(Lg, fit.apply(Lg), "-", lw=1.5, color="crimson", label="fit")
    a.set_xlabel("L = ln(R_T/R_0)")
    a.set_ylabel("reference T [degC]")
    a.set_title(f"{fit.channel}: {fit.n} pairs, {pairs.n_profiles()} profiles")
    a.legend(fontsize=8)

    a = ax[0, 1]
    for sel, lab, c in ((d > 0, "dive", "tab:blue"), (d < 0, "climb", "tab:orange")):
        m = keep & sel
        if np.any(m):
            a.plot(pairs.T_ref[m], fit.residual_K[m], ".", ms=1, alpha=0.3, label=lab, color=c)
    a.axhline(0, color="k", lw=0.6)
    a.set_xlabel("reference T [degC]")
    a.set_ylabel("residual [K]")
    a.set_title(f"dive/climb split {fit.dive_climb_split_K:+.4f} K")
    a.legend(fontsize=8, markerscale=6)

    a = ax[1, 0]
    m = keep & np.isfinite(pairs.pressure)
    if np.any(m):
        a.plot(fit.residual_K[m], pairs.pressure[m], ".", ms=1, alpha=0.3)
        a.invert_yaxis()
    a.axvline(0, color="k", lw=0.6)
    a.set_xlabel("residual [K]")
    a.set_ylabel("pressure [dbar]")
    a.set_title("residual vs depth")

    a = ax[1, 1]
    plotted = False
    if stab is not None and stab.blocks:
        t0 = min(b.t_start for b in stab.blocks)
        x = np.array([(b.t_mid - t0) / SECONDS_PER_DAY for b in stab.blocks])
        y = np.array([b.dT_K for b in stab.blocks])
        e = np.array([b.dT_se_K for b in stab.blocks])
        a.errorbar(x, y, yerr=e, fmt="o", capsize=3, label="blocked offset")
        if np.isfinite(stab.drift_K_per_day):
            a.plot(x, np.polyval([stab.drift_K_per_day, np.mean(y - stab.drift_K_per_day * x)], x),
                   "-", color="crimson", lw=1.2,
                   label=f"{stab.probe_drift_K_per_day:+.2e} K/day (probe)")
        a.set_title(stab.summary(), fontsize=8)
        plotted = True
        if t1t2 and t1t2.get("available") and t1t2["time"].size:
            a2 = a.twinx()
            xt = (t1t2["time"] - t0) / SECONDS_PER_DAY
            a2.plot(xt, t1t2["value"] - np.mean(t1t2["value"]), ".", ms=3,
                    color="tab:green", alpha=0.6)
            a2.set_ylabel("T1 − T2 (demeaned) [K]", color="tab:green")
    if not plotted:
        a.text(0.5, 0.5, "no stability blocks", ha="center", va="center",
               transform=a.transAxes)
    a.axhline(0, color="k", lw=0.6)
    a.set_xlabel("days since start")
    a.set_ylabel("offset [K]")
    a.legend(fontsize=8, loc="best")

    fig.tight_layout()
    out = Path(path)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out

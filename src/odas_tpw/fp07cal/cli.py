# Aug-2026, Claude and Pat Welch, pat@mousebrains.com
"""``fp07-cal`` — the pre-pipeline FP07 in-situ calibration tool.

Runs before perturb, once per deployment.  ``coverage`` answers "what reference
do I actually have?", ``fit`` produces coefficients plus the diagnostics that
say whether to believe them, and ``demo`` runs the whole thing on a synthetic
deployment with known answers so the machinery can be exercised without data.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from odas_tpw.fp07cal.fit import fit_calibration
from odas_tpw.fp07cal.lag import pressure_offset, temperature_lag
from odas_tpw.fp07cal.pairs import PairConfig, build_pairs_multi
from odas_tpw.fp07cal.report import coverage_text, figure, fit_text
from odas_tpw.fp07cal.series import load_hotel_reference, load_probe_series
from odas_tpw.fp07cal.stability import blocked_offsets, drift_fit, t1_t2_series

TEMPLATE = """\
# fp07-cal — FP07 in-situ calibration (a PRE-PIPELINE step)
#
# Run order:  perturb trim -> fp07-cal fit -> (patch) -> perturb run
#
# Paths may use <CONFIG_DIR>, which resolves to this file's directory.

files:
  p_file_root: "<CONFIG_DIR>"
  p_file_pattern: "**/*.p"
  output_dir: "<CONFIG_DIR>/fp07cal"

reference:
  # Read STRAIGHT from the hotel NetCDF, on the CTD's own clock. Deliberately
  # not through perturb's hotel merge, which interpolates across arbitrary gaps
  # and edge-holds outside coverage -- on a glider sampling CT every n-th yo
  # that would hand the fit a fabricated ramp over most of the record.
  file: "<CONFIG_DIR>/hotel.nc"
  time_var: "sci_ctd41cp_timestamp"
  value_var: "sci_water_temp"
  pressure_var: "sci_water_pressure"   # enables the clock-offset measurement
  pressure_scale: 1.0  # a Slocum reports sci_water_pressure in BAR. A hotel
                       # file from `dinkum-hotel` has already applied the x10,
                       # so 1.0 is right there -- but set 10.0 when pointing
                       # straight at a raw converted ebd.nc, which has not.
  valid_min: -5.0
  valid_max: 45.0

pairs:
  max_gap: 30.0        # [s] largest CTD spacing still counted as coverage.
                       # NOT a smoothing knob -- it defines where the
                       # reference exists at all.
  kernel_width: null   # [s] boxcar; null = the CTD's median sample interval
  kernel_tau: 0.5      # [s] single pole approximating the CTD's response
  min_speed: 0.05      # [m/s] below this the thermistor is not flushing
  require_profile: true

channels: ["T1", "T2"]

lag:
  max_lag: 20.0
  step: 0.25
  detrend_s: 30.0      # High-pass BEFORE scoring, and gate on peak sharpness
                       # rather than on r. A glider dive is a monotonic ramp,
                       # and shifting a straight line in time returns the same
                       # line plus a constant, which every correlation removes:
                       # measured on real data, raw pressure scored r=1.000000
                       # at EVERY lag over +/-30 s (dynamic range 2e-5). A high
                       # r is not evidence. Cross-check 20/30/60 s -- a window
                       # that is too short locks onto a wrong lobe.

fit:
  order: 1             # 1 by default on purpose: a glider rarely spans the
                       # >8 degC an order-2 fit needs, the Vandermonde is badly
                       # conditioned over a narrow range, and a line
                       # extrapolates gracefully onto yos that went outside it
  robust: true
  geometry: true       # Fit the sensor mounting offset JOINTLY with the
                       # coefficients. The FP07 and CTD sit at different depths
                       # at the same instant, contributing dz*dT/dz to the
                       # residual -- which looks exactly like a depth-dependent
                       # calibration. Estimating it afterwards does not work:
                       # dT/dz correlates with T on a monotone profile, so an
                       # ordinary fit absorbs it into t_0 (25 cm injected came
                       # back as 0.4 cm).

stability:
  n_blocks: 6
  block_days: null     # set this instead of n_blocks for a fixed cadence
  min_profiles: 3
"""


def _resolve(value, config_dir: Path):
    if isinstance(value, str):
        return value.replace("<CONFIG_DIR>", str(config_dir))
    return value


def _load_config(path: Path) -> dict:
    from ruamel.yaml import YAML

    yaml = YAML(typ="safe")
    with open(path) as fh:
        cfg = yaml.load(fh) or {}
    d = path.parent.resolve()
    for section in cfg.values():
        if isinstance(section, dict):
            for k, v in section.items():
                section[k] = _resolve(v, d)
    return cfg


def _gather(cfg: dict):
    files_cfg = cfg.get("files", {})
    root = Path(files_cfg.get("p_file_root", "."))
    pattern = files_cfg.get("p_file_pattern", "**/*.p")
    paths = sorted(root.glob(pattern))
    if not paths:
        raise SystemExit(f"no .p files matched {root}/{pattern}")
    probes = [load_probe_series(p) for p in paths]

    ref_cfg = cfg.get("reference", {})
    ref = load_hotel_reference(
        ref_cfg["file"],
        time_var=ref_cfg.get("time_var", "sci_ctd41cp_timestamp"),
        value_var=ref_cfg.get("value_var", "sci_water_temp"),
        pressure_var=ref_cfg.get("pressure_var", "sci_water_pressure"),
        pressure_scale=float(ref_cfg.get("pressure_scale", 1.0)),
        valid_min=float(ref_cfg.get("valid_min", -5.0)),
        valid_max=float(ref_cfg.get("valid_max", 45.0)),
    )
    return probes, ref


def _pair_config(cfg: dict) -> PairConfig:
    p = cfg.get("pairs", {}) or {}
    return PairConfig(
        max_gap=float(p.get("max_gap", 30.0)),
        kernel_width=p.get("kernel_width"),
        kernel_tau=float(p.get("kernel_tau", 0.5)),
        min_speed=float(p.get("min_speed", 0.05)),
        require_profile=bool(p.get("require_profile", True)),
        min_corr=float(p.get("min_corr", 0.7)),
    )


def run_calibration(probes, ref, cfg: dict, out_dir: Path, *, make_figure: bool = True) -> dict:
    """Lag -> pairs -> fit -> stability -> report, for each requested channel."""
    pc = _pair_config(cfg)
    lag_cfg = cfg.get("lag", {}) or {}
    fit_cfg = cfg.get("fit", {}) or {}
    st_cfg = cfg.get("stability", {}) or {}
    channels = cfg.get("channels") or sorted({c for p in probes for c in p.counts})

    out_dir.mkdir(parents=True, exist_ok=True)
    po = pressure_offset(probes, ref)
    clock = (po.lag, po.score)
    print(f"  {po.summary()}")
    t1t2 = t1_t2_series(probes)
    results: dict = {"clock_offset_s": clock[0], "clock_offset_r": clock[1], "channels": {}}

    for ch in channels:
        if not any(ch in p.counts for p in probes):
            print(f"  {ch}: not present in any file — skipped", file=sys.stderr)
            continue
        lr, pairs = temperature_lag(
            probes, ref, ch, cfg=pc,
            max_lag=float(lag_cfg.get("max_lag", 20.0)),
            step=float(lag_cfg.get("step", 0.25)),
            detrend_s=float(lag_cfg.get("detrend_s", 30.0)),
        )
        lag, r = lr.lag, lr.score
        print(f"  {lr.summary()}")
        if len(pairs) == 0:
            print(f"  {ch}: zero pairs — no usable reference coverage", file=sys.stderr)
            results["channels"][ch] = {"error": "zero pairs", "rejected": dict(pairs.rejected)}
            continue

        geo = None
        if fit_cfg.get("geometry", True):
            from odas_tpw.fp07cal.geometry import joint_fit

            fit, geo = joint_fit(
                pairs, order=int(fit_cfg.get("order", 1)),
                robust=bool(fit_cfg.get("robust", True)),
            )
            print(f"  {ch} geometry: {geo.summary()}")
            if np.isfinite(geo.collinearity) and geo.collinearity > 0.8:
                print(
                    f"      NOTE collinearity {geo.collinearity:.3f}: dz and the "
                    f"residual lag are NOT separately resolved (vertical speed "
                    f"one-signed — profiles in only one direction). Their "
                    f"combination is measured; the split is not."
                )
        else:
            fit = fit_calibration(
                pairs,
                order=int(fit_cfg.get("order", 1)),
                robust=bool(fit_cfg.get("robust", True)),
            )
        blocks = blocked_offsets(
            pairs, fit,
            n_blocks=st_cfg.get("n_blocks", 6),
            block_days=st_cfg.get("block_days"),
            min_profiles=int(st_cfg.get("min_profiles", 3)),
        )
        stab = drift_fit(blocks)
        stab.channel = ch

        md = fit_text(pairs, fit, clock_offset=clock, stab=stab, t1t2=t1t2,
                      min_corr=pc.min_corr)
        (out_dir / f"{ch}_report.md").write_text(md)
        if make_figure:
            figure(pairs, fit, stab=stab, t1t2=t1t2, path=out_dir / f"{ch}_diagnostics.png")

        results["channels"][ch] = {
            "lag_s": lag,
            "corr": r,
            "thermal_lag_s": (lag - clock[0]) if np.isfinite(clock[0]) else None,
            "n_pairs": len(pairs),
            "n_profiles": pairs.n_profiles(),
            "order": fit.order,
            "coefficients": fit.coeffs.tolist(),
            "config_equivalent": fit.config_equivalent,
            "rms_K": fit.rms_K,
            "dive_climb_split_K": fit.dive_climb_split_K,
            "beta1_bracket": list(fit.beta1_bracket),
            "condition": fit.condition,
            "T_range": list(fit.T_range),
            "lag_trustworthy": lr.trustworthy(),
            "lag_dynamic_range": lr.dynamic_range,
            "lag_width_s": lr.width,
            "clock_trustworthy": po.trustworthy(),
            "geometry": None if geo is None else {
                "dz_m": geo.dz_m, "dz_se_m": geo.dz_se_m,
                "tau_s": geo.tau_s, "tau_se_s": geo.tau_se_s,
                "collinearity": geo.collinearity,
                "separately_resolved": bool(
                    np.isfinite(geo.collinearity) and geo.collinearity <= 0.8
                ),
            },
            "stability": {
                "probe_drift_K_per_day": stab.probe_drift_K_per_day,
                "se_K_per_day": stab.drift_se_K_per_day,
                "permutation_p": stab.permutation_p,
                "significant": stab.significant,
                "n_blocks": len(stab.blocks),
                "reason": stab.reason,
            },
        }
        print(f"  {ch}: {stab.summary()}")
        print(
            f"      t_0={fit.config_equivalent['t_0']:.4f} "
            f"beta_1={fit.config_equivalent.get('beta_1', float('nan')):.2f} "
            f"lag={lag:+.2f}s r={r:.4f} rms={fit.rms_K:.4f}K "
            f"split={fit.dive_climb_split_K:+.4f}K"
        )

    (out_dir / "coefficients.json").write_text(json.dumps(results, indent=2, default=float))
    return results


def _cmd_init(args) -> int:
    out = Path(args.output)
    if out.exists() and not args.force:
        print(f"{out} exists; use --force", file=sys.stderr)
        return 1
    out.write_text(TEMPLATE)
    print(f"wrote {out}")
    return 0


def _cmd_coverage(args) -> int:
    cfg = _load_config(Path(args.config))
    probes, ref = _gather(cfg)
    pc = _pair_config(cfg)
    channels = cfg.get("channels") or sorted({c for p in probes for c in p.counts})
    ch = next((c for c in channels if any(c in p.counts for p in probes)), None)
    per_file = {}
    if ch:
        pairs = build_pairs_multi(probes, ref, ch, lag=0.0, cfg=pc)
        per_file = pairs.per_file
    text = coverage_text(ref.coverage_report(pc.max_gap), per_file, pc.max_gap)
    out_dir = Path(cfg.get("files", {}).get("output_dir", "fp07cal"))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "coverage.md").write_text(text)
    print(text)
    return 0


def _cmd_fit(args) -> int:
    cfg = _load_config(Path(args.config))
    probes, ref = _gather(cfg)
    out_dir = Path(cfg.get("files", {}).get("output_dir", "fp07cal"))
    print(f"{len(probes)} file(s), {ref.time.size} reference samples")
    run_calibration(probes, ref, cfg, out_dir, make_figure=not args.no_figure)
    print(f"wrote {out_dir}/")
    return 0


def _cmd_demo(args) -> int:
    """Exercise the whole chain on a synthetic deployment with known answers."""
    from odas_tpw.fp07cal.synth import SynthConfig, make_deployment

    scfg = SynthConfig(
        n_yos=args.yos,
        yo_seconds=args.yo_seconds,
        fs=4.0,
        ct_every_n=args.ct_every_n,
        files_per_deployment=max(1, args.yos // 10),
        clock_offset=args.clock_offset,
        ctd_delay=1.0,
        drift_K_per_day=args.drift,
    )
    probes, ref, truth = make_deployment(scfg, t2_drift_K_per_day=args.t2_drift)
    out_dir = Path(args.output)
    print(
        f"synthetic: {truth['n_yos']} yos, CT on {truth['n_yos_with_ct']} of them, "
        f"{len(probes)} files, {ref.time.size} reference samples"
    )
    print(
        f"truth: t_0={truth['t_0']:.4f} beta_1={truth['beta_1']:.2f} "
        f"clock_offset={truth['clock_offset']:+.2f}s "
        f"ctd_delay={truth['ctd_delay']:+.2f}s "
        f"probe_drift={truth['drift_K_per_day']:+.2e} K/day"
    )
    cfg = {
        "pairs": {"max_gap": 30.0, "min_corr": 0.7},
        "lag": {"max_lag": 12.0, "step": 0.5},
        "fit": {"order": 1},
        "stability": {"n_blocks": args.blocks},
        "channels": ["T1"],
    }
    res = run_calibration(probes, ref, cfg, out_dir, make_figure=not args.no_figure)
    r = res["channels"].get("T1", {})
    if "config_equivalent" in r:
        ce = r["config_equivalent"]
        print(
            f"recovered: t_0={ce['t_0']:.4f} (err {ce['t_0'] - truth['t_0']:+.2e} K) "
            f"beta_1={ce['beta_1']:.2f} (err {1e6 * (ce['beta_1'] / truth['beta_1'] - 1):+.1f} ppm)"
        )
        print(
            f"clock offset {res['clock_offset_s']:+.2f}s "
            f"(truth {truth['clock_offset']:+.2f}s), "
            f"thermal lag {r['thermal_lag_s']:+.2f}s "
            f"(truth {truth['ctd_delay']:+.2f}s)"
        )
    print(f"wrote {out_dir}/")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="fp07-cal",
        description="FP07 in-situ calibration — a pre-pipeline step. "
        "Fits one Steinhart-Hart coefficient set per deployment from whatever "
        "yos carried a CTD reference, and reports whether it is stable in time.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    q = sub.add_parser("init", help="write a commented template config")
    q.add_argument("-o", "--output", default="fp07-cal.yaml")
    q.add_argument("--force", action="store_true")
    q.set_defaults(func=_cmd_init)

    q = sub.add_parser("coverage", help="what reference coverage do we actually have?")
    q.add_argument("-c", "--config", required=True)
    q.set_defaults(func=_cmd_coverage)

    q = sub.add_parser("fit", help="fit coefficients + stability diagnostic")
    q.add_argument("-c", "--config", required=True)
    q.add_argument("--no-figure", action="store_true")
    q.set_defaults(func=_cmd_fit)

    q = sub.add_parser("demo", help="run the whole chain on synthetic data")
    q.add_argument("-o", "--output", default="fp07cal_demo")
    q.add_argument("--yos", type=int, default=120)
    q.add_argument("--yo-seconds", type=float, default=9000.0)
    q.add_argument("--ct-every-n", type=int, default=5)
    q.add_argument("--drift", type=float, default=0.002, help="probe drift [K/day]")
    q.add_argument("--t2-drift", type=float, default=0.0)
    q.add_argument("--clock-offset", type=float, default=3.0)
    q.add_argument("--blocks", type=int, default=6)
    q.add_argument("--no-figure", action="store_true")
    q.set_defaults(func=_cmd_demo)

    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

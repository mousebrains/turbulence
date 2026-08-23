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

from odas_tpw.fp07cal.fit import fit_calibration, select_order
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
  max_fit_files: 100   # how many files to hold in memory for the lag+fit stage,
                       # spread evenly across the deployment. The coefficients
                       # are then applied to EVERY file in a streaming second
                       # pass. Holding a full deployment at once is several GB.

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
  order: "auto"        # "auto" picks the order by HELD-OUT error, splitting on
                       # temperature (fit the warm half, predict the cold half).
                       # In-sample fit always improves with order, so it can
                       # only ever say "more"; what matters is extrapolation
                       # onto profiles outside the fitted range. On osu685
                       # (24 degC) order 2 halves the in-sample residual AND
                       # improves extrapolation 2-5x, while order 3 gains
                       # 0.013 mK in sample and makes extrapolation 4x WORSE.
                       # Set an integer to force it.
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
    return paths, ref


def _load_some(paths, limit: int):
    """Load up to *limit* files, spread evenly across the deployment.

    Every ``.p`` load is guarded.  A real deployment is full of startup and
    surface fragments that carry a config record but no data --- osu685 had 429
    of them among 1226 files --- and an unguarded list comprehension turns the
    first one into a crash before any science happens.
    """
    step = max(1, len(paths) // max(1, limit))
    out, skipped = [], 0
    for path in paths[::step][:limit]:
        try:
            probe = load_probe_series(path)
        except Exception as exc:
            skipped += 1
            print(f"    skip {Path(path).name}: {type(exc).__name__}: {exc}",
                  file=sys.stderr)
            continue
        if probe.profiles:
            out.append(probe)
        else:
            skipped += 1
    return out, skipped


def _stream(paths):
    """Yield every loadable probe with a detected profile, one at a time.

    Streaming matters: holding 1225 files of 200k slow samples at once is
    several GB against a pair set of a few MB.  The fit needs a subsample; the
    per-profile statistics need every file but only one at a time.
    """
    for path in paths:
        try:
            probe = load_probe_series(path)
        except Exception:
            continue
        if probe.profiles:
            yield probe


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
    """Lag -> pairs -> fit -> stability -> report, for each requested channel.

    *probes* is an in-memory list (the fit subsample).  See :func:`_cmd_fit` for
    the streaming pass that follows it.
    """
    pc = _pair_config(cfg)
    lag_cfg = cfg.get("lag", {}) or {}
    fit_cfg = cfg.get("fit", {}) or {}
    st_cfg = cfg.get("stability", {}) or {}
    channels = cfg.get("channels") or sorted({c for p in probes for c in p.counts})

    out_dir.mkdir(parents=True, exist_ok=True)
    # A coefficient set is meaningless without the L-definition it was fitted
    # against, so the bridge constants travel with it and the patch step
    # refuses on any mismatch.
    provenance = {
        "instrument_sn": probes[0].instrument_sn if probes else "?",
        "n_fit_files": len(probes),
        "time_start": float(min(p.time[0] for p in probes)) if probes else None,
        "time_end": float(max(p.time[-1] for p in probes)) if probes else None,
        "reference": ref.source,
    }
    po = pressure_offset(probes, ref)
    clock = (po.lag, po.score)
    print(f"  {po.summary()}")
    t1t2 = t1_t2_series(probes)
    results: dict = {
        "schema": "fp07-cal/1",
        "clock_offset_s": clock[0],
        "clock_offset_r": clock[1],
        **provenance,
        "channels": {},
    }

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

        order_cfg = fit_cfg.get("order", "auto")
        order_scores: dict = {}
        if order_cfg in (None, "auto"):
            order, order_scores = select_order(pairs)
            print(f"  {ch} order: {order} chosen by held-out error — " + ", ".join(
                f"order {o}: {v['held_out_K']*1e3:.1f} mK held-out "
                f"({v['in_sample_K']*1e3:.1f} in-sample)"
                for o, v in sorted(order_scores.items())
            ))
        else:
            order = int(order_cfg)

        geo = None
        if fit_cfg.get("geometry", True):
            from odas_tpw.fp07cal.geometry import joint_fit

            fit, geo = joint_fit(
                pairs, order=order, robust=bool(fit_cfg.get("robust", True)),
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
                pairs, order=order, robust=bool(fit_cfg.get("robust", True)),
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
            "order_selection": {str(k): v for k, v in order_scores.items()},
            "coefficients": fit.coeffs.tolist(),
            "config_equivalent": fit.config_equivalent,
            "rms_K": fit.rms_K,
            "dive_climb_split_K": fit.dive_climb_split_K,
            "beta1_bracket": list(fit.beta1_bracket),
            "condition": fit.condition,
            "T_range": list(fit.T_range),
            "bridge": next(
                (p.bridge[ch].as_dict() for p in probes if ch in p.bridge), None
            ),
            "beta_key": next(
                (p.beta_key.get(ch) for p in probes if ch in p.beta_key), "beta_1"
            ),
            "probe_sn": next(
                (p.probe_sn.get(ch) for p in probes if ch in p.probe_sn), "?"
            ),
            "factory": next(
                (p.factory[ch].tolist() for p in probes if ch in p.factory), None
            ),
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
    paths, ref = _gather(cfg)
    limit = int(cfg.get("files", {}).get("max_fit_files", 100) or 100)
    probes, skipped = _load_some(paths, limit)
    print(f"{len(paths)} .p file(s); sampled {len(probes)} with profiles "
          f"({skipped} skipped)")
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
    paths, ref = _gather(cfg)
    out_dir = Path(cfg.get("files", {}).get("output_dir", "fp07cal"))
    limit = int(cfg.get("files", {}).get("max_fit_files", 100) or 100)
    print(f"{len(paths)} .p file(s), {ref.time.size} reference samples")
    print(f"loading up to {limit} for the fit...")
    probes, skipped = _load_some(paths, limit)
    if not probes:
        print("no file yielded a detected profile — check profiles/W_min and the "
              "reference time range", file=sys.stderr)
        return 1
    print(f"fitting on {len(probes)} file(s) ({skipped} skipped)")
    res = run_calibration(probes, ref, cfg, out_dir, make_figure=not args.no_figure)

    if not args.no_stream and len(paths) > len(probes):
        print(f"streaming all {len(paths)} files for per-profile statistics...")
        res["per_profile"] = _stream_stats(paths, ref, cfg, res, out_dir)
        (out_dir / "coefficients.json").write_text(
            json.dumps(res, indent=2, default=float)
        )
    print(f"wrote {out_dir}/")
    return 0


def _stream_stats(paths, ref, cfg: dict, res: dict, out_dir: Path) -> dict:
    """Second pass: apply the fitted coefficients to EVERY file.

    Keeps only per-profile summaries.  The profile is the independent unit for
    every statistic downstream, so collapsing to it loses nothing and takes the
    memory from gigabytes to kilobytes.
    """
    from odas_tpw.fp07cal.logr import temperature
    from odas_tpw.fp07cal.pairs import build_pairs
    from odas_tpw.fp07cal.stability import SECONDS_PER_DAY, Block

    pc = _pair_config(cfg)
    st_cfg = cfg.get("stability", {}) or {}
    model = {
        ch: (np.asarray(e["coefficients"], dtype=float), float(e["lag_s"]))
        for ch, e in res["channels"].items()
        if "coefficients" in e
    }
    if not model:
        return {}

    recs: dict[str, list] = {ch: [] for ch in model}
    n_files = 0
    for probe in _stream(paths):
        n_files += 1
        for ch, (coeffs, lag) in model.items():
            if ch not in probe.counts:
                continue
            ps = build_pairs(probe, ref, ch, lag=lag, cfg=pc)
            if len(ps) < 100:
                continue
            y = 1.0 / (ps.T_ref + 273.15)
            hi = np.zeros_like(ps.L)
            for j, a in enumerate(coeffs):
                if j:
                    hi = hi + a * ps.L**j
            a0 = y - hi
            resid = ps.T_ref - temperature(ps.L, coeffs)
            uid = np.asarray(ps.profile_uid, dtype=object)
            for u in np.unique(uid):
                sel = uid == u
                good = np.isfinite(resid[sel])
                if good.sum() < 100:
                    continue
                recs[ch].append((float(np.mean(ps.time[sel])),
                                 float(np.mean(a0[sel][good])),
                                 float(np.mean(resid[sel][good])),
                                 float(np.median(ps.T_ref[sel]))))

    out: dict = {"n_files_streamed": n_files, "channels": {}}
    for ch, R in recs.items():
        if len(R) < 12:
            out["channels"][ch] = {"n_profiles": len(R), "note": "too few to block"}
            continue
        arr = np.array(R)
        o = np.argsort(arr[:, 0])
        t, a0, resid, Tm = arr[o, 0], arr[o, 1], arr[o, 2], arr[o, 3]
        nb = int(st_cfg.get("n_blocks", 12) or 12)
        edges = np.linspace(t.min(), t.max(), nb + 1)
        TK = float(np.median(Tm)) + 273.15
        blocks = []
        for b in range(nb):
            sel = (t >= edges[b]) & (t <= edges[b + 1] if b == nb - 1 else t < edges[b + 1])
            if sel.sum() < int(st_cfg.get("min_profiles", 3)):
                continue
            v = a0[sel]
            mean = float(np.mean(v))
            se = float(np.std(v, ddof=1) / np.sqrt(v.size)) if v.size > 1 else float("nan")
            blocks.append(Block(
                t_mid=float(np.mean(t[sel])), t_start=float(edges[b]),
                t_end=float(edges[b + 1]), n_pairs=0, n_profiles=int(v.size),
                a0=mean, a0_se=se, t_0=1.0 / mean if mean else float("nan"),
                dT_K=float(-(mean - np.mean(a0)) * TK**2),
                dT_se_K=float(se * TK**2),
            ))
        stab = drift_fit(blocks)
        stab.channel = ch
        print(f"  {stab.summary()}")
        out["channels"][ch] = {
            "n_profiles": len(R),
            "span_days": float((t.max() - t.min()) / SECONDS_PER_DAY),
            "resid_median_K": float(np.median(resid)),
            "resid_sd_K": float(np.std(resid)),
            "probe_drift_K_per_day": stab.probe_drift_K_per_day,
            "se_K_per_day": stab.drift_se_K_per_day,
            "permutation_p": stab.permutation_p,
            "significant": stab.significant,
            "n_blocks": len(stab.blocks),
        }
        np.savez_compressed(out_dir / f"{ch}_profiles.npz",
                            t=t, a0=a0, resid=resid, T=Tm)
    return out


def _cmd_patch(args) -> int:
    """Write the fitted coefficients into copies of the .p files."""
    from odas_tpw.fp07cal.patch import patch_deployment

    cfg = _load_config(Path(args.config))
    paths, _ref = _gather(cfg)
    files_cfg = cfg.get("files", {})
    out_root = Path(files_cfg.get("output_dir", "fp07cal"))
    record = Path(args.record) if args.record else out_root / "coefficients.json"
    if not record.exists():
        print(f"{record} not found — run `fp07-cal fit` first", file=sys.stderr)
        return 1
    dst = Path(args.output) if args.output else out_root / "patched"

    try:
        plan, results = patch_deployment(
            record, paths, dst,
            channels=cfg.get("channels"),
            note=args.note, dry_run=args.dry_run,
        )
    except ValueError as exc:
        print(f"refusing to patch: {exc}", file=sys.stderr)
        return 1

    for w in plan.warnings:
        print(f"  WARNING {w}", file=sys.stderr)
    for e in plan.errors:
        print(f"  ERROR   {e}", file=sys.stderr)
    if not plan.ok:
        print("no edits applied", file=sys.stderr)
        return 1

    print("edits per channel:")
    for ch, kv in plan.edits.items():
        print(f"  {ch}: " + "  ".join(f"{k}={v}" for k, v in kv.items()))
    written = [d for _s, d, _c in results if d is not None]
    print(f"{'would write' if args.dry_run else 'wrote'} {len(written)} "
          f"patched file(s) to {dst}/")
    if not args.dry_run and written:
        print("Now run perturb over the PATCHED files with `fp07.calibrate: false`.")
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
    q.add_argument("--no-stream", action="store_true",
                   help="skip the all-files per-profile pass (fit subsample only)")
    q.set_defaults(func=_cmd_fit)

    q = sub.add_parser(
        "patch",
        help="write fitted coefficients into copies of the .p files "
             "(the pre-pipeline sink)",
    )
    q.add_argument("-c", "--config", required=True)
    q.add_argument("--record", help="coefficients.json (default: output_dir/)")
    q.add_argument("-o", "--output", help="destination dir (default: output_dir/patched)")
    q.add_argument("--note", default="", help="note recorded in the provenance banner")
    q.add_argument("--dry-run", action="store_true")
    q.set_defaults(func=_cmd_patch)

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

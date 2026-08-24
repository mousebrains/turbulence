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
  # Read STRAIGHT from the hotel NetCDF, on the CTD's own clock -- real
  # samples only, never through perturb's hotel merge. How that merge treats
  # gaps is governed by perturb's hotel.max_gap / extrapolate settings (PR
  # #150); reading the file directly here guarantees the fit sees only what
  # the CTD actually measured, whatever those settings say.
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
  kernel_tau: 0.7      # [s] single pole modelling the CTD thermistor's own
                       # response. NOT the measured temperature-vs-pressure
                       # delay (~2.7 s on osu685): that is dominated by
                       # plumbing transit, a pure delay the lag search
                       # already removes.
                       #
                       # A pole mismatch is NOT a delay -- the lag search
                       # cannot remove it, and it leaks a few mK into a
                       # one-signed (climb-only) deployment.
                       #
                       # This is a property of the CTD MODEL, not of the unit
                       # or the deployment: 0.7 s for a Seabird GPCTD. Look it
                       # up; do not refit it per mission. What does need
                       # checking is WHICH CTD the glider carries -- TWR's
                       # masterdata names SBE41CP for both the unpumped CTDs
                       # and the GPCTD, and an unpumped CTD is slower and
                       # flow-dependent.
  min_speed: 0.05      # [m/s] below this the thermistor is not flushing
  require_profile: true

channels: ["T1", "T2"]

profiles:
  speed_var: "U_EM"    # channel used for the flushing gate; null to disable
  W_min: 0.05          # [dbar/s] minimum |fall rate| — the GLIDER value; the
                       # VMP-tuned 0.3 rejects every glide
  P_min: 0.5           # [dbar] minimum pressure for a profile sample
  min_duration: 60.0   # [s] shortest stretch counted as a profile

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


def _gather_paths(cfg: dict) -> list[Path]:
    """The input ``.p`` set, with anything under ``output_dir`` excluded.

    ``patch`` writes ``output_dir/patched/*.p`` INSIDE the default config
    layout, so a recursive glob would sweep the tool's own output back in as
    input on the next run --- double-counting files in ``coverage`` and making
    ``patch`` refuse its own products.
    """
    files_cfg = cfg.get("files", {}) or {}
    root = Path(files_cfg.get("p_file_root", "."))
    pattern = files_cfg.get("p_file_pattern", "**/*.p")
    out_dir = Path(files_cfg.get("output_dir", "fp07cal")).resolve()
    # glob_paths, not root.glob: `**` in pathlib does not traverse a
    # symlinked directory, which is how a deployment kept on an external
    # volume is normally wired in.
    from odas_tpw.perturb.discover import glob_paths

    paths = sorted(glob_paths(root, pattern))
    inside = [
        q for q in paths
        if out_dir == q.resolve() or out_dir in q.resolve().parents
    ]
    if inside:
        print(
            f"  NOTE ignoring {len(inside)} .p file(s) under the output_dir "
            f"{out_dir} (e.g. {inside[0].name}) — the tool's own output is "
            f"never an input",
            file=sys.stderr,
        )
        skip = {q.resolve() for q in inside}
        paths = [q for q in paths if q.resolve() not in skip]
    if not paths:
        raise SystemExit(f"no .p files matched {root}/{pattern}")
    return paths


def _load_reference(cfg: dict):
    """Load the CTD reference, failing with a message rather than a traceback."""
    ref_cfg = cfg.get("reference")
    if not isinstance(ref_cfg, dict) or not ref_cfg.get("file"):
        # perturb's config uses the same `files.p_file_root` /
        # `p_file_pattern` keys, so pointing fp07-cal at one gets far enough
        # to look plausible and then fails on a missing section. Name it.
        perturb_only = {"epsilon", "chi", "binning", "hotel", "profiles", "speed"} & set(cfg)
        if perturb_only:
            raise ValueError(
                "this looks like a perturb config, not an fp07-cal one (it has "
                f"{sorted(perturb_only)} and no `reference:`). fp07-cal needs its "
                "own file: `fp07-cal init -o fp07-cal.yaml`, or copy "
                "examples/slocum_glider_hotel/fp07-cal.yaml. Point its "
                "`reference.file` at the hotel file and its `files.p_file_root` "
                "at the TRIMMED .p files."
            )
        raise ValueError(
            "config has no `reference:` block with a `file:` entry — "
            "`fp07-cal init` writes a commented template"
        )
    ref_path = Path(ref_cfg["file"])
    if not ref_path.exists():
        raise ValueError(f"reference file {ref_path} does not exist")
    return load_hotel_reference(
        ref_path,
        time_var=ref_cfg.get("time_var", "sci_ctd41cp_timestamp"),
        value_var=ref_cfg.get("value_var", "sci_water_temp"),
        pressure_var=ref_cfg.get("pressure_var", "sci_water_pressure"),
        pressure_scale=float(ref_cfg.get("pressure_scale", 1.0)),
        valid_min=float(ref_cfg.get("valid_min", -5.0)),
        valid_max=float(ref_cfg.get("valid_max", 45.0)),
    )


def _profile_kwargs(cfg: dict) -> dict:
    """The `profiles:` block -> load_probe_series keyword arguments (P7)."""
    prof = cfg.get("profiles", {}) or {}
    return {
        "speed_var": prof.get("speed_var", "U_EM"),
        "W_min": float(prof.get("W_min", 0.05)),
        "P_min": float(prof.get("P_min", 0.5)),
        "min_duration": float(prof.get("min_duration", 60.0)),
    }


def _load_some(paths, limit: int, prof_kw: dict | None = None):
    """Load up to *limit* files, spread evenly across the whole deployment.

    Every ``.p`` load is guarded.  A real deployment is full of startup and
    surface fragments that carry a config record but no data --- osu685 had 429
    of them among 1226 files --- and an unguarded list comprehension turns the
    first one into a crash before any science happens.
    """
    if len(paths) <= limit:
        chosen = list(paths)
    else:
        # np.linspace over the INDEX so the sample covers the whole list;
        # paths[::step][:limit] truncates and never sees the deployment's tail.
        idx = np.unique(np.linspace(0, len(paths) - 1, max(1, limit)).round().astype(int))
        chosen = [paths[i] for i in idx]
    out, skipped = [], 0
    for path in chosen:
        try:
            probe = load_probe_series(path, **(prof_kw or {}))
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


def _stream(paths, prof_kw: dict | None = None, failures: list | None = None):
    """Yield every loadable probe with a detected profile, one at a time.

    Streaming matters: holding 1225 files of 200k slow samples at once is
    several GB against a pair set of a few MB.  The fit needs a subsample; the
    per-profile statistics need every file but only one at a time.

    Skips are never silent: each is appended to *failures* as
    ``(name, reason)`` for the caller to report and count.
    """
    for path in paths:
        try:
            probe = load_probe_series(path, **(prof_kw or {}))
        except Exception as exc:
            if failures is not None:
                failures.append((Path(path).name, f"{type(exc).__name__}: {exc}"))
            continue
        if probe.profiles:
            yield probe
        elif failures is not None:
            failures.append((Path(path).name, "no profiles detected"))


def _pair_config(cfg: dict) -> PairConfig:
    p = cfg.get("pairs", {}) or {}
    return PairConfig(
        max_gap=float(p.get("max_gap", 30.0)),
        kernel_width=p.get("kernel_width"),
        kernel_tau=float(p.get("kernel_tau", 0.7)),
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
    # A serial shared by two channels is a placeholder, not an identity
    # (sensor_inventory.py records `sn = T` on both T1 and T2 as common), so a
    # coefficient record must not key on it.
    sn_counts: dict[str, int] = {}
    for probe in probes:
        for _ch, sn in probe.probe_sn.items():
            sn_counts[sn] = sn_counts.get(sn, 0) + 1
    n_files = max(1, len(probes))

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
            order, order_scores = select_order(
                pairs, use_geometry=bool(fit_cfg.get("geometry", True)),
                robust=bool(fit_cfg.get("robust", True)),
            )
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
            # lag - clock_offset is the CTD's own response ONLY when the
            # clock offset is itself a resolved peak; a boundary/flat
            # pressure_offset would otherwise be laundered into a "sensor
            # response" number (N4).
            "thermal_lag_s": (
                (lag - clock[0])
                if np.isfinite(clock[0]) and po.trustworthy()
                else None
            ),
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
            "probe_sn_trusted": bool(
                (lambda sn: sn not in ("?", "", "(no SN)")
                 and sn_counts.get(sn, 0) <= n_files)(
                    next((p.probe_sn.get(ch) for p in probes if ch in p.probe_sn), "?")
                )
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


def _overlap_warning(probes, ref) -> str:
    """Do the .p files and the reference cover the same time at all?

    A reference paired with the wrong deployment produces zero pairs, and
    every stage downstream reports that as "sparse coverage" rather than as
    the mistake it is. Caught in the wild: an osu685 MicroRider (Jan-Apr 2025)
    pointed at a hotel file built from ru33 (Oct 2021). `coverage` cheerfully
    reported a 42.5% duty cycle -- which was true of the reference on its own,
    and irrelevant.

    Returns "" when they overlap, or a multi-line warning naming both spans.
    """
    import datetime as dt

    import numpy as np

    starts, ends = [], []
    for probe in probes:
        t = getattr(probe, "time", None)
        if t is None or not len(t):
            continue
        finite = np.asarray(t, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        if finite.size:
            starts.append(float(finite.min()))
            ends.append(float(finite.max()))
    if not starts or ref.time.size == 0:
        return ""
    p_lo, p_hi = min(starts), max(ends)
    r_lo, r_hi = float(ref.time[0]), float(ref.time[-1])
    if p_lo <= r_hi and r_lo <= p_hi:
        return ""

    def iso(x: float) -> str:
        return dt.datetime.fromtimestamp(x, dt.UTC).isoformat(timespec="seconds")

    gap_days = (r_lo - p_hi if r_lo > p_hi else p_lo - r_hi) / 86400.0
    return (
        "WARNING: the .p files and the reference do not overlap in time.\n"
        f"  .p files : {iso(p_lo)} .. {iso(p_hi)}\n"
        f"  reference: {iso(r_lo)} .. {iso(r_hi)}\n"
        f"  they are {gap_days:.0f} days apart, so the fit will find zero pairs.\n"
        "  Check that reference.file is the hotel file for THIS deployment."
    )


def _cmd_coverage(args) -> int:
    cfg = _load_config(Path(args.config))
    paths = _gather_paths(cfg)
    try:
        ref = _load_reference(cfg)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    limit = int(cfg.get("files", {}).get("max_fit_files", 100) or 100)
    probes, skipped = _load_some(paths, limit, _profile_kwargs(cfg))
    print(f"{len(paths)} .p file(s); sampled {len(probes)} with profiles "
          f"({skipped} skipped)")
    pc = _pair_config(cfg)
    channels = cfg.get("channels") or sorted({c for p in probes for c in p.counts})
    ch = next((c for c in channels if any(c in p.counts for p in probes)), None)
    per_file = {}
    if ch:
        pairs = build_pairs_multi(probes, ref, ch, lag=0.0, cfg=pc)
        per_file = pairs.per_file
    overlap = _overlap_warning(probes, ref)
    if overlap:
        print(overlap, file=sys.stderr)
    text = coverage_text(ref.coverage_report(pc.max_gap), per_file, pc.max_gap)
    if overlap:
        text = f"> **{overlap.splitlines()[0]}**\n>\n> " + "\n> ".join(
            overlap.splitlines()[1:]
        ) + "\n\n" + text
    out_dir = Path(cfg.get("files", {}).get("output_dir", "fp07cal"))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "coverage.md").write_text(text)
    print(text)
    return 0


def _cmd_fit(args) -> int:
    cfg = _load_config(Path(args.config))
    paths = _gather_paths(cfg)
    try:
        ref = _load_reference(cfg)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    out_dir = Path(cfg.get("files", {}).get("output_dir", "fp07cal"))
    limit = int(cfg.get("files", {}).get("max_fit_files", 100) or 100)
    print(f"{len(paths)} .p file(s), {ref.time.size} reference samples")
    print(f"loading up to {limit} for the fit...")
    probes, skipped = _load_some(paths, limit, _profile_kwargs(cfg))
    if not probes:
        print("no file yielded a detected profile — check profiles/W_min and the "
              "reference time range", file=sys.stderr)
        return 1
    print(f"fitting on {len(probes)} file(s) ({skipped} skipped)")
    res = run_calibration(probes, ref, cfg, out_dir, make_figure=not args.no_figure)

    if not args.no_stream and len(paths) > len(probes):
        print(f"streaming all {len(paths)} files for per-profile statistics...")
        res["per_profile"] = _stream_stats(paths, ref, cfg, res, out_dir)
        _record_validity(res)
        (out_dir / "coefficients.json").write_text(
            json.dumps(res, indent=2, default=float)
        )
    print(f"wrote {out_dir}/")
    return 0


def _record_validity(res: dict) -> None:
    """Compare what the deployment actually spans against what the fit covered.

    The temperature gate cannot be evaluated at patch time --- knowing a file's
    range means reading its data, and ``patch`` only reads the config.  The
    streaming pass already reads every file, so the comparison is made here and
    the verdict travels in the record for ``patch`` to act on.
    """
    pp = res.get("per_profile") or {}
    for ch, entry in res.get("channels", {}).items():
        seen = (pp.get("channels", {}).get(ch) or {})
        fit_lo, fit_hi = entry.get("T_range", [None, None])
        if fit_lo is None or seen.get("T_min") is None:
            continue
        below = fit_lo - seen["T_min"]
        above = seen["T_max"] - fit_hi
        entry["validity"] = {
            "T_fitted": [fit_lo, fit_hi],
            "T_seen": [seen["T_min"], seen["T_max"]],
            "extrapolated_below_K": max(0.0, float(below)),
            "extrapolated_above_K": max(0.0, float(above)),
            "n_profiles_outside": int(seen.get("n_outside", 0)),
            "n_profiles_total": int(seen.get("n_profiles", 0)),
            "time_start": res.get("time_start"),
            "time_end": res.get("time_end"),
        }


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
    failures: list[tuple[str, str]] = []
    for probe in _stream(paths, _profile_kwargs(cfg), failures):
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
                                 float(np.median(ps.T_ref[sel])),
                                 float(np.min(ps.T_ref[sel])),
                                 float(np.max(ps.T_ref[sel]))))

    if failures:
        by_reason: dict[str, int] = {}
        for _name, reason in failures:
            by_reason[reason] = by_reason.get(reason, 0) + 1
        print(f"  {len(failures)} of {len(paths)} file(s) skipped in the "
              f"streaming pass:", file=sys.stderr)
        for reason, n in sorted(by_reason.items(), key=lambda kv: -kv[1]):
            print(f"    {n} x {reason}", file=sys.stderr)
    out: dict = {
        "n_files_streamed": n_files,
        "n_files_failed": len(failures),
        "channels": {},
    }
    for ch, R in recs.items():
        if not R:
            out["channels"][ch] = {"n_profiles": 0, "note": "no profiles"}
            continue
        arr = np.array(R)
        o = np.argsort(arr[:, 0])
        t, a0, resid, Tm = arr[o, 0], arr[o, 1], arr[o, 2], arr[o, 3]
        T_lo_all, T_hi_all = arr[o, 4], arr[o, 5]
        fit_lo, fit_hi = res["channels"][ch].get("T_range", [-np.inf, np.inf])
        n_outside = int(np.sum((T_lo_all < fit_lo) | (T_hi_all > fit_hi)))

        # The temperature coverage is recorded even when there are too few
        # profiles to block, because it is what feeds the patch-time
        # extrapolation warning -- and a short deployment is exactly where the
        # fit is most likely to have missed part of the range.
        coverage = {
            "n_profiles": len(R),
            "T_min": float(np.min(T_lo_all)),
            "T_max": float(np.max(T_hi_all)),
            "n_outside": n_outside,
        }
        if len(R) < 12:
            out["channels"][ch] = {**coverage, "note": "too few to block"}
            continue
        # Same blocking policy as run_calibration: block_days wins when set,
        # else n_blocks (default 6, matching blocked_offsets).
        from odas_tpw.fp07cal.stability import _block_edges
        edges = _block_edges(t, st_cfg.get("n_blocks", 6), st_cfg.get("block_days"))
        nb = edges.size - 1
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
            **coverage,
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
    # Only the file list is needed here: patch edits configs, it never reads
    # the hotel reference — a missing hotel.nc must not block patching.
    paths = _gather_paths(cfg)
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
    except (ValueError, OSError) as exc:
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

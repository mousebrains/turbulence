# Aug-2026, Claude and Pat Welch, pat@mousebrains.com
"""``fp07-cal`` — the pre-pipeline FP07 in-situ calibration tool.

Runs before perturb, once per deployment.  ``coverage`` answers "what reference
do I actually have?", ``fit`` produces coefficients plus the diagnostics that
say whether to believe them, and ``demo`` runs the whole thing on a synthetic
deployment with known answers so the machinery can be exercised without data.
``registry`` keeps the per-bead calibration sheet
(``microstructure_sensors/fp07_calibrations.yaml``, see :mod:`.registry`).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from odas_tpw._version_info import add_version_argument, software_version
from odas_tpw.fp07cal.fit import fit_calibration, outlier_screen, select_order
from odas_tpw.fp07cal.lag import LagResult, pressure_offset, shares_clock, temperature_lag
from odas_tpw.fp07cal.pairs import SETTLE_TAUS, PairConfig, build_pairs_multi
from odas_tpw.fp07cal.patch import SCHEMA as RECORD_SCHEMA
from odas_tpw.fp07cal.prior import (
    PriorError,
    fp07_slope,
    is_ntc,
    load_sheet,
    no_prior_record,
    parse_prior_config,
    resolve_priors,
)
from odas_tpw.fp07cal.prior import choose as choose_with_prior
from odas_tpw.fp07cal.report import coverage_text, figure, fit_text
from odas_tpw.fp07cal.series import DEATH_RUN_FILES, load_hotel_reference, load_probe_series
from odas_tpw.fp07cal.stability import blocked_offsets, drift_fit, t1_t2_series

# Hard export gate [K]: the largest error the fitted calibration may make
# predicting its OWN reference over the fitted range. A working FP07 in-situ
# calibration lands near 1e-2 K rms (osu685: 4 mK), so 1 K is orders of
# magnitude clear of any healthy fit and only catches a design that is not
# identified at all -- the intercept-aliased case returned 6.5e33 K with
# nothing rejecting it (issue #180 F20). Keyed on a PHYSICAL failure rather
# than a condition number, which a healthy but badly scaled design could cross.
MAX_PREDICTION_ERROR_K = 1.0

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
  same_clock: auto     # Is the reference on the instrument's OWN clock (a VMP
                       # against its own JAC CT)? Then the clock offset is 0 by
                       # construction and the +/-40 s pressure search is
                       # skipped. auto: yes when the reference's samples are
                       # the instrument's own (time stamps to 1 us, pressure to
                       # 1 mm); true/false force it. Affects only the reported
                       # clock/thermal split of the lag, never the coefficients.
                       # With true the 0 is an assertion, not a measurement
                       # (the record's clock_source says so): never copy it
                       # into perturb's hotel.time_offset.

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
  direction: glide     # glide: both legs of a yo (a glider); down: a VMP's
                       # descents (its hauled-in ascent puts the sensors in
                       # the body's wake)

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
  # prior:               # A bead calibrated before has a prior in the FP07
  #                      # calibration sheet: its shape (beta terms) and t_0.
  #                      # With this block the fit is made two ways -- the full
  #                      # fit above, and t_0 alone with the prior's shape held
  #                      # fixed (with dz/tau when geometry is on) -- scored by
  #                      # HELD-OUT error on the same folds. The prior's shape is
  #                      # kept unless the full fit is clearly better, and always
  #                      # over a range narrower than 5 K or when no full order
  #                      # is a plausible curve for this bead. Either way the
  #                      # result is checked against the prior over the overlap
  #                      # of their ranges; beyond the prior's uncertainty is
  #                      # flagged: a possible unrecorded bead swap, reference
  #                      # problem or failing bead. Omit the block for the
  #                      # plain fit, unchanged.
  #   sheet: "<CONFIG_DIR>/microstructure_sensors/fp07_calibrations.yaml"
  #                      # REQUIRED: the sheet is in the repository, not in the
  #                      # installed package
  #   mode: auto         # auto: choose as above; offset: force the prior's
  #                      # shape; full: keep the full fit and use the prior for
  #                      # the QC only
  #   tie_fraction: 0.05 # the full fit must beat the prior's held-out error by
  #                      # more than this fraction (5%: what a shape error of the
  #                      # sheet's own +/-5 mK costs on a wide synthetic range)
  #   beads:             # per channel: auto = the serial in the .p configs
  #     T1: auto         # (refused if the files disagree or it is the
  #     T2: auto         # placeholder T), a serial such as T1592 to override,
  #                      # or none for no prior on that channel

stability:
  n_blocks: 6
  block_days: null     # set this instead of n_blocks for a fixed cadence
  min_profiles: 3
"""


def _resolve(value, config_dir: Path):
    if isinstance(value, str):
        return value.replace("<CONFIG_DIR>", str(config_dir))
    if isinstance(value, dict):  # a nested block such as fit.prior
        return {k: _resolve(v, config_dir) for k, v in value.items()}
    return value


def _load_config(path: Path) -> dict:
    from ruamel.yaml import YAML

    yaml = YAML(typ="safe")
    with open(path, encoding="utf-8") as fh:
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
    excluded = [str(x) for x in (files_cfg.get("exclude") or [])]
    out_dir = Path(files_cfg.get("output_dir", "fp07cal")).resolve()
    # glob_paths, not root.glob: `**` in pathlib does not traverse a
    # symlinked directory, which is how a deployment kept on an external
    # volume is normally wired in.
    from odas_tpw.perturb.discover import glob_paths

    paths = sorted(glob_paths(root, pattern))
    if excluded:
        # Same escape hatch as dinkum-hotel's files.exclude, and for the same
        # reason: a deployment usually has one or two files the glider never
        # finished writing, and `patch` is all-or-nothing by design.
        drop = {q for pat in excluded for q in glob_paths(root, pat)}
        if drop:
            paths = [q for q in paths if q not in drop]
            print(
                f"  files.exclude removed {len(drop)} file(s): "
                + ", ".join(sorted(q.name for q in list(drop)[:5]))
                + (" ..." if len(drop) > 5 else "")
            )
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


_DIRECTIONS = ("glide", "down", "up")


def _profile_kwargs(cfg: dict) -> dict:
    """The `profiles:` block -> load_probe_series keyword arguments (P7)."""
    prof = cfg.get("profiles", {}) or {}
    kw = {
        "speed_var": prof.get("speed_var", "U_EM"),
        "W_min": float(prof.get("W_min", 0.05)),
        "P_min": float(prof.get("P_min", 0.5)),
        "min_duration": float(prof.get("min_duration", 60.0)),
    }
    # Only when set, so a glider config reaches load_probe_series unchanged.
    if "direction" in prof:
        d = str(prof["direction"]).strip().lower()
        if d not in _DIRECTIONS:
            raise ValueError(
                f"profiles.direction: {prof['direction']!r} must be one of {list(_DIRECTIONS)}"
            )
        kw["direction"] = d
    return kw


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


def _same_clock_setting(cfg: dict) -> str:
    """``reference.same_clock`` -> ``"auto"``, ``"yes"`` or ``"no"``; anything else refused."""
    value = (cfg.get("reference") or {}).get("same_clock", "auto")
    if value is True:
        return "yes"
    if value is False:
        return "no"
    if isinstance(value, str) and value.strip().lower() == "auto":
        return "auto"
    raise ValueError(f"reference.same_clock: {value!r} must be auto, true or false")


def _clock_offset(probes, ref, cfg: dict) -> LagResult:
    """The instrument-vs-reference clock offset: searched, or zero by construction.

    ``reference.same_clock`` (default ``auto``): when the reference is on the
    instrument's own clock --- a VMP fitted against its own JAC CT --- the
    offset is zero by construction and the +/-40 s pressure search only spends
    time confirming it: 232 s of an 894 s run on SUNRISE 2022 PointSur (77
    files), to return -1.3 us.  ``auto`` decides by :func:`shares_clock`; ``true``
    asserts it (and, if wrong, misstates only the clock/thermal split of the
    lag, never the calibration, which the temperature lag search handles
    whole); ``false`` always searches.

    The record's ``clock_source`` says which: an asserted zero was never
    measured, so it must not be copied into perturb's ``hotel.time_offset``
    the way a measured ``clock_offset_s`` is.
    """
    setting = _same_clock_setting(cfg)
    if setting == "yes":
        label = ("clock offset 0 ASSERTED by reference.same_clock: true, not measured: do "
                 "not copy clock_offset_s into perturb's hotel.time_offset")
    elif setting == "auto" and shares_clock(probes, ref):
        label = ("clock offset 0 by construction, not searched (detected: the reference's "
                 "samples are the instrument's own, in time and pressure)")
    else:
        return pressure_offset(probes, ref)
    return LagResult(lag=0.0, exact=True, label=label)


def _gap_summary(pairs) -> dict:
    """What the thermistor gaps cost, for the record."""
    g = pairs.gaps
    return {
        "n_files": len(g),
        "n_gaps": int(sum(v.get("n_gaps", 0) for v in g.values())),
        "n_nonfinite_samples": int(sum(v.get("n_nonfinite", 0) for v in g.values())),
        "n_settling_samples": int(sum(v.get("n_settling", 0) for v in g.values())),
        "n_pairs_lost": int(pairs.rejected.get("thermistor_gap", 0)),
        "files": sorted(g)[:20],
    }


def _gap_note(s: dict) -> str:
    return (
        f"{s['n_nonfinite_samples']} non-finite thermistor sample(s) in {s['n_gaps']} "
        f"gap(s) across {s['n_files']} file(s). The single pole that slows the "
        f"thermistor to the reference restarted after each gap, with the first "
        f"{SETTLE_TAUS:g} time constants ({s['n_settling_samples']} samples) blanked "
        f"while it settled; {s['n_pairs_lost']} pair(s) were lost to them "
        f"(`thermistor_gap`)."
    )


def _iso(t: float) -> str:
    from datetime import UTC, datetime

    return datetime.fromtimestamp(t, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _channel_span(probes, pairs, fit, ch: str) -> dict:
    """When this channel's bead was ALIVE, over the files in memory (issue #249).

    The time a bead was in service, not the files that happened to give it a
    calibration pair (the PI's decision, 2026-10-29): ``time_start``/``time_end``
    run from the first sample of the first file in which its thermistor recorded
    live data (:func:`~odas_tpw.fp07cal.series.thermistor_alive`) to the last
    sample of the last, ``n_files`` counts those files and ``n_files_dead`` the
    others.  A bead that died before the run ended gets its own end: SUNRISE
    2021 Pelican's T1593, on the ADC rail through SR1P_516, ends with SR1P_515;
    one that died and then passes now and then ends at its death
    (:func:`~odas_tpw.fp07cal.series.alive_span`, ``first_dead``).
    ``first_kept_pair``/``last_kept_pair`` are the extent of the pairs the fit
    kept, on the reference's clock: the calibration's support.

    ``from``: ``fit files``.  ``fp07-cal fit`` replaces the file span with the
    streaming pass's over every file of the run (:func:`_cmd_fit`): the fit holds
    a subsample, whose stride can end a dying bead a file early.
    """
    from odas_tpw.fp07cal.series import alive_span, bead_alive

    held = [p for p in probes if ch in p.counts]
    kept = fit.kept if fit.kept.size == len(pairs) else np.ones(len(pairs), dtype=bool)
    span = alive_span([(float(p.time[0]), float(p.time[-1]), bead_alive(p, ch)) for p in held])
    if span is None:  # no live file judged, yet pairs kept: their files
        used = set(np.asarray(pairs.file_label, dtype=object)[kept].tolist())
        span = alive_span([(float(p.time[0]), float(p.time[-1]), p.label in used)
                           for p in held])
    t = np.asarray(pairs.time, dtype=np.float64)[kept]
    return {
        **(span or {}),
        "from": "fit files",
        "first_kept_pair": float(t.min()),
        "last_kept_pair": float(t.max()),
    }


def _death_note(ch: str, span: dict) -> str | None:
    """The NOTE for files that still passed the alive rule after the bead died."""
    if not span.get("n_files_alive_after_death"):
        return None
    return (f"{ch}: the bead died at {_iso(span['first_dead'])} (the first of "
            f"{DEATH_RUN_FILES} consecutive dead files), yet {span['n_files_alive_after_death']} "
            f"of the {span['n_files_after_death']} file(s) from then on still pass the alive "
            f"rule: its span ends at {_iso(span['time_end'])}. Check those files: a bead "
            f"failing intermittently, or one replaced without a config update.")


def _bad_buffer_note(s: dict) -> str:
    whole = ""
    if s.get("n_files_dropped"):
        whole = (
            f" In {s['n_files_dropped']} of those file(s) an RDL marker run fell on this "
            f"thermistor or its pre-emphasised partner, which PFile's deconvolution smears "
            f"over the whole file (issue #259), so the channel was left out of each such "
            f"file whole: {', '.join(s['files_dropped'])}."
        )
    return (
        f"{s['n_samples']} thermistor count(s) in {s['n_files']} file(s) were set to NaN, "
        f"not paired: samples PFile reports as RDL bad buffers or as record slots a framing "
        f"repair rebuilt (PFile.bad_buffer_report), with the deconvolution's local smear "
        f"after them, as chi masks them (issue #248).{whole} The pairs they cost are "
        f"counted under `thermistor_gap`. A v6.0 file's records flagged in header word 16 "
        f"are not in that report and are still paired."
    )


def _outliers_beyond_gate(pairs, pred_err: np.ndarray, outlier: np.ndarray) -> dict | None:
    """The pairs the robust fit rejected that the export gate would have refused on."""
    big = outlier & np.isfinite(pred_err) & (pred_err > MAX_PREDICTION_ERROR_K)
    if not big.any():
        return None
    k = int(np.flatnonzero(big)[np.argmax(pred_err[big])])
    return {
        "n": int(big.sum()),
        "n_rejected": int(outlier.sum()),
        "max_error_K": float(pred_err[k]),
        "worst_time": float(pairs.time[k]),
        "worst_pressure_dbar": float(pairs.pressure[k]),
        "worst_file": str(pairs.file_label[k]),
    }


def _outlier_warning(b: dict) -> str:
    from datetime import UTC, datetime

    when = datetime.fromtimestamp(b["worst_time"], tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    return (
        f"{b['n']} of the {b['n_rejected']} pairs the robust fit rejected mispredict the "
        f"reference by more than {MAX_PREDICTION_ERROR_K:g} K, worst {b['max_error_K']:.3g} K "
        f"at {when}, {b['worst_pressure_dbar']:.1f} dbar ({b['worst_file']}). They are "
        f"left out of the fit and of the export gate; check the reference there (a spike or "
        f"an instrument-wide transient)"
    )


# When the robust loop rejects a second POPULATION rather than a tail (issue
# #246). Judged about the median, it takes any minority offset by more than
# 4 sigma out whole, however small the offset: 40% of a reference 50 mK off is
# left out of the fit AND of the stability blocks, where before it was averaged
# in (rms 24 mK) and its step showed in the block table. Measured on 11 real
# deployments (7 VMP, 4 glider; 30 fitted channels with their variants):
# 0.1-6.9% of the pairs rejected, and the profiles that lost more than half
# their pairs (0-7 of them) held at most 12.7% of the rejections. A synthetic
# minority of 5-40% put 83-100% of the rejections in such profiles.
MAX_REJECTED_FRACTION = 0.15
"""Warn when the robust fit rejects more than this fraction of the pairs it judged."""
WHOLE_PROFILE_SHARE = 0.5
"""... or when profiles that lost more than half their pairs hold at least this share."""
WHOLE_PROFILE_MIN_FRACTION = 0.005
"""... and at least this fraction of every pair judged: one rejected pair in a one-pair
profile is not a population. (Real data: at most 0.43%, in one profile.)"""


def _robust_rejection(pairs, fit) -> dict:
    """What the robust fit rejected, and how much of it came as whole profiles."""
    outlier = (fit.outlier if fit.outlier.size == len(pairs)
               else np.zeros(len(pairs), dtype=bool))
    kept = fit.kept if fit.kept.size == len(pairs) else ~outlier
    # Every pair the loop judged: the joint fit judges pairs it does not fit.
    judged = kept | outlier | (
        np.isfinite(np.asarray(pairs.L, dtype=np.float64))
        & np.isfinite(np.asarray(pairs.T_ref, dtype=np.float64))
        & np.isfinite(np.asarray(pairs.w, dtype=np.float64))
    )
    uid = np.asarray(pairs.profile_uid, dtype=object)
    u, inv = np.unique(uid[judged], return_inverse=True)
    n_prof = np.bincount(inv, minlength=u.size)
    n_rej = np.bincount(inv, weights=outlier[judged].astype(np.float64), minlength=u.size)
    mostly = n_rej > 0.5 * n_prof
    whole = np.zeros(len(pairs), dtype=bool)
    whole[np.flatnonzero(judged)[mostly[inv]]] = True
    files: dict[str, int] = {}
    for f in np.asarray(pairs.file_label, dtype=object)[outlier]:
        files[str(f)] = files.get(str(f), 0) + 1
    n = int(outlier.sum())
    t = np.asarray(pairs.time, dtype=np.float64)
    return {
        "n_rejected": n,
        "n_judged": int(judged.sum()),
        "fraction": float(n / max(1, int(judged.sum()))),
        "n_profiles": int(u.size),
        "n_profiles_mostly_rejected": int(mostly.sum()),
        "share_in_those_profiles": float(n_rej[mostly].sum() / n) if n else 0.0,
        "whole_profile_fraction": float(n_rej[mostly].sum() / max(1, int(judged.sum()))),
        "first": float(t[whole].min()) if whole.any() else None,
        "last": float(t[whole].max()) if whole.any() else None,
        "files": dict(sorted(files.items(), key=lambda kv: -kv[1])[:5]),
    }


def _population_warning(rr: dict) -> str | None:
    """The WARNING for a rejected population, or None for an ordinary tail."""
    if not (rr["fraction"] > MAX_REJECTED_FRACTION
            or (rr["n_profiles_mostly_rejected"]
                and rr["share_in_those_profiles"] >= WHOLE_PROFILE_SHARE
                and rr["whole_profile_fraction"] >= WHOLE_PROFILE_MIN_FRACTION)):
        return None
    from datetime import UTC, datetime

    def iso(x: float) -> str:
        return datetime.fromtimestamp(x, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    where = ""
    if rr["first"] is not None:
        where = (f", {rr['n_profiles_mostly_rejected']} of {rr['n_profiles']} profiles "
                 f"lost more than half their pairs ({100 * rr['share_in_those_profiles']:.0f}% "
                 f"of the rejections, {iso(rr['first'])} .. {iso(rr['last'])})")
    return (
        f"the robust fit rejected {rr['n_rejected']} of {rr['n_judged']} pairs "
        f"({100 * rr['fraction']:.1f}%){where}, most in "
        + ", ".join(f"{k} ({v})" for k, v in rr["files"].items())
        + ". That is a second population, not a tail: a reference or bead episode that "
        "is left out of the fit and of the stability blocks, so neither shows it. "
        "Check those pairs before trusting either"
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
    # fit.prior: resolved for every channel BEFORE any fitting, so an ambiguous
    # bead refuses the run at once rather than after the first channel.
    pcfg = parse_prior_config(fit_cfg, list(channels))
    priors = {}
    if pcfg is not None:
        span = (
            (provenance["time_start"], provenance["time_end"]) if probes else None
        )
        priors = resolve_priors(pcfg, probes, list(channels), span=span)
    po = _clock_offset(probes, ref, cfg)
    clock = (po.lag, po.score)
    print(f"  {po.summary()}")
    results: dict = {
        "schema": RECORD_SCHEMA,
        "software_version": software_version(),
        "clock_offset_s": clock[0],
        "clock_offset_r": clock[1],
        **({"clock_source": po.label} if po.exact else {}),
        **provenance,
        "channels": {},
    }
    # Reports are written once every channel is fitted: the T1 - T2 check
    # needs both channels' coefficients (issue #238).
    reported: dict[str, dict] = {}

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

        notes: list[str] = []
        extra: dict = {}
        blanked = {p.label: n for p in probes
                   if (n := int((getattr(p, "bad_buffer", None) or {}).get(ch, 0)))}
        if blanked:
            whole = sorted(p.label for p in probes
                           if ch in (getattr(p, "bad_buffer_dropped", None) or []))
            extra["bad_buffer_samples"] = {
                "n_samples": sum(blanked.values()),
                "n_files": len(blanked),
                "files": sorted(blanked)[:20],
                # left out whole: a marker run smears the deconvolution file-wide
                **({"n_files_dropped": len(whole), "files_dropped": whole[:20]}
                   if whole else {}),
            }
            notes.append(_bad_buffer_note(extra["bad_buffer_samples"]))
            print(f"  {ch}: NOTE {notes[-1]}")
        if pairs.gaps:
            extra["thermistor_gaps"] = _gap_summary(pairs)
            notes.append(_gap_note(extra["thermistor_gaps"]))
            print(f"  {ch}: NOTE {notes[-1]}")
        # The geometry term needs a reference gradient, which needs profiles
        # spanning ~30 dbar. Without one, fit without the term and say so: a
        # crash here stopped the run before the prior was ever consulted.
        geometry = bool(fit_cfg.get("geometry", True))
        dTdz = None
        if geometry:
            from odas_tpw.fp07cal.geometry import geometry_unavailable, local_dTdz

            # Computed once, and handed to the joint fit below, which would
            # otherwise compute the identical array again.
            dTdz = local_dTdz(pairs)
            why_not = geometry_unavailable(pairs, dTdz)
            if why_not:
                geometry = False
                extra["geometry_skipped"] = why_not
                notes.append(
                    f"The sensor-geometry term was requested (fit.geometry) but skipped: "
                    f"{why_not}. The calibration is fitted without it, as with "
                    f"fit.geometry: false, so any dz * dT/dz is left in the residual."
                )
                print(f"  {ch}: NOTE geometry term skipped: {why_not}")

        order_cfg = fit_cfg.get("order", "auto")
        order_scores: dict = {}
        # Held-out folds leave out the pairs every candidate order rejects (a
        # reference glitch), or a 2 degC spike in 13-21 degC water sets the
        # warm/cold split and dominates the test error (issue #238 review).
        screen = None
        uses_folds = order_cfg in (None, "auto") or (
            pcfg is not None and ch in priors and priors[ch].prior is not None)
        if uses_folds and bool(fit_cfg.get("robust", True)):
            screen = outlier_screen(pairs, use_geometry=geometry, robust=True)
            if screen.any():
                print(f"  {ch}: {int(screen.sum())} pair(s) every order rejects are left out "
                      f"of the held-out folds")
        if order_cfg in (None, "auto"):
            order, order_scores = select_order(
                pairs, use_geometry=geometry,
                robust=bool(fit_cfg.get("robust", True)), exclude=screen,
            )
            print(f"  {ch} order: {order} chosen by held-out error — " + ", ".join(
                f"order {o}: {v['held_out_K']*1e3:.1f} mK held-out "
                f"({v['in_sample_K']*1e3:.1f} in-sample)"
                for o, v in sorted(order_scores.items())
            ))
        else:
            order = int(order_cfg)

        geo = None
        if geometry:
            from odas_tpw.fp07cal.geometry import joint_fit

            fit, geo = joint_fit(
                pairs, order=order, robust=bool(fit_cfg.get("robust", True)), dTdz=dTdz,
            )
            print(f"  {ch} geometry: {geo.summary()}")
            if np.isfinite(geo.collinearity) and geo.collinearity > 0.8:
                print(
                    f"      NOTE collinearity {geo.collinearity:.3f}: dz and the "
                    f"residual lag are NOT separately resolved (vertical speed "
                    f"one-signed — profiles in only one direction). Their "
                    f"combination is measured; the split is not."
                )
            if geo.rank_deficient:
                print(
                    f"      NOTE the FULL design is RANK-DEFICIENT (smallest "
                    f"singular value {geo.smallest_singular_value:.3e}): a "
                    f"geometry term is aliased with the calibration polynomial, "
                    f"so dz/tau are not identified even though their mutual "
                    f"correlation looks fine.",
                    file=sys.stderr,
                )
        else:
            fit = fit_calibration(
                pairs, order=order, robust=bool(fit_cfg.get("robust", True)),
            )
        prior_rec = None
        if pcfg is not None and ch in priors:
            cp = priors[ch]
            if cp.prior is None:
                prior_rec = no_prior_record(cp, pcfg.mode)
                print(f"  {ch} prior: none — {cp.reason}")
            else:
                chosen, geo, prior_rec = choose_with_prior(
                    pairs, cp, mode=pcfg.mode, tie_fraction=pcfg.tie_fraction,
                    full_fit=fit, full_geo=geo,
                    orders=(1, 2, 3) if order_cfg in (None, "auto") else (order,),
                    robust=bool(fit_cfg.get("robust", True)),
                    geometry=geometry, exclude=screen,
                )
                if chosen is None:
                    # mode offset and the offset-only fit failed: a channel-level
                    # refusal, recorded like any other, never a mid-run exception
                    # that would leave this run's reports beside an old record.
                    results["channels"][ch] = {"error": "offset-only fit failed",
                                               "prior": prior_rec}
                    continue
                fit = chosen
                if prior_rec["chosen_kind"] == "offset" and geo is not None:
                    print(f"  {ch} geometry (prior's shape): {geo.summary()}")
        # geo=... is load-bearing: without it the fitted dz/tau nuisance is
        # re-exposed to the blocks and a changing gradient across the deployment
        # is reported as bead drift (issue #180 F08).
        # HARD export gate: a fit that cannot reproduce its own reference over
        # the fitted range is not a calibration, whatever its condition number
        # says. Keyed on a PHYSICAL prediction failure rather than a
        # conditioning threshold, which a healthy but badly scaled design could
        # cross: a geometry column aliased with the intercept returned dz = 3448
        # +/- 0.68 m and a 6.5e33 K calibration error with nothing rejecting it
        # (issue #180 F20).
        #
        # Judged over every pair EXCEPT those the robust loop already rejected
        # as outliers (issue #238). Counting them refused SUNRISE 2022
        # PointSur, whose fit rms is 6-7 mK, over one 4.2 K reference event the
        # fit had already thrown out. An unidentified fit mispredicts the pairs
        # it kept too, so the F20 case is still refused; the outliers are
        # reported on their own instead.
        max_pred_err = float("nan")
        beyond = None
        if pairs.L.size and fit.coeffs.size:
            with np.errstate(over="ignore", invalid="ignore"):
                pred_err = np.abs(fit.apply(pairs.L) - pairs.T_ref)
            outlier = (
                fit.outlier if fit.outlier.size == len(pairs)
                else np.zeros(len(pairs), dtype=bool)
            )
            judged = np.isfinite(pred_err) & ~outlier
            max_pred_err = float(pred_err[judged].max()) if judged.any() else float("inf")
            beyond = _outliers_beyond_gate(pairs, pred_err, outlier)
        if not np.isfinite(max_pred_err) or max_pred_err > MAX_PREDICTION_ERROR_K:
            print(
                f"  {ch}: REFUSED — the fit mispredicts its own reference by up "
                f"to {max_pred_err:.3g} K over the fitted range (limit "
                f"{MAX_PREDICTION_ERROR_K:g} K). The design is not identified; "
                f"no coefficients are exported for this channel.",
                file=sys.stderr,
            )
            results["channels"][ch] = {
                "error": "unidentifiable fit",
                "max_prediction_error_K": max_pred_err,
                "condition": fit.condition,
                "rank_deficient": bool(geo.rank_deficient) if geo is not None else None,
                "smallest_singular_value": (
                    geo.smallest_singular_value if geo is not None else None
                ),
            }
            if prior_rec is not None:
                results["channels"][ch]["prior"] = prior_rec
            continue
        # A thermistor is NTC: T falls as L rises. A fit that does not, over its
        # own fitted range, is a fit to noise (water with no temperature range)
        # and predicts its pairs within the 1 K gate all the same: without a
        # prior, an isothermal synthetic exported beta_1 = -1.06e5 (issue #238).
        #
        # Nor, with the right sign, is a slope no FP07 has (issue #247): over
        # isothermal water 18 of 40 synthetic noise fits came out NTC with a
        # local beta 14-42x an FP07's. Judged against every bead in the
        # calibration sheet, so it needs no prior (prior.FP07_BETA_K).
        why = ""
        beta_span = None
        if not is_ntc(fit):
            lo, hi = fit.L_range
            why = (
                f"T does not fall as L rises over the fitted L range {lo:.5f}..{hi:.5f} "
                f"(dT/dL >= 0 somewhere, or 1/T_K <= 0), so this is not an FP07 curve: "
                f"typically water with too little temperature range, where a fit to "
                f"noise can take either sign"
            )
        else:
            slope_ok, slope_why, beta_span = fp07_slope(fit)
            if not slope_ok:
                why = slope_why
        if why:
            print(f"  {ch}: REFUSED — {why}. No coefficients are exported for this "
                  f"channel; a fit.prior block lends such a deployment the bead's shape.",
                  file=sys.stderr)
            results["channels"][ch] = {
                "error": "not an FP07 curve",
                "reason": why,
                "order": fit.order,
                "refused_coefficients": fit.coeffs.tolist(),
                "L_range": list(fit.L_range),
                "T_range": list(fit.T_range),
                **({"beta_range_K": beta_span} if beta_span is not None else {}),
            }
            if prior_rec is not None:
                results["channels"][ch]["prior"] = prior_rec
            continue
        warnings: list[str] = []
        if beyond is not None:
            extra["outliers_beyond_gate"] = beyond
            warnings.append(_outlier_warning(beyond))
            print(f"  {ch}: WARNING {_outlier_warning(beyond)}", file=sys.stderr)
        rr = _robust_rejection(pairs, fit)
        print(f"  {ch}: robust fit rejected {rr['n_rejected']} of {rr['n_judged']} pairs "
              f"({100 * rr['fraction']:.2f}%); {rr['n_profiles_mostly_rejected']} of "
              f"{rr['n_profiles']} profiles lost more than half, holding "
              f"{100 * rr['share_in_those_profiles']:.1f}% of the rejections")
        population = _population_warning(rr)
        if population is not None:
            extra["robust_rejection"] = rr
            warnings.append(population)
            print(f"  {ch}: WARNING {population}", file=sys.stderr)

        blocks = blocked_offsets(
            pairs, fit,
            n_blocks=st_cfg.get("n_blocks", 6),
            block_days=st_cfg.get("block_days"),
            min_profiles=int(st_cfg.get("min_profiles", 3)),
            geo=geo,
        )
        stab = drift_fit(blocks)
        stab.channel = ch
        own = _channel_span(probes, pairs, fit, ch)
        if own.get("n_files_dead") or own.get("first_dead") is not None:
            print(f"  {ch}: the bead was alive in {own['n_files']} of {len(probes)} file(s), "
                  f"{_iso(own['time_start'])} .. {_iso(own['time_end'])}: its span for the "
                  f"sheet, not the run's")
        reported[ch] = {"pairs": pairs, "fit": fit, "stab": stab, "prior": prior_rec,
                        "notes": notes, "warnings": warnings}

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
            "span": own,
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
                # Resolved from EACH OTHER *and* from the calibration
                # polynomial: the old predicate tested only the dz/tau
                # correlation, which an intercept-aliased design passes at
                # 4.7e-17 (issue #180 F20).
                "separately_resolved": bool(geo.separately_resolved()),
                "rank_deficient": bool(geo.rank_deficient),
                "smallest_singular_value": geo.smallest_singular_value,
                "dz_se_cluster_m": geo.dz_se_cluster_m,
                "tau_se_cluster_s": geo.tau_se_cluster_s,
                "n_profiles_clustered": geo.n_clusters,
                "cluster_warning": geo.cluster_warning,
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
        # Present only when they apply, so a record none of them touch is
        # unchanged byte for byte.
        if lr.narrow_window():
            results["channels"][ch]["lag_resolution"] = lr.resolution()
        results["channels"][ch].update(extra)
        if prior_rec is not None:
            results["channels"][ch]["prior"] = prior_rec
        print(f"  {ch}: {stab.summary()}")
        print(
            f"      t_0={fit.config_equivalent['t_0']:.4f} "
            f"beta_1={fit.config_equivalent.get('beta_1', float('nan')):.2f} "
            f"lag={lag:+.2f}s r={r:.4f} rms={fit.rms_K:.4f}K "
            f"split={fit.dive_climb_split_K:+.4f}K"
        )

    # The reference-free T1 - T2 check, with ONE coefficient set per channel:
    # this run's fit where there is one. Each file's embedded set used to be
    # taken instead, and where files had been patched one by one those differ
    # by several K at fixed L (ARCTERX-2022 Interior: 13 sets in 30 files,
    # 6.3 K apart for T1; issue #238).
    t1t2 = t1_t2_series(probes, coeffs={ch: d["fit"].coeffs for ch, d in reported.items()})
    for ch, d in reported.items():
        md = fit_text(d["pairs"], d["fit"], clock_offset=clock,
                      clock_note=po.label if po.exact else None, stab=d["stab"],
                      t1t2=t1t2, min_corr=pc.min_corr, prior=d["prior"],
                      notes=d["notes"], extra_warnings=d["warnings"])
        (out_dir / f"{ch}_report.md").write_text(md, encoding="utf-8")
        if make_figure:
            figure(d["pairs"], d["fit"], stab=d["stab"], t1t2=t1t2,
                   path=out_dir / f"{ch}_diagnostics.png")

    (out_dir / "coefficients.json").write_text(
        json.dumps(results, indent=2, default=float), encoding="utf-8"
    )
    return results


def _cmd_init(args) -> int:
    out = Path(args.output)
    if out.exists() and not args.force:
        print(f"{out} exists; use --force", file=sys.stderr)
        return 1
    out.write_text(TEMPLATE, encoding="utf-8")
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
        prof_kw = _profile_kwargs(cfg)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    limit = int(cfg.get("files", {}).get("max_fit_files", 100) or 100)
    probes, skipped = _load_some(paths, limit, prof_kw)
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
    (out_dir / "coverage.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


def _cmd_fit(args) -> int:
    cfg = _load_config(Path(args.config))
    try:
        # A bad fit.prior block fails here, before minutes of loading.
        pcfg = parse_prior_config(cfg.get("fit"), cfg.get("channels"))
        if pcfg is not None:
            load_sheet(pcfg)
        _same_clock_setting(cfg)
        _profile_kwargs(cfg)
    except ValueError as exc:  # PriorError is a ValueError
        print(f"refusing: {exc}", file=sys.stderr)
        return 1
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
    try:
        res = run_calibration(probes, ref, cfg, out_dir, make_figure=not args.no_figure)
    except PriorError as exc:
        print(f"refusing: {exc}", file=sys.stderr)
        return 1

    if not args.no_stream and len(paths) > len(probes):
        print(f"streaming all {len(paths)} files for per-profile statistics...")
        res["per_profile"] = _stream_stats(paths, ref, cfg, res, out_dir)
        _record_validity(res)
        # Each channel's alive span over every file of the run (issue #249);
        # the kept-pair extent stays the fit's own.
        for ch, s in res["per_profile"].pop("spans", {}).items():
            if "span" in res["channels"].get(ch, {}):
                res["channels"][ch]["span"].update(s)
                note = _death_note(ch, s)
                if note:
                    print(f"  NOTE {note}", file=sys.stderr)
    else:
        for e in res["channels"].values():
            if "span" in e:
                e["span"]["from"] = ("every file" if len(paths) <= limit else
                                     f"{len(probes)} fit files of {len(paths)}")
    (out_dir / "coefficients.json").write_text(
        json.dumps(res, indent=2, default=float), encoding="utf-8"
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
    from odas_tpw.fp07cal.geometry import local_dTdz
    from odas_tpw.fp07cal.logr import temperature
    from odas_tpw.fp07cal.pairs import build_pairs
    from odas_tpw.fp07cal.series import alive_span, bead_alive
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
    # The fitted nuisance geometry, per channel. The calibration was fitted
    # against T_ref - (dz*g + tau*w*g), so each profile has to be compared with
    # that same quantity: compared with the raw reference, a gradient exposure
    # that changes over the deployment shows up as a changing intercept, i.e.
    # as bead drift -- the issue #180 F08 defect, fixed in blocked_offsets but
    # not here. On osu685 (2025) it manufactured +3 mK of drift on a flat bead.
    geometry: dict[str, tuple[float, float]] = {}
    for ch, e in res["channels"].items():
        g = e.get("geometry") or {}
        dz = float(g.get("dz_m", np.nan))
        if np.isfinite(dz):
            tau = float(g.get("tau_s", np.nan))
            geometry[ch] = (dz, tau if np.isfinite(tau) else 0.0)

    recs: dict[str, list] = {ch: [] for ch in model}
    blanked: dict[str, int] = {ch: 0 for ch in model}
    dropped: dict[str, int] = {ch: 0 for ch in model}
    lives: dict[str, list] = {ch: [] for ch in model}
    n_files = 0
    failures: list[tuple[str, str]] = []
    for probe in _stream(paths, _profile_kwargs(cfg), failures):
        n_files += 1
        for ch, (coeffs, lag) in model.items():
            if ch not in probe.counts:
                continue
            blanked[ch] += int((getattr(probe, "bad_buffer", None) or {}).get(ch, 0))
            dropped[ch] += ch in (getattr(probe, "bad_buffer_dropped", None) or [])
            # When the bead was alive, over EVERY file of the run, not the fit's
            # subsample, whose stride can end a dying bead a file early (#249).
            t = getattr(probe, "time", None)
            if t is not None and len(t):
                lives[ch].append((float(t[0]), float(t[-1]), bead_alive(probe, ch)))
            ps = build_pairs(probe, ref, ch, lag=lag, cfg=pc)
            if not len(ps):
                continue
            T_cal = np.asarray(ps.T_ref, dtype=np.float64)
            if ch in geometry:
                dz, tau = geometry[ch]
                g = local_dTdz(ps)
                w = np.asarray(ps.w, dtype=np.float64)
                T_cal = T_cal - np.where(
                    np.isfinite(g) & np.isfinite(w), dz * g + tau * w * g, 0.0
                )
            y = 1.0 / (T_cal + 273.15)
            hi = np.zeros_like(ps.L)
            for j, a in enumerate(coeffs):
                if j:
                    hi = hi + a * ps.L**j
            a0 = y - hi
            resid = T_cal - temperature(ps.L, coeffs)
            # The temperature a profile covers, for the patch-time validity
            # verdict, leaves out reference glitches as the fit's own range
            # does: a pair the calibration mispredicts by more than the export
            # gate is not water the coefficients were asked to describe.
            with np.errstate(invalid="ignore"):
                cover = np.abs(resid) <= MAX_PREDICTION_ERROR_K
            if len(ps) < 100:
                continue
            uid = np.asarray(ps.profile_uid, dtype=object)
            for u in np.unique(uid):
                sel = uid == u
                good = np.isfinite(resid[sel])
                if good.sum() < 100:
                    continue
                Tc = ps.T_ref[sel & cover] if np.any(sel & cover) else ps.T_ref[sel]
                recs[ch].append((float(np.mean(ps.time[sel])),
                                 float(np.mean(a0[sel][good])),
                                 float(np.mean(resid[sel][good])),
                                 float(np.median(ps.T_ref[sel])),
                                 float(np.min(Tc)),
                                 float(np.max(Tc))))

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
        "spans": {ch: {**s, "from": "every file"}
                  for ch, s in ((c, alive_span(v)) for c, v in lives.items()) if s},
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
    for ch, n in blanked.items():  # thermistor counts set NaN as bad buffers (#248)
        if n and ch in out["channels"]:
            out["channels"][ch]["bad_buffer_samples"] = n
            if dropped[ch]:
                out["channels"][ch]["bad_buffer_files_dropped"] = dropped[ch]
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
    if args.dry_run:
        # patch_files returns dst=None for a dry run AND for a no-op, so
        # counting non-None destinations reported "would write 0" on every
        # dry run, whatever it was about to do. Count the files that actually
        # have changes instead -- which also distinguishes "nothing to do,
        # they already carry these coefficients" from "0 because dry run".
        would = [s for s, _d, changes in results if changes]
        print(f"would write {len(would)} patched file(s) to {dst}/")
        if not would and results:
            print("  (no file needs a change: they already carry these coefficients)")
    else:
        written = [d for _s, d, _c in results if d is not None]
        print(f"wrote {len(written)} patched file(s) to {dst}/")
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


def _cmd_registry_render(args) -> int:
    """Regenerate the sheet's .md and .csv views from its YAML."""
    from odas_tpw.fp07cal.registry import RegistryError, default_path, write_views

    path = Path(args.registry) if args.registry else default_path()
    try:
        changed = write_views(path)
    except RegistryError as exc:
        print(f"invalid: {exc}", file=sys.stderr)
        return 1
    for dst in changed:
        print(f"wrote {dst}")
    if not changed:
        print(f"{path.with_suffix('.md').name} and {path.with_suffix('.csv').name} "
              f"are already current")
    return 0


def _cmd_registry_check(args) -> int:
    """Validate the YAML; exit 1 if it is invalid or either view is stale."""
    from odas_tpw.fp07cal.registry import (
        RENDER_HINT,
        RegistryError,
        default_path,
        load,
        stale_views,
    )

    path = Path(args.registry) if args.registry else default_path()
    try:
        reg = load(path)
        stale = stale_views(path)
    except RegistryError as exc:
        print(f"invalid: {exc}", file=sys.stderr)
        return 1
    for dst in stale:
        what = "is missing" if not dst.exists() else f"differs from what {path.name} renders"
        print(f"stale: {dst} {what}; {RENDER_HINT}", file=sys.stderr)
    if stale:
        return 1
    print(f"ok: {len(reg.calibrations)} calibrations of {len(reg.beads())} beads in "
          f"{path}; both views current")
    return 0


def _cmd_registry_show(args) -> int:
    """One bead's calibrations and its prior for a fresh deployment."""
    from odas_tpw.fp07cal.registry import RegistryError, default_path, load

    path = Path(args.registry) if args.registry else default_path()
    try:
        reg = load(path)
        history = reg.history(args.serial)
        prior = reg.prior_for(args.serial, instrument=args.instrument,
                              channel=args.channel, as_of=args.as_of)
    except RegistryError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps({"bead": args.serial,
                          "calibrations": [c.to_dict() for c in history],
                          "prior": prior.to_dict()}, indent=2))
        return 0
    print(f"{args.serial}: {len(history)} calibration(s) in {path}")
    for i, c in enumerate(history, start=1):
        u = c.uncertainty
        block = "-" if u.block_sd_mK is None else f"{u.block_sd_mK:.1f}"
        print(f"  {i}. {c.deployment} | {c.instrument.label()} | "
              f"{c.first.date()} .. {c.last.date()} | order {c.fit.order} | "
              f"{c.valid_T[0]:.2f}-{c.valid_T[1]:.2f} degC | rms {u.fit_rms_mK:.1f}, "
              f"held-out {u.held_out_mK:.1f}, block SD {block} mK"
              + (f" | {', '.join(c.flags)}" if c.flags else ""))
        if c.response is not None:
            print("     " + _response_line(c.response))
    s = prior.source
    why = ("its widest-range calibration (NONE IS UNFLAGGED)" if prior.fallback
           else "the widest-range unflagged calibration"
           + (" on the target" if prior.target is not None and not prior.cross_instrument
              else ""))
    print()
    print(f"Prior: from {s.deployment} ({s.instrument.label()}, {s.first.date()} .. "
          f"{s.last.date()}), {why}; order {prior.order}")
    print("  config lines (full precision):")
    for line in prior.config_lines():
        print(f"    {line}")
    print(f"  t_0 ±{prior.t_0_uncertainty_mK:.0f} mK"
          + (" (cross-instrument term included)" if prior.cross_instrument else "")
          + f"; shape ±{prior.shape_mK:.1f} mK over "
          f"{prior.shape_span_K[0]:g}-{prior.shape_span_K[1]:g} K; fitted "
          f"{prior.valid_T[0]:.2f}-{prior.valid_T[1]:.2f} degC")
    print("  bridge (the L these coefficients belong to): "
          + " ".join(f"{k}={v:g}" for k, v in prior.bridge.items()))
    for note in prior.notes:
        print(f"  - {note}")
    print("  Pasting into a .p config: remove any legacy `beta` key (ODAS and the reader use "
          "it before beta_1, so a pasted beta_1 would be silently ignored), and set a beta "
          "term this fit lacks but the config carries to 1e30, never 0.")
    return 0


def _response_line(r) -> str:
    """One line describing a calibration's FP07 response block."""
    def ci(v) -> str:
        return "" if v is None else f" [{v[0]:.1f}-{v[1]:.1f}]"

    head = (f"FP07 response vs {r.partner}, W_ref {r.W_ref_m_s:.3f} m/s, "
            f"{r.band_Hz[0]:g}-{r.band_Hz[1]:g} Hz: ")
    if r.tau_ms is None:
        head += f"tau NOT identified ({r.reason})"
    else:
        head += f"single-pole-equivalent tau {r.tau_ms:.1f}{ci(r.tau_ms_16_84)} ms at W_ref"
    if r.halves is not None:
        head += "; halves " + " -> ".join(
            "-" if h.tau_ms is None else f"{h.tau_ms:.1f}{ci(h.tau_ms_16_84)}" for h in r.halves)
    head += "; power vs partner " + ", ".join(
        f"{'-' if v is None else f'{v:.2f}'} @{f0} Hz" for f0, v in r.power_ratio.items())
    head += f"; {r.n_chi_files} chi files" + (f"; flags {', '.join(r.flags)}" if r.flags else "")
    return head


def _cmd_registry_response(args) -> int:
    """Record an `fp07-pair tau` measurement as an entry's FP07 response."""
    from odas_tpw.fp07cal.registry import (
        RegistryError,
        add_response,
        default_path,
        load,
        response_from_record,
    )

    path = Path(args.registry) if args.registry else default_path()
    rec_path = Path(args.record)
    try:
        record = json.loads(rec_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"refusing: {rec_path}: {exc}", file=sys.stderr)
        return 1
    try:
        reg = load(path)
        hits = [c for c in reg.history(args.bead)
                if " ".join(c.deployment.split()).casefold()
                == " ".join(args.deployment.split()).casefold()]
        if not hits:
            raise RegistryError(f"{args.bead} has no entry for {args.deployment!r} in {path}")
        block = response_from_record(record, hits[0], notes=args.notes or "")
        text = add_response(path, args.bead, args.deployment, block, dry_run=args.dry_run)
    except (RegistryError, KeyError, TypeError, IndexError, ValueError, AttributeError,
            OverflowError) as exc:
        print(f"refusing: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(text, end="")
    if args.dry_run:
        print(f"(dry run: valid against {path}; nothing written)")
    else:
        print(f"added the FP07 response to {args.bead} / {args.deployment} in {path}; views "
              f"re-rendered")
    return 0


def _none(s: str | None) -> str | None:
    return None if s is None or s.strip().lower() in ("", "none", "null") else s.strip()


def _cmd_registry_add(args) -> int:
    """Record one channel of an `fp07-cal fit` result in the sheet."""
    from odas_tpw.fp07cal.registry import (
        RegistryError,
        add_entry,
        block_offsets_from_report,
        default_path,
        entry_from_record,
    )

    path = Path(args.registry) if args.registry else default_path()
    rec_path = Path(args.record)
    try:
        record = json.loads(rec_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"refusing: {rec_path}: {exc}", file=sys.stderr)
        return 1
    report = Path(args.report) if args.report else rec_path.parent / f"{args.channel}_report.md"
    if not report.exists():
        print(f"refusing: {report} not found. The block scatter comes from the report "
              f"`fp07-cal fit` wrote beside {rec_path.name}; give it with --report.",
              file=sys.stderr)
        return 1
    try:
        entry = entry_from_record(
            record, args.channel,
            deployment=args.deployment,
            instrument_model=args.instrument_model,
            method=args.method,
            reference_kind=args.reference_kind,
            reference_serial=_none(args.reference_serial),
            reference_cal_date=_none(args.reference_cal_date),
            reference_note=args.reference_note,
            reference_pole_s=args.reference_pole_s,
            block_offsets_K=block_offsets_from_report(report.read_text(encoding="utf-8")),
            bead=args.bead,
            instrument_serial=args.instrument_serial,
            platform=args.platform,
            robust=args.robust,
            flags=args.flag or [],
            notes=args.notes or "",
            allow_qc_flag=args.allow_qc_flag,
        )
        block = add_entry(path, entry, dry_run=args.dry_run)
    except RegistryError as exc:
        print(f"refusing: {exc}", file=sys.stderr)
        return 1
    print(block, end="")
    span = (record["channels"].get(args.channel) or {}).get("span")
    if span is None:
        print(f"NOTE {rec_path.name} is {record.get('schema')}, from before per-channel spans: "
              f"first, last and n_files are the run's. If this bead stopped giving pairs "
              f"before the run ended, correct them (issue #249).", file=sys.stderr)
    else:
        if span.get("from", "every file") != "every file":
            print(f"NOTE {args.channel}'s span comes from {span['from']}, not every file: "
                  f"first and last can be a file stride early or late, and n_files counts "
                  f"only those. `fp07-cal fit` without --no-stream measures it over every "
                  f"file.", file=sys.stderr)
        note = _death_note(args.channel, span)
        if note:
            print(f"NOTE {note}", file=sys.stderr)
    if args.dry_run:
        print(f"(dry run: valid against {path}; nothing written)")
    else:
        print(f"added {entry['bead']} / {entry['deployment']} to {path}; views re-rendered")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="fp07-cal",
        description="FP07 in-situ calibration — a pre-pipeline step. "
        "Fits one Steinhart-Hart coefficient set per deployment from whatever "
        "yos carried a CTD reference, and reports whether it is stable in time.",
    )
    add_version_argument(p)
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

    q = sub.add_parser(
        "registry",
        help="the FP07 calibration sheet (microstructure_sensors/fp07_calibrations.yaml)",
        description="The FP07 calibration sheet. fp07_calibrations.yaml is the single "
        "source of truth; its .md and .csv views are generated from it and must never be "
        "edited by hand.",
    )
    rsub = q.add_subparsers(dest="registry_command", required=True)
    reg_help = "path to fp07_calibrations.yaml (default: the repository's)"
    r = rsub.add_parser("render", help="regenerate the .md and .csv views from the YAML")
    r.add_argument("-r", "--registry", help=reg_help)
    r.set_defaults(func=_cmd_registry_render)
    r = rsub.add_parser(
        "check", help="validate the YAML; exit 1 if it is invalid or a view is stale",
    )
    r.add_argument("-r", "--registry", help=reg_help)
    r.set_defaults(func=_cmd_registry_check)
    r = rsub.add_parser("show", help="one bead's calibrations and its prior")
    r.add_argument("serial", help="bead serial, e.g. T1592")
    r.add_argument("-r", "--registry", help=reg_help)
    r.add_argument("--instrument", help="target instrument serial, e.g. 194")
    r.add_argument("--channel", help="target channel, e.g. T1 (with --instrument)")
    r.add_argument("--as-of", help="use only calibrations that had ended by this date "
                   "(YYYY-MM-DD or YYYY-MM-DDTHH:MM:SSZ)")
    r.add_argument("--json", action="store_true", help="machine-readable output")
    r.set_defaults(func=_cmd_registry_show)
    r = rsub.add_parser(
        "add",
        help="record one channel of an `fp07-cal fit` result in the sheet",
        description="Append one bead x deployment entry, built from the fit's "
        "coefficients.json and <channel>_report.md, to the sheet's YAML (existing text "
        "and comments untouched), validate the whole sheet, and re-render its views. "
        "Refuses a second entry for the same bead and deployment. The options below "
        "are what the record cannot supply.",
    )
    r.add_argument("record", help="coefficients.json written by `fp07-cal fit`")
    r.add_argument("--channel", required=True, help="the channel to record, e.g. T1")
    r.add_argument("--deployment", required=True,
                   help='a label unique for this bead, e.g. "SUNRISE 2021 WaltonSmith"')
    r.add_argument("--instrument-model", required=True,
                   help="as the sheet writes it, e.g. vmp-250 or MR1000RDL-EM")
    r.add_argument("--method", required=True,
                   help="one of the sheet's `methods` (e.g. vmp-own-jac, glider-ctd)")
    r.add_argument("--reference-kind", required=True, help='e.g. "JAC CT"')
    r.add_argument("--reference-serial", required=True,
                   help="the reference's serial as printed (e.g. 23), or none")
    r.add_argument("--reference-cal-date", required=True,
                   help="its last calibration, YYYY-MM-DD, or none")
    r.add_argument("--reference-note", help="required when a reference field is none")
    r.add_argument("--reference-pole-s", type=float, required=True,
                   help="the fit config's pairs.kernel_tau [s]")
    r.add_argument("--bead", help="override the bead serial (required when the config's is "
                   "a placeholder)")
    r.add_argument("--instrument-serial", help="override the record's instrument_sn")
    r.add_argument("--platform", help='e.g. "Slocum osu684"')
    r.add_argument("--no-robust", dest="robust", action="store_false",
                   help="the fit ran with fit.robust: false (fp07-cal's default is true)")
    r.add_argument("--flag", action="append", help="a flag from the sheet's vocabulary "
                   "(repeatable), e.g. lag-gate-overridden")
    r.add_argument("--notes", help="anything a later reader needs")
    r.add_argument("--report", help="<channel>_report.md (default: beside the record)")
    r.add_argument("--allow-qc-flag", action="store_true",
                   help="record a fit whose QC against the bead's prior was flagged")
    r.add_argument("--dry-run", action="store_true", help="validate and print; write nothing")
    r.add_argument("-r", "--registry", help=reg_help)
    r.set_defaults(func=_cmd_registry_add)
    r = rsub.add_parser(
        "response",
        help="record an `fp07-pair tau` measurement as an entry's FP07 time constant",
        description="Insert a `response:` block, built from an `fp07-pair tau` JSON record, "
        "into one existing bead x deployment entry (existing text and comments untouched), "
        "validate the whole sheet, and re-render its views. Refuses a record of another "
        "instrument or channel, a measurement outside the entry's span, and an entry that "
        "already has a response.",
    )
    r.add_argument("record", help="fp07tau_*.json written by `fp07-pair tau`")
    r.add_argument("--bead", required=True, help="the entry's bead, e.g. T1592")
    r.add_argument("--deployment", required=True, help="the entry's deployment label")
    r.add_argument("--notes", help="anything a later reader needs about this measurement")
    r.add_argument("--dry-run", action="store_true", help="validate and print; write nothing")
    r.add_argument("-r", "--registry", help=reg_help)
    r.set_defaults(func=_cmd_registry_response)

    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

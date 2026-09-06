# Aug-2026, Claude and Pat Welch, pat@mousebrains.com
"""Configuration for ``mr-clocksync``: what to sync, to what, and how hard."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML

DEFAULTS: dict[str, Any] = {
    "reference": {
        "file": None,
        "kind": "auto",          # auto | matlab | netcdf | pfile
        "time_var": "T",
        "pressure_var": "P",
        "time_units": "auto",    # auto | datenum | epoch | epoch_ms
        "name": "reference",
        "scale": 1.0,
        "offset": 0.0,
    },
    "targets": [],               # [{name, glob, channel}]
    "cache": "clocksync-cache",
    "output": "clocksync",
    "analysis": {
        "rate": 16.0,
        "band": [0.05, 0.35],
        "window": 900.0,
        "step": None,
        "max_lag": 900.0,
        "max_gap": 1.0,
        "fit_rate": True,
        "min_windows": 3,
    },
    "gates": {
        "min_amplitude": 0.005,
        "min_dynamic_range": 3.0,
        "min_coherence": 0.5,
        "max_chi2": 25.0,
        "coh_min": 0.3,
    },
    "continuity": {
        "n_sigma": 5.0,
        "window": 9,
        "wave_period": None,
    },
}

TEMPLATE = """\
# mr-clocksync configuration
#
# Solves a per-FILE clock offset (and rate) for each target against a reference
# pressure record, using the wave band.  See docs/clocksync/runbook.md.

reference:
  # The clock everything else is moved onto.  Pick the instrument you have
  # independent reason to trust -- then verify it by syncing a third record to
  # it and checking the result is consistent.
  file: "sig_lowest_bin.mat"
  kind: auto              # auto | matlab | netcdf | pfile
  time_var: T
  pressure_var: P
  time_units: auto        # auto-detects datenum vs epoch; set it if auto errors
  name: sig1000
  scale: 1.0              # to dbar, if the source is not already
  offset: 0.0

targets:
  - name: MR330
    glob: "MR330/*.p"
    channel: P
  - name: MR429
    glob: "MR429/*.p"
    channel: P

cache: clocksync-cache    # one small .npz per .p file; resumable
output: clocksync         # solutions and reports land here

analysis:
  rate: 16.0              # common grid, Hz.  Match the reference: the target is
                          # decimated DOWN onto it, never the reference up.
  band: [0.05, 0.35]      # the wave band.  Everything is band-passed to this --
                          # tides are a ramp, and a shifted ramp is the same
                          # ramp, which is how you get r = 1.0 at every lag.
  window: 900.0           # seconds per lag estimate
  step: null              # null = non-overlapping windows
  max_lag: 900.0          # +/- search, seconds.  START WIDE.  On Bank Seaspider
                          # the MR clocks were 43-149 s out, and a 60 s search
                          # silently failed most files with "peak at the search
                          # boundary".  Widen until nothing reports that, then
                          # narrow to a little beyond the observed spread --
                          # a wider search is more chances to lock onto the
                          # wrong wave cycle.
  max_gap: 1.0            # seconds; longer holes are blanked, never bridged
  fit_rate: true          # false = one offset per file, no drift term
  min_windows: 3          # fewer than this: offset only, and say so

gates:
  # Deliberately NOT gated on correlation -- see lag.py.  A high r is not
  # evidence; a sharp, coherent peak in a band that actually contains a wave is.
  min_amplitude: 0.005    # dbar of band-passed signal.  Below this there is no
                          # wave and any lag comes from filter ringing.
  min_dynamic_range: 3.0  # envelope peak / median envelope
  min_coherence: 0.5      # median coherence in band
  max_chi2: 25.0          # phase(f) must be a straight line for a pure delay
  coh_min: 0.3            # per-frequency floor for inclusion in the phase fit

continuity:
  # A clock drifts; it does not teleport.  A file whose offset jumps away from
  # its neighbours is usually a cycle slip, not a clock.
  n_sigma: 5.0
  window: 9               # files in the running median
  wave_period: null       # e.g. 8.3 -- lets a slip be NAMED as N wave periods
"""


def _yaml() -> YAML:
    y = YAML()
    y.preserve_quotes = True
    y.default_flow_style = False
    return y


def write_template(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(TEMPLATE)
    return path


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        elif v is not None or k not in out:
            out[k] = v
    return out


def load_config(path: str | Path) -> dict:
    path = Path(path)
    with open(path) as fh:
        raw = _yaml().load(fh) or {}
    cfg = _merge(DEFAULTS, dict(raw))
    cfg["_path"] = str(path)
    cfg["_root"] = str(path.parent)
    if not cfg["reference"].get("file"):
        raise ValueError(f"{path}: reference.file is required")
    if not cfg.get("targets"):
        raise ValueError(f"{path}: at least one entry under targets: is required")
    return cfg


def dump_config(cfg: dict) -> str:
    buf = io.StringIO()
    _yaml().dump({k: v for k, v in cfg.items() if not k.startswith("_")}, buf)
    return buf.getvalue()


def resolve(cfg: dict, value: str | Path) -> Path:
    """Config-relative paths, so a config is portable with its data."""
    p = Path(value)
    return p if p.is_absolute() else Path(cfg["_root"]) / p

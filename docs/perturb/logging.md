# Perturb Logging

Every `perturb` invocation that runs a pipeline stage emits a hierarchy of
log files under the output root, so you can diagnose problems after the
fact without re-running.  A single `logger.info(...)` fans out to several
files — the per-invocation run log, the worker that produced it, and the
stage-specific log inside the versioned output directory.

## Layout

```
<output_root>/
  logs/
    run_<stamp>.log              # whole CLI invocation, always written
    worker_<stamp>_<pid>.log     # one per parallel worker
  profiles_NN/    <stem>.log     # per-.p-file profile-stage records
  diss_NN/        <stem>.log
  chi_NN/         <stem>.log
  ctd_NN/         <stem>.log
  profiles_binned_NN/  <stem>.log
  diss_binned_NN/      <stem>.log
  chi_binned_NN/       <stem>.log
  combo_NN/       combo.log      # per-combo step
  diss_combo_NN/  combo.log
  chi_combo_NN/   combo.log
  ctd_combo_NN/   combo.log
```

`<stamp>` is a UTC timestamp like `20260504T230320Z` shared across the
run log and every worker log from the same invocation.  `<stem>` is the
`.p` file basename without extension (e.g. `ARCTERX_Thompson_2025_SN479_0002`),
recovered for binning logs by stripping the per-profile `_prof###`
suffix.

## What goes where

| Layer | Purpose | Useful when |
|-------|---------|-------------|
| `logs/run_<stamp>.log` | Top-level pipeline events: file discovery, trim/merge, per-file `Done:`, binning kickoff, combo writes. | You want a high-level timeline of the whole run. |
| `logs/worker_<stamp>_<pid>.log` | Everything that one parallel worker did, in chronological order. | A worker died or stalled; you want to see exactly which `.p` file it was on. |
| `<stage>_NN/<stem>.log` | All records emitted while processing one `.p` file in one stage. | A specific file's diss/chi/profile result looks wrong; this log captures FP07 cal warnings, profile-detection messages, dissipation errors for that file alone. |
| `*_binned_NN/<stem>.log` | Aggregated per-`.p`-file records during binning. Multiple per-profile NetCDFs from the same `.p` file append into one log. | A binned variable is missing for one source file; this isolates which one. |
| `*_combo_*/combo.log` | The single combo-assembly step's records. | A combo NetCDF failed to write or contains the wrong files. |

## Console behaviour

By default `perturb` prints **nothing** to stderr — the run log captures
everything.  Pass `--stdout` to also mirror records to stderr in a terse
format while you watch progress:

```bash
perturb run -o results/ -j 4 --stdout VMP/*.p
```

Combine with shell redirection if you want to keep a tee'd copy:

```bash
perturb run -o results/ --stdout VMP/*.p 2>&1 | tee live.log
```

The default-silent change is intentional: a pipeline run that produces
two `.p` files plus binning emits ~50 INFO records; on a 30-file campaign
this is hundreds of lines that previously scrolled the terminal.

## Log format

The file format is verbose for postmortem reading; the stderr format is
terse.

* File: `2026-05-04T16:03:21 INFO    odas_tpw.perturb.pipeline [SpawnProcess-1]: Trimmed: foo.p`
* stderr (`--stdout`): `Trimmed: foo.p`

`processName` (`MainProcess` vs. `SpawnProcess-N`) discriminates parent
events from worker events in shared files, which matters for the run log
when `jobs > 1`.

## Log volume

Rough numbers for the ARCTERX 2025 campaign (~25 profiles per `.p`
file, two shear probes, chi enabled):

| File | Lines per `.p` | Notes |
|------|----------------|-------|
| `run_*.log` | 8 + 1 per file | Discovery + trim + Done + binning + combo. |
| `worker_*_*.log` | 25–30 per file the worker handles | Profile extraction dominates. |
| `profiles_NN/<stem>.log` | 25–30 | One INFO per detected profile. |
| `diss_NN/<stem>.log` | 0 unless errors | Only logs warnings and exceptions. |
| `chi_NN/<stem>.log` | 0 unless errors | Same. |
| `*_binned_NN/<stem>.log` | 0 unless errors | Binning is silent on the happy path. |
| `combo*/combo.log` | 1 | The `Wrote …` line. |

Empty log files are intentional — the FileHandler is opened by the
stage-log context manager regardless of whether records flow through it,
so the existence of `diss_NN/X.log` proves the diss stage ran for `X.p`
even when there's nothing else to say.

## Programmatic access

The helpers are exposed as `odas_tpw.perturb.logging_setup`:

```python
from odas_tpw.perturb.logging_setup import (
    setup_root_logging,    # CLI wires this from --stdout/--log-level
    init_worker_logging,   # ProcessPoolExecutor initializer
    stage_log,             # context manager: with stage_log(dir, stem): ...
    current_run_stamp,     # shared timestamp for one invocation
)
```

`stage_log(dir, basename)` is a no-op when `dir is None`, so it's safe to
wrap optional stages without conditionals.

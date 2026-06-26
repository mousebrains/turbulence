# Multi-Round Deep Review — PARTIAL (2026-06-25/26)

**Status: INCOMPLETE.** The up-to-3-round adversarial review of the project
(branch `fix/audit-2026-06-25-majors` / PR #78) ran **round 1 + part of round 2**
across two quota windows (~246 agents), then hit the **weekly** usage limit
before the synthesis step. There is therefore **no deduplicated/synthesized
report**; this is the raw extraction recovered from the agent transcripts.

- Raw candidate findings: **108** — 20 major, 55 minor, 33 nit
- Verifier verdicts rendered: **120** → **92 confirmed / 28 refuted** (~77%;
  refute-by-default). Pairing of each verdict to its finding was not recoverable
  from transcripts, so individual major findings below are **candidates, not yet
  confirmed**; expect some to be downgraded/refuted on a final synthesis.
- Full raw dump: `2026-06-25-multiround-RAW-partial.json`.

## ⚠️ Findings against PR #78's own fixes (check before merge)

1. **`rsi/profile.py` (M2 regression):** the `_nc_filled` fix stops the 9.97e36
   leak but a single mid-cast `_FillValue`→NaN now propagates through the
   fall-rate (gradient + filtfilt smear) so **zero profiles are detected for the
   whole file** — trading a fill-leak for silent total loss.
2. **`rsi/profile.py`:** `extract_profiles` reportedly raises `AttributeError`
   on an external/partial CF NetCDF whose channel carries a `_FillValue`
   attribute — the exact files M2 claims to support.
3. **`processing/ct_align.py` (M3):** the NaN guard discards the **entire
   profile** on one isolated bad sample, silently wiping CT alignment for a
   whole file (the over-discard the single-fix reviewer flagged as possible).

## Other MAJOR candidates (grouped)

**Silent data loss / wrong numbers**
- `scor160/despike.py`: a single NaN disables ALL spike detection → real spikes
  pass into chi/epsilon.
- `chi/chi.py`: iterative MLE fit discards a valid earlier kB when a later
  refinement → NaN, dropping the whole window.
- `rsi/chi_io.py`: Method-1 epsilon time reference corrupted by tz-offset
  stripping → NaN all chi for non-UTC instruments.
- `rsi/chi_io.py`: Method-1 filters the wrong FOM statistic (variance-ratio
  `fom` vs Lueck `FM`; the 1.15 limit is defined for FM) → drops good probes.
- `processing/top_trim.py`: returns the shallowest sub-threshold bin → under-trim
  leaves the noisy near-surface section in the profile.
- `perturb/merge.py`: still no time-continuity/`restarted` check → 29 independent
  ARCTERX casts fuse into one chain (regressed/unfixed from the first audit).

**Physics / ODAS port-faithfulness**
- `chi/batchelor.py`: Kraichnan gradient spectrum uses a different functional
  form than the in-repo MATLAB reference → biases kB/chi on the default path.
- `rsi/p_file.py`: deconvolution runs BEFORE the unsigned-wrap → corrupts any
  pre-emphasized unsigned channel at the int16 wrap.
- `scor160/ocean.py`: `visc()` Sharqawy branch lacks the negative-viscosity
  floor `visc35` has → crashes nasmyth on a spurious-cold window mean.

**Config / infrastructure**
- `config_base.py`: `round(v, 10)` collapses 1e-13 minima to 0.0 → corrupts the
  version hash / provenance.
- `config_base.py`: the `{prefix}_[0-9][0-9]` glob misses dir 100+ → reuse
  failure / collision past 100 configs.
- `config_base.py`: `compute_hash`/canonicalize crash on mixed-type dict keys.
- `rsi/config_patch.py`: edit-spec keys "never validated" → stanza injection.
  **(Likely false positive — key validation WAS added in PR #77; verify.)**

**Robustness / packaging / metadata**
- `rsi/p_file.py`: no short-header length guard → raw `struct.error` on a
  truncated file.
- `rsi/convert.py`: `convert_all` aborts the whole batch on one corrupt `.p`.
- `perturb/gps.py`: undeclared `pandas` runtime dependency → crashes GPS/hotel
  CSV ingestion on a clean install.
- `rsi/binning.py`: L5/L6 binned/combined products carry no
  units/long_name/standard_name.

## Next steps
- The review needs its **synthesis** step (dedupe + verify pairing + severity
  recalibration) to become a trustworthy report — that requires re-running once
  the weekly limit resets, OR synthesizing from the raw JSON.
- Treat the three PR #78 items as **merge-blockers to verify first**; the M2
  NaN-propagation one is the highest priority (potential regression in the fix).

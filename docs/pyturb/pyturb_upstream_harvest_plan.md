# Current pyturb Harvest Implementation Plan

Date: 2026-06-07

Upstream reviewed: `oceancascades/pyturb` at commit
[`ec9f09469f6778b21c5a06d90d5eba1d7162722e`](https://github.com/oceancascades/pyturb/commit/ec9f09469f6778b21c5a06d90d5eba1d7162722e)
(`main`, fetched 2026-06-07).

## Goal

Bring this project's `odas_tpw.pyturb` compatibility layer up to the current
Jesse `pyturb` workflow where that improves interoperability, while keeping the
core `perturb`, `rsi`, `scor160`, and `chi` pipelines stable unless a change is
explicitly validated against their existing benchmarks.

This is a compatibility harvest, not a wholesale replacement of the processing
engine.

## Current Findings

The previous comparison notes are stale. They still say upstream lacks Goodman
cleaning and QC metrics, but current upstream has:

- p2nc optional despiking with `<probe>_clean` and `<probe>_despike_mask`.
- A `cutp` command for extracting valid P-file segments.
- Config-driven epsilon processing with serialized `pyturb_config` metadata.
- Delayed shear and gradT velocity scaling in the spectral stage.
- Accelerometer and EM-current coherent-noise cleaning.
- Per-window epsilon QC from speed, FM/MAD fit quality, and despike fraction.
- QC-aware binning with combined `eps` and `eps_qc`.
- New kmax tests around realistic rising shear-channel noise.

Local drift in `odas_tpw.pyturb`:

- `--aoa` and `--pitch-correction` are accepted but explicitly not implemented.
- `cutp` is absent from the local emulation command set.
- `p2nc` lacks upstream `--despike`.
- `bin` lacks upstream `--qc-thresh`, per-bin QC aggregation, and combined
  `eps` / `eps_qc`.
- Several command defaults and option shapes now differ, including packed
  `--peaks`, packed `--despike`, packed `--qc-thresh`, and worker/default
  behavior.
- Local p2nc/L1 adapters velocity-scale shear during conversion/adaptation,
  while upstream now intentionally keeps probe time series unscaled until
  spectral processing.
- Documentation still describes the old upstream state.

## Compatibility Contract

Implement compatibility in layers, so each phase is reviewable and testable.

1. CLI compatibility: current upstream options parse and map to real behavior.
2. Data compatibility: current upstream p2nc-style NetCDF can be read, and
   local legacy `L1_converted` NetCDF remains accepted.
3. Output compatibility: expected variables, QC variables, metadata, and binned
   products are present with current upstream names.
4. Numerical compatibility: results are scientifically comparable, but exact
   byte-for-byte parity is not required because this project intentionally uses
   SCOR-160 and ATOMIX-validated paths in places.

## Non-Goals

- Do not replace the core `scor160` epsilon estimator with upstream `pyturb`
  without a separate ATOMIX and existing-test validation.
- Do not make Typer a dependency for this project just because upstream uses it.
- Do not remove support for local `L1_converted` files or existing `pyturb-cli`
  tests during this harvest.
- Do not silently change output interpretation when data are already
  velocity-scaled.

## Implementation Phases

### Phase 0: Baseline and Documentation

Deliverables:

- Update `docs/perturb/pyturb_comparison.md` and `docs/pyturb/PYTURB_CLI.md`
  to reflect upstream commit `ec9f094`.
- Add a short compatibility matrix that separates CLI compatibility, data-format
  compatibility, and algorithmic parity.
- Add a small test or static check that records the upstream commit used for
  the comparison so future reviews know when the plan is stale.

Validation:

- Documentation-only diff is reviewed before behavior changes.
- No code paths changed.

### Phase 1: Config and CLI Surface

Deliverables:

- Add a compatibility config object for the `odas_tpw.pyturb` layer. It should
  cover upstream `ProfileConfig` fields that are user visible:
  `diss_len_sec`, `fft_len_sec`, pressure/speed/temperature variable names,
  pressure smoothing, speed estimation, pitch correction, AoA, probe names,
  despike settings, acceleration and EM-current cleaning settings, profile
  direction, peak detection settings, and QC thresholds.
- Replace the current `--aoa` and `--pitch-correction` warnings with real
  behavior.
- Add a command/default audit against upstream `ec9f094`. For each difference,
  choose one of: implement, keep local behavior and document why, or preserve
  both through aliases.
- Add `cutp` or explicitly document why it is out of scope. The preferred
  implementation is a thin wrapper around existing P-file segment extraction if
  equivalent local functionality is already present.
- Add upstream-style packed options:
  `--despike passes,thresh,smooth,replace_sec`,
  `--peaks height,distance,width,prominence`,
  and `bin --qc-thresh questionable,bad`.
- Preserve the current split peak flags (`--peaks-height`,
  `--peaks-distance`, `--peaks-prominence`) as aliases for one release cycle,
  with documented precedence if both forms are supplied.
- Add `--accel-clean/--no-accel-clean` and `--emc-clean/--no-emc-clean`.
  Default `emc_clean` should match current upstream for the compatibility
  layer, but only if channel detection is safe.
- Make auxiliary T/S/density opt-in like upstream: lat/lon default names remain,
  but auxiliary temperature, salinity, and density are applied only when named.

Validation:

- CLI parser unit tests cover defaults and all new options.
- Command-coverage tests verify that all upstream command names are either
  implemented or intentionally documented as unsupported.
- Existing local tests that assert `--aoa` only warns are updated to assert
  configuration is passed through.
- Backward-compatible flags continue to parse.

### Phase 2: Data Format Detection and p2nc Despiking

Deliverables:

- Teach `odas_tpw.pyturb.eps` to classify input NetCDFs:
  `pyturb_raw`, `local_l1_converted`, or unsupported.
- Add a scaling-state marker for generated outputs, for example
  `probe_scaling = "raw"` or `probe_scaling = "velocity_scaled"`.
- Implement upstream-style `p2nc --despike`:
  - despike `sh1`, `sh2`, `gradT1`, `gradT2` when present;
  - emit `<probe>_clean` and `<probe>_despike_mask`;
  - persist despike parameter attrs;
  - do not force despiking when the option is absent.
- Keep existing local `L1_converted` p2nc behavior available. If a default
  output-format change is proposed, gate it behind an explicit review because
  existing users may rely on the current ATOMIX-style output.

Recommended default:

- Add `--format pyturb|l1-converted` to `pyturb-cli p2nc`.
- For the first implementation, keep the current default and document
  `--format pyturb` as the current-upstream-compatible path.
- Revisit flipping the default only after external review.

Validation:

- Unit tests verify generated despike clean/mask variables and attrs.
- Tests verify `eps` refuses to double-scale a dataset marked
  `velocity_scaled`.
- Tests verify both raw upstream-style and local `L1_converted` inputs still
  process.

### Phase 3: Delayed Scaling Processing Path

Deliverables:

- Add a raw-probe processing path for upstream-style p2nc datasets:
  - smooth pressure/speed with gap-aware filtering;
  - estimate speed from pressure when the named speed variable is absent;
  - apply AoA/pitch correction when requested;
  - despike raw probes before velocity scaling;
  - high-pass shear after despiking;
  - compute spectra on raw cleaned probes;
  - apply shear `1 / W^4` and gradT `1 / W^2` spectral scaling using
    window-mean speed, matching upstream's delayed scaling intent.
- Keep the existing `scor160` path for already-scaled `L1_converted` files.
- Make scaling state explicit in logs and output attrs.

Validation:

- Synthetic turnaround test: raw-probe despiking should not be dominated by
  low-speed velocity scaling near turnarounds.
- Regression test for already-scaled inputs ensures they are not scaled again.
- Existing SCOR-160 tests must continue to pass.

### Phase 4: Coherent-Noise Cleaning Enhancements

Deliverables:

- Add a generic coherent-cleaning helper for the compatibility layer that can
  use accelerometer references and EM-current references.
- Preserve current `scor160.goodman` behavior for the core pipeline.
- Support configurable reference channel names, with defaults matching
  upstream (`Ax`, `Ay`, `Az`, `EMC_Cur`, `EM_Cur`) and local aliases where
  available.
- Log which reference channels were actually used.

Validation:

- Synthetic coherent-contamination test: cleaned spectra have reduced coherent
  power and no negative PSD after clipping.
- Missing requested reference channels produce a warning, not a crash, unless
  the user explicitly requests strict behavior.

### Phase 5: Per-Window QC and Metadata

Deliverables:

- Emit `eps_1_qc`, `eps_2_qc`, `eps_1_fm`, `eps_2_fm`,
  `<probe>_despike_frac`, and `k_max_N` for compatibility outputs.
- Compose QC from:
  - speed below `min_speed`;
  - FM/MAD thresholds;
  - per-window despike fraction thresholds;
  - missing epsilon.
- Stamp output metadata:
  `source_file`, `profile_index`, `profile_direction`, `pyturb_version` or
  local compatibility version, `pyturb_processed_utc`, serialized
  compatibility config, and instrument attrs where available.

Validation:

- QC composition unit tests cover speed-only, FM-only, despike-only, mixed,
  and missing-epsilon cases.
- Metadata tests verify config serialization and required attrs.

### Phase 6: QC-Aware Binning

Deliverables:

- Implement upstream-style bin masking:
  - `qc=2` values above `questionable_thresh` are excluded before binning;
  - `qc=4` values above `bad_thresh` are excluded before binning;
  - low-epsilon flagged values are retained by default;
  - excluded QC values are restored to valid missing flag `9` after binning.
- Aggregate QC variables with max and numeric variables with mean.
- Add combined binned `eps` and `eps_qc`:
  - average probes when they agree within a factor of 10;
  - use the lower finite value when both exist but disagree more;
  - use the surviving probe when only one is finite;
  - use missing when neither is finite.
- Add `--qc-thresh` CLI support.

Validation:

- Unit tests for masking thresholds, sentinel restoration, QC max aggregation,
  and combined epsilon behavior.
- Binning integration test with synthetic depth/pressure coordinates.

### Phase 7: kmax Test Harvest

Deliverables:

- Port the useful parts of upstream's latest synthetic kmax tests into this
  project's test suite, adjusted for the local estimator's intended constants.
- Use local `odas_tpw.rsi.shear_noise.noise_shearchannel` for realistic rising
  shear-channel noise fixtures.
- Treat failures as evidence to investigate, not as a mandate to copy upstream
  constants.

Validation:

- Confirm current `scor160.l4` behavior remains consistent with ATOMIX notes
  and existing benchmark tests.
- If these tests reveal a real bug, open a separate implementation branch for
  estimator changes.

### Phase 8: Cross-Tool Validation

Deliverables:

- Add a reproducible comparison script that can run local `pyturb-cli` and
  upstream `pyturb` on the same small dataset when upstream is available.
- Record output-variable presence, profile counts, pressure-grid alignment,
  and log10 epsilon ratio quantiles.
- Keep this script out of routine unit tests if it requires network or an
  external checkout.

Validation:

- Run normal unit tests locally.
- Run optional cross-tool comparison before changing default p2nc output format.

## Suggested Work Breakdown

1. Documentation refresh and compatibility matrix.
2. CLI/config parsing and tests.
3. p2nc format detection plus optional despiking.
4. Raw-probe delayed-scaling processor.
5. QC variables and metadata.
6. QC-aware binning.
7. kmax/noise test harvest.
8. Optional cross-tool comparison tooling.

Each step should be independently reviewable. Avoid bundling estimator changes
with CLI/data-format changes.

## Adversarial Review

### Challenge 1: "This plan may break existing users by changing p2nc output."

Risk:

The local `pyturb-cli p2nc` currently writes through `odas_tpw.rsi.convert.p_to_L1`,
which produces ATOMIX-style `L1_converted` data with velocity-scaled shear.
Current upstream `pyturb p2nc` writes raw probe variables and delays velocity
scaling. Flipping defaults would surprise existing users and tests.

Mitigation applied to plan:

- Add format detection.
- Keep local `L1_converted` accepted.
- Add `--format pyturb|l1-converted`.
- Defer any default flip to a separate review.

### Challenge 2: "Delayed scaling can silently double-scale data."

Risk:

If a dataset is already scaled by speed during conversion, then applying
upstream delayed spectral scaling again would corrupt epsilon.

Mitigation applied to plan:

- Require scaling-state metadata on generated outputs.
- Detect known local `L1_converted` structure.
- Add tests that fail on double scaling.
- Emit logs/attrs for the selected scaling path.

### Challenge 3: "Exact numerical parity is not realistic."

Risk:

This project intentionally uses SCOR-160 and ATOMIX-tuned processing in core
paths. Upstream `pyturb` has different implementation choices. Chasing exact
parity could degrade validated behavior.

Mitigation applied to plan:

- Define compatibility levels.
- Preserve the core estimator.
- Move kmax work into tests first.
- Make estimator changes a separate branch if evidence warrants it.

### Challenge 4: "QC-aware binning may discard scientifically important data."

Risk:

Upstream's default thresholding discards high-epsilon values only when QC flags
already mark them questionable/bad. Even so, threshold defaults can influence
products.

Mitigation applied to plan:

- Keep thresholds configurable through `--qc-thresh`.
- Document defaults.
- Preserve low-epsilon flagged values by default, matching upstream rationale.
- Add tests that make the masking behavior explicit.

### Challenge 5: "EM-current channel naming is not stable across instruments."

Risk:

Upstream defaults to `EMC_Cur` and `EM_Cur`; this project already handles
`U_EM` for speed and has multiple EM converter names. A narrow channel-name
implementation would miss data or produce noisy warnings.

Mitigation applied to plan:

- Make reference channel names configurable.
- Include local aliases after inventory.
- Log actual references used.
- Treat missing references as a warning by default.

### Challenge 6: "Adding upstream-like code may duplicate core code."

Risk:

Copying upstream profile processing wholesale could create parallel
implementations that diverge and become hard to maintain.

Mitigation applied to plan:

- Keep compatibility helpers inside `odas_tpw.pyturb`.
- Reuse existing primitives where the semantics match.
- Do not modify core `scor160` behavior for compatibility-only features.
- Add tests around the compatibility boundary, not just internals.

### Challenge 7: "The upstream reference can change again."

Risk:

The plan is pinned to `ec9f094`; future upstream commits may invalidate some
details.

Mitigation applied to plan:

- Record the exact upstream commit and fetch date.
- Add a doc/test marker for the comparison baseline.
- Treat future upstream updates as a follow-up harvest rather than mixing
  them into this scoped implementation.

### Challenge 8: "The plan under-scopes the CLI by focusing on flags, not commands."

Risk:

Upstream now has `cutp` and changed some command defaults. If the local layer
only adds individual flags, external users may still find the command surface
incomplete.

Mitigation applied to plan:

- Add a command/default audit in Phase 1.
- Require every upstream command to be implemented or explicitly documented as
  unsupported.
- Prefer implementing `cutp` as a thin wrapper if existing local extraction
  code is equivalent.

## Review Gate Before Coding

Before starting implementation, reviewers should answer:

1. Should `pyturb-cli p2nc` default remain the local `L1_converted` output for
   now, or should current-upstream raw output become default immediately?
2. Should numerical compatibility target upstream `pyturb`, local SCOR-160,
   or a documented hybrid?
3. Are EM-current cleaning defaults acceptable for all current instruments, or
   should `emc_clean` default off locally until channel aliases are audited?
4. Should `--peaks-height` style legacy flags remain indefinitely, or only for
   a compatibility window?
5. Should `cutp` be part of this harvest, or documented as unsupported until a
   data-backed use case appears?

The safest implementation path is to resolve these questions before Phase 2.

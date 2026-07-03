# 2026-07-01 Deep Audit — Remediation Log

Disposition of the confirmed findings from
[`2026-07-01-deep-audit.md`](./2026-07-01-deep-audit.md). All fixes land on
`main`; the full test suite is green throughout (2190 passed, 76 skipped) and
every code fix carries a regression test unless noted.

## Fixed

### Major findings (7) — committed individually
| Finding | Commit | One-line |
|---------|--------|----------|
| r1-1 mixing K_T/Gamma use in-situ (not conservative) T gradient | `09ffc08` | fit conservative-temperature gradient (Osborn-Cox) |
| r1-2 top_trim deep-transient over-trim | `ebb7749` | anchor prop-wash exit to the surface-attached run |
| r1-3 bottom-crash false positives delete deep data | `1958542` | gate crash detection to the near-bottom zone |
| r1-4 chi QC runs after mixing derivation | `72dd9e4` | apply chi QC before deriving K_T/Gamma/K_rho |
| r1-5 interrupted binned.nc validates as cache-current | `f1168f5` | atomic binned.nc write |
| r1-13 quick_look windowed eps/chi all-NaN after profile 1 | `533194f` | size interp to the full record |
| r2-3 glider vertical-speed epsilon (~5x) | `a99199d` | warn on `|dP/dt|` speed for glide/horizontal |

### Minor / nit findings — grouped commits
| Batch | Commit | Scope |
|-------|--------|-------|
| A (docs) | `2d233f7` | 15 doc-vs-code mismatches (math docs, ARCHITECTURE, DATA_FLOW, CLI, ARCTERX config) |
| B (comments) | `3f6b1ad` | 11 misleading code comments/docstrings (FM interpretation ×5, chi_combine step 7, quick_look M2 label, K_rho attribution, speed dbar/sign, GRADT, LnSigma, S_sh, top_trim) |
| C robustness | `2976a3b` | chi below-noise→NaN, top_trim never-settled clamp, fp07 lag guard, gps coverage window, parse_time zero-offset |
| C cache/QC/schema | `59feb41` | trim fingerprint order, two-sided chi fom QC (rsi+perturb), CHI_SCHEMA chiMean/chiLnSigma, pandas engine-dep |
| C ODAS parity | `fcbc091` | accel adc_bits 16→0 regression, bad-buffer warning, vehicle-tau profile detection |
| C disclosure | `d784d20` | fixed-kappa_T bias, perturb no-QC-default, diagnostics no-ops, chi-M2 epsilon provenance |
| C/D pyturb+plot | `de3aeff` | per-file FFT window, ignored-flag warnings, geometric-bin disclosure, bounded plot ffill |
| C cache integrity | `c95e6b3` | atomic per-profile diss/chi writes, stale-combo clearing |
| E test-validity | `33d5441` | consistent MLE fixture, literal Nasmyth values, tightened synthetic-recovery windows |

## Deferred — need a decision or data this session could not verify

These are **not** dropped; each is documented in-code and/or here so it is not
lost. They were held back because a blind change risks the scientific output or
cannot be verified without artifacts outside the repo.

| Finding | Why deferred | Recommended fix |
|---------|--------------|-----------------|
| `chi/batchelor.py:23` — fixed `kappa_T` biases chi ~−6.5% / epsilon_T ~−12.6% in warm water | Threading per-window `kappa_T(T,S,P)` touches ~13 sites in the chi core and changes **every** published chi/epsilon_T value | Implement the T-dependent kappa_T and regenerate chi products (bias documented in `chi_mathematics.md` §11 and at the constant) |
| `processing/mixing.py:515` — K_rho ceiling is one-sided; K_T/Gamma unbounded in the same windows | Choosing plausibility ceilings for K_T and Gamma is a domain judgement (what Γ / K_T is "implausible"?) | Add `K_T_max` / a Γ sanity bound, or mask K_T/Gamma at the K_rho-masked windows — needs your thresholds |
| `perturb/netcdf_schema.py:44` — `bin` coord holds pressure (dbar) but is labelled depth/m | Which fix is correct depends on whether a true-depth var or `P_mean` was binned, and it changes a **published CF coordinate**; two valid resolutions | Relabel to `dbar`/`sea_water_pressure` (match rsi) **or** convert P→depth before binning; then fix combo geospatial attrs + plot axis labels |
| `rsi/p_file.py:359` — `start_time` is +1 s vs the ODAS recsize convention | Shifts **every** absolute timestamp (GPS matching, merges); I could not confidently confirm the recsize source/value this session | Subtract recsize (default 1.0 s) after `parse_config`; pin `start_time` against the MATLAB `_allch.nc` in a test |
| Zero-input residual of `pipeline.py:2401` — stale `binned.nc` persists when input NCs shrink to exactly zero | Clearing needs the binned dirs resolved unconditionally (a dir-creation side effect); rare edge case | Resolve the binned dir even when the NC list is empty and clear its `binned.nc` |
| `test_pipeline_vs_matlab.py:421`, `test_matlab_epsilon.py:218`, `test_matlab_chi.py:228` — accuracy gates too loose | These skip in CI (VMP/ and AtomixData are gitignored); tightening blind could assert an accuracy the code misses on real data | With the reference data present: tighten to ~0.05–0.1 decades, add a signed-bias assert, restrict chi to `F ≤ f_AA` |
| `scripts/compare_atomix.py:282,630` — AA-cut fallback truncates integration; FOM plot mixes two statistics | Analysis **script**, not published-product code; lower priority | Use `0.9·f_AA` (not `HP_cut`) for the AA limit; plot the Lueck FM vs the package FM on separate axes |
| `rsi/dissipation.py:265` — perturb epsilon carries no ATOMIX spectral-fit QC by default | Behaviour/policy: adding an FM cut changes published epsilon | **Documented** (config `epsilon.fom_max` comment); decide whether to default an FM cut |

## Notes
- `docs/audits/2026-07-01-deep-audit-findings.json` keeps the original finding
  text and evidence; this log is the disposition layer on top of it.
- New regression tests live in `tests/test_audit_2026_07_robustness.py` and in
  `tests/test_p_file_branches.py::TestPFileBadBuffer`.

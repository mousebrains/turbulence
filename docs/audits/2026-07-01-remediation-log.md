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

### Follow-up round (2026-07-03, at Pat's direction)
| Finding | Commit | One-line |
|---------|--------|----------|
| `p_file.py:359` start_time +1 s vs ODAS | `795e622` | subtract recsize (default 1 s); verified vs MATLAB `_allch.nc` (offset → 7.6 µs) |
| `netcdf_schema.py:44` bin coord pressure-labelled-depth | `abfd748` | convert P→depth in **metres** via `gsw.z_from_p` (profile lat; 45° fallback) + fix plot labels |
| `mixing.py:515` K_rho-only ceiling | `3c0bfc3` | per-variable `K_T_max` (10 m²/s) and `Gamma_max` (5) ceilings |
| `compare_atomix.py:282,630` AA cut + FOM plot | `8e5b3a9` | never use HP_cut as f_AA + 0.9 margin (Baltic RMSD 0.192→0.008, all 6 PASS); FOM plot axes fixed |
| `batchelor.py:23` fixed `kappa_T` → chi bias | `30444a9` | per-window T-dependent `ocean.kappa_T(T,S,P)` (Sharqawy/Jamieson–Tudhope + gsw) threaded L3→L4→chi.py; removes a chi bias of ~−1% (−1 °C) to ~+8% (32 °C); epsilon_T ∝ kappa_T² verified |

## Deferred — need a decision or data this session could not verify

These are **not** dropped; each is documented in-code and/or here so it is not
lost. They were held back because a blind change risks the scientific output or
cannot be verified without artifacts outside the repo.

| Finding | Why deferred | Recommended fix |
|---------|--------------|-----------------|
| Zero-input residual of `pipeline.py:2401` — stale `binned.nc` persists when input NCs shrink to exactly zero | Clearing needs the binned dirs resolved unconditionally (a dir-creation side effect); rare edge case | Resolve the binned dir even when the NC list is empty and clear its `binned.nc` |
| `test_pipeline_vs_matlab.py:421`, `test_matlab_epsilon.py:218`, `test_matlab_chi.py:228` — accuracy gates too loose | These skip in CI (VMP/ and AtomixData stay gitignored — too large for GitHub) | With the reference data present locally: tighten to ~0.05–0.1 decades, add a signed-bias assert, restrict chi to `F ≤ f_AA` |
| `rsi/dissipation.py:265` — perturb epsilon carries no ATOMIX spectral-fit QC by default | Behaviour/policy: adding an FM cut changes published epsilon | **Documented** (config `epsilon.fom_max` comment); decide whether to default an FM cut |

## Notes
- `docs/audits/2026-07-01-deep-audit-findings.json` keeps the original finding
  text and evidence; this log is the disposition layer on top of it.
- New regression tests live in `tests/test_audit_2026_07_robustness.py` and in
  `tests/test_p_file_branches.py::TestPFileBadBuffer`.

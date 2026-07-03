# Deep Adversarial Audit — microstructure-tpw (2026-07-01, branch main @ HEAD)

## 1. Executive Summary

This audit re-swept the full `microstructure-tpw` chain at HEAD (post PR #79-#87):
RSI `.p` I/O and numeric conversion, epsilon/chi physics, the scor160 ATOMIX
benchmark path and its `scripts/compare_atomix.py` harness, instrument-agnostic
processing (trim / bottom-crash / ct_align / mixing), the perturb campaign
pipeline and its post-#85/#86 incremental cache, perturb plotting, the pyturb
CLI, through-water speed / fall-rate, ancillary-data ingestion (hotel / GPS /
CTD / seawater), and documentation self-consistency. It also re-verified the 24
prior-audit findings tracked across the fix PRs. Every itemized finding survived
a refute-by-default verifier; severities are the verifier's recalibration, not
the finder's.

**Overall risk read (publication use): moderate. The headline epsilon and chi on
the in-house ARCTERX/SN479 VMP path remain sound — no confirmed major produces a
wrong epsilon or chi on every run of the validated in-house data. But one
default-path physics defect reaches the *derived mixing coefficients* in the
published product: `K_T` and `Gamma` are computed against the IN-SITU temperature
gradient instead of the conservative/potential-temperature gradient (r1-1),
biasing them by 1.3x-11x (unbounded near cancellation) in weak-gradient warm
water and silently mis-masking valid weakly-stratified windows.** If the
publication uses `K_T` / `Gamma` / turbulent heat flux, r1-1 is a must-fix; if it
uses only epsilon and chi, no confirmed major bites the in-house numbers.

The remaining six majors are each conditional on a non-default feature, input, or
fault and do NOT affect the clean in-house epsilon/chi:
- **top_trim over-trim** (r1-2) and **bottom-crash false positives** (r1-3) are
  silent data-loss defects reachable only when those trimming stages are enabled
  (top_trim IS enabled on ARCTERX; bottom-crash is off by default).
- **Chi QC gate runs after mixing derivation** (r1-4) — QC-rejected chi still
  drives published `K_T`/`Gamma`/`K_rho`; latent until the documented QC-gate is
  configured with non-empty drop lists.
- **Interrupted `binned.nc` write validates as cache-current** (r1-5) — silent
  permanent corruption on re-run, but only after an I/O fault (ENOSPC / SMB drop
  / Ctrl-C).
- **quick_look all-NaN eps/chi panels** (r1-13) — an interactive-QA-viewer
  regression from the PR #75 M-7 fix; no saved product affected.
- **Glide-vehicle speed** (r2-3) — perturb silently uses vertical `|dP/dt|` as
  through-water speed for a Slocum glider (~5x epsilon), a warning-parity gap
  vs the rsi path; the user holds real Slocum MR data, but ARCTERX is a VMP
  (unaffected).

The dominant theme across the minors/nits is **documentation and metadata drift
behind the fix PRs** (12+ doc/code mismatches, several introduced when a code fix
landed but its doc/comment did not follow — e.g. r1-41 for prior #98, r1-24 for
PR #81), **CI/regression-suite blindness** (reference-data tests skip in CI; the
surviving synthetic gates tolerate factor 2-3 errors — r1-9/r1-10/r1-11), and a
cluster of **stale-cache / edge-input** robustness gaps in the new incremental
cache (r1-5/r1-6/r1-7/r1-31/r1-38). Two prior findings are **partially fixed**
(see §3): the chi variance-grid perf residual (#16/#64/#100/#107) and the
bottom-crash half-bin under-trim (#63). None regressed to broken; two NEW
findings are code regressions introduced by earlier PRs (r1-13 from #75, r1-26
from #74).

**Confirmed totals: 0 critical, 7 major, 48 minor, 8 nit (63 findings).**
Machine-readable record: `2026-07-01-deep-audit-findings.json`. Nine refuted
candidates are appended (§6) so future rounds do not rediscover them; zero
findings were left unverified.

### Headline issues (read these first)

1. **`K_T`/`Gamma` fit the in-situ temperature gradient, not the
   conservative-temperature gradient (r1-1, MAJOR, `processing/mixing.py:327`).**
   The single default-path science defect that reaches a published product. In
   ARCTERX 29 C water the adiabatic lapse rate (2.42e-4 K/dbar) is 2.4x the
   1e-4 K/m well-mixed floor, so `K_T` and `Gamma` are overestimated x1.76 at
   true `dCT/dz`=1e-3 K/m, x3.8 at 5e-4, x10.9 at 3.5e-4 (unbounded near the
   cancellation point), and genuinely-stratified windows are mis-masked as
   well-mixed. The *same window* already computes lapse-corrected CT for N² but
   uses in-situ T for `dTdz`, so `Gamma`'s numerator and denominator are
   physically inconsistent. epsilon and chi are unaffected.

2. **Two silent data-loss trimming defects on real casts (r1-2, r1-3, MAJOR).**
   `top_trim` takes the DEEPEST elevated bin as the prop-wash exit with no
   surface-contiguity requirement, so an isolated deep accelerometer transient
   over-trims valid near-surface epsilon/chi on ~5% of real ARCTERX profiles
   (up to +22 m, top_trim IS enabled on ARCTERX). `detect_bottom_crash` flags
   mid-column vibration transients as bottom strikes with no bottom-proximity or
   speed-drop confirmation, truncating up to ~130-178 m of valid deep cast when
   enabled. r1-2 is the unrecorded flip side of the prior #66 fix (§3).

3. **The MATLAB/ATOMIX reference validation does not run in CI and its
   surviving gates are 12-100x looser than actual accuracy (r1-9/r1-10/r1-11,
   MINOR ea.).** `VMP/` and `AtomixData` are gitignored, so all 64
   MATLAB-comparison tests plus both ATOMIX benchmarks skip in CI; the only
   quantitative accuracy gates left are synthetic recovery windows of
   0.5<ratio<2.0 (epsilon) and 0.3<ratio<3.0 (chi). A factor-1.9 epsilon or
   factor-2.9 chi systematic error passes the full CI suite green. The chi
   spectra test is additionally statistically blind in the band chi is fitted
   from (pools 79/128 out-of-band junk bins).

4. **Two code regressions from earlier fix PRs (r1-13, r1-26).** The PR #75 M-7
   quick_look fix built a segment-length temperature grid indexed with absolute
   fast indices, NaN-ing eps/chi for every profile after the first
   (`quick_look.py:223`). PR #74 flipped `convert_accel`'s missing-`adc_bits`
   default from ODAS's 0 to 16 (`channels.py:196`) — reintroducing a divergence
   the 2026-06-19 audit had explicitly refuted.

5. **Incremental-cache durability gaps (r1-5, r1-6, r1-7, r1-31).** The post-#86
   `binned.nc` write stamps its cache-validity manifest into the NetCDF header
   BEFORE the payload and writes direct to the final path (no tmp+os.replace),
   so an interrupted write publishes truncated bins that every later run treats
   as current. The trim cache fingerprints the source AFTER copying; a failed
   per-profile NC is locked in by the successful retry; a shrunk input set leaves
   a stale `combo.nc`.

---

## 2. Prior-Findings Fix-Status (24 tracked)

**22 fixed, 2 partially-fixed, 0 open, 0 regressed.** No tracked prior finding is
broken at HEAD. Four IDs (#20, #95, #98, #33) were tracked in two verification
batches (the merge-blocker group and a minors re-sample) and appear twice below;
both instances confirm the same fix.

> **PARTIALLY-FIXED — read these two.** Neither is dangerous, but each left a
> real residual the fix PR chose not to close:
>
> - **#16/#64/#100/#107 (chi variance-grid, PARTIAL).** The dead `n_fine` floor
>   was removed, but `_PTS_PER_KB` is still 2000, so `_variance_correction`
>   builds a 40*2000 = **80000-point grid unconditionally** (~0.83 ms/call, per
>   window per iteration, no caching). At the audit-suggested 500 pts/kB the
>   correction differs by 4.1e-5 relative (well under the 0.1% tolerance) and
>   runs 3.7x faster. Correctness-neutral perf residual only. See NEW finding
>   r1-15 (kappa_T) for the adjacent chi-accuracy item in the same module.
> - **#63 (bottom-crash under-trim, PARTIAL).** `bottom.py:152-153` still returns
>   the flagged bin's sample mean, which lands near bin center and under-trims by
>   up to ~depth_window/2 (repro at HEAD: +1.45 m early-in-bin, +2.54 m with
>   ring-down samples, slightly beyond the documented ~half-bin bound). PR #79
>   applied only the audit's stated MINIMUM remediation (a docstring note at
>   `bottom.py:74-79`); the stronger shallowest-flagged-sample / sub-bin-onset
>   fix was not implemented. Compounds with NEW finding r1-3 (false positives in
>   the same detector).

| Prior ID | Title (abbrev) | Status | Evidence (file / commit) |
|---|---|---|---|
| 14 | extract_profiles AttributeError on CF/ATOMIX `_FillValue` | fixed | `profile.py:286-287` skips reserved attrs; 93ebb33 (#79); repro extracts finite data; test `test_rsi_profile.py:362-433` |
| 13 | Mid-cast NaN slow-P poisons fall-rate → cast dropped | fixed | `profile.py:141` `_repair_nans` before smooth; 93ebb33; 2000/2000 finite W; tests :403-433,:441-472 |
| 106 | `_load_from_nc` leaks NetCDF handle on error paths | fixed | `profile.py:465-574` try/finally close; 93ebb33; lsof 0 leaks; test :480-506 |
| #42/#101 | Method-1 chi filters obs/Nasmyth fom at the FM limit | fixed | `chi_io.py:360-362` reads FM first; 93ebb33; tests `test_audit_2026_06_26_fixes.py:116-138` |
| #41 | tz-offset strip corrupts Method-1 epsilon time → chi NaN | fixed | `chi_io.py:395-402` offset-aware parse; 93ebb33; round-trip repro no 36000 s shift |
| 66 | compute_trim_depth returned shallowest sub-threshold bin | fixed | `top_trim.py:142-162` deepest-elevated rule; 93ebb33 + 83de66f (#80); **see NEW r1-2 over-trim tail** |
| 63 | Bottom crash-depth mean near bin center → under-trim | **partially-fixed** | `bottom.py:152-153` unchanged (sample mean); only docstring note :74-79; **compounds NEW r1-3** |
| 24/67 | All-NaN-depth RuntimeWarning escapes np.errstate | fixed | `bottom.py:107-108` guard before nanmax; 93ebb33; test :86 promotes warning to error |
| #20 | config_patch `_check_keymap` validates values not keys | fixed | `config_patch.py:156-160`+:358-364 apply-boundary re-validate; 93ebb33; `TestM20KeyInjection` |
| #95 | merge chains casts on file_number adjacency, no time gate | fixed | `merge.py:106-118` `_is_continuous` 5 s tol; 93ebb33; 29 real casts → 0 chains |
| #16/#64/#100/#107 | chi `_variance_correction` 80000-pt grid; dead n_fine floor | **partially-fixed** | dead floor removed but `_PTS_PER_KB`=2000 → 80000 pts unconditionally; 0.83 ms/call perf residual |
| #98 | `_mle_fit_kB` correction span [0,K_max] ≠ obs-var band | fixed | `chi.py:444-452` `K_min=K_fit_low`; 93ebb33; test `test_chi.py:1372`; **doc lag NEW r1-41** |
| #93 | kB=inf admitted by `kB>0` → inf-inf RuntimeWarning | fixed | `chi.py:106-109` `0<kB<inf` guard; 93ebb33; test :1554 |
| #33 | `_iterative_fit` returns chi from stale kB on break | fixed | `chi.py:602-626` recompute chi_obs from final kB; 93ebb33; test :1484 |
| #20 | config_patch edit-spec keys never validated (major) | fixed | `config_patch.py:146-161`+:358-364; 93ebb33; injection repro raises ValueError |
| #95 | merge fuses independent casts, no continuity check (major) | fixed | `merge.py:69-118`; 93ebb33; 29 ARCTERX casts → 0 chains, gaps 17.8 s/9289 s/70458 s rejected |
| #98 | `_mle_fit_kB` span mismatch biases chi low (minor) | fixed | `chi.py:444-452`; 93ebb33; test :1375-1443 |
| #33 | `_iterative_fit` stale kB on convergence break (nit) | fixed | `chi.py:602-626`; 93ebb33; test :1484 |
| #2 | epsilon viscosity ignored salinity/pressure (visc35) | fixed | `l4.py:90,171-203` optional salinity; `pipeline.py:121-143` per-window; 93ebb33; **doc lag NEW r1-43/r2-0** |
| #32 | rsi mixing used unfiltered chi_final, no fom/K_max QC | fixed | `pipeline.py:46-80` `_qc_chi_final`; 93ebb33; **NEW r1-48 one-sided-cut defect in this fix** |
| #23+#105 | ct_align discarded whole segment on one non-finite sample | fixed | `ct_align.py:34-56` `_repair_nonfinite`; 93ebb33; every-profile-one-NaN recovers lag; **cf NEW r2-6** |
| #27+#50 | despike both-empty replacement 0.0 not MATLAB NaN | fixed | `despike.py:195-203` NaN semantics; 93ebb33; test `test_scor160_despike.py:90` |
| #45 | binning `np.arange` edges spurious trailing bin | fixed | `binning.py:441-448` integer edge count; 93ebb33; 0-7 m @0.7 → 10 bins; **cf NEW r1-32 label** |
| #30 | K_rho N2_min floor didn't cap unphysical inflation | fixed | `mixing.py:510-523` `K_rho_max` mask; 93ebb33 (1.0)→292ce35/#81 (10.0); **NEW r1-23/r1-24 residuals** |

---

## 3. New Confirmed Findings — MAJOR (7)

#### r1-1. `K_T`/`Gamma` fit in-situ temperature, not conservative temperature — 1.3x-11x bias in warm weak-gradient water
- **Subsystem:** processing-mixing — `src/odas_tpw/processing/mixing.py:327` (and `window_stratification` :203); consumed by `rsi/pipeline.py:521-546` and `perturb/pipeline.py:750,885`
- **Mechanism:** `_stable_window` computes an adiabatically-correct conservative temperature `CT_stable` (mixing.py:324) and uses it for N² (:334-335), but fits `dTdz = polyfit(z_stable, T_stable)` on the **in-situ** temperature. `K_T = chi/(2 dTdz²)` and `Gamma = N2·chi/(2 eps dTdz²)` therefore mix a lapse-corrected numerator (N²) with an uncorrected denominator within one window. Osborn & Cox (1972) variance production is against the conserved-scalar gradient d(theta)/dz.
- **Quantified impact:** gsw at Saipan (SP=34.5, lat 15.2, T=29 C, P=10) gives adiabatic lapse 2.416e-4 K/dbar — 2.4x `DEFAULT_DTDZ_MIN`=1e-4 (mixing.py:79). K_T/Gamma bias `(dCT/dT_insitu)²`: x1.10 at true `dCT/dz`=-5e-3, x1.74 at -1e-3, x3.74 at -5e-4, x10.4 at -3.5e-4, unbounded near -2.4e-4 (in-situ slope cancels → window MASKED though stratified), x0.45 for a +5e-4 inversion. Default path of BOTH pipelines; ~10-30% even in moderate thermoclines. epsilon and chi themselves unaffected.
- **Fix:** fit the gradient of the already-computed conservative temperature in `_stable_window` and `window_stratification`; update `dTdz` comments/docs; add a uniform-CT warm-column regression asserting `|dTdz| < DEFAULT_DTDZ_MIN`.

#### r1-2. `top_trim` deepest-elevated-bin rule over-trims valid near-surface data on ~5% of real ARCTERX casts
- **Subsystem:** processing-trim — `src/odas_tpw/processing/top_trim.py:157`; caller `perturb/pipeline.py:512-521`
- **Mechanism:** `compute_trim_depth` takes `deepest = elevated_idx[-1]` (the deepest bin with std > noise_factor·background) as the prop-wash exit, with **no surface-contiguity requirement**. An isolated deep accelerometer transient (cable snap-load, 10-78x background on both Ax and Ay, so the median combine cannot reject it) inside the search range trims the entire quiet band above it.
- **Quantified impact:** with shipped campaign settings (top_trim.enable=true, max_depth=25, quantile=0.6) on the 29 real ARCTERX `.p` files: SN479_0026 prof 8 — quiet 3-24.5 dbar, isolated transient at the 25.0 m search-edge bin → trim pushed 1.14 → 25.01 dbar, discarding settled 2.75-24.5 m water while retaining contaminated 25-33 m. Campaign survey (464 profiles): excess over contiguous-surface exit p95=5.25 m, max=22.25 m; 24/464 (5.2%) over-trim >5 m, 4 over 15 m, ~598 m discarded. This heavy over-trim tail is the **unrecorded flip side of the prior #66 fix** (§2), which recorded only the opposite under-trim.
- **Fix:** locate the prop-wash exit as the end of the elevated run CONTIGUOUS with the surface (allow gaps of ~2-4 bins, still fixing the #66 momentary-lull case); log isolated deep elevated bins separately as mid-column contamination.

#### r1-3. `detect_bottom_crash` flags mid-column vibration transients as bottom strikes (no proximity/speed confirmation) → truncates up to ~178 m of valid deep cast
- **Subsystem:** processing-trim — `src/odas_tpw/processing/bottom.py:141-153`; caller `perturb/pipeline.py:550-563`
- **Mechanism:** returns the deepest depth-bin whose accel-magnitude std exceeds vibration_factor·median (default 4.0), with **no check that the flagged bin is near the cast bottom** and with the MATLAB speed-drop confirmation explicitly unimplemented (bottom.py:52-55). A synthetic cast descending smoothly to 234 dbar with one mid-column vibration burst at ~100 dbar returns 99.9998 dbar; the caller's `where(P_seg >= bottom_depth)` is non-empty and truncates ~134 m, logging only INFO.
- **Quantified impact:** finder's real-data survey (defaults, {Ax,Ay}, fs=512) reported 6/12 detections false (11.9-140 dbar flags on casts descending to 189-234 dbar). The specific VMP files are absent from this checkout so those exact numbers are unverified, but the mechanism reproduces and is reachable whenever bottom-crash trimming is enabled. Off the default path (`bottom.enable=false`), but the shipped yaml documents enabling it for suspected strikes; **compounds the prior #63 under-trim residual** (§2) in the same detector.
- **Fix:** require the flagged bin within ~2 bins of the deepest populated bin and/or implement the documented speed_factor confirmation; warn (not INFO) when the flagged depth is far above `nanmax(depth)`.

#### r1-4. Chi QC gate runs AFTER mixing-quantity derivation and never masks `K_T`/`Gamma`/`K_rho`
- **Subsystem:** perturb-pipeline — `src/odas_tpw/perturb/pipeline.py:1518` (order 1490→1495-1512→1518→1533)
- **Mechanism:** `mk_chi_mean` → `_add_mixing_quantities` computes `K_T`/`Gamma`/`K_rho` from the **pre-gate** `chiMean` → `apply_qc_to_dataset` with `value_vars=['chiMean','chiLnSigma','chi_1','chi_2','chi','epsilon_T']` (omits the three mixing vars) → `to_netcdf`. `qc_gate.py:162-176` NaNs only listed vars. `binning.py:250-272` then bins every 1-D numeric var, so the un-gated mixing coefficients leak into `chi_binned`/`chi_combo` while `chiMean` correctly excludes the flagged segments. The epsilon side is ordered correctly (QC at :1388-1399 precedes the diss write).
- **Quantified impact:** repro (flagged segments 1-4) → `chiMean=[nan,nan,nan,nan]` but `K_T=1.98e-6`, `Gamma=0.81`, `K_rho=4.9e-7` all finite and binned. Latent on current ARCTERX yaml (qc.enable=true but drop lists empty); fires on any configured use of the documented QC-gate. Distinct from prior #32 (rsi path + fom/K_max_ratio QC).
- **Fix:** move `apply_qc_to_dataset` before `_add_mixing_quantities`, or append `K_T`/`Gamma`/`K_rho` to the chi `value_vars`; add a regression asserting a QC-flagged segment is NaN in ALL chi-derived variables.

#### r1-5. Interrupted `binned.nc` write leaves a partial file that validates as cache-current (manifest before payload, no atomic write)
- **Subsystem:** perturb-cache — `src/odas_tpw/perturb/pipeline.py:117-119` (`_write_binned_or_clear`)
- **Mechanism:** `ds.assign_attrs(_input_manifest=manifest); ds.to_netcdf(out)` writes the cache-validity marker into the NetCDF header BEFORE the data payload completes, direct to the final path (no tmp+os.replace, unlike `trim.py:283-293` and `_write_marker` :333-335). Skip checks at :2252/:2273/:2290 compare only that attribute.
- **Quantified impact:** repro injecting `OSError(ENOSPC)` on the 2nd variable write left a readable 14144-B `binned.nc` with `_output_is_current=True`, epsilon 1000/1000 finite, chi 0/1000 (all fill). Every later run then logs "up to date (skipped)" and combo rebuilds from the truncated bins — silent permanent corruption. Requires an I/O fault (ENOSPC / SMB drop / Ctrl-C), realistic given the explicit exFAT/SMB/network-volume target (PR #86).
- **Fix:** write `binned.nc` to a temp file and `os.replace()` it in, and/or stamp `_input_manifest` only after `to_netcdf` returns (mirroring the combo path's `_stamp_manifest`).

#### r1-13. quick_look windowed eps/chi uses a segment-length T grid indexed with absolute fast indices → all-NaN panels after the first profile (regression from PR #75 M-7)
- **Subsystem:** rsi-viewer — `src/odas_tpw/rsi/quick_look.py:223` (grid built `_interp_slow_to_fast(T_slow, N)` with N=seg_end-seg_start, sliced with `s=seg_start+idx*step`)
- **Mechanism:** the temperature grid is built at segment length but sliced with absolute fast indices, so windows with `s >= N` get an empty slice → `mean_T=NaN` → `nu=visc35(NaN)=NaN` → NaN epsilon/chi; shallower windows compress the whole record onto the segment, biasing per-window viscosity. Correct pattern at `viewer_base.py:285` uses `len(P_fast)`.
- **Quantified impact:** on `tests/data/SN479_0006.p` (seg_start=49632, N=103352): finite eps 54/98 vs 98/98 with the correct grid (deeper 22 windows × 2 probes silently NaN); synthetic 25→10 C record biases first-window `mean_T` by -2.3 C. `git log -L 223` confirms the line was introduced by commit 3b7e467 (PR #75, the M-7 fix), postdating the recorded audits. Interactive QA viewer only — no saved product affected — but silent and always-on.
- **Fix:** `T_fast = _interp_slow_to_fast(T_slow, len(P_fast))` matching `viewer_base:285`; add a viewer regression for a profile with `sel.start > 0`.

#### r2-3. perturb silently uses vertical `|dP/dt|` as through-water speed for glide vehicles at the default `speed.method='pressure'` → ~5x epsilon on real Slocum data
- **Subsystem:** rsi-speed — `src/odas_tpw/rsi/speed.py:91-96`; injected by `perturb/pipeline.py:1073-1081`
- **Mechanism:** the `method=='pressure'` branch of `compute_speed_for_pfile` returns `|dP/dt|` unconditionally and never checks vehicle direction (`vehicle` is used only to resolve tau, :80). Because perturb precomputes `speed_fast`, `prepare_profiles` takes the precomputed branch (`helpers.py:292-293`) and never reaches its own glide warning (`helpers.py:303-309`) — the codebase's only warning for this case.
- **Quantified impact:** on real `MR/AIOP2_SL685_0450.p` (slocum_glider, 98.8% `|Incl_Y|>25 deg`): median `|dP/dt|`=0.291 vs EM-flowmeter through-water 0.428 m/s → 32.0% underestimate → `0.680^-4.3` = 5.26x epsilon inflation, with zero warnings. ARCTERX VMP is unaffected (vertical = through-water). The user holds ~30 GB of real Slocum MR data, so a realistic input; documentation (speed.py:14) is a mitigant, not a fix.
- **Fix:** in `compute_speed_for_pfile`, resolve the vehicle direction and, when glide/horizontal + method resolves to `pressure` with no explicit speed, emit the same warning `prepare_profiles` already issues (or auto-select the vehicle's ODAS speed_algorithm: glide→flight, emc→em).

---

## 4. New Confirmed Findings — MINOR (48)

Compact entries; each verified (reproduced where marked). File:line, mechanism +
quantified impact where measured, and fix.

#### r1-6. Bin/combo manifest blind to per-file NC content and per-file failure
`perturb/pipeline.py:2100-2101,133-142`. The `_inputs_manifest` hashes only `[nc.name, cachekey]` (a source-`.p` identity hash) set for EVERY file regardless of success; the marker is withheld on error but binning runs unconditionally over the glob and stamps `binned.nc`. A soft-ENOSPC partial diss NC that stays openable is binned in run 1 (epsilonMean nanmean 2.11e-7); the retry rewrites it completely under the same name → byte-identical manifest → re-bin skipped, publishing the stale bin (correct 3.4e-7). Bounded: the common truncation leaves an unopenable HDF5 that crashes binning loudly. **Fix:** unlink the partial NC in the per-profile except handler and/or fold a per-run nonce into failed-stem cache keys.

#### r1-7. Trim cache fingerprints the source AFTER the trim copy (TOCTOU)
`perturb/trim.py:135`. `_trim_cache_store` stats the source at :135, after the size used for the trim decision was read (:242-243) and after the copy (:283-293). An append landing in that window makes the cached fingerprint describe a state never trimmed to the stored dest; repro: run 1 trims to 960 B while source grows to 1316, run 2 with cache returns skipped/960 (record 4 permanently dropped) vs 1216 uncached. Requires a concurrent append during the copy AND the source freezing at the fingerprinted size next run; rsync uses temp+rename so does not trigger it. **Fix:** capture `_trim_fingerprint(source)` at the top of `trim_p_file` and pass the snapshot to `_trim_cache_store`.

#### r1-8. ARCHITECTURE.md falsely claims the rsi and perturb 2-probe epsilon combines are "mathematically identical"
`docs/ARCHITECTURE.md:423-426,431,446`. At HEAD `epsilon_combine.py:135-136` gates the CI-removal loop on `n_probes>=3`, so perturb ALWAYS keeps both probes for the 2-probe VMP-250, while `l4.py:766-773` applies the bit-4 flag and geomeans only flag==0 probes. Repro (probes {1e-9,1e-7}, ln-ratio 4.61 > threshold 0.62): rsi `epsilon_final=1e-9` vs perturb `epsilonMean=1e-8` → 10x. The doc text describes pre-#79 behavior. The underlying divergence duplicates 06-25 audit #121, but that finding cited the code, not ARCHITECTURE.md. **Fix:** rewrite the three sentences; for 2 probes perturb keeps both while rsi's bit-4 flag drops the larger.

#### r1-9. Chi-vs-MATLAB spectra test is statistically blind in the band chi is fitted from
`tests/test_matlab_chi.py:218` (clone `test_pipeline_vs_matlab.py:561`). The median `|log10(py/ml)|` pools ALL 128 bins with no `f_AA` mask: on ARCTERX_...0002 (20 profiles) full-range median 0.306 (near the 0.5 gate) but in-band (<=98 Hz) only 0.043. Injecting an in-band-only error: x2.0→0.315, x3.0→0.465, x0.5→0.414 all PASS. No test compares the chi VALUE to an independent reference. Real test defect; in-band agreement is excellent so no wrong published number. **Fix:** restrict to `0<F<=f_AA`, tighten to ~0.1 decades, add an independent chi-value reference.

#### r1-10. Epsilon-vs-MATLAB magnitude gate is one full decade vs 0.01-0.05-decade actual agreement
`tests/test_pipeline_vs_matlab.py:421`, `test_matlab_epsilon.py:218`. The only epsilon-magnitude assertion is `median|log10(py/ml)|<1.0`; every other check is corrcoef-of-logs (scale-invariant). Measured per-probe-profile agreement on files 0002/0003/0004 is 0.011-0.050 decades, so a multiplicative epsilon error up to ~9x passes this gate. The synthetic Nasmyth recovery test (0.5<ratio<2.0) narrows the full-suite escape to ~1.9x. **Fix:** tighten the gate to ~0.05-0.1 decades and add a signed-bias assertion.

#### r1-11. CI runs zero reference-data validation; surviving accuracy gates tolerate factor 2-3 errors
`tests/test_scor160_l4.py:122` et al. `VMP/` and `AtomixData` are gitignored, so all 64 MATLAB-comparison tests and both ATOMIX benchmarks skip in CI (76 skips at HEAD); `.github/workflows/ci.yml` runs plain pytest on a bare checkout. The only quantitative gates left are synthetic recovery windows 0.5<ratio<2.0 (epsilon, `test_dissipation.py:80`) and 0.3<ratio<3.0 (chi, `test_chi.py:457`); measured recovery is ~1.0, so a 1.9x epsilon or 2.9x chi shift passes green. **Fix:** tighten synthetic windows to ±15-20%; add one tracked golden-reference NC for `tests/data/SN479_0006.p`.

#### r1-12. QC-dropped/missing eps/chi bins repainted by `ffill_down` in eps-chi and profiles figures
`perturb/plot/layout.py:134,234`. Both pcolor kernels forward-fill every internal NaN down each cast column AFTER QC masking, so `--apply-qc`-dropped internal bins render with the shallower bin's value under a "QC applied" title. Repro: cast with bins 2-5 flagged prints "NaN 4 cells" yet the QuadMesh draws 4/4 with the copied value; contradicts the `plot_columns` set_bad docstring (:229-230) and `docs/perturb/plotting.md:219`. Saved NetCDFs untouched; only internal drops sandwiched between valid bins repainted (contiguous end drops stay masked). **Fix:** re-apply the pre-fill mask after `ffill_down`, or bound the fill to <=1 bin.

#### r1-15. Constant `kappa_T`=1.4e-7 ignores its temperature dependence → chi ~6.5% low, Method-2 epsilon_T ~12.6% low in 28 C water
`chi/batchelor.py:23`, used at every chi/epsilon_T site (`chi.py:263,436,456,...`; `l4_chi.py:245`) with NO T-dependent call. Computed `kappa_T=k/(rho·cp)` at S=34.5,p=10: 1.389e-7 at 0 C, 1.498e-7 at 28 C (+7.0%). chi is linear in `KAPPA_T` (-6.5% at 28 C); `epsilon_T=(2π·kB)⁴·nu·KAPPA_T²` quadratic (-12.6%). Internal inconsistency: the same pipeline resolves `nu=visc(T,S,P)` but freezes `kappa_T`. Documented convention (`chi_mathematics.md:567`), 6.5% within microstructure uncertainty. **Fix:** thread per-window `kappa_T(T)` through `batchelor_kB`/the 6·kappa_T integrals/epsilon_T as `nu` already is, or document the +7%/-13% envelope.

#### r1-16. Signal-free windows return hard-coded floor values as finite chi (1e-14 sentinel; 0.01·chi_vc grid edge)
`chi/chi.py:600,636` and :268-270. When every band bin is at/below the modeled noise floor, `_iterative_fit` returns `chi=1e-14` with `kB` pegged at the fine-grid ceiling (63245.55, so the kB=NaN guard is bypassed) and nonsense `epsilon_T`~489-587 W/kg; `_chi_from_epsilon` pins `chi=0.01·chi_vc` exactly with `fom=0.794`. `_compute_chi_final` (l4_chi.py:286-294) has no gate → probe pair {1e-9,1e-14} → `chi_final=3.16e-12`, 316x below the good probe. Bounded: perturb `chiMean` IS protected (`chi_combine.py:100` masks chi<=1e-13); only the secondary rsi `chi_final` is contaminated. **Fix:** return NaN when `chi_band<=0` and when the Method-1 grid search terminates on a grid edge.

#### r1-17. `compare_atomix.py` falls back to `HP_cut` as the anti-aliasing frequency and omits the ODAS 0.9 margin
`scripts/compare_atomix.py:282,400`. `f_AA=attrs.get('f_AA', attrs.get('HP_cut', 98.0))`: for MSS_Baltic.nc (`f_AA=None, HP_cut=0.15`) the harness uses a 0.15 Hz high-pass as the anti-alias limit, capping K_AA at 7.6 cpm and misreporting RMSD 0.192 vs the correct 0.008 (median eps ratio 1.070); contradicts `atomix_benchmark.md:230` (RMS 0.007-0.009). Also omits the `0.9·f_AA` margin production applies (`l4.py:213`), so the harness never exercises the production epsilon config. Benchmark harness only. **Fix:** never treat `HP_cut` as `f_AA`; apply `K_AA=0.9·f_AA/W`; fix the report text (:717 says `rsi.dissipation`, import is `scor160.l4`).

#### r1-19. `compare_atomix.py` FOM plot compares two different, anti-correlated statistics on a 1:1 axis
`scripts/compare_atomix.py:619-631`. Scatters `fom_atomix` (Lueck-2022 MAD FOM) against `fom_rsi` (observed/Nasmyth variance ratio) with a 1:1 line and 1.15 thresholds. On VMP250_TidalChannel_024.nc: medians 0.805 vs 1.010, Pearson r=-0.53 (anti-correlated). The harness never computes the comparable FM (`_estimate_epsilon` called without `num_ffts`, :437-444 → FM NaN). `scor160/compare.py:352-356` pairs them correctly. **Fix:** pass `num_ffts` and plot FM vs `fom_atomix`, or retitle and drop the 1:1 line.

#### r1-20. epsilon_mathematics.md says the shear probe attenuates LOW wavenumbers — Macoun-Lueck physics is backwards
`docs/epsilon_mathematics.md:264`. Finite probe size spatially averages small scales and attenuates HIGH wavenumbers; the correction `1+(k/48)²` in the same section grows with k (factor 2 at 48 cpm, :279) and the companion `atomix_benchmark.md:139-141` states the correct direction. Code (`l3.py:_apply_macoun_lueck`) is correct; only the doc sentence is inverted. **Fix:** "...spatially averages the velocity field, attenuating the measured shear at high wavenumbers (small scales)."

#### r1-22. epsilon_mathematics.md says singular accelerometer bins retain the original spectrum; code NaNs them
`docs/epsilon_mathematics.md:249` vs `scor160/goodman.py:189-199,45-71`. The code NaNs a singular AA[f] bin (`cond>1e13`) on every path (comment: "rather than passing the UNCLEANED raw UU through"); `l4.py:373-386` then interpolates-or-rejects. Retaining raw (as the doc says) would bias epsilon high in that bin. An exactly rank-deficient AA essentially never occurs on real 3-axis data, so behaviorally latent, but a reader implementing from the doc gets the worse fallback. **Fix:** update step 2 to state singular bins are set to NaN and excluded.

#### r1-23. `K_rho_max` ceiling is one-sided QC: identical `K_T` survives above 10 m²/s and `Gamma` is unbounded (500 emitted)
`processing/mixing.py:504-518`. Masks only `K_rho>K_rho_max`(10); by the module's own identity `K_T==Gamma·eps/N2` (:17-23) the exact windows PR #81 masks publish an equally-implausible `K_T` (repro: eps=4e-3,N2=7e-5 → K_rho NaN but K_T=11.43 finite), and `Gamma` has no plausibility bound (repro: Gamma=500 emitted silently). Formula-correct derived diagnostics, easily filtered, no eps/chi corruption. Not covered by prior #30 (K_rho-only) or #32. **Fix:** mask `K_T`/`Gamma` at the same windows (or add `K_T_max`), or add per-window QC-flag variables.

#### r1-24. `K_rho` NetCDF comment attrs and mixing_efficiency.md still attribute the >10 m²/s mask solely to "near-floor N2"
`rsi/pipeline.py:617`, `perturb/pipeline.py:927`, `docs/mixing_efficiency.md:68-69`. `git show 292ce35` (PR #81) changed only "K_rho > 1" → "> 10" in the published-file comment attrs; the "from near-floor N2" cause clause is unchanged, yet the same commit's message states the actually-masked 14 windows are extreme near-surface epsilon (N2~7e-5), and it DID update the code comment `mixing.py:71` to two causes. Published metadata now names the wrong cause, internally inconsistent with the code comment. No numeric effect. **Fix:** mirror `mixing.py` ("unbounded near-floor-N2 artifact OR contaminated near-surface epsilon") in both attrs and the doc. Cross-ref r1-23 (same ceiling).

#### r1-25. `PFile.start_time` omits the RSI `recsize` subtraction → every absolute timestamp is +1.000 s late vs ODAS
`rsi/p_file.py:359-377`. ODAS defines the data start as the record-0 header time MINUS `recsize` (`odas_p2mat.m:416-417`; `setupstr.m:523-525` injects default recsize=1 when absent). Python subtracts nothing; on `SN479_0006.p` (config omits recsize) ODAS would label the first sample 1.000 s earlier. Propagates to L1 time coords (`convert.py:454-461`), per-profile time, and any absolute-time GPS/CTD/hotel merge (~1 m at ship drift). Depth/pressure-indexed products (eps/chi/K/N2) unaffected. Distinct from #41/#95. **Fix:** read recsize from `config['root']` (default 1.0) and apply `start_time -= recsize`; pin `start_time` against the MATLAB `_allch.nc` attrs.

#### r1-26. `convert_accel` `adc_bits` default changed to 16, diverging from ODAS's 0 — a PR #74 regression against a refuted claim
`rsi/channels.py:196`. ODAS `odas_accel_internal` (`convert_odas.m:424`) defaults a missing `adc_bits` to 0 (counts-based coef0/coef1). Python defaults 16 (÷65536). Commit 579ce0f (PR #74) flipped it, even though the 2026-06-19 audit explicitly PARTIALLY REFUTED this exact case ("the Python port's adc_bits default of 0 for accel faithfully matches ODAS"). Repro (counts [1000,-2000,15000], {coef0:-100,coef1:1500}): Python [0.654,0.654,0.655] vs ODAS [7.19,-12.43,98.75], ratios 11x-151x with a sign flip. Latent for ARCTERX (SN479 uses piezo → `convert_piezo`); affects only legacy accel-type configs omitting adc_bits; warns loudly. **Fix:** restore `adc_bits`/`adc_fs` defaults 0/1 for accel; add a no-adc-params parity test.

#### r1-27. PFile never checks the per-record bad-buffer flag (header word 16) that ODAS mandatorily screens and repairs
`rsi/p_file.py:426`. Stores `_record_headers` but never inspects word 16; ODAS `odas_p2mat.m:348` unconditionally runs `check_bad_buffers` (`header(:,16)~=0`) and patches flagged records via patch_odas/fix_bad_buffers BEFORE conversion. Scan of all 29 ARCTERX files: 0 flagged (latent for this campaign), but a DAQ dropout would silently feed corrupt samples into eps/chi where ODAS detects and repairs. **Fix:** warn on nonzero word-16 counts (optionally NaN affected scan ranges); add a synthetic-`.p` test.

#### r1-28. `compute_trim_depth` exit bin can lie below the deepest sample → caller applies NO trim to a never-settled cast
`processing/top_trim.py:161`. When the deepest populated search bin is elevated (cast ends inside the search range), `exit_idx=deepest+1` is an unpopulated bin center below every sample; the caller's `where(P_seg>=trim_depth)` is empty → no trim, surface wash retained. Repro (default max_depth=50): cast 0.2-28 m with wash <4 m + transient >27.6 m → returns 28.5 m > cast max, no trim applied. Same geometry class as prior M4 but in `top_trim.py` (M4 was fixed in `bottom.py`). Unreachable on in-house data (all casts >74.9 dbar > max_depth). **Fix:** clamp the returned trim depth to the deepest observed sample; caller treats empty `above` + non-None trim as "never settled".

#### r1-29. `top_trim` docstring claims median robustness to one bad channel, but the production caller feeds exactly two channels (median = mean)
`processing/top_trim.py:66-67`, caller `perturb/pipeline.py:509` (Ax,Ay only). `np.median([a,b])==(a+b)/2`, so one misbehaving-but-live channel drags the trim halfway, and the abstention rule lets a single barely-elevated channel alone set the trim. Robustness tests use 3 channels (unreachable in production). Only the flat/dead-channel special case matches the docstring. **Fix:** reword to the two-channel reality, or make the combine robust for n=2 (min exit on large disagreement / require both channels to confirm deeper bins).

#### r1-30. perturb profile detection smooths fall-rate with the fixed default tau=1.5, not the vehicle-resolved tau
`perturb/pipeline.py:1140`. `_smooth_fall_rate(P_slow, fs_slow)` passes no tau → default 1.5, while direction IS vehicle-resolved 6 lines later and the speed channel uses `resolve_tau(vehicle)` (`speed.py:80`); ODAS smooths detection W with the vehicle tau (`odas_p2mat.m:696`). Non-VMP vehicles (glider 3.0, argo 60.0) get a 2-40x too-high cutoff → noisier W, more fragmented boundaries. Same gap at `rsi/profile.py:146`; `_VEHICLE_TAU` (:42) is dead code. Zero effect on ARCTERX VMP. **Fix:** pass `tau=resolve_tau(vehicle)` (or reuse the injected vehicle-tau W_slow); delete the dead alias.

#### r1-31. Stale `combo.nc` (and `binned.nc`) republished when a re-run's input set yields no binned data
`perturb/pipeline.py:2398-2406`. The #56 fix unlinks `binned.nc` on an empty ds, but `_run_combo` only writes/stamps when `make_combo` returns non-None and never deletes `dst/'combo.nc'`. Repro: run 1 writes combo.nc (2 profiles), run 2 with an empty dataset unlinks binned.nc but combo.nc still holds run-1 data. Dirs are config-hashed (reused across shrinking input sets). **Fix:** unlink `dst/'combo.nc'` when src yields no input NCs; clear both on the zero-input branch.

#### r1-32. perturb combo `bin` coordinate holds pressure (dbar) but is labeled depth in metres
`perturb/netcdf_schema.py:44-51`. Binning uses `_DEPTH_CANDIDATES=('depth','P','P_mean')` and the per-profile NC has no `depth` var — P is dbar/`sea_water_pressure`. Inspected real `combo.nc`: `bin` values 1.5..879.5 stamped `units='m', standard_name='depth'` + geospatial EPSG:5831, a systematic ~0.7-1% mislabel (500 dbar=496.48 m). rsi `binning.py:183-190` labels the identical coord `dbar`. Self-acknowledged: `profiles.py:18-21` calls it "a known mislabel" and labels its axis "Pressure (dbar)" while `eps_chi.py:439`/`scalar.py:223` label "Depth (m)". Merges two candidate findings. **Fix:** relabel the schema/geospatial attrs `dbar`/`sea_water_pressure` (or convert P→depth via `gsw.z_from_p` before binning); unify the plot axis labels.

#### r1-33. CHI_SCHEMA omits headline `chiMean`/`chiLnSigma` (published with no units) while 8+ entries name per-probe vars no code creates
`perturb/netcdf_schema.py:296-367`. Binning strips all var attrs and `apply_schema` is the sole attrs source, but CHI_SCHEMA lacks `chiMean`/`chiLnSigma` → published without units/long_name (repro: `make_combo` gives `chiMean units=None`). Meanwhile CHI_SCHEMA's `kB_1/2`, `fom_T_1/2`, `K_max_T_1/2` and COMBO_SCHEMA's `FM_1/2`, `K_max_1/2` describe per-probe vars never created (the base fom/FM/K_max/kB are 2-D and dropped by the 1-D binning skip, `binning.py:254`), so combos carry no per-probe QC. **Fix:** add `chiMean`/`chiLnSigma`/`qc_drop_*` entries; delete the dead per-probe entries or flatten QC to 1-D companions; assert every combo data_var has units.

#### r1-34. `epsilon.diagnostics` / `chi.diagnostics` / `profiles.diagnostics` are documented no-ops
`perturb/pipeline.py:1376,1431`. `process_file` strips `diagnostics` from the eps/chi kwargs, never reads `profiles.diagnostics`, and the compute functions have no such parameter (grep empty in `dissipation.py`/`chi_io.py`/`profile.py`); only `ctd.diagnostics` (:1241) and `binning.diagnostics` (:2176) are honored. Yet `config.py:556,607,635` and `configuration.md:81,167,191` promise specific diagnostic variables. Setting these keys silently does nothing. **Fix:** implement the outputs or mark the flags reserved/unimplemented (as the unused `bottom.*` keys are).

#### r1-35. Chi Method 2 (use_epsilon=false, chosen because shear epsilon is distrusted) still builds published `Gamma`/`K_rho` from that shear `epsilonMean`
`perturb/pipeline.py:1496-1502`. `diss_ds=None` for Method 2, but mixing re-opens `diss_path` and pairs its shear `epsilonMean` into `Gamma=N2·chi/(2·eps·dTdz²)` and `K_rho=0.2·eps/N2` (`mixing.py:507,510`), so the mixing coefficients are built from the epsilon `config.py:619-621` declares unreliable, with no distinguishing attribute. Not the default path (ARCTERX inherits use_epsilon=true); K_rho-from-shear-epsilon (Osborn) is standard, so debatable. **Fix:** derive mixing epsilon from the chi product's `epsilon_T` for use_epsilon=false, or record the shear-epsilon provenance and warn.

#### r1-36. ARCTERX config comments misstate the default dissipation window (2x/3x fft_length vs actual 4x)
`ARCTERX/perturb.yaml:76,93`. Annotates `diss_length: null` as "= 2 * fft_length" (epsilon) and "= 3 * fft_length" (chi); the code default is 4x everywhere (`dissipation.py:101`, `chi_io.py:68`, `pipeline.py:1343,1346`; template `config.py:589,612` correctly says 4x). So the epsilon window is 2 s (not the commented 1 s) and chi 4 s (not 3 s) — misstating vertical resolution, num_ffts, and spectral dof for a methods reader. `null` still resolves to 4x in code, so no wrong number. **Fix:** correct both YAML comments to "= 4 * fft_length".

#### r1-37. docs/perturb/pipeline.md lists the per-file sub-stages in an order that contradicts the code
`docs/perturb/pipeline.md:39-45`. Numbers them 2.CTD binning, 3.Profile extraction, 4.FP07 calibration, 5.CT alignment — implying CTD salinity uses unaligned conductivity and per-profile NCs carry factory FP07 cal. Actual `process_file` order runs CT alignment BEFORE the CTD fork (`pipeline.py:1164-1178`, comment confirms) and FP07 calibration BEFORE `extract_profiles` (:1266-1288). Line 128 also claims trim/bin "run serially" while both use `jobs=`. **Fix:** reorder the sub-stage list to match the code; fix line 128.

#### r1-38. pandas missing from the engine-fingerprint dependency set
`perturb/config.py:268`. `_ENGINE_DEPS=('numpy','scipy','gsw','netCDF4','xarray')` omits pandas, though its own comment says these are "deps whose version can change eps/chi/N2 outputs". pandas parses GPS (`gps.py:214` → lat/lon → gsw latitude in N2) and hotel (`hotel.py:177/186/285` → time alignment → hotel speed → epsilon scaling). A pandas 2.x datetime-parsing change would shift alignment yet reuse stale cached outputs. Narrow (upgrade + cache reuse). **Fix:** add `pandas` (and consider `ruamel.yaml`) to `_ENGINE_DEPS`.

#### r1-40. atomix_benchmark.md claims "Our flag system doesn't include the diss_ratio flag" — stale; bit 4 is implemented
`docs/atomix_benchmark.md:322`. bit 4 IS implemented at HEAD (`l4.py:78,766-773`; documented `epsilon_mathematics.md:568`). Git ordering: the doc last modified abb1930 (#54), diss_ratio added d9781b5 (#56), doc untouched since — so the present-tense sentence is false. The attributed 1.41 EPSI_FINAL number may be a valid pre-flag snapshot. **Fix:** update the Nemo paragraph; rerun the comparison or mark the 1.41 historical.

#### r1-41. chi_mathematics.md §7 still documents the pre-#98 V_resolved band (integral from 0)
`docs/chi_mathematics.md:410`. Writes "V_resolved = integral_0^{K_max_fit}", but the code (post audit-fix #98, §2) passes `K_min=K_fit_low` (`chi.py:444-453`) precisely because the 0-based band biased chi low; a reader re-implementing from the doc reproduces the pre-fix bias. Line 386's "first wavenumber bin above zero" also misstates the above-2×-noise fit mask. This is the doc lag of the fixed prior #98. **Fix:** change the lower limit to `K_fit_low` (noting it matches the obs_var band); reword line 386.

#### r1-42. epsilon_mathematics.md claims the polynomial fit order "is decreased until one is found" — no such retry exists
`docs/epsilon_mathematics.md:369`. `l4.py:522-538` does exactly ONE `np.polyfit` at `fit_order_eff=min(max(fit_order,3),8)` and falls straight to `K_95` when no qualifying root exists; the order-decreasing retry is ODAS behavior, not this port. Default fit_order=3 unaffected; users setting 4-8 get behavior differing from the doc. **Fix:** reword to "single polynomial fit, order clamped 3-8; fallback min(K_95,K_AA)".

#### r1-43. chi_mathematics.md says chi-window viscosity uses `visc35(T_mean)` — default pipeline uses `visc(T,S_measured,P)`
`docs/chi_mathematics.md:556`. On the default rsi-tpw path `_resolve_salinity` returns per-sample measured JAC C/T salinity first (`pipeline.py:114-115,419`) and `l3_chi.py:178-192` uses `visc(T,S,P)` whenever salinity is not None; `visc35` is only the no-conductivity fallback. ARCTERX carries JAC_C/JAC_T (measured branch = production default). Doc lag of the fixed prior #2. **Fix:** state `visc(T_mean,S,P_mean)` when salinity is available, falling back to `visc35` otherwise.

#### r1-44. `rsi-tpw pipeline --salinity` is silently ignored on conductivity-equipped instruments
`docs/rsi-tpw/cli.md:248` and CLI help. Help reads "Salinity [PSU] for viscosity (default: 35, fixed S)", but `_resolve_salinity` (`pipeline.py:114-118`) prefers measured JAC C/T salinity BEFORE the user value with no warning, so on the campaign VMP-250 a user's `--salinity` has no effect on eps/chi/N2 viscosity. The mixing subcommand's own help (`cli.py:1157`) correctly states the precedence; the pipeline/eps/chi help is stale. **Fix:** document the precedence in help/cli.md, or invert precedence so an explicit value wins with a warning.

#### r1-45. CLI reference tables omit the shipped `rsi-tpw ml` mixing viewer
`docs/rsi-tpw/cli.md:11-24`, `docs/DATA_FLOW.md:16-27`. `rsi-tpw --help` registers 13 subcommands including `ml` (backed by `mixing_look.py`); cli.md lists 12 with no `ml` row/section, DATA_FLOW.md lists 10 and also omits `patch-config`/`patch-template`. The N2/dTdz/K_T/Gamma/K_rho viewer is undiscoverable from the reference docs. **Fix:** add the `ml` rows (and a flags section); add the missing subcommands to DATA_FLOW.md.

#### r1-46. MLE chi fit test uses a physically inconsistent fixture (no FP07 rolloff, but H² handed to the fitter) and never asserts chi
`tests/test_chi.py:479-493`. Builds `spec_obs=batchelor_grad(K,kB,chi)` with NO H², yet passes H² to `_mle_fit_kB`, which de-attenuates variance never applied → on its own fixture returned chi is 7.9x true, kB 17% high (62% at eps=1e-8); the test discards `_chi_fit`, asserts only 0.5<kB<2.0, and blames "grid search". A consistent fixture recovers kB/chi/eps to 0.5-2%. This is the only synthetic-accuracy test of the user-selectable `fit_method='mle'` path; production `_mle_fit_kB` is correct. **Fix:** make the fixture consistent (`batchelor_grad·H²+noise`), tighten to ~5%, assert chi_fit/eps_fit.

#### r1-47. docs/DATA_FLOW.md still claims L5 depth-binning uses a geometric mean for epsilon/chi — stale since PR #68
`docs/DATA_FLOW.md:197-199`. States "Log-normal variables (epsilon, chi) use geometric mean", but `rsi/pipeline.py:650,662` call `bin_by_depth` WITHOUT `log_mean_vars` → arithmetic (`binning.py:96-97`; binning switched arithmetic in 6550022/#68, doc last touched abb1930/#54). ARCHITECTURE.md:408 already says "arithmetic mean", so the two overview docs contradict. A reader trusting DATA_FLOW.md misreads the binned products (median vs mean, factor exp(σ²/2)~1.6-3x). **Fix:** update the L5 section: all vars arithmetic; geometric only via the opt-in `log_mean_vars`.

#### r1-48. `_qc_chi_final` applies the epsilon-FM threshold (1.15) one-sidedly to the two-sided chi variance-ratio fom
`rsi/pipeline.py:67-73`. Gates the chi obs/model variance ratio with `fom<=1.15` only — a threshold defined for the Lueck FM (window.py:349), applied to the two-sided chi fom (`chi_mathematics.md:498`: "far from 1.0 in either direction"). So a window whose model overestimates observed variance 5x (fom=0.2) passes into K_T/Gamma while a 20%-high fit (fom=1.2) is dropped. On the default Method-1 path real SN479 data has 13.6% of windows with fom in [0.5,0.87) that a symmetric cut would reject. A NEW defect IN the prior #32 fix (§2). Bounded (chi anchored to obs_var). **Fix:** gate two-sidedly (`|log(fom)|<=log(1.15)`); same for perturb's `_apply_fom_cut`.

#### r1-49. perturb applies no ATOMIX spectral-fit QC to epsilon before combining/binning, despite computing and writing FM
`rsi/dissipation.py:265`. `_compute_epsilon` stores raw `e_4` and never calls `_compute_flags`/`_compute_epsi_final`; it writes FM (:481, comment "reject FM>~1.15") but nothing thresholds it. The only spectral knob `epsilon.fom_max` defaults None and cuts the variance-ratio fom, not FM; `mk_epsilon_mean`'s CI is inactive for 2-probe SN479. rsi run_pipeline instead bins flag-masked `epsi_final`. Repro on SN479_0006: 18% of windows FM>1.15; removing them drops the aggregate arithmetic-mean epsilon 3.85x (geo 1.18x) — contradicting ARCHITECTURE.md:433-448's claim the paths "agree numerically for two-probe data". Documented-intentional caps it at minor. **Fix:** add an FM_max cut in the perturb diss stage, or correct the doc's numeric claim.

#### r1-50. pyturb-cli `bin` silently geometric-means epsilon/chi (up to ~14x lower than the perturb/rsi arithmetic convention)
`pyturb/bin.py:40`. Hard-codes `log_mean_vars={eps_1,eps_2,eps_final,epsilon,chi,chi_final,epsi_final}` with no CLI flag, so binned dissipation estimates the per-bin MEDIAN. Measured geo/ari = 0.60/0.32/0.07 at σ_ln=1.0/1.5/2.3 (theory `exp(-σ²/2)`); `PYTURB_CLI.md:92-107` never mentions the convention. Geometric mean is a legitimate convention, but the undocumented, un-overridable inconsistency injects a hidden 1.6-14x factor when comparing tools or deriving K_rho. **Fix:** add a `--log-mean/--linear-mean` option (or default arithmetic); document the median semantics.

#### r1-51. `mk_chi_mean` docstring step 7 describes the OLD combine rule ("remove the largest probe value")
`processing/chi_combine.py:54-55`. Documents the pre-fix always-drop-max rule; the code (:158-188) drops the probe furthest from the cross-probe ln-mean and only with >=3 probes (inline comment even says "The old code always dropped the maximum"). Repro {1e-7,1.2e-7,1e-12}: chiMean=1.095e-7 (low outlier dropped) vs the docstring rule's 3.16e-10 — 346x. The epsilon twin docstring was updated; only chi is stale. **Fix:** rewrite step 7 to the symmetric furthest-from-ln-mean rule.

#### r1-52. quick_look `chi_method=2` computes the iterative fit but labels/highlights it as "M2-MLE"
`rsi/quick_look.py:413,568`. The chi-profile panel plots `compute_chi_window(method=2)` → `_iterative_fit` (the M2-Iter estimator) but titles it "M2-MLE", and the spectra panel stars the genuinely-distinct `_mle_fit_kB` curve (:142-165) whose chi need not match the profile panel. Docstring says "MLE Batchelor fit"; CLI help and `window.py:402` say "iterative spectral fit". Diagnostic viewer only. **Fix:** map `chi_method=2` to "M2-Iter" in both label dicts; fix the docstring.

#### r1-53. `parse_time` rejects valid unquoted UTC 'Z' timestamps in sections.yaml / figure specs
`perturb/plot/sections.py:107-120`. ruamel resolves unquoted `2025-01-20T00:00:00Z` to a tz-aware datetime whose `str()` renders `+00:00`; `parse_time` strips only a literal trailing 'Z', so the offset check raises ValueError telling the user to "use a trailing Z" — exactly what they wrote. Quoted form works. Clean error, no silent shift. **Fix:** accept tz-aware datetimes with `utcoffset()==0` (convert to naive UTC) before stringifying.

#### r1-54. pyturb-cli eps window parameters computed once at 512 Hz, never recomputed per file
`pyturb/eps.py:275-278`. `compute_window_parameters` is called once with `fs_default=512.0`; `_process_one_file` overwrites only `fs_fast`, never the sample counts, so at 1024 Hz `-f 1.0` still yields a 0.5 s FFT. Comments at :274/:73 falsely claim per-file override. ARCTERX SN479 is 512 Hz (unaffected); latent for other-rate instruments. **Fix:** move `compute_window_parameters` inside `_process_one_file` after `fs_fast` is known, or delete the misleading comments.

#### r1-55. pyturb-cli documents `--despike-passes`, `-t/--temperature`, `--speed` as functional but all three are ignored
`pyturb/eps.py:35,291`; `docs/pyturb/PYTURB_CLI.md:68-75`. `eps.py` never reads `args.temperature`/`args.speed`; `despike_passes` is in the signature but unused, so despike runs the scor160 default max_passes=10, not the documented 6. Unlike `--aoa`/`--pitch-correction`, these are not marked "(not implemented)". **Fix:** thread `despike_passes` into despike and implement temperature/speed selection, or mark all three "(not implemented)" with a runtime warning; fix the doc table.

#### r2-0. ARCTERX `epsilon.salinity` comment claims CTD-derived salinity, but the code uses fixed S=35 for epsilon viscosity
`ARCTERX/perturb.yaml:84`. Comment says "use CTD-derived salinity (preferred)", but `merge_config` drops the null (`config_base.py:230`), so `_compute_epsilon` runs with `salinity=None` → `dissipation.py:233` `nu=visc35` (fixed S=35). perturb has NO measured-salinity path for epsilon (only chi does). Contradicts `configuration.md:161`/`config.py:597` ("null = 35, fixed S"). Numeric impact tiny (visc(28,34.5) vs (28,35) = -0.099% for ARCTERX; <1.3% worst realistic campaign) but the comment implies a nonexistent capability. **Fix:** correct the comment, or add a measured-salinity epsilon path mirroring chi.

#### r2-1. GPS positions beyond the fix record are always extrapolated; the only guardrail warning is suppressed when the record extremes carry NaN lat/lon
`perturb/gps.py:237,319-320`. `_warn_outside_coverage` never gates to NaN (warn-only) and takes `[t_min,t_max]` from ALL decoded times, while the interpolator's real domain is the finite-node span (`_finite_interp1d` drops non-finite nodes). Repro (lat/lon NaN for t>800 in a [0,1000] record): queries at t=900/950/1000 return finite extrapolated positions with NO warning (only t>1060 warns), because `_t_max=1000`. Feeds per-profile lat/lon, geospatial attrs, and TEOS-10 SA. Opt-in path + edge trigger; negligible open-ocean SA effect. Residual after the prior `_finite_interp1d` fix. **Fix:** compute `_t_min/_t_max` from the finite-node span used by the interpolator.

#### r2-6. `_calc_lag` in FP07 in-situ calibration lacks the non-finite/flatline guard its sibling `ct_align` has
`perturb/fp07_cal.py:135`. `if norm > 0: corr = corr/norm` has no else/continue, so a flatlined (norm=0) or NaN (norm non-finite) segment leaves `argmax(|corr|)` over the negative-lag window returning index 0 = `-max_lag_seconds` (-10 s), appended unfiltered to `lags_list` then median. Repro: flat/NaN segment → -10.0 s vs clean -0.109 s; a majority-poisoned reference shifts the Steinhart-Hart fit → chi +9.15%, +0.47 C. `ct_align.py:121-126` guards exactly this (the identical pattern was fixed there per prior #23/#105, but not propagated here — a genuine unfixed parallel, not a duplicate). Latent on the default path: JAC_T is a polynomial that never emits NaN (0/464 in-house profiles poisoned) and the median is robust to a single bad profile. **Fix:** mirror `ct_align` — return `(nan,nan)` when `norm` is non-finite/<=0, and drop non-finite entries before `np.median`.

---

## 5. New Confirmed Findings — NIT (8)

#### r1-21. Five code/metadata comments claim good FM "approaches 0"; theoretical/measured good-fit FM is ~0.7-0.8
`rsi/dissipation.py:491`, `rsi/pipeline.py:751`, `rsi/chi_io.py:30,356,577`. For a correctly-modeled spectrum `FM = 0.798/T_M ∈ [0.74,0.86]` (MC median 0.75, p95 0.95); near-zero FM would be anomalous. `epsilon_mathematics.md:517` states the correct interpretation. Every site co-states the correct 1.15 reject threshold (which actually drives QC), so no QC decision or product is affected. **Fix:** reword to "expected near 0.7-0.8, below ~1 for 97.5% of spectra".

#### r1-57. docs/mixing_efficiency.md implies all derived estimates get both the N² and dT/dz masks
`docs/mixing_efficiency.md:58-61`. States estimates are NaN where N²<1e-9 "or" |dT/dz|<1e-4, but the code masks per-variable: `K_T` on dTdz only, `K_rho` on N² only, only `Gamma` on both (`mixing.py:504-510`). Repro: N²=1e-10,dTdz=0.01 → K_T finite while Gamma/K_rho NaN. Function docstring is correct; only the .md generalizes. **Fix:** rewrite per-variable mirroring `mixing.py:454-464`.

#### r1-58. L1 GRADT uses the legacy first-difference method vs the ODAS default high-pass gradient, undocumented
`rsi/convert.py:521`. Computes `fs·diff(T)/speed` (ODAS's legacy `first_difference` option) while ODAS defaults to `high_pass` (`make_gradT_odas.m:60`, `odas_p2mat.m:232`); grep confirms the method is documented nowhere. Export-only product (chi builds its own gradient spectrum); HF divergence from the sinc response of a first difference. (The finding's "second deviation" sub-claim is false — the last-sample handling matches ODAS.) **Fix:** port the high-pass method for parity, or document the legacy choice in the GRADT attrs.

#### r1-59. chi_mathematics.md Method-1 step 7 says chi is the grid argmin; code applies a parabolic log-space refinement
`docs/chi_mathematics.md:346`. `_chi_from_epsilon` refines the 200-point/4-decade grid minimum by a parabola in log10(chi) (`chi.py:283-296`), shifting the reported chi by up to ~2.3% vs the documented pure argmin. Code is the improved version; doc is incomplete. **Fix:** append the parabolic-refinement step to §6 step 7.

#### r1-60. Nasmyth "formula/coefficient" tests are tautological
`tests/test_scor160_nasmyth.py:30`, `test_epsilon.py:133`. Both recompute `8.05·x^(1/3)/(1+(20.6x)^3.715)` — character-identical to `nasmyth.py:72` — then `assert_allclose` the function to itself, so a coefficient transcription error would propagate to both and never be caught. Constants are currently correct (no numeric impact). **Fix:** pin literal G2 values from the published Lueck tabulation with a citation.

#### r1-61. Binned/combo `epsilonLnSigma`/`chiLnSigma` lose the "NOT the sigma of the combined mean" caveat
`perturb/netcdf_schema.py:233-237`. The per-profile var carries a comment that it is the single-probe σ_ln with "no sqrt(n) reduction applied" (`epsilon_combine.py:180-184`), but binning strips all attrs and the combo schema re-attaches only `units=''`/`long_name`, so the RMS-binned σ is published indistinguishable from an error bar on `epsilonMean` (which it overstates by ~sqrt(n)). `long_name` remains technically accurate (a spread), bounding misuse risk. **Fix:** add the caveat + RMS-of-per-window-σ semantics to the schema entries.

#### r1-62. pyturb eps `S_sh*` labeled "cleaned shear spectrum" even though Goodman cleaning is off by default
`pyturb/_compat.py:170-174`. With the default `--goodman` off (`cli.py:73`), `process_l3` sets `sh_spec_clean` to a copy of the raw spectrum (`l3.py:167-168`), but the long_name is written "cleaned shear spectrum" unconditionally. Secondary standalone tool; epsilon numbers unaffected (same array either way). **Fix:** set the long_name conditionally, or add a `goodman` global attribute.

#### r2-4. speed.py docstring inverts the dbar↔metre conversion and the sign of the systematic speed bias
`rsi/speed.py:65-68`. States "1 dbar ~ 1.01-1.02 m ... ~1-2% systematic speed bias", but 1 dbar is ~0.99 m for seawater (gsw: 0.993-0.994 m/dbar), so `|dP/dt|` OVER-states true speed by <1% (opposite sign to the docstring). The 1.01-1.02 figure is the inverse relation and holds only for fresh water. Purely explanatory QC note with zero effect on any computed value. **Fix:** correct to "1 dbar ~ 0.99 m ... over-states the true m/s speed by <1%".

---

## 6. Refuted Candidates (do not re-litigate)

Nine candidates were refuted by the verifier; recorded so future rounds skip them.

1. **Default chi fabricates finite chi/absurd epsilon_T for signal-free windows (chi.py).** Mechanism reproduces, but substantially duplicates prior finding #32; perturb `chiMean` is already protected by the 1e-13 floor.
2. **M2 fill-value leak in `helpers._channels_from_nc` / `chi_io._load_therm_channels` (helpers.py).** Unreachable on valid in-house data: real per-profile NCs carry no `_FillValue` on P/T1/sh1; reads return clean unmasked data.
3. **`compare_atomix.py` pre-zeroes NaN spectral bins (compare_atomix.py).** Reproduces synthetically, but the real benchmark inputs have zero interior-NaN bins (only Epsifish has single DC-bin NaNs, handled correctly).
4. **Plot resolver strips `_engine`, can pick a different-code-version dir (resolve.py).** Frames intended, documented behavior as a defect; the "silently" claim is false (resolve.py:166-171 warns).
5. **`dissipation.py:237` K_AA `max(...)` keeps NaN — unfixed sibling of #37 (dissipation.py).** Unreachable: `prepare_profiles` NaN-scrubs speed at `helpers.py:325` across all branches before this site.
6. **`GPSFromCSV` crashes on ISO timestamp columns (gps.py).** Intended/documented ("time column expected in epoch seconds"); fails loudly at construction on an opt-in path, no docstring contradiction.
7. **Flight-model docstring "matching ODAS Slocum trim" uses opposite AoA sign (speed.py).** Over-read: the phrase modifies the default value (3 deg), not the sign; the sign convention is correctly attributed to Merckelbach 2010.
8. **Diss/chi products emit identically-labeled N2/dTdz at different window scales (pipeline.py).** Intended, documented dual-scale N² design; per-variable NetCDF comments name the scale.
9. **Window-stratification path silently assumes S=35 with no fast-conductivity guard (pipeline.py).** On real data P/JAC_T/JAC_C share `time_slow`, so the default path uses measured salinity; the S=35 fallback is recorded in the NetCDF comment and docstring, not silent.

---

## 7. Unverified Findings

None. Every candidate that entered verification received a verdict (confirmed or
refuted); no verifier agent failed to return a judgment this round.

---

## 8. Methodology

**Finder phase — 13 lenses + fix-status phase.** Thirteen finder lenses swept the
codebase: rsi `.p` I/O and numeric conversion; epsilon physics; chi physics;
scor160/ATOMIX benchmark and the `compare_atomix.py` harness; instrument-agnostic
processing (trim / bottom / ct_align / mixing); the perturb campaign pipeline and
its incremental cache; config/hashing infrastructure; perturb plotting; pyturb;
documentation self-consistency; through-water speed / fall-rate; ancillary-data
(hotel / GPS / CTD / seawater) ingestion; and dual-N² stratification. A dedicated
fix-status phase re-verified the 24 prior-audit findings tracked across PRs
#79-#87 against HEAD.

**Dedup.** Raw candidates were semantically de-duplicated — merged where a defect
and its coverage/trigger sibling coincided (e.g. r1-32 merges two `bin`-label
candidates; r1-17, r1-50, r1-4, r1-33 each absorb a sibling).

**Verification — batched, adversarial, refute-by-default.** Survivors went through
batched refute-by-default verification: each verifier agent judged a small GROUP
of findings (majors in groups of 3; minors/nits in groups of 6) within one
context, applying all three perspectives per finding — **reproduce** (run the
repro), **reference** (check against the ODAS MATLAB library and published
physics), and **context** (default-path reachability, prior-audit overlap). A
finding is CONFIRMED only if the verifier did not refute it; the verifier's
recalibrated severity overrides the finder's (e.g. r1-6/r1-7/r1-8/r1-9/r1-10 were
demoted major→minor; r1-21/r2-4 minor→nit). A completeness-critic round 2 covered
gaps.

**Honest caveat on gate strength.** Verification was **batched — one agent per
group of findings — rather than one independent agent per lens**. This was an
adaptation forced by usage-window budget limits, and it is a slightly weaker
adversarial gate than independent per-lens voting: a single verifier context
judged several findings together, so cross-finding anchoring is possible and
each verdict rests on one agent's pass rather than a quorum. Every finding here
is therefore best read as "survived one adversarial pass," not "survived
independent replication." Confidence fields in the JSON record where the verifier
itself flagged medium/low certainty (e.g. r1-3 medium — the specific ARCTERX
false-positive counts are unverifiable because those VMP files are absent from
this checkout; r1-29 low; r1-38/r1-48/r1-61 medium).

**Baseline.** At HEAD with `PYTHONPATH=/Users/pat/tpw/tpw/turbulence/src`:
2154 passed, 76 skipped, 0 failed (the 76 skips are the reference-data tests of
r1-11). The severity tally in §1 was cross-checked programmatically against the
JSON record after writing (major 7 / minor 48 / nit 8 / 63 total; fix-statuses
22 fixed + 2 partially-fixed = 24).

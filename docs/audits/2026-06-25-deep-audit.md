# Deep Audit Report: microstructure-tpw (, branch main)

## 1. Executive Summary

This deep adversarial audit examined the full `microstructure-tpw` processing
chain — RSI `.p` I/O and numeric conversion, epsilon/chi physics, the scor160
ATOMIX benchmark path, instrument-agnostic profile processing, the perturb
campaign pipeline, config/hashing infrastructure, plotting, and packaging. Every
finding below survived a refute-by-default verifier; severities reflect the
verifier's recalibration, not the original finder's.

**Overall risk read: the package's *primary scientific outputs* (epsilon and chi
from shear/FP07 microstructure) are sound on the in-house ARCTERX/SN479 happy
path, but there are real correctness defects in (a) one default-path chi numeric
accuracy bug and (b) several silent-data-corruption paths in profile/CTD/bottom
processing that trigger on realistic — not merely adversarial — field-data and
external-NetCDF conditions.** None rises to a blocker (no defect produces a wrong
primary result on every run of valid in-house data), but three majors can
silently corrupt or discard real scientific output without raising.

### Headline issues (read these first)

1. **Chi variance-correction grid is under-resolved on the default path
   (`chi/chi.py:98-99`, MAJOR).** When `kB << K_max` (low-epsilon windows that
   still pass QC), the fixed 2000-point grid spread over `5·K_max` cannot resolve
   the spectral peak near `kB`. Verifier measured **9–42% chi error** on the
   production-default iterative/Kraichnan path with `K_min>0` — far worse than
   the docstring's claimed "<0.1% accuracy", and worse than the original ~2%
   estimate. The originally-proposed fix (`K_upper = max(K_max, 8·kB)`) was shown
   to make some cases *worse*; the correct fix is adaptive point density / fixed
   `dK`.

2. **NetCDF fill values leak into pressure and channels, silently truncating
   profiles (`rsi/profile.py:362-427`, MAJOR).** `_load_from_nc` reads
   `[:].data` off masked arrays, exposing the raw `9.97e36` fill buffer instead
   of NaN. A single fill value mid-descent produces a `4.47e35 dbar/s` fall-rate
   spike and silently drops the rest of the cast — no exception. Affects any
   partial/interrupted NC or any externally-produced (CF/ATOMIX) NC carrying
   `_FillValue`, both supported input paths.

3. **A single NaN T/C sample poisons the CT lag estimate, returning the
   maximally-wrong `-max_lag` shift (`processing/ct_align.py:77-124`, MAJOR).**
   The `norm <= 0` and `total <= 0` guards are False for NaN (NaN comparisons
   always False), so a NaN-poisoned profile is *not* skipped and drives a `-5 s`
   CT misalignment into the CTD/salinity/density product and the N²/stratification
   computed immediately after. The realistic trigger is built in: `convert_jac_c`
   deliberately emits `np.nan` for corrupt conductivity samples, and the pipeline
   calls `ct_align` on that channel with no NaN stripping.

4. **Bottom-crash trimming silently fails in ~50% of detections
   (`processing/bottom.py:98,129`, MAJOR).** The deepest depth bin overhangs the
   real max depth by up to one bin width, and the reported crash depth is the
   bin *center* — which lies *below* the deepest real sample whenever the bottom
   falls in the lower half of that overhanging bin (~50% of cases). The caller's
   `P_seg >= bottom_depth` match is then empty and the seafloor-impact samples are
   left in the deepest epsilon/chi estimates with no warning. The original finder
   self-rated this "low confidence / benign"; the verifier *refuted that hedge*
   and reproduced a 49.6% trim-failure rate.

The remaining issues are bounded: a cluster of robustness gaps in malformed-`.p`
handling (opaque errors / batch-abort instead of graceful skip), an inter-probe
combine rule that diverges from the benchmark convention on the 2-probe campaign
instrument, several CF/UDUNITS metadata non-compliances on auxiliary channels,
config-hashing edge cases (date values crash; a 2-digit-glob collision past 100
configs), and an assortment of presentation, port-faithfulness, and config-drift
nits. A notable share of the nits are *latent* — real but with no reachable
trigger in the current code/data (e.g. the unsigned-wrap ordering, the dead
`kB_best < 1` guard, the static-section `None` crash).

**Confirmed totals: 0 blocker, 4 major, 28 minor, 35 nit (67 findings)** — the
complete record is `2026-06-25-deep-audit-findings.json`. This report writes up
62 of them as itemized entries below (4 major, 23 minor, 35 nit); the remaining
5 minor findings were folded into related entries during de-duplication (see §3).
(The per-section counts and this paragraph were corrected post-synthesis: the
synthesis agent's own tally drifted from the findings it actually emitted.)

---

## 2. Findings by Severity

### BLOCKER (0)

None. No confirmed defect produces a wrong *primary* scientific result on every
run of valid in-house data, and none is a crash on the default happy path.

---

### MAJOR (4)

#### M1. Chi variance-correction integration grid under-resolved when `kB << K_max` (up to ~40% chi bias on the default path)
- **Subsystem:** chi-physics
- **Location:** `src/odas_tpw/chi/chi.py:98-99` (docstring claim at 93-94)
- **Evidence:** `K_upper = max(K_max*5, kB*5)` then `K_fine = np.linspace(K_upper/n_fine*0.01, K_upper, n_fine)` with `n_fine=2000`. When `kB` is small relative to `K_max` (low-epsilon windows where the anti-aliasing mask pushes `K_max` out near `K_AA`), the fixed 2000-point grid is spread so coarsely (`dK ~ 0.35 cpm` over ~700 cpm) that the steep Batchelor/Kraichnan peak near `kB` is unresolved. The verifier's refutation *failed for the production-default path*: `process_l4_chi_fit` defaults `fit_method="iterative"`, `spectrum_model="kraichnan"`, and `_iterative_fit` calls `_variance_correction` with `K_min = k_l_new > 0`, which breaks the V_total/V_resolved ratio cancellation that protects the `K_min=0` paths. Measured correction-factor error vs a converged `n=500000` grid: **kB=3 → +41.6%, kB=5 → +19.3%, kB=8 → +9.2%, kB=15 → +3.2%**, propagating linearly into `chi = chi_band * correction` (chi.py:573). These windows pass QC (only `kB<1` is cut; `K_max_ratio ~ 10–47` looks healthy). This directly contradicts the docstring's "<0.1% accuracy."
- **Fix:** Decouple grid *density* from `K_max`. Use an adaptive `n_fine` (or a fixed `dK`) so the peak near `kB` is always resolved while still covering `K_max` for the `[K_min, K_max]` mask. **Do not** use the originally-proposed `K_upper = max(K_max, 8·kB)` — the verifier showed it makes kB=3/kB=5 *worse* because it clamps the extent without raising point density. Add a convergence test asserting <0.1% error vs a high-n reference across `kB ∈ {3,5,8,15}` at realistic `K_max`, and correct the docstring.

#### M2. `_load_from_nc` leaks NetCDF `_FillValue` (9.97e36) into pressure and channels via masked-array `.data`
- **Subsystem:** rsi-pfile-robust
- **Location:** `src/odas_tpw/rsi/profile.py:362-368, 393, 397, 399, 427`
- **Evidence:** Every read uses `ds.variables[...][:].data`. netCDF4 returns a **masked** array when a variable has a `_FillValue` or unwritten elements; `.data` exposes the raw `~9.96921e36` fill buffer, *not* NaN. Verified end-to-end: an NC with one unwritten slow-P element makes `_load_from_nc(...)['P'] = [10.0, 9.96920997e+36]`. That value then drives `_smooth_fall_rate`/`get_profiles`, producing a `4.47e35 dbar/s` spike; the detected profile truncates from `(321, 4802)` to `(321, 2560)`, **silently dropping the entire second half of the cast with no error**. The `9.97e36` value also passes the `P_min < P` validity mask, and channel reads at line 427 carry it into every downstream epsilon/chi computation. The ATOMIX `L1_converted` group path (lines 397/399) is equally affected. Happy path is safe (the package's own NCs are written fully with no `_FillValue`), but external/partial NCs — both supported inputs — trigger corruption. No test in `tests/test_rsi_profile.py` covers masked/fill input.
- **Fix:** Replace `[:].data` with fill-aware extraction at all four sites (t_fast/t_slow 362-368, P 393/397/399, channel scan 427), e.g. `np.ma.filled(ds.variables[name][:].astype("f8"), np.nan)` (or `ds.set_auto_mask(True)` then `.filled(np.nan)`). Add a regression test writing an NC with an unwritten/fill element and asserting the result is NaN, not `>1e30`.

#### M3. NaN in T/C poisons the CT lag estimate, yielding the maximally-wrong `-max_lag_seconds` shift
- **Subsystem:** processing-ctalign
- **Location:** `src/odas_tpw/processing/ct_align.py:77-88, 121-124`
- **Evidence:** A single non-finite sample anywhere in a profile's T or C segment corrupts the whole lag estimate — the NaN twin of the flatline failure the audit-#42 guard was added for. Chain (all reproduced by running `ct_align`): the 2nd-order IIR `lfilter` (lines 77-78) smears one input NaN to **all** subsequent outputs → `dx/dy` all-NaN → `norm = sqrt(...) = NaN`. Line 82 `if norm <= 0:` is **False for NaN**, so the degenerate profile is not skipped (the comment on lines 83-87 describes exactly this argmax→index-0→`-max_lag` failure, but the guard only catches `norm==0`). `np.argmax(np.abs(all-NaN))` returns 0 → `lag = lags[0] = -max_lag_samples`. The per-profile weight `max_corr*n_samples = NaN`; line 121 `if total <= 0:` is again False for NaN; `searchsorted(cum_w, NaN/2)` returns 0 → `median_lag` = the poisoned profile's `-max_lag_seconds`. Measured: a clean profile (true 5-sample lag) + one profile with a single injected NaN → returned lag **`-5.0 s` (= -320 samples)** instead of `-5/64 s`; the valid profile's consensus is entirely discarded. Realistic trigger built in: `convert_jac_c` (channels.py:238) emits `np.nan` for corrupt samples (`v_part==0`), and `pipeline.py:1049` calls `ct_align` directly on `pf.channels[C_name]` (JAC_C) with no NaN stripping, feeding the CTD product and the N²/dT/dz stratification at pipeline.py:1059+. No NaN-injection test exists.
- **Fix:** Treat non-finite like flatlined. (1) Line 82 → `if not np.isfinite(norm) or norm <= 0:` (or pre-skip the profile if `seg_T`/`seg_C` is not all-finite, since `lfilter` smears any NaN). (2) Line 121 → `if not np.isfinite(total) or total <= 0:`. Optionally drop non-finite `per_profile` entries before the weighted median. Add a regression test injecting a single NaN into C within an otherwise-good profile and asserting the returned lag equals the clean-data consensus.

#### M4. Bottom-crash reports the deepest bin's CENTER, which lies below the deepest real sample → ~50% of crashes silently not trimmed
- **Subsystem:** processing-trim-crash
- **Location:** `src/odas_tpw/processing/bottom.py:98, 129`; caller `src/odas_tpw/perturb/pipeline.py:433-438`
- **Evidence:** `bins = np.arange(depth_minimum, max_depth + bin_size, bin_size)` (line 98) builds edges whose deepest bin `[bins[-2], bins[-1])` overhangs `nanmax(depth)` by up to `bin_size`; the crash depth is reported as the bin *center* `0.5*(bins[i]+bins[i+1])` (line 129). A bottom crash lands in that deepest, densely-filled (fs=512 Hz) overhanging bin, which is therefore eligible and selected. The center exceeds the deepest real sample whenever `max_depth` falls in the lower half of the bin — geometrically ~50%. The verifier *refuted the finder's "benign" hedge* and reproduced it: over 20000 realistic descent profiles, the returned depth exceeded `nanmax(depth)` in **10065/20000** trials; in the full caller logic (`below = np.where(P_seg >= bottom_depth)[0]; if len(below) > 0:`) the empty match leaves `new_e = e_slow` and the crash is **silently not trimmed in ~49.6% of detections**, leaving seafloor-contaminated samples in the deepest epsilon/chi estimates. The sibling `top_trim.py:71` deliberately builds *centered* edges and does not overhang, so the bottom.py:126-128 comment invoking the top_trim analogy is misleading.
- **Fix:** Clamp the reported crash depth to the data: `bottom_depth = min(0.5*(bins[i]+bins[i+1]), float(max_depth))`, or report the mean/max of the actual depth samples in the selected bin so the value never exceeds the deepest observed sample. Add a test asserting the reported crash depth is `<= nanmax(depth)` and that the caller trims when a crash is detected.

---

### MINOR (23 itemized; 28 confirmed)

#### m1. Inter-probe rejection diverges from the benchmark-validated ATOMIX/Lueck rule (drop-furthest vs anchor-to-minimum); dead-code CI loop on the 2-probe campaign instrument
- **Subsystem:** processing-combine
- **Location:** `src/odas_tpw/processing/epsilon_combine.py:128-167`; mirror in `chi_combine.py:163-188`
- **Evidence:** The benchmark-pinned reference `scor160/l4.py:740-746` flags only probes too high above the window minimum (`e_min` is the trusted baseline) and takes the geometric mean of survivors. `mk_epsilon_mean` instead uses the full `|ln(max)-ln(min)|` spread and drops the probe *furthest from the cross-probe ln-mean* (symmetric), iterating one drop at a time, **gated on `n_probes >= 3`**. For two probes `{1e-8, 1e-3}` the reference returns `1e-8`; `mk` keeps both and returns `3.16e-6` (**316× higher**). The campaign instrument (VMP-250 SN479) has exactly 2 shear probes, so on the real data the CI rejection loop is **dead code** and a disagreeing probe is never dropped. `epsilon_combine.py:128` mislabels the symmetric rule as "the Lueck/Rockland mk_diss_mean rule," while l4.py claims its *different* `e_min` rule "matches the benchmark convention" — both cite the same authority for opposite behavior. (Note: for the 3-probe one-LOW-junk case `mk` is actually superior, so the intent is defensible — this is a reconciliation/documentation defect, not a pure computational bug.)
- **Fix:** Reconcile and document. Either (a) document in both module docstrings that `mk_*_mean` intentionally departs from the `e_min`-anchored ATOMIX rule used in `scor160/l4.py`, with the physical justification, and cross-reference `DEFAULT_DISS_RATIO_LIMIT`/`l4._compute_flags`; or (b) adopt the reference's hybrid (flag relative to `e_min` but exclude the `e_min` probe from being sole survivor when the cluster sits above it). Fix the mislabeled provenance comment at epsilon_combine.py:128.

#### m2. `np.nanmean/nanmin/nanmax` RuntimeWarnings leak from the combine functions on any NaN row (common single-NaN-in-nu case)
- **Subsystem:** processing-combine
- **Location:** `src/odas_tpw/processing/epsilon_combine.py:123,140,141,158,167`; `chi_combine.py:146,167,168,183,191`
- **Evidence:** The `np.errstate(invalid="ignore", divide="ignore")` contexts suppress FP exceptions, **not** the Python `warnings`-module RuntimeWarnings (`"Mean of empty slice"`, `"All-NaN slice encountered"`) that `np.nanmean/nanmin/nanmax` emit. A single NaN in `nu` makes a row's `sigma_ln_epsilon` all-NaN, tripping line 123 even with zero all-NaN epsilon rows. Results stay numerically correct (NaN where expected), but the warning spam pollutes logs and would crash any caller under `-W error::RuntimeWarning`. `pyproject.toml` `filterwarnings` does not include `"error"`, and the combine tests assert only on values, so this slips through CI.
- **Fix:** Wrap the `np.nan*` reductions in `warnings.catch_warnings(); warnings.simplefilter("ignore", RuntimeWarning)` (`np.errstate` is insufficient), or guard all-NaN rows explicitly. Add a test asserting warning-cleanliness on a NaN-row input.

#### m3. `convert_all` per-file isolation defeated by too-narrow exception allowlist (incl. GRADT `IndexError` on a 1-sample fast channel)
- **Subsystem:** rsi-pfile-robust
- **Location:** `src/odas_tpw/rsi/convert.py:605 (serial), 616 (parallel)`; concrete trigger `convert.py:514-515`
- **Evidence:** Both loops catch only `(OSError, ValueError, RuntimeError)`, but the documented per-file-isolation intent (log `{name}: {e}` and continue) is defeated by other classes. `convert.py:514-515` `dTdt = np.append(dTdt, dTdt[-1])` raises `IndexError` for a length-1 fast channel (`np.diff` returns empty), and `IndexError`/`KeyError`/`struct.error` are not subclasses of the caught tuple (verified). In the **parallel** branch the exception escapes `future.result()`, propagates out of the `as_completed` loop, exits the `ProcessPoolExecutor` block, and **abandons all remaining files**; serial aborts the loop. (The GRADT length-1 path is itself only reachable on a structurally degenerate `n_rows==1` `.p`, never real instrument output — that part is a nit-level edge case — but the *exception-isolation gap* is the real defect.)
- **Fix:** Broaden both handlers to `except Exception as e:` (still logging `{p_path.name}: {e}` and continuing), or explicitly add `IndexError, KeyError, struct.error`. Separately, guard the GRADT degenerate case: `if T_fast_ch.size < 2: gradt_arrays.append(np.zeros_like(T_fast_ch)); continue`.

#### m4. `matrix_count == 0` yields empty channels/time vectors with no diagnostic
- **Subsystem:** rsi-pfile-robust
- **Location:** `src/odas_tpw/rsi/p_file.py:377-378, 475-476`
- **Evidence:** When `data_words < n_cols`, `scans_per_record`/`total_scans`/`matrix_count` become 0, producing empty `t_fast`/`t_slow` and empty channel arrays; nothing rejects this. Downstream consumers fail opaquely: `summary()` does `self.t_fast[-1]` → `IndexError`; `convert.py:472` `np.interp(pf.t_fast, pf.t_slow, P_slow)` raises `"array of sample points is empty"` (verified). Only reachable on a structurally malformed/truncated header (record body smaller than one `n_rows × n_cols` scan), never a valid RSI file. (The original framing that ODAS "rejects no-data files" is partly overstated — ODAS's `n_records <= 1` guard is the analog of the existing Python `n_records < 1` check; ODAS has no scan-count guard either.)
- **Fix:** After computing `matrix_count`, add `if matrix_count < 1: raise ValueError(f"{self.filepath.name}: file yields no complete scans (matrix_count=0)")`, matching the existing validation pattern in `extract_pfile_segment`.

#### m5. `edit_config_text` trusts unvalidated `EditSpec` values, allowing stanza injection that defeats the documented `[root]`/`[matrix]` unaddressability guarantee
- **Subsystem:** rsi-config-patch-security
- **Location:** `src/odas_tpw/rsi/config_patch.py:305-442` (apply); validation only at 111-153 (inside `load_edit_spec`)
- **Evidence:** All value sanitization (no-newline/`;`/control-chars) lives only in `load_edit_spec`; `edit_config_text` and the public `EditSpec` dataclass perform none. Verified exploits against the real code: `EditSpec(sections={'instrument_info': {'vehicle': 'rvmp\r\n[root]\r\nrate = 1'}})` makes `parse_config(active)['root']['rate'] == '1'` — overwriting an acquisition parameter the module docstring (lines 18-20) guarantees is "unreachable by design ... can never be corrupted." A second asymmetric path: `provenance()` (lines 356-357) interpolates `spec.note` **raw** into a `;`-comment, while `_build_banner` sanitizes it via `_one_line` — so `note='oops\r\n[root]\r\nrate = 7'` injects `rate=7`. **Not an exploitable security vuln** (the only production caller, the CLI, always goes through `load_edit_spec`; reaching the exploit requires already executing arbitrary Python to hand-build an `EditSpec`) — it is a defense-in-depth + doc-vs-code-invariant + provenance-sanitization-inconsistency gap.
- **Fix:** Enforce validation at the apply boundary: add a validating `__post_init__` to `EditSpec` (or a `validate()` at the top of `edit_config_text`) running `_check_value` over every section/channel value and over `note`/`author`. Route `ch.old`/`ch.new`/`spec.note` through `_one_line()` in `provenance()` to match `_build_banner`.

#### m6. `write_patched_pfile` raises uncaught `struct.error` on a corrupt `header_size` (< 24 bytes)
- **Subsystem:** rsi-config-patch-security
- **Location:** `src/odas_tpw/rsi/config_patch.py:527-528`; CLI catch list `cli.py:291`
- **Evidence:** `struct.pack_into(f"{endian}H", header_region, 22, ...)` needs a buffer ≥ 24 bytes; a source advertising `header_size < 24` makes `bytearray(f.read(header_size))` too short → `struct.error`. `struct.error` is not a subclass of `OSError`/`ValueError` (verified), and `_cmd_patch_config` catches only `(FileNotFoundError, FileExistsError, ValueError, OSError)`, so the user gets a raw traceback instead of `Error: ...` + `exit(1)`. Reproduced on a real VMP file with `header_size` corrupted to 16. Unlike `extract_pfile_segment` (p_file.py:138-139, validates `header_size >= HEADER_BYTES`), the patch path has no such guard.
- **Fix:** Validate `header_size` in `read_config_text` and `write_patched_pfile` (require `HEADER_BYTES <= header_size`, `config_size` consistent with file size) and raise an actionable `ValueError`, mirroring `extract_pfile_segment`. Alternatively add `struct.error` to the CLI's caught tuple.

#### m7. `write_patched_pfile` silently produces a corrupt copy when `header_size` is wrong-but-plausible (24..127)
- **Subsystem:** rsi-config-patch-security
- **Location:** `src/odas_tpw/rsi/config_patch.py:519-541`
- **Evidence:** The writer never validates `header_size` against `HEADER_BYTES` (128); its only guard (line 521, `first_record_size >= file_size`) catches truncation, not a wrong-but-plausible value. With `header_size=64` (real header 128) verified to return successfully with **no exception**, writing a truncated header and copying body from `first_record_size = header_size + config_size = 83` — an offset inside the real 128-byte header — silently mis-slicing the body. The silent-corruption sibling of m6. `read_config_text` shares the missing check.
- **Fix:** Add `header_size >= HEADER_BYTES` validation (and `header_size + config_size` consistency) before copying, raising `ValueError` like `extract_pfile_segment`. Best: a shared `_read_and_validate_header` helper used by `read_config_text`, `write_patched_pfile`, and `extract_pfile_segment`.

#### m8. `dissipation.py` does not scrub NaN window temperature before viscosity; sibling `l4.py` does, so a NaN window silently yields NaN epsilon
- **Subsystem:** rsi-dissipation
- **Location:** `src/odas_tpw/rsi/dissipation.py:213-225`
- **Evidence:** `nu_out = visc(l3.temp, sal_window, l3.pres)` / `visc35(l3.temp)` is computed directly from `l3.temp`, which can legitimately be NaN (a NaN fast-temperature sample NaN-poisons the window mean at l3.py:179). `visc35(nan)→nan`, and `_estimate_epsilon(..., nu=nan)` returns `epsilon=nan, method=0, fom=nan` (the `e_10<=0` floor guard is False for NaN; the ISR branch is skipped). The canonical `process_l4` path guards exactly this (`l4.py:154-160`: non-finite window temp → 10 °C + warning), so the two epsilon paths diverge on identical input: NaN viscosity → NaN epsilon vs 10 °C → finite `9.6e-11`. Net effect is data loss (a recoverable window dropped to NaN), not a silent wrong number, since `epsilon=NaN` remains the authoritative invalid marker and `method` is unused metadata.
- **Fix:** Mirror `l4.py`: `temp_w = np.where(np.isfinite(l3.temp), l3.temp, 10.0)` for both the `visc` and `visc35` branches (with a warning), or guard `nu` in the j-loop (`if not np.isfinite(nu): nu = visc35(10.0)`). Add a test feeding a NaN-temperature window and asserting finite epsilon.

#### m9. Chi from per-profile NetCDF silently discards FP07 `diff_gain` and electronics calibration, biasing the noise floor / QC
- **Subsystem:** rsi-chi-io
- **Location:** `src/odas_tpw/rsi/chi_io.py:765-792`
- **Evidence:** The NetCDF branch of `_load_therm_channels` hardcodes `diff_gains.append(0.94)` and never populates `therm_cal` (line 792 → `[{}]*len(therm)`), whereas the `.p` branch reads the real per-channel `diff_gain` and FP07 cal. Verified on SN479: `.p` source returns `diff_gains=[0.912, 0.92]` with full `therm_cal` (`beta_1=3143.55`, `e_b=0.6828`, …); the per-profile NC of the *same file* returns `[0.94, 0.94]` and `[{}, {}]`, no warning. The `configuration_string` global attr *is* written to the NC (profile.py:312-313) but never parsed here. The documented `prof → chi` workflow hits this path. **Important scope correction:** the headline claim that this "biases the observed gradient spectrum" is wrong — the `G_2` pre-emphasis is already baked into the stored `T1_dT1` channel; `bl_corrections` are diff_gain-insensitive. The real impact is a **~6–10% noise-floor bias** (`gradT_noise_batch` uses defaults) that propagates into `K_max`, `K_max_ratio`, and `fom`.
- **Fix:** In the NetCDF branch, parse the embedded `configuration_string` via `parse_config()` and extract per-channel `diff_gain` and FP07 cal the same way the `.p` branch does. If the attribute is absent, `warnings.warn` that the noise floor falls back to defaults, so the calibration loss is not silent.

#### m10. `kraichnan_grad` leaks divide-by-zero RuntimeWarning at `kB=0` (incomplete errstate guard; contradicts `batchelor_grad`'s "mirror" comment)
- **Subsystem:** chi-spectra
- **Location:** `src/odas_tpw/chi/batchelor.py:173-177`
- **Evidence:** `y = k / kB` (line 173) and `sq6q_y = sq6q * y` (line 175) are computed *outside* the `np.errstate(divide="ignore", invalid="ignore")` block, which wraps only the final `S` (176-177). `kraichnan_grad(k, 0.0, 1e-7)` raises `RuntimeWarning: divide by zero` (verified), while `batchelor_grad` does not — its comment (117-118) says it guards its division "to mirror kraichnan_grad," but the mirroring is one-directional. Output is still correct (`np.where(np.isfinite(S), S, 0.0)`), and production guards `kB < 1` before calling, so it is a stray cosmetic warning + faithfulness inconsistency, never fatal (no `"error"` in `filterwarnings`).
- **Fix:** Indent lines 173 and 175 under the existing `with np.errstate(...)` block. Add a test mirroring `test_batchelor_grad_zero_kB_no_warning` asserting `kraichnan_grad(k, 0.0, 1e-7)` raises no warning and returns finite values.

#### m11. `fp07_transfer_batch` silently treats any unknown model string as `double_pole` instead of raising
- **Subsystem:** chi-spectra
- **Location:** `src/odas_tpw/chi/fp07.py:175-178`
- **Evidence:** `if model == "single_pole": ...; else: double_pole`. Any other string (a typo like `'singel_pole'`) silently selects double-pole — verified `np.allclose` to the double-pole branch (a ~3.5× `|H|²` difference at 50 Hz). Inconsistent with `fp07_tau` (128-129), which raises `ValueError` on unknown models. The CLI restricts `--fp07-model` to valid choices, but the YAML path is value-unvalidated (`validate_config` checks only section/key *names*), so a typo'd `fp07_model` flows into `l3_chi.py:212` and produces a silently-wrong chi. The same `else`-fallthrough pattern recurs at `default_tau_model` (fp07.py:82), `l4_chi.py:88`, and `window.py:325`, so on a typo you get a never-intended double-pole-|H|² + lueck-tau hybrid.
- **Fix:** Make `fp07_transfer_batch` raise on unknown models (`elif model == "double_pole": ...; else: raise ValueError(...)`), matching `fp07_tau`. Apply the same to `default_tau_model`, `l4_chi.py:88`, and `window.py:325`. Optionally validate the value at the config layer.

#### m12. `despike` bad-region marking under-removes one trailing sample for odd N (MATLAB float-`N/2` + round vs Python floor)
- **Subsystem:** scor160-despike-nasmyth
- **Location:** `src/odas_tpw/scor160/despike.py:93, 131-132`
- **Evidence:** Python: `N_half = N // 2` (floor), marks `[s - N//2, s + 2*(N//2)]`. ODAS `despike.m` halves N as float (`N=N/2`) then `round(max(1,s-N):min(len,s+2*N))`. For **odd** N (e.g. N=21 → N=10.5): MATLAB covers `[s-10, s+21]`, Python covers `[s-10, s+20]` — Python removes one fewer trailing sample per spike (leading edge agrees). Confirmed by emulating MATLAB's colon+round semantics for N ∈ {9,11,19,21,41}. Default `N = round(0.04*fs)` is even on common rates (512→20, 256→10), so the benchmark/default path is unaffected; reachable when a caller passes odd N (e.g. `l2.py:121` `round(sh_hw*fs)`, or default rates fs=1024→41, fs=64→3). No test pins odd-N marking extent.
- **Fix:** Compute the trailing extent from the float half-width: `n_after = round(N)` so `hi = min(len(good), s + round(N) + 1)`, matching MATLAB. Or document/enforce even N at entry. Add a regression test asserting marked-region extent for both even and odd N against the MATLAB rule.

#### m13. `merge_p_files` splices trailing partial-record bytes mid-stream, shifting all subsequent records (silent corruption); merge is also never tested with a partial-record input
- **Subsystem:** perturb-trim-merge
- **Location:** `src/odas_tpw/perturb/merge.py:224-235`; test gap `tests/test_perturb_merge.py:10-41`
- **Evidence:** `merge_p_files` copies `chain[0]` in full and appends each continuation from `header_size+config_size` to EOF, with **no whole-record-boundary check anywhere in the module**. If `chain[0]` or any *middle* continuation ends in a fractional record, the stray bytes are spliced mid-stream and shift every following record. Verified: a 3-file chain whose middle file carried 37 trailing bytes produced merged data whose third-file records started 37 bytes late, with `data_bytes % record_size == 37`; `PFile._read` then warns "trailing partial record ignored" and silently drops the final partial while reading the shifted records as garbage. The chain is still detected (`_file_group_key` keys on config/geometry, not size). Reachable via the dedicated `perturb merge` subcommand (no trim) and `files.merge=true & files.trim=false`. The merge test helper `_make_p_file` has no `extra_bytes` knob (unlike the trim helper), so this path is unexercised by CI. Bounded because the default pipeline trims first.
- **Fix:** Clamp each spliced member to a whole-record body: for `chain[0]`, copy `first_record_size + (data_bytes // record_size)*record_size`; for continuations, copy `(file_size - skip) // record_size * record_size`; or raise if `(file_size - first_record_size) % record_size != 0`. Add a `trailing_partial` parameter to the merge test helper and a test asserting the merged data length is a whole multiple of `record_size`.

#### m14. `trim_p_file` and `_read_merge_info` accept geometrically-impossible headers (`record_size < header_size`, `header_size < 128`) that PFile cannot parse
- **Subsystem:** perturb-trim-merge
- **Location:** `src/odas_tpw/perturb/trim.py:132-146`; `merge.py:21-55`
- **Evidence:** `trim_p_file` validates only `record_size <= 0` and `data_bytes < 0`; never `header_size >= HEADER_BYTES` or `record_size >= header_size`. `_read_merge_info` validates nothing past the 128-byte read. Verified: a file with `header_size=512, record_size=128` returns `action='referenced'/'trimmed'` (with `data_words = (record_size-header_size)//2` negative), and `header_size=64` is accepted; both then crash/mis-parse far downstream in `PFile._read` (observed `ZeroDivisionError`), where the error is hard to attribute. `extract_pfile_segment` (p_file.py:138-148) rejects exactly these in the same package. Only bites malformed/non-RSI input.
- **Fix:** Mirror `extract_pfile_segment`: raise `ValueError` when `header_size < HEADER_BYTES` or `record_size < header_size`, so a malformed `.p` is rejected at the trim/merge stage with a clear, source-attributed message.

#### m15. `find_mergeable_files` chains files on sequential `file_number` alone, with no time-continuity / `restarted`-flag guard (concatenates the whole deployment)
- **Subsystem:** perturb-trim-merge
- **Location:** `src/odas_tpw/perturb/merge.py:112-132`
- **Evidence:** Chains any same-`(config_hash,endian,record_size,header_size,config_size)` files whose `file_number == previous+1`, consulting no time-continuity, the RSI `restarted` flag (word 16), `record_number`, `n_records_written`, or `buffer_status` (`merge.py` reads none of these). Reproduced on the real 29 ARCTERX SN479 files: all collapse into a **single** chain (file_numbers 2..30), though header timestamps prove they are independent casts (file 2 = 2025-01-14 15:30, file 3 = 2025-01-15 14:00, …, each `record_number=0`). With `files.merge=true`, `merge_p_files` byte-concatenates the whole deployment into one `.p`, destroying cast boundaries. Gated behind a flag that defaults off and is force-disabled on the common analysis subcommands, but honored by `run` and the dedicated `merge` subcommand. (Note: every ARCTERX file has `restarted=0`, so a `restarted==0` discriminator alone would *not* block the bad merge — only the start-time-contiguity check would.)
- **Fix:** Add a positive split-file discriminator beyond `file_number` adjacency — require start-time contiguity within a tolerance (and/or `restarted==0`) on the continuation before chaining, reading the timestamp words already in `_H`. Until then keep `files.merge` off by default and document that it must only be enabled for genuine size-rollover splits.

#### m16. `epsilon.salinity` has no `'measured'` resolution path (unlike `chi.salinity`), so configuring it crashes deep in `prepare_profiles`
- **Subsystem:** perturb-calibrate-compute
- **Location:** `src/odas_tpw/perturb/pipeline.py:1241-1257`
- **Evidence:** The pipeline resolves `chi.salinity='measured'` per-profile but passes `epsilon.salinity` straight through (`'salinity'` is not in the exclusion set at the `_compute_epsilon` call). A user setting `epsilon.salinity: 'measured'` (a natural symmetry expectation) reaches `prepare_profiles` (helpers.py:333-334) `np.asarray(salinity, dtype=float)` → `ValueError: could not convert string to float: 'measured'`, surfacing only as a per-profile `"diss for ..."` error log and silently empty diss output. The `epsilon` DEFAULTS comment (config.py:399) documents `null = 35` and never mentions `measured`, while the `chi` DEFAULTS do — making this a documentation-asymmetry foot-gun.
- **Fix:** Either reject a non-numeric `epsilon.salinity` at config-merge time with an actionable message ("the measured option is chi-only"), or extend the per-profile measured-salinity resolution to the epsilon path for symmetry.

#### m17. `profile_stratification` leaves unsampled depth gaps (and contradicts its "window/2 spacing" docstring) when `window < 2*min_dp`
- **Subsystem:** processing-mixing
- **Location:** `src/odas_tpw/processing/mixing.py:394-399`
- **Evidence:** `step = max(half_w, min_dp)` with `half_w = window/2`, then `target_P = np.arange(p_lo, p_hi + step, step)`, each window `±half_w` wide. When `window < 2*min_dp` (`half_w < min_dp`), `step = min_dp > half_w`, so consecutive windows do not overlap and `(step - 2*half_w)`-wide gaps are never evaluated for N²/dT/dz. Reproduced: `window=0.1, min_dp=0.2` gives targets 0.2 dbar apart with `±0.05` windows — exactly **50% of the column unsampled**, contradicting the docstring/test contract ("spaced by window/2"). The shipped default `window=2.0, min_dp=0.2` has `half_w=1.0 > 0.2`, so no gap — which is why no test catches it. (Impact is softened at the output level because the caller `np.interp`s the coarse results onto the slow grid, linearly filling gaps rather than leaving NaN holes.)
- **Fix:** Make target spacing equal the window half-width regardless of `min_dp` (`step = half_w`, keep `min_dp` as the per-window validity guard), or clamp `window >= 2*min_dp` with a warning. Update the docstring and add a test with `window < 2*min_dp` asserting no inter-window gap.

#### m18. `bin_by_depth` drops the deepest sample (and can add a spurious empty deepest bin) at non-binary-exact bin widths (`np.arange` float accumulation)
- **Subsystem:** perturb-bin
- **Location:** `src/odas_tpw/perturb/binning.py:440-446`
- **Evidence:** `bin_edges = np.arange(d_min, d_max + bin_width, bin_width)`; for non-binary-exact `bin_width`, float error makes `bin_edges[-1] != d_max`. **Overshoot** (e.g. `bw=0.2, g_max=49.7` → `bin_edges[-1]=50.0000…4`) creates an extra trailing bin that no in-range sample can fill → spurious all-NaN deepest row (reproduced in 4267/~50000 scanned width/range combos). **Undershoot** (e.g. `bw=0.1`, deepest sample at an exact bin-edge multiple) makes the `coords == bin_edges[-1]` last-bin guard miss, `np.digitize` returns `idx==n_bins`, and the deepest real sample is silently dropped (7610 such cases in the scan). Production ARCTERX configs use `width: 1.0` (binary-exact) so it is latent there, but sub-metre widths are valid supported inputs. `bin_by_time` is unaffected (it does not snap `t_max` up). (Note: the originally-proposed fix `d_min + bin_width*np.arange(n_bins+1)` pins the bin *count* but is not bit-exact to `d_max`.)
- **Fix:** Build edges from an exact integer bin count: `n_bins = int(round((d_max - d_min)/bin_width)); bin_edges = d_min + bin_width*np.arange(n_bins + 1)` — eliminates the extra/missing trailing bin. Combine with a tolerant last-bin snap in `_bin_indices` (`coords >= bin_edges[-1]` plus an explicit `coords <= d_max` range check) so the deepest sample is never dropped. Add a test resolving sub-metre widths and asserting no dropped/empty deepest bin.

#### m19. Legacy chi binning uses an order-dependent running 0.5-average, not a mean — biased toward the last sample and inconsistent with its own docstring
- **Subsystem:** perturb-plot
- **Location:** `src/odas_tpw/perturb/plot/eps_chi.py:87`
- **Evidence:** `chi[i, j] = v if np.isnan(cur) else 0.5*(cur + v)` when multiple source values land in one depth bin. For `[1,2,9]` this yields 5.25 (weights 0.25/0.25/0.5) vs true mean 4.0; reversing the order yields 3.25 — order-dependent and biased toward the last-digitized sample. The companion `_reindex_rows_to_depth` uses a true mean (`acc/cnt`) yet its docstring (line 112) claims `reduce='mean'` is "consistent with the legacy path." Reachable when 3+ chi time-windows fall in one (coarser) eps depth bin (chi and diss bin widths are separately configurable). Plotting-only and a legacy fallback (fires only when `chiMean` is absent from the combo), so blast radius is narrow.
- **Fix:** Accumulate parallel `acc`/`cnt` arrays and finalize `chi = np.where(cnt>0, acc/cnt, np.nan)` (a true nanmean), making the "consistent with the legacy path" docstring claim true.

#### m20. L1 converter emits non-UDUNITS unit strings on auxiliary channels (`micro_Tesla`, `FTU`, raw supplementary units); `test_cf_compliance` never validates units against UDUNITS
- **Subsystem:** cross-cf-units
- **Location:** `src/odas_tpw/rsi/convert.py:361 (MAG), 383 (TURB), 541 (supplementary)`; test gap `tests/test_convert.py:149-177`
- **Evidence:** The L1 product declares `Conventions="CF-1.13, ACDD-1.3"` (convert.py:478) but three classes of emitted units fail `cf_units`: (a) **MAG** hardcodes `units="micro_Tesla"` (the converter itself already returns the parseable `'uT'`, which is overridden) — *newly reachable* after the round-1 `magn`-classification fix; (b) **TURB** hardcodes `units="FTU"` — running `p_to_L1` on `SN479_0002.p` yields exactly this one UDUNITS failure, and the perturb schema already maps `FTU→"1"` via `canonicalize_units`, which the L1 path bypasses; (c) **supplementary** channels write `v.units = info["units"]` verbatim (e.g. `'umol_L-1'`, `'deg'` from `convert_poly`/`inclxy`), again skipping `canonicalize_units` that the per-profile `profile.py` path applies. `test_cf_compliance` asserts only `Conventions`/time/PRES and never parses any `units` with `cf_units`, and the SN479 sample has no magnetometer, so all three are untested. Metadata-only (no effect on computed values; NetCDF writing does not crash).
- **Fix:** Route all hardcoded and supplementary L1 unit strings through `perturb.netcdf_schema.canonicalize_units` (as `profile.py` already does), or directly set `MAG→"uT"`, `TURB→"1"` (document "formazin turbidity units" in `long_name`). Add a test looping `cf_units.Unit(v.units)` over every L1 variable with a `units` attr (asserting no parse error), and synthesize a config with a `magn` channel to exercise the MAG spec. (`cf_units` is not a declared dep — add it as a dev dep or use `importorskip`.)

#### m21. Stale mypy per-module override list references three nonexistent `rsi` modules (config drift)
- **Subsystem:** packaging-deps
- **Location:** `pyproject.toml:63-83`
- **Evidence:** The `[[tool.mypy.overrides]]` module list names `odas_tpw.rsi.spectral`, `odas_tpw.rsi.ocean`, `odas_tpw.rsi.nasmyth` — none of which exist (`find` confirms no such files; `git log --all` shows they never existed under `rsi/`; `rsi/__init__.py` has no shims). The real modules live in `scor160/` and *are* separately and correctly listed. mypy silently ignores overrides targeting nonexistent modules, so the broad `disable_error_code` suppression is a no-op for these names — but if a future `rsi/spectral.py` (etc.) is added, it would silently inherit broad error suppression. `mypy src/odas_tpw/` currently passes (no active breakage).
- **Fix:** Delete the three phantom entries. Optionally add a CI/pre-commit assertion that every module named in the override list resolves to a real file.

#### m22. `resolve_output_dir` 2-digit glob vs `:02d` format mismatch causes directory collision and silent provenance corruption past 100 configs
- **Subsystem:** rsi-config
- **Location:** `src/odas_tpw/config_base.py:241 (glob), 256 (format)`
- **Evidence:** Discovery globs `f"{prefix}_[0-9][0-9]"` (exactly two digits) but new dirs use `f"{prefix}_{next_seq:02d}"`, which widens to three digits for `next_seq >= 100`. Once >100 distinct configs accumulate under one base: (1) `eps_100+` are invisible to the glob, so `max_seq` caps at 99 and `next_seq` becomes 100 again, **colliding** with the existing `eps_100`; (2) reuse-on-hash-match silently breaks for all 3-digit dirs (their signature is never scanned). Reproduced: after 101 distinct configs (`eps_00..eps_100`), a new distinct config lands in `eps_100`, leaving **two distinct `.params_sha256_*` signature files in one directory** with co-mingled output products and no error — defeating the param→dir provenance the signature mechanism exists to guarantee. Triggering requires 100+ distinct configs under one base for one stage — realistic only for a large parameter sweep, hence minor.
- **Fix:** Make the glob and formatter agree: scan with `f"{prefix}_[0-9]*"` (or regex `^{prefix}_(\d+)$`) so 3+ digit dirs are discovered for both `max_seq` and signature reuse; keep `{next_seq:02d}` (it widens naturally). Add a test resolving 101+ distinct configs and asserting each gets a unique dir and reuse works for dir 100.

#### m23. YAML date/datetime config values crash `compute_hash`/`resolve_output_dir` with an opaque `TypeError`
- **Subsystem:** rsi-config
- **Location:** `src/odas_tpw/config_base.py:41` (reached via `canonicalize` json.dumps at 209)
- **Evidence:** ruamel.yaml parses unquoted ISO dates/timestamps into `datetime.date`/`datetime` objects. `_normalize_value` handles None/bool/int/float/str then falls through with a bare `return v`, passing the date object unchanged into `json.dumps`, which raises `TypeError: Object of type date is not JSON serializable`, crashing `compute_hash → resolve_output_dir`. **Reachable end-to-end through a valid perturb config**: the `netcdf` section DEFAULTS define date fields (`date_created`, `time_coverage_start`, …) that flow into `resolve_output_dir` via `_upstream_for`. The maintainer already added a non-finite-float guard (lines 31-35) for exactly this call chain "to avoid an opaque message" — but datetime (a far more common YAML artifact than NaN) was not guarded. The shipped template presents these as `null`, inviting a user to fill a bare ISO date. Easy workaround (quote the value), fails fast at directory setup (not data corruption).
- **Fix:** In `_normalize_value`, before the final `return v`, add `if isinstance(v, (datetime.date, datetime.datetime)): return v.isoformat()` (or `return repr(v)` mirroring the float branch, or raise a clear `ValueError` naming the section/key). Add a regression test hashing a `netcdf` config with a bare `date_created`.

> Note: m22 and m23 are both `config_base.py` defects but distinct root causes (sequence-glob mismatch vs JSON-unserializable scalar); kept separate.

---

### NIT (35)

#### n1. Unsigned-wrap step runs AFTER deconvolution, reversing ODAS order (latent for unsigned/32-bit deconvolved channels)
- **Subsystem:** rsi-pfile-numeric — `src/odas_tpw/rsi/p_file.py:498-523`
- **Evidence:** `_read` calls `_apply_deconvolution` then performs the signed→unsigned wrap; ODAS wraps (read_odas.m:370-398) *before* deconvolving (odas_p2mat.m:516-570). A synthetic deconvolve on signed vs pre-wrapped int16 differs by ~34731 counts. **Inert on ARCTERX** (deconvolved channels are all signed; unsigned channels carry no `diff_gain`, so the intersection is empty; MATLAB validation matches at rtol=1e-10). The 32-bit 2-id path already wraps before deconvolution.
- **Fix:** Move the unsigned-wrap loop before `_apply_deconvolution` to restore ODAS ordering.

#### n2. RINKO `aroft_o2`/`aroft_t` glitch filter omitted — spurious DO/DO_T spikes survive into output
- **Subsystem:** rsi-pfile-numeric — `src/odas_tpw/rsi/channels.py:262-271`
- **Evidence:** ODAS `odas_aroft_o2_internal`/`_t_internal` apply a "buggy RS232 receiver" glitch filter (`if input(i)<=50 || input(i)>=2^16-50, input(i)=input(i-1)`); the Python ports do only `d/100.0` / `d/1000.0-5.0`. RINKO is present in ARCTERX; mid-record dropouts in ~1/3 of files (e.g. file 0007 idx ~5065: DO drops to raw 0 for ~120 samples) make Python write `0.0 µmol/L` where ODAS holds the last-good ~243. Auxiliary DO sensor, not in the ε/χ pipeline; downstream-refilterable. `test_channels.py` never exercises a glitch value mid-array (`-1` wraps to 65535, a glitch value, yet asserts a valid output — confirming the filter is absent).
- **Fix:** Forward-fill masked positions `(d<=50)|(d>=2**16-50)` (index 0 never filtered) before scaling, in both converters. Add a regression test feeding glitch samples.

#### n3. `_load_from_nc` days→seconds conversion indexes `t_fast[0]` without an empty-array guard
- **Subsystem:** rsi-pfile-robust — `src/odas_tpw/rsi/profile.py:437-439`
- **Evidence:** `t_fast = (t_fast - t_fast[0]) * 86400.0` raises `IndexError` on a zero-length TIME dim; the seconds path doesn't index element 0. Only fires for a malformed ATOMIX-format NC whose time units contain "day" and TIME is empty.
- **Fix:** `if t_fast.size: t_fast = (t_fast - t_fast[0]) * 86400.0` (same for t_slow), or raise an explicit "no time samples" error earlier.

#### n4. PFile reshape raises opaque `ValueError` when `data_words` is not a positive multiple of `n_cols`
- **Subsystem:** rsi-pfile-robust — `src/odas_tpw/rsi/p_file.py:377-379`
- **Evidence:** `reshape(total_scans, n_cols)` mismatches the element count when `data_words % n_cols != 0` or `data_words < n_cols`, raising a cryptic `cannot reshape array of size N into shape (M,n_cols)`. Only bites a corrupted/forged header (real files have `data_words % (n_rows*n_cols) == 0`); a deterministic `ValueError` already, so this is purely diagnostic-message quality. (The "ODAS sidesteps this" claim is misleading — its reshape would throw equivalently.)
- **Fix:** Validate before the reshape and emit a clear, source-named error.

#### n5. PFile reshape — see n4 (merged: `matrix_count==0` empty-output case is m4, kept at minor; the opaque-reshape variant is n4).

#### n6. Profile detection uses raw `P` while extracted/L1 pressure uses deconvolved `P_dP` (inconsistent source)
- **Subsystem:** rsi-pfile-robust — `src/odas_tpw/rsi/profile.py:316`
- **Evidence:** `_load_from_pfile` returns `"P": pf.channels["P"]` (raw) for fall-rate/profile detection, whereas `convert.py:39/471`, `adapter.py`, `pipeline.py`, and even `profile.py:39` (`_compute_speed`) prefer `P_dP`. So `profile_mean_speed` derives from raw-P-smoothed W while the speed actually used in dissipation uses P_dP. Both slow-rate, so bounds are nearly identical — negligible numerical effect.
- **Fix:** `"P": pf.channels.get("P_dP", pf.channels["P"])` to mirror the rest of the pipeline.

#### n7. GRADT `IndexError` on a 1-sample fast thermistor channel — *merged into m3* (concrete trigger for the exception-allowlist gap). Listed here for traceability; reachable only on a degenerate `n_rows==1` `.p`.

#### n8. `scaffold_yaml` interpolates the `vehicle` value into the YAML template without escaping
- **Subsystem:** rsi-config-patch-security — `src/odas_tpw/rsi/config_patch.py:573`
- **Evidence:** The active `vehicle:` line uses raw f-string interpolation; a value containing `"` makes ruamel raise `ScannerError`. (The finding's lines 583/595 are *commented* lines and load cleanly — overstated; only line 573 holds.) The data source is the user's own `.p` config; real configs don't contain quotes; worst case is a clear YAML error, not corruption.
- **Fix:** Emit the value via a quoted scalar (ruamel `DoubleQuotedScalarString`) or skip/escape values containing `"`.

#### n9. Whitespace-only `author` collapses to an empty author in the provenance banner
- **Subsystem:** rsi-config-patch-security — `src/odas_tpw/rsi/config_patch.py:189-194`
- **Evidence:** `author = str(author).strip() if author else (...USER...)`. A `"   "` author passes validation, is truthy, then `.strip()` → `''`, bypassing the `$USER`/`'unknown'` fallback; the banner emits an empty author. `note` (line 183) *is* guarded against whitespace-only; `author` is not. Cosmetic (audit comment only).
- **Fix:** `stripped = str(author).strip() if isinstance(author, str) else ''; author = stripped or os.environ.get('USER') or 'unknown'`.

#### n10. Provenance banner accumulates without bound on repeated re-patches
- **Subsystem:** rsi-config-patch-security — `src/odas_tpw/rsi/config_patch.py:434-435`
- **Evidence:** `edit_config_text` always prepends a fresh banner; the prior banner is inside `active` and preserved, so banners stack (3 after 3 patches). `CONFIG_MARKER` stays at 1; inline `[PATCH ...]` lines are intended to accumulate. Cosmetic bloat bounded by the 65535-byte `config_size` limit (a clean `ValueError`, never corruption).
- **Fix:** On re-patch (`marker_idx >= 0`), strip any prior leading `; ===`-delimited banner before prepending, or emit the banner only on first patch.

#### n11. Reported `epsilonLnSigma`/`chiLnSigma` includes dropped-outlier probes' sigma (computed once before iterative removal)
- **Subsystem:** processing-combine — `src/odas_tpw/processing/epsilon_combine.py:120-123,167-186`; `chi_combine.py:146,205-209`
- **Evidence:** `mu_sigma` is computed once before the removal loop and written as `epsilonLnSigma`, so a dropped low-junk probe (large sigma) still inflates the *reported* uncertainty even though it didn't enter the mean. Faithful to the reference for the CF95 threshold; the docstring documents the cross-probe averaging design. Diagnostic-only (not fed back into selection), and gated on `n_probes>=3` so it never fires on the 2-probe campaign data.
- **Fix:** Recompute `mu_sigma` over the surviving (post-removal) mask before writing, or extend the docstring to state that dropped outliers still contribute.

#### n12. Unreachable `kB_best < 1` guard in `_iterative_fit`; NaN-leak guard keys on `isfinite`, not `>= 1`
- **Subsystem:** chi-physics — `src/odas_tpw/chi/chi.py:538`
- **Evidence:** `_mle_find_kB` can never return `kB<1` (`_KB_COARSE` min is exactly 1.0; fine grid clamps `max(...,1.0)`), so the `kB_best < 1` branch is dead. The final NaN-leak guard (582-588) keys on `np.isfinite(kB_best)`, not `>=1`, so if the grid floor were ever lowered below 1.0 a `kB_best in [floor,1)` would `break` yet still leak a finite chi/epsilon — inconsistent with Method 1, which NaNs out `kB<1`. Purely latent.
- **Fix:** Set `kB_best = np.nan` before the `break` when `kB_best < 1`, or gate the final chi/epsilon on `np.isfinite(kB_best) and kB_best >= 1`.

#### n13. `fp07_tau` (`lueck`/`peterson`) unguarded for non-positive speed, producing NaN/inf and warnings
- **Subsystem:** chi-spectra — `src/odas_tpw/chi/fp07.py:122-125`
- **Evidence:** `0.01*(1/speed)**0.5` and `0.012*speed**-0.32` give inf at speed=0 and NaN at negative speed (verified). All three live callers floor/`abs` speed upstream, so production impact is nil; but `fp07_tau`/`fp07_tau_batch` are public API and the module's noise paths use explicit errstate guards — these are the inconsistent outlier.
- **Fix:** Validate `speed > 0` at entry (raise or floor with a warning), or guard the branches so non-positive speed does not silently emit NaN.

#### n14. `despike` replacement value diverges from ODAS when one averaging side has no valid neighbors (Python finite/0.0 vs MATLAB NaN)
- **Subsystem:** scor160-despike-nasmyth — `src/odas_tpw/scor160/despike.py:157-164`
- **Evidence:** ODAS fills `sum([])/length([]) = NaN`; Python branches to a single-side mean or 0.0. Identical input yields NaN (ODAS) vs finite (Python). Reflected end-padding almost always guarantees both sides have neighbors; only reachable on contrived tiny (~2-60 sample) inputs, never real diss-length sections. Python's behavior is arguably more robust.
- **Fix:** Add a comment documenting the intentional, more-robust deviation (preferred), or write NaN for exact ODAS parity.

#### n15. `despike` raises uncaught `filtfilt` ValueError on very short arrays (n<=2 at default fs)
- **Subsystem:** scor160-despike-nasmyth — `src/odas_tpw/scor160/despike.py:98-114`
- **Evidence:** For n<=2, padded length `3*n <= 6 = filtfilt padlen` → opaque scipy `ValueError`. Mirrors MATLAB `filtfilt`; all real callers operate on long sections (one even guards `sig.size < 16`), so not triggerable in practice. No test for n<=2.
- **Fix:** Early guard: if padded length <= padlen (or `len(signal) < 3`), return the input unchanged with `n_passes=0`. Add a unit test for n in {0,1,2,3}.

#### n16. `np.real()` view aliasing — `_sanitize_autospectra` mutates a view into the discarded complex `clean_UU`
- **Subsystem:** scor160-spectral — `src/odas_tpw/scor160/goodman.py:201,267 (callers); 39 (mutation)`
- **Evidence:** `np.real(complex)` is a non-copying view; `_sanitize_autospectra` writes into it, overwriting the real part of the original. Harmless today only because the complex `clean_UU` is dead after the rebind and doesn't alias the returned `UU`. The test masks it by passing `m.copy()`. Pure maintainability/fragility; no current correctness impact.
- **Fix:** `_sanitize_autospectra(np.real(clean_UU).copy())` at both call sites, or document the in-place mutation contract.

#### n17. Custom-window length not validated → cryptic broadcast error instead of ODAS's clear message
- **Subsystem:** scor160-spectral — `src/odas_tpw/scor160/spectral.py:201`
- **Evidence:** `csd_matrix` accepts a user `window` with no length check; a wrong length fails deep in the loop with `operands could not be broadcast together (256,) (128,)`. ODAS `csd_matrix_odas.m:124-126` validates up-front. Internal callers always use the default window; loud failure, just a poor diagnostic.
- **Fix:** After resolving `window`, `if window.shape != (nfft,): raise ValueError(...)`, mirroring the ODAS message.

#### n18. Python CSD divides BOTH endpoints by 2 for Cxx/Cyy (intentional, more-correct divergence from a MATLAB asymmetry) — informational, no action
- **Subsystem:** scor160-spectral — `src/odas_tpw/scor160/spectral.py:247-250,440-443`
- **Evidence:** ODAS halves only Cxx-DC and Cyy-Nyquist (a long-standing asymmetry); Python symmetrically halves both DC and Nyquist for all three matrices — the physically correct single-counting for a one-sided real spectrum. The dissipation band excludes the exact DC/Nyquist bins, so no effect on ε/χ; the ATOMIX benchmark passes.
- **Fix:** No code change. Optionally add a one-line comment noting the intentional divergence so a reviewer does not "restore" the MATLAB behavior.

#### n19. `compare_l2` raises `ValueError` on empty `section_number` (no `n_common` guard like `compare_l3/l4`)
- **Subsystem:** scor160-levels — `src/odas_tpw/scor160/compare.py:51-52`
- **Evidence:** Unconditional `int(reference.section_number.max())` raises on a size-0 array; `compare_l3`/`compare_l4` early-return on `n_common==0`. Reachable since `_select_sections` now returns empty on empty input, but benchmark `.nc` files are never empty.
- **Fix:** `if ref_sec.size == 0 or comp_sec.size == 0: return results` (zeros), mirroring the siblings.

#### n20. scor160 L4 ignores the ODAS `f_limit` constraint when forming `K_AA` (uses `0.9*f_AA` only)
- **Subsystem:** scor160-levels — `src/odas_tpw/scor160/l4.py:180`
- **Evidence:** `K_AA = 0.9*f_AA/max(W,0.05)`; ODAS additionally applies `if f_limit < f_AA, f_AA = f_limit` (and the RSI sibling honors it). ODAS default `f_limit = inf`, and ATOMIX NCs carry no `f_limit`, so `min(0.9*f_AA, f_limit)` always reduces to `0.9*f_AA` for every benchmark/CLI invocation — bit-identical to ODAS.
- **Fix:** Accept an optional `f_limit` in `process_l4`/`_estimate_epsilon` and apply `f_AA_eff = min(0.9*f_AA, f_limit)`, or document that the scor160 path assumes `f_limit` non-constraining.

#### n21. Bottom-crash aborts on the first length-mismatched channel; `top_trim` skips it (asymmetric contract)
- **Subsystem:** processing-trim-crash — `src/odas_tpw/processing/bottom.py:80-86`
- **Evidence:** `detect_bottom_crash` does `if len(a) != len(depth): return None` (aborts crash detection for the whole profile), while `compute_trim_depth` does `continue` (skips the bad channel). In production all fast channels share one demultiplexed length sliced with identical bounds, so the mismatch branch is effectively dead code on real data.
- **Fix:** Replace the early `return None` with `continue` and require at least one usable channel afterward (`if mag_sq is None: return None`), matching `top_trim`. Add a "good-axis + bad-axis" test.

#### n22. Trim skip-guard still serves stale content for an equal-trimmed-size content change with equal/older mtime
- **Subsystem:** perturb-trim-merge — `src/odas_tpw/perturb/trim.py:165-171`
- **Evidence:** The size+mtime cache key is fooled by a source edited to the exact same trimmed size then restamped with an equal/older mtime (`cp -p`/rsync/backup-restore). Reproduced: re-trim returned `action='skipped'` and the output kept stale bytes. The inherent limitation of any size+mtime key; adversarial trigger; a `force_trim` escape hatch already exists; the docstring does not over-claim coverage.
- **Fix:** Document the residual hole, or for full content-correctness compare a content digest of the first record rather than size+mtime alone.

#### n23. FP07 in-situ lag search restricts argmax to negative lags instead of ODAS's full-window-search-then-clamp
- **Subsystem:** perturb-calibrate-compute — `src/odas_tpw/perturb/fp07_cal.py:149-156`
- **Evidence:** Python searches only the negative half; ODAS `cal_FP07_in_situ.m:502-533` finds the global `|corr|` peak then clamps `>0` to 0. They differ when the global peak is at positive lag (ODAS → 0 shift; Python → best negative-side peak). Per-profile median damps outliers; on ARCTERX FP07 physically leads the reference so peaks are expected negative.
- **Fix:** Search the full window for `argmax(|corr|)`, compute the lag, then clamp `lag>0` to 0.

#### n24. JAC low-pass cutoff uses `mean(|W|)` where ODAS uses `|mean(W)|`, differing on mixed up/down profile sets
- **Subsystem:** perturb-calibrate-compute — `src/odas_tpw/perturb/fp07_cal.py:95-101`
- **Evidence:** Identical for single-signed (pure downcast) profiles, so **zero** effect on the shipped downcast ARCTERX data; differs only when up+down casts are pooled (glide mode). The `sqrt` damps the effect. (The proposed `abs(mean)` fix is arguably *worse* for a symmetric glide where `mean(W)≈0`; the genuine deviation is the Python pooling all profiles vs ODAS being per-profile.)
- **Fix:** Document the intentional pooling, or match ODAS's per-profile convention if exact parity is desired. Low priority.

#### n25. Chi viscosity mixes FP07 (T1) temperature with JAC_T-derived salinity under `chi.salinity='measured'`
- **Subsystem:** perturb-calibrate-compute — `src/odas_tpw/rsi/chi_io.py:82,105-107`
- **Evidence:** Salinity comes from JAC_C/JAC_T/P, but the temperature feeding `visc()` is the calibrated FP07 (T1, the `load_channels` default), not JAC_T. After in-situ calibration T1 ≈ JAC_T (<0.1 °C residual) and viscosity is weakly T-sensitive, so the chi effect is sub-percent; the provenance is mixed and undocumented.
- **Fix:** Document that chi viscosity temperature is the FP07 channel while measured salinity uses JAC, or source the viscosity temperature from JAC_T for the measured-salinity path.

#### n26. `chi.salinity` string other than `'measured'` is passed through and only fails at a deep guard
- **Subsystem:** perturb-calibrate-compute — `src/odas_tpw/perturb/pipeline.py:1312-1351`
- **Evidence:** Only the exact (stripped, lowercased) `'measured'` is special-cased; a typo (`'meas'`) flows to `_compute_chi` and raises `ValueError` at `chi_io.py:71-79`, per-profile and late rather than once at config time. Loud and deterministic (cannot silently produce wrong chi).
- **Fix:** Validate `chi.salinity` once before the profile loop (accept `'measured'`, a number, or None; raise once naming the offending value), keeping the deep guard as defense-in-depth.

#### n27. Explicit `--clim`/`--vmin` minimum on a log variable is silently clamped to 6 decades below the max; no test covers the clamp or the profiles/eps_chi inconsistency
- **Subsystem:** perturb-plot — `src/odas_tpw/perturb/plot/profiles.py:155`; test gap `tests/test_perturb_plot_profiles.py:96`
- **Evidence:** `_make_norm` log path does `vmin = max(vmin, vmax/1e6)` *after* honoring the user's explicit limit, so an 8-decade `--clim` request (`1e-14`) is silently truncated to `1e-12`. `eps_chi`'s `_safe_lognorm` applies no decade clamp, so the same explicit limits are honored there but clamped here — the two subcommands diverge. No test asserts an explicit `--clim` minimum survives. Presentation-only (colorbar floor), narrow trigger (>6-decade explicit request).
- **Fix:** Gate the clamp on `if not explicit:` (the `explicit` flag already exists at line 114), so user-supplied minima pass through verbatim, matching `eps_chi`. Add a regression test asserting an explicit `--clim` minimum is preserved in `LogNorm.vmin`, and one asserting profiles and eps_chi treat the same explicit log limits identically.

#### n28. `eps_chi` negative `--eps-vmin`/`--chi-vmin` silently yields a "no data" panel instead of an error
- **Subsystem:** perturb-plot — `src/odas_tpw/perturb/plot/eps_chi.py:376`
- **Evidence:** `profiles._make_norm` rejects a non-positive log minimum with a clear `SystemExit`; `eps_chi` has no such guard — a negative `*_vmin` flows through `quantile_limits`, `_safe_lognorm` returns None on `vmn<=0`, and the panel renders the "no finite data" placeholder despite finite positive data. User misuse only; worst case a misleading label.
- **Fix:** Validate explicitly-supplied `*_vmin > 0` and `SystemExit` with a message, mirroring `profiles.py`.

#### n29. latitude/longitude x-axis is dateline-unsafe in profiles/scalar
- **Subsystem:** perturb-plot — `src/odas_tpw/perturb/plot/profiles.py:208`
- **Evidence:** Columns are sorted by raw `x`; the `longitude` x-axis method returns unwrapped degrees, so a transect crossing ±180° gets reordered and stretched (179.5°E and 179.5°W render 359 units apart). The distance/along_line/signed methods *are* dateline-safe. Root cause in `xaxis.py` (out of scope); presentation-only; Saipan/ARCTERX (~145°E) is far from the dateline.
- **Fix:** Document the `longitude` limitation, or (preferred) unwrap finite longitudes about their circular mean before sorting/binning when they span the dateline (centrally in `xaxis.compute`'s `longitude` branch).

#### n30. `_canonicalize_section`/`write_resolved_config` crash with `AttributeError` on `None` params for static sections (inconsistent with the dynamic-key guard)
- **Subsystem:** rsi-config — `src/odas_tpw/config_base.py:178,293,302`
- **Evidence:** Dynamic-key branches use `(params or {}).items()`; static branches do bare `params.items()`/`up_params.items()`. `canonicalize('chi', {}, upstream=[('epsilon', None)])` and `write_resolved_config(...)` raise `AttributeError`. Publicly re-exported, but no in-repo caller can trigger it (every `params` originates from `merge_config`, which always returns a dict; `load_config` normalizes None-bodied sections to `{}`). Purely latent.
- **Fix:** Make the static branches None-tolerant: `(params or {}).items()` / `(up_params or {}).items()`.

#### n31. `write_signature`/`write_resolved_config` reject a `str` directory argument, inconsistent with the `str | Path` contract of siblings
- **Subsystem:** rsi-config — `src/odas_tpw/config_base.py:271,307`
- **Evidence:** `load_config`/`resolve_output_dir`/`generate_template` accept `str | Path` and coerce; these two annotate `directory: Path` and do `directory / "..."`, so a plain `str` raises `TypeError`. Fully latent in-tree (only callers pass a Path) and mypy-guarded for type-checked callers.
- **Fix:** Change signatures to `directory: str | Path` and add `directory = Path(directory)` as the first line of both.

#### n32. Unknown keys in `params` are silently dropped by `canonicalize`/`merge_config` instead of validated (a typo'd key hashes identically to the clean config)
- **Subsystem:** rsi-config — `src/odas_tpw/config_base.py:164,178-179`
- **Evidence:** `validate_config` rejects unknown keys but is only called by `load_config`; the hashing/merge path keeps only `if k in base`/`if k in merged`. Verified: `compute_hash('epsilon', {'totally_bogus_key': 12345}) == compute_hash('epsilon', {})`. Every production path is shielded (CLI loads via `load_config` first; overrides come from a hardcoded mapping), so the blast radius for file/CLI users is nil; only a hand-built dict bypassing `load_config` hits it.
- **Fix:** Optionally re-validate at the top of `canonicalize`/`compute_hash`, or document the pre-validation precondition. At minimum add a test asserting an unknown key either raises or is documented as intentionally ignored.

#### n33. Test gap: `test_bin_edges_too_few_returns_none` does not exercise the `len(bin_edges) < 2` guard it claims
- **Subsystem:** processing-trim-crash — `tests/test_processing_top_trim.py:55-63`
- **Evidence:** With `dz=1.0, min_depth=max_depth=10.0`, `bin_edges = [9.5, 10.5]` (two edges) — the guard is *not* hit; the function returns None via the unrelated `np.sum(valid) < 3` path. The docstring/comment (and a cited line number) are wrong. The real `len(bin_edges) < 2` guard (reachable only via inverted bounds, e.g. `min_depth=20, max_depth=10`) is untested.
- **Fix:** Add a test with inverted bounds (`min_depth=20.0, max_depth=10.0`) that actually empties `bin_edges`, and correct the misleading comment.

#### n34. pandas used directly by the perturb subpackage but not declared as a direct dependency (relies on xarray transitively)
- **Subsystem:** packaging-deps — `pyproject.toml:16-25`
- **Evidence:** perturb imports pandas at `gps.py:214/216`, `hotel.py:177/186/285`, but pandas is absent from `[project].dependencies`. It resolves only because every perturb module imports xarray, whose metadata hard-requires `pandas>=2.2`. PEP 508 best practice is to declare every directly-imported runtime dependency; the lazy function-level imports mean any failure surfaces only at CSV/ISO runtime.
- **Fix:** Add `"pandas>=2.2,<4"` to `[project].dependencies` (floor matches xarray; pandas 3.x is already in use).

#### n35. `make_combo` schema overwrite downgrades correct `units="1"` to empty-string units on `epsilonLnSigma`/`FM`/`fom` diagnostics
- **Subsystem:** cross-cf-units — `src/odas_tpw/perturb/netcdf_schema.py:235,240,245,319,324`
- **Evidence:** `COMBO_SCHEMA`/`CHI_SCHEMA` assign `units=""` to five dimensionless diagnostics, and `apply_schema` unconditionally writes schema attrs, so the combo carries `units=""` — inconsistent with every other dimensionless var in the package (`mad`, `fom`, `K_max_ratio`, `Gamma` all use `"1"`). (The binning steps actually strip all per-variable attrs first, so apply_schema is the *sole* units source, not a downgrader — but the inconsistent end state is real.) Empty-string units are CF-tolerated as dimensionless; a compliance-checker warning + internal wart, not a data defect.
- **Fix:** Replace the five `units=""` entries with `units="1"`.

---

## 3. De-duplication Notes

The raw confirmed set was 49 findings; 43 remain after merging seven overlapping
clusters (each merge combines a defect with its coverage sibling or its concrete
trigger reported under a separate lens):

| Merged into | Absorbed finding(s) | Rationale |
|---|---|---|
| **m3** (convert_all narrow allowlist) | GRADT IndexError on 1-sample channel (nit) | The IndexError is the concrete trigger for the allowlist gap; same fix locus. Listed as n7 for traceability. |
| **m13** (merge splices partial-record bytes) | Test gap: merge never tested with partial record (nit) | The test gap is the direct coverage sibling of the splice bug. |
| **m20** (L1 non-UDUNITS units) | MAG `micro_Tesla`, TURB `FTU`, supplementary raw units, `test_cf_compliance` no-UDUNITS test gap (4 minors) | One root cause (L1 convert path never routes unit strings through `canonicalize_units`) + its coverage sibling. |
| **n27** (explicit-clim decade clamp) | No test covers the clamp / profiles-eps_chi inconsistency (nit) | The test gap is the coverage sibling of the clamp defect. |

The `epsilon.py does not exist` confirmed entry is **not a defect** (verifier
classified it `not-a-bug` — epsilon logic lives in `scor160/l4.py`, which was
audited); it is excluded from the counts and recorded here as a scope note only.

The two `config_base.py` config-resolution items (m22 sequence-glob mismatch, m23
date crash) and the two `config_base.py` static-section/None items (n30, n32)
were kept separate — they are distinct root causes, not the same defect viewed
through different lenses.

---

## 4. Coverage Gaps / Next Round

**Reported coverage gaps:** none. Every finder key ran (the orchestrator's
`COVERAGE GAPS` list was empty).

**Recommended focus for the next round** (areas this audit touched but did not
exhaust, ordered by expected payoff):

1. **chi numeric-accuracy regression suite (highest priority).** M1 shows the
   variance-correction grid is unconverged on the *default* path. Before fixing,
   add a convergence harness that sweeps `n_fine` against a high-`n` reference
   across the realistic `(kB, K_max, K_min)` envelope for both `iterative` and
   `mle` methods and both `batchelor`/`kraichnan` models — the current tests
   assert no such accuracy bound, which is how this persisted.

2. **External / partial NetCDF input fuzzing.** M2 (fill-value leak) was found in
   `_load_from_nc`; the same `[:].data`-vs-`filled` pattern, masked-array
   handling, and empty-dimension assumptions likely recur elsewhere in the NC
   read paths (`helpers._channels_from_nc`, ATOMIX `io.py`, adapter). Audit every
   `.data` / `[:]` extraction for masked-array correctness and add a CF/ATOMIX
   external-NC test corpus (declared `_FillValue`, days-unit time, empty TIME).

3. **NaN-propagation sweep across processing.** M3 (ct_align) and m8
   (dissipation viscosity) are two instances of "a guard compares `<= 0` and is
   False for NaN." Systematically grep the processing/perturb/chi code for
   `<= 0` / `>= 0` / `> 0` guards on quantities derived from instrument channels
   and verify each is NaN-safe; the IIR-filter NaN-smearing in particular
   (`lfilter`/`filtfilt`) deserves a dedicated check at every call site.

4. **Depth-binning and bin-center geometry.** M4 (bottom-crash center) and m18
   (depth-bin float `arange`) are both float-grid edge-geometry bugs near the
   deepest sample. Audit every `np.arange(min, max+step, step)` and every
   bin-center / bin-edge construction in `binning.py`, `top_trim.py`,
   `bottom.py`, and `mixing.py` for the overshoot/undershoot/over-read class.

5. **Config-value validation layer.** m11 (`fp07_transfer_batch` silent
   fallthrough), m16 (`epsilon.salinity`), m23 (date crash), n26
   (`chi.salinity`), and n32 (unknown-key drop) all stem from `validate_config`
   checking only key *names*, not *values* or *types*. A single value-validation
   pass (enum/range/type per key, run once at config load) would close most of
   them at once.

6. **CF/UDUNITS as a CI gate.** m20 and n35 are all metadata non-compliances that
   a single "every `units` attr must `cf_units.Unit()`-parse" test (over both the
   L1 convert output and the perturb combo) would catch and prevent regressing.

7. **`xaxis.py`** was explicitly out of scope for n29 (dateline) but is the root
   cause; include it next round.

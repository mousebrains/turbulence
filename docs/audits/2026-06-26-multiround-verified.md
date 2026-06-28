# Multi-Round Deep Review — Verified Findings (2026-06-26, PR #78 branch)

Branch: `fix/audit-2026-06-25-majors` (PR #78). 81 candidate findings survived adversarial refute-by-default verification; after deduplication (7 merge groups collapsing 16 raw indices into 7 canonical items) the confirmed set is **72 findings: 0 blocker, 6 major, 28 minor, 38 nit**. Corrected severities follow the verifier's recalibrated `severity` field; where the reasoning explicitly downgraded a candidate (e.g. #26 despike NaN, #53 fp07 speed guard, #89 mixed-key sort), the downgrade is applied.

## Executive Summary

**Overall risk read: moderate, no merge-blocker that corrupts already-validated production output, but PR #78's own M2 (`_nc_filled`) fix is incomplete and ships two genuine regressions for external/CF-NetCDF inputs.** The ARCTERX/Saipan production path (raw `.p` → profiles → epsilon → chi → bin) is not affected by any *new* data-corruption defect introduced by PR #78; the M1/M3/M4 fixes are correct as far as they go. The dominant theme across the verified set is **silent failure on degenerate/hostile/edge inputs** — single NaN samples, corrupt/truncated `.p` headers, duplicate pressures, tiny config floats, non-UTC timestamps — rather than systematic error on clean data. None rises to "blocker" (no defect silently corrupts the headline epsilon/chi numbers on the validated benchmark or the clean ARCTERX casts).

**Headline issues (the 6 majors):**
1. **`extract_profiles` crashes on the exact external CF/ATOMIX files PR #78 claims to support** (#14) — a confirmed `AttributeError` when any channel carries `_FillValue`. Reproduced.
2. **Method-1 chi filters the wrong epsilon QC statistic** (`fom` obs/Nasmyth ratio vs the Lueck `FM` it never reads, against a limit defined for `FM`) — silently drops good shear probes from the Method-1 epsilon mean (merge of #42/#101).
3. **Method-1 epsilon time reference corrupted by tz-offset stripping** (#41) — silently NaNs *all* chi for any non-UTC instrument timestamp.
4. **`compute_trim_depth` returns the shallowest sub-threshold bin, not the prop-wash exit** (#66) — a momentarily-quiet top bin causes severe under-trim, leaving noisy near-surface data in the profile.
5. **Edit-spec key injection** (#20) — `_check_keymap` validates only the value, not the key, defeating the "unaddressable acquisition params" guarantee of the config-patch tool (latent; from PR #77, untouched by #78).
6. **`merge` fuses 29 independent ARCTERX casts into one chain** (#95) — no time-continuity/restart gate; chains purely on `file_number` adjacency. Latent because `files.merge` defaults to `false`, but a hard data-fusion hazard if enabled.

**Verdict on PR #78's own fixes:** **Conditionally sound, but the M2 fix must not merge as-is.** The four fixes target M1 (`_variance_correction`), M2 (`_nc_filled`), M3 (`ct_align` NaN guard), M4 (`bottom` crash depth). M1, M3, M4 are correct fixes; M2 is incomplete and introduces regressions. Critically, **every PR #78 test verifies the fix *mechanism* but none tests the *downstream consequence*** the audit surfaces (see merge-blocker subsection). All 11 tests in `tests/test_audit_2026_06_25_fixes.py` pass, which masks the residual defects.

---

## PR #78 merge-blockers (files touched by the four fixes)

Files in scope: `chi/chi.py` (`_variance_correction`), `rsi/profile.py` (`_nc_filled`), `processing/ct_align.py`, `processing/bottom.py`. Each confirmed finding below is classified as **regression** (newly introduced/exposed by #78), **residual** (a real defect #78 did not fix but should have given its stated scope), or **refuted**.

### M2 — `_nc_filled` (`rsi/profile.py`) — INCOMPLETE; do not merge as-is

- **#14 (major) — REGRESSION/RESIDUAL, reproduced.** `rsi/profile.py:240` creates each per-profile channel via `createVariable(..., "f4", (dim,), zlib=True)` with **no `fill_value` kwarg**, then `:243-246` does a blind `setattr(var, k, v)` over every attr copied from the source at `:440-441` (`for a in var.ncattrs(): attrs[a] = getattr(var, a)`), **including `_FillValue`**. netCDF4 forbids setting `_FillValue` after creation. I reproduced end-to-end: an external CF NetCDF whose `T1` declares `fill_value=9.96921e36` makes the write path raise `AttributeError: _FillValue attribute must be set when variable is created`. The M2 docstring explicitly claims support for "external / partial (CF/ATOMIX) files that declare a `_FillValue`" — so the fix's own target input crashes. **Fix:** skip reserved attrs (`for k,v in attrs.items(): if k in {"_FillValue","_Netcdf4Coordinates","_Netcdf4Dimid","_Unsigned"}: continue`), or thread the source `_FillValue` into `createVariable(..., fill_value=...)`. Add a test that `extract_profiles` handles a `.nc` whose channels declare `_FillValue`.

- **#13 (minor, borderline-major) — REGRESSION, reproduced.** The M2 fix correctly stops the raw ~9.97e36 fill buffer from leaking, but converting fill→NaN poisons the *whole cast*: `scor160/profile.py:58-61` `smooth_fall_rate` does `np.gradient(P)` then `filtfilt`, both of which propagate a single NaN across the entire array. I reproduced: a 2000-sample descending pressure with **one** mid-cast NaN yields **0/2000 finite fall-rate** (vs 2000/2000 clean), so `get_profiles` returns `[]` and the cast is **silently dropped**. The PR #78 test `TestM2FillValueLeak::test_pressure_fill_becomes_nan` only asserts `P[5]` becomes NaN — it never runs profile detection, so it passes while the cast-loss behavior is live. A truncated cast becomes a *lost* cast. **Fix:** linearly interpolate isolated NaNs in slow pressure inside `_load_from_nc` before fall-rate, and warn/count when NaNs are repaired so a poisoned cast is visible. Add an `extract_profiles` regression with a mid-cast fill asserting the profile is still found. (Kept at minor per the verifier's corrected severity; flagged here because silent whole-cast loss is the most consequential M2 residual.)

- **#106 (nit) — RESIDUAL.** `_load_from_nc` opens `ds = nc.Dataset(...)` at `:349` and only closes at `:458` (success). The `raise ValueError` at `:385/:416/:418` and `AttributeError` from `_get_nc_attr` at `:370` leak the handle. With M2 now actively NaN-ing pressure (which can trigger the "no pressure"/empty paths more readily), this is adjacent to the fix. **Fix:** wrap in `try/finally: ds.close()` or use `with`.

### M1 — `_variance_correction` (`chi/chi.py`) — correct fix; cost/dead-code residuals

- **#16 / #64 / #100 / #107 (nit, merged → `chi_grid`) — RESIDUAL, confirmed.** `chi.py:113` `n = max(int(n_fine), _K_SPAN_KB * _PTS_PER_KB)` = `max(2000, 80000)` = **80000 unconditionally**; the `n_fine`/`_N_FINE=2000` floor is dead code, and the grid grew 2000→80000 (~40x) vs the pre-#78 code. This is a correctness-neutral performance cost in the per-window, per-iteration chi loop (two 80000-element float64 arrays per call). The M1 accuracy test passes; accuracy is not in question. **Fix:** lower `_PTS_PER_KB` to ~200–500 after confirming <0.1% error against the high-n reference, drop the dead `n_fine` floor (`n = _K_SPAN_KB * _PTS_PER_KB`), and remove/clarify the now-misleading `n_fine` knob.
- **#93 (nit) — RESIDUAL.** `chi.py:108` guard `if not (kB > 0)` admits `kB=inf` (`inf > 0` is True); with `kB=inf`, `K_fine` is all-inf and `np.trapezoid` does an `inf-inf` subtract, leaking a NumPy `RuntimeWarning`. **Fix:** tighten to `if not (0 < kB < np.inf): return np.nan`, or wrap the trapezoid in `np.errstate(invalid="ignore")`.
- **#98 (minor) — RESIDUAL (separate root from the grid).** `_mle_fit_kB` (`chi.py:440-447`) calls `_variance_correction` with **no `K_min`** (defaults to 0.0), so V_resolved integrates over `[~0, K_max]`, while `obs_var` (`:449-451`) integrates over `[K_fit_low, K_max]`. The mismatched lower band edge biases chi low for low-epsilon windows. `_iterative_fit` does this correctly. **Fix:** pass `K_min=float(K[fit_mask][0])` so both spans match.
- **#33 (nit) — RESIDUAL.** `_iterative_fit` (`chi.py:560-562`) breaks on convergence *before* the chi_obs recompute block (`:565-596`), returning a chi computed against a one-iteration-stale `kB`. **Fix:** recompute chi_obs once from the final `kB_best` after the loop.

### M3 — `ct_align` NaN guard (`processing/ct_align.py`) — correct fix; over-discard residual

- **#23 / #105 (minor, merged → `ct_align_discard`) — RESIDUAL, confirmed.** The M3 guard `ct_align.py:83-84` `if not (np.all(np.isfinite(seg_T)) and np.all(np.isfinite(seg_C))): continue` correctly stops a NaN profile from driving the −max_lag estimate, but it discards the **entire** profile segment on a *single* isolated non-finite sample (e.g. `convert_jac_c` emits NaN for one corrupt conductivity sample). If every profile carries one isolated NaN, all segments are skipped and alignment silently falls back to no-op. The M3 test only appends a single NaN to *one* of two profiles (the other clean), so the consensus survives and the over-discard is untested. **Fix:** mask/interpolate only the non-finite samples before `diff`/`lfilter`; `continue` only when the finite fraction is too low. Add a regression where *every* profile has an isolated NaN and assert the lag is still recovered.

### M4 — `bottom` crash depth (`processing/bottom.py`) — correct fix; under-trim + warning residuals

- **#63 (minor) — RESIDUAL, confirmed.** The M4 fix `bottom.py:135-136` returns `np.nanmean(depth[in_range & (idx==i)])` (mean of flagged-bin samples) to avoid the prior bin-center-overhang crash. But because the profiler keeps descending through the crash, the flagged bin holds samples both before and after onset, so the mean lands near bin center — under-trimming by up to half a bin width (~2 m at the 4 m default) when onset is early in the bin. The M4 test only asserts the depth does not *exceed* the deepest sample; the under-trim direction is untested. **Fix:** report the bin's shallowest flagged-sample depth, or sub-bin to locate the accel rise edge; at minimum document the ~half-bin under-trim.
- **#24 / #67 (nit, merged → `bottom_warn`) — RESIDUAL, confirmed.** `bottom.py:89-90` wraps `np.nanmax(depth)` in `np.errstate(invalid="ignore")`, but errstate governs only IEEE FP flags, not numpy's `warnings.warn`-issued "All-NaN slice encountered" RuntimeWarning, so a legitimate all-NaN-depth bail-out (the path the comment says it handles silently) leaks warning noise. **Fix:** short-circuit `if not np.any(np.isfinite(depth)): return None` before `nanmax`, or use `warnings.catch_warnings()`.

**Net merge-blocker verdict:** Block on **#14** (confirmed crash on the fix's own supported input) and require a guard/interp for **#13** (silent whole-cast loss) before merge. The M1/M3/M4 fixes are sound; their residuals (`chi_grid`, `ct_align_discard`, `#63`) are non-blocking but should be tracked. No finding in scope is refuted.

---

## Findings by Severity

### MAJOR (6)

- **#14 — `rsi/profile.py:240,243-246,440-441` — `extract_profiles` `AttributeError` on CF/ATOMIX `.nc` with channel `_FillValue`.** What's wrong: `createVariable` without `fill_value` followed by blind `setattr(var,"_FillValue",...)`, which netCDF4 forbids. Confirmed: reproduced end-to-end (`AttributeError: _FillValue attribute must be set when variable is created`); the source attrs dict is populated by an unfiltered `var.ncattrs()` copy. Fix: skip `{"_FillValue","_Netcdf4Coordinates","_Netcdf4Dimid","_Unsigned"}` in the copy loop, or pass the source `_FillValue` into `createVariable(..., fill_value=...)`; add a test covering a channel with `_FillValue`.

- **#20 — `rsi/config_patch.py:146-152,386` — edit-spec keys never validated → config-stanza injection.** What's wrong: `_check_value` rigorously rejects values containing non-ASCII/`;`/control chars to stop `[root]`/`[matrix]` stanza injection, but `_check_keymap` runs that validation only on the value, never the key, so a hostile key can carry a newline + injected stanza into the written config. Confirmed: from PR #77, untouched by #78; the asymmetry is exact. Fix: in `_check_keymap`, reject keys containing newline/control char/`=`/`;`/`[`/`]`/non-ASCII before storing, on both the YAML and public `EditSpec` paths.

- **#41 — `rsi/chi_io.py:382-390` — Method-1 epsilon time reference corrupted by tz-offset stripping.** What's wrong: `start = np.datetime64(re.sub(r"(Z|[+-]\d{2}:?\d{2})$","",start_str))` strips the offset; the inline comment claiming numpy can't accept a tz suffix is false — `np.datetime64('...+10:00')` correctly UTC-normalizes (only a UserWarning). Stripping yields naive local time (a 36000 s error for +10:00), which silently NaNs all chi for non-UTC instruments. Confirmed: reproduced the 36000 s divergence. Fix: parse offset-aware (`np.datetime64(pd.Timestamp(start_str))` or `datetime.fromisoformat(...).astimezone(UTC)`); add a regression with a non-UTC CF `units` that round-trips through NetCDF.

- **#42 + #101 — `rsi/chi_io.py:353-366` (merged `m1_fom`) — Method-1 epsilon mean filters the wrong FOM statistic.** What's wrong: `_epsilon_ds_to_l4data` reads `_probe_var("fom")` (the obs/Nasmyth variance ratio, good ≈ 1.0) and rejects probes where `fom > _EPS_FOM_LIMIT=1.15` — but 1.15 is the limit for the **Lueck `FM`** statistic (good ≈ 0), which the rsi epsilon NetCDF *also* writes and which this code never reads. Filtering the variance-ratio `fom` at the FM limit silently drops good shear probes from the cross-probe epsilon mean feeding Method-1 chi. Confirmed: both `fom` and `FM` are written by `dissipation.py:455-481` with distinct long_names and an explicit comment that `fom` is "NOT the MAD-based ATOMIX/Rockland FM statistic." Fix: read `FM` (fallback to `fom` only if absent, mirroring `l4.py`), gate on `FM > limit`, rename `_EPS_FOM_LIMIT`→`_EPS_FM_LIMIT`, and fix the two tests that encode the same mislabel.

- **#66 — `processing/top_trim.py:97-100` — `compute_trim_depth` returns the shallowest sub-threshold bin, not the prop-wash exit.** What's wrong: the per-channel loop returns the FIRST (shallowest) bin below threshold and breaks; `:106` then `np.max` across channels, but each channel already committed to its shallowest quiet bin, so a momentarily-quiet top bin causes severe under-trim that leaves the noisy near-surface section in the profile. Confirmed: reproduced empirically on a quiet-top/noisy-middle profile (untouched by #78). Fix: scan for the *deepest* bin still exceeding threshold and return the next bin below it (the prop-wash exit); add a quiet-top/noisy-middle/quiet-bottom regression. **Resolved 2026-06-27:** `compute_trim_depth` now returns the bin below the deepest still-elevated bin (threshold = `noise_factor`×background, background = the `quantile` of per-bin std) so a momentary near-surface lull no longer ends the search early; channels are combined by median (robust to a dead/misbehaving channel). **Corrected 2026-06-28 (real-data tuning):** on ARCTERX VMP casts the original channel set (sh1/sh2 + Ax/Ay, combined by `max`) over-trimmed to 30-50 m because the shear probes track deep ocean turbulence; top_trim now uses **only the accelerometers** (Ax/Ay), which mark the mechanical entry transient, with the median combine — trims settle at ~5 m (p90 9 m). The `quantile` default reverted 0.25→0.6 (0.25 over-trims patchy real data; the deep-prop-wash synthetic case it was tuned for does not occur with accelerometer signals).

- **#95 — `perturb/merge.py:78-132` — `merge` fuses independent casts: no time-continuity/restart check.** What's wrong: `find_mergeable_files` chains purely on strict `file_number` adjacency with no start-time/gap/restarted gate, so 29 sequential-but-independent ARCTERX casts fuse into one chain. Confirmed: code trace shows `_read_merge_info` reads no start time and the chain link has no gap test; untouched by #78. Latent because `files.merge` defaults false. Fix: compute `prev_end = prev.start_time + prev_n_scans/fs_fast` in `_read_merge_info` and chain only when `abs(next.start_time - prev_end) <= tol`; add an ARCTERX-gap regression.

### MINOR (28)

- **#0 + #80 — `rsi/config_patch.py:504-528` (merged `write_patched_hdr`) — `write_patched_pfile` lacks `header_size >= HEADER_BYTES` guard.** Unchecked `header_size` (word 17) is used for `f.read(header_size)` and `struct.pack_into` at offset `config_size*2 = 22`; a `header_size` in [12,23] yields a buffer too short → cryptic `struct.error`, and other small values silently corrupt the header. Confirmed: reproduced both failure modes. Fix: `if header_size < HEADER_BYTES: raise ValueError(...)` before any seek/pack, in both `write_patched_pfile` and `read_config_text`, matching `extract_pfile_segment`.

- **#2 — `scor160/l4.py:176` — production epsilon viscosity ignores salinity/pressure (`visc35`, S=35 hardcoded).** Inconsistent with chi/N² viscosity in the same pipeline run, which is salinity-aware. Confirmed: `process_l4` imports only `visc35` and has no salinity/pressure parameter. Fix: plumb `salinity` (and `pres`) into `process_l4`, compute `nu = visc(temp,sal,pres)` when provided (as `dissipation.py:213-220` does), keep `visc35` default for ATOMIX parity; at minimum document the divergence.

- **#8 — `rsi/binning.py:122`, `combine.py:82`, `pipeline.py:599,279` — L5/L6 products have no units/long_name/standard_name on any data variable.** `bin_by_depth` attaches only `cell_methods`; `combine_profiles` drops even that. Confirmed at runtime. Fix: merge a units/long_name/standard_name dict (reuse `COMBO_SCHEMA`/`CHI_SCHEMA`) into each binned var, carry source attrs through `combine_profiles`'s `(dims, data, attrs)` tuple; assert epsilon/chi carry units after combine.

- **#13 — `rsi/profile.py:330-342` / `scor160/profile.py:54-61` — M2 fill→NaN poisons whole fall-rate (one mid-cast fill ⇒ 0 profiles).** See merge-blocker subsection (M2). Confirmed (reproduced 0/2000 finite W). Fix: interpolate isolated NaNs in slow pressure before fall-rate and warn.

- **#17 — `rsi/convert.py:592` — `convert_all` flattens input paths to basename, colliding same-named `.p` from different dirs.** Confirmed: `VMP/a/X.p` and `VMP/b/X.p` both map to `out/X.nc`; parallel jobs race-write, serial silently overwrites. Fix: detect output-name collisions before dispatch and raise/disambiguate.

- **#21 — `rsi/config_patch.py:350-357,351,459 — provenance/`when` not control-char sanitized (banner is).** Defense-in-depth gap: `_one_line` is applied to author/note/source_label in the banner but never to `when` or to the provenance interpolations. Confirmed. Fix: route all comment-line fields through one sanitizer used by both banner and provenance.

- **#25 — `scor160/despike.py:98-110` — `despike` raises scipy `ValueError` on length-1/2 sections, reachable via temperature despike.** For length-2, padded array is exactly 6 samples and `filtfilt` padlen=6 → crash. Confirmed: reproduced for L∈{1,2}. Fix: pass `padlen=_matlab_padlen(...)` (=3) which both restores MATLAB parity and avoids the crash, or guard short sections.

- **#32 — `rsi/pipeline.py:466` (`chi/l4_chi.py:286-294`) — rsi mixing quantities use unfiltered `chi_final` geo-mean (no fom/K_max_ratio QC).** Unlike the epsilon path that co-drives Γ/K_ρ and perturb's `mk_chi_mean`. Confirmed: written K_T/Γ outputs use the unfiltered mean. Fix: drop per-probe chi exceeding the chi fom limit or K_max_ratio < ~0.5 before the geometric mean, falling back to all-probe only if none survive.

- **#35 — `perturb/plot/xaxis.py:326-327` — longitude x-axis passthrough not dateline-safe.** A track crossing ±180 is split/reordered while the distance/along_line methods were hardened with `_circmean_deg`/`_wrap_deg`. Confirmed (latitude part of the claim is spurious; longitude part is real). Low urgency for ~145E ARCTERX data. Fix: unwrap longitudes about their circular mean in the longitude branch; add a dateline regression.

- **#43 — `rsi/chi_io.py:857-872` — `load_epsilon_dataset` concatenates stale `_eps.nc` with per-profile `_prof00N_eps.nc`, double-counting times.** Confirmed: unconditional `files.insert(0, single)` then `xr.concat(dim="time")` with no overlap guard. Fix: treat single-profile and per-profile naming as mutually exclusive (prefer per-profile when any exist), or drop duplicate times and warn.

- **#45 — `perturb/binning.py:442,576` — `np.arange` bin edges produce a spurious trailing bin for non-binary-exact widths.** Confirmed: depths 0–7 m at bw=0.7 yield 11 bins not 10, reassigning the deepest boundary sample. Fix: compute `n_bins = int(round((d_max-d_min)/bw))` and build edges with `d_min + np.arange(n_bins+1)*bw`; apply to both depth and time grids.

- **#55 — `rsi/p_file.py:302-305` — `PFile._read()` lacks short-header length guard, raising raw `struct.error` on truncated files.** Confirmed: reproduced on a 32-byte file. Fix: `if len(raw_hdr) < HEADER_BYTES: raise ValueError(...)` after the read.

- **#56 — `rsi/convert.py:605,616` — `convert_all` batch loops don't catch `struct.error`, aborting the whole batch.** `struct.error` is not a subclass of OSError/ValueError/RuntimeError. Confirmed via MRO. Fix: guard the short header in `PFile._read()` (#55), and broaden the except to include `struct.error`.

- **#58 — `rsi/p_file.py:322-323` — zero `n_cols`/`n_rows` in a corrupt header → bare `ZeroDivisionError`.** Confirmed: reproduced both branches. Fix: validate `n_cols >= 1`, `n_rows >= 1`, `f_clock > 0` with a descriptive ValueError.

- **#63 — `processing/bottom.py:135-136` — bottom-crash sample-mean can under-trim by up to half a bin.** See merge-blocker subsection (M4). Confirmed. Fix: report shallowest flagged-sample depth or sub-bin; document the half-bin under-trim.

- **#74 — `scor160/spectral.py:201` — `csd_matrix` accepts a length-1/scalar custom window, silently broadcasting to rectangular.** MATLAB errors on non-`nfft`-length windows. Confirmed. Fix: validate `window.ndim==1 and window.shape[0]==nfft` after `np.asarray`.

- **#75 — `scor160/spectral.py:371-375` — `csd_matrix_batch` omits the `2*nfft` minimum-length requirement** that `csd_matrix` and the MATLAB reference enforce, allowing statistically-weak single-segment spectra. Confirmed. Fix: add `if diss_length < 2*nfft: raise ValueError(...)`.

- **#82 — `rsi/config_patch.py:530-542` — `write_patched_pfile` is not atomic.** Opens the final destination directly (`open(dst,"xb")`) and streams with no try/except; a mid-copy failure leaves a partial valid-looking patched `.p`. Confirmed. Fix: write to a sibling temp file then `os.replace` after fsync; unlink temp on failure.

- **#83 — `rsi/config_patch.py:631-657` — `patch_files` aborts the whole batch (and leaves the first output) when two sources share a basename.** Confirmed: `dst = out_dir/src.name`, no up-front collision check, second raises `FileExistsError` mid-loop. Fix: detect basename collisions up front, or disambiguate outputs, or make per-file errors recoverable with a summary.

- **#84 — `config_base.py:38` — `round(v,10)` collapses tiny config floats (`epsilon_minimum`/`chi_minimum=1e-13`) to 0.0, corrupting the version hash and signature provenance.** Confirmed: `round(1e-13,10)==0.0`; these keys are not hash-excluded. Fix: normalize by significant figures — `return float(f"{v:.10e}")`; add a hash-distinctness regression for 1e-13 vs 5e-13.

- **#85 — `config_base.py:241` — sequential output-dir glob `{prefix}_[0-9][0-9]` only matches 2-digit suffixes; dir 100 is invisible once >100 configs accumulate.** Dirs are created with `{next_seq:02d}` (3 digits at ≥100). Confirmed end-to-end: reuse-failure + collision. Fix: width-adaptive anchored glob `{prefix}_[0-9][0-9]*`.

- **#89 — `config_base.py:51,173` — `compute_hash`/`canonicalize` crash on mixed-type dict keys** (numeric + string instrument serials, `hotel.channels`, `qc.rules`). Confirmed (downgraded major→minor): `sorted(v.items())` on `{465: ..., 'SN479': ...}` raises `TypeError: '<' not supported`. Fix: sort by `str(k)` at `:51`, `:173`, and `pipeline.py:138`; optionally coerce numeric serial keys to str at load.

- **#90 — `config_base.py:240-258` — concurrent runs of two configs against one output_root can collide into one directory** (non-atomic glob→scan→mkdir; no lock/O_EXCL). Confirmed: no fcntl/flock/filelock anywhere. Fix: `mkdir(exist_ok=False)` in a retry loop incrementing `next_seq`, or take an advisory lock around scan+create.

- **#92 — `chi/fp07.py:82,175-178` — `fp07_model` config value never validated; a typo silently selects double-pole transfer + single-pole (lueck) tau.** Confirmed: all three else-fallthrough sites exist. Fix: validate `fp07_model ∈ {single_pole, double_pole}` at the entry point, mirroring `fp07_tau`'s `ValueError`.

- **#96 — `perturb/merge.py:224-235` — no record-boundary validation before splicing.** `merge_p_files` copies chain[0] in full then copies continuation remainder to EOF with no `(size - header - config) % record_size == 0` check, despite `record_size` being available. Confirmed. Fix: validate each member is record-aligned, or copy only the integer-record prefix.

- **#98 — `chi/chi.py:440-452` — `_mle_fit_kB` correction span `[0,K_max]` ≠ obs-var span `[K_fit_low,K_max]`, biasing chi low.** See merge-blocker subsection (M1). Confirmed. Fix: pass `K_min=float(K[fit_mask][0])`.

- **#23 + #105 — `processing/ct_align.py:83-84` (merged `ct_align_discard`) — whole-segment discard on a single non-finite C/T sample.** See merge-blocker subsection (M3). Confirmed (reproduced). Fix: mask/interpolate only bad samples; `continue` only when finite fraction is too low; add an all-profiles-have-one-NaN regression.

- **#27 + #50 — `scor160/despike.py:157-166` (merged `despike_zero`) — both-empty replacement falls back to 0.0 instead of MATLAB's NaN.** When a bad region spans the whole padded array with no valid neighbors, the if/elif falls through to `replacement = 0.0`, silently injecting a plausible physical shear/temperature value and hiding data loss (ODAS gives NaN via 0/0). Confirmed: reachability verified for short arrays. Fix: set `replacement = np.nan`; add a short-array regression.

### NIT (38)

- **#1 — `rsi/config_patch.py:324-331` — `edit_config_text` mis-detects a pre-existing `CONFIG_MARKER` inside the original config**, freezing everything below it and raising a misleading "section is not present" error on RSI-pre-patched files. Confirmed (reproduced). Fix: only treat post-marker text as frozen when it looks like a fully `; `-commented block this tool wrote; else treat the whole string as active.

- **#5 — `rsi/dissipation.py:218-220` — deprecated `_compute_epsilon` lacks the NaN-temperature viscosity guard `process_l4` has**, silently producing NaN epsilon. Confirmed: `visc35(nan)=nan`. Fix: `temp = np.where(np.isfinite(l3.temp), l3.temp, 10.0)` + warn, mirroring `l4.py:154-160`.

- **#9 — `rsi/combine.py:89-107`, `pipeline.py:595-599,277-279` — L6/L5 lack the `Conventions`/CF global attr** every other writer sets. Confirmed. Fix: set `attrs['Conventions']='CF-1.13, ACDD-1.3'` (+`featureType='profile'`) before `to_netcdf`.

- **#10 — `rsi/convert.py:541` — L1 supplementary channels write raw RSI units, bypassing `canonicalize_units`** (e.g. `deg`, `mS_cm-1`, `umol_L-1`). Confirmed. Fix: `v.units = canonicalize_units(info["units"])`.

- **#11 — `rsi/convert.py:360 — L1 MAG `units='micro_Tesla'` is not UDUNITS-parseable.** Confirmed via cf_units. Fix: use `uT` (or `microtesla`).

- **#12 — `perturb/ctd.py:419` — CTD per-file write emits `_FillValue` on the `time` coordinate (CF-1.13 §2.5.1 violation).** Confirmed: xarray auto-emits it with no `encoding=`. Fix: `encoding={'time': {'_FillValue': None}}`, mirroring `make_combo`.

- **#22 — `rsi/cli.py:280-293` / `config_patch.py:168-169` — `patch-config` surfaces an uncaught ruamel `YAMLError` traceback on malformed edit-spec YAML.** The except tuple omits `YAMLError`. Confirmed. Fix: catch `YAMLError` in the CLI except, or re-raise as `ValueError` inside `load_edit_spec`.

- **#26 — `scor160/despike.py:110-119` — a single NaN sample silently disables ALL spike detection for the whole array** (filtfilt propagates NaN → `ratio > thresh` all-False; real spikes pass through). Confirmed (reproduced; downgraded major→nit — purely QC-degrading, bounded reachability on real shear). Fix: NaN-guard at the top of `despike()` that warns and skips only the contaminated span.

- **#30 — `processing/mixing.py:475,486` — `K_rho` N2_min floor (1e-9 s⁻²) prevents div-by-zero but not unphysical inflation**; near-floor windows emit absurd Osborn diffusivities (e.g. K_rho=10 m²/s). Confirmed; physically realizable. Fix: raise `DEFAULT_N2_MIN` (~1e-7) or add an upper sanity clamp/mask; surface the near-floor count. **Resolved 2026-06-27:** added a configurable upper bound `DEFAULT_K_RHO_MAX` (1.0 m²/s); over-bound windows are masked to NaN and the masked count is surfaced via `warnings.warn`. Kept `DEFAULT_N2_MIN=1e-9` (raising the floor was rejected as it NaNs real weakly-stratified bins). NetCDF `K_rho` comment attributes and `docs/mixing_efficiency.md` updated.

- **#33 — `chi/chi.py:560-606` — `_iterative_fit` returns chi against a one-iteration-stale kB on convergence break.** See M1 subsection. Confirmed. Fix: recompute chi_obs from final kB after the loop.

- **#34 — `chi/chi.py:644` — `_bilinear_correction` raises `ValueError` for small config-derived `diff_gain` (cutoff ≥ 1).** Confirmed: `diff_gain ≤ ~6.2e-4` → `butter` raises. Fix: clamp/validate the normalized cutoff; degrade to an all-ones correction when out of range.

- **#36 — `perturb/plot/eps_chi.py:323` — realignment over-reports "eps slots have no chi" by counting time-matched but all-NaN chi columns as unmatched.** Confirmed. Fix: track matches with an explicit boolean array set inside the time-match branch.

- **#37 — `scor160/l4.py:175-180` — NaN-speed window yields wrong-but-finite epsilon via `K_AA = 0.9*f_AA/max(NaN,0.05)=NaN`.** L3 floors W only for the wavenumber axis but stores the un-floored mean. Confirmed (primary-impact claim was overstated). Fix: `K_AA = 0.9*f_AA/max(np.nan_to_num(W,nan=0.05),0.05)`.

- **#38 — `scor160/compare.py:50-56` — `compare_l2` raises `ValueError` on empty section arrays** (`.max()` on zero-size, before the `both_in.any()` guard). Confirmed; sibling `compare_l3`/`compare_l4` guard correctly. Fix: `int(ref_sec.max()) if ref_sec.size else 0`.

- **#39 — `scor160/io.py:198` — `L4Data.fom` docstring mislabels the benchmark FOM as the obs/Nasmyth variance ratio.** It actually holds the Lueck-2022 MAD-based `FM`. Confirmed (in-repo comments at `compare.py:350-352` agree). Fix: correct the comment to name the MAD-normalized statistic.

- **#47 — `scor160/spectral.py:197-200` — `overlap<0` clamps to `nfft//2` (50%) where MATLAB clamps to 0 (no overlap).** Confirmed port divergence; changes n_seg/ensemble averaging. Fix: split the clamp to mirror MATLAB (`<0 → 0`, `>0.9*nfft → nfft//2`) in both `csd_matrix` and `csd_matrix_batch`.

- **#48 — `scor160/spectral.py:180-181` — auto-transpose `shape[0] < shape[1]` flips legitimately wide matrices, unlike MATLAB's `isrow`-only force.** Confirmed (also `goodman.py:138-141`). Fix: transpose only the true row-vector case (`x.ndim==2 and x.shape[0]==1`).

- **#49 — `scor160/despike.py:110-114` — `filtfilt` uses scipy default padlen (6), not MATLAB's 3** — a port divergence the codebase already fixed in `l2.py` via `_matlab_padlen`. Confirmed. Fix: pass `padlen=_matlab_padlen(b,a)` (also resolves #25's length-2 crash).

- **#52 — `scor160/despike.py:93 — spike window uses `N//2` (floor) where MATLAB uses `round(N/2)`**, a ~1-sample difference for odd N. Confirmed (correctly nit; negligible physical effect, even default). Fix: compute the half-width as float and round the boundary indices, or document.

- **#53 — `chi/fp07.py:120-130,455-498,591-616` — `gradT_noise`/`gradT_noise_batch`/`fp07_tau` drop the MATLAB speed>0 validation**, yielding negative noise/inf/NaN for invalid speed. Confirmed (downgraded minor→nit — purely latent, no live path triggers it). Fix: `speed = np.maximum(np.abs(speed), 1e-3)` or raise for speed≤0.

- **#54 — `chi/fp07.py:574` — broken-thermistor `R_ratio` clamp warns in scalar `noise_thermchannel` but is silent in `noise_thermchannel_batch`** (the production path). Confirmed. Fix: add `if np.any(R_ratio_pre < 0.1): warnings.warn(...)` reporting the clamped count.

- **#57 — `rsi/p_file.py:361-379` — corrupt `record_size`/`n_cols` produces a cryptic "cannot reshape array" crash** instead of a clear diagnostic. Confirmed. Fix: validate header geometry (`record_size > header_size >= HEADER_BYTES`, data words a positive multiple of `n_cols`) before reshape.

- **#59 — `rsi/convert.py:514-515` — GRADT computation crashes on a single-sample fast channel** (`np.diff` size-0 → `dTdt[-1]` IndexError). Confirmed (reachability effectively nil on real data). Fix: `if T_fast_ch.size < 2: dTdt = np.zeros_like(...)`.

- **#61 — `rsi/p_file.py:471-473,503-523` — joined 32-bit (2-id) channels never get ODAS's signed-by-default correction.** Confirmed (latent; real 2-id channels are jac_c which is unsigned). Fix: apply `ch[ch >= 2**31] -= 2**32` for non-skip-set joined channels, or document the unsigned-only handling.

- **#69 — `pyproject.toml:65-67` — three mypy per-module overrides reference nonexistent modules** (`odas_tpw.rsi.spectral/ocean/nasmyth`; the real modules live under `scor160/`). Confirmed: suppress nothing. Fix: delete the three stale lines; add a CI check that override modules exist.

- **#72 — `rsi/deconvolve.py:66-68` — the sign-inversion branch (`p[0] < -0.5`) has zero test coverage.** Confirmed (downgraded minor→nit): faithful port of `deconvolve.m:189`, but all 8 tests use positively-correlated input. Fix: add a test with anti-correlated `X_dX` (and an int16 −32768 sample) asserting correct un-inverted recovery.

- **#86 — `config_base.py:173` — `hash_exclude_keys` ('diagnostics') is wrongly applied to top-level instrument keys**, so an instrument literally named 'diagnostics' is dropped from the canonical hash. Confirmed (correctly nit). Fix: drop the `if k not in hash_exclude_keys` filter from the dynamic-key branch (scope it to static sections).

- **#87 — `config_base.py:152` — `merge_config` silently discards ALL overrides for dynamic-key sections** (empty `{}` defaults) if ever called on 'instruments'. Confirmed (currently not invoked on dynamic sections). Fix: short-circuit dynamic sections in `merge_config`, or raise a clear "undefined for dynamic-key sections" error.

- **#93 — `chi/chi.py:108-117` — `_variance_correction` leaks a NumPy `RuntimeWarning` on degenerate `kB=inf`.** See M1 subsection. Confirmed. Fix: `if not (0 < kB < np.inf): return np.nan`.

- **#94 — `chi/fp07.py:175-178` — `fp07_transfer_batch` silently treats any unrecognized model string as double_pole**, unlike `fp07_tau` which raises. Confirmed. Fix: make the branch explicit with an `else: raise ValueError(...)`.

- **#97 — `perturb/config.py:323` — `files.merge` documented with no warning about the cast-fusion hazard** (the adjacent `force_trim` carries an explanatory comment). Confirmed. Fix: annotate the line; pair with the #95 continuity fix.

- **#99 — `processing/mixing.py:511-514` — `pair_nearest` auto-tolerance collapses to 0 when >50% of source-time spacings are zero** (duplicate src_times), dropping all but exact-match pairings. Confirmed (reproduced). Fix: `max_dt = med if (len>1 and med>0) else <positive floor>`.

- **#103 — `scor160/ocean.py:118-149` — `buoyancy_freq` default `lat=0` biases N² via TEOS-10 g(lat)** (+0.07% at Saipan ~15N, +0.94% at 70N). Confirmed; production path threads a real lat. Fix: drop the `lat=0` default (make it required) or warn when defaulted.

- **#104 — `scor160/ocean.py:146-149` — `buoyancy_freq` produces inf/nan N² on duplicate/non-monotonic input pressure** (unguarded dp=0 division). Confirmed (reproduced; latent — production passes monotonic P). Fix: validate strict-monotonic P before `gsw.Nsquared`, or NaN the offending entries; document the precondition.

- **#106 — `rsi/profile.py:349-458` — `_load_from_nc` leaks the open NetCDF Dataset handle on every error path.** See M2 subsection. Confirmed. Fix: `try/finally: ds.close()` or `with`.

- **#16 + #64 + #100 + #107 — `chi/chi.py:112-114` (merged `chi_grid`) — `_variance_correction` fixed at 80,000 points (dead `n_fine` floor), ~40x the prior grid.** See M1 subsection. Confirmed. Fix: lower `_PTS_PER_KB` after a <0.1% accuracy check and drop the dead floor.

- **#18 + #71 — `rsi/p_file.py:249,390` (merged `parse_config`) — bare `ValueError` on a corrupt matrix row or channel id token, with no file context.** Confirmed (reproduced both crashes; config decoded with `errors="replace"` can yield non-numeric tokens). Fix: wrap the `int()` conversions and re-raise naming the file, section, and offending token.

- **#24 + #67 — `processing/bottom.py:89-90` (merged `bottom_warn`) — "All-NaN slice encountered" RuntimeWarning escapes the surrounding `np.errstate`** on the intended-silent all-NaN-depth bail-out. See M4 subsection. Confirmed. Fix: short-circuit on `not np.any(np.isfinite(depth))` before `nanmax`, or use `warnings.catch_warnings()`.

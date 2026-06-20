# Deep audit round 2 (post-PR#74, 1-round loop) — odas_tpw

**Executive summary.** This re-audit of branch `fix/audit-pr73` (PR #74) confirmed **0 Blocker, 17 Major, 24 Minor, 15 Nit** findings still present in the current code after the prior deep audit's 24+33 fixes. No Blocker-class defects remain: the core happy-path .p→epsilon→chi pipeline on the ARCTERX 2-probe VMP-250 dataset is sound. The residual risk concentrates in (a) the `perturb/` campaign pipeline — config-default-not-applied bugs that silently mis-size windows and falsify provenance hashes, in-place trim/merge that can destroy source `.p` files, and stale/orphaned-data leaks into combos; (b) NaN/fill-value handling regressions where the audit's own fixes (`_nan_singular_bins`, despike change-mask, NaT/`_FillValue`→NaN policy) introduced crashes or silent QC corruption; and (c) interactive viewers/plots that crash or display biased epsilon/chi. The loop **did not converge**: it ran **1 of a 12-round cap** and is being reported after a single pass (the 2-consecutive-dry-round stop condition was not reached). Overall branch risk read: **moderate** — safe for the default RSI single-instrument workflow, but several Major defects bite real campaign re-runs, glider/non-VMP vehicles, NaN-bearing data, and the opt-in `mle` chi path.

**Overlap with known-DEFERRED areas.** Three findings below touch the deferred list and are flagged inline: the **dateline/antimeridian** note overlaps the GPS NaN-node finding (M-12 / Mn-15, *distinct* defect — NaN-node poisoning, not lon wrapping); the **`_load_from_nc` / `p_to_L1` handle leaks** are not re-reported but the adjacent `extract_profiles` over-run (Mn-8) lives in the same `_load_from_nc` path; **`run_pipeline` exit code** is not re-reported. New/more-specific defects only are listed.

---

## Blocker

*None.*

---

## Major

**M-1. Method-1 chi averages epsilon across both probes with NO QC filtering.**
`src/odas_tpw/rsi/chi_io.py:342-347` — On the documented `rsi-tpw chi --epsilon-dir` / disk-round-trip path, `_epsilon_ds_to_l4data` collapses per-probe epsilon via `np.nanmean(eps_vals, axis=0)` and passes `fom/mad/epsi_flags` as zeros, discarding the FM/fom the eps file carries. The viewer path (`window.py:347`) applies `fom_limit=1.15`; production does not, so a bad (high-FM) shear probe biases the epsilon driving every thermistor's chi. *Fix: mask FM>~1.15 estimates (geometric mean) before averaging, mirroring `compute_chi_window`, and propagate real fom/FM/mad.*

**M-2. NaN in shear/accel silently skips Goodman cleaning with a false "insufficient segments" warning.**
`src/odas_tpw/scor160/goodman.py:141-169` (root `spectral.py:52 _detrend_segment`) — `_detrend_segment` lacks the NaN guard that `_detrend_batch` has, so `scipy.detrend`/`polyfit` raise `ValueError` on any non-finite sample; `clean_shear_spec` catches it as "signal too short," warns wrongly, and returns uncleaned/zeroed spectra (vibration noise left in epsilon). Reachable from the viewer epsilon path (`window.py:145`). *Fix: add the `_detrend_batch` NaN guard to `_detrend_segment`; gate the `except ValueError` on an explicit `shear.shape[0] < 2*nfft` length check.*

**M-3. `_nan_singular_bins` crashes (LinAlgError: SVD did not converge) on NaN accelerometer data — a regression.**
`src/odas_tpw/scor160/goodman.py:58-62` (called `:194,:260`) — The audit's new `np.linalg.cond(AA)` runs SVD over the whole batch; SVD raises `LinAlgError` if any matrix has NaN/inf, and the surrounding `errstate` only suppresses warnings. The NaN-in-AA path is reachable by design (l2 `_filtfilt_nan` restores NaN → l3 vib windows), uncaught up the chain, aborting the whole `.p` file. Pre-fix code and ODAS `mrdivide` did not crash. *Fix: mask non-finite AA matrices (`np.all(np.isfinite(AA),axis=(-2,-1))`) and treat them as singular before `np.linalg.cond`; add a NaN-AA regression test.*

**M-4. process_l2 despike change-mask counts NaN samples as despiked (NaN!=NaN), inflating DESPIKE_FRACTION_SH and tripping QC flag bit 2.**
`src/odas_tpw/scor160/l2.py:130,146` — `despike_mask = cleaned != before` records every unchanged NaN as "replaced" because NaN!=NaN is True; `despike()` itself was hardened (`despike.py:84`) but the caller wasn't. On real benchmark data with shear gaps (e.g. `MSS_Baltic.nc`, 19,246 NaN), windows exceed `despike_fraction_limit=0.05` and get flag bit 2, dropping good epsilon from EPSI_FINAL. *Fix: `changed = (cleaned != before) & ~(np.isnan(cleaned) & np.isnan(before))` at both lines.*

**M-5. Pipeline passes raw `speed_tau` (default 1.5) into L2Params instead of vehicle-resolved `file_speed_tau`.**
`src/odas_tpw/rsi/pipeline.py:236` (vs `:180,:226`) — L1 speed and profile detection use `file_speed_tau` (glider tau 3.0/5.0), but L2's re-applied butter low-pass uses the raw 1.5, re-smoothing already-tau-smoothed glider speed at the wrong cutoff (0.68/1.5 vs 0.68/tau). VMP unaffected (tau already 1.5); only non-VMP vehicles. *Fix: pass `speed_tau=file_speed_tau` at `:236`.*

**M-6. MLE chi fit returns epsilon biased ~1.7x high (chi ~1.15x) from a too-low fixed `chi_obs`.**
`src/odas_tpw/chi/l4_chi.py:243-265` and `chi/chi.py:335,350` — In `fit_method='mle'` (Method 2), `chi_obs` is the attenuated, band-limited variance (no `/|H|^2`, no unresolved-variance correction), held FIXED while only kB varies; the NLL inflates kB to compensate, and `epsilon=(2π·kB)^4·ν·κ_T^2` amplifies it. Default `iterative` path is unbiased. *Fix: iterate the MLE (recompute `chi=6κ_T·obs_var·_variance_correction(...)` between kB fits) or profile out chi analytically.*

**M-7. All viewers use full-cast mean temperature for per-window viscosity (and chi noise), biasing displayed epsilon/chi up to ~14% with depth.**
`src/odas_tpw/rsi/viewer_base.py:93,265`, `quick_look.py:69,221` — `mean_T = np.mean(T_slow)` averages the entire-deployment `T1` (all profiles, out-of-water samples), then feeds `visc35(T_mean)` to every window and `gradT_noise(T_mean)`. On real data mean=26°C is applied to 19°C deep windows (visc error ~14-17%). Pipeline correctly uses per-window `l3.temp`; viewers diverge. Presentation-only. *Fix: interpolate T1 to the fast time base and average over the window slice.*

**M-8. Pipeline ignores documented `epsilon.fft_length` default (uses 1024 not 256); diss dir signature lies.**
`src/odas_tpw/perturb/pipeline.py:1138,1183-1199` — `eps_cfg = config.get("epsilon",{})` is the RAW (unmerged) config, so an omitted `fft_length` falls to `_compute_epsilon`'s own default 1024 instead of the perturb DEFAULT 256. Meanwhile `_setup_output_dirs` hashes `merge_config(...)` = 256, so the diss dir `.params_sha256` claims 256 while data used 1024; implicit and explicit `fft_length:256` runs collide/diverge. *Fix: `eps_cfg = merge_config("epsilon", config.get("epsilon"))` (and chi) before use.*

**M-9. `diss_length_seconds` for QC and stratification windows mis-computed when `fft_length` is defaulted (4x too short for eps, 2x for chi).**
`src/odas_tpw/perturb/pipeline.py:1162-1167,1222-1224,1336` — Hardcoded fallback `4*256` vs the actual `4*1024` used by `_compute_epsilon`. The undersized window is passed as the QC segment-drop window (`apply_qc_to_dataset`) and the stratification half-window (`_attach_window_stratification`), weakening the drop gate and biasing background N²/dT/dz/Γ/K_T. *Fix: derive from the merged config or the `diss_length` attr the NC records.*

**M-10. `trim_p_file` destroys the source file when the trim output resolves to the source path.**
`src/odas_tpw/perturb/trim.py:171-172` — `open(dest,"wb")` truncates before `src_f.read(...)`; when `dest==source` (files already under `<root>/trimmed/`, or `root=None` collision) the file is wiped to 0 bytes while `TrimResult` reports `action="trimmed"`. No `samefile` guard; `_check_unique_outputs` only checks dest-vs-dest. *Fix: no-op/raise when `dest.resolve()==source.resolve()`, or write to a temp file + `os.replace()`.*

**M-11. `merge_p_files` truncates/corrupts the base file when the merge output resolves to `chain[0]`.**
`src/odas_tpw/perturb/merge.py:213-216` — Same truncate-before-read footgun: `open(dest,"wb")` zeroes `chain[0]` before `copyfileobj`, yielding a headerless, config-less, unparseable file and destroying the base. *Fix: guard `dest.resolve()==chain[0].resolve()`, or build into a temp file then `os.replace()` after all reads.*

**M-12. Hotel pchip interpolation (the default) crashes on any channel containing a fill value / NaN.**
`src/odas_tpw/perturb/hotel.py:417-419` (`_nc_array:138-151`) — `_nc_array`/CSV map `_FillValue`→NaN by design, but `PchipInterpolator(...)` (default `kind='pchip'`, `config.py:46`) raises `ValueError: y must contain only finite values` on any NaN, aborting `merge_hotel_into_pfile` (and the campaign run). Telemetry gaps are routine. *Fix: mask `np.isfinite(data)` before building the interpolator (both pchip and interp1d branches); all-NaN → edge fill.*

**M-13. NaT datetime values decode to a bogus epoch (~-9.22e9 s) instead of NaN in GPS/hotel time parsing.** *(overlaps DEFERRED dateline area — distinct defect)*
`src/odas_tpw/perturb/gps.py:30` and `hotel.py:166,175` — `.astype(np.int64)/1e9` turns NaT (`int64.min`) into a finite ~1678-BCE time. `np.nanmin` then treats it as a real coverage bound, disabling the out-of-coverage warning and feeding interp1d a spurious node → corrupted positions / TEOS-10 salinity. *Fix (NOTE: the finding's own suggested float-cast is wrong — float cast keeps the sentinel): mask with `np.isnat()` (or `int64==iinfo.min`) and set those to NaN.*

**M-14. eps-chi plot panels crash on all-NaN data with no manual color limits (`LogNorm(None,None)`).**
`src/odas_tpw/perturb/plot/eps_chi.py:354-376` — `layout.quantile_limits` returns `(None,None)` when a panel has no finite-positive data, passed straight into `LogNorm`→`colorbar`, raising `ValueError: Invalid vmin or vmax` and aborting the whole figure even when eps is fine. Realistic when chi is fully QC-dropped or unalignable to eps. The `profiles` subcommand guards this; eps-chi doesn't. *Fix: skip/placeholder panels whose limits are `(None,None)` or which lack finite-positive data.*

**M-15. `profiles --var` on a 1-D (profile-only) variable crashes with a cryptic transpose error.**
`src/odas_tpw/perturb/plot/profiles.py:245` — `lat/lon/stime/etime` are in `data_vars` (1-D along `profile`), pass `_available`, then `dss[name].transpose("bin","profile")` raises `ValueError: Dimensions {'bin'} do not exist`, aborting the run on an apparently-valid request. *Fix: require `{'bin','profile'} ⊆ dims` in `_available`, routing profile-only vars to `missing`.*

**M-16. Sibling ULP-depth bug: per-profile eps/chi merge splits equal depth bins, causing silent data loss in `combine_profiles`.**
`src/odas_tpw/rsi/pipeline.py:556` (interacts with `combine.py:50-81`) — The `combine.py` fix added `_depth_key` quantization, but the earlier eps/chi `merge(extra, join="outer")` has no ULP tolerance. With non-binary-exact `bin_size` (0.1, 0.2…), eps and chi grids (different L4 pressure minima) split each overlap depth into two rows; `combine_profiles` then collapses both to one key and the NaN row overwrites the real value. Default `bin_size=1.0` is safe; reachable via `run_pipeline(..., bin_size=0.1)`. *Fix: pass a shared `pres_range` to both `bin_by_depth` calls, or snap chi's depth_bin onto eps's grid within `_DEPTH_ATOL` before the merge.*

**M-17. CTD per-file NCs are never pruned: orphaned casts from dropped `.p` files leak into the CTD combo.**
`src/odas_tpw/perturb/pipeline.py:1916-1918` (CTD combo `:2106-2117`; `ctd.py:407-409`; `combo.py:72`) — The orphan-prune fix covers only `('profiles','diss','chi')`. The config-hashed (reused) CTD dir holds `{stem}.nc`, which the combo globs `*.nc`; a re-run on a reduced input set leaves stale casts that get concatenated into the published CTD time series. Even adding `'ctd'` to the loop fails — the prune's `*_prof*.nc` glob/prefix never matches `{stem}.nc`. *Fix: a CTD-aware prune deleting `{stem}.nc` whose stem ∉ valid_stems (exact membership).*

---

## Minor

**Mn-1. 2-id (32-bit split) channel high/low word assigned by `sorted(ids)` instead of config order.**
`src/odas_tpw/rsi/p_file.py:451-468` — `id_even,id_odd = sorted(ids)` assumes the smaller ID is the low word; ODAS pairs by config-declared `_E`/`_O` order. Latent (ARCTERX JAC_C `id=48,49` ascending), but a high-word-first config would swap the 16-bit halves and corrupt conductivity/salinity. *Fix: `id_low,id_high = ids[0],ids[1]`.*

**Mn-2. `_adis_14bit` mis-decodes inputs with both status bits set (0xC000-0xDFFF).**
`src/odas_tpw/rsi/channels.py:41-60` — Strict `val < -(2**14)` excludes 0xC000 itself; 8192 inputs decode to a large negative instead of their low-14-bit value (faithful port of ODAS `adis.m:48`). Affects only the auxiliary inclinometer channels. *Fix: `low14=(word.astype(uint16)&0x3FFF); where(low14>=0x2000, low14-0x4000, low14)`.*

**Mn-3. `convert_jac_c` replaces `v_part==0` with 1, masking divide-by-zero with a finite wrong value.**
`src/odas_tpw/rsi/channels.py:222-231` — ODAS yields Inf (recognizable bad sample) for `v==0`; Python produces a plausible-but-meaningless conductivity that survives QC into salinity/density. *Fix: `ratio = np.where(v_part==0, np.nan, i_part/v_part)`.*

**Mn-4. `p_to_L1` hardcodes CHLA/TURB/DOXY units, overriding the actual converter/config units.**
`src/odas_tpw/rsi/convert.py:368-404` — CHLA labeled `ug L-1` while the poly converter produced `ppb` (numerically compatible here, but breaks provenance and risks mismatch for other configs). *Fix: use `pf.channel_info[ch]['units']`, falling back to the ATOMIX default only when empty.*

**Mn-5. Goodman cleaning silently skipped when `diss_length == 2*fft_length` (off-by-one).**
`src/odas_tpw/rsi/window.py:144` — Early-return uses `< 2*fft_length` but the Goodman guard uses `> 2*fft_length`; at exact equality neither fires and plain CSD runs with no accel noise removal. Viewer-only. *Fix: change `>` to `>=`.*

**Mn-6. Chi from NetCDF sources loses FP07 electronics-noise calibration (`therm_cal` never populated).**
`src/odas_tpw/rsi/chi_io.py:730-757` — The NetCDF branch never builds `therm_cal`, so it falls back to `[{}]`, and `gradT_noise_batch` uses defaults (beta_1 ~5% off, beta_2 dropped). NetCDF-sourced chi has a less-accurate noise floor than the same `.p` data. *Fix: re-parse the `configuration_string` global attr and call `_extract_therm_cal` like the `.p` branch.*

**Mn-7. `extract_profiles` computes `e_fast` without clamping to `len(t_fast)`; over-runs for full-record NetCDF inputs.** *(same `_load_from_nc` path as a DEFERRED handle-leak item)*
`src/odas_tpw/rsi/profile.py:134` — Unclamped `(e_slow+1)*ratio` over-runs when `len(t_fast) != len(t_slow)*round(fs_fast/fs_slow)` (non-integer ratio / trimmed fast record), raising a shape-mismatch on `t_fast_var[:]=`. `dissipation.py:169` clamps the same expression. *Fix: `e_fast = min((e_slow+1)*ratio, len(t_fast))`.*

**Mn-8. `quick_look` chi-spectra panel ignores user `diss_length`, can fall back to a whole-segment chi spectrum.**
`src/odas_tpw/rsi/quick_look.py:67` (sig `:33-46`, call `:542-555`) — `_compute_chi_spectra` hardcodes `diss_length=None` (→4*fft_length) while the epsilon feeding Method-1 chi used the real `self.diss_length`; when `self.diss_length < 4*fft_length` the chi spectrum spans many depths but is fit with single-window epsilon. *Fix: add a `diss_length` param and pass `self.diss_length`.*

**Mn-9. `process_l3_chi` divides by zero when `overlap == diss_length` (`diss_step==0`).**
`src/odas_tpw/chi/l3_chi.py:113,254` — Unvalidated `overlap` (public L3Params/CLI) yields `diss_step=0` → `ZeroDivisionError`, aborting chi for the file (uncaught by the pipeline's `except (ValueError,RuntimeError)`). Defaults safe. *Fix: validate `0 <= overlap < diss_length` or `diss_step = max(diss_length-overlap, 1)`.*

**Mn-10. `detect_bottom_crash` crashes (ValueError) on an all-NaN depth segment instead of returning None.**
`src/odas_tpw/processing/bottom.py:89-95` — `NaN < depth_minimum` is False, so `np.arange(..., NaN, ...)` raises, violating the return-None contract; reachable via an all-NaN `P_fast` slice (pipeline wraps it in try/except so it degrades to a warning there). *Fix: `if not np.isfinite(max_depth) or max_depth < depth_minimum: return None`.*

**Mn-11. `detect_bottom_crash` reports the shallow (left) edge of the spike bin, underestimating bottom depth by up to one bin width.**
`src/odas_tpw/processing/bottom.py:121-124` — Returns `bins[i]` (left edge) where the sibling `top_trim` uses `bin_centers[i]`; the caller trims the profile end up to ~4 m too shallow (conservative). *Fix: `bottom_depth = 0.5*(bins[i]+bins[i+1])`.*

**Mn-12. `mk_epsilon_mean`/`mk_chi_mean` iterative loop can reduce a row to a single (possibly worst) survivor, contradicting the "keep both with 2 probes" policy.**
`src/odas_tpw/processing/epsilon_combine.py:124-149` (mirror `chi_combine.py:152-172`) — Starting from ≥3 probes, the `range(n_probes-1)` loop keeps dropping until one remains; a log-symmetric triple `[1e-9,1e-7,1e-5]` collapses to the extreme `1e-9`, and the final equidistant pair is broken arbitrarily by `nanargmax`. Pre-existing; only ≥3-probe configs (VMP-250 has 2). *Fix: stop dropping once ≤2 finite probes remain (keep their geometric mean) or pick the survivor nearest the ln-median.*

**Mn-13. Hashing of list/dict config values is not type-normalized: `[1,99]` and `[1.0,99.0]` hash differently.**
`src/odas_tpw/config_base.py:163-167,20-40` — `_canonicalize_section` uses non-recursing `_normalize_value`, so list/dict elements bypass int/float collapse (violating `test_float_int_normalization` for list keys like `speed.amplitude_quantile`), causing spurious output-dir cache misses/recompute. *Fix: use `_normalize_nested`.*

**Mn-14. binned/combo directories re-versioned by unrelated binning sub-keys (`chi_width` churns `diss_binned`).**
`src/odas_tpw/perturb/pipeline.py:1944,1958,1927` — All binned/combo dirs hash the full merged `binning` section, so changing `chi_width` re-creates byte-identical `diss_binned`/`profiles_binned` dirs and forces re-binning. *Fix: hash each stage on only the sub-keys it consumes.*

**Mn-15. GPS interpolation with a fill-valued lat/lon node poisons whole spans with NaN, then seawater silently substitutes lat=0/lon=0.** *(overlaps DEFERRED dateline area — distinct: NaN-node poisoning)*
`src/odas_tpw/perturb/gps.py:307-308,271-272`; `seawater.py:50-51` — A single NaN node makes `interp1d` return NaN across the spanning interval; `add_seawater_properties` replaces NaN positions with 0,0 for `gsw.SA_from_SP`, silently degrading absolute salinity to equator/prime-meridian over a multi-sample span. *Fix: mask non-finite `(t,lat)`/`(t,lon)` pairs before building the interpolators; warn rather than substitute 0.*

**Mn-16. `add_seawater_properties` silently strips masked arrays, exposing `_FillValue` garbage as real T/C/P.**
`src/odas_tpw/perturb/seawater.py:43-47` — `np.asarray(MaskedArray)` discards the mask, feeding `-999`/`1e20` fills to gsw as measurements. Latent (current caller passes plain arrays) but inconsistent with the gps/hotel `.filled(nan)` convention. *Fix: `np.ma.filled(np.ma.asarray(x).astype(float64), np.nan)` per input.*

**Mn-17. Schema hardcodes non-UDUNITS units `'cpm'` and `'PSU'`, defeating the module's CF/UDUNITS goal.**
`src/odas_tpw/perturb/netcdf_schema.py:169,248,253,302,307,322,327` — `cf_units`/compliance-checker reject `cpm` and `PSU`; `canonicalize_units` is never applied to schema literals. (Same `cpm` also written in `rsi/dissipation.py:441,545,554,563`.) *Fix: wavenumber→`m-1`, practical salinity→`1` with the label in `long_name`.*

**Mn-18. `bin_by_time` silently ignores the `diagnostics` flag — time-binned diss/chi/profiles lose `n_samples` and `*_std`.**
`src/odas_tpw/perturb/binning.py:514-589` — The parameter is declared but never used; the depth path emits `*_std`/`*_n`/`n_samples` and the time path emits none, so `method='time'` + `diagnostics=true` silently drops every requested diagnostic. *Fix: implement diagnostics in `bin_by_time` mirroring `_bin_snapshot`, or warn when unsupported.*

**Mn-19. Linear/pchip interpolation of discrete hotel flags over-flags neighboring diss/chi segments.**
`src/odas_tpw/perturb/qc_gate.py:96-100` (with `hotel.py:446,418`) — A flag set at one hotel timestamp is interpolated to fractional values; `qc_gate` treats `|x|>=0.5` as set, so segments touching only the interpolated ramp get NaN'd (fail-safe over-flagging). Default empty `*_drop_from`. *Fix: force `interp:nearest`/step-hold for `*_drop_from` channels, or document the requirement.*

**Mn-20. `_time_bin` diagnostic `n_samples` counts NaN samples — overstates the effective averaging N.**
`src/odas_tpw/perturb/ctd.py:78` — `np.bincount` over all in-range samples regardless of finiteness, while the mean/median use only finite values; also a single shared count across channels with different NaN footprints. Diagnostics-only. *Fix: count finite-and-in-range (mirror `binning._bin_finite_counts`).*

**Mn-21. `make_combo` time-dedup silently drops data at coincident bin-center timestamps.**
`src/odas_tpw/perturb/combo.py:112-119` — `keep[1:] &= tk[1:]!=tk[:-1]` keeps only the first of exact float-equal timestamps; when two CTD files' phase-aligned windows produce identical centers, the second file's distinct measurements are discarded without warning. Rare (requires exact float collision). *Fix: aggregate duplicate-timestamp rows, or at least log a dropped-duplicate count.*

**Mn-22. eps-chi crashes when `chiMean` is on the combo but no `chi_NN` dir exists.**
`src/odas_tpw/perturb/plot/eps_chi.py:262-280` — Both productive branches require `chi_dir is not None`, but `_load_chi_from_combo` needs `chi_dir` only for cosmetic title attrs; if per-profile chi dirs were archived while the combo remains, it raises `SystemExit('No chi_NN dir')` despite having all panel data. *Fix: when `has_chi_mean` and `chi_dir is None`, still load from the combo with `attrs={}`.*

**Mn-23. Chi L4 produces NaN (silent window drop) on an all-NaN-temperature window, where epsilon substitutes 10 °C.**
`src/odas_tpw/chi/l3_chi.py:156` (`l4_chi.py:100,108,276`) — `np.nanmean` returns NaN for an all-NaN window → `nu=NaN` → NaN chi silently dropped, while the epsilon side (`scor160/l4.py:154-160`) substitutes 10 °C with a warning. The fix's comment claims parity it doesn't have (single-NaN only). Latent (production temp is interpolated/finite). *Fix: `T_means = np.where(np.isfinite(T_means), 10.0)` with a warning.*

**Mn-24. Chi L3 speed window uses `nan_to_num` (collapses whole window to 0.01 m/s) instead of `nanmean`, inconsistent with the temperature handling on the same lines.**
`src/odas_tpw/chi/l3_chi.py:148-150` — One NaN sample makes `np.mean`→NaN→`nan_to_num`→0.01 (a ~60x speed underestimate) corrupting the wavenumber axis, spectral correction, FP07 tau, and noise floor for that window; temperature on the adjacent line uses `nanmean`. Latent (rsi `prepare_profiles` scrubs speed upstream). *Fix: use `np.nanmean` then floor.*

**Mn-25. Epsilon L3 sibling speed path: `max(W,0.05)` does not floor a NaN, propagating NaN into the wavenumber axis.**
`src/odas_tpw/scor160/l3.py:182,201-203` — `np.mean` of a NaN-bearing window → NaN; Python `max(nan,0.05)==nan`, so the floor is ineffective and `kcyc_w=F/W` NaNs the window's epsilon. The chi sibling got a `nan_to_num` scrub; this epsilon sibling did not. Reachable on the scor160 benchmark/direct-NetCDF path. *Fix: `W = max(np.nan_to_num(W, nan=0.05), 0.05)`.*

---

## Nit

**Nt-1. `convert_therm` prefers `beta_1` over `beta`, opposite of ODAS precedence.** `src/odas_tpw/rsi/channels.py:101-106` — Only matters if a config carries both keys (rare). *Fix: check `beta` first.*

**Nt-2. `_epsilon_ds_to_l4data` emits "Mean of empty slice" RuntimeWarning for all-NaN epsilon windows.** `src/odas_tpw/rsi/chi_io.py:345` — Warning noise; NaN handled downstream. *Fix: wrap in `warnings.catch_warnings`.*

**Nt-3. Anti-aliasing marker drawn at `f_AA` but spectral fits use `0.9*f_AA`.** `src/odas_tpw/rsi/quick_look.py:494,607`; `viewer_base.py:629,889` — Marker sits ~11% high in wavenumber. *Fix: draw at `0.9*f_AA/speed` or relabel.*

**Nt-4. `batchelor_grad` does not sanitize divide-by-zero/NaN (unlike `kraichnan_grad`).** `src/odas_tpw/chi/batchelor.py:116-119` — Latent (all callers guard kB≥1). *Fix: mirror `kraichnan_grad`'s `errstate`+`np.where(isfinite,...,0.0)`.*

**Nt-5. `_select_sections` raises IndexError on empty speed/pressure input.** `src/odas_tpw/scor160/l2.py:231-234` — `good[0]` on an empty array; unreachable with valid benchmark data. *Fix: `if n==0: return section_number`.*

**Nt-6. `visc()` is discontinuous (~0.7%) at the S=35, P=0 seam between visc35 and Sharqawy.** `src/odas_tpw/scor160/ocean.py:65-84` — By-design ODAS-parity seam. *Fix: document the discontinuity (or blend).*

**Nt-7. Probe variables are lexicographically sorted, misordering `e_10+`/`chi_10+` before `e_2`.** `src/odas_tpw/processing/epsilon_combine.py:53` (`chi_combine.py:71`) — Cosmetic; outputs are order-invariant and named by index. *Fix: `key=lambda n: int(n.split('_')[1])`.*

**Nt-8. Malformed config section (list/scalar) raises an opaque TypeError instead of a clear validation error.** `src/odas_tpw/config_base.py:95` — `dict(v)` runs before `validate_config`. *Fix: `isinstance(v, Mapping)` check naming the section.*

**Nt-9. `fft_length: null` in config crashes `diss_length_seconds` with TypeError.** `src/odas_tpw/perturb/pipeline.py:1162,1165` — `4 * None` aborts the file (single-job mode aborts the run). *Fix: `eps_cfg.get("fft_length") or 256`, or route through `merge_config`.*

**Nt-10. `_apply_fom_cut`/`_nan_excluded_probes` `e_N`/`chi_N` masking is dead code (companions don't exist yet).** `src/odas_tpw/perturb/pipeline.py:443-449,803-807` — Branches never fire (companions created later in `mk_*_mean`); 2-D mask still correct. Docstrings overstate the effect. *Fix: drop the branches or move the cuts after `mk_*_mean`.*

**Nt-11. chi/epsilon colorbar units mislabeled with a spurious `s` factor.** `src/odas_tpw/perturb/plot/eps_chi.py:375` — `χ/ε` should be `K^2 kg J^-1`, not `K^2 s kg J^-1`. *Fix: drop the `s`.*

**Nt-12. Iterative-fit FOM mask diverges from Method 1/MLE despite a docstring claiming parity.** `src/odas_tpw/chi/chi.py:590-606` — Strict above-noise mask with no `_valid_wavenumber_mask` fallback; chi FOM is diagnostic-only (no QC gate by default), so output unaffected. *Fix: reword the comment or route through `_valid_wavenumber_mask`.*

**Nt-13. Hotel `time_format='auto'` rejects a whole file when a per-channel time grid is magnitude-ambiguous, even if the default grid is unambiguous.** `src/odas_tpw/perturb/hotel.py:180-194,291,332,401` — Per-channel grids re-run 'auto' disambiguation. Fails loudly. *Fix: resolve the relative/epoch sense once and pass explicit `seconds`/`epoch` per channel.*

**Nt-14. QC drop-flag reindex uses `np.maximum` instead of bitwise-OR, contradicting its "ORs a drop bitfield" contract.** `src/odas_tpw/perturb/plot/eps_chi.py:128` — Latent: 0b01∨0b10→3 but max→2; harmless today since all consumers test `>0`. *Fix: integer bitwise-OR accumulator for the flag path.*

**Nt-15. Chi all-NaN window emits an unhandled "Mean of empty slice" RuntimeWarning from `np.nanmean`.** `src/odas_tpw/chi/l3_chi.py:156,161` — Surface symptom of Mn-23's missing all-NaN guard; result (NaN chi) is correct. *Fix: suppress the RuntimeWarning and/or apply the 10 °C fallback.*

---

*Loop status: 1 of 12 rounds executed; not converged (the 2-consecutive-dry-round stop was not reached). Counts — Blocker 0, Major 17, Minor 24, Nit 15.*

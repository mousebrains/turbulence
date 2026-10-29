# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- **A bead's slow response path, divided out of its chi
  (`instruments.<key>.fp07_slow_response`, rsi-tpw `[chi] fp07_slow_response`;
  issue #252).** FP07 bead T2005 (VMP SN194 T2, 2021-06-29 to 2022-06-20) loses
  ~12% of its amplitude through a ~110 ms parallel path,
  `|H_slow|^2 = |(1-beta) + beta/(1 + i 2 pi f tau_s)|^2` (beta 0.124, tau_s
  110 ms common to its deployments; beta 0.09-0.13 across single-pole model
  choices). Above its ~1-1.5 Hz corner that is a near-flat 0.77-0.85 in power,
  so its chi read ~20-25% low. The new module `chi/fp07_slow.py` holds the
  response, the configuration parser and a per-file guard. Keyed by BEAD
  SERIAL, read from each file's configuration exactly as `fp07-pair` reads it,
  with a mandatory UTC span tested against the file's `start_time` (several
  spans per bead allowed, never overlapping). `l4_chi` divides the bead's
  observed AND noise spectra by `|H_slow|^2` in every bin, unweighted, before
  the `fp07_response` step, for Method 1, Method 2 (iterative and MLE), the
  detection gate, band, fom and K_max_ratio alike; stored spectra stay raw.
  Dividing S and N by one factor keeps each bin's S/N: under Method 1 the
  gates, band, K_max and K_max_ratio are unchanged and chi rises by a weighted
  mean of 1/e over the band. On one SUNRISE 2021 WaltonSmith file (136
  profiles, Method 1) T1's chi, K_max_ratio and chiQCFallback were identical,
  fom moved by at most 0.14%, and T2's chi rose by a median ln 0.252
  (per-window 1.20-1.30x). Under Method 2 the fitted kB moves too, by design:
  on synthetics the corrected chi, kB and epsilon_T recover what a bead
  without the path gives. A file whose bead is outside every span is left
  uncorrected and says so; a file without the bead keeps its chi and records
  only the beads it carries (`fp07_slow_response_serials`); a bead on two
  channels inside its span, or an unreadable configuration, is an error
  (outside every span a doubled serial is only recorded: it is a copy-paste
  placeholder there, and nothing would be applied). The guard measures the
  file's own change in median ln(S_bead/S_partner) from 0.25 to 2 Hz (32-s
  windows of 8-s Hann segments of the gradient), which the chi spectra cannot
  see, and warns when it differs from the configured path's prediction by more
  than 0.10 (set from 85 healthy T2005 files) with at least 40 windows; perturb
  pools a `.p` file's profiles for that verdict, since one SUNRISE cast is one
  window, and records it beside the chi products
  (`<stem>.fp07_slow_response.json`), so a rerun that skips the file repeats
  its notices. It catches a probe swap without a config update and a faded
  path. `rsi-tpw chi` ends with WARNING lines for unmatched bead keys, inputs
  outside every span, and inputs the guard flagged or could not judge, and one
  when the inputs are per-profile NetCDFs, which never reach 40 windows.
  Refused beside `fp07_tau_scale` and `fp07_response` (both fitted on spectra
  that already contain the path) and by `rsi-tpw pipeline`. Valid on
  calibrated temperature only: the path is relative to the DC level an in-situ
  calibration fixes. Products record `fp07_slow_response_*` attributes
  (applied, bead, beta, `tau_s_ms`, span, the guard's measured and predicted
  change, its verdict and windows); perturb-diag draws the factor into the
  model; the run summary names bead keys that matched nothing, files outside
  every span, and files the guard flagged or could not judge, keyed by output
  stem. Spans keep fractions of a second as hashed. Unset, every product and
  every hash is unchanged.
- **Each FP07 bead's time constant, measured and carried in the calibration
  sheet (`fp07-pair tau`, `fp07-cal registry response`; closes the last open
  question of issue #189).** `fp07-pair tau` fits one single pole per bead to
  the pair's median ln(S2/S1) (the `fp07-pair fit` machinery) over 1-25 Hz of
  reduced frequency at the pair's median speed, over an explicit time span on
  one instrument, with 16-84% intervals from a bootstrap over recordings
  (`.p` files; their profiles are not independent); it also records the
  model-free power ratio at 10 and 20 Hz and fits each time half, so a bead
  that slowed within a deployment shows. A rule set on synthetics decides
  whether a pair identifies a tau at all (>= 16 chi files from >= 5
  recordings, the beads differing at 20 Hz by >= 3 bootstrap sigma and 16% in
  power, each interval bounded and within a factor 1.5, flat gain below a
  factor 2, no bead dead in any file, neither probe's 60-100 Hz floor above
  1.25x its noise model); matched beads, a failing or dead bead, and
  unmodelled content on a probe give no tau, only ratios. The thresholds used
  (and `n_boot`) are recorded in the record and the sheet, and the loader
  refuses an identified block that breaks its own. One JSON record per pair;
  `fp07-cal registry response` inserts it into an entry as a strictly
  validated `response:` block (text insertion, as `registry add`), and the
  `.md`/`.csv` views render it at the pair's own speed `W_ref` only (W^-1/2
  across gliders and VMPs is not established, so no tau at 1 m/s). It is a single-pole-equivalent value: the pole form is not
  determined and no external reference anchors the absolute level. Nothing in
  chi processing reads it yet. `fp07-pair fit` is unchanged (pinned by a golden
  test against its pre-change output). The sheet now carries a response on 19
  of its 23 entries (10 identified; the four ARCTERX-2023 MicroRider entries
  have no chi products). The run found T1593 on the ADC rail (spectrum
  identically zero) through the last SUNRISE 2021 Pelican file before its
  replacement, a noise model 2-5x above the
  observed floor on SN194 from that swap through WaltonSmith, and excess
  high-frequency content on SN194 T2 in SUNRISE 2022 PointSur that makes T2005
  read 4.9 ms. That last case prompted rule 7 (the floor refusal), a
  tightening made after the real results were seen and set on synthetics; it
  leaves SUNRISE 2022 PointSur without a tau for either bead. T1593's SUNRISE
  2021 Pelican entry now ends with its own last file (SR1P_515), not the run's.
- **FP07 calibration sheet (`fp07-cal registry`)**:
  `microstructure_sensors/fp07_calibrations.yaml` records the in-situ
  calibrations of individual FP07 beads. An entry is one bead in one deployment:
  the instrument, serial, channel and dates; the reference thermometer with
  its serial and calibration date; the fit settings and software version; the
  coefficients in `.p` config form (`t_0`, `beta_1..n`); the six bridge
  constants that define L; and the fit's rms, held-out error, block scatter
  and fitted range, in mK and °C. It is the single source of truth.
  `fp07_calibrations.md` (one section per bead, readable without this code)
  and `fp07_calibrations.csv` (one row per fit) are generated from it by
  `fp07-cal registry render`. `fp07-cal registry check` (exit 1) and a test
  fail when either view is stale. The loader refuses unknown or missing keys,
  bad serials, dates out of order, a bead or channel in two places at once, an
  order that disagrees with its `beta` terms or held-out errors, duplicate
  entries, and coefficients that cannot describe an FP07 over the stated range
  (`t_0` not in kelvin, `beta_1` <= 0, |L| beyond the bridge clip, T not
  falling with L). Each defect has its own error class.
  `Registry.prior_for` (and `fp07-cal registry show <serial>`) gives a prior
  for a fresh deployment. Its source is the bead's widest-range calibration
  not flagged `short`, `fast-descent` or `episode`, preferring the target
  instrument and channel. It carries the sheet's transfer uncertainties from
  the cross-deployment comparisons in issue #189 (sizes and evidence are in the
  generated sheet), and reports how far the most recent calibration sits from
  it. It returns what a `t_0`-only refit needs; `fp07-cal fit` makes that
  refit with `fit.prior` (next entry). Seeded with 23 pooled fits of 12
  beads: T1592, T1593 and T2005 on
  VMP SN194 from SUNRISE 2019-2022 and ARCTERX-2022; T1592 and T2007 on VMP
  SN142 in ARCTERX-2025; and eight MicroRider beads on two Slocums from
  ARCTERX-2023 and 2025 Interior.
- **`fp07-cal fit` with a bead's prior (`fit.prior`)**: for a bead that is
  already in the calibration sheet, the fit is made two ways. One is the full
  fit (orders 1-3, as before). The other holds the prior's shape (`beta`
  terms) fixed and fits only `t_0`, with the same robust iteration, and
  jointly with `dz` and `tau` when `fit.geometry` is on. Every candidate is
  scored by the temperature-split held-out error that picks the order, on the
  same folds. In `mode: auto` the prior's shape is kept in three cases:
  - no full order is a plausible curve for this bead (T falling as L rises,
    dT/dL within 10% of the prior's; the sheet's 12 beads differ by at most
    1.65%);
  - the deployment spans less than the sheet's 5 K shape span;
  - its held-out error is within `tie_fraction` (default 5%) of the full
    fit's, so ties go to the prior.

  `mode: offset` and `mode: full` force the choice. Either way the result is
  checked against the prior at the same L over the overlap of their fitted
  ranges. The limit is 2σ of `t_0` at the center, and 2√(σ_t0² + σ_shape²)
  across the overlap, where σ_t0 is ±15 mK, or ±18 mK when the bridge
  constants or the instrument differ. Without an overlap the prior is
  compared over this deployment's range as an extrapolation. Beyond a limit,
  or with an implausible exported fit, a `WARNING` goes to the log and the
  report: a possible unrecorded bead swap, reference problem or failing bead.
  A warning also fires when the offset-only fit's robust clip drops markedly
  more pairs than the full fit's. The bead comes from the `.p` configs and is
  refused when it is a placeholder or differs between the files of a run;
  `fit.prior.beads` overrides it per channel. Sheet entries of the bead whose
  dates overlap the fitted span are excluded from its prior.
  `coefficients.json` and `<ch>_report.md` record:
  - the source entry, the sheet's path and SHA-256, and the bridge comparison;
  - every candidate's held-out error, rms over all pairs and over its kept
    pairs, dropped fraction and plausibility;
  - the margin, the choice and why, and the comparison and verdict.

  The coefficients stay in config form, so `patch` is unchanged. A channel
  whose forced offset-only fit fails is recorded as a channel error instead
  of stopping the run. Without the block, every output is byte-identical to
  before; checked on ARCTERX-2022 Wake (geometry on and off) and Interior.

  On bead T1592's SUNRISE 2021 WaltonSmith (27.2-29.2 degC, narrow) the
  prior's shape is kept, and its held-out error is also 5.4% better.
  Extrapolated to 18 degC, the order-1 fit is 30 mK off ARCTERX-2022 Wake's
  curve, where the prior's shape stays within 4-7 mK. That is the
  independent comparison: the prior comes from ARCTERX-2022 Interior, so
  agreement with Interior is circular. On ARCTERX-2022 Wake (18.4-28.2 degC)
  the full fit wins by 10.3%. With another bead's prior applied deliberately,
  the check flags -2.5 K.
- **`fp07-cal registry add`**: records one channel of a fit in the sheet, from
  its `coefficients.json` and the block table of its report. Options supply
  only what the record lacks: the deployment label, the instrument model, the
  method, the reference's identity, and the reference pole. The entry is
  inserted as text in the sheet's own layout, so every existing line and
  comment is kept. The whole sheet must then validate, and the views are
  re-rendered. `add` refuses a duplicate bead and deployment, a placeholder
  serial, an untrusted lag without `--flag lag-gate-overridden`, and a fit
  flagged against its prior without `--allow-qc-flag`. A fit that kept the
  prior's shape is recorded as `order_choice: prior`, with `fit.shape_from`
  naming its source. The loader checks that its `beta` terms are the
  source's, and such an entry never supplies a prior itself. `fit_rms_mK` is
  the rms over the pairs the robust fit kept, as for every entry; fits made
  with `fit.prior` also record `uncertainty.rms_all_mK`, the rms over every
  pair. The sheet's file mode and line endings (LF or CRLF) are kept, an
  indented comment after the last entry stays with it, an empty
  `calibrations:` is accepted, and a lock file serializes concurrent adds.
- **`epsilon.despike_half_width` (rsi-tpw and perturb)**: the shear
  despiker's removal half-width [s], hard-wired at 0.04 s until now. The
  despiker replaces `N//2` samples before each detected spike and `2(N//2)`
  after (`N = round(hw * fs)`), so 0.04 s replaces ~60 ms per spike and 0.01 s
  ~14 ms. A VMP-5500 from an outside user carries a ~16 ms electrical pulse on
  both shear channels; injected at 20x the record rms into clean VMP shear,
  it biased epsilon by -0.05 dex at 0.04 s, -0.01 to -0.02 at 0.02 s and
  +0.02 to +0.05 at 0.01 s (median over windows; two pulse shapes and host
  files). A pulse below `despike_thresh` is never
  detected, at any half-width. Validated at load to (0, 0.5] s; at 0.04 the
  key is left out of the stage signatures, so existing outputs keep their
  hashes. No CLI flag, like the other despike keys.
- **`epsilon.record_locked_shear` (rsi-tpw and perturb; off by default)**:
  removes shear interference locked to the `.p` data record. That VMP-5500
  carries two ~16 ms pulses at fixed samples (6 and 81-82 of 512) of every
  record on both shear channels, each with an undershoot and a slow recovery;
  its accelerometers carry it only weakly, so Goodman cannot remove it, and
  the despiker caught it in every record in only 21 of 88 cast-channels
  (replacing < 1% of the shear in 36). Per profile and channel, the mean of
  the high-passed shear over quiet complete records (rms <= 3x the median)
  decides where -- samples >= 5 standard errors from its median, grown
  through >= 2, dilated 2 -- and each record's leave-one-out mean over the 16
  records on either side decides what is subtracted there, repeated (up to 3
  passes) while anything stays significant. That mean is taken with the
  high-pass at 1/10 of the L2 cut, because it is subtracted before L2's own
  high-pass (at the L2 cut it left H(1 - H) of the pulse at its low
  harmonics: at 1 Hz, perturb `fft_sec: 0.5`, the median bias of a 20x
  synthetic pulse was +0.031 dex, now +0.014). The local mean is needed
  because the interference is not stationary: on those casts it drifts
  several-fold within a cast, and one whole-profile mean left 1.5-75 standard
  errors of it in an eighth of a cast (median over eighths), the local mean
  2-6 fitted out of sample. Every other sample is left bit-identical, and
  nothing is corrected on clean shear (no sample of 924 VMP-250
  profile-channels reached 5 standard errors). A sample's place in its record
  is its index modulo the new `PFile.samples_per_record`, never time
  (`fs_fast` was 512.03 Hz). In synthetic pulses injected into clean VMP-250
  shear at 2/5/20x the record rms, drifting 6x with 5% jitter, the median
  epsilon bias went from +0.05/+0.10/-0.06 dex to +0.002/+0.003/+0.012 (90th
  percentile of |bias| 0.01/0.02/0.11). On the 44 real casts: detected in all
  88 cast-channels; the record-locked SNR at the corrected samples fell from a
  median 67 to 1.1 (no sample above 5.0 left); the despiked fraction from 2.3%
  to 0.19%; epsilon moved -0.05 dex (median of cast-channel medians; range
  -0.51 to +0.09: down where the pulse sat below the despike threshold, up
  where the despiker had been cutting it out); the sh1/sh2 disagreement fell
  in 27 of 44 casts (median |log10 ratio| 0.20 to 0.18). What a mean cannot
  remove is measured, not hidden: `record_locked_<channel>_phase_power_ratio`
  (and `_loud`, over the turbulent records the detection never sees) is the
  power left at the corrected samples over that at the others, ~1 when
  nothing is left; it exceeded 1.2 in 9 of 88 cast-channels (27 over loud
  records, up to 4.8), mostly where the pulse was strong, and those windows
  stay biased high. A slow tail below significance at every sample is left
  too, and still shows at the ~1 Hz fundamental. Per-profile NetCDFs, and
  `rsi-tpw nc` files, now always carry `samples_per_record_fast` and
  `record_offset_fast` (global attrs, data unchanged), so perturb's diss stage
  and `rsi-tpw eps` on a profile NetCDF can use it; a source without them is
  refused when it is on. The diss products also carry
  `record_locked_<channel>_records/_samples/_peak/_snr/_peak_range`. The
  estimate is per profile, so the corrected samples follow the profile bounds
  (rsi-tpw and perturb's top-trimmed profiles differ). True re-versions diss,
  chi and their binned/combo stages, not profiles. `rsi-tpw pipeline` refuses
  it, and a `despike_half_width` other than 0.04, rather than ignore them. See
  `odas_tpw.rsi.record_locked`.
- **`perturb-diag`: temperature, delta-T, conductivity and fall-rate panels,
  and a raw row for casts with no product file** (issue #210). The raw
  diagnostics row (`--diag`, the default) now leads with T1/T2 against the CT
  temperature, FP07 minus CT (on a robust x-range, so a few surface spikes
  cannot flatten a millikelvin offset), the CT conductivity, and W_slow with
  the speed used for epsilon, ahead of the inclinometer, accelerometer, shear
  and temperature-gradient panels. The CT is the first of these (temperature,
  conductivity) pairs the cast's file holds: the run's own
  `ct.T_name`/`ct.C_name` (read from the profiles directory's signature), then
  JAC_T/JAC_C, a glider's hotel `sci_water_temp`/`sci_water_cond`, and
  SBT1/SBC1. The conductivity comes from the same CT as the temperature when
  the file has it. A `ct.T_name` naming an FP07 channel (T1/T2) or the TEOS-10
  `CT` is never used; those runs fall back to the platform pairs, and the
  legends name whichever channel is plotted. Each channel is read on the clock
  its own dimension names, and a panel whose channels a cast lacks reads
  "n/a". A cast with a profiles file but no usable diss (or chi) file now
  draws this row, with the spectra panel naming the missing file, instead of
  a bare "no per-profile file" placeholder; in `perturb-diag mixing` a cast
  with no usable chi file still shows its shear spectrum on a K_rho click.
- **chi `mad_shape`: a spectral SHAPE statistic beside `fom` (issue #191).**
  `fom` integrates observed and model variance over the fit band, so a
  mid-band deficit cancels against excesses at the band ends; on Method 1 and
  the MLE fit, whose chi is that band integral, it is pinned near 1 by
  construction. Every chi product that carries `fom` (`rsi-tpw chi`, the rsi
  pipeline's `L4_chi_*`, perturb's per-profile `chi_NN`) now also carries
  `mad_shape(probe, time)`: `mean |r - median(r)|` in dex, with
  `r = log10(observed/model)` per bin over the same bins and against the same
  model (Batchelor/Kraichnan x FP07 `|H|^2` + noise) as `fom`. It is the
  epsilon `mad` taken about the window's own median instead of 0, so neither a
  constant level error nor the chi-squared bias of a log estimate (median
  below mean) counts. A perfect fit sits at a spectral-scatter floor (0.166
  dex at the usual 9.5 degrees of freedom), not 0;
  `odas_tpw.chi.chi.mad_shape_floor(dof)` gives it for any window. Use the
  window's own DOF: near-bed windows laid out with `bbl_diss_length` average
  fewer FFT segments than the file's `dof_spec` attribute assumes, so their
  floor is higher. On the CASPER-West windows of the issue
  (`K_max_ratio > 1.2`, `fom` 0.985-1.002) it reads 0.27 (p10 0.19, p90 0.43)
  against a Welch-noise null of 0.16, and the top third carries the #191
  residual coherently (+1.5 dex at k/kB = 0.02, -0.35 to -0.44 at 0.27-0.61).
  A diagnostic only: no gate and no config key, and the binned and combo
  products carry neither it nor `fom`. Every pre-existing variable is
  bit-identical (rsi-tpw `eps`/`chi` Methods 1, 2, MLE, Batchelor, `pipeline`,
  and a perturb run, on SN479 0006 and 0008). See `docs/chi_mathematics.md`
  §9. The §8 note that the Ruddick et al. (2000) MAD was "available for
  post-processing" was wrong -- it was never computed; it is the linear-ratio
  cousin of `mad_shape`, and its `2*sqrt(2/d)` limit does not apply here.
  **API change:** `chi.chi.ChiEpsilonResult` grows from 7 to 8 fields and
  `ChiFitResult` from 8 to 9 (`mad_shape`, last), and a `chi_func` passed to
  `chi.l4_chi._process_l4_chi` must now return 8 values. Code that unpacks
  either result positionally now raises `ValueError: too many values to
  unpack`; bind by attribute (`res.chi`, `res.fom`, ...) instead.
- **`instruments.<key>.exclude`: one probe masked over a pressure and/or time
  window** (issue #190). A shear probe with a pressure-related fault is bad
  inside a depth window and fine above and below it; `exclude_shear_probes`
  could only drop it for the whole deployment, and could not name a
  thermistor at all. On CASPER-West 2017 SN 194, sh2 read 15-18x high in a
  20 m window (from 10-16) while 65-130 m stayed at 1.01-1.02, biasing
  `epsilonMean` up to 4.25x, and no QC dropped any of it. Each entry is
  `{probe, P_min, P_max, start, stop, reason}`: `P_min <= P_mean < P_max`
  [dbar] and `start <= t < stop` [ISO-8601, UTC], tested against each window's
  own stored `P_mean` and center time, with an omitted bound unbounded and
  `reason` required. The masking is exactly `exclude_shear_probes`'s,
  restricted to the window: the probe's own estimates are set NaN before
  `mk_epsilon_mean` / `mk_chi_mean` (`epsilon`/`e_N` for `sh<N>`;
  `chi`/`chi_N`/`epsilon_T` for `T<N>`), so a shear entry reaches
  `epsilonMean`, chi Method 1, `epsilon_paired`, `K_rho` and `Gamma`, a
  thermistor entry `chiMean`, `chiLnSigma`, `K_T` and `Gamma` (never
  epsilon), and per-probe metadata is kept. A malformed entry fails at config
  load naming the instrument and entry; a probe the data does not carry fails
  each profile where the entry's window selects a window, naming it (an
  entry for a probe removed later in the deployment does not fail the
  profiles after it). Every per-profile file an entry changed records it
  (`probe_exclusion_<n>_*`, with the windows masked; a chi file also records
  the shear entries it inherits through `epsilonMean`), and the run summary
  reports what each entry masked, including none, an instrument key carrying
  `exclude` that matches no input file, and a thermistor entry not applied
  because chi did not run. A shear entry versions diss and chi, a thermistor
  entry chi only. Products of a config without `exclude` are unchanged.
  `exclude_shear_probes` is unchanged. See `docs/perturb/configuration.md`.
- **`PFile` refuses or repairs a mis-framed `.p` file instead of
  mis-assembling it.** Every reader, ODAS included, cuts a v6 file at a fixed
  record size, so one record of the wrong length puts every later header in
  the data and vice versa. MicroRider SN 156 (2022) wrote 28 of 29 files that
  way (one record of 33-98 matrices instead of 64); PFile returned channels
  mis-assembled under three misleading warnings, and ODAS 4.5.1 flags ~98% of
  the records as bad buffers and then stops in `patch_odas`. Each record's
  header must now repeat record 0's sizes, clock and matrix shape (not the
  version, which ODAS `patch_setupstr` rewrites in record 0 only -- 94 of 308
  sampled SeaChest files differ there). On a mismatch, `framing="repair"`
  (default) places every intact record at its own slot by record number,
  rebuilds the damaged slots by interpolation and reports them as bad-buffer
  spans on every channel, so epsilon/chi drop the windows they touch
  (`PFile.framing_report`); it refuses when the damage cannot be placed in
  time, and `framing="refuse"` always refuses. Correctly framed files take the
  unchanged fast path. On the 28 real files, every kept sample equals a copy
  re-framed independently (which ODAS reads the same way), and the 3-73
  lead-in records that copy dropped are kept. **One production file had it
  too:** VMP SN 194's `ARC1A046.P` (ARCTERX-2022 Wake), set aside during the
  cruise for "bad buffers" that were this misread. Its products used the
  mis-framed read (1/64 of every channel's samples were header words, and
  everything after record #13 was 0.48 s early) and need a rerun. Each rebuilt
  record costs about 70 s of epsilon/chi: the P_dP span is dilated by three
  deconvolution time constants. See `docs/rsi-tpw/odas_file_format.md`,
  "Record framing".

### Fixed
- **The chi noise model now uses the FP07 coefficients the data were
  converted with (issue #245).** Under perturb's `fp07.calibrate: true` the
  channel data were on the per-file in-situ fit while the noise model
  converted its counts to kelvin with the `.p` file's embedded `T_0`,
  `beta_1` and `beta_2`. Those are the bead's R(T); the code comments (and
  the #131 m8 entry below) had called them the electronics. The fit's
  coefficients now ride on the base channel's `PFile.channel_info`
  (`in_situ_coefficients`, the configuration's form) into the `T*_dT*`
  attributes of the per-profile files, which chi reads. An order-1 fit has no
  quadratic term, so an embedded `beta_2` no longer survives under it; an
  order-3 fit's `beta_3` is not in the noise model (it has no cubic term), as
  for an `fp07-cal` patched configuration. A channel the fit refused keeps the
  embedded values, consistent with its data. The per-profile variable carries
  `bead_coefficients_source` (the fit's `calibration` tag) and the chi file
  `bead_coefficients_source_<channel>` when the fit supplied them; neither is
  written otherwise. Reading a per-profile file whose data were fitted in
  situ but whose coefficients are the embedded ones (written before this fix)
  logs a WARNING. `pf.config` is left as read: it still matches the
  `configuration_string` the products carry, and editing it would have had
  to reproduce `fp07-cal patch`'s handling of a stale `beta_2` and of the
  legacy `beta` key that `convert_therm` reads before `beta_1`. With
  `calibrate: false`, including `fp07-cal` patched input, every product is
  unchanged: on the SN479 fixture, a config-patched copy of it and two
  SUNRISE 2021 Pelican files, every variable and attribute matched `main`
  except the provenance stamps and `_input_manifest` (which hashes the engine).
  *SUNRISE 2021 Pelican SN194, SR1P_518 and SR1P_520 (order 1; 1511 and 1530
  chi windows):* the observed/model floor over 60-100 Hz in the quietest decile
  (the `fp07-pair tau` rule-7 statistic) went from 0.25 and 0.20 to 1.21 and
  1.50 on T2 (bead T2005), and from 1.11 and 0.76 to 1.22 and 1.45 on T1
  (T1592), whose embedded coefficients are wrong in some files too
  (SR1P_520: `beta_1` −1413.8). The excess left falls with frequency in
  every case (SR1P_520 T2: 1.93, 1.53, 1.39, 1.25 over 60-70 ... 90-100 Hz),
  as resolved signal does in these energetic casts (conjecture: not tested
  against a quieter record). Chi itself barely moved (Method 1): median
  ln(chi_new/chi_old) +0.0002 to +0.0055 per probe and file, 5-95% within
  −0.10 and +0.21; 26 T2 windows gained a finite chi. *Which products move:*
  in 2021 the SUNRISE `.p` configurations were rewritten FILE BY FILE, and
  many of the coefficients are not an FP07's (WaltonSmith SN194 alone:
  `beta_1` from −230435 to 95481). Read from every file's configuration
  against one scanned file's fit per bead (approximate: a bead's fits differ
  by a few percent), the noise model was off by more than 1.25x in T1/T2 of
  5/4 of Pelican SN194's 17 files (T2: 4 of the 5 after the swap), 18/14 of
  Pelican SN412's 78, 14/15 of WaltonSmith SN142's 47 and 18/17 of
  WaltonSmith SN194's 39, from 0.001x to 10^33x. Those deployments need
  reprocessing. In every other `calibrate: true` deployment scanned (a fit of
  one or two files each: SUNRISE 2019 and 2022, ARCTERX 2022-2025, ASTRAL
  2023, CASPER 2017 VMP, Taiwan 2017 VMP and the glider MicroRiders of
  CASPER, Taiwan and Tanaka) every file stays within 0.8-1.25x (medians
  0.85-1.04x), except Tanaka 2015-03 (1.25-1.27x, the Johnson term at 3 °C)
  and Taiwan 2013 husker (T2 1.9x). Not covered: beads no scanned file
  carried (ARCTERX 2023 Wake SN409, part of SN465, 35 CASPER 2015 files), and
  deployments whose configurations were not found on SeaChest (RIOT, the
  ARCTERX 2025 gliders). *Limitation:* the noise model inverts R(T) to first
  order in `beta_2`, close enough (< 2% in power) for every fit scanned but
  two: CASPER 2015 doug's CAS_150 (12-19%) and husker FSA_011, whose order-2
  fit (`beta_1` 115, `beta_2` 145; the parabola's vertex sits at the data)
  puts it 27-336x off at the file's median temperature. That is a fit perturb
  should refuse, left for a follow-up.
- **`fp07-cal` on a VMP against its own JAC CT (#238).** Six defects found
  fitting VMP beads against the instrument's own reference. On three glider
  deployments (osu684 2023/2025, osu685 2025) the coefficients, chosen
  orders and lag-trust decisions are unchanged; the record's held-out scores
  are lower (item 1's fold screen), and the report's T1 − T2 line and the
  figure's stability panel change (item 5).
  1. *The 1 K export gate counted pairs the robust fit had already
     rejected.* One 4.2 K reference event refused SUNRISE 2022 PointSur, fit
     rms 6-7 mK, on both channels. The gate is now judged over every pair but
     the robust loop's outliers (an unidentified fit mispredicts those it kept
     too, so the issue #180 F20 case is still refused); outliers beyond 1 K
     are reported in a `WARNING`, the report and the record's
     `outliers_beyond_gate`. Nor do rejected pairs set the fitted range
     (`T_range`/`L_range`: the report's do-not-extrapolate line, `patch`'s
     validity verdict, the sheet's `valid_T_degC`, the NTC check below) or the
     warm/cold folds that choose the order, from which pairs every order (1-3)
     rejects are left out: three 2 °C reference samples in 13-21 °C water had
     exported `T_range` [2.0, 21.0]. A fit that is not an FP07 curve (T not
     falling as L rises over its fitted range) is now refused at export,
     `error: "not an FP07 curve"`: without a prior, an isothermal synthetic
     had exported β₁ = −1.06×10⁵.
  2. *The lag-trust gate refused every VMP lag.* It asks a peak to fall by
     0.10 across the search, which a glider's ±20 s window shows and a VMP's
     ±1 s cannot. A smaller fall is now trusted when the peak's curvature,
     from the contiguous width around the peak (taken at its grid lower
     bound), implies 0.10 across ±20 s and two interleaved halves of the
     profiles find the peak again within two lag steps. The peak must be
     resolved (at least two steps wide), the only top within 1% of the fall,
     and no wider than a quarter of the search: two lobes, a flat top and a
     one-step bump on the 1/64 s grid are refused. Only searches narrower than
     ±20 s qualify, judged on the configured `max_lag`; a wider one measures
     the fall itself and is judged as before. On eight VMP deployments the
     implied fall is 2.3-49, the peaks 5-12 steps wide, and the halves agree to
     0.1-8 ms, so every lag is trusted and `patch` accepts where it refused all
     eight; a glider's mean-removed ramp (the plateau the gate exists for)
     gives 1.4×10⁻⁴-5.2×10⁻⁴ and stays refused. The record carries
     `lag_resolution` (with any `problem`) when the new rule decides.
  3. *The geometry term crashed on profiles under ~30 dbar* (`joint_fit`
     raised `ValueError`; on SUNRISE 2021 no descent's pairs span more than
     28.3 dbar), stopping the run before a `fit.prior` was consulted. The term
     is now skipped with a note in the report and `geometry_skipped` in the
     record.
  4. *One NaN count turned the rest of the file to NaN* through the single
     pole. The pole now restarts after a gap, blanking five time constants
     while it settles, and every pair whose kernel the gap or the blank
     touches is dropped; the pairs lost are counted (`thermistor_gap`, the
     record's `thermistor_gaps`). Finite input gives the same pairs bit for
     bit as before.
  5. *The reference-free T1 − T2 check used each file's embedded
     coefficients*, which per-file patches had left several K apart at fixed
     L (ARCTERX-2022 Interior: 13 sets in 30 files). Both channels now take
     their coefficients from one source, named in the report line: this run's
     fit when it fitted both, else the embedded sets when each is one set
     across the files, else the check is unavailable with the reason (never a
     fitted T1 against an embedded T2). On ARCTERX-2022 Interior the file-to-file steps in T1 − T2 fall
     from up to 2.5 K to 2.5 mK and its slope from +0.32 to +1.3×10⁻⁴ K/day;
     on osu685 2025 and osu684 2023, whose two beads carry identical nominal
     coefficients, the slope is 5× and 6.5× smaller (the old check read the
     change in water temperature as differential drift). This changes the
     T1 − T2 line of every two-channel report and the stability panel of the
     figure.
  6. *The ±40 s clock search ran against the instrument's own reference*,
     where the offset is zero by construction. `reference.same_clock: auto`
     (new; `true`/`false` force it) skips it when the reference's samples are
     the instrument's own in time (1 µs) and pressure (1 mm). On SUNRISE 2022
     PointSur (77 files) the search took 232 s of 894 s; the run now takes
     666 s, with identical coefficients (it had returned −1.3 µs). The clock
     offset is recorded as 0 with `clock_source`, which says when the zero
     was asserted (`true`) rather than detected: an asserted zero measured
     nothing and is not a value for perturb's `hotel.time_offset`.

  `profiles.direction` (new, default `glide`) selects a VMP's descents. The
  runbook has a VMP section (§8).
- **`fp07-cal fit` reported gradient changes as bead drift in its streaming
  pass.** The per-profile statistics that give a deployment's drift compared
  each profile with the raw CTD temperature, although the calibration was
  fitted with the sensor-geometry term (`dz·dT/dz + τ·w·dT/dz`) removed, so a
  change in stratification over the deployment read as drift. This is the
  issue #180 F08 defect, which had been fixed only in `blocked_offsets`. On
  osu685 (ARCTERX-2025) it had made a bead with no measurable drift read
  +0.049 mK/day (+3.1 mK over 66 days, *p* = 0.02); fixed, it reads +0.022 ±
  0.015 mK/day (*p* = 0.18), and the other bead +0.075 rather than +0.103
  mK/day. `docs/fp07cal/osu685_stability.md` is corrected. Fitted
  coefficients are unchanged.
- **`rsi-tpw prof` on an `rsi-tpw nc` (ATOMIX L1) file now gives usable
  profiles (#235).** It copied only the L1 file's 1-D variables, under their
  ATOMIX names, so every profile lacked the shear, vibration and FP07
  channels and `P`, and `rsi-tpw eps` on one died with `KeyError: 'P'`, while
  `prof` reported success. The L1 file is now read back by inverting
  `rsi-tpw nc`'s own role tables (`convert.l1_native_channels`): stacked rows
  by `sensor_names` (`SHEAR` times the square of the speed it was divided by),
  single-channel variables by the channel the embedded configuration gives
  their role, `PRES_SLOW` as `P`, and the channels the L1 file leaves out as
  copies (`T1` = every n-th `T1_dT1` sample, `P_dP` = `P`) rebuilt as `PFile`
  builds them; the decimal-day time axis snaps back to the sample grid it
  rounds, anchored at `start_time`. RDL bad-buffer spans travel too: the L1
  file's `L1_converted` group gains one attribute, `bad_buffer_spans_by_channel`
  (every channel's spans, the left-out `T1`/`P_dP` included; empty when
  clean), and the profiles carry them as the `.p` route's do. An L1 file
  written before it cannot say: `prof` warns, the profiles carry
  `bad_buffer_spans_unknown`, and epsilon/chi report `bad_buffer_fraction` and
  `interpolated_fraction` as NaN rather than a clean 0. `SHEAR` is restored
  with today's speed, unchanged since v0.3.0, with a warning when the file's
  `software_version` does not show v0.3.0 or later; a NaN in `PRES_SLOW` is
  repaired first, as for profile detection. On the committed SN479 file, the
  repo's full ARCTERX SN479_0006 (2 profiles) and two MicroRider-on-glider
  files, every channel of every profile is bit-identical to the `.p` route's,
  and epsilon and chi are bit-identical too, as they are on SN479 with
  injected dropouts and on SN479_0002, which has real ones; the profiles'
  attributes differ only in the piezo `units` (`1` in the L1 file, `counts`
  from `PFile`) and in the L1 file's `source` and history line, which they
  carry. `rsi-tpw
  eps`/`chi` given the full-record L1 file now stop with "extract its profiles
  first: rsi-tpw prof ..." instead of a bare netCDF4 `AttributeError`.
  `TEMP`'s `sensor_names` (`T1_dT1, T2_dT2`), suspected in the issue of
  naming the gradient channels, is correct and unchanged: it names the
  channel each row copies, and `PFile` keeps the deconvolved fast temperature
  under the pre-emphasized channel's name. The `.p` route is bit-identical
  to before, and so is every variable of the L1 writer's output.
- **An `instruments:` key that matches no input file is no longer silent.**
  perturb looks a file's per-instrument settings up by the raw directory name
  of its `.p` file, and a key that matched none was ignored without a word, so
  `exclude_shear_probes`, `fp07_tau_scale` and `fp07_response` could simply
  not be applied. A key YAML loads as a non-string (`465:` is an int,
  `2017-10-24:` a date), which can never match, is now refused at config load
  with a message to quote it. A string key that matches no input file of the
  run is logged at WARNING at run start and listed in the run summary's
  `notice:` lines, whichever settings it holds. This replaces the
  `exclude`-only notice added with `instruments.<key>.exclude`. None of the
  repo's example recipes was affected: the five with an `instruments:` block
  use quoted keys matching their input directories.
- **One Goodman vibration stack for every entry point (#168).** The adapter
  (`rsi-tpw pipeline`) chose the channels Goodman cleans shear against by
  sensor type, while `load_channels` (`rsi-tpw eps`/`chi`, perturb) and the
  quick-look viewers chose them by name (`^A[xyz]$`), and `nc_to_l1data` took
  a `p_to_L1` file's ACC alone. A piezo named anything else was therefore in
  one stack and not another. Rockland's own Zissou tutorial file `VMP_024.p`
  names its piezos `apx`/`apy`. On it, `rsi-tpw eps` and perturb's epsilon
  stage refused every profile ("No accelerometer channels found (required for
  Goodman)"), while chi and the viewers ran with no Goodman cleaning at all.
  Every route now uses `apx`/`apy`, so chi on such files changes (+8% median
  on that file's first profile). Next to a channel named Ax, such a piezo was dropped
  without a word. `select_vibration_channels` is now the one rule:
  every fast piezo whatever its name, plus the non-piezo channels named
  `^A[xyz]$`. **`load_channels(accel_pattern=...)` now names accelerometers
  only and can no longer exclude a piezo.** A piezo on a slow column is left
  out with a warning and a logged WARNING; the adapter used to crash on one
  in `np.stack`. No change on any instrument we fly (all piezo-only, named
  Ax/Ay): bit-identical on four VMP SN479 files through every route.
- **The v6.1 bad-buffer sentinel scan read the header version from record 0**,
  which ODAS `patch_setupstr` rewrites as v6.0 while leaving the data records
  as they were. It skipped the scan on such files (e.g. the ARCTERX-2023
  Seaspider MR429 set; impact unverified -- three of those files checked
  carry no sentinel runs). It now reads the data records' version.
- **Per-bead FP07 response equalization: `fp07-pair` and
  `instruments.<dir>.fp07_response`.** Two FP07 beads on one probe head
  respond differently, and chi gave both the same model response, so their chi
  disagreed by a factor that trends with `K_max_ratio`. `fp07-pair fit`
  measures each bead's response relative to its partner from the median ratio
  of their gradient spectra (no turbulence model), segmented in time where the
  pair changed, and writes one correction per segment. chi divides each bead's
  observed and noise spectra by its factor, weighted by the bin's S/N, so the
  pair's average stays on the tau model and only the disagreement is removed.
  On ARCTERX-2022 SN 194 / SN 428 (Method 1, 190 profiles) the `K_max_ratio`
  quartile spread of `ln(chi2/chi1)` fell from -0.37 / +0.24 to -0.04 / +0.03,
  the per-window scatter fell from 0.28 / 0.19 to 0.11 / 0.10 (MAD-sd), and
  the ratio landed on the pair's flat gain; the geometric-mean chi moved < 1%. Every
  file is checked against its correction (a mid-band disagreement > 0.3 in ln
  turns it off for that file), a file matching no correction is an error, and
  each chi product records what was applied and why not. Hashed by content:
  chi, chi_binned and chi_combo re-version, epsilon does not. Mutually
  exclusive with `fp07_tau_scale`. `rsi-tpw chi --fp07-response`. Runbook:
  `docs/fp07pair/runbook.md`; design record and acceptance:
  `docs/fp07_pair_response_PLAN.md`.
- **Converters for RSI's own JAC-CT board, `rsijac_t` and `rsijac_c`.** ODAS
  has had them since 2017 (`convert_odas.m`, WID); this package did not, so a
  MicroRider carrying that board kept `JAC_T` and `JAC_C` in raw Q3.28 counts
  (~3e8) under a "No converter for type" warning. perturb runs on, so every
  salinity, density and N² it then derives is garbage. Found on the MR-SE
  SN 240 of SeaExplorer SEA025 (Tanaka et al. 2022). The port matches ODAS
  v4.5.1 on every one of the file's 577,344 samples (to 6e-14), and JAC_T
  sits within 9 mK (median) of the FP07.
  - RSI writes some of these coefficients as arithmetic (`2^28`,
    `10.88/100`), which ODAS `eval`s and `float()` cannot read. A new
    `config_number` parses them with MATLAB's rules: `^` binds tighter than
    a unary sign and is left-associative (`2^3^2` is 64). Other names, hex,
    `**` and non-finite results are refused with a `ValueError` naming the
    channel and key. Every other reader of a channel coefficient now uses it
    too (see Fixed).
  - Like ODAS, `rsijac_t` reads `division_factor` (default 2^28) and ignores
    the `scale_factor` key RSI's SeaExplorer configs carry.
  - Unlike ODAS, a zero divisor (`T_0`, a beta, `cell_constant`,
    `division_factor`) raises instead of returning −273.15 °C or inf.
  - Expression-valued coefficients now reach the `cal_*` provenance; `float()`
    dropped them there.
  - `rsi-tpw nc` files the new types as the CTD (`COND_CTD`/`TEMP_CTD`), not
    as supplementary channels, and `rsi-tpw bench` range-checks `rsijac_t`
    like `jac_t`.
- **The v0.4.0 version DOI** (`10.5281/zenodo.22699141`) in `CITATION.cff`,
  the README and the suggested manuscript wording. It lands one commit behind
  the tag by necessity: Zenodo mints a version DOI only when the GitHub Release
  fires, and its GitHub integration cannot reserve one beforehand
  (zenodo-rdm#831).
- **chi takes epsilon's window layout: `anchor` and the `bbl_*` keys under
  `[chi]`.** chi laid out its own windows, always top-anchored, and refused
  these keys, so a bottom-up product had bottom-anchored epsilon and top-down
  chi. Its deepest window center then sat ~1.2 m above the bed: on SUNRISE
  2021/Pelican/SN194, chi occupancy was 0.0% at 0.62 m against epsilon's 95.7%,
  and Gamma and K_T had no near-bed values. Both paths now take their layout
  from one helper, `scor160.l3.window_groups`. `L3ChiData.dof_spec` is per
  window, because a shorter BBL window averages fewer FFT segments and the DOF
  gates chi's non-detection floor. The defaults (`anchor: top`, no BBL windows)
  are bit-identical to before: 167 variables in 11 epsilon, chi and pipeline
  outputs from `ARCTERX_Thompson_2025_SN479_0006.p` match exactly. The
  canonical config is unchanged too: with the engine fingerprint pinned, all
  308 stage hashes of the 28 example recipes that load on 0.4.0 match. (The
  fingerprint itself re-versions products, as it does for any engine change.)
  The 18 SUNRISE bottom-up recipes under `examples/` now set the chi keys
  their products were built with.

  The layout is now checked when the config loads, in both sections: an
  `anchor` other than `top`/`bottom` and a BBL window shorter than two FFT
  segments are refused (both used to fail per profile at run time, the second
  inside `csd_matrix_batch`, leaving an empty stage directory), and `bbl_*`
  under `anchor: top`, which is ignored, now warns. A `bbl_overlap` or
  `bbl_extent` given in samples beside a `bbl_diss_sec` window now reaches the
  stage hash; it used to be dropped, so changing it reused a stale product.

  **Known limitation, shared with epsilon's BBL windows since #203:** products
  still carry ONE `diss_length` and `dof_spec` per file, so the consumers that
  read them treat a BBL window as a full-length one. The N2/dTdz half-window
  behind Gamma and K_T, chi's `chiLnSigma`, and epsilon's `num_ffts` all use
  the long window's value for the near-bed windows.
- **`--version` on every command, with the git commit.** `rsi-tpw`,
  `perturb`, `perturb-plot`, `perturb-diag`, `scor160-tpw`, `fp07-cal`,
  `fp07-pair`, `mr-clocksync`, `dinkum-hotel` and `erddap-hotel` all print one
  line for a bug report, e.g. `perturb 0.4.0 (git v0.4.0-28-g5f6572b9a1e8)`.
  The release number alone did not identify the code, because `pyproject.toml`
  keeps the last release's number for every commit after it. The commit comes
  from the checkout the package was imported from (editable installs and
  `PYTHONPATH` worktrees, with `uncommitted changes` and a hash of the
  package's `.py` files when a file under `src/odas_tpw/` or
  `pyproject.toml` differs), else from pip's record of a
  `git+` install, else from a stamp the release workflow now writes into the
  PyPI wheel (`0.4.0 (release, git ...)`). The workflow also refuses a tag that
  does not match `project.version`. Anything it cannot establish prints
  `commit unknown` with the reason, so a wheel built elsewhere is never taken
  for the release. The editable install in the main checkout reported 0.3.1
  from metadata frozen at install time; the release is now read from the
  source tree's `pyproject.toml`. A test walks `[project.scripts]`, so a new
  command without `--version` fails CI. Like any code change this one moves
  perturb's engine fingerprint, so a campaign's next run recomputes; later
  edits to the version code will not (`_version_info.py` is excluded from the
  hash). See `docs/installation.md`.
- **Products name the code that made them.** `--version` names the code
  installed now; a product made last month needs the code that wrote it. Every
  NetCDF product (perturb profiles, diss, chi, CTD, binned and combo; rsi-tpw
  L1, profiles, epsilon, chi, L4-L6; hotel files; `fp07-pair` corrections) now
  carries a `software_version` attribute and a `history` line
  `<UTC time>: <step> by microstructure-tpw <version>`, appended to the history
  of the single input it came from (binned and combo files start fresh, keeping
  a `history` configured in `netcdf:`, and list every code that wrote their
  inputs in `input_software_versions`, since they are rebuilt from cached
  products another code state may have made). It replaces the old `Computed with
  microstructure-tpw on ...`, `Pipeline on ...`, `Created by perturb on ...`,
  `L1 converted from ... on ...` and `Profile N extracted on ...` lines, which
  named no version. perturb's run and worker logs, the `patch-config` /
  `fp07-cal patch` banner, `v1to6`'s `translator` (both of which recorded the
  stale installed-metadata version), `fp07-cal`'s `coefficients.json` and
  `mr-clocksync`'s report record it too. A test walks the source for NetCDF
  writers that do not stamp.

### Fixed
- **The `goto` FP07 time constant left out Goto et al.'s speed factor.** Goto
  et al. (2016, sections 2d and 3c) use `tau = tau0 * W^-0.32` with
  `tau0 = 3 ms`; the code used a fixed 3 ms, which is 13% short at their own
  0.64 m/s and 2.1x short at 0.1 m/s. It pairs with `fp07_model: double_pole`,
  which no shipped recipe selects. Chi from a user-run `double_pole` changes
  (tau +15% at 0.65 m/s), and `perturb-diag` now redraws an old `double_pole`
  product's FP07 response with the new tau, not the one its fit used.
  - The `lueck` model (`tau = 10 ms * W^-0.5`, single pole, the default) was
    credited to Lueck et al. (1977), which contains no such formula. Their
    0.15-mm beads barely depended on speed, measured at 0.5 and 1.25 m/s. The
    `-1/2` exponent is Vachon & Lueck's (1984) and Hill's (1987). The 10 ms
    is Sommer et al. (2013, Lueck a co-author), where it is a
    speed-independent double pole, not this single pole x `W^-1/2`
    (corrected 2026-10-28; an earlier version of this entry said the 10 ms
    had no source). The docstrings, `docs/chi_mathematics.md`
    and the bibliography now say so. The model itself is unchanged, but
    note what that means: it sets the FP07 correction in nearly every chi
    product. `fp07_tau_scale` is an optional per-thermistor multiplier
    (default 1, never fitted by the code), and the only shipped recipe that
    sets it is a sensitivity variant (CASPER-West `perturb_M1_tau2.yaml`).
    Changing the model would move every chi product.
  - Corrected while there: Peterson & Fer's `-0.32` is Gregg & Meagher's
    (1980), not Vachon & Lueck's, and ODAS `quick_look.m` implements Vachon &
    Lueck as a double pole with `f_3dB = 25*sqrt(W)` Hz, a response 1.6x
    faster than the `lueck` single pole.
- **perturb's in-pipeline FP07 fit no longer gives up on a glider file for
  reasons unrelated to the data.** On Tanaka 2015-11-13 (Slocum, CTD every
  4.5 s) it calibrated 3 of 12 files. It now calibrates all 12, and each fit
  agrees with the deployment-pooled `fp07-cal` fit to within 0.6% in gradient
  gain (median 0.9999) and 8 mK. Two defects caused it, as re-running with
  each fix reverted in turn shows. Without the first fix, 3 of 12 files are
  fitted; without the second, 11 (T1) and 10 (T2):
  - One NaN in a profile's reference (a hotel gap blanked by `max_gap`)
    passed through the lag filter and silenced the WHOLE profile, counted
    neither as accepted nor as rejected. A profile is now split at its gaps,
    and each finite stretch of at least twice the lag window is scored. A
    profile with no gap is scored exactly as before.
  - The low-pass started from zero, so its output climbed from 0 to the
    thermistor's counts (thousands) over seconds. A profile starting at the top
    of a file, as every glider file that begins mid-dive does, scored that
    ramp. A synthetic probe tracking its reference exactly scored |r| = 0.09.
    The filter now starts at rest on the first sample.
  - **This second fix also changes VMP output, slightly, on a file that
    starts mid-cast.** Of the 29 repo SN 479 files, 26 are bit-identical to
    before. In the other three (0004, 0018, 0029, starting at 128, 82 and
    134 dbar) the first profile begins at sample 0, its lag score rises from
    |r| 0.10-0.32 to 0.93-0.97, and it now enters the fit: one more profile,
    at most 0.56 mK change in temperature, and no change in lag. With the
    start-up fix alone reverted, all 29 are bit-identical, so the gap split
    changes nothing on gap-free profiles.
- **A channel coefficient written as arithmetic, or mistyped, no longer
  becomes a default without a word.** ODAS `eval`s every channel parameter,
  so `coef1 = 1/400` is a legal config. Fourteen converters (poly, voltage,
  piezo, accel, magn, inclxy, inclt, jac_c, jac_t, sbt, sbc, aem1g_a,
  aem1g_d, alec_emc) read it, or a decimal comma such as `0,5`, as 0 or their generic
  default. Every reader of a channel coefficient now goes through one parser,
  `config_number`: the converters and their `cal_*` provenance, the chi noise
  model's thermistor calibration, deconvolution's `diff_gain` (in six
  places), `fp07cal`'s bridge parameters, coefficients and patch check, the
  shear-sheet check, the `diff_gain` and sensor inventories, and the v1 setup
  cross-check. Previously they could read the same value three different
  ways. (Defaults for an absent key still differ where they always did:
  `t_0` is 289.0 K in `convert_therm`, 289.3 K in the noise model, and
  required by `fp07cal`.)
  - Arithmetic is evaluated with MATLAB's rules, including a signed exponent
    (`2^-3^2` is (2^-3)^2 = 0.015625, checked in MATLAB).
  - **Arithmetic outside JAC-CT warns**, naming the channel, key and value,
    once per file. In the configs of 19,064 real `.p` files (about 4,400
    distinct; local copies and SeaChest) the only arithmetic value not
    written by RSI is a typo (below); ODAS evaluates it without a word. The
    JAC-CT converters, where RSI writes `2^28` on purpose, stay silent.
  - A present value that is malformed, NaN or infinite raises a `ValueError`
    naming the channel and key. `ConfigNumberNotFinite` tells NaN/Inf from a
    malformed value, and now reaches callers of the converters too (except
    for shear `sens`, which keeps its own message pointing at `patch-config`).
  - An absent key behaves exactly as before. A blank one (`key =`) is
    treated as absent, with two exceptions that are refused: a blank
    polynomial coefficient, because ODAS drops the term and shifts every
    higher order down one, and a blank pre-emphasis `diff_gain` (`T1_dT1`,
    `P_dP`), because undeconvolved counts are no fallback.
  - A non-positive pre-emphasis `diff_gain` is refused, naming the channel.
    It used to fail inside scipy ("filter critical frequencies must be
    greater than 0") or with a bare `ZeroDivisionError`.
  - `beta_2`/`beta_3 = Inf` is no term, as in ODAS (1/Inf = 0). Rockland's own
    Zissou tutorial file (`VMP_024.p`) writes `beta_2 = Inf`; it was
    unreadable before.
  - A zero divisor (`g` for voltage, `coef1` for accel and magn), a
    non-positive Sea-Bird frequency term, and a non-positive AEM1-G `adc_fs`
    or `adc_bits` now raise instead of yielding inf, NaN or a fabricated
    speed. No real config trips any of these.
  - Whitespace inside an operand (`1 2`, `1e -5`) is refused, as MATLAB
    refuses it. The JAC-CT parser had deleted it and read 12.
  - A numpy float keeps its exact value rather than being re-read from its
    `str()`. A boolean is refused rather than read as 1 or 0.
  - Deliberate deviation: ODAS `deconvolve.m` reads `diff_gain` with
    `str2double`, which turns arithmetic into NaN; here it gets the same value
    the shear converter does. Root-section values (`recsize`) stay plain
    numbers, as in ODAS, and the sensor inventory compares `cal_date` as text
    (`2019-06-26` is valid arithmetic).
  - Checked against every `.p` config on SeaChest: 1,981 distinct configs in
    13,923 files, each channel converted by the old and new code. Output is
    bit-identical except in two places. The 64 JAC-CT files now convert
    (above). **One SUNRISE 2021 file now loads that did not:** Pelican SN 412
    `SR1P2_0003`, whose T1 (probe T2132) has `beta_1 = 3-1315.08`.
    It was refused before, so no SN 412 product contains it; it now reads
    -1312.08 (with the warning above), the same kind of garbage negative
    `beta_1` its 2021 siblings carry as plain numbers. The next SN 412
    rebuild picks it up. Its recipe sets `fp07.calibrate: true`, so a
    successful per-file fit replaces the coefficients; if the fit fails for
    this file, the garbage value reaches the product.
- **`perturb-plot mixing` failed on every bottom-up product.** It merges
  `diss_combo` and `chi_combo` by position and demanded identical bin axes, but
  near the bed epsilon's 1.0 s BBL window reaches bins chi's 2.0 s window does
  not (SUNRISE 2021/Pelican/SN194: 29 bins against 28). The merge now
  reindexes onto the primary product's bins by value, which is exact on a
  shared lattice and leaves the missing bins NaN. It warns on every reindex,
  and refuses when the two are binned at different widths
  (`binning.diss_width` vs `binning.chi_width`), when fewer than two bins
  match, or when the cast counts differ. Existing products need no rebuild.
- **`cutp` / `extract_pfile_segment` stamped segments early.** For
  `start_record > 0` it advanced the record-0 header clock by the nominal
  config `recsize` (1.0 s) per record, but a record lasts `data_words /
  f_clock` on the sampling clock that `PFile` puts every sample on: 1.0001067 s
  for a VMP-250 with 5120-word records at 5119.454 Hz. The error grows with
  the cut point: 53 ms after 500 records, 0.45 s after 4215 (found splitting a
  2023 Hydrographer Bank VMP file at a probe failure). The advance now uses
  the true duration and falls back to `recsize`, with a warning, only when the
  header carries no clock. Segment sample times now match the source within
  the header's 1 ms resolution.

### Changed
- **An in-pipeline FP07 fit against a HOTEL reference is refused unless
  `fp07.allow_hotel_reference: true`.** That fit is per `.p` file, so a file
  the glider CTD did not cover keeps the factory coefficients while its
  neighbors are fitted. In our own products, Taiwan13 husker carries T1
  on factory coefficients in 40 of 58 files. perturb now stops before any
  stage and names both ways forward: `fp07-cal` first (one calibration for
  the deployment), or the switch. The switch is excluded from the output-
  directory hash. The eight shipped glider recipes set it, so they run as
  before.
- **The run summary names every FP07 channel that kept the factory
  coefficients**, per channel, with the file count and the most common kind of
  reason, then the files. That used to be one warning per file in the worker
  logs. A new `notices` list on `PipelineResult` carries it; notices neither
  fail the run nor block the cache. The summary clips each line, so the notices
  are also logged whole, at WARNING. They cover the files processed in that
  run: a cached re-run reports none. `fp07_calibrate` returns a `factory` map
  giving the reason for every channel it did not fit. Its `n_profiles*` counts
  are now of scored segments (one per gap-free profile, so unchanged on a
  VMP).
- **SN 132's `diff_gain = 0.09` is correct (issue #178, confirmed by
  Rockland).** This supersedes two 0.4.0 notes below: that "the
  ladder points the other way", and the `--diff-gain` entry's "real miss ...
  inflating its epsilon by roughly 110x". A 512 Hz sampling rate did not imply
  a ~0.95 gain: Rockland converted SN 132 to a 1024 Hz tidal build in 2021, and
  its analog chain stayed on that rung when it went back to 512 Hz.
  Open-circuit bench noise fits 0.10–0.13 (the same fit reads SN 409's 0.949
  within 2% and SN 465's 0.988 3–14% high), and epsilon computed with 0.09
  matches SN 465's, cast near-simultaneously from a second boat in German
  Channel, at ~0.9–1.2× (95% CIs within ~0.7–1.7×), not the 109× a 0.94 gain
  would give. Rockland has since confirmed 0.09: the tidal build's 1/10 gain
  was never changed back, so SN 132 epsilon needs no gain correction and
  carries no extra gain uncertainty.
  `docs/deployment_procedures.md` no longer says a bench run cannot check
  `diff_gain` — the open-circuit noise spectrum estimates it against
  `noise_shearchannel` — and a test probe is documented as reading the config's
  `T_0`, which is ~16 °C only with nominal coefficients.

## [0.4.0] - 2026-09-10

### Added
- **`docs/validation_and_limitations.md`** — one place to point a reviewer or
  a co-author at. States what the epsilon estimator was validated against
  (ATOMIX, 1416 spectra, re-run at this release commit), what that check does
  *not* cover (agreement with a consensus reference is not agreement with
  truth; the calibration constants, not the estimator, usually dominate), and
  why chi must be reported as a **relative** field with an absolute systematic
  of order **1.5x** (#179) rather than at the ±25% its tuned cross-probe
  spread suggests. Also records that a chi `fom` of 1.001 sits on top of a
  factor-of-two structured spectral residual (#191), so `fom` alone is not
  evidence of a good fit. Includes suggested manuscript wording.
- **`docs/atomix_benchmark.md`: a re-verification table stamped with the
  release commit.** The benchmark numbers previously carried no provenance,
  and the generated report in the (untracked) `AtomixData/` tree had drifted
  four months behind the code. The comparison is now re-run at release and
  the command to regenerate it is written down next to the result.

### Changed (BREAKING)
- **`hotel.max_gap` is now required, and `hotel.extrapolate` defaults to
  `false`.** The hotel merge used to interpolate across arbitrary gaps and
  edge-hold outside coverage, with no way to say otherwise. On an instrument
  whose CTD ran on only some profiles that produces a smooth *fabricated* ramp
  between real samples hours apart, and every consumer — `ct`, `ctd`,
  `stratification`, `salinity: "measured"`, `epsilon.T_source` — reads it as
  data. It also silently undid gap control applied upstream: a builder that
  NaN-marks its own dropouts (`dinkum-hotel`'s `projection.max_gap`) had that
  erased, because the loader drops non-finite samples and *then* interpolates
  across the hole.

  There is no safe default for the limit — the right value is the sensor's own
  rate, tens of seconds for a 1 Hz CTD and minutes for a flight-state variable
  — so omitting it now raises rather than guessing. Both settings accept
  per-channel overrides.

  **Migration:** add `max_gap: <seconds>` to your `hotel:` block (30 is a good
  starting point for a 1 Hz CTD). To keep the old behaviour exactly — to
  regenerate a historical result, say — write `max_gap: "unlimited"` and
  `extrapolate: true`. Any channel that then bridges a gap wider than 10× its
  own median sample interval warns, naming the channel and the fabricated
  fraction.

  The gate is honoured downstream, not just at the merge: the resolved
  `max_gap` travels with each merged channel (a `hotel_max_gap` attr in the
  per-profile NetCDFs), and the stratification / viscosity-salinity refills
  that interp-fill non-finite samples refuse to rule across a wider hole
  (viscosity falls back to fixed 35 PSU there, with a warning). Epsilon's
  per-window temperature now averages the finite samples only, so a window
  merely touching a NaN-ed gap keeps a real temperature; the 10 °C fallback
  (still warned about) applies only to windows with no finite sample at all.
  A missing `max_gap` fails once at config-load time, not once per file in
  the worker pool.

### Changed
- **A perturb stage that produced an EMPTY product now fails the run.**
  `stage_errors` has always meant "a product the config asked for was not
  written", but only a stage that raised *as a whole* ever recorded one. A
  fault that hits every profile identically arrives as N file errors, absorbed
  one at a time — so the run created an empty `chi_NN/`, reported
  `0 stage failure(s)` and `N file error(s)`, and exited **0**. That is how the
  `_compute_chi()` regression below cost 1330 profiles of chi on ARCTERX-2022
  while reporting success.

  `run_pipeline` now records a stage error when a per-profile stage the config
  enabled holds no `.nc` files **while profiles exist**. Two things that make
  it safe rather than noisy:

  - **Gated on profiles existing.** Zero profiles is a legitimate outcome — a
    corpus of nothing but deck files — and diss/chi then having nothing to do
    is correct, not a failure.
  - **Counted from the product directory, never from the per-file results.**
    On an incremental re-run every file is served from cache and returns no
    paths while the products sit complete on disk; counting results would fail
    every cached re-run. The directory is what "the product exists" means.

  The message distinguishes empty-with-file-errors (something failed on every
  profile) from empty-with-none (the stage produced nothing without failing) —
  different bugs. `ctd` is not covered: it derives from the `.p` file rather
  than from profiles, so it needs a gate of its own.

  This is the class of failure, not the one instance: any per-profile
  exception — a missing therm channel, a bad speed source — reproduced the
  same silent-empty-product before this.

### Fixed
- **perturb `chi` produced nothing at all, on every profile.** `chi.mixing_use_qc_fallback`
  was added as a config key but not to the list that filters the `chi:` section
  before it is splatted into `_compute_chi(**kwargs)`, so every profile raised
  `TypeError: _compute_chi() got an unexpected keyword argument`. The
  per-profile handler logs that and continues, so the run reported "N file
  errors", wrote an **empty** `chi_NN/` directory, and still exited reporting
  stage success — on ARCTERX-2022 that is 1330 profiles and zero chi files.

  Both splat filters (epsilon's and chi's) are now module-level constants
  (`_EPSILON_KWARG_EXCLUDE`, `_CHI_KWARG_EXCLUDE`) and
  `tests/test_kwarg_splat_guard.py` asserts that what survives them is a subset
  of the callee's real signature, for both stages. The failure mode is silent
  product loss rather than a traceback, so it needs a test rather than care.
  Epsilon was checked and was not leaking.
- **`perturb/fp07_cal.py` no longer emits confident, wrong coefficients.** Five
  defects, each a way the per-file in-situ calibration could fit something that
  was not a calibration:
  - It carried its **own copy of `ln(R_T/R_0)`** whose defaults disagreed with
    the reader's (`e_b` 0 vs 0.68, `g` 1 vs 6.0, `adc_fs` 5 vs 4.096), so a
    config missing any of them produced coefficients that did not reproduce.
    The bridge algebra now comes from `fp07cal.logr` — the same code the reader
    is verified against — and a missing parameter is refused, not defaulted.
  - It computed a **lag correlation and never compared it to anything**. A
    reference fabricated by interpolating across a gap yielded a confident lag
    at r ≈ 0.02 and was folded into the median. Profiles below `fp07.min_corr`
    (default 0.5) are now dropped, and a channel with none above it keeps the
    factory calibration rather than being fitted against noise.
  - Its **non-JAC low-pass cutoff was `fs/3`** — ~21 Hz for a 1 Hz reference on
    a 64 Hz grid, i.e. no filtering at all, leaving the regressor carrying
    bandwidth the reference cannot see and attenuating the fitted `beta_1`. The
    cutoff is now the reference's own Nyquist. The hotel merge records each
    channel's native sample interval (`pf.hotel_native_dt`) and the pipeline
    passes it through, so this engages for every `hotel.interpolation` kind;
    `fp07.reference_interval` overrides it. Inference from the merged array
    (linear-merge knots, per finite run, robust threshold) remains only as a
    fallback, and warns when it cannot tell and no filtering is applied.
  - It fitted a **raw Vandermonde in `L`**, badly conditioned over a narrow
    temperature range. The fit is now centred and scaled, and composed back
    exactly, so the emitted coefficients still reproduce in the reader.
  - **Clipped samples were regressed silently.** A sample on the ±0.6 bridge
    rail carries a wrong `L` and looks ordinary downstream; they are now
    excluded, and a channel more than 5% railed is refused outright.

  Also removes the unreachable non-therm branch of `_compute_RT_R0`
  (`_find_fp07_channels` only admits therm channels), which existed solely to
  be covered and carried the default mismatch. The module docstring now says
  plainly that this is a per-file calibration and points at the
  deployment-scoped `fp07-cal` CLI for a sparse reference.

### Changed
- **`instruments.<SN>.*` is now hashed only into the stages that consume it.**
  `fp07_tau_scale` is read solely by the chi computation, but the whole
  `instruments` block versioned the **diss** directory, so setting a τ forced a
  full epsilon recompute that could not change a single value — verified
  bit-identical over 34 820 `epsilon`/`FM`/`fom`/`var_resolved` values on
  ARCTERX-2022. It now versions chi only; `exclude_shear_probes`, which reaches
  diss directly and chi transitively through Method 1's `epsilonMean`, still
  versions both. An instrument whose settings are all irrelevant to a stage is
  dropped from that stage's hash entirely, so a config carrying only a chi-side
  knob hashes identically to one with no `instruments` at all.

  **Effect on existing outputs:** a config that set `fp07_tau_scale` will now
  resolve to the diss directory it would have used *without* it — reusing that
  output instead of the one the old hash minted. The two are bit-identical, so
  this reuses work rather than discarding it. Configs using only
  `exclude_shear_probes` are unaffected.

- **The applied FP07 τ multipliers are recorded on the chi product** as
  `fp07_tau_scale_<channel>` plus `fp07_tau_model`. They are deployment-specific
  fits, not derivable from the `.p` file, so without this the number behind a
  chi file was unrecoverable from the product alone. Written only for probes
  actually scaled, so an untouched file carries no misleading `1.0`.

### Removed
- **The `pyturb` compatibility layer (`src/odas_tpw/pyturb/`, `pyturb-cli`).**
  A re-implementation of Jesse Cusack's standalone
  [pyturb](https://github.com/oceancascades/pyturb) on top of our readers,
  hosted here so results could be compared. Nobody was using it, upstream has
  moved on, and it carried the last two open findings from the issue #180
  review (F11, F12) — for code on a path no product depends on.

  Removed with its tests, its `docs/pyturb/` comparison report, and the
  `scripts/compare_pyturb.py` / `scripts/generate_comparison_md.py` pair that
  shelled out to the deleted entry point. The *ideas* worth keeping were ported
  into `rsi/` instead (see Added, above).

  `docs/perturb/pyturb_comparison.md` stays: it compares upstream pyturb against
  our main `rsi`/`perturb`/`scor160` packages, which is a live question.

  **Migration:** none for pipeline users — `rsi-tpw`, `perturb` and `scor160-tpw`
  are untouched. Anyone invoking `pyturb-cli` should use upstream pyturb
  directly, or `rsi-tpw eps`.

### Added
- **Plausible-range checks on shear calibration coefficients.** The strict
  parse from issue #180 (F06) fails closed on a coefficient that is not a
  number, but it cannot see one that parses cleanly and is still impossible.
  The canonical case is the un-filled `sens = 1.0` placeholder: it converted
  without a murmur and scaled epsilon by roughly 200x. `sens` outside
  [0.03, 0.15] V*s/m now warns.

  The bounds are checked against our own corpus rather than adopted from the
  vendor sheet: over **15 998 channel-configs** (ARCTERX, SUNRISE, RIOT,
  CASPER, Taiwan, ASTRAL, Keck, goflow) `sens` spans 0.041-0.123, so the window
  brackets every probe we have deployed and fires on none of them.

  `diff_gain` gets a deliberately much wider bound ([0.001, 10]), because that
  distribution turned out to be **bimodal** — and measuring *why* changed the
  design. The low band is one rung of a Rockland design ladder: for
  higher-energy environments they double the sampling frequency (more of the
  Nasmyth spectrum) and drop the differential gain 10x to avoid saturating the
  differentiator — 512 Hz/~0.95, 1024 Hz/~0.095, 2048 Hz/~0.0095. The
  same-model comparison isolates it: `MR1000RDL-EM` reads 0.927/0.941 on the
  512 Hz gliders SN 433/435 and 0.099/0.094 on the 1024 Hz SN 429.

  So a tight band would flag a seventh of every shear channel we own; and
  `convert_shear` sees only the channel config, never the rate, so it cannot
  pick the right rung itself.

  **The floor is 0.001, not the 0.01 the data alone suggests.** We own no
  2048 Hz instruments, so ~0.0095 appears nowhere in the corpus — and a floor
  fitted to what we happen to have would have warned on every channel of the
  first one we read. Absence from our files is not evidence of non-existence.

  This takes no position on SN 132 (issue #178) — if anything the ladder points
  the other way. Over all 217 of its files, unanimously `fs_fast` = 511.9454 Hz
  on a 10-column `vmp-250-IR`, so it belongs on the ~0.95 rung with
  SN 412/465/479; yet it carries 0.09, the 1024 Hz rung's value, identically on
  both channels where every measured pair differs. Separating a real low-gain
  differentiator from a transcription error needs an instrument-keyed
  comparison, which is what `rsi-tpw sensors --diff-gain` is for.

- **`cal_<key>` provenance attributes on every converted L1 variable.** An L1
  file is read years later without the config that produced it, and a shear
  record is meaningless without the `sens` and `diff_gain` that scaled it.
  `SHEAR` now carries `cal_sens = [0.1075, 0.113]` and
  `cal_diff_gain = [0.954, 0.933]`, as arrays parallel to `sensor_names` so
  probe 0's coefficient stays with probe 0. An audit can now answer "which
  sensitivity is baked into this epsilon?" from the product, rather than by
  re-deriving it from a `.p` file that `fp07-cal` or `patch-config` may since
  have rewritten.

  Recorded **by observation**: `channels.CalRecorder` notes which keys each
  converter actually reads, so the attributes cannot drift from the conversion
  the way a hand-maintained per-type table would. Keys probed but absent are
  not recorded (`convert_poly` walking `coef0..coef9` is a loop exit, not
  provenance), and a pass-through converter that consumes no coefficients
  honestly publishes none.

  The prefix is required: this covers every variable, and `TEMP_CTD`'s JAC
  coefficients are literally named `a`-`f`. The older bare-name therm attrs on
  *per-profile* files (#131 m8, read by `chi_io`) are unchanged — that
  mechanism is confined to gradient channels where the names do not clash.

  Numerically inert: the sample-exact `v6_golden_converted.npz` regression
  passes unchanged. See
  [docs/rsi-tpw/calibration_provenance.md](docs/rsi-tpw/calibration_provenance.md).
- **`rsi-tpw sensors --diff-gain`: a differential-gain audit keyed by
  INSTRUMENT.** `diff_gain` belongs to an instrument's amplifier chain, not to
  the probe screwed into it, which made it invisible to the sensor inventory in
  two ways: that tool is keyed on *sensor* serial, so a shear `diff_gain` was
  attributed to whichever probe was installed, and it deliberately drops the
  `X_dX` pre-emphasized channels (`T1_dT1`, `P_dP`) as the same physical sensor
  as their base — so the thermistor and pressure gains were never reported at
  all.

  The leverage is the shear sensitivity's: shear goes as
  `1/(diff_gain*sens)`, so **epsilon goes as `(diff_gain*sens)^-2` and chi as
  `diff_gain^-2`**. A 2% gain error is 4% in both.

  It exists because of a real miss. On ARCTERX-2023 one VMP carried
  `diff_gain = 0.09` on *both* shear channels where its three siblings in the
  same cruise carried 0.92-0.99 — inflating its epsilon by roughly 110x — and
  nothing in the chain noticed. The audit now flags it automatically.

  Outliers are called out relative to the **fleet median for the same channel
  name** (>2x), not against a hardcoded band: the plausible range differs by
  channel class (differentiator channels near 1, `P_dP` near 20) and hardcoding
  either would be a guess. The median takes one vote per *instrument*, not per
  file, so a thousand files from one unit cannot drag it, and flagging needs at
  least three instruments rather than inventing an outlier from a sample of one.

  Changes over time are **reported with dates, not flagged**: gains follow the
  electronics, so a Persistor CF2 VMP upgraded to RDL hardware legitimately
  changes them. (The `model` string can lag the hardware — SN 142's 2021 files
  still read `vmp-250` while already carrying its RDL gains.)

  `--diff-gain-csv` writes the per-(file, channel) table; `--diff-gain-strict`
  exits **4** on an outlier, distinct from `--cal-strict`'s 3.

- **`epsilon.spectral_qc`: the ATOMIX-style per-probe cut epsilon was missing.**
  `chi.spectral_qc` has shipped and defaulted `true` for some time; epsilon had
  only `fom_max`, which thresholds a *different statistic*. This adds bits 1
  (`FM > FM_max`) and 16 (`var_resolved < min`, **`method == 0` only**) of
  `scor160.l4._compute_flags`, applied per-probe before `mk_epsilon_mean` so a
  failing probe leaves the geometric mean individually.

  Named `spectral_qc`, not `atomix_qc`, because it **cannot** be the full flag
  set: bits 2 and 8 need despike diagnostics the perturb diss product does not
  carry. Claiming conformance we do not have would be worse than the gap.

  Three things that are easy to get wrong and are pinned by tests:
  - **The `method` gate on bit 16 is not optional.** An ISR fit never
    integrates the dissipation range, so a low resolved fraction is expected
    rather than diagnostic, and ATOMIX exempts it. Ungated on ARCTERX-2022 the
    criterion rejects 3.51% of probe-windows instead of 0.18%. With no `method`
    variable the criterion is skipped with a warning, never guessed.
  - **`FM` is not `fom`.** ATOMIX's 1.15 was written for the MAD-based `FM`;
    `fom_max` thresholds the variance-ratio `fom`. They are only weakly related
    (r ~ -0.35 on ARCTERX-2022). Relatedly, `fom_max`'s docs now carry the
    warning that on a corpus with ISR estimates a low value cuts them *as a
    class* — on ARCTERX-2022 every window with `fom >= 1.15` was an ISR fit.
  - **A window where no probe survives is dropped (NaN), not backfilled** —
    matching rsi's `_compute_epsi_final` and deliberately unlike
    `chi.spectral_qc`. A finite-but-wrong epsilon rescales Method-1 chi roughly
    linearly while the chi `fom` stays ~1, so nothing downstream can reject it.

  Provenance is written onto the diss product (`spectral_qc_*`), including
  `spectral_qc_rejected_fraction`. Note `spectral_qc_n_pair_windows_flagged`
  counts *windows* while the other two counts are *(probe, segment) cells*, and
  under `flag_only` it records a flag rather than a cut.

  Default **`false`** while `chi.spectral_qc` defaults `true` — a deliberate,
  documented asymmetry. On ARCTERX-2022 enabling it cuts 19.99% of
  probe-windows (bit 1 19.92%, bit 16 0.18%), removes 47% of the top epsilon
  decile and shifts the median epsilon by 1.14x. That is a reshaping of the
  distribution, not a tidy-up, so it is opt-in and the size of the cut is
  written into the product (`spectral_qc_*` attributes, including
  `spectral_qc_rejected_fraction`) rather than left in the documentation.

- **`epsilon.pair_policy`: the two-probe case both existing rules decline.**
  `mk_epsilon_mean`'s outlier rule needs `n_probes >= 3` — with two probes
  neither is identifiable as the outlier, and always dropping the maximum would
  bias `epsilonMean` low — so on a 2-probe VMP, the common configuration, there
  was no inter-probe consistency check at all. ATOMIX bit 4 *does* act on a
  pair, and keeps the minimum. Both are defensible, so the choice is exposed:
  `keep_both` (default, bit-identical to today), `drop_high` (ATOMIX's action,
  which makes a low junk probe authoritative — a factor of 50 on the test
  fixture), or `flag_only`. Windows with >= 3 probes stay with
  `mk_epsilon_mean`, so two rules never contest the same window. Threshold and
  sigma_ln come from `processing.probe_consistency`, shared with the
  cross-probe consistency log added in #174, so gate and diagnostic always
  describe the same statistic. On ARCTERX-2022, 6.40% of windows exceed the
  threshold, so the policy is not academic.

- **`instruments.<SN>.fp07_tau_scale`: a per-thermistor FP07 time constant.**
  `tau` previously came from a single model (`fp07_tau`: `lueck` for
  `single_pole`, `goto` for `double_pole`) and was shared by every thermistor
  on an instrument. Two beads on one probe head can have materially different
  response, and that assumption biases them apart: on ARCTERX-2022 the two
  FP07s disagreed on chi by 1.77x (SN 194) and 0.72x (SN 428) — in opposite
  directions, persistently, over 1330 profiles.

  The disagreement is a **response** problem, not a flat gain error, and the
  diagnostic is that it varies with `K_max_ratio` (`K_max/kB`): SN 194 ran
  2.27 → 1.54 and SN 428 0.59 → 0.80 across quartiles, both converging toward
  1 as more of the Batchelor rolloff is resolved rather than extrapolated. A
  flat gain error would be quartile-independent. Fitting one tau per bead
  collapsed both units to 1.03x, with the quartile spread falling from
  1.47x/1.37x to 1.13x/1.11x.

  The config value multiplies the tau *model* rather than replacing it, so the
  model's speed dependence survives — a scale factor on `tau(U)` is what the
  data actually constrain. Keys take the bead name (`T1`) or the gradient
  channel (`T1_dT1`); an unmatched key raises rather than being ignored,
  because a silently-uncorrected probe would bias chi by the square of the
  response error with no downstream symptom. Omitted, every probe stays on the
  model and results are bit-identical to before.

  Internally `L3ChiData.tau0` and `.H2` are now emitted per thermistor —
  `(N_GRADT, N_SPECTRA)` and `(N_GRADT, N_SPECTRA, N_FREQ)`. The previous
  shared-across-probes layout is still accepted on input; read either through
  the new `L3ChiData.tau0_for(ci, j)` / `.H2_for(ci, j)` accessors rather than
  indexing directly. Note `fom` cannot arbitrate between two beads (it was
  ~1.000 for both on both units), and fitting tau makes the beads agree with
  each other without establishing the absolute chi level. See
  `docs/perturb/configuration.md`.
- **`mr-clocksync`: per-file clock offset and rate for a MicroRider, from the
  surface-wave band.** An MR clock jumps between `.p` files and runs at the
  wrong rate, so a single per-deployment offset is wrong by construction. Given
  a reference pressure record on a trusted clock — a MATLAB, NetCDF or `.p`
  source, so that the trusted clock can itself be *chained and verified*
  (CTD → ADCP → MR → MR) — the shared surface-wave signal recovers
  `offset + rate * (t - t0)` for each file, with a real error bar. On the
  ARCTERX-2023 Bank Seaspider mooring the MR clocks were 43–149 s out and drifted
  ~33 s/day; all 40 files solved to a median σ of 22 ms.

  Four subcommands: `probe` (is there a wave at all — always run it first),
  `extract` (`.p` trees to a resumable `.npz` cache, the only slow step),
  `solve` (`clock_offsets.csv` + `clock_report.txt`) and `report`. The output
  feeds perturb's `hotel.time_offset`.

  The estimator is a band-passed cross-correlation *envelope* peak refined by
  cross-spectral phase slope, not `xcorr`-and-`argmax`: the envelope picks the
  right side lobe of a narrowband swell, the zero-phase filter contributes no
  group delay of its own to the quantity being measured, and the phase slope
  gives sub-sample precision plus a coherence-derived σ. Gates are on peak
  sharpness, in-band amplitude and coherence — **never** on `r`, which a tidal
  ramp drives to 1.000000 at every lag. A peak on the search boundary is
  refused rather than reported.
- **RDL bad-buffer dropouts are now repaired or rejected in ε and χ.** v6.1+
  files substitute `-32753` for individual missing samples (TN-051 rev.
  2026-01-12 §3.2); `PFile` already detected them, but the affected samples
  still entered the spectra as ordinary physical values. A run is now repaired
  only when **both** tests pass, and rejects its windows otherwise. First, the
  channel must be consumed as a slowly-varying **scalar** — speed, pressure,
  reference T/C. **Shear, vibration and the FP07 thermistors are never
  interpolated**, however short the gap: their high-frequency content *is* the
  measurement, so a linear bridge substitutes a smooth ramp for real variance
  inside the band being fitted, fabricating a spectrum rather than repairing
  one. Second, the gap must be **≤ 0.25 s**; a longer one has no information to
  interpolate across at any scale, and the threshold is where the observed
  dropouts separate, since the RDL's fixed 64-sample buffer loss is 0.125 s on a
  fast channel but 1.0 s on a slow one. A window is rejected anyway once more
  than 5% of it has been interpolated. Both are
  published per probe × time as `bad_buffer_fraction` and
  `interpolated_fraction`, with `mask_bad_buffers=False` to opt out. The
  dependency scoping is the other half: a `sh1`
  dropout rejects only probe 1; an accelerometer dropout rejects every probe but
  only under Goodman; and a **`U_EM` dropout rejects nothing when the speed came
  from the flight model**, since `U_EM` is not a flight-model input (pressure
  still is, through `W_slow`). `prepare_profiles` stamps
  `metadata["speed_channels"]` with the channels the selected speed actually
  read, rather than letting the mask builder guess from a method name. On the χ
  side the rejection lands before `chi_final` is formed, so a contaminated probe
  cannot be averaged back in. Masks survive the per-profile NetCDF, so the
  `prof → eps/chi` and perturb routes behave like the direct `.p` route. Null
  test: ε, χ, `fom` and `epsilon_T` are bit-identical to the pre-masking code on
  the SN479 set, whose only dropouts are on `DO_T` (feeds nothing). See
  `docs/rsi-tpw/odas_file_format.md`.

### Fixed
- **A 32-bit channel's bad buffers were reported once instead of twice.**
  `bad_buffer_report` was keyed on the matrix address, so both words of a 2-id
  channel mapped to the same channel name and the second silently overwrote the
  first. Findings are now grouped per channel and the spans unioned. `spans` is
  also always populated (previously `None` whenever an address occupied several
  matrix cells) and is aligned with the samples extraction actually keeps, which
  is what makes it usable as a mask.
- **`perturb run` no longer exits 0 after failing to write a product.** A combo
  stage that raised was caught, logged into that stage's own `combo.log`, and
  the run went on to log "Pipeline complete." and exit 0 — and since `perturb`
  installs **no console handler without `--stdout`**, the terminal showed
  nothing at all. The two together made a missing `combo.nc` indistinguishable
  from a clean run: observed on ARCTERX-2023-Wake, where a transient `OSError`
  reading `chi_binned_00/binned.nc` off an SMB mount lost `chi_combo_00/combo.nc`
  silently, surfacing only later as `perturb-plot mixing: mixing combo not
  found`. `run_pipeline` now returns a `PipelineResult` separating **stage**
  failures (a product the config asked for was not written — combo, ctd combo,
  merge) from **file** failures the pipeline absorbs by design (a startup
  fragment with no data records, one profile whose fit blew up — trim, and
  everything `process_file` records in `result["errors"]`). The CLI prints a
  one-line-per-failure summary to **stderr unconditionally**, names the run log,
  and **exits 1 on any stage failure**; file-level errors are reported but still
  exit 0, since a campaign with one bad `.p` among 200 is a normal successful run
  and failing on those would train the operator to ignore the status entirely.
  Both buckets survive the incremental cache — a file with recorded errors is
  not cached, and a failed stage left no output to skip — so a repeat run
  reproduces the status instead of laundering it into a clean exit.
  `run_trim`/`run_merge` take an optional `errors` out-parameter (their
  `list[Path]` returns are consumed directly by the `trim`/`merge` subcommands,
  so the return type is unchanged).
- **Flight-model glide angle now ADDS the angle of attack** (issue #131 M7).
  `speed.method: "flight"` computed `U = |W| / sin(|pitch| − aoa)`; steady-glide
  force balance (Merckelbach et al. 2010) and the ODAS reference
  (`odas_p2mat.m`: `glide_angle = abs(Incl_Y) + aoa`) both make the glide path
  STEEPER than the pitch attitude, `U = |W| / sin(|pitch| + aoa)`. The
  subtraction inflated U by sin(|pitch|+aoa)/sin(|pitch|−aoa) — 1.24× at a
  typical 26° Slocum pitch — and, through epsilon's ~U⁻⁴ speed leverage, biased
  flight-method epsilon ~2.4× LOW (up to ~5× at 15° pitch; ~1.55× on steep 44°
  SL685 climbs). On `MR/AIOP2_SL685_0450.p` the corrected flight speed
  (0.397 m/s) now sits within 10% of the independent `U_EM` flowmeter
  (0.428 m/s). Only `speed.method: "flight"` output is affected; `pressure`,
  `em`, and `constant` are unchanged. A new cross-check warns whenever the
  flight speed and a present `U_EM` channel disagree by more than 20% (median),
  since a bad `aoa_deg`, a mis-picked pitch axis, or a stale EM calibration all
  leak into epsilon the same way. `min_pitch_deg` now gates on the pitch
  attitude itself (`|pitch| < min_pitch_deg` → NaN), not the aoa-shifted path
  angle — the steady-glide model is invalid near dive/climb inflections
  regardless of aoa; the effective default gate moves from |pitch| > 8° to
  |pitch| ≥ 5°. The path angle is clamped at 90° (railed inclinometers).
  Stale `sin(|pitch|−aoa)·cos|roll|` formulas in the perturb template, the
  ARCTERX example config, `--speed-method` help, and
  docs/perturb/configuration.md were corrected to match the code.

### Changed
- **Γ (mixing efficiency) now plots on a diverging colormap anchored at 0.2**
  in `perturb-plot mixing`/`overview` (and the `epsilon`/`chi`/`profiles`
  products, which share the same tables). Γ used to use the sequential cmocean
  `matter` on a plain `LogNorm`, so the canonical Osborn (1980) value the whole
  χ→K_ρ conversion assumes had no visual position at all. It now uses `balance`
  (two hues, neutral midpoint) with the robust 1/99% range symmetrized **in log
  space** about 0.2 — equal color distance ⇔ equal *ratio* from the anchor,
  which is the only reading that makes sense for a quantity spanning decades,
  and the reason this is not a `TwoSlopeNorm` (linear within each half, it would
  crush the entire sub-0.2 range into a sliver of the ramp). Staying on a plain
  `LogNorm` also keeps the colorbar's decade ticks. A thin rule marks the anchor
  on the colorbar (`layout.draw_cbar_anchor`, placed through the norm so it is
  norm-shape agnostic) and the label reads `Γ (diverging about 0.2)`. An
  explicit `--clim Gamma MIN MAX` / `--vmin` / `--vmax` wins outright and drops
  the centering, matching what the existing zero-centered diverging branch
  already does. New anchors go in `profiles._DIVERGING_LOG`.

### Added
- **Real-glider MicroRider end-to-end fixture + test** (issue #131 m11) —
  `tests/data/MR_SL685_climb.p`, a 150 s / 1.4 MB climb segment (481 → 429 dbar
  at ~0.35 dbar/s, EM speed ~0.5 m/s) cut at record boundaries from
  `MR/AIOP2_SL685_0450.p` (MR1000RDL-EM SN 435 on Slocum osu685, ARCTERX IOP2),
  and `tests/test_mr_e2e.py`: the first test to run the batch machinery
  (PFile → vehicle resolution → glide profile detection → ε → χ Method 2)
  against real MicroRider-on-glider data instead of mocks. Direction, `W_min`,
  and speed (median `|U_EM|`) are passed explicitly so the test is independent
  of vehicle-resolved defaults.
- Docs drift fixes (issue #131 m13): `docs/perturb/cli.md` documents
  `perturb run --force`; `docs/rsi-tpw/cli.md` adds the missing `sensors` row
  to the subcommand table and a `rsi-tpw ml` reference section.
- **Cross-probe consistency diagnostics** (#131) — with two or more shear
  probes (or FP07s), every per-profile epsilon/chi dataset now carries
  per-pair global attrs `probe_ratio_pairs` / `probe_ratio_median` /
  `n_ratio_windows` / `probe_ratio_z` (`chi_`-prefixed on the chi product):
  the median first/second-probe ratio over the windows where both are finite,
  plus the significance z of the median ln-ratio given the Lueck (2022)
  per-window `sigma_ln`. A two-tier `logging` warning flags persistent
  inter-probe disagreement — statistical (z > 3, ≥ 20 windows) and practical
  (median ratio beyond 1.8× either way, ≥ 10 windows) — because per-window QC
  (fom/FM) cannot see a persistent systematic offset (vmp142's pairs disagreed
  1.8×/2.2× with clean per-window QC; CAS_080.P has sh1 ≈ 1000× sh2 with
  fom ≈ 1). Observational only, nothing is auto-dropped; the attrs do **not**
  survive depth binning / combining (attrs are rebuilt from a schema there) —
  read them from the per-profile files. Shared helper in
  `processing/probe_consistency.py`; see docs/rsi-tpw/pipeline.md.
- **Calibration staleness guard** for `rsi-tpw sensors --cal-dir` (#131 m2) —
  each sheet's "Recommended re-calibration" date is parsed onto its
  calibration point, and observations governed by a calibration past that
  date are annotated `[cal N months old at use; recal was recommended by
  YYYY-MM-DD — verify no newer sheet exists]`. Sheets without the line fall
  back to `--cal-max-age-months` (default **12**, Rockland's actual
  recommendation — the flag only sets the fallback). The mismatch summary
  reports the stale count even when no sensitivities mismatch. The check is
  only as good as the sheets directory (completeness assumption documented in
  docs/rsi-tpw/sensors.md).
- **Strict exit codes** (#131 m3) — `rsi-tpw bench` now exits **1** when any
  file failed to evaluate (per-file errors still don't abort the batch;
  checklist FAIL/REVIEW verdicts don't affect the code), and
  `rsi-tpw sensors --cal-strict` exits **3** (distinct from the scan-failure
  code 1) when the `--cal-dir` check finds sensitivity mismatches outside
  `--cal-tol`; default behavior stays report-only. `--cal-strict` without
  `--cal-dir` is an error. Both flags are available on `rsi-tpw sensors` and
  `python -m odas_tpw.rsi.sensor_inventory`.
- **Old-format (2013-2017 CASPER-era) MicroRider config dialects** (#131 m1,
  m5, m6, m7) — pre-2017 configs now inventory and convert faithfully:
  `serial_num` is honored wherever the instrument SN is read (`summary`,
  NetCDF `instrument_sn`, epsilon/chi metadata, `rsi-tpw sensors` platform SN);
  `[cruise info]` — and any unknown section — is kept by `parse_config`
  (section names normalized: lower-cased, internal whitespace folded to `_`;
  `config-patch`/`patch-template` agree on the normalized name); channels
  declared `accel` with `coef0 = 0` / `coef1 = 1` are rewritten to `piezo`
  during config parsing (setupstr.m parity, exact-string trigger), so
  CASPER Ax/Ay convert as piezo counts and route to the `VIB` role instead of
  a fake 9.81·counts "m/s²" ACC; `id_even`/`id_odd` channel sections with no
  `id` synthesize the 2-id (32-bit) join; and a matrix address with no usable
  `[channel]` section now warns (read_odas.m parity; special address 255
  exempt). Note one deliberate side effect: the adapter's vibration routing is
  now type-based, so modern files whose Ax/Ay are declared `type = piezo`
  (e.g. SN479) report `vib_type = "VIB"` instead of `"ACC"` — harmonizing with
  `p_to_netcdf`'s classification. On mixed hardware the vibration stack is the
  name-sorted union of true accelerometers and piezo channels (labeled `ACC`),
  so Goodman noise removal keeps every coherent reference; the label is
  descriptive only and epsilon/chi values are unchanged.
- **Hotel-telemetry speed method** (issue #131 finding M10). New perturb
  `speed.method: "hotel"` (+ `speed.hotel_var`, default `"speed"`) consumes a
  hotel-merged channel as the through-water speed for epsilon/chi — previously
  a merged hotel `speed` channel was written to the outputs but consumed by
  nothing, while the docs claimed it fed the pipeline (finding m10; the
  pipeline/configuration docs now describe the real mechanism, which this
  change makes true). Slow-grid channels are interpolated and
  Butterworth-smoothed onto the fast grid exactly like the em/flight methods;
  fast-grid channels (the default `hotel.fast_channels` placement for
  `"speed"`) get the same NaN-interp/smooth/floor treatment without
  regridding. An explicitly requested hotel speed that is unusable (channel
  missing, matching neither time grid, or < 50% finite) is a per-file
  **error** that aborts the file with a recorded `errors` entry — never a
  silent fall-back to the 0.05 m/s `speed_cutout` floor, and never a silent
  downstream substitution with |dP/dt|. The same abort applies to any
  explicitly selected non-pressure method (em/flight/constant/hotel) whose
  speed stage fails; only the default pressure method keeps the historical
  warn-and-continue (its downstream fallback recomputes the same |dP/dt|).
  And the failures now actually fire: an `em`/`flight` speed with **zero**
  finite samples (dead flowmeter; level-flight/all-inflection pitch) errors
  before the cutout floor is applied — previously the all-NaN fill published
  a constant 0.05 m/s with provenance `"em"`/`"flight"`, indistinguishable
  from a real 0.05 m/s speed — and a non-finite `speed.value` errors for
  `constant`. em/flight deliberately use a zero-finite threshold rather than
  hotel's 50% rule: the flight model legitimately NaNs every sample below
  `min_pitch_deg` at dive/climb inflections, so a fraction cut could
  false-error real casts.
  Product provenance: `speed_source = "hotel:<var>"` on the per-profile
  NetCDFs, carried through to the diss/chi attrs by the precomputed-speed
  mechanism (which now retains upstream source strings that say more than the
  method name, e.g. `"hotel:speed"`, `"constant:0.4"`). And the other half of
  M10: when a hotel merge injected channels literally named
  `speed_fast`/`W_slow` whose values the speed stage recomputes and discards,
  the pipeline now **warns** and names the remedy instead of overwriting
  silently — `W_slow` always (it is always recomputed as smoothed |dP/dt|,
  method-independent), and `speed_fast` unless `speed.method: "hotel"` with
  `speed.hotel_var: "speed_fast"` actually consumes it. The rsi `--speed-method` layer
  keeps `hotel` perturb-only (hotel channels are merged there) and says so
  when asked for it. Note — hash churn: the new `speed.hotel_var` key changes
  the perturb `speed` section hash, so stage directories keyed on it recompute
  into fresh directories on the next run (the key set is part of the
  provenance signature).
- **Legacy ODAS header-v1 `.p` support: the `rsi-tpw v1to6` translator**
  (issue #141). Pre-2015 files (header word 11 == 1; record 0 holds a binary
  address matrix, configuration in an external old-dialect setup file) are
  translated to standard v6 files — vendor `patch_setupstr.m` header contract
  (words 11-14 → `0x0600`/config-size/0/0, everything else preserved), data
  records copied byte-for-byte (unlike the vendor tool, per-record headers,
  timestamps, and bad-buffer flags survive), and the embedded INI is
  **synthesized** from the setup file (old `key: values` dialect parser in
  `rsi/setup_v1.py`; auto-detected `setup.txt` > `setup*.txt` > `setup*.cfg`
  siblings with INI-dialect sniffing, cross-candidate consistency warnings,
  and a hard record-0-matrix assertion) with machine-readable provenance keys
  in `[root]`. The **complete** provenance set (`translated_from`,
  `v1_source_file`, `setup_file_source`, `setup_file_md5`, `sens_source`,
  `translator`, `translated_on`) is carried onto every derived product —
  full-record and per-profile NetCDF, standalone epsilon/chi files, and the
  pipeline L4 writers — on both routes (on-disk translated file and direct
  raw-v1 read); the setup-file md5 + sens source are the audit trail for the
  sens⁻² epsilon scaling. `PFile` reads raw v1 files directly via the same
  translation in memory (`setup_file=` kwarg to override discovery;
  `translated_from_v1` / `v1_provenance` attributes), refuses any other
  pre-v6 version loudly (raw + decoded version in the message), and the v6
  path is pinned by golden per-channel regressions of the three committed
  fixtures (exact raw-count hashes; converted channels via stats plus
  order-sensitive shape/finite-mask/decimated-value goldens).
  Shear sens is never defaulted: `--sens sh1=…,sh2=…`, `<name>_sens:` setup
  keys, or `patch-config --add-keys` on translated files (the per-epoch
  workflow); a sens-less shear channel errors at conversion time. FP07
  thermistors stay raw counts on v1 corpora (no authoritative calibration —
  the ~4.2 °C offset of the only on-disk candidate coefficients makes them
  untrustworthy); chi on v1 is deferred. See `docs/rsi-tpw/legacy_v1.md`.
- **Sea-Bird SBE3/SBE4 (`sbt`/`sbc`) converters** (vendor
  `odas_sbt/sbc_internal` parity; coef4-6 carry f0/f_ref/n_periods; verified
  to ≤1e-8 °C / ≤1e-8 mS/cm against the 2013 Taiwan channel-level ground
  truth; zero counts yield NaN instead of a division blow-up). Files whose
  configs declare `sbt`/`sbc`-type channels previously kept raw counts with a
  "no converter" warning — they now convert; files without such channels are
  bit-identical. The `auto` reference-temperature chain gains an `sbt`-typed
  candidate tail (after `JAC_T`, matched by channel TYPE, not name), so a v1
  corpus with counts-valued FP07s lands on the SeaBird automatically, and
  `--conductivity SBC1 --salinity measured` works (mS/cm, SP_from_C-ready).
- **v1 refusal guards** in every other positional `.p` reader — perturb trim
  (which would otherwise write a corrupt "trimmed" copy of every v1 file),
  perturb merge, `rsi-tpw sensors` (silently-empty inventory), patch-config /
  patch-template, `rsi-tpw config` (`read_config_string`), and `cutp`
  (`extract_pfile_segment`) — each names the `rsi-tpw v1to6` remedy.
- **Selectable reference temperature/conductivity with plausibility QC**
  (issue #131 finding B1). The reference temperature that drives seawater
  properties (viscosity ν for ε; ν and κ_T for χ; the published `T_mean`) is
  no longer hard-coded to `T1`:
  - **rsi**: new `epsilon.temperature` / `epsilon.conductivity` config keys
    (and the same in `chi`), plus `--temperature` / `--conductivity` flags on
    `rsi-tpw eps`, `chi`, and `pipeline`. Default `"auto"` picks the first
    plausible of `T1`..`Tn`, bare `T`, `JAC_T` — a railed/drifting/mostly-NaN
    channel is skipped with a warning (QC evaluates in-water samples,
    P > 0.5 dbar, when pressure is available), and a file with **no**
    plausible reference temperature errors per file instead of publishing
    wrong-viscosity products (on a railed-T1 corpus the old behavior was
    ε ≈ 5× low with all-green QC). An explicitly named channel is honored
    even when it fails QC (loud warning); a **number** is a constant
    reference temperature in °C (ODAS `constant_temp` parity). ODAS
    divergence is deliberate: ODAS uses T1 unchecked and silently
    substitutes 10 °C when temperature is missing.
  - `--salinity` (eps/chi) now also accepts `"measured"`: per-sample
    practical salinity from the resolved conductivity/temperature pair and
    pressure (TEOS-10), preferring the co-located `JAC_C`/`JAC_T` pair and
    falling back to the selected reference temperature (with a warning) when
    JAC_T is implausible; on `pipeline`, `--salinity measured` maps to the
    automatic path (measured JAC salinity was already preferred there).
  - Product provenance: the diss/chi NetCDFs gain `temperature_source`,
    `temperature_qc`, and `conductivity_source` (the resolved conductivity
    channel — consumed only under `salinity: measured`), plus
    `salinity_pair_temperature` when a measured salinity was actually
    computed; the run_pipeline L4 products gain `temperature_source`/
    `temperature_qc`. A constant reference outside the plausible ocean
    range is recorded in `temperature_qc` (and warns) instead of claiming
    "pass".
  - **perturb**: `epsilon.T_source` is now actually implemented (it was
    parsed but stripped before the computation — a dead key). `null`/`"auto"`
    = the QC chain above; a channel name or a number work as in rsi. One knob
    serves both the diss and chi stages. The interactive rsi viewers
    (`rsi-tpw ql`/`dl`/`ml`) reuse the same resolver for their viscosity
    preview.
  - **Note — hash churn**: adding the `temperature`/`conductivity` keys to
    the rsi `epsilon`/`chi` sections (and removing perturb's `T1_norm`/
    `T2_norm`, below) changes the config hashes, so existing `eps_NN`/
    `chi_NN` (and perturb stage) directories will not be reused (epsilon/chi always; prof dirs only when the merged profiles key-set changed — an explicit numeric W_min keeps them) — the next
    run recomputes into fresh directories. Deliberate: the key set is part
    of the provenance signature.
- **Salinity from a hotel file** — `epsilon.salinity`, `chi.salinity`, and the
  new `stratification.salinity` now accept `"hotel"` (or `"hotel:<var>"`) to draw
  practical salinity from a hotel-injected channel (default variable `salinity`),
  in addition to the existing `null` / number / `"measured"` forms. This feeds ε
  and χ **viscosity** and the `N2`/`Γ`/`K_ρ` **mixing** path from an external CTD
  feed, so gliders and MicroRiders with no onboard conductivity get correct
  viscosity and stratification instead of the fixed 35-PSU fallback. Set all
  three keys together (an unset `stratification.salinity` while ε/χ use `"hotel"`
  now warns). `epsilon.salinity` also gains the `"measured"` form for parity with
  `chi`. Note: the practical-/absolute-salinity/density (`SP`/`SA`/`ρ`) CTD and
  profile products still require onboard conductivity and are unaffected. See
  docs/perturb/configuration.md.
- **Shell tab-completion** for the argparse CLIs (`rsi-tpw`, `perturb`,
  `perturb-plot`, `perturb-diag`) via the optional
  [argcomplete](https://github.com/kislyuk/argcomplete) dependency (new
  `completion` extra: `pip install 'microstructure-tpw[completion]'`). Each CLI
  calls a shared `enable_argcomplete()` hook before parsing; it is a no-op unless
  argcomplete is installed and the shell's completion machinery is driving the
  process, so normal runs are unaffected. Enable it per shell with
  `eval "$(register-python-argcomplete rsi-tpw)"`. See docs/rsi-tpw/completion.md.
- **`rsi-tpw config FILE...`** — print a `.p` file's raw embedded configuration
  (INI) record — the `setup.cfg`-style text with the address matrix and every
  channel's calibration coefficients — to stdout. Useful for inspecting a
  coefficient in place (e.g. a suspect pressure `coef2`). Reads only the header
  and config record, so unlike `info` it also works on startup/truncated files
  that carry a config but no data records; backed by the new
  `PFile`-independent `read_config_string()`. See docs/rsi-tpw/cli.md.
- **`rsi-tpw sensors --cal-dir DIR`** — check each shear probe's configured
  `sens` against Rockland shear-probe calibration sheets (PDFs) in `DIR` and
  report mismatches. Parses `Probe SN`, sensitivity, calibration date, and the
  optional previous calibration date/sensitivity; the sensitivity applied to an
  observation is the calibration **in effect** at its date (hold-previous;
  linear interpolation between calibrations is intentionally deferred).
  `--cal-tol` sets the flag threshold as an **absolute** sensitivity difference
  (same units as `sens`, default `0.00005` — half the sheets' 4th-decimal
  resolution), not a percentage. PDF reading uses the new
  optional `cal` extra (`pip install 'microstructure-tpw[cal]'`, i.e. `pypdf`);
  it is imported lazily so the rest of `sensors` works without it. See
  docs/rsi-tpw/sensors.md.
- Zenodo DOI badge + identifiers (concept DOI `10.5281/zenodo.21366142`,
  version DOI `10.5281/zenodo.21366143`) in the README and `CITATION.cff`,
  plus a PyPI version badge — back-filled after v0.3.0 was archived on Zenodo.
- **`--speed-method {pressure,em,flight}` + `--aoa` on `rsi-tpw eps` and `chi`**
  (issue #131 findings M1/M2) — the through-water speed models that `pipeline`
  and perturb already had are now reachable from the modular commands (config
  keys `epsilon.speed_method` / `epsilon.aoa_deg`, same in `chi`; the
  `em`/`flight` methods are mutually exclusive with a fixed `--speed`, and
  the same exclusivity is now enforced by `run_pipeline`). An **explicit**
  `--speed-method pressure` forces the |dP/dt| path even when the input (a
  perturb per-profile file) carries a precomputed `speed_fast` channel;
  leaving the method unset prefers that channel, as before. `rsi-tpw
  pipeline` additionally honors the YAML `epsilon.speed_method`/`aoa_deg`
  (previously CLI-flag-only) and warns when the `[chi]` section sets a
  different `speed_method`/`aoa_deg`/`W_min` (single per-file resolution,
  like `temperature`/`conductivity`).
- **Speed provenance** (issue #131 finding M8): the diss/chi products now carry
  a `speed_source` attr stamped at the speed precedence point —
  `"fixed --speed <v>"` / `"precomputed speed_fast (perturb speed.method=em)"`
  / `"em (U_EM)"` / `"flight (aoa=3)"` / `"pressure |dP/dt|"` — and
  `compute_speed_for_pfile` returns its source string
  (`"pressure" | "em" | "flight" | "constant:<v>"`) as a third value so callers
  stamp what was actually computed. A stale upstream `speed_method` attr is
  cleared when a fixed speed or the pressure path is used, so the attrs can
  never claim e.g. `speed_method="em"` next to a fixed-speed source. The
  perturb pipeline writes `speed_method`/`speed_source` global attrs into the
  per-profile NetCDFs (new `extra_attrs` parameter on `extract_profiles`;
  written whenever the speed channel was computed) and the diss/chi stages
  inherit them.
- **perturb warning capture** (issue #131 finding M8): `warnings.warn(...)`
  (e.g. the glider |dP/dt| speed-bias warning) is now routed into the perturb
  run/worker/stage log files via `logging.captureWarnings(True)`, enabled in
  both `setup_root_logging` and `init_worker_logging` (spawn workers never run
  the former) — never at module import.

### Changed
- **Glider-correct detection defaults** (issue #131 findings M3/M13/M7):
  `W_min` now defaults to **auto** everywhere — 0.3 dbar/s for a free-falling
  profiler, 0.05 for glide/horizontal platforms — in rsi `prof`/`eps`/`chi`/
  `pipeline` (config `profiles.W_min: null`, new `epsilon.W_min`/`chi.W_min`
  keys + `--W-min` flags) and in perturb (`profiles.W_min: null`). In
  `run_pipeline` the resolved value also feeds `L2Params.profile_min_W`, so
  the L2 section selector no longer discards glider data that detection
  accepted. perturb's `profiles.direction` default flips **"down" → "auto"**
  (vehicle-resolved; a Slocum gets `glide` = up + down), and its detection
  site now resolves the same merged defaults that the cache signature hashes.
  Scope qualifier: **auto helps only vehicle-declaring corpora** — files whose
  `instrument_info` carries no `vehicle` (e.g. CASPER-era MicroRiders) still
  resolve to `down`/0.3 and need an explicit `direction: glide`. Note for
  local Rutgers configs: the `Rutgers/perturb.yaml` workaround comment
  ("REQUIRED: batch default 0.3 rejects gliders") is now obsolete — `W_min`
  and `direction` can be left unset there.
- **`rsi-tpw prof` is vehicle-aware** (issue #131 finding M13):
  `extract_profiles` now resolves `vehicle`/`direction`/`W_min`/tau before
  detection — `--vehicle slocum_glider` previously crashed with a
  `TypeError` (unexpected keyword) and `--direction auto` silently behaved
  as `down`. Full-record NetCDF sources resolve the vehicle from their
  `platform_type` attr (falling back to an instrument-model heuristic). VMP
  defaults are bit-identical (`down`, tau 1.5, W_min 0.3).
- **No more silent-empty products** (issue #131 finding M4): when profile
  detection finds nothing, `eps`/`chi` warn with the observed pressure span
  and peak fall rate vs the thresholds (and the flag to relax), print a
  per-file "no profiles detected" plus a final "N of M file(s) produced
  output" summary. **Exit-code change**: `rsi-tpw eps`/`chi` now exit **1**
  when *no* input file produced output (all failed and/or all empty), so a
  `set -e` batch script fails instead of reporting success over an empty
  directory — deliberate.
- **`rsi-tpw info`/`prof` batch robustness** (issue #131 finding M5): a
  startup file (valid config, no data records), 0-byte, or truncated `.p`
  file no longer aborts the batch — per-file errors print `ERROR: ...` and
  processing continues (exit 1 only when every file failed). The `eps`/`chi`
  loops additionally catch `struct.error` (truncated mid-file `.p`), matching
  `nc`.
- **Note — hash churn**: the `profiles`/`epsilon`/`chi` key-set and default
  changes above alter the rsi and perturb config hashes, so existing
  `prof_NN`/`eps_NN`/`chi_NN` and perturb stage directories will not be
  reused — the next run recomputes into fresh directories. Deliberate: the
  key set is part of the provenance signature, and this lands in the same
  release as the reference-temperature keys (one churn, not two).
- **FP07 factory calibration travels with per-profile NetCDFs** (issue #131
  finding m8). `extract_profiles` now writes `diff_gain` and the base
  thermistor's calibration (`e_b`, `b`, `gain`, `beta_1`, `beta_2`, `adc_fs`,
  `adc_bits`, `T_0` — exactly the set `_extract_therm_cal` can produce,
  including `b`, consumed by the electronics-noise model's eta term; the
  noise model's remaining knobs keep their defaults) as float variable attrs
  on every `T*_dT*` gradient channel — and on plain fast `T*` channels for
  instruments without pre-emphasis (the first-difference fallback) — and the
  chi loader reads them back from NetCDF sources. A per-profile file without
  the attrs (written before this change) falls back to the old generic
  defaults (`diff_gain=0.94`, ODAS-default thermistor coefficients) with a
  logged warning naming the file — never silently. The attrs carry the **factory** coefficients from the embedded
  `.p` config string; perturb's FP07 in-situ calibration may rewrite the
  channel *data*, which is compatible — the attrs describe the electronics.
  *(Corrected in issue #245: `T_0`/`beta_1`/`beta_2` are the bead's R(T), not
  the electronics, so under an in-situ fit they must be the fit's. See the
  Unreleased Fixed entry.)*
- **`load_channels` carries vibration-sensor types** (`channel_types`: `.p`
  from `pf.channel_info`, NetCDF from the `sensor_type` attr
  `extract_profiles` writes), and the shared L1 builder labels an
  all-piezo-typed vibration stack `VIB` instead of `ACC`, matching the
  adapter's piezo convention (#131 W5-ii, W3 review F5). Label-only: no
  numeric consumer branches on `vib_type` (Goodman treats both alike), but
  displays now say `VIB` for piezo instruments (e.g. SN479's `Ax`/`Ay`).
- **`perturb-plot gamma-scaling` temperature Thorpe route resolves its
  channel** through the QC'd reference-temperature resolver (`T1` first,
  then `T2`…/`T`/`JAC_T`) instead of hard-coding slow `T1`. A profile whose
  every candidate is missing or implausible loses only its temperature route
  (NaN `LT_temp`, counted in the "skipped/degraded" notes) — a figure is
  never crashed by a railed thermistor.

### Changed
- **A shear channel without a usable `sens` is now a hard per-file error**
  (was: a warning plus a fabricated default of 1.0 — plausible-looking shear
  that scales epsilon by sens⁻²). "Usable" means parseable, **finite**, and
  positive: `nan` would bypass a bare sign check (every NaN comparison is
  False → all-NaN shear) and `inf` yields all-zero shear, so both are
  rejected everywhere sens enters — conversion, the `v1to6 --sens` CLI
  parser, and the Python-API overrides. ODAS `convert_odas.m` parity: the
  vendor errors outright too. Modern configs always carry `sens`, so only
  genuinely broken configs and un-patched v1 translations are affected; the
  error names the three remedies (`patch-config --add-keys`, `v1to6 --sens`,
  `<name>_sens:` setup keys). (#141)

### Removed
- **perturb `epsilon.T1_norm` / `epsilon.T2_norm` config keys** — they were
  never implemented (stripped before the computation, a silent no-op) and the
  template comment ("null = blend T1/T2") described behavior that never
  existed. **Breaking**: `validate_config` is strict, so a config that still
  sets them now fails loudly with an unknown-key error — delete the two lines
  (they never did anything).

### Fixed
- **`rsi-tpw cutp` / `extract_pfile_segment` absolute time** — a segment cut
  with `--start N>0` used to copy record 0's header timestamp verbatim, so the
  output's derived start time read N records too early (embedded provenance
  that would mislead any hotel/GPS join or absolute-time consumer). The tool
  now advances the record-0 timestamp by `N x recsize` (record duration from
  the embedded config, default 1.0 s; datetime arithmetic carries milliseconds
  across minute/hour/day boundaries; a year-0 startup clock is left unchanged
  with a warning). `tests/data/MR_SL685_climb.p` was regenerated with the
  fixed tool — data records byte-identical, only the record-0 timestamp words
  changed — and `test_mr_e2e.py` now pins the fixture's absolute start time.
- **chi `dof_spec` now subtracts the Goodman DOF loss** (#131 m9) — the chi
  product's `dof_spec` attribute is `1.9 * max(num_ffts − n_vib, 1)` when
  Goodman noise removal ran (each coherently-removed vibration signal costs
  one FFT segment of DOF, Lueck 2022b), matching the epsilon convention in
  `dissipation.py` instead of the uncorrected `1.9 * num_ffts`. **Note for
  downstream consumers of `dof_spec`:** values on the chi product are now
  smaller wherever Goodman ran with vibration channels present (e.g. 9.5
  instead of 13.3 for the default 4-segment window with 2 accelerometers);
  chi values themselves are unchanged.
- **Stratification salinity is now scrubbed, with a truthful provenance note**
  (#131 M6). The N²/dT/dz paths (window-scale in the diss/chi products,
  slow-grid in the profile product) previously read a hotel-sourced salinity
  channel raw: an **all-NaN merged channel** (a hotel variable with fewer than
  two finite source samples) silently NaNed out N² and every mixing product
  (K_T/Γ/K_ρ) while the metadata still claimed "salinity from hotel channel",
  and **partial NaNs** (a `"hotel:<var>"` pointing at a derived profile
  variable, e.g. an SP computed from bad conductivity) NaNed the affected
  windows with no note. (The hotel merge itself never produces NaN outside the
  hotel file's time coverage — it boundary-holds and bridges interior gaps —
  so the earlier "outside CTD coverage" description of this failure mode was
  wrong.) Both stratification consumers now scrub the salinity: non-finite
  samples are filled by **interpolation over the finite samples**
  (nearest-finite hold at the edges) — **per cast** on the slow-grid path,
  because interpolating across a cast boundary would blend one cast's deep
  salinity toward the next cast's shallow values and fabricate a wrong-sign
  within-cast dS/dz (collapsing N² near the boundary). A channel — or a cast
  slice — with **fewer than two finite samples** falls through to
  conductivity/35 PSU (the hotel merge's own <2-finite convention; a single
  sample must not constant-fill a profile and beat a valid conductivity
  channel). The N² `comment` records exactly what was used: interpolated vs
  edge-held counts ("N/M non-finite salinity samples interpolated", "N/M edge
  samples held at the nearest finite value; N2 approaches temperature-only
  where held") or the fallback reason ("hotel channel 'x' entirely
  non-finite"). The **conductivity fallback is gated the same way**: a
  conductivity channel that is present but whose derived `SP_from_C` salinity
  has fewer than two finite samples (e.g. an all-NaN `JAC_C`) now falls
  through to the documented 35 PSU fallback with the reason noted ("JAC_C
  present but yielded no finite salinity"), instead of producing all-NaN N²
  under a note claiming C-derived salinity; a partially-NaN derived SP (two
  or more finite samples) keeps the existing per-window masking
  (interp-filling derived SP is deferred). The slow-grid N² comment also no
  longer claims the salinity
  came "from the profile's own C/T/P" when it did not. The viscosity path's
  scrub switches from a whole-profile **median** fill to the same
  interpolation fill: N² is first-order in dS/dz, so a constant fill would
  insert a spurious salinity step at every fill boundary (spurious N² ~
  g·β·ΔS/Δz, orders above thermocline values, and Thorpe sorting keeps it);
  for viscosity the difference is negligible.
- **Old-format MicroRider deconvolution** (#131 M11, M12) — two bugs that
  corrupted or crashed processing of old-MR corpora (e.g. CASPER 2015_East
  CAS_001-006, whose 4×10 matrix samples T1 *and* T1_dT1 as full fast columns
  and P/P_dP twice per scan). **M11**: a natively-fast base channel (T1) kept
  its "fast" classification after deconvolution overwrote it with slow-length
  data, so `is_fast()` lied and `rsi-tpw nc` crashed with a broadcast error in
  the gradT builder; the base is now reclassified slow (the full fast-rate
  deconvolved signal lives in `T1_dT1`, matching the modern-config invariant).
  **M12**: a duplicate-sampled slow pair (P/P_dP, 2× per scan) was deconvolved
  at the 2× matrix-occurrence rate instead of the decimated array's true rate,
  applying the pre-emphasis crossover at twice its design frequency and
  mis-blending P and P_dP — the pressure feeding fall-rate/speed and thus ε/χ.
  The deconvolution rate (and fast/slow branch) is now derived from the array
  actually extracted.
- **Perturb chi no longer computed with generic FP07 calibration** (issue
  #131 finding m8). The perturb pipeline computes chi from per-profile
  NetCDFs, whose loader hard-coded `diff_gain=0.94` and an empty thermistor
  calibration; with the new attrs (see Added) it now uses the instrument's
  real coefficients — on SN479, `diff_gain=0.912`/`0.920` and `b=0.99861`/
  `0.99927`, so the 0.94/1.0 hardcodes were provably wrong there. **Numerics
  note**: chi's FP07 noise floor and bilinear differentiator correction on
  the perturb path change wherever the real coefficients differ from the
  defaults — expected, and in the correct direction (`rsi-tpw chi` on the
  `.p` file and perturb-on-NetCDF now agree; measured on the SN479 fixture
  the residual matched-window difference drops from ≲3.5% to ≲0.15%).
- **Measured-salinity CT pairing QC** (rsi `run_pipeline` adapter): the
  practical salinity computed from `JAC_C` now pairs with `JAC_T` only when
  JAC_T passes the plausibility QC, falling back to the resolved reference
  temperature with a warning — previously a railed JAC_T silently poisoned
  the measured salinity.
- **`rsi-tpw pipeline` batch robustness**: one unreadable/implausible file no
  longer aborts a multi-file run; the pipeline logs the per-file error and
  continues (mirroring the `eps`/`chi` loops).
- **`rsi-tpw patch-template`** now scaffolds *every* per-channel calibration
  field, not just `coef0`/`coef1`. The previous hardcoded whitelist silently
  dropped higher-order polynomial coefficients (a pressure channel's `coef2`,
  `coef3`…) and thermistor Steinhart-Hart terms (`a`, `b`, `beta_1`, `t_0`, …),
  so a coefficient a user needed to patch never appeared in the template. The
  scaffold now emits all fields except the structural identifiers
  (`id`, `name`, `type`, `units`, `sign`), in config-file order.
- **Interactive viewers (`ql`/`dl`/`ml`) — direction-aware `W_min` default.**
  The fall-rate floor for profile detection now defaults to **0.05 dbar/s** when
  the resolved direction is `glide` or `horizontal` (slow glider/AUV motion),
  and stays **0.3 dbar/s** for `down`/`up` (free-falling profilers). Previously a
  fixed 0.3 rejected every cast from a slow platform, so `--direction glide`
  alone still found nothing; now it works without also passing `--W-min`. An
  explicit `--W-min` still overrides. The batch `prof` command is unchanged.
- **Interactive viewers (`ql`/`dl`/`ml`)** now explain *why* no profiles were
  detected instead of the bare "No profiles detected in this file". The message
  reports the observed pressure span and peak fall/rise rate against the
  `P_min`/`W_min` thresholds and suggests the fix — e.g. a slow or glider-style
  cast whose fall rate never reaches the `W_min` default is told to lower
  `--W-min` and/or use `--direction glide`.

## [0.3.0] - 2026-07-14

### Added
- **Release/citation metadata** — `CITATION.cff` (drives GitHub's "Cite this
  repository" button) and `.zenodo.json` (controls the Zenodo archive record),
  PyPI trove classifiers and expanded `[project.urls]`, a package-level
  `odas_tpw.__version__` sourced from the installed distribution, and a
  Citation section in the README. Groundwork for a tagged, DOI-archived
  `v0.3.0` release published to PyPI.
- **`rsi-tpw bench`** — a bench-test diagnostic, ported from ODAS
  `quick_bench.m` and extended to auto-evaluate the *Rockland Bench Test
  Review Checklist* (V3). Produces raw-count time-series and
  counts²·Hz⁻¹ spectra figures (plus a CT/CLTU figure when those
  channels exist) and a checklist report that marks each quantitative
  criterion PASS/FAIL and flags the subjective ones REVIEW. Reads with a
  new `PFile(path, deconvolve=False)` option so the pre-emphasized
  channels (`T1_dT1`, `T2_dT2`, `P_dP`) carry the raw counts the
  checklist thresholds are defined on. See docs/rsi-tpw/bench.md.

### Changed
- **perturb epsilon/chi windows are now duration-based** (`fft_sec`,
  `diss_sec`, `overlap_sec`), resolved to sample counts at each
  instrument's sampling rate, and the **epsilon FFT default changed from
  256 samples to 1 second** (512 samples on a 512-Hz VMP-250; 1-2 kHz
  coastal units scale automatically) per Lueck et al. (2024) best
  practices — see docs/perturb/dissipation_length.md. Sample-count keys
  (`fft_length`/`diss_length`/`overlap`) remain as expert overrides that
  win when set; legacy configs pinning them keep byte-identical stage
  signatures and resolve their existing output directories unchanged.

### Fixed
- Reference library (`papers/`) DOI corrections: fixed three wrong DOIs —
  Rehmann & Hwang (2005) was a dead link (`JPO2676.1` → `JPO-2676.1`) and
  the two Smyth & Moum (2000) companion papers were cross-linked to each
  other / an unrelated paper — plus nine old-AMS DOI hrefs whose literal
  parentheses truncated the clickable link (now percent-encoded), and two
  citation nits (Nash et al. 1999 page range, Shapiro et al. glider-paper
  year). All 53 DOIs now resolve to the correct paper.
- CT alignment (`ct_align`) and the despike pass-count flag: conductivity
  is now shifted with edge-hold instead of `np.roll` (which wrapped one
  end of the record into the other), and the too-many-passes QC flag
  uses each window's own section pass count
  (`PASS_COUNT_SH(probe, section)` semantics) instead of the per-channel
  maximum.
- Method-1 chi: parabolic refinement of the log-grid search removes the
  ~4.7 % grid quantization on chi (synthetic recovery now better
  than 0.5 %).
- No-Goodman shear spectra are detrended parabolically, matching ODAS
  `get_diss_odas.m` (`method='parabolic'` in the no-goodman branch);
  Goodman-cleaned spectra remain linear, as in `clean_shear_spec.m`.
- Fall-rate smoothing no longer reuses the shear high-pass cutoff
  (`HP_cut`) as a pre-smoothing knob when `speed_tau > 0`: the speed is
  smoothed exactly once, at 0.68/tau (the ODAS convention). Benchmark
  speed and epsilon agreement unchanged.
- Documented the FP07 tau/transfer-function pairings (`double_pole`
  pairs with the fixed 3 ms Goto tau, NOT Peterson & Fer's
  speed-dependent tau — and how to reproduce Peterson & Fer exactly)
  and the iterative chi fit's actual limit-refinement behavior.

### Added
- Complete ATOMIX QC flag set for epsilon (`EPSI_FLAGS`): in addition to
  the existing FM (bit 1) and variance-resolved (bit 16) criteria, the
  flags now include despike fraction (bit 2), inter-probe consistency
  (bit 4, `ln(e_i/e_min) > 1.96*sqrt(2)*sigma_ln` with the Lueck 2022a
  variance model — only the larger estimates are flagged), and
  too-many-despike-passes (bit 8). The variance-resolved criterion is
  now applied only to variance-method estimates (not ISR), matching the
  benchmark convention. Per-window `DESPIKE_FRACTION` is computed in
  L2/L3 and carried in `L4Data`. The benchmark comparison
  (`scor160-tpw l1-l4`/`l2-l4`) now reports QC flag agreement using each
  dataset's own thresholds (`read_l4_qc_limits`): 83-100% exact match
  on the VMP250 tidal and MR1000 datasets.
- Derived mixing quantities in the `perturb` chi stage (`chi.mixing`,
  default true): `N2`, `dTdz`, `K_T`, `Gamma`, `K_rho` on the chi window
  grid, with practical salinity from the profile's own C/T/P (TEOS-10)
  so `N2` is fully constrained.
- Derived mixing quantities in the `rsi-tpw pipeline` Method-1 chi output
  (`L4_chi_epsilon.nc`, propagated to `L5_binned.nc`/`L6_combined.nc`):
  window-scale `N2` (TEOS-10) and background `dTdz`, Osborn-Cox heat
  diffusivity `K_T`, measured mixing coefficient `Gamma` (Oakey 1982),
  and Osborn diffusivity `K_rho` (Gamma_0 = 0.2), with
  stratification/gradient validity masking. New module
  `odas_tpw.processing.mixing`.
- Robustness tests for corrupted/truncated `.p` files and for the
  `accel`/`magn` channel converters.
- `perturb` now runs profile detection and CT alignment *before* the CTD
  fork, so the CTD product's salinity/density are computed from
  time-aligned conductivity whenever profiles are detectable (previously
  always unaligned).
- Per-CLI / per-worker / per-stage / per-combo log fan-out for `perturb`. Every
  invocation now writes `<output_root>/logs/run_<stamp>.log`, plus
  `worker_<stamp>_<pid>.log` per parallel worker, `<stem>.log` inside each
  versioned `profiles_NN/`, `diss_NN/`, `chi_NN/`, `ctd_NN/`,
  `*_binned_NN/` dir, and `combo.log` in each combo dir. See
  `docs/perturb/logging.md`.
- `--stdout` and `--log-level` flags on every pipeline-running subcommand.
- YAML configuration file system with `rsi-tpw init` template generation
- Three-way parameter merge: defaults <- config file <- CLI flags
- Cumulative hash-tracked sequential output directories (`eps_00/`, `chi_00/`, ...)
- Configuration validation with clear error messages for unknown sections/keys
- Resolved `config.yaml` written to each output directory for reproducibility
- Comprehensive documentation in `docs/`: configuration reference, CLI reference, Python API, pipeline guide
- `scor160-tpw` CLI reference documentation (`docs/scor160/cli.md`)
- Regression test for parallel Method 1 chi (`test_cmd_chi_method1_parallel`)
- CHANGELOG

### Fixed
- `rsi-tpw chi --epsilon-dir` now works in parallel mode (`-j > 1`); previously `epsilon_dir` was ignored and all workers used Method 2
- Serial chi path now properly closes epsilon datasets via try/finally (prevents resource leak)
- `perturb run` merge stage now feeds merged files to downstream processing (return value was previously discarded)

### Changed
- `perturb` console output is now silent by default — every record goes to the
  per-invocation log file.  Pass `--stdout` to also stream to stderr.  Previously
  `logging.basicConfig` printed everything to stderr unconditionally.
- `*_binned_NN/` and `*_combo*/` output dirs are now created up-front (before
  the bin/combo work runs) so per-input log files can be opened inside them.
  An empty bin/combo step now leaves a dir with just `.params_sha256_*` and the
  log files, instead of no dir at all.
- Output directories now use sequential naming with parameter hash deduplication
- Each output directory includes a `config.yaml` recording the exact parameters used
- `README.md`, `CLAUDE.md`, and `docs/rsi-tpw/python_api.md` now use `run_pipeline()` / `compute_diss_file()` / `compute_chi_file()` instead of deprecated `get_diss()` / `get_chi()`
- `docs/rsi-tpw/pipeline.md` and `docs/rsi-tpw/output_directories.md` updated to reflect actual `{stem}/profile_NNN/` pipeline output layout
- `docs/perturb/pipeline.md` updated: `perturb run` covers the full chain trim→merge→process→bin→combo (`run_pipeline()` now calls `_run_combo`); `perturb combo` remains available to re-run combo assembly on its own
- `docs/perturb/cli.md` now documents `--hotel-file` and `--p-file-root` flags
- Fixed ~30 stale source paths in `docs/chi_mathematics.md`, `docs/epsilon_mathematics.md`, and `docs/rsi-tpw/vectorization.md` (`rsi/` → `chi/` or `scor160/` as appropriate)

## [0.1.0] - 2025-03-08

### Added
- `PFile` class for reading Rockland Scientific `.p` binary files
- Channel conversion functions (raw counts to physical units) for all sensor types
- NetCDF4 conversion with CF-1.13 compliance
- Profile detection from pressure time series
- Epsilon (TKE dissipation rate) calculation from shear probe spectra
  - Nasmyth universal spectrum fitting (Lueck improved coefficients)
  - Iterative despiking of shear signals
  - Goodman coherent noise removal using accelerometer cross-spectra
  - Macoun & Lueck wavenumber correction
  - Figure of merit (fom) and K_max_ratio QC metrics
- Chi (thermal variance dissipation rate) calculation from FP07 thermistor spectra
  - Method 1: from known epsilon with FP07 transfer function correction
  - Method 2a: MLE Batchelor spectrum fitting (Ruddick et al. 2000)
  - Method 2b: Iterative integration (Peterson & Fer 2014)
  - Batchelor and Kraichnan theoretical spectrum models
  - FP07 single-pole and double-pole transfer functions
  - Electronics noise model
- Unified `rsi-tpw` CLI with subcommands: `info`, `nc`, `prof`, `eps`, `chi`, `pipeline`
- Parallel processing support (`-j` flag)
- Seawater property functions via gsw/TEOS-10: viscosity, density, buoyancy frequency
- MATLAB implementation of chi calculation
- CI pipeline: ruff linting, mypy type checking, pytest on Python 3.12/3.13
- Mathematical documentation for epsilon and chi algorithms

[Unreleased]: https://github.com/mousebrains/turbulence/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/mousebrains/turbulence/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/mousebrains/turbulence/releases/tag/v0.3.0

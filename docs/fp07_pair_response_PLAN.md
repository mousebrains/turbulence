# Per-bead FP07 response equalization from the T1/T2 ratio: plan

Status: IMPLEMENTED 2026-10-13 on `feat/fp07-pair-response`. Precedence where sections differ:
"Implementation and acceptance" (last) > "Review outcome" > the draft.

## Why

Two FP07 beads on one probe head differ in response by a factor ~1.3 in power
at 20 Hz and ~1.6 at 60 Hz (1 sd, 14 independent VMP pairs;
`docs/fp07_tau_prefactor_PLAN.md` on branch `analysis/fp07-tau-prefactor`),
and their response is often not a single pole. The pipeline gives both beads
the same model response (`lueck`, 10 ms * W^-1/2, single pole), optionally
scaled per bead by `fp07_tau_scale` — a single number fitted to make
integrated chi agree, which cannot follow a shape difference.

The ratio of the two beads' gradient spectra measures their RELATIVE
response without a turbulence model:

    median over windows of ln(S2/S1) = ln G + ln|H2(f)|^2 - ln|H1(f)|^2

The absolute level is NOT identifiable from in-situ data (the analysis
branch's synthetic check), so the default model stays the absolute reference
(Pat, 2026-10-13: keep 10 ms) and only the relative part is corrected.

## What

**Convention: geometric mean unity.** Each bead's response is the default
model times a measured factor, and the pair's factors multiply to one:

    |H_i(f)|^2 = |H_model(f; W)|^2 * e_i(f; W),     e_1 * e_2 = 1
    ln e_2 = +lnR'/2,   ln e_1 = -lnR'/2,   lnR' = ln(S2/S1) - <ln(S2/S1)>_{1-3 Hz}

So the pair's average response is exactly the default, nothing about the
absolute level changes, and the two beads stop disagreeing because of their
response. Chi from each bead, and their mean, move relative to the default
only by the bead's departure from the pair average.

**Speed.** e_i is measured at the fit data's median speed W_ref and applied at
any W by assuming the response scales like the default model (time constant
~ W^-1/2): e_i(f; W) = e_i,ref(f * (W_ref / W)^(1/2)). For a VMP W is nearly
constant, so this barely matters; for a glider it is an assumption, recorded
in the file.

**Outside the measured band** [f_lo, f_hi]: below f_lo, e = 1 (both responses
-> 1); above f_hi, e is held at its f_hi value. Recorded in the file; the
report states how much of each chi fit band lies above f_hi.

## Components

1. **`odas_tpw/chi/fp07_pair.py`**
   - `ResponseCorrection`: probe channel names, bead serials, f_ref grid,
     ln_e (2 x n), W_ref, speed exponent, f_lo, f_hi, optional t_start/t_end,
     provenance (source glob, n_files, n_windows, created, estimator, SNR
     gate); NetCDF read/write; `factor(probe, F, W) -> e` with the band and
     speed rules above.
   - `estimate(chi_files, ...)`: group windows by (instrument, bead pair) from
     each chi product's source config (`configuration_string` or the `.p`);
     median-of-log-ratio per frequency bin over windows, gated on a running
     mean of S/N >= 10 on both probes (`medlog.py`, validated on synthetics);
     smooth in log-frequency bins; normalize at 1-3 Hz; report halves (drift)
     and quality (files, windows, band, rms).
2. **CLI `fp07-pair`** (entry point): `fit CHI_GLOB -o OUTDIR`, writing one
   `.nc` per (instrument, pair) plus a report and the YAML lines for
   perturb; `show FILE`.
3. **chi integration** (`l3_chi`, `l4_chi`, `chi_io`): an optional per-gradient-
   channel correction. `l3_chi` multiplies `H2_all[ci]` by
   `factor(ci, F, W_window)`; `l4_chi` wraps the `_h2` callable passed to the
   variance correction with the SAME factor, so the fit and the resolved-
   variance correction see one response. `compute_chi_file(fp07_response=...)`
   matches a correction to a file by bead serials (and time range if given);
   no match -> no correction and a warning naming the serials.
4. **perturb**: `instruments.<SN>.fp07_response: path | [paths]` (relative to
   the config dir; a list covers probe swaps within a deployment). Validated
   on load (file readable, two probes, finite). Hashed by CONTENT (sha256
   replaces the path when the chi stage hash is canonicalized), so editing a
   correction re-versions chi, and only chi. Provenance on every chi product:
   `fp07_response_<channel>` = file name, sha256, bead serial, W_ref, f_hi.
5. **rsi-tpw**: `chi --fp07-response FILE` (repeatable).
6. Coexists with `fp07_tau_scale` (that scales the model; this multiplies the
   result). Documented; using both is allowed but rarely sensible.

## Workflow

Run perturb once -> `fp07-pair fit <output>/chi_NN/*.nc -o fp07pair/` ->
read the report (drift, band, identifiability is not needed: a relative
correction always exists) -> add `fp07_response:` to the instrument -> rerun
(only chi re-versions).

## Validation (acceptance)

- Unit: factor interpolation, band rules, speed scaling, e_1 e_2 = 1, NetCDF
  round trip, serial matching, config validation and content hashing.
- Synthetic: two beads with known different responses (single pole, and a
  shallow free-shape) -> the estimated factors reproduce |H_i|^2/|H_model|^2
  up to the common geometric-mean level, within a stated tolerance.
- Integration: chi with the correction uses H2 * e in both the fit and the
  variance correction (a test that would fail if only one were changed).
- **Real data: ARCTERX-22 SN194** (ratio 0.51 at 20 Hz). chi(T1)/chi(T2) by
  K_max_ratio quartile, before and after. The response-error signature is a
  ratio that trends with K_max_ratio; after correction it should flatten and
  approach 1. Also SN428 (1.43), CASPER-West VMP (0.90, near matched: should
  barely move), and one glider.
- Invariance: correction from H1 applied to H2 (and vice versa) — drift shows
  up as residual disagreement.

## Risks / open questions

- The ratio assumes both probes sample the same turbulence statistics;
  probe separation is a few cm, below the scales of interest.
- Noise-model error near f_hi bends the ratio (SN428 noise 1.3-1.45x model);
  the smoothed-SNR gate limits this; f_hi is reported.
- Drift within a deployment (SN194 +25-50% between halves): v1 makes one
  correction per file; the report flags drift so the user can split by time
  (several files, each with t_start/t_end, listed under one instrument).
- Gliders: W varies (0.15-0.45 m/s); the W^-1/2 frequency scaling is an
  assumption.


## Review outcome (adversarial review, 2026-10-13) — supersedes the draft

**Blockers, adopted:**

1. **Never stack on `fp07_tau_scale`.** ARCTERX-22's production configs set
   SN194 {T1 0.30, T2 0.75}; the ratio is measured on spectra WITHOUT H2, so a
   correction on top corrects the relative part twice (SN194 chi1/chi2 1.03 ->
   ~0.64, inverted). Config validation REFUSES both keys on one instrument.
   The acceptance baseline is the default model without tau_scale; moving
   ARCTERX-22 off tau_scale raises its chiMean/Gamma by about the tau_scale
   geometric mean (x~1.4 for SN194), which must be stated with the change.
2. **Segment in time; check every file.** One correction per deployment made
   one half worse in 4 of 17 VMP pairs (all with placeholder or reused serials:
   probably undocumented swaps). The estimator builds a per-file timeline of
   the mid-band (15-30 Hz) ln ratio, segments it (change points well above the
   per-file precision, 0.075-0.155 ln), and writes one correction per stable
   segment with MANDATORY t_start/t_end. At chi time each file's own mid-band
   ratio is compared with its correction; |difference| > 0.3 ln -> not
   applied, recorded, counted.
3. **Match on (instrument, channel->serial map of BOTH channels, time span).**
   The correction is relative to the partner bead, serials can be
   placeholders ("T", "?"), and a channel swap would apply it inverted
   (e1 e2 = 1 doubles the log disagreement). Placeholder serials require the
   time span. A configured instrument whose file matches nothing is REFUSED
   (file error; no uncorrected product mixed into a corrected chi_NN).

**Majors, adopted:** the correction is stored IN each chi product (f_ref,
ln_e per probe, W_ref, f_hi, digest, segment, guard result) so the product
stands alone; `perturb/diag/render.py` and `rsi/window.py` reconstruct H2 from
those variables (and from `fp07_tau_scale_*`, which they ignore today);
`rsi-tpw chi` AND `pipeline` take `--fp07-response`; the config digest is the
SEMANTIC content, computed at load (moving or re-timestamping the file does
not re-version; chi, chi_binned and chi_combo re-version, diss does not);
spectra on different grids (100 vs 101 bins, 2 Hz grids) are interpolated
onto one log-f grid and every file is accounted for; f_hi comes from a
bootstrap interval (half-width < 0.1 ln) and the factor is HELD above it (the
ratio is monotone); the reference level is the lowest well-sampled bins, and
the flat gain ln G is reported — the acceptance target is chi1/chi2 -> G, and
|ln G| > ln 2 (an amplitude-loss signature) is refused; the speed scaling uses
the tau model itself (f -> f * tau(W)/tau(W_ref)), so `goto`'s speed
dependence and any other model stay consistent; gliders are binned on that
reduced axis; the anchor is recorded (geomean default; T1 or T2 selectable);
pairs with |ln R20| > 0.9 (beyond the fleet spread) are flagged as possible
failing beads; implied |H|^2 > 1 is clipped and flagged.

**Tests required:** constant e = c on one probe scales Method-1 chi by exactly
1/c (fails if only the fit or only the variance correction sees e); the
slow-bead sign (T2 = model x R gives e2 < 1 and a smaller |ln(chi1/chi2)|);
channels matched by name not index; factor(f, W) vs e_ref(f * tau(W)/tau(W_ref));
Method 1, iterative and MLE; refit on corrected products is idempotent;
render consistency; hashing (diss unchanged, chi re-versions, file move does
not).

**Scope:** the non-parametric shape buys 5-15% per probe over a single-pole
scale, the same size as unhandled drift, so segmentation and the per-file
guard come first; the shape is kept because it costs little once those exist.


## Implementation and acceptance (2026-10-13) — supersedes both sections above

### Design change: equalize the DATA, not the model

The draft and the review multiplied each bead's model |H|^2 by e_i, in the
fit AND in the variance correction. The acceptance run showed that this leaves
most of the response signature in chi, and the reason is the estimator: chi
Method 1 (and the iterative Method 2) computes

    chi = 6 kappa * Int(S_obs - N) dK * V_total / V_resolved,
    V_resolved = Int B(K; kB) |H(K)|^2 dK     (model shape, kB from epsilon)

so the response enters through a **model-weighted** integral. A relative
correction e_2/e_1 is then averaged with the Kraichnan model's weights rather
than the observed spectrum's, and whenever the observed spectrum is redder
than the model (most in low-epsilon windows) the correction is over-applied.

**Implemented instead:** each bead's observed spectrum AND noise spectrum are
divided by e_i before chi is computed (`l4_chi._process_l4_chi`); H2, the fit
model and the variance correction stay on the default tau model. The bead is
mapped onto the pair-average response, and the two beads' chi then differ only
by the flat gain G and sampling, **whatever the spectral shape**. For a pure
spectral fit (log-space or MLE) the two placements are mathematically
identical; only the moment estimators differ. The stored `spec_gradT` and
`spec_noise` stay raw (the estimator and the guard read them).

Verified three ways: (1) offline recomputation of Method 1 from the stored
spectra reproduces the stored chi to 1e-15 (base) and 1e-7 (corrected);
(2) a unit test with white spectra where probe 2 = probe 1 seen through e_2
gives chi_2 = chi_1 to 1e-9 under data equalization, and fails by 49% under
the model placement (mutation-checked); (3) the real-data table below.

### Acceptance, ARCTERX-2022 Interior, Method 1 (190-profile subset)

Corrections fitted on the full deployment's chi_00 with the final code
(4 segments: SN194 2, SN428 2; bands 0.9-99.8 Hz; no flags), rerun after the
code review. 184 of 189 files applied; 5 turned off by the per-file guard
(|delta| > 0.3; guard delta median +0.008, p5/p95 -0.184/+0.183). The "data / e" column is the final code
(S/N-weighted, below); the unweighted version gave -0.128 / -0.057 / 0.118 and
+0.082 / +0.036 / 0.103 for the same rows. Windows with both probes finite and 0.5 < fom < 2: SN194
2995, SN428 3055. ln(chi2/chi1), median:

| | base | model x e (draft) | data / e (built) | target ln G |
|---|---|---|---|---|
| SN194 all | -0.570 | +0.013 | **-0.142** | -0.140 / -0.144 |
| SN194 Q1..Q4 of K_max_ratio | -0.78 -0.60 -0.52 -0.41 | -0.11 -0.01 +0.04 +0.14 | -0.17 -0.15 -0.15 -0.13 | |
| SN194 Q1-Q4 | -0.367 | -0.251 | **-0.041** | 0 |
| SN194 per-window MAD-sd | 0.280 | 0.229 | **0.109** | |
| SN428 all | +0.321 | -0.012 | **+0.089** | +0.088 / +0.095 |
| SN428 Q1..Q4 | +0.47 +0.34 +0.26 +0.23 | +0.08 +0.01 -0.05 -0.07 | +0.12 +0.09 +0.08 +0.08 | |
| SN428 Q1-Q4 | +0.242 | +0.153 | **+0.030** | 0 |
| SN428 per-window MAD-sd | 0.193 | 0.161 | **0.100** | |

Geometric-mean chi of the pair moves by -0.007 (SN194) and -0.002 (SN428) in
ln: the anchor holds and the absolute level is untouched. The residual
ln(chi2/chi1) sits at ln G, the flat gain at ~1 Hz, which this correction
deliberately does not remove (calibration or diff_gain, unknown which bead).

For comparison, `docs/perturb/configuration.md` reports per-bead
`fp07_tau_scale` (model-side, fitted so integrated chi agrees) reducing the
quartile spread to 1.13x/1.11x (ln 0.12/0.10) on the full deployment; the data
equalization gives 1.04x/1.03x on this subset. Not strictly commensurable
(subset vs full; tau_scale also absorbs G, ending at 1.03x overall).

Varying the COMMON response (tau x 1-2.5) or kB (x 0.7-1.4) offline under the
model placement moved the level but left the Q1-Q4 trend unchanged
(-0.21..-0.29; +0.14..+0.18): the residual was the model-vs-data shape, not
the absolute tau.

### A pre-existing finding, not addressed here

Per probe, Method 1 chi exceeds a data-weighted variant (in-band response
divided out of the observed spectrum, model used only for the unresolved
tail) by +0.198 ln (SN194; p16/p84 -0.06/+0.54) and +0.116 (SN428), rising to
+0.26 in the highest K_max_ratio quartile. Neither estimator is established as
correct (dividing by |H|^2 up to 88 Hz amplifies noise-model error), but the
in-band response correction is model-weighted in every chi product today.
Worth an issue; it is independent of this branch.

### Acceptance, osu684 / MR433 Slocum glider, Method 2 (13 profiles)

The real perturb path: a base run, then the same config plus
`instruments.patched.fp07_response` (the key is the `.p` directory,
`fp07cal/patched`). profiles_00 and diss_00 were reused; chi, chi_binned and
chi_combo went to a new generation, as designed. Corrections: 4 segments, one
refused (band ends at 22.3 Hz, still after the band-edge fix, so a real
limit, not the bootstrap artifact) and covered by a not-applied record; per-file
mid-band spread 0.14-0.29 (VMP 0.08-0.12). Files in the refused span ran
uncorrected with `fp07_response_reason = refused at fit: ...`.

Segment from 2025-02-04 22:12 (ln R'(20 Hz) = -0.40, ln G = +0.01),
ln(chi2/chi1) by low-band window S/N:

| window S/N | base | data / e, unweighted | data / e, S/N-weighted (final) |
|---|---|---|---|
| top 5% | -0.379 | -0.115 | **-0.127** |
| 80-95% | -0.185 | +0.010 | **-0.008** |
| 50-80% | +0.004 | +0.206 | +0.111 |
| bottom 50% | +0.431 | +0.786 | **+0.477** |

In signal-dominated windows the correction lands on ln G. The low-S/N half
carries a **pre-existing probe-dependent noise-floor bias** (+0.43: T2's
spectrum exceeds its noise model by more than T1's), which unweighted
equalization amplified by scaling clipped noise residuals; the S/N weighting
removed most of that. Over all windows the median moves +0.15 -> +0.24,
because the base's apparent agreement in the middle band was the noise bias
cancelling the response error. Segment from 2025-02-08 (ln R'(20 Hz) =
+0.02): +0.27 -> +0.25, i.e. nothing to correct; its disagreement is the
noise floor too (+0.40 in the low-S/N half, +0.12..+0.16 above it).

Verdict: on this glider the correction is right where there is signal, and
the pair's agreement is limited by the noise model, not the response. The
runbook says to check the S/N split before trusting a glider correction.

### Other changes found by acceptance

- **Refused segments leave a not-applied record.** With review item B3 (no
  match is an error), one refused segment made the whole deployment unrunnable
  with corrections (osu684: a 22-Hz band in one segment). The estimator now
  writes `*_notapplied.nc` for each refused span (flag `not_applied`, reason in
  provenance); files there match it, run uncorrected, and carry
  `fp07_response_applied = 0` with `fp07_response_reason`. A refusal before a
  span exists (no start_time) still writes nothing.
- **Output names carry the span start to the minute.** Date-only names let
  two segments starting on one day overwrite each other (osu684 lost one of
  four). The CLI also refuses a name collision outright.
- **The guard's reason is recorded** (`fp07_response_reason`), as is the
  not-applied record's.
- **The factor is S/N-weighted** (`fp07_pair.signal_weighted`): ln e times
  w, w = 0 at smoothed S/N <= 2 rising to 1 at >= 10 (the S/N the factor was
  measured at). See the glider section; on ARCTERX-2022 it slightly improved
  the result.
- **Segments split only between distinct start_times.** Profiles cut from
  one .p share its start_time (the chi-time match key); a split inside such a
  group produced an empty span (ARCTERX-2022 SN194, 11 profiles at one start).
- **`_smoothed_snr` is edge-normalized**: the zero-padded running mean read
  S/N low in the first and last 3 bins, where the reference level is taken.

## Code review outcome (2026-10-13)

An adversarial review of the implementation (scripted checks, not a read-
through) found nine items. Each was verified against the code before fixing;
every new test was mutation-checked (it fails with its fix reverted).

1. **HIGH, fixed.** `rsi-tpw chi --fp07-response` never reached the output
   hash: rsi's canonicalizer keeps only keys in `DEFAULTS["chi"]`, so a
   corrected run reused the uncorrected run's `chi_NN`. `fp07_response` is
   now a chi default, dropped from the canonical view when unset (every
   existing hash unchanged), and hashed as content digests when set (also
   from a `[chi]` config value). `rsi-tpw pipeline` refuses the key.
2. **MEDIUM, fixed.** Matching never compared the instrument, so a
   placeholder serial matched another unit's correction (an SN194 correction
   applied itself to SN479, whose beads read `T`). Both sides now derive
   (instrument, serials) from the source's configuration string through
   `fp07_pair.pair_identity`, and the instrument must be equal.
3. **MEDIUM, fixed.** One lumpy bootstrap bin ended the band at a few Hz and
   refused clean small segments (7 of 30 seeds at 12 files). The half-width is
   now median-smoothed over the +/-2 nearest *estimated* bins (index
   neighbors are empty at low f), and the weighted median averages the two
   middle values at an exact half: 0 of 30 refused at 10 and at 12 files.
4. **MEDIUM-LOW, fixed.** Groups of one instrument whose serials changed
   (real <-> placeholder) could both match a boundary file through their
   one-hour margins. `select` now prefers the record whose serials equal the
   file's; equal rank is still an error.
5. **LOW-MEDIUM, fixed.** The guard was centered on the smoothed curve, not
   on the per-file statistic it recomputes (fit files sat +0.04..+0.08 off
   center). `mid_lnR` is now the median of the segment's per-file statistics.
6. **LOW-MEDIUM, fixed.** An unreadable configuration at chi time silently
   became `?` serials (a time-only wildcard). It is now an error when
   `fp07_response` is configured.
7. **LOW, partly fixed.** Corrections are now cached per process (path,
   mtime, size), so perturb workers stop reopening every file per profile.
   Not done: verifying at chi time that the applied content equals the hashed
   content. Each product records its `fp07_response_digest`, so an in-place
   refit during a run is detectable afterwards.
8. **LOW, fixed.** `fp07-pair fit` prints absolute paths; `fp07_response`
   follows perturb's path convention (`<CONFIG_DIR>/...` or relative to the
   working directory, previously relative to the config); a relative `source`
   in a chi product is also looked up against the product's parent
   directories.
9. **LOW, documented.** The interactive viewers (`rsi/window.py`) do not
   apply the correction (nor `fp07_tau_scale`); `rsi-tpw pipeline` refuses it.

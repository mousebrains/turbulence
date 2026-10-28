# `fp07-pair` runbook

Per-bead FP07 response equalization. Two FP07 beads on one probe head
respond differently (on ARCTERX-2022 the T2/T1 power ratio at 20 Hz was 0.47
on SN 194 and 1.43 on SN 428), and chi gives both the same model response.
The two chi estimates then disagree by a factor that **trends with
`K_max_ratio`**, the signature of a response error. `fp07-pair` measures each
bead's response *relative to its partner* from the ratio of their gradient
spectra, a measurement that needs no turbulence model, and chi applies it.

```
perturb run (no fp07_response) ─► chi_NN/*.nc ─► fp07-pair fit ─► fp07pair_*.nc + report
                                                                      │
              instruments.<dir>.fp07_response: [...] ◄────────────────┘
                                │
perturb run ─► chi_NN+1 (diss and profiles reused)
```

What it does **not** do:

- **It does not set the absolute response.** The pair's geometric-mean
  response stays the default tau model (`lueck`, 10 ms at 1 m/s). The absolute
  level is not identifiable from in-situ spectra (see
  `docs/fp07_tau_prefactor_PLAN.md`).
  The pair's geometric-mean chi moved by less than 1% on ARCTERX-2022.
- **It does not remove a flat gain.** The ratio at ~1 Hz (`ln G`) is a
  calibration or `diff_gain` difference, and nothing says which bead is wrong.
  After correction, `ln(chi2/chi1)` sits at `ln G`, not at 0.
- **It does not rescue a failing bead.** A failing bead loses microscale
  amplitude, which no response factor models; `|ln G| > ln 2` is refused as
  that signature.

## 1. Produce chi without it

Run perturb (or `rsi-tpw chi`) as usual, **without `fp07_tau_scale`** on the
instrument. The estimator reads only `spec_gradT`, `spec_noise`, `F`, `speed`,
`start_time` and the `source` attribute (for the bead serials), all of which
every chi product carries, so the method and fit settings do not matter here.

## 2. Fit

```bash
fp07-pair fit 'results/chi_00/*.nc' -o fp07pair/
```

One NetCDF per (instrument, both channels' bead serials, time segment), plus
`fp07pair_report.txt`. Per pair the estimator:

1. computes, per window and frequency bin, `ln((S2-N2)/(S1-N1))`, gated on a
   running mean of S/N >= 10 on **both** probes;
2. maps every window onto one reduced frequency axis at the pair's median
   speed `W_ref`, through the tau model (`f -> f * tau(W)/tau(W_ref)`);
3. builds a per-file timeline of the mid-band (15-30 Hz) ratio and splits it
   where the pair changed (a swapped probe, drift). Splits fall only between
   **different** `start_time`s, because every profile cut from one `.p` file
   carries that file's start, and that is the key a correction is matched on;
4. per segment: takes the median over files, references it to the lowest
   well-sampled bins (`ln G`), and sets the upper band edge `f_hi` where a
   bootstrap over files says the curve is known to +/-0.1 in ln.

Read the report before using anything:

| field | what to look for |
|---|---|
| segments | Several per deployment is normal (SN 194: 2; SN 428: 2; the osu684 glider: 4). A segment boundary at a known service or swap is a good sign. |
| `band` | `f_hi` well above 30 Hz. Above `f_hi` the factor is held constant. |
| `ln G` | Small (<0.2). Large values mean a calibration problem, not response. |
| `ln R' at 20 Hz` | The fleet spread is about +/-0.3. `|ln R'| > 0.9` is flagged `outlier`, possibly a failing bead. |
| per-file mid-band spread | 0.08-0.16 on VMPs. Larger means the ratio is not stable file to file (glider: 0.14-0.29). |
| REFUSED | A segment with too few files, too short a band, or an amplitude signature. A `*_notapplied.nc` record covers its span (see §4). |

`fp07-pair show fp07pair/*.nc` prints what a file contains.

## 3. Configure

```yaml
instruments:
  SN194:                     # the RAW DIRECTORY NAME of the .p files
    fp07_response:
      - <CONFIG_DIR>/fp07pair/fp07pair_SN194_T1592-T2005_20220321T0359Z.nc
      - <CONFIG_DIR>/fp07pair/fp07pair_SN194_T1592-T2005_20220328T2114Z.nc
```

- **List every file the fit wrote for that instrument**, the not-applied
  records included. A file whose instrument, channels, bead serials and start
  time match **no** listed correction is an error, never a silently
  uncorrected product. Instrument and serials are read from the source's
  configuration string on both sides; a file whose configuration cannot be
  read is an error too, never a wildcard.
- **The instrument must match.** A placeholder bead serial (`T`, `?`, as on
  SN 479) falls back to the time span, but never across instruments. Where a
  serial group's one-hour outer margin reaches a neighboring group of the same
  instrument (serials edited mid-deployment), the record whose serials equal
  the file's wins.
- The key is `p_path.parent.name`, **not** the serial: the osu684 files live
  in `fp07cal/patched/`, so the key there is `patched`. An unmatched
  instrument key is silently ignored by perturb (a pre-existing behavior), so
  verify the result (§5).
- Paths follow perturb's convention: `<CONFIG_DIR>/...` resolves against the
  config's directory, a plain relative path against the working directory.
  `fp07-pair fit` prints absolute paths. Each file is hashed by **content**:
  refitting re-versions chi, chi_binned and chi_combo; moving or renaming the
  file does not; epsilon never re-versions.
- **`fp07_response` and `fp07_tau_scale` are mutually exclusive** (config
  load refuses both). The correction is measured on spectra without any tau
  scale, so stacking them corrects the relative part twice. Moving an
  instrument off `fp07_tau_scale` changes its absolute chi by roughly the
  geometric mean of the scales. ARCTERX-2022 production used SN 194
  `{T1: 0.30, T2: 0.75}` and SN 428 `{T1: 0.75, T2: 0.45}`; dropping them
  raises SN 194's chi (and Gamma) by about 1.4x. State that with the change.

`rsi-tpw chi ... --fp07-response FILE` (repeatable, or `[chi]
fp07_response` in the rsi config) does the same outside perturb, and versions
its `chi_NN` by the corrections' content. `rsi-tpw pipeline` computes chi on
its own path and **refuses** the key rather than ignoring it. The interactive
viewers (`quick_look`, `diss_look`, via `rsi/window.py`) recompute chi per
window without the correction (and without `fp07_tau_scale`), so viewer chi
can differ from product chi on a corrected instrument.

## 4. What chi does with it

Each bead's observed spectrum and noise spectrum are divided by its factor
`e_i(f, W)` before chi is computed; the model, the fit and the variance
correction keep the default response. Two details matter:

- **The data, not the model.** For a pure spectral fit the two placements are
  identical. But Method 1 and the iterative Method 2 apply the response
  through a resolved-variance correction weighted by the Kraichnan model,
  which over-applies a relative correction whenever the observed spectrum is
  redder than the model. Multiplying the model left most of the `K_max_ratio`
  trend on ARCTERX-2022; dividing the data removed it.
- **Only where the signal dominates.** `ln e` is weighted by the bin's S/N:
  in full at S/N >= 10 (where the factor was measured), not at all below 2.
  A near-floor window's integrand is mostly clipped noise residual and
  noise-model error, which the bead did not shape. Scaling it by `1/e`
  inflated the slow bead's chi in the low-S/N half of the osu684 windows.

Per file, before applying it, the file's own mid-band ratio is compared with
the correction's. A difference > 0.3 in ln (a swap or drift since the fit)
computes that file **without** it. Every chi product records what happened:

| attribute | meaning |
|---|---|
| `fp07_response_applied` | 1 or 0 |
| `fp07_response_reason` | why not: `guard: ...` or `refused at fit: ...` |
| `fp07_response_digest`, `_span`, `_serials`, `_channels`, `_band`, `_W_ref`, `_anchor`, `_tau_model`, `_guard_delta` | which correction and how well the file agreed |
| variables `fp07_response_f`, `fp07_response_ln_e` | the factor itself, so the product stands alone |

`spec_gradT` and `spec_noise` stay raw. `perturb-diag` draws the model
through the same factor.

## 5. Verify

- Count `fp07_response_applied` = 0 and read the reasons. A handful of guard
  rejections is normal (5 of 189 on ARCTERX-2022).
- Compare `ln(chi2/chi1)` by `K_max_ratio` quartile before and after. After
  correction the quartiles should be flat and sit near `ln G`.

- **On a glider, also split by window S/N.** On osu684 the correction landed
  on `ln G` in signal-dominated windows (top 5% of S/N: -0.38 -> -0.13;
  80-95%: -0.19 -> -0.01), but the low-S/N half carried a pre-existing
  probe-dependent noise-floor bias (+0.43) that no response correction can
  remove, and the overall median moved away from `ln G`. If the low-S/N
  windows dominate, the pair's disagreement is the noise model's, not the
  response's.

ARCTERX-2022 Interior, Method 1, 190-profile subset (median over windows):

| | before | after | `ln G` |
|---|---|---|---|
| SN 194 `ln(chi2/chi1)` | -0.570 | -0.142 | -0.140 / -0.144 |
| SN 194 quartile spread Q1-Q4 | -0.367 | -0.041 | |
| SN 194 per-window MAD-sd | 0.280 | 0.109 | |
| SN 428 `ln(chi2/chi1)` | +0.321 | +0.089 | +0.088 / +0.095 |
| SN 428 quartile spread Q1-Q4 | +0.242 | +0.030 | |
| SN 428 per-window MAD-sd | 0.193 | 0.100 | |

The full record, including the design alternatives that were tried and
rejected, is `docs/fp07_pair_response_PLAN.md`.

## 6. Each bead's time constant for the FP07 sheet (`fp07-pair tau`)

The same ratio, fitted with a single pole per bead, gives each bead a time
constant, which is recorded in the FP07 calibration sheet next to its
Steinhart-Hart coefficients. Tau is carried per bead because the beads
scatter more than any change to the default would move them (factor ~1.9,
geometric 1 sd, about a fleet median of ~12 ms; `docs/fp07_tau_prefactor_PLAN.md`
§7 and "Fleet results"), and because the electronics are much faster than the
bead (the PI, 2026-10-28; the anti-alias filters sit near 110 Hz).

```bash
fp07-cal registry show T1592 --json        # the entry's instrument, first, last
fp07-pair tau 'VMP_results/chi_00/*.nc' --instrument 194 \
    --first 2022-03-21T04:59:26Z --last 2022-03-30T18:45:16Z -o fp07tau/
fp07-cal registry response fp07tau/fp07tau_SN194_T1592-T2005_20220321T0459Z.json \
    --bead T1592 --deployment "ARCTERX-2022 Interior"      # and again for T2005
```

Use the sheet entry's own span, so the tau describes the same stretch of the
same pair as the calibration. A file belongs to the span when its
`start_time` does, and the instrument and bead serials come from each file's
source configuration, never from its name, so one directory may hold several
instruments (ARCTERX-2025 Interior holds SN142 and SN194). Each bead pair in
the span gets its own record: SUNRISE 2021 Pelican's T1 bead T1592 had T1593
and then T2005 as its partner.

### What it computes

Per bead pair, from the chi products' raw `spec_gradT` and `spec_noise`:

1. the pooled curve exactly as `fp07-pair fit` builds it (§2): per window and
   bin `ln((S2-N2)/(S1-N1))` behind the S/N >= 10 gate on both probes, mapped
   to reduced frequency at the pair's median speed `W_ref`, per-file medians,
   their count-weighted median, a 3-bin running median;
2. that curve read at fixed points 1, 2, ..., 25 Hz, and
   `ln R = ln G + ln(1 + (2 pi x tau1)^2) - ln(1 + (2 pi x tau2)^2)` fitted there
   by least squares, ln G free. Fixed points keep the fit's weighting
   independent of the products' FFT resolution (1 Hz on ARCTERX-2022, 2 Hz on
   SUNRISE) and of the speed spread. A point is read off the curve only
   between measured bins at most 2.5 Hz apart, never across a hole in the band;
3. 16-84% intervals from 200 bootstrap resamples of whole **recordings**
   (the chi files sharing one `.p` start time): profiles cut from one
   recording are not independent draws, so resampling them one by one would
   make the intervals too narrow;
4. the model-free power ratio at 10 and 20 Hz, relative to the reference
   level (the lowest four usable bins below 4 Hz, the `ln G` that `fp07-pair
   fit` reports), with its bootstrap interval;
5. the same for the first and second halves of the files in time, split
   between different start times and fitted at the same `W_ref`. A bead whose
   halves differ by more than twice their combined 1-sigma in ln tau is
   flagged `drift`; a pair whose halves differ that much in the model-free 20 Hz
   ratio is flagged `ratio-shift` (more sensitive, but it cannot say which bead
   moved: on the osu685 MicroRider in 2025 the ratio went 1.23 to 0.81 while
   neither bead's tau moved beyond its own 2-sigma test);
6. dead beads: a chi file in which a bead's spectrum is identically zero over
   1-10 Hz in at least half its windows (a dead or railed channel) is left
   out and listed, and the pair flagged `dead-bead`;
7. a noise-model check (a diagnostic): each probe's observed/model spectrum
   at 60-100 Hz in the quietest 10% of the windows it can use (S/N >= 10 over
   1-10 Hz), flagged outside 0.67-1.5.

Each pair writes `fp07tau_<instrument>_<beads>_<start>.json` (the record) and
a `.txt` report.

### What it is not

- **A single-pole-equivalent value.** The pole form is not determined: a
  double pole fits the ratio as well and moves the 3-dB point 30-40% lower.
- **Not an absolute measurement.** The level depends on the response family
  and the band, and no external reference anchors it; the in-situ anchor
  failed its synthetic check (`docs/fp07_tau_prefactor_PLAN.md`). Compare beads
  and deployments with it. The 10 and 20 Hz power ratios are the robust part.
- **Not consumed by chi.** Nothing reads the sheet's tau yet. Using it is the
  follow-up: a per-bead tau in perturb (a generalization of
  `instruments.<dir>.fp07_tau_scale`) or a patch step. `fp07_tau_scale` and
  `fp07_response` are mutually exclusive, and a per-bead tau would be a third
  way of setting the same response, so it must exclude both.

### When a pair identifies no tau

Matched beads give a flat ratio, and a single pole then fits any pair of
equal taus; a failing bead loses amplitude or goes dead, and neither is a
time constant. The rule (`fp07_bead_tau.RULE`) was set on synthetics; what was
decided before the first real measurement and what after is recorded below
the validation table. A pair's taus are identified only when:

1. there are >= 16 chi files from >= 5 recordings, and >= 12 of the 25 fit
   points are measured, reaching >= 20 Hz;
2. the flat gain is within a factor 2 (`|ln G| <= ln 2`, the `fp07-pair fit`
   amplitude refusal);
3. the beads differ beyond bootstrap noise: `|ln R'(20 Hz)|` is at least 3
   half-widths of its 16-84% interval and at least 0.15 (16% in power);
4. each tau and its 16-84% interval lie inside the fit bounds (0.5-200 ms,
   with a 1.05x margin), and each interval spans at most a factor 1.5;
5. at least 90% of the bootstrap replicates fit;
6. neither bead is dead in any chi file of the span (a half is refused only
   when a dead file falls in its time).

Otherwise the record's tau is null and the reasons are listed; the ratios
are still recorded. A half is judged by the same rule, so a short deployment
can have a tau overall and none per half. To measure a bead before it died,
give a span that ends before the dead files; the refusal is deliberate, since
a bead's response usually degrades before it dies (on SUNRISE 2021 Pelican
the T1592/T1593 ratio at 20 Hz went from matched to 0.71 in the days before
T1593's channel went dead). `drift` is flagged when a bead's
halves differ by more than twice their combined 1-sigma (in ln tau).

**Synthetic validation** (`scripts/fp07_tau/bead_tau_synth.py`): synthetic chi
products with known single-pole beads, run through `measure` exactly as real
ones. Every window draws its own speed (5% scatter), Batchelor wavenumber,
level and noise floor, and the spectral shape is deliberately non-universal (a
low-k excess and a mid-band dip that vary window to window). 120 files of 60
windows, VMP speed 0.95 m/s, unless stated:

| case | truth tau1/tau2 [ms] | recovered [16-84%] | verdict |
|---|---|---|---|
| matched beads | 10/10 | 0.5/0.6, intervals to the bounds | **not identified** (beads not distinguishable) |
| near-matched | 11/10 | 11.1 [8.7-14.7] / 10.1 [7.8-13.5] | **not identified** (intervals x1.7) |
| | 12/10 | 12.0 [11.0-13.7] / 10.0 [9.2-11.5] | identified |
| ARCTERX-22 SN194-like | 13/19.5 | 13.1 [12.8-13.9] / 19.5 [19.0-20.6] | identified |
| | 25/10 | 25.1 [24.6-25.5] / 10.2 [9.9-10.3] | identified, `outlier` |
| glider, W 0.3 m/s | 20/15 | 21.0 [19.6-22.2] / 16.0 [14.9-16.9] | identified |
| noise 1.5x model on one probe, peak S/N 60 | 13/19.5 | 13.4 [12.8-14.2] / 19.6 [18.8-20.7] | identified |
| noise 0.67x model on one probe, peak S/N 60 | 13/19.5 | 13.8 [13.0-14.5] / 20.3 [19.3-21.3] | identified, `noise-model` |
| peak S/N 20 | 13/19.5 | 12.5 [10.9-13.6] / 18.4 [16.5-19.9] | identified |
| 2 Hz FFT bins (SUNRISE), 15 windows/file | 13/19.5 | 13.5 [11.5-15.1] / 20.1 [17.2-22.5] | identified |
| 8 files | 13/19.5 | 15.9 [13.7-17.9] / 23.2 [20.0-25.9] | **not identified** (8 files < 16) |
| 16 files | 13/19.5 | 14.4 [12.9-16.1] / 21.0 [18.9-23.2] | identified |
| failing bead (gain 0.3) | 13/19.5 | 12.9 / 19.4 | **not identified**, `amplitude` |
| drift between halves | 10.8/17.7 → 16.1/22.0 | halves 10.7/17.3 → 16.7/22.8 | identified, `drift` |
| double-pole truth, single pole fitted | DP 8/5 | 9.4 [9.1-9.7] / 3.2 [2.8-3.5] | identified: single-pole *equivalent*, not the double pole |

Over 20 seeds per case (100 bootstrap replicates; the bootstrap resampling
whole recordings, the files sharing a start time):

| case | identified | 16-84% interval covers the truth (tau1 / tau2; nominal 0.68) | median recovered [ms] | drift false alarms |
|---|---|---|---|---|
| 13/19.5, 16 files = 16 recordings | 20/20 | 0.85 / 0.85 | 13.7 / 20.3 | — |
| 13/19.5, 60 files in 6 recordings | 20/20 | 0.75 / 0.65 | 13.2 / 19.6 | — |
| 13/19.5, 120 files in 12 recordings | 20/20 | 0.50 / 0.70 | 13.3 / 19.7 | 2/20 |
| 13/19.5, 120 files in 5 recordings | 20/20 | 0.45 / 0.55 | 13.3 / 19.7 | — |
| as above, 12 recordings that differ (own speed offset and noise-model error, sd 0.3) | 20/20 | 0.60 / 0.75 | 13.3 / 19.7 | 3/20 |
| 13/19.5, 2 Hz FFT bins, 120 files of 15 windows | 20/20 | 0.70 / 0.90 | 13.7 / 20.6 | 0/18 |
| glider 20/15, W 0.3, 60 files | 20/20 | 0.60 / 0.50 | 21.2 / 16.1 | 1/20 |
| 11/10, 60 / 120 files | 1/20, 0/20 | (the one identified covered) | — | — |
| matched 10/10, 16 / 60 / 120 files | 0/20 each | — | — | — |

With 20 seeds the coverage itself is uncertain by about ±0.1, but three
limitations are real. With few recordings (5-12) the interval under-covers
(0.45-0.70): treat it as a lower bound on the sampling error. At a glider's
0.3 m/s the taus read 6-7% high and the interval covers the truth about
half the time; with 2 Hz FFT bins they read ~5% high. All three are well
inside the bead-to-bead spread the sheet exists to record, and smaller than
the estimator and band dependence of the absolute level (below). A noise model
1.5x off on one probe moves tau by less than its interval, because the S/N >=
10 gate keeps such bins out (`noise-model` flags it all the same).

On synthetics the recording bootstrap is no better than resampling profiles:
the generator's recordings are nearly independent of each other even when
they differ. Real recordings are not. On ARCTERX-2022 Interior it widened the
intervals 2.5x (T1592 [16.4-17.0] to [15.9-17.7] ms) with the same estimates,
which is between-recording variance the profile bootstrap had hidden.

What was decided when. The first rule (minimum 8 files, a 0.05 floor,
disjoint 16-84% intervals for `drift`, the fit made on the log-grid bins as
filled) was set on synthetics before any real product was measured; a
multi-seed study then raised the minimum to 16 files (8 files covered the
truth 50-56% of the time) and made `drift` a 2-sigma test (disjoint 1-sigma
intervals false-alarmed on 4 of 20 stationary pairs). One real measurement
(ARCTERX-2022 Interior SN194) was then made, and its diagnosis (below) showed
that the log-grid weighting depends on the products' FFT resolution and speed
spread, so the fit moved to fixed 1 Hz points; and the re-validation showed
11/10 ms pairs occasionally identified with intervals missing the truth by
35-50%, so the floor became 0.15. Both changes rest on synthetic evidence and
on the cross-deployment comparability the sheet needs, not on any real tau;
the real pair seen meanwhile sits far from every threshold (|ln R'(20 Hz)| =
0.69). A code review and the first SUNRISE runs then added the rest: the
bootstrap resamples recordings and needs 5 of them (profiles of one `.p` are
not independent); a dead bead is refused (SUNRISE 2021 Pelican); a fit point
is never read across a hole in the band; the noise check ignores dead windows,
which had read as a noise model 30x too high. These are data-quality and
statistical corrections, not thresholds tuned to a tau.

### Acceptance: ARCTERX-2022 Interior SN194 (T1592/T2005)

`docs/fp07_tau_prefactor_PLAN.md` published, for this pair, single-pole taus
of 13.0 [11.9-14.0] and 19.5 [17.8-20.9] ms at W 0.95, halves 10.8/17.7 then
16.1/22.0. On the same chi_00 products (536 files, 37,212 windows), in ms:

| estimator | T1592 | T2005 | halves (T1592/T2005) |
|---|---|---|---|
| analysis-era first pass, spectral sums over a gated 1-26 Hz band (`run_fleet.py`), as published | 13.0 [11.9-14.0] | 19.5 [17.8-20.9] | 10.8/17.7 → 16.1/22.0 |
| the same code rerun on the products today | 13.0 [11.9-14.0] | 19.5 [17.8-20.9] | 10.8/17.7 → 16.1/22.0 |
| analysis-era median of log ratio, 1-25 Hz (`analyze_cache.py` `b25_p1`) | 17.4 [17.1-17.6] | 25.6 [25.2-25.9] | 15.3/23.7 → 21.0/28.7 |
| `fp07-pair tau` (recording bootstrap) | 16.7 [15.9-17.7] | 24.8 [24.0-25.7] | 14.8/23.0 → 20.3/28.1 |

The first pass reproduces exactly, so the products have not changed since it
was run; the disagreement is the estimator. Spectral sums let a few energetic
windows dominate (the PLAN's own later finding), and the median of the log
ratio over the same files reads 1.3x higher. `fp07-pair tau` agrees with the
analysis-era median-of-log fit to 3-4%, inside its own interval (the
analysis-era intervals resampled profiles, so they are narrower than
resampling recordings: 30 recordings here). Walking from one to the other one step
at a time (point estimates): the shared S/N gate changes nothing; the reduced-
frequency axis with the log grid's bins as filled reads 7-9% higher; read at
fixed 1 Hz points it returns to within 4%. The tau ratio (1.47-1.50), the
pair's power ratio (T1592 2.0x T2005 at 20 Hz relative to the reference band),
and the slowing between halves (T1592 +37%, T2005 +22%; the first pass gave
+49%/+24%) agree across every estimator. The absolute level moves by up to
1.3x with the estimator and by 1.5x with the band (fitted over 1-60 Hz:
10.9/16.7 ms). The bootstrap intervals above are sampling intervals only; they
do not include that. The noise model checks out on both probes (observed/model
1.02 and 1.05 at 60-100 Hz in the quietest windows).

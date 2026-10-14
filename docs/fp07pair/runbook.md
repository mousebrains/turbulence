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
  `docs/fp07_tau_prefactor_PLAN.md` on branch `analysis/fp07-tau-prefactor`).
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

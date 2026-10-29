# FP07 time-constant prefactor: plan

Status: CLOSED 2026-10-13. Decision (Pat): **keep the `lueck` default
(10 ms * W^-1/2)** until a more significant result; the fleet's single-pole-
equivalent median is 12 ms, and the absolute level is not identifiable in
situ. The per-bead correction the conclusions point to was built:
`fp07-pair` / `instruments.<dir>.fp07_response` (PR #217). Section 7's other
branch ("per-bead tau goes into the #189 registry") was built 2026-10-28:
`fp07-pair tau` measures each bead's single-pole-equivalent tau and the FP07
calibration sheet carries it (`docs/fp07pair/runbook.md` §6). No lab measurement
will be made, and the micro-conductivity route is closed (Starbuck is the only
deployment with one, and its `C1_dC1` behaves as a temperature sensor, not a
conductivity cell: section "Starbuck micro-conductivity", revisited 2026-10-28). Sections are kept in the order they
were written; later sections supersede earlier ones where they differ (the
section-3 method was BLOCKED on VMP data, and the "a ~ 1" shape claim was
WITHDRAWN; see "Fleet results and conclusions").

## 1. The question

perturb and rsi-tpw correct every temperature-gradient spectrum for the FP07
response with the default `lueck` model: single pole,
`|H(f)|^2 = 1/(1 + (2 pi f tau)^2)`, `tau = 10 ms * W^-0.5`. Nothing fits
that prefactor (`fp07_tau_scale` is an optional multiplier, default 1), so it
sets the correction in nearly every chi product. It has no known source
(PR #216): Lueck et al. (1977) does not contain it, the -1/2 exponent is
Vachon & Lueck (1984) / Hill (1987), and ODAS `quick_look.m` builds the same
law as a double pole with `f_3dB = 25 sqrt(W)` Hz, a response 1.6x faster at
the 3-dB point.

**Goal:** the best-supported FP07 response for this fleet — pole form,
prefactor, speed exponent, and the sensor-to-sensor spread — with an
uncertainty, and a decision on the default.

**Why it matters:** chi responds to tau as d(log chi)/d(log tau) ~ 0.76
(measured on CASPER-West). A 1.6x change in the response moves chi by tens of
percent, in every campaign at once.

## 2. What is already known (do not re-litigate)

From [[project_arcterx2022_interior]], [[project_gamma_absolute_tension]],
[[reference_fp07_bead_attenuation]], issues #189 and #191:

- **Absolute tau from a spectral-shape fit is blocked by the model.** The
  Kraichnan fit leaves a residual structured by ~2x across `k/kB` (#191:
  median obs/model 0.92, 0.45, 0.53, 0.76, 0.94 by `k/kB` band on
  CASPER-West), in the band where noise is negligible. Two tau estimators
  that share that model agreeing (16-19 ms) was withdrawn as corroboration.
- **Method-1/Method-2 kB-crossing fits** (`fit_tau_method12.py`) gave
  s = 1.3-2.9 (tau 13-29 ms at 1 m/s) over 10 beads on 4 instruments: SLOWER
  than the default, opposite in sign to the ODAS model. Same caveat: they use
  the Kraichnan shape.
- **Inter-probe fits identify only tau(T1)/tau(T2)**, never the common level.
- **A failing bead loses amplitude, not bandwidth** (CASPER MR SN 134); no tau
  models it. Such beads must be excluded, not fitted.
- **The epsilon side is clean** (obs/Nasmyth 0.9-1.1 through the peak), so
  shear epsilon can anchor kB.

Literature: see section 8 (survey pending).

## 3. Method: separate the response from the spectral shape

The blocker is that a tau fit needs a spectral model, and the model is
wrong by ~2x in shape. The two errors live on different axes:

- a **spectral-shape** error is a function of `x = k/kB` (wavenumber relative
  to the Batchelor wavenumber);
- a **response** error is a function of **frequency** `f = k W`.

Across windows, `kB` (from the shear epsilon) and `W` both vary, so a given
`x` is seen at many `f`. Model the log of each window's observed spectrum,
where it is well above noise, as

    log S_obs(k; w) = a_w + g(log x) + h(log f) + e

- `a_w`: one amplitude per window (absorbs chi, and the constant log-bias of a
  chi-squared spectral estimate, which is the same for every window at fixed
  DOF);
- `g`: the universal spectral shape, **non-parametric** (bins in log x) — no
  Batchelor or Kraichnan form is imposed, so #191's misfit is absorbed rather
  than aliased into tau;
- `h`: the **residual response** relative to the response already divided out
  (or the full response, if the observed spectrum is uncorrected).

**Identifiability.** `log f = log x + log(kB W)`, so `g` and `h` trade only a
common power law (`+alpha log x`, `-alpha log f`), which the per-window
constant absorbs. That tilt is pinned by requiring `h = 0` for `f` below a
floor `f0`. No floor is safe for every candidate: if the fleet really is
s ~ 3 (tau ~ 50 ms at 0.3 m/s), |H|^2 is already 0.9 at 1 Hz. So `f0` is
varied (0.5, 1, 2 Hz) and the bias it causes is measured on synthetics; the
parametric fit (below) carries its own low-f flatness and is the primary
estimate. Two fits:

1. **Non-parametric `h`** (bins in log f): the empirical response shape, no
   pole form assumed. This is the headline diagnostic.
2. **Parametric `h`** = `log |H(f; tau, n)|^2`, n = 1 or 2 poles, tau scanned:
   gives tau with a profile-likelihood interval, and which pole form fits.

`kB = (1/2pi) (eps / (nu kappa_T^2))^(1/4)` [cpm] from the SHEAR epsilon
(chi Method 1), never a fitted kB (circular: it was fitted with the old tau).
An epsilon error rescales `x` by a constant factor, which `g` absorbs; random
epsilon error smears `g` (a loss of power, not a bias in `h`, to be confirmed
by synthetics).

## 4. Validation before any real result (synthetic injection, null tests)

Using the REAL `(kB, W, K, noise)` of each dataset, synthesize spectra
`S = chi * G(x) * |H_true(f)|^2 + noise`, with chi-squared estimation noise
at the dataset's DOF, then run the identical fit:

- **Recovery:** `H_true` single pole 5, 10, 20 ms and double pole 3, 6 ms;
  recover tau within the stated interval, and the right pole form.
- **Shape immunity:** distort `G` by the #191 residual (x0.45 at x = 0.3, x1.3
  at x = 1) — recovered tau must not move.
- **Null:** `H_true = 1` — recovered tau ~ 0, `h` flat.
- **Epsilon error:** multiply eps by 2 (constant) and by lognormal noise of
  0.3 dex — tau must not move beyond its interval.
- **Speed error:** W x 1.1 (an EM zero offset) — quantify the tau bias; it
  enters as `f` mis-scaled.

Invariance on real data (each must leave tau within its interval):

- the SNR gate (obs/noise > 3, 5, 10) and the upper frequency limit
  (60, 70, 80 Hz);
- the x-bin width and the low-f anchor;
- splits by epsilon (tercile), by Re_b, by profile direction, by probe, and
  by time (first/second half of the deployment: a drifting tau is fouling).

## 5. Data

| set | platform | W [m/s] | kB source | windows | status |
|---|---|---|---|---|---|
| Kemedukl | VMP-250-IR | 1.15-1.40 | Method 1 in product | 2,927 | ready |
| SN479 (repo `VMP/`) | VMP-250-IR, ARCTERX-2025 Wake | ~0.65 | rerun chi Method 1 | ~? | to process |
| Rutgers RU33 | Slocum MR1000EM | 0.25-0.35 | diss_00, windows aligned | 10,327 | ready |
| Hydrographer Bank | moored MR1000RDL-EM | 0.05-0.33 | diss_00, time stamps differ | 2.16 M | check alignment |
| SeaChest campaigns | ASTRAL SN92 (0.8), CASPER-West VMP+MR, ARCTERX, Taiwan, Tanaka | 0.3-0.8 | rerun Method 1 | large | SeaChest unmounted |

The Hydrographer MR alone spans 0.05-0.33 m/s on the SAME beads, so it can
measure the speed exponent without bead-to-bead confusion — but below
~0.1-0.15 m/s Taylor's hypothesis fails (turbulent velocity ~ mean flow), so
it is gated on W and on the turbulence-intensity ratio where available.

Per bead, per deployment: tau is fitted for each thermistor separately (beads
differ, #189); the fleet answer is the distribution across beads.

## 6. Confounders and how each is handled

| confounder | acts on | handling |
|---|---|---|
| anti-alias filters (2 x 4-pole Butterworth ~110 Hz), not corrected in `spec_gradT` | f | fit only f <= 70 Hz (|H_AA|^2 > 0.999), and a variant including the known AA response |
| electronics noise | high k | SNR gate; second variant adds the noise model instead of gating |
| first-difference and bilinear corrections | f | already applied in `spec_gradT` (`l3_chi`); verify on the product |
| Goodman coherent-noise removal | specific f | compare with Goodman off on one dataset |
| speed error (EM zero offset, flight-model AOA) | f scale | VMP pressure-derived W is the reference; quantify via synthetics (sec. 4) |
| shear-epsilon error (sens, isotropy at low Re_b) | x scale | absorbed by g as a constant; Re_b split |
| bead attenuation / fouling | amplitude, drift | time split; exclude beads with a known failure signature |
| spectral-shape dependence on Re_b or on the instrument | g not universal | fit g per dataset (and per Re_b class), never pooled across instruments |
| low-W Taylor failure | everything | W gate on the moored MR |
| **low-k temperature excess** (2-5x above Kraichnan at small x in every campaign; finestructure?) — a function of k itself, neither x nor f | at fixed W, k and f are proportional, so it ALIASES INTO h | a `k_min` cut (2, 4, 8 cpm) as an invariance test; across platforms W varies 25x, which separates a k-dependence from an f-dependence; a three-way variant `c(log k) + g(log x) + h(log f)` on the pooled multi-speed set |

## 7. Decision criteria (set before seeing the answer)

- **Change the default** only if: (a) the non-parametric `h` is reproduced by
  the parametric fit (pole form identified), (b) synthetics recover tau within
  +/-15%, (c) at least three independent beads on two or more platforms give
  prefactors whose spread is smaller than their distance from 10 ms, and (d)
  the speed exponent is constrained well enough to choose between -0.5 and
  -0.32 (or the choice is shown not to matter over 0.25-1.4 m/s).
- **If beads scatter more than the shift**: the default stays a nominal value,
  documented as such, and per-bead tau goes into the #189 registry instead.
- **Never overwrite `lueck`**: a new default gets a new model key, so every
  existing product stays reproducible, and products record the model used.
- Product impact is quantified before any switch: chi ratio new/old per
  campaign, measured on a subset by rerunning, not via the 0.76 elasticity.

## 8. Literature (survey 2026-10-13)

> **Corrected 2026-10-28.** The 10 ms *is* published: Sommer et al. (2013),
> doi:10.1175/JTECH-D-12-00272.1, with Rolf Lueck a co-author, found tau = 10 ms
> for the FP07 as a speed-independent **double** pole (half-power 10.2 Hz, Lake
> Kivu, 0.19-0.89 m/s), and rejected velocity scaling. What still holds below: no
> source gives the repo's **single pole x W^-1/2** form, so the hypothesis
> sharpens to Sommer's number combined with the Vachon & Lueck exponent and the
> wrong pole count.

**No source gives single pole, tau = 10 ms * W^-1/2.** It entered the repo in
commit 2dbf6da (2026-03-08, the original chi port, co-authored with an AI
model) credited to Lueck et al. 1977, together with the fixed-3-ms `goto`.
Hypothesis (moderate confidence): a conflation of "~7-10 ms" with the Vachon
& Lueck exponent.

Half-power frequency f_3dB [Hz] (single pole 1/(2 pi tau); double pole
0.6436/(2 pi tau)):

| source | form | measured W | f_3dB @0.3 | @0.7 | @1.0 | method |
|---|---|---|---|---|---|---|
| **package default** | SP 10 ms W^-1/2 | none | 8.7 | 13.3 | 15.9 | none |
| Vachon & Lueck 1984 (via ODAS, Sommer) | DP 4.1 ms W^-1/2 | ~0.4-1.3 | 13.7 | 20.9 | 25.0 | lab, vs fast C sensor |
| Hill 1987 (Fig 4, digitized) | between SP and DP | 0.54-1.97 | ~10.7 | ~16.3 | ~19.5 | lab plume, n = 1 |
| Nash et al. 1999 | T1 DP / T2 SP | 0.3, 1.2 | 16-20 | 18-23 | 19-24 | in situ vs uC probe |
| Nash & Moum 2002 | DP f_c 29 Hz | 0.25-0.35 | 18.7 | (18.7) | (18.7) | in situ vs uC / thermocouple |
| Sommer et al. 2013 | DP 10 ms fixed | 0.19-0.89 | 10.2 | 10.2 | 10.2 | interface histograms, by eye |
| Ijichi & St. Laurent 2025 glider | SP 8 ms U^-1/2 | 0.23-0.37 | 10.9 | (16.6) | (19.9) | obs / Kraichnan(q 5.26, shear eps) |
| Ijichi & St. Laurent 2025 VMP | SP 6 ms fixed | 0.67 | (26.5) | 26.5 | (26.5) | same |
| Goto et al. 2016 | DP 3 ms W^-0.32 | 0.6-0.7 | (23.9) | 30.5 | (34.1) | eps_T vs eps_S |
| Rockland TN-005 | "about 7 ms at 1 m/s" | — | — | — | 22.7 (SP) | vendor; "not convincingly determined" |

(parentheses: extrapolated outside the measured speeds.)

- **VMP, ~0.7 m/s:** FP07 measurements cluster at 18-30 Hz (median ~20); the
  default's 13.3 Hz over-corrects. Sommer's 10.2 is the outlier.
- **Glider, ~0.3 m/s:** the one glider measurement (Ijichi) gives 10.9 Hz;
  the default's 8.7 Hz is 0.8x of it. Ijichi report a ~2x glider-vs-VMP
  difference at 30 Hz that they cannot explain — a platform effect as large
  as the sensor spread.
- **Sensor-to-sensor spread** up to 1.7x in f_3dB within one study (Nash
  1999), ~18% in tau (Goto); fouling up to 3x. A single default therefore
  carries roughly +/-30-50% in f_3dB whatever it is.
- **Every in-situ estimate except the uC/thermocouple comparisons depends on
  a spectral model:** Ijichi's f_3dB moves 2x for q between 3.2 and 7.9. This
  is exactly the dependence the decomposition in section 3 is built to avoid.
- The exponent: -1/2 from V&L, Hill and Ijichi (narrow range); -0.32 is from
  glass rods (Gregg & Meagher) and adopted, not measured, for the FP07.

## 9. Deliverables

1. `scripts/fp07_tau/` — the decomposition fit, the synthetic harness, and a
   driver per dataset (resumable, checkpointed per file).
2. `docs/fp07_tau_prefactor.md` — results with n, method and spread.
3. If the criteria pass: a new tau model in `chi/fp07.py` and a PR; otherwise
   a documented nominal and the per-bead path.


## Review outcome and first results (2026-10-13)

**Blockers (adversarial review, confirmed on real data):**

1. **The FP07 knee sits on the low-k excess.** At VMP speeds the response
   turns over at x = k/kB ~ 0.03-0.2, the same band as the universal low-k
   excess (#191, crossover x ~ 0.04-0.15). With W nearly constant, a
   k-dependent excess and h(f) are degenerate (the three-way model is not
   identifiable), and a k_min cut removes the knee. Confirmed on Kemedukl
   (W 1.25 m/s): single-pole tau 2.9 ms at k_min 2 cpm, 9.1 at 8, 16 with
   f_max 40. Synthetics on the same geometry recover tau to one grid step,
   so the method is sound and the data violate its assumption.
2. **The moored Hydrographer MR is unusable as planned**: `speed.py` takes
   |U_EM| (flow reverses ~36% of the time: probes in the body's wake), shear
   epsilon is implausible (median 2e-3, p90 0.15 W/kg; probe pairs 4x worse
   than the VMPs), dead/railed beads.
3. **The cross-platform exponent is undecidable** (opposite-signed biases by
   geometry; water temperature differs by platform).

**Errors in this plan, corrected:** the anti-alias response is
1/(1+(f/f_AA)^8)^2, i.e. 0.95 at 70 Hz for f_AA = 110 Hz, not > 0.999 (0.999
only below ~40 Hz); a speed error is not "f mis-scaled" (x is invariant to a
per-window W error through eps ~ W^-4); section 2's s = 1.3-2.9 is stale —
#191's later diagnosis (x > 0.1) put the M1/M2 crossing on tau_lueck
(s 1.00/1.29/1.11). The per-bin SNR gate biases tau high when band-edge SNR
is ~5 (Rutgers); Kemedukl is safe (median 461).

**The evidence now conflicts:**

| line | method | says |
|---|---|---|
| literature, VMP ~0.7 m/s (V&L, Hill, Nash, Goto, Ijichi) | lab and in situ | f_3dB 18-30 Hz: FASTER than lueck (13.3 Hz) |
| #191 M1/M2 crossing, x > 0.1 (CASPER-West, ARCTERX-22) | Kraichnan, shear eps | tau ~ tau_lueck |
| Ijichi & St. Laurent 2025 VMP | Kraichnan, shear eps | SP 6 ms fixed (26.5 Hz) — 2x faster than the line above |
| SN479 T2/T1 spectral ratio (below) | model-free, weak | faster bead ~7 ms at 1.04 m/s (21 Hz), unstable between W bins |

**SN479 decomposition** (28 files, W ~1.07 m/s): the same failure — single
pole 3.8 ms at k_min 2 cpm, 5.6 at 8; double pole 2.0-2.4 ms. Both VMPs give
f_3dB 40-57 Hz from the decomposition, faster than every published value:
treated as biased by blocker 1, not as a measurement.

**SN479 T1xT2:** coherence collapses above ~6 Hz (0.26 at 6 Hz, < 0.01 by 15
Hz), so the cross-phase method fails at these probe separations. The POWER
ratio needs no coherence: <S2>/<S1> = G |H2|^2/|H1|^2 rises from ~1.1 to
~1.5 between 4 and 50 Hz, and its turnover fixes each tau absolutely when the
beads differ. Fits: 8.9/7.4 ms at W 1.04 (7,670 windows), 8.4/6.9 at 0.96,
4.9/4.0 at 1.19 — not yet stable; ±10% wiggles between neighboring bins
suggest narrowband (vibration) contamination; beads differ only 1.2x here.
`scripts/fp07_tau/ratio.py`. Needs: synthetics, vibration masking, and
datasets whose beads differ strongly (ARCTERX-22 ratios 0.40 / 1.67).

**Revised direction (proposed):**
1. Ratio method, validated on synthetics, on strongly mismatched bead pairs
   (SeaChest: ARCTERX-22 SN194/SN428).
2. A direct transfer function against a faster reference — micro-conductivity
   (Nash et al. 1999) — if any fleet instrument carries one.
3. The #189 single-transition rise-time envelope with its noise control, as an
   upper bound on tau.
4. The section-3 decomposition only where kB is high enough that the knee
   clears the excess (high Re_b), or on a W-varying same-bead dataset with
   validated epsilon.


## Source of the 10 ms, and the SBE7 (2026-10-13, later)

> **Superseded 2026-10-28:** the 10 ms is Sommer et al. (2013), which Rolf Lueck
> co-authored (as a speed-independent double pole; see the note in section 8).
> The paragraph below is kept as the record of what was believed on 2026-10-13.

Pat confirms the 10 ms did NOT come from Rolf Lueck: it came out of an earlier
AI model. He has not found published measurements he trusts, and the
bead-to-bead variance is unknown — measuring it is a goal in itself, and the
T1/T2 ratio measures it directly for every bead pair.

The ARCTERX-2022 Wake Slocum "Starbuck" MicroRider carries a micro-conductivity
channel, `C1_dC1` (type `ucond`, diff_gain 0.4), in all 203 files
(`ARCTERX/2022/Wake/slocom/Starbuck MicroRider/DATA`). The package does not
convert `ucond`; a ratio needs no gain, only the pre-emphasis and a common AA.

## Synthetic check of the ratio estimators (`ratio_synth.py`, `uc_gate_synth.py`)

Each synthetic window has its own kB, chi and W and a deliberately
non-universal shape (a per-window low-k excess, a mid-band dip); spectra carry
chi-squared noise at 9.5 DOF; 3,000 windows per case.

**T1/T2 (single pole, 1-60 Hz):**

| truth tau1/tau2 | W | recovered |
|---|---|---|
| 10/10 ms | 1.0 | 0.6/0.5 — **no information when the beads match** |
| 12/10 | 1.0 | 11.5/9.6 |
| 15/10 | 1.0 | 15.1/10.0 |
| 25/10 | 1.0 | 24.5/9.8 |
| 6/4 | 1.0 | 5.9/3.9 |
| 20/15 | 0.3 | 18.4/13.9 |
| 30/12 | 0.3 | 29.5/11.8 |

The non-universal shape does not move it. Failure modes: matched beads
(unidentifiable — the fit must say so), low SNR (peak S/N 20: ~20% low),
the wrong pole form (single-pole fit to a double-pole truth: 9.7/3.7 for 8/5;
fitting both forms recovers 7.8/4.8), narrowband lines (small bias, removed by
a 7-bin running median).

**T/uC (FP07 18 ms, W 0.3, uC flat):** exact without salinity (17.97 ms). A
salinity contribution, with its own kB_S = 10 kB_T, biases it hard unless each
window is gated on x = k/kB (shear epsilon):

| salinity share | all x | x <= 0.3 | x <= 0.15 | x <= 0.08 |
|---|---|---|---|---|
| 0 | 17.9 | 18.4 | 19.5 | 15.7 |
| 10% | 31.3 | 20.5 | **18.0** | 17.5 |
| 30% | 45.9 | 25.1 | **18.7** | 20.9 |


## Starbuck micro-conductivity: not usable as a fast reference (2026-10-13)

`uc_ratio.py` on 83 Starbuck profiles (75 of 203 files have enough
flight-model coverage for shear epsilon) gave a ratio that RISES ~f^2 from 1
to 12 Hz — not an FP07 response. The channel itself:

- **Files 001-~100: dead.** Counts pinned at -326..-334 (std 0.1-0.8 counts);
  coherence with the FP07 ~0 at every frequency.
- **Files ~101-203: working at low f only.** Coherence with T1 0.98 at
  0.06-0.2 Hz, 0.75-0.86 at 0.5-1 Hz, ~0.3 at 2-4 Hz, ~0 at 8 Hz. The T->C
  transfer is ~60 counts/K, ~5x below the config's a/b/k (292 counts/K), and
  it FALLS with f (60 -> 12 counts/K by 8 Hz) where a pre-emphasized channel's
  should rise 5x by 2 Hz.
- **Resolution:** the high-f level is 1-count quantization noise (-36.7 dB
  measured, -37.2 predicted); by the measured transfer 1 count ~ 17 mK.
- ODAS `odas_ucond_internal`: C = 10*((counts*adc_fs/2^bits + adc_zero) - a)/b/k.
  Its mean reads -60 then -92 mS/cm — the channel is probably AC-coupled, and
  its config constants do not describe it.

It cannot resolve the FP07 knee (~10 Hz at 0.5 m/s). Closed for this dataset.

> **Revisited 2026-10-28: still not usable, but for a different reason.** Most
> of the reasons above were analysis artifacts:
> - **De-emphasized twice.** `PFile` overwrites `channels_raw['C1_dC1']` with
>   the de-emphasized series, as it does every channel with a `diff_gain`.
>   `uc_ratio.py` then divided by 1 + (2 pi f G)^2 again: 7x too small at 1 Hz,
>   ~900x at 12 Hz. That is the ratio that "rises ~f^2", and the reason the
>   transfer seemed to fall where it should rise.
> - **The pre-emphasis matches the config.** Fitted against T1 at 0.03-1 Hz:
>   G_C 0.357 s (MAD 0.033, n 11 files) against 0.4 configured.
> - **It is not quantization-limited.** The floor is 0.2-0.3 mK/sqrt(Hz),
>   white from 10 to 80 Hz. In the strongest 1-5% of windows C1 clears 3x its
>   noise to 16-21 Hz, past the knee at 0.3-0.4 m/s.
> - **No other C channel.** The address matrix never samples one. The channel
>   is DC-coupled: it tracks the glider CTD's temperature from 12 to 28 C to
>   0.6 counts rms.
>
> **The real reason: C1 does not see salinity.**
> - **Salinity test.** Its fit gain is -2.2 counts/psu [file bootstrap 16-84%:
>   -2.9 to -1.5; 101 files, 82,207 CTD samples], against +89 for a
>   conductivity cell at its temperature gain. Its temperature curve has an NTC
>   thermistor's curvature (C''/C' 0.028/K, against 0.006 for seawater
>   conductivity).
> - **Spectral ratio.** C1/T1 FALLS above the FP07 knee: 0.50-0.60 at 16-22 Hz
>   in the strongest 1% of windows, against 0.95 at 1 Hz. A pole-free
>   reference would make it rise. The same estimator on a synthetic pole-free
>   C1 recovers a single-pole 15 ms T1 as 14.6 ms and a double-pole 10 ms as
>   9.2 ms, so it would have seen a real reference.
>
> That C1 is a slow thermistor is a conjecture (~85%). Every file carries the
> same 2012 configuration template, so whether an SBE7 was ever fitted is
> unverified. `uc_ratio.py` is corrected.


## T1/T2 ratio on real data, first pass (2026-10-13; `run_fleet.py`)

Single-pole tau [16-84% file bootstrap] at the deployment's median W; double-pole
fits are equally good (pole form undetermined) and give 3-dB points ~30-40% lower.

| deployment / bead pair | W | windows | tau1 [ms] | tau2 [ms] | f_3dB T1/T2 [Hz] | lueck f_3dB |
|---|---|---|---|---|---|---|
| ARCTERX-22 Interior SN194 | 0.95 | 37,212 | 13.0 [11.9-14.0] | 19.5 [17.8-20.9] | 12.3 / 8.1 | 15.5 |
| — first half | 0.93 | 23,292 | 10.8 [9.6-12.6] | 17.7 [16.4-19.3] | | |
| — second half | 0.97 | 13,920 | 16.1 [14.1-18.4] | 22.0 [19.4-25.2] | | |
| ARCTERX-22 Interior SN428 | 1.03 | 50,903 | 16.0 [13.6-18.0] | 12.7 [10.7-14.4] | 10.0 / 12.5 | 16.1 |
| ARCTERX-22 Wake SN428 | 0.97 | 57,461 | 13.0 [11.2-14.7] | 9.3 [7.7-10.7] | 12.2 / 17.1 | 15.7 |
| ARCTERX-22 Wake SN194 | 0.94 | 3,682 | not identifiable (6.1 / 9.3) | | | |
| ARCTERX-25 Wake SN479 | 1.07 | 22,393 | not identifiable — beads matched (6.4 / 5.4) | | | |

> **Superseded 2026-10-28 (`fp07-pair tau`, issue #189).** These first-pass
> values come from spectral SUMS (`ratio.collect`/`analyse`), which the section
> "Robust estimator" below found not robust. Rerun on the same ARCTERX-22
> Interior chi_00 products, that code reproduces the SN194 row exactly
> (13.0 [11.9-14.0] / 19.5 [17.8-20.9], halves 10.8/17.7 and 16.1/22.0), so the
> products have not changed. The median-of-log estimator on the same files gives
> 17.4 [17.1-17.6] / 25.6 [25.2-25.9] ms (`analyze_cache.py` `b25_p1`) and the
> production tool 16.7 [15.9-17.7] / 24.8 [24.0-25.7] ms (halves 14.8/23.0 then
> 20.3/28.1), the values the FP07 calibration sheet records. The tau ratio
> (1.47-1.50) and the slowing between halves agree across all three; the
> absolute level does not, by up to 1.3x. That is the estimator dependence of the
> absolute level, not a change in the beads. The "Robust estimator" section's
> 16.3/23.5 (1-25 Hz) could not be reproduced exactly (17.4/25.6 here; its
> cache no longer exists). Two later changes to `fp07-pair tau` (2026-10-28,
> after an independent review): the ratios' reference level is the curve at
> 2-4 Hz, and a probe whose 60-100 Hz floor exceeds 1.25x its noise model
> identifies no tau. The second TIGHTENS the rule after real results were seen
> (SUNRISE 2022 PointSur); its threshold was set on synthetics. Neither moves
> this pair's taus (floors 1.02/1.05); its 20 Hz power ratio now reads 0.57
> relative to 2-4 Hz, against 0.50 relative to the lowest bins (~1 Hz), because
> 2-4 Hz is already inside the slower bead's roll-off (docs/fp07pair/runbook.md
> section 6).

- The four ARCTERX-22 beads read 9-20 ms at ~1 m/s: SLOWER than lueck
  (10 ms), and ~2x slower than the published FP07 values.
- SN194's beads both slowed by 25-50% between the halves of the Interior
  effort: fouling or aging, consistent with #189's split-in-thirds drift.
- Directions match the earlier inter-probe fits (SN194 T2 slower, SN428 T1
  slower); magnitudes are smaller (1.5 vs 2.5; 1.26-1.40 vs 1.67).

**Caveat — noise-model sensitivity (`ratio_synth_noise.py`):** truth 13/19.5 ms,
1-26 Hz band, no SNR gate: a correct model recovers 13.3/20.0; a 1.5x noise
error on one probe gives 16.3/23.3 at peak S/N 300 and 29.4/37.2 at peak S/N
60. The real fits keep only bins where both probes are >= 10x the model in
>= 80% of windows, which limits this, but each probe's noise model has to be
checked against its measured floor before these absolute values are trusted.


## Robust estimator, response shape, and what the ratio does NOT give (2026-10-13)

(The "a ~ 1" response shape claimed below was WITHDRAWN once the fleet showed
the same beads giving a = 1.80 and 1.10 in two efforts; see "Fleet results and
conclusions". The estimator and the noise check stand.)

**Noise model checked** (`noise_check.py`, lowest-decile-chi windows, obs/model
at 80-100 Hz): SN194 0.9-1.0 on both probes; SN428 1.3-1.45 on both. Rerunning
ARCTERX-22 with SN428's noise x1.4 and/or the gate 10x -> 30x moves Interior
tau by <= 10%; Wake SN428 up to 25% (band shrinks to 17-20 Hz, flagged).

**Plain sums are not robust.** Window levels span orders of magnitude, so a
few energetic windows dominate a sum of spectra; that produced a spurious step
at 30-35 Hz in SN194's ratio (the per-probe median spectra are smooth there).
The estimator is now the median over windows of ln(S2/S1), gated on a running
mean of S/N (`medlog.py`); synthetics recover single poles to 4-9% with no
band dependence.

**The beads' responses are not single (or double) poles.** On real data the
single-pole tau depends on the band (SN194: 16.3/23.5 ms on 1-25 Hz, 10.3/15.5
on 1-60 Hz), which synthetic single poles never show. A free-shape response
|H|^2 = 1/(1 + (f/fc)^a) fits 3x better than either pole form, with
**a ~ 1** — power falling as f^-1 at high frequency:

| pair | a [16-84%] | fc1 / fc2 [Hz] | rms free / SP / DP |
|---|---|---|---|
| Taiwan13 VMP SN002 | 0.97 [0.88-1.13] | 31.9 / 107 | 0.014 / 0.023 / 0.023 |
| ARCTERX-22 Int SN194 (T1592/T2005) | 0.94 [0.90-0.98] | 21.0 / 5.1 | 0.011 / 0.036 / 0.037 |

Synthetics (`shape_synth.py`): single-pole truths return a = 1.9-2.05 (no bias
toward 1) and no band dependence; a = 1 truths reproduce the real band
dependence closely (synthetic 15.9/23.9 -> 11.1/17.2 ms vs real 17.4/25.6 ->
10.7/16.3); window-to-window tau scatter of +/-50% only reaches a ~1.5; mixed
single/double-pole pairs fit nothing like the data. A diffusive response
H = 1/(1 + sqrt(i w tau)) (heat through a coating into a core; f^-1 at high f)
fits far better than a single pole but worse than the free shape, and gives
implausible absolute values (f_3dB 1 Hz).

**Correction to an earlier claim in this plan:** "the turnover fixes each tau
absolutely" holds only WITHIN an assumed response family. For the same data
the faster SN194 bead's half-power point is 21 Hz (free a), 15 Hz (single pole,
1-70 Hz) or 6.8 Hz (half-order). What the ratio gives without a model is the
RELATIVE response of the two beads, and the evidence that the roll-off is
shallower than a single pole. An absolute level needs an external anchor —
e.g. the Kraichnan shape where it fits (x > 0.1, #191), with the response
family taken from the ratio.

**Implication if a ~ 1 holds fleet-wide (hypothesis):** the pipeline's
single-pole correction grows as f^2 where the real response needs f^1, so it
over-corrects the high-wavenumber end of every chi fit that reaches it.


## Absolute anchor from in-situ spectra: fails its synthetic check (2026-10-13)

`abs_anchor.py` takes the response family and fc1/fc2 from the ratio and fits
ONE common scale c, with a free shape g(x) at x >= 0.1, per-window levels and
per-probe gains, on shear-epsilon kB. Synthetic check on ARCTERX-22 SN194's
real geometry (37,212 windows; sd(ln kB) 0.66, sd(ln W) 0.12):

| synthetic truth | realistic shape (low-k excess + mid-band dip) | clean shape |
|---|---|---|
| c = 1.0 | 0.57 | 0.85 |
| c = 2.0 | 0.85 | 1.38 |

Biased low and compressed even with a clean shape: a smooth, shallow response
trades against the free shape g. **The absolute FP07 response cannot be
anchored from in-situ VMP spectra with the data in hand**, even with the
family known. It needs an external reference: a lab measurement, or a
working fast sensor co-located with the FP07.

## Data-handling defects found and fixed (2026-10-13)

- `analyze_cache.load()` dropped every chunk whose frequency grid differed
  from the first: ARCTERX-22 SN194 has 100 bins below 100 Hz, SN428 101, so
  SN428 was silently missing from both ARCTERX-22 rows of the first fleet
  table. It now returns one dataset per grid.
- `cache_spectra.py` skipped files whose grid differed WITHIN a chunk: ~480
  files (428 ARCTERX-22, 55 ARCTERX-23). It now writes one bucket per grid
  (`chunk_NNN_gK.npz`); those three deployments are being rebuilt.


## Fleet results and conclusions (2026-10-13)

Spectra of every window cached from 16 deployments (`cache_spectra.py`,
`speccache*`), analysed per bead pair and time half (`analyze_cache.py`,
median-of-log-ratio estimator, 100-file bootstrap). 14 independent VMP bead
pairs, 4 usable glider pairs (Tanaka's three gliders are too noisy; CASPER-West
doug's pair includes T1's known failure and is excluded).

**1. Bead-to-bead spread, model-free.** From each pair's power ratio at f,
relative to 1-3 Hz, assuming beads scatter independently (Var ln ratio =
2 sigma^2). 1-sigma factor in a single bead's power response:

| f [Hz] | VMP (14 pairs, 0.7-1.6 m/s) [16-84%] | glider (4 pairs, 0.2-0.3 m/s) |
|---|---|---|
| 5 | x1.07 [1.04-1.08] | x1.10 |
| 10 | x1.15 [1.12-1.17] | x1.22 |
| 20 | x1.29 [1.22-1.34] | x1.29 |
| 30 | x1.39 [1.28-1.45] | x1.29 |
| 40 | x1.47 [1.34-1.55] | x1.32 |
| 60 | x1.59 [1.40-1.70] | x1.38 |

This is the number that was unknown. It is robust to the response family.

**2. Response shape: not settled.** The free-shape exponent varies by pair
(VMP 0.94-3.1) and, for the SAME beads, between efforts (SN428 1.80 Interior
vs 1.10 Wake; SN194 0.94 vs 1.36) with tight intervals each time — so the
fitted shape is not a stable bead property, plausibly because residual noise
near the top of the band moves it. Withdrawn: "FP07s are diffusive (a ~ 1)".
What holds: for most pairs a single pole is not the best description, and the
fitted single-pole tau depends on the band.

**3. Absolute level, single-pole family, 1-25 Hz band, scaled to 1 m/s by
W^-1/2** (only beads whose pair identifies the level):

| | beads | median | geometric spread | range |
|---|---|---|---|---|
| VMP | 25 | 12.0 ms | x1.94 | 2.4-41 |
| glider | 8 | 12.2 ms | x1.83 | 5.7-27 |
| `lueck` default | | 10.0 | | |

The VMP and glider medians agree under W^-1/2 (weak support for that
exponent: different beads). The absolute level is family- and band-dependent
(ratio) and the in-situ anchor failed its synthetic check, so these are
single-pole-equivalent values over 1-25 Hz, not a measurement of the bead.
They point the same way as #191's x > 0.1 crossing (tau ~ tau_lueck) and
AWAY from the faster published values (V&L, Goto, Ijichi: 4-7 ms at 1 m/s).

**4. What this means for the default.**
- The 10 ms default sits near the fleet's single-pole-equivalent median
  (12 ms). There is no evidence-based case for a large move, and none for
  the faster published values.
- The dominant error is not the prefactor but the bead: a factor ~2
  (geometric 1 sd) between beads, plus drift within a deployment (SN194:
  +25-50% between halves). No single default can remove it.
- The robust improvement is per bead: the T1/T2 ratio measures each pair's
  relative response, shape-aware, and could equalize the two beads (a
  generalization of `fp07_tau_scale`, which fits only an integrated ratio).
- The absolute level needs an external reference: a lab measurement with a
  fast reference (thermocouple, as Nash et al. 1999, or a working
  micro-conductivity probe), or Rockland.

**5. Decisions (Pat, 2026-10-13).** Keep 10 ms until a more significant result.
Build the per-bead equalization: done as `fp07-pair` (PR #217), which on
ARCTERX-2022 brought each pair's chi ratio onto its flat gain (SN194 -0.570 ->
-0.142, SN428 +0.321 -> +0.088 in ln) and removed the K_max_ratio trend. No
lab measurement. Starbuck is the only deployment with an SBE7, and its
micro-conductivity channel cannot serve (above: revisited 2026-10-28, it
behaves as a temperature sensor), so there is no in-hand external reference. "The pair comparison is what is most practical
now."

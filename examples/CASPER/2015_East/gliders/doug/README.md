# MicroRider SN 134 on glider doug — CASPER-East 2015

Microstructure from the Rockland MicroRider carried by Slocum glider `doug`
during CASPER-East (R/V *Atlantic Explorer*), on the 50 m shelf off Duck,
North Carolina. **4,653 profiles from 137 of 173 files**, 2015-10-12 13:13 →
2015-11-05 18:22, 2.5–45.5 m, processed 2026-09-20.

Twin of `../../../../2017_West/gliders/20170929_doug_CASPER/analysis/` — same
glider, same MicroRider, same recipe. Read that README too; this one records
what is different, and **χ is the product here, not ε.**

## Verdict up front

| | status |
|---|---|
| **χ (thermal)** | **usable.** Both thermistors healthy, T1/T2 = 0.843 |
| **ε (shear)** | **NOT VALIDATED — do not use.** See below |
| CTD / hotel | verified bit-identical to the 2018 conversion |
| clock | solved, 3 blocks, 0.29–0.30 s RMS, independently confirmed |
| speed | flight model, α consistent with two other deployments |

## Results (1-m bins, 4,652 profiles)

| | median | spread (MAD) | n |
|---|---|---|---|
| χ (paired) | 9.7e-9 K²/s | 0.76 dex | 137,321 |
| χ(T1) | 8.5e-9 | 0.76 dex | 134,199 |
| χ(T2) | 1.07e-8 | 0.74 dex | 132,954 |
| N² | 6.5e-5 s⁻² | 0.63 dex | 141,405 |
| speed | 0.313 m/s | 0.046 | 141,405 |
| ~~ε (paired)~~ | ~~5.8e-8 W/kg~~ | 0.76 dex | 141,405 |
| ~~Γ~~, ~~K_ρ~~, ~~K_T~~ | *derived from ε — inherit its problem* | | |

This shelf is **~35× more energetic than CASPER-West** on the ε median, which
is expected for 50 m of tidally and wind-mixed water off the Outer Banks — but
see the ε caveat before quoting that number.

**The thermistors agree: χ(T1)/χ(T2) = 0.843, MAD 0.17 dex over 4,557
profiles.** Both are healthy for the whole deployment by the relative test in
`thermistor_health.csv` (T1_sd/T2_sd median 0.968, 5–95 %: 0.963–1.067; 138
files ok, 33 bench/frozen, 2 briefly railed). That is the opposite of
CASPER-West, where T1 died. χ here comes from a Method-2 spectral fit, which
does **not** use the shear probes, so it is unaffected by the problem below.

## ⚠️ ε is not trustworthy — the shear probes flip-flop by ±3 decades

The per-file median of log₁₀(e₁/e₂), over 132 files and 665 profiles, spans
**−3.46 to +3.57** and is **bimodal**:

| | share of files |
|---|---|
| sh1 more than 3× above sh2 | **33 %** |
| the two within a factor of 3 | 14 % |
| sh2 more than 3× above sh1 | **54 %** |

Things this is **not**:

- **Not a sensitivity or gain error.** A wrong `sens` or `diff_gain` is a fixed
  multiplier; it cannot change sign from file to file. The config values
  predict e₁/e₂ = 0.61 (era 1) and 1.70 (era 3); neither resembles what is
  seen, in either direction.
- **Not a dead channel.** The raw shear signals are comparable throughout —
  per-file σ ratios 0.62–2.13 in both ADC counts and converted units, in all
  eras and in 2017. The 2–4 decade differences appear only in ε, i.e. in the
  *resolved spectral variance*, not in the data.
- **Not fixed by QC.** Gating on the Lueck FM statistic makes the pooled
  disagreement **worse**, not better (era 3: 0.634 all windows → 0.020 at
  FM < 1.15 → 0.004 at FM < 0.8). A real common signal would converge.
- **Not a probe property.** It survives a probe change on *both* channels
  (M1247→M1344, M1249→M1371).

What it looks like, stated as a hypothesis and **not** established: in 50 m of
water the glider is never far from a boundary, and each yo turns every few
minutes, so one probe or the other is frequently sitting on its noise floor
while the other sees signal — whichever is momentarily clean wins. Consistent
with it: in `CAS_100_prof005` sh2 sits at ~2e-10 W/kg with FM 1.3–2.0 (a
rejected spectral shape) while sh1 reads 1e-7–1e-6 with FM 0.5–1.2; median FM
is 0.99 (sh1) and 1.02 (sh2), both at the acceptance boundary, with 23 % and
38 % of files failing it.

**Before ε from this deployment is used it needs a per-window noise-floor
model and a decision rule for which probe to believe** — not a pooled average.
`epsilonMean` is a geometric mean of the two and is therefore meaningless
wherever they disagree, which is 86 % of files.

Note this is **specific to CASPER-East**. The same MicroRider at CASPER-West,
in 195 m of water, gave sh1/sh2 = 0.939 with 0.02 dex of scatter.

## Three probe eras, and the config tracks them

Unique among SN 134 deployments: the config was edited on the day of each
change, so `sens` is right per file and one perturb run is correct across all
three. Do **not** override the sensitivity globally.

| files | dates | sh1 | sh2 | T1 | T2 |
|---|---|---|---|---|---|
| `CAS_021`–`056` | 10-11 → 10-14 | M1247 0.0960 | M1249 0.0748 | T987 | T988 |
| `CAS_057`–`060` | 10-14 → 10-22 | M1253 0.0622 | M1249 0.0748 | T990 | T991 |
| `CAS_061`–`169` | 10-23 → 11-05 | M1344 0.0655 | M1371 0.0855 | T865 | T866 |

The "config never edited 2015-10-23 → 2017-10-23" window that makes Taiwan 2017
and CASPER-West probe identities unverifiable begins with the **last** of these
edits, so CASPER-East is on the good side of it.

## The clock jumped twice

`glider_time = MR_time + L`.

| block | files | L at t_ref | drift | residual RMS |
|---|---|---|---|---|
| 10-12 → 10-14 | 26 | −63.99 s | −0.192 s/day | 0.289 s |
| 10-23 → 10-28 | 35 | +8.20 s | −0.520 s/day | 0.297 s |
| 10-28 → 11-05 | 65 | −7.60 s | −0.498 s/day | 0.305 s |

**+73.50 s** across the 234 h probe-change gap, **−12.75 s** on 10-28. One line
through all three gives 16.5 s RMS and a fictitious +2.6 s/day drift.
Confirmed independently by MR thermistor vs glider CTD (r = 0.917–0.999,
consistent **+3.1 s**, the CTD's own print lag — uncorrected, ~0.45 m).

`MIN_YO_DBAR` is **20** dbar here. The inherited 40 dbar gate is right for
Taiwan's 200 m yos and discards 115 of the 126 good files on this shelf, where
every long file has a 25.6–43.6 dbar yo.

## Speed, and the compressibility that could not be fitted

Cd₀ = 0.1829 global; per-day 0.1673–0.1919, **13.9 %** spread (CASPER-West:
5.3 %). U 0.313 m/s. α = **−2.86° diving, +2.91° climbing**, against
CASPER-West's −2.84°/+3.16° and Taiwan17's −2.90°/+2.89° — three deployments,
two oceans, the same glider. `w_water` medians to +0.0002 / +0.0000 m/s.

**Hull compressibility is imported, not fitted.** In 50 m of water the usable
glide window is 5–30 m — a 2.5 bar lever arm — and fitting ε returns
**−5.38e-9 Pa⁻¹**, a hull that expands under pressure, dragging Cd₀ to a 74 %
day-to-day spread. Held at doug's own CASPER-West value (**+8.219e-10 Pa⁻¹**,
measured over 0–195 m), the spread falls to 14 %. The rejected fit is kept as
`calibration.fitted-epsilon-REJECTED.json`.

This is well conditioned: over 0–30 m, ε contributes only ~12 cc of the ±233 cc
pump range, so it barely affects the speed — which is exactly why it is not
identifiable here, and why fitting it merely let it absorb noise.

## The 36 files that produced nothing

`CAS_001`–`023` (pre-cruise on R/V *Elakha*, plus the first two minutes of the
glider record), `CAS_055`–`062` (the probe-change window, where the glider
record itself has 37 h / 17 h / 14 h gaps, so no speed exists), `CAS_066` (the
glider sat on the surface — 0.43 dbar for a 49-minute record, against 33–39
dbar either side), `CAS_170` (header dated 2016-08-08) and `DAT_001`–`003`
(bench). All correct refusals.

## A note on running this again

`parallel.jobs` is **2**, not 8. At 8 the binning stage died with
`NetCDF: HDF error` on a profile file — which was **not** corrupt: the same
bytes copied off the share opened fine locally. It is an SMB read failure
under concurrency. `perturb bin -c perturb.yaml` re-runs just that stage.

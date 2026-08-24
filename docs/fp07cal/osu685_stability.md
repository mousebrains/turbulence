# FP07 in-situ calibration stability — osu685 (ARCTERX 2025 Interior)

MicroRider MR1000RDL-EM SN 435 on a Slocum with an extended energy bay.
Reference: the glider's own glider CTD, read directly from `PASS0/ebd.nc`.

| | |
|---|---|
| deployment | 2025-01-28 → 2025-04-10 (72.8 days) |
| reference | 5,040,313 CTD samples at 1 Hz, 80.1% duty cycle |
| MR files | 1228 total; 1225 share probes T2811 (T1) / T2813 (T2) |
| usable profiles | 757 with pairs, spanning 65.9 days |
| **recording** | **climbs only — 0 dives.** Every file starts at 507–1016 dbar |
| fit set | 95 files, 202,149 pairs, T range 3.61–27.57 °C |

Both probes carry the same **nominal** coefficients (`t_0 = 289.301`,
`beta_1 = 3143.55`, `beta_2 = 2.5e5`). These are not a calibration of either
probe — they are generic values, identical on two physically different beads.
So the in-situ fit is not a *correction* to a factory calibration; it is the
**only** calibration these probes have. Applying the nominal values instead
costs **−2.66 K (T1) and −1.62 K (T2)**, measured by reading a patched file
against an unpatched one.

## Coefficients

| | T1 (T2811) | T2 (T2813) | factory |
|---|---|---|---|
| `t_0` | 286.6529 | 287.7327 | 289.301 |
| `beta_1` | 3051.45 | 3024.73 | 3143.55 |
| `beta_2` | 2.647e5 | 2.628e5 | 2.5e5 |
| fit rms | 0.00553 K | 0.00569 K | — |

The two probes land ~1.1 K apart in `t_0`, as expected for different beads
sharing one generic number.

> **`beta_2` is present in the nominal config**, so an order-1 emission would
> leave a live quadratic term fighting new linear coefficients (plan A6).
> `fp07-cal patch` neutralises such a term with `1e30` rather than 0 — see the
> runbook. Here order 2 is selected anyway; see below.

## Is `beta_2` worth fitting?

Yes, decisively — and `beta_3` is actively harmful. Measured on 47.7k pairs
spanning 24 °C, splitting the fit by **temperature** (fit the warm half,
predict the cold half) rather than at random, because extrapolation onto
profiles outside the fitted range is the failure mode that matters:

| order | in-sample | held-out (warm→cold) | verdict |
|---|---|---|---|
| 1 | 22.3 mK | 70.0 mK | insufficient |
| **2** | **10.7 mK** | **31.0 mK** | **best** |
| 3 | 10.7 mK | 126.3 mK | overfit |

`beta_2` halves the in-sample residual and improves extrapolation two- to
five-fold; its coefficient carries a t-statistic of 417. `beta_3` gains
0.013 mK in sample while making extrapolation **four times worse** — and its
t-statistic is 10, which a significance test alone would have called
"significant". That is precisely why the tool selects order by held-out error
(`fit.order: "auto"`) and not by in-sample fit or by a t-test.

Note this contradicts the plan's original order-1 default, which was reasoned
for a narrow-range (<8 °C) glider deployment. With 24 °C of range the quadratic
is essential. The rule is the range, not a fixed default.

## Lag

Two independent estimators agree to 0.04 s, and both track a deliberately
injected shift exactly:

| term | T1 | T2 |
|---|---|---|
| pressure offset (clock skew + geometry) | +4.30 s | +4.30 s |
| total temperature lag | +7.36 s | +7.40 s |
| **CTD sensor response** | **+3.06 s** | **+3.10 s** |

The two probes agree on the response to 0.04 s — a genuine cross-check, since
they are independent sensors sharing one reference.

## Temporal stability — small, real, and independently confirmed

| | T1 | T2 |
|---|---|---|
| drift | +0.049 mK/day | +0.103 mK/day |
| over 66 days | **+3.1 mK** | **+6.5 mK** |
| permutation *p* (24 blocks) | 0.021 | 0.001 |
| Theil–Sen (robust) | +0.041 mK/day | +0.106 mK/day |
| dropping first+last block | +0.042 mK/day | +0.096 mK/day |

Robust to the estimator and to endpoint removal. T1 is borderline (*p* = 0.13 /
0.067 / 0.021 at 6 / 12 / 24 blocks); T2 is solid.

**The independent check works.** `T1 − T2` needs no CTD at all, so it is
available on every profile:

```
predicted from the two per-probe drifts:  -0.054 mK/day
observed T1 - T2 (reference-free):        -0.046 mK/day     85% agreement
```

Two probes drifting at measurably different rates, and their *difference*
predicted from CTD-referenced fits matches the CTD-free differential. That is
the corroboration the plan's D5.4 was designed to provide, and it says the
drift is **probe-specific and real**, not an artifact of the reference.

## Depth stability — flat below the thermocline

Quoted from the **deep (~1000 m) climbs only**: the ~500 m climbs turn at
~508 dbar, so their deepest bins are a few profiles scraping their own apogee
and produced 3–5 mK swings at 475–575 dbar that are sampling noise.

| | T1 | T2 |
|---|---|---|
| full range peak-to-peak | 3.58 mK | 4.59 mK |
| thermocline (≤175 dbar) | −3.23 … −1.70 mK | −3.19 … −1.52 mK |
| **below 300 dbar** | **−0.22 … +0.35 mK (sd 0.17 mK)** | **−0.08 … +1.39 mK (sd 0.42 mK)** |

All the structure sits in the top ~200 dbar, where `|dT/dz|` is largest. Below
300 dbar the residual is flat to **0.17 mK (T1)**. So it tracks the *gradient*,
not depth — leftover sensor-response mismatch that the linear
`dz·dT/dz + tau·w·dT/dz` model does not fully capture (the CTD response is a
filter, not a pure delay), rather than a depth-dependent calibration.

The `deep − shallow` comparison at matched depth, where a genuine depth
dependence cancels, leaves a median of −0.9 mK (T1) / −1.1 mK (T2) — i.e. about
1 mK is elapsed-time-related, the rest is gradient-driven.

## Sensor geometry

The joint fit returns `dz = +76.8 cm` (T1) and `+87.6 cm` (T2) with residual
lags of 2.50 s and 2.91 s. **The split is not resolved** — collinearity 0.916,
because vertical speed is one-signed on a climb-only deployment. Only the
combination is measured, and it agrees with two independent routes:

| route | result |
|---|---|
| joint calibration+geometry fit | 77–88 cm |
| direct `P_ctd − P_mr` | 91 cm |
| stated build at the observed 44° climb pitch: `1.0·sin44 + 0.17·cos44` | 81 cm |

That T1 and T2 differ by 11 cm in `dz` is itself a symptom of the unresolved
split: two beads millimetres apart on one sensor head cannot really differ in
mounting, so the difference is their own response times leaking into `dz`.

## Bottom line

**A single static coefficient set is good to ~5 mK over this deployment in both
time and depth** — drift is 3–7 mK over 66 days, and depth structure below the
thermocline is under 0.5 mK. Both are far inside what turbulence work needs, and
both are smaller than the per-profile residual scatter (sd ≈ 4 mK), so the drift
is only visible in aggregate over hundreds of profiles.

Ship static coefficients. The drift is measurable but not worth modelling here.

## What a climb-only deployment cost

1. No dive/climb residual split — the sharpest unremoved-lag diagnostic.
2. Vertical speed one-signed, so `dz` and residual lag stay 0.92 collinear.
3. Depth confounded with elapsed-time-since-file-start; only the 500 m/1000 m
   split separates them, and only over their overlap (25–475 dbar).

A handful of recorded dives would fix all three at essentially no cost.

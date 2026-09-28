# VMP SN 142 thermistors vs the ship's SBE 911plus — rosette cast DAT_166

For cast RR1704_10 (2017-02-26 03:38Z, to 250 dbar), VMP SN 142 was strapped to
the R/V Revelle CTD rosette. That gives a direct comparison of the VMP's
temperature chain against the ship's dual-sensor SBE 911plus. Done 2026-09-17.

## Result

| chain | uniform water: median (MAD), n bins | gain vs SBE, 95% interval |
|---|---|---|
| SBE secondary − primary (the reference's own spread) | −0.5 mK (0.7), 55 | +0.2 mK/K [−0.1, +0.5] |
| **JAC_T** − SBE | **+2.1 mK (2.3)**, 55 | **+1.1 mK/K [−0.2, +2.1]** |
| **FP07 T1**, calibrated in situ vs JAC_T − SBE | **+1.1 mK (2.2)**, 55 | **+1.7 mK/K [+0.2, +3.7]** |
| FP07 T2, same | +0.8 mK (2.8), 55 | — |
| SP from JAC_C/JAC_T − SBE | **+14 mPSU (7)**, 246 | — |

"Uniform water" means |dT/dp| < 0.005 K/dbar; salinity uses all 246 bins. The
JAC_T and T1 gains come from the uniform-water bins at |dT/dp| < 0.01 K/dbar
(see *A gain that did not survive*). The SBE row is a whole-cast fit, since its
two sensors sit at the same height.

- **The temperature chain is good to a few mK.** JAC_T, which the FP07s are
  calibrated against, agrees with the SBE to ~2 mK. Its gain is not
  distinguishable from zero. The calibrated FP07s land within ~1 mK in uniform
  water.
- **Calibration is not a meaningful χ error.** χ scales as (1 + gain)². The
  uniform-water estimate for T1 gives ×1.0035 [1.0003, 1.0075]. The least
  favorable estimate here comes from the one-offset fit on the pipeline's
  short profile, a model shown below to inflate gain: ×1.011 [1.004, 1.016].
  Either way, calibration contributes about 1% or less.
- **JAC salinity reads ~0.014 PSU high** (conductivity +13 µS/cm in uniform
  water, against +1.2 µS/cm between the SBE's own two cells). That is a constant,
  so N² is unaffected. Subtract it if absolute salinity from the VMP is ever used.

**What this does NOT test:** the FP07 **frequency response** (τ), which governs χ
far more than calibration does. It is also **one cast**, rosette-mounted and
heaving: 10% of downcast samples have the rosette moving up, and 25% fall slower
than perturb's 0.3 dbar/s profile threshold. The SBE's own calibration dates
were not checked.

## Rosette motion

The rosette hangs on a tensioned wire. Until enough wire is out, the ship's roll
and pitch move it (Pat, 2026-09-17). The fall rate from DAT_166's pressure,
downcast only:

| dbar | median fall rate | moving up | below perturb's 0.3 dbar/s | oscillation half-range (p95−p5)/2 |
|---|---|---|---|---|
| 0–25 | 0.24 dbar/s | **30%** | 55% | 0.73 dbar/s |
| 25–50 | 0.49 | 2% | 21% | 0.45 |
| 50–75 | 0.45 | 2% | 25% | 0.40 |
| 75–100 | 0.52 | 0% | 22% | 0.37 |
| 100–125 | 0.78 | 8% | 14% | 0.67 |
| 125–225 | 0.92–1.05 | 0% | 0% | 0.31–0.54 |
| 225–250 | 0.88 | 5% | 14% | 0.64 |

- **The reversals are where the wire is short**: 30% of samples in the top 25
  dbar, none from 75 to 225 dbar. At 225–250 dbar the winch slows toward the
  stop.
- **What split perturb's profile at ~100 dbar was winch speed, not reversals.**
  Above 100 dbar the payout was slow (median ~0.5 dbar/s). The remaining
  oscillation then pushed about a quarter of the samples under the 0.3 dbar/s
  threshold. Below 125 dbar the median is ~0.95 dbar/s and none fall under it.
  The 100–125 dbar band is the speed change.
- **An oscillation persists at depth**, with a spectral peak near **7 s**
  (150–245 dbar). That is consistent with ship motion, though not checked
  against the ship's attitude record. It never reverses the rosette once the
  descent is ~1 dbar/s.

This is why the comparison bins by pressure over every downcast sample. As a
check, dropping samples slower than 0.2 dbar/s moves no temperature median or
fitted offset by more than 0.6 mK (`crosscheck.out`, the two passes of chain A).

## How

**Reference.** `../../uctd/calibration_casts/RR1704_10.cnv` is Seasave output
**bin-averaged to 1 dbar**, 1–252 dbar, with **no time column**, and carries
two temperature and two conductivity sensors. So every comparison is made in
**pressure**, on the SBE's own bins: VMP downcast samples are averaged over
(k − ½, k + ½] dbar. Bins shallower than 5 dbar are dropped, where pumps
start and the surface soak sits.

**Vertical offset.** The VMP rides at a different height on the rosette, with
its own pressure sensor and sensor lags. Where temperature changes with depth,
that offset looks exactly like a temperature error. Each difference is therefore
modeled as `ΔX = a + b·dRef/dp + s·(T − T̄)`, and the **uniform-water bins**
(|dT/dp| < 0.005 K/dbar, still spanning 15.6–23.7 °C) are reported as the
model-free check. The fitted offset is ~0.6 dbar above 100 dbar and ~1.1 dbar
below. That is geometry and lag, and irrelevant to profiling products.

**Three chains** (`crosscheck.py`):
- **A.** JAC_T and JAC_C vs SBE, the reference the FP07s are fit to. JAC_C is
  time-aligned to JAC_T with `ct_align`, as the pipeline does.
- **B.** `perturb run` on DAT_166 with Taiwan17's own config (`perturb_rosette.yaml`).
  Its real output. **Heave broke the downcast up: perturb found one profile,
  102.7–248.7 dbar**, so its FP07 fit spans only 6.9 K (it warns under 8 K).
  In that range the model puts T1/T2 at −11 to −13 mK from the SBE, against
  −4 to −5 mK for JAC_T there. That is not representative: the processed casts' fits
  span a median 8.2 K.
- **C.** The same `fp07_calibrate`, with Taiwan17's settings (order 2, JAC_T,
  lag search to 10 s), fitted over the **whole** downcast. This stands in for a
  free-fall cast, and it is the FP07 row in the table. Fitted lag: −0.078 s on
  both. Order 2 reproduces JAC_T with a residual MAD of 4.1 mK, against 9.0 mK
  for order 1, which confirms `fp07.order: 2`.

## A gain that did not survive

The first fit, with one vertical offset for the whole cast, gave JAC_T a gain of
+3.8 mK/K [+2.4, +5.2]. That would be a 0.8% χ error with a tight interval.
It is an artifact of that model (`gain_invariance.py`):

| model | JAC_T gain, mK/K [95%] | residual MAD |
|---|---|---|
| one vertical offset | +3.77 [+2.40, +5.22] | 15.8 mK |
| offset above / below 100 dbar | +2.11 [+0.99, +3.31] | 10.4 mK |
| offset + lag × descent rate | +2.71 [+1.51, +4.04] | 12.6 mK |
| both | +1.75 [+0.57, +2.98] | 10.8 mK |
| **uniform-water bins only, no offset term** (\|dT/dp\| < 0.01, 15.6–23.7 °C) | **+1.08 [−0.18, +2.11]** | 3.2 mK |

The gain shrinks as the offset is described better, and the residual falls with
it. The bins that cannot alias a gradient show nothing distinguishable from
zero. (At the tighter 0.005 K/dbar threshold, resampling can collapse the
temperature span, so that bootstrap interval is degenerate and is not quoted.)
Intervals are 10-dbar block bootstraps, because neighboring bins are not
independent.

## Files

| | |
|---|---|
| `perturb_rosette.yaml` | `../perturb.yaml` with the input, output and GPS paths changed and diagnostics on. **Output stays in `./Processed`**, never `../Processed`. |
| `crosscheck.py` | Chains A, B, C: offsets, uniform-water medians, vertical-offset fits. |
| `gain.py` | Offset, vertical offset and gain with bootstrap intervals. |
| `gain_invariance.py` | Whether the gain survives changes to the vertical-offset model. |
| `*.out`, `perturb_run.log`, `Processed/` | The outputs of the run recorded above. |

```bash
cd /Volumes/SeaChest/Taiwan/Taiwan17/vmp/rosette_crosscheck
perturb run -c perturb_rosette.yaml -j 1      # ~1 min; one profile
python crosscheck.py > crosscheck.out
python gain.py > gain.out                     # ~1 min (bootstrap)
python gain_invariance.py > gain_invariance.out
```

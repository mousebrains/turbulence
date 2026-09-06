# Will dives break the α / U_EM-scale degeneracy? — a forecast, pre-registered

**Status: PREDICTION, recorded before the data.** The OSU dives have flown, but
the data had not reached this machine when this was written (2026-09-06). The
numbers below are therefore a forecast, not a result, and they are on the record
so the comparison later is honest.

Produced by `scripts/will_dives_work.py` — a Monte Carlo, deterministic at
`seed=11`, 400 trials per configuration.

---

## 1. The problem

For a steady glide the observable per profile is

```
excess(θ) = U_EM · sin θ / |W| = b · sin θ / sin(θ + α)
```

with two unknowns: the angle of attack **α** and the EM scale error **b**. A
single pitch cannot separate them — they trade off almost exactly, which is why
every climbs-only deployment returns a bracket on `c` and `k` that straddles
zero. Two well-separated pitches can separate them, through the curvature of
`sin()`. *How well* is the question, and it depends on the separation, the
per-profile scatter, and how many profiles of each.

## 2. What was assumed

| | value | where from |
|---|---|---|
| per-profile scatter σ | 0.013 (1.3 %) in `excess` | measured: 721 osu685 climb profiles, bootstrap SEM 0.0005 on the median |
| α (truth) | 3.0° | assumed |
| b (truth) | 1.10 | assumed |

The σ is measured; the two truths are not, and the forecast is conditional on
them. It is a statement about **precision** — how tightly the geometry lets you
recover α and b — not about accuracy.

## 3. The forecast

68 % intervals (16th/50th/84th percentile) recovered from simulated data:

```
                             configuration   sep     alpha 16/50/84        b 16/50/84   corr
  osu685 as flown (climbs only, 40.4-48.0)   7.6   2.59/ 2.98/ 3.34   1.093/1.099/1.106  +0.998
osu684 post-failure (climbs only, 35.8-39.4) 3.6   0.87/ 2.80/ 5.41   1.051/1.096/1.154  +0.999
  PLANNED: dives 25 + climbs 37.5, 10% dives 12.5   2.87/ 2.99/ 3.13   1.097/1.100/1.103  +0.987
  PLANNED: dives 25 + climbs 37.5, 25% dives 12.5   2.91/ 2.99/ 3.08   1.097/1.100/1.102  +0.977
  PLANNED: dives 25 + climbs 37.5, 50% dives 12.5   2.93/ 3.00/ 3.07   1.098/1.100/1.102  +0.972
if climbs were 45 instead of 37.5, 25% dives 20.0   2.94/ 2.99/ 3.06   1.098/1.100/1.101  +0.943
```

**Yes, dives work — and a small fraction is nearly all of the benefit.**

- **Dives are worth far more than more climbs.** Climbs-only at 7.6° separation
  gives α to ±0.4°. Adding dives at 25° — even at **10 %** of profiles — gives
  **±0.13°**, a threefold improvement, because the separation goes 7.6° → 12.5°.
- **Going past ~10 % dives buys almost nothing.** 10 % → 50 % improves α only
  from ±0.13° to ±0.07°. **Do not trade away science time for a large dive
  fraction**; roughly one profile in ten is enough.
- **osu684 post-failure was never going to work.** At 3.6° separation the
  interval is 0.87–5.41°, which is no measurement at all. That record cannot
  answer this question and should not be asked to.
- **The degeneracy is broken, not eliminated.** The α–b correlation falls from
  +0.998 to +0.972, so the two are still strongly coupled. Report them jointly,
  with the covariance, and never quote one having fixed the other.
- Steeper climbs would help further (20° separation → ±0.06°) but the gain over
  dives-at-10 % is small; the separation is what matters, not which end supplies
  it.

For scale: **±1° in α is ±2 % in U and ±8 % in ε**, since ε goes as U⁻⁴. So
±0.13° is ~1 % in ε, and ±0.4° is ~3 %.

## 4. What would refute this

When the dive data arrives, run `scripts/will_dives_work.py` again with the
geometry **as actually flown** (real dive/climb pitches and the real profile
counts), then fit the real `excess(θ)` and compare:

1. **The recovered interval is much wider than forecast.** Most likely cause is
   that the real per-profile σ exceeds the 1.3 % measured on osu685 — dives may
   simply be noisier than climbs, which this forecast assumes they are not.
2. **The fit is poor rather than imprecise** — α and b do not reconcile
   dive-side and climb-side at all. That falsifies the *model*, not the
   geometry, and the first suspect is that α is not a single constant.
   `ru33_depth.py` already saw depth structure in `excess` (1.12 at 55 m → 0.98
   at 1000 m) that a climb-only record could not decompose, because pressure,
   speed, temperature and buoyancy trim all move together. Dives break that
   confound too, so this is the more interesting failure.
3. **α comes back near zero or negative.** That would say the geometry is fine
   and the premise was wrong — the excess is an EM error, not an angle of
   attack.

Record the outcome here rather than editing the forecast.

## 5. Related

- `scripts/em_bench_zero.py` — the bench soak that attacks the same degeneracy
  from the other side, by varying conductivity instead of pitch.
- `docs/glider_speed_problem.html` — the statement of the speed problem.
- `../AEM1-G/scripts/glider_aoa_check.py` and `aoa_histogram.py` — the α
  analysis on the climbs-only records, which is what this forecast says is
  limited to ±0.4°.

# Trimming the top of a VMP profile — what we learned, and how to check it

**Status:** derived across four campaigns (ASTRAL 2023, Taiwan17, CASPER-West
2017, SUNRISE 2019/2021/2022) between 2026-09-20 and 2026-09-21. Every number
here was measured on real data; where something is inferred rather than
measured it says so.

---

## 1. There are THREE distinct contaminations, not one

They have different physics, different depth scales, and different sensors
that can see them. Conflating them is how the trim goes wrong.

| # | contamination | cause | depth scale | what can see it |
|---|---|---|---|---|
| 1 | **Instrument attitude** | the VMP is launched near-horizontal and must right itself | 3.8–4.1 m to vertical, +2 m to settle | `Incl_Y` (directly) |
| 2 | **Prop wash** | ship's propellers, advected under the hull | 5–30 m, vessel-dependent | high-passed fall-rate residual |
| 3 | **Entry transient / body ringing** | mechanical shock of water entry | < 5 m | `Ax`/`Ay` (piezo vibration) |

**Attitude is the largest and the most often missed.** Until the instrument is
vertical the shear probes see a **mean cross-flow** rather than turbulence.
Since `epsilon ~ shear² / U⁴`, the reported dissipation there is *meaningless,
not merely contaminated*. Measured on SUNRISE, `epsilon` runs **4.5×10⁵** the
interior value at 1 m. The magnitude follows from geometry alone: at 38° off
vertical the mean cross-flow is `U·sin(38°) ≈ 0.46 m/s` against turbulent
fluctuations of order 10⁻³ m/s — a velocity ratio ~460, so ~2×10⁵ in shear
variance.

### Attitude is universal across the fleet

| campaign | Incl_Y at descent start | \|Incl_X\| | z_vert median | p90 |
|---|---|---|---|---|
| SUNRISE 2021 WS SN194 | 22.6° | 51.4° | 3.92 m | 5.70 |
| CASPER-West SN194 | 57.5° | 20.1° | 3.80 m | 5.66 |
| Taiwan17 SN142 | 29.4° | 27.0° | 3.87 m | 5.01 |
| ASTRAL 2023 SN92 | 44.2° | 23.6° | 4.14 m | 6.71 |

Three ships, four instruments, eight years, and the settling depth is
**3.8–4.1 m everywhere**. A VMP is always put over the side at some attitude
and has to right itself.

**But it only BINDS where the prop wash is weak.** Where the wash trim already
lands deeper, attitude contamination is removed incidentally:

| campaign | trim median | z_vert + 2 m | outcome |
|---|---|---|---|
| ASTRAL | 11.01 | 6.1 | covered by 4.9 m |
| Taiwan17 | 9.01 | 5.9 | covered by 3.1 m |
| CASPER-West | 7.01 | 5.8 | **marginal (+1.2 m) — and it failed, see §5** |
| SUNRISE at the ASTRAL default | 3.50 | 5.9 | **short by 2.4 m** |

---

## 2. Which sensor sees what — and the trap in each

### `Ax`/`Ay` are VIBRATION sensors, not accelerometers

On a Rockland VMP these are piezo sensors for Goodman coherent-noise removal,
and **there is no `Az`**. They measure body ringing, which damps within ~5 m,
so they are **blind to prop wash** — an advected *flow* perturbation that
persists far deeper. Measured on ASTRAL: Ax/Ay std peaks at 1.44× background
and is flat below 5 m, while the fall-rate residual peaks at 44× and decays
over ~30 m. Feeding only Ax/Ay trims to a median 3 dbar and leaves **two
decades of prop-wash epsilon** in the product.

### The fall-rate RESIDUAL, not the raw fall rate

Raw fall rate stays elevated at depth through genuine ocean turbulence. The
high-passed residual about a smooth descent settles to a flat background.

> **THE SMOOTHING WINDOW IS NOT A FREE PARAMETER.** The residual is
> `w − movmean(w, window)`, so the window sets the high-pass corner and is
> only meaningful *relative to the cast length*. Once it is a large fraction
> of the cast, the "smooth descent" baseline absorbs the wash and the residual
> loses the signal it exists to detect.

| group | @2 m | @4 m | @6 m | @10 m |
|---|---|---|---|---|
| 2019 Pelican SN194 (60–67 m casts) | 10.5× | 5.9× | 4.0× | 2.6× |
| 2022 PointSur SN194 (34 m) | 5.7× | 4.1× | 1.6× | 1.0× |
| 2021 Pelican SN412 (20 m) | 3.6× | 1.3× | **0.8×** | — |

At perturb's 6.0 m default, **two of five SUNRISE vessel-years read as having
no prop wash at all**. At 2 m the wash is plainly there in every one. Roughly
**10% of the cast span** worked on both ASTRAL (long casts) and SUNRISE
(short). The caller now warns when the window exceeds 30% of the cast.

This is the same class of error as the earlier `residual_smooth_sec` →
`residual_smooth_m` fix: units in seconds meant 12.8 m on a fast-falling
Taiwan13 cast but 6.3 m on a slow ASTRAL one. Fixing the units removed the
fall-rate dependence but **not** the cast-length dependence.

### `Incl_Y` — read the ANGLE, not its variance

The inclinometer's *variance* responds to ocean turbulence and would
over-trim, which is why it must not be fed to `compute_trim_depth`. Its
*absolute value* answers a different question — "is the instrument vertical
yet?" — and is the right input to `attitude_trim_depth`. Both statements are
true; they use different properties of the same channel.

**Its decisive advantage: no parameter to tune against cast length.** On 466
real SUNRISE casts, validated against an independently measured epsilon knee
of 5–6 dbar:

| setting | median | p25–75 | p90 |
|---|---|---|---|
| ASTRAL default (6 m resid, 50 m range) | 3.50 | 2.01–4.51 | 6.00 |
| tuned residual only (2 m, nf 1.5) | 4.51 | 3.01–6.01 | 7.51 |
| **inclinometer only** | **5.94** | 5.14–6.95 | 7.72 |
| inclinometer + tuned residual | 6.50 | 5.50–7.43 | 8.01 |

The attitude voter lands on target with **no tuning**; the residual voter
needed its window changed and still came in ~1.5 m shallow.

**VMP-ONLY.** `+90° = nose down` is a VMP convention. On a MicroRider `Incl_Y`
is roughly pitch — do not enable it there. And a glider-borne MR has no ship
wash at all, so none of §1's items 1–2 apply to it.

---

## 3. Hardware fails, and the obvious guard does not catch it

**VMP SN 412's inclinometers drop out intermittently** — both channels, on 61%
(2021) and 83% (2022) of casts. Within a cast the reading is either
bit-identical or a normal ~68° sweep; the intermediate bins are **empty**, so
detection is unambiguous.

> **The trap:** SN 412's value is frozen *within* each cast but **changes
> between** casts, so across a whole file it spans 6.3–90.0° over 3,033
> distinct values. **Any whole-file "is this sensor alive" test passes it**
> and then hands the voter a frozen angle. A file-level health check was
> nearly shipped before the data was looked at.

The dangerous case is frozen **above** threshold: it mimics "already
vertical", the voter abstains, the caller reads "no attitude transient", and a
cast whose first metres are meaningless passes through untrimmed. Hence
`inclinometer_is_usable()` is **per-profile**, is called both inside
`attitude_trim_depth` and by the pipeline caller, and the caller counts and
names the frozen casts.

Do not characterise such a fault from a couple of files: two different 2-file
samples of SN 412's 2022 deployment gave 94% and 27% frozen, against the
13-file value of 83%.

**Fallback when the sensor is dead.** Where the VMPs were swapped so one could
charge, each deployment has a healthy companion under identical handling, and
the transfer can be *verified* on the broken unit's working casts rather than
assumed:

| deployment | companion | SN412 working casts | Δ |
|---|---|---|---|
| 2021 Pelican | SN194 4.77 m | 5.35 m | +0.58 |
| 2022 Pelican | SN142 3.98 m | 3.73 m | −0.25 |

Caveat: SN 412's dropout is largely whole-file, so its working casts are a
*time-selected* subset — the likeliest reason 2021's tail runs deeper.

---

## 4. `min_depth` is NOT a floor

`top_trim.min_depth` is the **top of the search window**, and the elevated run
is anchored to it. Raising it to 20 would find nothing at 20 m, abstain,
return `None`, and trim **nothing**. A floor must be imposed with
`profiles.P_min`, which masks samples.

**A FLOOR AND THE DETECTOR COMPOUND — they do not bound each other.**
`profiles.P_min` does not merely mask samples below itself; it moves where
PROFILE DETECTION starts, so `top_trim`'s search window then runs on an
already-truncated segment and trims FURTHER from there. Measured on
CASPER-West, the detector's extra depth is **additive and near-constant at
~4.5 m**:

| floor | effective median trim | 10-12 dbar windows retained |
|---|---|---|
| none | 8.50 | 2530 |
| 8 dbar | **12.50** | **357** |
| 6 dbar | **10.50** | 2421 |

At an 8 dbar floor the effective trim landed 4.5 m deeper than intended and
emptied the 8-12 dbar band — which carries the STRONGEST wind correlation
(0.29 at 10-12), i.e. genuinely wind-forced mixing, not contamination.
Dropping the floor to 6 recovered that band essentially intact (2421 of 2530)
while leaving only 90 windows of residual excess in 6-8 dbar, 0.08% of the
data and down 92% from the 1105 with no floor at all.

**So set the floor to the depth you want MINUS the detector's contribution,
and verify the effective trim rather than the configured one.** Do not assume
the floor is what the product got.

**When a floor is right, and when it is not.** A flat floor over-trims quiet
stations and under-trims active ones, which is why it was dropped on ASTRAL —
wash depth tracks thruster usage and varies within a cruise. Use one only
where the contamination is *invariant*. On SUNRISE both conditions exist in
different bands:

| z | p10/ref | p50/ref | p90/ref | % windows > 3× |
|---|---|---|---|---|
| 1 | 107,204 | 527,021 | 2.9e6 | **100.0%** |
| 2 | 760 | 75,738 | 299,035 | **99.7%** |
| 3 | 1.6 | 23.7 | 13,399 | 81.7% |
| 6 | 0.4 | 1.2 | 14.0 | 28.2% |

0–2 m is contaminated on essentially every window of all 36 stations —
invariant, so a floor is right. 3–6 m varies by three orders of magnitude
between stations — the per-cast voter's job.

---

## 5. How to VALIDATE a trim — four techniques, in order of strength

### 5.1 Measure contaminated epsilon directly (strongest)

Process one deployment with `top_trim.enable: false` and look at epsilon
versus depth. This decides the trade on **the quantity the product actually
reports**, not a proxy. SUNRISE generation `_00` of 2021 Walton Smith SN194
(5,553 profiles, 110,379 windows):

| z | ε median | ε/interior | Γ |
|---|---|---|---|
| 1 | 1.1e-2 | 4.5×10⁵ | — |
| 2 | 1.5e-3 | 6.2×10⁴ | 0.196 |
| 3 | 5.0e-7 | 20.9 | 0.232 |
| 5 | 3.5e-8 | 1.48 | 0.164 |
| 6 | 2.3e-8 | 0.96 | 0.146 |

A **3,200× fall in the single metre between 2 and 3 m** is an interface, not
the gradual decay of advected wash — the signature of attitude, later
confirmed directly from `Incl_Y`.

### 5.2 Wind correlation and law-of-the-wall (independent of every motion proxy)

Wind-forced turbulence obeys `eps = u*³/(κz)` and **correlates with wind
speed**; ship-injected turbulence does not — the propellers do not care how
hard it is blowing. The **correlation is the sharper half**: it is
dimensionless and needs no drag coefficient or assumption that the wall layer
applies.

ASTRAL (RR2306 met):

| z | eps/(u*³/κz) | corr(log U, log eps) |
|---|---|---|
| 4.5 m | 252 | −0.26 ← ship, not ocean |
| 8.5 m | 20 | −0.08 |
| 10.5 m | 2.7 | +0.03 |
| 22–35 m | 0.7–1.5 | +0.46…+0.63 ← genuinely wind-forced |

CASPER-West (Sally Ride SCS met, 2,097 profiles, 100% wind-matched):

| z | n | ε median | ε/wall | corr |
|---|---|---|---|---|
| **4–6** | 287 | **5.0e-6** | 744 | 0.11 |
| 6–8 | 1105 | 3.5e-7 | 69 | 0.17 |
| 8–10 | 2075 | 7.7e-8 | 24.7 | **0.24** |
| 20–22 | 2629 | 1.7e-8 | 11.9 | 0.19 |
| 100–130 | 22550 | 8.2e-10 | 4.6 | 0.07 |

The correlation is **lowest exactly where ε is most anomalous** and peaks at
8–12 dbar. **This found a real defect**: 900 of 2,097 CASPER-West profiles
(42.9%) retain data shallower than 8 dbar, and for those, ε at 4–8 dbar is a
median **25.2×** *their own* 15–30 dbar value (p90 1,487×). The per-profile
comparison is the strong form — it controls for spatial and temporal
variability.

> **Caveat on the absolute ratio.** CASPER-West was nearly windless (median
> 2.01 m/s, max 10.96), so `u*³` is tiny and `eps/wall` is inflated
> everywhere; a narrow wind range also weakens the correlation's power. The
> *depth structure* of the correlation remains meaningful; the absolute ratio
> is far less informative than on ASTRAL.

> **Settle the wind UNITS from the data.** `eps_wall ~ U³`, so a knots/m-s
> confusion is a factor of **7.3**. On Sally Ride the SCS `.acq` declared no
> units; ship speed maxing at 12.70 pins it to knots (she cannot make 21 kt),
> and a true-wind vector closure — `TW` against `WS`/`SP`/heading — agreed to
> a median 0.65 kt over 2.49 M samples, proving the three share one unit.

#### 5.2.1 On a NEW ship or platform, this is the method of first resort

Everything else in §5 needs something you do not yet have. The residual
voter's window must be scaled to the cast length; its `noise_factor` must be
scaled to how strong the wash is; a floor needs to know how deep the
contamination reaches. On a vessel you have worked before you can carry those
over. **On a new ship you have no prior at all**, and tuning a proxy against
data you do not understand is how you end up confirming your own assumption.

The wind test needs none of that. It requires no vessel knowledge, no
tuning constant, and no assumption about the depth scale — and it is the
**only technique here that identifies the SOURCE rather than the depth**.
The motion proxies answer "where is the instrument still disturbed?"; the
wind test answers "is this the ship or the ocean?", which is the question the
trim actually turns on.

Recommended sequence for a platform you have not characterised:

1. **Process one deployment with `top_trim.enable: false`.** The test runs on
   epsilon, so there must be a product to test, and it must be untrimmed or
   the evidence has already been thrown away.
2. **Secure the ship's underway met**, which is a task in its own right (see
   below).
3. **Run the wind test.** The depth at which `corr(log U, log eps)` turns
   positive and the ratio approaches O(1) is the bottom of the
   ship-contaminated layer. That number is the trim, and it was derived
   without touching a motion proxy.
4. **Set the trim, re-run, and repeat the test** to confirm the excess is
   gone. This is not ceremony: on CASPER-West the medians looked reasonable
   while 43% of profiles were under-trimmed.

**Get the met early, because it is routinely missing or misfiled.** In four
campaigns it was never simply present alongside the microstructure:

* **SUNRISE** had no met on disk at all. It came from the project's shared
  Google Drive (2019 MIDAS, 2021 raw SCS streams, 2022 a processed `ShipDas`
  NetCDF), 7.3 GB fetched after the fact.
* **ASTRAL**'s `Data/ship/met` held R/V Revelle **RR2202** — a different
  cruise entirely, misfiled. The real met had to come from R2R.
* **CASPER-West**'s met existed in four separate places (`met/`, `ship/met/`,
  `tpw/met/`, `share/ships_met/`), and the small, obvious-looking
  `met_avg.mat` contains **no wind at all** — only air and water temperature.
  The wind is in a 105-column SCS table under terse names (`WS`/`WD`
  relative, `TW`/`TI` true) with the clock in a column called `ZD`.
* **Rutgers RU33** publishes to ERDDAP, but the curated product carries no
  buoyancy variable and the raw files needed a sensor-list cache from the
  operator.

If a cruise is still being planned, ask for the underway record explicitly.
If it is already over, R2R (`rvdata.us`) holds processed navigation for many
UNOLS vessels — though not all: PE22-01 (Pelican, SUNRISE 2021) returned "No
products Found" while WS21170 (Walton Smith) had a full Applanix POS/MV
product.

**Check the wind distribution before trusting the result.** The test's power
comes from wind *variability*, and a calm cruise cannot supply it:

| campaign | wind median | max | verdict |
|---|---|---|---|
| ASTRAL 2023 | — (wide range) | — | ratio AND correlation both informative |
| CASPER-West 2017 | 2.01 m/s | 10.96 | ratio inflated everywhere; use the correlation's DEPTH STRUCTURE only |

With `u*³` tiny in near-calm, `eps/wall` is large at every depth and says
little. The *shape* of the correlation against depth survives; the absolute
ratio does not. Report which one you are relying on.


#### 5.2.2 Ship SPEED THROUGH WATER predicts the wash — ask for the speed log

Pat's operational observation: *"there are times when the VMP is falling
through 'clean' water and others when it is not."* Speed through water (STW)
is the covariate that predicts which.

Measured on CASPER-West, generation `_01` (no floor), each speed class
normalised by **its own** 20-40 dbar median so ocean variability cannot
explain the difference:

| STW class | n | 4-6 | 6-8 | 8-10 | 10-12 | 12-16 | 20-30 |
|---|---|---|---|---|---|---|---|
| 0.0-0.5 kt (hove to / DP) | 436 | **897x** | 52.3 | 11.6 | 6.9 | 5.0 | 1.5 |
| 0.5-1.5 kt (drifting) | 1581 | 535x | 42.6 | 7.7 | 4.1 | 2.9 | 1.3 |
| 1.5-3.0 kt (slow) | 79 | — | **17.0** | 8.1 | 5.9 | 2.5 | 1.2 |

Absolute 2-8 m median epsilon: **1.01e-6 -> 7.01e-7 -> 1.44e-7**, a factor of
**7 cleaner** making way than holding station. The mechanism is
straightforward: on station the thrusters run continuously and the wash has
nowhere to go; making way it is advected astern and the instrument falls
through cleaner water.

**This is the strongest argument yet against a flat per-cruise floor** — the
contamination has a known, measured, per-cast driver.

**STW is NOT speed over ground.** On Sally Ride `SL - SOG` had a median of
-0.20 kt (p10 -0.70, p90 +0.23) — that difference is the current. Use the
speed log. On an SCS ship it is an NMEA `$VDVBW` sentence
(`SENSOR = Speed Log`); on Sally Ride it came from the UHDAS/RDI unit under
tags `SL` (longitudinal) and `SX` (transverse), in KNOTS. **Ask for it when
you ask for the met** — it is a different sensor and is easy to omit.

**Bounds on this result.** Per-profile STW here was median 0.77 kt, p75 1.03,
max 3.68 — an almost entirely on-station cruise, so the genuinely-underway
regime (>3 kt) is unsampled. Speed and location are also confounded (the ship
is stationary *at* stations), so the classes differ in place as well as
speed; the monotonic trend across three classes is hard to explain that way
but is not isolated from it. And the 2-3x residual at 12-20 dbar is probably
REAL: the wind test gives that band a correlation of 0.19-0.24. Normalising
to 20-40 dbar makes legitimate near-surface wind-forced mixing look like an
excess. The contamination proper is the 40-900x below 8 dbar.

#### 5.2.3 PROPULSION TYPE decides which covariate to use

The STW result above (§5.2.2) came from **Sally Ride: twin fixed-shaft
controllable-pitch propellers**, diesel-electric, with azimuthing bow and
stern tunnel thrusters for DP. It should NOT be assumed to transfer to a
different propulsion arrangement.

| propulsion | ships | wash behaviour | covariate to use |
|---|---|---|---|
| **fixed shaft** (incl. CPP) | Pelican, Point Sur, Walton Smith (cat, variable-pitch), **Sally Ride** | wash direction fixed relative to the hull; intensity set by thrust and flushing | **STW** works — monotonic, factor 7 (§5.2.2) |
| **Z-drive / azimuthing main** | **Revelle, Thompson** | drives act as the RUDDER: they swing on a macro scale with wind and current, AND continuously on short scales to hold heading or COG. **Wash direction is a fast-varying unknown.** | STW is NOT sufficient — need drive azimuth, which we do not have logged |
| **jet DP, main prop LOCKED** | **E/V Nautilus (2024)** | no propeller wash at all on DP; the source is jet wash, with different geometry | unknown — to be characterised |

**Why Z-drive ships are the hard case** (Pat, operational): on Revelle and
Thompson the Z-drives *are* the rudder. They rotate on a macro time scale as
wind and current change, and on short time scales they are constantly
adjusting to hold course-over-ground or heading. So the wash **direction**
relative to the instrument changes faster than any cruise-level setting can
track. This is the mechanism behind the observation that started this work —
*"there are times when the VMP is falling through 'clean' water and others
when it is not"* — and it is why ASTRAL's per-cast wash depth spanned
3.5-29.5 m (IQR 13.5-25.5), a factor of 4.4, on a single cruise.

**Consequence: on a Z-drive ship the per-cast detector is not a refinement,
it is a requirement.** A flat floor cannot track a covariate that changes on
the timescale of a single cast.

**We have no drive-azimuth logging** for either Revelle campaign (Taiwan17's
`ship_data/met/met.mat`, ASTRAL's R2R-derived met). If you can get it, ask
for thruster azimuth and RPM/pitch. An available proxy is the **crab angle**,
heading minus course-over-ground: a ship holding heading against wind and
current with azimuthing drives will crab, and the rate of change of that
angle indicates how hard the drives are working athwartships.

**A CPP detail worth knowing:** on a controllable-pitch ship the shafts keep
turning at zero net pitch, so "stopped" is not "off". That is consistent with
hove-to being the DIRTIEST class on Sally Ride (§5.2.2) — spinning blades,
DP thrusters running, and no flushing.

**E/V Nautilus 2024 will be the interesting test.** Its DP uses jets with the
main prop **locked**, so the propeller-wash source is absent entirely and
whatever contamination exists has a different geometry. Operations ran from
stationary-over-ground to very slow, ~0.3-0.5 kt. Treat it as a NEW platform
(§5.2.1): characterise from an untrimmed pass rather than carrying any of the
numbers in this document across.

### 5.3 Instrument attitude (direct, per-cast, no tuning)

`Incl_Y` measures the thing itself. Use it as a voter, and as a check that any
other method's trim clears `z_vert + margin`.

### 5.4 Window-invariance test (catches the analyst, not the data)

Vary the residual smoothing window (2/4/6/10 m). **If the apparent wash depth
or peak ratio moves with the window, the number is a filter artifact.** A peak
ratio near 1.0 is evidence about the *window* until the short-window case has
been checked.

---

## 6. Hypothesis zero is that your own analysis is wrong

Four attributions were made and falsified in two days. All four were mine.

| claim | how it died |
|---|---|
| "SN412 is the noisy instrument" | pooled instruments within a vessel-year; splitting them showed 2021 Pelican **SN194** also had no peak |
| "the descent filter is clipping the surface" | descents start at a median **0.17–1.87 m** — it was not |
| "wash reaches 10–15 m" | came from ratios computed with the suppressing 6 m window against a contaminated background; the real answer is 5–6 m |
| "the near-surface spike is bubbles / tether snatch" | no evidence; **attitude** explains the universality and the step |
| "it is a free-fall speed artifact" | at 1 m the VMP is at 0.752 vs 1.040 m/s terminal — worth **3.7×**, against 4.5×10⁵ observed; `eps·U⁴` shows the probe really measured it |

Cheap tests that killed them: split the pooling, remove the filter, vary the
window, check the geometry's order of magnitude, and look at the raw quantity
(`eps·U⁴`) rather than the derived one.

**And one silent failure worth its own line.** A flight-model calibration
returned a plausible `Cd0` that was actually its untouched seed, because
xarray decoded a time coordinate to `datetime64[ns]` and a bare `.astype(float)`
yielded **nanoseconds** — so every `dt > 20 s` test fired and the mask
excluded 100% of samples. The only tell was `n=0` in the output. **A believable
number is not evidence that the code ran.**

---

## 7. Practical checklist

### A platform you have worked before

1. **Measure the cast span** (`p10/p50/p90`), do not assume it. A descent
   filter requiring >=8 m span hides short casts; a 10-file sample misled the
   2019 Pelican estimate by a factor of 3 in the other direction.
2. **Set `residual_smooth_m` to ~10% of the cast span.** Never copy it.
3. **Set `max_depth` to fit inside the cast** — with `quantile: 0.6` the wash
   must be < 40% of the search range.
4. **Enable `use_inclinometer`** on any VMP (never on a MicroRider), and read
   the run log for frozen-channel and never-vertical warnings.
5. **Run one deployment with `top_trim.enable: false`** and look at eps vs
   depth.
6. **Check per-profile, not just per-campaign.** CASPER-West's medians looked
   fine; 43% of its profiles were under-trimmed.

### A NEW ship or platform — different order, and the wind leads

The steps above all need a prior you do not have: a window scaled to the
cast, a `noise_factor` scaled to the wash, a floor scaled to a depth nobody
has measured on this hull. Start with the one technique that needs no prior
at all (§5.2.1).

1. **Ask for the ship's underway met AND the speed log before the cruise**,
   or chase them immediately after. The met was missing, misfiled or
   scattered in all four campaigns examined here. The speed log is a
   SEPARATE sensor and is easy to omit — but speed through water predicts
   the wash (§5.2.2), and it is not the same as speed over ground. R2R
   (`rvdata.us`) covers many UNOLS vessels after the fact, but not all.
2. **Check the wind distribution first.** A calm cruise cannot support the
   test: with `u*^3` tiny, `eps/wall` is inflated at every depth. Decide up
   front whether you will be able to use the ratio, the correlation's depth
   structure, or neither.
3. **Settle the wind UNITS from the data**, not from a document — `eps_wall ~
   U^3`, so knots vs m/s is a factor of 7.3. Anchor on ship speed (a hull has
   a known top speed) and confirm with a true-wind vector closure.
4. **Process one deployment untrimmed** (`top_trim.enable: false`).
5. **Run the wind test.** Where `corr(log U, log eps)` turns positive and the
   ratio approaches O(1) is the bottom of the ship-contaminated layer — the
   trim, derived without a motion proxy.
6. **Set the trim, re-run, and repeat the test** to confirm the excess is
   gone.
7. **Only then tune the motion proxies**, using the wind-derived depth as the
   target to reproduce. That is the order that keeps you from tuning a
   detector to agree with your own assumption.

## Related

`processing/top_trim.py`, `perturb/pipeline.py` (`_adjust_profile_bounds`),
`docs/mixing_efficiency.md`. Campaign findings:
`/Volumes/SeaChest/SUNRISE/Data/VMP_README.md`,
`/Volumes/SeaChest/CASPER/2017_West/vmp/analysis/wind_assessment.py`.

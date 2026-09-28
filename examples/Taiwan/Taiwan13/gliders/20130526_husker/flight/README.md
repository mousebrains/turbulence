# Through-water speed for MicroRider MR046 on glider husker — flight model

MR046 carried **no EM flowmeter**, so the speed that scales shear and
temperature-gradient spectra must come from a glider flight model. This
directory builds it and writes the perturb hotel file that carries it.

**Model: Lucas Merckelbach's**, through his own implementation `gliderflight`
1.2.1 (MIT, [doi:10.5281/zenodo.2222694](https://doi.org/10.5281/zenodo.2222694)):

- Merckelbach, Smeed & Griffiths (2010), *JTECH* 27, 547–563,
  [doi:10.1175/2009JTECHO710.1](https://doi.org/10.1175/2009JTECHO710.1) — steady state;
- Merckelbach, Berger, Krahmann, Dengler & Carpenter (2019), *JTECH* 36, 281–296,
  [doi:10.1175/JTECH-D-18-0168.1](https://doi.org/10.1175/JTECH-D-18-0168.1) — the
  **dynamic** model used for the product, built for microstructure on Slocums.

Built 2026-09-15.

## Files

| | |
|---|---|
| `dinkum-flight.yaml` → `flight_inputs.nc` | Flight-model inputs on the **flight computer's clock** (`m_present_time`, ~4.4 s): `m_pressure`, `m_pitch`, `m_roll`, **`m_ballast_pumped`** (measured) and `c_ballast_pumped` (commanded), `m_battpos`, plus the CTD projected from its own clock. Not the same file as `../hotel/hotel.nc` (CTD clock, no ballast/roll). |
| `mr_clock.py` → `mr1hz/`, `mr_clock_offsets.csv`, `mr_clock_model.json` | MR-vs-glider clock, per file and per clock block. |
| `make_flight_speed.py calibrate` → `calibration.json` | Model calibration, resumable. |
| `make_flight_speed.py run` → **`husker_flight_hotel.nc`** | **The perturb hotel file**: speed, angle of attack, CTD, **on the MR clock**. |
| `calibrate.log` | Log of the calibration run. |

## Reproduce

```bash
cd /Volumes/SeaChest/Taiwan/Taiwan13/gliders/20130526_husker/flight
dinkum-hotel build -c dinkum-flight.yaml           # ~5 s (cache in ../hotel/cache)
python mr_clock.py extract                          # needs the named-stanza reader fix (below); ~30 min, resumable
python mr_clock.py solve
uv run make_flight_speed.py calibrate               # ~30 min, resumable; pinned deps in the script header
uv run make_flight_speed.py run                     # ~1 min
```

`mr_clock.py extract` needs a `microstructure-tpw` that reads MR046's config
dialect (one named stanza per channel: `[pitch]`, `[shear1]` …): `main` at or
after **29f3ea9** (PR #196, merged 2026-09-16). **v0.4.0 and earlier build zero
channels** from these files.

## Inputs, and what was checked about them

- **Glider type: 200 m "shallow" pump Slocum.** Buoyancy is `m_ballast_pumped`
  (bang-bang ±200 cc in both blocks, commanded ±200); `m_de_oil_vol` is unused.
- **Density** from the glider CTD by TEOS-10, positions from `../hotel/hotel.nc`'s
  drift-corrected track.
- **Clocks.** The CTD clock and the flight clock agree to ±0.5 s (pressure vs
  pressure, sharpness 40–46). The MR clock does not (below).
- **Two data blocks**: 2013-05-02 04:45 → 05-03, and 05-19 → 05-26 07:42.
  Nothing between.

## MR clock

`glider_time = MR_time + L`, estimated per MR file by regressing glider pressure
at `MR_time + L` on MR pressure and minimizing the residual. A yo sawtooth makes
the minimum sharp, but it is periodic, so three gates must all pass:
sharpness > 5, minimum not on the ±3600 s boundary, and **ambiguity** (best cost
more than 300 s away ÷ best cost) > 3.

| clock block | files | L | drift | residual |
|---|---|---|---|---|
| A, 2013-05-01 … 05-03 | 2 | **−2.1 s** | — | 0.13 s |
| B, 2013-05-17 … 05-26 | 49 | **−14.30 s** at 2013-05-23T16:35Z | **+0.373 s/day** | 0.28 s RMS |

The MR clock was evidently reset between blocks. **FSA_011 was refused**: from
1,174 s of overlap it fitted −2715.5 s, one ~45-min yo cycle away from −2 s —
the aliasing the ambiguity gate exists to catch.

perturb's `hotel.time_offset` is a single number and cannot hold two blocks and
a drift, so **the hotel file is written already on the MR clock** (glider time
− L(t)) — use `hotel.time_offset: 0`. `glider_time` and `mr_clock_lag` are
carried in the file for audit.

## Calibration

`gliderflight` minimizes the mean square of (modeled glider w − observed depth
rate). Held-out samples (`calibration_excluded = 1`): depth < 10 m or > 175 m,
|pitch| < 15°, |measured − commanded ballast| > 10 cc (pump running), ±90 s of a
commanded ballast reversal, sample spacing > 20 s, |dh/dt| > 0.6 m/s.

1. **Whole deployment, steady state**, fits C_D0, V_g and hull compressibility ε:
   C_D0 = 0.157, ε = **5.83 × 10⁻¹⁰ Pa⁻¹** (then held fixed), 114,153 samples.
2. **Per UTC day, dynamic model**, fits C_D0 and V_g (±1800 s of warm-up data
   each side, excluded from the cost). Parameters are interpolated linearly
   between day centers and held beyond the ends.

| day (UTC) | C_D0 dynamic | C_D0 steady | V_g − m_g/ρ₀ | U dive | U climb | α dive | α climb | w_water MAD | n cal |
|---|---|---|---|---|---|---|---|---|---|
| 2013-05-02 | 0.1531 | 0.1531 | 76 cc | 0.343 | 0.308 | −2.45° | +2.42° | 1.61 cm/s | 8781 |
| 2013-05-03 | skipped (< 3000 samples) | | | 0.332 | 0.327 | −2.48° | +2.24° | 1.34 cm/s | 2734 |
| 2013-05-19 | 0.1490 | 0.1480 | 82 cc | 0.271 | 0.374 | −2.28° | +2.43° | 1.58 cm/s | 11208 |
| 2013-05-20 | 0.1514 | 0.1505 | 81 cc | 0.265 | 0.374 | −2.37° | +2.50° | 1.54 cm/s | 14772 |
| 2013-05-21 | 0.1480 | 0.1471 | 83 cc | 0.263 | 0.383 | −2.36° | +2.38° | 1.43 cm/s | 14892 |
| 2013-05-22 | 0.1514 | 0.1508 | 80 cc | 0.272 | 0.370 | −2.33° | +2.49° | 1.53 cm/s | 15007 |
| 2013-05-23 | 0.1565 | 0.1555 | 76 cc | 0.266 | 0.355 | −2.46° | +2.64° | 1.56 cm/s | 14620 |
| 2013-05-24 | 0.1684 | 0.1674 | 73 cc | 0.259 | 0.328 | −2.63° | +2.77° | 1.52 cm/s | 14140 |
| 2013-05-25 | 0.1791 | 0.1779 | 71 cc | 0.267 | 0.304 | −2.80° | +2.99° | 1.64 cm/s | 14405 |
| 2013-05-26 | 0.1840 | 0.1826 | 67 cc | 0.248 | 0.290 | −2.85° | +3.13° | 1.74 cm/s | 3594 |

U and α are medians over calibration samples; speeds in m/s. |α| over all
calibration samples: 2.10 / 2.50 / 3.11° (5 / 50 / 95%), with 99.6% inside
Kokoszka et al. (2025)'s 1.5–4.5° QC band — a plausibility check, not a
validation (the band is itself flight-model based).

### What the table says

- **Drag rises monotonically from 2013-05-22** (C_D0 0.151 → 0.184, +22%) while
  the glider gets heavier (V_g − m_g/ρ₀ 83 → 67 cc). Both are what biofouling
  would do; **that attribution is a hypothesis**, not verified (no photos or
  recovery notes examined). Climb speed falls 0.38 → 0.29 m/s. **This is why the
  calibration is per day**: one deployment-wide fit would put the late climbs
  ~20% fast — roughly 2× in ε, since ε ∝ U⁻⁴.
- **May 2–3 dive FASTER than they climb**, the reverse of May 19–26, with the
  same ±200 cc. In-situ density explains it: 1022.4 kg m⁻³ (26.8 °C) then
  against 1024.3 (22.5 °C) later, which makes the glider ~90 cc heavier relative
  to the water. Not a model artifact.
- **Dynamic and steady-state agree** to 0.45% median in U over steady flight
  (5–95%: 0.989–1.027, one day checked), and C_D0 to within the optimizer's
  tolerance.

## Invariance tests (steady state, whole deployment)

| case | C_D0 | U dive / climb | α dive / climb |
|---|---|---|---|
| baseline (m_g = 52 kg) | 0.1570 | 0.2714 / 0.3491 | −2.48° / +2.60° |
| m_g = 60 kg | 0.1566 | 0.2701 / 0.3506 | −2.48° / +2.59° |
| start C_D0 = 0.10, 0.25, or ε = 1e-9 | 0.1570 | identical | identical |
| depth 20–160 m only | 0.1572 | 0.2701 / 0.3496 | −2.49° / +2.60° |
| \|pitch\| > 20° only | 0.1571 | 0.2714 / 0.3489 | −2.49° / +2.60° |
| block A only / block B only | 0.1545 / 0.1573 | 0.270 / 0.353, 0.271 / 0.349 | |
| block B first half / second half | **0.1490 / 0.1658** | 0.274 / 0.364, 0.268 / 0.335 | |

Only **time** moves the answer — the finding that led to the per-day calibration.
The mass is not identifiable (V_g absorbs it: U changes < 0.5% for 52 → 60 kg).

## An independent check exists — and it says the angle of attack is too small

A 2 MHz Nortek Aquadopp HR profiler flew on the sister glider **jane**
(2013-04-29 → 05-04) and measures the flow past the hull directly. Processed
2026-09-15 (`/Volumes/SeaChest/Taiwan/Taiwan13/aquadopp/analysis/`, with its own
README), and compared against the same Merckelbach recipe applied to jane over
326,280 samples:

| | U Aquadopp | U model | ratio | \|α\| Aquadopp | \|α\| model |
|---|---|---|---|---|---|
| dive | 0.257 m/s | 0.274 m/s | 1.065 | 4.55° | 2.91° |
| climb | 0.346 m/s | 0.359 m/s | 1.037 | 2.99° | 2.90° |

**On dives the model's α is ~1.6° too small, so its U is ~5–6% too high and any
ε built on it is ~20–25% too low**; climbs agree to 0.3%. The measured dive
value (4.3–4.6° at 22–25° pitch) matches Tanaka et al. (2022)'s ADCP-measured
Slocum α of −4.4° at −23.7° pitch, the only directly measured Slocum value
published.

**This has NOT been applied here, by decision (Pat, 2026-09-15): do not
propagate the angle of attack for now.** The ~20–25% dive-ε low bias is
therefore a known and deliberate state of the product, not an oversight — do
not "fix" it on rereading the Aquadopp result. It is a different glider on
different days,
the climb branch is less stable than the dive, and the systematics (±0.4°
alignment, ±0.5° flow distortion) are real though smaller than the discrepancy.
Treat it as the best available estimate of how wrong the speed is, and read that
README before using it to correct anything.

The full write-up — how the glider was identified, how the HR ambiguity was
unwrapped without a flight model, and every invariance test — is
`docs/aquadopp_flight_model_validation.html` in the microstructure-tpw repo
("The Aquadopp Check").

## Limitations — read before quoting ε

1. **No independent speed reference.** No EM, no ADCP. The calibration makes the
   modeled glider *vertical* velocity match the depth rate; the horizontal
   speed follows from the lift/drag parameterization (gliderflight's Slocum
   defaults: a_h = 3.8, AR = 7, Ω = 45°, C_D1 from its parameterization), which
   **was not calibrated** — pitch spans only ±25–27°, too narrow to separate the
   lift coefficient from C_D0. Tanaka et al. (2022) show coefficients and speed
   scale cannot be determined uniquely without another velocity source.
2. **Roll is not modeled.** husker flew with a persistent roll of −14.7°
   median (5–95%: −17.0 to −9.7°, calibration samples); `gliderflight` ignores
   roll. How a steady bank of that size changes the lift balance was not
   quantified here.
3. **Mass is nominal** (52 kg); harmless for U (above).
4. **C_D0 resolution ≈ ±0.0015** (~1%; ~0.5% in U, ~2% in ε). `gliderflight`
   calls scipy `fmin` with `xtol = 1e-2` (scaled units) and its default
   `ftol = 1e-4` — absolute, and comparable to the cost itself (~3e-4) — so it
   halts within that resolution of wherever it starts. Consequences: the
   per-day **dynamic** coefficients are the steady-state seeds to within
   resolution (2013-05-02's is the seed exactly; on doug, Taiwan 2017, every
   day's is), and the ~0.001 dynamic-minus-steady differences in the table are
   optimizer noise, not physics. Re-running 05-02 from C_D0 = 0.17 gave 0.1540
   at an indistinguishable cost. The dynamic model's contribution is its time
   integration, not different coefficients.
5. **2013-05-03** has too few calibration samples; it takes 05-02's parameters.

## `gliderflight` 1.2.1 pitfalls (handled in the script)

- `pressure` is in **bar**, though `set_input_data`'s docstring says Pa.
- `DynamicCalibrate` defaults to **k1 = 0.02**, but the paper gives m₁₁ = 0.2 m_g,
  m₂₂ = 0.92 m_g (below its eq. 11); k1 = 0.20 is set explicitly.
- A **callable** C_D0 / V_g / m_g reaches an undefined name `ti` in
  `DynamicGliderModel.integrate`; pass arrays.
- The dynamic model uses a process pool, so scripts need `if __name__ == "__main__"`
  (macOS spawn), or they die with `BrokenProcessPool`.

## Using it in perturb

```yaml
hotel:
  enable: true
  file: "<path>/husker_flight_hotel.nc"
  time_column: "time"          # ALREADY on the MR clock
  time_offset: 0.0
  max_gap: 30.0
  channels:
    speed: {fast: true}
    sci_water_temp: {units: "degree_Celsius", fast: false}
    sci_water_cond: {units: "mS/cm", fast: false}   # already mS/cm: no scale
speed:
  method: "hotel"
  hotel_var: "speed"
```

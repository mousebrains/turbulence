# Through-water speed for the MicroRider (SN 134) on glider doug — flight model

The MicroRider on doug (`../microrider/CAS_###.P`) carried **no EM flowmeter**,
so its speed comes from a glider flight model: **Lucas Merckelbach's**, through
his own `gliderflight` 1.2.1. The method, the choices and the `gliderflight`
pitfalls are identical to Taiwan 2013's and are documented once, in
**`/Volumes/SeaChest/Taiwan/Taiwan13/gliders/20130526_husker/flight/README.md`
— read that first.** This file records what is specific to doug.

Built 2026-09-15. Scripts here are copies of the husker ones; the only code
differences are paths, the output name, a fixed position, and a single clock
block.

## Files

| | |
|---|---|
| `dinkum-flight.yaml` → `flight_inputs.nc` | Flight-model inputs on the flight clock, from `../../20170216_doug_Taiwan/post-recovery/{flight,science}/logs` |
| `cache/` | A **copy** of `../../20170216_doug_Taiwan/cache` (so a build never writes into the source tree) |
| `mr_clock.py` → `mr1hz/`, `mr_clock_offsets.csv`, `mr_clock_model.json` | MR-vs-glider clock |
| `make_flight_speed.py calibrate` → `calibration.json` | Model calibration |
| `make_flight_speed.py run` → **`doug_flight_hotel.nc`** | **The perturb hotel file**, already on the MR clock |
| `make_gps.py` → `gps.nc` | Drift-corrected dead-reckoned track (husker's `make_corrected_track.py` algorithm), written on the MR clock; perturb's `gps:` block reads it. Needs `mr_clock_model.json`, so run it after `mr_clock.py solve`. |

Reproduce exactly as in the husker README (`dinkum-hotel build`, `python
mr_clock.py extract|solve`, `uv run make_flight_speed.py calibrate|run`), then
`python make_gps.py`, which husker does not have. This
MR's config uses modern `[channel]` stanzas, so any `microstructure-tpw`
reads it.

## Inputs

- **Source**: `20170216_doug_Taiwan/`, added to SeaChest 2026-09-15 (after the
  Taiwan17 reorganization, so not in `REORG_MANIFEST.csv`): 156 each of
  `.dbd/.mbd/.mlg/.sbd` and `.ebd/.nbd/.nlg/.tbd`, lowercase extensions.
- **200 m pump Slocum**: `m_ballast_pumped` ±233 cc (commanded ±233);
  `m_de_oil_vol` unused.
- **The flight record starts when doug started flying, 2017-02-19 11:43Z**, and
  nothing is missing. The mission log shows segment `0090` was `STATUS.MI` on
  2017-02-16, open 06:38:48 to 06:41:33 — a 2¾-minute deck check (30 records);
  `0091`/`0093` are further STATUS checks and `0092` is `lastgasp.mi`, all on
  02-19 morning. The science mission **`TW-CM-NS.MI` begins at segment `0094`,
  2017-02-19 11:43Z** (then `0096` on 02-22 and `0097` on 02-27; 54, 78 and 19
  segments). The MicroRider files from before that are out of the water too:
  CAS_001/002 (02-16) and CAS_003/004 (02-19) run 0.1–2.1 min at 0.3–1.3 dbar,
  while CAS_005 starts 02-19 11:55Z and dives to 159 dbar. **Every in-water MR
  file has flight coverage.**
- **Position** for TEOS-10 is fixed at the median of the 661 GPS fixes in
  `../NH_201702160638_doug.mat` (21.462 N, 119.420 E; fixes span 21.27–22.12 N,
  119.18–120.15 E) — at most ~1e-3 kg m⁻³ in density.
- **Clocks**: the CTD clock runs 0.5–1.0 s behind the flight clock
  (pressure vs pressure, sharpness 58, n = 647,200) — ~13 cm at doug's fall
  rate, negligible for salinity.

## MR clock — the MicroRider was 9 minutes 21 seconds fast

`glider_time = MR_time + L`, same estimator and gates as husker.

| clock block | files | L | drift | residual |
|---|---|---|---|---|
| 2017-02-19 … 02-28 | 70 | **−561.21 s** at 2017-02-23T21:31Z | **−0.444 s/day** | 0.33 s RMS |

Per-file minima are very sharp (sharpness 236–558; ambiguity 15–385, all above
the gate of 3). **An uncorrected 9.4-minute offset would put every merged CTD
sample tens of meters from where the MicroRider was** — this correction is not
optional. The hotel file is written on the MR clock: use `hotel.time_offset: 0`.

## Calibration

Whole deployment, steady state: **C_D0 = 0.1753, ε = 7.57 × 10⁻¹⁰ Pa⁻¹**
(then held fixed), 124,468 samples. Per UTC day:

| day (UTC) | C_D0 dynamic | C_D0 steady | V_g − m_g/ρ₀ | U dive | U climb | α dive | α climb | w_water MAD | n cal |
|---|---|---|---|---|---|---|---|---|---|
| 2017-02-19 | 0.1741 | 0.1741 | −51 cc | 0.395 | 0.239 | −2.83° | +2.86° | 1.55 cm/s | 6598 |
| 2017-02-20 | 0.1707 | 0.1707 | −62 cc | 0.393 | 0.246 | −2.88° | +2.82° | 1.26 cm/s | 15156 |
| 2017-02-21 | 0.1711 | 0.1711 | −67 cc | 0.392 | 0.249 | −2.82° | +2.81° | 1.29 cm/s | 14563 |
| 2017-02-22 | 0.1719 | 0.1719 | −65 cc | 0.398 | 0.241 | −2.80° | +2.89° | 1.24 cm/s | 9504 |
| 2017-02-23 | 0.1761 | 0.1761 | −64 cc | 0.385 | 0.247 | −2.95° | +2.92° | 1.29 cm/s | 15012 |
| 2017-02-24 | 0.1790 | 0.1790 | −66 cc | 0.384 | 0.246 | −2.94° | +2.90° | 1.27 cm/s | 15031 |
| 2017-02-25 | 0.1765 | 0.1765 | −67 cc | 0.386 | 0.244 | −2.92° | +2.94° | 1.29 cm/s | 14961 |
| 2017-02-26 | 0.1773 | 0.1773 | −64 cc | 0.384 | 0.244 | −2.97° | +2.98° | 1.35 cm/s | 14938 |
| 2017-02-27 | 0.1777 | 0.1777 | −64 cc | 0.387 | 0.246 | −2.89° | +2.92° | 1.33 cm/s | 14685 |
| 2017-02-28 | 0.1811 | 0.1811 | −64 cc | 0.386 | 0.236 | −2.95° | +2.86° | 1.14 cm/s | 4020 |

Speeds in m/s, medians over calibration samples. |α| over all calibration
samples: 2.50 / 2.90 / 3.28° (5 / 50 / 95%), 99.7% inside Kokoszka et al.
(2025)'s 1.5–4.5° band (plausibility, not validation). Roll: −11.9° median
(5–95%: −13.0 to −10.0°), not modeled.

### What the table says

- **doug dives much faster than it climbs** (0.39 vs 0.24 m/s): V_g − m_g/ρ₀ ≈
  −64 cc, i.e. it flew heavy for the water it was in (in-situ ρ median 1024.6).
- **Drag rises only slightly**: C_D0 0.171–0.174 (02-19 … 02-22) to 0.176–0.181
  (02-23 … 02-28), about +4%, and speeds barely move — unlike husker's +22%.
- **The "dynamic" coefficients equal the steady-state seeds exactly, every
  day.** Not a failure: the dynamic cost is finite with a real minimum at the
  seed (checked for 02-24: 0.179; starting from 0.20 returns 0.1788).
  `gliderflight` stops on scipy `fmin`'s default `ftol = 1e-4` (absolute —
  comparable to the cost itself, ~2e-4) and `xtol = 1e-2` (≈ ±0.0015 in C_D0),
  so it halts within resolution of where it starts. The dynamic model's
  contribution is its time integration, not different coefficients; the
  resolution limit is ~±0.5% in U, ~±2% in ε.

## Invariance tests (steady state, whole deployment)

| case | C_D0 | ε | U dive / climb | α dive / climb |
|---|---|---|---|---|
| baseline (m_g = 52 kg) | 0.1753 | 7.57e-10 | 0.3876 / 0.2441 | −2.90° / +2.91° |
| m_g = 60 kg | 0.1753 | 8.97e-10 | 0.3875 / 0.2439 | −2.90° / +2.91° |
| start C_D0 = 0.10 | 0.1753 | 7.57e-10 | 0.3876 / 0.2441 | −2.90° / +2.91° |
| start C_D0 = 0.25 | 0.1753 | 7.57e-10 | 0.3876 / 0.2441 | −2.90° / +2.91° |
| first half only | 0.1729 | 7.64e-10 | 0.3897 / 0.2470 | −2.86° / +2.87° |
| second half only | 0.1777 | 7.55e-10 | 0.3856 / 0.2413 | −2.94° / +2.95° |

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
("The Aquadopp Check"). doug is four years later than jane, so the transfer is
weaker still.

## Limitations

All of the husker README's apply (no independent speed reference; lift
coefficient not calibratable from a ±25° pitch spread; roll not modeled;
nominal mass). Nothing specific to doug beyond those: the deck files
(CAS_001–004, 030, 031, 078–080) get no clock solution and no speed, which is
correct — they hold no profiles.

## Using it in perturb

As in the husker README, with `file: "<path>/doug_flight_hotel.nc"`,
`time_column: "time"`, `time_offset: 0.0`.

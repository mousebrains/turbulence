# Aquadopp HR on glider jane — an independent check on the glider flight model

`../TAI102.PRF` is a **2 MHz Nortek Aquadopp HR profiler** (hardware AQD 8396,
head ASP 5444) that flew on Slocum glider **jane**, 2013-04-29 05:00Z –
05-04 06:43Z: 438,201 records at 1 Hz, 29 cells, 3 beams. It measures the water
moving past the hull, so it yields the glider's **through-water speed and angle
of attack directly** — the one thing the Merckelbach flight models used for the
Taiwan MicroRiders have no measurement of.

Done 2026-09-15. **Result: on dives the flight model's angle of attack is ~1.6°
too small, its speed ~5–6% too high, and any ε derived from it ~20–25% too low.
On climbs the two agree to 0.3%.**

The narrative write-up of all of this — "The Aquadopp Check" — is
`docs/aquadopp_flight_model_validation.html` in the microstructure-tpw repo.
This README is the operator's copy: same numbers, next to the code.

## Which glider, and the clock

The Aquadopp is not labeled with a platform. jane is identified by overlap:
its record sits inside `gliders/20130427_jane_tn` (04-28 09:45 → 05-04 05:10),
and matching the two pressure records gives

    glider_time = aqd_time + 38.0 s     (37.0 → 39.0 s over the record)
    glider_P    = 0.600 + 1.0125 * aqd_P   (dbar)

with a 0.184 dbar residual and a sharpness of 117, on 385,489 in-water samples.

## Files

| | |
|---|---|
| `read_prf.py` | Nortek reader. Structures were **verified against the file**, not assumed — see its docstring. |
| `beams.py` | Undoes the instrument's own ENU rotation → XYZ → beam. |
| `aoa.py` | Unwraps the HR ambiguity and solves for speed and angle of attack. |
| `dinkum-flight.yaml`, `make_flight_speed_jane.py`, `jane_calibration.json` | The same Merckelbach recipe used for the MicroRider gliders, applied to jane. |
| `compare.py` | Measured vs modeled. |

## Two things the file itself settles

- **Velocity scaling and orientation come from the status byte** (0x62 on every
  record): bit 1 set = **0.1 mm/s**, not mm/s; bit 0 clear = head looks **up**.
  Read as mm/s the velocities are 10× too big — and still look plausible, which
  is the trap.
- **The transformation matrix sits at `hdSystem + 8`** in the head record:
  `[[1.5774, −0.7891, −0.7891], [0, −1.3662, 1.3662], [0.3677, 0.3677, 0.3677]]`.

## The hard part: HR wrapping

Beam velocities saturate at **±0.1285 m/s** — the ambiguity velocity — while
the glider flies at ~0.3 m/s, so every beam wraps once or twice. Range
continuity cannot fix it: all 29 cells see nearly the same velocity and wrap
together.

**Unwrapping is anchored on measured pressure, never on a flight model.** The
prior is `U = |dh/dt| / sin(|pitch| + α)` with α starting at 0; it picks the
nearest branch, the data then set the vector within ±V_a, and α is iterated.
The prior only needs to be right to ~0.14 m/s and is tied to a measured
vertical velocity, so it cannot manufacture the answer — confirmed below.

## Result

Over 326,280 samples matched to the flight model (|pitch| > 15°, calibration
samples only):

| | U Aquadopp | U model | ratio | \|α\| Aquadopp | \|α\| model | difference |
|---|---|---|---|---|---|---|
| dive | 0.257 m/s | 0.274 m/s | 1.065 | 4.55° | 2.91° | −1.64° |
| climb | 0.346 m/s | 0.359 m/s | 1.037 | 2.99° | 2.90° | −0.09° |

Both fall with pitch, the measured always the larger:

| \|pitch\| | n | \|α\| Aquadopp | \|α\| model | U Aquadopp | U model |
|---|---|---|---|---|---|
| 18–22° | 7,718 | 4.34° | 3.19° | 0.304 | 0.311 |
| 22–25° | 176,858 | 4.32° | 3.04° | 0.300 | 0.314 |
| 25–28° | 134,515 | 3.99° | 2.73° | 0.311 | 0.328 |
| 28–32° | 7,022 | 3.07° | 2.57° | 0.330 | 0.343 |

The measured dive value, 4.3–4.6° at 22–25° pitch, sits on top of **Tanaka et
al. (2022)'s ADCP-measured Slocum angle of attack, −4.4° at −23.7° pitch** —
the only directly measured Slocum value in the literature — and well above what
`gliderflight`'s uncalibrated lift/drag defaults give.

## What was tested

| test | U dive | \|α\| dive | U climb | \|α\| climb |
|---|---|---|---|---|
| baseline | 0.251 | 4.55° | 0.349 | 2.99° |
| start α = +6° | 0.251 | 4.55° | 0.347 | 3.02° |
| start α = −6° | 0.251 | 4.55° | 0.373 | 3.71° |
| V_a = 0.1275 / 0.1295 | 0.251 | 4.55° | 0.346 / 0.352 | 2.90 / 3.09° |
| cells 8–20 / 2–10 | 0.251 / 0.252 | 4.20 / 5.09° | 0.350 | 3.33 / 2.43° |
| correlation > 65 | 0.246 | 4.54° | 0.349 | 2.98° |
| prior pitch ±1° | 0.251–0.252 | 4.55° | 0.347–0.353 | 2.99° |

- **The prior does not drive the answer**: shifting the prior's pitch by ±1°
  leaves α unchanged to 0.01°.
- **Dives are stable; climbs are not as stable** — starting from α = −6° moves
  the climb to a different wrap branch (0.373 m/s, 3.71°). Trust the dive.
- **Flow distortion is visible**: α falls from 5.09° (cells 2–10, nearest the
  hull) to 4.20° (cells 8–20). ~±0.5° of systematic.

## Systematics, and one thing that does NOT matter

- **Mounting alignment is what α is sensitive to**: perturbing the un-rotation
  attitude by 1° moves α by 1°, one for one.
- **The two tilt sensors disagree, but it does not propagate.**
  `aqd_pitch − glider_pitch = −0.42 − 0.032·pitch − 0.032·roll` (correlation
  0.9991, residual MAD 0.43°): a **3.2% gain** difference, which at ±25° is
  ∓0.8° and flips sign between legs. It is *proportional to pitch*, so it is a
  sensor gain, not a mounting tilt (which would be constant); the constant part,
  the actual alignment, is only ~0.2–0.4°. And a gain error cancels exactly:
  the instrument rotated into ENU with its own attitude and `beams.py` undoes it
  with the same numbers. The roll sensors disagree more
  (`d_roll = +2.56 + 0.035·pitch − 0.226·roll`), and roll enters only through
  that same self-canceling rotation.
  Tested against the obvious geometric alternative: a yaw misalignment big
  enough to explain 3.2% (14.6°) would put a −0.25·pitch term in the roll
  difference; the measured slope is +0.035, so that is excluded.
- Total α systematic ≈ ±0.4° (alignment) + ±0.5° (flow distortion), smaller
  than the 1.6° dive discrepancy.

## What this does and does not license

- It is **jane**, 2013-04-29 → 05-04 — *not* husker (MR046, 2013-05) or doug
  (2017). Same glider class, same mission pattern, comparable speed and pitch,
  but transferring the correction to them is an assumption, not a measurement.
  **Decided 2026-09-15 (Pat): do not propagate the angle of attack for now.**
  The MicroRider products keep their modeled speed, and the dive-ε low bias
  stands as a documented, unapplied correction.
- The dive/climb asymmetry in the measured α (4.55° vs 2.99°) may be real
  (Slocum trim differs between legs) or a residual wrap-branch effect on climbs.
  It is not resolved here.
- Nothing here validates the *absolute* speed scale of the MicroRider products;
  it bounds one term — the angle of attack — that the flight model guesses.

## Not attempted

Turbulence from the Aquadopp itself (structure functions along the beams).
At ε = 1e-8 W/kg the velocity difference across 0.5 m is ~3 mm/s against ~0.1
m/s of single-ping noise, so only very energetic water would register. This was
estimated, not tested.

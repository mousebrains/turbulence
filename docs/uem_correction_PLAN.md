# U_EM correction — scale (Merckelbach 2019, Tanaka 2022) and zero

**Status:** plan, **on hold** (Pat, 2026-09-15) until the AEM1-G tank-test
analysis is complete. That analysis decides between gain, zero or both (§2), so
§2, D1 and Phase 1's defaults are to be revisited with its results before
anything here is implemented. Nothing here is implemented.
**Date:** 2026-09-15. A skeptical review of the first draft (code facts, math,
design, acceptance criteria) was folded in before this was committed; the
changes it forced are marked *(review)*. §7 (intermittent EM data and an
EM-anchored flight model) was added the same day at Pat's request.
**Literature:** `papers/gliders-and-platforms/README.md` (flight models, the
published scale factors) and `papers/current-meters/README.md` (what sets an EM
meter's gain and zero). Current evidence on this sensor lives in the AEM1-G
repository; §2 summarizes it as of this date.

**Notation.** θ = \|pitch\|, α = angle of attack (magnitude), W = \|dP/dt\|,
U = through-water path speed. **k is always the factor that multiplies U_EM**:
U_axial = k·(U_EM − c). A published "scale factor 0.93" is k = 0.93. AEM1-G's
regression slope s is the inverse, s = 1/k *(review)*.

---

## 0. The problem

A MicroRider's AEM1-G flowmeter (`U_EM`) is the only *measured* through-water
speed on a glider, and ε goes as roughly U⁻⁴ (measured U^-3.9 on osu684). Two
published papers correct it with a multiplicative factor. Merckelbach et al.
(2019) used k = **0.93**. Tanaka et al. (2022) measured v_ADCP/v_EM = **0.85**
(Slocum) and **0.90** (SeaExplorer). That means U_EM reads 7.5–17.6% high, and
through U⁻⁴ it moves ε by ×1.34 (k = 0.93) to ×1.92 (k = 0.85).
`speed.method: em` today uses U_EM as recorded, so neither can be applied.

Adding a `scale` keyword is a one-line change. The plan exists because three
things make the obvious version wrong or misleading:

1. **A pressure-and-pitch factor is conditional on the angle of attack
   assumed.** Steady glide gives W = U·sin(θ + α) with U = k·U_EM/cos α. At one
   pitch that is one equation in k and α *(review: sign corrected)*. Tanaka's
   Appendix E makes this concrete: on the same Slocum descents, the direct ADCP
   ratio is 0.85 but Merckelbach's recipe gives 0.90. Their words: the
   coefficients and scale "cannot be determined uniquely without having another
   source of velocity measurement". That a pitch spread would also separate them
   is this repository's reasoning, not Tanaka's. Every OSU MicroRider record
   analyzed so far is climbs only.
2. **A gain may be the wrong error model.** AEM1-G's per-leg regression on the
   2025 OSU climbs gives s ≈ 1 (0.97–1.09) and a positive **offset** of +34 to
   +55 mm/s, with α fixed at 4.5°. Still-water tank zeros are per unit and
   positive (EM 066 ≈ +19 mm/s, EM 079 ≈ +13 mm/s, fresh water). Over the speed
   range a glider flies (CV(U) 0.12 on osu685-2025), slope and offset are nearly
   degenerate too. Both published papers fit a gain because they had
   effectively one flight speed per glider.
3. **Neither number transfers.** The zero is per unit and shifts ~2 mm/s on
   redeployment. AEM1-G leans, on n = 2, toward part of it being a
   *mounting* offset, i.e. a property of the MicroRider body as well as the
   head. EM 040 drifted ~−2 mm/s/day on the mooring. Tanaka found 0.85 "does not
   necessarily work best for all the individual casts", and ~0.91 on ascent
   against ~0.85 on descent. Whatever is applied must be scoped to an
   installation and a deployment, and must say what it assumed.

The deliverable is therefore not "a scale factor". It has three parts:
- (a) a correction the pipeline can apply as a gain, a zero, or both, with its
  assumptions carried into the ε and χ products;
- (b) estimators that reproduce the Merckelbach and Tanaka recipes *and* the
  offset regression, and report whether the data can tell them apart;
- (c) a validation that threads a published result end to end before any OSU
  number is trusted.

---

## 1. What exists today (verified 2026-09-15)

| Piece | Location | What it does |
|---|---|---|
| Speed dispatcher | `rsi/speed.py:58` `compute_speed_for_pfile` | Shared by `rsi` and `perturb`. Returns `(speed_fast, W_slow, source)`: provenance is **one string**. |
| `em` branch | `rsi/speed.py:161-193` | `abs(U_EM)` on the slow grid, coverage/gap gates, interp + Butterworth to fast, `speed_cutout` floor. Provenance `"em"` or `"em(cov=…,gap=…s)"` (`_measured_provenance`, `:316`). No correction of any kind. |
| `flight` branch | `rsi/speed.py:195-246`, `_flight_model_slow` `:449` | U = W / sin(θ + aoa_deg), constant `aoa_deg` (default 3.0), gate `min_pitch_deg`. |
| Flight-vs-EM cross-check | `rsi/speed.py:511-530` | Median(flight/U_EM) over the file. Warns only outside [0.8, 1.25], so the published 0.85–0.93 never trigger it. Runs only in `flight`, on raw U_EM. |
| **`max_gap_s` bug** *(review)* | `rsi/speed.py:176,234` read `speed.max_gap_s`; the errors at `:184,:241` tell users to "raise speed.max_gap_s" | perturb's validator **rejects** the key: `Unknown key(s) in [speed]: ['max_gap_s']` (reproduced). The advice in the error cannot be followed. |
| perturb config | `perturb/config.py:226-235` (defaults), `:1146` (template) | `speed:` keys `method`, `value`, `hotel_var`, `aoa_deg`, `min_pitch_deg`, `speed_cutout`, `tau`, `amplitude_quantile`. |
| perturb call site | `perturb/pipeline.py:2397-2412` | Passes the merged `speed` dict; stamps **only** `speed_method` / `speed_source` into per-profile attrs. |
| Profile → diss/chi provenance | `rsi/helpers.py:698-713` | The NC-route metadata whitelist carries `speed_method`, `speed_source` and the v1 provenance keys. Any new speed attr stops at the profile NetCDF unless added here *(review)*. |
| rsi `pipeline` adapter | `rsi/adapter.py:179` | Calls `compute_speed_for_pfile` and **discards** `_source`, so `rsi-tpw pipeline` products carry no speed provenance today *(review)*. |
| rsi shim | `rsi/helpers.py:1047` `speed_from_method` | Passes `{"method", "aoa_deg"}` only. **Bug, reproduced by review on synthetic input:** `labels.get(source, "pressure \|dP/dt\|")` matches `"em"`/`"flight"` exactly. A 10 s U_EM gap gives source `em(cov=0.983,gap=10.0s)`, which is stamped `'pressure \|dP/dt\|'`. The same happens to `flight` below `min_pitch_deg`. Not observed on the three real MicroRider files tried. Labels are pinned in `tests/test_glider_defaults.py:271-318`. |
| rsi config | `rsi/config.py` `epsilon`/`chi` | Flat `speed_method`, `aoa_deg`; CLI `--speed-method`, `--aoa` (`rsi/cli.py:185-239`). `aoa_deg` threads through 17 code sites. `min_pitch_deg` is unreachable from rsi; `tau` is reachable via the adapter. |
| U_EM conversion | `rsi/channels.py:719` `convert_aem1g_d` (`:700` analog) | `U = a/100 + (b/100)·d`. Real coefficients: a ≈ −26.6, b ≈ 0.0116. **U_EM is signed, forward positive**: climb median +0.43 m/s, with 1–215 negative samples per 30-min file. |
| Serials | `.p` config: channel `U_EM` key `sn` (e.g. `066`); `instrument_info` `sn` (e.g. `435`) via `p_file.instrument_sn` `:488` | Both populated in the tracked fixtures `tests/data/MR_SL435.p` and `MR_SL685_climb.p`. Formats differ across sources (`066` in the `.p`, `S/N 0066` in AEM1-G output). |
| Other U_EM consumer | `fp07cal/pairs.py:335` | Flushing gate on raw U_EM at `min_speed` 0.05 m/s. A correction of the size discussed does not move a 0.05 gate materially; document, do not change. |
| Config patcher | `rsi/config_patch.py` | Rewrites record-0 values into a new `.p` with provenance (the `fp07-cal patch` route). |
| Speed tests | `tests/test_perturb_speed.py` | `TestEMMethod`, `TestFlightMethod`, `TestFlightParameterValidation`, `TestSourceReturn`, … |
| Bench zero | `scripts/em_bench_zero.py` | Fits c(σ) = A + B/σ from still-water cans. |

An offset correction is already *possible* without code, by patching `U_EM`'s
`a` with `config_patch` (a' = a − 100·c). Nobody should have to know that, and it
leaves no trace in the ε product.

---

## 2. What the evidence says, with its uncertainty

**Published** (checked against the PDFs; see `papers/gliders-and-platforms/README.md`):

| Source | Platform, EM | k | Reference for U | Conditional on |
|---|---|---|---|---|
| Merckelbach 2019 | Slocum G1 IFM03, MicroRider + AEM1-G, Peru 2017 | 0.93 | depth rate, via pitch + modeled α, **with** ÷cos α | steady-state α "with lift angle and induced drag settings found for glider COMET" |
| Tanaka 2022 | Slocum G2 095, MicroRider + AEM1-GA, descents | 0.85 | ADCP nearest bin, along-glider | nothing about α (direct) |
| Tanaka 2022 | same data, Merckelbach's recipe | 0.90 | depth rate, via pitch + α | MEA19 a and C_D1 with C_D0 = 0.164 |
| Tanaka 2022 | "our MR data", ascents; glider and reference not stated. Cannot be ADCP for the Slocum, whose ADCP skipped ascents | ~0.91 | not stated | — |
| Tanaka 2022 | SeaExplorer SEA025, MicroRider + EM | 0.90 | ADCP | nothing about α |
| Rockland (pers. comm. in MEA19) | SeaExplorer + ADCP | "similar" | ADCP | unpublished |

**AEM1-G, as of 2026-09-15.** Sources: `docs/findings/09_freshwater_tank_zero.md`,
`docs/preregistration/2026-09-11_conductivity_series.md` (partly uncommitted),
`paper/main.tex`, `results/glider_aoa_check.txt`.

- **Glider per-leg regression.** U_MR = s·U_flight + c, with U_flight =
  W/sin(θ+α) (no cos α) and α fixed at 4.5°. Bootstrap over legs, **16–84%
  intervals**:
  - osu685-2025 (EM 066): s 0.991 [0.981, 1.002], c +49 [+46, +53] mm/s.
  - osu684-2025 pre/post wing failure (EM 051): s 1.094 / 0.967, c +34 / +55.
  - Errors-in-variables make s a lower bound.
  - At the flight-model α (1.2–2.4° at these ~42° pitches), c would drop by
    ~12–19 mm/s. **That rerun is an open TODO there.**
- **Read as a gain,** the same data need k = 0.905–0.925 at the flight-model
  α, within a few percent of Merckelbach's 0.93. *Both readings fit climbs-only
  data. That is the degeneracy, not agreement.*
- **Still-water fresh-tank zeros:**
  - EM 066: +18.4 to +20.9 mm/s. EM 079: +13.2. EM 077: +18.8 to +19.3.
  - Handling scatter is ≤ 0.23 mm/s, but redeploy shifts are ~2 mm/s.
  - Conductivity dependence is **unresolved**. Temperature: −0.07 ± 0.05
    mm/s/°C (uncommitted). No seawater soak yet.
- **Unresolved anomaly.** Even at flight-model α, the glider offset (~30–37
  mm/s, if the TODO lands as estimated) exceeds EM 066's fresh-tank zero (~19).
  A 10–20 mm/s gap is unexplained. Separately, AEM1-G's `findings/08` §3h still
  reports a slope-only model winning by AICc, which contradicts the per-leg
  offset reading, and it is not marked superseded.
- **No recommendation.** AEM1-G offers no corrected coefficients and no
  processing recommendation. Mooring wave-band gains are not believed to
  transfer to glider flight.

**Consequences for the design:**
- Support a gain *and* an offset without privileging either.
- Any estimated number travels with its α assumption and an identifiability
  report.
- Defaults reproduce today's products exactly.

---

## 3. Design decisions

**D1. One affine form, the axial projection optional.**

    U_axial = k · (U_EM − c)       k = em_scale (default 1),  c = em_offset [m/s] (default 0)
    U_path  = U_axial / cos α      only when em_axial_aoa is set (default: not applied)

- The Merckelbach/Tanaka correction is `c = 0`, `k = 0.93 | 0.85 | 0.90`. The
  AEM1-G reading is `k = 1`, `c = <unit zero>`. Both at once is allowed.
- **c is subtracted from the signed value, before `abs()`.** U_EM is signed and
  forward positive (§1), and a zero is signed in the sensor frame. Subtracting
  after `abs()` would be wrong whenever U_EM < c: on deck, at inflections, in
  the negative samples seen in every file. k multiplies the offset-removed
  signal, so a tank zero (measured as U_EM in still water) plugs in directly.
- **The ÷cos α term is Merckelbach's eq. (13)**, and his 0.93 was derived with
  it. The docs must say that typing 0.93 with the projection off is
  inconsistent by 1 − cos α: 0.3% in U, ~1.2% in ε, at 4.5° *(review)*. It stays
  off by default so existing products do not move.
- **Exact equivalence for `aem1g_d`:** a' = k·(a − 100c), b' = k·b. Analog
  (`aem1g_a`) also involves `bias`. Patched config text is decimal, so a test of
  this equivalence uses a tolerance, not bitwise equality.
- **What D1 cannot express** *(review)*: Tanaka's descent/ascent difference, or
  any factor that depends on θ. The symmetric steady-glide model cannot
  generate a dive/climb asymmetry either. Say so in the user docs, and let V4's
  residual structure decide whether a direction- or θ-dependent term is ever
  needed.

**D2. Apply through the speed config, not by patching `.p` files — for now.**
`fp07-cal` patches coefficients because Steinhart–Hart is a settled form. Here
the *form* is the open question. As config keys:
- a gain run and an offset run of the same deployment differ by one line, so
  the "speed arms" sensitivity (osu684: ε ratio 1.76 / 1.28 between flight and
  EM) becomes routine output;
- the values land in the ε/χ product attributes (D7).

The cost: ODAS MATLAB and any other `.p` reader will not see the correction.
Revisit (D6) once AEM1-G settles gain vs zero.

**D3. Scope estimated values to an installation, not a serial** *(review)*.
Values can be typed inline (`em_scale: 0.93` with a required
`em_correction_note`, e.g. "Merckelbach 2019, IFM03 — not this unit"). They can
also come from a **correction record** (YAML, written by the D5 tool). A record
carries:

| Field | What it holds |
|---|---|
| **Key** | `em_sn` + `mr_sn` (normalized: strip `S/N`, spaces and leading zeros), plus the `U_EM` `a`, `b` and `cal_date` it was derived against |
| **Scope** | `valid_from` / `valid_to`, deployment label |
| **Result** | form (gain / zero / both), values with 16–84% and 2.5–97.5% intervals |
| **Assumptions** | the α assumption (fixed value, or preset + C_D0) |
| **Levers** | pitch spread, speed lever |
| **Fit quality** | n legs, identifiability metrics |
| **Provenance** | estimator, code version |

The pipeline **refuses** a record that does not match the file on every key
field, or whose date range excludes it. A zero derived against one `a`/`b` is
meaningless after a recalibration, and AEM1-G's evidence says the MicroRider
body is part of the installation. This is the same domain-of-validity rule
`fp07cal` uses.

**D4. A steady-state AOA as a function of pitch, shared by the estimators and
optionally by `flight`.**

- **Equation.** Solve a·α·tan(θ + α) = C_D0 + C_D1·α² per sample. It needs no
  buoyancy or volume.
- **Admissible branch** *(review)*: 0 < α ≤ α* = √(C_D0/C_D1), the best-glide
  angle of attack.
  - Below a pitch of about 11–12° the only root lies past α*, and it is
    non-physical: 40.5° at 5° with MEA19 coefficients, 21.7° with Tanaka's.
  - Return NaN there and let the existing `min_pitch_deg` gating and gap rules
    handle it. The effective minimum pitch is then a property of the
    coefficients, and should be reported.
- **Presets** carry (a, C_D1) with citations:
  - MEA10: 6.1 rad⁻¹, 2.88 rad⁻², as quoted by Tanaka 2022.
  - MEA19: 7.5, 10.5.
  - Tanaka Slocum: 5.4, 5.92.
  - Tanaka SeaExplorer: 4.0, 5.0.
- **C_D0 is glider-specific and must be supplied.** Values run from 0.136 to
  0.21 across the papers, with a probe guard at the top.
- **No independent anchor exists above ~32° pitch** *(review)*. Tanaka's 4.25°
  at 23.7° reproduces a fit to those data, so it is circular. Tests of D4 are
  tests of the *solver*, not of the physics.

**D5. Estimators live in a pre-pipeline tool, `em-cal`, patterned on
`mr-clocksync`:** `init`, `extract` (slow, resumable `.p` → per-leg npz cache),
`fit`, `report`, `record`. Three estimators run on the same legs and are
reported side by side:

| Estimator | Model | Needs | Returns |
|---|---|---|---|
| **E1 Merckelbach** | W·cos α = k·U_EM·sin(θ+α), c = 0 | pressure, pitch, α (fixed or D4) | k, conditional on α |
| **E2 Tanaka** | v_ref,along = k·(U_EM − c) | a Doppler along-glider velocity as a hotel channel | k (and c if the speed lever allows), independent of α |
| **E3 offset** | U_EM = s·W/sin(θ+α) + c, **no cos α**, exactly AEM1-G's form; report k = 1/s | pressure, pitch, α, a speed lever across legs | s (and k), c; conditional on α |

A **joint identifiability report** is not optional. It gives:
- the pitch spread (10th–90th percentile of θ, per leg direction);
- the speed lever (CV(U) and U_max/U_min across legs at matched θ);
- bootstrap correlations of (k, α) and (s, c);
- the parameter shift between α presets.

The labeling rules below use thresholds from the Phase 3 Monte Carlo:
- **No pitch spread:** E1 is labeled "conditional on α = …".
- **Speed lever too short:** E3's (s, c) are labeled "degenerate", and only
  the α-pinned c at s = 1 is reported.
- **Only with both levers** is a joint (α, k) or (α, c) fit reported as a fit.

E3 ports AEM1-G's `scripts/glider_aoa_check.py::slope_zero_regression` (OLS,
bootstrap over legs, pitch band) **with attribution, not by import**: the
dependency arrow points AEM1-G → turbulence (`docs/aem1g_split_disposition.md`
§1). Its parity test needs per-leg inputs that are in neither repository's git
*(review)*. Export a small per-leg fixture from AEM1-G's `legs_*.npy` for
osu685-2025 and commit it to `tests/data/` (a few hundred rows of medians, no
raw data). The test then reproduces `results/glider_aoa_check.txt:55-57` in CI.

**D6. Later, once the form settles:** `em-cal patch` writes a' and b' (and
`bias` for analog) through `config_patch`, so ODAS MATLAB and every other `.p`
consumer see the correction. Out of scope until then.

**D7. Provenance that actually reaches the products** *(review)*.
- `compute_speed_for_pfile` gains a structured provenance dict (method, k, c,
  axial α, note or record sha256, coverage, gap, cross-check ratio). The
  existing string return is kept for compatibility.
- perturb stamps the dict's fields into per-profile attrs.
- The `rsi/helpers.py:704-713` whitelist carries them on to diss/chi.
- `rsi/adapter.py:179` stops discarding provenance.
- **Token grammar,** specified once:
  `em[k=0.930,c=+0.0000,axial=4.5](cov=0.983,gap=10.0s)`. The bracket appears
  only when the correction is not the identity, and the parenthesis only when
  there was imputation. The identity with no imputation keeps the exact token
  `"em"`. Nothing in code compares against that token today, but the pinned
  label tests do.
- The flight-vs-EM cross-check runs on *corrected* U_EM, and its median ratio
  becomes an attribute rather than only a warning. A windowed version goes into
  the `em-cal` report, because the 2023 MR685 episodes (pegged high, dead low)
  are invisible to one median per file.

---

## 4. Phases

Each phase is its own PR. Before pushing, run ruff (`ruff check` only, never
`ruff format`), mypy (`mypy src/odas_tpw/`) and pytest locally. Do a code-review
pass before opening.

*(review)* **The first PR is perturb-only and delivers the user-facing ask.**
perturb already hands its precomputed `speed_fast` and `speed_source` to the ε
and χ stages through `prepare_profiles`. A correction applied in perturb
therefore reaches ε and χ without touching the rsi kwarg plumbing. The rsi
refactor moves to Phase 2.

### Phase 1 — apply a given correction in perturb (the Merckelbach/Tanaka factor, usable)

1. **Fix the `max_gap_s` bug.**
   - Add `max_gap_s` to perturb's `speed` defaults and template.
   - Add a test that every `cfg.get(...)` key read in `rsi/speed.py` is a valid
     perturb `speed` key. That guard fails today, and would have caught the bug.
2. **Fix the rsi label bug without losing information.**
   - Map the method prefix to its label and append the provenance suffix
     unchanged, e.g. `em (U_EM)(cov=0.983,gap=10.0s)`. The cov/gap suffix
     exists for issue #180 F07 and must survive.
   - Write the failing test first, covering `em` and `flight`.
   - Update the pinned labels in `test_glider_defaults.py`.
3. **Keys:**
   - `speed.em_scale` (default 1.0);
   - `speed.em_offset` (default 0.0 m/s);
   - `speed.em_axial_aoa` (default null; `true` uses `aoa_deg`);
   - `speed.em_correction_note` (required when any of the above is not the
     identity);
   - `speed.em_record` (path; Phase 3 writes these, Phase 1 only validates the
     key's absence or presence).
4. **Validation:**
   - k must be finite and > 0.
   - |c| < 0.2 m/s is a hard limit *(review: 0.5 was a whole glider speed)*.
   - Warn when k is outside [0.7, 1.3] or |c| > 0.1 m/s.
   - Setting any em key with a `method` that does not take its speed from
     U_EM is an **error** (decided, §6). A correction typed into a flight
     config must not silently do nothing. The em-taking methods are `em`, and
     `em_flight` once §7 lands. The message names the offending keys and both
     ways out:

         ValueError: speed.em_scale, speed.em_offset are set but speed.method is
         'flight'. The U_EM correction is applied only when speed comes from
         U_EM (method 'em' or 'em_flight'); with method 'flight' it would be
         silently ignored and the products would not be corrected. Remove
         those keys, or set speed.method: em.

     Tests assert this text, so a later edit to the message is deliberate.
5. **Apply in the `em` branch,** in the D1 order: offset on the signed value,
   then scale, optional ÷cos α, `abs`, the unchanged gates, interp, smoothing,
   cutout. If the correction pushes real samples under `speed_cutout`, warn and
   give the fraction affected.
6. **Provenance per D7,** perturb side only: the dict, the attrs, the helpers
   whitelist.
7. **Tests:**
   - The identity is bit-identical *on arrays*: speed, and ε/χ variables.
   - Product attrs compare equal **excluding `history`**, which carries `now()`
     (`dissipation.py:413`, `chi_io.py:373`) *(review)*. Run on the tracked
     MicroRider fixtures.
   - k/c math, including U_EM < c and negative U_EM.
   - The ÷cos α magnitude.
   - The token grammar, including combination with imputation.
   - Each validation error names its key.
   - The in-pipeline form equals the `config_patch` a'/b' form on a copied
     fixture, within a stated tolerance.
8. **Docs:**
   - `docs/perturb/configuration.md` and the template comments.
   - A new `docs/em_correction.md` carrying:
     - the §2 published table *with its conditions*;
     - a plain statement that those are other units on other gliders;
     - the ε multiplier each factor implies;
     - the cos α caveat, and what D1 cannot express;
     - the `fp07cal` flushing gate note.

### Phase 2 — rsi reach, and the steady-state AOA model

1. **rsi plumbing.**
   - Replace the 17-site `aoa_deg` kwarg threading with one `speed_opts` dict
     from the rsi config to `compute_speed_for_pfile`. Keep `aoa_deg` as a
     legacy kwarg for one release.
   - Add the em keys and CLI flags `--em-scale` and `--em-offset`.
   - Carry provenance through `adapter.py`.
   - Characterization test: `rsi-tpw eps` and `chi` on a VMP file and a
     MicroRider fixture, before and after. Variables and attrs must be equal,
     excluding `history`.
2. **`rsi/flight_aoa.py`** per D4:
   - `steady_aoa(theta, a, cd0, cd1)`, vectorized, returning NaN off the
     admissible branch;
   - presets with citations;
   - input validation.
3. **Optional `speed.aoa_model: {preset, cd0}`** for `method: flight`. The
   default stays the constant `aoa_deg`.
4. **Tests (of the solver):**
   - The residual of the equation is ≤ 1e-10 on the admissible branch.
   - α ≤ α* everywhere.
   - NaN below each preset's θ_min, at 5° and 8°.
   - α is monotonic decreasing in θ above θ_min.
   - The θ_min reported matches where the branch ends.
   - Tanaka's 4.25° at 23.7° appears as a regression check, labeled circular.

### Phase 3 — `em-cal` estimators and identifiability (D5)

1. **`extract`:** legs from `.p`, plus an optional hotel for thruster state,
   glider pitch and E2's reference. Apply AEM1-G's steadiness gates: thruster
   off and *known* off; nose-up with dP/dt < 0 on climbs, mirrored on dives; a
   settle time after each inflection. Per-leg medians go to a resumable npz
   cache, checkpointed per file.
2. **`fit`:** E1 and E3 always; E2 when a reference channel is configured.
   Bootstrap over legs, not samples.
3. **Monte Carlo, before any real data** *(review: independent seeds)*:
   - Synthesize legs from known (α(θ), k, c), with noise and speed/pitch
     distributions matched to each target: climbs-only osu685-2025; climbs plus
     10% dives; RU33-like both legs at matched pitch.
   - **Seed set A** sets the identifiability thresholds, where recoveries
     break down.
   - **Seed set B** (disjoint) tests recovery against tolerances stated in ε
     terms: |Δk| ≤ 0.01 (≈4% in ε) and |Δc| ≤ 3 mm/s (≈3% in ε at 0.4 m/s),
     inside the identifiable region.
   - Null: k = 1 and c = 0 give back the identity.
   - Injection: each estimator recovers its own truth. The *other*
     estimator's answer under that truth is recorded, which makes the
     degeneracy visible.
4. **`report`:**
   - all estimators side by side, under each α preset;
   - the windowed cross-check ratio over time;
   - the share of legs with α > 0 after each correction;
   - the implied ε multiplier.
5. **`record`:** writes the D3 YAML, only for a result that passed the
   identifiability rules or is explicitly marked conditional.
6. **Parity:** E3 against the committed osu685-2025 leg fixture reproduces
   AEM1-G's s, c and 16–84% intervals within bootstrap noise (same seed).

### Phase 4 — validation, criteria stated before running

| # | Test | Data | Criterion |
|---|---|---|---|
| V1 | **Reproduce Merckelbach's 0.93** end to end with E1 | IFM03, Zenodo [10.5281/zenodo.2270123](https://doi.org/10.5281/zenodo.2270123). *Contents unverified: first check whether it holds MicroRider pitch, pressure and U_EMC at a usable rate.* | Coefficient set **pinned before running**: a = 7.4 rad⁻¹ (COMET's DVL optimum, MEA19 §5a) and C_D1 = 10.5 rad⁻² with C_D0 = 0.147, ÷cos α on. **Pass: \|k − 0.93\| ≤ 0.02.** Separately, report C_D0 = 0.136 and a = 7.5, but do not select among them afterwards. **Fail → stop**: the estimator or our reading of the recipe is wrong. |
| V2 | Synthetic recovery (Phase 3.3, seed set B) | synthetic | Pass: the ε-stated tolerances inside the identifiable region, with every degenerate configuration labeled and never reported as a fit. |
| V3 | Tank zero vs glider offset — **a measurement, no pass/fail** | E3 on osu685-2025 (EM 066) and sl684-2026 (EM 079) against fresh-tank zeros (+19, +13 mm/s) | Report the difference with intervals. Agreement is not expected (§2 anomaly); a difference is a finding, not something to average away. |
| V4 | Gain vs zero on a record with dives | the OSU deployment now recording ~25° dives with 35–40° climbs (path to confirm) | Rule, stated now: **leave-legs-out cross-validation**, as `fp07-cal` chooses polynomial order by held-out error. Gain, zero and both are fit on 80% of legs and scored by RMS error on the held-out 20%, over 200 splits. A form is "preferred" only if its median held-out RMSE beats the next by more than twice the split-to-split MAD. Otherwise report "not distinguishable". First the Phase 3 rules must call the record identifiable. |
| V5 | ε sensitivity arms — **a measurement** | osu684-2025, osu685-2025, sl684/sl685-2026 | perturb products for {flight, em raw, em gain (E1), em zero (E3), em both}, with ε ratio tables. For any deployment that fails V4's identifiability, this is the deliverable. |
| V6 | E2 route, if any MicroRider glider carries a Doppler along-glider velocity | none known — ask | k independent of α, against E1 on the same legs: Tanaka's 0.85-vs-0.90 test on our data. A measurement. |

### Phase 5 — documentation

- `docs/emcal/runbook.md`.
- A "facts that are easy to get wrong" block in `CLAUDE.md`:
  - k multiplies U_EM, and s = 1/k;
  - the offset comes off before `abs`;
  - published factors are conditional on α and on ÷cos α;
  - the pitch spread and the speed lever gate what can be fit;
  - records key on the installation.
- Revisit D6.

---

## 5. Assumptions carried as priors, and what would change the plan

- **That the error is affine in U_EM at all.** Aubrey & Trowbridge measured
  flow-configuration-dependent sensitivity. Bevir (1970) proves point-electrode
  meters read the flow *pattern*. A gain varying with θ or leg direction
  (Tanaka's 0.85 / ~0.91) would show as structure in E1/E3 residuals across θ.
  If V4 shows it, D1 needs a θ- or direction-dependent term, and the config keys
  are the wrong abstraction.
- **That a fresh-tank zero means anything in seawater.** Unresolved in AEM1-G.
  Until a seawater soak lands, a tank c used as `em_offset` must carry a note
  saying so. Phase 1 does not encode c(σ).
- **That per-deployment constancy holds.** Redeploy shifts of ~2 mm/s and EM
  040's drift argue it may not. The windowed report exists to catch this. A
  drifting unit needs c(t), which is out of scope.
- **That the steady-state α is right at 42°.** No paper measures Slocum AOA
  above ~32° pitch. E1 and E3 at 42° rest on extrapolation until V4.
- **That IFM03's public data suffice for V1.** Unverified. If they do not, V1
  falls back to reproducing Tanaka's Merckelbach-recipe 0.90. That needs their
  data; ask the authors (T. P. Welch is one).
- **That "climbs only" holds for every OSU record.** This is what the AEM1-G
  analyses report; it is not re-checked here. Phase 3's `extract` counts legs
  per direction and reports it.
- **What would stop the plan:**
  - V1 failing.
  - V4 finding no form distinguishable at the best lever we can fly. Then the
    deliverable is V5's sensitivity arms plus a flight-plan request for more
    pitch settings, not a correction.

## 6. Decisions (Pat, 2026-09-15)

1. **The first PR is Phase 1 only: perturb**, both bug fixes, keys, provenance,
   docs.
2. **Em keys with a method that does not use U_EM are an error**, with the
   message given in Phase 1.4.
3. **Data.**
   - **Pat holds Tanaka et al.'s data.** That enables V1's fallback
     (reproduce Tanaka's Merckelbach-recipe 0.90), and E2/V6 on the Slocum
     descents, which carry a Doppler along-glider velocity.
   - **A newer dataset exists that cannot be analyzed yet.** Do not design
     around it.
   - Still open: the path to the OSU deployment recording dives (V4), and
     whether RU33 per-leg data can be regenerated.
4. **This stays a plan until the AEM1-G tank-test analysis is done.** Its
   results decide the form of the correction. The AEM1-G per-leg rerun at
   flight-model α belongs to that work.

Still to decide when the plan is picked up: whether §7's validity mask (which
does not depend on gain vs zero) goes before Phase 1.

---

## 7. Intermittent EM data: a validity mask, and an EM-anchored flight model

*Added 2026-09-15 at Pat's request, then corrected the same day by a skeptical
review that re-ran the numbers on ~45 files. The first version of §7.1
misdescribed files 200, 281, 300, 500 and 600. The key reversals were
re-verified before this text was written, and are marked **(corrected)**.*

### 7.0 The problem

On some deployments U_EM is sane for a while, then wildly insane. The ask has
three parts: find the good stretches, use them to refine a flight model, and use
that model for speed where the EM is invalid.

The motivating case is ARCTERX-2023 Interior, osu685:
- **Data.** `/Volumes/SeaChest/ARCTERX/2023/Interior/MR685/`: 640 `.p`
  files, 15 GB, running **2023-06-06 (file 100) to 2023-08-08 (file 640)**.
- **EM.** `U_EM` is type `aem1g_d`, sn `046`, cal 2021-09-10. Glider hotel
  `.mat` and SFMC data are under `glider/`.
- **Directory README.** "The flooding caused problems with the EM sensor
  starting in A685_0622.p". Header buffer-status words were zeroed from
  `A685_0626`.
- **Not yet mapped.** Pat has not yet mapped the sane and insane periods.

### 7.1 Reconnaissance (a sample, not a characterization)

**How the numbers were made.** 14 files were sampled first; the review then
loaded ~45. Stats are on the slow channel. x = W/sin θ with W from a linear fit
to P over 60 s windows of steady climb. r = x/U_EM, which is about 1.0 for a
correct U_EM at these attitudes. Counts are recovered exactly from
U = a/100 + (b/100)·count. Rows marked *(review)* are the reviewer's numbers
and were not re-run here.

| File (date) | Pitch | U_EM p5 / p50 / p95 [m/s] | What it is |
|---|---|---|---|
| 60, 100 (06-06), 130 | ~−40 | 0.40 / 0.59 / 0.66 (100) | **clean.** Per-minute fit U_EM = 0.97·x + 0.028, corr 0.988 (100). r 0.974–0.983 across the three *(review)*. Longest constant count run 0.6 s. |
| 160–210, e.g. 200 (06-17) | −40.1 | 0.31 / 0.42 / 0.46 (200) | **(corrected) compressed response, not an offset.** Counts are clean and the magnitude is plausible, but U_EM barely follows speed: U_EM = 0.30·x + 0.25, corr 0.73. r runs from 1.51 (P > 700) to 1.16 (P < 300). Files 160, 195 and 198–210 look the same *(review)*, and file 190 reads 1.53 m/s *(review)*. The "clean 0025–0209" of the August coarse scan is wrong from 160 at the latest. |
| 217, 337 | ~−41 | median 5.1–5.4, p95 7.395 | pegged; 7.395 m/s is count 65535, full scale |
| 233, 273 | ~−40 | median 1.9–3.1 | high and noisy |
| 281, 300 | −39.7 | median 0.71–0.74 | **(corrected) decorrelated, not "19% high".** Corr 0.83–0.84, and the per-file affine fits disagree (1.15·x + 0.03 against 0.95·x + 0.17) *(review)*. |
| 401, 425 | ~−39 | 0.003 constant; −0.037 constant | stuck: 96.8% of 401 at count 2295 |
| 500 | −38.9 | −0.265 / −0.210 / 0.785 | **(corrected) contiguous rail blocks** at counts 0, 32, 287 and 468, each lasting 271–518 s *(review)* |
| 600 (08-04) | −38.4 | −0.265 / 0.547 / 0.656 | **(corrected) not interleaved.** Three contiguous blocks: count 0 for 311 s (1014 → 917 dbar), valid for 1809 s (917 → 225 dbar), count 7 for 1016 s (225 dbar → surface). The median is the valid middle. Entry to each rail is a sub-second ramp (62 → 22 → 7) *(review)*. |
| 621 | 26–37 | clean counts | flooding onset, earlier than the README says: U_EM climbs 0.62 → 1.21 while x falls 0.59 → 0.25 *(review)* |
| 622 | −29.6 | 0.053 / 0.747 / 7.395 | 37% > 1.5 m/s, noisy |
| 633 | −21.3 | 3.567 constant (count 32784) | stuck; flooding era through 640 (08-08) |

`EMC_Cur` is the 15 Hz square-wave coil drive: harmonics at 15, 45 and 75 Hz,
with per-minute p95 − p5 of 1.33–1.36 *(review)*. It is normal through every
failure before the flooding. It rises from 1.36 to 1.43 across 622 and has
collapsed by 629 *(review)*. Every `EMC_Cur` failure in this sample is already
stuck in the counts.

**What follows:**
1. **Failures come in contiguous blocks, from sub-second to multi-week.** They
   show up as rail blocks of minutes inside a file, whole stuck files, and
   degraded stretches spanning files 160–210. Rails are entered through
   sub-second ramps that are neither count 0 nor long runs. A per-file median
   misclassifies: file 600's is its valid middle. So the mask is per sample,
   with its edges dilated.
2. **"Plausible but wrong" is a response failure, not an offset.** File 200 is
   compressed (s ≈ 0.3); files 281 and 300 are decorrelated. A magnitude band
   cannot see either, and neither can a ratio mode. What sees them is a per-leg
   affine regression of U_EM on x: its slope, correlation and residual scatter.
   Each climb spans CV(x) ≈ 0.13–0.17 *(review)*, its own speed lever. On clean
   file 100 that regression gives s 0.97, corr 0.99.
3. **The ratio r is not invariant to pitch or speed** *(review)*, so a
   deployment-wide r_ref cannot be the test.
   - **Pitch.** With D4 α(θ), sin(θ+α)/sin θ is 1.028, 1.060, 1.092, 1.155 and
     1.303 at 40°, 30°, 25°, 20° and 15° (MEA19, C_D0 0.15). A perfect EM at the
     flooding era's 17–30° would be flagged.
   - **Inclinometer bias.** A 1° error moves r by 2.1% at 40° and 6.5% at 15°.
   - **Offset.** An offset makes r speed-dependent: clean file 100 goes from
     0.953 to 0.987 as x goes from 0.41 to 0.65.
   - **Mode lock.** The degraded state fills most count-clean files in
     160–210, so a "robust mode" can lock onto it.
4. **`EMC_Cur` corroborates but is narrow.** It catches only coil-drive loss.
   It must be two-sided, since it also rises across 622.
5. **The main case is a long gap at the training attitude, not pitch
   extrapolation** *(corrected)*. Invalid behavior appears from file 160
   (mid-June). Files 217–590, most of the ~7 weeks to early August, are
   **largely unsampled**, so whether valid stretches hide there is unknown. The
   flooding era (files 621–640, 08-06 to 08-08) is days long, at 17–30° pitch.
6. **The serial-framing hypothesis is weakened.** Contiguous rails entered
   through sub-second ramps look more like a signal driven to a rail inside the
   sensor's electronics than garbled RS-232 frames. That is reasoning, with
   moderate confidence. The question for Rockland stands.

### 7.1b Follow-up the same day: testing a soft-body fouling hypothesis

**Hypothesis (Pat).** Something soft — a jellyfish — covered the rounded EM
head. The pointed FP07 and shear probes would pierce such a body rather than be
covered, which would explain why only the EM is affected. Surfacing usually
shakes such bodies off the MicroRider's nose.

**Method.** A per-minute scan of files 95–230 and 581–640, plus every 10th file
in 240–580. The glider dbd gives 851 surfacings (depth < 2 m), median 3.0 h
apart and 10.7 min long, so nearly every recorded climb ends at the surface.

**Classification.**
- **Rail/stuck:** count 0, counts ≥ 16384, or a constant-count run > 5 s.
- **Clean reference:** files 95–150, U_EM = 0.977·x + 0.026, residual
  σ = 13.6 mm/s.
- **Response-bad:** residual > 54 mm/s. False-flag rate on the reference
  files: 0.2%.
- **Settling:** the first 5 minutes of each climb and P > 950 dbar are
  excluded (climb-start acceleration otherwise masquerades as episodes).

**What it predicts and what the data say:**

| Prediction if the EM were covered | Observed | Verdict |
|---|---|---|
| A covered sensor still reads water, with noise | Rail windows have within-minute std **1.3 × 10⁻¹⁵ m/s** (n = 3375), against 0.011 m/s in clean water | Rails are an electronics or data-path state, not a physical reading. Not explained by a covering. |
| Episodes end at a surfacing | Response-bad dominates **files 159–178 across 19 consecutive logged surfacings**, and 194–205 across 12. From file 159 (mid-June) no densely sampled file is clean again until the mixed files near 580 (every-10th sampling in between). | Hard to reconcile with "usually flushed at surfacing". It would need a body that survives dozens of surfacings, or re-fouling on nearly every climb. |
| No dependence on depth | Late mixed files 581–622: rails in **95%** of windows above 200 dbar and **81%** below 900 dbar, **0%** between 300 and 900 dbar (96–98% clean). The switch sits at 916 ± 8 dbar (deep recovery, n = 36) and 218 ± 14 dbar (shallow onset, n = 32), MAD-scaled. Files that begin above 900 dbar (596, 605, 614 start at ~795) have no deep rail. Clean June files show no depth dependence. | **Excludes a covering for the late period.** Nothing soft leaves at 900 dbar and returns at 220 dbar on ~40 consecutive climbs. |
| Other sensors unaffected | FP07 T1 goes wild in files 619–621 (49–58 °C readings), the same days as the flooding the directory README records. A matched shear/accelerometer comparison of response-bad vs clean windows was too sparse to read (n = 18–26 per speed bin). | The late failure is not EM-only. The early-period comparison remains untested. |

**Reading, with confidence levels:**
- **Rails and the late depth-banded failures** point at something electrical
  whose state depends on depth. That fits a progressive leak or water ingress
  ending in the recorded flooding (moderate confidence).
  - Pressure and temperature co-vary here and are **not yet separated**.
    Separating them needs isotherm heave between climbs compared against the
    transition scatter. T1's absolute calibration is also unreliable (factory
    FP07 coefficients).
- **The early response-failure period (159–213)** is the only part a covering
  could still explain. Its persistence through surfacings argues against it
  (low-to-moderate confidence), and it may instead be an early stage of the
  same ingress. Unresolved.
- **Consequence for M1/M2.** On this deployment the valid EM in the mixed
  period exists only between ~300 and ~900 dbar. An anchor trained there is
  applied, every climb, to the surface layer and the deep layer. That is a
  depth and speed extrapolation, and exactly the case §7.2's trained-domain
  flags must catch.
- **Reproducibility.** The scan took ~2 min on a warm cache; its scripts and
  per-file CSVs are session scratch and must be re-created in 7A, not relied
  on.

### 7.2 Design

**M1. A validity mask per slow sample, with dilation and human overrides.**

- **T1 — digital.** Flag count 0, counts ≥ 16384 (1.65 m/s here; AEM1-G's leg
  gate), full scale, and constant-count runs longer than N s. Healthy files
  never exceed 0.6 s; N is set from clean records.
  - **Dilate** every flagged block by ≥ 2 s plus the support of the slow→fast
    Butterworth, so rail-entry ramps are masked too.
- **T2 — hardware, corroboration only.** Windowed `EMC_Cur` amplitude outside
  a **two-sided** band around its deployment median.
- **T3 — kinematic response, per leg.**
  - **Regression.** For each steady climb or dive, regress U_EM on
    x = W / sin(θ + α(θ)), using D4's α(θ) so pitch dependence is modeled rather
    than absorbed.
  - **Test.** A leg is valid when its slope s, correlation and robust residual
    scatter lie inside bands taken from a **reference set** of legs.
  - **Reference set.** Seeded by human-confirmed clean files (e.g. 60–130 on
    osu685-2023), never by a deployment-wide mode, which can lock onto a
    degraded state. The record states what the reference was.
  - **Localization.** Minute windows within a leg locate where a response
    failure starts. Legs too short or too uniform in x to regress inherit from
    their neighbors and are marked "untested", not "valid".
  - **Consequence** *(corrected)*: 7A therefore needs D4 unless a deployment
    flies a single attitude.
- **Combining.** A sample is invalid if any test fails. Minimum episode
  durations suppress flicker. Human intervals `[start, end, valid|invalid,
  note]` apply last.
- **Output.** A mask record, with the test that fired for each interval, plus
  a timeline report meant for Pat's by-eye mapping. The report shows depth and
  x distributions of valid and invalid samples side by side, because a mask that
  preferentially removes slow or shallow water biases whatever is trained on
  what remains *(review: in files 590–620 the valid samples sit at median 490–540
  dbar, x ≈ 0.6; the invalid ones at 105–150 dbar, x ≈ 0.4)*.
- **Home.** `em-cal validity` in the D5 tool: a resumable scan with a per-file
  npz of window stats, checkpointed (640 files is well over 10 min).

**M2. An EM-anchored flight model — an affine anchor, not a gain.**

- **Form** *(corrected)*. U_fm = s_a·x + c_a with x = W / sin(θ + α(θ)),
  fitted to (corrected, per D1) U_EM on valid legs.
  - A gain-only anchor is misspecified. With file 100's s ≈ 0.96,
    c ≈ +0.035, a gain trained at x = 0.6 is −3% in U at x = 0.4 and −10% at
    x = 0.2 (×1.5 in ε). Late flooding-era files fly at x ≈ 0.16–0.32
    *(review)*.
- **The fit carries its own degeneracies, and the record says so.**
  - c_a depends on the α curve: +0.028, +0.036 and +0.053 m/s at α = 0°, 1.35°
    and 4.5° *(review)*.
  - Within a climb, pitch and speed are collinear (corr 0.958, n = 46
    *(review)*), so they are one lever, not two.
  - Gap-filling at the trained (θ, x) is insensitive to both; extrapolation is
    not.
- **Trained domain.** The record stores both the θ range and the x range.
  Predictions outside either are flagged as extrapolation, with a sensitivity
  band across D4 presets and across an affine vs gain anchor.
- **Effective negative angle.** Nothing lands in `aoa_deg`. The EM's
  calibration lives in (s_a, c_a); α(θ) stays physical and ≥ 0.
- **Time dependence.**
  - (s_a, c_a) are fitted in windows and reported as a time series with
    uncertainty, so slow drift (fouling, EM 040-style) is visible and tracked
    rather than silently absorbed. Abrupt changes are left to T3.
  - Interpolation across gaps is flagged beyond a length limit and for
    one-sided extrapolation.
  - A detected trim or regime change (osu684's wing failure: roll 17 → 25°,
    climb pitch 43 → 37° within a minute) stops a window carrying across without
    a flag.
- **Inputs.** Only the MicroRider's pressure and inclinometers. A
  buoyancy-driven dynamic model is a later option.

**M3. `speed.method: em_flight`.**

- **Speed.** Corrected U_EM where the mask is valid; U_fm where it is not.
- **Switches.** At each EM↔model switch, blend over a short window and mark ε
  and χ segments that contain a switch. The size of the step the Butterworth
  would otherwise smear across is unmeasured.
- **Per-sample flag.** A `speed_flag` channel (uint8): 1 = EM; 2 = model inside
  the trained (θ, x) domain and gap limit; 3 = model extrapolated; 0 = cutout.
- **Per-segment provenance.** The fraction of each flag, record sha256s, and
  the (s_a, c_a) used (D7).
- **QC.** Reuse `qc.epsilon_drop_from` / `chi_drop_from` to drop flag-3
  segments.
- **Errors.** Phase 1.4's error lists `em_flight` as a U_EM-taking method.

### 7.3 Validation, criteria stated before running

| # | Test | Data | Criterion |
|---|---|---|---|
| H1a | **Long real gap at the training attitude** — the main case *(review)* | osu685-2023: train on valid legs in files ≤ 150 (June), predict the valid windows of 590–620 (early August, ~7 weeks later) | Report the median signed ΔU/U, p84 and p95 of \|ΔU/U\|, and E[(U_fm/U)^−3.9] (the mean ε factor). Pass if the \|median signed\| ≤ 2% and p84 ≤ 5%; thresholds for Pat to confirm. Late r (0.977–0.998) already sits near early (0.974–0.983), which is encouraging but not the test. |
| H1b | **Synthetic gaps, hold-out by whole file** | valid files of osu685-2023, osu685-2025, sl685-2026 | Whole files held out, with a buffer at least as long as the anchor window, never blocks inside a training leg *(review: autocorrelated leakage)*. Same statistics per gap length (1 file, 6 h, 1 d, 3 d, 1 wk). |
| H2 | **Pitch and speed extrapolation** | needs valid EM at two attitudes: the dive-recording deployment (V4). osu685-2023 cannot test it. | Train at one (θ, x) region, predict the other. Until run, flag 3 carries no accuracy claim. |
| H3 | **Mask null and injection with the failures actually seen** *(review)* | clean records, and the same records with injected rail blocks entered through sub-second ramps, stuck files, a compressed response (s ≈ 0.3, corr ≈ 0.7), decorrelation (corr ≈ 0.8), and a 621-style divergence ramp | False-invalid ≤ 1% of clean steady samples. ≥ 99% of rail and stuck samples flagged, including ramps. Compressed and decorrelated legs flagged at the leg level. Thresholds for Pat to confirm. |
| H4 | **Mask bias** | every record the mask runs on | Depth and x distributions of valid vs invalid samples, with a statement of what a model trained on the valid set can and cannot predict. A measurement. |
| H5 | **Against Pat's by-eye map** of osu685-2023 | once mapped | Every disagreement listed individually. |
| H6 | **ε arms** — a measurement | osu685-2023 | {em masked → NaN, em_flight, flight only}: ε ratios over valid periods, and the spread across presets and anchor forms over invalid periods. |

### 7.4 Sequencing, and what stays open

- **7A — mask, scan, timeline report.** Independent of gain vs zero. It needs
  D4 for T3's α(θ) *(corrected)*, though a single-attitude record can run with
  a constant α meanwhile. It directly serves Pat's mapping, and could precede
  Phase 1 (§6).
- **7B — the affine anchor.** Needs D4, and D1 only to say which U_EM it
  anchors to.
- **7C — `em_flight`.** After 7A, 7B and Phase 1.
- **Open: what drives the failures.** Files 160–600 fail with the coil still
  driven, in two modes: response failures (compressed, decorrelated) and rail
  blocks with ramps. The mechanism is unknown; see §7.1 item 6. Ask Rockland.
- **Open: common cause?** Scan MR684 (`MR684/`, 135 files) from the same
  cruise. Matching episodes would point at firmware or telemetry, not at
  the unit.
- **Open: is anything valid in 217–590?** That ~7-week span is largely
  unsampled. The first 7A scan answers it, and decides whether H1a's gap is
  real.

# U_EM correction — scale (Merckelbach 2019, Tanaka 2022) and zero

**Status:** plan, not started. Nothing here is implemented.
**Date:** 2026-09-15. A skeptical review of the first draft (code facts, math,
design, acceptance criteria) was folded in before this was committed; the
changes it forced are marked *(review)*.
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
   - Setting any em key with `method` ≠ `em` is an **error** naming the key. A
     correction typed into a flight config must not silently do nothing. The
     flight cross-check reads the same keys when present with `method: em`.
     *(Decision for Pat, §6.)*
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

## 6. Decisions needed from Pat

1. **Scope of the first PR.** Recommended: Phase 1 only (perturb, both bug
   fixes, keys, provenance, docs). It delivers "apply Merckelbach's or Tanaka's
   factor" with honest provenance and forecloses nothing.
2. **Em keys with `method: flight`.** Recommended: error, so a correction is
   never silently ignored.
3. **Data:**
   - the path to the deployment now recording dives (V4);
   - whether any MicroRider glider carries a DVL or ADCP (E2/V6);
   - whether RU33 per-leg data can be regenerated (AEM1-G says they are not
     held).
4. **Sequencing with AEM1-G.** Should its per-leg regression rerun at the
   flight-model α land before V3? It moves c by 12–19 mm/s, and it is the
   number V3 compares against.

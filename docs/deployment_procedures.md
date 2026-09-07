# Deployment procedures that make the data analysable

**Forward-looking.** Every item below exists because a specific question could
not be answered on data we already have. Nothing here fixes past cruises — the
past is the past — but each is cheap at deployment time and expensive or
impossible to reconstruct afterwards.

Each procedure states what it costs, what it buys, and the concrete failure it
would have prevented.

---

## 1. Bench test before AND after every deployment

**Do:** record a short (~2–5 min) static bench file with Rockland's test probes
installed, immediately before and immediately after the deployment. Use the
**deployment's config** so the channel layout and `diff_gain` are right — the
probe `SN` and `sens` are essentially unused for a bench run, and a common
template value across channels is actually convenient, since it cancels in a
channel-to-channel comparison.

**Name and log them so they are identifiable WITHOUT reading the data.** A
convention such as `bench_pre_YYYYMMDD` / `bench_post_YYYYMMDD`, plus a line in
the cruise log, costs nothing and matters more than it looks: see the warning
below.

**Why it works.** The test probes are diagnostic by construction:

- the **shear test probe is electrically an OPEN CIRCUIT**, so the amplifier
  sees no piezo and the recording is *pure electronics noise with zero probe
  contribution*;
- the **FP07 test probe is a fixed resistance corresponding to ~16 °C**, so a
  bench trace should sit pinned at a constant ~16 °C.

**What it buys.** Subtracting bench noise from the in-water noise floor
separates *channel* noise from *probe + vibration* noise. That is the only way
to answer "is this probe noisy, or is that channel noisy?".

**The pre/post pair matters more than either alone.** A post-deployment bench
run bounds what changed *during* the cruise. Without it, a probe that was
damaged on cast 3 is indistinguishable from one that arrived wrong.

**What it does NOT do:** it cannot check `diff_gain` (an open circuit carries no
signal — see §2) and it can never give `sens`.

> **Do not identify a bench file by its readings.** It is tempting to find bench
> runs by testing "FP07 sits at `T_0`, shear near zero". That test is CIRCULAR:
> a bench run exists to catch bad electronics, so on a faulty instrument the
> readings will NOT be nominal — and the check would reject exactly the file you
> most need to find. Identify bench runs by NAME and LOG ENTRY; treat the
> readings as the result, never the identifier.
>
> What the readings mean once you have the file:
>
> | | FP07 | shear |
> |---|---|---|
> | healthy, test probes fitted | `T_0` (the test resistance is R₀, so ln(R/R₀)=0) | ~0 with small real noise (SN 428: 3.6e-4) |
> | open circuit | the rail (SN 428: −17.087 °C) | ~0 with real noise |
> | dead / railed acquisition | the rail, **zero variance** | a large constant, **zero variance** |
>
> Variance is the discriminator between an open circuit and a dead record, not
> the value.
>
> **A metric that works when provenance does not.** Names and logs are routinely
> unreliable, so `scripts/detect_bench_runs.py` classifies from the data alone,
> without assuming the readings are correct:
>
> - **FP07 — constancy, at least one channel.** `std(T)` over the record, with a
>   30 s minimum so a short file is not "constant" by brevity. Nearness to `T_0`
>   is reported as corroboration, never as a gate, and only ONE thermistor need
>   be constant — the other may be open or dead. A constant thermistor far from
>   `T_0` is still a bench run, flagged as suspect electronics.
> - **Shear — `std` in ADC COUNTS**, which is unit-free and comparable across
>   instruments:
>
>       counts = std_physical x (2*sqrt(2) * diff_gain * sens) / (adc_fs / 2^bits)
>
>   An open circuit sits at the quantisation floor, ~1.1 counts (sub-LSB dither).
>   Exactly 0 counts is dead, not open.
>
> Measured separation on labelled files:
>
> | | shear [counts] | FP07 std [°C] |
> |---|---|---|
> | bench, test probes | **1.06 – 1.14** (10–12 codes) | 2e-5 – 6e-5 |
> | dead / railed | 0.00 (1 code) | 3.6e-15 |
> | real probes, on deck | 48 – 148 | 0.08 – 0.39 |
> | in water | 4550 – 19700 | 1.5 – 2.0 |
>
> A ~50x margin separates an open circuit from the nearest connected case. SN 428's `dat_0001.p` has n_unique = 1 on *every* channel
> (`Ax`/`Ay` at −32752, the ODAS `sp_char` comms-integrity value; `V_Bat`
> −20.47 V) — a dead acquisition that a value-based test would have filed as a
> bench run.

> **ARCTERX-2022.** SN 194's two probes differ by ~50% in noise floor
> (10⁻⁹·²⁷ vs 10⁻⁹·¹⁰). Probe or channel? **Permanently unanswerable.** We
> searched ARCTERX-2023-Interior and all of SUNRISE — 2069 `.p` files, 185 of
> them SN 194, every one short enough to be a bench run opened and tested — and
> **no SN 194 bench run exists anywhere**. The four files that looked promising
> were 3–54 s deck recordings whose FP07s sat at 25–30 °C rather than the test
> probe's ~16 °C, and whose shear std was 10⁻¹ rather than an open circuit's
> 10⁻⁴.
>
> SN 428 *does* have one (`dat_0021.p`, 101 s), and it settled the equivalent
> question there in a single line: ~8% channel noise difference against a 23% ε
> offset, proving that instrument's problem was gain, not noise. One 2-minute
> recording, four years earlier, is the whole difference between a question
> answered and a question closed.

---

## 2. A signal-injection gain measurement whenever the electronics are touched

**Do:** inject a known voltage at each shear channel's input and measure the
output. Record the measured `diff_gain` per channel to 3 decimal places, with
the date. Repeat after any electronics work.

**What it buys.** `diff_gain` belongs to the CHANNEL and has *exactly* the same
leverage as `sens`: ε ∝ (`diff_gain`·`sens`)⁻², χ ∝ `diff_gain`⁻². A 2% gain
error is 4% in ε.

**It ages well, which is the real argument.** Electronics gains are stable;
probe sensitivities are not. A gain measurement made years later is still good
evidence about the deployment (absent a hardware rebuild), whereas a probe
recalibration years later says almost nothing about it (§4).

**Relationship to the swap (§3).** Both answer the same question — channel or
probe? — and they are complements, not alternatives:

| | swap (§3) | bench gain (§2) |
|---|---|---|
| where | in the field, mid-deployment | on the bench, at servicing |
| needs | nothing but hands and discipline | a calibrated source injected at the channel input |
| signal | the ocean | a known voltage |
| yields | the RATIOS `a/b` and `p/q` — **which** is at fault | the ABSOLUTE channel gains `a`, `b` |

Bench gain is strictly the more informative: it is the only one of the two that
produces a number you could actually apply, and combined with an
already-measured in-water pair ratio it gives `p/q` with no field procedure at
all. The swap is the zero-equipment version that infers attribution from ratios.

Neither recovers the absolute level — `sens` still needs hydrodynamic
calibration. **Do both:** the swap is free, the two are independent, and their
agreement is itself a check.

> **ARCTERX-2022.** SN 194 carries `diff_gain = 0.98` on *both* shear channels —
> a 2-dp approximation never replaced with measured values. Independently
> measured amplifiers do not agree to 2 dp. Its residual pair inconsistency is
> 2.7% in `diff_gain·sens`, exactly the size by which SN 142's measured values
> fell below its 2-dp ones when it finally got them.

---

## 3. Swap the probes between channels at least once per deployment

**Do:** exchange the two shear probes between channels partway through, take a
few casts each way (interleave if convenient), and — critically — **edit the
setup file so `SN` and `sens` travel WITH the probe**, and **record the swap
time in the cast sheet**.

**What it buys.** `diff_gain` follows the channel and `sens` follows the probe.
If the two are never separated in the field they cannot be separated in
analysis. With channel errors `a`, `b` and probe errors `p`, `q`:

    ratio_before = (a·p)² / (b·q)²
    ratio_after  = (a·q)² / (b·p)²

    before × after = (a/b)⁴   →  CHANNEL error
    before ÷ after = (p/q)⁴   →  PROBE error

By inspection: **the pair ratio inverts if the PROBES are at fault, and is
unchanged if the CHANNELS are.** No absolute calibration is needed anywhere,
which is exactly why it still works when the sheets are stale.

**Cost: negligible.** The offsets this resolves reach z ≈ 5 in a *single*
profile, so a few casts each way suffice. It does not need cruise time — a tank,
or any forcing both probes see in common, will do.

**Two ways to waste the effort:** not updating the config (the data is then
mislabelled rather than swapped, and the test is void), and not recording the
swap time (once the config is correctly updated, nothing in the files says which
configuration a profile belongs to).

> **ARCTERX-2022.** SN 428's shear channels disagree by a constant ~23% in ε
> (~14% in `diff_gain·sens`), stable across two cruises and flat across 1.4
> decades of ε — unambiguously a gain error. Which of the two is at fault is
> **permanently unrecoverable**, because the pairs were never swapped.

---

## 4. Calibrate close in time, and bracket the deployment

**Do:** deploy on a current calibration, and recalibrate **soon after**
recovery. Record every calibration date, and the pressure-test date.

**Why bracketing, specifically.** Shear probes are **not continuous between
calibrations** — sensitivity steps on an impact, a re-tipping or a repair, at an
unrecorded moment. A calibration constrains the probe *at its own date* and says
nothing about the interval to the next one. So a single later recalibration does
**not** bound the deployment; only a calibration close in time does, and one on
each side brackets it.

**Also:** the calibration must not fall within ~10 days *after* a pressure test —
the probe may not have re-settled. Rockland's guidance is that the pressure test
should come after the calibration, or well before it.

> **ARCTERX-2022.** SN 194's probes ran ~12 months past their recal due date.
> M2245 was later measured at 0.1007 against 0.0682 — +48% over six years — but
> because probes are not C⁰ that bounds *nothing* about 2022: the cruise-time
> error is unbounded in magnitude **and undetermined in sign**. Separately, both
> SN 428 sheets sit 7 days after a pressure test, the suspect window, which
> makes its ε ~4% low as a common-mode offset.

---

## 5. Overlap two instruments in space and time, even briefly

**Do:** when two profilers are deployed on one cruise, arrange at least one
period where both are in the water simultaneously and co-located.

**What it buys.** An inter-instrument anchor. Without it, every cross-instrument
comparison is confounded with position and time, and the confound is usually far
larger than the instrumental effect being chased.

> **ARCTERX-2022.** The two VMPs alternated every 2–3 h and *never* profiled
> simultaneously. Day-matched medians differ by up to 0.9 decades (27 Mar: one
> unit read 10× the other) against the 0.06 decades we needed to resolve. A few
> hours of genuine overlap would have made SN 194 usable as an arbiter.

---

## 6. Config hygiene

**Do:**

- Put the **real probe serial and sensitivity** in the setup file before
  deploying. Never leave the template values (`sens = 0.0700`, `SN = M`,
  `SN = T`).
- Update the config **whenever a probe changes**, and note it in the log.
- Prefer **measured 3-dp `diff_gain`** over 2-dp nominal values.
- Treat the config embedded in the `.p` file as the authoritative record — it
  travels with the data, and a separate setup file on a laptop does not.

> **ARCTERX-2022.** The only SN 428 setup files that survive in the cruise share
> folder still carry `sens = 0.0700` and `SN = M`. The real values exist solely
> inside the `.p` files. Had those been lost, the sensitivities behind every
> epsilon would have been unrecoverable.

---

## 7. Log what the files cannot record

The cast sheet is the only place some of this can live. Record:

- probe serial **per channel**, and every change with its time;
- **swap times** (§3) and bench-run file names (§1);
- any **impact, grounding or handling incident**, with the cast number — this is
  what later distinguishes "arrived wrong" from "broke on cast 12";
- which instrument was in the water when, if several alternate.

---

## 8. Keep the originals, and don't silently discard "junk"

**Do:** archive untouched originals separately from anything merged, trimmed or
repaired, and when a file is set aside as bad, write down *why*.

> **ARCTERX-2022.** `ARC1A046.P` was set aside as junk for bad buffers. It holds
> **5 good casts to 204 dbar** whose ε distribution is indistinguishable from the
> known-good cast, and it was recovered four years later only because the
> original had been kept. Note also that the vendor's *repair* of that file
> discarded a quarter of it, including the deepest cast — so the original beat
> the repaired copy.

---

## Quick checklist

| When | Do |
|---|---|
| Before deployment | bench run (§1); config carries real SN + `sens` (§6); calibration current, and not within 10 days after a pressure test (§4) |
| During | swap probes between channels once, updating the config and logging the time (§3); overlap instruments if more than one (§5); log incidents (§7) |
| After recovery | bench run again (§1); recalibrate soon (§4) |
| On servicing | signal-injection `diff_gain` measurement, recorded with date (§2) |
| Always | keep untouched originals; say why a file was junked (§8) |

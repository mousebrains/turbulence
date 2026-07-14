# Bench test (`rsi-tpw bench`)

A **bench test** is a pre-deployment health check for a Rockland Scientific
instrument. It is a Python port of the ODAS `quick_bench.m` figures, extended
with an automatic evaluation of the **Rockland Bench Test Review Checklist**
(V3). It catches corroded connections, dead channels, and excessive electronic
noise *before* the instrument goes in the water. For the equivalent check on
real ocean profiles, use [`ql`](cli.md) / [`dl`](cli.md).

## Procedure (how to record a bench file)

1. Install **dummy probes** on the instrument.
2. Rest it horizontally on something soft (open-cell foam), pressure port/magnet
   centered and facing up.
3. Record **≥ 60 s**, minimizing vibration and shocks.
4. Run `rsi-tpw bench` on the resulting `.p` file.

```bash
rsi-tpw bench VMP/SN479_0001.p                  # figures + checklist → ./bench/
rsi-tpw bench VMP/SN479_0001.p -o bench/ --show # save AND open windows
rsi-tpw bench VMP/SN479_0001.p --show           # display only, write nothing
```

## Output

Up to four files per input (SN and stem in the name), plus `--show` windows:

| File | Contents |
|------|----------|
| `QB_<SN>_<stem>_timeseries.png` | Every channel in **raw counts** (inclinometers in physical units); fast thermistor-gradient channels are mean-subtracted with the offset in the legend. |
| `QB_<SN>_<stem>_spectra.png` | Log-log auto-spectra in **counts²·Hz⁻¹**, to compare against the noise floor in the instrument's RSI calibration report. |
| `QB_<SN>_<stem>_ctclu.png` | *(if present)* JAC-T/C, turbidity, chlorophyll time series (CT/CLTU page). |
| `QB_<SN>_<stem>_checklist.txt` | The evaluated checklist (below). |

`--format` selects the figure format: `png` (default), `pdf`, or `both` write
one file per figure; `pdf-bundle` instead writes a single multi-page
`QB_<SN>_<stem>.pdf` with all figures **plus the checklist as a final page**,
matching `quick_bench.m`'s single-PDF output.

Everything is in **raw counts** on purpose: the checklist thresholds — and the
RSI calibration report you compare against — are defined on the raw signal, not
on physical units.

## Why raw counts need `deconvolve=False`

The pre-emphasized channels `T1_dT1`, `T2_dT2`, `P_dP` are recorded as the
electrical output of an analog differentiator; that is why their spectra *rise*
with frequency. `PFile` normally reconstructs the deconvolved high-resolution
temperature/pressure (Mudge & Lueck 1994), overwriting the raw counts. The
bench tool reads with `PFile(path, deconvolve=False)` so those channels stay as
the raw pre-emphasized counts the checklist thresholds refer to. Undeconvolved
channels (`Ax`, `sh1`, …) are unaffected.

## Checklist mapping

Quantitative criteria are evaluated to **PASS**/**FAIL** with the measured
value; genuinely subjective criteria are marked **REVIEW** for a human (with a
helper number where one is cheap). Optional channels that are absent are
reported **N/A**. Thresholds are from the *Rockland Bench Test Review Checklist
V3*.

The checklist's "**typically** within ±N counts" criteria are evaluated against
the **99.9th percentile** of |x − mean|, not the single worst sample: a lone DAQ
glitch or a bench bump (accelerometers register any nearby movement) must not
fail an otherwise-clean channel. So up to 0.1% of samples may sit outside the
range without failing the check — to keep that from silently masking real
excursions, whenever the true maximum exceeds the limit it is surfaced in the
measured value (e.g. `±8 counts (max ±1600)`), and the accelerometers also carry
a "large spikes?" peak/std hint.

### Time series (raw counts)

| Check | Metric | Verdict |
|-------|--------|---------|
| `Ax`, `Ay` within ±500 counts | p99.9 \|x−mean\| | PASS/FAIL |
| `Ax`, `Ay` similar, `Ax` larger; large spikes? | std ratio, peak/std | REVIEW |
| `Incl_T`, `Incl_X`, `Incl_Y` reasonable & constant | mean, std (physical) | REVIEW |
| `T1_dT1`, `T2_dT2` within ±40 counts | p99.9 \|x−mean\| | PASS/FAIL |
| `T1_dT1`, `T2_dT2` offset < 100 counts | \|mean\| | PASS/FAIL |
| `sh1`, `sh2` mean < 10 counts | \|mean\| | PASS/FAIL |
| `sh1`, `sh2` within ±30 counts | p99.9 \|x−mean\| | PASS/FAIL |
| `P` within ±2 counts | p99.9 \|x−mean\| | PASS/FAIL |
| `P_dP` within ±10 counts | p99.9 \|x−mean\| | PASS/FAIL |
| `P_dP` seemingly random | — | REVIEW |
| `C1_dC1` within ±50 / offset < 6000 | p99.9 \|x−mean\|, \|mean\| | PASS/FAIL, else N/A |

### Spectra (counts²·Hz⁻¹)

| Check | Metric | Verdict |
|-------|--------|---------|
| `P_dP` density everywhere < 10 | max PSD | PASS/FAIL |
| `P_dP` peak < 3 (rolloff ~2 Hz) | max PSD, peak freq | PASS/FAIL |
| `Ax`, `Ay` peaks < 100 | max PSD | PASS/FAIL |
| `T1`, `T2` rising to ~10⁻¹ near 100 Hz; similar | PSD @ 100 Hz | REVIEW |
| `sh1`, `sh2` rising to ~10⁻² near 100 Hz; similar | PSD @ 100 Hz | REVIEW |

### CT/CLTU (raw counts)

`JAC_T` within ±50, `Turbidity` within ±50, `Chlorophyll` within ±400 →
PASS/FAIL; `JAC_C`'s I (≤ ±5) and V (~10⁴, ≤ ±100) sub-channels are drawn in the
CT/CLTU figure for visual review.

## Reading the spectra

Spectra are expected to be smooth curves. **Narrow-band** spikes from explainable
sources — AC power (50/60 Hz), the EM current meter (15 Hz), and their harmonics —
are fine. **Broadband** noise, especially in a single channel, warrants
investigation. Always compare against the instrument's ASTP calibration report.

## Flags

See the [`rsi-tpw bench` section of the CLI reference](cli.md#rsi-tpw-bench) for
the full flag list (`-o/--output`, `--show`, `--sn`, `--fft-sec`, `--dpi`,
`--format`).

## References

- ODAS `quick_bench.m` (Rockland Scientific ODAS MATLAB Library v4.5.1).
- *Rockland Bench Test Review Checklist* (V3), Rockland Scientific International.

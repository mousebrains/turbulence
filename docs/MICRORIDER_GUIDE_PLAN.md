# MicroRider End-to-End User Guide — Build Plan

**Status:** planning (prep for the microrider work session).
**Goal:** a standalone, reproducible walkthrough that takes *someone other than Pat*
from raw MicroRider `.p` files all the way to publication figures, using this
repo's tools. Includes commands, explanatory text, and graphics.

The guide covers the full chain:

```
inspect .p  →  build YAML  →  perturb run  →  extract sections  →  perturb-diag  →  perturb-plot
 (rsi-tpw)      (perturb init)   (batch)         (perturb sections)  (interactive)    (figures)
```

---

## 0. What makes this an *MR* guide (not a VMP guide)

These are the MR-specific facts the guide must teach; they drive every config choice
below. Each is grounded in code (file refs for our reference, not for the reader).

| Topic | VMP | MicroRider (glider) | Where in code |
|---|---|---|---|
| Vehicle/direction | `vmp` → free-fall `down`, τ≈1.5 s | `slocum_glider` → `glide`, τ≈3 s; profiles go up **and** down | `rsi/vehicle.py:11-24`; `pipeline` direction resolve |
| Through-water speed | `|dP/dt|` (pressure) is correct | **`|dP/dt|` is vertical speed → biases ε** (~U⁴). Use `flight` model, `em` (`U_EM`), hotel telemetry, or `constant` | `rsi/speed.py:44-149` |
| Salinity / CTD | JAC C/T on board → measured S | **No onboard conductivity** → N²/Γ/K_ρ use assumed S unless glider CTD via hotel | `rsi/adapter.py:78-84` |
| Inclinometers | — | **Pitch axis auto-detected** (larger-amplitude of `Incl_X`/`Incl_Y`); mount-dependent, not fixed | `perturb/qc_rules.py:71-100`, `speed.py:196-213` |
| Shear ε reliability | good | Thruster/surface vibration contaminates shear → prefer **chi Method 2** (`chi.use_epsilon: false`) | `config.py:142-146`, `pipeline.py:1641` |
| `top_trim` | accelerometer top-trim | **VMP-only** (MRs trimmed separately) | `pipeline.py:536-537` |
| Window lengths | time-based OK | Slow platform (0.3–0.5 m/s) + host-body length → size spatially (~2 m FFT) and convert per speed | `docs/perturb/dissipation_length.md:35-37,150-199` |
| Startup files | rare | Glider power-up writes a **year-0 clock**; tools keep the file as "undated" | `sensor_inventory.py:156-197` |

---

## 1. Dataset

**Recommended primary dataset: the in-repo `MR/` directory** — 10 MicroRider `.p`
files `AIOP2_SL685_0450.p … 0459.p` (ARCTERX 2025 Interior, Slocum "osu685",
~17 MB each). Self-contained, real, and small enough to commit examples against.
Note `AIOP2_SL685_0455.p` is a 9 KB **startup file** (year-0 clock) — a perfect,
already-present example of the undated-file behavior.

- Shear probes on this data: **M3038 / M3039** → matching sheets already exist in
  `microstructure_sensors/`, so the calibration-check step produces a real result.
- Tiny reproducible fixture for text/screenshots: `tests/data/MR_SL435.p` (196 KB).
- Larger real corpora if we want a "scale-up" appendix: `/Volumes/SeaChest/ARCTERX/2023/Interior/MR685/` (640 files), CASPER `MR_Doug/` sets.

**Open question (blocking for §3):** does osu685 have **glider telemetry** (speed /
pitch / CTD) we can supply as a hotel file? That decides speed method and whether we
get measured salinity. See Open Decisions.

---

## 2. Guide outline (chapters)

Each chapter = prose + the exact command(s) + one or more figures/outputs.

**Ch. 0 — Setup.** Install (`pip install 'microstructure-tpw[cal]'` — the `cal`
extra is needed for the calibration check). Link `docs/installation.md`. State the
example dataset and where to get it.

**Ch. 1 — Know your instrument.** Short conceptual section: MR vs VMP (the table in
§0, in reader-friendly form). Sets expectations: no free-fall, no onboard salinity,
vibration.

**Ch. 2 — Inspect the raw `.p` files (`rsi-tpw`).**
- `rsi-tpw info MR/*.p` — channels, `fs_fast`/`fs_slow`, start time, matrix. Point out
  the MR channel set (sh1/sh2, T1/T2 + T1_dT1/T2_dT2, P/P_dP, Ax/Ay, Incl_X/Y/T, maybe
  `U_EM`) and the **absence of JAC_C/JAC_T**. Show the startup file reading as undated.
- `rsi-tpw sensors MR/ --cal-dir microstructure_sensors --shear` — probe inventory +
  real M3038/M3039 calibration check (hold-previous model; link `docs/rsi-tpw/sensors.md`).
- `rsi-tpw bench <bench_file>.p -o bench/` — **only if a bench-test file exists** for
  this deployment. If not, cover it briefly and link `docs/rsi-tpw/bench.md`.
- *(optional peek)* `rsi-tpw nc` / `rsi-tpw prof --vehicle slocum_glider` to eyeball
  one file before committing to the batch config.

**Ch. 3 — Build the `perturb` config YAML.**
- `perturb init perturb.yaml` → walk the generated, commented template.
- The **MR edits** (this is the heart of the guide):
  - `files.p_file_root` / `p_file_pattern` → the MR data.
  - Vehicle/direction: rely on `.p` `instrument_info` = `slocum_glider`; set
    `profiles.direction: auto` (resolves to `glide`).
  - `speed.method`: **decision point** — `flight` (needs pitch; `aoa_deg` ~3),
    `em` (needs `U_EM`), hotel telemetry, or `constant`. Explain the ε bias of the
    default `pressure` on a glider.
  - `chi.enable: true`, `chi.use_epsilon: false` (Method 2), pick `spectrum_model`.
  - Salinity: `epsilon.salinity` / `chi.salinity` — assumed constant, **or**
    `"measured"` only if a glider CTD is supplied via hotel.
  - Window lengths: `fft_sec` / `diss_sec` reasoning from `dissipation_length.md`.
  - `top_trim.enable: false`.
  - `hotel:` block if telemetry exists (`hotel.py` channel spec; example
    `examples/arcterx_2025_interior/perturb.yaml:28-41`).
  - `netcdf:` metadata (title/summary/creator/platform/instrument).
  - Model file: `examples/arcterx_2025_interior/perturb.yaml`, adapted for MR.

**Ch. 4 — Run the pipeline.**
- `perturb run -c perturb.yaml -j 4 --stdout`.
- Explain the stages trim → merge → process → bin → combo, and the versioned output
  layout: `profiles_NN/`, `diss_NN/`, `chi_NN/`, (`ctd_NN/` only if CTD),
  `combo_NN/`, `diss_combo_NN/`, `chi_combo_NN/`, (`ctd_combo_NN/`), plus
  `logs/run_<stamp>.log`. Link `docs/perturb/pipeline.md`, `cli.md`, `logging.md`.
- Note re-run/caching behavior (`--force`).

**Ch. 5 — Extract sections.**
- `perturb sections -c perturb.yaml -o sections.yaml --gap 2h --xaxis signed_distance --units km`
  → auto-split casts on time gaps; then hand-edit `sections.yaml` (name / start / stop /
  xaxis method). Document the schema (methods: `time`, `latitude`, `longitude`,
  `distance_from_point`, `along_line`, `signed_distance`). Model file:
  `examples/arcterx_2025_interior/sections.yaml`.

**Ch. 6 — Inspect interactively (`perturb-diag`).**
- `perturb-diag epsilon -c perturb.yaml` — interactive overview (cast × depth) + click/
  arrow drill-down into shear spectra + Nasmyth fits + a diagnostics strip.
- `perturb-diag chi …`, `perturb-diag mixing …`.
- Narrowing: `--sections sections.yaml --select section_00`. QC toggle `--no-qc`,
  color `--clim`. Static capture for the doc: `--out fig.png`.
- (No dedicated perturb-diag doc exists → `--help` is the source of truth; the guide
  effectively becomes its first documentation.)

**Ch. 7 — Publication graphics (`perturb-plot`).**
- Presets: `perturb-plot figure --list-presets`, `--dump-preset <name>`,
  `perturb-plot figure --spec figure.yaml --output-pdf mr.pdf`.
- Direct products: `eps-chi` (ε/χ/χ-over-ε pcolor), `overview` (per-section ε+χ+context),
  `epsilon` / `chi` / `mixing` binned meshes, `profiles` (T1/T2/N²/dTdz),
  `scalar` (**CTD only — skip/appendix for MR unless hotel CTD**), `gamma-scaling`
  (advanced; needs ADCP). Link `docs/perturb/plotting.md`.

**Ch. 8 — Interpreting results & MR caveats.** ε vertical-speed bias, vibration
contamination, assumed-salinity effect on mixing, QC flags. Short, honest limitations
section.

---

## 3. Materials catalog (what we must produce/collect)

### A. Data (mostly already present)
- [x] `MR/` 10 `.p` files (primary).
- [x] `tests/data/MR_SL435.p` (tiny fixture for reproducible snippets).
- [ ] A **bench-test `.p`** for osu685 (for Ch. 2 bench) — *do we have one?*
- [ ] Glider **hotel telemetry** file (speed/pitch/CTD) — *do we have one?* (see decisions)
- [ ] Calibration PDFs for the MR probes — M3038/M3039 already in `microstructure_sensors/`. ✔

### B. Config / spec artifacts to author (candidate new example dir `examples/arcterx_2025_mr/`)
- [ ] `perturb.yaml` — the MR-tuned config (the reference config the guide builds).
- [ ] `sections.yaml` — generated then curated for this deployment.
- [ ] `figure.yaml` — a `perturb-plot figure` spec producing the guide's figure set.
- [ ] (optional) `hotel.csv`/`.nc` example if telemetry exists.

### C. Graphics to generate (embed as PNG/PDF in the guide)
| # | Command | Figure |
|---|---|---|
| G1 | `rsi-tpw bench …` | raw-count time series + counts²/Hz spectra (if bench file) |
| G2 | `perturb-diag epsilon --out …` | ε overview + shear-spectra drill-down snapshot |
| G3 | `perturb-diag chi --out …` | χ overview + temp-gradient spectra snapshot |
| G4 | `perturb-diag mixing --out …` | K_ρ / K_T / Γ overview snapshot |
| G5 | `perturb-plot eps-chi` | stacked log ε / χ / χ-over-ε pcolor |
| G6 | `perturb-plot overview --sections …` | per-section ε + χ + context row |
| G7 | `perturb-plot epsilon` / `chi` / `mixing` | binned depth-vs-cast meshes |
| G8 | `perturb-plot profiles` | binned T1/T2/N²/dTdz |
| G9 | *(optional)* `perturb-plot gamma-scaling` | Γ vs R_OT/Re_b/Ri_g (needs ADCP) |

### D. Prose to write
- One narrative pass per chapter (Ch. 0–8 above), each ending in a runnable command block.
- The MR-vs-VMP explainer (Ch. 1) and the caveats (Ch. 8) are the highest-value original text.

---

## 4. Open decisions (need Pat's input before/at the session)

1. **Audience & depth.** A colleague fluent in turbulence but new to this toolchain
   (my assumption), or a broader/greener reader? Affects how much theory vs. recipe.
2. **Speed source for the MR.** Glider hotel telemetry, `flight` model, `em` (`U_EM`),
   or `constant`? → *Recommended default:* `flight` if no telemetry, hotel if we have it.
   **This is the biggest single config decision.**
3. **Salinity / CTD.** Assume a constant S, or feed glider CTD via hotel to get real
   N²/Γ/K_ρ? Determines whether the mixing chapters use measured stratification.
4. **Guide home & format.** Markdown in-repo (proposed `docs/guides/microrider.md`),
   and/or a rendered shareable page (Artifact)? Should the worked example live as a new
   `examples/arcterx_2025_mr/` directory?
5. **Scope of a first pass.** Epsilon-only end-to-end first (fastest to a complete
   loop), then add chi/mixing/gamma — or full-fat in one go?
6. **Bench chapter.** Include only if an osu685 bench file exists; otherwise link the
   bench doc and move on.

*Recommended defaults if you just say "go":* dataset = repo `MR/`; audience =
turbulence-literate new user; speed = `flight` (or hotel if telemetry supplied);
salinity = assumed constant unless CTD provided; guide = `docs/guides/microrider.md`
+ a new `examples/arcterx_2025_mr/` worked example; scope = full chain but epsilon
figures first, chi/mixing second.

---

## 5. Suggested build order for the session

1. Confirm the Open Decisions (esp. #2/#3 — speed & salinity).
2. Ch. 2 dry-run: `info` + `sensors --cal-dir` on `MR/*.p`; capture real output.
3. Author `examples/arcterx_2025_mr/perturb.yaml`; `perturb run` to completion; verify
   the `{stage}_NN` outputs and logs.
4. `perturb sections` → curate `sections.yaml`.
5. Generate G2–G8 figures (diag snapshots + plot products); pick the keepers.
6. Draft prose around the captured commands/figures; wire into `docs/guides/microrider.md`.
7. Review pass; decide whether to commit the example dir + figures.

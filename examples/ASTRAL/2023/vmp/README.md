# ASTRAL 2023 — VMP-250IR-DL3 SN 92, Arabian Sea

**183 profiles from 13 deployments (D2–D14), no errors.** 2023-06-17 15:08 →
06-24 13:55, 11.77–12.26 N / 66.58–67.80 E, casts to 263 m. Rebuilt and
reorganized 2026-09-20; the earlier `Processed/` had been deleted.

**This is not our deployment** — everything below was measured from the files,
not taken from a cruise document.

```
ASTRAL/2023/
  vmp/raw/                 13 astral_VMP_D*.p   (they used to sit in the analysis dir)
  vmp/analysis/            perturb.yaml, calibration/, sections.yaml, this README
    VMP_results/           profiles / diss / chi / ctd (+ _binned, _combo), generation _00
  gps/                     gps.mat (v7.3 HDF5 table), gps.nc, make_gps.py
  FOREIGN_RR2202_met/      NOT ASTRAL -- see below
  REORG_MANIFEST.csv       every move of the 2026-09-20 reorganization
```

## Results (1-m bins, 183 profiles)

| | median | spread (MAD) | n |
|---|---|---|---|
| ε | 2.31e-9 W/kg | 0.25 dex | 24,435 |
| χ | 7.50e-9 K²/s | 0.61 dex | 24,435 |
| N² | 1.21e-4 s⁻² | 0.28 dex | 24,435 |
| Γ | 0.080 | 0.40 dex | 20,189 |
| K_ρ | 3.81e-6 m²/s | 0.36 dex | 24,412 |
| K_T | 1.06e-6 m²/s | 0.53 dex | 20,928 |

Fall rate 0.792 m/s (0.615–0.851 over all 183 profiles).

## The probe-vs-channel experiment

**The channels never changed and the probes changed six times in eight days.**
That is the experiment ARCTERX could not run, and it separates a *probe*
sensitivity error from a *channel* gain error.

| dep | sh1 | sh2 | ε₁/ε₂ | MAD |
|---|---|---|---|---|
| D2–D4 | M1021 | M1516 | 1.214 / 1.207 / 1.165 | 0.01 |
| D5 | M1021 | **M1264** | 1.627 | 0.03 |
| D6–D7 | **M1038** | M1264 | 0.882 / 0.904 | 0.04 |
| D8 | M1038 | **M1848** | 1.203 | 0.06 |
| D9 | **M1268** | M1848 | 1.512 | 0.03 |
| D10 | **M1196** | M1848 | **0.179** | **0.21** |
| D11–D14 | **M1252** | M1848 | 1.371 / 1.360 / 1.295 / 1.417 | 0.02 |

**The ratio is stable inside a combination and jumps at every change** —
8.4× at D9→D10 and 7.7× back at D10→D11. Fitting
`ln(ε₁/ε₂) = 2(ln f_i − ln f_j)` over all 13 deployments, with f = true /
configured sensitivity, closes to **2.1 % RMS in ε**. Per-probe sensitivity
explains essentially all of the pair disagreement; the channels are innocent.

| | f (normalised within its channel group) |
|---|---|
| sh1 | M1021 1.485 · M1268 1.233 · M1252 1.170 · M1038 1.100 · **M1196 0.424** |
| sh2 | M1516 1.165 · M1264 0.998 · M1848 0.860 |

**Two things this cannot tell you.** No probe was ever on both channels, so a
global shift of all f is unconstrained: the sh1 and sh2 groups are **not
comparable to each other**, and the *absolute* level of each group is
arbitrary. Only ratios **within** a group are determined. (This reproduces the
2026-09-07 fit exactly where it is determined — M1021/M1038 = 1.350 here
against 1.354 then, M1516/M1848 = 1.355 against 1.344 — and differs only by
that normalisation.)

### ⚠️ D10 / M1196 is damaged — do not use its sh1 ε

M1196 sits **2.6–3.5× below every other sh1 probe** (0.424 against 1.100–1.485),
and that comparison is *within* its group, so it does not depend on the
normalisation. Its D10 pair ratio is also the only one with disorderly scatter
(MAD 0.21 dex against 0.01–0.06 everywhere else). It lasted a single
deployment, as did M1268. **Treat D10's ε₁ as invalid**; `epsilonMean` for D10
is a geometric mean of one good probe and one bad one and is ~2.4× low.
`exclude_shear_probes` is per-INSTRUMENT, not per-file, so removing it needs a
separate run or a post-hoc mask on D10.

**No calibration sheet exists for any of the 8 probes** — not in the
collection, not under ASTRAL, nowhere on SeaChest by filename. So the absolute
ε scale rests on configured sensitivities that the data show to be wrong by
10–49 % (and 2.4× for M1196).

## FP07 tau — both beads converge

`fit_tau_method12.py` (Method-1/2 consistency; k_B,fit/k_B,ε crosses 1 at the
correct tau). 2,392 windows, U 0.77 m/s, Lueck nominal 11.4 ms; model control
0.0203 dex.

| bead | crossing | tau |
|---|---|---|
| T1 | **s = 1.70** | 19.3 ms |
| T2 | **s = 1.50** | 17.1 ms |

The two agree to 13 % — the tightest pair measured so far — and land in the
middle of the fleet (s = 1.3–2.9 over 10 beads on 4 instruments; see
`CASPER/2017_West/gliders/doug/README.md`). Note the products here were
computed at s = 1, so **χ is biased low by roughly 1.4×** (χ ∝ s^0.76). That
is a correction worth applying before quoting χ or Γ absolutely.

## D14 is NOT anomalous — the earlier flag was a statistic artifact

A previous note recorded D14's median in-water dP/dt as **negative** (−0.32
dbar/s) and unexplained. It is explained, and it is nothing:

- The in-water record contains **both** the fall and the winch-up recovery, so
  a median over all in-water samples just reflects which occupied more time.
  D14 is 47 % descending / 52 % ascending → −0.317. **D13 is the same**
  (47/53 → −0.310), and was never flagged.
- **Within detected profiles**, D14's fall rate is **0.783 m/s**, dead centre
  of the 0.770–0.824 range across all 13 deployments.

The right diagnostic is the fall rate inside the detected profiles, not over
the whole record.

## Other facts worth keeping

- **`FOREIGN_RR2202_met/` is a different cruise.** 13 `.MET` files headed
  `CRUISE:RR2202`, 2022-03-20 → 04-01 — R/V *Revelle*, ARCTERX-2022 Interior.
  All 13 are **md5-identical** to `ARCTERX/2022/Interior/cruise/data/met/data`,
  so this is a misfiled duplicate and is safe to delete. `hotel.enable` is
  false accordingly.
- **`fp07.order: 2`, and measured** — per-cast JAC_T span is 9–15 K with 100 %
  of casts ≥ 8 K (ARCTERX: 3.4–7.5 K). The per-cast *span* decides the order,
  not the cruise. Note the perturb fit is pooled per `.p` FILE, so the
  per-cast span is a proxy for the fitted range.
- A DL3 is **CF2-based**; do not infer hardware from matrix size or `fs_fast`,
  which are `[matrix]` config choices (`fs_fast = f_clock / n_cols`).
  Header v6.1, **little-endian** (ARCTERX units are big-endian).
- `gps.mat` is a MATLAB **v7.3 (HDF5) table** read with netCDF4 (`h5py` is not
  installed). The top-level `gps` is an HDF5 object reference; the columns live
  in `#refs#` under one-letter names (t=c, lat=f, lon=g), and `t` is epoch
  **milliseconds**. lat/lon cannot be told apart by range here (lon 66–72 is
  inside ±90), so `make_gps.py` takes table order and validates against the
  working area. The pre-existing `mat2nc.py` is unfinished (prints, writes
  nothing) and was left alone.
- `trim: true` — 12 of 13 files end in a partial record.

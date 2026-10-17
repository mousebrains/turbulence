# MicroRider SN 247 on Slocum G2 unit095 — Oyashio, July–August 2017

The Slocum of Tanaka et al. (2022, *JTECH*, doi:10.1175/JTECH-D-21-0074.1). It
dove to 1000 m across the Oyashio east of Japan, 40.4–42.3 N, 143.1–145.3 E.
The glider record runs 2017-07-24 → 08-30; the MicroRider record runs
07-24 23:34 → 08-14 (158 files, 16.9 GB). The MR (SN 247, top-mounted)
carried two shear probes, two FP07s and AEM1-G EM sensor 010. A 600 kHz RDI
DVL looked down and recorded descents only, 07-24 → 07-31.
**Reprocessed 2026-10-17 on main 98b4206** (`microstructure-tpw 0.4.0 (git
v0.4.0-29-g98b42062069c)`, stamped in every product); first processed
2026-10-13.

## Verdict

| | status |
|---|---|
| ε | both probes; **near the floor** (about 1e-10 W/kg), and about half the windows fail the FM shape test |
| **χ** | **T1 only, and only in DAT_023–025** (6 profiles, to 07-25 ~04 UTC): `Processed_chi_T1/`, from `perturb_chi_T1.yaml`. Off everywhere else: T2 never worked, T1 broke from DAT_026 (below) |
| speed | dynamic flight model with the paper's lift and C_D1; **matches the paper's DVL to a median ratio of 0.997** |
| clock | 157/158 files, −0.81 s/day, 0.90 s RMS (one anomaly, below) |
| EM | not used. model/U_EM = 0.839 on dives, 0.887 on climbs |

## Products: `Processed/`, generation _00 (ε, N², CTD)

345 profiles (174 dives, 171 climbs) from all 158 files, in 1-m bins.

| | dives | climbs |
|---|---|---|
| speed (model) | 0.280 m/s | 0.365 m/s |
| ε, mean of the two probes | 1.1e-10 W/kg (MAD 0.45 dex) | 1.3e-10 (MAD 0.67 dex) |
| sh1 / sh2, paired | 1.00 (MAD 0.23 dex) | 0.81 (MAD 0.24 dex) |
| FM > 1.15, sh1 / sh2 | 49% / 46% | 51% / 36% |

Binned N² (> 0) median 1.2e-5 s⁻² (MAD 0.49 dex, 196,096 bins); N² ≤ 0 in 18.9%
of bins, because stratification is temperature-sorted (#219) and salinity
scrambles the density order in 128 of 158 files (run notice).

ε is low: the Oyashio below the pycnocline is quiet. Half the windows fail the
Lueck FM test, so much of this record sits at or near the shear-probe noise
floor. Treat medians as upper-bound-ish until a per-window noise model is
applied (CONJECTURE). Probes M1672 @ 0.0837 and M1673 @ 0.0799 have no sheet in
`microstructure_sensors/`.

## The thermistors: T2 never worked, T1 only at the start

Both FP07s carry the placeholder serial "T" and nominal coefficients.

- **T2 is pinned at 16.136 °C**, the config's T₀: an open circuit or a dummy
  plug (compare the SN 465 test-probe casts). perturb's per-file fit refuses
  it in all 158 files.
- **T1 worked in DAT_023-025** (07-24 23:38 → 07-25 ~04 UTC, 6 profiles to
  ~500 m). **CORRECTED 2026-10-17:** this README said neither FP07 ever worked
  and that the fit failed its lag gate on all files; that was the gap bug
  fixed in #214. Fitted per file against the CTD, T1 tracks it at r = 1.000
  with a median |T1 − CTD| of 4–10 mK over 1.7–17 °C. Whether its
  pre-emphasized gradient channel is clean is untested.
- **T1 breaks from DAT_026** (07-25 04:42 UTC, the first 1000-m dive) and never
  recovers. There it is already 0.44 °C off at 0–100 m, so this is not a depth
  window; whether that dive damaged the bead is a guess. After that:
  - railed at −17 °C (DAT_030);
  - wandering 17–27 °C in 1.4–5.6 °C water, following pressure rather than
    temperature (Spearman ρ with the CTD 0.24, with pressure 0.67; DAT_050);
  - on the +57.6 °C rail from about DAT_120 onward (raw SD 4 counts).
- **The per-file fit accepted T1 in three broken files**, DAT_026, 028 and 048.
  It fits each file from whichever profiles pass the lag gate (|r| ≥ 0.5), then
  applies those coefficients to every profile in the file. The result is
  0.1–2.6 °C off the CTD, and 17.6 °C off on DAT_028 prof003. Their `T1`
  carries an `in-situ` calibration tag; ignore it. Nothing uses T1 here because
  χ is off.

The paper does not depend on this: it used only the MR's pressure, pitch and
EM.

## χ from T1, DAT_023–025: `Processed_chi_T1/`

`perturb_chi_T1.yaml` is `perturb.yaml` with χ on and the file pattern cut to
`DAT_02[345].P`, writing to its own output root so these products never mix
with `Processed/`. Method 2 (spectral fit), as on the other gliders. Six
profiles (3 dives, 3 climbs), 2017-07-24 23:38 → 07-25 ~04 UTC, to ~505 m.

**T1 behaves like a working bead.** Per profile it detects χ in 87–96% of
windows, the Batchelor rolloff is resolved (median K_max/k_B 4.5–5.7), and the
spectral fit is good (median fom 0.91–0.99). Whether the bead's response
(τ) is nominal cannot be checked here: there is no second working bead.

1-m bins, median (MAD of log10, n):

| | |
|---|---|
| χ from T1 (`chi_1`) | 1.8e-10 K²/s (0.77 dex, 1,983) |
| ε (paired) | 4.9e-11 W/kg (0.53, 2,183) |
| N² (> 0) | 1.7e-5 s⁻² (0.42, 1,983) |
| Γ | 0.33 (0.78, 1,094) |
| K_T | 2.3e-6 m²/s (1.11, 1,298) |

**Use `chi_1`, not `chiMean`.** perturb has no per-thermistor exclusion, so χ
is computed for the pinned T2 too. T2 returns χ in 0–48% of windows per
profile, all poorly resolved (K_max/k_B 0.7–1.2), and 2–3× above T1 where both
exist (the run flags `chi_1/chi_2` medians of 0.31 and 0.47). That spurious χ
enters `chiMean` in 127 of 1,990 bins (6%), and Γ, K_T and K_ρ are computed
from `chiMean`, so those 6% carry it too.

ε here is near the shear-probe floor (above). A floor-limited ε reads high,
which would bias Γ low (reasoning, not separately tested).

## What changed from the 2026-10-13 set

The same 345 profiles, 1-m bins.
- **ε: bit-identical** (243,018 bins).
- **N²** (#219, temperature sort): where positive in both sets it is unchanged
  (paired ratio 1.000), but 18.9% of bins now have N² ≤ 0 (0% before). In the
  same depth bins, N² dive/climb went from 1.23 to 1.07.
- **No χ in `Processed/`.** The old set kept its first run's χ as
  `INVALID_chi_*_no_working_FP07`; it was deleted with the old set
  (2026-10-17). χ from T1's three working files is now in
  `Processed_chi_T1/` (above).
- The CTD combo is bit-identical. The first attempt to write it segfaulted
  (rc 139) after every other product was done, leaving an orphan
  `.combo.nc.<pid>.tmp`; a resumed `perturb run` wrote it in 28 s. The cause,
  found the same day on CASPER-West, is HDF5's file locking on the SeaChest SMB
  mount, which can make the close of a large netCDF4 file fail and libnetcdf
  crash; `HDF5_USE_FILE_LOCKING=FALSE` avoids it.

## Recipe (run from this directory)

1. **`dinkum-hotel build -c dinkum-flight.yaml`** → `flight_inputs.nc`
   (655,401 flight-clock samples).
   - **The CTD is a Neil Brown unpumped cell whose `sci_nbctd_timestamp` is 0
     in every record.** It is projected from `sci_m_present_time` (the
     logging time), and exact-zero rows are rejected by `valid_min`.
   - Buoyancy is `m_de_oil_vol`; `m_ballast_pumped` reads 0.
2. **`python mr_clock.py extract`, then `solve`** → `mr_clock_model.json`.
   - glider_time = MR_time + L, with L = −5.98 s at 2017-08-03 and drift
     −0.81 s/day (+2.3 → −16.8 s over the record). 157 files.
   - **Anomaly.** On 08-01..08-03 the per-file lags sit about +1.2 s above
     the line, against −0.5 s elsewhere. Those days are the glider's shallow
     150–550 m dives, one dive per file. It tracks dive geometry, not the
     clock (a pressure-sensor response lag weighted differently by short
     dives is the guess, unverified). About 0.2 m vertically; not modeled.
3. **`uv run make_flight_speed.py calibrate`, then `run`** →
   `unit095_flight_hotel.nc` (already on the MR clock).
   - gliderflight 1.2.1 dynamic model.
   - Fixed: a = 5.4, C_D1 = 5.92 (the paper's DVL values for this glider)
     and m = 70.0 kg.
   - Global steady fit: C_D0 0.192, V_GL 68.22 L, c 4.76e-10 Pa⁻¹. The paper
     has 0.18–0.19, 68.2 L and 4.5–4.6e-10.
   - Per-day dynamic C_D0: 0.188–0.198 over 22 days (median 0.192).
   - Modeled AOA: −4.8° on dives at −22.5° pitch, +3.9° on climbs.
4. **`perturb run -c perturb.yaml -j 8`** (~8.5 min; these files were made on
   main 98b4206).
5. **`perturb run -c perturb_chi_T1.yaml -j 3`** → `Processed_chi_T1/` (~1.5 min).

## The speed is checked against the paper's DVL

`../../../paper_TanakaEtAl_2022_JTECH/JTECH/TanakaEtAl_2022_Revise2/data_for_TanakaEtAl_FLM_20211216.mat`
holds the paper's processed nearest-bin DVL along-glider velocity. On 39,069
descent samples (07-25 → 07-31, 50–950 dbar, calibration mask), the model
U·cos α divided by the DVL v_p has a **median of 0.997** (IQR 0.961–1.038,
mean 1.002). By depth band:

| depth | median ratio |
|---|---|
| 50–300 dbar | 1.002 |
| 300–600 dbar | 0.985 |
| 600–950 dbar | 1.005 |

In ε terms, mean (v_ADCP/v_model)⁴ = 1.028. Climbs have no DVL reference.

## The EM sensor: the gain-or-zero question this raises

At 1 Hz over the calibration mask, model/U_EM is **0.839 on dives** (U = 0.280)
and **0.887 on climbs** (U = 0.368). The paper's factor against the DVL on
descents is 0.85.

- A **pure zero offset**, U_EM ≈ U + 0.050 m/s, reproduces both legs' ratios.
  That is the per-unit zero this repository found on AEM1-G sensors in the
  tank.
- Against the DVL directly on descents, though, a total-least-squares fit
  prefers mostly gain: U_EM = 1.126·v + 0.015 (review, Sect. 4).
- Both fits have narrow speed levers. **Unresolved.** See
  `../../../paper_TanakaEtAl_2022_JTECH/review_2026-10/REVIEW.md`.

# MicroRider SN 247 on Slocum G2 unit095 — Oyashio, July–August 2017

The Slocum of Tanaka et al. (2022, *JTECH*, doi:10.1175/JTECH-D-21-0074.1). It
dove to 1000 m across the Oyashio east of Japan, 40.4–42.3 N, 143.1–145.3 E.
The glider record runs 2017-07-24 → 08-30; the MicroRider record runs
07-24 23:34 → 08-14 (158 files, 16.9 GB). The MR (SN 247, top-mounted)
carried two shear probes, two FP07s and AEM1-G EM sensor 010. A 600 kHz RDI
DVL looked down and recorded descents only, 07-24 → 07-31.
**Processed 2026-10-13** with microstructure-tpw (main 0f65101 + branch
`feat/rsijac-ct`, whose change does not touch these files).

## Verdict

| | status |
|---|---|
| ε | both probes; **near the floor** (about 1e-10 W/kg), and about half the windows fail the FM shape test |
| **χ** | **NONE: no working FP07** (see below). χ is off in `perturb.yaml`; the first run's χ is kept as `Processed/INVALID_chi_*` |
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

Binned N² median 1.3e-5 s⁻² (MAD 0.31 dex).

ε is low: the Oyashio below the pycnocline is quiet. Half the windows fail the
Lueck FM test, so much of this record sits at or near the shear-probe noise
floor. Treat medians as upper-bound-ish until a per-window noise model is
applied (CONJECTURE). Probes M1672 @ 0.0837 and M1673 @ 0.0799 have no sheet in
`microstructure_sensors/`.

## No working thermistor

Both FP07s carry the placeholder serial "T" and nominal coefficients.

- **T2 is pinned at 16.136 °C**, the config's T₀: an open circuit or a dummy
  plug (compare the SN 465 test-probe casts).
- **T1 is broken in stages**:
  - railed at −17 °C (DAT_030);
  - wandering 17–27 °C in 1.4–5.6 °C water, following pressure rather than
    temperature (Spearman ρ with the CTD 0.24, with pressure 0.67; DAT_050);
  - on the +57.6 °C rail from about DAT_120 onward (raw SD 4 counts).
- `fp07.calibrate` failed its lag gate on all 157 files for both channels.

The paper does not depend on this: it used only the MR's pressure, pitch and
EM.

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
4. **`PYTHONPATH=<repo>/src perturb run -c perturb.yaml`.**

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

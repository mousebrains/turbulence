# rockland-technical-notes

Rockland Scientific's **Technical Notes (TN) series** — the vendor grey literature
behind the instruments this repository processes (VMP profilers and MicroRider /
RDL packages). These are the primary source for the RSI signal chain: physical-unit
conversion, the ODAS data-file format, noise floors, sensor calibration, and field
technique. They complement the peer-reviewed reading list in the sibling groups.

As with the rest of `papers/`, the PDFs are **not tracked in git** (only this
`README.md` is). The public notes are on Rockland's
[Technical Notes page](https://rocklandscientific.com/support/technical-notes/);
the note number in each table links to the source PDF where Rockland publishes it
openly. Notes marked **†** are distributed by Rockland on request (shown as
"Contact Support" on that page) and are not openly downloadable; they are held here
locally for reference only.

Three notes in the series are reprints of journal papers rather than Rockland
reports — those are filed with the papers, not here (see
[Notes that are journal papers](#notes-that-are-journal-papers) below).

## Dissipation, spectra & signal conversion

| Note | File | Why it matters |
|---|---|---|
| [TN-005](https://rocklandscientific.com/wp-content/uploads/2021/12/TN_005.pdf) — Converting Shear Probe, Thermistor & Micro-Conductivity Signals into Physical Units | `TN-005_Converting_Signals_to_Physical_Units.pdf` | The count→physical-unit conversions implemented in `rsi/channels.py`; the reference for the shear, FP07, and micro-C scalings. |
| [TN-028](https://rocklandscientific.com/wp-content/uploads/2021/12/TN-028-Calculating-the-Rate-of-Dissipation-of-Turbulent-Kinetic-Energy.pdf) — Calculating the Rate of Dissipation of Turbulent Kinetic Energy (Lueck, 2013) | `TN-028_Calculating_TKE_Dissipation_Rate.pdf` | Rockland's canonical ε recipe: shear-spectrum integration, Nasmyth comparison, wavenumber limits — the algorithm mirrored in `rsi/dissipation.py` / `scor160/`. |
| [TN-030](https://rocklandscientific.com/wp-content/uploads/2021/12/TN-030-On-the-Forms-of-the-Velocity-Shear-and-Rate-of-Strain-Spectra.pdf) — On the Forms of the Velocity, Shear, and Rate-of-Strain Spectra (Lueck, 2014) | `TN-030_Velocity_Shear_Strain_Spectra.pdf` | Definitions and inter-relations of the velocity / shear / rate-of-strain spectra — the bookkeeping behind spectral variance and the Nasmyth form. |
| TN-043 † — Band of Interest | `TN-043_Band_of_Interest.pdf` | How Rockland picks the spectral integration band for a dissipation estimate; context for the fit-range logic in the ε path. |
| [TN-061](https://rocklandscientific.com/wp-content/uploads/2022/06/TN_61_goodman_bias.pdf) — Goodman Coherent Noise Removal — Spectral Bias | `TN-061_Goodman_Coherent_Noise_Spectral_Bias.pdf` | The bias the Goodman accelerometer-based coherent-noise removal introduces — directly relevant to `scor160/goodman.py`. |

## Noise, calibration & sensor response

| Note | File | Why it matters |
|---|---|---|
| [TN-010](https://rocklandscientific.com/wp-content/uploads/2021/12/AN_010_Anti_Aliasing.pdf) — Design and Optimization of Anti-Aliasing Filters | `TN-010_Anti_Aliasing_Filters.pdf` | The analog anti-aliasing filter design ahead of the ADC; sets the usable bandwidth of the fast channels. (Also numbered AN-010.) |
| [TN-040](https://rocklandscientific.com/wp-content/uploads/2021/12/TN_040_Thermistor_Noise.pdf) — Noise in Temperature Gradient Measurements | `TN-040_Noise_in_Temperature_Gradient.pdf` | FP07 temperature-gradient noise floor — the reference for the χ noise model and where a spectrum stops being signal. |
| TN-041 † — EMC Noise (Removing EMC Interference) | `TN-041_EMC_Noise.pdf` | Electromagnetic-compatibility interference: how it shows up in the spectra and how to suppress it at the instrument. |
| [TN-042](https://rocklandscientific.com/wp-content/uploads/2022/12/TN_042_Shear_Noise.pdf) — Noise in Shear Probe Measurements | `TN-042_Noise_in_Shear_Probe.pdf` | Shear-probe noise floor — the reference for the ε lower limit and the K_max / figure-of-merit rejection logic. |
| [TN-046](https://rocklandscientific.com/wp-content/uploads/2021/12/TN-046_Inclinometers.pdf) — Rockland Inclinometers | `TN-046_Rockland_Inclinometers.pdf` | Inclinometer geometry and sign conventions (Incl_X / Incl_Y / Incl_T) — the basis for the pitch/roll notes in `CLAUDE.md` and the profile-orientation checks. |
| [TN-047](https://rocklandscientific.com/wp-content/uploads/2021/12/TN_047_Why_calibrate_the_FP07.pdf) — Why Calibrate the FP07 Thermistor? | `TN-047_Why_Calibrate_the_FP07.pdf` | Why the FP07 needs a per-probe Steinhart–Hart calibration; background for `rsi/fp07_cal.py`. |
| [TN-048](https://rocklandscientific.com/wp-content/uploads/2021/12/TN_048_Interpreting_Calibrate_All.pdf) — Interpreting Calibrate All Results | `TN-048_Interpreting_Calibrate_All.pdf` | Reading Rockland's `calibrate_all` output — the bench/calibration report the `rsi-tpw bench` checklist is compared against. |
| [TN-067](https://rocklandscientific.com/wp-content/uploads/2024/12/TN-067-Microstructure-Probe-Testing-Guide.pdf) — Microstructure Probe Testing Guide | `TN-067_Microstructure_Probe_Testing_Guide.pdf` | Bench/handling test procedure for shear and thermistor probes — the field companion to the `rsi-tpw bench` diagnostic. |

## Data format, logging & processing software

| Note | File | Why it matters |
|---|---|---|
| [TN-003](https://rocklandscientific.com/wp-content/uploads/2014/01/TN_003_Transmission_Reception_Time.pdf) — Transmission-Reception Time | `TN-003_Transmission_Reception_Time.pdf` | Timing/latency of the acquisition chain; context for time-base alignment of the recorded channels. |
| TN-039 † — A Guide to Data Processing (ODAS Data Processing Manual) | `TN-039_Guide_to_Data_Processing.pdf` | The ODAS v4.4 data-processing manual — the end-to-end reference for the vendor `odas/` MATLAB library mirrored by this package. |
| TN-044 † — Legacy CF2 Persistor MicroRider Proglet for Slocum Glider | `TN-044_Legacy_CF2_MicroRider_Proglet_Slocum.pdf` | The legacy CF2/Persistor glider proglet — provenance for older MicroRider-on-glider records. |
| TN-050 † — Integration Guidance for RDL MicroRider Instruments | `TN-050_RDL_MicroRider_Integration.pdf` | Integrating the Rockland Data Logger (RDL) MicroRider onto a host vehicle — the platform behind the modern MR records. |
| [TN-051](https://rocklandscientific.com/wp-content/uploads/2026/01/TN_051_Rockland_ODAS_v6_Data_File_Format.pdf) — Rockland ODAS v6 Data File Format | `TN-051_ODAS_v6_Data_File_Format.pdf` | **The `.p` file-format spec** parsed by `rsi/p_file.py`: 128-byte record headers, the config string, the address matrix, endianness. Cited throughout the code. |
| TN-052 † — Rockland Data Logger (RDL) Overview vs Persistor CF2 | `TN-052_RDL_Overview_vs_CF2.pdf` | What changed between the CF2 and RDL loggers — matters when reconciling old vs new MR data conventions. |
| [TN-054](https://rocklandscientific.com/wp-content/uploads/2026/06/TN-54-2026-06-23-Rockland-Data-Logger-In-Situ-Data-Processing.pdf) — In-Situ Data Processing (ISDP) on Rockland Instruments | `TN-054_In_Situ_Data_Processing.pdf` | On-instrument (real-time) dissipation processing — the on-board counterpart to this package's shore processing. |
| TN-057 † — Rockland API | `TN-057_Rockland_API.pdf` | The Rockland instrument API. (Not listed on the public Technical Notes index; held for reference.) |
| [TN-063](https://rocklandscientific.com/wp-content/uploads/2024/04/TN_63_RDL_Cyclic_Sampling.pdf) — Cyclic Sampling with RDL-Based Instruments | `TN-063_Cyclic_Sampling_RDL.pdf` | Duty-cycled (cyclic) sampling on RDL instruments — affects the time structure of long MR deployments. |

## Platform, deployment & field methods

| Note | File | Why it matters |
|---|---|---|
| [TN-022](https://rocklandscientific.com/wp-content/uploads/2021/12/TN-022-Turbulence-Measurements-from-a-Glider.pdf) — Turbulence Measurements from a Glider (Wolk, Lueck & St. Laurent, 2009) | `TN-022_Turbulence_Measurements_from_a_Glider.pdf` | Early MicroRider-on-Slocum feasibility study (glider vibration spectra, sensor placement) — precedent for the glider-platform noise choices. Presented at the 13th Workshop on Physical Processes in Natural Waters. |
| [TN-024](https://rocklandscientific.com/wp-content/uploads/2021/12/TN_024_VMP_in_Tidal_Channels.pdf) — VMP Measurements in a Tidal Channel (Wolk & Lueck, 2012) | `TN-024_VMP_Measurements_in_a_Tidal_Channel.pdf` | Profiling in strong flow — the high-dissipation / high-shear regime, companion to McMillan et al. (2016) in `epsilon-shear/`. |
| [TN-033](https://rocklandscientific.com/wp-content/uploads/2021/12/TN_33_SBE7_Probe_Prep_Cleaning.pdf) — Probe Preparation & Treatment of the SBE7 Micro-Conductivity Sensor | `TN-033_SBE7_MicroConductivity_Probe_Prep.pdf` | Handling/prep of the SBE7 micro-conductivity probe — the sensor behind the C1 / micro-C channels. |
| TN-036 † — Estimating the Steady-State Load on a Winch during VMP Recovery | `TN-036_Winch_Load_During_VMP_Recovery.pdf` | Recovery-load estimate for winch/line selection — deployment-planning reference. |
| [TN-049](https://rocklandscientific.com/wp-content/uploads/2021/12/TN_049_Tow_yo.pdf) — Tow-yoing with the VMP-250-IR | `TN-049_Tow_Yoing_VMP_250_IR.pdf` | Tow-yo technique for the VMP-250-IR (the SN 479 instrument class in this repo's `VMP/` data). |

## Hardware & mechanical

| Note | File | Why it matters |
|---|---|---|
| TN-014 † — VMP Tether Termination Splice | `TN-014_VMP_Tether_Termination_Splice.pdf` | Mechanical procedure for terminating the VMP tether. |
| [TN-023](https://rocklandscientific.com/wp-content/uploads/2021/12/TN_023_SMC_Heatshink_Replacement.pdf) — SMC Heat-shrink Tubing Replacement | `TN-023_SMC_Heatshrink_Tubing_Replacement.pdf` | Replacing the SMC connector heat-shrink — maintenance procedure. |
| [TN-034](https://rocklandscientific.com/wp-content/uploads/2021/12/TN_034_Whipping_Woodhead_Cable_Grips.pdf) — Whipping Woodhead Cable Grips | `TN-034_Whipping_Woodhead_Cable_Grips.pdf` | Cable-grip whipping procedure — maintenance/rigging. |

## Notes that are journal papers

Per the collection's convention, a technical note that is really a reprint of a
published paper is filed with the papers and cited as the paper — not duplicated
here:

| Note | Is really | Filed under |
|---|---|---|
| TN-002 — Digital Signal Processing to Enhance Oceanographic Observations | Mudge & Lueck (1994), *JTECH* **11**, 825–836. [doi:10.1175/1520-0426(1994)011<0825:DSPTEO>2.0.CO;2](https://doi.org/10.1175/1520-0426%281994%29011%3C0825:DSPTEO%3E2.0.CO;2) | [`../spectra-and-sensor-response/`](../spectra-and-sensor-response/README.md) → `Mudge_Lueck_1994_DSP_Oceanographic_Observations.pdf` |
| TN-015 — Modeling the Spatial Response of the Airfoil Shear Probe Using Different Sized Probes | Macoun & Lueck (2004), *JAOT* **21**, 284–297. [doi:10.1175/1520-0426(2004)021<0284:MTSROT>2.0.CO;2](https://doi.org/10.1175/1520-0426%282004%29021%3C0284:MTSROT%3E2.0.CO;2) | [`../spectra-and-sensor-response/`](../spectra-and-sensor-response/README.md) → `Macoun_Lueck_2004_Shear_Probe_Spatial_Response.pdf` |
| TN-016 — Small-Scale Structure of Strongly Stratified Turbulence | Rehmann & Hwang (2005), *JPO* **35**, 151–164. [doi:10.1175/JPO-2676.1](https://doi.org/10.1175/JPO-2676.1) | [`../stratified-turbulence-anisotropy/`](../stratified-turbulence-anisotropy/README.md) → `Rehmann_Hwang_2005_SmallScale_Stratified_Turbulence.pdf` |

## Restricted, not held

These are "Contact Support" notes that are **not held** in this collection. They
are legacy / not relevant to modern VMP / MicroRider sampling, so they were not
pursued:

- **TN-012** — VMP 5500/6000 Release Logic
- **TN-013** — Using the Magnetometer
- **TN-021** (a: 2011, b: 2018) — Cyclic Sampling using the LPPS board (P050)
- **TN-037** — MicroXM Configuration Files (.XMP)

---

30 technical notes held here; 3 more (TN-002, TN-015, TN-016) filed as journal
papers in the sibling groups. Assembled 2026-07-14 from Rockland's public
Technical Notes page plus notes supplied by Rockland on request.

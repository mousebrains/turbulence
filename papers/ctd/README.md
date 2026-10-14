# ctd

CTD measurements from moving platforms. The group covers the conductivity
cell's thermal lag on glider CTDs, temperature–conductivity alignment, pumped
versus free-flushed flow, the spatial resolution that response time and sample
rate allow, and pressure-to-depth conversion. Opened 2026-10-10. None of these
papers concerns microstructure; they are about the CTD in its own right.

The repository uses CTD data in four places:

- **Hotel files**, which `dinkum-hotel` and `erddap-hotel` build from a Slocum
  glider's CTD as logged by the glider.
- **The reference for in-situ FP07 calibration.** `fp07-cal` decimates the
  thermistor onto real CTD samples and slows it with a single pole
  (`pairs.kernel_tau`) that stands for the CTD thermistor's own response.
- **`processing/ct_align`**, which shifts conductivity onto temperature by
  cross-correlating their first differences (by default on the instrument's own
  JAC CT).
- **Pressure to depth**, done with TEOS-10 (`gsw.z_from_p`).

The code applies no conductivity-cell thermal-mass correction anywhere (search
of `src/`, 2026-10-10). The papers below say when such a correction matters and
what it needs: a known, steady flow past the cell, a constant sample rate, and,
on an unpumped CTD, a through-water speed.

| Paper | File | Why it matters |
|---|---|---|
| Janzen (2011), "Improving CTD Data from Gliders by Optimizing Sample Rate and Flow Past Sensors," *Ocean News & Technology* **17**(7), 22–23 (August 2011). No DOI. | `Janzen_2011_Improving_CTD_Data_Gliders_Optimizing_Sample.pdf` | Sea-Bird's case for replacing free-flushed glider CTDs with a pumped, ducted, self-recording module: a free-flushed cell's response depends on the glider's varying speed and its thermistor need not sample the water that enters the cell, whereas a fixed flow and a **constant sample rate** make T–C alignment and cell-thermal-mass corrections workable. Beside a corrected 24 Hz SBE 9plus lowered at ~50 cm s⁻¹, a prototype pumped continuously at 10 mL s⁻¹ and sampled at 0.5 Hz agrees closely once aligned and corrected. One sampled every 8 s with intermittent pumping does not, because 8 s is too coarse to resolve the T–C misalignment and close to the cell's thermal-mass lag (**~10 s** at 10 mL s⁻¹). The module's internal memory records a constant-rate series, which a CTD logged through the glider's own data system need not be, and the hotel files here are built from the glider's log. |
| Garau et al. (2011), "Thermal Lag Correction on Slocum CTD Glider Data," *JTECH* **28**, 1065–1071. [doi:10.1175/JTECH-D-10-05030.1](https://doi.org/10.1175/JTECH-D-10-05030.1) | `Garau_2011_Thermal_Lag_Correction_Slocum_CTD_Glider.pdf` | The thermal-lag correction for an **unpumped** Slocum CTD (a modified SBE 41) that Wang et al. (2020) and Sheehan et al. (2023) both apply; it names three problems of that installation: the glider's speed sets the flow through the cell, sampling is ~0.5 Hz, and the sampling interval is irregular. It generalizes Morison et al. (1994) by letting the error amplitude and time constant vary with flow speed (α = α₀ + α_s/V_f, τ = τ₀ + τ_s/√V_f) and fits the four coefficients by minimizing the area between two T–S curves, either a downcast–upcast pair or glider against ship CTD. Against a pumped 8 Hz SBE 25 in a 25.8 → 15 °C thermocline, the salinity error of **~0.3 psu** at 30 m fell to **0.04 psu**, and over 220 profile pairs the T–S area came out 3 times smaller than with Morison's constant coefficients. On a glider the flow speed V_f comes from a flight model, so the correction inherits the speed questions in `../gliders-and-platforms/`. |
| Wang et al. (2020), "Modified Thermal Lag Correction of CTD Data from Underwater Gliders," *J. Coastal Res.* Special Issue No. 99, 137–143. [doi:10.2112/SI99-020.1](https://doi.org/10.2112/SI99-020.1) | `Wang_2020_Modified_Thermal_Lag_Correction_CTD_Data.pdf` | Garau's correction applied to a **pumped** Seabird GPCTD (1 Hz) on a Petrel glider, checked against an SBE 911plus in the South China Sea. Each down and up profile is split at a depth chosen from the curvature of the temperature profile, so that the upper part of upcast n−1 is paired with the upper part of downcast n, closer in time and place on a 4–6 h, 1000 m dive. The downcast discrepancy fell by up to **0.05 psu** near 90 m, and in three example profiles a 260–400 m separation between adjacent profiles, which the unsplit method leaves, is removed; no aggregate statistic is given. Mainly evidence that a pumped glider CTD's salinity is thermal-lag corrected in practice; it also tabulates the real-time QC tests of IOOS, LAGER, ANFOG, SOCIB, UW-APL, EGO and Argo. |
| RBR Ltd. (2024), "Dynamic corrections for the RBRlegato³," document RBR#0009591, revision C, 27 February 2024, 17 pp.; the revision history credits M. Dever and J.-M. Leconte. No DOI. | `RBRLtd_2024_Dynamic_Corrections_RBRlegato3.pdf` | Lab-derived dynamic corrections, with Python and MATLAB code, for RBR's inductive glider CTD (conductivity response ~32 ms; thermistor ~1 s standard or ~0.1 s fast). The C–T lag (sensor response plus the T-to-C separation), found by maximizing T–C cross-correlation in 7 s segments (segments with maximum covariance below 0.5, or in the mixed layer, excluded), is **0.90 ± 0.06 s** for the standard thermistor and **0.06 ± 0.02 s** for the fast one from more than 50 000 segments of Slocum, Seaglider and SeaExplorer data, with no measurable flow-speed dependence over 20–45 cm s⁻¹; in a flume it cut the salinity spike of a ~12 °C step from **9 to 0.5 PSU**. Thermal inertia then takes up to three recursive filters (the thermistor, then long-term O(100 s) and short-term O(10 s) cell terms), whose coefficients are power laws in the flow speed past the sensor. This is the nearest published counterpart to `processing/ct_align`'s cross-correlation lag, and another correction that needs a through-water speed. |
| Sea-Bird Scientific (2024), "Considerations for CTD Spatial and Temporal Resolutions on Moving Platforms," Application Note 98, revised May 2024, 2 pp.; text first published in *Ocean News & Technology* **15**(6), September 2009. No DOI. | `SeaBirdScientific_2024_Considerations_CTD_Spatial_Temporal_Resolutions_Moving.pdf` | On a moving platform, resolution is set by platform speed, sensor response time and sample rate together, and a feature is resolved only if half its wavelength is, so Table 1 doubles the "measurement" resolution to a "realized" one for a 3.6 m s⁻¹ tow; for example, a 0.5 s sensor sampled at 4 Hz gives 1.8 m measured and **3.6 m** realized. On an unpumped sensor the conductivity response depends on flow while the temperature response does not, giving "uncorrectable, mismatched sensor measurements", and "some commercial gliders and AUVs do not acquire data at a constant sampling frequency, making it difficult to align data streams during post processing". The same speed × response-time arithmetic sets the density-route L_T floor in `processing/thorpe.py` (~0.1 s ⇒ ~7 cm at 0.7 m s⁻¹ for an unpumped JAC CT) and bounds what a glider CTD can resolve as an FP07 reference. |
| Sea-Bird Scientific (2024), "Conversion of Pressure to Depth," Application Note 69, revised April 2024, 1 p. No DOI. | `SeaBirdScientific_2024_Conversion_Pressure_Depth.pdf` | Sea-Bird's depth algorithms, used by SEASOFT and by instruments that output depth. For seawater it uses the UNESCO Technical Paper 44 formula, which assumes a 0 °C, 35 PSU water column and lets gravity vary with latitude and pressure. For fresh water it uses simply depth = 1.019716 × pressure (dbar). The repository converts with TEOS-10 `gsw.z_from_p`; the second bullet below says when a Sea-Bird depth can be mixed with ours. |

**What these mean here.** These are checks against the code made on 2026-10-10,
not claims of the papers.

- **A hotel file's channel names do not say whether its CTD was pumped.** Wang
  et al. (2020) give the SBE 41 as unpumped and the GPCTD as pumped. Fer (2024,
  in `../gliders-and-platforms/`) describes a pumped CTD as "CTD41CP".
  `fp07cal/cli.py` notes that TWR's masterdata names `SBE41CP` for both. The
  correction regime (Garau's flow-dependent one, or fixed coefficients for a
  pumped cell) therefore has to come from the deployment record, not from the
  file.
- **Our depths are commensurable with Sea-Bird seawater depths.** With gsw
  3.6.23 (the repository venv), `-gsw.z_from_p` reproduces AN69's seawater
  formula to within 1 mm at 10–1000 dbar and at 0°, 45° and 70° latitude.
  AN69's fresh-water factor reads 2.5–3.3% deeper, +30 m at 1000 dbar and 45°.
- **The CTD thermistor's time constant was measured here, not taken from these
  papers.** `fp07-cal`'s 0.7 s default comes from
  `docs/fp07cal/ctd_response_time.md`. None of the six documents gives a GPCTD
  thermistor response time.

## Known gaps in this group

| Wanted | Why |
|---|---|
| Lueck & Picklo (1990), "Thermal inertia of conductivity cells: Observations with a Sea-Bird cell," *JTECH* **7**, 756–768. | The thermal-inertia model and recursive filter that Garau, Wang and RBR all build on. |
| Morison, Andersen, Larson, D'Asaro & Boyd (1994), "The correction for thermal-lag effects in Sea-Bird CTD data," *JTECH* **11**, 1151–1164. | The dependence of α and τ on flow speed that Garau generalizes. |
| Johnson, Toole & Larson (2007), "Sensor corrections for Sea-Bird SBE-41CP and SBE-41 CTDs," *JTECH* **24**, 1117–1130. | Corrections for the `SBE41CP` family whose name the Slocum channels carry. |
| A Sea-Bird GPCTD manual or specification sheet | The vendor's thermistor response time and pump rate, to set beside the 0.7 s measured in `docs/fp07cal/ctd_response_time.md`. |

## Cross-references

- [`../gliders-and-platforms/`](../gliders-and-platforms/README.md) holds the
  through-water speed that Garau's and RBR's flow-dependent corrections need.
  There, Frajka-Williams et al. (2011) and Sheehan et al. (2023) take the CT
  cell's flushing speed from a flight model, Leadbitter (2022) calibrates an
  FP07 against a Seaglider CT sail, and Fer (2024) lists its CTD as a pumped
  CTD41CP.
- [`../rockland-technical-notes/`](../rockland-technical-notes/README.md):
  TN-039, the ODAS processing guide, documents `cal_FP07_in_situ`, the vendor's
  in-situ FP07 calibration against a CTD.

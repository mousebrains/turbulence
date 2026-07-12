# chi-thermal

Thermal-variance dissipation (χ): the Batchelor / Kraichnan temperature-gradient
spectrum, its fitting (least-squares vs maximum-likelihood), and
temperature-microstructure methods for the FP07 chi path.

| Paper | File | Why it matters |
|---|---|---|
| Dillon & Caldwell (1980), "The Batchelor spectrum and dissipation in the upper ocean," *JGR* **85**(C4), 1910–1916. [doi:10.1029/JC085iC04p01910](https://doi.org/10.1029/JC085iC04p01910) | `Dillon_Caldwell_1980_Batchelor_Spectrum_Upper_Ocean.pdf` | Batchelor-spectrum fitting foundations for the chi path. |
| Ruddick, Anis & Thompson (2000), "Maximum likelihood spectral fitting: The Batchelor spectrum," *JTECH* **17**, 1541–1555. [doi:10.1175/1520-0426(2000)017<1541:MLSFTB>2.0.CO;2](https://doi.org/10.1175/1520-0426%282000%29017%3C1541:MLSFTB%3E2.0.CO;2) | `Ruddick_2000_ML_Batchelor_Fitting.pdf` | The MLE Batchelor-fitting method family our chi path belongs to; the unbiased amplitude estimator that fixes the log-space least-squares bias (issue #104 U3-C1). |
| Ijichi & St. Laurent (2025), "Revisiting issues in estimating spectra of ocean temperature microstructure," *JTECH* **42**, 1137–1148. [doi:10.1175/JTECH-D-24-0087.1](https://doi.org/10.1175/JTECH-D-24-0087.1) | `Ijichi_StLaurent_2025_Temperature_Microstructure_Spectra.pdf` | Current state of the art on temperature-microstructure spectra. |
| Goto, Yasuda & Nagasawa (2016), "Turbulence estimation using fast-response thermistors attached to a free-fall vertical microstructure profiler," *JTECH* **33**, 2065–2078. [doi:10.1175/JTECH-D-15-0220.1](https://doi.org/10.1175/JTECH-D-15-0220.1) | `Goto_2016_FastThermistor_Turbulence_VMP.pdf` | Thermistor-based ε estimation — chi-side methods. |
| Peterson & Fer (2014), "Dissipation measurements using temperature microstructure from an underwater glider," *Methods Oceanogr.* **10**, 44–69. [doi:10.1016/j.mio.2014.05.002](https://doi.org/10.1016/j.mio.2014.05.002) | `Peterson_Fer_2014_Glider_Temperature_Microstructure.pdf` | Chi-based ε from a glider (MR-relevant). Source of the "least squares is biased, MLE is unbiased" result (issue #104 U3-C1) and the single-pole FP07 response the pipeline defaults to (U3-C4). Also a glider-platform paper — see `gliders-and-platforms/`. |

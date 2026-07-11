# papers/ — general reference collection

Topic-specific collections live in subdirectories (see
[anisotropy/README.md](anisotropy/README.md) for the isotropy +
mixing-efficiency library). This root holds general instrument/methods
references:

| File | Citation | Why |
|---|---|---|
| `Lueck_2024_Best_Practices_Shear_Probes.pdf` | Lueck et al. (2024), "Best practices recommendations for estimating dissipation rates from shear probes," *Front. Mar. Sci.* **11**, 1334327. [doi:10.3389/fmars.2024.1334327](https://doi.org/10.3389/fmars.2024.1334327) | The ATOMIX community standard; source of the FFT-length "sandwich" in `docs/perturb/dissipation_length.md`. |
| `Fer_2024_ATOMIX_Benchmark_Datasets.pdf` | Fer, Dengler, Holtermann, Le Boyer & Lueck (2024), "ATOMIX benchmark datasets for dissipation rate measurements using shear probes," *Sci. Data* **11**, 518. [doi:10.1038/s41597-024-03323-y](https://doi.org/10.1038/s41597-024-03323-y) | Benchmark datasets validating the epsilon pipeline (see `AtomixData/`). |
| `Whalen_2021_Best_Practices_Comparing_Ocean_Turbulence.pdf` | Whalen (2021), "Best practices for comparing ocean turbulence measurements across spatiotemporal scales," *JTECH* **38**, 837–841. [doi:10.1175/JTECH-D-20-0175.1](https://doi.org/10.1175/JTECH-D-20-0175.1) | Lognormal averaging-scale pitfalls when comparing ε products. |
| `LeBoyer_2021_Epsilometer.pdf` | Le Boyer et al. (2021), "Modular, flexible, low-cost microstructure measurements: The Epsilometer," *JTECH* **38**, 657–668. [doi:10.1175/JTECH-D-20-0116.1](https://doi.org/10.1175/JTECH-D-20-0116.1) | Alternative microstructure instrument; processing comparison point. |
| `Nash_1999_Thermocouple_Probe_HighSpeed.pdf` | Nash, Caldwell, Zelman & Moum (1999), "A thermocouple probe for high-speed temperature measurement in the ocean," *JTECH* **16**, 1474–1483. [doi:10.1175/1520-0426(1999)016<1474:ATPFHS>2.0.CO;2](https://doi.org/10.1175/1520-0426(1999)016%3C1474:ATPFHS%3E2.0.CO;2) | Fast-temperature sensing background (OCR text layer added locally). |
| `Dillon_Caldwell_1980_Batchelor_Spectrum_Upper_Ocean.pdf` | Dillon & Caldwell (1980), "The Batchelor spectrum and dissipation in the upper ocean," *JGR* **85**(C4), 1910–1916. [doi:10.1029/JC085iC04p01910](https://doi.org/10.1029/JC085iC04p01910) | Batchelor-spectrum fitting foundations for the chi path. |
| `Ruddick_2000_ML_Batchelor_Fitting.pdf` | Ruddick, Anis & Thompson (2000), "Maximum likelihood spectral fitting: The Batchelor spectrum," *JTECH* **17**, 1541–1555. [doi:10.1175/1520-0426(2000)017<1541:MLSFTB>2.0.CO;2](https://doi.org/10.1175/1520-0426%282000%29017%3C1541:MLSFTB%3E2.0.CO;2) | The MLE Batchelor-fitting method family our chi path belongs to. |
| `Bluteau_2025_SinglePoint_Dissipation_Best_Practices_preprint.pdf` | Bluteau, Wain, Mullarney & Stevens (2025, EGUsphere preprint), "Best practices for estimating turbulent dissipation from oceanic single-point measurements." [doi:10.5194/egusphere-2025-4433](https://doi.org/10.5194/egusphere-2025-4433) | Companion best-practices to Lueck 2024 (point-velocity side of ATOMIX). |
| `Mudge_Lueck_1994_DSP_Oceanographic_Observations.pdf` | Mudge & Lueck (1994), "Digital signal processing to enhance oceanographic observations," *JTECH* **11**, 825–836. [doi:10.1175/1520-0426(1994)011<0825:DSPTEO>2.0.CO;2](https://doi.org/10.1175/1520-0426%281994%29011%3C0825:DSPTEO%3E2.0.CO;2) | Pre-emphasis/deconvolution background for the RSI signal chain (OCR added locally). |
| `McMillan_2016_Dissipation_HighRe_Tidal_Channel.pdf` | McMillan, Hay, Lueck & Wolk (2016), "Rates of dissipation of turbulent kinetic energy in a high Reynolds number tidal channel," *JTECH* **33**, 817–837. [doi:10.1175/JTECH-D-15-0167.1](https://doi.org/10.1175/JTECH-D-15-0167.1) | Inertial-subrange ε fitting at high Re — the high-ε method branch. |
| `Goto_2016_FastThermistor_Turbulence_VMP.pdf` | Goto, Yasuda & Nagasawa (2016), "Turbulence estimation using fast-response thermistors attached to a free-fall vertical microstructure profiler," *JTECH* **33**, 2065–2078. [doi:10.1175/JTECH-D-15-0220.1](https://doi.org/10.1175/JTECH-D-15-0220.1) | Thermistor-based ε estimation — chi-side methods. |
| `Ijichi_StLaurent_2025_Temperature_Microstructure_Spectra.pdf` | Ijichi & St. Laurent (2025), "Revisiting issues in estimating spectra of ocean temperature microstructure," *JTECH* **42**, 1137–1148. [doi:10.1175/JTECH-D-24-0087.1](https://doi.org/10.1175/JTECH-D-24-0087.1) | Current state of the art on temperature-microstructure spectra. |
| `Kraichnan_1968_Scalar_Field_Turbulence.pdf` | Kraichnan (1968), "Small-scale structure of a scalar field convected by turbulence," *Phys. Fluids* **11**, 945–953. [doi:10.1063/1.1692063](https://doi.org/10.1063/1.1692063) | The Kraichnan scalar spectrum our chi fitting uses. |
| `Welch_1967_FFT_Power_Spectra.pdf` | Welch (1967), "The use of fast Fourier transform for the estimation of power spectra," *IEEE Trans. Audio Electroacoust.* **AU-15**, 70–73. [doi:10.1109/TAU.1967.1161901](https://doi.org/10.1109/TAU.1967.1161901) | The Welch method behind all our spectral estimation. |
| `Lueck_1977_Spectral_Response_Thermistors.pdf` | Lueck, Hertzman & Osborn (1977), "The spectral response of thermistors," *Deep-Sea Res.* **24**, 951–970. [doi:10.1016/0146-6291(77)90565-3](https://doi.org/10.1016/0146-6291%2877%2990565-3) | Thermistor frequency-response foundations for the FP07 corrections. |
| `Gregg_Meagher_1980_Glass_Rod_Thermistor_Response.pdf` | Gregg & Meagher (1980), "The dynamic response of glass rod thermistors," *JGR* **85**(C5), 2779–2786. [doi:10.1029/JC085iC05p02779](https://doi.org/10.1029/JC085iC05p02779) | Companion thermistor-response reference. |
| `Peterson_Fer_2014_Glider_Temperature_Microstructure.pdf` | Peterson & Fer (2014), "Dissipation measurements using temperature microstructure from an underwater glider," *Methods Oceanogr.* **10**, 44–69. [doi:10.1016/j.mio.2014.05.002](https://doi.org/10.1016/j.mio.2014.05.002) | Chi-based ε from a glider — MR-relevant methods. |
| `Mater_Venayagamoorthy_2014_Mixing_Efficiency_Parameterization.pdf` | Mater & Venayagamoorthy (2014), "The quest for an unambiguous parameterization of mixing efficiency...," *GRL* **41**, 4646–4653. [doi:10.1002/2014GL060571](https://doi.org/10.1002/2014GL060571) | Mixing-efficiency parameterization context (gamma-scaling). |
| `Ijichi_2020_Mixing_Efficiency_Abyss.pdf` | Ijichi, St. Laurent, Polzin & Toole (2020), "How variable is mixing efficiency in the abyss?" *GRL* **47**, e2019GL086813. [doi:10.1029/2019GL086813](https://doi.org/10.1029/2019GL086813) | Abyssal Γ variability — companion to Ijichi & Hibiya 2018. |
| `Bogucki_1997_DNS_Passive_Scalars_Pr1.pdf` | Bogucki, Domaradzki & Yeung (1997), "Direct numerical simulations of passive scalars with Pr>1 advected by turbulent flow," *JFM* **343**, 111–130. [doi:10.1017/S0022112097005727](https://doi.org/10.1017/S0022112097005727) | DNS basis for the Kraichnan scalar spectrum used in chi fitting. |

## Cited in the repository docs, no local PDF yet

The full annotated bibliography behind `docs/chi_mathematics.md` /
`docs/epsilon_mathematics.md` lives in `docs/bibliography.md` (git-tracked).
Papers cited there but not yet in this collection:

| Citation | DOI |
|---|---|
| Batchelor (1959), Small-scale variation of convected quantities, *JFM* 5, 113–133 | [10.1017/S002211205900009X](https://doi.org/10.1017/S002211205900009X) |

(The BODC `10.5285/...` and Zenodo DOIs cited in `docs/atomix_benchmark.md` are
*dataset* DOIs — data lives in `AtomixData/`, not here.)

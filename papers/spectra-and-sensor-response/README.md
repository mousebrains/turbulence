# spectra-and-sensor-response

Spectral-estimation foundations and the thermistor / shear-probe frequency
response — the transfer functions and windowing behind the FP07 chi corrections
and the shear-probe ε path.

| Paper | File | Why it matters |
|---|---|---|
| Welch (1967), "The use of fast Fourier transform for the estimation of power spectra," *IEEE Trans. Audio Electroacoust.* **AU-15**, 70–73. [doi:10.1109/TAU.1967.1161901](https://doi.org/10.1109/TAU.1967.1161901) | `Welch_1967_FFT_Power_Spectra.pdf` | The Welch method behind all our spectral estimation. |
| Mudge & Lueck (1994), "Digital signal processing to enhance oceanographic observations," *JTECH* **11**, 825–836. [doi:10.1175/1520-0426(1994)011<0825:DSPTEO>2.0.CO;2](https://doi.org/10.1175/1520-0426%281994%29011%3C0825:DSPTEO%3E2.0.CO;2) | `Mudge_Lueck_1994_DSP_Oceanographic_Observations.pdf` | Pre-emphasis/deconvolution background for the RSI signal chain (OCR added locally). |
| Kraichnan (1968), "Small-scale structure of a scalar field convected by turbulence," *Phys. Fluids* **11**, 945–953. [doi:10.1063/1.1692063](https://doi.org/10.1063/1.1692063) | `Kraichnan_1968_Scalar_Field_Turbulence.pdf` | The Kraichnan scalar spectrum our chi fitting uses. |
| Bogucki, Domaradzki & Yeung (1997), "Direct numerical simulations of passive scalars with Pr>1 advected by turbulent flow," *JFM* **343**, 111–130. [doi:10.1017/S0022112097005727](https://doi.org/10.1017/S0022112097005727) | `Bogucki_1997_DNS_Passive_Scalars_Pr1.pdf` | DNS basis for the Kraichnan scalar spectrum used in chi fitting. |
| Lueck, Hertzman & Osborn (1977), "The spectral response of thermistors," *Deep-Sea Res.* **24**, 951–970. [doi:10.1016/0146-6291(77)90565-3](https://doi.org/10.1016/0146-6291%2877%2990565-3) | `Lueck_1977_Spectral_Response_Thermistors.pdf` | Thermistor frequency-response foundations for the FP07 corrections. |
| Gregg & Meagher (1980), "The dynamic response of glass rod thermistors," *JGR* **85**(C5), 2779–2786. [doi:10.1029/JC085iC05p02779](https://doi.org/10.1029/JC085iC05p02779) | `Gregg_Meagher_1980_Glass_Rod_Thermistor_Response.pdf` | Companion thermistor-response reference. |
| Nash, Caldwell, Zelman & Moum (1999), "A thermocouple probe for high-speed temperature measurement in the ocean," *JTECH* **16**, 1474–1483. [doi:10.1175/1520-0426(1999)016<1474:ATPFHS>2.0.CO;2](https://doi.org/10.1175/1520-0426%281999%29016%3C1474:ATPFHS%3E2.0.CO;2) | `Nash_1999_Thermocouple_Probe_HighSpeed.pdf` | Fast-temperature sensing background (OCR text layer added locally). |
| Nash & Moum (2002), *JPO* **32**, 2312–2333. [doi:10.1175/1520-0485(2002)032<2312:MEOTSF>2.0.CO;2](https://doi.org/10.1175/1520-0485%282002%29032%3C2312:MEOTSF%3E2.0.CO;2) | `Nash_Moum_2002_Salinity_Dissipation_FP07.pdf` | FP07 double-pole response (τ = 5.5 ms) context; thermistor response for the χ pair. See `docs/chi_mathematics.md` for which pole model the pipeline actually ships (single-pole default). |
| Macoun & Lueck (2004), *JTECH* **21**, 284–297. [doi:10.1175/1520-0426(2004)021<0284:MTSROT>2.0.CO;2](https://doi.org/10.1175/1520-0426%282004%29021%3C0284:MTSROT%3E2.0.CO;2) | `Macoun_Lueck_2004_Shear_Probe_Spatial_Response.pdf` | Shear-probe spatial response — probe separation/averaging effects on two-probe comparisons. |

Two of these are also in Rockland's Technical Note series — Mudge & Lueck (1994)
is **TN-002** and Macoun & Lueck (2004) is **TN-015**; they are filed here as the
papers. The vendor conversions and sensor/calibration notes live in
[`../rockland-technical-notes/`](../rockland-technical-notes/README.md):
TN-005 (counts → physical units), TN-010 (anti-aliasing filters), TN-030 (spectral
forms), TN-040 (temperature-gradient noise), TN-046 (inclinometers), and
TN-047 / TN-048 (FP07 calibration).

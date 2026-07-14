# epsilon-shear

TKE dissipation (ε) from shear probes: the ATOMIX best-practices and benchmark
papers, the Lueck (2022) statistics that the `epsilonLnSigma` uncertainty model
implements, Goodman coherent-noise removal, and cross-method / cross-instrument
comparison references.

| Paper | File | Why it matters |
|---|---|---|
| Lueck et al. (2024), "Best practices recommendations for estimating dissipation rates from shear probes," *Front. Mar. Sci.* **11**, 1334327. [doi:10.3389/fmars.2024.1334327](https://doi.org/10.3389/fmars.2024.1334327) | `Lueck_2024_Best_Practices_Shear_Probes.pdf` | The ATOMIX community standard; source of the FFT-length "sandwich" in `docs/perturb/dissipation_length.md`. |
| Fer, Dengler, Holtermann, Le Boyer & Lueck (2024), "ATOMIX benchmark datasets for dissipation rate measurements using shear probes," *Sci. Data* **11**, 518. [doi:10.1038/s41597-024-03323-y](https://doi.org/10.1038/s41597-024-03323-y) | `Fer_2024_ATOMIX_Benchmark_Datasets.pdf` | Benchmark datasets validating the epsilon pipeline (see `AtomixData/`). |
| Bluteau, Wain, Mullarney & Stevens (2025, EGUsphere preprint), "Best practices for estimating turbulent dissipation from oceanic single-point measurements." [doi:10.5194/egusphere-2025-4433](https://doi.org/10.5194/egusphere-2025-4433) | `Bluteau_2025_SinglePoint_Dissipation_Best_Practices_preprint.pdf` | Companion best-practices to Lueck 2024 (point-velocity side of ATOMIX). |
| Lueck (2022), Part I: Shear variance and dissipation rates, *JTECH* **39**, 1259–1271. [doi:10.1175/JTECH-D-21-0051.1](https://doi.org/10.1175/JTECH-D-21-0051.1) | `Lueck_2022_Statistics_Part1_Shear_Variance.pdf` | The σ_lnε variance model our `epsilonLnSigma` implements — also the null for ln(ε₁/ε₂) in the two-probe isotropy work; vmp194 matches it to 8%. |
| Lueck (2022), Part II: Shear spectra and a model spectrum, *JTECH* **39**, 1273–1282. [doi:10.1175/JTECH-D-21-0050.1](https://doi.org/10.1175/JTECH-D-21-0050.1) | `Lueck_2022_Statistics_Part2_Spectral_Model.pdf` | Companion spectral model; the FM misfit statistic in our diss product. |
| Goodman, Levine & Lueck (2006), *JTECH* **23**, 977–990. [doi:10.1175/JTECH1889.1](https://doi.org/10.1175/JTECH1889.1) | `Goodman_2006_TKE_Budget_AUV_Coherent_Noise.pdf` | The Ax/Ay coherent-noise removal in our ε path — and the tool for a vibration-cleaned covariance test (isotropy next step). |
| McMillan, Hay, Lueck & Wolk (2016), "Rates of dissipation of turbulent kinetic energy in a high Reynolds number tidal channel," *JTECH* **33**, 817–837. [doi:10.1175/JTECH-D-15-0167.1](https://doi.org/10.1175/JTECH-D-15-0167.1) | `McMillan_2016_Dissipation_HighRe_Tidal_Channel.pdf` | Inertial-subrange ε fitting at high Re — the high-ε method branch. |
| Le Boyer et al. (2021), "Modular, flexible, low-cost microstructure measurements: The Epsilometer," *JTECH* **38**, 657–668. [doi:10.1175/JTECH-D-20-0116.1](https://doi.org/10.1175/JTECH-D-20-0116.1) | `LeBoyer_2021_Epsilometer.pdf` | Alternative microstructure instrument; processing comparison point. |
| Whalen (2021), "Best practices for comparing ocean turbulence measurements across spatiotemporal scales," *JTECH* **38**, 837–841. [doi:10.1175/JTECH-D-20-0175.1](https://doi.org/10.1175/JTECH-D-20-0175.1) | `Whalen_2021_Best_Practices_Comparing_Ocean_Turbulence.pdf` | Lognormal averaging-scale pitfalls when comparing ε products. |

Rockland's own ε recipe and noise floors are in the vendor Technical Notes:
[TN-028](../rockland-technical-notes/README.md) (calculating the TKE dissipation
rate), [TN-030](../rockland-technical-notes/README.md) (forms of the shear /
strain spectra), [TN-042](../rockland-technical-notes/README.md) (shear-probe
noise), [TN-043](../rockland-technical-notes/README.md) (band of interest), and
[TN-061](../rockland-technical-notes/README.md) (Goodman coherent-noise spectral
bias). See [`../rockland-technical-notes/`](../rockland-technical-notes/README.md).

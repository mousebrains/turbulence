# Anisotropy & mixing-efficiency reference library

Reading list for (A) the VMP two-probe isotropy/anisotropy investigation and
(C) the mixing-efficiency scaling work (`perturb-plot gamma-scaling`).
Assembled 2026-07-09/10 from publisher open access + Pat's library copies.

## A. Isotropy / anisotropy of ocean turbulence (the active investigation)

| Paper | File | Why it matters |
|---|---|---|
| Gargett, Osborn & Nasmyth (1984), *JFM* **144**, 231–280. [doi:10.1017/S0022112084001592](https://doi.org/10.1017/S0022112084001592) | `Gargett_1984_Local_Isotropy_Stratified_Decay.pdf` | The isotropy benchmark: spectral isotropy requires ε/νN² ≳ 200; defines the decay/anisotropy regimes our Re_b U-curve should be read against. |
| Yamazaki & Osborn (1990), *JGR* **95**(C6), 9739–9744. [doi:10.1029/JC095iC06p09739](https://doi.org/10.1029/JC095iC06p09739) | `Yamazaki_Osborn_1990_Dissipation_Stratified.pdf` | Isotropic ε formula usable for Re_b > 20 — the classic threshold for two-probe work. |
| Itsweire, Koseff, Briggs & Ferziger (1993), *JPO* **23**, 1508–1522. [doi:10.1175/1520-0485(1993)023<1508:TISSFI>2.0.CO;2](https://doi.org/10.1175/1520-0485(1993)023%3C1508:TISSFI%3E2.0.CO;2) | `Itsweire_1993_Stratified_Shear_Flows_Mixing.pdf` | DNS of stratified shear flows: how shear-probe-style estimates degrade with buoyancy — context for anisotropy at low Re_b. |
| Smyth & Moum (2000), *Phys. Fluids* **12**, 1343–1362. [doi:10.1063/1.870385](https://doi.org/10.1063/1.870385) | `Smyth_Moum_2000_Anisotropy_Stratified_Mixing_Layers.pdf` | Anisotropy of dissipation-range turbulence in stratified mixing layers — length-scale ratios we can mirror with L_T/L_O. |
| Smyth & Moum (2000), "Length scales of turbulence in stably stratified mixing layers", *Phys. Fluids* **12**, 1327–1342. [doi:10.1063/1.870384](https://doi.org/10.1063/1.870384) | `Smyth_Moum_2000_Length_Scales_Stratified.pdf` | The L_T/L_O/L_E foundations — companion to the Anisotropy paper; underpins the L_T-adaptive dissipation-window controller. |

## B. Statistics of dissipation estimates (the two-probe null hypothesis)

| Paper | File | Why it matters |
|---|---|---|
| Lueck (2022), Part I: Shear variance and dissipation rates, *JTECH* **39**, 1259–1271. [doi:10.1175/JTECH-D-21-0051.1](https://doi.org/10.1175/JTECH-D-21-0051.1) | `Lueck_2022_Statistics_Part1_Shear_Variance.pdf` | The σ_lnε variance model our `epsilonLnSigma` implements — the null for ln(ε₁/ε₂). vmp194 matches it to 8%. |
| Lueck (2022), Part II: Shear spectra and a model spectrum, *JTECH* **39**, 1273–1282. [doi:10.1175/JTECH-D-21-0050.1](https://doi.org/10.1175/JTECH-D-21-0050.1) | `Lueck_2022_Statistics_Part2_Spectral_Model.pdf` | Companion spectral model; the FM misfit statistic in our diss product. |
| ATOMIX shear-probe benchmark (already in `papers/`): `../LeBoyer_2021_Epsilometer.pdf`, `../Fer_2024_ATOMIX_Benchmark_Datasets.pdf` | — | Processing-standard cross-check for our ε pipeline. |

## C. Mixing efficiency / gamma-scaling chain (Lewin Fig. 5)

| Paper | File | Why it matters |
|---|---|---|
| Lewin, Kaminski, McSweeney & Waterhouse (2025), *JPO* **55**, 1735–1750. [doi:10.1175/JPO-D-25-0012.1](https://doi.org/10.1175/JPO-D-25-0012.1) | `Lewin_2025_Multiscale_Mixing_Inner_Shelf.pdf` | The Fig. 5 we reproduce: Γ vs R_OT/Re_b/Ri_g, QC recipe (Re_b>20, Cox>50). |
| Kaminski, D'Asaro, Shcherbina & Harcourt (2021), *JPO* **51**, 3163–3181. [doi:10.1175/JPO-D-21-0032.1](https://doi.org/10.1175/JPO-D-21-0032.1) | `Kaminski_2021_North_Pacific_Transition_Layer_preprint.pdf` (EarthArXiv) | Operational patch-N² definition (their Eqs. 9–10, the **rms** form) that `processing/thorpe.py` implements. |
| Smyth, Moum & Caldwell (2001), *JPO* **31**, 1969–1992. [doi:10.1175/1520-0485(2001)031<1969:TEOMIT>2.0.CO;2](https://doi.org/10.1175/1520-0485(2001)031%3C1969:TEOMIT%3E2.0.CO;2) | `Smyth_2001_Efficiency_Mixing_Turbulent_Patches.pdf` | Origin of the overturn-weighted stratification and Γ–R_OT interpretation (patch age). |
| Smyth (2020), *JPO* **50**, 2141–2150. [doi:10.1175/JPO-D-20-0083.1](https://doi.org/10.1175/JPO-D-20-0083.1) | `Smyth_2020_Marginal_Instability_Mixing.pdf` | Marginal instability picture; Γ ~ (L_T/L_O)^{4/3} scaling context. |
| Gregg, D'Asaro, Riley & Kunze (2018), *Annu. Rev. Mar. Sci.* **10**, 443–473. [doi:10.1146/annurev-marine-121916-063643](https://doi.org/10.1146/annurev-marine-121916-063643) | `Gregg_2018_Mixing_Efficiency_Review.pdf` | The mixing-efficiency review — definitions and pitfalls (Γ vs R_f vs Γ₀=0.2). |
| Monismith, Koseff & White (2018), *GRL* **45**, 5627–5634. [doi:10.1029/2018GL077229](https://doi.org/10.1029/2018GL077229) | `Monismith_2018_Mixing_Efficiency_Constant.pdf` | Γ ~ Re_b^{-1/2} at high Re_b — panel (b)'s slope guide. |
| Ijichi & Hibiya (2018), *JPO* **48**, 1815–1830. [doi:10.1175/JPO-D-17-0275.1](https://doi.org/10.1175/JPO-D-17-0275.1) | `Ijichi_Hibiya_2018_Mixing_Efficiency_Deep_Ocean.pdf` | Observed Γ ~ R_OT^{-4/3} — panel (a)'s slope guide. |
| Shih, Koseff, Ivey & Ferziger (2005), *JFM* **525**, 193–214. [doi:10.1017/S0022112004002587](https://doi.org/10.1017/S0022112004002587) | `Shih_2005_Reb_Parameterization.pdf` | Re_b regime framework (molecular/transitional/energetic) behind the K_ρ parameterizations. |
| Osborn (1980), *JPO* **10**, 83–89. [doi:10.1175/1520-0485(1980)010<0083:EOTLRO>2.0.CO;2](https://doi.org/10.1175/1520-0485(1980)010%3C0083:EOTLRO%3E2.0.CO;2) | `Osborn_1980_Vertical_Diffusion_Dissipation.pdf` | K_ρ = Γ ε/N², Γ₀ = 0.2 — the Osborn relation in `processing/mixing.py`. |
| Osborn & Cox (1972), *Geophys. Fluid Dyn.* **3**, 321–345. [doi:10.1080/03091927208236085](https://doi.org/10.1080/03091927208236085) | `Osborn_Cox_1972_Oceanic_Fine_Structure.pdf` | Osborn–Cox K_T and the Cox number (our C_x > 50 QC). |
| Oakey (1982), *JPO* **12**, 256–271. [doi:10.1175/1520-0485(1982)012<0256:DOTROD>2.0.CO;2](https://doi.org/10.1175/1520-0485(1982)012%3C0256:DOTROD%3E2.0.CO;2) | `Oakey_1982_Epsilon_Chi_Simultaneous.pdf` | First simultaneous ε+χ mixing coefficient — the measured Γ our pipeline computes. |
| Holleman, Geyer & Ralston (2016), *JPO* **46**, 1769–1783. [doi:10.1175/JPO-D-15-0193.1](https://doi.org/10.1175/JPO-D-15-0193.1) | `Holleman_2016_Salt_Wedge_Mixing_Efficiency.pdf` | High-Γ observations in energetic stratified turbulence — context for our Γ ≳ 0.2 medians. |

## D. Overturns / Thorpe scales

| Paper | File | Why it matters |
|---|---|---|
| Thorpe (1977), *Phil. Trans. Roy. Soc. A* **286**, 125–181. [doi:10.1098/rsta.1977.0112](https://doi.org/10.1098/rsta.1977.0112) | `Thorpe_1977_Scottish_Loch.pdf` | The original Thorpe sort and displacement scale. |
| Dillon (1982), *JGR* **87**(C12), 9601–9613. [doi:10.1029/JC087iC12p09601](https://doi.org/10.1029/JC087iC12p09601) | `Dillon_1982_Thorpe_Ozmidov_Scales.pdf` | L_O/L_T ≈ 0.8 — the Thorpe–Ozmidov link behind R_OT. |
| Galbraith & Kelley (1996), *JTECH* **13**, 688–702. [doi:10.1175/1520-0426(1996)013<0688:IOICP>2.0.CO;2](https://doi.org/10.1175/1520-0426(1996)013%3C0688:IOICP%3E2.0.CO;2) | `Galbraith_Kelley_1996_Identifying_Overturns.pdf` | Run-length + water-mass overturn validity tests — our `--min-run` gate. |
| Mater, Venayagamoorthy, St. Laurent & Moum (2015), *JPO* **45**, 2497–2521. [doi:10.1175/JPO-D-14-0128.1](https://doi.org/10.1175/JPO-D-14-0128.1) | `Mater_2015_Thorpe_Biases_Part1.pdf` | Thorpe-scale ε biases from ocean overturns — caveats for the R_OT panel. |
| Scotti (2015), *JPO* **45**, 2522–2543. [doi:10.1175/JPO-D-14-0092.1](https://doi.org/10.1175/JPO-D-14-0092.1) | `Scotti_2015_Thorpe_Biases_Part2.pdf` | Part II: energetics arguments — convective vs shear overturns bias differently. |

## E2. Processing best practices & platforms

| Paper | File | Why it matters |
|---|---|---|
| Lueck et al. (2024), "Best practices recommendations for estimating dissipation rates from shear probes," *Front. Mar. Sci.*, **11**, 1334327. [doi:10.3389/fmars.2024.1334327](https://doi.org/10.3389/fmars.2024.1334327) | `../Lueck_2024_Best_Practices_Shear_Probes.pdf` | THE community best-practices paper (the ATOMIX benchmark defers to it) — fft/diss window guidance for the dissipation-length design note. |
| Fer, Peterson & Ullgren (2014), "Microstructure measurements from an underwater glider in the turbulent Faroe Bank Channel overflow," *JTECH*, **31**, 1128–1150. [doi:10.1175/JTECH-D-13-00221.1](https://doi.org/10.1175/JTECH-D-13-00221.1) | `Fer_2014_Glider_Microstructure_FBC.pdf` | MicroRider-on-glider processing choices; the dataset behind Lueck 2022's low-ε validation. |
| Shapiro, Ferris, Kassis, Lueck, Merrifield & St. Laurent (2025), "Near real-time processing and telemetry of turbulent dissipation rate estimates by autonomous underwater gliders," *JTECH*. [doi:10.1175/JTECH-D-25-0058.1](https://doi.org/10.1175/JTECH-D-25-0058.1) | `Shapiro_2025_Glider_Realtime_Dissipation.pdf` | Recent glider/MR-class processing decisions (window choices under power/telemetry constraints). |
| Scheifele, Waterman, Merckelbach & Carpenter (2018), "Measuring the dissipation rate of turbulent kinetic energy in strongly stratified, low-energy environments: A case study from the Arctic Ocean," *JGR Oceans*, **123**, 5459–5480. [doi:10.1029/2017JC013731](https://doi.org/10.1029/2017JC013731) | `Scheifele_2018_LowEnergy_Dissipation_Arctic.pdf` | MicroRider-on-glider at ε → 1e-12: the closest published precedent for our deep-MR window/noise-floor choices. |

## E. VMP sensor methods (used directly by the pipeline)

| Paper | File | Why it matters |
|---|---|---|
| Goodman, Levine & Lueck (2006), *JTECH* **23**, 977–990. [doi:10.1175/JTECH1889.1](https://doi.org/10.1175/JTECH1889.1) | `Goodman_2006_TKE_Budget_AUV_Coherent_Noise.pdf` | The Ax/Ay coherent-noise removal in our ε path — and the tool for a vibration-cleaned covariance test (isotropy next step). |
| Macoun & Lueck (2004), *JTECH* **21**, 284–297. [doi:10.1175/1520-0426(2004)021<0284:MTSROT>2.0.CO;2](https://doi.org/10.1175/1520-0426(2004)021%3C0284:MTSROT%3E2.0.CO;2) | `Macoun_Lueck_2004_Shear_Probe_Spatial_Response.pdf` | Shear-probe spatial response — probe separation/averaging effects on two-probe comparisons. |
| Nash & Moum (2002), *JPO* **32**, 2312–2333. [doi:10.1175/1520-0485(2002)032<2312:MEOTSF>2.0.CO;2](https://doi.org/10.1175/1520-0485(2002)032%3C2312:MEOTSF%3E2.0.CO;2) | `Nash_Moum_2002_Salinity_Dissipation_FP07.pdf` | FP07 double-pole response (τ = 5.5 ms) used by Lewin et al.; thermistor response context for the χ pair. |

## Status

- **Complete: all 30 papers in hand (+ Lueck 2024 best practices in `papers/`).** (The three legacy AMS scans — Osborn
  1980, Oakey 1982, Galbraith & Kelley 1996 — carry OCR text layers added
  locally; Osborn & Cox 1972 and Gargett et al. 1984 are ILL scans with their
  cover sheets on page 1.)

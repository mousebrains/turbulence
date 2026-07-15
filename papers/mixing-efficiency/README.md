# mixing-efficiency

Mixing efficiency Γ and the gamma-scaling chain we reproduce (Lewin et al. 2025
Fig. 5): the Osborn and Osborn–Cox relations in `processing/mixing.py`, the
Re_b and R_OT parameterizations, and the Γ observational literature.

| Paper | File | Why it matters |
|---|---|---|
| Lewin, Kaminski, McSweeney & Waterhouse (2025), *JPO* **55**, 1735–1750. [doi:10.1175/JPO-D-25-0012.1](https://doi.org/10.1175/JPO-D-25-0012.1) | `Lewin_2025_Multiscale_Mixing_Inner_Shelf.pdf` | The Fig. 5 we reproduce: Γ vs R_OT/Re_b/Ri_g, QC recipe (Re_b>20, Cox>50). |
| Kaminski, D'Asaro, Shcherbina & Harcourt (2021), *JPO* **51**, 3163–3181. [doi:10.1175/JPO-D-21-0032.1](https://doi.org/10.1175/JPO-D-21-0032.1) | `Kaminski_2021_North_Pacific_Transition_Layer_preprint.pdf` (EarthArXiv) | Operational patch-N² definition (their Eqs. 9–10, the **rms** form) that `processing/thorpe.py` implements. |
| Oakey (1982), *JPO* **12**, 256–271. [doi:10.1175/1520-0485(1982)012<0256:DOTROD>2.0.CO;2](https://doi.org/10.1175/1520-0485%281982%29012%3C0256:DOTROD%3E2.0.CO;2) | `Oakey_1982_Epsilon_Chi_Simultaneous.pdf` | First simultaneous ε+χ mixing coefficient — the measured Γ our pipeline computes. |
| Osborn (1980), *JPO* **10**, 83–89. [doi:10.1175/1520-0485(1980)010<0083:EOTLRO>2.0.CO;2](https://doi.org/10.1175/1520-0485%281980%29010%3C0083:EOTLRO%3E2.0.CO;2) | `Osborn_1980_Vertical_Diffusion_Dissipation.pdf` | K_ρ = Γ ε/N², Γ₀ = 0.2 — the Osborn relation in `processing/mixing.py`. |
| Osborn & Cox (1972), *Geophys. Fluid Dyn.* **3**, 321–345. [doi:10.1080/03091927208236085](https://doi.org/10.1080/03091927208236085) | `Osborn_Cox_1972_Oceanic_Fine_Structure.pdf` | Osborn–Cox K_T and the Cox number (our C_x > 50 QC). |
| Gregg, D'Asaro, Riley & Kunze (2018), *Annu. Rev. Mar. Sci.* **10**, 443–473. [doi:10.1146/annurev-marine-121916-063643](https://doi.org/10.1146/annurev-marine-121916-063643) | `Gregg_2018_Mixing_Efficiency_Review.pdf` | The mixing-efficiency review — definitions and pitfalls (Γ vs R_f vs Γ₀=0.2). |
| Monismith, Koseff & White (2018), *GRL* **45**, 5627–5634. [doi:10.1029/2018GL077229](https://doi.org/10.1029/2018GL077229) | `Monismith_2018_Mixing_Efficiency_Constant.pdf` | Γ ~ Re_b^{-1/2} at high Re_b — panel (b)'s slope guide. |
| Ijichi & Hibiya (2018), *JPO* **48**, 1815–1830. [doi:10.1175/JPO-D-17-0275.1](https://doi.org/10.1175/JPO-D-17-0275.1) | `Ijichi_Hibiya_2018_Mixing_Efficiency_Deep_Ocean.pdf` | Observed Γ ~ R_OT^{-4/3} — panel (a)'s slope guide. |
| Ijichi, St. Laurent, Polzin & Toole (2020), "How variable is mixing efficiency in the abyss?" *GRL* **47**, e2019GL086813. [doi:10.1029/2019GL086813](https://doi.org/10.1029/2019GL086813) | `Ijichi_2020_Mixing_Efficiency_Abyss.pdf` | Abyssal Γ variability — companion to Ijichi & Hibiya 2018. |
| Mater & Venayagamoorthy (2014), "The quest for an unambiguous parameterization of mixing efficiency...," *GRL* **41**, 4646–4653. [doi:10.1002/2014GL060571](https://doi.org/10.1002/2014GL060571) | `Mater_Venayagamoorthy_2014_Mixing_Efficiency_Parameterization.pdf` | Mixing-efficiency parameterization context (gamma-scaling). |
| Shih, Koseff, Ivey & Ferziger (2005), *JFM* **525**, 193–214. [doi:10.1017/S0022112004002587](https://doi.org/10.1017/S0022112004002587) | `Shih_2005_Reb_Parameterization.pdf` | Re_b regime framework (molecular/transitional/energetic) behind the K_ρ parameterizations. |
| Holleman, Geyer & Ralston (2016), *JPO* **46**, 1769–1783. [doi:10.1175/JPO-D-15-0193.1](https://doi.org/10.1175/JPO-D-15-0193.1) | `Holleman_2016_Salt_Wedge_Mixing_Efficiency.pdf` | High-Γ observations in energetic stratified turbulence — context for our Γ ≳ 0.2 medians. |
| Smyth, Moum & Caldwell (2001), *JPO* **31**, 1969–1992. [doi:10.1175/1520-0485(2001)031<1969:TEOMIT>2.0.CO;2](https://doi.org/10.1175/1520-0485%282001%29031%3C1969:TEOMIT%3E2.0.CO;2) | `Smyth_2001_Efficiency_Mixing_Turbulent_Patches.pdf` | Origin of the overturn-weighted stratification and Γ–R_OT interpretation (patch age). |
| Smyth (2020), *JPO* **50**, 2141–2150. [doi:10.1175/JPO-D-20-0083.1](https://doi.org/10.1175/JPO-D-20-0083.1) | `Smyth_2020_Marginal_Instability_Mixing.pdf` | Marginal instability picture; Γ ~ (L_T/L_O)^{4/3} scaling context. |

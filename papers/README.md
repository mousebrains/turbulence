# papers/ — annotated reference library

Reading list for the microstructure-tpw processing chain and the associated
analyses (epsilon, chi, mixing efficiency, overturns, isotropy). The PDFs are
**not** tracked in git (they are large and mostly copyrighted); only these
`README.md` indexes are. Each subject subdirectory holds its own PDFs plus a
`README.md` with a `Paper | File | Why it matters` table.

## Subject groups

| Group | Papers | Scope |
|---|---|---|
| [spectra-and-sensor-response/](spectra-and-sensor-response/README.md) | 9 | Spectral-estimation foundations (Welch, DSP) and thermistor / shear-probe frequency response — the transfer functions behind FP07 and shear corrections. |
| [epsilon-shear/](epsilon-shear/README.md) | 10 | TKE dissipation (ε) from shear probes: ATOMIX best practices and benchmark, the Lueck (2022) statistics, Goodman coherent-noise removal, cross-method comparisons, and the moored-instrument precedent. |
| [chi-thermal/](chi-thermal/README.md) | 7 | Thermal-variance dissipation (χ): the Batchelor (1959) origin papers, Batchelor / Kraichnan spectrum fitting, MLE estimation, and temperature-microstructure methods. |
| [mixing-efficiency/](mixing-efficiency/README.md) | 14 | Mixing efficiency Γ and the gamma-scaling chain (Lewin Fig. 5): Osborn / Osborn–Cox, Re_b and R_OT parameterizations, and Γ observations. |
| [overturns-thorpe/](overturns-thorpe/README.md) | 5 | Thorpe-scale overturn analysis: the sort, the Thorpe–Ozmidov link, overturn-validity tests, and the Thorpe-scale ε biases. |
| [stratified-turbulence-anisotropy/](stratified-turbulence-anisotropy/README.md) | 6 | Isotropy criteria and the structure of stratified turbulence — the backdrop for the VMP two-probe isotropy investigation. |
| [gliders-and-platforms/](gliders-and-platforms/README.md) | 17 | MicroRider-on-glider / AUV platform processing and the through-water speed behind `speed.method`: the Slocum flight models, angle of attack, and the published `U_EM` scale factors (Merckelbach et al. 2019; Tanaka et al. 2022). |
| [current-meters/](current-meters/README.md) | 18 | Calibration and dynamic response of EM current meters: the Aubrey–Trowbridge / Guza dispute over gain error, and the flowmeter physics of gain, zero and conductivity behind the `em_bench_zero.py` bench test. |
| [rockland-technical-notes/](rockland-technical-notes/README.md) | 30 | Rockland Scientific vendor Technical Notes: the `.p`/ODAS file format, count→physical-unit conversion, the ε recipe, shear/thermistor noise floors, FP07 calibration, and field/deployment technique for the VMP and MicroRider. |

86 references across the eight subject groups, mostly peer-reviewed papers plus
a few reports, a patent, a handbook chapter, a preprint and a vendor working
note, and 30 Rockland technical notes in the ninth (`rockland-technical-notes/`).

## Cited in the repository docs, no local PDF yet

The full annotated bibliography behind `docs/chi_mathematics.md` /
`docs/epsilon_mathematics.md` lives in `docs/bibliography.md` (git-tracked).
Every paper cited there is now in this collection — Batchelor (1959) Parts 1 & 2
were the last gap, now filed under `chi-thermal/`.

(The BODC `10.5285/...` and Zenodo DOIs cited in `docs/atomix_benchmark.md` are
*dataset* DOIs — the data lives in `AtomixData/`, not here.)

## Provenance notes

- Assembled 2026-07-09/12 from publisher open access plus Pat's library copies.
- `rockland-technical-notes/` added 2026-07-14 from Rockland's public
  [Technical Notes page](https://rocklandscientific.com/support/technical-notes/)
  plus notes supplied by Rockland on request (the "Contact Support" ones, marked
  **†** in that group's README). Three notes that are journal-paper reprints
  (TN-002, TN-015, TN-016) are filed as the papers, not as notes — TN-016 =
  Rehmann & Hwang (2005) is the one new paper this brought in.
- `current-meters/` opened 2026-08-26 for the AEM1-G in-situ calibration work.
  Velocity-sensor calibration is a separate literature from the microstructure
  groups here — largely nearshore, largely 1980s — and the group holds the complete
  Aubrey--Trowbridge / Guza exchange -- the WHOI-84-20 laboratory report, the
  1985 paper, the 1988 comment and the 1988 reply -- with four named gaps
  remaining in its own README, of which Dibble & Sollitt (1981) matters most.
- `gliders-and-platforms/` and `current-meters/` extended 2026-09-15 from the
  AEM1-G repository's reference library (`../AEM1-G/docs/references/`), by the
  split rule in `docs/aem1g_split_disposition.md`: a paper comes here if the
  next deployment would need it with the AEM1-G paper abandoned. 29 PDFs came
  over: 14 on glider flight models, angle of attack and MicroRider-on-glider
  practice; 14 on EM flowmeter gain, zero and conductivity; and Lueck et al.
  (1997) into `epsilon-shear/`. Ten more were already here under other
  filenames or as TN-022. Left in AEM1-G as manuscript context only: Seaglider
  and Spray navigation (Eriksen 2001, Frajka-Williams 2011, Rudnick 2018,
  Bennett 2021), glider hydrodynamics without in-situ data (Graver 2005,
  Jenkins 2003, Williams 2008, Park 2016), glider ADCP, navigation and review
  papers (Woithe 2011, Thurnherr 2015, de Fommervault 2019, Rudnick 2016),
  glider chi and Thorpe work without a MicroRider (Sheehan 2023, Leadbitter
  2022, Fer 2024 dataset summary), and EM papers that serve the manuscript's
  history and framing rather than the handling of a `U_EM` record (Shercliff
  1954, McCullough 1979, Beardsley 1981, Clay & Longworth 1986, Onishi & Otobe
  2006, Woodward 1978 proceedings, Chen 2023, Zhang 2025, Matos 2026). The PDFs
  here are the same bytes as there, under this library's naming (Kolås et al.
  2022 was filed there as `Fer_etal_2022_…`).
- OCR text layers were added locally to three legacy AMS scans — Osborn (1980)
  and Oakey (1982) in `mixing-efficiency/`, Galbraith & Kelley (1996) in
  `overturns-thorpe/` — and to Nash (1999) and Mudge & Lueck (1994) in
  `spectra-and-sensor-response/`.
- Osborn & Cox (1972, `mixing-efficiency/`) and Gargett et al. (1984,
  `stratified-turbulence-anisotropy/`) are inter-library-loan scans with their
  cover sheets on page 1.
- Several papers sit at a topic boundary and are cross-referenced from the
  neighboring group's README where relevant (e.g. the Lueck 2022 statistics
  under `epsilon-shear/` also underpin the two-probe isotropy null; Peterson &
  Fer 2014 under `chi-thermal/` is also a glider-platform paper).

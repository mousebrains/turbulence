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
| [epsilon-shear/](epsilon-shear/README.md) | 9 | TKE dissipation (ε) from shear probes: ATOMIX best practices and benchmark, the Lueck (2022) statistics, Goodman coherent-noise removal, and cross-method comparisons. |
| [chi-thermal/](chi-thermal/README.md) | 7 | Thermal-variance dissipation (χ): the Batchelor (1959) origin papers, Batchelor / Kraichnan spectrum fitting, MLE estimation, and temperature-microstructure methods. |
| [mixing-efficiency/](mixing-efficiency/README.md) | 14 | Mixing efficiency Γ and the gamma-scaling chain (Lewin Fig. 5): Osborn / Osborn–Cox, Re_b and R_OT parameterizations, and Γ observations. |
| [overturns-thorpe/](overturns-thorpe/README.md) | 5 | Thorpe-scale overturn analysis: the sort, the Thorpe–Ozmidov link, overturn-validity tests, and the Thorpe-scale ε biases. |
| [stratified-turbulence-anisotropy/](stratified-turbulence-anisotropy/README.md) | 6 | Isotropy criteria and the structure of stratified turbulence — the backdrop for the VMP two-probe isotropy investigation. |
| [gliders-and-platforms/](gliders-and-platforms/README.md) | 3 | MicroRider-on-glider / AUV platform processing — precedent for the deep-MR window and noise-floor choices. |
| [rockland-technical-notes/](rockland-technical-notes/README.md) | 30 | Rockland Scientific vendor Technical Notes: the `.p`/ODAS file format, count→physical-unit conversion, the ε recipe, shear/thermistor noise floors, FP07 calibration, and field/deployment technique for the VMP and MicroRider. |

53 peer-reviewed papers across the seven subject groups, plus 30 Rockland
technical notes in the eighth (`rockland-technical-notes/`).

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

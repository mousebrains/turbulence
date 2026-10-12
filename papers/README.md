# papers/ — annotated reference library

Reading list for the microstructure-tpw processing chain and the associated
analyses (epsilon, chi, mixing efficiency, overturns, isotropy). The PDFs are
**not** tracked in git (they are large and mostly copyrighted); only these
`README.md` indexes are. Each subject subdirectory holds its own PDFs plus a
`README.md` with a `Paper | File | Why it matters` table.

## Subject groups

| Group | Papers | Scope |
|---|---|---|
| [spectra-and-sensor-response/](spectra-and-sensor-response/README.md) | 10 | Spectral-estimation foundations (Welch, DSP) and thermistor / shear-probe frequency response — the transfer functions behind FP07 and shear corrections. |
| [epsilon-shear/](epsilon-shear/README.md) | 21 | TKE dissipation (ε) from shear probes: ATOMIX best practices and benchmark, the Lueck (2022) statistics, Goodman coherent-noise removal, cross-method comparisons, and the moored-instrument precedent. |
| [chi-thermal/](chi-thermal/README.md) | 8 | Thermal-variance dissipation (χ): the Batchelor (1959) origin papers, Batchelor / Kraichnan spectrum fitting, MLE estimation, and temperature-microstructure methods. |
| [mixing-efficiency/](mixing-efficiency/README.md) | 17 | Mixing efficiency Γ and the gamma-scaling chain (Lewin Fig. 5): Osborn / Osborn–Cox, Re_b and R_OT parameterizations, and Γ observations. |
| [overturns-thorpe/](overturns-thorpe/README.md) | 5 | Thorpe-scale overturn analysis: the sort, the Thorpe–Ozmidov link, overturn-validity tests, and the Thorpe-scale ε biases. |
| [stratified-turbulence-anisotropy/](stratified-turbulence-anisotropy/README.md) | 9 | Isotropy criteria and the structure of stratified turbulence — the backdrop for the VMP two-probe isotropy investigation. |
| [gliders-and-platforms/](gliders-and-platforms/README.md) | 31 | MicroRider-on-glider / AUV platform processing and the through-water speed behind `speed.method`: the Slocum flight models, angle of attack, and the published `U_EM` scale factors (Merckelbach et al. 2019; Tanaka et al. 2022). Also Seaglider and Spray flight-model accuracy against independent references, glider-mounted ADCPs and DVLs, and FP07-only glider microstructure. |
| [current-meters/](current-meters/README.md) | 44 | Calibration and dynamic response of EM current meters: the Aubrey–Trowbridge / Guza dispute over gain error, and the flowmeter physics of gain, zero and conductivity behind the `em_bench_zero.py` bench test; the weight-function theory of where the gain comes from; and field performance and intercomparisons. |
| [ctd/](ctd/README.md) | 6 | CTD response and sampling on moving platforms: conductivity-cell thermal lag on glider CTDs, T–C alignment, pumped versus free-flushed flow, spatial resolution, and pressure-to-depth. The background for the hotel files, the `fp07-cal` reference, `ct_align` and depth conversion. |
| [rockland-technical-notes/](rockland-technical-notes/README.md) | 30 | Rockland Scientific vendor Technical Notes: the `.p`/ODAS file format, count→physical-unit conversion, the ε recipe, shear/thermistor noise floors, FP07 calibration, and field/deployment technique for the VMP and MicroRider. |

151 references across the nine subject groups, mostly peer-reviewed papers plus
reports, theses, book chapters, vendor application notes, a patent and a
processing manual, and 30 Rockland technical notes in the tenth
(`rockland-technical-notes/`). Counts are README rows; every PDF on disk has one.

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
- 2026-10-05 (PRs #204-#206): a literature search for the Hydrographer Bank /
  Palau white paper filed 80 open-access PDFs, opening eight white-paper folders
  (`topographic-wakes/` first, with Chor & Wenegrat 2025) and adding four rows to
  `epsilon-shear/` (Guerra & Thomson 2017, McCaffrey et al. 2015, Scannell et al.
  2017, Thomson et al. 2012), and wrote `TO_FETCH.md`. The same day the loose
  top-level PDFs were filed: Bluteau (2025, *L&O Methods*) to `epsilon-shear/`,
  Li et al. (2023) to `mixing-efficiency/`, and Reidenbach et al. (2006) to
  `bottom-boundary-layers/`; a second download of Lueck (2022) Part II was a
  duplicate. The eight folders have since moved out (below); the five
  `epsilon-shear/` papers and Li et al. stay here.
- `current-meters/` opened 2026-08-26 for the AEM1-G in-situ calibration work.
  Velocity-sensor calibration is a separate literature from the microstructure
  groups here — largely nearshore, largely 1980s — and the group holds the complete
  Aubrey--Trowbridge / Guza exchange -- the WHOI-84-20 laboratory report, the
  1985 paper, the 1988 comment and the 1988 reply. All four gaps its own README
  named were filed 2026-10-10: Dibble & Sollitt (1981), the one that mattered
  most, Cunningham, Guza & Lowe (1979), Bowden & White (1966) and Guza &
  Thornton (1980).
- `gliders-and-platforms/` and `current-meters/` extended 2026-09-15 from the
  AEM1-G repository's reference library (`../AEM1-G/docs/references/`), by the
  split rule in `docs/aem1g_split_disposition.md`: a paper comes here if the
  next deployment would need it with the AEM1-G paper abandoned. 29 PDFs came
  over: 14 on glider flight models, angle of attack and MicroRider-on-glider
  practice; 14 on EM flowmeter gain, zero and conductivity; and Lueck et al.
  (1997) into `epsilon-shear/`. Ten more were already here under other
  filenames or as TN-022. Left in AEM1-G as manuscript context only: Seaglider
  and Spray navigation (Eriksen 2001, Frajka-Williams 2011, Rudnick 2018,
  Bennett 2021), glider hydrodynamics without an independent speed reference
  (Graver 2005, Jenkins 2003, Williams 2008, Park 2016), glider ADCP, navigation and review
  papers (Woithe 2011, Thurnherr 2015, de Fommervault 2019, Rudnick 2016),
  glider FP07 microstructure on MicroPod Seagliders (Sheehan 2023, Leadbitter
  2022) and a MicroRider-on-Slocum dataset summary (Fer 2024), and EM papers that serve the manuscript's
  history and framing rather than the handling of a `U_EM` record (Shercliff
  1954, McCullough 1979, Beardsley 1981, Clay & Longworth 1986, Onishi & Otobe
  2006, Woodward 1978 proceedings, Chen 2023, Zhang 2025, Matos 2026). The PDFs
  here are the same bytes as there, under this library's naming (Kolås et al.
  2022 was filed there as `Fer_etal_2022_…`). **Revised 2026-10-10:** Pat
  judged that papers useful for diagnosing the sensor belong here too, so all
  of the above except Rudnick 2016, Onishi & Otobe 2006, Zhang 2025, Matos 2026
  and the Woodward 1978 proceedings were copied in from `~/Desktop/Papers`, with more EM
  flowmeter physics and field intercomparisons (see the 2026-10-10 notes below).
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
- Synced 2026-10-10 against Pat's private library `~/Desktop/Papers` (the
  catalogued superset; its `catalog.yaml` entries record each file's `source`
  and `sha256`). 175 PDFs here were already byte-identical to it. Replaced by
  the library's copy: three preprints by their published versions, renamed
  (Bluteau et al. 2026 in `epsilon-shear/`, Kokoszka et al. 2026 in
  `gliders-and-platforms/`, Kaminski et al. 2021 in `mixing-efficiency/`);
  six accepted manuscripts or arXiv copies by the published versions under the
  same names (Ardhuin 2009 and 2014, Bertin 2018, Aravind 2026, Liu 2025,
  Puthan 2022); and nine scans by copies with an OCR text layer added, page
  images unchanged (Kraichnan 1968, Itsweire 1993, Herbers 1995, Roemmich
  1983, Molinari & Kirwan 1975, Jagannathan 2021, Puthan 2021, Gill 1984,
  Price 1981). TN-028 gained its 2016-05-03 edition; the 2013 edition is kept
  as `TN-028_…_2013.pdf`. 81 more were copied in under the library's
  filenames: the two `current-meters/` gaps above, and 79 from `TO_FETCH.md`
  into the folder it names. `TO_FETCH.md` is the Hydrographer Bank white
  paper's fetch list, not this repository's (see the next note), so most of
  those 79 went into the white paper's folders, which have since moved; four went into `epsilon-shear/`
  (Nasmyth 1970, Bluteau et al. 2011, Wiles et al. 2006, Lumley & Terray 1983).
  Seven PDFs differ from the library's copy only as a different download of the
  same published version and were left as they are.
- Eight folders that used to sit here — `bottom-boundary-layers/`,
  `drifters-and-mapping/`, `infragravity-and-swell/`,
  `internal-bores-and-intrusions/`, `palau-fleat/`, `remote-sensing-currents/`,
  `topographic-wakes/` and `typhoons-and-upper-ocean/` (154 PDFs; their READMEs
  came with PRs #204-#206 and were removed from git with the move) —
  were created 2026-10-05 by an automated literature search for the
  Hydrographer Bank (Palau) white paper, which took "our local papers
  directory" to mean this one; it also wrote `TO_FETCH.md`. On 2026-10-10 the
  folders and `TO_FETCH.md` were moved to
  `../Hydrographer-Analysis/docs/references/`, where the white paper lives.
- Second pass 2026-10-10: Pat asked for the library's relevant papers, noting
  that some AEM1-G papers are particular to diagnosing that sensor. 52 more were
  copied from `~/Desktop/Papers`, chosen for bearing on what this repository
  computes or on the instruments it processes: 22 into `current-meters/` (EM
  flowmeter weight-function, zero and eddy-current physics; field
  intercomparisons; the Signature1000 reference), 14 into
  `gliders-and-platforms/`, 6 into the new `ctd/`, 10 into the turbulence groups
  (one of them, Salon et al. 2008, into `bottom-boundary-layers/`, which then
  moved to Hydrographer-Analysis with the other white-paper folders). Left in the
  library: whole topics outside that scope (climate, general engineering,
  eddies and drifters, submesoscale dynamics, island wakes, internal tides,
  remote sensing, vehicle autonomy, mixed-layer-depth definitions), older
  editions of TN-051 and TN-054, and papers judged peripheral within these
  groups (e.g. glider reviews and platform-origin papers, the Woodward 1978
  proceedings, Matos 2026, the Miles–Howard Richardson-number papers). Rows were drafted from each paper's text and checked against it.

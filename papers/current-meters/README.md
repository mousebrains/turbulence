# current-meters

Calibration and dynamic response of *velocity* sensors — electromagnetic
current meters in particular. Opened for the AEM1-G in-situ calibration work
(`../../../AEM1-G/`), where the question is not how to estimate ε but whether
the along-body speed feeding it is right.

The subject has its own literature, largely from the nearshore community in the
1980s, and it is a different one from the microstructure groups here: the
sensors are EM rather than acoustic or shear, and the recurring failure modes
are gain error, zero offset, angular response, and electronic filtering in the
wave band.

| Paper | File | Why it matters |
|---|---|---|
| Guza (1988), "Comment on 'Kinematic and dynamic estimates from electromagnetic current meter data' by D. G. Aubrey and J. H. Trowbridge," *JGR* **93**(C2), 1337–1343. [doi:10.1029/JC093iC02p01337](https://doi.org/10.1029/JC093iC02p01337) | `Guza_1988_Comment_EM_Current_Meter_Calibration.pdf` | The central dispute over how badly EM current meters are calibrated, and the origin of the method the AEM1-G work uses. Aubrey & Trowbridge (1985) reported laboratory sensitivity errors up to **45%** with free-stream turbulence; Guza rebuts with colocated pressure–velocity–elevation comparisons against linear theory, finding **u–P and u–η gains of 1.0 ± 0.1** outside the surf zone across EM, MMI and acoustic sensors. Also: §4 shows a single regression across a nonlinear response producing a spurious intercept *and* a distorted slope (Fig. 2) — the 1988 form of the AEM1-G false positive; §6 and Fig. 3 give amplitude reduction **and** phase delay for the MMI electronic filters, negligible below 0.26 Hz. |
| Aubrey & Trowbridge (1985), "Kinematic and dynamic estimates from electromagnetic current meter data," *JGR* **90**(C5), 9137–9146. [doi:10.1029/JC090iC05p09137](https://doi.org/10.1029/JC090iC05p09137) | `Aubrey_Trowbridge_1985_EM_Current_Meter_Response.pdf` | The laboratory study the dispute is about. Tow-tank calibration of Marsh-McBirney EM meters in pure steady, pure oscillatory and **combined steady/oscillatory** flow. Headline results: horizontal cosine response departs systematically from cosine with **intercardinal undersensitivity up to 25%** and a Reynolds-dependent "shoulder" near head-on; combined flows depress steady sensitivity 7–10% and oscillatory sensitivity 9–21% — *the two differ*; sensitivity rises with steady Reynolds number; **numerical offsets up to 4 cm s⁻¹** appear in mixed flow; grid turbulence changed sensitivity by 24% and 45%. Also states that EMCM time constants "arise from analog electrical filter characteristics" and should be removed by inverting the transfer function. |
| Aubrey & Trowbridge (1988), "Reply," *JGR* **93**(C2), 1344–1346. [doi:10.1029/JC093iC02p01344](https://doi.org/10.1029/JC093iC02p01344) | `Aubrey_Trowbridge_1988_Reply_to_Guza.pdf` | Where the dispute lands. Concedes the peak-to-peak sensitivity B′ "is not the one to be applied to field data" and that **"to correct field data for calibration results, both positive and negative velocities are needed"**. The consensus both sides sign: **AST, AT and Guza all report 5–10% differences between manufacturer-supplied gains and predeployment calibrations**, and "some researchers still persist in using manufacturer-supplied gains." Also cautions that AT's 5% field agreement holds for sensors **≥1 m above the bottom, outside the wave boundary layer**. |
| Aubrey, Spencer & Trowbridge (1984), *Dynamic Response of Electromagnetic Current Meters*, WHOI-84-20 / CRC-84-3, 150 pp. [NOAA repository 39165](https://repository.library.noaa.gov/view/noaa/39165) | `Aubrey_Spencer_Trowbridge_1984_Dynamic_Response_EMCM.pdf` | The full laboratory report behind AT (1985) — tables, error budget, methodology and recommendations that the paper only summarises. Its literature review names **Dibble & Sollitt (1981)**, who characterised current meters as low-order linear differential systems to extract time constants and phase — the antecedent for any "first-order response" claim. Table 6 lists instrument time constant and zero drift as standard calibration error sources. Its recommendations prefigure most of a modern calibration spec: pre- **and** post-calibration, ≤1–2 month deployments against fouling, electrode scrubbing for surface wetting, in-situ zero checks, mounting ≥3 probe diameters from the bed *or any material of different electrical conductivity*, Reynolds-dependent angular-response correction, and — still open — "investigate the response of EM sensors to broad-band forcing." **U.S. Government work; reproduction permitted.** |

## Known gaps in this group

The 1984 report, the 1985 paper, the 1988 comment and the 1988 reply are all
held — the complete Aubrey--Trowbridge / Guza exchange. Everything
below is cited within them and is still wanted for the AEM1-G manuscript.

| Wanted | Why |
|---|---|
| Dibble & Sollitt (1981), "Frequency response characterization of current meters," *OCEANS 81*, 250–256, [doi:10.1109/OCEANS.1981.1151454](https://doi.org/10.1109/OCEANS.1981.1151454). | **Abstract read, full text paywalled (IEEE Xplore); 7 pp.** The direct antecedent of the AEM1-G τ measurement. Per the abstract: current meters modelled as **2nd-order** differential systems — static sensitivity from constant-speed tows, **time constant from a rapid deceleration ramp test**, natural frequency and damping from **pendulum oscillator tests** — yielding corrections for "velocity amplitude attenuation and phase shift." **An electromagnetic meter is one of the two examined.** Worth obtaining for the test protocol, but the abstract already fixes the priority. |
| Cunningham, Guza & Lowe (1979), "Dynamic calibration of electromagnetic flow meters," *Oceans '79*, 298–301. | Broad-banded dynamic calibration; Guza cites it for ~10% apparent-gain variation across a test spectrum. |
| Guza & Thornton (1980), "Local and shoaled comparisons of sea surface elevations, pressures, and velocities," *JGR* **85**(C3), 1524–1530. | The colocated-sensor method, and a spectrally measured 5% gain change over a factor-of-5 in peak velocity. |
| Bowden & White (1966), *Geophys. J. R. Astron. Soc.* **12**, 33–54. | The origin of using pressure and linear theory to check a current meter. |

## Cross-references

- [`../epsilon-shear/`](../epsilon-shear/README.md) — McMillan et al. (2016) put a
  MicroRider, an AD2CP **and** a JFE Advantech EM current meter on one bottom
  frame in a tidal channel, and report the EM meter "in general agreement" with
  the ADV without quantifying it. It is the closest published precedent to the
  AEM1-G deployment and is filed there as an ε paper.
- [`../rockland-technical-notes/`](../rockland-technical-notes/README.md) —
  **TN-041** is the AEM1-G-on-MicroRider note: coil, graphite electrodes,
  synchronous demodulation, and the fact that JAC calibrates the sensor on a
  simulated MicroRider front end. **TN-039** and **TN-048** carry the `U_EM`
  channel definition.
- `../../../AEM1-G/docs/datasheets/README.md` — the JFE vendor datasheet:
  ±0.5 cm s⁻¹ or ±2 %RD, 0–500 cm s⁻¹, 10 Hz output, and **no response time,
  zero drift, angular response or conductivity dependence specified**. The PDF
  itself is held in the AEM1-G repository (untracked there, like these papers);
  that index carries every number off the sheet.

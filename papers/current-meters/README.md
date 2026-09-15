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

The second table is the flowmeter's own physics (what sets the gain and the
zero, and which of them depends on the water's conductivity). It is the
literature behind the still-water bench test in `scripts/em_bench_zero.py`.
For how a `U_EM` over-read shows up in glider flight, see
[`../gliders-and-platforms/`](../gliders-and-platforms/README.md).

## The Aubrey–Trowbridge / Guza exchange

| Paper | File | Why it matters |
|---|---|---|
| Guza (1988), "Comment on 'Kinematic and dynamic estimates from electromagnetic current meter data' by D. G. Aubrey and J. H. Trowbridge," *JGR* **93**(C2), 1337–1343. [doi:10.1029/JC093iC02p01337](https://doi.org/10.1029/JC093iC02p01337) | `Guza_1988_Comment_EM_Current_Meter_Calibration.pdf` | The central dispute over how badly EM current meters are calibrated, and the origin of the method the AEM1-G work uses. Aubrey & Trowbridge (1985) reported laboratory sensitivity errors up to **45%** with free-stream turbulence; Guza rebuts with colocated pressure–velocity–elevation comparisons against linear theory, finding **u–P and u–η gains of 1.0 ± 0.1** outside the surf zone across EM, MMI and acoustic sensors. Also: §4 shows a single regression across a nonlinear response producing a spurious intercept *and* a distorted slope (Fig. 2) — the 1988 form of the AEM1-G false positive; §6 and Fig. 3 give amplitude reduction **and** phase delay for the MMI electronic filters, negligible below 0.26 Hz. |
| Aubrey & Trowbridge (1985), "Kinematic and dynamic estimates from electromagnetic current meter data," *JGR* **90**(C5), 9137–9146. [doi:10.1029/JC090iC05p09137](https://doi.org/10.1029/JC090iC05p09137) | `Aubrey_Trowbridge_1985_EM_Current_Meter_Response.pdf` | The laboratory study the dispute is about. Tow-tank calibration of Marsh-McBirney EM meters in pure steady, pure oscillatory and **combined steady/oscillatory** flow. Headline results: horizontal cosine response departs systematically from cosine with **intercardinal undersensitivity up to 25%** and a Reynolds-dependent "shoulder" near head-on; combined flows depress steady sensitivity 7–10% and oscillatory sensitivity 9–21% — *the two differ*; sensitivity rises with steady Reynolds number; **numerical offsets up to 4 cm s⁻¹** appear in mixed flow; grid turbulence changed sensitivity by 24% and 45%. Also states that EMCM time constants "arise from analog electrical filter characteristics" and should be removed by inverting the transfer function. |
| Aubrey & Trowbridge (1988), "Reply," *JGR* **93**(C2), 1344–1346. [doi:10.1029/JC093iC02p01344](https://doi.org/10.1029/JC093iC02p01344) | `Aubrey_Trowbridge_1988_Reply_to_Guza.pdf` | Where the dispute lands. Concedes the peak-to-peak sensitivity B′ "is not the one to be applied to field data" and that **"to correct field data for calibration results, both positive and negative velocities are needed"**. The consensus both sides sign: **AST, AT and Guza all report 5–10% differences between manufacturer-supplied gains and predeployment calibrations**, and "some researchers still persist in using manufacturer-supplied gains." Also cautions that AT's 5% field agreement holds for sensors **≥1 m above the bottom, outside the wave boundary layer**. |
| Aubrey, Spencer & Trowbridge (1984), *Dynamic Response of Electromagnetic Current Meters*, WHOI-84-20 / CRC-84-3, 150 pp. [NOAA repository 39165](https://repository.library.noaa.gov/view/noaa/39165) | `Aubrey_Spencer_Trowbridge_1984_Dynamic_Response_EMCM.pdf` | The full laboratory report behind AT (1985) — tables, error budget, methodology and recommendations that the paper only summarises. Its literature review names **Dibble & Sollitt (1981)**, who characterised current meters as low-order linear differential systems to extract time constants and phase — the antecedent for any "first-order response" claim. Table 6 lists instrument time constant and zero drift as standard calibration error sources. Its recommendations prefigure most of a modern calibration spec: pre- **and** post-calibration, ≤1–2 month deployments against fouling, electrode scrubbing for surface wetting, in-situ zero checks, mounting ≥3 probe diameters from the bed *or any material of different electrical conductivity*, Reynolds-dependent angular-response correction, and — still open — "investigate the response of EM sensors to broad-band forcing." **U.S. Government work; reproduction permitted.** |

## Gain, zero, and conductivity: calibration practice and flowmeter physics

| Paper | File | Why it matters |
|---|---|---|
| Guza, Clifton & Rezvani (1988), "Field intercomparisons of electromagnetic current meters," *JGR* **93**(C8), 9302–9314. [doi:10.1029/JC093iC08p09302](https://doi.org/10.1029/JC093iC08p09302) | `Guza_Clifton_Rezvani_1988_Field_Intercomparisons_EM_Current_Meters.pdf` | The field data behind Guza's 1988 comment: colocated biaxial EM meters with spherical and open-frame heads. The rms velocity ratio between any two meters was within 1.0 ± 0.07, consistent with ~±5% gain calibration. Offsets proved sensitive to "subtle differences in the electromagnetic environment". Also states, citing Olson (1972) and giving no numbers, that the seawater/tap-water conductivity difference does not significantly change gain or offset. The same sentence goes on to say this "is not necessarily true for multiple instrument arrays", because interactions between sensors depend on conductivity, "our calibrations are done in seawater". So it is a prior on the fresh-versus-seawater question the bench test measures for one unit, and a caution for any multi-sensor mount. |
| Griffiths & Collar (1980), "Some comparative studies on electromagnetic sensor heads in laminar and near-turbulent flows in a towing tank," *OCEANS '80*, 323–329. [doi:10.1109/OCEANS.1980.1151375](https://doi.org/10.1109/OCEANS.1980.1151375) | `Griffiths_Collar_1980_EM_Sensor_Heads_Towing_Tank.pdf` | A tow-tank comparison of disc, annulus and sphere heads that puts numbers on one zero mechanism. With a 200 Ω source, a 12 V coil drive and 100 µV per m/s, a 20% drop in the 10⁹ Ω coil-to-seawater insulation resistance shifts the zero by 1 cm/s. A 2000 h fresh-water test moved the annulus zero by <0.7 cm/s; if that was insulation drift, sea water "should be an order lower due to reduced source resistance". Temperature moved the zero by <1 cm/s between 1 and 25 °C. |
| Collar & Griffiths (2001), "Single point current meters," in *Encyclopedia of Ocean Sciences*, Academic Press, 2796–2803. [doi:10.1006/rwos.2001.0326](https://doi.org/10.1006/rwos.2001.0326) | `Collar_Griffiths_2001_Single_Point_Current_Meters.pdf` | Review entry. EM zero stability "within a few mm s⁻¹ over many months" should be achievable with modern electronics; "good practice is represented by regular calibration checks in water"; and "no amount of simple rectilinear calibration in steady flow conditions can reveal the instrument response" to broadband motion in the sea. (The PDF held is the 2001 first edition; the second-edition DOI, 10.1016/B978-012374473-9.00326-X, paginates it 428–435.) |
| Cushing (1976), "Electromagnetic water current meter," *OCEANS '76*, 663–679. [doi:10.1109/OCEANS.1976.1154309](https://doi.org/10.1109/OCEANS.1976.1154309) | `Cushing_1976_Electromagnetic_Water_Current_Meter.pdf` | An EM water current meter's theory and tests, from Cushing Engineering. A nearby flow boundary "degenerates sensor sensitivity": the effect is neutralized by a conductive boundary and doubled by an insulating one. Separately, it warns of "the hazard of appending a water current meter to a sizable subsurface housing". That hazard is hydrodynamic, not electrical: the housing's circulation and lift distort the flow at the sensor, so tilt response is no longer sinusoidal. Cushing's remedy is to build the meter into the middle of the housing. A MicroRider nose mount is the appended case. |
| Cushing (1976), U.S. Patent Re. 28,989, "Electromagnetic water current meter," reissued 5 October 1976. No DOI. | `Cushing_1976_USRE28989_EM_Current_Meter_Patent.pdf` | Known meters are sensitive to "the conductivity of the metered water whereby the meter factor is not constant in different kinds of water". The zero is its example: it "moves around" with conductivity and changes with electrochemical aging at the electrodes. Leakage of one part in 10⁷–10⁸ of the magnet voltage into the signal circuit is enough to move it. It gives no source-resistance divider and no 1/σ form; Griffiths & Collar (1980) do. |
| Hemp (1995), "Theory of a simple electromagnetic velocity probe with prediction of the effect on sensitivity of a nearby wall," *Meas. Sci. Technol.* **6**, 376–382. [doi:10.1088/0957-0233/6/4/006](https://doi.org/10.1088/0957-0233/6/4/006) | `Hemp_1995_EM_Velocity_Probe_Nearby_Wall.pdf` | Gain of a small wafer-shaped EM probe as a function of distance to a plane wall, conducting or insulating, magnetic or not (the insulating non-magnetic case numerically). The closest published model for an EM head mounted near an instrument housing. |
| Bevir (1970), "The theory of induced voltage electromagnetic flowmeters," *JFM* **43**, 577–590. [doi:10.1017/S0022112070002586](https://doi.org/10.1017/S0022112070002586) | `Bevir_1970_Theory_Induced_Voltage_EM_Flowmeters.pdf` | Weight-vector theory: the signal is ∫ v·W dτ with W = B × j, where the virtual current j depends on electrode shape and wall conditions "(and by the conductivity distribution if non-uniform)". A uniform conductivity therefore drops out, under the stated assumption that no current is drawn from the electrodes. Also proves that flowmeters with point electrodes "cannot be made ideal": the reading depends on the flow pattern in the sensing volume, not the mean velocity alone. The AEM1-G senses with small graphite electrodes (TN-041). |
| Bevir (1972), "The effect of conducting pipe connections and surrounding liquid on the sensitivity of electromagnetic flowmeters," *J. Phys. D* **5**, 717–729. [doi:10.1088/0022-3727/5/4/311](https://doi.org/10.1088/0022-3727/5/4/311) | `Bevir_1972_Conducting_Connections_Surrounding_Liquid.pdf` | A conductor near the electrodes changes the **gain** through the virtual current. In the limiting case of surrounding liquid, half the virtual current flows outside and the signal halves. Also notes that "there is a liquid conductivity below which it may be difficult to measure the signal", without giving a value. |
| Baker (2016), "Electromagnetic flowmeters," ch. 12 in *Flow Measurement Handbook*, 2nd ed., Cambridge University Press, 362–407. [doi:10.1017/CBO9781107054141.013](https://doi.org/10.1017/CBO9781107054141.013) | `Baker_2016_Flow_Measurement_Handbook_Ch12_EM_Flowmeters.pdf` | Engineering practice. Sensitivity is conductivity-independent provided the electrode output resistance is "two or more orders of magnitude" below the amplifier input resistance: ~25 Ω in sea water against ~10 kΩ in tap water for a 1 cm electrode. Even so, "severe variations in conductivity may cause zero errors in AC-type magnetic flowmeters". Square-wave excitation lets quadrature signals decay before sampling, and the baseline still drifts "due to electrochemical and other effects". |
| Maalouf (2006), "The derivation and validation of the practical operating equation for electromagnetic flowmeters: Case of having an electrolytic conductor flowing through," *IEEE Sensors J.* **6**(1), 89–96. [doi:10.1109/JSEN.2005.860360](https://doi.org/10.1109/JSEN.2005.860360) | `Maalouf_2006_Practical_Operating_Equation.pdf` | For an electrolyte the flow signal "is independent of the conductivity of the medium". The quotable statement that the **gain** should not change between fresh water and seawater. |
| Maalouf (2006a), "A validated model for the electromagnetic flowmeter's measuring cell: Case of having an electrolytic conductor flowing through," *IEEE Sensors J.* **6**(3), 623–630. [doi:10.1109/JSEN.2006.874461](https://doi.org/10.1109/JSEN.2006.874461) | `Maalouf_2006a_Measuring_Cell_Electrode_Impedance.pdf` | The electrode–electrolyte interface as a measured impedance. Two electrodes of the same metal in the same electrolyte show "a small residual potential difference … which is not always stable". This is the electrode side of the zero. |
| Maalouf (2006b), "A validated model for the zero drift due to transformer signals in electromagnetic flowmeters operating with electrolytic conductors," *IEEE Sensors J.* **6**(6), 1502–1510. [doi:10.1109/JSEN.2006.884176](https://doi.org/10.1109/JSEN.2006.884176) | `Maalouf_2006b_Zero_Drift_Transformer_Signals.pdf` | The zero from flux linkage between the coils and the electrode-lead loop depends on excitation frequency. For electrolytes, conductivity and permittivity are expected to have "no effect" on it, whereas for dielectric liquids they do. This is the conductivity-*independent* part of the zero. |
| Maalouf (2007), "A validated model for the zero drift due to eddy currents in electromagnetic flowmeters operating with electrolytic conductors," *IEEE Sensors J.* **7**(11), 1497–1505. [doi:10.1109/JSEN.2007.907562](https://doi.org/10.1109/JSEN.2007.907562) | `Maalouf_2007_Zero_Drift_Eddy_Currents.pdf` | The eddy-current zero, and how it depends on the liquid's properties and the excitation frequency. The route by which a zero can depend on the water. |
| Al-Rabeh, Baker & Hemp (1978), "Induction flow-measurement theory for poorly conducting fluids," *Proc. R. Soc. Lond. A* **361**, 93–107. [doi:10.1098/rspa.1978.0093](https://doi.org/10.1098/rspa.1978.0093) | `AlRabeh_Baker_Hemp_1978_Poorly_Conducting_Fluids.pdf` | Where ordinary EM flowmeter theory stops holding. Tap water (~10⁻² S/m) sits in the "moderately conducting" category the ordinary theory covers, while the poorly conducting case is petrol-like, ~10⁻¹⁵ S/m. A tap-water can on the bench is not a low-conductivity experiment in this sense. |

**What the second table says about the bench model.** `em_bench_zero.py` fits
c(σ) = A + B/σ. The literature supports that split and says what could break
it:

- **Gain** is conductivity-independent (Bevir 1970, Maalouf 2006), provided the
  source resistance stays well below the input resistance (Baker 2016).
- **A zero ∝ 1/σ** has a published mechanism: leakage through the coil
  insulation divided against the electrode source resistance (Griffiths &
  Collar 1980). That is not the mechanism `em_bench_zero.py` names for B/σ
  (amplifier bias current × source impedance), though both scale with source
  resistance and the fit cannot tell them apart. Cushing's patent reports
  conductivity dependence of both meter factor and zero, with no functional
  form.
- **A σ-independent zero** also has one: flux-linkage quadrature leaking through
  the detector (Maalouf 2006b). That belongs in A.
- **The eddy-current and electrode-interface routes** (Maalouf 2007, 2006a)
  depend on the liquid without a stated 1/σ form. The third can tests for them.

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

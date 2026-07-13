# Choosing dissipation and FFT lengths (and when to make them adaptive)

Guidance for `epsilon.fft_length` / `epsilon.diss_length` (and the chi
analogs), grounded in the ATOMIX best-practices literature and a controlled
2-s vs 4-s twin processing of the ARCTERX-2025-Interior VMP dataset. The
short version: **no single fixed length is right across a profile that spans
energetic and quiet water — the optimum is set by the larger of two
physically distinct constraints, both of which vary along the cast.**

## The constraints

**1. Spectral coverage — FFT length (ε-dependent).** The shear spectrum's
peak sits near `k ≈ 0.02/L_K` cpm (Lueck 2022b), i.e. it moves to *lower*
wavenumber as ε falls (`L_K = (ν³/ε)^{1/4}`). Lueck et al. (2024, §"FFT
length") make the requirement explicit and ε-tiered: the lowest resolved
wavenumber `k_l = 1/(t_fft·W)` should be

| ε [W/kg] | k_l needed | FFT span `t_fft·W` |
|---|---|---|
| ≲ 1e-9 (low) | ≈ 0.5 cpm | ≈ 2 m |
| ≲ 1e-7 (moderate) | ≈ 1 cpm | ≈ 1 m |
| ≳ 1e-7 (high) | ≈ 2 cpm | ≈ 0.5 m |

with a **hard ceiling**: the FFT span "should not exceed the length of a
free profiler because the profiler will be advected by eddies comparable to
and larger than the profiler" (Lueck et al. 2024). So k_l cannot honestly go
below ~1/(profiler length) no matter how quiet the water.

For the **VMP-250 (body length ≈ 1 m)** the ceiling means k_l ≥ ~1 cpm —
the low-ε tier above is unreachable, and the platform's full-fidelity floor
is where the spectral peak stays resolvable: **ε ≳ 1–3e-10 W/kg** (below
that, estimates rest on the viscous rolloff alone). The practical VMP-250
optimum is `fft_length: 512` (1 s ≈ 0.75 m at fall speed, just under the
ceiling); the historical 256-sample FFT leaves a factor of ~2 of legitimate
low-wavenumber coverage unused. A MicroRider inherits its *host's* length
(~2 m on a glider), so k_l ≈ 0.5 cpm — the tier the deep ocean needs — is
reachable there.

**2. Statistical reliability — dissipation length (also ε-dependent).**
Lueck (2022a, Eq. 11): `σ²_lnε = 5.5/(1+(L̂_f/4)^{7/9})` with
`L̂_f = (l_ε/L_K)·V_f^{3/4}` — the window length **in Kolmogorov lengths**
(shear samples are independent only when separated by > 4 L_K), derated by
the resolved-variance fraction. Holding σ_lnε ≤ ~0.3 needs
`l_ε ≳ 800·L_K`:

| ε [W/kg] | L_K | l_ε for σ_lnε ≤ 0.3 |
|---|---|---|
| 1e-7 | 2.2 mm | 1.7 m |
| 1e-8 | 3.8 mm | 3.1 m |
| 1e-9 | 6.8 mm | 5.5 m |
| 1e-10 | 12 mm | 9.7 m |
| 1e-11 | 22 mm | 17 m |

ATOMIX floor/format constraints: ≥ 3 half-overlapping FFT segments per
estimate (N_f = 3 minimum; N_f ≈ 7 in the ATOMIX Faroe Bank example — 8-s
records, 2-s FFTs), N_f ≥ ~10 for near-normal spectral statistics, and
Goodman coherency is only *well*-estimated for N_f ≳ 19 (Lueck 2022b) —
below that the bias correction `1/(1−1.02·N_A/N_f)` (implemented in
`scor160.goodman`) carries the load.

**3. Ceilings — stationarity and resolution.** Longer is not free: ε is
lognormal and patchy, so long windows average across patches (mean-of-ε
weighting; Whalen 2021 shows order-10 discrepancies from mismatched
averaging scales in strong turbulence), the estimate's vertical resolution
is `l_ε` itself, and Lueck (2022a) notes pdf departures by `L̂ ~ 10⁴`.

## Evidence from the ARCTERX 2-s vs 4-s twin runs

The same 349 casts were processed with `diss_length` = 1024 (2 s ≈ 1.5 m)
and 2048 (4 s ≈ 3 m), `fft_length` = 256 (0.5 s) in both:

- **The Lueck model verifies quantitatively**: doubling the window reduced
  the median per-window `epsilonLnSigma` by ×0.76 in every ε decade —
  exactly the predicted `(L̂)^{-7/18}` scaling.
- **A low-ε window-length bias**: pairing the runs window-by-window, the
  2-s estimates run ~**1.2–1.4× lower** (estimator-sensitive) than the 4-s
  estimates for ε ∈ [1e-10, 1e-9], converging to parity above 1e-8. The exact
  ratio depends on the ε patch estimator (arithmetic vs log/geometric mean),
  so it is not a single fixed factor. Both runs share the
  same 0.5-s FFT, so this is NOT the unresolved-peak effect — it is a
  dissipation-window effect at fixed FFT: lognormal patch-averaging
  (longer windows arithmetic-mean across patches and read higher where
  turbulence is intermittent) and/or Goodman coherency dof (N_f = 7 vs 15;
  low dof over-removes at low signal-to-vibration ratio, biasing the short
  window low).
- **The FFT-span effect, isolated** (`perturb.4f1.yaml`: 4-s windows, 1-s
  FFT, vs the same windows at 0.5-s FFT): the 1-s-FFT ε reads ×1.31 higher
  at ε ∈ [1e-10, 1e-9] (MAD 0.147 — tightly systematic), ×1.18 at
  [1e-9, 1e-8], converging to ×1.02 above 1e-7 — the unresolved-peak
  signature, now measured in our own data, with `epsilonLnSigma` unchanged
  (statistics depend on the window, not the FFT). **Combined**, the
  historical 2-s/0.5-s configuration underestimates ε by up to ×1.8 in the
  quietest decade; any low-ε science from those products (thermocline
  Re_b, K_rho, Gamma) inherits that bias.
- **Overturn containment**: at 2 s, ~80% of mixed-layer windows had
  Thorpe overturns clipped by the window (`perturb-plot gamma-scaling`
  edge-truncation diagnostics); 4 s roughly halves the cap deficit.

## Adaptive lengths: what theory supports

Two natural "turbulence-aware" controllers exist, and they rule different
regimes:

- **Eddy containment** wants `l_ε ≥ N_e·L_O` (N_e ≈ 5–10 energy-containing
  eddies; L_T ≈ L_O by Dillon's 0.8) — windows shorter than the overturns
  sample single correlated structures, not statistics.
- **Statistics/coverage** wants `l_ε ≳ 800·L_K` and the FFT tier above.

They are linked by the identity **`L_O/L_K = Re_b^{3/4}`**, so an
L_O-proportional window carries `L̂ = N_e·Re_b^{3/4}` — its statistical
quality *collapses* in quiet stratified water (at ε = 1e-10, N² = 1e-5:
L_O ≈ 6 cm; a 5·L_O window gives σ_lnε ≈ 0.9). Setting the two equal gives
the crossover

  `Re_b* = (800/N_e)^{4/3} ≈ 350–900`

- **Re_b ≳ 900** (mixed layer, wakes): eddy containment dominates —
  L_T/L_O-adaptive windows are the theoretically right choice.
- **Re_b ≲ 350** (thermocline, deep MicroRider water): statistics dominate —
  the window must scale with `L_K ∝ ε^{-1/4}`, i.e. grow as ε falls.

So the defensible adaptive rule is the **max of both, clamped**:

  `l_ε = clamp( max(N_e·L_O, C·L_K), l_min, l_max )`,
  FFT span `= clamp( tier(ε), 0.5 m, profiler length )`

L_T is the attractive *operational* controller for the first term because it
is measurable without ε (no circularity): grow the window until the Thorpe
edge-truncation flag clears — the smallest window that contains its own
overturns (`processing.thorpe` already computes the flag). The `L_K` term
needs an ε first guess (previous window, prior pass, or noise-floor prior).

## MicroRider (> 1000 m range) implications

A deep MR cast spends most of its range at ε ~ 1e-11–1e-10 where the tables
demand FFT spans of ~2 m (at the platform-length ceiling) and dissipation
lengths of 10–17 m for σ_lnε ≤ 0.3 — versus ~0.5 m / 1–2 m optima in the
upper ocean. A fixed VMP-style 0.5-s/2-s configuration would be biased low
(unresolved peak) *and* noisy over most of the profile; a fixed deep-water
configuration would smear the energetic upper ocean. Adaptivity (or at
minimum a depth schedule) is not cosmetic for the MR — it is required.
Also note: MR platforms are slow (0.3–0.5 m/s), so lengths must be
configured in **spatial units** and converted per-window by measured speed
(the Lueck statistics are formulated spatially; ATOMIX already converts by
per-spectrum mean speed).

## Fall-speed profiles interact with window units

Tethered VMPs decelerate with depth (body compression is less than the
potential-density increase, and the ~0.98-specific-gravity line
increasingly retards the fall). This is a hardware win exactly where ε is
low: at fixed *spatial* spans, the slow deep fall provides more raw
samples per meter (despiking and Goodman statistics), maps the Batchelor
rolloff into the FP07's honest response band (k_B·W ≈ 35 Hz at 1 m/s but
~21 Hz at 0.6 m/s for ε = 1e-10 — χ improves with depth), and moves the
whole resolvable shear band to low, quiet frequencies. **But the win is
conditional on spatial-unit windows**: with `fft_length`/`diss_length`
fixed in samples, deceleration *shrinks* the spatial spans at depth —
raising k_l and lowering L̂ precisely where both need to grow. The
near-neutral line keeps the profiler quasi-free, so the body-length
ceiling applies as stated.

## Recommendations for this pipeline

1. **Now (shipped)**: the pipeline default is `fft_sec: 1.0` with
   `diss_sec: null` (= 4 s), specified as **durations** and converted per
   instrument via its sampling rate — so a 512-Hz VMP-250 gets 512-sample
   FFTs and a 2-kHz coastal unit gets 2048 automatically, and the 1-s
   choice sits at the VMP-250's 1-m body ceiling. Explicit sample keys
   (`fft_length` etc.) remain as expert overrides and win (legacy configs
   keep bit-identical signatures). Treat historical 2-s/0.5-s ε below
   1e-9 as biased low (up to ×1.8 at 1e-10). Multiple configurations
   coexist under one `output_root` (config signatures).
2. **Near term (small feature)**: spatial-unit windows (meters, converted
   per window by measured speed — the remaining step beyond durations,
   needed for decelerating VMPs and slow MR platforms) plus an optional
   depth-banded schedule — static per band, no algorithmic risk.
3. **Longer term (prototype)**: two-pass ε-adaptive lengths per the
   max-rule above, with the L_T fixed-point (grow until edge-truncation
   clears) supplying the eddy term in pass 1. Validate against the twin-run
   pattern: the diagnostic of success is the disappearance of the paired
   low-ε bias without loss of upper-ocean resolution.
4. **Always**: report per-window `epsilonLnSigma` (we do), the resolved
   variance fraction, and window length in the products, so downstream
   users can filter on statistical quality rather than trust a global
   choice.

## References

- Lueck, R.G., 2022a: Statistics Part I. JTECH 39, 1259–1271.
  https://doi.org/10.1175/JTECH-D-21-0051.1
- Lueck, R.G., 2022b: Statistics Part II. JTECH 39, 1273–1282.
  https://doi.org/10.1175/JTECH-D-21-0050.1
- Lueck, R.G., et al., 2024: Best practices recommendations for estimating
  dissipation rates from shear probes. Front. Mar. Sci. 11, 1334327.
  https://doi.org/10.3389/fmars.2024.1334327
- Fer, I., et al., 2024: ATOMIX benchmark datasets. Sci. Data 11, 518.
  https://doi.org/10.1038/s41597-024-03323-y
- Whalen, C.B., 2021: Best practices for comparing ocean turbulence
  measurements across spatiotemporal scales. JTECH 38, 837–841.
  https://doi.org/10.1175/JTECH-D-20-0175.1
- Dillon, T.M., 1982: Vertical overturns. JGR 87, 9601–9613.
  https://doi.org/10.1029/JC087iC12p09601

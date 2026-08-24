# The glider CTD's thermistor response time

**Result: τ ≈ 0.7 s**, from two independent estimates on osu685 that bracket
0.62–0.87 s, agreeing with the 0.5–0.75 s inferred independently across many
deployments and hundreds of millions of samples.

**Not 2.7 s.** That number is the *delay* between the CTD's temperature and its
own pressure, and it is a different quantity — see below.

---

## Why the distinction matters

`PairConfig.kernel_tau` is a **model**, not a stored measurement: a single pole
applied to the FP07 to slow it to the reference's bandwidth so the regression
compares like with like. Set it wrong and the fitted `beta_1` is wrong, because
un-matched bandwidth in the regressor is textbook errors-in-variables.

The measured temperature-vs-pressure gap on osu685 was **+2.7 s** (total
temperature lag +7.4 s minus the pressure offset +4.5 s). That gap has two
physically different parts:

| part | nature | does the lag search remove it? |
|---|---|---|
| plumbing transit — pumped flow from intake to thermistor | **pure delay**: shifts, does not attenuate | **yes** |
| thermistor thermal response | **pole**: shifts *and* attenuates | **no** |

So the 2.7 s is dominated by transit, which the lag search already handles.
Feeding it back in as a pole over-filters the probe and re-attenuates `beta_1`
— destroying the thing the bandwidth match exists to protect.

## Method A — residual sweep

Sweep the pole applied to the FP07, re-estimate the lag at each value (the pole
changes group delay), refit, and record the fit residual. A pole *mismatch* is
not a delay, so it cannot be absorbed and should leave a minimum near truth.

| τ [s] | lag [s] | rms [mK] |
|---|---|---|
| 0.05 | 7.78 | 5.9339 |
| 0.45 | 7.41 | 5.9338 |
| **0.65** | 7.23 | **5.9169** |
| 0.85 | 7.07 | 5.9200 |
| 1.50 | 6.58 | 6.2156 |
| 2.60 | 5.88 | 7.2407 |
| 3.80 | 5.24 | 8.7276 |

Minimum at **0.62–0.71 s** across two runs (parabolic refinement).

**Read the asymmetry, not just the minimum.** Across 0.05–0.85 s the residual
moves only **0.4 %** — this criterion barely constrains the lower bound,
because under-filtering leaves FP07 variance that is small against the
thermocline signal. It constrains the *upper* bound sharply: at 2.7 s the
residual is **25 % worse**, and over-filtering destroys real signal. So this
method says "certainly not 2.7, probably around 0.65" and should not be quoted
to two decimals.

## Method B — transfer function

Block-average the FP07 onto the CTD's own 1 Hz timestamps (a boxcar whose sinc
is known and divided out), then fit

```
|H(f)| = 1 / sqrt(1 + (2πfτ)²)
```

over the band where the two sensors are still coherent.

Result: **τ = 0.87 s** over 0.023–0.230 Hz, 49 bins with coherence > 0.5.

Coherence falls from 1.00 at 0.004 Hz to ~0.70 by 0.14 Hz, and `|H|` from 1.01
to ~0.77 over the same span.

**Expect this to be biased high**, and it is. The two sensors sit ~1 m apart, so
they sample different water below that scale: at *w* ≈ 0.3 m/s, 1 m is ≈ 3 s
≈ 0.3 Hz — exactly where a 0.7 s pole rolls off. Spatial decorrelation and the
pole are not separable in this band, and decorrelation looks like extra
roll-off. 0.87 s is therefore an upper bound rather than an estimate.

## The sampling limitation

The GPCTD's fastest sampling is **1 Hz**, so τ ≈ 0.5–0.75 s is **sub-sample**.
Its corner, `1/(2πτ)` = 0.21–0.32 Hz, does sit below the 0.5 Hz Nyquist, so the
roll-off is partially in band — but only just, which is why neither method
above pins it tightly.

**Use `sci_ctd41cp_timestamp`, never `sci_m_present_time`.** The science
computer's clock smears the signal at exactly the sub-sample scale being
measured, and no amount of averaging recovers it. All the numbers above are on
the CTD's own timestamps.

## Naming: what the masterdata does and does not tell you

TWR's Slocum masterdata uses the sensor name **`SBE41CP`** for *both* the older
unpumped CTDs and the pumped **GPCTD**. So `sci_ctd41cp_*` variable names
**cannot** identify which instrument is fitted, and neither this code nor its
documentation should infer one. Everything here says "the glider CTD".

Newer gliders carry **RBR** CTDs, which *are* identifiable — they appear under
different masterdata names. Practically that means a config keyed on
`sci_ctd41cp_timestamp` will simply not find its variable on an RBR glider,
which is a loud config failure rather than a silent wrong answer.

## `kernel_tau` is a property of the CTD model, not of the deployment

The response time belongs to the **sensor design** --- geometry, glass bead,
flow past it --- not to the individual unit or the mission. Across a large body
of GPCTD data (hundreds of millions of samples, PW) the value is close to
instrument-independent, and the 0.62--0.71 s measured here on osu685 sits
squarely in it.

So `kernel_tau` is a constant to be *looked up* per CTD model, not a knob to be
refitted per deployment. 0.7 s for a Seabird GPCTD. Refit it only if you have
reason to believe the sensor is not what you think it is.

What *does* need checking per deployment is **which CTD the glider carries**,
and that is not free: TWR's masterdata uses `SBE41CP` for both the older
unpumped CTDs and the pumped GPCTD, so the sensor name alone does not
distinguish them (see above). An unpumped CTD's response is slower and
flow-dependent, and 0.7 s does not transfer to it.

## The asymmetry that makes this work

Worth stating plainly, because the two halves of the calibration behave in
opposite ways:

| | FP07 (the thing being calibrated) | GPCTD (the thing calibrating it) |
|---|---|---|
| coefficients | **sensor-specific** --- must be fit per probe | model-consistent |
| what it reports | needs the in-situ fit to be trusted | units agree closely with each other |
| response time | --- | ~0.7 s, near enough constant |

On osu685 both FP07 probes shipped with *identical* factory nominal
coefficients, and the in-situ fit was worth -2.66 K and -1.62 K on them
respectively --- two probes, one nominal, two very different truths. That is
what "sensor-specific" means here, and why the fit is per channel and never
transferred.

The reference is the other way round. GPCTDs agree closely enough with each
other, in both what they report and how fast they report it, that the reference
can be taken as given rather than itself calibrated per unit. If it were not
so, this method would not work: a reference carrying its own per-unit offset
would simply transfer that offset into the FP07 coefficients, and the fit would
look every bit as clean while being wrong by the reference's error.

So: refit the FP07 for every probe. Look up the GPCTD's numbers once.

## Still to do

Nothing blocking. The Rutgers bidirectional data would still be a *useful*
check --- dives make `w` change sign, which separates the pole from the spatial
decorrelation that limits Method B here, and enables the dive/climb hysteresis
criterion a climb-only deployment cannot provide --- but it is a confirmation
of a known constant, not a measurement the pipeline is waiting on.

Reproduce with `scripts/fp07cal_ctd_tau.py`.

# Is the FP07–CTD offset a time shift or a depth shift?

**Short answer: a depth shift, and pairing on measured pressure is the better
formulation — but osu685 cannot demonstrate it, because it recorded climbs
only. Bidirectional data settles it.**

## The argument for depth

The MR's pressure sensor sits 1–2 cm from the FP07; the CTD's pressure sensor
is colocated with its thermistor. So **each instrument measures the depth of its
own temperature sample.** In a steady, horizontally homogeneous column
`T = T(z)`, matching an FP07 sample to the CTD sample at the *same measured
pressure* makes the sensor separation cancel identically — longitudinal and
perpendicular alike — without modelling it at all. Clock skew cancels too,
because time never enters the pairing.

That is strictly better than what the tool does today (pair in time at a fitted
lag, then absorb the residual with a `dz·dT/dz` term). It removes two nuisance
parameters instead of estimating them.

## Why it does not help on this deployment

**On a monotonic profile, a depth offset and a time shift are the same thing.**
With `w ≈ 0.3 dbar/s`, a 0.81 dbar depth offset is indistinguishable from a
2.7 s time shift. Changing coordinates does not break a degeneracy; it just
relabels it.

Concretely, what survives pressure-pairing is the CTD's **thermal response**.
Its pressure sensor reports the current depth, but its thermistor reports water
from `w·τ` ago. So pairing on pressure matches the FP07 at depth `P` against a
temperature that belongs to depth `P ∓ w·τ` — which reappears as an *apparent*
depth offset. Measured, depth-paired, on 13 files:

| terms fitted | rms (T1) | recovered |
|---|---|---|
| none | 11.90 mK | — |
| `τ·w·dT/dz` only | 11.77 mK | τ = −0.51 s |
| `dz·dT/dz` only | 11.67 mK | dz = 0.18 m |
| both | 11.54 mK | dz = 0.64 m, τ = 1.85 s |

The two terms remain collinear and the residual barely moves — the same 0.92
collinearity as the time-paired fit, in different clothing. Depth-pairing also
*raised* the residual (5.1 → 6.8 mK on a larger subset) precisely because it
removes the fitted lag that had been absorbing the response.

## This also reinterprets two earlier numbers

The "+4.52 s clock offset" from pressure-vs-pressure was never purely a clock
offset. On a monotonic ramp the estimator cannot separate a genuine time shift
from the geometric depth offset divided by `w`:

```
4.52 s  =  clock skew  +  (geometric depth offset) / w
        ≈  1.8 s       +  0.81 dbar / 0.3 dbar s⁻¹
```

And the "P_ctd − P_mr = +0.91 dbar" figure is the same ambiguity seen from the
other side: measured at the 4.52 s alignment it is +0.91 dbar; at the 7.4 s
temperature-lag alignment it is **+0.09 dbar**. The number is a function of the
assumed alignment, so on climb-only data neither value stands alone. The genuine
cross-calibration of the two pressure sensors is good — the shallow-water
difference is −0.013 dbar — which is what makes depth-pairing viable at all.

## What bidirectional data resolves

Sort the terms by how they behave when the glider reverses:

| term | depth-space form | dive vs climb |
|---|---|---|
| longitudinal separation | `sep_x·sin θ` | **flips** (nose down vs up) |
| perpendicular separation | `sep_z·cos θ` | does **not** flip |
| CTD thermal response | `w·τ = U·τ·sin θ` | **flips** |

Time-paired, dives separate `sep_z` from the rest, but `sep_x` and `τ` both
scale as `sin θ` and stay entangled except through speed variation.

**Pressure-paired, both geometric terms are already gone, leaving exactly one
parameter — `τ` — whose sign flips with `w`.** One parameter, one clean
discriminant. That is the experiment to run the moment the Rutgers data lands,
and it should determine the CTD response time directly and unambiguously.

## Status

`scripts/fp07cal_depthpair.py` implements pressure-pairing and the comparison
above. It is deliberately **not** wired into the CLI yet: on climb-only data it
is not an improvement, and switching the default on evidence that cannot
distinguish the two would be exactly the kind of move the rest of this work
exists to avoid. Promote it once bidirectional data shows `τ` resolving.

# Calibration coefficients: range checks and provenance

Two guards on the numbers that turn counts into physics. Both are adaptations of
ideas from [oceancascades/pyturb](https://github.com/oceancascades/pyturb)
(Jesse Cusack, Oregon State).

- A **plausible-range check** warns when a coefficient parses cleanly but cannot
  be right.
- **`cal_<key>` variable attributes** record, on every converted L1 variable,
  the coefficients that actually produced it.

---

## 1. Why a range check, on top of the strict parse

`channels._parse_finite_float` (issue #180, F06) already refuses a coefficient
that is *present but not a number* — a stray decimal comma, `nan`, `inf`, a
negative sensitivity. It fails closed rather than falling back to a generic
default, because a fabricated default is invisible: `diff_gain = 0,09` silently
became `1.0` and rescaled shear variance by 0.0081×.

That guard cannot see the other failure mode. A coefficient can parse, be
finite, be positive, and still be impossible:

```
sens = 1.0      # the un-filled setup.cfg placeholder
```

Real RSI shear probes have sensitivities near 0.04–0.12 V·s/m. Since
ε ∝ (diff_gain·sens)⁻², converting with `sens = 1.0` scales epsilon by roughly
**200×** — and before this check, produced **no warning at all**.

### Where the bounds come from

Not from the vendor sheet, and not adopted from upstream on faith. They are
checked against our own corpus: the shear inventory
(`/Volumes/SeaChest/Shear Inventory/shear_sensors.csv`, read 2026-09-07),
**15 998 channel-configs** read out of real `.p` files across ARCTERX, SUNRISE,
RIOT, CASPER, Taiwan, ASTRAL, Keck and goflow.

| coefficient | min | median | max | distinct values |
|---|---|---|---|---|
| `sens` | 0.041 | 0.1041 | 0.123 | 67 |
| `diff_gain` | 0.09 | 0.941 | 1.01 | 28 |

```python
_SHEAR_SENS_MIN = 0.03      # V*s/m
_SHEAR_SENS_MAX = 0.15
```

These are Jesse's bounds, and the corpus validates them: they bracket every
probe we have ever deployed with ~35 % margin below and ~22 % above, and fire on
**none** of the 15 998 rows.

### Why `diff_gain` gets a much wider bound

The `diff_gain` distribution is **bimodal**, and this is the finding that shaped
the design:

| band | rows | share | instruments | `fs_fast` |
|---|---|---|---|---|
| 0.090 – 0.099 | 2 236 | 14 % | SN **330**, **429** | **1024 Hz** |
| 0.09 | 434 | 3 % | SN **132** | 512 Hz |
| 0.905 – 1.01 | 13 762 | 86 % | 13 instruments | 512 Hz |

The low band is a **real population, not corruption**, and what separates it is
the **sampling rate**: every 1024 Hz instrument in the corpus is low-band, every
512 Hz instrument is high-band. SN 330 and 429 are high-energy/tidal builds,
where less pre-emphasis gain (and more bandwidth) is exactly what an energetic
flow calls for.

The comparison that isolates the rate from everything correlated with it is
**same-model**:

| model | `fs_fast` | `diff_gain` |
|---|---|---|
| `MR1000RDL-EM` | 512 Hz | SN 433: 0.927, SN 435: 0.941 |
| `MR1000RDL-EM` | **1024 Hz** | SN 429: 0.099 / 0.094 |

Identical hardware model, ~10× apart. Note the ~10× gain step is *not* the 2×
rate step, so the rate **marks** a different differentiator build rather than
scaling it; vehicle class is confounded with the rate and is not the driver.

A tight band around the upper mode would therefore flag a seventh of every shear
channel we own, and *an alarm that fires on a seventh of the corpus is an alarm
that gets muted*. `convert_shear` also sees only the **channel** config — not
the rate, not the vehicle — so it could not apply a class-aware bound even if
that were desirable.

> **This is not a verdict on SN 132 — if anything the opposite.** It is the one
> instrument the sampling-rate explanation does *not* cover: 511.95 Hz on a
> 10-column `vmp-250-IR`, the same sampling configuration as SN 412, 465 and 479
> at 0.937–0.99. Its `0.09 / 0.09` is also *identical* across channels, where
> all 13 high-band instruments and low-band SN 429 differ between theirs. See
> issue #178; the range check deliberately takes no position.

Distinguishing "0.09 is this instrument's real differentiator" from "0.09 is a
transcription error" needs that instrument's own history and class, not a static
window.
That is exactly what the [`rsi-tpw sensors --diff-gain` audit](sensors.md)
(`odas_tpw.rsi.diff_gain`, PR #177) does — it is instrument-keyed and compares across time, and it is the
right tool for this question. The bound here is deliberately only a floor and
ceiling for a value that cannot be a differentiator gain **at all**:

```python
_SHEAR_DIFF_GAIN_MIN = 0.01   # ~9x below the observed minimum
_SHEAR_DIFF_GAIN_MAX = 10.0   # ~10x above the observed maximum
```

### It warns; it does not substitute

A range violation is *"this looks wrong"*, not *"this cannot be honoured"*. The
value parsed, so it may be a real coefficient for hardware we have not met.
Conversion proceeds with the value as given, and the physical units follow from
it. Refusing would make an unfamiliar-but-valid instrument unreadable; only the
operator can tell those two cases apart.

This is the same principle as the F06 work, applied one level out: never
fabricate a coefficient, and never hide the one you were given.

### Not extended to FP07 or EM coefficients

Deliberately. We have no comparable fleet-wide inventory of `beta_1`/`t_0`
ranges, and a bound invented without data is a bound that fires on the first
unusual instrument. Add the check when the corpus exists to justify it.

---

## 2. `cal_<key>` provenance attributes

A converted L1 file is routinely read years later, on another machine, without
the instrument config that produced it — and a shear record is meaningless
without the `sens` and `diff_gain` that scaled it. Every L1 variable now carries
the coefficients that made it:

```
SHEAR:cal_sens      = [0.1075, 0.113]
SHEAR:cal_diff_gain = [0.954, 0.933]
SHEAR:sensor_names  = "sh1, sh2"
```

For a stacked variable the value is an array **parallel to `sensor_names`**, so
probe 0's coefficient stays attached to probe 0. A coefficient carried by only
some probes in a stack is NaN-filled for the others rather than dropped: a short
or re-ordered array would silently mis-attribute a coefficient.

This makes an audit answerable from the product — *"which sensitivity is baked
into this epsilon?"* — instead of by re-deriving it from a `.p` file that may
since have been re-patched by `fp07-cal` or `rsi-tpw patch-config`.

### Recorded by observation, not by a table

`channels.CalRecorder` is a `dict` subclass that notes which keys a converter
actually reads. The recorded set is therefore, by construction, exactly the
coefficients that were consumed:

- add a coefficient to a converter and it documents itself;
- stop reading one and the attribute disappears on its own;
- there is no parallel "coefficients for sensor type X" list to drift.

Keys *probed but absent* are not recorded — `convert_poly` walks `coef0..coef9`
to find the end of the polynomial, and a missing `coef7` is its exit condition,
not provenance. Structural keys a converter reads but that are not coefficients
(`name`, `units`) are excluded explicitly rather than relying on `float()` to
reject them, because a channel named `2` is legal in an RSI config.

An empty `cal` is a real answer, not a gap: `convert_piezo` reads only `a_0`, so
a `VIB` channel on an instrument that does not set it carries no `cal_*` attrs
because the conversion consumed no coefficients.

### What `cal_*` does *not* record

Coefficients **taken from the config** — not every number that entered the
arithmetic. Where a config omits a key and the converter falls back to a
documented default (`adc_fs = 4.096`, `adc_bits = 16`), no `cal_` attribute
appears.

That is the more useful of the two readings: it separates what the instrument
declared from what we assumed on its behalf. But read it correctly — a missing
`cal_adc_fs` means *"the config was silent"*, not *"no ADC scaling was
applied"*.

### Why the `cal_` prefix, when per-profile files use bare names

Per-profile NetCDFs written by `profile.extract_profiles` (#131 m8) already
carry `diff_gain`, `beta_1`, `e_b`, … as **bare** attribute names, and
`chi_io.py` reads them back. That mechanism is targeted: it is confined to
pre-emphasized gradient channels feeding the chi path, where those names happen
not to clash.

The L1 mechanism is general — it covers every converted variable — so it cannot
use bare names. `TEMP_CTD`'s JAC coefficients are literally called `a`, `b`,
`c`, `d`, `e`, `f`, and `convert_poly` reads a config key named `units`. Bare
attributes would collide with CF attributes and with each other. The two
conventions coexist deliberately; unifying them would break readers of existing
per-profile files for no gain.

---

## Effect on existing results

**None numerically.** The range check only warns, and `CalRecorder` is a
transparent `dict` subclass — the sample-exact `v6_golden_converted.npz`
regression (shape + finite-mask hash + decimated values) passes unchanged.

L1 files gain attributes; no variable, dimension or value changes.

The perturb engine fingerprint does change, since it hashes all of
`odas_tpw/*.py` — so cached stages recompute. On this branch that cost is
already paid: issue #180's fixes changed `perturb/config.py` in the same
release.

# Rutgers RU33 — MicroRider MR410, deployment 20211001T1841

Everything here is the **same instrument**: Rockland MicroRider **MR1000EM
SN 410**, EM flowmeter **SN 042**, on Rutgers Slocum glider **RU33**. There is
no VMP data in this tree — `dat_0377/VMP_results/` is named that way only
because `VMP_results` is perturb's default output directory name.

RU33 is the useful contrast to the OSU gliders: it flies **near-zero roll**
(≈ −1°, against 9–13°) and records **both legs** of the yo, which is what makes
the dive/climb geometry checks possible.

## Layout

```
Glider/            CTD source: Rutgers delayed-mode QC'd trajectory subset
                   (2021-10-01 → 10-20, 346 518 samples, CTD only — no attitude)
dinkum-hotel.yaml  Glider/*.nc  ->  hotel.nc      (sanitise + unit conversion)
hotel.nc           the shared CTD reference: T [degC], C [mS/cm], P [dbar]

MR/                THE MR TREE — 7 pressure-patched files, 2021-10-11 18:20 →
                   2021-10-12 09:54 UTC, 14.8 h recorded across 15.6 h wall
                   clock (six 5–11 min gaps between files).
  original/        the unpatched originals + Rockland's edits.yaml
fp07-cal.yaml      in-situ FP07 calibration for MR/
perturb.yaml       epsilon / chi processing for MR/

dat_0377/          an EARLIER, self-contained single-file run: dat_0377.p,
                   2021-10-17 15:20, 209 s. Same instrument, same deployment,
                   six days later. Its own patch.yaml / perturb.yaml / CSV
                   hotel feed / products. Left intact; all its paths are
                   <CONFIG_DIR>-relative so it still runs as-is.
```

## The patch these files carry

`coef2` on the pressure channel was `1.5879e8` where it should be `1.5879e-8`,
which put depth into the trillions of dbar. Both `MR/*.p` and `dat_0377.p`
carry the corrected value plus `vehicle = slocum_glider`. The unpatched
originals are kept beside them.

Note a separate, **uncorrected** issue: the MR pressure reads **−6 dbar at the
surface** where the glider CTD reads +0.11. That ~6 dbar zero offset does not
affect `dP/dt` (and so does not affect the speed or the profile detection), but
it does mean MR pressure is not absolute depth. True depth ≈ `P + 6`.

## Why the calibration lever arm is short

The MR window spans only **16.2–21.0 °C** (4.77 K), against 24 K on osu685 and
21 K on osu684. So `fit.order: auto` will pick order 1, `beta_1` (a slope) is
weakly constrained while `t_0` (an offset) is well constrained, and the
coefficients must not be extrapolated outside 16–21 °C. RU33's full 18-day
record spans 10.8 K — these coefficients describe less than half of it.

Folding `dat_0377` into the calibration does not help: its 209 s sit at
20.1–20.2 °C, entirely inside the existing range (77 CTD samples against
13 605).

---

## What was run, and what came out

```
dinkum-hotel build -c Rutgers/dinkum-hotel.yaml     # Glider/*.nc -> hotel.nc
fp07-cal fit   -c Rutgers/fp07-cal.yaml             # -> fp07cal/
fp07-cal patch -c Rutgers/fp07-cal.yaml             # -> fp07cal/patched/
rsi-tpw patch-config Rutgers/fp07cal/patched/*.p \
    --edits Rutgers/patch-pressure.yaml --out Rutgers/MR_final --batch-cal
perturb run -c Rutgers/perturb.yaml                 # -> MR_results/
perturb run -c Rutgers/perturb-flight.yaml          # -> MR_results_flight/
```

### FP07 calibration

The lag gate **initially refused** the fit — "NOT TRUSTWORTHY (flat peak)" on
both probes — and it was right to. In a well-mixed autumn shelf the top 25 dbar
is isothermal to ~0.1 K, so a 15–30 s high-pass keeps noise and discards the
signal. The lag itself was stable at 6.74–7.00 s across every window tried
(0.26 s spread); only the sharpness changed. `lag.detrend_s: 60` admits the
band the signal is actually in and the gate passes. See the config for the
full table.

| | factory | in-situ (order 1) | factory error over 16.3–21.0 °C |
|---|---|---|---|
| T1 | t₀ 289.301, β₁ 3143.55 | t₀ 285.270, β₁ 3092.89 | **+4.09 K** |
| T2 | t₀ 289.301, β₁ 3143.55 | t₀ 285.678, β₁ 3077.59 | **+3.63 K** |

In-sample rms 14.1 / 14.5 mK; **held-out 83.7 / 84.4 mK**. That 6× ratio is the
narrow-lever-arm signature — do not extrapolate outside 16–21 °C.

Clock offset (MR vs glider CTD): **+1.45 s**, much smaller than osu684's
+5.06 s. Applied as `hotel.time_offset: -1.45`.

### Products (199 profiles, 0.5 m bins, 2.2–42.2 dbar)

| | `em` arm | `flight` arm (α = 6.2°) |
|---|---|---|
| speed | 0.295 m/s | 0.299 m/s |
| ε | 2.15 × 10⁻⁷ W/kg | 2.34 × 10⁻⁷ |
| χ | 1.21 × 10⁻⁶ K²/s | 1.24 × 10⁻⁶ |
| K_ρ | 4.45 × 10⁻⁴ m²/s | 5.28 × 10⁻⁴ |
| Γ | 0.563 | 0.546 |

(medians; N² 7.93 × 10⁻⁵ s⁻² both arms; 64% of bins carry finite ε)

**The two arms agree**: speed ratio 0.997, ε ratio 0.912 (IQR 0.66–1.24,
MAD 0.138 dex). Contrast osu684-pre, where the identical comparison gave a
**1.76×** ε ratio. That is the difference between a deployment whose EM reading
is consistent with a physical angle of attack and one whose is not.

α = 6.2° was fitted *from* U_EM, so the two agree by construction in the mean;
the remaining per-sample scatter (~37% in ε) is real, and is what the
within-leg settling drift (~1°/100 s) looks like downstream.

Γ ≈ 0.56 here and ≈ 0.60 on osu684 — both well above the canonical 0.2, on
unrelated platforms. Worth a look.

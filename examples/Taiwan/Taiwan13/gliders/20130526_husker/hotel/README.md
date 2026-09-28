# Hotel file for MicroRider MR046 — glider `husker`, Taiwan 2013

`hotel.nc` — 312,510 times × 14 channels, 2013-05-02T04:45Z .. 2013-05-26T07:41Z.
Built by `dinkum-hotel build -c dinkum-hotel.yaml`, then `make_corrected_track.py`
adds the drift-corrected position.

| channel | source | notes |
|---|---|---|
| `sci_water_temp` | science CTD | 14.97 – 33.28 °C, 312,510 finite |
| `sci_water_cond` | science CTD | S/m → mS/cm (`scale: 10`), applied **once**, here |
| `sci_water_pressure` | science CTD | bar → dbar (`scale: 10`), −0.20 – 187.79 dbar |
| **`lat` / `lon`** | **derived** | **drift-corrected, decimal degrees** — use these |
| `lat_corrected` | derived | False where the track is raw DR (outside a GPS-bracketed dive) |
| `m_lat` / `m_lon` | flight | raw dead-reckoned, Slocum ddmm.mmmm, kept for provenance |
| `m_gps_lat` / `m_gps_lon` | flight | raw fixes, ddmm.mmmm; only 789 finite (surface only) |
| `m_depth`, `m_pitch` | flight | for flight-model / AOA work |
| `m_water_vx`, `m_water_vy` | flight | **all NaN — see below** |

## Did the 2013 flight computer dead-reckon? Yes.

Verified on `00210000.DBD`: **380 distinct `m_lat` values against only 4 GPS
fixes**; 352 distinct while deeper than 10 m, spread ~184 m; and `m_lat` never
equals a forward-fill of `m_gps_lat` (0 of 404 samples). `m_x_lmc`/`m_y_lmc`
and `m_dr_time` are present too. So `m_lat`/`m_lon` are genuine DR.

## The drift correction

This is Pat's algorithm, stated by him as:

> clean both datasets, DR and GPS; find the last DR position **strictly before**
> a GPS point and use the difference as the DR drift; then correct linearly in
> time from the **previous valid GPS point**.

Verified by independent re-implementation from that description: the resulting
track matches the one in `hotel.nc` to **0.0000 m median and 0.0000 m maximum
over 312,497 samples**, and the ramp anchor is a valid GPS fix in **65 of 65**
segments.

Two properties of 2013 Dinkum data that this formulation handles correctly, and
that a looser reading would not:

- **"strictly before" is load-bearing.** At a valid fix `m_lat` has *already*
  been snapped to `m_gps_lat`, so differencing at the fix sample itself yields a
  closing error of exactly **0.0 on every segment** — a null result that looks
  entirely plausible.
- **"last DR position" cannot be simplified to `n-1`.** A Slocum sensor is
  written only when it updates, so the record before a fix carries no `m_lat`:
  NaN in **65 of 65** segments, nearest real value a median of **16 records**
  back (max 245). Taking the last *valid* DR sample sidesteps this; indexing
  `n-1` directly would fail on every segment.

Related, and worth knowing though this algorithm does not depend on it:
`m_lat(n) == m_gps_lat(n)` at a valid fix holds to sub-meter at 1,650 of 1,653
fixes (median 0.0001 m) but is **never bit-identical**, and 3 fixes differ by up
to 33 m — so never test it with exact equality.

The DR endpoint sits a median of 73 s before the fix (max 152 s): the ascent and
surface-acquisition interval, through which the glider is still drifting, so
attributing it to the segment is correct.

Result over 65 dive segments (190 surface intervals):

| | median | p90 | max |
|---|---|---|---|
| closing drift | 1,588 m | 3,568 m | 5,213 m |
| dive duration | 2.69 h | | 6.09 h |
| implied depth-averaged current | 0.182 m/s | 0.334 m/s | 0.607 m/s |

138,708 of 140,522 DR positions lie inside a corrected segment.

### Independent validation

The glider computes its own depth-averaged current in `m_water_vx`/`m_water_vy`.
Comparing that against the current implied by my closing error, per segment:

```
n = 65 segments
  my implied |current|       median 0.182 m/s
  glider m_water_v |current| median 0.183 m/s
  u: corr 0.9987   median diff -0.0009 m/s
  v: corr 0.9996   median diff -0.0036 m/s
  vector difference: median 0.0074 m/s, p90 0.0124 m/s
```

So the correction reproduces the flight computer's own solution to ~7 mm/s.
Currents of 0.18–0.6 m/s are also physically right for this Kuroshio-influenced
part of the South China Sea.

## Things to know before using this

- **`m_water_vx`/`m_water_vy` are all NaN in `hotel.nc`.** They are reported only
  at surfacings — 140 samples for the whole deployment — so `projection.max_gap:
  120` correctly refuses to bridge them. `make_corrected_track.py` reads them
  straight from the DBD stream instead. Do not raise `max_gap` to "fix" this;
  that would interpolate a 2.7 h-sparse quantity across whole dives.
- **13 of the 77 MR `.p` files start outside hotel coverage.** Those are the
  pre-deployment bench files (3 in 2012-10, 2 in 2013-04) plus the 2013-05-01
  block — the science computer's record begins 2013-05-02T04:45Z. **64 of 77 are
  covered.**
- **Two segments are excluded by name** in `dinkum-hotel.yaml`: `00120000` and
  `00180000` are zero bytes on both computers and in every file type. Named
  explicitly rather than raising `files.max_skipped`, so a *new* undecodable file
  still fails the build loudly.
- **Uppercase `.DBD`/`.EBD` in the glob patterns is load-bearing** — these are
  DOS 8.3 names and Python's glob is case-sensitive even on this SMB mount.
- **Time bounds exclude pre-deployment leftovers.** The file system was not
  cleaned before deployment; `LOGS` retains 2012-09 segments (Massachusetts, and
  an earlier South China Sea visit). The clocks are sound — the fixes form two
  chronological runs with no interleaving, separated by a 227.5-day gap.
- **`sci_water_cond` reaches 0.0193 mS/cm and `sci_water_temp` 33.3 °C** — those
  are out-of-water/on-deck samples that pass the permissive `valid_min/max`.
  Trim on pressure or a profile mask before using.
- 2013 Dinkum files are self-describing, so `cache/` starts empty and the
  glider's own `STATE/CACHE` being empty is expected, not a fault.

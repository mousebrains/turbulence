"""Is the 0.0 'not sampled' sentinel row-wise or per-variable?

If row-wise -- timestamp is 0.0 exactly when the values are -- then dropping
rows by timestamp validity removes them all and no per-variable zero rule is
needed. If not, a value-level rule is required, and valid_range will NOT catch
it because 0.0 degC is inside any sane temperature range.
"""
import numpy as np
import xarray as xr

D = "/Volumes/SeaChest/ARCTERX/2025/Interior/Gliders/osu685"
ds = xr.open_dataset(f"{D}/PASS0/ebd.nc", decode_times=False, mask_and_scale=False)

t = np.asarray(ds["sci_ctd41cp_timestamp"].values, float).ravel()
V = {v: np.asarray(ds[v].values, float).ravel()
     for v in ("sci_water_temp", "sci_water_cond", "sci_water_pressure")}
n = t.size
print(f"{n:,} rows\n")

t_bad = ~np.isfinite(t) | (t <= 100)
print(f"timestamp unusable (NaN or <=100): {t_bad.sum():,}  ({100*t_bad.mean():.2f}%)")

for name, v in V.items():
    zero = v == 0.0
    both = zero & t_bad
    only_val = zero & ~t_bad
    print(f"\n{name}")
    print(f"  == 0.0                       : {zero.sum():,}")
    print(f"  ...and timestamp also unusable: {both.sum():,}")
    print(f"  ...but timestamp LOOKS FINE   : {only_val.sum():,}  <-- the ones a")
    print("                                     row-wise drop would MISS")
    if only_val.sum():
        idx = np.flatnonzero(only_val)[:3]
        print(f"  examples (row, t, value): "
              f"{[(int(i), float(t[i]), float(v[i])) for i in idx]}")

# And the converse: after dropping bad timestamps, what survives?
keep = ~t_bad
print(f"\nafter dropping unusable timestamps: {keep.sum():,} rows remain")
for name, v in V.items():
    z = (v[keep] == 0.0).sum()
    print(f"  {name}: {z:,} still exactly 0.0")
ds.close()

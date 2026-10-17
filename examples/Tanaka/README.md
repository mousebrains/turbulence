# Tanaka: glider MicroRider recipes, 2015–2017

Recipes for the four MicroRider deployments in `/Volumes/SeaChest/Tanaka/`.
Two of them are the Slocum and SeaExplorer of Tanaka, Hasegawa, Okunishi,
Yasuda & Welch (2022), "In situ calibration of underwater glider flight model
using acoustic Doppler current profilers," *JTECH* **39**, 1331–1352,
[doi:10.1175/JTECH-D-21-0074.1](https://doi.org/10.1175/JTECH-D-21-0074.1).

These are copies of `gliders/<deployment>/analysis/` on SeaChest, without the
products. Each script resolves its inputs relative to its own location
(`../post-recovery/...`), so run it from the SeaChest copy. Each deployment's
`README.md` gives the order and the results.

| deployment | platform | microstructure | speed | notes |
|---|---|---|---|---|
| `20170724_unit095` | Slocum G2, Oyashio, to 1000 m | MR SN 247 + AEM1-G EM | dynamic flight model with the paper's lift and C_D1 | ε; χ from T1 in its three working files only (`perturb_chi_T1.yaml`): T2 never worked and T1 broke from DAT_026. The model's along-glider speed matches the paper's DVL on descents to a median ratio of 0.997 |
| `20170620_sea025` | SeaExplorer, Kuroshio, to 700 m | MR-SE SN 240, RSI JAC-CT, EM `U_EMC` | steady-state model with the paper's SeaExplorer coefficients | needs the `rsijac_t`/`rsijac_c` converters (in main since b1270b5, #213) |
| `20151113_unit211` | Slocum, 29.4 N 127.7 E | MR1000-LP SN 058 | flight model, default lift | **sh1 dead**, excluded; FP07 fitted per file in all 12 files |
| `20151119_unit211` | Slocum, 32.3 N 133.2 E | MR1000-LP SN 058 | flight model, default lift | **sh1 dead**, excluded; FP07 fitted per file in all 11 files |

All four were reprocessed on main 98b4206 on 2026-10-17; each README's "What
changed" section gives the differences from the 2026-10-13 products.

The steps are the same as for the CASPER and Taiwan gliders: decode the glider
(`dinkum-hotel`, or `make_nav.py` for the SeaExplorer), solve the MicroRider
clock against glider pressure (`mr_clock.py`), compute a flight-model speed on
the MR clock (`make_flight_speed.py`, a `uv run` script that pins its
dependencies, including gliderflight 1.2.1), then run `perturb`.

One change here goes beyond the CASPER copies. `mr_clock.py` now **refuses a
file that holds a single leg**. On a near-monotonic pressure ramp the fitted
offset absorbs any time shift, so the lag is unconstrained even when the cost
minimum looks sharp. On unit211 (2015-11-13) two such files returned −9.0
and −6.25 s against +1.75 to +4.25 s from the full yos.

A review of the paper's own analysis, and what a re-analysis could improve, is
in `/Volumes/SeaChest/Tanaka/paper_TanakaEtAl_2022_JTECH/review_2026-10/REVIEW.md`.

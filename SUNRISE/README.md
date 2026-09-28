# SUNRISE — VMP recipes

Recipe only: **three** perturb configs plus `sections.yaml` per deployment, and
the helper scripts. **No data.** The data and products live on SeaChest at
`/Volumes/SeaChest/SUNRISE/Data/`, whose `VMP_README.md` and
`CAMPAIGN_RESULTS.md` carry the full findings, the file inventory and the
deletion manifest.

Gulf of Mexico shelf. Three instruments over three years and three vessels:

| year | ship | hull | SN142 | SN194 | SN412 |
|---|---|---|---|---|---|
| 2019 | Pelican | monohull | 12 | 43 | — |
| 2021 | Pelican | monohull | — | 22 | 81 |
| 2021 | Walton Smith | **catamaran** | 58 | 42 | — |
| 2022 | Pelican | monohull | 57 | — | 78 |
| 2022 | Point Sur | monohull | — | 78 | — |

471 raw `.p`, 43.7 GB. Layout `Data/<year>/<ship>/VMP/SN<nnn>/`; analysis in
`.../VMP/analysis/SN<nnn>/`.

## Three analyses from the same raw files

| config | anchor | bins | products |
|---|---|---|---|
| `perturb_topdown.yaml` | top-down | 1.0 m depth | `VMP_results/` |
| `perturb_bottomup.yaml` | bottom-up | 1.0 m depth | `VMP_results_bbl/` |
| `perturb_bottomup_hab.yaml` | bottom-up | 0.25 m **altitude** | `VMP_results_hab/` |

All three share one `sections.yaml` (names and time ranges, no paths), so the
same line can be plotted from each. Separate TREES rather than three
generations in one directory because `perturb-plot`'s `latest_stage_dir()`
always takes the highest-numbered generation and has no selector — co-locating
them would mean every plot silently got whichever was built last.

    ./run_three.sh        # all three analyses x all nine, resumable
    TREE=VMP_results_bbl ./check_runs.sh
    ./summarize.py VMP_results_hab    # tree name is the argument

**Do not compare the hab tree's campaign median with the other two**: it is
binned at 0.25 m against 1.0 m, so it averages less and reads higher for that
reason alone. The commensurable pair is top-down vs bbl.

## The three things that are easy to get wrong here

**1. The first metres are an ATTITUDE artifact, not prop wash.** These are
tow-yo casts: the VMP is launched nearly parallel to the surface, the reel
brake is released, and it pitches down as it falls. Until it is vertical the
shear probes see a MEAN CROSS-FLOW, so epsilon (~ shear^2/U^4) is meaningless
rather than merely contaminated — 4.5e5 x the interior at 1 m. `Incl_Y`
reaches 85 deg at a median 3.9 m but anywhere in 2.7-5.7 m, which is why the
configs use the per-cast `top_trim.use_inclinometer` voter and not a flat
floor.

**2. `residual_smooth_m` must scale with the CAST, not be copied.** The
residual is `w - movmean(w, window)`, so a window that is a large fraction of
a ~20 s cast absorbs the wash into its own baseline. At perturb's 6.0 default
two of five SUNRISE vessel-years read as having *no prop wash at all*; at 2 m
the wash is plainly there in every one. The configs use ~10% of each
deployment's cast span (2.0 m for the 22 m legs, 6.0 m for 2019's 64 m casts,
3.0 m for Point Sur's 34 m).

**3. VMP SN 412's inclinometers drop out.** Both channels, intermittently, on
61% (2021) and 83% (2022) of casts — frozen *within* a cast but varying
*between* casts, so a whole-file health test passes it. The voter is guarded
per-profile and abstains; those deployments carry a floor derived from the
healthy companion instrument that ran on the same ship (the VMPs were swapped
so one could charge), verified against SN 412's own working casts to within
0.25-0.58 m.

## Known data-quality limits

* **2021 embedded FP07 `beta_1` is garbage** (down to -230,435; negative is
  impossible). Harmless only because `fp07.calibrate: true` refits in situ.
* **Nine shear probes have no calibration sheet**: M1277, M1344, M1371,
  M1701, M1702, M2375, M2381, M2382, M289. Six others match their sheet
  exactly. Epsilon goes as sens^-2.
* **SN 142's probe pair disagrees** (ratios 0.4-1.8 within one deployment,
  inconsistent in direction) — the same pathology seen in the isotropy work.
* **Chi is weakly constrained by construction**: SUNRISE is energetic
  (kB ~165 cpm) so `K_max/kB` medians run ~0.49 and the Batchelor rolloff is
  never resolved.
* **2022 Point Sur's `gps.mat` has 4305 flipped longitude signs**; the `.nc`
  built by `make_gps_nc.py` repairs them. 2021 Walton Smith GPS covers only
  6 of SN 194's 42 files.

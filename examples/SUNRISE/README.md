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
* **Every shear probe now has a calibration sheet** (the last nine, M1277,
  M1344, M1371, M1701, M1702, M2375, M2381, M2382 and M289, arrived
  2026-10-12; see `microstructure_sensors/`). All configured values match the
  sheet in effect, with two exceptions. **M289 on SN 194, 2022 Point Sur, sh2**
  is configured 0.0725 (its 2016-04-11 sheet), but Rockland recalibrated it
  to **0.0794 on 2021-04-27**. If that is right, sh2 epsilon there is 1.20×
  high. **M2382 on SN 412, 2021 Pelican, sh2** carries 0.0691 (M2381's value)
  on some files instead of its own 0.0670. Several matches are stale: the
  2019 probes' calibrations were 23–37 months old. Epsilon goes as sens^-2.
* **SN 142's probe pair disagrees** (ratios 0.4-1.8 within one deployment,
  inconsistent in direction) — the same pathology seen in the isotropy work.
* **Chi is weakly constrained by construction**: SUNRISE is energetic
  (kB ~165 cpm) so `K_max/kB` medians run ~0.49 and the Batchelor rolloff is
  never resolved.
* **2022 Point Sur's `gps.mat` has 4305 flipped longitude signs**; the `.nc`
  built by `make_gps_nc.py` repairs them. The 2021 `gps.mat` extracts stop
  mid-cruise (Walton Smith covered only 6 of SN 194's 42 files). `gps.nc` for
  both 2021 ships was rebuilt from the ships' underway records on 2026-09-21
  (SeaChest `Data/UNDERWAY_README.md`), and every 2021 profile now has a
  position.

## Reprocessed 2026-10-20 on main 98b4206

All 54 runs were rebuilt with `microstructure-tpw 0.4.0 (git
v0.4.0-29-g98b42062069c)`, stamped in every product: the three configs plus a
generated `_dense` presentation twin of each, for all nine deployments. The
configs are the recipes here. 52 of the 54 runs had an old product to compare
against (engine 2755a1c, the #211 branch tip). The other two had failed in the
old campaign.

* **ε: bit-identical** in all 52.
* **χ: bit-identical except in five files**: SR1P2_0027/0080 (2021 Pelican
  SN412), SR2E1_0040/0073 (2022 Pelican SN412) and SR2E2_0028 (2022 Pelican
  SN142).
  * Each probe moves by at most 0.10 dex in the top-down tree, and up to
    0.27 dex in the bottom-up trees. `chiMean` moves up to 0.75 dex in a few
    bins where the two probes disagree by 1.5 dex, because the small change
    flips which probe passes QC.
  * The cause is #214: the FP07 fit's low-pass now starts at rest. A bisect ran
    SR2E1_0040 alone at six commits. 2755a1c, 0f65101 and b1270b5 reproduce
    the old product exactly; 53787f0 (#214), cc5107d and 98b4206 reproduce the
    new one.
  * Of the 433 files that yield profiles, these are exactly the five whose
    first profile starts within 193 slow samples (3.02 s) of the file start.
    The next four start 4.9–9.2 s in, and did not change.
* **2021 Pelican SN412 gains 152 profiles** from `SR1P2_0003.p`, which the old
  code refused at load for its T1 `beta_1 = "3-1315.08"`. #215 evaluates the
  arithmetic as ODAS does. The in-situ fit then replaces the coefficient
  anyway, and T1 agrees with JAC_T as well as the next file's does: a few mK
  either way, inside the ~4 mK scatter.
* **Stratification moves** with #219's temperature sort. N² is no longer forced
  positive, so N² ≤ 0 in 3–7% of the top-down tree's bins (0% before). In the
  two deployments checked, at least 98% of those bins carry no Γ or K_ρ.

Top-down tree. Medians are over each set's own bins; paired ratios are new/old,
the median over bins present in both:

| deployment | Γ old → new | paired Γ | paired N² | paired \|dT/dz\| | N² ≤ 0 |
|---|---|---|---|---|---|
| 2019 Pelican SN142 | 0.118 → 0.082 | 0.85 | 0.95 | 1.06 | 3.3% |
| 2019 Pelican SN194 | 0.076 → 0.061 | 0.91 | 0.98 | 1.04 | 3.2% |
| 2021 Pelican SN194 | 0.120 → 0.093 | 0.94 | 0.95 | 1.01 | 7.1% |
| 2021 Pelican SN412 | 0.126 → 0.095 | 0.91 | 0.93 | 1.01 | 5.5% |
| 2021 Walton Smith SN142 | 0.176 → 0.131 | 0.93 | 0.93 | 1.01 | 6.1% |
| 2021 Walton Smith SN194 | 0.068 → 0.050 | 0.90 | 0.95 | 1.02 | 4.5% |
| 2022 Pelican SN142 | 0.113 → 0.104 | 1.05 | 1.00 | 0.97 | 5.3% |
| 2022 Pelican SN412 | 0.118 → 0.107 | 1.04 | 1.00 | 0.98 | 4.6% |
| 2022 Point Sur SN194 | 0.048 → 0.043 | 1.03 | 1.00 | 0.98 | 5.6% |

In 2022 the paired ratio and the median move in opposite directions, and the
reason is a skewed change, not different bins. Most bins rise a few percent,
but 20–21% move by more than 0.1 dex, and 70–73% of those move down. On the
bins present in both sets, the median Γ falls ×0.93–0.94.

The full campaign table, regenerated on these products, is in SeaChest
`Data/CAMPAIGN_RESULTS.md`.

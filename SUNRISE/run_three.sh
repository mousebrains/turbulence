#!/bin/bash
# Build all THREE SUNRISE VMP analyses, from the same raw files.
#
#   1  perturb_topdown.yaml       -> VMP_results/      top-down,  depth bins
#   2  perturb_bottomup.yaml      -> VMP_results_bbl/  bottom-up, depth bins
#   3  perturb_bottomup_hab.yaml  -> VMP_results_hab/  bottom-up, ALTITUDE bins
#
# Three trees rather than three generations in one: perturb-plot's
# latest_stage_dir() always takes the highest-numbered generation and has no
# selector, so co-locating them would mean every plot silently got whichever
# was built last.
#
# All three share one sections.yaml (it carries names and time ranges, no
# paths), so the same line can be plotted from each:
#   perturb-plot scalar --root VMP_results_hab --sections sections.yaml --select line_003
#
# Resumable by construction: perturb versions output by a params hash and
# skips stages already up to date; the DONE file additionally stops a
# completed (deployment, analysis) pair being re-walked.
#
#   caffeinate -is nohup ./run_three.sh > run_three.log 2>&1 &
#
# AFTERWARDS run ./check_runs.sh with TREE= for each tree. "Pipeline complete:
# no errors" and rc=0 do NOT mean the data are clean -- the per-file warnings
# live in the worker logs, and a transient SMB "NetCDF: HDF error" can drop a
# profile without failing the run.
set -u
BASE=/Volumes/SeaChest/SUNRISE/Data
PERTURB=/Users/pat/tpw/turbulence/.venv/bin/perturb
DONE=$BASE/.run_three_done

DEPS=(
  "2019/Pelican/VMP/analysis/SN142"
  "2021/Pelican/VMP/analysis/SN194"
  "2021/WaltonSmith/VMP/analysis/SN194"
  "2019/Pelican/VMP/analysis/SN194"
  "2022/Pelican/VMP/analysis/SN142"
  "2021/WaltonSmith/VMP/analysis/SN142"
  "2022/Pelican/VMP/analysis/SN412"
  "2022/PointSur/VMP/analysis/SN194"
  "2021/Pelican/VMP/analysis/SN412"
)
CFGS=(perturb_topdown.yaml perturb_bottomup.yaml perturb_bottomup_hab.yaml)

touch "$DONE"
for cfg in "${CFGS[@]}"; do
  for d in "${DEPS[@]}"; do
    key="$cfg|$d"
    if grep -qxF "$key" "$DONE"; then
      echo "=== SKIP (done) $key"
      continue
    fi
    echo "=== START $key   $(date -u +%FT%TZ)"
    cd "$BASE/$d" || { echo "    cd failed"; continue; }
    "$PERTURB" run -c "$cfg"
    rc=$?
    echo "=== END   $key   rc=$rc   $(date -u +%FT%TZ)"
    if [ $rc -eq 0 ]; then echo "$key" >> "$DONE"; fi
  done
done
echo "=== ALL DONE $(date -u +%FT%TZ)"

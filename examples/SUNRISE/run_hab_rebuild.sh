#!/bin/bash
# Rebuild the THIRD analysis (bottom-up, ALTITUDE bins) for all nine
# deployments, after the binning.method routing fix.
#
# WHY A FULL REBUILD. Every routing site tested `method == "depth"` and fell
# through to the TIME branch for "altitude", so in the 2026-09-23/24 build the
# hab tree's profiles were binned by TIME and its combos glued lengthwise with
# featureType "trajectory" and a spurious time:1 dimension. diss_binned and
# chi_binned were correct (bin_diss/bin_chi mapped altitude properly); the
# profiles stage and all three combos were not. On 2022/PointSur/SN194 the time
# path then died with a bare SIGSEGV, which is the only reason it was caught.
#
# The fix changes the engine fingerprint, so every stage gets a new *_NN dir and
# the whole tree recomputes. That is deliberate: a pinned fingerprint would give
# this tree a provenance stamp that does not match the code that built it.
#
# VMP_results/ and VMP_results_bbl/ are NOT rebuilt. Both use method: depth,
# for which the fix is bit-identical (the depth branch is unchanged, the retry
# only fires on a failure, and the new validation only rejects bad values), so
# their products stand as built.
#
#   cd /Volumes/SeaChest/SUNRISE/Data
#   caffeinate -is nohup ./run_hab_rebuild.sh > run_hab_rebuild.log 2>&1 &
set -u
BASE=/Volumes/SeaChest/SUNRISE/Data
PERTURB=/Users/pat/tpw/turbulence/.venv/bin/perturb
DONE=$BASE/.run_hab_rebuild_done
CFG=perturb_bottomup_hab.yaml

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

touch "$DONE"
for d in "${DEPS[@]}"; do
  key="$CFG|$d"
  if grep -qxF "$key" "$DONE"; then echo "=== SKIP (done) $key"; continue; fi
  echo "=== START $key   $(date -u +%FT%TZ)"
  cd "$BASE/$d" || { echo "    cd failed"; continue; }
  "$PERTURB" run -c "$CFG"
  rc=$?
  echo "=== END   $key   rc=$rc   $(date -u +%FT%TZ)"
  if [ $rc -eq 0 ]; then echo "$key" >> "$DONE"; fi
done
echo "=== ALL DONE $(date -u +%FT%TZ)"

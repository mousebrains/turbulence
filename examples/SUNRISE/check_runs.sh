#!/bin/bash
# Audit every deployment's NEWEST run for the warnings that rc=0 hides.
#
# WHY THIS EXISTS. perturb's top-level run_*.log reports stage failures and
# file-level load errors. It does NOT aggregate the per-file warnings, which
# go to worker_<stamp>_<pid>.log. On 2026-09-21 the 2021 WaltonSmith SN194 run
# printed "Pipeline complete: no errors" with rc=0 while its workers emitted
# 19,952 GPS-extrapolation warnings -- the product carried a FABRICATED ship
# track 712 km long, because gps.max_time_diff only WARNS and positions past
# the end of coverage are linearly extrapolated without bound. The product
# reported 100% position coverage, so no coverage check could catch it.
#
# "rc=0 and a clean run log" is not evidence of clean data.
set -u
BASE=/Volumes/SeaChest/SUNRISE/Data
for d in $(cd "$BASE" && ls -d 20*/*/VMP/analysis/SN* 2>/dev/null); do
  TREE=${TREE:-VMP_results}
  L="$BASE/$d/$TREE/logs"
  [ -d "$L" ] || { printf '%-34s no logs\n' "$d"; continue; }
  run=$(ls -t "$L"/run_*.log 2>/dev/null | head -1)
  [ -n "$run" ] || { printf '%-34s no run log\n' "$d"; continue; }
  stamp=$(basename "$run" | sed -E 's/run_(.*)\.log/\1/')
  shopt -s nullglob
  workers=("$L"/worker_${stamp}_*.log)
  shopt -u nullglob
  gps=0; fp07=0; other=0
  if [ ${#workers[@]} -gt 0 ]; then
    gps=$(grep -h  "outside the GPS record" "${workers[@]}" 2>/dev/null | wc -l | tr -d ' ')
    fp07=$(grep -hE "fp07|Steinhart|thermistor" "${workers[@]}" 2>/dev/null | grep -ci warning | tr -d ' ')
    other=$(grep -h "UserWarning" "${workers[@]}" 2>/dev/null | wc -l | tr -d ' ')
  fi
  errs=$(grep -c "file error" "$run" 2>/dev/null | tr -d ' ')
  printf '%-34s %-18s workers=%-2d gps_extrap=%-6s fp07_warn=%-5s all_warn=%-7s file_err=%s\n' \
         "$d" "$stamp" "${#workers[@]}" "$gps" "$fp07" "$other" "$errs"
done

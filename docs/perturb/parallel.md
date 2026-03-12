# Parallel Scaling

The `perturb` pipeline supports parallel file processing via Python's `ProcessPoolExecutor`. Parallelism applies to the per-file processing stage (profile extraction, FP07 calibration, CT alignment, dissipation, chi, CTD binning). Trimming, merging, depth/time binning, and combo assembly run serially.

## Usage

```bash
perturb run -j 4 -o results/ VMP/*.p   # 4 workers
perturb run -j 0 -o results/ VMP/*.p   # auto (all cores)
perturb run -j 1 -o results/ VMP/*.p   # serial (default)
```

Or via configuration file:

```yaml
parallel:
  jobs: 4
```

CLI `-j` overrides the config file value.

## Benchmark Results

**Hardware:** Apple M4 Max, 16 physical cores, 128 GB RAM

**Workload:** 11 `.p` files (ARCTERX Thompson 2025, files `*002*`), 158 profiles total, ~200 dbar depth range. File sizes range from 1 profile (file 0027) to 30 profiles (file 0028).

| Workers (`-j`) | Wall time | Speedup | CPU utilization | Efficiency |
|:-:|:-:|:-:|:-:|:-:|
| 1 | 14:43 (883s) | 1.0x | 99% | 100% |
| 2 | 8:56 (536s) | 1.65x | 176% | 82% |
| 4 | 5:54 (354s) | 2.49x | 277% | 62% |
| 8 | 3:58 (238s) | 3.71x | 422% | 46% |
| 12 | 3:59 (239s) | 3.69x | 423% | 31% |

**Efficiency** is defined as speedup / workers. Perfect scaling would be 100%.

## Scaling Analysis

Scaling plateaus at approximately 8 workers for this workload. Several factors contribute:

1. **Amdahl's law:** Trimming, binning, and combo assembly are serial. These contribute a fixed ~10 s overhead regardless of worker count.

2. **Load imbalance:** With only 11 files of uneven size (1 to 30 profiles per file), the largest file becomes the bottleneck. At j=8+, most workers finish early and wait for the slowest file.

3. **Process spawn overhead:** Each `ProcessPoolExecutor` worker is a full Python process. Spawn + import time is ~1-2 s per worker.

4. **Memory bandwidth:** The dissipation computation is FFT-heavy. At high core counts, memory bandwidth becomes a limiting factor.

## Recommendations

- For small datasets (< 20 files), `-j 4` offers the best efficiency.
- For large datasets (100+ files), `-j 0` (all cores) will approach ideal scaling since load imbalance is reduced.
- Serial (`-j 1`) is useful for debugging — error messages are printed in file order.

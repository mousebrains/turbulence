# pyturb vs pyturb-cli: Quantitative Comparison

Comparison of epsilon (TKE dissipation rate) estimates from
[Jesse's pyturb](https://github.com/oceancascades/pyturb) and
`pyturb-cli` (this repository) on ARCTERX VMP-250 data
(SN 479, R/V Thompson, January 2025).

Both tools were run with matching parameters:
- `--fft-len 1.0` (512 samples at 512 Hz)
- `--diss-len 4.0` (2048 samples at 512 Hz)
- Goodman cleaning: **off** (`pyturb-cli` default matches pyturb)
- Overlap: n_fft // 2 = 256 samples

Epsilon values are compared by interpolating `log10(epsilon)` from
pyturb-cli onto pyturb's pressure grid. The median ratio
`pyturb-cli / pyturb` is reported per profile.

*Generated 2026-03-15 from 29 VMP .p files.*

## Summary

| Metric | Value |
|--------|-------|
| .p files processed | 28 |
| Total profiles | 459 |
| Profiles compared | 448 |
| pyturb-cli only | 0 |
| pyturb only | 11 |

## Epsilon Agreement

Agreement categories based on median |log10(ratio)| per profile:
- **good**: < 0.15 (within a factor of 1.41)
- **fair**: 0.15 -- 0.301 (within a factor of 2.0)
- **poor**: > 0.301 (outside a factor of 2.0)

| Variable | median ratio | 5th--95th pctl | range | good | fair | poor |
|----------|--------------|----------------|-------|------|------|------|
| `eps_1` | 1.047 | 0.97 -- 1.10 | 0.89 -- 1.16 | 437 | 11 | 0 |
| `eps_2` | 1.046 | 0.98 -- 1.10 | 0.94 -- 1.21 | 437 | 11 | 0 |

Overall, pyturb-cli epsilon is ~9% higher than pyturb for eps_1 and ~9% higher for eps_2. This small systematic offset is attributable to differences in the spectral estimation method (SCOR-160 vs custom) and Macoun & Lueck spatial response correction (applied in pyturb-cli, not in pyturb).

## Distribution of Per-Profile Epsilon Ratios

![Epsilon ratio histograms](epsilon_ratio_histograms.png)

## Processing Differences

| Feature | pyturb | pyturb-cli |
|---------|--------|------------|
| Shear spectrum | Custom | SCOR-160 (Lueck 2024) |
| Noise removal | None | Goodman (opt-in via `--goodman`) |
| Spatial correction | None | Macoun & Lueck (2004) |
| Window conversion | `int(s*fs)`, even | Same (matching pyturb) |
| Overlap | n_fft // 2 | Same (matching pyturb) |
| Profile detection | profinder | scipy.signal.find_peaks |
| CLI framework | Typer | argparse |

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
- **good**: < 0.15 (within ~40%)
- **fair**: 0.15 -- 0.5 (within ~3x)
- **poor**: > 0.5

| Variable | mean log10(ratio) | std | median ratio range | good | fair | poor |
|----------|-------------------|-----|--------------------|------|------|------|
| `eps_1` | +0.0357 | 0.0195 | 0.89 -- 1.16 | 437 | 11 | 0 |
| `eps_2` | +0.0362 | 0.0190 | 0.94 -- 1.21 | 437 | 11 | 0 |

Overall, pyturb-cli epsilon is ~9% higher than pyturb for eps_1 and ~9% higher for eps_2. This small systematic offset is attributable to differences in the spectral estimation method (SCOR-160 vs custom) and Macoun & Lueck spatial response correction (applied in pyturb-cli, not in pyturb).

## Per-File Summary

| File | Profiles | eps_1 ratio | eps_2 ratio | eps_1 | eps_2 |
|------|----------|-------------|-------------|-------|-------|
| 0002 | 20/20 | 1.04 (0.95--1.11) | 1.04 (0.99--1.10) | 20g/0f/0p | 20g/0f/0p |
| 0003 | 24/25 | 1.04 (0.98--1.11) | 1.05 (0.97--1.19) | 24g/0f/0p | 24g/0f/0p |
| 0004 | 7/7 | 1.04 (1.00--1.07) | 1.05 (1.02--1.09) | 7g/0f/0p | 7g/0f/0p |
| 0005 | 19/20 | 1.03 (0.92--1.14) | 1.02 (0.96--1.08) | 16g/3f/0p | 17g/2f/0p |
| 0006 | 2/2 | 1.04 (1.02--1.05) | 1.02 (1.01--1.03) | 2g/0f/0p | 2g/0f/0p |
| 0007 | 7/7 | 1.04 (0.99--1.12) | 1.02 (0.98--1.06) | 7g/0f/0p | 6g/1f/0p |
| 0008 | 4/4 | 1.02 (1.00--1.07) | 1.02 (1.00--1.08) | 4g/0f/0p | 4g/0f/0p |
| 0009 | 25/25 | 1.02 (0.91--1.09) | 1.04 (0.97--1.21) | 25g/0f/0p | 24g/1f/0p |
| 0010 | 11/11 | 1.05 (0.99--1.12) | 1.03 (0.97--1.08) | 11g/0f/0p | 11g/0f/0p |
| 0011 | 21/21 | 1.04 (0.97--1.10) | 1.05 (0.98--1.13) | 21g/0f/0p | 21g/0f/0p |
| 0012 | 15/15 | 1.02 (0.94--1.07) | 1.05 (0.98--1.11) | 15g/0f/0p | 15g/0f/0p |
| 0013 | 26/27 | 1.04 (0.89--1.16) | 1.04 (0.95--1.10) | 24g/2f/0p | 25g/1f/0p |
| 0014 | 11/12 | 1.06 (1.00--1.09) | 1.05 (0.98--1.10) | 10g/1f/0p | 10g/1f/0p |
| 0015 | 29/30 | 1.06 (1.02--1.09) | 1.06 (1.02--1.11) | 29g/0f/0p | 29g/0f/0p |
| 0016 | 14/15 | 1.06 (1.03--1.11) | 1.04 (0.95--1.08) | 14g/0f/0p | 14g/0f/0p |
| 0017 | 24/24 | 1.05 (0.96--1.15) | 1.05 (0.95--1.12) | 22g/2f/0p | 22g/2f/0p |
| 0018 | 9/9 | 1.01 (0.96--1.10) | 1.02 (0.94--1.08) | 9g/0f/0p | 9g/0f/0p |
| 0019 | 28/28 | 1.06 (1.02--1.12) | 1.06 (0.99--1.12) | 27g/1f/0p | 28g/0f/0p |
| 0020 | 6/8 | 1.06 (1.01--1.08) | 1.05 (1.03--1.08) | 6g/0f/0p | 6g/0f/0p |
| 0022 | 24/24 | 1.02 (0.94--1.09) | 1.03 (0.94--1.11) | 24g/0f/0p | 24g/0f/0p |
| 0023 | 9/9 | 1.04 (1.00--1.08) | 1.05 (1.02--1.17) | 8g/1f/0p | 7g/2f/0p |
| 0024 | 27/27 | 1.04 (0.98--1.09) | 1.05 (0.97--1.10) | 27g/0f/0p | 27g/0f/0p |
| 0025 | 14/14 | 1.04 (0.99--1.08) | 1.05 (0.96--1.11) | 14g/0f/0p | 14g/0f/0p |
| 0026 | 10/10 | 1.06 (1.02--1.09) | 1.04 (0.98--1.07) | 10g/0f/0p | 10g/0f/0p |
| 0027 | 1/1 | 1.09 (1.09--1.09) | 1.06 (1.06--1.06) | 1g/0f/0p | 1g/0f/0p |
| 0028 | 26/29 | 1.05 (0.96--1.16) | 1.05 (0.99--1.13) | 25g/1f/0p | 25g/1f/0p |
| 0029 | 15/15 | 1.02 (0.93--1.10) | 1.01 (0.94--1.08) | 15g/0f/0p | 15g/0f/0p |
| 0030 | 20/20 | 1.06 (1.00--1.11) | 1.05 (0.99--1.10) | 20g/0f/0p | 20g/0f/0p |

## Per-Profile Detail

Median ratio = `median(pyturb-cli / pyturb)` for pressure-matched estimates.

### File 0002

| Profile | n_pyturb | n_cli | eps_1 | ratio | eps_2 | ratio |
|---------|----------|-------|-------|-------|-------|-------|
| 0 | 57 | 58 | good | 1.05 | good | 1.05 |
| 1 | 57 | 58 | good | 1.04 | good | 1.07 |
| 2 | 57 | 58 | good | 0.95 | good | 0.99 |
| 3 | 57 | 57 | good | 1.05 | good | 1.08 |
| 4 | 57 | 58 | good | 1.05 | good | 1.07 |
| 5 | 56 | 57 | good | 1.06 | good | 1.04 |
| 6 | 57 | 58 | good | 1.01 | good | 1.10 |
| 7 | 57 | 58 | good | 1.09 | good | 1.05 |
| 8 | 56 | 57 | good | 1.05 | good | 1.05 |
| 9 | 56 | 57 | good | 0.98 | good | 1.03 |
| 10 | 59 | 59 | good | 0.99 | good | 1.02 |
| 11 | 57 | 58 | good | 1.11 | good | 1.05 |
| 12 | 57 | 57 | good | 1.06 | good | 1.06 |
| 13 | 56 | 57 | good | 1.00 | good | 1.06 |
| 14 | 57 | 58 | good | 1.03 | good | 1.05 |
| 15 | 57 | 58 | good | 1.03 | good | 1.04 |
| 16 | 57 | 58 | good | 1.02 | good | 1.05 |
| 17 | 57 | 57 | good | 1.01 | good | 1.01 |
| 18 | 57 | 58 | good | 1.10 | good | 1.02 |
| 19 | 57 | 57 | good | 1.07 | good | 1.01 |

### File 0003

*1 profile(s) in pyturb only (not in pyturb-cli).*

| Profile | n_pyturb | n_cli | eps_1 | ratio | eps_2 | ratio |
|---------|----------|-------|-------|-------|-------|-------|
| 0 | 57 | 58 | good | 1.00 | good | 1.04 |
| 1 | 57 | 57 | good | 1.00 | good | 1.01 |
| 2 | 57 | 57 | good | 1.09 | good | 1.09 |
| 3 | 59 | 60 | good | 1.03 | good | 1.02 |
| 4 | 56 | 57 | good | 1.01 | good | 1.00 |
| 5 | 57 | 57 | good | 1.11 | good | 1.19 |
| 6 | 57 | 59 | good | 1.00 | good | 1.03 |
| 7 | 57 | 57 | good | 1.03 | good | 1.07 |
| 8 | 57 | 58 | good | 1.05 | good | 1.06 |
| 9 | 57 | 57 | good | 1.02 | good | 0.97 |
| 11 | 57 | 58 | good | 1.04 | good | 1.13 |
| 12 | 57 | 57 | good | 0.99 | good | 0.97 |
| 13 | 57 | 57 | good | 1.05 | good | 1.01 |
| 14 | 58 | 59 | good | 0.99 | good | 1.01 |
| 15 | 58 | 59 | good | 0.98 | good | 1.09 |
| 16 | 58 | 59 | good | 1.07 | good | 1.05 |
| 17 | 58 | 58 | good | 1.06 | good | 1.00 |
| 18 | 57 | 57 | good | 1.05 | good | 1.03 |
| 19 | 56 | 57 | good | 1.07 | good | 1.07 |
| 20 | 57 | 57 | good | 1.03 | good | 1.04 |
| 21 | 56 | 57 | good | 1.06 | good | 1.05 |
| 22 | 56 | 57 | good | 1.04 | good | 1.06 |
| 23 | 57 | 57 | good | 1.03 | good | 1.13 |
| 24 | 57 | 57 | good | 1.07 | good | 1.09 |

### File 0004

| Profile | n_pyturb | n_cli | eps_1 | ratio | eps_2 | ratio |
|---------|----------|-------|-------|-------|-------|-------|
| 0 | 19 | 19 | good | 1.03 | good | 1.04 |
| 1 | 56 | 57 | good | 1.00 | good | 1.02 |
| 2 | 57 | 59 | good | 1.06 | good | 1.09 |
| 3 | 57 | 57 | good | 1.02 | good | 1.04 |
| 4 | 57 | 58 | good | 1.03 | good | 1.05 |
| 5 | 53 | 53 | good | 1.07 | good | 1.08 |
| 6 | 56 | 57 | good | 1.06 | good | 1.07 |

### File 0005

*1 profile(s) in pyturb only (not in pyturb-cli).*

| Profile | n_pyturb | n_cli | eps_1 | ratio | eps_2 | ratio |
|---------|----------|-------|-------|-------|-------|-------|
| 0 | 56 | 56 | good | 1.02 | good | 0.99 |
| 1 | 57 | 57 | good | 1.04 | good | 1.01 |
| 2 | 56 | 57 | good | 1.06 | good | 1.08 |
| 3 | 56 | 57 | good | 1.03 | good | 1.07 |
| 4 | 57 | 58 | good | 1.06 | good | 1.07 |
| 5 | 57 | 57 | good | 0.96 | good | 0.96 |
| 6 | 57 | 57 | fair | 1.00 | fair | 0.96 |
| 7 | 57 | 58 | good | 0.97 | good | 0.96 |
| 9 | 57 | 58 | good | 0.92 | good | 0.96 |
| 10 | 56 | 58 | good | 0.98 | good | 0.98 |
| 11 | 56 | 56 | good | 1.02 | good | 1.04 |
| 12 | 57 | 58 | good | 1.03 | good | 1.03 |
| 13 | 57 | 58 | good | 1.05 | good | 1.06 |
| 14 | 57 | 58 | good | 1.14 | good | 1.03 |
| 15 | 58 | 58 | good | 1.01 | good | 1.04 |
| 16 | 57 | 58 | good | 1.02 | good | 1.04 |
| 17 | 57 | 57 | fair | 1.09 | good | 1.00 |
| 18 | 57 | 57 | good | 1.10 | good | 1.05 |
| 19 | 57 | 57 | fair | 1.14 | fair | 1.08 |

### File 0006

| Profile | n_pyturb | n_cli | eps_1 | ratio | eps_2 | ratio |
|---------|----------|-------|-------|-------|-------|-------|
| 0 | 57 | 57 | good | 1.02 | good | 1.01 |
| 1 | 57 | 59 | good | 1.05 | good | 1.03 |

### File 0007

| Profile | n_pyturb | n_cli | eps_1 | ratio | eps_2 | ratio |
|---------|----------|-------|-------|-------|-------|-------|
| 0 | 57 | 58 | good | 1.06 | good | 1.06 |
| 1 | 56 | 57 | good | 1.01 | good | 0.98 |
| 2 | 57 | 57 | good | 0.99 | good | 0.99 |
| 3 | 57 | 58 | good | 1.12 | fair | 1.03 |
| 4 | 57 | 58 | good | 1.03 | good | 1.05 |
| 5 | 51 | 51 | good | 1.03 | good | 1.05 |
| 6 | 51 | 51 | good | 1.06 | good | 0.99 |

### File 0008

| Profile | n_pyturb | n_cli | eps_1 | ratio | eps_2 | ratio |
|---------|----------|-------|-------|-------|-------|-------|
| 0 | 57 | 58 | good | 1.07 | good | 1.08 |
| 1 | 57 | 57 | good | 1.01 | good | 1.02 |
| 2 | 57 | 57 | good | 1.03 | good | 1.00 |
| 3 | 56 | 57 | good | 1.00 | good | 1.00 |

### File 0009

| Profile | n_pyturb | n_cli | eps_1 | ratio | eps_2 | ratio |
|---------|----------|-------|-------|-------|-------|-------|
| 0 | 57 | 57 | good | 0.99 | good | 1.02 |
| 1 | 54 | 57 | good | 0.94 | good | 1.02 |
| 2 | 56 | 57 | good | 1.05 | good | 1.04 |
| 3 | 57 | 57 | good | 1.04 | fair | 1.14 |
| 4 | 57 | 57 | good | 1.04 | good | 1.01 |
| 5 | 57 | 57 | good | 1.01 | good | 1.02 |
| 6 | 54 | 57 | good | 1.09 | good | 1.01 |
| 7 | 57 | 57 | good | 1.09 | good | 1.21 |
| 8 | 55 | 57 | good | 1.06 | good | 1.09 |
| 9 | 56 | 57 | good | 1.03 | good | 1.07 |
| 10 | 56 | 57 | good | 1.05 | good | 1.03 |
| 11 | 55 | 57 | good | 1.02 | good | 1.03 |
| 12 | 59 | 59 | good | 1.02 | good | 1.05 |
| 13 | 56 | 57 | good | 1.04 | good | 1.04 |
| 14 | 57 | 59 | good | 1.05 | good | 1.06 |
| 15 | 56 | 58 | good | 1.01 | good | 1.01 |
| 16 | 57 | 58 | good | 1.07 | good | 1.00 |
| 17 | 55 | 57 | good | 1.07 | good | 1.02 |
| 18 | 57 | 57 | good | 1.00 | good | 1.06 |
| 19 | 57 | 58 | good | 0.91 | good | 0.97 |
| 20 | 57 | 57 | good | 0.96 | good | 1.04 |
| 21 | 56 | 58 | good | 1.05 | good | 1.06 |
| 22 | 56 | 57 | good | 1.05 | good | 1.05 |
| 23 | 56 | 57 | good | 1.00 | good | 0.99 |
| 24 | 56 | 59 | good | 0.96 | good | 1.00 |

### File 0010

| Profile | n_pyturb | n_cli | eps_1 | ratio | eps_2 | ratio |
|---------|----------|-------|-------|-------|-------|-------|
| 0 | 51 | 52 | good | 1.05 | good | 1.05 |
| 1 | 56 | 57 | good | 1.05 | good | 1.02 |
| 2 | 57 | 57 | good | 0.99 | good | 1.00 |
| 3 | 57 | 57 | good | 1.03 | good | 1.04 |
| 4 | 57 | 58 | good | 1.11 | good | 1.05 |
| 5 | 57 | 58 | good | 1.04 | good | 1.04 |
| 6 | 57 | 58 | good | 1.06 | good | 1.08 |
| 7 | 58 | 59 | good | 1.02 | good | 1.06 |
| 8 | 57 | 58 | good | 1.12 | good | 1.00 |
| 9 | 58 | 58 | good | 1.01 | good | 0.97 |
| 10 | 57 | 58 | good | 1.07 | good | 1.02 |

### File 0011

| Profile | n_pyturb | n_cli | eps_1 | ratio | eps_2 | ratio |
|---------|----------|-------|-------|-------|-------|-------|
| 0 | 57 | 57 | good | 0.98 | good | 1.12 |
| 1 | 56 | 58 | good | 1.06 | good | 1.07 |
| 2 | 57 | 59 | good | 1.03 | good | 1.04 |
| 3 | 56 | 57 | good | 1.07 | good | 1.04 |
| 4 | 56 | 57 | good | 1.05 | good | 1.01 |
| 5 | 57 | 58 | good | 1.00 | good | 0.99 |
| 6 | 57 | 58 | good | 1.03 | good | 1.10 |
| 7 | 57 | 58 | good | 1.05 | good | 1.04 |
| 8 | 57 | 57 | good | 1.09 | good | 1.07 |
| 9 | 59 | 60 | good | 1.02 | good | 1.02 |
| 10 | 57 | 57 | good | 0.97 | good | 1.00 |
| 11 | 60 | 61 | good | 1.10 | good | 1.08 |
| 12 | 57 | 58 | good | 0.99 | good | 1.02 |
| 13 | 57 | 58 | good | 1.07 | good | 1.12 |
| 14 | 57 | 58 | good | 1.04 | good | 1.13 |
| 15 | 57 | 58 | good | 1.04 | good | 1.06 |
| 16 | 57 | 58 | good | 1.08 | good | 1.11 |
| 17 | 57 | 57 | good | 1.05 | good | 1.07 |
| 18 | 57 | 57 | good | 1.00 | good | 0.98 |
| 19 | 54 | 55 | good | 1.05 | good | 1.05 |
| 20 | 57 | 57 | good | 0.99 | good | 0.99 |

### File 0012

| Profile | n_pyturb | n_cli | eps_1 | ratio | eps_2 | ratio |
|---------|----------|-------|-------|-------|-------|-------|
| 0 | 57 | 58 | good | 1.07 | good | 1.08 |
| 1 | 57 | 58 | good | 1.05 | good | 1.08 |
| 2 | 57 | 57 | good | 1.02 | good | 1.07 |
| 3 | 57 | 57 | good | 0.99 | good | 1.02 |
| 4 | 57 | 57 | good | 1.00 | good | 1.07 |
| 5 | 57 | 57 | good | 0.94 | good | 1.00 |
| 6 | 57 | 58 | good | 1.02 | good | 1.07 |
| 7 | 57 | 58 | good | 1.05 | good | 1.02 |
| 8 | 57 | 58 | good | 1.05 | good | 1.06 |
| 9 | 58 | 59 | good | 0.98 | good | 1.03 |
| 10 | 59 | 60 | good | 1.02 | good | 1.02 |
| 11 | 59 | 60 | good | 1.06 | good | 1.06 |
| 12 | 57 | 58 | good | 1.05 | good | 1.07 |
| 13 | 57 | 58 | good | 0.98 | good | 0.98 |
| 14 | 57 | 57 | good | 0.98 | good | 1.11 |

### File 0013

*1 profile(s) in pyturb only (not in pyturb-cli).*

| Profile | n_pyturb | n_cli | eps_1 | ratio | eps_2 | ratio |
|---------|----------|-------|-------|-------|-------|-------|
| 0 | 56 | 57 | good | 1.09 | good | 1.08 |
| 1 | 57 | 58 | good | 1.00 | good | 1.02 |
| 2 | 56 | 57 | good | 1.02 | good | 1.03 |
| 3 | 58 | 58 | good | 0.99 | good | 1.04 |
| 4 | 57 | 58 | good | 1.04 | good | 1.07 |
| 5 | 56 | 57 | good | 1.03 | good | 1.10 |
| 6 | 56 | 57 | good | 1.01 | good | 1.04 |
| 7 | 57 | 58 | good | 1.16 | good | 1.08 |
| 8 | 57 | 58 | good | 1.00 | good | 0.97 |
| 9 | 17 | 18 | good | 1.06 | good | 1.08 |
| 10 | 57 | 58 | good | 1.05 | good | 1.02 |
| 11 | 57 | 57 | good | 1.15 | good | 1.06 |
| 12 | 57 | 57 | good | 1.07 | good | 1.06 |
| 13 | 56 | 57 | good | 0.97 | good | 0.95 |
| 14 | 57 | 57 | good | 1.09 | good | 1.04 |
| 15 | 57 | 58 | fair | 1.10 | fair | 1.08 |
| 16 | 57 | 58 | good | 1.04 | good | 1.07 |
| 17 | 56 | 57 | good | 0.99 | good | 0.99 |
| 18 | 57 | 57 | fair | 1.16 | good | 1.04 |
| 19 | 57 | 58 | good | 1.01 | good | 1.04 |
| 21 | 57 | 57 | good | 1.09 | good | 1.07 |
| 22 | 56 | 57 | good | 0.89 | good | 1.00 |
| 23 | 56 | 57 | good | 1.06 | good | 1.07 |
| 24 | 56 | 57 | good | 1.00 | good | 1.05 |
| 25 | 56 | 57 | good | 1.03 | good | 1.03 |
| 26 | 56 | 56 | good | 0.99 | good | 1.06 |

### File 0014

*1 profile(s) in pyturb only (not in pyturb-cli).*

| Profile | n_pyturb | n_cli | eps_1 | ratio | eps_2 | ratio |
|---------|----------|-------|-------|-------|-------|-------|
| 0 | 57 | 57 | good | 1.05 | good | 1.07 |
| 1 | 56 | 56 | good | 1.09 | good | 0.98 |
| 2 | 56 | 57 | good | 1.05 | good | 1.07 |
| 3 | 56 | 56 | good | 1.07 | good | 1.07 |
| 4 | 58 | 58 | good | 1.01 | good | 0.99 |
| 5 | 58 | 59 | good | 1.00 | good | 1.04 |
| 6 | 56 | 57 | good | 1.07 | good | 1.10 |
| 7 | 57 | 58 | good | 1.08 | good | 1.05 |
| 9 | 57 | 58 | good | 1.06 | good | 1.04 |
| 10 | 57 | 58 | fair | 1.05 | fair | 1.06 |
| 11 | 57 | 57 | good | 1.09 | good | 1.04 |

### File 0015

*1 profile(s) in pyturb only (not in pyturb-cli).*

| Profile | n_pyturb | n_cli | eps_1 | ratio | eps_2 | ratio |
|---------|----------|-------|-------|-------|-------|-------|
| 0 | 57 | 57 | good | 1.08 | good | 1.07 |
| 1 | 56 | 57 | good | 1.06 | good | 1.08 |
| 2 | 58 | 58 | good | 1.07 | good | 1.05 |
| 3 | 57 | 57 | good | 1.04 | good | 1.10 |
| 4 | 57 | 58 | good | 1.06 | good | 1.06 |
| 6 | 57 | 57 | good | 1.06 | good | 1.08 |
| 7 | 57 | 58 | good | 1.07 | good | 1.09 |
| 8 | 57 | 58 | good | 1.05 | good | 1.04 |
| 9 | 57 | 58 | good | 1.06 | good | 1.04 |
| 10 | 57 | 57 | good | 1.08 | good | 1.02 |
| 11 | 56 | 57 | good | 1.03 | good | 1.07 |
| 12 | 57 | 58 | good | 1.07 | good | 1.11 |
| 13 | 57 | 58 | good | 1.07 | good | 1.06 |
| 14 | 57 | 58 | good | 1.09 | good | 1.06 |
| 15 | 57 | 57 | good | 1.05 | good | 1.08 |
| 16 | 57 | 58 | good | 1.05 | good | 1.05 |
| 17 | 58 | 59 | good | 1.02 | good | 1.05 |
| 18 | 57 | 58 | good | 1.07 | good | 1.07 |
| 19 | 57 | 59 | good | 1.09 | good | 1.03 |
| 20 | 58 | 58 | good | 1.06 | good | 1.09 |
| 21 | 56 | 57 | good | 1.05 | good | 1.05 |
| 22 | 56 | 58 | good | 1.05 | good | 1.09 |
| 23 | 56 | 57 | good | 1.06 | good | 1.07 |
| 24 | 57 | 58 | good | 1.07 | good | 1.04 |
| 25 | 57 | 58 | good | 1.05 | good | 1.06 |
| 26 | 57 | 58 | good | 1.07 | good | 1.08 |
| 27 | 57 | 57 | good | 1.02 | good | 1.09 |
| 28 | 59 | 60 | good | 1.06 | good | 1.06 |
| 29 | 57 | 57 | good | 1.06 | good | 1.03 |

### File 0016

*1 profile(s) in pyturb only (not in pyturb-cli).*

| Profile | n_pyturb | n_cli | eps_1 | ratio | eps_2 | ratio |
|---------|----------|-------|-------|-------|-------|-------|
| 0 | 57 | 57 | good | 1.10 | good | 1.04 |
| 1 | 57 | 58 | good | 1.11 | good | 1.08 |
| 2 | 57 | 58 | good | 1.04 | good | 0.95 |
| 3 | 57 | 57 | good | 1.04 | good | 1.03 |
| 4 | 57 | 59 | good | 1.05 | good | 1.06 |
| 5 | 56 | 57 | good | 1.07 | good | 1.05 |
| 6 | 57 | 58 | good | 1.07 | good | 1.08 |
| 7 | 57 | 57 | good | 1.08 | good | 1.08 |
| 8 | 58 | 59 | good | 1.03 | good | 1.03 |
| 10 | 58 | 59 | good | 1.06 | good | 1.02 |
| 11 | 57 | 59 | good | 1.06 | good | 1.07 |
| 12 | 56 | 57 | good | 1.06 | good | 1.03 |
| 13 | 57 | 58 | good | 1.08 | good | 1.08 |
| 14 | 57 | 57 | good | 1.05 | good | 1.02 |

### File 0017

| Profile | n_pyturb | n_cli | eps_1 | ratio | eps_2 | ratio |
|---------|----------|-------|-------|-------|-------|-------|
| 0 | 57 | 57 | good | 1.08 | good | 1.06 |
| 1 | 57 | 58 | good | 1.07 | good | 0.97 |
| 2 | 57 | 58 | good | 1.01 | good | 1.02 |
| 3 | 56 | 57 | good | 1.06 | good | 1.07 |
| 4 | 57 | 57 | good | 1.11 | good | 1.07 |
| 5 | 59 | 60 | good | 1.15 | good | 1.10 |
| 6 | 57 | 57 | fair | 0.99 | good | 1.00 |
| 7 | 57 | 58 | good | 1.05 | good | 1.01 |
| 8 | 57 | 58 | good | 1.05 | good | 1.02 |
| 9 | 68 | 69 | good | 1.02 | good | 1.05 |
| 10 | 56 | 57 | good | 1.01 | good | 0.95 |
| 11 | 57 | 58 | good | 1.05 | good | 1.04 |
| 12 | 57 | 57 | fair | 1.05 | fair | 1.11 |
| 13 | 56 | 57 | good | 1.01 | good | 1.02 |
| 14 | 56 | 57 | good | 1.05 | good | 1.09 |
| 15 | 56 | 57 | good | 1.06 | good | 1.06 |
| 16 | 57 | 58 | good | 1.05 | good | 1.02 |
| 17 | 56 | 58 | good | 0.96 | fair | 0.96 |
| 18 | 56 | 57 | good | 1.08 | good | 1.09 |
| 19 | 57 | 57 | good | 1.06 | good | 1.11 |
| 20 | 57 | 57 | good | 0.97 | good | 1.00 |
| 21 | 57 | 58 | good | 1.04 | good | 1.07 |
| 22 | 56 | 57 | good | 1.07 | good | 1.12 |
| 23 | 57 | 58 | good | 1.05 | good | 1.06 |

### File 0018

| Profile | n_pyturb | n_cli | eps_1 | ratio | eps_2 | ratio |
|---------|----------|-------|-------|-------|-------|-------|
| 0 | 38 | 38 | good | 0.96 | good | 0.98 |
| 1 | 57 | 58 | good | 1.06 | good | 1.07 |
| 2 | 57 | 58 | good | 1.10 | good | 1.02 |
| 3 | 57 | 57 | good | 0.97 | good | 1.03 |
| 4 | 56 | 57 | good | 0.99 | good | 1.01 |
| 5 | 58 | 59 | good | 0.98 | good | 0.94 |
| 6 | 56 | 57 | good | 0.96 | good | 1.08 |
| 7 | 57 | 57 | good | 1.06 | good | 0.99 |
| 8 | 57 | 59 | good | 1.03 | good | 1.06 |

### File 0019

| Profile | n_pyturb | n_cli | eps_1 | ratio | eps_2 | ratio |
|---------|----------|-------|-------|-------|-------|-------|
| 0 | 57 | 58 | good | 1.05 | good | 1.07 |
| 1 | 56 | 57 | good | 1.06 | good | 1.08 |
| 2 | 57 | 58 | good | 1.06 | good | 0.99 |
| 3 | 56 | 57 | good | 1.07 | good | 1.07 |
| 4 | 56 | 58 | good | 1.06 | good | 1.05 |
| 5 | 56 | 57 | good | 1.03 | good | 1.02 |
| 6 | 57 | 58 | good | 1.11 | good | 1.04 |
| 7 | 56 | 57 | good | 1.05 | good | 1.05 |
| 8 | 57 | 57 | good | 1.09 | good | 1.04 |
| 9 | 57 | 58 | good | 1.05 | good | 1.06 |
| 10 | 57 | 58 | good | 1.12 | good | 1.06 |
| 11 | 57 | 58 | good | 1.08 | good | 1.10 |
| 12 | 57 | 58 | good | 1.03 | good | 1.06 |
| 13 | 59 | 60 | good | 1.05 | good | 1.05 |
| 14 | 57 | 57 | fair | 1.06 | good | 1.03 |
| 15 | 57 | 58 | good | 1.06 | good | 1.12 |
| 16 | 57 | 57 | good | 1.08 | good | 1.07 |
| 17 | 57 | 57 | good | 1.02 | good | 1.02 |
| 18 | 57 | 59 | good | 1.09 | good | 1.09 |
| 19 | 58 | 59 | good | 1.04 | good | 1.00 |
| 20 | 59 | 59 | good | 1.04 | good | 1.05 |
| 21 | 57 | 58 | good | 1.06 | good | 1.06 |
| 22 | 57 | 58 | good | 1.05 | good | 1.06 |
| 23 | 57 | 58 | good | 1.06 | good | 1.08 |
| 24 | 57 | 58 | good | 1.08 | good | 1.10 |
| 25 | 57 | 59 | good | 1.08 | good | 1.06 |
| 26 | 58 | 59 | good | 1.09 | good | 1.05 |
| 27 | 56 | 57 | good | 1.08 | good | 1.06 |

### File 0020

*2 profile(s) in pyturb only (not in pyturb-cli).*

| Profile | n_pyturb | n_cli | eps_1 | ratio | eps_2 | ratio |
|---------|----------|-------|-------|-------|-------|-------|
| 0 | 56 | 57 | good | 1.05 | good | 1.04 |
| 1 | 56 | 57 | good | 1.06 | good | 1.06 |
| 2 | 57 | 57 | good | 1.08 | good | 1.04 |
| 3 | 57 | 58 | good | 1.01 | good | 1.03 |
| 4 | 56 | 57 | good | 1.08 | good | 1.08 |
| 7 | 56 | 58 | good | 1.06 | good | 1.06 |

### File 0022

| Profile | n_pyturb | n_cli | eps_1 | ratio | eps_2 | ratio |
|---------|----------|-------|-------|-------|-------|-------|
| 0 | 57 | 58 | good | 1.05 | good | 1.01 |
| 1 | 57 | 58 | good | 1.03 | good | 1.02 |
| 2 | 57 | 57 | good | 1.04 | good | 0.97 |
| 3 | 57 | 57 | good | 0.94 | good | 1.00 |
| 4 | 47 | 48 | good | 1.09 | good | 1.11 |
| 5 | 57 | 57 | good | 1.01 | good | 1.05 |
| 6 | 58 | 58 | good | 1.01 | good | 1.02 |
| 7 | 58 | 58 | good | 1.02 | good | 0.94 |
| 8 | 58 | 60 | good | 1.06 | good | 1.05 |
| 9 | 57 | 58 | good | 1.03 | good | 1.04 |
| 10 | 57 | 58 | good | 1.06 | good | 1.05 |
| 11 | 56 | 56 | good | 0.98 | good | 1.05 |
| 12 | 57 | 58 | good | 1.03 | good | 1.09 |
| 13 | 57 | 57 | good | 1.05 | good | 1.00 |
| 14 | 56 | 57 | good | 1.03 | good | 1.00 |
| 15 | 59 | 59 | good | 1.01 | good | 1.03 |
| 16 | 58 | 58 | good | 1.02 | good | 1.04 |
| 17 | 57 | 58 | good | 1.02 | good | 1.10 |
| 18 | 57 | 57 | good | 1.02 | good | 1.06 |
| 19 | 57 | 57 | good | 1.02 | good | 1.02 |
| 20 | 57 | 57 | good | 1.01 | good | 0.99 |
| 21 | 59 | 60 | good | 1.02 | good | 1.00 |
| 22 | 56 | 56 | good | 1.05 | good | 1.06 |
| 23 | 57 | 59 | good | 0.97 | good | 1.02 |

### File 0023

| Profile | n_pyturb | n_cli | eps_1 | ratio | eps_2 | ratio |
|---------|----------|-------|-------|-------|-------|-------|
| 0 | 57 | 58 | good | 1.00 | good | 1.02 |
| 1 | 56 | 57 | good | 1.04 | good | 1.02 |
| 2 | 56 | 57 | good | 1.04 | good | 1.05 |
| 3 | 56 | 57 | good | 1.08 | good | 1.02 |
| 4 | 57 | 58 | good | 1.04 | good | 1.06 |
| 5 | 57 | 58 | good | 1.05 | good | 1.06 |
| 6 | 58 | 59 | good | 1.07 | good | 1.03 |
| 7 | 57 | 57 | fair | 1.05 | fair | 1.17 |
| 8 | 57 | 57 | good | 1.03 | fair | 1.03 |

### File 0024

| Profile | n_pyturb | n_cli | eps_1 | ratio | eps_2 | ratio |
|---------|----------|-------|-------|-------|-------|-------|
| 0 | 57 | 57 | good | 0.98 | good | 1.10 |
| 1 | 57 | 59 | good | 1.02 | good | 1.03 |
| 2 | 57 | 58 | good | 1.03 | good | 1.04 |
| 3 | 57 | 58 | good | 1.04 | good | 1.04 |
| 4 | 56 | 57 | good | 1.01 | good | 0.97 |
| 5 | 56 | 58 | good | 1.06 | good | 1.05 |
| 6 | 56 | 58 | good | 1.05 | good | 1.05 |
| 7 | 56 | 58 | good | 1.06 | good | 1.06 |
| 8 | 57 | 58 | good | 1.06 | good | 1.06 |
| 9 | 57 | 58 | good | 1.06 | good | 1.08 |
| 10 | 57 | 58 | good | 1.06 | good | 1.03 |
| 11 | 57 | 58 | good | 1.03 | good | 1.03 |
| 12 | 58 | 58 | good | 1.00 | good | 1.01 |
| 13 | 62 | 63 | good | 1.07 | good | 1.09 |
| 14 | 58 | 59 | good | 1.04 | good | 1.02 |
| 15 | 57 | 59 | good | 1.04 | good | 1.02 |
| 16 | 57 | 58 | good | 1.09 | good | 1.04 |
| 17 | 57 | 58 | good | 1.06 | good | 1.07 |
| 18 | 56 | 57 | good | 1.02 | good | 1.03 |
| 19 | 56 | 57 | good | 1.02 | good | 1.04 |
| 20 | 57 | 58 | good | 1.03 | good | 1.09 |
| 21 | 57 | 59 | good | 1.05 | good | 1.06 |
| 22 | 57 | 58 | good | 1.05 | good | 1.09 |
| 23 | 57 | 57 | good | 1.09 | good | 1.10 |
| 24 | 61 | 61 | good | 1.03 | good | 1.03 |
| 25 | 56 | 57 | good | 1.09 | good | 1.03 |
| 26 | 56 | 57 | good | 1.06 | good | 0.99 |

### File 0025

| Profile | n_pyturb | n_cli | eps_1 | ratio | eps_2 | ratio |
|---------|----------|-------|-------|-------|-------|-------|
| 0 | 56 | 57 | good | 1.07 | good | 1.11 |
| 1 | 57 | 57 | good | 0.99 | good | 1.02 |
| 2 | 57 | 58 | good | 1.08 | good | 1.07 |
| 3 | 57 | 58 | good | 1.07 | good | 1.05 |
| 4 | 57 | 58 | good | 1.06 | good | 1.06 |
| 5 | 26 | 26 | good | 1.08 | good | 1.01 |
| 6 | 57 | 57 | good | 1.00 | good | 1.03 |
| 7 | 56 | 57 | good | 1.03 | good | 1.06 |
| 8 | 57 | 58 | good | 1.06 | good | 1.07 |
| 9 | 57 | 58 | good | 1.06 | good | 1.06 |
| 10 | 57 | 57 | good | 0.99 | good | 1.00 |
| 11 | 56 | 58 | good | 1.04 | good | 1.08 |
| 12 | 56 | 58 | good | 1.03 | good | 1.06 |
| 13 | 56 | 58 | good | 0.99 | good | 0.96 |

### File 0026

| Profile | n_pyturb | n_cli | eps_1 | ratio | eps_2 | ratio |
|---------|----------|-------|-------|-------|-------|-------|
| 0 | 63 | 63 | good | 1.09 | good | 1.05 |
| 1 | 57 | 57 | good | 1.05 | good | 1.04 |
| 2 | 57 | 59 | good | 1.04 | good | 1.05 |
| 3 | 57 | 58 | good | 1.07 | good | 1.06 |
| 4 | 57 | 58 | good | 1.02 | good | 1.01 |
| 5 | 57 | 57 | good | 1.06 | good | 1.03 |
| 6 | 56 | 57 | good | 1.06 | good | 1.05 |
| 7 | 57 | 57 | good | 1.05 | good | 1.07 |
| 8 | 57 | 58 | good | 1.05 | good | 0.98 |
| 9 | 57 | 58 | good | 1.08 | good | 1.06 |

### File 0027

| Profile | n_pyturb | n_cli | eps_1 | ratio | eps_2 | ratio |
|---------|----------|-------|-------|-------|-------|-------|
| 0 | 57 | 57 | good | 1.09 | good | 1.06 |

### File 0028

*3 profile(s) in pyturb only (not in pyturb-cli).*

| Profile | n_pyturb | n_cli | eps_1 | ratio | eps_2 | ratio |
|---------|----------|-------|-------|-------|-------|-------|
| 3 | 53 | 54 | good | 1.07 | good | 1.12 |
| 4 | 57 | 57 | good | 1.02 | good | 1.00 |
| 5 | 56 | 57 | good | 1.04 | good | 0.99 |
| 6 | 57 | 59 | good | 0.97 | good | 1.00 |
| 7 | 57 | 58 | good | 0.97 | good | 1.02 |
| 8 | 57 | 57 | good | 1.01 | good | 1.02 |
| 9 | 56 | 58 | good | 1.07 | good | 1.03 |
| 10 | 56 | 57 | good | 1.06 | good | 1.07 |
| 11 | 57 | 57 | good | 1.09 | good | 1.13 |
| 12 | 58 | 59 | good | 1.06 | good | 1.06 |
| 13 | 57 | 58 | good | 1.04 | good | 1.10 |
| 14 | 56 | 57 | good | 1.11 | good | 1.09 |
| 15 | 57 | 58 | good | 1.01 | good | 1.05 |
| 16 | 58 | 59 | good | 1.07 | good | 1.08 |
| 17 | 57 | 57 | good | 1.05 | good | 1.04 |
| 18 | 56 | 57 | good | 1.07 | good | 1.05 |
| 19 | 57 | 58 | good | 1.08 | good | 1.09 |
| 20 | 57 | 58 | good | 1.05 | good | 0.99 |
| 21 | 57 | 58 | good | 1.11 | good | 1.08 |
| 22 | 57 | 59 | good | 1.08 | good | 1.00 |
| 23 | 57 | 59 | good | 1.05 | good | 1.02 |
| 24 | 58 | 59 | good | 1.05 | good | 1.04 |
| 25 | 57 | 57 | good | 1.00 | good | 0.99 |
| 26 | 53 | 54 | good | 0.96 | good | 1.01 |
| 27 | 17 | 18 | fair | 1.16 | fair | 1.08 |
| 28 | 20 | 21 | good | 1.01 | good | 1.07 |

### File 0029

| Profile | n_pyturb | n_cli | eps_1 | ratio | eps_2 | ratio |
|---------|----------|-------|-------|-------|-------|-------|
| 0 | 21 | 22 | good | 0.93 | good | 0.95 |
| 1 | 57 | 58 | good | 1.02 | good | 1.03 |
| 2 | 57 | 58 | good | 1.00 | good | 1.00 |
| 3 | 57 | 58 | good | 1.04 | good | 1.08 |
| 4 | 57 | 57 | good | 0.96 | good | 0.98 |
| 5 | 57 | 58 | good | 1.04 | good | 1.03 |
| 6 | 57 | 58 | good | 1.10 | good | 1.07 |
| 7 | 57 | 58 | good | 0.96 | good | 0.94 |
| 8 | 57 | 58 | good | 1.06 | good | 1.07 |
| 9 | 57 | 59 | good | 1.03 | good | 1.03 |
| 10 | 57 | 59 | good | 1.06 | good | 1.01 |
| 11 | 56 | 57 | good | 1.01 | good | 0.96 |
| 12 | 57 | 58 | good | 1.07 | good | 1.05 |
| 13 | 69 | 70 | good | 1.03 | good | 1.02 |
| 14 | 57 | 58 | good | 1.07 | good | 0.99 |

### File 0030

| Profile | n_pyturb | n_cli | eps_1 | ratio | eps_2 | ratio |
|---------|----------|-------|-------|-------|-------|-------|
| 0 | 57 | 58 | good | 1.07 | good | 1.09 |
| 1 | 57 | 59 | good | 1.06 | good | 1.03 |
| 2 | 58 | 59 | good | 1.09 | good | 1.06 |
| 3 | 57 | 57 | good | 1.06 | good | 1.03 |
| 4 | 57 | 58 | good | 1.04 | good | 1.05 |
| 5 | 57 | 59 | good | 1.05 | good | 1.05 |
| 6 | 57 | 58 | good | 1.04 | good | 1.05 |
| 7 | 57 | 57 | good | 1.05 | good | 1.05 |
| 8 | 56 | 58 | good | 1.08 | good | 1.03 |
| 9 | 57 | 57 | good | 1.09 | good | 1.09 |
| 10 | 57 | 58 | good | 1.09 | good | 1.10 |
| 11 | 57 | 58 | good | 1.08 | good | 1.08 |
| 12 | 57 | 58 | good | 1.11 | good | 1.07 |
| 13 | 57 | 58 | good | 1.07 | good | 1.06 |
| 14 | 57 | 57 | good | 1.01 | good | 1.01 |
| 15 | 57 | 58 | good | 1.00 | good | 1.06 |
| 16 | 57 | 58 | good | 1.07 | good | 1.06 |
| 17 | 57 | 58 | good | 1.01 | good | 0.99 |
| 18 | 57 | 58 | good | 1.06 | good | 1.05 |
| 19 | 57 | 58 | good | 1.09 | good | 1.07 |

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

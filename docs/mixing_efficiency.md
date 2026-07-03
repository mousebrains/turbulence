# Mixing Efficiency

## Overview

This package computes and plots (in the `rsi-tpw dl` viewer) the
**dissipation ratio** of the two directly measured dissipation rates:

$$\lambda = \frac{\chi}{\varepsilon}$$

where

| Symbol | Quantity | Units |
|--------|----------|-------|
| $\chi$ | Thermal variance dissipation rate | K$^2$ s$^{-1}$ |
| $\varepsilon$ | TKE dissipation rate | W kg$^{-1}$ = m$^2$ s$^{-3}$ |
| $\lambda = \chi/\varepsilon$ | Dissipation ratio | K$^2$ s$^2$ m$^{-2}$ |

Note that $\chi/\varepsilon$ is **dimensional** (K$^2$ s$^2$ m$^{-2}$),
so it cannot be compared with the canonical dimensionless mixing
coefficient $\Gamma \approx 0.2$ of Osborn (1980).  The dissipation
ratio is a useful diagnostic of the relative strength of thermal
versus mechanical dissipation in a profile, but its magnitude depends
on the background temperature gradient and stratification and has no
universal threshold value.

## The Dimensionless Mixing Coefficient

The proper dimensionless mixing coefficient (flux coefficient) is
obtained by combining the Osborn--Cox and Osborn models below
([Oakey 1982](https://doi.org/10.1175/1520-0485(1982)012%3C0256:DOTROD%3E2.0.CO;2)):

$$\Gamma = \frac{N^2\,\chi}{2\,\varepsilon\,\left(\partial \overline{T}/\partial z\right)^2}$$

where $N^2$ is the squared buoyancy frequency and
$\partial \overline{T}/\partial z$ is the background (mean) vertical
temperature gradient.  This quantity is dimensionless and is the one
for which the canonical value $\Gamma \approx 0.2$ applies.

**The package computes $\Gamma$ in both processing pipelines** — the
`rsi-tpw pipeline` Method-1 chi output (`L4_chi_epsilon.nc`, propagated
into the binned `L5_binned.nc` and combined `L6_combined.nc` products)
and the `perturb` chi stage (`chi.mixing: true`, the default; written
into the per-profile chi NetCDFs).  In the perturb path, practical
salinity comes from the profile's own conductivity/temperature/pressure
via TEOS-10, so $N^2$ is fully constrained; the rsi path may assume
35 PSU (caveat 1 below).  The supporting quantities, all on the chi
window grid:

| Variable | Definition | Units |
|----------|------------|-------|
| `N2` | TEOS-10 buoyancy frequency squared between the shallow- and deep-half means of each window | s$^{-2}$ |
| `dTdz` | Least-squares background temperature gradient vs depth | K m$^{-1}$ |
| `K_T` | Osborn–Cox heat diffusivity $\chi/(2(\partial\overline{T}/\partial z)^2)$ | m$^2$ s$^{-1}$ |
| `Gamma` | Measured mixing coefficient (Oakey 1982) | 1 |
| `K_rho` | Osborn diffusivity $0.2\,\varepsilon/N^2$ with the canonical constant | m$^2$ s$^{-1}$ |

Implementation: [`odas_tpw.processing.mixing`](../src/odas_tpw/processing/mixing.py).
Estimates are masked (NaN) **per variable**, mirroring which gradient each
one divides by: `K_T` where $|\partial\overline{T}/\partial z| < 10^{-4}$ K/m
(well-mixed — the temperature-variance budget no longer constrains a
diffusivity); `K_rho` where $N^2 < 10^{-9}$ s$^{-2}$ (unstratified — the
Osborn scaling does not apply); and `Gamma` where *either* floor is crossed
(it divides by both). So `K_T` survives in unstratified water and `K_rho`
survives in well-mixed water — only their intersection removes both.
`K_rho` is additionally masked where it exceeds an upper sanity bound
(`K_rho_max`, default $10$ m$^2$ s$^{-1}$): as $N^2$ approaches the floor
the Osborn diffusivity grows without bound (tens to thousands of
m$^2$ s$^{-1}$), an artifact rather than real mixing. The bound sits above
even the most energetic real mixing — overflows and hydraulic jumps reach
$\sim 0.1$–$1$ m$^2$ s$^{-1}$, and energetic near-surface mixing on ARCTERX
VMP casts reaches a few m$^2$ s$^{-1}$ — so genuine signal is retained and
only physically implausible magnitudes are removed: the unbounded
near-floor-$N^2$ artifact, or contaminated near-surface windows whose
epsilon is itself spurious. The number of masked windows is reported via a
warning.

`K_T` and `Gamma` carry their **own** upper ceilings for the same reason,
gated per variable rather than wherever `K_rho` is masked (they blow up in
different regimes): `K_T` above `K_T_max` (default $10$ m$^2$ s$^{-1}$; a weak
$\partial\overline{T}/\partial z$ with a contaminated $\chi$ can push the
thermal diffusivity past any real value even where $N^2$ is fine), and `Gamma`
above `Gamma_max` (default $5$; the measured mixing coefficient is $\sim 0.2$
canonically and rarely exceeds $\sim 1$–$2$, so values of tens to hundreds —
produced when a near-zero shear $\varepsilon$ lands in the denominator — are
artifacts). Gating each on its own ceiling keeps a legitimately large `K_T` in
low-$N^2$ water instead of discarding it with `K_rho`.

Two caveats for interpretation:

1. When no salinity is supplied to the pipeline, $N^2$ assumes a uniform
   35 PSU and reflects temperature stratification only — it can be badly
   wrong where salinity stratification matters (the `N2` variable's
   `comment` attribute records which case applies).
2. Because $\Gamma$ is *derived from* the measured $\chi$ and
   $\varepsilon$, the identity $\Gamma\,\varepsilon/N^2 \equiv K_T$
   holds by construction.  `K_rho` therefore uses the canonical
   $\Gamma_0 = 0.2$, so comparing `K_rho` with `K_T` (equivalently,
   `Gamma` with 0.2) is a meaningful consistency check rather than a
   tautology.

## Connection to Turbulent Diffusivity

### Osborn--Cox model

The turbulent thermal diffusivity $K_T$ relates the thermal variance
dissipation rate to the mean temperature gradient (Osborn & Cox, 1972):

$$K_T = \frac{\chi}{2\,\left(\partial \overline{T}/\partial z\right)^2}$$

$K_T$ measures the effective diffusivity that turbulence imposes on the
mean temperature field, analogous to the molecular thermal diffusivity
$\kappa_T$ but typically orders of magnitude larger.

### Osborn model

The Osborn (1980) model relates the turbulent (eddy) diffusivity of
density $K_\rho$ to the TKE dissipation rate and the buoyancy
frequency:

$$K_\rho = \Gamma\,\frac{\varepsilon}{N^2}$$

where $N^2 = -(g/\rho)\,\partial\rho/\partial z$ is the squared
buoyancy frequency and $\Gamma$ is the **mixing efficiency** (also
called the flux coefficient).  In steady, homogeneous turbulence the
production of turbulent kinetic energy balances the sum of viscous
dissipation $\varepsilon$ and the buoyancy flux $J_b$:

$$P = \varepsilon + J_b$$

The flux Richardson number is

$$R_f = \frac{J_b}{P} = \frac{J_b}{\varepsilon + J_b}$$

so that

$$\Gamma = \frac{R_f}{1 - R_f}$$

The canonical value $\Gamma = 0.2$ (equivalently $R_f \approx 0.17$)
is widely used as a default, but observational and numerical studies
show that $\Gamma$ varies with turbulence intensity, stratification,
and the nature of the forcing.

Equating $K_T$ (Osborn--Cox) with $K_\rho$ (Osborn) yields the Oakey
(1982) expression for $\Gamma$ given above.

## Ratio of Kinetic to Thermal Energy Dissipation

The dissipation ratio can also be viewed from an energy-budget
perspective.  Turbulence dissipates kinetic energy at rate
$\varepsilon$ through the strain-rate tensor, while it dissipates
thermal variance at rate $\chi$ through molecular thermal diffusion
acting on temperature gradients at the smallest scales:

$$\varepsilon = 2\,\nu\,\langle s_{ij}\,s_{ij}\rangle,
  \qquad
  \chi = 2\,\kappa_T\,\langle |\nabla T'|^2 \rangle$$

where $\nu$ is kinematic viscosity and $\kappa_T$ is molecular thermal
diffusivity.  The dimensional ratio $\chi/\varepsilon$ reflects the
relative importance of thermal vs mechanical dissipation; its
interpretation depends on the Prandtl number $Pr = \nu/\kappa_T$, the
turbulence intensity (buoyancy Reynolds number
$Re_b = \varepsilon/(\nu N^2)$), and the background temperature
gradient.

## References

Key references: Osborn (1980), Osborn & Cox (1972), Oakey (1982),
Gregg et al. (2018), Mater & Venayagamoorthy (2014).
See [Bibliography](bibliography.md) for full citations.

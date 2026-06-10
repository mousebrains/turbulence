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

**This package does not currently compute $\Gamma$**: doing so requires
the background stratification ($N^2$) and mean temperature gradient in
addition to $\chi$ and $\varepsilon$.  Only the dimensional ratio
$\chi/\varepsilon$ is produced.

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

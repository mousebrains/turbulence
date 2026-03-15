# Mixing Efficiency

## Overview

The mixing efficiency parameter $\Gamma$ quantifies how effectively
turbulent kinetic energy is converted into irreversible mixing of the
temperature (or density) field.  In this package it is defined as the
ratio of the thermal variance dissipation rate to the TKE dissipation
rate:

$$\Gamma = \frac{\chi}{\varepsilon}$$

where

| Symbol | Quantity | Units |
|--------|----------|-------|
| $\chi$ | Thermal variance dissipation rate | K$^2$ s$^{-1}$ |
| $\varepsilon$ | TKE dissipation rate | W kg$^{-1}$ = m$^2$ s$^{-3}$ |
| $\Gamma$ | Dissipation ratio | K$^2$ s$^2$ m$^{-2}$ |

A threshold of $\Gamma = 0.2$ separates two regimes:

- $\Gamma > 0.2$: **highly efficient mixing** -- a large fraction of
  turbulent energy goes into irreversible stirring of the temperature
  field.
- $\Gamma < 0.2$: **less efficient mixing** -- turbulent kinetic
  energy is dissipated viscously with relatively little thermal mixing.

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

## Ratio of Kinetic to Thermal Energy Dissipation

The ratio can also be viewed from an energy-budget perspective.
Turbulence dissipates kinetic energy at rate $\varepsilon$ through the
strain-rate tensor, while it dissipates thermal variance at rate
$\chi$ through molecular thermal diffusion acting on temperature
gradients at the smallest scales:

$$\varepsilon = 2\,\nu\,\langle s_{ij}\,s_{ij}\rangle,
  \qquad
  \chi = 2\,\kappa_T\,\langle |\nabla T'|^2 \rangle$$

where $\nu$ is kinematic viscosity and $\kappa_T$ is molecular thermal
diffusivity.  Their ratio $\Gamma = \chi/\varepsilon$ reflects the
relative importance of thermal vs mechanical dissipation and depends on
the Prandtl number $Pr = \nu/\kappa_T$ and the turbulence intensity
(buoyancy Reynolds number $Re_b = \varepsilon/(\nu N^2)$).

## References

Key references: Osborn (1980), Osborn & Cox (1972), Gregg et al. (2018),
Mater & Venayagamoorthy (2014).
See [Bibliography](bibliography.md) for full citations.

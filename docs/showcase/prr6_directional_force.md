# Sensing a directional noise bath (PRR 6, 2024)

J. M. H. Gosling, A. Pontin, J. H. Iacoponi, P. F. Barker and T. S. Monteiro, *Sensing directional
noise baths in levitated optomechanics*, Phys. Rev. Research **6**, 013129 (2024),
[doi:10.1103/PhysRevResearch.6.013129](https://doi.org/10.1103/PhysRevResearch.6.013129).

:::{figure} ../guide/figures/schematics/prr6_setup_b.png
:width: 70%
:alt: A nanoparticle in a tweezer between four electrodes

The particle is trapped at the focus of an aspheric lens (NA 0.77) between four electrodes, which
apply a white-noise force along a chosen direction $\psi$ in the $x$-$y$ plane.
:::

:::{figure} ../guide/figures/schematics/prr6_setup_a.png
:width: 100%
:alt: The optical table of the experiment

The table: the trapping laser and its polarisation optics, the tweezer and electrodes, and the
balanced detection of $x$ and $y$.
:::

A stream of impulses too weak to detect one by one, arriving from a fixed direction, acts on a
levitated particle as a white-noise force along that direction. Its power spectral densities only
show it as extra heating, which must be separated from the thermal part by a calibration, and which is
the same for $\psi$ and $-\psi$. In this paper we showed that the $x$-$y$ cross-correlation spectrum is a
calibration-free signature of such a force and of its quadrant, since uncorrelated thermal noise
leaves $S_{xy}$ at zero on average. I simulated all of it in levisim: the directed force enters the
stochastic Runge-Kutta integration of `FreeSpaceSimulation` as one random force shared between $x$ and $y$ in
proportion to $\cos\psi$ and $\sin\psi$, and I worked out the error analysis of the spectral estimates in
the paper's appendix.

```{raw} html
:file: ../_generated/showcase/prr6_directional_force.html
```

The default free-space `Trap` (485 mW at 1064 nm, a 60.1 nm sphere, $4 \times 10^{-3}$ mbar, feedback
damping $\Gamma_{\text{fb},x}/2\pi = 2.1$ kHz and $\Gamma_{\text{fb},y}/2\pi = 2.5$ kHz) with a directed force
twice the thermal force PSD ($\beta = \sqrt{2}$). Points are one seeded 40 ms simulation per angle,
averaged over 40 segments; lines are `SpectralDensityTheory`. **(a)** $S_{xx}$ and $S_{yy}$, one-sided:
the directed force raises them above the thermal-only spectra (shaded) by the same amount for
$\psi = +45°$ and $-45°$. **(b)** The co-spectrum $C_{xy} = \mathrm{Re}\,S_{xy}$ changes sign with $\psi$:
its lobes, $\pm 4.3 \times 10^{-23}$ m²/Hz near each resonance ($\Omega_y/2\pi = 135.8$ kHz,
$\Omega_x/2\pi = 156.3$ kHz), tell the quadrant of the force directly.

The essential lines:

```{literalinclude} ../guide/scripts/showcase/prr6_directional_force.py
:language: python
:pyobject: compute
```

The model, the Kasdin noise weights of the stochastic integrator, and the estimator options are in
the guide page [Stochastic simulations of a bead in a free-space tweezer](../guide/stochastic_simulations.md).

:::{figure} ../guide/figures/showcase/prr6_directional_force.png
:width: 100%
:alt: PSDs and co-spectra for two force directions

The static version of the figure, as shown in the README.
:::

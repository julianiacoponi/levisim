# levisim

`levisim` simulates levitated optomechanics. Its core is the quantum linear theory (QLT) of
several optically levitated nanoparticles coupled by coherent scattering to one shared cavity
mode: given the traps, the gas and the cavity, it calculates the homodyne and heterodyne spectra
of the cavity output and the power and cross spectral densities of every particle's motion, by
any of seven calculation methods that can be checked against each other. Around that core it has
a stochastic simulator of a bead in a free-space tweezer, covariance-matrix tools for
steady-state entanglement under a two-tone drive, and a Fortran reference implementation that the
Python spectra are tested against.

:::{figure} guide/figures/quickstart_homodyne.png
:width: 80%
:alt: Homodyne spectrum of two particles in one cavity

Homodyne spectrum of the cavity output for two particles moving in $x$ and $y$
(the [quickstart](getting_started.md#quickstart)), with each motion's bare trap frequency
$\Omega_{j_n}$ and the four normal modes marked; below it, the displacement spectrum of each motion.
:::

```{toctree}
:maxdepth: 2
:caption: Using levisim

getting_started
interactive_labs
showcase/index
concepts/conventions
concepts/calculation_methods
concepts/stability
concepts/static_cavity_shifts
concepts/interactions
concepts/particles_and_materials
concepts/traps_as_fields
concepts/rotation
guide/index
notebooks
```

```{toctree}
:maxdepth: 2
:caption: Reference

api
```

```{toctree}
:maxdepth: 1
:caption: Development

testing
contributing
provenance
```

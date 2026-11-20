# A mechanical mode comb and its collective bright mode (PRR 8, 2026)

J. H. Iacoponi, M. Rademacher and T. S. Monteiro, *Nanoparticle arrays levitated in a cavity for
quantum sensing*, Phys. Rev. Research **8**, 033286 (2026),
[doi:10.1103/wrd3-t5cf](https://doi.org/10.1103/wrd3-t5cf).

:::{figure} ../guide/figures/schematics/prr8_array.png
:width: 100%
:alt: An array of tweezers along one cavity mode

$N$ tweezers along one cavity mode, each with its own phase $\Phi_n$, trap particles whose motions
have frequencies $\Omega_j$.
:::

When many particles share one cavity mode, the light couples every motion to every other. In this
paper we treated such arrays with the quantum linear theory, found closed forms
for the spectra in terms of the tweezer phases, and showed that a comb of evenly spaced mechanical
frequencies is reorganised by the cavity into one collective bright mode, which takes almost all of
the coupling to the light, and a comb of nearly dark modes, which stay close to the bare frequencies
and keep their narrow linewidths. levisim calculates everything in the paper; its analytic methods
(6) and (7) are the paper's closed forms.

```{raw} html
:file: ../_generated/showcase/prr8_mode_comb.html
```

$N$ particles, each moving in $x$ and $y$, at the node of the default cavity
($\Delta/2\pi = -450$ kHz, $\kappa/2\pi = 390$ kHz, $10^{-6}$ mbar), with $2N$ mechanical frequencies spaced
6.67 kHz apart about 300 kHz and $g/2\pi = 40$ kHz for every motion (toy parameters, as in the paper's
Fig. 1). **(a)** The heterodyne spectrum of the cavity output for $N = 8$: the comb of 16 narrow peaks
at the bare frequencies, and the bright mode as a broad feature far below them. **(b)** Every
mechanical eigenfrequency of the drift matrix against $N$, coloured by its linewidth $\Gamma/2\pi$
($-2\,\mathrm{Re}\,\lambda/2\pi$); the grey bars are the spans of the bare frequencies. The bright mode
moves from 286.7 kHz for one particle to 156.9 kHz for eight, broadening from 17.4 kHz to 90.8 kHz,
while the narrowest dark mode narrows from 0.84 kHz to 0.42 kHz. All eight systems are linearly
stable; with these couplings the bright mode reaches zero frequency (a static instability) for $N = 12$.

The essential lines:

```{literalinclude} ../guide/scripts/showcase/prr8_mode_comb.py
:language: python
:pyobject: array
```

```{literalinclude} ../guide/scripts/showcase/prr8_mode_comb.py
:language: python
:pyobject: mechanical_modes
```

The paper's figures themselves are recomputed in the research notebook
`notebooks/multiphysics/mechanical-mode-comb/reproduce_Figs_1_3_4_5.ipynb`.

:::{figure} ../guide/figures/showcase/prr8_mode_comb.png
:width: 100%
:alt: Heterodyne spectrum of an eight-particle comb and its eigenfrequencies

The static version of the figure, as shown in the README.
:::

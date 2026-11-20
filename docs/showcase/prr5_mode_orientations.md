# Mode orientations and the cancellation point (PRR 5, 2023)

A. Pontin, H. Fu, J. H. Iacoponi, P. F. Barker and T. S. Monteiro, *Controlling mode orientations and
frequencies in levitated cavity optomechanics*, Phys. Rev. Research **5**, 013013 (2023),
[doi:10.1103/PhysRevResearch.5.013013](https://doi.org/10.1103/PhysRevResearch.5.013013).

:::{figure} ../guide/figures/schematics/prr5_geometry.png
:width: 85%
:alt: Geometry of a nanosphere trapped on the cavity standing wave

The tweezer's polarisation makes an angle $\theta$ with the cavity axis $Y_\text{c}$, and the
particle sits at a phase $\phi$ on the standing wave. The cavity backaction rotates the $x$-$y$ normal
modes one way ($\Theta_\text{dyn}$), the static coherent-scattering field the other
($\Theta_\text{CS}$).
:::

A nanosphere in a tweezer inside a cavity has two transverse motions, $x$ and $y$, which the light
couples in two ways. Through the cavity's dynamics, each motion scatters photons into the cavity mode
with a coupling $g_x$ or $g_y \propto \sin\Phi$, which mixes them into normal modes that are rotated
away from the tweezer axes; this is the route by which a bright mode and a dark mode form. The static
part of the coherent-scattering field, which builds up in the cavity when the particle is not at a
node, adds a direct coupling $g_{xy} \propto A_0 \cos\Phi$ that rotates the modes the opposite way.
In this paper we measured the $x$-$y$ cross-correlation spectrum $S_{xy}(\omega)$ as the particle was
moved along the standing wave, and found the point at which the two rotations cancel: there the modes
keep their unperturbed orientation, $S_{xy}$ vanishes, and the particle is still cavity-cooled.
My part of the work was the analysis of that cancellation point, including the map of the mode
rotation over the trap position and the detuning, which is what the figure below recomputes.

```{raw} html
:file: ../_generated/showcase/prr5_mode_orientations.html
```

**(a)** $\mathrm{Re}\,S_{xy}(\omega)$ for one 60.1 nm sphere in levisim's default 1064 nm tweezer
($\theta = 45°$), with the default cavity ($\Delta/2\pi = -450$ kHz, $\kappa/2\pi = 390$ kHz) and gas
($10^{-6}$ mbar), at trap positions from the node ($\Phi = \pi/2$) towards the antinode, each curve
scaled to its own largest value. The $x$ and $y$ motions are always anticorrelated (peaks of opposite
sign at $\Omega_y/2\pi \approx 134$ kHz and $\Omega_x/2\pi \approx 155$ kHz), and the sign of the
pattern flips between $\Phi = 0.25\pi$ and $0.20\pi$. **(b)** The rotation of the $x$-like normal mode
away from the tweezer $x$ axis, from the eigenvectors of the drift matrix, over $\Phi$ and $\Delta$. The
black contour, where the rotation is zero, is the cancellation point: $\Phi_\text{c} = 0.248\pi$ at
$\Delta/2\pi = -450$ kHz and $-1000$ kHz, moving towards the node as the detuning approaches the
mechanical frequencies ($0.256\pi$ at $-300$ kHz, $0.313\pi$ at $-150$ kHz), as the paper anticipates.
It hardly depends on $\theta$ (within $10^{-3}\pi$ between 20° and 70°), since both couplings carry
the same factor $\sin\theta\cos\theta$. Every system in the map is linearly stable; the least stable is at the antinode, where the light
does not couple to $x$ and $y$ and only the gas damps them ($\mathrm{Re}\,\lambda = -3.6 \times 10^{-3}$ s⁻¹).

The essential lines:

```{literalinclude} ../guide/scripts/showcase/prr5_mode_orientations.py
:language: python
:pyobject: one_particle
```

```{literalinclude} ../guide/scripts/showcase/prr5_mode_orientations.py
:language: python
:pyobject: mode_rotations
```

**What levisim does not include.** The paper also shows that the static field stiffens the mechanical
frequencies away from the node, opposing the optical spring of the cavity dynamics, and that the two
frequency shifts cancel at the same point. levisim includes the static coupling $g_{xy}$, and so the
rotation, but not the static coherent-scattering frequency shifts, so its mode frequencies away from
the node carry only the optical spring.

:::{figure} ../guide/figures/showcase/prr5_mode_orientations.png
:width: 100%
:alt: Cross spectra and the mode rotation map

The static version of the figure, as shown in the README.
:::

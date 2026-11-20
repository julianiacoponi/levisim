# Linear stability as a tweezer phase is swept

Every system levisim models is a network of damped linear resonators, $\dot{\hat{\mathbf c}} = \mathbf A
\hat{\mathbf c} + \hat{\mathbf c}_\text{in}$, and it has a steady state only if every eigenvalue $\lambda$ of
the drift matrix $\mathbf A$ has $\mathrm{Re}\,\lambda < 0$. The stationary spectra do not show it when
this fails: they remain finite numbers. `MultiTrap` therefore reports its stability explicitly
(`is_stable`, `max_growth_rate`, `unstable_modes`, `drift_eigenvalues`) and logs a WARNING for an
unstable system. The theory is on the concepts page
[Eigenvalue stability of coupled resonators](../concepts/stability.md), and the guide page
[Linear stability of a system](../guide/stability.md) lists which ready-made systems are unstable.

```{raw} html
:file: ../_generated/showcase/stability_tweezer_phase.html
```

The README quickstart system (two particles moving in $x$ and $y$, levisim's default tweezers, beads,
cavity and gas) with the second tweezer's phase $\phi_2$ swept from 0° to 180°. With unequal phases the
cavity-mediated coupling between the particles gains a non-reciprocal part
$\propto \sin(\phi_2 - \phi_1)$, which anti-damps the hybridised pair of $y$ motions. **(a)** The
mechanical eigenvalues in the complex plane for $\phi_2$ from 0° to 90°; the eigenvalues at $\phi_2$ and
$180° - \phi_2$ are the same, so this covers the whole sweep. One mode near 135 kHz crosses into the
unstable half-plane. **(b)** The largest growth rate: the system is unstable for
$54.40° < \phi_2 < 125.60°$ (edges found by root-finding), with
$\mathrm{Re}\,\lambda = +2.68 \times 10^3$ s⁻¹ at 90°, the built-in phase of the second tweezer, which is why
the quickstart uses $\phi_2 = 30°$ ($\mathrm{Re}\,\lambda = -294$ s⁻¹).

```{literalinclude} ../guide/scripts/showcase/stability_tweezer_phase.py
:language: python
:pyobject: compute
```

:::{figure} ../guide/figures/showcase/stability_tweezer_phase.png
:width: 100%
:alt: Drift eigenvalues and growth rate against the tweezer phase

The static version of the figure, as shown in the README.
:::

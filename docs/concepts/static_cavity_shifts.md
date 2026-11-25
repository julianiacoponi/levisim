# Cavity wavenumber and static cavity shifts

Two corrections of the physical (non-toy) model are available from levisim 0.1.1. Both are
**off by default**, so every number of levisim 0.1.0 is unchanged unless you ask for them:

```python
from levisim import MultiTrap

mt = MultiTrap(
    num_traps=1, Cartesian_axes_per_trap=3,
    cavity_wavenumber=True,     # standing wave uses the cavity wavenumber k_c
    static_cavity_shifts=True,  # cotrapping + dispersive shifts of the trap frequencies
)
```

Both keywords are ignored (with a warning) for toy parameters, which have no standing wave and no
static cavity field.

## `cavity_wavenumber=True`

The cavity standing wave is $\cos(k_\mathrm{c} Y_\mathrm{c} + \Phi)$ with the cavity wavenumber
$k_\mathrm{c} = \omega_\mathrm{c}/c = (\omega_\mathrm{tw} - \Delta)/c$ (`Cavity.k`). By default levisim
uses the tweezer wavenumber $k_\mathrm{tw}$ there, a relative difference of
$|\Delta|/\omega_\mathrm{tw} \approx 6 \times 10^{-10}$ for $\Delta/2\pi = -180$ kHz at 1064 nm. With
the keyword, `Dimension.standing_wave_k` is $k_\mathrm{c}$, which enters

- the transverse couplings $g_x, g_y \propto k_\mathrm{c}$,
- the direct couplings $g_{xy} \propto k_\mathrm{c}^2$ and $g_{xz}, g_{yz} \propto k_\mathrm{c} k_z$,
- the static shifts below.

The axial factor $k_z = k_\mathrm{tw} - 1/z_\mathrm{R}$ comes from the propagation phase of the tweezer
(with its Gouy phase), so it keeps the tweezer wavenumber either way.

## `static_cavity_shifts=True`

The dipole potential of the bead in the total field is
$U = U_\mathrm{tw} + V_\mathrm{CS} + V_\mathrm{c}$
(`levisim.physics.coherent_scattering.optical_potential`). The linearised couplings use only the
off-diagonal second derivatives of the coherent-scattering term $V_\mathrm{CS}$ (they are $g_{xy}$,
$g_{xz}$, $g_{yz}$). Its diagonal second derivatives, the **cotrapping** terms, and the curvature of
the cavity's own intensity, the **dispersive** potential $V_\mathrm{c}$, add a static stiffness
$K$ (`static_cavity_hessian`), with $s, c = \sin\theta, \cos\theta$, $\hbar\omega_\mathrm{d} = E_\mathrm{d}$
and $\bar{\alpha} = (A_0 + iP_0)/2$:

$$
\begin{aligned}
K^\mathrm{CS}_{xx} &= \hbar\omega_\mathrm{d}\, s A_0 \cos\Phi\,\left(k_\mathrm{c}^2 s^2 + 2/w_x^2\right),\\
K^\mathrm{CS}_{yy} &= \hbar\omega_\mathrm{d}\, s A_0 \cos\Phi\,\left(k_\mathrm{c}^2 c^2 + 2/w_y^2\right),\\
K^\mathrm{CS}_{zz} &= \hbar\omega_\mathrm{d}\, s A_0 \cos\Phi\,\left(k_z^2 + 1/z_\mathrm{R}^2\right),\\
K^\mathrm{c}_{jk} &= 2\alpha\,\varepsilon_\mathrm{c}^2 |\bar{\alpha}|^2 k_\mathrm{c}^2 \cos 2\Phi\; d_j d_k,
\qquad \mathbf{d} = (s, c, 0).
\end{aligned}
$$

Each frequency becomes $\Omega_j = \Omega_j^\mathrm{tw}\sqrt{1 + K_{jj}/(m\,\Omega_j^{\mathrm{tw}\,2})}$, and
$K^\mathrm{c}_{xy}$ adds $-K^\mathrm{c}_{xy} x_\mathrm{zpf} y_\mathrm{zpf}/\hbar$ to $g_{xy}$. The shifted
frequency is used everywhere: it is `Dimension.Ω` (the mechanical susceptibilities and the drift
matrix), it sets the zero-point fluctuation (so every coupling scales by
$\sqrt{\Omega_j^\mathrm{tw}/\Omega_j}$), the thermal occupancy and the photon-recoil heating rate.
`Dimension.static_cavity_shift` is $\Omega_j - \Omega_j^\mathrm{tw}$.

For red detuning and a bead between the antinode and the node, $A_0\cos\Phi > 0$ and the cotrapping
stiffens every axis; the dispersive term stiffens the motion along the cavity axis on the antinode
side ($\cos 2\Phi > 0$) and softens it on the node side. Both vanish at the node, where the static
field is zero. Frequencies given as measured values (`override_Ω_dict`) are not shifted again, since
a measurement already contains the shifts.

The test `tests/test_cavity_wavenumber_static_shifts.py` checks $K$, and the shifted frequencies,
against a finite-difference Hessian of the full potential.

Not included: the static force of the cavity field (`static_cavity_force`), which displaces the
equilibrium of the bead; levisim still linearises about the trap centre.

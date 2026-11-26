# The static cavity field: shifts, couplings and the equilibrium

levisim 0.1.1 introduced the cavity wavenumber and the static cavity shifts as opt-in keywords. Since
levisim 1.0 the static cavity shifts are **on by default** for physical parameters, together with
four corrections of the static field that make `MultiTrap`'s drift matrix equal to the Jacobian of
the nonlinear cavity model (`CavitySimulation(model='nonlinear')`) at the trap centres, to round-off:

| keyword (default) | what it adds | size off the node |
| --- | --- | --- |
| `static_cavity_shifts` (True) | the cotrapping and dispersive static stiffness | δΩ/2π ≈ +1.0, +1.2, +6.7 kHz |
| `static_cavity_phases` (True) | the tweezer phases $e^{i\phi_n}$ in the static field | 2-4 % of the drift matrix with unequal phases |
| `cavity_dispersive_coupling` (True) | the light coupling and the frequency pull of the cavity's own standing wave | $\lvert G_\text{disp}\rvert/\lvert G\rvert \approx 7 \times 10^{-3}$, pull 1.7 kHz |
| `cavity_envelope` (True) | the curvature of the cavity mode's transverse envelope | $\sim 2 \times 10^{-5}$ of the static stiffness |
| `cavity_effective_polarisability` (True) | Re $\alpha_\text{eff}$ in every cavity term | $-5.4 \times 10^{-5}$ of every cavity coupling (also at the node) |
| `cavity_wavenumber` (False) | the cavity's wavenumber in the standing wave | $6 \times 10^{-10}$ |

(sizes for one 60.1 nm sphere at $\Phi = \pi/4$ in the default cavity). Passing all of the first five
as False gives the numbers of levisim <= 0.1.1 bit for bit. All are ignored, with a warning, for toy
parameters, which have no standing wave and no static cavity field. Each is described below, followed
by the opt-in linearisation at the true equilibrium (`MultiTrap.from_nonlinear_equilibrium`) and the
warning for dropped light-coupled axes.

Off the node, the light a particle scatters into the cavity builds a static intracavity field
$\alpha = (A_0 + iP_0)/2$, of order $10^6$ photons for one 60 nm sphere in the default cavity of
{mod}`levisim.multiphysics.utils.shelf`. `MultiTrap` linearises the coherent-scattering interaction
about this field. The linear terms give the light-mechanics couplings $g_{j_n}$, and the
second-order terms in the displacement give the direct couplings $g_{xy} \propto A_0\cos\Phi$ and
$g_{xz}, g_{yz} \propto P_0\sin\Phi$. The same second-order expansion also has diagonal terms, and
the particle also sits in the cavity's own standing-wave intensity. Together these stiffen or soften
each motion. Since round 5, `MultiTrap` includes them by default.

## The cavity wavenumber (`cavity_wavenumber=True`)

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

`cavity_wavenumber` is off by default (as in 0.1.1); the static shifts below use the same
standing-wave wavenumber.

## The potential

The geometry is the one of {attr}`~levisim.multiphysics.dimension.Dimension.light_mech_coupling`.
The tweezer propagates along $z$ and is polarised along $y$. The cavity axis is
$\mathbf d = (\sin\theta, \cos\theta, 0)$, so $\theta$ is the angle between the tweezer polarisation
and the cavity axis ($\theta = 90°$ puts the cavity along $x$). The standing wave is
$\cos(k\,\mathbf d\cdot\mathbf r + \Phi)$, with $\Phi = \pi/2$ a node. The time-averaged dipole
potential of the total field, with $\mathcal E_\text{c} = 2E_0\,\alpha\cos(k\,\mathbf d\cdot\mathbf r + \Phi)$
and $E_0$ the field per photon, is

$$
U = -\frac{\alpha_\text{pol}}{4}\,\lvert \mathcal E_\text{tw} + \mathcal E_\text{c}\rvert^2
  = U_\text{tw} + U_\text{CS} + U_\text{cav},
$$

$$
\begin{aligned}
U_\text{CS} &= -E_d \sin\theta\, f(\mathbf r)\cos(k\,\mathbf d\cdot\mathbf r + \Phi)\,
  \big[A_0\cos\psi(z) + P_0\sin\psi(z)\big], \\
U_\text{cav} &= -\alpha_\text{pol} E_0^2\, \lvert\alpha\rvert^2 \cos^2(k\,\mathbf d\cdot\mathbf r + \Phi),
\end{aligned}
$$

where $E_d = \alpha_\text{pol} E_\text{tw} E_0/2$ (`Trap.Ed`, with $\alpha_\text{pol}$ = Re $\alpha_\text{eff}$, see below), $f$ is the tweezer's field envelope and
$\psi(z) = kz - \arctan(z/z_R)$ its phase along the beam. With the cavity quadratures as operators,
$U_\text{CS}$ is levisim's coherent-scattering interaction $V_\text{CS}$. This one function is
{func}`~levisim.physics.coherent_scattering.optical_potential`. The nonlinear cavity model uses it
too (its closed-form force and drive), so that its Jacobian equals the drift matrix (see Validation).

## The static stiffness

The second derivatives of $U_\text{CS} + U_\text{cav}$ at the trap centre that the linearised
couplings leave out are, with $s, c = \sin\theta, \cos\theta$ and $k_z = k - 1/z_R$,

$$
\begin{aligned}
K_{xx} &= E_d s A_0\cos\Phi\,\big(k^2 s^2 + 2/w_{x0}^2\big) + K^\text{cav}_{xx}, \\
K_{yy} &= E_d s A_0\cos\Phi\,\big(k^2 c^2 + 2/w_{y0}^2\big) + K^\text{cav}_{yy}, \\
K_{zz} &= E_d s A_0\cos\Phi\,\big(k_z^2 + 1/z_R^2\big), \\
K^\text{cav} &= 2\alpha_\text{pol}E_0^2\lvert\alpha\rvert^2 k^2\cos 2\Phi\;\mathbf d\,\mathbf d^\top ,
\end{aligned}
$$

plus the envelope term below. Here $A_0$ (and $P_0$ in $g_{xz}, g_{yz}$) is the static field in the
frame of the particle's own tweezer (see the tweezer phases below).
{func}`~levisim.physics.coherent_scattering.static_cavity_hessian` returns them. The off-diagonal
part of the $U_\text{CS}$ Hessian is already $g_{xy}$, $g_{xz}$ and $g_{yz}$. The dispersive term
adds an $x$-$y$ entry along the cavity axis, which goes into $g_{xy}$ as
$-K^\text{cav}_{xy}\,x_\text{zpf}\,y_\text{zpf}/\hbar$. Each frequency becomes

$$
\Omega_{j_n}^2 = \Omega_{j_n,\text{tw}}^2 + K_{jj}/m_n .
$$

For red detuning $A_0\cos\Phi > 0$, so the coherent-scattering part stiffens every axis, most of
all $z$, whose tweezer stiffness is weakest. The dispersive part vanishes at $\Phi = \pi/4$ and has
the sign of $\cos 2\Phi$. Both parts vanish when the static field is zero, which is the case when
every particle sits at a node. The frequencies there are bit-identical to those without the shifts.

**Self-consistency.** The shifted $\Omega_{j_n}$ is used everywhere a frequency enters: in the
mechanical susceptibilities, in the zero-point fluctuation (so each coupling $g$ is rescaled by
$\sqrt{\Omega_\text{tw}/\Omega}$ per dimension), in the thermal and photon-recoil phonon numbers and in
the cooling estimates. The shift itself does not depend on $\Omega$, so no iteration is needed.

**Sizes.** For one 60.1 nm sphere at $\Phi = \pi/4$ in the default cavity ($\Delta/2\pi = -450$ kHz,
$\lvert\alpha\rvert^2 = 1.86 \times 10^6$ photons), $\delta\Omega/2\pi$ = +1.05, +1.17 and +6.72 kHz on
156.3, 135.8 and 35.1 kHz for $x$, $y$ and $z$ (+1.04, +1.17 and +6.71 kHz, with
$\lvert\alpha\rvert^2 = 1.85 \times 10^6$, without the pull and with the static polarisability). At
$\Phi = 0.05\pi$ the $z$ shift is about 12 kHz. With two or three particles the field adds up, and the
shifts reach 3-5 kHz in $x$, $y$ and 17 kHz in $z$.

## The tweezer phases (`static_cavity_phases=True`)

Each particle's linear coupling carries its tweezer's phase, $G_{j_n} = g_{j_n} e^{i\phi_n}$: the
cavity mode seen from tweezer $n$ is $a_n = a\,e^{-i\phi_n}$. The static drive of particle $n$ carries
the same phase, so the static field in the cavity frame is

$$
\alpha = \sum_n \omega_{d,n}\sin\theta_n\cos\Phi_n\,e^{i\phi_n}\;
  \frac{-\Delta_\text{eff} + i\kappa/2}{\Delta_\text{eff}^2 + (\kappa/2)^2},
$$

and particle $n$ sees $\alpha_n = \alpha\,e^{-i\phi_n}$ (`MultiTrap.static_bead_frame_amplitudes`),
which sets its $g_{xy} \propto \mathrm{Re}\,\alpha_n$, $g_{xz}, g_{yz} \propto \mathrm{Im}\,\alpha_n$ and
its static stiffness. This is what the nonlinear model derives from one energy. $\alpha_n$ is summed
from the phase differences $\phi_m - \phi_n$, so with equal phases (or one particle) every number of
the particles is bit-identical to the sum without phases, and only the cavity-frame field
(`cavity.A0`, `cavity.P0`) is rotated by the common phase. At the node every spectrum and frequency is
bit-identical too; drift-matrix entries at the round-off of the node's static field
($\cos(\pi/2) \approx 6 \times 10^{-17}$) change. Off the node with unequal phases the static field
changes at first order: for two particles at $\Phi = \pi/4, 1.1$ rad with $\phi = 0°, 70°$ the drift
matrix changes by 1.8 % of its largest entry, the frequencies by up to 4 kHz, and the growth rate of
the unstable mode from 5180 to 2750 s⁻¹. `static_cavity_phases=False` sums the drives without phases
(levisim <= 0.1.1), which agrees with the nonlinear model only at the node or with equal phases.

## The dispersive terms (`cavity_dispersive_coupling=True`)

The cavity's own standing-wave potential $U_\text{cav} = -\alpha_\text{pol}E_0^2\lvert\alpha\rvert^2\cos^2(k\,\mathbf d\cdot\mathbf r + \Phi)$
is quadratic in the field, so besides its static stiffness it has two linear terms:

- **a light-mechanical coupling** along the cavity axis, from the term linear in both $\mathbf r$ and
  the fluctuation $a$, $\alpha_\text{pol}E_0^2 k\sin 2\Phi\,(\mathbf d\cdot\mathbf r)(\bar\alpha a^\dagger + \bar\alpha^* a)$:

  $$
  G_{\text{disp},j} = -\delta\,k\sin 2\Phi\, d_j\, x_{\text{zpf},j}\,\bar\alpha, \qquad \delta = \alpha_\text{pol}E_0^2/\hbar,
  $$

  added to `Dimension.G` (kept separately as `Dimension.dispersive_G`). It has the phase of the
  static field, is zero along $z$, at the node and at the antinode, and is $7 \times 10^{-3}$ of
  $\lvert G\rvert$ for the example above. For a libration it is
  $(E_0^2/\hbar)\cos^2\Phi\,\partial_j c\,\theta_{\text{zpf},j}\bar\alpha$ with $c = \mathbf e_c\cdot\alpha(\theta)\cdot\mathbf e_c$.
- **the pull of the cavity frequency**, $\delta_n\cos^2\Phi_n$ per particle: the light sees the detuning
  $\Delta_\text{eff} = \Delta + \sum_n\delta_n\cos^2\Phi_n$ (`Cavity.dispersive_shift`, `Cavity.Δ_eff`),
  in the static field, the drift matrix and the cavity susceptibility `Cavity.χC`. $\delta/2\pi \approx 3.4$
  kHz for a 60 nm sphere in the default cavity, so one particle at $\Phi = \pi/4$ pulls the cavity by
  1.7 kHz, which raises $\lvert\alpha\rvert^2$ by 0.6 % at $\Delta/2\pi = -450$ kHz. `Cavity.ω`, `k` and
  `E0` keep $\Delta$ (a relative change of $\sim 10^{-14}$). The multi-tone model
  ({mod}`levisim.multiphysics.multitone`) uses its tones' own detunings, without the pull.

Off the node the two together change the drift matrix by ~0.5 % (`tests/test_static_cavity_field.py`).

## The cavity mode envelope (`cavity_envelope=True`)

The standing wave carries the cavity mode's transverse envelope $e^{-\rho^2/w^2}$ about its axis
through the trap centre, $\rho^2 = r^2 - (\mathbf d\cdot\mathbf r)^2$, with $w$ = `Cavity.waist`. Its
gradient vanishes at the centre, so it adds no force and no linear coupling, but its curvature adds to
the static stiffness, including an $x$-$y$ entry (added to $g_{xy}$):

$$
K_\text{env} = \frac{2}{w^2}\big[E_d s A_0\cos\Phi + 2\alpha_\text{pol}E_0^2\lvert\alpha\rvert^2\cos^2\Phi\big]\,
  (\mathbb 1 - \mathbf d\,\mathbf d^\top),
$$

$\sim 2/(kw)^2 \approx 2 \times 10^{-5}$ of the stiffness above. The nonlinear model includes the envelope
in its force and drive when the MultiTrap does.

## The polarisability (`cavity_effective_polarisability=True`)

The tweezer stiffness `Trap.kj` uses the radiation-reaction corrected polarisability, Re
$\alpha_\text{eff}$ = Re $[\alpha/(1 - ik^3\alpha/6\pi\varepsilon_0)]$. The coherent-scattering drive
$E_d$ (`Trap.Ed`) and the cavity terms now use the same polarisability, at the wavenumber of the light
the particle scatters into the cavity, instead of the static `Bead.α`; for an anisotropic particle the
real part of the lab-frame tensor $\alpha(\mathbb 1 - ik^3\alpha/6\pi\varepsilon_0)^{-1}$, also in the
librational terms ({func}`~levisim.physics.coherent_scattering.drive_polarisability`,
{func}`~levisim.physics.coherent_scattering.cavity_mode_polarisability`). For a 60 nm silica sphere at
1064 nm the two differ by $(k^3\alpha/6\pi\varepsilon_0)^2 \approx 5.4 \times 10^{-5}$, so every cavity
coupling changes at that level, also at the node (the growth rate of the unstable built-in 2 × 2
system moves from 2677.5 to 2676.7 s⁻¹).

## Measured frequencies

A trap given measured frequencies (`override_Ω_dict`) keeps them: a measured frequency already
contains every static stiffness, so neither the static cavity shift nor the self-stiffness of the
`interactions` is added to it (the latter is added to the frequencies levisim computes). The couplings
are still added: the dispersive $x$-$y$ coupling, $g_{xy}$, $g_{xz}$, $g_{yz}$ and the pair couplings.
`interaction_frequency_shifts='all'` adds the interactions' self-stiffness to measured frequencies
too, which describes frequencies that are the traps' own, without the interaction (levisim <= 0.1.1
did this by default; the Krisnanda et al. 2020 notebook passes it, since the paper's $\omega$ is the
bare trap frequency), and `interaction_frequency_shifts=False` adds it to no frequency.

## Validation

- **The nonlinear model.** With every correction on, `MultiTrap.drift_matrix` (in quadratures) equals
  the Jacobian of the drift of `NonlinearCavityModel(damping='symmetric')` evaluated at the
  MultiTrap's static field and the trap centres to $1.5$-$2.5 \times 10^{-13}$ of its largest entry,
  off the node and with unequal tweezer phases, for 1 × 3, 2 × 2, 2 × 3 and 3 × 3 systems; the static
  field is the fixed point of the nonlinear drive there. Without the phases the two differ by 3-5 %,
  without the dispersive terms by 0.4-0.6 % (`tests/test_static_cavity_field.py`).
- **Finite differences.** Fourth-order central differences of `optical_potential` at the trap
  centre reproduce the tweezer stiffness `Trap.kj`, the couplings $g_x, g_y, g_z$ and the dispersive
  coupling (through $\partial^2 U/\partial r_j\partial(A, P)$), $g_{xy}, g_{xz}, g_{yz}$ and every entry
  of `static_cavity_hessian`, with the envelope. The agreement is better than $10^{-6}$ relative
  (typically $10^{-10}$) for seven $(\Phi, \theta)$ cases (`tests/test_static_cavity_shifts.py`). The
  envelope's force and drive match central differences of the potential away from the centre.
- **PRR 5.** In the showcase [Mode orientations along the standing wave](../showcase/prr5_mode_orientations.md),
  the static field rotates the $x$-$y$ modes against the cavity dynamics, and the rotation cancels at
  $\Phi_\text{c} = 0.247\pi$. The paper found that the frequency shifts cancel at the same point. With
  the static shifts, the net shift of each mode (static stiffening minus optical spring) crosses zero
  at $0.255$-$0.256\pi$. Without them it does not cross zero near there.

## Turning them off

`MultiTrap(static_cavity_shifts=False)` leaves out the static stiffness; with
`static_cavity_phases=False, cavity_dispersive_coupling=False, cavity_envelope=False,
cavity_effective_polarisability=False` as well it gives the couplings and frequencies of levisim
0.1.0 exactly. The paper-reproduction notebooks pass `static_cavity_shifts=False`; the PRR 5 showcase
figure includes the shifts since levisim 1.0. Toy parameters have no static field, so the keywords have
no effect on them.

## The equilibrium (`MultiTrap.from_nonlinear_equilibrium`)

The static force {func}`~levisim.physics.coherent_scattering.static_cavity_force` moves the particle off
the trap centre: for the example above by $(-2.8, -3.8, 34.5)$ nm ($k_z\,\delta z \approx 0.2$ rad).
Recoil (a `PhotonRecoil` element), `Gravity` and the pair forces of the `interactions` displace it as
well. `MultiTrap(...)` linearises at the trap centres (the closed forms above), so off the node its
drift matrix differs from the linearisation at the equilibrium by 3.5 % (one particle) to 9 % (two
particles at $\Phi = \pi/4, 1.1$ rad). The opt-in constructor

```python
mtrap = MultiTrap.from_nonlinear_equilibrium(**the_same_keyword_arguments)
```

solves for the equilibrium of the nonlinear model and builds the MultiTrap from the Jacobian there:
the frequencies $\Omega_j = \sqrt{K_{jj}/m}$ of the stiffness $K = -\partial\mathbf F/\partial\mathbf r$ at
the equilibrium, $G_j$ from $\partial F_j/\partial(A, P)$, every direct coupling
$g_{jk} = -K_{jk}x_{\text{zpf},j}x_{\text{zpf},k}/\hbar$ (within and between particles), and the static field
and pull of the cavity at the equilibrium (`levisim.multiphysics.multitrap.base.equilibrium_linearisation`).
Its drift matrix equals the Jacobian of the nonlinear drift at the equilibrium (in the quadratures of
the new frequencies) to $3 \times 10^{-13}$, and every method uses it. For the example above the
frequencies move from 157.39, 136.97, 41.78 kHz to 157.42, 137.01, 42.11 kHz. The model and the
displacements are kept as `mtrap.nonlinear_model` and `mtrap.equilibrium_displacements`. Measured
frequencies (`override_Ω_dict`) are replaced by the curvature of the force elements, and librations are
not supported (the nonlinear model has none).

## Dropped light-coupled axes

A MultiTrap with fewer than three axes per particle leaves out the others. Off the node $z$ couples
to the phase quadrature ($g_z \propto \cos\Phi$) and to $x$ and $y$ ($g_{xz}, g_{yz} \propto P_0\sin\Phi$),
so a 2-axis model drops a light-coupled axis, and the dropped dynamics can change the stability: two
60 nm spheres at $\Phi = \pi/4$ with $\phi_2 = 90°$ in the default cavity are stable with two axes each
(largest Re $\lambda = -1088$ s⁻¹) but unstable with three (+6080 s⁻¹). `MultiTrap` logs a WARNING when a dropped axis couples to the light or to a modelled axis by more
than $10^{-3}$ of the largest modelled $\lvert G\rvert$ (`MultiTrap.dropped_axis_couplings`,
`warn_about_dropped_light_coupled_axes`); at the node, where $z$ decouples, it does not.
`dropped_axis_warning=False` silences it.

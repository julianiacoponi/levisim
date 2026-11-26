# Cavity wavenumber and static cavity shifts

levisim 0.1.1 introduced both corrections as opt-in keywords. Since levisim 1.0 the static cavity
shifts are **on by default** for physical parameters (`static_cavity_shifts=False` gives the numbers
of levisim <= 0.1.1); the cavity wavenumber stays opt-in. Both are ignored, with a warning, for toy
parameters, which have no standing wave and no static cavity field.

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

where $E_d = \alpha_\text{pol} E_\text{tw} E_0/2$ (`Trap.Ed`), $f$ is the tweezer's field envelope and
$\psi(z) = kz - \arctan(z/z_R)$ its phase along the beam. With the cavity quadratures as operators,
$U_\text{CS}$ is levisim's coherent-scattering interaction $V_\text{CS}$. This one function is
{func}`~levisim.physics.coherent_scattering.optical_potential`. The nonlinear cavity simulation is
meant to use it too, so that its Jacobian at equilibrium equals the drift matrix.

## The static stiffness

The second derivatives of $U_\text{CS} + U_\text{cav}$ at the trap centre that the linearised
couplings leave out are, with $s, c = \sin\theta, \cos\theta$ and $k_z = k - 1/z_R$,

$$
\begin{aligned}
K_{xx} &= E_d s A_0\cos\Phi\,\big(k^2 s^2 + 2/w_{x0}^2\big) + K^\text{cav}_{xx}, \\
K_{yy} &= E_d s A_0\cos\Phi\,\big(k^2 c^2 + 2/w_{y0}^2\big) + K^\text{cav}_{yy}, \\
K_{zz} &= E_d s A_0\cos\Phi\,\big(k_z^2 + 1/z_R^2\big), \\
K^\text{cav} &= 2\alpha_\text{pol}E_0^2\lvert\alpha\rvert^2 k^2\cos 2\Phi\;\mathbf d\,\mathbf d^\top .
\end{aligned}
$$

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
$\lvert\alpha\rvert^2 = 1.85 \times 10^6$ photons), $\delta\Omega/2\pi$ = +1.04, +1.17 and +6.71 kHz on
156.3, 135.8 and 35.1 kHz for $x$, $y$ and $z$. At $\Phi = 0.05\pi$ the $z$ shift is about 12 kHz. With
two or three particles the field adds up, and the shifts reach 3-5 kHz in $x$, $y$ and 17 kHz in $z$.

## Validation

- **Finite differences.** Fourth-order central differences of `optical_potential` at the trap
  centre reproduce the tweezer stiffness `Trap.kj`, the couplings $g_x, g_y, g_z$ (through
  $\partial^2 U/\partial r_j\partial(A, P)$), $g_{xy}, g_{xz}, g_{yz}$ and every entry of
  `static_cavity_hessian`. The agreement is better than $10^{-6}$ relative (typically $10^{-10}$) for
  seven $(\Phi, \theta)$ cases (`tests/test_static_cavity_shifts.py`).
- **PRR 5.** In the showcase [Mode orientations along the standing wave](../showcase/prr5_mode_orientations.md),
  the static field rotates the $x$-$y$ modes against the cavity dynamics, and the rotation cancels at
  $\Phi_\text{c} = 0.248\pi$. The paper found that the frequency shifts cancel at the same point. With
  the static shifts, the net shift of each mode (static stiffening minus optical spring) crosses zero
  at $0.255$-$0.257\pi$. Without them it does not cross zero near there.

## Turning them off

`MultiTrap(static_cavity_shifts=False)` gives the couplings and frequencies of levisim 0.1.0 exactly.
The paper-reproduction notebooks pass it, and so does the PRR 5 showcase figure. Toy parameters
have no static field, so the keyword has no effect on them. A trap given measured frequencies
(`override_Ω_dict`) keeps them, since a measured frequency already contains the static shift. The
dispersive $x$-$y$ coupling is still added there.

## Not included

- **The equilibrium displacement.** The static force
  {func}`~levisim.physics.coherent_scattering.static_cavity_force` moves the particle off the trap
  centre. For the example above this is about 33 nm along $z$ ($k_z\,\delta z \approx 0.19$ rad), and
  levisim still linearises about the centre.
- **The cavity mode envelope.** Its curvature changes the second derivatives by about
  $2/(k w)^2 \sim 10^{-5}$. `optical_potential(..., cavity_waist=w)` includes it, but `MultiTrap`
  does not.
- **The tweezer phase.** The static field does not rotate with each tweezer's phase $\phi_n$. This
  is the convention of the existing couplings, which the shifts follow.
- **The dispersive dynamical coupling.** The $\lvert\alpha\rvert$-linear coupling of $U_\text{cav}$ to
  the cavity fluctuations is about $\lvert\mathcal E_\text{c}\rvert/E_\text{tw} \sim 10^{-3}$ of $g$.

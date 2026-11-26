# Rotation and libration

A non-spherical particle, a birefringent sphere or a sphere in circularly polarised light feels a
torque. levisim treats the particle as a rigid body: its orientation is a unit quaternion
(`Particle.orientation`), and `levisim.physics.rotation` gives the torques, their stiffness and their
noise. Two solvers use them:

- `RigidBodySimulation`, the stochastic time domain of one particle (translation and rotation);
- `MultiTrap`, whose dimensions may include the librations `'α'`, `'β'`, `'γ'` of each particle.

## Conventions

| Quantity | Convention |
|---|---|
| orientation | unit quaternion $q = (w, x, y, z)$, body frame → lab frame, rotation matrix $R(q)$ |
| angular momentum | body frame $L_b$; $\omega = I^{-1}L_b$; lab $\omega_\text{lab} = R\,\omega$ |
| librations | small rotation vector $\theta = (\theta_\alpha, \theta_\beta, \theta_\gamma)$ about the **lab** axes $x, y, z$ at the reference orientation $R_0$: $R = e^{[\theta]_\times}R_0$ |
| fields | complex amplitudes, $E(t) = \mathrm{Re}(E\,e^{-i\omega t})$; levisim's tweezer propagates along $z$ and is polarised along $y$ by default |

So for a linearly polarised tweezer, `'α'` and `'γ'` are the tilts of a particle's long axis out of
the polarisation towards $z$ and $x$, and `'β'` is the rotation about the polarisation itself, which
the tweezer does not confine.

## Torques

```python
from levisim.physics import Spheroid, Trap, Tweezer
from levisim.physics.particle import body_axis_to_quaternion

rod = Spheroid(a=70e-9, b=50e-9, orientation=body_axis_to_quaternion((0, 1, 0)))   # long axis along y
trap = Trap(tweezer=Tweezer(), particle=rod)
trap.Ωθ            # {'α': ..., 'β': 0.0, 'γ': ...}, librational frequencies [2π·Hz]
trap.torque_stiffness_tensor, trap.Γrot_j, trap.libration_recoil_heating_rates
```

| Element | Torque | Stiffness $K_{ij} = -\partial\tau_i/\partial\theta_j$ |
|---|---|---|
| `Tweezer` | $\tfrac12\mathrm{Re}(p^*\times E) - \frac{k^3}{12\pi\varepsilon_0}\mathrm{Im}(p^*\times p)$, $p = \alpha_\text{eff,lab}E$ | the derivative of the torque for $\alpha \to \alpha + \theta_j\partial_j\alpha$, $\partial_j\alpha = G_j\alpha - \alpha G_j$ |
| `UniformElectricField` | $p_0\times E + (\alpha_\text{static}E)\times E$ | $-(G_jp_0)\times E - (\partial_j\alpha\,E)\times E$ |
| magnetic (`rotation.magnetic_torque`) | $\mu\times B + \tfrac{V}{\mu_0}(\chi B)\times B$ | (in the field elements) |

$G_j$ generates rotations about lab axis $j$ ($G_jv = \hat e_j\times v$). The first optical term is the
torque of the incident field on the induced dipole, the second the recoil of the angular momentum the
dipole radiates (Chaumet and Rahmani, Opt. Express **17**, 2224 (2009)). Together they give the alignment
torque of the anisotropic real polarisability and the **spin torque** of circular and elliptical light.
On a sphere the spin torque is the absorbed power over $\omega$: a lossless sphere is not spun up.
A lossless anisotropic particle is, through the anisotropy of its scattering: rotating in the plane of
circularly polarised light, it feels $k^3E_0^2(\alpha_1 - \alpha_2)^2/(24\pi\varepsilon_0)$ at every angle
(tested). A tweezer gets any polarisation through a Jones vector,
`Tweezer(polarisation=Tweezer.elliptical_polarisation(ψ, ζ))`, $\hat e = \cos\psi\,\hat e_1 + i\sin\psi\,\hat e_2$
(ψ = π/4 circular); the default (linear along $y$) keeps every existing number.

For linear polarisation along $y$ and a particle whose principal axes lie along the lab axes, the
librational stiffness is $\tfrac12E_0^2(\alpha_y - \alpha_z)$ about $x$ and $\tfrac12E_0^2(\alpha_y - \alpha_x)$
about $z$ (Re $\alpha_\text{eff}$, up to the recoil term, a relative 10⁻⁴), and the frequency is $\Omega_j = \sqrt{K_{jj}/I_{jj}}$.

### Recoil torque noise

The induced dipole interferes with the vacuum field. The torque linear in the vacuum field $\delta E$
is $\tau_i = M_{il}\delta E_l$ with $M_{il} = \varepsilon_{ijk}\alpha_{jl}E_k + \varepsilon_{ijl}p_j$, and
the two-sided noise is

$$
S_\tau = \frac{\hbar k^3}{12\pi\varepsilon_0}\,\mathrm{Re}(MM^\dagger),
$$

normalised so that the same construction for the force gives levisim's translational recoil noise
(`Tweezer.noise`). In the principal frame, for rotation about axis $i$, it is
$\hbar k^3(\alpha_j - \alpha_k)^2(|E_j|^2 + |E_k|^2)/(12\pi\varepsilon_0)$, which is the rotational recoil
heating of Rudolph et al., Phys. Rev. A **103**, 043514 (2021), Eq. (31b), including its weights
$\cos^2\psi$ and $\sin^2\psi$ for elliptical light (tested). A sphere of isotropic material feels none.

### Gas damping

The rotational friction and diffusion tensors are those of Martinetz et al., Phys. Rev. E **97**, 052112
(2018), from the shapes (`Particle.rotational_damping_tensor`, `rotational_diffusion_tensor`):
$\dot J = -\Gamma_\text{rot}J$ with angular-momentum diffusion $2k_BT\,\Gamma_\text{rot}I$.

### Finite-size alignment in an anisotropic trap

A particle that is not small compared with the trap's length scales also samples the curvature of
the intensity over its own extent. In the generalised Rayleigh–Gans picture this adds
$U = \tfrac12\mathrm{Tr}(k\,\langle r r^\top\rangle)$, with $k$ the translational stiffness tensor and
$\langle r r^\top\rangle = S/m$ the particle's mass-weighted second moment in the lab frame
(`rotation.second_moment_tensor`, `finite_size_potential`, `finite_size_torque`,
`finite_size_torque_stiffness`). It turns the longer axes towards the weaker confinement; for an
ellipsoid with radii $p, q, r$ along the lab axes it gives
$\Omega_{lx}^2 = (\Omega_z^2 - \Omega_y^2)(q^2 - r^2)/(q^2 + r^2)$, the trap-anisotropy term of
Kamba et al., Nat. Commun. **14**, 7943 (2023), Eq. (3). It is opt-in, `Trap(finite_size_torque=True)`
(then also the default of `RigidBodySimulation`), so the point-dipole torques are unchanged by default;
`Trap` uses $m\Omega_j^2$ from `override_Ω_dict` when that gives x, y and z.

## Time domain: `RigidBodySimulation`

```python
from levisim.simulation import RigidBodySimulation
sim = RigidBodySimulation(trap=trap, num_samples=200_000, Δt=5e-9, seed=1)
sim.traces['γ'], sim.lab_directions((0, 0, 1)), sim.angular_velocities_lab()
```

The state is $(r, p, q, L_b)$, with the angular momentum in the body frame (so that the rotational
friction and its noise are constant and additive) and the translational momentum in the lab frame:

$$
\dot r = p/m,\quad \dot p = F - R\Gamma_\text{cm}R^\top p + \xi_p,\quad
\dot q = \tfrac12 q\otimes(0,\omega),\quad \dot L_b = \tau_b - \omega\times L_b - \Gamma_\text{rot}L_b + \xi_L .
$$

The rotational noise satisfies the fluctuation-dissipation relation exactly; the translational noise
is that of the initial orientation, exact for a sphere and for librations (see the module docstring).

The tweezer force uses the orientation-dependent $\hat e^*\cdot\alpha_\text{eff}\cdot\hat e$, so the
translational and rotational motions are coupled. After every step the quaternion is renormalised
(a projection method; `StochasticSimulation.project_state`). Options: `recoil=True` adds the
recoil force and torque noise (at the initial orientation, fixed in the body frame), `power_modulation`
modulates the tweezer intensity (parametric feedback), `external_torque` and `external_force` take
callables of $(t, X)$ (drives and feedback); the built-in `parametric_feedback`, `torque_drive` and
`angular_damping` also run in the compiled loop.

Kasdin's stochastic Runge–Kutta scheme is slightly dissipative on an undamped oscillator, at a rate
growing as $(\Omega\Delta t)^3$: an 800 kHz libration loses energy at about 6 × 10³ s⁻¹ with 25 steps
per period and 6 s⁻¹ with 250. Where heating or cooling rates matter, resolve the fastest libration
with a few hundred steps per period.

For a particle aligned with a linear polarisation, the point-dipole torque depends only on the angle
between the axis and the polarisation, so the two tilts (α and γ) are degenerate and the angular
momentum of the conical motion about the polarisation is conserved by any modulation of the tweezer
power: single-channel parametric feedback cannot take the pair below $\Omega|\ell|$ (see the
van der Laan et al. (2021) reproduction).

```{figure} figures/rotation_rigid_body.png
:width: 100%

Left: the body-frame angular velocity of a torque-free prolate spheroid (100:50 nm) precesses at
$\Omega_p = (I_3 - I_1)\omega_3/I_1$; the largest deviation from the analytic solution is 3.6 × 10⁻⁷ of
$\omega_\perp$, the energy drift 4 × 10⁻⁹ and $||q| - 1| \le 2.2\times10^{-16}$ over 2 × 10⁴ steps.
Right: spin about the minor and major axes of inertia of a 90:60:40 nm ellipsoid is stable, spin about
the intermediate axis turns over (the tennis-racket effect).
```

## Librational dimensions in `MultiTrap`

`Cartesian_axes_per_trap` takes a tuple of axis names; the integer form is unchanged:

```python
from levisim.multiphysics.multitrap import MultiTrap
mt = MultiTrap(num_traps=1, Cartesian_axes_per_trap=('x', 'y', 'z', 'α', 'γ'), beads=[rod], tweezers=[Tweezer()])
mt.labels    # ['x1', 'y1', 'z1', 'α1', 'γ1']
```

A librational dimension has the moment of inertia about its lab axis in place of the mass (in the
zero-point amplitude $\theta_\text{zpf} = \sqrt{\hbar/2I\Omega}$), the frequency from the torque
stiffness of the trap's elements (plus the static cavity field with `static_cavity_shifts`, like the
translations), the rotational gas damping, and the recoil torque heating as photon recoil. Its light
coupling and its couplings to the other dimensions come from the same coherent-scattering potential
as the translations, with the polarisability tensor rotated
(`coherent_scattering.librational_expansion`):

$$
U_\text{CS} = -E_dE_0\,\mathrm{Re}\big(\alpha_c^*\,F(r)\,s(\theta)\big),\quad s(\theta) = e_c\cdot\alpha(\theta)\cdot\hat e,
\qquad G_j = \frac{E_dE_0}{2\hbar}\cos\Phi\,\partial_js\;\theta_{\text{zpf},j}.
$$

Librations couple at the antinode ($\cos\Phi$), translations $x, y$ at the node ($\sin\Phi$), as in
Stickler et al., Phys. Rev. A **94**, 033818 (2016). The ratio of the libration and $x$ couplings equals
that of Rudolph et al. (2021), Eqs. (D5a) and (D5d), to 10⁻¹⁰ (test). levisim's cavity has one mode,
polarised in the $x$-$y$ plane ($e_c$); the rotation about $x$ scatters into the orthogonal mode (along $z$),
which is not modelled, so it has no light coupling here.

The same expression fixes the coherent-scattering drive of an anisotropic particle: it is
$e_c\cdot\alpha\cdot\hat e$, which includes the cross-polarised scattering of a tilted particle
(`coherent_scattering.drive_polarisability`; `Bead.α` itself for a sphere).

Limits of the linear model: the lab-frame inertia and rotational damping are taken diagonal (principal
axes along the lab axes; a WARNING otherwise), a free rotation (no restoring torque, e.g. `'β'` in
linear polarisation) cannot be a dimension (ValueError), and a static torque at the reference
orientation is logged but the orientation is not moved.

```{figure} figures/rotation_libration_psd.png
:width: 75%

The libration $\gamma$ of a 70:50 nm spheroid aligned with the polarisation, at 10 mbar of 3 K gas
(small, linear librations): the Welch spectrum of a `RigidBodySimulation` (4 × 10⁵ steps of 5 ns),
the thermal Lorentzian from $\Omega_\gamma = 2\pi\times829.6$ kHz, $\Gamma_\gamma$ and the
fluctuation-dissipation torque noise, and the `MultiTrap` spectrum of $\gamma_1$. The time-domain
variance is 1.06 × $k_BT/K$. MultiTrap's rotating-wave damping gives a peak area 14 % lower at this
large damping ($\Gamma/\Omega = 0.14$).
```

`docs/concepts/scripts/rotation_figures.py` makes both figures and prints the numbers quoted here.

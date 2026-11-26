# Traps as sums of fields

A {class}`~levisim.physics.trap.Trap` is a particle, a gas and a list of *force elements*, the
`fields`. Each element is one source of force on the particle: the optical tweezer, a harmonic
potential, gravity, feedback damping, a kick, the scattering force of the tweezer light. The trap's
stiffness, frequencies and couplings are sums over the elements, and the time-domain simulation adds
up their forces. A trap that is not optical, or that combines several fields, is therefore written
in the same way as the single-tweezer trap levisim has always had, and that trap still gives the
same numbers bit for bit.

## Building a trap

```python
from levisim import Trap, Tweezer, Bead, Feedback, HarmonicPotential, Gravity
from levisim.core.units import kHz

# the shorthand: one tweezer plus feedback cold damping (Γfb,x, Γfb,y = 2.1, 2.5 kHz by default)
trap = Trap(tweezer=Tweezer(), bead=Bead())
trap.fields          # [Tweezer(...), Feedback(Γ_fbx=2.1 kHz, Γ_fby=2.5 kHz, Γ_fbz=0 kHz)]

# the same trap, spelled out
trap = Trap(fields=[Tweezer(), Feedback(2.1*kHz, 2.5*kHz)], particle=Bead())

# a tweezer with an additional harmonic potential, and gravity along -x
hybrid = Trap(fields=[Tweezer(), HarmonicPotential([1e-6, 0, 0]), Gravity()], particle=Bead())
```

`Trap(tweezer=..., bead=...)` builds `fields=[tweezer, Feedback(Γ_fbx, Γ_fby, Γ_fbz)]`, adds a
{class}`~levisim.physics.fields.elements.PhotonRecoil` element if `photon_recoil=True`, and appends
the `kicks`. With `fields=` the trap holds exactly the elements given: there is no feedback damping
unless a {class}`~levisim.physics.fields.elements.Feedback` element is among them, or `Γ_fbx`, `Γ_fby`,
`Γ_fbz` are passed (which then add one), and no directed force noise unless `β` is passed (the
shorthand keeps the historical, experiment-specific $\beta = \sqrt2$, which doubles the $x$-$y$ force
noise; see {class}`~levisim.analysis.spectral_density_theory.SpectralDensityTheory`). `trap.Γ_fbx`, `trap.kicks`, `trap.photon_recoil` and
`trap.tweezer` read and write the corresponding elements. `trap.fields` is an ordinary list and may be
edited in place.

## The force-element interface

Every element derives from {class}`~levisim.physics.fields.base.ForceElement` and provides

| Method | Meaning | Units |
|---|---|---|
| `force(r, t, particle)` | force at position $\mathbf{r}$ and time $t$ | N |
| `torque(r, t, particle, orientation)` | torque about the centre of mass (zero for a sphere) | N·m |
| `potential(r, particle)` | potential energy, for a conservative element | J |
| `stiffness(r0, particle)` | $K_{ij} = -\partial F_i/\partial r_j$ at $\mathbf{r}_0$ | N/m |
| `noise(particle)` | two-sided force spectral density matrix it adds | N²/Hz |
| `damping_rates(particle)` | linear damping rates $(\Gamma_x, \Gamma_y, \Gamma_z)$, force $-\Gamma_j p_j$ | 2π·Hz |

and the flag `is_optical`. `stiffness` is written in closed form where one exists; otherwise the base
class takes central differences of `force` with step `finite_difference_step`, the same scheme as
`Trap.Ωj_at_z_eq`. One definition of the force thus serves both the nonlinear time domain and the
linearised frequency domain, and a test can compare the two.

| Element | Force | Stiffness | Noise |
|---|---|---|---|
| `Tweezer` | gradient force $\tfrac{1}{4}\mathrm{Re}(\alpha_\mathrm{eff})\nabla\lvert E\rvert^2$, exactly the arithmetic of `Trap.Ftw` | closed form at the focus, $\mathrm{Re}(\alpha_\mathrm{eff})E_0^2/w_{j0}^2$; central differences elsewhere | photon-recoil force noise |
| `HarmonicPotential(K, centre)` | $-K(\mathbf{r}-\mathbf{c})$ | $K$ | - |
| `Gravity(g, direction)` | $m_n g\,\hat{\mathbf{d}}$ | 0 | - |
| `Feedback(Γ_fbx, Γ_fby, Γ_fbz)` | $-\Gamma_{\mathrm{fb},j}\,p_j$ | 0 | - (loop noise not modelled) |
| `PhotonRecoil(tweezer)` | scattering force along $+z$, `Trap.F_scattering` | central differences | - |
| `Kick` | $F_0(\cos\theta, \sin\theta, 0)$ while active | 0 | - |
| `PaulTrap(V_rf, Ω_rf, r0, U_dc, geometry, ...)` | full rf + dc quadrupole force $-Q[c^\mathrm{dc}_j + c^\mathrm{rf}_j\cos\Omega_\mathrm{rf}t](r_j - c_j)$ | secular (pseudopotential) stiffness | electrode voltage noise $Q\,E_\mathrm{noise}$ |
| `UniformElectricField(E)` | $Q\mathbf{E}$ | 0 | - |
| `MagneticTrap(source)` | $(V/\mu_0) J^\mathsf{T}\chi_\mathrm{eff}\mathbf{B}$ from $U = -(V/2\mu_0)\mathbf{B}\cdot\chi_\mathrm{eff}\mathbf{B}$ | analytic, from $\mathbf{B}$, $J$ and $\partial^2\mathbf{B}$ | - |
| `EddyCurrentDamping(source, position)` | $-\Gamma_j p_j$, $\Gamma = \sigma R^2 J^\mathsf{T}J/10\rho$ | 0 | thermal, $2k_BT\gamma$ |

The tweezer's `noise` is the force spectral density
$S_j = \frac{\pi\hbar}{15\varepsilon_0}\left(\frac{\mathrm{Re}(\alpha_\mathrm{eff}) E_0}{2\pi}\right)^2 k^5 c_j$,
$(c_x, c_y, c_z) = (2, 1, 7)$, which heats a mode of frequency $\Omega_j$ at
$S_j/(2 m_n \hbar \Omega_j)$ phonons per second, i.e. at `Trap.photon_recoil_heating_rates`.
`Trap.field_noise_psd` sums the elements' noise; the gas noise $S_\mathrm{th}$ and the directed noise
$S_\mathrm{dir}$ stay properties of the trap.

## Stiffness, frequencies and couplings

The total stiffness tensor is the sum of the elements' tensors at the trap's `linearisation_point`,

$$
K = \sum_{e\,\in\,\mathrm{fields}} K^{(e)}(\mathbf{r}_0),
$$

with $\mathbf{r}_0$ the origin (the tweezer focus) by default, as levisim has always used.
`Trap(linearisation_point='equilibrium')` linearises about the root of the static force instead
(`Trap.find_equilibrium`), e.g. the point $z_\mathrm{eq}$ where the scattering force balances the
gradient force. The diagonal of $K$ gives `kx`, `ky`, `kz` and $\Omega_j = \sqrt{K_{jj}/m_n}$
(`override_Ω_dict` still replaces the frequencies). For the default trap $\Omega_x/2\pi$ = 156.34 kHz,
as before; adding `HarmonicPotential([1e-6, 0, 0])` raises it to 198.74 kHz.

When the elements have different principal axes, $K$ has off-diagonal entries. They are kept as
`Trap.intra_trap_coupling_stiffness`, and converted to levisim's coupling convention (the term
$-g_{jk} q_j q_k$ of $H/\hbar$, $q_j = b_j + b_j^\dagger$, as in
`MultiTrapBase.mechanical_coupling_matrix`) by

$$
g_{jk} = -\frac{K_{jk}\, x_{j,\mathrm{zpf}}\, x_{k,\mathrm{zpf}}}{\hbar}.
$$

`Trap.intra_trap_coupling_matrix` is that $3\times3$ matrix (zero diagonal) and
`Trap.intra_trap_couplings` the dict `{'xy': gxy, 'yx': gyx, ...}`, where `jk` is the rate at which
axis $k$ drives axis $j$. For a tweezer plus a harmonic potential of stiffness
(1, 0.5, 0) μN/m rotated by 0.3 rad about $z$, $g_{xy}/2\pi$ = -5.93 kHz. `MultiTrap` adds the same
conversion of $K_{jk}$, at its dimensions' (possibly shifted) zero-point fluctuations, to its
`mechanical_coupling_matrix` for every pair of modelled axes of one particle, so a hybrid trap with
misaligned axes has the normal modes of its full Hessian: the eigenfrequencies $\sqrt{\lambda_i/m_n}$ of
$K$ (tested against central differences of the summed force). For a plain tweezer trap $K$ is diagonal
and nothing is added; `MultiTrap(intra_trap_couplings=False)` leaves the couplings out.

Gravity has no stiffness, but it moves the equilibrium: for a harmonic trap the sag along each axis
is $g\,\hat d_j/\Omega_j^2$, about 10 pm along $x$ for the default tweezer
(`Trap(fields=[Tweezer(), Gravity()]).find_equilibrium()` gives $-1.016 \times 10^{-11}$ m).

## Optical-only quantities

The polarisability $\alpha_\mathrm{eff}$, the trap depth `U_depth`, `photon_recoil_heating_rates`,
`σ_scattering`, `z_eq`, `Utw`, `Ftw`, `F_scattering` and the cavity drive all refer to the trap's
optical element, the first element with `is_optical` (`Trap.tweezer`). A trap without one raises
{class}`~levisim.physics.trap.MissingOpticalFieldError`, a `ValueError` that names the trap's
elements. A particle held by a non-optical trap scatters no light into a cavity unless a
`cavity_drive_tweezer` is given. The mechanical quantities, $\Omega_j$, $\Gamma_j$, $A$, $T_\mathrm{eff}$,
the zero-point fluctuations and the theory spectra, work for every trap.

`Utw` (and `Tweezer.potential`) is the cycle-averaged dipole energy
$-\tfrac14\mathrm{Re}(\alpha_\mathrm{eff})\lvert E\rvert^2$, whose gradient is the force `Ftw` and
whose curvature at the focus is `kx`, `ky`, `kz`; `U_depth` $= \tfrac14\mathrm{Re}(\alpha_\mathrm{eff})E_0^2$
and so `v_escape`, `p_escape` follow from it. Up to levisim 0.1 these used the peak energy
$-\tfrac12\mathrm{Re}(\alpha_\mathrm{eff})\lvert E\rvert^2$, twice the potential of the force
(`U_depth` was 2×, `v_escape` and `p_escape` √2× too large); the forces, frequencies and spectra are
unchanged, bit for bit.

## In the simulation

{class}`~levisim.simulation.free_space.FreeSpaceSimulation` adds the forces in a fixed order, set by
each element's `simulation_group`:

$$
\dot p_j = \underbrace{\textstyle\sum F^{\mathrm{field}}_j}_{\text{tweezers, potentials}}
- \Gamma_j p_j + F^{\mathrm{kick}}_j \;\Big[+ F^{\mathrm{recoil}}_z\Big] \;\Big[+ F^{\mathrm{gravity}}_j\Big],
\qquad \Gamma_j = \Gamma_\mathrm{th} + \Gamma_{\mathrm{fb},j}.
$$

This is the order of the single-tweezer code, so every seeded trace of an existing configuration is
unchanged. `gravity=True` adds a `Gravity` element (a trap may instead hold one in its fields, but not
both). The compiled numba kernel covers the set it always did: one tweezer (or `override_Ω_dict`),
feedback, kicks, the tweezer's photon recoil and gravity. A trap with any other element runs the
NumPy loop, and says so in a debug message. A generic compiled path would need each element to
supply a numba-compilable force function and a packed parameter array.

## Adding a new kind of trap

A new trap is a new element. Two non-optical ones are in place:

- The **Paul trap** (`PaulTrap`, see the guide [Paul and hybrid traps](../guide/paul_traps.md)) has a
  time-dependent `force`, $Q\,\mathbf{E}_\mathrm{rf}(\mathbf{r})\cos\Omega_\mathrm{rf}t$ plus the static
  part, so the time-domain simulation shows the micromotion and the Mathieu stability boundary. Its
  `stiffness` returns the secular (pseudopotential) stiffness, to lowest order in the Mathieu
  parameters, which is what `MultiTrap` and $\Omega_j$ use, and `Trap.static_force` takes its
  time-averaged `secular_force` (elements declare this with `is_time_dependent`). Its `noise` is the
  electrode voltage noise as a force noise $Q\,E_\mathrm{noise}$, which `FreeSpaceSimulation` adds.
- The **magnetic trap** (`MagneticTrap`, see the guide [Magnetic levitation](../guide/magnetic_levitation.md))
  for a diamagnetic particle is conservative, with `potential` $U = -(V/2\mu_0)\mathbf{B}\cdot\chi_\mathrm{eff}\mathbf{B}$
  (shape-corrected, tensor χ in the body frame); its `force` is $-\nabla U$ and its `stiffness` the
  Hessian of $U$, analytic in the derivatives of the field of a `MagneticFieldSource` (local expansion,
  multipoles, current loops, coil pairs, magnet blocks or a user callable).

Combined with `Gravity`, and with each other or a `Tweezer`, they form hybrid traps without any change
to `Trap`: its frequencies come from the summed stiffness, its couplings from the off-diagonal part.

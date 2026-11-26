# Interactions between particles

`levisim.physics.interactions` describes forces between levitated particles. Each interaction gives
the force on every particle at laboratory-frame positions, for the time domain, and the stiffness
blocks at an equilibrium, for `MultiTrap`. One definition therefore feeds both solvers.

```python
from levisim.physics.interactions import Coulomb, NewtonianGravity, Casimir

F = Coulomb().forces(positions, particles)              # (N, 3) [N]
K = Coulomb().stiffness_blocks(positions, particles)    # (N, N, 3, 3) [N/m], K[j, k] = -∂F_j/∂r_k
```

Every simulation that takes `interactions=[...]` calls `forces` once per drift evaluation. An
interaction also has `potential(positions, particles)` [J] when it is conservative, and the flags
`is_conservative` and `is_reciprocal`. A non-reciprocal interaction has $K_{jk} \neq K_{kj}^\top$, and
`MultiTrap` keeps the two directions apart. `pairs=[(j, k), ...]` restricts an interaction to some
pairs (0-based particle indices).

## The interactions

All three are central pair potentials of the form $U = C\,(r - s)^{-n}$ in the centre-to-centre
distance $r$ (`PowerLawPairInteraction`). The forces $F_j = -U'(r)\hat r = -F_k$ and the blocks
$K_{jj} = K_{kk} = -K_{jk} = U''\hat r\hat r^\top + (U'/r)(\mathbb 1 - \hat r\hat r^\top)$ follow in
closed form. They obey Newton's third law exactly.

| Interaction | $U(r)$ | Validity |
|---|---|---|
| `Coulomb` | $q_jq_k/(4\pi\varepsilon_0 r)$, with $q$ = `Particle.q` = charge·e | point charges, or spherically symmetric charge, outside contact; no image charges |
| `NewtonianGravity` | $-Gm_jm_k/r$ (CODATA $G$) | spheres of spherically symmetric density outside contact (shell theorem) |
| `Casimir(model='casimir_polder')` | $-\dfrac{23\hbar c}{4\pi}\dfrac{\alpha_j\alpha_k}{r^7}$, $\alpha = R^3\dfrac{\varepsilon - 1}{\varepsilon + 2}$ | $r \gg R_j + R_k$ (dipole limit); retarded limit, $r$ much larger than the material's absorption wavelengths |
| `Casimir(model='proximity_force')` | $-\eta\,\dfrac{\pi^3\hbar c R_\text{eff}}{720\,d^2}$, $d = r - R_j - R_k$, $R_\text{eff} = \dfrac{R_jR_k}{R_j + R_k}$ | gap $d \ll R$ (proximity force); $d$ much larger than the absorption wavelengths |

Sources of the Casimir formulas, checked against the papers:

- The Casimir-Polder form is that of two polarisable dipoles (Casimir and Polder, Phys. Rev. **73**, 360
  (1948)) with the static polarisability volume of a dielectric sphere. It is written in the same form
  as Eq. (4) of van de Kamp et al., Phys. Rev. A **102**, 062807 (2020), which uses it for
  the gravity-entanglement (QGEM) proposal.
- The proximity-force form is the Derjaguin approximation for two spheres,
  $F = 2\pi R_\text{eff}\,\mathcal E(d)$, with $\mathcal E$ the energy per area between half-spaces.
  For ideal mirrors $\mathcal E = -\pi^2\hbar c/(720 d^3)$ (Klimchitskaya, Mohideen and Mostepanenko,
  Rev. Mod. Phys. **81**, 1827 (2009), Eqs. (2) and (16)). For two equal spheres this gives
  $-\pi^3\hbar cR/(1440d^2)$, the expression quoted by Krisnanda et al., npj Quantum Inf. **6**, 12 (2020).
- $\eta$ = `lifshitz_plate_factor(ε_j, ε_k)` is the zero-temperature Lifshitz energy of two dielectric
  half-spaces with frequency-independent permittivity, relative to ideal mirrors (RMP **81**, Eqs. (3)-(7)).
  With constant $\varepsilon$ the wavevector integral is done in closed form:
  $\eta = \tfrac{45}{\pi^4}\int_0^1\mathrm dp\,[\mathrm{Li}_4(r_\text{TM}^2) + \mathrm{Li}_4(r_\text{TE}^2)]$.
  It tends to 1 for $\varepsilon\to\infty$. For $\varepsilon \to 1$ it tends to
  $69(\varepsilon - 1)^2/(8\pi^4)$, the pairwise sum of the Casimir-Polder $r^{-7}$ energy, which ties
  the two Casimir limits together (tested). For silica with static $\varepsilon$ = 3.8, $\eta$ = 0.124.

Both Casimir forms are zero-temperature results. They hold for separations below the thermal
wavelength $\hbar c/k_BT$ (7.6 μm at 300 K, 0.57 mm at 4 K). Beyond it the zero-frequency thermal
term dominates, and it is not modelled. Between the two limits ($d \sim R$) neither form is accurate,
and a WARNING is logged once per instance when either is used outside its range. `permittivity=`
sets the static $\varepsilon$. A particle's `εr` is levisim's *optical* permittivity (1.98 for the
default silica), which underestimates the static value.

## Optical binding

`OpticalBinding` is the force that light scattered by one particle exerts on another. It is neither
conservative nor reciprocal, so it is not a pair potential. Each particle $n$ sits in its own tweezer
(trap $n$, focus at `trap.position`), which induces the dipole $p_n = \alpha_n E_{\mathrm{tw},n}(r_n)$. The
dipole radiates the full free-space dipole field $\mathfrak G(r - r_n)\,p_n$, near, intermediate and far
terms included:

$$
\mathfrak G(\mathbf r) = \frac{e^{ikr}}{4\pi\varepsilon_0}\left[\frac{k^2}{r}(\mathbb 1 - \hat r\hat r) + (3\hat r\hat r - \mathbb 1)\left(\frac1{r^3} - \frac{ik}{r^2}\right)\right].
$$

Particle $n$ feels the time-averaged force of the field at its position. With levisim's complex
amplitudes, $E(t) = \mathrm{Re}[E\,e^{-i\omega t}]$, the force is

$$
\mathbf F_n = \tfrac12\,\mathrm{Re}\sum_i p_{n,i}^*\,\nabla E_{\mathrm{loc},n,i}(\mathbf r_n)
\;-\;(\text{force of its own tweezer on the isolated particle}),
\qquad
E_{\mathrm{loc},n} = E_{\mathrm{tw},n} + \sum_{m\neq n}\mathfrak G(\mathbf r - \mathbf r_m)\,p_m ,
$$

where the gradient acts on the field and not on the dipole. Two orders are available.

- `order='first'` (default) keeps every term of second order in the polarisabilities. This is the
  optical binding force of Rieser et al., Science **377**, 987 (2022), Eq. (S2), and Rudolph et al.,
  PRL **133**, 233603 (2024), Eq. (1), generalised to complex tensor polarisabilities. It has two parts:
  the field scattered by $p_m$ acting on $p_n$, and the tweezer acting on the dipole
  $\delta p_n = \alpha_n\mathfrak G p_m$ that this field induces. Its stiffness blocks are analytic, from
  closed-form second derivatives of $\mathfrak G$ and of the tweezer field.
- `order='self_consistent'` solves the coupled-dipole equations $p = \alpha(E_\mathrm{tw} + \mathfrak Gp)$
  exactly, which includes multiple scattering. Its stiffness blocks are fourth-order central
  differences of the forces. The two orders differ by a relative amount of order
  $|k^3\alpha/4\pi\varepsilon_0|/kd$: 0.04 % at 2.2 μm for 105 nm silica spheres.

`α_n` is the lab-frame polarisability with the radiation reaction:
`Tweezer.effective_polarisability`·𝟙 for an isotropic sphere, and
`Particle.effective_polarisability_tensor` for shapes, so a tilted spheroid's dipole leaves the
polarisation direction. Each tweezer is levisim's paraxial elliptical Gaussian beam,

$$
E_\mathrm{tw}(s) = E_0 e^{i\phi}\,\hat e\;\frac{\exp\!\left(ikz - \dfrac{x^2/w_{x0}^2 + y^2/w_{y0}^2}{1 + iz/z_R}\right)}{1 + iz/z_R},
$$

at the displacement $s$ from the focus, with $\phi$ = `Tweezer.φ`. On the axis its phase is
$kz - \arctan(z/z_R) + \phi$, the phase of the coherent-scattering model. The phase gradient at the
focus is therefore $k' = k - 1/z_R$, not $k$. `tweezer_field='plane_wave'` uses $E_0e^{i\phi}\hat e\,e^{ikz}$
instead, for comparison with plane-wave theory.

```python
from levisim.physics import OpticalBinding

mtrap = MultiTrap(num_traps=2, Cartesian_axes_per_trap=3, trap_positions=[(0, 0, 0), (3.2e-6, 0, 0)],
                  φn_dict={1: 0.5 * π, 2: 0.0}, interactions=[OpticalBinding()])
binding = mtrap.interactions[0]                   # MultiTrap attached its traps
parts = binding.coupling_decomposition(mtrap.interaction_positions, [t.bead for t in mtrap.traps])
parts['reciprocal'], parts['antireciprocal']      # (K ± Kᵀ)/2 over [x1, y1, z1, x2, ...] [N/m]
```

**Traps.** The interaction needs the tweezers and their foci, which the generic
`forces(positions, particles)` interface does not carry. Give them with `OpticalBinding(traps=[...])`,
or let the host attach its own. `MultiTrap`, `MultiParticleFreeSpaceSimulation` and the nonlinear
`CavitySimulation` call `Interaction.attach_traps(traps)` for every interaction, and the base-class
method does nothing.

**Reciprocal and anti-reciprocal parts.** For two particles a distance $d$ apart across the
polarisation, coupled along their tweezer axes, the far-field limit of the stiffness is Rieser et
al.'s Eq. (1). Writing $\Delta\phi = \phi_1 - \phi_2$,

$$
K_{zz} \to \begin{pmatrix} k_1 + k_2 & -(k_1 + k_2)\\ -(k_1 - k_2) & k_1 - k_2\end{pmatrix},\qquad
k_1 = \mathcal G\frac{\cos kd\,\cos\Delta\phi}{kd},\quad k_2 = \mathcal G\frac{\sin kd\,\sin\Delta\phi}{kd},
\quad \mathcal G = \frac{\alpha^2k^3k'^2E_1E_2}{8\pi\varepsilon_0}.
$$

$k_1$ is reciprocal, and $k_2$ is the curl of the force. `MultiTrap` turns these into the rates
$g_{jk} = -K_{jk}x_{\text{zpf},j}x_{\text{zpf},k}/\hbar$, so $g_r \pm g_a$ = `G[z1, z2]`, `G[z2, z1]`. The tests
check this limit to $2\times10^{-3}$ of $\mathcal G/kd$ at $d$ = 64 μm and 1 mm, for a 20 nm sphere. Without
the Gouy phase (`tweezer_field='plane_wave'`, $k' = k$) the limit is the published one. On the diagonal, the axial fall of the Gaussian amplitude adds $1/(k'z_R)^2$ to
the curvature. With complex $\alpha$, the radiation reaction adds $O(\mathrm{Im}\,\alpha/\mathrm{Re}\,\alpha)$.
For 105 nm silica that is 4 %.

```{figure} figures/optical_binding.png
Axial coupling of two 105 nm silica spheres in 400 mW tweezers with 730 nm waists (Rieser et
al.'s parameters), in units of the trap frequency. Left: at $\Delta\phi = \pi/2$ the two directions differ
in sign and magnitude. Middle: reciprocal and anti-reciprocal parts against the optical phase at
3.2 μm. Right: a dipole along $y$ radiates no far field along $y$. Separated along the polarisation,
the coupling has no $1/d$ term and falls as $1/d^2$. Script: `docs/concepts/scripts/optical_binding_figures.py`.
```

**Momentum balance.** The two forces are not equal and opposite, $F_1 + F_2 \neq 0$. The difference is
momentum carried away by light. It has two parts: the interference of the fields the two dipoles
radiate (each dipole alone radiates symmetrically and takes no net momentum), and the change in
the light the particles remove from the tweezer beams (the $\delta p$ terms). The first part is tested
exactly for two fixed dipoles: $F_{12} + F_{21} = -\tfrac{\varepsilon_0}{2}\oint|E|^2\hat n\,r^2\,\mathrm d\Omega$,
cross term only, to $10^{-9}$ (`tests/test_optical_binding.py`). The second part, the momentum
exchanged with the beams, is not represented in levisim, which has no field degrees of freedom for
the tweezers.

**Approximations.**

- Point dipoles: a WARNING is logged once when $kR > 0.6$.
- Paraxial tweezers, with no longitudinal field.
- Vacuum between the particles.
- Each particle is lit by its own tweezer only. Another tweezer's Gaussian field at a particle is
  neglected, and a WARNING is logged once when it exceeds `overlap_tolerance` = 10⁻³ of the
  particle's own. Real high-NA foci have Airy tails, which Rieser et al. had to include in their
  fits.
- Tweezers of different wavelengths do not interfere on average. Their particles do not bind (the
  pair is skipped, with a WARNING).
- Interference between the recoil noise of the two particles (correlated shot noise, Rudolph et al.
  Eq. (8)) is not modelled. Each particle's recoil heating stays independent.

The time domain evaluates the full, un-linearised force at every step, with the Gaussian envelope
and the Gouy phase. This is what saturates the non-reciprocal instability on a limit cycle. Only
the NumPy loop runs it: `MultiParticleFreeSpaceSimulation` with `backend='auto'` falls back from numba
(an explicit `backend='numba'` raises `UnsupportedBackendError`), and the nonlinear `CavitySimulation`
is not compilable with interactions. One force evaluation for
two particles takes about 70 μs.

Reproductions: `notebooks/papers/rieser2022_optical_binding.ipynb` (coupling against separation,
optical phase and polarisation) and `notebooks/papers/reisenbauer2024_nonreciprocal_binding.ipynb`
(exceptional points and limit cycles).

## Positions

`Trap(position=(X, Y, Z))` places a trap (its origin, the tweezer focus) in the laboratory frame,
default $(0, 0, 0)$. Its axes are parallel to the laboratory's. `MultiTrap(trap_positions=[...])` (or
`position` in each trap's `trap_kwargs`) sets them for every trap. The cavity coupling still uses each
bead's standing-wave phase $\Phi$. If two traps are displaced along the cavity axis
$\hat d = (\sin\theta, \cos\theta, 0)$ by an amount inconsistent with their phases,
$\Phi_n - \Phi_1 \not\equiv k\hat d\cdot(X_n - X_1) \pmod{2\pi}$, a WARNING is logged.

## Linearisation in `MultiTrap`

`MultiTrap(interactions=[...])` evaluates the stiffness blocks at the particles' positions
`trap.position + trap.linearisation_point`, taken as their equilibria (`setup_interactions`):

- the diagonal blocks add to each particle's static stiffness, like the static cavity shifts:
  $\Omega_j^2 = \Omega_{j,\text{tw}}^2 + K_{jj}/m$. Every zero-point fluctuation, occupation and
  coupling uses the shifted $\Omega$.
- every other entry becomes a direct coupling of `mechanical_coupling_matrix`. A stiffness term
  $\tfrac12 K_{jk}x_jx_k$ in $H$, with $x = x_\text{zpf}(b + b^\dagger) = x_\text{zpf}\,q$, gives the
  force $F_j = -K_{jk}x_k$ and $\mathrm d b_j/\mathrm dt \ni iF_jx_{\text{zpf},j}/\hbar$. In levisim's
  convention $H/\hbar \ni -g_{jk}q_jq_k$ (so $\mathrm db_j/\mathrm dt \ni +ig_{jk}q_k$) this is

  $$
  g_{jk} = -\frac{K_{jk}\,x_{\text{zpf},j}\,x_{\text{zpf},k}}{\hbar}.
  $$

The static forces of the interactions at those positions are kept in `interaction_static_forces`.
They do not move the particles: place the traps at the equilibrium positions instead. The
displacement each force would cause in its trap alone is logged at DEBUG. In toy mode,
`toy_params['g_direct'] = {('z1', 'z2'): (g_12, g_21)}` sets the rates directly (a single number for
$g_{12} = g_{21}$). All seven calculation methods support these couplings
([calculation methods](calculation_methods.md)), and so does the stability analysis
([stability](stability.md)).

## Time domain

`MultiParticleFreeSpaceSimulation(traps=[...], interactions=[...])` integrates $N$ particles, each with
its own trap physics (the equations of `FreeSpaceSimulation`), plus the pair forces at
$R_n = $ `trap.position` + $(x_n, y_n, z_n)$. The noises of different particles are independent. One
particle without interactions reproduces `FreeSpaceSimulation` bit for bit. The numba kernel covers
single-tweezer traps (or `override_Ω_dict`) with feedback, photon recoil and gravity, and the
power-law interactions. It accumulates each pair's force in local scalars and writes it once to both
particles. Other configurations run the NumPy reference loop.

Validation (`tests/test_multi_particle_simulation.py`): two particles of 20 e each, 4 μm apart along
the tweezer $y$ axis, both with $\Omega_y/2\pi$ = 100 kHz. The stiffness blocks at the equilibrium
predict normal modes split by 427.76 Hz. A displacement of particle 1 alone passes completely to
particle 2 after $\pi/\Delta\omega$ = 1.169 ms, to within two oscillation periods, and the spectra of
the sum and the difference of the two records peak at the predicted normal modes. The notebook
`notebooks/papers/penny2023_sympathetic_cooling.ipynb` reproduces the normal modes of Penny et al.,
Phys. Rev. Research **5**, 013070 (2023), and simulates sympathetic cooling.

## Entanglement between particles

levisim's states are Gaussian (the linearised dynamics), so entanglement between particles is the
logarithmic negativity of their covariance matrix. The tools are in `levisim.multiphysics.covariance`:

- `quadrature_matrices(mtrap, cavity=True, extra_heating=None)`: the drift $A$ and diffusion $D$ in
  quadratures. `cavity=False` keeps only the mechanical modes: the particles in their traps with their
  direct couplings and static shifts, as without the cavity. `extra_heating={label: Γ_h}` adds heating
  rates [phonons/s] as $2\Gamma_h$ on both quadratures, the same as raising the bath occupation by
  $\Gamma_h/\Gamma$.
- `steady_state_covariance(mtrap, cavity=..., extra_heating=...)`: the Lyapunov steady state.
- `product_state_covariance(mtrap, occupations, cavity=..., squeezing=...)`: uncorrelated thermal or
  squeezed thermal initial states (`gaussian_states.squeezed_thermal_covariance`).
- `evolve_covariance(mtrap, V0, times, ...)`: the exact solution of $\dot V = AV + VA^\top + D$ at the
  given times (Van Loan's block exponential on substeps with $\lVert A\rVert_1\tau \le 1$, recombined
  exactly), for protocols that start from a cooled product state.
- `particle_log_negativity(mtrap, V, particles_A, particles_B=None)`: $E_N$ between all modelled motions of
  some particles and those of others, with the other modes traced out (natural logarithm).

Validation (`tests/test_entanglement_outputs.py`) uses two osmium spheres coupled by gravity in
harmonic traps, the trapped-mass case of Krisnanda et al., npj Quantum Inf. **6**, 12 (2020). The tests
check the figure of merit $\eta\omega^2$ = 1.364 × 10⁻⁶ s⁻² (paper 1.36 × 10⁻⁶), the ground-state
maximum $E_N \approx \eta$ (to $O(\eta^2)$), the thermal amplitudes $\eta - \ln(2n + 1)$ and the
squeezed maxima $\approx |s_A + s_B|$. The notebook
`notebooks/papers/krisnanda2020_gravity_entanglement.ipynb` maps where the entanglement survives gas
collisions and blackbody radiation. The spin-superposition protocol of Bose et al., Phys. Rev. Lett.
**119**, 240401 (2017), is not Gaussian and is out of scope.

## Decoherence budget

`levisim.multiphysics.decoherence.decoherence_budget(mtrap, T_environment=None, T_internal=None,
ε_bb=2.1 + 0.57i, electric_field_noise=0)` returns, for every mode, the heating rates in phonons per
second:

| Mechanism | Rate | Source |
|---|---|---|
| gas | $\Gamma_\mathrm{gas}n_\mathrm{th}$ | levisim's gas bath (`Dimension.Γ`, Bose-Einstein $n_\mathrm{th}$) |
| photon recoil | `Trap.photon_recoil_heating_rates`, scaled to the mode's $\Omega$ | as in `MultiTrap(photon_recoil=True)` |
| blackbody scattering, emission, absorption | $2\Lambda x_\mathrm{zpf}^2 = \hbar\Lambda/(m\Omega)$ | localisation parameters, Romero-Isart, Phys. Rev. A **84**, 052121 (2011), Eqs. (28)-(29) |
| electric-field noise | $q^2S_E(\Omega)/(4m\hbar\Omega)$, $S_E$ single-sided | Brownnutt et al., Rev. Mod. Phys. **87**, 1419 (2015), Eq. (10) without micromotion |

The rate $2\Lambda x_\mathrm{zpf}^2$ follows from $\dot\rho = -\Lambda[x, [x, \rho]]$, which gives
$\mathrm d\langle p^2\rangle/\mathrm dt = 2\hbar^2\Lambda$. The default $\varepsilon_\mathrm{bb}$ is the
silica value of Romero-Isart's Table III. The blackbody rates are those of a point dipole, so they hold
for $Rk_BT/\hbar c \ll 1$. Above 0.1 a WARNING is logged and the modes are listed in
`budget.warnings`, because the $R^6T^9$ scattering term then grossly overestimates the rate.
`budget.extra_heating()` sums the mechanisms that the MultiTrap's noise does not already contain, for
`extra_heating=`. `budget.table()` prints the budget and `budget.to_dataframe()` returns it as a pandas
DataFrame.

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

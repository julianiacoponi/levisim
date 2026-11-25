"""
Optical binding: the force between particles from the light that each scatters into the other's
tweezer

Every particle n sits in its own tweezer (Trap n, focus at ``trap.position``), which polarises it.
The induced dipole radiates the full free-space dipole field (near, intermediate and far terms), and
particle n feels the time-averaged force of the field at its position (levisim's convention for the
complex amplitudes, E(r, t) = Re[E(r)·e^{-iωt}])::

    F_n = ½ Re Σ_i p_n,i* ∂E_loc,n,i/∂r_n  -  (the force of its own tweezer on the isolated particle)

    E_loc,n(r) = E_tw,n(r) + Σ_{m≠n} 𝔊(r - r_m)·p_m,     p_n = α_n·E_loc,n(r_n)

with 𝔊 the free-space dipole Green tensor (`dipole_green_tensor`) and α_n the particle's lab-frame
polarisability with the radiation reaction (a scalar times 𝟙 for an isotropic sphere, the tensor of
`Particle.effective_polarisability_tensor` for any other particle). The derivative acts on the field
only; the dipoles are held fixed. Two orders are available:

- ``order='first'`` (the optical binding force of Rieser et al., Science 377, 987 (2022), Eq. (S2),
  and Rudolph et al., PRL 133, 233603 (2024), Eq. (1), generalised to complex tensor
  polarisabilities): every term of second order in the polarisabilities, i.e. the field scattered
  by the tweezer-induced dipole p_m = α_m·E_tw,m(r_m) acting on p_n = α_n·E_tw,n(r_n), plus the
  tweezer acting on the dipole δp_n = α_n·Σ_m 𝔊·p_m that this scattered field induces::

      F_n,a = Σ_m ½ Re[ u_n†·α_n†·∂_a𝔊(r_n - r_m)·α_m·u_m  +  (∂_a u_n)†·α_n·𝔊(r_n - r_m)·α_m·u_m ]

  with u_n = E_tw,n(r_n). Terms of third and higher order (multiple scattering, and the scattering
  force of the scattered light alone) are left out.
- ``order='self_consistent'``: the coupled-dipole equations p = α·(E_tw + 𝔊·p) are solved exactly
  (all orders of scattering), and the force is the total force on each dipole in its local field,
  less the force of its own tweezer on the isolated particle. It agrees with 'first' to relative
  order k³·abs(α)/(4πε0·kd) at large separations d.

The forces on the particles are computed separately: they are NOT equal and opposite. The
difference, F_1 + F_2 ≠ 0, is the momentum carried away by the scattered light (the interference
of the fields the particles radiate, and the change in the light they remove from the tweezers);
see ``docs/concepts/interactions.md``. The interaction is therefore non-conservative and
non-reciprocal (`is_conservative = is_reciprocal = False`), and its stiffness blocks have both a
reciprocal and an anti-reciprocal part (`coupling_decomposition`).

Tweezer fields. Each tweezer is levisim's paraxial elliptical Gaussian beam, propagating along +z
and polarised along the tweezer's `polarisation` (lab y), with the complex amplitude ::

    E_tw(s) = E0·e^{iφ}·ê · exp(i k z) · exp(-(x²/wx0² + y²/wy0²)/(1 + i z/zR)) / (1 + i z/zR)

at the displacement s = (x, y, z) from the focus. Its modulus is `Tweezer.E_field_amplitude` and
its phase on axis is k·z - arctan(z/zR) + φ, the phase of the coherent-scattering model, with φ
the tweezer's phase `Tweezer.φ`. ``tweezer_field='plane_wave'`` replaces it by E0·e^{iφ}·ê·e^{ikz}
(the approximation of Reisenbauer et al., Nat. Phys. 20, 1629 (2024), Methods), which is useful
to compare with plane-wave theory.

Approximations:

- point dipoles (the particle radius R must be well below the wavelength; a WARNING is logged once
  when kR > 0.6), paraxial tweezers (no longitudinal field component), vacuum between the particles
- each particle is illuminated by its own tweezer only: the field of tweezer m at particle n is
  neglected. For Gaussian beams it is exp(-d²/w0²) of the local field at a separation d, but real
  high-NA foci have Airy tails that interfere with the scattered light (Rieser et al.,
  supplementary material). A WARNING is logged once when the neglected Gaussian field of another
  tweezer exceeds `overlap_tolerance` (default 10⁻³) of a particle's own
- particles in tweezers of different wavelengths do not interfere: their time-averaged binding force
  is zero (the pair is skipped, with a WARNING logged once)

The traps (for the tweezers and their foci) are given with ``OpticalBinding(traps=[...])`` or are
attached by the host: `MultiTrap(interactions=[...])`, `MultiParticleFreeSpaceSimulation` and the
nonlinear `CavitySimulation` call `attach_traps` with their own traps.
"""

import numpy as np

from levisim.core.constants import ε0, π
from levisim.core.logger import logger
from levisim.physics.interactions.base import Interaction, as_positions, particle_radius

__all__ = [
    'OpticalBinding',
    'dipole_field',
    'dipole_green_tensor',
    'dipole_green_tensor_derivatives',
    'tweezer_field',
]

COULOMB_FACTOR = 1 / (4 * π * ε0)


# --------------------------------------------------------------------------------------------------
# FREE-SPACE DIPOLE GREEN TENSOR
# --------------------------------------------------------------------------------------------------
def _radial_coefficients(k):
    """
    The two radial functions of 𝔊_ij = f(r)·δ_ij + h(r)·x_i·x_j as {power: coefficient} of
    e^{ikr}·Σ c_p r^p, times 1/(4πε0)
    """
    f = {-1: k**2, -2: 1j * k, -3: -1.0}
    h = {-3: -k**2, -4: -3j * k, -5: 3.0}
    return f, h


def _derivative(coefficients, k):
    """d/dr of e^{ikr}·Σ c_p r^p, as the coefficients of e^{ikr}·Σ c'_p r^p"""
    result = {}
    for p, c_p in coefficients.items():
        result[p] = result.get(p, 0) + 1j * k * c_p
        if p != 0:
            result[p - 1] = result.get(p - 1, 0) + p * c_p
    return result


def _evaluate(coefficients, r, phase):
    return phase * sum(c_p * r**p for p, c_p in coefficients.items())


def dipole_green_tensor(r_vector, k):
    """
    Free-space dipole Green tensor 𝔊(r) at the displacement r from a point dipole, so that its
    field is E(r) = 𝔊(r)·p (levisim's E(t) = Re[E·e^{-iωt}], outgoing e^{ikr}) ::

        𝔊(r) = e^{ikr}/(4πε0) · [ k²·(𝟙 - r̂r̂)/r + (3r̂r̂ - 𝟙)·(1/r³ - ik/r²) ]

    with the near (1/r³), intermediate (1/r²) and far (1/r) terms.
    r_vector: shape (..., 3) [m]; k [1/m]
    [m/F] (shape (..., 3, 3))
    """
    return dipole_green_tensor_derivatives(r_vector, k, order=0)[0]


def dipole_green_tensor_derivatives(r_vector, k, order=2):
    """
    𝔊(r) and its derivatives with respect to the field point, in closed form ::

        G[..., i, j]          = 𝔊_ij
        dG[..., a, i, j]      = ∂_a 𝔊_ij
        d2G[..., a, b, i, j]  = ∂_a ∂_b 𝔊_ij

    (as many as `order` asks for). With 𝔊_ij = f(r)·δ_ij + h(r)·x_i·x_j ::

        ∂_a 𝔊_ij = f'·x_a/r·δ_ij + h'·x_a/r·x_i·x_j + h·(δ_ai·x_j + δ_aj·x_i)

    and the second derivatives by the product rule once more.
    [m/F, m/F/m, m/F/m²]
    """
    x = np.asarray(r_vector, dtype=float)
    r = np.sqrt(np.sum(x**2, axis=-1))
    if np.any(r <= 0):
        raise ValueError('dipole_green_tensor: the field point coincides with the dipole (r = 0)')
    phase = COULOMB_FACTOR * np.exp(1j * k * r)
    f_c, h_c = _radial_coefficients(k)
    f = _evaluate(f_c, r, phase)
    h = _evaluate(h_c, r, phase)
    eye = np.eye(3)
    xx = x[..., :, None] * x[..., None, :]
    G = f[..., None, None] * eye + h[..., None, None] * xx
    results = [G]
    if order < 1:
        return results
    f1_c, h1_c = _derivative(f_c, k), _derivative(h_c, k)
    f1 = _evaluate(f1_c, r, phase)
    h1 = _evaluate(h1_c, r, phase)
    n = x / r[..., None]
    # δ_ai·x_j + δ_aj·x_i
    sym = eye[:, :, None] * x[..., None, None, :] + eye[:, None, :] * x[..., None, :, None]
    dG = (
        (f1[..., None] * n)[..., :, None, None] * eye
        + (h1[..., None] * n)[..., :, None, None] * xx[..., None, :, :]
        + h[..., None, None, None] * sym
    )
    results.append(dG)
    if order < 2:
        return results
    f2 = _evaluate(_derivative(f1_c, k), r, phase)
    h2 = _evaluate(_derivative(h1_c, k), r, phase)
    nn = n[..., :, None] * n[..., None, :]
    # ∂_b (x_a/r) = (δ_ab - n_a n_b)/r
    dn = (eye - nn) / r[..., None, None]
    radial_f = f2[..., None, None] * nn + f1[..., None, None] * dn
    radial_h = h2[..., None, None] * nn + h1[..., None, None] * dn
    d2G = (
        radial_f[..., :, :, None, None] * eye
        + radial_h[..., :, :, None, None] * xx[..., None, None, :, :]
        # h'·n_a·(δ_bi x_j + δ_bj x_i) + h'·n_b·(δ_ai x_j + δ_aj x_i)
        + (h1[..., None] * n)[..., :, None, None, None] * sym[..., None, :, :, :]
        + (h1[..., None] * n)[..., None, :, None, None] * sym[..., :, None, :, :]
        # h·(δ_ai δ_bj + δ_aj δ_bi)
        + h[..., None, None, None, None] * (
            eye[:, None, :, None] * eye[None, :, None, :] + eye[:, None, None, :] * eye[None, :, :, None]
        )
    )
    results.append(d2G)
    return results


def dipole_field(r, r_dipole, p, k):
    """
    Complex electric field of the point dipole p at r_dipole, at the point r, written out as in
    Jackson, Classical Electrodynamics, 3rd ed., Eq. (9.18) (SI) ::

        E = e^{ikR}/(4πε0) · { k²·(n × p) × n / R + [3n(n·p) - p]·(1/R³ - ik/R²) },   R = |r - r_dipole|

    It is the independent reference of `dipole_green_tensor` in the tests.
    [V/m]
    """
    R_vector = np.asarray(r, dtype=float) - np.asarray(r_dipole, dtype=float)
    R = np.sqrt(R_vector @ R_vector)
    n = R_vector / R
    p = np.asarray(p, dtype=complex)
    far = k**2 * np.cross(np.cross(n, p), n) / R
    near = (3 * n * (n @ p) - p) * (1 / R**3 - 1j * k / R**2)
    return COULOMB_FACTOR * np.exp(1j * k * R) * (far + near)


# --------------------------------------------------------------------------------------------------
# TWEEZER FIELD
# --------------------------------------------------------------------------------------------------
def tweezer_field(tweezer, s, order=2, model='gaussian'):
    """
    Complex field of a Tweezer at the displacement s from its focus, and its derivatives ::

        E[i], dE[a, i] = ∂_a E_i, d2E[a, b, i] = ∂_a ∂_b E_i

    model='gaussian': the paraxial elliptical Gaussian beam (see the module docstring) ::

        E = E0·e^{iφ}·ê·exp(S),   S = i k z - ln(1 + iz/zR) - (x²/wx0² + y²/wy0²)/(1 + iz/zR)

    whose modulus is `Tweezer.E_field_amplitude`; model='plane_wave': S = i k z.
    s: shape (3,) [m]
    [V/m, V/m², V/m³]
    """
    x, y, z = (float(value) for value in s)
    k = tweezer.k
    amplitude = tweezer.E0 * np.exp(1j * tweezer.φ) * np.asarray(tweezer.polarisation, dtype=float)
    dS = np.zeros(3, dtype=complex)
    d2S = np.zeros((3, 3), dtype=complex)
    if model == 'plane_wave':
        S = 1j * k * z
        dS[2] = 1j * k
    elif model == 'gaussian':
        zR, a2, b2 = tweezer.zR, tweezer.wx0**2, tweezer.wy0**2
        iz = 1j / zR
        q = 1 + iz * z
        Q = x**2 / a2 + y**2 / b2
        S = 1j * k * z - np.log(q) - Q / q
        dS[0] = -2 * x / (a2 * q)
        dS[1] = -2 * y / (b2 * q)
        dS[2] = 1j * k - iz / q + Q * iz / q**2
        d2S[0, 0] = -2 / (a2 * q)
        d2S[1, 1] = -2 / (b2 * q)
        d2S[0, 2] = d2S[2, 0] = 2 * x * iz / (a2 * q**2)
        d2S[1, 2] = d2S[2, 1] = 2 * y * iz / (b2 * q**2)
        d2S[2, 2] = iz**2 / q**2 - 2 * Q * iz**2 / q**3
    else:
        raise ValueError(f"tweezer field model must be 'gaussian' or 'plane_wave', not {model!r}")
    E = amplitude * np.exp(S)
    results = [E]
    if order >= 1:
        results.append(dS[:, None] * E[None, :])
    if order >= 2:
        results.append((dS[:, None] * dS[None, :] + d2S)[:, :, None] * E[None, None, :])
    return results


# --------------------------------------------------------------------------------------------------
# THE INTERACTION
# --------------------------------------------------------------------------------------------------
class OpticalBinding(Interaction):
    """
    Optical binding between particles in phase-coherent tweezers (see the module docstring)

    Keyword arguments (defaults in brackets):

    - traps [None]: the Traps of the particles, in order (their tweezers and foci `position`); None
      to let the host attach them (`attach_traps`)
    - order ['first']: 'first' (second order in the polarisabilities, the published optical binding
      force) or 'self_consistent' (the coupled-dipole equations solved exactly)
    - tweezer_field ['gaussian']: 'gaussian' (levisim's paraxial beam, with the Gouy phase) or
      'plane_wave'
    - pairs [None]: the pairs (j, k) that interact (0-based); None for every pair
    - overlap_tolerance [1e-3]: the ratio of another tweezer's (neglected) Gaussian field to a
      particle's own above which a WARNING is logged once
    - wavelength_rtol [1e-9]: tweezers whose wavelengths differ by more than this (relative) do not
      interfere, and their particles do not bind

    The stiffness blocks are analytic for order='first' and fourth-order (Richardson-extrapolated)
    central differences of `forces` for 'self_consistent'; `stiffness_blocks(..., method=...)`
    chooses either for cross-validation.
    """

    is_conservative = False
    is_reciprocal = False

    ORDERS = ('first', 'self_consistent')
    FIELD_MODELS = ('gaussian', 'plane_wave')

    def __init__(self, traps=None, order='first', tweezer_field='gaussian', pairs=None,
                 overlap_tolerance=1e-3, wavelength_rtol=1e-9):
        super().__init__(pairs=pairs)
        if order not in self.ORDERS:
            raise ValueError(f'OpticalBinding order must be one of {self.ORDERS}, not {order!r}')
        if tweezer_field not in self.FIELD_MODELS:
            raise ValueError(f'OpticalBinding tweezer_field must be one of {self.FIELD_MODELS}, not {tweezer_field!r}')
        self.order = order
        self.tweezer_field = tweezer_field
        self.overlap_tolerance = float(overlap_tolerance)
        self.wavelength_rtol = float(wavelength_rtol)
        self.traps = None
        self._given_traps = traps is not None
        self._warned = set()
        if traps is not None:
            self._set_traps(traps)

    # ----------------------------------------------------------------------------------------------
    # traps
    # ----------------------------------------------------------------------------------------------
    def _set_traps(self, traps):
        traps = list(traps)
        tweezers = []
        for n, trap in enumerate(traps):
            try:
                tweezers.append(trap.tweezer)
            except ValueError as error:
                raise ValueError(f'OpticalBinding: trap {n} has no tweezer to polarise its particle') from error
        self.traps = traps
        self.tweezers = tweezers
        self.foci = np.array([np.asarray(getattr(trap, 'position', np.zeros(3)), dtype=float) for trap in traps])

    def attach_traps(self, traps):
        """
        Use the host's traps (MultiTrap, MultiParticleFreeSpaceSimulation, nonlinear CavitySimulation)
        for the tweezers and foci, unless traps were given to the constructor. Called by the host.
        """
        if self._given_traps:
            if len(list(traps)) != len(self.traps):
                raise ValueError(f'OpticalBinding was given {len(self.traps)} traps, but the host has {len(list(traps))}')
            return
        self._set_traps(traps)

    def _check_ready(self, num_particles):
        if self.traps is None:
            raise ValueError('OpticalBinding needs the traps: give OpticalBinding(traps=[...]) or use it inside a host'
                             ' (MultiTrap, MultiParticleFreeSpaceSimulation, CavitySimulation) that attaches them')
        if len(self.traps) != num_particles:
            raise ValueError(f'OpticalBinding has {len(self.traps)} traps for {num_particles} particles')

    def _warn_once(self, key, message, *args):
        if key in self._warned:
            return
        self._warned.add(key)
        logger.warning('OpticalBinding: ' + message + ' (logged once per instance)', *args)

    @property
    def label(self):
        return f'OpticalBinding({self.order})'

    @property
    def finite_difference_step(self):
        """
        Step of the central differences of `stiffness_blocks` (method='finite_difference'): 10⁻³/k of
        the first tweezer, or 0.1 nm without traps
        [m]
        """
        if self.traps is None:
            return 1e-10
        return 1e-3 / self.tweezers[0].k

    # ----------------------------------------------------------------------------------------------
    # optics
    # ----------------------------------------------------------------------------------------------
    def polarisability(self, n, particle):
        """
        Lab-frame effective polarisability (with the radiation reaction) of particle n at its
        tweezer's wavenumber, as a 3×3 tensor: `Tweezer.effective_polarisability`·𝟙 for an isotropic
        sphere (the same arithmetic as the trap), `Particle.effective_polarisability_tensor` otherwise
        [complex Fm²]
        """
        tweezer = self.tweezers[n]
        flag = getattr(particle, 'has_scalar_polarisability', None)
        scalar = np.ndim(particle.α) == 0 if flag is None else bool(flag)
        if scalar:
            return tweezer.effective_polarisability(particle) * np.eye(3, dtype=complex)
        return np.asarray(particle.effective_polarisability_tensor(tweezer.k), dtype=complex)

    def interacting_pairs(self, num_particles):
        """The pairs (n, m), n < m as given, whose tweezers share a wavelength"""
        pairs = []
        for n, m in self.pair_indices(num_particles):
            λn, λm = self.tweezers[n].λ, self.tweezers[m].λ
            if abs(λn - λm) > self.wavelength_rtol * max(λn, λm):
                self._warn_once(
                    ('wavelength', n, m),
                    'the tweezers of particles {} and {} have different wavelengths ({:.6g} nm, {:.6g} nm): their'
                    ' fields do not interfere on average, so their binding force is set to zero', n, m,
                    λn * 1e9, λm * 1e9,
                )
                continue
            pairs.append((n, m))
        return pairs

    def _local_fields(self, positions, particles, order):
        """Tweezer fields (and derivatives) at the particles, and their polarisabilities"""
        N = len(particles)
        u, du, d2u, α = [], [], [], []
        for n in range(N):
            fields = tweezer_field(self.tweezers[n], positions[n] - self.foci[n], order=order, model=self.tweezer_field)
            u.append(fields[0])
            du.append(fields[1] if order >= 1 else None)
            d2u.append(fields[2] if order >= 2 else None)
            α.append(self.polarisability(n, particles[n]))
        return u, du, d2u, α

    def _check_validity(self, positions, particles, pairs):
        if 'validity' in self._warned:
            return
        self._warned.add('validity')
        for n, particle in enumerate(particles):
            radius = particle_radius(particle)
            if radius is not None and self.tweezers[n].k * radius > 0.6:
                logger.warning('OpticalBinding: particle {} has kR = {:.3g} > 0.6, where the point-dipole model is'
                               ' inaccurate (logged once per instance)', n, self.tweezers[n].k * radius)
        if self.tweezer_field != 'gaussian':
            return
        for n, m in pairs:
            for a, b in ((n, m), (m, n)):
                own = abs(tweezer_field(self.tweezers[a], positions[a] - self.foci[a], order=0)[0]).max()
                other = abs(tweezer_field(self.tweezers[b], positions[a] - self.foci[b], order=0)[0]).max()
                if own > 0 and other / own > self.overlap_tolerance:
                    logger.warning('OpticalBinding: the (neglected) field of tweezer {} at particle {} is {:.3g} of its'
                                   ' own tweezer\'s: the tweezer envelopes overlap (logged once per instance)',
                                   b, a, other / own)

    def induced_dipoles(self, positions, particles):
        """
        The dipoles (N, 3) of the particles at the positions: the tweezer-induced dipoles
        p_n = α_n·E_tw,n(r_n) for order='first', the solution of the coupled-dipole equations
        p = α·(E_tw + 𝔊·p) for 'self_consistent'
        [Cm]
        """
        positions = as_positions(positions, len(particles))
        self._check_ready(len(particles))
        u, _, _, α = self._local_fields(positions, particles, order=0)
        p0 = np.array([α[n] @ u[n] for n in range(len(particles))])
        if self.order == 'first':
            return p0
        return self._self_consistent_dipoles(positions, u, α, self.interacting_pairs(len(particles)))

    def _self_consistent_dipoles(self, positions, u, α, pairs):
        N = len(u)
        M = np.eye(3 * N, dtype=complex)
        for n, m in pairs:
            k = self.tweezers[n].k
            G_nm = dipole_green_tensor(positions[n] - positions[m], k)
            # 𝔊(r_m - r_n) = 𝔊(r_n - r_m) (even in r)
            M[3*n:3*n + 3, 3*m:3*m + 3] -= α[n] @ G_nm
            M[3*m:3*m + 3, 3*n:3*n + 3] -= α[m] @ G_nm
        source = np.concatenate([α[n] @ u[n] for n in range(N)])
        return np.linalg.solve(M, source).reshape(N, 3)

    # ----------------------------------------------------------------------------------------------
    # forces
    # ----------------------------------------------------------------------------------------------
    def forces(self, positions, particles, t=0.0):
        """
        Optical binding force on every particle at the laboratory-frame positions (N, 3); t is not
        used (the force is time-averaged over the optical period)
        [N]
        """
        positions = as_positions(positions, len(particles))
        N = len(particles)
        self._check_ready(N)
        pairs = self.interacting_pairs(N)
        F = np.zeros((N, 3))
        if not pairs:
            return F
        self._check_validity(positions, particles, pairs)
        u, du, _, α = self._local_fields(positions, particles, order=1)
        if self.order == 'first':
            for n, m in pairs:
                G, dG = dipole_green_tensor_derivatives(positions[n] - positions[m], self.tweezers[n].k, order=1)
                for a, b in ((n, m), (m, n)):
                    # dG is odd in r_a - r_b: ∂_a 𝔊(r_a - r_b) for the pair seen from a
                    sign = 1.0 if a == n else -1.0
                    p0_b = α[b] @ u[b]
                    first = np.conj(α[a] @ u[a]) @ (sign * dG) @ p0_b
                    second = np.conj(du[a]) @ (α[a] @ G @ p0_b)
                    F[a] += 0.5 * np.real(first + second)
            return F
        p = self._self_consistent_dipoles(positions, u, α, pairs)
        gradient = [du[n].astype(complex) for n in range(N)]
        for n, m in pairs:
            dG = dipole_green_tensor_derivatives(positions[n] - positions[m], self.tweezers[n].k, order=1)[1]
            gradient[n] = gradient[n] + dG @ p[m]
            gradient[m] = gradient[m] - dG @ p[n]
        for n in range(N):
            p0 = α[n] @ u[n]
            F[n] = 0.5 * np.real(gradient[n] @ np.conj(p[n])) - 0.5 * np.real(du[n] @ np.conj(p0))
        return F

    # ----------------------------------------------------------------------------------------------
    # linearisation
    # ----------------------------------------------------------------------------------------------
    def stiffness_blocks(self, positions0, particles, method=None):
        """
        Stiffness blocks K[n, m] = -∂F_n/∂r_m (3 × 3 each) at the positions0 (N, 3)
        [N/m]

        method: 'analytic' (order='first' only: second derivatives of the Green tensor and of the
        tweezer fields in closed form), 'finite_difference' (central differences of `forces` with
        steps h and h/2, Richardson-extrapolated to fourth order, h = `finite_difference_step`), or
        None for 'analytic' when available.
        """
        positions0 = as_positions(positions0, len(particles))
        self._check_ready(len(particles))
        if method is None:
            method = 'analytic' if self.order == 'first' else 'finite_difference'
        if method == 'analytic':
            if self.order != 'first':
                raise ValueError("OpticalBinding: analytic stiffness blocks are for order='first' only")
            return self._analytic_stiffness(positions0, particles)
        if method != 'finite_difference':
            raise ValueError(f"method must be 'analytic' or 'finite_difference', not {method!r}")
        h = self.finite_difference_step
        K_h = self._central_difference(positions0, particles, h)
        K_h2 = self._central_difference(positions0, particles, h / 2)
        return (4 * K_h2 - K_h) / 3

    def _central_difference(self, positions0, particles, h):
        N = len(particles)
        K = np.zeros((N, N, 3, 3))
        for k in range(N):
            for b in range(3):
                step = np.zeros((N, 3))
                step[k, b] = h
                F_plus = self.forces(positions0 + step, particles)
                F_minus = self.forces(positions0 - step, particles)
                K[:, k, :, b] = -(F_plus - F_minus) / (2*h)
        return K

    def _analytic_stiffness(self, positions, particles):
        N = len(particles)
        K = np.zeros((N, N, 3, 3))
        pairs = self.interacting_pairs(N)
        if not pairs:
            return K
        u, du, d2u, α = self._local_fields(positions, particles, order=2)
        for n, m in pairs:
            G, dG, d2G = dipole_green_tensor_derivatives(positions[n] - positions[m], self.tweezers[n].k, order=2)
            for a, b in ((n, m), (m, n)):
                sign = 1.0 if a == n else -1.0
                dG_ab = sign * dG          # ∂/∂r_a of 𝔊(r_a - r_b), [c, i, j]
                d2G_ab = d2G               # even: ∂²/∂r_a² of 𝔊(r_a - r_b)
                pa_c = np.conj(α[a] @ u[a])                     # (α_a u_a)*
                dpa_c = np.conj(du[a] @ α[a].T)                 # [b, i]: (α_a ∂_b u_a)*
                pb = α[b] @ u[b]
                dpb = du[b] @ α[b].T                            # [b, j]: α_b ∂_b u_b
                Gpb = G @ pb                                    # [i]
                dGpb = dG_ab @ pb                               # [c, i]
                # F_a,c = ½Re[ pa*·∂_c𝔊·pb + (∂_c u_a)*·α_a·𝔊·pb ]
                # derivative with respect to r_a,e
                d_self = (
                    np.einsum('ei,ci->ce', dpa_c, dGpb)                          # ∂_e pa*
                    + np.einsum('i,ceij,j->ce', pa_c, d2G_ab, pb)                # ∂_e ∂_c 𝔊
                    + np.einsum('cei,i->ce', np.conj(d2u[a]), α[a] @ Gpb)        # ∂_e ∂_c u_a*
                    + np.einsum('ci,ei->ce', np.conj(du[a]), (α[a] @ dGpb.T).T)  # ∂_e 𝔊
                )
                # derivative with respect to r_b,e (𝔊 depends on r_a - r_b; pb on r_b)
                d_other = (
                    -np.einsum('i,ceij,j->ce', pa_c, d2G_ab, pb)
                    + np.einsum('i,cij,ej->ce', pa_c, dG_ab, dpb)
                    - np.einsum('ci,ei->ce', np.conj(du[a]), (α[a] @ dGpb.T).T)
                    + np.einsum('ci,ij,ej->ce', np.conj(du[a]), α[a] @ G, dpb)
                )
                K[a, a] -= 0.5 * np.real(d_self)
                K[a, b] -= 0.5 * np.real(d_other)
        return K

    def coupling_decomposition(self, positions0, particles, method=None):
        """
        The linearised coupling and its reciprocal and anti-reciprocal parts, as a dict of
        (3N × 3N) matrices over [x1, y1, z1, x2, ...]:

        - 'K': the full stiffness -∂F/∂r [N/m]
        - 'reciprocal': (K + Kᵀ)/2, the part a potential could give (it is K for a conservative force)
        - 'antireciprocal': (K - Kᵀ)/2, the curl of the force (non-conservative)
        - 'blocks': the stiffness blocks (N, N, 3, 3)

        For two particles coupled along their tweezer axes this is Rieser et al.'s (Science 377,
        987 (2022), Eq. (1)) K_zz = [[k1 + k2, -(k1 + k2)], [-(k1 - k2), k1 - k2]]: the off-diagonal
        reciprocal part is -k1 and the anti-reciprocal part ∓k2; k2 also makes the diagonal
        shifts of the two particles differ by 2·k2.
        """
        blocks = self.stiffness_blocks(positions0, particles, method=method)
        N = blocks.shape[0]
        K = blocks.transpose(0, 2, 1, 3).reshape(3 * N, 3 * N)
        return {
            'K': K,
            'reciprocal': (K + K.T) / 2,
            'antireciprocal': (K - K.T) / 2,
            'blocks': blocks,
        }

    def __repr__(self):
        options = [f"order={self.order!r}"]
        if self.tweezer_field != 'gaussian':
            options.append(f'tweezer_field={self.tweezer_field!r}')
        if self.pairs is not None:
            options.append(f'pairs={self.pairs}')
        return f'OpticalBinding({", ".join(options)})'

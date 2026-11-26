"""
Magnetic force elements for a magnetisable (diamagnetic or weakly paramagnetic) particle in a static
field: `MagneticTrap`, and `EddyCurrentDamping` for a conducting particle

A particle of volume V and small volume susceptibility (χ of magnitude ≪ 1) in a static field B acquires the
moment m_B = V χ_eff B/μ0 and has the energy ::

    U = -(V / 2μ0) B·χ_eff·B

(Simon and Geim, J. Appl. Phys. 87, 6200 (2000), doi:10.1063/1.372654; Hsu et al., Sci. Rep. 6,
30125 (2016), doi:10.1038/srep30125, Eq. (5)). χ_eff is the susceptibility corrected for the
particle's own demagnetising field, χ_eff = χ (𝟙 + N χ)⁻¹ in the body frame, with N = diag(N_x,
N_y, N_z) the demagnetising factors of its shape; for an ellipsoid these equal the depolarisation
factors of electrostatics (`levisim.physics.shapes.depolarisation_factors`; 1/3 for a sphere). For a
diamagnet the correction is of relative size -χ/3 ~ 10⁻⁵ - 10⁻⁴. An anisotropic material (pyrolytic
graphite) has a body-frame tensor χ, turned into the lab frame by the particle's (fixed) orientation.

With a `Gravity` element a diamagnet (χ < 0) levitates where the magnetic force balances its weight,
which needs (for isotropic χ and the field along the vertical ŷ) ::

    B ∂B/∂y = μ0 ρ g / |χ_eff|      (the levitation condition, e.g. 2400 T²/m for silica)

independent of the particle's size, as are the trap frequencies Ω_j² = (-χ/2μ0ρ) ∂²B²/∂r_j² (B² = B·B).
`Trap.magnetic_trap_report` solves for the equilibrium and reports the stiffness, frequencies, the
levitation margin and the trap depth; see the guide page "Magnetic levitation".

The field comes from any `levisim.physics.fields.magnetic_sources.MagneticFieldSource` (its B, gradB
and hessB), so the force and the stiffness tensor are analytic in the field derivatives ::

    F_k = (V/μ0) Σ_ij J_ik χ_ij B_j,
    K_kl = -∂F_k/∂r_l = -(V/μ0) [ (Jᵀ χ J)_kl + Σ_i (χ B)_i H_ikl ]

with J_ij = ∂B_i/∂r_j and H_ijk = ∂²B_i/∂r_j∂r_k.
"""

import numpy as np

from levisim.core.constants import kB, μ0
from levisim.physics.fields.base import ForceElement, as_position
from levisim.physics.fields.magnetic_sources import MagneticFieldSource

__all__ = [
    'EddyCurrentDamping',
    'MagneticTrap',
    'demagnetising_factors_of',
    'effective_susceptibility',
    'find_escape_saddle',
    'static_energy',
]


def demagnetising_factors_of(particle):
    """
    The body-frame demagnetising factors (N_x, N_y, N_z) of a particle's shape: its
    `depolarisation_factors` (ellipsoids, spheroids, cylinders), or (1/3, 1/3, 1/3) for a `Sphere`
    [·]
    """
    factors = getattr(particle, 'depolarisation_factors', None)
    if factors is not None:
        return np.asarray(factors, dtype=float)
    from levisim.physics.particle import Sphere
    if isinstance(particle, Sphere):
        return np.full(3, 1 / 3)
    raise ValueError(
        f'No demagnetising factors known for a {type(particle).__name__}: give MagneticTrap(..., '
        'demagnetising_factors=(N_x, N_y, N_z)) or shape_correction=False'
    )


def effective_susceptibility(χ, N):
    """
    The shape-corrected susceptibility χ_eff = χ (𝟙 + N χ)⁻¹ (body frame) of a uniformly magnetised
    ellipsoid with demagnetising factors N = (N_x, N_y, N_z), so that its moment in an applied field
    B is V χ_eff B/μ0
    [·]
    """
    χ = np.asarray(χ, dtype=float)
    χ = χ * np.eye(3) if χ.ndim == 0 else χ
    return χ @ np.linalg.inv(np.eye(3) + np.diag(np.asarray(N, dtype=float)) @ χ)


class MagneticTrap(ForceElement):
    """
    A static magnetic field acting on a magnetisable particle through U = -(V/2μ0) B·χ_eff·B (see the
    module docstring)

    ::

        MagneticTrap(source, shape_correction=True, demagnetising_factors=None)

    - source: a `MagneticFieldSource` (`LocalFieldExpansion`, `MultipoleField`, `CurrentLoop`,
      `CoilPair`, `CuboidMagnet`, `FieldSum`, `CallableField`), positioned in the trap's coordinates
    - shape_correction [True]: use χ_eff = χ(𝟙 + Nχ)⁻¹ with the particle's demagnetising factors N;
      False uses the bare material χ (as Hsu et al. and Slezak et al. do; the relative difference is about -χ/3)
    - demagnetising_factors: override N = (N_x, N_y, N_z) in the body frame (default: from the
      particle's shape, see `demagnetising_factors_of`)

    The susceptibility is the particle material's `χ_m` (scalar or body-frame tensor), in the lab frame
    through the particle's orientation. The element is conservative; it adds no noise or damping
    (see `EddyCurrentDamping` for a conductor). A particle with a remanent magnetisation (a magnet) is
    not handled here.
    """

    is_conservative = True

    def __init__(self, source, shape_correction=True, demagnetising_factors=None):
        if not isinstance(source, MagneticFieldSource):
            raise TypeError(f'source must be a MagneticFieldSource, not {type(source).__name__}')
        self.source = source
        self.shape_correction = bool(shape_correction)
        self.demagnetising_factors = (
            None if demagnetising_factors is None else np.asarray(demagnetising_factors, dtype=float).reshape(3)
        )

    @property
    def finite_difference_step(self):
        """Step of the base class's central differences (not used: the stiffness is analytic) [m]"""
        return 1e-6 * self.source.length_scale

    # ----------------------------------------------------------------------------------------------
    # the particle's response
    # ----------------------------------------------------------------------------------------------
    def _check_particle(self, particle):
        if particle is None:
            raise ValueError('MagneticTrap needs the particle (its volume and susceptibility): pass particle=...')
        if getattr(particle.material, 'magnetisation', None) is not None:
            raise NotImplementedError(
                'MagneticTrap treats induced magnetisation only; a particle with a remanent magnetisation '
                '(a permanent magnet) is not supported yet'
            )

    def body_susceptibility(self, particle):
        """
        Effective susceptibility tensor of the particle in its body frame (shape-corrected if
        shape_correction)
        [·]
        """
        self._check_particle(particle)
        χ = np.asarray(particle.material.χ_m_tensor, dtype=float)
        if not self.shape_correction:
            return χ
        N = self.demagnetising_factors if self.demagnetising_factors is not None else demagnetising_factors_of(particle)
        return effective_susceptibility(χ, N)

    def susceptibility(self, particle):
        """
        Effective susceptibility tensor of the particle in the lab frame, R χ_eff Rᵀ
        [·]
        """
        χ = np.asarray(particle.to_lab_frame(self.body_susceptibility(particle)), dtype=float)
        return χ * np.eye(3) if χ.ndim == 0 else χ

    def _response(self, particle):
        """
        V/μ0 and the lab-frame χ_eff, cached for the last particle (recomputed if its material,
        orientation, volume or this element's settings change), since a simulation calls this every step
        """
        key = (
            id(particle), id(particle.material), particle.orientation.tobytes(), particle.V,
            self.shape_correction, None if self.demagnetising_factors is None else self.demagnetising_factors.tobytes(),
        )
        cached = self.__dict__.get('_response_cache')
        if cached is not None and cached[0] == key and cached[1] is particle:
            return cached[2]
        value = (particle.V / μ0, self.susceptibility(particle))
        self.__dict__['_response_cache'] = (key, particle, value)
        return value

    # ----------------------------------------------------------------------------------------------
    # ForceElement interface
    # ----------------------------------------------------------------------------------------------
    def magnetic_moment(self, r, particle):
        """
        Induced magnetic moment V χ_eff B/μ0 at r
        [A·m²]
        """
        scale, χ = self._response(particle)
        return scale * χ @ self.source.B(as_position(r))

    def potential(self, r, particle):
        """
        U = -(V/2μ0) B·χ_eff·B
        [J]
        """
        scale, χ = self._response(particle)
        B = self.source.B(as_position(r))
        return -0.5 * scale * float(B @ χ @ B)

    def force(self, r, t, particle):
        """
        F = (V/μ0) Jᵀ χ_eff B (J_ij = ∂B_i/∂r_j)
        [N]
        """
        r = as_position(r)
        scale, χ = self._response(particle)
        B = self.source.B(r)
        J = self.source.gradB(r)
        return scale * (J.T @ (χ @ B))

    def stiffness(self, r0=None, particle=None):
        """
        K_kl = -∂F_k/∂r_l = -(V/μ0) [(Jᵀ χ J)_kl + Σ_i (χ B)_i H_ikl] at r0 (default the origin),
        from the field and its first and second derivatives (exact for the polynomial sources)
        [N/m]
        """
        r0 = np.zeros(3) if r0 is None else as_position(r0)
        scale, χ = self._response(particle)
        B = self.source.B(r0)
        J = self.source.gradB(r0)
        H = self.source.hessB(r0)
        K = -scale * (J.T @ χ @ J + np.einsum('i,ikl->kl', χ @ B, H))
        return 0.5 * (K + K.T)

    def torque(self, r, t, particle, orientation=None):
        """
        Torque m_B × B on the induced moment (zero for an isotropic χ)
        [N·m]
        """
        r = as_position(r)
        B = self.source.B(r)
        return np.cross(self.magnetic_moment(r, particle), B)

    # ----------------------------------------------------------------------------------------------
    # levitation
    # ----------------------------------------------------------------------------------------------
    def susceptibility_along(self, particle, direction):
        """
        χ_eff projected on a unit field direction b̂, b̂·χ_eff·b̂ (the susceptibility that the energy
        -(V/2μ0) χ B² sees for a field along b̂)
        [·]
        """
        b = np.asarray(direction, dtype=float)
        b = b / np.linalg.norm(b)
        return float(b @ self.susceptibility(particle) @ b)

    def required_B_gradB(self, particle, g=None, field_direction=None):
        """
        The field-gradient product B ∂B/∂h along the upward vertical that holds the particle against
        gravity, μ0 ρ g / (-χ) (χ < 0 the effective susceptibility along the field direction; for an
        isotropic particle field_direction is not needed)
        [T²/m]
        """
        from levisim.core.constants import g as g_earth
        g = g_earth if g is None else float(g)
        χ = self.susceptibility(particle)
        if field_direction is None:
            if not np.allclose(χ, χ[0, 0] * np.eye(3), rtol=1e-12, atol=0):
                raise ValueError('an anisotropic particle needs field_direction (the direction of B at the particle)')
            χ_B = χ[0, 0]
        else:
            χ_B = self.susceptibility_along(particle, field_direction)
        return μ0 * particle.ρ * g / abs(χ_B)

    def __repr__(self):
        return f'MagneticTrap({self.source!r}, shape_correction={self.shape_correction})'


class EddyCurrentDamping(ForceElement):
    """
    Eddy-current damping of a conducting sphere moving through a field gradient, with its thermal
    force noise

    A sphere of radius R and conductivity σ moving with velocity v through a field of Jacobian J sees,
    in its own frame, the uniform field change Ḃ = J v. The induced electric field E = -½ Ḃ × r (which
    is tangential at the surface, so no surface charge is needed) dissipates P = σ ∫E·E dV =
    (2π/15) σ R⁵ Ḃ·Ḃ, so the friction tensor is γ = (2π/15) σ R⁵ JᵀJ and the damping rates, in levisim's
    F = -Γ·p form, are ::

        Γ = γ/m = σ R² JᵀJ / (10 ρ)

    the low-frequency (skin depth ≫ R), small-particle (B varies little over the sphere) limit: Eq. (4)
    of Xiong et al., Commun. Phys. 8, 80 (2025), doi:10.1038/s42005-025-02010-7,
    γ_ij = (2π/15m) σR⁵ Σ_n ∂B_n/∂i ∂B_n/∂j (there also checked against a finite-element simulation).
    The Lorentz force ∫ j × (B0 + J r) dV of these currents is -(2π/15) σ R⁵ (Jᵀ Ḃ - Ḃ tr J), the same
    γ since tr J = ∇·B = 0. tests/test_eddy_current_numerical.py checks γ independently: it solves for
    the induced currents in another gauge by minimising the dissipation over polynomial potentials and
    integrates the Lorentz force (agreement to 1e-9). The conductivity must be isotropic; for a
    conductivity tensor the smallest and largest principal values bound γ from below and above.
    The fluctuating force of the same dissipation at the conductor's temperature T,
    〈F_i F_j〉 = 2 kB T γ_ij δ(t - t'), is added in the time-domain simulation (noise_in_simulation).

    `Trap` uses the diagonal of Γ (`damping_rates`); the off-diagonal part is in `damping_tensor`.

    - source: the `MagneticFieldSource` (normally the MagneticTrap's)
    - position: where J is evaluated, normally the equilibrium (default the origin) [m]
    - conductivity: σ [S/m]; default the particle material's `conductivity`
    - temperature: T of the conductor [K] (default 300 K)
    """

    simulation_group = 'damping'
    noise_in_simulation = True

    def __init__(self, source, position=(0.0, 0.0, 0.0), conductivity=None, temperature=300.0):
        if not isinstance(source, MagneticFieldSource):
            raise TypeError(f'source must be a MagneticFieldSource, not {type(source).__name__}')
        self.source = source
        self.position = as_position(position)
        self.conductivity = None if conductivity is None else float(conductivity)
        self.temperature = float(temperature)

    def _σ(self, particle):
        σ = self.conductivity if self.conductivity is not None else getattr(particle.material, 'conductivity', None)
        if σ is not None and np.ndim(σ) == 2:
            σ = np.asarray(σ, dtype=float)
            if not np.all(σ == σ[0, 0] * np.eye(3)):
                raise ValueError(
                    'EddyCurrentDamping holds for an isotropic conductivity (E = -½ Ḃ × r needs no surface '
                    'charge only then); the particle material has an anisotropic conductivity tensor, pass '
                    'conductivity=... [S/m] explicitly (e.g. the in-plane value as an upper bound)'
                )
            σ = float(σ[0, 0])
        if σ is None or not σ > 0:
            raise ValueError(
                'EddyCurrentDamping needs a conductivity: the particle material has none, pass conductivity=... [S/m]'
            )
        return σ

    def friction_tensor(self, particle):
        """
        γ = (2π/15) σ R⁵ JᵀJ, so that F = -γ v
        [kg/s]
        """
        from levisim.physics.particle import Sphere
        if not isinstance(particle, Sphere):
            raise NotImplementedError('EddyCurrentDamping is written for a sphere only')
        J = self.source.gradB(self.position)
        return (2 * np.pi / 15) * self._σ(particle) * particle.r**5 * (J.T @ J)

    def damping_tensor(self, particle):
        """
        Γ = γ/m = σ R² JᵀJ / (10 ρ), F = -Γ·p
        [2π·Hz]
        """
        return self.friction_tensor(particle) / particle.m

    def damping_rates(self, particle=None):
        """
        Diagonal of `damping_tensor`, (Γx, Γy, Γz)
        [2π·Hz]
        """
        return np.diag(self.damping_tensor(particle)).copy()

    def force(self, r, t, particle):
        """
        Zero: the eddy-current force depends on the momentum only, see `damping_rates`
        [N]
        """
        return np.zeros(3)

    def stiffness(self, r0=None, particle=None):
        """Zero [N/m]"""
        return np.zeros((3, 3))

    def noise(self, particle):
        """
        Two-sided force noise of the dissipation, 2 kB T γ (diagonal part, as the damping in `Trap`)
        [N²/Hz]
        """
        return np.diag(2 * kB * self.temperature * np.diag(self.friction_tensor(particle)))

    def __repr__(self):
        return (
            f'EddyCurrentDamping(source={type(self.source).__name__}, position={self.position.tolist()} m, '
            f'conductivity={self.conductivity}, temperature={self.temperature} K)'
        )


####################################################################################################
# The report of a Trap holding a MagneticTrap
# --------------------------------------------------------------------------------------------------
_STATIC_GROUPS = ('field', 'recoil', 'gravity')


def _static_elements(trap):
    return [element for element in trap.fields if element.simulation_group in _STATIC_GROUPS]


def _total_stiffness(trap, r):
    """Sum of the stiffness tensors of the trap's static (field, recoil, gravity) elements at r [N/m]"""
    return sum((element.stiffness(r, trap.bead) for element in _static_elements(trap)), np.zeros((3, 3)))


def _gravity_vector(trap):
    """The summed gravitational acceleration vector of the trap's Gravity elements [m/s²] (zero if none)"""
    from levisim.physics.fields.elements import Gravity
    return sum((element.g * element.direction for element in trap.fields_of_type(Gravity)), np.zeros(3))


def _vertical_line_guess(trap, element, r_start, scan_range, n=801):
    """
    The stable force-balance point along the vertical line through r_start (the root of the vertical
    static force with a restoring slope) nearest to r_start, or r_start if there is none or no gravity
    """
    g_vec = _gravity_vector(trap)
    if not np.any(g_vec):
        return r_start
    up = -g_vec / np.linalg.norm(g_vec)
    s = np.linspace(-scan_range, scan_range, n)
    F_up = np.array([trap.static_force(r_start + si * up) @ up for si in s])
    roots = []
    for i in range(n - 1):
        if np.isfinite(F_up[i]) and np.isfinite(F_up[i + 1]) and F_up[i] > 0 >= F_up[i + 1]:
            # restoring: the upward force falls through zero as the particle rises
            roots.append(s[i] - F_up[i] * (s[i + 1] - s[i]) / (F_up[i + 1] - F_up[i]))
    if not roots:
        return r_start
    best = min(roots, key=abs)
    return r_start + best * up


def find_static_equilibrium(trap, r_guess, scan_range, max_iterations=60, tolerance=None):
    """
    Root of the trap's static force near r_guess by Newton's method on the analytic stiffness,
    r ← r + K⁻¹ F, with steps limited to scan_range/10; first moved to the stable force balance on the
    vertical line through r_guess (if the trap has gravity)
    [m]
    """
    element_scale = scan_range
    tolerance = 1e-12 * element_scale if tolerance is None else tolerance
    r = _vertical_line_guess(trap, None, as_position(r_guess), scan_range)
    for _ in range(max_iterations):
        F = trap.static_force(r)
        K = _total_stiffness(trap, r)
        try:
            step = np.linalg.solve(K, F)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(K, F, rcond=None)[0]
        norm = np.linalg.norm(step)
        if norm > scan_range / 10:
            step *= (scan_range / 10) / norm
        r = r + step
        if norm < tolerance:
            return r
    raise RuntimeError(
        f'find_static_equilibrium did not converge from {np.asarray(r_guess).tolist()} (last step {norm:.3g} m)'
    )


def _ray_depth(trap, r_eq, direction, scan_range, n=400):
    """Work W(s) = -∫₀ˢ F·d̂ ds' along a ray from r_eq; returns (max W, whether the maximum is at the end)"""
    s = np.linspace(0.0, scan_range, n)
    F_along = np.array([trap.static_force(r_eq + si * direction) @ direction for si in s])
    W = -np.concatenate([[0.0], np.cumsum(0.5 * (F_along[1:] + F_along[:-1]) * np.diff(s))])
    i = int(np.argmax(W))
    return float(W[i]), i == n - 1


def static_energy(trap, points):
    """
    Potential energy of the trap's static (field, recoil, gravity) elements at each of `points`, shape
    (N, 3) or (3,), from the elements' `potential` (vectorised for `MagneticTrap`, `Gravity` and
    `HarmonicPotential`, point by point for the others)
    [J]

    Raises NotImplementedError if an element has no potential energy (e.g. it is not conservative).
    """
    from levisim.physics.fields.elements import Gravity, HarmonicPotential
    points = np.asarray(points, dtype=float)
    single = points.ndim == 1
    points = np.atleast_2d(points)
    particle = trap.bead
    U = np.zeros(len(points))
    for element in _static_elements(trap):
        if isinstance(element, MagneticTrap):
            scale, χ = element._response(particle)
            B = np.asarray(element.source.B(points), dtype=float).reshape(len(points), 3)
            U += -0.5 * scale * np.einsum('ni,ij,nj->n', B, χ, B)
        elif isinstance(element, Gravity):
            U += -particle.m * element.g * (points @ element.direction)
        elif isinstance(element, HarmonicPotential):
            d = points - element.centre
            U += 0.5 * np.einsum('ni,ij,nj->n', d, element.K, d)
        else:
            U += np.array([element.potential(point, particle) for point in points])
    return float(U[0]) if single else U


def _component_reaches_boundary(below, centre_index):
    """Whether the connected (6-neighbour) component of `below` holding centre_index touches the box faces"""
    from scipy import ndimage
    labels, _ = ndimage.label(below)
    label = labels[centre_index]
    if label == 0:
        return False
    faces = (labels[0], labels[-1], labels[:, 0], labels[:, -1], labels[:, :, 0], labels[:, :, -1])
    return any(np.any(face == label) for face in faces)


def find_escape_saddle(trap, r_eq, scan_range, grid_points=41, refine=True, max_iterations=50):
    """
    The lowest energy barrier for escape from the equilibrium r_eq out of the cube of half-width
    scan_range around it, and the saddle point that sets it, as a dict

    The static energy U (`static_energy`) is evaluated on a grid of grid_points³ nodes (odd, so that
    r_eq is a node). The barrier on the grid is the minimax energy of all 6-neighbour grid paths from
    r_eq to a face of the cube: the lowest level U* for which the connected region {U ≤ U*} holding
    r_eq reaches a face (a bisection over the sorted node energies, with `scipy.ndimage.label`). The
    node at that level is the grid saddle. With refine=True it is refined by Newton's method on the
    analytic stiffness, r ← r + K⁻¹F (as `find_static_equilibrium`, steps limited to two grid
    spacings), and accepted if it converges within three grid spacings of the node to a point with
    exactly one negative stiffness eigenvalue (an index-1 saddle); the barrier is then U(saddle) -
    U(r_eq), exact to the convergence of Newton's method.

    If the minimax path reaches its highest energy on a face, the energy is still rising there and the
    true barrier lies outside the cube: the result is a lower bound ('is_lower_bound' True, no saddle).
    Any path out of the trap, the rays of `magnetic_trap_report` included, crosses an energy at least
    as high as the minimax, so the saddle barrier is never above the ray estimate (to the resolution
    of the grid and of the ray integration).

    Keys:

    - 'barrier' [J]: U(saddle) - U(r_eq) (or the grid value if not refined, or the lower bound)
    - 'saddle' [m]: the saddle point (None for a lower bound)
    - 'grid_barrier' [J] and 'grid_saddle' [m]: the grid values before refinement
    - 'is_lower_bound' [bool], 'refined' [bool]
    - 'saddle_stiffness_eigenvalues' [N/m]: eigenvalues of the symmetrised stiffness at the saddle
      (None for a lower bound)
    - 'direction' [·]: unit vector from r_eq to the saddle (or to the grid node that bounds it)
    """
    grid_points = int(grid_points)
    if grid_points < 5 or grid_points % 2 == 0:
        raise ValueError(f'grid_points must be an odd number ≥ 5, not {grid_points}')
    r_eq = as_position(r_eq)
    scan_range = float(scan_range)
    axis = np.linspace(-scan_range, scan_range, grid_points)
    spacing = axis[1] - axis[0]
    X, Y, Z = np.meshgrid(axis, axis, axis, indexing='ij')
    nodes = r_eq + np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    U = static_energy(trap, nodes).reshape(X.shape)
    centre = (grid_points // 2,) * 3
    U_eq = static_energy(trap, r_eq)

    # bisection over the sorted energies for the lowest level whose basin around r_eq reaches a face
    levels = np.unique(U[np.isfinite(U)])
    levels = levels[levels >= U[centre]]
    low, high = 0, len(levels) - 1
    if not _component_reaches_boundary(U <= levels[high], centre):
        raise RuntimeError('find_escape_saddle: the region around r_eq never reaches the faces of the cube')
    while low < high:
        middle = (low + high) // 2
        if _component_reaches_boundary(U <= levels[middle], centre):
            high = middle
        else:
            low = middle + 1
    level = levels[low]
    # the grid saddle: a node at that level, adjacent to the basin below it
    from scipy import ndimage
    labels, _ = ndimage.label(U < level)
    basin = labels == labels[centre] if labels[centre] else np.zeros_like(U, dtype=bool)
    basin[centre] = True
    neighbours = ndimage.binary_dilation(basin)
    candidates = np.argwhere(np.isclose(U, level, rtol=0, atol=0) & neighbours)
    if len(candidates) == 0:
        candidates = np.argwhere(U == level)
    node = tuple(candidates[0])
    grid_saddle = nodes[np.ravel_multi_index(node, U.shape)]
    on_face = any(index in (0, grid_points - 1) for index in node)
    out = {
        'grid_barrier': float(level - U_eq),
        'grid_saddle': grid_saddle,
        'is_lower_bound': bool(on_face),
        'refined': False,
        'saddle': None if on_face else grid_saddle,
        'barrier': float(level - U_eq),
        'saddle_stiffness_eigenvalues': None,
    }
    if not on_face:
        K = _total_stiffness(trap, grid_saddle)
        out['saddle_stiffness_eigenvalues'] = np.linalg.eigvalsh(0.5 * (K + K.T))
    if refine and not on_face:
        r = grid_saddle.copy()
        converged = False
        for _ in range(max_iterations):
            F = trap.static_force(r)
            K = _total_stiffness(trap, r)
            try:
                step = np.linalg.solve(K, F)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(K, F, rcond=None)[0]
            norm = np.linalg.norm(step)
            if norm > 2 * spacing:
                step *= 2 * spacing / norm
            r = r + step
            if norm < 1e-12 * scan_range:
                converged = True
                break
        if converged and np.linalg.norm(r - grid_saddle) <= 3 * spacing:
            K = _total_stiffness(trap, r)
            λ = np.linalg.eigvalsh(0.5 * (K + K.T))
            if np.sum(λ < 0) == 1:
                out.update(saddle=r, barrier=float(static_energy(trap, r) - U_eq), refined=True,
                           saddle_stiffness_eigenvalues=λ)
    target = out['saddle'] if out['saddle'] is not None else grid_saddle
    offset = target - r_eq
    out['direction'] = offset / np.linalg.norm(offset) if np.linalg.norm(offset) > 0 else offset
    return out


def magnetic_trap_report(trap, element=None, r_guess=None, scan_range=None, trap_depth_method='saddle',
                         saddle_grid_points=41):
    """
    Characteristics of a trap holding a `MagneticTrap` (normally with `Gravity`, possibly with other
    elements such as a tweezer), as a dict:

    - 'equilibrium' [m]: root of the total static force (`find_static_equilibrium`, from r_guess,
      default the trap's linearisation point, after a scan of the vertical line through it)
    - 'B' [T], 'B_magnitude' [T], 'gradB' [T/m]: the field and its Jacobian there
    - 'χ_eff' [·]: the lab-frame effective susceptibility tensor; 'χ_along_B' its projection on B̂
    - 'magnetic_force' [N] and 'weight' [N] (m g, 0 without gravity) at the equilibrium
    - 'required_B_gradB' [T²/m]: μ0 ρ g divided by the magnitude of χ_along_B, the levitation condition (None without gravity)
    - 'B_gradB' [T²/m]: the actual ½ ∂(B·B)/∂h along the upward vertical at the equilibrium, signed
      positive when it lifts the particle (equal to 'required_B_gradB' when the field alone holds the
      weight)
    - 'levitation_margin' [·]: the largest upward magnetic force on the vertical line through the
      equilibrium (± scan_range) divided by the weight; ≥ 1 means the field can hold the particle
    - 'stiffness' [N/m] (all static elements) and 'magnetic_stiffness' [N/m] (this element) at the
      equilibrium; 'Ω_axes' [2π·Hz] = √(K_jj/m) (nan for a non-restoring axis);
      'Ω_normal_modes' [2π·Hz] and 'normal_modes' (columns) from the eigen-decomposition of K/m;
      'stable' (all eigenvalues of K positive); 'coupling_stiffness' [N/m]: the off-diagonal part of K,
      which `Trap(linearisation_point=...)` turns into `intra_trap_couplings`
    - 'trap_depth' [J]: the energy barrier for escape from the equilibrium, by `trap_depth_method`:
      'saddle' (default) the lowest barrier over all paths out of the cube of half-width scan_range,
      from the saddle-point search `find_escape_saddle` (saddle_grid_points³ grid, refined by Newton's
      method); 'rays' the ray estimate below (the value of levisim before the saddle search).
      'trap_depth_direction' (unit vector towards the saddle, or along the lowest ray) and
      'trap_depth_is_lower_bound' (True if the barrier lies beyond scan_range) belong to the same
      method; 'trap_depth_method' records it
    - 'trap_depth_rays' [J]: the smallest energy barrier along rays from the equilibrium in the ±
      normal-mode and ± vertical directions, an upper bound on the saddle-point barrier (which can lie
      off these rays); 'trap_depth_rays_direction' and 'trap_depth_rays_is_lower_bound' (True if the
      barrier along that ray lies beyond scan_range)
    - 'saddle_point' [m]: the saddle point that sets the barrier, None for a lower bound or with
      trap_depth_method='rays'; 'saddle_refined' (whether Newton's method converged to an index-1
      saddle) and 'saddle_search' (the full `find_escape_saddle` dict, or None)
    - 'eddy_damping_rates' [2π·Hz]: diagonal Γ of eddy-current damping at the equilibrium if the
      particle is a sphere of a material with an isotropic conductivity (else None);
      'eddy_damping_rate_bounds' [2π·Hz]: for an anisotropic conductivity tensor (e.g. pyrolytic
      graphite), the rates with its smallest and largest principal conductivity, which bound the true
      rates from below and above (else None)

    scan_range defaults to the field source's length_scale [m]. The saddle search needs the potential
    energy of every static element (`static_energy`); if one has none, the report falls back to the
    rays and says so in 'trap_depth_method'.
    """
    from levisim.physics.fields.elements import Gravity
    if trap_depth_method not in ('saddle', 'rays'):
        raise ValueError(f"trap_depth_method must be 'saddle' or 'rays', not {trap_depth_method!r}")
    if element is None:
        elements = trap.fields_of_type(MagneticTrap)
        if not elements:
            raise ValueError('This Trap has no MagneticTrap in its fields')
        element = elements[0]
    particle = trap.bead
    scan_range = element.source.length_scale if scan_range is None else float(scan_range)
    r_guess = trap.linearisation_point if r_guess is None else as_position(r_guess)
    r_eq = find_static_equilibrium(trap, r_guess, scan_range)

    B = element.source.B(r_eq)
    J = element.source.gradB(r_eq)
    B_norm = float(np.linalg.norm(B))
    χ = element.susceptibility(particle)
    χ_B = element.susceptibility_along(particle, B) if B_norm > 0 else float(np.trace(χ) / 3)
    m = particle.m

    g_vec = _gravity_vector(trap)
    has_gravity = bool(trap.fields_of_type(Gravity)) and bool(np.any(g_vec))
    weight = m * float(np.linalg.norm(g_vec))
    out = {
        'equilibrium': r_eq,
        'B': B,
        'B_magnitude': B_norm,
        'gradB': J,
        'χ_eff': χ,
        'χ_along_B': χ_B,
        'magnetic_force': element.force(r_eq, 0.0, particle),
        'weight': weight,
    }
    if has_gravity:
        up = -g_vec / np.linalg.norm(g_vec)
        out['required_B_gradB'] = μ0 * particle.ρ * float(np.linalg.norm(g_vec)) / abs(χ_B)
        # ½ ∂|B|²/∂h along the upward vertical, signed so that it is positive when the field lifts the
        # particle (∂|B|²/∂h < 0 for a diamagnet, > 0 for a paramagnet)
        out['B_gradB'] = float(np.sign(χ_B) * (up @ (J.T @ B)))
        s = np.linspace(-scan_range, scan_range, 801)
        F_up = np.array([element.force(r_eq + si * up, 0.0, particle) @ up for si in s])
        out['levitation_margin'] = float(np.nanmax(F_up) / weight)
    else:
        out['required_B_gradB'] = out['B_gradB'] = out['levitation_margin'] = None

    K = _total_stiffness(trap, r_eq)
    K = 0.5 * (K + K.T)
    out['stiffness'] = K
    out['magnetic_stiffness'] = element.stiffness(r_eq, particle)
    with np.errstate(invalid='ignore'):
        out['Ω_axes'] = np.sqrt(np.diag(K) / m)
    λ, modes = np.linalg.eigh(K / m)
    with np.errstate(invalid='ignore'):
        out['Ω_normal_modes'] = np.sqrt(λ)
    out['normal_modes'] = modes
    out['stable'] = bool(np.all(λ > 0))
    coupling = K.copy()
    np.fill_diagonal(coupling, 0.0)
    out['coupling_stiffness'] = coupling

    directions = [modes[:, j] * sign for j in range(3) for sign in (1, -1)]
    if has_gravity:
        directions += [up, -up]
    depths = [(*_ray_depth(trap, r_eq, d, scan_range), d) for d in directions]
    depth, beyond, direction = min(depths, key=lambda item: item[0])
    out['trap_depth_rays'] = depth
    out['trap_depth_rays_direction'] = direction
    out['trap_depth_rays_is_lower_bound'] = beyond
    out['trap_depth'] = depth
    out['trap_depth_direction'] = direction
    out['trap_depth_is_lower_bound'] = beyond
    out['trap_depth_method'] = 'rays'
    out['saddle_point'] = None
    out['saddle_refined'] = False
    out['saddle_search'] = None
    if trap_depth_method == 'saddle':
        try:
            saddle = find_escape_saddle(trap, r_eq, scan_range, grid_points=saddle_grid_points)
        except NotImplementedError as error:
            out['trap_depth_method'] = f'rays (no saddle search: {error})'
        else:
            out['saddle_search'] = saddle
            out['trap_depth'] = saddle['barrier']
            out['trap_depth_direction'] = saddle['direction']
            out['trap_depth_is_lower_bound'] = saddle['is_lower_bound']
            out['trap_depth_method'] = 'saddle'
            out['saddle_point'] = saddle['saddle']
            out['saddle_refined'] = saddle['refined']

    out['eddy_damping_rates'] = None
    out['eddy_damping_rate_bounds'] = None
    material = particle.material
    if getattr(material, 'is_conductor', False):
        try:
            if getattr(material, 'is_electrically_anisotropic', False):
                # the dissipation min_φ ∫ e·σ e is monotonic in σ, so the smallest and largest principal
                # conductivities bound the damping of the anisotropic sphere from below and above
                σ = np.linalg.eigvalsh(material.conductivity_tensor)
                out['eddy_damping_rate_bounds'] = tuple(
                    EddyCurrentDamping(element.source, r_eq, conductivity=float(value)).damping_rates(particle)
                    for value in (σ[0], σ[-1])
                )
            else:
                out['eddy_damping_rates'] = EddyCurrentDamping(element.source, r_eq).damping_rates(particle)
        except NotImplementedError:
            pass  # EddyCurrentDamping is written for a sphere only
    return out


def magnetic_trap_report_str(report):
    """A `magnetic_trap_report` dict as a printable summary"""
    from levisim.core.units import Hz, μm
    r = report
    lines = [
        f'  equilibrium = {np.array2string(r["equilibrium"] / μm, precision=4)} μm',
        f'  |B| = {r["B_magnitude"]:.4g} T, χ_eff along B = {r["χ_along_B"]:.4g}',
    ]
    if r['required_B_gradB'] is not None:
        lines.append(
            f'  levitation: B∂B/∂h = {r["B_gradB"]:.5g} T²/m (needed {r["required_B_gradB"]:.5g} T²/m), '
            f'margin = {r["levitation_margin"]:.4g}'
        )
    lines.append(f'  Ω_axes/2π = {np.array2string(r["Ω_axes"] / Hz, precision=5)} Hz')
    lines.append(f'  normal modes Ω/2π = {np.array2string(r["Ω_normal_modes"] / Hz, precision=5)} Hz, stable = {r["stable"]}')
    method = r.get('trap_depth_method', 'rays')
    if method == 'saddle':
        lines.append(
            f'  trap depth = {r["trap_depth"]:.4g} J = {r["trap_depth"] / kB:.4g} K·kB'
            + (' (lower bound: no saddle within the scan range)' if r['trap_depth_is_lower_bound'] else
               f' (saddle at {np.array2string(r["saddle_point"] / μm, precision=4)} μm)')
        )
        lines.append(
            f'  ray estimate ≈ {r["trap_depth_rays"]:.4g} J = {r["trap_depth_rays"] / kB:.4g} K·kB'
            + (' (lower bound)' if r['trap_depth_rays_is_lower_bound'] else '')
        )
    else:
        lines.append(
            f'  trap depth ≈ {r["trap_depth"]:.4g} J = {r["trap_depth"] / kB:.4g} K·kB'
            + (' (lower bound)' if r['trap_depth_is_lower_bound'] else '')
        )
    if r['eddy_damping_rates'] is not None:
        lines.append(f'  eddy-current damping Γ/2π = {np.array2string(r["eddy_damping_rates"] / Hz, precision=4)} Hz')
    if r.get('eddy_damping_rate_bounds') is not None:
        low, high = r['eddy_damping_rate_bounds']
        lines.append(
            f'  eddy-current damping Γ/2π between {np.array2string(low / Hz, precision=4)} and '
            f'{np.array2string(high / Hz, precision=4)} Hz (anisotropic σ)'
        )
    return '\n'.join(lines)

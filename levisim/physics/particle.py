"""
Levitated particles: the Particle base class and the Sphere

A `Particle` combines a `Material` with a shape. It supplies what the dynamics need from the
particle: mass m, volume V, body-frame inertia tensor I, charge q, polarisability tensor α (with
an optional radiation-reaction correction) and the gas-damping tensors. `Sphere` is the only shape
implemented here; `Bead` (levisim.physics.bead) is a `Sphere` that keeps the original keyword
arguments and numbers. The non-spherical shapes (Spheroid, Ellipsoid, Cylinder, Rod, Disk, Dumbbell)
are in levisim.physics.shapes and `TabulatedParticle` in levisim.physics.tabulated. A particle has a
fixed orientation (a unit quaternion) that turns body-frame tensors into lab-frame ones; rotational
dynamics come later.
"""

from functools import cached_property

import numpy as np

from levisim.core.constants import e, g, kB, ε0, π
from levisim.core.units import aN, fg, nm
from levisim.physics.material import Material, _as_tensor

# trapping position phase along the cavity axis: antinode, halfnode, node = 0, π/4, π/2
_NODE = π/2

# levisim's tweezer polarisation direction (lab frame): every Tweezer propagates along z and is
# polarised along y (see levisim.physics.coherent_scattering)
TWEEZER_POLARISATION = np.array([0.0, 1.0, 0.0])


def _radiation_reaction(α, k):
    """α (𝟙 - i k³/(6πε0) α)⁻¹ for a 3×3 polarisability α [Fm²] at wavenumber k [1/m]"""
    return α @ np.linalg.inv(np.eye(3) - 1j * (k**3 / (6*π*ε0)) * α)


def as_unit_quaternion(orientation):
    """
    The orientation as a float array (w, x, y, z), checked to be a unit quaternion (to 1e-8)
    [·]
    """
    q = np.asarray(orientation, dtype=float)
    if q.shape != (4,) or not np.all(np.isfinite(q)) or not np.isclose(np.linalg.norm(q), 1.0):
        raise ValueError(f'orientation must be a unit quaternion (w, x, y, z), not {orientation}')
    return q


def quaternion_to_rotation_matrix(orientation):
    """
    Rotation matrix R (body frame → lab frame) of the unit quaternion q = (w, x, y, z)

    ::

        R = [[1 - 2(y² + z²), 2(xy - wz),     2(xz + wy)    ],
             [2(xy + wz),     1 - 2(x² + z²), 2(yz - wx)    ],
             [2(xz - wy),     2(yz + wx),     1 - 2(x² + y²)]]

    A body-frame vector v has lab-frame components R v; a body-frame tensor T is R T Rᵀ in the lab.
    q = (cos(θ/2), sin(θ/2) n̂) rotates by θ about the unit axis n̂. The quaternion is normalised
    before use, so R is orthogonal to round-off.
    [·]
    """
    w, x, y, z = np.asarray(orientation, dtype=float) / np.linalg.norm(orientation)
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ])


def rotation_matrix_to_quaternion(R):
    """
    Unit quaternion (w, x, y, z) with w ≥ 0 of a proper rotation matrix R (body frame → lab frame),
    the inverse of `quaternion_to_rotation_matrix` (Shepperd's method)
    [·]
    """
    R = np.asarray(R, dtype=float)
    if R.shape != (3, 3) or not np.allclose(R @ R.T, np.eye(3), atol=1e-10) or np.linalg.det(R) <= 0:
        raise ValueError('R must be a proper rotation matrix (orthogonal, det +1)')
    trace = np.trace(R)
    i = int(np.argmax([trace, R[0, 0], R[1, 1], R[2, 2]]))
    if i == 0:
        w = 0.5 * np.sqrt(1 + trace)
        x, y, z = (R[2, 1] - R[1, 2]) / (4*w), (R[0, 2] - R[2, 0]) / (4*w), (R[1, 0] - R[0, 1]) / (4*w)
    elif i == 1:
        x = 0.5 * np.sqrt(1 + 2*R[0, 0] - trace)
        w, y, z = (R[2, 1] - R[1, 2]) / (4*x), (R[0, 1] + R[1, 0]) / (4*x), (R[0, 2] + R[2, 0]) / (4*x)
    elif i == 2:
        y = 0.5 * np.sqrt(1 + 2*R[1, 1] - trace)
        w, x, z = (R[0, 2] - R[2, 0]) / (4*y), (R[0, 1] + R[1, 0]) / (4*y), (R[1, 2] + R[2, 1]) / (4*y)
    else:
        z = 0.5 * np.sqrt(1 + 2*R[2, 2] - trace)
        w, x, y = (R[1, 0] - R[0, 1]) / (4*z), (R[0, 2] + R[2, 0]) / (4*z), (R[1, 2] + R[2, 1]) / (4*z)
    q = np.array([w, x, y, z])
    q = q / np.linalg.norm(q)
    return -q if q[0] < 0 else q


def axis_angle_quaternion(axis, angle):
    """
    Unit quaternion (w, x, y, z) of a rotation by `angle` [rad] about `axis` (a 3-vector, or one of
    'x', 'y', 'z'); e.g. axis_angle_quaternion('x', -π/2) turns the body z axis onto the lab y axis
    [·]
    """
    if isinstance(axis, str):
        axis = {'x': (1, 0, 0), 'y': (0, 1, 0), 'z': (0, 0, 1)}[axis]
    n = np.asarray(axis, dtype=float)
    n = n / np.linalg.norm(n)
    return np.concatenate([[np.cos(angle / 2)], np.sin(angle / 2) * n])


def body_axis_to_quaternion(direction):
    """
    A unit quaternion whose rotation takes the body z axis onto `direction` (a lab-frame 3-vector)
    by the shortest rotation; e.g. body_axis_to_quaternion((0, 1, 0)) aligns a spheroid's symmetry
    axis with levisim's tweezer polarisation y
    [·]
    """
    d = np.asarray(direction, dtype=float)
    d = d / np.linalg.norm(d)
    c = float(d[2])
    if c > 1 - 1e-15:
        return np.array([1.0, 0.0, 0.0, 0.0])
    if c < -1 + 1e-15:
        return np.array([0.0, 1.0, 0.0, 0.0])
    axis = np.cross([0.0, 0.0, 1.0], d)
    return axis_angle_quaternion(axis, np.arccos(np.clip(c, -1, 1)))


class Particle:
    """
    A levitated rigid particle made of one `Material` (base class)

    ::

        Particle(material, charge=0, λ=1064 nm, Φ=π/2, orientation=(1, 0, 0, 0))

    - material: a `Material` (see levisim.physics.materials for a library)
    - charge: net charge in units of the elementary charge e (an integer for a real particle) [·]
    - λ: vacuum wavelength at which a dispersive permittivity is evaluated for `εr`, `n` and `α` [m]
    - Φ: trapping position phase along the cavity axis, Φ = k·x0 [·] (π/2: node, 0: antinode)
    - orientation: unit quaternion (w, x, y, z) of the body frame relative to the lab frame (see
      `quaternion_to_rotation_matrix`). It turns the body-frame tensors into lab-frame ones
      (`lab_polarisability_tensor`, `lab_inertia_tensor`, ...). It is fixed (there are no
      rotational dynamics yet) but may be reassigned.

    Subclasses define the shape through `V`, `A`, `inertia_tensor` and `static_polarisability_tensor`,
    and may define the gas-damping tensors. Body-frame quantities are cached on first access, as in
    the original `Bead`; lab-frame quantities are computed from the current orientation.

    A non-spherical particle is trapped translationally through its polarisability projected on the
    tweezer polarisation (levisim's lab y axis) in the current orientation: `α` is that projection of
    the static tensor, and `effective_polarisability_along` applies the radiation reaction to the
    whole lab-frame tensor before projecting.
    """
    def __init__(
        self,
        material,
        charge=0,
        λ=1064*nm,
        Φ=_NODE,
        orientation=(1.0, 0.0, 0.0, 0.0),
    ):
        """
        Initialises a Particle instance
        """
        if not isinstance(material, Material):
            raise TypeError(f'material must be a Material, not {type(material).__name__}')
        self.material = material

        # Net charge
        # [e]
        self.charge = charge

        # Vacuum wavelength at which the permittivity is evaluated
        # [m]
        self.λ = float(λ)

        # Trapping position phase away from the antinode of the cavity standing wave
        # [·]
        self.Φ = float(Φ)

        # Orientation of the body frame (unit quaternion w, x, y, z)
        # [·]
        self.orientation = orientation

    @property
    def orientation(self):
        """
        Unit quaternion (w, x, y, z) of the body frame relative to the lab frame
        [·]
        """
        return self._orientation

    @orientation.setter
    def orientation(self, value):
        self._orientation = as_unit_quaternion(value)

    @property
    def rotation_matrix(self):
        """
        Rotation matrix R (body frame → lab frame) of the current orientation
        [·]
        """
        return quaternion_to_rotation_matrix(self._orientation)

    def to_lab_frame(self, tensor):
        """
        A body-frame 3×3 tensor T in the lab frame, R T Rᵀ (a scalar is returned unchanged)
        [units of T]
        """
        if np.ndim(tensor) == 0:
            return tensor
        R = self.rotation_matrix
        return R @ np.asarray(tensor) @ R.T

    ################################################################################################
    # MATERIAL PROPERTIES
    # ----------------------------------------------------------------------------------------------
    @property
    def ρ(self):
        """
        Mass density of the particle's material
        [kg/m³]
        """
        return self.material.ρ

    @ρ.setter
    def ρ(self, value):
        self.material = self.material.replace(name=f'{self.material.name} (ρ={float(value)!r})', ρ=float(value))

    @property
    def εr(self):
        """
        Relative permittivity of the particle's material at the particle's wavelength λ (a 3×3
        body-frame tensor for an anisotropic material)
        [·]
        """
        return self.material.εr(self.λ)

    @εr.setter
    def εr(self, value):
        self.material = self.material.replace(name=f'{self.material.name} (εr={value!r})', permittivity=value)

    @property
    def q(self):
        """
        Net charge
        [C]
        """
        return self.charge * e

    ################################################################################################
    # SHAPE (defined by subclasses)
    # ----------------------------------------------------------------------------------------------
    @cached_property
    def V(self):
        """
        Volume of the particle
        [m³]
        """
        raise NotImplementedError(f'{type(self).__name__} does not define its volume V')

    @cached_property
    def A(self):
        """
        Surface area of the particle
        [m²]
        """
        raise NotImplementedError(f'{type(self).__name__} does not define its surface area A')

    @cached_property
    def inertia_tensor(self):
        """
        Moment of inertia tensor about the centre of mass, in the body frame
        [kg·m²]
        """
        raise NotImplementedError(f'{type(self).__name__} does not define its inertia tensor')

    def static_polarisability_tensor(self, λ=None):
        """
        Polarisability tensor (body frame) without radiation reaction, at vacuum wavelength λ
        (default: the particle's λ)
        [Fm²]
        """
        raise NotImplementedError(f'{type(self).__name__} does not define its polarisability')

    ################################################################################################
    # DERIVED PROPERTIES
    # ----------------------------------------------------------------------------------------------
    @cached_property
    def m(self):
        """
        Mass of the particle
        [kg]
        """
        return self.ρ * self.V

    @cached_property
    def weight(self):
        """
        Particle weight under Earth's gravity
        [N]
        """
        return self.m * g

    @cached_property
    def n(self):
        """
        Refractive index of the (non-magnetic, μr=1) particle at its wavelength λ
        [·]
        """
        return np.sqrt(self.εr)

    def polarisability_tensor(self, λ=None, radiation_reaction=False):
        """
        Polarisability tensor (body frame) at vacuum wavelength λ (default: the particle's λ)

        With radiation_reaction=True, the dipole's self-field is included (the optical theorem)::

            α_eff = α (1 - i k³/(6πε0) α)⁻¹,   k = 2π/λ

        which for a scalar α is `Trap.α_eff`.
        [complex Fm²]
        """
        λ = self.λ if λ is None else float(λ)
        α = _as_tensor(self.static_polarisability_tensor(λ))
        if not radiation_reaction:
            return α
        k = 2*π / λ
        return _radiation_reaction(α, k)

    def lab_polarisability_tensor(self, λ=None, radiation_reaction=False):
        """
        Polarisability tensor in the lab frame, R α Rᵀ, at vacuum wavelength λ (default: the
        particle's λ), optionally with the radiation reaction (see `polarisability_tensor`;
        R α_eff Rᵀ = (R α Rᵀ)(𝟙 - i k³/(6πε0) R α Rᵀ)⁻¹, so the order does not matter)
        [complex Fm²]
        """
        return self.to_lab_frame(_as_tensor(self.polarisability_tensor(λ, radiation_reaction=radiation_reaction)))

    @property
    def has_scalar_polarisability(self):
        """
        Whether `α` is one scalar for every field direction (an isotropic sphere). `Tweezer` then uses
        levisim's original scalar arithmetic; otherwise the tensor projections
        [bool]
        """
        return False

    @property
    def α(self):
        """
        Static polarisability seen by light polarised along levisim's tweezer polarisation (lab y)
        in the current orientation, ŷ·(R α_body Rᵀ)·ŷ, at the particle's wavelength λ. It is the
        scalar that `Trap` and `MultiTrap` use for the coherent-scattering drive `Ed` and the cavity
        terms; the tweezer itself uses `effective_polarisability_along`.
        [Fm²]
        """
        value = TWEEZER_POLARISATION @ self.lab_polarisability_tensor() @ TWEEZER_POLARISATION
        return value.real if np.imag(value) == 0 else complex(value)

    def effective_polarisability_tensor(self, k):
        """
        Lab-frame polarisability with the radiation reaction at the wavenumber k [1/m] of the light,
        from the static tensor at the particle's wavelength λ (as `Tweezer.effective_polarisability`
        does for a sphere): α_eff = α_lab (𝟙 - i k³/(6πε0) α_lab)⁻¹
        [complex Fm²]
        """
        return _radiation_reaction(self.lab_polarisability_tensor(), k)

    def effective_polarisability_along(self, direction, k):
        """
        Effective polarisability for light polarised along the real unit lab-frame vector
        `direction` at wavenumber k [1/m]: ê·α_eff·ê, with the radiation reaction applied to the whole
        tensor first. For a fixed orientation the time-averaged dipole energy in the field E ê is
        -(1/4) Re(ê·α_eff·ê)|E|² (the induced dipole's components perpendicular to ê do no work),
        so this sets the translational trap frequencies.
        [complex Fm²]
        """
        e = np.asarray(direction, dtype=float)
        return complex(e @ self.effective_polarisability_tensor(k) @ e)

    ################################################################################################
    # GAS DAMPING
    # ----------------------------------------------------------------------------------------------
    def translational_damping_tensor(self, gas, accommodation=1.0, surface_temperature_ratio=1.0):
        """
        Centre-of-mass gas damping-rate tensor Γ_cm, in the body frame
        [2π·Hz]
        """
        raise NotImplementedError(
            f'{type(self).__name__} has no translational gas-damping tensor yet (only Sphere does)'
        )

    def rotational_damping_tensor(self, gas, accommodation=1.0, surface_temperature_ratio=1.0):
        """
        Rotational gas damping-rate tensor Γ_rot, in the body frame
        [2π·Hz]
        """
        raise NotImplementedError(
            f'{type(self).__name__} has no rotational gas-damping tensor yet (only Sphere does)'
        )

    def lab_translational_damping_tensor(self, gas, accommodation=1.0, surface_temperature_ratio=1.0):
        """
        Centre-of-mass gas damping-rate tensor in the lab frame, R Γ_cm Rᵀ
        [2π·Hz]
        """
        return self.to_lab_frame(self.translational_damping_tensor(gas, accommodation, surface_temperature_ratio))

    def lab_rotational_damping_tensor(self, gas, accommodation=1.0, surface_temperature_ratio=1.0):
        """
        Rotational gas damping-rate tensor in the lab frame, R Γ_rot Rᵀ
        [2π·Hz]
        """
        return self.to_lab_frame(self.rotational_damping_tensor(gas, accommodation, surface_temperature_ratio))

    def translational_diffusion_tensor(self, gas, accommodation=1.0, surface_temperature_ratio=1.0):
        """
        Centre-of-mass momentum diffusion tensor D_cm in the body frame, d〈P_i P_j〉/dt = 2 D_ij
        (Martinetz et al., Phys. Rev. E 97, 052112 (2018)). With a surface at the gas temperature
        (γs = 1) it is D_cm = kB T m_n Γ_cm (their Eq. (47a)), the default here.
        [N²·s]
        """
        if surface_temperature_ratio != 1.0:
            raise NotImplementedError(
                f'{type(self).__name__} gives the diffusion tensor for γs = 1 only (D = kB T m_n Γ_cm)'
            )
        Γ = self.translational_damping_tensor(gas, accommodation, surface_temperature_ratio)
        return kB * gas.T * self.m * Γ

    def rotational_diffusion_tensor(self, gas, accommodation=1.0, surface_temperature_ratio=1.0):
        """
        Angular-momentum diffusion tensor D_rot in the body frame, d〈J_i J_j〉/dt = 2 D_ij; with γs = 1
        it is D_rot = kB T Γ_rot I (Martinetz et al. 2018, Eq. (47b)), the default here
        [N²·m²·s]
        """
        if surface_temperature_ratio != 1.0:
            raise NotImplementedError(
                f'{type(self).__name__} gives the diffusion tensor for γs = 1 only (D = kB T Γ_rot I)'
            )
        Γ = self.rotational_damping_tensor(gas, accommodation, surface_temperature_ratio)
        return kB * gas.T * Γ @ self.inertia_tensor

    @property
    def lab_inertia_tensor(self):
        """
        Moment of inertia tensor about the centre of mass in the lab frame, R I Rᵀ
        [kg·m²]
        """
        return self.to_lab_frame(self.inertia_tensor)

    ################################################################################################
    # INFORMATION PRINTING FUNCTIONS
    # ----------------------------------------------------------------------------------------------
    @property
    def info(self):
        """
        Prints information on the Particle instance
        """
        print(self.info_str)

    @property
    def info_str(self):
        """
        String describing the Particle instance
        """
        return (
            f'{type(self).__name__}(id={id(self)}'
            f'\n    material = {self.material.name}'
            f'\n    ρ = {self.ρ:.1f} kg/m³'
            f'\n    charge = {self.charge} e'
            f'\n    λ = {self.λ / nm:.1f} nm'
            f'\n'
            f'\n    volume = {self.V:.5e} m³'
            f'\n    mass = {self.m / fg:.5f} fg'
            f'\n    weight = {self.weight / aN:.5f} aN'
            f'\n'
            f'\n    trapping position phase, Φ/2π = x0/λtw = {self.Φ /(2*π):.3f}'
            f'\n)'
        )


class Sphere(Particle):
    """
    A homogeneous sphere of radius r

    ::

        Sphere(r=60.1 nm, material=LEVISIM_DEFAULT_SILICA, charge=0, λ=1064 nm, Φ=π/2)

    V = (4/3)πr³, A = 4πr², m = ρV, I = (2/5) m r² 𝟙, and the Clausius-Mossotti polarisability

        α = 3 V ε0 (εr - 1)/(εr + 2)

    (the matrix form 3Vε0 (ε - 𝟙)(ε + 2𝟙)⁻¹ for an anisotropic material). With the default
    arguments every number equals that of `Bead()`.
    """
    def __init__(
        self,
        r=60.1*nm,
        material=None,
        **kwargs,
    ):
        """
        Initialises a Sphere instance
        """
        if material is None:
            from levisim.physics.materials import LEVISIM_DEFAULT_SILICA
            material = LEVISIM_DEFAULT_SILICA
        super().__init__(material, **kwargs)

        # Sphere radius
        # [m]
        self.r = float(r)
        if not np.isfinite(self.r) or self.r <= 0:
            raise ValueError(f'r must be a positive, finite radius in m, not {r}')

    @property
    def info_str(self):
        """
        String describing the Sphere instance
        """
        α = self.α if np.ndim(self.εr) == 0 else np.diag(self.α)
        return (
            f'Sphere(id={id(self)}'
            f'\n    r = {self.r / nm:.1f} nm'
            f'\n    material = {self.material.name}'
            f'\n    ρ = {self.ρ:.1f} kg/m³'
            f'\n    εr = {self.εr}'
            f'\n    charge = {self.charge} e'
            f'\n    λ = {self.λ / nm:.1f} nm'
            f'\n'
            f'\n    volume = {self.V:.5e} m³'
            f'\n    area = {self.A:.5e} m²'
            f'\n    mass = {self.m / fg:.5f} fg'
            f'\n    weight = {self.weight / aN:.5f} aN'
            f'\n    α = {α} Fm²'
            f'\n'
            f'\n    trapping position phase, Φ/2π = x0/λtw = {self.Φ /(2*π):.3f}'
            f'\n)'
        )

    ################################################################################################
    # DERIVED PROPERTIES
    # ----------------------------------------------------------------------------------------------
    @cached_property
    def V(self):
        """
        Volume of the sphere
        [m³]
        """
        return (4/3)*π * self.r**3

    @cached_property
    def A(self):
        """
        Surface area of the sphere
        [m²]
        """
        return 4*π * self.r**2

    @cached_property
    def inertia_tensor(self):
        """
        Moment of inertia tensor of the homogeneous sphere, I = (2/5) m r² 𝟙 (any frame)
        [kg·m²]
        """
        return (2/5) * self.m * self.r**2 * np.eye(3)

    @cached_property
    def I(self):  # the conventional symbol
        """
        Moment of inertia of the homogeneous sphere about any axis through its centre, (2/5) m r²
        [kg·m²]
        """
        return (2/5) * self.m * self.r**2

    @staticmethod
    def _clausius_mossotti(V, εr):
        """3Vε0(εr - 1)/(εr + 2), or its matrix form for a 3×3 εr"""
        if np.ndim(εr) == 0:
            return 3 * V * ε0 * (εr - 1) / (εr + 2)
        identity = np.eye(3)
        return 3 * V * ε0 * (εr - identity) @ np.linalg.inv(εr + 2*identity)

    @cached_property
    def α(self):
        """
        Static polarisability of the sphere at its wavelength λ (Clausius-Mossotti; a 3×3
        body-frame tensor for an anisotropic material)
        [Capacitance × Area] = [Fm²]

        [F] = [Farad] = Capacitance = Charge / Electric Potential = [C/V]
        NOTE: https://en.wikipedia.org/wiki/Clausius–Mossotti_relation#Lorentz–Lorenz_equation
        """
        return self._clausius_mossotti(self.V, self.εr)

    @property
    def has_scalar_polarisability(self):
        """
        Whether `α` is a scalar (an isotropic material)
        [bool]
        """
        return np.ndim(self.α) == 0

    def effective_polarisability_along(self, direction, k):
        """
        Effective polarisability for light polarised along `direction` at wavenumber k [1/m]. For an
        isotropic material this is the scalar α/(1 - i k³α/(6πε0)) whatever the direction, with
        exactly the arithmetic of the original `Trap.α_eff`; for an anisotropic material (a 3×3 `α`)
        it is ê·α_eff·ê of the rotated tensor (see `Particle.effective_polarisability_along`).
        [complex Fm²]
        """
        if self.has_scalar_polarisability:
            return self.α / (1 - 1j * (k**3 / (6*π*ε0)) * self.α)
        return super().effective_polarisability_along(direction, k)

    def static_polarisability_tensor(self, λ=None):
        """
        Clausius-Mossotti polarisability tensor (body frame) at vacuum wavelength λ (default: the
        sphere's λ, for which it equals α·𝟙)
        [Fm²]
        """
        if λ is None or float(λ) == self.λ:
            return _as_tensor(self.α)
        return _as_tensor(self._clausius_mossotti(self.V, self.material.εr(float(λ))))

    ################################################################################################
    # GAS DAMPING
    # ----------------------------------------------------------------------------------------------
    def translational_damping_tensor(self, gas, accommodation=1.0, surface_temperature_ratio=1.0):
        """
        Centre-of-mass gas damping-rate tensor of the sphere in the free-molecular regime
        [2π·Hz]

        Table I of Martinetz et al., Phys. Rev. E 97, 052112 (2018), doi:10.1103/PhysRevE.97.052112,
        for specular and diffuse reflection with accommodation coefficient αc and surface temperature
        Ts = γs² T::

            Γ_cm = 8 n_g R² √(2π m_g kB T) (1 + αc γs π/8) / (3M) 𝟙
                 = (1 + αc γs π/8) · m_g v̄ P/(kB T) / (ρ R) 𝟙,   v̄ = √(8 kB T/(π m_g))

        With αc = γs = 1 (the defaults) this is `Trap.Γth` exactly. `surface_temperature_ratio` is γs.
        """
        geometric_factor = (1 + π/8 * accommodation * surface_temperature_ratio)
        particle_factor = 1 / (self.ρ * self.r)
        gas_factor = (gas.m * gas.vth * gas.P) / (kB * gas.T)
        return geometric_factor * particle_factor * gas_factor * np.eye(3)

    def rotational_damping_tensor(self, gas, accommodation=1.0, surface_temperature_ratio=1.0):
        """
        Rotational gas damping-rate tensor of the sphere in the free-molecular regime
        [2π·Hz]

        Table I of Martinetz et al., Phys. Rev. E 97, 052112 (2018), doi:10.1103/PhysRevE.97.052112::

            Γ_rot = 10 αc n_g R² √(2π m_g kB T) / (3M) 𝟙 = (5/4) αc · m_g v̄ P/(kB T) / (ρ R) 𝟙

        Only diffusely reflected gas molecules (αc > 0) exert a torque on a sphere; the result does
        not depend on the surface temperature ratio γs (accepted for a uniform signature).
        """
        gas_factor = (gas.m * gas.vth * gas.P) / (kB * gas.T)
        return (5/4) * accommodation * gas_factor / (self.ρ * self.r) * np.eye(3)


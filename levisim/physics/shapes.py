"""
Non-spherical levitated particles: Ellipsoid, Spheroid, Cylinder (Rod, Disk) and Dumbbell

Each shape is a `Particle` made of one homogeneous `Material` and supplies its volume V, surface
area A, mass m_n = ρV, body-frame inertia tensor I, body-frame polarisability tensor in the
quasi-static (Rayleigh) approximation, and the free-molecular gas friction and diffusion tensors.
The body frame has the shape's symmetry axis along z; the particle's `orientation` (a unit
quaternion) turns body-frame tensors into lab-frame ones (`Particle.lab_polarisability_tensor`, ...).

POLARISABILITY. A homogeneous ellipsoid in a uniform field E0 is uniformly polarised, with internal
field E_in = [𝟙 + L(ε - 𝟙)]⁻¹ E0, so its polarisability is ::

    α = ε0 V (ε - 𝟙) [𝟙 + L (ε - 𝟙)]⁻¹,   L = diag(L_x, L_y, L_z),   L_x + L_y + L_z = 1

(Bohren and Huffman, Absorption and Scattering of Light by Small Particles (Wiley, 1983), §5.3;
Landau and Lifshitz, Electrodynamics of Continuous Media, §4), with depolarisation factors ::

    L_i = (a_x a_y a_z / 2) ∫₀^∞ ds / [(s + a_i²) √((s + a_x²)(s + a_y²)(s + a_z²))]
        = (a_x a_y a_z / 3) R_D(a_j², a_k², a_i²)                         (Carlson's integral)

For a scalar εr, α_i = ε0 V (εr - 1)/(1 + L_i(εr - 1)); a 3×3 εr (a birefringent material, its
principal axes given in the particle's body frame) enters through the matrix form. At L_i = 1/3
this is Clausius-Mossotti, `Sphere.α`. The radiation reaction α_eff = α (𝟙 - i k³α/(6πε0))⁻¹ is
applied to the whole tensor (`Particle.polarisability_tensor(radiation_reaction=True)`).

Cylinders (rods, disks) have no closed-form polarisability: levisim uses the depolarisation factors
of the spheroid with the same aspect ratio ℓ/(2R), with the cylinder's own volume (an approximation
that becomes exact for the infinitely long rod, L = (1/2, 1/2, 0), and the infinitely thin disk,
L = (0, 0, 1)). A dumbbell is two coupled point dipoles (see `Dumbbell`).

GAS DAMPING. Martinetz, Hornberger and Stickler, Phys. Rev. E 97, 052112 (2018),
doi:10.1103/PhysRevE.97.052112, for a mixture of specular and diffuse reflection (accommodation
coefficient αc, surface temperature Ts = γs² T; levisim's defaults αc = γs = 1 are those of
`Trap.Γth`). Spheres and cylinders use the closed forms of their Table I, ellipsoids and dumbbells
the surface integrals of their Eqs. (40)-(41) evaluated numerically once (levisim.physics.gas_damping).

SPHERE LIMIT. An Ellipsoid with a = b = c and a Spheroid with a = b reproduce `Sphere` (and so
`Bead`) bit for bit: in that limit they use the sphere's own formulas.
"""
from functools import cached_property

import numpy as np
from scipy.special import ellipeinc, ellipkinc, elliprd

from levisim.core.constants import kB, ε0, π
from levisim.core.units import fg, nm
from levisim.physics.gas_damping import (
    cylinder_surface_quadrature,
    diffusion_matrix,
    ellipsoid_surface_quadrature,
    friction_matrix,
    gas_prefactor,
    spheres_surface_quadrature,
    surface_integrals,
)
from levisim.physics.material import _as_tensor
from levisim.physics.particle import Particle, Sphere

__all__ = [
    'Cylinder',
    'Disk',
    'Dumbbell',
    'Ellipsoid',
    'Rod',
    'Spheroid',
    'depolarisation_factors',
    'ellipsoid_polarisability',
    'ellipsoid_surface_area',
    'spheroid_depolarisation_factors',
]


####################################################################################################
# FORMULAS
# --------------------------------------------------------------------------------------------------
def depolarisation_factors(a, b, c):
    """
    Depolarisation factors (L_x, L_y, L_z) of the ellipsoid with semi-axes a, b, c along x, y, z,
    L_x = (abc/3)·R_D(b², c², a²) (Carlson's symmetric elliptic integral), and cyclically; they sum
    to 1, and are exactly (1/3, 1/3, 1/3) for a sphere
    [·]
    """
    a, b, c = (float(_) for _ in (a, b, c))
    if min(a, b, c) <= 0 or not np.all(np.isfinite((a, b, c))):
        raise ValueError(f'semi-axes must be positive and finite, not {(a, b, c)}')
    if a == b == c:
        return (1/3, 1/3, 1/3)
    a2, b2, c2 = a*a, b*b, c*c
    prefactor = a * b * c / 3
    return (
        prefactor * float(elliprd(b2, c2, a2)),
        prefactor * float(elliprd(c2, a2, b2)),
        prefactor * float(elliprd(a2, b2, c2)),
    )


def spheroid_depolarisation_factors(aspect_ratio):
    """
    Depolarisation factors (L_⊥, L_⊥, L_∥) of a spheroid with aspect ratio p = a/b (a the semi-axis
    along the symmetry axis z, b the equatorial semi-axis), in closed form ::

        prolate (p > 1), e² = 1 - 1/p²:  L_∥ = (1 - e²)/e² · (artanh(e)/e - 1)
        oblate  (p < 1), e² = 1 - p²:    L_∥ = 1/e² · (1 - √(1 - e²) arcsin(e)/e)
        L_⊥ = (1 - L_∥)/2

    (Bohren and Huffman (1983), Eqs. (5.33)-(5.34)). Within 1e-2 of a sphere, where these forms lose
    digits to cancellation, Carlson's integral is used instead.
    [·]
    """
    p = float(aspect_ratio)
    if not np.isfinite(p) or p <= 0:
        raise ValueError(f'the aspect ratio must be positive and finite, not {aspect_ratio}')
    if p == 1:
        return (1/3, 1/3, 1/3)
    if abs(p - 1) < 1e-2:
        return depolarisation_factors(1.0, 1.0, p)
    if p > 1:
        e = np.sqrt(1 - 1/p**2)
        L_par = (1 - e**2) / e**2 * (np.arctanh(e) / e - 1)
    else:
        e = np.sqrt(1 - p**2)
        L_par = (1 - np.sqrt(1 - e**2) * np.arcsin(e) / e) / e**2
    L_perp = (1 - L_par) / 2
    return (float(L_perp), float(L_perp), float(L_par))


def ellipsoid_polarisability(V, εr, L):
    """
    Quasi-static polarisability tensor of a homogeneous ellipsoid of volume V [m³] with relative
    permittivity εr (scalar or 3×3 body-frame tensor) and depolarisation factors L = (L_x, L_y, L_z),
    α = ε0 V (ε - 𝟙)[𝟙 + L(ε - 𝟙)]⁻¹ (diagonal α_i = ε0 V (εr - 1)/(1 + L_i(εr - 1)) for a scalar εr)
    [Fm²]
    """
    L = np.asarray(L, dtype=float)
    if np.ndim(εr) == 0:
        return np.diag(ε0 * V * (εr - 1) / (1 + L * (εr - 1)))
    identity = np.eye(3)
    ε = np.asarray(εr)
    return ε0 * V * (ε - identity) @ np.linalg.inv(identity + np.diag(L) @ (ε - identity))


def ellipsoid_surface_area(a, b, c):
    """
    Surface area of the ellipsoid with semi-axes a, b, c (Legendre's form, with the semi-axes sorted
    a ≥ b ≥ c) ::

        A = 2πc² + (2πab/sin φ)·(E(φ, k) sin²φ + F(φ, k) cos²φ),
        cos φ = c/a,   k² = a²(b² - c²)/(b²(a² - c²))

    with the incomplete elliptic integrals F and E (DLMF 19.33.2); exact closed forms for spheres
    [m²]
    """
    a, b, c = sorted((float(a), float(b), float(c)), reverse=True)
    if a == c:
        return 4*π * a**2
    φ = np.arccos(c / a)
    m = a**2 * (b**2 - c**2) / (b**2 * (a**2 - c**2))
    return float(2*π*c**2 + 2*π*a*b / np.sin(φ) * (ellipeinc(φ, m) * np.sin(φ)**2 + ellipkinc(φ, m) * np.cos(φ)**2))


def _spheroid_surface_area(a, b):
    """
    Surface area of the spheroid with semi-axis a along the symmetry axis and equatorial b ::

        prolate (a > b), e² = 1 - b²/a²:  A = 2πb² (1 + a arcsin(e)/(b e))
        oblate  (a < b), e² = 1 - a²/b²:  A = 2πb² (1 + (1 - e²) artanh(e)/e)
    [m²]
    """
    if a == b:
        return 4*π * a**2
    if a > b:
        e = np.sqrt(1 - b**2 / a**2)
        return float(2*π*b**2 * (1 + a * np.arcsin(e) / (b * e)))
    e = np.sqrt(1 - a**2 / b**2)
    return float(2*π*b**2 * (1 + (1 - e**2) * np.arctanh(e) / e))


####################################################################################################
# BASE CLASS
# --------------------------------------------------------------------------------------------------
class _Shape(Particle):
    """
    Common machinery of the analytic shapes: default material, gas damping from the cached surface
    integrals, polarisability from depolarisation factors, information string
    """
    def __init__(self, material=None, **kwargs):
        if material is None:
            from levisim.physics.materials import LEVISIM_DEFAULT_SILICA
            material = LEVISIM_DEFAULT_SILICA
        super().__init__(material, **kwargs)

    @staticmethod
    def _positive(name, value):
        value = float(value)
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be a positive, finite length in m, not {value}')
        return value

    # ----------------------------------------------------------------------------------------------
    # shape description (subclasses)
    # ----------------------------------------------------------------------------------------------
    @property
    def dimensions(self):
        """{name: length} of the shape's defining lengths [m]"""
        raise NotImplementedError

    def _surface_quadrature(self):
        """(points, normals, weights) of the surface about the centre of mass"""
        raise NotImplementedError

    @cached_property
    def equivalent_radius(self):
        """
        Radius of the sphere of the same volume, (3V/4π)^(1/3)
        [m]
        """
        return (3 * self.V / (4*π))**(1/3)

    # ----------------------------------------------------------------------------------------------
    # polarisability
    # ----------------------------------------------------------------------------------------------
    @cached_property
    def depolarisation_factors(self):
        """
        Depolarisation factors (L_x, L_y, L_z) in the body frame used for the polarisability
        [·]
        """
        raise NotImplementedError

    def static_polarisability_tensor(self, λ=None):
        """
        Quasi-static polarisability tensor in the body frame at vacuum wavelength λ (default: the
        particle's λ), α = ε0 V (ε - 𝟙)[𝟙 + L(ε - 𝟙)]⁻¹ (`ellipsoid_polarisability`)
        [Fm²]
        """
        εr = self.εr if λ is None else self.material.εr(float(λ))
        return ellipsoid_polarisability(self.V, εr, self.depolarisation_factors)

    # ----------------------------------------------------------------------------------------------
    # gas damping
    # ----------------------------------------------------------------------------------------------
    @cached_property
    def gas_surface_integrals(self):
        """
        The geometric surface integrals (G_n, G_d) of Martinetz et al. (2018), Eqs. (40)-(41)
        (`levisim.physics.gas_damping.surface_integrals`), computed once
        [m², m³, m⁴ blocks]
        """
        return surface_integrals(*self._surface_quadrature())

    def rotranslational_friction_matrix(self, gas, accommodation=1.0, surface_temperature_ratio=1.0):
        """
        The full 6×6 body-frame friction matrix Γ acting on (P, J), Martinetz et al. (2018) Eq. (40)
        (`levisim.physics.gas_damping.friction_matrix`); its diagonal blocks are
        `translational_damping_tensor` and `rotational_damping_tensor`
        [2π·Hz blocks]
        """
        return friction_matrix(self.gas_surface_integrals, self.m, self.inertia_tensor, gas,
                               accommodation, surface_temperature_ratio)

    def rotranslational_diffusion_matrix(self, gas, accommodation=1.0, surface_temperature_ratio=1.0):
        """
        The full 6×6 body-frame momentum diffusion matrix D, d〈(P, J)(P, J)ᵀ〉/dt = 2D, Martinetz et
        al. (2018) Eq. (41), valid for any γs
        [N²·s, N²·m·s, N²·m²·s blocks]
        """
        return diffusion_matrix(self.gas_surface_integrals, gas, accommodation, surface_temperature_ratio)

    def translational_damping_tensor(self, gas, accommodation=1.0, surface_temperature_ratio=1.0):
        """
        Centre-of-mass gas damping-rate tensor Γ_cm in the body frame, free-molecular regime
        (Martinetz et al. (2018), Eq. (40), evaluated on the surface)
        [2π·Hz]
        """
        return self.rotranslational_friction_matrix(gas, accommodation, surface_temperature_ratio)[:3, :3]

    def rotational_damping_tensor(self, gas, accommodation=1.0, surface_temperature_ratio=1.0):
        """
        Rotational gas damping-rate tensor Γ_rot (acting on the angular momentum J) in the body
        frame, free-molecular regime (Martinetz et al. (2018), Eq. (40))
        [2π·Hz]
        """
        return self.rotranslational_friction_matrix(gas, accommodation, surface_temperature_ratio)[3:, 3:]

    def translational_diffusion_tensor(self, gas, accommodation=1.0, surface_temperature_ratio=1.0):
        """
        Centre-of-mass momentum diffusion tensor D_cm in the body frame (Eq. (41) of Martinetz et
        al. (2018), any γs); D_cm = kB T m_n Γ_cm for γs = 1
        [N²·s]
        """
        return self.rotranslational_diffusion_matrix(gas, accommodation, surface_temperature_ratio)[:3, :3]

    def rotational_diffusion_tensor(self, gas, accommodation=1.0, surface_temperature_ratio=1.0):
        """
        Angular-momentum diffusion tensor D_rot in the body frame (Eq. (41) of Martinetz et al.
        (2018), any γs); D_rot = kB T Γ_rot I for γs = 1
        [N²·m²·s]
        """
        return self.rotranslational_diffusion_matrix(gas, accommodation, surface_temperature_ratio)[3:, 3:]

    # ----------------------------------------------------------------------------------------------
    # information
    # ----------------------------------------------------------------------------------------------
    @property
    def info_str(self):
        """
        String describing the particle
        """
        dims = ', '.join(f'{name} = {value / nm:.1f} nm' for name, value in self.dimensions.items())
        α = np.real_if_close(np.diag(_as_tensor(self.static_polarisability_tensor())))
        return (
            f'{type(self).__name__}(id={id(self)}'
            f'\n    {dims}'
            f'\n    material = {self.material.name}'
            f'\n    ρ = {self.ρ:.1f} kg/m³'
            f'\n    charge = {self.charge} e'
            f'\n    λ = {self.λ / nm:.1f} nm'
            f'\n    orientation (w, x, y, z) = {np.array2string(self.orientation, precision=5)}'
            f'\n'
            f'\n    volume = {self.V:.5e} m³'
            f'\n    area = {self.A:.5e} m²'
            f'\n    mass = {self.m / fg:.5f} fg'
            f'\n    inertia (body) = {np.array2string(np.diag(self.inertia_tensor), precision=5)} kg·m²'
            f'\n    depolarisation factors = {np.array2string(np.asarray(self.depolarisation_factors), precision=5)}'
            f'\n    α (body, diagonal) = {α} Fm²'
            f'\n'
            f'\n    trapping position phase, Φ/2π = x0/λtw = {self.Φ /(2*π):.3f}'
            f'\n)'
        )


####################################################################################################
# ELLIPSOIDS
# --------------------------------------------------------------------------------------------------
class Ellipsoid(_Shape):
    """
    A homogeneous ellipsoid with semi-axes a, b, c along the body x, y, z axes

    ::

        Ellipsoid(a, b, c, material=LEVISIM_DEFAULT_SILICA, charge=0, λ=1064 nm, Φ=π/2,
                  orientation=(1, 0, 0, 0))

    V = (4/3)π abc, m_n = ρV, I = (m_n/5)·diag(b² + c², a² + c², a² + b²), A from Legendre's
    elliptic-integral form, α from the depolarisation factors (module docstring), gas damping from
    the numerically integrated Eq. (40) of Martinetz et al. (2018). With a = b = c every number is
    that of `Sphere(r=a)` exactly.
    """
    def __init__(self, a, b, c, material=None, **kwargs):
        """
        Initialises an Ellipsoid instance
        """
        super().__init__(material, **kwargs)
        # Semi-axes along the body x, y, z axes
        # [m]
        self.a = self._positive('a', a)
        self.b = self._positive('b', b)
        self.c = self._positive('c', c)

    @property
    def semi_axes(self):
        """(a, b, c) along the body x, y, z axes [m]"""
        return (self.a, self.b, self.c)

    @property
    def dimensions(self):
        """{name: length} [m]"""
        return {'a': self.a, 'b': self.b, 'c': self.c}

    @property
    def is_sphere(self):
        """Whether all three semi-axes are equal (then the Sphere formulas are used) [bool]"""
        return self.a == self.b == self.c

    # ----------------------------------------------------------------------------------------------
    @cached_property
    def V(self):
        """
        Volume, (4/3)π abc
        [m³]
        """
        if self.is_sphere:
            return (4/3)*π * self.a**3
        return (4/3)*π * self.a * self.b * self.c

    @cached_property
    def A(self):
        """
        Surface area (`ellipsoid_surface_area`)
        [m²]
        """
        if self.is_sphere:
            return 4*π * self.a**2
        return ellipsoid_surface_area(self.a, self.b, self.c)

    @cached_property
    def inertia_tensor(self):
        """
        Moment of inertia tensor about the centre of mass, body frame, (m_n/5)·diag(b² + c², a² + c², a² + b²)
        [kg·m²]
        """
        if self.is_sphere:
            return (2/5) * self.m * self.a**2 * np.eye(3)
        a2, b2, c2 = self.a**2, self.b**2, self.c**2
        return (self.m / 5) * np.diag([b2 + c2, a2 + c2, a2 + b2])

    @cached_property
    def depolarisation_factors(self):
        """
        Depolarisation factors (L_x, L_y, L_z) (`depolarisation_factors`)
        [·]
        """
        return depolarisation_factors(self.a, self.b, self.c)

    # ----------------------------------------------------------------------------------------------
    # sphere limit: the Sphere arithmetic, so that the numbers are bit-identical
    # ----------------------------------------------------------------------------------------------
    @property
    def has_scalar_polarisability(self):
        """Whether α is one scalar for every direction (a sphere of an isotropic material) [bool]"""
        return self.is_sphere and np.ndim(self.εr) == 0

    @property
    def α(self):
        """
        Static polarisability seen by light polarised along the lab y axis in the current orientation
        (`Particle.α`); for a sphere of an isotropic material the Clausius-Mossotti scalar of `Sphere.α`
        [Fm²]
        """
        if self.has_scalar_polarisability:
            return Sphere._clausius_mossotti(self.V, self.εr)
        return super().α

    def effective_polarisability_along(self, direction, k):
        """
        ê·α_eff·ê (`Particle.effective_polarisability_along`); for a sphere of an isotropic material
        the scalar of `Sphere.effective_polarisability_along`
        [complex Fm²]
        """
        if self.has_scalar_polarisability:
            α = self.α
            return α / (1 - 1j * (k**3 / (6*π*ε0)) * α)
        return super().effective_polarisability_along(direction, k)

    def static_polarisability_tensor(self, λ=None):
        """
        Quasi-static polarisability tensor in the body frame (`ellipsoid_polarisability`); for a
        sphere, Clausius-Mossotti with exactly the arithmetic of `Sphere`
        [Fm²]
        """
        if self.is_sphere:
            εr = self.εr if λ is None else self.material.εr(float(λ))
            return _as_tensor(Sphere._clausius_mossotti(self.V, εr))
        return super().static_polarisability_tensor(λ)

    def _surface_quadrature(self):
        return ellipsoid_surface_quadrature(self.a, self.b, self.c)

    def translational_damping_tensor(self, gas, accommodation=1.0, surface_temperature_ratio=1.0):
        """
        Centre-of-mass gas damping-rate tensor Γ_cm in the body frame (free-molecular, Martinetz et
        al. (2018)); for a sphere, Table I with exactly the arithmetic of `Sphere`
        [2π·Hz]
        """
        if self.is_sphere:
            geometric_factor = (1 + π/8 * accommodation * surface_temperature_ratio)
            particle_factor = 1 / (self.ρ * self.a)
            gas_factor = (gas.m * gas.vth * gas.P) / (kB * gas.T)
            return geometric_factor * particle_factor * gas_factor * np.eye(3)
        return super().translational_damping_tensor(gas, accommodation, surface_temperature_ratio)

    def rotational_damping_tensor(self, gas, accommodation=1.0, surface_temperature_ratio=1.0):
        """
        Rotational gas damping-rate tensor Γ_rot in the body frame (free-molecular, Martinetz et al.
        (2018)); for a sphere, Table I with exactly the arithmetic of `Sphere`
        [2π·Hz]
        """
        if self.is_sphere:
            gas_factor = (gas.m * gas.vth * gas.P) / (kB * gas.T)
            return (5/4) * accommodation * gas_factor / (self.ρ * self.a) * np.eye(3)
        return super().rotational_damping_tensor(gas, accommodation, surface_temperature_ratio)


class Spheroid(Ellipsoid):
    """
    A homogeneous spheroid: semi-axis a along the symmetry axis (body z) and equatorial semi-axis b

    ::

        Spheroid(a, b, material=LEVISIM_DEFAULT_SILICA, charge=0, λ=1064 nm, Φ=π/2,
                 orientation=(1, 0, 0, 0))

    Prolate (needle-like) for a > b, oblate (disk-like) for a < b. An `Ellipsoid(b, b, a)` with
    closed-form depolarisation factors (`spheroid_depolarisation_factors`) and surface area. With
    a = b every number is that of `Sphere(r=a)` exactly.
    """
    def __init__(self, a, b, material=None, **kwargs):
        """
        Initialises a Spheroid instance
        """
        a, b = self._positive('a', a), self._positive('b', b)
        super().__init__(b, b, a, material=material, **kwargs)
        # semi-axis along the symmetry axis (body z) and equatorial semi-axis
        # [m]
        self.a_axial, self.b_equatorial = a, b

    @property
    def dimensions(self):
        """{name: length} [m]"""
        return {'a (axial)': self.a_axial, 'b (equatorial)': self.b_equatorial}

    @property
    def aspect_ratio(self):
        """a/b: > 1 prolate, < 1 oblate [·]"""
        return self.a_axial / self.b_equatorial

    @cached_property
    def depolarisation_factors(self):
        """
        (L_⊥, L_⊥, L_∥) in closed form (`spheroid_depolarisation_factors`)
        [·]
        """
        return spheroid_depolarisation_factors(self.aspect_ratio)

    @cached_property
    def A(self):
        """
        Surface area in closed form (prolate: 2πb²(1 + a arcsin(e)/(b e)); oblate:
        2πb²(1 + (1 - e²) artanh(e)/e))
        [m²]
        """
        if self.is_sphere:
            return 4*π * self.a**2
        return _spheroid_surface_area(self.a_axial, self.b_equatorial)


####################################################################################################
# CYLINDERS
# --------------------------------------------------------------------------------------------------
class Cylinder(_Shape):
    """
    A homogeneous solid circular cylinder of `length` ℓ and `radius` R, symmetry axis along body z

    ::

        Cylinder(length, radius, material=LEVISIM_DEFAULT_SILICA, charge=0, λ=1064 nm, Φ=π/2,
                 orientation=(1, 0, 0, 0))

    V = πR²ℓ, A = 2πRℓ + 2πR², and (Martinetz et al. (2018), Eq. (48b))

        I = m_n (ℓ²/12 + R²/4)(𝟙 - ẑẑ) + (m_n R²/2) ẑẑ

    The gas damping is the closed form of Table I of Martinetz et al. (2018) (flat end caps
    included). The polarisability uses the depolarisation factors of the spheroid with the same
    aspect ratio ℓ/(2R) and the cylinder's volume: an approximation (the field inside a finite
    cylinder is not uniform), exact in the long-rod and thin-disk limits; use `TabulatedParticle`
    for numerically exact values.
    """
    def __init__(self, length, radius, material=None, **kwargs):
        """
        Initialises a Cylinder instance
        """
        super().__init__(material, **kwargs)
        # Length along the symmetry axis and radius
        # [m]
        self.length = self._positive('length', length)
        self.radius = self._positive('radius', radius)

    @property
    def dimensions(self):
        """{name: length} [m]"""
        return {'length': self.length, 'radius': self.radius}

    @property
    def aspect_ratio(self):
        """ℓ/(2R): > 1 rod-like, < 1 disk-like [·]"""
        return self.length / (2 * self.radius)

    @cached_property
    def V(self):
        """
        Volume, πR²ℓ
        [m³]
        """
        return π * self.radius**2 * self.length

    @cached_property
    def A(self):
        """
        Surface area including both end caps, 2πRℓ + 2πR²
        [m²]
        """
        return 2*π * self.radius * self.length + 2*π * self.radius**2

    @cached_property
    def inertia_tensor(self):
        """
        Moment of inertia tensor about the centre of mass, body frame,
        m_n·diag(ℓ²/12 + R²/4, ℓ²/12 + R²/4, R²/2)
        [kg·m²]
        """
        perp = self.m * (self.length**2 / 12 + self.radius**2 / 4)
        return np.diag([perp, perp, self.m * self.radius**2 / 2])

    @cached_property
    def depolarisation_factors(self):
        """
        (L_⊥, L_⊥, L_∥) of the spheroid with the same aspect ratio ℓ/(2R) (an approximation, see
        the class docstring)
        [·]
        """
        return spheroid_depolarisation_factors(self.aspect_ratio)

    def _surface_quadrature(self):
        return cylinder_surface_quadrature(self.length, self.radius)

    def _table_I(self, gas, accommodation, surface_temperature_ratio):
        """(Γ⊥cm, Γ∥cm, Γ⊥rot, Γ∥rot) of Table I of Martinetz et al. (2018) [2π·Hz]"""
        αc, γs = float(accommodation), float(surface_temperature_ratio)
        R, ℓ = self.radius, self.length
        x = R / ℓ
        # n_g √(2π m_g kB T) R ℓ / m_n
        scale = 2*π * gas_prefactor(gas) * R * ℓ / self.m
        perp_cm = scale * (2 + αc * (-1/2 + π*γs/4 + x))
        par_cm = scale * (4*x + αc * (1 - 2*x + π*γs/2 * x))
        perp_rot = scale * ℓ**2 / (3*R**2 + ℓ**2) * (
            2 + 12*x**3 + αc * (-1/2 + π*γs/4 + 3*x + 6*x**2 + (3*π*γs/2 - 6) * x**3)
        )
        par_rot = scale * αc * (2 + x)
        return perp_cm, par_cm, perp_rot, par_rot

    def translational_damping_tensor(self, gas, accommodation=1.0, surface_temperature_ratio=1.0):
        """
        Centre-of-mass gas damping-rate tensor, body frame, Table I of Martinetz et al. (2018) ::

            Γ_cm = Γ⊥(𝟙 - ẑẑ) + Γ∥ ẑẑ,   with n_g √(2π m_g kB T) R ℓ / m_n times
            Γ⊥: 2 + αc(-1/2 + πγs/4 + R/ℓ),   Γ∥: 4R/ℓ + αc(1 - 2R/ℓ + πγs R/(2ℓ))

        [2π·Hz]
        """
        perp, par, _, _ = self._table_I(gas, accommodation, surface_temperature_ratio)
        return np.diag([perp, perp, par])

    def rotational_damping_tensor(self, gas, accommodation=1.0, surface_temperature_ratio=1.0):
        """
        Rotational gas damping-rate tensor, body frame, Table I of Martinetz et al. (2018) ::

            Γ⊥rot = n_g √(2π m_g kB T) R ℓ/m_n · ℓ²/(3R² + ℓ²) · {2 + 12R³/ℓ³
                    + αc[-1/2 + πγs/4 + 3R/ℓ + 6R²/ℓ² + (3πγs/2 - 6)R³/ℓ³]}
            Γ∥rot = n_g √(2π m_g kB T) R ℓ/m_n · αc (2 + R/ℓ)

        Rotation about the symmetry axis is damped only by diffusely reflected molecules (αc > 0).
        [2π·Hz]
        """
        _, _, perp, par = self._table_I(gas, accommodation, surface_temperature_ratio)
        return np.diag([perp, perp, par])


class Rod(Cylinder):
    """
    A rod: a `Cylinder(length, radius)` (the name for ℓ ≫ R, e.g. silicon nanorods); the same
    formulas and approximations
    """


class Disk(Cylinder):
    """
    A disk: a `Cylinder` with a `thickness` along its symmetry axis (body z) and a `radius`

    ::

        Disk(radius, thickness, material=LEVISIM_DEFAULT_SILICA, ...)

    The same formulas and approximations as `Cylinder` (for a thin disk the spheroid depolarisation
    factors tend to the exact (0, 0, 1)).
    """
    def __init__(self, radius, thickness, material=None, **kwargs):
        """
        Initialises a Disk instance
        """
        super().__init__(length=thickness, radius=radius, material=material, **kwargs)

    @property
    def thickness(self):
        """Thickness along the symmetry axis (the cylinder length) [m]"""
        return self.length

    @property
    def dimensions(self):
        """{name: length} [m]"""
        return {'radius': self.radius, 'thickness': self.thickness}


####################################################################################################
# DUMBBELL
# --------------------------------------------------------------------------------------------------
class Dumbbell(_Shape):
    """
    Two homogeneous spheres of radii r1 and r2 whose centres are a `separation` d ≥ r1 + r2 apart
    (touching by default), along the body z axis, with the origin at the centre of mass

    ::

        Dumbbell(r1, r2=r1, separation=r1 + r2, material=LEVISIM_DEFAULT_SILICA, ...)

    Sphere 1 sits at z1 = -m2 d/(m1 + m2), sphere 2 at z2 = m1 d/(m1 + m2). Volume, area and mass
    are sums; the inertia tensor follows from the parallel-axis theorem ::

        I_zz = (2/5)(m1 r1² + m2 r2²),   I_xx = I_yy = I_zz + m1 z1² + m2 z2² = I_zz + μ d²

    with μ = m1 m2/(m1 + m2). The polarisability is that of two coupled point dipoles at the centres,
    each with its sphere's Clausius-Mossotti α_i, in the quasi-static near field of the other ::

        p_1 = α_1 (E + G p_2),   p_2 = α_2 (E + G p_1),   G = (3ẑẑ - 𝟙)/(4πε0 d³)

    so α = p_1 + p_2 per unit field; for equal scalar α_s, α_∥ = 2α_s/(1 - 2α_s/(4πε0 d³)) and
    α_⊥ = 2α_s/(1 + α_s/(4πε0 d³)). Approximations: point dipoles (the higher multipoles induced near
    the contact of touching spheres are left out, which underestimates the anisotropy), no
    retardation across the dumbbell (k d ≪ 1). The radiation reaction is applied to the total
    tensor. The gas damping integrates Eq. (40) of Martinetz et al. (2018) over both sphere
    surfaces, neglecting that each sphere shadows part of the other from the gas (exact for
    separations ≫ r); unequal spheres have rotranslational coupling (`rotranslational_friction_matrix`).
    """
    def __init__(self, r1, r2=None, separation=None, material=None, **kwargs):
        """
        Initialises a Dumbbell instance
        """
        super().__init__(material, **kwargs)
        # Sphere radii
        # [m]
        self.r1 = self._positive('r1', r1)
        self.r2 = self.r1 if r2 is None else self._positive('r2', r2)
        # Centre-to-centre separation
        # [m]
        contact = self.r1 + self.r2
        self.separation = contact if separation is None else self._positive('separation', separation)
        if self.separation < contact * (1 - 1e-12):
            raise ValueError(
                f'separation {self.separation} m < r1 + r2 = {contact} m: overlapping spheres are not supported'
            )

    @property
    def dimensions(self):
        """{name: length} [m]"""
        return {'r1': self.r1, 'r2': self.r2, 'separation': self.separation}

    @cached_property
    def sphere_volumes(self):
        """(V1, V2) [m³]"""
        return ((4/3)*π * self.r1**3, (4/3)*π * self.r2**3)

    @cached_property
    def sphere_masses(self):
        """(m1, m2) [kg]"""
        return tuple(self.ρ * V for V in self.sphere_volumes)

    @cached_property
    def sphere_positions(self):
        """
        Body-frame z coordinates (z1, z2) of the sphere centres relative to the centre of mass
        [m]
        """
        m1, m2 = self.sphere_masses
        d = self.separation
        return (-m2 * d / (m1 + m2), m1 * d / (m1 + m2))

    @cached_property
    def V(self):
        """
        Volume, (4/3)π(r1³ + r2³)
        [m³]
        """
        return sum(self.sphere_volumes)

    @cached_property
    def A(self):
        """
        Surface area, 4π(r1² + r2²) (the contact point of touching spheres has no area)
        [m²]
        """
        return 4*π * (self.r1**2 + self.r2**2)

    @cached_property
    def inertia_tensor(self):
        """
        Moment of inertia tensor about the centre of mass, body frame (parallel-axis theorem)
        [kg·m²]
        """
        (m1, m2), (z1, z2) = self.sphere_masses, self.sphere_positions
        own = (2/5) * (m1 * self.r1**2 + m2 * self.r2**2)
        perp = own + m1 * z1**2 + m2 * z2**2
        return np.diag([perp, perp, own])

    def coupled_dipole_polarisabilities(self, λ=None):
        """
        (α_1, α_2): the polarisability tensors of each sphere in the dumbbell (its induced dipole per
        unit applied field, including the near field of the other), body frame, at vacuum wavelength λ
        [Fm²]
        """
        εr = self.εr if λ is None else self.material.εr(float(λ))
        α1 = _as_tensor(Sphere._clausius_mossotti(self.sphere_volumes[0], εr))
        α2 = _as_tensor(Sphere._clausius_mossotti(self.sphere_volumes[1], εr))
        G = (3 * np.diag([0.0, 0.0, 1.0]) - np.eye(3)) / (4*π*ε0 * self.separation**3)
        system = np.block([[np.eye(3), -α1 @ G], [-α2 @ G, np.eye(3)]])
        solution = np.linalg.solve(system, np.vstack([α1, α2]))
        return solution[:3], solution[3:]

    @cached_property
    def depolarisation_factors(self):
        """
        Effective depolarisation factors L_i that give the dumbbell's α_i with its total volume,
        α_i = ε0 V (εr - 1)/(1 + L_i(εr - 1)), for a real scalar εr (for information; they sum to 1
        only approximately); NaN for complex or tensor εr
        [·]
        """
        εr = self.εr
        if np.ndim(εr) != 0 or np.iscomplexobj(εr):
            return (np.nan, np.nan, np.nan)
        α = np.diag(self.static_polarisability_tensor())
        return tuple(float(L) for L in (ε0 * self.V * (εr - 1) / α - 1) / (εr - 1))

    def static_polarisability_tensor(self, λ=None):
        """
        Coupled-dipole polarisability tensor of the dumbbell in the body frame (class docstring)
        [Fm²]
        """
        α1, α2 = self.coupled_dipole_polarisabilities(λ)
        return α1 + α2

    def _surface_quadrature(self):
        z1, z2 = self.sphere_positions
        return spheres_surface_quadrature(((self.r1, (0.0, 0.0, z1)), (self.r2, (0.0, 0.0, z2))))


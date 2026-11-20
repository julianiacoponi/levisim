"""
Levitated particles: the Particle base class and the Sphere

A `Particle` combines a `Material` with a shape. It supplies what the dynamics need from the
particle: mass m, volume V, body-frame inertia tensor I, charge q, polarisability tensor α (with
an optional radiation-reaction correction) and the gas-damping tensors. `Sphere` is the only shape
implemented here; `Bead` (levisim.physics.bead) is a `Sphere` that keeps the original keyword
arguments and numbers. Non-spherical shapes, orientation dynamics and their drag tensors come later.
"""

from functools import cached_property

import numpy as np

from levisim.core.constants import e, g, kB, ε0, π
from levisim.core.units import aN, fg, nm
from levisim.physics.material import Material, _as_tensor

# trapping position phase along the cavity axis: antinode, halfnode, node = 0, π/4, π/2
_NODE = π/2


class Particle:
    """
    A levitated rigid particle made of one `Material` (base class)

    ::

        Particle(material, charge=0, λ=1064 nm, Φ=π/2, orientation=(1, 0, 0, 0))

    - material: a `Material` (see levisim.physics.materials for a library)
    - charge: net charge in units of the elementary charge e (an integer for a real particle) [·]
    - λ: vacuum wavelength at which a dispersive permittivity is evaluated for `εr`, `n` and `α` [m]
    - Φ: trapping position phase along the cavity axis, Φ = k·x0 [·] (π/2: node, 0: antinode)
    - orientation: unit quaternion (w, x, y, z) of the body frame relative to the lab frame. A
      placeholder for the rotation work: it is stored but not used by any calculation yet.

    Subclasses define the shape through `V`, `A`, `inertia_tensor` and `static_polarisability_tensor`,
    and may define the gas-damping tensors. Derived quantities are cached on first access, as in
    the original `Bead`.
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

        # Orientation of the body frame (unit quaternion w, x, y, z); placeholder, unused so far
        # [·]
        orientation = np.asarray(orientation, dtype=float)
        if orientation.shape != (4,) or not np.isclose(np.linalg.norm(orientation), 1.0):
            raise ValueError(f'orientation must be a unit quaternion (w, x, y, z), not {orientation}')
        self.orientation = orientation

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
        return α @ np.linalg.inv(np.eye(3) - 1j * (k**3 / (6*π*ε0)) * α)

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

    def rotational_damping_tensor(self, gas, accommodation=1.0):
        """
        Rotational gas damping-rate tensor Γ_rot, in the body frame
        [2π·Hz]
        """
        raise NotImplementedError(
            f'{type(self).__name__} has no rotational gas-damping tensor yet (only Sphere does)'
        )

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

    def rotational_damping_tensor(self, gas, accommodation=1.0):
        """
        Rotational gas damping-rate tensor of the sphere in the free-molecular regime
        [2π·Hz]

        Table I of Martinetz et al., Phys. Rev. E 97, 052112 (2018), doi:10.1103/PhysRevE.97.052112::

            Γ_rot = 10 αc n_g R² √(2π m_g kB T) / (3M) 𝟙 = (5/4) αc · m_g v̄ P/(kB T) / (ρ R) 𝟙

        Only diffusely reflected gas molecules (αc > 0) exert a torque on a sphere.
        """
        gas_factor = (gas.m * gas.vth * gas.P) / (kB * gas.T)
        return (5/4) * accommodation * gas_factor / (self.ρ * self.r) * np.eye(3)


"""
The optical tweezer (Tweezer): a focussed, elliptical Gaussian beam
"""

from functools import cached_property

import numpy as np

from levisim.core.constants import c, ħ, ε0, π
from levisim.core.units import (
    THz,
    degrees,
    mW,
    nm,
    μm,
)
from levisim.physics.fields.base import ForceElement, as_position


def _has_scalar_polarisability(particle):
    """
    Whether the particle's `α` is one scalar for every polarisation (an isotropic sphere, or any
    object with a scalar `α` that does not say otherwise), so the original scalar arithmetic applies
    """
    flag = getattr(particle, 'has_scalar_polarisability', None)
    if flag is None:
        return np.ndim(particle.α) == 0
    return bool(flag)


class Tweezer(ForceElement):
    """
    Defines an optical tweezer: a focussed, elliptical Gaussian beam

    A Tweezer is the optical `ForceElement` of a `Trap`: `force` is the time-averaged gradient force
    (the same arithmetic as `Trap.Ftw`), `potential` the dipole energy ``-(Re α_eff/2)|E|²``, `stiffness`
    the Hessian (closed form at the focus) and `noise` the photon-recoil force noise.

    The light is polarised along the lab y axis (`polarisation`). For a non-spherical or
    birefringent particle in a fixed orientation, α_eff is the projection ê·α_eff·ê of its lab-frame
    tensor on that polarisation (`effective_polarisability`), which sets the translational trap.
    """

    is_optical = True
    is_conservative = True
    def __init__(
        self,
        **kwargs,
    ):
        """
        Initialises a Tweezer instance
        """
        ############################################################################################
        # DEFINED PROPERTIES
        # ------------------------------------------------------------------------------------------
        # Tweezer laser power
        # [W]
        # Jon says 200mW, Tania uses 485mW
        self.Power = float(kwargs.get('Power', 485*mW))

        # ------------------------------------------------------------------------------------------
        # Tweezer laser wavelength
        # [m]
        self.λ = float(kwargs.get('λ', 1064*nm))

        # ------------------------------------------------------------------------------------------
        # Tweezer beam focus waist in x-axis
        # [m]
        self.wx0 = float(kwargs.get('wx0', 0.9276*μm))

        # ------------------------------------------------------------------------------------------
        # Tweezer beam focus waist in y-axis
        # [m]
        self.wy0 = float(kwargs.get('wy0', 1.0679*μm))

        ############################################################################################
        # MULTI-TRAP PROPERTIES
        # ------------------------------------------------------------------------------------------
        # Rotation angle of polarisation direction relative to cavity axis
        # [rad]
        # i.e. y_tw angle relative to x_cav
        self.θ = float(kwargs.get('θ', 0.0*degrees))

        # ------------------------------------------------------------------------------------------
        # Tweezer phase
        # [·]
        # relevant if making two tweezers interact via a cavity
        self.φ = float(kwargs.get('φ', 0.0*degrees))

        # ------------------------------------------------------------------------------------------
        # Polarisation (Jones vector in the focal plane)
        # [·]
        # None: linear along y, levisim's default. Otherwise a complex Jones vector (Ex, Ey) or a
        # lab-frame 3-vector with no z component, normalised here; see `elliptical_polarisation`
        polarisation = kwargs.get('polarisation', None)
        self._polarisation = None if polarisation is None else self._as_jones_vector(polarisation)

    ################################################################################################
    # INFORMATION PRINTING FUNCTIONS
    # ----------------------------------------------------------------------------------------------
    @property
    def info(self):
        """
        Prints information on the Tweezer instance
        """
        print(self.info_str)

    @property
    def info_str(self):
        """
        String describing the Tweezer instance
        """
        return (
            f'Tweezer(id={id(self)}'
            f'\n    Power = {self.Power / mW:.5f} mW'
            f'\n    λ = {self.λ / nm:.5f} nm'
            f'\n    wx0 = {self.wx0 / μm:.5f} μm'
            f'\n    wy0 = {self.wy0 / μm:.5f} μm'
            f'\n'
            f'\n    k = {self.k:.5e} m⁻¹'
            f'\n    ω/2π = {self.ω / THz:.5e} THz'
            f'\n    w0 = {self.w0 / μm:.5f} μm'
            f'\n    zR = {self.zR / μm:.5f} μm'
            f'\n    wz0 = {self.wz0 / μm:.5f} μm'
            f'\n    E0 = {self.E0:.5e} V/m'
            f'\n    I0 = {self.I0:.5e} W/m²'
            f'\n'
            f'\n    ellipticity = {self.ellipticity:.5f}'
            f'\n    flattening = {self.flattening:.5f}'
            f'\n'
            f'\n    θ = {self.θ / degrees:.5f} °'
            f'\n    φ = {self.φ / degrees:.5f} °'
            f'\n)'
        )

    ################################################################################################
    # DERIVED PROPERTIES
    # ----------------------------------------------------------------------------------------------
    @cached_property
    def k(self):
        """
        Angular wavenumber of the tweezer light
        [2π/m]
        """
        return 2*π / self.λ

    @cached_property
    def ω(self):
        """
        Angular frequency of the tweezer light
        [2π·Hz]
        """
        return c * self.k

    @cached_property
    def w0(self):
        """
        Gaussian beam waist (geometric mean of x, y waists)
        [m]
        """
        return np.sqrt(self.wx0 * self.wy0)

    @cached_property
    def zR(self):
        """
        Rayleigh range for the tweezer beam in its propagation axis (z-axis)
        (distance where beam waist becomes √2 its waist at the origin)
        [m]
        """
        return π * self.w0**2 / self.λ

    @cached_property
    def wz0(self):
        """
        Analogue to wx0, wy0 but for the tweezer's z-axis
        [m]
        """
        return np.sqrt(2) * self.zR

    @cached_property
    def waists(self):
        """
        A dict of each beam waist
        [m]
        """
        return {
            'x': self.wx0,
            'y': self.wy0,
            'z': self.wz0,
        }

    @cached_property
    def E0(self):
        """
        Magnitude of the electric field at the origin
        [V/m]
        """
        return np.sqrt(4 * self.Power / (π * self.w0**2 * ε0 * c))

    @cached_property
    def I0(self):
        """
        Magnitude of the electric field intensity at the origin
        [W/m²]
        """
        return 2 * self.Power / (π * self.w0**2)

    @cached_property
    def ellipticity(self):
        """
        Ellipticity of the Gaussian beam
        a.k.a. its eccentricity 0 < e < 1
        NOTE: https://mathworld.wolfram.com/Ellipticity.html
        [·]
        """
        a, b = max(self.wx0, self.wy0), min(self.wx0, self.wy0)
        return np.sqrt((a**2 - b**2)/a**2)

    @cached_property
    def flattening(self):
        """
        Flattening of the Gaussian beam

        f = (a - b)/a = 1 - b/a
        NOTE: https://en.wikipedia.org/wiki/Flattening
        [·]
        """
        return 1 - np.sqrt(1 - self.ellipticity**2)

    ################################################################################################
    # SPATIO-TEMPORAL FUNCTIONS
    # ----------------------------------------------------------------------------------------------
    # for use in the stochastic calculation of the tweezer potential done in Simulation
    def w(self, z):
        """
        The beam profile in units of the beam waist
        [·]
        """
        return np.sqrt(1 + (z / self.zR)**2)

    def wx(self, z):
        """
        The beam profile in the x direction
        [m]
        """
        return self.wx0 * self.w(z)

    def wy(self, z):
        """
        The beam profile in the y direction
        [m]
        """
        return self.wy0 * self.w(z)

    def wz(self, z):
        """
        The beam profile in the z direction
        [m]
        """
        return self.wz0 * self.w(z)

    def q(self, z):
        """
        Complex beam parameter
        [complex m]
        """
        return z - 1j * self.zR

    def ellipse(self, x, y, z):
        """
        The beam ellipse profile
        [·]
        """
        return (x / self.wx(z))**2 + (y / self.wy(z))**2

    def Gouy(self, z):
        """
        The Gouy phase shift
        [·]
        NOTE: The lack of minus sign is due to the choice of convention for wave propagation
        i.e. using the physicists exp i(kz - ωt)
        """
        # NOTE: this was np.arctan2(z / self.zR), which raises a TypeError (arctan2 needs two arguments)
        return np.arctan(z / self.zR)

    def R(self, z):
        """
        The radius of curvature of the wavefront
        [m]
        """
        return z * (1 + self.zR**2 / z**2)

    def E_field_amplitude(self, x, y, z):
        """
        The peak spatial amplitude of the electric field at a given point in 3D space
        [V/m]
        """
        return (self.E0 / self.w(z)) * np.exp(-self.ellipse(x, y, z))

    def E_field_phase(self, z, t):
        """
        The time-dependent phase of the electric field
        [·]
        """
        # NOTE: this is the only temporal function, but is not actually used
        # since this oscillation is averaged out via the rotating wave approximation
        return self.k * z + self.Gouy(z) + self.ω * t + self.φ

    def E_field(self, x, y, z, t):
        """
        The tweezer's electric field
        [V/m]
        """
        # NOTE: 1064nm tweezer light oscillates at about 282 THz >> kHz oscillations of the bead.
        # This means we do not do the above (which would be proper).
        # Instead, we consider that the cos²(θ) oscillation averages to ½ (when finding the potential Utw ~ |E|² in the Trap class).
        # So, we *artificially* include this ½ in the forces calculated in the Trap class.
        return  self.E_field_amplitude(x, y, z) * np.cos(self.E_field_phase(z, t))

    ################################################################################################
    # FORCE ELEMENT INTERFACE (see levisim.physics.fields)
    # ----------------------------------------------------------------------------------------------
    @staticmethod
    def _as_jones_vector(polarisation):
        e = np.asarray(polarisation, dtype=complex)
        if e.shape == (2,):
            e = np.array([e[0], e[1], 0.0], dtype=complex)
        if e.shape != (3,) or not np.all(np.isfinite(e)):
            raise ValueError(f'polarisation must be a Jones vector (Ex, Ey) or a 3-vector, not {polarisation}')
        norm = np.linalg.norm(e)
        if norm == 0:
            raise ValueError('polarisation must not be zero')
        if abs(e[2]) > 1e-12 * norm:
            raise ValueError('the polarisation of a Tweezer (propagating along z) must lie in the x-y plane')
        e = e / norm
        if not np.any(np.imag(e)):
            # a linear polarisation stays a real vector
            return np.real(e)
        return e

    @staticmethod
    def elliptical_polarisation(ψ, ζ=0.0):
        """
        Jones vector ê = cos ψ·ê1 + i sin ψ·ê2 of an elliptical polarisation with ellipticity angle ψ
        (0: linear along ê1, π/4: circular) and major axis ê1 = (-sin ζ, cos ζ, 0) at the angle ζ
        from levisim's y axis, ê2 = ẑ × ê1. The light's spin Im(ê* × ê) is sin 2ψ·ẑ, so ψ > 0 turns
        a particle about +z, the propagation direction (as Rudolph et al., Phys. Rev. A 103, 043514
        (2021), Eq. (12), whose ζ is measured from x)
        [·]
        """
        e1 = np.array([-np.sin(ζ), np.cos(ζ), 0.0])
        e2 = np.cross([0.0, 0.0, 1.0], e1)
        return np.cos(ψ) * e1 + 1j * np.sin(ψ) * e2

    @property
    def has_default_polarisation(self):
        """Whether the light is linearly polarised along y, levisim's default [bool]"""
        return getattr(self, '_polarisation', None) is None

    @property
    def polarisation(self):
        """
        Unit polarisation vector of the tweezer light at the focus, in the lab frame: by default
        linear along y (θ rotates the cavity axis, not the polarisation); a complex Jones vector
        for circular or elliptical light (`Tweezer(polarisation=...)`)
        [·]
        """
        if self.has_default_polarisation:
            return np.array([0.0, 1.0, 0.0])
        return self._polarisation.copy()

    @property
    def spin_density(self):
        """
        Im(ê* × ê): the light's spin per unit intensity, (0, 0, sin 2ψ) for an elliptical
        polarisation of ellipticity angle ψ, zero for linear light
        [·]
        """
        e = np.asarray(self.polarisation, dtype=complex)
        return np.imag(np.cross(np.conj(e), e))

    def field_vector(self, r):
        """
        Complex amplitude vector of the tweezer field at r = (x, y, z), E_field_amplitude(r)·ê (the
        phase k·z + Gouy, common to all components, is left out: it does not enter the torques)
        [V/m]
        """
        x, y, z = as_position(r)
        return self.E_field_amplitude(x, y, z) * np.asarray(self.polarisation, dtype=complex)

    def lab_effective_polarisability(self, particle, orientation=None):
        """
        Lab-frame effective polarisability tensor of the particle at this tweezer's wavenumber,
        R α_eff,body Rᵀ for `orientation` (default the particle's own); α_eff·𝟙 for a sphere of
        isotropic material
        [complex Fm²]
        """
        if _has_scalar_polarisability(particle):
            return (particle.α / (1 - 1j * (self.k**3 / (6*π*ε0)) * particle.α)) * np.eye(3)
        return particle.lab_tensor(particle.body_effective_polarisability_tensor(self.k), orientation)

    def torque(self, r, t, particle, orientation=None):
        """
        Time-averaged optical torque ½ Re(p* × E) on the particle at r, p = α_eff,lab E
        (`levisim.physics.rotation.optical_torque`); `orientation` (a unit quaternion) defaults to
        the particle's own. It contains the alignment torque of an anisotropic polarisability and
        the spin torque ½ Im(α_eff)·‖E‖²·Im(ê* × ê) of circularly or elliptically polarised light,
        which acts on a sphere too. t is not used.
        [N·m]
        """
        from levisim.physics.rotation import optical_torque
        α = self.lab_effective_polarisability(particle, orientation)
        return optical_torque(α, self.field_vector(r))

    def torque_stiffness(self, particle, orientation=None, r0=None):
        """
        Torque stiffness K_ij = -∂τ_i/∂θ_j of `torque` for small rotations θ about the lab axes at
        `orientation` (default the particle's own) and position r0 (default the focus) ::

            K_ij = -½ Re((∂_j α E)* × E)_i,   ∂_j α = G_j α - α G_j

        For linear polarisation along y and a particle whose principal axes lie along the lab axes
        it is diag(½E0²(α_y - α_z), 0, ½E0²(α_y - α_x)) with Re α_eff: rotations about x and z are
        confined when the most polarisable axis is along the polarisation, and rotation about the
        polarisation is free.
        [N·m/rad]
        """
        from levisim.physics.rotation import rotated_tensor_derivatives
        r0 = np.zeros(3) if r0 is None else as_position(r0)
        E = self.field_vector(r0)
        α = self.lab_effective_polarisability(particle, orientation)
        dα, _ = rotated_tensor_derivatives(α)
        K = np.zeros((3, 3))
        for j in range(3):
            K[:, j] = -0.5 * np.real(np.cross(np.conj(dα[j] @ E), E))
        return K

    def recoil_torque_noise(self, particle, orientation=None, r0=None):
        """
        Two-sided recoil torque noise of the scattered tweezer light, S_τ = ħk³/(12πε0)·Re(M M†)
        (`levisim.physics.rotation.recoil_torque_noise`), the rotational counterpart of `noise`
        (Rudolph et al., Phys. Rev. A 103, 043514 (2021), Eq. (31b)). Zero for a sphere of isotropic
        material.
        [N²·m²/Hz]
        """
        from levisim.physics.rotation import recoil_torque_noise
        r0 = np.zeros(3) if r0 is None else as_position(r0)
        α = self.lab_effective_polarisability(particle, orientation)
        return recoil_torque_noise(α, self.field_vector(r0), self.k)

    def torque_noise(self, particle, orientation=None, r0=None):
        """
        The element's torque noise (`ForceElement.torque_noise`): `recoil_torque_noise`
        [N²·m²/Hz]
        """
        return self.recoil_torque_noise(particle, orientation, r0)

    def effective_polarisability(self, particle):
        """
        Effective polarisability of the particle at this tweezer's wavenumber, including the
        radiation reaction: α_eff = α / (1 - i k³α/(6πε0)) (`Trap.α_eff`) for an isotropic sphere.
        For any other particle (a non-spherical shape, a birefringent material or a tabulated
        particle) it is ê·α_eff·ê, the lab-frame tensor α_eff = α_lab (𝟙 - i k³α_lab/(6πε0))⁻¹ of the
        particle's current orientation projected on the polarisation ê = ŷ
        (`Particle.effective_polarisability_along`).
        [complex Fm²]
        """
        if _has_scalar_polarisability(particle):
            return particle.α / (1 - 1j * (self.k**3 / (6*π*ε0)) * particle.α)
        return particle.effective_polarisability_along(self.polarisation, self.k)

    def recoil_coefficients(self, particle):
        """
        (‖Re α_eff ê‖, (c_x, c_y, c_z)): the polarisability and direction weights of the
        photon-recoil force noise, S_j = π ħ/(15 ε0)·(‖Re α_eff ê‖·E0/2π)²·k⁵·c_j (see `noise`)

        The induced dipole p ∝ α_eff ê radiates with the pattern 1 - (p̂·n)², whose mean of n_j² is
        (2 - p̂_j²)/5; the absorbed tweezer photons add 1 along the beam (z). So ::

            c_j = 2 - p̂_j² + 5 δ_jz,   p̂_j² = (Re α_eff ê)_j²/‖Re α_eff ê‖²

        which is (2, 1, 7) for a dipole along the polarisation y (an isotropic particle: then the
        first entry is Re α_eff exactly, as in the original code).
        [Fm², ·]
        """
        if self.has_default_polarisation:
            if _has_scalar_polarisability(particle):
                return np.real(self.effective_polarisability(particle)), np.array([2.0, 1.0, 7.0])
            p = np.real(particle.effective_polarisability_tensor(self.k) @ self.polarisation)
            magnitude = float(np.linalg.norm(p))
            p_hat2 = (p / magnitude)**2 if magnitude > 0 else np.array([0.0, 1.0, 0.0])
            return magnitude, 2 - p_hat2 + np.array([0.0, 0.0, 5.0])
        # a general (complex) polarisation: the cycle-averaged emission pattern of the dipole
        # p ∝ Re(α_eff) ê gives the weights 2 - |p_j|²/‖p‖² (plus 5 along the beam)
        p = np.real(self.lab_effective_polarisability(particle)) @ np.asarray(self.polarisation, dtype=complex)
        magnitude = float(np.linalg.norm(p))
        p_hat2 = np.abs(p / magnitude)**2 if magnitude > 0 else np.array([0.0, 1.0, 0.0])
        return magnitude, 2 - p_hat2 + np.array([0.0, 0.0, 5.0])

    @property
    def finite_difference_step(self):
        """
        1e-4 of the waist w0 (as `Trap.Ωj_at_z_eq`)
        [m]
        """
        return 1e-4 * self.w0

    def potential(self, r, particle):
        """
        Dipole energy of the particle in the (time-averaged) tweezer field, ``-(Re α_eff / 2) |E(r)|²``
        (`Trap.Utw`)
        [J]
        """
        x, y, z = r
        return -(np.real(self.effective_polarisability(particle)) / 2) * self.E_field_amplitude(x, y, z)**2

    def force(self, r, t, particle):
        """
        Gradient force of the tweezer, (Ftw,x, Ftw,y, Ftw,z), with exactly the arithmetic of `Trap.Ftw`
        (including the ½ of the time average of the light's cos²); t is not used
        [N]
        """
        x, y, z = r
        U = self.potential(r, particle)
        wx = self.wx(z)
        wy = self.wy(z)
        wz = self.wz(z)
        ellipse = self.ellipse(x, y, z)
        return np.array([
            0.5 * (4 * x / wx**2) * U,
            0.5 * (4 * y / wy**2) * U,
            0.5 * (4 * z / wz**2) * U * (1 - 2 * ellipse),
        ])

    def stiffness(self, r0=None, particle=None):
        """
        Stiffness tensor K_ij = -∂Ftw,i/∂r_j at r0
        [N/m]

        At the focus (r0 = None or the origin) in closed form, K = Re(α_eff) E0² diag(1/wx0², 1/wy0²,
        1/wz0²) (`Trap.kx`, `ky`, `kz`); elsewhere from central differences of `force` with step
        1e-4·w0.
        """
        if r0 is None or not np.any(as_position(r0)):
            α = np.real(self.effective_polarisability(particle))
            return np.diag([
                α * self.E0**2 / self.wx0**2,
                α * self.E0**2 / self.wy0**2,
                α * self.E0**2 / self.wz0**2,
            ])
        return super().stiffness(r0, particle)

    def noise(self, particle):
        """
        Photon-recoil force noise of the tweezer light scattered by the particle, two-sided, diagonal
        [N²/Hz]

        ::

            S_j = π ħ/(15 ε0) · (Re α_eff · E0/2π)² · k⁵ · c_j,   (c_x, c_y, c_z) = (2, 1, 7)

        It is the force noise that heats a mode of frequency Ωj at the rate S_j/(2 m ħ Ωj) phonons
        per second, `Trap.photon_recoil_heating_rates` (Piotrowski et al., Nat. Phys. 19, 1009
        (2023), Eq. (4)). Not used by FreeSpaceSimulation, which models no recoil heating. For an
        anisotropic particle the induced dipole need not lie along the polarisation, and Re α_eff and
        c_j become ‖Re α_eff ê‖ and the weights of `recoil_coefficients`.
        """
        if _has_scalar_polarisability(particle) and self.has_default_polarisation:
            α = np.real(self.effective_polarisability(particle))
            prefactor = π * ħ / (15 * ε0) * (α * self.E0 / (2*π))**2 * self.k**5
            return np.diag([2 * prefactor, prefactor, 7 * prefactor])
        α, coefficients = self.recoil_coefficients(particle)
        prefactor = π * ħ / (15 * ε0) * (α * self.E0 / (2*π))**2 * self.k**5
        return np.diag(coefficients * prefactor)

    def __repr__(self):
        return (f'Tweezer(Power={self.Power / mW:.6g} mW, λ={self.λ / nm:.6g} nm, '
                f'wx0={self.wx0 / μm:.6g} μm, wy0={self.wy0 / μm:.6g} μm)')

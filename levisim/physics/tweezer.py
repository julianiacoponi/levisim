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


class Tweezer(ForceElement):
    """
    Defines an optical tweezer: a focussed, elliptical Gaussian beam

    A Tweezer is the optical `ForceElement` of a `Trap`: `force` is the time-averaged gradient force
    (the same arithmetic as `Trap.Ftw`), `potential` the dipole energy -(Re α_eff/2)|E|², `stiffness`
    the Hessian (closed form at the focus) and `noise` the photon-recoil force noise.
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
    def effective_polarisability(self, particle):
        """
        Effective polarisability of the particle at this tweezer's wavenumber, including the
        radiation reaction: α_eff = α / (1 - i k³α/(6πε0)) (`Trap.α_eff`)
        [complex Fm²]
        """
        return particle.α / (1 - 1j * (self.k**3 / (6*π*ε0)) * particle.α)

    @property
    def finite_difference_step(self):
        """
        1e-4 of the waist w0 (as `Trap.Ωj_at_z_eq`)
        [m]
        """
        return 1e-4 * self.w0

    def potential(self, r, particle):
        """
        Dipole energy of the particle in the (time-averaged) tweezer field, -(Re α_eff / 2) |E(r)|²
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
        (2023), Eq. (4)). Not used by FreeSpaceSimulation, which models no recoil heating.
        """
        α = np.real(self.effective_polarisability(particle))
        prefactor = π * ħ / (15 * ε0) * (α * self.E0 / (2*π))**2 * self.k**5
        return np.diag([2 * prefactor, prefactor, 7 * prefactor])

    def __repr__(self):
        return (f'Tweezer(Power={self.Power / mW:.6g} mW, λ={self.λ / nm:.6g} nm, '
                f'wx0={self.wx0 / μm:.6g} μm, wy0={self.wy0 / μm:.6g} μm)')

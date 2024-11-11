import numpy as np
from functools import cached_property

from levisim.core.constants import π, c, ε0
from levisim.core.units import (
    degrees,
    μm, nm,
    THz,
    mW,
)


class Tweezer:
    """
    Defines an optical tweezer: a focussed, elliptical Gaussian beam
    """
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
        # [J/s]
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
        # MULTI-PHYSICS PROPERTIES
        # ------------------------------------------------------------------------------------------
        # Rotation angle of polarisation direction relative to cavity axis
        # [rad]
        # i.e. y_tw angle relative to x_cav
        self.θ = float(kwargs.get('θ', 0.0*degrees))

        # ------------------------------------------------------------------------------------------
        # Tweezer phase
        # [rad]
        # relevant if making two tweezers interact via a cavity
        self.φ = float(kwargs.get('φ', 0.0*degrees))

    ################################################################################################
    # INFORMATION PRINTING FUNCTIONS
    # ----------------------------------------------------------------------------------------------
    @property
    def info(self):
        """
        Prints info on the tweezer
        """
        print(
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
            f'\n    εtw = {self.εtw:.5e} V/m'
            f'\n'
            f'\n    ellipticity = {self.ellipticity:.5f}'
            f'\n    flattening = {self.flattening:.5f}'
            f'\n'
            f'\n    θ = {self.θ / degrees:.5f} °'
            f'\n    φ = {self.φ / degrees:.5f} °'
            f'\n)'
        )
        return

    ################################################################################################
    # DERIVED PROPERTIES
    # ----------------------------------------------------------------------------------------------
    @cached_property
    def k(self):
        """
        Angular wavenumber of the tweezer light
        [rad/m]
        """
        return 2*π / self.λ

    @cached_property
    def ω(self):
        """
        Angular frequency of the tweezer light
        [rad/s]
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
        # NOTE: could use a separate zR for x and y, i.e.
        # zRx = π * wx**2 / λ
        # zRy = π * wy**2 / λ
        # Conversation with Antonio suggests I could, but it clearly won't make much difference for these parameters
        # TODO: It would require redoing of the z-derivative of Utw
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
    def εtw(self):
        """
        Magnitude of the electric field at the origin
        [V/m]
        """
        # NOTE: As above, this could have different amplitudes for x, y direction
        # return np.sqrt(4 * self.Power / (self.zR * self.λ * ε0 * c))
        return np.sqrt(4 * self.Power / (π * self.w0**2 * ε0 * c))

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

    def ellipse(self, x, y, z):
        """
        The beam ellipse profile
        [·]
        """
        return (x / self.wx(z))**2 + (y / self.wy(z))**2

    def Gouy(self, z):
        """
        The Gouy phase shift
        [rad]
        NOTE: https://www.rp-photonics.com/gouy_phase_shift.html
        """
        return -np.arctan2(z / self.zR)

    def amplitude(self, x, y, z):
        """
        The peak spatial amplitude of the electric field at a given point in 3D space
        [V/m]
        """
        return (self.εtw / self.w(z)) * np.exp(-self.ellipse(x, y, z))

    def phase(self, z, t):
        """
        The time-dependent phase of the electric field
        [rad]
        """
        # NOTE: this is the only temporal function, but is not actually used
        # since this oscillation is averaged out via the rotating wave approximation
        return self.k * z + self.Gouy(z) + self.ω * t + self.φ

    def E(self, x, y, z):
        """
        The tweezer's electric field
        [V/m]
        """
        # A = self.amplitude(x, y, z)
        # θ = self.phase(z)
        # return  A * np.cos(θ)

        # NOTE: 1064nm tweezer light oscillates at about 2.8 THz >> kHz oscillations of the bead.
        # This means we do not do the above (which would be proper).
        # Instead, we consider that the cos²(θ) oscillation averages to ½ (when finding the potential Utw ~ |E|² in the Physics class).

        # So, we *artificially* include this ½ in the forces calculated in Physics.
        return self.amplitude(x, y, z)

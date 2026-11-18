import numpy as np

from levisim.core.constants import π, ħ, c, ε0
from levisim.core.units import (
    degrees,
    cm, μm,
    Hz, kHz, THz,
)

from levisim.physics.tweezer import Tweezer

class Cavity:
    """
    Defines a Fabry-Perot optical cavity
    """
    def __init__(
        self,
        **kwargs,
    ):
        """
        Initialises a Cavity instance
        """
        ############################################################################################
        # DEFINED PROPERTIES
        # ------------------------------------------------------------------------------------------
        # Length (distance between mirrors)
        # [m]
        self.L = float(kwargs.get('L', 1.223*cm))

        # ------------------------------------------------------------------------------------------
        # Cavity mode profile waist at centre
        # [m]
        self.waist = float(kwargs.get('w', 61*μm))

        # ------------------------------------------------------------------------------------------
        # Finesse
        # [·]
        # ~ number of times a photon reflects off the mirrors before leaving the cavity
        self.Finesse = float(kwargs.get('Finesse', 30_000))

        # ------------------------------------------------------------------------------------------
        # Detuning from tweezer
        # [2π·Hz]
        self.Δ = float(kwargs.get('Δ', -180*kHz))

        # ------------------------------------------------------------------------------------------
        # Manually overriden value for the cavity energy decay rate
        # [2π·Hz]
        self.override_κ = kwargs.get('κ')

        # ------------------------------------------------------------------------------------------
        # Tweezer angular frequency
        # [2π·Hz]
        # NOTE: one tweezer frequency is enough here. ωtw only sets the frame and the cavity's own
        # optical frequency ω = ωtw - Δ (and so k, λ, xR, E0). Further tones are handled by
        # levisim.multiphysics.multitone, which works in the frame of the cavity resonance and gives
        # each tone its own detuning Δ_k; using a different tone's frequency here would change ω
        # (hence k and E0) by |Δ_k - Δ|/ωtw ~ 1.6 × 10⁻⁹ for Δ/2π = 450 kHz at 1064 nm, which is
        # negligible against every other parameter uncertainty.
        self.ωtw = float(kwargs.get('ωtw', Tweezer().ω))

        ############################################################################################
        # MULTI-TRAP PROPERTIES
        # ------------------------------------------------------------------------------------------
        # Real and imaginary parts of the static part of the cavity mode photon field
        # [complex ·]
        self.αR = complex()
        self.αI = complex()

        # ------------------------------------------------------------------------------------------
        # Amplitude and phase quadratures of the static part of the cavity mode photon field
        # [·]
        self.A0 = float()
        self.P0 = float()

        # ------------------------------------------------------------------------------------------
        # Total cavity drive angular frequency of all tweeer bead traps scattering into the cavity
        # [2π·Hz]
        self.ωd_total = float()

        # ------------------------------------------------------------------------------------------
        # Total cavity drive energy of all tweeer bead traps scattering into the cavity
        # [J]
        self.Ed_total = float()

        # ------------------------------------------------------------------------------------------
        # Homodyne phase
        # [·]
        self.θhom = float(kwargs.get('θhom', 0*degrees))

        # ------------------------------------------------------------------------------------------
        # Heterodyne frequency
        # [2π·Hz]
        self.Ωhet = float(kwargs.get('Ωhet', 0*kHz))

    ################################################################################################
    # INFORMATION PRINTING FUNCTIONS
    # ----------------------------------------------------------------------------------------------
    @property
    def info(self):
        """
        Prints information on the Cavity instance
        """
        print(self.info_str)

    @property
    def info_str(self):
        """
        String describing the Cavity instance
        """
        multitrap_str = ''
        if self.A0:
            multitrap_str = (
                f'\n'
                f'\n    |α|² = {np.abs(self.αR + 1j*self.αI)**2:.5e} photons'
                f'\n    A0 = {self.A0:.5e}'
                f'\n    P0 = {self.P0:.5e}'
                f'\n    ωd_total/2π = {self.ωd_total / Hz:.5e} Hz'
                f'\n    Ed_total = {self.Ed_total:.5e} J'
            )
        return (
            f'Cavity(id={id(self)}'
            f'\n    Length = {self.L / cm:.5f} cm'
            f'\n    waist = {self.waist / μm:.5f} μm'
            f'\n    Finesse = {self.Finesse:.3e}'
            f'\n    Δ/2π = {self.Δ / kHz:.5f} kHz'
            f'\n'
            f'\n    κ/2π = {self.κ / kHz:.5f} kHz'
            f'\n    Volume = {self.V:.5e} m³'
            f'\n    ω/2π = {self.ω / THz:.5f} THz'
            f'\n    k = {self.k:.5e} m⁻¹'
            f'\n    λ = {self.λ:.5e} m'
            f'\n    xR = {self.xR:.5e} m'
            f'\n    E0 = {self.E0:.5e} V/m'
            f'\n'
            f'\n    θhom = {self.θhom / degrees:.5f} °'
            f'\n    Ωhet = {self.Ωhet / kHz:.5f} kHz'
            f'{multitrap_str}'
            f'\n)'
        )

    ################################################################################################
    # DERIVED PROPERTIES
    # ----------------------------------------------------------------------------------------------
    @property
    def κ(self):
        """
        Cavity energy decay rate
        [2π·Hz]
        """
        if self.override_κ:
            return float(self.override_κ)
        return π * c / (self.Finesse * self.L)

    @property
    def V(self):
        """
        Cavity mode volume
        [m³]
        """
        return self.L * π * self.waist**2 / 4

    # NOTE: these derived properties are plain @property, not cached_property. A cached_property
    # is computed once and then frozen, so changing an input afterwards silently left it stale:
    # e.g. after `cavity.info_str` (which MultiTrap logs on construction) set `cavity.Finesse =
    # 20_000` and κ stayed at the old 390.35 kHz instead of 780.73 kHz, and χC(ω) - which reads
    # self.κ - used the stale value; likewise `cavity.Δ = ...` left ω, k, λ, xR and E0 at the old
    # detuning. Each property is a few float operations, so recomputing on access costs nothing
    # measurable and gives bit-identical values. NOTE: a MultiTrap still copies the couplings it
    # derives from the cavity at construction, so changing a cavity input after building a
    # MultiTrap does not update that MultiTrap - build a new one.
    @property
    def ω(self):
        """
        Angular frequency of the cavity mode
        [2π·Hz]
        """
        return self.ωtw - self.Δ

    @property
    def k(self):
        """
        Angular wavenumber of the cavity mode
        [2π/m]
        """
        return self.ω / c

    @property
    def λ(self):
        """
        Wavelength of the cavity mode
        [m]
        """
        return 2*π / self.k

    @property
    def xR(self):
        """
        Rayleigh range for the cavity mode, in its cavity axis (x-axis)
        (distance where cavity mode waist becomes √2 its waist at the origin)
        [m]
        """
        return π * self.waist**2 / self.λ

    @property
    def E0(self):
        """
        Magnitude of the cavity's electric field at the origin
        [V/m]
        """
        return np.sqrt(ħ * self.ω / (2 * ε0 * self.V))

    ################################################################################################
    # FREQUENCY-DEPENDENT FUNCTIONS
    # ----------------------------------------------------------------------------------------------
    # for use in QLT calculations
    def χC(self, ω):
        """
        The optical susceptibility
        for photon annihilation in the cavity
        [complex (2π·Hz)⁻¹]
        """
        return (-1j*(ω + self.Δ) + self.κ / 2)**(-1)

    def χC_star(self, ω):
        """
        The optical susceptibility
        for photon creation in the cavity
        [complex (2π·Hz)⁻¹]
        """
        return np.conjugate(self.χC(-ω))

    def ηm(self, ω):
        """
        The difference of the Hermitian conjugate optical susceptibilities
        for photon annihilation and creation in the cavity
        [complex (2π·Hz)⁻¹]
        """
        return self.χC(ω) - np.conjugate(self.χC(-ω))

    def ηp(self, ω):
        """
        The sum of the Hermitian conjugate optical susceptibilities
        for photon annihilation and creation in the cavity
        [complex (2π·Hz)⁻¹]
        """
        return self.χC(ω) + np.conjugate(self.χC(-ω))

    def ξC(self, ω):
        """
        Product of the Hermitian conjugate optical susceptibilities
        for photon annihilation in the cavity
        [complex (2π·Hz)⁻¹]
        """
        return self.χC(ω) * np.conjugate(self.χC(-ω))

    def ηφ_A(self, φ, ω):
        """
        Generalised phase-dependent difference of the Hermitian conjugate optical susceptibilities
        for photon annihilation and creation in the cavity
        (Amplitude form ~ χC* + χC)
        [complex (2π·Hz)⁻¹]
        """
        return  np.conjugate(self.χC(-ω)) * np.exp(1j*φ) + self.χC(ω) * np.exp(-1j*φ)

    def ηφ_P(self, φ, ω):
        """
        Generalised phase-dependent difference of the Hermitian conjugate optical susceptibilities
        for photon annihilation and creation in the cavity
        (Phase form ~ i(χC* - χC))
        [complex (2π·Hz)⁻¹]
        """
        return  1j* (np.conjugate(self.χC(-ω)) * np.exp(1j*φ) - self.χC(ω) * np.exp(-1j*φ))

    def ηφ_T(self, φ, ω):
        """
        Generalised phase-dependent difference of the Hermitian conjugate optical susceptibilities
        for photon annihilation and creation in the cavity
        (Tania's form ~ χC - χC*)
        [complex (2π·Hz)⁻¹]
        """
        return  self.χC(ω) * np.exp(-1j*φ) - np.conjugate(self.χC(-ω)) * np.exp(1j*φ)

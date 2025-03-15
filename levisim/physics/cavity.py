import numpy as np
from functools import cached_property

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
        # [rad/s]
        self.Δ = float(kwargs.get('Δ', -180*kHz))

        # ------------------------------------------------------------------------------------------
        # Manually overriden value for the cavity energy decay rate
        # [rad/s]
        self.override_κ = kwargs.get('κ')

        # ------------------------------------------------------------------------------------------
        # Tweezer angular frequency
        # [rad/s]
        # TODO: for now, can only handle one tweezer frequency, even if many fill the cavity
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
        # [rad/s]
        self.ωd_total = float()

        # ------------------------------------------------------------------------------------------
        # Total cavity drive energy of all tweeer bead traps scattering into the cavity
        # [J]
        self.Ed_total = float()

        # ------------------------------------------------------------------------------------------
        # Homodyne angle
        # [rad]
        self.θhom = float(kwargs.get('θhom', 0))

        # ------------------------------------------------------------------------------------------
        # Heterodyne frequency
        # [rad/s]
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
            f'\n    k = {self.k:.5e} rad/m'
            f'\n    λ = {self.λ:.5e} m'
            f'\n    xR = {self.xR:.5e} m'
            f'\n    εcav = {self.εcav:.5e} V/m'
            f'\n'
            f'\n    θhom = {self.θhom / degrees:.5f} °'
            f'\n    Ωhet = {self.Ωhet / kHz:.5f} kHz'
            f'{multitrap_str}'
            f'\n)'
        )

    ################################################################################################
    # DERIVED PROPERTIES
    # ----------------------------------------------------------------------------------------------
    @cached_property
    def κ(self):
        """
        Cavity energy decay rate
        [rad/s]
        """
        if self.override_κ:
            return float(self.override_κ)
        return π * c / (self.Finesse * self.L)

    @cached_property
    def V(self):
        """
        Cavity mode volume
        [m³]
        """
        return self.L * π * self.waist**2 / 4

    # TODO: figure out whether it is important to use property vs. cached_property
    # if deciding to dynamically change a variable that affects a given property...
    @cached_property
    def ω(self):
        """
        Angular frequency of the cavity mode
        [rad/s]
        """
        return self.ωtw - self.Δ

    @cached_property
    def k(self):
        """
        Angular wavenumber of the cavity mode
        [rad/m]
        """
        return self.ω / c

    @cached_property
    def λ(self):
        """
        Wavelength of the cavity mode
        [m]
        """
        return 2*π / self.k

    @cached_property
    def xR(self):
        """
        Rayleigh range for the cavity mode, in its cavity axis (x-axis)
        (distance where cavity mode waist becomes √2 its waist at the origin)
        [m]
        """
        return π * self.waist**2 / self.λ

    @cached_property
    def εcav(self):
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
        [complex (rad/s)⁻¹]
        """
        return (-1j*(ω + self.Δ) + self.κ / 2)**(-1)

    def ηm(self, ω):
        """
        The difference of the conjugate optical susceptibilities
        for photon annihilation and creation in the cavity
        [complex (rad/s)⁻¹]
        """
        return self.χC(ω) - np.conjugate(self.χC(-ω))

    def ηp(self, ω):
        """
        The sum of the conjugate optical susceptibilities
        for photon annihilation and creation in the cavity
        [complex (rad/s)⁻¹]
        """
        return self.χC(ω) + np.conjugate(self.χC(-ω))

    def ηφ_A(self, φ, ω):
        """
        Generalised optical susceptibilities
        for photon annihilation and creation in the cavity
        (Amplitude form ~ χC* + χC)
        [complex (rad/s)⁻¹]
        """
        return  np.conjugate(self.χC(-ω)) * np.exp(1j*φ) + self.χC(ω) * np.exp(-1j*φ)

    def ηφ_P(self, φ, ω):
        """
        Generalised optical susceptibilities
        for photon annihilation and creation in the cavity
        (Phase form ~ i(χC* - χC))
        [complex (rad/s)⁻¹]
        """
        return  1j* (np.conjugate(self.χC(-ω)) * np.exp(1j*φ) - self.χC(ω) * np.exp(-1j*φ))

    def ηφ_T(self, φ, ω):
        """
        Generalised optical susceptibilities
        for photon annihilation and creation in the cavity
        (Tania's form ~ χC - χC*)
        [complex (rad/s)⁻¹]
        """
        return  self.χC(ω) * np.exp(-1j*φ) - np.conjugate(self.χC(-ω)) * np.exp(1j*φ)

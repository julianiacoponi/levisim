"""
The Fabry-Perot optical cavity (Cavity)
"""

import numpy as np

from levisim.core.constants import c, ħ, ε0, π
from levisim.core.units import (
    Hz,
    THz,
    cm,
    degrees,
    kHz,
    μm,
)
from levisim.physics.tweezer import Tweezer


class Cavity:
    """
    Defines a Fabry-Perot optical cavity

    Optional Fabry-Perot ports: ``κ_ports=(κ_1, κ_2, ...)`` gives the decay rate through each mirror
    and ``κ_loss`` the rate of every other loss (scattering, absorption), so that
    κ = κ_1 + κ_2 + ... + κ_loss (κ is then set from them, and an explicit κ must agree). Each port
    and the loss channel has its own vacuum input. ``detected_port`` (default 0) is the port whose
    output the spectra of `MultiTrap` describe. The default, one port carrying all of κ, is what
    every levisim spectrum assumed before ports existed. κ_loss alone keeps one port with κ − κ_loss.
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
        # Fabry-Perot ports: κ = κ_1 + κ_2 + ... + κ_loss (see port_rates)
        # [2π·Hz]
        # NOTE: by default one port carries all of κ (κ_ports=None, κ_loss=0), which is what every
        # spectrum of levisim assumed before ports existed; that default changes no number.
        κ_ports = kwargs.get('κ_ports')
        self.κ_ports = None if κ_ports is None else tuple(float(κ_j) for κ_j in np.atleast_1d(κ_ports))
        self.κ_loss = float(kwargs.get('κ_loss', 0.0))
        self.detected_port = int(kwargs.get('detected_port', 0))
        if self.κ_ports is not None:
            total = sum(self.κ_ports) + self.κ_loss
            if self.override_κ and not np.isclose(float(self.override_κ), total, rtol=1e-12, atol=0):
                raise ValueError(
                    f'κ = {float(self.override_κ):.6g} differs from the sum of κ_ports and κ_loss ({total:.6g}):'
                    ' give κ_ports (and κ_loss) only, or rates that add up to κ'
                )
            self.override_κ = total
        self._check_ports()

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
        self.αR = 0j
        self.αI = 0j

        # ------------------------------------------------------------------------------------------
        # Amplitude and phase quadratures of the static part of the cavity mode photon field
        # [·]
        self.A0 = 0.0
        self.P0 = 0.0

        # ------------------------------------------------------------------------------------------
        # Total cavity drive angular frequency of all tweeer bead traps scattering into the cavity
        # [2π·Hz]
        self.ωd_total = 0.0

        # ------------------------------------------------------------------------------------------
        # Total cavity drive energy of all tweeer bead traps scattering into the cavity
        # [J]
        self.Ed_total = 0.0

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
            f'{self._ports_info_str()}'
            f'\n    θhom = {self.θhom / degrees:.5f} °'
            f'\n    Ωhet = {self.Ωhet / kHz:.5f} kHz'
            f'{multitrap_str}'
            f'\n)'
        )

    def _ports_info_str(self):
        """Lines of info_str for non-default ports (empty for the default single port)"""
        if self.has_default_ports:
            return ''
        ports = ', '.join(f'{κ_j / kHz:.5f}' for κ_j in self.port_rates)
        return (
            f'\n    κ_ports/2π = ({ports}) kHz, κ_loss/2π = {self.κ_loss / kHz:.5f} kHz'
            f'\n    detected_port = {self.detected_port} (η_port = {self.detected_port_efficiency:.5f})'
        )

    ################################################################################################
    # FABRY-PEROT PORTS
    # ----------------------------------------------------------------------------------------------
    def _check_ports(self):
        rates = self.κ_ports if self.κ_ports is not None else ()
        if any(κ_j < 0 for κ_j in rates) or self.κ_loss < 0:
            raise ValueError(f'port rates {rates} and κ_loss = {self.κ_loss} must not be negative')
        if self.κ_ports is not None and len(self.κ_ports) == 0:
            raise ValueError('κ_ports needs at least one port')
        num_ports = len(rates) if self.κ_ports is not None else 1
        if not 0 <= self.detected_port < num_ports:
            raise ValueError(f'detected_port = {self.detected_port}, but the cavity has ports 0..{num_ports - 1}')
        if self.κ_ports is None and self.κ_loss and self.κ_loss >= self.κ:
            raise ValueError(f'κ_loss = {self.κ_loss:.6g} must be smaller than κ = {self.κ:.6g}')

    @property
    def has_default_ports(self):
        """True for the default single port carrying all of κ (no κ_ports, no κ_loss)"""
        return self.κ_ports is None and self.κ_loss == 0

    @property
    def port_rates(self):
        """
        Decay rates through the ports, (κ_1, κ_2, ...): `κ_ports`, or (κ − κ_loss,) for a single port
        [2π·Hz]
        """
        if self.κ_ports is not None:
            return self.κ_ports
        return (self.κ - self.κ_loss,)

    @property
    def detected_port_rate(self):
        """κ_j of the detected port [2π·Hz]"""
        return self.port_rates[self.detected_port]

    @property
    def detected_port_efficiency(self):
        """
        η_port = κ_j/κ of the detected port: the output of port j is √η_port·a_out + √(1 − η_port)·v,
        with a_out = −√κ·a + a_in the single-port output and v a vacuum independent of it
        (the other ports and the loss channel together), so a port acts on the spectra as a
        detection efficiency η_port
        [·]
        """
        return self.detected_port_rate / self.κ

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

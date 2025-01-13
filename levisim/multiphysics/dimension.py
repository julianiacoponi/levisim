import sympy
import numpy as np

from levisim.core.constants import π, ħ, kB
from levisim.core.units import (
    degrees,
    Hz, kHz,
)


class Dimension:
    """ Class to represent a mechanical degree of freedom in the system """
    def __init__(self, multitrap, label, **kwargs):

        # the multi-dimensional context in which this dimension exists!
        self.multitrap = multitrap

        # should be a string with either of 'xyz' as the first character
        # remaining characters are the bead index
        self.label = label
        self.axis, self.trap_index = label[0], label[1:]
        self.sub = multitrap.label_to_sub[label]
        # needed to know to assign Q1D
        self.noise_index = multitrap.label_to_noise_index[label]

        ############################################################################################
        # TOY MODEL PARAMETERS
        # ------------------------------------------------------------------------------------------
        # Relative phase of the tweezer associated with this dimension
        self.φ = kwargs.get('φ')

        self.T = kwargs.get('T')
        self.m = kwargs.get('m')

        # self.Ω = kwargs.get('Ω', self.trap.Ω_CS(self.axis))
        self.Ω = kwargs.get('Ω')
        self.Γ = kwargs.get('Γ')

        self.g = kwargs.get('g')
        self.gxy = kwargs.get('gxy')
        self.gxz = kwargs.get('gxz')
        self.gyz = kwargs.get('gyz')

        ############################################################################################
        # PHYSICAL MODEL PARAMETERS
        # ------------------------------------------------------------------------------------------
        self.Φ = None
        if not self.multitrap.toy_params:
            # e.g. label 'x1' will take traps[0]
            self.trap = multitrap.traps[int(self.trap_index) - 1]
            # Trapping position of the bead in the cavity's standing wave
            # [rad]
            self.Φ = self.trap.bead.Φ

        if self.φ is None:
            self.φ = self.trap.tweezer.φ

        if self.T is None:
            self.T = multitrap.gas.T

        if self.m is None:
            self.m = self.trap.bead.m

        if self.Ω is None:
            self.Ω = self.trap.Ωj[self.axis]

        if self.Γ is None:
            self.Γ = self.trap.Γth

        # Light-mech coupling goes H/ħ = -XL
        # X = b† + b
        # L = g.an† + g*.an
        # g = gAn + igPn
        # L = gAn.An + gPn.Pn
        # An = (an† + an)
        # Pn = i(an† - an)

        # a†,a = creation, annihilation operator for photons in the single cavity mode
        # an†,an = creation, annihilation operator for photons per-tweezer, 'rotated' cavity mode
        # φ = per tweezer phase
        # an = a.exp(-iφ)
        # an† = a†.exp(iφ)

        # L = g.a†.exp(iφ) + g*.a.exp(-iφ)
        # L = G.a† + G*.a = GA.A + GP.P
        # G = g.exp(iφ)
        # G* = g*.exp(-iφ)

        # L = GA.A + GP.P
        # A = (a† + a)
        # P = i(a† - a)
        # G = GA + iGP

        if self.g is None:
            self.g = self.light_mech_coupling

        if self.gxy is None:
            if multitrap.Cartesian_axes_per_trap == 1:
                self.gxy = 0
                self.gxz = 0
                self.gyz = 0

            if multitrap.Cartesian_axes_per_trap == 2:
                self.gxy = self.mech_mech_coupling('xy')
                self.gxz = 0
                self.gyz = 0

            if self.gxz is None and self.gyz is None:
                if multitrap.Cartesian_axes_per_trap == 3:
                    self.gxy = self.mech_mech_coupling('xy')
                    self.gxz = self.mech_mech_coupling('xz')
                    self.gyz = self.mech_mech_coupling('yz')

        # all coupling rates apart from gz should be negative
        for coupling in [
            self.gxy,
            self.gxz,
            self.gyz,
        ]:
            assert coupling <= 0, f'(gxy, gxz, gyz) = ({self.gxy / kHz}, {self.gxz / kHz}, {self.gyz / kHz}) kHz'

        if self.label[0] in ['x', 'y']:
            assert self.g <= 0, f'g{label} = {self.g / kHz} kHz'
        elif self.label[0] == 'z':
            assert self.g >= 0, f'g{label} = {self.g / kHz} kHz'

        ############################################################################################
        # DERIVED PROPERTIES
        # ------------------------------------------------------------------------------------------
        # Linear coupling of (x,y,z)-motion to the cavity light quadratures with tweezer phase included
        # [rad/s]
        if not self.φ:
            arg = self.CS_coupling_phase
            # NOTE: only sympy gives exact cos(π/2) = 0
            self.G = self.g * (float(sympy.cos(arg)) + 1j * float(sympy.sin(arg)))
        else:
            self.G = self.g * np.exp(1j * (self.CS_coupling_phase + self.φ))

        # ------------------------------------------------------------------------------------------
        # Number of phonons due to the thermal bath, via the Bose-Einstein distribution
        # [·]
        ħΩ = ħ * self.Ω
        kBT = kB * self.T
        self.noise_phonons_exact = (np.exp(ħΩ / kBT) - 1)**(-1)
        self.noise_phonons_approx = kBT / ħΩ

        # TODO: derive this properly?
        Δ, κ = self.multitrap.cavity.Δ, self.multitrap.cavity.κ
        Cp = 1 / ((Δ + self.Ω)**2 + (κ/2)**2)
        Cm = 1 / ((Δ - self.Ω)**2 + (κ/2)**2)
        self.cooling_rate = κ * np.abs(self.G)**2 * (Cp - Cm)
        self.cooling_factor = self.Γ / (self.Γ + self.cooling_rate)
        self.noise_phonons_with_cooling = self.noise_phonons_approx * self.cooling_factor
        self.T_with_cooling = ħΩ * self.noise_phonons_with_cooling / kB

        self.zpf = np.sqrt(ħ/2 / (self.m * self.Ω))

    def __repr__(self):
        return f'Dimension({self.label})'

    @property
    def info(self):
        """ Prints information on the dimension """
        Φ_str = '' if not self.Φ else f'\n    (Φ{self.trap_index}/2π) = (x0/λ) = {self.Φ / (2*π):.5f}'

        print(
            f'Dimension('
            f'\n    jn = label = {self.label}'
            f'\n    subscript = {self.sub}'
            f'\n    j = axis = {self.axis}'
            f'\n    n = trap_index = {self.trap_index}'
            f'\n    noise_index = {self.noise_index}'
            f'\n'
            f'\n    φ{self.trap_index} = {self.φ / degrees:.5f} °'
            f'{Φ_str}'
            f'\n'
            f'\n    Ω{self.label} = {self.Ω / kHz:.5f} kHz'
            f'\n    Γ{self.label} = {self.Γ / Hz:.5e} Hz'
            f'\n'
            f'\n    n{self.label} = {self.noise_phonons_exact:.5e} phonons (exact)'
            f'\n    n{self.label} = {self.noise_phonons_approx:.5e} phonons (approx.)'
            f'\n'
            f'\n    cooling_rate = {self.cooling_rate / Hz:.5e} Hz'
            f'\n    cooling_factor = {self.cooling_factor:.5e}'
            f'\n    cooled n{self.label} = {self.noise_phonons_with_cooling:.5e} phonons'
            f'\n    cooled T{self.label} = {self.T_with_cooling:.5e} K'
            f'\n'
            f'\n    g{self.label} = {self.g / kHz:.5f} kHz'
            f'\n    G{self.label} = {self.G / kHz:.5f} kHz'
            f'\n'
            f'\n    gxy = {self.gxy / kHz:.5f} kHz'
            f'\n    gxz = {self.gxz / kHz:.5f} kHz'
            f'\n    gyz = {self.gyz / kHz:.5f} kHz'
            f'\n'
            f'\n    {self.label}_zpf = {self.zpf:.5e} m'
            f'\n)'
        )

    def zero_point_fluctuation(self, axis):
        """
        Zero-point fluctuation of the position of the oscillator, per axis
        [m]
        """
        # TODO: sort CS frequency corrections
        # return np.sqrt(ħ/2 / (self.bead.m * self.Ω_CS(axis)))
        return np.sqrt(ħ/2 / (self.trap.bead.m * self.trap.Ωj[axis]))

    @property
    def CS_coupling_phase(self):
        """
        Phase of linear coherent scattering coupling of (x,y,z)-motion to the cavity light
        C(Φ) = Acos(Φ) + Psin(Φ) = a†.exp(iΦ) + a.exp(-iΦ)
        - Φ = 0 couples to amplitude quadrature, C(0) = A = a† + a
        - Φ = π/2 couples to phase quadrature, C(π/2) = P = i(a† - a)
        [rad]
        """
        return {
            'x': 0,
            'y': 0,
            'z': π/2 if self.φ else sympy.pi / 2,
            'z': π/2,
        }[self.axis]

    @property
    def light_mech_coupling(self):
        """
        Linear coupling of (x,y,z)-motion to the cavity light quadratures
        [rad/s]
        """
        kc, θtw = self.trap.tweezer.k, self.trap.tweezer.θ
        ωdsinθ = self.trap.ωd * np.sin(θtw)
        Φ = self.trap.bead.Φ
        zpf = self.zero_point_fluctuation(self.axis)

        # x, y couple to amplitude, have negative signs
        if self.axis == 'x':
            return -ωdsinθ * kc * zpf * np.sin(Φ) * np.sin(θtw)

        if self.axis == 'y':
            return -ωdsinθ * kc * zpf * np.sin(Φ) * np.cos(θtw)

        # z couples to the phase quadrature, has positive sign!
        if self.axis == 'z':
            kz = (kc - 1 / self.trap.tweezer.zR)
            return ωdsinθ * kz * zpf * np.cos(Φ)

    def mech_mech_coupling(self, axes):
        """
        Linear coupling between 2 axes of motion
        [rad/s]
        """
        kc, θtw = self.trap.tweezer.k, self.trap.tweezer.θ
        Φ = self.trap.bead.Φ

        A0, P0 = self.multitrap.cavity.A0, self.multitrap.cavity.P0
        ωdsinθ = self.trap.ωd * np.sin(θtw)

        # all have negative signs
        if axes == 'xy':
            x_zpf = self.zero_point_fluctuation(axes[0])
            y_zpf = self.zero_point_fluctuation(axes[1])
            return -A0 * ωdsinθ * (kc * x_zpf) * (kc * y_zpf) * np.sin(θtw) * np.cos(θtw) * np.cos(Φ)

        if axes == 'xz':
            x_zpf = self.zero_point_fluctuation(axes[0])
            z_zpf = self.zero_point_fluctuation(axes[1])
            kz = (kc - 1 / self.trap.tweezer.zR)
            return -P0 * ωdsinθ * (kc * x_zpf) * (kz * z_zpf) * np.sin(θtw) * np.sin(Φ)

        if axes == 'yz':
            y_zpf = self.zero_point_fluctuation(axes[0])
            z_zpf = self.zero_point_fluctuation(axes[1])
            kz = (kc - 1 / self.trap.tweezer.zR)
            return -P0 * ωdsinθ * (kc * y_zpf) * (kz * z_zpf) * np.cos(θtw) * np.sin(Φ)

    ################################################################################################
    # FREQUENCY RESPONSE FUNCTIONS
    # ----------------------------------------------------------------------------------------------
    def χM(self, ω):
        """
        The mechanical susceptibility for phonon annihilation
        [complex (rad/s)⁻¹]
        """
        return (-1j*(ω - self.Ω) + self.Γ / 2)**(-1)

    def μm(self, ω):
        """
        The difference of the conjugate mechanical susceptibilities
        for phonon annihilation and creation
        [complex (rad/s)⁻¹]
        """
        return self.χM(ω) - np.conjugate(self.χM(-ω))

    def μp(self, ω):
        """
        The sum of the conjugate mechanical susceptibilities
        for phonon annihilation and creation
        [complex (rad/s)⁻¹]
        """
        return self.χM(ω) + np.conjugate(self.χM(-ω))

    def M(self, ω):
        """
        The 1D normalisation factor for position
        [·]
        """
        return 1 + self.μm(ω) * self.multitrap.cavity.ηm(ω) * np.abs(self.G)**2

    def Q1D(self, ω):
        """
        The 1D noise vector for this dimension
        [·]
        """
        κ = self.multitrap.cavity.κ
        Γ = self.Γ
        G = self.G
        G_star = np.conjugate(G)

        χC, χC_star = self.multitrap.cavity.χC(ω), np.conjugate(self.multitrap.cavity.χC(-ω))
        χM, χM_star = self.χM(ω), np.conjugate(self.χM(-ω))

        μ = self.μm(ω)
        M = self.M(ω)

        Q = np.zeros(self.multitrap.num_operators, dtype=np.complex128)
        index = self.noise_index

        # a_in
        Q[0] = 1j * μ * np.sqrt(κ) * G_star * χC / M
        # a†_in
        Q[1] = 1j * μ * np.sqrt(κ) * G * χC_star / M
        # bj_in
        Q[index] = np.sqrt(Γ) * χM / M
        # bjn†_in
        Q[index + 1] = np.sqrt(Γ) * χM_star / M
        # Q should be empty in all other indices, since it is just the vector for the 1D position
        return Q


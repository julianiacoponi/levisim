import sympy
import numpy as np

from levisim.core.constants import π, ħ, kB
from levisim.core.units import (
    degrees,
    Hz, kHz,
)


__all__ = [
    'Dimension',
]


class Dimension:
    """
    Class to represent a mechanical degree of freedom in the system
        - φ: phase of tweezer that is trapping this dimension [rad]

        - T: thermal bath temperature [K]
        - m: mass of bead associated with this dimension [kg]

        - Ω: resonant frequency of motion associated with this dimension [2π·Hz]
        - Γ: mechanical damping rate [2π·Hz]

        - g: light-mech coupling rate [2π·Hz]
        - gxy, gxz, gyz: mech-mech coupling rates [2π·Hz]

    There are two ways to initialise this class:
    1) TOY MODEL:
    This takes 'toy model' parameters that define the physics directly

    2) PHYSICS MODEL:
    This derives the parameters from the original physics classes, and includes:
        - Φ: trapping position along the cavity's standing wave [rad]
    """
    def __init__(self, multitrap, label, **params):

        # the multi-dimensional context in which this dimension exists!
        self.multitrap = multitrap

        # should be a string with either of 'xyz' as the first character
        # remaining characters are the bead index
        self.label = label
        self.axis, self.trap_index = label[0], int(label[1:])
        self.sub = self.multitrap.label_to_sub[label]
        # needed to assign q1D
        self.noise_index = self.multitrap.label_to_noise_index[label]

        j, n = f'{self.axis}', f'{self.trap_index}'
        # the $$ are left out here so these can be slotted in with other latex more easily
        jn = f'{j}_{{{n}}}'
        self.latex = rf'{{{jn}}}'
        self.latex_hat = rf'\hat{{{j}}}_{{{n}}}'
        self.latex_PSD = rf'S_{{{jn + jn}}}'
        self.latex_Ω = rf'\Omega_{{{jn}}}'
        self.latex_Γ = rf'\Gamma_{{{jn}}}'
        self.latex_χ = rf'\chi_{{{jn}}}(\omega)'
        self.latex_χ_star = rf'\chi^*_{{{jn}}}(-\omega)'
        self.latex_μ = rf'\mu_{{{jn}}}'
        self.latex_μm = rf'\mu^-_{{{jn}}}'
        self.latex_μp = rf'\mu^+_{{{jn}}}'

        ############################################################################################
        # TOY MODEL PARAMETERS
        # ------------------------------------------------------------------------------------------
        # Relative phase of the tweezer associated with this dimension
        self.φ = params.get('φ')

        self.T = params.get('T')
        self.m = params.get('m')

        # self.Ω = params.get('Ω', self.trap.Ω_CS(self.axis))
        self.Ω = params.get('Ω')
        self.Γ = params.get('Γ')

        self.g = params.get('g')
        self.gxy = params.get('gxy')
        self.gxz = params.get('gxz')
        self.gyz = params.get('gyz')

        ############################################################################################
        # PHYSICAL MODEL PARAMETERS
        # ------------------------------------------------------------------------------------------
        self.Φ = None
        if not self.multitrap.toy_params:
            # e.g. label 'x1' will take traps[0]
            self.trap = self.multitrap.traps[self.trap_index - 1]
            # Trapping position of the bead in the cavity's standing wave
            # [rad]
            self.Φ = self.trap.bead.Φ

        if self.φ is None:
            self.φ = self.trap.tweezer.φ

        if self.T is None:
            self.T = self.multitrap.gas.T

        if self.m is None:
            self.m = self.trap.bead.m

        if self.Ω is None:
            self.Ω = self.trap.Ωj[self.axis]

        if self.Γ is None:
            self.Γ = self.trap.Γth

        # Light-mech coupling goes H/ħ = -X·L
        # X = b† + b
        # L = g·an† + g*·an
        # g = gAn + igPn
        # L = gAn·An + gPn·Pn
        # An = an† + an
        # Pn = i(an† - an)

        # NOTE: the below is the '+' tweezer phase convention since G = g·exp(+iφ)
        # a†, a = creation, annihilation operator for photons in the single, shared cavity mode
        # an†, an = creation, annihilation operator for photons per-tweezer, 'rotated' cavity mode
        # φn = per tweezer phase
        # an = a·exp(-iφn)
        # an† = a†·exp(iφn)

        # L = g·a†·exp(iφn) + g*·a·exp(-iφn)
        # L = G·a† + G*·a = GA·A + GP·P
        # G = g·exp(iφn)
        # G* = g·exp(-iφn)

        # L = GA·A + GP·P
        # A = a† + a
        # P = i(a† - a)
        # G = GA + iGP

        if self.g is None:
            self.g = self.light_mech_coupling

        if self.gxy is None:
            if self.multitrap.Cartesian_axes_per_trap == 1:
                self.gxy = 0
                self.gxz = 0
                self.gyz = 0

            if self.multitrap.Cartesian_axes_per_trap == 2:
                self.gxy = self.mech_mech_coupling('xy')
                self.gxz = 0
                self.gyz = 0

            if self.gxz is None and self.gyz is None:
                if self.multitrap.Cartesian_axes_per_trap == 3:
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
        Φjn_CS = self.linearised_coherent_scattering_coupling_phase
        sign = self.multitrap.tweezer_phase_sign_convention
        # ------------------------------------------------------------------------------------------
        # Linear coupling of (x, y, z)-motion to the cavity light quadratures with tweezer phase included
        # [2π·Hz]
        if not self.φ:
            # NOTE: only sympy gives exact cos(π/2) = 0
            self.G = self.g * (float(sympy.cos(sign * Φjn_CS)) + 1j * float(sympy.sin(sign * Φjn_CS)))
        else:
            self.G = self.g * np.exp(sign * 1j * self.φ_coupling)

        # NOTE: "+1" convention is what the majority of my code's derivation implies: G = g·exp(iφ)
        # i.e. positive phase to the creation operator when using a coupling Hamiltonian
        # H_CS = q(G*·a + G·a†) = g·q(a†·exp(iφ) + a·exp(-iφ))

        # NOTE: "-1" convention is what Tania's analytic derivation implies: G = g·exp(-iφ)
        # i.e. positive phase to the annihilation operator when using the coupling Hamiltonian
        # H_CS = q(G*·a + G·a†) = g·q(a†·exp(-iφ) + a·exp(iφ))

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
        """
        Prints information on the dimension
        """
        print(self.info_str)

    @property
    def info_str(self):
        """
        A string with information on the dimension
        """
        Φ_str = '' if not self.Φ else f'\n    (Φ{self.trap_index}/2π) = (x0/λ) = {self.Φ / (2*π):.5f}'
        return (
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

    @property
    def all_dimensions_in_other_traps(self):
        """
        Returns a list of all other dimensions that are part of the `MultiTrap` instance
        (useful for summing over k≠j)
        """
        return [dim for dim in self.multitrap.dimensions.values() if self.trap_index != dim.trap_index]

    @property
    def linearised_coherent_scattering_coupling_phase(self):
        """
        Phase of linear coherent scattering coupling of (x, y, z)-motion to the cavity light
        C(Φ) = A·cos(Φ) + P·sin(Φ) = a†·exp(iΦ) + a·exp(-iΦ)
        - Φ = 0 couples to amplitude quadrature, C(0) = A = a† + a
        - Φ = π/2 couples to phase quadrature, C(π/2) = P = i(a† - a)
        [rad]
        """
        return {
            'x': 0,
            'y': 0,
            'z': π/2 if self.φ else sympy.pi / 2,
        }[self.axis]

    @property
    def φ_coupling(self):
        """
        Phase defining which quadrature of cavity light this dimension is coupled to
        C(φ) = A·cos(φ) + P·sin(φ) = a†·exp(iφ) + a·exp(-iφ)
        [rad]
        """
        # TODO: why is the sign needed here *and* in the analytic noise vector method functions in self.multitrap?
        # make this choice at a level that also covers the definition of momentum being i(dag - hat) or i(hat - dag)
        Φjn_CS = self.linearised_coherent_scattering_coupling_phase
        return self.multitrap.tweezer_phase_sign_convention * float(Φjn_CS + self.φ)

    @property
    def light_mech_coupling(self):
        """
        Linear coupling of (x, y, z)-motion to the cavity light quadratures
        [2π·Hz]
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
        [2π·Hz]
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

    def zero_point_fluctuation(self, axis):
        """
        Zero-point fluctuation of the position of the oscillator, per axis
        [m]
        """
        # TODO: sort CS frequency corrections
        # return np.sqrt(ħ/2 / (self.bead.m * self.Ω_CS(axis)))
        return np.sqrt(ħ/2 / (self.trap.bead.m * self.trap.Ωj[axis]))

    ################################################################################################
    # FREQUENCY RESPONSE FUNCTIONS
    # ----------------------------------------------------------------------------------------------
    def χ(self, ω):
        """
        The mechanical susceptibility for phonon annihilation
        [complex (2π·Hz)⁻¹]
        """
        return (-1j*(ω - self.Ω) + self.Γ / 2)**(-1)

    def χ_star(self, ω):
        """
        The mechanical susceptibility for phonon creation
        [complex (2π·Hz)⁻¹]
        """
        return np.conjugate(self.χ(-ω))

    def μm(self, ω):
        """
        The difference of the conjugate mechanical susceptibilities
        for phonon annihilation and creation
        [complex (2π·Hz)⁻¹]
        """
        return self.χ(ω) - np.conjugate(self.χ(-ω))

    def μp(self, ω):
        """
        The sum of the conjugate mechanical susceptibilities
        for phonon annihilation and creation
        [complex (2π·Hz)⁻¹]
        """
        return self.χ(ω) + np.conjugate(self.χ(-ω))

    def M(self, ω):
        """
        The 1D normalisation factor for position
        [complex]
        """
        return 1 +  self.multitrap.cavity.ηm(ω) * np.abs(self.G)**2 * self.μm(ω)

    def Q_THERM(self, ω):
        """
        The thermal noise vector for this dimension
        [complex 1/√(2π·Hz)]
        """
        χ, χ_star = self.χ(ω), np.conjugate(self.χ(-ω))
        Q_TH = np.zeros(self.multitrap.num_operators, dtype=np.complex128)
        index = self.noise_index
        # bj_in
        Q_TH[index] = χ
        # bjn†_in
        Q_TH[index + 1] = χ_star
        return Q_TH

    def q1D(self, ω):
        """
        The 1D noise vector for this dimension
        [complex 1/√(2π·Hz)]
        """
        κ = self.multitrap.cavity.κ
        Γ = self.Γ
        G = self.G
        G_star = np.conjugate(G)

        χC, χC_star = self.multitrap.cavity.χC(ω), np.conjugate(self.multitrap.cavity.χC(-ω))
        χ, χ_star = self.χ(ω), np.conjugate(self.χ(-ω))

        μ = self.μm(ω)
        M = self.M(ω)

        Q = np.zeros(self.multitrap.num_operators, dtype=np.complex128)
        index = self.noise_index

        # a_in
        Q[0] = 1j * μ * np.sqrt(κ) * G_star * χC / M
        # a†_in
        Q[1] = 1j * μ * np.sqrt(κ) * G * χC_star / M
        # bj_in
        Q[index] = np.sqrt(Γ) * χ / M
        # bjn†_in
        Q[index + 1] = np.sqrt(Γ) * χ_star / M
        # Q should be empty in all other indices, since it is just the vector for the 1D position
        return Q

    ################################################################################################
    # PHASE DEPENDENT ZOO OF FUNCTIONS
    # ----------------------------------------------------------------------------------------------
    def K(self, ω):
        """
        Core phase-dependent function
        [complex 2π·Hz]
        """
        return 2 * np.sum([
            dim.g**2
            * dim.μm(ω)
            * np.sin(self.φ_coupling - dim.φ_coupling)**2
            for dim in self.all_dimensions_in_other_traps
        ])

    def H(self, ω):
        """
        Phase-dependent function that depends on K
        [complex]
        """
        return -np.conjugate(self.multitrap.cavity.χC(-ω)) * self.K(ω)

    def H_dag(self, ω):
        """
        Hermitian conjugate of H
        [complex]
        """
        return self.multitrap.cavity.χC(ω) * self.K(ω)

    def U(self, ω):
        """
        Phase-dependent function that depends on K
        [complex]
        """
        return 1 - np.conjugate(self.multitrap.cavity.χC(-ω)) * self.K(ω)

    def U_dag(self, ω):
        """
        Hermitian conjugate of U
        [complex]
        """
        return 1 + self.multitrap.cavity.χC(ω) * self.K(ω)

    def N(self, option, ω):
        """
        Phase-dependent function that appears in the normalisation function
        [complex]
        """
        match option:
            case 'K':
                return self.multitrap.cavity.ηm(ω) - self.multitrap.cavity.ξC(ω) * self.K(ω)
            case 'H':
                return self.multitrap.cavity.ηm(ω) + self.multitrap.cavity.χC(ω) * self.H(ω)
            case 'U':
                return self.multitrap.cavity.χC(ω) * self.U(ω) - np.conjugate(self.multitrap.cavity.χC(-ω))
            case 'H_dag':
                return self.multitrap.cavity.ηm(ω) - np.conjugate(self.multitrap.cavity.χC(-ω)) * self.H_dag(ω)
            case 'U_dag':
                return self.multitrap.cavity.χC(ω) - np.conjugate(self.multitrap.cavity.χC(-ω)) * self.U_dag(ω)

    def K_2(self, ω):
        """
        Alternative definition of K
        [complex 2π·Hz]
        """
        return np.sum([
            dim.g**2
            * dim.μm(ω)
            * (1 - np.exp(2j * (self.φ_coupling - dim.φ_coupling)))
            for dim in self.all_dimensions_in_other_traps
        ])

    def H_2(self, ω):
        """
        Alternative definition of H
        [complex]
        """
        return np.conjugate(self.multitrap.cavity.χC(-ω)) * np.sum([
            dim.g**2
            * dim.μm(ω)
            * (np.exp(2j * (self.φ_coupling - dim.φ_coupling)) - 1)
            for dim in self.all_dimensions_in_other_traps
        ])

    def H_dag_2(self, ω):
        """
        Alternative definition of H_dag
        [complex]
        """
        return self.multitrap.cavity.χC(ω) * np.sum([
            dim.g**2
            * dim.μm(ω)
            # NOTE: changing this -2j to 2j means the definition of N_2 with H_dag_2 (and hence U_dag_2)
            # lines up with the other N_2 definitions.
            # I don't know why the usual 1j to -1j rule for Hermitian conjugation doesn't work here...
            * (1 - np.exp(-2j * (self.φ_coupling - dim.φ_coupling)))
            for dim in self.all_dimensions_in_other_traps
        ])

    def U_2(self, ω):
        """
        Alternative definition of U_dag
        [complex]
        """
        return 1 + self.H_2(ω)

    def U_dag_2(self, ω):
        """
        Alternative definition of U_dag
        [complex]
        """
        return 1 + self.H_dag_2(ω)

    def N_2(self, option, ω):
        """
        Alternative definitions of N
        [complex]
        """
        match option:
            case 'K':
                return self.multitrap.cavity.ηm(ω) - self.multitrap.cavity.ξC(ω) * self.K_2(ω)
            case 'H':
                return self.multitrap.cavity.ηm(ω) + self.multitrap.cavity.χC(ω) * self.H_2(ω)
            case 'U':
                return self.multitrap.cavity.χC(ω) * self.U_2(ω) - np.conjugate(self.multitrap.cavity.χC(-ω))
            case 'H_dag':
                return self.multitrap.cavity.ηm(ω) - np.conjugate(self.multitrap.cavity.χC(-ω)) * self.H_dag_2(ω)
            case 'U_dag':
                return self.multitrap.cavity.χC(ω) - np.conjugate(self.multitrap.cavity.χC(-ω)) * self.U_dag_2(ω)

    def P(self, ω):
        """
        Phase-dependent function that equals zero when all phases are zero
        [complex]
        """
        return self.g**2 * self.μm(ω) * self.K(ω)

    def P_2(self, ω):
        """
        Alternative definition of P
        [complex]
        """
        return self.g**2 * self.μm(ω) * self.K_2(ω)


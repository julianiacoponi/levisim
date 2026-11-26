"""
One mechanical degree of freedom (Dimension) of a multi-particle cavity trap
"""

import numpy as np

from levisim.core.constants import kB, ħ, π
from levisim.core.units import (
    Hz,
    degrees,
    kHz,
)
from levisim.physics.coherent_scattering import hessian_to_coupling_rate

__all__ = [
    'Dimension',
]


# (cos, sin) of the coupling phases Φjn_CS = 0 (x, y) and ±π/2 (z), keyed by the multiple of π/2
EXACT_COS_SIN_OF_QUARTER_TURNS = {0: (1.0, 0.0), 1: (0.0, 1.0), -1: (0.0, -1.0)}


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
        # a libration 'α', 'β', 'γ' (rotation about the lab x, y, z axis) rather than a translation
        self.is_libration = self.axis in ('α', 'β', 'γ')
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
            # for a libration 'α', 'β', 'γ' the moment of inertia about that lab axis
            self.m = self.trap.Ij[self.axis] if self.is_libration else self.trap.bead.m

        if self.Ω is None:
            # the tweezer frequency, plus the static cavity shifts if MultiTrap(static_cavity_shifts=True)
            self.Ω = self.mechanical_frequency(self.axis)

        if self.Γ is None:
            # the gas damping of this axis (Trap.Γth for a sphere; the rotational gas damping
            # Trap.Γrot_j for a libration)
            self.Γ = self.trap.Γrot_j[self.axis] if self.is_libration else self.trap.Γth_j[self.axis]

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

        if self.gxy is None and not isinstance(self.multitrap.Cartesian_axes_per_trap, (int, np.integer)):
            # a tuple of axis names: the coherent-scattering couplings of the Cartesian pairs present
            # (those with a libration are built by MultiTrap.librational_coupling_rate)
            axes = self.multitrap.axes_per_trap
            self.gxy = self.mech_mech_coupling('xy') if 'x' in axes and 'y' in axes else 0
            self.gxz = self.mech_mech_coupling('xz') if 'x' in axes and 'z' in axes else 0
            self.gyz = self.mech_mech_coupling('yz') if 'y' in axes and 'z' in axes else 0

        if self.gxy is None:
            if self.multitrap.Cartesian_axes_per_trap == 1:
                self.gxy = 0
                self.gxz = 0
                self.gyz = 0

            if self.multitrap.Cartesian_axes_per_trap == 2:
                self.gxy = self.mech_mech_coupling('xy')
                self.gxz = 0
                self.gyz = 0

            if self.gxz is None and self.gyz is None and self.multitrap.Cartesian_axes_per_trap == 3:
                self.gxy = self.mech_mech_coupling('xy')
                self.gxz = self.mech_mech_coupling('xz')
                self.gyz = self.mech_mech_coupling('yz')

        # NOTE: the signs of the couplings are not fixed. For red detuning (Δ < 0), 0 ≤ θ ≤ π/2 and
        # beads with 0 ≤ Φ ≤ π/2 they are gx, gy, gxy, gxz, gyz ≤ 0 and gz ≥ 0 (this used to be
        # asserted), but gx, gy ∝ -sin(Φ), gz ∝ cos(Φ), gxy ∝ -A0·cos(Φ) and gxz, gyz ∝ -P0·sin(Φ) with
        # A0 ∝ -Δ·∑cos(Φn) and P0 ∝ ∑cos(Φn), so a bead on the other side of a node or antinode, blue
        # detuning, or toy couplings of either sign legitimately change them (and round-off gave
        # gxz ~ +1e-16 at Φ = π). Only relative signs matter: e.g. Φ → π - Φ for every bead flips
        # gz, gxz and gyz, which is the relabelling z → -z, and leaves every spectrum unchanged.

        ############################################################################################
        # DERIVED PROPERTIES
        # ------------------------------------------------------------------------------------------
        Φjn_CS = self.linearised_coherent_scattering_coupling_phase
        sign = self.multitrap.tweezer_phase_sign_convention
        # ------------------------------------------------------------------------------------------
        # Linear coupling of (x, y, z)-motion to the cavity light quadratures with tweezer phase included
        # [2π·Hz]
        if not self.φ and round(sign * Φjn_CS / (π/2)) * (π/2) == sign * Φjn_CS:
            # NOTE: exact cos and sin of Φjn_CS = 0 or ±π/2, since np.cos(π/2) = 6.1e-17 would leave a
            # spurious amplitude-quadrature coupling of z (these are the values sympy gave, which was
            # used here before; it took about a second to import)
            cos_Φ, sin_Φ = EXACT_COS_SIN_OF_QUARTER_TURNS[round(sign * Φjn_CS / (π/2))]
            self.G = self.g * (cos_Φ + 1j * sin_Φ)
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

        # NOTE: this is the standard weak-coupling sideband-cooling (optical damping) rate,
        #   Γ_opt = |G|²·κ·[1/((Δ + Ω)² + (κ/2)²) - 1/((Δ - Ω)² + (κ/2)²)] = A₋ - A₊
        # (anti-Stokes minus Stokes scattering rates; Aspelmeyer, Kippenberg & Marquardt, Rev. Mod.
        # Phys. 86, 1391 (2014), quantum-noise treatment of radiation-pressure cooling), with levisim's Δ = ω_tweezer - ω_cavity (red detuning
        # Δ < 0 cools: the first term peaks at Δ = -Ω) and |G| the coupling of H/ħ = -(b + b†)(G·a† + G*·a),
        # which is the g of that reference. No factor or sign is missing: checked against the exact
        # optical damping -2·Re(λ) - Γ of the 1×1 drift-matrix eigenvalue λ, the ratio is 1.00007,
        # 1.0012 and 1.006 for g = -5, -20 and -40 kHz (κ = 2π·390 kHz); the residual is the
        # strong-coupling correction O(|G|²/κ²). Two limits of the derived quantities below: they
        # ignore the backaction floor n_min = A₊/(A₋ - A₊) (so the cooled phonon number is an
        # underestimate near the ground state), and they are per dimension, with no hybridisation.
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
            f'\n    static cavity shift δΩ{self.label} = {self.static_cavity_shift / kHz:.5f} kHz'
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

        ::

            C(Φ) = A·cos(Φ) + P·sin(Φ) = a†·exp(iΦ) + a·exp(-iΦ)

        - Φ = 0 couples to amplitude quadrature, C(0) = A = a† + a
        - Φ = π/2 couples to phase quadrature, C(π/2) = P = i(a† - a)

        [rad]
        """
        if self.is_libration:
            # the phase of the complex coupling (0 for a linearly polarised tweezer), see
            # light_mech_coupling
            if not hasattr(self, '_libration_coupling_phase'):
                _ = self.light_mech_coupling  # sets _libration_coupling_phase
            return self._libration_coupling_phase
        return {
            'x': 0,
            'y': 0,
            'z': π/2,
        }[self.axis]

    @property
    def φ_coupling(self):
        """
        Phase defining which quadrature of cavity light this dimension is coupled to

        ::

            C(φ) = A·cos(φ) + P·sin(φ) = a†·exp(iφ) + a·exp(-iφ)

        [rad]
        """
        # NOTE: the sign is applied here and again in G = g·exp(sign·i·φ_coupling) (and, until Nov 2026,
        # again in the analytic noise-vector functions, which used exp(±sign·i·φ_coupling)), so it always
        # cancels: G = g·exp(+i(Φ_CS + φ)) for both values of `tweezer_phase_sign_convention`, and every
        # spectrum is bit-identical for sign = ±1 (checked for LAM and the analytic methods on the 2×2
        # PHASE_DEPENDENT toy system). The second sign was only ever needed to undo the first. The
        # analytic functions now use G (or `MultiTrap.coupling_phases`) directly. Choosing the "-1"
        # (G = g·exp(-iφ)) convention of PRR 8 is equivalent to relabelling φ → -φ (PRR 8's tweezer
        # phase is Φ_j = -φ_j), i.e. it is a choice of input parameters, not of the equations.
        Φjn_CS = self.linearised_coherent_scattering_coupling_phase
        return self.multitrap.tweezer_phase_sign_convention * float(Φjn_CS + self.φ)

    @property
    def light_mech_coupling(self):
        """
        Linear coupling of (x, y, z)-motion to the cavity light quadratures
        [2π·Hz]

        The wavenumber k, polarisation angle θ and Rayleigh range (Gouy term of the z coupling) are
        those of the light the bead scatters into the cavity, `Trap.drive_tweezer` (the trapping
        tweezer unless the trap has a `cavity_drive_tweezer`). For a bead away from that tweezer's
        focus the Gouy term is still the focal one, k - 1/zR; the x, y couplings do not have it.
        """
        if self.is_libration:
            # G = c·θ_zpf from levisim.physics.coherent_scattering.librational_expansion: a real
            # coupling (to the amplitude quadrature, like x and y) for a linearly polarised tweezer;
            # otherwise g = |G| and the phase goes into linearised_coherent_scattering_coupling_phase
            from levisim.physics.rotation import LIBRATION_AXIS_INDEX
            c = self.multitrap.librational_light_couplings[self.trap_index][LIBRATION_AXIS_INDEX[self.axis]]
            G0 = complex(c * self.zero_point_fluctuation(self.axis))
            if G0.imag == 0:
                self._libration_coupling_phase = 0
                return G0.real
            self._libration_coupling_phase = float(np.angle(G0))
            return abs(G0)
        k, θtw = self.trap.drive_tweezer.k, self.trap.drive_tweezer.θ
        kc = self.standing_wave_k
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
            # the propagation phase of the tweezer, so its wavenumber (with the Gouy term)
            kz = (k - 1 / self.trap.drive_tweezer.zR)
            return ωdsinθ * kz * zpf * np.cos(Φ)

    def mech_mech_coupling(self, axes):
        """
        Linear coupling between 2 axes of motion
        [2π·Hz]
        """
        k, θtw = self.trap.drive_tweezer.k, self.trap.drive_tweezer.θ
        kc = self.standing_wave_k
        kz = (k - 1 / self.trap.drive_tweezer.zR)
        Φ = self.trap.bead.Φ

        A0, P0 = self.multitrap.cavity.A0, self.multitrap.cavity.P0
        ωdsinθ = self.trap.ωd * np.sin(θtw)

        # all have negative signs
        if axes == 'xy':
            x_zpf = self.zero_point_fluctuation(axes[0])
            y_zpf = self.zero_point_fluctuation(axes[1])
            gxy = -A0 * ωdsinθ * (kc * x_zpf) * (kc * y_zpf) * np.sin(θtw) * np.cos(θtw) * np.cos(Φ)
            if getattr(self.multitrap, 'static_cavity_shifts', False):
                # the cavity's dispersive potential also couples x and y, since its stiffness is
                # along the cavity axis (see MultiTrap.setup_static_cavity_shifts)
                K_xy = self.multitrap.static_cavity_hessians[self.trap_index][0, 1]
                if K_xy != 0:
                    gxy = gxy + hessian_to_coupling_rate(K_xy, x_zpf, y_zpf)
            return gxy

        if axes == 'xz':
            x_zpf = self.zero_point_fluctuation(axes[0])
            z_zpf = self.zero_point_fluctuation(axes[1])
            return -P0 * ωdsinθ * (kc * x_zpf) * (kz * z_zpf) * np.sin(θtw) * np.sin(Φ)

        if axes == 'yz':
            y_zpf = self.zero_point_fluctuation(axes[0])
            z_zpf = self.zero_point_fluctuation(axes[1])
            return -P0 * ωdsinθ * (kc * y_zpf) * (kz * z_zpf) * np.cos(θtw) * np.sin(Φ)

    @property
    def standing_wave_k(self):
        """
        Wavenumber of the cavity standing wave cos(k·Y_c + Φ) along which the bead moves: the
        cavity's, `Cavity.k` = (ωtw - Δ)/c, with `MultiTrap(cavity_wavenumber=True)`, otherwise
        (the levisim 0.1.0 default) the drive tweezer's. It sets the x, y factors of the couplings;
        the z factor kz = k - 1/zR is the tweezer's propagation phase either way.
        [m⁻¹]
        """
        if getattr(self.multitrap, 'cavity_wavenumber', False):
            return self.multitrap.cavity.k
        return self.trap.drive_tweezer.k

    def mechanical_frequency(self, axis):
        """
        The frequency of this bead's oscillator along `axis`: the tweezer frequency `Trap.Ωj`,
        shifted by the static cavity field when `MultiTrap(static_cavity_shifts=True)`, and by the
        particle's own block of the interactions' stiffness with `MultiTrap(interactions=[...])`; for
        a hybrid Paul trap the exact (Floquet) secular frequency with
        `MultiTrap(paul_exact_frequencies=True)` (see `MultiTrap.setup_static_cavity_shifts`)
        [2π·Hz]
        """
        if axis in ('α', 'β', 'γ'):
            return self.multitrap.librational_Ω[self.trap_index][axis]
        if (getattr(self.multitrap, 'static_cavity_shifts', False) or getattr(self.multitrap, 'interactions', None)
                or self.trap_index in getattr(self.multitrap, 'floquet_paul_traps', ())):
            return self.multitrap.static_shifted_Ωj[self.trap_index][axis]
        return self.trap.Ωj[axis]

    @property
    def static_cavity_shift(self):
        """
        The static cavity shift of this dimension's frequency, Ω - Ωtw (zero for toy parameters
        or with static_cavity_shifts=False); for a hybrid Paul trap with exact frequencies Ωtw is the
        trap's exact (Floquet) frequency without the static cavity field (`MultiTrap.paul_floquet_Ωj`)
        [2π·Hz]
        """
        if self.multitrap.toy_params:
            return 0.0
        if self.is_libration:
            return self.Ω - self.trap.Ωθ[self.axis]
        floquet = getattr(self.multitrap, 'paul_floquet_Ωj', {})
        if self.trap_index in floquet:
            return self.Ω - floquet[self.trap_index][self.axis]
        return self.Ω - self.trap.Ωj[self.axis]

    def zero_point_fluctuation(self, axis):
        """
        Zero-point fluctuation of the position of the oscillator, per axis
        [m]
        """
        # NOTE: this deliberately uses the same Ω as Dimension.Ω (and so the mechanical
        # susceptibilities): the tweezer frequency trap.Ωj[axis], shifted by the static cavity field
        # with MultiTrap(static_cavity_shifts=True). The zero-point fluctuation sets the scale of
        # bjn, so it must use the frequency of the oscillator whose bjn appears in the Langevin
        # equations - including the static shifts, which are part of that oscillator's potential -
        # whereas the dynamical optical spring is already in those equations and must not be folded
        # in here. (The old commented-out line called self.Ω_CS, which no longer exists anywhere in
        # levisim.)
        if axis in ('α', 'β', 'γ'):
            # a libration: θ_zpf = √(ħ/(2 I Ω)) with the moment of inertia about the lab axis [rad]
            return np.sqrt(ħ/2 / (self.trap.Ij[axis] * self.mechanical_frequency(axis)))
        return np.sqrt(ħ/2 / (self.trap.bead.m * self.mechanical_frequency(axis)))

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


import os
import time
import random
import sympy
import scipy
import numpy as np

from pathlib import Path
from copy import copy
from functools import cached_property, partial

from levisim.core.constants import (
    π, ħ, kB,
    m_air,
    shot_noise,
    antinode, halfnode, node,
)
from levisim.core.units import (
    nm,
    degrees,
    Hz, kHz,
    mbar,
)

from levisim.core.logger import logger
from levisim.core.utils import load_variable_from_pickle
from levisim.core.wrappers import timer, profiler, progress_bar

from levisim.physics.tweezer import Tweezer
from levisim.physics.bead import Bead
from levisim.physics.gas import Gas
from levisim.physics.cavity import Cavity
from levisim.physics.trap import Trap

from levisim.multiphysics.dimension import Dimension
from levisim.multiphysics.quantum_linear_theory_method import QuantumLinearTheoryMethod
from levisim.multiphysics.hybridisation import (
    HybridisationTerm,
    generate_subscripts_and_labels,
    generate_all_hybridisation_functions,
    generate_determinant_and_adjugate_by_order,
)

# some debug options
TIMER = False
PROFILER = False
PROGRESS_BAR = False


class MultiTrap:
    """
    Class to calculate Quantum Linear Theory for a multi-particle cavity trap
    """
    def __init__(
        self,
        **kwargs,
    ):
        """
        Initialises a MultiTrap instance
        """
        self.num_traps = kwargs.get('num_traps', 2)
        self.Cartesian_axes_per_trap = kwargs.get('Cartesian_axes_per_trap', 2)
        self.ω_range = kwargs.get('ω_range', np.arange(-400, 400, 0.5) * kHz)

        logger.info(f'Initialising {self}: ({self.num_traps=}, {self.Cartesian_axes_per_trap=})')
        logger.info(
            f'Spectra will be calculated over'
            f' ω_range=[{self.ω_range.min() / kHz:.1f}, {self.ω_range.max() / kHz:.1f}] kHz'
            f' over {len(self.ω_range)} steps'
            f' with resolution Δω = {(self.ω_range.max() - self.ω_range.min()) / len(self.ω_range) / kHz:.1f} kHz'
        )

        self.num_total_dims = self.num_traps * self.Cartesian_axes_per_trap
        # we assume there is one light degree of freedom, with just 2 in the noise vector: a_in, a_in^dagger
        # for each dimension we have 2 entries in the noise vector: bjn,in, bjn,in^dagger
        self.num_operators = 2 * self.num_total_dims + 2

        self.subscripts, self.labels = generate_subscripts_and_labels(self.num_traps, self.Cartesian_axes_per_trap)

        # e.g. 'c': 'z1' 1 particle, xyz
        self.sub_to_label = {sub: label for sub, label in zip(self.subscripts, self.labels)}
        # e.g. 'y2': 'd' if 2 particle, xy
        self.label_to_sub = {label: sub for sub, label in self.sub_to_label.items()}

        # + 2 to account for the two conjugate light field operators
        # e.g 'x2' will be the 7th and 8th entries in the noise vector for 2 particle, xy
        self.label_to_noise_index = {label: 2*ii + 2 for ii, label in enumerate(self.labels)}
        self.noise_index_to_label = {noise_index: label for label, noise_index in self.label_to_noise_index.items()}

        # ensure the odd indices also get assigned to the correct label
        noise_indices = list(self.noise_index_to_label.keys())
        for noise_index in noise_indices:
            self.noise_index_to_label[noise_index + 1] = self.noise_index_to_label[noise_index]
        self.noise_index_to_label[0] = 'A'
        self.noise_index_to_label[1] = 'A'

        gas_kwargs = kwargs.get('gas_kwargs', {})
        self.gas = Gas(**gas_kwargs)

        cavity_kwargs = kwargs.get('cavity_kwargs', {})
        self.cavity = Cavity(**cavity_kwargs)

        self.toy_params = kwargs.get('toy_params')
        if self.toy_params:
            self.setup_toy_parameters(**self.toy_params)
        else:
            self.setup_levitated_physics(**kwargs)
            self.setup_linearised_coherent_scattering_physics(**kwargs)
            self.dimensions = {label: Dimension(self, label) for label in self.labels}

        self.G_dict = {label: dim.G for label, dim in self.dimensions.items()}
        self.G_array = np.array(list(self.G_dict.values()))

        self.quantum_factor_per_noise_operator = [
            # photon annihilation operator, <a_in(ω).a_in†(ω)> = np + 1
            shot_noise + 1,
            # photon creation operator, <a_in(ω)†.a_in(ω)> = np
            shot_noise,
        ]

        for label in self.labels:
            # NOTE: this is to match Tania's original fortran's values
            # njn = 1.4e-23 * 300 / (1.05e-34 * self.dimensions[label].Ω)

            # NOTE: there is basically no difference between approx. and exact calcs
            # since kBT >> ħΩ for T = 300K
            # njn = self.dimensions[label].noise_phonons_approx
            njn = self.dimensions[label].noise_phonons_exact

            # phonon annihilation operators, <bjn_in(ω).bjn_in†(ω)> = njn + 1
            self.quantum_factor_per_noise_operator.append(njn + 1)
            # phonon creation operators, <bjn_in†(ω).bjn_in(ω)> = njn
            self.quantum_factor_per_noise_operator.append(njn)

        # this is the quickest, because it is the direct numerical inversion of the smallest matrix
        self.QLT_method = kwargs.get('QLT_method', QuantumLinearTheoryMethod.NUMERICAL_HYBRIDISATION)
        logger.info(f'Setting {self}.QLT_method={self.QLT_method}')

        self.setup_linear_amplifier_model_matrices()

        self.all_Rjk = generate_all_hybridisation_functions(self.labels)

        # don't allow a larger max order than 9
        self.max_order = min(kwargs.get('max_order', 9), self.num_total_dims)
        self.evaluation_order = kwargs.get('evaluation_order', self.max_order)

        self.load_in_power_series = kwargs.get('load_in_power_series', False)
        self.generate_power_series = kwargs.get('generate_power_series', False)

        if self.load_in_power_series:
            N, ax = self.num_traps, self.Cartesian_axes_per_trap
            n , max_order = N * ax, self.max_order

            # NOTE: these pickle files were made using the save_power_series_to_pickle.ipynb notebook
            filename = f'{n}D_{N}x{ax}_up_to_{max_order}.pickle'
            current_file_path = Path(__file__).resolve()
            # currently this file sits in levisim/levisim/multiphysics/
            # we want to go to the base directeroy levisim/
            base_levisim_dir = current_file_path.parent.parent.parent
            power_series_file_path = (
                base_levisim_dir
                / 'notebooks'
                / 'multiphysics'
                / 'power_series'
                / 'pickle'
                / f'{filename}'
            )
            power_series_data = load_variable_from_pickle(power_series_file_path)
            self.deranged_determinant_by_order, self.deranged_adjugate_by_order = power_series_data

            if self.num_total_dims > 1:
                a, b = random.sample(self.labels, 2)
                aa, ab = f'{a}{a}', f'{a}{b}'
                for order in range(1, max_order + 1):
                    logger.trace(f'{len(self.deranged_determinant_by_order[order])} terms in determinant of order {order} of {n}')
                    logger.trace(f'{len(self.deranged_adjugate_by_order[order][aa])} terms in {a} diagonal adjugate of order {order} of {n}')
                    logger.trace(f'{len(self.deranged_adjugate_by_order[order][ab])} terms in {ab} off-diagonal adjugate of order {order} of {n}')

        elif self.generate_power_series:
            self.deranged_determinant_by_order, self.deranged_adjugate_by_order = generate_determinant_and_adjugate_by_order(
                n=self.num_total_dims,
                labels=self.labels,
                max_order=self.max_order
            )

        self.compare_Rjk_calculations = kwargs.get('compare_Rjk_calculations', False)

        logger.info(self.info_str)
        logger.info(self.cavity.info_str)
        logger.info(self.gas.info_str)
        if self.toy_params:
            logger.info(self.toy_params_info_str)
        for dimension in self.dimensions.values():
            logger.info(dimension.info_str)

    def __repr__(self):
        """
        String to represent an instance of MultiTrap
        """
        return f'MultiTrap({self.num_traps}×{self.Cartesian_axes_per_trap})'

    ################################################################################################
    # INFORMATION PRINTING FUNCTIONS
    # ----------------------------------------------------------------------------------------------
    @property
    def info_str(self):
        """
        String describing the MultiTrap instance
        """
        return (
            f'MultiTrap('
            f'\n    num_traps = {self.num_traps}'
            f'\n    Cartesian_axes_per_trap = {self.Cartesian_axes_per_trap}'
            f'\n    ω_range = [{self.ω_range.min() / kHz}, {self.ω_range.max() / kHz}] kHz'
            f'\n    num_ω_points = {len(self.ω_range)}'
            f'\n'
            f'\n    QLT_method = {self.QLT_method}'
            f'\n'
            f'\n    load_in_power_series = {self.load_in_power_series}'
            f'\n    generate_power_series = {self.generate_power_series}'
            f'\n    max_order = {self.max_order}'
            f'\n    evaluation_order = {self.evaluation_order}'
            f'\n'
            f'\n    compare_Rjk_calculations = {self.compare_Rjk_calculations}'
            f'\n'
            f'\n    id(gas) = {id(self.gas)}'
            f'\n    id(cavity) = {id(self.cavity)}'
            f'\n)'
        )

    @property
    def info(self):
        """
        Prints information on the MultiTrap instance
        """
        print(self.info_str)
        if self.toy_params:
            self.toy_params_info

    @property
    def info_per_dimension(self):
        """
        Prints information on each Dimension instance
        """
        for dimension in self.dimensions.values():
            dimension.info

    @property
    def info_per_trap(self):
        """
        Prints information on each Trap instance
        """
        if not self.toy_params:
            for trap in self.traps:
                trap.info

    @property
    def toy_params_info_str(self):
        """
        String describing the toy model parameters
        """
        if not self.toy_params:
            return

        if self.use_Γ_formula:
            Γ_str = (
                f'\n    Γ_formula = {self.toy_Γ_formula / Hz:.5e} Hz'
                f'\n    Γ_formula_from_Antonio_paper = (1 + π/8) m_gas〈v〉P / (kBT) / (ρr) = {self.toy_Γ_formula_from_Antonio_paper / Hz:.5e} Hz'
                f'\n    Γ_formula_from_Tania_paper = 15Pr²/(m〈v〉) = {self.toy_Γ_formula_from_Tania_paper / Hz:.5e} Hz'
                f'\n    Γ_formula_from_Tania_fortran = 16P/(πρr〈v〉) = {self.toy_Γ_formula_from_Tania_fortran / Hz:.5e} Hz'
            )
        else:
            Γ_str = (
                f'\n    Γ = {self.toy_Γ / kHz:.5e} kHz'
            )
        return (
            'toy_params = {'
            f'\n    Ω_model = {self.Ω_model}'
            f'{self.Ω_model_description}'
            f'\n'
            f'\n'
            f'\n    ρ = {self.toy_ρ:.1f} kg/m³'
            f'\n    r = {self.toy_r / nm:.1f} nm'
            f'\n    m = {self.toy_m:.5e} kg'
            f'\n'
            f'\n    P = {self.toy_P / mbar:.1e} mbar'
            f'\n    T = {self.toy_T:.1f} K'
            f'\n    〈v〉 = {self.toy_vth:.1f} m/s'
            f'\n'
            f'{Γ_str}'
            f'\n'
            f'\n    gx = {self.toy_gx / kHz:.5f} kHz'
            f'\n    gy = {self.toy_gy / kHz:.5f} kHz'
            f'\n    gz = {self.toy_gz / kHz:.5f} kHz'
            f'\n'
            f'\n    gxy = {self.toy_gxy / kHz:.5f} kHz'
            f'\n    gxz = {self.toy_gxz / kHz:.5f} kHz'
            f'\n    gyz = {self.toy_gyz / kHz:.5f} kHz'
            f'\n'
            '\n}'
        )

    @property
    def toy_params_info(self):
        """
        Prints information on the toy parameters
        """
        print(self.toy_params_info_str)

    ################################################################################################
    # SETUP FUNCTIONS
    # ----------------------------------------------------------------------------------------------
    def setup_levitated_physics(self, **kwargs):
        """
        Setup the physics of many beads levitated in tweezer traps
        """
        # roughly 42 degrees
        θtw = 0.731991
        θtw = 45*degrees
        self.tweezers = [
            Tweezer(θ=θtw),
            Tweezer(wx0=950*nm, wy0=1050*nm, θ=θtw, φ=90*degrees),
            Tweezer(wx0=975*nm, wy0=1025*nm, θ=θtw, φ=20*degrees),
            Tweezer(wx0=965*nm, wy0=1015*nm, θ=θtw, φ=30*degrees),
            Tweezer(wx0=940*nm, wy0=1020*nm, θ=θtw, φ=40*degrees),
            Tweezer(wx0=930*nm, wy0=1010*nm, θ=θtw, φ=50*degrees),
            Tweezer(wx0=915*nm, wy0=1055*nm, θ=θtw, φ=60*degrees),
            Tweezer(wx0=925*nm, wy0=1045*nm, θ=θtw, φ=70*degrees),
            Tweezer(wx0=935*nm, wy0=1035*nm, θ=θtw, φ=80*degrees),
        ][:self.num_traps]

        # NOTE: all mech-mech, and z-light couplings are 0 at the node
        # for the mech-mech couplings:
        # x-y couplings depend on A0, and j-z couplings on P0
        # A0, and P0 depend on the static light field
        # the static light field sums each trap's 'drive' energy into the cavity
        # each of these cavity drive energies goes as cosΦ, due to the linearised potential Vcs
        # z-light couplings also go with cosΦ, due to the linearised potential Vcs
        Φbead = node
        self.beads = [
            Bead(r=60.1*nm, Φ=Φbead),
            Bead(r=60.2*nm, Φ=Φbead),
            Bead(r=60.3*nm, Φ=Φbead),
            Bead(r=60.4*nm, Φ=Φbead),
            Bead(r=60.5*nm, Φ=Φbead),
            Bead(r=60.6*nm, Φ=Φbead),
            Bead(r=60.7*nm, Φ=Φbead),
            Bead(r=60.8*nm, Φ=Φbead),
            Bead(r=60.9*nm, Φ=Φbead),
        ][:self.num_traps]

        # each particle/trap/tweezer has its own physics!
        self.traps = [
            Trap(
                tweezer=self.tweezers[_],
                bead=self.beads[_],
                gas=self.gas,
                trap_index= _ + 1
            )
            for _ in range(self.num_traps)
        ]

    def setup_linearised_coherent_scattering_physics(self, **kwargs):
        """
        Setup the linearised properties of tweezers levitating beads
        which are all coherently scattering into the same cavity

        Linearisation splits the optical degrees of freedom into 2 parts
        α = a dominant, static, complex valued part
        a = a small, dynamic, operator valued part

        a → α + a
        a† → α* + a†
        A → A0 + (a† + a)
        P → P0 + i(a† - a)

        α = 1/2 (A0 + iP0)
        α* = 1/2 (A0 - iP0)

        To derive α (the mean static cavity photon field), consider the full Hamiltonian
        H = Hfree + Hint
        Hfree = -ħΔa†a
        NOTE: *all* tweezers are now included in a multi-particle setup!
        Ed --> Etot = Ed1 + Ed2 + ... = ∑ Ed
        Hint = ∑ Vcs = Σ -Edsinθ(cosΦ(...)...)(A ...)
        Vcs(0) = -Ed.sinθ.cosΦ.A

        Heisenberg's EoMs + dissipation
        da/dt = i[H/ħ, a] - (κ/2)a
        da/dt = (iΔ - κ/2)a + ∑ i[Vcs/ħ, a]
        static part exists at origin
        dα/dt = 0 = (iΔ - κ/2)α +  ∑ i[Vcs(0)/ħ, a]
        i[Vcs(0)/ħ, a] = -iωd.sinθ.cosΦ[A,a] = +iωd.sinθ.cosΦ, since [A,a] = -1
        dα/dt = 0 = (iΔ - κ/2)α + ∑ iωd.sinθ.cosΦ
        α = -∑ iωd.sinθ.cosΦ / (iΔ - κ/2)
        α = +∑ ωd.sinθ.cosΦ * (-Δ + iκ/2) / (Δ² + (κ/2)²)
        """

        for trap in self.traps:
            # Cavity drive energy for a particular bead in a particular tweezer trap
            # [J]
            trap.Ed = 0.5 * trap.bead.α * trap.tweezer.E0 * self.cavity.E0
            # Cavity drive frequency for a particular bead in a particular tweezer trap
            # [rad/s]
            trap.ωd = trap.Ed / ħ

        # ∑ ωd.sinθ.cosΦ
        self.ωd_total = sum(
            [trap.ωd * np.sin(trap.tweezer.θ) * np.cos(trap.bead.Φ) for trap in self.traps]
        )
        α_prefactor = self.ωd_total / (self.cavity.Δ**2 + (self.cavity.κ/2)**2)
        self.Ed_total = ħ * self.ωd_total

        # Real part of the static part of the photon annihilation operator [·]
        self.cavity.αR = α_prefactor * (-self.cavity.Δ)

        # Imaginary part of the static part of the photon annihilation operator [·]
        self.cavity.αI = α_prefactor * (self.cavity.κ/2)

        # Static part of the amplitude quadrature [·]
        # A0 = α* + α = 2αR
        self.cavity.A0 = 2 * self.cavity.αR

        # Static part of the phase quadrautre [·]
        # P0 = i(α* - α) = 2αI
        self.cavity.P0 = 2 * self.cavity.αI

        # TODO: figure out how to add optical spring corrections (per tweezer?) and
        # coherent scattering corrections (overall?) to this

    def setup_toy_parameters(self, **params):
        """ Setup parameters for the model directly """
        # __________________________________________________________________________________________
        # MECHANICAL FREQUENCIES

        # the type of model determines how the x or x,y frequencies are set
        self.Ω_model = params.get('Ω_model', 'pairs')
        logger.info(f'Setting up toy model {self.Ω_model=}')

        # NOTE: this is copying Tania's 'ParametersNpair.f90'
        if self.Ω_model == 'pairs':
            self.Ω_first = params.get('Ω_first', 300*kHz)
            self.Ω_intrapair_gap = params.get('Ω_intrapair_gap', 1.5*kHz)
            self.Ω_interpair_gap = params.get('Ω_interpair_gap', 15*kHz)

            Ωs = {}
            Ω_value = self.Ω_first
            count = 0
            for label in self.labels:
                if label[0] != 'z':
                    if not count % 2:
                        # even index is the first in a pair
                        # start with the first value, add intrapair gap for next value
                        Ωs[label] = Ω_value
                        Ω_value += self.Ω_intrapair_gap
                    else:
                        # odd index the first in a pair
                        # minus the intrapair gap
                        Ωs[label] = Ω_value
                        Ω_value -= (self.Ω_interpair_gap + self.Ω_intrapair_gap)
                    count += 1

            self.Ω_model_description = (
                f'\n    Ω_first = {self.Ω_first / kHz:.1f} kHz'
                f'\n    Ω_intrapair_gap = {self.Ω_intrapair_gap / kHz:.1f} kHz'
                f'\n    Ω_interpair_gap = {self.Ω_interpair_gap / kHz:.1f} kHz'
            )

        # NOTE: this is attempting to copy the 'Mechanical Mode Comb' paper
        # define x,y frequencies across a certain span around a midpoint
        elif self.Ω_model == 'equal spacing':
            self.Ω_mid = params.get('Ω_mid', 300*kHz)
            self.Ω_span = params.get('Ω_span', 100*kHz)

            Ω_low = self.Ω_mid - self.Ω_span / 2
            Ω_high = self.Ω_mid + self.Ω_span / 2

            num_non_z_dims = len([_ for _ in self.labels if _[0] != 'z'])
            # divide by one fewer than the number of things to fill the span
            Ω_gap = self.Ω_span / (num_non_z_dims - 1)

            Ωs = {}
            Ω_value = Ω_low
            for label in self.labels:
                if label[0] != 'z':
                    Ωs[label] = Ω_value
                    Ω_value += Ω_gap

            self.Ω_model_description = (
                f'\n    Ω_mid = {self.Ω_mid / kHz:.1f} kHz'
                f'\n    Ω_span = {self.Ω_span / kHz:.1f} kHz'
            )

        elif self.Ω_model == 'degenerate':
            self.Ω0 = params.get('Ω0', 300*kHz)
            Ωs = {label: self.Ω0 for label in self.labels if label[0] != 'z'}
            self.Ω_model_description = (
                f'\n    Ω0 = {self.Ω0 / kHz:.1f} kHz'
            )

        elif self.Ω_model == 'multiband':
            if self.Cartesian_axes_per_trap == 3:
                raise Exception(f'{self.Ω_model=} only supports x,y frequencies')

            self.Ω_model_description = (
                f'\n    Ω_centers = {self.Ω_centers / kHz:.1f} kHz'
                f'\n    Ω_spans = {self.Ω_spans / kHz:.1f} kHz'
            )
            for center, span in zip(self.Ω_centers, self.Ω_spans):
                Ω_low = center - span / 2
                Ω_high = center + span / 2

        elif self.Ω_model == 'custom':
            Ωs = params.get('toy_Ωs')
            self.Ω_model_description = ''

        if self.Cartesian_axes_per_trap == 3:
            # allow for custom z frequencies, but if not provided, then always do a factor of x freqs
            if self.Ω_model != 'custom':
                self.Ωz_to_Ωx_multiplier = params.get('Ωz_to_Ωx_multiplier', 10)
                for ii in range(self.num_traps):
                    n = ii + 1
                    xn, zn = f'x{n}', f'z{n}'
                    Ωs[zn] = Ωs[xn] / self.Ωz_to_Ωx_multiplier
                    self.Ω_model_description += f'\n    Ωz_to_Ωx_multiplier = {self.Ωz_to_Ωx_multiplier:.1f}'

        for label, Ω in Ωs.items():
            logger.trace(f'{label} frequency = {Ω / kHz:.1f} kHz')

        if self.Cartesian_axes_per_trap >= 1:
            Ωxs = np.array([Ωs[label] for label in self.labels if label[0] == 'x'])
            self.Ω_model_description += f'\n    all Ωx = {np.array2string(Ωxs / kHz, precision=2, separator=", ")} kHz'

        if self.Cartesian_axes_per_trap >= 2:
            Ωys = np.array([Ωs[label] for label in self.labels if label[0] == 'y'])
            self.Ω_model_description += f'\n    all Ωy = {np.array2string(Ωys / kHz, precision=2, separator=", ")} kHz'

        if self.Cartesian_axes_per_trap == 3:
            Ωzs = np.array([Ωs[label] for label in self.labels if label[0] == 'z'])
            self.Ω_model_description += f'\n    all Ωz = {np.array2string(Ωzs / kHz, precision=2, separator=", ")} kHz'


        # __________________________________________________________________________________________
        # BEAD
        self.toy_ρ = params.get('ρ', 2200)  # kg/m³
        self.toy_r = params.get('r', 75*nm)
        self.toy_m = params.get('m', self.toy_ρ * (4/3) * π * self.toy_r**3)

        # __________________________________________________________________________________________
        # THERMAL BATH
        self.toy_P = params.get('P', self.gas.P)
        self.toy_T = params.get('T', self.gas.T)
        self.toy_vth = params.get('vth', self.gas.vth)

        # define damping rate either with formula
        self.use_Γ_formula = params.get('use_Γ_formula', False)
        if self.use_Γ_formula:

            self.toy_Γ_formula_from_Antonio_paper = (1 + π/8) * m_air * self.toy_vth * self.toy_P / (kB * self.toy_T) / (self.toy_ρ * self.toy_r)
            self.toy_Γ_formula_from_Tania_paper = 15 * self.toy_P * self.toy_r**2 / (self.toy_m * self.toy_vth)
            self.toy_Γ_formula_from_Tania_fortran = 16 * self.toy_P / π / self.toy_vth / self.toy_ρ / self.toy_r

            # choose Tania's fortran formula for now
            self.toy_Γ_formula = self.toy_Γ_formula_from_Tania_fortran

            Γs = {label: self.toy_Γ_formula for label in self.labels}

        # or define damping rate directly
        else:
            self.toy_Γ = params.get('toy_Γ', 1e-5*Hz)
            Γs_dict = {label: self.toy_Γ for label in self.labels}
            Γs = params.get('Γs', Γs_dict)

        # __________________________________________________________________________________________
        # LINEAR COUPLINGS
        self.toy_gx = params.get('toy_gx', 0*kHz)
        self.toy_gy = params.get('toy_gy', 0*kHz)
        self.toy_gz = params.get('toy_gz', 0*kHz)

        # assuming each bead is at the node, z and mech-mech are zero
        gs_dict = {}
        for label in self.labels:
            if 'x' in label:
                gs_dict[label] = self.toy_gx
            if 'y' in label:
                gs_dict[label] = self.toy_gy
            if 'z' in label:
                gs_dict[label] = self.toy_gz
        gs = params.get('gs', gs_dict)

        self.toy_gxy = params.get('toy_gxy', 0*kHz)
        self.toy_gxz = params.get('toy_gxz', 0*kHz)
        self.toy_gyz = params.get('toy_gyz', 0*kHz)

        # __________________________________________________________________________________________
        # TWEEZER PHASES
        φ1 = params.get('φ1', 0*degrees)
        φ2 = params.get('φ2', 0*degrees)
        φs_dict = {}
        for label in self.labels:
            j, n = label[0], int(label[1:])
            if n % 2:
                φs_dict[n] = φ1
            else:
                φs_dict[n] = φ2
        φs = params.get('φs', φs_dict)

        # __________________________________________________________________________________________
        # DIMENSION DEFINITIONS
        self.dimensions = {}
        for _ in range(self.num_total_dims):
            label = self.labels[_]
            j, n = label[0], int(label[1:])
            self.dimensions[label] = Dimension(
                self,
                label,
                φ=φs[n],

                T=self.toy_T,
                m=self.toy_m,

                Ω=Ωs[label],
                Γ=Γs[label],

                g=gs[label],

                gxy=self.toy_gxy,
                gxz=self.toy_gxz,
                gyz=self.toy_gyz,
            )

    def setup_linear_amplifier_model_matrices(self):
        """
        Setup the drift matrix (A) and input noise vector (c_in)
        to help solve the quantum Langevin equations directly
        via a Linear Amplifier Model (LAM) approach

        dc(t)/dt = A.c(t) + c_in(t)

        Heisenberg (commutator with Hamiltonian)
        +
        Langevin (minus dissipation, plus input noise)
        = (
            da/dt = i[H/ħ, a] - (κ/2)a + √κ.a_in
            da†/dt = i[H/ħ, a†] - (κ/2)a† + √κ.a_in†
            ...
            dbjn/dt = i[H/ħ, bjn] - (Γjn/2)bjn + √Γjn.bjn_in
            dbjn†/dt = i[H/ħ, bjn†] - (Γjn/2)bjn† + √Γjn.bjn†_in
            ...
        )

        c = (a, a†, ..., bjn, bjn†, ...)^T = vector of annihilation and creation operators

        c_in
        = (√κ.a_in, √κ.a_in†, ..., √Γjn.bjn_in, √Γjn.bjn_in†, ...)^T
        = noise column vector times the square root of the damping rate

        A = matrix of couplings between them
        """
        N = self.num_operators
        self.identity_matrix = np.matrix(np.eye(self.num_operators))
        self.drift_matrix = np.matrix(np.zeros((N, N), dtype=np.complex128))

        Δ, κ = self.cavity.Δ, self.cavity.κ

        # the overall dampings for the nosies in the PSDs
        self.γ = np.zeros(N)
        self.γ[0] = κ
        self.γ[1] = κ

        self.drift_matrix[0, 0] = 1j*Δ - κ/2
        self.drift_matrix[1, 1] = -1j*Δ - κ/2

        for ii in range(2, N):
            label_jn = self.noise_index_to_label[ii]
            j, n = label_jn[0], label_jn[1:]

            dim_jn = self.dimensions[label_jn]
            Γj = dim_jn.Γ
            Ωj = dim_jn.Ω
            self.γ[ii] = Γj

            Gj = dim_jn.G
            Gj_star = np.conjugate(Gj)
            iG = 1j*Gj
            iG_star = 1j*Gj_star

            # fill out the 2 rows for the 2 light operators
            # with their couplings to the mechanical operators
            self.drift_matrix[0, ii] = iG  # a couples to bjn†, bjn
            self.drift_matrix[1, ii] = -iG_star  # a† couples to bjn†, bjn

            # annihilation operators will have an even index
            # creation operators will have an odd index
            # e.g.
            # ii = 2 corresponds to bx1 (annihilation)
            # ii = 3 corresponds to bx1† (creation)
            # ii % 2 = 0 for even numbers (False), 1 for odd numbers (True)
            ii_is_odd = bool(ii % 2)
            is_annihilation_operator = not ii_is_odd

            if is_annihilation_operator:
                self.drift_matrix[ii, 0] = iG_star
                self.drift_matrix[ii, 1] = iG
                self.drift_matrix[ii, ii] = -1j*Ωj - Γj/2
            else:
                self.drift_matrix[ii, 0] = -iG_star
                self.drift_matrix[ii, 1] = -iG
                self.drift_matrix[ii, ii] = 1j*Ωj - Γj/2

            # add the mech-to-mech couplings
            igxy = 1j*dim_jn.gxy
            igxz = 1j*dim_jn.gxz
            igyz = 1j*dim_jn.gyz
            if self.Cartesian_axes_per_trap == 2:
                if j == 'x':
                    # k = y
                    if is_annihilation_operator:
                        # takes bxn to byn,byn†
                        self.drift_matrix[ii, ii + 2] = igxy
                        self.drift_matrix[ii, ii + 3] = igxy
                    else:
                        # takes bxn† to byn,byn†
                        self.drift_matrix[ii, ii + 1] = -igxy
                        self.drift_matrix[ii, ii + 2] = -igxy
                elif j == 'y':
                    # k = x
                    if is_annihilation_operator:
                        # takes byn to bxn,bxn†
                        self.drift_matrix[ii, ii - 2] = igxy
                        self.drift_matrix[ii, ii - 1] = igxy
                    else:
                        # takes byn† to bxn,bxn†
                        self.drift_matrix[ii, ii - 3] = -igxy
                        self.drift_matrix[ii, ii - 2] = -igxy

            if self.Cartesian_axes_per_trap == 3:
                if j == 'x':
                    # k, l = y, z
                    if is_annihilation_operator:
                        # takes bxn to byn,byn†
                        self.drift_matrix[ii, ii + 2] = igxy
                        self.drift_matrix[ii, ii + 3] = igxy
                        # takes bxn to bzn,bzn†
                        self.drift_matrix[ii, ii + 4] = igxz
                        self.drift_matrix[ii, ii + 5] = igxz
                    else:
                        # takes bxn† to byn,byn†
                        self.drift_matrix[ii, ii + 1] = -igxy
                        self.drift_matrix[ii, ii + 2] = -igxy
                        # takes bxn† to bzn,bzn†
                        self.drift_matrix[ii, ii + 3] = -igxz
                        self.drift_matrix[ii, ii + 4] = -igxz
                elif j == 'y':
                    # k, l = x, z
                    if is_annihilation_operator:
                        # takes byn to bxn,bxn†
                        self.drift_matrix[ii, ii - 2] = igxy
                        self.drift_matrix[ii, ii - 1] = igxy
                        # takes byn to bzn,bzn†
                        self.drift_matrix[ii, ii + 2] = igyz
                        self.drift_matrix[ii, ii + 3] = igyz
                    else:
                        # takes byn† to bxn,bxn†
                        self.drift_matrix[ii, ii - 3] = -igxy
                        self.drift_matrix[ii, ii - 2] = -igxy
                        # takes byn† to bzn,bzn†
                        self.drift_matrix[ii, ii + 1] = -igyz
                        self.drift_matrix[ii, ii + 2] = -igyz
                elif j == 'z':
                    # k, l = x, y
                    if is_annihilation_operator:
                        # takes bzn to bxn,bxn†
                        self.drift_matrix[ii, ii - 4] = igxz
                        self.drift_matrix[ii, ii - 3] = igxz
                        # takes bzn to bxn,bxn†
                        self.drift_matrix[ii, ii - 2] = igyz
                        self.drift_matrix[ii, ii - 1] = igyz
                    else:
                        # takes bzn† to bxn,bxn†
                        self.drift_matrix[ii, ii - 5] = -igxz
                        self.drift_matrix[ii, ii - 4] = -igxz
                        # takes bzn† to byn,byn†
                        self.drift_matrix[ii, ii - 3] = -igyz
                        self.drift_matrix[ii, ii - 2] = -igyz

        # (√κ.a_in, √κ.a_in†, ..., √Γjn.bjn_in, √Γjn.bjn_in†, ...)
        self.c_in = np.sqrt(self.γ)

        # Noise matrix
        # N = diag(
        #       κ.<a_in(ω),a_in†(ω)>,
        #       κ<a_in†(ω),a_in(ω)>,
        #       ...,
        #       Γjn<bjn_in(ω),bjn_in†(ω)>,
        #       Γjn<bjn_in†(ω),bjn_in(ω)>,
        #       ...,
        #   )
        #   = diag(
        #       κ.(np + 1),
        #       κ.np,
        #       ...,
        #       Γjn.(njn + 1),
        #       Γjn.njn,
        #       ...,
        #   )
        # np = photons due to optical noise = 0 for shot noise
        # njn = phonons due to jn degree of freedom ~ kBT/ħΩjn (Bose-Einstein distribution)
        self.noise_matrix = np.matrix(np.diag(self.γ * self.quantum_factor_per_noise_operator))

    ################################################################################################
    # FREQUENCY FUNCTIONS
    # ----------------------------------------------------------------------------------------------
    # Linear Amplifier Model matrices
    # ----------------------------------------------------------------------------------------------
    def T_matrix(self, ω):
        """
        The transformation matrix of input noises into operator values
        dc(t)/dt = drift @ c(t) + c_in
        dc(t)/dt = -iω.c(ω)
        so
        c(ω) = T(ω) @ c_in
        where
        T(ω) = (-iω.I - drift)⁻¹
        """
        return np.linalg.inv(-1j*ω * self.identity_matrix - self.drift_matrix)
        # return scipy.linalg.inv(-1j*ω * self.identity_matrix - self.drift_matrix)

    def c_noise_vector(self, ω):
        """
        Calculates the vector of operators, without summing all the noises together
        c = (a, a†, bx1, bx1†, ...)
        c(ω) = T(ω) @ c_in
        i.e. where each operator is a vector of all the input noises
        """
        c_noise = np.zeros((self.num_operators, self.num_operators), dtype=np.complex128)
        for ii in range(self.num_operators):
            T_row = self.T_matrix(ω)[ii, :]
            # https://stackoverflow.com/questions/3337301/numpy-matrix-to-array#3338368
            T_row = np.squeeze(np.asarray(T_row))
            c_noise[ii] = T_row * self.c_in
        return c_noise

    # ----------------------------------------------------------------------------------------------
    # Derangement and Hybridisation matrices
    # ----------------------------------------------------------------------------------------------
    def derangement_matrix(self, ω):
        """
        The matrix that transforms x to X1D
        x = R @ x + X1D
        D @ x = X1D
        D = I - R

        Invert it to solve the fully hybridised position vector using the 1D noise vectors
        x = D⁻¹ @ X1D
        """
        D_mat = np.matrix(np.eye(self.num_total_dims), dtype=np.complex128)
        for jj, label_jn in enumerate(self.labels):
            for kk, label_km in enumerate(self.labels):
                if jj == kk:
                    continue
                term = HybridisationTerm(sign=-1, pairs=[(label_jn, label_km)])
                D_mat[jj, kk] = self.hybridisation_term_value(term, ω)
        return D_mat

    # calculation step functions - these are calculated for a given frequency
    # TODO: make this a partial() function? i.e. precalculate all N^2 - N of them
    def hybridisation_term_value(self, term, ω):
        """ The value of a hybridisation term at a given frequency """
        value = complex(1)
        for pair in term.pairs:
            label_jn, label_km = pair

            dim_jn = self.dimensions[label_jn]
            dim_km = self.dimensions[label_km]

            j, n = label_jn[0], label_jn[1:]
            k, m = label_km[0], label_km[1:]

            Gjn = dim_jn.G
            Gkm_star = np.conjugate(dim_km.G)

            μj = dim_jn.μm(ω)
            Mj = dim_jn.M(ω)

            ηC_m = self.cavity.ηm(ω)
            ηC_p = self.cavity.ηp(ω)

            jk = j + k

            if self.compare_Rjk_calculations:
                gjn = dim_jn.g
                gkm = dim_km.g

            # on-bead, intra-particle hybridisation
            if m == n:
                if jk in ['xy', 'yx']:
                    gxy = dim_jn.gxy
                    gxy_copy = dim_km.gxy
                    assert gxy == gxy_copy
                    gjk = gxy

                if jk in ['xz', 'zx']:
                    gxz = dim_jn.gxz
                    gxz_copy = dim_km.gxz
                    assert gxz == gxz_copy
                    gjk = gxz

                if jk in ['yz', 'zy']:
                    gyz = dim_jn.gyz
                    gyz_copy = dim_km.gyz
                    assert gyz == gyz_copy
                    gjk = gyz

                if self.compare_Rjk_calculations:
                    if jk in ['xy', 'yx']:
                        Rjk_A = (1j * μj / Mj) * (1j * ηC_m * gjn * gkm + gjk)

                    elif jk in ['xz', 'yz']:
                        Rjk_A = (1j * μj / Mj) * (-ηC_p * gjn * gkm + gjk)

                    elif jk in ['zx', 'zy']:
                        Rjk_A = (1j * μj / Mj) * (ηC_p * gjn * gkm + gjk)

            # off-bead, inter-particle hybridisation
            else:
                # no mech-mech coupling if not in the same tweezer trap
                gjk = 0

                if self.compare_Rjk_calculations:
                    φnm = dim_jn.φ - dim_km.φ

                    # amplitude-amplitude coupling or phase-phase coupling
                    if jk in ['xy', 'yx', 'xx', 'yy'] or jk in ['zz']:
                        Rjk_B1 = (1j * μj / Mj) * (1j * ηC_m * np.cos(φnm) + ηC_p * np.sin(φnm)) * gjn * gkm
                        ηC_φ_P = self.cavity.ηφ_P(φnm, ω)
                        Rjk_B2 = (1j * μj / Mj) * (-ηC_φ_P) * gjn * gkm

                        # NOTE: Tania's form is related to the phase form
                        # ηC_φ_T = iηC_φ_P
                        # iηC_φ_T = -ηC_φ_P
                        #  ηC_φ_T = self.cavity.ηφ_T(φnm, ω)
                        # Rjk_B2 = (1j * μj / Mj) * (1j * ηC_φ_T) * gjn * gkm

                    # amplitude-phase coupling
                    elif jk in ['xz', 'yz']:
                        Rjk_B1 = (1j * μj / Mj) * (1j * ηC_m * np.sin(φnm) - ηC_p * np.cos(φnm)) * gjn * gkm
                        ηC_φ_A = self.cavity.ηφ_A(φnm, ω)
                        Rjk_B2 = (1j * μj / Mj) * (-ηC_φ_A) * gjn * gkm

                    # phase-amplitude coupling
                    elif jk in ['zx', 'zy']:
                        Rjk_B1 = (1j * μj / Mj) * (-1j * ηC_m * np.sin(φnm) + ηC_p * np.cos(φnm)) * gjn * gkm
                        ηC_φ_A = self.cavity.ηφ_A(φnm, ω)
                        Rjk_B2 = (1j * μj / Mj) * (ηC_φ_A) * gjn * gkm


            # NOTE: This expression works regardless of on-bead/off-bead
            # so long as the gjk = 0 if off-bead
            Rjk = (1j * μj / Mj) * (1j * ηC_m * np.real(Gjn * Gkm_star) + ηC_p * np.imag(Gjn * Gkm_star) + gjk)

            if self.compare_Rjk_calculations:
                # 1e-16 fails the float comparision trivially
                rtol = 1e-15
                atol = 1e-15
                debug_str = (
                    f'\n{ω / kHz} kHz'
                    f'\n{label_jn=}, {label_km=}'
                    f'\ng{label_jn} = {dim_jn.g / kHz} kHz'
                    f'\ng{label_km} = {dim_km.g / kHz} kHz'
                    f'\nφ{label_jn[1:]} = {dim_jn.φ / degrees}°'
                    f'\nφ{label_km[1:]} = {dim_km.φ / degrees}°'
                    f'\nG{label_jn} = {Gjn / kHz} kHz'
                    f'\nG{label_km}* = {Gkm_star / kHz} kHz'
                )
                if m == n:
                    debug_str += f'\n{Rjk_A=}, {Rjk=}'
                    assert np.isclose(Rjk_A, Rjk, rtol=rtol, atol=atol), debug_str
                else:
                    debug_str_1 = debug_str + f'\n{Rjk_B1=}, {Rjk=}'
                    debug_str_2 = debug_str + f'\n{Rjk_B2=}, {Rjk=}'
                    debug_str_12 = debug_str + f'\n{Rjk_B1=}, {Rjk_B2=}'
                    try:
                        # assert np.isclose(Rjk_B1, Rjk_B2, rtol=rtol, atol=atol), debug_str_12
                        # assert np.isclose(Rjk_B1, Rjk, rtol=rtol, atol=atol), debug_str_1
                        # assert np.isclose(Rjk_B2, Rjk, rtol=rtol, atol=atol), debug_str_2
                        assert np.isclose(Rjk_B1, Rjk, rtol=rtol, atol=atol)
                        assert np.isclose(Rjk_B2, Rjk, rtol=rtol, atol=atol)
                        assert np.isclose(Rjk_B1, Rjk_B2, rtol=rtol, atol=atol)
                    except AssertionError as error:
                        logger.critical(debug_str + f'\n{Rjk=}, {Rjk_B1=}, {Rjk_B2=}')
                        raise error()

            value *= Rjk

        return term.sign * value

    def deranged_determinant_value(self, ω):
        """
        The value of the deranged determinant at a given frequency
        """
        value = 0
        for term in self.deranged_determinant_by_order[self.evaluation_order]:
            value += self.hybridisation_term_value(term, ω)
        return value


    def deranged_adjugate_entry_value(self, ij, ω):
        """
        The value of given adjugate entry of the derangement matrix at a given frequency
        """
        value = 0
        for term in self.deranged_adjugate_by_order[self.evaluation_order][ij]:
            value += self.hybridisation_term_value(term, ω)
        return value

    def hybridisation_matrix(self, method, ω):
        """
        The matrix that transforms x to X1D
        X = R @ X + X1D
        D @ X = X1D
        D = I - R

        Invert it to solve the fully hybridised position vector using the 1D noise vectors
        X = D⁻¹ @ X1D
        """
        # numerical method simply inverts D directly
        if method == QuantumLinearTheoryMethod.NUMERICAL_HYBRIDISATION:
            return np.linalg.inv(self.derangement_matrix(ω))
            # return scipy.linalg.inv(self.derangement_matrix(ω))

        if method == QuantumLinearTheoryMethod.POWER_SERIES_HYBRIDISATION:
            # 'power series' method inverts D symbolically
            # D⁻¹ = adj(D)/det(D)

            # det(D) = determinant of D = power series of deranged cycles ν
            # det(D) = 1 - ∑ R²ν - ∑ R³ν + ...

            # adj(D) = adjugate matrix of D = tranpose of the matrix of cofactors
            # cofactor = determinant of a minor of D
            # minor = matrix left when one row and one column have been removed
            # diagonal entry adj(D)ii = det(D\i)
            # i.e. det(D) but with row i and column i removed

            # off-diagonal entry adj(D)ij = power series of 'path-cycles' ρ
            # adj(D)ij = Rij + ∑ R²ρ + ∑ R³ρ + ...

            adjugate_matrix_ω = np.matrix(np.eye(self.num_total_dims), dtype=np.complex128)
            for jj, label_jn in enumerate(self.labels):
                for kk, label_km in enumerate(self.labels):
                    adjugate_matrix_ω[jj, kk] = self.deranged_adjugate_entry_value(label_jn + label_km, ω)
            return adjugate_matrix_ω / self.deranged_determinant_value(ω)

    def X1D_noise_vector(self, ω):
        """
        Column vector of the 1D solutions to the QLT equations of motion for each dimension, as a noise vector
        """
        X1D = np.zeros((self.num_total_dims, self.num_operators), dtype=np.complex128)
        for ii, dimension in enumerate(self.dimensions.values()):
            X1D[ii] = dimension.Q1D(ω)
        return X1D

    # ----------------------------------------------------------------------------------------------
    # QuantumLinearTheoryMethod dependent functions
    # ----------------------------------------------------------------------------------------------
    def hybridised_X(self, method, ω):
        """
        Column vector of the solutions to the QLT equations of motion for each dimension
        Each entry in the column vector is a noise vector
        """
        if (
            method == QuantumLinearTheoryMethod.LINEAR_AMPLIFIER_MODEL
            or
            method == QuantumLinearTheoryMethod.HYBRID_LINEAR_AMPLIFIER_MODEL
        ):
            hybrid_X = np.zeros((self.num_total_dims, self.num_operators), dtype=np.complex128)
            c = self.c_noise_vector(ω)
            for ii, label in enumerate(self.labels):
                # annihlation operator index
                ai = self.label_to_noise_index[label]
                # creation operator index
                ci = ai + 1
                # Qjn = bjn† + bjn
                hybrid_X[ii] = c[ci] + c[ai]
            return hybrid_X

        # hybridisation matrix methods
        return self.hybridisation_matrix(method, ω) @ self.X1D_noise_vector(ω)

    def light_noise_vectors(self, method, ω):
        """
        The light field inside the cavity calcualted via either
        - Numerical inversion of derangement matrix
        - Symbolic inversion of derangement matrix
        - Linear Amplifier Model (LAM)
        """

        if method == QuantumLinearTheoryMethod.LINEAR_AMPLIFIER_MODEL:
            # goes straight to the LAM solution for the light field operators
            # already accounts for noises a_in, a_in†
            c = self.c_noise_vector(ω)
            a_hat, a_dag = c[0], c[1]

        else:
            # a = χC (i ∑GX  + √κ a_in)
            # a† = χC* (-i ∑GX + √κ a_in†)

            # this is the light coupling multiplied by the position operator (as a noise vector)
            # i.e. ∑GX = (Gx1 * x1) + (Gy1 * y1) + ...
            # each jn=x1,y1,... is a noise vector

            χC, χC_star = self.cavity.χC(ω), np.conjugate(self.cavity.χC(-ω))
            κ = self.cavity.κ
            hybrid_X = self.hybridised_X(method, ω)
            if method == QuantumLinearTheoryMethod.HYBRID_LINEAR_AMPLIFIER_MODEL:
                # calculates the hybridised_X using LAM, and then uses the QLT Fourier solution
                # this is already an array (c_noise_vector converts it from matrix)
                a_hat = χC * 1j * self.G_array @ hybrid_X
                a_dag = χC_star * -1j * np.conjugate(self.G_array) @ hybrid_X
            else:
                a_hat = χC * 1j * self.G_array * hybrid_X
                a_dag = χC_star * -1j * np.conjugate(self.G_array) * hybrid_X
                # convert from matrix back into a 1D array, so that indexing later works ok
                # https://stackoverflow.com/questions/3337301/numpy-matrix-to-array#3338368
                a_hat = np.squeeze(np.asarray(a_hat))
                a_dag = np.squeeze(np.asarray(a_dag))

            # adds cavity-filtered noise contribution
            # χC (√κ a_in)
            a_hat[0] += χC * np.sqrt(κ)
            # χC* (√κ a_in†)
            a_dag[1] += χC_star * np.sqrt(κ)

        return a_hat, a_dag

    def output_light_noise_vectors(self, method, ω):
        """
        Account for input-output theory to determine output light field
        a_out = -√κ a + a_in
        a_out† = -√κ a† + a_in†
        """
        a_hat, a_dag = self.light_noise_vectors(method, ω)

        κ = self.cavity.κ
        a_out_hat = -np.sqrt(κ) * a_hat
        a_out_dag = -np.sqrt(κ) * a_dag

        a_out_hat[0] = a_out_hat[0] + 1
        a_out_dag[1] = a_out_dag[1] + 1

        return a_out_hat, a_out_dag

    def homodyne_noise_vector(self, method, ω):
        """
        The homodyne quadrature of the output light, defined by the homodyne angle of the cavity
        A_out(ω) = a†_out(ω) + a_out(ω)
        P_out(ω) = i(a†_out(ω) - a_out(ω))
        θ = homodyne phase angle
        S_hom(ω) = a†_out(ω).exp(iθ) + a_out(ω).exp(-iθ) = A_out(ω).cos(θ) + P_out(ω).sin(θ)
        [1/√Hz]
        """
        a_out_hat, a_out_dag = self.output_light_noise_vectors(method, ω)

        # A_out = a_out_dag + a_out_hat
        # P_out = 1j*(a_out_dag - a_out_hat)
        # return A_out * np.cos(self.cavity.θhom) + P_out * np.sin(self.cavity.θhom)
        # return a_out_dag * np.exp(1j * self.cavity.θhom) + a_out_hat * np.exp(-1j * self.cavity.θhom)

        # NOTE: Tania defines the opposite phase quadratrue P = i(a-a†)
        A_out = a_out_hat + a_out_dag
        P_out = 1j*(a_out_hat - a_out_dag)
        return A_out * np.cos(self.cavity.θhom) + P_out * np.sin(self.cavity.θhom)

    def heterodyne_noise_vector(self, method, ω):
        """
        The heterodyne quadrature of the output light
        S_het(ω) = S_aa†(ω + Ω) + S_a†a(ω - Ω)
        [1/√Hz]
        """
        # if self.cavity.Ωhet:
            # TODO: should this be a sum of CSDs?
            # a_out_hat_p, a_out_dag_p = self.output_light_noise_vectors(method, ω + self.cavity.Ωhet)
            # a_out_hat_m, a_out_dag_m = self.output_light_noise_vectors(method, ω - self.cavity.Ωhet)
            # return a_out_hat_p * a_out_dag_p + a_out_dag_m * a_out_hat_m

        a_out_hat, a_out_dag = self.output_light_noise_vectors(method, ω)
        # TODO: clarify which vector for heterodyne...
        # technically heterodyne is sum of two CSDs, S_aa†(ω + Ω) + S_a†a(ω - Ω)
        # return a_out_hat * a_out_dag + a_out_dag * a_out_hat
        return a_out_hat

    def noise_vector_to_power_spectrum(self, noise_vector):
        """
        Convert a noise vector into its power spectrum
        """
        return np.sum(self.quantum_factor_per_noise_operator * np.abs(noise_vector)**2)

    ################################################################################################
    # SPECTRAL CALCULATION FUNCTIONS
    # ----------------------------------------------------------------------------------------------
    @timer(TIMER)
    @profiler(PROFILER)
    @progress_bar('self.ω_range', PROGRESS_BAR)
    def calculate_all_hybridised_position_spectra(self, method, zpf_units=True):
        """
        Invert the derangement matrix D to get the hybridisation matrix H
        This solves the n-dimensional hybridised position vector, x
        in terms of the 1D noise vectors, X1D
        D @ X = X1D
        H = D⁻¹
        X = H @ X1D

        Calculate PSDs based on the squares of x as a noise vector
        """
        logger.info(f'Calculating all PSDs with {method=} {zpf_units=}')
        if method == QuantumLinearTheoryMethod.TNT_LINEAR_AMPLIFIER_MODEL:
            psd_array = self.position_power_spectra_array

        else:
            num_ω_points = len(self.ω_range)
            N = self.num_operators
            hybrid_X_array = np.zeros((num_ω_points, self.num_total_dims, self.num_operators), dtype=np.complex128)
            psd_array = np.zeros((num_ω_points, self.num_total_dims))

            for ii, ω in enumerate(self.ω_range):
                hybrid_X_array[ii] = self.hybridised_X(method, ω)

                # for each dimension in X, calculate the psd
                for dd, dimension in enumerate(self.dimensions.values()):
                    psd_array[ii, dd] = self.noise_vector_to_power_spectrum(hybrid_X_array[ii, dd])
                    if zpf_units:
                        psd_array[ii, dd] *= dimension.zpf**2

        psd_dict = {}
        for dd, dimension in enumerate(self.dimensions.values()):
            psd_dict[dimension.label] = psd_array[:, dd]

        return psd_array, psd_dict

    @timer(TIMER)
    @profiler(PROFILER)
    @progress_bar('self.ω_range', PROGRESS_BAR)
    def calculate_homodyne_spectrum(self, method):
        """
        Calculates homodyne spectrum of output light from the cavity, S_hh
        h = a†_out.exp(iθ) + a_out.exp(-iθ) = A_out.cos(θ) + P_out.sin(θ)
        θ = homodyne angle
        """
        logger.info(f'Calculating homodyne spectrum with {method=}')
        homodyne_spectrum = []
        for ii, ω in enumerate(self.ω_range):
            homodyne_spectrum.append(self.noise_vector_to_power_spectrum(self.homodyne_noise_vector(method, ω)))
        return np.array(homodyne_spectrum)

    @timer(TIMER)
    @profiler(PROFILER)
    @progress_bar('self.ω_range', PROGRESS_BAR)
    def calculate_heterodyne_spectrum(self, method):
        """
        Calculates the heterodyne spectrum of output light from the cavity, S_het
        S_het(ω) = S_aa†(ω + Ω) + S_a†a(ω - Ω)
        """
        logger.info(f'Calculating heterodyne spectrum with {method=}')
        heterodyne_spectrum = []
        for ii, ω in enumerate(self.ω_range):
            heterodyne_value = self.noise_vector_to_power_spectrum(self.heterodyne_noise_vector(method, ω))
            # the value is allowed to be 'broken' < 1 if using the lower order approximations...
            if method != QuantumLinearTheoryMethod.POWER_SERIES_HYBRIDISATION and self.evaluation_order != self.num_total_dims:
                assert heterodyne_value >= 1, f'{heterodyne_value=} < 1 - this should never happen since 1 is the shot noise limit!'
            heterodyne_spectrum.append(heterodyne_value)
        return np.array(heterodyne_spectrum)

    # ----------------------------------------------------------------------------------------------
    # TODO: fix these remainining functions
    # ----------------------------------------------------------------------------------------------
    def hybridised_position_operator(self, label, ω):
        """A specific hybridised position operator (a specific entry in fully hybridised x) """
        # fill out the adjugate row desired
        adj_row = np.zeros(self.num_total_dims, dtype=np.complex128)
        for ii, column_label in enumerate(self.labels):
            ij = label + column_label
            adj_row[ii] = self.deranged_adjugate_entry_value(ij, ω)

        # X = adj(D)/det(D) * X1D
        value = (adj_row @ self.X1D_noise_vector(ω)) / self.deranged_determinant_value(ω)
        return value

    @cached_property
    @timer(TIMER)
    @profiler(PROFILER)
    @progress_bar('self.ω_range', PROGRESS_BAR)
    def quantum_operator_spectral_density_array(self):
        """
        Calculate all the spectra 'at once' using 'TNT' Linear Amplifier Model (TNT LAM)
        S_cc†(ω) = T(ω) @ N @ T(ω)†
        """
        # TODO: figure out which values in this output matrix translate to which PSDs, CSDs...
        N = self.num_operators
        num_ω_points = len(self.ω_range)
        Scc_array = np.zeros((num_ω_points, N, N), dtype=np.complex128)
        for ii, ω in enumerate(self.ω_range):
            T = self.T_matrix(ω)
            Scc_array[ii] = T @ self.noise_matrix @ T.H
        return Scc_array

    @cached_property
    def position_power_spectra_array(self):
        SXX_array = np.zeros((len(self.ω_range), self.num_total_dims))
        Scc_array = self.quantum_operator_spectral_density_array
        for dd, label in enumerate(self.labels):
            # annihlation operator index
            ai = self.label_to_noise_index[label]
            # creation operator index
            ci = ai + 1
            SXX_array[:, dd] = np.real(
                Scc_array[:, ci, ci]
                + Scc_array[:, ai, ai]
                # NOTE: make sure to add the cross bb† and b†b contributions!
                # this fixes https://github.com/julianiacoponi/levisim/issues/1
                + Scc_array[:, ai, ci]
                + Scc_array[:, ci, ai]
            )
        return SXX_array

    @timer(TIMER)
    @profiler(PROFILER)
    @progress_bar('self.ω_range', PROGRESS_BAR)
    def calculate_power_spectrum(self, label, zpf_units=True):
        """ Calculates the power spectrum of a given dimension """
        logger.info(f'Calculating PSD({label})')

        j, n = label[0], label[1:]
        zpf = self.dimensions[label_A].zpf if not zpf_units else 1
        psd = []
        for ω in self.ω_range:
            value = self.noise_vector_to_power_spectrum(self.hybridised_position_operator(label, ω))
            psd.append(value)
        return np.array(psd) * zpf**2

    @timer(TIMER)
    @profiler(PROFILER)
    @progress_bar('self.ω_range', PROGRESS_BAR)
    def calculate_cross_spectrum(self, label_A, label_B, zpf_units=True):
        """ Calculates the cross correlation spectrum of the given dimensions """
        logger.info(f'Calculating CSD({label_A},{label_B})')

        j, n = label_A[0], label_A[1:]
        zpf_A = self.dimensions[label_A].zpf if not zpf_units else 1

        k, m = label_B[0], label_B[1:]
        zpf_B = self.dimensions[label_B].zpf if not zpf_units else 1

        csd = []
        for ω in self.ω_range:
            position_A = self.hybridised_position_operator(label_A, ω)
            position_B = np.conjugate(self.hybridised_position_operator(label_B, ω))
            position_AB = position_A * position_B
            value = np.sum(self.quantum_factor_per_noise_operator * position_AB)
            csd.append(value)
        return np.array(csd) * zpf_A * zpf_B

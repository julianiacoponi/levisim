import os
import time
import random
import sympy
import numpy as np
import itertools

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
    degrees,
    metre, nm,
    Hz, kHz,
    kg,
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

from levisim.multiphysics.enums.quantum_linear_theory_method import QuantumLinearTheoryMethod
from levisim.multiphysics.dimension import Dimension
from levisim.multiphysics.hybridisation import (
    HybridisationTerm,
    generate_subscripts_and_labels,
    generate_all_hybridisation_functions,
    generate_determinant_and_adjugate_by_order,
)


__all__ = [
    'MultiTrap',
    'ΩModel',
]


TIMER = True
PROFILER = False
PROGRESS_BAR = True


class ΩModel:
    """
    Model for the x and y mechanical frequencies of a toy-parameter MultiTrap

    Kinds and their parameters (defaults in KINDS), filling the x, y labels in order:
      'pairs'                  Ω_first, Ω_intrapair_gap, Ω_interpair_gap - x,y of each bead
                               split by the intrapair gap, beads split by the interpair gap,
                               descending (copies Tania's ParametersNpair.f90)
      'equal spacing: simple'  Ω_start, Ω_gap - descending in steps of Ω_gap
      'equal spacing'          Ω_mid, Ω_span - ascending, evenly filling Ω_mid ± Ω_span/2
                               (after the 'Mechanical Mode Comb' paper)
      'degenerate'             Ω0 - every frequency equal
      'multiband'              Ω_centers, Ω_spans - the labels are shared out over the bands in
                               order (any remainder to the first bands) and each band is filled
                               evenly across center ± span/2, or sits at its center if it holds one
      'custom'                 toy_Ωjn_dict - {label: Ω} given directly (z labels too if 3 axes)

    z frequencies (3 axes) are not part of the model: MultiTrap sets Ωz = Ωx / Ωz_to_Ωx_multiplier
    for every kind except 'custom'. A model can be passed to MultiTrap as toy_params['Ω_model'],
    or given the legacy way as the kind string plus its parameters in the same toy_params dict.
    [2π·Hz]
    """
    KINDS = {
        'pairs': dict(Ω_first=300*kHz, Ω_intrapair_gap=1.5*kHz, Ω_interpair_gap=15*kHz),
        'equal spacing: simple': dict(Ω_start=300*kHz, Ω_gap=5*kHz),
        'equal spacing': dict(Ω_mid=300*kHz, Ω_span=100*kHz),
        'degenerate': dict(Ω0=300*kHz),
        'multiband': dict(Ω_centers=None, Ω_spans=None),
        'custom': dict(toy_Ωjn_dict=None),
    }

    def __init__(self, kind='pairs', **params):
        """
        Initialises an ΩModel of the given `kind`, with defaults for parameters not given
        """
        if kind not in self.KINDS:
            raise ValueError(f'Unknown Ω_model {kind!r}; expected one of {list(self.KINDS)}')
        unknown = set(params) - set(self.KINDS[kind])
        if unknown:
            raise ValueError(f'Ω_model {kind!r} does not take {sorted(unknown)}; it takes {list(self.KINDS[kind])}')
        self.kind = kind
        self.params = {**self.KINDS[kind], **params}

        if kind == 'multiband':
            if self.params['Ω_centers'] is None or self.params['Ω_spans'] is None:
                raise ValueError("Ω_model 'multiband' needs both Ω_centers and Ω_spans")
            self.params['Ω_centers'] = np.atleast_1d(np.asarray(self.params['Ω_centers'], dtype=float))
            self.params['Ω_spans'] = np.atleast_1d(np.asarray(self.params['Ω_spans'], dtype=float))
            if self.params['Ω_centers'].shape != self.params['Ω_spans'].shape:
                raise ValueError('Ω_centers and Ω_spans must have the same length')
        if kind == 'custom' and self.params['toy_Ωjn_dict'] is None:
            raise ValueError("Ω_model 'custom' needs toy_Ωjn_dict")

    def __repr__(self):
        """
        String to represent an ΩModel instance
        """
        return f'ΩModel({self.kind!r})'

    @classmethod
    def from_toy_params(cls, toy_params):
        """
        Builds the model from a toy_params dict: toy_params['Ω_model'] is either an ΩModel, or a
        kind string whose parameters are read from the same dict (other keys are ignored)
        """
        model = toy_params.get('Ω_model', 'pairs')
        if isinstance(model, cls):
            return model
        if model not in cls.KINDS:
            raise ValueError(f'Unknown Ω_model {model!r}; expected one of {list(cls.KINDS)}')
        return cls(model, **{name: toy_params[name] for name in cls.KINDS[model] if name in toy_params})

    def as_toy_params(self):
        """
        The legacy toy_params form, dict(Ω_model=kind, **parameters), for splatting into toy_params
        """
        return dict(Ω_model=self.kind, **self.params)

    def frequencies(self, labels):
        """
        {label: Ω} for every x and y label in `labels` (z labels are skipped, except for 'custom',
        which returns a copy of toy_Ωjn_dict as given)
        [2π·Hz]
        """
        p = self.params
        non_z_labels = [label for label in labels if label[0] != 'z']
        Ωjn_dict = {}

        if self.kind == 'pairs':
            Ω_value = p['Ω_first']
            for count, label in enumerate(non_z_labels):
                Ωjn_dict[label] = Ω_value
                if not count % 2:
                    # even index is the first in a pair: add the intrapair gap for the next value
                    Ω_value += p['Ω_intrapair_gap']
                else:
                    # odd index is the second in a pair: step down to the next pair
                    Ω_value -= (p['Ω_interpair_gap'] + p['Ω_intrapair_gap'])

        elif self.kind == 'equal spacing: simple':
            Ω_value = p['Ω_start']
            for label in non_z_labels:
                Ωjn_dict[label] = Ω_value
                Ω_value -= p['Ω_gap']

        elif self.kind == 'equal spacing':
            Ω_low = p['Ω_mid'] - p['Ω_span'] / 2
            # divide by one fewer than the number of things to fill the span
            Ω_gap = 0 if len(non_z_labels) == 1 else p['Ω_span'] / (len(non_z_labels) - 1)
            Ω_value = Ω_low
            for label in non_z_labels:
                Ωjn_dict[label] = Ω_value
                Ω_value += Ω_gap

        elif self.kind == 'degenerate':
            Ωjn_dict = {label: p['Ω0'] for label in non_z_labels}

        elif self.kind == 'multiband':
            num_bands = len(p['Ω_centers'])
            base, remainder = divmod(len(non_z_labels), num_bands)
            band_sizes = [base + (1 if band < remainder else 0) for band in range(num_bands)]
            remaining_labels = iter(non_z_labels)
            for center, span, size in zip(p['Ω_centers'], p['Ω_spans'], band_sizes):
                for ii in range(size):
                    label = next(remaining_labels)
                    Ωjn_dict[label] = center if size == 1 else center - span / 2 + ii * span / (size - 1)

        elif self.kind == 'custom':
            Ωjn_dict = dict(p['toy_Ωjn_dict'])

        return Ωjn_dict

    @property
    def description(self):
        """
        The model's parameters, one indented line each, as printed in MultiTrap.toy_params_info_str
        """
        p = self.params
        if self.kind == 'multiband':
            return (
                f'\n    Ω_centers = {np.array2string(p["Ω_centers"] / kHz, precision=1, separator=", ")} kHz'
                f'\n    Ω_spans = {np.array2string(p["Ω_spans"] / kHz, precision=1, separator=", ")} kHz'
            )
        if self.kind == 'custom':
            return ''
        return ''.join(f'\n    {name} = {value / kHz:.1f} kHz' for name, value in p.items())

    @property
    def info_str(self):
        """
        String describing the ΩModel instance
        """
        return f'ΩModel(\n    kind = {self.kind}{self.description}\n)'

    @property
    def info(self):
        """
        Prints information on the ΩModel instance
        """
        print(self.info_str)


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
        # ------------------------------------------------------------------------------------------
        # SETUP
        # ------------------------------------------------------------------------------------------
        self.num_traps = kwargs.get('num_traps', 2)
        self.Cartesian_axes_per_trap = kwargs.get('Cartesian_axes_per_trap', 2)
        # ω_range is either one array of angular frequencies, or a list/tuple of arrays that are
        # stitched into one sorted array without duplicates - e.g. a fine grid around each
        # resonance plus a coarse background grid - kept per segment in self.ω_ranges
        ω_range = kwargs.get('ω_range', np.arange(-400, 400, 0.5) * kHz)
        if isinstance(ω_range, (list, tuple)):
            self.ω_ranges = [np.asarray(segment, dtype=float) for segment in ω_range]
            self.ω_range = np.unique(np.concatenate(self.ω_ranges))
        else:
            self.ω_ranges = [ω_range]
            self.ω_range = ω_range
        # NOTE: for a stitched range this is the step of the first segment only; it is used for
        # logging and plot labels, never in a calculation
        self.ω_step = self.ω_ranges[0][1] - self.ω_ranges[0][0] if len(self.ω_ranges[0]) > 1 else 0.0

        logger.info(f'Initialising {self}: ({self.num_traps=}, {self.Cartesian_axes_per_trap=})')
        for segment in self.ω_ranges:
            logger.info(
                f'Spectra will be calculated over'
                f' ω_range=[{segment.min() / kHz:.1f}, {segment.max() / kHz:.1f}] kHz'
                f' over {len(segment)} steps'
                f' with resolution Δω = {(segment[1] - segment[0] if len(segment) > 1 else 0.0) / kHz:.3f} kHz'
            )
        if len(self.ω_ranges) > 1:
            logger.info(f'{len(self.ω_ranges)} ω ranges stitched into {len(self.ω_range)} unique points')

        self.num_dimensions = self.num_traps * self.Cartesian_axes_per_trap
        # we assume there is one light degree of freedom, with just 2 in the noise vector: a_in, a_in^dagger
        # for each dimension we have 2 entries in the noise vector: bjn,in, bjn,in^dagger
        self.num_operators = 2 * self.num_dimensions + 2

        self.subscripts, self.labels = generate_subscripts_and_labels(self.num_traps, self.Cartesian_axes_per_trap)

        # e.g. 'c': 'z1' 1 particle, xyz
        self.sub_to_label = {sub: label for sub, label in zip(self.subscripts, self.labels)}
        # e.g. 'y2': 'd' if 2 particle, xy
        self.label_to_sub = {label: sub for sub, label in self.sub_to_label.items()}

        # e.g. 'z1': 3 if 3 particle, xyz
        self.label_to_index = {label: ii for ii, label in enumerate(self.labels)}
        # e.g. 4: 'y2' if 2 particle, xy
        self.index_to_label = {index: label for label, index in self.label_to_index.items()}

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

        self.tweezer_phase_sign_convention = kwargs.get('tweezer_phase_sign_convention', +1)
        self.tweezer_phase_sign_convention_str = '+' if self.tweezer_phase_sign_convention == 1 else '-'

        # ------------------------------------------------------------------------------------------
        # PHYSICS
        # ------------------------------------------------------------------------------------------
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

        # ------------------------------------------------------------------------------------------
        # PHASE DEPENDENCE
        # ------------------------------------------------------------------------------------------
        # NOTE: the two setup functions above should each define self.φn_dict
        self.φns = np.array(list(self.φn_dict.values()))
        self.δφnm_dict = {}

        for (n, m), (φn, φm) in zip(
            itertools.combinations(list(range(1, self.num_traps + 1)), 2),
            itertools.combinations(self.φns, 2),
        ):
            self.δφnm_dict[f'δφ{n},{m}'] = np.abs(φn - φm)

        # NOTE: could use itertools.permutations instead, and not bother with asserting the symmetry
        # assert all([
        #     self.δφnm_dict[f'φ{n}{m}'] == -self.δφnm_dict[f'φ{m}{n}']
        #     for n, m in permutations(list(range(1, len(self.φns) + 1)), 2)
        # ])

        # ------------------------------------------------------------------------------------------
        # COUPLING RATES
        # ------------------------------------------------------------------------------------------
        self.G_dict = {label: dim.G for label, dim in self.dimensions.items()}
        self.G_array = np.array(list(self.G_dict.values()))


        # ------------------------------------------------------------------------------------------
        # NOISE CORRELATIONS
        # ------------------------------------------------------------------------------------------
        self.quantum_factor_per_noise_operator = [
            # photon annihilation operator, 〈a_in(ω)·a_in†(-ω)〉= np + 1
            shot_noise + 1,
            # photon creation operator, 〈a_in(ω)†·a_in(-ω)〉= np
            shot_noise,
        ]

        for label in self.labels:
            # NOTE: this is to match Tania's original fortran's values
            # njn = 1.4e-23 * 300 / (1.05e-34 * self.dimensions[label].Ω)

            # NOTE: there is basically no difference between approx. and exact calcs
            # since kB·T >> ħΩ for T = 300K
            # njn = self.dimensions[label].noise_phonons_approx
            njn = self.dimensions[label].noise_phonons_exact

            # phonon annihilation operators, 〈bjn_in(ω)·bjn_in†(-ω)〉= njn + 1
            self.quantum_factor_per_noise_operator.append(njn + 1)
            # phonon creation operators, 〈bjn_in†(ω)·bjn_in(-ω)〉= njn
            self.quantum_factor_per_noise_operator.append(njn)

        # ------------------------------------------------------------------------------------------
        # QUANTUM LINEAR THEORY METHOD
        # ------------------------------------------------------------------------------------------
        # this is the quickest, because it is the direct numerical inversion of the smallest matrix
        # self.QLT_method = kwargs.get('QLT_method', QuantumLinearTheoryMethod.NUMERICAL_HYBRIDISATION)
        # logger.info(f'Setting {self}.QLT_method={self.QLT_method}')

        self.setup_linear_amplifier_model_matrices()

        self.all_Rjk = generate_all_hybridisation_functions(self.labels)

        # don't allow a larger max order than 9
        self.max_order = min(kwargs.get('max_order', 9), self.num_dimensions)
        self.evaluation_order = kwargs.get('evaluation_order', self.max_order)

        self.load_in_power_series = kwargs.get('load_in_power_series', False)
        self.generate_power_series = kwargs.get('generate_power_series', False)

        if self.load_in_power_series:
            N, d = self.num_traps, self.Cartesian_axes_per_trap
            n , max_order = N * d, self.max_order

            # NOTE: these pickle files were made using the save_power_series_to_pickle.ipynb notebook
            filename = f'{n}D_{N}x{d}_up_to_{max_order}.pickle'
            current_file_path = Path(__file__).resolve()
            # currently this file sits in levisim/levisim/multiphysics/
            # and the base directory is levisim/
            base_levisim_dir = current_file_path.parent.parent.parent
            power_series_file_path = (
                base_levisim_dir
                / 'notebooks'
                / 'multiphysics'
                / 'power-series'
                / 'pickle'
                / f'{filename}'
            )
            logger.trace(f'Loading in power series from {power_series_file_path=}')
            power_series_data = load_variable_from_pickle(power_series_file_path)
            self.deranged_determinant_by_order, self.deranged_adjugate_by_order = power_series_data

            if self.num_dimensions > 1:
                a, b = random.sample(self.labels, 2)
                aa, ab = f'{a}{a}', f'{a}{b}'
                for order in range(1, max_order + 1):
                    logger.trace(f'{len(self.deranged_determinant_by_order[order])} terms in determinant of order {order} of {n}')
                    logger.trace(f'{len(self.deranged_adjugate_by_order[order][aa])} terms in {a} diagonal adjugate of order {order} of {n}')
                    logger.trace(f'{len(self.deranged_adjugate_by_order[order][ab])} terms in {ab} off-diagonal adjugate of order {order} of {n}')

        elif self.generate_power_series:
            logger.trace('Generating power series from scratch')
            self.deranged_determinant_by_order, self.deranged_adjugate_by_order = generate_determinant_and_adjugate_by_order(
                n=self.num_dimensions,
                labels=self.labels,
                max_order=self.max_order
            )

        self.compare_Rjk_calculations = kwargs.get('compare_Rjk_calculations', False)

        # ------------------------------------------------------------------------------------------
        # INFO PRINTING
        # ------------------------------------------------------------------------------------------
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
            f'\n    ω_step = {self.ω_step / kHz:.3f}'
            f'\n    num_ω_points = {len(self.ω_range)}'
            # f'\n'
            # f'\n    QLT_method = {self.QLT_method}'
            f'\n'
            f'\n    tweezer_phase_sign_convention = {self.tweezer_phase_sign_convention}'
            f'\n    g_RMS = {self.g_RMS / kHz:.3f} kHz'
            f'\n    gbar = {self.gbar / kHz:.3f} kHz'
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
        if getattr(self, 'custom_gjn_dict', False):
            g_str = '\n    gx, gy, gz = N/A: custom gjn_dict, per dimension:' + ''.join(
                f'\n        g{label} = {g / kHz:.5f} kHz' for label, g in self.gjn_dict.items()
            )
        else:
            g_str = (
                f'\n    gx = {self.toy_gx / kHz:.5f} kHz'
                f'\n    gy = {self.toy_gy / kHz:.5f} kHz'
                f'\n    gz = {self.toy_gz / kHz:.5f} kHz'
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
            f'{g_str}'
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
    # COLLECTIVE PROPERTIES
    # ----------------------------------------------------------------------------------------------
    @cached_property
    def g_RMS(self):
        """
        Root-mean-square of all the dimensions' light-mech coupling rates
        [2π·Hz]
        """
        return np.sqrt(
            np.sum([dim.g**2 for dim in self.dimensions.values()]) / self.num_dimensions
        )

    @cached_property
    def gbar(self):
        """
        The total coupling rate for the phase-dependent collective modes Q,Q†
        [2π·Hz]
        """
        return np.sqrt(self.num_dimensions) * self.g_RMS

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
        self.φn_dict = {ii + 1: tweezer.φ for ii, tweezer in enumerate(self.tweezers)}

        # NOTE: all mech-mech, and z-light couplings are 0 at the node
        # for the mech-mech couplings:
        # x-y couplings depend on A0
        # x-z and y-z couplings on P0
        # A0, and P0 depend on the static light field
        # the static light field sums each trap's 'drive' energy into the cavity
        # each of these cavity drive energies goes as cos(Φ), due to the linearised potential V_CS
        # z-light couplings also go with cos(Φ), due to the linearised potential V_CS
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
        α ≡ a dominant, static, complex valued part
        a ≡ a small, dynamic, operator valued part

        a → α + a
        a† → α* + a†
        A → A0 + (a† + a)
        P → P0 + i(a† - a)

        α = 1/2 (A0 + iP0)
        α* = 1/2 (A0 - iP0)

        To derive α (the mean static cavity photon field), consider the full Hamiltonian
        H ≡ H_free + H_int
        H_free ≡ -ħΔ·a†a

        NOTE: *all* tweezers are now included in a multi-particle setup!
        Ed --> Etot = Ed1 + Ed2 + ... = ∑ Ed
        H_int ≡ ∑ V_CS = Σ -Ed·sin(θ)[cos(Φ)·(...)...]·(A ...)
        V_CS(0) = -Ed·sin(θ)·cos(Φ).A

        Heisenberg's EoMs + dissipation
        da/dt = i[H/ħ, a] - (κ/2)a
        da/dt = (iΔ - κ/2)a + ∑ i[V_CS/ħ, a]

        Static part exists at origin
        dα/dt = 0 = (iΔ - κ/2)α +  ∑ i[V_CS(0)/ħ, a]
        i[V_CS(0)/ħ, a] = -iωd·sin(θ)·cos(Φ)·[A, a] = +iωd·sin(θ)·cos(Φ), since [A, a] = -1
        dα/dt = 0 = (iΔ - κ/2)α + ∑ iωd·sin(θ)·cos(Φ)
        α = -∑ iωd·sin(θ)·cos(Φ) / (iΔ - κ/2)
        α = +∑ ωd·sin(θ)·cos(Φ) · [(-Δ + iκ/2) / (Δ² + (κ/2)²]
        """

        for trap in self.traps:
            # Cavity drive energy for a particular bead in a particular tweezer trap
            # [J]
            trap.Ed = 0.5 * trap.bead.α * trap.tweezer.E0 * self.cavity.E0
            # Cavity drive frequency for a particular bead in a particular tweezer trap
            # [2π·Hz]
            trap.ωd = trap.Ed / ħ

        # ∑ ωd·sin(θ)·cos(Φ)
        self.ωd_total = sum(
            [trap.ωd * np.sin(trap.tweezer.θ) * np.cos(trap.bead.Φ) for trap in self.traps]
        )
        α_prefactor = self.ωd_total / (self.cavity.Δ**2 + (self.cavity.κ/2)**2)
        self.Ed_total = ħ * self.ωd_total

        # Real part of the static part of the photon annihilation operator
        # [·]
        self.cavity.αR = α_prefactor * (-self.cavity.Δ)

        # Imaginary part of the static part of the photon annihilation operator
        # [·]
        self.cavity.αI = α_prefactor * (self.cavity.κ/2)

        # Static part of the amplitude quadrature
        # A0 = α* + α = 2αR
        # [·]
        self.cavity.A0 = 2 * self.cavity.αR

        # Static part of the phase quadrautre
        # P0 = i(α* - α) = 2αI
        # [·]
        self.cavity.P0 = 2 * self.cavity.αI

        # TODO: figure out how to add optical spring corrections (per tweezer?) and
        # coherent scattering corrections (overall?) to this

    def setup_toy_parameters(self, **params):
        """
        Setup 'toy' parameters to insert into the model directly, without need to calculate them
        with underlying physics
        """
        # __________________________________________________________________________________________
        # MECHANICAL FREQUENCIES

        # the type of model determines how the x or x,y frequencies are set
        # see ΩModel above, and plot_vertical_stack_varying_1_parameter.ipynb for more
        self.Ω_model_object = ΩModel.from_toy_params(params)
        self.Ω_model = self.Ω_model_object.kind
        logger.info(f'Setting up toy model {self.Ω_model=}')
        # the model's parameters stay available as attributes, e.g. self.Ω_first for 'pairs'
        for name, value in self.Ω_model_object.params.items():
            setattr(self, name, value)
        self.Ωjn_dict = self.Ω_model_object.frequencies(self.labels)
        self.Ω_model_description = self.Ω_model_object.description

        if self.Cartesian_axes_per_trap == 3:
            # allow for custom z frequencies, but if not provided, then always do a factor of x freqs
            if self.Ω_model != 'custom':
                self.Ωz_to_Ωx_multiplier = params.get('Ωz_to_Ωx_multiplier', 10)
                for ii in range(self.num_traps):
                    n = ii + 1
                    xn, zn = f'x{n}', f'z{n}'
                    self.Ωjn_dict[zn] = self.Ωjn_dict[xn] / self.Ωz_to_Ωx_multiplier
                self.Ω_model_description += f'\n    Ωz_to_Ωx_multiplier = {self.Ωz_to_Ωx_multiplier:.1f}'

        for label, Ω in self.Ωjn_dict.items():
            logger.trace(f'{label} frequency = {Ω / kHz:.1f} kHz')

        if self.Cartesian_axes_per_trap >= 1:
            Ωxs = np.array([self.Ωjn_dict[label] for label in self.labels if label[0] == 'x'])
            self.Ω_model_description += f'\n    all Ωx = {np.array2string(Ωxs / kHz, precision=2, separator=", ")} kHz'

        if self.Cartesian_axes_per_trap >= 2:
            Ωys = np.array([self.Ωjn_dict[label] for label in self.labels if label[0] == 'y'])
            self.Ω_model_description += f'\n    all Ωy = {np.array2string(Ωys / kHz, precision=2, separator=", ")} kHz'

        if self.Cartesian_axes_per_trap == 3:
            Ωzs = np.array([self.Ωjn_dict[label] for label in self.labels if label[0] == 'z'])
            self.Ω_model_description += f'\n    all Ωz = {np.array2string(Ωzs / kHz, precision=2, separator=", ")} kHz'


        # __________________________________________________________________________________________
        # BEAD
        self.toy_ρ = params.get('ρ', 2200*kg/metre**3)
        self.toy_r = params.get('r', 75*nm)
        self.toy_m = params.get('m', self.toy_ρ * (4/3) * π * self.toy_r**3)

        # __________________________________________________________________________________________
        # THERMAL BATH
        self.toy_P = params.get('P', self.gas.P)
        self.toy_T = params.get('T', self.gas.T)
        self.toy_vth = params.get('vth', self.gas.vth)

        # define damping rate either with formula
        self.use_Γ_formula = params.get('use_Γ_formula', True)
        if self.use_Γ_formula:

            self.toy_Γ_formula_from_Antonio_paper = (1 + π/8) * m_air * self.toy_vth * self.toy_P / (kB * self.toy_T) / (self.toy_ρ * self.toy_r)
            self.toy_Γ_formula_from_Tania_paper = 15 * self.toy_P * self.toy_r**2 / (self.toy_m * self.toy_vth)
            self.toy_Γ_formula_from_Tania_fortran = 16 * self.toy_P / π / self.toy_vth / self.toy_ρ / self.toy_r

            # choose Tania's fortran formula for now
            self.toy_Γ_formula = self.toy_Γ_formula_from_Tania_fortran

            self.Γjn_dict = {label: self.toy_Γ_formula for label in self.labels}

        # or define damping rate directly
        else:
            self.toy_Γ = params.get('toy_Γ', 1e-5*Hz)
            Γjn_dict = {label: self.toy_Γ for label in self.labels}
            self.Γjn_dict = params.get('Γjn_dict', Γjn_dict)

        # __________________________________________________________________________________________
        # LINEAR COUPLINGS
        # NOTE: if a custom gjn_dict is given, toy_gx/gy/gz are not used and toy_params_info_str
        # prints the per-dimension couplings instead of these
        self.custom_gjn_dict = 'gjn_dict' in params
        self.toy_gx = params.get('toy_gx', 0*kHz)
        self.toy_gy = params.get('toy_gy', 0*kHz)
        self.toy_gz = params.get('toy_gz', 0*kHz)

        # assuming each bead is at the node, z and mech-mech are zero
        gjn_dict = {}
        for label in self.labels:
            if 'x' in label:
                gjn_dict[label] = self.toy_gx
            if 'y' in label:
                gjn_dict[label] = self.toy_gy
            if 'z' in label:
                gjn_dict[label] = self.toy_gz
        self.gjn_dict = params.get('gjn_dict', gjn_dict)

        self.toy_gxy = params.get('toy_gxy', 0*kHz)
        self.toy_gxz = params.get('toy_gxz', 0*kHz)
        self.toy_gyz = params.get('toy_gyz', 0*kHz)

        # __________________________________________________________________________________________
        # TWEEZER PHASES
        φ_odd_n = params.get('φ_odd_n', 0*degrees)
        φ_even_n = params.get('φ_even_n', 0*degrees)
        φn_dict = {}
        for label in self.labels:
            j, n = label[0], int(label[1:])
            if n % 2:
                φn_dict[n] = φ_odd_n
            else:
                φn_dict[n] = φ_even_n
        self.φn_dict = params.get('φn_dict', φn_dict)

        # __________________________________________________________________________________________
        # DIMENSION DEFINITIONS
        self.dimensions = {}
        for _ in range(self.num_dimensions):
            label = self.labels[_]
            j, n = label[0], int(label[1:])
            self.dimensions[label] = Dimension(
                self,
                label,
                φ=self.φn_dict[n],

                T=self.toy_T,
                m=self.toy_m,

                Ω=self.Ωjn_dict[label],
                Γ=self.Γjn_dict[label],

                g=self.gjn_dict[label],

                gxy=self.toy_gxy,
                gxz=self.toy_gxz,
                gyz=self.toy_gyz,
            )

    def setup_linear_amplifier_model_matrices(self):
        """
        Setup the drift matrix (A) and input noise vector (c_in)
        to help solve the quantum Langevin equations directly
        via a Linear Amplifier Model (LAM) approach

        dc(t)/dt = A @ c(t) + c_in(t)

        Heisenberg (commutator with Hamiltonian)
        +
        Langevin (minus dissipation, plus input noise)
        = (
            da/dt  = i[H/ħ, a]  - (κ/2)a  + √κ·a_in
            da†/dt = i[H/ħ, a†] - (κ/2)a† + √κ·a_in†
            ...
            dbjn/dt  = i[H/ħ, bjn]  - (Γjn/2)bjn  + √Γjn·bjn_in
            dbjn†/dt = i[H/ħ, bjn†] - (Γjn/2)bjn† + √Γjn·bjn†_in
            ...
        )

        c ≡ vector of annihilation and creation operators
        c ≡ (a, a†, ..., bjn, bjn†, ...)^⊤

        c_in ≡ noise column vector times the square root of the damping rate
        c_in ≡ (√κ·a_in, √κ·a_in†, ..., √Γjn·bjn_in, √Γjn·bjn_in†, ...)^⊤

        A ≡ 'drift' matrix of couplings between them
        """
        self.identity_matrix = np.eye(self.num_operators)
        self.drift_matrix = np.zeros((self.num_operators, self.num_operators), dtype=np.complex128)

        Δ, κ = self.cavity.Δ, self.cavity.κ

        # the overall dampings for the nosies in the PSDs
        self.γ = np.zeros(self.num_operators)
        self.γ[0] = κ
        self.γ[1] = κ

        self.drift_matrix[0, 0] = 1j*Δ - κ/2
        self.drift_matrix[1, 1] = -1j*Δ - κ/2

        for ii in range(2, self.num_operators):
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

        # c_in ≡ (
        #   √κ·a_in
        #   √κ·a_in†
        #   ...
        #   √Γjn·bjn_in
        #   √Γjn·bjn_in†
        #   ...
        # )
        self.c_in = np.sqrt(self.γ)

        # Noise matrix
        # N ≡ diag(
        #       κ〈a_in(ω)[a_in(ω)]†
        #       κ〈a_in†(ω)[a_in†(ω)]†〉
        #       ...
        #       Γjn〈bjn_in(ω)[bjn_in(ω)]†〉
        #       Γjn〈bjn_in†(ω)[bjn_in†(ω)]†〉
        #       ...
        #   )
        #   = diag(
        #       κ(nc + 1)
        #       κ·nc
        #       ...
        #       Γjn(njn + 1)
        #       Γjn·njn
        #       ...
        #   )
        # nc ≡ photons due to optical noise inside cavity = 0 for shot noise
        # njn ≡ phonons due to jn degree of freedom ~ kB·T/ħ·Ωjn (Bose-Einstein distribution)
        self.noise_matrix = np.diag(self.γ * self.quantum_factor_per_noise_operator)

    ################################################################################################
    # FUNCTIONS OF ONLY FREQUENCY
    # ----------------------------------------------------------------------------------------------
    # for linear amplifier model methods
    # ----------------------------------------------------------------------------------------------
    def T_matrix(self, ω):
        """
        The transformation matrix of input noises into operator values
        c(ω) ≡ T(ω) @ c_in(ω)
        with
        T(ω) ≡ (-iω·I - A(ω))⁻¹
        (where I is the identity matrix)

        Derived from the Fourier-transformed drift matrix relation
        dc(t)/dt = A(t) @ c(t) + c_in(t)
        -iω.c(ω) = A(ω) @ c(ω) + c_in(ω)
        c(ω) = (-iω·I - A(ω))⁻¹ @ c_in(ω)
        """
        return np.linalg.inv(-1j*ω * self.identity_matrix - self.drift_matrix)

    def c_noise_vector(self, ω):
        """
        Calculates the vector of operators, without summing all the noises together
        c ≡ (a, a†, ..., bjn, bjn†, ...)^⊤
        c(ω) = T(ω) @ c_in(ω)
        i.e. where each operator is a noise vector of all the input noises
        """
        c_noise = np.zeros((self.num_operators, self.num_operators), dtype=np.complex128)
        T = self.T_matrix(ω)
        for ii in range(self.num_operators):
            T_row = T[ii, :]
            # https://stackoverflow.com/questions/3337301/numpy-matrix-to-array#3338368
            T_row = np.squeeze(np.asarray(T_row))
            c_noise[ii] = T_row * self.c_in
        return c_noise

    # ----------------------------------------------------------------------------------------------
    # for hybridisation methods
    # ----------------------------------------------------------------------------------------------
    def x1D_noise_vector(self, ω):
        """
        Column vector of the 1D solutions to the QLT equations of motion for each dimension, as a noise vector
        """
        x1D = np.zeros((self.num_dimensions, self.num_operators), dtype=np.complex128)
        for ii, dimension in enumerate(self.dimensions.values()):
            x1D[ii] = dimension.q1D(ω)
        return x1D

    def derangement_matrix(self, ω):
        """
        The matrix that transforms x to x1D
        x = R @ x + x1D
        D @ x = x1D
        D = I - R

        Invert it to solve the fully hybridised position vector using the 1D noise vectors
        x = D⁻¹ @ x1D
        """
        D_matrix = np.eye(self.num_dimensions, dtype=np.complex128)
        for jj, label_jn in enumerate(self.labels):
            for kk, label_km in enumerate(self.labels):
                if jj == kk:
                    continue
                term = HybridisationTerm(sign=-1, pairs=[(label_jn, label_km)])
                D_matrix[jj, kk] = self.hybridisation_term_value(term, ω)
        return D_matrix

    def deranged_determinant(self, ω):
        """
        The value of the deranged determinant at a given frequency
        """
        if ω == 300*kHz:
            logger.trace(f'{self.evaluation_order=}')
        R_values = self.hybridisation_function_values(ω)
        value = 0
        for term in self.deranged_determinant_by_order[self.evaluation_order]:
            value += self.hybridisation_term_value(term, ω, R_values=R_values)
        return value

    def deranged_adjugate_matrix_entry(self, ij, ω):
        """
        The value of given adjugate entry of the derangement matrix at a given frequency
        """
        if ω == 300*kHz:
            # NOTE: this dumps the entire adjugate dictionary - 104 kB of stdout for a 2x2
            # system - so it is behind logger.trace rather than an unconditional print
            logger.trace(f'{self.deranged_adjugate_by_order[self.evaluation_order]=}')
        R_values = self.hybridisation_function_values(ω)
        value = 0
        for term in self.deranged_adjugate_by_order[self.evaluation_order][ij]:
            value += self.hybridisation_term_value(term, ω, R_values=R_values)
        return value

    # calculation step functions - these are calculated for a given frequency
    def hybridisation_function_values(self, ω):
        """
        All N² - N hybridisation functions at a given frequency, as {(label_jn, label_km): Rjk}

        NOTE: the power series evaluate the same N² - N functions in every one of their terms
        (n!·n factors for the determinant of order n), so they are computed once per frequency
        here and looked up by hybridisation_term_value. Each value comes from the same
        hybridisation_function_value call as before, so the sums are bit-identical.
        [complex ·]
        """
        return {
            (label_jn, label_km): self.hybridisation_function_value(label_jn, label_km, ω)
            for label_jn in self.labels
            for label_km in self.labels
            if label_jn != label_km
        }

    def hybridisation_term_value(self, term, ω, R_values=None):
        """
        The value of a hybridisation term (a signed product of hybridisation functions) at a
        given frequency

        `R_values` is an optional table from hybridisation_function_values(ω); without it each
        hybridisation function is evaluated directly.
        [complex ·]
        """
        value = complex(1)
        for pair in term.pairs:
            if R_values is None:
                Rjk = self.hybridisation_function_value(pair[0], pair[1], ω)
            else:
                Rjk = R_values[pair]
            value *= Rjk

        return term.sign * value

    def hybridisation_function_value(self, label_jn, label_km, ω):
        """
        The hybridisation function Rjn,km(ω): the response of dimension jn to dimension km, via
        the cavity (on- and off-bead) and the mech-mech coupling gjk (on-bead only)
        [complex ·]
        """

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
                    # ηC_φ_T = self.cavity.ηφ_T(φnm, ω)
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
                except AssertionError:
                    logger.critical(debug_str + f'\n{Rjk=}, {Rjk_B1=}, {Rjk_B2=}')
                    raise

        return Rjk

    # ----------------------------------------------------------------------------------------------
    # for analytic methods
    # ----------------------------------------------------------------------------------------------
    @cached_property
    def coupling_phases(self):
        """
        Phase θjn of each dimension's complex light-mech coupling, Gjn = gjn·exp(iθjn)

        θjn = Φjn_CS + φn, where Φjn_CS = 0 for x, y (amplitude quadrature) and π/2 for z (phase
        quadrature), and φn is the phase of the tweezer trapping bead n. This does not depend on
        `tweezer_phase_sign_convention`: that sign multiplies `Dimension.φ_coupling` and multiplies
        it again inside Gjn = gjn·exp(sign·i·φ_coupling), so it cancels (see `Dimension.φ_coupling`).
        Ordered as `self.labels`.
        [rad]
        """
        return np.array([
            self.tweezer_phase_sign_convention * dim.φ_coupling
            for dim in self.dimensions.values()
        ])

    @cached_property
    def coupling_quadrature_overlap_matrix(self):
        """
        Real antisymmetric matrix I_lp = Im(Gl*·Gp) = gl·gp·sin(θp - θl) over all pairs of dimensions

        It measures how far apart (in quadrature angle) two dimensions couple to the cavity light,
        and is the only way the coupling phases enter the cavity-mediated terms that can cancel
        (P and the thermal transduction coefficients). It is evaluated from the phase difference
        so that it is EXACTLY zero for pairs that couple to the same quadrature (equal tweezer
        phases, or x and y on the same bead) - which is what keeps those terms free of
        catastrophic cancellation when one dimension is driven on resonance.
        Ordered as `self.labels` on both axes.
        [(2π·Hz)²]
        """
        θ = self.coupling_phases
        g = np.array([dim.g for dim in self.dimensions.values()])
        return np.outer(g, g) * np.sin(θ[np.newaxis, :] - θ[:, np.newaxis])

    @cached_property
    def trap_dimension_indices(self):
        """
        For each trap (in order), the indices into `self.labels` of the dimensions of its bead
        [list of int arrays]
        """
        trap_indices = [dim.trap_index for dim in self.dimensions.values()]
        return [
            np.flatnonzero(np.array(trap_indices) == trap_index)
            for trap_index in sorted(set(trap_indices))
        ]

    @cached_property
    def intra_bead_coupling_matrix(self):
        """
        Symmetric matrix of the direct mechanical couplings between dimensions of the same bead

        Entry (j, k) is gxy, gxz or gyz when dimensions j and k belong to the same bead, and zero
        otherwise (levisim has no direct mechanical coupling between different beads), matching the
        mech-mech terms of the drift matrix. Ordered as `self.labels` on both axes.
        [2π·Hz]
        """
        couplings = np.zeros((self.num_dimensions, self.num_dimensions))
        dims = list(self.dimensions.values())
        for jj, dim_j in enumerate(dims):
            for kk, dim_k in enumerate(dims):
                if jj == kk or dim_j.trap_index != dim_k.trap_index:
                    continue
                axes = ''.join(sorted(dim_j.axis + dim_k.axis))
                couplings[jj, kk] = getattr(dim_j, f'g{axes}')
        return couplings

    def analytic_kernel(self, ω):
        """
        Frequency-dependent building blocks shared by every analytic-method function at ω

        Eliminating the cavity field from the quantum Langevin equations
            a(ω)   = χC(ω)·[ i·Σ Gjn·qjn(ω) + √κ·a_in]
            a†(ω)  = χC*(-ω)·[-i·Σ Gjn*·qjn(ω) + √κ·a_in†]
            qjn(ω) = μjn(ω)·[i(Gjn*·a + Gjn·a†) + ...] + √Γjn·Q_THERMjn(ω)
        leaves, for the positions q (ordered as `self.labels`),
            (W⁻¹ - iC)·q = sources,   C = iχC·G*Gᵀ - iχC*·GG†  (rank 2: the two light channels)
        where W = (diag(1/μ) - i·g_intra)⁻¹ is the mechanical response matrix, μjn = χjn(ω) - χjn*(-ω),
        and g_intra holds the direct mechanical couplings gxy, gxz, gyz between the dimensions of the
        same bead (`intra_bead_coupling_matrix`). W is block diagonal per bead (W = diag(μ) when a
        bead's dimensions are uncoupled), symmetric, and only needs one (axes × axes) inverse per
        bead - the closed form never inverts an N×N matrix. Each thermal drive enters through
        B⁻¹ = W·diag(1/μ).
        Because C has rank 2, the cavity field only ever sees three collective response sums
            D(0) = Gᵀ·W·G*,   D(2) = Gᵀ·W·G,   D(-2) = G*ᵀ·W·G*
        and the determinant of the remaining 2×2 light-only problem is the normalisation
            M_norm = 1 + ηm·D(0) - ξC·P,   P = D(0)² - D(2)·D(-2).

        NOTE: P is NOT evaluated as D(0)² - D(2)·D(-2). When the frequency grid hits a mechanical
        resonance exactly, |gjn²·μjn| ~ 1e13 rad/s for the toy parameters, and that subtraction
        loses ~ε·|D(0)|², i.e. ~1e-9 of P. With unequal tweezer phases this error was then
        amplified ~|Mjn| ~ 1e7 in the resonant dimension's position (where the thermal drive and
        the cavity backaction nearly cancel), which was the 0.2-0.6% disagreement of the analytic
        position PSDs with LAM. The identity
            P = 2i·hᵀ·I·h̃,   h = W·G,   h̃ = W·G*,   I = `coupling_quadrature_overlap_matrix`
        involves no such subtraction, since I is exactly zero between dimensions coupling to the
        same quadrature. The same identity gives the brackets multiplying each dimension's
        thermal noise in a and a† without cancellation:
            F_Q·G + F_Q†·G*     → B⁻ᵀ·[G  - 2i·χC*(-ω)·Iᵀ·h ]
            FF_Q·G + FF_Q†·G*   → B⁻ᵀ·[G* - 2i·χC(ω)·Iᵀ·h̃]
        (with g_intra = 0, B⁻¹ = 1 and these are exactly the brackets of PRR 8).

        The result for the last ω requested is cached, since the analytic noise vectors call
        D, P and M_norm several times per frequency.
        [dict of complex arrays and scalars]
        """
        cached = getattr(self, '_analytic_kernel_cache', None)
        if cached is not None and cached[0] == ω:
            return cached[1]

        χC, χC_star = self.cavity.χC(ω), np.conjugate(self.cavity.χC(-ω))
        G = self.G_array
        G_star = np.conjugate(G)
        μ = np.array([dim.μm(ω) for dim in self.dimensions.values()])
        I = self.coupling_quadrature_overlap_matrix

        # W⁻¹ = diag(1/μ) - i·g_intra per bead, and its inverse W. W is built by inverting W⁻¹
        # rather than as (I - i·diag(μ)·g_intra)⁻¹·diag(μ): on a resonance the latter matrix has a
        # condition number ~5e6 (3×2 OFF_NODE toy system), which cost ~4e-10 in W. Beads without
        # intra-bead coupling keep W = diag(μ) exactly.
        g_intra = self.intra_bead_coupling_matrix
        W = np.diag(μ)
        # B⁻¹ = W·diag(1/μ) maps each dimension's thermal drive to the bead's uncoupled response
        B_inv = np.eye(self.num_dimensions, dtype=np.complex128)
        W_inv_blocks = []
        for indices in self.trap_dimension_indices:
            block = np.ix_(indices, indices)
            W_inv_n = np.diag(1 / μ[indices]) - 1j * g_intra[block]
            W_inv_blocks.append(W_inv_n)
            if np.any(g_intra[block]):
                W[block] = np.linalg.inv(W_inv_n)
                B_inv[block] = W[block] / μ[np.newaxis, indices]

        h, h_tilde = W @ G, W @ G_star
        D0 = G @ h_tilde
        D2 = G @ h
        Dm2 = G_star @ h_tilde
        P = 2j * h @ I @ h_tilde

        kernel = dict(
            μ=μ,
            W=W,
            W_inv_blocks=W_inv_blocks,
            h=h,
            h_tilde=h_tilde,
            D={0: D0, 2: D2, -2: Dm2},
            P=P,
            M_norm=1 + self.cavity.ηm(ω) * D0 - self.cavity.ξC(ω) * P,
            # brackets multiplying each dimension's thermal noise in a and in a† (B⁻ᵀ is the identity
            # for beads without intra-bead coupling)
            thermal_a=B_inv.T @ (G - 2j * χC_star * (I.T @ h)),
            thermal_a_dag=B_inv.T @ (G_star - 2j * χC * (I.T @ h_tilde)),
        )
        self._analytic_kernel_cache = (ω, kernel)
        return kernel

    def D(self, n, ω):
        """
        Phase-dependent collective coupling and mechanical susceptibility rate

        D(0) = Gᵀ·W·G*,   D(2) = Gᵀ·W·G,   D(-2) = G*ᵀ·W·G*   (see `analytic_kernel`)
        which without intra-bead coupling (W = diag(μ)) are D(0) = Σ gjn²·μjn and
        D(±2) = Σ gjn²·μjn·exp(±2iθjn). Only n = 0, ±2 appear in the closed form.
        [complex 2π·Hz]
        """
        if n not in (0, 2, -2):
            raise ValueError(f'D(n) is only defined for n = 0, ±2, not {n=}')
        return self.analytic_kernel(ω)['D'][n]

    def P(self, ω):
        """
        Phase-dependent function that equals zero when all phases are equal

        P = D(0)² - D(2)·D(-2), evaluated in the cancellation-free form 2i·hᵀ·I·h̃ (see
        `analytic_kernel`). Without intra-bead coupling this equals
        4·Σ_(j<k) gj²·μj·gk²·μk·sin²(θj - θk), i.e. `P_via_Pjk` and `P_via_Pj`; those
        alternative forms (and M_norm_2 ... M_norm_N_2, and the K/H/U/N functions of `Dimension`)
        do NOT include intra-bead coupling.
        [complex (2π·Hz)²]
        """
        return self.analytic_kernel(ω)['P']

    def P_via_Pj(self, ω):
        """
        Alternative definition of P as a sum of contributions from each dimension
        [complex (2π·Hz)²]
        """
        return np.sum([
            dim.P(ω)
            for dim in self.dimensions.values()
        ])

    def P_via_Pj_2(self, ω):
        """
        Alternative definition of P as a sum of contributions from each dimension
        [complex (2π·Hz)²]
        """
        return np.sum([
            dim.P_2(ω)
            for dim in self.dimensions.values()
        ])

    def P_via_Pjk(self, ω):
        """
        Alternative definition of P as a sum of contributions from each pair of dimensions

        Only pairs coupling to different quadratures contribute: pairs with equal coupling
        phases θ (x and y on the same bead, or any two x/y dimensions in tweezers of equal phase)
        give sin²(0) = 0 exactly and are skipped. Pairs on the SAME bead still contribute when
        one of them is z, since z couples to the phase quadrature (θz - θx = π/2).
        [complex (2π·Hz)²]
        """
        return np.sum([
            self.Pjk(dim_pair, ω)
            for dim_pair in itertools.combinations(self.dimensions.values(), 2)
            if dim_pair[0].φ_coupling != dim_pair[1].φ_coupling
        ])

    def Pjk(self, dim_pair, ω):
        """
        Component of P contributed by a given pair of dimensions
        [complex (2π·Hz)²]
        """
        dim_jn, dim_km = dim_pair[0], dim_pair[1]
        return 4 * (
            dim_jn.g**2
            * dim_jn.μm(ω)
            * dim_km.g**2
            * dim_km.μm(ω)
            * np.sin(dim_jn.φ_coupling - dim_km.φ_coupling)**2
        )

    def M_norm(self, ω):
        """
        Normalisation function for transducing all input noises into the collective modes Q, Q†

        M_norm = 1 + ηm·D(0) - ξC·P, i.e. the determinant of the 2×2 light-only problem.
        NOTE: this used to show a larger relative difference than the alternative forms below
        because P was evaluated as D(0)² - D(2)·D(-2), which cancels catastrophically on a
        mechanical resonance (e.g. 7.8e-10 relative error at ω = Ωy2 for the 2×2
        PHASE_DEPENDENT toy system). The alternative forms sum K(ω), which is already built from
        sin² of phase differences. With P from `analytic_kernel` the error is at machine
        precision.
        [complex]
        """
        return self.analytic_kernel(ω)['M_norm']

    def M_norm_2(self, ω):
        """
        Alternative formulation of M_norm
        [complex]
        """
        return 1 + np.sum([
            dim.g**2
            * dim.μm(ω)
            * (self.cavity.ηm(ω) - self.cavity.ξC(ω) * dim.K(ω))
            for dim in self.dimensions.values()
        ])

    def M_norm_3(self, ω):
        """
        Alternative formulation of M_norm
        [complex]
        """
        return 1 + np.sum([
            dim.g**2
            * dim.μm(ω)
            * (self.cavity.ηm(ω) + self.cavity.χC(ω) * dim.H(ω))
            for dim in self.dimensions.values()
        ])

    def M_norm_4(self, ω):
        """
        Alternative formulation of M_norm
        [complex]
        """
        return 1 + np.sum([
            dim.g**2
            * dim.μm(ω)
            * (self.cavity.ηm(ω) - np.conjugate(self.cavity.χC(-ω)) * dim.H_dag(ω))
            for dim in self.dimensions.values()
        ])

    def M_norm_5(self, ω):
        """
        Alternative formulation of M_norm
        [complex]
        """
        return 1 + np.sum([
            dim.g**2
            * dim.μm(ω)
            * (self.cavity.χC(ω) * dim.U(ω) - np.conjugate(self.cavity.χC(-ω)))
            for dim in self.dimensions.values()
        ])

    def M_norm_6(self, ω):
        """
        Alternative formulation of M_norm
        [complex]
        """
        return 1 + np.sum([
            dim.g**2
            * dim.μm(ω)
            * (self.cavity.χC(ω) - np.conjugate(self.cavity.χC(-ω)) * dim.U_dag(ω))
            for dim in self.dimensions.values()
        ])

    def M_norm_7(self, ω):
        """
        Alternative formulation of M_norm
        [complex]
        """
        return 1 + 2j * self.cavity.ξC(ω) * np.sum([
            dim.g**2
            * dim.μm(ω)
            * (self.cavity.Δ + 1j * dim.K(ω) / 2)
            for dim in self.dimensions.values()
        ])

    def M_norm_2_2(self, ω):
        """
        Alternative formulation of M_norm
        [complex]
        """
        return 1 + np.sum([
            dim.g**2
            * dim.μm(ω)
            * (self.cavity.ηm(ω) - self.cavity.ξC(ω) * dim.K_2(ω))
            for dim in self.dimensions.values()
        ])

    def M_norm_3_2(self, ω):
        """
        Alternative formulation of M_norm
        [complex]
        """
        return 1 + np.sum([
            dim.g**2
            * dim.μm(ω)
            * (self.cavity.ηm(ω) + self.cavity.χC(ω) * dim.H_2(ω))
            for dim in self.dimensions.values()
        ])

    def M_norm_4_2(self, ω):
        """
        Alternative formulation of M_norm
        [complex]
        """
        return 1 + np.sum([
            dim.g**2
            * dim.μm(ω)
            * (self.cavity.ηm(ω) - np.conjugate(self.cavity.χC(-ω)) * dim.H_dag_2(ω))
            for dim in self.dimensions.values()
        ])

    def M_norm_5_2(self, ω):
        """
        Alternative formulation of M_norm
        [complex]
        """
        return 1 + np.sum([
            dim.g**2
            * dim.μm(ω)
            * (self.cavity.χC(ω) * dim.U_2(ω) - np.conjugate(self.cavity.χC(-ω)))
            for dim in self.dimensions.values()
        ])

    def M_norm_6_2(self, ω):
        """
        Alternative formulation of M_norm
        [complex]
        """
        return 1 + np.sum([
            dim.g**2
            * dim.μm(ω)
            * (self.cavity.χC(ω) - np.conjugate(self.cavity.χC(-ω)) * dim.U_dag_2(ω))
            for dim in self.dimensions.values()
        ])

    def M_norm_7_2(self, ω):
        """
        Alternative formulation of M_norm
        [complex]
        """
        return 1 + 2j * self.cavity.ξC(ω) * np.sum([
            dim.g**2
            * dim.μm(ω)
            * (self.cavity.Δ + 1j * dim.K_2(ω) / 2)
            for dim in self.dimensions.values()
        ])

    def M_norm_N(self, option, ω):
        """
        Alternative formulation of M_norm
        [complex]
        """
        return 1 + np.sum([
            dim.g**2
            * dim.μm(ω)
            * dim.N(option, ω)
            for dim in self.dimensions.values()
        ])

    def M_norm_N_2(self, option, ω):
        """
        Alternative formulation of M_norm
        [complex]
        """
        return 1 + np.sum([
            dim.g**2
            * dim.μm(ω)
            * dim.N_2(option, ω)
            for dim in self.dimensions.values()
        ])

    def Fa(self, ω):
        """
        Transduction function to collective mode Q
        from annihilation operator of cavity input noise photons a_in
        Q ∝ Fa·a_in
        [complex 1/√(2π·Hz)]
        """
        iκg = 1j * np.sqrt(self.cavity.κ) / self.gbar
        return iκg * (self.cavity.χC(ω) * self.D(0, ω) - self.cavity.ξC(ω) * self.P(ω))

    def FFa_dag(self, ω):
        """
        Transduction function to collective mode Q†
        from creation operator of cavity input noise photons a_in†
        Q† ∝ FFa_dag·a_in†
        [complex 1/√(2π·Hz)]
        """
        iκg = 1j * np.sqrt(self.cavity.κ) / self.gbar
        return iκg * (np.conjugate(self.cavity.χC(-ω)) * self.D(0, ω) + self.cavity.ξC(ω) * self.P(ω))

    def Fa_dag(self, ω):
        """
        Transduction function to collective mode Q
        from creation operator of cavity input noise photons a_in†
        Q ∝ Fa_dag·a_in†
        [complex 1/√(2π·Hz)]
        """
        iκg = 1j * np.sqrt(self.cavity.κ) / self.gbar
        return iκg * np.conjugate(self.cavity.χC(-ω)) * self.D(2, ω)

    def FFa(self, ω):
        """
        Transduction function to collective mode Q†
        Q† ∝ FFa·a_in
        i.e. from annihilation operator of cavity input noise photons a_in
        [complex 1/√(2π·Hz)]
        """
        iκg = 1j * np.sqrt(self.cavity.κ) / self.gbar
        return iκg * self.cavity.χC(ω) * self.D(-2, ω)

    def FQ(self, ω):
        """
        Transduction function to collective mode Q
        from annihilation operator of collective thermal noise mode Q_TOT
        Q ∝ FQ·Q_TOT
        [complex]
        """
        return 1 - np.conjugate(self.cavity.χC(-ω)) * self.D(0, ω)

    def FQ_dag(self, ω):
        """
        Transduction function to collective mode Q
        from creation operator of collective thermal noise mode Q_TOT†
        Q ∝ FQ_dag·Q_TOT†
        [complex]
        """
        return np.conjugate(self.cavity.χC(-ω)) * self.D(2, ω)

    def FFQ(self, ω):
        """
        Transduction function to collective mode Q†
        from annihilation operator of collective thermal noise mode Q_TOT
        Q† ∝ FFQ·Q_TOT
        [complex]
        """
        return -self.cavity.χC(ω) * self.D(-2, ω)

    def FFQ_dag(self, ω):
        """
        Transduction function to collective mode Q†
        from creation operator of collective thermal noise mode Q_TOT†
        Q† ∝ FFQ_dag·Q_TOT†
        [complex]
        """
        return 1 + self.cavity.χC(ω) * self.D(0, ω)

    def analytic_thermal_noise_vector(self, coefficients, ω):
        """
        Noise vector Σ cjn·√Γjn·(χjn(ω)·bjn_in + χjn*(-ω)·bjn_in†) for per-dimension coefficients cjn

        `coefficients` is ordered as `self.labels`; the light-noise entries (a_in, a_in†) are
        left at zero for the caller to fill.
        [complex 1/√(2π·Hz) × units of the coefficients]
        """
        noise_vector = np.zeros(self.num_operators, dtype=np.complex128)
        for coefficient, dim in zip(coefficients, self.dimensions.values()):
            index = dim.noise_index
            noise_vector[index] = coefficient * np.sqrt(dim.Γ) * dim.χ(ω)
            noise_vector[index + 1] = coefficient * np.sqrt(dim.Γ) * np.conjugate(dim.χ(-ω))
        return noise_vector

    ################################################################################################
    # FUNCTIONS OF METHOD AND FREQUENCY
    # ----------------------------------------------------------------------------------------------
    def light_noise_vectors(self, method, ω):
        """
        The noise vectors for the annihilation and creation operators of photons
        in the shared light mode inside the cavity
        a(ω), a†(ω)
        """
        if method == QuantumLinearTheoryMethod.LINEAR_AMPLIFIER_MODEL:
            # goes straight to the LAM solution for the light field operators
            # already accounts for noises a_in, a_in†
            c = self.c_noise_vector(ω)
            a_hat, a_dag = c[0], c[1]
            return a_hat, a_dag

        if method == QuantumLinearTheoryMethod.ANALYTIC_INSIDE_CAVITY:
            gbar, κ = self.gbar, self.cavity.κ
            χC, χC_star = self.cavity.χC(ω), np.conjugate(self.cavity.χC(-ω))
            M_norm_m1 = self.M_norm(ω)**(-1)
            kernel = self.analytic_kernel(ω)

            # a ∝ χC·[F_Q·Gjn + F_Q†·Gjn*]·√Γjn·(χjn·bjn_in + χjn*·bjn_in†), etc. - with the bracket
            # evaluated in the cancellation-free form of `analytic_kernel`
            a_hat = self.analytic_thermal_noise_vector(1j * χC * M_norm_m1 * kernel['thermal_a'], ω)
            a_hat[0] = χC * (1j * gbar * M_norm_m1 * self.Fa(ω) + np.sqrt(κ))
            a_hat[1] = χC * 1j * gbar * M_norm_m1 * self.Fa_dag(ω)

            a_dag = self.analytic_thermal_noise_vector(-1j * χC_star * M_norm_m1 * kernel['thermal_a_dag'], ω)
            a_dag[0] = χC_star * -1j * gbar * M_norm_m1 * self.FFa(ω)
            a_dag[1] = χC_star * (-1j * gbar * M_norm_m1 * self.FFa_dag(ω) + np.sqrt(κ))
            return a_hat, a_dag

        if method == QuantumLinearTheoryMethod.ANALYTIC_CAVITY_OUTPUT:
            # reverse the input-output relations!
            a_out_hat, a_out_dag = self.output_light_noise_vectors(method, ω)
            a_out_hat[0] = a_out_hat[0] - 1
            a_out_dag[1] = a_out_dag[1] - 1
            κ = self.cavity.κ
            a_hat = a_out_hat / -np.sqrt(κ)
            a_dag = a_out_dag / -np.sqrt(κ)
            return a_hat, a_dag

        # NOTE: remaining possible methods are
        #   QuantumLinearTheoryMethod.HYBRID_LINEAR_AMPLIFIER_MODEL
        #   QuantumLinearTheoryMethod.NUMERICAL_HYBRIDISATION
        #   QuantumLinearTheoryMethod.POWER_SERIES_HYBRIDISATION
        # a(ω)  = χC(ω)  [ i ∑ Gjn·qjn(ω) + √κ·a_in]
        # a†(ω) = χC*(-ω)[-i ∑ Gjn·qjn(ω) + √κ·a_in†]

        # where ∑Gjn·qjn(ω) is the phase-dependent light coupling multiplied by the position operator
        # where each qjn(ω) is a noise vector entry in the hybridised_positions column vector

        χC, χC_star = self.cavity.χC(ω), np.conjugate(self.cavity.χC(-ω))
        hybrid_positions = self.hybridised_positions(method, ω)

        if self.num_dimensions == 1:
            # using @ doesn't work for single value @ 1D arrays
            a_hat = χC * 1j * self.G_array * hybrid_positions
            a_dag = χC_star * -1j * np.conjugate(self.G_array) * hybrid_positions
            # result of using * is (1, num_operators) shaped array, so flatten it
            a_hat, a_dag = a_hat.flatten(), a_dag.flatten()
        else:
            a_hat = χC * 1j * self.G_array @ hybrid_positions
            a_dag = χC_star * -1j * np.conjugate(self.G_array) @ hybrid_positions

        κ = self.cavity.κ
        # adds cavity-filtered noise contribution
        # χC(ω) (√κ·a_in)
        a_hat[0] += χC * np.sqrt(κ)
        # χC*(-ω) (√κ·a_in†)
        a_dag[1] += χC_star * np.sqrt(κ)

        return a_hat, a_dag

    def output_light_noise_vectors(self, method, ω):
        """
        The noise vectors for the annihilation and creation operators of photons leaving the cavity
        accounted for via input-output theory
        a_out(ω)  = -√κ·a(ω)  + a_in(ω)
        a_out†(ω) = -√κ·a†(ω) + a_in†(ω)
        """
        if method == QuantumLinearTheoryMethod.ANALYTIC_CAVITY_OUTPUT:
            gbar, κ = self.gbar, self.cavity.κ
            χC, χC_star = self.cavity.χC(ω), np.conjugate(self.cavity.χC(-ω))
            M_norm_m1 = self.M_norm(ω)**(-1)
            kernel = self.analytic_kernel(ω)

            a_out_hat = self.analytic_thermal_noise_vector(
                -1j * np.sqrt(κ) * χC * M_norm_m1 * kernel['thermal_a'], ω
            )
            a_out_hat[0] = -1j * gbar * np.sqrt(κ) * χC * M_norm_m1 * self.Fa(ω) + 1 - κ * χC
            a_out_hat[1] = -1j * gbar * np.sqrt(κ) * χC * M_norm_m1 * self.Fa_dag(ω)

            a_out_dag = self.analytic_thermal_noise_vector(
                1j * np.sqrt(κ) * χC_star * M_norm_m1 * kernel['thermal_a_dag'], ω
            )
            a_out_dag[0] = 1j * gbar * np.sqrt(κ) * χC_star * M_norm_m1 * self.FFa(ω)
            a_out_dag[1] = 1j * gbar * np.sqrt(κ) * χC_star * M_norm_m1 * self.FFa_dag(ω) + 1 - κ * χC_star
            return a_out_hat, a_out_dag

        a_hat, a_dag = self.light_noise_vectors(method, ω)
        κ = self.cavity.κ
        a_out_hat = -np.sqrt(κ) * a_hat
        a_out_dag = -np.sqrt(κ) * a_dag
        a_out_hat[0] = a_out_hat[0] + 1
        a_out_dag[1] = a_out_dag[1] + 1
        return a_out_hat, a_out_dag

    def hybridisation_matrix(self, method, ω):
        """
        The matrix that transforms the column vector of 1D position operators (as noise vectors)
        to the column vector of fully hybridised position operators (as noise vectors)
        i.e. the inverse of the derangement matrix
        x = D⁻¹ @ x1D
        via
        x = R @ x + x1D
        D @ x = x1D
        D = I - R
        """
        if method == QuantumLinearTheoryMethod.NUMERICAL_HYBRIDISATION:
            # this method inverts D directly/numerically
            return np.linalg.inv(self.derangement_matrix(ω))

        if method == QuantumLinearTheoryMethod.POWER_SERIES_HYBRIDISATION:
            # this method inverts D symbolically
            # D⁻¹ = adj(D)/det(D)

            # det(D) = determinant of D = power series of deranged cycles ν
            # det(D) = 1 - ∑ R²ν - ∑ R³ν + ...

            # adj(D) = adjugate matrix of D = transpose of the matrix of cofactors
            # cofactor = determinant of a minor of D
            # minor = matrix left when one row and one column have been removed
            # diagonal entry adj(D)ii = det(D\i)
            # i.e. det(D) but with row i and column i removed

            # off-diagonal entry adj(D)ij = power series of 'path-cycles' ρ
            # adj(D)ij = Rij + ∑ R²ρ + ∑ R³ρ + ...

            deranged_adjugate_matrix = np.eye(self.num_dimensions, dtype=np.complex128)
            for jj, label_jn in enumerate(self.labels):
                for kk, label_km in enumerate(self.labels):
                    deranged_adjugate_matrix[jj, kk] = self.deranged_adjugate_matrix_entry(label_jn + label_km, ω)
            return deranged_adjugate_matrix / self.deranged_determinant(ω)

    def hybridised_positions(self, method, ω):
        """
        Column vector of the solutions to the QLT equations of motion for each dimension's position
        Each entry in the column vector is a noise vector
        """
        hybrid_positions = np.zeros((self.num_dimensions, self.num_operators), dtype=np.complex128)
        if (
            method.is_linear_amplifier_model
            and
            method != QuantumLinearTheoryMethod.TNT_LINEAR_AMPLIFIER_MODEL
        ):
            c = self.c_noise_vector(ω)
            for ii, label in enumerate(self.labels):
                # annihlation operator index
                ai = self.label_to_noise_index[label]
                # creation operator index
                ci = ai + 1
                # qjn(ω) = bjn(ω)† + bjn(ω)
                hybrid_positions[ii] = c[ci] + c[ai]
            return hybrid_positions

        if method.is_hybridisation:
            hybrid_positions = self.hybridisation_matrix(method, ω) @ self.x1D_noise_vector(ω)
            # https://stackoverflow.com/questions/3337301/numpy-matrix-to-array#3338368
            return np.squeeze(np.asarray(hybrid_positions))

        if method.is_analytic:
            return self.analytic_hybridised_positions(method, ω)

    ################################################################################################
    # NOISE VECTORS FOR CALCULATING SPECTRA
    # ----------------------------------------------------------------------------------------------
    def homodyne_noise_vector(self, method, ω):
        """
        The homodyne quadrature of the output light, defined by the homodyne angle of the cavity
        A_out(ω) = a†_out(ω) + a_out(ω)
        P_out(ω) = i(a†_out(ω) - a_out(ω))
        θ = homodyne phase angle
        S_hom(ω) = a†_out(ω)·exp(iθ) + a_out(ω)·exp(-iθ) = A_out(ω)·cos(θ) + P_out(ω)·sin(θ)
        [1/√Hz]
        """
        a_out_hat, a_out_dag = self.output_light_noise_vectors(method, ω)

        # A_out = a_out_dag + a_out_hat
        # P_out = 1j*(a_out_dag - a_out_hat)
        # return A_out * np.cos(self.cavity.θhom) + P_out * np.sin(self.cavity.θhom)
        # return a_out_dag * np.exp(1j * self.cavity.θhom) + a_out_hat * np.exp(-1j * self.cavity.θhom)

        # NOTE: Tania defines the opposite phase quadratrue P = i(a - a†)
        A_out = a_out_hat + a_out_dag
        P_out = 1j * (a_out_hat - a_out_dag)
        return A_out * np.cos(self.cavity.θhom) + P_out * np.sin(self.cavity.θhom)

    def heterodyne_noise_vector(self, method, ω):
        """
        The heterodyne quadrature of the output light
        S_het(ω) = S_aa†(ω + Ω) + S_a†a(ω - Ω)
        [1/√Hz]
        """
        # if self.cavity.Ωhet:
            # TODO: clarify which vector for heterodyne...  should it be a sum of CSDs?
            # technically heterodyne is sum of two CSDs, S_aa†(ω + Ω) + S_a†a(ω - Ω)
            # return a_out_hat * a_out_dag + a_out_dag * a_out_hat
            # a_out_hat_p, a_out_dag_p = self.output_light_noise_vectors(method, ω + self.cavity.Ωhet)
            # a_out_hat_m, a_out_dag_m = self.output_light_noise_vectors(method, ω - self.cavity.Ωhet)
            # return a_out_hat_p * a_out_dag_p + a_out_dag_m * a_out_hat_m

        a_out_hat, a_out_dag = self.output_light_noise_vectors(method, ω)
        return a_out_hat

    def homodyne_and_heterodyne_noise_vectors(self, method, ω):
        """
        Returns both homodyne and heterodyne noise vectors
        (in order to only call `output_light_noise_vectors` once)
        [1/√Hz, 1/√Hz]
        """
        # if self.cavity.Ωhet:
            # TODO: clarify which vector for heterodyne...  should it be a sum of CSDs?
            # technically heterodyne is sum of two CSDs, S_aa†(ω + Ω) + S_a†a(ω - Ω)
            # return a_out_hat * a_out_dag + a_out_dag * a_out_hat
            # a_out_hat_p, a_out_dag_p = self.output_light_noise_vectors(method, ω + self.cavity.Ωhet)
            # a_out_hat_m, a_out_dag_m = self.output_light_noise_vectors(method, ω - self.cavity.Ωhet)
            # return a_out_hat_p * a_out_dag_p + a_out_dag_m * a_out_hat_m

        a_out_hat, a_out_dag = self.output_light_noise_vectors(method, ω)
        # NOTE: this uses the -1 sign convention (Tania's) implicitly
        homodyne = a_out_hat * np.exp(self.cavity.θhom) + a_out_dag * np.exp(-self.cavity.θhom)
        heterodyne = a_out_hat
        return homodyne, heterodyne

    def analytic_response_matrix(self, ω):
        """
        Closed-form mechanical response matrix H ≡ (W⁻¹ - iC)⁻¹ of all dimensions, including the cavity

        (see `analytic_kernel` for W and C). The off-bead blocks follow from the Woodbury identity
            H = W + W·Φ·W / M_norm,   Φ = iC - 4ξC·I·W·I
        but on a bead's own block that sum cancels almost completely when one of its dimensions
        is driven on resonance (|W| ~ |μ| is then up to ~1e7 times larger than the result). The
        on-bead blocks are therefore taken from the Schur complement instead,
            H_nn = (W_n⁻¹ - Φ_nn^(¬n) / M_norm^(¬n))⁻¹
        where ^(¬n) means the same quantity built from all the OTHER beads only, with
        Φ_nn^(¬n) = iC_nn - 4ξC·I_n,¬n·W_¬n·I_¬n,n. This is one small (axes × axes) inverse per
        bead, so the closed form never inverts an N×N matrix. For a single uncoupled dimension it
        reduces to H_jj = μj·M_norm^(¬j)/M_norm: the dimension's susceptibility, renormalised by
        the cavity with and without it.
        Ordered as `self.labels` on both axes.
        [complex (2π·Hz)⁻¹]
        """
        kernel = self.analytic_kernel(ω)
        χC, χC_star = self.cavity.χC(ω), np.conjugate(self.cavity.χC(-ω))
        ηm, ξC = self.cavity.ηm(ω), self.cavity.ξC(ω)
        G, G_star = self.G_array, np.conjugate(self.G_array)
        I = self.coupling_quadrature_overlap_matrix
        W, M_norm = kernel['W'], kernel['M_norm']
        h, h_tilde = kernel['h'], kernel['h_tilde']

        C = 1j * χC * np.outer(G_star, G) - 1j * χC_star * np.outer(G, G_star)
        Φ = 1j * C - 4 * ξC * (I @ W @ I)
        H = W + W @ Φ @ W / M_norm

        traps = self.trap_dimension_indices
        D0_per_trap = np.array([G[n] @ h_tilde[n] for n in traps])
        P_per_trap_pair = np.array([[2j * h[n] @ I[np.ix_(n, m)] @ h_tilde[m] for m in traps] for n in traps])
        for ii, n in enumerate(traps):
            other_traps = [jj for jj in range(len(traps)) if jj != ii]
            others = np.concatenate([np.zeros(0, dtype=int)] + [traps[jj] for jj in other_traps])
            M_norm_without_n = (
                1
                + ηm * np.sum(D0_per_trap[other_traps])
                - ξC * np.sum(P_per_trap_pair[np.ix_(other_traps, other_traps)])
            )
            Φ_nn_without_n = (
                1j * C[np.ix_(n, n)]
                - 4 * ξC * I[np.ix_(n, others)] @ W[np.ix_(others, others)] @ I[np.ix_(others, n)]
            )
            H[np.ix_(n, n)] = np.linalg.inv(kernel['W_inv_blocks'][ii] - Φ_nn_without_n / M_norm_without_n)
        return H

    def analytic_hybridisation_matrix(self, ω):
        """
        The hybridisation matrix D⁻¹ (inverse of `derangement_matrix`) from the closed form

        The derangement matrix is D = diag(μjn/Mjn)·(W⁻¹ - iC), so D⁻¹ = H·diag(Mjn/μjn) with H from
        `analytic_response_matrix`; it equals np.linalg.inv(self.derangement_matrix(ω)) without
        inverting an N×N matrix.
        Ordered as `self.labels` on both axes.
        [complex]
        """
        M_per_dimension = np.array([dim.M(ω) for dim in self.dimensions.values()])
        return self.analytic_response_matrix(ω) * (M_per_dimension / self.analytic_kernel(ω)['μ'])[np.newaxis, :]

    def analytic_hybridised_positions(self, method, ω):
        """
        All hybridised position noise vectors from the analytic closed form

        Each dimension's thermal noise reaches the positions through H·diag(1/μ) (H from
        `analytic_response_matrix`), and the cavity input noise through the light force
            q = W·i(G*·a + G·a†)   per unit a_in, a_in†
        written so that the resonant terms of D(0) and D(±2) do not have to cancel numerically.
        The positions are the same for both analytic methods (ANALYTIC_INSIDE_CAVITY and
        ANALYTIC_CAVITY_OUTPUT share this closed form). Rows are ordered as `self.labels`.
        [complex 1/√(2π·Hz)]
        """
        kernel = self.analytic_kernel(ω)
        κ = self.cavity.κ
        χC, χC_star = self.cavity.χC(ω), np.conjugate(self.cavity.χC(-ω))
        G, G_star = self.G_array, np.conjugate(self.G_array)
        I = self.coupling_quadrature_overlap_matrix
        μ, W, M_norm = kernel['μ'], kernel['W'], kernel['M_norm']

        positions = np.zeros((self.num_dimensions, self.num_operators), dtype=np.complex128)
        # cavity input noise
        positions[:, 0] = W @ (1j * χC * np.sqrt(κ) / M_norm * (G_star + 2j * χC_star * (I.T @ kernel['h_tilde'])))
        positions[:, 1] = W @ (1j * χC_star * np.sqrt(κ) / M_norm * (G + 2j * χC * (I.T @ kernel['h'])))
        # thermal noise
        thermal_response = self.analytic_response_matrix(ω) / μ[np.newaxis, :]
        for kk, dim in enumerate(self.dimensions.values()):
            index = dim.noise_index
            positions[:, index] = thermal_response[:, kk] * np.sqrt(dim.Γ) * dim.χ(ω)
            positions[:, index + 1] = thermal_response[:, kk] * np.sqrt(dim.Γ) * np.conjugate(dim.χ(-ω))
        return positions

    def hybridised_position_noise_vector(self, label, method, ω):
        """
        The noise vector for the specific hybridised position operator qjn
        qjn(ω) = bjn(ω) + bjn†(ω)
        """
        if method.is_analytic:
            return self.analytic_hybridised_positions(method, ω)[self.label_to_index[label]]

        if method == QuantumLinearTheoryMethod.POWER_SERIES_HYBRIDISATION:
            # NOTE: only fill out the adjugate row desired, ignoring the rest of the matrix
            deranged_adjugate_matrix_row = np.zeros(self.num_dimensions, dtype=np.complex128)
            for ii, column_label in enumerate(self.labels):
                ij = label + column_label
                deranged_adjugate_matrix_row[ii] = self.deranged_adjugate_matrix_entry(ij, ω)

            # x = adj(D)/det(D) * x1D
            # https://stackoverflow.com/questions/3337301/numpy-matrix-to-array#3338368
            return np.squeeze(np.asarray((deranged_adjugate_matrix_row @ self.x1D_noise_vector(ω)) / self.deranged_determinant(ω)))

        # NOTE: getting to an individual position operator via any other method sadly requires
        # inefficiently calculating all positions together and then picking the desired one
        return self.hybridised_positions(method, ω)[self.label_to_index[label]]

    def noise_vector_to_power_spectrum(self, noise_vector):
        """
        Convert a noise vector into its power spectrum
        """
        # NOTE: should be equivalent to
        # return self.noise_vectors_to_cross_spectrum(noise_vector, noise_vector)
        return np.sum(self.quantum_factor_per_noise_operator * np.abs(noise_vector)**2)

    def noise_vectors_to_cross_spectrum(self, noise_vector_A, noise_vector_B):
        """
        Convert two noise vectors into their cross spectrum
        """
        return np.sum(self.quantum_factor_per_noise_operator * noise_vector_A * np.conjugate(noise_vector_B))

    ################################################################################################
    # SPECTRA
    # ----------------------------------------------------------------------------------------------
    @timer(TIMER)
    @profiler(PROFILER)
    @progress_bar('self.ω_range', PROGRESS_BAR)
    def homodyne_spectrum(self, method):
        """
        Calculates homodyne spectrum of output light from the cavity, S_hh
        h = a†_out·exp(iθ) + a_out·exp(-iθ) = A_out·cos(θ) + P_out·sin(θ)
        θ = homodyne angle
        """
        msg = f'🧮 Calculating homodyne spectrum with {method=}'
        print(msg), logger.info(msg)

        homodyne_spectrum = []
        for ii, ω in enumerate(self.ω_range):
            homodyne_spectrum.append(self.noise_vector_to_power_spectrum(self.homodyne_noise_vector(method, ω)))
        return np.array(homodyne_spectrum)

    @timer(TIMER)
    @profiler(PROFILER)
    @progress_bar('self.ω_range', PROGRESS_BAR)
    def heterodyne_spectrum(self, method):
        """
        Calculates the heterodyne spectrum of output light from the cavity, S_het
        S_het(ω) = S_aa†(ω + Ω) + S_a†a(ω - Ω)
        """
        msg = f'🧮 Calculating heterodyne spectrum with {method=}'
        print(msg), logger.info(msg)

        heterodyne_spectrum = []
        for ii, ω in enumerate(self.ω_range):
            heterodyne_value = self.noise_vector_to_power_spectrum(self.heterodyne_noise_vector(method, ω))
            # the value is allowed to be 'broken' < 1 if using the lower order approximations...
            # if method != QuantumLinearTheoryMethod.POWER_SERIES_HYBRIDISATION and self.evaluation_order != self.num_dimensions:
            #     assert heterodyne_value >= 1, f'{heterodyne_value=} < 1 - this should never happen since 1 is the shot noise limit!'
            heterodyne_spectrum.append(heterodyne_value)
        return np.array(heterodyne_spectrum)

    @timer(TIMER)
    @profiler(PROFILER)
    @progress_bar('self.ω_range', PROGRESS_BAR)
    def homodyne_and_heterodyne_spectra(self, method):
        """
        Calculates the heterodyne spectrum of output light from the cavity, S_het
        S_het(ω) = S_aa†(ω + Ω) + S_a†a(ω - Ω)
        """
        msg = f'🧮 Calculating both homodyne and heterodyne spectra with {method=}'
        print(msg), logger.info(msg)

        homodyne_spectrum = []
        heterodyne_spectrum = []
        for ii, ω in enumerate(self.ω_range):
            homodyne_noise_vector_value, heterodyne_noise_vector_value = self.homodyne_and_heterodyne_noise_vectors(method, ω)

            homodyne_value = self.noise_vector_to_power_spectrum(homodyne_noise_vector_value)
            heterodyne_value = self.noise_vector_to_power_spectrum(heterodyne_noise_vector_value)

            # the value is allowed to be 'broken' < 1 if using the lower order approximations...
            if method != QuantumLinearTheoryMethod.POWER_SERIES_HYBRIDISATION and self.evaluation_order != self.num_dimensions:
                assert heterodyne_value >= 1, f'{heterodyne_value=} < 1 - this should never happen since 1 is the shot noise limit!'

            homodyne_spectrum.append(homodyne_value)
            heterodyne_spectrum.append(heterodyne_value)

        return np.array(homodyne_spectrum), np.array(heterodyne_spectrum)


    @timer(TIMER)
    @profiler(PROFILER)
    @progress_bar('self.ω_range', PROGRESS_BAR)
    def all_position_power_spectra(self, method, zpf_scaled=True):
        """
        Invert the derangement matrix D to get the hybridisation matrix H
        This solves for the multi-dimensional hybridised position vector, x
        in terms of the 1D noise vectors, x1D
        D @ x = x1D
        H ≡ D⁻¹
        x = H @ x1D
        """
        msg = f'🧮 Calculating all position PSDs with {method=} ({zpf_scaled=})'
        print(msg), logger.info(msg)

        if method == QuantumLinearTheoryMethod.TNT_LINEAR_AMPLIFIER_MODEL:
            # NOTE: this is the only valid place for the TNT method
            psd_array = self.position_power_spectra_array

        else:
            num_ω_points = len(self.ω_range)
            psd_array = np.zeros((num_ω_points, self.num_dimensions))
            hybrid_positions_array = np.zeros((num_ω_points, self.num_dimensions, self.num_operators), dtype=np.complex128)
            for ii, ω in enumerate(self.ω_range):
                hybrid_positions_array[ii] = self.hybridised_positions(method, ω)
                # for each dimension in x, calculate the psd
                for dd, dimension in enumerate(self.dimensions.values()):
                    psd_array[ii, dd] = self.noise_vector_to_power_spectrum(hybrid_positions_array[ii, dd])

        psd_dict = {}
        for dd, dimension in enumerate(self.dimensions.values()):
            if not zpf_scaled:
                # zpf_scaled means the position operator remains dimensionless
                # i.e. rescaled by the zpf, or in units of the zpf
                psd_array[:, dd] *= dimension.zpf**2
            psd_dict[dimension.label] = psd_array[:, dd]

        return psd_array, psd_dict


    @timer(TIMER)
    @profiler(PROFILER)
    @progress_bar('self.ω_range', PROGRESS_BAR)
    def position_power_spectrum(self, label, method, zpf_scaled=True):
        """
        Calculates the power spectrum of a given dimension
        """
        msg = f'🧮 Calculating {label} position PSD via {method=} ({zpf_scaled=})'
        print(msg), logger.info(msg)

        j, n = label[0], int(label[1:])
        zpf = 1 if zpf_scaled else self.dimensions[label].zpf
        psd = []
        for ω in self.ω_range:
            value = self.noise_vector_to_power_spectrum(self.hybridised_position_noise_vector(label, method, ω))
            psd.append(value)
        return np.array(psd) * zpf**2


    @timer(TIMER)
    @profiler(PROFILER)
    @progress_bar('self.ω_range', PROGRESS_BAR)
    def position_cross_spectrum(self, label_A, label_B, method, zpf_scaled=True):
        """
        Calculates the cross correlation spectrum of the given dimensions
        """
        msg = f'🧮 Calculating {label_A},{label_B} position CSD via {method=} ({zpf_scaled=})'
        print(msg), logger.info(msg)

        j, n = label_A[0], int(label_A[1:])
        zpf_A = 1 if zpf_scaled else self.dimensions[label_A].zpf

        k, m = label_B[0], int(label_B[1:])
        zpf_B = 1 if zpf_scaled else self.dimensions[label_B].zpf

        csd = []
        for ω in self.ω_range:
            if method.is_analytic or method == QuantumLinearTheoryMethod.NUMERICAL_HYBRIDISATION:
                position_A = self.hybridised_position_noise_vector(label_A, method, ω)
                position_B = self.hybridised_position_noise_vector(label_B, method, ω)
            else:
                hybrid_positions = self.hybridised_positions(method, ω)
                position_A = hybrid_positions[self.label_to_index[label_A]]
                position_B = hybrid_positions[self.label_to_index[label_B]]
            value = self.noise_vectors_to_cross_spectrum(position_A, position_B)
            csd.append(value)
        return np.array(csd) * zpf_A * zpf_B


    @cached_property
    @timer(TIMER)
    @profiler(PROFILER)
    @progress_bar('self.ω_range', PROGRESS_BAR)
    def quantum_operator_spectral_density_array(self):
        """
        Calculate all spectra 'at once' using 'TNT' Linear Amplifier Model (TNT LAM)
        Scc†(ω) = T(ω) @ N @ T(ω)†
        """
        msg = '🧮 Calculating the quantum operator spectral density array Scc† via TNT method'
        print(msg), logger.info(msg)

        N_ops = self.num_operators
        Scc_array = np.zeros((len(self.ω_range), N_ops, N_ops), dtype=np.complex128)
        for ii, ω in enumerate(self.ω_range):
            T = self.T_matrix(ω)
            Scc_array[ii] = T @ self.noise_matrix @ T.conj().T
        return Scc_array

    @cached_property
    def position_power_spectra_array(self):
        """
        Calculates an array of the PSDs of all dimensions' zpf-scaled position operators
        (see position_cross_spectra_array for all the cross spectra)
        """
        msg = '🧮 Calculating array of all (zpf-scaled) position power spectra'
        print(msg), logger.info(msg)

        Sxx_array = np.zeros((len(self.ω_range), self.num_dimensions))
        Scc_array = self.quantum_operator_spectral_density_array
        for dd, label in enumerate(self.labels):
            # annihlation operator index
            ai = self.label_to_noise_index[label]
            # creation operator index
            ci = ai + 1
            # q² ~ (b + b†)² = (b + b†)(b + b†) = b·b + b†·b† + b·b† + b†·b
            Sxx_array[:, dd] = np.real(
                Scc_array[:, ci, ci]
                + Scc_array[:, ai, ai]
                # NOTE: make sure to add the cross b·b† and b†·b contributions!
                # this fixes https://github.com/julianiacoponi/levisim/issues/1
                + Scc_array[:, ai, ci]
                + Scc_array[:, ci, ai]
            )
        return Sxx_array

    @cached_property
    def position_cross_spectra_array(self):
        """
        Calculates an array of the cross spectra of all pairs of dimensions' zpf-scaled position
        operators, from the same TNT LAM array as position_power_spectra_array

        S_qjqk(ω)[ii, jj, kk] = Σ Scc†(ω) over the 2 × 2 annihilation/creation index pairs of
        dimensions jj and kk, since qjn = bjn + bjn†. The diagonal is position_power_spectra_array
        (real), and S_qkqj = S_qjqk*. Same convention as position_cross_spectrum(label_A, label_B),
        i.e. 〈qA(ω)·qB(ω)†〉.
        [complex ·], shape (len(ω_range), num_dimensions, num_dimensions)
        """
        msg = '🧮 Calculating array of all (zpf-scaled) position cross spectra'
        print(msg), logger.info(msg)

        Scc_array = self.quantum_operator_spectral_density_array
        indices = [(self.label_to_noise_index[label], self.label_to_noise_index[label] + 1) for label in self.labels]
        Sqq_array = np.zeros((len(self.ω_range), self.num_dimensions, self.num_dimensions), dtype=np.complex128)
        for jj, (aj, cj) in enumerate(indices):
            for kk, (ak, ck) in enumerate(indices):
                Sqq_array[:, jj, kk] = (
                    Scc_array[:, aj, ak]
                    + Scc_array[:, aj, ck]
                    + Scc_array[:, cj, ak]
                    + Scc_array[:, cj, ck]
                )
        return Sqq_array

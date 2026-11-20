"""
Shared state and method dispatch of `MultiTrap`

`MultiTrapBase` builds the system (trap physics or toy parameters, the dimensions, the noise
correlations), holds the information printing functions, and owns the spectrum entry points
(`homodyne_spectrum`, `all_position_power_spectra`, ...) together with the functions that dispatch
a `QuantumLinearTheoryMethod` to the mixin that implements it:

- `light_noise_vectors` and `output_light_noise_vectors` (the cavity light, inside and outside)
- `hybridised_positions` and `hybridised_position_noise_vector` (the mechanical positions)

The dispatch lives here rather than in any one mixin because, for the methods that solve for the
positions first (HYBRID_LINEAR_AMPLIFIER_MODEL, NUMERICAL_HYBRIDISATION, POWER_SERIES_HYBRIDISATION),
the light comes from the positions, while for ANALYTIC_CAVITY_OUTPUT the intracavity light comes
from the output light: the two families of functions call each other across mixins.

The method-specific calculations are in the mixins, which `MultiTrap` combines with this class:

- `levisim.multiphysics.multitrap.linear_amplifier_mixin.LinearAmplifierMixin` - methods 1-3
- `levisim.multiphysics.multitrap.hybridisation_mixin.HybridisationMixin` - methods 4-5
- `levisim.multiphysics.multitrap.analytic_mixin.AnalyticMixin` - methods 6-7

`MultiTrapBase` is not meant to be built on its own: its constructor calls
`setup_linear_amplifier_model_matrices` and `setup_hybridisation` from the mixins.
"""

import numpy as np
import itertools

from copy import copy
from functools import cached_property

from levisim.core.constants import (
    π, ħ, kB,
    c as speed_of_light,
    m_air,
    shot_noise,
    node,
)
from levisim.core.units import (
    degrees,
    metre, nm,
    Hz, kHz,
    kg,
    mbar,
)

from levisim.core.logger import logger
from levisim.core.wrappers import timer, profiler, progress_bar

from levisim.physics.tweezer import Tweezer
from levisim.physics.bead import Bead
from levisim.physics.gas import Gas
from levisim.physics.cavity import Cavity
from levisim.physics.coherent_scattering import static_cavity_hessian
from levisim.physics.trap import Trap

from levisim.multiphysics.enums.quantum_linear_theory_method import QuantumLinearTheoryMethod
from levisim.multiphysics.dimension import Dimension
from levisim.multiphysics.hybridisation import generate_subscripts_and_labels
from levisim.multiphysics.multitrap.omega_model import ΩModel


__all__ = [
    'MultiTrapBase',
]


TIMER = True
PROFILER = False
PROGRESS_BAR = True


# keyword arguments of MultiTrap that replace the built-in per-trap physics (one entry per trap)
PHYSICAL_PER_TRAP_KWARGS = ('tweezers', 'beads', 'trap_kwargs')


class MultiTrapBase:
    """
    Shared state, information printing, spectrum entry points and method dispatch of `MultiTrap`

    Use `levisim.multiphysics.multitrap.MultiTrap`, which adds the method-specific mixins.
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
        # in physical mode, lists of per-trap objects (see setup_levitated_physics) set the default
        default_num_traps = 2
        if not kwargs.get('toy_params'):
            per_trap_lists = [kwargs[name] for name in PHYSICAL_PER_TRAP_KWARGS if kwargs.get(name) is not None]
            if per_trap_lists:
                default_num_traps = len(per_trap_lists[0])
        self.num_traps = kwargs.get('num_traps', default_num_traps)
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

        # the one-line INFO summary of the system is logged at the end of __init__
        if len(self.ω_ranges) > 1:
            for segment in self.ω_ranges:
                logger.debug(
                    'ω_range segment [{:.1f}, {:.1f}] kHz with {} points (Δω = {:.3f} kHz)',
                    segment.min() / kHz, segment.max() / kHz, len(segment),
                    (segment[1] - segment[0] if len(segment) > 1 else 0.0) / kHz,
                )
            logger.debug('{} ω_range segments stitched into {} unique points', len(self.ω_ranges), len(self.ω_range))

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

        self.toy_params = kwargs.get('toy_params')

        # static coherent-scattering frequency shifts and the cavity's dispersive potential (see
        # setup_static_cavity_shifts): on by default with physical parameters; toy parameters have
        # no static cavity field, so there they are always off
        static_cavity_shifts = kwargs.get('static_cavity_shifts')
        if self.toy_params:
            if static_cavity_shifts:
                logger.warning(
                    'static_cavity_shifts=True has no effect with toy_params, which have no static cavity'
                    ' field (give the shifted frequencies through the toy Ω model instead)'
                )
            self.static_cavity_shifts = False
        else:
            self.static_cavity_shifts = True if static_cavity_shifts is None else bool(static_cavity_shifts)

        if self.toy_params:
            self.cavity = Cavity(**cavity_kwargs)
            ignored = [name for name in (*PHYSICAL_PER_TRAP_KWARGS, 'φn_dict') if kwargs.get(name) is not None]
            if ignored:
                logger.warning(
                    'MultiTrap keyword arguments {} only apply to physical parameters and are ignored with'
                    ' toy_params (give toy tweezer phases as toy_params["φn_dict"])', ignored,
                )
            self.setup_toy_parameters(**self.toy_params)
        else:
            self.setup_levitated_physics(**kwargs)
            # the cavity is built after the tweezers, so that it is driven at their frequency
            self.cavity = Cavity(**self._cavity_kwargs_with_tweezer_frequency(cavity_kwargs))
            self.setup_linearised_coherent_scattering_physics(**kwargs)
            self.setup_static_cavity_shifts()
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

        # direct mechanical couplings between dimensions, in one matrix (see its docstring)
        self.mechanical_coupling_matrix = self.build_mechanical_coupling_matrix()


        # ------------------------------------------------------------------------------------------
        # PHOTON RECOIL HEATING (on by default with physical parameters)
        # ------------------------------------------------------------------------------------------
        # each dimension heats at Γ_rec (Trap.photon_recoil_heating_rates) from the recoil of the
        # tweezer photons its bead scatters. Recoil adds no damping, so it is included by raising the
        # bath occupation, n → n + Γ_rec/Γ: the same d〈n〉/dt = -Γ(〈n〉 - n) + Γ_rec. It does not
        # change the drift matrix, so the dynamics, their stability and the mode frequencies are the
        # same with or without it; only the noise is larger. Recoil is a momentum diffusion, whereas
        # levisim's bath noise drives both quadratures; the two are equivalent in the rotating-wave
        # sense used throughout (Ω much larger than every damping, heating and optical rate).
        #   - physical parameters: on unless photon_recoil=False. If it was not asked for explicitly
        #     and a dimension has no damping (gas pressure 0), there is no bath to carry the heating:
        #     a WARNING is logged and recoil is left out. photon_recoil=True then raises ValueError.
        #   - toy parameters: there is no tweezer or bead to scatter from, so recoil is always off;
        #     photon_recoil=True raises ValueError.
        photon_recoil = kwargs.get('photon_recoil')
        recoil_requested = photon_recoil is not None and bool(photon_recoil)
        if photon_recoil is None:
            photon_recoil = not self.toy_params
        self.photon_recoil = bool(photon_recoil)
        self.recoil_heating_rate_dict = {}
        if self.photon_recoil:
            if self.toy_params:
                msg = 'photon_recoil=True needs physical parameters (tweezers and beads), not toy_params'
                logger.error(msg)
                raise ValueError(msg)
            undamped = [label for label, dimension in self.dimensions.items() if not dimension.Γ > 0]
            if undamped and recoil_requested:
                msg = f'photon_recoil=True needs a non-zero damping Γ{undamped[0]} (gas pressure > 0) to carry the recoil heating'
                logger.error(msg)
                raise ValueError(msg)
            if undamped:
                logger.warning(
                    '{}: photon recoil heating is left out, since the dimensions {} have no damping (gas'
                    ' pressure 0) to carry it; pass photon_recoil=False to silence this',
                    self, ', '.join(undamped),
                )
                self.photon_recoil = False
        if self.photon_recoil:
            for label, dimension in self.dimensions.items():
                Γ_rec = dimension.trap.photon_recoil_heating_rates[dimension.axis]
                # the rate in phonons per second goes as x_zpf² ∝ 1/Ω, so it follows a static shift
                # of the dimension's frequency (Trap uses the bare tweezer frequency)
                Ω_bare = dimension.trap.Ωj[dimension.axis]
                if dimension.Ω != Ω_bare:
                    Γ_rec = Γ_rec * Ω_bare / dimension.Ω
                self.recoil_heating_rate_dict[label] = Γ_rec
            logger.debug(
                'Photon recoil heating added to the baths of {}: {}', self,
                ', '.join(f'Γ_rec,{label}/2π = {Γ_rec / kHz:.4g} kHz' for label, Γ_rec in self.recoil_heating_rate_dict.items()),
            )

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
            if self.photon_recoil:
                njn = njn + self.recoil_heating_rate_dict[label] / self.dimensions[label].Γ

            # phonon annihilation operators, 〈bjn_in(ω)·bjn_in†(-ω)〉= njn + 1
            self.quantum_factor_per_noise_operator.append(njn + 1)
            # phonon creation operators, 〈bjn_in†(ω)·bjn_in(-ω)〉= njn
            self.quantum_factor_per_noise_operator.append(njn)

        # ------------------------------------------------------------------------------------------
        # QUANTUM LINEAR THEORY METHOD
        # ------------------------------------------------------------------------------------------
        # this is the quickest, because it is the direct numerical inversion of the smallest matrix
        # self.QLT_method = kwargs.get('QLT_method', QuantumLinearTheoryMethod.NUMERICAL_HYBRIDISATION)

        # the drift matrix A of the linearised Langevin equations (LinearAmplifierMixin); it is built
        # for every method, since the linear-stability check at construction uses it
        self.setup_linear_amplifier_model_matrices()

        # the hybridisation functions, and the power series of the deranged determinant and
        # adjugate if asked for (HybridisationMixin)
        self.setup_hybridisation(**kwargs)

        # ------------------------------------------------------------------------------------------
        # LOGGING
        # ------------------------------------------------------------------------------------------
        self._log_construction()

    def __repr__(self):
        """
        String to represent an instance of MultiTrap
        """
        return f'MultiTrap({self.num_traps}×{self.Cartesian_axes_per_trap})'

    def _log_construction(self):
        """
        Logs a one-line summary of the built system at INFO, and every parameter dump at DEBUG

        The parameter dumps are built lazily, so they cost nothing when no sink takes DEBUG.
        """
        if self.toy_params:
            parameters = f'toy parameters (Ω_model {self.Ω_model!r})'
        else:
            parameters = 'physical parameters (tweezers, beads, gas)'
        if self.load_in_power_series:
            power_series = f'; power series loaded up to order {self.max_order}'
        elif self.generate_power_series:
            power_series = f'; power series generated up to order {self.max_order}'
        else:
            power_series = ''
        logger.info(
            'Built {}: {} dimensions ({}) from {}; cavity Δ = {:.1f} kHz, κ = {:.1f} kHz;'
            ' ω_range = [{:.1f}, {:.1f}] kHz over {} points{}',
            self, self.num_dimensions, ', '.join(self.labels), parameters,
            self.cavity.Δ / kHz, self.cavity.κ / kHz,
            self.ω_range.min() / kHz, self.ω_range.max() / kHz, len(self.ω_range), power_series,
        )

        def parameter_dumps():
            dumps = [self.info_str, self.cavity.info_str, self.gas.info_str]
            if self.toy_params:
                dumps.append(self.toy_params_info_str)
            dumps.extend(dimension.info_str for dimension in self.dimensions.values())
            return '\n'.join(dumps)

        logger.opt(lazy=True).debug('Parameters of {}:\n{}', lambda: self, parameter_dumps)

    def _check_linear_stability(self):
        """
        Logs a WARNING if the drift matrix A has an eigenvalue with a positive real part

        Every QLT method returns the stationary spectra of the linearised Langevin equations, e.g.
        (-iω·I - A)⁻¹ for the LAM. Those describe a physical steady state only if all eigenvalues
        of A have negative real parts; otherwise a mode grows as exp(Re λ·t) and there is no
        steady state, although the formulas still return finite numbers. One eigvals call on the
        small (2N + 2)-square drift matrix per build, so the check is cheap, and it changes no
        result. The test is `is_stable`, with the tolerance of `stability_tolerance`.
        """
        worst = self.drift_eigenvalues[0]
        logger.debug('{}: largest drift-matrix eigenvalue real part Re λ = {:+.4e} /s', self, worst.real)
        num_unstable = int(np.sum(self.drift_eigenvalues.real > self.stability_tolerance))
        if num_unstable:
            logger.warning(
                '{}: {} of the {} drift-matrix eigenvalues have a positive real part (largest'
                ' Re λ = {:+.4g} /s, an e-folding time of {:.3g} ms, on the mode at |Im λ| = {:.2f} kHz),'
                ' so the linearised dynamics are unstable and have no steady state: the stationary'
                ' spectra are formally finite but do not describe a physical steady state.'
                ' Check the detuning Δ, the couplings and the tweezer phases φn.',
                self, num_unstable, len(self.drift_eigenvalues), worst.real, 1e3 / worst.real, abs(worst.imag) / kHz,
            )

    ################################################################################################
    # LINEAR STABILITY
    # ----------------------------------------------------------------------------------------------
    # The linearised Langevin equations dc/dt = A·c + c_in have a steady state only if every
    # eigenvalue λ of the drift matrix A has Re λ < 0. Each eigenvalue is a normal mode: |Im λ| is
    # its angular frequency and -Re λ its amplitude damping rate (half its energy damping rate, the
    # half-width of its Lorentzian). Eigenvalues come in pairs λ, λ* (a mode and its conjugate), so
    # every mode appears twice in `drift_eigenvalues` and once in `unstable_modes`. See the docs page
    # concepts/stability for the theory.
    # ----------------------------------------------------------------------------------------------
    @cached_property
    def drift_eigenvalues(self):
        """
        Eigenvalues λ of the drift matrix A, sorted by decreasing real part (largest growth first)

        Re λ is minus the amplitude damping rate of the normal mode, ``|Im λ|`` its angular frequency.
        [1/s, complex] (divide Im λ by `kHz` for the frequency in kHz)
        """
        eigenvalues = np.linalg.eigvals(self.drift_matrix)
        return eigenvalues[np.argsort(-eigenvalues.real, kind='stable')]

    @property
    def stability_tolerance(self):
        """
        Largest Re λ still counted as stable: 1e-8 of the largest ``|A|`` entry

        It ignores round-off in the decay rates of dark modes, which can be as small as the gas
        damping.
        [1/s]
        """
        return 1e-8 * np.abs(self.drift_matrix).max()

    @property
    def max_growth_rate(self):
        """
        Largest real part of the drift-matrix eigenvalues, max Re λ: negative (minus the damping of
        the least-damped mode) for a stable system, positive (the growth rate of the fastest-growing
        mode, whose amplitude grows as exp(Re λ·t)) for an unstable one
        [1/s]
        """
        return float(self.drift_eigenvalues[0].real)

    @property
    def is_stable(self):
        """
        Whether the linearised dynamics have a steady state: every Re λ ≤ `stability_tolerance`
        [bool]
        """
        return bool(self.max_growth_rate <= self.stability_tolerance)

    @property
    def least_stable_mode(self):
        """
        (Ω, Re λ) of the normal mode with the largest Re λ: its angular frequency Ω = ``|Im λ|`` and
        its growth rate (positive) or minus its damping rate (negative)
        [2π·Hz, 1/s]
        """
        worst = self.drift_eigenvalues[0]
        return float(abs(worst.imag)), float(worst.real)

    @property
    def unstable_modes(self):
        """
        The growing normal modes, as a list of (Ω, Re λ) pairs, fastest-growing first: Ω = ``|Im λ|``
        is the mode's angular frequency and Re λ > `stability_tolerance` its growth rate. Each
        conjugate pair of eigenvalues λ, λ* counts once (through the member with Im λ > 0); a
        purely real eigenvalue (Ω = 0, a static instability) counts on its own.
        Empty for a stable system.
        [2π·Hz, 1/s]
        """
        modes = []
        for eigenvalue in self.drift_eigenvalues:
            if eigenvalue.real <= self.stability_tolerance:
                continue
            # a purely real eigenvalue (a static, non-oscillating instability, e.g. a negative net
            # spring constant) is its own conjugate, so it counts on its own; numerically its
            # imaginary part is only zero to within round-off
            is_real = abs(eigenvalue.imag) <= self.stability_tolerance
            if is_real or eigenvalue.imag > 0:
                modes.append((0.0 if is_real else float(eigenvalue.imag), float(eigenvalue.real)))
        return modes

    @property
    def stability_info_str(self):
        """
        Lines describing the linear stability, as printed in `info_str`
        """
        Ω, growth_rate = self.least_stable_mode
        lines = (
            f'\n    linearly_stable = {self.is_stable}'
            f'\n    max_growth_rate = {growth_rate:+.4g} /s (max Re λ, on the mode at |Im λ|/2π = {Ω / kHz:.3f} kHz)'
        )
        for Ω, growth_rate in self.unstable_modes:
            lines += f'\n    unstable mode at {Ω / kHz:.3f} kHz grows at Re λ = {growth_rate:+.4g} /s'
        return lines

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
            f'\n    photon_recoil = {self.photon_recoil}'
            f'{self.stability_info_str}'
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

    ################################################################################################
    # SETUP FUNCTIONS
    # ----------------------------------------------------------------------------------------------
    def build_mechanical_coupling_matrix(self):
        """
        The direct mechanical couplings between the dimensions, as one matrix

        Entry [j, k] is the rate gjk at which dimension k drives dimension j, in levisim's g
        convention: the term -gjk·qj·qk of H/ħ (qj = bj + bj†), so that ::

            dbj/dt ∋ +i·gjk·qk,   dbj†/dt ∋ -i·gjk·qk

        Both axes are ordered as `self.labels`. The matrix is real and need not be symmetric (a
        non-reciprocal coupling has gjk ≠ gkj). It is what the drift matrix, the hybridisation
        functions and the analytic methods read their mechanical couplings from.

        Built from the Dimensions, it holds the couplings between the axes of one bead (gxy, gxz
        and gyz, from the coherent-scattering expansion or the toy parameters toy_gxy, toy_gxz,
        toy_gyz) and zeros elsewhere: with two axes per trap only gxy is used, with three all of
        them, with one none. Dimensions of different beads are not coupled directly.
        [2π·Hz]
        """
        couplings = np.zeros((self.num_dimensions, self.num_dimensions))
        dims = [self.dimensions[label] for label in self.labels]
        for jj, dim_j in enumerate(dims):
            for kk, dim_k in enumerate(dims):
                if jj == kk or dim_j.trap_index != dim_k.trap_index:
                    continue
                axes = ''.join(sorted(dim_j.axis + dim_k.axis))
                couplings[jj, kk] = getattr(dim_j, f'g{axes}')
        return couplings

    def setup_levitated_physics(self, **kwargs):
        """
        Setup the physics of many beads levitated in tweezer traps

        By default the tweezers and beads are the built-in lists below, cut to the first `num_traps`:
        1064 nm tweezers at θ = 45° with phases φn = 0°, 90°, 20°, 30°, ..., 80° and slightly different
        waists, and beads (default `Bead` material) of radius 60.1, 60.2, ..., 60.9 nm, all at the
        node. Each list can be replaced through `MultiTrap` keyword arguments:

            - tweezers: list of `Tweezer` instances, one per trap
            - beads: list of `Bead` instances, one per trap
            - trap_kwargs: list of dicts of extra `Trap` keyword arguments, one per trap, e.g.
              `override_Ω_dict` to use measured mechanical frequencies, or `cavity_drive_tweezer`
              and `cavity_drive_field_ratio` for a bead that scatters into the cavity the light of
              another tweezer than the one trapping it (e.g. a bead in a 976 nm trap, which the
              cavity does not resonate with, scattering the out-of-focus field ζ·E0 of a 1064 nm
              tweezer; see `Trap.cavity_drive_field`). The coherent-scattering couplings then use
              that tweezer's wavelength, polarisation angle θ and Rayleigh range, and the cavity
              frequency follows the cavity-drive tweezers
            - φn_dict: {n: φn} tweezer phases [rad] for any of the traps n = 1, ..., num_traps; they
              replace the phase of that trap's tweezer (given or built-in). The `Tweezer` instances
              passed in are not modified: a trap whose phase is replaced gets a copy of its tweezer.
            - photon_recoil: add the photon-recoil heating of each bead to the bath of each of its
              dimensions (default True with physical parameters; see the PHOTON RECOIL HEATING
              block of `__init__`)

        If `num_traps` is not given, it is the length of the first of these lists that is given.
        """
        tweezers = kwargs.get('tweezers')
        beads = kwargs.get('beads')
        trap_kwargs = kwargs.get('trap_kwargs')
        φn_overrides = kwargs.get('φn_dict')

        def check_length(name, items):
            if items is not None and len(items) != self.num_traps:
                msg = f'{name} has {len(items)} entries but num_traps = {self.num_traps}: give one per trap'
                logger.error(msg)
                raise ValueError(msg)

        check_length('tweezers', tweezers)
        check_length('beads', beads)
        check_length('trap_kwargs', trap_kwargs)

        if tweezers is None:
            # roughly 42 degrees
            θtw = 0.731991
            θtw = 45*degrees
            builtin_tweezers = [
                Tweezer(θ=θtw),
                Tweezer(wx0=950*nm, wy0=1050*nm, θ=θtw, φ=90*degrees),
                Tweezer(wx0=975*nm, wy0=1025*nm, θ=θtw, φ=20*degrees),
                Tweezer(wx0=965*nm, wy0=1015*nm, θ=θtw, φ=30*degrees),
                Tweezer(wx0=940*nm, wy0=1020*nm, θ=θtw, φ=40*degrees),
                Tweezer(wx0=930*nm, wy0=1010*nm, θ=θtw, φ=50*degrees),
                Tweezer(wx0=915*nm, wy0=1055*nm, θ=θtw, φ=60*degrees),
                Tweezer(wx0=925*nm, wy0=1045*nm, θ=θtw, φ=70*degrees),
                Tweezer(wx0=935*nm, wy0=1035*nm, θ=θtw, φ=80*degrees),
            ]
            if self.num_traps > len(builtin_tweezers):
                msg = (
                    f'there are {len(builtin_tweezers)} built-in tweezers but num_traps = {self.num_traps}:'
                    ' pass tweezers=[...] (one Tweezer per trap)'
                )
                logger.error(msg)
                raise ValueError(msg)
            tweezers = builtin_tweezers[:self.num_traps]
        else:
            tweezers = list(tweezers)

        if φn_overrides:
            unknown = sorted(set(φn_overrides) - set(range(1, self.num_traps + 1)))
            if unknown:
                msg = f'φn_dict has trap indices {unknown} outside 1, ..., num_traps = {self.num_traps}'
                logger.error(msg)
                raise ValueError(msg)
            for n, φn in φn_overrides.items():
                tweezer = copy(tweezers[n - 1])
                tweezer.φ = float(φn)
                tweezers[n - 1] = tweezer
            logger.debug(
                'Tweezer phases replaced by φn_dict: {}',
                {n: f'{φn / degrees:.3f}°' for n, φn in φn_overrides.items()},
            )
        self.tweezers = tweezers
        self.φn_dict = {ii + 1: tweezer.φ for ii, tweezer in enumerate(self.tweezers)}

        # NOTE: all mech-mech, and z-light couplings are 0 at the node
        # for the mech-mech couplings:
        # x-y couplings depend on A0
        # x-z and y-z couplings on P0
        # A0, and P0 depend on the static light field
        # the static light field sums each trap's 'drive' energy into the cavity
        # each of these cavity drive energies goes as cos(Φ), due to the linearised potential V_CS
        # z-light couplings also go with cos(Φ), due to the linearised potential V_CS
        if beads is None:
            Φbead = node
            builtin_beads = [
                Bead(r=60.1*nm, Φ=Φbead),
                Bead(r=60.2*nm, Φ=Φbead),
                Bead(r=60.3*nm, Φ=Φbead),
                Bead(r=60.4*nm, Φ=Φbead),
                Bead(r=60.5*nm, Φ=Φbead),
                Bead(r=60.6*nm, Φ=Φbead),
                Bead(r=60.7*nm, Φ=Φbead),
                Bead(r=60.8*nm, Φ=Φbead),
                Bead(r=60.9*nm, Φ=Φbead),
            ]
            if self.num_traps > len(builtin_beads):
                msg = (
                    f'there are {len(builtin_beads)} built-in beads but num_traps = {self.num_traps}:'
                    ' pass beads=[...] (one Bead per trap)'
                )
                logger.error(msg)
                raise ValueError(msg)
            beads = builtin_beads[:self.num_traps]
        self.beads = list(beads)

        if trap_kwargs is None:
            trap_kwargs = [{} for _ in range(self.num_traps)]

        # each particle/trap/tweezer has its own physics!
        self.traps = [
            Trap(
                tweezer=self.tweezers[_],
                bead=self.beads[_],
                gas=self.gas,
                trap_index=_ + 1,
                **trap_kwargs[_],
            )
            for _ in range(self.num_traps)
        ]

    def _cavity_kwargs_with_tweezer_frequency(self, cavity_kwargs):
        """
        The cavity keyword arguments, with `ωtw` set to the first cavity-drive tweezer's frequency
        unless given

        The cavity-drive tweezer of a trap is its own tweezer unless `trap_kwargs` gives it a
        `cavity_drive_tweezer` (see `Trap.drive_tweezer`). The cavity mode is driven by the light the
        beads scatter from these tweezers, so its optical
        frequency (ω = ωtw - Δ, hence k and the field per photon E0 in every coupling) must be the
        tweezers'. `Cavity` alone defaults to `Tweezer().ω` (1064 nm), which used to apply even to
        a MultiTrap of 1550 nm tweezers. One drive frequency is assumed for all traps, so a WARNING
        is logged if the tweezers have different wavelengths, or if an explicit `ωtw` differs from
        all of them by more than 1 ppm (multi-tone detunings shift it by only ~1e-9).
        """
        # the light each bead scatters into the cavity: its trapping tweezer's, unless a trap was
        # given a cavity_drive_tweezer (e.g. a bead held by 976 nm light that scatters 1064 nm light)
        drive_tweezers = [trap.drive_tweezer for trap in self.traps]
        tweezer_ωs = np.array([tweezer.ω for tweezer in drive_tweezers])
        tweezer_λs_nm = ', '.join(f'{tweezer.λ / nm:.6g}' for tweezer in drive_tweezers)
        if not np.allclose(tweezer_ωs, tweezer_ωs[0], rtol=1e-12, atol=0):
            logger.warning(
                'The cavity-drive tweezers have different wavelengths (λ = {} nm), but the model has'
                ' one cavity drive frequency: the cavity uses ωtw = {} (λ = {:.6g} nm) for every trap.'
                ' A tweezer far from any cavity resonance scatters nothing into the mode; give that'
                ' trap the cavity_drive_tweezer whose light it does scatter (Trap keyword arguments)',
                tweezer_λs_nm,
                "cavity_kwargs['ωtw']" if 'ωtw' in cavity_kwargs else 'the first tweezer frequency',
                2 * π * speed_of_light / cavity_kwargs.get('ωtw', tweezer_ωs[0]) / nm,
            )
        if 'ωtw' in cavity_kwargs:
            ωtw = cavity_kwargs['ωtw']
            if np.all(np.abs(tweezer_ωs / ωtw - 1) > 1e-6):
                logger.warning(
                    "cavity_kwargs['ωtw'] (λ = {:.6g} nm) differs from every tweezer frequency (λ = {} nm):"
                    ' the cavity field and the couplings use the ωtw given',
                    2 * π * speed_of_light / ωtw / nm, tweezer_λs_nm,
                )
            return cavity_kwargs
        return {**cavity_kwargs, 'ωtw': tweezer_ωs[0]}

    def setup_linearised_coherent_scattering_physics(self, **kwargs):
        """
        Setup the linearised properties of tweezers levitating beads
        which are all coherently scattering into the same cavity

        Linearisation splits the optical degrees of freedom into 2 parts ::

            α ≡ a dominant, static, complex valued part
            a ≡ a small, dynamic, operator valued part

            a → α + a
            a† → α* + a†
            A → A0 + (a† + a)
            P → P0 + i(a† - a)

            α = 1/2 (A0 + iP0)
            α* = 1/2 (A0 - iP0)

        To derive α (the mean static cavity photon field), consider the full Hamiltonian ::

            H ≡ H_free + H_int
            H_free ≡ -ħΔ·a†a

        NOTE: *all* tweezers are now included in a multi-particle setup! ::

            Ed --> Etot = Ed1 + Ed2 + ... = ∑ Ed
            H_int ≡ ∑ V_CS = Σ -Ed·sin(θ)[cos(Φ)·(...)...]·(A ...)
            V_CS(0) = -Ed·sin(θ)·cos(Φ).A

        Heisenberg's EoMs + dissipation ::

            da/dt = i[H/ħ, a] - (κ/2)a
            da/dt = (iΔ - κ/2)a + ∑ i[V_CS/ħ, a]

        Static part exists at origin ::

            dα/dt = 0 = (iΔ - κ/2)α +  ∑ i[V_CS(0)/ħ, a]
            i[V_CS(0)/ħ, a] = -iωd·sin(θ)·cos(Φ)·[A, a] = +iωd·sin(θ)·cos(Φ), since [A, a] = -1
            dα/dt = 0 = (iΔ - κ/2)α + ∑ iωd·sin(θ)·cos(Φ)
            α = -∑ iωd·sin(θ)·cos(Φ) / (iΔ - κ/2)
            α = +∑ ωd·sin(θ)·cos(Φ) · [(-Δ + iκ/2) / (Δ² + (κ/2)²]
        """

        for trap in self.traps:
            # Cavity drive energy for a particular bead in a particular tweezer trap
            # [J]
            # (the field the bead scatters is its trapping tweezer's focal field E0 by default, or
            # cavity_drive_field_ratio × E0 of another tweezer: see Trap.cavity_drive_field)
            trap.Ed = 0.5 * trap.bead.α * trap.cavity_drive_field * self.cavity.E0
            # Cavity drive frequency for a particular bead in a particular tweezer trap
            # [2π·Hz]
            trap.ωd = trap.Ed / ħ

        # ∑ ωd·sin(θ)·cos(Φ)
        self.ωd_total = sum(
            [trap.ωd * np.sin(trap.drive_tweezer.θ) * np.cos(trap.bead.Φ) for trap in self.traps]
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

    def setup_static_cavity_shifts(self):
        """
        Setup the static stiffness that the static cavity field adds to each bead's trap

        The linearisation of `setup_linearised_coherent_scattering_physics` keeps, at second order
        in the bead displacement, only the off-diagonal terms of the coherent-scattering potential
        (gxy ∝ A0, gxz and gyz ∝ P0). The same expansion has diagonal terms ∝ A0·cos Φ, and the
        bead also sits in the cavity's own standing-wave intensity ``-α_pol·E0²·|α|²·cos²(k·d·r + Φ)``,
        whose curvature ``∝ |α|²·cos 2Φ`` adds diagonal terms and an x-y term along the cavity axis
        d = (sin θ, cos θ, 0). These are the 3 × 3 Hessians K of
        `levisim.physics.coherent_scattering.static_cavity_hessian` [N/m], one per trap, kept in
        `self.static_cavity_hessians`. With `static_cavity_shifts=True` (the default with physical
        parameters) each dimension's frequency becomes ::

            Ωjn = Ωjn,tw·√(1 + Kjj/(m·Ωjn,tw²))      (Ωjn,tw = Trap.Ωj, the tweezer's)

        and Kxy adds -Kxy·x_zpf·y_zpf/ħ to gxy. A trap given measured frequencies
        (`override_Ω_dict` in its `trap_kwargs`) keeps them as they are, since a measured
        frequency already contains the static shift. The shifted Ωjn is used self-consistently: it is
        Dimension.Ω (the mechanical susceptibilities), it sets the zero-point fluctuation (so
        every coupling g, gxy, gxz, gyz is rescaled by √(Ωtw/Ω) per dimension), the thermal and
        recoil phonon occupations and the cooling estimates. Both terms vanish when the static
        cavity field is zero, i.e. when every bead is at a node, and the frequencies are then
        bit-identical to those without shifts. `static_cavity_shifts=False` reproduces the
        couplings and frequencies of levisim ≤ 0.1.0.

        Raises ValueError if a modelled dimension is statically unstable (Ωtw² + K/m ≤ 0).
        """
        α = (self.cavity.A0 + 1j * self.cavity.P0) / 2
        modelled_axes = 'xyz'[:self.Cartesian_axes_per_trap]
        self.static_cavity_hessians = {}
        self.static_shifted_Ωj = {}
        for trap in self.traps:
            n = trap.trap_index
            if self.static_cavity_shifts:
                K = static_cavity_hessian(trap, self.cavity.E0, α)
            else:
                K = np.zeros((3, 3))
            self.static_cavity_hessians[n] = K
            shifted = {}
            for ii, axis in enumerate('xyz'):
                Ω_tw = trap.Ωj[axis]
                # NOTE: frequencies given through override_Ω_dict are measured ones, which already
                # contain whatever static shift the cavity field causes, so it is not added again
                # (the x-y coupling of the cavity's dispersive potential still is)
                if K[ii, ii] == 0 or trap.override_Ω_dict:
                    shifted[axis] = Ω_tw
                    continue
                relative = K[ii, ii] / (trap.bead.m * Ω_tw**2)
                if 1 + relative <= 0 and axis in modelled_axes:
                    msg = (
                        f'the static cavity field makes {axis}{n} statically unstable: Ω{axis}{n},tw/2π ='
                        f' {Ω_tw / kHz:.4g} kHz and the static stiffness K/m = {relative:.4g}·Ω² gives Ω² ≤ 0'
                        ' (pass static_cavity_shifts=False to leave the static shifts out)'
                    )
                    logger.error(msg)
                    raise ValueError(msg)
                shifted[axis] = Ω_tw * np.sqrt(1 + relative) if 1 + relative > 0 else np.nan
            self.static_shifted_Ωj[n] = shifted
        if self.static_cavity_shifts:
            logger.debug(
                'Static cavity shifts of {}: {}', self,
                ', '.join(
                    f'δΩ{axis}{n}/2π = {(self.static_shifted_Ωj[n][axis] - trap.Ωj[axis]) / kHz:+.4g} kHz'
                    for trap in self.traps for n in [trap.trap_index] for axis in modelled_axes
                ),
            )

    # every toy_params key that setup_toy_parameters reads, besides those of the chosen ΩModel kind
    TOY_PARAMETER_KEYS = frozenset({
        'Ω_model', 'Ωz_to_Ωx_multiplier',
        'P', 'T', 'm', 'r', 'ρ', 'vth', 'use_Γ_formula', 'toy_Γ', 'Γjn_dict',
        'toy_gx', 'toy_gy', 'toy_gz', 'toy_gxy', 'toy_gxz', 'toy_gyz', 'gjn_dict',
        'φ_odd_n', 'φ_even_n', 'φn_dict',
    })

    def warn_about_unknown_toy_parameters(self, params):
        """
        Logs a WARNING naming any toy_params key that nothing reads

        Unknown keys used to be ignored silently, so a renamed or misspelt key (e.g. 'φ2' for
        'φ_even_n', 'gs' for 'gjn_dict', or 'Ω_gap' with the 'pairs' model) changed the physics
        without any message. Parameters of an ΩModel kind other than the chosen one count as
        unknown, since they have no effect.
        """
        known = set(self.TOY_PARAMETER_KEYS) | set(ΩModel.KINDS[self.Ω_model_object.kind])
        unknown = sorted(set(params) - known)
        if unknown:
            logger.warning(
                'toy_params keys {} are not used by MultiTrap (Ω_model {!r}) and have no effect;'
                ' check for renamed or misspelt keys', unknown, self.Ω_model_object.kind,
            )
        return unknown

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
        self.warn_about_unknown_toy_parameters(params)
        logger.debug('Setting up toy parameters with {!r}', self.Ω_model_object)
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

    ################################################################################################
    # METHOD DISPATCH (the light and the positions call each other across the mixins)
    # ----------------------------------------------------------------------------------------------
    def light_noise_vectors(self, method, ω):
        """
        The noise vectors for the annihilation and creation operators of photons
        in the shared light mode inside the cavity
        a(ω), a†(ω)

        Dispatches to `linear_amplifier_light_noise_vectors` (LINEAR_AMPLIFIER_MODEL),
        `analytic_inside_cavity_light_noise_vectors` (ANALYTIC_INSIDE_CAVITY),
        `analytic_cavity_output_light_noise_vectors` (ANALYTIC_CAVITY_OUTPUT), and otherwise
        `light_noise_vectors_from_positions`.
        """
        if method == QuantumLinearTheoryMethod.LINEAR_AMPLIFIER_MODEL:
            return self.linear_amplifier_light_noise_vectors(ω)

        if method == QuantumLinearTheoryMethod.ANALYTIC_INSIDE_CAVITY:
            return self.analytic_inside_cavity_light_noise_vectors(ω)

        if method == QuantumLinearTheoryMethod.ANALYTIC_CAVITY_OUTPUT:
            return self.analytic_cavity_output_light_noise_vectors(ω)

        # NOTE: remaining possible methods are
        #   QuantumLinearTheoryMethod.HYBRID_LINEAR_AMPLIFIER_MODEL
        #   QuantumLinearTheoryMethod.NUMERICAL_HYBRIDISATION
        #   QuantumLinearTheoryMethod.POWER_SERIES_HYBRIDISATION
        return self.light_noise_vectors_from_positions(method, ω)

    def light_noise_vectors_from_positions(self, method, ω):
        """
        The intracavity light noise vectors a(ω), a†(ω) from the hybridised positions ::

            a(ω)  = χC(ω)  [ i ∑ Gjn·qjn(ω) + √κ·a_in]
            a†(ω) = χC*(-ω)[-i ∑ Gjn*·qjn(ω) + √κ·a_in†]

        (the creation operator couples through the conjugate Gjn*, since qjn is Hermitian), used
        by the methods that solve for the positions first: HYBRID_LINEAR_AMPLIFIER_MODEL,
        NUMERICAL_HYBRIDISATION and POWER_SERIES_HYBRIDISATION.
        """
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
        accounted for via input-output theory ::

            a_out(ω)  = -√κ·a(ω)  + a_in(ω)
            a_out†(ω) = -√κ·a†(ω) + a_in†(ω)

        ANALYTIC_CAVITY_OUTPUT has its own closed form (`analytic_output_light_noise_vectors`);
        every other method applies the relations above to `light_noise_vectors`.
        """
        if method == QuantumLinearTheoryMethod.ANALYTIC_CAVITY_OUTPUT:
            return self.analytic_output_light_noise_vectors(ω)

        a_hat, a_dag = self.light_noise_vectors(method, ω)
        κ = self.cavity.κ
        a_out_hat = -np.sqrt(κ) * a_hat
        a_out_dag = -np.sqrt(κ) * a_dag
        a_out_hat[0] = a_out_hat[0] + 1
        a_out_dag[1] = a_out_dag[1] + 1
        return a_out_hat, a_out_dag

    def hybridised_positions(self, method, ω):
        """
        Column vector of the solutions to the QLT equations of motion for each dimension's position
        Each entry in the column vector is a noise vector

        Dispatches to `linear_amplifier_hybridised_positions` (LINEAR_AMPLIFIER_MODEL and
        HYBRID_LINEAR_AMPLIFIER_MODEL), `hybridisation_hybridised_positions` (NUMERICAL_HYBRIDISATION and
        POWER_SERIES_HYBRIDISATION) and `analytic_hybridised_positions` (both analytic methods).
        TNT_LINEAR_AMPLIFIER_MODEL has no per-frequency noise vectors and raises ValueError.
        """
        if (
            method.is_linear_amplifier_model
            and
            method != QuantumLinearTheoryMethod.TNT_LINEAR_AMPLIFIER_MODEL
        ):
            return self.linear_amplifier_hybridised_positions(ω)

        if method.is_hybridisation:
            return self.hybridisation_hybridised_positions(method, ω)

        if method.is_analytic:
            return self.analytic_hybridised_positions(method, ω)

        # only the TNT LAM reaches here: it has no per-frequency noise vectors (it builds the
        # position PSDs from T·N·T† in position_power_spectra_array), so say so instead of returning
        # None, which failed later with an opaque matmul error in the light noise vectors
        msg = (
            f'{method!r} has no per-frequency noise vectors (it only gives position power spectra,'
            ' through all_position_power_spectra): use another method for noise vectors and optical spectra'
        )
        logger.error(msg)
        raise ValueError(msg)

    def hybridised_position_noise_vector(self, label, method, ω):
        """
        The noise vector for the specific hybridised position operator qjn

        qjn(ω) = bjn(ω) + bjn†(ω)
        """
        if method.is_analytic:
            return self.analytic_hybridised_positions(method, ω)[self.label_to_index[label]]

        if method == QuantumLinearTheoryMethod.POWER_SERIES_HYBRIDISATION:
            return self.power_series_position_noise_vector(label, ω)

        # NOTE: getting to an individual position operator via any other method sadly requires
        # inefficiently calculating all positions together and then picking the desired one
        return self.hybridised_positions(method, ω)[self.label_to_index[label]]

    ################################################################################################
    # NOISE VECTORS FOR CALCULATING SPECTRA
    # ----------------------------------------------------------------------------------------------
    def homodyne_quadrature(self, a_out_hat, a_out_dag):
        """
        The output-light quadrature selected by the homodyne angle θ = `cavity.θhom`

        h(ω) = A_out(ω)·cos(θ) + P_out(ω)·sin(θ) = a_out(ω)·exp(iθ) + a_out†(ω)·exp(-iθ)
        with A_out = a_out + a_out† and P_out = i(a_out - a_out†) (the phase quadrature convention
        used in Tania's derivation; the opposite sign i(a† - a) would mirror θ → -θ).
        [same units as the noise vectors]
        """
        A_out = a_out_hat + a_out_dag
        P_out = 1j * (a_out_hat - a_out_dag)
        return A_out * np.cos(self.cavity.θhom) + P_out * np.sin(self.cavity.θhom)

    def homodyne_noise_vector(self, method, ω):
        """
        The homodyne quadrature of the output light, defined by the homodyne angle of the cavity

        h(ω) = a_out(ω)·exp(iθ) + a_out†(ω)·exp(-iθ) = A_out(ω)·cos(θ) + P_out(ω)·sin(θ)
        (see `homodyne_quadrature`), θ = `cavity.θhom`
        [1/√Hz]
        """
        return self.homodyne_quadrature(*self.output_light_noise_vectors(method, ω))

    def heterodyne_noise_vector(self, method, ω):
        """
        The output light annihilation operator, whose PSD levisim calls the heterodyne spectrum

        S_het(ω) ≡ S_aout,aout(ω) = ``〈|a_out(ω)|²〉``, as in PRR 8, Eq. (5) and below, i.e. the
        spectrum in the frame of the local oscillator (Ωhet = 0).
        NOTE: this is a single PSD, not a sum of cross-spectral densities. A real heterodyne
        measurement with the local oscillator detuned by Ωhet records the two sidebands
        S_aout,aout(Ωhet - ω) + S_aout†,aout†(ω + Ωhet) (PRR 8, Sec. IV): a sum of two PSDs at
        shifted frequencies, which is not the PSD of any single noise vector at ω. That shift is
        not applied anywhere in levisim; `cavity.Ωhet` is currently unused.
        [1/√Hz]
        """
        a_out_hat, a_out_dag = self.output_light_noise_vectors(method, ω)
        return a_out_hat

    def homodyne_and_heterodyne_noise_vectors(self, method, ω):
        """
        Returns both homodyne and heterodyne noise vectors
        (in order to only call `output_light_noise_vectors` once)

        Identical to (`homodyne_noise_vector`, `heterodyne_noise_vector`).
        [1/√Hz, 1/√Hz]
        """
        a_out_hat, a_out_dag = self.output_light_noise_vectors(method, ω)
        return self.homodyne_quadrature(a_out_hat, a_out_dag), a_out_hat

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
    def _method_branch_description(self, method):
        """
        One line saying how `method` computes the noise vectors of this system, for the DEBUG log
        """
        n_ops, n_dims = self.num_operators, self.num_dimensions
        light = 'light from a(ω) = χC(ω)[i∑Gjn·qjn(ω) + √κ·a_in(ω)]'
        if method == QuantumLinearTheoryMethod.LINEAR_AMPLIFIER_MODEL:
            return f'inverting the {n_ops}×{n_ops} matrix (-iω·I - A) at each ω, c(ω) = T(ω)·c_in(ω)'
        if method == QuantumLinearTheoryMethod.HYBRID_LINEAR_AMPLIFIER_MODEL:
            return f'positions from inverting the {n_ops}×{n_ops} matrix (-iω·I - A) at each ω; {light}'
        if method == QuantumLinearTheoryMethod.TNT_LINEAR_AMPLIFIER_MODEL:
            return f'Scc†(ω) = T(ω)·N·T(ω)† for all {n_ops} operators at once (position spectra only)'
        if method == QuantumLinearTheoryMethod.NUMERICAL_HYBRIDISATION:
            return f'inverting the {n_dims}×{n_dims} derangement matrix D = I - R numerically at each ω; {light}'
        if method == QuantumLinearTheoryMethod.POWER_SERIES_HYBRIDISATION:
            order = self.evaluation_order
            truncation = f', truncated below the full order {n_dims}, so approximate' if order < n_dims else ''
            return (
                f'D⁻¹ = adj(D)/det(D) from the power series at evaluation order {order}'
                f' ({len(self.deranged_determinant_by_order.get(order, ()))} determinant terms{truncation}); {light}'
            )
        if method == QuantumLinearTheoryMethod.ANALYTIC_INSIDE_CAVITY:
            return 'closed form (no N×N inversion), with the intracavity light from the analytic transduction functions'
        if method == QuantumLinearTheoryMethod.ANALYTIC_CAVITY_OUTPUT:
            return 'closed form (no N×N inversion), with the output light from the analytic a_out expressions'
        return 'unknown method'

    def _log_spectrum_call(self, quantity, method, allow_TNT=False, **options):
        """
        Logs the spectrum about to be calculated (INFO) and the branch `method` takes (DEBUG)

        Raises ValueError for the TNT LAM outside `all_position_power_spectra`, and AttributeError
        for the power-series method on a system built without a power series - both with a message
        that says why, instead of the TypeError or AttributeError deep inside the ω loop.
        """
        if method == QuantumLinearTheoryMethod.TNT_LINEAR_AMPLIFIER_MODEL and not allow_TNT:
            msg = f'{method!r} only gives position power spectra: use all_position_power_spectra, or another method for the {quantity}'
            logger.error(msg)
            raise ValueError(msg)
        if method == QuantumLinearTheoryMethod.POWER_SERIES_HYBRIDISATION and not hasattr(self, 'deranged_determinant_by_order'):
            msg = f'{method!r} needs a power series: build the MultiTrap with load_in_power_series=True or generate_power_series=True'
            logger.error(msg)
            raise AttributeError(msg)
        options_str = ''.join(f', {name}={value}' for name, value in options.items())
        logger.info('🧮 Calculating {} via {!r} over {} ω points{}', quantity, method, len(self.ω_range), options_str)
        logger.opt(lazy=True).debug(
            '{} for {}: {}', lambda: repr(method), lambda: self, lambda: self._method_branch_description(method),
        )

    def _warn_if_not_finite(self, quantity, method, values):
        """
        Logs a WARNING if a calculated spectrum has NaN or infinite values (e.g. ω exactly on a pole)
        """
        num_not_finite = values.size - np.count_nonzero(np.isfinite(values))
        if num_not_finite:
            logger.warning(
                '{} {} via {!r} has {} non-finite value(s) out of {}',
                self, quantity, method, num_not_finite, values.size,
            )

    @timer(TIMER)
    @profiler(PROFILER)
    @progress_bar('self.ω_range', PROGRESS_BAR)
    def homodyne_spectrum(self, method):
        """
        Calculates homodyne spectrum of output light from the cavity, S_hh

        ::

            h = a_out·exp(iθ) + a_out†·exp(-iθ) = A_out·cos(θ) + P_out·sin(θ)

        θ = homodyne angle (see `homodyne_quadrature`)
        """
        self._log_spectrum_call('homodyne spectrum', method)

        homodyne_spectrum = []
        for ii, ω in enumerate(self.ω_range):
            homodyne_spectrum.append(self.noise_vector_to_power_spectrum(self.homodyne_noise_vector(method, ω)))
        homodyne_spectrum = np.array(homodyne_spectrum)
        self._warn_if_not_finite('homodyne spectrum', method, homodyne_spectrum)
        return homodyne_spectrum

    @timer(TIMER)
    @profiler(PROFILER)
    @progress_bar('self.ω_range', PROGRESS_BAR)
    def heterodyne_spectrum(self, method):
        """
        Calculates the heterodyne spectrum of output light from the cavity, S_het

        ``S_het(ω) = S_aout,aout(ω) = 〈|a_out(ω)|²〉`` (see `heterodyne_noise_vector`)
        """
        self._log_spectrum_call('heterodyne spectrum', method)

        heterodyne_spectrum = []
        for ii, ω in enumerate(self.ω_range):
            heterodyne_value = self.noise_vector_to_power_spectrum(self.heterodyne_noise_vector(method, ω))
            # the value is allowed to be 'broken' < 1 if using the lower order approximations...
            # if method != QuantumLinearTheoryMethod.POWER_SERIES_HYBRIDISATION and self.evaluation_order != self.num_dimensions:
            #     assert heterodyne_value >= 1, f'{heterodyne_value=} < 1 - this should never happen since 1 is the shot noise limit!'
            heterodyne_spectrum.append(heterodyne_value)
        heterodyne_spectrum = np.array(heterodyne_spectrum)
        self._warn_if_not_finite('heterodyne spectrum', method, heterodyne_spectrum)
        return heterodyne_spectrum

    @timer(TIMER)
    @profiler(PROFILER)
    @progress_bar('self.ω_range', PROGRESS_BAR)
    def homodyne_and_heterodyne_spectra(self, method):
        """
        Calculates the homodyne and heterodyne spectra of output light from the cavity together

        ``S_hom(ω) = 〈|h(ω)|²〉,  S_het(ω) = 〈|a_out(ω)|²〉``
        (see `homodyne_noise_vector` and `heterodyne_noise_vector`)
        """
        self._log_spectrum_call('homodyne and heterodyne spectra', method)

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

        homodyne_spectrum, heterodyne_spectrum = np.array(homodyne_spectrum), np.array(heterodyne_spectrum)
        self._warn_if_not_finite('homodyne spectrum', method, homodyne_spectrum)
        self._warn_if_not_finite('heterodyne spectrum', method, heterodyne_spectrum)
        return homodyne_spectrum, heterodyne_spectrum


    @timer(TIMER)
    @profiler(PROFILER)
    @progress_bar('self.ω_range', PROGRESS_BAR)
    def all_position_power_spectra(self, method, zpf_scaled=True):
        """
        Invert the derangement matrix D to get the hybridisation matrix H
        This solves for the multi-dimensional hybridised position vector, x
        in terms of the 1D noise vectors, x1D ::

            D @ x = x1D
            H ≡ D⁻¹
            x = H @ x1D
        """
        self._log_spectrum_call('all position PSDs', method, allow_TNT=True, zpf_scaled=zpf_scaled)

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

        self._warn_if_not_finite('position PSDs', method, psd_array)
        return psd_array, psd_dict


    @timer(TIMER)
    @profiler(PROFILER)
    @progress_bar('self.ω_range', PROGRESS_BAR)
    def position_power_spectrum(self, label, method, zpf_scaled=True):
        """
        Calculates the power spectrum of a given dimension
        """
        self._log_spectrum_call(f'{label} position PSD', method, zpf_scaled=zpf_scaled)

        j, n = label[0], int(label[1:])
        zpf = 1 if zpf_scaled else self.dimensions[label].zpf
        psd = []
        for ω in self.ω_range:
            value = self.noise_vector_to_power_spectrum(self.hybridised_position_noise_vector(label, method, ω))
            psd.append(value)
        psd = np.array(psd) * zpf**2
        self._warn_if_not_finite(f'{label} position PSD', method, psd)
        return psd


    @timer(TIMER)
    @profiler(PROFILER)
    @progress_bar('self.ω_range', PROGRESS_BAR)
    def position_cross_spectrum(self, label_A, label_B, method, zpf_scaled=True):
        """
        Calculates the cross correlation spectrum of the given dimensions
        """
        self._log_spectrum_call(f'{label_A},{label_B} position CSD', method, zpf_scaled=zpf_scaled)

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
        csd = np.array(csd) * zpf_A * zpf_B
        self._warn_if_not_finite(f'{label_A},{label_B} position CSD', method, csd)
        return csd

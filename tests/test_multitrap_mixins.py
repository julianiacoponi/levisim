"""
The layout of the multitrap package: MultiTrap = three method mixins + MultiTrapBase
"""
from levisim.multiphysics import multitrap
from levisim.multiphysics.multitrap import MultiTrap, ΩModel
from levisim.multiphysics.multitrap.analytic_mixin import AnalyticMixin
from levisim.multiphysics.multitrap.base import MultiTrapBase
from levisim.multiphysics.multitrap.hybridisation_mixin import HybridisationMixin
from levisim.multiphysics.multitrap.linear_amplifier_mixin import LinearAmplifierMixin
from levisim.multiphysics.multitrap.omega_model import ΩModel as ΩModel_from_module
from levisim.multiphysics.multitrap.simulation_mixin import SimulationMixin


def test_public_names_and_import_paths_are_unchanged():
    assert multitrap.__all__ == ['MultiTrap', 'ΩModel']
    assert MultiTrap.__module__ == 'levisim.multiphysics.multitrap'
    assert ΩModel is ΩModel_from_module
    # module attributes of the former single-module multitrap.py
    assert multitrap.PHYSICAL_PER_TRAP_KWARGS == ('tweezers', 'beads', 'trap_kwargs')
    assert {multitrap.TIMER, multitrap.PROFILER, multitrap.PROGRESS_BAR} <= {True, False}


def test_multitrap_is_the_three_mixins_the_simulation_mixin_and_the_base():
    assert MultiTrap.__bases__ == (LinearAmplifierMixin, HybridisationMixin, AnalyticMixin, SimulationMixin, MultiTrapBase)


def owner(name):
    """the class in MultiTrap's MRO that defines `name`"""
    return next(klass for klass in MultiTrap.__mro__ if name in vars(klass))


def test_each_method_family_lives_in_its_mixin_and_the_dispatch_on_the_base():
    expected = {
        LinearAmplifierMixin: ['setup_linear_amplifier_model_matrices', 'T_matrix', 'c_noise_vector',
                               'quantum_operator_spectral_density_array', 'position_power_spectra_array'],
        HybridisationMixin: ['setup_hybridisation', 'derangement_matrix', 'hybridisation_matrix',
                             'deranged_determinant', 'deranged_adjugate_matrix_entry', 'x1D_noise_vector'],
        AnalyticMixin: ['gbar', 'M_norm', 'D', 'P', 'Fa', 'FFa_dag', 'FQ', 'analytic_kernel'],
        SimulationMixin: ['simulate'],
        MultiTrapBase: ['light_noise_vectors', 'output_light_noise_vectors', 'hybridised_positions',
                        'hybridised_position_noise_vector', 'homodyne_spectrum', 'all_position_power_spectra',
                        'noise_vector_to_power_spectrum', 'noise_vectors_to_cross_spectrum'],
    }
    for klass, names in expected.items():
        for name in names:
            assert owner(name) is klass, (name, owner(name))
    # no name is defined on more than one class of the MRO (nothing is silently overridden)
    seen = {}
    for klass in MultiTrap.__mro__[1:-1]:
        for name in vars(klass):
            if not name.startswith('__'):
                assert name not in seen, (name, klass, seen.get(name))
                seen[name] = klass

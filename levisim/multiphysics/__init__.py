"""
levisim.multiphysics
===============
A module to handle all the physics involved in multitrap cavity levitation
"""

# Every submodule below declares its own __all__, so these wildcard imports have a
# bounded, declared surface. __all__ here restates it so that `from levisim.multiphysics
# import *` and dir() agree. NOTE: the lazily-built shelf fixtures are deliberately
# absent - see the __getattr__ forwarder at the bottom of this file.
__all__ = [
    'CavityGeometry',
    'DEFAULT_cavity_kwargs',
    'DEFAULT_gas_kwargs',
    'DEFAULT_Ω_model_pairs',
    'DEFAULT_ω_range',
    'DEPLANO2025',
    'Dimension',
    'FORTRAN_toy_params',
    'FORTRAN_Ω_model_pairs',
    'FORTRAN_ω_range',
    'HybridisationTerm',
    'MultiToneDrive',
    'MultiTrap',
    'NoiseVectorType',
    'OFF_NODE_toy_params_equal_spacing_simple',
    'OFF_NODE_toy_params_pairs',
    'PHASE_DEPENDENT_toy_params',
    'PHASE_EQUAL_toy_params',
    'Parameter',
    'ParameterType',
    'PeriodicSteadyState',
    'QuadraticPotential',
    'QuantumLinearTheoryMethod',
    'RelativeDifferenceTooHigh',
    'SpectrumType',
    'Tone',
    'TooManyErrantPixelsInImageDiff',
    'TrapModel',
    'TweezerOptics',
    'TweezerTone',
    'TwoToneSteadyState',
    'air_pressure',
    'all_parameters',
    'analytic_log_negativity',
    'analytic_optimum',
    'analytic_threshold_cooperativity',
    'animate_spectra',
    'assert_arrays_agree_within_threshold',
    'assert_images_agree_within_errant_pixels',
    'bogoliubov_coefficients',
    'bogoliubov_occupations',
    'calculate_phonons_and_temperatures_from_PSD',
    'calculate_spectra_across_y_range_for_one_frame',
    'calculate_spectra_for_all_frames',
    'calculate_temperature_from_position_PSD',
    'cavity_detuning',
    'common_period',
    'complex_to_quadrature_transform',
    'delete_local_png_files',
    'deplano2025_multitrap',
    'deplano2025_tweezer',
    'dimension_covariance',
    'evaluation_order',
    'find_resonances',
    'generate_all_hybridisation_functions',
    'generate_deranged_terms',
    'generate_determinant_and_adjugate_by_order',
    'generate_heatmap_animation_frames',
    'generate_slices_animation_frames',
    'generate_subscripts_and_labels',
    'get_latex_labels',
    'greyscale_diff',
    'latex_ω_units',
    'light_mech_coupling',
    'mode_index',
    'multitone_drive',
    'noise_vector_latex',
    'noise_vector_string',
    'optical_damping',
    'phase_of_2nd_tweezer',
    'phase_per_tweezer_string',
    'plot_mean_abs_relative_diffs_matrix',
    'plot_pass_fail_matrix',
    'principal_axes_log_negativity',
    'quadrature_diffusion_matrix',
    'quadrature_drift_matrix',
    'quantum_cooperativity',
    'relative_tweezer_phases_string',
    'sieve_power_series',
    'sieve_power_series_for_sequence',
    'spectrum_type_to_latex',
    'spectrum_type_to_units',
    'steady_state_covariance',
    'subsystem_log_negativity',
    'symmetric_group_with_min_fixed_points',
    'term_latex',
    'term_string',
    'title_string_for_one_frame',
    'to_rotating_frame',
    'truncate_power_series',
    'tweezer_trap',
    'two_tone_drive',
    'two_tone_rwa_steady_state',
    'two_tone_steady_state',
    'two_tone_tones',
    'verify_equivalence_of_spectral_functions',
    'verify_equivalence_of_spectrum_calculation_methods',
    'visualise_relative_difference_of_two_arrays',
    'visualise_two_noise_component_calculation_methods',
    'visualise_two_noise_component_calculation_methods_over_entire_spectrum',
    'visualise_two_spectrum_calculation_methods',
]

from levisim.multiphysics.multitrap import *
from levisim.multiphysics.dimension import *
from levisim.multiphysics.hybridisation import *
from levisim.multiphysics.enums import *
from levisim.multiphysics.utils import *
from levisim.multiphysics.covariance import *
from levisim.multiphysics.multitone import *
from levisim.multiphysics.two_tone import *
from levisim.multiphysics.two_tone_traps import *


def __getattr__(name):
    """
    Forwards the lazily-built `shelf` fixtures (PEP 562)

    `FORTRAN_8x2`, `PHASE_EQUAL_xy_dict` and `PHASE_DEPENDENT_xy_dict` are built on first
    access rather than at import time, so they are not re-exported by the `import *` above.
    Forwarding them here keeps `from levisim.multiphysics import FORTRAN_8x2` working as it always did.
    """
    from levisim.multiphysics.utils import shelf
    if name in shelf._LAZY_ATTRIBUTES:
        return getattr(shelf, name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')

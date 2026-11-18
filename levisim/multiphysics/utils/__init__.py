"""
levisim.multiphysics.utils
===============
A module for helpful utilities for plotting and analysing multiphysics
"""

# Every submodule below declares its own __all__, so these wildcard imports have a
# bounded, declared surface. __all__ here restates that surface, so `from levisim.multiphysics.utils import *`
# and dir() agree and a reader can see what the package exports without opening each file.
__all__ = [
    'DEFAULT_cavity_kwargs',
    'DEFAULT_gas_kwargs',
    'DEFAULT_Ω_model_pairs',
    'DEFAULT_ω_range',
    'FORTRAN_toy_params',
    'FORTRAN_Ω_model_pairs',
    'FORTRAN_ω_range',
    'OFF_NODE_toy_params_equal_spacing_simple',
    'OFF_NODE_toy_params_pairs',
    'PHASE_DEPENDENT_toy_params',
    'PHASE_EQUAL_toy_params',
    'RelativeDifferenceTooHigh',
    'TooManyErrantPixelsInImageDiff',
    'animate_spectra',
    'assert_arrays_agree_within_threshold',
    'assert_images_agree_within_errant_pixels',
    'calculate_phonons_and_temperatures_from_PSD',
    'calculate_spectra_across_y_range_for_one_frame',
    'calculate_spectra_for_all_frames',
    'calculate_temperature_from_position_PSD',
    'delete_local_png_files',
    'find_resonances',
    'generate_heatmap_animation_frames',
    'generate_slices_animation_frames',
    'get_latex_labels',
    'greyscale_diff',
    'latex_ω_units',
    'noise_vector_latex',
    'noise_vector_string',
    'phase_per_tweezer_string',
    'plot_mean_abs_relative_diffs_matrix',
    'plot_pass_fail_matrix',
    'relative_tweezer_phases_string',
    'spectrum_type_to_latex',
    'spectrum_type_to_units',
    'title_string_for_one_frame',
    'verify_equivalence_of_spectral_functions',
    'verify_equivalence_of_spectrum_calculation_methods',
    'visualise_relative_difference_of_two_arrays',
    'visualise_two_noise_component_calculation_methods',
    'visualise_two_noise_component_calculation_methods_over_entire_spectrum',
    'visualise_two_spectrum_calculation_methods',
]

from levisim.multiphysics.utils.animations import *
from levisim.multiphysics.utils.comparison import *
from levisim.multiphysics.utils.shelf import *


def __getattr__(name):
    """
    Forwards the lazily-built `shelf` fixtures (PEP 562)

    `FORTRAN_8x2`, `PHASE_EQUAL_xy_dict` and `PHASE_DEPENDENT_xy_dict` are built on first
    access rather than at import time, so they are not re-exported by the `import *` above.
    Forwarding them here keeps `from levisim.multiphysics.utils import FORTRAN_8x2` working as it always did.
    """
    from levisim.multiphysics.utils import shelf
    if name in shelf._LAZY_ATTRIBUTES:
        return getattr(shelf, name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')

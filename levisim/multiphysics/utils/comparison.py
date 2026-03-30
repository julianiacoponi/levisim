import os
import cv2
import itertools
import matplotlib
import scipy

import numpy as np
import matplotlib.pyplot as plt

from levisim.core.logger import logger
from levisim.core.constants import ħ, kB
from levisim.core.units import degrees, Hz, kHz
from levisim.core.plotting import set_plt_rcParams_to_xmgrace_style, LATEX_FIG_WIDTH, get_latex_doc_lims
from levisim.core.colours import (
    all_colours_list,
    distinct_16_colours_list,
    blue, orange, green, red, purple,
    black, white, grey,
    RGB_red, RGB_green, RGB_yellow, RGB_magenta,
)

from levisim.multiphysics.enums.quantum_linear_theory_method import QuantumLinearTheoryMethod
from levisim.multiphysics.enums.spectrum import SpectrumType
from levisim.multiphysics.enums.noise_vector import NoiseVectorType


__all__ = [
    # custom exceptions
    'TooManyErrantPixelsInImageDiff',
    'RelativeDifferenceTooHigh',

    # verification functions (run tests)
    'verify_equivalence_of_spectrum_calculation_methods',
    'verify_equivalence_of_spectral_functions',

    # test functions (raise exceptions)
    'test_pairs_of_images_for_errant_pixels',
    'test_pairs_of_arrays_for_relative_difference',

    # visualisation functions (show plots)
    'visualise_two_spectrum_calculation_methods',
    'visualise_two_noise_component_calculation_methods',
    'visualise_two_noise_component_calculation_methods_over_entire_spectrum',
    'visualise_relative_difference_of_two_arrays',

    # results matrix plotting
    'plot_pass_fail_matrix',
    'plot_mean_abs_relative_diffs_matrix',

    # helpers
    'greyscale_diff',
    'delete_local_png_files',

    'calculate_temperature_from_position_PSD',

    'get_latex_labels',

    'noise_vector_string',
    'noise_vector_latex',

    'phase_per_tweezer_string',
    'relative_tweezer_phases_string',

    'find_resonances',
]


def test_thresholds(difficulty_level):
    """
    Returns tuple of
        (
            errant_pixel_allowed_percentage,
            relative_diff_threshold,
            relative_diff_allowed_failed_points_percentage,
        )
    """
    match difficulty_level:
        case 'very easy':
            msg = f'👶 Setting {difficulty_level=}'
            print(msg), logger.info(msg)
            # 1e-3 relative absolute diff is equivalent to 0.1% error
            return (0.1, 1e-3, 0.5)
        case 'easy':
            msg = f'🙂 Setting {difficulty_level=}'
            print(msg), logger.info(msg)
            return (0.05, 1e-6, 0.5)
        case 'hard':
            msg = f'😰 Setting {difficulty_level=}'
            print(msg), logger.info(msg)
            return (0.01, 1e-10, 0.25)
        case 'very hard':
            msg = f'🤬 Setting {difficulty_level=}'
            print(msg), logger.info(msg)
            return (0.005, 1e-12, 0.1)


# --------------------------------------------------------------------------------------------------
# CUSTOM EXCEPTIONS
# --------------------------------------------------------------------------------------------------
class TooManyErrantPixelsInImageDiff(Exception):
    pass

class RelativeDifferenceTooHigh(Exception):
    pass


# --------------------------------------------------------------------------------------------------
# VERIFICATION FUNCTIONS
# --------------------------------------------------------------------------------------------------
def verify_equivalence_of_spectrum_calculation_methods(mtrap, spectrum_types, methods, **kwargs):
    """
    Calculate all `spectrum_types` via all `methods`
    Verify their equivalnece by
        - checking for sufficently few errant pixels in a greyscale image diff
        - checking the relative absolute difference is sufficiently low
    """
    set_plt_rcParams_to_xmgrace_style()

    msg = f'⚖️ Using {mtrap} ({mtrap.tweezer_phase_sign_convention_str}), verifying calculation of {spectrum_types=} using {methods=} via image diff'
    print(msg), logger.info(msg)

    xlims = mtrap.ω_range.min() / kHz, mtrap.ω_range.max() / kHz
    yscale = kwargs.get('yscale', 'log')

    all_image_dicts = {}
    all_values_dict = {}
    for spectrum_type in spectrum_types:
        all_image_dicts[spectrum_type] = {}
        all_values_dict[spectrum_type] = {}
        for method in methods:

            # optical spectra cannot be calculated with TNT method since it does not account for input-output theory
            if spectrum_type.is_optical and method == QuantumLinearTheoryMethod.TNT_LINEAR_AMPLIFIER_MODEL:
                continue

            msg = f'🎨 Plotting {spectrum_type=} using {method=}'
            print(msg), logger.info(msg)

            match spectrum_type:
                case SpectrumType.HOMODYNE:
                    label = repr(spectrum_type)
                    vals = mtrap.homodyne_spectrum(method)
                    ylabel = '[photons/Hz]'
                    ylims = kwargs.get('ylims', (1e-3, 1e6))
                case SpectrumType.HETERODYNE:
                    label = repr(spectrum_type)
                    vals = mtrap.heterodyne_spectrum(method)
                    ylabel = '[photons/Hz]'
                    ylims = kwargs.get('ylims', (1e-3, 1e6))
                case SpectrumType.POSITION_PSD:
                    label = get_latex_labels(mtrap, PSD=True)
                    zpf_scaled = kwargs.get('zpf_scaled', True)
                    vals, _ = mtrap.all_position_power_spectra(method, zpf_scaled=zpf_scaled)
                    ylabel = '[phonons/Hz]' if zpf_scaled else '[m²/Hz]'
                    ylims = kwargs.get('ylims', (1e-9, 1e6))

                    if kwargs.get('separate_PSDs', False):
                        # if mtrap.num_traps >= 3 and mtrap.Cartesian_axes_per_trap >= 2:
                        mosaic = [[dim_label] for dim_label in mtrap.labels]
                        fig = plt.figure(figsize=(LATEX_FIG_WIDTH, 3.0 * len(mosaic)), dpi=400)
                        ax_dict = fig.subplot_mosaic(mosaic)
                        for ii, ax in enumerate(ax_dict.values()):
                            ax.set_title(rf'zpf-scaled {label[ii]}')
                            ax.set_xlabel('Frequency [kHz]')
                            ax.set_ylabel(ylabel)
                            ax.set_xlim(*xlims)
                            ax.set_ylim(1e-7, 1e1)
                            ax.vlines([_.Ω / kHz for _ in mtrap.dimensions.values()], *ylims, label=r'All $\Omega_j$', color=black, alpha=0.1, ls='--')
                            ax.vlines([-_.Ω / kHz for _ in mtrap.dimensions.values()], *ylims, color=black, alpha=0.1, ls='--')
                            dim = mtrap.dimensions[mtrap.labels[ii]]
                            ax.vlines(dim.Ω / kHz, *ylims, label=dim.Ωj_str, color=RGB_red, alpha=0.5, ls='--', lw=1.25)
                            ax.plot(mtrap.ω_range / kHz, vals[:, ii], color=colours_50[ii])
                            ax.set_yscale(yscale)
                            ax.legend(loc='upper right')
                            ax.grid(alpha=0.1)
                        plt.show()

            all_values_dict[spectrum_type][repr(method)] = vals

            fig_kwargs = kwargs.get('fig_kwargs', dict(figsize=(LATEX_FIG_WIDTH, 3.5), dpi=400))
            fig, ax = plt.subplots(**fig_kwargs)
            ax.set_title(kwargs.get('title', ''))
            ax.set_xlabel('Frequency [kHz]')
            ax.set_ylabel(ylabel)
            ax.set_xlim(*xlims)
            ax.set_ylim(*ylims)
            ax.vlines([_.Ω / kHz for _ in mtrap.dimensions.values()], *ylims, label=r'All $\Omega_j$', color=black, alpha=0.1, ls='--')
            ax.vlines([-_.Ω / kHz for _ in mtrap.dimensions.values()], *ylims, color=black, alpha=0.1, ls='--')
            if spectrum_type.is_optical:
                ax.hlines(1, *xlims, color=black, alpha=0.1)
                ax.plot(mtrap.ω_range / kHz, vals, label=label)
                ax.set_yscale(yscale)
            else:
                for ii in range(mtrap.num_dimensions):
                    ax.plot(mtrap.ω_range / kHz, vals[:, ii], label=label[ii], color=colours_50[ii])
                    ax.set_yscale(yscale)
            ax.legend(loc='upper left', prop=dict(size=8), ncol=1 + mtrap.num_dimensions // 8)
            ax.grid(alpha=0.1)

            figname = f'{str(spectrum_type)} -- {repr(method)}.png'
            plt.savefig(figname)
            all_image_dicts[spectrum_type][repr(method)] = cv2.imread(figname)
            if kwargs.get('show', False):
                plt.show()
            else:
                plt.close(fig)

    if len(methods) > 1:
        for image_dict in all_image_dicts.values():
            msg = f'🔦 Checking for errant pixels in the image diffs of all plots of {spectrum_type=}'
            print(msg), logger.info(msg)
            test_pairs_of_images_for_errant_pixels(image_dict, **kwargs)

        for spectrum_type, array_dict in all_values_dict.items():
            msg = f'🔦 Checking the relative difference is sufficiently small in all pairs of arrays of {spectrum_type=}'
            print(msg), logger.info(msg)
            test_pairs_of_arrays_for_relative_difference(mtrap, array_dict, **kwargs)

        msg = f'🎉🎉🎉🎉🎉 All spectral calculation methods are equivalent for {mtrap} ☺️ 🎉🎉🎉🎉🎉'
        print(msg), logger.info(msg)

    else:
        # otherwise this is only called inside the test functions
        delete_local_png_files()

    if kwargs.get('return_PSD_temperatures', False) and spectrum_type == SpectrumType.POSITION_PSD:
        temperatures = {}
        for ii, label in enumerate(mtrap.labels):
            temperatures[label] = calculate_temperature_from_position_PSD(mtrap, vals, ii, label)
        return temperatures


def verify_equivalence_of_spectral_functions(mtrap, func_dict, **kwargs):
    """
    Calculate all functions in `func_dict` across the given `mtrap.ω_range`
    Verify their equivalnece by
        - checking for sufficently few errant pixels in a greyscale image diff of their plots
        - checking the relative absolute difference is sufficiently low
    """
    set_plt_rcParams_to_xmgrace_style()

    msg = f'⚖️ Using {mtrap} ({mtrap.tweezer_phase_sign_convention_str}), verifying spectral {func_dict=} via image diff'
    print(msg), logger.info(msg)

    ω_range = mtrap.ω_range
    image_dict = {}
    all_values_dict = {}
    for func_key, func in func_dict.items():
        msg = f'🎨 Plotting {func_key=}'
        print(msg), logger.info(msg)
        vals = np.array([func(ω) for ω in ω_range])

        all_values_dict[func_key] = vals

        xlims = ω_range.min() / kHz, ω_range.max() / kHz
        ylim = 1.2 * np.abs(vals).max()
        yscale = kwargs.get('yscale', 'log')
        ylims = (-ylim, +ylim) if yscale == 'linear' else (1e-3 * np.abs(vals).min(), 1e3 * ylim)

        fig, ax = plt.subplots(**kwargs.get('fig_kwargs', dict(figsize=(LATEX_FIG_WIDTH, 3.5), dpi=400)))
        ax.set_title(kwargs.get('title', ''))
        ax.set_xlabel('Frequency [kHz]')
        ax.set_ylabel(kwargs.get('ylabel', ''))
        ax.set_xlim(*xlims)
        ax.set_ylim(*ylims)
        ax.plot(ω_range / kHz, np.real(vals), label='Re', alpha=0.5, lw=2, ls='--')
        ax.plot(ω_range / kHz, np.imag(vals), label='Im', alpha=0.5, lw=2, ls='--')
        ax.plot(ω_range / kHz, np.abs(vals), label='Abs', alpha=0.3, lw=1.5)
        ax.hlines(0, *xlims, color=black, alpha=0.1)
        ax.vlines([_.Ω / kHz for _ in mtrap.dimensions.values()], *ylims, label=r'All $\Omega_j$', color=black, alpha=0.1, ls='--')
        ax.set_yscale(yscale)
        ax.grid(alpha=0.1)
        ax.legend(loc='lower right', ncol=2)

        figname = f'{func_key}.png'
        fig.savefig(figname)
        image_dict[func_key] = cv2.imread(figname)
        if kwargs.get('show', False):
            plt.show()
        else:
            plt.close(fig)

    test_pairs_of_arrays_for_relative_difference(mtrap, all_values_dict, **kwargs)
    test_pairs_of_images_for_errant_pixels(image_dict, **kwargs)

    msg = f'🎉🎉🎉🎉🎉 All functions are equivalent for {mtrap}! ☺️ 🎉🎉🎉🎉🎉'
    print(msg), logger.info(msg)


# --------------------------------------------------------------------------------------------------
# TEST FUNCTIONS
# --------------------------------------------------------------------------------------------------
def test_pairs_of_images_for_errant_pixels(image_dict, **kwargs):
    """
    Test the greyscale diff of all pairs of images in `image_dict` for errant pixels
    """
    set_plt_rcParams_to_xmgrace_style()

    image_diff_results_by_image_pair = {}
    image_names = list(image_dict.keys())
    num_names = len(image_names)
    pass_fail_matrix = np.zeros((num_names, num_names), dtype=bool)

    errant_pixel_allowed_percentage, _, _ = test_thresholds(kwargs.get('difficulty_level', 'very hard'))
    msg = (
        f'🔬 Testing all pairs of image diffs of {image_names=}'
        f' for errant pixel percentage below {errant_pixel_allowed_percentage:.3f}%'
    )
    print(msg), logger.info(msg)

    all_results_msg = '🗂️ ALL IMAGE DIFF RESULTS (🟩 = success, 🟥 = failure):'
    for pair in itertools.combinations(image_names, 2):
        image_A, image_B = image_dict[pair[0]], image_dict[pair[1]]
        # greyscale image diff
        greyscale_image_diff = cv2.cvtColor(cv2.subtract(image_A, image_B), cv2.COLOR_BGR2GRAY)
        errant_pixels = cv2.countNonZero(greyscale_image_diff)
        success = errant_pixels == 0

        h, w, _ = image_A.shape
        total_pixels = h * w
        errant_pixels_percentage = 100 * errant_pixels / total_pixels

        if not success:
            msg = (
                f'⚠️⚠️⚠️⚠️⚠️ WARNING! ⚠️⚠️⚠️⚠️⚠️'
                f'\n{pair=} image diff has {errant_pixels=}'
                f'\n📊 IMAGE DIFF STATISTICS:'
                f'\n    {image_A.shape=}'
                f'\n    image_A has {total_pixels=}'
                f'\n    {errant_pixels_percentage=:.3f}% = {errant_pixels}/{total_pixels}'
                f'\n    {errant_pixel_allowed_percentage=:.1f}%'
            )
            print(msg), logger.warning(msg)
            success = errant_pixels_percentage < errant_pixel_allowed_percentage

            # https://stackoverflow.com/questions/39994831/difference-between-cv2-findnonzero-and-numpy-nonzero#39996246
            errant_pixel_locations = np.nonzero(greyscale_image_diff)
            errant_pixel_y, errant_pixel_x = errant_pixel_locations[0], errant_pixel_locations[1]

            plt.figure(dpi=500)
            plt.imshow(image_A)
            [plt.axhline(_, lw=0.1, color=RGB_magenta, alpha=0.5) for _ in errant_pixel_locations[0]]
            [plt.axvline(_, lw=0.1, color=RGB_magenta, alpha=0.5) for _ in errant_pixel_locations[1]]
            plt.title(f'image_A = "{pair[0]}"\n(errant pixel coordinates indicated with lines)'), plt.axis('off'), plt.show()

            plt.figure(dpi=500)
            plt.imshow(image_B)
            [plt.axhline(_, lw=0.1, color=RGB_magenta, alpha=0.5) for _ in errant_pixel_locations[0]]
            [plt.axvline(_, lw=0.1, color=RGB_magenta, alpha=0.5) for _ in errant_pixel_locations[1]]
            plt.title(f'image_B = "{pair[1]}"\n(errant pixel coordinates indicated with lines)'), plt.axis('off'), plt.show()

            plt.figure(dpi=500)
            plt.imshow(greyscale_image_diff)
            plt.axhspan(errant_pixel_y.min(), errant_pixel_y.max(), color=RGB_yellow, alpha=0.2)
            plt.axvspan(errant_pixel_x.min(), errant_pixel_x.max(), color=RGB_yellow, alpha=0.2)
            plt.title(f'greyscale image diff = "{pair[0]}" vs. "{pair[1]}"\n(errant pixels inside highlighted region)'), plt.axis('off'), plt.show()

        image_diff_results_by_image_pair[pair] = success
        index_A, index_B = image_names.index(pair[0]), image_names.index(pair[1])
        # only fill lower triangle
        pass_fail_matrix[index_B, index_A] = success

        msg = f'🟩 {pair=}' if success else f'🟥 {pair=}'
        all_results_msg += f'\n    {msg}'

    print(all_results_msg), logger.info(all_results_msg)

    delete_local_png_files()
    overall_failure = not all(image_diff_results_by_image_pair.values())

    # show table if there is a failure or on total success if this kwarg is present
    if kwargs.get('show_pass_fail_matrix_on_success', overall_failure):
        title = 'PASS/FAIL matrix for\nerrant pixel check in image diffs'
        plot_pass_fail_matrix(pass_fail_matrix, image_names, **kwargs)

    if overall_failure:
        msg = f'🏴‍☠️🏴‍☠️🏴‍☠️🏴‍☠️🏴‍☠️ Failure present! 🏴‍☠️🏴‍☠️🏴‍☠️🏴‍☠️🏴‍☠️\n{image_diff_results_by_image_pair=}'
        print(msg), logger.error(msg)
        # NOTE: do not use this option if calling this test function within a try:except block
        # that relies on catching raised exceptions :-)
        if kwargs.get('fail_quietly', False):
            pass
        else:
            raise TooManyErrantPixelsInImageDiff(f'{image_diff_results_by_image_pair=}')

    msg = f'✅ Success on all errant pixel checks!'
    print(msg), logger.info(msg)


def test_pairs_of_arrays_for_relative_difference(mtrap, array_dict, **kwargs):
    """
    Test the relative difference of all pairs of arrays in `array_dict` is below threshold
    """
    set_plt_rcParams_to_xmgrace_style()

    successes_by_pair = {}
    array_names = list(array_dict.keys())
    num_names = len(array_names)
    pass_fail_matrix = np.zeros((num_names, num_names), dtype=bool)
    mean_abs_relative_diffs_matrix = np.zeros((num_names, num_names), dtype=complex)

    _, relative_diff_threshold, allowed_failed_points_percentage = test_thresholds(kwargs.get('difficulty_level', 'very hard'))
    msg = (
        f'🔬 Testing that all pairs of {array_names=}'
        f' have mean relative absolute difference below {relative_diff_threshold=:.0e}'
    )
    print(msg), logger.info(msg)

    all_results_msg = '🗂️ ALL RELATIVE DIFF RESULTS (🟢 = success, 🔴 = failure):'
    for pair in itertools.combinations(array_names, 2):
        array_A, array_B = array_dict[pair[0]], array_dict[pair[1]]
        diffs = array_B - array_A
        abs_relative_diffs = np.abs(diffs) / np.abs(array_A)

        failed_points = abs_relative_diffs[abs_relative_diffs > relative_diff_threshold]
        num_failed_points = len(failed_points)

        success = np.mean(abs_relative_diffs) < relative_diff_threshold

        # either pass the test quietly (there are no failed points)
        # or warn about some failed points, and pass on later condition
        # TODO: the failed points tend to be on resonances so could make the number of dimensions
        # a threshold for number of failed points
        if num_failed_points:
            num_array_points = len(array_A)

            if len(array_A.shape) == 2:
                # sum all the array points across all PSDs
                num_array_points = array_A.shape[0] * array_A.shape[1]

            failed_points_percentage = 100 * num_failed_points / num_array_points

            failed_point_indices = np.where(abs_relative_diffs > relative_diff_threshold)

            # NOTE: this indicates the arrays are all position PSDs bundled together from `all_position_power_spectra`
            if len(abs_relative_diffs.shape) == 2:
                num_positions = abs_relative_diffs.shape[1]
                assert num_positions == mtrap.num_dimensions, f'❓ Expecting value for {abs_relative_diffs.shape=} to be {mtrap.num_dimensions=}'

                failed_ω_str = ''
                for ii, failed_point in enumerate(failed_points):
                    failed_ω = mtrap.ω_range[failed_point_indices[0][ii]]
                    jn = mtrap.index_to_label[failed_point_indices[1][ii]]
                    failed_ω_str += f'{jn} position PSD has {failed_point=:.5e} @ {failed_ω / kHz:.3f} kHz'
                    if ii != num_failed_points:
                        failed_ω_str += '\n    '

            elif len(abs_relative_diffs.shape) == 1:
                failed_ω_values = mtrap.ω_range[failed_point_indices]
                failed_ω_str = f'{failed_ω_values / kHz=}'

            else:
                msg = f'❓ Expecting either 1 or 2 for length of {abs_relative_diffs.shape=}'
                print(msg), logger.error(msg)
                failed_ω_str = msg

            msg = (
                f'⚠️⚠️⚠️⚠️⚠️ WARNING! ⚠️⚠️⚠️⚠️⚠️'
                f'\n{pair=} has {num_failed_points=} greater than {relative_diff_threshold=:.0e}'
                f' out of {num_array_points=}'
                f'\n📊 RELATIVE DIFF STATISTICS:'
                f'\n    {failed_points_percentage=:.3f}% = {num_failed_points}/{num_array_points}'
                f'\n    {allowed_failed_points_percentage=:.3f}%'
                f'\n    {failed_points=}'
                f'\n    {failed_ω_str}'
                f'\n    {np.max(abs_relative_diffs)=:.2e}'
                f'\n    {np.mean(abs_relative_diffs)=:.2e}'
                f'\n    {np.median(abs_relative_diffs)=:.2e}'
            )
            print(msg), logger.warning(msg)

            visualise_relative_difference_of_two_arrays(mtrap, array_A, array_B, pair[0], pair[1], **kwargs)

            # pass the test if there is a sufficiently small proportion of failing points
            # even if the mean is above threshold
            if not success:
                success = failed_points_percentage < allowed_failed_points_percentage

        successes_by_pair[pair] = success
        index_A, index_B = array_names.index(pair[0]), array_names.index(pair[1])
        # only fill lower triangle
        pass_fail_matrix[index_B, index_A] = success
        mean_abs_relative_diffs_matrix[index_B, index_A] = np.mean(abs_relative_diffs)

        msg = f'🟢 {pair=}' if success else f'🔴 {pair=}'
        all_results_msg += f'\n    {msg}'

    print(all_results_msg), logger.info(all_results_msg)
    overall_failure = not all(successes_by_pair.values())

    # show table if there is a failure or on total success if this kwarg is present
    if kwargs.get('show_pass_fail_matrix_on_success', overall_failure):
        kwargs['pass_fail_matrix_title'] = 'PASS/FAIL matrix for\n relative absolute difference'
        plot_pass_fail_matrix(pass_fail_matrix, array_names, **kwargs)
        plot_mean_abs_relative_diffs_matrix(mean_abs_relative_diffs_matrix, array_names, **kwargs)

    if overall_failure:
        msg = f'🏴‍☠️🏴‍☠️🏴‍☠️🏴‍☠️🏴‍☠️ Failure present! 🏴‍☠️🏴‍☠️🏴‍☠️🏴‍☠️🏴‍☠️\n{successes_by_pair=}'
        print(msg), logger.error(msg)
        # NOTE: do not use this option if calling this test function within a try:except block
        # that relies on catching raised exceptions :-)
        if kwargs.get('fail_quietly', False):
            pass
        else:
            raise RelativeDifferenceTooHigh(f'{successes_by_pair=}')

    msg = f'✅ Success on all relative diff checks!'
    print(msg), logger.info(msg)


# --------------------------------------------------------------------------------------------------
# VISUALISATION FUNCTIONS
# --------------------------------------------------------------------------------------------------
def visualise_two_spectrum_calculation_methods(
    mtrap, spectrum_type,
    method_A, method_B,
    **kwargs,
):
    """
    Visualise a given `spectrum_type` calculated via `method_A` vs. `method_B`
    """
    set_plt_rcParams_to_xmgrace_style()

    msg = f'🖼️ Using {mtrap} ({mtrap.tweezer_phase_sign_convention_str}) to visualise the calculation of {spectrum_type=} via {method_A=} vs. {method_B=}'
    print(msg), logger.info(msg)

    fig_kwargs = dict(figsize=(LATEX_FIG_WIDTH, 3.5), dpi=400)
    legend_kwargs = dict(loc='upper left', prop=dict(size=8), ncol=2)
    zpf_scaled = kwargs.get('zpf_scaled', True)

    ω_range = mtrap.ω_range
    print('Tweezer phases:')
    print(phase_per_tweezer_string(mtrap))
    print('Relative tweezer phases:')
    print(relative_tweezer_phases_string(mtrap))

    match spectrum_type:
        case SpectrumType.HOMODYNE:
            vals_A = mtrap.homodyne_spectrum(method_A)
            vals_B = mtrap.homodyne_spectrum(method_B)
            title_str = r'$S_\mathregular{homodyne}(\omega)$'
            ylabel = 'Optical PSD [photons/Hz]'
        case SpectrumType.HETERODYNE:
            vals_A = mtrap.heterodyne_spectrum(method_A)
            vals_B = mtrap.heterodyne_spectrum(method_B)
            title_str = r'$S_\mathregular{heterodyne}(\omega)$'
            ylabel = 'Optical PSD [photons/Hz]'
        case SpectrumType.POSITION_PSD:
            dim_label = kwargs.get('dim_label', mtrap.index_to_label[0])
            vals_A = mtrap.position_power_spectrum(dim_label, method_A, zpf_scaled=zpf_scaled)
            vals_B = mtrap.position_power_spectrum(dim_label, method_B, zpf_scaled=zpf_scaled)
            dim = mtrap.dimensions[dim_label]
            title_str = (
                f'PSD of ${dim.latex_hat}$ position\n'
                rf'${dim.latex_PSD}(\omega)$'
            )
            ylabel = rf'[${dim.latex}$ phonons/Hz]' if zpf_scaled else '[m²/Hz]'
        case SpectrumType.POSITION_CSD:
            dim_label_A = kwargs.get('dim_label_A', mtrap.index_to_label[0])
            dim_label_B = kwargs.get('dim_label_B', mtrap.index_to_label[1])
            vals_A = mtrap.position_cross_spectrum(dim_label_A, dim_label_B, method_A, zpf_scaled=zpf_scaled)
            vals_B = mtrap.position_cross_spectrum(dim_label_A, dim_label_B, method_B, zpf_scaled=zpf_scaled)
            j, n = dim_label_A[0], dim_label_A[1:]
            k, m = dim_label_B[0], dim_label_B[1:]
            jn_string = f'{j}_{n}'
            km_string = f'{k}_{m}'
            jn_km_string = f'{jn_string}{km_string}'
            title_str = (
                f'CSD of ${jn_string}$ and ${km_string}$ positions\n'
                rf'$S_{{{jn_km_string}}}(\omega)$'
            )
            dim_A, dim_B = mtrap.dimensions[dim_label_A], mtrap.dimensions[dim_label_B]
            ylabel = rf'[$\sqrt{{{dim_A.latex + dim_B.latex}}}$ phonons/Hz]' if zpf_scaled else '[m²/Hz]'

    xlims = np.min(ω_range) / kHz, np.max(ω_range) / kHz
    ylims = 1e-1 * min(np.abs(vals_A).min(), np.abs(vals_B).min()), 1e1 * max(np.abs(vals_A).max(), np.abs(vals_B).max())

    plt.figure(**fig_kwargs)
    plt.title(title_str + kwargs.get('title', ''))
    plt.xlabel('Frequency [kHz]')
    plt.ylabel(ylabel)
    plt.xlim(*xlims)
    plt.ylim(*ylims)
    plt.text(0.1, 0.3, phase_per_tweezer_string(mtrap, latex=True), color=black, fontsize=14, bbox=dict(facecolor=grey, alpha=0.2, boxstyle='round,pad=0.5'), transform=plt.gca().transAxes)

    if spectrum_type.is_optical:
        plt.hlines(1, *xlims, color=black, alpha=0.1)

    if spectrum_type == SpectrumType.POSITION_CSD:
        plt.vlines(dim_A.Ω / kHz, *ylims, label=rf'$\Omega_{{{jn_string}}}$', color=red, alpha=0.5, ls='--')
        plt.vlines(dim_B.Ω / kHz, *ylims, label=rf'$\Omega_{{{km_string}}}$', color=blue, alpha=0.5, ls='--')
        plt.vlines([_.Ω / kHz for _ in mtrap.dimensions.values()], *ylims, label=r'all other $\Omega_j$', color=black, alpha=0.1, ls='--')
        plt.vlines([-_.Ω / kHz for _ in mtrap.dimensions.values()], *ylims, color=black, alpha=0.1, ls='--')
        plt.plot(ω_range / kHz, np.abs(np.real(vals_A)), label=f'|Re[{repr(method_A)}]|', alpha=0.5, color=blue)
        plt.plot(ω_range / kHz, np.abs(np.real(vals_B)), label=f'|Re[{repr(method_B)}]|', alpha=0.5, color=blue, ls='--', lw=2)
        plt.plot(ω_range / kHz, np.abs(np.imag(vals_A)), label=f'|Im[{repr(method_A)}]|', alpha=0.5, color=orange)
        plt.plot(ω_range / kHz, np.abs(np.imag(vals_B)), label=f'|Im[{repr(method_B)}]|', alpha=0.5, color=orange, ls='--', lw=2)
        plt.plot(ω_range / kHz, np.abs(vals_A), label=f'Abs[{repr(method_A)}]', alpha=0.5, color=green)
        plt.plot(ω_range / kHz, np.abs(vals_B), label=f'Abs[{repr(method_B)}]', alpha=0.5, color=green, ls='--', lw=2)
        plt.yscale('log')
    else:
        if spectrum_type == SpectrumType.POSITION_PSD:
            dim = mtrap.dimensions[dim_label]
            plt.vlines(dim.Ω / kHz, *ylims, label=rf'${dim.latex_Ω}$', color=red, alpha=0.5, ls='--')
            plt.vlines([_.Ω / kHz for _ in mtrap.dimensions.values()], *ylims, label=r'all other $\Omega_j$', color=black, alpha=0.1, ls='--')
        else:
            plt.vlines([_.Ω / kHz for _ in mtrap.dimensions.values()], *ylims, label=r'all $\Omega_j$', color=black, alpha=0.1, ls='--')
            plt.vlines([-_.Ω / kHz for _ in mtrap.dimensions.values()], *ylims, color=black, alpha=0.1, ls='--')
        plt.semilogy(ω_range / kHz, vals_A, label=f'{repr(method_A)}', alpha=0.5, color=blue, lw=1.5)
        plt.semilogy(ω_range / kHz, vals_B, label=f'{repr(method_B)}', alpha=0.5, color=blue, lw=2.5, ls='--')

    plt.legend(**legend_kwargs)
    plt.show()

    if spectrum_type == SpectrumType.POSITION_CSD:
        plt.figure(**fig_kwargs)
        plt.title(f'Magnitude of {title_str} coloured by frequency')
        plt.xlabel('Frequency [kHz]')
        plt.ylabel(ylabel)
        plt.xlim(*xlims)
        plt.ylim(*ylims)
        plt.hlines(0, *xlims, color=black, alpha=0.1)
        plt.vlines([_.Ω / kHz for _ in mtrap.dimensions.values()], *ylims, label=r'All $\Omega_j$', color=black, alpha=0.1, ls='--')
        plt.vlines([-_.Ω / kHz for _ in mtrap.dimensions.values()], *ylims, color=black, alpha=0.1, ls='--')
        plt.vlines(dim_A.Ω / kHz, *ylims, label=rf'$\Omega_{{{jn_string}}}$', color=red, alpha=0.5, ls='--')
        plt.vlines(dim_B.Ω / kHz, *ylims, label=rf'$\Omega_{{{km_string}}}$', color=blue, alpha=0.5, ls='--')
        plt.scatter(ω_range / kHz, np.abs(vals_A), s=10, marker='o', label=f'{repr(method_A)}', alpha=0.3, c=ω_range, cmap='gist_rainbow')
        plt.scatter(ω_range / kHz, np.abs(vals_B), s=10, marker='x', label=f'{repr(method_B)}', alpha=0.3, c=ω_range, cmap='gist_rainbow')
        plt.yscale('log')
        plt.legend(**legend_kwargs)
        plt.show()

        plt.figure(**fig_kwargs)
        plt.title(f'{title_str} in the complex plane coloured by frequency')
        plt.xlabel('Real part')
        plt.ylabel('Imag. part')
        max_Re = 1.2 * max(np.abs(np.real(vals_A)).max(), np.abs(np.real(vals_B)).max())
        max_Im = 1.2 * max(np.abs(np.imag(vals_A)).max(), np.abs(np.imag(vals_B)).max())
        xlims = -max_Re, max_Re
        ylims = -max_Im, max_Im
        plt.xlim(*xlims)
        plt.ylim(*ylims)
        plt.hlines(0, *xlims, color=black, alpha=0.1, label=rf'$\omega_\mathregular{{step}} = ${mtrap.ω_step / Hz:.0f} Hz')
        plt.vlines(0, *ylims, color=black, alpha=0.1)
        plt.scatter(np.real(vals_A), np.imag(vals_A), s=10, marker='o', label=f'{repr(method_A)}', alpha=0.3, c=ω_range, cmap='gist_rainbow')
        plt.scatter(np.real(vals_B), np.imag(vals_B), s=15, marker='x', label=f'{repr(method_B)}', alpha=0.3, c=ω_range, cmap='gist_rainbow')
        plt.legend(**legend_kwargs)
        plt.show()


def visualise_two_noise_component_calculation_methods(
    mtrap, noise_vector_type,
    dim_index, noise_index,
    vals_A, vals_B,
    method_A, method_B,
    **kwargs,
):
    """
    Visualise a given `noise_vector_type`'s component (determined by `dim_index` and `noise_index`)
    calculated via `method_A` vs. `method_B`

    For efficiency, `vals_A` and `vals_B` are pre-calculated from the full noise vector functions
    and passed in as arguments directly here
    """
    set_plt_rcParams_to_xmgrace_style()

    name_str = noise_vector_string(mtrap, noise_vector_type, dim_index, noise_index)
    name_latex = noise_vector_latex(mtrap, noise_vector_type, dim_index, noise_index)

    msg = f'🖼️ Using {mtrap} ({mtrap.tweezer_phase_sign_convention_str}) to visualise the calculation of {name_str} via {method_A=} vs. {method_B=}'
    print(msg), logger.info(msg)

    visualise = kwargs.get('visualise', True)
    return_resonances = kwargs.get('return_resonances', False)
    msg = f'{visualise=}, {return_resonances=}'
    print(msg), logger.info(msg)

    fig_kwargs = dict(figsize=(LATEX_FIG_WIDTH, 3.5), dpi=400)
    legend_kwargs = dict(loc='lower left', prop=dict(size=8), ncol=3)

    ω_range = mtrap.ω_range
    ω_step = np.abs(ω_range[1] - ω_range[0])
    xlims = np.min(ω_range) / kHz, np.max(ω_range) / kHz
    ylim = 1.2 * max(np.abs(vals_A).max(), np.abs(vals_B).max())
    ylims = -ylim, +ylim
    ylabel = kwargs.get('ylabel', '')

    if visualise:
        plt.figure(**fig_kwargs)
        plt.title(f'Complex parts of {name_latex}')
        plt.xlabel('Frequency [kHz]')
        plt.ylabel(ylabel)
        plt.xlim(*xlims)
        plt.ylim(*ylims)
        plt.hlines(0, *xlims, color=black, alpha=0.1)
        plt.vlines([_.Ω / kHz for _ in mtrap.dimensions.values()], *ylims, label=r'All $\Omega_j$', color=black, alpha=0.1, ls='--')
        plt.plot(ω_range / kHz, np.real(vals_A), label=f'Re[{repr(method_A)}]', color=blue)
        plt.plot(ω_range / kHz, np.real(vals_B), label=f'Re[{repr(method_B)}]', color=blue, ls='--', lw=2)
        plt.plot(ω_range / kHz, np.imag(vals_A), label=f'Im[{repr(method_A)}]', color=orange)
        plt.plot(ω_range / kHz, np.imag(vals_B), label=f'Im[{repr(method_B)}]', color=orange, ls='--', lw=2)
        plt.plot(ω_range / kHz, np.abs(vals_A), label=f'Abs[{repr(method_A)}]', color=green)
        plt.plot(ω_range / kHz, np.abs(vals_B), label=f'Abs[{repr(method_B)}]', color=green, ls='--', lw=2)
        plt.legend(**legend_kwargs)
        plt.show()

        plt.figure(**fig_kwargs)
        plt.title(f'Magnitude of {name_latex} coloured by frequency')
        plt.xlabel('Frequency [kHz]')
        plt.ylabel(ylabel)
        plt.xlim(*xlims)
        plt.ylim(*ylims)
        plt.hlines(0, *xlims, color=black, alpha=0.1)
        plt.vlines([_.Ω / kHz for _ in mtrap.dimensions.values()], *ylims, label=r'All $\Omega_j$', color=black, alpha=0.1, ls='--')
        plt.scatter(ω_range / kHz, np.abs(vals_A), s=10, marker='o', label=f'{repr(method_A)}', alpha=0.3, c=ω_range, cmap='gist_rainbow')
        plt.scatter(ω_range / kHz, np.abs(vals_B), s=10, marker='x', label=f'{repr(method_B)}', alpha=0.3, c=ω_range, cmap='gist_rainbow')
        plt.legend(**legend_kwargs)
        plt.show()

        plt.figure(**fig_kwargs)
        plt.title(f'{name_latex} in the complex plane coloured by frequency')
        plt.xlabel('Real part')
        plt.ylabel('Imag. part')
        equal_xlims, equal_ylims = get_latex_doc_lims(ylim, 3e-1)
        plt.xlim(*equal_xlims)
        plt.ylim(*equal_ylims)
        plt.axhline(0, color=black, alpha=0.1, label=rf'$\omega_\mathregular{{step}} = ${ω_step / Hz:.0f} Hz')
        plt.axvline(0, color=black, alpha=0.1)
        [plt.axline((0, 0), slope=np.tan(mtrap.φn_dict[ii]), alpha=0.5 - 0.1 * ii, color=purple, ls='--', label=rf'$\phi_{{{ii}}}$') for ii in range(2, mtrap.num_traps + 1)]
        plt.gca().set_aspect('equal')
        plt.scatter(np.real(vals_A), np.imag(vals_A), s=10, marker='o', label=f'{repr(method_A)}', alpha=0.3, c=ω_range, cmap='gist_rainbow')
        plt.scatter(np.real(vals_B), np.imag(vals_B), s=15, marker='x', label=f'{repr(method_B)}', alpha=0.3, c=ω_range, cmap='gist_rainbow')
        plt.legend(**legend_kwargs)
        plt.show()

        if kwargs.get('plot_by_magnitude', False):
            plt.figure(**fig_kwargs)
            plt.title(f'Magnitude of {name_latex} coloured by magnitude')
            plt.xlabel('Frequency [kHz]')
            plt.ylabel(ylabel)
            plt.xlim(*xlims)
            plt.ylim(*ylims)
            plt.hlines(0, *xlims, color=black, alpha=0.1)
            plt.vlines([_.Ω / kHz for _ in mtrap.dimensions.values()], *ylims, label=r'All $\Omega_j$', color=black, alpha=0.1, ls='--')
            plt.scatter(ω_range / kHz, np.abs(vals_A), s=10, marker='o', label=f'{repr(method_A)}', alpha=0.3, c=np.abs(vals_A), cmap='gist_rainbow')
            plt.scatter(ω_range / kHz, np.abs(vals_B), s=15, marker='x', label=f'{repr(method_B)}', alpha=0.3, c=np.abs(vals_B), cmap='gist_rainbow')
            plt.legend(**legend_kwargs)
            plt.show()

            plt.figure(**fig_kwargs)
            plt.title(f'{name_latex} in the complex plane coloured by magnitude')
            plt.xlabel('Real part')
            plt.ylabel('Imag. part')
            equal_xlims, equal_ylims = get_latex_doc_lims(ylim, 3e-1)
            plt.xlim(*equal_xlims)
            plt.ylim(*equal_ylims)
            plt.axhline(0, color=black, alpha=0.1, label=rf'$\omega_\mathregular{{step}} = ${ω_step / Hz:.0f} Hz')
            plt.axvline(0, color=black, alpha=0.1)
            [plt.axline((0, 0), slope=np.tan(mtrap.φn_dict[ii]), alpha=0.5 - 0.1 * ii, color=purple, ls='--', label=rf'$\phi_{{{ii}}}$') for ii in range(2, mtrap.num_traps + 1)]
            plt.gca().set_aspect('equal')
            plt.scatter(np.real(vals_A), np.imag(vals_A), s=10, marker='o', label=f'{repr(method_A)}', alpha=0.3, c=np.abs(vals_A), cmap='gist_rainbow')
            plt.scatter(np.real(vals_B), np.imag(vals_B), s=15, marker='x', label=f'{repr(method_B)}', alpha=0.3, c=np.abs(vals_B), cmap='gist_rainbow')
            plt.legend(**legend_kwargs)
            plt.show()

    if return_resonances:
        peaks_per_method = {}

        _, peaks_A = find_resonances(ω_range, np.abs(vals_A))
        peaks_per_method[method_A] = peaks_A

        _, peaks_B = find_resonances(ω_range, np.abs(vals_B))
        peaks_per_method[method_B] = peaks_B

        return peaks_per_method


def visualise_two_noise_component_calculation_methods_over_entire_spectrum(
    mtrap, noise_vector_type,
    HD_ω_range, SD_ω_range,
    HD_minus_vals_A, HD_minus_vals_B,
    SD_vals_A, SD_vals_B,
    HD_plus_vals_A, HD_plus_vals_B,
    dim_index, noise_index,
    method_A, method_B,
    **kwargs,
):
    """
    Visualise a given `noise_vector_type`'s component (determined by `dim_index` and `noise_index`)
    calculated via `method_A` vs. `method_B`, across 3 different spectral ranges:
    - -`HD_ω_range`: negative high resolution range, around the negative resonances
    - `SD_ω_range`: lower resolution range, between positive and negative resonances
    - `HD_ω_range`: positive high definition range, around the positive resonances

    For efficiency, all the values are pre-calculated from the full noise vector functions
    and passed in as arguments directly here
    """
    set_plt_rcParams_to_xmgrace_style()

    ylabel = kwargs.get('ylabel', '')
    log = kwargs.get('log', False)

    if noise_vector_type.is_light:
        assert dim_index <= 1

    name_str = noise_vector_string(mtrap, noise_vector_type, dim_index, noise_index)
    name_latex = noise_vector_latex(mtrap, noise_vector_type, dim_index, noise_index)

    msg = f'🖼️ Using {mtrap} ({mtrap.tweezer_phase_sign_convention_str}) to compare the calculation of {name_str} via {method_A=} vs. {method_B=}'
    print(msg), logger.info(msg)

    fig_kwargs = dict(figsize=(LATEX_FIG_WIDTH, 3.5), dpi=400)
    legend_kwargs = dict(loc='lower left', prop=dict(size=8))

    mosaic = [
        ['-HD 1', 'SD 1', '+HD 1'],
        ['-HD 2', 'SD 2', '+HD 2'],
        ['-HD 3', 'SD 3', '+HD 3'],
    ]
    fig = plt.figure(figsize=(2 * LATEX_FIG_WIDTH, 4.5 * len(mosaic)), dpi=200)
    ax_dict = fig.subplot_mosaic(mosaic)
    fig.suptitle(f'Complex parts of {name_latex}')

    HD_ω_step = np.abs(HD_ω_range[1] - HD_ω_range[0])
    SD_ω_step = np.abs(SD_ω_range[1] - SD_ω_range[0])

    ω_range = -HD_ω_range
    mtrap.ω_range = ω_range
    vals_A = HD_minus_vals_A[:, dim_index, noise_index]
    vals_B = HD_minus_vals_B[:, dim_index, noise_index]
    HD_minus_xlims = np.min(ω_range) / kHz, np.max(ω_range) / kHz
    HD_minus_ylim = 1.2 * max(np.abs(vals_A).max(), np.abs(vals_B).max())
    HD_minus_ylims = -HD_minus_ylim, +HD_minus_ylim
    HD_minus_log_ylims = 1e-2 * min(np.abs(vals_A).min(), np.abs(vals_B).min()), 1e2 * max(np.abs(vals_A).max(), np.abs(vals_B).max())

    ω_range = SD_ω_range
    mtrap.ω_range = ω_range
    vals_A = SD_vals_A[:, dim_index, noise_index]
    vals_B = SD_vals_B[:, dim_index, noise_index]
    SD_xlims = np.min(ω_range) / kHz, np.max(ω_range) / kHz
    SD_ylim = 1.2 * max(np.abs(vals_A).max(), np.abs(vals_B).max())
    SD_ylims = -SD_ylim, +SD_ylim
    SD_log_ylims = 1e-2 * min(np.abs(vals_A).min(), np.abs(vals_B).min()), 1e2 * max(np.abs(vals_A).max(), np.abs(vals_B).max())

    ω_range = HD_ω_range
    mtrap.ω_range = ω_range
    vals_A = HD_plus_vals_A[:, dim_index, noise_index]
    vals_B = HD_plus_vals_B[:, dim_index, noise_index]
    HD_plus_xlims = np.min(ω_range) / kHz, np.max(ω_range) / kHz
    HD_plus_ylim = 1.2 * max(np.abs(vals_A).max(), np.abs(vals_B).max())
    HD_plus_ylims = -HD_plus_ylim, +HD_plus_ylim
    HD_plus_log_ylims = 1e-2 * min(np.abs(vals_A).min(), np.abs(vals_B).min()), 1e2 * max(np.abs(vals_A).max(), np.abs(vals_B).max())

    all_log_ylims = *HD_minus_log_ylims, *SD_log_ylims, *HD_plus_log_ylims
    overall_log_ylims = min(*all_log_ylims), max(*all_log_ylims)

    # ----------------------------------------------------------------------------------------------
    # LEFT PANEL: high resolution (small ω_step) around NEGATIVE frequency resonances
    # ----------------------------------------------------------------------------------------------
    ω_range = -HD_ω_range
    vals_A = HD_minus_vals_A[:, dim_index, noise_index]
    vals_B = HD_minus_vals_B[:, dim_index, noise_index]
    xlims, ylims = HD_minus_xlims, HD_minus_ylims

    ax = ax_dict['-HD 1']
    ax.set_xlabel('Frequency [kHz]')
    ax.set_ylabel(ylabel)
    ax.set_xlim(*xlims), ax.set_ylim(*ylims)
    ax.hlines(0, *xlims, color=black, alpha=0.1)
    ax.vlines([-_.Ω / kHz for _ in mtrap.dimensions.values()], *ylims, label=r'All $-\Omega_j$', color=black, alpha=0.1, ls='--')
    ax.plot(ω_range / kHz, np.real(vals_A), label=f'Re[{repr(method_A)}]', color=blue)
    ax.plot(ω_range / kHz, np.real(vals_B), label=f'Re[{repr(method_B)}]', color=blue, ls='--', lw=2)
    ax.plot(ω_range / kHz, np.imag(vals_A), label=f'Im[{repr(method_A)}]', color=orange)
    ax.plot(ω_range / kHz, np.imag(vals_B), label=f'Im[{repr(method_B)}]', color=orange, ls='--', lw=2)
    ax.plot(ω_range / kHz, np.abs(vals_A), label=f'Abs[{repr(method_A)}]', color=green)
    ax.plot(ω_range / kHz, np.abs(vals_B), label=f'Abs[{repr(method_B)}]', color=green, ls='--', lw=2)
    ax.legend(**legend_kwargs)
    ax.grid(alpha=0.1)

    ax = ax_dict['-HD 2']
    ax.set_xlabel('Frequency [kHz]')
    ax.set_xlim(*xlims)
    ax.hlines(0, *xlims, color=black, alpha=0.1)
    if log:
        ax.set_ylim(*overall_log_ylims)
        ax.vlines([-_.Ω / kHz for _ in mtrap.dimensions.values()], *overall_log_ylims, label=r'All -$\Omega_j$', color=black, alpha=0.1, ls='--')
        ax.set_yscale('log')
    else:
        ax.set_ylim(*ylims)
        ax.vlines([-_.Ω / kHz for _ in mtrap.dimensions.values()], *ylims, label=r'All $\Omega_j$', color=black, alpha=0.1, ls='--')
    ax.scatter(ω_range / kHz, np.abs(vals_A), s=10, marker='o', label=f'{repr(method_A)}', alpha=0.3, c=ω_range, cmap='gist_rainbow')
    ax.scatter(ω_range / kHz, np.abs(vals_B), s=10, marker='x', label=f'{repr(method_B)}', alpha=0.3, c=ω_range, cmap='gist_rainbow')
    ax.legend(**legend_kwargs)
    ax.grid(alpha=0.1)

    ax = ax_dict['-HD 3']
    ax.set_xlabel('Real part')
    ax.set_ylabel('Imag. part')
    ax.set_xlim(*ylims), ax.set_ylim(*ylims)
    ax.axhline(0, color=black, alpha=0.1, label=rf'$\omega_\mathregular{{step}} = ${HD_ω_step / Hz:.0f} Hz')
    ax.axvline(0, color=black, alpha=0.1)
    [ax.axline((0, 0), slope=np.tan(mtrap.φn_dict[ii]), alpha=0.5 - 0.1 * ii, color=purple, ls='--', label=rf'$\phi_{{{ii}}}$') for ii in range(2, mtrap.num_traps + 1)]
    ax.set_aspect('equal')
    ax.scatter(np.real(vals_A), np.imag(vals_A), s=10, marker='o', label=f'{repr(method_A)}', alpha=0.3, c=ω_range, cmap='gist_rainbow')
    ax.scatter(np.real(vals_B), np.imag(vals_B), s=15, marker='x', label=f'{repr(method_B)}', alpha=0.3, c=ω_range, cmap='gist_rainbow')
    ax.legend(**legend_kwargs)
    ax.grid(alpha=0.1)

    # ----------------------------------------------------------------------------------------------
    # MIDDLE PANEL: low resolution (large ω_step) in large stretch between +ve and -ve resonances
    # ----------------------------------------------------------------------------------------------
    ω_range = SD_ω_range
    vals_A = SD_vals_A[:, dim_index, noise_index]
    vals_B = SD_vals_B[:, dim_index, noise_index]
    xlims, ylims = SD_xlims, SD_ylims

    ax = ax_dict['SD 1']
    ax.set_xlabel('Frequency [kHz]')
    ax.set_xlim(*xlims), ax.set_ylim(*ylims)
    ax.hlines(0, *xlims, color=black, alpha=0.1)
    ax.vlines([-_.Ω / kHz for _ in mtrap.dimensions.values()], *ylims, label=r'All $-\Omega_j$', color=black, alpha=0.1, ls='--')
    ax.plot(ω_range / kHz, np.real(vals_A), label=f'Re[{repr(method_A)}]', color=blue)
    ax.plot(ω_range / kHz, np.real(vals_B), label=f'Re[{repr(method_B)}]', color=blue, ls='--', lw=2)
    ax.plot(ω_range / kHz, np.imag(vals_A), label=f'Im[{repr(method_A)}]', color=orange)
    ax.plot(ω_range / kHz, np.imag(vals_B), label=f'Im[{repr(method_B)}]', color=orange, ls='--', lw=2)
    ax.plot(ω_range / kHz, np.abs(vals_A), label=f'Abs[{repr(method_A)}]', color=green)
    ax.plot(ω_range / kHz, np.abs(vals_B), label=f'Abs[{repr(method_B)}]', color=green, ls='--', lw=2)
    ax.grid(alpha=0.1)

    ax = ax_dict['SD 2']
    ax.set_title(f'Magnitude of {name_latex} coloured by frequency')
    ax.set_xlabel('Frequency [kHz]')
    ax.set_xlim(*xlims)
    ax.hlines(0, *xlims, color=black, alpha=0.1)
    if log:
        ax.set_ylim(*overall_log_ylims)
        ax.set_yscale('log')
    else:
        ax.set_ylim(*ylims)
    ax.scatter(ω_range / kHz, np.abs(vals_A), s=10, marker='o', label=f'{repr(method_A)}', alpha=0.3, c=ω_range, cmap='gist_rainbow_r')
    ax.scatter(ω_range / kHz, np.abs(vals_B), s=10, marker='x', label=f'{repr(method_B)}', alpha=0.3, c=ω_range, cmap='gist_rainbow_r')
    ax.grid(alpha=0.1)

    ax = ax_dict['SD 3']
    ax.set_title(f'{name_latex} in the complex plane coloured by frequency')
    ax.set_xlabel('Real part')
    ax.set_ylabel('Imag. part')
    ax.set_xlim(*ylims), ax.set_ylim(*ylims)
    ax.axhline(0, color=black, alpha=0.1, label=rf'$\omega_\mathregular{{step}} = ${SD_ω_step / kHz:.0f} kHz')
    ax.axvline(0, color=black, alpha=0.1)
    [ax.axline((0, 0), slope=np.tan(mtrap.φn_dict[ii]), alpha=0.5 - 0.1 * ii, color=purple, ls='--', label=rf'$\phi_{{{ii}}}$') for ii in range(2, mtrap.num_traps + 1)]
    ax.set_aspect('equal')
    ax.scatter(np.real(vals_A), np.imag(vals_A), s=10, marker='o', label=f'{repr(method_A)}', alpha=0.3, c=ω_range, cmap='gist_rainbow_r')
    ax.scatter(np.real(vals_B), np.imag(vals_B), s=15, marker='x', label=f'{repr(method_B)}', alpha=0.3, c=ω_range, cmap='gist_rainbow_r')
    ax.legend(**legend_kwargs)
    ax.grid(alpha=0.1)

    # ----------------------------------------------------------------------------------------------
    # RIGHT PANEL: high resolution (small ω_step) around POSITIVE frequency resonances
    # ----------------------------------------------------------------------------------------------
    ω_range = HD_ω_range
    vals_A = HD_plus_vals_A[:, dim_index, noise_index]
    vals_B = HD_plus_vals_B[:, dim_index, noise_index]
    xlims, ylims = HD_plus_xlims, HD_plus_ylims

    ax = ax_dict['+HD 1']
    ax.set_xlabel('Frequency [kHz]')
    ax.set_xlim(*xlims), ax.set_ylim(*ylims)
    ax.hlines(0, *xlims, color=black, alpha=0.1)
    ax.vlines([_.Ω / kHz for _ in mtrap.dimensions.values()], *ylims, label=r'All $-\Omega_j$', color=black, alpha=0.1, ls='--')
    ax.plot(ω_range / kHz, np.real(vals_A), label=f'Re[{repr(method_A)}]', color=blue)
    ax.plot(ω_range / kHz, np.real(vals_B), label=f'Re[{repr(method_B)}]', color=blue, ls='--', lw=2)
    ax.plot(ω_range / kHz, np.imag(vals_A), label=f'Im[{repr(method_A)}]', color=orange)
    ax.plot(ω_range / kHz, np.imag(vals_B), label=f'Im[{repr(method_B)}]', color=orange, ls='--', lw=2)
    ax.plot(ω_range / kHz, np.abs(vals_A), label=f'Abs[{repr(method_A)}]', color=green)
    ax.plot(ω_range / kHz, np.abs(vals_B), label=f'Abs[{repr(method_B)}]', color=green, ls='--', lw=2)
    ax.grid(alpha=0.1)

    ax = ax_dict['+HD 2']
    ax.set_xlabel('Frequency [kHz]')
    ax.set_xlim(*xlims)
    ax.hlines(0, *xlims, color=black, alpha=0.1)
    if log:
        ax.set_ylim(*overall_log_ylims)
        ax.vlines([_.Ω / kHz for _ in mtrap.dimensions.values()], *overall_log_ylims, label=r'All -$\Omega_j$', color=black, alpha=0.1, ls='--')
        ax.set_yscale('log')
    else:
        ax.set_ylim(*ylims)
        ax.vlines([_.Ω / kHz for _ in mtrap.dimensions.values()], *ylims, label=r'All $\Omega_j$', color=black, alpha=0.1, ls='--')
    ax.scatter(ω_range / kHz, np.abs(vals_A), s=10, marker='o', label=f'{repr(method_A)}', alpha=0.3, c=ω_range, cmap='gist_rainbow')
    ax.scatter(ω_range / kHz, np.abs(vals_B), s=10, marker='x', label=f'{repr(method_B)}', alpha=0.3, c=ω_range, cmap='gist_rainbow')
    ax.legend(**legend_kwargs)
    ax.grid(alpha=0.1)

    ax = ax_dict['+HD 3']
    ax.set_xlabel('Real part')
    ax.set_ylabel('Imag. part')
    ax.set_xlim(*ylims), ax.set_ylim(*ylims)
    ax.axhline(0, color=black, alpha=0.1, label=rf'$\omega_\mathregular{{step}} = ${HD_ω_step / Hz:.0f} Hz')
    ax.axvline(0, color=black, alpha=0.1)
    [ax.axline((0, 0), slope=np.tan(mtrap.φn_dict[ii]), alpha=0.5 - 0.1 * ii, color=purple, ls='--', label=rf'$\phi_{{{ii}}}$') for ii in range(2, mtrap.num_traps + 1)]
    ax.set_aspect('equal')
    ax.scatter(np.real(vals_A), np.imag(vals_A), s=10, marker='o', label=f'{repr(method_A)}', alpha=0.3, c=ω_range, cmap='gist_rainbow')
    ax.scatter(np.real(vals_B), np.imag(vals_B), s=15, marker='x', label=f'{repr(method_B)}', alpha=0.3, c=ω_range, cmap='gist_rainbow')
    ax.legend(**legend_kwargs)
    ax.grid(alpha=0.1)

    plt.show()


def visualise_relative_difference_of_two_arrays(
    mtrap,
    array_A, array_B,
    name_A, name_B,
    **kwargs,
):
    """
    Visualise the relative difference of `array_A` vs. `array_B`
    """
    # relative_diff_threshold = kwargs.get('relative_diff_threshold', RELATIVE_DIFF_threshold)
    _, relative_diff_threshold, _ = test_thresholds(kwargs.get('difficulty_level', 'very hard'))
    xlims = mtrap.ω_range.min() / kHz, mtrap.ω_range.max() / kHz
    ylims = 1e-17, 1e9

    fig_kwargs = dict(figsize=(LATEX_FIG_WIDTH, LATEX_FIG_WIDTH), dpi=400)
    legend_kwargs = dict(loc='upper center', prop=dict(size=8))

    diffs = array_B - array_A
    abs_relative_diffs = np.abs(diffs) / np.abs(array_A)

    # NOTE: only plot absolute values as complex parts seem to mimic
    # abs_real_relative_diff = np.abs(np.real(diffs) / np.real(array_A))
    # abs_imag_relative_diff = np.abs(np.imag(diffs) / np.imag(array_A))

    mosaic = [
        # ['real'],
        # ['imag'],
        ['abs'],
    ]
    colour_per_ax = {'real': blue, 'imag': orange, 'abs': green}

    fig = plt.figure(figsize=(LATEX_FIG_WIDTH, 4 * len(mosaic)), dpi=400)
    ax_dict = fig.subplot_mosaic(mosaic)
    fig.suptitle(f'Magnitude of relative difference of\n{name_A} vs. {name_B}')

    # vals = abs_real_relative_diff
    # ax = ax_dict['real']
    # ax.set_title('Real values')

    # vals = abs_imag_relative_diff
    # ax = ax_dict['imag']
    # ax.set_title('Imaginary values')

    vals = abs_relative_diffs
    ax = ax_dict['abs']
    ax.set_title('Absolute values')

    for ax_name, ax in ax_dict.items():
        ax.set_xlabel('Frequency [kHz]')
        ax.set_yscale('log')
        ax.set_xlim(*xlims)
        ax.set_ylim(*ylims)
        ax.vlines([_.Ω / kHz for _ in mtrap.dimensions.values()], *ylims, label=r'All $\Omega_j$', color=black, alpha=0.1, ls='--')
        ax.vlines([-_.Ω / kHz for _ in mtrap.dimensions.values()], *ylims, color=black, alpha=0.1, ls='--')
        ax.axhline(relative_diff_threshold, lw=3, ls='--', color=RGB_yellow, label=f'failure = {relative_diff_threshold:.0e}', alpha=0.8, gapcolor=black)
        ax.plot(mtrap.ω_range / kHz, vals, '.', alpha=0.1, color=colour_per_ax[ax_name])
        ax.axhline(np.max(vals), lw=0.5, ls=':', color=red, label=f'max = {np.max(vals):.2e}')
        ax.axhline(np.mean(vals), lw=0.5, ls='--', color=red, label=f'mean = {np.mean(vals):.2e}')
        ax.axhline(np.median(vals), lw=0.5, ls='-', color=red, label=f'median = {np.median(vals):.2e}')
        # https://numpy.org/doc/stable/reference/generated/numpy.finfo.html
        # np.finfo(float).eps = 2.220446049250313e-16
        ax.axhline(np.finfo(float).eps, lw=3, ls='-', color=RGB_magenta, label=f'{np.finfo(float).eps=:.2e}', alpha=0.5)
        ax.legend(**legend_kwargs)
    plt.show()


# --------------------------------------------------------------------------------------------------
# RESULTS MATRIX PLOTTING FUNCTIONS
# --------------------------------------------------------------------------------------------------
def plot_pass_fail_matrix(matrix, names, **kwargs):
    """
    Plots the PASS/FAIL matrix for the pair-wise comparison of named items (e.g. images, arrays)
    """
    num_names = len(names)
    fig, ax = plt.subplots(**kwargs.get('fig_kwargs', dict(figsize=(LATEX_FIG_WIDTH, LATEX_FIG_WIDTH), dpi=400)))

    font_size = 8 if num_names < 10 else 6
    # remove last tick label the last from x-axis
    ax.set_xticks(range(num_names), labels=names[:-1] + [''], rotation=45, ha='right', rotation_mode='anchor', fontsize=font_size)
    # remove first tick label the last from y-axis
    ax.set_yticks(range(num_names), labels=[''] + names[1:], fontsize=font_size)

    # keep only lower diagonal
    np.fill_diagonal(matrix, 0)
    upper_triangle_mask = np.triu(np.ones_like(matrix, dtype=bool))
    matrix = np.ma.array(matrix, mask=upper_triangle_mask)

    # this makes sure its all green if all tests pass
    # but still green/red for pass/fail if some tests fail
    cmap_list = [RGB_green] if matrix.all() else [RGB_red, RGB_green]
    im = ax.imshow(matrix, cmap=matplotlib.colors.ListedColormap(cmap_list))

    for ii in range(num_names):
        for jj in range(num_names):
            if ii > jj:
                mark = 'PASS' if matrix[ii, jj] else 'FAIL'
                text = ax.text(jj, ii, mark, ha='center', va='center', size=font_size)

    ax.set_title(kwargs.get('pass_fail_matrix_title', ''))
    ax.spines[['top', 'bottom', 'left', 'right']].set_visible(False)
    ax.tick_params(top=False, bottom=False, left=False, right=False)
    plt.show()


def plot_mean_abs_relative_diffs_matrix(matrix, names, **kwargs):
    """
    Plots the matrix for the pair-wise comparison of the mean absolute difference
    """
    num_names = len(names)
    fig, ax = plt.subplots(figsize=(LATEX_FIG_WIDTH, LATEX_FIG_WIDTH), dpi=400)

    font_size = 8 if num_names < 10 else 6
    # remove last tick label the last from x-axis
    ax.set_xticks(range(num_names), labels=names[:-1] + [''], rotation=45, ha='right', rotation_mode='anchor', fontsize=font_size)
    # remove first tick label the last from y-axis
    ax.set_yticks(range(num_names), labels=[''] + names[1:], fontsize=font_size)

    np.fill_diagonal(matrix, np.nan)
    upper_triangle_mask = np.triu(np.ones_like(matrix, dtype=bool))
    matrix = np.ma.array(matrix, mask=upper_triangle_mask)

    vals = np.log10(np.abs(matrix))
    # np.finfo(float).precision = 16
    im = ax.imshow(vals, vmin=-17, vmax=-3)

    for ii in range(num_names):
        for jj in range(num_names):
            if ii > jj:
                # when the difference is exactly zero, the log10 gives a masked value
                if getattr(vals[ii, jj], 'mask', False):
                    note = 'exact\nzero\ndiff'
                    colour = black
                else:
                    note = f'{vals[ii, jj]:.2f}'
                    colour = white
                text = ax.text(jj, ii, note, ha='center', va='center', color=colour, size=font_size)

    title = kwargs.get('mean_abs_relative_diffs_matrix_title', '')
    ax.set_title(f'{title}\n' + r'$\mathrm{log}_{10}($mean of absolute relative difference$)$')
    ax.spines[['top', 'bottom', 'left', 'right']].set_visible(False)
    ax.tick_params(top=False, bottom=False, left=False, right=False)
    plt.show()


# --------------------------------------------------------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------------------------------------------------------
def greyscale_diff(image_A, image_B):
    """
    Convert the diff of two images into greyscale
    """
    return cv2.cvtColor(cv2.subtract(image_A, image_B), cv2.COLOR_BGR2GRAY)


def delete_local_png_files():
    """
    Delete all local .png files
    """
    files = os.listdir()
    for file in files:
        if file.endswith('.png'):
            try:
                logger.trace(f'Deleting {file=}')
                os.remove(file)
            except Exception as exception:
                logger.error(f'Error deleting {file}: {exception}')


def calculate_temperature_from_position_PSD(mtrap, vals, ii, label):
    """
    Calculates the temperature from the PSD via two methods:
        - phonons
        - variance
    """
    jn = mtrap.dimensions[label]
    ħΩ = ħ * jn.Ω

    SQQ_area = np.trapz(vals[:, ii], mtrap.ω_range / Hz)
    njn = 0.5 * (SQQ_area - 1)
    logger.trace(f'n{label} = {njn:.6f} phonons')
    T_via_phonons = njn * ħΩ / kB
    logger.trace(f'T{label} via phonons = {T_via_phonons:.6e} K')

    eqp_f = jn.m * jn.Ω**2 / kB
    q_variance = SQQ_area * jn.zpf**2
    T_via_variance = eqp_f * q_variance
    logger.trace(f'T{label} via variance = {T_via_variance:.6e} K')

    return {
        'phonons': T_via_phonons,
        'variance': T_via_variance,
    }


def get_latex_labels(mtrap, hat=False, PSD=False):
    """
    Returns a list of latex strings of dimension labels
    e.g. $x_1$ or $\hat{x}_1$
    """
    subscripted_labels = [f'{label[0]}_{{{label[1:]}}}' for label in mtrap.labels]
    latex_labels = [rf'${subscripted_label}$' for subscripted_label in subscripted_labels]
    if hat:
        return [rf'$\hat{{{subscripted_label}}}$' for subscripted_label in subscripted_labels]
    if PSD:
        return [rf'$S_{{{subscripted_label + subscripted_label}}}(\omega)$' for subscripted_label in subscripted_labels]
    return latex_labels


def noise_vector_string(mtrap, noise_vector_type, dim_index, noise_index):
    """
    Returns a string representing the noise vector and component
    """
    match noise_vector_type:
        case NoiseVectorType.HYBRIDISED_POSITIONS:
            operator_name = mtrap.index_to_label[dim_index]
        case NoiseVectorType.LIGHT_INSIDE_CAVITY:
            if dim_index not in [0, 1]:
                return ''
            operator_name = {0: 'a', 1: 'a†'}[dim_index]
        case NoiseVectorType.CAVITY_OUTPUT_LIGHT:
            if dim_index not in [0, 1]:
                return ''
            operator_name = {0: 'a_out', 1: 'a†_out'}[dim_index]

    if noise_index == 0:
        return operator_name + '[a_in]'
    if noise_index == 1:
        return operator_name + '[a†_in]'

    label = mtrap.noise_index_to_label[noise_index]
    if noise_index % 2:
        return operator_name + f'[b†_{label}_in]'
    else:
        return operator_name + f'[b_{label}_in]'


def noise_vector_latex(mtrap, noise_vector_type, dim_index, noise_index):
    """
    Returns a latex string representing the noise vector and component
    """
    match noise_vector_type:
        case NoiseVectorType.HYBRIDISED_POSITIONS:
            dim_label = mtrap.index_to_label[dim_index]
            j, n = dim_label[0], dim_label[1:]
            operator_name = rf'$\hat{{{j}}}_{n}$'
        case NoiseVectorType.LIGHT_INSIDE_CAVITY:
            if dim_index not in [0, 1]:
                return ''
            operator_name = {0: r'$\hat{a}$', 1: r'$\hat{a}^\dagger$'}[dim_index]
        case NoiseVectorType.CAVITY_OUTPUT_LIGHT:
            if dim_index not in [0, 1]:
                return ''
            operator_name = {0: r'$\hat{a}_\mathregular{out}$', 1: r'$\hat{a}^\dagger_\mathregular{out}$'}[dim_index]

    if noise_index == 0:
        return operator_name + r'$[\tilde{a}_{\mathregular{in}}]$'
    if noise_index == 1:
        return operator_name + r'$[\tilde{a}^\dagger_{\mathregular{in}}]$'

    label = mtrap.noise_index_to_label[noise_index]
    j, n = label[0], label[1:]
    jn_in_string = f'{j}_{n}' + r',\mathregular{in}'
    if noise_index % 2:
        return operator_name + rf'$[\tilde{{b}}^\dagger_{{{jn_in_string}}}]$'
    else:
        return operator_name + rf'$[\tilde{{b}}_{{{jn_in_string}}}]$'


def phase_per_tweezer_string(mtrap, latex=False, uppercase=False, unit=degrees, unit_symbol='°'):
    """
    Returns string of the values of the tweezer phases with lowercase φ
    """
    string = ''
    if latex:
        symbol = 'Phi' if uppercase else 'phi'
    else:
        symbol = 'Φ' if uppercase else 'φ'

    for trap_index in range(1, mtrap.num_traps + 1):
        value = f'{mtrap.φn_dict[trap_index] / unit:.0f}{unit_symbol}'

        if latex:
            string += rf'$\{symbol}_{{{trap_index}}} = $ {value}'
        else:
            string += f'{symbol}{trap_index} = {value}'

        if trap_index != mtrap.num_traps:
            string += '\n'

    return string


def relative_tweezer_phases_string(mtrap):
    """
    Returns string of the relative phase values between tweezer traps
    """
    string = ''
    for ii, (δφnm_key, δφnm_value) in enumerate(mtrap.δφnm_dict.items()):
        string += rf'{δφnm_key} = {δφnm_value / degrees:.0f}°'
        if ii != len(mtrap.δφnm_dict) - 1:
            string += '\n'
    return string


def find_resonances(ω_range, values):
    """
    Find resonance peaks and frequencies
    """
    resonance_indices = scipy.signal.find_peaks(values)[0]
    ω_peaks = ω_range[resonance_indices]
    peak_values = values[resonance_indices]
    logger.info(f'find_resonances: {ω_peaks=}')
    logger.info(f'find_resonances: {peak_values=}')
    return ω_peaks, peak_values

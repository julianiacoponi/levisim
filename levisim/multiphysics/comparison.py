import os
import cv2

import numpy as np
import matplotlib.pyplot as plt

from levisim.core.logger import logger
from levisim.core.constants import ħ, kB
from levisim.core.units import Hz, kHz
from levisim.multiphysics.spectrum_type import SpectrumType
from levisim.multiphysics.quantum_linear_theory_method import QuantumLinearTheoryMethod


__all__ = [
    'greyscale_diff',
    'compare_methods',
]


def greyscale_diff(image_1, image_2):
    return cv2.cvtColor(cv2.subtract(image_1, image_2), cv2.COLOR_BGR2GRAY)


def compare_methods(
    multitrap,
    methods_to_compare,
    spectra_to_compare,
    return_PSD_temperatures=False,
    **kwargs,
):
    logger.info(f'Using {multitrap}, comparing {spectra_to_compare} using methods {methods_to_compare}')

    # plt.close('all')

    methods = methods_to_compare
    spectrum_types = spectra_to_compare
    N, axes = multitrap.num_traps, multitrap.Cartesian_axes_per_trap

    if QuantumLinearTheoryMethod.TNT_LINEAR_AMPLIFIER_MODEL in methods:
        spectrum_types = [SpectrumType.PSD]
    if QuantumLinearTheoryMethod.is_analytic:
        spectrum_types = [SpectrumType.HETERODYNE, SpectrumType.HOMODYNE]

    ω_low = min(multitrap.ω_range) / kHz
    ω_high = max(multitrap.ω_range) / kHz
    ω_step = (multitrap.ω_range[1] - multitrap.ω_range[0]) / kHz

    image_dicts = {}
    all_plot_values = {}
    for spectrum_type in spectrum_types:
        image_dicts[spectrum_type] = {}
        for method in methods:
            logger.info(f'Comparing {spectrum_type=} using {method=}')
            all_plot_values[method] = {}
            print(f'{spectrum_type} -- {method}')

            label = spectrum_type
            calc_kwargs = {}
            if spectrum_type == SpectrumType.PSD:
                plot_values, _ = multitrap.calculate_all_hybridised_position_spectra(method, zpf_units=False)
                label = multitrap.labels
            elif spectrum_type == SpectrumType.HETERODYNE:
                plot_values = multitrap.calculate_heterodyne_spectrum(method)
            elif spectrum_type == SpectrumType.HOMODYNE:
                plot_values = multitrap.calculate_homodyne_spectrum(method)

            all_plot_values[method][spectrum_type] = plot_values

            if spectrum_type == SpectrumType.PSD and N >= 3 and axes >= 2:
                PSD_fig = plt.figure(layout='constrained', figsize=(8, 16))
                PSD_ax_dict = PSD_fig.subplot_mosaic([
                    ['X1'],
                    ['Y1'],
                    ['X2'],
                    ['Y2'],
                    ['X3'],
                    ['Y3'],
                ])
                PSD_ax_list = list(PSD_ax_dict.values())
                for ii, PSD_ax in enumerate(PSD_ax_list):
                    PSD_ax.plot(multitrap.ω_range / kHz, plot_values[:, ii], label=label[ii])
                    PSD_ax.set_yscale('log')
                    PSD_ax.legend()
                    PSD_ax.grid()
                    PSD_ax.set_xlim(ω_low, ω_high)
                    PSD_ax.set_ylim(1e-7, 1e1)
                plt.show()

            plt.plot(
                multitrap.ω_range / kHz,
                plot_values,
                label=label,
            )
            if kwargs.get('yscale_log', True):
                plt.yscale('log')
            plt.legend()
            plt.grid()
            plt.xlim(ω_low, ω_high)
            if spectrum_type == SpectrumType.PSD:
                plt.ylim(1e-8, 1e1)
            # if spectrum_type == SpectrumType.HETERODYNE:
            #     het_ylims = kwargs.get('heterodyne_ylims', (5e-1, 1e1))
            #     plt.ylim(*het_ylims)
            # if spectrum_type == SpectrumType.HOMODYNE:
            #     hom_ylims = kwargs.get('homodyne_ylims', (0, 5))
            #     plt.ylim(*hom_ylims)
            figname = f'{str(spectrum_type)} -- {method}.png'
            plt.savefig(figname)
            image_dicts[spectrum_type][method] = cv2.imread(figname)
            plt.show()

            temperatures = {}
            if spectrum_type == SpectrumType.PSD:
                for ii, label in enumerate(multitrap.labels):
                    temperatures[label] = calculate_PSD_temperature(multitrap, plot_values, ii, label)
                if return_PSD_temperatures:
                    return temperatures

    method_pairs = []
    # comparing 2
    if len(methods) >= 2:
        method_pairs.append((methods[0], methods[1]))

    # comparing 3
    if len(methods) >= 3:
        method_pairs.append((methods[1], methods[2]))
        method_pairs.append((methods[0], methods[2]))

    # comparing 4
    if len(methods) >= 4:
        method_pairs.append((methods[2], methods[3]))
        method_pairs.append((methods[0], methods[3]))
        method_pairs.append((methods[1], methods[3]))

    # comparing 5
    if len(methods) >= 5:
        method_pairs.append((methods[3], methods[4]))
        method_pairs.append((methods[0], methods[4]))
        method_pairs.append((methods[1], methods[4]))
        method_pairs.append((methods[2], methods[4]))

    results_dict = {}
    all_results = []
    for spectrum_type, image_dict in image_dicts.items():

        results_dict[spectrum_type] = {}

        for method_pair in method_pairs:
            diff = greyscale_diff(image_dict[method_pair[0]], image_dict[method_pair[1]])
            result = cv2.countNonZero(diff) == 0

            all_results.append(result)
            results_dict[spectrum_type][method_pair] = result

    files = os.listdir()
    for file in files:
        if file.endswith('.png'):
            try:
                os.remove(file)
            except Exception as e:
                logger.error(f'Error deleting {file}: {e}')

    assert all(all_results), f'🏴‍☠️🏴‍☠️🏴‍☠️🏴‍☠️🏴‍☠️ Failure present in {multitrap}! 🏴‍☠️🏴‍☠️🏴‍☠️🏴‍☠️🏴‍☠️\n{results_dict=}'
    print(f'🎉🎉🎉🎉🎉 All spectral calculation methods agree for {multitrap} :-) 🎉🎉🎉🎉🎉')
    print(results_dict)

    return


def calculate_PSD_temperature(multitrap, plot_values, ii, label):
    """
    Calculates the temperature from the PSD via two methods:
        - phonons
        - variance
    """
    jn = multitrap.dimensions[label]
    ħΩ = ħ * jn.Ω

    SQQ_area = np.trapz(plot_values[:, ii], multitrap.ω_range / Hz)
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

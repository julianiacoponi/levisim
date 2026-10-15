import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from matplotlib.animation import ArtistAnimation
from itertools import cycle
from copy import copy
from tqdm import tqdm

from levisim.core.logger import logger
from levisim.core.constants import ħ, kB
from levisim.core.units import Hz, kHz
from levisim.core.colours import discrete_viridis, discrete_plasma, discrete_magma

from levisim.multiphysics.multitrap import MultiTrap
from levisim.multiphysics.enums.quantum_linear_theory_method import QuantumLinearTheoryMethod
from levisim.multiphysics.enums.spectrum import SpectrumType
from levisim.multiphysics.enums.parameter import ParameterType


__all__ = [
    'spectrum_type_to_latex',
    'spectrum_type_to_units',
    'calculate_phonons_and_temperatures_from_PSD',
    'calculate_spectra_across_y_range_for_one_frame',
    'title_string_for_one_frame',
    'calculate_spectra_for_all_frames',

    'generate_heatmap_animation_frames',
    'generate_slices_animation_frames',
    'animate_spectra',
]


def spectrum_type_to_latex(spectrum_type, **kwargs):
    """
    Returns a latex symbol string given a choice of spectrum
    """
    if spectrum_type == SpectrumType.HETERODYNE:
        return '$S_{het}$'

    if spectrum_type == SpectrumType.HOMODYNE:
        return '$S_{hom}$'

    PSD_label = kwargs.get('PSD_label')
    j, n = PSD_label[0], PSD_label[1:]
    sub = f'{j}_{n} {j}_{n}'
    return f'$S_{{{sub}}}$'


def spectrum_type_to_units(spectrum_type):
    """
    Returns a units string given a choice of spectrum
    """
    if spectrum_type.is_optical:
        return '[photons/Hz]'
    return '[m²/Hz]'


def calculate_area(mtrap, values):
    """
    Take the area under the given spectral values
    """
    return np.trapz(values, mtrap.ω_range / Hz)


def calculate_phonons_and_temperatures_from_PSD(mtrap, Sqq_values, label):
    """
    Take the area under the given PSD to find the number of phonons and temperature
    - area under zpf-normalised PSD = 2n + 1, where n = phonon occupation
        area = 2n + 1
        ∴
        n = ½(area - 1)
    - area under (two-sided) PSD of variable = variance of variable
        〈q²〉= (1/2π) ∫ Sqq(Ω)dΩ
        ∴
        〈q²〉= ∫ Sxx(f)df
    - equipartition theorem states: kinetic energy from oscillation = thermal energy
        ½mΩ²〈q²〉 = ½kB×T
        ∴
        T = (mΩ² / kB) × 〈q²〉
    """
    jn = mtrap.dimensions[label]
    ħΩ = ħ * jn.Ω

    # convert Sqq(Ω)dΩ to Sqq(f)df with Hz = 2π
    q_variance = np.trapz(Sqq_values, mtrap.ω_range / Hz)

    # Convert to Q = q/q_zpf
    SQQ_area = q_variance / jn.zpf**2
    phonons = 0.5 * (SQQ_area - 1)
    temperature_via_phonons = phonons * ħΩ / kB

    equipartition_factor = jn.m * jn.Ω**2 / kB
    q_variance = SQQ_area * jn.zpf**2
    temperature_via_variance = equipartition_factor * q_variance

    return phonons, temperature_via_phonons, temperature_via_variance


def update_parameter_in_multitrap_kwargs(multitrap_kwargs, parameter, new_value):
    """
    Updates the given parameter from `Multitrap` instance to the given value
    """
    logger.info(f'Updating {parameter.type} to {new_value / parameter.units:{parameter.format_specifier}}{parameter.unit_symbol}')

    if parameter.type == ParameterType.LIGHT_MECH_COUPLING:
        multitrap_kwargs['toy_params']['toy_gx'] = new_value
        multitrap_kwargs['toy_params']['toy_gy'] = new_value

    elif parameter.type == ParameterType.PHASE_OF_2ND_TWEEZER:
        multitrap_kwargs['toy_params']['φ2'] = new_value

    elif parameter.type == ParameterType.CAVITY_DETUNING:
        multitrap_kwargs['cavity_kwargs']['Δ'] = new_value

    elif parameter.type == ParameterType.OPTICAL_DAMPING:
        multitrap_kwargs['cavity_kwargs']['κ'] = new_value

    elif parameter.type == ParameterType.AIR_PRESSURE:
        multitrap_kwargs['gas_kwargs']['P'] = new_value
    elif parameter.type == ParameterType.AIR_TEMPERATURE:
        multitrap_kwargs['gas_kwargs']['T'] = new_value

    elif parameter.type == ParameterType.EVALUATION_ORDER:
        multitrap_kwargs['evaluation_order'] = int(new_value)

        # change the calculation method depending on whether evaluating at highest order or not
        num_dimensions = multitrap_kwargs['num_traps'] * multitrap_kwargs['Cartesian_axes_per_trap']
        if multitrap_kwargs['evaluation_order'] == num_dimensions:
            multitrap_kwargs['QLT_method'] = QuantumLinearTheoryMethod.NUMERICAL_HYBRIDISATION
        elif multitrap_kwargs['evaluation_order'] < num_dimensions:
            multitrap_kwargs['QLT_method'] = QuantumLinearTheoryMethod.POWER_SERIES_HYBRIDISATION

    return multitrap_kwargs

def calculate_spectra_across_y_range_for_one_frame(
    multitrap_kwargs,
    spectrum_type,
    y,
    frame,
    frame_value,
    method=QuantumLinearTheoryMethod.NUMERICAL_HYBRIDISATION,
    **kwargs,
):
    """
    Given a value of the frame variable, returns a spectrum calculated at each value in the
    y-variable's range
    """
    if spectrum_type == SpectrumType.POSITION_PSD:
        PSD_label = kwargs.get('PSD_label', 'x1')

    # define the value for this frame
    multitrap_kwargs = update_parameter_in_multitrap_kwargs(multitrap_kwargs, frame, frame_value)

    # calculate the spectrum at each y-value
    spectra_across_y_range = np.zeros(shape=(len(multitrap_kwargs['ω_range']), len(y.range)))

    area_across_y_range = []
    phonons_across_y_range = []
    T_ph_across_y_range = []
    T_var_across_y_range = []

    for yy, y_value in enumerate(y.range):
        multitrap_kwargs = update_parameter_in_multitrap_kwargs(multitrap_kwargs, y, y_value)
        mtrap = MultiTrap(**multitrap_kwargs)

        if spectrum_type == SpectrumType.HETERODYNE:
            spectra_across_y_range[:, yy] = mtrap.calculate_heterodyne_spectrum(mtrap.QLT_method)
            area = calculate_area(mtrap, spectra_across_y_range[:, yy])
            area_across_y_range.append(area)

        elif spectrum_type == SpectrumType.HOMODYNE:
            spectra_across_y_range[:, yy] = mtrap.calculate_homodyne_spectrum(mtrap.QLT_method)
            area = calculate_area(mtrap, spectra_across_y_range[:, yy])
            area_across_y_range.append(area)

        elif spectrum_type == SpectrumType.POSITION_PSD:
            _, psd_dict = mtrap.calculate_all_position_power_spectra(mtrap.QLT_method)
            spectra_across_y_range[:, yy] = psd_dict[PSD_label]

            ph, T_ph, T_var = calculate_phonons_and_temperatures_from_PSD(
                mtrap,
                spectra_across_y_range[:, yy],
                PSD_label,
            )
            phonons_across_y_range.append(ph)
            T_ph_across_y_range.append(T_ph)
            T_var_across_y_range.append(T_var)

    info_for_one_frame = {
        'Area': area_across_y_range,
        'Phonons': phonons_across_y_range,
        'Temperature via phonons': T_ph_across_y_range,
        'Temperature via variance': T_var_across_y_range,
    }

    return spectra_across_y_range.T, info_for_one_frame


def title_string_for_one_frame(
    multitrap_kwargs,
    spectrum_type,
    y, frame, frame_value,
    all_parameters,
    **kwargs,
):
    """
    Generate a title that updates each frame
    """
    other_params = copy(all_parameters)
    other_params.pop(y.type)
    other_params.pop(frame.type)
    if spectrum_type != SpectrumType.HOMODYNE:
        other_params.pop(ParameterType.HOMODYNE_ANGLE)

    symbols_str = ''
    defaults_str = ''
    for param in other_params.values():
        fs = param.format_specifier
        symbols_str = symbols_str + f'{param.symbol}, '
        defaults_str = defaults_str + f'{param.default_value / param.units:{param.format_specifier}}{param.unit_symbol}, '

    # remove the final extra ', '
    symbols_str = symbols_str[:-2]
    defaults_str = defaults_str[:-2]

    return (
        f'{multitrap_kwargs["num_traps"]}×{multitrap_kwargs["Cartesian_axes_per_trap"]} system'
        f'\n'
        f'{spectrum_type_to_latex(spectrum_type, **kwargs)}(ω, {y.symbol})'
        f' at '
        f'{frame.symbol}'
        f'='
        f'{frame_value / frame.units:{frame.format_specifier}}{frame.unit_symbol}'
        f'\n'
        f'({symbols_str})'
        f'='
        f'({defaults_str})'
    )


def calculate_spectra_for_all_frames(
    multitrap_kwargs,
    spectrum_type,
    y, frame,
    **kwargs,
):
    all_spectra = []
    info_for_all_frames = []
    global_min = float('inf')
    global_max = float('-inf')
    description = f'{repr(frame.type)} calculation'
    for frame_value in tqdm(frame.range, desc=description):
        spectra, info_for_one_frame = calculate_spectra_across_y_range_for_one_frame(
            multitrap_kwargs,
            spectrum_type,
            y,
            frame,
            frame_value,
            **kwargs,
        )
        info_for_all_frames.append(info_for_one_frame)
        all_spectra.append(spectra)
        global_min = min(global_min, spectra.min())
        global_max = max(global_max, spectra.max())

    limits = global_min, global_max

    return all_spectra, limits, info_for_all_frames

def generate_heatmap_animation_frames(
    multitrap_kwargs,
    all_parameters,
    x, y, frame,
    spectrum_type,

    all_spectra_by_frame,
    limits,
    info_for_all_frames,

    **kwargs,
):

    global_min, global_max = limits

    if spectrum_type == SpectrumType.HOMODYNE:
        squeezed_threshold = 1
        norm_positive = mcolors.LogNorm(vmin=squeezed_threshold, vmax=global_max)
    else:
        norm = mcolors.LogNorm(vmin=global_min, vmax=global_max)

    animation_frames = []
    fig, heatmap_ax = plt.subplots(figsize=(8, 8), dpi=200)
    for ii, frame_value in enumerate(frame.range[::-1]):
        heatmap_ax.set_xlabel('Frequency, ω/2π (kHz)')
        heatmap_ax.set_ylabel(f'{y.symbol} {y.unit_string}')
        title_str = title_string_for_one_frame(
            multitrap_kwargs,
            spectrum_type,
            y, frame,
            frame_value,
            all_parameters,
            **kwargs,
        )
        title = heatmap_ax.text(
            # position above the plot
            0.5, 1.03,
            title_str,
            size='small',
            ha='center',
            transform=heatmap_ax.transAxes,
        )

        # next, generate the heatmap frames
        if spectrum_type == SpectrumType.HOMODYNE:
            S_hom = all_spectra_by_frame[ii]

            # split into squeezed and non-squeezed
            non_squeezed = np.where(S_hom >= squeezed_threshold, S_hom, np.nan)
            squeezed = np.where(S_hom < squeezed_threshold, S_hom, np.nan)

            # plot non-squeezed values on a log scale
            pc_positive = heatmap_ax.pcolormesh(
                x,
                y.axis_range,
                non_squeezed,
                cmap='Wistia_r',
                norm=norm_positive,
                shading='nearest',
            )
            # plot squeezed values on a linear scale
            pc_negative = heatmap_ax.pcolormesh(
                x,
                y.axis_range,
                squeezed,
                cmap='coolwarm',
                vmin=global_min,
                vmax=squeezed_threshold,
                shading='nearest',
            )
            fig.subplots_adjust(bottom=0.2)
            animation_frames.append([pc_positive, pc_negative, title])

        else:
            pc_mesh = heatmap_ax.pcolormesh(
                x,
                y.axis_range,
                all_spectra_by_frame[ii],
                shading='nearest',
                norm=norm,
                cmap='plasma' if spectrum_type == SpectrumType.HETERODYNE else 'viridis',
            )
            animation_frames.append([pc_mesh, title])

    # lastly, add legend/colourbar outside of frame loop
    if spectrum_type == SpectrumType.HOMODYNE:
        # [left, bottom, width, height]
        cax_positive = fig.add_axes([0.1, 0.05, 0.35, 0.03])
        cbar_positive = plt.colorbar(pc_positive, cax=cax_positive, orientation='horizontal')
        cbar_positive.ax.xaxis.set_label_position('top')
        cbar_positive.set_label('Non-Squeezed $S_{hom} ≥ 1$')

        cax_negative = fig.add_axes([0.5, 0.05, 0.35, 0.03])
        cbar_negative = plt.colorbar(pc_negative, cax=cax_negative, orientation='horizontal')
        cbar_negative.ax.xaxis.set_label_position('top')
        cbar_negative.set_label('Squeezed $S_{hom} < 1$')
    else:
        cbar = fig.colorbar(pc_mesh, ax=heatmap_ax, location='right', pad=0.1)
        cbar.set_label(f'{spectrum_type_to_latex(spectrum_type, **kwargs)} {spectrum_type_to_units(spectrum_type)}')

    return fig, animation_frames


def generate_slices_animation_frames(
    multitrap_kwargs,
    all_parameters,
    x, y, frame,
    spectrum_type,

    all_spectra_by_frame,
    limits,
    info_for_all_frames,

    **kwargs,
):
    global_min, global_max = limits

    figsize = kwargs.get('figsize', (6, 6))
    dpi = kwargs.get('dpi', 150)
    fig = plt.figure(layout='constrained', figsize=(6, 6), dpi=150)
    if spectrum_type.is_optical:
        if spectrum_type == SpectrumType.HETERODYNE:
            colours = discrete_plasma[len(y.axis_range)]
        elif spectrum_type == SpectrumType.HOMODYNE:
            colours = discrete_magma[len(y.axis_range)]

        ax_dict = fig.subplot_mosaic(
            [
                ['spectra', 'spectra', 'spectra'],
                ['spectra', 'spectra', 'spectra'],
                ['spectra', 'spectra', 'spectra'],
                ['Area', 'Area', 'Area'],
                ['Area', 'Area', 'Area'],
                ['frame', 'frame', 'frame'],
            ],
        )

    else:
        colours = discrete_viridis[len(y.axis_range)]
        ax_dict = fig.subplot_mosaic(
            [
                ['spectra', 'spectra', 'spectra'],
                ['spectra', 'spectra', 'spectra'],
                ['spectra', 'spectra', 'spectra'],
                ['Phonons', 'Temperature via phonons', 'Temperature via variance'],
                ['Phonons', 'Temperature via phonons', 'Temperature via variance'],
                ['frame', 'frame', 'frame'],
            ],
        )

    spectrum_ax = ax_dict['spectra']
    spectrum_ax.grid(True)
    spectrum_ax.set_yscale('log')

    frame_ax = ax_dict['frame']
    frame_ax.grid(True)
    frame_ax.set_yticks(ticks=[1], labels=[f'{frame.symbol}'])
    frame_ax.set_xlabel(f'{frame.unit_string}')

    animation_frames = []
    for frame_index, frame_value in enumerate(frame.range):
        spectrum_ax.set_xlabel('Frequency, ω/2π (kHz)')
        spectrum_ax.set_ylabel(f'{spectrum_type_to_latex(spectrum_type, **kwargs)}(ω) {spectrum_type_to_units(spectrum_type)}')
        title_str = title_string_for_one_frame(
            multitrap_kwargs,
            spectrum_type,
            y, frame,
            frame_value,
            all_parameters,
            **kwargs,
        )
        title = spectrum_ax.text(
            # position above the plot
            0.5, 1.03,
            title_str,
            # size='large',
            ha='center',
            transform=spectrum_ax.transAxes,
        )
        frame_components = [title]

        frame_bar_container = frame_ax.barh(1, frame_value / frame.units, color='k')
        frame_bar = frame_bar_container.get_children()[0]
        frame_components.append(frame_bar)

        # plot in reverse order so highest order is on top
        for y_index, spectrum in enumerate(all_spectra_by_frame[frame_index]):
            line, = spectrum_ax.plot(
                x, spectrum,
                color=colours[y_index],
                # lw=8 - 2 * y_index,
                # ls=next(linestyles),
                label=f'{y.range[y_index] / y.units:{y.format_specifier}}' if frame_index == 0 else '',
            )
            frame_components.append(line)
        spectrum_ax.legend(
            title=f'{repr(y.type)} {y.unit_string}',
            prop={'size': 'x-small'},
            # fontsize='x-small',
        )

        if spectrum_type.is_optical:
            info_keys = ['Area']
        else:
            info_keys = [
                'Phonons',
                'Temperature via phonons',
                'Temperature via variance',
            ]
        for key in info_keys:
            info_ax = ax_dict[key]
            info_ax.set_title(f'(Normalised)\n{key}', size='small')
            info_ax.set_xlabel(f'{repr(y.type)}')
            # info_ax.set_yscale('symlog', linthresh=0.1)
            info_ax.set_yscale('log')
            info_ax.set_ylim(1e-1, 1e1)
            info_ax.tick_params(axis='y', labelsize='x-small')
            info_ax.set_xticks(y.range)
            info_ax.grid(True)

            max_order_info = info_for_all_frames[frame_index][key][-1]
            for y_index, info in enumerate(info_for_all_frames[frame_index][key]):
                bar_container = info_ax.bar(
                    y.range[y_index],
                    info / max_order_info,
                    color=colours[y_index],
                )
                bar = bar_container.get_children()[0]
                frame_components.append(bar)

        animation_frames.append(frame_components)

    return fig, animation_frames


def animate_spectra(
    animation_type,
    multitrap_kwargs,
    all_parameters,
    x, y, frame,
    spectrum_type,
    **kwargs,
):
    all_spectra_by_frame, limits, info_for_all_frames = calculate_spectra_for_all_frames(
        multitrap_kwargs,
        spectrum_type,
        y, frame,
        **kwargs,
    )

    frame_func = {
        'heatmap': generate_heatmap_animation_frames,
        'slices': generate_slices_animation_frames,
    }[animation_type]

    fig, animation_frames = frame_func(
        multitrap_kwargs,
        all_parameters,
        x, y, frame,
        spectrum_type,

        all_spectra_by_frame,
        limits,
        info_for_all_frames,

        **kwargs,
    )

    # make each animation 5s long
    interval = kwargs.get('animation_length_ms', 5_000) // len(frame.range)
    logger.debug(f'animation frame {interval=} ms')
    if kwargs.get('reverse_animation', False):
        animation_frames = animation_frames[::-1]
    animation = ArtistAnimation(fig, animation_frames, interval=interval, blit=False)

    if kwargs.get('save_mp4', False):
        if spectrum_type.is_optical:
            label = repr(spectrum_type)
        else:
            label = f'{kwargs.get("PSD_label")} PSD'

        num_traps = multitrap_kwargs['num_traps']
        Cartesian_axes_per_trap = multitrap_kwargs['Cartesian_axes_per_trap']
        N = num_traps * Cartesian_axes_per_trap

        save_name = (
            f'{N}D'
            f' {num_traps}x{Cartesian_axes_per_trap} system'
            f' - {repr(y.type)} vs {repr(frame.type)}'
            f' {label}'
            f' {animation_type}'
            f'.mp4'
        )
        animation.save(save_name, writer='ffmpeg')

    return animation

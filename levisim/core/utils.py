__all__ = [
    # folders where data is kept
    'CALIBRATION_DATA_PATH_ROOT',
    'DIRECTED_FORCE_DATA_PATH_ROOT',
    'KICKS_DATA_PATH_ROOT',

    # binary pickle files
    'save_variable_to_pickle',
    'load_variable_from_pickle',

    # time arrays
    'generate_time_array',
    'time_slice',

    # pyplot animations
    'create_func_animation',
    'create_artist_animation',

    # signal processing
    'apply_filter',
    'calculate_contrast',

    # kicks
    'kick_data_path',
    'generate_kick_list',
    'find_kicks',

    # MultiTrap QLT spectra comparison tools
    'greyscale_diff',
    'compare_methods',
]

import cv2
import pickle
import scipy.signal
import matplotlib.animation

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.path import Path
from IPython.display import HTML
from pprint import pprint as pp

from levisim.core.constants import ħ, kB
from levisim.core.units import Hz, kHz
from levisim.physics.kick import Kick
from levisim.physics.multitrap import QuantumLinearTheoryMethod as QLTM


####################################################################################################
# data to calculate the V/m calibration factor
CALIBRATION_DATA_PATH_ROOT = ''

# data with directed stochastic noise at varying angles
DIRECTED_FORCE_DATA_PATH_ROOT = '/Users/julian/Documents/PhD/projects/2023-01 -- C15 Experiment - Sensing Force Directionality in Cross-Correlation Spectra/C15-final-data/'

# data with kicks of varying strengths
KICKS_DATA_PATH_ROOT = '/Users/julian/Downloads/jon-kick-data/Kicksforcalibration/'


####################################################################################################
# binary pickle files

def save_variable_to_pickle(variable, save_name='default.pickle'):
    """ Save any given python variable to a binary file """
    print(f'Saving {variable} to "{save_name}"')
    file = open(save_name, 'wb')  # 'wb' = write binary
    pickle.dump(variable, file)
    file.close()
    return


def load_variable_from_pickle(save_name):
    """ Load any previously saved python variable from a binary file """
    print(f'Loading variable from "{save_name}"')
    file = open(save_name, 'rb')  # 'rb' = read binary
    variable = pickle.load(file)
    file.close()
    return variable


####################################################################################################
# time arrays
def generate_time_array(start_time, end_time, num_samples):
    """ Returns a `num_samples + 1` length array of time values, i.e. including `end_time` """
    return np.linspace(start_time, end_time, num_samples + 1, endpoint=True)


def time_slice(array, start_time, end_time, Δt):
    """
    Returns a slice of a time series `array` (of resolution `Δt`)
    from a given `start_time` to `end_time`
    """
    start_index = int(start_time // Δt)
    end_index = int(end_time // Δt)
    return array[start_index:end_index]


####################################################################################################
# pyplot animations

def create_func_animation(
    fig,
    func,

    frames=None,
    init_func=None,
    fargs=None,

    interval=200,
    blit=True,
):
    anim = matplotlib.animation.FuncAnimation(
        fig,
        func,

        frames=frames,
        init_func=init_func,
        fargs=fargs,

        interval=interval,
        blit=blit,
    )
    return anim


# TODO: set this up! https://matplotlib.org/stable/users/explain/animations/animations.html
def create_artist_animation(
    fig,
    artists,
    interval=200,
):
    anim = matplotlib.animation.ArtistAnimation(
        fig,
        artists,
        interval=interval,
    )
    return anim


####################################################################################################
# signal processing

def apply_filter(signal, Nyquist_freq, centre_freq, filter_type, bandwidth, plot=True):
    kiloHz, MegaHz = 1e3, 1e6
    print(f'Signal has {len(signal)} samples and Nyquist frequency={Nyquist_freq / MegaHz} MHz')

    if filter_type == 'butter':
        order = 5
        print(f'Making a {bandwidth / kiloHz}kHz Butterworth bandpass filter, centred @ {centre_freq / kiloHz}kHz ({order}th order)')
        lowcut = centre_freq - bandwidth / 2
        highcut = centre_freq + bandwidth / 2
        low, high = lowcut / Nyquist_freq, highcut / Nyquist_freq
        b, a = scipy.signal.butter(order, [low, high], btype='band')

    elif filter_type == 'iirpeak':
        Q_factor = centre_freq / bandwidth
        print(f'Making a {bandwidth / kiloHz}kHz linewidth IIR peak filter, centred @ {centre_freq / kiloHz}kHz (Q={Q_factor})')
        b, a = scipy.signal.iirpeak(centre_freq, Q_factor, fs=2*Nyquist_freq)

    if plot:
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.iirpeak.html#scipy.signal.iirpeak
        freq, h = scipy.signal.freqz(b, a, fs=2*Nyquist_freq)
        _, ax = plt.subplots(2, 1, figsize=(8, 6))
        ax[0].plot(freq, 20*np.log10(np.maximum(abs(h), 1e-5)), '.-', color='blue')
        ax[0].set_title("Frequency Response")
        ax[0].set_ylabel('Amplitude (dB)')
        ax[0].grid(True)
        ax[1].plot(freq, np.unwrap(np.angle(h, deg=True)), '.-', color='green')
        ax[1].set_ylabel('Angle (degrees)')
        ax[1].set_xlabel('Frequency (Hz)')
        ax[1].grid(True)
        plt.show()

    # TODO: implement sosfilt ?
    return scipy.signal.filtfilt(b, a, signal)


def calculate_contrast(data):
    """ Calculates the ratio of the maximum to the root-mean-square of given data """
    abs_max = np.max(np.abs(data))
    rms = np.sqrt(np.mean(np.abs(data)**2))
    return abs_max / rms


####################################################################################################
# kicks

def kick_data_path(gain, run_choice):
    """ Returns the data path """
    root = KICKS_DATA_PATH_ROOT + '20231124_KickTest+45DegreeGain'
    folder_path_per_gain = {
        0.1: root + '0_1/',
        0.3: root + '0_3/',
        0.6: root + '0_6/',
        1.2: root + '1_2/',
        1.6: root + '1_6/',
        2.4: root + '2_4/',
        3.0: root + '3/',
    }
    file_name_root_per_gain = {
        0.1: '20231124_171458_343490KickTest+45DegreeGain0_1_000',
        0.3: '20231124_172007_930760KickTest+45DegreeGain0_3_000',
        0.6: '20231124_172358_645377KickTest+45DegreeGain0_6_000',
        1.2: '20231124_174201_881084KickTest+45DegreeGain1_2_000',
        1.6: '20231124_175042_798883KickTest+45DegreeGain1_6_000',
        2.4: '20231124_180556_599698KickTest+45DegreeGain2_4_000',
        3.0: '20231124_181505_502314KickTest+45DegreeGain3_000',
    }
    runs_per_gain = {
        0.1: 15,
        0.3: 13,
        0.6: 13,
        1.2: 13,
        1.6: 13,
        2.4: 14,
        3.0: 14,
    }

    assert 0 < int(run_choice) <= runs_per_gain[gain], f'For gain {gain} choice must be between 1 and {runs_per_gain[gain]}'
    print(f'Getting data path for kicks with gain {gain} -- run choice {run_choice}')
    return folder_path_per_gain[gain] + file_name_root_per_gain[gain] + run_choice + '.bin'


def generate_kick_list(
    num_kicks=None,
    starts=None,
    durations=None,
    strengths=None,  # multiple of to the thermal gas momentum
    angles=None,  # radians in x-y plane
):
    """ Provides a list of Kicks that can be assigned to the kicks attribute of a Trap instance """
    # default to returning an empty list
    if num_kicks is None:
        return []

    num_kicks = int(num_kicks)

    # if any of the variables are single values and not indexable arrays / lists, make them so
    # and apply their value to every kick
    if type(starts) in [float, int]:
        starts = [starts for _ in range(num_kicks)]
    else:
        assert len(starts) == num_kicks

    if type(durations) in [float, int]:
        durations = [durations for _ in range(num_kicks)]
    else:
        assert len(durations) == num_kicks

    if type(strengths) in [float, int]:
        strengths = [strengths for _ in range(num_kicks)]
    else:
        assert len(strengths) == num_kicks

    if type(angles) in [float, int]:
        angles = [angles for _ in range(num_kicks)]
    else:
        assert len(angles) == num_kicks

    # TODO: see if this works
    # for var in [starts, durations, strengths, angles]:
    #     if type(var) in [float, int]:
    #         var = [var for _ in range(num_kicks)]
    #     else:
    #         assert len(var) == num_kicks

    return [
        Kick(
            start=starts[ii],
            duration=durations[ii],
            K=strengths[ii],
            θ=angles[ii],
        )
        for ii in range(num_kicks)
    ]


def find_kicks(
    kick_trace,
    number_of_kicks_to_look_for,
    plot=True,
):
    n = number_of_kicks_to_look_for

    # arg partition sorts lowest to highest by default so make all the values negative
    kick_indices = np.argpartition(-kick_trace.values, n)[:n]

    # doing the diff means even if the voltage kick is negative, it should be picked out by this sorting
    kick_indices_2 = np.argpartition(np.diff(-kick_trace.values), n)[:n]

    # TODO: some sort of cross of the two above should rule out any spurious detections...
    # aa = set(kick_indices) & set(kick_indices_2)
    # bb = np.array(kick_indices) - np.array(kick_indices_2)

    kick_times = kick_trace.time_array[kick_indices]
    kick_times_2 = kick_trace.time_array[kick_indices_2]
    kick_voltages = kick_trace.values[kick_indices]

    if plot:
        for kick_time, kick_voltage in zip(kick_times, kick_voltages):
            plt.vlines(kick_time, 0, kick_voltage)

        plt.plot(0, 0, '-')
        # plt.plot(max(kick_data.time.get_array()), 0, '-')
        plt.plot(np.max(kick_trace.time_array), 0, '-')
        plt.plot(kick_times, kick_voltages, 'x', label='max-to-min sorting')
        plt.plot(kick_times_2, kick_voltages, 'x', label='min-to-max sorting of the diffs' )
        plt.title(f'Looking for {n} kicks...')
        plt.legend()
        plt.show()

    return kick_times, kick_times_2, kick_voltages

####################################################################################################
# MultiTrap QLT spectra comparison tools

def greyscale_diff(image_1, image_2):
    return cv2.cvtColor(cv2.subtract(image_1, image_2), cv2.COLOR_BGR2GRAY)

def compare_methods(multitrap_instance, methods_to_compare, spectra_to_compare):
    mph = multitrap_instance
    methods = methods_to_compare
    spectral_types = spectra_to_compare
    N, ax = mph.num_traps, mph.Cartesian_axes_per_trap

    if QLTM.TNT_LINEAR_AMPLIFIER_MODEL in methods:
        spectral_types = ['PSDs']

    ω_low = min(mph.ω_range) / kHz
    ω_high = max(mph.ω_range) / kHz
    ω_step = (mph.ω_range[1] - mph.ω_range[0]) / kHz

    image_dicts = {}
    all_plot_values = {}
    for spectral_type in spectral_types:
        image_dicts[spectral_type] = {}
        for method in methods:
            all_plot_values[method] = {}
            print(f'{spectral_type} -- {method}')

            label = spectral_type
            if spectral_type == 'PSDs':
                calc_func = mph.calculate_all_hybridised_position_spectra
                label = mph.labels
            elif spectral_type == 'homodyne':
                calc_func = mph.calculate_homodyne_spectrum
            elif spectral_type == 'heterodyne':
                calc_func = mph.calculate_heterodyne_spectrum

            plot_values = calc_func(method)
            all_plot_values[method][spectral_type] = plot_values

            if spectral_type == 'PSDs' and N >= 3 and ax >= 2:
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
                    PSD_ax.plot(mph.ω_range / kHz, plot_values[:, ii], label=label[ii])
                    PSD_ax.set_yscale('log')
                    PSD_ax.legend()
                    PSD_ax.grid()
                    PSD_ax.set_xlim(ω_low, ω_high)
                    PSD_ax.set_ylim(1e-7, 1e1)
                plt.show()

            plt.plot(
                mph.ω_range / kHz,
                plot_values,
                label=label,
            )
            plt.yscale('log')
            plt.legend()
            plt.grid()
            plt.xlim(ω_low, ω_high)
            if spectral_type == 'PSDs':
                plt.ylim(1e-8, 1e1)
            figname = f'{spectral_type} -- {method}.png'
            plt.savefig(figname)
            image_dicts[spectral_type][method] = cv2.imread(figname)
            plt.show()

            if spectral_type == 'PSDs':
                for ii, label in enumerate(mph.labels):
                    jn = mph.dimensions[label]
                    ħΩ = ħ * jn.Ω

                    SQQ_area = np.trapz(plot_values[:, ii], mph.ω_range / Hz)
                    njn = 0.5 * (SQQ_area - 1)
                    print(f'{label=}')
                    print(f'n{label} = {njn:.6f} phonons')
                    print(f'T{label} via phonons = {njn * ħΩ / kB:.6e} K')

                    eqp_f = jn.m * jn.Ω**2 / kB
                    q_variance = SQQ_area * jn.zpf**2
                    print(f'T{label} via variance = {eqp_f * q_variance:.6e} K')


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
    for spectral_type, image_dict in image_dicts.items():

        results_dict[spectral_type] = {}

        for method_pair in method_pairs:
            diff = greyscale_diff(image_dict[method_pair[0]], image_dict[method_pair[1]])
            result = cv2.countNonZero(diff) == 0

            all_results.append(result)
            results_dict[spectral_type][method_pair] = result

    assert all(all_results), f'🏴‍☠️🏴‍☠️🏴‍☠️🏴‍☠️🏴‍☠️ Failure present! 🏴‍☠️🏴‍☠️🏴‍☠️🏴‍☠️🏴‍☠️\n{results_dict=}'
    print('🎉🎉🎉🎉🎉 all spectral calculation methods agree :) 🎉🎉🎉🎉🎉')
    pp(results_dict)


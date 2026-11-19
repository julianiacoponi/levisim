"""
General helpers: pickling, time arrays and slices, filtering, and generating and finding kicks
"""

import pickle

import numpy as np
import matplotlib.pyplot as plt

from levisim.core.logger import logger
from levisim.core.constants import ħ, kB
from levisim.core.units import Hz, kHz, kiloHz, megaHz

from levisim.physics.kick import Kick


__all__ = [
    # folders where experimental data is kept
    'CALIBRATION_DATA_PATH_ROOT',
    'DIRECTED_FORCE_DATA_PATH_ROOT',
    'KICKS_DATA_PATH_ROOT',

    # binary pickle files
    'save_variable_to_pickle',
    'load_variable_from_pickle',

    # time arrays
    'generate_time_array',
    'time_slice',

    # signal processing
    'apply_filter',
    'calculate_contrast',

    # kicks
    'kick_data_path',
    'generate_kick_list',
    'find_kicks',
]


####################################################################################################
# data to calculate the V/m calibration factor
# --------------------------------------------------------------------------------------------------
CALIBRATION_DATA_PATH_ROOT = ''

# data with directed stochastic noise at varying angles
DIRECTED_FORCE_DATA_PATH_ROOT = '/Users/julian/Documents/PhD/projects/2023-01 -- C15 Experiment - Sensing Force Directionality in Cross-Correlation Spectra/C15-final-data/'

# data with kicks of varying strengths
KICKS_DATA_PATH_ROOT = '/Users/julian/Downloads/jon-kick-data/Kicksforcalibration/'


####################################################################################################
# binary pickle files
# --------------------------------------------------------------------------------------------------

def save_variable_to_pickle(variable, save_name='default.pickle'):
    """ Save any given python variable to a binary file """
    logger.info('Saving variable to "{}"', save_name)
    file = open(save_name, 'wb')  # 'wb' = write binary
    pickle.dump(variable, file)
    file.close()
    return


def load_variable_from_pickle(save_name):
    """ Load any previously saved python variable from a binary file """
    logger.debug('Loading variable from "{}"', save_name)
    file = open(save_name, 'rb')  # 'rb' = read binary
    variable = pickle.load(file)
    file.close()
    return variable


####################################################################################################
# time arrays
# --------------------------------------------------------------------------------------------------
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
# signal processing
# --------------------------------------------------------------------------------------------------
def apply_filter(signal, Nyquist_freq, centre_freq, filter_type, bandwidth, plot=True):
    """
    Zero-phase (forward-backward) band-pass filtering of a signal
    filter_type: 'butter' (5th-order Butterworth band-pass of the given bandwidth) or 'iirpeak'
    (2nd-order IIR peak of quality factor centre_freq / bandwidth)
    The filter is applied as second-order sections (scipy.signal.sosfiltfilt), which avoids the
    round-off problems of the transfer-function (b, a) form for high orders or narrow bands.
    [units of signal]
    """
    # imported here, not at the top: scipy.signal takes about half a second to import
    import scipy.signal

    logger.debug('Filtering a signal of {} samples with Nyquist frequency {} MHz', len(signal), Nyquist_freq / megaHz)

    if filter_type == 'butter':
        order = 5
        logger.debug('Making a {} kHz Butterworth bandpass filter, centred at {} kHz (order {})', bandwidth / kiloHz, centre_freq / kiloHz, order)
        lowcut = centre_freq - bandwidth / 2
        highcut = centre_freq + bandwidth / 2
        low, high = lowcut / Nyquist_freq, highcut / Nyquist_freq
        sos = scipy.signal.butter(order, [low, high], btype='band', output='sos')

    elif filter_type == 'iirpeak':
        Q_factor = centre_freq / bandwidth
        logger.debug('Making a {} kHz linewidth IIR peak filter, centred at {} kHz (Q = {})', bandwidth / kiloHz, centre_freq / kiloHz, Q_factor)
        b, a = scipy.signal.iirpeak(centre_freq, Q_factor, fs=2*Nyquist_freq)
        sos = scipy.signal.tf2sos(b, a)

    if plot:
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.iirpeak.html#scipy.signal.iirpeak
        freq, h = scipy.signal.sosfreqz(sos, fs=2*Nyquist_freq)
        _, ax = plt.subplots(2, 1, figsize=(8, 6))
        ax[0].plot(freq, 20*np.log10(np.maximum(abs(h), 1e-5)), '.-', color='blue')
        ax[0].set_title('Frequency Response')
        ax[0].set_ylabel('Amplitude [dB]')
        ax[0].grid(True)
        ax[1].plot(freq, np.unwrap(np.angle(h, deg=True)), '.-', color='green')
        ax[1].set_ylabel('Angle [degrees]')
        ax[1].set_xlabel('Frequency [Hz]')
        ax[1].grid(True)
        plt.show()

    # NOTE: implemented sosfilt (as sosfiltfilt, keeping the zero-phase forward-backward filtering of
    # the earlier filtfilt(b, a)). Away from the edges the outputs agree to round-off (see
    # tests/test_simulation_data_utils.py); the default edge padding differs slightly between the two.
    return scipy.signal.sosfiltfilt(sos, signal)


def calculate_contrast(data):
    """ Calculates the ratio of the maximum to the root-mean-square of given data """
    abs_max = np.max(np.abs(data))
    rms = np.sqrt(np.mean(np.abs(data)**2))
    return abs_max / rms


####################################################################################################
# kicks
# --------------------------------------------------------------------------------------------------
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
    logger.debug('Getting the data path for kicks with gain {}, run choice {}', gain, run_choice)
    return folder_path_per_gain[gain] + file_name_root_per_gain[gain] + run_choice + '.bin'


def _per_kick_values(values, num_kicks, name):
    """
    Returns `values` as a list of length num_kicks: a scalar (int, float or numpy scalar) is repeated,
    a sequence must already have num_kicks entries
    """
    if np.ndim(values) == 0:
        return [values for _ in range(num_kicks)]
    values = list(values)
    assert len(values) == num_kicks, f'{name} has {len(values)} entries, expected {num_kicks}'
    return values


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
    # NOTE: replaces a commented loop `for var in [...]: var = [...]`, which could not work: rebinding
    # the loop variable does not change the list entries (or starts, durations, ...)
    starts = _per_kick_values(starts, num_kicks, 'starts')
    durations = _per_kick_values(durations, num_kicks, 'durations')
    strengths = _per_kick_values(strengths, num_kicks, 'strengths')
    angles = _per_kick_values(angles, num_kicks, 'angles')

    return [
        Kick(
            start=starts[ii],
            duration=durations[ii],
            K=strengths[ii],
            θ=angles[ii],
        )
        for ii in range(num_kicks)
    ]


def cross_check_kick_indices(indices_a, indices_b, tolerance=10):
    """
    Indices of indices_a that have a partner in indices_b within ±tolerance samples (sorted)
    Used by find_kicks to flag detections that only one of its two methods finds.
    [samples]
    """
    indices_b = np.sort(np.asarray(indices_b))
    confirmed = [
        i for i in np.sort(np.asarray(indices_a))
        if len(indices_b) and np.min(np.abs(indices_b - i)) <= tolerance
    ]
    return np.array(confirmed, dtype=int)


def find_kicks(
    kick_trace,
    number_of_kicks_to_look_for,
    plot=True,
    tolerance=10,
):
    """
    Finds the times of the n largest kicks in a kick-monitor trace, by two methods:
    1. the n largest values, and 2. the n steepest rises (largest positive sample-to-sample steps)
    and logs a warning if the two disagree (see cross_check_kick_indices, ±tolerance samples).
    Returns (kick_times, kick_times_2, kick_voltages) for methods 1 and 2.
    [s, s, units of the trace]
    """
    n = number_of_kicks_to_look_for

    # arg partition sorts lowest to highest by default so make all the values negative
    kick_indices = np.argpartition(-kick_trace.values, n)[:n]

    # doing the diff means even if the voltage kick is negative, it should be picked out by this sorting
    kick_indices_2 = np.argpartition(np.diff(-kick_trace.values), n)[:n]

    # cross-check the two index sets: a real kick should be found by both methods within a few samples
    # (the steepest rise sits just before the maximum); detections found by only one are suspect
    confirmed = cross_check_kick_indices(kick_indices, kick_indices_2, tolerance)
    if len(confirmed) < n:
        logger.warning(
            f'find_kicks: only {len(confirmed)} of {n} kicks were found by both the max and the steepest-rise '
            f'sorting (within {tolerance} samples); the others may be spurious'
        )

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

import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat


from levisim.core.units import (
    Megasamples,
    ms, ns,
    MegaHz,
)
from levisim.core.utils import generate_time_array
from levisim.physics.physics import Physics
from levisim.data.trace import Trace


class Experiment:
    """ Class for importing the experimental data """

    def __init__(
        self,
        data_path,
        **kwargs,
    ):
        self.data_path = data_path

        self.channel_dict = kwargs.get('channel_dict', None)
        self.V_per_m_calibrations_dict = kwargs.get('V_per_m_calibrations_dict', {})
        self.physics = kwargs.get('physics', Physics())

        self.file_name = self.data_path.split('/')[-1]
        self.file_extension = self.data_path.split('.')[-1]

        self.traces = {}
        self.load_data()

    @property
    def info(self):
        """ Prints info on the trace """
        print(
            f'Experiment(id={id(self)}'
            f'\n    file_name = {self.file_name}'
            f'\n    channel_dict = {self.channel_dict}'
            f'\n    V_per_m_calibrations_dict = {self.V_per_m_calibrations_dict}'
            f'\n'
            f'\n    num_samples = {self.num_samples / Megasamples:.6f} Megasamples'
            f'\n    Δt = {self.Δt / ns:.6f} ns'
            f'\n'
            f'\n    sample_rate = {self.sample_rate:.6f} Hz'
            f'\n    Nyquist_frequency = {self.Nyquist_frequency / MegaHz:.6f} MHz'
            f'\n'
            f'\n    time_length = {self.time_length / ms:.6f} ms'
            f'\n    min_Δf = {self.min_Δf:.6f} Hz'
            f'\n)'
        )

    @property
    def time_length(self):
        """ Time length of the experimental run (seconds) """
        return self.num_samples * self.Δt

    @property
    def min_Δf(self):
        """ Minimum frequency resolution the time trace is capable of producing (Hertz) """
        return 1 / self.time_length

    @property
    def sample_rate(self):
        """ Sample rate of the experiment - how many samples are taken per second? (samples/second) """
        return 1 / self.Δt

    @property
    def Nyquist_frequency(self):
        """
        The Nyquist frequency, a.k.a. the bandwidth of the experimental data collection
        This will be the largest frequency in the spectral density estimate
        NOTE: https://en.wikipedia.org/wiki/Nyquist_frequency
        """
        return self.sample_rate / 2

    @property
    def time_array(self):
        """ Array of all the time steps in the experimental data """
        return generate_time_array(0, self.time_length, self.num_samples)

    def load_data(self):
        """
        Runs correct load function given the `file_extension`
        The aim of each of these functions is
        - to acquire `num_samples` and `Δt`
        - to assign the trace data to the traces dict
        """
        return {
            'mat': self.load_matfile(),
            'bin': self.load_binfile(),
        }[self.file_extension]


    def load_matfile(self):
        """ Load time traces from a .mat file """
        if self.channel_dict is None:
            # these are letters corresponding to data capture channels on the Picoscope
            # e.g. 4 channels A, B, C & D (channel C is unused here)
            self.channel_dict = {
                'x': 'B',
                'y': 'A',
                'kick': 'D',
            }

        matfile_data = loadmat(self.data_path)
        self.num_samples = matfile_data['Length'][0, 0]
        self.Δt = matfile_data['Tinterval'][0, 0]

        for axis, channel in self.channel_dict.items():
            trace_values = matfile_data[channel][:, 0]
            self.assign_trace(trace_values, 0, self.time_length, axis, 'V')

    def load_binfile(self):
        """ Load time traces from a .bin file - adapted from optoanalysis.DataObject.load_time_data """
        if self.channel_dict is None:
            # these are channel numbers relative to the total number captured
            # e.g. 4 channels 0, 1, 2, & 3 (channel 2 is unused here)
            self.channel_dict = {
                'x': 0,
                'y': 1,
                'kick': 3,
            }

        # get the streaming parameters - sample rate, number of channels
        streaming_params_file = self.data_path.replace(self.file_name, '') + 'streaming_parameter.log'
        for line in open(streaming_params_file, 'r'):
            _, sample_rate, num_channels, _, _ = line.split(';')[0:]
            num_channels = int(num_channels)
            # slightly roundabout way, oh well
            self.Δt = 1 / float(sample_rate)

        data = np.fromfile(self.data_path, dtype='int16')

        for axis, channel in self.channel_dict.items():
            trace_values = data[channel:len(data):num_channels]
            self.assign_trace(trace_values, 0, self.time_length, axis, 'V')


    def assign_trace(self, values, start, end, axis, units):
        """ Assign the values of a trace to the trace dict according to the channel dict"""
        V_per_m = self.V_per_m_calibrations_dict.get(axis)
        if V_per_m:
            # converts Volts to metres
            values = values / V_per_m
            units = 'm'

        self.traces[axis] = Trace(
            values,
            start, end, self.Δt,
            axis, units, self,
        )



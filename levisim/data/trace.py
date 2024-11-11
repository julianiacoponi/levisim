import numpy as np

from levisim.core.units import (
    Megasamples,
    ms, ns,
)
from levisim.core.utils import generate_time_array, calculate_contrast


class Trace:
    """
    Class to represent a time trace a.k.a. time series data that helps keep track of:
        - values: all values of the time trace

        - start_time: what point in time does this trace start
        - end_time: what point in time does this trace end
        - Δt: the time step of the time series

        - axis: degree of freedom measured in the trace
            e.g. 'x', 'y', 'z', 'px', 'py', 'pz', 'kick'
        - units:
            'm' for simulation of x,y,z translation in metres
            'kg.m/s' for simulation of px,py,pz momentum in kilogram-metres-per-second
            'V' for experimental data in Volts
        - source: either of type Simulation, or Experiment
    """

    def __init__(
        self,
        values,

        start_time,
        end_time,
        Δt,

        axis,
        units,
        source,

        zero_mean=True,
    ):
        if axis == 'kick':
            zero_mean = False
        self.zero_mean = zero_mean

        self.mean = np.mean(values)
        if zero_mean:
            self.values = values - self.mean
        else:
            self.values = values

        self.start_time = float(start_time)
        self.end_time = float(end_time)
        self.Δt = float(Δt)

        self.axis = axis
        self.units = units
        self.source = source

    @property
    def info(self):
        """ Prints info on the trace """
        print(
            f'Trace(id={id(self)}'
            f'\n    start = {self.start_time / ms:.6f} ms'
            f'\n    end = {self.end_time / ms:.6f} ms'
            f'\n    Δt = {self.Δt / ns:.6f} ns'
            f'\n'
            f'\n    time_length = {self.time_length / ms:.6f} ms'
            f'\n    num_samples = {self.num_samples / Megasamples:.6f} Megasamples'
            f'\n    min_Δf = {self.min_Δf:.6f} Hz'
            f'\n'
            f'\n    mean = {self.mean:.3f}'
            f'\n    contrast = {self.contrast:.3f}'
            f'\n    zero_mean = {self.zero_mean}'
            f'\n'
            f'\n    axis = {self.axis}'
            f'\n    units = {self.units}'
            f'\n    source = {self.source}'
            f'\n)'
        )

    @property
    def num_samples(self):
        """ Number of samples in the trace """
        return len(self.values)

    @property
    def time_length(self):
        """ Time length of the trace (seconds)"""
        return self.end_time - self.start_time

    @property
    def sample_rate(self):
        """ Sample rate of the trace - how many samples are taken per second? (samples/second) """
        return 1 / self.Δt

    @property
    def Nyquist_frequency(self):
        """
        The Nyquist frequency, a.k.a. the bandwidth of the time trace
        This will be the largest frequency in the spectral density estimate
        NOTE: https://en.wikipedia.org/wiki/Nyquist_frequency
        """
        return self.sample_rate / 2

    @property
    def min_Δf(self):
        """ Minimum frequency resolution the time trace is capable of producing (Hertz) """
        return 1 / self.time_length

    @property
    def time_array(self):
        """ Array of all the time steps in the trace (seconds)"""
        return generate_time_array(self.start_time, self.end_time, self.num_samples)

    @property
    def contrast(self):
        """ Ratio of the absolute maximum of the data to its root-mean-square value """
        return calculate_contrast(self.values)


class TracePair:
    """
    Class to represent a pair of Traces, to help calculate
    - PSDs, trace_1 = trace_2
    - CSDs, trace_1 ≠ trace_2
    """
    def __init__(self, trace_1, trace_2=None):

        self.num_samples = trace_1.num_samples
        self.Δt = trace_1.Δt
        self.time_length = trace_1.time_length
        self.time_array = trace_1.time_array

        self.Nyquist_frequency = trace_1.Nyquist_frequency
        self.physics = trace_1.source.physics

        if trace_2 is None:
            trace_2 = trace_1

        self.traces = {
            trace_1.axis: trace_1,
            trace_2.axis: trace_2,
        }
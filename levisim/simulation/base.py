"""
StochasticSimulation: the parts of a stochastic time-domain simulation that do not depend on the
physical system

A subclass describes the system through a small set of hooks (see `StochasticSimulation`); the base
class owns the time axis, the random numbers and their seeding, the chunked noise generation, the
choice of integrator and backend, the recording into `Trace` objects (including a white detection
floor), saving and the summary printed by `info`.
"""

import time
from os import makedirs

import numpy as np

from levisim.core.logger import logger
from levisim.core.units import (
    megasamples,
    ms,
    μs,
)
from levisim.core.utils import generate_time_array, save_variable_to_pickle
from levisim.data.trace import Trace
from levisim.simulation.backends import (
    BACKENDS,
    resolve_backend,
    unsupported_backend_error,
)
from levisim.simulation.integrators import get_integrator

__all__ = [
    'StochasticSimulation',
]



def _no_hook():
    """Stand-in for an absent optional hook"""


class StochasticSimulation:
    """
    Base class of levisim's stochastic time-domain simulations, dX = F(t, X) dt + G dW

    Common keyword arguments (defaults in brackets)::

        num_samples [1e6]: number of time steps
        Δt [0.1 μs]: time step [s]
        RK_method ['vector_4']: 'vector_4' (Kasdin stochastic RK4) or 'vector_2' (improved Euler,
            Heun), or an Integrator instance (see levisim.simulation.integrators)
        seed [None]: None uses numpy's legacy global RandomState (reproducible with np.random.seed);
            an int (or SeedSequence / Generator) uses np.random.default_rng(seed)
        backend ['auto']: 'numpy' (the reference time loop), 'numba' (compiled kernels, needs
            `pip install levisim[fast]`; ImportError without numba) or 'auto' (numba when it is
            installed and the run has at least AUTO_NUMBA_MIN_STEPS = 2¹⁴ steps, NumPy otherwise).
            Both consume the same random numbers; trajectories agree to round-off. With 'auto' a
            configuration without a compiled kernel falls back to NumPy; an explicit 'numba' raises
            UnsupportedBackendError for it instead. self.backend_used says which ran.
        detection_floor [0]: one-sided white noise PSD added to the recorded traces only (a number,
            or a dict per trace label) [trace units²/Hz]
        zero_mean [True]: subtract each trace's mean
        save, save_dir, save_description, save_fmt: optional pickling of the instance
        debug, rk_loop_debug [False]: extra DEBUG logging, and per-step TRACE logging of the loop

    A subclass provides the system through these hooks:

    - `configure(**kwargs)`: read the system's own keyword arguments (called before integration)
    - `initial_state()`: the initial state vector X0
    - `drift(t, X)`: the deterministic part F(t, X)
    - `noise_matrix()`: the constant noise matrix G, shape (len(X0), number of noises)
    - `noise_spectral_densities()`: the two-sided spectral densities S_k of the independent noises
    - `record(X)`: turn the integrated states (shape (num_samples + 1, len(X0))) into Traces,
      using `make_trace` and `add_detection_floor`, and fill `self.traces`
    - optionally `compiled_integrate(integrator, G, S, X0, time_array)`: a compiled time loop that
      consumes the random numbers exactly as `integrator.integrate` does (return None when the
      configuration has no kernel, after setting self.compiled_kernel_unsupported to the reason;
      NumPy is then used with backend='auto' and UnsupportedBackendError raised with 'numba')
    - optionally `prepare_numpy_loop()` and `release_numpy_loop()`: called just before and just after
      the NumPy time loop (e.g. to resolve the force dispatch once per run instead of once per drift
      evaluation)
    """

    # optional callable X -> X applied after every step (see Integrator.integrate); None: no projection
    project_state = None

    default_num_samples = 1*megasamples
    default_Δt = 0.1*μs

    def __init__(
        self,
        **kwargs,
    ):
        self.debug = kwargs.pop('debug', False)
        self.rk_loop_debug = kwargs.pop('rk_loop_debug', False)

        self.num_samples = int(kwargs.get('num_samples', self.default_num_samples))
        self.Δt = float(kwargs.get('Δt', self.default_Δt))
        self.RK_method = kwargs.get('RK_method', 'vector_4')
        self.backend = kwargs.get('backend', 'auto')
        if self.backend not in BACKENDS:
            raise ValueError(f"Unknown backend={self.backend!r}; use 'auto', 'numpy' or 'numba'")
        self.detection_floor = kwargs.get('detection_floor', 0)  # one-sided PSD, units²/Hz
        self.seed = kwargs.get('seed', None)
        self.rng = None if self.seed is None else np.random.default_rng(self.seed)
        self.zero_mean = kwargs.get('zero_mean', True)  # subtract the mean of the time traces from all values

        self.configure(**kwargs)

        self.simulate_dynamics()

        self.save = kwargs.get('save', False)
        self.save_dir = kwargs.get('save_dir', 'pickle')
        self.save_description = kwargs.get('save_description', 'default')
        self.save_fmt = kwargs.get('save_fmt', '.0e')

        if self.save:
            makedirs(self.save_dir, exist_ok=True)
            save_variable_to_pickle(self, self.save_dir + '/' + self.save_name)

    # ------------------------------------------------------------------------------------------
    # hooks for subclasses
    # ------------------------------------------------------------------------------------------
    def configure(self, **kwargs):
        """Reads the system's own keyword arguments; called before the integration"""

    def initial_state(self):
        """Initial state vector X0"""
        raise NotImplementedError

    def drift(self, t, X):
        """Deterministic part F(t, X) of dX = F(t, X) dt + G dW"""
        raise NotImplementedError

    def noise_matrix(self):
        """Constant noise matrix G of dX = F(t, X) dt + G dW, shape (len(X0), number of noises)"""
        raise NotImplementedError

    def noise_spectral_densities(self):
        """Two-sided spectral densities S_k of the independent noises dW_k (Var dW_k = S_k dt)"""
        raise NotImplementedError

    def record(self, X):
        """Turns the integrated states X into Traces and fills self.traces"""
        raise NotImplementedError

    def compiled_integrate(self, integrator, G, S, X0, time_array):
        """
        A compiled time loop for this configuration, or None if there is none (the NumPy loop is
        then used with backend='auto'; backend='numba' raises UnsupportedBackendError). Must consume
        the random numbers exactly as `integrator.integrate` does. A subclass returning None sets
        self.compiled_kernel_unsupported to the reason, which the error message quotes.
        """
        self.compiled_kernel_unsupported = f'{type(self).__name__} has no compiled time loop'

    # ------------------------------------------------------------------------------------------
    # random numbers
    # ------------------------------------------------------------------------------------------
    def standard_normals(self, shape):
        """
        Standard normal random numbers N(0, 1) of the given shape, drawn from the simulation's
        random number generator

        - `seed=None` (default): numpy's legacy global RandomState (np.random.normal), so runs
          seeded with np.random.seed(...) reproduce the numbers of earlier levisim versions
        - otherwise: a numpy Generator, np.random.default_rng(seed)

        Drawing an array from the legacy RandomState gives the same numbers, in the same (C) order,
        as drawing them one at a time, so pre-generating per chunk does not change the stream.
        [·]
        """
        if self.rng is None:
            return np.random.normal(size=shape)
        return self.rng.standard_normal(shape)

    def _noise_chunks(self, num_steps, num_stages, num_noises, chunk_size=2**15):
        """
        Yields (start_index, normals) where normals has shape (n, num_stages, num_noises), n ≤ chunk_size,
        covering num_steps time steps in order. Keeps memory bounded for long runs.
        """
        for start in range(0, num_steps, chunk_size):
            n = min(chunk_size, num_steps - start)
            yield start, self.standard_normals((n, num_stages, num_noises))

    # ------------------------------------------------------------------------------------------
    # integration
    # ------------------------------------------------------------------------------------------
    @property
    def integrator(self):
        """The Integrator selected by RK_method (raises ValueError for an unknown method)"""
        return get_integrator(self.RK_method)

    def simulate_dynamics(self):
        """
        Integrates dX = F(t, X) dt + G dW from initial_state() with the selected integrator,
        stores the states as self.X and records the traces (see `record`)

        The random numbers for each chunk of time steps are pre-generated (see standard_normals).
        """
        integrator = self.integrator
        X0 = self.initial_state()
        # the noise is additive (G is independent of t, X), so build G once
        G = self.noise_matrix()
        S = self.noise_spectral_densities()
        # the time array has num_samples + 1 entries; the penultimate step fills the final entry
        time_array = self.time_array[:-1]
        logger.info(
            'Simulating {} time steps of Δt = {:.4g} μs ({:.4g} ms) with RK_method={!r}, seed={!r}',
            self.num_samples, self.Δt / μs, self.num_samples * self.Δt / ms, integrator.name, self.seed,
        )
        start_time = time.perf_counter()

        X = None
        self.backend_used = 'numpy'
        if resolve_backend(self.backend, self.num_samples) == 'numba':
            X = self.compiled_integrate(integrator, G, S, X0, time_array)
            if X is None:
                reason = getattr(self, 'compiled_kernel_unsupported', '') or 'no compiled kernel'
                if self.backend == 'numba':
                    raise unsupported_backend_error(type(self).__name__, reason)
                logger.debug('No compiled kernel for this {} configuration ({}): using the NumPy loop',
                             type(self).__name__, reason)
            else:
                self.backend_used = 'numba'
        if X is None:
            # a subclass with a constrained state (RigidBodySimulation) projects after every step;
            # the keyword is passed only then, so integrators written before it keep working
            project_kwargs = {} if self.project_state is None else {'project': self.project_state}
            # (looked up with getattr: duck-typed systems that borrow simulate_dynamics may lack them)
            getattr(self, 'prepare_numpy_loop', _no_hook)()
            try:
                X = integrator.integrate(
                    self.drift, G, S, X0, time_array, self.Δt, self._noise_chunks,
                    debug=self.debug, trace_steps=self.rk_loop_debug, **project_kwargs,
                )
            finally:
                getattr(self, 'release_numpy_loop', _no_hook)()

        elapsed_seconds = time.perf_counter() - start_time
        logger.debug(
            'Integrated {} steps in {:.3f} s ({:.2f} μs per step, {} backend)',
            self.num_samples, elapsed_seconds, 1e6 * elapsed_seconds / max(self.num_samples, 1), self.backend_used,
        )

        self.X = X
        self.record(X)

        logger.info(
            'Simulated {} samples ({:.4g} ms) in {:.2f} s: traces {}',
            self.num_samples, self.time_length / ms, time.perf_counter() - start_time, list(self.traces),
        )

    # ------------------------------------------------------------------------------------------
    # recording
    # ------------------------------------------------------------------------------------------
    def make_trace(self, values, label, units):
        """A Trace of `values` sampled every Δt over the simulation, labelled `label` [units]"""
        return Trace(
            values,
            0, self.time_length, self.Δt,
            label, units, self,
            zero_mean=self.zero_mean,
        )

    def detection_floor_for(self, axis):
        """
        One-sided PSD of the white detection (imprecision) noise for a given trace label
        `detection_floor` may be a single number (all traces) or a dict {'x': ..., 'y': ..., 'z': ...}
        [m²/Hz for positions]
        """
        if isinstance(self.detection_floor, dict):
            return float(self.detection_floor.get(axis, 0))
        return float(self.detection_floor or 0)

    def add_detection_floor(self, values, axis):
        """
        Adds white measurement noise to a *recorded* trace (never to the integrated state)

        Convention: `detection_floor` G_floor is the one-sided PSD [m²/Hz] of the noise, i.e. the flat
        level it produces far from resonance in a (default, one-sided) SpectralDensityEstimate.
        A white sequence of variance σ² sampled every Δt has one-sided PSD G = 2σ²Δt, hence
        σ = √(G_floor / (2Δt)).
        [m]
        """
        G_floor = self.detection_floor_for(axis)
        if not G_floor:
            return values
        σ_floor = np.sqrt(G_floor / (2 * self.Δt))
        return values + σ_floor * self.standard_normals(values.shape)

    # ------------------------------------------------------------------------------------------
    # time axis
    # ------------------------------------------------------------------------------------------
    @property
    def time_length(self):
        """ Time length of the simulation (seconds) """
        return self.num_samples * self.Δt

    @property
    def min_Δf(self):
        """ Minimum frequency resolution the time trace is capable of producing (Hertz) """
        return 1 / self.time_length

    @property
    def sample_rate(self):
        """ Sample rate of the simulation - how many samples are taken per second? (samples/second) """
        return 1 / self.Δt

    @property
    def Nyquist_frequency(self):
        """
        The Nyquist frequency, a.k.a. the bandwidth of the simulation
        This will be the largest frequency in the spectral density estimate
        NOTE: https://en.wikipedia.org/wiki/Nyquist_frequency
        """
        return self.sample_rate / 2

    @property
    def time_array(self):
        """ Array of all the time steps in the simulation """
        return generate_time_array(0, self.time_length, self.num_samples)

    # ------------------------------------------------------------------------------------------
    # summary and saving
    # ------------------------------------------------------------------------------------------
    @property
    def info_str(self):
        """Summary of the simulation's settings (subclasses append their traces' statistics)"""
        return (
            f'{type(self).__name__}(id={id(self)}'
            f'\n    RK_method = {self.RK_method}'
            f'\n    num_samples = {self.num_samples:.5e}'
            f'\n    Δt = {self.Δt / μs:.5f} μs'
            f'\n'
            f'\n    time_length = {self.time_length / ms:.5f} ms'
            f'\n    sample_rate = {self.sample_rate / 1e6:.5f} MHz'
            f'\n    Nyquist_freq = {self.Nyquist_frequency / 1e6:.5f} MHz'
            f'\n    zero_mean = {self.zero_mean}'
            f'{self._info_body()}'
            f'\n)'
        )

    def _info_body(self):
        """Extra lines of info_str (subclass hook)"""
        return ''

    @property
    def info(self):
        """Prints a summary of the simulation's settings and of its traces"""
        print(self.info_str)

    @property
    def save_name(self):
        """ Returns standardised naming format for the pickle binaries """
        fmt = self.save_fmt
        return (
            f'{self.save_description.replace(" ", "_")}'
            f'_t{format(self.Δt, fmt)}'
            f'_N{format(self.num_samples, fmt)}'
        )

r"""
CavitySimulation: time traces of the light leaving a cavity, and of the particles in it, from the
same linear model that `MultiTrap` uses for spectra

Model 'linear' (the exact linear model)
---------------------------------------
The state is the real quadrature vector u of `QuadratureSystem.from_multitrap` (cavity A, P,
then X, P of every dimension; the vacuum covariance is the identity), ::

    du = A u dt + B dW,      ½〈{dW, dWᵀ}〉 = N dt,

with A levisim's drift matrix in quadratures and B·N·Bᵀ levisim's diffusion matrix. Every record
(`levisim.simulation.records`) adds auxiliary integrals Y driven by the SAME increments dW,
dY = (C·u) dt + D·dW + E·dV + R·Y dt (dV private vacuum noises of the detectors), reset to zero at
the start of every sample interval. The augmented linear system z = (u, Y) is discretised
EXACTLY: over one sample interval Δt ::

    z_{k+1} = Φ·(u_k, 0) + ξ_k,     Φ = exp(Ã·Δt),     ξ_k ~ N(0, Q),
    Q = ∫₀^Δt exp(Ã·s)·B̃ÑB̃ᵀ·exp(Ãᵀ·s) ds,

with Φ and Q from one matrix exponential of Van Loan's block matrix (C. F. Van Loan, IEEE Trans.
Autom. Control 23, 395 (1978), doi:10.1109/tac.1978.1101743). There is no time-step error at
any Δt: the sampled state has exactly the statistics of the continuous model, and each recorded
sample is exactly the average of the continuous output over its interval (a sinc²(fΔt) response
and no aliasing of white noise). Q is factorised by a symmetric eigen-decomposition (it is only
positive SEMI-definite when the noise has fewer columns than the augmented state).

The simulation starts in the steady state (a Cholesky factor of `steady_state_covariance`) by
default. An unstable system has no steady state: it must start from 'zero' or a given state, and
its growth is then simulated faithfully (until the linearisation no longer holds physically).

Ensembles: `num_trajectories=K` advances K independent trajectories as one array, which is the
cheap way to smooth averaged spectra.

Model 'nonlinear' is the split-step model of the design plan; it is not part of this module yet.
"""

import time

import numpy as np
from scipy.linalg import expm

from levisim.core.logger import logger
from levisim.core.units import ms, μs
from levisim.multiphysics.output_modes import QuadratureSystem
from levisim.simulation.backends import (
    choose_threading,
    resolve_backend,
)
from levisim.simulation.base import StochasticSimulation
from levisim.simulation.records import (
    Heterodyne,
    Homodyne,
    OpticalRecord,
    Position,
    Record,
)

__all__ = [
    'CavitySimulation',
    'van_loan_discretisation',
]

MODELS = ('linear', 'nonlinear')

# backend='auto' compiles the time loop only up to this state dimension n. Measured (8-core arm64,
# one trajectory, one homodyne record, per step): n = 4: NumPy 3.6 μs, numba 0.35 μs; n = 34: 10.0
# vs 4.5 μs; n = 66: 16.4 vs 8.8 μs; n = 130: 31.9 vs 31.6 μs. Beyond that the per-step cost is the
# BLAS products (the noise L·N(0, 1) and Φ·u), and NumPy's are faster than the compiled loop.
AUTO_NUMBA_MAX_STATE = 128

# keyword arguments read by StochasticSimulation and by CavitySimulation
_BASE_KWARGS = {
    'debug', 'rk_loop_debug', 'num_samples', 'Δt', 'RK_method', 'backend', 'detection_floor', 'seed',
    'zero_mean', 'save', 'save_dir', 'save_description', 'save_fmt',
}
_CAVITY_KWARGS = {
    'model', 'detection', 'num_trajectories', 'initial_state', 'store_state', 'chunk_size',
}


def van_loan_discretisation(A, Q_c, Δt, max_norm_step=0.5):
    """
    Exact discretisation of dz = A z dt + dη with white noise of covariance Q_c dt, over Δt

    Returns (Φ, Q) with Φ = exp(A·Δt) and Q = ∫₀^Δt exp(A·s)·Q_c·exp(Aᵀ·s) ds, from the block
    exponential exp([[-A, Q_c], [0, Aᵀ]]·h) = [[·, F₁₂], [0, F₂₂]]: Φ(h) = F₂₂ᵀ and Q(h) = Φ(h)·F₁₂
    (Van Loan, IEEE Trans. Autom. Control 23, 395 (1978)). The block contains exp(-A·h), which
    grows like exp(κh/2) and cancels in Q; to keep that cancellation harmless, h = Δt/2^s with
    ‖A‖₁·h ≤ max_norm_step, and the step is doubled s times exactly:
    Φ(2h) = Φ(h)², Q(2h) = Φ(h)·Q(h)·Φ(h)ᵀ + Q(h). Q is symmetrised.
    A [1/s], Q_c [state²/s], Δt [s]
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    norm = np.abs(A).sum(axis=0).max() * Δt if n else 0.0
    squarings = max(0, int(np.ceil(np.log2(norm / max_norm_step)))) if norm > max_norm_step else 0
    h = Δt / 2**squarings
    M = np.zeros((2 * n, 2 * n))
    M[:n, :n] = -A
    M[:n, n:] = Q_c
    M[n:, n:] = A.T
    F = expm(M * h)
    Φ = F[n:, n:].T
    Q = Φ @ F[:n, n:]
    Q = (Q + Q.T) / 2
    for _ in range(squarings):
        Q = Φ @ Q @ Φ.T + Q
        Q = (Q + Q.T) / 2
        Φ = Φ @ Φ
    return Φ, Q


def _psd_square_root(Q):
    """
    L with L·Lᵀ = Q for a symmetric positive semi-definite Q (negative round-off clipped)

    The entries of Q span many orders of magnitude (thermal mechanical quadratures against the
    sample integrals of shot noise, ~Δt), so Q is first scaled to unit diagonal,
    Q = S·Q'·S with S = diag(√Q_ii), and the eigen-decomposition is taken of Q'; otherwise its
    round-off, relative to the largest entry, would swamp the smallest ones.
    """
    d = np.sqrt(np.clip(np.diag(Q), 0, None))
    scale = np.where(d > 0, d, 1.0)
    Q_scaled = Q / np.outer(scale, scale)
    w, V = np.linalg.eigh((Q_scaled + Q_scaled.T) / 2)
    largest = max(np.abs(w).max(), 1e-300)
    if w.min() < -1e-9 * largest:
        logger.warning('one-step noise covariance has a negative eigenvalue {:.3e} (largest {:.3e}, unit-diagonal scaling); clipped',
                       w.min(), largest)
    return scale[:, None] * (V * np.sqrt(np.clip(w, 0, None)))


class _TrajectoryView:
    """The traces of one trajectory of an ensemble, with what SpectralDensityEstimate reads"""

    def __init__(self, sim, index):
        self.sim = sim
        self.index = index
        self.Δt = sim.Δt
        self.traces = {label: sim._make_trace(label, index) for label in sim.records}

    @property
    def num_samples(self):
        return self.sim.num_samples

    @property
    def time_array(self):
        return self.sim.time_array

    @property
    def time_length(self):
        return self.sim.time_length


class CavitySimulation(StochasticSimulation):
    """
    Stochastic simulation of particles coherently scattering into a cavity, with output records

    ``CavitySimulation(system, **kwargs)``, where `system` is a `MultiTrap` (held by composition;
    the usual entry point is ``mtrap.simulate(**kwargs)``) or a `QuadratureSystem` (e.g. two cavity
    modes from `levisim.multiphysics.output_modes.stack_cavity_modes`).

    Keyword arguments (defaults in brackets)::

        model ['linear']: 'linear', the exact linear model (see the module docstring); 'nonlinear'
            raises NotImplementedError (a later round-5 track, round5/wip/cavity-nonlinear)
        detection [None]: a list of records (levisim.simulation.records: Homodyne, Heterodyne,
            PhotonNumber, Position). None records Homodyne() of every cavity mode (labelled
            'homodyne', or 'homodyne_<cavity>' with several modes) and Position of every mechanical mode
        num_samples [1e6], Δt [0.1 μs]: number and length of the sample intervals [·, s]
        num_trajectories [1]: K independent trajectories advanced together
        initial_state ['steady']: 'steady' (drawn from steady_state_covariance; ValueError for an
            unstable system), 'zero', or an array u0 of shape (n,) or (K, n) [quadratures]
        store_state [False]: keep the state at every sample time, self.states, shape
            (K, num_samples + 1, n); the final state is always kept as self.u_final
        seed [None], backend ['auto'], detection_floor [0], zero_mean [True], save...: see
            StochasticSimulation. With backend 'numba' (or 'auto' and numba installed) the time loop
            is compiled; it consumes the same random numbers as the NumPy loop
        chunk_size [2¹⁴]: time steps per block of pre-generated random numbers

    Results: self.traces (label -> Trace of trajectory 0, so the analysis path of `Experiment`
    data applies, e.g. SpectralDensityEstimate(sim, ('homodyne', 'homodyne'))), self.records
    (label -> array (K, num_samples) of every trajectory), `trajectory(k)` (the traces of
    trajectory k), `averaged_psd` and the theory spectra `record_spectrum`.

    Sample k is the average of its record over [t_k, t_k + Δt], t_k = k·Δt; the traces' time
    array labels each sample by t_k.
    """

    def __init__(self, system, **kwargs):
        unknown = set(kwargs) - _BASE_KWARGS - _CAVITY_KWARGS
        if unknown:
            raise TypeError(f'CavitySimulation got unknown keyword arguments {sorted(unknown)}')
        if 'RK_method' in kwargs:
            raise ValueError("CavitySimulation's linear model is discretised exactly; it takes no RK_method")
        self.source_system = system
        super().__init__(**kwargs)

    # ------------------------------------------------------------------------------------------
    # configuration
    # ------------------------------------------------------------------------------------------
    def configure(self, **kwargs):
        """Reads the model, the system, the records, the ensemble size and the initial state"""
        self.model = kwargs.get('model', 'linear')
        if self.model not in MODELS:
            raise ValueError(f'Unknown model={self.model!r}; use one of {MODELS}')
        if self.model == 'nonlinear':
            raise NotImplementedError(
                "model='nonlinear' (the split-step nonlinear cavity simulation) is not implemented yet: it is a"
                " later round-5 track (round5/wip/cavity-nonlinear). Use model='linear'."
            )
        self.RK_method = 'exact (Van Loan)'

        system = self.source_system
        if isinstance(system, QuadratureSystem):
            self.mtrap = None
            self.system = system
        elif hasattr(system, 'drift_matrix') and hasattr(system, 'quantum_factor_per_noise_operator'):
            self.mtrap = system
            self.system = QuadratureSystem.from_multitrap(system)
        else:
            raise TypeError(f'CavitySimulation needs a MultiTrap or a QuadratureSystem, not {type(system).__name__}')

        self.num_trajectories = int(kwargs.get('num_trajectories', 1))
        if self.num_trajectories < 1:
            raise ValueError(f'num_trajectories = {self.num_trajectories} should be ≥ 1')
        self.initial_state_spec = kwargs.get('initial_state', 'steady')
        self.store_state = bool(kwargs.get('store_state', False))
        self.chunk_size = int(kwargs.get('chunk_size', 2**14))

        detection = kwargs.get('detection')
        if detection is None:
            detection = self.default_detection()
        elif isinstance(detection, Record):
            detection = [detection]
        self.detection = list(detection)
        labels = [record.label for record in self.detection]
        duplicates = sorted({label for label in labels if labels.count(label) > 1})
        if duplicates:
            raise ValueError(f'records with the same label {duplicates}: give each record its own label=...')

    def default_detection(self):
        """Homodyne() of every cavity mode and Position of every mechanical mode"""
        cavities = list(self.system.cavities)
        records = []
        for cavity in cavities:
            label = 'homodyne' if len(cavities) == 1 else f'homodyne_{cavity}'
            θ = None if self._is_multitrap_cavity(cavity) else 0.0
            records.append(Homodyne(θ=θ, cavity=cavity, label=label))
        units = 'm' if self.mtrap is not None and self.mtrap.toy_params is None else 'zpf'
        for label in self.system.labels:
            if label not in self.system.cavities:
                records.append(Position(label, units=self._position_units(label, units)))
        return records

    def _position_units(self, label, units):
        if units == 'm':
            try:
                self.zero_point_fluctuation(label)
            except (AttributeError, KeyError, ValueError):
                return 'zpf'
        return units

    # ------------------------------------------------------------------------------------------
    # what the records read from the system
    # ------------------------------------------------------------------------------------------
    def _is_multitrap_cavity(self, cavity):
        return self.mtrap is not None and cavity == self.system.labels[0]

    def resolve_cavity(self, cavity):
        """The label of a cavity mode (None: the only one)"""
        if cavity is None:
            if len(self.system.cavities) != 1:
                raise ValueError(f'the system has several cavity modes {list(self.system.cavities)}: pass cavity=...')
            return next(iter(self.system.cavities))
        if cavity not in self.system.cavities:
            raise ValueError(f'unknown cavity mode {cavity!r}; the system has {list(self.system.cavities)}')
        return cavity

    def port(self, cavity, port=None):
        """
        (κ_j, (column of Q_in, column of P_in), j) of a port of a cavity mode: port=None is the
        detected port (`QuadratureSystem.cavities`)
        """
        ports = getattr(self.system, 'ports', {}).get(cavity)
        if port is None:
            κ, columns = self.system.cavities[cavity]
            index = 0
            if ports:
                index = next(j for j, (κ_j, cols) in enumerate(ports) if tuple(cols) == tuple(columns))
            return κ, columns, index
        if not ports:
            if port != 0:
                raise ValueError(f'cavity {cavity!r} has a single port (index 0), not {port!r}')
            κ, columns = self.system.cavities[cavity]
            return κ, columns, 0
        if not 0 <= port < len(ports):
            raise ValueError(f'cavity {cavity!r} has ports 0..{len(ports) - 1}, not {port!r}')
        κ_j, columns = ports[port]
        return κ_j, columns, port

    def _cavity_object(self, purpose):
        if self.mtrap is None:
            raise ValueError(f'{purpose} is read from a MultiTrap; for a QuadratureSystem pass it to the record explicitly')
        return self.mtrap.cavity

    def default_homodyne_angle(self):
        """cavity.θhom of the MultiTrap [rad]"""
        return float(self._cavity_object('the homodyne angle θ').θhom)

    def default_heterodyne_frequency(self):
        """cavity.Ωhet of the MultiTrap [2π·Hz]"""
        return float(self._cavity_object('the heterodyne frequency Ωhet').Ωhet)

    def static_field(self, cavity):
        """Static intracavity field α = (A0 + i·P0)/2 of the MultiTrap's cavity [√photons]"""
        if not self._is_multitrap_cavity(cavity):
            raise ValueError(f'the static field of cavity mode {cavity!r} is not known: pass α=... to PhotonNumber')
        c = self.mtrap.cavity
        return complex(c.A0 + 1j * c.P0) / 2

    def optical_angular_frequency(self):
        """Angular frequency of the cavity light, cavity.ω [rad/s]"""
        return float(self._cavity_object('the optical frequency').ω)

    def zero_point_fluctuation(self, mode):
        """x_zpf of a mechanical mode, from the MultiTrap's Dimension [m]"""
        if self.mtrap is None:
            raise ValueError(f"x_zpf of {mode!r} is not known for a QuadratureSystem: use Position({mode!r}, units='zpf')")
        zpf = float(self.mtrap.dimensions[mode].zpf)
        if not np.isfinite(zpf) or zpf <= 0:
            raise ValueError(f"x_zpf of {mode!r} is {zpf}: use Position({mode!r}, units='zpf')")
        return zpf

    # ------------------------------------------------------------------------------------------
    # the augmented system and its exact discretisation
    # ------------------------------------------------------------------------------------------
    def build_augmented_system(self):
        """
        Ã, B̃, Ñ of the augmented state z = (u, Y_1, Y_2, ...) and the slices of each record's Y

        Ñ = blockdiag(N, identity for the records' private vacuum noises).
        """
        system = self.system
        n, m = system.B.shape
        rows = [record.rows(self) for record in self.detection]
        q = sum(r.C.shape[0] for r in rows)
        e = sum(r.E.shape[1] for r in rows)
        A = np.zeros((n + q, n + q))
        B = np.zeros((n + q, m + e))
        N = np.zeros((m + e, m + e))
        A[:n, :n] = system.A
        B[:n, :m] = system.B
        N[:m, :m] = system.N
        N[m:, m:] = np.eye(e)
        slices = {}
        i, j = n, m
        for record, r in zip(self.detection, rows):
            qr, er = r.C.shape[0], r.E.shape[1]
            A[i:i + qr, :n] = r.C
            A[i:i + qr, i:i + qr] = r.R
            B[i:i + qr, :m] = r.D
            B[i:i + qr, j:j + er] = r.E
            slices[record.label] = slice(i - n, i - n + qr)
            i += qr
            j += er
        self.augmented = {'A': A, 'B': B, 'N': N, 'n': n, 'q': q, 'slices': slices}
        return self.augmented

    def discretise(self):
        """Φ (restricted to the columns of u, since the records restart from zero), Q and L·Lᵀ = Q"""
        aug = self.augmented
        Φ, Q = van_loan_discretisation(aug['A'], aug['B'] @ aug['N'] @ aug['B'].T, self.Δt)
        self.Φ = Φ[:, :aug['n']]
        self.Q = Q
        self.L = _psd_square_root(Q)
        return self.Φ, self.Q

    # ------------------------------------------------------------------------------------------
    # initial state
    # ------------------------------------------------------------------------------------------
    def initial_state(self):
        """
        Initial quadratures, shape (K, n): drawn from the steady state ('steady'), zero ('zero'),
        or as given [·]
        """
        n = self.system.A.shape[0]
        K = self.num_trajectories
        spec = self.initial_state_spec
        if isinstance(spec, str):
            if spec == 'zero':
                return np.zeros((K, n))
            if spec != 'steady':
                raise ValueError(f"initial_state={spec!r}: use 'steady', 'zero' or an array")
            growth = self.system.largest_growth_rate()
            if growth >= 0:
                raise ValueError(
                    f'the system is linearly unstable (largest growth rate {growth:.4g} s⁻¹), so it has no steady'
                    " state to start in: pass initial_state='zero' (or an array) to simulate its growth"
                )
            V = self.system.steady_state_covariance()
            try:
                L = np.linalg.cholesky(V)
            except np.linalg.LinAlgError:
                L = _psd_square_root(V)
            return self.standard_normals((K, n)) @ L.T
        u0 = np.asarray(spec, dtype=float)
        if u0.shape == (n,):
            return np.broadcast_to(u0, (K, n)).copy()
        if u0.shape == (K, n):
            return u0.copy()
        raise ValueError(f'initial_state has shape {u0.shape}; expected ({n},) or ({K}, {n})')

    # ------------------------------------------------------------------------------------------
    # integration
    # ------------------------------------------------------------------------------------------
    def simulate_dynamics(self):
        """
        Advances the augmented system exactly over num_samples intervals and records the traces
        """
        start_time = time.perf_counter()
        aug = self.build_augmented_system()
        self.discretise()
        n, q = aug['n'], aug['q']
        K, N = self.num_trajectories, self.num_samples
        growth = self.system.largest_growth_rate()
        if growth > 0:
            logger.warning(
                'Simulating a linearly unstable system (largest growth rate {:.4g} s⁻¹): amplitudes grow by e^{:.3g}'
                ' over the run', growth, growth * self.time_length,
            )
        logger.info(
            'Simulating {} samples of Δt = {:.4g} μs ({:.4g} ms), {} trajectories, exact linear model with {} records, seed={!r}',
            N, self.Δt / μs, self.time_length / ms, K, len(self.detection), self.seed,
        )
        u = self.initial_state()

        Y = np.empty((N, K, q))
        states = np.empty((N + 1, K, n)) if self.store_state else None
        if states is not None:
            states[0] = u
        ΦT = np.ascontiguousarray(self.Φ.T)       # (n, n + q), for the NumPy loop
        Φc = np.ascontiguousarray(self.Φ)         # (n + q, n), for the compiled loop
        LT = np.ascontiguousarray(self.L.T)       # (n + q, n + q)

        self.backend_used = 'numpy'
        kernel = None
        use_numba = resolve_backend(self.backend, N) == 'numba'
        if use_numba and self.backend == 'auto' and n > AUTO_NUMBA_MAX_STATE:
            use_numba = False
        if use_numba:
            from levisim.simulation.backends.numba_linear import exact_linear_chunk
            kernel = exact_linear_chunk
            self.backend_used = 'numba'
            self.threading = choose_threading(num_setups=K)

        for start in range(0, N, self.chunk_size):
            nn = min(self.chunk_size, N - start)
            # one (nn·K, n + q) product is about 3× faster than the batched (nn, K, n + q) @ LT
            ξ = (self.standard_normals((nn, K, n + q)).reshape(nn * K, n + q) @ LT).reshape(nn, K, n + q)
            if kernel is not None:
                u = kernel(Φc, ξ, u, Y[start:start + nn], states[start + 1:start + nn + 1] if states is not None else None,
                           self.threading.axis == 'setups')
            else:
                for k in range(nn):
                    z = u @ ΦT
                    z += ξ[k]
                    u = z[:, :n]
                    Y[start + k] = z[:, n:]
                    if states is not None:
                        states[start + k + 1] = u
        self.u_final = u
        self.states = None if states is None else np.transpose(states, (1, 0, 2))
        self.X = self.states
        elapsed = time.perf_counter() - start_time
        logger.debug('Integrated {} steps × {} trajectories in {:.3f} s ({:.3f} μs per step, {} backend)',
                     N, K, elapsed, 1e6 * elapsed / max(N, 1), self.backend_used)

        self.record(np.transpose(Y, (1, 0, 2)) / self.Δt)
        logger.info('Simulated {} samples ({:.4g} ms) in {:.2f} s: traces {}',
                    N, self.time_length / ms, time.perf_counter() - start_time, list(self.traces))

    # ------------------------------------------------------------------------------------------
    # recording
    # ------------------------------------------------------------------------------------------
    def record(self, values):
        """
        Turns the sample averages of the auxiliary integrals, shape (K, num_samples, q), into the
        records (self.records, arrays (K, num_samples)) and the Traces of trajectory 0
        """
        slices = self.augmented['slices']
        self.records = {}
        for record in self.detection:
            self.records[record.label] = record.finish(values[..., slices[record.label]], self)
        self.traces = {label: self._make_trace(label, 0) for label in self.records}
        for label, trace in self.traces.items():
            if label.isidentifier() and not hasattr(type(self), label) and label not in vars(self):
                setattr(self, label, trace)

    def _make_trace(self, label, index):
        record = next(r for r in self.detection if r.label == label)
        return self.make_trace(self.records[label][index], label, record.units)

    def trajectory(self, index):
        """The traces of trajectory `index` (an object SpectralDensityEstimate accepts as a source)"""
        if not 0 <= index < self.num_trajectories:
            raise IndexError(f'trajectory {index} of {self.num_trajectories}')
        return _TrajectoryView(self, index)

    def averaged_psd(self, label, num_segments=16, window='hann', one_sided=False):
        """
        PSD of a record averaged over every trajectory and every segment (Welch-like, without overlap)

        Each trajectory is cut into num_segments segments, each windowed (normalised to unit mean
        square, as SpectralDensityEstimate does) and Fourier transformed; the periodograms are
        averaged. Normalisation: two-sided by default, ∫S df = 〈h²〉 (levisim's convention; the
        vacuum of an optical record is 1); one_sided=True doubles f > 0 as SpectralDensityEstimate.
        Returns (f [Hz], S [units²/Hz]) for f ≥ 0, sorted.
        """
        from levisim.analysis.spectral_density_estimate import SpectralDensityEstimate
        x = self.records[label]
        K, N = x.shape
        M = N // num_segments
        w = SpectralDensityEstimate.make_window(window, M)
        segments = x[:, :M * num_segments].reshape(K, num_segments, M)
        segments = segments - segments.mean(axis=-1, keepdims=True) if self.zero_mean else segments
        F = np.fft.rfft(segments * w, axis=-1)
        S = (np.abs(F)**2).mean(axis=(0, 1)) * self.Δt / M
        f = np.fft.rfftfreq(M, self.Δt)
        if one_sided:
            S[1:] *= 2
            if M % 2 == 0:
                S[-1] /= 2
        return f, S

    # ------------------------------------------------------------------------------------------
    # theory
    # ------------------------------------------------------------------------------------------
    def record_spectrum(self, label, f, kind='discrete'):
        """
        Theory spectrum of a record (two-sided, symmetric, ∫S df = variance) at frequencies f [Hz]

        - kind='discrete': the EXACT spectrum of the sampled record, from the discretised augmented
          system (it includes the sinc² response of the sample average and the aliasing of
          everything beyond the Nyquist frequency)
        - kind='continuous': the symmetrised spectrum of the continuous output from the
          QuadratureSystem's transfer functions, times sinc²(fΔt) (no aliasing); this is the form
          to compare with ½[S(ω) + S(-ω)] of MultiTrap

        Includes the detection efficiency, electronic noise, detection floor and conversion to
        volts; excludes LO phase noise. [units²/Hz]
        """
        record = next((r for r in self.detection if r.label == label), None)
        if record is None:
            raise KeyError(f'no record {label!r}; records are {list(self.records)}')
        f = np.atleast_1d(np.asarray(f, dtype=float))
        ω = 2 * np.pi * f
        cross = self._aux_spectral_matrix if kind == 'discrete' else self._aux_spectral_matrix_continuous
        if kind not in ('discrete', 'continuous'):
            raise ValueError(f"kind={kind!r}: use 'discrete' or 'continuous'")
        sl = self.augmented['slices'][label]

        if isinstance(record, Heterodyne):
            v = np.array([1.0, 1j])
            Ω = record.Ωhet_used
            s_lo = np.einsum('i,wij,j->w', v, cross(ω - Ω, sl), v.conj()).real
            s_hi = np.einsum('i,wij,j->w', v, cross(-ω - Ω, sl), v.conj()).real
            S = (s_lo + s_hi) / 4
        elif isinstance(record, OpticalRecord):
            θ = record.θ_used
            v = np.array([np.cos(θ), np.sin(θ)])
            S = np.einsum('i,wij,j->w', v, cross(ω, sl), v).real
        else:
            S = cross(ω, sl)[:, 0, 0].real
        if kind == 'continuous':
            S = S * np.sinc(f * self.Δt)**2
        if isinstance(record, OpticalRecord):
            S = S * record.scale(self)**2 + record.electronic_noise / 2
        else:
            S = S * record.factor**2
            floor = record.detection_floor if record.detection_floor is not None else self.detection_floor_for(record.mode)
            S = S + floor / 2
        return S

    def _aux_spectral_matrix(self, ω, sl):
        """
        Exact spectral matrix of the sampled averages of the auxiliary integrals `sl` [·²/Hz]

        With levisim's Fourier convention, z = e^{-iωΔt}: u(ω) = (z·I - Φ_u)⁻¹ ξ_u(ω),
        y(ω) = Φ_y·u(ω) + ξ_y(ω), and S = H·Q·H† / Δt with y the averages (Y/Δt).
        """
        n = self.augmented['n']
        Φu = self.Φ[:n]
        Φy = self.Φ[n:][sl]
        idx_y = n + np.arange(self.Φ.shape[0] - n)[sl]
        z = np.exp(-1j * ω * self.Δt)
        Minv = np.linalg.inv(z[:, None, None] * np.eye(n)[None] - Φu[None])
        H = np.zeros((len(ω), Φy.shape[0], self.Φ.shape[0]), dtype=complex)
        H[:, :, :n] = Φy[None] @ Minv
        H[:, np.arange(len(idx_y)), idx_y] += 1
        return H @ self.Q @ np.conj(np.transpose(H, (0, 2, 1))) / self.Δt

    def _aux_spectral_matrix_continuous(self, ω, sl):
        """Symmetrised spectral matrix of the continuous outputs C·u + D·w + E·v of records `sl`"""
        aug = self.augmented
        n = aug['n']
        C = aug['A'][n:, :n][sl]
        DE = aug['B'][n:][sl]
        GB = np.linalg.solve(-1j * ω[:, None, None] * np.eye(n)[None] - aug['A'][:n, :n][None],
                             np.broadcast_to(aug['B'][:n].astype(complex), (len(ω), n, aug['B'].shape[1])))
        r = C[None] @ GB + DE[None]
        return r @ aug['N'] @ np.conj(np.transpose(r, (0, 2, 1)))

    # ------------------------------------------------------------------------------------------
    # export
    # ------------------------------------------------------------------------------------------
    def save_matfile(self, path, channel_dict=None, trajectory=0):
        """
        Writes records of one trajectory to a .mat file in the layout `levisim.data.Experiment`
        loads (variables 'Length', 'Tinterval' and one column per channel)

        channel_dict: {label: channel name}, default the records in order as channels 'A', 'B', ...
        Returns the channel_dict to pass as Experiment(path, channel_dict={label: channel}).
        """
        from scipy.io import savemat
        if channel_dict is None:
            channel_dict = {label: chr(ord('A') + i) for i, label in enumerate(self.records)}
        data = {'Length': np.array([[self.num_samples]]), 'Tinterval': np.array([[self.Δt]])}
        for label, channel in channel_dict.items():
            data[channel] = np.asarray(self.records[label][trajectory])[:, None]
        savemat(path, data)
        return {label: channel for label, channel in channel_dict.items()}

    # ------------------------------------------------------------------------------------------
    # unused hooks of StochasticSimulation (the linear model has its own exact stepper)
    # ------------------------------------------------------------------------------------------
    def drift(self, t, X):
        """A·u (the deterministic part, for reference; the stepper uses exp(AΔt))"""
        return self.system.A @ X

    def noise_matrix(self):
        """B of du = A u dt + B dW"""
        return self.system.B

    def noise_spectral_densities(self):
        """Diagonal of the symmetrised noise correlation N (1 for vacuum)"""
        return np.diag(self.system.N)

    # ------------------------------------------------------------------------------------------
    # summary
    # ------------------------------------------------------------------------------------------
    def _info_body(self):
        lines = [
            '',
            f'    model = {self.model}',
            f'    system = {self.mtrap if self.mtrap is not None else "QuadratureSystem"} with modes {self.system.labels}',
            f'    num_trajectories = {self.num_trajectories}',
            f'    largest growth rate = {self.system.largest_growth_rate():.5g} s⁻¹',
            f'    backend = {self.backend_used}',
            '',
        ]
        for label, values in self.records.items():
            record = next(r for r in self.detection if r.label == label)
            lines.append(f'    {record!r}: Var = {np.var(values):.5e} {record.units}²')
        return '\n'.join(lines)

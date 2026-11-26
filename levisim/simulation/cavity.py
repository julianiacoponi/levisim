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

Model 'nonlinear' (split-step)
------------------------------
The state is the cavity amplitude and the 3D position and momentum of every particle, with the
full (un-linearised) tweezer and coherent-scattering potential and the cavity drive derived from
the same energy (`levisim.simulation.nonlinear_cavity`). It is stored as the deviation u from the
TRUE equilibrium of the nonlinear drift (found by Newton's method), in the quadratures of the
linear model, so that the records, ports and detection are those of the linear model. The drift
is split as du = J·u dt + B dW + R(u) dt, with J the Jacobian at the equilibrium and R the
nonlinear remainder. Each sample interval Δt is cut into `substeps` steps h = Δt/M, and each step
is a Strang splitting ::

    L(h/2) ∘ N(h) ∘ L(h/2)

where L is the EXACT linear flow of the augmented system (state and record integrals), with its
Van Loan noise, and N(h) integrates du/dt = R(u) with one Runge-Kutta step ('rk4' or 'rk2'); the
record integrals are linear in u and are advanced only by L. Consecutive half steps inside a sample
are merged into full ones. With R = 0 (every term linearised) the scheme is the exact linear model.
`integrator='heun'` integrates the same equations with the plain Heun scheme instead (a
cross-check; it has time-step error in the fast cavity too).
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
    unsupported_backend_error,
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
# keyword arguments of model='nonlinear' only
_NONLINEAR_KWARGS = {
    'damping', 'terms', 'linearised_terms', 'interactions', 'trap_positions', 'feedback',
    'integrator', 'substeps', 'nonlinear_stage',
}
INTEGRATORS = ('split', 'heun')
NONLINEAR_STAGES = ('rk4', 'rk2')


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

        model ['linear']: 'linear', the exact linear model, or 'nonlinear', the split-step model of
            the full tweezer and coherent-scattering potential (see the module docstring; needs a
            MultiTrap with physical parameters)
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
            is compiled; it consumes the same random numbers as the NumPy loop. A nonlinear model
            without a compiled kernel (integrator='heun', pair interactions, force elements outside
            the compiled set) falls back to NumPy with 'auto' and raises UnsupportedBackendError
            with 'numba'
        chunk_size [2¹⁴]: time steps per block of pre-generated random numbers

    Keyword arguments of model='nonlinear' only (see `NonlinearCavityModel` for the first six)::

        damping ['velocity']: 'velocity' (-Γ·p) or 'symmetric' (MultiTrap's rotating-wave form)
        terms, linearised_terms: which drift terms to include, and which to linearise about the
            equilibrium ('elements', 'coherent_scattering', 'cavity', 'interactions')
        interactions [()], trap_positions [None]: pair forces (objects with
            forces(positions, particles, t)) and the lab-frame trap centres [m]
        feedback [False]: include the damping of the traps' Feedback elements
        integrator ['split']: 'split' (exact linear flow + Runge-Kutta remainder) or 'heun'
        substeps [1]: integration steps per sample interval
        nonlinear_stage ['rk4']: Runge-Kutta step of the remainder, 'rk4' or 'rk2'

    With model='nonlinear', the state (initial_state arrays, self.states, self.u_final) is the
    deviation from the equilibrium `self.nonlinear_model.u_eq`, in the quadratures of
    `self.system` (the model linearised at that equilibrium, labels 'cavity', 'x1', 'y1', 'z1',
    ...); 'zero' or 'equilibrium' starts at the equilibrium, 'steady' samples the linearised
    steady state about it. Position records are displacements from the equilibrium position.

    Results: self.traces (label -> Trace of trajectory 0, so the analysis path of `Experiment`
    data applies, e.g. SpectralDensityEstimate(sim, ('homodyne', 'homodyne'))), self.records
    (label -> array (K, num_samples) of every trajectory), `trajectory(k)` (the traces of
    trajectory k), `averaged_psd` and the theory spectra `record_spectrum`.

    Sample k is the average of its record over [t_k, t_k + Δt], t_k = k·Δt; the traces' time
    array labels each sample by t_k.
    """

    def __init__(self, system, **kwargs):
        unknown = set(kwargs) - _BASE_KWARGS - _CAVITY_KWARGS - _NONLINEAR_KWARGS
        if unknown:
            raise TypeError(f'CavitySimulation got unknown keyword arguments {sorted(unknown)}')
        if 'RK_method' in kwargs:
            raise ValueError(
                "CavitySimulation takes no RK_method: the linear model is discretised exactly, and the nonlinear"
                " model is chosen with integrator='split' or 'heun'"
            )
        nonlinear_only = sorted(set(kwargs) & _NONLINEAR_KWARGS)
        if nonlinear_only and kwargs.get('model', 'linear') != 'nonlinear':
            raise TypeError(f"keyword arguments {nonlinear_only} apply only to model='nonlinear'")
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
        self.RK_method = 'exact (Van Loan)'
        self.nonlinear_model = None

        system = self.source_system
        if self.model == 'nonlinear':
            self._configure_nonlinear(system, kwargs)
        elif isinstance(system, QuadratureSystem):
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

    def _configure_nonlinear(self, system, kwargs):
        """Builds the NonlinearCavityModel and its linearisation, and reads the integrator options"""
        from levisim.simulation.nonlinear_cavity import (
            NONLINEAR_TERMS,
            NonlinearCavityModel,
        )
        if isinstance(system, QuadratureSystem) or not hasattr(system, 'drift_matrix'):
            raise TypeError(
                "model='nonlinear' needs a MultiTrap with physical parameters (tweezers and particles), not a"
                f' {type(system).__name__}'
            )
        self.mtrap = system
        self.nonlinear_model = NonlinearCavityModel(
            system,
            damping=kwargs.get('damping', 'velocity'),
            terms=kwargs.get('terms', NONLINEAR_TERMS),
            linearised_terms=kwargs.get('linearised_terms', ()),
            interactions=kwargs.get('interactions', ()),
            trap_positions=kwargs.get('trap_positions'),
            feedback=kwargs.get('feedback', False),
        )
        self.system = self.nonlinear_model.quadrature_system
        self.integrator_name = kwargs.get('integrator', 'split')
        if self.integrator_name not in INTEGRATORS:
            raise ValueError(f'integrator={self.integrator_name!r}: use one of {INTEGRATORS}')
        self.substeps = int(kwargs.get('substeps', 1))
        if self.substeps < 1:
            raise ValueError(f'substeps = {self.substeps} should be ≥ 1')
        self.nonlinear_stage = kwargs.get('nonlinear_stage', 'rk4')
        if self.nonlinear_stage not in NONLINEAR_STAGES:
            raise ValueError(f'nonlinear_stage={self.nonlinear_stage!r}: use one of {NONLINEAR_STAGES}')
        if self.integrator_name == 'split':
            self.RK_method = f'split step (exact linear flow, {self.nonlinear_stage} remainder, {self.substeps} per sample)'
        else:
            self.RK_method = f'Heun ({self.substeps} per sample)'

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
        """
        Static intracavity field α = (A0 + i·P0)/2 of the MultiTrap's cavity; with model='nonlinear',
        the field at the true equilibrium of the nonlinear drift [√photons]
        """
        if not self._is_multitrap_cavity(cavity):
            raise ValueError(f'the static field of cavity mode {cavity!r} is not known: pass α=... to the record')
        if self.nonlinear_model is not None:
            return self.nonlinear_model.α_eq
        c = self.mtrap.cavity
        return complex(c.A0 + 1j * c.P0) / 2

    def optical_angular_frequency(self):
        """Angular frequency of the cavity light, cavity.ω [rad/s]"""
        return float(self._cavity_object('the optical frequency').ω)

    def zero_point_fluctuation(self, mode):
        """x_zpf of a mechanical mode, from the MultiTrap's Dimension (or the nonlinear model's scaling) [m]"""
        if self.nonlinear_model is not None:
            model = self.nonlinear_model
            index = model.labels.index(mode) - 1
            return float(model.zpf[index // 3, index % 3])
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
        # records sampled instantaneously (Position(sampling='instantaneous')): state components
        # copied at every sample time t_k, outside the augmented system
        instantaneous = [r for r in self.detection if getattr(r, 'instantaneous', False)]
        self.augmented = {'A': A, 'B': B, 'N': N, 'n': n, 'q': q, 'slices': slices,
                          'instantaneous': {r.label: index for index, r in enumerate(instantaneous)},
                          'instantaneous_indices': np.array([r.state_index for r in instantaneous], dtype=int)}
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
            if spec == 'zero' or (spec == 'equilibrium' and self.nonlinear_model is not None):
                return np.zeros((K, n))
            if spec != 'steady':
                options = "'steady', 'zero', 'equilibrium'" if self.nonlinear_model is not None else "'steady', 'zero'"
                raise ValueError(f'initial_state={spec!r}: use {options} or an array')
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
        if self.nonlinear_model is not None:
            return self._simulate_nonlinear()
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
        # instantaneous samples u_k[idx] at t_k: the NumPy loop copies them before each step; the
        # compiled loop gets extra rows of Φ that select them, with zero noise (no random numbers)
        idx = aug['instantaneous_indices']
        r = len(idx)
        U = np.empty((N, K, r)) if r else None
        if r:
            Φc_inst = np.ascontiguousarray(np.vstack([self.Φ, np.eye(n)[idx]]))

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
            if kernel is not None and r:
                ξ_inst = np.concatenate([ξ, np.zeros((nn, K, r))], axis=-1)
                YU = np.empty((nn, K, q + r))
                u = kernel(Φc_inst, ξ_inst, u, YU, states[start + 1:start + nn + 1] if states is not None else None,
                           self.threading.axis == 'setups')
                Y[start:start + nn] = YU[..., :q]
                U[start:start + nn] = YU[..., q:]
            elif kernel is not None:
                u = kernel(Φc, ξ, u, Y[start:start + nn], states[start + 1:start + nn + 1] if states is not None else None,
                           self.threading.axis == 'setups')
            else:
                for k in range(nn):
                    if r:
                        U[start + k] = u[:, idx]
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

        self.record(np.transpose(Y, (1, 0, 2)) / self.Δt, None if U is None else np.transpose(U, (1, 0, 2)))
        logger.info('Simulated {} samples ({:.4g} ms) in {:.2f} s: traces {}',
                    N, self.time_length / ms, time.perf_counter() - start_time, list(self.traces))

    # ------------------------------------------------------------------------------------------
    # integration of the nonlinear model
    # ------------------------------------------------------------------------------------------
    def nonlinear_flow(self, u, h, t=0.0):
        """
        One Runge-Kutta step ('rk4' or 'rk2', `nonlinear_stage`) of du/dt = R(u), the nonlinear
        remainder, over h from time t; u has shape (K, n) [quadratures]
        """
        R = self.nonlinear_model.remainder
        if self.nonlinear_stage == 'rk2':
            k1 = R(u, t)
            return u + h * R(u + (h / 2) * k1, t + h / 2)
        k1 = R(u, t)
        k2 = R(u + (h / 2) * k1, t + h / 2)
        k3 = R(u + (h / 2) * k2, t + h / 2)
        k4 = R(u + h * k3, t + h)
        return u + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    def discretise_split_step(self):
        """
        Transition matrices and noise factors of the exact linear flows of the split step, over h/2
        and h (h = Δt/substeps), for the WHOLE augmented state (the records' integrals continue
        across the flows of one sample): self.split = {'Φ_half', 'L_half', 'Φ_full', 'L_full', 'h'}
        """
        aug = self.augmented
        Q_c = aug['B'] @ aug['N'] @ aug['B'].T
        h = self.Δt / self.substeps
        Φ_half, Q_half = van_loan_discretisation(aug['A'], Q_c, h / 2)
        Φ_full, Q_full = van_loan_discretisation(aug['A'], Q_c, h)
        self.split = {
            'h': h,
            'Φ_half': Φ_half, 'L_half': _psd_square_root(Q_half),
            'Φ_full': Φ_full, 'L_full': _psd_square_root(Q_full),
        }
        return self.split

    def _simulate_nonlinear(self):
        """
        Integrates the nonlinear model (split step or Heun) over num_samples intervals and records
        the traces; the state is the deviation from the equilibrium
        """
        start_time = time.perf_counter()
        model = self.nonlinear_model
        aug = self.build_augmented_system()
        self.discretise()        # Φ, Q of the linearised model over Δt, for record_spectrum
        n, q = aug['n'], aug['q']
        K, N, M = self.num_trajectories, self.num_samples, self.substeps
        growth = self.system.largest_growth_rate()
        if growth > 0:
            logger.warning('Simulating a linearly unstable system (largest growth rate {:.4g} s⁻¹ at the equilibrium)', growth)
        logger.info(
            'Simulating {} samples of Δt = {:.4g} μs ({:.4g} ms), {} trajectories, nonlinear model ({}) with {} records, seed={!r}',
            N, self.Δt / μs, self.time_length / ms, K, self.RK_method, len(self.detection), self.seed,
        )
        u = self.initial_state()
        Y = np.empty((N, K, q))
        # instantaneous samples are read from the stored states (kept internally when store_state=False)
        idx = aug['instantaneous_indices']
        states = np.empty((N + 1, K, n)) if (self.store_state or len(idx)) else None
        if states is not None:
            states[0] = u

        self.backend_used = 'numpy'
        kernel = None
        if self.integrator_name == 'split':
            self.discretise_split_step()
            use_numba = resolve_backend(self.backend, N * M) == 'numba'
            if use_numba and self.backend == 'auto' and n > AUTO_NUMBA_MAX_STATE:
                # the threshold of the linear model, kept for consistency: above it the dense linear flow
                # dominates the step. Measured for the nonlinear model at n = 386 (64 particles) the two
                # backends were within the run-to-run spread of a shared machine (NumPy 690-750 μs,
                # numba 256-910 μs per step for one trajectory), so this choice is not benchmark-proven
                use_numba = False
            if use_numba and not model.compilable:
                if self.backend == 'numba':
                    raise unsupported_backend_error('nonlinear CavitySimulation', model.not_compilable_reason)
                logger.debug('No compiled kernel for this nonlinear configuration ({}): using NumPy',
                             model.not_compilable_reason)
                use_numba = False
            if use_numba:
                from levisim.simulation.backends.numba_nonlinear import split_step_chunk
                kernel = split_step_chunk
                self.backend_used = 'numba'
                self.threading = choose_threading(num_setups=K, num_particles=model.num_particles)
        else:
            if self.backend == 'numba':
                raise unsupported_backend_error('nonlinear CavitySimulation', "integrator='heun'")
            B̃ = aug['B']
            w, V = np.linalg.eigh(aug['N'])
            L_c = B̃ @ (V * np.sqrt(np.clip(w, 0, None)))      # L_c·L_cᵀ = B̃ÑB̃ᵀ
        for start in range(0, N, self.chunk_size):
            nn = min(self.chunk_size, N - start)
            if self.integrator_name == 'split':
                normals = self.standard_normals((nn, M + 1, K, n + q))
                ξ = np.empty_like(normals)
                ξ[:, [0, M]] = normals[:, [0, M]] @ self.split['L_half'].T
                if M > 1:
                    ξ[:, 1:M] = normals[:, 1:M] @ self.split['L_full'].T
                if kernel is not None:
                    u = kernel(self, ξ, u, Y[start:start + nn],
                               states[start + 1:start + nn + 1] if states is not None else None,
                               self.threading.axis == 'setups')
                else:
                    u = self._split_step_numpy(ξ, u, start, Y[start:start + nn],
                                               states[start + 1:start + nn + 1] if states is not None else None)
            else:
                normals = self.standard_normals((nn, M, K, L_c.shape[1]))
                u = self._heun_numpy(normals, L_c, u, start, Y[start:start + nn],
                                     states[start + 1:start + nn + 1] if states is not None else None)
            if not np.all(np.isfinite(u)):
                logger.warning('The nonlinear simulation diverged (non-finite state) before sample {}', start + nn)
        self.u_final = u
        U = np.transpose(states[:N][..., idx], (1, 0, 2)) if len(idx) else None
        if not self.store_state:
            states = None
        self.states = None if states is None else np.transpose(states, (1, 0, 2))
        self.X = self.states
        elapsed = time.perf_counter() - start_time
        logger.debug('Integrated {} steps × {} trajectories in {:.3f} s ({:.3f} μs per sample, {} backend)',
                     N, K, elapsed, 1e6 * elapsed / max(N, 1), self.backend_used)
        self.record(np.transpose(Y, (1, 0, 2)) / self.Δt, U)
        logger.info('Simulated {} samples ({:.4g} ms) in {:.2f} s: traces {}',
                    N, self.time_length / ms, time.perf_counter() - start_time, list(self.traces))

    def _split_step_numpy(self, ξ, u, start, Y, states):
        """The NumPy reference loop of the split step over one chunk (see the module docstring)"""
        n = u.shape[1]
        M, h = self.substeps, self.split['h']
        Φh_T = self.split['Φ_half'].T
        Φf_T = self.split['Φ_full'].T
        z = np.zeros((u.shape[0], ξ.shape[-1]))
        for k in range(ξ.shape[0]):
            z[:, :n] = u
            z[:, n:] = 0.0
            z = z @ Φh_T + ξ[k, 0]
            t0 = (start + k) * self.Δt
            for i in range(M):
                z[:, :n] = self.nonlinear_flow(z[:, :n], h, t0 + i * h)
                z = z @ (Φf_T if i < M - 1 else Φh_T) + ξ[k, i + 1]
            u = z[:, :n].copy()
            Y[k] = z[:, n:]
            if states is not None:
                states[k] = u
        return u

    def _heun_numpy(self, normals, L_c, u, start, Y, states):
        """Heun's scheme for the augmented system (state and record integrals) over one chunk"""
        aug = self.augmented
        A_T = aug['A'].T
        n = u.shape[1]
        M = self.substeps
        h = self.Δt / M
        R = self.nonlinear_model.remainder
        L_T = L_c.T * np.sqrt(h)
        z = np.zeros((u.shape[0], A_T.shape[0]))
        for k in range(normals.shape[0]):
            z[:, :n] = u
            z[:, n:] = 0.0
            t0 = (start + k) * self.Δt
            for i in range(M):
                t = t0 + i * h
                dW = normals[k, i] @ L_T
                f1 = z @ A_T
                f1[:, :n] += R(z[:, :n], t)
                z_pred = z + h * f1 + dW
                f2 = z_pred @ A_T
                f2[:, :n] += R(z_pred[:, :n], t + h)
                z = z + (h / 2) * (f1 + f2) + dW
            u = z[:, :n].copy()
            Y[k] = z[:, n:]
            if states is not None:
                states[k] = u
        return u

    @property
    def absolute_states(self):
        """With model='nonlinear' and store_state=True: the states including the equilibrium, u_eq + u"""
        if self.nonlinear_model is None or self.states is None:
            raise ValueError("absolute_states needs model='nonlinear' and store_state=True")
        return self.states + self.nonlinear_model.u_eq

    # ------------------------------------------------------------------------------------------
    # recording
    # ------------------------------------------------------------------------------------------
    def record(self, values, instantaneous=None):
        """
        Turns the sample averages of the auxiliary integrals, shape (K, num_samples, q), and the
        instantaneous state samples, shape (K, num_samples, r) (None if no record samples the state),
        into the records (self.records, arrays (K, num_samples)) and the Traces of trajectory 0
        """
        slices = self.augmented['slices']
        columns = self.augmented.get('instantaneous', {})
        self.records = {}
        for record in self.detection:
            if record.label in columns:
                c = columns[record.label]
                self.records[record.label] = record.finish(instantaneous[..., c:c + 1], self)
            else:
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

        A thin wrapper of `SpectralDensityEstimate` with `ensemble=True`: each trajectory is cut into
        num_segments segments, each (with zero_mean, as the simulation's) has its mean subtracted,
        is windowed (normalised to unit mean square) and Fourier transformed; the periodograms are
        averaged. Normalisation: two-sided by default, ∫S df = 〈h²〉 (levisim's convention; the
        vacuum of an optical record is 1); one_sided=True doubles 0 < f < f_Nyquist as
        SpectralDensityEstimate does. Returns (f [Hz], S [units²/Hz]) for 0 ≤ f ≤ f_Nyquist, sorted
        (the np.fft.rfftfreq grid of one segment). Use SpectralDensityEstimate(sim, (label, label),
        ensemble=True, ...) directly for CSDs, the per-segment arrays or the other options.
        """
        from levisim.analysis.spectral_density_estimate import SpectralDensityEstimate
        estimate = SpectralDensityEstimate(
            self, (label, label), num_segments=num_segments, one_sided=False, ensemble=True, window=window,
            subtract_segment_mean=self.zero_mean, keep_segments=False, fit=False,
        )
        M = estimate.num_samples_per_segment
        # the record is real: S(-f) = S(f), so the f ≥ 0 half (with the Nyquist bin for even M) is all of it
        S = np.real(estimate.fft_order_values[:M // 2 + 1])
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
        volts; excludes LO phase noise and the static carrier (static_carrier=True). For an
        instantaneous Position record, kind='discrete' is the exact spectrum of the sampled state and
        kind='continuous' has no sinc² factor. [units²/Hz]
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

        if getattr(record, 'instantaneous', False):
            S = self._state_spectrum(ω, record.state_index, kind)
        elif isinstance(record, Heterodyne):
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
        if kind == 'continuous' and not getattr(record, 'instantaneous', False):
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

    def _state_spectrum(self, ω, index, kind):
        """
        Spectrum of state component `index` [·²/Hz]: kind='discrete', the exact spectrum of its
        instantaneous samples u_k (aliasing included), S = Δt·[M⁻¹·Q_uu·M⁻†]_ii with M = z·I − Φ_u,
        z = e^{-iωΔt}; kind='continuous', the symmetrised spectrum of the continuous u_i(t)
        """
        n = self.augmented['n']
        if kind == 'discrete':
            z = np.exp(-1j * ω * self.Δt)
            Minv = np.linalg.inv(z[:, None, None] * np.eye(n)[None] - self.Φ[:n][None])
            h = Minv[:, index, :]
            return np.einsum('wi,ij,wj->w', h, self.Q[:n, :n], h.conj()).real * self.Δt
        aug = self.augmented
        GB = np.linalg.solve(-1j * ω[:, None, None] * np.eye(n)[None] - aug['A'][:n, :n][None],
                             np.broadcast_to(aug['B'][:n].astype(complex), (len(ω), n, aug['B'].shape[1])))
        g = GB[:, index, :]
        return np.einsum('wi,ij,wj->w', g, aug['N'], g.conj()).real

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

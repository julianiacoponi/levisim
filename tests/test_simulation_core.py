"""
Tests of the simulation core: StochasticSimulation, the integrators, FreeSpaceSimulation and the
deprecated Simulation alias

The free-space physics itself is tested in test_simulation.py and test_trap_physics.py; here the
refactor into levisim.simulation is pinned: identical numbers to the pre-refactor loops, the alias
and its warning, and that the base class and integrators work for a system other than a bead.
"""
import pickle
import warnings

import numpy as np
import pytest

import levisim
from levisim.data.simulation import Simulation
from levisim.physics.kick import Kick
from levisim.physics.trap import Trap
from levisim.simulation import FreeSpaceSimulation, StochasticSimulation
from levisim.simulation.integrators import INTEGRATORS, Heun, KasdinRK4, get_integrator

Δt = 1e-7


def legacy_vector_2(trap, num_samples, Δt, standard_normals):
    """
    The 'vector_2' (Heun) loop exactly as written in levisim 0.1.0 (levisim/data/simulation.py),
    with the trap's default physics (no kicks, gravity or recoil)
    """
    m = trap.bead.m
    ψ = trap.ψ
    G = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                  [1, 0, 0, np.cos(ψ)], [0, 1, 0, np.sin(ψ)], [0, 0, 1, 0]])

    def dynamics(t, X):
        x, y, z, px, py, pz = X
        Ftwx, Ftwy, Ftwz = trap.Ftw(x, y, z)
        return np.array([px / m, py / m, pz / m, Ftwx - trap.Γx*px, Ftwy - trap.Γy*py, Ftwz - trap.Γz*pz])

    X = np.zeros([num_samples + 1, 6])
    time_array = np.linspace(0, num_samples*Δt, num_samples + 1)[:-1]
    σ = np.sqrt(Δt * np.array([trap.Sth, trap.Sth, trap.Sth, trap.Sdir]))
    chunk = 2**15
    for start in range(0, num_samples, chunk):
        normals = standard_normals((min(chunk, num_samples - start), 1, 4))
        dP = (normals[:, 0, :] * σ) @ G.T
        for k in range(len(normals)):
            n = start + k
            t = time_array[n]
            K1 = dynamics(t, X[n])*Δt + dP[k]
            K2 = dynamics(t + Δt, X[n] + K1)*Δt + dP[k]
            X[n + 1] = X[n] + 0.5 * (K1 + K2)
    return X


@pytest.mark.parametrize('seed', [None, 5])
def test_vector_2_bit_identical_to_0_1_0_loop(seed):
    """Heun regression: the refactored integrator reproduces the 0.1.0 loop bit for bit (across a chunk edge)"""
    trap = Trap(β=0.4, ψ=0.7)
    num_samples = 2**15 + 500
    if seed is None:
        np.random.seed(99)
        X_legacy = legacy_vector_2(trap, num_samples, Δt, lambda shape: np.random.normal(size=shape))
        np.random.seed(99)
    else:
        rng = np.random.default_rng(seed)
        X_legacy = legacy_vector_2(trap, num_samples, Δt, rng.standard_normal)
    sim = FreeSpaceSimulation(trap=trap, num_samples=num_samples, Δt=Δt, RK_method='vector_2', seed=seed,
                              zero_mean=False, backend='numpy')
    assert np.array_equal(sim.X, X_legacy)


def test_simulation_alias_is_a_deprecated_free_space_simulation():
    Simulation._deprecation_warned = False
    with pytest.warns(DeprecationWarning, match='FreeSpaceSimulation'):
        a = Simulation(num_samples=500, Δt=Δt, seed=3, store_momenta=True)
    # once per session
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        Simulation(num_samples=10, Δt=Δt, seed=3)
    b = FreeSpaceSimulation(num_samples=500, Δt=Δt, seed=3, store_momenta=True)
    assert isinstance(a, FreeSpaceSimulation) and isinstance(a, StochasticSimulation)
    assert np.array_equal(a.X, b.X)
    assert list(a.traces) == list(b.traces) == ['x', 'y', 'z', 'px', 'py', 'pz']
    assert levisim.Simulation is Simulation
    assert levisim.FreeSpaceSimulation is FreeSpaceSimulation
    assert levisim.data.Simulation is Simulation


def test_free_space_simulation_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter('error', DeprecationWarning)
        FreeSpaceSimulation(num_samples=10, Δt=Δt, seed=1)


def make_options_trap():
    return Trap(photon_recoil=True, kicks=[Kick(start=5*Δt, duration=3*Δt, K=10, θ=0.5)])


@pytest.mark.parametrize('RK_method', ['vector_4', 'vector_2'])
def test_all_options_seeded_and_picklable(RK_method):
    """Every optional feature runs through the base class; the instance pickles and keeps its numbers"""
    kwargs = dict(
        num_samples=300, Δt=Δt, seed=2, RK_method=RK_method, gravity=True,
        detection_floor={'x': 1e-27}, store_momenta=True, store_radial=True,
    )
    a = FreeSpaceSimulation(trap=make_options_trap(), **kwargs)
    b = FreeSpaceSimulation(trap=make_options_trap(), **kwargs)
    assert np.array_equal(a.X, b.X)
    assert set(a.traces) == {'x', 'y', 'z', 'px', 'py', 'pz', 'r'}
    assert a.trap.kicks[0].Fx is not None  # the kick was applied
    c = pickle.loads(pickle.dumps(a))
    assert np.array_equal(c.X, a.X) and np.array_equal(c.r.values, a.r.values)


def test_time_axis_properties():
    sim = FreeSpaceSimulation(num_samples=1000, Δt=Δt, seed=1)
    assert sim.time_length == 1000 * Δt
    assert sim.sample_rate == 1 / Δt and sim.Nyquist_frequency == 0.5 / Δt
    assert sim.min_Δf == 1 / sim.time_length
    assert len(sim.time_array) == 1001 and sim.time_array[-1] == pytest.approx(sim.time_length)
    assert 'FreeSpaceSimulation(id=' in sim.info_str and 'Tx via Var(x)' in sim.info_str


def test_integrator_registry():
    assert isinstance(get_integrator('vector_4'), KasdinRK4) and isinstance(get_integrator('vector_2'), Heun)
    assert get_integrator(INTEGRATORS['vector_2']) is INTEGRATORS['vector_2']
    with pytest.raises(ValueError, match='scalar_2'):
        get_integrator('scalar_2')
    with pytest.raises(ValueError, match='Unknown'):
        get_integrator('vector_3')
    assert KasdinRK4().noise_std([1.0, 4.0], 0.5).shape == (4, 2)
    # Kasdin's stage times are consistent with its coefficients
    assert KasdinRK4().stage_offsets[2] == pytest.approx(0.63836696743415)


class OrnsteinUhlenbeck(StochasticSimulation):
    """A system that is not a bead: dv = -γ v dt + dW, Var dW = S dt, recorded as 'v'"""

    def configure(self, **kwargs):
        self.γ = kwargs.get('γ', 1e5)
        self.S = kwargs.get('S', 2.0)

    def initial_state(self):
        return np.array([0.0])

    def drift(self, t, X):
        return -self.γ * X

    def noise_matrix(self):
        return np.array([[1.0]])

    def noise_spectral_densities(self):
        return np.array([self.S])

    def record(self, X):
        self.v = self.make_trace(self.add_detection_floor(X[:, 0], 'v'), 'v', 'm/s')
        self.traces = {'v': self.v}


@pytest.mark.parametrize('RK_method', ['vector_4', 'vector_2'])
def test_base_class_and_integrators_serve_another_system(RK_method):
    """Stationary variance of an Ornstein-Uhlenbeck process, S/(2γ), from the generic machinery"""
    γ, S = 1e5, 2.0
    sim = OrnsteinUhlenbeck(num_samples=200_000, Δt=Δt, seed=4, RK_method=RK_method, γ=γ, S=S)
    # γT = 2000 correlation times => relative standard error ≈ √(2/(γT)) ≈ 3 %
    assert np.var(sim.v.values) == pytest.approx(S / (2 * γ), rel=0.1)
    assert sim.v.units == 'm/s'

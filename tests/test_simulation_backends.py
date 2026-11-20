"""
Tests of the simulation backends: backend selection, the threading policy, and the compiled (numba)
free-space loops against the NumPy reference on identical random numbers

The numba tests are skipped when numba is not installed; the selection tests run either way.
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from levisim.core.units import ms, ns
from levisim.physics.kick import Kick
from levisim.physics.trap import Trap
from levisim.simulation import FreeSpaceSimulation, backends
from levisim.simulation.backends import (
    AUTO_NUMBA_MIN_STEPS,
    choose_threading,
    numba_available,
    resolve_backend,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
Δt = 1e-7
needs_numba = pytest.mark.skipif(not numba_available(), reason='numba is not installed')


@pytest.fixture
def without_numba(monkeypatch):
    """Pretend numba is not installed"""
    monkeypatch.setenv('LEVISIM_DISABLE_NUMBA', '1')
    backends.numba_available.cache_clear()
    yield
    monkeypatch.delenv('LEVISIM_DISABLE_NUMBA')
    backends.numba_available.cache_clear()


def test_backend_keyword_validation():
    with pytest.raises(ValueError, match='backend'):
        FreeSpaceSimulation(num_samples=10, Δt=Δt, seed=1, backend='jax')


def test_numba_backend_without_numba_raises_import_error(without_numba):
    with pytest.raises(ImportError, match=r'levisim\[fast\]'):
        FreeSpaceSimulation(num_samples=10, Δt=Δt, seed=1, backend='numba')
    # 'auto' quietly uses NumPy
    sim = FreeSpaceSimulation(num_samples=AUTO_NUMBA_MIN_STEPS, Δt=Δt, seed=1, backend='auto')
    assert sim.backend_used == 'numpy'


def test_resolve_backend(without_numba):
    assert resolve_backend('numpy', 10**9) == 'numpy'
    assert resolve_backend('auto', 10**9) == 'numpy'
    with pytest.raises(ImportError):
        resolve_backend('numba', 10)


def test_threading_policy():
    """One free-space bead is serial; ensembles thread over setups, large systems over particles"""
    assert choose_threading(1, 1, max_threads=8).axis == 'serial'
    assert choose_threading(1, 64, max_threads=8).axis == 'serial'
    assert choose_threading(1, 256, max_threads=8) == backends.ThreadingPlan('particles', 8)
    assert choose_threading(16, 2, max_threads=8) == backends.ThreadingPlan('setups', 8)
    assert choose_threading(4096, 2, max_threads=1).axis == 'serial'


def test_import_levisim_does_not_import_numba():
    code = 'import sys, levisim; levisim.FreeSpaceSimulation(num_samples=10, seed=1); print("numba" in sys.modules)'
    env = {**os.environ, 'PYTHONPATH': str(REPO_ROOT), 'LEVISIM_LOG_FILE': '0'}
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, env=env, cwd=REPO_ROOT, check=True)
    assert result.stdout.strip().splitlines()[-1] == 'False'


def run_both(RK_method, num_samples=AUTO_NUMBA_MIN_STEPS + 1000, seed=3, trap=None, **kwargs):
    """The same seeded simulation with the NumPy and the numba loops"""
    sims = {}
    for backend in ('numpy', 'numba'):
        np.random.seed(11)  # used when seed=None
        sims[backend] = FreeSpaceSimulation(
            trap=trap() if trap else Trap(), num_samples=num_samples, Δt=Δt, seed=seed,
            RK_method=RK_method, backend=backend, **kwargs,
        )
        assert sims[backend].backend_used == backend
    return sims['numpy'], sims['numba']


def max_relative_difference(a, b):
    return np.max(np.abs(a - b)) / np.max(np.abs(a))


def kicked_trap():
    return Trap(photon_recoil=True, kicks=[
        Kick(start=0.5*ms, duration=200*ns, K=20, θ=0.3),
        Kick(start=1*ms, duration=100*ns, K=5, θ=2.0),
        Kick(start=99*ms, K=1),  # never active
    ])


CASES = {
    'default': {},
    'legacy RandomState': {'seed': None},
    'gravity': {'gravity': True, 'gravity_direction': (0, 1, -2)},
    'photon recoil': {'trap': lambda: Trap(photon_recoil=True)},
    'override_Ω_dict': {'trap': lambda: Trap(override_Ω_dict={'x': 2*np.pi*150e3, 'y': 2*np.pi*170e3, 'z': 2*np.pi*40e3})},
    'directed force': {'trap': lambda: Trap(β=0.3, ψ=0.4)},
    'X0': {'X0': np.array([1e-9, -2e-9, 5e-9, 0, 1e-20, 0])},
    'kicks, recoil, gravity, floor, stored traces': {
        'trap': kicked_trap, 'gravity': True, 'detection_floor': {'y': 1e-27}, 'store_momenta': True, 'store_radial': True,
    },
}


@needs_numba
@pytest.mark.parametrize('RK_method', ['vector_4', 'vector_2'])
@pytest.mark.parametrize('case', list(CASES))
def test_numba_matches_numpy_to_round_off(RK_method, case):
    """
    Same random numbers, same arithmetic: positions agree to ≲ 1e-12 relative (observed ≲ 3e-13 over
    33 768 steps; the compiled exp can differ from NumPy's in the last bit)
    """
    a, b = run_both(RK_method, **CASES[case])
    assert max_relative_difference(a.X[:, :3], b.X[:, :3]) < 1e-12
    assert max_relative_difference(a.X[:, 3:], b.X[:, 3:]) < 1e-12
    assert list(a.traces) == list(b.traces)
    for label in a.traces:
        assert max_relative_difference(a.traces[label].values, b.traces[label].values) < 1e-12
    # the kicks record the same applied impulses
    for kick_a, kick_b in zip(a.trap.kicks, b.trap.kicks):
        assert (kick_a.p, kick_a.F0, kick_a.Fx, kick_a.Fy) == (kick_b.p, kick_b.F0, kick_b.Fx, kick_b.Fy)
    # the random stream is consumed identically: the next draw agrees
    if a.rng is not None:
        assert a.rng.standard_normal() == b.rng.standard_normal()


@needs_numba
def test_auto_backend_choice():
    short = FreeSpaceSimulation(num_samples=AUTO_NUMBA_MIN_STEPS - 1, Δt=Δt, seed=1)
    long = FreeSpaceSimulation(num_samples=AUTO_NUMBA_MIN_STEPS, Δt=Δt, seed=1)
    assert short.backend_used == 'numpy'
    assert long.backend_used == 'numba'


class ModifiedPhysics(FreeSpaceSimulation):
    """A subclass that changes the drift, which the compiled kernel cannot know about"""

    def dynamics_vector(self, t, X):
        return 1.01 * super().dynamics_vector(t, X)


@needs_numba
def test_unsupported_configurations_fall_back_to_numpy():
    assert ModifiedPhysics(num_samples=100, Δt=Δt, seed=1, backend='numba').backend_used == 'numpy'
    assert FreeSpaceSimulation(num_samples=100, Δt=Δt, seed=1, backend='numba', rk_loop_debug=True).backend_used == 'numpy'
    assert FreeSpaceSimulation(num_samples=100, Δt=Δt, seed=1, backend='numba').backend_used == 'numba'

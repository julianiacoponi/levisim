"""
levisim.simulation.backends
===========================
Optional compiled backends of the stochastic time loops

The NumPy loops of `levisim.simulation.integrators` are the reference. With numba installed
(`pip install levisim[fast]`), compiled loops consume exactly the same random numbers, so
trajectories agree with NumPy to round-off: `numba_free_space` (FreeSpaceSimulation),
`numba_multi_particle` (MultiParticleFreeSpaceSimulation), `numba_linear` and `numba_nonlinear`
(CavitySimulation), `numba_rigid_body` (RigidBodySimulation). They share the compiled forces of
`numba_elements` (PaulTrap, MagneticTrap and its field sources, HarmonicPotential,
UniformElectricField) and `numba_binding` (OpticalBinding).

This module is imported by every simulation but does not import numba itself: numba is imported
only when a compiled loop is actually used, so `import levisim` stays fast without or with numba.

- `numba_available()`: whether numba can be imported
- `require_numba()`: raises an ImportError with installation advice when it cannot
- `resolve_backend(backend, num_steps)`: which backend a simulation should use
- `UnsupportedBackendError`: raised when backend='numba' is requested explicitly for a configuration
  without a compiled kernel (backend='auto' falls back to NumPy instead)
- `choose_threading(num_setups, num_particles)`: the threading plan for compiled kernels
"""

import importlib.util
import os
from dataclasses import dataclass
from functools import lru_cache

__all__ = [
    'AUTO_NUMBA_MIN_STEPS',
    'BACKENDS',
    'ThreadingPlan',
    'UnsupportedBackendError',
    'choose_threading',
    'numba_available',
    'require_numba',
    'resolve_backend',
]

BACKENDS = ('auto', 'numpy', 'numba')

# 'auto' uses numba only for runs of at least this many steps: below it, loading the cached
# kernels costs more than the NumPy loop itself. Re-measured on a quiet 8-core arm64 machine
# (benchmarks/auto_backend_thresholds.py; default Trap, fresh process, median of 3): loading the
# kernels costs 0.31 s once per process; the first run takes NumPy 0.197 s vs numba 0.316 s at 2¹³
# steps and 0.408 s vs 0.310 s at 2¹⁴ (break-even ≈ 1.2 × 10⁴ steps). Later runs in the same process
# are 30-65× faster with numba at every length
AUTO_NUMBA_MIN_STEPS = 2**14


class UnsupportedBackendError(ValueError):
    """
    backend='numba' was requested explicitly for a configuration that has no compiled kernel

    An explicit request never silently runs another backend: use backend='numpy', or backend='auto'
    (which falls back to NumPy for such configurations).
    """


def unsupported_backend_error(owner, reason):
    """The UnsupportedBackendError for simulation class name owner and the reason it cannot compile"""
    return UnsupportedBackendError(
        f"backend='numba' was requested, but this {owner} configuration has no compiled kernel "
        f"({reason}). Use backend='numpy', or backend='auto' to fall back to NumPy automatically."
    )


@lru_cache(maxsize=1)
def numba_available():
    """True if numba is installed (and not disabled with LEVISIM_DISABLE_NUMBA=1)"""
    if os.environ.get('LEVISIM_DISABLE_NUMBA', '0') not in ('', '0'):
        return False
    return importlib.util.find_spec('numba') is not None


def require_numba():
    """Raises ImportError with installation advice if numba is not available"""
    if not numba_available():
        raise ImportError(
            "backend='numba' needs the optional dependency numba, which is not installed (or is disabled "
            "with LEVISIM_DISABLE_NUMBA). Install it with `pip install levisim[fast]` (or `pip install "
            "numba`), or use backend='numpy' or backend='auto'."
        )


def resolve_backend(backend, num_steps):
    """
    The backend a simulation of num_steps steps should try, 'numpy' or 'numba'

    - 'numpy': always NumPy
    - 'numba': numba (ImportError if it is not installed)
    - 'auto': numba if it is installed and num_steps ≥ AUTO_NUMBA_MIN_STEPS, otherwise NumPy

    With 'auto' the simulation may still fall back to NumPy when its configuration has no compiled
    kernel; with an explicit 'numba' it raises UnsupportedBackendError instead.
    """
    if backend not in BACKENDS:
        raise ValueError(f"Unknown backend={backend!r}; use one of {BACKENDS}")
    if backend == 'numpy':
        return 'numpy'
    if backend == 'numba':
        require_numba()
        return 'numba'
    return 'numba' if (numba_available() and num_steps >= AUTO_NUMBA_MIN_STEPS) else 'numpy'


@dataclass(frozen=True)
class ThreadingPlan:
    """
    How a compiled kernel should use threads

    axis: 'serial' (one thread), 'setups' (independent trajectories or parameter sets in parallel,
    each integrated serially) or 'particles' (the force/pair loop of one large system in parallel)
    num_threads: threads to use (1 for 'serial')
    """
    axis: str
    num_threads: int


# thresholds from the round-5 backend benchmark (bench_summary.md, 8-core arm64): threading over
# setups paid off from K = 16 (2.6x) to K = 4096 (4.2x); threading the pair loop of one system lost
# at N = 8 and N = 64 (thread start-up per step dominates) and won only at N = 256 (1.4x)
MIN_SETUPS_TO_THREAD = 8
MIN_PARTICLES_TO_THREAD = 128


def choose_threading(num_setups=1, num_particles=1, max_threads=None):
    """
    Chooses the threading of a compiled kernel from the problem's shape

    - several independent setups (ensemble members, parameter sweeps): thread over the setups,
      which share nothing and scale almost linearly
    - one setup with many particles: thread over the particles (pair forces accumulated into
      per-particle local scalars, never into a shared array inside the inner loop)
    - otherwise (e.g. one free-space bead): serial, since there is nothing worth a thread

    max_threads defaults to NUMBA_NUM_THREADS if set, else os.cpu_count().
    """
    if max_threads is None:
        max_threads = int(os.environ.get('NUMBA_NUM_THREADS', '0') or '0') or os.cpu_count() or 1
    if num_setups >= MIN_SETUPS_TO_THREAD and max_threads > 1:
        return ThreadingPlan('setups', min(max_threads, num_setups))
    if num_setups == 1 and num_particles >= MIN_PARTICLES_TO_THREAD and max_threads > 1:
        return ThreadingPlan('particles', max_threads)
    return ThreadingPlan('serial', 1)

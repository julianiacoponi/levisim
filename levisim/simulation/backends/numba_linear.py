"""
Compiled (numba) time loop of the exact linear CavitySimulation

One step is z = Φ·u + ξ for each trajectory, with the transition matrix Φ (restricted to the
state columns) and the correlated noise ξ = L·N(0, 1) computed in NumPy per chunk, exactly as the
NumPy loop of `levisim.simulation.cavity` does; the kernel only runs the recursion. Trajectories
agree with NumPy to round-off (the order of the sums differs).

Threading: with several trajectories (`choose_threading(num_setups=K)` says 'setups') each
trajectory is advanced in its own thread; one trajectory runs serially.
"""

import numpy as np
from numba import njit, prange

__all__ = [
    'exact_linear_chunk',
]


@njit(cache=True, fastmath=True)
def _advance_one(Φ, ξ, u, Y, states, store, j):
    nn = ξ.shape[0]
    n = u.shape[1]
    nq = Φ.shape[0]
    z = np.empty(nq)
    for k in range(nn):
        for c in range(nq):
            s = 0.0
            for i in range(n):
                s += Φ[c, i] * u[j, i]
            z[c] = s + ξ[k, j, c]
        for i in range(n):
            u[j, i] = z[i]
            if store:
                states[k, j, i] = z[i]
        for c in range(nq - n):
            Y[k, j, c] = z[n + c]


@njit(cache=True)
def _serial(Φ, ξ, u, Y, states, store):
    for j in range(u.shape[0]):
        _advance_one(Φ, ξ, u, Y, states, store, j)


@njit(cache=True, parallel=True)
def _parallel(Φ, ξ, u, Y, states, store):
    for j in prange(u.shape[0]):
        _advance_one(Φ, ξ, u, Y, states, store, j)


def exact_linear_chunk(Φ, ξ, u, Y, states, threaded):
    """
    Advances K trajectories over one chunk of steps, in place

    Φ: (n + q, n) transition matrix restricted to the state columns (C-contiguous); ξ: (nn, K, n + q) noise; u: (K, n) state (the
    returned array holds the final state); Y: (nn, K, q) output of the records' integrals;
    states: (nn, K, n) or None; threaded: thread over the trajectories.
    """
    u = np.ascontiguousarray(u, dtype=np.float64).copy()
    store = states is not None
    buffer = states if store else np.empty((0, 0, 0))
    if threaded:
        _parallel(Φ, ξ, u, Y, buffer, store)
    else:
        _serial(Φ, ξ, u, Y, buffer, store)
    return u

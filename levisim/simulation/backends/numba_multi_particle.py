"""
Compiled (numba) time loops of MultiParticleFreeSpaceSimulation

Each particle's own forces are those of the free-space kernel (`numba_free_space._drift`, one row
of parameters per particle); the interactions are power-law pair potentials U = C·(r - s)^(-n)
(`PowerLawPairInteraction.power_law`), given as rows [j, k, C, s, n]. The pair loop accumulates
each pair's force in local scalars and writes it once to both particles. The random numbers are
the simulation's own (`_noise_chunks`), scaled in NumPy exactly as the NumPy loops scale them, so
trajectories agree with the NumPy reference to round-off.

Supported: traps of one Gaussian-beam tweezer (or override_Ω_dict) with feedback, photon recoil
and gravity, without kicks; power-law interactions (Coulomb, NewtonianGravity, Casimir).
"""

import numpy as np
from numba import njit

from levisim.simulation.backends.numba_free_space import _drift, free_space_parameters

__all__ = [
    'integrate_multi_particle',
    'pair_terms',
]


def pair_terms(sim):
    """Rows [j, k, C, s, n] of every interacting pair of every interaction of the simulation"""
    rows = []
    for interaction in sim.interactions:
        for j, k in interaction.pair_indices(sim.num_particles):
            C, s, n = interaction.power_law(sim.particles[j], sim.particles[k])
            rows.append([j, k, C, s, n])
    return np.array(rows, dtype=np.float64).reshape(-1, 5)


@njit(cache=True)
def _multi_drift(t, X, P, kicks, origins, terms, out):
    """drift of MultiParticleFreeSpaceSimulation, written into out (length 6N)"""
    N = P.shape[0]
    for n in range(N):
        _drift(t, X[6*n:6*n + 6], P[n], kicks, out[6*n:6*n + 6])
    for row in range(terms.shape[0]):
        j = int(terms[row, 0])
        k = int(terms[row, 1])
        C = terms[row, 2]
        s = terms[row, 3]
        p = terms[row, 4]
        dx = (origins[j, 0] + X[6*j]) - (origins[k, 0] + X[6*k])
        dy = (origins[j, 1] + X[6*j + 1]) - (origins[k, 1] + X[6*k + 1])
        dz = (origins[j, 2] + X[6*j + 2]) - (origins[k, 2] + X[6*k + 2])
        r = np.sqrt(dx*dx + dy*dy + dz*dz)
        d = r - s
        U = C * d**(-p)
        dU = -p * U / d
        # F_j = -U'·r̂ = -F_k, accumulated locally then written once
        fx = -dU * dx / r
        fy = -dU * dy / r
        fz = -dU * dz / r
        out[6*j + 3] += fx
        out[6*j + 4] += fy
        out[6*j + 5] += fz
        out[6*k + 3] -= fx
        out[6*k + 4] -= fy
        out[6*k + 5] -= fz


@njit(cache=True)
def _kasdin_chunk(X, start, time_array, dW, G, Δt, c, P, kicks, origins, terms):
    """Kasdin RK4 over one chunk of steps, in place on X (see numba_free_space._kasdin_chunk)"""
    a21, a31, a32, a41, a42, a43 = c[0], c[1], c[2], c[3], c[4], c[5]
    α1, α2, α3, α4 = c[6], c[7], c[8], c[9]
    dim = X.shape[1]
    num_noises = G.shape[1]
    f = np.empty(dim)
    Xs = np.empty(dim)
    K1 = np.empty(dim)
    K2 = np.empty(dim)
    K3 = np.empty(dim)
    K4 = np.empty(dim)
    for k in range(dW.shape[0]):
        n = start + k
        t = time_array[n]
        Xn = X[n]

        _multi_drift(t, Xn, P, kicks, origins, terms, f)
        for i in range(dim):
            g = 0.0
            for j in range(num_noises):
                g += G[i, j] * dW[k, 0, j]
            K1[i] = (f[i] + g)*Δt

        for i in range(dim):
            Xs[i] = Xn[i] + a21*K1[i]
        _multi_drift(t + a21*Δt, Xs, P, kicks, origins, terms, f)
        for i in range(dim):
            g = 0.0
            for j in range(num_noises):
                g += G[i, j] * dW[k, 1, j]
            K2[i] = (f[i] + g)*Δt

        for i in range(dim):
            Xs[i] = Xn[i] + a31*K1[i] + a32*K2[i]
        _multi_drift(t + (a31 + a32)*Δt, Xs, P, kicks, origins, terms, f)
        for i in range(dim):
            g = 0.0
            for j in range(num_noises):
                g += G[i, j] * dW[k, 2, j]
            K3[i] = (f[i] + g)*Δt

        for i in range(dim):
            Xs[i] = Xn[i] + a41*K1[i] + a42*K2[i] + a43*K3[i]
        _multi_drift(t + (a41 + a42 + a43)*Δt, Xs, P, kicks, origins, terms, f)
        for i in range(dim):
            g = 0.0
            for j in range(num_noises):
                g += G[i, j] * dW[k, 3, j]
            K4[i] = (f[i] + g)*Δt

        for i in range(dim):
            X[n + 1, i] = Xn[i] + α1*K1[i] + α2*K2[i] + α3*K3[i] + α4*K4[i]


@njit(cache=True)
def _heun_chunk(X, start, time_array, dP, Δt, P, kicks, origins, terms):
    """Heun over one chunk of steps, in place on X; dP = (normals·σ) Gᵀ per step"""
    dim = X.shape[1]
    f = np.empty(dim)
    K1 = np.empty(dim)
    Xs = np.empty(dim)
    for k in range(dP.shape[0]):
        n = start + k
        t = time_array[n]
        Xn = X[n]
        _multi_drift(t, Xn, P, kicks, origins, terms, f)
        for i in range(dim):
            K1[i] = f[i]*Δt + dP[k, i]
            Xs[i] = Xn[i] + K1[i]
        _multi_drift(t + Δt, Xs, P, kicks, origins, terms, f)
        for i in range(dim):
            X[n + 1, i] = Xn[i] + 0.5 * (K1[i] + (f[i]*Δt + dP[k, i]))


def integrate_multi_particle(sim, integrator, G, S, X0, time_array):
    """
    The compiled equivalent of integrator.integrate(sim.drift, G, S, X0, ...) for a
    MultiParticleFreeSpaceSimulation; returns the states, shape (len(time_array) + 1, 6N)
    """
    Δt = sim.Δt
    num_steps = len(time_array)
    P = np.array([free_space_parameters(single)[0] for single in sim.single_particle_systems])
    kicks = np.zeros((0, 4))
    origins = np.ascontiguousarray(sim.trap_positions, dtype=np.float64)
    terms = pair_terms(sim)
    G = np.ascontiguousarray(G, dtype=np.float64)
    time_array = np.ascontiguousarray(time_array, dtype=np.float64)
    X = np.zeros([num_steps + 1, len(X0)])
    X[0] = X0
    if integrator.name == 'vector_4':
        c = np.array([
            integrator.a21, integrator.a31, integrator.a32, integrator.a41, integrator.a42, integrator.a43,
            integrator.α1, integrator.α2, integrator.α3, integrator.α4,
        ], dtype=np.float64)
        Σ = integrator.noise_std(S, Δt)
        for start, normals in sim._noise_chunks(num_steps, integrator.num_stages, len(S)):
            dW = normals * Σ
            _kasdin_chunk(X, start, time_array, dW, G, Δt, c, P, kicks, origins, terms)
    elif integrator.name == 'vector_2':
        σ = integrator.noise_std(S, Δt)
        for start, normals in sim._noise_chunks(num_steps, integrator.num_stages, len(S)):
            dP = np.ascontiguousarray((normals[:, 0, :] * σ) @ G.T)
            _heun_chunk(X, start, time_array, dP, Δt, P, kicks, origins, terms)
    else:
        raise ValueError(f'No compiled kernel for the integrator {integrator!r}')
    return X

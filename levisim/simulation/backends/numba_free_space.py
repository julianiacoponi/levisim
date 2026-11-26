"""
Compiled (numba) time loops of FreeSpaceSimulation

The kernels advance the state [x, y, z, px, py, pz] with the arithmetic of
`FreeSpaceSimulation.dynamics_vector` and of the NumPy integrators (Kasdin RK4 'vector_4', Heun
'vector_2'), and take the random numbers from the simulation's own `_noise_chunks`, scaled in NumPy
exactly as the NumPy loops scale them. Trajectories therefore agree with the NumPy reference to
round-off (the compiled exp, and the order of the products in G·dW, can differ in the last bit).

Supported: the Gaussian-beam tweezer force (or override_Ω_dict), gas + feedback damping, kicks,
photon recoil, gravity, both integrators. Kernels are compiled on first use and cached on disk
(numba's cache=True), so later processes only load them.

Threading: one free-space bead has nothing to thread over, so these kernels are serial (see
`levisim.simulation.backends.choose_threading` for the multi-particle and ensemble policy).
"""

import numpy as np
from numba import njit

__all__ = [
    'free_space_parameters',
    'integrate_free_space',
]

# indices into the parameter vector P built by free_space_parameters
_M, _GAMMA_X, _GAMMA_Y, _GAMMA_Z = 0, 1, 2, 3
_OVERRIDE, _OMEGA_X, _OMEGA_Y, _OMEGA_Z = 4, 5, 6, 7
_E0, _ZR, _WX0, _WY0, _WZ0, _RE_ALPHA, _IM_ALPHA, _K = 8, 9, 10, 11, 12, 13, 14, 15
_RECOIL, _GRAVITY, _GX, _GY, _GZ = 16, 17, 18, 19, 20
_NUM_PARAMETERS = 21


def free_space_parameters(sim):
    """
    The physics of a FreeSpaceSimulation as (P, kicks): a float vector of the trap's parameters and a
    (num_kicks, 4) array of [start, end, F_kick,x, F_kick,y]
    """
    trap = sim.trap
    tweezer = trap.tweezer
    P = np.zeros(_NUM_PARAMETERS)
    P[_M] = trap.bead.m
    P[_GAMMA_X], P[_GAMMA_Y], P[_GAMMA_Z] = trap.Γx, trap.Γy, trap.Γz
    if trap.override_Ω_dict:
        P[_OVERRIDE] = 1.0
        P[_OMEGA_X], P[_OMEGA_Y], P[_OMEGA_Z] = (trap.override_Ω_dict[axis] for axis in 'xyz')
    else:
        α_eff = trap.α_eff
        P[_E0], P[_ZR] = tweezer.E0, tweezer.zR
        P[_WX0], P[_WY0], P[_WZ0] = tweezer.wx0, tweezer.wy0, tweezer.wz0
        P[_RE_ALPHA], P[_IM_ALPHA] = np.real(α_eff), np.imag(α_eff)
    if trap.photon_recoil:
        α_eff = trap.α_eff
        P[_RECOIL] = 1.0
        P[_E0], P[_ZR], P[_K] = tweezer.E0, tweezer.zR, tweezer.k
        P[_WX0], P[_WY0] = tweezer.wx0, tweezer.wy0
        P[_IM_ALPHA] = np.imag(α_eff)
    if sim.gravity:
        P[_GRAVITY] = 1.0
        P[_GX], P[_GY], P[_GZ] = sim.F_gravity

    kicks = np.zeros((len(trap.kicks), 4))
    for index, kick in enumerate(trap.kicks):
        # the same expressions as dynamics_vector, evaluated once
        p = kick.K * trap.gas.pth
        F0 = p / kick.duration
        kicks[index] = kick.start, kick.end, F0 * np.cos(kick.θ), F0 * np.sin(kick.θ)
    return P, kicks


@njit(cache=True)
def _drift(t, X, P, kicks, out):
    """dynamics_vector of FreeSpaceSimulation, written into out (length 6)"""
    x, y, z, px, py, pz = X[0], X[1], X[2], X[3], X[4], X[5]
    m = P[_M]
    if P[_OVERRIDE] != 0.0:
        Ftwx = -m * (P[_OMEGA_X])**2 * x
        Ftwy = -m * (P[_OMEGA_Y])**2 * y
        Ftwz = -m * (P[_OMEGA_Z])**2 * z
    else:
        w = np.sqrt(1 + (z / P[_ZR])**2)
        wx = P[_WX0] * w
        wy = P[_WY0] * w
        wz = P[_WZ0] * w
        ellipse = (x / wx)**2 + (y / wy)**2
        E = (P[_E0] / w) * np.exp(-ellipse)
        U = -(P[_RE_ALPHA] / 2) * E**2
        Ftwx = 0.5 * (4 * x / wx**2) * U
        Ftwy = 0.5 * (4 * y / wy**2) * U
        Ftwz = 0.5 * (4 * z / wz**2) * U * (1 - 2 * ellipse)

    # as in dynamics_vector, the last active kick wins
    F_kick_x = 0.0
    F_kick_y = 0.0
    for index in range(kicks.shape[0]):
        if kicks[index, 0] <= t < kicks[index, 1]:
            F_kick_x = kicks[index, 2]
            F_kick_y = kicks[index, 3]

    out[0] = px / m
    out[1] = py / m
    out[2] = pz / m
    out[3] = Ftwx - P[_GAMMA_X]*px + F_kick_x
    out[4] = Ftwy - P[_GAMMA_Y]*py + F_kick_y
    out[5] = Ftwz - P[_GAMMA_Z]*pz

    if P[_RECOIL] != 0.0:
        zR = P[_ZR]
        w = np.sqrt(1 + (z / zR)**2)
        ellipse = (x / (P[_WX0] * w))**2 + (y / (P[_WY0] * w))**2
        E = (P[_E0] / w) * np.exp(-ellipse)
        phase_gradient = P[_K] - zR / (zR**2 + z**2)
        out[5] += 0.5 * P[_IM_ALPHA] * E**2 * phase_gradient
    if P[_GRAVITY] != 0.0:
        out[3] += P[_GX]
        out[4] += P[_GY]
        out[5] += P[_GZ]


@njit(cache=True)
def _kasdin_chunk(X, start, time_array, dW, G, Δt, c, P, kicks):
    """
    Kasdin RK4 over the steps start .. start + len(dW) - 1, in place on X

    dW: stage noises (n, 4, num_noises), already scaled by Σ; c: the coefficients
    [a21, a31, a32, a41, a42, a43, α1, α2, α3, α4]
    """
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

        _drift(t, Xn, P, kicks, f)
        for i in range(dim):
            g = 0.0
            for j in range(num_noises):
                g += G[i, j] * dW[k, 0, j]
            K1[i] = (f[i] + g)*Δt

        t2 = t + a21*Δt
        for i in range(dim):
            Xs[i] = Xn[i] + a21*K1[i]
        _drift(t2, Xs, P, kicks, f)
        for i in range(dim):
            g = 0.0
            for j in range(num_noises):
                g += G[i, j] * dW[k, 1, j]
            K2[i] = (f[i] + g)*Δt

        t3 = t + (a31 + a32)*Δt
        for i in range(dim):
            Xs[i] = Xn[i] + a31*K1[i] + a32*K2[i]
        _drift(t3, Xs, P, kicks, f)
        for i in range(dim):
            g = 0.0
            for j in range(num_noises):
                g += G[i, j] * dW[k, 2, j]
            K3[i] = (f[i] + g)*Δt

        t4 = t + (a41 + a42 + a43)*Δt
        for i in range(dim):
            Xs[i] = Xn[i] + a41*K1[i] + a42*K2[i] + a43*K3[i]
        _drift(t4, Xs, P, kicks, f)
        for i in range(dim):
            g = 0.0
            for j in range(num_noises):
                g += G[i, j] * dW[k, 3, j]
            K4[i] = (f[i] + g)*Δt

        for i in range(dim):
            X[n + 1, i] = Xn[i] + α1*K1[i] + α2*K2[i] + α3*K3[i] + α4*K4[i]


@njit(cache=True)
def _heun_chunk(X, start, time_array, dP, Δt, P, kicks):
    """Heun over the steps start .. start + len(dP) - 1, in place on X; dP = (normals·σ) Gᵀ per step"""
    dim = X.shape[1]
    f = np.empty(dim)
    K1 = np.empty(dim)
    Xs = np.empty(dim)
    for k in range(dP.shape[0]):
        n = start + k
        t = time_array[n]
        Xn = X[n]
        _drift(t, Xn, P, kicks, f)
        for i in range(dim):
            K1[i] = f[i]*Δt + dP[k, i]
            Xs[i] = Xn[i] + K1[i]
        _drift(t + Δt, Xs, P, kicks, f)
        for i in range(dim):
            X[n + 1, i] = Xn[i] + 0.5 * (K1[i] + (f[i]*Δt + dP[k, i]))


def _mark_applied_kicks(sim, integrator, time_array):
    """
    Sets kick.p, F0, Fx, Fy on every Kick that was active at some evaluated stage time, as the NumPy
    loop does as a side effect of dynamics_vector
    """
    trap = sim.trap
    if not trap.kicks:
        return
    Δt = sim.Δt
    if integrator.name == 'vector_4':
        stage_times = (
            time_array,
            time_array + integrator.a21*Δt,
            time_array + (integrator.a31 + integrator.a32)*Δt,
            time_array + (integrator.a41 + integrator.a42 + integrator.a43)*Δt,
        )
    else:
        stage_times = (time_array, time_array + Δt)
    for kick in trap.kicks:
        if any(np.any((kick.start <= t) & (t < kick.end)) for t in stage_times):
            kick.p = kick.K * trap.gas.pth
            kick.F0 = kick.p / kick.duration
            kick.Fx = kick.F0 * np.cos(kick.θ)
            kick.Fy = kick.F0 * np.sin(kick.θ)


def integrate_free_space(sim, integrator, G, S, X0, time_array):
    """
    The compiled equivalent of integrator.integrate(sim.drift, G, S, X0, time_array, sim.Δt,
    sim._noise_chunks) for a FreeSpaceSimulation; returns the states, shape (len(time_array) + 1, 6)
    """
    Δt = sim.Δt
    num_steps = len(time_array)
    P, kicks = free_space_parameters(sim)
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
            dW = normals * Σ  # exactly as the NumPy loop
            _kasdin_chunk(X, start, time_array, dW, G, Δt, c, P, kicks)
    elif integrator.name == 'vector_2':
        σ = integrator.noise_std(S, Δt)
        for start, normals in sim._noise_chunks(num_steps, integrator.num_stages, len(S)):
            dP = np.ascontiguousarray((normals[:, 0, :] * σ) @ G.T)  # exactly as the NumPy loop
            _heun_chunk(X, start, time_array, dP, Δt, P, kicks)
    else:
        raise ValueError(f'No compiled kernel for the integrator {integrator!r}')
    _mark_applied_kicks(sim, integrator, time_array)
    return X

"""
Compiled (numba) time loops of RigidBodySimulation

The kernels advance the 13-entry state [r, p, q, L_body] with the arithmetic of
`RigidBodySimulation.drift` for a trap of at most one `Tweezer` (any polarisation) plus `Feedback`,
with gas damping, recoil noise and the built-in controls (parametric_feedback, torque_drive,
angular_damping), for both integrators. They take the random numbers from the simulation's own
`_noise_chunks`, scaled as the NumPy loops scale them, and renormalise the quaternion after every
step, so trajectories agree with the NumPy reference to round-off.
"""

import numpy as np
from numba import njit

from levisim.core.constants import ε0, π

__all__ = [
    'integrate_rigid_body',
    'rigid_body_parameters',
]

# indices into the float parameter vector P
_M, _TWEEZER, _E0, _ZR, _WX0, _WY0, _WZ0 = 0, 1, 2, 3, 4, 5, 6
_FB = 7          # 3 feedback rates (lab)
_DRIVE = 10      # flag, offset (3), amplitude (3), ω, phase
_ANG = 19        # flag, 3 angular damping rates (lab)
_RECOIL_C = 23   # k³/(12π ε0) of the tweezer (recoil of the radiated angular momentum)
_FS = 24         # flag: finite-size alignment torque (tensors 3: k, 4: S_body/m, 5-7: generators)
_NUM_PARAMETERS = 25


def rigid_body_parameters(sim):
    """(P, α_body, e, tensors, parametric) arrays of a RigidBodySimulation for the kernels"""
    P = np.zeros(_NUM_PARAMETERS)
    P[_M] = sim.m
    α_body = np.zeros((3, 3), dtype=np.complex128)
    e = np.zeros(3, dtype=np.complex128)
    if sim._optics:
        tweezer, α, polarisation = sim._optics[0]
        P[_TWEEZER] = 1.0
        P[_E0], P[_ZR] = tweezer.E0, tweezer.zR
        P[_WX0], P[_WY0], P[_WZ0] = tweezer.wx0, tweezer.wy0, tweezer.wz0
        α_body[:] = α
        e[:] = polarisation
        P[_RECOIL_C] = tweezer.k**3 / (12 * π * ε0)
    P[_FB:_FB + 3] = sim.Γ_feedback
    if sim.torque_drive is not None:
        d = sim.torque_drive
        P[_DRIVE] = 1.0
        P[_DRIVE + 1:_DRIVE + 4] = d['offset']
        P[_DRIVE + 4:_DRIVE + 7] = d['amplitude']
        P[_DRIVE + 7], P[_DRIVE + 8] = d['ω'], d['phase']
    if sim.angular_damping is not None:
        P[_ANG] = 1.0
        P[_ANG + 1:_ANG + 4] = sim.angular_damping
    if sim.finite_size_torque:
        P[_FS] = 1.0
    from levisim.physics.rotation import GENERATORS
    tensors = np.ascontiguousarray(
        np.stack([sim.inertia_inverse, sim.Γ_cm, sim.Γ_rot, sim._fs_k, sim._fs_S_body, *GENERATORS]), dtype=np.float64,
    )
    parametric = np.ascontiguousarray(sim.parametric_feedback, dtype=np.float64).reshape(-1, 8)
    return P, α_body, e, tensors, parametric


@njit(cache=True)
def _drift(t, X, P, A, e, T, Q, out):
    """RigidBodySimulation.drift, written into out (length 13)"""
    x, y, z = X[0], X[1], X[2]
    qw, qx, qy, qz = X[6], X[7], X[8], X[9]
    norm = np.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
    w, a, b, c = qw / norm, qx / norm, qy / norm, qz / norm
    R = np.empty((3, 3))
    R[0, 0] = 1 - 2*(b*b + c*c)
    R[0, 1] = 2*(a*b - w*c)
    R[0, 2] = 2*(a*c + w*b)
    R[1, 0] = 2*(a*b + w*c)
    R[1, 1] = 1 - 2*(a*a + c*c)
    R[1, 2] = 2*(b*c - w*a)
    R[2, 0] = 2*(a*c - w*b)
    R[2, 1] = 2*(b*c + w*a)
    R[2, 2] = 1 - 2*(a*a + b*b)
    m = P[_M]
    ω = np.zeros(3)
    for i in range(3):
        for j in range(3):
            ω[i] += T[0, i, j] * X[10 + j]
    ω_lab = np.zeros(3)
    L_lab = np.zeros(3)
    p_body = np.zeros(3)
    for i in range(3):
        for j in range(3):
            ω_lab[i] += R[i, j] * ω[j]
            L_lab[i] += R[i, j] * X[10 + j]
            p_body[i] += R[j, i] * X[3 + j]

    factor = 1.0
    for k in range(Q.shape[0]):
        n0 = R[0, 0]*Q[k, 1] + R[0, 1]*Q[k, 2] + R[0, 2]*Q[k, 3]
        n1 = R[1, 0]*Q[k, 1] + R[1, 1]*Q[k, 2] + R[1, 2]*Q[k, 3]
        n2 = R[2, 0]*Q[k, 1] + R[2, 1]*Q[k, 2] + R[2, 2]*Q[k, 3]
        s = Q[k, 4]*n0 + Q[k, 5]*n1 + Q[k, 6]*n2
        c0 = ω_lab[1]*n2 - ω_lab[2]*n1
        c1 = ω_lab[2]*n0 - ω_lab[0]*n2
        c2 = ω_lab[0]*n1 - ω_lab[1]*n0
        ṡ = Q[k, 4]*c0 + Q[k, 5]*c1 + Q[k, 6]*c2
        η = Q[k, 0] * s * ṡ
        η_max = Q[k, 7]
        if η > η_max:
            η = η_max
        elif η < -η_max:
            η = -η_max
        factor += η

    F = np.zeros(3)
    τ_b = np.zeros(3)
    if P[_TWEEZER] != 0.0:
        zR = P[_ZR]
        wz = np.sqrt(1 + (z / zR)**2)
        wx = P[_WX0] * wz
        wy = P[_WY0] * wz
        wzz = P[_WZ0] * wz
        ellipse = (x / wx)**2 + (y / wy)**2
        amplitude = (P[_E0] / wz) * np.exp(-ellipse)
        amplitude2 = factor * amplitude**2
        u = np.zeros(3, dtype=np.complex128)
        for i in range(3):
            for j in range(3):
                u[i] += R[j, i] * e[j]
        v = np.zeros(3, dtype=np.complex128)
        for i in range(3):
            for j in range(3):
                v[i] += A[i, j] * u[j]
        av = 0.0
        for i in range(3):
            av += (np.conj(u[i]) * v[i]).real
        U = -(av / 2) * amplitude2
        F[0] = 0.5 * (4 * x / wx**2) * U
        F[1] = 0.5 * (4 * y / wy**2) * U
        F[2] = 0.5 * (4 * z / wzz**2) * U * (1 - 2 * ellipse)
        vc0, vc1, vc2 = np.conj(v[0]), np.conj(v[1]), np.conj(v[2])
        cr = P[_RECOIL_C]
        τ_b[0] = amplitude2 * (0.5 * (vc1*u[2] - vc2*u[1]).real - cr * (vc1*v[2] - vc2*v[1]).imag)
        τ_b[1] = amplitude2 * (0.5 * (vc2*u[0] - vc0*u[2]).real - cr * (vc2*v[0] - vc0*v[2]).imag)
        τ_b[2] = amplitude2 * (0.5 * (vc0*u[1] - vc1*u[0]).real - cr * (vc0*v[1] - vc1*v[0]).imag)

    if P[_FS] != 0.0:
        # τ_j = -½ Tr(k dS_j), dS_j = G_j S - S G_j, S = R S_body Rᵀ (levisim.physics.rotation)
        S_lab = np.zeros((3, 3))
        for a in range(3):
            for b in range(3):
                acc = 0.0
                for c in range(3):
                    for d in range(3):
                        acc += R[a, c] * T[4, c, d] * R[b, d]
                S_lab[a, b] = acc
        B = np.zeros((3, 3))
        for a in range(3):
            for b in range(3):
                acc = 0.0
                for c in range(3):
                    acc += S_lab[a, c] * T[3, c, b] - T[3, a, c] * S_lab[c, b]
                B[a, b] = acc
        τ_fs = np.zeros(3)
        for j in range(3):
            acc = 0.0
            for a in range(3):
                for b in range(3):
                    acc += T[5 + j, a, b] * B[b, a]
            τ_fs[j] = -0.5 * acc
        for i in range(3):
            for j in range(3):
                τ_b[i] += factor * R[j, i] * τ_fs[j]

    τ_lab = np.zeros(3)
    extra = False
    if P[_DRIVE] != 0.0:
        cosine = np.cos(P[_DRIVE + 7] * t + P[_DRIVE + 8])
        for i in range(3):
            τ_lab[i] += P[_DRIVE + 1 + i] + P[_DRIVE + 4 + i] * cosine
        extra = True
    if P[_ANG] != 0.0:
        for i in range(3):
            τ_lab[i] -= P[_ANG + 1 + i] * L_lab[i]
        extra = True
    if extra:
        for i in range(3):
            for j in range(3):
                τ_b[i] += R[j, i] * τ_lab[j]

    for i in range(3):
        F[i] -= P[_FB + i] * X[3 + i]

    L0, L1, L2 = X[10], X[11], X[12]
    # R Γ_cm Rᵀ p, with Γ_cm in the body frame
    damp_body = np.zeros(3)
    for i in range(3):
        damp_body[i] = T[1, i, 0]*p_body[0] + T[1, i, 1]*p_body[1] + T[1, i, 2]*p_body[2]
    for i in range(3):
        out[i] = X[3 + i] / m
        out[3 + i] = F[i] - (R[i, 0]*damp_body[0] + R[i, 1]*damp_body[1] + R[i, 2]*damp_body[2])
    out[6] = 0.5 * (-qx*ω[0] - qy*ω[1] - qz*ω[2])
    out[7] = 0.5 * (qw*ω[0] + qy*ω[2] - qz*ω[1])
    out[8] = 0.5 * (qw*ω[1] - qx*ω[2] + qz*ω[0])
    out[9] = 0.5 * (qw*ω[2] + qx*ω[1] - qy*ω[0])
    for i in range(3):
        damp = T[2, i, 0]*L0 + T[2, i, 1]*L1 + T[2, i, 2]*L2
        out[10 + i] = τ_b[i] - damp
    out[10] -= ω[1]*L2 - ω[2]*L1
    out[11] -= ω[2]*L0 - ω[0]*L2
    out[12] -= ω[0]*L1 - ω[1]*L0


@njit(cache=True)
def _normalise(X, n):
    norm = np.sqrt(X[n, 6]**2 + X[n, 7]**2 + X[n, 8]**2 + X[n, 9]**2)
    for i in range(6, 10):
        X[n, i] = X[n, i] / norm


@njit(cache=True)
def _kasdin_chunk(X, start, time_array, dW, G, Δt, c, P, A, e, T, Q):
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
        _drift(t, Xn, P, A, e, T, Q, f)
        for i in range(dim):
            g = 0.0
            for j in range(num_noises):
                g += G[i, j] * dW[k, 0, j]
            K1[i] = (f[i] + g)*Δt
        for i in range(dim):
            Xs[i] = Xn[i] + a21*K1[i]
        _drift(t + a21*Δt, Xs, P, A, e, T, Q, f)
        for i in range(dim):
            g = 0.0
            for j in range(num_noises):
                g += G[i, j] * dW[k, 1, j]
            K2[i] = (f[i] + g)*Δt
        for i in range(dim):
            Xs[i] = Xn[i] + a31*K1[i] + a32*K2[i]
        _drift(t + (a31 + a32)*Δt, Xs, P, A, e, T, Q, f)
        for i in range(dim):
            g = 0.0
            for j in range(num_noises):
                g += G[i, j] * dW[k, 2, j]
            K3[i] = (f[i] + g)*Δt
        for i in range(dim):
            Xs[i] = Xn[i] + a41*K1[i] + a42*K2[i] + a43*K3[i]
        _drift(t + (a41 + a42 + a43)*Δt, Xs, P, A, e, T, Q, f)
        for i in range(dim):
            g = 0.0
            for j in range(num_noises):
                g += G[i, j] * dW[k, 3, j]
            K4[i] = (f[i] + g)*Δt
        for i in range(dim):
            X[n + 1, i] = Xn[i] + α1*K1[i] + α2*K2[i] + α3*K3[i] + α4*K4[i]
        _normalise(X, n + 1)


@njit(cache=True)
def _heun_chunk(X, start, time_array, dP, Δt, P, A, e, T, Q):
    dim = X.shape[1]
    f = np.empty(dim)
    K1 = np.empty(dim)
    Xs = np.empty(dim)
    for k in range(dP.shape[0]):
        n = start + k
        t = time_array[n]
        Xn = X[n]
        _drift(t, Xn, P, A, e, T, Q, f)
        for i in range(dim):
            K1[i] = f[i]*Δt + dP[k, i]
            Xs[i] = Xn[i] + K1[i]
        _drift(t + Δt, Xs, P, A, e, T, Q, f)
        for i in range(dim):
            X[n + 1, i] = Xn[i] + 0.5 * (K1[i] + (f[i]*Δt + dP[k, i]))
        _normalise(X, n + 1)


def integrate_rigid_body(sim, integrator, G, S, X0, time_array):
    """
    The compiled equivalent of integrator.integrate(sim.drift, G, S, X0, time_array, sim.Δt,
    sim._noise_chunks, project=sim.project_state); returns the states, shape (len(time_array) + 1, 13)
    """
    Δt = sim.Δt
    num_steps = len(time_array)
    P, A, e, T, Q = rigid_body_parameters(sim)
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
            _kasdin_chunk(X, start, time_array, dW, G, Δt, c, P, A, e, T, Q)
    elif integrator.name == 'vector_2':
        σ = integrator.noise_std(S, Δt)
        for start, normals in sim._noise_chunks(num_steps, integrator.num_stages, len(S)):
            dP = np.ascontiguousarray((normals[:, 0, :] * σ) @ G.T)
            _heun_chunk(X, start, time_array, dP, Δt, P, A, e, T, Q)
    else:
        raise ValueError(f'No compiled kernel for the integrator {integrator!r}')
    return X

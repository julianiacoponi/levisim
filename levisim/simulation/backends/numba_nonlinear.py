"""
Compiled (numba) time loop of the nonlinear CavitySimulation (split step)

The kernel runs exactly the scheme of `CavitySimulation._split_step_numpy`: per sample interval,
an exact linear half step of the augmented state (transition matrix Φ_half and the noise ξ
computed in NumPy per chunk), then `substeps` times a Runge-Kutta step of the nonlinear remainder
followed by an exact linear step (the last one a half step). The remainder is evaluated from the
same per-particle parameter table as the NumPy drift (`NonlinearCavityModel.parameters`, columns
defined in `levisim.simulation.nonlinear_cavity`), so the two agree to round-off (the order of the
sums differs). Configurations with interactions or force elements other than one Tweezer, its
PhotonRecoil, Gravity, HarmonicPotential and Feedback have no kernel (NumPy is used).

Threading: with several trajectories (`choose_threading` says 'setups') each trajectory is advanced
in its own thread; otherwise serially.
"""

import math

import numpy as np
from numba import njit, prange

__all__ = [
    'remainder',
    'split_step_chunk',
]

# column offsets, as in levisim.simulation.nonlinear_cavity
_INV_P = 3
_TW = 6
_REC = 12
_GRAV = 15
_HARM_K = 18
_HARM_KC = 27
_CS = 30


@njit(cache=True, fastmath=False)
def _term_drift(U, P, flags, out):
    """Drift of the flagged terms (elements, coherent scattering, cavity) at absolute quadratures U"""
    n = U.shape[0]
    for i in range(n):
        out[i] = 0.0
    N = P.shape[0]
    α = 0.5 * (U[0] + 1j * U[1])
    dα = 0.0 + 0.0j
    use_elements, use_cs, use_cav = flags[0], flags[1], flags[2]
    for p in range(N):
        base = 2 + 6 * p
        x = U[base] * P[p, 0]
        y = U[base + 2] * P[p, 1]
        z = U[base + 4] * P[p, 2]
        Fx = 0.0
        Fy = 0.0
        Fz = 0.0
        if use_elements:
            if P[p, _TW + 5] > 0:
                aRe = P[p, _TW]
                E0 = P[p, _TW + 1]
                zR = P[p, _TW + 4]
                w = math.sqrt(1 + (z / zR) ** 2)
                wx = P[p, _TW + 2] * w
                wy = P[p, _TW + 3] * w
                wz = math.sqrt(2.0) * zR * w
                ellipse = (x / wx) ** 2 + (y / wy) ** 2
                E = (E0 / w) * math.exp(-ellipse)
                Utw = -(aRe / 2) * E ** 2
                Fx += 0.5 * (4 * x / wx ** 2) * Utw
                Fy += 0.5 * (4 * y / wy ** 2) * Utw
                Fz += 0.5 * (4 * z / wz ** 2) * Utw * (1 - 2 * ellipse)
                if P[p, _REC + 2] > 0:
                    Fz += P[p, _REC] * E ** 2 * (P[p, _REC + 1] - zR / (zR ** 2 + z ** 2))
            Fx += P[p, _GRAV]
            Fy += P[p, _GRAV + 1]
            Fz += P[p, _GRAV + 2]
            Fx += -(P[p, _HARM_K] * x + P[p, _HARM_K + 1] * y + P[p, _HARM_K + 2] * z) + P[p, _HARM_KC]
            Fy += -(P[p, _HARM_K + 3] * x + P[p, _HARM_K + 4] * y + P[p, _HARM_K + 5] * z) + P[p, _HARM_KC + 1]
            Fz += -(P[p, _HARM_K + 6] * x + P[p, _HARM_K + 7] * y + P[p, _HARM_K + 8] * z) + P[p, _HARM_KC + 2]
        if use_cs or use_cav:
            s = P[p, _CS]
            c = P[p, _CS + 1]
            kk = P[p, _CS + 2]
            ph = kk * (s * x + c * y) + P[p, _CS + 3]
            cos_u = math.cos(ph)
            sin_u = math.sin(ph)
            if use_cs:
                wx0 = P[p, _CS + 4]
                wy0 = P[p, _CS + 5]
                zR = P[p, _CS + 6]
                w2 = 1 + (z / zR) ** 2
                e = (x ** 2 / wx0 ** 2 + y ** 2 / wy0 ** 2) / w2
                f = math.exp(-e) / math.sqrt(w2)
                ψ = kk * z - math.atan(z / zR)
                phase = math.cos(ψ) + 1j * math.sin(ψ)
                D = s * f * cos_u * phase
                dD_sin = s * f * sin_u * phase * kk
                gx = D * (-2 * x / (wx0 ** 2 * w2)) - dD_sin * s
                gy = D * (-2 * y / (wy0 ** 2 * w2)) - dD_sin * c
                gz = D * ((z / zR ** 2) / w2 * (2 * e - 1)) + 1j * (kk - zR / (zR ** 2 + z ** 2)) * D
                e_iφ = P[p, _CS + 9] + 1j * P[p, _CS + 10]
                αn_conj = (α.real - 1j * α.imag) * e_iφ
                two_Ed = P[p, _CS + 8]
                Fx += two_Ed * (αn_conj * gx).real
                Fy += two_Ed * (αn_conj * gy).real
                Fz += two_Ed * (αn_conj * gz).real
                dα += 1j * P[p, _CS + 7] * e_iφ * D
            if use_cav:
                Fcav = -P[p, _CS + 12] * (α.real ** 2 + α.imag ** 2) * kk * 2 * sin_u * cos_u
                Fx += Fcav * s
                Fy += Fcav * c
                dα += 1j * α * P[p, _CS + 11] * cos_u ** 2
        out[base + 1] = Fx * P[p, _INV_P]
        out[base + 3] = Fy * P[p, _INV_P + 1]
        out[base + 5] = Fz * P[p, _INV_P + 2]
    out[0] = 2 * dα.real
    out[1] = 2 * dα.imag


@njit(cache=True, fastmath=False)
def remainder(u, U_eq, P, flags, F0, J, out, work):
    """
    R(u) = drift_nl(U_eq + u) - F0 - J·u for one state u (n,), with J = (indptr, indices, data) in
    CSR form (it couples each particle only to itself and to the cavity); `work` is a scratch (n,)
    """
    indptr, indices, data = J
    n = u.shape[0]
    for i in range(n):
        work[i] = U_eq[i] + u[i]
    _term_drift(work, P, flags, out)
    for i in range(n):
        s = 0.0
        for jj in range(indptr[i], indptr[i + 1]):
            s += data[jj] * u[indices[jj]]
        out[i] -= F0[i] + s


@njit(cache=True, fastmath=False)
def _rk_step(u, h, U_eq, P, flags, F0, J, rk4, k1, k2, k3, k4, tmp, work):
    n = u.shape[0]
    remainder(u, U_eq, P, flags, F0, J, k1, work)
    if not rk4:
        for i in range(n):
            tmp[i] = u[i] + 0.5 * h * k1[i]
        remainder(tmp, U_eq, P, flags, F0, J, k2, work)
        for i in range(n):
            u[i] += h * k2[i]
        return
    for i in range(n):
        tmp[i] = u[i] + 0.5 * h * k1[i]
    remainder(tmp, U_eq, P, flags, F0, J, k2, work)
    for i in range(n):
        tmp[i] = u[i] + 0.5 * h * k2[i]
    remainder(tmp, U_eq, P, flags, F0, J, k3, work)
    for i in range(n):
        tmp[i] = u[i] + h * k3[i]
    remainder(tmp, U_eq, P, flags, F0, J, k4, work)
    for i in range(n):
        u[i] += (h / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i])


@njit(cache=True, fastmath=False)
def _matvec(Φ, z, out):
    m = z.shape[0]
    for c in range(m):
        s = 0.0
        for i in range(m):
            s += Φ[c, i] * z[i]
        out[c] = s


@njit(cache=True, fastmath=False)
def _advance_one(Φh, Φf, ξ, u, Y, states, store, j, M, h, U_eq, P, flags, F0, J, rk4, has_nl):
    nn = ξ.shape[0]
    n = u.shape[1]
    m = Φh.shape[0]
    z = np.zeros(m)
    z2 = np.empty(m)
    uj = np.empty(n)
    k1 = np.empty(n)
    k2 = np.empty(n)
    k3 = np.empty(n)
    k4 = np.empty(n)
    tmp = np.empty(n)
    work = np.empty(n)
    for k in range(nn):
        for i in range(n):
            z[i] = u[j, i]
        for i in range(n, m):
            z[i] = 0.0
        _matvec(Φh, z, z2)
        for i in range(m):
            z[i] = z2[i] + ξ[k, 0, j, i]
        for step in range(M):
            if has_nl:
                for i in range(n):
                    uj[i] = z[i]
                _rk_step(uj, h, U_eq, P, flags, F0, J, rk4, k1, k2, k3, k4, tmp, work)
                for i in range(n):
                    z[i] = uj[i]
            if step < M - 1:
                _matvec(Φf, z, z2)
            else:
                _matvec(Φh, z, z2)
            for i in range(m):
                z[i] = z2[i] + ξ[k, step + 1, j, i]
        for i in range(n):
            u[j, i] = z[i]
            if store:
                states[k, j, i] = z[i]
        for c in range(m - n):
            Y[k, j, c] = z[n + c]


@njit(cache=True)
def _serial(Φh, Φf, ξ, u, Y, states, store, M, h, U_eq, P, flags, F0, J, rk4, has_nl):
    for j in range(u.shape[0]):
        _advance_one(Φh, Φf, ξ, u, Y, states, store, j, M, h, U_eq, P, flags, F0, J, rk4, has_nl)


@njit(cache=True, parallel=True)
def _parallel(Φh, Φf, ξ, u, Y, states, store, M, h, U_eq, P, flags, F0, J, rk4, has_nl):
    for j in prange(u.shape[0]):
        _advance_one(Φh, Φf, ξ, u, Y, states, store, j, M, h, U_eq, P, flags, F0, J, rk4, has_nl)


def _csr(J):
    """(indptr, indices, data) of the non-zero entries of a dense matrix, row by row"""
    rows, cols = np.nonzero(J)
    indptr = np.zeros(J.shape[0] + 1, dtype=np.int64)
    np.add.at(indptr, rows + 1, 1)
    return np.cumsum(indptr), cols.astype(np.int64), np.ascontiguousarray(J[rows, cols], dtype=np.float64)


def kernel_arguments(sim):
    """The arrays the kernel reads from a nonlinear CavitySimulation"""
    model = sim.nonlinear_model
    flags = np.array([t in model.nonlinear_terms for t in ('elements', 'coherent_scattering', 'cavity')], dtype=np.bool_)
    return {
        'Φh': np.ascontiguousarray(sim.split['Φ_half']),
        'Φf': np.ascontiguousarray(sim.split['Φ_full']),
        'M': int(sim.substeps), 'h': float(sim.split['h']),
        'U_eq': np.ascontiguousarray(model.u_eq, dtype=np.float64),
        'P': np.ascontiguousarray(model.parameters, dtype=np.float64),
        'flags': flags,
        'F0': np.ascontiguousarray(model.nonlinear_drift_at_equilibrium, dtype=np.float64),
        'J': _csr(model.nonlinear_jacobian),
        'rk4': sim.nonlinear_stage == 'rk4',
        'has_nl': bool(model.nonlinear_terms),
    }


def split_step_chunk(sim, ξ, u, Y, states, threaded):
    """
    Advances K trajectories over one chunk of samples, in place

    ξ: (nn, substeps + 1, K, n + q) noise of the linear flows; u: (K, n) deviations (the returned
    array holds the final state); Y: (nn, K, q) output of the records' integrals; states:
    (nn, K, n) or None; threaded: thread over the trajectories.
    """
    a = kernel_arguments(sim)
    u = np.ascontiguousarray(u, dtype=np.float64).copy()
    ξ = np.ascontiguousarray(ξ)
    store = states is not None
    buffer = states if store else np.empty((0, 0, 0))
    Yc = np.empty(Y.shape)
    run = _parallel if threaded else _serial
    run(a['Φh'], a['Φf'], ξ, u, Yc, buffer, store, a['M'], a['h'], a['U_eq'], a['P'], a['flags'], a['F0'], a['J'],
        a['rk4'], a['has_nl'])
    Y[...] = Yc
    return u

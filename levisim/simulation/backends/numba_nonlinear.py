"""
Compiled (numba) time loop of the nonlinear CavitySimulation (split step)

The kernel runs exactly the scheme of `CavitySimulation._split_step_numpy`: per sample interval,
an exact linear half step of the augmented state (transition matrix Φ_half and the noise ξ
computed in NumPy per chunk), then `substeps` times a Runge-Kutta step of the nonlinear remainder
followed by an exact linear step (the last one a half step). The remainder is evaluated from the
same per-particle parameter table as the NumPy drift (`NonlinearCavityModel.parameters`, columns
defined in `levisim.simulation.nonlinear_cavity`), so the two agree to round-off (the order of the
sums differs). Besides one Tweezer, its PhotonRecoil, Gravity, HarmonicPotential and Feedback, the
kernel evaluates the static elements of `numba_elements` (MagneticTrap with polynomial, current-loop,
cuboid or summed sources, UniformElectricField) and the interactions: power-law pair potentials
(Coulomb, NewtonianGravity, Casimir) and one OpticalBinding (`numba_binding`), at the laboratory
positions trap_positions + r. Other configurations (a PaulTrap, whose force depends on time, other
elements or interactions) have no kernel (NumPy is used).

Threading: with several trajectories (`choose_threading` says 'setups') each trajectory is advanced
in its own thread; otherwise serially.
"""

import math

import numpy as np
from numba import njit, prange

from levisim.simulation.backends.numba_binding import binding_forces
from levisim.simulation.backends.numba_elements import field_force

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


@njit(cache=True, fastmath=False, error_model='numpy')
def _interaction_forces(U, P, extra):
    """The pair forces (N, 3) at the laboratory positions trap_positions + r of the quadratures U"""
    origins, terms, use_binding, meta, real, cplx = extra[4], extra[5], extra[6], extra[7], extra[8], extra[9]
    N = P.shape[0]
    positions = np.empty((N, 3))
    for p in range(N):
        base = 2 + 6 * p
        for a in range(3):
            positions[p, a] = origins[p, a] + U[base + 2*a] * P[p, a]
    F = np.zeros((N, 3))
    for row in range(terms.shape[0]):
        j = int(terms[row, 0])
        k = int(terms[row, 1])
        C = terms[row, 2]
        s = terms[row, 3]
        power = terms[row, 4]
        dx = positions[j, 0] - positions[k, 0]
        dy = positions[j, 1] - positions[k, 1]
        dz = positions[j, 2] - positions[k, 2]
        r = math.sqrt(dx*dx + dy*dy + dz*dz)
        d = r - s
        Upair = C * d**(-power)
        dU = -power * Upair / d
        fx = -dU * dx / r
        fy = -dU * dy / r
        fz = -dU * dz / r
        F[j, 0] += fx
        F[j, 1] += fy
        F[j, 2] += fz
        F[k, 0] -= fx
        F[k, 1] -= fy
        F[k, 2] -= fz
    if use_binding:
        binding_forces(positions, meta, real, cplx, F)
    return F


@njit(cache=True, fastmath=False, error_model='numpy')
def _term_drift(U, P, flags, extra, out):
    """
    Drift of the flagged terms (elements, coherent scattering, cavity, interactions) at absolute
    quadratures U
    """
    n = U.shape[0]
    for i in range(n):
        out[i] = 0.0
    N = P.shape[0]
    α = 0.5 * (U[0] + 1j * U[1])
    dα = 0.0 + 0.0j
    use_elements, use_cs, use_cav, use_int = flags[0], flags[1], flags[2], flags[3]
    ranges, rows, sources, data = extra[0], extra[1], extra[2], extra[3]
    if use_int:
        F_int = _interaction_forces(U, P, extra)
    else:
        F_int = np.zeros((0, 3))
    element_force = np.empty(3)
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
                # Utw here is TWICE the cycle-averaged potential -(Re α/4)|E|² (the 0.x convention of
                # Trap.Utw); the factors 0.5 below make the force exactly -∇[-(Re α/4)|E|²], the
                # tweezer stiffness of Trap (checked against Trap.kx/ky/kz and the MultiTrap drift matrix)
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
            if ranges[p, 1] > ranges[p, 0]:
                # the static elements of the program (their forces do not depend on time)
                field_force(0.0, x, y, z, ranges[p, 0], ranges[p, 1], rows, sources, data, element_force)
                Fx += element_force[0]
                Fy += element_force[1]
                Fz += element_force[2]
        if use_cs or use_cav:
            s = P[p, _CS]
            c = P[p, _CS + 1]
            kk = P[p, _CS + 2]
            ph = kk * (s * x + c * y) + P[p, _CS + 3]
            cos_u = math.cos(ph)
            sin_u = math.sin(ph)
            inv_w2 = P[p, _CS + 13]
            env = 1.0
            glx = 0.0
            gly = 0.0
            glz = 0.0
            if inv_w2 > 0:
                along = s * x + c * y
                tx = x - along * s
                ty = y - along * c
                env = math.exp(-(tx ** 2 + ty ** 2 + z ** 2) * inv_w2)
                glx = -2 * tx * inv_w2
                gly = -2 * ty * inv_w2
                glz = -2 * z * inv_w2
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
                if inv_w2 > 0:
                    gx = env * (gx + D * glx)
                    gy = env * (gy + D * gly)
                    gz = env * (gz + D * glz)
                    D = D * env
                e_iφ = P[p, _CS + 9] + 1j * P[p, _CS + 10]
                αn_conj = (α.real - 1j * α.imag) * e_iφ
                two_Ed = P[p, _CS + 8]
                Fx += two_Ed * (αn_conj * gx).real
                Fy += two_Ed * (αn_conj * gy).real
                Fz += two_Ed * (αn_conj * gz).real
                dα += 1j * P[p, _CS + 7] * e_iφ * D
            if use_cav:
                Fcav = -P[p, _CS + 12] * (α.real ** 2 + α.imag ** 2) * kk * 2 * sin_u * cos_u
                if inv_w2 > 0:
                    intensity = P[p, _CS + 12] * (α.real ** 2 + α.imag ** 2) * env ** 2
                    Fx += env ** 2 * Fcav * s + intensity * 2 * cos_u ** 2 * glx
                    Fy += env ** 2 * Fcav * c + intensity * 2 * cos_u ** 2 * gly
                    Fz += intensity * 2 * cos_u ** 2 * glz
                    dα += 1j * α * P[p, _CS + 11] * cos_u ** 2 * env ** 2
                else:
                    Fx += Fcav * s
                    Fy += Fcav * c
                    dα += 1j * α * P[p, _CS + 11] * cos_u ** 2
        if use_int:
            Fx += F_int[p, 0]
            Fy += F_int[p, 1]
            Fz += F_int[p, 2]
        out[base + 1] = Fx * P[p, _INV_P]
        out[base + 3] = Fy * P[p, _INV_P + 1]
        out[base + 5] = Fz * P[p, _INV_P + 2]
    out[0] = 2 * dα.real
    out[1] = 2 * dα.imag


@njit(cache=True, fastmath=False)
def remainder(u, U_eq, P, flags, extra, F0, J, out, work):
    """
    R(u) = drift_nl(U_eq + u) - F0 - J·u for one state u (n,), with J = (indptr, indices, data) in
    CSR form (it couples each particle only to itself and to the cavity); `work` is a scratch (n,)
    """
    indptr, indices, data = J
    n = u.shape[0]
    for i in range(n):
        work[i] = U_eq[i] + u[i]
    _term_drift(work, P, flags, extra, out)
    for i in range(n):
        s = 0.0
        for jj in range(indptr[i], indptr[i + 1]):
            s += data[jj] * u[indices[jj]]
        out[i] -= F0[i] + s


@njit(cache=True, fastmath=False)
def _rk_step(u, h, U_eq, P, flags, extra, F0, J, rk4, k1, k2, k3, k4, tmp, work):
    n = u.shape[0]
    remainder(u, U_eq, P, flags, extra, F0, J, k1, work)
    if not rk4:
        for i in range(n):
            tmp[i] = u[i] + 0.5 * h * k1[i]
        remainder(tmp, U_eq, P, flags, extra, F0, J, k2, work)
        for i in range(n):
            u[i] += h * k2[i]
        return
    for i in range(n):
        tmp[i] = u[i] + 0.5 * h * k1[i]
    remainder(tmp, U_eq, P, flags, extra, F0, J, k2, work)
    for i in range(n):
        tmp[i] = u[i] + 0.5 * h * k2[i]
    remainder(tmp, U_eq, P, flags, extra, F0, J, k3, work)
    for i in range(n):
        tmp[i] = u[i] + h * k3[i]
    remainder(tmp, U_eq, P, flags, extra, F0, J, k4, work)
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
def _advance_one(Φh, Φf, ξ, u, Y, states, store, j, M, h, U_eq, P, flags, extra, F0, J, rk4, has_nl):
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
                _rk_step(uj, h, U_eq, P, flags, extra, F0, J, rk4, k1, k2, k3, k4, tmp, work)
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
def _serial(Φh, Φf, ξ, u, Y, states, store, M, h, U_eq, P, flags, extra, F0, J, rk4, has_nl):
    for j in range(u.shape[0]):
        _advance_one(Φh, Φf, ξ, u, Y, states, store, j, M, h, U_eq, P, flags, extra, F0, J, rk4, has_nl)


@njit(cache=True, parallel=True)
def _parallel(Φh, Φf, ξ, u, Y, states, store, M, h, U_eq, P, flags, extra, F0, J, rk4, has_nl):
    for j in prange(u.shape[0]):
        _advance_one(Φh, Φf, ξ, u, Y, states, store, j, M, h, U_eq, P, flags, extra, F0, J, rk4, has_nl)


def _csr(J):
    """(indptr, indices, data) of the non-zero entries of a dense matrix, row by row"""
    rows, cols = np.nonzero(J)
    indptr = np.zeros(J.shape[0] + 1, dtype=np.int64)
    np.add.at(indptr, rows + 1, 1)
    return np.cumsum(indptr), cols.astype(np.int64), np.ascontiguousarray(J[rows, cols], dtype=np.float64)


def kernel_arguments(sim):
    """The arrays the kernel reads from a nonlinear CavitySimulation"""
    model = sim.nonlinear_model
    flags = np.array([t in model.nonlinear_terms for t in ('elements', 'coherent_scattering', 'cavity', 'interactions')],
                     dtype=np.bool_)
    return {
        'Φh': np.ascontiguousarray(sim.split['Φ_half']),
        'Φf': np.ascontiguousarray(sim.split['Φ_full']),
        'M': int(sim.substeps), 'h': float(sim.split['h']),
        'U_eq': np.ascontiguousarray(model.u_eq, dtype=np.float64),
        'P': np.ascontiguousarray(model.parameters, dtype=np.float64),
        'flags': flags,
        'extra': model.compiled_extra_arguments(),
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
    run(a['Φh'], a['Φf'], ξ, u, Yc, buffer, store, a['M'], a['h'], a['U_eq'], a['P'], a['flags'], a['extra'], a['F0'],
        a['J'], a['rk4'], a['has_nl'])
    Y[...] = Yc
    return u

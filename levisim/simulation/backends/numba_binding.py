"""
Compiled (numba) optical binding forces (`OpticalBinding`, orders 'first' and 'self_consistent',
Gaussian or plane-wave tweezer fields, any tweezer polarisation)

`encode_binding` packs one OpticalBinding once per run: the interacting pairs, and per particle the
tweezer (wavenumber, waists, Rayleigh range, focus, complex amplitude E0·e^{iφ}·ê with the
tweezer's Jones vector ê), the lab-frame polarisability tensor and the coefficients of the radial
functions of the dipole Green tensor (`optical_binding._radial_coefficients`, `_derivative`).
`binding_forces` evaluates the expressions of `OpticalBinding.forces` from these, so the two agree to
round-off (≲ 1e-13 relative; the complex exp and log, and the order of the sums, differ). The
polarisabilities and the pairs are fixed for the run (the particles' orientations do not change in
the translational simulations that use this).

Layout::

    meta (2 + 2·num_pairs,) int64: [order (0 first, 1 self_consistent), model (0 gaussian,
        1 plane_wave), n0, m0, n1, m1, ...]
    real (N, 8) float64: [k, zR, wx0², wy0², focus x, y, z, 0]
    cplx (N, 26) complex128: [amplitude (3), α (9, row-major), f (3), h (3), f' (4), h' (4)]
"""

import cmath
import math

import numpy as np
from numba import njit

__all__ = [
    'binding_forces',
    'encode_binding',
]


def encode_binding(binding, particles):
    """(meta, real, cplx) of an OpticalBinding whose traps are attached, for the given particles"""
    from levisim.physics.interactions.optical_binding import (
        COULOMB_FACTOR,
        _derivative,
        _radial_coefficients,
    )
    N = len(particles)
    binding._check_ready(N)
    pairs = binding.interacting_pairs(N)
    meta = [0 if binding.order == 'first' else 1, 0 if binding.tweezer_field == 'gaussian' else 1]
    for n, m in pairs:
        meta.extend([n, m])
    real = np.zeros((N, 8))
    cplx = np.zeros((N, 26), dtype=complex)
    for n in range(N):
        tweezer = binding.tweezers[n]
        k = tweezer.k
        real[n, :7] = [k, tweezer.zR, tweezer.wx0**2, tweezer.wy0**2, *binding.foci[n]]
        amplitude = tweezer.E0 * np.exp(1j * tweezer.φ) * binding.polarisation_of(n)
        f, h = _radial_coefficients(k)
        f1, h1 = _derivative(f, k), _derivative(h, k)
        # the powers are -1, -2, -3 (f), -3, -4, -5 (h), -1..-4 (f'), -3..-6 (h'), in this order
        assert list(f) == [-1, -2, -3] and list(h) == [-3, -4, -5]
        assert list(f1) == [-1, -2, -3, -4] and list(h1) == [-3, -4, -5, -6]
        cplx[n] = np.concatenate([
            amplitude, np.ravel(binding.polarisability(n, particles[n])),
            list(f.values()), list(h.values()), list(f1.values()), list(h1.values()),
        ])
    real[:, 7] = COULOMB_FACTOR
    return np.array(meta, dtype=np.int64), np.ascontiguousarray(real), np.ascontiguousarray(cplx)


@njit(cache=True, error_model='numpy')
def _tweezer(n, s, real, cplx, model, u, du):
    """tweezer_field (order 1) of particle n at the displacement s from its focus: u (3,), du[a, i]"""
    k = real[n, 0]
    x = s[0]
    y = s[1]
    z = s[2]
    if model == 1:
        S = 1j * k * z
        dS0 = 0.0 + 0.0j
        dS1 = 0.0 + 0.0j
        dS2 = 1j * k
    else:
        zR = real[n, 1]
        a2 = real[n, 2]
        b2 = real[n, 3]
        iz = 1j / zR
        q = 1 + iz * z
        Q = x**2 / a2 + y**2 / b2
        S = 1j * k * z - cmath.log(q) - Q / q
        dS0 = -2 * x / (a2 * q)
        dS1 = -2 * y / (b2 * q)
        dS2 = 1j * k - iz / q + Q * iz / q**2
    e = cmath.exp(S)
    for i in range(3):
        u[i] = cplx[n, i] * e
    for i in range(3):
        du[0, i] = dS0 * u[i]
        du[1, i] = dS1 * u[i]
        du[2, i] = dS2 * u[i]


@njit(cache=True, error_model='numpy')
def _green(n, xv, real, cplx, order, G, dG):
    """dipole_green_tensor_derivatives at the displacement xv with the coefficients of particle n"""
    k = real[n, 0]
    r = math.sqrt(xv[0]**2 + xv[1]**2 + xv[2]**2)
    phase = real[n, 7] * cmath.exp(1j * k * r)
    f = 0.0 + 0.0j
    for p in range(3):
        f += cplx[n, 12 + p] * r ** float(-1 - p)
    f = phase * f
    h = 0.0 + 0.0j
    for p in range(3):
        h += cplx[n, 15 + p] * r ** float(-3 - p)
    h = phase * h
    for i in range(3):
        for j in range(3):
            G[i, j] = h * xv[i] * xv[j]
        G[i, i] += f
    if order < 1:
        return
    f1 = 0.0 + 0.0j
    for p in range(4):
        f1 += cplx[n, 18 + p] * r ** float(-1 - p)
    f1 = phase * f1
    h1 = 0.0 + 0.0j
    for p in range(4):
        h1 += cplx[n, 22 + p] * r ** float(-3 - p)
    h1 = phase * h1
    for c in range(3):
        nc = xv[c] / r
        for i in range(3):
            for j in range(3):
                value = h1 * nc * xv[i] * xv[j]
                if i == j:
                    value += f1 * nc
                if c == i:
                    value += h * xv[j]
                if c == j:
                    value += h * xv[i]
                dG[c, i, j] = value


@njit(cache=True, error_model='numpy')
def _alpha_dot(cplx, n, v, out):
    """out = α_n · v"""
    for i in range(3):
        acc = 0.0 + 0.0j
        for j in range(3):
            acc += cplx[n, 3 + 3*i + j] * v[j]
        out[i] = acc


@njit(cache=True, error_model='numpy')
def binding_forces(positions, meta, real, cplx, F):
    """OpticalBinding.forces at the lab-frame positions (N, 3), added into F (N, 3)"""
    N = positions.shape[0]
    num_pairs = (meta.shape[0] - 2) // 2
    if num_pairs == 0:
        return
    order = meta[0]
    model = meta[1]
    u = np.empty((N, 3), dtype=np.complex128)
    du = np.empty((N, 3, 3), dtype=np.complex128)
    p0 = np.empty((N, 3), dtype=np.complex128)
    s = np.empty(3)
    for n in range(N):
        for i in range(3):
            s[i] = positions[n, i] - real[n, 4 + i]
        _tweezer(n, s, real, cplx, model, u[n], du[n])
        _alpha_dot(cplx, n, u[n], p0[n])
    G = np.empty((3, 3), dtype=np.complex128)
    dG = np.empty((3, 3, 3), dtype=np.complex128)
    xv = np.empty(3)
    v = np.empty(3, dtype=np.complex128)
    w = np.empty(3, dtype=np.complex128)
    if order == 0:
        for q in range(num_pairs):
            n = meta[2 + 2*q]
            m = meta[3 + 2*q]
            for i in range(3):
                xv[i] = positions[n, i] - positions[m, i]
            _green(n, xv, real, cplx, 1, G, dG)
            for side in range(2):
                a = n if side == 0 else m
                b = m if side == 0 else n
                sign = 1.0 if side == 0 else -1.0
                # α_a·𝔊·p0_b
                for i in range(3):
                    acc = 0.0 + 0.0j
                    for j in range(3):
                        acc += G[i, j] * p0[b, j]
                    v[i] = acc
                _alpha_dot(cplx, a, v, w)
                for c in range(3):
                    first = 0.0 + 0.0j
                    for i in range(3):
                        inner = 0.0 + 0.0j
                        for j in range(3):
                            inner += (sign * dG[c, i, j]) * p0[b, j]
                        first += p0[a, i].conjugate() * inner
                    second = 0.0 + 0.0j
                    for i in range(3):
                        second += du[a, c, i].conjugate() * w[i]
                    F[a, c] += 0.5 * (first + second).real
        return
    # self-consistent: (𝟙 - α𝔊) p = α u
    M = np.zeros((3 * N, 3 * N), dtype=np.complex128)
    for i in range(3 * N):
        M[i, i] = 1.0
    for q in range(num_pairs):
        n = meta[2 + 2*q]
        m = meta[3 + 2*q]
        for i in range(3):
            xv[i] = positions[n, i] - positions[m, i]
        _green(n, xv, real, cplx, 0, G, dG)
        for side in range(2):
            a = n if side == 0 else m
            b = m if side == 0 else n
            for i in range(3):
                for j in range(3):
                    acc = 0.0 + 0.0j
                    for l in range(3):
                        acc += cplx[a, 3 + 3*i + l] * G[l, j]
                    M[3*a + i, 3*b + j] -= acc
    source = np.empty(3 * N, dtype=np.complex128)
    for n in range(N):
        for i in range(3):
            source[3*n + i] = p0[n, i]
    p = np.linalg.solve(M, source)
    gradient = du.copy()
    for q in range(num_pairs):
        n = meta[2 + 2*q]
        m = meta[3 + 2*q]
        for i in range(3):
            xv[i] = positions[n, i] - positions[m, i]
        _green(n, xv, real, cplx, 1, G, dG)
        for c in range(3):
            for i in range(3):
                to_n = 0.0 + 0.0j
                to_m = 0.0 + 0.0j
                for j in range(3):
                    to_n += dG[c, i, j] * p[3*m + j]
                    to_m += dG[c, i, j] * p[3*n + j]
                gradient[n, c, i] += to_n
                gradient[m, c, i] -= to_m
    for n in range(N):
        for c in range(3):
            total = 0.0 + 0.0j
            own = 0.0 + 0.0j
            for i in range(3):
                total += gradient[n, c, i] * p[3*n + i].conjugate()
                own += du[n, c, i] * p0[n, i].conjugate()
            F[n, c] += 0.5 * total.real - 0.5 * own.real

"""
Time-domain (covariance matrix) description of a `MultiTrap`

The rest of levisim solves the quantum Langevin equations in the frequency domain to get
spectra. Entanglement is a property of the STATE, i.e. of equal-time second moments, so here the
same drift and noise matrices are used in the Lyapunov equation instead:

    dV/dt = A·V + V·Aᵀ + D        steady state:  A·V + V·Aᵀ + D = 0

with A the drift matrix and D the diffusion matrix, both in the real quadrature basis
u = (A, P, X_x1, P_x1, X_y1, P_y1, ...) of levisim's operator ordering c = (a, a†, b_x1, b_x1†, ...),
where for every mode X = b + b† and P = i(b† - b). The covariance V_ij = ½〈{Δu_i, Δu_j}〉 then
has the vacuum as the identity, which is the convention of `levisim.analysis.gaussian_states`.

The noise is levisim's: each operator pair (c_k, c_k†) with damping γ_k and occupation n_k
(the factors in `MultiTrap.quantum_factor_per_noise_operator`) contributes γ_k(2n_k + 1) to the
diffusion of both of its quadratures.
"""
import numpy as np

from scipy.linalg import solve_continuous_lyapunov


__all__ = [
    'complex_to_quadrature_transform',
    'quadrature_drift_matrix',
    'quadrature_diffusion_matrix',
    'steady_state_covariance',
    'mode_index',
    'dimension_covariance',
]


# maps one (c, c†) pair onto its quadratures (X, P) = (c + c†, i(c† - c))
_PAIR_TO_QUADRATURES = np.array([[1, 1], [-1j, 1j]])


def complex_to_quadrature_transform(num_operators):
    """
    Matrix T with u = T @ c, from levisim's operator ordering c = (a, a†, b_x1, b_x1†, ...)
    """
    assert num_operators % 2 == 0, f'{num_operators=} should be even (operators come in pairs)'
    return np.kron(np.eye(num_operators // 2), _PAIR_TO_QUADRATURES)


def quadrature_drift_matrix(complex_drift_matrix):
    """
    Drift matrix in the quadrature basis, T·A·T⁻¹, which is real for any Hermitian Hamiltonian
    """
    A = np.asarray(complex_drift_matrix)
    T = complex_to_quadrature_transform(A.shape[0])
    A_quadratures = T @ A @ np.linalg.inv(T)
    scale = max(1.0, np.abs(A_quadratures).max())
    assert np.abs(A_quadratures.imag).max() <= 1e-9 * scale, 'drift matrix is not real in quadratures'
    return A_quadratures.real


def quadrature_diffusion_matrix(mtrap):
    """
    Diffusion matrix D for the Lyapunov equation, from the MultiTrap's dampings and occupations

    D = diag(γ_k(2n_k + 1)) on both quadratures of every mode, with 2n_k + 1 being the sum of
    the pair of factors (n_k + 1, n_k) levisim keeps per operator pair.
    [2π·Hz]
    """
    γ = np.asarray(mtrap.γ, dtype=float)
    factors = np.asarray(mtrap.quantum_factor_per_noise_operator, dtype=float)
    per_mode = [γ[2*k] * (factors[2*k] + factors[2*k + 1]) for k in range(len(γ) // 2)]
    return np.diag(np.repeat(per_mode, 2))


def steady_state_covariance(mtrap):
    """
    Steady-state covariance matrix of the MultiTrap's single-tone Linear Amplifier Model

    Solves A·V + V·Aᵀ + D = 0 with levisim's own (time-independent, tweezer-frame) drift matrix.
    Raises if the drift matrix is unstable, because then no steady state exists.
    """
    A = quadrature_drift_matrix(mtrap.drift_matrix)
    largest_growth_rate = np.max(np.linalg.eigvals(A).real)
    if largest_growth_rate >= 0:
        raise ValueError(f'{mtrap} is unstable (largest Re eigenvalue {largest_growth_rate:.3e}), so it has no steady state')
    return solve_continuous_lyapunov(A, -quadrature_diffusion_matrix(mtrap))


def mode_index(mtrap, label):
    """
    Index of a mode in the quadrature covariance matrix: 0 for the cavity, then the dimensions
    """
    if label in ('cavity', 'a', 'A'):
        return 0
    return 1 + mtrap.label_to_index[label]


def dimension_covariance(mtrap, V, labels):
    """
    Covariance matrix of the given dimensions (e.g. ['x1', 'y1']), in the order given
    """
    indices = np.ravel([[2*mode_index(mtrap, label), 2*mode_index(mtrap, label) + 1] for label in labels])
    return np.asarray(V)[np.ix_(indices, indices)]

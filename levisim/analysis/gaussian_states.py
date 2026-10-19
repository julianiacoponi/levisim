"""
Gaussian states of bosonic modes, described by their covariance matrices

Everything here acts on covariance matrices (plain numpy arrays), so it applies equally to the
mechanical modes of a `MultiTrap`, to its cavity mode, or to any other set of bosonic modes.

Conventions (the same as levisim's light quadratures A = a† + a, P = i(a† - a)):
    quadratures per mode      X = b + b†,  P = i(b† - b),  so [X, P] = 2i
    quadrature ordering       u = (X1, P1, X2, P2, ...)
    covariance matrix         V_ij = ½〈{Δu_i, Δu_j}〉, so the vacuum is the identity matrix
    uncertainty principle     V + iΩ ≥ 0, i.e. every symplectic eigenvalue ν ≥ 1
    logarithms                natural, so a two-mode squeezed vacuum S(r)|0,0〉 has E_N = 2r

The entanglement measure used throughout levisim is the logarithmic negativity E_N, computed
from the symplectic spectrum of the partially transposed covariance matrix. For two modes the
positivity of the partial transpose (PPT) is necessary AND sufficient for separability of a
Gaussian state (Simon, PRL 84, 2726 (2000)), so E_N > 0 if and only if the state is entangled.
The other quantities here (Duan and Hillery-Zubairy witnesses, mutual information, discord,
purity) are provided because they are often used in its place, and docs/two_tone_entanglement
shows where each of them gives the wrong answer for levitated-particle states.
"""
import numpy as np

from scipy.optimize import minimize


__all__ = [
    'symplectic_form',
    'symplectic_eigenvalues',
    'is_physical',
    'reduced_covariance',
    'partial_transpose',
    'partially_transposed_symplectic_eigenvalues',
    'log_negativity',
    'negativity',
    'is_entangled',
    'mean_phonon_numbers',
    'symplectic_occupations',
    'phonon_moments',
    'purity',
    'von_neumann_entropy',
    'mutual_information',
    'gaussian_discord',
    'duan_criterion',
    'optimised_duan_criterion',
    'hillery_zubairy_ratio',
    'local_rotation',
    'common_phase_averaged',
    'rotate_transverse_axes',
    'two_mode_squeezed_vacuum',
    'thermal_covariance',
]


# ----------------------------------------------------------------------------------------------
# SYMPLECTIC STRUCTURE
# ----------------------------------------------------------------------------------------------
def symplectic_form(num_modes):
    """
    Ω = ⊕ [[0, 1], [-1, 0]], so that [u_i, u_j] = 2iΩ_ij
    """
    return np.kron(np.eye(num_modes), np.array([[0, 1], [-1, 0]]))


def _modes_of(V):
    num_quadratures = np.shape(V)[0]
    assert num_quadratures % 2 == 0, f'covariance matrix has odd dimension {num_quadratures}'
    return num_quadratures // 2


def symplectic_eigenvalues(V):
    """
    The symplectic eigenvalues ν_k of a covariance matrix, sorted in ascending order

    They are the moduli of the eigenvalues of iΩV, which come in pairs ±ν_k. A physical state
    has every ν_k ≥ 1, with equality for every mode only if the state is pure.
    """
    num_modes = _modes_of(V)
    eigenvalues = np.linalg.eigvals(1j * symplectic_form(num_modes) @ np.asarray(V, dtype=float))
    return np.sort(np.abs(eigenvalues))[::2]


def is_physical(V, tolerance=1e-9):
    """
    True if V is symmetric, positive definite and satisfies the uncertainty principle
    """
    V = np.asarray(V, dtype=float)
    if not np.allclose(V, V.T, atol=tolerance * max(1, np.abs(V).max())):
        return False
    if np.min(np.linalg.eigvalsh(V)) <= 0:
        return False
    return bool(symplectic_eigenvalues(V).min() >= 1 - tolerance)


def reduced_covariance(V, modes):
    """
    Covariance matrix of a subset of the modes (the partial trace of a Gaussian state)
    """
    indices = np.ravel([[2*m, 2*m + 1] for m in modes])
    return np.asarray(V)[np.ix_(indices, indices)]


def partial_transpose(V, modes=(0,)):
    """
    Covariance matrix after partial transposition of the given modes

    Transposition is time reversal, P → -P, on the transposed modes only.
    """
    flip = np.ones(2 * _modes_of(V))
    for m in modes:
        flip[2*m + 1] = -1
    return flip[:, None] * np.asarray(V, dtype=float) * flip[None, :]


def partially_transposed_symplectic_eigenvalues(V, modes=(0,)):
    """
    Symplectic eigenvalues ν̃_k of the partially transposed covariance matrix

    The state is PPT (and, for 1 × N mode bipartitions, separable) if and only if every ν̃_k ≥ 1.
    """
    return symplectic_eigenvalues(partial_transpose(V, modes))


# ----------------------------------------------------------------------------------------------
# ENTANGLEMENT
# ----------------------------------------------------------------------------------------------
def log_negativity(V, modes=(0,)):
    """
    Logarithmic negativity E_N of the bipartition (modes | all other modes)

    E_N = Σ_k max(0, -ln ν̃_k)

    For two modes only the smallest ν̃_- can drop below 1, so E_N = max(0, -ln ν̃_-). It is an
    entanglement monotone (Plenio, PRL 95, 090503 (2005)), an upper bound on the distillable
    entanglement, additive, and invariant under local unitaries such as the free rotation of
    each mode at its own frequency.
    [·]
    """
    ν_tilde = partially_transposed_symplectic_eigenvalues(V, modes)
    return float(np.sum(np.maximum(0.0, -np.log(ν_tilde))))


def negativity(V, modes=(0,)):
    """
    Negativity N = (exp(E_N) - 1)/2, the absolute sum of the negative eigenvalues of ρ^{T_A}
    [·]
    """
    return float((np.exp(log_negativity(V, modes)) - 1) / 2)


def is_entangled(V, modes=(0,), tolerance=1e-12):
    """
    PPT test: True if any partially transposed symplectic eigenvalue is below 1
    """
    return bool(partially_transposed_symplectic_eigenvalues(V, modes).min() < 1 - tolerance)


# ----------------------------------------------------------------------------------------------
# OCCUPATIONS, PURITY, ENTROPY, CORRELATIONS
# ----------------------------------------------------------------------------------------------
def mean_phonon_numbers(V):
    """
    〈b_j†b_j〉 for each mode (zero mean assumed), including any local squeezing energy
    [·]
    """
    V = np.asarray(V, dtype=float)
    diagonal = np.diag(V)
    return (diagonal[0::2] + diagonal[1::2]) / 4 - 0.5


def symplectic_occupations(V):
    """
    Effective thermal occupation of each mode on its own, from 2n + 1 = √det(V_j)

    This is the occupation that sets the purity of each reduced state, and is what Deplano et
    al., Nat. Commun. 16, 4215 (2025) quote as n_x, n_y.
    [·]
    """
    return np.array([
        (np.sqrt(np.linalg.det(reduced_covariance(V, [m]))) - 1) / 2 for m in range(_modes_of(V))
    ])


def phonon_moments(V):
    """
    Normally ordered second moments (N, M) with N_jk = 〈b_j†b_k〉 and M_jk = 〈b_j b_k〉

    b_j = (X_j + iP_j)/2. The commutator [X_j, P_j] = 2i only contributes on the diagonal.
    """
    V = np.asarray(V, dtype=float)
    num_modes = _modes_of(V)
    X, P = slice(0, None, 2), slice(1, None, 2)
    VXX, VPP, VXP, VPX = V[X, X], V[P, P], V[X, P], V[P, X]
    N = (VXX + VPP + 1j*(VXP - VPX)) / 4 - 0.5 * np.eye(num_modes)
    M = (VXX - VPP + 1j*(VXP + VPX)) / 4
    return N, M


def purity(V):
    """
    Tr(ρ²) = 1/√det(V)
    [·]
    """
    return float(1 / np.sqrt(np.linalg.det(np.asarray(V, dtype=float))))


def _entropy_function(ν):
    """g(ν) = ((ν+1)/2) ln((ν+1)/2) - ((ν-1)/2) ln((ν-1)/2), with g(1) = 0"""
    ν = np.atleast_1d(np.asarray(ν, dtype=float))
    plus, minus = (ν + 1) / 2, (ν - 1) / 2
    with np.errstate(divide='ignore', invalid='ignore'):
        minus_term = np.where(minus > 1e-15, minus * np.log(np.where(minus > 1e-15, minus, 1)), 0.0)
    return plus * np.log(plus) - minus_term


def von_neumann_entropy(V):
    """
    S(ρ) = Σ_k g(ν_k) over the symplectic eigenvalues
    [nats]
    """
    return float(np.sum(_entropy_function(np.maximum(symplectic_eigenvalues(V), 1.0))))


def mutual_information(V, modes_A=(0,), modes_B=None):
    """
    I(A:B) = S(A) + S(B) - S(AB)

    Counts classical AND quantum correlations, so it is positive for separable states too.
    [nats]
    """
    num_modes = _modes_of(V)
    if modes_B is None:
        modes_B = [m for m in range(num_modes) if m not in modes_A]
    V_AB = reduced_covariance(V, list(modes_A) + list(modes_B))
    S_A = von_neumann_entropy(reduced_covariance(V, modes_A))
    S_B = von_neumann_entropy(reduced_covariance(V, modes_B))
    return float(S_A + S_B - von_neumann_entropy(V_AB))


def gaussian_discord(V, measured_mode=1):
    """
    Gaussian quantum discord of a two-mode state, for Gaussian measurements on `measured_mode`

    Closed form of Adesso & Datta, PRL 105, 030501 (2010), written in the local symplectic
    invariants A = det α, B = det β, C = det γ, D = det V of V = [[α, γ], [γᵀ, β]], where β is
    the block of the measured mode. Discord is positive for almost every correlated state,
    separable ones included, so it does not witness entanglement (Ferraro et al., PRA 81,
    052318 (2010)). It reproduces Deplano et al. (2025)'s published D_X|Y and D_Y|X.
    [nats]
    """
    V = np.asarray(V, dtype=float)
    assert V.shape == (4, 4), 'gaussian_discord is defined here for two modes'
    if measured_mode == 0:
        swap = np.array([2, 3, 0, 1])
        V = V[np.ix_(swap, swap)]
    α, β, γ = V[:2, :2], V[2:, 2:], V[:2, 2:]
    A, B, C, D = np.linalg.det(α), np.linalg.det(β), np.linalg.det(γ), np.linalg.det(V)

    if (D - A*B)**2 <= (1 + B) * C**2 * (A + D):
        E_min = (2*C**2 + (B - 1)*(D - A) + 2*np.abs(C)*np.sqrt(C**2 + (B - 1)*(D - A))) / (B - 1)**2
    else:
        E_min = (A*B - C**2 + D - np.sqrt(C**4 + (D - A*B)**2 - 2*C**2*(A*B + D))) / (2*B)

    ν_minus, ν_plus = np.maximum(symplectic_eigenvalues(V), 1.0)
    g = lambda x: float(_entropy_function(max(x, 1.0))[0])
    return g(np.sqrt(B)) - g(ν_minus) - g(ν_plus) + g(np.sqrt(E_min))


# ----------------------------------------------------------------------------------------------
# WITNESSES THAT ARE ONLY SUFFICIENT
# ----------------------------------------------------------------------------------------------
def local_rotation(V, angles):
    """
    Rotates each mode's quadratures by its angle θ_j, i.e. applies b_j → b_j·exp(-iθ_j)

    This is free evolution for a time θ_j/Ω_j: X → X cos θ + P sin θ, P → P cos θ - X sin θ.
    """
    blocks = [np.array([[np.cos(θ), np.sin(θ)], [-np.sin(θ), np.cos(θ)]]) for θ in angles]
    R = np.zeros((2*len(blocks), 2*len(blocks)))
    for m, block in enumerate(blocks):
        R[2*m:2*m + 2, 2*m:2*m + 2] = block
    return R @ np.asarray(V, dtype=float) @ R.T


def duan_criterion(V, θ1=0.0, θ2=0.0, a=1.0):
    """
    Duan et al., PRL 84, 2722 (2000) inequality for one choice of local quadratures and weight

    u = a·X1(θ1) + X2(θ2)/a,  v = a·P1(θ1) - P2(θ2)/a
    Every separable state has Var(u) + Var(v) ≥ 2(a² + 1/a²) in these units.

    Returns the ratio (Var u + Var v) / 2(a² + 1/a²); a value below 1 proves entanglement, but
    a value above 1 proves nothing, because the criterion is only sufficient. With fixed lab
    quadratures it misses entanglement whose correlations sit in rotated quadratures, which is
    always the case for the time-periodic states of a multi-tone drive.
    [·]
    """
    W = local_rotation(V, [θ1, θ2])
    u = np.array([a, 0, 1/a, 0])
    v = np.array([0, a, 0, -1/a])
    return float((u @ W @ u + v @ W @ v) / (2 * (a**2 + 1/a**2)))


def optimised_duan_criterion(V, num_starts=12, seed=0):
    """
    Duan ratio minimised over both local rotations and the weight a

    Still only a sufficient criterion: local squeezing is not optimised over, so it can miss
    entanglement that the PPT test (log_negativity) detects. Its verdict (ratio < 1) is exactly
    the Hillery-Zubairy one (hillery_zubairy_ratio > 1); see that docstring.
    Returns (ratio, θ1, θ2, a).
    """
    rng = np.random.default_rng(seed)

    def objective(p):
        return duan_criterion(V, p[0], p[1], np.exp(p[2]))

    best = None
    for _ in range(num_starts):
        start = np.array([rng.uniform(0, 2*np.pi), rng.uniform(0, 2*np.pi), rng.normal(0, 0.5)])
        result = minimize(objective, start, method='Nelder-Mead', options=dict(xatol=1e-8, fatol=1e-10, maxiter=4000))
        if best is None or result.fun < best.fun:
            best = result
    θ1, θ2, log_a = best.x
    return float(best.fun), float(θ1 % (2*np.pi)), float(θ2 % (2*np.pi)), float(np.exp(log_a))


def hillery_zubairy_ratio(V):
    """
    |〈b1 b2〉|² / (〈b1†b1〉〈b2†b2〉), Hillery & Zubairy, PRL 96, 050503 (2006)

    A value above 1 proves entanglement; it is only sufficient, and blind to entanglement that
    is not carried by the pair correlation 〈b1 b2〉.

    It is the same test as optimised_duan_criterion: with u = a·X1 + X2/a, v = a·P1 - P2/a,
    Var(u) + Var(v) = a²(4n1 + 2) + (4n2 + 2)/a² + 8·Re(〈b1 b2〉e^{-i(θ1 + θ2)}), which contains
    neither local squeezing nor the exchange correlation 〈b1†b2〉; minimising over the phases and
    over a gives Var(u) + Var(v) < 2(a² + 1/a²) exactly when |〈b1 b2〉|² > n1·n2.
    [·]
    """
    N, M = phonon_moments(V)
    n1, n2 = np.real(N[0, 0]), np.real(N[1, 1])
    return float(np.abs(M[0, 1])**2 / (n1 * n2))


# ----------------------------------------------------------------------------------------------
# PHASE REFERENCES AND PARTITIONS
# ----------------------------------------------------------------------------------------------
def common_phase_averaged(V, num_angles=16):
    """
    Averages a two-mode state over a common, unknown phase: b_j → b_j·exp(-iθ) for both modes

    This is what an observer without a phase reference sees, e.g. a time average of a state
    whose pair correlation 〈b1 b2〉 rotates at Ω1 + Ω2, or not knowing the relative phase of two
    drive tones. Pair (two-mode squeezing) correlations average to zero; exchange correlations
    〈b1†b2〉 survive. The averaged state is not Gaussian; this returns its covariance matrix,
    whose PPT test is still a valid (second-moment) separability test for any state.
    """
    angles = 2*np.pi * np.arange(num_angles) / num_angles
    return np.mean([local_rotation(V, [θ, θ]) for θ in angles], axis=0)


def rotate_transverse_axes(V, Ω1, Ω2, φ):
    """
    Covariance matrix of the motion along transverse axes rotated by φ from the principal axes

    Mode 1 and mode 2 must be the motions of ONE particle along orthogonal principal axes with
    frequencies Ω1 and Ω2. Positions and momenta are rotated together (a point transformation),
    and each new axis is renormalised; local renormalisation does not change entanglement.

    Because the two zero-point widths differ when Ω1 ≠ Ω2, the trap's product ground state looks
    entangled along rotated axes - exactly E_N = ½|ln(Ω1/Ω2)| at 45° - so 'x-y entanglement' is
    only meaningful for a stated partition, normally the principal axes.
    """
    S = np.diag([1/np.sqrt(2*Ω1), np.sqrt(Ω1/2), 1/np.sqrt(2*Ω2), np.sqrt(Ω2/2)])
    c, s = np.cos(φ), np.sin(φ)
    R = np.array([[c, 0, s, 0], [0, c, 0, s], [-s, 0, c, 0], [0, -s, 0, c]])
    return R @ (2 * S @ np.asarray(V, dtype=float) @ S.T) @ R.T


# ----------------------------------------------------------------------------------------------
# REFERENCE STATES
# ----------------------------------------------------------------------------------------------
def two_mode_squeezed_vacuum(r, θ=0.0):
    """
    Covariance matrix of the two-mode squeezed vacuum with squeezing r and pair phase θ

    〈b1 b2〉 = exp(iθ)·sinh(r)·cosh(r); E_N = 2r and ν̃_- = exp(-2r).
    """
    c, s = np.cosh(2*r), np.sinh(2*r)
    V = np.array([
        [c, 0, s, 0],
        [0, c, 0, -s],
        [s, 0, c, 0],
        [0, -s, 0, c],
    ])
    return local_rotation(V, [-θ/2, -θ/2])


def thermal_covariance(occupations):
    """
    Product of thermal states with the given mean occupations
    """
    return np.diag(np.repeat(2*np.asarray(occupations, dtype=float) + 1, 2))

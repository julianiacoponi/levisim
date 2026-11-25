"""
Time-domain (covariance matrix) description of a `MultiTrap`

The rest of levisim solves the quantum Langevin equations in the frequency domain to get
spectra. Entanglement is a property of the STATE, i.e. of equal-time second moments, so here the
same drift and noise matrices are used in the Lyapunov equation instead::

    dV/dt = A·V + V·Aᵀ + D        steady state:  A·V + V·Aᵀ + D = 0

with A the drift matrix and D the diffusion matrix, both in the real quadrature basis
u = (A, P, X_x1, P_x1, X_y1, P_y1, ...) of levisim's operator ordering c = (a, a†, b_x1, b_x1†, ...),
where for every mode X = b + b† and P = i(b† - b). The covariance V_ij = ½〈{Δu_i, Δu_j}〉 then
has the vacuum as the identity, which is the convention of `levisim.analysis.gaussian_states`.

The noise is levisim's: each operator pair (c_k, c_k†) with damping γ_k and occupation n_k
(the factors in `MultiTrap.quantum_factor_per_noise_operator`) contributes γ_k(2n_k + 1) to the
diffusion of both of its quadratures. An extra heating rate Γ_h [phonons/s] of a mode (e.g. from
`levisim.multiphysics.decoherence.decoherence_budget`) adds 2Γ_h, the same as raising its bath
occupation n → n + Γ_h/γ, which is how MultiTrap already adds photon recoil.

Directly coupled particles (MultiTrap(interactions=[...])) are described with or without the cavity
mode: `cavity=False` keeps only the mechanical quadratures, i.e. the particles in their traps with
their direct couplings and the static shifts, as if the cavity were removed. Entanglement between
particles is `particle_log_negativity`, for the steady state (`steady_state_covariance`) or along a
protocol that starts from a product state (`product_state_covariance`, `evolve_covariance`, which
solves dV/dt = A·V + V·Aᵀ + D exactly between the requested times).
"""
import numpy as np

from scipy.linalg import block_diag, expm, solve_continuous_lyapunov

from levisim.analysis.gaussian_states import log_negativity, reduced_covariance, squeezed_thermal_covariance


__all__ = [
    'complex_to_quadrature_transform',
    'quadrature_drift_matrix',
    'quadrature_diffusion_matrix',
    'steady_state_covariance',
    'mode_index',
    'dimension_covariance',
    'quadrature_matrices',
    'covariance_evolution',
    'evolve_covariance',
    'product_state_covariance',
    'particle_mode_indices',
    'particle_log_negativity',
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


def quadrature_matrices(mtrap, cavity=True, extra_heating=None):
    """
    (A, D) of the MultiTrap in quadratures, with or without the cavity mode

    - cavity=True: the full drift and diffusion matrices, mode order (cavity, *mtrap.labels)
    - cavity=False: the mechanical modes only, mode order mtrap.labels: the rows and columns of the
      cavity quadratures are dropped, which describes the particles with their direct couplings and
      static shifts but without the cavity (exact when no dimension couples to the cavity, G = 0)
    - extra_heating: {label: Γ_h} heating rates [phonons/s] added to the diffusion, 2Γ_h on both
      quadratures of each listed mechanical mode
    [2π·Hz, 2π·Hz]
    """
    A = quadrature_drift_matrix(mtrap.drift_matrix)
    D = quadrature_diffusion_matrix(mtrap)
    for label, rate in (extra_heating or {}).items():
        k = mode_index(mtrap, label)
        D[2*k, 2*k] += 2 * rate
        D[2*k + 1, 2*k + 1] += 2 * rate
    if not cavity:
        A, D = A[2:, 2:], D[2:, 2:]
    return A, D


def steady_state_covariance(mtrap, cavity=True, extra_heating=None):
    """
    Steady-state covariance matrix of the MultiTrap's single-tone Linear Amplifier Model

    Solves A·V + V·Aᵀ + D = 0 with levisim's own (time-independent, tweezer-frame) drift matrix.
    Raises if the drift matrix is unstable, because then no steady state exists. With
    cavity=False or extra_heating, A and D are those of `quadrature_matrices` (the default
    arguments give the same numbers as before).
    """
    if not cavity or extra_heating:
        A, D = quadrature_matrices(mtrap, cavity=cavity, extra_heating=extra_heating)
        largest_growth_rate = np.max(np.linalg.eigvals(A).real)
        if largest_growth_rate >= 0:
            raise ValueError(f'{mtrap} is unstable (largest Re eigenvalue {largest_growth_rate:.3e}), so it has no steady state')
        V = solve_continuous_lyapunov(A, -D)
        return (V + V.T) / 2
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


def covariance_evolution(A, D, V0, times):
    """
    Covariance V(t) of du/dt = A·u + noise with diffusion D, from V(0) = V0, at the given times

    Solves dV/dt = A·V + V·Aᵀ + D exactly between successive times with Van Loan's block exponential:
    over a step τ, V → Φ·V·Φᵀ + Q with Φ = exp(Aτ) and Q = ∫₀^τ exp(As)·D·exp(Aᵀs) ds, both read off
    exp([[-A, D], [0, Aᵀ]]·τ), on substeps with ‖A‖₁·τ ≤ 1 that are recombined exactly. It needs no
    stability, so it also follows growing or undamped motion.
    `times` must be non-decreasing and start at or after 0.
    [shape (len(times), n, n)]
    """
    A, D = np.asarray(A, dtype=float), np.asarray(D, dtype=float)
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or np.any(np.diff(times) < 0) or (len(times) and times[0] < 0):
        raise ValueError('times must be a non-decreasing 1D array starting at or after 0')
    n = A.shape[0]
    V = np.array(V0, dtype=float)
    out = np.empty((len(times), n, n))
    t_previous = 0.0
    # Van Loan needs exp(-Aτ), which grows as exp(|Re λ|·τ), and expm loses accuracy for a large
    # ‖Aτ‖: long steps are split into 2^k substeps with ‖A‖₁·τ ≤ 1 and recombined exactly,
    # (Φ, Q) → (Φ·Φ, Φ·Q·Φᵀ + Q)
    rate = float(np.linalg.norm(A, 1)) if n else 0.0
    cache = {}
    for index, t in enumerate(times):
        τ = t - t_previous
        if τ > 0:
            key = float(τ)
            if key not in cache:
                doublings = max(0, int(np.ceil(np.log2(max(rate * τ, 1.0)))))
                step = τ / 2**doublings
                block = np.zeros((2*n, 2*n))
                block[:n, :n] = -A * step
                block[:n, n:] = D * step
                block[n:, n:] = A.T * step
                F = expm(block)
                Φ = F[n:, n:].T
                Q = Φ @ F[:n, n:]
                for _ in range(doublings):
                    Q = Φ @ Q @ Φ.T + Q
                    Φ = Φ @ Φ
                cache[key] = (Φ, (Q + Q.T) / 2)
            Φ, Q = cache[key]
            V = Φ @ V @ Φ.T + Q
            V = (V + V.T) / 2
        out[index] = V
        t_previous = t
    return out


def evolve_covariance(mtrap, V0, times, cavity=True, extra_heating=None):
    """
    `covariance_evolution` with the MultiTrap's (A, D) from `quadrature_matrices`
    [shape (len(times), n, n)]
    """
    A, D = quadrature_matrices(mtrap, cavity=cavity, extra_heating=extra_heating)
    return covariance_evolution(A, D, V0, times)


def product_state_covariance(mtrap, occupations, cavity=True, squeezing=None):
    """
    Covariance of an uncorrelated (product) initial state of the MultiTrap's modes

    - occupations: {label: n} thermal occupations of the mechanical modes (a number for all)
    - squeezing: {label: (r, θ)} or {label: r}: a squeezed thermal state of that mode,
      `gaussian_states.squeezed_thermal_covariance`
    - cavity: with the cavity mode (in its vacuum) first, as in `quadrature_matrices`
    """
    blocks = [np.eye(2)] if cavity else []
    for label in mtrap.labels:
        n = occupations if np.ndim(occupations) == 0 else occupations[label]
        r, θ = 0.0, 0.0
        if squeezing and label in squeezing:
            value = squeezing[label]
            r, θ = (value, 0.0) if np.ndim(value) == 0 else value
        blocks.append(squeezed_thermal_covariance(n, r, θ))
    return block_diag(*blocks)


def particle_mode_indices(mtrap, trap_indices, cavity=True):
    """Mode indices (in the covariance matrix) of every dimension of the given traps (1-based)"""
    trap_indices = [trap_indices] if np.ndim(trap_indices) == 0 else list(trap_indices)
    offset = 1 if cavity else 0
    return [offset + mtrap.label_to_index[label] for label in mtrap.labels
            if mtrap.dimensions[label].trap_index in trap_indices]


def particle_log_negativity(mtrap, V, particles_A=1, particles_B=None, cavity=True):
    """
    Logarithmic negativity E_N between the motion of particles A and particles B

    All modelled dimensions of each particle belong to its side; every other mode (the cavity, other
    particles) is traced out. particles_B defaults to every particle not in A. Natural logarithm, so
    a two-mode squeezed vacuum of squeezing r has E_N = 2r (divide by ln 2 for the base-2 value).
    V may be one covariance matrix or a stack (times, n, n).
    [·]
    """
    A_list = [particles_A] if np.ndim(particles_A) == 0 else list(particles_A)
    if particles_B is None:
        particles_B = [n for n in range(1, mtrap.num_traps + 1) if n not in A_list]
    modes_A = particle_mode_indices(mtrap, A_list, cavity)
    modes_B = particle_mode_indices(mtrap, particles_B, cavity)
    V = np.asarray(V)
    keep = modes_A + modes_B
    if V.ndim == 3:
        return np.array([log_negativity(reduced_covariance(v, keep), modes=range(len(modes_A))) for v in V])
    return log_negativity(reduced_covariance(V, keep), modes=range(len(modes_A)))

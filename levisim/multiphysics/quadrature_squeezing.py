"""
Spectral covariance matrix of the output-light quadratures, and squeezing read out from it

`MultiTrap.homodyne_spectrum` gives the spectrum of one output quadrature at a time (the angle is
`cavity.θhom`). Optical squeezing is a property of the full 2 × 2 spectral covariance matrix of the
output quadratures (Q, P) at each frequency: from it follow the spectrum of every quadrature, the
optimal (least-noisy) quadrature at each frequency, and the effect of detection inefficiency and
phase noise. This module computes that matrix from the same drift and noise matrices that levisim
uses everywhere else, and provides the read-outs used by Pontin, Deplano, Marino and Marin,
Phys. Rev. A 114, 053704 (2026), Eqs. (1)-(3) and Appendix A.

Conventions
-----------
- Quadratures as in `levisim.multiphysics.covariance` (and in the paper above):
  Q = a_out + a_out†, P = i(a_out† − a_out), so the vacuum (shot noise) is S_QQ = S_PP = 1.
- Spectra are symmetrised, S̄_AB(ω) = ½〈{A(ω), B†(ω)}〉, and the cross spectrum returned is the
  real part, S_QP = ½(S̄_QP + S̄_PQ). For output-field quadratures these are even in ω.
- The quadrature at detection phase φ is X_φ = Q cos φ + P sin φ, with spectrum
  S_φ = S_QQ cos²φ + S_PP sin²φ + S_QP sin 2φ (Eq. (2) of the paper without phase noise).
  `MultiTrap.homodyne_quadrature` uses P_out = i(a_out − a_out†) = −P, so
  ``MultiTrap.homodyne_spectrum`` with ``cavity.θhom = θ`` equals ``quadrature_spectrum(..., φ=-θ)``.
- Input-output: a_out = a_in − √κ a, the whole cavity linewidth κ coupled to the detected port, as
  in `MultiTrap.output_light_noise_vectors`. Losses (a second mirror, detection) enter through the
  detection efficiency η, `with_detection_efficiency`.

Mechanical noise
----------------
levisim's bath for each mode is a damping γ with occupation n acting on b and b†, so its diffusion
γ(2n + 1) enters both the position X and the momentum P of the mode (`quadrature_diffusion_matrix`).
Gas collisions and photon recoil are force noises, which diffuse only the momentum. For the same
heating rate dn/dt the two choices give identical occupations at weak coupling, but different
spectra away from the mechanical resonance: the symmetric bath drives X(ω) with weight
(ω² + Ω²)/(2Ω²) relative to a force noise. ``noise='force'`` moves each mechanical mode's position
diffusion onto its momentum (same dn/dt), which is the model of Deplano et al., Nat. Commun. 16,
4215 (2025), Eq. (3), and of the PRA's Appendix A. The cavity's diffusion is unchanged.
"""
import numpy as np

from scipy.linalg import solve_continuous_lyapunov

from levisim.multiphysics.covariance import quadrature_diffusion_matrix, quadrature_drift_matrix


__all__ = [
    'NOISE_MODELS',
    'diffusion_matrix',
    'steady_state_covariance_with_noise',
    'output_quadrature_spectra',
    'with_detection_efficiency',
    'phase_noise_averaged',
    'quadrature_spectrum',
    'optimal_quadrature_spectrum',
]


NOISE_MODELS = ('levisim', 'force')


def diffusion_matrix(mtrap, noise='levisim'):
    """
    Quadrature diffusion matrix D for the noise model `noise`

    'levisim': levisim's own D = diag(γ_k(2n_k + 1)) on both quadratures of every mode
    (`levisim.multiphysics.covariance.quadrature_diffusion_matrix`).
    'force': the same, except that each mechanical mode's position diffusion is moved onto its
    momentum, D_PP → D_XX + D_PP and D_XX → 0, which keeps the heating rate dn/dt = (D_XX + D_PP)/4.
    [2π·Hz]
    """
    if noise not in NOISE_MODELS:
        raise ValueError(f'{noise=} should be one of {NOISE_MODELS}')
    D = quadrature_diffusion_matrix(mtrap).astype(float)
    if noise == 'force':
        for k in range(1, D.shape[0] // 2):     # mode 0 is the cavity
            D[2*k + 1, 2*k + 1] += D[2*k, 2*k]
            D[2*k, 2*k] = 0.0
    return D


def steady_state_covariance_with_noise(mtrap, noise='levisim'):
    """
    Steady-state quadrature covariance V (vacuum = identity) for the noise model `noise`

    Solves A·V + V·Aᵀ + D = 0 with levisim's drift matrix and `diffusion_matrix(mtrap, noise)`;
    with noise='levisim' this is `levisim.multiphysics.covariance.steady_state_covariance`.
    Raises ValueError if the drift matrix is unstable.
    """
    A = quadrature_drift_matrix(mtrap.drift_matrix)
    largest_growth_rate = np.max(np.linalg.eigvals(A).real)
    if largest_growth_rate >= 0:
        raise ValueError(f'{mtrap} is unstable (largest Re eigenvalue {largest_growth_rate:.3e}), so it has no steady state')
    return solve_continuous_lyapunov(A, -diffusion_matrix(mtrap, noise))


def output_quadrature_spectra(mtrap, ω, noise='levisim'):
    """
    Symmetrised spectral covariance matrix of the output quadratures, (S_QQ, S_PP, S_QP)

    Linear Amplifier Model in the quadrature basis u = (Q_c, P_c, X_1, P_1, ...): with the drift
    matrix A and white noises of covariance D (`diffusion_matrix`), u(ω) = (−iω − A)⁻¹ n(ω), and
    the output quadratures are u_out = −√κ u_c + u_in, where the cavity's noise is n_c = √κ u_in.
    So u_out(ω) = L(ω) n(ω) with L = −√κ [(−iω − A)⁻¹]_c + [1/√κ on the cavity columns], and
    S̄(ω) = L(ω) D L(ω)†. Returns three real arrays over ω, normalised to shot noise (vacuum = 1).
    ω: angular frequencies [2π·Hz] (scalar or array)
    """
    ω = np.atleast_1d(np.asarray(ω, dtype=float))
    A = quadrature_drift_matrix(mtrap.drift_matrix)
    D = diffusion_matrix(mtrap, noise)
    n = A.shape[0]
    κ = mtrap.cavity.κ
    M = -1j * ω[:, None, None] * np.eye(n)[None] - A[None]
    G = np.linalg.inv(M)                                    # (len(ω), n, n)
    L = -np.sqrt(κ) * G[:, :2, :]
    L[:, :, :2] += np.eye(2)[None] / np.sqrt(κ)
    S = np.einsum('wij,jk,wlk->wil', L, D, L.conj())
    return S[:, 0, 0].real, S[:, 1, 1].real, S[:, 0, 1].real


def with_detection_efficiency(S_QQ, S_PP, S_QP, η):
    """
    Spectra seen by a detector of total efficiency η: S → η S + (1 − η) on the diagonal, S_QP → η S_QP

    For a heterodyne (or dual-homodyne) measurement of both quadratures, η includes the factor 1/2
    of the simultaneous measurement (PRA 114, 053704, Appendix A and the text after Eq. (1)).
    """
    if not 0 <= η <= 1:
        raise ValueError(f'{η=} should be in [0, 1]')
    S_QQ, S_PP, S_QP = (np.asarray(S, dtype=float) for S in (S_QQ, S_PP, S_QP))
    return η * S_QQ + 1 - η, η * S_PP + 1 - η, η * S_QP


def phase_noise_averaged(S_QQ, S_PP, S_QP, σθ2):
    """
    Spectral covariance matrix averaged over a Gaussian phase jitter of variance σθ2 [rad²]

    R(θ) V R(θ)ᵀ averaged over θ ~ N(0, σθ²) (PRA 114, 053704, Eq. (1) with S_QP = S_PQ):
    S_QQ → e^(−σ²)(S_QQ cosh σ² + S_PP sinh σ²), S_PP → e^(−σ²)(S_PP cosh σ² + S_QQ sinh σ²),
    S_QP → e^(−2σ²) S_QP. The vacuum (identity) is unchanged, so the order relative to
    `with_detection_efficiency` does not matter.
    """
    if σθ2 < 0:
        raise ValueError(f'{σθ2=} should be ≥ 0')
    S_QQ, S_PP, S_QP = (np.asarray(S, dtype=float) for S in (S_QQ, S_PP, S_QP))
    e1, ch, sh = np.exp(-σθ2), np.cosh(σθ2), np.sinh(σθ2)
    return e1 * (S_QQ * ch + S_PP * sh), e1 * (S_PP * ch + S_QQ * sh), np.exp(-2 * σθ2) * S_QP


def quadrature_spectrum(S_QQ, S_PP, S_QP, φ, σθ2=0.0):
    """
    Spectrum of the quadrature X_φ = Q cos φ + P sin φ, with optional phase noise σθ2 [rad²]

    S_φ = (S_QQ + S_PP)/2 + (S_QQ − S_PP)/2 · e^(−2σ²) cos 2φ + S_QP e^(−2σ²) sin 2φ
    (PRA 114, 053704, Eq. (2)). φ may be an array; it broadcasts against the spectra, e.g.
    φ[:, None] with spectra over ω gives a (φ, ω) map.
    """
    S_QQ, S_PP, S_QP = (np.asarray(S, dtype=float) for S in (S_QQ, S_PP, S_QP))
    φ = np.asarray(φ, dtype=float)
    d = np.exp(-2 * σθ2)
    return (S_QQ + S_PP) / 2 + (S_QQ - S_PP) / 2 * d * np.cos(2 * φ) + S_QP * d * np.sin(2 * φ)


def optimal_quadrature_spectrum(S_QQ, S_PP, S_QP, σθ2=0.0):
    """
    Least-noisy quadrature at every frequency: (S_opt, φ_opt)

    S_opt = ½[S_QQ + S_PP − e^(−2σ²) √((S_QQ − S_PP)² + 4 S_QP²)], the minimum of
    `quadrature_spectrum` over φ, reached at φ_opt = ½ atan2(−2 S_QP, −(S_QQ − S_PP)) ∈ (−π/2, π/2].
    (PRA 114, 053704, Eq. (3) as printed lacks the overall factor ½, without which the vacuum would
    give S_opt = 2 instead of 1.)
    """
    S_QQ, S_PP, S_QP = (np.asarray(S, dtype=float) for S in (S_QQ, S_PP, S_QP))
    root = np.sqrt((S_QQ - S_PP)**2 + 4 * S_QP**2)
    S_opt = 0.5 * (S_QQ + S_PP - np.exp(-2 * σθ2) * root)
    φ_opt = 0.5 * np.arctan2(-2 * S_QP, -(S_QQ - S_PP))
    return S_opt, φ_opt

"""
Output-quadrature spectral covariance matrix and squeezing read-outs (quadrature_squeezing.py)

- the matrix reproduces `MultiTrap.homodyne_spectrum` at every homodyne angle (θhom = −φ);
- with the force-noise model it equals the closed form of Pontin et al., PRA 114, 053704 (2026),
  Appendix A (Eqs. (A9)-(A16), with the factor i in (A11) removed, as (A2) requires);
- the phase-noise average equals a numerical Gaussian average of rotated matrices (Eq. (1));
- the optimal spectrum is the minimum over φ (Eq. (3) with its factor ½);
- the vacuum is invariant under detection inefficiency and phase noise, and the force-noise
  model keeps the occupations of levisim's bath at weak coupling.
"""
import numpy as np
import pytest

from levisim.core.units import kHz, nm
from levisim.multiphysics.covariance import steady_state_covariance
from levisim.multiphysics.enums.quantum_linear_theory_method import QuantumLinearTheoryMethod
from levisim.multiphysics.multitrap import MultiTrap
from levisim.multiphysics.quadrature_squeezing import (
    diffusion_matrix,
    optimal_quadrature_spectrum,
    output_quadrature_spectra,
    phase_noise_averaged,
    quadrature_spectrum,
    steady_state_covariance_with_noise,
    with_detection_efficiency,
)
from levisim.physics.tweezer import Tweezer

LAM = QuantumLinearTheoryMethod.LINEAR_AMPLIFIER_MODEL
Ω = {'x1': 122.0 * kHz, 'y1': 109.0 * kHz}
G = {'x1': 14.0 * kHz, 'y1': 10.0 * kHz}
κ, Δ = 57.0 * kHz, -111.0 * kHz
ω = np.linspace(60, 180, 121) * kHz + 0.0137 * kHz      # avoid ω exactly on a bare resonance


def toy_system(g=G, γ=2 * np.pi * 1e-4, T=300.0):
    """One particle, two axes, strongly coupled and sideband resolved (n_bath ≈ 5 × 10⁷, γ·n ≈ 2π × 5 kHz)"""
    toy_params = dict(Ω_model='custom', toy_Ωjn_dict=dict(Ω), gjn_dict=dict(g), use_Γ_formula=False,
                      toy_Γ=γ, T=T, m=1e-18)
    return MultiTrap(num_traps=1, Cartesian_axes_per_trap=2, toy_params=toy_params, ω_range=ω,
                     cavity_kwargs=dict(κ=κ, Δ=Δ, ωtw=Tweezer(λ=1064 * nm).ω))


@pytest.fixture(scope='module')
def mtrap():
    m = toy_system()
    assert np.max(np.linalg.eigvals(m.drift_matrix).real) < 0, 'the test system should be stable'
    return m


@pytest.mark.parametrize('θ', [0.0, 0.3, np.pi / 4, 1.1, np.pi / 2, 2.5])
def test_quadrature_spectrum_reproduces_homodyne_spectrum(mtrap, θ):
    S = output_quadrature_spectra(mtrap, ω)
    mtrap.cavity.θhom = θ
    homodyne = mtrap.homodyne_spectrum(LAM)
    np.testing.assert_allclose(quadrature_spectrum(*S, φ=-θ), homodyne, rtol=1e-10)


def test_spectra_are_shot_noise_without_coupling():
    m = toy_system(g={'x1': 0.0, 'y1': 0.0})
    for noise in ('levisim', 'force'):
        S_QQ, S_PP, S_QP = output_quadrature_spectra(m, ω, noise=noise)
        np.testing.assert_allclose(S_QQ, 1, atol=1e-12)
        np.testing.assert_allclose(S_PP, 1, atol=1e-12)
        np.testing.assert_allclose(S_QP, 0, atol=1e-12)


def test_spectra_are_even_in_frequency(mtrap):
    for noise in ('levisim', 'force'):
        np.testing.assert_allclose(output_quadrature_spectra(mtrap, -ω, noise),
                                   output_quadrature_spectra(mtrap, ω, noise), rtol=1e-10)


def test_force_noise_matches_the_pra_appendix_a_closed_form(mtrap):
    """PRA 114, 053704 (2026), Eqs. (A9)-(A16), no mechanical damping, heating Γ_j = γ(n_j + ½)"""
    factors = np.asarray(mtrap.quantum_factor_per_noise_operator)
    Γ = {label: mtrap.dimensions[label].Γ * (factors[mtrap.label_to_noise_index[label]]
                                              + factors[mtrap.label_to_noise_index[label] + 1]) / 2
         for label in Ω}
    gx, gy = (abs(G[k]) for k in ('x1', 'y1'))

    def χc(w):
        return 1 / (-1j * (Δ + w) + κ / 2)

    def coefficients(w):
        Gw = gx**2 * Ω['x1'] / (Ω['x1']**2 - w**2) + gy**2 * Ω['y1'] / (Ω['y1']**2 - w**2)
        den = 1 - 2j * (χc(w) - np.conj(χc(-w))) * Gw
        A_Q = 1j * np.sqrt(κ) * (χc(w) - np.conj(χc(-w))) / den
        B_Q = κ * χc(w) / den - 1
        A_P = np.sqrt(κ) * (χc(w) + np.conj(χc(-w))) / den        # (A11) without its factor i
        B_P = -1j * (κ * χc(w) * (1 + 4j * np.conj(χc(-w)) * Gw) / den - 1)
        return A_Q, B_Q, A_P, B_P

    A_Q, B_Q, A_P, B_P = coefficients(ω)
    _, B_Qm, _, B_Pm = coefficients(-ω)
    thermal = 4 * sum(g**2 * (Ω[k] / (Ω[k]**2 - ω**2))**2 * Γ[k] for g, k in ((gx, 'x1'), (gy, 'y1')))
    S_QQ = thermal * abs(A_Q)**2 + 0.5 * (abs(B_Qm)**2 + abs(B_Q)**2)
    S_PP = thermal * abs(A_P)**2 + 0.5 * (abs(B_Pm)**2 + abs(B_P)**2)
    S_QP = thermal * np.real(np.conj(A_Q) * A_P) + 0.5 * np.real(np.conj(B_Qm) * B_Pm + np.conj(B_Q) * B_P)

    levisim = output_quadrature_spectra(mtrap, ω, noise='force')
    for ours, theirs in zip(levisim, (S_QQ, S_PP, S_QP)):
        np.testing.assert_allclose(ours, theirs, atol=1e-6)
    assert np.ptp(S_QP) > 0.1, 'the cross spectrum should be non-trivial here'


def test_force_noise_differs_from_levisim_bath_off_resonance(mtrap):
    S_levisim = output_quadrature_spectra(mtrap, ω, 'levisim')
    S_force = output_quadrature_spectra(mtrap, ω, 'force')
    assert np.max(np.abs(S_levisim[0] - S_force[0])) > 1e-3


def test_force_noise_keeps_the_heating_rate(mtrap):
    D, D_force = diffusion_matrix(mtrap, 'levisim'), diffusion_matrix(mtrap, 'force')
    np.testing.assert_array_equal(D[:2, :2], D_force[:2, :2])
    for k in (1, 2):
        assert D_force[2 * k, 2 * k] == 0
        assert D_force[2 * k + 1, 2 * k + 1] == pytest.approx(D[2 * k, 2 * k] + D[2 * k + 1, 2 * k + 1])
    with pytest.raises(ValueError):
        diffusion_matrix(mtrap, 'thermal')


def test_covariance_with_levisim_noise_is_levisim_covariance(mtrap):
    np.testing.assert_allclose(steady_state_covariance_with_noise(mtrap, 'levisim'), steady_state_covariance(mtrap))


def test_force_noise_occupations_agree_at_weak_coupling():
    m = toy_system(g={'x1': 0.5 * kHz, 'y1': 0.4 * kHz})
    V, V_force = steady_state_covariance(m), steady_state_covariance_with_noise(m, 'force')
    for k in (1, 2):
        n = (V[2*k, 2*k] + V[2*k + 1, 2*k + 1] - 2) / 4
        n_force = (V_force[2*k, 2*k] + V_force[2*k + 1, 2*k + 1] - 2) / 4
        assert n_force == pytest.approx(n, rel=1e-3)
        assert V_force[2*k, 2*k + 1] == pytest.approx(0, abs=1e-6 * V_force[2*k, 2*k])


def test_phase_noise_average_matches_numerical_average(mtrap):
    S = np.array(output_quadrature_spectra(mtrap, ω[::10]))
    σ2 = 0.062
    θ = np.linspace(-8, 8, 4001) * np.sqrt(σ2)
    weights = np.exp(-θ**2 / (2 * σ2))
    weights /= weights.sum()
    c, s = np.cos(θ)[:, None], np.sin(θ)[:, None]
    S_QQ, S_PP, S_QP = S[0][None], S[1][None], S[2][None]
    rotated = (c*c*S_QQ + s*s*S_PP - 2*c*s*S_QP,
               s*s*S_QQ + c*c*S_PP + 2*c*s*S_QP,
               (c*c - s*s)*S_QP + c*s*(S_QQ - S_PP))
    numerical = [np.sum(weights[:, None] * r, axis=0) for r in rotated]
    np.testing.assert_allclose(phase_noise_averaged(*S, σ2), numerical, rtol=1e-9, atol=1e-12)
    # and Eq. (2) is the quadrature spectrum of the averaged matrix
    φ = 0.7
    np.testing.assert_allclose(quadrature_spectrum(*S, φ, σθ2=σ2), quadrature_spectrum(*phase_noise_averaged(*S, σ2), φ))


@pytest.mark.parametrize('σ2', [0.0, 0.062])
def test_optimal_spectrum_is_the_minimum_over_phase(mtrap, σ2):
    S = output_quadrature_spectra(mtrap, ω)
    φ = np.linspace(-np.pi / 2, np.pi / 2, 20001)[:, None]
    brute = quadrature_spectrum(*S, φ, σθ2=σ2).min(axis=0)
    S_opt, φ_opt = optimal_quadrature_spectrum(*S, σθ2=σ2)
    np.testing.assert_allclose(S_opt, brute, atol=1e-7)
    np.testing.assert_allclose(quadrature_spectrum(*S, φ_opt, σθ2=σ2), S_opt, atol=1e-12)
    assert S_opt.min() < 1, 'this strongly coupled, low-occupation system squeezes the output light'


def test_vacuum_is_invariant_and_efficiency_scales_the_excess(mtrap):
    vacuum = (np.ones(3), np.ones(3), np.zeros(3))
    for out in (with_detection_efficiency(*vacuum, 0.3), phase_noise_averaged(*vacuum, 0.2)):
        np.testing.assert_allclose(out, vacuum)
    S = output_quadrature_spectra(mtrap, ω)
    S_opt, _ = optimal_quadrature_spectrum(*S)
    S_opt_η, _ = optimal_quadrature_spectrum(*with_detection_efficiency(*S, 0.16))
    np.testing.assert_allclose(S_opt_η - 1, 0.16 * (S_opt - 1), atol=1e-12)
    with pytest.raises(ValueError):
        with_detection_efficiency(*S, 1.2)

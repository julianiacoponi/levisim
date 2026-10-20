"""
Gaussian-state tools in levisim.analysis.gaussian_states

Every expected value here is a closed-form result for a reference state, so these tests pin the
conventions (vacuum = identity, natural logarithms, X = b + b†, P = i(b† - b)) as well as the
code. The 'witness' tests document, rather than hide, where the sufficient-only criteria fail.
"""
import numpy as np
import pytest

from levisim.analysis.gaussian_states import (
    common_phase_averaged,
    duan_criterion,
    gaussian_discord,
    hillery_zubairy_ratio,
    is_entangled,
    is_physical,
    local_rotation,
    log_negativity,
    mean_phonon_numbers,
    mutual_information,
    optimised_duan_criterion,
    partially_transposed_symplectic_eigenvalues,
    phonon_moments,
    purity,
    reduced_covariance,
    rotate_transverse_axes,
    symplectic_eigenvalues,
    symplectic_occupations,
    thermal_covariance,
    two_mode_squeezed_vacuum,
    von_neumann_entropy,
)


def single_mode_squeezed(r):
    """Squeezed vacuum with Var(X) = exp(-2r)"""
    return np.diag([np.exp(-2*r), np.exp(2*r)])


def beam_splitter(V):
    """50:50 beam splitter acting on both quadratures of a two-mode covariance matrix"""
    s = 1 / np.sqrt(2)
    BS = np.array([[s, 0, s, 0], [0, s, 0, s], [s, 0, -s, 0], [0, s, 0, -s]])
    return BS @ V @ BS.T


@pytest.mark.parametrize('r', [0.05, 0.4, 1.3])
def test_two_mode_squeezed_vacuum_has_log_negativity_2r(r):
    V = two_mode_squeezed_vacuum(r, θ=0.3)
    assert is_physical(V)
    assert np.isclose(partially_transposed_symplectic_eigenvalues(V).min(), np.exp(-2*r))
    assert np.isclose(log_negativity(V), 2*r)
    assert np.isclose(purity(V), 1)


def test_pair_phase_convention():
    """two_mode_squeezed_vacuum(r, θ) has 〈b1 b2〉 = exp(iθ)·sinh(r)·cosh(r)"""
    r, θ = 0.6, 1.1
    N, M = phonon_moments(two_mode_squeezed_vacuum(r, θ))
    assert np.isclose(M[0, 1], np.exp(1j*θ) * np.sinh(r) * np.cosh(r))
    assert np.allclose(np.real(np.diag(N)), np.sinh(r)**2)
    assert np.allclose(mean_phonon_numbers(two_mode_squeezed_vacuum(r, θ)), np.sinh(r)**2)


def test_thermal_states_are_separable_with_no_correlations():
    V = thermal_covariance([0.3, 4.0])
    assert log_negativity(V) == 0
    assert not is_entangled(V)
    assert np.isclose(mutual_information(V), 0)
    assert abs(gaussian_discord(V)) < 1e-12
    assert np.allclose(symplectic_occupations(V), [0.3, 4.0])
    assert np.allclose(symplectic_eigenvalues(V), [1.6, 9.0])


def test_unphysical_matrix_is_rejected():
    assert not is_physical(np.diag([0.5, 0.5, 1, 1]))


def test_pure_state_discord_and_mutual_information_are_entanglement_entropies():
    V = two_mode_squeezed_vacuum(0.7)
    S = von_neumann_entropy(reduced_covariance(V, [0]))
    assert np.isclose(gaussian_discord(V, measured_mode=1), S)
    assert np.isclose(gaussian_discord(V, measured_mode=0), S)
    assert np.isclose(mutual_information(V), 2*S)


@pytest.mark.parametrize('n', [0.0, 0.5, 3.0])
def test_squeezed_light_on_a_beam_splitter_entangles_only_above_the_thermal_threshold(n):
    """
    Squeezed vacuum and a thermal state of n on a 50:50 beam splitter: ν̃_-² = (2n+1)·exp(-2r),
    so the output is entangled iff exp(2r) > 2n + 1
    """
    threshold = 0.5 * np.log(2*n + 1)
    for r, entangled in [(threshold + 0.05, True), (max(threshold - 0.05, 0.01), n == 0)]:
        V_in = np.zeros((4, 4))
        V_in[:2, :2] = single_mode_squeezed(r)
        V_in[2:, 2:] = (2*n + 1) * np.eye(2)
        V = beam_splitter(V_in)
        assert np.isclose(partially_transposed_symplectic_eigenvalues(V).min(), np.sqrt((2*n + 1) * np.exp(-2*r)))
        assert is_entangled(V) == entangled


def test_log_negativity_is_invariant_under_local_rotations():
    V = beam_splitter(np.diag([np.exp(-1.2), np.exp(1.2), 2.0, 2.0]))
    assert np.isclose(log_negativity(local_rotation(V, [0.4, -2.1])), log_negativity(V))


def test_fixed_quadrature_duan_test_misses_rotated_entanglement():
    """
    The same entangled state, viewed with lab quadratures a quarter-period later, fails the
    fixed-quadrature Duan test; optimising the local phases recovers it, and PPT never misses it
    """
    # for this pair phase u = X1 + X2 and v = P1 - P2 are the ANTI-squeezed combinations
    V = two_mode_squeezed_vacuum(0.5, θ=0.0)
    assert log_negativity(V) > 0.99
    assert duan_criterion(V) > 1
    # half a pair-phase cycle later (θ = π) the same fixed test succeeds
    assert np.isclose(duan_criterion(two_mode_squeezed_vacuum(0.5, θ=np.pi)), np.exp(-1.0))
    ratio, *_ = optimised_duan_criterion(V)
    assert np.isclose(ratio, np.exp(-1.0), atol=1e-6)


def test_hillery_zubairy_ratio_of_two_mode_squeezed_vacuum():
    r = 0.4
    assert np.isclose(hillery_zubairy_ratio(two_mode_squeezed_vacuum(r)), 1 / np.tanh(r)**2)


def test_phase_averaging_destroys_pair_correlations():
    """Without a phase reference a two-mode squeezed state is only classically correlated"""
    V = two_mode_squeezed_vacuum(0.8)
    V_averaged = common_phase_averaged(V)
    N, M = phonon_moments(V_averaged)
    assert np.isclose(M[0, 1], 0)
    assert not is_entangled(V_averaged)
    assert mutual_information(V_averaged) > 0


@pytest.mark.parametrize('ratio', [1.117, 3.0])
def test_rotated_axes_make_the_anisotropic_ground_state_look_entangled(ratio):
    """The principal-axis ground state is a product state, but not along axes at 45° to them"""
    V = np.eye(4)
    assert log_negativity(rotate_transverse_axes(V, ratio, 1.0, 0.0)) == 0
    assert np.isclose(log_negativity(rotate_transverse_axes(V, ratio, 1.0, np.pi/4)), 0.5*np.log(ratio))


def random_two_mode_state(rng):
    """Thermal state → two-mode squeezer → beam splitter → local squeezers → local rotations"""
    V = thermal_covariance(rng.uniform(0, 1.5, size=2))
    r = rng.uniform(0, 1.2)
    c, s = np.cosh(r), np.sinh(r)
    I, Z = np.eye(2), np.diag([1, -1])
    two_mode_squeezer = np.block([[c*I, s*Z], [s*Z, c*I]])
    θ = rng.uniform(0, np.pi)
    beam_splitter_matrix = np.block([[np.cos(θ)*I, np.sin(θ)*I], [-np.sin(θ)*I, np.cos(θ)*I]])
    r1, r2 = rng.normal(0, 0.5, size=2)
    local_squeezers = np.diag([np.exp(-r1), np.exp(r1), np.exp(-r2), np.exp(r2)])
    S = local_squeezers @ beam_splitter_matrix @ two_mode_squeezer
    return local_rotation(S @ V @ S.T, rng.uniform(0, 2*np.pi, size=2))


def test_optimised_duan_and_hillery_zubairy_are_the_same_test():
    """Both see only n_1, n_2 and |<b1 b2>|, so neither sees local squeezing; PPT does"""
    rng = np.random.default_rng(1)
    agree, ppt_only = 0, 0
    for _ in range(30):
        V = random_two_mode_state(rng)
        assert is_physical(V)
        hz = hillery_zubairy_ratio(V)
        if abs(np.log(hz)) < 0.05:
            continue  # too close to the boundary for a numerical optimiser to settle
        duan = optimised_duan_criterion(V)[0]
        assert (duan < 1) == (hz > 1)
        agree += 1
        ppt_only += is_entangled(V) and hz <= 1
    assert agree >= 20
    assert ppt_only > 0

"""
Time-domain covariance dynamics, the multi-tone (Floquet) drive and the two-tone entanglement scheme

The anchors, in order of how much they rely on the new code:
  1. levisim's own single-tone drift matrix, solved as a Lyapunov equation, reproduces the
     published 2D coherent-scattering state of Deplano et al., Nat. Commun. 16, 4215 (2025).
  2. MultiToneDrive with ONE tone is that same model moved to the cavity frame.
  3. With two tones, limits with known answers: the ideal limit is the two-mode squeezed vacuum,
     the RWA is exact when sidebands are resolved, degenerate modes leave a dark mode, a blue
     tone stronger than the red one amplifies, and the published noise level does not entangle.
"""
import numpy as np
import pytest

from levisim.core.units import kHz
from levisim.analysis.gaussian_states import (
    common_phase_averaged,
    gaussian_discord,
    is_entangled,
    log_negativity,
    purity,
    symplectic_occupations,
)
from levisim.multiphysics.covariance import (
    dimension_covariance,
    quadrature_drift_matrix,
    steady_state_covariance,
)
from levisim.multiphysics.multitone import MultiToneDrive, Tone, common_period
from levisim.multiphysics.two_tone import (
    analytic_log_negativity,
    analytic_optimum,
    analytic_threshold_cooperativity,
    deplano2025_multitrap,
    quantum_cooperativity,
    two_tone_drive,
    two_tone_rwa_steady_state,
    two_tone_steady_state,
)


def rwa_log_negativity(mtrap, ratio, **kwargs):
    stable, _, V = two_tone_rwa_steady_state(mtrap, ratio, **kwargs)
    return log_negativity(V) if stable else np.nan


# ----------------------------------------------------------------------------------------------
# 1. single tone
# ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize('system', ['1×2', '2×2'])
def test_quadrature_drift_matrix_is_real(make_phase_dependent_mtrap, system):
    mtrap = make_phase_dependent_mtrap(system, load_in_power_series=False)
    A = quadrature_drift_matrix(mtrap.drift_matrix)
    assert np.isrealobj(A)
    assert A.shape == (mtrap.num_operators, mtrap.num_operators)


def test_quadrature_drift_matrix_is_real_with_z_and_physical_couplings(make_mtrap):
    # the phase-dependent toy parameters only define x, y couplings, so use the physics model
    mtrap = make_mtrap('1×3')
    assert np.isrealobj(quadrature_drift_matrix(mtrap.drift_matrix))


def test_single_tone_steady_state_reproduces_deplano2025():
    """
    Published (Table 1 parameters, Eq. 4 and text): n_x = 0.55 ± 0.03, n_y = 0.74 ± 0.04,
    purity μ = 0.209 ± 0.013, discord D_X|Y = 0.0423 ± 0.0007, D_Y|X = 0.0471 ± 0.0012, and no
    entanglement. The paper's bath model is momentum diffusion only; levisim damps both
    quadratures, which shifts the covariance entries by ~1%.
    """
    mtrap = deplano2025_multitrap()
    V = dimension_covariance(mtrap, steady_state_covariance(mtrap), ['x1', 'y1'])
    published = np.array([
        [2.13, 0, -0.32, -0.59],
        [0, 2.07, 0.52, -0.34],
        [-0.32, 0.52, 2.47, 0],
        [-0.59, -0.34, 0, 2.48],
    ])
    assert np.abs(V - published).max() < 0.04
    assert np.allclose(symplectic_occupations(V), [0.55, 0.74], atol=0.02)
    assert np.isclose(purity(V), 0.209, atol=0.005)
    assert np.isclose(gaussian_discord(V, measured_mode=1), 0.0423, atol=0.001)
    assert np.isclose(gaussian_discord(V, measured_mode=0), 0.0471, atol=0.001)
    assert log_negativity(V) == 0


@pytest.mark.parametrize('system', ['1×2', '2×2'])
def test_one_tone_multitone_drive_is_levisims_model_in_the_cavity_frame(make_phase_dependent_mtrap, system):
    mtrap = make_phase_dependent_mtrap(system, load_in_power_series=False)
    V_levisim = steady_state_covariance(mtrap)
    drive = MultiToneDrive(mtrap, [Tone(Δ=mtrap.cavity.Δ)])
    steady_state = drive.periodic_steady_state(num_samples=4)
    assert steady_state.stable
    for V in steady_state.covariances:
        # the frames differ only in the cavity's rotation, so the mechanical block is identical
        mechanical = slice(2, None)
        assert np.abs(V[mechanical, mechanical] - V_levisim[mechanical, mechanical]).max() < 1e-7 * np.abs(V_levisim).max()


def test_frame_offset_is_only_a_change_of_frame():
    """Tones at ω_c ∓ ν + s: the cavity frame and the frame rotating at ω_c + s give the same state"""
    mtrap = deplano2025_multitrap(Γ_scale=0.3, balanced=True)
    ν = (mtrap.dimensions['x1'].Ω + mtrap.dimensions['y1'].Ω) / 2
    tones = two_tone_drive(mtrap, 0.55, common_shift=ν/10)
    cavity_frame = MultiToneDrive(mtrap, tones)
    shifted_frame = MultiToneDrive(mtrap, tones, frame_offset=ν/10)
    assert np.isclose(cavity_frame.period, 10 * shifted_frame.period)
    V_cavity_frame = cavity_frame.periodic_steady_state(num_samples=1).covariances[0]
    V_shifted_frame = shifted_frame.periodic_steady_state(num_samples=1).covariances[0]
    assert np.allclose(V_cavity_frame, V_shifted_frame, rtol=1e-7, atol=1e-7)


def test_common_period():
    assert np.isclose(common_period([3*kHz, -3*kHz]), 1 / (3e3))
    assert np.isclose(common_period([2*kHz, 3*kHz]), 1 / 1e3)
    with pytest.raises(ValueError):
        common_period([1*kHz, np.sqrt(2)*kHz], max_denominator=50)


# ----------------------------------------------------------------------------------------------
# 2. two tones: limits with known answers
# ----------------------------------------------------------------------------------------------
def test_ideal_limit_is_the_two_mode_squeezed_vacuum():
    """Negligible heating and resolved sidebands: E_N → 2r with tanh r = G_+/G_-"""
    mtrap = deplano2025_multitrap(Γ_scale=1e-4, balanced=True, Ωx=1222.17*kHz, Ωy=1209.37*kHz)
    result = two_tone_steady_state(mtrap, 0.5, num_samples=4)
    assert np.isclose(result.E_N, 2*np.arctanh(0.5), rtol=0.01)
    assert max(result.bogoliubov) < 0.01


def test_rwa_is_exact_when_sidebands_are_resolved():
    """κ/Ω̄ = 0.05: RWA and the full Floquet model agree. At the experiment's κ/Ω̄ ≈ 0.5 they do not."""
    resolved = deplano2025_multitrap(Γ_scale=0.3, balanced=True, Ωx=1222.17*kHz, Ωy=1209.37*kHz)
    assert np.isclose(two_tone_steady_state(resolved, 0.55, num_samples=4).E_N, rwa_log_negativity(resolved, 0.55), rtol=0.01)
    experiment = deplano2025_multitrap(Γ_scale=0.3, balanced=True)
    assert two_tone_steady_state(experiment, 0.55, num_samples=8).E_N < 0.85 * rwa_log_negativity(experiment, 0.55)


def test_analytic_estimate_matches_rwa_at_high_cooperativity():
    mtrap = deplano2025_multitrap(Γ_scale=0.03, balanced=True)
    C = quantum_cooperativity(mtrap)
    for ratio in [0.5, 0.65, 0.8]:
        assert np.isclose(analytic_log_negativity(np.arctanh(ratio), C), rwa_log_negativity(mtrap, ratio), rtol=0.03)


def test_analytic_optimum_closed_forms():
    for C in [6, 10, 30, 100]:
        r, E_N = analytic_optimum(C)
        assert np.isclose(r, np.log(1 + C) / 4, atol=1e-4)
        assert np.isclose(E_N, np.log(C / (2*np.sqrt(1 + C))), atol=1e-7)
        r, E_N = analytic_optimum(C, fixed_total_power=False)
        assert np.isclose(r, np.log(1 + 2*C) / 4, atol=1e-4)
        assert np.isclose(E_N, np.log(C / (1 + np.sqrt(1 + 2*C))), atol=1e-7)


def test_analytic_thresholds():
    assert np.isclose(analytic_threshold_cooperativity(fixed_total_power=True), 4.83, atol=0.01)
    assert np.isclose(analytic_threshold_cooperativity(fixed_total_power=False), 4.00, atol=0.01)
    assert analytic_optimum(4.5)[1] == 0
    r, E_N = analytic_optimum(100)
    assert 0.5 < r < 1.5 and E_N > 1


def test_published_noise_level_does_not_entangle():
    """At the Deplano et al. (2025) heating rates C ≈ 2.9, below every threshold"""
    mtrap = deplano2025_multitrap(balanced=True)
    assert quantum_cooperativity(mtrap) < analytic_threshold_cooperativity()
    for ratio in np.arange(0.1, 0.95, 0.1):
        assert two_tone_steady_state(mtrap, ratio, num_samples=4).E_N == 0
        assert rwa_log_negativity(mtrap, ratio) == 0


def test_a_third_of_the_published_heating_entangles():
    result = two_tone_steady_state(deplano2025_multitrap(Γ_scale=0.3, balanced=True), 0.55)
    assert result.stable
    assert 0.2 < result.E_N_min <= result.E_N_max < 0.3


def test_degenerate_modes_leave_a_dark_bogoliubov_mode():
    """For Ω_x = Ω_y one combination of the Bogoliubov modes decouples and heats to the bath"""
    Ω = 115.77*kHz
    result = two_tone_steady_state(deplano2025_multitrap(Γ_scale=0.1, balanced=True, Ωx=Ω, Ωy=Ω), 0.5, num_samples=4)
    assert result.floquet_multiplier > 1 - 1e-6
    assert result.E_N == 0
    assert min(result.mean_phonons) > 1e6


def test_blue_tone_stronger_than_red_amplifies():
    result = two_tone_steady_state(deplano2025_multitrap(Γ_scale=0.3, balanced=True), 1.2, num_samples=4)
    assert not result.stable
    assert result.floquet_multiplier > 1


def test_unsuppressed_intensity_beat_makes_the_trap_unstable():
    """Co-polarised tones beat at Ω_x + Ω_y; the trap must not see more than a few % of it"""
    mtrap = deplano2025_multitrap(Γ_scale=0.3, balanced=True)
    assert not two_tone_steady_state(mtrap, 0.55, beat_visibility=1.0, num_samples=4).stable
    assert two_tone_steady_state(mtrap, 0.55, beat_visibility=0.1, num_samples=4).E_N == 0
    assert two_tone_steady_state(mtrap, 0.55, beat_visibility=0.01, num_samples=8).E_N > 0.2


def test_only_the_tone_difference_is_critical():
    """The difference must match Ω_x + Ω_y to ~1 kHz; shifting both tones by 5 kHz costs little"""
    mtrap = deplano2025_multitrap(Γ_scale=0.3, balanced=True)
    E_N = two_tone_steady_state(mtrap, 0.55, num_samples=4).E_N
    # offset moves the tones apart symmetrically: the difference is off by 2 kHz
    assert two_tone_steady_state(mtrap, 0.55, offset=1*kHz, num_samples=4).E_N < 0.5 * E_N
    for shift in (-5*kHz, 5*kHz):
        assert two_tone_steady_state(mtrap, 0.55, common_shift=shift, num_samples=4).E_N > 0.85 * E_N
        assert rwa_log_negativity(mtrap, 0.55, common_shift=shift) > 0.9 * rwa_log_negativity(mtrap, 0.55)


def test_entanglement_does_not_depend_on_the_relative_tone_phase():
    """The relative phase only rotates the squeezed quadratures, which is a local unitary"""
    mtrap = deplano2025_multitrap(Γ_scale=0.3, balanced=True)
    values = [rwa_log_negativity(mtrap, 0.55, relative_phase=phase) for phase in (0.0, 1.3, 3.0)]
    assert np.allclose(values, values[0])


def test_entanglement_needs_a_phase_reference():
    """Averaging the lab-frame state over the period (no phase reference) leaves it separable"""
    result = two_tone_steady_state(deplano2025_multitrap(Γ_scale=0.3, balanced=True), 0.55)
    assert all(is_entangled(V) for V in result.mechanical_covariances)
    assert not is_entangled(np.mean(result.mechanical_covariances, axis=0))
    assert not is_entangled(common_phase_averaged(result.mechanical_covariances[0]))

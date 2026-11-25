"""
Entanglement outputs for directly coupled particles (covariance with and without the cavity, exact
Lyapunov evolution, particle bipartitions) and the decoherence budget
"""
import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.special import zeta

from levisim.analysis.gaussian_states import (
    is_physical,
    squeezed_thermal_covariance,
    two_mode_squeezed_vacuum,
)
from levisim.core.constants import c, e, kB, ħ, π
from levisim.core.units import nm, μm
from levisim.multiphysics.covariance import (
    covariance_evolution,
    evolve_covariance,
    particle_log_negativity,
    particle_mode_indices,
    product_state_covariance,
    quadrature_matrices,
    steady_state_covariance,
)
from levisim.multiphysics.decoherence import (
    blackbody_localisation_parameters,
    decoherence_budget,
    electric_field_noise_heating_rate,
)
from levisim.multiphysics.multitrap import MultiTrap
from levisim.multiphysics.utils.shelf import (
    DEFAULT_cavity_kwargs,
    DEFAULT_gas_kwargs,
    PHASE_DEPENDENT_toy_params,
)
from levisim.physics.bead import Bead
from levisim.physics.interactions import Coulomb, G_newton, NewtonianGravity
from levisim.physics.material import Material
from levisim.physics.particle import Sphere

OSMIUM = Material('osmium', ρ=22590.0, permittivity=4.0)


def gravity_pair(ω=0.01, R=1e-3, separation_factor=2.1, P=0.0, **kwargs):
    """Two osmium spheres coupled by gravity along x (Krisnanda et al., npj Quantum Inf. 6, 12 (2020))"""
    particles = [Sphere(r=R, material=OSMIUM), Sphere(r=R, material=OSMIUM)]
    L = separation_factor * R
    kwargs.setdefault('photon_recoil', False)
    kwargs.setdefault('static_cavity_shifts', False)
    return MultiTrap(
        num_traps=2, Cartesian_axes_per_trap=1, beads=particles, trap_positions=[(0, 0, 0), (L, 0, 0)],
        trap_kwargs=[{'override_Ω_dict': {'x': ω, 'y': ω, 'z': ω}}] * 2, interactions=[NewtonianGravity()],
        gas_kwargs={'P': P}, **kwargs,
    ), 2 * G_newton * particles[0].m / (ω**2 * L**3)


# --------------------------------------------------------------------------------------------------
# Gaussian states
# --------------------------------------------------------------------------------------------------
def test_squeezed_thermal_covariance():
    r = 15 * math.log(10) / 20
    assert r == pytest.approx(1.727, abs=1e-3)
    V = squeezed_thermal_covariance(0.3, r, 0.4)
    assert np.linalg.det(V) == pytest.approx(1.6**2, rel=1e-12)
    assert is_physical(V)
    assert np.allclose(squeezed_thermal_covariance(2.0), 5 * np.eye(2))
    assert squeezed_thermal_covariance(0, 0.5)[0, 0] == pytest.approx(np.exp(-1.0))


def test_particle_log_negativity_of_a_two_mode_squeezed_vacuum():
    mtrap, _ = gravity_pair()
    V = two_mode_squeezed_vacuum(0.4)
    assert particle_log_negativity(mtrap, V, 1, cavity=False) == pytest.approx(0.8, rel=1e-12)
    assert particle_mode_indices(mtrap, 2, cavity=True) == [2]


# --------------------------------------------------------------------------------------------------
# covariance with and without the cavity
# --------------------------------------------------------------------------------------------------
def test_default_steady_state_is_unchanged_and_cavity_false_drops_the_cavity():
    mtrap = MultiTrap(num_traps=2, toy_params=PHASE_DEPENDENT_toy_params, gas_kwargs=DEFAULT_gas_kwargs,
                      cavity_kwargs=DEFAULT_cavity_kwargs)
    A, D = quadrature_matrices(mtrap)
    assert A.shape == (10, 10)
    V = steady_state_covariance(mtrap)
    assert np.allclose(A @ V + V @ A.T + D, 0, atol=1e-9 * np.abs(D).max())
    A_mech, D_mech = quadrature_matrices(mtrap, cavity=False)
    assert np.array_equal(A_mech, A[2:, 2:]) and np.array_equal(D_mech, D[2:, 2:])
    V_mech = steady_state_covariance(mtrap, cavity=False)
    # without the cavity each toy mode is alone in its bath
    for k, label in enumerate(mtrap.labels):
        dim = mtrap.dimensions[label]
        assert V_mech[2*k, 2*k] == pytest.approx(2 * dim.noise_phonons_exact + 1, rel=1e-9)
    heated = steady_state_covariance(mtrap, cavity=False, extra_heating={'x1': 10 * mtrap.dimensions['x1'].Γ})
    assert heated[0, 0] == pytest.approx(V_mech[0, 0] + 20, rel=1e-9)


def test_covariance_evolution_matches_a_direct_ode_and_the_steady_state():
    mtrap = MultiTrap(num_traps=2, toy_params=PHASE_DEPENDENT_toy_params, gas_kwargs=DEFAULT_gas_kwargs,
                      cavity_kwargs=DEFAULT_cavity_kwargs)
    A, D = quadrature_matrices(mtrap, cavity=False)
    V0 = product_state_covariance(mtrap, 0.0, cavity=False)
    times = np.array([0.0, 1e-6, 3e-6, 1e-5])
    V = covariance_evolution(A, D, V0, times)
    n = A.shape[0]
    solution = solve_ivp(lambda t, v: (A @ v.reshape(n, n) + v.reshape(n, n) @ A.T + D).ravel(),
                         (0, times[-1]), V0.ravel(), t_eval=times, rtol=1e-11, atol=1e-11, method='DOP853')
    for k in range(len(times)):
        np.testing.assert_allclose(V[k], solution.y[:, k].reshape(n, n), rtol=1e-7, atol=1e-7)
    long = covariance_evolution(A, D, V0, [0.0, 50 / mtrap.dimensions['x1'].Γ])[-1]
    V_ss = steady_state_covariance(mtrap, cavity=False)
    # ~10⁸ oscillation periods in ~30 exact doublings: round-off accumulates to ~1e-7
    np.testing.assert_allclose(long, V_ss, rtol=1e-6, atol=1e-12 * np.abs(V_ss).max())
    with pytest.raises(ValueError):
        covariance_evolution(A, D, V0, [1.0, 0.5])


# --------------------------------------------------------------------------------------------------
# gravity-mediated entanglement: Krisnanda et al., npj Quantum Inf. 6, 12 (2020)
# --------------------------------------------------------------------------------------------------
def test_gravity_coupling_and_frequency_shift_from_the_interaction():
    ω = 0.01
    mtrap, η = gravity_pair(ω)
    # η = 8πGρ/(3·2.1³·ω²) = 1.36e-6/ω² for osmium (the paper's value)
    assert η * ω**2 == pytest.approx(1.36e-6, rel=5e-3)
    assert mtrap.dimensions['x1'].Ω == pytest.approx(ω * np.sqrt(1 - η), rel=1e-12)
    unshifted, _ = gravity_pair(ω, interaction_frequency_shifts=False)
    assert unshifted.dimensions['x1'].Ω == ω


def test_ground_state_entanglement_reaches_eta():
    """
    E_max ≈ η/ln 2 bits = η nats, at ωt ≈ π/(2(1 - η)), from the ground states without damping

    With the frequency shift ω²(1 - η) the model is exactly the paper's Eq. (4); the exact maximum
    exceeds the paper's leading-order estimate η by O(η²) (here η·1.4 %, with η = 0.0136)
    """
    ω = 0.01
    mtrap, η = gravity_pair(ω)
    t_star = π / (2 * (1 - η) * ω)
    times = np.linspace(0, 2 * t_star, 1601)
    V = evolve_covariance(mtrap, product_state_covariance(mtrap, 0.0, cavity=False), times, cavity=False)
    E = particle_log_negativity(mtrap, V, 1, cavity=False)
    assert E.max() == pytest.approx(η, rel=2 * η)
    assert times[np.argmax(E)] == pytest.approx(t_star, rel=0.02)
    # thermal initial states: amplitude η - ln(2n + 1) (η/ln 2 - log2(2n + 1) bits)
    for n in (0.001, 0.003):
        V = evolve_covariance(mtrap, product_state_covariance(mtrap, n, cavity=False), times, cavity=False)
        assert particle_log_negativity(mtrap, V, 1, cavity=False).max() == pytest.approx(η - np.log(2*n + 1), abs=2 * η**2)


def test_squeezed_states_reach_the_sum_of_squeezings():
    """E_max ≈ |s_A + s_B| (nats) at t = π/(2ηω), for any squeezing direction (η ≪ s)"""
    ω = 0.01
    mtrap, η = gravity_pair(ω)
    s = 0.5
    times = np.linspace(0, π / (η * ω), 2001)
    for θ in (0.0, π / 4):
        V0 = product_state_covariance(mtrap, 0.0, cavity=False, squeezing={'x1': (s, θ), 'x2': (s, θ)})
        E = particle_log_negativity(mtrap, evolve_covariance(mtrap, V0, times, cavity=False), 1, cavity=False)
        assert E.max() == pytest.approx(2 * s, rel=0.02)


def test_damping_destroys_gravitational_entanglement():
    ω = 0.01
    mtrap, _η = gravity_pair(ω, P=1e-6)
    times = np.linspace(0, π / ω, 401)
    V = evolve_covariance(mtrap, product_state_covariance(mtrap, 0.0, cavity=False), times, cavity=False)
    assert particle_log_negativity(mtrap, V, 1, cavity=False).max() < 1e-12


# --------------------------------------------------------------------------------------------------
# decoherence budget
# --------------------------------------------------------------------------------------------------
def charged_beads():
    beads = [Bead(r=60.1*nm, Φ=0.0, charge=100), Bead(r=60.2*nm, Φ=0.0, charge=0)]
    return MultiTrap(num_traps=2, beads=beads, trap_positions=[(0, 0, 0), (0, 10*μm, 0)], interactions=[Coulomb()],
                     static_cavity_shifts=False)


def test_decoherence_budget_entries():
    mtrap = charged_beads()
    budget = decoherence_budget(mtrap, T_environment=300.0, T_internal=[500.0, 300.0], electric_field_noise=1e-6)
    assert set(budget) == set(mtrap.labels)
    assert budget.included == ('gas', 'photon_recoil')
    for label, rates in budget.items():
        dim = mtrap.dimensions[label]
        assert rates['gas'] == pytest.approx(dim.Γ * dim.noise_phonons_exact, rel=1e-14)
        assert rates['photon_recoil'] == pytest.approx(mtrap.recoil_heating_rate_dict[label], rel=1e-14)
        assert rates['total'] == pytest.approx(sum(v for k, v in rates.items() if k != 'total'), rel=1e-14)
    # emission at T_i = 500 K exceeds absorption at 300 K by (5/3)⁶ for particle 1, equal for particle 2
    x1, x2 = budget['x1'], budget['x2']
    assert x1['blackbody_emission'] / x1['blackbody_absorption'] == pytest.approx((5/3)**6, rel=1e-12)
    assert x2['blackbody_emission'] == pytest.approx(x2['blackbody_absorption'], rel=1e-14)
    # the charge-free particle feels no field noise
    assert x2['electric_field_noise'] == 0
    dim = mtrap.dimensions['x1']
    assert x1['electric_field_noise'] == pytest.approx((100 * e)**2 * 1e-6 / (4 * dim.m * ħ * dim.Ω), rel=1e-14)
    extra = budget.extra_heating()
    assert extra['x1'] == pytest.approx(x1['total'] - x1['gas'] - x1['photon_recoil'], rel=1e-12)
    assert 'electric_field_noise' in budget.table()


def test_blackbody_localisation_parameters_formulas():
    R, T, ε = 50*nm, 4.5, 2.1 + 0.57j
    Λ_sc, Λ_e, Λ_a = blackbody_localisation_parameters(R, T, 200.0, ε)
    response = (ε - 1) / (ε + 2)
    assert Λ_sc == pytest.approx(math.factorial(8) * 8 * zeta(9) * c * R**6 / (9*π) * (kB*T/(ħ*c))**9 * response.real**2, rel=1e-14)
    assert Λ_e == pytest.approx(16 * π**5 * c * R**3 / 189 * (kB*200/(ħ*c))**6 * response.imag, rel=1e-14)
    assert Λ_e / Λ_a == pytest.approx((200 / 4.5)**6, rel=1e-12)
    assert electric_field_noise_heating_rate(e, 1.0, 1.0, 1.0) == pytest.approx(e**2 / (4 * ħ))


def test_budget_heating_enters_the_covariance():
    mtrap = charged_beads()
    budget = decoherence_budget(mtrap, electric_field_noise=1e-3)
    _A, D = quadrature_matrices(mtrap, cavity=False)
    _, D_heated = quadrature_matrices(mtrap, cavity=False, extra_heating=budget.extra_heating())
    k = mtrap.label_to_index['x1']
    assert D_heated[2*k, 2*k] - D[2*k, 2*k] == pytest.approx(2 * budget.extra_heating()['x1'], rel=1e-12)


def test_budget_needs_physical_parameters():
    mtrap = MultiTrap(num_traps=2, toy_params=PHASE_DEPENDENT_toy_params)
    with pytest.raises(ValueError, match='physical'):
        decoherence_budget(mtrap)

"""
Photon-recoil heating in the quantum linear theory: MultiTrap(photon_recoil=...), on by default with
physical parameters

Reference numbers are those of the Delić et al., Science 367, 892 (2020) reproduction notebook
(notebooks/papers on branch round2/paper-reproductions), which added the same heating by hand
with paper_tools.recoil_heating_rates and paper_tools.add_heating.
"""
import numpy as np
import pytest

from levisim.core.constants import π, ħ
from levisim.core.units import cm, degrees, kHz, mbar, mW, nm, μm
from levisim.multiphysics.covariance import mode_index, steady_state_covariance
from levisim.multiphysics.multitrap import MultiTrap
from levisim.physics.bead import Bead
from levisim.physics.tweezer import Tweezer


Ω_DELIC = {'x': 305*kHz, 'y': 275*kHz, 'z': 80*kHz}


def delic_multitrap(Δ_paper, P=1e-6*mbar, axes=3, photon_recoil=True):
    """Delić 2020 at the paper's detuning Δ_paper (> 0 is red there, so Δ = -Δ_paper here)"""
    return MultiTrap(
        tweezers=[Tweezer(Power=400*mW, λ=1064*nm, wx0=0.67*μm, wy0=0.77*μm, θ=90*degrees)],
        beads=[Bead(r=143/2*nm, ρ=1850.0, permittivity_model='Malitson', λ=1064*nm, Φ=π/2)],
        trap_kwargs=[dict(override_Ω_dict=Ω_DELIC)],
        Cartesian_axes_per_trap=axes,
        gas_kwargs=dict(P=P, T=300.0),
        cavity_kwargs=dict(L=1.07*cm, w=41.1*μm, Finesse=73_000, κ=193*kHz, Δ=-Δ_paper),
        photon_recoil=photon_recoil,
        ω_range=np.array([0.0, 1.0]),
    )


def occupation(mtrap, label):
    V = steady_state_covariance(mtrap)
    k = mode_index(mtrap, label)
    return (V[2*k, 2*k] + V[2*k + 1, 2*k + 1] - 2) / 4


def test_on_by_default_with_physical_parameters():
    mtrap = MultiTrap(num_traps=2, Cartesian_axes_per_trap=2)
    explicit = MultiTrap(num_traps=2, Cartesian_axes_per_trap=2, photon_recoil=True)
    assert mtrap.photon_recoil is True and set(mtrap.recoil_heating_rate_dict) == set(mtrap.labels)
    assert mtrap.quantum_factor_per_noise_operator == explicit.quantum_factor_per_noise_operator
    assert np.array_equal(mtrap.noise_matrix, explicit.noise_matrix)


def test_off_gives_the_gas_bath_alone():
    off = MultiTrap(num_traps=2, Cartesian_axes_per_trap=2, photon_recoil=False)
    assert off.photon_recoil is False and off.recoil_heating_rate_dict == {}
    gas_only = [off.quantum_factor_per_noise_operator[0], off.quantum_factor_per_noise_operator[1]]
    for label in off.labels:
        n = off.dimensions[label].noise_phonons_exact
        gas_only += [n + 1, n]
    assert off.quantum_factor_per_noise_operator == gas_only


def test_recoil_changes_the_noise_but_not_the_drift_matrix_or_its_stability():
    on = MultiTrap(num_traps=2, Cartesian_axes_per_trap=2)
    off = MultiTrap(num_traps=2, Cartesian_axes_per_trap=2, photon_recoil=False)
    assert np.array_equal(on.drift_matrix, off.drift_matrix)
    assert np.array_equal(np.linalg.eigvals(on.drift_matrix), np.linalg.eigvals(off.drift_matrix))
    assert not np.array_equal(on.noise_matrix, off.noise_matrix)
    # the light noise is untouched, every mechanical bath occupation is raised
    assert on.quantum_factor_per_noise_operator[:2] == off.quantum_factor_per_noise_operator[:2]
    assert all(a > b for a, b in zip(on.quantum_factor_per_noise_operator[2:], off.quantum_factor_per_noise_operator[2:]))


def test_off_by_default_with_toy_parameters():
    toy_params = dict(Ω_model='custom', toy_Ωjn_dict={'x1': 100*kHz}, gjn_dict={'x1': -10*kHz})
    mtrap = MultiTrap(num_traps=1, Cartesian_axes_per_trap=1, toy_params=toy_params)
    assert mtrap.photon_recoil is False and mtrap.recoil_heating_rate_dict == {}
    off = MultiTrap(num_traps=1, Cartesian_axes_per_trap=1, toy_params=toy_params, photon_recoil=False)
    assert mtrap.quantum_factor_per_noise_operator == off.quantum_factor_per_noise_operator


def test_default_is_left_out_with_a_warning_without_damping_but_explicit_true_raises():
    from loguru import logger
    messages = []
    sink = logger.add(messages.append, level='WARNING', format='{message}')
    try:
        mtrap = MultiTrap(num_traps=1, Cartesian_axes_per_trap=2, gas_kwargs=dict(P=0.0))
    finally:
        logger.remove(sink)
    assert mtrap.photon_recoil is False and mtrap.recoil_heating_rate_dict == {}
    assert any('photon recoil heating is left out' in message for message in messages), messages
    with pytest.raises(ValueError, match='needs a non-zero damping'):
        MultiTrap(num_traps=1, Cartesian_axes_per_trap=2, gas_kwargs=dict(P=0.0), photon_recoil=True)


def test_recoil_rates_match_the_delic_notebook():
    mtrap = delic_multitrap(315*kHz)
    rates = mtrap.recoil_heating_rate_dict
    # 5.59 kHz along x (the paper: 6 kHz calculated, 4.3 ± 1 kHz inferred)
    assert rates['x1'] / kHz == pytest.approx(5.5947, rel=1e-4)
    # c = (2, 1, 7) and x_zpf² ∝ 1/Ω
    assert rates['y1'] / rates['x1'] == pytest.approx(0.5 * Ω_DELIC['x'] / Ω_DELIC['y'], rel=1e-12)
    assert rates['z1'] / rates['x1'] == pytest.approx(3.5 * Ω_DELIC['x'] / Ω_DELIC['z'], rel=1e-12)
    # the bath occupation is raised by Γ_rec/Γ, for both noise factors of each mode
    for label, Γ_rec in rates.items():
        index = mtrap.label_to_noise_index[label]
        n = mtrap.dimensions[label].noise_phonons_exact + Γ_rec / mtrap.dimensions[label].Γ
        assert mtrap.quantum_factor_per_noise_operator[index] == pytest.approx(n + 1, rel=1e-14)
        assert mtrap.quantum_factor_per_noise_operator[index + 1] == pytest.approx(n, rel=1e-14)
    # n_x at 315 kHz: 0.398 with recoil (measured 0.43 ± 0.03), 0.322 without
    assert occupation(mtrap, 'x1') == pytest.approx(0.39756, rel=1e-4)
    assert occupation(delic_multitrap(315*kHz, photon_recoil=False), 'x1') == pytest.approx(0.32189, rel=1e-4)


def test_recoil_limited_minimum_occupation():
    Δs = np.arange(300, 326, 1.0) * kHz
    n_x = np.array([occupation(delic_multitrap(Δ, P=1e-9*mbar, axes=1), 'x1') for Δ in Δs])
    # 0.145 at 312-313 kHz; the paper's "< 1e-8 mbar" curve has 0.154 at 304 kHz
    assert n_x.min() == pytest.approx(0.1454, abs=5e-4)
    assert 310*kHz <= Δs[n_x.argmin()] <= 315*kHz


def test_rate_equals_photon_counting_estimate():
    """Γ_rec,j = (c_j/5)·R·k²·x_zpf² with R = σ_scat·I0/(ħω) the rate of scattered photons"""
    trap = delic_multitrap(315*kHz).traps[0]
    tweezer = trap.tweezer
    R = trap.σ_scattering * tweezer.I0 / (ħ * tweezer.ω)
    for axis, c_j in zip('xyz', (2, 1, 7)):
        estimate = c_j / 5 * R * tweezer.k**2 * ħ / (2 * trap.bead.m * trap.Ωj[axis])
        # σ_scat uses Im(α_eff) and the rate Re(α_eff)²; they differ by the radiation-reaction term
        assert trap.photon_recoil_heating_rates[axis] == pytest.approx(estimate, rel=1e-3)


def test_photon_recoil_needs_physical_parameters():
    toy_params = dict(Ω_model='custom', toy_Ωjn_dict={'x1': 100*kHz}, gjn_dict={'x1': -10*kHz})
    with pytest.raises(ValueError, match='needs physical parameters'):
        MultiTrap(num_traps=1, Cartesian_axes_per_trap=1, toy_params=toy_params, photon_recoil=True)

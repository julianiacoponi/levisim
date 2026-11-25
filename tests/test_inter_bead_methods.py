"""
Every QLT method with direct couplings between beads (interactions, toy g_direct), reciprocal or not

The power-series and analytic methods were generalised (round 5) instead of raising: Rjk includes
the inter-bead gjk, and the analytic closed form uses the full mechanical response
W = (diag(1/μ) - i·g)⁻¹ with the rank-2 cavity part eliminated by Woodbury. Each must agree with
the Linear Amplifier Model point by point.
"""
import numpy as np
import pytest

from levisim.core.units import kHz, nm, μm
from levisim.multiphysics.enums.quantum_linear_theory_method import (
    QuantumLinearTheoryMethod as Q,
)
from levisim.multiphysics.multitrap import MultiTrap
from levisim.multiphysics.utils.shelf import (
    DEFAULT_cavity_kwargs,
    DEFAULT_gas_kwargs,
    OFF_NODE_toy_params_pairs,
)
from levisim.physics.bead import Bead
from levisim.physics.interactions import Coulomb

OMEGA_RANGE = np.arange(250, 320, 0.5) * kHz
TOLERANCE = 1e-12


def toy(num_traps, axes, g_direct):
    return MultiTrap(
        num_traps=num_traps, Cartesian_axes_per_trap=axes, ω_range=OMEGA_RANGE,
        toy_params=dict(OFF_NODE_toy_params_pairs, g_direct=g_direct),
        gas_kwargs=DEFAULT_gas_kwargs, cavity_kwargs=DEFAULT_cavity_kwargs,
        load_in_power_series=(axes == 2 and num_traps <= 3),
    )


def coulomb_three():
    beads = [Bead(r=60.1*nm, Φ=0.7, charge=300), Bead(r=60.2*nm, Φ=1.1, charge=-200), Bead(r=60.3*nm, Φ=1.3, charge=0)]
    return MultiTrap(
        num_traps=3, beads=beads, trap_positions=[(0, 0, 0), (0, 5*μm, 0), (0, 10*μm, 0)],
        interactions=[Coulomb()], ω_range=np.linspace(100, 320, 400) * kHz, load_in_power_series=True,
    )


CASES = {
    'toy 2×2 non-reciprocal': lambda: toy(2, 2, {('x1', 'x2'): (3*kHz, -1*kHz), ('y1', 'x2'): 2*kHz}),
    'toy 3×2 two groups': lambda: toy(3, 2, {('x1', 'y2'): (3*kHz, -1*kHz)}),
    'toy 3×2 one group': lambda: toy(3, 2, {('x1', 'x3'): (2*kHz, 4*kHz), ('y2', 'y3'): 1.5*kHz}),
    'toy 2×3': lambda: toy(2, 3, {('z1', 'z2'): 2*kHz, ('x1', 'y2'): (1*kHz, 0)}),
    'toy 2×2 asymmetric intra-bead': lambda: toy(2, 2, {('x1', 'y1'): (3*kHz, -1*kHz)}),
    'physical 3×2 Coulomb off node': coulomb_three,
}


@pytest.fixture(scope='module', params=list(CASES))
def mtrap(request):
    return CASES[request.param]()


def relative(a, b):
    return np.max(np.abs(np.asarray(a) / np.asarray(b) - 1))


def test_structure_is_general(mtrap):
    is_bead_local, groups = mtrap.mechanical_coupling_structure()
    assert not is_bead_local
    assert sorted(np.concatenate(groups)) == list(range(mtrap.num_dimensions))


@pytest.mark.parametrize('method', [m for m in Q if m != Q.LINEAR_AMPLIFIER_MODEL], ids=lambda m: m.name)
def test_position_spectra_agree_with_LAM(mtrap, method):
    if method == Q.POWER_SERIES_HYBRIDISATION and not getattr(mtrap, 'deranged_determinant_by_order', None):
        pytest.skip('no shipped power series for this system size')
    reference = mtrap.all_position_power_spectra(Q.LINEAR_AMPLIFIER_MODEL)[0]
    assert relative(mtrap.all_position_power_spectra(method)[0], reference) < TOLERANCE


@pytest.mark.parametrize('method', [m for m in Q if m not in (Q.LINEAR_AMPLIFIER_MODEL, Q.TNT_LINEAR_AMPLIFIER_MODEL)],
                         ids=lambda m: m.name)
def test_optical_spectra_agree_with_LAM(mtrap, method):
    if method == Q.POWER_SERIES_HYBRIDISATION and not getattr(mtrap, 'deranged_determinant_by_order', None):
        pytest.skip('no shipped power series for this system size')
    assert relative(mtrap.homodyne_spectrum(method), mtrap.homodyne_spectrum(Q.LINEAR_AMPLIFIER_MODEL)) < TOLERANCE
    assert relative(mtrap.heterodyne_spectrum(method), mtrap.heterodyne_spectrum(Q.LINEAR_AMPLIFIER_MODEL)) < TOLERANCE


def test_analytic_response_matrix_is_the_dense_inverse(mtrap):
    """H from Woodbury and Schur complements over groups equals inv(W⁻¹ - iC), also on a resonance"""
    G = mtrap.G_array
    for ω in (283.25*kHz, mtrap.dimensions[mtrap.labels[0]].Ω):
        kernel = mtrap.analytic_kernel(ω)
        χC, χC_star = mtrap.cavity.χC(ω), np.conjugate(mtrap.cavity.χC(-ω))
        W_inv = np.zeros((mtrap.num_dimensions,) * 2, dtype=complex)
        for indices, block in zip(kernel['groups'], kernel['W_inv_blocks']):
            W_inv[np.ix_(indices, indices)] = block
        C = 1j * χC * np.outer(np.conjugate(G), G) - 1j * χC_star * np.outer(G, np.conjugate(G))
        dense = np.linalg.inv(W_inv - 1j * C)
        H = mtrap.analytic_response_matrix(ω)
        assert np.max(np.abs(H - dense)) < 1e-12 * np.max(np.abs(dense))
        np.testing.assert_allclose(
            mtrap.analytic_hybridisation_matrix(ω), np.linalg.inv(mtrap.derangement_matrix(ω)),
            rtol=0, atol=1e-12 * np.max(np.abs(np.linalg.inv(mtrap.derangement_matrix(ω)))),
        )


def test_collective_sums_and_normalisation(mtrap):
    """D(0), D̄(0) and M_norm from their definitions; P in the cancellation-free form"""
    ω = 283.25*kHz
    kernel = mtrap.analytic_kernel(ω)
    G, W = mtrap.G_array, kernel['W']
    D0, D0_bar = G @ W @ np.conjugate(G), np.conjugate(G) @ W @ G
    D2, Dm2 = G @ W @ G, np.conjugate(G) @ W @ np.conjugate(G)
    assert mtrap.D(0, ω) == pytest.approx(D0, rel=1e-13)
    assert mtrap.D0_bar(ω) == pytest.approx(D0_bar, rel=1e-13)
    assert mtrap.P(ω) == pytest.approx(D0 * D0_bar - D2 * Dm2, rel=1e-10)
    χC, χC_star, ξC = mtrap.cavity.χC(ω), np.conjugate(mtrap.cavity.χC(-ω)), mtrap.cavity.ξC(ω)
    determinant = (1 + χC * D0) * (1 - χC_star * D0_bar) + ξC * D2 * Dm2
    assert mtrap.M_norm(ω) == pytest.approx(determinant, rel=1e-12)


def test_bead_local_systems_keep_the_original_kernel():
    mtrap = MultiTrap(num_traps=2, ω_range=OMEGA_RANGE, toy_params=OFF_NODE_toy_params_pairs,
                      gas_kwargs=DEFAULT_gas_kwargs, cavity_kwargs=DEFAULT_cavity_kwargs)
    is_bead_local, _groups = mtrap.mechanical_coupling_structure()
    assert is_bead_local
    kernel = mtrap.analytic_kernel(283.25*kHz)
    assert kernel['is_bead_local'] and kernel['D0_bar'] is kernel['D'][0]
    assert kernel['h_left'] is kernel['h']

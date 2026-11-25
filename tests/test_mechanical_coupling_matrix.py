"""
MultiTrapBase.mechanical_coupling_matrix: the one place the direct mechanical couplings live

Entry [j, k] is the rate at which dimension k drives dimension j (H/ħ ∋ -gjk·qj·qk). The drift
matrix, the hybridisation functions and the analytic methods all read it. For the systems levisim
builds it holds the intra-bead gxy, gxz, gyz and zeros elsewhere; the drift matrix built from it is
bit-identical to the one built with the old hard-coded index offsets (checked against main when it
was introduced, and here against the documented block structure in test_drift_matrix.py).
"""
import numpy as np
import pytest

from levisim.core.constants import π
from levisim.core.units import kHz, nm
from levisim.multiphysics.enums.quantum_linear_theory_method import (
    QuantumLinearTheoryMethod,
)
from levisim.multiphysics.utils.shelf import OFF_NODE_toy_params_pairs
from levisim.physics.bead import Bead

LAM = QuantumLinearTheoryMethod.LINEAR_AMPLIFIER_MODEL
NUMERICAL_H = QuantumLinearTheoryMethod.NUMERICAL_HYBRIDISATION
ANALYTIC_INSIDE = QuantumLinearTheoryMethod.ANALYTIC_INSIDE_CAVITY
ANALYTIC_OUTPUT = QuantumLinearTheoryMethod.ANALYTIC_CAVITY_OUTPUT
OMEGA_RANGE = np.arange(250, 320, 5.0) * kHz


def expected_matrix(mtrap):
    """gxy, gxz, gyz between axes of one bead (only those the axes per trap use), zero elsewhere"""
    used = {1: set(), 2: {'xy'}, 3: {'xy', 'xz', 'yz'}}[mtrap.Cartesian_axes_per_trap]
    expected = np.zeros((mtrap.num_dimensions, mtrap.num_dimensions))
    for jj, label_j in enumerate(mtrap.labels):
        for kk, label_k in enumerate(mtrap.labels):
            axes = ''.join(sorted(label_j[0] + label_k[0]))
            if jj != kk and label_j[1:] == label_k[1:] and axes in used:
                expected[jj, kk] = getattr(mtrap.dimensions[label_j], f'g{axes}')
    return expected


@pytest.mark.parametrize('system', ['1×1', '2×1', '1×2', '2×2', '1×3', '2×3', '3×2'])
def test_toy_matrix_holds_intra_bead_couplings_only(make_mtrap, system):
    mtrap = make_mtrap(system, ω_range=OMEGA_RANGE, toy_params=OFF_NODE_toy_params_pairs)
    matrix = mtrap.mechanical_coupling_matrix
    assert matrix.shape == (mtrap.num_dimensions, mtrap.num_dimensions)
    assert matrix.dtype == float
    np.testing.assert_array_equal(matrix, expected_matrix(mtrap))
    np.testing.assert_array_equal(matrix, matrix.T)
    if mtrap.Cartesian_axes_per_trap >= 2:
        assert matrix[mtrap.label_to_index['x1'], mtrap.label_to_index['y1']] == OFF_NODE_toy_params_pairs['toy_gxy']


@pytest.mark.parametrize('system', ['2×2', '2×3'])
def test_physical_matrix_off_node(make_mtrap, system):
    num_traps = int(system[0])
    beads = [Bead(r=(60.1 + 0.1 * ii) * nm, Φ=π / 4 + 0.1 * ii) for ii in range(num_traps)]
    mtrap = make_mtrap(system, ω_range=OMEGA_RANGE, beads=beads)
    np.testing.assert_array_equal(mtrap.mechanical_coupling_matrix, expected_matrix(mtrap))
    assert np.any(mtrap.mechanical_coupling_matrix != 0)


def test_analytic_intra_bead_matrix_is_the_mechanical_coupling_matrix(make_mtrap):
    mtrap = make_mtrap('2×3', ω_range=OMEGA_RANGE, toy_params=OFF_NODE_toy_params_pairs)
    np.testing.assert_array_equal(mtrap.intra_bead_coupling_matrix, mtrap.mechanical_coupling_matrix)


def test_drift_matrix_is_written_from_the_matrix_including_off_bead_and_asymmetric_entries(make_mtrap):
    """An inter-bead, non-reciprocal entry lands where the H/ħ ∋ -gjk·qj·qk convention puts it"""
    mtrap = make_mtrap('2×2', ω_range=OMEGA_RANGE, toy_params=OFF_NODE_toy_params_pairs)
    before = mtrap.drift_matrix.copy()
    matrix = mtrap.mechanical_coupling_matrix.copy()
    jj, kk = mtrap.label_to_index['x1'], mtrap.label_to_index['y2']
    matrix[jj, kk] = 3 * kHz  # y2 drives x1 ...
    matrix[kk, jj] = -1 * kHz  # ... and x1 drives y2 at a different rate
    mtrap.mechanical_coupling_matrix = matrix
    mtrap.setup_linear_amplifier_model_matrices()
    after = mtrap.drift_matrix

    row, col = mtrap.label_to_noise_index['x1'], mtrap.label_to_noise_index['y2']
    assert after[row, col] == after[row, col + 1] == 1j * 3 * kHz
    assert after[row + 1, col] == after[row + 1, col + 1] == -1j * 3 * kHz
    assert after[col, row] == after[col, row + 1] == -1j * 1 * kHz
    assert after[col + 1, row] == after[col + 1, row + 1] == 1j * 1 * kHz
    changed = np.argwhere(after != before)
    assert len(changed) == 8


def test_off_bead_entry_is_used_by_hybridisation_and_analytic_methods(make_mtrap):
    """An inter-bead entry (formerly NotImplementedError) enters Rjk and the analytic closed form"""
    mtrap = make_mtrap('2×2', ω_range=OMEGA_RANGE, toy_params=OFF_NODE_toy_params_pairs)
    R_before = mtrap.hybridisation_function_value('x1', 'x2', 300 * kHz)
    matrix = mtrap.mechanical_coupling_matrix.copy()
    matrix[mtrap.label_to_index['x1'], mtrap.label_to_index['x2']] = 1 * kHz
    mtrap.mechanical_coupling_matrix = matrix
    mtrap.setup_linear_amplifier_model_matrices()
    μ, M = mtrap.dimensions['x1'].μm(300 * kHz), mtrap.dimensions['x1'].M(300 * kHz)
    R_after = mtrap.hybridisation_function_value('x1', 'x2', 300 * kHz)
    assert R_after - R_before == pytest.approx(1j * μ / M * 1 * kHz, rel=1e-12)
    lam = np.asarray(mtrap.homodyne_spectrum(LAM))
    mtrap._analytic_kernel_cache = None
    analytic = np.asarray(mtrap.homodyne_spectrum(ANALYTIC_INSIDE))
    np.testing.assert_allclose(analytic, lam, rtol=1e-12)


def test_asymmetric_intra_bead_entry_is_kept_by_the_analytic_methods(make_mtrap):
    mtrap = make_mtrap('1×2', ω_range=OMEGA_RANGE, toy_params=OFF_NODE_toy_params_pairs)
    matrix = mtrap.mechanical_coupling_matrix.copy()
    matrix[0, 1] += 1 * kHz
    mtrap.mechanical_coupling_matrix = matrix
    mtrap.setup_linear_amplifier_model_matrices()
    assert mtrap.mechanical_coupling_structure()[0] is False
    np.testing.assert_array_equal(mtrap.intra_bead_coupling_matrix, matrix)
    np.testing.assert_allclose(
        np.asarray(mtrap.homodyne_spectrum(ANALYTIC_OUTPUT)), np.asarray(mtrap.homodyne_spectrum(LAM)), rtol=1e-12,
    )


def test_asymmetric_intra_bead_coupling_agrees_between_LAM_and_numerical_hybridisation(make_mtrap):
    """The hybridisation function reads gjk = matrix[j, k], the rate at which k drives j"""
    mtrap = make_mtrap('1×2', ω_range=OMEGA_RANGE, toy_params=OFF_NODE_toy_params_pairs)
    matrix = mtrap.mechanical_coupling_matrix.copy()
    matrix[0, 1] = -5 * kHz
    matrix[1, 0] = +1 * kHz
    mtrap.mechanical_coupling_matrix = matrix
    mtrap.setup_linear_amplifier_model_matrices()
    lam, _ = mtrap.all_position_power_spectra(LAM)
    hyb, _ = mtrap.all_position_power_spectra(NUMERICAL_H)
    np.testing.assert_allclose(np.asarray(hyb), np.asarray(lam), rtol=1e-9)

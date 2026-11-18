"""
The analytic closed form with direct mechanical coupling between the dimensions of a bead

Off the node, levisim couples the dimensions of one bead directly (gxy, gxz, gyz from the
coherent-scattering potential; zero between different beads). The closed form of PRR 8 has only
the rank-2 cavity coupling, so it left these out and disagreed with LAM by factors of 7-75 on
the homodyne spectrum of the OFF_NODE toy systems. With the per-bead response matrix
W = (diag(1/μ) - i·g_intra)⁻¹ the cavity coupling stays rank 2, so the closed form keeps its
2×2 capacitance (M_norm) and only needs one (axes × axes) inverse per bead
(see `MultiTrap.analytic_kernel` and `MultiTrap.analytic_response_matrix`).

Tolerances: the measured worst case against LAM over all cases below is 3.3e-13 (2×2-4×2 with
gxy = -30 kHz, in the far wings of the spectrum where LAM itself is off a 40-digit mpmath
reference by up to 3.3e-13); rtol = 1e-12 leaves 3x headroom. The closed-form hybridisation
matrix and determinant agree with np.linalg.inv / det of the derangement matrix to <= 6.7e-13
relative to max|D⁻¹|, so those use 5e-12.
"""
import io
import contextlib
from copy import copy

import numpy as np
import pytest

from levisim.core.units import kHz
from levisim.multiphysics.enums.quantum_linear_theory_method import QuantumLinearTheoryMethod
from levisim.multiphysics.utils.shelf import (
    OFF_NODE_toy_params_equal_spacing_simple,
    OFF_NODE_toy_params_pairs,
    PHASE_DEPENDENT_toy_params,
)

LAM = QuantumLinearTheoryMethod.LINEAR_AMPLIFIER_MODEL
ANALYTIC_INSIDE = QuantumLinearTheoryMethod.ANALYTIC_INSIDE_CAVITY
ANALYTIC_OUTPUT = QuantumLinearTheoryMethod.ANALYTIC_CAVITY_OUTPUT

OMEGA_RANGE = np.arange(250, 320, 0.5) * kHz
RTOL = 1e-12


def with_gxy(gxy):
    """PHASE_DEPENDENT toy parameters (unequal phases, x and y only) plus a direct x-y coupling"""
    params = copy(PHASE_DEPENDENT_toy_params)
    params['toy_gxy'] = gxy
    return params


CASES = {
    'off node 1×2': ('1×2', OFF_NODE_toy_params_pairs),
    'off node 2×2': ('2×2', OFF_NODE_toy_params_pairs),
    'off node 1×3': ('1×3', OFF_NODE_toy_params_pairs),
    'off node 3×2': ('3×2', OFF_NODE_toy_params_pairs),
    'off node 2×3': ('2×3', OFF_NODE_toy_params_pairs),
    'off node equal spacing 1×2': ('1×2', OFF_NODE_toy_params_equal_spacing_simple),
    'off node equal spacing 2×2': ('2×2', OFF_NODE_toy_params_equal_spacing_simple),
    'unequal phases 2×2, gxy = -12 kHz': ('2×2', with_gxy(-12 * kHz)),
    'unequal phases 3×2, gxy = -12 kHz': ('3×2', with_gxy(-12 * kHz)),
    'unequal phases 3×2, gxy = -30 kHz': ('3×2', with_gxy(-30 * kHz)),
    'unequal phases 4×2, gxy = -30 kHz': ('4×2', with_gxy(-30 * kHz)),
    # a vanishingly weak coupling keeps the on-resonance blocks of W nearly singular
    'unequal phases 3×2, gxy = -1 mHz': ('3×2', with_gxy(-1e-6 * kHz)),
}


def quietly(function, *args, **kwargs):
    """The spectrum functions print a progress line per call"""
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return function(*args, **kwargs)


def spectra(mtrap, method):
    """(homodyne, heterodyne, zero-point-scaled position PSDs) for one method"""
    return (
        np.asarray(quietly(mtrap.homodyne_spectrum, method)),
        np.asarray(quietly(mtrap.heterodyne_spectrum, method)),
        np.asarray(quietly(mtrap.all_position_power_spectra, method, zpf_scaled=True)[0]),
    )


@pytest.fixture
def build(make_mtrap):
    def _build(case):
        system, toy_params = CASES[case]
        return quietly(
            make_mtrap, system, ω_range=OMEGA_RANGE, toy_params=toy_params, load_in_power_series=False,
        )
    return _build


@pytest.mark.parametrize('method', [ANALYTIC_INSIDE, ANALYTIC_OUTPUT])
@pytest.mark.parametrize('case', list(CASES))
def test_analytic_spectra_match_LAM(build, case, method):
    """Homodyne, heterodyne and every position PSD agree with LAM to rtol = 1e-12"""
    mtrap = build(case)
    assert np.any(mtrap.intra_bead_coupling_matrix), 'case has no intra-bead coupling to test'
    for name, analytic, reference in zip(
        ['homodyne', 'heterodyne', 'position PSDs'], spectra(mtrap, method), spectra(mtrap, LAM),
    ):
        np.testing.assert_allclose(analytic, reference, rtol=RTOL, atol=0, err_msg=f'{case}: {name}')


@pytest.mark.parametrize('case', [
    'off node 3×2', 'off node 2×3', 'unequal phases 3×2, gxy = -30 kHz', 'unequal phases 4×2, gxy = -30 kHz',
])
def test_closed_form_hybridisation_matrix_and_determinant(build, case):
    """
    D⁻¹ from the block/Woodbury closed form equals np.linalg.inv(derangement matrix), and
    det D = M_norm · Π_n det(W_n⁻¹) · Π_jn μjn/Mjn, on and off resonance
    """
    mtrap = build(case)
    for ω in list(mtrap.ω_range[::10]) + [dim.Ω for dim in mtrap.dimensions.values()]:
        D = mtrap.derangement_matrix(ω)
        numerical = np.linalg.inv(D)
        np.testing.assert_allclose(
            mtrap.analytic_hybridisation_matrix(ω), numerical,
            rtol=0, atol=5e-12 * np.max(np.abs(numerical)), err_msg=f'{ω / kHz} kHz',
        )
        kernel = mtrap.analytic_kernel(ω)
        M_per_dimension = np.array([dim.M(ω) for dim in mtrap.dimensions.values()])
        determinant = (
            kernel['M_norm']
            * np.prod([np.linalg.det(block) for block in kernel['W_inv_blocks']])
            * np.prod(kernel['μ'] / M_per_dimension)
        )
        np.testing.assert_allclose(determinant, np.linalg.det(D), rtol=5e-12, err_msg=f'{ω / kHz} kHz')


def test_response_matrix_equals_dense_inverse(build):
    """
    H from Woodbury (off-bead blocks) and Schur complements (on-bead blocks) equals the dense
    inverse of W⁻¹ - iC, with C = iχC·G*Gᵀ - iχC*·GG† (not symmetric: the cavity-mediated coupling
    is non-reciprocal for unequal phases)
    """
    mtrap = build('off node 2×3')
    G = mtrap.G_array
    for ω in (mtrap.dimensions['x1'].Ω, 283.25 * kHz):
        χC, χC_star = mtrap.cavity.χC(ω), np.conjugate(mtrap.cavity.χC(-ω))
        W_inv = np.zeros((mtrap.num_dimensions, mtrap.num_dimensions), dtype=np.complex128)
        for indices, block in zip(mtrap.trap_dimension_indices, mtrap.analytic_kernel(ω)['W_inv_blocks']):
            W_inv[np.ix_(indices, indices)] = block
        C = 1j * χC * np.outer(np.conjugate(G), G) - 1j * χC_star * np.outer(G, np.conjugate(G))
        dense = np.linalg.inv(W_inv - 1j * C)
        np.testing.assert_allclose(
            mtrap.analytic_response_matrix(ω), dense, rtol=0, atol=1e-12 * np.max(np.abs(dense)),
        )


def test_reduces_exactly_without_intra_bead_coupling(make_mtrap):
    """
    With gxy = gxz = gyz = 0, W is exactly diag(μ) and the thermal brackets are exactly those of
    PRR 8 (no B⁻¹ factor): the generalised code path reproduces the uncoupled one bit for bit
    """
    mtrap = quietly(
        make_mtrap, '3×2', ω_range=OMEGA_RANGE, toy_params=with_gxy(0 * kHz), load_in_power_series=False,
    )
    assert not np.any(mtrap.intra_bead_coupling_matrix)
    χC = mtrap.cavity.χC
    for ω in (mtrap.dimensions['y2'].Ω, 271.5 * kHz):
        kernel = mtrap.analytic_kernel(ω)
        np.testing.assert_array_equal(kernel['W'], np.diag(kernel['μ']))
        I, G = mtrap.coupling_quadrature_overlap_matrix, mtrap.G_array
        np.testing.assert_array_equal(
            kernel['thermal_a'], G - 2j * np.conjugate(χC(-ω)) * (I.T @ kernel['h']),
        )
        np.testing.assert_array_equal(
            kernel['thermal_a_dag'], np.conjugate(G) - 2j * χC(ω) * (I.T @ kernel['h_tilde']),
        )


def test_intra_bead_coupling_matrix_matches_drift_matrix(build):
    """The couplings used by the closed form are the mech-mech entries of the LAM drift matrix"""
    mtrap = build('off node 2×3')
    g_intra = mtrap.intra_bead_coupling_matrix
    np.testing.assert_array_equal(g_intra, g_intra.T)
    for jj, label_j in enumerate(mtrap.labels):
        for kk, label_k in enumerate(mtrap.labels):
            if jj == kk:
                continue
            # bjn row, bkm column of the drift matrix is i·gjk (see setup_linear_amplifier_model_matrices)
            entry = mtrap.drift_matrix[mtrap.label_to_noise_index[label_j], mtrap.label_to_noise_index[label_k]]
            if label_j[1:] == label_k[1:]:
                assert entry == 1j * g_intra[jj, kk], (label_j, label_k)
            else:
                assert g_intra[jj, kk] == 0

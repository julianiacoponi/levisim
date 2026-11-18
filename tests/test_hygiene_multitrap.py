"""
Bookkeeping in MultiTrap, the shelf, the comparator and Cavity: ΩModel, stitched ω ranges,
toy info strings for custom couplings, the per-frequency hybridisation-function table, the
array of all position cross spectra, the per-resonance failed-point allowance, and Cavity's
derived properties following their inputs.
"""
import numpy as np
import pytest

import matplotlib.pyplot as plt

from levisim.core.constants import π, c, ħ, ε0
from levisim.core.units import kHz
from levisim.physics.cavity import Cavity
from levisim.multiphysics.multitrap import ΩModel
from levisim.multiphysics.enums.quantum_linear_theory_method import QuantumLinearTheoryMethod
from levisim.multiphysics.utils import shelf
from levisim.multiphysics.utils.comparison import (
    RelativeDifferenceTooHigh,
    assert_arrays_agree_within_threshold,
)

LAM = QuantumLinearTheoryMethod.LINEAR_AMPLIFIER_MODEL
OMEGA_RANGE = np.arange(280, 320, 1.0) * kHz


# --------------------------------------------------------------------------------------------------
# ΩModel
# --------------------------------------------------------------------------------------------------
def test_shelf_Ω_model_dicts_keep_their_values():
    """The shelf dicts are now built from ΩModel but must equal the literals they replaced"""
    assert shelf.FORTRAN_Ω_model_pairs == dict(Ω_model='pairs', Ω_first=312*kHz, Ω_intrapair_gap=1.5*kHz, Ω_interpair_gap=15*kHz)
    assert shelf.FORTRAN_Ω_model_equal_spacing_simple == dict(Ω_model='equal spacing: simple', Ω_start=312*kHz, Ω_gap=7*kHz)
    assert shelf.DEFAULT_Ω_model_pairs == dict(Ω_model='pairs', Ω_first=305*kHz, Ω_intrapair_gap=2.5*kHz, Ω_interpair_gap=7*kHz)
    assert shelf.DEFAULT_Ω_model_equal_spacing == dict(Ω_model='equal spacing', Ω_mid=275*kHz, Ω_span=50*kHz)
    assert shelf.DEFAULT_Ω_model_equal_spacing_simple == dict(Ω_model='equal spacing: simple', Ω_start=305*kHz, Ω_gap=7*kHz)


def test_Ω_model_frequencies():
    labels = ['x1', 'y1', 'x2', 'y2']
    pairs = ΩModel('pairs', Ω_first=305*kHz, Ω_intrapair_gap=2.5*kHz, Ω_interpair_gap=7*kHz).frequencies(labels)
    assert np.allclose([pairs[_] / kHz for _ in labels], [305, 307.5, 298, 300.5], rtol=0, atol=1e-9)
    simple = ΩModel('equal spacing: simple', Ω_start=305*kHz, Ω_gap=7*kHz).frequencies(labels)
    assert np.allclose([simple[_] / kHz for _ in labels], [305, 298, 291, 284], rtol=0, atol=1e-9)
    spaced = ΩModel('equal spacing', Ω_mid=300*kHz, Ω_span=60*kHz).frequencies(labels)
    assert np.allclose([spaced[_] / kHz for _ in labels], [270, 290, 310, 330], rtol=0, atol=1e-9)
    assert set(ΩModel('degenerate', Ω0=250*kHz).frequencies(labels).values()) == {250*kHz}
    # z labels are left to MultiTrap
    assert 'z1' not in ΩModel('degenerate').frequencies(['x1', 'y1', 'z1'])


def test_multiband_model_shares_labels_over_bands():
    """5 labels over 2 bands: 3 in the first (evenly filling 200 ± 10), 2 in the second"""
    model = ΩModel('multiband', Ω_centers=[200*kHz, 300*kHz], Ω_spans=[20*kHz, 4*kHz])
    labels = ['x1', 'y1', 'x2', 'y2', 'x3']
    Ω = model.frequencies(labels)
    assert np.allclose([Ω[_] / kHz for _ in labels], [190, 200, 210, 298, 302], rtol=0, atol=1e-9)
    single = ΩModel('multiband', Ω_centers=[200*kHz, 300*kHz], Ω_spans=[20*kHz, 4*kHz]).frequencies(['x1', 'y1'])
    assert np.allclose([single['x1'] / kHz, single['y1'] / kHz], [200, 300], rtol=0, atol=1e-9)
    assert 'Ω_centers = [200., 300.] kHz' in model.description


def test_Ω_model_validation():
    with pytest.raises(ValueError):
        ΩModel('pears')
    with pytest.raises(ValueError):
        ΩModel('pairs', Ω_gap=1*kHz)
    with pytest.raises(ValueError):
        ΩModel('multiband', Ω_centers=[300*kHz])
    with pytest.raises(ValueError):
        ΩModel('custom')
    assert ΩModel.from_toy_params({'Ω_model': 'degenerate', 'Ω0': 1.0, 'toy_gx': 2.0}).params == {'Ω0': 1.0}


def test_multitrap_accepts_Ω_model_instance(make_mtrap):
    """Passing an ΩModel gives the same frequencies and attributes as the legacy dict form"""
    legacy = make_mtrap('2×2', ω_range=OMEGA_RANGE, toy_params=shelf.OFF_NODE_toy_params_pairs)
    params = {k: v for k, v in shelf.OFF_NODE_toy_params_pairs.items() if k not in ΩModel.KINDS['pairs']}
    params['Ω_model'] = ΩModel('pairs', **{k: shelf.OFF_NODE_toy_params_pairs[k] for k in ΩModel.KINDS['pairs']})
    modern = make_mtrap('2×2', ω_range=OMEGA_RANGE, toy_params=params)
    assert modern.Ωjn_dict == legacy.Ωjn_dict
    assert modern.Ω_model == legacy.Ω_model == 'pairs'
    assert modern.Ω_first == legacy.Ω_first == 305*kHz
    assert np.array_equal(modern.homodyne_spectrum(LAM), legacy.homodyne_spectrum(LAM))


def test_z_multiplier_described_once(make_mtrap):
    mtrap = make_mtrap('3×3', ω_range=OMEGA_RANGE, toy_params=shelf.OFF_NODE_toy_params_pairs)
    assert mtrap.Ω_model_description.count('Ωz_to_Ωx_multiplier') == 1
    assert mtrap.Ωjn_dict['z2'] == mtrap.Ωjn_dict['x2'] / 10


# --------------------------------------------------------------------------------------------------
# info strings, ω ranges, spectra bookkeeping
# --------------------------------------------------------------------------------------------------
def test_toy_info_str_shows_custom_couplings(make_mtrap):
    custom = make_mtrap('2×2', ω_range=OMEGA_RANGE, toy_params=shelf.PHASE_EQUAL_toy_params)
    assert 'N/A: custom gjn_dict' in custom.toy_params_info_str
    assert 'gy2 = -37.00000 kHz' in custom.toy_params_info_str
    assert 'gx = 0.00000 kHz' not in custom.toy_params_info_str
    default = make_mtrap('2×2', ω_range=OMEGA_RANGE, toy_params=shelf.OFF_NODE_toy_params_pairs)
    assert 'gx = -40.00000 kHz' in default.toy_params_info_str
    assert 'custom gjn_dict' not in default.toy_params_info_str


def test_stitched_ω_ranges(make_mtrap):
    """A list of ranges is stitched into one sorted, duplicate-free grid; spectra are pointwise"""
    coarse = np.arange(250, 320, 5.0) * kHz
    fine = np.arange(295, 305, 0.5) * kHz
    stitched = make_mtrap('1×2', ω_range=[coarse, fine], toy_params=shelf.OFF_NODE_toy_params_pairs)
    expected = np.unique(np.concatenate([coarse, fine]))
    assert np.array_equal(stitched.ω_range, expected)
    assert len(stitched.ω_ranges) == 2
    assert stitched.ω_step == coarse[1] - coarse[0]
    single = make_mtrap('1×2', ω_range=expected, toy_params=shelf.OFF_NODE_toy_params_pairs)
    assert np.array_equal(stitched.homodyne_spectrum(LAM), single.homodyne_spectrum(LAM))
    # a single array is used as given
    assert single.ω_range is expected


def test_hybridisation_function_table_matches_direct_evaluation(make_mtrap):
    """The per-ω table gives exactly the values of evaluating each term directly"""
    mtrap = make_mtrap('2×2', ω_range=OMEGA_RANGE, toy_params=shelf.OFF_NODE_toy_params_pairs, load_in_power_series=True)
    ω = 301.3 * kHz
    R_values = mtrap.hybridisation_function_values(ω)
    assert len(R_values) == 4**2 - 4
    for term in mtrap.deranged_determinant_by_order[mtrap.evaluation_order]:
        assert mtrap.hybridisation_term_value(term, ω, R_values=R_values) == mtrap.hybridisation_term_value(term, ω)
    D = mtrap.derangement_matrix(ω)
    for (jn, km), R in R_values.items():
        assert D[mtrap.label_to_index[jn], mtrap.label_to_index[km]] == -R


def test_position_cross_spectra_array(make_mtrap):
    mtrap = make_mtrap('2×2', ω_range=OMEGA_RANGE, toy_params=shelf.PHASE_DEPENDENT_toy_params)
    Sqq = mtrap.position_cross_spectra_array
    n = mtrap.num_dimensions
    assert Sqq.shape == (len(OMEGA_RANGE), n, n)
    diagonal = np.real(np.einsum('wjj->wj', Sqq))
    assert np.allclose(diagonal, mtrap.position_power_spectra_array, rtol=1e-12, atol=0)
    assert np.allclose(Sqq, np.conjugate(np.swapaxes(Sqq, 1, 2)), rtol=1e-12, atol=0)
    csd = mtrap.position_cross_spectrum('x1', 'y2', LAM)
    jj, kk = mtrap.label_to_index['x1'], mtrap.label_to_index['y2']
    assert np.allclose(Sqq[:, jj, kk], csd, rtol=1e-9, atol=0)


def test_failed_points_per_resonance_allowance(make_mtrap):
    """Off by default; with k given, k·num_dimensions failed points per spectrum are allowed"""
    mtrap = make_mtrap('1×2', ω_range=OMEGA_RANGE, toy_params=shelf.OFF_NODE_toy_params_pairs)
    rng = np.random.default_rng(7)
    A = 1 + rng.random((len(OMEGA_RANGE), 2))
    B = A.copy()
    # 3 failed points per PSD, 6 in total, i.e. 7.5% of the 80 points
    B[[3, 10, 20], 0] *= 2
    B[[4, 11, 21], 1] *= 2
    arrays = {'A': A, 'B': B}
    with pytest.raises(RelativeDifferenceTooHigh):
        assert_arrays_agree_within_threshold(mtrap, arrays, difficulty_level='very hard')
    with pytest.raises(RelativeDifferenceTooHigh):
        assert_arrays_agree_within_threshold(mtrap, arrays, difficulty_level='very hard', allowed_failed_points_per_resonance=1)
    # 2 per resonance × 2 dimensions × 2 PSDs = 8 >= 6
    assert_arrays_agree_within_threshold(mtrap, arrays, difficulty_level='very hard', allowed_failed_points_per_resonance=2)
    plt.close('all')


# --------------------------------------------------------------------------------------------------
# Cavity
# --------------------------------------------------------------------------------------------------
def test_cavity_derived_properties_follow_their_inputs():
    """Changing an input after a derived property was read must not leave it stale"""
    cavity = Cavity(Δ=-450*kHz, L=0.96e-2, Finesse=40_000)
    _ = cavity.info_str
    cavity.Finesse = 20_000
    assert cavity.κ == π * c / (20_000 * cavity.L)
    cavity.Δ = -300*kHz
    assert cavity.ω == cavity.ωtw + 300*kHz
    assert cavity.k == cavity.ω / c
    assert cavity.E0 == np.sqrt(ħ * cavity.ω / (2 * ε0 * cavity.V))

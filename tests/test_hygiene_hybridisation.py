"""
The symbolic power-series ('derangement') machinery in levisim.multiphysics.hybridisation

Pins three things: the generated terms are the exact power series of det(D) and of
det(D)·D⁻¹ for the derangement matrix D = I - R; the single-pass generator returns exactly what
the per-entry generator does; and the power-series pickles shipped in levisim/multiphysics/power_series_data/ (written
when HybridisationTerm was a plain namedtuple) still load and equal freshly generated terms.
"""
import pickle
from pathlib import Path

import numpy as np
import pytest

from levisim.core.utils import load_variable_from_pickle
from levisim.multiphysics.hybridisation import (
    HybridisationTerm,
    term_string,
    term_latex,
    generate_subscripts_and_labels,
    generate_deranged_terms,
    generate_determinant_and_adjugate_terms,
    generate_determinant_and_adjugate_by_order,
    truncate_power_series,
)

from levisim.multiphysics.multitrap import POWER_SERIES_DATA_DIR as PICKLE_DIR  # noqa: E402


def evaluate(terms, R, label_to_index):
    """Numerical value of a power series for a numerical hybridisation-function matrix R"""
    total = 0
    for term in terms:
        value = term.sign
        for origin, destination in term.pairs:
            value = value * R[label_to_index[origin], label_to_index[destination]]
        total += value
    return total


@pytest.mark.parametrize('n', [1, 2, 3, 4, 5])
def test_power_series_equals_determinant_and_scaled_inverse(n):
    """Σ terms = det(I - R) and adjugate entry ij = (det(D)·D⁻¹)ij for a random complex R"""
    rng = np.random.default_rng(20261118 + n)
    labels = [f'x{ii}' for ii in range(1, n + 1)]
    label_to_index = {label: ii for ii, label in enumerate(labels)}
    R = 0.4 * (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))
    np.fill_diagonal(R, 0)
    D = np.eye(n) - R

    determinant, adjugate = generate_determinant_and_adjugate_terms(n, labels)
    assert abs(evaluate(determinant, R, label_to_index) - np.linalg.det(D)) < 1e-12
    expected = np.linalg.det(D) * np.linalg.inv(D)
    for ii, a in enumerate(labels):
        for jj, b in enumerate(labels):
            assert abs(evaluate(adjugate[a + b], R, label_to_index) - expected[ii, jj]) < 1e-12


@pytest.mark.parametrize('system', [(1, 2), (2, 2), (1, 3), (3, 2)])
@pytest.mark.parametrize('max_order', [1, 2, None])
def test_single_pass_matches_per_entry_generation(system, max_order):
    """generate_determinant_and_adjugate_terms gives the per-entry lists, in the same order"""
    n = system[0] * system[1]
    _, labels = generate_subscripts_and_labels(*system)
    determinant, adjugate = generate_determinant_and_adjugate_terms(n, labels, max_order=max_order)
    assert determinant == generate_deranged_terms(n, dimension_labels=labels, max_order=max_order)
    for ii, a in enumerate(labels):
        for jj, b in enumerate(labels):
            assert adjugate[a + b] == generate_deranged_terms(n, ii, jj, dimension_labels=labels, max_order=max_order)


def test_term_counts_match_partial_derangement_numbers():
    """det of order n has n! terms; truncated to order 2 it is 1 + C(n,2) (identity + transpositions)"""
    _, labels = generate_subscripts_and_labels(3, 2)
    determinant, adjugate = generate_determinant_and_adjugate_by_order(6, labels, max_order=6)
    assert len(determinant[6]) == 720
    assert len(determinant[2]) == 1 + 15
    assert len(adjugate[6]['x1x1']) == 120
    assert len(adjugate[6]['x1y2']) == 120
    assert truncate_power_series(determinant[6], 2) == determinant[2]


def test_hybridisation_term_class():
    """Hashable, immutable, picklable, with pairs normalised to tuples"""
    term = HybridisationTerm(sign=-1, pairs=[['x1', 'y1'], ('y1', 'x1')])
    assert term.pairs == (('x1', 'y1'), ('y1', 'x1'))
    assert term.order == 2
    assert term.leading_dimensions == ['x1', 'y1']
    assert str(term) == term_string(term) == '-Rx1y1·Ry1x1'
    assert term.latex(with_ω=False) == term_latex(term, with_ω=False)
    assert len({term, HybridisationTerm(-1, [('x1', 'y1'), ('y1', 'x1')])}) == 1
    assert pickle.loads(pickle.dumps(term)) == term
    assert type(pickle.loads(pickle.dumps(term))) is HybridisationTerm
    with pytest.raises(AttributeError):
        term.sign = 1
    with pytest.raises(ValueError):
        HybridisationTerm(sign=2, pairs=[])


@pytest.mark.parametrize('N', [1, 2, 3])
def test_committed_pickles_load_and_equal_generated_terms(N):
    """The shipped power-series pickles still load, as HybridisationTerm, and are unchanged"""
    n = 2 * N
    path = PICKLE_DIR / f'{n}D_{N}x2_up_to_{n}.pickle'
    determinant, adjugate = load_variable_from_pickle(path)
    term = determinant[n][-1]
    assert type(term) is HybridisationTerm
    assert isinstance(term.pairs, tuple)
    hash(term)

    _, labels = generate_subscripts_and_labels(N, 2)
    generated_determinant, generated_adjugate = generate_determinant_and_adjugate_by_order(n, labels, max_order=n)
    assert determinant == generated_determinant
    assert adjugate == generated_adjugate

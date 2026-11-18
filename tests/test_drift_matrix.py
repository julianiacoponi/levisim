"""
Drift-matrix assembly for the Linear Amplifier Model

Converted from notebooks/multiphysics/tests/test_drift_matrix_generation.ipynb, which checked
the 1×3, 2×2 and 2×3 systems by writing out every expected row by hand (about 400 lines of
literal row definitions). Here the expected matrix is rebuilt from the documented block
structure, which covers the same three systems plus any others added to the parametrisation.

Block structure, for N traps with d Cartesian axes each (so n = N*d mechanical dimensions and
2n + 2 quantum operators):

  row 0        cavity:            [cav,  0  ] then [iG_k,  iG_k ] for every dimension k
  row 1        cavity conjugate:  [0,    cav*] then [-iG_k*, -iG_k*] for every dimension k
  row 2j + 2   dimension j:       [iG_j*, iG_j] then, per dimension k: (m_j, 0) if k == j,
                                  (ig_jk, ig_jk) if k shares a trap with j, else (0, 0)
  row 2j + 3   dimension j conj:  [-iG_j*, -iG_j] then, per k: (0, m_j*) if k == j,
                                  (-ig_jk, -ig_jk) if k shares a trap with j, else (0, 0)

where cav = iΔ - κ/2, m_j = -iΩ_j - Γ_j/2, and a starred quantity is its complex conjugate.
Dimensions in different traps do not couple mechanically, which is the zero block above.
"""
import numpy as np
import pytest


def intra_trap_coupling(mtrap, label_j, label_k):
    """
    Mechanical coupling g between two dimensions of the same trap

    The Dimension objects expose gxy, gxz and gyz; which one applies depends on the pair of
    Cartesian axes, and the notebook asserted these are symmetric (gxy == gyx).
    """
    axes = {label_j[0], label_k[0]}
    dim_j = mtrap.dimensions[label_j]
    if axes == {'x', 'y'}:
        return dim_j.gxy
    if axes == {'x', 'z'}:
        return dim_j.gxz
    if axes == {'y', 'z'}:
        return dim_j.gyz
    raise AssertionError(f'unexpected axis pair {axes} for ({label_j}, {label_k})')


def expected_drift_matrix(mtrap):
    """Rebuilds the drift matrix from the block structure documented above"""
    labels = list(mtrap.labels)
    trap_of = {label: label[1:] for label in labels}

    cav = 1j * mtrap.cavity.Δ - mtrap.cavity.κ / 2
    rows = []

    cavity_row = [cav, 0]
    cavity_conj_row = [0, np.conjugate(cav)]
    for label in labels:
        G = mtrap.dimensions[label].G
        cavity_row += [1j * G, 1j * G]
        cavity_conj_row += [-1j * np.conjugate(G), -1j * np.conjugate(G)]
    rows.append(cavity_row)
    rows.append(cavity_conj_row)

    for label_j in labels:
        dim_j = mtrap.dimensions[label_j]
        G_j = dim_j.G
        m_j = -1j * dim_j.Ω - dim_j.Γ / 2

        row = [1j * np.conjugate(G_j), 1j * G_j]
        conj_row = [-1j * np.conjugate(G_j), -1j * G_j]
        for label_k in labels:
            if label_k == label_j:
                row += [m_j, 0]
                conj_row += [0, np.conjugate(m_j)]
            elif trap_of[label_k] == trap_of[label_j]:
                ig = 1j * intra_trap_coupling(mtrap, label_j, label_k)
                row += [ig, ig]
                conj_row += [-ig, -ig]
            else:
                row += [0, 0]
                conj_row += [0, 0]
        rows.append(row)
        rows.append(conj_row)

    return np.array(rows, dtype=np.complex128)


SYSTEMS = ['1×3', '2×2', '2×3']


@pytest.mark.parametrize('system', SYSTEMS)
def test_drift_matrix_matches_expected_block_structure(make_mtrap, system):
    """Every entry of the assembled drift matrix must equal the documented block structure"""
    mtrap = make_mtrap(system)
    actual = np.asarray(mtrap.drift_matrix)
    expected = expected_drift_matrix(mtrap)

    assert actual.shape == expected.shape
    mismatches = np.argwhere(actual != expected)
    assert mismatches.size == 0, (
        f'{len(mismatches)} mismatched entries for {system}, first at '
        f'{tuple(mismatches[0])}: actual={actual[tuple(mismatches[0])]!r} '
        f'expected={expected[tuple(mismatches[0])]!r}'
    )


@pytest.mark.parametrize('system', SYSTEMS)
def test_drift_matrix_has_two_operators_per_degree_of_freedom(make_mtrap, system):
    """The matrix is square with 2n + 2 rows: two per mechanical dimension, two for the cavity"""
    mtrap = make_mtrap(system)
    expected_size = 2 * mtrap.num_dimensions + 2
    assert mtrap.num_operators == expected_size
    assert np.asarray(mtrap.drift_matrix).shape == (expected_size, expected_size)


@pytest.mark.parametrize('system', SYSTEMS)
def test_intra_trap_couplings_are_symmetric(make_mtrap, system):
    """
    g_jk == g_kj for every pair of axes within a trap

    The notebook asserted this inline for each named pair; here it is checked for all pairs.
    """
    mtrap = make_mtrap(system)
    labels = list(mtrap.labels)
    for ii, label_j in enumerate(labels):
        for label_k in labels[ii + 1:]:
            if label_j[1:] != label_k[1:]:
                continue
            forward = intra_trap_coupling(mtrap, label_j, label_k)
            backward = intra_trap_coupling(mtrap, label_k, label_j)
            assert forward == backward, f'({label_j}, {label_k}): {forward} != {backward}'


@pytest.mark.parametrize('system', ['2×2', '2×3'])
def test_dimensions_in_different_traps_do_not_couple(make_mtrap, system):
    """Mechanical cross-coupling exists only within a trap, so inter-trap blocks are zero"""
    mtrap = make_mtrap(system)
    actual = np.asarray(mtrap.drift_matrix)
    labels = list(mtrap.labels)
    for jj, label_j in enumerate(labels):
        for kk, label_k in enumerate(labels):
            if label_j[1:] == label_k[1:]:
                continue
            block = actual[2 + 2 * jj:4 + 2 * jj, 2 + 2 * kk:4 + 2 * kk]
            assert not block.any(), f'non-zero block between {label_j} and {label_k}:\n{block}'

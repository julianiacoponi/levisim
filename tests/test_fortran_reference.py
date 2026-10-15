"""
levisim checked against the Fortran reference implementation

Converted from notebooks/multiphysics/tests/test_fortran_vs_levisim.ipynb, which could not run
on a clean clone because the fortran/ directory was untracked. It now ships with the repo, so
this comparison can be a real test.

The reference data is an 8x2 = 16-dimensional system (8 nanoparticles, near-degenerate x and y
frequencies in pairs), produced by `make pairs` in fortran/ and matching the FORTRAN_8x2 fixture.

TOLERANCE, AND WHY
Measured agreement between levisim's NUMERICAL_HYBRIDISATION spectra and the Fortran output is
a maximum relative difference of 7.4e-7 with a median of ~9e-8, across the heterodyne spectrum,
the homodyne spectrum and all six position PSDs.

That floor is not a disagreement between the two implementations: NParticle.f90 writes with
`101 FORMAT(7E14.7)`, i.e. seven significant digits, so the file cannot record a value more
precisely than ~1e-7 relative. The two agree to the limit of what the data format can express.

The assertion tolerance is therefore set an order of magnitude above the observed maximum, and
a separate test pins the measured maximum so that real drift still fails even though the
tolerance is loose.
"""
from pathlib import Path

import numpy as np
import pytest

from levisim.core.units import kHz
from levisim.multiphysics.enums.quantum_linear_theory_method import QuantumLinearTheoryMethod

FORTRAN_DIR = Path(__file__).resolve().parent.parent / 'fortran'
PSD_FILE = FORTRAN_DIR / 'FTanXYZ.dat'
OPTICAL_FILE = FORTRAN_DIR / 'FTanOPT.dat'

# generous enough not to be brittle, tight enough to catch a real divergence
COMPARISON_RTOL = 1e-5
# the agreement actually measured, pinned separately - see the module docstring
MEASURED_MAX_RELATIVE_DIFFERENCE = 1e-6

pytestmark = pytest.mark.skipif(
    not (PSD_FILE.exists() and OPTICAL_FILE.exists()),
    reason=f'Fortran reference data not found under {FORTRAN_DIR}',
)


@pytest.fixture(scope='module')
def fortran_psd_data():
    """ω (kHz), then the position PSD for x1, y1, x2, y2, x3, y3"""
    return np.loadtxt(PSD_FILE)


@pytest.fixture(scope='module')
def fortran_optical_data():
    """ω (kHz), heterodyne, symmetrised homodyne"""
    return np.loadtxt(OPTICAL_FILE)


@pytest.fixture(scope='module')
def levisim_spectra():
    """The same spectra from levisim, via the method the notebook compared against"""
    from levisim.multiphysics.utils.shelf import FORTRAN_8x2
    method = QuantumLinearTheoryMethod.NUMERICAL_HYBRIDISATION
    position_psds, _ = FORTRAN_8x2.all_position_power_spectra(method)
    return {
        'ω_range': np.asarray(FORTRAN_8x2.ω_range),
        'position_psd': np.asarray(position_psds),
        'heterodyne': np.asarray(FORTRAN_8x2.heterodyne_spectrum(method)),
        'homodyne': np.asarray(FORTRAN_8x2.homodyne_spectrum(method)),
    }


def aligned(fortran_column, levisim_array):
    """
    Pairs a Fortran column with the matching levisim values

    The Fortran sweep starts at 0.5 kHz and levisim's at 0, so Fortran row ii corresponds to
    levisim index ii + 1. The Fortran files are also shorter than the levisim sweep: a complete
    run writes 800 rows but the committed files hold 799 (PSD) and 769 (optical), so the
    comparison runs over however many rows the reference actually provides.
    """
    count = len(fortran_column)
    return np.asarray(fortran_column), np.asarray(levisim_array[1:count + 1])


def relative_difference(fortran_column, levisim_array):
    reference, computed = aligned(fortran_column, levisim_array)
    return np.abs(reference - computed) / np.maximum(np.abs(reference), 1e-300)


def test_frequency_axes_line_up(fortran_psd_data, levisim_spectra):
    """The off-by-one alignment this comparison relies on must actually hold"""
    fortran_omega = fortran_psd_data[:, 0]
    levisim_omega = levisim_spectra['ω_range'] / kHz
    assert fortran_omega[0] == pytest.approx(levisim_omega[1])
    reference, computed = aligned(fortran_omega, levisim_omega)
    assert np.allclose(reference, computed, rtol=1e-9)


def test_heterodyne_spectrum_matches_fortran(fortran_optical_data, levisim_spectra):
    reference, computed = aligned(fortran_optical_data[:, 1], levisim_spectra['heterodyne'])
    assert np.allclose(reference, computed, rtol=COMPARISON_RTOL)


def test_homodyne_spectrum_matches_fortran(fortran_optical_data, levisim_spectra):
    reference, computed = aligned(fortran_optical_data[:, 2], levisim_spectra['homodyne'])
    assert np.allclose(reference, computed, rtol=COMPARISON_RTOL)


@pytest.mark.parametrize(
    'column, label',
    [(1, 'x1'), (2, 'y1'), (3, 'x2'), (4, 'y2'), (5, 'x3'), (6, 'y3')],
)
def test_position_psd_matches_fortran(fortran_psd_data, levisim_spectra, column, label):
    """
    Only the first six position PSDs are written by the Fortran

    NParticle.f90 writes seven columns via `101 FORMAT(7E14.7)` - ω plus six PSDs - so the
    remaining ten dimensions of the 16D system have no reference values.
    """
    reference, computed = aligned(
        fortran_psd_data[:, column], levisim_spectra['position_psd'][:, column - 1],
    )
    assert np.allclose(reference, computed, rtol=COMPARISON_RTOL), (
        f'{label} PSD: max relative difference '
        f'{relative_difference(fortran_psd_data[:, column], levisim_spectra["position_psd"][:, column - 1]).max():.3e}'
    )


def test_agreement_is_at_the_limit_of_the_file_format(
    fortran_psd_data, fortran_optical_data, levisim_spectra,
):
    """
    Pins the measured agreement, so drift fails even though COMPARISON_RTOL is loose

    Every spectrum should agree to within ~1e-6, which is the quantisation of the Fortran's
    seven-significant-digit output format rather than a limit of either implementation.
    """
    worst = {
        'heterodyne': relative_difference(fortran_optical_data[:, 1], levisim_spectra['heterodyne']).max(),
        'homodyne': relative_difference(fortran_optical_data[:, 2], levisim_spectra['homodyne']).max(),
    }
    for column in range(1, 7):
        worst[f'PSD column {column}'] = relative_difference(
            fortran_psd_data[:, column], levisim_spectra['position_psd'][:, column - 1],
        ).max()
    for name, value in worst.items():
        assert value < MEASURED_MAX_RELATIVE_DIFFERENCE, f'{name}: {value:.3e}'

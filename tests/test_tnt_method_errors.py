"""
The TNT LAM only gives position power spectra: every other use raises a clear ValueError

The spectrum methods check this up front (MultiTrap._log_spectrum_call); the per-frequency noise
vector functions used to return None from hybridised_positions, which then failed with an opaque
'matmul: Input operand 1 does not have enough dimensions' in homodyne_noise_vector.
"""
import numpy as np
import pytest

from levisim.core.units import degrees, kHz
from levisim.multiphysics.enums import QuantumLinearTheoryMethod
from levisim.multiphysics.multitrap import MultiTrap

TNT = QuantumLinearTheoryMethod.TNT_LINEAR_AMPLIFIER_MODEL
ω = 130*kHz


@pytest.fixture(scope='module')
def mtrap():
    return MultiTrap(num_traps=2, Cartesian_axes_per_trap=2, φn_dict={2: 30*degrees}, ω_range=np.arange(125, 135, 1.0) * kHz)


@pytest.mark.parametrize('call', [
    lambda m: m.homodyne_spectrum(TNT),
    lambda m: m.heterodyne_spectrum(TNT),
    lambda m: m.homodyne_and_heterodyne_spectra(TNT),
    lambda m: m.position_power_spectrum('x1', TNT),
    lambda m: m.position_cross_spectrum('x1', 'y1', TNT),
], ids=['homodyne', 'heterodyne', 'homodyne_and_heterodyne', 'position_power', 'position_cross'])
def test_spectrum_methods_raise_up_front(mtrap, call):
    with pytest.raises(ValueError, match='only gives position power spectra'):
        call(mtrap)


@pytest.mark.parametrize('call', [
    lambda m: m.hybridised_positions(TNT, ω),
    lambda m: m.hybridised_position_noise_vector('x1', TNT, ω),
    lambda m: m.light_noise_vectors(TNT, ω),
    lambda m: m.homodyne_noise_vector(TNT, ω),
    lambda m: m.heterodyne_noise_vector(TNT, ω),
], ids=['hybridised_positions', 'position_noise_vector', 'light', 'homodyne', 'heterodyne'])
def test_noise_vector_functions_raise_a_clear_error(mtrap, call):
    with pytest.raises(ValueError, match='has no per-frequency noise vectors'):
        call(mtrap)


def test_all_position_power_spectra_still_works_and_matches_the_LAM(mtrap):
    tnt = mtrap.all_position_power_spectra(TNT)[0]
    lam = mtrap.all_position_power_spectra(QuantumLinearTheoryMethod.LINEAR_AMPLIFIER_MODEL)[0]
    np.testing.assert_allclose(tnt, lam, rtol=1e-10)

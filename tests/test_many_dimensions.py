"""
MultiTrap systems with more than 26 dimensions

The dimension subscripts used to be single letters (alphabet[:n]), so a system with more than 26
dimensions failed with a KeyError in Dimension. They now continue 'aa', 'ab', ... after 'z'.
"""
import string

import numpy as np
import pytest

from levisim.core.units import degrees, kHz
from levisim.multiphysics.enums import QuantumLinearTheoryMethod
from levisim.multiphysics.hybridisation import alphabetic_subscript, generate_subscripts_and_labels
from levisim.multiphysics.multitrap import MultiTrap


def test_subscripts_start_with_the_single_letters_and_stay_unique():
    assert [alphabetic_subscript(index) for index in range(26)] == list(string.ascii_lowercase)
    assert [alphabetic_subscript(index) for index in (26, 27, 51, 52, 701, 702)] == ['aa', 'ab', 'az', 'ba', 'zz', 'aaa']
    subscripts, labels = generate_subscripts_and_labels(2, 2)
    assert subscripts == ['a', 'b', 'c', 'd'] and labels == ['x1', 'y1', 'x2', 'y2']
    subscripts, _ = generate_subscripts_and_labels(1000, 3)
    assert len(set(subscripts)) == 3000


@pytest.mark.parametrize('num_traps, axes', [(30, 1), (10, 3), (15, 2)])
def test_thirty_dimension_system_builds_and_methods_agree(num_traps, axes):
    labels = [f'{axis}{n}' for n in range(1, num_traps + 1) for axis in 'xyz'[:axes]]
    toy_params = dict(
        Ω_model='custom',
        toy_Ωjn_dict={label: (120 + 3*int(label[1:]) + 20*'xyz'.index(label[0])) * kHz for label in labels},
        gjn_dict={label: -5*kHz for label in labels},
        φn_dict={n: (n % 2) * 30*degrees for n in range(1, num_traps + 1)},
    )
    mtrap = MultiTrap(
        num_traps=num_traps, Cartesian_axes_per_trap=axes, toy_params=toy_params,
        cavity_kwargs=dict(Δ=-300*kHz, κ=200*kHz), ω_range=np.arange(100, 200, 0.5) * kHz,
    )
    assert mtrap.num_dimensions == 30
    assert mtrap.dimensions[labels[-1]].sub == 'ad'
    assert np.linalg.eigvals(mtrap.drift_matrix).real.max() < 0
    lam = mtrap.all_position_power_spectra(QuantumLinearTheoryMethod.LINEAR_AMPLIFIER_MODEL)[0]
    numerical = mtrap.all_position_power_spectra(QuantumLinearTheoryMethod.NUMERICAL_HYBRIDISATION)[0]
    np.testing.assert_allclose(numerical, lam, rtol=1e-11, atol=0)
    np.testing.assert_allclose(
        mtrap.heterodyne_spectrum(QuantumLinearTheoryMethod.NUMERICAL_HYBRIDISATION),
        mtrap.heterodyne_spectrum(QuantumLinearTheoryMethod.LINEAR_AMPLIFIER_MODEL),
        rtol=1e-12, atol=0,
    )

"""
MultiTrap's physical-mode overrides: tweezers=, beads=, trap_kwargs= and φn_dict=

Without them, a physical MultiTrap is built from the built-in tweezers and beads in
MultiTrap.setup_levitated_physics, and must be bit-identical to what it was before the overrides
were added; passing the same objects explicitly must give exactly the same arrays.
"""
import numpy as np
import pytest

from levisim.core.logger import logger
from levisim.core.units import degrees, kHz, nm
from levisim.multiphysics.multitrap import MultiTrap
from levisim.multiphysics.enums.quantum_linear_theory_method import QuantumLinearTheoryMethod
from levisim.physics.bead import Bead
from levisim.physics.tweezer import Tweezer


LAM = QuantumLinearTheoryMethod.LINEAR_AMPLIFIER_MODEL
ω_RANGE = np.arange(-300, 300, 2.5) * kHz


@pytest.fixture
def captured_warnings():
    captured = []
    handler_id = logger.add(lambda message: captured.append(message.record), level='WARNING', format='{message}')
    yield captured
    logger.remove(handler_id)


def spectra(mtrap):
    return mtrap.heterodyne_spectrum(LAM), mtrap.all_position_power_spectra(LAM)[0]


@pytest.mark.parametrize('num_traps, axes', [(2, 1), (2, 2), (1, 3)])
def test_explicit_builtin_objects_are_bit_identical(num_traps, axes):
    default = MultiTrap(num_traps=num_traps, Cartesian_axes_per_trap=axes, ω_range=ω_RANGE)
    explicit = MultiTrap(
        num_traps=num_traps, Cartesian_axes_per_trap=axes, ω_range=ω_RANGE,
        tweezers=default.tweezers, beads=default.beads,
    )
    assert np.array_equal(default.drift_matrix, explicit.drift_matrix)
    for a, b in zip(spectra(default), spectra(explicit)):
        assert np.array_equal(a, b)


def test_φn_dict_replaces_one_tweezer_phase_without_touching_the_input():
    default = MultiTrap(num_traps=2, Cartesian_axes_per_trap=2, ω_range=ω_RANGE)
    shifted = MultiTrap(num_traps=2, Cartesian_axes_per_trap=2, ω_range=ω_RANGE, φn_dict={2: 30*degrees})
    assert shifted.φn_dict == {1: 0.0, 2: 30*degrees}
    assert shifted.dimensions['x2'].φ == 30*degrees
    assert default.tweezers[1].φ == 90*degrees
    # same as building the second tweezer with that phase directly
    t1, t2 = default.tweezers
    t2_30 = Tweezer(wx0=t2.wx0, wy0=t2.wy0, θ=t2.θ, φ=30*degrees)
    direct = MultiTrap(num_traps=2, Cartesian_axes_per_trap=2, ω_range=ω_RANGE, tweezers=[t1, t2_30])
    assert np.array_equal(shifted.drift_matrix, direct.drift_matrix)
    # the Tweezer instances passed in are not modified
    given = [Tweezer(), Tweezer(φ=90*degrees)]
    MultiTrap(num_traps=2, Cartesian_axes_per_trap=1, tweezers=given, φn_dict={2: 10*degrees})
    assert given[1].φ == 90*degrees


def test_beads_and_trap_kwargs_are_used():
    beads = [Bead(r=75*nm), Bead(r=80*nm)]
    Ωs = [dict(x=150*kHz, y=170*kHz, z=40*kHz), dict(x=155*kHz, y=175*kHz, z=45*kHz)]
    mtrap = MultiTrap(Cartesian_axes_per_trap=3, beads=beads, trap_kwargs=[dict(override_Ω_dict=Ω) for Ω in Ωs])
    assert mtrap.num_traps == 2
    assert [trap.bead for trap in mtrap.traps] == beads
    for n, Ω in enumerate(Ωs, start=1):
        for axis in 'xyz':
            assert mtrap.dimensions[f'{axis}{n}'].Ω == Ω[axis]


def test_num_traps_follows_the_given_lists_and_can_exceed_the_builtin_nine():
    tweezers = [Tweezer(φ=0*degrees) for _ in range(10)]
    beads = [Bead() for _ in range(10)]
    mtrap = MultiTrap(Cartesian_axes_per_trap=1, tweezers=tweezers, beads=beads)
    assert mtrap.num_traps == 10
    assert mtrap.labels[-1] == 'x10'


@pytest.mark.parametrize('kwargs, match', [
    (dict(num_traps=3, tweezers=[Tweezer(), Tweezer()]), 'tweezers has 2 entries'),
    (dict(num_traps=2, beads=[Bead()]), 'beads has 1 entries'),
    (dict(num_traps=2, trap_kwargs=[{}]), 'trap_kwargs has 1 entries'),
    (dict(num_traps=10), 'built-in tweezers'),
    (dict(num_traps=10, tweezers=[Tweezer() for _ in range(10)]), 'built-in beads'),
    (dict(num_traps=2, φn_dict={3: 0.0}), 'outside 1, ..., num_traps'),
])
def test_bad_override_raises_value_error(kwargs, match):
    with pytest.raises(ValueError, match=match):
        MultiTrap(Cartesian_axes_per_trap=1, **kwargs)


def test_overrides_with_toy_params_log_a_warning(captured_warnings):
    toy_params = dict(Ω_model='custom', toy_Ωjn_dict={'x1': 100*kHz}, gjn_dict={'x1': -10*kHz})
    MultiTrap(num_traps=1, Cartesian_axes_per_trap=1, toy_params=toy_params, φn_dict={1: 0.0})
    assert any('ignored with toy_params' in record['message'] for record in captured_warnings)

"""
MultiTrap warns about toy_params keys that nothing reads

Several notebooks carried renamed keys ('φ2' for 'φ_even_n', 'gs' for 'gjn_dict', 'first_Ωx' and
'Ω_gap' with the 'pairs' model) that were silently ignored, so they ran a different system than
intended. The shelf fixtures must produce no such warning.
"""
import pytest
from loguru import logger

from levisim.core.units import kHz
from levisim.multiphysics.multitrap import MultiTrap
from levisim.multiphysics.utils import shelf


@pytest.fixture
def warnings_seen():
    records = []
    handler = logger.add(lambda message: records.append(message.record['message']), level='WARNING')
    yield records
    logger.remove(handler)


def build(toy_params):
    return MultiTrap(
        num_traps=2, Cartesian_axes_per_trap=1, toy_params=toy_params,
        gas_kwargs=shelf.DEFAULT_gas_kwargs, cavity_kwargs=shelf.DEFAULT_cavity_kwargs,
        load_in_power_series=False,
    )


@pytest.mark.parametrize('name', [
    'FORTRAN_toy_params', 'OFF_NODE_toy_params_pairs', 'OFF_NODE_toy_params_equal_spacing',
    'OFF_NODE_toy_params_equal_spacing_simple', 'PHASE_EQUAL_toy_params', 'PHASE_DEPENDENT_toy_params',
])
def test_shelf_toy_params_use_every_key(name, warnings_seen):
    mtrap = build(getattr(shelf, name))
    assert mtrap.warn_about_unknown_toy_parameters(getattr(shelf, name)) == []
    assert not [w for w in warnings_seen if 'toy_params keys' in w]


def test_renamed_and_misspelt_keys_are_reported(warnings_seen):
    toy_params = dict(shelf.PHASE_EQUAL_toy_params, φ2=0.5, gs=[1, 2], Ω_gap=3*kHz)
    build(toy_params)
    messages = [w for w in warnings_seen if 'toy_params keys' in w]
    assert len(messages) == 1
    for key in ('φ2', 'gs', 'Ω_gap'):
        assert repr(key) in messages[0]

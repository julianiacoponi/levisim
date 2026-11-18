"""
The logging convention (docs/contributing.md, 'Logging levels')

Each test adds a temporary loguru sink that records every message from TRACE up, so it sees the
records whatever sinks levisim.core.logger has configured, and removes it afterwards.
"""
import ast
import importlib
import pathlib
import sys
import warnings

import numpy as np
import pytest
from loguru import logger

from levisim.core.units import kHz
from levisim.core.wrappers import timer
from levisim.core import testing
from levisim.data.simulation import Simulation
from levisim.multiphysics.enums.quantum_linear_theory_method import QuantumLinearTheoryMethod
from levisim.multiphysics.hybridisation import generate_determinant_and_adjugate_by_order
from levisim.multiphysics.multitrap import MultiTrap
from levisim.multiphysics.utils.shelf import (
    DEFAULT_cavity_kwargs,
    DEFAULT_gas_kwargs,
    OFF_NODE_toy_params_pairs,
    PHASE_EQUAL_toy_params,
)

OMEGA_RANGE = np.arange(280, 320, 2.0) * kHz
OPTICAL_METHODS = [
    method for method in QuantumLinearTheoryMethod
    if method != QuantumLinearTheoryMethod.TNT_LINEAR_AMPLIFIER_MODEL
]
LEVISIM_ROOT = pathlib.Path(__file__).resolve().parent.parent / 'levisim'


@pytest.fixture
def records():
    """Every log record emitted while the test runs, as loguru record dicts"""
    captured = []
    handler_id = logger.add(lambda message: captured.append(message.record), level='TRACE', format='{message}')
    yield captured
    logger.remove(handler_id)


def at(records, level):
    """The messages of the records at exactly `level`"""
    return [record['message'] for record in records if record['level'].name == level]


def at_least(records, level):
    """The messages of the records at `level` or above"""
    threshold = logger.level(level).no
    return [record['message'] for record in records if record['level'].no >= threshold]


def stable_mtrap(system=(2, 2), **kwargs):
    """A MultiTrap whose drift matrix is stable (equal tweezer phases, red-detuned cavity)"""
    num_traps, axes = system
    return MultiTrap(
        num_traps=num_traps,
        Cartesian_axes_per_trap=axes,
        toy_params=PHASE_EQUAL_toy_params,
        gas_kwargs=DEFAULT_gas_kwargs,
        cavity_kwargs=DEFAULT_cavity_kwargs,
        ω_range=OMEGA_RANGE,
        **kwargs,
    )


def test_building_a_multitrap_logs_one_info_summary(records):
    stable_mtrap()
    infos = at(records, 'INFO')
    assert len(infos) == 1, infos
    assert infos[0].startswith('Built MultiTrap(2×2): 4 dimensions (x1, y1, x2, y2) from toy parameters')
    assert f'over {len(OMEGA_RANGE)} points' in infos[0]
    assert not at_least(records, 'WARNING')


def test_loading_a_power_series_adds_one_info_line(records):
    stable_mtrap(load_in_power_series=True)
    infos = at(records, 'INFO')
    assert len(infos) == 2, infos
    assert infos[0] == 'Loading the 4D power series up to order 4 from 4D_2x2_up_to_4.pickle'
    assert infos[1].endswith('power series loaded up to order 4')


def test_parameter_dumps_are_logged_at_debug(records):
    stable_mtrap()
    dumps = [message for message in at(records, 'DEBUG') if message.startswith('Parameters of MultiTrap(2×2)')]
    assert len(dumps) == 1 and 'Cavity(' in dumps[0] and 'Dimension(' in dumps[0]


def test_parameter_dumps_are_not_built_when_no_sink_takes_debug(monkeypatch):
    """The DEBUG dumps are lazy: with only a WARNING sink, building a MultiTrap never calls info_str"""
    calls = []
    original = MultiTrap.info_str
    monkeypatch.setattr(MultiTrap, 'info_str', property(lambda self: calls.append(1) or original.fget(self)))
    logger.remove()
    logger.add(lambda message: None, level='WARNING')
    try:
        stable_mtrap()
    finally:
        logger.remove()
        # restore the sinks levisim.core.logger configures, for the rest of the session (the module
        # is taken from sys.modules: the attribute levisim.core.logger is the loguru logger object,
        # rebound by `from levisim.core.logger import *` in levisim/core/__init__.py)
        importlib.reload(sys.modules['levisim.core.logger'])
    assert calls == []


@pytest.mark.needs_power_series
@pytest.mark.parametrize('system', [(1, 2), (2, 2)])
def test_spectra_of_a_well_posed_system_log_no_warning_and_print_nothing(records, capsys, system):
    mtrap = stable_mtrap(system, load_in_power_series=True)
    for method in OPTICAL_METHODS:
        mtrap.homodyne_spectrum(method)
        mtrap.heterodyne_spectrum(method)
        mtrap.homodyne_and_heterodyne_spectra(method)
        mtrap.position_power_spectrum(mtrap.labels[0], method)
        mtrap.position_cross_spectrum(mtrap.labels[0], mtrap.labels[-1], method)
    for method in QuantumLinearTheoryMethod:
        mtrap.all_position_power_spectra(method)
    assert not at_least(records, 'WARNING'), at_least(records, 'WARNING')
    assert capsys.readouterr().out == ''


def test_spectrum_call_logs_its_method_and_branch(records):
    mtrap = stable_mtrap()
    records.clear()
    mtrap.homodyne_spectrum(QuantumLinearTheoryMethod.NUMERICAL_HYBRIDISATION)
    assert at(records, 'INFO') == [f'🧮 Calculating homodyne spectrum via Numerical H over {len(OMEGA_RANGE)} ω points']
    debug = at(records, 'DEBUG')
    assert any('derangement matrix D = I - R' in message for message in debug), debug
    assert any(message.startswith('⏱️ "MultiTrap.homodyne_spectrum" executed in') for message in debug), debug


def test_unstable_drift_matrix_logs_one_warning(records):
    """Unequal tweezer phases (0°, 45°) make one collective mode grow here: Re λ ≈ +574 /s"""
    MultiTrap(
        num_traps=2, Cartesian_axes_per_trap=2, toy_params=OFF_NODE_toy_params_pairs,
        gas_kwargs=DEFAULT_gas_kwargs, cavity_kwargs=DEFAULT_cavity_kwargs, ω_range=OMEGA_RANGE,
    )
    warnings = at(records, 'WARNING')
    assert len(warnings) == 1, warnings
    assert 'eigenvalues have a positive real part' in warnings[0]
    assert 'largest Re λ = +574' in warnings[0]


def test_invalid_method_choices_log_an_error_before_raising(records):
    mtrap = stable_mtrap()
    with pytest.raises(ValueError, match='only gives position power spectra'):
        mtrap.homodyne_spectrum(QuantumLinearTheoryMethod.TNT_LINEAR_AMPLIFIER_MODEL)
    with pytest.raises(AttributeError, match='needs a power series'):
        mtrap.heterodyne_spectrum(QuantumLinearTheoryMethod.POWER_SERIES_HYBRIDISATION)
    assert len(at(records, 'ERROR')) == 2


def test_timer_logs_at_debug_and_prints_nothing(records, capsys):
    @timer(True, debug=True)
    def add(a, b=1):
        return a + b

    assert add(2, b=3) == 5
    debug = at(records, 'DEBUG')
    assert len(debug) == 2, debug
    assert 'called with args=(2,), kwargs={\'b\': 3}' in debug[0]
    assert debug[1].startswith('⏱️ "test_timer_logs_at_debug_and_prints_nothing.<locals>.add" executed in')
    assert not at(records, 'INFO')
    assert capsys.readouterr().out == ''


def test_power_series_generation_logs_start_and_finish_at_info(records):
    generate_determinant_and_adjugate_by_order(3, ['x1', 'y1', 'x2'], max_order=3)
    infos = at(records, 'INFO')
    assert infos[0] == 'Generating the 3D power series of det(D) and adj(D) up to order 3'
    assert infos[1].startswith('Generated the 3D power series up to order 3 in ')
    assert infos[1].endswith('6 determinant terms at order 3')


def test_simulation_logs_start_and_finish_with_sample_counts(records, capsys):
    Simulation(num_samples=2_000, seed=1)
    infos = at(records, 'INFO')
    assert len(infos) == 2, infos
    assert infos[0].startswith('Simulating 2000 time steps of Δt = 0.1 μs')
    assert infos[1].startswith('Simulated 2000 samples')
    assert capsys.readouterr().out == ''


def test_notify_end_is_success_when_passed_and_error_when_failed(records):
    passed, failed = testing.Test('passes'), testing.Test('fails')
    for test in (passed, failed):
        test.notify_start()
    failed.exceptions.append(RuntimeError('boom'))
    passed.notify_end()
    failed.notify_end()
    assert len(at(records, 'SUCCESS')) == 1
    errors = at(records, 'ERROR')
    assert len(errors) == 1 and 'boom' in errors[0]


# functions (by qualified name, per module) whose printing IS their documented output
DOCUMENTED_PRINTERS = {
    'levisim/core/wrappers.py': {'profiler.decorator.wrapper'},
    'levisim/core/testing.py': {'Test.notify_start', 'Test.notify_end'},
}


def print_calls(path):
    """(qualified name of the enclosing function, line) of every print() call in a module"""
    calls = []

    def visit(node, scope):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                visit(child, scope + [child.name])
            else:
                if (
                    isinstance(child, ast.Call)
                    and isinstance(child.func, ast.Name)
                    and child.func.id == 'print'
                ):
                    calls.append(('.'.join(scope), child.lineno))
                visit(child, scope)

    with warnings.catch_warnings():
        # some docstrings hold LaTeX with invalid escape sequences, which ast.parse warns about
        warnings.simplefilter('ignore')
        tree = ast.parse(path.read_text())
    visit(tree, [])
    return calls


def test_print_is_only_used_where_printing_is_the_documented_output():
    """
    print() belongs only in the `info`-style properties (which print their info_str) and the
    few helpers whose console output is their purpose; everything else goes through the logger
    """
    offenders = []
    for path in sorted(LEVISIM_ROOT.rglob('*.py')):
        relative = path.relative_to(LEVISIM_ROOT.parent).as_posix()
        if relative == 'levisim/core/logger.py':
            continue
        for scope, line in print_calls(path):
            function = scope.split('.')[-1] if scope else ''
            if function == 'info' or function.endswith('_info') or function.startswith('info_'):
                continue
            if scope in DOCUMENTED_PRINTERS.get(relative, set()):
                continue
            offenders.append(f'{relative}:{line} ({scope or "module level"})')
    assert not offenders, offenders

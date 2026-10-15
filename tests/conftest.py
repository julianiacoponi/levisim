"""
Shared fixtures and markers for the levisim test suite

Several of these tests are converted from the notebooks in notebooks/multiphysics/tests/.
Those notebooks built `MultiTrap` systems and compared spectra or noise vectors across
calculation methods, but reported failures by appending to a `Test.exceptions` list and
printing PASSED/FAILED - so a failing notebook still ran to completion and the failure was
only visible by reading the printed output. Here they raise, as tests should.
"""
import pytest

from levisim.multiphysics.multitrap import MultiTrap
from levisim.multiphysics.utils.shelf import (
    DEFAULT_cavity_kwargs,
    DEFAULT_gas_kwargs,
    PHASE_DEPENDENT_toy_params,
)


def pytest_configure(config):
    config.addinivalue_line(
        'markers',
        'slow: test takes more than ~10 seconds (deselect with -m "not slow")',
    )
    config.addinivalue_line(
        'markers',
        'needs_power_series: test needs a power-series pickle under notebooks/',
    )


def parse_system(system):
    """'3×2' -> (3, 2), i.e. 3 traps with 2 Cartesian axes each"""
    traps, axes = system.split('×')
    return int(traps), int(axes)


@pytest.fixture
def make_mtrap():
    """
    Builds a fresh MultiTrap per test

    Deliberately does NOT reuse the cached shelf fixtures (PHASE_DEPENDENT_xy_dict etc.):
    the notebooks assigned to `mtrap.ω_range` on those shared objects, which leaks state
    between tests now that the fixtures are built once and cached.
    """
    def _make(system, ω_range=None, **kwargs):
        num_traps, axes = parse_system(system)
        kwargs.setdefault('gas_kwargs', DEFAULT_gas_kwargs)
        kwargs.setdefault('cavity_kwargs', DEFAULT_cavity_kwargs)
        if ω_range is not None:
            kwargs['ω_range'] = ω_range
        return MultiTrap(
            num_traps=num_traps,
            Cartesian_axes_per_trap=axes,
            **kwargs,
        )
    return _make


@pytest.fixture
def make_phase_dependent_mtrap(make_mtrap):
    """A MultiTrap using the phase-dependent toy parameters, as the noise-vector notebooks do"""
    def _make(system, ω_range=None, sign=+1, **kwargs):
        num_traps, axes = parse_system(system)
        kwargs.setdefault('toy_params', PHASE_DEPENDENT_toy_params)
        kwargs.setdefault('tweezer_phase_sign_convention', sign)
        kwargs.setdefault('load_in_power_series', num_traps * axes <= 6)
        return make_mtrap(system, ω_range=ω_range, **kwargs)
    return _make

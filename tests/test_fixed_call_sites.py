"""
Regression tests for the three call sites that raised NameError

Each of these functions was unreachable - calling it raised immediately - and none was
exercised anywhere in the repo, which is why the bugs survived.
"""
import numpy as np
import pytest

from levisim.data.trace import Trace
from levisim.physics.focussing_optics import FocussingOptics


@pytest.fixture
def trace():
    """A short synthetic trace, enough to exercise the info property"""
    rng = np.random.default_rng(seed=0)
    return Trace(
        values=rng.standard_normal(1_000),
        start_time=0.0,
        end_time=1e-3,
        Δt=1e-6,
        axis='x',
        units='m',
        source=None,
    )


def test_trace_info_does_not_raise(trace, capsys):
    """`Trace.info` is a property, so merely accessing it used to raise NameError"""
    trace.info
    printed = capsys.readouterr().out
    assert 'num_samples' in printed
    assert 'Megasamples' in printed


def test_trace_info_reports_sample_count_in_megasamples(trace, capsys):
    """1,000 samples is 0.001 megasamples - checks the divisor, not just that it runs"""
    trace.info
    printed = capsys.readouterr().out
    line = next(l for l in printed.splitlines() if 'num_samples' in l)
    assert '0.001000' in line, line


def test_focussing_optics_scattering_force_does_not_raise():
    """F_scattering referenced a bare `k`; the wavenumber is a cached_property on the class"""
    optics = FocussingOptics()
    force = optics.F_scattering(S=1.0)
    assert np.isfinite(force)


def test_scattering_force_is_linear_in_intensity():
    """The expression is sigma/c * S, so doubling S must double the force"""
    optics = FocussingOptics()
    assert optics.F_scattering(S=2.0) == pytest.approx(2 * optics.F_scattering(S=1.0))


def test_scattering_force_uses_the_instance_wavenumber():
    """A shorter wavelength means a larger k, and so a larger scattering force"""
    red = FocussingOptics(λ=1550e-9)
    green = FocussingOptics(λ=532e-9)
    assert green.k > red.k
    assert green.F_scattering(S=1.0) > red.F_scattering(S=1.0)


def test_comparison_module_exposes_colours_50():
    """comparison.py plotted with colours_50 but never imported it"""
    from levisim.multiphysics.utils import comparison
    assert len(comparison.colours_50) >= 10
    assert all(colour.startswith('#') for colour in comparison.colours_50)

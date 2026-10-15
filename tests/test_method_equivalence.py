"""
Equivalence of the spectrum calculation methods

Converted from notebooks/multiphysics/tests/test_all_systems.ipynb and
test_systems_from_papers.ipynb. Those notebooks compared methods by rendering each spectrum to
a PNG and counting differing pixels (verify_equivalence_of_spectrum_calculation_methods, which
needs OpenCV and writes image files). These tests compare the spectrum arrays directly via
mtrap.homodyne_spectrum / heterodyne_spectrum / all_position_power_spectra, using the project's
own relative-difference comparator so the thresholds stay the project's convention.

The analytic methods are known to disagree with the others - the notebook section comparing them
is headed "BROKEN!" and multitrap.py carries "TODO: fix analytic methods for z motion". That
disagreement is recorded here as an xfail rather than hidden, so it stays visible and any change
in its status is reported by the suite.
"""
import numpy as np
import pytest

from levisim.core.units import kHz
from levisim.multiphysics.enums.quantum_linear_theory_method import QuantumLinearTheoryMethod
from levisim.multiphysics.enums.spectrum import SpectrumType
from levisim.multiphysics.utils.comparison import (
    RelativeDifferenceTooHigh,
    assert_arrays_agree_within_threshold as assert_arrays_agree,
)
from levisim.multiphysics.utils.shelf import OFF_NODE_toy_params_pairs, OFF_NODE_toy_params_equal_spacing_simple

LAM = QuantumLinearTheoryMethod.LINEAR_AMPLIFIER_MODEL
HYBRID_LAM = QuantumLinearTheoryMethod.HYBRID_LINEAR_AMPLIFIER_MODEL
TNT_LAM = QuantumLinearTheoryMethod.TNT_LINEAR_AMPLIFIER_MODEL
NUMERICAL_H = QuantumLinearTheoryMethod.NUMERICAL_HYBRIDISATION
POWER_SERIES_H = QuantumLinearTheoryMethod.POWER_SERIES_HYBRIDISATION
ANALYTIC_INSIDE = QuantumLinearTheoryMethod.ANALYTIC_INSIDE_CAVITY
ANALYTIC_OUTPUT = QuantumLinearTheoryMethod.ANALYTIC_CAVITY_OUTPUT

# a coarse sweep across the mechanical resonances - enough to catch a method disagreeing,
# small enough to keep the suite fast (the notebooks swept 1500+ points and rendered PNGs,
# which is why they took minutes; comparing the arrays directly takes under 2 seconds)
OMEGA_RANGE = np.arange(250, 320, 0.5) * kHz

SYSTEMS = ['1×2', '2×2']


def build(make_mtrap, system, **kwargs):
    """A MultiTrap on the off-node toy parameters the all-systems notebook used"""
    num_traps, axes = (int(_) for _ in system.split('×'))
    toy_params = (
        OFF_NODE_toy_params_equal_spacing_simple if axes == 1
        else OFF_NODE_toy_params_pairs
    )
    kwargs.setdefault('toy_params', toy_params)
    kwargs.setdefault('load_in_power_series', num_traps * axes <= 6)
    return make_mtrap(system, ω_range=OMEGA_RANGE, **kwargs)


def position_psd_arrays(mtrap, methods):
    """{label: array} of zero-point-scaled position power spectra, one entry per method"""
    arrays = {}
    for method in methods:
        values, _ = mtrap.all_position_power_spectra(method, zpf_scaled=True)
        arrays[f'position PSD via {method!r}'] = np.asarray(values)
    return arrays


def optical_spectrum_arrays(mtrap, methods, spectrum_type):
    """{label: array} of homodyne or heterodyne spectra, one entry per method"""
    getter = {
        SpectrumType.HOMODYNE: mtrap.homodyne_spectrum,
        SpectrumType.HETERODYNE: mtrap.heterodyne_spectrum,
    }[spectrum_type]
    return {f'{spectrum_type!r} via {method!r}': np.asarray(getter(method)) for method in methods}


@pytest.mark.parametrize('system', SYSTEMS)
def test_position_psds_agree_across_non_analytic_methods(make_mtrap, system):
    """
    LAM, Hybrid LAM, TNT LAM, Numerical H and Power Series H must give the same position PSDs
    """
    mtrap = build(make_mtrap, system)
    methods = [LAM, HYBRID_LAM, TNT_LAM, NUMERICAL_H]
    if mtrap.num_dimensions <= 6:
        methods.append(POWER_SERIES_H)
    assert_arrays_agree(mtrap, position_psd_arrays(mtrap, methods))


@pytest.mark.parametrize('spectrum_type', [SpectrumType.HOMODYNE, SpectrumType.HETERODYNE])
@pytest.mark.parametrize('system', SYSTEMS)
def test_optical_spectra_agree_across_non_analytic_methods(make_mtrap, system, spectrum_type):
    """
    Homodyne and heterodyne spectra must agree across the non-analytic methods

    TNT LAM is excluded: it does not account for input-output theory, so it cannot produce
    optical spectra (comparison.py skips it for optical spectrum types for the same reason).
    """
    mtrap = build(make_mtrap, system)
    methods = [LAM, HYBRID_LAM, NUMERICAL_H]
    if mtrap.num_dimensions <= 6:
        methods.append(POWER_SERIES_H)
    assert_arrays_agree(
        mtrap, optical_spectrum_arrays(mtrap, methods, spectrum_type)
    )


@pytest.mark.parametrize('system', SYSTEMS)
@pytest.mark.xfail(
    raises=RelativeDifferenceTooHigh,
    reason=(
        'the analytic methods disagree with LAM and Numerical H - known issue, flagged in '
        'test_all_systems.ipynb under the heading "BROKEN!" and by the "TODO: fix analytic '
        'methods for z motion" note in multitrap.py'
    ),
    strict=False,
)
def test_analytic_methods_agree_with_the_rest(make_mtrap, system):
    """Documents the known analytic-vs-rest disagreement rather than leaving it unrecorded"""
    mtrap = build(make_mtrap, system)
    methods = [ANALYTIC_INSIDE, ANALYTIC_OUTPUT, LAM, NUMERICAL_H]
    assert_arrays_agree(
        mtrap, optical_spectrum_arrays(mtrap, methods, SpectrumType.HOMODYNE)
    )

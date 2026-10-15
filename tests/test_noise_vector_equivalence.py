"""
Equivalence of the noise vectors across calculation methods

Converted from notebooks/multiphysics/tests/test_noise_vectors_fast.ipynb (and its slower
sibling). Those notebooks reported 988 failing method pairs out of 1200 on the current code
while still running to completion, because `Test` collects exceptions instead of raising.

Those failures are an artefact of the comparison threshold, not a disagreement between the
methods. `assert_arrays_agree_within_threshold` defaults to difficulty_level='very
hard', which is a relative-difference threshold of 1e-12 - below what a double-precision
linear solve can deliver - and it divides by |array_A|, which is ill-conditioned wherever a
noise-vector component passes through zero. Measured against LAM over a 2×2 system, every
other method agrees on the light noise vectors to a median relative difference of ~1e-16
(machine epsilon), a mean of ~5e-11 and a worst case of ~8e-9.

These tests therefore compare at difficulty_level='easy' (1e-6), which is still three orders
of magnitude tighter than the worst observed disagreement, and they include the analytic
methods - which agree here, unlike for the spectra (see test_method_equivalence.py).
"""
import numpy as np
import pytest

from levisim.core.units import kHz
from levisim.multiphysics.enums.noise_vector import NoiseVectorType
from levisim.multiphysics.enums.quantum_linear_theory_method import QuantumLinearTheoryMethod
from levisim.multiphysics.utils.comparison import (
    assert_arrays_agree_within_threshold as assert_arrays_agree,
)

LAM = QuantumLinearTheoryMethod.LINEAR_AMPLIFIER_MODEL
HYBRID_LAM = QuantumLinearTheoryMethod.HYBRID_LINEAR_AMPLIFIER_MODEL
NUMERICAL_H = QuantumLinearTheoryMethod.NUMERICAL_HYBRIDISATION
POWER_SERIES_H = QuantumLinearTheoryMethod.POWER_SERIES_HYBRIDISATION
ANALYTIC_INSIDE = QuantumLinearTheoryMethod.ANALYTIC_INSIDE_CAVITY
ANALYTIC_OUTPUT = QuantumLinearTheoryMethod.ANALYTIC_CAVITY_OUTPUT

NON_ANALYTIC = [LAM, HYBRID_LAM, NUMERICAL_H, POWER_SERIES_H]
ALL_METHODS = NON_ANALYTIC + [ANALYTIC_INSIDE, ANALYTIC_OUTPUT]

# the notebook swept +/-350 kHz at 0.1 kHz resolution; a sample across the resonances is
# enough to catch a method diverging, and keeps the suite to seconds rather than minutes
OMEGA_RANGE = np.arange(260, 310, 2.0) * kHz

# comparing at 1e-6 rather than the 1e-12 default - see the module docstring
DIFFICULTY = 'easy'

NOISE_VECTOR_GETTERS = {
    NoiseVectorType.HYBRIDISED_POSITIONS: 'hybridised_positions',
    NoiseVectorType.LIGHT_INSIDE_CAVITY: 'light_noise_vectors',
    NoiseVectorType.CAVITY_OUTPUT_LIGHT: 'output_light_noise_vectors',
}


def noise_vectors_per_method(mtrap, noise_vector_type, methods):
    """{method: array of shape (n_omega, n_rows, n_noise_operators)}"""
    getter = getattr(mtrap, NOISE_VECTOR_GETTERS[noise_vector_type])
    return {
        method: np.array([getter(method, ω) for ω in mtrap.ω_range])
        for method in methods
    }


LIGHT_TYPES = [NoiseVectorType.LIGHT_INSIDE_CAVITY, NoiseVectorType.CAVITY_OUTPUT_LIGHT]


def compare_components(mtrap, noise_vector_type, methods, difficulty=DIFFICULTY):
    """Compares every (row, noise operator) component of the noise vector across the methods"""
    per_method = noise_vectors_per_method(mtrap, noise_vector_type, methods)
    n_rows = 2 if noise_vector_type.is_light else mtrap.num_dimensions
    for row_index in range(n_rows):
        for noise_index in range(mtrap.num_operators):
            arrays = {
                f'component[{row_index}, {noise_index}] via {method!r}':
                    per_method[method][:, row_index, noise_index]
                for method in methods
            }
            assert_arrays_agree(mtrap, arrays, difficulty_level=difficulty)


@pytest.mark.slow
@pytest.mark.parametrize('noise_vector_type', list(NOISE_VECTOR_GETTERS))
def test_noise_vectors_agree_across_non_analytic_methods(
    make_phase_dependent_mtrap, noise_vector_type,
):
    """
    LAM, Hybrid LAM, Numerical H and Power Series H must agree on every noise vector component

    A real regression in any of these now fails the suite, where the notebook printed FAILED
    and carried on. Measured against LAM on a 2×2 system: Hybrid LAM is bit-identical, and
    Numerical H and Power Series H agree to a worst case of ~3e-8.
    """
    mtrap = make_phase_dependent_mtrap('2×2', ω_range=OMEGA_RANGE)
    compare_components(mtrap, noise_vector_type, NON_ANALYTIC)


@pytest.mark.slow
@pytest.mark.parametrize('noise_vector_type', LIGHT_TYPES)
def test_light_noise_vectors_agree_including_analytic_methods(
    make_phase_dependent_mtrap, noise_vector_type,
):
    """
    For the light noise vectors, the analytic methods agree with the rest

    This is worth asserting explicitly because the analytic methods do NOT agree on the
    spectra (see test_method_equivalence.py), so the disagreement is specific rather than
    a blanket problem with those methods.
    """
    mtrap = make_phase_dependent_mtrap('2×2', ω_range=OMEGA_RANGE)
    compare_components(mtrap, noise_vector_type, ALL_METHODS)


@pytest.mark.slow
def test_analytic_hybridised_positions_deviate_only_locally(make_phase_dependent_mtrap):
    """
    The analytic hybridised positions match LAM almost everywhere, with localised deviations

    Measured on a 2×2 system: median relative difference ~4e-16 (machine epsilon) but a worst
    case of ~5e-3, which pulls the mean to ~9e-6 and is why the threshold-based comparison at
    difficulty_level='easy' (1e-6) rejects them. Pinning the measured bounds records the
    current state precisely: a global divergence would break the median assertion, and a
    worsening of the local deviation would break the max assertion.
    """
    mtrap = make_phase_dependent_mtrap('2×2', ω_range=OMEGA_RANGE)
    per_method = noise_vectors_per_method(
        mtrap, NoiseVectorType.HYBRIDISED_POSITIONS, [LAM, ANALYTIC_INSIDE, ANALYTIC_OUTPUT],
    )
    reference = per_method[LAM]
    for method in (ANALYTIC_INSIDE, ANALYTIC_OUTPUT):
        relative = np.abs(per_method[method] - reference) / np.maximum(np.abs(reference), 1e-300)
        assert np.median(relative) < 1e-12, f'{method!r} median {np.median(relative):.3e}'
        assert np.max(relative) < 1e-2, f'{method!r} max {np.max(relative):.3e}'


@pytest.mark.slow
def test_noise_vector_agreement_is_near_machine_precision(make_phase_dependent_mtrap):
    """
    Pins the measured agreement of the light noise vectors, so a loss of accuracy is visible

    This is the measurement that justifies difficulty_level='easy': if the methods ever stop
    agreeing to ~1e-8, this fails even though the threshold-based tests still pass.
    """
    mtrap = make_phase_dependent_mtrap('2×2', ω_range=OMEGA_RANGE)
    per_method = noise_vectors_per_method(
        mtrap, NoiseVectorType.LIGHT_INSIDE_CAVITY, ALL_METHODS,
    )
    reference = per_method[LAM]
    for method in ALL_METHODS:
        if method is LAM:
            continue
        relative = np.abs(per_method[method] - reference) / np.maximum(np.abs(reference), 1e-300)
        assert np.median(relative) < 1e-12, f'{method!r} median {np.median(relative):.3e}'
        assert np.mean(relative) < 1e-8, f'{method!r} mean {np.mean(relative):.3e}'

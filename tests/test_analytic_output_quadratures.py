"""
Output-light quadratures and the per-dimension cooling rate

- `homodyne_and_heterodyne_spectra` used exp(θhom) (a real number) instead of exp(iθhom) for the
  homodyne quadrature, so for the default θhom = π/4 its homodyne spectrum was 1.4-12.8 times
  `homodyne_spectrum` (2×2 PHASE_DEPENDENT system). Both now use `homodyne_quadrature`.
- `Dimension.cooling_rate` is the weak-coupling optical damping rate; it is checked against the
  exact damping of the drift-matrix eigenvalue, which it matches to O(|G|²/κ²).
"""
import io
import contextlib
from copy import copy

import numpy as np
import pytest

from levisim.core.units import kHz
from levisim.multiphysics.enums.quantum_linear_theory_method import QuantumLinearTheoryMethod
from levisim.multiphysics.utils.shelf import OFF_NODE_toy_params_pairs, PHASE_DEPENDENT_toy_params

LAM = QuantumLinearTheoryMethod.LINEAR_AMPLIFIER_MODEL
ANALYTIC_OUTPUT = QuantumLinearTheoryMethod.ANALYTIC_CAVITY_OUTPUT
OMEGA_RANGE = np.arange(250, 320, 0.5) * kHz


def quietly(function, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return function(*args, **kwargs)


@pytest.mark.parametrize('method', [LAM, ANALYTIC_OUTPUT])
def test_combined_spectra_equal_the_separate_ones(make_mtrap, method):
    """homodyne_and_heterodyne_spectra returns exactly homodyne_spectrum and heterodyne_spectrum"""
    mtrap = quietly(make_mtrap, '2×2', ω_range=OMEGA_RANGE, toy_params=PHASE_DEPENDENT_toy_params, load_in_power_series=False)
    assert mtrap.cavity.θhom != 0, 'θhom = 0 would hide a missing factor of i'
    homodyne, heterodyne = quietly(mtrap.homodyne_and_heterodyne_spectra, method)
    np.testing.assert_array_equal(homodyne, quietly(mtrap.homodyne_spectrum, method))
    np.testing.assert_array_equal(heterodyne, quietly(mtrap.heterodyne_spectrum, method))


def test_homodyne_quadrature_is_a_rotation(make_mtrap):
    """h = a_out·exp(iθ) + a_out†·exp(-iθ), as the docstrings state"""
    mtrap = quietly(make_mtrap, '2×2', ω_range=OMEGA_RANGE, toy_params=PHASE_DEPENDENT_toy_params, load_in_power_series=False)
    a_out_hat, a_out_dag = mtrap.output_light_noise_vectors(LAM, 300 * kHz)
    θ = mtrap.cavity.θhom
    np.testing.assert_allclose(
        mtrap.homodyne_quadrature(a_out_hat, a_out_dag),
        a_out_hat * np.exp(1j * θ) + a_out_dag * np.exp(-1j * θ),
        rtol=1e-14, atol=1e-14 * np.max(np.abs(a_out_hat)),
    )


@pytest.mark.parametrize('g, rtol', [(-5 * kHz, 2e-4), (-20 * kHz, 3e-3), (-40 * kHz, 1e-2)])
def test_cooling_rate_matches_drift_matrix_damping(make_mtrap, g, rtol):
    """
    Γ_opt = κ|G|²[1/((Δ+Ω)²+(κ/2)²) - 1/((Δ-Ω)²+(κ/2)²)] versus -2·Re(λ) - Γ of the mechanical
    eigenvalue λ of the 1×1 drift matrix (measured ratios 1.00007, 1.0012, 1.006: the weak-coupling
    formula is accurate to O(|G|²/κ²), κ = 2π·390 kHz)
    """
    params = copy(OFF_NODE_toy_params_pairs)
    params['toy_gx'] = g
    mtrap = quietly(make_mtrap, '1×1', ω_range=OMEGA_RANGE, toy_params=params, load_in_power_series=False)
    dim = mtrap.dimensions['x1']
    eigenvalues = np.linalg.eigvals(mtrap.drift_matrix)
    mechanical = eigenvalues[np.argmin(np.abs(eigenvalues.imag + dim.Ω))]
    assert dim.cooling_rate > 0, 'red detuning must cool'
    np.testing.assert_allclose(dim.cooling_rate, -2 * mechanical.real - dim.Γ, rtol=rtol)

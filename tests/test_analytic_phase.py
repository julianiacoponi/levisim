"""
The analytic closed form (ANALYTIC_INSIDE_CAVITY / ANALYTIC_CAVITY_OUTPUT) versus LAM, for systems
without intra-bead mechanical coupling: at the node, and with unequal tweezer phases

Until Nov 2026 the analytic methods agreed with LAM to only ~1e-9 on the optical spectra and
0.2-0.6% on the position PSDs whenever the tweezer phases were unequal. The cause was numerical,
not a sign convention: the frequency grid hits mechanical resonances exactly (e.g. Ωy2 = 300.5 kHz
on a 0.5 kHz grid), where |g²·μ| ~ 1e13 rad/s, and P = D(0)² - D(2)·D(-2) and the thermal
brackets F_Q·G + F_Q†·G* were evaluated by subtracting such terms. The resonant dimension's
position then amplified that ~1e-9 error by |M_jn| ~ 1e7. See `MultiTrap.analytic_kernel` and
`MultiTrap.analytic_hybridised_positions` for the cancellation-free forms.

Tolerances: LAM is itself accurate to ~1e-14 here (checked below against a 40-digit mpmath
solve of the same Langevin equations), and the analytic closed form agrees with it to at most
7.5e-14 on every system in this file (measured, 0.5 kHz grid over 250-320 kHz). The comparisons
use rtol = 1e-12, i.e. >10x headroom over the measured worst case, and 1e3 times tighter than
the old 1e-9 floor, so a reintroduced cancellation fails.
"""
import io
import contextlib

import numpy as np
import pytest

from levisim.core.units import kHz
from levisim.multiphysics.enums.quantum_linear_theory_method import QuantumLinearTheoryMethod
from levisim.multiphysics.utils.shelf import (
    OFF_NODE_toy_params_pairs,
    PHASE_DEPENDENT_toy_params,
    PHASE_EQUAL_toy_params,
)

LAM = QuantumLinearTheoryMethod.LINEAR_AMPLIFIER_MODEL
ANALYTIC_INSIDE = QuantumLinearTheoryMethod.ANALYTIC_INSIDE_CAVITY
ANALYTIC_OUTPUT = QuantumLinearTheoryMethod.ANALYTIC_CAVITY_OUTPUT

# 0.5 kHz steps land exactly on the toy mechanical frequencies (305, 307.5, 298, 300.5, ... kHz),
# which is the worst case for the closed form
OMEGA_RANGE = np.arange(250, 320, 0.5) * kHz
RTOL = 1e-12

# (system, toy parameters): none of these has intra-bead coupling (1 axis per trap, or the
# PHASE_* parameters, which only set x and y couplings)
CASES = {
    'node 2×2': ('2×2', PHASE_EQUAL_toy_params),
    'node 3×2': ('3×2', PHASE_EQUAL_toy_params),
    'unequal phases 2×2': ('2×2', PHASE_DEPENDENT_toy_params),
    'unequal phases 3×2': ('3×2', PHASE_DEPENDENT_toy_params),
    'unequal phases 4×2': ('4×2', PHASE_DEPENDENT_toy_params),
    'off node 1×1': ('1×1', OFF_NODE_toy_params_pairs),
    'off node, unequal phases 2×1': ('2×1', OFF_NODE_toy_params_pairs),
}


def quietly(function, *args, **kwargs):
    """The spectrum functions print a progress line per call"""
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return function(*args, **kwargs)


def spectra(mtrap, method):
    """(homodyne, heterodyne, zero-point-scaled position PSDs) for one method"""
    return (
        np.asarray(quietly(mtrap.homodyne_spectrum, method)),
        np.asarray(quietly(mtrap.heterodyne_spectrum, method)),
        np.asarray(quietly(mtrap.all_position_power_spectra, method, zpf_scaled=True)[0]),
    )


@pytest.fixture
def build(make_mtrap):
    def _build(case, **kwargs):
        system, toy_params = CASES[case]
        return quietly(
            make_mtrap, system, ω_range=OMEGA_RANGE, toy_params=toy_params,
            load_in_power_series=False, **kwargs,
        )
    return _build


@pytest.mark.parametrize('method', [ANALYTIC_INSIDE, ANALYTIC_OUTPUT])
@pytest.mark.parametrize('case', list(CASES))
def test_analytic_spectra_match_LAM(build, case, method):
    """Homodyne, heterodyne and every position PSD agree with LAM to rtol = 1e-12"""
    mtrap = build(case)
    for name, analytic, reference in zip(
        ['homodyne', 'heterodyne', 'position PSDs'], spectra(mtrap, method), spectra(mtrap, LAM),
    ):
        np.testing.assert_allclose(analytic, reference, rtol=RTOL, atol=0, err_msg=f'{case}: {name}')


def test_P_cancellation_free_form(build):
    """
    P = 2i·hᵀ·I·h̃ equals D(0)² - D(2)·D(-2) away from resonance, and equals the pairwise sum
    4·Σ gj²μj·gk²μk·sin²(θj - θk) everywhere - including exactly on a resonance, where the
    subtraction form is only good to ~1e-9
    """
    mtrap = build('unequal phases 2×2')
    off_resonance, on_resonance = 290 * kHz, mtrap.dimensions['y2'].Ω
    D = mtrap.D
    for ω in (off_resonance, on_resonance):
        np.testing.assert_allclose(mtrap.P(ω), mtrap.P_via_Pjk(ω), rtol=1e-14)
        np.testing.assert_allclose(mtrap.P(ω), mtrap.P_via_Pj(ω), rtol=1e-14)
    np.testing.assert_allclose(mtrap.P(off_resonance), D(0, off_resonance)**2 - D(2, off_resonance) * D(-2, off_resonance), rtol=1e-14)
    subtraction_error = abs((D(0, on_resonance)**2 - D(2, on_resonance) * D(-2, on_resonance)) / mtrap.P(on_resonance) - 1)
    assert subtraction_error > 1e-12, 'the resonance is no longer a stress test for P'


def test_M_norm_agrees_with_all_alternative_forms_on_resonance(build):
    """M_norm (via the cancellation-free P) agrees with the K-based forms to ~1e-15 on resonance"""
    mtrap = build('unequal phases 2×2')
    ω = mtrap.dimensions['y2'].Ω
    reference = mtrap.M_norm(ω)
    for suffix in ['2', '3', '4', '5', '6', '7', '2_2', '3_2', '4_2', '5_2', '6_2', '7_2']:
        np.testing.assert_allclose(getattr(mtrap, f'M_norm_{suffix}')(ω), reference, rtol=1e-14, err_msg=suffix)


def test_thermal_bracket_identity(build):
    """D(0)·G - D(2)·G* = 2i·Iᵀ·h and D(0)·G* - D(-2)·G = -2i·Iᵀ·h̃, used by the thermal brackets"""
    mtrap = build('unequal phases 3×2')
    ω = 287.3 * kHz
    kernel = mtrap.analytic_kernel(ω)
    G, I = mtrap.G_array, mtrap.coupling_quadrature_overlap_matrix
    D0, D2, Dm2 = (kernel['D'][n] for n in (0, 2, -2))
    np.testing.assert_allclose(D0 * G - D2 * np.conjugate(G), 2j * I.T @ kernel['h'], rtol=1e-12, atol=0)
    np.testing.assert_allclose(D0 * np.conjugate(G) - Dm2 * G, -2j * I.T @ kernel['h_tilde'], rtol=1e-12, atol=0)


@pytest.mark.parametrize('method', [LAM, ANALYTIC_INSIDE])
def test_tweezer_phase_sign_convention_is_inert(build, method):
    """
    The sign multiplies φ_coupling and multiplies it again in G = g·exp(sign·i·φ_coupling), so it
    cancels: the spectra are bit-identical for sign = ±1 (see the NOTE in Dimension.φ_coupling)
    """
    plus = spectra(build('unequal phases 2×2', tweezer_phase_sign_convention=+1), method)
    minus = spectra(build('unequal phases 2×2', tweezer_phase_sign_convention=-1), method)
    for a, b in zip(plus, minus):
        np.testing.assert_array_equal(a, b)


def test_exact_resonance_against_high_precision_reference(build):
    """
    On the y2 resonance of the unequal-phase 2×2 system, LAM and the analytic positions both match
    a 40-digit mpmath solve of (-iω·I - A)·c = c_in to ~1e-14, i.e. LAM is a valid reference
    """
    mpmath = pytest.importorskip('mpmath')
    mtrap = build('unequal phases 2×2')
    ω = mtrap.dimensions['y2'].Ω
    with mpmath.workdps(40):
        n = mtrap.num_operators
        A = mtrap.drift_matrix
        matrix = mpmath.matrix(n, n)
        for ii in range(n):
            for jj in range(n):
                matrix[ii, jj] = (-1j * mpmath.mpf(ω) if ii == jj else 0) - mpmath.mpc(A[ii, jj].real, A[ii, jj].imag)
        T = mpmath.inverse(matrix)
        c_in = [mpmath.sqrt(mpmath.mpf(γ)) for γ in mtrap.γ]
        factors = [mpmath.mpf(f) for f in mtrap.quantum_factor_per_noise_operator]
        exact = []
        for label in mtrap.labels:
            ai = mtrap.label_to_noise_index[label]
            exact.append(float(sum(
                factors[jj] * abs((T[ai, jj] + T[ai + 1, jj]) * c_in[jj])**2 for jj in range(n)
            )))
    exact = np.array(exact)
    for method in (LAM, ANALYTIC_INSIDE, ANALYTIC_OUTPUT):
        positions = mtrap.hybridised_positions(method, ω)
        psds = np.array([mtrap.noise_vector_to_power_spectrum(q) for q in positions])
        np.testing.assert_allclose(psds, exact, rtol=1e-13, err_msg=repr(method))

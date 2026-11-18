"""
Complex-conjugation relations and equivalent forms of the analytic (closed-form) functions

Ported from the notebook notebooks/multiphysics/tests/test_analytic_complex_conjugation_relations
(deleted in Nov 2026). The notebook checked, on the 2×2 PHASE_DEPENDENT toy system:

1. the normalisation function is real on the real axis in the Fourier sense,
   M_norm*(ω) = M_norm(-ω);
2. each light-noise transduction function into Q is the conjugate of the matching one into Q†
   at -ω: Fa*(ω) = FFa_dag(-ω) and FFa*(ω) = Fa_dag(-ω);
3. likewise for the thermal-noise transduction functions: FQ*(ω) = FFQ_dag(-ω) and
   FFQ*(ω) = FQ_dag(-ω);
4. all alternative closed forms of M_norm (M_norm_2 ... M_norm_7, M_norm_2_2 ... M_norm_7_2,
   M_norm_N and M_norm_N_2 for each option K, H, U, H_dag, U_dag) equal M_norm.

(FFa, FFa_dag, FFQ, FFQ_dag are levisim's names for the transduction functions into the
collective mode Q†; the doubled letter is not a conjugation.)

The notebook compared products such as Fa(ω)·FFa_dag(-ω) with |Fa(ω)|², which is equivalent
to the conjugation relation wherever Fa ≠ 0; here the relation is tested directly.

Tolerances: the notebook noted that differences grew to ~1e-10 near mechanical resonances,
and deliberately left its checks failing at that level. That growth was the catastrophic
cancellation in P = D(0)² - D(2)·D(-2) that `MultiTrap.analytic_kernel` now avoids. Measured
on current main (Nov 2026) on the notebook's grid plus the exact resonances ±Ωjn, the
largest relative difference is 2.2e-14 (M_norm on 3×2); the tests use rtol = 1e-12, ~50x
headroom, so a reintroduced cancellation (~1e-9) fails.
"""
import io
import contextlib

import numpy as np
import pytest

from levisim.core.units import kHz
from levisim.multiphysics.utils.shelf import PHASE_DEPENDENT_toy_params

RTOL = 1e-12

# the notebook's grid: dense around both mechanical bands (±250-350 kHz), sparse in between
OMEGA_GRID = np.concatenate((
    np.arange(-350, -250, 0.1) * kHz,
    np.arange(-200, 200, 5) * kHz,
    np.arange(250, 350 + 0.1, 0.1) * kHz,
))

# (system, tweezer phase sign convention): the notebook ran 2×2 with sign +1, and had 2×2 with
# sign -1 and 3×2 commented out. None of these has intra-bead coupling (the PHASE_DEPENDENT toy
# parameters only set x and y couplings), which the alternative M_norm forms do not include.
SYSTEMS = [('2×2', +1), ('2×2', -1), ('3×2', +1)]

# f*(ω) = g(-ω)
CONJUGATE_PAIRS = [
    ('M_norm', 'M_norm'),
    ('Fa', 'FFa_dag'),
    ('FFa', 'Fa_dag'),
    ('FQ', 'FFQ_dag'),
    ('FFQ', 'FQ_dag'),
]


def quietly(function, *args, **kwargs):
    """MultiTrap prints a progress line on construction"""
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return function(*args, **kwargs)


def relative_difference(a, b):
    a, b = np.asarray(a), np.asarray(b)
    scale = np.maximum(np.abs(a), np.abs(b))
    return np.abs(a - b) / np.where(scale == 0, 1, scale)


def evaluate(function, ω_values):
    return np.array([function(ω) for ω in ω_values])


@pytest.fixture(params=SYSTEMS, ids=[f'{system} sign {sign:+d}' for system, sign in SYSTEMS])
def mtrap_and_grid(request, make_mtrap):
    system, sign = request.param
    mtrap = quietly(
        make_mtrap, system, ω_range=OMEGA_GRID,
        toy_params=PHASE_DEPENDENT_toy_params,
        tweezer_phase_sign_convention=sign,
        load_in_power_series=False,
    )
    # add the exact resonances, where any cancellation is worst
    Ωs = np.array(sorted({dim.Ω for dim in mtrap.dimensions.values()}))
    return mtrap, np.concatenate((OMEGA_GRID, Ωs, -Ωs))


@pytest.mark.parametrize('f_name, g_name', CONJUGATE_PAIRS, ids=[f'{f}*(ω)={g}(-ω)' for f, g in CONJUGATE_PAIRS])
def test_conjugation_relation(mtrap_and_grid, f_name, g_name):
    mtrap, ω_values = mtrap_and_grid
    f_conjugated = np.conjugate(evaluate(getattr(mtrap, f_name), ω_values))
    g_at_minus_ω = evaluate(getattr(mtrap, g_name), -ω_values)
    worst = relative_difference(f_conjugated, g_at_minus_ω).max()
    assert worst < RTOL, f'{f_name}*(ω) vs {g_name}(-ω): max relative difference {worst:.2e}'


def alternative_M_norm_forms(mtrap):
    forms = {}
    for k in range(2, 8):
        forms[f'M_norm_{k}'] = getattr(mtrap, f'M_norm_{k}')
        forms[f'M_norm_{k}_2'] = getattr(mtrap, f'M_norm_{k}_2')
    for option in ('K', 'H', 'U', 'H_dag', 'U_dag'):
        forms[f'M_norm_N({option!r})'] = lambda ω, option=option: mtrap.M_norm_N(option, ω)
        forms[f'M_norm_N_2({option!r})'] = lambda ω, option=option: mtrap.M_norm_N_2(option, ω)
    return forms


def test_alternative_forms_of_M_norm_agree(mtrap_and_grid):
    mtrap, ω_values = mtrap_and_grid
    reference = evaluate(mtrap.M_norm, ω_values)
    worst = {
        name: relative_difference(evaluate(form, ω_values), reference).max()
        for name, form in alternative_M_norm_forms(mtrap).items()
    }
    assert len(worst) == 22
    failing = {name: f'{value:.2e}' for name, value in worst.items() if not value < RTOL}
    assert not failing, f'alternative M_norm forms disagree with M_norm: {failing}'

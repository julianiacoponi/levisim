"""
Tests of levisim.multiphysics.output_modes: covariance of filtered (propagating) cavity-output modes

The two routes to a filtered-mode covariance (frequency integration of the output cross-spectra,
and a cascaded virtual filter cavity in the drift matrix) are checked against each other, the
building blocks against levisim's own Lyapunov covariance, and a two-cavity-mode system against
the worked example of Deplano et al., Science 394, 113 (2026), supplement Eq. (S45).
"""
import numpy as np
import pytest

from levisim.core.constants import kB, ħ
from levisim.core.units import kHz
from levisim.analysis.gaussian_states import is_physical, partially_transposed_symplectic_eigenvalues
from levisim.multiphysics import MultiTrap
from levisim.multiphysics.covariance import quadrature_diffusion_matrix, steady_state_covariance
from levisim.multiphysics.output_modes import (
    QuadratureSystem,
    add_mechanical_mode,
    cascade_lorentzian_filter,
    filter_norm,
    lorentzian_filter,
    output_mode_covariance,
    rectangular_filter,
    rectangular_filter_scan,
    stack_cavity_modes,
)

T_BATH = 300.0


def toy_multitrap(Ω, g, heating, Δ, κ=58*kHz):
    """
    One particle, one cavity mode; heating {label: phonon heating rate} set through the bath
    damping Γ = heating/n_th of a 300 K bath
    """
    n_th = {label: 1/np.expm1(ħ*Ω[label]/(kB*T_BATH)) for label in Ω}
    toy_params = dict(Ω_model='custom', toy_Ωjn_dict=dict(Ω), gjn_dict=dict(g), use_Γ_formula=False,
                      Γjn_dict={label: heating[label]/n_th[label] for label in Ω}, T=T_BATH)
    return MultiTrap(num_traps=1, Cartesian_axes_per_trap=len(Ω), toy_params=toy_params,
                     cavity_kwargs=dict(κ=κ, Δ=Δ))


# the parameter set of supplement Eq. (S45): (Ωx, Ωy, gx, gy, Γx, Γy, ΔA, ΔB)/2π [kHz], P
S45 = dict(Ωx=110.5, Ωy=98.3, gx=9.0, gy=7.5, Γx=3.0, Γy=2.7, ΔA=-98.6, ΔB=96.6, P=0.291)
S45_MATRIX = np.array([
    [1.5, 0.0, 0.35, 1.29],
    [0.0, 1.5, 1.31, -0.36],
    [0.35, 1.31, 3.89, 0.0],
    [1.29, -0.36, 0.0, 3.81],
])


def deplano_system(p=S45):
    Ω = dict(x1=p['Ωx']*kHz, y1=p['Ωy']*kHz)
    heating = dict(x1=p['Γx']*kHz, y1=p['Γy']*kHz)
    g_A = dict(x1=p['gx']*kHz, y1=p['gy']*kHz)
    g_B = {label: np.sqrt(p['P'])*g for label, g in g_A.items()}
    red = toy_multitrap(Ω, g_A, heating, p['ΔA']*kHz)
    blue = toy_multitrap(Ω, g_B, heating, p['ΔB']*kHz)
    return stack_cavity_modes([red, blue], ['A', 'B'])


def bright_mode_transform(p=S45):
    """
    (Q_ξ, P_ξ, x, p_x, y, p_y) → (Q_ξ, P_ξ, x_b, p_b), Eqs. (S6, S8) with tan θ = √(Ωx/Ωy)·gx/gy
    """
    Ωx, Ωy = p['Ωx'], p['Ωy']
    θ = np.arctan(np.sqrt(Ωx/Ωy) * p['gx']/p['gy'])
    Ωb = np.sqrt(Ωx**2*np.sin(θ)**2 + Ωy**2*np.cos(θ)**2)
    T = np.zeros((4, 6))
    T[0, 0] = T[1, 1] = 1
    T[2, 2], T[2, 4] = np.sqrt(Ωb/Ωx)*np.sin(θ), np.sqrt(Ωb/Ωy)*np.cos(θ)
    T[3, 3], T[3, 5] = np.sqrt(Ωx/Ωb)*np.sin(θ), np.sqrt(Ωy/Ωb)*np.cos(θ)
    return T, Ωb*kHz


@pytest.fixture(scope='module')
def single():
    return toy_multitrap(dict(x1=110*kHz, y1=98*kHz), dict(x1=9*kHz, y1=7.5*kHz), dict(x1=3*kHz, y1=2.7*kHz), -100*kHz)


@pytest.fixture(scope='module')
def deplano():
    return deplano_system()


def test_from_multitrap_matches_levisim_covariance(single):
    system = QuadratureSystem.from_multitrap(single)
    np.testing.assert_allclose(system.diffusion_matrix, quadrature_diffusion_matrix(single), rtol=1e-12)
    V = steady_state_covariance(single)
    np.testing.assert_allclose(system.steady_state_covariance(), V, rtol=1e-9, atol=1e-9*np.abs(V).max())


def test_stack_of_one_is_the_multitrap_and_uncoupled_mode_is_vacuum(single):
    alone = QuadratureSystem.from_multitrap(single, 'A')
    stacked = stack_cavity_modes([single], ['A'])
    np.testing.assert_array_equal(stacked.A, alone.A)
    # a second cavity mode with zero couplings changes nothing and sits in vacuum
    idle = toy_multitrap(dict(x1=110*kHz, y1=98*kHz), dict(x1=0.0, y1=0.0), dict(x1=3*kHz, y1=2.7*kHz), 50*kHz)
    two = stack_cavity_modes([single, idle], ['A', 'B'])
    V1, V2 = alone.steady_state_covariance(), two.steady_state_covariance()
    keep = [0, 1, 4, 5, 6, 7]
    np.testing.assert_allclose(V2[np.ix_(keep, keep)], V1, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(V2[2:4, 2:4], np.eye(2), atol=1e-12)


def test_stack_rejects_different_mechanics(single):
    other = toy_multitrap(dict(x1=111*kHz, y1=98*kHz), dict(x1=9*kHz, y1=7.5*kHz), dict(x1=3*kHz, y1=2.7*kHz), 100*kHz)
    with pytest.raises(AssertionError, match='mechanical drift'):
        stack_cavity_modes([single, other], ['A', 'B'])


def test_add_mechanical_mode_matches_a_multitrap_dimension():
    full = QuadratureSystem.from_multitrap(toy_multitrap(dict(x1=110*kHz), dict(x1=9*kHz), dict(x1=3*kHz), -100*kHz))
    cavity_only = QuadratureSystem(full.A[:2, :2], full.B[:2, :2], full.N[:2, :2], ['cavity'], {'cavity': full.cavities['cavity']})
    Γ = full.B[2, 2]**2
    rebuilt = add_mechanical_mode(cavity_only, 'x1', 110*kHz, {'cavity': 9*kHz}, heating=(full.N[2, 2] - 1)/2*Γ, damping=Γ)
    np.testing.assert_allclose(rebuilt.A, full.A, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(rebuilt.diffusion_matrix, full.diffusion_matrix, rtol=1e-12)


@pytest.mark.parametrize('spectral_filter', [rectangular_filter(-100*kHz, 30*kHz), lorentzian_filter(-100*kHz, 30*kHz)])
def test_filters_are_normalised(spectral_filter):
    assert filter_norm(spectral_filter, ω_limits=(-1e6*kHz, 1e6*kHz), panels=20_000) == pytest.approx(1, abs=1e-4)


def test_uncoupled_or_undetected_output_is_vacuum(single):
    idle = toy_multitrap(dict(x1=110*kHz, y1=98*kHz), dict(x1=0.0, y1=0.0), dict(x1=3*kHz, y1=2.7*kHz), 100*kHz)
    V = output_mode_covariance(QuadratureSystem.from_multitrap(idle), 'cavity', rectangular_filter(-100*kHz, 30*kHz))
    np.testing.assert_allclose(V[:2, :2], np.eye(2), atol=1e-12)
    np.testing.assert_allclose(V[:2, 2:], 0, atol=1e-12)
    V = output_mode_covariance(QuadratureSystem.from_multitrap(single), 'cavity', rectangular_filter(100*kHz, 30*kHz), η=0.0)
    np.testing.assert_allclose(V[:2, :2], np.eye(2), atol=1e-12)
    np.testing.assert_allclose(V[:2, 2:], 0, atol=1e-12)


def test_cascaded_filter_cavity_equals_frequency_integration(deplano):
    """
    Two independent routes to the same Lorentzian-filtered output mode
    """
    center, width = -106*kHz, 25*kHz
    cascaded = cascade_lorentzian_filter(deplano, 'B', center, width, η=0.283)
    V_cascaded = cascaded.steady_state_covariance()
    order = [*cascaded.quadrature_indices('filter'), *cascaded.quadrature_indices('x1'), *cascaded.quadrature_indices('y1')]
    V_cascaded = V_cascaded[np.ix_(order, order)]
    V_integrated = output_mode_covariance(deplano, 'B', lorentzian_filter(center, width), η=0.283, modes=['x1', 'y1'],
                                          panels=400, extra_breakpoints=[-106*kHz, 106*kHz])
    np.testing.assert_allclose(V_integrated, V_cascaded, atol=1e-7)
    assert is_physical(V_cascaded)


def test_rectangular_scan_equals_direct_integration(deplano):
    widths = np.array([20, 27.5, 35, 50]) * kHz
    scan = rectangular_filter_scan(deplano, 'B', -106*kHz, widths, η=0.283, modes=['x1', 'y1'])
    for V, width in zip(scan, widths):
        direct = output_mode_covariance(deplano, 'B', rectangular_filter(-106*kHz, width), η=0.283, modes=['x1', 'y1'])
        np.testing.assert_allclose(V, direct, atol=1e-12)


def test_deplano2026_supplement_example(deplano):
    """
    Eq. (S45) of the supplement: V for (Q_ξ, P_ξ, x_b, p_b) of the propagating B mode (flat window
    centred at −Ω_b) and the bright mode. The paper's model diffuses momentum only, levisim's bath
    both quadratures, which gives 〈x_b p_b〉 ≈ −D/(2Ω_b) ≈ −0.03 instead of 0; every other entry
    agrees to the printed precision at Γ_ξ/2π = 40 kHz.
    """
    assert deplano.largest_growth_rate() < 0
    T, Ωb = bright_mode_transform()
    V = T @ output_mode_covariance(deplano, 'B', rectangular_filter(-Ωb, 40*kHz), η=0.283, modes=['x1', 'y1']) @ T.T
    np.testing.assert_allclose(V, S45_MATRIX, atol=0.035)
    off_xp = np.abs(V - S45_MATRIX)
    off_xp[2, 3] = off_xp[3, 2] = 0
    assert off_xp.max() < 0.015
    ν_minus = partially_transposed_symplectic_eigenvalues(V, (0,)).min()
    assert 0.87 < ν_minus < 0.9


def test_unstable_system_raises():
    blue = toy_multitrap(dict(x1=110*kHz), dict(x1=9*kHz), dict(x1=3*kHz), 110*kHz)
    with pytest.raises(ValueError, match='unstable'):
        output_mode_covariance(QuadratureSystem.from_multitrap(blue), 'cavity', rectangular_filter(-110*kHz, 30*kHz))

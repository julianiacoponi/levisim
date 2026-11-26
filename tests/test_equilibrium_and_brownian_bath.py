"""
MultiTrap.from_nonlinear_equilibrium (the linearisation at the true equilibrium) and the opt-in
Brownian (velocity) bath
"""
import numpy as np
import pytest

from levisim.core.constants import kB, ħ, π
from levisim.core.units import Hz, degrees, kHz, mbar, nm, μm
from levisim.multiphysics.covariance import quadrature_drift_matrix, steady_state_covariance
from levisim.multiphysics.enums import QuantumLinearTheoryMethod
from levisim.multiphysics.multitrap import MultiTrap
from levisim.multiphysics.output_modes import QuadratureSystem
from levisim.multiphysics.utils.shelf import DEFAULT_cavity_kwargs, DEFAULT_gas_kwargs
from levisim.physics.bead import Bead
from levisim.physics.gas import Gas
from levisim.physics.interactions import Coulomb
from levisim.simulation.nonlinear_cavity import NonlinearCavityModel

Q = QuantumLinearTheoryMethod
ω_RANGE = np.arange(120, 180, 0.5) * kHz


def kwargs_for(Φs, φs, axes=3, **kwargs):
    return dict(num_traps=len(Φs), Cartesian_axes_per_trap=axes, ω_range=ω_RANGE,
                beads=[Bead(r=(60.1 + 0.1*n)*nm, Φ=Φ) for n, Φ in enumerate(Φs)],
                φn_dict={n + 1: φ for n, φ in enumerate(φs)},
                cavity_kwargs=DEFAULT_cavity_kwargs, gas_kwargs=DEFAULT_gas_kwargs, **kwargs)


# --------------------------------------------------------------------------------------------------
# linearisation at the nonlinear equilibrium
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('case', [([π/4], [0.0], 3), ([π/4, 1.1], [0.0, 70*degrees], 3), ([π/4, 1.1], [0.0, 70*degrees], 2)])
def test_drift_matrix_is_the_jacobian_at_the_equilibrium(case):
    kwargs = kwargs_for(*case)
    centre = MultiTrap(**kwargs)
    mtrap = MultiTrap.from_nonlinear_equilibrium(**kwargs)
    # the nonlinear model of the new MultiTrap has the same equilibrium; in the quadratures of the
    # new frequencies its Jacobian there is the new drift matrix (measured 1.3-2.8e-13)
    model = NonlinearCavityModel(mtrap, damping='symmetric')
    np.testing.assert_allclose(model.r_eq, mtrap.equilibrium_displacements, rtol=1e-8, atol=1e-15)
    idx = model.modelled_indices()
    A = quadrature_drift_matrix(mtrap.drift_matrix)
    assert np.abs(model.jacobian[np.ix_(idx, idx)] - A).max() <= 1e-11 * np.abs(A).max()
    # the frequencies are √(K/m) of the stiffness there, and differ from the trap-centre closed forms
    K = model.stiffness_at_equilibrium()
    for label, dim in mtrap.dimensions.items():
        n, a = dim.trap_index - 1, 'xyz'.index(dim.axis)
        assert dim.Ω == pytest.approx(np.sqrt(K[n, a, a] / dim.m), rel=1e-9)
    A_centre = quadrature_drift_matrix(centre.drift_matrix)
    assert np.abs(A_centre - A).max() > 1e-2 * np.abs(A).max()   # 3.5-8.8 % here
    assert mtrap.equilibrium_displacements[0, 2] > 25 * nm
    # every method sees the same linearisation
    lam = mtrap.homodyne_spectrum(Q.LINEAR_AMPLIFIER_MODEL)
    for method in (Q.NUMERICAL_HYBRIDISATION, Q.ANALYTIC_CAVITY_OUTPUT):
        np.testing.assert_allclose(mtrap.homodyne_spectrum(method), lam, rtol=1e-9)


def test_equilibrium_at_the_node_is_the_trap_centre():
    kwargs = kwargs_for([π/2, π/2], [0.0, 90*degrees], 2, cavity_effective_polarisability=False)
    centre, mtrap = MultiTrap(**kwargs), MultiTrap.from_nonlinear_equilibrium(**kwargs)
    assert np.abs(mtrap.equilibrium_displacements).max() < 1e-20
    np.testing.assert_allclose(mtrap.drift_matrix, centre.drift_matrix, rtol=0, atol=1e-8 * np.abs(centre.drift_matrix).max())


def test_equilibrium_with_pair_forces():
    beads = [Bead(r=60.1*nm, Φ=0.7, charge=10), Bead(r=60.2*nm, Φ=1.1, charge=-10)]
    kwargs = dict(num_traps=2, Cartesian_axes_per_trap=3, beads=beads, ω_range=ω_RANGE, interactions=[Coulomb()],
                  trap_positions=[(0, 0, 0), (0, 4*μm, 0)], cavity_kwargs=DEFAULT_cavity_kwargs, gas_kwargs=DEFAULT_gas_kwargs)
    mtrap = MultiTrap.from_nonlinear_equilibrium(**kwargs)
    # the charges attract along y: relative to the equilibrium without them (cavity forces alone) the
    # particles move towards each other
    alone = MultiTrap.from_nonlinear_equilibrium(**{**kwargs, 'interactions': []})
    δr = mtrap.equilibrium_displacements - alone.equilibrium_displacements
    assert δr[0, 1] > 0 > δr[1, 1]
    # inter-particle couplings from the Jacobian's off-diagonal blocks
    i, j = mtrap.labels.index('y1'), mtrap.labels.index('y2')
    assert mtrap.mechanical_coupling_matrix[i, j] != 0
    model = NonlinearCavityModel(mtrap, damping='symmetric', interactions=mtrap.interactions)
    idx = model.modelled_indices()
    A = quadrature_drift_matrix(mtrap.drift_matrix)
    assert np.abs(model.jacobian[np.ix_(idx, idx)] - A).max() <= 1e-9 * np.abs(A).max()


def test_from_nonlinear_equilibrium_rejects_toy_parameters():
    with pytest.raises(ValueError, match='physical parameters'):
        MultiTrap.from_nonlinear_equilibrium(toy_params={'Ω_model': 'custom'})


# --------------------------------------------------------------------------------------------------
# the Brownian bath
# --------------------------------------------------------------------------------------------------
def penny_pair(**kwargs):
    """Two axial modes with the shifted frequencies and the Coulomb coupling of the Penny et al. (2023)
    notebook (372.74 and 298.03 Hz, g12 = g21 = 77.82 Hz), no cavity coupling, at 293 K"""
    gas = Gas(P=1.3e-2 * mbar, T=293.0)
    Ω1, Ω2, g = 2*π*372.74*Hz, 2*π*298.03*Hz, 2*π*77.82*Hz
    return MultiTrap(
        num_traps=2, Cartesian_axes_per_trap=1, ω_range=np.linspace(100, 700, 61) * Hz,
        toy_params=dict(Ω_model='custom', toy_Ωjn_dict={'x1': Ω1, 'x2': Ω2}, m=5.3e-17, r=193.5*nm, ρ=1850,
                        P=gas.P, T=293.0, vth=gas.vth, toy_gx=0, g_direct={('x1', 'x2'): g}),
        **kwargs,
    ), (Ω1, Ω2, g)


def gibbs_covariance(Ω1, Ω2, g, T):
    """Classical Gibbs covariance of H/ħ = Σ Ωj(Xj² + Pj²)/4 - g·X1·X2 in quadratures (X = b + b†)"""
    W = ħ * np.array([[Ω1 / 2, -g], [-g, Ω2 / 2]])          # potential ½ Xᵀ W X
    V = np.zeros((4, 4))
    V[np.ix_([0, 2], [0, 2])] = kB * T * np.linalg.inv(W)
    V[1, 1], V[3, 3] = 2 * kB * T / (ħ * Ω1), 2 * kB * T / (ħ * Ω2)
    return V


def test_brownian_bath_thermalises_the_hybridised_modes():
    brownian, (Ω1, Ω2, g) = penny_pair(brownian_bath=True)
    rotating, _ = penny_pair()
    assert g / Ω2 > 0.25   # ultrastrong: g/Ω ≈ 0.2-0.26 of the bare frequencies
    V_gibbs = gibbs_covariance(Ω1, Ω2, g, 293.0)
    V_b = steady_state_covariance(brownian, cavity=False)
    V_r = steady_state_covariance(rotating, cavity=False)
    scale = np.abs(V_gibbs).max()
    # one temperature for every normal mode: the Gibbs state. The quantum correction is (ħΩ/kT)² ~ 1e-21;
    # the residual (measured 1.4e-7) is the round-off of the occupation 1/(exp(ħΩ/kT) - 1) at
    # ħΩ/kT ~ 6e-11 (Dimension.noise_phonons_exact)
    assert np.abs(V_b - V_gibbs).max() <= 1e-5 * scale
    # the rotating-wave bath does not thermalise the hybridised modes
    assert np.abs(V_r - V_gibbs).max() > 1e-2 * scale
    # the same covariance from QuadratureSystem.from_multitrap (the cavity is decoupled here)
    V_q = QuadratureSystem.from_multitrap(brownian).steady_state_covariance()[2:, 2:]
    np.testing.assert_allclose(V_q, V_b, rtol=1e-9, atol=1e-12 * scale)
    # and the LAM spectra integrate to it: ∫ S_qq dω/2π over a wide grid gives ⟨q²⟩ (= V_XX)
    wide = penny_pair(brownian_bath=True)[0]
    wide.ω_range = np.linspace(-20, 20, 400_001) * kHz * 2 * π / (2 * π)
    S = wide.all_position_power_spectra(Q.LINEAR_AMPLIFIER_MODEL)[0]
    variance = np.trapezoid(S[:, 0], wide.ω_range) / (2 * π)
    assert variance == pytest.approx(V_b[0, 0], rel=2e-3)


def test_brownian_bath_drift_and_noise():
    mtrap, _ = penny_pair(brownian_bath=True)
    A = quadrature_drift_matrix(mtrap.drift_matrix)
    Γ = mtrap.dimensions['x1'].Γ
    # X undamped, P damped at Γ
    assert A[2, 2] == pytest.approx(0.0, abs=1e-12 * Γ) and A[3, 3] == pytest.approx(-Γ, rel=1e-12)
    for method in (Q.NUMERICAL_HYBRIDISATION, Q.ANALYTIC_CAVITY_OUTPUT):
        with pytest.raises(NotImplementedError, match='brownian_bath'):
            mtrap.homodyne_spectrum(method)
    # the default (rotating-wave) bath is unchanged
    default, _ = penny_pair()
    A0 = quadrature_drift_matrix(default.drift_matrix)
    assert A0[2, 2] == pytest.approx(-Γ / 2, rel=1e-12) and A0[3, 3] == pytest.approx(-Γ / 2, rel=1e-12)

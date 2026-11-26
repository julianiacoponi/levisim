"""
The opt-in physics of levisim 0.1.1: `MultiTrap(cavity_wavenumber=True)` (the standing wave uses the
cavity wavenumber) and `MultiTrap(static_cavity_shifts=True)` (the trap frequencies include the
cotrapping shifts of the coherent-scattering potential and the dispersive shifts of the cavity
intensity, with the zero-point fluctuations and everything downstream at the shifted frequencies).

The shifts are checked against a finite-difference Hessian of the full optical potential
`levisim.physics.coherent_scattering.optical_potential`.
"""
import numpy as np
import pytest

from levisim.core.constants import ħ, kB
from levisim.core.units import kHz, degrees
from levisim.multiphysics.multitrap import MultiTrap
from levisim.multiphysics.enums.quantum_linear_theory_method import QuantumLinearTheoryMethod as QLT
from levisim.physics.bead import Bead
from levisim.physics.trap import Trap
from levisim.physics.tweezer import Tweezer
from levisim.physics.coherent_scattering import (
    hessian_to_coupling_rate,
    optical_potential,
    static_cavity_force,
    static_cavity_hessian,
)

GAS = dict(P=1e-6 * 1e2, T=300.0)  # 1e-6 mbar in Pa


class OneBead(MultiTrap):
    """One bead, three axes, at standing-wave phase Φ and polarisation angle θ"""

    def __init__(self, Φ=0.35 * np.pi, θ=49 * degrees, trap_kwargs=None, **kwargs):
        self._Φ, self._θ, self._trap_kwargs = Φ, θ, trap_kwargs or {}
        kwargs.setdefault('num_traps', 1)
        kwargs.setdefault('Cartesian_axes_per_trap', 3)
        kwargs.setdefault('gas_kwargs', GAS)
        kwargs.setdefault('ω_range', np.arange(1, 250, 1.0) * kHz)
        super().__init__(**kwargs)

    def setup_levitated_physics(self, **kwargs):
        self.tweezers = [Tweezer(θ=self._θ)]
        self.φn_dict = {1: self.tweezers[0].φ}
        self.beads = [Bead(Φ=self._Φ)]
        self.traps = [Trap(tweezer=self.tweezers[0], bead=self.beads[0], gas=self.gas, trap_index=1, **self._trap_kwargs)]


def _hessian(f, h):
    """Central finite-difference Hessian of f: R³ → R at the origin"""
    H = np.zeros((3, 3))
    e = np.eye(3) * h
    for i in range(3):
        for j in range(3):
            H[i, j] = (f(e[i] + e[j]) - f(e[i] - e[j]) - f(-e[i] + e[j]) + f(-e[i] - e[j])) / (4 * h * h)
    return H


def _gradient(f, h):
    e = np.eye(3) * h
    return np.array([(f(e[i]) - f(-e[i])) / (2 * h) for i in range(3)])


def _α(mt):
    return (mt.cavity.A0 + 1j * mt.cavity.P0) / 2


@pytest.fixture(scope='module')
def bare():
    # the numbers of levisim <= 0.1.1: since 1.0 the static shifts are on by default
    return OneBead(static_cavity_shifts=False)


@pytest.fixture(scope='module')
def full():
    return OneBead(cavity_wavenumber=True, static_cavity_shifts=True)


def test_defaults():
    # since levisim 1.0 the static shifts are on by default; the cavity wavenumber stays opt-in
    default = OneBead()
    assert default.cavity_wavenumber is False and default.static_cavity_shifts is True
    assert all(d.static_cavity_shift != 0.0 for d in default.dimensions.values())


def test_opt_out_gives_the_0_1_1_numbers(bare):
    explicit = OneBead(cavity_wavenumber=False, static_cavity_shifts=False)
    assert bare.cavity_wavenumber is False and bare.static_cavity_shifts is False
    for label, d in bare.dimensions.items():
        assert d.Ω == d.trap.Ωj[d.axis]
        assert d.static_cavity_shift == 0.0
        assert d.standing_wave_k == d.trap.drive_tweezer.k
        e = explicit.dimensions[label]
        assert (d.Ω, d.g, d.gxy, d.gxz, d.gyz) == (e.Ω, e.g, e.gxy, e.gxz, e.gyz)
    assert np.all(bare.static_cavity_hessians[1] == 0)
    np.testing.assert_array_equal(bare.drift_matrix, explicit.drift_matrix)


def test_cavity_wavenumber_scales_the_standing_wave_couplings(bare):
    mt = OneBead(cavity_wavenumber=True, static_cavity_shifts=False)
    r = mt.cavity.k / mt.traps[0].drive_tweezer.k
    # the cavity resonance is above the tweezer for red detuning, so kc > ktw by |Δ|/ωtw
    assert r - 1 == pytest.approx(-mt.cavity.Δ / mt.traps[0].tweezer.ω, rel=1e-6)
    b, c = bare.dimensions, mt.dimensions
    assert c['x1'].standing_wave_k == mt.cavity.k
    assert c['x1'].g == pytest.approx(b['x1'].g * r, rel=1e-14)
    assert c['y1'].g == pytest.approx(b['y1'].g * r, rel=1e-14)
    assert c['z1'].g == b['z1'].g  # kz is the tweezer's propagation phase
    assert c['x1'].gxy == pytest.approx(b['x1'].gxy * r**2, rel=1e-14)
    assert c['x1'].gxz == pytest.approx(b['x1'].gxz * r, rel=1e-14)
    assert c['y1'].gyz == pytest.approx(b['y1'].gyz * r, rel=1e-14)


@pytest.mark.parametrize('kc_choice', ['tweezer', 'cavity'])
@pytest.mark.parametrize('Φ', [0.0, 0.2 * np.pi, 0.35 * np.pi, 0.45 * np.pi])
def test_static_hessian_matches_finite_differences(Φ, kc_choice):
    mt = OneBead(Φ=Φ, static_cavity_shifts=False)  # bare frequencies (the default has the shifts on)
    trap, α = mt.traps[0], _α(mt)
    kc = mt.cavity.k if kc_choice == 'cavity' else None
    h = 2e-10

    def U(terms):
        return lambda r: float(optical_potential(r, trap, mt.cavity.E0, α, terms=terms, kc=kc))

    H_CS = _hessian(U(('coherent_scattering',)), h)
    H_cav = _hessian(U(('cavity',)), h)
    K = static_cavity_hessian(trap, mt.cavity.E0, α, kc=kc)
    scale = max(np.abs(H_CS).max(), np.abs(H_cav).max())
    # diagonal of U_CS (cotrapping) plus all of U_cav (dispersive)
    expected = np.diag(np.diag(H_CS)) + H_cav
    np.testing.assert_allclose(K, expected, rtol=0, atol=2e-5 * scale)
    np.testing.assert_allclose(
        static_cavity_hessian(trap, mt.cavity.E0, α, kc=kc, terms=('cavity',)), H_cav, rtol=0, atol=2e-5 * scale,
    )
    # the off-diagonal of U_CS is levisim's direct mechanical couplings (bare frequencies here)
    if kc_choice == 'tweezer':
        d = mt.dimensions['x1']
        zpf = {a: d.zero_point_fluctuation(a) for a in 'xyz'}
        for (i, a), (j, b) in [((0, 'x'), (1, 'y')), ((0, 'x'), (2, 'z')), ((1, 'y'), (2, 'z'))]:
            g = getattr(d, f'g{a}{b}')
            g_fd = hessian_to_coupling_rate(H_CS[i, j], zpf[a], zpf[b])
            assert g_fd == pytest.approx(g, rel=1e-4, abs=1e-6 * abs(d.g))
    # the static force is minus the gradient
    F = static_cavity_force(trap, mt.cavity.E0, α, kc=kc)
    F_fd = -_gradient(U(('coherent_scattering', 'cavity')), h)
    np.testing.assert_allclose(F, F_fd, rtol=0, atol=1e-6 * np.abs(F_fd).max())


def test_shifted_frequencies_are_the_curvature_of_the_full_potential(full):
    trap, α = full.traps[0], _α(full)
    U = lambda r: float(optical_potential(r, trap, full.cavity.E0, α, kc=full.cavity.k))
    H = _hessian(U, 1e-10)
    for ii, axis in enumerate('xyz'):
        d = full.dimensions[f'{axis}1']
        assert trap.bead.m * d.Ω**2 == pytest.approx(H[ii, ii], rel=2e-4)
    # x-y: the CS cross term (levisim's gxy) plus the dispersive term, both in gxy
    d = full.dimensions['x1']
    g_fd = hessian_to_coupling_rate(H[0, 1], d.zero_point_fluctuation('x'), d.zero_point_fluctuation('y'))
    assert g_fd == pytest.approx(d.gxy, rel=1e-4)


def test_shifts_propagate_downstream(bare, full):
    for label, d in full.dimensions.items():
        Ω_tw = d.trap.Ωj[d.axis]
        assert d.Ω == full.static_shifted_Ωj[1][d.axis]
        assert d.static_cavity_shift == pytest.approx(d.Ω - Ω_tw, rel=1e-14)
        assert d.zero_point_fluctuation(d.axis) == pytest.approx(np.sqrt(ħ / (2 * d.m * d.Ω)), rel=1e-14)
        assert d.noise_phonons_exact == pytest.approx(1 / np.expm1(ħ * d.Ω / (kB * d.T)), rel=1e-7)  # exp(x) - 1 at x ~ 1e-8
        # recoil heating in phonons/s goes as x_zpf² ∝ 1/Ω
        assert full.recoil_heating_rate_dict[label] == pytest.approx(
            bare.recoil_heating_rate_dict[label] * Ω_tw / d.Ω, rel=1e-12,
        )
    # red detuning, between node and antinode: cotrapping stiffens every axis
    assert all(d.static_cavity_shift > 0 for d in full.dimensions.values())
    # the drift matrix uses the shifted frequencies: its mechanical eigenvalues move with them
    Ωs = {label: d.Ω for label, d in full.dimensions.items()}
    λ = np.linalg.eigvals(full.drift_matrix)
    assert np.any(np.isclose(np.abs(λ.imag), Ωs['x1'], rtol=2e-2))
    assert not np.any(np.isclose(np.abs(λ.imag), bare.dimensions['z1'].Ω, rtol=1e-3))


def test_methods_agree_with_shifts(full):
    _, S_lam = full.all_position_power_spectra(QLT.TNT_LINEAR_AMPLIFIER_MODEL, zpf_scaled=False)
    _, S_hyb = full.all_position_power_spectra(QLT.NUMERICAL_HYBRIDISATION, zpf_scaled=False)
    for label in full.labels:
        np.testing.assert_allclose(S_hyb[label], S_lam[label], rtol=1e-9)


def test_no_shift_at_the_node():
    mt = OneBead(Φ=np.pi / 2, cavity_wavenumber=True, static_cavity_shifts=True)
    np.testing.assert_allclose(mt.static_cavity_hessians[1], 0, atol=1e-30)
    for d in mt.dimensions.values():
        assert d.Ω == d.trap.Ωj[d.axis]


def test_measured_frequencies_are_not_shifted():
    Ω = {'x': 2 * np.pi * 150e3, 'y': 2 * np.pi * 130e3, 'z': 2 * np.pi * 40e3}
    mt = OneBead(static_cavity_shifts=True, trap_kwargs=dict(override_Ω_dict=Ω))
    for d in mt.dimensions.values():
        assert d.Ω == Ω[d.axis]
    assert mt.static_cavity_hessians[1][0, 1] != 0  # the dispersive x-y coupling is still added


def test_toy_parameters_ignore_the_keywords():
    from levisim.multiphysics.utils.shelf import PHASE_EQUAL_toy_params
    mt = MultiTrap(num_traps=1, toy_params=PHASE_EQUAL_toy_params, static_cavity_shifts=True, cavity_wavenumber=True)
    assert mt.static_cavity_shifts is False and mt.cavity_wavenumber is False

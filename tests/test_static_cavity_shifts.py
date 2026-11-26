"""
Static coherent-scattering frequency shifts and the cavity's dispersive potential

`MultiTrap(static_cavity_shifts=True)` (the default with physical parameters) adds to each bead's
trap the static stiffness of the static cavity field that the linearised couplings leave out
(`MultiTrap.setup_static_cavity_shifts`). These tests check

- the one potential function, `levisim.physics.coherent_scattering.optical_potential`: its
  numerical second derivatives at the trap centre reproduce the tweezer stiffness, the existing
  couplings g, gxy, gxz, gyz, the new diagonal terms and the cavity dispersive Hessian
- that the shifts vanish exactly at the node, and that opting out restores the bare frequencies
- that the shifted Ω is used self-consistently (zero-point fluctuation, couplings, recoil)
"""
import numpy as np
import pytest

from levisim.core.constants import ħ, π
from levisim.core.logger import logger
from levisim.core.units import degrees, kHz, nm, μm
from levisim.multiphysics.enums import QuantumLinearTheoryMethod
from levisim.multiphysics.multitrap import MultiTrap
from levisim.multiphysics.utils.shelf import (
    DEFAULT_cavity_kwargs,
    DEFAULT_gas_kwargs,
    OFF_NODE_toy_params_pairs,
)
from levisim.physics.bead import Bead
from levisim.physics.coherent_scattering import (
    cavity_axis,
    optical_potential,
    static_cavity_force,
    static_cavity_hessian,
)
from levisim.physics.tweezer import Tweezer

LAM = QuantumLinearTheoryMethod.LINEAR_AMPLIFIER_MODEL
ω_RANGE = np.arange(100, 200, 1.0) * kHz


def build(Φs, axes=3, θ=45*degrees, **kwargs):
    beads = [Bead(r=(60.1 + 0.1*n)*nm, Φ=Φ) for n, Φ in enumerate(Φs)]
    kwargs.setdefault('tweezers', [Tweezer(θ=θ, wx0=(0.93 + 0.02*n)*μm, wy0=1.07*μm) for n in range(len(Φs))])
    kwargs.setdefault('cavity_kwargs', DEFAULT_cavity_kwargs)
    kwargs.setdefault('gas_kwargs', DEFAULT_gas_kwargs)
    return MultiTrap(num_traps=len(Φs), Cartesian_axes_per_trap=axes, beads=beads, ω_range=ω_RANGE, **kwargs)


def static_α(mtrap):
    return (mtrap.cavity.A0 + 1j*mtrap.cavity.P0) / 2


# --------------------------------------------------------------------------------------------------
# finite differences (fourth-order central stencils)
# --------------------------------------------------------------------------------------------------
def fd_gradient(f, h):
    e = np.eye(3)
    return np.array([
        (-f(2*h[i]*e[i]) + 8*f(h[i]*e[i]) - 8*f(-h[i]*e[i]) + f(-2*h[i]*e[i])) / (12*h[i])
        for i in range(3)
    ])


def fd_hessian(f, h):
    e = np.eye(3)
    H = np.zeros((3, 3))
    for i in range(3):
        H[i, i] = (-f(2*h[i]*e[i]) + 16*f(h[i]*e[i]) - 30*f(0*e[i]) + 16*f(-h[i]*e[i]) - f(-2*h[i]*e[i])) / (12*h[i]**2)
        for j in range(i + 1, 3):
            def g(a, b, i=i, j=j):
                return f(a*h[i]*e[i] + b*h[j]*e[j])
            H[i, j] = H[j, i] = (
                8*(g(1, -2) + g(2, -1) + g(-2, 1) + g(-1, 2))
                - 8*(g(-1, -2) + g(-2, -1) + g(1, 2) + g(2, 1))
                - (g(2, -2) + g(-2, 2) - g(-2, -2) - g(2, 2))
                + 64*(g(-1, -1) + g(1, 1) - g(1, -1) - g(-1, 1))
            ) / (144*h[i]*h[j])
    return H


def potential(trap, mtrap, α, *terms):
    return lambda r: optical_potential(r, trap, mtrap.cavity.E0, α, terms=terms)


def steps(trap):
    k, zR = trap.drive_tweezer.k, trap.drive_tweezer.zR
    return np.array([2e-3/k, 2e-3/k, 2e-3/(k - 1/zR)])


def assert_close(fd, exact, scale, rtol=1e-6):
    assert abs(fd - exact) <= rtol * abs(scale), (fd, exact, scale)


CASES = [(45, 45), (30, 45), (70, 80), (135, 45), (10, 90), (100, 30), (60, 60)]


@pytest.mark.parametrize('Φ_deg, θ_deg', CASES)
@pytest.mark.parametrize('shifts', [True, False])
def test_finite_differences_of_the_potential_reproduce_every_coupling(Φ_deg, θ_deg, shifts):
    """∂²U/∂rj∂rk and ∂²U/∂rj∂(A, P) at the trap centre give g, gxy, gxz, gyz (to ≲1e-6)"""
    mtrap = build([Φ_deg*degrees, 0.3], θ=θ_deg*degrees, static_cavity_shifts=shifts)
    trap, α = mtrap.traps[0], static_α(mtrap)
    dims = {axis: mtrap.dimensions[f'{axis}1'] for axis in 'xyz'}
    zpf = {axis: dims['x'].zero_point_fluctuation(axis) for axis in 'xyz'}
    h = steps(trap)

    H_CS = fd_hessian(potential(trap, mtrap, α, 'coherent_scattering'), h)
    H_cav = fd_hessian(potential(trap, mtrap, α, 'cavity'), h)
    K = static_cavity_hessian(trap, mtrap.cavity.E0, α)
    to_g = {ax: -zpf[ax[0]] * zpf[ax[1]] / ħ for ax in ('xy', 'xz', 'yz')}
    g_scale = max(abs(dims['x'].mech_mech_coupling(ax)) for ax in ('xy', 'xz', 'yz'))
    for (i, j, ax) in [(0, 1, 'xy'), (0, 2, 'xz'), (1, 2, 'yz')]:
        expected = dims['x'].mech_mech_coupling(ax)
        fd = H_CS[i, j] * to_g[ax] + (H_cav[i, j] * to_g[ax] if shifts else 0.0)
        assert_close(fd, expected, g_scale)
        assert getattr(dims['x'], f'g{ax}') == expected

    # light-mechanics couplings: U_CS is linear in α, so α = 1/2 and i/2 pick out A and P
    grad_A = fd_gradient(potential(trap, mtrap, 0.5, 'coherent_scattering'), h)
    grad_P = fd_gradient(potential(trap, mtrap, 0.5j, 'coherent_scattering'), h)
    g_light_scale = max(abs(dims[axis].g) for axis in 'xyz')
    assert_close(-grad_A[0] * zpf['x'] / ħ, dims['x'].g, g_light_scale)
    assert_close(-grad_A[1] * zpf['y'] / ħ, dims['y'].g, g_light_scale)
    assert_close(-grad_P[2] * zpf['z'] / ħ, dims['z'].g, g_light_scale)
    assert_close(grad_P[0], 0.0, max(abs(grad_A)))  # x, y do not couple to P, z not to A
    assert_close(grad_A[2], 0.0, max(abs(grad_A)))

    # the new static terms: the diagonal of U_CS and the whole of U_cav
    K_scale = np.max(np.abs(np.diag(H_CS)))
    for i in range(3):
        assert_close(H_CS[i, i] + H_cav[i, i], K[i, i], K_scale)
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        assert_close(H_cav[i, j], K[i, j], K_scale)
        assert K[i, j] == K[j, i]

    # the tweezer stiffness is Trap.kj (a check of the |E|² normalisation)
    H_tw = fd_hessian(potential(trap, mtrap, α, 'tweezer'), 1e-3 * np.array([trap.tweezer.wx0, trap.tweezer.wy0, trap.tweezer.zR]))
    for i, axis in enumerate('xyz'):
        assert H_tw[i, i] == pytest.approx(trap.kj[axis], rel=1e-6)

    # the static force
    F = static_cavity_force(trap, mtrap.cavity.E0, α)
    F_fd = -fd_gradient(potential(trap, mtrap, α, 'coherent_scattering', 'cavity'), h)
    np.testing.assert_allclose(F_fd, F, rtol=0, atol=1e-6 * np.max(np.abs(F)))


def test_the_cavity_hessian_is_along_the_cavity_axis_and_follows_cos_2Φ():
    for Φ in (0.3, 0.9, 1.3):
        mtrap = build([Φ, 0.2])
        trap, α = mtrap.traps[0], static_α(mtrap)
        K_cav = static_cavity_hessian(trap, mtrap.cavity.E0, α) - np.diag(np.diag(static_cavity_hessian(trap, mtrap.cavity.E0, α)))
        d = cavity_axis(trap.drive_tweezer.θ)
        expected = 2 * trap.bead.α * mtrap.cavity.E0**2 * abs(α)**2 * trap.drive_tweezer.k**2 * np.cos(2*Φ) * np.outer(d, d)
        np.testing.assert_allclose(K_cav, expected - np.diag(np.diag(expected)), rtol=1e-12, atol=0)


def test_the_potential_with_a_wide_cavity_waist_tends_to_the_plane_standing_wave():
    mtrap = build([0.6, 0.3])
    trap, α = mtrap.traps[0], static_α(mtrap)
    r = np.array([[30*nm, -20*nm, 50*nm]])
    plane = optical_potential(r, trap, mtrap.cavity.E0, α, terms=('coherent_scattering', 'cavity'))
    wide = optical_potential(r, trap, mtrap.cavity.E0, α, terms=('coherent_scattering', 'cavity'), cavity_waist=1.0)
    narrow = optical_potential(r, trap, mtrap.cavity.E0, α, terms=('coherent_scattering', 'cavity'), cavity_waist=61*μm)
    assert wide == pytest.approx(plane, rel=1e-12)
    assert narrow != plane and narrow == pytest.approx(plane, rel=1e-5)
    with pytest.raises(ValueError, match='unknown'):
        optical_potential(r, trap, mtrap.cavity.E0, α, terms=('tweezers',))


# --------------------------------------------------------------------------------------------------
# MultiTrap
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('system', [(1, 2), (2, 3), (3, 2)])
def test_shifts_are_exactly_zero_at_the_node(system):
    num_traps, axes = system
    on = MultiTrap(num_traps=num_traps, Cartesian_axes_per_trap=axes, ω_range=ω_RANGE)
    off = MultiTrap(num_traps=num_traps, Cartesian_axes_per_trap=axes, ω_range=ω_RANGE, static_cavity_shifts=False)
    assert on.static_cavity_shifts and not off.static_cavity_shifts
    for label in on.labels:
        assert on.dimensions[label].Ω == off.dimensions[label].Ω == on.dimensions[label].trap.Ωj[label[0]]
        assert on.dimensions[label].static_cavity_shift == 0.0
        assert on.dimensions[label].zpf == off.dimensions[label].zpf
        assert on.dimensions[label].g == off.dimensions[label].g
    for x, y in zip(
        (on.homodyne_spectrum(LAM), on.all_position_power_spectra(LAM)[0]),
        (off.homodyne_spectrum(LAM), off.all_position_power_spectra(LAM)[0]),
    ):
        assert np.array_equal(x, y)


def test_opting_out_keeps_the_tweezer_frequencies():
    mtrap = build([π/4, 0.4], static_cavity_shifts=False)
    for dim in mtrap.dimensions.values():
        assert dim.Ω == dim.trap.Ωj[dim.axis]
        assert dim.static_cavity_shift == 0.0
    assert all(not np.any(K) for K in mtrap.static_cavity_hessians.values())


def test_shifted_frequencies_are_used_self_consistently():
    on, off = build([π/4, 0.4]), build([π/4, 0.4], static_cavity_shifts=False)
    for label in on.labels:
        d_on, d_off = on.dimensions[label], off.dimensions[label]
        n, axis = d_on.trap_index, d_on.axis
        K = on.static_cavity_hessians[n]['xyz'.index(axis), 'xyz'.index(axis)]
        assert d_on.Ω**2 == pytest.approx(d_off.Ω**2 + K / d_on.m, rel=1e-12)
        assert d_on.zpf == pytest.approx(np.sqrt(ħ / (2 * d_on.m * d_on.Ω)), rel=1e-15)
        assert d_on.zero_point_fluctuation(axis) == d_on.zpf
        # every coupling scales with the zero-point fluctuation(s) of its dimension(s)
        assert d_on.g == pytest.approx(d_off.g * np.sqrt(d_off.Ω / d_on.Ω), rel=1e-12)
        assert d_on.noise_phonons_exact < d_off.noise_phonons_exact  # stiffened: fewer thermal phonons
        assert on.recoil_heating_rate_dict[label] == pytest.approx(
            off.recoil_heating_rate_dict[label] * d_off.Ω / d_on.Ω, rel=1e-12)
    dx_on, dx_off = on.dimensions['x1'], off.dimensions['x1']
    scale = np.sqrt(dx_off.Ω / dx_on.Ω * off.dimensions['z1'].Ω / on.dimensions['z1'].Ω)
    assert dx_on.gxz == pytest.approx(dx_off.gxz * scale, rel=1e-12)


def test_the_dispersive_term_adds_an_x_y_coupling():
    Φ = 0.35 * π  # cos 2Φ ≠ 0
    on, off = build([Φ]), build([Φ], static_cavity_shifts=False)
    d_on, d_off = on.dimensions['x1'], off.dimensions['x1']
    scale = np.sqrt(d_off.Ω / d_on.Ω * off.dimensions['y1'].Ω / on.dimensions['y1'].Ω)
    K_xy = on.static_cavity_hessians[1][0, 1]
    assert K_xy != 0
    extra = -K_xy * d_on.zpf * on.dimensions['y1'].zpf / ħ
    assert d_on.gxy == pytest.approx(d_off.gxy * scale + extra, rel=1e-12)
    assert abs(extra) < 1e-2 * abs(d_on.gxy)  # ~|E_cav|/E_tw smaller than the coherent-scattering gxy


def test_magnitude_for_one_bead_at_a_quarter_wave_in_the_default_cavity():
    """|α|² ≈ 1.85 × 10⁶ photons; δΩ/2π ≈ +1.04, +1.17, +6.71 kHz on 156.3, 135.8, 35.1 kHz"""
    mtrap = MultiTrap(num_traps=1, Cartesian_axes_per_trap=3, beads=[Bead(r=60.1*nm, Φ=π/4)],
                      cavity_kwargs=DEFAULT_cavity_kwargs, gas_kwargs=DEFAULT_gas_kwargs, ω_range=ω_RANGE)
    assert abs(static_α(mtrap))**2 == pytest.approx(1.847e6, rel=1e-3)
    shifts = {axis: mtrap.dimensions[f'{axis}1'].static_cavity_shift / kHz for axis in 'xyz'}
    assert shifts == pytest.approx({'x': 1.045, 'y': 1.167, 'z': 6.707}, rel=2e-3)
    # at Φ = π/4 the dispersive term vanishes (cos 2Φ = 0): only the coherent-scattering diagonal
    K = mtrap.static_cavity_hessians[1]
    assert abs(K[0, 1]) < 1e-12 * abs(K[0, 0])


def test_toy_parameters_are_unaffected_and_warn_if_asked():
    captured = []
    handler_id = logger.add(lambda message: captured.append(message.record['message']), level='WARNING', format='{message}')
    try:
        default = MultiTrap(num_traps=2, Cartesian_axes_per_trap=2, toy_params=OFF_NODE_toy_params_pairs, ω_range=ω_RANGE)
        asked = MultiTrap(num_traps=2, Cartesian_axes_per_trap=2, toy_params=OFF_NODE_toy_params_pairs, ω_range=ω_RANGE,
                          static_cavity_shifts=True)
    finally:
        logger.remove(handler_id)
    assert not default.static_cavity_shifts and not asked.static_cavity_shifts
    assert np.array_equal(default.drift_matrix, asked.drift_matrix)
    assert any('static_cavity_shifts only apply to physical parameters' in message for message in captured)


def test_measured_frequencies_are_kept():
    Ω = {'x': 150*kHz, 'y': 130*kHz, 'z': 40*kHz}
    mtrap = build([π/4, 0.4], trap_kwargs=[{'override_Ω_dict': Ω}, {}])
    for axis in 'xyz':
        assert mtrap.dimensions[f'{axis}1'].Ω == Ω[axis]
        assert mtrap.dimensions[f'{axis}2'].static_cavity_shift != 0


def test_a_statically_unstable_dimension_raises():
    # blue detuning softens: A0·cos Φ < 0 near the antinode, and z is the softest axis
    with pytest.raises(ValueError, match='statically unstable'):
        build([0.05], cavity_kwargs=dict(DEFAULT_cavity_kwargs, Δ=+200*kHz))
    # the same system without z, or without the shifts, builds
    build([0.05], axes=2, cavity_kwargs=dict(DEFAULT_cavity_kwargs, Δ=+200*kHz))
    build([0.05], cavity_kwargs=dict(DEFAULT_cavity_kwargs, Δ=+200*kHz), static_cavity_shifts=False)

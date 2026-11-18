"""
Multi-tone coherent-scattering traps built from polarisations and cavity geometry
(levisim.multiphysics.two_tone_traps), and the QuadraticPotential terms of MultiToneDrive

Anchors:
  1. One red tone reproduces the Deplano et al. (2025) frequencies and couplings.
  2. Co-polarised tones reproduce levisim.multiphysics.two_tone exactly, beat included.
  3. Orthogonal tones give the closed-form static trap, and their beat is a pure x-y term.
  4. A static QuadraticPotential is levisim's own mech-mech coupling, sign included.
  5. Geometry: the node has no carrier field; a static coupling only rotates the axes; off the
     node the carrier adds a (u·r)² stiffness, i.e. frequency shifts as large as g_xy.
"""
import numpy as np
import pytest

from levisim.core.units import kHz
from levisim.core.constants import π
from levisim.analysis.gaussian_states import log_negativity
from levisim.multiphysics.covariance import dimension_covariance
from levisim.multiphysics.multitone import MultiToneDrive, QuadraticPotential, Tone
from levisim.multiphysics.multitrap import MultiTrap
from levisim.multiphysics.two_tone import DEPLANO2025, deplano2025_multitrap, two_tone_steady_state
from levisim.multiphysics.two_tone_traps import (
    CavityGeometry,
    TweezerTone,
    deplano2025_tweezer,
    multitone_drive,
    tweezer_trap,
    two_tone_tones,
)

P = DEPLANO2025


def _heating(scale, labels=('x1', 'y1')):
    base = {'x': P['Γx_decoherence'], 'y': P['Γy_decoherence']}
    return {label: scale * base[label[0]] for label in labels}


def _trap(ρ, ψ_blue, θ=None, f_pol=1.0, **kwargs):
    optics, geometry, ω_d, m = deplano2025_tweezer(f_pol=f_pol)
    if θ is not None:
        geometry = CavityGeometry(θ=θ)
    projections = tuple(np.array([np.cos(ψ), np.sin(ψ), 0.0]) @ geometry.polarisation for ψ in (π/2, ψ_blue))
    ν = (P['Ωx'] + P['Ωy']) / 2
    for _ in range(2):
        tones = two_tone_tones(ν, ρ, π/2, ψ_blue, projections=projections, **kwargs)
        trap = tweezer_trap(optics, tones, geometry, ω_d, m, κ=P['κ'])
        ν = np.mean(list(trap.frequencies.values()))
    return trap


def test_one_red_tone_reproduces_deplano():
    optics, geometry, ω_d, m = deplano2025_tweezer()
    trap = tweezer_trap(optics, [TweezerTone(Δ=-115*kHz, amplitude=1.0)], geometry, ω_d, m)
    assert trap.frequencies['x1'] == pytest.approx(P['Ωx'])
    assert trap.frequencies['y1'] == pytest.approx(P['Ωy'])
    assert abs(trap.tones[0].couplings['x1']) == pytest.approx(P['gx'], rel=1e-12)
    assert abs(trap.tones[0].couplings['y1']) == pytest.approx(P['gy'], rel=1e-12)
    assert trap.axes_angle == 0.0 and trap.potentials == []


@pytest.mark.parametrize('Γ_scale, ratio', [(1/3, 0.525), (0.1, 0.675)])
def test_copolarised_tones_reproduce_the_two_tone_model(Γ_scale, ratio):
    trap = _trap(ratio, π/2, θ=deplano2025_tweezer()[1].θ)
    assert [abs(trap.tones[0].couplings[l]) for l in ('x1', 'y1')] == pytest.approx([P['gx'], P['gy']])
    # with the beat: the same Floquet multiplier as two_tone's beat_visibility = 1
    mtrap, drive = multitone_drive([trap], P['κ'], _heating(Γ_scale))
    reference = two_tone_steady_state(deplano2025_multitrap(Γ_scale=Γ_scale), ratio, beat_visibility=1.0, num_samples=4)
    assert drive.periodic_steady_state(4).largest_multiplier == pytest.approx(reference.floquet_multiplier, rel=1e-6)
    # without it: the same entanglement as v = 0
    trap.potentials = []
    mtrap, drive = multitone_drive([trap], P['κ'], _heating(Γ_scale))
    ss = drive.periodic_steady_state(8)
    E = np.mean([log_negativity(dimension_covariance(mtrap, V, ['x1', 'y1'])) for V in ss.covariances])
    assert E == pytest.approx(two_tone_steady_state(deplano2025_multitrap(Γ_scale=Γ_scale), ratio, num_samples=8).E_N,
                              rel=1e-6)


@pytest.mark.parametrize('f_pol', [1.0, 0.7])
def test_orthogonal_tones_give_the_closed_form_trap(f_pol):
    optics = deplano2025_tweezer(f_pol=f_pol)[0]
    trap = _trap(0.4, 0.0, θ=np.radians(45), f_pol=f_pol)
    c_red, c_blue = trap.tones[0].amplitude, trap.tones[1].amplitude
    assert c_red**2 + c_blue**2 == pytest.approx(1.0)
    Δp, Δs = f_pol * optics.anisotropy, (1 - f_pol) * optics.anisotropy
    Kx = optics.mean_square + (Δs + Δp * (c_red**2 - c_blue**2)) / 2
    Ky = optics.mean_square - (Δs + Δp * (c_red**2 - c_blue**2)) / 2
    assert trap.frequencies['x1'] == pytest.approx(np.sqrt(Kx), rel=1e-12)
    assert trap.frequencies['y1'] == pytest.approx(np.sqrt(Ky), rel=1e-12)
    assert trap.axes_angle == pytest.approx(0.0, abs=1e-12)
    # the beat has no diagonal part: a two-mode squeezer χ(b_x b_y + h.c.) with χ = c₋c₊Δp/(4√(Ω_xΩ_y))
    (beat,) = trap.potentials
    assert set(beat.coefficients) == {('x1', 'y1')}
    assert beat.frequency == pytest.approx(trap.tones[0].Δ - trap.tones[1].Δ)
    W = beat.coefficients[('x1', 'y1')]
    χ = c_red * c_blue * Δp / (4 * np.sqrt(trap.frequencies['x1'] * trap.frequencies['y1']))
    assert abs(W) / 2 == pytest.approx(χ, rel=1e-12)


def test_blue_tone_projection_is_minus_cot_theta():
    θ = np.radians(55.0)
    trap = _trap(0.4, 0.0, θ=θ)
    red, blue = trap.tones
    for label in ('x1', 'y1'):
        assert blue.couplings[label] / red.couplings[label] == pytest.approx(-1 / np.tan(θ), rel=1e-12)
    # and the requested coupling ratio is met once the projections are accounted for
    G_blue, G_red = blue.amplitude * blue.couplings['x1'], red.amplitude * red.couplings['x1']
    assert abs(G_blue / G_red) == pytest.approx(0.4, rel=1e-12)


def test_orthogonal_entanglement_does_not_depend_on_the_relative_tone_phase():
    values = []
    for phase in (0.0, 1.3, 2.9):
        trap = _trap(0.35, 0.0, θ=np.radians(45), offset=-1.0*kHz, relative_phase=phase)
        mtrap, drive = multitone_drive([trap], P['κ'], _heating(1/3))
        ss = drive.periodic_steady_state(16)
        values.append(np.mean([log_negativity(dimension_covariance(mtrap, V, ['x1', 'y1'])) for V in ss.covariances]))
    assert values[0] > 0.1
    assert np.ptp(values) < 2e-3 * values[0]


def test_static_quadratic_potential_is_levisims_mech_mech_coupling():
    g = -2.0 * kHz                    # levisim's convention: g_xy ≤ 0, with H/ħ ⊃ -g_xy·X_xX_y
    toy = {
        'Ω_model': 'custom', 'toy_Ωjn_dict': {'x1': P['Ωx'], 'y1': P['Ωy']}, 'use_Γ_formula': False,
        'Γjn_dict': {'x1': 1e-4, 'y1': 1e-4}, 'gjn_dict': {'x1': -P['gx'], 'y1': -P['gy']}, 'T': 300.0,
    }
    common = dict(num_traps=1, Cartesian_axes_per_trap=2, cavity_kwargs={'κ': P['κ'], 'Δ': P['Δ']},
                  ω_range=np.arange(-10, 10, 1) * kHz)
    with_g = MultiTrap(toy_params={**toy, 'toy_gxy': g}, **common)
    without = MultiTrap(toy_params=toy, **common)
    tones = [Tone(Δ=-110*kHz, amplitude=0.9), Tone(Δ=+110*kHz, amplitude=0.4)]
    a = MultiToneDrive(with_g, tones)
    b = MultiToneDrive(without, tones, potentials=[QuadraticPotential(0.0, {('x1', 'y1'): -g})])
    for t in (0.0, 1.7e-6, 3.1e-6):
        assert np.allclose(a.drift_matrix(t), b.drift_matrix(t), rtol=1e-12, atol=1e-9 * np.abs(a.drift_matrix(t)).max())


def test_the_node_has_no_carrier_field():
    optics, geometry, ω_d, m = deplano2025_tweezer(include_z=True)
    tones = two_tone_tones(115*kHz, 0.5, π/2, 0.0)
    trap = tweezer_trap(optics, tones, geometry, ω_d, m, κ=P['κ'])
    assert np.all(trap.coherent_amplitudes == 0)
    assert all(not q.label.startswith('cavity field') for q in trap.potentials)
    assert abs(trap.tones[0].couplings['z1']) == pytest.approx(0.0, abs=1e-9)


def test_a_static_coupling_only_rotates_the_axes():
    optics, geometry, ω_d, m = deplano2025_tweezer()
    K_xy = -0.08 * optics.anisotropy
    trap = tweezer_trap(optics, [TweezerTone(Δ=-115*kHz, amplitude=1.0)], geometry, ω_d, m,
                        extra_stiffness=np.array([[0, K_xy], [K_xy, 0]]))
    K = np.array([[optics.Ω_perp**2, K_xy], [K_xy, optics.Ω_par**2]])
    assert sorted(trap.frequencies.values()) == pytest.approx(sorted(np.sqrt(np.linalg.eigvalsh(K))))
    assert trap.axes_angle == pytest.approx(0.5 * np.arctan2(2 * K_xy, K[0, 0] - K[1, 1]), rel=1e-10)
    assert trap.potentials == []
    # the couplings follow the rotated axes: their quadrature sum is unchanged
    G2 = sum(abs(trap.tones[0].couplings[l])**2 * trap.frequencies[l] for l in ('x1', 'y1'))
    assert G2 == pytest.approx(P['gx']**2 * P['Ωx'] + P['gy']**2 * P['Ωy'], rel=1e-10)


def test_off_the_node_the_carrier_adds_a_rank_one_stiffness():
    optics, geometry, ω_d, m = deplano2025_tweezer()
    off = CavityGeometry(θ=geometry.θ, Φ=π/2 - 0.4)
    tones = [TweezerTone(Δ=-115*kHz, amplitude=1.0)]
    node = tweezer_trap(optics, tones, geometry, ω_d, m, κ=P['κ'])
    trap = tweezer_trap(optics, tones, off, ω_d, m, κ=P['κ'])
    K = lambda t: t.axes[:2, :2] @ np.diag([t.frequencies['x1']**2, t.frequencies['y1']**2]) @ t.axes[:2, :2].T
    ΔK = K(trap) - K(node)
    u = off.axis[:2]
    # ΔK ∝ u uᵀ: the diagonal shifts come with the x-y term in the ratio sin²θ : sinθcosθ : cos²θ
    assert ΔK == pytest.approx(ΔK[0, 1] / (u[0] * u[1]) * np.outer(u, u), rel=1e-9)
    assert abs(ΔK[0, 0]) > abs(ΔK[0, 1]) > 0
    assert np.sum(np.abs(trap.coherent_amplitudes)**2) > 1e5

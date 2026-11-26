"""
The escape barrier of a static trap: `static_energy` and the saddle-point search `find_escape_saddle`
(closed-form saddles, independent finite differences), its use in `magnetic_trap_report` against the
ray estimate, and the Slezak et al. (2018) magneto-gravitational trap
"""
import numpy as np
import pytest

from levisim.core.constants import kB
from levisim.core.units import μm
from levisim.physics import Sphere, Trap
from levisim.physics.fields import (
    ForceElement,
    Gravity,
    HarmonicPotential,
    LocalFieldExpansion,
    MagneticTrap,
    MultipoleField,
    find_escape_saddle,
    static_energy,
)
from levisim.physics.fields.base import as_position
from levisim.physics.materials import BISMUTH, FUSED_SILICA

PARTICLE = Sphere(r=100e-9)


class CubicWell(ForceElement):
    """U = ½ Σ k_j r_j² - (λ/3) u³ with u = r·n̂: a well with one index-1 saddle along n̂ (test element)"""

    is_conservative = True

    def __init__(self, k, λ, n):
        self.k = np.asarray(k, dtype=float)
        self.λ = float(λ)
        self.n = np.asarray(n, dtype=float) / np.linalg.norm(n)

    def potential(self, r, particle):
        r = as_position(r)
        return 0.5 * np.sum(self.k * r**2) - self.λ * (r @ self.n)**3 / 3

    def force(self, r, t, particle):
        r = as_position(r)
        return -self.k * r + self.λ * (r @ self.n)**2 * self.n

    def stiffness(self, r0=None, particle=None):
        r = np.zeros(3) if r0 is None else as_position(r0)
        return np.diag(self.k) - 2 * self.λ * (r @ self.n) * np.outer(self.n, self.n)

    def closed_form(self):
        # stationary point off the origin: k_j r_j = λ u² n_j, so u = 1/(λ s) with s = Σ n_j²/k_j, and
        # the barrier U* = ½ λ² u⁴ s - λ u³/3 = 1/(6 λ² s³)
        s = np.sum(self.n**2 / self.k)
        u = 1 / (self.λ * s)
        return self.λ * u**2 * self.n / self.k, 1 / (6 * self.λ**2 * s**3)


class ForceOnly(ForceElement):
    """A harmonic force without a potential method (to test the fallback)"""

    def force(self, r, t, particle):
        return -1e-6 * as_position(r)


def _slezak_trap():
    # the fit of the Slezak et al. (2018) notebook: three multipoles of Hsu et al. (2016), y0 = 75 μm
    silica = FUSED_SILICA.replace(ρ=2200.0)
    source = MultipoleField({(2, -2): -0.793625, (3, 1): 0.007238, (4, -4): 0.223192}, r0=75 * μm)
    return Trap(fields=[MagneticTrap(source, shape_correction=False), Gravity(g=9.80665, direction=(0, -1, 0))],
                particle=Sphere(r=0.77 * μm, material=silica))


def test_saddle_off_the_rays_matches_closed_form():
    # normal modes along the axes, saddle at 30° in the x-y plane: the rays miss it
    element = CubicWell(k=[1e-6, 2e-6, 3e-6], λ=2e5, n=[np.cos(np.pi / 6), np.sin(np.pi / 6), 0])
    trap = Trap(fields=[element], particle=PARTICLE)
    saddle, barrier = element.closed_form()
    result = find_escape_saddle(trap, np.zeros(3), 3 * np.max(np.abs(saddle)), grid_points=31)
    assert result['refined'] and not result['is_lower_bound']
    np.testing.assert_allclose(result['barrier'], barrier, rtol=1e-12)
    np.testing.assert_allclose(result['saddle'], saddle, atol=1e-12 * np.linalg.norm(saddle))
    assert np.sum(result['saddle_stiffness_eigenvalues'] < 0) == 1
    np.testing.assert_allclose(result['direction'], saddle / np.linalg.norm(saddle), atol=1e-12)
    # the grid value before refinement is within the grid resolution
    assert abs(result['grid_barrier'] / barrier - 1) < 0.1
    # a ray along the x normal mode crosses U = k_x³/(6 λ² n_x⁶), 1.59 times the saddle
    ray_x = element.k[0]**3 / (6 * element.λ**2 * element.n[0]**6)
    assert ray_x / barrier == pytest.approx(1.5879629629629626, rel=1e-12)


def test_saddle_on_a_ray_agrees_with_the_ray_estimate():
    # saddle along +z, a normal mode: rays and saddle search must agree (rays: trapezoid on 400 points)
    element = CubicWell(k=[3e-6, 2e-6, 1e-6], λ=2e5, n=[0, 0, 1])
    trap = Trap(fields=[element, MagneticTrap(LocalFieldExpansion(B0=(0, 0, 0)))], particle=PARTICLE)
    saddle, barrier = element.closed_form()
    rep = trap.magnetic_trap_report(r_guess=np.zeros(3), scan_range=2 * np.linalg.norm(saddle))
    assert rep['trap_depth_method'] == 'saddle' and rep['saddle_refined']
    np.testing.assert_allclose(rep['trap_depth'], barrier, rtol=1e-12)
    np.testing.assert_allclose(rep['saddle_point'], saddle, atol=1e-12 * np.linalg.norm(saddle))
    np.testing.assert_allclose(rep['trap_depth_rays'], barrier, rtol=1e-4)
    assert rep['trap_depth'] <= rep['trap_depth_rays'] * (1 + 1e-4)
    assert not rep['trap_depth_is_lower_bound']


def test_magnetic_saddle_checked_by_finite_differences():
    # a levitated bismuth sphere (vectorised magnetic and gravitational energies) plus the cubic term
    # (point by point), which opens a saddle off the rays; the refined saddle is checked against
    # independent central differences of static_energy
    particle = Sphere(r=2e-6, material=BISMUTH)
    probe = MagneticTrap(LocalFieldExpansion(B0=(0, 0, 1)))
    need = probe.required_B_gradB(particle)
    source = LocalFieldExpansion.axisymmetric(1.0, -need, -2e5)
    trap = Trap(fields=[MagneticTrap(source), Gravity(direction=(0, 0, -1)),
                        CubicWell(k=[0, 0, 0], λ=1e-4, n=[1, 1, 0.3])], particle=particle)
    rep = trap.magnetic_trap_report(r_guess=np.zeros(3), scan_range=40e-6)
    assert rep['trap_depth_method'] == 'saddle' and rep['saddle_refined'] and not rep['trap_depth_is_lower_bound']
    r_s, h = rep['saddle_point'], 1e-8
    E = np.eye(3) * h
    grad = np.array([(static_energy(trap, r_s + E[j]) - static_energy(trap, r_s - E[j])) / (2 * h) for j in range(3)])
    scale = rep['trap_depth'] / np.linalg.norm(r_s - rep['equilibrium'])
    assert np.linalg.norm(grad) < 1e-6 * scale
    hessian = np.array([[(static_energy(trap, r_s + E[i] + E[j]) - static_energy(trap, r_s + E[i] - E[j])
                          - static_energy(trap, r_s - E[i] + E[j]) + static_energy(trap, r_s - E[i] - E[j])) / (4 * h**2)
                         for j in range(3)] for i in range(3)])
    assert np.sum(np.linalg.eigvalsh(0.5 * (hessian + hessian.T)) < 0) == 1
    np.testing.assert_allclose(rep['trap_depth'], static_energy(trap, r_s) - static_energy(trap, rep['equilibrium']), rtol=1e-12)
    assert rep['trap_depth'] < rep['trap_depth_rays']


def test_static_energy_is_the_potential_of_the_static_force():
    trap = _slezak_trap()
    rng = np.random.default_rng(3)
    h = 1e-9
    for r in rng.uniform(-60e-6, 60e-6, size=(5, 3)):
        grad = np.array([(static_energy(trap, r + h * e) - static_energy(trap, r - h * e)) / (2 * h) for e in np.eye(3)])
        np.testing.assert_allclose(-grad, trap.static_force(r), rtol=1e-6, atol=1e-6 * np.linalg.norm(trap.static_force(r)))
    points = rng.uniform(-60e-6, 60e-6, size=(4, 3))
    np.testing.assert_allclose(static_energy(trap, points), [static_energy(trap, p) for p in points], rtol=1e-14)
    harmonic = Trap(fields=[HarmonicPotential([1e-6, 2e-6, 3e-6], centre=(1e-6, 0, 0))], particle=PARTICLE)
    np.testing.assert_allclose(static_energy(harmonic, points), [harmonic.fields[0].potential(p, PARTICLE) for p in points],
                               rtol=1e-14)


def test_slezak_trap_depth_against_rays():
    # the fitted three-multipole field has no saddle within 75 or 150 μm: the energy still rises along
    # the weak axis (z) at the faces of the cube, so the search returns a lower bound, set by the
    # face centre on the z axis, the same point as the lowest ray
    trap = _slezak_trap()
    for scan_range, expected in ((75 * μm, 1674.7086), (150 * μm, 6843.966)):
        rep = trap.magnetic_trap_report(r_guess=(0, -1 * μm, 0), scan_range=scan_range)
        assert rep['trap_depth_method'] == 'saddle'
        assert rep['trap_depth_is_lower_bound'] and rep['trap_depth_rays_is_lower_bound']
        assert rep['saddle_point'] is None and not rep['saddle_refined']
        np.testing.assert_allclose(rep['trap_depth'] / kB, expected, rtol=1e-6)
        np.testing.assert_allclose(rep['trap_depth'], rep['trap_depth_rays'], rtol=1e-6)
        np.testing.assert_allclose(np.abs(rep['trap_depth_direction']), [0, 0, 1], atol=1e-9)  # ±z
    text = trap.magnetic_trap_report_str(r_guess=(0, -1 * μm, 0))
    assert 'lower bound: no saddle within the scan range' in text and 'ray estimate' in text


def test_rays_option_and_fallback():
    trap = _slezak_trap()
    rays = trap.magnetic_trap_report(r_guess=(0, -1 * μm, 0), trap_depth_method='rays')
    assert rays['trap_depth_method'] == 'rays' and rays['saddle_search'] is None
    assert rays['trap_depth'] == rays['trap_depth_rays']
    assert np.array_equal(rays['trap_depth_direction'], rays['trap_depth_rays_direction'])
    with pytest.raises(ValueError, match='trap_depth_method'):
        trap.magnetic_trap_report(trap_depth_method='grid')
    no_potential = Trap(fields=[MagneticTrap(LocalFieldExpansion(B0=(0, 0, 0))), ForceOnly()], particle=PARTICLE)
    rep = no_potential.magnetic_trap_report(r_guess=np.zeros(3), scan_range=1e-6)
    assert rep['trap_depth_method'].startswith('rays (no saddle search')
    assert rep['trap_depth'] == rep['trap_depth_rays']
    with pytest.raises(ValueError, match='odd'):
        find_escape_saddle(trap, np.zeros(3), 1e-5, grid_points=40)

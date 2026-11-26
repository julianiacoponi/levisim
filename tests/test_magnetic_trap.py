"""
Static magnetic fields and diamagnetic levitation: field sources (Maxwell identities, exact formulas),
MagneticTrap (levitation condition, stiffness, frequencies, equilibrium, torque), EddyCurrentDamping
and the time-domain simulation
"""
import warnings

import numpy as np
import pytest
from scipy.signal import welch

from levisim.core.constants import g as g_earth
from levisim.core.constants import μ0
from levisim.core.units import Hz, mbar
from levisim.physics import Sphere, Trap, Tweezer
from levisim.physics.fields import (
    CallableField,
    CoilPair,
    CuboidMagnet,
    CurrentLoop,
    EddyCurrentDamping,
    FieldSum,
    Gravity,
    LocalFieldExpansion,
    MagneticTrap,
    MultipoleField,
    effective_susceptibility,
    real_solid_harmonic,
)
from levisim.physics.gas import Gas
from levisim.physics.materials import BISMUTH, FUSED_SILICA, NDFEB, PYROLYTIC_GRAPHITE
from levisim.physics.particle import axis_angle_quaternion
from levisim.physics.shapes import Spheroid
from levisim.simulation import FreeSpaceSimulation
from levisim.simulation.backends import resolve_backend

RNG = np.random.default_rng(12)


def _rotation(axis, angle):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def _biot_savart_loop(r, loop, n=20000):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    R = loop.rotation if loop.rotation is not None else np.eye(3)
    points = (np.stack([np.cos(t), np.sin(t), 0 * t], -1) * loop.radius) @ R.T + loop.centre
    dl = (np.stack([-np.sin(t), np.cos(t), 0 * t], -1) * loop.radius * 2 * np.pi / n) @ R.T
    d = r - points
    return μ0 * loop.current / (4 * np.pi) * np.sum(np.cross(dl, d) / np.linalg.norm(d, axis=1)[:, None]**3, 0)


def _dipole_field(m):
    def B(r):
        r = np.asarray(r, float)
        rn = np.linalg.norm(r, axis=-1, keepdims=True)
        n = r / rn
        return μ0 / (4 * np.pi) * (3 * n * np.sum(n * m, axis=-1, keepdims=True) - m) / rn**3
    return B


SOURCES = {
    'loop': (CurrentLoop(2e-3, 1.5, centre=(1e-4, -2e-4, 3e-4), axis=(1, 2, 2)), 1e-3),
    'coil_pair': (CoilPair.anti_helmholtz(5e-3, 10.0, axis=(0, 1, 0)), 2e-3),
    'cuboid': (CuboidMagnet((1e-3, 2e-3, 0.5e-3), magnetisation=(3e5, -2e5, 8e5), centre=(0, 0, -2e-3),
                            rotation=_rotation((1, 1, 0), 0.4)), 1e-3),
    'multipole': (MultipoleField({(2, -2): -0.8, (3, 1): 0.007, (4, -4): 0.22, (2, 0): 0.1}, r0=75e-6), 3e-5),
    'axisymmetric': (LocalFieldExpansion.axisymmetric(1.0, -30.0, 400.0, axis=(0, 1, 1)), 1e-3),
    'callable': (CallableField(_dipole_field(np.array([0.0, 0.0, 1e-3])), length_scale=1e-3, centre=(0, 0, -3e-3)), 1e-3),
}


####################################################################################################
# Field identities
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('name', sorted(SOURCES))
def test_field_is_divergence_and_curl_free(name):
    source, scale = SOURCES[name]
    points = RNG.normal(scale=scale, size=(6, 3))
    J = source.gradB(points)
    norm = np.linalg.norm(J, axis=(-2, -1))
    assert np.all(np.abs(source.divergence(points)) <= 1e-8 * norm)
    assert np.all(np.linalg.norm(source.curl(points), axis=-1) <= 1e-8 * norm)
    # hessB is symmetric in its derivative indices and each component is harmonic (∇²B_i = 0)
    H = source.hessB(points)
    h = np.max(np.abs(H))
    assert np.max(np.abs(H - np.swapaxes(H, -1, -2))) <= 1e-6 * h
    assert np.max(np.abs(np.einsum('...ijj->...i', H))) <= 1e-6 * h


@pytest.mark.parametrize('name', sorted(SOURCES))
def test_derivatives_match_differences_of_the_field(name):
    source, scale = SOURCES[name]
    r = RNG.normal(scale=scale, size=3)
    h = 1e-4 * source.length_scale
    J_fd = np.stack([(source.B(r + h * e) - source.B(r - h * e)) / (2 * h) for e in np.eye(3)], axis=-1)
    np.testing.assert_allclose(source.gradB(r), J_fd, atol=1e-6 * np.abs(J_fd).max())
    H_fd = np.stack([(source.gradB(r + h * e) - source.gradB(r - h * e)) / (2 * h) for e in np.eye(3)], axis=-1)
    np.testing.assert_allclose(source.hessB(r), H_fd, atol=1e-5 * np.abs(H_fd).max())


def test_vectorised_and_pointwise_evaluation_agree():
    for source, scale in SOURCES.values():
        points = RNG.normal(scale=scale, size=(2, 3, 3))
        B = source.B(points)
        assert B.shape == (2, 3, 3) and source.gradB(points).shape == (2, 3, 3, 3)
        # round-off is relative to the field's magnitude, not to each (possibly near-zero) component
        B_point = source.B(points[1, 2])
        np.testing.assert_allclose(B[1, 2], B_point, rtol=0, atol=1e-12 * np.linalg.norm(B_point))


def test_current_loop_matches_biot_savart_and_on_axis_formula():
    loop = SOURCES['loop'][0]
    for r in (np.array([1e-3, 2e-4, -5e-4]), np.array([3e-3, -1e-3, 2e-3]), loop.centre + 1e-9 * loop.axis + 1e-8):
        exact = _biot_savart_loop(r, loop)
        np.testing.assert_allclose(loop.B(r), exact, atol=1e-12 * np.linalg.norm(exact))
    s = np.array([-3e-3, 0.0, 1e-3, 5e-3])
    np.testing.assert_allclose(
        loop.B(loop.centre + s[:, None] * loop.axis) @ loop.axis, loop.on_axis_field(s), rtol=1e-13
    )
    np.testing.assert_allclose(loop.on_axis_field(0.0), μ0 * 1.5 / (2 * 2e-3), rtol=1e-15)


def test_current_loop_near_axis_series_is_continuous():
    # the on-axis series (ρ < 1e-3 a) and the elliptic-integral form on either side of the switch both
    # agree with direct Biot-Savart quadrature
    loop = CurrentLoop(1e-2, 2.0)
    for z in (0.0, 3e-3):
        for ρ in (0.9999e-5, 1.0001e-5):
            r = np.array([ρ, 0.0, z])
            exact = _biot_savart_loop(r, loop)
            np.testing.assert_allclose(loop.B(r)[2], exact[2], rtol=1e-11)
            np.testing.assert_allclose(loop.B(r)[0], exact[0], rtol=1e-9, atol=1e-11 * abs(exact[2]))


def test_anti_helmholtz_pair_gradient_and_zero():
    a, I = 5e-3, 10.0
    pair = CoilPair(a, I, 4e-3)
    np.testing.assert_allclose(pair.B(np.zeros(3)), 0.0, atol=1e-18)
    expected = 3 * μ0 * I * a**2 * 2e-3 / (a**2 + 4e-6)**2.5
    np.testing.assert_allclose(pair.gradB(np.zeros(3))[2, 2], expected, rtol=1e-9)
    # the radial gradients are -½ the axial one (∇·B = 0)
    np.testing.assert_allclose(np.diag(pair.gradB(np.zeros(3))), [-expected / 2, -expected / 2, expected], rtol=1e-9)
    h = CoilPair.helmholtz(a, I)
    np.testing.assert_allclose(h.hessB(np.zeros(3))[2, 2, 2], 0.0, atol=1e-6 * μ0 * I / a**3)


def test_cuboid_on_axis_closed_form_and_dipole_far_field():
    a, b, c, Br = 1e-3, 2e-3, 3e-3, 1.2
    block = CuboidMagnet((2 * a, 2 * b, 2 * c), remanence=Br)

    def on_axis(z):
        f = lambda t: np.arctan(a * b / (t * np.sqrt(a * a + b * b + t * t)))
        return Br / np.pi * (f(z - c) - f(z + c))

    for z in (3.5e-3, 5e-3, 2e-2):
        np.testing.assert_allclose(block.B(np.array([0, 0, z])), [0, 0, on_axis(z)], rtol=1e-13, atol=1e-16)
    r = np.array([0.3, 0.2, -0.4])
    m = block.magnetic_moment
    n = r / np.linalg.norm(r)
    dipole = μ0 / (4 * np.pi) * (3 * n * (m @ n) - m) / np.linalg.norm(r)**3
    np.testing.assert_allclose(block.B(r), dipole, rtol=1e-4)
    # inside, B = μ0 (H + M): the normal component is continuous through the top face
    np.testing.assert_allclose(block.B(np.array([0, 0, c - 1e-9]))[2], block.B(np.array([0, 0, c + 1e-9]))[2], rtol=1e-5)


def test_cuboid_components_by_superposition():
    M = np.array([2e5, -1e5, 5e5])
    whole = CuboidMagnet((1e-3, 2e-3, 3e-3), magnetisation=M)
    parts = FieldSum([CuboidMagnet((1e-3, 2e-3, 3e-3), magnetisation=M * e) for e in np.eye(3)])
    r = np.array([1.1e-3, -0.4e-3, 2.2e-3])
    np.testing.assert_allclose(whole.B(r), parts.B(r), rtol=1e-13)
    # rotating the block by a quarter turn about z maps M_x onto M_y
    turned = CuboidMagnet((2e-3, 1e-3, 3e-3), magnetisation=(2e5, 0, 0), rotation=_rotation((0, 0, 1), np.pi / 2))
    upright = CuboidMagnet((1e-3, 2e-3, 3e-3), magnetisation=(0, 2e5, 0))
    np.testing.assert_allclose(turned.B(r), upright.B(r), rtol=1e-12, atol=1e-18)


@pytest.mark.parametrize('l', range(1, 6))
def test_real_solid_harmonics_are_harmonic_and_normalised(l):
    for m in range(-l, l + 1):
        p = real_solid_harmonic(l, m)
        laplacian = p.derivative(0).derivative(0) + p.derivative(1).derivative(1) + p.derivative(2).derivative(2)
        assert all(abs(c) < 1e-12 for c in laplacian.terms.values())
        # normalised on the unit sphere: Gauss-Legendre in cos θ and the trapezoid rule in φ are exact
        # for these polynomials
        c, w = np.polynomial.legendre.leggauss(12)
        φ = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        C, Φ = np.meshgrid(c, φ, indexing='ij')
        S = np.sqrt(1 - C**2)
        u = np.stack([S * np.cos(Φ), S * np.sin(Φ), C], -1)
        integral = np.sum(w[:, None] * p(u)**2) * 2 * np.pi / len(φ)
        np.testing.assert_allclose(integral, 1.0, rtol=1e-12)


def test_solid_harmonics_match_the_tabulated_forms():
    # the forms written out in Hsu et al., Sci. Rep. 6, 30125 (2016), Eq. (1)-(3)
    u = RNG.normal(size=(5, 3))
    x, y, z = u.T
    np.testing.assert_allclose(real_solid_harmonic(2, -2)(u), 0.5 * np.sqrt(15 / np.pi) * x * y, rtol=1e-14)
    np.testing.assert_allclose(
        real_solid_harmonic(3, 1)(u), 0.25 * np.sqrt(21 / (2 * np.pi)) * x * (4 * z**2 - x**2 - y**2), rtol=1e-13
    )
    np.testing.assert_allclose(
        real_solid_harmonic(4, -4)(u), 0.75 * np.sqrt(35 / np.pi) * x * y * (x**2 - y**2), rtol=1e-13
    )


def test_local_expansion_checks_maxwell():
    with pytest.warns(UserWarning, match='does not satisfy'):
        LocalFieldExpansion(gradient=np.diag([1.0, 1.0, 1.0]))
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        expansion = LocalFieldExpansion.axisymmetric(0.5, 20.0, -300.0, axis=(1, 0, 0))
    assert max(expansion.maxwell_residuals().values()) < 1e-14
    s = np.linspace(-1e-3, 1e-3, 5)
    np.testing.assert_allclose(expansion.B(s[:, None] * [1, 0, 0])[:, 0], 0.5 + 20 * s - 150 * s**2, rtol=1e-14)


####################################################################################################
# MagneticTrap: levitation condition, stiffness, frequencies
# --------------------------------------------------------------------------------------------------
def test_effective_susceptibility_of_a_sphere():
    χ = -170e-6
    np.testing.assert_allclose(effective_susceptibility(χ, [1 / 3] * 3), χ / (1 + χ / 3) * np.eye(3), rtol=1e-15)
    element = MagneticTrap(LocalFieldExpansion(B0=(0, 0, 1)))
    np.testing.assert_allclose(element.susceptibility(Sphere(r=1e-6, material=BISMUTH)), χ / (1 + χ / 3) * np.eye(3))
    bare = MagneticTrap(LocalFieldExpansion(B0=(0, 0, 1)), shape_correction=False)
    np.testing.assert_array_equal(bare.susceptibility(Sphere(r=1e-6, material=BISMUTH)), χ * np.eye(3))


@pytest.mark.parametrize('material, χ', [(BISMUTH, -170e-6), (FUSED_SILICA, -11.28e-6)])
def test_levitation_condition_isotropic(material, χ):
    particle = Sphere(r=2e-6, material=material)
    bare = MagneticTrap(LocalFieldExpansion(B0=(0, 0, 1)), shape_correction=False)
    np.testing.assert_allclose(bare.required_B_gradB(particle), μ0 * material.ρ * g_earth / abs(χ), rtol=1e-15)
    corrected = MagneticTrap(LocalFieldExpansion(B0=(0, 0, 1)))
    np.testing.assert_allclose(
        corrected.required_B_gradB(particle), μ0 * material.ρ * g_earth / abs(χ / (1 + χ / 3)), rtol=1e-14
    )


def test_levitation_condition_matches_houlton_table_for_silica():
    # Houlton et al., Rev. Sci. Instrum. 89, 125107 (2018), Table 1: SiO2, ρ = 2200 kg/m³, χ = -11.28e-6,
    # levitation (B∇B)_z = 2400 T²/m (quoted to 2 significant figures)
    particle = Sphere(r=1e-6, material=FUSED_SILICA.replace(ρ=2200.0))
    value = MagneticTrap(LocalFieldExpansion(B0=(0, 0, 1)), shape_correction=False).required_B_gradB(particle)
    assert abs(value / 2400 - 1) < 5e-3


def test_levitation_condition_pyrolytic_graphite_depends_on_field_direction():
    particle = Sphere(r=5e-6, material=PYROLYTIC_GRAPHITE)  # body c axis along lab z
    element = MagneticTrap(LocalFieldExpansion(B0=(0, 0, 1)), shape_correction=False)
    with pytest.raises(ValueError, match='anisotropic'):
        element.required_B_gradB(particle)
    ρ = PYROLYTIC_GRAPHITE.ρ
    np.testing.assert_allclose(element.required_B_gradB(particle, field_direction=(0, 0, 1)), μ0 * ρ * g_earth / 450e-6)
    np.testing.assert_allclose(element.required_B_gradB(particle, field_direction=(1, 0, 0)), μ0 * ρ * g_earth / 85e-6)
    # turned by 90° about x, the c axis lies along lab y
    particle.orientation = axis_angle_quaternion((1, 0, 0), np.pi / 2)
    np.testing.assert_allclose(
        element.required_B_gradB(particle, field_direction=(0, 1, 0)), μ0 * ρ * g_earth / 450e-6, rtol=1e-12
    )


def _quadrupole(G0=1e5, ratios=(1.0, 0.5, -1.5), centre=(0, 0, 0)):
    return LocalFieldExpansion(gradient=G0 * np.diag(ratios), centre=centre)


@pytest.mark.filterwarnings('ignore:invalid value encountered in sqrt:RuntimeWarning')
def test_frequencies_in_a_pure_quadratic_field():
    particle = Sphere(r=1e-6, material=BISMUTH)
    χ = MagneticTrap(_quadrupole()).susceptibility(particle)[0, 0]
    trap = Trap(fields=[MagneticTrap(_quadrupole())], particle=particle)
    expected_K = -χ * particle.V / μ0 * (1e5)**2 * np.diag([1.0, 0.25, 2.25])
    np.testing.assert_allclose(trap.stiffness_tensor, expected_K, rtol=1e-14)
    Ω = np.sqrt(abs(χ) / (μ0 * BISMUTH.ρ)) * 1e5 * np.array([1.0, 0.5, 1.5])
    np.testing.assert_allclose([trap.Ωx, trap.Ωy, trap.Ωz], Ω, rtol=1e-14)
    # independent of the particle size
    small = Trap(fields=[MagneticTrap(_quadrupole())], particle=Sphere(r=50e-9, material=BISMUTH))
    np.testing.assert_allclose([small.Ωx, small.Ωy, small.Ωz], Ω, rtol=1e-12)
    # a paramagnet is expelled from the field minimum
    para = Trap(fields=[MagneticTrap(_quadrupole())], particle=Sphere(r=1e-6, material=BISMUTH.replace(χ_m=1e-4)))
    assert np.all(np.diag(para.stiffness_tensor) < 0)


def test_force_is_minus_gradient_of_potential_and_stiffness_its_derivative():
    for particle in (Sphere(r=1e-6, material=BISMUTH), Spheroid(1e-6, 2.0, material=PYROLYTIC_GRAPHITE)):
        particle.orientation = axis_angle_quaternion((1, 1, 0), 0.3)
        element = MagneticTrap(CoilPair(3e-3, 50.0, 2.5e-3))
        r = np.array([2e-4, -1e-4, 3e-4])
        h = 1e-7
        grad_U = np.array([(element.potential(r + h * e, particle) - element.potential(r - h * e, particle)) / (2 * h)
                           for e in np.eye(3)])
        F = element.force(r, 0.0, particle)
        np.testing.assert_allclose(F, -grad_U, rtol=1e-6, atol=1e-8 * np.abs(F).max())
        K_fd = -np.stack([(element.force(r + h * e, 0, particle) - element.force(r - h * e, 0, particle)) / (2 * h)
                          for e in np.eye(3)], axis=-1)
        np.testing.assert_allclose(element.stiffness(r, particle), K_fd, rtol=1e-5, atol=1e-6 * np.abs(K_fd).max())


def test_torque_on_induced_moment():
    element = MagneticTrap(LocalFieldExpansion.axisymmetric(1.0, 10.0, 0.0))
    sphere = Sphere(r=1e-6, material=BISMUTH)
    np.testing.assert_allclose(element.torque((1e-4, 0, 0), 0, sphere), 0.0, atol=1e-30)
    flake = Sphere(r=1e-6, material=PYROLYTIC_GRAPHITE, orientation=axis_angle_quaternion((1, 0, 0), 0.4))
    τ = element.torque(np.zeros(3), 0, flake)
    m = element.magnetic_moment(np.zeros(3), flake)
    np.testing.assert_allclose(τ, np.cross(m, [0, 0, 1.0]), rtol=1e-14)
    assert np.linalg.norm(τ) > 0
    # the torque turns the c axis (the most diamagnetic direction) away from the field: τ·x̂ > 0 for a
    # c axis tilted towards -y (rotation +0.4 about x), i.e. it increases the tilt
    assert τ[0] > 0


def test_remanent_particle_is_rejected():
    with pytest.raises(NotImplementedError, match='remanent'):
        MagneticTrap(_quadrupole()).force(np.zeros(3), 0, Sphere(r=1e-6, material=NDFEB))


####################################################################################################
# Equilibrium with gravity and the report
# --------------------------------------------------------------------------------------------------
def _levitating_bismuth(z_eq=0.0):
    """A bismuth sphere in an axisymmetric field along z whose B·dB/dz exactly balances gravity at z_eq"""
    particle = Sphere(r=2e-6, material=BISMUTH)
    element = MagneticTrap(LocalFieldExpansion(B0=(0, 0, 1)))
    need = element.required_B_gradB(particle)
    # [T], [T/m²]: stable when B1² + B0 B2 > 0 (vertical) and B1²/4 - B0 B2/2 > 0 (radial), with
    # B1 ≈ -707 T/m here
    B0, B2 = 1.0, -2e5
    B1 = -need / B0  # |B| decreasing upwards: B ∂B/∂z = -need at the origin
    source = LocalFieldExpansion.axisymmetric(B0, B1, B2, centre=(0, 0, z_eq))
    return particle, Trap(fields=[MagneticTrap(source), Gravity(direction=(0, 0, -1))], particle=particle)


def test_equilibrium_and_report_levitating_bismuth():
    particle, trap = _levitating_bismuth()
    rep = trap.magnetic_trap_report(r_guess=(0, 0, 3e-6), scan_range=1e-4)
    np.testing.assert_allclose(rep['equilibrium'], 0.0, atol=1e-15)
    np.testing.assert_allclose(rep['B_gradB'], rep['required_B_gradB'], rtol=1e-12)
    np.testing.assert_allclose(rep['magnetic_force'], [0, 0, particle.m * g_earth], rtol=1e-12, atol=1e-25)
    assert rep['levitation_margin'] > 1 and rep['stable']
    # stiffness at the equilibrium in closed form: U = -(χV/2μ0)|B|², |B|² to second order
    χ = rep['χ_along_B']
    B0, B1, B2 = 1.0, -MagneticTrap(LocalFieldExpansion(B0=(0, 0, 1))).required_B_gradB(particle), -2e5
    k_z = -χ * particle.V / μ0 * (B1**2 + B0 * B2)
    k_ρ = -χ * particle.V / μ0 * (B1**2 / 4 - B0 * B2 / 2)
    np.testing.assert_allclose(np.diag(rep['stiffness']), [k_ρ, k_ρ, k_z], rtol=1e-12)
    assert rep['trap_depth'] > 0
    assert 'Ω_axes/2π' in trap.magnetic_trap_report_str(r_guess=(0, 0, 3e-6), scan_range=1e-4)


def test_report_eddy_damping_for_isotropic_and_anisotropic_conductors():
    particle, trap = _levitating_bismuth()
    rep = trap.magnetic_trap_report(r_guess=(0, 0, 3e-6), scan_range=1e-4)
    source = trap.fields[0].source
    np.testing.assert_allclose(rep['eddy_damping_rates'],
                               EddyCurrentDamping(source, rep['equilibrium']).damping_rates(particle), rtol=1e-14)
    assert rep['eddy_damping_rate_bounds'] is None
    graphite = Sphere(r=2e-6, material=PYROLYTIC_GRAPHITE)
    source = _quadrupole(G0=1e3)
    g_trap = Trap(fields=[MagneticTrap(source)], particle=graphite)
    g_rep = g_trap.magnetic_trap_report(r_guess=(0, 0, 0), scan_range=1e-4)
    assert g_rep['eddy_damping_rates'] is None
    low, high = g_rep['eddy_damping_rate_bounds']
    np.testing.assert_allclose(high, 1000 * low, rtol=1e-12)
    np.testing.assert_allclose(high, EddyCurrentDamping(source, g_rep['equilibrium'], conductivity=2e5)
                               .damping_rates(graphite), rtol=1e-14)
    assert '(anisotropic σ)' in g_trap.magnetic_trap_report_str(r_guess=(0, 0, 0), scan_range=1e-4)


def test_find_equilibrium_and_linearisation_at_equilibrium():
    _, trap = _levitating_bismuth(z_eq=7e-6)
    r = trap.find_equilibrium(r_guess=(0, 0, 0))
    np.testing.assert_allclose(r, [0, 0, 7e-6], atol=1e-14)
    at_eq = Trap(fields=trap.fields, particle=trap.particle, linearisation_point='equilibrium')
    np.testing.assert_allclose(at_eq.linearisation_point, [0, 0, 7e-6], atol=1e-14)
    assert np.all(np.isfinite([at_eq.Ωx, at_eq.Ωy, at_eq.Ωz]))


def test_off_diagonal_stiffness_becomes_intra_trap_couplings():
    # a quadrupole whose principal axes are turned by 30° about z couples x and y
    G = _rotation((0, 0, 1), np.pi / 6) @ np.diag([1.0, 0.5, -1.5]) @ _rotation((0, 0, 1), np.pi / 6).T * 1e5
    trap = Trap(fields=[MagneticTrap(LocalFieldExpansion(gradient=G))], particle=Sphere(r=1e-6, material=BISMUTH))
    assert abs(trap.intra_trap_coupling_stiffness[0, 1]) > 0
    np.testing.assert_allclose(trap.intra_trap_coupling_stiffness, trap.intra_trap_coupling_stiffness.T)
    assert trap.intra_trap_couplings['xy'] != 0 and trap.intra_trap_couplings['xz'] == 0


@pytest.mark.filterwarnings('ignore:invalid value encountered in sqrt:RuntimeWarning')
def test_hsu_2016_multipole_trap_regression():
    # Hsu et al., Sci. Rep. 6, 30125 (2016): a2 = -1.3 T, a3 = 0.018 T, a4 = 0.72 T, y0 = 75 μm give, for their
    # diamond cluster, ω/2π = (104, 130, 9.6) Hz and y_eq = -19 μm. Their χ is not stated; with
    # χ = -2.2e-5, ρ = 3510 kg/m³ (assumed here) levisim gives (106.1, 130.3, 9.56) Hz and -18.87 μm,
    # within the 2-significant-figure rounding of the published coefficients
    diamond = FUSED_SILICA.replace(name='diamond (test)', ρ=3510.0, χ_m=-2.2e-5, permittivity=None)
    source = MultipoleField({(2, -2): -1.3, (3, 1): 0.018, (4, -4): 0.72}, r0=75e-6)
    trap = Trap(fields=[MagneticTrap(source, shape_correction=False), Gravity(g=9.80665, direction=(0, -1, 0))],
                particle=Sphere(r=0.5e-6, material=diamond))
    rep = trap.magnetic_trap_report(r_guess=(0, -1e-6, 0))
    np.testing.assert_allclose(rep['equilibrium'][1], -19e-6, rtol=0.02)
    np.testing.assert_allclose(rep['Ω_axes'] / Hz, [104, 130, 9.6], rtol=0.025)
    np.testing.assert_allclose(rep['B_magnitude'], 0.2, rtol=0.1)  # "Beq ≈ 0.2 T"


####################################################################################################
# Eddy-current damping
# --------------------------------------------------------------------------------------------------
def test_eddy_current_damping_formula_and_trap_rates():
    source = _quadrupole(G0=1e3)
    σ = 7.7e5
    particle = Sphere(r=20e-6, material=BISMUTH)
    eddy = EddyCurrentDamping(source, conductivity=σ)
    J = source.gradB(np.zeros(3))
    np.testing.assert_allclose(eddy.damping_tensor(particle), σ * particle.r**2 * J.T @ J / (10 * BISMUTH.ρ), rtol=1e-14)
    gas = Gas(P=1e-6 * mbar)
    without = Trap(fields=[MagneticTrap(source)], particle=particle, gas=gas)
    with_eddy = Trap(fields=[MagneticTrap(source), eddy], particle=particle, gas=gas)
    np.testing.assert_allclose(
        [with_eddy.Γx, with_eddy.Γy, with_eddy.Γz],
        np.array([without.Γx, without.Γy, without.Γz]) + np.diag(eddy.damping_tensor(particle)),
        rtol=1e-14,
    )
    np.testing.assert_allclose(np.diag(eddy.noise(particle)), 2 * 1.380649e-23 * 300 * particle.m
                               * np.diag(eddy.damping_tensor(particle)), rtol=1e-12)
    with pytest.raises(ValueError, match='conductivity'):
        EddyCurrentDamping(source).damping_rates(Sphere(r=20e-6, material=BISMUTH.replace(conductivity=None)))
    # the library bismuth carries a cited conductivity (decision 16), used when none is given
    np.testing.assert_allclose(EddyCurrentDamping(source).damping_rates(particle),
                               eddy.damping_rates(particle) * BISMUTH.conductivity / σ, rtol=1e-14)
    with pytest.raises(NotImplementedError, match='sphere'):
        eddy.damping_rates(Spheroid(1e-6, 2.0, material=BISMUTH))


####################################################################################################
# Time domain
# --------------------------------------------------------------------------------------------------
def _peak_frequencies(sim, axes='xyz'):
    out = []
    for axis in axes:
        f, S = welch(sim.traces[axis].values, fs=1 / sim.Δt, nperseg=4096)
        i = int(np.argmax(S[1:])) + 1
        # parabolic interpolation of the log-peak
        a, b, c = np.log(S[i - 1:i + 2])
        out.append(f[i] + 0.5 * (a - c) / (a - 2 * b + c) * (f[1] - f[0]))
    return np.array(out)


def test_simulated_psd_peaks_match_stiffness_with_gravity():
    particle = Sphere(r=1e-6, material=BISMUTH)
    fields = [MagneticTrap(_quadrupole()), Gravity(direction=(0, 0, -1))]
    trap = Trap(fields=fields, particle=particle, gas=Gas(P=5 * mbar), β=0.0, linearisation_point='equilibrium')
    # the gravitational sag g/Ωz² along -z
    np.testing.assert_allclose(trap.linearisation_point, [0, 0, -g_earth / trap.Ωz**2], rtol=1e-10, atol=1e-20)
    X0 = np.r_[trap.linearisation_point, 0, 0, 0]
    sim = FreeSpaceSimulation(trap=trap, num_samples=30_000, Δt=10e-6, X0=X0, seed=3, backend='auto')
    sim.simulate_dynamics()
    # NumPy below AUTO_NUMBA_MIN_STEPS (or without numba); the compiled MagneticTrap kernel above it
    assert sim.backend_used == resolve_backend('auto', 30_000)
    expected = np.array([trap.Ωx, trap.Ωy, trap.Ωz]) / Hz
    np.testing.assert_allclose(_peak_frequencies(sim), expected, rtol=0.02)


def test_hybrid_magneto_optical_trap_sums_stiffness():
    particle = Sphere(r=1e-6, material=FUSED_SILICA)
    magnetic = MagneticTrap(_quadrupole(G0=1e4))
    tweezer = Tweezer()
    hybrid = Trap(fields=[tweezer, magnetic], particle=particle)
    np.testing.assert_allclose(
        hybrid.stiffness_tensor, tweezer.stiffness(None, particle) + magnetic.stiffness(None, particle), rtol=1e-15
    )
    sim = FreeSpaceSimulation(trap=hybrid, num_samples=200, Δt=0.1e-6, seed=1, backend='auto')
    sim.simulate_dynamics()
    assert sim.backend_used == 'numpy' and np.all(np.isfinite(sim.traces['x'].values))

"""
Non-spherical particles: analytic shapes, orientation, gas damping tensors, TabulatedParticle

- orientation: quaternion ↔ rotation matrix, lab-frame tensors
- depolarisation factors (closed forms, Carlson's integral, a direct quadrature), areas, volumes,
  inertia tensors (against direct volume integrals)
- the sphere limit of Ellipsoid and Spheroid is bit-identical to Sphere and Bead, alone and in a Trap
- polarisability tensors with radiation reaction and birefringent materials
- free-molecular gas damping: the numerical surface integrals of Martinetz et al. (2018), Eq. (40),
  against the closed forms of their Table I (sphere and cylinder)
- translational trapping of a spheroid uses the polarisability projected on the tweezer polarisation
- TabulatedParticle: validation and round trips
"""
import warnings

import numpy as np
import pytest
from scipy.integrate import quad

from levisim.core.constants import kB, ε0, π
from levisim.core.units import nm, μm
from levisim.physics import materials as M
from levisim.physics.bead import Bead
from levisim.physics.gas import Gas
from levisim.physics.gas_damping import (
    cylinder_surface_quadrature,
    diffusion_matrix,
    ellipsoid_surface_quadrature,
    friction_matrix,
    gas_prefactor,
    surface_integrals,
)
from levisim.physics.particle import (
    Sphere,
    axis_angle_quaternion,
    body_axis_to_quaternion,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
)
from levisim.physics.shapes import (
    Cylinder,
    Disk,
    Dumbbell,
    Ellipsoid,
    Rod,
    Spheroid,
    depolarisation_factors,
    ellipsoid_polarisability,
    ellipsoid_surface_area,
    spheroid_depolarisation_factors,
)
from levisim.physics.tabulated import TabulatedParticle
from levisim.physics.trap import Trap

R0 = 60.1*nm
GAS = Gas()
RNG = np.random.default_rng(20261120)


def mixed_close(A, B, rtol):
    """A ≈ B for 6×6 matrices whose blocks carry different units (scaled by B's diagonal)"""
    d = np.sqrt(np.abs(np.diag(B)))
    return np.allclose(A / np.outer(d, d), B / np.outer(d, d), rtol=rtol, atol=rtol)


def random_quaternion(rng=RNG):
    q = rng.normal(size=4)
    return q / np.linalg.norm(q)


def bitwise_equal(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


# --------------------------------------------------------------------------------------------------
# orientation
# --------------------------------------------------------------------------------------------------
def test_quaternion_rotation_matrix_round_trip():
    for _ in range(50):
        q = random_quaternion()
        R = quaternion_to_rotation_matrix(q)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-14)
        assert np.isclose(np.linalg.det(R), 1.0)
        q_back = rotation_matrix_to_quaternion(R)
        assert np.allclose(q_back, q if q[0] >= 0 else -q, atol=1e-12)


def test_axis_angle_and_body_axis_quaternions():
    R = quaternion_to_rotation_matrix(axis_angle_quaternion('z', π/2))
    assert np.allclose(R @ [1, 0, 0], [0, 1, 0], atol=1e-15)
    for direction in ([0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 0, -1], [1, 2, -3]):
        d = np.asarray(direction, float) / np.linalg.norm(direction)
        R = quaternion_to_rotation_matrix(body_axis_to_quaternion(direction))
        assert np.allclose(R @ [0, 0, 1], d, atol=1e-14)


def test_orientation_is_validated_and_rotates_lab_tensors():
    with pytest.raises(ValueError):
        Spheroid(100*nm, 50*nm, orientation=(1, 1, 0, 0))
    sp = Spheroid(100*nm, 50*nm)
    body = sp.polarisability_tensor()
    assert np.array_equal(sp.lab_polarisability_tensor(), body)
    sp.orientation = body_axis_to_quaternion((0, 1, 0))
    lab = sp.lab_polarisability_tensor()
    assert np.isclose(lab[1, 1], body[2, 2], rtol=1e-14)
    assert np.isclose(lab[0, 0], body[0, 0], rtol=1e-14)
    assert np.isclose(sp.α, body[2, 2], rtol=1e-14)
    assert np.allclose(sp.lab_inertia_tensor, sp.rotation_matrix @ sp.inertia_tensor @ sp.rotation_matrix.T)
    with pytest.raises(ValueError):
        sp.orientation = (0.5, 0, 0, 0)


# --------------------------------------------------------------------------------------------------
# depolarisation factors, areas, volumes, inertia
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('abc', [(1, 2, 3), (3, 1, 2), (1, 1, 4), (5, 5, 1), (1, 1.0001, 1), (0.1, 2, 7)])
def test_depolarisation_factors_sum_to_one_and_match_the_integral(abc):
    L = depolarisation_factors(*abc)
    assert np.isclose(sum(L), 1.0, rtol=0, atol=1e-14)
    a, b, c = abc
    for i, ai in enumerate(abc):
        integrand = lambda s, ai=ai: 1 / ((s + ai**2) * np.sqrt((s + a**2) * (s + b**2) * (s + c**2)))
        direct = a * b * c / 2 * quad(integrand, 0, np.inf, epsabs=0, epsrel=1e-12, limit=200)[0]
        assert np.isclose(L[i], direct, rtol=1e-9)


@pytest.mark.parametrize('p', [1e-3, 0.05, 0.3, 0.9, 0.999, 1.001, 1.2, 3, 20, 1e3])
def test_spheroid_closed_forms_match_carlson(p):
    closed = spheroid_depolarisation_factors(p)
    carlson = depolarisation_factors(1, 1, p)
    assert np.allclose(closed, carlson, rtol=1e-10, atol=1e-15)
    assert np.isclose(sum(closed), 1.0, atol=1e-14)


def test_depolarisation_limits():
    assert depolarisation_factors(2, 2, 2) == (1/3, 1/3, 1/3)
    assert spheroid_depolarisation_factors(1.0) == (1/3, 1/3, 1/3)
    assert np.allclose(spheroid_depolarisation_factors(1e6), (0.5, 0.5, 0.0), atol=1e-10)
    assert np.allclose(spheroid_depolarisation_factors(1e-6), (0.0, 0.0, 1.0), atol=1e-5)
    # prolate along z: the field along the long axis is screened least
    L = spheroid_depolarisation_factors(3)
    assert L[2] < 1/3 < L[0]


@pytest.mark.parametrize('abc', [(1, 2, 3), (1, 1, 3), (3, 1, 1), (2, 2, 1), (0.2, 1, 5)])
def test_ellipsoid_area_matches_quadrature(abc):
    _, _, weights = ellipsoid_surface_quadrature(*abc)
    assert np.isclose(ellipsoid_surface_area(*abc), weights.sum(), rtol=1e-12)


@pytest.mark.parametrize('a, b', [(300*nm, 100*nm), (50*nm, 200*nm)])
def test_spheroid_area_closed_form(a, b):
    assert np.isclose(Spheroid(a, b).A, ellipsoid_surface_area(b, b, a), rtol=1e-13)


def volume_integrals(points, weights, ρ):
    """mass and inertia tensor from a volume quadrature"""
    m = ρ * weights.sum()
    r2 = np.sum(points**2, axis=1)
    I = ρ * (np.einsum('k,k->', weights, r2) * np.eye(3) - np.einsum('k,ki,kj->ij', weights, points, points))
    return m, I


def ellipsoid_volume_quadrature(a, b, c, n=24):
    x, w = np.polynomial.legendre.leggauss(n)
    ρ_, wρ = (x + 1) / 2, w / 2
    θ, wθ = (x + 1) * π/2, w * π/2
    φ = np.arange(2*n) * π / n
    P, T, F = np.meshgrid(ρ_, θ, φ, indexing='ij')
    W = np.einsum('i,j,k->ijk', wρ, wθ, np.full(2*n, π/n)) * a*b*c * P**2 * np.sin(T)
    pts = np.stack([a*P*np.sin(T)*np.cos(F), b*P*np.sin(T)*np.sin(F), c*P*np.cos(T)], -1).reshape(-1, 3)
    return pts, W.ravel()


def test_ellipsoid_volume_mass_inertia():
    e = Ellipsoid(50*nm, 80*nm, 130*nm, material=M.SILICON)
    pts, w = ellipsoid_volume_quadrature(e.a, e.b, e.c)
    m, I = volume_integrals(pts, w, e.ρ)
    assert np.isclose(e.V, w.sum(), rtol=1e-12)
    assert np.isclose(e.m, m, rtol=1e-12)
    assert np.allclose(e.inertia_tensor, I, rtol=1e-11, atol=1e-12 * np.max(I))


def test_cylinder_volume_area_inertia():
    c = Cylinder(length=400*nm, radius=50*nm)
    x, w = np.polynomial.legendre.leggauss(16)
    z, wz = x * c.length/2, w * c.length/2
    ρ_, wρ = (x + 1) * c.radius/2, w * c.radius/2
    φ = np.arange(32) * 2*π/32
    Z, P, F = np.meshgrid(z, ρ_, φ, indexing='ij')
    W = np.einsum('i,j,k->ijk', wz, wρ, np.full(32, 2*π/32)) * P
    pts = np.stack([P*np.cos(F), P*np.sin(F), Z], -1).reshape(-1, 3)
    _, I = volume_integrals(pts, W.ravel(), c.ρ)
    assert np.isclose(c.V, W.sum(), rtol=1e-12)
    assert np.allclose(c.inertia_tensor, I, rtol=1e-12, atol=1e-12 * np.max(I))
    assert np.isclose(c.A, cylinder_surface_quadrature(c.length, c.radius)[2].sum(), rtol=1e-13)
    assert isinstance(Rod(1*μm, 50*nm), Cylinder)
    disk = Disk(radius=200*nm, thickness=20*nm)
    assert disk.length == disk.thickness == 20*nm and disk.aspect_ratio == pytest.approx(0.05)


def test_dumbbell_inertia_parallel_axis():
    r = 50*nm
    d = Dumbbell(r)
    m_s = Sphere(r=r).m
    assert np.isclose(d.m, 2*m_s, rtol=1e-15)
    # two touching equal spheres: I_zz = 2·(2/5) m r², I_xx = I_zz + 2 m r²
    assert np.allclose(np.diag(d.inertia_tensor), [2.8*m_s*r**2, 2.8*m_s*r**2, 0.8*m_s*r**2], rtol=1e-14)
    u = Dumbbell(50*nm, 30*nm, separation=120*nm)
    (m1, m2), (z1, z2) = u.sphere_masses, u.sphere_positions
    assert np.isclose(m1*z1 + m2*z2, 0, atol=1e-30)
    μ = m1*m2/(m1 + m2)
    assert np.isclose(u.inertia_tensor[0, 0] - u.inertia_tensor[2, 2], μ * 120e-9**2, rtol=1e-13)
    with pytest.raises(ValueError):
        Dumbbell(50*nm, 30*nm, separation=70*nm)


# --------------------------------------------------------------------------------------------------
# the sphere limit is bit-identical to Sphere and Bead
# --------------------------------------------------------------------------------------------------
SPHERE_ATTRIBUTES = ['V', 'A', 'm', 'weight', 'inertia_tensor', 'α', 'n']
TRAP_ATTRIBUTES = ['α_eff', 'kx', 'ky', 'kz', 'Ωx', 'Ωy', 'Ωz', 'Γth', 'Γx', 'Γy', 'Γz', 'fth', 'Sth',
                   'σ_scattering', 'z_eq', 'x_zpf', 'U_depth', 'stiffness_tensor', 'field_noise_psd']


def sphere_like(r, **kwargs):
    return [Ellipsoid(r, r, r, **kwargs), Spheroid(r, r, **kwargs)]


@pytest.mark.parametrize('orientation', [(1, 0, 0, 0), tuple(random_quaternion())])
def test_sphere_limit_is_bit_identical_to_bead(orientation):
    bead = Bead()
    for shape in sphere_like(R0, orientation=orientation):
        for name in SPHERE_ATTRIBUTES:
            assert bitwise_equal(getattr(shape, name), getattr(bead, name)), name
        assert bitwise_equal(shape.polarisability_tensor(radiation_reaction=True),
                             bead.polarisability_tensor(radiation_reaction=True))
        assert bitwise_equal(shape.lab_polarisability_tensor(), bead.lab_polarisability_tensor())
        for args in [(GAS,), (GAS, 0.6, 1.3), (Gas(P=1e-3, T=77), 0.0, 1.0)]:
            assert bitwise_equal(shape.translational_damping_tensor(*args), bead.translational_damping_tensor(*args))
            assert bitwise_equal(shape.rotational_damping_tensor(*args[:2]), bead.rotational_damping_tensor(*args[:2]))
        t_shape, t_bead = Trap(particle=shape), Trap(bead=bead)
        for name in TRAP_ATTRIBUTES:
            assert bitwise_equal(getattr(t_shape, name), getattr(t_bead, name)), name
        assert t_shape.photon_recoil_heating_rates == t_bead.photon_recoil_heating_rates
        assert t_shape.Γj == t_bead.Γj


def test_sphere_limit_of_a_birefringent_material():
    sphere = Sphere(r=200*nm, material=M.VATERITE)
    for shape in sphere_like(200*nm, material=M.VATERITE):
        assert bitwise_equal(shape.static_polarisability_tensor(), sphere.static_polarisability_tensor())
        assert not shape.has_scalar_polarisability


def test_spheroid_approaches_the_sphere_continuously():
    sphere = Sphere(r=R0)
    near = Spheroid(R0 * (1 + 1e-7), R0)
    assert np.allclose(near.translational_damping_tensor(GAS), sphere.translational_damping_tensor(GAS), rtol=1e-6)
    assert np.allclose(near.rotational_damping_tensor(GAS), sphere.rotational_damping_tensor(GAS), rtol=1e-6)
    assert np.allclose(near.static_polarisability_tensor(), sphere.static_polarisability_tensor(), rtol=1e-6)


# --------------------------------------------------------------------------------------------------
# polarisability
# --------------------------------------------------------------------------------------------------
def test_spheroid_polarisability_components():
    sp = Spheroid(150*nm, 50*nm)
    εr = sp.εr
    L = sp.depolarisation_factors
    α = sp.static_polarisability_tensor()
    for i in range(3):
        assert np.isclose(α[i, i], ε0 * sp.V * (εr - 1) / (1 + L[i]*(εr - 1)), rtol=1e-15)
    assert α[2, 2] > α[0, 0] == α[1, 1]
    oblate = Spheroid(50*nm, 150*nm).static_polarisability_tensor()
    assert oblate[2, 2] < oblate[0, 0]


def test_birefringent_spheroid_aligned_axes():
    sp = Spheroid(150*nm, 100*nm, material=M.VATERITE)
    ε = np.diag(M.VATERITE.εr_tensor())
    L = sp.depolarisation_factors
    α = sp.static_polarisability_tensor()
    assert np.allclose(np.diag(α), ε0 * sp.V * (ε - 1) / (1 + np.asarray(L)*(ε - 1)), rtol=1e-14)
    assert np.allclose(α - np.diag(np.diag(α)), 0)
    # a general (rotated) permittivity tensor: matrix form, symmetric
    R = quaternion_to_rotation_matrix(random_quaternion())
    εR = R @ M.VATERITE.εr_tensor() @ R.T
    αR = ellipsoid_polarisability(sp.V, εR, L)
    assert np.allclose(αR, αR.T, rtol=1e-13, atol=1e-13 * np.abs(αR).max())


def test_radiation_reaction_tensor_obeys_the_optical_theorem():
    sp = Spheroid(150*nm, 50*nm, orientation=random_quaternion())
    k = 2*π / sp.λ
    α_eff = sp.lab_polarisability_tensor(radiation_reaction=True)
    for e in np.eye(3):
        assert np.isclose(np.imag(e @ α_eff @ e), k**3 / (6*π*ε0) * np.linalg.norm(α_eff @ e)**2, rtol=1e-10)
    assert np.allclose(sp.effective_polarisability_tensor(k), α_eff, rtol=1e-14)


def test_dumbbell_polarisability():
    r = 50*nm
    α_s = Sphere(r=r).α
    far = Dumbbell(r, separation=100*μm).static_polarisability_tensor()
    assert np.allclose(np.diag(far), 2*α_s, rtol=1e-9)
    d = Dumbbell(r)
    α = np.diag(d.static_polarisability_tensor())
    g = α_s / (4*π*ε0 * (2*r)**3)
    assert np.isclose(α[2], 2*α_s / (1 - 2*g), rtol=1e-13)
    assert np.isclose(α[0], 2*α_s / (1 + g), rtol=1e-13)
    assert α[2] > 2*α_s > α[0]
    assert np.isclose(sum(d.depolarisation_factors), 1.0, atol=0.05)


def test_cylinder_uses_the_equivalent_spheroid_factors():
    c = Cylinder(length=600*nm, radius=50*nm)
    assert c.depolarisation_factors == spheroid_depolarisation_factors(6.0)
    α = c.static_polarisability_tensor()
    assert np.isclose(α[2, 2], ε0 * c.V * (c.εr - 1) / (1 + c.depolarisation_factors[2]*(c.εr - 1)))


# --------------------------------------------------------------------------------------------------
# gas damping (Martinetz et al., Phys. Rev. E 97, 052112 (2018))
# --------------------------------------------------------------------------------------------------
ACCOMMODATIONS = [(1.0, 1.0), (0.6, 1.3), (0.0, 1.0), (1.0, 0.5)]


@pytest.mark.parametrize('αc, γs', ACCOMMODATIONS)
def test_numerical_sphere_drag_matches_table_I(αc, γs):
    s = Sphere(r=100*nm)
    Γ = friction_matrix(surface_integrals(*ellipsoid_surface_quadrature(s.r, s.r, s.r)), s.m, s.inertia_tensor, GAS, αc, γs)
    assert np.allclose(Γ[:3, :3], s.translational_damping_tensor(GAS, αc, γs), rtol=1e-13)
    assert np.allclose(Γ[3:, 3:], s.rotational_damping_tensor(GAS, αc), rtol=1e-12, atol=1e-12 * Γ[0, 0])
    # centrosymmetric: no rotranslational coupling (relative to Γ_cm/r)
    assert np.max(np.abs(Γ[:3, 3:])) < 1e-12 * Γ[0, 0] / s.r


@pytest.mark.parametrize('αc, γs', ACCOMMODATIONS)
@pytest.mark.parametrize('length, radius', [(400*nm, 50*nm), (20*nm, 200*nm), (100*nm, 50*nm)])
def test_cylinder_table_I_matches_the_surface_integral(length, radius, αc, γs):
    c = Cylinder(length, radius)
    numerical = c.rotranslational_friction_matrix(GAS, αc, γs)
    scale = np.max(np.abs(np.diag(numerical)))
    assert np.allclose(np.diag(numerical)[:3], np.diag(c.translational_damping_tensor(GAS, αc, γs)), rtol=1e-12)
    assert np.allclose(np.diag(numerical)[3:], np.diag(c.rotational_damping_tensor(GAS, αc, γs)),
                       rtol=1e-12, atol=1e-14 * scale)
    off = numerical - np.diag(np.diag(numerical))
    assert np.max(np.abs(off[:3, :3])) < 1e-13 * scale
    assert np.max(np.abs(off[3:, 3:])) < 1e-13 * scale


def test_diffusion_obeys_fluctuation_dissipation_at_gamma_s_1():
    for shape in [Spheroid(150*nm, 50*nm), Ellipsoid(40*nm, 60*nm, 90*nm), Cylinder(300*nm, 40*nm),
                  Dumbbell(50*nm, 30*nm)]:
        Γ = shape.rotranslational_friction_matrix(GAS, 0.8)
        D = shape.rotranslational_diffusion_matrix(GAS, 0.8)
        T = np.zeros((6, 6))
        T[:3, :3] = shape.m * np.eye(3)
        T[3:, 3:] = shape.inertia_tensor
        assert mixed_close(D, kB * GAS.T * Γ @ T, rtol=1e-12)
        assert np.allclose(shape.translational_diffusion_tensor(GAS, 0.8),
                           kB * GAS.T * shape.m * shape.translational_damping_tensor(GAS, 0.8), rtol=1e-12)
        # a hot surface (γs ≠ 1) breaks the relation
        assert not mixed_close(shape.rotranslational_diffusion_matrix(GAS, 0.8, 1.5),
                               kB * GAS.T * shape.rotranslational_friction_matrix(GAS, 0.8, 1.5) @ T, rtol=1e-3)


def test_spheroid_drag_properties():
    prolate = Spheroid(300*nm, 50*nm)
    Γ = prolate.translational_damping_tensor(GAS)
    assert Γ[0, 0] == pytest.approx(Γ[1, 1], rel=1e-12) and Γ[2, 2] < Γ[0, 0]
    # specular reflection exerts no torque about the symmetry axis of a body of revolution
    assert prolate.rotational_damping_tensor(GAS, accommodation=0.0)[2, 2] == pytest.approx(0, abs=1e-12)
    assert prolate.rotational_damping_tensor(GAS, accommodation=0.0)[0, 0] > 0
    # converged quadrature
    from levisim.physics.gas_damping import ellipsoid_surface_quadrature as q
    finer = friction_matrix(surface_integrals(*q(50*nm, 50*nm, 300*nm, order=128)), prolate.m, prolate.inertia_tensor, GAS)
    T = np.diag([prolate.m] * 3 + list(np.diag(prolate.inertia_tensor)))
    assert mixed_close(prolate.rotranslational_friction_matrix(GAS) @ T, finer @ T, rtol=1e-11)


def test_damping_scales_with_gas():
    sp = Spheroid(150*nm, 50*nm)
    other = Gas(P=GAS.P * 7, T=150.0)
    ratio = gas_prefactor(other) / gas_prefactor(GAS)
    assert np.allclose(sp.translational_damping_tensor(other), ratio * sp.translational_damping_tensor(GAS), rtol=1e-13)


def test_dumbbell_drag():
    r = 50*nm
    equal = Dumbbell(r)
    Γ = equal.rotranslational_friction_matrix(GAS)
    # translational drag is that of two independent spheres (shadowing is neglected)
    assert np.allclose(Γ[:3, :3], Sphere(r=r).translational_damping_tensor(GAS), rtol=1e-12)
    assert np.max(np.abs(Γ[:3, 3:])) < 1e-12 * Γ[0, 0] / r
    unequal = Dumbbell(50*nm, 30*nm).rotranslational_friction_matrix(GAS)
    assert np.max(np.abs(unequal[:3, 3:])) > 1e-3 * unequal[0, 0] / r
    integrals = Dumbbell(50*nm, 30*nm).gas_surface_integrals
    for G in integrals:
        assert np.allclose(G, G.T)


def test_diffusion_matrix_is_positive_semidefinite():
    D = diffusion_matrix(Dumbbell(50*nm, 30*nm).gas_surface_integrals, GAS, 0.7, 1.2)
    assert np.linalg.eigvalsh(D).min() > -1e-12 * np.abs(D).max()


# --------------------------------------------------------------------------------------------------
# translational trapping of anisotropic particles
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('axis', [(0, 1, 0), (1, 0, 0), (0, 0, 1), (1, 1, 1)])
def test_spheroid_trap_uses_the_projected_polarisability(axis):
    sp = Spheroid(120*nm, 60*nm, orientation=body_axis_to_quaternion(axis))
    trap = Trap(particle=sp)
    tw = trap.tweezer
    e = np.array([0.0, 1.0, 0.0])
    α_eff = e @ sp.lab_polarisability_tensor(radiation_reaction=False) @ e
    expected = np.complex128(e @ sp.effective_polarisability_tensor(tw.k) @ e)
    assert trap.α_eff == pytest.approx(expected, rel=1e-15)
    assert trap.α_eff.real == pytest.approx(α_eff, rel=1e-3)
    for j, w in zip('xyz', (tw.wx0, tw.wy0, tw.wz0)):
        assert trap.Ωj[j] == pytest.approx(np.sqrt(trap.α_eff.real * tw.E0**2 / w**2 / sp.m), rel=1e-14)
    # the closed-form stiffness at the focus equals the finite-difference Hessian of the force
    numerical = super(type(tw), tw).stiffness(np.array([1e-12, 0, 0]), sp)
    assert np.allclose(np.diag(numerical), np.diag(tw.stiffness(None, sp)), rtol=1e-4)


def test_spheroid_aligned_with_the_polarisation_is_trapped_more_stiffly():
    along = Trap(particle=Spheroid(120*nm, 60*nm, orientation=body_axis_to_quaternion((0, 1, 0))))
    across = Trap(particle=Spheroid(120*nm, 60*nm, orientation=body_axis_to_quaternion((1, 0, 0))))
    ratio = along.Ωx / across.Ωx
    α = Spheroid(120*nm, 60*nm).static_polarisability_tensor()
    assert ratio == pytest.approx(np.sqrt(α[2, 2] / α[0, 0]), rel=1e-3)
    assert ratio > 1


def test_recoil_coefficients_generalise():
    tw = Trap().tweezer
    along = Spheroid(120*nm, 60*nm, orientation=body_axis_to_quaternion((0, 1, 0)))
    _, c = tw.recoil_coefficients(along)
    assert np.allclose(c, [2, 1, 7], atol=1e-12)
    tilted = Spheroid(120*nm, 60*nm, orientation=body_axis_to_quaternion((1, 1, 0)))
    _, c = tw.recoil_coefficients(tilted)
    assert np.isclose(c.sum(), 10) and c[0] < 2 and c[1] > 1
    assert np.allclose(np.diag(tw.noise(tilted)) / np.diag(tw.noise(tilted))[2], c / c[2])


def test_tilted_spheroid_gas_damping_per_axis():
    sp = Spheroid(300*nm, 50*nm, orientation=body_axis_to_quaternion((1, 0, 0)))
    trap = Trap(particle=sp)
    body = sp.translational_damping_tensor(trap.gas)
    assert trap.Γth_j['x'] == pytest.approx(body[2, 2], rel=1e-14)
    assert trap.Γth_j['y'] == pytest.approx(body[0, 0], rel=1e-14)
    assert trap.Γth == pytest.approx(np.trace(body) / 3, rel=1e-14)
    assert trap.Sth_j['x'] == pytest.approx(2 * sp.m * kB * trap.gas.T * trap.Γth_j['x'], rel=1e-14)


def test_anisotropic_particle_in_multitrap_and_simulation():
    from levisim import MultiTrap
    from levisim.simulation import FreeSpaceSimulation
    sp = Spheroid(100*nm, 50*nm, orientation=body_axis_to_quaternion((1, 0, 0)))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        mtrap = MultiTrap(num_traps=1, Cartesian_axes_per_trap=3, beads=[sp])
    trap = mtrap.traps[0]
    for label in mtrap.labels:
        assert mtrap.dimensions[label].Γ == trap.Γth_j[label[0]]
    sim = FreeSpaceSimulation(trap=Trap(particle=sp), num_samples=200, seed=1, backend='numpy')
    S = sim.noise_spectral_densities()
    t = sim.trap
    assert np.array_equal(S[:3], [t.Sth_j['x'], t.Sth_j['y'], t.Sth_j['z']])
    assert S[0] != S[1]


def test_birefringent_sphere_in_a_trap_is_scalar():
    trap = Trap(particle=Sphere(r=100*nm, material=M.VATERITE))
    assert np.ndim(trap.α_eff) == 0 and np.ndim(trap.kx) == 0
    no2 = M.VATERITE.εr_tensor()[0, 0]
    # optic axis (body z) is along the lab z: the y-polarised light sees n_o
    assert trap.α_eff.real == pytest.approx(Sphere(r=100*nm, material=M.VATERITE.replace(name='o', permittivity=no2)).α, rel=1e-3)


# --------------------------------------------------------------------------------------------------
# TabulatedParticle
# --------------------------------------------------------------------------------------------------
def test_tabulated_round_trip_of_a_spheroid():
    sp = Spheroid(150*nm, 50*nm, orientation=body_axis_to_quaternion((1, 1, 0)), charge=3)
    tab = TabulatedParticle.from_particle(sp, gas=GAS)
    assert tab.m == sp.m and tab.V == sp.V and tab.A == sp.A and tab.charge == 3
    assert np.array_equal(tab.inertia_tensor, sp.inertia_tensor)
    assert np.array_equal(tab.static_polarisability_tensor(), sp.static_polarisability_tensor())
    assert np.array_equal(tab.translational_damping_tensor(GAS), sp.translational_damping_tensor(GAS))
    other = Gas(P=3e-3, T=200)
    assert np.allclose(tab.rotational_damping_tensor(other), sp.rotational_damping_tensor(other), rtol=1e-13)
    t_tab, t_sp = Trap(particle=tab), Trap(particle=sp)
    for name in ['α_eff', 'kx', 'Ωx', 'Ωy', 'Ωz', 'Γx', 'Γy', 'Γz']:
        assert np.array_equal(getattr(t_tab, name), getattr(t_sp, name)), name
    again = TabulatedParticle(**tab.to_dict())
    for name in ['m', 'V', 'A', 'inertia_tensor', 'α', 'orientation']:
        assert np.array_equal(getattr(again, name), getattr(tab, name)), name
    assert np.array_equal(again.translational_damping_tensor(other), tab.translational_damping_tensor(other))
    with pytest.raises(ValueError):
        tab.translational_damping_tensor(GAS, accommodation=0.5)


def test_tabulated_wavelength_table():
    s = Sphere(r=100*nm, material=M.FUSED_SILICA)
    λs = np.linspace(800, 1600, 9) * nm
    tab = TabulatedParticle.from_particle(s, wavelengths=λs)
    for λ in λs:
        assert np.array_equal(tab.static_polarisability_tensor(λ), s.static_polarisability_tensor(λ))
    λ = 1064*nm
    assert np.allclose(tab.static_polarisability_tensor(λ), s.static_polarisability_tensor(λ), rtol=1e-4)
    assert tab.wavelength_range == (λs[0], λs[-1])
    with pytest.raises(ValueError, match='outside'):
        tab.static_polarisability_tensor(2000*nm)


def test_tabulated_callable_and_constant_inputs():
    α0 = Sphere().α
    tab = TabulatedParticle(mass=1e-18, inertia=[1, 1, 1.5], polarisability=lambda λ: α0 * (1064*nm / λ))
    assert tab.static_polarisability_tensor(532*nm)[0, 0] == pytest.approx(2*α0)
    constant = TabulatedParticle(mass=1e-18, inertia=1.0, polarisability=α0)
    assert np.array_equal(constant.static_polarisability_tensor(), α0 * np.eye(3))
    with pytest.raises(ValueError, match='no volume'):
        _ = constant.V


@pytest.mark.parametrize('kwargs, message', [
    ({'mass': -1}, 'mass'),
    ({'inertia': [[1, 0.5, 0], [0, 1, 0], [0, 0, 1]]}, 'symmetric'),
    ({'inertia': [1, 1, -1]}, 'positive definite'),
    ({'inertia': [1, 1, 3]}, 'triangle'),
    ({'polarisability': np.array([[1, 0.3, 0], [0, 1, 0], [0, 0, 1]]) * 1e-32}, 'symmetric'),
    ({'polarisability': np.diag([1, 1, 1 - 0.1j]) * 1e-32}, 'passive'),
    ({'polarisability': np.ones((2, 2))}, '3×3'),
    ({'polarisability': {'λ': [1e-6, 0.9e-6], 'α': np.zeros((2, 3, 3))}}, 'increasing'),
    ({'translational_damping': np.eye(3)}, 'damping_gas'),
    ({'translational_damping': -np.eye(3), 'damping_gas': GAS}, 'non-negative'),
    ({'volume': 0.0}, 'volume'),
])
def test_tabulated_validation(kwargs, message):
    arguments = {'mass': 1e-18, 'inertia': [1e-33, 1e-33, 1.5e-33], 'polarisability': 1e-32}
    arguments.update(kwargs)
    with pytest.raises(ValueError, match=message):
        TabulatedParticle(**arguments)


def test_info_strings():
    for particle in [Spheroid(100*nm, 50*nm), Ellipsoid(1e-7, 2e-7, 3e-7), Cylinder(4e-7, 5e-8), Disk(2e-7, 2e-8),
                     Dumbbell(5e-8, 3e-8), TabulatedParticle.from_particle(Spheroid(100*nm, 50*nm))]:
        text = particle.info_str
        assert type(particle).__name__ in text and 'mass' in text

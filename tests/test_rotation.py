"""
Rigid-body rotation: kinematics, torques, recoil torque noise, RigidBodySimulation and the
librational dimensions of MultiTrap (levisim.physics.rotation, levisim.simulation.rigid_body)
"""

import numpy as np
import pytest
from scipy.linalg import expm

from levisim.core.constants import kB, ħ, ε0, π
from levisim.core.logger import logger
from levisim.core.units import degrees, kHz, mbar, nm
from levisim.multiphysics.enums.quantum_linear_theory_method import (
    QuantumLinearTheoryMethod as Q,
)
from levisim.multiphysics.multitrap import MultiTrap
from levisim.physics import (
    Bead,
    Ellipsoid,
    Gas,
    HarmonicPotential,
    Spheroid,
    Trap,
    Tweezer,
    UniformElectricField,
)
from levisim.physics import coherent_scattering as cs
from levisim.physics.particle import (
    axis_angle_quaternion,
    body_axis_to_quaternion,
    quaternion_to_rotation_matrix,
)
from levisim.physics.rotation import (
    GENERATORS,
    quaternion_from_rotation_vector,
    quaternion_multiply,
    recoil_torque_noise,
    rotated_tensor_derivatives,
    rotation_vector_between,
)
from levisim.simulation.rigid_body import RigidBodySimulation

logger.remove()


def rotate(T, θ):
    R = expm(np.einsum('j,jab->ab', θ, GENERATORS))
    return R @ T @ R.T


def aligned_spheroid(**kwargs):
    """A prolate spheroid with its symmetry axis along the tweezer polarisation y"""
    return Spheroid(a=70*nm, b=50*nm, orientation=body_axis_to_quaternion((0, 1, 0)), **kwargs)


# --------------------------------------------------------------------------------------------------
# kinematics and tensor expansions
# --------------------------------------------------------------------------------------------------
def test_generators_are_cross_products():
    v = np.array([0.3, -1.2, 0.7])
    for j in range(3):
        assert np.allclose(GENERATORS[j] @ v, np.cross(np.eye(3)[j], v))


def test_rotated_tensor_derivatives_are_the_taylor_expansion():
    rng = np.random.default_rng(1)
    T = rng.normal(size=(3, 3))
    dT, d2T = rotated_tensor_derivatives(T)
    errors = []
    for scale in (1e-2, 5e-3):
        θ = scale * np.array([1.0, -2.0, 1.5])
        approx = T + np.einsum('j,jab->ab', θ, dT) + 0.5 * np.einsum('i,j,ijab->ab', θ, θ, d2T)
        errors.append(np.max(np.abs(rotate(T, θ) - approx)))
    # third order: halving θ divides the error by 8
    assert 6 < errors[0] / errors[1] < 10
    assert np.allclose(d2T, d2T.transpose(1, 0, 2, 3))


def test_quaternion_helpers_round_trip():
    q0 = axis_angle_quaternion((1, 2, 3), 0.8)
    θ = np.array([1e-3, -2e-3, 4e-4])
    q = quaternion_multiply(quaternion_from_rotation_vector(θ), q0)
    assert np.allclose(rotation_vector_between(q0, q), θ, atol=1e-15)
    R = expm(np.einsum('j,jab->ab', θ, GENERATORS))
    assert np.allclose(quaternion_to_rotation_matrix(q), R @ quaternion_to_rotation_matrix(q0))


# --------------------------------------------------------------------------------------------------
# torques
# --------------------------------------------------------------------------------------------------
def test_aligned_particle_has_no_torque_and_the_analytic_stiffness():
    particle = aligned_spheroid()
    tweezer = Tweezer()
    assert np.allclose(tweezer.torque(np.zeros(3), 0.0, particle), 0, atol=1e-30)
    α = np.real(particle.body_effective_polarisability_tensor(tweezer.k))
    α_perp, α_par = α[0, 0], α[2, 2]  # body z is the symmetry axis
    E0 = tweezer.E0
    K = tweezer.torque_stiffness(particle)
    expected = 0.5 * E0**2 * (α_par - α_perp)
    # the recoil of the radiated angular momentum changes it at O(k³α) ~ 1e-4
    assert np.allclose(np.diag(K), [expected, 0, expected], rtol=1e-3)
    trap = Trap(tweezer=tweezer, particle=particle)
    assert trap.Ωθ['α'] == pytest.approx(np.sqrt(K[0, 0] / trap.Ij['α']), rel=1e-12)
    assert trap.Ωθ['β'] == 0.0


@pytest.mark.parametrize('ψ', [0.0, 0.3, π/4])
def test_torque_stiffness_is_the_derivative_of_the_torque(ψ):
    particle = Ellipsoid(80*nm, 60*nm, 45*nm, orientation=axis_angle_quaternion((1, -1, 2), 0.5))
    tweezer = Tweezer(polarisation=Tweezer.elliptical_polarisation(ψ, ζ=0.2))
    K = tweezer.torque_stiffness(particle)
    h = 1e-6
    q0 = particle.orientation
    K_fd = np.zeros((3, 3))
    for j in range(3):
        step = np.zeros(3)
        step[j] = h
        τp = tweezer.torque(np.zeros(3), 0, particle, orientation=quaternion_multiply(quaternion_from_rotation_vector(step), q0))
        τm = tweezer.torque(np.zeros(3), 0, particle, orientation=quaternion_multiply(quaternion_from_rotation_vector(-step), q0))
        K_fd[:, j] = -(τp - τm) / (2 * h)
    assert np.allclose(K, K_fd, rtol=1e-6, atol=1e-8 * np.max(np.abs(K)))


def test_spin_torque_on_a_sphere_is_the_absorbed_power_over_ω():
    from levisim.physics import Material, Sphere
    lossless = Bead()
    absorbing = Sphere(r=60*nm, material=Material('lossy glass', 2200.0, permittivity=2.1 + 0.02j))
    for ψ in (π/4, 0.2, -0.3):
        tweezer = Tweezer(polarisation=Tweezer.elliptical_polarisation(ψ))
        k, E0 = tweezer.k, tweezer.E0
        τ0 = tweezer.torque(np.zeros(3), 0.0, lossless)
        α_eff = lossless.α / (1 - 1j * k**3 / (6 * π * ε0) * lossless.α)
        # the extinction torque alone would be ½ Im α_eff E0² sin 2ψ; the radiated angular momentum cancels it
        assert np.max(np.abs(τ0)) < 1e-10 * 0.5 * np.imag(α_eff) * E0**2
        τ = tweezer.torque(np.zeros(3), 0.0, absorbing)
        α = absorbing.α
        α_eff = α / (1 - 1j * k**3 / (6 * π * ε0) * α)
        absorbed = 0.5 * (np.imag(α_eff) - k**3 * abs(α_eff)**2 / (6 * π * ε0)) * E0**2  # P_abs/ω
        assert np.allclose(τ, [0, 0, absorbed * np.sin(2 * ψ)], rtol=1e-10, atol=0)


def test_spin_torque_on_a_lossless_anisotropic_particle_in_circular_light():
    tweezer = Tweezer(polarisation=Tweezer.elliptical_polarisation(π/4))
    k, E0 = tweezer.k, tweezer.E0
    for angle in (0.0, 0.4, 1.3):
        q = quaternion_multiply(axis_angle_quaternion('z', angle), body_axis_to_quaternion((1, 0, 0)))
        particle = Spheroid(a=70*nm, b=50*nm, orientation=q)  # symmetry axis in the focal plane
        α = np.real(np.diag(particle.body_effective_polarisability_tensor(k)))
        τ = tweezer.torque(np.zeros(3), 0.0, particle)
        expected = k**3 * E0**2 * (α[2] - α[0])**2 / (24 * π * ε0)
        assert τ[2] == pytest.approx(expected, rel=2e-3)
        assert np.allclose(τ[:2], 0, atol=1e-6 * expected)


def test_default_and_explicit_y_polarisation_agree():
    for particle in (Bead(), aligned_spheroid(), Ellipsoid(80*nm, 60*nm, 45*nm, orientation=axis_angle_quaternion((1, 2, 0), 0.3))):
        a = Trap(tweezer=Tweezer(), particle=particle)
        b = Trap(tweezer=Tweezer(polarisation=(0, 1)), particle=particle)
        for axis in 'xyz':
            assert b.Ωj[axis] == pytest.approx(a.Ωj[axis], rel=1e-13)
            assert b.photon_recoil_heating_rates[axis] == pytest.approx(a.photon_recoil_heating_rates[axis], rel=1e-12)
        assert np.allclose(b.torque_stiffness_tensor, a.torque_stiffness_tensor, rtol=1e-12, atol=1e-40)


def test_circular_polarisation_recoil_weights():
    tweezer = Tweezer(polarisation=Tweezer.elliptical_polarisation(π/4))
    _, c = tweezer.recoil_coefficients(Bead())
    assert np.allclose(c, [1.5, 1.5, 7.0])


# --------------------------------------------------------------------------------------------------
# recoil torque noise
# --------------------------------------------------------------------------------------------------
def test_recoil_torque_noise_principal_frame_and_rudolph_weights():
    α = np.diag([1.0, 2.0, 3.5]) * 1e-32
    k, E0 = 2 * π / 1064e-9, 1e7
    prefactor = ħ * k**3 * E0**2 / (12 * π * ε0)
    for ψ in (0.0, 0.25, π/4):
        # Rudolph et al. (2021): polarisation cos ψ ê_c + i sin ψ ê_b with axes (a, b, c) = (x, y, z)
        E = E0 * np.array([0, 1j * np.sin(ψ), np.cos(ψ)])
        S = recoil_torque_noise(α, E, k)
        a, b, c = np.diag(α)
        expected = prefactor * np.array([(c - b)**2, (c - a)**2 * np.cos(ψ)**2, (b - a)**2 * np.sin(ψ)**2])
        assert np.allclose(np.diag(S), expected, rtol=1e-12, atol=1e-12 * expected.max())
    assert np.allclose(recoil_torque_noise(2e-32 * np.eye(3), E0 * np.array([0, 1, 0]), k), 0)
    S = Tweezer().recoil_torque_noise(Bead())
    assert np.all(S == 0)


def test_libration_recoil_heating_rate():
    trap = Trap(tweezer=Tweezer(), particle=aligned_spheroid())
    S = np.diag(trap.torque_noise_psd)
    for i, axis in enumerate(('α', 'γ')):
        j = (0, 2)[i]
        assert trap.libration_recoil_heating_rates[axis] == pytest.approx(S[j] / (2 * trap.Ij[axis] * ħ * trap.Ωθ[axis]))
    assert trap.libration_recoil_heating_rates['β'] == 0.0


# --------------------------------------------------------------------------------------------------
# electric torques
# --------------------------------------------------------------------------------------------------
def test_permanent_dipole_torque_and_libration_frequency():
    p0 = 1e-24  # C·m
    E = 1e4  # V/m
    particle = Bead(r=50*nm)
    particle.dipole_moment = (0, 0, p0)
    field = UniformElectricField((0, 0, E))
    tilted = axis_angle_quaternion('x', 0.01)
    τ = field.torque(np.zeros(3), 0.0, particle, orientation=tilted)
    assert τ == pytest.approx([-p0 * E * np.sin(0.01), 0, 0], rel=1e-12)
    K = field.torque_stiffness(particle)
    assert np.allclose(K, np.diag([p0 * E, p0 * E, 0]), rtol=1e-12, atol=1e-40)
    trap = Trap(fields=[field], particle=particle)
    assert trap.Ωθ['α'] == pytest.approx(np.sqrt(p0 * E / trap.Ij['α']))


# --------------------------------------------------------------------------------------------------
# rigid-body time domain
# --------------------------------------------------------------------------------------------------
def free_trap(particle):
    return Trap(fields=[HarmonicPotential(np.eye(3) * 1e-6)], particle=particle)


def test_torque_free_symmetric_top_precesses_analytically():
    particle = Spheroid(a=100*nm, b=50*nm)
    I1, _, I3 = np.diag(particle.inertia_tensor)
    ω0 = np.array([1e5, 0, 3e5])
    sim = RigidBodySimulation(trap=free_trap(particle), num_samples=5000, Δt=1e-8, ω0=ω0, gas_damping=False, seed=1)
    ω = sim.angular_velocities_body()
    Ωp = (I3 - I1) / I1 * ω0[2]
    t = sim.time_array
    assert np.max(np.abs(ω[:, 0] - ω0[0] * np.cos(Ωp * t))) < 1e-6 * ω0[0]
    assert np.max(np.abs(ω[:, 1] - ω0[0] * np.sin(Ωp * t))) < 1e-6 * ω0[0]
    assert np.max(np.abs(ω[:, 2] - ω0[2])) < 1e-9 * ω0[2]
    assert sim.quaternion_norm_error() < 1e-14
    E = sim.rotational_kinetic_energies()
    assert np.ptp(E) / E[0] < 1e-7
    L = sim.angular_momenta_lab()
    assert np.max(np.ptp(L, axis=0)) / np.linalg.norm(L[0]) < 1e-7


def test_intermediate_axis_instability():
    particle = Ellipsoid(90*nm, 60*nm, 40*nm)
    I = np.diag(particle.inertia_tensor)
    order = np.argsort(I)  # minor, intermediate, major axes of inertia
    flips = {}
    for name, axis in zip(('minor', 'intermediate', 'major'), order):
        ω0 = np.full(3, 1e3)
        ω0[axis] = 1e6
        sim = RigidBodySimulation(trap=free_trap(particle), num_samples=8000, Δt=2e-8, ω0=ω0, gas_damping=False)
        ω = sim.angular_velocities_body()
        flips[name] = np.min(ω[:, axis]) / ω0[axis]
        assert sim.quaternion_norm_error() < 1e-14
        E = sim.rotational_kinetic_energies()
        assert np.ptp(E) / E[0] < 1e-6
    assert flips['minor'] > 0.99 and flips['major'] > 0.99
    assert flips['intermediate'] < -0.5  # the spin about the intermediate axis turns over


def test_libration_frequency_and_equipartition_in_time_domain():
    particle = aligned_spheroid()
    trap = Trap(tweezer=Tweezer(), particle=particle, gas=Gas(P=10 * mbar, T=3.0))
    Ω = trap.Ωθ['γ']
    # deterministic: a small libration about z rings at Ωγ (no gas)
    sim = RigidBodySimulation(trap=trap, num_samples=4000, Δt=2e-9, ω0=(0, 0, 1e3), gas_damping=False)
    γ = sim.angles[:, 2]
    crossings = np.where(np.diff(np.sign(γ)) != 0)[0]
    period = 2 * np.mean(np.diff(crossings)) * sim.Δt
    assert 2 * π / period == pytest.approx(Ω, rel=2e-3)
    # thermal: 〈θ²〉 = kB T/K for the confined librations
    sim = RigidBodySimulation(trap=trap, num_samples=60000, Δt=5e-9, seed=4)
    K = np.diag(trap.torque_stiffness_tensor)
    # the tilt of the symmetry axis (body z, along y at rest) towards x is the libration γ; the
    # rotation about the symmetry axis itself diffuses freely
    n = sim.lab_directions((0, 0, 1))
    assert np.var(-n[10000:, 0]) == pytest.approx(kB * trap.gas.T / K[2], rel=0.25)
    assert np.var(n[10000:, 2]) == pytest.approx(kB * trap.gas.T / K[0], rel=0.25)


# --------------------------------------------------------------------------------------------------
# librational dimensions of MultiTrap
# --------------------------------------------------------------------------------------------------
ω_range = np.linspace(-1000, 1000, 401) * kHz


def libration_multitrap(Φ, axes=('x', 'y', 'z', 'α', 'γ'), **kwargs):
    particle = aligned_spheroid(Φ=Φ)
    return MultiTrap(num_traps=1, Cartesian_axes_per_trap=axes, beads=[particle], tweezers=[Tweezer(θ=60*degrees)],
                     ω_range=ω_range, **kwargs)


def test_axis_tuple_labels_and_integer_form():
    mt = libration_multitrap(π/4)
    assert mt.labels == ['x1', 'y1', 'z1', 'α1', 'γ1']
    assert mt.num_dimensions == 5
    a = MultiTrap(num_traps=2, Cartesian_axes_per_trap=('x', 'y'), ω_range=ω_range)
    b = MultiTrap(num_traps=2, Cartesian_axes_per_trap=2, ω_range=ω_range)
    assert a.labels == b.labels
    assert np.array_equal(a.drift_matrix, b.drift_matrix)


def test_free_rotation_about_the_polarisation_is_rejected():
    with pytest.raises(ValueError, match='no restoring torque'):
        libration_multitrap(π/4, axes=('x', 'β'))


def test_librations_couple_at_the_antinode_and_translations_at_the_node():
    node = libration_multitrap(π/2)
    antinode = libration_multitrap(0.0)
    assert abs(node.dimensions['γ1'].g) < 1e-12 * abs(antinode.dimensions['γ1'].g) and abs(node.dimensions['x1'].g) > 0
    assert antinode.dimensions['x1'].g == 0 and abs(antinode.dimensions['γ1'].g) > 0
    # the rotation about x needs the cavity mode polarised along z, which levisim does not have
    assert abs(antinode.dimensions['α1'].g) < 1e-9 * abs(antinode.dimensions['γ1'].g)


def test_light_couplings_against_rudolph_2021():
    # Rudolph et al., PRA 103, 043514 (2021), Eqs. (D5a), (D5d) with their ζ = -π/2 (polarisation
    # along levisim's y) and ψ = 0: G1α/G1x = (χc - χb) cos φ sin(θ - ζ) / (k sin θ sin φ χc cos(θ - ζ))
    Φ = π/3
    mt = libration_multitrap(Φ)
    trap = mt.traps[0]
    θ = trap.tweezer.θ
    α = np.real(np.diag(cs._static_lab_polarisability(trap.bead)))
    χc, χb = α[1], α[0]
    ζ = -π/2
    ratio = (χc - χb) * np.cos(Φ) * np.sin(θ - ζ) / (trap.tweezer.k * np.sin(θ) * np.sin(Φ) * χc * np.cos(θ - ζ))
    x, γ = mt.dimensions['x1'], mt.dimensions['γ1']
    assert (γ.g / γ.zpf) / (x.g / x.zpf) == pytest.approx(ratio, rel=1e-10)


def test_generic_expansion_reproduces_the_translational_couplings():
    mt = MultiTrap(num_traps=1, Cartesian_axes_per_trap=3, beads=[aligned_spheroid(Φ=0.7)], tweezers=[Tweezer(θ=50*degrees)],
                   ω_range=ω_range)
    trap = mt.traps[0]
    s0 = cs.drive_polarisability(trap) * np.sin(trap.drive_tweezer.θ)
    k, zR, Φ = trap.drive_tweezer.k, trap.drive_tweezer.zR, trap.bead.Φ
    grad_F = -k * np.sin(Φ) * cs.cavity_axis(trap.drive_tweezer.θ) + 0j
    grad_F[2] += 1j * (k - 1 / zR) * np.cos(Φ)
    for j, axis in enumerate('xyz'):
        dim = mt.dimensions[f'{axis}1']
        G = trap.cavity_drive_field * mt.cavity.E0 / (2 * ħ) * grad_F[j] * s0 * dim.zpf
        assert dim.G == pytest.approx(G, rel=1e-12)


def test_tilted_particle_drive_uses_cross_polarised_scattering():
    q = axis_angle_quaternion('z', 0.3)
    particle = Spheroid(a=70*nm, b=50*nm, orientation=quaternion_multiply(q, body_axis_to_quaternion((0, 1, 0))))
    mt = MultiTrap(num_traps=1, Cartesian_axes_per_trap=2, beads=[particle], tweezers=[Tweezer(θ=60*degrees)], ω_range=ω_range,
                   photon_recoil=False)
    trap = mt.traps[0]
    α_lab = cs._static_lab_polarisability(particle)
    e_c = cs.cavity_polarisation(trap.tweezer.θ)
    assert cs.drive_polarisability(trap) * np.sin(trap.tweezer.θ) == pytest.approx(e_c @ α_lab @ np.array([0, 1, 0]), rel=1e-12, abs=0)
    assert cs.drive_polarisability(trap) != pytest.approx(particle.α, rel=1e-3, abs=0)
    assert cs.drive_polarisability(Trap(tweezer=Tweezer(), particle=Bead())) == Bead().α


def test_static_libration_terms_against_finite_differences():
    """The θθ and rθ static stiffness of the coherent-scattering and cavity terms, from finite
    differences of the full interference potential with the rotated polarisability tensor"""
    mt = libration_multitrap(0.6)
    trap = mt.traps[0]
    α_c = (mt.cavity.A0 + 1j * mt.cavity.P0) / 2
    E0c, Ed = mt.cavity.E0, trap.cavity_drive_field
    drive = trap.drive_tweezer
    α_lab = cs._static_lab_polarisability(trap.bead)
    e_c = cs.cavity_polarisation(drive.θ)
    ê = np.array([0, 1, 0])
    d = cs.cavity_axis(drive.θ)

    def U(v):
        r, θ = v[:3], v[3:]
        α = rotate(α_lab, θ)
        u = drive.k * (d @ r) + trap.bead.Φ
        f = drive.E_field_amplitude(*r) / drive.E0
        ψ = drive.k * r[2] - np.arctan(r[2] / drive.zR)
        E_tw = Ed * f * np.exp(1j * ψ) * ê
        E_c = 2 * E0c * α_c * np.cos(u) * e_c
        return -0.5 * np.real(np.conj(E_c) @ α @ E_tw) - 0.25 * np.real(np.conj(E_c) @ α @ E_c)

    scale = np.array([1e-9] * 3 + [1e-5] * 3)
    H = np.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            ei, ej = np.eye(6)[i] * scale[i], np.eye(6)[j] * scale[j]
            H[i, j] = (U(ei + ej) - U(ei - ej) - U(-ei + ej) + U(-ei - ej)) / (4 * scale[i] * scale[j])
    exp = cs.librational_expansion(trap, E0c, α_c)
    K_θθ = exp['K_θθ_CS'] + exp['K_θθ_cav']
    K_rθ = exp['K_rθ_CS'] + exp['K_rθ_cav']
    assert np.allclose(H[3:, 3:], K_θθ, rtol=1e-5, atol=1e-6 * np.max(np.abs(K_θθ)))
    assert np.allclose(H[:3, 3:], K_rθ, rtol=1e-5, atol=1e-6 * np.max(np.abs(K_rθ)))


def test_librational_frequencies_and_damping():
    mt = libration_multitrap(π/2, static_cavity_shifts=False)
    trap = mt.traps[0]
    for axis in ('α', 'γ'):
        dim = mt.dimensions[f'{axis}1']
        assert dim.Ω == pytest.approx(trap.Ωθ[axis], rel=1e-12)
        assert dim.Γ == trap.Γrot_j[axis]
        assert dim.zpf == pytest.approx(np.sqrt(ħ / (2 * trap.Ij[axis] * dim.Ω)))
        assert dim.m == trap.Ij[axis]


def test_all_methods_agree_with_librations():
    mt = libration_multitrap(π/4)
    reference = mt.homodyne_spectrum(Q.LINEAR_AMPLIFIER_MODEL)
    for method in (Q.HYBRID_LINEAR_AMPLIFIER_MODEL, Q.NUMERICAL_HYBRIDISATION, Q.ANALYTIC_INSIDE_CAVITY, Q.ANALYTIC_CAVITY_OUTPUT):
        assert np.allclose(mt.homodyne_spectrum(method), reference, rtol=1e-8)
    for label in ('α1', 'γ1'):
        spectrum = mt.position_power_spectrum(label, Q.LINEAR_AMPLIFIER_MODEL)
        assert np.all(np.isfinite(spectrum)) and np.max(spectrum) > 0


def test_free_space_jacobian_matches_trap_libration_frequencies():
    """Jacobian of the rigid-body drift at the aligned equilibrium: the libration eigenfrequencies
    are Trap.Ωθ (the linear model of MultiTrap without a cavity)"""
    particle = aligned_spheroid()
    trap = Trap(tweezer=Tweezer(), particle=particle)
    sim = RigidBodySimulation(trap=trap, num_samples=1, Δt=1e-9, gas_damping=False)
    X0 = sim.initial_state()
    # coordinates: small rotation vector θ (lab) and the lab angular momentum
    def f(v):
        θ, L_lab = v[:3], v[3:]
        q = quaternion_multiply(quaternion_from_rotation_vector(θ), X0[6:10])
        R = quaternion_to_rotation_matrix(q)
        X = X0.copy()
        X[6:10] = q
        X[10:13] = R.T @ L_lab
        dX = sim.drift(0.0, X)
        ω_lab = R @ (sim.inertia_inverse @ X[10:13])
        return np.concatenate([ω_lab, R @ dX[10:13]])  # dθ/dt ≈ ω_lab, dL_lab/dt = R τ_b (no ω × L at rest)
    h = np.concatenate([np.full(3, 1e-7), np.full(3, 1e-30)])
    J = np.zeros((6, 6))
    for j in range(6):
        e = np.eye(6)[j] * h[j]
        J[:, j] = (f(e) - f(-e)) / (2 * h[j])
    eigenvalues = np.linalg.eigvals(J)
    frequencies = np.sort(np.abs(eigenvalues.imag))[::2]
    expected = np.sort([trap.Ωθ['α'], trap.Ωθ['β'], trap.Ωθ['γ']])
    assert np.allclose(frequencies, expected, rtol=1e-6, atol=1.0)


# --------------------------------------------------------------------------------------------------
# built-in controls and the compiled loop
# --------------------------------------------------------------------------------------------------
def controlled_simulation(**kwargs):
    trap = Trap(tweezer=Tweezer(polarisation=Tweezer.elliptical_polarisation(0.2)), particle=aligned_spheroid(), gas=Gas(P=1 * mbar))
    common = {'trap': trap, 'num_samples': 1500, 'Δt': 5e-9, 'seed': 3, 'recoil': True,
                  'parametric_feedback': {'gain': 1e-7, 'lab_axis': (1, 0, 0)},
                  'torque_drive': {'amplitude': (0, 0, 1e-22), 'ω': 2 * π * 1e5}, 'angular_damping': (1e3, 0, 0),
                  'ω0': (1e4, 2e4, 3e4)}
    common.update(kwargs)
    return RigidBodySimulation(**common)


@pytest.mark.parametrize('method', ['vector_4', 'vector_2'])
def test_numba_rigid_body_matches_numpy(method):
    pytest.importorskip('numba')
    from levisim.simulation.backends import numba_available
    if not numba_available():
        pytest.skip('numba disabled')
    a = controlled_simulation(backend='numpy', RK_method=method)
    b = controlled_simulation(backend='numba', RK_method=method)
    assert b.backend_used == 'numba'
    scale = np.abs(a.X).max(axis=0) + 1e-300
    assert np.max(np.abs(a.X - b.X) / scale) < 1e-12
    assert b.quaternion_norm_error() < 1e-14


def test_callables_run_the_numpy_loop():
    sim = controlled_simulation(backend='auto', num_samples=50, external_torque=lambda t, X: np.zeros(3))
    assert sim.backend_used == 'numpy'


def test_angular_damping_and_torque_drive():
    particle = Spheroid(a=70*nm, b=50*nm)
    I = particle.inertia_tensor[2, 2]
    # a constant torque about the symmetry axis spins the particle up linearly; damping saturates it
    sim = RigidBodySimulation(trap=free_trap(particle), num_samples=2000, Δt=1e-8, gas_damping=False, backend='numpy',
                              torque_drive={'offset': (0, 0, 1e-24)}, angular_damping=(0, 0, 2e5))
    ω = sim.angular_velocities_lab()[:, 2]
    t = sim.time_array
    expected = 1e-24 / (I * 2e5) * (1 - np.exp(-2e5 * t))
    assert np.max(np.abs(ω - expected)) < 1e-6 * expected[-1]


def test_parametric_feedback_cools_at_the_harmonic_rate():
    # η = G·s·ṡ with s = x̂·(R ẑ_body) = -θγ removes the energy of a harmonic librator at G Ω² E²/2K,
    # so E(t) = E0/(1 + G Ω² E0 t/2K); a negative gain heats
    trap = Trap(tweezer=Tweezer(), particle=aligned_spheroid(), gas=Gas(P=1 * mbar))
    Ω, K, I = trap.Ωθ['γ'], trap.torque_stiffness_tensor[2, 2], trap.Ij['γ']
    E0 = 0.5 * I * 1e4**2
    period = 2 * π / Ω
    num_samples, Δt = 40_000, period / 400
    T = num_samples * Δt
    gain = 2 * K / (Ω**2 * E0 * T)  # halves the energy over the run
    energies = {}
    for sign in (1, -1):
        sim = RigidBodySimulation(trap=trap, num_samples=num_samples, Δt=Δt, gas_damping=False, ω0=(0, 0, 1e4),
                                  parametric_feedback={'gain': sign * gain, 'body_axis': (0, 0, 1), 'lab_axis': (1, 0, 0)})
        n = sim.lab_directions((0, 0, 1))
        ω = sim.angular_velocities_lab()
        θ, θ_dot = -n[:, 0], -np.cross(ω, n)[:, 0]
        E = 0.5 * I * θ_dot**2 + 0.5 * K * θ**2
        energies[sign] = E[-400:].mean()  # over the last period
    assert energies[1] == pytest.approx(E0 / 2, rel=0.03)
    assert energies[-1] > E0

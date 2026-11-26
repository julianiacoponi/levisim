"""
Tests of levisim.physics.fields.superconducting: a permanent magnet above an ideal superconducting
plane (image dipole; Vinante et al., Phys. Rev. Applied 13, 064027 (2020))
"""

import numpy as np
import pytest

from levisim.core.constants import g, μ0, π
from levisim.core.units import nm
from levisim.physics import Gas, Sphere, Trap
from levisim.physics.fields import (
    Gravity,
    MeissnerImagePlane,
    meissner_levitation_height,
)
from levisim.physics.material import Material
from levisim.physics.particle import body_axis_to_quaternion
from levisim.physics.rotation import (
    quaternion_from_rotation_vector,
    quaternion_multiply,
)

MAGNET = Material('Nd alloy (Vinante et al. 2020)', 7430.0, magnetisation=0.71 / μ0)


def magnet(R=30.1e-6, axis=(0, 1, 0)):
    """A uniformly magnetised sphere with its moment along the lab `axis`"""
    return Sphere(R, material=MAGNET, orientation=body_axis_to_quaternion(axis))


def levitated_trap(particle, P=0.0):
    μ = np.linalg.norm(particle.magnetic_moment)
    h0 = meissner_levitation_height(μ, particle.m)
    plane = MeissnerImagePlane(surface_point=(-h0, 0, 0), normal=(1, 0, 0))
    return Trap(particle=particle, fields=[plane, Gravity()], gas=Gas(P=P or 1e-12)), h0


def test_vinante_image_method_frequencies():
    particle = magnet()
    trap, h0 = levitated_trap(particle)
    assert np.allclose(trap.static_force(np.zeros(3)), 0, atol=1e-12 * particle.m * g)
    assert np.abs(trap.static_torque).max() == 0
    # Eqs. (2), (5), (6): ωz² = 4g/z0, ωβ² = 5 z0 g/(3R²); the tilt of μ towards the normal is a rotation about z
    K = trap.stiffness_tensor
    assert K[0, 0] / particle.m == pytest.approx(4 * g / h0, rel=1e-12)
    assert K[1, 1] == 0 and K[2, 2] == 0
    Kθ = trap.torque_stiffness_tensor
    I = particle.inertia_tensor[0, 0]
    assert Kθ[2, 2] / I == pytest.approx(5 * h0 * g / (3 * particle.r**2), rel=1e-12)
    assert np.allclose(np.delete(np.delete(Kθ, 2, 0), 2, 1), 0, atol=1e-12 * Kθ[2, 2])
    # Eq. (7): R = √(20/3)·g/(ωz ωβ), independent of density and magnetisation
    ωz, ωβ = np.sqrt(4 * g / h0), np.sqrt(5 * h0 * g / (3 * particle.r**2))
    assert particle.r == pytest.approx(np.sqrt(20 / 3) * g / (ωz * ωβ), rel=1e-12)
    # the paper's numbers: z0 = 311 μm, 56.5 Hz and 377 Hz for R = 30.1 μm, ρ = 7430 kg/m³, μ0M = 0.71 T
    assert h0 == pytest.approx(311e-6, rel=0.01)
    assert ωz / (2 * π) == pytest.approx(56.5, rel=0.01)
    assert ωβ / (2 * π) == pytest.approx(377, rel=0.01)


def test_image_force_torque_and_stiffness_against_finite_differences():
    particle = magnet(axis=(0.3, 0.8, 0.5))
    plane = MeissnerImagePlane(surface_point=(-300e-6, 10e-6, 0), normal=(1, 0.1, -0.2))
    r = np.array([5e-6, -3e-6, 2e-6])
    q0 = particle.orientation
    U = plane.potential(r, particle)
    assert U > 0
    h = 1e-9
    F_fd = np.array([-(plane.potential(r + h * e, particle) - plane.potential(r - h * e, particle)) / (2 * h) for e in np.eye(3)])
    assert np.allclose(plane.force(r, 0, particle), F_fd, rtol=1e-6)

    def U_rot(θ):
        return plane.potential(r, particle, orientation=quaternion_multiply(quaternion_from_rotation_vector(θ), q0))

    a = 1e-5
    τ_fd = np.array([-(U_rot(a * e) - U_rot(-a * e)) / (2 * a) for e in np.eye(3)])
    assert np.allclose(plane.torque(r, 0, particle), τ_fd, rtol=1e-6, atol=1e-9 * np.abs(τ_fd).max())
    K = plane.torque_stiffness(particle, r0=r)
    K_fd = np.array([[(U_rot(a * (ei + ej)) - U_rot(a * (ei - ej)) - U_rot(a * (ej - ei)) + U_rot(-a * (ei + ej))) / (4 * a**2)
                      for ej in np.eye(3)] for ei in np.eye(3)])
    # absolute tolerance from the largest entry: the small off-diagonal stiffness (~3e-3 of it) carries
    # finite-difference noise of ~1e-6 of the largest entry, which differs between platforms
    assert np.allclose(K, K_fd, rtol=1e-5, atol=1e-5 * np.abs(K_fd).max())
    with pytest.raises(ValueError, match='not above'):
        plane.force((-400e-6, 0, 0), 0, particle)
    with pytest.raises(ValueError, match='permanent magnetic moment'):
        plane.force(r, 0, Sphere(100 * nm))


def test_rigid_body_frequencies_of_the_levitated_magnet():
    from levisim.simulation import RigidBodySimulation
    particle = magnet()
    trap, h0 = levitated_trap(particle)
    Ωz = np.sqrt(4 * g / h0)
    Ωβ = np.sqrt(trap.torque_stiffness_tensor[2, 2] / particle.inertia_tensor[0, 0])
    # small vertical and tilt oscillations, no damping
    sim = RigidBodySimulation(trap=trap, num_samples=40_000, Δt=2 * π / Ωβ / 200, gas_damping=False,
                              r0=(0.1e-6, 0, 0), ω0=(0, 0, 1e-4 * Ωβ))
    assert sim.backend_used == 'numpy'
    x = sim.X[:, 0]
    θ = sim.lab_directions((0, 0, 1))[:, 0]  # the moment's tilt towards the normal

    def frequency(signal):
        s = signal - signal.mean()
        up = np.flatnonzero((s[:-1] < 0) & (s[1:] >= 0))
        times = (up - s[up] / (s[up + 1] - s[up])) * sim.Δt
        return 2 * π / np.diff(times).mean()

    assert frequency(x) == pytest.approx(Ωz, rel=2e-3)
    assert frequency(θ) == pytest.approx(Ωβ, rel=1e-3)

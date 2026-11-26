"""
Gas momentum diffusion of Sphere and TabulatedParticle for any surface temperature ratio γs (owner
decision 31), against the surface integrals of Martinetz et al. (2018) that the other shapes use
"""
import numpy as np
import pytest

from levisim.core.constants import kB
from levisim.core.units import mbar, nm
from levisim.physics import Bead, Ellipsoid, Gas, Sphere, Spheroid, TabulatedParticle

GAS = Gas(P=1e-3 * mbar)
CASES = [(1.0, 1.0), (1.0, 1.5), (0.7, 2.0), (0.3, 0.6), (0.0, 3.0), (1.0, 0.0)]


@pytest.mark.parametrize('αc, γs', CASES)
def test_sphere_diffusion_matches_the_surface_integral_for_any_gamma_s(αc, γs):
    r = 80*nm
    sphere = Sphere(r=r)
    # Ellipsoid(a, a, a) evaluates the diffusion from the Gauss quadrature of Eq. (41), not Table I
    numerical = Ellipsoid(r, r, r).rotranslational_diffusion_matrix(GAS, αc, γs)
    np.testing.assert_allclose(sphere.translational_diffusion_tensor(GAS, αc, γs), numerical[:3, :3],
                               rtol=1e-12, atol=1e-12 * np.max(np.abs(numerical[:3, :3])))
    np.testing.assert_allclose(sphere.rotational_diffusion_tensor(GAS, αc, γs), numerical[3:, 3:],
                               rtol=1e-12, atol=1e-12 * np.max(np.abs(numerical[:3, :3])) * r**2)


def test_gamma_s_1_keeps_the_fluctuation_dissipation_arithmetic_bit_for_bit():
    bead = Bead()
    Γ = bead.translational_damping_tensor(GAS)
    assert np.array_equal(bead.translational_diffusion_tensor(GAS), kB * GAS.T * bead.m * Γ)
    assert np.array_equal(bead.rotational_diffusion_tensor(GAS), kB * GAS.T * bead.rotational_damping_tensor(GAS) @ bead.inertia_tensor)
    # the closed form for γs ≠ 1 is continuous through γs = 1
    np.testing.assert_allclose(bead.translational_diffusion_tensor(GAS, 0.8, 1 + 1e-12),
                               bead.translational_diffusion_tensor(GAS, 0.8, 1.0), rtol=1e-10)
    np.testing.assert_allclose(bead.rotational_diffusion_tensor(GAS, 0.8, 1 - 1e-12),
                               bead.rotational_diffusion_tensor(GAS, 0.8, 1.0), rtol=1e-10)


def test_sphere_diffusion_limits():
    s = Sphere(r=100*nm)
    # specular reflection (αc = 0): the surface temperature does not enter, D = kB T m Γ (Epstein,
    # specular) for every γs, and there is no torque noise
    for γs in (0.5, 2.0):
        np.testing.assert_allclose(s.translational_diffusion_tensor(GAS, 0.0, γs),
                                   kB * GAS.T * s.m * s.translational_damping_tensor(GAS, 0.0, γs), rtol=1e-13)
        assert np.all(s.rotational_diffusion_tensor(GAS, 0.0, γs) == 0)
    # a hot surface (γs > 1) diffuses more than one at the gas temperature
    assert np.all(np.diag(s.translational_diffusion_tensor(GAS, 1.0, 2.0)) > np.diag(s.translational_diffusion_tensor(GAS)))
    assert np.all(np.diag(s.rotational_diffusion_tensor(GAS, 1.0, 2.0)) > np.diag(s.rotational_diffusion_tensor(GAS)))
    # equals the spheroid of equal axes (also a surface integral) and is isotropic
    np.testing.assert_allclose(s.translational_diffusion_tensor(GAS, 0.9, 1.7),
                               Spheroid(100*nm, 100*nm).translational_diffusion_tensor(GAS, 0.9, 1.7),
                               rtol=1e-12, atol=1e-12 * s.translational_diffusion_tensor(GAS, 0.9, 1.7)[0, 0])


def test_tabulated_particle_diffusion_for_any_gamma_s():
    sphere = Sphere(r=70*nm)
    tab = TabulatedParticle.from_particle(sphere, gas=GAS, accommodation=0.8, surface_temperature_ratio=1.6)
    other = Gas(P=3e-2 * mbar, T=250.0)
    for gas in (GAS, other):
        np.testing.assert_allclose(tab.translational_diffusion_tensor(gas, 0.8, 1.6),
                                   sphere.translational_diffusion_tensor(gas, 0.8, 1.6), rtol=1e-12)
        np.testing.assert_allclose(tab.rotational_diffusion_tensor(gas, 0.8, 1.6),
                                   sphere.rotational_diffusion_tensor(gas, 0.8, 1.6), rtol=1e-12)
    with pytest.raises(ValueError, match='holds for αc = 0.8'):
        tab.translational_diffusion_tensor(GAS, 0.8, 1.0)
    # round trip
    again = TabulatedParticle(**tab.to_dict())
    assert np.array_equal(again.rotational_diffusion_tensor(GAS, 0.8, 1.6), tab.rotational_diffusion_tensor(GAS, 0.8, 1.6))
    # a callable covers every αc, γs
    callable_tab = TabulatedParticle(
        mass=sphere.m, inertia=sphere.inertia_tensor, polarisability=sphere.static_polarisability_tensor(),
        translational_damping=sphere.translational_damping_tensor, rotational_damping=sphere.rotational_damping_tensor,
        translational_diffusion=sphere.translational_diffusion_tensor, rotational_diffusion=sphere.rotational_diffusion_tensor,
    )
    for αc, γs in CASES:
        assert np.array_equal(callable_tab.translational_diffusion_tensor(GAS, αc, γs),
                              sphere.translational_diffusion_tensor(GAS, αc, γs))


def test_tabulated_particle_without_diffusion_tables_keeps_gamma_s_1_and_raises_otherwise():
    sphere = Sphere(r=70*nm)
    tab = TabulatedParticle.from_particle(sphere, gas=GAS)                   # γs = 1: no tables stored
    assert tab.to_dict()['translational_diffusion'] is None
    assert np.array_equal(tab.translational_diffusion_tensor(GAS),
                          kB * GAS.T * tab.m * tab.translational_damping_tensor(GAS))
    callable_only = TabulatedParticle(mass=sphere.m, inertia=sphere.inertia_tensor, polarisability=1e-32,
                                      translational_damping=sphere.translational_damping_tensor,
                                      rotational_damping=sphere.rotational_damping_tensor)
    with pytest.raises(NotImplementedError, match='translational_diffusion'):
        callable_only.translational_diffusion_tensor(GAS, 1.0, 1.5)
    with pytest.raises(NotImplementedError, match='rotational_diffusion'):
        callable_only.rotational_diffusion_tensor(GAS, 1.0, 1.5)
    with pytest.raises(ValueError, match='positive semi-definite'):
        TabulatedParticle(mass=1e-18, inertia=1e-30, polarisability=1e-32, damping_gas=GAS,
                          translational_diffusion=np.diag([1.0, -1.0, 1.0]))

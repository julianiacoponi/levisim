"""
Corrections of `Trap` and its use in `MultiTrap` (levisim 1.0)

- the tweezer potential `Trap.Utw` (and `Tweezer.potential`, `U_depth`) is the cycle-averaged dipole
  energy, whose curvature is the stiffness and whose gradient is the force (bit-identical to before)
- the off-diagonal stiffness of a trap's combined fields couples the axes of its particle in
  `MultiTrap`: the normal modes of a misaligned hybrid trap against the eigenvalues of its Hessian
- a hybrid Paul trap in `MultiTrap` has the exact (Floquet) secular frequencies, checked against an
  independent Hill-determinant solution of Mathieu's equation
- directed force noise β: √2 for the tweezer shorthand, 0 for traps built with fields=
- the finite-size torque is on by default for non-spherical particles, off for spheres
"""
import numpy as np
import pytest
from scipy.optimize import brentq

from levisim.core.constants import ħ
from levisim.core.units import Hz, kHz, mbar, mm, nm
from levisim.multiphysics.multitrap import MultiTrap
from levisim.physics import Spheroid
from levisim.physics.bead import Bead
from levisim.physics.fields import HarmonicPotential, PaulTrap
from levisim.physics.gas import Gas
from levisim.physics.particle import Sphere
from levisim.physics.shapes import Dumbbell, Ellipsoid
from levisim.physics.trap import Trap
from levisim.physics.tweezer import Tweezer


def rotation_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


# --------------------------------------------------------------------------------------------------
# decision 33: the tweezer potential is the cycle-averaged one
# --------------------------------------------------------------------------------------------------
def test_potential_curvature_is_the_stiffness_and_its_gradient_the_force():
    trap = Trap()
    h = 1e-3 * trap.tweezer.w0
    for j, axis in enumerate('xyz'):
        e = np.eye(3)[j] * h
        curvature = (trap.Utw(*e) - 2 * trap.Utw(0, 0, 0) + trap.Utw(*(-e))) / h**2
        assert curvature == pytest.approx(trap.kj[axis], rel=1e-5)
    r = np.array([0.13, -0.21, 0.4]) * trap.tweezer.w0
    gradient = np.array([(trap.Utw(*(r + h*e)) - trap.Utw(*(r - h*e))) / (2*h) for e in np.eye(3)])
    assert -gradient == pytest.approx(np.array(trap.Ftw(*r)), rel=1e-6)
    assert trap.U_depth == pytest.approx(np.real(trap.α_eff) * trap.tweezer.E0**2 / 4, rel=1e-14)
    assert trap.v_escape == pytest.approx(np.sqrt(2 * trap.U_depth / trap.bead.m), rel=1e-14)
    assert trap.tweezer.potential(r, trap.bead) == trap.Utw(*r)


def test_tweezer_force_is_bit_identical_to_the_peak_energy_form():
    """The levisim <= 0.1 force 0.5·(4x/wx²)·U with the peak energy U = -(Re α/2)|E|²"""
    trap = Trap()
    tweezer, α = trap.tweezer, np.real(trap.α_eff)
    rng = np.random.default_rng(3)
    for r in rng.normal(scale=0.3 * tweezer.w0, size=(50, 3)):
        x, y, z = r
        U_peak = -(α / 2) * tweezer.E_field_amplitude(x, y, z)**2
        legacy = (
            0.5 * (4 * x / tweezer.wx(z)**2) * U_peak,
            0.5 * (4 * y / tweezer.wy(z)**2) * U_peak,
            0.5 * (4 * z / tweezer.wz(z)**2) * U_peak * (1 - 2 * tweezer.ellipse(x, y, z)),
        )
        assert trap.Ftw(x, y, z) == legacy
        assert (trap.Ftwx(x, y, z), trap.Ftwy(x, y, z), trap.Ftwz(x, y, z)) == legacy
        assert tuple(tweezer.force(r, 0.0, trap.bead)) == legacy


# --------------------------------------------------------------------------------------------------
# decision 36: directed force noise β
# --------------------------------------------------------------------------------------------------
def test_directed_noise_default_depends_on_how_the_trap_is_built():
    paul = PaulTrap.ring(10.0, 2e4 * Hz)
    particle = Sphere(r=100*nm, charge=500)
    assert Trap().β == np.sqrt(2)
    assert Trap(tweezer=Tweezer(), extra_fields=[paul], bead=Bead(charge=500)).β == np.sqrt(2)
    fields_trap = Trap(fields=[paul], particle=particle)
    assert fields_trap.β == 0.0 and fields_trap.Sdir == 0.0
    assert Trap(fields=[Tweezer()]).β == 0.0
    assert Trap(fields=[paul], particle=particle, β=np.sqrt(2)).β == np.sqrt(2)

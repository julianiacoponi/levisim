"""
Permanent magnets above an ideal superconductor: the Meissner image dipole

A type-I superconductor in the Meissner state expels the field of a nearby magnet (B·n̂ = 0 on its
surface). Above an infinite flat surface this is the field of an image dipole: a moment μ at height h
has its image μ' = μ - 2(μ·n̂)n̂ at depth h, and the energy of the magnet in the field of the image
(half the dipole-dipole energy, since the image is induced) is ::

    U(h, μ) = μ0 (‖μ‖² + (μ·n̂)²) / (64π h³) = μ0 (μ∥² + 2μ⊥²) / (64π h³)

(μ∥ parallel to the surface, μ⊥ = μ·n̂ normal to it). It repels the magnet (F = 3U/h along n̂) and
turns its moment parallel to the surface. With gravity along -n̂ the magnet levitates at ::

    h0⁴ = 3μ0‖μ‖² / (64π m g),     Ωz² = 4g/h0,     κ_tilt = 2μ0‖μ‖²/(64π h0³) = (2/3) m g h0

(the tilt of μ out of the surface plane; Vinante et al., Phys. Rev. Applied 13, 064027 (2020),
Eqs. (1)-(6)). A flat surface gives no lateral confinement and no confinement of the azimuth of μ or
of the rotation about μ; a real trap (a cylindrical well) adds them through its side walls, which this
element does not model.

`MeissnerImagePlane` is a `ForceElement` and works with `Gravity` in `Trap` (Ωj, Ωθ) and
`RigidBodySimulation` (the NumPy loop, through `force` and `torque`). The particle's moment is
`Particle.magnetic_moment` (from its material's remanent magnetisation).

Reconciliation with the stage-1 magnetic sources (round5/wip/diamagnetic): that track's
`MagneticTrap` describes induced moments, U = -(V/2μ0) B·χ_eff·B with the demagnetising correction
χ_eff = χ(𝟙 + Nχ)⁻¹; it rejects remanent magnetisation. A superconducting sphere in a source field
(an anti-Helmholtz `CoilPair`) is then the material χ = -1 in that element, whose shape correction
gives the Meissner value χ_eff = -3/2 (N = 1/3) with no new element. This module covers the opposite
case, a permanent magnet next to a superconducting surface, which has no external source; a permanent
magnet in a source field (U = -μ·B(r), τ = μ × B) belongs with the sources and is left to that track.
"""

import numpy as np

from levisim.core.constants import g as g_earth
from levisim.core.constants import μ0, π
from levisim.physics.fields.base import ForceElement, as_position

__all__ = [
    'MeissnerImagePlane',
    'meissner_levitation_height',
]


def meissner_levitation_height(moment, mass, g=g_earth):
    """
    Equilibrium height h0 = (3μ0μ²/64π m g)^¼ of a magnet with moment ‖μ‖ and mass m above an
    infinite superconducting plane, its moment parallel to the surface (Vinante et al. (2020), Eq. (2))
    [m]
    """
    return float((3 * μ0 * float(moment)**2 / (64 * π * float(mass) * float(g)))**0.25)


class MeissnerImagePlane(ForceElement):
    """
    An infinite, ideal (Meissner) superconducting plane acting on a particle's permanent magnetic
    moment through its image dipole (see the module docstring)

    ::

        MeissnerImagePlane(surface_point=(0, 0, 0), normal=(1, 0, 0))

    - surface_point: a point on the surface [m]
    - normal: the surface normal pointing into the vacuum, where the particle is (normalised); the
      default +x is opposite to levisim's default `Gravity` direction (-x)

    The particle must have a `magnetic_moment` (body frame, e.g. from `Material(magnetisation=...)`)
    and be above the surface (h > 0).
    """

    is_conservative = True
    # the image force depends on μ·n̂: RigidBodySimulation passes the current orientation to `force`
    orientation_dependent_force = True

    def __init__(self, surface_point=(0.0, 0.0, 0.0), normal=(1.0, 0.0, 0.0)):
        self.surface_point = as_position(surface_point)
        n = np.asarray(normal, dtype=float)
        if n.shape != (3,) or not np.linalg.norm(n) > 0:
            raise ValueError(f'normal must be a non-zero 3-vector, not {normal}')
        self.normal = n / np.linalg.norm(n)

    # ----------------------------------------------------------------------------------------------
    def _moment(self, particle, orientation=None):
        μ_body = getattr(particle, 'magnetic_moment', None)
        if μ_body is None:
            raise ValueError(
                f'MeissnerImagePlane acts on a permanent magnetic moment, and the {type(particle).__name__}'
                ' has none (give its material a remanent magnetisation)'
            )
        return np.asarray(particle.lab_vector(μ_body, orientation), dtype=float)

    def height(self, r):
        """
        Height h = n̂·(r - surface_point) of the particle's centre above the surface
        [m]
        """
        h = float(self.normal @ (as_position(r) - self.surface_point))
        if h <= 0:
            raise ValueError(f'the particle is not above the superconducting surface (h = {h:.3e} m)')
        return h

    def _energy_scale(self, h):
        return μ0 / (64 * π * h**3)

    def potential(self, r, particle, orientation=None):
        """
        U = μ0(‖μ‖² + (μ·n̂)²)/(64π h³)
        [J]
        """
        μ = self._moment(particle, orientation)
        μn = μ @ self.normal
        return self._energy_scale(self.height(r)) * (μ @ μ + μn**2)

    def force(self, r, t, particle, orientation=None):
        """
        F = -∂U/∂h n̂ = (3U/h) n̂, repulsive
        [N]
        """
        h = self.height(r)
        return 3 * self.potential(r, particle, orientation) / h * self.normal

    def stiffness(self, r0=None, particle=None):
        """
        K = (12U/h²) n̂n̂ᵀ: only along the normal (a flat surface does not confine sideways)
        [N/m]
        """
        r0 = np.zeros(3) if r0 is None else as_position(r0)
        h = self.height(r0)
        return 12 * self.potential(r0, particle) / h**2 * np.outer(self.normal, self.normal)

    def torque(self, r, t, particle, orientation=None):
        """
        τ = -(μ0 (μ·n̂)/(32π h³)) μ × n̂, turning μ parallel to the surface
        [N·m]
        """
        μ = self._moment(particle, orientation)
        μn = μ @ self.normal
        return -2 * self._energy_scale(self.height(r)) * μn * np.cross(μ, self.normal)

    def torque_stiffness(self, particle, orientation=None, r0=None):
        """
        K_ij = ∂²U/∂θ_i∂θ_j = 2c[(n̂·G_iμ)(n̂·G_jμ) + (n̂·μ)(n̂·½{G_i, G_j}μ)], c = μ0/(64π h³); with μ
        parallel to the surface, 2c‖μ‖² for the tilt out of the surface plane and zero otherwise
        [N·m/rad]
        """
        from levisim.physics.rotation import GENERATORS
        r0 = np.zeros(3) if r0 is None else as_position(r0)
        c = self._energy_scale(self.height(r0))
        μ = self._moment(particle, orientation)
        n = self.normal
        first = np.array([n @ (G @ μ) for G in GENERATORS])
        K = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                second = n @ (0.5 * (GENERATORS[i] @ GENERATORS[j] + GENERATORS[j] @ GENERATORS[i]) @ μ)
                K[i, j] = 2 * c * (first[i] * first[j] + (n @ μ) * second)
        return K

    def __repr__(self):
        return f'MeissnerImagePlane(surface_point={self.surface_point.tolist()}, normal={self.normal.tolist()})'

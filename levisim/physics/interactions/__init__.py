"""
levisim.physics.interactions
============================
Forces between levitated particles

An `Interaction` returns the force on every particle at laboratory-frame positions (N, 3), and the
stiffness blocks K[j, k] = -∂F_j/∂r_k (N, N, 3, 3) that `MultiTrap(interactions=[...])` linearises.
Simulations that take ``interactions=[...]`` call `forces` once per drift evaluation.

- `Interaction`: the base class (forces, potential, stiffness_blocks, is_reciprocal)
- `PairInteraction`: a central pair potential U(r), with closed-form forces and stiffness
- `Coulomb`: q_j·q_k/(4πε0·r), from `Particle.q`
- `NewtonianGravity`: -G·m_j·m_k/r
- `Casimir`: retarded Casimir-Polder (r ≫ R) or proximity-force (gap ≪ R) attraction of
  dielectric spheres, at zero temperature
- `OpticalBinding`: the non-conservative, non-reciprocal force of the light that each particle
  scatters into the others' tweezers (point dipoles, free-space Green tensor)
"""

from levisim.physics.interactions.base import (
    Interaction,
    PairInteraction,
    as_positions,
    attach_interaction_traps,
    particle_radius,
)
from levisim.physics.interactions.central import (
    Casimir,
    Coulomb,
    G_newton,
    NewtonianGravity,
    PowerLawPairInteraction,
    lifshitz_plate_factor,
)
from levisim.physics.interactions.optical_binding import (
    OpticalBinding,
    dipole_field,
    dipole_green_tensor,
    dipole_green_tensor_derivatives,
    tweezer_field,
)

__all__ = [
    'Casimir',
    'Coulomb',
    'G_newton',
    'Interaction',
    'NewtonianGravity',
    'OpticalBinding',
    'PairInteraction',
    'PowerLawPairInteraction',
    'as_positions',
    'attach_interaction_traps',
    'dipole_field',
    'dipole_green_tensor',
    'dipole_green_tensor_derivatives',
    'lifshitz_plate_factor',
    'particle_radius',
    'tweezer_field',
]

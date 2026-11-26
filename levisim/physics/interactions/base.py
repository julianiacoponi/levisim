"""
Interaction: the base class of forces between levitated particles

An interaction acts on a set of particles at laboratory-frame positions (shape (N, 3)) and
returns one force per particle. It also supplies the stiffness blocks at a configuration, which
is what `MultiTrap` linearises. The two methods are the interface shared with every simulation
(``interactions=[...]``):

- ``forces(positions, particles, t=0.0)`` -> ndarray (N, 3) [N]
- ``stiffness_blocks(positions0, particles)`` -> ndarray (N, N, 3, 3) [N/m], with
  ``K[j, k, a, b] = -∂F_j,a/∂r_k,b`` at positions0

`PairInteraction` is the common case of a central pair potential U(r) of the distance r between two particles, for which the
forces, the potential and the stiffness blocks follow from U'(r) and U''(r) in closed form.
"""

import itertools

import numpy as np

__all__ = [
    'Interaction',
    'PairInteraction',
    'as_positions',
    'attach_interaction_traps',
    'particle_radius',
]


def attach_interaction_traps(interaction, traps):
    """
    Give an interaction the host's Traps (`Interaction.attach_traps`), if it takes them; objects that
    only implement `forces` and `stiffness_blocks` are left alone
    """
    attach = getattr(interaction, 'attach_traps', None)
    if attach is not None:
        attach(traps)


def as_positions(positions, num_particles=None):
    """Laboratory-frame positions as a float array of shape (N, 3) [m]"""
    positions = np.asarray(positions, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f'positions must have shape (N, 3), not {positions.shape}')
    if num_particles is not None and positions.shape[0] != num_particles:
        raise ValueError(f'{positions.shape[0]} positions for {num_particles} particles')
    return positions


def particle_radius(particle):
    """
    The radius of a spherical particle, or None if the particle has no radius `r`
    [m]
    """
    radius = getattr(particle, 'r', None)
    return None if radius is None else float(radius)


class Interaction:
    """
    Base class of the forces between levitated particles

    Subclasses implement `forces` (and `potential` if conservative), and may override
    `stiffness_blocks`, whose default is the central difference of `forces`.

    Keyword arguments:

    - pairs [None]: the pairs (j, k) of particle indices (0-based) that interact; None means every
      pair. An interaction that is not a sum over pairs may ignore it.

    Class attributes:

    - is_conservative [True]: `forces` = -∇`potential`
    - is_reciprocal [True]: the stiffness blocks obey K_jk = K_kjᵀ (always true for a conservative
      interaction). A non-reciprocal interaction (e.g. non-conservative optical binding) has
      K_jk ≠ K_kjᵀ, and `MultiTrap` keeps the two directions apart.
    """

    is_conservative = True
    is_reciprocal = True

    def __init__(self, pairs=None):
        self.pairs = None if pairs is None else [tuple(int(index) for index in pair) for pair in pairs]
        if self.pairs is not None:
            for j, k in self.pairs:
                if j == k:
                    raise ValueError(f'an interaction pair needs two different particles, not ({j}, {k})')

    @property
    def label(self):
        """A short name of the interaction, for messages and summaries"""
        return type(self).__name__

    @property
    def finite_difference_step(self):
        """
        Step of the central differences in the default `stiffness_blocks`
        [m]
        """
        return 1e-10

    def pair_indices(self, num_particles):
        """The interacting pairs (j, k), j < k as given, for num_particles particles"""
        if self.pairs is None:
            return list(itertools.combinations(range(num_particles), 2))
        for j, k in self.pairs:
            if max(j, k) >= num_particles:
                raise ValueError(f'pair ({j}, {k}) refers to a particle beyond the {num_particles} given')
        return list(self.pairs)

    def attach_traps(self, traps):
        """
        Called by every host (MultiTrap, MultiParticleFreeSpaceSimulation, the nonlinear
        CavitySimulation) with its Traps, in the order of the particles, before the first `forces`.
        Interactions that need more than the particles (e.g. `OpticalBinding`, which needs the tweezers
        and their foci) keep them; the default does nothing.
        """

    def forces(self, positions, particles, t=0.0):
        """
        Force on every particle at the laboratory-frame positions (N, 3) and time t
        [N]
        """
        raise NotImplementedError(f'{type(self).__name__} does not implement forces(positions, particles, t)')

    def potential(self, positions, particles):
        """
        Total interaction energy at the positions (N, 3), for a conservative interaction
        [J]
        """
        raise NotImplementedError(f'{type(self).__name__} has no potential energy')

    def stiffness_blocks(self, positions0, particles):
        """
        Stiffness blocks K[j, k] = -∂F_j/∂r_k (3 × 3 each) at the positions0 (N, 3)
        [N/m]

        By default from central differences of `forces` at t = 0 with step h =
        `finite_difference_step`. Block K[j, j] is the stiffness the interaction adds to particle
        j's own trap; K[j, k] (j ≠ k) couples particle k's displacement to the force on j.
        """
        positions0 = as_positions(positions0, len(particles))
        N = len(particles)
        h = self.finite_difference_step
        K = np.zeros((N, N, 3, 3))
        for k in range(N):
            for b in range(3):
                step = np.zeros((N, 3))
                step[k, b] = h
                F_plus = np.asarray(self.forces(positions0 + step, particles, 0.0), dtype=float)
                F_minus = np.asarray(self.forces(positions0 - step, particles, 0.0), dtype=float)
                K[:, k, :, b] = -(F_plus - F_minus) / (2*h)
        return K

    def __repr__(self):
        pairs = '' if self.pairs is None else f'pairs={self.pairs}'
        return f'{type(self).__name__}({pairs})'


class PairInteraction(Interaction):
    """
    A central pair potential U_jk(r) of the centre-to-centre distance r, summed over the pairs

    Subclasses implement `pair_derivatives(r, particle_j, particle_k)`, returning (U, dU/dr,
    d²U/dr²) at the centre-to-centre distance r [J, N, N/m]. Then, with r̂ = (r_j - r_k)/r, ::

        F_j = -U'(r)·r̂ = -F_k
        K_jj = K_kk = -K_jk = -K_kj = U''·r̂r̂ᵀ + (U'/r)·(1 - r̂r̂ᵀ)

    so the interaction is conservative and reciprocal, and obeys Newton's third law exactly.
    """

    def pair_derivatives(self, r, particle_j, particle_k):
        """(U, dU/dr, d²U/dr²) of one pair at centre-to-centre distance r [J, N, N/m]"""
        raise NotImplementedError

    def check_separation(self, r, particle_j, particle_k):
        """Hook to reject separations outside the model's validity (raises ValueError)"""
        if not r > 0:
            raise ValueError(f'{self.label}: two particles at the same position')

    def _pair_geometry(self, positions, j, k, particles):
        displacement = positions[j] - positions[k]
        r = float(np.sqrt(displacement @ displacement))
        self.check_separation(r, particles[j], particles[k])
        return displacement / r, r

    def forces(self, positions, particles, t=0.0):
        positions = as_positions(positions, len(particles))
        F = np.zeros_like(positions)
        for j, k in self.pair_indices(len(particles)):
            r_hat, r = self._pair_geometry(positions, j, k, particles)
            _, dU, _ = self.pair_derivatives(r, particles[j], particles[k])
            F[j] -= dU * r_hat
            F[k] += dU * r_hat
        return F

    def potential(self, positions, particles):
        positions = as_positions(positions, len(particles))
        energy = 0.0
        for j, k in self.pair_indices(len(particles)):
            _, r = self._pair_geometry(positions, j, k, particles)
            energy += self.pair_derivatives(r, particles[j], particles[k])[0]
        return energy

    def pair_hessian(self, r_hat, r, particle_j, particle_k):
        """Hessian of U_jk with respect to r_j, i.e. the block K_jj of this pair [N/m]"""
        _, dU, d2U = self.pair_derivatives(r, particle_j, particle_k)
        projector = np.outer(r_hat, r_hat)
        return d2U * projector + (dU / r) * (np.eye(3) - projector)

    def stiffness_blocks(self, positions0, particles):
        positions0 = as_positions(positions0, len(particles))
        N = len(particles)
        K = np.zeros((N, N, 3, 3))
        for j, k in self.pair_indices(N):
            r_hat, r = self._pair_geometry(positions0, j, k, particles)
            H = self.pair_hessian(r_hat, r, particles[j], particles[k])
            K[j, j] += H
            K[k, k] += H
            K[j, k] -= H
            K[k, j] -= H
        return K

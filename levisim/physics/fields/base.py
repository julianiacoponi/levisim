"""
ForceElement: the base class of everything that pushes on a levitated particle

A `Trap` is a list of force elements (its ``fields``) acting on one particle. Each element supplies
the same small interface, so that one definition feeds both the time-domain simulations (through the
nonlinear `force`) and the frequency-domain `MultiTrap` (through the `stiffness` tensor at the
linearisation point):

- ``force(r, t, particle)``: the force on the particle at position r and time t [N]
- ``torque(r, t, particle, orientation)``: the torque [N·m] (zero for a point particle)
- ``potential(r, particle)``: the potential energy, for a conservative element [J]
- ``stiffness(r0, particle)``: the 3×3 tensor K_ij = -∂F_i/∂r_j at r0 [N/m], in closed form where
  an element has one, otherwise from central differences of `force`
- ``noise(particle)``: the (two-sided) force spectral density the element adds, a 3×3 matrix [N²/Hz]
- ``damping_rates(particle)``: linear velocity damping rates (Γx, Γy, Γz), F = -Γ·p [2π·Hz]
- ``is_optical``: whether the element is a light field that polarises the particle
- ``is_time_dependent``, ``secular_force(r, particle)``: for an explicitly time-dependent element
  (a `PaulTrap`), the time-averaged force that static quantities (`Trap.find_equilibrium`) use
- ``max_time_step(particle)``: the largest time step that resolves the element in a time-domain
  simulation (None if it sets no limit)
"""

import numpy as np

__all__ = [
    'SIMULATION_GROUPS',
    'ForceElement',
]

# The slots in which FreeSpaceSimulation adds the elements' forces to dp/dt. The order of the slots
# fixes the floating-point order of the sum, which is what keeps seeded traces bit-identical to the
# single-tweezer code that preceded the force elements:
#   dp/dt = (Σ 'field' forces) - Γ·p + 'kick' force [x, y], then += 'recoil' [z], then += 'gravity'
SIMULATION_GROUPS = ('field', 'damping', 'kick', 'recoil', 'gravity')


def as_position(r):
    """A position (x, y, z) as a float array of shape (3,) [m]"""
    r = np.asarray(r, dtype=float)
    if r.shape != (3,):
        raise ValueError(f'A position must have three components (x, y, z), not shape {r.shape}')
    return r


class ForceElement:
    """
    Base class of the forces acting on one levitated particle (one element of `Trap.fields`)

    Subclasses implement `force`, and override the other methods where they can do better than the
    defaults:

    - `stiffness` defaults to central differences of `force` at t = 0, with step
      `finite_difference_step` (the same scheme as `Trap.Ωj_at_z_eq`)
    - `potential` raises NotImplementedError (non-conservative or not yet written)
    - `torque` and `noise` are zero, and `damping_rates` are zero

    Class attributes:

    - is_optical [False]: True for a light field that the particle polarises (a `Tweezer`). A
      `Trap`'s optical-only quantities (U_depth, α_eff, photon_recoil_heating_rates, z_eq, ...)
      use its first optical element and raise a ValueError if it has none.
    - is_conservative [False]: True if `force` = -∇`potential`
    - simulation_group ['field']: the slot in which `FreeSpaceSimulation` adds the force, one of
      'field', 'damping', 'kick', 'recoil', 'gravity' (see SIMULATION_GROUPS)
    - is_time_dependent [False]: True if `force` depends explicitly on t beyond switching on and off
      (a `PaulTrap`); `Trap.static_force` then uses the time-averaged `secular_force`
    - noise_in_simulation [False]: True if `FreeSpaceSimulation` adds the element's `noise` as extra
      white force noise (the tweezer's recoil noise is not added, as before; a `PaulTrap`'s electrode
      noise is)
    """

    is_optical = False
    is_conservative = False
    is_time_dependent = False
    noise_in_simulation = False
    simulation_group = 'field'

    @property
    def finite_difference_step(self):
        """
        Step of the central differences in the default `stiffness`
        [m]
        """
        return 1e-10

    @property
    def label(self):
        """A short name of the element, for messages and summaries"""
        return type(self).__name__

    def force(self, r, t, particle):
        """
        Force on the particle at position r = (x, y, z) and time t
        [N]
        """
        raise NotImplementedError(f'{type(self).__name__} does not implement force(r, t, particle)')

    def secular_force(self, r, particle):
        """
        Time-averaged force at r, used for static quantities such as `Trap.find_equilibrium`
        [N]

        For an element that is not time dependent this is force(r, 0, particle).
        """
        return self.force(r, 0.0, particle)

    def max_time_step(self, particle=None):
        """
        Largest time step of a time-domain simulation that resolves this element, or None
        [s]

        None by default: a static element sets no limit.
        """

    def torque(self, r, t, particle, orientation=None):
        """
        Torque on the particle about its centre of mass
        [N·m]

        Zero by default: a sphere of isotropic material feels no torque from these elements.
        Elements acting on anisotropic particles override it.
        """
        return np.zeros(3)

    def torque_stiffness(self, particle, orientation=None, r0=None):
        """
        Torque stiffness K_ij = -∂τ_i/∂θ_j for small rotations θ about the lab axes, at `orientation`
        (default the particle's own) and position r0 (default the origin)
        [N·m/rad]

        Zero by default.
        """
        return np.zeros((3, 3))

    def torque_noise(self, particle, orientation=None, r0=None):
        """
        Two-sided torque spectral density matrix that the element adds, 〈τ_i(t) τ_j(t')〉 = S_ij δ(t - t')
        [N²·m²/Hz]

        Zero by default (the `Tweezer` gives its recoil torque noise).
        """
        return np.zeros((3, 3))

    def potential(self, r, particle):
        """
        Potential energy of the particle at r, for a conservative element
        [J]
        """
        raise NotImplementedError(
            f'{type(self).__name__} has no potential energy (it is not conservative, or it is not written yet)'
        )

    def stiffness(self, r0=None, particle=None):
        """
        Stiffness tensor K_ij = -∂F_i/∂r_j at r0 (default the origin)
        [N/m]

        By default from central differences of `force` at t = 0, with step h =
        `finite_difference_step`: K_ij = -(F_i(r0 + h ê_j) - F_i(r0 - h ê_j)) / 2h. For a
        conservative element K is the Hessian of the potential, and symmetric.
        """
        r0 = np.zeros(3) if r0 is None else as_position(r0)
        h = self.finite_difference_step
        K = np.zeros((3, 3))
        for j in range(3):
            step = np.zeros(3)
            step[j] = h
            F_plus = np.asarray(self.force(r0 + step, 0.0, particle), dtype=float)
            F_minus = np.asarray(self.force(r0 - step, 0.0, particle), dtype=float)
            K[:, j] = -(F_plus - F_minus) / (2*h)
        return K

    def noise(self, particle):
        """
        Two-sided force spectral density matrix that the element adds, 〈F_i(t) F_j(t')〉 = S_ij δ(t - t')
        [N²/Hz]

        Zero by default.
        """
        return np.zeros((3, 3))

    def damping_rates(self, particle=None):
        """
        Linear velocity-damping rates (Γx, Γy, Γz) of the element, i.e. a force -Γj·pj on axis j
        [2π·Hz]

        Zero by default (only `Feedback` damps).
        """
        return np.zeros(3)

    def __repr__(self):
        return f'{type(self).__name__}()'

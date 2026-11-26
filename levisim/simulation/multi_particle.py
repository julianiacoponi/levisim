"""
MultiParticleFreeSpaceSimulation: several free-space traps whose particles interact
"""

import numpy as np

from levisim.core.logger import logger
from levisim.core.units import nm
from levisim.physics.interactions.base import as_positions
from levisim.physics.interactions.base import attach_interaction_traps as _attach_traps
from levisim.physics.trap import Trap
from levisim.simulation.base import StochasticSimulation
from levisim.simulation.free_space import FreeSpaceSimulation

__all__ = [
    'MultiParticleFreeSpaceSimulation',
]


class MultiParticleFreeSpaceSimulation(StochasticSimulation):
    """
    Stochastic simulation of N particles, each in its own free-space `Trap`, coupled by pair forces

    The state is [x1, y1, z1, px1, py1, pz1, x2, ...] (6 entries per particle, the displacements
    from each trap's origin and the momenta). Each particle obeys the equations of motion of
    `FreeSpaceSimulation` in its own trap (its force elements, gas and feedback damping, noise), and
    the `interactions` add, once per drift evaluation, the forces at the laboratory-frame positions ::

        R_n = trap_n.position + (x_n, y_n, z_n)
        dp_n/dt ∋ Σ_interactions F_n(R_1, ..., R_N, t)

    The noises of different particles are independent (the noise matrix is block diagonal, four
    Wiener processes per particle, in the order of the traps), so a single particle without
    interactions consumes the random numbers exactly as `FreeSpaceSimulation` and gives the same
    trajectory, bit for bit.

    Keyword arguments (defaults in brackets), besides those of `StochasticSimulation`::

        traps [two default Traps]: the Traps, one per particle; their `position` places them
        interactions [[]]: levisim.physics.interactions objects (Coulomb, NewtonianGravity, Casimir, ...)
        X0 [zeros]: initial state, shape (N, 6) or (6N,)
        gravity [False], gravity_direction [(-1, 0, 0)]: the weight of every particle, as in
            FreeSpaceSimulation
        store_momenta [False]: also store the momentum traces 'px1', ...
        backend ['auto']: the numba kernel covers the traps that FreeSpaceSimulation's kernel
            covers (one Tweezer or override_Ω_dict, or any sum of the compiled 'field' elements:
            tweezer, PaulTrap, MagneticTrap, HarmonicPotential, UniformElectricField; feedback,
            photon recoil and gravity), but no kicks, and interactions that are power laws
            (`PowerLawPairInteraction`: Coulomb, NewtonianGravity, Casimir) plus at most one
            `OpticalBinding`; anything else runs the NumPy loop

    Traces are labelled like MultiTrap's dimensions: 'x1', 'y1', 'z1', 'x2', ... (self.traces).
    """

    def configure(self, **kwargs):
        """Reads the traps, interactions, initial state and the optional physics"""
        self.traps = list(kwargs.get('traps') or [Trap(), Trap(position=(0.0, 10e-6, 0.0))])
        self.num_particles = len(self.traps)
        self.interactions = list(kwargs.get('interactions') or [])
        X0 = kwargs.get('X0')
        X0 = np.zeros(6 * self.num_particles) if X0 is None else np.asarray(X0, dtype=float).reshape(-1)
        if X0.shape != (6 * self.num_particles,):
            raise ValueError(f'X0 must have shape ({self.num_particles}, 6) or ({6 * self.num_particles},)')
        self.X0 = X0
        self.gravity = bool(kwargs.get('gravity', False))
        self.gravity_direction_given = kwargs.get('gravity_direction', (-1, 0, 0))
        self.store_momenta = bool(kwargs.get('store_momenta', False))
        # one FreeSpaceSimulation per particle supplies that particle's own physics (it is configured,
        # never run), so the single-trap equations of motion are not written twice
        self.single_particle_systems = []
        for n, trap in enumerate(self.traps):
            single = object.__new__(FreeSpaceSimulation)
            single.rk_loop_debug = self.rk_loop_debug
            # the per-particle configure reads the time step and integrator (e.g. to warn when Δt does
            # not resolve a PaulTrap's rf drive)
            single.Δt = self.Δt
            single.RK_method = self.RK_method
            single.configure(trap=trap, X0=self.X0[6*n:6*n + 6], gravity=self.gravity,
                             gravity_direction=self.gravity_direction_given)
            self.single_particle_systems.append(single)
        self.particles = [trap.bead for trap in self.traps]
        self.trap_positions = np.array([trap.position for trap in self.traps])
        self.labels = [f'{axis}{n + 1}' for n in range(self.num_particles) for axis in 'xyz']
        for interaction in self.interactions:
            _attach_traps(interaction, self.traps)
        if self.interactions:
            # rejects overlapping particles before the integration starts
            self.interaction_forces(self.X0)
            logger.debug(
                'MultiParticleFreeSpaceSimulation: {} particles, interactions {}; initial pair forces {} N',
                self.num_particles, self.interactions, np.array2string(self.interaction_forces(self.X0), precision=4),
            )

    # ------------------------------------------------------------------------------------------
    # physics
    # ------------------------------------------------------------------------------------------
    def positions(self, X):
        """Laboratory-frame positions (N, 3) of the particles in the state X [m]"""
        return self.trap_positions + np.asarray(X).reshape(self.num_particles, 6)[:, :3]

    def interaction_forces(self, X, t=0.0):
        """Sum of the interactions' forces on every particle, shape (N, 3) [N]"""
        positions = as_positions(self.positions(X), self.num_particles)
        F = np.zeros((self.num_particles, 3))
        for interaction in self.interactions:
            F = F + interaction.forces(positions, self.particles, t)
        return F

    def initial_state(self):
        """Initial state [x1, y1, z1, px1, py1, pz1, x2, ...]"""
        return self.X0

    def drift(self, t, X):
        """
        Each particle's FreeSpaceSimulation.dynamics_vector, plus the interaction forces on the
        momentum rows
        [m/s, N]
        """
        dynamics = np.concatenate([
            single.dynamics_vector(t, X[6*n:6*n + 6]) for n, single in enumerate(self.single_particle_systems)
        ])
        if self.interactions:
            dynamics.reshape(self.num_particles, 6)[:, 3:] += self.interaction_forces(X, t)
        return dynamics

    def prepare_numpy_loop(self):
        """Resolves every particle's force dispatch once for the run (FreeSpaceSimulation.prepare_numpy_loop)"""
        for single in self.single_particle_systems:
            single.prepare_numpy_loop()

    def release_numpy_loop(self):
        """Drops the per-run force dispatch of every particle"""
        for single in self.single_particle_systems:
            single.release_numpy_loop()

    def noise_matrix(self):
        """
        Block-diagonal noise matrix, one FreeSpaceSimulation block per particle: 6 × 4, plus one column
        per principal axis of an element's simulated noise (e.g. a PaulTrap's electrode noise) [·]
        """
        blocks = [single.noise_matrix() for single in self.single_particle_systems]
        G = np.zeros((6 * self.num_particles, sum(block.shape[1] for block in blocks)))
        column = 0
        for n, block in enumerate(blocks):
            G[6*n:6*n + 6, column:column + block.shape[1]] = block
            column += block.shape[1]
        return G

    def noise_spectral_densities(self):
        """The four force spectral densities of every particle, in order [N²/Hz]"""
        return np.concatenate([single.noise_spectral_densities() for single in self.single_particle_systems])

    def record(self, X):
        """Stores the position traces 'x1', 'y1', 'z1', 'x2', ... (and momenta if asked for)"""
        self.traces = {}
        for n in range(self.num_particles):
            for index, axis in enumerate('xyz'):
                label = f'{axis}{n + 1}'
                values = self.add_detection_floor(X[:, 6*n + index], label)
                self.traces[label] = self.make_trace(values, label, 'm')
            if self.store_momenta:
                for index, axis in enumerate(('px', 'py', 'pz')):
                    label = f'{axis}{n + 1}'
                    self.traces[label] = self.make_trace(X[:, 6*n + 3 + index], label, 'kg·m/s')

    def compiled_integrate(self, integrator, G, S, X0, time_array):
        """The numba time loop (levisim.simulation.backends.numba_multi_particle), or None"""
        reason = self._compiled_kernel_unsupported_reason(integrator)
        if reason:
            logger.debug('MultiParticleFreeSpaceSimulation: NumPy loop instead of numba ({})', reason)
            return None
        from levisim.simulation.backends.numba_multi_particle import (
            integrate_multi_particle,
        )
        return integrate_multi_particle(self, integrator, G, S, X0, time_array)

    def _compiled_kernel_unsupported_reason(self, integrator):
        """Why the compiled kernel cannot reproduce this configuration ('' if it can)"""
        from levisim.physics.interactions.central import PowerLawPairInteraction
        from levisim.physics.interactions.optical_binding import OpticalBinding
        cls = type(self)
        for name in ('drift', 'noise_matrix', 'noise_spectral_densities', 'initial_state', 'interaction_forces', 'positions'):
            if getattr(cls, name) is not getattr(MultiParticleFreeSpaceSimulation, name):
                return f'{cls.__name__} overrides {name}'
        for n, single in enumerate(self.single_particle_systems):
            if single.trap.kicks:
                return f'trap {n + 1} has kicks'
            reason = single._compiled_kernel_unsupported_reason(integrator, single.noise_matrix(), single.X0)
            if reason:
                return f'trap {n + 1}: {reason}'
        bindings = 0
        for interaction in self.interactions:
            if type(interaction) is OpticalBinding:
                bindings += 1
                continue
            if not isinstance(interaction, PowerLawPairInteraction):
                return f'no compiled kernel for the {interaction.label} interaction'
            if type(interaction).forces is not PowerLawPairInteraction.forces:
                return f'{interaction.label} overrides forces'
        if bindings > 1:
            return 'the compiled kernel supports one OpticalBinding'
        return ''

    # ------------------------------------------------------------------------------------------
    # summary
    # ------------------------------------------------------------------------------------------
    def _info_body(self):
        lines = [f'\n    {self.num_particles} particles, interactions {self.interactions}']
        for label, trace in self.traces.items():
            if label[0] in 'xyz':
                lines.append(f'\n    σ{label} = {np.std(trace.values) / nm:.5f} nm')
        return ''.join(lines)

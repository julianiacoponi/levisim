"""
FreeSpaceSimulation: stochastic Runge-Kutta simulation of the motion of a bead in a free-space
tweezer trap (formerly `levisim.data.Simulation`)
"""

import numpy as np

from levisim.core.constants import kB
from levisim.core.units import (
    Hz,
    mbar,
    nm,
)
from levisim.physics.trap import Trap
from levisim.simulation.base import StochasticSimulation

__all__ = [
    'FreeSpaceSimulation',
]


class FreeSpaceSimulation(StochasticSimulation):
    """
    Tweezer-bead dynamics simulated using a Runge-Kutta (RK) stochastic numerical solver

    Integrates the Langevin equations of motion of a bead in a free-space Trap, ::

        dX = F(t, X) dt + G dW,   X = [x, y, z, px, py, pz]

    (see dynamics_vector and noise_matrix_RK4), and stores the recorded position traces
    self.x, self.y, self.z (Trace instances, also in self.traces).

    Keyword arguments (defaults in brackets)::

        trap [Trap()]: the Trap whose physics is simulated
        num_samples [1e6]: number of time steps
        Δt [0.1 μs]: time step [s]
        X0 [zeros]: initial [x, y, z, px, py, pz]
        (photon recoil is switched on with Trap(photon_recoil=True), see Trap.F_scattering)
        RK_method ['vector_4']: 'vector_4' (Kasdin stochastic RK4) or 'vector_2' (improved Euler)
        seed [None]: None uses numpy's legacy global RandomState (reproducible with np.random.seed);
            an int (or SeedSequence / Generator) uses np.random.default_rng(seed)
        gravity [False]: add the bead's weight m·g along gravity_direction [(-1, 0, 0), i.e. -x];
            it shifts the equilibrium by g/Ωj² (≈ 10 pm in x for the defaults)
        detection_floor [0]: one-sided white displacement-noise PSD [m²/Hz] added to the recorded
            traces only (a number, or a dict per axis)
        zero_mean [True]: subtract each trace's mean
        store_momenta [False]: also store px, py, pz traces [kg·m/s] (self.px etc., traces 'px', ...)
        store_radial [False]: also store the radial trace r = √(x² + y²) [m] (self.r, traces 'r')
        save, save_dir, save_description, save_fmt: optional pickling of the instance

    The time axis, seeding, noise generation, integrators and recording are shared with every
    levisim time-domain simulation, see `StochasticSimulation`.
    """

    state_labels = ('x', 'y', 'z', 'px', 'py', 'pz')

    def configure(self, **kwargs):
        """Reads the trap, initial state and the optional physics (gravity, stored traces)"""
        self.trap = kwargs.get('trap', Trap())
        self.X0 = kwargs.get('X0', np.array([0, 0, 0, 0, 0, 0]))  # initial conditions of [x, y, z, px, py, pz]
        # gravity, off by default; when on, it points along gravity_direction (default -x: in the
        # experiment the tweezer axis is horizontal and, per Antonio, gravity is along negative x)
        self.gravity = bool(kwargs.get('gravity', False))
        gravity_direction = np.asarray(kwargs.get('gravity_direction', (-1, 0, 0)), dtype=float)
        self.gravity_direction = gravity_direction / np.linalg.norm(gravity_direction)
        self.store_momenta = bool(kwargs.get('store_momenta', False))
        self.store_radial = bool(kwargs.get('store_radial', False))

    # ------------------------------------------------------------------------------------------
    # system hooks of StochasticSimulation
    # ------------------------------------------------------------------------------------------
    def initial_state(self):
        """Initial [x, y, z, px, py, pz] [m, m, m, kg·m/s, kg·m/s, kg·m/s]"""
        return self.X0

    def drift(self, t, X):
        """Deterministic part of the equations of motion, see dynamics_vector"""
        return self.dynamics_vector(t, X)

    def noise_matrix(self):
        """Constant noise matrix G, see noise_matrix_RK4 [·]"""
        return self.noise_matrix_RK4(0, self.X0)

    def noise_spectral_densities(self):
        """
        Two-sided force spectral densities of the four independent noises (thermal x, y, z and
        directed), (Sth, Sth, Sth, Sdir)
        [N²/Hz]
        """
        return np.array([self.trap.Sth, self.trap.Sth, self.trap.Sth, self.trap.Sdir])

    def record(self, X):
        """
        Stores the position traces x, y, z (plus the detection floor, if any), and optionally the
        momenta and the radial distance, as Trace instances in self.traces
        """
        self.position_values = {
            'x': self.add_detection_floor(X[:, 0], 'x'),
            'y': self.add_detection_floor(X[:, 1], 'y'),
            'z': self.add_detection_floor(X[:, 2], 'z'),
        }

        units = 'm'
        self.x, self.y, self.z = (
            self.make_trace(self.position_values[axis], axis, units)
            for axis in ('x', 'y', 'z')
        )

        self.traces = {
            'x': self.x,
            'y': self.y,
            'z': self.z,
        }

        if self.store_momenta:
            # the momenta are the integrated state: no detection floor is added to them
            self.px, self.py, self.pz = (
                self.make_trace(X[:, index], axis, 'kg·m/s')
                for index, axis in ((3, 'px'), (4, 'py'), (5, 'pz'))
            )
            self.traces.update({'px': self.px, 'py': self.py, 'pz': self.pz})

        if self.store_radial:
            # radial distance from the beam axis, from the recorded x, y (i.e. including any floor)
            self.r = self.make_trace(
                np.hypot(self.position_values['x'], self.position_values['y']), 'r', units,
            )
            self.traces['r'] = self.r

    # ------------------------------------------------------------------------------------------
    # physics
    # ------------------------------------------------------------------------------------------
    def dynamics_vector(self, t, X):
        """
        Deterministic part of the equations of motion, dX = F(t, X) dt + G dW

        ::

            F = (px/m, py/m, pz/m, Ftw,x - Γx px + F_kick,x, Ftw,y - Γy py + F_kick,y, Ftw,z - Γz pz)

        i.e. Hamilton's equations for ``H = |p|²/2m + Utw(r)``, plus the (gas + feedback) damping and
        any Kick impulses active at time t.
        [m/s, m/s, m/s, N, N, N]
        """
        x, y, z, px, py, pz = X
        m = self.trap.bead.m
        Γx, Γy, Γz = self.trap.Γx, self.trap.Γy, self.trap.Γz

        # allow for manually input frequencies
        if self.trap.override_Ω_dict:
            Ftwx = -m * (self.trap.override_Ω_dict['x'])**2 * x
            Ftwy = -m * (self.trap.override_Ω_dict['y'])**2 * y
            Ftwz = -m * (self.trap.override_Ω_dict['z'])**2 * z
        else:
            Ftwx, Ftwy, Ftwz = self.trap.Ftw(x, y, z)

        F_kick_x, F_kick_y = 0, 0
        for kick in self.trap.kicks:
            if kick.start <= t < kick.end:
                # K = p_kick / pth
                kick.p = kick.K * self.trap.gas.pth

                # since
                # p_kick = F0 * duration
                # then
                # F0 = p_kick / duration
                kick.F0 = kick.p / kick.duration

                F_kick_x = kick.F0 * np.cos(kick.θ)
                F_kick_y = kick.F0 * np.sin(kick.θ)

                kick.Fx = F_kick_x
                kick.Fy = F_kick_y

        dynamics = np.array([
            px / m,
            py / m,
            pz / m,
            Ftwx - Γx*px + F_kick_x,
            Ftwy - Γy*py + F_kick_y,
            Ftwz - Γz*pz,
        ])

        # opt-in forces, added only when enabled so that the default arithmetic is unchanged
        if self.trap.photon_recoil:
            # scattering force along the beam propagation axis, +z (see Trap.F_scattering)
            dynamics[5] += self.trap.F_photon_recoil(x, y, z)
        if self.gravity:
            # m g ≈ 16.5 aN for the default 1.68 fg bead, along gravity_direction (default -x)
            dynamics[3:] += self.F_gravity

        return dynamics

    def noise_matrix_RK4(self, t, X):
        """
        Noise matrix G of the equations of motion dX = F dt + G dW (used by every RK_method)

        ::

                          dynamics                  noise matrix (6 x 4)
            |  dx |   |     px/m     |       |   0      0      0      0     |
            |  dy |   |     py/m     |       |   0      0      0      0     |
            |  dz |   |     pz/m     |       |   0      0      0      0     ||   dWx  |
            |     | = |              | dt +  |                              ||   dWy  |
            | dpx |   | -dU/dx - Γpx |       |   1      0      0    cos(ψ)  ||   dWz  |
            | dpy |   | -dU/dy - Γpy |       |   0      1      0    sin(ψ)  ||  dWdir |
            | dpz |   | -dU/dz - Γpz |       |   0      0      1      0     |

        dWx,y,z and dWdir are four *independent* Wiener processes that carry their physical
        variances, i.e. the (two-sided) force spectral densities Sth and Sdir = β²Sth, whose square
        roots are the force noise floors fth and fdir. The same dWdir enters x and y, weighted by
        cos(ψ) and sin(ψ), which is what correlates the x, y motion (a non-zero Sxy).

        G does not depend on (t, X), i.e. the noise is additive, so the Itô and Stratonovich
        interpretations of the equations of motion coincide.
        [·]
        """
        # this means that there are separate Wiener processes applied to the momenta
        # stemming from the thermal and directed stochastic force noises
        ψ = self.trap.ψ
        return np.array([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 0, 0, np.cos(ψ)],
            [0, 1, 0, np.sin(ψ)],
            [0, 0, 1, 0],
        ])

    @property
    def F_gravity(self):
        """
        Weight of the bead along gravity_direction
        [N]
        """
        return self.trap.bead.weight * self.gravity_direction

    def temperature_via_variance(self, axis, Ωj=None):
        """
        Temperature estimated via the variance, using equipartition theorem

        ::

            ½mΩj²〈j²〉 = ½kb*Tj
            〈j²〉= Var(j)
            Therefore expecting T = (mΩ² / kb) * Var(j)
            j = x, y, z

        Ωj may be passed to account for how the simulated frequencies are shifted from the first
        order (harmonic) approximations; by default the Trap's harmonic Ωj is used.
        NOTE: includes the detection floor's variance if one was added to the trace.
        [K]
        """
        if Ωj is None:
            return self.trap.equipartition_factor[axis] * np.var(self.traces[axis].values)
        return self.trap.bead.m * Ωj**2 / kB * np.var(self.traces[axis].values)

    # ------------------------------------------------------------------------------------------
    # summary and saving
    # ------------------------------------------------------------------------------------------
    def _info_body(self):
        return (
            f'\n'
            f'\n    〈x(t)〉 = {np.mean(self.x.values) / nm:.5f} nm'
            f'\n    〈y(t)〉 = {np.mean(self.y.values) / nm:.5f} nm'
            f'\n    〈z(t)〉 = {np.mean(self.z.values) / nm:.5f} nm'
            f'\n'
            f'\n    Var(x) = {np.var(self.x.values) / nm**2:.5f} nm²'
            f'\n    Var(y) = {np.var(self.y.values) / nm**2:.5f} nm²'
            f'\n    Var(z) = {np.var(self.z.values) / nm**2:.5f} nm²'
            f'\n'
            f'\n    σx = {np.std(self.x.values) / nm:.5f} nm'
            f'\n    σy = {np.std(self.y.values) / nm:.5f} nm'
            f'\n    σz = {np.std(self.z.values) / nm:.5f} nm'
            f'\n'
            f'\n    Tx via Var(x) = {self.temperature_via_variance("x"):.5f} K'
            f'\n    Ty via Var(y) = {self.temperature_via_variance("y"):.5f} K'
            f'\n    Tz via Var(z) = {self.temperature_via_variance("z"):.5f} K'
            f'\n'
            f'\n    id(trap)={id(self.trap)}'
        )

    @property
    def save_name(self):
        """ Returns standardised naming format for the pickle binaries """
        kick_str = ''
        if self.trap.kicks:
            kick_str = f'_{len(self.trap.kicks)}_kicks'

        fmt = self.save_fmt
        save_name = (
            f'{self.save_description.replace(" ", "_")}'
            f'{kick_str}'
            f'_P{format(self.trap.gas.P / mbar, fmt)}'
            f'_fbx{format(self.trap.Γ_fbx / Hz, fmt)}'
            f'_t{format(self.Δt, fmt)}'
            f'_N{format(self.num_samples, fmt)}'
        )
        return save_name

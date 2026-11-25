"""
FreeSpaceSimulation: stochastic Runge-Kutta simulation of the motion of a bead in a free-space
tweezer trap (formerly `levisim.data.Simulation`)
"""

import warnings

import numpy as np

from levisim.core.constants import kB
from levisim.core.logger import logger
from levisim.core.units import (
    Hz,
    mbar,
    nm,
)
from levisim.physics.fields.elements import Gravity
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
            it shifts the equilibrium by g/Ωj² (≈ 10 pm in x for the defaults). Internally this is a
            `Gravity` force element (self.gravity_field); a trap may instead hold a Gravity element in
            its fields (not both)
        detection_floor [0]: one-sided white displacement-noise PSD [m²/Hz] added to the recorded
            traces only (a number, or a dict per axis)
        zero_mean [True]: subtract each trace's mean
        store_momenta [False]: also store px, py, pz traces [kg·m/s] (self.px etc., traces 'px', ...)
        store_radial [False]: also store the radial trace r = √(x² + y²) [m] (self.r, traces 'r')
        (a trap with a PaulTrap needs Δt ≤ 1/(20 f_rf), else a UserWarning; its electrode noise is
            added as extra white force noise, see field_noise_modes)
        backend ['auto']: 'auto', 'numpy' or 'numba' (compiled loop, `pip install levisim[fast]`),
            see StochasticSimulation; every option above has a compiled kernel, for a trap made of
            one tweezer plus feedback, kicks, photon recoil and gravity (or override_Ω_dict); traps
            with other force elements run the NumPy loop
        save, save_dir, save_description, save_fmt: optional pickling of the instance

    The forces are those of the trap's force elements (`Trap.fields`), added in a fixed order
    (see dynamics_vector). The time axis, seeding, noise generation, integrators and recording are
    shared with every levisim time-domain simulation, see `StochasticSimulation`.
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
        # the gravity keyword is a Gravity force element added to the trap's own elements
        # (built from the raw direction, which Gravity normalises exactly as above)
        self.gravity_field = Gravity(direction=gravity_direction) if self.gravity else None
        if self.gravity and self.trap.fields_in_group('gravity'):
            raise ValueError(
                'gravity=True would add gravity twice: the trap already holds a Gravity element in its fields'
            )
        self.store_momenta = bool(kwargs.get('store_momenta', False))
        self.store_radial = bool(kwargs.get('store_radial', False))
        self._check_time_step()

    def _check_time_step(self):
        """
        Warns if Δt is longer than an element's max_time_step (a PaulTrap's 1/(20 f_rf)): the
        integrators evaluate an explicitly time-dependent force at their stage times, which is accurate
        only when Δt resolves the drive
        """
        heun = getattr(self.RK_method, 'name', self.RK_method) == 'vector_2'
        for element in self.trap.fields:
            Δt_max = element.max_time_step(self.trap.bead)
            if Δt_max is None:
                continue
            if self.Δt > Δt_max:
                warnings.warn(
                    f'FreeSpaceSimulation: Δt = {self.Δt:.4g} s does not resolve {element.label}: use '
                    f'Δt ≤ {Δt_max:.4g} s (20 steps per rf period or more)',
                    UserWarning,
                    stacklevel=4,
                )
            elif heun and self.Δt > Δt_max / 5:
                warnings.warn(
                    f"FreeSpaceSimulation: RK_method='vector_2' (Heun, second order) shifts the secular "
                    f'frequency of a {element.label} by ~1.6% at 20 steps per rf period: use '
                    f"RK_method='vector_4' or Δt ≤ {Δt_max / 5:.4g} s (100 steps per period)",
                    UserWarning,
                    stacklevel=4,
                )

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
        """
        Constant noise matrix G, see noise_matrix_RK4 [·]; with elements whose noise enters the
        simulation (noise_in_simulation, e.g. a PaulTrap's electrode noise) one more column per
        principal axis of their summed force noise (see field_noise_modes)
        """
        G = self.noise_matrix_RK4(0, self.X0)
        modes = self.field_noise_modes()
        if modes is None:
            return G
        _, directions = modes
        extra = np.vstack([np.zeros((3, directions.shape[1])), directions])
        return np.hstack([G, extra])

    def noise_spectral_densities(self):
        """
        Two-sided force spectral densities of the four independent noises (thermal x, y, z and
        directed), (Sth,x, Sth,y, Sth,z, Sdir) (Sth on every axis for a sphere, Trap.Sth_j for a
        non-spherical particle whose gas damping differs between axes), followed by those of the
        elements' simulated noise, if any (see field_noise_modes)
        [N²/Hz]
        """
        Sth = self.trap.Sth_j
        S = np.array([Sth['x'], Sth['y'], Sth['z'], self.trap.Sdir])
        modes = self.field_noise_modes()
        if modes is None:
            return S
        return np.concatenate([S, modes[0]])

    def field_noise_modes(self):
        """
        The force noise of the elements with noise_in_simulation (a PaulTrap's electrode noise), as
        independent white noises: (spectral densities λ_k [N²/Hz], unit directions as the columns of a
        3×K array) from the eigen-decomposition of the summed two-sided 3×3 matrix; None if there is
        none, which leaves G and S (and so the random-number stream) exactly as without such elements
        """
        elements = [element for element in self.trap.fields if element.noise_in_simulation]
        if not elements:
            return None
        S = sum(np.asarray(element.noise(self.trap.bead), dtype=float) for element in elements)
        if not np.any(S):
            return None
        λ, U = np.linalg.eigh(0.5 * (S + S.T))
        keep = λ > 1e-12 * np.max(np.abs(λ))
        return λ[keep], U[:, keep]

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

    def compiled_integrate(self, integrator, G, S, X0, time_array):
        """
        The numba time loop (levisim.simulation.backends.numba_free_space), or None when this
        configuration must use the NumPy loop: a subclass that overrides the physics methods, a Trap
        or Tweezer subclass that overrides the forces, a custom integrator, or rk_loop_debug
        """
        reason = self._compiled_kernel_unsupported_reason(integrator, G, X0)
        if reason:
            logger.debug('FreeSpaceSimulation: NumPy loop instead of numba ({})', reason)
            return None
        from levisim.simulation.backends.numba_free_space import integrate_free_space
        return integrate_free_space(self, integrator, G, S, X0, time_array)

    def _compiled_kernel_unsupported_reason(self, integrator, G, X0):
        """Why the compiled kernel cannot reproduce this configuration ('' if it can)"""
        from levisim.physics.tweezer import Tweezer
        from levisim.simulation.integrators import Heun, KasdinRK4
        cls = type(self)
        for name in ('dynamics_vector', 'drift', 'noise_matrix', 'noise_matrix_RK4',
                     'noise_spectral_densities', 'initial_state', 'F_gravity'):
            if getattr(cls, name) is not getattr(FreeSpaceSimulation, name):
                return f'{cls.__name__} overrides {name}'
        reason = self._compiled_kernel_unsupported_fields()
        if reason:
            return reason
        checks = [(self.trap, Trap, ('Ftw', 'Utw', 'F_scattering', 'F_photon_recoil', 'field_force',
                                     'kick_force', 'element_force'))]
        if self.trap.has_optical_field:
            checks.append((self.trap.tweezer, Tweezer, ('w', 'wx', 'wy', 'wz', 'ellipse', 'E_field_amplitude')))
        for obj, base, names in checks:
            for name in names:
                if getattr(type(obj), name) is not getattr(base, name):
                    return f'{type(obj).__name__} overrides {name}'
        if type(integrator) not in (KasdinRK4, Heun):
            return f'no compiled kernel for {integrator!r}'
        if self.rk_loop_debug:
            return 'rk_loop_debug logs every step'
        if np.shape(G) != (6, 4) or len(X0) != 6:
            return 'unexpected state or noise dimensions'
        return ''

    def _compiled_kernel_unsupported_fields(self):
        """
        Why the trap's force elements are outside the compiled kernel's fixed set ('' if inside)

        The kernel knows one Gaussian-beam tweezer (or override_Ω_dict in its place), any Feedback
        elements (through Γx, Γy, Γz), kicks, one PhotonRecoil of that tweezer, and gravity. A generic
        compiled path would need each element to provide a numba-compilable force with its parameters
        packed into an array (see the PR notes of the force-element refactor).
        """
        from levisim.physics.fields.elements import Feedback, Gravity, PhotonRecoil
        from levisim.physics.kick import Kick
        from levisim.physics.tweezer import Tweezer
        trap = self.trap
        field_elements = trap.fields_in_group('field')
        single_tweezer = len(field_elements) == 1 and isinstance(field_elements[0], Tweezer)
        if not trap.override_Ω_dict and not single_tweezer:
            names = ', '.join(element.label for element in field_elements) or 'none'
            return f'no compiled kernel for the field elements [{names}] (only one Tweezer)'
        for element in trap.fields:
            if element.simulation_group == 'field':
                continue
            if type(element) not in (Feedback, Kick, PhotonRecoil, Gravity):
                return f'no compiled kernel for the {element.label} element'
        recoils = trap.fields_of_type(PhotonRecoil)
        if recoils and (len(recoils) > 1 or recoils[0].tweezer is not trap.tweezer):
            return 'the compiled kernel supports one PhotonRecoil of the trap tweezer only'
        return ''

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

        With force elements (`Trap.fields`) the momentum rows are assembled in this fixed order, by
        each element's simulation_group: the sum of the 'field' forces (Ftw above, in the order of
        fields; replaced by -m Ωj² j with override_Ω_dict), minus the gas + 'damping' (Feedback)
        damping Γj·pj, plus the 'kick' force; then the 'recoil' force is added to dpz/dt and last the
        'gravity' force. This is the order of the single-tweezer code, so its numbers are unchanged.
        """
        x, y, z, px, py, pz = X
        m = self.trap.bead.m
        Γx, Γy, Γz = self.trap.Γx, self.trap.Γy, self.trap.Γz
        # the trap's elements by role (cached until the list of elements changes)
        plan = self.trap._element_plan

        # allow for manually input frequencies
        if self.trap.override_Ω_dict:
            Ftwx = -m * (self.trap.override_Ω_dict['x'])**2 * x
            Ftwy = -m * (self.trap.override_Ω_dict['y'])**2 * y
            Ftwz = -m * (self.trap.override_Ω_dict['z'])**2 * z
        elif plan['single_tweezer']:
            Ftwx, Ftwy, Ftwz = self.trap.Ftw(x, y, z)
        else:
            # the sum of the 'field' elements' forces, in the order of Trap.fields
            Ftwx, Ftwy, Ftwz = self.trap.field_force(x, y, z, t)

        F_kick_x, F_kick_y = 0, 0
        if plan['kicks']:
            # as always, the last kick active at t wins (Trap.kick_force)
            F_kick_x, F_kick_y = self.trap.kick_force(t)

        dynamics = np.array([
            px / m,
            py / m,
            pz / m,
            Ftwx - Γx*px + F_kick_x,
            Ftwy - Γy*py + F_kick_y,
            Ftwz - Γz*pz,
        ])

        # opt-in forces, added only when enabled so that the default arithmetic is unchanged
        if plan['recoils']:
            # scattering force along the beam propagation axis, +z (see Trap.F_scattering)
            dynamics[5] += self.trap.F_photon_recoil(x, y, z)
        if self.gravity_field is not None or 'gravity' in plan['groups']:
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
    def gravity_elements(self):
        """
        The Gravity force elements acting in this simulation: the one of the gravity keyword (if on)
        and any 'gravity' elements of the trap's fields
        """
        trap_gravity = self.trap._element_plan['groups'].get('gravity', [])
        if self.gravity_field is None:
            return trap_gravity
        return [self.gravity_field, *trap_gravity]

    @property
    def F_gravity(self):
        """
        Total force of the gravity elements (the bead's weight m·g along gravity_direction with
        gravity=True); with no gravity element, the weight along gravity_direction that gravity=True
        would add
        [N]
        """
        elements = self.gravity_elements
        if not elements:
            return self.trap.bead.weight * self.gravity_direction
        # gravity elements are uniform: the position does not enter
        F = elements[0].force(None, 0.0, self.trap.bead)
        for element in elements[1:]:
            F = F + element.force(None, 0.0, self.trap.bead)
        return F

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

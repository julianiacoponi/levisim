import numpy as np

from os import makedirs

from levisim.core.units import (
    megasamples,
    nm,
    ms, μs,
    Hz,
    mbar,
)
from levisim.core.constants import kB
from levisim.core.logger import logger
from levisim.core.utils import save_variable_to_pickle, generate_time_array
from levisim.physics.trap import Trap
from levisim.data.trace import Trace

__all__ = [
    'Simulation',
]

class Simulation:
    """
    Tweezer-bead dynamics simulated using a Runge-Kutta (RK) stochastic numerical solver

    Integrates the Langevin equations of motion of a bead in a free-space Trap,
        dX = F(t, X) dt + G dW,   X = [x, y, z, px, py, pz]
    (see dynamics_vector and noise_matrix_RK4), and stores the recorded position traces
    self.x, self.y, self.z (Trace instances, also in self.traces).

    Keyword arguments (defaults in brackets):
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
    """

    def __init__(
        self,
        **kwargs,
    ):
        self.debug = kwargs.pop('debug', False)
        self.rk_loop_debug = kwargs.pop('rk_loop_debug', False)

        self.num_samples = int(kwargs.get('num_samples', 1*megasamples))
        self.Δt = float(kwargs.get('Δt', 0.1*μs))
        self.trap = kwargs.get('trap', Trap())
        self.X0 = kwargs.get('X0', np.array([0, 0, 0, 0, 0, 0]))  # initial conditions of [x, y, z, px, py, pz]
        self.RK_method = kwargs.get('RK_method', 'vector_4')
        self.detection_floor = kwargs.get('detection_floor', 0)  # one-sided PSD, m²/Hz
        # gravity, off by default; when on, it points along gravity_direction (default -x: in the
        # experiment the tweezer axis is horizontal and, per Antonio, gravity is along negative x)
        self.gravity = bool(kwargs.get('gravity', False))
        gravity_direction = np.asarray(kwargs.get('gravity_direction', (-1, 0, 0)), dtype=float)
        self.gravity_direction = gravity_direction / np.linalg.norm(gravity_direction)
        self.seed = kwargs.get('seed', None)
        self.store_momenta = bool(kwargs.get('store_momenta', False))
        self.store_radial = bool(kwargs.get('store_radial', False))
        self.rng = None if self.seed is None else np.random.default_rng(self.seed)
        self.zero_mean = kwargs.get('zero_mean', True)  # subtract the mean of the time traces from all values

        self.simulate_dynamics()

        self.save = kwargs.get('save', False)
        self.save_dir = kwargs.get('save_dir', 'pickle')
        self.save_description = kwargs.get('save_description', 'default')
        self.save_fmt = kwargs.get('save_fmt', '.0e')

        if self.save:
            makedirs(self.save_dir, exist_ok=True)
            save_variable_to_pickle(self, self.save_dir + '/' + self.save_name)

    @property
    def info(self):
        print(
            f'Simulation(id={id(self)}'
            f'\n    RK_method = {self.RK_method}'
            f'\n    num_samples = {self.num_samples:.5e}'
            f'\n    Δt = {self.Δt / μs:.5f} μs'
            f'\n'
            f'\n    time_length = {self.time_length / ms:.5f} ms'
            f'\n    sample_rate = {self.sample_rate / 1e6:.5f} MHz'
            f'\n    Nyquist_freq = {self.Nyquist_frequency / 1e6:.5f} MHz'
            f'\n    zero_mean = {self.zero_mean}'
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
            f'\n)'
        )
        return

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

    def dynamics_vector(self, t, X):
        """
        Deterministic part of the equations of motion, dX = F(t, X) dt + G dW

        F = (px/m, py/m, pz/m, Ftw,x - Γx px + F_kick,x, Ftw,y - Γy py + F_kick,y, Ftw,z - Γz pz)
        i.e. Hamilton's equations for H = |p|²/2m + Utw(r), plus the (gas + feedback) damping and
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

    def standard_normals(self, shape):
        """
        Standard normal random numbers N(0, 1) of the given shape, drawn from the simulation's
        random number generator

        - `seed=None` (default): numpy's legacy global RandomState (np.random.normal), so runs
          seeded with np.random.seed(...) reproduce the numbers of earlier levisim versions
        - otherwise: a numpy Generator, np.random.default_rng(seed)

        Drawing an array from the legacy RandomState gives the same numbers, in the same (C) order,
        as drawing them one at a time, so pre-generating per chunk does not change the stream.
        [·]
        """
        if self.rng is None:
            return np.random.normal(size=shape)
        return self.rng.standard_normal(shape)

    def _noise_chunks(self, num_steps, num_stages, num_noises, chunk_size=2**15):
        """
        Yields (start_index, normals) where normals has shape (n, num_stages, num_noises), n ≤ chunk_size,
        covering num_steps time steps in order. Keeps memory bounded for long runs.
        """
        for start in range(0, num_steps, chunk_size):
            n = min(chunk_size, num_steps - start)
            yield start, self.standard_normals((n, num_stages, num_noises))

    def simulate_dynamics(self):
        """
        Integrates the stochastic equations of motion dX = F(t, X) dt + G dW for X = [x, y, z, px, py, pz]
        and stores the position traces (plus the detection floor, if any) as Trace instances

        RK_method:
        - 'vector_4' (default): Kasdin's 4-stage stochastic Runge-Kutta method (see below)
        - 'vector_2': improved Euler (Heun) scheme for additive noise, strong order 1

        The random numbers for each chunk of time steps are pre-generated (see standard_normals).
        """
        if self.RK_method not in ('vector_4', 'vector_2'):
            if self.RK_method == 'scalar_2':
                raise ValueError(
                    "RK_method='scalar_2' has been removed: it drove x, y, z and the directed force with a "
                    "single Wiener process, i.e. perfectly correlated thermal noise. Use 'vector_2' (improved "
                    "Euler with independent Wiener processes) or 'vector_4' (Kasdin RK4)."
                )
            raise ValueError(f"Unknown RK_method={self.RK_method!r}; use 'vector_4' or 'vector_2'")

        # simulation array - stores [x, y, z, px, py, pz] at each time point
        # +1 to include the initial condition X0
        X = np.zeros([self.num_samples + 1, len(self.X0)])
        X[0] = self.X0

        Δt = self.Δt
        dynamics = self.dynamics_vector  # a.k.a. deterministic
        # the noise is additive (G is independent of t, X), so build G once
        G = self.noise_matrix_RK4(0, X[0])
        # the time array has num_samples + 1 entries; the penultimate step fills the final entry
        time_array = self.time_array[:-1]
        logger.info(f'Simulating {self.num_samples} time steps ({self.RK_method})...')

        if self.RK_method == 'vector_2':
            # NOTE: https://en.wikipedia.org/wiki/Runge–Kutta_method_(SDE)#Variation_of_the_Improved_Euler_is_flexible
            # (A. J. Roberts, arXiv:1210.0933). For dX = a dt + b dW the scheme is
            #     K1 = a(t, X) Δt + (ΔW - S√Δt) b(t, X)
            #     K2 = a(t + Δt, X + K1) Δt + (ΔW + S√Δt) b(t + Δt, X + K1)
            #     X[n + 1] = X[n] + (K1 + K2) / 2,   S = ±1 with equal probability
            # Here b = G·√S_k is constant (additive noise), so the ±S√Δt terms cancel exactly in K1 + K2
            # and S need not be drawn: the scheme reduces to Heun's method with one ΔW per step.
            # ΔW holds four independent increments: thermal x, y, z ~ N(0, Sth Δt) and directed ~ N(0, Sdir Δt)
            σ = np.sqrt(Δt * np.array([self.trap.Sth, self.trap.Sth, self.trap.Sth, self.trap.Sdir]))
            for start, normals in self._noise_chunks(self.num_samples, 1, 4):
                dP = (normals[:, 0, :] * σ) @ G.T  # momentum kicks [N·s], shape (n, 6)
                for k in range(len(normals)):
                    n = start + k
                    t = time_array[n]
                    K1 = dynamics(t, X[n])*Δt + dP[k]
                    K2 = dynamics(t + Δt, X[n] + K1)*Δt + dP[k]
                    X[n + 1] = X[n] + 0.5 * (K1 + K2)

        elif self.RK_method == 'vector_4':
            # coefficients are from: doi: 10.1109/5.381848
            # 1995-05 -- Kasdin -- Discrete Simulation of Colored Noise and Stochastic Processes and 1\f^a Power Law Noise Generation

            # noise variance weights, q1, q2, q3, q4
            q = [3.99956364361748, 1.64524970733585, 1.59330355118722, 0.26330006501868]

            # stage 2 weight
            a21 = 0.66667754298442
            # stage 3 weights
            a3 = np.array([0.63493935027993, 0.00342761715422])
            a31, a32 = a3
            # stage 4 weights
            a4 = np.array([-2.32428921184321, 2.69723745129487, 0.29093673271592])
            a41, a42, a43 = a4
            # solution weights
            α = np.array([0.25001351164789, 0.67428574806272, -0.00831795169360, 0.08401868181222])
            α1, α2, α3, α4 = α

            # qi's help adjust the variance of the noise for each step!
            # After Eq. 80, (h = Δt here)
            # "The IID noise, Wi, has variance equal to qi.Q/h, where Q is the spectral density of the input white noise"

            # The force noises have units N/√Hz
            # The definition asks for spectral densities i.e. N²/Hz, so:
            #     | Sth  |
            # Q = |      |
            #     | Sdir |
            # the angle for x, y is applied in the noise matrix

            # these are the standard deviations [N] of the stage forces, applied for each step of the RK method
            # Σ[i, k] = √(q_i S_k / Δt), stage i = 1..4, noise k = (thermal x, y, z, directed)
            S = np.array([self.trap.Sth, self.trap.Sth, self.trap.Sth, self.trap.Sdir])
            Σ = np.sqrt(np.outer(q, S) / Δt)

            if self.debug:
                # here, 1/√Δt is applied to the dW's outside the loop, then Δt in the loop
                # overall: √Δt
                logger.debug(f'Thermal force noise st.devs √(qi.Q/Δt) * Δt: {Σ[:, 0] * Δt}')
                logger.debug(f'Directed force noise st.devs √(qi.Q/Δt) * Δt: {Σ[:, 3] * Δt}')

            # '4th order' = 4 steps of calculation per loop!
            for start, normals in self._noise_chunks(self.num_samples, 4, 4):
                # stage forces [N]; equivalent to the per-step dWi = Σ[i] * N(0, 1) of earlier versions,
                # drawn in the same order (dW1[0..3], dW2[0..3], dW3[0..3], dW4[0..3] per step)
                dW = normals * Σ
                for k in range(len(normals)):
                    n = start + k
                    t = time_array[n]
                    dW1, dW2, dW3, dW4 = dW[k]

                    # .dot() must be used to apply the noise matrix to the dW vector!
                    # step 1
                    K1 = (dynamics(t, X[n]) + G.dot(dW1))*Δt

                    # step 2
                    t2 = t + a21*Δt
                    X2 = X[n] + a21*K1
                    K2 = (dynamics(t2, X2) + G.dot(dW2))*Δt

                    # step 3
                    t3 = t + (a31 + a32)*Δt
                    X3 = X[n] + a31*K1 + a32*K2
                    K3 = (dynamics(t3, X3) + G.dot(dW3))*Δt

                    # step 4
                    t4 = t + (a41 + a42 + a43)*Δt
                    X4 = X[n] + a41*K1 + a42*K2 + a43*K3
                    K4 = (dynamics(t4, X4) + G.dot(dW4))*Δt

                    # solution
                    X[n + 1] = X[n] + α1*K1 + α2*K2 + α3*K3 + α4*K4

                    if self.rk_loop_debug:
                        logger.debug(
                            f'\n LOOP #{n + 1} ######################'
                            f'\n{dW[k]=}'
                            f'\n{K1=}\n{K2=}\n{K3=}\n{K4=}'
                            f'\n{X[n + 1]=}'
                        )

        self.X = X
        self.position_values = {
            'x': self.add_detection_floor(X[:, 0], 'x'),
            'y': self.add_detection_floor(X[:, 1], 'y'),
            'z': self.add_detection_floor(X[:, 2], 'z'),
        }

        start, end, units = 0, self.time_length, 'm'
        self.x, self.y, self.z = (
            Trace(
                self.position_values[axis],
                start, end, Δt,
                axis, units, self,
                zero_mean=self.zero_mean,
            )
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
                Trace(
                    X[:, index],
                    start, end, Δt,
                    axis, 'kg·m/s', self,
                    zero_mean=self.zero_mean,
                )
                for index, axis in ((3, 'px'), (4, 'py'), (5, 'pz'))
            )
            self.traces.update({'px': self.px, 'py': self.py, 'pz': self.pz})

        if self.store_radial:
            # radial distance from the beam axis, from the recorded x, y (i.e. including any floor)
            self.r = Trace(
                np.hypot(self.position_values['x'], self.position_values['y']),
                start, end, Δt,
                'r', units, self,
                zero_mean=self.zero_mean,
            )
            self.traces['r'] = self.r

        logger.info('Done!')

    def detection_floor_for(self, axis):
        """
        One-sided displacement PSD of the white detection (imprecision) noise for a given axis
        `detection_floor` may be a single number (all axes) or a dict {'x': ..., 'y': ..., 'z': ...}
        [m²/Hz]
        """
        if isinstance(self.detection_floor, dict):
            return float(self.detection_floor.get(axis, 0))
        return float(self.detection_floor or 0)

    def add_detection_floor(self, values, axis):
        """
        Adds white measurement noise to a *recorded* position trace (never to the integrated state)

        Convention: `detection_floor` G_floor is the one-sided PSD [m²/Hz] of the noise, i.e. the flat
        level it produces far from resonance in a (default, one-sided) SpectralDensityEstimate.
        A white sequence of variance σ² sampled every Δt has one-sided PSD G = 2σ²Δt, hence
        σ = √(G_floor / (2Δt)).
        [m]
        """
        G_floor = self.detection_floor_for(axis)
        if not G_floor:
            return values
        σ_floor = np.sqrt(G_floor / (2 * self.Δt))
        return values + σ_floor * self.standard_normals(values.shape)

    @property
    def F_gravity(self):
        """
        Weight of the bead along gravity_direction
        [N]
        """
        return self.trap.bead.weight * self.gravity_direction

    @property
    def time_length(self):
        """ Time length of the simulation (seconds) """
        return self.num_samples * self.Δt

    @property
    def min_Δf(self):
        """ Minimum frequency resolution the time trace is capable of producing (Hertz) """
        return 1 / self.time_length

    @property
    def sample_rate(self):
        """ Sample rate of the simulation - how many samples are taken per second? (samples/second) """
        return 1 / self.Δt

    @property
    def Nyquist_frequency(self):
        """
        The Nyquist frequency, a.k.a. the bandwidth of the simulation
        This will be the largest frequency in the spectral density estimate
        NOTE: https://en.wikipedia.org/wiki/Nyquist_frequency
        """
        return self.sample_rate / 2

    @property
    def time_array(self):
        """ Array of all the time steps in the simulation """
        return generate_time_array(0, self.time_length, self.num_samples)

    def temperature_via_variance(self, axis, Ωj=None):
        """
        Temperature estimated via the variance, using equipartition theorem
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

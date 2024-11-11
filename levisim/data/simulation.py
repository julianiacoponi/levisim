import numpy as np

from os import makedirs

from levisim.core.units import (
    Megasamples,
    nm,
    ms, μs,
    Hz,
    mbar,
)
from levisim.core.utils import save_variable_to_pickle, generate_time_array
from levisim.physics.physics import Physics
from levisim.data.trace import Trace


class Simulation:
    """ Tweezer-bead dynamics simulated using a Runge-Kutta (RK) stochastic numerical solver """

    def __init__(
        self,
        **kwargs,
    ):
        self.debug = kwargs.pop('debug', False)
        self.rk_loop_debug = kwargs.pop('rk_loop_debug', False)

        self.num_samples = int(kwargs.get('num_samples', 1*Megasamples))
        self.Δt = float(kwargs.get('Δt', 0.1*μs))
        self.physics = kwargs.get('physics', Physics())
        self.X0 = kwargs.get('X0', np.array([0, 0, 0, 0, 0, 0]))  # initial conditions of [x, y, z, px, py, pz]
        self.RK_method = kwargs.get('RK_method', 'vector_4')
        self.detection_floor = kwargs.get('detection_floor', 0)  # m²/Hz
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
            f'\n    id(physics)={id(self.physics)}'
            f'\n)'
        )
        return

    @property
    def save_name(self):
        """ Returns standardised naming format for the pickle binaries """
        kick_str = ''
        if self.physics.kicks:
            kick_str = f'_{len(self.physics.kicks)}_kicks'

        fmt = self.save_fmt
        save_name = (
            f'{self.save_description.replace(" ", "_")}'
            f'{kick_str}'
            f'_P{format(self.physics.gas.P / mbar, fmt)}'
            f'_fbx{format(self.physics.Γ_fbx / Hz, fmt)}'
            f'_t{format(self.Δt, fmt)}'
            f'_N{format(self.num_samples, fmt)}'
        )
        return save_name

    def dynamics_vector(self, t, X):
        """
        Vector governing the dynamics of the bead in the tweezer
        This is essentially the gradient of the Hamiltonian (dH) applied to the symplectic matrix J
        """
        x, y, z, px, py, pz = X
        m = self.physics.bead.m
        Γx, Γy, Γz = self.physics.Γx, self.physics.Γy, self.physics.Γz

        # allow for manually input frequencies
        if self.physics.override_Ω0:
            Ftwx = -m * (self.physics.override_Ω0_dict['x'])**2 * x
            Ftwy = -m * (self.physics.override_Ω0_dict['y'])**2 * y
            Ftwz = -m * (self.physics.override_Ω0_dict['z'])**2 * z
        else:
            Ftwx = self.physics.Ftwx(x, y, z)
            Ftwy = self.physics.Ftwy(x, y, z)
            Ftwz = self.physics.Ftwz(x, y, z)

        # F_photon_recoil = self.physics.F_photon_recoil

        F_kick_x, F_kick_y = 0, 0
        for kick in self.physics.kicks:
            if kick.start <= t < kick.end:
                # K = p_kick / pth
                kick.p = kick.K * self.physics.gas.pth

                # since
                # p_kick = F0 * duration
                # then
                # F0 = p_kick / duration
                kick.F0 = kick.p / kick.duration

                F_kick_x = kick.F0 * np.cos(kick.θ)
                F_kick_y = kick.F0 * np.sin(kick.θ)

                kick.Fx = F_kick_x
                kick.Fy = F_kick_y

        return np.array([
            px / m,
            py / m,
            pz / m,
            Ftwx - Γx*px + F_kick_x,
            Ftwy - Γy*py + F_kick_y,
            # photon recoil goes in -z direction!
            # Ftwz - Γz*pz - F_photon_recoil(x, y, z),
            Ftwz - Γz*pz,
        ])

        # TODO: could add gravity somewhere, it is roughly 10 aN for a 1fg mass!
        # - m*g,
        # Antonio says it's in the direction of negative x

    def noise_vector_RK2(self, t, X):
        """
        NOTE: 2nd order runge kutta method used here only considers *one* Wiener process
        THIS IS WRONG!
        *Two* Wiener processes are needed, one for the thermal gas, one for the directed force

                     dynamics                  noise vector
        |  dx |   |     px/m      |       |          0         |
        |  dy |   |     py/m      |       |          0         |
        |  dz |   |     pz/m      |       |          0         |
        |     | = |               | dt +  |                    | dW
        | dpx |   | -Γpx - dU/dx  |       |   Fth + Fd.cos(ψ)  |
        | dpy |   | -Γpy - dU/dy  |       |   Fth + Fd.sin(ψ)  |
        | dpz |   | -Γpz - dU/dz  |       |   Fth + (Fdz = 0)  |

        If doing *scalar* 2nd order, dW is a single normally distributed value N~(0, Δt), i.e. zero mean, variance Δt
        If doing *vector* 2nd order, dW is a vector of separately generated normally distributed values
        """
        Fth = self.physics.fth
        return np.array([
            0,
            0,
            0,
            (Fth + self.physics.Fdx),
            (Fth + self.physics.Fdy),
            (Fth + self.physics.Fdz),
        ])

    def noise_matrix_RK4(self, t, X):
        """
                      dynamics                  noise matrix (6 x 4)
        |  dx |   |     px/m     |       |   0      0      0      0     |
        |  dy |   |     py/m     |       |   0      0      0      0     |
        |  dz |   |     pz/m     |       |   0      0      0      0     ||   dWx  |
        |     | = |              | dt +  |                              ||   dWy  |
        | dpx |   | -dU/dx - Γpx |       |   1      0      0    cos(ψ)  ||   dWz  |
        | dpy |   | -dU/dy - Γpy |       |   0      1      0    sin(ψ)  ||  dWdir |
        | dpz |   | -dU/dz - Γpz |       |   0      0      1      0     |

        dWx,y,z and dWdir include their physical variances
        i.e. the force spectral densities Sth and Sdir,
        whose square roots are the standard deviations
        i.e. the force noise floors fth and fd
        """
        # this means that there are separate Wiener processes applied to the momenta
        # stemming from the thermal and directed stochastic force noises
        ψ = self.physics.ψ
        return np.array([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 0, 0, np.cos(ψ)],
            [0, 1, 0, np.sin(ψ)],
            [0, 0, 1, 0],
        ])

    def noise_matrix_RK4_with_detection_floor(self, t, X):
        """
        The same as noise_matrix_RK4, but with 3 more Wiener processes to model 3 detection noise floors
                      dynamics                  noise matrix (6 x 7)
        |  dx |   |     px/m     |       |   0      0      0      0     0     0     0     ||    dWx    |
        |  dy |   |     py/m     |       |   0      0      0      0     0     0     0     ||    dWy    |
        |  dz |   |     pz/m     |       |   0      0      0      0     0     0     0     ||    dWz    |
        |     | = |              | dt +  |                                                || dWx_floor |
        | dpx |   | -dU/dx - Γpx |       |   1      0      0      1     0     0   cos(ψ)  || dWy_floor |
        | dpy |   | -dU/dy - Γpy |       |   0      1      0      0     1     0   sin(ψ)  || dWz_floor |
        | dpz |   | -dU/dz - Γpz |       |   0      0      1      0     0     1     0     ||  dWdir    |

        dWx,y,z, dWx,y,z_floor and dWdir include their physical variances
        i.e. the force spectral densities Sth and Sdir,
        whose square roots are the standard deviations
        i.e. the force noise floors fth and fd
        """
        # this means that there are separate Wiener processes applied to the momenta
        # stemming from the thermal and directed stochastic force noises
        ψ = self.physics.ψ
        return np.array([
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 1, 0, 0, np.cos(ψ)],
            [0, 1, 0, 0, 1, 0, np.sin(ψ)],
            [0, 0, 1, 0, 0, 1, 0],
        ])

    def simulate_dynamics(self):
        """ Run a Runge-Kutta numerical method to solve the equations of motion """
        # simulation array - stores [x, y, z, px, py, pz] at each time point
        # +1 to include the initial condition X0
        X = np.zeros([self.num_samples + 1, len(self.X0)])
        X[0] = self.X0

        Δt = self.Δt
        # NOTE: https://en.wikipedia.org/wiki/Runge–Kutta_method_(SDE)#Variation_of_the_Improved_Euler_is_flexible
        if '2' in self.RK_method:
            a = self.dynamics_vector  # a.k.a. deterministic
            b = self.noise_vector_RK2  # a.k.a. stochastic
            𝛿 = np.sqrt(Δt)

            if self.debug:
                print(self.RK_method)
                print(f'Thermal variance = {𝛿 * self.physics.Sth}')
                print(f'Directed variance (x) = {𝛿 * self.physics.Sdirx}')
                print(f'Directed variance (y) = {𝛿 * self.physics.Sdiry}')
                print(f'Directed variance (z) = {𝛿 * self.physics.Sdirz}')
                print(f'Combined variance (x) = {𝛿 * (self.physics.Sth + self.physics.Sdirx)}')
                print(f'Combined variance (y) = {𝛿 * (self.physics.Sth + self.physics.Sdiry)}')
                print(f'Combined variance (z) = {𝛿 * (self.physics.Sth + self.physics.Sdirz)}')

            for n, t in enumerate(self.time_array):
                # generate random numbers on the fly
                if self.RK_method == 'scalar_2':
                    dW = np.random.normal()

                if self.RK_method == 'vector_2':
                    dW = np.array([
                        # the dx, dy, dz equations don't have any noise
                        # so don't bother to generate random numbers for them
                        0, 0, 0,
                        np.random.normal(),
                        np.random.normal(),
                        np.random.normal(),
                    ])

                S = np.random.choice([+1, -1])
                # '2nd order' = 2 steps of calculation per loop!
                # step 1
                K1 = a(t, X[n])*Δt + 𝛿*(dW - S)*b(t, X[n])

                # step 2
                t2 = t + Δt
                X2 = X[n] + K1
                K2 = a(t2, X2)*Δt + 𝛿*(dW + S)*b(t2, X2)

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

            dynamics = self.dynamics_vector  # a.k.a. determinisitc
            noises = self.noise_matrix_RK4  # a.k.a. stochastic

            # qi's help adjust the variance of the noise for each step!
            # After Eq. 80, (h = Δt here)
            # "The IID noise, Wi, has variance equal to qi.Q/h, where Q is the spectral density of the input white noise"

            # The force noises have units N/√Hz
            # The definition asks for spectral densities i.e. N²/Hz, so:
            #     | Sth  |
            # Q = |      |
            #     | Sdir |
            # the angle for x, y is applied in the noise matrix

            # these are the standard deviations to be applied for each step of the RK method
            Σth1, Σth2, Σth3, Σth4 = (np.sqrt(qi * self.physics.Sth / Δt) for qi in q)
            Σdir1, Σdir2, Σdir3, Σdir4 = (np.sqrt(qi * self.physics.Sdir() / Δt) for qi in q)

            # if self.detection_floor:
            #     noises = self.noise_matrix_RK4_with_detection_floor
            #     Σfl1, Σfl2, Σfl3, Σfl4 = (np.sqrt(qi * self.detection_floor / Δt) for qi in q)

            if self.debug:
                # in the RK2 methods, 𝛿 = √Δt is applied in the loop
                # here, 1/√Δt is applied to the dW's outside the loop, then Δt in the loop
                # overall: √Δt
                print(
                    f'\nThermal force noise st.devs √(qi.Q/Δt) * Δt:'
                    f'\n{Σth1*Δt=}'
                    f'\n{Σth2*Δt=}'
                    f'\n{Σth3*Δt=}'
                    f'\n{Σth4*Δt=}'
                )
                print(
                    f'\nDirected force noise st.devs √(qi.Q/Δt) * Δt:'
                    f'\n{Σdir1*Δt=}'
                    f'\n{Σdir2*Δt=}'
                    f'\n{Σdir3*Δt=}'
                    f'\n{Σdir4*Δt=}'
                )

            # '4th order' = 4 steps of calculation per loop!
            print('Simulating time step...')
            # the penultimate time step loop fills the final time step value in the array
            for n, t in enumerate(self.time_array[:-1]):
                # progress bar in 10% steps
                if not n % (self.num_samples // 10):
                    print(f'{n} of {self.num_samples}')

                # generate the random numbers in the loop
                # if self.detection_floor:
                #     # dW must include 7 independent noises (3 thermal, 3 detection, 1 directional)
                #     dW1 = np.array([
                #         Σth1 * np.random.normal(),
                #         Σth1 * np.random.normal(),
                #         Σth1 * np.random.normal(),
                #         Σfl1 * np.random.normal(),
                #         Σfl1 * np.random.normal(),
                #         Σfl1 * np.random.normal(),
                #         Σdir1 * np.random.normal(),
                #     ])
                #     dW2 = np.array([
                #         Σth2 * np.random.normal(),
                #         Σth2 * np.random.normal(),
                #         Σth2 * np.random.normal(),
                #         Σfl2 * np.random.normal(),
                #         Σfl2 * np.random.normal(),
                #         Σfl2 * np.random.normal(),
                #         Σdir2 * np.random.normal(),
                #     ])
                #     dW3 = np.array([
                #         Σth3 * np.random.normal(),
                #         Σth3 * np.random.normal(),
                #         Σth3 * np.random.normal(),
                #         Σfl3 * np.random.normal(),
                #         Σfl3 * np.random.normal(),
                #         Σfl3 * np.random.normal(),
                #         Σdir3 * np.random.normal(),
                #     ])
                #     dW4 = np.array([
                #         Σth4 * np.random.normal(),
                #         Σth4 * np.random.normal(),
                #         Σth4 * np.random.normal(),
                #         Σfl4 * np.random.normal(),
                #         Σfl4 * np.random.normal(),
                #         Σfl4 * np.random.normal(),
                #         Σdir4 * np.random.normal(),
                #     ])

                # else:
                dW1 = np.array([
                    Σth1 * np.random.normal(),
                    Σth1 * np.random.normal(),
                    Σth1 * np.random.normal(),
                    Σdir1 * np.random.normal(),
                ])
                dW2 = np.array([
                    Σth2 * np.random.normal(),
                    Σth2 * np.random.normal(),
                    Σth2 * np.random.normal(),
                    Σdir2 * np.random.normal(),
                ])
                dW3 = np.array([
                    Σth3 * np.random.normal(),
                    Σth3 * np.random.normal(),
                    Σth3 * np.random.normal(),
                    Σdir3 * np.random.normal(),
                ])
                dW4 = np.array([
                    Σth4 * np.random.normal(),
                    Σth4 * np.random.normal(),
                    Σth4 * np.random.normal(),
                    Σdir4 * np.random.normal(),
                ])


                # .dot() must be used to apply the noise matrix to the dW vector!
                # step 1
                K1 = (dynamics(t, X[n]) + noises(t, X[n]).dot(dW1))*Δt

                # step 2
                t2 = t + a21*Δt
                X2 = X[n] + a21*K1
                K2 = (dynamics(t2, X2) + noises(t2, X2).dot(dW2))*Δt

                # step 3
                t3 = t + (a31 + a32)*Δt
                X3 = X[n] + a31*K1 + a32*K2
                K3 = (dynamics(t3, X3) + noises(t3, X3).dot(dW3))*Δt

                # step 4
                t4 = t + (a41 + a42 + a43)*Δt
                X4 = X[n] + a41*K1 + a42*K2 + a43*K3
                K4 = (dynamics(t4, X4) + noises(t4, X4).dot(dW4))*Δt

                # solution
                X[n + 1] = X[n] + α1*K1 + α2*K2 + α3*K3 + α4*K4

                # TODO: make detection_floor work
                # add a fixed variance, random amount to the x, y, z coordinates
                if self.detection_floor:
                    Σfl = np.sqrt(self.detection_floor / Δt)
                    X[n + 1][0] += Σfl * np.random.normal()
                    X[n + 1][1] += Σfl * np.random.normal()
                    X[n + 1][2] += Σfl * np.random.normal()

                if self.rk_loop_debug:
                    print(
                        f'\n LOOP #{n + 1} ######################'
                        f'\n{dW1=}'
                        f'\n{dW2=}'
                        f'\n{dW3=}'
                        f'\n{dW4=}'

                        f'\n stage 1 ------------------------'
                        f'\n{t=}'
                        f'\n{X[n]=}'
                        f'\n{dynamics(t, X[n])=}'
                        f'\n{noises(t, X[n])=}'
                        f'\n{K1=}'

                        f'\n stage 2 ------------------------'
                        f'\n{t2=}'
                        f'\n{X2=}'
                        f'\n{dynamics(t2, X2)=}'
                        f'\n{noises(t2, X2)=}'
                        f'\n{K2=}'

                        f'\n stage 3 ------------------------'
                        f'\n{t3=}'
                        f'\n{X3=}'
                        f'\n{dynamics(t3, X3)=}'
                        f'\n{noises(t3, X3)=}'
                        f'\n{K3=}'

                        f'\n stage 4 ------------------------'
                        f'\n{t4=}'
                        f'\n{X4=}'
                        f'\n{dynamics(t4, X4)=}'
                        f'\n{noises(t4, X4)=}'
                        f'\n{K4=}'

                        f'\n\n RESULT ------------------------'
                        f'\n{X[n + 1]=}'
                    )
                    print(f'{Σfl=}')

        start, end, units = 0, self.time_length, 'm'
        self.x, self.y, self.z = (
            Trace(
                X[:, 0],
                start, end, Δt,
                'x', units, self,
                zero_mean=self.zero_mean,
            ),
            Trace(
                X[:, 1],
                start, end, Δt,
                'y', units, self,
                zero_mean=self.zero_mean,
            ),
            Trace(
                X[:, 2],
                start, end, Δt,
                'z', units, self,
                zero_mean=self.zero_mean,
            ),
        )

        # NOTE: ignore saving the momenta, not needed
        # start, end, units = 0, self.time_length, 'kg.m/s'
        # self.px, self.py, self.pz = (
        #     Trace(
        #         X[:, 3],
        #         start, end, Δt,
        #         'px', units, self,
        #         zero_mean=self.zero_mean,
        #     ),
        #     Trace(
        #         X[:, 4],
        #         start, end, Δt,
        #         'py', units, self,
        #         zero_mean=self.zero_mean,
        #     ),
        #     Trace(
        #         X[:, 5],
        #         start, end, Δt,
        #         'pz', units, self,
        #         zero_mean=self.zero_mean,
        #     ),
        # )


        self.traces = {
            'x': self.x,
            'y': self.y,
            'z': self.z,
            # 'px': self.px,
            # 'py': self.py,
            # 'pz': self.pz,
        }

        print('Done!')

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
        """
        # accounts for how the simulated frequencies are shifted from the first order approximations
        if Ωj is None:
            Ωj = self.physics.Ω0(axis)
        return self.physics.equipartition_factor(axis, Ωj) * np.var(self.traces[axis].values)

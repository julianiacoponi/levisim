"""
Stochastic integrators for additive-noise Langevin equations, dX = F(t, X) dt + G dW

Each integrator advances a state vector X with a user-supplied drift function F(t, X), a constant
noise matrix G (shape (state dimension, number of noises)) and a vector S of the noises' two-sided
spectral densities, so that dW_k has variance S_k dt. The random numbers come from a
`noise_chunks(num_steps, num_stages, num_noises)` generator (see
`StochasticSimulation._noise_chunks`), which yields standard normals in blocks. An integrator
declares how many normals it consumes per noise and per step (`num_stages`), so any simulation that
provides F, G and S can use any integrator without knowing its internals.

These are the schemes of levisim's free-space `Simulation`, moved here unchanged: the arithmetic is
written exactly as before, so seeded trajectories are bit-identical to earlier versions.

- `KasdinRK4` ('vector_4'): Kasdin's four-stage stochastic Runge-Kutta method
- `Heun` ('vector_2'): improved Euler (Heun) scheme for additive noise, strong order 1
"""

import numpy as np

from levisim.core.logger import logger

__all__ = [
    'INTEGRATORS',
    'Heun',
    'Integrator',
    'KasdinRK4',
    'get_integrator',
]


class Integrator:
    """
    Base class of the stochastic integrators

    Subclasses define

    - `name`: the `RK_method` string that selects them
    - `num_stages`: standard normals drawn per noise and per step
    - `stage_offsets`: the times t + c Δt (as the fractions c) at which the drift is evaluated
      within a step, used e.g. to tell which time-dependent forces were active
    - `noise_std(S, Δt)`: the standard deviations that scale the standard normals
    - `integrate(...)`: the time loop
    """
    name = None
    num_stages = None
    stage_offsets = ()

    def noise_std(self, S, Δt):
        """Standard deviations applied to the standard normals of one step [units of √S·√Hz or √S/√s]"""
        raise NotImplementedError

    def integrate(self, drift, G, S, X0, time_array, Δt, noise_chunks, *, debug=False, trace_steps=False, project=None):
        """
        Integrates dX = drift(t, X) dt + G dW from X0 over the time points in time_array

        Arguments::

            drift: callable (t, X) -> dX/dt (deterministic part)
            G: constant noise matrix, shape (len(X0), len(S))
            S: two-sided spectral densities of the independent noises dW_k
            X0: initial state
            time_array: start times of the steps (one per step) [s]
            Δt: time step [s]
            noise_chunks: generator function (num_steps, num_stages, num_noises) ->
                (start_index, standard normals of shape (n, num_stages, num_noises))
            debug, trace_steps: extra DEBUG / per-step TRACE logging
            project: optional callable X -> X applied to the state after every full step (a
                projection onto a constraint manifold, e.g. renormalising the quaternion of a
                rigid body); None (the default) leaves the steps untouched

        Returns X of shape (len(time_array) + 1, len(X0)), with X[0] = X0.
        """
        raise NotImplementedError

    def __repr__(self):
        return f'{type(self).__name__}(name={self.name!r})'


def _allocate(X0, num_steps):
    """State array of num_steps + 1 rows (the first is the initial condition X0)"""
    X = np.zeros([num_steps + 1, len(X0)])
    X[0] = X0
    return X


class KasdinRK4(Integrator):
    """
    Kasdin's four-stage stochastic Runge-Kutta method ('vector_4')

    Coefficients from N. J. Kasdin, Proceedings of the IEEE 83, 802-827 (1995),
    doi:10.1109/5.381848 (after Eq. 80, with h = Δt): the IID stage noises W_i have variance
    q_i Q/h, where Q is the spectral density of the input white noise.
    """
    name = 'vector_4'
    num_stages = 4

    # noise variance weights, q1, q2, q3, q4
    q = (3.99956364361748, 1.64524970733585, 1.59330355118722, 0.26330006501868)
    # stage 2 weight
    a21 = 0.66667754298442
    # stage 3 weights
    a31, a32 = np.array([0.63493935027993, 0.00342761715422])
    # stage 4 weights
    a41, a42, a43 = np.array([-2.32428921184321, 2.69723745129487, 0.29093673271592])
    # solution weights
    α1, α2, α3, α4 = np.array([0.25001351164789, 0.67428574806272, -0.00831795169360, 0.08401868181222])

    @property
    def stage_offsets(self):
        return (0.0, self.a21, self.a31 + self.a32, self.a41 + self.a42 + self.a43)

    def noise_std(self, S, Δt):
        """
        Σ[i, k] = √(q_i S_k / Δt): standard deviations of the stage forces, stage i = 1..4, noise k
        [√S/√s]
        """
        return np.sqrt(np.outer(self.q, S) / Δt)

    def integrate(self, drift, G, S, X0, time_array, Δt, noise_chunks, *, debug=False, trace_steps=False, project=None):
        num_steps = len(time_array)
        X = _allocate(X0, num_steps)
        a21, a31, a32, a41, a42, a43 = self.a21, self.a31, self.a32, self.a41, self.a42, self.a43
        α1, α2, α3, α4 = self.α1, self.α2, self.α3, self.α4
        Σ = self.noise_std(S, Δt)

        if debug:
            # here, 1/√Δt is applied to the dW's outside the loop, then Δt in the loop
            # overall: √Δt
            logger.debug(f'Thermal force noise st.devs √(qi.Q/Δt) * Δt: {Σ[:, 0] * Δt}')
            logger.debug(f'Directed force noise st.devs √(qi.Q/Δt) * Δt: {Σ[:, -1] * Δt}')

        # '4th order' = 4 steps of calculation per loop!
        for start, normals in noise_chunks(num_steps, self.num_stages, len(S)):
            # stage forces; equivalent to the per-step dWi = Σ[i] * N(0, 1) of earlier versions,
            # drawn in the same order (dW1[0..K-1], dW2[...], dW3[...], dW4[...] per step)
            dW = normals * Σ
            for k in range(len(normals)):
                n = start + k
                t = time_array[n]
                dW1, dW2, dW3, dW4 = dW[k]

                # .dot() must be used to apply the noise matrix to the dW vector!
                # step 1
                K1 = (drift(t, X[n]) + G.dot(dW1))*Δt

                # step 2
                t2 = t + a21*Δt
                X2 = X[n] + a21*K1
                K2 = (drift(t2, X2) + G.dot(dW2))*Δt

                # step 3
                t3 = t + (a31 + a32)*Δt
                X3 = X[n] + a31*K1 + a32*K2
                K3 = (drift(t3, X3) + G.dot(dW3))*Δt

                # step 4
                t4 = t + (a41 + a42 + a43)*Δt
                X4 = X[n] + a41*K1 + a42*K2 + a43*K3
                K4 = (drift(t4, X4) + G.dot(dW4))*Δt

                # solution
                X[n + 1] = X[n] + α1*K1 + α2*K2 + α3*K3 + α4*K4
                if project is not None:
                    X[n + 1] = project(X[n + 1])

                if trace_steps:
                    logger.trace(
                        'RK loop #{}: dW={}, K1={}, K2={}, K3={}, K4={}, X={}',
                        n + 1, dW[k], K1, K2, K3, K4, X[n + 1],
                    )
        return X


class Heun(Integrator):
    """
    Improved Euler (Heun) scheme for additive noise ('vector_2'), strong order 1

    Following A. J. Roberts, arXiv:1210.0933 (see
    https://en.wikipedia.org/wiki/Runge–Kutta_method_(SDE)#Variation_of_the_Improved_Euler_is_flexible),
    for dX = a dt + b dW ::

        K1 = a(t, X) Δt + (ΔW - S√Δt) b(t, X)
        K2 = a(t + Δt, X + K1) Δt + (ΔW + S√Δt) b(t + Δt, X + K1)
        X[n + 1] = X[n] + (K1 + K2) / 2,   S = ±1 with equal probability

    With b = G·√S_k constant (additive noise) the ±S√Δt terms cancel exactly in K1 + K2, so S need
    not be drawn: the scheme reduces to Heun's method with one ΔW per step.
    """
    name = 'vector_2'
    num_stages = 1
    stage_offsets = (0.0, 1.0)

    def noise_std(self, S, Δt):
        """σ_k = √(S_k Δt): standard deviations of the increments ΔW_k of one step [√S·√s]"""
        return np.sqrt(Δt * np.asarray(S))

    def integrate(self, drift, G, S, X0, time_array, Δt, noise_chunks, *, debug=False, trace_steps=False, project=None):
        num_steps = len(time_array)
        X = _allocate(X0, num_steps)
        σ = self.noise_std(S, Δt)
        for start, normals in noise_chunks(num_steps, self.num_stages, len(S)):
            dP = (normals[:, 0, :] * σ) @ G.T  # noise increments of the state, shape (n, len(X0))
            for k in range(len(normals)):
                n = start + k
                t = time_array[n]
                K1 = drift(t, X[n])*Δt + dP[k]
                K2 = drift(t + Δt, X[n] + K1)*Δt + dP[k]
                X[n + 1] = X[n] + 0.5 * (K1 + K2)
                if project is not None:
                    X[n + 1] = project(X[n + 1])
                if trace_steps:
                    logger.trace('Heun loop #{}: dP={}, K1={}, K2={}, X={}', n + 1, dP[k], K1, K2, X[n + 1])
        return X


INTEGRATORS = {
    KasdinRK4.name: KasdinRK4(),
    Heun.name: Heun(),
}


def get_integrator(RK_method):
    """
    The Integrator registered under `RK_method` ('vector_4' or 'vector_2'); an Integrator instance
    is returned unchanged
    """
    if isinstance(RK_method, Integrator):
        return RK_method
    if RK_method in INTEGRATORS:
        return INTEGRATORS[RK_method]
    if RK_method == 'scalar_2':
        raise ValueError(
            "RK_method='scalar_2' has been removed: it drove x, y, z and the directed force with a "
            "single Wiener process, i.e. perfectly correlated thermal noise. Use 'vector_2' (improved "
            "Euler with independent Wiener processes) or 'vector_4' (Kasdin RK4)."
        )
    raise ValueError(f"Unknown RK_method={RK_method!r}; use 'vector_4' or 'vector_2'")

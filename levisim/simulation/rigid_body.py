"""
RigidBodySimulation: translation and rotation of one rigid particle in a free-space trap

The state is ::

    X = [x, y, z,  px, py, pz,  qw, qx, qy, qz,  L1, L2, L3]

with the lab-frame position r and momentum p, the orientation quaternion q (body → lab, R = R(q))
and the angular momentum L_b in the BODY frame. The equations of motion are ::

    dr/dt   = p / m
    dp/dt   = F(r, q, t) - R Γ_cm Rᵀ p                      + ξ_p
    dq/dt   = ½ q ⊗ (0, ω),            ω = I⁻¹ L_b
    dL_b/dt = Rᵀ τ(r, q, t) - ω × L_b - Γ_rot L_b           + ξ_L

with the body-frame gas friction tensors Γ_cm, Γ_rot (Martinetz et al., Phys. Rev. E 97, 052112
(2018), from levisim.physics.shapes). The angular momentum is kept in the body frame, so that the
rotational friction and its diffusion tensor 2 kB T Γ_rot I are constant and the noise is additive
(constant G in dX = F dt + G dW, as the integrators of levisim.simulation.integrators need); the
rotational fluctuation-dissipation relation holds exactly. The translational momentum is kept in the
lab frame (a body-frame momentum would rotate with a spinning particle and force Δt ≪ 1/Ω_rot); its
noise is the lab-frame diffusion tensor at the INITIAL orientation, 2 kB T m R0 Γ_cm R0ᵀ, while the
friction follows the orientation. This is exact for a sphere and for librations about the initial
orientation; for a freely rotating anisotropic particle the translational noise lags the friction
by the anisotropy of Γ_cm (a few per cent for the shapes of levisim). After every step the
quaternion is renormalised (a projection method), so ‖q‖ = 1 to round-off; the energy error of the
deterministic part is that of the integrator (fourth order for 'vector_4').
"""

import numpy as np

from levisim.core.constants import kB, ε0, π
from levisim.core.logger import logger
from levisim.core.units import μs
from levisim.physics.fields.elements import Feedback
from levisim.physics.particle import quaternion_to_rotation_matrix
from levisim.physics.rotation import finite_size_torque, rotation_vector_between
from levisim.physics.trap import Trap
from levisim.physics.tweezer import Tweezer
from levisim.simulation.base import StochasticSimulation

__all__ = [
    'RigidBodySimulation',
]


def _cross(a, b):
    """a × b for 3-vectors (faster than np.cross for single vectors)"""
    return np.array([
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0],
    ])


def _rotation_matrix(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ])


def _quaternion_rate(q, ω):
    """½ q ⊗ (0, ω) for a body-frame angular velocity ω"""
    w, x, y, z = q
    a, b, c = ω
    return 0.5 * np.array([
        -x*a - y*b - z*c,
        w*a + y*c - z*b,
        w*b - x*c + z*a,
        w*c + x*b - y*a,
    ])


def _noise_factor(S):
    """(eigenvalues, eigenvectors) of a symmetric positive semi-definite noise matrix S"""
    S = 0.5 * (np.asarray(S, dtype=float) + np.asarray(S, dtype=float).T)
    values, vectors = np.linalg.eigh(S)
    values = np.where(values > 0, values, 0.0)
    return values, vectors


class RigidBodySimulation(StochasticSimulation):
    """
    Stochastic translation and rotation of one rigid particle in a `Trap` (levisim.simulation.rigid_body)

    Keyword arguments (defaults in brackets)::

        trap [Trap(particle=...)]: the Trap; its particle's `orientation` is the initial orientation
            and the reference of the recorded angles. Its elements give the forces and torques: a
            `Tweezer` (gradient force with the orientation-dependent polarisability ê*·α_eff·ê,
            alignment and spin torque, any polarisation), `Feedback` (cold damping of the lab
            momenta), and every other element through its own `force` and `torque`
        num_samples [1e6], Δt [0.01 μs], RK_method ['vector_4'], seed, detection_floor, zero_mean:
            as `StochasticSimulation`. Kasdin's stochastic Runge–Kutta scheme ('vector_4') is slightly
            dissipative on an undamped oscillator, at a rate growing as (ΩΔt)³: for a libration at
            Ω/2π = 800 kHz the energy decays at ≈ 6 × 10³ s⁻¹ with 25 steps per period and ≈ 6 s⁻¹
            with 250, so resolve the fastest libration with a few hundred steps when its energy
            matters (heating and feedback cooling rates)
        X0 [None]: initial state (13 entries, see the module docstring); by default the particle at
            rest at the origin in its orientation
        r0, v0, ω0 [zeros]: initial position [m], lab velocity [m/s] and lab angular velocity
            [rad/s], used when X0 is not given
        gas_damping [True]: rotational and translational gas friction and their thermal noise
        gas_noise [True]: with gas_damping, include the thermal noise (False keeps only the friction,
            e.g. to see a driven rotation's deterministic spin-up)
        recoil [False]: the recoil force noise (`Tweezer.noise`) and torque noise
            (`Tweezer.recoil_torque_noise`) of the tweezer light, both evaluated at the initial
            orientation and kept fixed in the body frame (exact for librations about it)
        power_modulation [None]: callable (t, X) -> η; the tweezer intensity is multiplied by 1 + η
            (parametric feedback, or a modulated laser)
        external_torque [None]: callable (t, X) -> lab-frame torque [N·m] added to the elements'
            (a modulated drive, or a feedback torque from a measured angle)
        external_force [None]: callable (t, X) -> lab-frame force [N]
        record_angular_velocity [False]: also record the lab angular velocity traces 'ωx', 'ωy', 'ωz'

    Built-in controls (they also run in the compiled loop, unlike the callables above)::

        parametric_feedback [None]: a dict, or a list of dicts, of parametric feedback on the tweezer
            intensity (van der Laan et al., Phys. Rev. Lett. 127, 123605 (2021)): with the measured
            angle s = ê·(R b̂), the lab component ê of a body axis b̂, the intensity is multiplied by
            1 + η, η = clip(gain·s·ṡ, ±max_modulation); a positive gain stiffens the trap while the angle grows and so cools it. Keys: gain [s], body_axis, lab_axis,
            max_modulation [0.5]
        torque_drive [None]: dict(amplitude=(τx, τy, τz) [N·m], ω [rad/s], phase [rad], offset [N·m]):
            a lab-frame torque offset + amplitude·cos(ω t + phase)
        angular_damping [None]: (Γx, Γy, Γz) [1/s]: a lab-frame cold-damping torque -Γj·L_lab,j
        finite_size_torque [trap.finite_size_torque]: the finite-size alignment torque of the extended
            particle in the trap's translational stiffness (`Trap.finite_size_tensors`, harmonic,
            evaluated at the trap centre; scaled with the tweezer intensity modulation)

    backend ['auto']: 'numba' compiles the loop for a trap of at most one `Tweezer` plus `Feedback`
    elements, with the built-in controls but no callables; anything else runs the NumPy loop.

    Traces (self.traces): 'x', 'y', 'z' [m] and 'α', 'β', 'γ' [rad], the rotation vector of the
    orientation relative to the initial one about the lab axes x, y, z (the librational coordinates
    of `MultiTrap`; meaningful for librations, not for a free rotor, whose angle grows without
    bound). Only the NumPy loop exists (backend='numba' falls back to it).
    """

    default_Δt = 0.01*μs
    state_labels = ('x', 'y', 'z', 'p1', 'p2', 'p3', 'qw', 'qx', 'qy', 'qz', 'L1', 'L2', 'L3')

    def configure(self, **kwargs):
        self.trap = kwargs.get('trap') or Trap()
        particle = self.trap.bead
        self.particle = particle
        self.m = float(particle.m)
        self.inertia = np.asarray(particle.inertia_tensor, dtype=float)
        self.inertia_inverse = np.linalg.inv(self.inertia)
        self.q_reference = np.array(particle.orientation, dtype=float)
        R0 = quaternion_to_rotation_matrix(self.q_reference)

        self.gas_damping = bool(kwargs.get('gas_damping', True))
        self.gas_noise = bool(kwargs.get('gas_noise', True))
        self.recoil = bool(kwargs.get('recoil', False))
        self.power_modulation = kwargs.get('power_modulation')
        self.external_torque = kwargs.get('external_torque')
        self.external_force = kwargs.get('external_force')
        self.record_angular_velocity = bool(kwargs.get('record_angular_velocity', False))
        parametric = kwargs.get('parametric_feedback')
        if parametric is None:
            parametric = []
        elif isinstance(parametric, dict):
            parametric = [parametric]
        self.parametric_feedback = np.zeros((len(parametric), 8))
        for index, entry in enumerate(parametric):
            unknown = set(entry) - {'gain', 'body_axis', 'lab_axis', 'max_modulation'}
            if unknown:
                raise ValueError(f'unknown parametric_feedback keys {sorted(unknown)}')
            b = np.asarray(entry.get('body_axis', (0, 0, 1)), dtype=float)
            e = np.asarray(entry.get('lab_axis', (1, 0, 0)), dtype=float)
            self.parametric_feedback[index] = [
                float(entry['gain']), *(b / np.linalg.norm(b)), *(e / np.linalg.norm(e)),
                float(entry.get('max_modulation', 0.5)),
            ]
        drive = kwargs.get('torque_drive')
        self.torque_drive = None
        if drive is not None:
            unknown = set(drive) - {'amplitude', 'ω', 'phase', 'offset'}
            if unknown:
                raise ValueError(f'unknown torque_drive keys {sorted(unknown)}')
            self.torque_drive = {
                'amplitude': np.asarray(drive.get('amplitude', np.zeros(3)), dtype=float),
                'ω': float(drive.get('ω', 0.0)),
                'phase': float(drive.get('phase', 0.0)),
                'offset': np.asarray(drive.get('offset', np.zeros(3)), dtype=float),
            }
        angular_damping = kwargs.get('angular_damping')
        self.angular_damping = None if angular_damping is None else np.asarray(angular_damping, dtype=float).reshape(3)
        self.finite_size_torque = bool(kwargs.get('finite_size_torque', getattr(self.trap, 'finite_size_torque', False)))
        if self.finite_size_torque:
            self._fs_k, _ = self.trap.finite_size_tensors
            from levisim.physics.rotation import second_moment_tensor
            self._fs_S_body = second_moment_tensor(self.particle.inertia_tensor) / self.particle.m
        else:
            self._fs_k = np.zeros((3, 3))
            self._fs_S_body = np.zeros((3, 3))

        # the elements by role
        self.tweezers = [element for element in self.trap.fields if isinstance(element, Tweezer)]
        self.feedbacks = [element for element in self.trap.fields if isinstance(element, Feedback)]
        self.other_elements = [
            element for element in self.trap.fields
            if not isinstance(element, (Tweezer, Feedback))
        ]
        self.Γ_feedback = np.zeros(3)
        for element in self.feedbacks:
            self.Γ_feedback = self.Γ_feedback + np.asarray(element.damping_rates(particle), dtype=float)

        # per tweezer: (tweezer, body-frame α_eff, Jones vector)
        self._optics = [
            (tweezer, particle.body_effective_polarisability_tensor(tweezer.k), np.asarray(tweezer.polarisation, dtype=complex))
            for tweezer in self.tweezers
        ]

        # gas friction (body frame) and its noise; for a sphere the translational rate is the
        # Trap's Γth (Epstein), for other shapes Martinetz's tensor. The translational noise is the
        # lab-frame tensor at the initial orientation (see the module docstring)
        if self.gas_damping:
            Γ_lab0 = np.asarray(self.trap.Γth_tensor, dtype=float)
            self.Γ_cm = R0.T @ Γ_lab0 @ R0
            self.Γ_rot = np.asarray(particle.rotational_damping_tensor(self.trap.gas), dtype=float)
            T = self.trap.gas.T if self.gas_noise else 0.0
            S_p = 2 * kB * T * self.m * Γ_lab0
            S_L = 2 * kB * T * self.Γ_rot @ self.inertia
        else:
            self.Γ_cm = np.zeros((3, 3))
            self.Γ_rot = np.zeros((3, 3))
            S_p = np.zeros((3, 3))
            S_L = np.zeros((3, 3))
        if self.recoil:
            for tweezer in self.tweezers:
                S_p = S_p + np.asarray(tweezer.noise(particle), dtype=float)
                S_L = S_L + R0.T @ tweezer.recoil_torque_noise(particle) @ R0
        self.momentum_noise_psd = S_p
        self.angular_momentum_noise_psd = S_L

        X0 = kwargs.get('X0')
        if X0 is None:
            r0 = np.asarray(kwargs.get('r0', np.zeros(3)), dtype=float)
            v0 = np.asarray(kwargs.get('v0', np.zeros(3)), dtype=float)
            ω0 = np.asarray(kwargs.get('ω0', np.zeros(3)), dtype=float)
            L_b = self.inertia @ (R0.T @ ω0)
            X0 = np.concatenate([r0, self.m * v0, self.q_reference, L_b])
        self.X0 = np.asarray(X0, dtype=float)
        if self.X0.shape != (13,):
            raise ValueError(f'X0 must have 13 entries (r, p, q, L_body), not shape {self.X0.shape}')
        logger.debug(
            'RigidBodySimulation of {} with {} tweezer(s), {} feedback, other elements {}',
            type(particle).__name__, len(self.tweezers), len(self.feedbacks), self.other_elements,
        )

    # ------------------------------------------------------------------------------------------
    # hooks of StochasticSimulation
    # ------------------------------------------------------------------------------------------
    def initial_state(self):
        return self.X0

    @staticmethod
    def project_state(X):
        """Renormalises the quaternion after every step"""
        X = X.copy()
        X[6:10] = X[6:10] / np.sqrt(X[6]**2 + X[7]**2 + X[8]**2 + X[9]**2)
        return X

    def forces_and_torques(self, t, X):
        """
        Lab-frame force F [N] and body-frame torque τ_b [N·m] on the particle in the state X at time t
        (from the elements and the external callables; no gas friction)
        """
        r = X[0:3]
        q = X[6:10]
        R = _rotation_matrix(q / np.sqrt(q @ q))
        F = np.zeros(3)
        τ_b = np.zeros(3)
        factor = 1.0
        if len(self.parametric_feedback):
            ω_lab = R @ (self.inertia_inverse @ X[10:13])
            for gain, bx, by, bz, ex, ey, ez, η_max in self.parametric_feedback:
                n = R @ np.array([bx, by, bz])
                e = np.array([ex, ey, ez])
                s = e @ n
                ṡ = e @ _cross(ω_lab, n)
                factor = factor + min(max(gain * s * ṡ, -η_max), η_max)
        if self.power_modulation is not None:
            factor = factor + float(self.power_modulation(t, X))
        for tweezer, α_body, e in self._optics:
            x, y, z = r
            amplitude2 = factor * tweezer.E_field_amplitude(x, y, z)**2
            u = R.T @ e  # the polarisation in the body frame
            v = α_body @ u
            # ê*·α_lab·ê = u*·α_body·u; the gradient force of levisim's Tweezer.force
            a = float(np.real(np.conj(u) @ v))
            U = -(a / 2) * amplitude2
            wx, wy, wz = tweezer.wx(z), tweezer.wy(z), tweezer.wz(z)
            ellipse = tweezer.ellipse(x, y, z)
            F = F + np.array([
                0.5 * (4 * x / wx**2) * U,
                0.5 * (4 * y / wy**2) * U,
                0.5 * (4 * z / wz**2) * U * (1 - 2 * ellipse),
            ])
            # τ_b = Rᵀ·[½Re(p* × E) - c Im(p* × p)] = ‖E‖²·[½Re(v* × u) - c Im(v* × v)], v = α_body u
            τ_b = τ_b + amplitude2 * (
                0.5 * np.real(_cross(np.conj(v), u)) - tweezer.k**3 / (12 * π * ε0) * np.imag(_cross(np.conj(v), v))
            )
        if self.other_elements:
            orientation = q / np.sqrt(q @ q)
            for element in self.other_elements:
                if getattr(element, 'orientation_dependent_force', False):
                    F = F + np.asarray(element.force(r, t, self.particle, orientation=orientation), dtype=float)
                else:
                    F = F + np.asarray(element.force(r, t, self.particle), dtype=float)
                τ = np.asarray(element.torque(r, t, self.particle, orientation=orientation), dtype=float)
                if np.any(τ):
                    τ_b = τ_b + R.T @ τ
        if self.finite_size_torque:
            τ_b = τ_b + factor * (R.T @ finite_size_torque(self._fs_k, R @ self._fs_S_body @ R.T))
        τ_lab = np.zeros(3)
        if self.torque_drive is not None:
            d = self.torque_drive
            τ_lab = τ_lab + d['offset'] + d['amplitude'] * np.cos(d['ω'] * t + d['phase'])
        if self.angular_damping is not None:
            τ_lab = τ_lab - self.angular_damping * (R @ X[10:13])
        if self.torque_drive is not None or self.angular_damping is not None:
            τ_b = τ_b + R.T @ τ_lab
        if self.external_force is not None:
            F = F + np.asarray(self.external_force(t, X), dtype=float)
        if self.external_torque is not None:
            τ_b = τ_b + R.T @ np.asarray(self.external_torque(t, X), dtype=float)
        return F, τ_b, R

    def drift(self, t, X):
        p = X[3:6]
        q = X[6:10]
        L_b = X[10:13]
        F, τ_b, R = self.forces_and_torques(t, X)
        ω = self.inertia_inverse @ L_b
        if self.feedbacks:
            F = F - self.Γ_feedback * p
        dX = np.empty(13)
        dX[0:3] = p / self.m
        dX[3:6] = F - R @ (self.Γ_cm @ (R.T @ p))
        dX[6:10] = _quaternion_rate(q, ω)
        dX[10:13] = τ_b - _cross(ω, L_b) - self.Γ_rot @ L_b
        return dX

    def compiled_integrate(self, integrator, G, S, X0, time_array):
        """The numba loop (levisim.simulation.backends.numba_rigid_body), or None when this configuration needs NumPy"""
        reason = self._compiled_kernel_unsupported_reason(integrator)
        if reason:
            from levisim.core.logger import logger as _logger
            _logger.debug('RigidBodySimulation runs the NumPy loop: {}', reason)
            return None
        from levisim.simulation.backends.numba_rigid_body import integrate_rigid_body
        return integrate_rigid_body(self, integrator, G, S, X0, time_array)

    def _compiled_kernel_unsupported_reason(self, integrator):
        if integrator.name not in ('vector_4', 'vector_2') or type(integrator).__module__ != 'levisim.simulation.integrators':
            return f'no kernel for the integrator {integrator!r}'
        if self.power_modulation is not None or self.external_torque is not None or self.external_force is not None:
            return 'callables (power_modulation, external_torque, external_force)'
        if self.other_elements:
            return f'force elements {self.other_elements}'
        if len(self.tweezers) > 1:
            return 'more than one tweezer'
        if type(self).drift is not RigidBodySimulation.drift or type(self).forces_and_torques is not RigidBodySimulation.forces_and_torques:
            return 'a subclass overrides the physics'
        if any(type(tweezer) is not Tweezer for tweezer in self.tweezers):
            return 'a Tweezer subclass'
        if self.rk_loop_debug:
            return 'rk_loop_debug'
        return None

    def noise_matrix(self):
        values_p, vectors_p = _noise_factor(self.momentum_noise_psd)
        values_L, vectors_L = _noise_factor(self.angular_momentum_noise_psd)
        self._noise_psd = np.concatenate([values_p, values_L])
        G = np.zeros((13, 6))
        G[3:6, 0:3] = vectors_p
        G[10:13, 3:6] = vectors_L
        return G

    def noise_spectral_densities(self):
        if not hasattr(self, '_noise_psd'):
            self.noise_matrix()
        return self._noise_psd

    def record(self, X):
        self.positions = X[:, 0:3]
        self.quaternions = X[:, 6:10]
        self.angles = rotation_vector_between(self.q_reference, self.quaternions)
        self.traces = {}
        for index, axis in enumerate('xyz'):
            values = self.add_detection_floor(X[:, index], axis)
            trace = self.make_trace(values, axis, 'm')
            setattr(self, axis, trace)
            self.traces[axis] = trace
        for index, axis in enumerate(('α', 'β', 'γ')):
            values = self.add_detection_floor(self.angles[:, index], axis)
            trace = self.make_trace(values, axis, 'rad')
            setattr(self, axis, trace)
            self.traces[axis] = trace
        if self.record_angular_velocity:
            ω_lab = self.angular_velocities_lab()
            for index, axis in enumerate(('ωx', 'ωy', 'ωz')):
                self.traces[axis] = self.make_trace(ω_lab[:, index], axis, 'rad/s')

    # ------------------------------------------------------------------------------------------
    # derived quantities of the integrated states
    # ------------------------------------------------------------------------------------------
    def rotation_matrices(self, X=None):
        """R(q) for every recorded state, shape (n, 3, 3)"""
        X = self.X if X is None else np.atleast_2d(X)
        q = X[:, 6:10] / np.linalg.norm(X[:, 6:10], axis=1)[:, None]
        w, x, y, z = q.T
        return np.stack([
            np.stack([1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)], axis=-1),
            np.stack([2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)], axis=-1),
            np.stack([2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)], axis=-1),
        ], axis=-2)

    def lab_directions(self, body_vector, X=None):
        """
        The lab-frame direction R v of a body-frame vector v for every state, shape (n, 3), e.g. the
        symmetry axis of a spheroid (v = (0, 0, 1)): its tilt is the libration even while the
        particle spins freely about that axis, when the rotation vector of the traces mixes the two
        [·]
        """
        return np.einsum('nij,j->ni', self.rotation_matrices(X), np.asarray(body_vector, dtype=float))

    def angular_velocities_body(self, X=None):
        """Body-frame angular velocity ω = I⁻¹ L_b of every state, shape (n, 3) [rad/s]"""
        X = self.X if X is None else np.atleast_2d(X)
        return X[:, 10:13] @ self.inertia_inverse.T

    def angular_velocities_lab(self, X=None):
        """Lab-frame angular velocity R ω of every state, shape (n, 3) [rad/s]"""
        X = self.X if X is None else np.atleast_2d(X)
        return np.einsum('nij,nj->ni', self.rotation_matrices(X), self.angular_velocities_body(X))

    def angular_momenta_lab(self, X=None):
        """Lab-frame angular momentum R L_b of every state, shape (n, 3) [kg·m²/s]"""
        X = self.X if X is None else np.atleast_2d(X)
        return np.einsum('nij,nj->ni', self.rotation_matrices(X), X[:, 10:13])

    def rotational_kinetic_energies(self, X=None):
        """½ L_b·I⁻¹L_b of every state [J]"""
        X = self.X if X is None else np.atleast_2d(X)
        return 0.5 * np.einsum('ni,ni->n', X[:, 10:13], self.angular_velocities_body(X))

    def translational_kinetic_energies(self, X=None):
        """‖p‖²/2m of every state [J]"""
        X = self.X if X is None else np.atleast_2d(X)
        return 0.5 * np.einsum('ni,ni->n', X[:, 3:6], X[:, 3:6]) / self.m

    def quaternion_norm_error(self, X=None):
        """max abs(‖q‖ - 1) over the states [·]"""
        X = self.X if X is None else np.atleast_2d(X)
        return float(np.max(np.abs(np.linalg.norm(X[:, 6:10], axis=1) - 1)))

    def _info_body(self):
        return (
            f'\n    particle = {type(self.particle).__name__}'
            f'\n    gas_damping = {self.gas_damping}, recoil = {self.recoil}'
            f'\n    tweezers = {len(self.tweezers)}, other elements = {self.other_elements}'
        )

    @property
    def save_name(self):
        return f'rigid_body_simulation_{self.num_samples}_samples'

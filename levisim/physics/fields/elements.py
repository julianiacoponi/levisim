"""
Force elements other than the optical tweezer: a harmonic potential, gravity, feedback cold damping
and the photon-recoil (scattering) force of a tweezer
"""

import numpy as np

from levisim.core.constants import g as g_earth
from levisim.core.units import kHz
from levisim.physics.fields.base import ForceElement, as_position

__all__ = [
    'Feedback',
    'Gravity',
    'HarmonicPotential',
    'PhotonRecoil',
]


class HarmonicPotential(ForceElement):
    """
    A harmonic potential about a centre c, U = ½ (r - c)·K·(r - c), so F = -K·(r - c)

    For tests and toy models, and as the linearised form of any trap. Keyword arguments:

    - stiffness: K, a number (isotropic), three numbers (Kx, Ky, Kz), or a symmetric 3×3 tensor [N/m]
    - centre [(0, 0, 0)]: the centre c [m]

    `HarmonicPotential.from_frequencies((Ωx, Ωy, Ωz), m)` builds K = m·diag(Ωx², Ωy², Ωz²).
    """

    is_conservative = True

    def __init__(self, stiffness, centre=(0.0, 0.0, 0.0)):
        K = np.asarray(stiffness, dtype=float)
        if K.ndim == 0:
            K = K * np.eye(3)
        elif K.shape == (3,):
            K = np.diag(K)
        if K.shape != (3, 3):
            raise ValueError(f'stiffness must be a number, three numbers or a 3×3 tensor, not shape {K.shape}')
        if not np.allclose(K, K.T, rtol=1e-12, atol=0):
            raise ValueError('a harmonic potential has a symmetric stiffness tensor')
        self.K = K
        self.centre = as_position(centre)

    @classmethod
    def from_frequencies(cls, Ω, m, centre=(0.0, 0.0, 0.0)):
        """
        The harmonic potential with frequencies Ω = (Ωx, Ωy, Ωz) [2π·Hz] for a particle of mass m [kg]
        """
        return cls(m * np.asarray(Ω, dtype=float)**2, centre=centre)

    def force(self, r, t, particle):
        """
        F = -K·(r - c)
        [N]
        """
        return -self.K @ (as_position(r) - self.centre)

    def potential(self, r, particle):
        """
        U = ½ (r - c)·K·(r - c)
        [J]
        """
        d = as_position(r) - self.centre
        return 0.5 * d @ self.K @ d

    def stiffness(self, r0=None, particle=None):
        """
        K (independent of r0)
        [N/m]
        """
        return self.K.copy()

    def __repr__(self):
        return f'HarmonicPotential(stiffness={np.diag(self.K).tolist()} N/m (diagonal), centre={self.centre.tolist()} m)'


class Gravity(ForceElement):
    """
    A uniform gravitational field: F = m g d̂ for a particle of mass m

    Keyword arguments:

    - g [levisim.core.constants.g]: gravitational acceleration [m/s²]
    - direction [(-1, 0, 0)]: the direction of the pull (normalised); levisim's default is -x, the
      direction of gravity in a horizontal-tweezer experiment

    It shifts a harmonic equilibrium by g/Ωj² along each axis (`Trap.find_equilibrium`).
    `FreeSpaceSimulation(gravity=True)` adds one of these to the simulated forces.
    """

    is_conservative = True
    simulation_group = 'gravity'

    def __init__(self, g=g_earth, direction=(-1, 0, 0)):
        self.g = float(g)
        direction = np.asarray(direction, dtype=float)
        self.direction = direction / np.linalg.norm(direction)

    def force(self, r, t, particle):
        """
        F = m g d̂ (the same at every r and t)
        [N]
        """
        return particle.m * self.g * self.direction

    def potential(self, r, particle):
        """
        U = -m g d̂·r
        [J]
        """
        return -particle.m * self.g * (self.direction @ as_position(r))

    def stiffness(self, r0=None, particle=None):
        """
        Zero: a uniform force has no gradient
        [N/m]
        """
        return np.zeros((3, 3))

    def __repr__(self):
        return f'Gravity(g={self.g} m/s², direction={self.direction.tolist()})'


class Feedback(ForceElement):
    """
    Feedback cold damping, a force -Γfb,j·pj on axis j

    Keyword arguments (all [2π·Hz], default 0):

    - Γ_fbx, Γ_fby, Γ_fbz: the feedback damping rates of the (x, y, z)-motion

    The damping adds to the gas damping, Γj = Γth + Γfb,j (`Trap.Γx`, ...). The noise injected by a
    real feedback loop (detection imprecision fed back as a force) is not modelled: `noise` is zero.
    """

    simulation_group = 'damping'

    def __init__(self, Γ_fbx=0.0, Γ_fby=0.0, Γ_fbz=0.0):
        self.Γ_fbx = float(Γ_fbx)
        self.Γ_fby = float(Γ_fby)
        self.Γ_fbz = float(Γ_fbz)

    def force(self, r, t, particle):
        """
        Zero: the feedback force depends on the momentum only, see `damping_rates`
        [N]
        """
        return np.zeros(3)

    def damping_rates(self, particle=None):
        """
        (Γfb,x, Γfb,y, Γfb,z)
        [2π·Hz]
        """
        return np.array([self.Γ_fbx, self.Γ_fby, self.Γ_fbz])

    def stiffness(self, r0=None, particle=None):
        """
        Zero: cold damping does not change the stiffness
        [N/m]
        """
        return np.zeros((3, 3))

    def __repr__(self):
        return (f'Feedback(Γ_fbx={self.Γ_fbx / kHz:.6g} kHz, Γ_fby={self.Γ_fby / kHz:.6g} kHz, '
                f'Γ_fbz={self.Γ_fbz / kHz:.6g} kHz)')


class PhotonRecoil(ForceElement):
    """
    The scattering (photon recoil, radiation pressure) force of a tweezer's light along its axis, +z

    ::

        F_scat,z = (Im α_eff / 2) |E(r)|² (k - zR/(zR² + z²))

    (`Trap.F_scattering`, with the Gouy-phase gradient). It pushes the particle downstream, to the
    equilibrium `Trap.z_eq_exact`. Transverse components and the recoil heating are not included in
    the force; the heating is the `noise` of the `Tweezer` element itself.

    `Trap(photon_recoil=True)` adds one of these for the trap's tweezer.

    - tweezer: the `Tweezer` whose light is scattered
    """

    simulation_group = 'recoil'

    def __init__(self, tweezer):
        self.tweezer = tweezer

    @property
    def finite_difference_step(self):
        """
        1e-4 of the tweezer waist w0 (as `Trap.Ωj_at_z_eq`)
        [m]
        """
        return 1e-4 * self.tweezer.w0

    def force(self, r, t, particle):
        """
        (0, 0, F_scat,z)
        [N]
        """
        x, y, z = r
        tweezer = self.tweezer
        α_eff = tweezer.effective_polarisability(particle)
        zR, k = tweezer.zR, tweezer.k
        phase_gradient = k - zR / (zR**2 + z**2)
        Fz = 0.5 * np.imag(α_eff) * tweezer.E_field_amplitude(x, y, z)**2 * phase_gradient
        return np.array([0.0, 0.0, Fz])

    def __repr__(self):
        return f'PhotonRecoil(tweezer={type(self.tweezer).__name__} at λ = {self.tweezer.λ * 1e9:.1f} nm)'

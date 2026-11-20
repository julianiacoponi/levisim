"""
Electric force elements on a charged particle: the quadrupole `PaulTrap` (radio-frequency trap) and a
`UniformElectricField` (a stray or compensation field)

The Paul trap is described at two levels, mirroring the rest of levisim:

- the full time-dependent force `PaulTrap.force(r, t, particle)`, for the time-domain simulations
  (micromotion, the Mathieu stability boundary, stray-field-induced excess micromotion), and
- the secular (pseudopotential) stiffness `PaulTrap.stiffness(r0, particle)`, the time-averaged
  restoring force that `Trap` and `MultiTrap` use, lowest order in the Mathieu parameters by default.

The stability analysis uses Mathieu's equation in the form of Leibfried et al., Rev. Mod. Phys. 75, 281
(2003), doi:10.1103/revmodphys.75.281, Eq. (3)-(6)::

    d²r_j/dτ² + (a_j + 2 q_j cos 2τ) r_j = 0,     τ = Ω_rf t / 2

with the characteristic exponent β_j (secular frequency Ω_j = β_j Ω_rf / 2), computed here exactly from
the Floquet monodromy matrix, and the first stability region a_0(q) < a < b_1(q) from Mathieu's
characteristic values (scipy.special.mathieu_a, mathieu_b).
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import mathieu_a, mathieu_b

from levisim.core.constants import e, ħ
from levisim.core.units import Hz, kHz, mm
from levisim.physics.fields.base import ForceElement, as_position

__all__ = [
    'PaulTrap',
    'UniformElectricField',
    'mathieu_characteristic_exponent',
    'mathieu_first_region_stable',
    'mathieu_floquet_multiplier',
    'mathieu_secular_beta',
]

AXES = 'xyz'
# the two radial axes of a trap whose symmetry axis is the key, in cyclic order
_RADIAL_AXES = {'x': (1, 2), 'y': (2, 0), 'z': (0, 1)}


def _particle_charge(particle, element='PaulTrap'):
    """
    The particle's net charge Q = charge·e, raising a clear error for a neutral (or missing) particle
    [C]
    """
    if particle is None:
        raise ValueError(f'{element} needs the particle (its charge and mass): pass particle=...')
    Q = particle.q
    if Q == 0:
        raise ValueError(
            f'{element} acts on the net charge of the particle, but this {type(particle).__name__} has '
            'charge = 0: give it a charge in elementary charges, e.g. Sphere(r=..., charge=300)'
        )
    return Q


def _axis_index(axis):
    if axis not in _RADIAL_AXES:
        raise ValueError(f"axis must be one of 'x', 'y', 'z', not {axis!r}")
    return AXES.index(axis)


# --------------------------------------------------------------------------------------------------
# Mathieu's equation
# --------------------------------------------------------------------------------------------------
def _monodromy_trace(a, q):
    """
    Trace Δ = y₁(π) + y₂'(π) of the monodromy matrix of y'' + (a + 2q cos 2τ) y = 0 over one period π
    of the drive, from the fundamental solutions y₁(0) = 1, y₁'(0) = 0 and y₂(0) = 0, y₂'(0) = 1
    [·]
    """
    def rhs(τ, Y):
        k = a + 2*q*np.cos(2*τ)
        return [Y[1], -k*Y[0], Y[3], -k*Y[2]]

    solution = solve_ivp(rhs, (0.0, np.pi), [1.0, 0.0, 0.0, 1.0], method='DOP853', rtol=1e-12, atol=1e-14)
    Y = solution.y[:, -1]
    return Y[0] + Y[3]


def mathieu_characteristic_exponent(a, q):
    """
    Characteristic exponent β of y'' + (a + 2q cos 2τ) y = 0 in the first stability region, from the
    Floquet monodromy matrix: cos(πβ) = Δ/2 with Δ its trace (numerically exact, to ~1e-10)

    Returns β in [0, 1] when |Δ| ≤ 2 and the point lies in the first stability region
    (`mathieu_first_region_stable`), and nan otherwise (unstable, or a higher stability region).
    The secular angular frequency is β Ω_rf/2.
    [·]
    """
    a, q = float(a), float(q)
    if not mathieu_first_region_stable(a, q):
        return np.nan
    Δ = _monodromy_trace(a, q)
    return float(np.arccos(np.clip(Δ / 2, -1.0, 1.0)) / np.pi)


def mathieu_floquet_multiplier(a, q):
    """
    Largest modulus |λ| of the Floquet multipliers of y'' + (a + 2q cos 2τ) y = 0 over one drive period
    (λ² - Δλ + 1 = 0, Δ the monodromy trace): 1 where the motion is stable, > 1 where it grows by |λ|
    per rf period, i.e. at the rate ln|λ| · Ω_rf/2π [1/s]
    [·]
    """
    Δ = _monodromy_trace(float(a), float(q))
    if abs(Δ) <= 2:
        return 1.0
    return float(abs(Δ)/2 + np.sqrt(Δ**2/4 - 1))


def mathieu_first_region_stable(a, q):
    """
    Whether (a, q) lies in the first stability region of Mathieu's equation, a_0(q) < a < b_1(q), with
    Mathieu's characteristic values a_0 and b_1 (scipy.special.mathieu_a, mathieu_b; even in q)

    At a = 0 this is |q| < 0.908046 (Leibfried et al. 2003, Fig. 1).
    [bool]
    """
    q = abs(float(q))
    return bool(mathieu_a(0, q) < float(a) < mathieu_b(1, q))


def mathieu_secular_beta(a, q, order='lowest'):
    """
    Characteristic exponent β (secular frequency β Ω_rf/2) at a given order in the Mathieu parameters

    - 'lowest': β² = a + q²/2 (the pseudopotential; Leibfried et al. 2003, Eq. 6), nan if a + q²/2 ≤ 0
    - 'next': β² = a - (a - 1)q²/(2(a - 1)² - q²) - (5a + 7)q⁴/(32(a - 1)³(a - 4)), the first two terms of
      the continued-fraction expansion of β (March, J. Mass Spectrom. 32, 351 (1997), Eq. 30); relative
      error ~5e-6 at q = 0.2, ~2e-4 at q = 0.4 (a = 0), against ~8e-3 and ~3e-2 at lowest order
    - 'exact': the Floquet value, `mathieu_characteristic_exponent` (nan outside the first region)

    [·]
    """
    a, q = float(a), float(q)
    if order == 'lowest':
        β2 = a + q**2/2
    elif order == 'next':
        β2 = a - (a - 1)*q**2/(2*(a - 1)**2 - q**2) - (5*a + 7)*q**4/(32*(a - 1)**3*(a - 4))
    elif order == 'exact':
        return mathieu_characteristic_exponent(a, q)
    else:
        raise ValueError(f"order must be 'lowest', 'next' or 'exact', not {order!r}")
    return float(np.sqrt(β2)) if β2 > 0 else np.nan


def _geometry(values, name):
    values = np.asarray(values, dtype=float)
    if values.shape != (3,):
        raise ValueError(f'{name} must be three numbers (a_x, a_y, a_z), not shape {values.shape}')
    scale = max(float(np.max(np.abs(values))), 1e-300)
    if abs(float(np.sum(values))) > 1e-9 * scale:
        raise ValueError(
            f'{name} = {values.tolist()} does not satisfy Laplace\'s equation: a_x + a_y + a_z must be 0 '
            'for a potential in free space'
        )
    return values


# --------------------------------------------------------------------------------------------------
# the quadrupole Paul trap
# --------------------------------------------------------------------------------------------------
class PaulTrap(ForceElement):
    """
    An ideal quadrupole Paul (radio-frequency) trap acting on a charged particle

    The electric potential about the trap centre c is ::

        φ(r, t) = κ_dc U_dc Σ_j a^dc_j (r_j - c_j)²/(2 r_dc²) + κ_geom V_rf cos(Ω_rf t + φ_rf) Σ_j a_j (r_j - c_j)²/(2 r0²)

    with Σ a_j = Σ a^dc_j = 0 (Laplace). With ``dc_geometry=None`` the dc voltage acts on the rf
    geometry, φ = (U_dc + V_rf cos Ω_rf t)(a_x x² + a_y y² + a_z z²)/(2 r0²) (times κ_geom). The force on
    a particle of charge Q = charge·e is F = -Q∇φ, i.e. along each axis ::

        F_j = -Q [κ_dc U_dc a^dc_j/r_dc² + κ_geom V_rf a_j cos(Ω_rf t + φ_rf)/r0²] (r_j - c_j)

    so each axis obeys Mathieu's equation (Leibfried et al. 2003, Eq. 3-5) with ::

        a_j = 4 Q κ_dc U_dc a^dc_j / (m r_dc² Ω_rf²),     q_j = 2 Q κ_geom V_rf a_j / (m r0² Ω_rf²)

    Keyword arguments (defaults in brackets):

    - V_rf: amplitude of the rf voltage [V]
    - Ω_rf: angular frequency of the rf drive [2π·Hz]
    - r0 [1 mm]: length scale of the rf quadrupole (for hyperbolic electrodes, the distance from the
      centre to the electrodes along an axis with a_j = 1) [m]
    - U_dc [0]: dc voltage [V]
    - geometry [(1, 1, -2)]: the rf curvatures (a_x, a_y, a_z), Σa = 0; (1, 1, -2) is the ring
      (endcap, "3D") trap with symmetry axis z, (1, -1, 0) the rf of a linear trap with axis z
    - dc_geometry [None]: the dc curvatures, if they differ from the rf ones (a linear trap's endcaps
      give (-½, -½, 1)); None uses `geometry`
    - dc_r0 [None]: length scale of the dc potential (r0 if None) [m]
    - κ_geom [1]: geometric efficiency of real electrodes, the ratio of the curvature of the real rf
      potential to that of ideal hyperbolic electrodes at the same r0 (typically 0.3-0.9) [·]
    - κ_dc [None]: the same for the dc potential (κ_geom if None) [·]
    - φ_rf [0]: phase of the rf drive at t = 0 [rad]
    - centre [(0, 0, 0)]: the trap centre c, e.g. to offset the Paul trap from a tweezer focus [m]
    - electrode_distances [None]: the distance d_j from the centre to the electrodes along each axis,
      for the trap depth; None gives d_j = r0/√|a_j| (or r_dc/√|a^dc_j| on an axis without rf
      curvature), the ideal hyperbolic electrodes [m]
    - secular_order ['lowest']: the order of the secular stiffness `stiffness` returns, 'lowest'
      (pseudopotential plus dc, k_j = m Ω_rf²(a_j + q_j²/2)/4), 'next' or 'exact' (from the Mathieu
      exponent, `mathieu_secular_beta`); see `stiffness` for why 'lowest' is the right choice in a
      hybrid trap
    - voltage_noise_psd [0]: one-sided spectral density S_V of the electrode voltage noise, white over
      the secular band [V²/Hz] (e.g. (38 nV/√Hz)²)
    - noise_distances [None]: effective distance d_j that converts the voltage noise to an electric
      field noise at the particle along each axis, E_j = V_noise/d_j: a number (all axes) or three
      numbers, np.inf for an axis without field noise [m]
    - noise_in_simulation [True]: whether `FreeSpaceSimulation` adds the electrode force noise as an
      extra white noise (it always adds the gas noise)

    Presets: `PaulTrap.ring` (= `PaulTrap.endcap`, the 3D quadrupole) and `PaulTrap.linear`.

    Levels of description:

    - `force(r, t, particle)`: the full rf + dc force (time domain)
    - `secular_force(r, particle)`, `potential(r, particle)` and `stiffness(r0, particle)`: the
      time-averaged (pseudopotential) force, energy and stiffness, ::

          U_sec = Q²|∇φ_rf|²/(4 m Ω_rf²) + Q φ_dc,     φ_rf = κ_geom V_rf Σ a_j r_j²/(2r0²)

    - `mathieu_parameters`, `secular_frequencies`, `is_stable`, `trap_depth`, `report`

    The trap's axes are the lab axes (no rotation). Gravity, a stray `UniformElectricField` or an offset
    tweezer displace the particle from the rf null, which drives excess micromotion of amplitude
    |q_j| u_j / 2 at Ω_rf (`excess_micromotion_amplitude`).
    """

    is_conservative = False
    is_time_dependent = True

    def __init__(
        self,
        V_rf,
        Ω_rf,
        r0=1*mm,
        U_dc=0.0,
        geometry=(1.0, 1.0, -2.0),
        dc_geometry=None,
        dc_r0=None,
        κ_geom=1.0,
        κ_dc=None,
        φ_rf=0.0,
        centre=(0.0, 0.0, 0.0),
        electrode_distances=None,
        secular_order='lowest',
        voltage_noise_psd=0.0,
        noise_distances=None,
        noise_in_simulation=True,
    ):
        self.V_rf = float(V_rf)
        self.Ω_rf = float(Ω_rf)
        if not np.isfinite(self.Ω_rf) or self.Ω_rf <= 0:
            raise ValueError(f'Ω_rf must be a positive angular frequency, not {Ω_rf}')
        self.r0 = float(r0)
        if not np.isfinite(self.r0) or self.r0 <= 0:
            raise ValueError(f'r0 must be a positive length, not {r0}')
        self.U_dc = float(U_dc)
        self.geometry = _geometry(geometry, 'geometry')
        self.dc_geometry = self.geometry.copy() if dc_geometry is None else _geometry(dc_geometry, 'dc_geometry')
        self.dc_r0 = self.r0 if dc_r0 is None else float(dc_r0)
        if not np.isfinite(self.dc_r0) or self.dc_r0 <= 0:
            raise ValueError(f'dc_r0 must be a positive length, not {dc_r0}')
        self.κ_geom = float(κ_geom)
        self.κ_dc = self.κ_geom if κ_dc is None else float(κ_dc)
        self.φ_rf = float(φ_rf)
        self.centre = as_position(centre)
        if secular_order not in ('lowest', 'next', 'exact'):
            raise ValueError(f"secular_order must be 'lowest', 'next' or 'exact', not {secular_order!r}")
        self.secular_order = secular_order
        self._electrode_distances = None if electrode_distances is None else self._three(electrode_distances, 'electrode_distances')
        self.voltage_noise_psd = float(voltage_noise_psd)
        if self.voltage_noise_psd < 0:
            raise ValueError('voltage_noise_psd is a spectral density and cannot be negative')
        self.noise_distances = None if noise_distances is None else self._three(noise_distances, 'noise_distances')
        if self.voltage_noise_psd > 0 and self.noise_distances is None:
            raise ValueError('voltage_noise_psd needs noise_distances (the distances d_j with E_j = V_noise/d_j)')
        self.noise_in_simulation = bool(noise_in_simulation)

    @staticmethod
    def _three(values, name):
        values = np.asarray(values, dtype=float)
        if values.ndim == 0:
            values = np.full(3, float(values))
        if values.shape != (3,) or np.any(values <= 0):
            raise ValueError(f'{name} must be a positive number or three positive numbers [m], not {values}')
        return values

    # ----------------------------------------------------------------------------------------------
    # presets
    # ----------------------------------------------------------------------------------------------
    @classmethod
    def ring(cls, V_rf, Ω_rf, r0=1*mm, U_dc=0.0, axis='z', **kwargs):
        """
        The ring (endcap, "3D") quadrupole trap with symmetry axis `axis`: curvatures 1 on the two
        radial axes and -2 on the axis, φ = (U_dc + V_rf cos Ω_rf t)(x² + y² - 2z²)/(2r0²) for axis z,
        i.e. the ideal ring trap with r0² = 2z0² (Leibfried et al. 2003, Eq. 2 with α = α' = (1, 1, -2)/r0²)

        So q_axis = -2 q_radial and a_axis = -2 a_radial. Other keyword arguments as `PaulTrap`
        (κ_geom for real, non-hyperbolic electrodes).
        """
        geometry = np.ones(3)
        geometry[_axis_index(axis)] = -2.0
        return cls(V_rf, Ω_rf, r0=r0, U_dc=U_dc, geometry=geometry, **kwargs)

    endcap = ring

    @classmethod
    def linear(cls, V_rf, Ω_rf, r0=1*mm, U_endcap=0.0, z0=None, axis='z', κ_endcap=None, **kwargs):
        """
        The linear Paul trap with trap axis `axis`: rf on the two radial axes with curvatures (1, -1)
        (two rf rods and two grounded rods at distance r0), and the axial confinement by the dc endcaps
        at distance z0, curvatures (-½, -½, 1) [radial, radial, axial] ::

            φ = κ_geom V_rf cos(Ω_rf t)(x² - y²)/(2r0²) + κ_endcap U_endcap (z² - (x² + y²)/2)/(2z0²)

        so the axial frequency is Ω_z = √(Q κ_endcap U_endcap / (m z0²)) and the radial secular frequencies
        are (Ω_rf/2)√(a_j + q_j²/2) with a_x = a_y = -2Q κ_endcap U_endcap/(m z0² Ω_rf²) (the endcaps
        defocus radially). Keyword arguments:

        - U_endcap [0]: the endcap (dc) voltage [V]
        - z0 [r0]: the endcap length scale (the distance to the endcaps for ideal electrodes) [m]
        - κ_endcap [None]: geometric efficiency of the endcaps (κ_geom if None) [·]
        - other keyword arguments as `PaulTrap`
        """
        index = _axis_index(axis)
        radial = _RADIAL_AXES[axis]
        geometry = np.zeros(3)
        geometry[radial[0]], geometry[radial[1]] = 1.0, -1.0
        dc_geometry = np.full(3, -0.5)
        dc_geometry[index] = 1.0
        return cls(
            V_rf, Ω_rf, r0=r0, U_dc=U_endcap, geometry=geometry, dc_geometry=dc_geometry,
            dc_r0=r0 if z0 is None else z0, κ_dc=κ_endcap, **kwargs,
        )

    # ----------------------------------------------------------------------------------------------
    # curvatures and Mathieu parameters
    # ----------------------------------------------------------------------------------------------
    def rf_curvatures(self):
        """
        Curvatures c_j = κ_geom V_rf a_j/r0² of the rf potential amplitude, φ_rf = Σ c_j r_j²/2
        [V/m²]
        """
        return self.κ_geom * self.V_rf * self.geometry / self.r0**2

    def dc_curvatures(self):
        """
        Curvatures κ_dc U_dc a^dc_j/r_dc² of the dc potential, φ_dc = Σ c_j r_j²/2
        [V/m²]
        """
        return self.κ_dc * self.U_dc * self.dc_geometry / self.dc_r0**2

    def mathieu_parameters(self, particle, extra_stiffness=None):
        """
        Mathieu parameters (a, q), each an array over the axes (x, y, z) ::

            a_j = 4 Q κ_dc U_dc a^dc_j/(m r_dc² Ω_rf²) + 4 K_jj/(m Ω_rf²),     q_j = 2 Q κ_geom V_rf a_j/(m r0² Ω_rf²)

        with the optional extra static stiffness K (a 3×3 tensor, or three numbers, from other elements
        such as a tweezer; only its diagonal enters) [N/m]. Leibfried et al. 2003, Eq. 5.
        [·]
        """
        Q = _particle_charge(particle)
        m = particle.m
        a = 4 * Q * self.dc_curvatures() / (m * self.Ω_rf**2)
        q = 2 * Q * self.rf_curvatures() / (m * self.Ω_rf**2)
        if extra_stiffness is not None:
            K = np.asarray(extra_stiffness, dtype=float)
            K = np.diag(K) if K.ndim == 2 else K
            a = a + 4 * K / (m * self.Ω_rf**2)
        return a, q

    def secular_β(self, particle, order=None, extra_stiffness=None):
        """
        Mathieu characteristic exponents (β_x, β_y, β_z) at `order` (default `secular_order`), see
        `mathieu_secular_beta`; nan on an unconfined or unstable axis
        [·]
        """
        order = self.secular_order if order is None else order
        a, q = self.mathieu_parameters(particle, extra_stiffness)
        return np.array([mathieu_secular_beta(a_j, q_j, order) for a_j, q_j in zip(a, q)])

    def secular_frequencies(self, particle, order=None, extra_stiffness=None):
        """
        Secular angular frequencies Ω_j = β_j Ω_rf/2 over (x, y, z), at `order` ('lowest', 'next',
        'exact'; default `secular_order`), with the optional extra stiffness of other elements folded
        into a_j (see `mathieu_parameters`); nan on an unconfined or unstable axis
        [2π·Hz]
        """
        return self.secular_β(particle, order, extra_stiffness) * self.Ω_rf / 2

    def is_stable(self, particle, extra_stiffness=None):
        """
        Whether the motion is stable on every axis, i.e. each (a_j, q_j) lies in the first stability
        region of Mathieu's equation, a_0(q_j) < a_j < b_1(q_j) (see `mathieu_first_region_stable`)
        [bool]
        """
        return all(self.stable_axes(particle, extra_stiffness))

    def stable_axes(self, particle, extra_stiffness=None):
        """
        Per axis (x, y, z): whether (a_j, q_j) lies in the first stability region
        [bool, bool, bool]
        """
        a, q = self.mathieu_parameters(particle, extra_stiffness)
        return tuple(mathieu_first_region_stable(a_j, q_j) for a_j, q_j in zip(a, q))

    def adiabaticity_warning(self, particle):
        """
        A message if the pseudopotential is a poor approximation (|q_j| > 0.4 or |a_j| > 0.1 on an axis),
        else ''
        """
        a, q = self.mathieu_parameters(particle)
        bad = [AXES[j] for j in range(3) if abs(q[j]) > 0.4 or abs(a[j]) > 0.1]
        if not bad:
            return ''
        return (f'PaulTrap: |q| > 0.4 or |a| > 0.1 on axes {", ".join(bad)}; the lowest-order secular '
                "stiffness is off by more than ~3% there (use secular_order='exact')")

    # ----------------------------------------------------------------------------------------------
    # forces
    # ----------------------------------------------------------------------------------------------
    def force(self, r, t, particle):
        """
        Full rf + dc force F_j = -Q [c^dc_j + c^rf_j cos(Ω_rf t + φ_rf)] (r_j - c_j)
        [N]
        """
        Q = _particle_charge(particle)
        d = as_position(r) - self.centre
        curvature = self.dc_curvatures() + self.rf_curvatures() * np.cos(self.Ω_rf * t + self.φ_rf)
        return -Q * curvature * d

    def secular_stiffness_diagonal(self, particle, order=None):
        """
        Secular stiffnesses (k_x, k_y, k_z) on their own (no other elements): at lowest order
        k_j = Q c^dc_j + Q²(c^rf_j)²/(2 m Ω_rf²) = m Ω_rf² (a_j + q_j²/2)/4, otherwise m (β_j Ω_rf/2)² from
        `secular_β` (nan where unstable)
        [N/m]
        """
        order = self.secular_order if order is None else order
        Q = _particle_charge(particle)
        if order == 'lowest':
            return Q * self.dc_curvatures() + Q**2 * self.rf_curvatures()**2 / (2 * particle.m * self.Ω_rf**2)
        β = self.secular_β(particle, order)
        return particle.m * (β * self.Ω_rf / 2)**2

    def stiffness(self, r0=None, particle=None):
        """
        Secular (time-averaged) stiffness tensor diag(k_x, k_y, k_z), independent of r0 for the ideal
        quadrupole
        [N/m]

        At `secular_order` = 'lowest' (default) this is the pseudopotential plus the dc curvature,
        k_j = m Ω_rf²(a_j + q_j²/2)/4, which is additive: in a hybrid trap the total secular frequency
        is √((k_j + K_other,jj)/m), exactly the lowest-order Mathieu result with a_j increased by
        4K_other,jj/(mΩ_rf²). At 'next' or 'exact' the stiffness is m(β_j Ω_rf/2)² of the Paul trap on
        its own; adding another element's stiffness to it is then not exact (β is not additive in a),
        so for a hybrid trap at higher order use `secular_frequencies(particle, 'exact',
        extra_stiffness=K_other)` or `Trap.paul_trap_report`.
        """
        return np.diag(self.secular_stiffness_diagonal(particle))

    def secular_force(self, r, particle):
        """
        Time-averaged (secular) force -K·(r - c), with K the `stiffness`
        [N]
        """
        d = as_position(r) - self.centre
        return -self.secular_stiffness_diagonal(particle) * d

    def potential(self, r, particle):
        """
        Secular (pseudo)potential energy ½ Σ k_j (r_j - c_j)², with k_j the `stiffness`
        [J]

        The full time-dependent potential energy is Q φ(r, t); this is its time average plus the
        kinetic energy of the micromotion (Dehmelt's pseudopotential).
        """
        d = as_position(r) - self.centre
        return 0.5 * float(np.sum(self.secular_stiffness_diagonal(particle) * d**2))

    # ----------------------------------------------------------------------------------------------
    # depth, micromotion, noise
    # ----------------------------------------------------------------------------------------------
    @property
    def electrode_distances(self):
        """
        Distances d_j from the trap centre to the electrodes along (x, y, z): as given, or for ideal
        electrodes r0/√|a_j|, or r_dc/√|a^dc_j| on an axis without rf curvature (inf if neither)
        [m]
        """
        if self._electrode_distances is not None:
            return self._electrode_distances.copy()
        d = np.full(3, np.inf)
        for j in range(3):
            if self.geometry[j] != 0:
                d[j] = self.r0 / np.sqrt(abs(self.geometry[j]))
            elif self.dc_geometry[j] != 0:
                d[j] = self.dc_r0 / np.sqrt(abs(self.dc_geometry[j]))
        return d

    def trap_depth(self, particle, per_axis=False):
        """
        Secular (pseudopotential) trap depth, the energy ½ k_j d_j² at the electrodes along each axis,
        with the lowest-order stiffness and `electrode_distances`; the depth is the smallest over the
        axes (0 if an axis is not confining)
        [J]

        Valid for |q| ≲ 0.4 (the pseudopotential) and ideal quadrupole fields out to the electrodes;
        real traps are shallower (anharmonic fields near the electrodes).
        """
        k = self.secular_stiffness_diagonal(particle, 'lowest')
        depth = 0.5 * k * self.electrode_distances**2
        depth = np.where(k > 0, depth, 0.0)
        return depth if per_axis else float(np.min(depth))

    def excess_micromotion_amplitude(self, displacement, particle):
        """
        Amplitude of the micromotion at Ω_rf of a particle displaced by u from the rf null (by gravity, a
        stray field, an offset tweezer), |q_j| u_j/2 per axis (Leibfried et al. 2003, Eq. 10 and 13)
        [m]
        """
        _, q = self.mathieu_parameters(particle)
        return 0.5 * np.abs(q) * np.abs(np.asarray(displacement, dtype=float) - self.centre)

    def electric_field_noise_psd(self):
        """
        One-sided spectral densities S_E,j = S_V/d_j² of the electric field noise at the particle along
        (x, y, z), from `voltage_noise_psd` and `noise_distances`
        [(V/m)²/Hz]
        """
        if self.noise_distances is None or self.voltage_noise_psd == 0:
            return np.zeros(3)
        return self.voltage_noise_psd / self.noise_distances**2

    def noise(self, particle):
        """
        Two-sided force spectral density matrix of the electrode noise, diag(Q² S_E,j / 2), so that
        〈F_j(t) F_j(t')〉 = Q² S_E,j/2 · δ(t - t') (S_E one-sided, `electric_field_noise_psd`)
        [N²/Hz]
        """
        S_E = self.electric_field_noise_psd()
        if not np.any(S_E):
            return np.zeros((3, 3))
        Q = _particle_charge(particle)
        return np.diag(Q**2 * S_E / 2)

    def heating_rate(self, particle, Ω=None):
        """
        Heating rates (Γ_x, Γ_y, Γ_z) of the secular motion by the electrode noise, in quanta per second ::

            Γ_j = Q² S_E,j / (4 m ħ Ω_j)

        with S_E one-sided (Turchette et al., Phys. Rev. A 61, 063418 (2000); the S_ff = 4mħΩΓ of Dania et
        al., PRL 132, 133602 (2024)). The energy heating rate is ħΩ_jΓ_j = Q²S_E,j/(4m). Ω defaults
        to the secular frequencies at `secular_order`.
        [1/s]
        """
        Q = _particle_charge(particle)
        Ω = self.secular_frequencies(particle) if Ω is None else np.broadcast_to(np.asarray(Ω, dtype=float), (3,))
        return Q**2 * self.electric_field_noise_psd() / (4 * particle.m * ħ * Ω)

    def max_time_step(self, particle=None):
        """
        Largest time step that resolves the rf drive in a time-domain simulation, 1/(20 f_rf)
        [s]

        With the Kasdin RK4 integrator ('vector_4') the secular frequency of the simulated (noise-free)
        motion is then within ~2e-4 of the Floquet value (q = 0.3, a = 0; ~2e-5 at 40 steps per period);
        Heun ('vector_2') is second order and is off by ~1.6% at 20 steps (~0.1% at 80).
        """
        return 2*np.pi / (20 * self.Ω_rf)

    # ----------------------------------------------------------------------------------------------
    # summaries
    # ----------------------------------------------------------------------------------------------
    def report(self, particle, extra_stiffness=None):
        """
        Dict of the trap's characteristics for a particle: Mathieu 'a', 'q' (arrays over x, y, z),
        'stable_axes', 'stable', secular frequencies 'Ω_lowest', 'Ω_next', 'Ω_exact' [2π·Hz] (with the
        optional extra stiffness of other elements folded into a), 'stiffness' (lowest order, this
        element only) [N/m], 'trap_depth' [J] and 'heating_rate' [1/s]
        """
        a, q = self.mathieu_parameters(particle, extra_stiffness)
        out = {
            'a': a,
            'q': q,
            'stable_axes': self.stable_axes(particle, extra_stiffness),
            'stable': self.is_stable(particle, extra_stiffness),
        }
        for order in ('lowest', 'next', 'exact'):
            out[f'Ω_{order}'] = self.secular_frequencies(particle, order, extra_stiffness)
        out['stiffness'] = self.secular_stiffness_diagonal(particle, 'lowest')
        out['trap_depth'] = self.trap_depth(particle)
        out['heating_rate'] = self.heating_rate(particle, out['Ω_exact'])
        return out

    def report_str(self, particle, extra_stiffness=None):
        """The `report` as a printable summary"""
        rep = self.report(particle, extra_stiffness)
        lines = [repr(self), f'  particle charge = {particle.charge} e, mass = {particle.m:.4e} kg']
        for j, axis in enumerate(AXES):
            lines.append(
                f'  {axis}: a = {rep["a"][j]:+.5g}, q = {rep["q"][j]:+.5g}, stable = {rep["stable_axes"][j]}, '
                f'Ω/2π = {rep["Ω_lowest"][j] / kHz:.6g} (lowest), {rep["Ω_next"][j] / kHz:.6g} (next), '
                f'{rep["Ω_exact"][j] / kHz:.6g} (exact) kHz'
            )
        lines.append(f'  trap depth = {rep["trap_depth"] / e:.4g} eV')
        return '\n'.join(lines)

    def __repr__(self):
        return (
            f'PaulTrap(V_rf={self.V_rf:.6g} V, Ω_rf/2π={self.Ω_rf / Hz:.6g} Hz, r0={self.r0:.4g} m, '
            f'U_dc={self.U_dc:.6g} V, geometry={self.geometry.tolist()}, dc_geometry={self.dc_geometry.tolist()}, '
            f'κ_geom={self.κ_geom:.4g}, centre={self.centre.tolist()} m)'
        )


class UniformElectricField(ForceElement):
    """
    A uniform static electric field E acting on the particle's net charge, F = Q E (a stray field, or
    the field of compensation electrodes)

    With a `PaulTrap` it displaces the particle from the rf null by u_j = Q E_j/k_j and so drives excess
    micromotion of amplitude |q_j| u_j/2 at Ω_rf.

    - E: the field (E_x, E_y, E_z) [V/m]
    """

    is_conservative = True

    def __init__(self, E):
        self.E = as_position(E)

    def force(self, r, t, particle):
        """
        F = Q E
        [N]
        """
        return _particle_charge(particle, 'UniformElectricField') * self.E

    def potential(self, r, particle):
        """
        U = -Q E·r
        [J]
        """
        return -_particle_charge(particle, 'UniformElectricField') * float(self.E @ as_position(r))

    def stiffness(self, r0=None, particle=None):
        """
        Zero: a uniform force has no gradient
        [N/m]
        """
        return np.zeros((3, 3))

    def __repr__(self):
        return f'UniformElectricField(E={self.E.tolist()} V/m)'


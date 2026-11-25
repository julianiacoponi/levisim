"""
Central pair interactions between spheres: Coulomb, Newtonian gravity and Casimir

Each is a power law in a (possibly shifted) distance, ::

    U_jk(r) = C_jk · (r - s_jk)^(-n)

with r the centre-to-centre distance. `PowerLawPairInteraction` evaluates U, U' and U'' from the
triple (C, s, n), which is also what the compiled multi-particle kernel reads, so the NumPy and
numba time loops share one definition of each force.
"""

import numpy as np
from scipy.constants import G as G_newton
from scipy.integrate import quad

from levisim.core.constants import c, ħ, ε0, π
from levisim.core.logger import logger
from levisim.physics.interactions.base import PairInteraction, particle_radius

__all__ = [
    'Casimir',
    'Coulomb',
    'G_newton',
    'NewtonianGravity',
    'PowerLawPairInteraction',
    'lifshitz_plate_factor',
]

# Coulomb constant 1/(4πε0) [N·m²/C²]
k_Coulomb = 1 / (4 * π * ε0)


class PowerLawPairInteraction(PairInteraction):
    """
    A central pair potential U = C·(r - s)^(-n), see `power_law(particle_j, particle_k)`
    """

    def power_law(self, particle_j, particle_k):
        """(C, s, n) of the pair: U = C·(r - s)^(-n) [J·mⁿ, m, ·]"""
        raise NotImplementedError

    def pair_derivatives(self, r, particle_j, particle_k):
        C, s, n = self.power_law(particle_j, particle_k)
        d = r - s
        U = C * d**(-n)
        return U, -n * U / d, n * (n + 1) * U / d**2


class Coulomb(PowerLawPairInteraction):
    """
    Coulomb interaction of charged particles, U = q_j·q_k / (4πε0·r)

    The charges are `Particle.q` = charge·e [C] (`Particle(charge=...)` in units of e). A sphere
    with a uniform (or spherically symmetric) charge acts as a point charge outside contact;
    induced (image) dipoles are not included.
    """

    def power_law(self, particle_j, particle_k):
        return k_Coulomb * particle_j.q * particle_k.q, 0.0, 1

    def check_separation(self, r, particle_j, particle_k):
        super().check_separation(r, particle_j, particle_k)
        _check_outside_contact(self.label, r, particle_j, particle_k)


class NewtonianGravity(PowerLawPairInteraction):
    """
    Newtonian gravity between particles, U = -G·m_j·m_k / r

    Exact for spheres of spherically symmetric density outside contact (shell theorem); for other
    shapes it is the leading (monopole) term. G is CODATA's (scipy.constants.G).
    """

    def power_law(self, particle_j, particle_k):
        return -G_newton * particle_j.m * particle_k.m, 0.0, 1

    def check_separation(self, r, particle_j, particle_k):
        super().check_separation(r, particle_j, particle_k)
        _check_outside_contact(self.label, r, particle_j, particle_k)


def _check_outside_contact(label, r, particle_j, particle_k):
    R_j, R_k = particle_radius(particle_j), particle_radius(particle_k)
    if R_j is not None and R_k is not None and r <= R_j + R_k:
        raise ValueError(
            f'{label}: the particles overlap (centre-to-centre distance {r:.4g} m ≤ R_j + R_k = {R_j + R_k:.4g} m)'
        )


def _li4(x):
    """Polylogarithm Li₄(x) = Σ xⁿ/n⁴ for 0 ≤ x ≤ 1 [·]"""
    if x >= 1:
        return π**4 / 90
    if x <= 0:
        return 0.0
    n = np.arange(1, 2001)
    return float(np.sum(x**n / n**4))


def lifshitz_plate_factor(ε_j, ε_k=None):
    """
    Ratio η of the zero-temperature Casimir energy between two dielectric half-spaces of
    frequency-independent permittivities ε_j, ε_k to that between ideal mirrors, -π²ħc/(720a³)

    From the Lifshitz formula at T = 0 (Lifshitz, Sov. Phys. JETP 2, 73 (1956); in the form of
    Klimchitskaya, Mohideen and Mostepanenko, Rev. Mod. Phys. 81, 1827 (2009), Eqs. (3)-(7) with
    kB·T/2π·Σ' → ħ/4π²·∫dξ), ::

        E(a) = ħ/(4π²) ∫dξ ∫k dk Σ_TM,TE ln(1 - r_j·r_k·e^(-2aq)),   q = √(k² + ξ²/c²)

    With ε independent of frequency the reflection coefficients depend only on p = ξ/(cq), so the
    q-integral is done in closed form, ∫q² ln(1 - R e^(-2aq)) dq = -Li₄(R)/(4a³), and ::

        η = (45/π⁴) ∫₀¹ dp [Li₄(r_TM,j·r_TM,k) + Li₄(r_TE,j·r_TE,k)]
        r_TM = (ε - s)/(ε + s),  r_TE = (1 - s)/(1 + s),  s = √(1 - p² + ε·p²)

    η → 1 for ε → ∞ (ideal mirrors). A frequency-independent ε is the nondispersive
    approximation: it describes the retarded regime only, where the gap is much larger than the
    wavelengths of the material's absorption lines, with ε their static value.
    [·]
    """
    ε_k = ε_j if ε_k is None else ε_k
    if not (ε_j > 1 and ε_k > 1):
        raise ValueError(f'lifshitz_plate_factor needs permittivities > 1, not {ε_j}, {ε_k}')

    def integrand(p):
        total = 0.0
        r_TM, r_TE = [], []
        for ε in (ε_j, ε_k):
            s = np.sqrt(1 - p**2 + ε * p**2)
            r_TM.append((ε - s) / (ε + s))
            r_TE.append((1 - s) / (1 + s))
        total += _li4(r_TM[0] * r_TM[1])
        total += _li4(r_TE[0] * r_TE[1])
        return total

    value, _ = quad(integrand, 0.0, 1.0, epsabs=1e-13, epsrel=1e-11, limit=200)
    return 45 / π**4 * value


class Casimir(PowerLawPairInteraction):
    """
    Casimir (retarded van der Waals) attraction between two dielectric spheres at zero temperature

    Two limits, chosen with `model` (there is no closed form in between; use the one whose
    validity range contains the separations of interest):

    - ``model='casimir_polder'`` (default): two polarisable dipoles, separation r ≫ R_j + R_k, ::

          U = -(23ħc / 4π) · α_j·α_k / r⁷,   α = R³·(ε - 1)/(ε + 2)

      (Casimir and Polder, Phys. Rev. 73, 360 (1948), with the static polarisability volume of a
      dielectric sphere; the same form as Eq. (4) of van de Kamp et al., Phys. Rev. A 102, 062807
      (2020)). Valid for r ≫ R_j, R_k (dipole limit) and r much larger than the wavelengths of the
      material's absorption lines (retarded limit, static ε).
    - ``model='proximity_force'``: the proximity-force (Derjaguin) approximation for a gap
      d = r - R_j - R_k ≪ R_j, R_k: the force is 2π·R_eff times the energy per area between
      half-spaces (Klimchitskaya et al., Rev. Mod. Phys. 81, 1827 (2009), Eq. (16) and Eq. (2)), ::

          U = -η · π³ħc·R_eff / (720·d²),   R_eff = R_j·R_k/(R_j + R_k)

      with η = `lifshitz_plate_factor`(ε_j, ε_k) (η = 1 with ``ideal_mirrors=True``). For two equal
      spheres this is the expression -f₀·π³ħcR/(1440·d²) quoted by Krisnanda et al., npj Quantum
      Inf. 6, 12 (2020). Valid for d ≪ R (PFA) and d much larger than the absorption wavelengths
      (static ε).

    Both are zero-temperature results: they hold for separations below the thermal wavelength
    λ_T = ħc/(kB·T) (7.6 μm at 300 K, 0.57 mm at 4 K); at larger separations the thermal
    (zero-Matsubara-frequency) term takes over, which is not modelled.

    Keyword arguments:

    - model ['casimir_polder']: 'casimir_polder' or 'proximity_force'
    - permittivity [None]: the static relative permittivity ε of every particle (a number), or None
      to use each particle's `εr`. NOTE: levisim's `εr` is the OPTICAL permittivity (1.98 for the
      default silica); the static value is larger (≈ 3.8 for fused silica), and makes the
      Casimir-Polder energy ≈ 3× larger, so pass it explicitly for a quantitative estimate.
    - ideal_mirrors [False]: proximity_force only, use η = 1 (perfectly reflecting surfaces, an
      upper bound)
    - validity_warning_ratio [3.0]: a WARNING is logged once if 'casimir_polder' is evaluated at
      r < ratio·(R_j + R_k), or 'proximity_force' at a gap d > min(R_j, R_k)/ratio; 0 switches the
      warning off
    """

    MODELS = ('casimir_polder', 'proximity_force')

    def __init__(self, model='casimir_polder', permittivity=None, ideal_mirrors=False, pairs=None,
                 validity_warning_ratio=3.0):
        super().__init__(pairs=pairs)
        if model not in self.MODELS:
            raise ValueError(f'Casimir model must be one of {self.MODELS}, not {model!r}')
        if ideal_mirrors and model != 'proximity_force':
            raise ValueError("ideal_mirrors=True applies to model='proximity_force' only")
        self.model = model
        self.permittivity = permittivity
        self.ideal_mirrors = bool(ideal_mirrors)
        self.validity_warning_ratio = float(validity_warning_ratio)
        self._warned = False
        self._plate_factors = {}

    def __repr__(self):
        return f'Casimir(model={self.model!r}, permittivity={self.permittivity})'

    def _ε(self, particle):
        ε = self.permittivity if self.permittivity is not None else np.real(particle.εr)
        return float(ε)

    @staticmethod
    def _radii(particle_j, particle_k):
        R_j, R_k = particle_radius(particle_j), particle_radius(particle_k)
        if R_j is None or R_k is None:
            raise ValueError('Casimir needs spherical particles with a radius r')
        return R_j, R_k

    def plate_factor(self, ε_j, ε_k):
        """η of the proximity-force model for these permittivities (cached) [·]"""
        if self.ideal_mirrors:
            return 1.0
        key = (ε_j, ε_k)
        if key not in self._plate_factors:
            self._plate_factors[key] = lifshitz_plate_factor(ε_j, ε_k)
        return self._plate_factors[key]

    def power_law(self, particle_j, particle_k):
        R_j, R_k = self._radii(particle_j, particle_k)
        ε_j, ε_k = self._ε(particle_j), self._ε(particle_k)
        if self.model == 'casimir_polder':
            α_j = R_j**3 * (ε_j - 1) / (ε_j + 2)
            α_k = R_k**3 * (ε_k - 1) / (ε_k + 2)
            return -23 * ħ * c / (4 * π) * α_j * α_k, 0.0, 7
        R_eff = R_j * R_k / (R_j + R_k)
        return -self.plate_factor(ε_j, ε_k) * π**3 * ħ * c * R_eff / 720, R_j + R_k, 2

    def check_separation(self, r, particle_j, particle_k):
        super().check_separation(r, particle_j, particle_k)
        R_j, R_k = self._radii(particle_j, particle_k)
        gap = r - R_j - R_k
        if gap <= 0:
            raise ValueError(f'Casimir: the spheres touch or overlap (gap {gap:.4g} m)')
        ratio = self.validity_warning_ratio
        if self._warned or ratio <= 0:
            return
        if self.model == 'casimir_polder' and r < ratio * (R_j + R_k):
            message = f'r = {r:.3g} m < {ratio:g}·(R_j + R_k): the dipole (Casimir-Polder) limit is not accurate here'
        elif self.model == 'proximity_force' and gap > min(R_j, R_k) / ratio:
            message = f'gap {gap:.3g} m > min(R)/{ratio:g}: the proximity-force approximation is not accurate here'
        else:
            return
        self._warned = True
        logger.warning('Casimir(model={!r}): {} (logged once per instance)', self.model, message)

"""
Mie theory for spheres beyond the dipole limit (optional backend: ``pip install levisim[mie]``)

levisim's particles use the quasi-static (Rayleigh) dipole polarisability with the radiation
reaction, α_eff = α/(1 - i k³α/(6πε0)), which is accurate while the size parameter x = k r is small.
For larger spheres this module evaluates the exact Mie solution with the `miepython` package
(Prahl, MIT licence, https://github.com/scottprahl/miepython) and gives

- the extinction, scattering and absorption cross sections σ = Q π r² (summing all multipoles)
- the dipole-equivalent polarisability from the electric-dipole Mie coefficient a₁ ::

      α_Mie = 6πi ε0 a₁ / k³

  which includes the radiation reaction exactly and reduces to α_eff for x → 0
  (a₁ ≈ -i(2x³/3)(εr - 1)/(εr + 2) + ...). It is the polarisability that sets the gradient force on
  a sphere in a weakly focused beam as long as the higher multipoles (b₁, a₂, ...) are negligible.

Conventions: Bohren and Huffman's coefficients with levisim's e^{-iωt} time dependence, so an
absorbing material has Im εr > 0 and Im α > 0. miepython takes the refractive index as m = n - iκ,
so the conjugate of levisim's index is passed. The sphere is in vacuum (or a dilute gas).

Agreement with the dipole model for the default bead (r = 60.1 nm, εr = 1.98, λ = 1064 nm,
x ≈ 0.355): see `compare_with_dipole` and the "Particles and materials" concepts page.
"""
from dataclasses import dataclass

import numpy as np

from levisim.core.constants import ε0, π

__all__ = [
    'MieComparison',
    'compare_with_dipole',
    'dipole_cross_sections',
    'mie_available',
    'mie_coefficients',
    'mie_cross_sections',
    'mie_dipole_polarisability',
]

_INSTALL_HINT = 'the Mie backend needs miepython: pip install "levisim[mie]" (or pip install miepython)'


def _miepython():
    try:
        import miepython
    except ImportError as error:  # pragma: no cover - exercised only without the extra
        raise ImportError(_INSTALL_HINT) from error
    return miepython


def mie_available():
    """Whether the optional Mie backend (miepython) can be imported [bool]"""
    try:
        _miepython()
    except ImportError:
        return False
    return True


def _refractive_index(εr):
    """√εr with Im n ≥ 0 (levisim's e^{-iωt}) for a scalar εr; anisotropic materials are refused"""
    if np.ndim(εr) != 0:
        raise ValueError('Mie theory is for isotropic spheres; this material has a tensor permittivity')
    n = np.sqrt(complex(εr))
    return n if n.imag >= 0 else -n


def mie_coefficients(r, εr, λ, n_max=None):
    """
    The Mie coefficients (a_n, b_n), n = 1, 2, ..., of a sphere of radius r [m] and relative
    permittivity εr in vacuum at vacuum wavelength λ [m], in Bohren and Huffman's convention for
    e^{-iωt} (Re a_n ≥ 0 and Re b_n ≥ 0 for passive spheres). n_max truncates the series (default:
    miepython's Wiscombe criterion, about x + 4x^(1/3) + 2 terms).
    [·]
    """
    miepython = _miepython()
    n = _refractive_index(εr)
    x = 2*π * float(r) / float(λ)
    a, b = miepython.coefficients(np.conj(n), x, n_pole=0 if n_max is None else int(n_max))
    a, b = np.atleast_1d(np.asarray(a, dtype=complex)), np.atleast_1d(np.asarray(b, dtype=complex))
    return a, b


def mie_cross_sections(r, εr, λ):
    """
    Extinction, scattering and absorption cross sections of the sphere from all Mie multipoles ::

        σ_ext = (2π/k²) Σ (2n + 1) Re(a_n + b_n),   σ_sca = (2π/k²) Σ (2n + 1)(|a_n|² + |b_n|²)

    Returns {'σ_ext', 'σ_sca', 'σ_abs'} [m²].
    """
    a, b = mie_coefficients(r, εr, λ)
    k = 2*π / float(λ)
    order = 2 * np.arange(1, a.size + 1) + 1
    σ_ext = 2*π / k**2 * float(np.sum(order * (a + b).real))
    σ_sca = 2*π / k**2 * float(np.sum(order * (np.abs(a)**2 + np.abs(b)**2)))
    return {'σ_ext': σ_ext, 'σ_sca': σ_sca, 'σ_abs': σ_ext - σ_sca}


def mie_dipole_polarisability(r, εr, λ):
    """
    Dipole-equivalent polarisability of the sphere from the electric-dipole Mie coefficient,
    α_Mie = 6πi ε0 a₁/k³ (radiation reaction included)
    [complex Fm²]
    """
    a, _ = mie_coefficients(r, εr, λ, n_max=1)
    k = 2*π / float(λ)
    return complex(6j * π * ε0 * a[0] / k**3)


def dipole_cross_sections(α_eff, λ):
    """
    Cross sections of a point dipole with effective polarisability α_eff [Fm²] at vacuum wavelength
    λ [m] ::

        σ_ext = k Im(α_eff)/ε0,   σ_sca = k⁴|α_eff|²/(6π ε0²)

    (equal for a lossless particle once α_eff includes the radiation reaction, the optical theorem).
    Returns {'σ_ext', 'σ_sca', 'σ_abs'} [m²].
    """
    k = 2*π / float(λ)
    σ_ext = k * float(np.imag(α_eff)) / ε0
    σ_sca = k**4 * float(np.abs(α_eff))**2 / (6*π * ε0**2)
    return {'σ_ext': σ_ext, 'σ_sca': σ_sca, 'σ_abs': σ_ext - σ_sca}


@dataclass(frozen=True)
class MieComparison:
    """
    Mie theory against levisim's dipole model for one sphere (`compare_with_dipole`)

    - r, λ, εr, x = k r: the sphere and the light
    - α_dipole: Clausius-Mossotti with radiation reaction (`Particle.polarisability_tensor`) [Fm²]
    - α_mie: 6πi ε0 a₁/k³ [Fm²]
    - mie, dipole: the cross sections {'σ_ext', 'σ_sca', 'σ_abs'} [m²]
    - multipole_fraction: the share of σ_sca not in the electric dipole a₁,
      ``1 - 3|a₁|²/Σ(2n+1)(|a_n|²+|b_n|²)`` [·]
    """
    r: float
    λ: float
    εr: complex
    x: float
    α_dipole: complex
    α_mie: complex
    mie: dict
    dipole: dict
    multipole_fraction: float

    @property
    def polarisability_ratio(self):
        """Re α_dipole / Re α_Mie, the ratio that sets the trap stiffness [·]"""
        return self.α_dipole.real / self.α_mie.real

    @property
    def scattering_ratio(self):
        """σ_sca(dipole)/σ_sca(Mie), the ratio that sets the recoil heating [·]"""
        return self.dipole['σ_sca'] / self.mie['σ_sca']


def compare_with_dipole(sphere, λ=None):
    """
    Compares Mie theory with levisim's dipole model for a `Sphere` (at its λ by default)
    [MieComparison]
    """
    λ = sphere.λ if λ is None else float(λ)
    εr = sphere.material.εr(λ)
    α_dipole = complex(np.asarray(sphere.polarisability_tensor(λ, radiation_reaction=True))[0, 0])
    a, b = mie_coefficients(sphere.r, εr, λ)
    order = 2 * np.arange(1, a.size + 1) + 1
    total = float(np.sum(order * (np.abs(a)**2 + np.abs(b)**2)))
    return MieComparison(
        r=sphere.r,
        λ=λ,
        εr=complex(εr),
        x=2*π * sphere.r / λ,
        α_dipole=α_dipole,
        α_mie=mie_dipole_polarisability(sphere.r, εr, λ),
        mie=mie_cross_sections(sphere.r, εr, λ),
        dipole=dipole_cross_sections(α_dipole, λ),
        multipole_fraction=1 - 3 * float(np.abs(a[0]))**2 / total,
    )

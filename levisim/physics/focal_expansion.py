"""
Second-order expansion of a tightly focused tweezer about its focus, for any linear polarisation

FocussingOptics evaluates the Richards-Wolf focal field (Novotny & Hecht, Principles of
Nano-Optics, §3.6) by quadrature at every point. Near the focus its three Bessel integrals
I00, I01, I02 have Taylor expansions whose coefficients are single θ-integrals ('moments')
⟨g⟩ = ∫₀^θmax f_w(θ)·cos^½θ·g(θ) dθ. For x-polarised input light, with Σ the field prefactor:

    E_x/Σ ≈ a0 + ikz·a1 - ½k²z²·a2 - ¼k²(x² + y²)·b0 + ⅛k²(x² - y²)·d0
    E_y/Σ ≈ ¼k²·d0·xy
    E_z/Σ ≈ -ik·e0·x

    a_n = ⟨sinθ(1 + cosθ)cosⁿθ⟩,  b0 = ⟨sin³θ(1 + cosθ)⟩,  d0 = ⟨sin³θ(1 - cosθ)⟩,  e0 = ⟨sin³θ⟩

so that |E|²/|Σ|² ≈ a0² - k²(κ_∥·x² + κ_⊥·y² + κ_z·z²) with

    κ_∥ = a0·b0/2 - a0·d0/4 - e0²      (along the polarisation)
    κ_⊥ = a0·b0/2 + a0·d0/4            (across it)
    κ_z = a0·a2 - a1²

The focus is softer along the polarisation because of the longitudinal field E_z and the
cross-polarised I02 term; this is where Ω_∥ < Ω_⊥ comes from for a round input beam.

TWO POLARISATIONS. The interference term of an x- and a y-polarised field is

    E^(x)·E^(y)*/|Σ|² ≈ k²·(κ_⊥ - κ_∥)·xy

exactly the anisotropy, and nothing else to this order (no x², y², z² or xz, yz terms). This is
forced by rotational symmetry: the field polarised at 45° is the x-polarised one rotated by 45°,
whose quadratic form has the cross term (κ_⊥ - κ_∥)·xy. In general, for polarisations ψ1, ψ2
measured from x,

    ∂²Re(E^(ψ1)·E^(ψ2)*)/|Σ|² = -2k²[cos(ψ1 - ψ2)·κ̄·1 + ½(κ_⊥ - κ_∥)·σ(ψ1 + ψ2)]   (transverse)
                               = -2k²·cos(ψ1 - ψ2)·κ_z                            (axial)

    σ(Ψ) = [[-cos Ψ, -sin Ψ], [-sin Ψ, cos Ψ]],  κ̄ = (κ_∥ + κ_⊥)/2

`levisim.multiphysics.two_tone_traps` uses this in units of measured trap frequencies.
"""
import numpy as np

from functools import cached_property
from scipy.integrate import quad


__all__ = [
    'FocalExpansion',
    'polarisation_matrix',
]


def polarisation_matrix(Ψ):
    """
    σ(Ψ) = [[-cos Ψ, -sin Ψ], [-sin Ψ, cos Ψ]], the polarisation-dependent part of a transverse Hessian
    """
    return np.array([[-np.cos(Ψ), -np.sin(Ψ)], [-np.sin(Ψ), np.cos(Ψ)]])


class FocalExpansion:
    """
    Taylor coefficients of a FocussingOptics focal field about the focus
        - optics: a FocussingOptics instance (wavelength, NA, filling factor, apodisation)
        - apodisation: 'levisim' uses FocussingOptics._fw as it stands; 'novotny' uses
          exp(-sin²θ / (f0²·sin²θmax)) (Novotny & Hecht Eq. 3.52). The two differ by sinθmax in the
          exponent, i.e. by an effective filling factor f0·sinθmax^½.
    """
    def __init__(self, optics, apodisation='levisim'):
        assert apodisation in ('levisim', 'novotny'), f'{apodisation=}'
        self.optics = optics
        self.apodisation = apodisation

    def __repr__(self):
        return f'FocalExpansion(NA={self.optics.NA}, f0={self.optics.f0:.3f}, apodisation={self.apodisation!r})'

    @property
    def info(self):
        """
        Prints information on the expansion
        """
        print(self.info_str)

    @property
    def info_str(self):
        """
        A string with information on the expansion
        """
        κ = self.curvatures
        return (
            f'{self!r}'
            f'\n    κ_∥, κ_⊥, κ_z = {κ["par"]:.6f}, {κ["perp"]:.6f}, {κ["z"]:.6f}'
            f'\n    Ω_⊥/Ω_∥ = {self.frequency_ratios["perp/par"]:.5f}'
            f'\n    Ω_⊥/Ω_z = {self.frequency_ratios["perp/z"]:.5f}'
        )

    # ------------------------------------------------------------------------------------------
    # MOMENTS
    # ------------------------------------------------------------------------------------------
    def _weight(self, θ):
        if self.apodisation == 'levisim':
            fw = self.optics._fw(θ)
        else:
            fw = np.exp(-(np.sin(θ)**2 / np.sin(self.optics.θ_max)**2) / self.optics.f0**2)
        return fw * np.sqrt(np.cos(θ))

    def _moment(self, g):
        return quad(lambda θ: self._weight(θ) * g(θ), 0, self.optics.θ_max, epsabs=0, epsrel=1e-12)[0]

    @cached_property
    def moments(self):
        """
        The θ-integrals a0, a1, a2, b0, d0, e0 [·]
        """
        s, c = np.sin, np.cos
        return {
            'a0': self._moment(lambda θ: s(θ) * (1 + c(θ))),
            'a1': self._moment(lambda θ: s(θ) * (1 + c(θ)) * c(θ)),
            'a2': self._moment(lambda θ: s(θ) * (1 + c(θ)) * c(θ)**2),
            'b0': self._moment(lambda θ: s(θ)**3 * (1 + c(θ))),
            'd0': self._moment(lambda θ: s(θ)**3 * (1 - c(θ))),
            'e0': self._moment(lambda θ: s(θ)**3),
        }

    @cached_property
    def curvatures(self):
        """
        κ_∥, κ_⊥, κ_z with |E|²/|Σ|² ≈ a0² - k²(κ_∥x² + κ_⊥y² + κ_z z²) for x-polarised light [·]
        """
        m = self.moments
        a0, a1, a2, b0, d0, e0 = (m[key] for key in ('a0', 'a1', 'a2', 'b0', 'd0', 'e0'))
        return {
            'par': a0*b0/2 - a0*d0/4 - e0**2,
            'perp': a0*b0/2 + a0*d0/4,
            'z': a0*a2 - a1**2,
        }

    @property
    def anisotropy(self):
        """
        κ_⊥ - κ_∥, also the xy-coefficient of E^(x)·E^(y)*/(k²|Σ|²) [·]
        """
        return self.curvatures['perp'] - self.curvatures['par']

    @property
    def frequency_ratios(self):
        """
        Ω_⊥/Ω_∥ and Ω_⊥/Ω_z of a single linearly polarised tone (harmonic approximation) [·]
        """
        κ = self.curvatures
        return {'perp/par': np.sqrt(κ['perp'] / κ['par']), 'perp/z': np.sqrt(κ['perp'] / κ['z'])}

    # ------------------------------------------------------------------------------------------
    # HESSIANS AT THE FOCUS (units of k²|Σ|²)
    # ------------------------------------------------------------------------------------------
    def interference_hessian(self, ψ1, ψ2):
        """
        3×3 Hessian of Re(E^(ψ1)·E^(ψ2)*) at the focus, for polarisations ψ1, ψ2 from x [k²|Σ|²]

        ψ1 = ψ2 = ψ gives the Hessian of the intensity of light polarised at ψ.
        """
        κ = self.curvatures
        κ̄ = (κ['par'] + κ['perp']) / 2
        H = np.zeros((3, 3))
        H[:2, :2] = -2 * (np.cos(ψ1 - ψ2) * κ̄ * np.eye(2) + 0.5 * self.anisotropy * polarisation_matrix(ψ1 + ψ2))
        H[2, 2] = -2 * np.cos(ψ1 - ψ2) * κ['z']
        return H

    def intensity_hessian(self, ψ):
        """
        3×3 Hessian of |E^(ψ)|² at the focus for light polarised at ψ from x [k²|Σ|²]
        """
        return self.interference_hessian(ψ, ψ)

    def numerical_hessian(self, ψ1, ψ2, step=0.02):
        """
        The same Hessian by finite differences of FocussingOptics' full focal field (for checks)
            - step: finite-difference step in units of 1/k [·]
        Uses the optics' own apodisation, so compare with apodisation='levisim'.
        """
        optics = self.optics

        def field(ψ, x, y, z):
            # rotate the x-polarised field: E^(ψ)(r) = R(ψ)·E^(x)(R(ψ)⁻¹r)
            c, s = np.cos(ψ), np.sin(ψ)
            xr, yr = c*x + s*y, -s*x + c*y
            Ex, Ey, Ez = (complex(optics._IE_00(component, xr, yr, z)) for component in 'xyz')
            return np.array([c*Ex - s*Ey, s*Ex + c*Ey, Ez])

        h = step / optics.k
        F = lambda r: np.real(np.vdot(field(ψ2, *r), field(ψ1, *r)))
        e = np.eye(3) * h
        H = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                H[i, j] = (F(e[i] + e[j]) - F(e[i] - e[j]) - F(-e[i] + e[j]) + F(-e[i] - e[j])) / (4 * h * h)
        return H / optics.k**2

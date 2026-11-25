"""
Static magnetic field sources behind one small interface, for `MagneticTrap` and later the magnet and
superconductor elements

Every source supplies ::

    B(r)       the field at r, r of shape (..., 3) → (..., 3)                           [T]
    gradB(r)   the Jacobian J_ij = ∂B_i/∂r_j, (..., 3) → (..., 3, 3)                   [T/m]
    hessB(r)   the second derivatives H_ijk = ∂²B_i/∂r_j∂r_k, (..., 3) → (..., 3, 3, 3)  [T/m²]

in the laboratory frame. A source that knows only B (a current loop, a magnet block, a user callable)
gets gradB and hessB from fourth-order central differences with steps set by its `length_scale`;
the polynomial sources (`LocalFieldExpansion`, `MultipoleField`) give them exactly.

Sources:

- `LocalFieldExpansion`: B = B0 + G·u + ½ C:uu about a point (value, gradient, curvature)
- `MultipoleField`: B = -μ0∇Φ from a sum of regular solid harmonics of the magnetic scalar potential,
  in the normalisation of Hsu et al., Sci. Rep. 6, 30125 (2016), doi:10.1038/srep30125, Eq. (1)-(4)
- `CurrentLoop`: a circular loop, exact field from complete elliptic integrals
- `CoilPair`: two coaxial loops (anti-Helmholtz, Helmholtz, or any currents)
- `CuboidMagnet`: a uniformly magnetised rectangular block, exact closed form
- `FieldSum`: the sum of several sources (``source_a + source_b`` builds one)
- `CallableField`: a user function B(r), with finite-difference derivatives

Every source can be placed with ``centre`` (the origin of its own coordinates, in the lab frame) and
``rotation`` (a 3×3 rotation matrix, own frame → lab frame); the transformation is applied to B, J
and H. `divergence` and `curl` evaluate ∇·B and ∇×B from gradB, which vanish for a static field in
a current-free region (Maxwell's equations); the tests check both for every source.
"""

from math import comb, factorial

import numpy as np
from scipy.special import ellipe, ellipk

from levisim.core.constants import μ0, π

__all__ = [
    'CallableField',
    'CoilPair',
    'CuboidMagnet',
    'CurrentLoop',
    'FieldSum',
    'LocalFieldExpansion',
    'MagneticFieldSource',
    'MultipoleField',
    'real_solid_harmonic',
]

# fourth-order central-difference stencils: first derivative (f(-2h), f(-h), f(h), f(2h))/h, and the
# offsets at which they are evaluated
_D1_OFFSETS = np.array([-2.0, -1.0, 1.0, 2.0])
_D1_WEIGHTS = np.array([1.0, -8.0, 8.0, -1.0]) / 12.0


def _as_points(r):
    """r as a float array of shape (..., 3)"""
    r = np.asarray(r, dtype=float)
    if r.shape[-1:] != (3,):
        raise ValueError(f'positions must have a last axis of length 3 (x, y, z), not shape {r.shape}')
    return r


def _as_rotation(rotation):
    """A 3×3 rotation matrix (proper, orthonormal), or None for the identity"""
    if rotation is None:
        return None
    R = np.asarray(rotation, dtype=float)
    if R.shape != (3, 3):
        raise ValueError(f'rotation must be a 3×3 matrix, not shape {R.shape}')
    if not np.allclose(R @ R.T, np.eye(3), atol=1e-10) or np.linalg.det(R) < 0:
        raise ValueError('rotation must be a proper rotation matrix (R Rᵀ = 𝟙, det R = +1)')
    return R


def _rotation_to_axis(axis):
    """A rotation matrix that takes the local z axis to the unit vector along axis (None if axis is z)"""
    a = np.asarray(axis, dtype=float)
    if a.shape != (3,) or not np.linalg.norm(a) > 0:
        raise ValueError(f'axis must be a non-zero 3-vector, not {axis!r}')
    a = a / np.linalg.norm(a)
    ez = np.array([0.0, 0.0, 1.0])
    c = float(a @ ez)
    if np.isclose(c, 1.0, rtol=0, atol=1e-15):
        return None
    if np.isclose(c, -1.0, rtol=0, atol=1e-15):
        return np.diag([1.0, -1.0, -1.0])  # a rotation by π about x
    v = np.cross(ez, a)
    s = np.linalg.norm(v)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / s**2)


class MagneticFieldSource:
    """
    Base class of the static magnetic field sources (see the module docstring)

    Subclasses implement `_B_local(u)` (the field in their own frame at local positions u of shape
    (..., 3)) and set `length_scale` [m], the distance over which their field varies, which sets the
    finite-difference steps of the default `_gradB_local` and `_hessB_local` (1e-3 and 3e-3 of it).
    Polynomial sources override those with exact derivatives.

    - centre: the origin of the source's own coordinates in the lab frame [m]
    - rotation: 3×3 rotation matrix from the source's frame to the lab frame (None: aligned)
    """

    length_scale = 1.0

    def __init__(self, centre=(0.0, 0.0, 0.0), rotation=None):
        self.centre = np.asarray(centre, dtype=float)
        if self.centre.shape != (3,):
            raise ValueError(f'centre must be a position (x, y, z), not shape {self.centre.shape}')
        self.rotation = _as_rotation(rotation)

    # ----------------------------------------------------------------------------------------------
    # frame transformations
    # ----------------------------------------------------------------------------------------------
    def _to_local(self, r):
        u = _as_points(r) - self.centre
        if self.rotation is not None:
            u = u @ self.rotation  # u_local = Rᵀ (r - centre), row-vector form
        return u

    def _vector_to_lab(self, v):
        return v if self.rotation is None else v @ self.rotation.T

    def _matrix_to_lab(self, M):
        if self.rotation is None:
            return M
        R = self.rotation
        return np.einsum('ia,...ab,jb->...ij', R, M, R)

    def _tensor3_to_lab(self, T):
        if self.rotation is None:
            return T
        R = self.rotation
        return np.einsum('ia,jb,kc,...abc->...ijk', R, R, R, T)

    # ----------------------------------------------------------------------------------------------
    # the interface
    # ----------------------------------------------------------------------------------------------
    def B(self, r):
        """
        Magnetic flux density at r (shape (..., 3))
        [T]
        """
        return self._vector_to_lab(self._B_local(self._to_local(r)))

    def gradB(self, r):
        """
        Jacobian J_ij = ∂B_i/∂r_j at r, shape (..., 3, 3)
        [T/m]
        """
        return self._matrix_to_lab(self._gradB_local(self._to_local(r)))

    def hessB(self, r):
        """
        Second derivatives H_ijk = ∂²B_i/∂r_j∂r_k at r, shape (..., 3, 3, 3)
        [T/m²]
        """
        return self._tensor3_to_lab(self._hessB_local(self._to_local(r)))

    def B_squared(self, r):
        """
        |B|² at r
        [T²]
        """
        B = self.B(r)
        return np.sum(B * B, axis=-1)

    def divergence(self, r):
        """
        ∇·B at r (zero for any magnetic field)
        [T/m]
        """
        return np.trace(self.gradB(r), axis1=-2, axis2=-1)

    def curl(self, r):
        """
        ∇×B at r (zero for a static field outside currents and magnetised matter)
        [T/m]
        """
        J = self.gradB(r)
        return np.stack([
            J[..., 2, 1] - J[..., 1, 2],
            J[..., 0, 2] - J[..., 2, 0],
            J[..., 1, 0] - J[..., 0, 1],
        ], axis=-1)

    # ----------------------------------------------------------------------------------------------
    # defaults: fourth-order central differences of the local field
    # ----------------------------------------------------------------------------------------------
    @property
    def gradient_step(self):
        """
        Step of the central differences of the default gradB
        [m]
        """
        return 1e-3 * self.length_scale

    @property
    def hessian_step(self):
        """
        Step of the central differences of gradB in the default hessB
        [m]
        """
        return 3e-3 * self.length_scale

    @staticmethod
    def _central_difference(function, u, h):
        """
        ∂f/∂u_k by fourth-order central differences, f(u) of shape (..., *S) → (..., *S, 3)
        """
        u = _as_points(u)
        steps = np.eye(3)[:, None, :] * (h * _D1_OFFSETS)[None, :, None]  # (3 directions, 4 offsets, 3)
        points = u[..., None, None, :] + steps  # (..., 3, 4, 3)
        values = function(points)  # (..., 3, 4, *S)
        extra = values.ndim - (u.ndim - 1) - 2
        weights = _D1_WEIGHTS.reshape((4,) + (1,) * extra)
        derivative = np.sum(values * weights, axis=u.ndim) / h  # (..., 3, *S)
        return np.moveaxis(derivative, u.ndim - 1, -1)  # (..., *S, 3)

    def _gradB_local(self, u):
        return self._central_difference(self._B_local, u, self.gradient_step)

    def _hessB_local(self, u):
        return self._central_difference(self._gradB_local, u, self.hessian_step)

    def _B_local(self, u):
        raise NotImplementedError(f'{type(self).__name__} does not implement _B_local(u)')

    def __add__(self, other):
        if not isinstance(other, MagneticFieldSource):
            return NotImplemented
        mine = self.sources if type(self) is FieldSum else [self]
        theirs = other.sources if type(other) is FieldSum else [other]
        return FieldSum([*mine, *theirs])

    def _pose_repr(self):
        parts = []
        if np.any(self.centre != 0):
            parts.append(f'centre={self.centre.tolist()}')
        if self.rotation is not None:
            parts.append('rotation=...')
        return (', ' + ', '.join(parts)) if parts else ''


####################################################################################################
# Polynomial sources (exact derivatives)
# --------------------------------------------------------------------------------------------------
class _Polynomial3:
    """
    A polynomial in (x, y, z), stored as {(i, j, k): coefficient} for x^i y^j z^k, with exact
    derivatives and vectorised evaluation
    """

    def __init__(self, terms=None):
        self.terms = {}
        for powers, coefficient in (terms or {}).items():
            self.add_term(powers, coefficient)

    def add_term(self, powers, coefficient):
        powers = tuple(int(p) for p in powers)
        value = self.terms.get(powers, 0.0) + coefficient
        if value == 0:
            self.terms.pop(powers, None)
        else:
            self.terms[powers] = value

    def __add__(self, other):
        result = _Polynomial3(self.terms)
        for powers, coefficient in other.terms.items():
            result.add_term(powers, coefficient)
        return result

    def __mul__(self, other):
        if np.isscalar(other):
            return _Polynomial3({p: c * other for p, c in self.terms.items()})
        result = _Polynomial3()
        for p1, c1 in self.terms.items():
            for p2, c2 in other.terms.items():
                result.add_term((p1[0] + p2[0], p1[1] + p2[1], p1[2] + p2[2]), c1 * c2)
        return result

    __rmul__ = __mul__

    def derivative(self, axis):
        result = _Polynomial3()
        for powers, coefficient in self.terms.items():
            if powers[axis] > 0:
                new = list(powers)
                new[axis] -= 1
                result.add_term(new, coefficient * powers[axis])
        return result

    def __call__(self, u):
        u = _as_points(u)
        total = np.zeros(u.shape[:-1])
        for (i, j, k), coefficient in self.terms.items():
            total = total + coefficient * u[..., 0]**i * u[..., 1]**j * u[..., 2]**k
        return total


class _PolynomialField(MagneticFieldSource):
    """A field whose components are polynomials of the local coordinates (exact gradB and hessB)"""

    def _set_components(self, components):
        self._components = components
        gradient = [[c.derivative(j) for j in range(3)] for c in components]
        hessian = [[[g.derivative(k) for k in range(3)] for g in row] for row in gradient]
        # one monomial basis for B, J and H, with coefficient matrices, so that a call evaluates every
        # monomial once: B = m(u) @ C_B, J = (m(u) @ C_J).reshape(3, 3), ...
        flat = {
            'B': components,
            'J': [g for row in gradient for g in row],
            'H': [h for row in hessian for column in row for h in column],
        }
        powers = sorted({p for polys in flat.values() for poly in polys for p in poly.terms})
        index = {p: i for i, p in enumerate(powers)}
        self._powers = np.array(powers, dtype=int).reshape(-1, 3)
        self._max_power = int(self._powers.max()) if len(powers) else 0
        self._coefficients = {}
        for key, polys in flat.items():
            C = np.zeros((len(powers), len(polys)))
            for column, poly in enumerate(polys):
                for p, c in poly.terms.items():
                    C[index[p], column] = c
            self._coefficients[key] = C

    def _monomials(self, u):
        u = _as_points(u)
        # u_a^k for k = 0..max_power, then the products x^i y^j z^k of the basis
        powers = u[..., None, :] ** np.arange(self._max_power + 1)[:, None]  # (..., k, 3)
        P = self._powers
        return powers[..., P[:, 0], 0] * powers[..., P[:, 1], 1] * powers[..., P[:, 2], 2]

    def _evaluate(self, u, key, shape):
        u = _as_points(u)
        if not len(self._powers):
            return np.zeros(u.shape[:-1] + shape)
        return (self._monomials(u) @ self._coefficients[key]).reshape(u.shape[:-1] + shape)

    def _B_local(self, u):
        return self._evaluate(u, 'B', (3,))

    def _gradB_local(self, u):
        return self._evaluate(u, 'J', (3, 3))

    def _hessB_local(self, u):
        return self._evaluate(u, 'H', (3, 3, 3))


class LocalFieldExpansion(_PolynomialField):
    """
    A field given by its local expansion to second order about `centre` ::

        B_i(r) = B0_i + G_ij u_j + ½ C_ijk u_j u_k,     u = r - centre

    - B0: the field at the centre [T]
    - gradient: G_ij = ∂B_i/∂r_j [T/m] (default zero)
    - curvature: C_ijk = ∂²B_i/∂r_j∂r_k [T/m²], symmetric in j, k (default zero)

    Maxwell's equations in a current-free region (∇·B = 0, ∇×B = 0) require G to be symmetric and
    traceless and C to be fully symmetric and traceless in every pair of indices. `maxwell_residuals`
    reports how far a given expansion is from that; the constructor warns if it is not consistent
    (check=True), because a force from an inconsistent field is not that of any real magnet. See
    `LocalFieldExpansion.axisymmetric` for the consistent form about a symmetry axis.
    """

    def __init__(self, B0=(0.0, 0.0, 0.0), gradient=None, curvature=None, centre=(0.0, 0.0, 0.0), check=True):
        super().__init__(centre=centre)
        self.B0 = np.asarray(B0, dtype=float).reshape(3)
        self.gradient = np.zeros((3, 3)) if gradient is None else np.asarray(gradient, dtype=float).reshape(3, 3)
        self.curvature = (
            np.zeros((3, 3, 3)) if curvature is None else np.asarray(curvature, dtype=float).reshape(3, 3, 3)
        )
        if not np.allclose(self.curvature, np.swapaxes(self.curvature, 1, 2), rtol=1e-12, atol=0):
            raise ValueError('curvature C_ijk must be symmetric in its last two indices (∂²B_i/∂r_j∂r_k)')
        components = []
        for i in range(3):
            polynomial = _Polynomial3({(0, 0, 0): self.B0[i]})
            for j in range(3):
                powers = [0, 0, 0]
                powers[j] += 1
                polynomial.add_term(powers, self.gradient[i, j])
                for k in range(3):
                    powers2 = [0, 0, 0]
                    powers2[j] += 1
                    powers2[k] += 1
                    polynomial.add_term(powers2, 0.5 * self.curvature[i, j, k])
            components.append(polynomial)
        self._set_components(components)
        if check:
            residuals = self.maxwell_residuals()
            if max(residuals.values()) > 1e-9:
                import warnings
                warnings.warn(
                    f'LocalFieldExpansion does not satisfy ∇·B = 0 and ∇×B = 0 (relative residuals {residuals});'
                    ' the gradient must be symmetric and traceless and the curvature symmetric and traceless',
                    UserWarning,
                    stacklevel=2,
                )

    @property
    def length_scale(self):
        """
        |B0|/|G| or |G|/|C| (whichever is defined), the scale of the expansion's variation; 1 m if the
        field is uniform
        [m]
        """
        nB, nG, nC = (np.linalg.norm(v) for v in (self.B0, self.gradient, self.curvature))
        if nG > 0 and nB > 0:
            return float(nB / nG)
        if nC > 0 and nG > 0:
            return float(nG / nC)
        return 1.0

    def maxwell_residuals(self):
        """
        Relative residuals of the Maxwell conditions on the expansion, {'gradient_trace',
        'gradient_antisymmetry', 'curvature_trace', 'curvature_antisymmetry'} (each 0 for a valid field)
        [·]
        """
        G, C = self.gradient, self.curvature
        nG = max(np.linalg.norm(G), 1e-300)
        nC = max(np.linalg.norm(C), 1e-300)
        return {
            'gradient_trace': float(abs(np.trace(G)) / nG),
            'gradient_antisymmetry': float(np.linalg.norm(G - G.T) / nG),
            # ∂_k(∇·B) = C_iik and ∂_k(∇×B) from C_ijk - C_jik
            'curvature_trace': float(np.linalg.norm(np.einsum('iik->k', C)) / nC),
            'curvature_antisymmetry': float(np.linalg.norm(C - np.swapaxes(C, 0, 1)) / nC),
        }

    @classmethod
    def axisymmetric(cls, B0, B1=0.0, B2=0.0, axis=(0, 0, 1), centre=(0.0, 0.0, 0.0)):
        """
        The second-order expansion of an axisymmetric field from its on-axis profile
        b(s) = B0 + B1 s + ½ B2 s² along the unit vector `axis` (s the coordinate along it) ::

            B_s = B0 + B1 s + ½ B2 (s² - ρ²/2),     B_ρ = -½ B1 ρ - ½ B2 s ρ

        (ρ the distance from the axis), which satisfies ∇·B = 0 and ∇×B = 0 exactly.

        - B0 [T], B1 = db/ds [T/m], B2 = d²b/ds² [T/m²]
        """
        a = np.asarray(axis, dtype=float)
        a = a / np.linalg.norm(a)
        P = np.eye(3) - np.outer(a, a)  # projector onto the plane normal to the axis
        # B = a (B0 + B1 s + ½B2(s² - ρ²/2)) - ½ (B1 + B2 s) ρ_vec,  s = a·u, ρ_vec = P u
        G = B1 * (np.outer(a, a) - 0.5 * P)
        C = np.zeros((3, 3, 3))
        # ∂²B_i/∂u_j∂u_k: from a_i ½B2 (s² - ½ρ²): a_i B2 (a_j a_k - ½ P_jk); from -½ B2 s ρ_i:
        # -½ B2 (a_j P_ik + a_k P_ij)
        C += B2 * np.einsum('i,jk->ijk', a, np.outer(a, a) - 0.5 * P)
        C -= 0.5 * B2 * (np.einsum('j,ik->ijk', a, P) + np.einsum('k,ij->ijk', a, P))
        return cls(B0=B0 * a, gradient=G, curvature=C, centre=centre)

    def __repr__(self):
        return (
            f'LocalFieldExpansion(B0={self.B0.tolist()} T, |G|={np.linalg.norm(self.gradient):.4g} T/m, '
            f'|C|={np.linalg.norm(self.curvature):.4g} T/m²{self._pose_repr()})'
        )


def real_solid_harmonic(l, m):
    """
    The regular solid harmonic r^l Y_lm(θ, φ) as a polynomial in (x, y, z), for the real,
    orthonormal spherical harmonics without the Condon-Shortley phase: m > 0 ∝ cos(mφ), m < 0 ∝
    sin(|m|φ), e.g. r² Y_2,-2 = ½√(15/π) xy and r³ Y_3,1 = ¼√(21/2π) x(4z² - x² - y²)

    Returned as the internal polynomial type (callable on points of shape (..., 3)).
    """
    l, m = int(l), int(m)
    if l < 0 or abs(m) > l:
        raise ValueError(f'need l ≥ 0 and |m| ≤ l, not l={l}, m={m}')
    am = abs(m)
    x = _Polynomial3({(1, 0, 0): 1.0})
    y = _Polynomial3({(0, 1, 0): 1.0})
    z = _Polynomial3({(0, 0, 1): 1.0})
    r2 = x * x + y * y + z * z
    # A_m + i B_m = (x + i y)^m
    A, Bp = _Polynomial3({(0, 0, 0): 1.0}), _Polynomial3()
    for _ in range(am):
        A, Bp = A * x + Bp * y * -1.0, A * y + Bp * x
    # Π_l^m(z, r) without its factorial normalisation
    Pi = _Polynomial3()
    for k in range((l - am) // 2 + 1):
        coefficient = (
            (-1)**k * 2.0**(-l) * comb(l, k) * comb(2*l - 2*k, l) * factorial(l - 2*k) / factorial(l - 2*k - am)
        )
        term = _Polynomial3({(0, 0, l - 2*k - am): coefficient})
        for _ in range(k):
            term = term * r2
        Pi = Pi + term
    norm = np.sqrt((2*l + 1) / (4*π) * factorial(l - am) / factorial(l + am))
    if m == 0:
        return Pi * norm
    return Pi * (A if m > 0 else Bp) * (norm * np.sqrt(2.0))


class MultipoleField(_PolynomialField):
    """
    A field from a magnetic scalar potential expanded in regular solid harmonics ::

        Φ = Σ_lm (a_lm r0 / (l μ0)) (r/r0)^l Y_lm(θ, φ),      B = -μ0 ∇Φ

    the normalisation of Hsu et al., Sci. Rep. 6, 30125 (2016), doi:10.1038/srep30125, Eq. (1)-(4),
    used by Slezak et al., New J. Phys. 20, 063028 (2018), doi:10.1088/1367-2630/aacac1, to model their
    magneto-gravitational trap (Y_lm: `real_solid_harmonic`). The field is a polynomial, so it satisfies
    ∇·B = 0 and ∇×B = 0 exactly, and gradB and hessB are exact.

    - coefficients: {(l, m): a_lm} [T]; a_lm is the field scale of the 2^l-pole at r = r0
    - r0: the reference radius (Hsu: the distance from the trap centre to the pole pieces) [m]
    """

    def __init__(self, coefficients, r0, centre=(0.0, 0.0, 0.0), rotation=None):
        super().__init__(centre=centre, rotation=rotation)
        self.coefficients = {(int(l), int(m)): float(a) for (l, m), a in dict(coefficients).items()}
        self.r0 = float(r0)
        if not self.r0 > 0:
            raise ValueError(f'r0 must be a positive length, not {r0}')
        potential = _Polynomial3()  # μ0 Φ
        for (l, m), a in self.coefficients.items():
            if l < 1:
                raise ValueError(f'l must be ≥ 1 (l = 0 is a constant potential), not {l}')
            potential = potential + real_solid_harmonic(l, m) * (a * self.r0**(1 - l) / l)
        self._potential = potential
        self._set_components([potential.derivative(j) * -1.0 for j in range(3)])

    @property
    def length_scale(self):
        """The reference radius r0 [m]"""
        return self.r0

    def scalar_potential(self, r):
        """
        Magnetic scalar potential Φ at r (B = -μ0∇Φ)
        [A]
        """
        return self._potential(self._to_local(r)) / μ0

    def __repr__(self):
        terms = ', '.join(f'({l}, {m}): {a:.6g}' for (l, m), a in self.coefficients.items())
        return f'MultipoleField({{{terms}}} T, r0={self.r0:.6g} m{self._pose_repr()})'


####################################################################################################
# Current loops
# --------------------------------------------------------------------------------------------------
class CurrentLoop(MagneticFieldSource):
    """
    A circular current loop of radius a carrying current I, centred at `centre` with its normal along
    `axis` (current anticlockwise about the axis, so B points along +axis at the centre)

    Exact field from the complete elliptic integrals K(k²), E(k²) (scipy's parameter convention; the
    standard result, e.g. Jackson, Classical Electrodynamics, 3rd ed., §5.5, written in cylindrical
    components, and checked against direct Biot-Savart quadrature in the tests) ::

        α² = a² + ρ² + z² - 2aρ,  β² = a² + ρ² + z² + 2aρ,  k² = 1 - α²/β²,  C = μ0 I/π
        B_ρ = C z / (2 α² β ρ) [(a² + ρ² + z²) E(k²) - α² K(k²)]
        B_z = C / (2 α² β) [(a² - ρ² - z²) E(k²) + α² K(k²)]

    Close to the axis (ρ < 1e-3 a), where B_ρ is a difference of nearly equal terms, the on-axis series
    B_z = b - ρ² b''/4, B_ρ = -ρ b'/2 + ρ³ b'''/16 with b(z) = μ0 I a²/(2(a² + z²)^{3/2}) is used
    instead (error O(ρ⁴/a⁴)). gradB and hessB come from central differences (`length_scale` = a).

    - radius a [m], current I [A]
    """

    def __init__(self, radius, current, centre=(0.0, 0.0, 0.0), axis=(0.0, 0.0, 1.0)):
        super().__init__(centre=centre, rotation=_rotation_to_axis(axis))
        self.radius = float(radius)
        self.current = float(current)
        self.axis = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
        if not self.radius > 0:
            raise ValueError(f'radius must be positive, not {radius}')

    @property
    def length_scale(self):
        """The loop radius [m]"""
        return self.radius

    def on_axis_field(self, s):
        """
        B along the axis at signed distance s from the centre, μ0 I a²/(2(a² + s²)^{3/2})
        [T]
        """
        a = self.radius
        return μ0 * self.current * a**2 / (2 * (a**2 + np.asarray(s, dtype=float)**2)**1.5)

    def _B_local(self, u):
        u = _as_points(u)
        a, I = self.radius, self.current
        x, y, z = u[..., 0], u[..., 1], u[..., 2]
        ρ = np.hypot(x, y)
        near_axis = ρ < 1e-3 * a
        Bρ = np.zeros_like(ρ)
        Bz = np.zeros_like(ρ)

        # exact elliptic-integral form away from the axis
        far = ~near_axis
        if np.any(far):
            ρf, zf = ρ[far], z[far]
            r2 = a**2 + ρf**2 + zf**2
            α2 = r2 - 2*a*ρf
            β2 = r2 + 2*a*ρf
            β = np.sqrt(β2)
            k2 = 1 - α2 / β2
            K, E = ellipk(k2), ellipe(k2)
            C = μ0 * I / π
            Bρ[far] = C * zf / (2 * α2 * β * ρf) * (r2 * E - α2 * K)
            Bz[far] = C / (2 * α2 * β) * ((a**2 - ρf**2 - zf**2) * E + α2 * K)

        # on-axis series near the axis
        if np.any(near_axis):
            ρn, zn = ρ[near_axis], z[near_axis]
            q = a**2 + zn**2
            b0 = μ0 * I * a**2 / 2
            b = b0 * q**-1.5
            b1 = -3 * b0 * zn * q**-2.5
            b2 = b0 * (-3 * q**-2.5 + 15 * zn**2 * q**-3.5)
            b3 = b0 * (45 * zn * q**-3.5 - 105 * zn**3 * q**-4.5)
            Bz[near_axis] = b - ρn**2 * b2 / 4
            Bρ[near_axis] = -ρn * b1 / 2 + ρn**3 * b3 / 16

        with np.errstate(invalid='ignore', divide='ignore'):
            cosφ = np.where(ρ > 0, x / np.where(ρ > 0, ρ, 1.0), 1.0)
            sinφ = np.where(ρ > 0, y / np.where(ρ > 0, ρ, 1.0), 0.0)
        return np.stack([Bρ * cosφ, Bρ * sinφ, Bz], axis=-1)

    def __repr__(self):
        return (
            f'CurrentLoop(radius={self.radius:.6g} m, current={self.current:.6g} A, '
            f'centre={self.centre.tolist()}, axis={self.axis.tolist()})'
        )


class FieldSum(MagneticFieldSource):
    """
    The sum of several field sources (each placed by its own centre and rotation); B, gradB and hessB
    are the sums of the sources' (exact where theirs are)

    - sources: a list of `MagneticFieldSource`
    """

    def __init__(self, sources):
        super().__init__()
        self.sources = list(sources)
        if not self.sources:
            raise ValueError('FieldSum needs at least one source')
        for source in self.sources:
            if not isinstance(source, MagneticFieldSource):
                raise TypeError(f'FieldSum sources must be MagneticFieldSource instances, not {type(source).__name__}')

    @property
    def length_scale(self):
        """The smallest length scale of the sources [m]"""
        return min(source.length_scale for source in self.sources)

    def B(self, r):
        return sum(source.B(r) for source in self.sources)

    def gradB(self, r):
        return sum(source.gradB(r) for source in self.sources)

    def hessB(self, r):
        return sum(source.hessB(r) for source in self.sources)

    def __repr__(self):
        return 'FieldSum([' + ', '.join(repr(source) for source in self.sources) + '])'


class CoilPair(FieldSum):
    """
    Two coaxial current loops of radius a at ±d/2 along `axis` from `centre`

    - anti=True (default): opposite currents (anti-Helmholtz), a field zero at the centre with on-axis
      gradient dB_s/ds = 3 μ0 I a² (d/2) / (a² + d²/4)^{5/2}; anti=False: equal currents (Helmholtz for
      d = a)
    - radius a [m], current I [A] (of the loop at +d/2), separation d [m]
    """

    def __init__(self, radius, current, separation, anti=True, centre=(0.0, 0.0, 0.0), axis=(0.0, 0.0, 1.0)):
        a = np.asarray(axis, dtype=float)
        a = a / np.linalg.norm(a)
        centre = np.asarray(centre, dtype=float)
        self.radius = float(radius)
        self.current = float(current)
        self.separation = float(separation)
        self.anti = bool(anti)
        self.axis = a
        upper = CurrentLoop(radius, current, centre=centre + 0.5 * separation * a, axis=a)
        lower = CurrentLoop(radius, -current if anti else current, centre=centre - 0.5 * separation * a, axis=a)
        super().__init__([upper, lower])
        self.centre = centre

    @classmethod
    def anti_helmholtz(cls, radius, current, centre=(0.0, 0.0, 0.0), axis=(0.0, 0.0, 1.0)):
        """Opposite currents at separation d = √3 a (the zero-curvature, most linear gradient)"""
        return cls(radius, current, np.sqrt(3) * radius, anti=True, centre=centre, axis=axis)

    @classmethod
    def helmholtz(cls, radius, current, centre=(0.0, 0.0, 0.0), axis=(0.0, 0.0, 1.0)):
        """Equal currents at separation d = a (the most uniform field)"""
        return cls(radius, current, radius, anti=False, centre=centre, axis=axis)

    def on_axis_field(self, s):
        """
        B along the axis at signed distance s from the centre (sum of the loops' on-axis fields)
        [T]
        """
        s = np.asarray(s, dtype=float)
        h = self.separation / 2
        return self.sources[0].on_axis_field(s - h) + self.sources[1].on_axis_field(s + h) * (-1 if self.anti else 1)

    def __repr__(self):
        return (
            f'CoilPair(radius={self.radius:.6g} m, current={self.current:.6g} A, separation={self.separation:.6g} m, '
            f'anti={self.anti}, centre={self.centre.tolist()}, axis={self.axis.tolist()})'
        )


####################################################################################################
# Uniformly magnetised rectangular block
# --------------------------------------------------------------------------------------------------
def _safe_log_sum(Y, R):
    """ln(Y + R) with R = √(X² + Y² + Z²) ≥ |Y|, without cancellation for Y < 0"""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(Y >= 0, np.log(Y + R), np.log((R**2 - Y**2) / (R - Y)))


def _block_field_Mz(u, half, M):
    """
    H field [A/m] of a block |u_x| ≤ half_x, |u_y| ≤ half_y, |u_z| ≤ half_z magnetised uniformly along
    +z with magnetisation M [A/m], from its two surface-charge sheets σ = ±M at u_z = ±half_z
    """
    x, y, z = u[..., 0], u[..., 1], u[..., 2]
    a, b, c = half
    Hx = np.zeros(u.shape[:-1])
    Hy = np.zeros(u.shape[:-1])
    Hz = np.zeros(u.shape[:-1])
    for zs, σ in ((c, M), (-c, -M)):
        Z = z - zs
        for i, xs in enumerate((-a, a)):
            X = x - xs
            for j, ys in enumerate((-b, b)):
                Y = y - ys
                R = np.sqrt(X**2 + Y**2 + Z**2)
                # Σ_ij (-1)^(i+j) G(x - x_i, y - y_j) with the rectangle's corners x_1 = -a, x_2 = a, ...;
                # (-1)^(i+j) is +1 for (x_1, y_1) and (x_2, y_2)
                sign = σ * (1.0 if i == j else -1.0) / (4 * π)
                with np.errstate(divide='ignore', invalid='ignore'):
                    Hz += sign * np.arctan(X * Y / (Z * R))
                Hx += sign * -_safe_log_sum(Y, R)
                Hy += sign * -_safe_log_sum(X, R)
    return np.stack([Hx, Hy, Hz], axis=-1)


class CuboidMagnet(MagneticFieldSource):
    """
    A uniformly magnetised rectangular block (a permanent magnet, or a saturated pole piece in a crude
    model), exact closed-form field

    The magnetisation M is replaced by surface charges σ = M·n̂ on the faces; each rectangular sheet
    has a closed-form H field (e.g. Furlani, Permanent Magnet and Electromechanical Devices (Academic
    Press, 2001), §4.3; Engel-Herbert and Hesjedal, J. Appl. Phys. 97, 074504 (2005),
    doi:10.1063/1.1883308) ::

        H_z = σ/4π Σ_ij (-1)^(i+j) arctan(X_i Y_j / (Z R_ij)),   H_x = -σ/4π Σ_ij (-1)^(i+j) ln(Y_j + R_ij)

    (X_i = x - x_i, ... relative to the sheet's corners). B = μ0 H outside the block and μ0 (H + M)
    inside. The field is singular on the edges, so stay a few gradient steps (1e-3 of the smallest
    side) away from them.

    - dimensions: the full side lengths (L_x, L_y, L_z) in the block's own frame [m]
    - magnetisation: M [A/m] in the block's frame, a 3-vector; or give remanence B_r [T] along
      `direction` instead (M = B_r/μ0)
    - centre, rotation: placement in the lab frame (rotation: block frame → lab frame)
    """

    def __init__(self, dimensions, magnetisation=None, remanence=None, direction=(0.0, 0.0, 1.0),
                 centre=(0.0, 0.0, 0.0), rotation=None):
        super().__init__(centre=centre, rotation=rotation)
        self.dimensions = np.asarray(dimensions, dtype=float).reshape(3)
        if np.any(self.dimensions <= 0):
            raise ValueError(f'dimensions must be positive side lengths, not {dimensions}')
        if (magnetisation is None) == (remanence is None):
            raise ValueError('give either magnetisation (a 3-vector, A/m) or remanence (T) with a direction')
        if magnetisation is not None:
            self.magnetisation = np.asarray(magnetisation, dtype=float).reshape(3)
        else:
            d = np.asarray(direction, dtype=float)
            self.magnetisation = float(remanence) / μ0 * d / np.linalg.norm(d)

    @property
    def length_scale(self):
        """The smallest side of the block [m]"""
        return float(np.min(self.dimensions))

    @property
    def magnetic_moment(self):
        """
        Total magnetic moment M V in the block's frame
        [A·m²]
        """
        return self.magnetisation * float(np.prod(self.dimensions))

    def _B_local(self, u):
        u = _as_points(u)
        half = self.dimensions / 2
        H = np.zeros(u.shape)
        # each magnetisation component in a cyclically permuted frame in which it points along z
        for axis in range(3):
            M = self.magnetisation[axis]
            if M == 0:
                continue
            order = [(axis + 1) % 3, (axis + 2) % 3, axis]  # local (x', y', z') = (u[order])
            H_perm = _block_field_Mz(u[..., order], half[order], M)
            H[..., order] += H_perm
        inside = np.all(np.abs(u) < half, axis=-1)
        B = μ0 * H
        if np.any(inside):
            B[inside] += μ0 * self.magnetisation
        return B

    def __repr__(self):
        return (
            f'CuboidMagnet(dimensions={self.dimensions.tolist()} m, magnetisation={self.magnetisation.tolist()} A/m'
            f'{self._pose_repr()})'
        )


####################################################################################################
# User callables
# --------------------------------------------------------------------------------------------------
class CallableField(MagneticFieldSource):
    """
    A field given by a user function

    - B: callable B(r) → field [T]; it must accept positions of shape (..., 3) and return (..., 3)
    - gradB: optional callable giving J_ij = ∂B_i/∂r_j, (..., 3) → (..., 3, 3) [T/m]; by default from
      fourth-order central differences of B with step 1e-3·length_scale
    - hessB: optional callable for H_ijk [T/m²]; by default central differences of gradB
    - length_scale: the distance over which the field varies [m] (sets the difference steps)
    """

    def __init__(self, B, gradB=None, hessB=None, length_scale=1e-3, centre=(0.0, 0.0, 0.0), rotation=None):
        super().__init__(centre=centre, rotation=rotation)
        if not callable(B):
            raise TypeError('B must be a callable B(r)')
        self._B_function = B
        self._gradB_function = gradB
        self._hessB_function = hessB
        self._length_scale = float(length_scale)
        if not self._length_scale > 0:
            raise ValueError(f'length_scale must be positive, not {length_scale}')

    @property
    def length_scale(self):
        """The length scale given at construction [m]"""
        return self._length_scale

    def _B_local(self, u):
        u = _as_points(u)
        B = np.asarray(self._B_function(u), dtype=float)
        if B.shape != u.shape:
            raise ValueError(f'the B callable returned shape {B.shape} for positions of shape {u.shape}')
        return B

    def _gradB_local(self, u):
        if self._gradB_function is None:
            return super()._gradB_local(u)
        return np.asarray(self._gradB_function(_as_points(u)), dtype=float)

    def _hessB_local(self, u):
        if self._hessB_function is None:
            return super()._hessB_local(u)
        return np.asarray(self._hessB_function(_as_points(u)), dtype=float)

    def __repr__(self):
        return f'CallableField({getattr(self._B_function, "__name__", "B")}, length_scale={self._length_scale:.4g} m{self._pose_repr()})'

"""
Compiled (numba) forces of the static and rf force elements

A trap's 'field' elements are packed once per run into an element program, a few flat arrays that
the compiled drifts read (`encode_elements`):

- the trap's Gaussian-beam tweezer (the arithmetic of `Trap.Ftw`)
- `PaulTrap`: F_j = -Q [c^dc_j + c^rf_j cos(Ω_rf t + φ_rf)] (r_j - c_j)
- `MagneticTrap`: F = (V/μ0) Jᵀ χ_eff B, with B and J = ∂B/∂r of its source, which may be a
  `LocalFieldExpansion` or `MultipoleField` (polynomials, exact derivatives), a `CurrentLoop`
  (complete elliptic integrals by the arithmetic-geometric mean, J by the same fourth-order central
  differences as NumPy), a `CuboidMagnet` (closed form B and J), or a `FieldSum`/`CoilPair` of these,
  each placed by its centre and rotation
- `HarmonicPotential` (F = -K·(r - c)) and `UniformElectricField` (F = Q·E)

Every expression is that of the element's NumPy `force`, so the compiled and NumPy forces agree to
round-off: ≲ 1e-15 relative for the Paul trap, harmonic and polynomial fields; ≲ 1e-13 for a current
loop, whose J is a central difference (step 10⁻³ of the radius) of fields that scipy's elliptic
integrals and the arithmetic-geometric mean give to ~1e-16; the cuboid's sums over its eight corners
are added in a different order. `CallableField` sources, element subclasses and other elements have
no compiled force (the NumPy loop is used).

The program, for N particles::

    ranges (N, 2): rows [start, end) of `rows` for each particle (start = end: no program, the
        drift uses its own single-tweezer or override_Ω_dict path)
    rows (E, 4) int64: [code, offset into data, first source, end source]
    sources (S, 4) int64: [kind, offset into data, number of monomials, 0]
    data (D,) float64: the parameters, at the offsets above
"""

import math

import numpy as np
from numba import njit

__all__ = [
    'ElementProgram',
    'empty_program',
    'encode_elements',
    'field_force',
]

# element codes
TWEEZER, PAUL, MAGNETIC, HARMONIC, UNIFORM_E = 0, 1, 2, 3, 4
# magnetic source kinds
POLYNOMIAL, LOOP, CUBOID = 0, 1, 2


class ElementProgram:
    """The packed field elements of N particles (see the module docstring)"""

    def __init__(self, ranges, rows, sources, data):
        self.ranges = np.ascontiguousarray(ranges, dtype=np.int64).reshape(-1, 2)
        self.rows = np.ascontiguousarray(rows, dtype=np.int64).reshape(-1, 4)
        self.sources = np.ascontiguousarray(sources, dtype=np.int64).reshape(-1, 4)
        self.data = np.ascontiguousarray(data, dtype=np.float64).reshape(-1)

    @property
    def arrays(self):
        """(rows, sources, data), the tuple the compiled drifts take"""
        return self.rows, self.sources, self.data


def empty_program(num_particles):
    """A program with no elements for any particle"""
    return ElementProgram(np.zeros((num_particles, 2)), np.zeros((0, 4)), np.zeros((0, 4)), np.zeros(1))


class _Builder:
    def __init__(self):
        self.rows, self.sources, self.data = [], [], [0.0]

    def put(self, values):
        offset = len(self.data)
        self.data.extend(float(v) for v in np.ravel(values))
        return offset


def _flatten_sources(source):
    """The primitive sources of a (possibly nested) FieldSum, or raises ValueError with the reason"""
    from levisim.physics.fields.magnetic_sources import (
        CoilPair,
        CuboidMagnet,
        CurrentLoop,
        FieldSum,
        LocalFieldExpansion,
        MultipoleField,
    )
    if type(source) in (FieldSum, CoilPair):
        return [primitive for part in source.sources for primitive in _flatten_sources(part)]
    if type(source) in (LocalFieldExpansion, MultipoleField, CurrentLoop, CuboidMagnet):
        return [source]
    raise ValueError(f'no compiled field for the magnetic source {type(source).__name__}')


def _put_source(builder, source):
    from levisim.core.constants import μ0
    from levisim.physics.fields.magnetic_sources import CuboidMagnet, CurrentLoop
    R = np.eye(3) if source.rotation is None else source.rotation
    offset = builder.put(np.concatenate([source.centre, np.ravel(R), [0.0 if source.rotation is None else 1.0, μ0]]))
    if isinstance(source, CurrentLoop):
        builder.put([source.radius, source.current, source.gradient_step])
        builder.sources.append([LOOP, offset, 0, 0])
    elif isinstance(source, CuboidMagnet):
        builder.put(np.concatenate([source.dimensions / 2, source.magnetisation]))
        builder.sources.append([CUBOID, offset, 0, 0])
    else:
        powers = source._powers
        builder.put(np.concatenate([np.ravel(powers).astype(float), np.ravel(source._coefficients['B']),
                                    np.ravel(source._coefficients['J'])]))
        builder.sources.append([POLYNOMIAL, offset, len(powers), 0])


def _put_element(builder, trap, element, allow_time_dependent):
    from levisim.physics.fields.electric import PaulTrap, UniformElectricField
    from levisim.physics.fields.elements import HarmonicPotential
    from levisim.physics.fields.magnetic import MagneticTrap
    from levisim.physics.tweezer import Tweezer
    bead = trap.bead
    kind = type(element)
    if kind is Tweezer and element is trap.tweezer:
        α_eff = trap.α_eff
        offset = builder.put([element.E0, element.zR, element.wx0, element.wy0, element.wz0, np.real(α_eff)])
        builder.rows.append([TWEEZER, offset, 0, 0])
    elif kind is PaulTrap:
        if not allow_time_dependent:
            raise ValueError('no compiled kernel for the time-dependent PaulTrap here')
        offset = builder.put(np.concatenate([[bead.q], element.dc_curvatures(), element.rf_curvatures(),
                                             [element.Ω_rf, element.φ_rf], element.centre]))
        if bead.q == 0:
            element.force(np.zeros(3), 0.0, bead)  # raises the element's own error for a neutral particle
        builder.rows.append([PAUL, offset, 0, 0])
    elif kind is MagneticTrap:
        primitives = _flatten_sources(element.source)
        scale, χ = element._response(bead)
        offset = builder.put(np.concatenate([[scale], np.ravel(χ)]))
        first = len(builder.sources)
        for source in primitives:
            _put_source(builder, source)
        builder.rows.append([MAGNETIC, offset, first, len(builder.sources)])
    elif kind is HarmonicPotential:
        offset = builder.put(np.concatenate([np.ravel(element.K), element.centre]))
        builder.rows.append([HARMONIC, offset, 0, 0])
    elif kind is UniformElectricField:
        offset = builder.put(element.force(np.zeros(3), 0.0, bead))
        builder.rows.append([UNIFORM_E, offset, 0, 0])
    else:
        raise ValueError(f'no compiled force for the {element.label} element')


def encode_elements(traps, field_elements, allow_time_dependent=True):
    """
    The ElementProgram of the given 'field' elements of each trap (field_elements[n] for traps[n],
    an empty list for a particle that uses its drift's own single-tweezer or override path)

    Raises ValueError with the reason when an element has no compiled force.
    """
    builder = _Builder()
    ranges = []
    for trap, elements in zip(traps, field_elements):
        start = len(builder.rows)
        for element in elements:
            _put_element(builder, trap, element, allow_time_dependent)
        ranges.append([start, len(builder.rows)])
    return ElementProgram(ranges, builder.rows, builder.sources, builder.data)


def unsupported_reason(traps, field_elements, allow_time_dependent=True):
    """Why encode_elements cannot pack these elements ('' if it can)"""
    try:
        encode_elements(traps, field_elements, allow_time_dependent)
    except ValueError as error:
        return str(error)
    return ''


# --------------------------------------------------------------------------------------------------
# compiled magnetic sources
# --------------------------------------------------------------------------------------------------
@njit(cache=True, error_model='numpy')
def _ellipke(m):
    """
    Complete elliptic integrals K(m), E(m) of parameter m = k² (scipy's ellipk, ellipe), 0 ≤ m < 1,
    by the arithmetic-geometric mean: K = π/(2 a_∞), E = K (1 - Σ_n 2^(n-1) c_n²) (Abramowitz and
    Stegun, Handbook of Mathematical Functions, §17.6)
    """
    a = 1.0
    b = math.sqrt(1.0 - m)
    c = math.sqrt(m)
    total = 0.5 * c * c
    power = 0.5
    for _ in range(40):
        if c <= 1e-17 * a:
            break
        c = 0.5 * (a - b)
        a, b = 0.5 * (a + b), math.sqrt(a * b)
        power *= 2.0
        total += power * c * c
    K = math.pi / (2.0 * a)
    return K, K * (1.0 - total)


@njit(cache=True, error_model='numpy')
def _loop_B_local(u0, u1, u2, a, current, μ0, out):
    """CurrentLoop._B_local at one local point"""
    ρ = math.hypot(u0, u1)
    z = u2
    if ρ < 1e-3 * a:
        q = a**2 + z**2
        b0 = μ0 * current * a**2 / 2
        b = b0 * q**-1.5
        b1 = -3 * b0 * z * q**-2.5
        b2 = b0 * (-3 * q**-2.5 + 15 * z**2 * q**-3.5)
        b3 = b0 * (45 * z * q**-3.5 - 105 * z**3 * q**-4.5)
        Bz = b - ρ**2 * b2 / 4
        Bρ = -ρ * b1 / 2 + ρ**3 * b3 / 16
    else:
        r2 = a**2 + ρ**2 + z**2
        α2 = r2 - 2*a*ρ
        β2 = r2 + 2*a*ρ
        β = math.sqrt(β2)
        k2 = 1 - α2 / β2
        K, E = _ellipke(k2)
        C = μ0 * current / math.pi
        Bρ = C * z / (2 * α2 * β * ρ) * (r2 * E - α2 * K)
        Bz = C / (2 * α2 * β) * ((a**2 - ρ**2 - z**2) * E + α2 * K)
    if ρ > 0:
        cosφ = u0 / ρ
        sinφ = u1 / ρ
    else:
        cosφ = 1.0
        sinφ = 0.0
    out[0] = Bρ * cosφ
    out[1] = Bρ * sinφ
    out[2] = Bz


@njit(cache=True, error_model='numpy')
def _loop_BJ_local(u, a, current, h, μ0, B, J):
    """B and the fourth-order central-difference Jacobian of CurrentLoop at one local point"""
    _loop_B_local(u[0], u[1], u[2], a, current, μ0, B)
    offsets = (-2.0, -1.0, 1.0, 2.0)
    weights = (1.0 / 12.0, -8.0 / 12.0, 8.0 / 12.0, -1.0 / 12.0)
    value = np.empty(3)
    point = np.empty(3)
    for k in range(3):
        acc0 = 0.0
        acc1 = 0.0
        acc2 = 0.0
        for o in range(4):
            point[0] = u[0]
            point[1] = u[1]
            point[2] = u[2]
            point[k] += h * offsets[o]
            _loop_B_local(point[0], point[1], point[2], a, current, μ0, value)
            acc0 += value[0] * weights[o]
            acc1 += value[1] * weights[o]
            acc2 += value[2] * weights[o]
        J[0, k] = acc0 / h
        J[1, k] = acc1 / h
        J[2, k] = acc2 / h


@njit(cache=True, error_model='numpy')
def _log_sum(A, B2, R):
    if A >= 0:
        return math.log(abs(A) + R)
    return math.log(B2) - math.log(R + abs(A))


@njit(cache=True, error_model='numpy')
def _inverse_sum(A, B2, R):
    if A >= 0:
        return 1 / (R * (R + abs(A)))
    return (R + abs(A)) / (R * B2)


@njit(cache=True, error_model='numpy')
def _block_Mz(u0, u1, u2, ha, hb, hc, M, H, Jp):
    """H field and Jacobian of a block magnetised along z (_block_field_Mz, _block_jacobian_Mz)"""
    for i in range(3):
        H[i] = 0.0
        for j in range(3):
            Jp[i, j] = 0.0
    for corner in range(8):
        xs = -ha if corner in (0, 1, 4, 5) else ha
        ys = -hb if corner % 2 == 0 else hb
        zs = hc if corner < 4 else -hc
        sign = 1.0 if corner in (0, 3, 5, 6) else -1.0
        w = sign * M / (4 * math.pi)
        X = u0 - xs
        Y = u1 - ys
        Z = u2 - zs
        R2 = X**2 + Y**2 + Z**2
        R = math.sqrt(R2)
        H[0] += -_log_sum(Y, X**2 + Z**2, R) * w
        H[1] += -_log_sum(X, Y**2 + Z**2, R) * w
        H[2] += math.atan(X * Y / (Z * R)) * w
        inv_y = _inverse_sum(Y, X**2 + Z**2, R)
        inv_x = _inverse_sum(X, Y**2 + Z**2, R)
        Jp[0, 0] += -X * inv_y * w
        Jp[0, 1] += -1 / R * w
        Jp[0, 2] += -Z * inv_y * w
        Jp[1, 0] += -1 / R * w
        Jp[1, 1] += -Y * inv_x * w
        Jp[1, 2] += -Z * inv_x * w
        Jp[2, 0] += Y * Z / ((X**2 + Z**2) * R) * w
        Jp[2, 1] += X * Z / ((Y**2 + Z**2) * R) * w
        Jp[2, 2] += -X * Y * (X**2 + Y**2 + 2 * Z**2) / ((X**2 * Y**2 + Z**2 * R2) * R) * w


@njit(cache=True, error_model='numpy')
def _cuboid_BJ_local(u, data, offset, μ0, B, J):
    """CuboidMagnet._B_local and _gradB_local at one local point"""
    H = np.zeros(3)
    Jt = np.zeros((3, 3))
    Hp = np.empty(3)
    Jp = np.empty((3, 3))
    order = np.empty(3, dtype=np.int64)
    for axis in range(3):
        M = data[offset + 3 + axis]
        if M == 0:
            continue
        order[0] = (axis + 1) % 3
        order[1] = (axis + 2) % 3
        order[2] = axis
        _block_Mz(u[order[0]], u[order[1]], u[order[2]], data[offset + order[0]], data[offset + order[1]],
                  data[offset + order[2]], M, Hp, Jp)
        for i in range(3):
            H[order[i]] += Hp[i]
            for j in range(3):
                Jt[order[i], order[j]] += Jp[i, j]
    inside = True
    for i in range(3):
        if not abs(u[i]) < data[offset + i]:
            inside = False
    for i in range(3):
        B[i] = μ0 * H[i]
        if inside:
            B[i] += μ0 * data[offset + 3 + i]
        for j in range(3):
            J[i, j] = μ0 * Jt[i, j]


@njit(cache=True, error_model='numpy')
def _polynomial_BJ_local(u, data, offset, n, B, J):
    """_PolynomialField._B_local and _gradB_local at one local point (monomials m(u) @ C)"""
    for i in range(3):
        B[i] = 0.0
        for j in range(3):
            J[i, j] = 0.0
    C_B = offset + 3 * n
    C_J = C_B + 3 * n
    for row in range(n):
        monomial = (u[0] ** data[offset + 3*row] * u[1] ** data[offset + 3*row + 1]
                    * u[2] ** data[offset + 3*row + 2])
        for i in range(3):
            B[i] += monomial * data[C_B + 3*row + i]
        for i in range(9):
            J[i // 3, i % 3] += monomial * data[C_J + 9*row + i]


@njit(cache=True, error_model='numpy')
def magnetic_field(r, sources, first, end, data, B, J):
    """
    B and J = ∂B/∂r (lab frame) of the sources [first, end) at r, summed (MagneticFieldSource.B,
    gradB; FieldSum)
    """
    for i in range(3):
        B[i] = 0.0
        for j in range(3):
            J[i, j] = 0.0
    u = np.empty(3)
    d = np.empty(3)
    Bs = np.empty(3)
    Js = np.empty((3, 3))
    for s in range(first, end):
        kind = sources[s, 0]
        offset = sources[s, 1]
        rotated = data[offset + 12] != 0.0
        μ0 = data[offset + 13]
        for i in range(3):
            d[i] = r[i] - data[offset + i]
        if rotated:
            # u = (r - centre) @ R
            for a in range(3):
                u[a] = d[0] * data[offset + 3 + a] + d[1] * data[offset + 6 + a] + d[2] * data[offset + 9 + a]
        else:
            for a in range(3):
                u[a] = d[a]
        body = offset + 14
        if kind == LOOP:
            _loop_BJ_local(u, data[body], data[body + 1], data[body + 2], μ0, Bs, Js)
        elif kind == CUBOID:
            _cuboid_BJ_local(u, data, body, μ0, Bs, Js)
        else:
            _polynomial_BJ_local(u, data, body, sources[s, 2], Bs, Js)
        if rotated:
            # B = R B_local, J = R J_local Rᵀ
            for i in range(3):
                acc = 0.0
                for a in range(3):
                    acc += data[offset + 3 + 3*i + a] * Bs[a]
                B[i] += acc
            for i in range(3):
                for j in range(3):
                    acc = 0.0
                    for a in range(3):
                        for b in range(3):
                            acc += data[offset + 3 + 3*i + a] * Js[a, b] * data[offset + 3 + 3*j + b]
                    J[i, j] += acc
        else:
            for i in range(3):
                B[i] += Bs[i]
                for j in range(3):
                    J[i, j] += Js[i, j]


# --------------------------------------------------------------------------------------------------
# the element forces
# --------------------------------------------------------------------------------------------------
@njit(cache=True, error_model='numpy')
def field_force(t, x, y, z, start, end, rows, sources, data, out):
    """The sum of the forces of the program rows [start, end) at (x, y, z), time t, into out (3,)"""
    Fx = 0.0
    Fy = 0.0
    Fz = 0.0
    for e in range(start, end):
        code = rows[e, 0]
        o = rows[e, 1]
        if code == TWEEZER:
            E0, zR, wx0, wy0, wz0, aRe = data[o], data[o + 1], data[o + 2], data[o + 3], data[o + 4], data[o + 5]
            w = np.sqrt(1 + (z / zR)**2)
            wx = wx0 * w
            wy = wy0 * w
            wz = wz0 * w
            ellipse = (x / wx)**2 + (y / wy)**2
            E = (E0 / w) * np.exp(-ellipse)
            U = -(aRe / 2) * E**2
            Fx = Fx + 0.5 * (4 * x / wx**2) * U
            Fy = Fy + 0.5 * (4 * y / wy**2) * U
            Fz = Fz + 0.5 * (4 * z / wz**2) * U * (1 - 2 * ellipse)
        elif code == PAUL:
            Q = data[o]
            c = np.cos(data[o + 7] * t + data[o + 8])
            Fx = Fx + -Q * (data[o + 1] + data[o + 4] * c) * (x - data[o + 9])
            Fy = Fy + -Q * (data[o + 2] + data[o + 5] * c) * (y - data[o + 10])
            Fz = Fz + -Q * (data[o + 3] + data[o + 6] * c) * (z - data[o + 11])
        elif code == MAGNETIC:
            r = np.empty(3)
            B = np.empty(3)
            J = np.empty((3, 3))
            r[0] = x
            r[1] = y
            r[2] = z
            magnetic_field(r, sources, rows[e, 2], rows[e, 3], data, B, J)
            scale = data[o]
            v0 = data[o + 1] * B[0] + data[o + 2] * B[1] + data[o + 3] * B[2]
            v1 = data[o + 4] * B[0] + data[o + 5] * B[1] + data[o + 6] * B[2]
            v2 = data[o + 7] * B[0] + data[o + 8] * B[1] + data[o + 9] * B[2]
            Fx = Fx + scale * (J[0, 0] * v0 + J[1, 0] * v1 + J[2, 0] * v2)
            Fy = Fy + scale * (J[0, 1] * v0 + J[1, 1] * v1 + J[2, 1] * v2)
            Fz = Fz + scale * (J[0, 2] * v0 + J[1, 2] * v1 + J[2, 2] * v2)
        elif code == HARMONIC:
            dx = x - data[o + 9]
            dy = y - data[o + 10]
            dz = z - data[o + 11]
            Fx = Fx + (-data[o] * dx - data[o + 1] * dy - data[o + 2] * dz)
            Fy = Fy + (-data[o + 3] * dx - data[o + 4] * dy - data[o + 5] * dz)
            Fz = Fz + (-data[o + 6] * dx - data[o + 7] * dy - data[o + 8] * dz)
        else:
            Fx = Fx + data[o]
            Fy = Fy + data[o + 1]
            Fz = Fz + data[o + 2]
    out[0] = Fx
    out[1] = Fy
    out[2] = Fz

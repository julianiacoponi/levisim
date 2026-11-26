"""
Free-molecular gas friction and diffusion of rigid particles of any convex shape

Martinetz, Hornberger and Stickler, Phys. Rev. E 97, 052112 (2018), doi:10.1103/PhysRevE.97.052112,
derive the rotranslational friction and diffusion tensors of a rigid body in a dilute gas (mean
free path ≫ particle) whose molecules are reflected either specularly or diffusely, the diffuse
fraction being the accommodation coefficient αc, diffusely re-emitted molecules leaving at the
surface temperature Ts = γs² T. For a particle moving slowly compared with the gas molecules, their
Eqs. (40) and (41) are ::

    Γ = n_g √(m_g kB T/2π) ∮ dA { c₁ (n; r×n)⊗(n; r×n) + αc [[𝟙, -r̂], [r̂, -r̂²]] } T⁻¹
    D = kB T n_g √(m_g kB T/2π) ∮ dA { c₂ (n; r×n)⊗(n; r×n) + c₃ [[𝟙, -r̂], [r̂, -r̂²]] }

    c₁ = 4 - 3αc + π αc γs/2,   c₂ = 4 - (7 - γs²) αc/2 + π αc γs/2,   c₃ = (1 + γs²) αc/2

over the particle surface, with r the surface point relative to the centre of mass, n the outward
normal, r̂ the matrix of r× (r̂a = r × a), n_g = P/(kB T) the gas number density and
T = diag(m_n 𝟙, I) (mass and body-frame inertia tensor). Γ (6×6, [2π·Hz] blocks) acts on the
momenta (P, J): dP/dt = -Γ_cm P - Γ_tr J, dJ/dt = -Γ_rt P - Γ_rot J. D is the momentum diffusion
matrix, d〈(P, J)(P, J)ᵀ〉/dt = 2D, which equals kB T Γ T for γs = 1.

ACCOMMODATION MODEL. levisim uses this mixed specular/diffuse model throughout; the defaults
αc = 1 (all molecules diffusely re-emitted) and γs = 1 (surface at the gas temperature) are those
of `Trap.Γth` (the Epstein drag with the (1 + π/8) factor).

ASSUMPTIONS. Every surface element receives the full, unshadowed thermal flux and each molecule
hits the particle once: exact for convex bodies (spheres, spheroids, ellipsoids, cylinders), an
approximation for non-convex ones (a `Dumbbell` near its neck). The particle is much smaller than
the mean free path and moves much slower than the gas molecules.

Here the surface integrals are evaluated numerically on Gauss quadratures (exact for spheres and
cylinders, spectrally convergent for ellipsoids) once per shape and cached; the sphere and cylinder
closed forms of Table I of the paper are in `Sphere` and `Cylinder`, and the tests check them
against these integrals.
"""
from functools import lru_cache

import numpy as np

from levisim.core.constants import kB, π

__all__ = [
    'cross_matrix',
    'cylinder_surface_quadrature',
    'diffusion_matrix',
    'ellipsoid_surface_quadrature',
    'friction_matrix',
    'gas_prefactor',
    'spheres_surface_quadrature',
    'surface_integrals',
]

# default quadrature order: the number of Gauss-Legendre nodes in the polar angle (twice as many
# equally spaced azimuthal nodes); ellipsoid integrals with aspect ratios up to ~10 converge to
# ~1e-12 relative at this order
DEFAULT_ORDER = 64


def cross_matrix(r):
    """
    The matrix r̂ of the cross product with r, r̂ a = r × a, for r of shape (..., 3)
    [units of r]
    """
    r = np.asarray(r, dtype=float)
    x, y, z = r[..., 0], r[..., 1], r[..., 2]
    zero = np.zeros_like(x)
    return np.stack([
        np.stack([zero, -z, y], axis=-1),
        np.stack([z, zero, -x], axis=-1),
        np.stack([-y, x, zero], axis=-1),
    ], axis=-2)


def gas_prefactor(gas):
    """
    n_g √(m_g kB T/2π), the mass flux scale of Eqs. (40)-(41) of Martinetz et al. (2018)
    [kg/(m²·s)]
    """
    n_g = gas.P / (kB * gas.T)
    return n_g * np.sqrt(gas.m * kB * gas.T / (2*π))


def surface_integrals(points, normals, weights):
    """
    The two geometric surface integrals of Eqs. (40)-(41) of Martinetz et al. (2018) ::

        G_n = ∮ dA (n; r×n)⊗(n; r×n),   G_d = ∮ dA [[𝟙, -r̂], [r̂, -r̂²]]

    from a quadrature of the surface: points r (N, 3) relative to the centre of mass, unit outward
    normals n (N, 3) and area weights (N,). Returns (G_n, G_d), two symmetric 6×6 matrices whose
    blocks are [m²], [m³] and [m⁴].
    """
    r = np.asarray(points, dtype=float)
    n = np.asarray(normals, dtype=float)
    w = np.asarray(weights, dtype=float)
    v = np.concatenate([n, np.cross(r, n)], axis=1)
    G_n = np.einsum('k,ki,kj->ij', w, v, v)
    R = cross_matrix(r)
    total = w.sum()
    R_sum = np.einsum('k,kij->ij', w, R)
    R2_sum = np.einsum('k,kij,kjl->il', w, R, R)
    G_d = np.block([[total * np.eye(3), -R_sum], [R_sum, -R2_sum]])
    # symmetrise away round-off (both are symmetric by construction)
    return 0.5 * (G_n + G_n.T), 0.5 * (G_d + G_d.T)


def _coefficients(accommodation, surface_temperature_ratio):
    αc, γs = float(accommodation), float(surface_temperature_ratio)
    if not 0 <= αc <= 1:
        raise ValueError(f'the accommodation coefficient αc must be in [0, 1], not {accommodation}')
    if γs < 0:
        raise ValueError(f'the surface temperature ratio γs = √(Ts/T) must be ≥ 0, not {surface_temperature_ratio}')
    c1 = 4 - 3*αc + π*αc*γs/2
    c2 = 4 - (7 - γs**2)*αc/2 + π*αc*γs/2
    c3 = (1 + γs**2)*αc/2
    return αc, c1, c2, c3


def friction_matrix(integrals, mass, inertia, gas, accommodation=1.0, surface_temperature_ratio=1.0):
    """
    The 6×6 rotranslational friction matrix Γ of Eq. (40) of Martinetz et al. (2018) in the body
    frame, from the `surface_integrals` (G_n, G_d), the mass [kg] and the body-frame inertia tensor
    [kg·m²]. Blocks: Γ[:3, :3] = Γ_cm, Γ[3:, 3:] = Γ_rot (both [2π·Hz]), and the rotranslational
    couplings Γ[:3, 3:] [1/(m·s)] and Γ[3:, :3] [m/s], which vanish for centrosymmetric shapes.
    """
    G_n, G_d = integrals
    αc, c1, _, _ = _coefficients(accommodation, surface_temperature_ratio)
    G = gas_prefactor(gas) * (c1 * G_n + αc * G_d)
    T_inv = np.zeros((6, 6))
    T_inv[:3, :3] = np.eye(3) / mass
    T_inv[3:, 3:] = np.linalg.inv(np.asarray(inertia, dtype=float))
    return G @ T_inv


def diffusion_matrix(integrals, gas, accommodation=1.0, surface_temperature_ratio=1.0):
    """
    The 6×6 rotranslational momentum diffusion matrix D of Eq. (41) of Martinetz et al. (2018) in
    the body frame, d〈(P, J)(P, J)ᵀ〉/dt = 2D; blocks [N²·s], [N²·m·s], [N²·m²·s]. For γs = 1 it
    equals kB T Γ T (their Eq. (42)).
    """
    G_n, G_d = integrals
    _, _, c2, c3 = _coefficients(accommodation, surface_temperature_ratio)
    return kB * gas.T * gas_prefactor(gas) * (c2 * G_n + c3 * G_d)


# --------------------------------------------------------------------------------------------------
# surface quadratures (cached: each shape's integrals are computed once)
# --------------------------------------------------------------------------------------------------
def _frozen(*arrays):
    for array in arrays:
        array.setflags(write=False)
    return arrays


@lru_cache(maxsize=256)
def ellipsoid_surface_quadrature(a, b, c, centre=(0.0, 0.0, 0.0), order=DEFAULT_ORDER):
    """
    Quadrature of the surface of the ellipsoid x²/a² + y²/b² + z²/c² = 1 (semi-axes a, b, c [m]
    along x, y, z, shifted by `centre`): (points, unit normals, area weights), from
    r = (a sinθ cosφ, b sinθ sinφ, c cosθ) with Gauss-Legendre nodes in θ and equally spaced φ.
    The area element is the norm of ∂r/∂θ × ∂r/∂φ = sinθ·(bc sinθ cosφ, ac sinθ sinφ, ab cosθ).
    """
    θ_nodes, θ_weights = np.polynomial.legendre.leggauss(order)
    θ = (θ_nodes + 1) * π/2
    wθ = θ_weights * π/2
    φ = np.arange(2*order) * (2*π / (2*order))
    wφ = 2*π / (2*order)
    T, P = np.meshgrid(θ, φ, indexing='ij')
    W = np.outer(wθ, np.full(2*order, wφ))
    s, c_ = np.sin(T), np.cos(T)
    points = np.stack([a*s*np.cos(P), b*s*np.sin(P), c*c_], axis=-1).reshape(-1, 3)
    area_vector = (s[..., None] * np.stack([b*c*s*np.cos(P), a*c*s*np.sin(P), a*b*c_], axis=-1)).reshape(-1, 3)
    dA = np.linalg.norm(area_vector, axis=1)
    normals = area_vector / dA[:, None]
    weights = dA * W.ravel()
    points = points + np.asarray(centre, dtype=float)
    return _frozen(points, normals, weights)


@lru_cache(maxsize=256)
def cylinder_surface_quadrature(length, radius, order=DEFAULT_ORDER):
    """
    Quadrature of the surface of the closed cylinder of `length` ℓ and `radius` R [m] about the z
    axis, centred at the origin: the side (Gauss-Legendre in z) and the two flat end caps
    (Gauss-Legendre in the radial coordinate with weight ρ). Exact for the polynomial integrands of
    `surface_integrals`. Returns (points, unit normals, area weights).
    """
    ℓ, R = float(length), float(radius)
    nodes, weights = np.polynomial.legendre.leggauss(order)
    φ = np.arange(2*order) * (2*π / (2*order))
    wφ = 2*π / (2*order)
    # side
    z = nodes * ℓ/2
    wz = weights * ℓ/2
    Z, P = np.meshgrid(z, φ, indexing='ij')
    side_points = np.stack([R*np.cos(P), R*np.sin(P), Z], axis=-1).reshape(-1, 3)
    side_normals = np.stack([np.cos(P), np.sin(P), np.zeros_like(P)], axis=-1).reshape(-1, 3)
    side_weights = (np.outer(wz, np.full(2*order, wφ)) * R).ravel()
    # caps
    ρ = (nodes + 1) * R/2
    wρ = weights * R/2 * ρ
    Q, P = np.meshgrid(ρ, φ, indexing='ij')
    cap_weights = np.outer(wρ, np.full(2*order, wφ)).ravel()
    caps = []
    for sign in (+1, -1):
        points = np.stack([Q*np.cos(P), Q*np.sin(P), np.full_like(Q, sign*ℓ/2)], axis=-1).reshape(-1, 3)
        normals = np.tile([0.0, 0.0, float(sign)], (points.shape[0], 1))
        caps.append((points, normals, cap_weights))
    points = np.concatenate([side_points, caps[0][0], caps[1][0]])
    normals = np.concatenate([side_normals, caps[0][1], caps[1][1]])
    weights = np.concatenate([side_weights, caps[0][2], caps[1][2]])
    return _frozen(points, normals, weights)


@lru_cache(maxsize=256)
def spheres_surface_quadrature(spheres, order=DEFAULT_ORDER):
    """
    Quadrature of the surfaces of several spheres, `spheres` = ((radius, (x, y, z)), ...) [m],
    concatenated: no shadowing between the spheres is accounted for (see the module docstring).
    Returns (points, unit normals, area weights).
    """
    parts = [ellipsoid_surface_quadrature(r, r, r, tuple(centre), order) for r, centre in spheres]
    return _frozen(*(np.concatenate([part[i] for part in parts]) for i in range(3)))

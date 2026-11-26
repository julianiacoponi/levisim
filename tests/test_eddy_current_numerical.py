"""
Independent numerical check of the eddy-current damping of a conducting sphere (owner decision 17)

EddyCurrentDamping uses γ = (2π/15) σ R⁵ JᵀJ, i.e. Γ = σR²JᵀJ/(10ρ), which is Eq. (4) of Xiong et al.,
Commun. Phys. 8, 80 (2025), doi:10.1038/s42005-025-02010-7. Here the induced currents are solved for
numerically instead: in the frame of a sphere moving with velocity v through the field
B(x) = B0 + J x, the field changes uniformly, Ḃ = J v. With the vector potential in a non-symmetric
gauge, Ȧ = (Ḃ_y z, Ḃ_z x, Ḃ_x y) (curl Ȧ = Ḃ), the electric field is E = -(Ȧ + ∇φ), and φ is found by
minimising the dissipation ∫ (Ȧ + ∇φ)·σ(Ȧ + ∇φ) dV over polynomials (its stationarity is ∇·j = 0
inside and j·n = 0 on the surface), on an exact Gauss quadrature of the ball. The dissipated power
and the Lorentz force ∫ j × B dV are then compared with the closed form, and for an anisotropic σ the
damping is checked to lie between the closed forms with the smallest and largest principal σ.
"""
import itertools

import numpy as np
import pytest

from levisim.physics import EddyCurrentDamping, Material, Sphere
from levisim.physics.fields.magnetic_sources import LocalFieldExpansion


def _ball_quadrature(R, n=10):
    """Points (N, 3) and weights (N,) integrating polynomials of degree < 2n - 2 over the ball exactly"""
    t, w = np.polynomial.legendre.leggauss(n)
    r, wr = R * (t + 1) / 2, R / 2 * w * (R * (t + 1) / 2)**2
    c, wc = t, w
    φ = 2 * np.pi * np.arange(2 * n) / (2 * n)
    wφ = np.full(φ.size, 2 * np.pi / φ.size)
    rr, cc, pp = np.meshgrid(r, c, φ, indexing='ij')
    W = np.einsum('i,j,k->ijk', wr, wc, wφ).ravel()
    s = np.sqrt(1 - cc**2)
    X = np.stack([rr * s * np.cos(pp), rr * s * np.sin(pp), rr * cc], axis=-1).reshape(-1, 3)
    return X, W


def _gradients_of_monomials(X, degree):
    """∇(x^a y^b z^c) for 1 ≤ a + b + c ≤ degree, shape (K, N, 3)"""
    grads = []
    for a, b, c in itertools.product(range(degree + 1), repeat=3):
        if not 1 <= a + b + c <= degree:
            continue
        x, y, z = X.T
        gx = a * x**max(a - 1, 0) * y**b * z**c if a else 0 * x
        gy = b * x**a * y**max(b - 1, 0) * z**c if b else 0 * x
        gz = c * x**a * y**b * z**max(c - 1, 0) if c else 0 * x
        grads.append(np.stack([gx, gy, gz], axis=-1))
    return np.array(grads)


def _solve(R, σ, Bdot, degree=3):
    """e = Ȧ + ∇φ minimising ∫ e·σ e dV; returns (points, weights, e)"""
    X, W = _ball_quadrature(R)
    x, y, z = X.T
    a = np.stack([Bdot[1] * z, Bdot[2] * x, Bdot[0] * y], axis=-1)
    G = _gradients_of_monomials(X / R, degree) / R          # scaled for conditioning
    σa = a @ σ
    M = np.einsum('n,kni,ij,lnj->kl', W, G, σ, G)
    b = np.einsum('n,kni,ni->k', W, G, σa)
    coefficients = np.linalg.lstsq(M, -b, rcond=1e-13)[0]
    e = a + np.einsum('k,kni->ni', coefficients, G)
    return X, W, e


def _dissipation_and_force(R, σ, J, B0, v):
    Bdot = J @ v
    X, W, e = _solve(R, σ, Bdot)
    j = -(e @ σ)                                             # j = σ E, σ symmetric
    P = float(np.einsum('n,ni,ni->', W, e @ σ, e))
    B = B0 + X @ J.T
    F = np.einsum('n,ni->i', W, np.cross(j, B))
    return P, F


def _random_gradient(rng):
    """A curl- and divergence-free field gradient: symmetric and traceless [T/m]"""
    A = rng.normal(size=(3, 3))
    J = A + A.T
    return 300.0 * (J - np.trace(J) / 3 * np.eye(3))


def test_ball_quadrature_is_exact_for_the_moments_used():
    X, W = _ball_quadrature(2.0)
    assert W.sum() == pytest.approx(4/3 * np.pi * 8, rel=1e-13)
    assert np.einsum('n,n->', W, X[:, 0]**2) == pytest.approx(4 * np.pi * 2.0**5 / 15, rel=1e-13)


def test_numerical_eddy_currents_reproduce_the_friction_tensor_of_eddy_current_damping():
    rng = np.random.default_rng(7)
    R, σ = 20e-6, 8.7e5
    J = _random_gradient(rng)
    B0 = np.array([0.1, -0.2, 0.5])
    source = LocalFieldExpansion(B0=B0, gradient=J)
    sphere = Sphere(r=R, material=Material('conductor', 9747.0, conductivity=σ))
    γ = EddyCurrentDamping(source).friction_tensor(sphere)
    γ_numerical = np.zeros((3, 3))
    for i in range(3):
        v = np.eye(3)[i]
        P, F = _dissipation_and_force(R, σ * np.eye(3), J, B0, v)
        γ_numerical[:, i] = -F
        # the dissipated power is v·γ·v and equals (2π/15) σ R⁵ |J v|²
        assert P == pytest.approx(v @ γ @ v, rel=1e-10)
    np.testing.assert_allclose(γ_numerical, γ, rtol=1e-9, atol=1e-12 * np.max(np.abs(γ)))
    # in levisim's F = -Γ p form, Γ = σR²JᵀJ/(10ρ)
    np.testing.assert_allclose(EddyCurrentDamping(source).damping_tensor(sphere),
                               γ_numerical / sphere.m, rtol=1e-9, atol=1e-12 * np.max(np.abs(γ / sphere.m)))


def test_numerical_eddy_damping_of_an_anisotropic_sphere_lies_between_the_principal_bounds():
    rng = np.random.default_rng(3)
    R = 20e-6
    σ = np.diag([2e5, 2e5, 200.0])                           # pyrolytic graphite, c axis along z
    J = _random_gradient(rng)
    for _ in range(5):
        v = rng.normal(size=3)
        P, _ = _dissipation_and_force(R, σ, J, np.zeros(3), v)
        per_σ = (2 * np.pi / 15) * R**5 * float((J @ v) @ (J @ v))
        assert 200.0 * per_σ * (1 - 1e-12) <= P <= 2e5 * per_σ * (1 + 1e-12)
        # the anisotropic sphere is not the isotropic one: currents along z are suppressed
        assert P < 0.999 * 2e5 * per_σ

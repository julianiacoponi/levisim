"""
Rigid-body rotation of a levitated particle: kinematics, torques and librational expansions

Conventions ::

    orientation   unit quaternion q = (w, x, y, z), body frame → lab frame, R = R(q)
                  (`levisim.physics.particle.quaternion_to_rotation_matrix`)
    angular       L = body-frame angular momentum, ω = I⁻¹ L (body frame), lab ω_lab = R ω
    kinematics    dq/dt = ½ q ⊗ (0, ω),   dL/dt = τ_body - ω × L   (Euler's equations)
    librations    small rotation vector θ = (θα, θβ, θγ) about the LAB axes x, y, z at a reference
                  orientation R0:  R = exp([θ]×) R0.  The librational dimensions of `MultiTrap`
                  are labelled 'α', 'β', 'γ' (rotations about lab x, y, z); levisim's tweezer
                  propagates along z and is polarised along y, so for a linearly polarised tweezer
                  'β' is the rotation about the polarisation, which the tweezer does not confine.

Torques (time-averaged over the optical cycle; complex field amplitudes E with
E(t) = Re(E e^{-iωt}), the convention in which levisim's tweezer force is -∇(-¼ Re α‖E‖²)) ::

    optical          τ = ½ Re(p* × E) - k³/(12π ε0) Im(p* × p),   p = α_eff,lab E,   α_eff,lab = R α_eff,body Rᵀ
    static electric  τ = p0 × E + (α_static,lab E) × E          (p0 = R p0,body, permanent dipole)
    magnetic         τ = μ × B + (V/μ0) (χ_lab B) × B            (μ = R μ_body; small ‖χ‖)

The first optical term is the torque of the incident field on the induced dipole (it removes the
light's angular momentum at the extinction rate); the second is the recoil of the angular momentum
that the dipole radiates (Chaumet and Rahmani, Opt. Express 17, 2224 (2009), Eq. (5)). Together they
give the alignment torque of the real anisotropic polarisability, which confines the librations of a
linearly polarised tweezer (Hoang et al., Phys. Rev. Lett. 117, 123604 (2016)), and the spin torque
of circular and elliptical light, which drives continuous rotation (Arita et al., Nat. Commun. 4,
2374 (2013); Reimann et al., Phys. Rev. Lett. 121, 033602 (2018)). For an isotropic sphere the spin
torque is the absorbed power over ω, ½ (Im α_eff - k³‖α_eff‖²/6πε0) ‖E‖²·Im(ê* × ê): a lossless sphere
is not spun up. A lossless anisotropic particle is, through the anisotropy of its scattering: in
circular light, rotating in the plane of the field, τ = k³ E0² (α_1 - α_2)²/(24π ε0) (E0 the field
amplitude, α_1, α_2 its in-plane polarisabilities), whatever its angle.

Recoil torque noise. The induced dipole interferes with the vacuum field; the torque that is linear
in the vacuum field δE is τ_i = M_il δE_l with ::

    M_il = ε_ijk (α_jl E_k) + ε_ijl p_j,     p = α E

and the two-sided torque noise is ::

    S_τ = ħ k³/(12π ε0) · Re(M M†)                                                [N²m²/Hz]

normalised so that the same construction for the force reproduces levisim's translational recoil
noise (`Tweezer.noise`, (2, 1, 7)·ħ k⁵ ‖α‖² E0²/(60π ε0)). In the principal frame of α, for
rotation about axis i (j, k the other two) it is ħ k³ (α_j - α_k)² (‖E_j‖² + ‖E_k‖²)/(12π ε0),
which is the rotational recoil heating of Rudolph et al., Phys. Rev. A 103, 043514 (2021), Eq. (31b)
(their ξ_q = S_τ,q·θ_zpf²/ħ², with the photon scattering rate γsc ε² χ² written as k³‖α‖²E0²/(12π ε0 ħ)).
"""

import numpy as np

from levisim.core.constants import ħ, ε0, π

__all__ = [
    'GENERATORS',
    'LIBRATION_AXES',
    'LIBRATION_AXIS_INDEX',
    'angular_velocity_body',
    'dipole_moment_torque',
    'finite_size_potential',
    'finite_size_torque',
    'finite_size_torque_stiffness',
    'induced_static_torque',
    'is_libration_axis',
    'magnetic_torque',
    'optical_torque',
    'quaternion_conjugate',
    'quaternion_from_rotation_vector',
    'quaternion_multiply',
    'recoil_torque_noise',
    'rotated_tensor_derivatives',
    'rotation_vector_between',
    'rotational_energy',
    'second_moment_tensor',
    'symmetric_hessian_of_projection',
]

# small rotations about the lab x, y, z axes
LIBRATION_AXES = ('α', 'β', 'γ')
LIBRATION_AXIS_INDEX = {'α': 0, 'β': 1, 'γ': 2}

# G_j: generator of rotations about the lab axis j, (G_j)_ab = -ε_jab, so exp(θ G_z) turns x
# towards y by θ, and G_j v = ê_j × v
GENERATORS = np.zeros((3, 3, 3))
for _j in range(3):
    for _a in range(3):
        for _b in range(3):
            GENERATORS[_j, _a, _b] = -np.sign((_a - _j) * (_b - _a) * (_b - _j)) if len({_j, _a, _b}) == 3 else 0.0
del _j, _a, _b


def is_libration_axis(axis):
    """Whether `axis` is one of the librational axis names 'α', 'β', 'γ'"""
    return axis in LIBRATION_AXIS_INDEX


# --------------------------------------------------------------------------------------------------
# quaternions
# --------------------------------------------------------------------------------------------------
def quaternion_multiply(p, q):
    """
    Hamilton product p ⊗ q of quaternions (w, x, y, z); broadcasts over leading axes
    [·]
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    pw, px, py, pz = p[..., 0], p[..., 1], p[..., 2], p[..., 3]
    qw, qx, qy, qz = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([
        pw*qw - px*qx - py*qy - pz*qz,
        pw*qx + px*qw + py*qz - pz*qy,
        pw*qy - px*qz + py*qw + pz*qx,
        pw*qz + px*qy - py*qx + pz*qw,
    ], axis=-1)


def quaternion_conjugate(q):
    """Conjugate (w, -x, -y, -z) of a quaternion (the inverse of a unit quaternion) [·]"""
    q = np.asarray(q, dtype=float)
    return q * np.array([1.0, -1.0, -1.0, -1.0])


def quaternion_from_rotation_vector(θ):
    """
    Unit quaternion of the rotation by ‖θ‖ about θ/‖θ‖ (the exponential map); θ in rad
    [·]
    """
    θ = np.asarray(θ, dtype=float)
    angle = np.linalg.norm(θ)
    if angle < 1e-12:
        # second-order Taylor expansion, exact to round-off for tiny angles
        q = np.concatenate([[1.0 - angle**2 / 8], 0.5 * θ])
        return q / np.linalg.norm(q)
    return np.concatenate([[np.cos(angle / 2)], np.sin(angle / 2) * θ / angle])


def rotation_vector_between(q_reference, q):
    """
    The lab-frame rotation vector θ with R(q) = exp([θ]×) R(q_reference), i.e. the small-angle
    librational coordinates (θα, θβ, θγ) of `q` about `q_reference`. The sign of q is chosen so that
    ‖θ‖ ≤ π.
    [rad]
    """
    δ = quaternion_multiply(q, quaternion_conjugate(q_reference))
    δ = np.where(δ[..., :1] < 0, -δ, δ)
    s = np.linalg.norm(δ[..., 1:], axis=-1)
    angle = 2 * np.arctan2(s, δ[..., 0])
    with np.errstate(invalid='ignore', divide='ignore'):
        factor = np.where(s > 1e-15, angle / np.where(s > 0, s, 1.0), 2.0)
    return δ[..., 1:] * factor[..., None]


# --------------------------------------------------------------------------------------------------
# rigid-body kinematics
# --------------------------------------------------------------------------------------------------
def angular_velocity_body(L_body, inertia_body):
    """ω = I⁻¹ L in the body frame [rad/s]"""
    return np.linalg.solve(inertia_body, L_body)


def rotational_energy(L_body, inertia_body):
    """Kinetic energy ½ L·I⁻¹L of the rotation [J]"""
    return 0.5 * float(L_body @ np.linalg.solve(inertia_body, L_body))


# --------------------------------------------------------------------------------------------------
# torques
# --------------------------------------------------------------------------------------------------
def optical_torque(α_lab, E, k=None):
    """
    Time-averaged torque of a light field with complex amplitude E (3,) on a dipole p = α_lab E
    (α_lab the lab-frame effective polarisability, complex 3×3) ::

        τ = ½ Re(p* × E) - k³/(12π ε0) Im(p* × p)

    The second term, the recoil of the angular momentum the dipole radiates, needs the wavenumber k
    [1/m]; with k=None only the first is returned (the torque of the incident field alone).
    [N·m]
    """
    E = np.asarray(E, dtype=complex)
    p = np.asarray(α_lab) @ E
    τ = 0.5 * np.real(np.cross(np.conj(p), E))
    if k is not None:
        τ = τ - k**3 / (12 * π * ε0) * np.imag(np.cross(np.conj(p), p))
    return τ


def dipole_moment_torque(p, E):
    """Torque p × E of a static field E on a permanent dipole p (lab frame) [N·m]"""
    return np.cross(np.asarray(p, dtype=float), np.asarray(E, dtype=float))


def induced_static_torque(α_static_lab, E):
    """Torque (α E) × E of a static field E on the dipole it induces (real α, lab frame) [N·m]"""
    E = np.asarray(E, dtype=float)
    return np.cross(np.real(np.asarray(α_static_lab)) @ E, E)


def magnetic_torque(B, magnetic_moment=None, χ_lab=None, V=None):
    """
    Torque μ × B + (V/μ0)(χ B) × B of a static magnetic field B (lab frame, T) on a permanent moment
    μ [A·m²] and on the moment V χ B/μ0 induced in a body of volume V [m³] and susceptibility tensor χ
    (lab frame; small ‖χ‖, no demagnetisation correction)
    [N·m]
    """
    from levisim.core.constants import μ0
    B = np.asarray(B, dtype=float)
    τ = np.zeros(3)
    if magnetic_moment is not None:
        τ = τ + np.cross(np.asarray(magnetic_moment, dtype=float), B)
    if χ_lab is not None and V is not None:
        τ = τ + np.cross((V / μ0) * (np.asarray(χ_lab, dtype=float) @ B), B)
    return τ


def recoil_torque_noise(α_lab, E, k):
    """
    Two-sided recoil torque noise S_τ (3×3, lab frame) of a dipole α_lab (complex 3×3, lab frame)
    in a light field of complex amplitude E (3,) and wavenumber k [1/m] (see the module docstring)

        S_τ = ħ k³/(12π ε0) · Re(M M†),   M_il = ε_ijk α_jl E_k + ε_ijl p_j,   p = α E

    Re α is used for the scattering amplitudes, as `Tweezer.noise` does.
    [N²·m²/Hz]
    """
    α = np.real(np.asarray(α_lab))
    E = np.asarray(E, dtype=complex)
    p = α @ E
    ε = -GENERATORS  # ε[i, j, k] = ε_ijk
    M = np.einsum('ijk,jl,k->il', ε, α, E) + np.einsum('ijl,j->il', ε, p)
    return ħ * k**3 / (12 * π * ε0) * np.real(M @ np.conj(M).T)


# --------------------------------------------------------------------------------------------------
# librational expansions of a rotated tensor
# --------------------------------------------------------------------------------------------------
def rotated_tensor_derivatives(T):
    """
    First and second derivatives of T(θ) = R T Rᵀ, R = exp(Σ θ_j G_j), at θ = 0

    Returns (dT, d2T) of shapes (3, 3, 3) and (3, 3, 3, 3): dT[j] = G_j T - T G_j and the symmetric
    d2T[i, j] = ½{G_i, G_j}T - (G_i T G_j + G_j T G_i) + ½T{G_i, G_j}, so that
    T(θ) = T + Σ θ_j dT[j] + ½ Σ θ_i θ_j d2T[i, j] + O(θ³).
    """
    T = np.asarray(T)
    G = GENERATORS
    dT = np.einsum('jab,bc->jac', G, T) - np.einsum('ab,jbc->jac', T, G)
    GG = np.einsum('iab,jbc->ijac', G, G)
    anti = 0.5 * (GG + GG.transpose(1, 0, 2, 3))
    d2T = (
        np.einsum('ijab,bc->ijac', anti, T)
        - np.einsum('iab,bc,jcd->ijad', G, T, G)
        - np.einsum('jab,bc,icd->ijad', G, T, G)
        + np.einsum('ab,ijbc->ijac', T, anti)
    )
    return dT, d2T


def symmetric_hessian_of_projection(T, u, v=None):
    """
    Gradient and Hessian of the projection u*·T(θ)·v (complex vectors u, v; v = u by default) with
    respect to the small lab rotation θ of T, at θ = 0

    Returns (gradient (3,), Hessian (3, 3)), complex.
    """
    u = np.asarray(u, dtype=complex)
    v = u if v is None else np.asarray(v, dtype=complex)
    dT, d2T = rotated_tensor_derivatives(T)
    gradient = np.einsum('a,jab,b->j', np.conj(u), dT, v)
    hessian = np.einsum('a,ijab,b->ij', np.conj(u), d2T, v)
    return gradient, hessian


# --------------------------------------------------------------------------------------------------
# finite-size (Rayleigh–Gans) alignment in an anisotropic trap
# --------------------------------------------------------------------------------------------------
def second_moment_tensor(inertia):
    """
    Second moment S = ∫ρ r rᵀ dV of a rigid body about its centre of mass from its inertia tensor,
    S = ½Tr(I)·𝟙 - I (in the frame of I); for an ellipsoid of semi-axes a, b, c it is
    (m/5)·diag(a², b², c²)
    [kg·m²]
    """
    I = np.asarray(inertia, dtype=float)
    return 0.5 * np.trace(I) * np.eye(3) - I


def finite_size_potential(stiffness, second_moment):
    """
    Orientation-dependent energy of an extended particle in a harmonic trap of stiffness tensor k,
    U = ½·Tr(k·⟨r rᵀ⟩) with ⟨r rᵀ⟩ = S/m the mass-weighted second moment in the lab frame
    (`second_moment` is S/m). In the generalised Rayleigh–Gans picture the potential energy density
    follows the intensity, so the particle's own extent samples the trap curvature: the longer axis
    turns towards the weaker confinement (Kamba et al., Nat. Commun. 14, 7943 (2023), Eq. (3);
    Seberson and Robicheaux, Phys. Rev. Research 2, 033437 (2020))
    [J]
    """
    return 0.5 * float(np.trace(np.asarray(stiffness, dtype=float) @ np.asarray(second_moment, dtype=float)))


def finite_size_torque(stiffness, second_moment):
    """
    Lab-frame torque τ_j = -∂U/∂θ_j of `finite_size_potential` for small lab rotations θ
    [N·m]
    """
    k = np.asarray(stiffness, dtype=float)
    S = np.asarray(second_moment, dtype=float)
    dS, _ = rotated_tensor_derivatives(S)
    return -0.5 * np.einsum('ab,jba->j', k, dS)


def finite_size_torque_stiffness(stiffness, second_moment):
    """
    Torque stiffness K_ij = ∂²U/∂θ_i∂θ_j of `finite_size_potential` at the current orientation;
    for an ellipsoid with principal axes along the lab axes and k = m·diag(Ωx², Ωy², Ωz²),
    K_x = (Ωz² - Ωy²)·(S_yy - S_zz), so that Ω_lx² = (Ωz² - Ωy²)(q² - r²)/(q² + r²) (Kamba et al.)
    [N·m/rad]
    """
    k = np.asarray(stiffness, dtype=float)
    S = np.asarray(second_moment, dtype=float)
    _, d2S = rotated_tensor_derivatives(S)
    K = 0.5 * np.einsum('ab,ijba->ij', k, d2S)
    return 0.5 * (K + K.T)

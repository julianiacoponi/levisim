"""
The optical potential of a bead in its tweezer and a cavity standing wave (coherent scattering)

One function, `optical_potential`, gives the full (un-linearised) dipole potential of one bead in
its trapping tweezer plus a static cavity field, as a function of the bead's position. Its second
derivatives at the trap centre are what `MultiTrap` uses: the tweezer stiffness (`Trap.kj`), the
mechanical couplings gxy, gxz, gyz (`Dimension.mech_mech_coupling`), the light-mechanics couplings
(`Dimension.light_mech_coupling`, through its derivative with respect to the cavity quadratures),
and the static frequency shifts of `static_cavity_hessian`. The nonlinear time-domain simulation
is meant to use the same function, so that its Jacobian at equilibrium equals the drift matrix.

Geometry and conventions (levisim's, as in `Dimension.light_mech_coupling`) ::

    tweezer: propagates along z, polarised along y, focal field E_d (`Trap.cavity_drive_field`
             for the light scattered into the cavity, `Tweezer.E0` for the trapping light)
    cavity:  axis d = (sin θ, cos θ, 0), so θ is the angle between the tweezer polarisation (y)
             and the cavity axis (θ = 90° puts the cavity along x); polarisation e_c in the x-y
             plane with |e_c·y| = sin θ; standing wave cos(k·d·r + Φ), Φ = `Bead.Φ` (Φ = π/2 is a
             node); static field 𝓔_c = 2·E0·α·cos(k·d·r + Φ)·e_c, with E0 = `Cavity.E0` the field
             per photon and α = (A0 + iP0)/2 the static intracavity amplitude (|α|² photons)

The time-averaged dipole potential of the total field is ::

    U = -(α_pol/4)·|𝓔_tw + 𝓔_c|² = U_tw + U_CS + U_cav

    U_tw  = -(Re α_eff/4)·|E_tw(r)|²                                  (tweezer trap)
    U_CS  = -Ed·sin θ·f(r)·cos(k·d·r + Φ)·[A0·cos ψ(z) + P0·sin ψ(z)]  (coherent scattering)
    U_cav = -α_pol·E0²·|α|²·cos²(k·d·r + Φ)                           (cavity dispersive term)

with ``Ed = α_pol·E_d·E0/2`` (`Trap.Ed`), ``f(r) = |E_d(r)|/E_d`` the drive tweezer's envelope
``exp(-x²/wx(z)² - y²/wy(z)²)/w(z)``, and ``ψ(z) = k·z - arctan(z/zR)`` its phase along the beam (the
Gouy phase, whose linear term gives levisim's kz = k - 1/zR). U_CS with (A0, P0) replaced by the
operators (A, P) is levisim's V_CS; expanding it to second order in the bead displacement and the
light fluctuations gives

- the light-mechanics couplings gx, gy (A) and gz (P), from ∂²U_CS/∂r_j∂(A or P)
- the mechanical couplings gxy (∝ A0) and gxz, gyz (∝ P0), from ∂²U_CS/∂r_j∂r_k
- DIAGONAL static terms ∂²U_CS/∂r_j² ∝ A0·cos Φ, which `MultiTrap` used to leave out
  (`static_cavity_hessian`)

and U_cav, the bead in the cavity's own intensity, adds a static stiffness ``∝ |α|²·cos 2Φ`` along
the cavity axis (diagonal terms and an x-y coupling). Both vanish when the static cavity field is
zero, which is the case when every bead is at a node. Further approximations, as in the rest of
levisim: the cavity mode's transverse envelope (waist ~ 61 μm) is left out unless `cavity_waist` is
given (its curvature changes the second derivatives by ~2/(k·w)² ~ 10⁻⁵), the standing wave uses
the tweezer wavenumber (the cavity's differs by ``|Δ|/ω ~ 10⁻⁹``), and the cross term uses the static
polarisability `Bead.α` (as Ed does) where the tweezer term uses Re `Trap.α_eff` (as `Trap.kj`
does); the two differ by ~5 × 10⁻⁵ for a 60 nm silica bead.
"""
import numpy as np

from levisim.core.constants import ħ

__all__ = [
    'cavity_axis',
    'cavity_drive',
    'cavity_mode_polarisability',
    'cavity_polarisation',
    'drive_polarisability',
    'hessian_to_coupling_rate',
    'librational_expansion',
    'optical_force',
    'optical_potential',
    'scattering_amplitude',
    'scattering_amplitude_gradient',
    'scattering_geometry',
    'static_cavity_force',
    'static_cavity_hessian',
]


def cavity_axis(θ):
    """
    Unit vector along the cavity axis in the tweezer frame (x, y, z), d = (sin θ, cos θ, 0)
    [·]
    """
    return np.array([np.sin(θ), np.cos(θ), 0.0])


def cavity_polarisation(θ):
    """
    Unit polarisation e_c = (-cos θ, sin θ, 0) of levisim's cavity mode, in the x-y plane and
    perpendicular to the cavity axis d (`cavity_axis`), with e_c·ŷ = sin θ
    [·]
    """
    return np.array([-np.cos(θ), np.sin(θ), 0.0])


def _static_lab_polarisability(particle):
    from levisim.physics.material import _as_tensor
    return particle.lab_tensor(_as_tensor(particle.static_polarisability_tensor()))


def _uses_scalar_drive(trap):
    from levisim.physics.tweezer import _has_scalar_polarisability
    return _has_scalar_polarisability(trap.bead) and trap.drive_tweezer.has_default_polarisation


def drive_polarisability(trap):
    """
    The polarisability α_pol that scatters the drive tweezer's light into the cavity mode, defined so
    that α_pol·sin θ = e_c·α·ê (e_c `cavity_polarisation`, ê the drive tweezer's polarisation, α the
    particle's static lab-frame tensor); it enters Ed = α_pol·E_d·E0/2 (`Trap.Ed`)

    For an isotropic sphere in levisim's y-polarised tweezer it is `Bead.α` itself (the same object,
    so every number is unchanged). For an anisotropic particle it includes the cross-polarised
    scattering of a tilted particle (e_c·α·ŷ rather than ŷ·α·ŷ). A drive whose e_c·α·ê is complex
    (elliptical light) or non-zero at sin θ = 0 cannot be written in this form: NotImplementedError.
    [Fm²]
    """
    if _uses_scalar_drive(trap):
        return trap.bead.α
    θ = trap.drive_tweezer.θ
    s = complex(cavity_polarisation(θ) @ _static_lab_polarisability(trap.bead) @ np.asarray(trap.drive_tweezer.polarisation, dtype=complex))
    if abs(np.sin(θ)) < 1e-12:
        if s == 0:
            return 0.0
        raise NotImplementedError(
            'the particle scatters into the cavity mode at θ = 0 (cross-polarised scattering, e_c·α·ê ='
            f' {s:.4g} F·m²), which levisim\'s translational couplings (∝ sin θ) cannot express'
        )
    value = s / np.sin(θ)
    if abs(value.imag) > 1e-12 * abs(value):
        raise NotImplementedError(
            'e_c·α·ê is complex (an elliptically polarised drive): the translational coherent-scattering'
            ' couplings of MultiTrap need a real scattering amplitude'
        )
    return float(value.real)


def cavity_mode_polarisability(trap):
    """
    The polarisability e_c·α·e_c of the particle for the cavity mode's own light (the cavity
    dispersive term U_cav and the cavity pull); `Bead.α` for an isotropic sphere
    [Fm²]
    """
    from levisim.physics.tweezer import _has_scalar_polarisability
    if _has_scalar_polarisability(trap.bead):
        return trap.bead.α
    e_c = cavity_polarisation(trap.drive_tweezer.θ)
    return float(np.real(e_c @ _static_lab_polarisability(trap.bead) @ e_c))


def librational_expansion(trap, cavity_E0, α):
    """
    The coherent-scattering and cavity terms of the librations (small rotations θ = (θα, θβ, θγ) about
    the lab axes at the particle's orientation) of one particle at its trap centre, from the same
    potential as `optical_potential` with the polarisability tensor rotated, α(θ) = R(θ) α Rᵀ(θ) ::

        U_CS  = -E_d·E0·Re(α_c*·F(r)·s(θ)),   s(θ) = e_c·α(θ)·ê,   F = f(r)·cos(k·d·r + Φ)·e^{iψ(z)}
        U_cav = -E0²·|α_c|²·cos²(k·d·r + Φ)·c(θ),   c(θ) = e_c·α(θ)·e_c

    (with s = α_pol·sin θ and c = α_pol for an isotropic sphere they are U_CS and U_cav of the module
    docstring). At r = 0, F = cos Φ and ∇F = -k·sin Φ·d + i·(k - 1/zR)·cos Φ·ẑ. Returns a dict:

        light       c_j such that the light-libration coupling is G_j = c_j·θ_zpf,j (complex, without
                    the tweezer phase), c_j = (E_d·E0/2ħ)·cos Φ·∂_j s: rotations couple at the
                    antinode, translations x, y at the node (Stickler et al., Phys. Rev. A 94,
                    033818 (2016); Rudolph et al., Phys. Rev. A 103, 043514 (2021), Eq. (D5)). It
                    reproduces Dimension.G of x, y, z with ∂_j s → ∇_j F·s/cos Φ (tests).
        K_θθ_CS     -E_d·E0·Re(α_c*·cos Φ·∂_i∂_j s)                     [N·m/rad²]
        K_θθ_cav    -E0²·|α_c|²·cos²Φ·∂_i∂_j c                          [N·m/rad²]
        K_rθ_CS     -E_d·E0·Re(α_c*·∇_i F·∂_j s)  (rows x, y, z)         [N/rad]
        K_rθ_cav    E0²·|α_c|²·k·sin 2Φ·d_i·∂_j c                        [N/rad]
        torque      static torque -∂U/∂θ of the cavity terms at θ = 0     [N·m]

    E_d = `Trap.cavity_drive_field`, E0 = `Cavity.E0` (field per photon), α_c the static intracavity
    amplitude (bead frame), the static polarisability tensor α as for `Trap.Ed`. The cavity mode
    polarised along z (perpendicular to levisim's cavity polarisation e_c) is not modelled, so the
    rotations it would couple (about x for the default geometry) have no light coupling here.
    """
    from levisim.physics.rotation import rotated_tensor_derivatives
    drive = trap.drive_tweezer
    θc, k, zR = drive.θ, drive.k, drive.zR
    Φ = trap.bead.Φ
    α = complex(α)
    e_c = cavity_polarisation(θc).astype(complex)
    ê = np.asarray(drive.polarisation, dtype=complex)
    d = cavity_axis(θc)
    α_lab = _static_lab_polarisability(trap.bead)
    dα, d2α = rotated_tensor_derivatives(α_lab)
    # e_c is real, so e_c·T·v needs no conjugation
    s_j = np.einsum('a,jab,b->j', e_c, dα, ê)
    s_ij = np.einsum('a,ijab,b->ij', e_c, d2α, ê)
    c_j = np.real(np.einsum('a,jab,b->j', e_c, dα, e_c))
    c_ij = np.real(np.einsum('a,ijab,b->ij', e_c, d2α, e_c))
    EdE0 = trap.cavity_drive_field * cavity_E0
    F0 = np.cos(Φ)
    grad_F = (-k * np.sin(Φ) * d).astype(complex)
    grad_F[2] += 1j * (k - 1 / zR) * np.cos(Φ)
    intensity = cavity_E0**2 * abs(α)**2
    return {
        'light': EdE0 / (2 * ħ) * F0 * s_j,
        'K_θθ_CS': -EdE0 * np.real(np.conj(α) * F0 * s_ij),
        'K_θθ_cav': -intensity * np.cos(Φ)**2 * c_ij,
        'K_rθ_CS': -EdE0 * np.real(np.conj(α) * np.outer(grad_F, s_j)),
        'K_rθ_cav': intensity * k * np.sin(2 * Φ) * np.outer(d, c_j),
        'torque': EdE0 * np.real(np.conj(α) * F0 * s_j) + intensity * np.cos(Φ)**2 * c_j,
    }


def _split_position(r):
    r = np.asarray(r, dtype=float)
    return r[..., 0], r[..., 1], r[..., 2]


def optical_potential(r, trap, cavity_E0, α, terms=('tweezer', 'coherent_scattering', 'cavity'), cavity_waist=None):
    """
    Dipole potential of one bead in its tweezer and a static cavity field, at displacement r

        - r: displacement(s) of the bead from its trap centre, shape (..., 3), in the tweezer
          frame (x, y, z) [m]
        - trap: the `Trap` (tweezer, bead with its standing-wave phase Φ, drive tweezer and field)
        - cavity_E0: the cavity's electric field per photon, `Cavity.E0` [V/m]
        - α: static intracavity amplitude α = (A0 + iP0)/2 in the frame of this bead's tweezer
          [complex √photons]. `MultiTrap` passes (cavity.A0 + 1j·cavity.P0)/2, the same α its
          couplings use (levisim's static field does not rotate with the tweezer phase φn).
        - terms: which of 'tweezer' (U_tw), 'coherent_scattering' (U_CS) and 'cavity' (U_cav) to
          include (see the module docstring)
        - cavity_waist: if given, the cavity mode's transverse Gaussian envelope
          exp(-ρ²/w²) about the cavity axis through the trap centre is included in the cavity
          field [m]; None (default) treats the mode as a plane standing wave, as the couplings do

    U is linear in α through U_CS, so U_CS(r; α=1/2) - U_CS(r; 0) and U_CS(r; α=i/2) are its
    derivatives with respect to the quadratures A and P.
    [J]
    """
    x, y, z = _split_position(r)
    tweezer = trap.tweezer
    drive = trap.drive_tweezer
    θ, k = drive.θ, drive.k
    Φ = trap.bead.Φ
    α = complex(α)

    u = k * (np.sin(θ) * x + np.cos(θ) * y) + Φ
    standing_wave = np.cos(u)
    if cavity_waist is not None:
        # distance² from the cavity axis through the trap centre: |r|² - (d·r)²
        along = np.sin(θ) * x + np.cos(θ) * y
        ρ2 = x**2 + y**2 + z**2 - along**2
        standing_wave = standing_wave * np.exp(-ρ2 / cavity_waist**2)

    U = np.zeros(np.broadcast(x, y, z).shape)
    if 'tweezer' in terms:
        U = U - (np.real(trap.α_eff) / 4) * tweezer.E_field_amplitude(x, y, z)**2
    if 'coherent_scattering' in terms:
        A0, P0 = 2 * α.real, 2 * α.imag
        envelope = drive.E_field_amplitude(x, y, z) / drive.E0
        ψ = k * z - np.arctan(z / drive.zR)
        Ed = 0.5 * drive_polarisability(trap) * trap.cavity_drive_field * cavity_E0
        U = U - Ed * np.sin(θ) * envelope * standing_wave * (A0 * np.cos(ψ) + P0 * np.sin(ψ))
    if 'cavity' in terms:
        U = U - cavity_mode_polarisability(trap) * cavity_E0**2 * abs(α)**2 * standing_wave**2
    unknown = set(terms) - {'tweezer', 'coherent_scattering', 'cavity'}
    if unknown:
        raise ValueError(f'unknown optical potential terms {sorted(unknown)}')
    return U


def static_cavity_hessian(trap, cavity_E0, α):
    """
    The static stiffness the cavity field adds that the linearised couplings leave out, at the
    trap centre: a symmetric 3 × 3 matrix K over (x, y, z), with δU = ½·Σ Kjk·rj·rk

    It is the Hessian of U_CS (its diagonal only: its off-diagonal part is already levisim's
    gxy, gxz, gyz) plus the full Hessian of U_cav (see the module docstring). With
    s, c = sin θ, cos θ, kz = k - 1/zR and the drive tweezer's waists wx0, wy0 ::

        K_CS,xx = Ed·s·A0·cos Φ·(k²s² + 2/wx0²)
        K_CS,yy = Ed·s·A0·cos Φ·(k²c² + 2/wy0²)
        K_CS,zz = Ed·s·A0·cos Φ·(kz² + 1/zR²)
        K_cav   = 2·α_pol·E0²·|α|²·k²·cos 2Φ · d dᵀ,   d = (s, c, 0)

    The envelope terms 2/wx0², 2/wy0² and 1/zR² are the curvature of the drive tweezer's focal
    amplitude (about 10⁻² of k² for a 1 μm waist). A dimension's frequency becomes
    Ω² → Ω² + Kjj/m, and Kxy adds -Kxy·x_zpf·y_zpf/ħ to gxy.
    [N/m]
    """
    drive = trap.drive_tweezer
    θ, k, zR = drive.θ, drive.k, drive.zR
    kz = k - 1 / zR
    Φ = trap.bead.Φ
    α = complex(α)
    A0 = 2 * α.real
    s, c = np.sin(θ), np.cos(θ)
    Ed = 0.5 * drive_polarisability(trap) * trap.cavity_drive_field * cavity_E0

    cs_prefactor = Ed * s * A0 * np.cos(Φ)
    K_CS = np.diag([
        cs_prefactor * (k**2 * s**2 + 2 / drive.wx0**2),
        cs_prefactor * (k**2 * c**2 + 2 / drive.wy0**2),
        cs_prefactor * (kz**2 + 1 / zR**2),
    ])
    d = cavity_axis(θ)
    K_cav = 2 * cavity_mode_polarisability(trap) * cavity_E0**2 * abs(α)**2 * k**2 * np.cos(2 * Φ) * np.outer(d, d)
    return K_CS + K_cav


def static_cavity_force(trap, cavity_E0, α):
    """
    The static force of the cavity field on the bead at the trap centre, -∇(U_CS + U_cav) at r = 0

    levisim linearises about the trap centre and does not move the bead to the displaced
    equilibrium r_eq ≈ K_tw⁻¹·F that this force implies; this function quantifies that
    approximation (e.g. kz·δz ~ 0.2 rad for one 60 nm bead at Φ = π/4 in the default cavity).
    [N]
    """
    drive = trap.drive_tweezer
    θ, k = drive.θ, drive.k
    kz = k - 1 / drive.zR
    Φ = trap.bead.Φ
    α = complex(α)
    A0, P0 = 2 * α.real, 2 * α.imag
    s = np.sin(θ)
    Ed = 0.5 * drive_polarisability(trap) * trap.cavity_drive_field * cavity_E0
    d = cavity_axis(θ)
    # -∂U_CS/∂r = Ed·s·[-k·sin Φ·A0·d + cos Φ·P0·kz·ẑ];  -∂U_cav/∂r = -α_pol·E0²·|α|²·k·sin 2Φ·d
    F_CS = Ed * s * (-k * np.sin(Φ) * A0 * d + np.array([0.0, 0.0, np.cos(Φ) * P0 * kz]))
    F_cav = -cavity_mode_polarisability(trap) * cavity_E0**2 * abs(α)**2 * k * np.sin(2 * Φ) * d
    return F_CS + F_cav


def hessian_to_coupling_rate(K_jk, zpf_j, zpf_k):
    """
    The rate g of H/ħ ∋ -g·qj·qk equivalent to a static stiffness term Kjk·rj·rk (j ≠ k)
    [2π·Hz]
    """
    return -K_jk * zpf_j * zpf_k / ħ


# --------------------------------------------------------------------------------------------------
# UN-LINEARISED FORCE AND CAVITY DRIVE (for the nonlinear CavitySimulation)
# --------------------------------------------------------------------------------------------------
def scattering_geometry(r, trap):
    """
    The pieces of U_CS and U_cav at displacement(s) r (shape (..., 3)) [m], as a dict:

        u = k·d·r + Φ          standing-wave phase [rad]
        f, grad_ln_f           drive tweezer's envelope exp(-x²/wx(z)² - y²/wy(z)²)/w(z) [·] and ∇ln f [1/m]
        ψ, dψ_dz               its phase k·z - arctan(z/zR) [rad] and ∂ψ/∂z [1/m]
        s, c, k, d             sin θ, cos θ, the wavenumber [1/m] and the cavity axis
    """
    x, y, z = _split_position(r)
    drive = trap.drive_tweezer
    θ, k, zR = drive.θ, drive.k, drive.zR
    s, c = np.sin(θ), np.cos(θ)
    u = k * (s * x + c * y) + trap.bead.Φ
    w2 = 1 + (z / zR)**2
    e = (x**2 / drive.wx0**2 + y**2 / drive.wy0**2) / w2
    f = np.exp(-e) / np.sqrt(w2)
    grad_ln_f = np.stack([
        -2 * x / (drive.wx0**2 * w2),
        -2 * y / (drive.wy0**2 * w2),
        (z / zR**2) / w2 * (2 * e - 1),
    ], axis=-1)
    ψ = k * z - np.arctan(z / zR)
    dψ_dz = k - zR / (zR**2 + z**2)
    return {'s': s, 'c': c, 'k': k, 'u': u, 'f': f, 'grad_ln_f': grad_ln_f, 'ψ': ψ, 'dψ_dz': dψ_dz,
            'd': cavity_axis(θ)}


def scattering_amplitude(r, trap):
    """
    The complex amplitude with which the bead at displacement r scatters its drive tweezer's light
    into the cavity mode, in its tweezer's frame ::

        D(r) = sin θ · f(r) · cos(k·d·r + Φ) · e^{iψ(z)}

    (f the drive tweezer's envelope, ψ = k·z - arctan(z/zR) its phase; see the module docstring),
    so that U_CS = -Ed·(α*·D + α·D*) = -2·Ed·Re(α*·D) for the bead-frame amplitude α. At the trap
    centre D = sin θ·cos Φ, the factor of levisim's static drive ωd·sin θ·cos Φ.
    r: shape (..., 3) [m]
    [·]
    """
    g = scattering_geometry(r, trap)
    return g['s'] * g['f'] * np.cos(g['u']) * np.exp(1j * g['ψ'])


def _amplitude_gradient(g):
    phase = np.exp(1j * g['ψ'])
    D = g['s'] * g['f'] * np.cos(g['u']) * phase
    grad = D[..., None] * g['grad_ln_f']
    grad = grad - (g['s'] * g['f'] * np.sin(g['u']) * phase)[..., None] * (g['k'] * g['d'])
    grad[..., 2] = grad[..., 2] + 1j * g['dψ_dz'] * D
    return grad


def scattering_amplitude_gradient(r, trap):
    """
    ∇D of `scattering_amplitude`, shape (..., 3), in closed form ::

        ∇D = D·∇ln f - sin θ·f·sin(k·d·r + Φ)·e^{iψ}·k·d + i·(∂ψ/∂z)·D·ẑ

    [1/m]
    """
    return _amplitude_gradient(scattering_geometry(r, trap))


def optical_force(r, trap, cavity_E0, α, terms=('tweezer', 'coherent_scattering', 'cavity')):
    """
    The force -∇U of `optical_potential` (plane-wave cavity mode), in closed form ::

        F_tw  = the trapping tweezer's gradient force (the arithmetic of `Tweezer.force`, `Trap.Ftw`)
        F_CS  = 2·Ed·Re(α*·∇D)                         (D: `scattering_amplitude`)
        F_cav = -α_pol·E0²·|α|²·k·sin(2(k·d·r + Φ))·d

    - r: displacement(s) from the trap centre, shape (..., 3) [m]
    - trap, cavity_E0, α (bead-frame static amplitude), terms: as `optical_potential`

    The tests check it against central differences of `optical_potential`. Returns shape (..., 3).
    [N]
    """
    r = np.asarray(r, dtype=float)
    α = complex(α)
    unknown = set(terms) - {'tweezer', 'coherent_scattering', 'cavity'}
    if unknown:
        raise ValueError(f'unknown optical force terms {sorted(unknown)}')
    F = np.zeros(r.shape)
    if 'tweezer' in terms:
        x, y, z = _split_position(r)
        tweezer = trap.tweezer
        U = -(np.real(trap.α_eff) / 2) * tweezer.E_field_amplitude(x, y, z)**2
        ellipse = tweezer.ellipse(x, y, z)
        F = F + np.stack([
            0.5 * (4 * x / tweezer.wx(z)**2) * U,
            0.5 * (4 * y / tweezer.wy(z)**2) * U,
            0.5 * (4 * z / tweezer.wz(z)**2) * U * (1 - 2 * ellipse),
        ], axis=-1)
    if 'coherent_scattering' in terms or 'cavity' in terms:
        g = scattering_geometry(r, trap)
    if 'coherent_scattering' in terms:
        Ed = 0.5 * drive_polarisability(trap) * trap.cavity_drive_field * cavity_E0
        F = F + 2 * Ed * np.real(np.conj(α) * _amplitude_gradient(g))
    if 'cavity' in terms:
        F = F - (cavity_mode_polarisability(trap) * cavity_E0**2 * abs(α)**2 * g['k'] * np.sin(2 * g['u']))[..., None] * g['d']
    return F


def cavity_drive(r, trap, cavity_E0, α, φ=0.0, terms=('coherent_scattering', 'cavity')):
    """
    The bead's contribution to dα/dt of the cavity amplitude α (lab frame) at displacement r ::

        dα/dt ∋ -(i/ħ)·∂U/∂α* = i·ωd·e^{iφ}·D(r) + i·(α_pol·E0²/ħ)·cos²(k·d·r + Φ)·α

    with ωd = Ed/ħ, φ the tweezer phase (the bead-frame amplitude is α·e^{-iφ}) and D the
    `scattering_amplitude`. The first term is the coherent-scattering drive; at the trap centre it
    is levisim's static drive i·ωd·sin θ·cos Φ (times e^{iφ}, which `MultiTrap`'s static field
    leaves out). The second is the frequency pull of the cavity by the bead (U_cav), which
    `MultiTrap` leaves out of the detuning. Drive and force come from the one energy
    H = -ħΔ|α|² + Σ U_n through dα/dt = -(i/ħ)·∂H/∂α* and dp/dt = -∇H, so they are conjugate.
    r: shape (..., 3) [m]; α [√photons]; φ [rad]
    [√photons/s]
    """
    α = complex(α)
    g = scattering_geometry(r, trap)
    drive = np.zeros(np.shape(g['u']), dtype=complex)
    if 'coherent_scattering' in terms:
        ωd = 0.5 * drive_polarisability(trap) * trap.cavity_drive_field * cavity_E0 / ħ
        D = g['s'] * g['f'] * np.cos(g['u']) * np.exp(1j * g['ψ'])
        drive = drive + 1j * ωd * np.exp(1j * φ) * D
    if 'cavity' in terms:
        δ = cavity_mode_polarisability(trap) * cavity_E0**2 / ħ
        drive = drive + 1j * δ * np.cos(g['u'])**2 * α
    return drive

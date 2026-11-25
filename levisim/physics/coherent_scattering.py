"""
The optical potential of a bead in its tweezer and a cavity standing wave (coherent scattering)

`optical_potential` gives the full (un-linearised) dipole potential of one bead in its trapping
tweezer plus the static cavity field, as a function of the bead's displacement. Its second
derivatives at the trap centre are what `MultiTrap` uses: the tweezer stiffness (`Trap.kj`), the
mechanical couplings gxy, gxz, gyz (`Dimension.mech_mech_coupling`), and, with
`MultiTrap(static_cavity_shifts=True)`, the static frequency shifts of `static_cavity_hessian`.

Geometry and conventions (levisim's, as in `Dimension.light_mech_coupling`) ::

    tweezer: propagates along z, polarised along y, focal field E_d (`Trap.cavity_drive_field`
             for the light scattered into the cavity, `Tweezer.E0` for the trapping light)
    cavity:  axis d = (sin θ, cos θ, 0), so θ is the angle between the tweezer polarisation (y)
             and the cavity axis; standing wave cos(kc·d·r + Φ), Φ = `Bead.Φ` (Φ = π/2 is a node);
             static field 2·E0·α·cos(kc·d·r + Φ), with E0 = `Cavity.E0` the field per photon and
             α = (A0 + iP0)/2 the static intracavity amplitude (|α|² photons)

The time-averaged dipole potential of the total field is ::

    U = -(α_pol/4)·|E_tw + E_c|² = U_tw + U_CS + U_cav

    U_tw  = -(Re α_eff/4)·|E_tw(r)|²                                     (tweezer trap)
    U_CS  = -Ed·sin θ·f(r)·cos(kc·d·r + Φ)·[A0·cos ψ(z) + P0·sin ψ(z)]   (coherent scattering)
    U_cav = -α_pol·E0²·|α|²·cos²(kc·d·r + Φ)                             (cavity intensity, V_c)

with ``Ed = α_pol·E_d·E0/2`` (= ħ·ωd, `Trap.Ed`), ``f(r) = |E_d(r)|/E_d`` the drive tweezer's
envelope and ``ψ(z) = k·z - arctan(z/zR)`` its phase along the beam, whose linear term gives
levisim's kz = k - 1/zR (k the TWEEZER wavenumber). Second order in the displacement gives

- the mechanical couplings gxy (∝ A0) and gxz, gyz (∝ P0): the off-diagonal part of ∂²U_CS
- DIAGONAL "cotrapping" terms ∂²U_CS/∂r_j² ∝ A0·cos Φ, which stiffen (red detuning, A0 > 0) or
  soften the trap
- the dispersive stiffness of U_cav, ``∝ |α|²·cos 2Φ``, along the cavity axis (diagonal terms and
  an x-y term)

The wavenumber kc of the standing wave is the cavity's, `Cavity.k` = (ωtw - Δ)/c. Every function
here takes it as the keyword `kc`; ``kc=None`` uses the tweezer wavenumber instead, which is what
`MultiTrap` does unless `cavity_wavenumber=True` (the two differ by abs(Δ)/ωtw ~ 6 × 10⁻¹⁰).
The cavity mode's transverse envelope (waist ~ 61 μm) is left out unless `cavity_waist` is given
(it changes the second derivatives by ~2/(k·w)² ~ 10⁻⁵), and the cross term uses the static
polarisability `Bead.α` (as Ed does) where the tweezer term uses Re `Trap.α_eff` (as `Trap.kj` does).
"""
import numpy as np

from levisim.core.constants import ħ

__all__ = [
    'cavity_axis',
    'hessian_to_coupling_rate',
    'optical_potential',
    'static_cavity_force',
    'static_cavity_hessian',
]

_TERMS = ('tweezer', 'coherent_scattering', 'cavity')


def cavity_axis(θ):
    """
    Unit vector along the cavity axis in the tweezer frame (x, y, z), d = (sin θ, cos θ, 0)
    [·]
    """
    return np.array([np.sin(θ), np.cos(θ), 0.0])


def _standing_wave_wavenumber(trap, kc):
    return trap.drive_tweezer.k if kc is None else float(kc)


def optical_potential(r, trap, cavity_E0, α, terms=_TERMS, cavity_waist=None, kc=None):
    """
    Dipole potential of one bead in its tweezer and a static cavity field, at displacement r

        - r: displacement(s) of the bead from its trap centre, shape (..., 3), in the tweezer
          frame (x, y, z) [m]
        - trap: the `Trap` (tweezer, bead with its standing-wave phase Φ, drive tweezer and field)
        - cavity_E0: the cavity's electric field per photon, `Cavity.E0` [V/m]
        - α: static intracavity amplitude α = (A0 + iP0)/2 [complex √photons]
        - terms: which of 'tweezer' (U_tw), 'coherent_scattering' (U_CS) and 'cavity' (U_cav) to
          include (see the module docstring)
        - cavity_waist: if given, the cavity mode's transverse Gaussian envelope exp(-ρ²/w²) about
          the cavity axis through the trap centre is included [m]
        - kc: wavenumber of the standing wave [m⁻¹]; None uses the drive tweezer's

    [J]
    """
    unknown = set(terms) - set(_TERMS)
    if unknown:
        raise ValueError(f'unknown optical potential terms {sorted(unknown)}')
    r = np.asarray(r, dtype=float)
    x, y, z = r[..., 0], r[..., 1], r[..., 2]
    drive = trap.drive_tweezer
    θ, k = drive.θ, drive.k
    k_sw = _standing_wave_wavenumber(trap, kc)
    Φ = trap.bead.Φ
    α = complex(α)

    along = np.sin(θ) * x + np.cos(θ) * y
    standing_wave = np.cos(k_sw * along + Φ)
    if cavity_waist is not None:
        ρ2 = x**2 + y**2 + z**2 - along**2
        standing_wave = standing_wave * np.exp(-ρ2 / cavity_waist**2)

    U = np.zeros(np.broadcast(x, y, z).shape)
    if 'tweezer' in terms:
        U = U - (np.real(trap.α_eff) / 4) * trap.tweezer.E_field_amplitude(x, y, z)**2
    if 'coherent_scattering' in terms:
        A0, P0 = 2 * α.real, 2 * α.imag
        envelope = drive.E_field_amplitude(x, y, z) / drive.E0
        ψ = k * z - np.arctan(z / drive.zR)
        Ed = 0.5 * trap.bead.α * trap.cavity_drive_field * cavity_E0
        U = U - Ed * np.sin(θ) * envelope * standing_wave * (A0 * np.cos(ψ) + P0 * np.sin(ψ))
    if 'cavity' in terms:
        U = U - trap.bead.α * cavity_E0**2 * abs(α)**2 * standing_wave**2
    return U


def static_cavity_hessian(trap, cavity_E0, α, kc=None, terms=('coherent_scattering', 'cavity')):
    """
    The static stiffness that the cavity field adds to the trap and that the linearised couplings
    leave out, at the trap centre: a symmetric 3 × 3 matrix K over (x, y, z), δU = ½·Σ Kjk·rj·rk

    It is the diagonal of the Hessian of U_CS (the "cotrapping" terms; its off-diagonal part is
    already levisim's gxy, gxz, gyz) plus the full Hessian of U_cav (the dispersive potential V_c).
    With s, c = sin θ, cos θ, kz = k - 1/zR (tweezer k), the drive tweezer's waists wx0, wy0 and
    the standing-wave wavenumber kc ::

        K_CS,xx = Ed·s·A0·cos Φ·(kc²s² + 2/wx0²)
        K_CS,yy = Ed·s·A0·cos Φ·(kc²c² + 2/wy0²)
        K_CS,zz = Ed·s·A0·cos Φ·(kz² + 1/zR²)
        K_cav   = 2·α_pol·E0²·|α|²·kc²·cos 2Φ · d dᵀ,   d = (s, c, 0)

    A dimension's frequency becomes Ω² → Ω² + Kjj/m, and Kxy adds -Kxy·x_zpf·y_zpf/ħ to gxy.

        - terms: 'coherent_scattering' (cotrapping) and/or 'cavity' (dispersive)
        - kc: wavenumber of the standing wave [m⁻¹]; None uses the drive tweezer's

    [N/m]
    """
    drive = trap.drive_tweezer
    θ, k, zR = drive.θ, drive.k, drive.zR
    k_sw = _standing_wave_wavenumber(trap, kc)
    kz = k - 1 / zR
    Φ = trap.bead.Φ
    α = complex(α)
    A0 = 2 * α.real
    s, c = np.sin(θ), np.cos(θ)
    K = np.zeros((3, 3))
    if 'coherent_scattering' in terms:
        Ed = 0.5 * trap.bead.α * trap.cavity_drive_field * cavity_E0
        prefactor = Ed * s * A0 * np.cos(Φ)
        K = K + np.diag([
            prefactor * (k_sw**2 * s**2 + 2 / drive.wx0**2),
            prefactor * (k_sw**2 * c**2 + 2 / drive.wy0**2),
            prefactor * (kz**2 + 1 / zR**2),
        ])
    if 'cavity' in terms:
        d = cavity_axis(θ)
        K = K + 2 * trap.bead.α * cavity_E0**2 * abs(α)**2 * k_sw**2 * np.cos(2 * Φ) * np.outer(d, d)
    return K


def static_cavity_force(trap, cavity_E0, α, kc=None):
    """
    The static force of the cavity field on the bead at the trap centre, -∇(U_CS + U_cav) at r = 0

    levisim linearises about the trap centre and does not move the bead to the displaced
    equilibrium r_eq ≈ K_tw⁻¹·F that this force implies; this function quantifies that
    approximation.
    [N]
    """
    drive = trap.drive_tweezer
    θ, k = drive.θ, drive.k
    k_sw = _standing_wave_wavenumber(trap, kc)
    kz = k - 1 / drive.zR
    Φ = trap.bead.Φ
    α = complex(α)
    A0, P0 = 2 * α.real, 2 * α.imag
    s = np.sin(θ)
    Ed = 0.5 * trap.bead.α * trap.cavity_drive_field * cavity_E0
    d = cavity_axis(θ)
    F_CS = Ed * s * (-k_sw * np.sin(Φ) * A0 * d + np.array([0.0, 0.0, np.cos(Φ) * P0 * kz]))
    F_cav = -trap.bead.α * cavity_E0**2 * abs(α)**2 * k_sw * np.sin(2 * Φ) * d
    return F_CS + F_cav


def hessian_to_coupling_rate(K_jk, zpf_j, zpf_k):
    """
    The rate g of H/ħ ∋ -g·qj·qk equivalent to a static stiffness term Kjk·rj·rk (j ≠ k)
    [2π·Hz]
    """
    return -K_jk * zpf_j * zpf_k / ħ

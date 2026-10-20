"""
Multi-tone coherent-scattering traps from their geometry: polarisations, cavity axis, node position

`two_tone` treats the two tones as copies of one field that differ only in amplitude and phase.
In a real tweezer each tone also has a POLARISATION, and the particle sits at some phase Φ of
the cavity standing wave. This module derives, for one or more tweezers each carrying several
tones, everything `MultiToneDrive` needs:

1. THE TRAP. Light polarised at ψ (from the trap x axis) gives the stiffness K(ψ, ψ), and two
   tones k, l interfere with K(ψ_k, ψ_l) (levisim.physics.focal_expansion, in frequency² units)

       K(ψ1, ψ2) = cos(ψ1 - ψ2)·[Ω̄²·1 + ½ΔΩ_s²·diag(1, -1)] + ½ΔΩ_p²·σ(ψ1 + ψ2)  ⊕  cos(ψ1 - ψ2)·Ω_z²
       σ(Ψ) = [[-cos Ψ, -sin Ψ], [-sin Ψ, cos Ψ]]

   ΔΩ_p² = f_pol·(Ω_⊥² - Ω_∥²) is the part of the anisotropy the polarisation causes (it turns with
   the polarisation), ΔΩ_s² = (1 - f_pol)·(Ω_⊥² - Ω_∥²) a fixed beam ellipticity. The trap is

       m⁻¹·K(t) = Σ_k |c_k|²·K(ψ_k, ψ_k) + Σ_(k<l) 2Re(c_k c_l*·K(ψ_k, ψ_l)·exp(-i(Δ_k - Δ_l)t))

   For CO-POLARISED tones the beat modulates the stiffnesses at the difference frequency. For
   ORTHOGONAL tones (ψ_l = ψ_k + 90°) cos(ψ_k - ψ_l) = 0: the beat has no diagonal part at all,
   only the x-y term -ΔΩ_p²·xy, and the static anisotropy is reduced to (c_k² - c_l²)·ΔΩ_p² + ΔΩ_s².

2. THE COUPLINGS. The cavity axis u_c lies in the focal plane at angle θ from the trap y axis
   (Deplano et al.'s θ), its mode polarisation ε_c in the focal plane perpendicular to it. Tone k
   induces a dipole along its polarisation e_k, which scatters into the mode with e_k·ε_c, so

       G_jk = -ω_d·(e_k·ε_c)·sinΦ·k·zpf_j·(u_c·ê_j)        (j along the trap's principal axes)
       G_zk = i·ω_d·(e_k·ε_c)·cosΦ·k_z·zpf_z              (couples to the other quadrature)

   ω_d is the coherent-scattering drive rate of a full-amplitude tone with the dipole parallel
   to ε_c. These are levisim's Dimension.light_mech_coupling expressions, with sinθ there
   replaced by e_k·ε_c so that each tone has its own projection.

3. OFF A NODE (Φ ≠ π/2) the carrier of every tone scatters into the cavity, building a coherent
   intracavity field α_l = i·ω_d(e_l·ε_c)c_l·cosΦ/(κ/2 - iΔ_l). Its interference with the tweezer
   adds, with Z(t) = Σ_(k,l) D_k·α_l*·exp(-i(Δ_k - Δ_l)t) and D_k = 2ω_d(e_k·ε_c)c_k,

       H/ħ ⊃ Re[Z(t)]·(½cosΦ·k²(u_c·r)² + ½cosΦ·k_z²z²) + Re[i·Z(t)]·k·k_z·sinΦ·z(u_c·r)

   whose static (k = l) part is levisim's mech-mech coupling (Dimension.mech_mech_coupling gives
   its xy entry, with A0 = 2Re α), together with frequency shifts of the same size, and whose
   k ≠ l part beats at the tones' difference frequency. Interference with the tweezer's own
   curvature is (k·w0)⁻² ≈ 3% of this and is neglected, as is the cavity field's own dipole potential.

The x and y dimensions of the returned model are the principal axes of the full static
potential, so a static x-y coupling shows up as a ROTATION of the modes (`axes_angle`) rather
than as a coupling term.
"""
import numpy as np

from dataclasses import dataclass, field

from levisim.core.constants import π, ħ
from levisim.core.units import kHz, nm, K
from levisim.multiphysics.multitrap import MultiTrap
from levisim.multiphysics.multitone import MultiToneDrive, QuadraticPotential, Tone
from levisim.multiphysics.covariance import dimension_covariance
from levisim.physics.focal_expansion import polarisation_matrix
from levisim.analysis.gaussian_states import log_negativity


__all__ = [
    'TweezerOptics',
    'CavityGeometry',
    'TweezerTone',
    'TrapModel',
    'tweezer_trap',
    'multitone_drive',
    'deplano2025_tweezer',
    'two_tone_tones',
    'subsystem_log_negativity',
    'principal_axes_log_negativity',
]


# ----------------------------------------------------------------------------------------------
# INGREDIENTS
# ----------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class TweezerOptics:
    """
    Single-tone trap of one tweezer, used to build the multi-tone potential
        - Ω_perp, Ω_par: frequencies across and along the polarisation for ONE tone carrying
          the full power, polarised along the trap y axis [2π·Hz]
        - Ω_z: axial frequency, or None to leave z out [2π·Hz]
        - f_pol: fraction of Ω_perp² - Ω_par² that the polarisation causes; 1 for a round,
          aberration-free beam, where the vector focal field is the only source [·]
    """
    Ω_perp: float
    Ω_par: float
    Ω_z: float = None
    f_pol: float = 1.0

    @property
    def mean_square(self):
        return (self.Ω_perp**2 + self.Ω_par**2) / 2

    @property
    def anisotropy(self):
        return self.Ω_perp**2 - self.Ω_par**2

    def cross_stiffness(self, ψ1, ψ2):
        """
        K(ψ1, ψ2) in frequency² units: (3×3 with z, else 2×2) [(2π·Hz)²]
        """
        Δp = self.f_pol * self.anisotropy
        Δs = (1 - self.f_pol) * self.anisotropy
        transverse = (
            np.cos(ψ1 - ψ2) * (self.mean_square * np.eye(2) + 0.5 * Δs * np.diag([1.0, -1.0]))
            + 0.5 * Δp * polarisation_matrix(ψ1 + ψ2)
        )
        if self.Ω_z is None:
            return transverse
        K = np.zeros((3, 3))
        K[:2, :2] = transverse
        K[2, 2] = np.cos(ψ1 - ψ2) * self.Ω_z**2
        return K


@dataclass(frozen=True)
class CavityGeometry:
    """
    Where the cavity is relative to one tweezer
        - θ: angle between the trap y axis (the reference polarisation) and the cavity axis [rad]
        - Φ: phase of the standing wave at the particle; π/2 is a node, where x, y couple most [rad]
    """
    θ: float
    Φ: float = π/2

    @property
    def axis(self):
        """Cavity axis u_c in trap coordinates"""
        return np.array([np.sin(self.θ), np.cos(self.θ), 0.0])

    @property
    def polarisation(self):
        """Cavity-mode polarisation ε_c (in the focal plane, perpendicular to u_c)"""
        return np.array([np.cos(self.θ), -np.sin(self.θ), 0.0])


@dataclass(frozen=True)
class TweezerTone:
    """
    One tone of one tweezer
        - Δ: detuning from the cavity resonance [2π·Hz]
        - amplitude: field amplitude, with Σ amplitude² = 1 for a tweezer's full power [·]
        - phase: optical phase [rad]
        - ψ: polarisation angle from the trap x axis (π/2: along y, the reference) [rad]
    """
    Δ: float
    amplitude: float
    phase: float = 0.0
    ψ: float = π/2
    label: str = ''

    @property
    def complex_amplitude(self):
        return self.amplitude * np.exp(1j * self.phase)

    @property
    def polarisation(self):
        return np.array([np.cos(self.ψ), np.sin(self.ψ), 0.0])


@dataclass
class TrapModel:
    """
    Everything one tweezer contributes to a MultiToneDrive (see `tweezer_trap`)
    """
    index: int
    labels: tuple
    frequencies: dict
    axes: np.ndarray
    axes_angle: float
    tones: list
    potentials: list
    coherent_amplitudes: np.ndarray = field(default=None, repr=False)

    @property
    def info(self):
        """
        Prints information on the trap model
        """
        print(self.info_str)

    @property
    def info_str(self):
        """
        A string with information on the trap model
        """
        frequencies = ', '.join(f'Ω_{label}/2π = {Ω / kHz:.3f} kHz' for label, Ω in self.frequencies.items())
        return (
            f'TrapModel(trap {self.index}: {frequencies}'
            f'\n    principal axes rotated by {np.degrees(self.axes_angle):.3f}° from the tweezer axes'
            f'\n    {len(self.tones)} tones, {len(self.potentials)} quadratic potential terms)'
        )


# ----------------------------------------------------------------------------------------------
# ONE TWEEZER
# ----------------------------------------------------------------------------------------------
def _quadrature_coefficients(K, frequencies, labels):
    """
    {(label_i, label_j): W_ij} with H/ħ = Σ W_ij·X_i X_j equal to ½m·rᵀ(K)r, K in frequency² units
    """
    coefficients = {}
    n = len(labels)
    cutoff = 1e-12 * np.abs(K).max()           # round-off, e.g. cos(π/2) for orthogonal tones
    for i in range(n):
        if abs(K[i, i]) > cutoff:
            coefficients[(labels[i], labels[i])] = K[i, i] / (4 * frequencies[i])
        for j in range(i + 1, n):
            if abs(K[i, j]) > cutoff:
                coefficients[(labels[i], labels[j])] = K[i, j] / (2 * np.sqrt(frequencies[i] * frequencies[j]))
    return coefficients


def tweezer_trap(
    optics,
    tones,
    geometry,
    ω_d,
    m,
    index=1,
    κ=None,
    λ=1064*nm,
    kz_over_k=0.936,
    coherent_field=True,
    extra_stiffness=None,
):
    """
    Trap, couplings and quadratic potentials of one tweezer carrying several tones

        - optics: TweezerOptics of this tweezer
        - tones: list of TweezerTone (amplitudes normalised to this tweezer's full power)
        - geometry: CavityGeometry of this tweezer
        - ω_d: coherent-scattering drive rate of a full-amplitude tone whose dipole is parallel
          to the cavity-mode polarisation [2π·Hz]
        - m: particle mass [kg]
        - index: trap number used in the labels (x1, y1, z1 for index 1)
        - κ: cavity linewidth, needed for the coherent intracavity field off a node [2π·Hz]
        - kz_over_k: effective axial wavenumber (k - 1/z_R)/k of the tweezer [·]
        - coherent_field: include the coherent-intracavity-field potentials off a node [bool]
        - extra_stiffness: any further static stiffness along the tweezer axes, e.g. a toy x-y
          coupling or a misaligned beam, in frequency² units (2×2, or 3×3 with z) [(2π·Hz)²]

    Returns a TrapModel whose transverse labels are the principal axes of the full static potential.
    """
    k = 2 * π / λ
    kz = kz_over_k * k
    include_z = optics.Ω_z is not None
    n_axes = 3 if include_z else 2
    u, ε = geometry.axis[:n_axes], geometry.polarisation[:n_axes]
    sinΦ, cosΦ = np.sin(geometry.Φ), np.cos(geometry.Φ)
    c = np.array([tone.complex_amplitude for tone in tones])
    Δs = np.array([tone.Δ for tone in tones])
    projections = np.array([tone.polarisation[:n_axes] @ ε for tone in tones])

    # ---- tweezer potential: static part and beats, in frequency² units, tweezer axes
    K_static = np.zeros((n_axes, n_axes))
    beats = []
    for a in range(len(tones)):
        for b in range(a, len(tones)):
            K_ab = optics.cross_stiffness(tones[a].ψ, tones[b].ψ)
            weight = c[a] * np.conjugate(c[b])
            if a == b:
                K_static += np.real(weight) * K_ab
            elif np.isclose(Δs[a], Δs[b]):
                K_static += 2 * np.real(weight) * K_ab
            else:
                beats.append((Δs[a] - Δs[b], 2 * weight * K_ab, f'beat {tones[a].label}-{tones[b].label}'))

    # ---- coherent intracavity field off a node
    coherent = np.zeros(len(tones), dtype=complex)
    coherent_static = np.zeros((n_axes, n_axes))
    coherent_terms = []
    if coherent_field and not np.isclose(cosΦ, 0):
        assert κ is not None, 'κ is needed for the coherent intracavity field off a node'
        D = 2 * ω_d * projections * c
        coherent = 1j * ω_d * projections * c * cosΦ / (κ/2 - 1j * Δs)
        for a in range(len(tones)):
            for b in range(len(tones)):
                Z = D[a] * np.conjugate(coherent[b])
                # Re[Z e^{-iωt}]·(½cosΦ k²(u·r)² + ½cosΦ kz² z²) + Re[iZ e^{-iωt}]·k kz sinΦ z(u·r), as m⁻¹·K
                S = (ħ / m) * cosΦ * k**2 * np.outer(u, u)
                if include_z:
                    S[2, 2] += (ħ / m) * cosΦ * kz**2
                    S_xz = np.zeros((n_axes, n_axes))
                    S_xz[2, :2] = S_xz[:2, 2] = (ħ / m) * k * kz * sinΦ * u[:2]
                if np.isclose(Δs[a], Δs[b]):
                    coherent_static += np.real(Z) * S
                    if include_z:
                        coherent_static += np.real(1j * Z) * S_xz
                else:
                    W = Z * S + (1j * Z * S_xz if include_z else 0)
                    coherent_terms.append((Δs[a] - Δs[b], W, f'cavity field {tones[a].label}-{tones[b].label}'))

    # ---- principal axes of the transverse static potential (z is kept separate)
    K_total = K_static + coherent_static
    if extra_stiffness is not None:
        K_total = K_total + np.asarray(extra_stiffness, dtype=float)
    eigenvalues, eigenvectors = np.linalg.eigh(K_total[:2, :2])
    order = np.argsort(eigenvalues)[::-1]          # x: the stiffer axis, as in Deplano et al.
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    for col in range(2):
        # keep each axis within 90° of the tweezer axis it continues
        if eigenvectors[col, col] < 0:
            eigenvectors[:, col] *= -1
    assert np.all(eigenvalues > 0), 'the static potential does not trap'
    E = np.eye(n_axes)
    E[:2, :2] = eigenvectors
    axes_angle = float(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    frequencies = list(np.sqrt(eigenvalues))
    labels = [f'x{index}', f'y{index}']
    if include_z:
        frequencies.append(float(np.sqrt(K_total[2, 2])))
        labels.append(f'z{index}')
    frequencies = np.array(frequencies)

    # ---- per-tone couplings along the principal axes
    zpf = np.sqrt(ħ / (2 * m * frequencies))
    model_tones = []
    for a, tone in enumerate(tones):
        couplings = {}
        for j, label in enumerate(labels):
            if label.startswith('z'):
                couplings[label] = 1j * ω_d * projections[a] * cosΦ * kz * zpf[j]
            else:
                couplings[label] = -ω_d * projections[a] * sinΦ * k * zpf[j] * (u @ E[:, j])
        model_tones.append(Tone(Δ=tone.Δ, amplitude=tone.amplitude, phase=tone.phase, label=tone.label,
                                couplings=couplings))

    # ---- quadratic potentials in quadratures along the principal axes
    potentials = []
    static_rest = E.T @ K_total @ E
    static_rest[:2, :2] = 0                        # diagonalised into the frequencies
    static_rest[2:, 2:] = 0 if include_z else static_rest[2:, 2:]
    if np.abs(static_rest).max() > 0:
        potentials.append(QuadraticPotential(0.0, _quadrature_coefficients(static_rest, frequencies, labels),
                                             label='static xz, yz'))
    for ω, W, label in beats + coherent_terms:
        W_rotated = E.T @ W @ E
        potentials.append(QuadraticPotential(ω, _quadrature_coefficients(W_rotated, frequencies, labels),
                                             label=label))

    return TrapModel(
        index=index,
        labels=tuple(labels),
        frequencies=dict(zip(labels, frequencies)),
        axes=E,
        axes_angle=axes_angle,
        tones=model_tones,
        potentials=potentials,
        coherent_amplitudes=coherent,
    )


# ----------------------------------------------------------------------------------------------
# SEVERAL TWEEZERS IN ONE CAVITY
# ----------------------------------------------------------------------------------------------
def multitone_drive(traps, κ, heating_rates, T=300*K, frame_offset=0.0, period=None):
    """
    A toy MultiTrap holding every trap's dimensions, and the MultiToneDrive of all their tones

        - traps: list of TrapModel (indices 1, 2, ...), all with or all without z
        - κ: cavity linewidth [2π·Hz]
        - heating_rates: {label: Γ_j·n_j}, the decoherence rate of each dimension [2π·Hz]
        - frame_offset, period: as for MultiToneDrive
    Returns (mtrap, drive).
    """
    axes = {len(trap.labels) for trap in traps}
    assert len(axes) == 1, 'all traps need the same number of axes'
    frequencies = {label: Ω for trap in traps for label, Ω in trap.frequencies.items()}
    n_BE = lambda Ω: 1 / (np.exp(ħ * Ω / (1.380649e-23 * T)) - 1)
    toy_params = {
        'Ω_model': 'custom',
        'toy_Ωjn_dict': frequencies,
        'use_Γ_formula': False,
        'Γjn_dict': {label: heating_rates[label] / n_BE(Ω) for label, Ω in frequencies.items()},
        # every tone carries its own couplings, so Dimension.G is a placeholder
        'gjn_dict': {label: (1e-9 if label.startswith('z') else -1e-9) for label in frequencies},
        'T': T,
    }
    mtrap = MultiTrap(
        num_traps=len(traps),
        Cartesian_axes_per_trap=axes.pop(),
        toy_params=toy_params,
        cavity_kwargs={'κ': κ, 'Δ': 0.0},
        ω_range=np.arange(-10, 10, 1) * kHz,
    )
    tones = [tone for trap in traps for tone in trap.tones]
    potentials = [potential for trap in traps for potential in trap.potentials]
    drive = MultiToneDrive(mtrap, tones, frame_offset=frame_offset, potentials=potentials, period=period)
    return mtrap, drive


# ----------------------------------------------------------------------------------------------
# DEPLANO ET AL. (2025) CALIBRATION
# ----------------------------------------------------------------------------------------------
def deplano2025_tweezer(f_pol=1.0, include_z=False, radius=50*nm, density=2200.0):
    """
    TweezerOptics, CavityGeometry and (ω_d, m) reproducing Deplano et al. (2025)'s single tone

    The published frequencies (Table 1) are taken as Ω_⊥ = Ω_X, Ω_∥ = Ω_Y (Y is the polarisation
    axis in the paper), and Ω_Z/2π = 21.4 kHz from the text. θ follows from g_X/g_Y = tanθ·(Ω_Y/Ω_X)^½,
    giving 55.2° (the paper quotes '~45°'), and ω_d from |g_X| at the node with the dipole projection
    sinθ. The mass (a 100 nm diameter silica sphere) only matters off a node. Returns
    (optics, geometry, ω_d, m).
    """
    from levisim.multiphysics.two_tone import DEPLANO2025
    p = DEPLANO2025
    m = density * 4/3 * π * radius**3
    optics = TweezerOptics(Ω_perp=p['Ωx'], Ω_par=p['Ωy'], Ω_z=(21.4*kHz if include_z else None), f_pol=f_pol)
    θ = np.arctan((p['gx'] / p['gy']) * np.sqrt(p['Ωx'] / p['Ωy']))
    geometry = CavityGeometry(θ=θ)
    k = 2 * π / (1064*nm)
    zpf_x = np.sqrt(ħ / (2 * m * p['Ωx']))
    ω_d = p['gx'] / (np.sin(θ) * np.sin(θ) * k * zpf_x)
    return optics, geometry, ω_d, m


def two_tone_tones(ν, ratio, ψ_red=π/2, ψ_blue=π/2, projections=None, offset=0.0, common_shift=0.0,
                   relative_phase=0.0, label=''):
    """
    Red and blue TweezerTones at ω_c ∓ ν(+offset) + common_shift, at fixed total power

        - ratio: the coupling ratio G_+/G_- the tones should give [·]
        - ψ_red, ψ_blue: their polarisations [rad]
        - projections: (e_red·ε_c, e_blue·ε_c), to convert the coupling ratio into an amplitude
          ratio when the polarisations project differently on the cavity mode
    """
    amplitude_ratio = ratio
    if projections is not None:
        amplitude_ratio = ratio * abs(projections[0] / projections[1])
    c_red, c_blue = 1 / np.sqrt(1 + amplitude_ratio**2), amplitude_ratio / np.sqrt(1 + amplitude_ratio**2)
    ν = ν + offset
    return [
        TweezerTone(Δ=-ν + common_shift, amplitude=c_red, phase=0.0, ψ=ψ_red, label=f'red{label}'),
        TweezerTone(Δ=+ν + common_shift, amplitude=c_blue, phase=relative_phase, ψ=ψ_blue, label=f'blue{label}'),
    ]


# ----------------------------------------------------------------------------------------------
# ANALYSIS
# ----------------------------------------------------------------------------------------------
def subsystem_log_negativity(mtrap, V, labels_A, labels_B):
    """
    Log-negativity between two groups of dimensions (e.g. ['x1', 'y1'] vs ['x2', 'y2']) [·]

    For one mode against one mode the PPT test is necessary and sufficient; for larger groups a
    positive value still proves entanglement, but zero does not prove separability.
    """
    V_AB = dimension_covariance(mtrap, V, list(labels_A) + list(labels_B))
    return log_negativity(V_AB, modes=tuple(range(len(labels_A))))


def principal_axes_log_negativity(mtrap, V, trap, rotation=None):
    """
    x-y log-negativity of one trap in a rotated pair of axes (default: back to the tweezer axes)

    The model's x, y are the principal axes of the full static potential; passing rotation=0
    evaluates them, and the default -trap.axes_angle the tweezer's own axes.
    """
    from levisim.analysis.gaussian_states import rotate_transverse_axes
    x, y = trap.labels[:2]
    V_xy = dimension_covariance(mtrap, V, [x, y])
    φ = -trap.axes_angle if rotation is None else rotation
    return log_negativity(rotate_transverse_axes(V_xy, trap.frequencies[x], trap.frequencies[y], φ))

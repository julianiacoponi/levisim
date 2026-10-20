"""
Two-tone coherent scattering: steady-state entanglement of the transverse motions of a particle

docs/two_tone_entanglement/README.md derives the scheme step by step; in brief:

1. With one tweezer tone, making a pair of phonons in x and y costs ħ(Ω_x + Ω_y), which a single
   frequency cannot supply resonantly, so the cavity mediates (almost) only phonon EXCHANGE.
   Exchange plus thermal noise never entangles.
2. A second tone, blue of the cavity by Ω̄ = (Ω_x + Ω_y)/2 while the first is red by Ω̄, supplies
   exactly that energy: blue photon → red photon + x phonon + y phonon is resonant.
3. In the frame rotating at Ω̄ (rotating-wave approximation, RWA) the cavity couples to two
   Bogoliubov modes, β_1 ∝ G_-·G_x·b_x + G_+·G_y·b_y† and β_2 ∝ G_-·G_y·b_y + G_+·G_x·b_x†, sitting at
   ±(Ω_x - Ω_y)/2. For balanced couplings (G_x = G_y) these are b_x, b_y two-mode squeezed by
   tanh(r) = G_+/G_-.
4. The cavity is a zero-temperature bath, so it cools β_1 and β_2 just as it cools x and y in
   ordinary 2D cooling. Their joint vacuum is the two-mode squeezed vacuum of x and y: E_N = 2r.
5. The non-degeneracy Ω_x ≠ Ω_y that a tweezer provides naturally is essential: for Ω_x = Ω_y,
   β_1 - β_2 decouples (a dark Bogoliubov mode) and heats without bound.
6. The price: thermal noise on x and y reaches the Bogoliubov modes amplified by cosh(2r), and
   (at fixed total power) the cooling rate drops by cosh(2r). Entanglement needs the quantum
   cooperativity C = 4G²/(κΓn) to beat that.

Everything here is a thin layer over `MultiToneDrive` (the exact, time-periodic model with all
counter-rotating terms) plus a time-independent RWA model for insight and fast scans.
"""
import numpy as np

from dataclasses import dataclass, field

from scipy.linalg import solve_continuous_lyapunov
from scipy.optimize import minimize_scalar

from levisim.core.constants import ħ, kB
from levisim.core.units import kHz, K
from levisim.multiphysics.multitrap import MultiTrap
from levisim.multiphysics.multitone import Tone, MultiToneDrive
from levisim.multiphysics.covariance import (
    quadrature_drift_matrix,
    quadrature_diffusion_matrix,
    dimension_covariance,
)
from levisim.analysis.gaussian_states import (
    local_rotation,
    log_negativity,
    mean_phonon_numbers,
    partially_transposed_symplectic_eigenvalues,
    phonon_moments,
    symplectic_occupations,
)


__all__ = [
    'DEPLANO2025',
    'deplano2025_multitrap',
    'two_tone_drive',
    'TwoToneSteadyState',
    'two_tone_steady_state',
    'two_tone_rwa_steady_state',
    'to_rotating_frame',
    'bogoliubov_coefficients',
    'bogoliubov_occupations',
    'quantum_cooperativity',
    'analytic_log_negativity',
    'analytic_optimum',
    'analytic_threshold_cooperativity',
]


# Fitted parameters of the 2D coherent-scattering experiment of Deplano, Pontin, Ranfagni, Marino
# & Marin, Nat. Commun. 16, 4215 (2025), Table 1 and text. Γ_x, Γ_y are the fitted total
# decoherence (heating) rates, i.e. phonons per second gained from all classical noise.
DEPLANO2025 = {
    'Ωx': 122.17*kHz,
    'Ωy': 109.37*kHz,
    'gx': 14.13*kHz,
    'gy': 10.37*kHz,
    'Γx_decoherence': 4.03*kHz,
    'Γy_decoherence': 3.05*kHz,
    'κ': 57*kHz,
    'Δ': -111*kHz,
    'detection_efficiency': 0.32,
    'reference': 'Deplano et al., Nat. Commun. 16, 4215 (2025), Table 1',
}


def _bose_einstein(Ω, T):
    return 1 / (np.exp(ħ * Ω / (kB * T)) - 1)


def deplano2025_multitrap(
    Γ_scale=1.0,
    coupling_scale=1.0,
    balanced=False,
    gx_over_gy=None,
    Ωx=None,
    Ωy=None,
    κ=None,
    Δ=None,
    T=300*K,
    **kwargs,
):
    """
    A 1×2 toy MultiTrap with the Deplano et al. (2025) parameters

    levisim's thermal bath is gas damping Γ_j at temperature T, so the fitted decoherence rates
    are reproduced as Γ_j = Γ_decoherence,j / n_j(T): the heating rate Γ_j·n_j is the fitted one
    and the damping itself is negligible (~1e-4 Hz).
        - Γ_scale: multiplies both decoherence rates [·]
        - coupling_scale: multiplies both couplings g_x, g_y [·]
        - balanced: replace g_x, g_y by their RMS value, so G_x = G_y [bool]
        - gx_over_gy: impose this coupling ratio at the same RMS coupling (overrides balanced) [·]
        - Ωx, Ωy, κ, Δ: override the experimental values [2π·Hz]
    Other kwargs go to MultiTrap (e.g. ω_range).
    """
    p = DEPLANO2025
    Ωx = p['Ωx'] if Ωx is None else Ωx
    Ωy = p['Ωy'] if Ωy is None else Ωy
    gx, gy = p['gx'] * coupling_scale, p['gy'] * coupling_scale
    if balanced:
        gx = gy = np.sqrt((gx**2 + gy**2) / 2)
    if gx_over_gy is not None:
        g_RMS = np.sqrt((gx**2 + gy**2) / 2)
        gy = g_RMS * np.sqrt(2 / (1 + gx_over_gy**2))
        gx = gx_over_gy * gy

    toy_params = {
        'Ω_model': 'custom',
        'toy_Ωjn_dict': {'x1': Ωx, 'y1': Ωy},
        'use_Γ_formula': False,
        'Γjn_dict': {
            'x1': Γ_scale * p['Γx_decoherence'] / _bose_einstein(Ωx, T),
            'y1': Γ_scale * p['Γy_decoherence'] / _bose_einstein(Ωy, T),
        },
        # levisim's convention is g_x, g_y ≤ 0 (Dimension asserts it); only g_x·g_y matters here
        'gjn_dict': {'x1': -gx, 'y1': -gy},
        'T': T,
    }
    cavity_kwargs = {
        'κ': p['κ'] if κ is None else κ,
        'Δ': p['Δ'] if Δ is None else Δ,
    }
    kwargs.setdefault('ω_range', np.arange(-10, 10, 1) * kHz)
    return MultiTrap(
        num_traps=1,
        Cartesian_axes_per_trap=2,
        toy_params=toy_params,
        cavity_kwargs=cavity_kwargs,
        **kwargs,
    )


def two_tone_drive(mtrap, ratio, relative_phase=0.0, offset=0.0, labels=('x1', 'y1'), fixed_total_power=True,
                   common_shift=0.0):
    """
    A red tone at ω_c - ν + s and a blue tone at ω_c + ν + s, with ν = (Ω_x + Ω_y)/2 + offset

    The scheme needs the tones' DIFFERENCE 2ν to equal Ω_x + Ω_y, so `offset` (a mismatch of
    2·offset in the difference) is critical, while `common_shift` s (both tones moved together,
    i.e. the cavity drifting relative to the laser) only moves the sidebands within the cavity line.

        - ratio: blue/red field amplitude ratio G_+/G_- (for balanced couplings, tanh r) [·]
        - relative_phase: optical phase of the blue tone relative to the red tone [rad]
        - offset: moves the tones apart (offset > 0) or together, symmetrically [2π·Hz]
        - common_shift: moves both tones by the same amount [2π·Hz]
        - fixed_total_power: if True the two amplitudes satisfy c_-² + c_+² = 1, so the total
          tweezer power (hence the trap) is unchanged and the red tone loses what the blue gains;
          if False the red tone keeps c_- = 1 and the blue tone is added on top
    Returns (red Tone, blue Tone).
    """
    ν = np.mean([mtrap.dimensions[label].Ω for label in labels]) + offset
    if fixed_total_power:
        c_red, c_blue = 1 / np.sqrt(1 + ratio**2), ratio / np.sqrt(1 + ratio**2)
    else:
        c_red, c_blue = 1.0, float(ratio)
    return (
        Tone(Δ=-ν + common_shift, amplitude=c_red, phase=0.0, label='red'),
        Tone(Δ=+ν + common_shift, amplitude=c_blue, phase=relative_phase, label='blue'),
    )


def to_rotating_frame(V_mechanical, ν, t):
    """
    Removes the free rotation at ν from mechanical covariance matrices sampled at time t

    The rotating-frame operators are b̃ = b·exp(iνt). In the RWA they are static, so this is the
    frame in which the two-mode squeezing correlation 〈b̃_x b̃_y〉 has a fixed phase.
    """
    num_modes = np.shape(V_mechanical)[0] // 2
    return local_rotation(V_mechanical, [-ν * t] * num_modes)


def bogoliubov_coefficients(mtrap, ratio, relative_phase=0.0, labels=('x1', 'y1'), fixed_total_power=True):
    """
    (u_1, v_1, u_2, v_2) with β_1 = u_1·b_x + v_1·b_y† and β_2 = u_2·b_y + v_2·b_x†

    β_1 ∝ c_-·G_x·b_x + c̃_+·G_y·b_y† is the combination the cavity sees rotating with b_x in the
    RWA, and |u|² - |v|² = 1 normalises it. If |c_+·G_y| ≥ |c_-·G_x| that combination is not an
    annihilation operator: the cavity then AMPLIFIES it and there is no steady state, which is
    returned as NaN coefficients.
    """
    red, blue = two_tone_drive(mtrap, ratio, relative_phase, labels=labels, fixed_total_power=fixed_total_power)
    c_minus, c_plus = red.complex_amplitude, blue.complex_amplitude
    Gx, Gy = (mtrap.dimensions[label].G for label in labels)
    coefficients = []
    for G_same, G_other in [(Gx, Gy), (Gy, Gx)]:
        norm_squared = np.abs(c_minus * G_same)**2 - np.abs(c_plus * G_other)**2
        if norm_squared <= 0:
            coefficients += [np.nan, np.nan]
            continue
        norm = np.sqrt(norm_squared)
        coefficients += [c_minus * G_same / norm, c_plus * G_other / norm]
    return tuple(coefficients)


def bogoliubov_occupations(V_rotating, mtrap, ratio, relative_phase=0.0, labels=('x1', 'y1'), fixed_total_power=True):
    """
    〈β_1†β_1〉, 〈β_2†β_2〉 from a ROTATING-frame x-y covariance matrix

    〈β†β〉 = |u|²〈b†b〉 + |v|²(〈b'†b'〉 + 1) + 2Re(u·v*·〈b b'〉). Both are zero exactly when the
    state is the two-mode squeezed vacuum the scheme aims for.
    """
    u1, v1, u2, v2 = bogoliubov_coefficients(mtrap, ratio, relative_phase, labels, fixed_total_power)
    N, M = phonon_moments(V_rotating)
    nx, ny, mxy = np.real(N[0, 0]), np.real(N[1, 1]), M[0, 1]
    β1 = np.abs(u1)**2 * nx + np.abs(v1)**2 * (ny + 1) + 2 * np.real(u1 * np.conjugate(v1) * mxy)
    β2 = np.abs(u2)**2 * ny + np.abs(v2)**2 * (nx + 1) + 2 * np.real(u2 * np.conjugate(v2) * mxy)
    return float(β1), float(β2)


@dataclass
class TwoToneSteadyState:
    """
    The periodic steady state of a two-tone drive, restricted to two mechanical dimensions

    `log_negativities` are per sampled time. In the RWA E_N is constant over the period (the
    lab-frame state differs from the rotating-frame one by local rotations only); any variation
    comes from counter-rotating terms and from the trap modulation.
    """
    ratio: float
    labels: tuple
    ν: float
    stable: bool
    floquet_multiplier: float
    times: np.ndarray = field(default=None, repr=False)
    covariances: list = field(default=None, repr=False)
    mechanical_covariances: list = field(default=None, repr=False)
    rotating_covariances: list = field(default=None, repr=False)
    log_negativities: np.ndarray = field(default=None, repr=False)
    ν_tildes: np.ndarray = field(default=None, repr=False)
    mean_phonons: np.ndarray = field(default=None, repr=False)
    thermal_occupations: np.ndarray = field(default=None, repr=False)
    bogoliubov: tuple = field(default=None, repr=False)

    @property
    def E_N(self):
        """Period-averaged logarithmic negativity (0 if unstable)"""
        return float(np.mean(self.log_negativities)) if self.stable else 0.0

    @property
    def E_N_min(self):
        return float(np.min(self.log_negativities)) if self.stable else 0.0

    @property
    def E_N_max(self):
        return float(np.max(self.log_negativities)) if self.stable else 0.0

    @property
    def summary_str(self):
        if not self.stable:
            return f'G+/G- = {self.ratio:.3f}: UNSTABLE (|Floquet multiplier| = {self.floquet_multiplier:.4f})'
        return (
            f'G+/G- = {self.ratio:.3f}: E_N = {self.E_N:.4f} (min {self.E_N_min:.4f}, max {self.E_N_max:.4f}),'
            f' n = {np.array2string(self.mean_phonons, precision=3)},'
            f' 〈β†β〉 = ({self.bogoliubov[0]:.3f}, {self.bogoliubov[1]:.3f}),'
            f' |Floquet multiplier| = {self.floquet_multiplier:.4f}'
        )


def two_tone_steady_state(
    mtrap,
    ratio,
    labels=('x1', 'y1'),
    relative_phase=0.0,
    offset=0.0,
    beat_visibility=0.0,
    fixed_total_power=True,
    num_samples=16,
    common_shift=0.0,
):
    """
    Exact (linearised, all counter-rotating terms kept) periodic steady state of the two-tone drive
    """
    tones = two_tone_drive(mtrap, ratio, relative_phase, offset, labels, fixed_total_power, common_shift)
    # light in the frame rotating at ω_c + common_shift, where the drive has period π/ν
    drive = MultiToneDrive(mtrap, tones, beat_visibility=beat_visibility, frame_offset=common_shift)
    ν = (tones[1].Δ - tones[0].Δ) / 2
    steady_state = drive.periodic_steady_state(num_samples)
    multiplier = steady_state.largest_multiplier
    if not steady_state.stable:
        return TwoToneSteadyState(float(ratio), tuple(labels), float(ν), False, multiplier)

    times, Vs = steady_state.times, steady_state.covariances
    mechanical = [dimension_covariance(mtrap, V, labels) for V in Vs]
    rotating = [to_rotating_frame(Vm, ν, t) for Vm, t in zip(mechanical, times)]
    log_negativities = np.array([log_negativity(Vm) for Vm in mechanical])
    ν_tildes = np.array([partially_transposed_symplectic_eigenvalues(Vm).min() for Vm in mechanical])
    V_rotating_mean = np.mean(rotating, axis=0)
    return TwoToneSteadyState(
        ratio=float(ratio),
        labels=tuple(labels),
        ν=float(ν),
        stable=True,
        floquet_multiplier=multiplier,
        times=times,
        covariances=Vs,
        mechanical_covariances=mechanical,
        rotating_covariances=rotating,
        log_negativities=log_negativities,
        ν_tildes=ν_tildes,
        mean_phonons=np.mean([mean_phonon_numbers(Vm) for Vm in mechanical], axis=0),
        thermal_occupations=np.mean([symplectic_occupations(Vm) for Vm in mechanical], axis=0),
        bogoliubov=bogoliubov_occupations(V_rotating_mean, mtrap, ratio, relative_phase, labels, fixed_total_power),
    )


def two_tone_rwa_steady_state(
    mtrap,
    ratio,
    labels=('x1', 'y1'),
    relative_phase=0.0,
    offset=0.0,
    beat_visibility=0.0,
    fixed_total_power=True,
    common_shift=0.0,
):
    """
    Steady state of the rotating-wave model, in the frame rotating at ν for every dimension

    H/ħ = -s·a†a + Σ_j (Ω_j - ν)·b̃_j†b̃_j - [a†·Σ_j G_j(c̃_-·b̃_j + c̃_+·b̃_j†) + h.c.]
          + Σ_j (Ω_j·v/4P)(c̃_-·c̃_+*·b̃_j² + h.c.)            (trap beat, visibility v)
          - Σ_pairs g_jk (b̃_j†b̃_k + h.c.)                    (exchange part of mech-mech terms)

    (s = common_shift; the light is in the frame rotating at ω_c + s.)
    It is time independent, so one Lyapunov solve gives the state. Comparing it with
    `two_tone_steady_state` isolates what the counter-rotating terms do.
    Returns (stable, full rotating-frame covariance matrix or None, mechanical covariance or None).
    """
    red, blue = two_tone_drive(mtrap, ratio, relative_phase, offset, labels, fixed_total_power, common_shift)
    c_minus, c_plus = red.complex_amplitude, blue.complex_amplitude
    power = np.abs(c_minus)**2 + np.abs(c_plus)**2
    ν = (blue.Δ - red.Δ) / 2
    n = mtrap.num_operators
    A = np.zeros((n, n), dtype=np.complex128)
    # light in the frame rotating at ω_c + common_shift, where the cavity is detuned by -common_shift
    A[0, 0] = -mtrap.cavity.κ / 2 + 1j * common_shift
    A[1, 1] = -mtrap.cavity.κ / 2 - 1j * common_shift

    for ii in range(2, n, 2):
        label = mtrap.noise_index_to_label[ii]
        dimension = mtrap.dimensions[label]
        G, G_star = dimension.G, np.conjugate(dimension.G)
        # cavity rows: da/dt ⊃ i·G(c̃_-·b̃ + c̃_+·b̃†), da†/dt its conjugate
        A[0, ii], A[0, ii + 1] = 1j * c_minus * G, 1j * c_plus * G
        A[1, ii + 1], A[1, ii] = -1j * np.conjugate(c_minus) * G_star, -1j * np.conjugate(c_plus) * G_star
        # mechanical rows: db̃/dt ⊃ i(c̃_-*·G*·a + c̃_+·G·a†), db̃†/dt its conjugate
        A[ii, 0], A[ii, 1] = 1j * np.conjugate(c_minus) * G_star, 1j * c_plus * G
        A[ii + 1, 1], A[ii + 1, 0] = -1j * c_minus * G, -1j * np.conjugate(c_plus) * G_star
        A[ii, ii] = -1j * (dimension.Ω - ν) - dimension.Γ / 2
        A[ii + 1, ii + 1] = 1j * (dimension.Ω - ν) - dimension.Γ / 2
        if beat_visibility:
            squeeze = dimension.Ω * beat_visibility / (2 * power)
            A[ii, ii + 1] = -1j * squeeze * np.conjugate(c_minus) * c_plus
            A[ii + 1, ii] = 1j * squeeze * c_minus * np.conjugate(c_plus)

    # exchange (number-conserving) part of any intra-trap mech-mech coupling, -g_jk X_j X_k
    for ii in range(2, n, 2):
        for kk in range(2, n, 2):
            if ii == kk:
                continue
            g_jk = mtrap.drift_matrix[ii, kk].imag  # levisim stores i·g_jk at (b_j, b_k)
            A[ii, kk] = 1j * g_jk
            A[ii + 1, kk + 1] = -1j * g_jk

    A_quadratures = quadrature_drift_matrix(A)
    if np.max(np.linalg.eigvals(A_quadratures).real) >= 0:
        return False, None, None
    V = solve_continuous_lyapunov(A_quadratures, -quadrature_diffusion_matrix(mtrap))
    return True, V, dimension_covariance(mtrap, V, labels)


# ----------------------------------------------------------------------------------------------
# ANALYTIC ESTIMATE
# ----------------------------------------------------------------------------------------------
def quantum_cooperativity(mtrap, labels=('x1', 'y1'), include_lorentzian=True):
    """
    C = 4〈|G|²〉/(κ·〈Γn〉) for a pair of dimensions, optionally times L = 1/(1 + ((Ω_x - Ω_y)/κ)²)

    4|G|²/κ is the resolved-sideband cooling rate a single red tone of the full power gives one
    dimension at Δ = -Ω, and Γ·n its thermal heating rate, so C is that dimension's quantum
    cooperativity; the pair's couplings and heating rates are averaged. In the two-tone scheme
    the Bogoliubov modes sit at ±(Ω_x - Ω_y)/2 from the cavity resonance, which the Lorentzian
    factor accounts for.
    [·]
    """
    dimensions = [mtrap.dimensions[label] for label in labels]
    mean_G_squared = np.mean([np.abs(dim.G)**2 for dim in dimensions])
    mean_heating_rate = np.mean([dim.Γ * dim.noise_phonons_exact for dim in dimensions])
    C = 4 * mean_G_squared / (mtrap.cavity.κ * mean_heating_rate)
    if include_lorentzian:
        Ωs = [dim.Ω for dim in dimensions]
        C /= 1 + ((max(Ωs) - min(Ωs)) / mtrap.cavity.κ)**2
    return float(C)


def _added_noise(r, cooperativity, fixed_total_power):
    """n_add = Γn/Γ_c(r): fixed total power gives Γ_c ∝ 1/cosh(2r), a fixed red tone 1/cosh²(r)"""
    return (np.cosh(2*r) if fixed_total_power else np.cosh(r)**2) / cooperativity


def analytic_log_negativity(r, cooperativity, fixed_total_power=True):
    """
    E_N ≈ max(0, -ln(exp(-2r) + 2·n_add)) with n_add = cosh(2r)/C (fixed total power)

    Ideal limit of the scheme: balanced couplings, the RWA, equal heating Γn on both modes, and
    Bogoliubov modes spectrally resolved from each other but inside the cavity line. Then the
    cavity damps x and y at Γ_c(r) = 4G²(c_-² - c_+²)/κ = C·Γn/cosh(2r) (at fixed total power)
    TOWARDS the two-mode squeezed vacuum, i.e. each sees a two-mode squeezed reservoir, while the
    thermal bath adds n_add = Γn/Γ_c phonons to each. That gives n_x = n_y = sinh²(r) + n_add,
    |〈b_x b_y〉| = sinh(r)cosh(r), and ν̃_- = exp(-2r) + 2n_add.

    NOTE: the Bogoliubov modes then hold n_β = n_add·cosh(2r) each, but they are CORRELATED (the
    image of uncorrelated x and y noise), so ν̃_- ≠ (2n_β + 1)exp(-2r); treating them as
    independent thermal modes overstates the entanglement.
    [·]
    """
    ν_tilde = np.exp(-2*r) + 2 * _added_noise(r, cooperativity, fixed_total_power)
    return float(max(0.0, -np.log(ν_tilde)))


def analytic_optimum(cooperativity, fixed_total_power=True):
    """
    (r*, E_N*) maximising analytic_log_negativity over the squeezing parameter r

    Closed forms, found numerically here so that the noise model can be changed:
        fixed total power:  exp(4r*) = 1 + C,   E_N* = ln[C / (2·sqrt(1 + C))]
        fixed red tone:     exp(4r*) = 1 + 2C,  E_N* = ln[C / (1 + sqrt(1 + 2C))]
    so E_N* ≈ ½·ln(C/4) for large C: each factor of e² ≈ 7.4 in C buys one nat.
    """
    objective = lambda r: np.exp(-2*r) + 2 * _added_noise(r, cooperativity, fixed_total_power)
    result = minimize_scalar(objective, bounds=(0, 4), method='bounded')
    return float(result.x), float(max(0.0, -np.log(result.fun)))


def analytic_threshold_cooperativity(fixed_total_power=True):
    """
    Smallest C for which the ideal scheme can entangle: min_r 2·n_add(r)·C/(1 - exp(-2r))

    C = 2 + 2·sqrt(2) ≈ 4.83 (at r = ln(1 + C)/4 ≈ 0.44) at fixed total power, and exactly 4 (at
    r = ln(9)/4 ≈ 0.55) if the blue tone is added on top of a fixed red tone.
    [·]
    """
    factor = (lambda r: np.cosh(2*r)) if fixed_total_power else (lambda r: np.cosh(r)**2)
    result = minimize_scalar(lambda r: 2 * factor(r) / -np.expm1(-2*r), bounds=(1e-3, 3), method='bounded')
    return float(result.fun)

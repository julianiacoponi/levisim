"""
Coherent scattering driven by several tweezer tones: a time-periodic Linear Amplifier Model

With one tweezer tone, levisim works in the frame rotating at the tweezer frequency, where the
linearised Hamiltonian is time independent:

    H/ħ = -Δ·a†a + Σ_j Ω_j·b_j†b_j - Σ_j (G_j·a† + G_j*·a)(b_j + b_j†)   [+ mech-mech terms]

With several tones k at ω_k = ω_c + Δ_k (Δ_k uses levisim's sign convention, Δ = ω_L - ω_c),
there is no frame in which all tones are static. In the frame rotating at the CAVITY resonance
each tone keeps its own time dependence, and the coupling of dimension j becomes

    G_j → F_j(t) = G_j · Σ_k c_k·exp(iϑ_k)·exp(-iΔ_k·t)

where c_k is the field amplitude of tone k relative to the tone that defines `Dimension.G` and
ϑ_k its optical phase. One tone with c = 1 at Δ_1 = cavity.Δ is exactly levisim's model, moved
to the cavity frame (the tests check this). Because every tone scatters off the same particle at
the same place in the standing wave, all tones share the per-dimension couplings G_j; only the
overall amplitude and phase differ. (The light can equally be described in a frame rotating at
ω_c + frame_offset, which replaces Δ_k by Δ_k - frame_offset in F_j and detunes the cavity by
-frame_offset; this changes no mechanical moment, only how short the drive period is.)

THE DRIVE IS ALSO THE TRAP. Tones that share a polarisation and spatial mode interfere in the
tweezer intensity, so the trap stiffness is modulated at every difference frequency Δ_k - Δ_l:

    Ω_j² → Ω_j²·(1 + ε(t)),   ε(t) = beat_visibility · 2 Σ_k<l Re(c̃_k c̃_l* exp(-i(Δ_k - Δ_l)t)) / Σ_k |c_k|²

beat_visibility = 1 is the co-polarised case and 0 a perfectly suppressed beat. This term has
no analogue in the microwave and membrane experiments the two-tone schemes come from.

MORE GENERAL GEOMETRIES. Two extensions cover tones that differ in more than amplitude and phase:
  - Tone.couplings gives a tone its own coupling G_jk to each dimension, replacing Dimension.G
    (dimensions not listed are not coupled to that tone), e.g. a second tweezer that only
    reaches its own particle, or a tone whose polarisation projects differently on the cavity
    mode;
  - QuadraticPotential terms add H/ħ ⊃ Σ Re(W_ij·exp(-iω_m t))·X_i X_j for any pairs of
    dimensions, e.g. the x-y coupling that orthogonally polarised tones beat at, or the static
    and beating potentials of the coherent intracavity field off a node
    (levisim.multiphysics.two_tone_traps builds both from the focal-field geometry).

The drift matrix is then periodic, A(t + T) = A(t), and the asymptotic state is periodic too. It
is found exactly from one period of evolution (the monodromy, i.e. a Floquet treatment of the
Lyapunov equation): with Φ = 𝒯exp∫A and W the noise accumulated over one period,

    V(0) = Φ·V(0)·Φᵀ + W

which is a discrete Lyapunov equation. NOTE: the trap frequencies Ω_j are taken from the
MultiTrap as given. Keep Σ_k c_k² = 1 (total tweezer power fixed) for this to be consistent.
"""
import math
import itertools

import numpy as np

from dataclasses import dataclass, field
from fractions import Fraction

from scipy.integrate import solve_ivp
from scipy.linalg import solve_discrete_lyapunov

from levisim.core.units import kHz, μs
from levisim.multiphysics.covariance import (
    complex_to_quadrature_transform,
    mode_index,
    quadrature_drift_matrix,
    quadrature_diffusion_matrix,
)


__all__ = [
    'Tone',
    'QuadraticPotential',
    'MultiToneDrive',
    'PeriodicSteadyState',
    'common_period',
]


@dataclass(frozen=True)
class Tone:
    """
    One frequency component of the tweezer light
        - Δ: detuning from the cavity resonance, ω_tone - ω_c [2π·Hz] (red detuned is Δ < 0)
        - amplitude: field amplitude relative to the tone that defines Dimension.G [·]
        - phase: optical phase relative to that tone [rad]
        - couplings: optional {label: G_jk}, this tone's own full-amplitude coupling to each
          dimension, used instead of Dimension.G; unlisted dimensions are not coupled [2π·Hz]
    """
    Δ: float
    amplitude: float = 1.0
    phase: float = 0.0
    label: str = ''
    couplings: dict = field(default=None, compare=False)

    @property
    def complex_amplitude(self):
        return self.amplitude * np.exp(1j * self.phase)

    def coupling(self, dimension):
        """Full-amplitude coupling of this tone to a Dimension [2π·Hz]"""
        if self.couplings is None:
            return dimension.G
        return complex(self.couplings.get(dimension.label, 0.0))


@dataclass(frozen=True)
class QuadraticPotential:
    """
    A quadratic mechanical term H/ħ = Σ_(i,j) Re(W_ij·exp(-iω·t))·X_i X_j
        - frequency: ω, 0 for a static term [2π·Hz]
        - coefficients: {(label_i, label_j): W_ij}; each entry is one product X_i X_j, so
          (x1, y1) and (y1, x1) would be counted twice [2π·Hz]
    """
    frequency: float
    coefficients: dict = field(compare=False)
    label: str = ''


def common_period(frequencies, max_denominator=1000, tolerance=1e-9):
    """
    Smallest T > 0 for which ω·T is a multiple of 2π for every angular frequency given

    Frequencies are expressed as rational multiples of the largest one; raises if they are not
    commensurate to within the tolerance, in which case pass an explicit period instead.
    [s]
    """
    magnitudes = [abs(f) for f in frequencies if abs(f) > 0]
    if not magnitudes:
        raise ValueError('need at least one non-zero frequency to define a period')
    reference = max(magnitudes)
    ratios = [Fraction(f / reference).limit_denominator(max_denominator) for f in magnitudes]
    for f, ratio in zip(magnitudes, ratios):
        if abs(float(ratio) - f / reference) > tolerance:
            raise ValueError(f'frequencies are not commensurate (ratio {f / reference}); give an explicit period')
    common_denominator = math.lcm(*[ratio.denominator for ratio in ratios])
    integers = [ratio.numerator * common_denominator // ratio.denominator for ratio in ratios]
    fundamental = reference / common_denominator * math.gcd(*integers)
    return 2 * np.pi / fundamental


class MultiToneDrive:
    """
    The Linear Amplifier Model of a MultiTrap driven by several coherently scattering tones
    """
    def __init__(self, mtrap, tones, beat_visibility=0.0, period=None, frame_offset=0.0, potentials=None):
        """
            - tones: the Tones; their Δ are always measured from the cavity resonance [2π·Hz]
            - beat_visibility: fraction of the tones' intensity beat that modulates the trap [·]
            - period: drive period, if not the common period of the tone frequencies [s]
            - frame_offset: the light is described in the frame rotating at ω_c + frame_offset.
              This is only a choice of frame (it rotates the cavity quadratures and leaves every
              mechanical moment unchanged), but it can make the drive periodic with a much shorter
              period, e.g. for tones at ω_c ∓ ν + s use frame_offset = s [2π·Hz]
            - potentials: QuadraticPotential terms added to the mechanical Hamiltonian
        """
        self.mtrap = mtrap
        self.tones = list(tones)
        self.beat_visibility = float(beat_visibility)
        self.frame_offset = float(frame_offset)

        self.Δs = np.array([tone.Δ for tone in self.tones], dtype=float)
        self.complex_amplitudes = np.array([tone.complex_amplitude for tone in self.tones])
        self.total_power_fraction = float(np.sum(np.abs(self.complex_amplitudes)**2))

        # pairs of tones beat in the trap intensity at their difference frequency
        self.beat_pairs = list(itertools.combinations(range(len(self.tones)), 2))
        self.potentials = list(potentials or [])
        frequencies = list(self.Δs - self.frame_offset)
        if self.beat_visibility:
            frequencies += [self.Δs[k] - self.Δs[l] for k, l in self.beat_pairs]
        frequencies += [potential.frequency for potential in self.potentials if potential.frequency]
        self.period = float(period) if period is not None else common_period(frequencies)

        self.setup_matrices()

    def __repr__(self):
        return f'MultiToneDrive({self.mtrap}, {len(self.tones)} tones)'

    @property
    def info(self):
        """
        Prints information on the drive
        """
        print(self.info_str)

    @property
    def info_str(self):
        """
        A string with information on the drive
        """
        tone_lines = ''.join(
            f'\n    {tone.label or f"tone {k}"}: Δ/2π = {tone.Δ / kHz:.3f} kHz,'
            f' amplitude = {tone.amplitude:.5f}, phase = {tone.phase:.5f} rad'
            for k, tone in enumerate(self.tones)
        )
        return (
            f'MultiToneDrive('
            f'{tone_lines}'
            f'\n    Σ amplitude² = {self.total_power_fraction:.5f}'
            f'\n    beat_visibility = {self.beat_visibility:.5f}'
            f'\n    frame_offset/2π = {self.frame_offset / kHz:.3f} kHz'
            f'\n    tones with their own couplings = {sum(tone.couplings is not None for tone in self.tones)}'
            f'\n    quadratic potential terms = {len(self.potentials)}'
            f'\n    period = {self.period / μs:.5f} μs'
            f'\n)'
        )

    # ------------------------------------------------------------------------------------------
    # MATRICES
    # ------------------------------------------------------------------------------------------
    def setup_matrices(self):
        """
        Splits the drift matrix as A(t) = A_static + 2Re(Σ_k c̃_k·exp(-iΔ_k t)·K_k) + ε(t)·B + P(t)

        K_k carries tone k's couplings G_jk (Dimension.G unless the tone has its own), so that
        Σ_k c̃_k·exp(-iΔ_k t)·G_jk multiplies every coupling; the static part is levisim's drift
        matrix with the light-mech couplings removed and the cavity at zero detuning. P(t) holds
        the QuadraticPotential terms.
        """
        mtrap = self.mtrap
        n = mtrap.num_operators
        A_static = np.array(mtrap.drift_matrix, dtype=np.complex128)
        B = np.zeros((n, n), dtype=np.complex128)

        # in the cavity frame the cavity no longer rotates at the (single) detuning; in the frame
        # rotating at ω_c + frame_offset it rotates at -frame_offset (levisim's -Δ·a†a with Δ → frame_offset)
        A_static[0, 0] = -mtrap.cavity.κ / 2 + 1j * self.frame_offset
        A_static[1, 1] = -mtrap.cavity.κ / 2 - 1j * self.frame_offset
        A_static[0, 2:] = A_static[1, 2:] = 0
        A_static[2:, 0] = A_static[2:, 1] = 0

        for ii in range(2, n, 2):
            dimension = mtrap.dimensions[mtrap.noise_index_to_label[ii]]
            # trap modulation (Ω_j ε/4)·X_j² gives db_j/dt += -i(Ω_j ε/2)(b_j + b_j†)
            B[ii, ii] = B[ii, ii + 1] = -0.5j * dimension.Ω
            B[ii + 1, ii] = B[ii + 1, ii + 1] = 0.5j * dimension.Ω

        T = complex_to_quadrature_transform(n)
        T_inverse = np.linalg.inv(T)
        self.A_static = quadrature_drift_matrix(A_static)
        self.Ks = [self._coupling_matrix(tone, T, T_inverse) for tone in self.tones]
        # tones sharing Dimension.G share one coupling matrix, as in the original formulation
        self.K = self.Ks[0]
        self.B = quadrature_drift_matrix(B)
        self.D = quadrature_diffusion_matrix(mtrap)
        self.potential_matrices = self._potential_matrices()

    def _coupling_matrix(self, tone, T, T_inverse):
        """
        Quadrature-basis matrix K_k of tone k, with A ⊃ 2Re(c̃_k·exp(-iΔ_k t)·K_k)
        """
        mtrap = self.mtrap
        n = mtrap.num_operators
        K = np.zeros((n, n), dtype=np.complex128)
        K_bar = np.zeros((n, n), dtype=np.complex128)
        for ii in range(2, n):
            dimension = mtrap.dimensions[mtrap.noise_index_to_label[ii]]
            G = tone.coupling(dimension)
            G_star = np.conjugate(G)
            # the same pattern as setup_linear_amplifier_model_matrices, with G → G·f(t)
            K[0, ii] = 1j * G
            K_bar[1, ii] = -1j * G_star
            if ii % 2 == 0:
                # annihilation operator b_j
                K_bar[ii, 0] = 1j * G_star
                K[ii, 1] = 1j * G
            else:
                # creation operator b_j†
                K_bar[ii, 0] = -1j * G_star
                K[ii, 1] = -1j * G
        K, K_bar = T @ K @ T_inverse, T @ K_bar @ T_inverse
        assert np.allclose(K_bar, np.conjugate(K)), 'coupling split is inconsistent'
        return K

    def _potential_matrices(self):
        """
        {ω: M_ω} such that the QuadraticPotential terms add Re(M_ω·exp(-iωt)) to the drift matrix

        H/ħ = h·X_i X_j gives dP_i/dt = -2h·X_j and dP_j/dt = -2h·X_i (i ≠ j), or dP_i/dt = -4h·X_i.
        """
        n = self.mtrap.num_operators
        matrices = {}
        for potential in self.potentials:
            M = matrices.setdefault(float(potential.frequency), np.zeros((n, n), dtype=np.complex128))
            for (label_i, label_j), W in potential.coefficients.items():
                i, j = 2 * mode_index(self.mtrap, label_i), 2 * mode_index(self.mtrap, label_j)
                if i == j:
                    M[i + 1, i] += -4 * W
                else:
                    M[i + 1, j] += -2 * W
                    M[j + 1, i] += -2 * W
        return matrices

    def f(self, t):
        """
        Common time dependence of every coupling, Σ_k c̃_k exp(-i(Δ_k - frame_offset)t) [·]
        (only meaningful when no tone has its own couplings)
        """
        return np.sum(self.complex_amplitudes * np.exp(-1j * (self.Δs - self.frame_offset) * t))

    def trap_modulation(self, t):
        """
        Fractional modulation ε(t) of the trap stiffness from the tones' intensity beat [·]
        """
        if not self.beat_visibility:
            return 0.0
        c = self.complex_amplitudes
        beat = sum(
            2 * np.real(c[k] * np.conjugate(c[l]) * np.exp(-1j * (self.Δs[k] - self.Δs[l]) * t))
            for k, l in self.beat_pairs
        )
        return self.beat_visibility * beat / self.total_power_fraction

    def drift_matrix(self, t):
        """
        Real quadrature-basis drift matrix A(t) in the cavity frame [2π·Hz]
        """
        phases = self.complex_amplitudes * np.exp(-1j * (self.Δs - self.frame_offset) * t)
        A = self.A_static + 2 * np.real(sum(phase * K for phase, K in zip(phases, self.Ks)))
        ε = self.trap_modulation(t)
        if ε:
            A = A + ε * self.B
        for ω, M in self.potential_matrices.items():
            A = A + np.real(M * np.exp(-1j * ω * t))
        return A

    # ------------------------------------------------------------------------------------------
    # DYNAMICS
    # ------------------------------------------------------------------------------------------
    def monodromy(self, rtol=1e-10, atol=1e-12):
        """
        Propagator Φ = 𝒯exp∫₀ᵀA dt and accumulated noise W over one period, from V(T) = ΦV(0)Φᵀ + W
        """
        n = self.D.shape[0]

        def rhs(t, y):
            Φ, W, A = y[:n*n].reshape(n, n), y[n*n:].reshape(n, n), self.drift_matrix(t)
            return np.concatenate([(A @ Φ).ravel(), (A @ W + W @ A.T + self.D).ravel()])

        y0 = np.concatenate([np.eye(n).ravel(), np.zeros(n*n)])
        solution = solve_ivp(rhs, (0.0, self.period), y0, method='DOP853', rtol=rtol, atol=atol)
        y = solution.y[:, -1]
        return y[:n*n].reshape(n, n), y[n*n:].reshape(n, n)

    def floquet_multipliers(self):
        """
        Eigenvalues of the one-period propagator: stable if and only if all |μ| < 1
        """
        Φ, _ = self.monodromy()
        return np.linalg.eigvals(Φ)

    def evolve(self, V0, times, rtol=1e-10, atol=1e-12):
        """
        Covariance matrices V(t) at the requested times, starting from V(times[0]) = V0
        """
        n = self.D.shape[0]

        def rhs(t, y):
            V, A = y.reshape(n, n), self.drift_matrix(t)
            return (A @ V + V @ A.T + self.D).ravel()

        times = np.asarray(times, dtype=float)
        max_step = self.period / 20
        solution = solve_ivp(rhs, (times[0], times[-1]), np.asarray(V0, dtype=float).ravel(),
                             t_eval=times, method='DOP853', rtol=rtol, atol=atol, max_step=max_step)
        return [solution.y[:, ii].reshape(n, n) for ii in range(solution.y.shape[1])]

    def periodic_steady_state(self, num_samples=16):
        """
        The asymptotic, T-periodic state, sampled at `num_samples` times over one period

        If any Floquet multiplier has |μ| ≥ 1 the drive amplifies some combination of modes and
        no steady state exists; the result then has stable=False and no covariances.
        """
        Φ, W = self.monodromy()
        multipliers = np.linalg.eigvals(Φ)
        if np.max(np.abs(multipliers)) >= 1:
            return PeriodicSteadyState(False, multipliers, monodromy=Φ)
        V0 = solve_discrete_lyapunov(Φ, W)
        times = np.linspace(0, self.period, num_samples, endpoint=False)
        covariances = [V0] if num_samples == 1 else self.evolve(V0, times)
        return PeriodicSteadyState(True, multipliers, times, covariances, monodromy=Φ)


@dataclass
class PeriodicSteadyState:
    """
    Result of MultiToneDrive.periodic_steady_state
    """
    stable: bool
    floquet_multipliers: np.ndarray
    times: np.ndarray = None
    covariances: list = None
    monodromy: np.ndarray = field(default=None, repr=False)     # Φ, whose eigenvalues are the multipliers

    @property
    def largest_multiplier(self):
        return float(np.max(np.abs(self.floquet_multipliers)))

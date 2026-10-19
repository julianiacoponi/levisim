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

beat_visibility = 1 is the co-polarised case and 0 a perfectly suppressed beat (e.g. orthogonal
polarisations, ignoring the small cross-polarised focal fields). This term has no analogue in
the microwave and membrane experiments the two-tone schemes come from.

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

from dataclasses import dataclass
from fractions import Fraction

from scipy.integrate import solve_ivp
from scipy.linalg import solve_discrete_lyapunov

from levisim.core.units import kHz, μs
from levisim.multiphysics.covariance import (
    complex_to_quadrature_transform,
    quadrature_drift_matrix,
    quadrature_diffusion_matrix,
)


__all__ = [
    'Tone',
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
    """
    Δ: float
    amplitude: float = 1.0
    phase: float = 0.0
    label: str = ''

    @property
    def complex_amplitude(self):
        return self.amplitude * np.exp(1j * self.phase)


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
    def __init__(self, mtrap, tones, beat_visibility=0.0, period=None, frame_offset=0.0):
        """
            - tones: the Tones; their Δ are always measured from the cavity resonance [2π·Hz]
            - beat_visibility: fraction of the tones' intensity beat that modulates the trap [·]
            - period: drive period, if not the common period of the tone frequencies [s]
            - frame_offset: the light is described in the frame rotating at ω_c + frame_offset.
              This is only a choice of frame (it rotates the cavity quadratures and leaves every
              mechanical moment unchanged), but it can make the drive periodic with a much shorter
              period, e.g. for tones at ω_c ∓ ν + s use frame_offset = s [2π·Hz]
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
        frequencies = list(self.Δs - self.frame_offset)
        if self.beat_visibility:
            frequencies += [self.Δs[k] - self.Δs[l] for k, l in self.beat_pairs]
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
            f'\n    period = {self.period / μs:.5f} μs'
            f'\n)'
        )

    # ------------------------------------------------------------------------------------------
    # MATRICES
    # ------------------------------------------------------------------------------------------
    def setup_matrices(self):
        """
        Splits the complex drift matrix as A(t) = A_static + f(t)·K + conj(f(t))·K̄ + ε(t)·B

        f(t) = Σ_k c̃_k exp(-iΔ_k t) multiplies every coupling G_j; the static part is levisim's
        drift matrix with the light-mech couplings removed and the cavity at zero detuning.
        """
        mtrap = self.mtrap
        n = mtrap.num_operators
        A_static = np.array(mtrap.drift_matrix, dtype=np.complex128)
        K = np.zeros((n, n), dtype=np.complex128)
        K_bar = np.zeros((n, n), dtype=np.complex128)
        B = np.zeros((n, n), dtype=np.complex128)

        # in the cavity frame the cavity no longer rotates at the (single) detuning; in the frame
        # rotating at ω_c + frame_offset it rotates at -frame_offset (levisim's -Δ·a†a with Δ → frame_offset)
        A_static[0, 0] = -mtrap.cavity.κ / 2 + 1j * self.frame_offset
        A_static[1, 1] = -mtrap.cavity.κ / 2 - 1j * self.frame_offset
        A_static[0, 2:] = A_static[1, 2:] = 0
        A_static[2:, 0] = A_static[2:, 1] = 0

        for ii in range(2, n):
            dimension = mtrap.dimensions[mtrap.noise_index_to_label[ii]]
            G, G_star = dimension.G, np.conjugate(dimension.G)
            # the same pattern as setup_linear_amplifier_model_matrices, with G → G·f(t)
            K[0, ii] = 1j * G
            K_bar[1, ii] = -1j * G_star
            if ii % 2 == 0:
                # annihilation operator b_j
                K_bar[ii, 0] = 1j * G_star
                K[ii, 1] = 1j * G
                # trap modulation (Ω_j ε/4)·X_j² gives db_j/dt += -i(Ω_j ε/2)(b_j + b_j†)
                B[ii, ii] = B[ii, ii + 1] = -0.5j * dimension.Ω
            else:
                # creation operator b_j†
                K_bar[ii, 0] = -1j * G_star
                K[ii, 1] = -1j * G
                B[ii, ii - 1] = B[ii, ii] = 0.5j * dimension.Ω

        T = complex_to_quadrature_transform(n)
        T_inverse = np.linalg.inv(T)
        self.A_static = quadrature_drift_matrix(A_static)
        self.K = T @ K @ T_inverse
        self.K_bar = T @ K_bar @ T_inverse
        assert np.allclose(self.K_bar, np.conjugate(self.K)), 'coupling split is inconsistent'
        self.B = quadrature_drift_matrix(B)
        self.D = quadrature_diffusion_matrix(mtrap)

    def f(self, t):
        """
        Common time dependence of every coupling, Σ_k c̃_k exp(-i(Δ_k - frame_offset)t) [·]
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
        A = self.A_static + 2 * np.real(self.f(t) * self.K)
        ε = self.trap_modulation(t)
        if ε:
            A = A + ε * self.B
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
            return PeriodicSteadyState(False, multipliers)
        V0 = solve_discrete_lyapunov(Φ, W)
        times = np.linspace(0, self.period, num_samples, endpoint=False)
        covariances = [V0] if num_samples == 1 else self.evolve(V0, times)
        return PeriodicSteadyState(True, multipliers, times, covariances)


@dataclass
class PeriodicSteadyState:
    """
    Result of MultiToneDrive.periodic_steady_state
    """
    stable: bool
    floquet_multipliers: np.ndarray
    times: np.ndarray = None
    covariances: list = None

    @property
    def largest_multiplier(self):
        return float(np.max(np.abs(self.floquet_multipliers)))

"""
Unit tests of SpectralDensityEstimate on synthetic traces with known spectra
"""
import numpy as np
import pytest

from levisim.analysis.spectral_density_estimate import SpectralDensityEstimate
from levisim.core.units import Hz
from levisim.data.simulation import Simulation
from levisim.data.trace import Trace
from levisim.physics.trap import Trap


Δt = 1e-6
N = 2**16


class WhiteSource:
    """Minimal stand-in for Simulation/Experiment: white traces 'a', 'b' with known variances"""
    def __init__(self, σa=2.0, σb=0.5, seed=0):
        rng = np.random.default_rng(seed)
        self.num_samples = N
        self.Δt = Δt
        self.time_length = N * Δt
        self.min_Δf = 1 / self.time_length
        self.Nyquist_frequency = 1 / (2 * Δt)
        self.time_array = np.arange(N) * Δt
        a = σa * rng.standard_normal(N)
        b = σb * rng.standard_normal(N)
        self.traces = {
            'a': Trace(a, 0, self.time_length, Δt, 'a', 'm', self, zero_mean=False),
            'b': Trace(b, 0, self.time_length, Δt, 'b', 'm', self, zero_mean=False),
            'a2': Trace(a.copy(), 0, self.time_length, Δt, 'a2', 'm', self, zero_mean=False),
        }


@pytest.fixture(scope='module')
def source():
    return WhiteSource()


def test_white_noise_level(source):
    """White noise of variance σ² has two-sided PSD σ²Δt and one-sided 2σ²Δt (DC bin undoubled)"""
    σ2 = np.var(source.traces['a'].values)
    two = SpectralDensityEstimate(source, 'aa', num_segments=64, one_sided=False)
    one = SpectralDensityEstimate(source, 'aa', num_segments=64)
    # 64 averages per bin, 512 bins: the mean level has a ~0.6% standard error
    assert np.mean(np.real(two.values)) == pytest.approx(σ2 * Δt, rel=0.03)
    assert np.mean(np.real(one.values[1:])) == pytest.approx(2 * σ2 * Δt, rel=0.03)


def test_parseval_two_sided(source):
    """Rectangular window: Σ S(f_k) Δf equals the mean square of the averaged samples exactly"""
    for num_segments in (1, 16, 64):
        est = SpectralDensityEstimate(source, 'aa', num_segments=num_segments, one_sided=False)
        used = source.traces['a'].values[:num_segments * est.num_samples_per_segment]
        assert np.sum(np.real(est.values)) * est.Δf == pytest.approx(np.mean(used**2), rel=1e-12)


def test_one_vs_two_sided(source):
    one = SpectralDensityEstimate(source, 'aa', num_segments=16)
    two = SpectralDensityEstimate(source, 'aa', num_segments=16, one_sided=False)
    h = two.halfway_index
    positive = two.values[h:]  # two-sided values reordered: [-f_Nyq, ..., -Δf, 0, Δf, ...]
    assert np.allclose(one.freq_range, two.freq_range[h:])
    assert one.values[0] == pytest.approx(positive[0])
    assert np.allclose(one.values[1:], 2 * positive[1:], rtol=1e-12)


def test_csd_of_identical_traces_is_the_psd(source):
    psd = SpectralDensityEstimate(source, 'aa', num_segments=16)
    csd = SpectralDensityEstimate(source, ('a', 'a2'), num_segments=16)
    assert csd.type == 'CSD'
    assert np.allclose(csd.values, psd.values, rtol=1e-12, atol=0)
    # S_ba = S_ab*
    ab = SpectralDensityEstimate(source, 'ab', num_segments=16)
    ba = SpectralDensityEstimate(source, 'ba', num_segments=16)
    assert np.allclose(ba.values, np.conj(ab.values))


def test_windows_preserve_broadband_level(source):
    rect = SpectralDensityEstimate(source, 'aa', num_segments=64)
    for window in ('hann', 'blackmanharris'):
        est = SpectralDensityEstimate(source, 'aa', num_segments=64, window=window)
        assert np.mean(np.real(est.values[1:])) == pytest.approx(np.mean(np.real(rect.values[1:])), rel=0.03)
    # Jon's Blackman-Harris normalisation 1/((287/800)√2) is the unit-mean-square one to within 0.15%
    w = SpectralDensityEstimate.make_window('blackmanharris', 4096)
    jon = __import__('scipy').signal.windows.blackmanharris(4096) / (287/800) / np.sqrt(2)
    assert np.mean(jon / w) == pytest.approx(1, abs=1.5e-3)
    assert SpectralDensityEstimate.make_window(None, 10) == 1


def test_spectrum_array(source):
    est = SpectralDensityEstimate(source, 'aa', num_segments=16)
    assert est.spectrum.shape == (len(est.freq_range), 2)
    assert np.array_equal(est.spectrum[:, 1], est.values)


def test_momentum_spectrum_and_radial_trace():
    """Spp = m²ω² Sxx, and momentum equipartition Var(px) = m kB T_eff"""
    from levisim.core.constants import kB
    from levisim.core.units import kHz
    trap = Trap(β=0.0, Γ_fbx=20*kHz, Γ_fby=20*kHz, Γ_fbz=5*kHz)
    sim = Simulation(trap=trap, num_samples=100_000, Δt=1e-7, seed=13, store_momenta=True, store_radial=True)
    assert set(sim.traces) == {'x', 'y', 'z', 'px', 'py', 'pz', 'r'}
    m = trap.bead.m
    assert np.var(sim.px.values) / (m * kB) == pytest.approx(trap.T_eff['x'], rel=0.1)
    sxx = SpectralDensityEstimate(sim, 'xx', num_segments=10)
    spp = SpectralDensityEstimate(sim, ('px', 'px'), num_segments=10)
    assert not hasattr(spp, 'Afit')  # not fitted: 'px' is not a position axis
    f0 = trap.Ωx / Hz
    band = (sxx.freq_range > f0 - 20e3) & (sxx.freq_range < f0 + 20e3)
    ratio = np.real(spp.values[band]) / (m**2 * (2*np.pi*sxx.freq_range[band])**2 * np.real(sxx.values[band]))
    assert np.allclose(ratio, 1, rtol=0.05)
    assert np.all(sim.r.values + sim.r.mean >= 0)


# ----------------------------------------------------------------------------------------------
# ensemble axis
# ----------------------------------------------------------------------------------------------
def test_ensemble_of_sources_is_the_mean_of_their_estimates_and_obeys_parseval():
    members = [WhiteSource(seed=seed) for seed in range(5)]
    ensemble = SpectralDensityEstimate(members, 'ab', num_segments=8, one_sided=False)
    singles = [SpectralDensityEstimate(m, 'ab', num_segments=8, one_sided=False) for m in members]
    np.testing.assert_allclose(ensemble.values, np.mean([s.values for s in singles], axis=0), rtol=1e-12, atol=1e-25)
    # Parseval over the ensemble (rectangular window): Σ S Δf = mean over members of 〈a b〉
    covariance = np.mean([np.mean(m.traces['a'].values * m.traces['b'].values) for m in members])
    assert np.sum(ensemble.values).real * ensemble.Δf == pytest.approx(covariance, rel=1e-10)
    assert ensemble.is_ensemble and ensemble.num_ensemble_members == 5
    assert ensemble.averaging_time == pytest.approx(5 * N * Δt)
    # the per-segment arrays gain a leading ensemble axis
    M = ensemble.num_samples_per_segment
    assert ensemble.fft_per_segment_A.shape == (5, 8, M) and ensemble.SDs_per_segment.shape == (5, 8, M)
    assert ensemble.covariance_matrices_per_segment.shape == (5, 8, 2, 2)
    np.testing.assert_array_equal(ensemble.fft_per_segment_A[3], singles[3].fft_per_segment_A)
    assert singles[0].fft_per_segment_A.shape == (8, M) and not singles[0].is_ensemble


def test_ensemble_scatter_falls_as_one_over_root_members_times_segments():
    """Bartlett: the relative scatter of a white PSD per bin is 1/√(members × segments)"""
    members = [WhiteSource(seed=100 + seed) for seed in range(16)]
    for num_members in (1, 16):
        estimate = SpectralDensityEstimate(members[:num_members], 'aa', num_segments=8, one_sided=False)
        S = estimate.values.real
        assert np.std(S) / np.mean(S) == pytest.approx(1 / np.sqrt(8 * num_members), rel=0.1)


def test_ensemble_argument_checks():
    with pytest.raises(ValueError, match='needs a source with trajectories'):
        SpectralDensityEstimate(WhiteSource(), 'aa', num_segments=4, ensemble=True)
    with pytest.raises(ValueError, match='not both'):
        SpectralDensityEstimate([WhiteSource()], 'aa', num_segments=4, ensemble=True)
    other = WhiteSource()
    other.Δt = 2 * Δt
    with pytest.raises(ValueError, match='same Δt'):
        SpectralDensityEstimate([WhiteSource(), other], 'aa', num_segments=4)
    lean = SpectralDensityEstimate([WhiteSource(), WhiteSource(seed=1)], 'aa', num_segments=4, keep_segments=False)
    assert lean.fft_per_segment_A is None and lean.SDs_per_segment is None

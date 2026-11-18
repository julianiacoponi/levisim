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

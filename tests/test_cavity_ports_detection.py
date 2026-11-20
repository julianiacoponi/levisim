"""
Fabry-Perot ports of `Cavity` and detection realism of the cavity-simulation records

- the default cavity is one port carrying all of κ, and an explicit single port changes no number;
- port j's output is the whole-κ output mixed with vacuum at efficiency κ_j/κ, in MultiTrap's
  spectra, in quadrature_squeezing, in QuadratureSystem (one noise pair per port and loss channel)
  and in the simulated records, which agree with each other;
- detection efficiency η scales the shot-noise-normalised signal, S − 1 → η(S − 1);
- slow LO phase noise averages the spectrum as quadrature_squeezing.phase_noise_averaged;
- conversion to volts gives the textbook shot noise 2·e·I·G², an electronic floor adds to it, and
  a record saved with `save_matfile` loads as `Experiment` data with the same spectrum.
"""
import numpy as np
import pytest

from levisim.analysis.spectral_density_estimate import SpectralDensityEstimate
from levisim.core.constants import node
from levisim.core.units import Hz, degrees, kHz, nm
from levisim.data.experiment import Experiment
from levisim.multiphysics.enums.quantum_linear_theory_method import (
    QuantumLinearTheoryMethod,
)
from levisim.multiphysics.multitrap import MultiTrap
from levisim.multiphysics.output_modes import (
    QuadratureSystem,
    _output_transfer,
    stack_cavity_modes,
)
from levisim.multiphysics.quadrature_squeezing import (
    output_quadrature_spectra,
    phase_noise_averaged,
    quadrature_spectrum,
)
from levisim.multiphysics.utils import shelf
from levisim.physics.bead import Bead
from levisim.physics.cavity import Cavity
from levisim.physics.tweezer import Tweezer
from levisim.simulation import (
    Heterodyne,
    Homodyne,
    PhaseNoise,
    PhotonNumber,
    VoltageConversion,
)
from levisim.simulation.records import ELEMENTARY_CHARGE

LAM = QuantumLinearTheoryMethod.LINEAR_AMPLIFIER_MODEL
F = np.linspace(20e3, 390e3, 75)          # Hz


def shelf_multitrap(**cavity_kwargs):
    return MultiTrap(num_traps=1, Cartesian_axes_per_trap=2, cavity_kwargs={**shelf.DEFAULT_cavity_kwargs, **cavity_kwargs},
                     gas_kwargs=shelf.DEFAULT_gas_kwargs)


@pytest.fixture(scope='module')
def single():
    return shelf_multitrap()


@pytest.fixture(scope='module')
def κ(single):
    return single.cavity.κ


@pytest.fixture(scope='module')
def two_port(κ):
    return shelf_multitrap(κ_ports=(0.6 * κ, 0.3 * κ), κ_loss=0.1 * κ, detected_port=1)


def toy():
    """The strongly coupled system of tests/test_quadrature_squeezing.py (squeezed near 80 kHz)"""
    toy_params = {'Ω_model': 'custom', 'toy_Ωjn_dict': {'x1': 122.0 * kHz, 'y1': 109.0 * kHz},
                  'gjn_dict': {'x1': 14.0 * kHz, 'y1': 10.0 * kHz}, 'use_Γ_formula': False, 'toy_Γ': 2 * np.pi * 1e-4,
                  'T': 300.0, 'm': 1e-18}
    return MultiTrap(num_traps=1, Cartesian_axes_per_trap=2, toy_params=toy_params, ω_range=np.array([100 * kHz]),
                     cavity_kwargs={'κ': 57 * kHz, 'Δ': -111 * kHz, 'ωtw': Tweezer(λ=1064 * nm).ω})


def at_frequencies(mtrap, ω, spectrum):
    old = mtrap.ω_range
    mtrap.ω_range = np.asarray(ω, dtype=float)
    try:
        return spectrum()
    finally:
        mtrap.ω_range = old


def symmetrised_homodyne(mtrap, f, Δt=0.0):
    ω = 2 * np.pi * np.asarray(f)
    S = at_frequencies(mtrap, np.concatenate([ω, -ω]), lambda: mtrap.homodyne_spectrum(LAM))
    return (S[:len(ω)] + S[len(ω):]) / 2 * np.sinc(np.asarray(f) * Δt)**2


# ----------------------------------------------------------------------------------------------
# Cavity ports
# ----------------------------------------------------------------------------------------------
def test_default_cavity_is_one_port_carrying_all_of_kappa():
    cavity = Cavity()
    assert cavity.has_default_ports and cavity.κ_ports is None and cavity.κ_loss == 0
    assert cavity.port_rates == (cavity.κ,) and cavity.detected_port == 0
    assert cavity.detected_port_efficiency == 1.0
    assert 'κ_ports' not in cavity.info_str


def test_ports_set_kappa_and_are_checked():
    cavity = Cavity(κ_ports=(100e3, 50e3), κ_loss=10e3, detected_port=1)
    assert cavity.κ == 160e3 and cavity.port_rates == (100e3, 50e3)
    assert cavity.detected_port_efficiency == 50e3 / 160e3
    assert 'κ_ports' in cavity.info_str
    assert Cavity(κ=160e3, κ_ports=(100e3, 50e3), κ_loss=10e3).κ == 160e3
    lossy = Cavity(κ=160e3, κ_loss=40e3)
    assert lossy.port_rates == (120e3,) and not lossy.has_default_ports
    with pytest.raises(ValueError, match='differs from the sum'):
        Cavity(κ=150e3, κ_ports=(100e3, 60e3))
    with pytest.raises(ValueError, match='detected_port'):
        Cavity(κ_ports=(100e3, 50e3), detected_port=2)
    with pytest.raises(ValueError, match='negative'):
        Cavity(κ_ports=(100e3, -1.0))
    with pytest.raises(ValueError, match='smaller than κ'):
        Cavity(κ=100e3, κ_loss=100e3)


def test_an_explicit_single_port_changes_no_number(single, κ):
    explicit = shelf_multitrap(κ_ports=(κ,))
    assert not explicit.cavity.has_default_ports and explicit.cavity.κ == κ
    assert np.array_equal(explicit.drift_matrix, single.drift_matrix)
    for name in ('homodyne_spectrum', 'heterodyne_spectrum'):
        assert np.array_equal(getattr(explicit, name)(LAM), getattr(single, name)(LAM)), name
    hom, het = explicit.homodyne_and_heterodyne_spectra(LAM)
    assert np.array_equal(hom, single.homodyne_spectrum(LAM)) and np.array_equal(het, single.heterodyne_spectrum(LAM))
    ω = 2 * np.pi * F
    for a, b in zip(output_quadrature_spectra(explicit, ω), output_quadrature_spectra(single, ω)):
        assert np.array_equal(a, b)


def test_detected_port_acts_as_an_efficiency(single, two_port):
    assert np.array_equal(two_port.drift_matrix, single.drift_matrix)
    η = two_port.cavity.detected_port_efficiency
    np.testing.assert_allclose(two_port.homodyne_spectrum(LAM), η * single.homodyne_spectrum(LAM) + 1 - η, rtol=1e-13)
    np.testing.assert_allclose(two_port.heterodyne_spectrum(LAM), η * single.heterodyne_spectrum(LAM) + 1 - η, rtol=1e-13)
    # quadrature_squeezing gives the same port spectrum (θhom = θ ⇔ φ = −θ)
    ω = two_port.ω_range[::40]
    S = quadrature_spectrum(*output_quadrature_spectra(two_port, ω), φ=-two_port.cavity.θhom)
    np.testing.assert_allclose(S, at_frequencies(two_port, ω, lambda: two_port.homodyne_spectrum(LAM)), rtol=1e-9)


def test_quadrature_system_has_one_noise_pair_per_port_and_loss(single, two_port):
    one = QuadratureSystem.from_multitrap(single)
    ported = QuadratureSystem.from_multitrap(two_port)
    assert one.B.shape == (6, 6) and ported.B.shape == (6, 10) and one.ports == {}
    np.testing.assert_allclose(ported.diffusion_matrix, one.diffusion_matrix, rtol=1e-13)
    assert np.array_equal(ported.A, one.A)
    rates = two_port.cavity.port_rates
    assert ported.ports['cavity'] == [(rates[0], (0, 1)), (rates[1], (2, 3))]
    assert ported.cavities['cavity'] == (rates[1], (2, 3))
    # the filtered-output machinery of output_modes sees the detected port
    θ = two_port.cavity.θhom
    m = ported.B.shape[1]
    N_total = np.zeros((m + 2, m + 2))
    N_total[:m, :m] = ported.N
    N_total[m:, m:] = np.eye(2)
    R = _output_transfer(ported, 'cavity', 2 * np.pi * F, 1.0)
    r = np.cos(θ) * R[:, 0] - np.sin(θ) * R[:, 1]
    S = np.einsum('wi,ij,wj->w', r.conj(), N_total, r).real
    np.testing.assert_allclose(S, symmetrised_homodyne(two_port, F), rtol=1e-9)


def test_stacking_ported_cavity_modes(two_port, single):
    stacked = stack_cavity_modes([two_port, single], ['A', 'B'])
    default = stack_cavity_modes([single, single], ['A', 'B'])
    assert stacked.B.shape == (8, 12) and default.B.shape == (8, 8)
    assert np.array_equal(stacked.A, default.A)
    np.testing.assert_allclose(stacked.diffusion_matrix, default.diffusion_matrix, rtol=1e-13)
    assert stacked.cavities['A'] == (two_port.cavity.port_rates[1], (2, 3))
    assert stacked.cavities['B'] == (single.cavity.κ, (6, 7))
    assert [columns for _, columns in stacked.ports['A']] == [(0, 1), (2, 3)]


def test_every_port_of_the_simulation_matches_multitrap(κ):
    mtrap = shelf_multitrap(κ_ports=(0.6 * κ, 0.3 * κ), κ_loss=0.1 * κ, detected_port=1)
    sim = mtrap.simulate(num_samples=2**17, seed=4, num_trajectories=8,
                         detection=[Homodyne(label='port1'), Homodyne(port=0, label='port0')])
    for label, port in (('port0', 0), ('port1', 1)):
        mtrap.cavity.detected_port = port
        np.testing.assert_allclose(sim.record_spectrum(label, F, kind='continuous'), symmetrised_homodyne(mtrap, F, sim.Δt),
                                   rtol=1e-9)
        f, S = sim.averaged_psd(label, num_segments=16)
        band = (f > 20e3) & (f < 390e3)
        r = S[band] / symmetrised_homodyne(mtrap, f[band], sim.Δt)
        assert abs(np.median(r) - 1) < 0.03 and abs(np.mean(r) - 1) < 0.03, (label, np.median(r), np.mean(r))
    mtrap.cavity.detected_port = 1
    # the two ports see the same cavity field, so their records are correlated
    c = np.corrcoef(sim.records['port0'].ravel(), sim.records['port1'].ravel())[0, 1]
    assert abs(c) > 0.02


# ----------------------------------------------------------------------------------------------
# detection realism
# ----------------------------------------------------------------------------------------------
def test_detection_efficiency_scales_the_signal_above_shot_noise(single):
    η = 0.4
    sim = single.simulate(num_samples=2**17, seed=6, num_trajectories=8,
                          detection=[Homodyne(label='ideal'), Homodyne(η=η, label='lossy')])
    # exact: S_η − 1 = η·(S_1 − 1) for the sampled records (white vacuum is not aliased)
    np.testing.assert_allclose(sim.record_spectrum('lossy', F) - 1, η * (sim.record_spectrum('ideal', F) - 1), rtol=1e-6)
    f, S_ideal = sim.averaged_psd('ideal', num_segments=16)
    _, S_lossy = sim.averaged_psd('lossy', num_segments=16)
    peaks = (f > 120e3) & (f < 170e3)
    ratio = (np.mean(S_lossy[peaks]) - 1) / (np.mean(S_ideal[peaks]) - 1)
    assert abs(ratio / η - 1) < 0.03, ratio


def test_phase_noise_models_have_their_statistics():
    rng = np.random.default_rng(0)
    Δt = 1e-7
    static = PhaseNoise.ornstein_uhlenbeck(σ=0.5, τ=1.0).sample(rng.standard_normal((4000, 200)), Δt)
    assert abs(static[:, 0].std() / 0.5 - 1) < 0.04 and np.abs(static[:, -1] - static[:, 0]).max() < 0.03
    ou = PhaseNoise.ornstein_uhlenbeck(σ=0.5, τ=1e-5).sample(rng.standard_normal((200, 20000)), Δt)
    lag = 100
    assert abs(ou.std() / 0.5 - 1) < 0.02
    assert abs(np.corrcoef(ou[:, :-lag].ravel(), ou[:, lag:].ravel())[0, 1] - np.exp(-lag * Δt / 1e-5)) < 0.03
    walk = PhaseNoise.random_walk(D=1e3).sample(rng.standard_normal((4000, 1000)), Δt)
    assert np.all(walk[:, 0] == 0) and abs(walk[:, -1].var() / (1e3 * 999 * Δt) - 1) < 0.06
    white = PhaseNoise.white(S_φ=1e-9).sample(rng.standard_normal((400, 1000)), Δt)
    assert abs(white.var() / (1e-9 / (2 * Δt)) - 1) < 0.02
    with pytest.raises(ValueError, match='Unknown phase-noise model'):
        PhaseNoise('pink')


def test_slow_phase_noise_averages_the_spectrum_as_phase_noise_averaged():
    """
    A static phase error per trajectory, φ ~ N(0, σ²): the spectrum of each trajectory is that of
    the quadrature θ + φ, and the ensemble average converges to phase_noise_averaged(σ²)
    """
    mtrap = toy()
    σ, θ, K = 0.5, -1.41, 256
    sim = mtrap.simulate(num_samples=2**14, seed=3, num_trajectories=K,
                         detection=[Homodyne(θ=θ, phase_noise=PhaseNoise.ornstein_uhlenbeck(σ, τ=100.0))])
    record = sim.detection[0]
    phases = record.phase_noise_samples[:, 0]
    assert np.abs(record.phase_noise_samples[:, -1] - phases).max() < 1e-2
    f, S = sim.averaged_psd('homodyne', num_segments=4)
    band = (f > 60e3) & (f < 180e3)
    spectra = output_quadrature_spectra(mtrap, 2 * np.pi * f[band])
    sinc2 = np.sinc(f[band] * sim.Δt)**2
    # 1) against the phases actually drawn (the MultiTrap convention θhom = θ ⇔ φ = −θ)
    drawn = np.mean([quadrature_spectrum(*spectra, φ=-(θ + φk)) for φk in phases], axis=0) * sinc2
    r = S[band] / drawn
    assert abs(np.mean(r) - 1) < 0.01 and abs(np.median(r) - 1) < 0.02
    # 2) the phases drawn average to the Gaussian of phase_noise_averaged (sampling error ~1/√K)
    gaussian = quadrature_spectrum(*phase_noise_averaged(*spectra, σ**2), φ=-θ) * sinc2
    unaveraged = quadrature_spectrum(*spectra, φ=-θ) * sinc2
    assert np.max(np.abs(drawn / gaussian - 1)) < 4 * np.max(np.abs(gaussian / unaveraged - 1)) / np.sqrt(K) + 0.01
    assert abs(np.mean(S[band] / gaussian) - 1) < 0.02
    assert np.max(np.abs(gaussian / unaveraged - 1)) > 0.08          # the averaging is visible


def test_voltage_conversion_and_electronic_noise(single):
    volts = VoltageConversion(power=1e-3, responsivity=0.7, transimpedance_gain=1e4)
    floor = 1e-15                                   # V²/Hz, one-sided
    sim = single.simulate(num_samples=2**16, seed=8, num_trajectories=4,
                          detection=[Homodyne(volts=volts, electronic_noise=floor), Homodyne(label='snu')])
    assert sim.traces['homodyne'].units == 'V' and sim.traces['snu'].units == 'shot noise units'
    shot = 2 * ELEMENTARY_CHARGE * 0.7 * 1e-3 * 1e4**2      # one-sided 2·e·I·G²
    f = np.array([3e6, 4e6])
    np.testing.assert_allclose(2 * sim.record_spectrum('homodyne', f), shot + floor, rtol=1e-4)
    np.testing.assert_allclose(sim.record_spectrum('homodyne', F),
                               volts.volts_per_shot_noise_unit()**2 * sim.record_spectrum('snu', F) + floor / 2, rtol=1e-12)
    fs, S = sim.averaged_psd('homodyne', num_segments=8, one_sided=True)
    far = (fs > 2e6) & (fs < 4.9e6)
    assert abs(np.mean(S[far]) / (shot + floor) - 1) < 0.01


def test_photon_number_in_volts_uses_the_mean_detected_power():
    mtrap = MultiTrap(num_traps=1, Cartesian_axes_per_trap=2, cavity_kwargs=shelf.DEFAULT_cavity_kwargs,
                      gas_kwargs=shelf.DEFAULT_gas_kwargs, beads=[Bead(Φ=node - 20 * degrees)])
    volts = VoltageConversion(responsivity=0.6, transimpedance_gain=2e3)
    sim = mtrap.simulate(num_samples=64, seed=0, detection=[PhotonNumber(η=0.5, volts=volts), PhotonNumber(η=0.5, label='snu')])
    record = sim.detection[0]
    power = 0.5 * record.mean_flux * 1.054571817e-34 * mtrap.cavity.ω
    np.testing.assert_allclose(record.shot_noise_power(sim), power, rtol=1e-9)
    np.testing.assert_allclose(record.scale(sim), 2e3 * np.sqrt(ELEMENTARY_CHARGE * 0.6 * power), rtol=1e-9)
    with pytest.raises(ValueError, match='no local oscillator'):
        PhotonNumber(phase_noise=PhaseNoise.white(1e-9))


def test_records_load_as_experiment_data(single, tmp_path):
    volts = VoltageConversion(power=2e-3)
    sim = single.simulate(num_samples=2**14, seed=9, detection=[Homodyne(volts=volts), Heterodyne(Ωhet=1e6 * Hz, volts=volts)])
    path = str(tmp_path / 'cavity_run.mat')
    channels = sim.save_matfile(path)
    assert channels == {'homodyne': 'A', 'heterodyne': 'B'}
    experiment = Experiment(path, channel_dict=channels)
    assert experiment.Δt == sim.Δt and experiment.num_samples == sim.num_samples
    for label in channels:
        np.testing.assert_allclose(experiment.traces[label].values, sim.traces[label].values, rtol=0, atol=1e-12 * np.abs(sim.traces[label].values).max())
        assert experiment.traces[label].units == 'V'
        from_experiment = SpectralDensityEstimate(experiment, (label, label), num_segments=8)
        from_simulation = SpectralDensityEstimate(sim, (label, label), num_segments=8)
        np.testing.assert_allclose(from_experiment.values, from_simulation.values, rtol=1e-9)

"""
CavitySimulation, model='linear': exact discretisation, output records and their validation

Fast versions of the validation ladder of the design plan (the long versions, with more samples,
are in docs/guide/scripts/cavity_simulation_validation.py):

- the Van Loan step is exact (closed form for one mode; the steady state is a fixed point of one step);
- the equal-time covariance of the simulated state is steady_state_covariance;
- homodyne and position PSDs match the SYMMETRISED MultiTrap spectra times sinc²(fΔt) for 1×1, 1×2,
  2×2 (stable phase) and a stacked two-mode system, and the homodyne angle has levisim's sign;
- the heterodyne record is the two sidebands S_aout,aout(ω + Ωhet) + S_aout†,aout†(ω − Ωhet);
- shot noise alone is flat at 1; a squeezing configuration dips below 1 at the right quadrature;
- an unstable system refuses to start in a steady state and grows at max_growth_rate from zero.

Spectral estimates: `CavitySimulation.averaged_psd` (Hann window normalised to unit mean square,
non-overlapping segments, two-sided, ∫S df = variance), whose relative scatter per bin is
1/√(segments × trajectories).
"""
import numpy as np
import pytest

from levisim.analysis.spectral_density_estimate import SpectralDensityEstimate
from levisim.core.constants import node
from levisim.core.units import Hz, degrees, kHz, nm, μs
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
    optimal_quadrature_spectrum,
    output_quadrature_spectra,
    quadrature_spectrum,
)
from levisim.multiphysics.utils import shelf
from levisim.physics.bead import Bead
from levisim.physics.tweezer import Tweezer
from levisim.simulation import (
    CavitySimulation,
    Heterodyne,
    Homodyne,
    PhotonNumber,
    Position,
)
from levisim.simulation.backends import numba_available
from levisim.simulation.cavity import van_loan_discretisation

LAM = QuantumLinearTheoryMethod.LINEAR_AMPLIFIER_MODEL
BAND = (20e3, 390e3)      # Hz, where the shelf systems have their features


# ----------------------------------------------------------------------------------------------
# systems and theory helpers
# ----------------------------------------------------------------------------------------------
def shelf_multitrap(num_traps=1, axes=2, **kwargs):
    return MultiTrap(num_traps=num_traps, Cartesian_axes_per_trap=axes, cavity_kwargs=shelf.DEFAULT_cavity_kwargs,
                     gas_kwargs=shelf.DEFAULT_gas_kwargs, **kwargs)


@pytest.fixture(scope='module')
def m1x2():
    return shelf_multitrap()


@pytest.fixture(scope='module')
def m2x2():
    m = shelf_multitrap(2, 2, φn_dict={2: 30 * degrees})
    assert m.is_stable
    return m


def toy(g=None, κ=57 * kHz, Δ=-111 * kHz):
    """The strongly coupled, sideband-resolved system of tests/test_quadrature_squeezing.py"""
    g = {'x1': 14.0 * kHz, 'y1': 10.0 * kHz} if g is None else g
    toy_params = {'Ω_model': 'custom', 'toy_Ωjn_dict': {'x1': 122.0 * kHz, 'y1': 109.0 * kHz}, 'gjn_dict': g,
                  'use_Γ_formula': False, 'toy_Γ': 2 * np.pi * 1e-4, 'T': 300.0, 'm': 1e-18}
    return MultiTrap(num_traps=1, Cartesian_axes_per_trap=2, toy_params=toy_params, ω_range=np.array([100 * kHz]),
                     cavity_kwargs={'κ': κ, 'Δ': Δ, 'ωtw': Tweezer(λ=1064 * nm).ω})


def at_frequencies(mtrap, ω, spectrum):
    """A MultiTrap spectrum function evaluated on another ω grid"""
    old = mtrap.ω_range
    mtrap.ω_range = np.asarray(ω, dtype=float)
    try:
        return spectrum()
    finally:
        mtrap.ω_range = old


def symmetrised(mtrap, f, spectrum):
    """½[S(ω) + S(-ω)] of a MultiTrap spectrum, at f [Hz]"""
    ω = 2 * np.pi * np.asarray(f)
    S = at_frequencies(mtrap, np.concatenate([ω, -ω]), spectrum)
    return (S[:len(ω)] + S[len(ω):]) / 2


def homodyne_theory(mtrap, f, Δt, θ=None):
    old = mtrap.cavity.θhom
    if θ is not None:
        mtrap.cavity.θhom = θ
    try:
        return symmetrised(mtrap, f, lambda: mtrap.homodyne_spectrum(LAM)) * np.sinc(f * Δt)**2
    finally:
        mtrap.cavity.θhom = old


def position_theory(mtrap, label, f, Δt):
    zpf = mtrap.dimensions[label].zpf
    return symmetrised(mtrap, f, lambda: mtrap.position_power_spectrum(label, LAM)) * zpf**2 * np.sinc(f * Δt)**2


def ratio_stats(S, theory):
    r = S / theory
    return np.median(r), np.mean(r), np.mean(np.abs(np.log(r)))


# ----------------------------------------------------------------------------------------------
# exact discretisation
# ----------------------------------------------------------------------------------------------
def test_van_loan_matches_the_ornstein_uhlenbeck_closed_form():
    γ, σ2, Δt = 3.7e5, 2.0e4, 0.13e-6
    Φ, Q = van_loan_discretisation(np.array([[-γ]]), np.array([[σ2]]), Δt)
    np.testing.assert_allclose(Φ[0, 0], np.exp(-γ * Δt), rtol=1e-14)
    np.testing.assert_allclose(Q[0, 0], σ2 * -np.expm1(-2 * γ * Δt) / (2 * γ), rtol=1e-12)


@pytest.mark.parametrize('Δt', [0.1 * μs, 1 * μs, 10 * μs])
def test_the_steady_state_is_a_fixed_point_of_one_exact_step(m1x2, m2x2, Δt):
    """Φ·V·Φᵀ + Q = V at ANY Δt: there is no time-step error"""
    for mtrap in (m1x2, m2x2):
        sim = CavitySimulation(mtrap, num_samples=4, Δt=Δt, seed=0)
        n = sim.system.A.shape[0]
        V = sim.system.steady_state_covariance()
        Φ = sim.Φ[:n]
        np.testing.assert_allclose(Φ @ V @ Φ.T + sim.Q[:n, :n], V, rtol=0, atol=1e-9 * np.abs(V).max())


def test_equal_time_covariance_matches_steady_state_covariance(m1x2):
    sim = CavitySimulation(m1x2, num_samples=2**17, seed=3, num_trajectories=8, store_state=True, detection=[])
    states = sim.states.reshape(-1, sim.states.shape[-1])
    V_sim = states.T @ states / len(states)
    V = sim.system.steady_state_covariance()
    d = np.sqrt(np.diag(V))
    # 8 trajectories × 13 ms against mechanical correlation times of ~0.4 ms: a few % sampling error
    assert np.max(np.abs(V_sim - V) / np.outer(d, d)) < 0.06
    np.testing.assert_allclose(np.diag(V_sim), np.diag(V), rtol=0.08)


# ----------------------------------------------------------------------------------------------
# homodyne and positions against the symmetrised MultiTrap spectra
# ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize('shape', [(1, 1), (1, 2), (2, 2)])
def test_homodyne_and_position_psds_match_symmetrised_multitrap(shape, m1x2, m2x2):
    mtrap = {(1, 1): None, (1, 2): m1x2, (2, 2): m2x2}[shape] or shelf_multitrap(*shape)
    # the 2×2 system at φ2 = 30° has a mode damped at only 294 s⁻¹ (a 47 Hz half-width): its
    # segments must be long (6.6 ms, 4 per trajectory) for the Hann window's leakage around that
    # narrow peak not to bias the mean ratio
    segments, trajectories = ((16, 8) if shape != (2, 2) else (4, 24))
    sim = mtrap.simulate(num_samples=2**18, seed=5, num_trajectories=trajectories)
    f, S = sim.averaged_psd('homodyne', num_segments=segments)
    band = (f > BAND[0]) & (f < BAND[1])
    median, mean, log_dev = ratio_stats(S[band], homodyne_theory(mtrap, f[band], sim.Δt))
    # ~100 averages: ±10% per bin; over thousands of bins the median and mean are good to ~1%
    assert abs(median - 1) < 0.03 and abs(mean - 1) < 0.03, (median, mean)
    assert log_dev < 0.11
    for label in mtrap.labels:
        f, S = sim.averaged_psd(label, num_segments=segments)
        median, mean, _ = ratio_stats(S[band], position_theory(mtrap, label, f[band], sim.Δt))
        assert abs(median - 1) < 0.03 and abs(mean - 1) < 0.03, (label, median, mean)


def test_homodyne_angle_has_levisims_sign(m1x2):
    """h = X_out cos θ + P_out sin θ with P_out = i(a_out − a_out†): the record follows +θ, not −θ"""
    sim = m1x2.simulate(num_samples=2**18, seed=11, num_trajectories=4)
    f, S = sim.averaged_psd('homodyne', num_segments=16)
    band = (f > BAND[0]) & (f < BAND[1])
    θ = m1x2.cavity.θhom
    _, _, right = ratio_stats(S[band], homodyne_theory(m1x2, f[band], sim.Δt, θ))
    _, _, wrong = ratio_stats(S[band], homodyne_theory(m1x2, f[band], sim.Δt, -θ))
    assert right < 0.13, right
    assert wrong > 0.2, wrong


def test_continuous_record_spectrum_is_the_symmetrised_multitrap_spectrum(m1x2, m2x2):
    """The theory of `record_spectrum(kind='continuous')` is MultiTrap's, to round-off"""
    f = np.linspace(5e3, 600e3, 157)
    for mtrap in (m1x2, m2x2):
        sim = mtrap.simulate(num_samples=4, seed=0)
        np.testing.assert_allclose(sim.record_spectrum('homodyne', f, kind='continuous'),
                                   homodyne_theory(mtrap, f, sim.Δt), rtol=1e-9)
        for label in mtrap.labels:
            np.testing.assert_allclose(sim.record_spectrum(label, f, kind='continuous'),
                                       position_theory(mtrap, label, f, sim.Δt), rtol=1e-9)


def test_discrete_record_spectrum_differs_from_sinc_squared_only_by_aliasing(m1x2):
    """
    The exact spectrum of the sampled record = Σ_k S_c(f + k/Δt)·sinc²(fΔt + k); its white part
    (the vacuum, level 1) is not aliased at all, since Σ_k sinc²(x + k) = 1
    """
    sim = m1x2.simulate(num_samples=4, seed=0)
    f = np.linspace(10e3, 4.9e6, 97)
    discrete = sim.record_spectrum('homodyne', f)
    fs = 1 / sim.Δt
    aliased = sum(sim.record_spectrum('homodyne', f + k * fs, kind='continuous') for k in range(-60, 61))
    k = np.arange(-60, 61)
    tail = 1 - np.sinc(f[:, None] * sim.Δt + k[None]).__pow__(2).sum(axis=1)     # aliases beyond ±60, at the vacuum level
    np.testing.assert_allclose(discrete, aliased + tail, rtol=2e-5)


# ----------------------------------------------------------------------------------------------
# heterodyne
# ----------------------------------------------------------------------------------------------
def heterodyne_record_theory(mtrap, Ωhet, f):
    """½[S_h(ω) + S_h(−ω)] with S_h(ω) = S_aout,aout(ω + Ωhet) + S_aout†,aout†(ω − Ωhet), from MultiTrap"""
    def S_aa(ω):
        return at_frequencies(mtrap, ω, lambda: mtrap.heterodyne_spectrum(LAM))

    def S_adad(ω):
        return np.array([mtrap.noise_vector_to_power_spectrum(mtrap.output_light_noise_vectors(LAM, w)[1]) for w in ω])

    ω = 2 * np.pi * np.asarray(f)
    S_h = lambda w: S_aa(w + Ωhet) + S_adad(w - Ωhet)
    return (S_h(ω) + S_h(-ω)) / 2


def test_heterodyne_record_is_the_two_shifted_sidebands(m1x2):
    Ωhet = 1.0 * 1e6 * Hz
    sim = m1x2.simulate(num_samples=4, seed=0, detection=[Heterodyne(Ωhet=Ωhet)])
    f = np.linspace(500e3, 1500e3, 211)
    np.testing.assert_allclose(sim.record_spectrum('heterodyne', f, kind='continuous'),
                               heterodyne_record_theory(m1x2, Ωhet, f) * np.sinc(f * sim.Δt)**2, rtol=1e-9)
    # sideband asymmetry: the two mechanical sidebands of the record have different weights
    Ωx = m1x2.dimensions['x1'].Ω / Hz
    lower, upper = sim.record_spectrum('heterodyne', [1e6 - Ωx, 1e6 + Ωx], kind='continuous')
    assert abs(lower / upper - 1) > 0.05


def test_simulated_heterodyne_matches_its_exact_spectrum(m1x2):
    Ωhet = 1.0 * 1e6 * Hz
    sim = m1x2.simulate(num_samples=2**18, seed=3, num_trajectories=8, detection=[Heterodyne(Ωhet=Ωhet)])
    f, S = sim.averaged_psd('heterodyne', num_segments=16)
    band = (f > 600e3) & (f < 1400e3)
    median, mean, log_dev = ratio_stats(S[band], sim.record_spectrum('heterodyne', f[band]))
    assert abs(median - 1) < 0.03 and abs(mean - 1) < 0.03 and log_dev < 0.1, (median, mean, log_dev)


def test_heterodyne_needs_a_nonzero_frequency(m1x2):
    assert m1x2.cavity.Ωhet == 0
    with pytest.raises(ValueError, match='non-zero Ωhet'):
        m1x2.simulate(num_samples=4, detection=[Heterodyne()])


# ----------------------------------------------------------------------------------------------
# shot noise and squeezing
# ----------------------------------------------------------------------------------------------
def test_shot_noise_alone_is_flat_at_one():
    mtrap = toy(g={'x1': 0.0, 'y1': 0.0})
    sim = mtrap.simulate(num_samples=2**16, seed=2, num_trajectories=8,
                         detection=[Homodyne(θ=0.4), Homodyne(θ=0.4, η=0.3, label='lossy'), Heterodyne(Ωhet=1e6 * Hz)])
    for label in ('homodyne', 'lossy', 'heterodyne'):
        f, S = sim.averaged_psd(label, num_segments=8)
        inside = (f > 0) & (f < 0.99 / (2 * sim.Δt))
        # white noise is not aliased by the sample average: level 1 up to the Nyquist frequency
        assert abs(np.mean(S[inside]) - 1) < 0.01, label
        assert abs(np.var(sim.records[label]) * sim.Δt - 1) < 0.01, label


def test_squeezing_dips_below_shot_noise_at_the_optimal_quadrature():
    mtrap = toy()
    f_probe = np.linspace(1, 400, 2000) * 1e3 + 13.7
    S_opt, φ_opt = optimal_quadrature_spectrum(*output_quadrature_spectra(mtrap, 2 * np.pi * f_probe))
    i = np.argmin(S_opt)
    φ = φ_opt[i]
    assert S_opt[i] < 0.92
    # MultiTrap.homodyne_spectrum at θhom = θ is quadrature_spectrum(φ = −θ)
    sim = mtrap.simulate(num_samples=2**19, seed=7, num_trajectories=8,
                         detection=[Homodyne(θ=-φ, label='squeezed'), Homodyne(θ=-φ + np.pi / 2, label='anti')])
    f, S = sim.averaged_psd('squeezed', num_segments=64)
    band = (f > f_probe[i] - 10e3) & (f < f_probe[i] + 10e3)
    theory = quadrature_spectrum(*output_quadrature_spectra(mtrap, 2 * np.pi * f[band]), φ=φ) * np.sinc(f[band] * sim.Δt)**2
    σ = 1 / np.sqrt(band.sum() * 64 * 8)                              # relative error of the band average
    assert np.mean(theory) < 1 - 6 * σ
    assert np.mean(S[band]) < 1 - 4 * σ                               # below shot noise
    assert abs(np.mean(S[band]) / np.mean(theory) - 1) < 3 * σ
    _, S_anti = sim.averaged_psd('anti', num_segments=64)
    assert np.mean(S_anti[band]) > 1


# ----------------------------------------------------------------------------------------------
# stacked two-mode system (no MultiTrap describes it)
# ----------------------------------------------------------------------------------------------
def test_stacked_two_mode_system_matches_its_output_transfer():
    red = toy(Δ=-111 * kHz)
    blue = toy(g={'x1': 0.6 * 14 * kHz, 'y1': 0.6 * 10 * kHz}, Δ=+250 * kHz)
    system = stack_cavity_modes([red, blue], ['A', 'B'])
    assert system.largest_growth_rate() < 0
    sim = CavitySimulation(system, num_samples=2**17, seed=9, num_trajectories=8,
                           detection=[Homodyne(θ=0.3, cavity='A', label='hA'), Homodyne(θ=1.1, cavity='B', label='hB'),
                                      Position('x1', units='zpf')])
    m = system.B.shape[1]
    N_total = np.zeros((m + 2, m + 2))
    N_total[:m, :m] = system.N
    N_total[m:, m:] = np.eye(2)
    for label, cavity, θ in (('hA', 'A', 0.3), ('hB', 'B', 1.1)):
        f, S = sim.averaged_psd(label, num_segments=16)
        band = (f > BAND[0]) & (f < BAND[1])
        R = _output_transfer(system, cavity, 2 * np.pi * f[band], 1.0)      # (Q_out, P_out) with P = i(a† − a)
        r = np.cos(θ) * R[:, 0] - np.sin(θ) * R[:, 1]                         # levisim's P_out = −P
        theory = np.einsum('wi,ij,wj->w', r.conj(), N_total, r).real * np.sinc(f[band] * sim.Δt)**2
        np.testing.assert_allclose(sim.record_spectrum(label, f[band], kind='continuous'), theory, rtol=1e-9)
        median, mean, _ = ratio_stats(S[band], theory)
        assert abs(median - 1) < 0.04 and abs(mean - 1) < 0.04, (label, median, mean)


# ----------------------------------------------------------------------------------------------
# unstable systems
# ----------------------------------------------------------------------------------------------
def test_unstable_system_refuses_a_steady_start_and_grows_at_max_growth_rate():
    mtrap = shelf_multitrap(2, 2)            # default tweezer phases: φ2 = 90°, unstable
    assert not mtrap.is_stable
    with pytest.raises(ValueError, match='no steady'):
        mtrap.simulate(num_samples=4)
    sim = mtrap.simulate(num_samples=40_000, seed=2, num_trajectories=16, initial_state='zero', store_state=True)
    eigenvalues, left = np.linalg.eig(sim.system.A.T)
    i = np.argmax(eigenvalues.real)
    amplitude = sim.states @ left[:, i]      # the least stable mode
    power = np.mean(np.abs(amplitude)**2, axis=0)
    t = np.arange(power.size) * sim.Δt
    late = t > 1.5e-3
    rate = np.polyfit(t[late], np.log(power[late]), 1)[0] / 2
    np.testing.assert_allclose(rate, mtrap.max_growth_rate, rtol=0.01)


# ----------------------------------------------------------------------------------------------
# records, interface, backends
# ----------------------------------------------------------------------------------------------
def test_photon_number_is_the_amplitude_quadrature_times_the_static_field():
    mtrap = shelf_multitrap(beads=[Bead(Φ=node - 20 * degrees)])
    α = (mtrap.cavity.A0 + 1j * mtrap.cavity.P0) / 2
    θ_α = -np.angle(-np.sqrt(mtrap.cavity.κ) * α)
    sim = mtrap.simulate(num_samples=2**12, seed=1, detection=[
        PhotonNumber(), PhotonNumber(units='photons/s', label='flux'), Homodyne(θ=θ_α, label='amplitude')])
    scale = np.std(sim.records['amplitude'])
    # identical rows of the one-step covariance: equal to the √ε round-off of its factorisation
    np.testing.assert_allclose(sim.records['photon_number'], sim.records['amplitude'], rtol=0, atol=1e-6 * scale)
    np.testing.assert_allclose(sim.records['flux'], np.sqrt(mtrap.cavity.κ) * abs(α) * sim.records['photon_number'],
                               rtol=0, atol=1e-6 * np.std(sim.records['flux']))
    # at a node the static field vanishes, and so does the linear flux fluctuation
    with pytest.raises(ValueError, match='static output field is zero'):
        toy().simulate(num_samples=4, detection=[PhotonNumber()])


def test_traces_feed_the_experiment_analysis_path(m1x2):
    sim = m1x2.simulate(num_samples=2**16, seed=1, num_trajectories=2)
    assert set(sim.traces) == {'homodyne', 'x1', 'y1'}
    assert sim.homodyne is sim.traces['homodyne'] and sim.traces['x1'].units == 'm'
    sde = SpectralDensityEstimate(sim, ('homodyne', 'homodyne'), num_segments=16, one_sided=False)
    other = SpectralDensityEstimate(sim.trajectory(1), ('homodyne', 'homodyne'), num_segments=16, one_sided=False)
    assert sde.values.shape == other.values.shape
    assert not np.allclose(sde.values, other.values)            # independent trajectories
    # far above the cavity linewidth the two-sided level is the vacuum, 1
    far = np.abs(sde.freq_range) > 2e6
    assert abs(np.mean(sde.values[far].real) - 1) < 0.03
    one_sided = SpectralDensityEstimate(sim, ('homodyne', 'homodyne'), num_segments=16)
    assert abs(np.mean(one_sided.values[one_sided.freq_range > 2e6].real) - 2) < 0.06


def test_seeded_runs_are_reproducible_and_backends_agree(m1x2):
    a = m1x2.simulate(num_samples=3000, seed=4, num_trajectories=3, backend='numpy', store_state=True)
    b = m1x2.simulate(num_samples=3000, seed=4, num_trajectories=3, backend='numpy', store_state=True)
    for label in a.records:
        assert np.array_equal(a.records[label], b.records[label])
    assert not np.allclose(a.records['homodyne'][0], a.records['homodyne'][1])
    if numba_available():
        c = m1x2.simulate(num_samples=3000, seed=4, num_trajectories=3, backend='numba', store_state=True)
        assert c.backend_used == 'numba'
        for label in a.records:
            np.testing.assert_allclose(c.records[label], a.records[label], rtol=1e-9, atol=1e-9 * np.abs(a.records[label]).max())
        np.testing.assert_allclose(c.states, a.states, rtol=1e-9, atol=1e-9 * np.abs(a.states).max())


def test_initial_states_and_argument_checks(m1x2):
    n = 6
    zero = m1x2.simulate(num_samples=2, initial_state='zero', store_state=True, seed=0)
    assert np.array_equal(zero.states[:, 0], np.zeros((1, n)))
    given = m1x2.simulate(num_samples=2, initial_state=np.arange(n, dtype=float), store_state=True, seed=0)
    assert np.array_equal(given.states[0, 0], np.arange(n))
    with pytest.raises(TypeError, match="only to model='nonlinear'"):
        m1x2.simulate(num_samples=2, substeps=2)
    with pytest.raises(ValueError, match='RK_method'):
        m1x2.simulate(RK_method='vector_4')
    with pytest.raises(TypeError, match='unknown keyword'):
        m1x2.simulate(num_sample=10)
    with pytest.raises(ValueError, match='same label'):
        m1x2.simulate(num_samples=2, detection=[Homodyne(), Homodyne()])
    with pytest.raises(ValueError, match='shape'):
        m1x2.simulate(num_samples=2, initial_state=np.zeros(3))


def test_quadrature_system_input_and_default_records():
    system = QuadratureSystem.from_multitrap(toy())
    sim = CavitySimulation(system, num_samples=64, seed=0)
    assert set(sim.records) == {'homodyne', 'x1', 'y1'}
    assert sim.traces['x1'].units == 'zpf'
    with pytest.raises(ValueError, match='pass α'):
        CavitySimulation(system, num_samples=4, detection=[PhotonNumber()])
    with pytest.raises(ValueError, match='MultiTrap'):
        CavitySimulation(system, num_samples=4, detection=[Homodyne()])

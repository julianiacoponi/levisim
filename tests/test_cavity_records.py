"""
Opt-in record options of CavitySimulation: the static output carrier of homodyne and heterodyne
records (with the conversion of local-oscillator phase noise into the record), and instantaneous
Position sampling

Closed forms used:
- carrier of a homodyne at angle θ: 2·√η·abs(ᾱ_out)·cos(θ − θ_α), ᾱ_out = −√κ·α, θ_α = −arg ᾱ_out,
  with α = αR + i·αI of the MultiTrap's cavity (an expression independent of the A0, P0 the
  record uses); with white phase noise φ of one-sided PSD S_φ at θ = θ_α + π/2, the record's
  two-sided PSD rises by 4η·abs(ᾱ_out)²·S_φ/2
- heterodyne carrier: the average of 2√η·abs(ᾱ_out)·cos(Ωhet·t + φ0 − θ_α) over each sample
  interval, checked against a 2001-point Simpson quadrature
- instantaneous samples of a stationary state: variance x_zpf²·V_ii (V = steady_state_covariance),
  and ∫ S df over the Nyquist band of the discrete record_spectrum equals the same variance
"""

import numpy as np
import pytest
from scipy.integrate import simpson

from levisim.core.constants import node
from levisim.core.units import degrees, μs
from levisim.multiphysics.enums.quantum_linear_theory_method import (
    QuantumLinearTheoryMethod,
)
from levisim.multiphysics.multitrap import MultiTrap
from levisim.multiphysics.utils import shelf
from levisim.physics.bead import Bead
from levisim.simulation import (
    CavitySimulation,
    Heterodyne,
    Homodyne,
    PhotonNumber,
    Position,
)
from levisim.simulation.backends import numba_available
from levisim.simulation.records import PhaseNoise, VoltageConversion

LAM = QuantumLinearTheoryMethod.LINEAR_AMPLIFIER_MODEL


def shelf_multitrap(num_traps=1, axes=2, **kwargs):
    return MultiTrap(num_traps=num_traps, Cartesian_axes_per_trap=axes, cavity_kwargs=shelf.DEFAULT_cavity_kwargs,
                     gas_kwargs=shelf.DEFAULT_gas_kwargs, **kwargs)


@pytest.fixture(scope='module')
def off_node():
    """A bead 20° from the node: a non-zero static output field"""
    return shelf_multitrap(beads=[Bead(Φ=node - 20 * degrees)])


def carrier_closed_form(mtrap, η=1.0):
    """(2√η·abs(ᾱ_out), θ_α) from α = αR + i·αI of the cavity"""
    c = mtrap.cavity
    α_out = -np.sqrt(c.κ) * complex(c.αR + 1j * c.αI)
    return 2 * np.sqrt(η) * abs(α_out), -np.angle(α_out)


# ----------------------------------------------------------------------------------------------
# static carrier (decision 18)
# ----------------------------------------------------------------------------------------------
def test_carrier_is_off_by_default_and_adds_exactly_the_dc_term(off_node):
    A, θ_α = carrier_closed_form(off_node, η=0.8)
    θs = (θ_α, θ_α + np.pi / 3, θ_α + np.pi / 2)

    def run(carrier):
        detection = [Homodyne(θ=θ, η=0.8, static_carrier=carrier, label=f'h{i}') for i, θ in enumerate(θs)]
        return off_node.simulate(num_samples=2**12, seed=4, num_trajectories=2, detection=detection + [Position('x1')])
    off, on, default = run(False), run(True), off_node.simulate(
        num_samples=2**12, seed=4, num_trajectories=2,
        detection=[Homodyne(θ=θ, η=0.8, label=f'h{i}') for i, θ in enumerate(θs)] + [Position('x1')])
    for label in off.records:
        np.testing.assert_array_equal(off.records[label], default.records[label])
    np.testing.assert_array_equal(on.records['x1'], off.records['x1'])
    for i, θ in enumerate(θs):
        dc = on.records[f'h{i}'] - off.records[f'h{i}']
        np.testing.assert_allclose(dc, A * np.cos(θ - θ_α), rtol=1e-12, atol=1e-12 * A)
    # the amplitude quadrature carries all of it, the phase quadrature none
    assert A > 1e5 and abs(on.records['h2'].mean()) < 1e-6 * A
    # a zero-mean trace removes the DC term; the records keep it
    assert abs(on.traces['h0'].values.mean()) < 1e-9 * A
    assert on.records['h0'].mean() == pytest.approx(A, rel=1e-3)


def test_white_lo_phase_noise_converts_the_carrier_into_the_record(off_node):
    """At θ = θ_α + π/2 the carrier converts φ at first order: S → S + A²·S_φ/2 (two-sided)"""
    A, θ_α = carrier_closed_form(off_node)
    S_φ = 2.0 / A**2          # one-sided [rad²/Hz]: adds 1 shot-noise unit (two-sided)
    noise = PhaseNoise.white(S_φ)
    θ = θ_α + np.pi / 2
    sim = off_node.simulate(num_samples=2**16, seed=6, num_trajectories=8, detection=[
        Homodyne(θ=θ, phase_noise=noise, static_carrier=True, label='with'),
        Homodyne(θ=θ, phase_noise=noise, label='without'),
        Homodyne(θ=θ_α, phase_noise=noise, static_carrier=True, label='amplitude'),
    ])
    f, S_with = sim.averaged_psd('with', num_segments=16)
    _, S_without = sim.averaged_psd('without', num_segments=16)
    _, S_amplitude = sim.averaged_psd('amplitude', num_segments=16)
    band = (f > 600e3) & (f < 4e6)          # away from the mechanical features
    theory = sim.record_spectrum('with', f[band])
    added = A**2 * S_φ / 2
    assert added == pytest.approx(1.0)
    assert np.median(S_with[band] / (theory + added)) == pytest.approx(1, abs=0.02)
    assert np.median(S_without[band] / theory) == pytest.approx(1, abs=0.02)
    # the amplitude quadrature converts φ only at second order (∝ φ², here 10⁻¹⁰ of the carrier)
    assert np.median(S_amplitude[band] / sim.record_spectrum('amplitude', f[band])) == pytest.approx(1, abs=0.02)


def test_heterodyne_carrier_is_the_beat_note_averaged_over_each_sample(off_node):
    A, θ_α = carrier_closed_form(off_node)
    Ωhet, φ0, Δt = 2 * np.pi * 1.3e6, 0.4, 0.1 * μs
    kw = {'num_samples': 64, 'seed': 2, 'Δt': Δt}
    on = off_node.simulate(detection=[Heterodyne(Ωhet=Ωhet, φ0=φ0, static_carrier=True)], **kw)
    off = off_node.simulate(detection=[Heterodyne(Ωhet=Ωhet, φ0=φ0)], **kw)
    dc = (on.records['heterodyne'] - off.records['heterodyne'])[0]
    t = np.arange(64)[:, None] * Δt + np.linspace(0, Δt, 2001)[None, :]
    quadrature = simpson(A * np.cos(Ωhet * t + φ0 - θ_α), x=t, axis=-1) / Δt
    np.testing.assert_allclose(dc, quadrature, rtol=0, atol=1e-9 * A)


def test_carrier_in_volts_and_with_an_explicit_field(off_node):
    A, θ_α = carrier_closed_form(off_node)
    volts = VoltageConversion(power=1e-3)
    kw = {'num_samples': 16, 'seed': 1}
    on = off_node.simulate(detection=[Homodyne(θ=θ_α, volts=volts, static_carrier=True)], **kw)
    off = off_node.simulate(detection=[Homodyne(θ=θ_α, volts=volts)], **kw)
    np.testing.assert_allclose(on.records['homodyne'] - off.records['homodyne'],
                               A * volts.volts_per_shot_noise_unit(), rtol=1e-12)
    # a QuadratureSystem has no static field of its own: α must be given
    from levisim.multiphysics.output_modes import QuadratureSystem
    system = QuadratureSystem.from_multitrap(off_node)
    with pytest.raises(ValueError, match='pass α='):
        CavitySimulation(system, num_samples=4, detection=[Homodyne(θ=0.1, static_carrier=True)])
    α = complex(off_node.cavity.αR + 1j * off_node.cavity.αI)
    a = CavitySimulation(system, num_samples=16, seed=1, detection=[Homodyne(θ=θ_α, static_carrier=True, α=α)])
    np.testing.assert_allclose(a.records['homodyne'].mean(), A, rtol=1e-3)
    with pytest.raises(ValueError, match='mean flux'):
        PhotonNumber(static_carrier=True)


# ----------------------------------------------------------------------------------------------
# instantaneous Position sampling (decision 21)
# ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize('backend', ['numpy', 'numba'])
def test_instantaneous_samples_are_the_state_at_the_sample_times(backend):
    if backend == 'numba' and not numba_available():
        pytest.skip('numba is not installed or is disabled')
    mtrap = shelf_multitrap(2, 2, φn_dict={2: 30 * degrees})
    kw = {'num_samples': 3000, 'seed': 9, 'num_trajectories': 3, 'backend': backend, 'store_state': True}
    base = [Homodyne(), Position('x1'), Position('y2', units='zpf')]
    plain = mtrap.simulate(detection=base, **kw)
    base = [Homodyne(), Position('x1'), Position('y2', units='zpf')]
    both = mtrap.simulate(detection=base + [Position('x1', sampling='instantaneous', trace_label='x1_inst'),
                                            Position('y2', units='zpf', sampling='instantaneous', trace_label='y2_inst')], **kw)
    assert both.backend_used == backend
    # the instantaneous records draw no random numbers: everything else is bit-identical
    for label in plain.records:
        np.testing.assert_array_equal(both.records[label], plain.records[label])
    np.testing.assert_array_equal(both.states, plain.states)
    ix = both.system.quadrature_indices('x1')[0]
    iy = both.system.quadrature_indices('y2')[0]
    np.testing.assert_array_equal(both.records['x1_inst'], mtrap.dimensions['x1'].zpf * both.states[:, :-1, ix])
    np.testing.assert_array_equal(both.records['y2_inst'], both.states[:, :-1, iy])


def test_instantaneous_variance_and_spectrum_against_closed_forms():
    mtrap = shelf_multitrap()
    Δt = 1 * μs           # coarse: the averaged record is visibly filtered by sinc²(fΔt)
    sim = mtrap.simulate(num_samples=2**15, Δt=Δt, seed=11, num_trajectories=16, zero_mean=False, detection=[
        Position('x1'), Position('x1', sampling='instantaneous', trace_label='x1_inst')])
    zpf = mtrap.dimensions['x1'].zpf
    i = sim.system.quadrature_indices('x1')[0]
    V = sim.system.steady_state_covariance()[i, i] * zpf**2
    # ∫ S df over the Nyquist band of the exact discrete spectrum is the variance of the samples
    f = np.linspace(-0.5 / Δt, 0.5 / Δt, 400_001)
    S_inst = sim.record_spectrum('x1_inst', f)
    assert simpson(S_inst, x=f) == pytest.approx(V, rel=1e-4)
    assert np.var(sim.records['x1_inst']) == pytest.approx(V, rel=0.05)
    # the averaged record is the filtered one: smaller variance
    assert simpson(sim.record_spectrum('x1', f), x=f) < 0.95 * V      # 0.92 V here
    # the simulated PSD follows the discrete theory, and the continuous theory has no sinc²
    fs, S = sim.averaged_psd('x1_inst', num_segments=8)
    band = (fs > 20e3) & (fs < 400e3)
    assert np.median(S[band] / sim.record_spectrum('x1_inst', fs[band])) == pytest.approx(1, abs=0.03)
    fc = np.array([150e3, 250e3])
    ω = 2 * np.pi * fc
    old = mtrap.ω_range
    mtrap.ω_range = np.concatenate([ω, -ω])
    try:
        S_mt = mtrap.position_power_spectrum('x1', LAM)
    finally:
        mtrap.ω_range = old
    symmetrised = (S_mt[:2] + S_mt[2:]) / 2 * zpf**2
    np.testing.assert_allclose(sim.record_spectrum('x1_inst', fc, kind='continuous'), symmetrised, rtol=1e-6)
    np.testing.assert_allclose(sim.record_spectrum('x1', fc, kind='continuous'),
                               symmetrised * np.sinc(fc * Δt)**2, rtol=1e-6)


def test_instantaneous_sampling_of_the_nonlinear_model_and_argument_checks():
    mtrap = shelf_multitrap()
    kw = {'model': 'nonlinear', 'num_samples': 500, 'seed': 3, 'num_trajectories': 2, 'backend': 'numpy'}
    stored = CavitySimulation(mtrap, store_state=True,
                              detection=[Homodyne(), Position('x1', sampling='instantaneous')], **kw)
    i = stored.system.quadrature_indices('x1')[0]
    np.testing.assert_array_equal(stored.records['x1'], stored.zero_point_fluctuation('x1') * stored.states[:, :-1, i])
    unstored = CavitySimulation(mtrap, detection=[Homodyne(), Position('x1', sampling='instantaneous')], **kw)
    assert unstored.states is None
    np.testing.assert_array_equal(unstored.records['x1'], stored.records['x1'])
    np.testing.assert_array_equal(unstored.records['homodyne'], stored.records['homodyne'])
    with pytest.raises(ValueError, match='sampling'):
        Position('x1', sampling='midpoint')
    assert 'instantaneous' in repr(Position('x1', sampling='instantaneous'))

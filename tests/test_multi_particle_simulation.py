"""
MultiParticleFreeSpaceSimulation: several free-space traps with pair forces
"""
import numpy as np
import pytest

from levisim.core.units import kHz, mbar, μm, μs
from levisim.physics.bead import Bead
from levisim.physics.gas import Gas
from levisim.physics.interactions import Coulomb, Interaction, NewtonianGravity
from levisim.physics.kick import Kick
from levisim.physics.trap import Trap
from levisim.simulation import FreeSpaceSimulation, MultiParticleFreeSpaceSimulation
from levisim.simulation.backends import numba_available

needs_numba = pytest.mark.skipif(not numba_available(), reason='numba is not installed (or disabled)')

Ω = 100*kHz


def harmonic_trap(position, charge=20):
    return Trap(
        bead=Bead(charge=charge), position=position, override_Ω_dict={'x': 1.3*Ω, 'y': Ω, 'z': 0.4*Ω},
        Γ_fbx=0, Γ_fby=0, gas=Gas(P=1e-12*mbar),
    )


@pytest.mark.parametrize('backend', ['numpy', pytest.param('numba', marks=needs_numba)])
@pytest.mark.parametrize('RK_method', ['vector_4', 'vector_2'])
def test_one_particle_is_bit_identical_to_free_space_simulation(backend, RK_method):
    kwargs = {'num_samples': 3000, 'seed': 11, 'RK_method': RK_method, 'backend': backend}
    multi = MultiParticleFreeSpaceSimulation(traps=[Trap(photon_recoil=True)], gravity=True, **kwargs)
    single = FreeSpaceSimulation(trap=Trap(photon_recoil=True), gravity=True, **kwargs)
    assert np.array_equal(multi.X, single.X)
    assert np.array_equal(multi.traces['x1'].values, single.x.values)


def test_two_particles_without_interactions_are_independent():
    traps = [Trap(), Trap(position=(0, 5*μm, 0))]
    multi = MultiParticleFreeSpaceSimulation(traps=traps, num_samples=500, seed=2, backend='numpy')
    assert multi.X.shape == (501, 12)
    assert list(multi.traces) == ['x1', 'y1', 'z1', 'x2', 'y2', 'z2']
    # the drift of particle 1 does not depend on particle 2's state
    X = multi.X[100].copy()
    drift = multi.drift(0.0, X)
    X[6:] += 1e-8
    assert np.array_equal(multi.drift(0.0, X)[:6], drift[:6])


def test_drift_adds_the_interaction_forces():
    traps = [harmonic_trap((0, 0, 0)), harmonic_trap((0, 4*μm, 0))]
    plain = MultiParticleFreeSpaceSimulation(traps=traps, num_samples=10, seed=1, backend='numpy')
    coupled = MultiParticleFreeSpaceSimulation(traps=traps, interactions=[Coulomb()], num_samples=10, seed=1, backend='numpy')
    X = np.zeros(12)
    X[1] = 1e-9
    F = Coulomb().forces(np.array([[0, 1e-9, 0], [0, 4*μm, 0]]), [trap.bead for trap in traps])
    difference = (coupled.drift(0.0, X) - plain.drift(0.0, X)).reshape(2, 6)
    np.testing.assert_allclose(difference[:, 3:], F, rtol=1e-12)
    assert np.all(difference[:, :3] == 0)


@needs_numba
@pytest.mark.parametrize('RK_method', ['vector_4', 'vector_2'])
def test_numba_agrees_with_numpy_to_round_off(RK_method):
    traps = [Trap(bead=Bead(charge=300)), Trap(bead=Bead(charge=-200), position=(0, 3*μm, 0)),
             Trap(bead=Bead(charge=100), position=(2*μm, 0, 1*μm))]
    kwargs = {'traps': traps, 'interactions': [Coulomb(), NewtonianGravity()], 'num_samples': 2000, 'seed': 5, 'RK_method': RK_method}
    numpy_run = MultiParticleFreeSpaceSimulation(backend='numpy', **kwargs)
    numba_run = MultiParticleFreeSpaceSimulation(backend='numba', **kwargs)
    assert numba_run.backend_used == 'numba'
    scale = np.abs(numpy_run.X).max(axis=0)
    assert np.all(np.abs(numba_run.X - numpy_run.X) <= 1e-9 * scale)


class StubInteraction(Interaction):
    def forces(self, positions, particles, t=0.0):
        return np.zeros((len(particles), 3))


@needs_numba
def test_numba_raises_and_auto_falls_back_for_kicks_and_other_interactions():
    from levisim.simulation.backends import (
        AUTO_NUMBA_MIN_STEPS,
        UnsupportedBackendError,
    )
    with pytest.raises(UnsupportedBackendError, match='interaction'):
        MultiParticleFreeSpaceSimulation(traps=[Trap(), Trap(position=(0, 3*μm, 0))],
                                         interactions=[StubInteraction()], num_samples=100, backend='numba')
    kicked = Trap(kicks=[Kick(start=1e-6, duration=1e-6)])
    with pytest.raises(UnsupportedBackendError, match='kicks'):
        MultiParticleFreeSpaceSimulation(traps=[kicked, Trap(position=(0, 3*μm, 0))], num_samples=100, backend='numba')
    sim = MultiParticleFreeSpaceSimulation(traps=[kicked, Trap(position=(0, 3*μm, 0))],
                                           num_samples=AUTO_NUMBA_MIN_STEPS, backend='auto')
    assert sim.backend_used == 'numpy'


def equilibrium(traps, interaction):
    """Equilibrium displacements of the particles in their harmonic traps, by fixed-point iteration"""
    positions = np.array([trap.position for trap in traps])
    particles = [trap.bead for trap in traps]
    stiffness = np.array([[trap.bead.m * trap.override_Ω_dict[axis]**2 for axis in 'xyz'] for trap in traps])
    δ = np.zeros_like(positions)
    for _ in range(60):
        δ = interaction.forces(positions + δ, particles) / stiffness
    return δ


def test_coulomb_normal_modes_and_beating_match_the_stiffness_blocks():
    """
    Two equal charged particles 4 μm apart along y: the stiffness blocks predict normal modes split by
    Δω, and a displacement of particle 1 alone is passed to particle 2 completely after π/Δω
    """
    traps = [harmonic_trap((0, 0, 0)), harmonic_trap((0, 4*μm, 0))]
    particles = [trap.bead for trap in traps]
    δ = equilibrium(traps, Coulomb())
    K = Coulomb().stiffness_blocks(np.array([t.position for t in traps]) + δ, particles)
    m = particles[0].m
    dynamical = np.array([[m*Ω**2 + K[0, 0, 1, 1], K[0, 1, 1, 1]], [K[1, 0, 1, 1], m*Ω**2 + K[1, 1, 1, 1]]]) / m
    ω_minus, ω_plus = np.sqrt(np.linalg.eigvalsh(dynamical))
    assert (ω_plus - ω_minus) / (2*np.pi) == pytest.approx(427.76, rel=1e-3)
    T_exchange = np.pi / (ω_plus - ω_minus)

    X0 = np.concatenate([δ[0], np.zeros(3), δ[1], np.zeros(3)])
    X0[1] += 1e-9
    Δt = 0.1*μs
    num_samples = int(1.6 * T_exchange / Δt)
    sim = MultiParticleFreeSpaceSimulation(traps=traps, interactions=[Coulomb()], X0=X0, num_samples=num_samples,
                                           Δt=Δt, seed=1, zero_mean=False)
    y1 = sim.traces['y1'].values - δ[0, 1]
    y2 = sim.traces['y2'].values - δ[1, 1]
    period = round(2*np.pi / Ω / Δt)
    starts = np.arange(0, num_samples - period, period)
    amplitude_1 = np.array([np.abs(y1[s:s + period]).max() for s in starts])
    amplitude_2 = np.array([np.abs(y2[s:s + period]).max() for s in starts])
    i = np.argmin(amplitude_1)
    assert starts[i] * Δt == pytest.approx(T_exchange, abs=2 * period * Δt)
    assert amplitude_1[i] < 0.03e-9
    assert amplitude_2[i] > 0.97e-9
    # the frequencies in the record are the predicted normal modes
    spectrum = np.abs(np.fft.rfft((y1 + y2) * np.hanning(len(y1))))
    frequencies = 2*np.pi * np.fft.rfftfreq(len(y1), Δt)
    assert frequencies[np.argmax(spectrum)] == pytest.approx(ω_minus, abs=2*np.pi / (num_samples * Δt))
    spectrum = np.abs(np.fft.rfft((y1 - y2) * np.hanning(len(y1))))
    assert frequencies[np.argmax(spectrum)] == pytest.approx(ω_plus, abs=2*np.pi / (num_samples * Δt))


def test_paul_traps_with_electrode_noise_and_coulomb():
    """Multi-particle runs accept per-particle noise blocks wider than 6 × 4 (a PaulTrap's electrode noise)"""
    import warnings

    from levisim.physics import Coulomb, Gas, PaulTrap, Sphere, Trap
    from levisim.simulation.multi_particle import MultiParticleFreeSpaceSimulation
    traps = []
    for X in (-150e-6, 150e-6):
        paul = PaulTrap.ring(V_rf=300.0, Ω_rf=2 * np.pi * 200e3, r0=0.5e-3, voltage_noise_psd=1e-14,
                             noise_distances=(1e-3, 1e-3, 1e-3))
        traps.append(Trap(fields=[paul], particle=Sphere(r=200e-9, charge=300), gas=Gas(P=0.1), position=(X, 0, 0)))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        sim = MultiParticleFreeSpaceSimulation(traps=traps, interactions=[Coulomb()], num_samples=2000, Δt=0.05e-6,
                                               seed=1, backend='numpy')
    assert sim.noise_matrix().shape[0] == 12
    assert sim.noise_matrix().shape[1] == len(sim.noise_spectral_densities()) > 8
    assert np.all(np.isfinite(sim.traces['x2'].values))

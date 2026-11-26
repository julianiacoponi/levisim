"""
Interactions between particles (levisim.physics.interactions) and their linearisation in MultiTrap
"""
import itertools

import numpy as np
import pytest

from levisim.core.constants import c, e, ħ, ε0, π
from levisim.core.units import kHz, nm, μm
from levisim.multiphysics.multitrap import MultiTrap
from levisim.physics.bead import Bead
from levisim.physics.interactions import (
    Casimir,
    Coulomb,
    G_newton,
    Interaction,
    NewtonianGravity,
    lifshitz_plate_factor,
)
from levisim.physics.particle import Sphere
from levisim.physics.trap import Trap

RNG = np.random.default_rng(7)


def particles_and_positions(charges=(300, -200, 500), radii=(60*nm, 80*nm, 100*nm)):
    particles = [Sphere(r=r, charge=q) for r, q in zip(radii, charges)]
    positions = np.array([[0.0, 0.0, 0.0], [3*μm, 1*μm, -2*μm], [-1*μm, 4*μm, 2.5*μm]])
    return particles, positions


def all_interactions():
    return [
        Coulomb(),
        NewtonianGravity(),
        Casimir(model='casimir_polder', validity_warning_ratio=0.0),
        Casimir(model='proximity_force', permittivity=3.8),
    ]


def close_configuration():
    """Two spheres 20 nm apart (for the proximity-force model)"""
    particles = [Sphere(r=1*μm), Sphere(r=0.8*μm)]
    positions = np.array([[0.0, 0.0, 0.0], [1.82*μm, 0.0, 0.0]])
    return particles, positions


def five_point(f, x, h):
    """Fourth-order central difference of f at x along the unit step h (an array like x)"""
    return (-f(x + 2*h) + 8*f(x + h) - 8*f(x - h) + f(x - 2*h)) / 12


def fd_step(interaction, positions):
    if isinstance(interaction, Casimir) and interaction.model == 'proximity_force':
        return 2e-10
    return 1e-3 * np.min(np.linalg.norm(positions[1:] - positions[0], axis=1))


def configuration_for(interaction):
    if isinstance(interaction, Casimir) and interaction.model == 'proximity_force':
        return close_configuration()
    return particles_and_positions()


@pytest.mark.parametrize('interaction', all_interactions(), ids=lambda i: repr(i))
def test_newtons_third_law(interaction):
    particles, positions = configuration_for(interaction)
    F = interaction.forces(positions, particles)
    assert F.shape == positions.shape
    np.testing.assert_allclose(F.sum(axis=0), 0, atol=1e-12 * np.abs(F).max())


@pytest.mark.parametrize('interaction', all_interactions(), ids=lambda i: repr(i))
def test_force_is_minus_the_gradient_of_the_potential(interaction):
    particles, positions = configuration_for(interaction)
    F = interaction.forces(positions, particles)
    h = fd_step(interaction, positions)
    gradient = np.zeros_like(positions)
    for n in range(len(particles)):
        for a in range(3):
            step = np.zeros_like(positions)
            step[n, a] = h
            gradient[n, a] = five_point(lambda x: interaction.potential(x, particles), positions, step) / h
    # the proximity-force energy differences lose ~7 digits to cancellation at a 20 nm gap
    tolerance = 1e-6 if getattr(interaction, 'model', '') == 'proximity_force' else 1e-8
    np.testing.assert_allclose(-gradient, F, rtol=tolerance, atol=tolerance * np.abs(F).max())


@pytest.mark.parametrize('interaction', all_interactions(), ids=lambda i: repr(i))
def test_analytic_stiffness_blocks_equal_finite_differences(interaction):
    particles, positions = configuration_for(interaction)
    K = interaction.stiffness_blocks(positions, particles)
    h = fd_step(interaction, positions)
    N = len(particles)
    K_fd = np.zeros_like(K)
    for k in range(N):
        for b in range(3):
            step = np.zeros_like(positions)
            step[k, b] = h
            K_fd[:, k, :, b] = -five_point(lambda x: interaction.forces(x, particles), positions, step) / h
    # the base class's default (second-order central differences) agrees to its truncation error
    np.testing.assert_allclose(Interaction.stiffness_blocks(interaction, positions, particles),
                               K, rtol=2e-3, atol=2e-3 * np.abs(K).max())
    assert K.shape == (len(particles), len(particles), 3, 3)
    tolerance = 1e-6 if getattr(interaction, 'model', '') == 'proximity_force' else 1e-7
    np.testing.assert_allclose(K, K_fd, rtol=tolerance, atol=tolerance * np.abs(K).max())
    # conservative and reciprocal: K[j, k] = K[k, j]ᵀ, and each row of blocks sums to zero
    np.testing.assert_allclose(K, np.transpose(K, (1, 0, 3, 2)), rtol=0, atol=1e-14 * np.abs(K).max())
    np.testing.assert_allclose(K.sum(axis=1), 0, atol=1e-12 * np.abs(K).max())


def test_coulomb_and_gravity_values():
    particles = [Sphere(r=100*nm, charge=1000), Sphere(r=100*nm, charge=-500)]
    positions = np.array([[0, 0, 0], [10*μm, 0, 0]], dtype=float)
    F = Coulomb().forces(positions, particles)
    expected = 1000 * 500 * e**2 / (4*π*ε0 * (10*μm)**2)  # attractive
    assert F[0, 0] == pytest.approx(expected, rel=1e-14)
    assert F[1, 0] == pytest.approx(-expected, rel=1e-14)
    F_g = NewtonianGravity().forces(positions, particles)
    assert F_g[0, 0] == pytest.approx(G_newton * particles[0].m**2 / (10*μm)**2, rel=1e-14)
    # along the line of centres a repulsive Coulomb pair stiffens: K_xx = U'' = 2kq1q2/r³ > 0
    K = Coulomb().stiffness_blocks(positions, [Sphere(r=100*nm, charge=10), Sphere(r=100*nm, charge=10)])
    assert K[0, 0, 0, 0] == pytest.approx(2 * 100 * e**2 / (4*π*ε0 * (10*μm)**3), rel=1e-14)
    assert K[0, 0, 1, 1] == pytest.approx(-K[0, 0, 0, 0] / 2, rel=1e-14)


def test_casimir_polder_and_proximity_force_values():
    R, ε = 1*μm, 5.7
    particles = [Sphere(r=R), Sphere(r=R)]
    r = 50*μm
    positions = np.array([[0, 0, 0], [r, 0, 0]], dtype=float)
    U = Casimir(permittivity=ε).potential(positions, particles)
    # van de Kamp et al., Phys. Rev. A 102, 062807 (2020), Eq. (4)
    assert U == pytest.approx(-23 * ħ * c / (4*π) * R**6 / r**7 * ((ε - 1)/(ε + 2))**2, rel=1e-14)
    # proximity force with ideal mirrors: Krisnanda et al. npj QI 6, 12 (2020), -π³ħcR/(1440 d²) with f0 = 1
    d = 30*nm
    positions = np.array([[0, 0, 0], [2*R + d, 0, 0]], dtype=float)
    U = Casimir(model='proximity_force', ideal_mirrors=True).potential(positions, particles)
    assert U == pytest.approx(-π**3 * ħ * c * R / (1440 * d**2), rel=1e-12)


def test_lifshitz_plate_factor_limits():
    # dilute limit: pairwise summation of the Casimir-Polder r⁻⁷ energy, -23ħc(ε-1)²/(1920π²a³)
    δ = 1e-4
    assert lifshitz_plate_factor(1 + δ) == pytest.approx(69 / (8 * π**4) * δ**2, rel=1e-3)
    # ideal mirrors, ε → ∞
    assert lifshitz_plate_factor(1e9) == pytest.approx(1.0, abs=2e-3)
    values = [lifshitz_plate_factor(ε) for ε in (2, 3.8, 10, 100)]
    assert all(0 < a < b < 1 for a, b in itertools.pairwise(values))


def test_overlap_and_contact_raise():
    particles = [Sphere(r=1*μm), Sphere(r=1*μm)]
    positions = np.array([[0, 0, 0], [1.5*μm, 0, 0]], dtype=float)
    for interaction in (NewtonianGravity(), Casimir(), Coulomb()):
        with pytest.raises(ValueError):
            interaction.forces(positions, particles)


def test_pairs_restrict_the_interaction():
    particles, positions = particles_and_positions()
    F = Coulomb(pairs=[(0, 1)]).forces(positions, particles)
    assert np.all(F[2] == 0)
    with pytest.raises(ValueError):
        Coulomb(pairs=[(1, 1)])


# --------------------------------------------------------------------------------------------------
# MultiTrap
# --------------------------------------------------------------------------------------------------
def charged_pair(axes=2, separation=(0, 10*μm, 0), charges=(500, 800), **kwargs):
    beads = [Bead(r=60.1*nm, Φ=0.0, charge=charges[0]), Bead(r=60.2*nm, Φ=0.0, charge=charges[1])]
    kwargs.setdefault('static_cavity_shifts', False)
    kwargs.setdefault('photon_recoil', False)
    return MultiTrap(
        num_traps=2, Cartesian_axes_per_trap=axes, beads=beads, trap_positions=[(0, 0, 0), separation], **kwargs,
    )


def test_trap_position_defaults_and_keyword():
    assert np.array_equal(Trap().position, np.zeros(3))
    assert np.array_equal(Trap(position=(1, 2, 3)).position, [1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        Trap(position=(1, 2))


def test_trap_positions_and_trap_kwargs_position_clash():
    with pytest.raises(ValueError, match='both give a position'):
        MultiTrap(num_traps=2, trap_positions=[(0, 0, 0), (0, 1e-5, 0)], trap_kwargs=[{'position': (0, 0, 0)}, {}])
    mtrap = MultiTrap(trap_positions=[(0, 0, 0), (0, 1e-5, 0), (0, 2e-5, 0)])
    assert mtrap.num_traps == 3
    assert np.array_equal(mtrap.traps[2].position, [0, 2e-5, 0])


def test_without_interactions_positions_change_nothing():
    plain = MultiTrap(num_traps=2, static_cavity_shifts=False)
    placed = MultiTrap(num_traps=2, static_cavity_shifts=False, trap_positions=[(0, 0, 0), (0, 3*μm, 0)])
    assert np.array_equal(plain.drift_matrix, placed.drift_matrix)


def captured_warnings(build):
    from levisim.core.logger import logger
    captured = []
    handler_id = logger.add(lambda message: captured.append(message.record['message']), level='WARNING', format='{message}')
    try:
        build()
    finally:
        logger.remove(handler_id)
    return ' '.join(captured)


def test_phase_mismatch_warning():
    k = MultiTrap(num_traps=1).cavity.k
    θ = 45 * π / 180
    d = np.array([np.sin(θ), np.cos(θ), 0])
    # trap 2 a quarter wavelength further along the cavity axis: Φ2 - Φ1 = π/2 is consistent
    X2 = d * (π/2) / k
    beads = [Bead(r=60.1*nm, Φ=0.3), Bead(r=60.2*nm, Φ=0.3 + π/2)]
    text = captured_warnings(lambda: MultiTrap(num_traps=2, beads=beads, trap_positions=[(0, 0, 0), X2]))
    assert 'disagree with the beads' not in text
    beads = [Bead(r=60.1*nm, Φ=0.3), Bead(r=60.2*nm, Φ=0.3)]
    text = captured_warnings(lambda: MultiTrap(num_traps=2, beads=beads, trap_positions=[(0, 0, 0), X2]))
    assert 'disagree with the beads' in text


def test_coupling_rate_conversion_and_frequency_shifts():
    plain = charged_pair()
    mtrap = charged_pair(interactions=[Coulomb()])
    K = mtrap.interaction_stiffness_blocks
    for label in mtrap.labels:
        dim, bare = mtrap.dimensions[label], plain.dimensions[label]
        a = 'xyz'.index(dim.axis)
        n = dim.trap_index - 1
        expected = np.sqrt(bare.Ω**2 + K[n, n, a, a] / dim.m)
        assert dim.Ω == pytest.approx(expected, rel=1e-13)
        assert dim.zpf == pytest.approx(np.sqrt(ħ / (2 * dim.m * dim.Ω)), rel=1e-14)
    G = mtrap.mechanical_coupling_matrix
    for jj, lj in enumerate(mtrap.labels):
        for kk, lk in enumerate(mtrap.labels):
            dj, dk = mtrap.dimensions[lj], mtrap.dimensions[lk]
            Kjk = K[dj.trap_index - 1, dk.trap_index - 1, 'xyz'.index(dj.axis), 'xyz'.index(dk.axis)]
            if jj != kk and dj.trap_index != dk.trap_index:
                assert G[jj, kk] == pytest.approx(-Kjk * dj.zpf * dk.zpf / ħ, rel=1e-14)
    # repulsion along y (the separation) stiffens y, softens x
    assert mtrap.dimensions['y1'].Ω > plain.dimensions['y1'].Ω
    assert mtrap.dimensions['x1'].Ω < plain.dimensions['x1'].Ω


def test_normal_modes_of_the_drift_matrix_match_the_stiffness_blocks():
    """With no cavity coupling (beads at the antinode, x and y only) the drift-matrix frequencies
    are those of the classical dynamical matrix M^(-1/2)·(K_tw + K_int)·M^(-1/2)"""
    mtrap = charged_pair(interactions=[Coulomb()], gas_kwargs={'P': 1e-9})
    plain = charged_pair(gas_kwargs={'P': 1e-9})
    assert np.all(mtrap.G_array == 0)
    labels = mtrap.labels
    K_total = np.zeros((4, 4))
    masses = np.array([mtrap.dimensions[label].m for label in labels])
    for jj, lj in enumerate(labels):
        dj = mtrap.dimensions[lj]
        K_total[jj, jj] = dj.trap.bead.m * dj.trap.Ωj[dj.axis]**2
        for kk, lk in enumerate(labels):
            dk = mtrap.dimensions[lk]
            K_total[jj, kk] += mtrap.interaction_stiffness_blocks[
                dj.trap_index - 1, dk.trap_index - 1, 'xyz'.index(dj.axis), 'xyz'.index(dk.axis)]
            if jj != kk and dj.trap_index == dk.trap_index:
                # the cavity's own static x-y coupling at the antinode (A0·cos Φ), as a stiffness
                pj, pk = plain.dimensions[lj], plain.dimensions[lk]
                K_total[jj, kk] += -plain.mechanical_coupling_matrix[jj, kk] * ħ / (pj.zpf * pk.zpf)
    dynamical = K_total / np.sqrt(np.outer(masses, masses))
    expected = np.sort(np.sqrt(np.linalg.eigvalsh(dynamical)))
    λ = np.linalg.eigvals(mtrap.drift_matrix[2:, 2:])
    found = np.sort(np.abs(λ.imag))[::2]
    np.testing.assert_allclose(found, expected, rtol=1e-9)


class NonReciprocalStub(Interaction):
    """A linear, non-reciprocal force F_0 = -κ01·(r_1 - X1), F_1 = 0 (a test stub)"""
    is_conservative = False
    is_reciprocal = False

    def __init__(self, κ01):
        super().__init__()
        self.κ01 = κ01

    def forces(self, positions, particles, t=0.0):
        F = np.zeros((len(particles), 3))
        F[0] = -self.κ01 * (positions[1] - self.reference)
        return F


def test_non_reciprocal_interactions_are_kept_apart():
    stub = NonReciprocalStub(κ01=1e-9)
    stub.reference = np.array([0, 10*μm, 0])
    mtrap = charged_pair(charges=(0, 0), interactions=[stub])
    G = mtrap.mechanical_coupling_matrix
    i_x1, i_x2 = mtrap.label_to_index['x1'], mtrap.label_to_index['x2']
    assert G[i_x1, i_x2] != 0
    assert G[i_x2, i_x1] == 0
    # K_01 = -∂F_0/∂r_1 = +κ01, so g = -K·x_zpf,1·x_zpf,2/ħ
    assert G[i_x1, i_x2] == pytest.approx(
        -1e-9 * mtrap.dimensions['x1'].zpf * mtrap.dimensions['x2'].zpf / ħ, rel=1e-6)


def test_toy_g_direct():
    from levisim.multiphysics.utils.shelf import OFF_NODE_toy_params_pairs
    params = dict(OFF_NODE_toy_params_pairs, g_direct={('x1', 'x2'): (3*kHz, -1*kHz), ('y1', 'y2'): 2*kHz})
    mtrap = MultiTrap(num_traps=2, toy_params=params)
    base = MultiTrap(num_traps=2, toy_params=OFF_NODE_toy_params_pairs)
    difference = mtrap.mechanical_coupling_matrix - base.mechanical_coupling_matrix
    index = mtrap.label_to_index
    assert difference[index['x1'], index['x2']] == 3*kHz
    assert difference[index['x2'], index['x1']] == -1*kHz
    assert difference[index['y1'], index['y2']] == difference[index['y2'], index['y1']] == 2*kHz
    assert np.count_nonzero(difference) == 4
    with pytest.raises(ValueError):
        MultiTrap(num_traps=2, toy_params=dict(OFF_NODE_toy_params_pairs, g_direct={('x1', 'q9'): 1.0}))
    with pytest.raises(ValueError, match='physical parameters'):
        MultiTrap(num_traps=2, toy_params=OFF_NODE_toy_params_pairs, interactions=[Coulomb()])

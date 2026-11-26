"""
Tests of levisim.physics.interactions.optical_binding: the dipole Green tensor, the tweezer field,
the OpticalBinding forces and stiffness blocks, and their use in MultiTrap and the time domain
"""

import numpy as np
import pytest

from levisim.core.constants import ħ, ε0, π
from levisim.core.units import mbar, mW, nm, μm
from levisim.multiphysics.covariance import quadrature_matrices
from levisim.multiphysics.multitrap import MultiTrap
from levisim.physics import Bead, Gas, OpticalBinding, Spheroid, Trap, Tweezer
from levisim.physics.interactions import (
    dipole_field,
    dipole_green_tensor,
    dipole_green_tensor_derivatives,
    tweezer_field,
)
from levisim.simulation import MultiParticleFreeSpaceSimulation

k_1064 = 2 * π / (1064 * nm)


def pair(d0, φ1=0.0, φ2=0.0, axis=(1, 0, 0), P=(400*mW, 400*mW), w0=730*nm, r=105*nm, λ2=1064*nm, particles=None):
    """Two traps (Rieser et al. 2022 parameters by default) with foci d0 apart along `axis`"""
    particles = particles or [Bead(r=r), Bead(r=r)]
    return [
        Trap(bead=particles[0], tweezer=Tweezer(Power=P[0], wx0=w0, wy0=w0, φ=φ1)),
        Trap(bead=particles[1], tweezer=Tweezer(Power=P[1], wx0=w0, wy0=w0, φ=φ2, λ=λ2),
             position=tuple(d0 * np.asarray(axis, dtype=float))),
    ]


def at_foci(traps):
    return np.array([trap.position for trap in traps]), [trap.bead for trap in traps]


def far_field_constants(traps):
    """Rieser et al. (2022) Eq. (S3): (k1, k2) = Gm·(cos kd cos Δφ, sin kd sin Δφ)/kd, with the
    Gouy-reduced wavenumber k' = k - 1/zR in the phase of each tweezer (Rudolph et al. 2024, Eq. (A4))"""
    t1, t2 = traps[0].tweezer, traps[1].tweezer
    α = np.real(t1.effective_polarisability(traps[0].bead))
    k = t1.k
    kp = k - 1 / t1.zR
    d = np.linalg.norm(traps[1].position - traps[0].position)
    Gm = α**2 * k**3 * kp**2 * t1.E0 * t2.E0 / (8 * π * ε0)
    Δφ = t1.φ - t2.φ
    return Gm * np.cos(k * d) * np.cos(Δφ) / (k * d), Gm * np.sin(k * d) * np.sin(Δφ) / (k * d), Gm / (k * d)


# --------------------------------------------------------------------------------------------------
# Green tensor and tweezer field
# --------------------------------------------------------------------------------------------------
def test_green_tensor_is_the_dipole_field():
    rng = np.random.default_rng(3)
    for scale in (50*nm, 1*μm, 20*μm):
        r = rng.normal(size=3) * scale
        p = rng.normal(size=3) + 1j * rng.normal(size=3)
        E = dipole_field(r + 1e-6, np.full(3, 1e-6), p, k_1064)
        np.testing.assert_allclose(dipole_green_tensor(r, k_1064) @ p, E, rtol=1e-13, atol=1e-13 * np.abs(E).max())
        G = dipole_green_tensor(r, k_1064)
        np.testing.assert_allclose(G, G.T, rtol=0, atol=1e-15 * np.abs(G).max())
        np.testing.assert_allclose(dipole_green_tensor(-r, k_1064), G, rtol=1e-15)


def test_green_tensor_far_field_is_transverse_and_radiates_the_dipole_power():
    # far from the dipole the field is k²e^{ikr}/(4πε0 r)·(𝟙 - r̂r̂)·p
    r = np.array([0.3, -0.4, 1.2]) * 1e-2
    G = dipole_green_tensor(r, k_1064)
    n = r / np.linalg.norm(r)
    far = k_1064**2 * np.exp(1j * k_1064 * np.linalg.norm(r)) / (4 * π * ε0 * np.linalg.norm(r)) * (np.eye(3) - np.outer(n, n))
    np.testing.assert_allclose(G, far, rtol=0, atol=1e-4 * np.abs(far).max())
    # the imaginary part at r → 0 is the radiation-reaction term k³/(6πε0)·𝟙
    small = dipole_green_tensor(np.array([1e-10, 0, 0]), k_1064)
    np.testing.assert_allclose(np.imag(small), k_1064**3 / (6 * π * ε0) * np.eye(3), rtol=1e-5)


def test_green_tensor_derivatives_match_finite_differences():
    r = np.array([1.3, -0.4, 0.7]) * μm
    _, dG, d2G = dipole_green_tensor_derivatives(r, k_1064)
    h = 1e-12
    for a in range(3):
        e = np.zeros(3)
        e[a] = h
        G_p, dG_p = dipole_green_tensor_derivatives(r + e, k_1064, order=1)
        G_m, dG_m = dipole_green_tensor_derivatives(r - e, k_1064, order=1)
        np.testing.assert_allclose((G_p - G_m) / (2*h), dG[a], rtol=0, atol=1e-8 * np.abs(dG).max())
        np.testing.assert_allclose((dG_p - dG_m) / (2*h), d2G[:, a], rtol=0, atol=1e-8 * np.abs(d2G).max())
    np.testing.assert_allclose(d2G, d2G.transpose(1, 0, 2, 3), rtol=0, atol=1e-15 * np.abs(d2G).max())


def test_tweezer_field_amplitude_phase_and_derivatives():
    tweezer = Tweezer(wx0=0.8*μm, wy0=1.1*μm, φ=0.7)
    for s in ([0, 0, 0], [0.2*μm, -0.15*μm, 0.4*μm], [0, 0, -1.3*μm]):
        s = np.asarray(s, dtype=float)
        E, dE, d2E = tweezer_field(tweezer, s)
        assert np.abs(E) @ [0, 1, 0] == pytest.approx(tweezer.E_field_amplitude(*s), rel=1e-14)
        assert E[0] == 0 and E[2] == 0
        if s[0] == 0 and s[1] == 0:
            # on axis the phase is levisim's k·z - arctan(z/zR), plus the tweezer phase φ
            expected = tweezer.k * s[2] - np.arctan(s[2] / tweezer.zR) + tweezer.φ
            assert np.angle(E[1] * np.exp(-1j * expected)) == pytest.approx(0, abs=1e-12)
        for a in range(3):
            e = np.zeros(3)
            e[a] = 1e-12
            E_p, dE_p = tweezer_field(tweezer, s + e, order=1)
            E_m, dE_m = tweezer_field(tweezer, s - e, order=1)
            np.testing.assert_allclose((E_p - E_m) / 2e-12, dE[a], rtol=0, atol=1e-8 * np.abs(dE).max())
            np.testing.assert_allclose((dE_p - dE_m) / 2e-12, d2E[:, a], rtol=0, atol=1e-7 * np.abs(d2E).max())
    # the Gouy phase lowers the axial phase gradient at the focus to k - 1/zR
    _, dE = tweezer_field(tweezer, np.zeros(3), order=1)
    assert np.imag(dE[2, 1] / tweezer_field(tweezer, np.zeros(3), order=0)[0][1]) == pytest.approx(
        tweezer.k - 1 / tweezer.zR, rel=1e-14)
    E, dE = tweezer_field(tweezer, np.array([0.3*μm, 0, 0.2*μm]), order=1, model='plane_wave')
    assert abs(E[1]) == pytest.approx(tweezer.E0) and dE[2, 1] == pytest.approx(1j * tweezer.k * E[1])
    with pytest.raises(ValueError):
        tweezer_field(tweezer, np.zeros(3), model='bessel')


# --------------------------------------------------------------------------------------------------
# forces and stiffness
# --------------------------------------------------------------------------------------------------
def test_requires_traps_and_validates_options():
    with pytest.raises(ValueError):
        OpticalBinding(order='second')
    with pytest.raises(ValueError):
        OpticalBinding(tweezer_field='airy')
    with pytest.raises(ValueError):
        OpticalBinding().forces(np.zeros((2, 3)), [Bead(), Bead()])
    traps = pair(3*μm)
    with pytest.raises(ValueError):
        OpticalBinding(traps=traps).forces(np.zeros((3, 3)), [Bead()] * 3)
    with pytest.raises(ValueError):
        OpticalBinding(traps=traps).attach_traps(traps + [Trap()])


@pytest.mark.parametrize('order', ['first', 'self_consistent'])
def test_forces_are_not_equal_and_opposite(order):
    traps = pair(3.15*μm, φ1=0.3)
    positions, particles = at_foci(traps)
    F = OpticalBinding(traps=traps, order=order).forces(positions, particles)
    assert np.abs(F).max() > 1e-15
    assert np.abs(F.sum(axis=0)).max() > 0.1 * np.abs(F).max()
    # in the plane of polarisation symmetry (y = 0) there is no y force
    assert np.all(F[:, 1] == 0)


@pytest.mark.parametrize('geometry', ['x', 'oblique', 'three'])
def test_analytic_stiffness_matches_finite_differences(geometry):
    if geometry == 'three':
        traps = pair(2.5*μm, φ1=0.4, P=(300*mW, 450*mW)) + [
            Trap(bead=Bead(r=80*nm), tweezer=Tweezer(Power=350*mW, φ=1.1), position=(1*μm, 3*μm, 0.2*μm))]
    elif geometry == 'oblique':
        traps = pair(2.2*μm, φ1=-0.8, axis=(0.6, 0.8, 0.0), P=(300*mW, 500*mW))
    else:
        traps = pair(3.15*μm, φ1=0.3)
    positions, particles = at_foci(traps)
    positions = positions + np.random.default_rng(1).normal(size=positions.shape) * 30*nm
    binding = OpticalBinding(traps=traps)
    K_analytic = binding.stiffness_blocks(positions, particles)
    K_numerical = binding.stiffness_blocks(positions, particles, method='finite_difference')
    np.testing.assert_allclose(K_analytic, K_numerical, rtol=0, atol=1e-9 * np.abs(K_analytic).max())


def test_tensor_polarisability_of_a_tilted_spheroid():
    rod = Spheroid(150*nm, 60*nm, orientation=(np.cos(0.3), np.sin(0.3), 0.0, 0.0))
    traps = pair(2.8*μm, φ1=0.5, particles=[rod, Bead(r=90*nm)])
    positions, particles = at_foci(traps)
    binding = OpticalBinding(traps=traps)
    α = binding.polarisability(0, rod)
    assert np.abs(α - np.diag(np.diag(α))).max() > 1e-3 * np.abs(α).max()
    p = binding.induced_dipoles(positions, particles)
    assert abs(p[0, 2]) > 1e-3 * abs(p[0, 1])      # the dipole leaves the polarisation direction
    K = binding.stiffness_blocks(positions, particles)
    np.testing.assert_allclose(K, binding.stiffness_blocks(positions, particles, method='finite_difference'),
                               rtol=0, atol=1e-9 * np.abs(K).max())


def test_first_order_and_self_consistent_agree_at_large_separation():
    differences = []
    separations = np.array([2, 4, 8, 16]) * μm
    for d0 in separations:
        traps = pair(d0, φ1=0.25)
        positions, particles = at_foci(traps)
        first = OpticalBinding(traps=traps).stiffness_blocks(positions, particles)
        exact = OpticalBinding(traps=traps, order='self_consistent').stiffness_blocks(positions, particles)
        zz_first = np.array([first[0, 1, 2, 2], first[1, 0, 2, 2]])
        zz_exact = np.array([exact[0, 1, 2, 2], exact[1, 0, 2, 2]])
        differences.append(np.abs(zz_exact - zz_first).max() / np.abs(zz_first).max())
    tweezer, bead = traps[0].tweezer, traps[0].bead
    scattering_strength = abs(k_1064**3 * tweezer.effective_polarisability(bead) / (4 * π * ε0))
    differences = np.array(differences)
    # multiple scattering is of relative order |k³α/4πε0|/(kd)
    assert np.all(differences < 3 * scattering_strength / (k_1064 * separations))
    assert differences[-1] < differences[0] / 3


@pytest.mark.parametrize('field', ['gaussian', 'plane_wave'])
def test_far_field_coupling_is_rieser_eq_s3(field):
    # The zz stiffness tends to Rieser et al.'s Eq. (S3), K = [[k1 + k2, -(k1 + k2)], [-(k1 - k2), k1 - k2]],
    # up to O(1/kd), and O(Im α/Re α) from the radiation reaction (a 20 nm sphere keeps it at 3e-4).
    # Gaussian tweezers: the Gouy phase gives k' = k - 1/zR (Rudolph et al. 2024, Eq. (A4)), and on the
    # diagonal only the axial fall of the amplitude adds 1/(k'·zR)² to the curvature.
    for Δφ in (0.0, 0.6, π/2 + 0.2):
        for d0 in (64*μm, 1024*μm):
            traps = pair(d0, φ1=Δφ, r=20*nm)
            K = OpticalBinding(traps=traps, tweezer_field=field).stiffness_blocks(*at_foci(traps))[:, :, 2, 2]
            k1, k2, envelope = far_field_constants(traps)
            tweezer = traps[0].tweezer
            kp = tweezer.k - 1 / tweezer.zR
            if field == 'plane_wave':
                scale, diagonal = (tweezer.k / kp)**2, 1.0
            else:
                scale, diagonal = 1.0, 1 + 1 / (kp * tweezer.zR)**2
            k1, k2, envelope = scale * k1, scale * k2, scale * envelope
            expected = np.array([[(k1 + k2) * diagonal, -(k1 + k2)], [-(k1 - k2), (k1 - k2) * diagonal]])
            assert np.abs(K - expected).max() / envelope < 2e-3


def test_first_order_force_is_the_gradient_of_rieser_eq_s2():
    # independent check with Jackson's dipole field: for real polarisabilities the first-order force
    # is F_n = ∇_n Re[α_n·α_m/(2ε0)·E_n*(r_n)·G(r_n - r_m)·E_m(r_m)] (Rieser et al. 2022, Eq. (S2),
    # whose G·p/ε0 is Jackson's field of p). A 30 nm sphere keeps Im α/Re α (not in S2) at 1e-3.
    traps = pair(2.3*μm, φ1=0.7, axis=(0.8, 0.6, 0), r=30*nm, P=(300*mW, 500*mW))
    positions, particles = at_foci(traps)
    positions = positions + np.array([[10, -20, 30], [-15, 5, 25]]) * nm
    α = [np.real(trap.tweezer.effective_polarisability(trap.bead)) for trap in traps]

    def u(n, r):
        return tweezer_field(traps[n].tweezer, r - traps[n].position, order=0)[0]

    def W(n, r_n, r_m):
        m = 1 - n
        return np.real(α[n] * α[m] / 2 * np.conj(u(n, r_n)) @ dipole_field(r_n, r_m, u(m, r_m), k_1064))

    h = 1e-10
    F = np.zeros((2, 3))
    for n in range(2):
        for a in range(3):
            e = np.zeros(3)
            e[a] = h
            F[n, a] = (W(n, positions[n] + e, positions[1 - n]) - W(n, positions[n] - e, positions[1 - n])) / (2*h)
    np.testing.assert_allclose(OpticalBinding(traps=traps).forces(positions, particles), F, rtol=0,
                               atol=5e-3 * np.abs(F).max())


def test_reciprocal_and_antireciprocal_parts():
    traps = pair(3.0*μm, φ1=0.0)
    positions, particles = at_foci(traps)
    parts = OpticalBinding(traps=traps).coupling_decomposition(positions, particles)
    np.testing.assert_allclose(parts['K'], parts['reciprocal'] + parts['antireciprocal'])
    # equal phases and identical particles: the zz coupling is reciprocal
    z1, z2 = 2, 5
    assert parts['antireciprocal'][z1, z2] == pytest.approx(0, abs=1e-12 * abs(parts['K'][z1, z2]))
    traps = pair(3.0*μm, φ1=π/2)
    parts = OpticalBinding(traps=traps).coupling_decomposition(*at_foci(traps))
    _, k2, _ = far_field_constants(traps)
    # Rieser Eq. (1): K_12 = -(k1 + k2), K_21 = -(k1 - k2): anti-reciprocal part -k2 on [z1, z2]
    assert parts['antireciprocal'][z1, z2] == pytest.approx(-k2, rel=0.15)
    assert parts['K'][z1, z1] - parts['K'][z2, z2] == pytest.approx(2 * k2, rel=0.15)


def test_zero_far_field_along_the_polarisation():
    # a dipole along y radiates no far field along y: separated along y, the zz coupling has no 1/d
    # term and falls as 1/d² (envelope), while separated along x it falls as 1/d
    def envelope(axis, d0):
        values = []
        for shift in np.linspace(0, 1064*nm, 9, endpoint=False):
            traps = pair(d0 + shift, axis=axis)
            K = OpticalBinding(traps=traps).stiffness_blocks(*at_foci(traps))
            values.append(abs(K[0, 1, 2, 2]))
        return max(values)
    ratio_x = envelope((1, 0, 0), 16*μm) / envelope((1, 0, 0), 4*μm)
    ratio_y = envelope((0, 1, 0), 16*μm) / envelope((0, 1, 0), 4*μm)
    assert ratio_x == pytest.approx(1/4, rel=0.15)
    assert ratio_y == pytest.approx(1/16, rel=0.25)
    # Rieser et al. (2022), Eq. (S9): at d0 = 3λ the residual coupling for polarisation along the
    # separation is estimated as 2/(kd)² = 5.6e-3 of the coupling at Θ = 0 (radial near field only);
    # the full model stays below that estimate
    d0 = 3 * 1064*nm
    for field in ('plane_wave', 'gaussian'):
        K_x = OpticalBinding(traps=pair(d0), tweezer_field=field).stiffness_blocks(*at_foci(pair(d0)))
        traps_y = pair(d0, axis=(0, 1, 0))
        K_y = OpticalBinding(traps=traps_y, tweezer_field=field).stiffness_blocks(*at_foci(traps_y))
        assert abs(K_y[0, 1, 2, 2] / K_x[0, 1, 2, 2]) < 2 / (6*π)**2


def test_mismatched_wavelengths_do_not_bind():
    from levisim.core.logger import logger
    messages = []
    handler = logger.add(lambda m: messages.append(m.record['message']), level='WARNING', format='{message}')
    try:
        traps = pair(3*μm, λ2=976*nm)
        F = OpticalBinding(traps=traps).forces(*at_foci(traps))
        K = OpticalBinding(traps=traps).stiffness_blocks(*at_foci(traps))
    finally:
        logger.remove(handler)
    assert np.all(F == 0) and np.all(K == 0)
    assert any('different wavelengths' in message for message in messages)


def test_momentum_balance_of_two_radiating_dipoles():
    # two fixed dipoles: the total force on them is minus the momentum per unit time that their
    # radiation carries away, which is the far-field interference term (each dipole alone radiates
    # symmetrically). This is the part of F1 + F2 ≠ 0 that is not taken from the tweezer beams.
    k = k_1064
    r1, r2 = np.zeros(3), np.array([1.7, 0.4, -0.3]) * μm
    p1 = np.array([0.2, 1.0, 0.1j]) * 1e-25
    p2 = np.array([0.0, 0.7 * np.exp(0.9j), 0.3]) * 1e-25
    dG = dipole_green_tensor_derivatives(r1 - r2, k, order=1)[1]
    F1 = 0.5 * np.real(np.conj(p1) @ dG @ p2)
    F2 = 0.5 * np.real(np.conj(p2) @ (-dG) @ p1)
    # far-field momentum flux, (ε0/2)·∮|E|² n r² dΩ, cross term only
    x, w = np.polynomial.legendre.leggauss(200)
    φ = np.linspace(0, 2*π, 400, endpoint=False)
    θ = np.arccos(x)
    n = np.stack([np.outer(np.sin(θ), np.cos(φ)), np.outer(np.sin(θ), np.sin(φ)), np.outer(np.cos(θ), np.ones_like(φ))], -1)

    def far(p, r):
        transverse = p - n * (n @ p)[..., None]
        return k**2 / (4*π*ε0) * np.exp(-1j * k * (n @ r))[..., None] * transverse

    cross = np.real(np.sum(np.conj(far(p1, r1)) * far(p2, r2), axis=-1))
    weights = np.outer(w, np.full(φ.size, 2*π / φ.size))
    P_dot = ε0 * np.einsum('ij,ijk->k', weights * cross, n)
    np.testing.assert_allclose(F1 + F2, -P_dot, rtol=0, atol=1e-9 * np.abs(F1).max())


# --------------------------------------------------------------------------------------------------
# hosts
# --------------------------------------------------------------------------------------------------
def rieser_mtrap(d0, Δφ, η=0.0, **kwargs):
    tweezers = [Tweezer(Power=400*mW * (1 + η), wx0=730*nm, wy0=730*nm),
                Tweezer(Power=400*mW * (1 - η), wx0=730*nm, wy0=730*nm)]
    return MultiTrap(num_traps=2, Cartesian_axes_per_trap=3, tweezers=tweezers,
                     beads=[Bead(r=105*nm, Φ=π/2), Bead(r=105*nm, Φ=π/2)],
                     trap_positions=[(0, 0, 0), (d0, 0, 0)], φn_dict={1: Δφ, 2: 0.0},
                     interactions=[OpticalBinding(**kwargs)], static_cavity_shifts=False)


def test_multitrap_attaches_its_traps_and_converts_the_stiffness():
    mtrap = rieser_mtrap(3.15*μm, 0.7)
    binding = mtrap.interactions[0]
    assert binding.traps == mtrap.traps
    K = binding.stiffness_blocks(mtrap.interaction_positions, [trap.bead for trap in mtrap.traps])
    np.testing.assert_array_equal(K, mtrap.interaction_stiffness_blocks)
    G = mtrap.mechanical_coupling_matrix
    labels = mtrap.labels
    for j, k in (('z1', 'z2'), ('z2', 'z1'), ('x1', 'x2'), ('x2', 'z1')):
        dj, dk = mtrap.dimensions[j], mtrap.dimensions[k]
        Kjk = K[dj.trap_index - 1, dk.trap_index - 1, 'xyz'.index(dj.axis), 'xyz'.index(dk.axis)]
        assert G[labels.index(j), labels.index(k)] == pytest.approx(-Kjk * dj.zpf * dk.zpf / ħ, rel=1e-13)
    # non-reciprocal: g_12 ≠ g_21
    assert G[labels.index('z1'), labels.index('z2')] != pytest.approx(G[labels.index('z2'), labels.index('z1')], rel=1e-3)


def test_rieser_normal_mode_splitting_at_equal_frequencies():
    # Δφ = 0 at d0 = 3.15 μm: conservative coupling, the CoM mode keeps Ω and the breathing mode
    # moves by ≈ 2g (Rieser et al. 2022, Eq. (S7))
    mtrap = rieser_mtrap(3.15*μm, 0.0)
    labels = mtrap.labels
    g = mtrap.mechanical_coupling_matrix[labels.index('z1'), labels.index('z2')]
    A, _ = quadrature_matrices(mtrap, cavity=False)
    index = [2 * labels.index('z1'), 2 * labels.index('z1') + 1, 2 * labels.index('z2'), 2 * labels.index('z2') + 1]
    frequencies = np.sort(np.abs(np.linalg.eigvals(A[np.ix_(index, index)]).imag))[::2]
    Ω = mtrap.dimensions['z1'].Ω
    assert frequencies[1] - frequencies[0] == pytest.approx(2 * g, rel=0.05)
    assert g / Ω == pytest.approx(0.0876, abs=0.002)


def test_multi_particle_simulation_adds_the_binding_forces():
    traps = pair(3*μm, φ1=0.4)
    for trap in traps:
        trap.gas = Gas(P=1e-6*mbar)
    plain = MultiParticleFreeSpaceSimulation(traps=traps, num_samples=10, seed=1, backend='auto')
    binding = OpticalBinding()
    coupled = MultiParticleFreeSpaceSimulation(traps=traps, interactions=[binding], num_samples=10, seed=1, backend='auto')
    assert coupled.backend_used == 'numpy' and binding.traps == traps
    X = np.zeros(12)
    X[2], X[8] = 20*nm, -10*nm
    F = binding.forces(coupled.positions(X), [trap.bead for trap in traps])
    difference = (coupled.drift(0.0, X) - plain.drift(0.0, X)).reshape(2, 6)
    np.testing.assert_allclose(difference[:, 3:], F, rtol=1e-12)


def test_nonlinear_cavity_simulation_uses_the_binding():
    from levisim.simulation import CavitySimulation
    mtrap = rieser_mtrap(3.15*μm, 0.3)
    binding = OpticalBinding()
    sim = CavitySimulation(mtrap, model='nonlinear', interactions=[binding], num_samples=20, seed=1, backend='auto')
    assert binding.traps == list(mtrap.traps)
    assert sim.backend_used == 'numpy'

"""
The quadrupole Paul trap element (levisim.physics.fields.electric)

- Mathieu parameters of the presets, the first stability region (characteristic values against the
  Floquet monodromy), the small-q limit of the secular frequency and the next-order correction
- the secular (pseudopotential) stiffness in Trap and MultiTrap, hybrid Paul-optical traps
- time domain: simulated secular frequency against the Floquet value, micromotion sidebands at
  Ω_rf ± Ω_sec, exponential growth beyond the stability boundary, gravitational sag g/Ω² and its
  excess micromotion, a stray field, electrode noise
- clear errors: neutral particle, non-Laplacian geometry, unresolved rf
"""

import warnings

import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.special import mathieu_a, mathieu_b

from levisim.core.constants import g, ħ
from levisim.core.units import Hz, kHz, mbar, mm, nm
from levisim.physics.bead import Bead
from levisim.physics.fields import (
    Gravity,
    PaulTrap,
    UniformElectricField,
    mathieu_characteristic_exponent,
    mathieu_first_region_stable,
    mathieu_floquet_multiplier,
    mathieu_secular_beta,
)
from levisim.physics.gas import Gas
from levisim.physics.particle import Sphere
from levisim.physics.trap import Trap
from levisim.physics.tweezer import Tweezer
from levisim.simulation import FreeSpaceSimulation

PARTICLE = Sphere(r=100*nm, charge=500)
F_RF = 10e3                      # Hz
Ω_RF = F_RF * Hz
UHV = Gas(P=1e-9*mbar)          # thermal force noise negligible over these runs


def voltage_for_q(q, particle=PARTICLE, Ω_rf=Ω_RF, r0=1*mm):
    """rf amplitude giving Mathieu q on an axis with curvature 1"""
    return q * particle.m * r0**2 * Ω_rf**2 / (2 * particle.q)


def ring(q=0.3, **kwargs):
    return PaulTrap.ring(voltage_for_q(q), Ω_RF, r0=1*mm, **kwargs)


def peak_frequency(values, Δt, f_min, f_max):
    """Frequency of the largest periodogram peak in (f_min, f_max), parabolic interpolation in log [Hz]"""
    window = np.hanning(len(values))
    spectrum = np.abs(np.fft.rfft((values - values.mean()) * window))**2
    f = np.fft.rfftfreq(len(values), Δt)
    band = np.flatnonzero((f > f_min) & (f < f_max))
    i = band[np.argmax(spectrum[band])]
    lo, mid, hi = np.log(spectrum[i - 1:i + 2])
    return f[i] + 0.5 * (lo - hi) / (lo - 2*mid + hi) * (f[1] - f[0])


def periodic_response(a, q, Ω, acceleration, N=30):
    """
    Exact periodic (micromotion) solution of x'' + (Ω²/4)(a + 2q cos Ωt) x = acceleration, by
    harmonic balance: (mean, amplitude of the Ω component, x(0), dx/dt(0))
    """
    n = np.arange(-N, N + 1)
    M = np.diag(-(n*Ω)**2 + Ω**2/4*a) + np.diag(np.full(2*N, Ω**2/4*q), 1) + np.diag(np.full(2*N, Ω**2/4*q), -1)
    b = np.zeros(2*N + 1)
    b[N] = acceleration
    c = np.linalg.solve(M.astype(complex), b.astype(complex))
    return c[N].real, 2*abs(c[N + 1]), np.sum(c).real, np.sum(1j*n*Ω*c).real


# --------------------------------------------------------------------------------------------------
# Mathieu's equation
# --------------------------------------------------------------------------------------------------
def test_first_stability_region_at_a_zero_ends_at_q_0_908():
    q_max = brentq(lambda q: mathieu_b(1, q), 0.5, 1.0)
    assert q_max == pytest.approx(0.908046, abs=1e-6)
    assert mathieu_first_region_stable(0.0, q_max - 1e-4)
    assert not mathieu_first_region_stable(0.0, q_max + 1e-4)


@pytest.mark.parametrize('q', [0.1, 0.3, 0.6, 0.85, 1.5])
def test_characteristic_values_bound_the_floquet_stable_region(q):
    a0, b1 = mathieu_a(0, q), mathieu_b(1, q)
    # just inside: β → 0 at the lower edge and → 1 at the upper edge, |Floquet multiplier| = 1
    assert mathieu_characteristic_exponent(a0 + 1e-6, q) < 0.01
    assert mathieu_characteristic_exponent(b1 - 1e-6, q) > 0.99
    assert mathieu_floquet_multiplier(0.5*(a0 + b1), q) == 1.0
    # just outside, on both edges, the motion grows
    assert mathieu_floquet_multiplier(a0 - 1e-3, q) > 1.0
    assert mathieu_floquet_multiplier(b1 + 1e-3, q) > 1.0
    assert np.isnan(mathieu_characteristic_exponent(b1 + 1e-3, q))


@pytest.mark.parametrize('a', [-0.01, 0.0, 0.01])
def test_small_q_limit_and_next_order(a):
    for q in (0.02, 0.05, 0.1, 0.2, 0.4):
        if a + q**2/2 <= 0.01:
            continue                                 # unconfined or marginal (a < a_0(q))
        exact = mathieu_secular_beta(a, q, 'exact')
        lowest = mathieu_secular_beta(a, q, 'lowest')
        nxt = mathieu_secular_beta(a, q, 'next')
        # lowest order: β² = a + q²/2, relative error O(q²) (0.8% at q = 0.2 for a = 0)
        assert abs(lowest - exact) / exact < 0.25 * q**2 + 1e-6
        # the next order is better by at least an order of magnitude here
        assert abs(nxt - exact) <= 0.1 * abs(lowest - exact) + 1e-9
    # Leibfried et al. 2003 Eq. 6 in the limit
    assert mathieu_secular_beta(0.0, 0.01, 'exact') == pytest.approx(0.01/np.sqrt(2), rel=1e-4)


def test_secular_beta_rejects_unknown_order():
    with pytest.raises(ValueError, match='order'):
        mathieu_secular_beta(0, 0.1, 'third')


# --------------------------------------------------------------------------------------------------
# the element: presets, parameters, errors
# --------------------------------------------------------------------------------------------------
def test_ring_preset_parameters():
    trap = ring(0.3, U_dc=0.2)
    a, q = trap.mathieu_parameters(PARTICLE)
    assert q == pytest.approx([0.3, 0.3, -0.6])
    m, Q = PARTICLE.m, PARTICLE.q
    a_x = 4 * Q * 0.2 / (m * (1*mm)**2 * Ω_RF**2)
    assert a == pytest.approx([a_x, a_x, -2*a_x])
    # the symmetry axis can be any lab axis
    _, q_y = PaulTrap.ring(voltage_for_q(0.3), Ω_RF, r0=1*mm, axis='y').mathieu_parameters(PARTICLE)
    assert q_y == pytest.approx([0.3, -0.6, 0.3])
    assert PaulTrap.endcap(1.0, Ω_RF).geometry.tolist() == PaulTrap.ring(1.0, Ω_RF).geometry.tolist()


def test_linear_preset_parameters_and_axial_frequency():
    z0, U = 2*mm, 3.0
    trap = PaulTrap.linear(voltage_for_q(0.2), Ω_RF, r0=1*mm, U_endcap=U, z0=z0, κ_endcap=0.5)
    a, q = trap.mathieu_parameters(PARTICLE)
    m, Q = PARTICLE.m, PARTICLE.q
    assert q == pytest.approx([0.2, -0.2, 0.0])
    a_z = 4 * Q * 0.5 * U / (m * z0**2 * Ω_RF**2)
    assert a == pytest.approx([-a_z/2, -a_z/2, a_z])
    Ω = trap.secular_frequencies(PARTICLE, 'exact')
    # the axis without rf is a dc harmonic oscillator: exact = lowest = √(Qκ U/(m z0²))
    assert Ω[2] == pytest.approx(np.sqrt(Q * 0.5 * U / (m * z0**2)), rel=1e-8)
    assert trap.electrode_distances == pytest.approx([1*mm, 1*mm, z0])
    assert trap.is_stable(PARTICLE)
    # a dc defocusing beyond the rf focusing makes the radial axes unstable
    strong = PaulTrap.linear(voltage_for_q(0.05), Ω_RF, r0=1*mm, U_endcap=100*U, z0=z0)
    assert strong.stable_axes(PARTICLE) == (False, False, True)


def test_errors():
    with pytest.raises(ValueError, match='Laplace'):
        PaulTrap(100, Ω_RF, geometry=(1, 1, 1))
    neutral = Sphere(r=100*nm)
    trap = ring()
    for call in (lambda: trap.force((0, 0, 0), 0.0, neutral),
                 lambda: trap.stiffness(None, neutral),
                 lambda: trap.mathieu_parameters(neutral),
                 lambda: Trap(fields=[trap], particle=neutral),
                 lambda: UniformElectricField((1, 0, 0)).force((0, 0, 0), 0.0, neutral)):
        with pytest.raises(ValueError, match='charge = 0'):
            call()
    with pytest.raises(ValueError, match='particle'):
        trap.stiffness()
    with pytest.raises(ValueError, match='noise_distances'):
        PaulTrap(100, Ω_RF, voltage_noise_psd=1e-15)


def test_force_and_secular_stiffness():
    trap = ring(0.3, U_dc=0.1, centre=(10*nm, 0, -5*nm))
    m, Q = PARTICLE.m, PARTICLE.q
    r = np.array([110*nm, -40*nm, 25*nm])
    d = r - trap.centre
    # full force at t = 0 and its average over an rf period (the dc part)
    F0 = trap.force(r, 0.0, PARTICLE)
    assert F0 == pytest.approx(-Q * (trap.dc_curvatures() + trap.rf_curvatures()) * d)
    ts = np.arange(64) / 64 * 2*np.pi / Ω_RF
    F_mean = np.mean([trap.force(r, t, PARTICLE) for t in ts], axis=0)
    assert F_mean == pytest.approx(-Q * trap.dc_curvatures() * d, rel=1e-10, abs=1e-30)
    # pseudopotential stiffness: Q c_dc + Q²c_rf²/(2mΩ²) = mΩ²(a + q²/2)/4 = m Ω_lowest²
    a, q = trap.mathieu_parameters(PARTICLE)
    K = np.diag(trap.stiffness(None, PARTICLE))
    assert K == pytest.approx(m * Ω_RF**2 * (a + q**2/2) / 4, rel=1e-12)
    assert K == pytest.approx(m * trap.secular_frequencies(PARTICLE, 'lowest')**2, rel=1e-12)
    assert trap.secular_force(r, PARTICLE) == pytest.approx(-K * d)
    assert trap.potential(r, PARTICLE) == pytest.approx(0.5 * np.sum(K * d**2))
    # a negative charge has the same pseudopotential, a and q change sign
    negative = Sphere(r=100*nm, charge=-500)
    a_n, q_n = trap.mathieu_parameters(negative)
    assert q_n == pytest.approx(-q) and a_n == pytest.approx(-a)
    assert trap.secular_stiffness_diagonal(negative) - negative.q * trap.dc_curvatures() == pytest.approx(
        K - Q * trap.dc_curvatures(), rel=1e-12)
    # higher orders on request
    exact = ring(0.3, U_dc=0.1, secular_order='exact')
    assert np.diag(exact.stiffness(None, PARTICLE)) == pytest.approx(
        m * exact.secular_frequencies(PARTICLE, 'exact')**2)


def test_trap_depth():
    trap = ring(0.2)
    k = trap.secular_stiffness_diagonal(PARTICLE)
    # ideal ring electrodes: r0 radially, r0/√2 axially; q_z = -2q_x gives 4× the stiffness axially
    assert trap.trap_depth(PARTICLE, per_axis=True) == pytest.approx(0.5 * k * np.array([1, 1, 0.5]) * (1*mm)**2)
    assert trap.trap_depth(PARTICLE) == pytest.approx(0.5 * k[0] * (1*mm)**2)
    # an unconfined axis has zero depth
    assert PaulTrap.linear(voltage_for_q(0.2), Ω_RF).trap_depth(PARTICLE) == 0.0


# --------------------------------------------------------------------------------------------------
# Trap and MultiTrap
# --------------------------------------------------------------------------------------------------
def test_trap_takes_the_secular_stiffness_and_report():
    paul = ring(0.3)
    trap = Trap(fields=[paul], particle=PARTICLE, gas=UHV)
    assert [trap.Ωx, trap.Ωy, trap.Ωz] == pytest.approx(paul.secular_frequencies(PARTICLE, 'lowest'), rel=1e-12)
    report = trap.paul_trap_report()
    assert report['stable'] and report['Ω_exact'] == pytest.approx(paul.secular_frequencies(PARTICLE, 'exact'))
    assert 'q = +0.3' in paul.report_str(PARTICLE)


def test_hybrid_trap_adds_stiffnesses_and_folds_them_into_a():
    tweezer = Tweezer(Power=1e-3)
    paul = ring(0.3)
    bead = Bead(charge=500)
    hybrid = Trap(tweezer=tweezer, bead=bead, extra_fields=[paul])
    optical = Trap(tweezer=tweezer, bead=bead)
    K_paul = paul.secular_stiffness_diagonal(bead)
    assert hybrid.kx == pytest.approx(optical.kx + K_paul[0], rel=1e-14)
    assert hybrid.Ωz == pytest.approx(np.sqrt((optical.kz + K_paul[2]) / bead.m), rel=1e-14)
    report = hybrid.paul_trap_report()
    a, _ = paul.mathieu_parameters(bead)
    assert report['a'] == pytest.approx(a + 4 * np.diag(optical.stiffness_tensor) / (bead.m * paul.Ω_rf**2))
    # lowest-order Mathieu with a folded in is exactly the summed stiffness
    assert report['Ω_lowest'] == pytest.approx([hybrid.Ωx, hybrid.Ωy, hybrid.Ωz], rel=1e-12)
    with pytest.raises(ValueError, match='no PaulTrap'):
        optical.paul_trap_report()
    with pytest.raises(TypeError, match='extra_fields'):
        Trap(extra_fields=['not an element'])


def test_multitrap_uses_the_hybrid_secular_frequencies_and_they_match_the_simulation():
    """
    A weak tweezer plus a Paul trap (q = -0.2 on its axis x, 0.1 radially) in MultiTrap: its Ωj are
    √((k_tw + k_ps)/m); the simulated (noise-free) motion of the same Trap oscillates at the exact
    (Floquet, `Trap.paul_trap_report`) frequencies to 1e-3, and at MultiTrap's lowest-order ones to
    within their pseudopotential error (0.9% on x here, 0.2-0.3% on y, z)
    """
    from levisim.multiphysics.multitrap import MultiTrap
    bead = Bead(charge=2000)
    tweezer = Tweezer(Power=4e-3)                    # Ω_tw/2π ≈ 10 kHz radially
    f_rf = 300e3
    V = 0.1 * bead.m * (1*mm)**2 * (f_rf*Hz)**2 / (2 * bead.q)
    paul = PaulTrap.ring(V, f_rf*Hz, r0=1*mm, axis='x')
    mt = MultiTrap(
        tweezers=[tweezer], beads=[bead], trap_kwargs=[{'extra_fields': [paul], 'Γ_fbx': 0.0, 'Γ_fby': 0.0}],
        Cartesian_axes_per_trap=3, static_cavity_shifts=False, gas_kwargs={'P': 1e-9*mbar},
    )
    trap = mt.traps[0]
    K_tw = Trap(tweezer=tweezer, bead=bead).stiffness_tensor
    K_ps = paul.secular_stiffness_diagonal(bead)
    for j, axis in enumerate('xyz'):
        assert mt.dimensions[f'{axis}1'].Ω == pytest.approx(np.sqrt((K_tw[j, j] + K_ps[j]) / bead.m), rel=1e-12)
    Δt = 1 / (40 * f_rf)
    sim = FreeSpaceSimulation(trap=trap, num_samples=60_000, Δt=Δt, seed=11, zero_mean=False,
                              X0=np.array([5*nm, 5*nm, 5*nm, 0, 0, 0]))
    exact = trap.paul_trap_report()['Ω_exact']
    for j, axis in enumerate('xyz'):
        f_sim = peak_frequency(sim.traces[axis].values, Δt, 2e3, 60e3)
        Ω_mt = mt.dimensions[f'{axis}1'].Ω
        assert f_sim == pytest.approx(Ω_mt / Hz, rel=1.2e-2)
        assert f_sim == pytest.approx(exact[j] / Hz, rel=1e-3)


# --------------------------------------------------------------------------------------------------
# time domain
# --------------------------------------------------------------------------------------------------
@pytest.fixture(scope='module')
def ring_simulation():
    paul = ring(0.3)
    trap = Trap(fields=[paul], particle=PARTICLE, gas=UHV)
    Δt = 1 / (40 * F_RF)
    sim = FreeSpaceSimulation(trap=trap, num_samples=40_000, Δt=Δt, seed=5, zero_mean=False,
                              X0=np.array([100*nm, 60*nm, 80*nm, 0, 0, 0]))
    return paul, trap, sim, Δt


def test_simulated_secular_frequency_is_the_floquet_value(ring_simulation):
    paul, _, sim, Δt = ring_simulation
    exact = paul.secular_frequencies(PARTICLE, 'exact') / Hz
    lowest = paul.secular_frequencies(PARTICLE, 'lowest') / Hz
    for j, axis in enumerate('xyz'):
        f_sim = peak_frequency(sim.traces[axis].values, Δt, 300, 4000)
        assert f_sim == pytest.approx(exact[j], rel=1e-3)
    # z (q = -0.6) is 8% above the lowest-order value, which the simulation resolves
    assert abs(peak_frequency(sim.z.values, Δt, 300, 4000) / lowest[2] - 1) > 0.05


def test_micromotion_sidebands_at_rf_plus_minus_secular(ring_simulation):
    paul, _, sim, Δt = ring_simulation
    f_sec = paul.secular_frequencies(PARTICLE, 'exact')[0] / Hz
    for f_expected in (F_RF - f_sec, F_RF + f_sec):
        f_sim = peak_frequency(sim.x.values, Δt, f_expected - 300, f_expected + 300)
        assert f_sim == pytest.approx(f_expected, abs=3.0)
    # relative sideband amplitudes C∓2/C0 = q/((β ∓ 2)² - a) from the Floquet recursion
    # [a - (2n + β)²] C_2n + q (C_2n-2 + C_2n+2) = 0, neglecting C±4 (→ q/4 each for β → 0,
    # Leibfried et al. 2003 Eq. 10); the lower sideband is 25% stronger than q/4 at q = 0.3
    x = sim.x.values - sim.x.values.mean()
    f = np.fft.rfftfreq(len(x), Δt)
    amplitude = np.abs(np.fft.rfft(x * np.hanning(len(x))))

    def line(f0):
        i = np.argmin(abs(f - f0))
        return amplitude[i - 3:i + 4].max()

    β = paul.secular_β(PARTICLE, 'exact')[0]
    carrier = line(f_sec)
    assert line(F_RF - f_sec) / carrier == pytest.approx(0.3 / (β - 2)**2, rel=0.05)
    assert line(F_RF + f_sec) / carrier == pytest.approx(0.3 / (β + 2)**2, rel=0.05)


def test_motion_grows_beyond_the_stability_boundary():
    """q = 0.95 > 0.908: the radial motion grows by the Floquet multiplier |λ| every rf period"""
    for q, stable in ((0.85, True), (0.95, False)):
        paul = PaulTrap.linear(voltage_for_q(q), Ω_RF, r0=1*mm)
        assert paul.stable_axes(PARTICLE)[:2] == (stable, stable)
        trap = Trap(fields=[paul], particle=PARTICLE, gas=UHV)
        n_per = 40
        sim = FreeSpaceSimulation(trap=trap, num_samples=40 * n_per, Δt=1/(n_per*F_RF), seed=1,
                                  zero_mean=False, X0=np.array([1*nm, 0, 0, 0, 0, 0]))
        envelope = np.abs(sim.x.values[:-1]).reshape(40, n_per).max(axis=1)
        if stable:
            assert envelope.max() < 10*nm
        else:
            λ = mathieu_floquet_multiplier(0.0, q)
            rate = np.polyfit(np.arange(20, 40), np.log(envelope[20:]), 1)[0]
            assert rate == pytest.approx(np.log(λ), rel=0.02)
            assert envelope[-1] > 1e3 * nm


def test_gravitational_sag_and_its_excess_micromotion():
    paul = ring(0.3)
    trap = Trap(fields=[paul, Gravity()], particle=PARTICLE, gas=UHV)
    a, q = paul.mathieu_parameters(PARTICLE)
    # pseudopotential sag g/Ωx² (lowest order) from the static (secular) force
    sag = trap.find_equilibrium()
    assert sag == pytest.approx([-g / trap.Ωx**2, 0, 0], abs=1e-15)
    # the exact periodic orbit of the Mathieu oscillator under gravity
    mean, micromotion, x0, v0 = periodic_response(a[0], q[0], Ω_RF, -g)
    assert mean == pytest.approx(sag[0], rel=3e-3)        # pseudopotential error at q = 0.3
    assert micromotion == pytest.approx(paul.excess_micromotion_amplitude(sag, PARTICLE)[0], rel=3e-3)
    # the simulation started on the periodic orbit stays on it
    Δt = 1 / (40 * F_RF)
    sim = FreeSpaceSimulation(trap=trap, num_samples=40_000, Δt=Δt, seed=2, zero_mean=False,
                              X0=np.array([x0, 0, 0, PARTICLE.m * v0, 0, 0]))
    x = sim.x.values
    t = np.arange(len(x)) * Δt
    assert x.mean() == pytest.approx(mean, rel=1e-3)
    amplitude = 2 * np.hypot(np.mean(x * np.cos(Ω_RF*t)), np.mean(x * np.sin(Ω_RF*t)))
    assert amplitude == pytest.approx(micromotion, rel=2e-3)


def test_stray_field_displacement_and_micromotion():
    paul = ring(0.2)
    E = np.array([0.0, 2.0, -1.0])                    # V/m
    trap = Trap(fields=[paul, UniformElectricField(E)], particle=PARTICLE, gas=UHV)
    u = trap.find_equilibrium()
    K = paul.secular_stiffness_diagonal(PARTICLE)
    assert u == pytest.approx(PARTICLE.q * E / K, rel=1e-8, abs=1e-18)
    assert paul.excess_micromotion_amplitude(u, PARTICLE) == pytest.approx(0.5 * np.abs([0.2, 0.2, 0.4]) * np.abs(u))
    assert UniformElectricField(E).potential((1, 2, 3), PARTICLE) == pytest.approx(-PARTICLE.q * (E @ [1, 2, 3]))


# --------------------------------------------------------------------------------------------------
# noise
# --------------------------------------------------------------------------------------------------
def test_heating_rate_reproduces_dania_et_al_2024():
    """
    Dania et al., PRL 132, 133602 (2024): q = 300 e, m = 4.3e-17 kg, Ωz/2π = 1.28 kHz; S_v = 38 nV/√Hz
    at d = 0.92 mm ⇔ S_EE = 1.7e-9 (V/m)²/Hz, S_ff = 4e-42 N²/Hz; measured 3.1(8)e4 phonon/s (dark)
    """
    ρ = Sphere().ρ
    particle = Sphere(r=(3 * 4.3e-17 / (4*np.pi*ρ))**(1/3), charge=300)
    assert particle.m == pytest.approx(4.3e-17)
    paul = PaulTrap.linear(500, 5*kHz, r0=1*mm, voltage_noise_psd=(38e-9)**2, noise_distances=0.92*mm)
    S_E = paul.electric_field_noise_psd()
    assert S_E == pytest.approx(np.full(3, 1.7e-9), rel=0.01)
    S_ff = particle.q**2 * S_E[2]
    assert S_ff == pytest.approx(4e-42, rel=0.02)
    Γ = paul.heating_rate(particle, Ω=1.28*kHz)
    assert Γ[2] == pytest.approx(S_ff / (4 * particle.m * ħ * 1.28*kHz))
    assert abs(Γ[2] - 3.1e4) < 0.8e4                   # within the measurement's 1σ
    # two-sided force noise for the simulation: 〈FF〉 = Q² S_E / 2 · δ
    assert np.diag(paul.noise(particle)) == pytest.approx(particle.q**2 * S_E / 2)


def test_electrode_noise_raises_the_simulated_temperature():
    """
    Electrode noise on x only, equal to the thermal force noise Sth: the x variance grows by
    1 + S_el/S_x relative to y (T_eff = T (1 + S_el/S_x)); without noise the noise matrix is the
    usual 6×4
    """
    gas = Gas(P=0.3*mbar)
    probe = Trap(fields=[ring(0.3)], particle=PARTICLE, gas=gas)
    S_V = 2 * probe.Sth * (1*mm)**2 / PARTICLE.q**2  # two-sided Q² S_V/(2d²) = Sth
    paul = ring(0.3, voltage_noise_psd=S_V, noise_distances=(1*mm, np.inf, np.inf))
    trap = Trap(fields=[paul], particle=PARTICLE, gas=gas)
    Δt = 1 / (40 * F_RF)
    sim = FreeSpaceSimulation(trap=trap, num_samples=100_000, Δt=Δt, seed=4)
    G, S = sim.noise_matrix(), sim.noise_spectral_densities()
    assert G.shape == (6, 5) and S[4] == pytest.approx(probe.Sth)
    assert np.abs(G[3:, 4]) == pytest.approx([1, 0, 0])
    # x and y share the directed noise Sdir (cos ψ, sin ψ); the electrode noise adds Sth on x
    S_x = probe.Sth + probe.Sdir * np.cos(probe.ψ)**2
    S_y = probe.Sth + probe.Sdir * np.sin(probe.ψ)**2
    ratio = np.var(sim.x.values) / np.var(sim.y.values)
    assert ratio == pytest.approx((S_x + probe.Sth) / S_y, rel=0.15)
    # without noise the matrices are those of every other trap
    quiet = FreeSpaceSimulation(trap=Trap(fields=[ring(0.3)], particle=PARTICLE, gas=gas), num_samples=10, Δt=Δt, seed=4)
    assert quiet.noise_matrix().shape == (6, 4)
    off = ring(0.3, voltage_noise_psd=S_V, noise_distances=1*mm, noise_in_simulation=False)
    assert FreeSpaceSimulation(trap=Trap(fields=[off], particle=PARTICLE, gas=gas), num_samples=10, Δt=Δt).noise_matrix().shape == (6, 4)
    assert trap.field_noise_psd[0, 0] == pytest.approx(probe.Sth)


def test_time_step_must_resolve_the_rf():
    trap = Trap(fields=[ring(0.3)], particle=PARTICLE, gas=UHV)
    with pytest.warns(UserWarning, match='does not resolve'):
        FreeSpaceSimulation(trap=trap, num_samples=10, Δt=1/(10*F_RF))
    with pytest.warns(UserWarning, match='Heun'):
        FreeSpaceSimulation(trap=trap, num_samples=10, Δt=1/(40*F_RF), RK_method='vector_2')
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        FreeSpaceSimulation(trap=trap, num_samples=10, Δt=1/(40*F_RF))
        FreeSpaceSimulation(trap=trap, num_samples=10, Δt=1/(200*F_RF), RK_method='vector_2')


def test_static_force_uses_the_secular_force_only_for_time_dependent_elements():
    paul = ring(0.3)
    trap = Trap(tweezer=Tweezer(), bead=Bead(charge=500), extra_fields=[paul])
    r = (20*nm, -10*nm, 30*nm)
    expected = np.array(trap.Ftw(*r)) + paul.secular_force(r, trap.bead)
    assert trap.static_force(r, t=1.234e-5) == pytest.approx(expected, rel=1e-14)
    assert not Tweezer.is_time_dependent and PaulTrap.is_time_dependent

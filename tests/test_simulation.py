"""
Tests of the stochastic (Langevin) Simulation of a bead in a free-space tweezer

All runs are seeded, so every number checked here is deterministic. The statistical tolerances
are nonetheless set at roughly 3 standard errors of the estimator, so that they would also pass
for (almost) any other seed: e.g. the variance of an Ornstein-Uhlenbeck-like trace of length T
with correlation time 1/Γ has a relative standard error of about √(2/(ΓT)).
"""
import numpy as np
import pytest
from scipy.integrate import trapezoid

from levisim.analysis.spectral_density_estimate import SpectralDensityEstimate
from levisim.core.units import kHz, megaHz, Hz
from levisim.data.simulation import Simulation
from levisim.physics.trap import Trap


Δt = 1e-7
# large feedback damping => short correlation times => tight statistics from short runs
COLD = dict(Γ_fbx=20*kHz, Γ_fby=20*kHz, Γ_fbz=5*kHz)


@pytest.fixture(scope='module')
def cold_sim():
    """β = 0 (no directed force): pure cold damping, T_eff = T Γth / (Γth + Γfb)"""
    return Simulation(trap=Trap(β=0.0, **COLD), num_samples=200_000, Δt=Δt, seed=3)


def legacy_vector_4(trap, num_samples, Δt):
    """
    The 'vector_4' loop as written before pre-generating the random numbers: one np.random.normal()
    call per number, the noise matrix rebuilt every stage, three separate tweezer-force calls
    """
    q = [3.99956364361748, 1.64524970733585, 1.59330355118722, 0.26330006501868]
    a21 = 0.66667754298442
    a31, a32 = 0.63493935027993, 0.00342761715422
    a41, a42, a43 = -2.32428921184321, 2.69723745129487, 0.29093673271592
    α1, α2, α3, α4 = 0.25001351164789, 0.67428574806272, -0.00831795169360, 0.08401868181222
    m = trap.bead.m

    def dynamics(t, X):
        x, y, z, px, py, pz = X
        return np.array([
            px / m, py / m, pz / m,
            trap.Ftwx(x, y, z) - trap.Γx*px,
            trap.Ftwy(x, y, z) - trap.Γy*py,
            trap.Ftwz(x, y, z) - trap.Γz*pz,
        ])

    def noises(t, X):
        ψ = trap.ψ
        return np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                         [1, 0, 0, np.cos(ψ)], [0, 1, 0, np.sin(ψ)], [0, 0, 1, 0]])

    Σth = [np.sqrt(qi * trap.Sth / Δt) for qi in q]
    Σdir = [np.sqrt(qi * trap.Sdir / Δt) for qi in q]
    X = np.zeros([num_samples + 1, 6])
    time_array = np.linspace(0, num_samples*Δt, num_samples + 1)
    for n, t in enumerate(time_array[:-1]):
        dW = [np.array([Σth[i]*np.random.normal(), Σth[i]*np.random.normal(),
                        Σth[i]*np.random.normal(), Σdir[i]*np.random.normal()]) for i in range(4)]
        K1 = (dynamics(t, X[n]) + noises(t, X[n]).dot(dW[0]))*Δt
        t2 = t + a21*Δt
        X2 = X[n] + a21*K1
        K2 = (dynamics(t2, X2) + noises(t2, X2).dot(dW[1]))*Δt
        t3 = t + (a31 + a32)*Δt
        X3 = X[n] + a31*K1 + a32*K2
        K3 = (dynamics(t3, X3) + noises(t3, X3).dot(dW[2]))*Δt
        t4 = t + (a41 + a42 + a43)*Δt
        X4 = X[n] + a41*K1 + a42*K2 + a43*K3
        K4 = (dynamics(t4, X4) + noises(t4, X4).dot(dW[3]))*Δt
        X[n + 1] = X[n] + α1*K1 + α2*K2 + α3*K3 + α4*K4
    return X


def test_vector_4_bit_identical_to_legacy_loop():
    """Pre-generating the normals (seed=None => legacy global RandomState) changes no number"""
    trap = Trap()
    np.random.seed(1234)
    X_legacy = legacy_vector_4(trap, 300, Δt)
    np.random.seed(1234)
    sim = Simulation(trap=trap, num_samples=300, Δt=Δt, zero_mean=False)
    assert np.array_equal(sim.X, X_legacy)


def test_seed_makes_runs_reproducible():
    a = Simulation(num_samples=2000, Δt=Δt, seed=7)
    b = Simulation(num_samples=2000, Δt=Δt, seed=7)
    c = Simulation(num_samples=2000, Δt=Δt, seed=8)
    assert np.array_equal(a.X, b.X)
    assert not np.array_equal(a.X, c.X)


def test_rk_method_validation():
    with pytest.raises(ValueError, match='scalar_2'):
        Simulation(num_samples=10, Δt=Δt, RK_method='scalar_2')
    with pytest.raises(ValueError, match='Unknown'):
        Simulation(num_samples=10, Δt=Δt, RK_method='vector_3')


def test_T_eff_formula():
    trap = Trap(β=0.0, **COLD)
    for axis in 'xyz':
        assert trap.T_eff[axis] == pytest.approx(trap.gas.T * trap.Γth / (trap.Γth + getattr(trap, f'Γ_fb{axis}')))
    # with the directed force, x and y get the extra Sdir,j = β² Sth cos²ψ (sin²ψ); z gets nothing
    trap = Trap(β=np.sqrt(2), ψ=np.pi/4, **COLD)
    assert trap.T_eff['x'] == pytest.approx(2 * trap.gas.T * trap.Γth / trap.Γx)
    assert trap.T_eff['z'] == pytest.approx(trap.gas.T * trap.Γth / trap.Γz)


def test_theory_uses_total_damping():
    trap = Trap()
    assert (trap.theory.Γx, trap.theory.Γy, trap.theory.Γz) == (trap.Γx, trap.Γy, trap.Γz)


def test_fit_amplitude_conventions():
    trap = Trap()
    assert trap.A_Hz * Hz**3 == pytest.approx(trap.A)
    assert trap.A_rad == pytest.approx(trap.A / Hz)


def test_equipartition_cold_damping(cold_sim):
    """T via Var(j) reproduces T_eff = T Γth/(Γth + Γfb,j); 2e5 samples, std. error ≈ 3% (x, y), 8% (z)"""
    trap = cold_sim.trap
    for axis, tol in (('x', 0.08), ('y', 0.08), ('z', 0.25)):
        assert cold_sim.temperature_via_variance(axis) == pytest.approx(trap.T_eff[axis], rel=tol)


def test_equipartition_improved_euler():
    trap = Trap(β=0.0, **COLD)
    sim = Simulation(trap=trap, num_samples=200_000, Δt=Δt, RK_method='vector_2', seed=11)
    for axis in 'xy':
        assert sim.temperature_via_variance(axis) == pytest.approx(trap.T_eff[axis], rel=0.08)


def test_equipartition_with_directed_force():
    trap = Trap(β=np.sqrt(2), ψ=np.pi/4, **COLD)
    sim = Simulation(trap=trap, num_samples=200_000, Δt=Δt, seed=4)
    for axis in 'xy':
        assert sim.temperature_via_variance(axis) == pytest.approx(trap.T_eff[axis], rel=0.08)
    # the shared directed force correlates x and y: theory gives Cov(x, y) = ∫ Cxy df > 0 for ψ = 45°
    theory = trap.theory
    f = np.linspace(1, 2*megaHz, 400_001)
    cov_theory = trapezoid(theory.Cxy(2*np.pi*f), f)
    cov_sim = np.cov(sim.x.values, sim.y.values)[0, 1]
    assert cov_sim == pytest.approx(cov_theory, rel=0.25)


def test_psd_fit_and_area_temperatures(cold_sim):
    trap = cold_sim.trap
    for axis in 'xy':
        est = SpectralDensityEstimate(cold_sim, axis*2, num_segments=20)
        assert est.Ωfit == pytest.approx(trap.Ωj[axis], rel=0.01)
        assert est.Γfit == pytest.approx(trap.Γj[axis], rel=0.15)
        assert est.temperature_via_fit == pytest.approx(trap.T_eff[axis], rel=0.1)
        assert est.temperature_via_area == pytest.approx(trap.T_eff[axis], rel=0.1)


def test_psd_matches_theory(cold_sim):
    """Band-averaged one-sided Sxx estimate / SpectralDensityTheory.Sxx within 10% around resonance"""
    trap = cold_sim.trap
    est = SpectralDensityEstimate(cold_sim, 'xx', num_segments=20)
    f0 = trap.Ωx / Hz
    band = (est.freq_range > f0 - 2*trap.Γx/Hz) & (est.freq_range < f0 + 2*trap.Γx/Hz)
    ratio = np.real(est.values[band]) / trap.theory.Sxx(2*np.pi*est.freq_range[band])
    assert np.mean(ratio) == pytest.approx(1, rel=0.1)


def test_detection_floor_is_measurement_noise_only():
    """The floor is added to the recorded traces, not the state, and sets the far-off-resonance PSD"""
    G_floor = 1e-26  # one-sided [m²/Hz], ~3 orders above the motional PSD at > 1 MHz
    clean = Simulation(num_samples=100_000, Δt=Δt, seed=21)
    noisy = Simulation(num_samples=100_000, Δt=Δt, seed=21, detection_floor=G_floor)
    # same seed => identical dynamics: the floor did not feed back into X
    assert np.array_equal(clean.X, noisy.X)
    est = SpectralDensityEstimate(noisy, 'xx', num_segments=10)
    band = (est.freq_range > 1*megaHz) & (est.freq_range < 4.5*megaHz)
    assert np.mean(np.real(est.values[band])) == pytest.approx(G_floor, rel=0.03)
    # per-axis dict: only y gets a floor
    per_axis = Simulation(num_samples=1000, Δt=Δt, seed=21, detection_floor={'y': G_floor})
    assert np.array_equal(per_axis.position_values['x'], per_axis.X[:, 0])
    assert not np.array_equal(per_axis.position_values['y'], per_axis.X[:, 1])

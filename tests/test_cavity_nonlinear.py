"""
Tests of the nonlinear CavitySimulation (levisim.simulation.nonlinear_cavity, model='nonlinear')

- the closed-form force and cavity drive against the optical potential they come from
- the Jacobian at the node equals MultiTrap's drift matrix (1×2, 2×2, 1×3)
- the off-node equilibrium, and the scattering-force equilibrium against Trap.z_eq_exact
- small-amplitude convergence to the linear model (spectra and pathwise)
- the measured Δt-convergence order of the split step
- the unstable 2×2 system: initial growth at max_growth_rate, then saturation
- the Duffing frequency shift of the tweezer against the exact period and FreeSpaceSimulation
- NumPy against numba, Heun against the split step, pair-force interactions, argument checks
"""
import numpy as np
import pytest
from scipy.integrate import quad

from levisim.core.units import K, degrees, mbar, nm, μs
from levisim.multiphysics.covariance import quadrature_drift_matrix
from levisim.multiphysics.multitrap import MultiTrap
from levisim.multiphysics.output_modes import QuadratureSystem
from levisim.multiphysics.utils import shelf
from levisim.physics import coherent_scattering as cs
from levisim.physics.bead import Bead
from levisim.simulation import FreeSpaceSimulation, Homodyne, PhotonNumber, Position
from levisim.simulation.backends import numba_available
from levisim.simulation.cavity import (
    CavitySimulation,
    _psd_square_root,
    van_loan_discretisation,
)
from levisim.simulation.nonlinear_cavity import NONLINEAR_TERMS, NonlinearCavityModel

ALL_TERMS = ('elements', 'coherent_scattering', 'cavity')
needs_numba = pytest.mark.skipif(not numba_available(), reason='long statistical run: needs numba')


def physical(num_traps, axes, Φ=None, **kwargs):
    if Φ is not None:
        kwargs['beads'] = [Bead(r=(60.1 + 0.1 * i) * nm, Φ=Φ[i]) for i in range(num_traps)]
    return MultiTrap(num_traps=num_traps, Cartesian_axes_per_trap=axes, cavity_kwargs=shelf.DEFAULT_cavity_kwargs,
                     gas_kwargs=kwargs.pop('gas_kwargs', shelf.DEFAULT_gas_kwargs), **kwargs)


def cold(T, **kwargs):
    """1×2 at the node with the bath at temperature T (pressure scaled to keep Γ), recoil off"""
    gas = dict(shelf.DEFAULT_gas_kwargs, T=T * K, P=shelf.DEFAULT_gas_kwargs['P'] * T / 300)
    return physical(1, 2, gas_kwargs=gas, photon_recoil=False, **kwargs)


@pytest.fixture(scope='module')
def off_node():
    return physical(1, 3, Φ=[np.pi / 4])


# --------------------------------------------------------------------------------------------------
# the force and the drive
# --------------------------------------------------------------------------------------------------
def test_optical_force_is_minus_the_gradient_of_the_potential(off_node):
    trap, E0 = off_node.traps[0], off_node.cavity.E0
    α = (off_node.cavity.A0 + 1j * off_node.cavity.P0) / 2 * np.exp(-0.3j)
    rng = np.random.default_rng(0)
    points = [np.zeros(3), np.array([30e-9, -20e-9, 40e-9]), rng.normal(size=3) * 200e-9, np.array([0.5e-6, 0.3e-6, 1e-6])]
    for terms in (('tweezer',), ('coherent_scattering',), ('cavity',), ('tweezer', 'coherent_scattering', 'cavity')):
        for r in points:
            F = cs.optical_force(r, trap, E0, α, terms)
            h = 1e-11
            F_fd = np.zeros(3)
            for j in range(3):
                e = np.zeros(3)
                e[j] = h
                U = [cs.optical_potential(r + s * e, trap, E0, α, terms) for s in (2, 1, -1, -2)]
                F_fd[j] = -(-U[0] + 8 * U[1] - 8 * U[2] + U[3]) / (12 * h)
            # the scale floor is the tweezer force of a 1 nm displacement (the force vanishes at the centre)
            scale = max(np.max(np.abs(F)), trap.kx * 1e-9)
            assert np.max(np.abs(F - F_fd)) < 1e-6 * scale, (terms, r)
        stacked = cs.optical_force(np.array(points), trap, E0, α, terms)
        assert np.array_equal(stacked[1], cs.optical_force(points[1], trap, E0, α, terms))
    # the tweezer term is Tweezer.force bit for bit, and the static force matches static_cavity_force
    for r in points:
        assert np.array_equal(cs.optical_force(r, trap, E0, 0, ('tweezer',)), trap.tweezer.force(r, 0.0, trap.bead))
    α0 = (off_node.cavity.A0 + 1j * off_node.cavity.P0) / 2
    np.testing.assert_allclose(cs.optical_force(np.zeros(3), trap, E0, α0, ('coherent_scattering', 'cavity')),
                               cs.static_cavity_force(trap, E0, α0), rtol=1e-12)
    with pytest.raises(ValueError, match='unknown'):
        cs.optical_force(np.zeros(3), trap, E0, α0, ('nope',))


def test_cavity_drive_is_conjugate_to_the_force(off_node):
    """dα/dt = -(i/ħ)∂U/∂α*: the drive and the force come from the same energy"""
    from levisim.core.constants import ħ
    trap, E0 = off_node.traps[0], off_node.cavity.E0
    α = (off_node.cavity.A0 + 1j * off_node.cavity.P0) / 2
    terms = ('coherent_scattering', 'cavity')
    for r in (np.zeros(3), np.array([40e-9, -10e-9, 60e-9])):
        h = 1e-3 * abs(α)
        dU_dR = (cs.optical_potential(r, trap, E0, α + h, terms) - cs.optical_potential(r, trap, E0, α - h, terms)) / (2 * h)
        dU_dI = (cs.optical_potential(r, trap, E0, α + 1j * h, terms) - cs.optical_potential(r, trap, E0, α - 1j * h, terms)) / (2 * h)
        expected = -1j / ħ * 0.5 * (dU_dR + 1j * dU_dI)
        drive = cs.cavity_drive(r, trap, E0, α)
        assert abs(drive - expected) < 1e-10 * abs(drive)
    # at the trap centre the scattering drive is levisim's static drive i·ωd·sinθ·cosΦ·e^{iφ}
    φ = 0.7
    drive0 = cs.cavity_drive(np.zeros(3), trap, E0, 0.0, φ=φ, terms=('coherent_scattering',))
    expected0 = 1j * trap.ωd * np.sin(trap.drive_tweezer.θ) * np.cos(trap.bead.Φ) * np.exp(1j * φ)
    assert abs(drive0 - expected0) < 1e-12 * abs(expected0)
    # amplitude and its gradient
    r = np.array([20e-9, 30e-9, -50e-9])
    D, gradD = cs.scattering_amplitude(r, trap), cs.scattering_amplitude_gradient(r, trap)
    for j in range(3):
        e = np.zeros(3)
        e[j] = 1e-11
        fd = (cs.scattering_amplitude(r + e, trap) - cs.scattering_amplitude(r - e, trap)) / 2e-11
        assert abs(fd - gradD[j]) < 1e-5 * np.abs(gradD).max()
    assert abs(D) <= np.sin(trap.drive_tweezer.θ)


# --------------------------------------------------------------------------------------------------
# linearisation
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('num_traps, axes', [(1, 2), (2, 2), (1, 3)])
def test_jacobian_equals_the_drift_matrix_at_the_node(num_traps, axes):
    mtrap = physical(num_traps, axes)
    model = NonlinearCavityModel(mtrap, damping='symmetric')
    idx = model.modelled_indices()
    J = model.jacobian[np.ix_(idx, idx)]
    A = quadrature_drift_matrix(mtrap.drift_matrix)
    scale = np.abs(A).max()
    assert np.abs(J - A).max() <= 1e-8 * scale
    # every entry above round-off (MultiTrap's cos(π/2) leaves ~1e-18 relative entries) to 1e-8
    big = np.abs(A) > 1e-12 * scale
    assert np.all(np.abs(J - A)[big] <= 1e-8 * np.abs(A[big]))
    assert abs(model.α_eq) < 1e-9 and np.abs(model.r_eq).max() < 1e-20
    # the modelled block has MultiTrap's eigenvalues; the axes MultiTrap leaves out are decoupled at
    # the node (gz ∝ cos Φ = 0), so they only add their own weakly damped modes (-Γ/2)
    eig = lambda M: np.sort_complex(np.round(np.linalg.eigvals(M), 3))
    np.testing.assert_allclose(eig(J), eig(A), rtol=1e-8)
    # the physical (velocity) damping differs only in the damping entries: -Γ on P instead of -Γ/2 on X and P
    velocity = NonlinearCavityModel(mtrap, damping='velocity')
    D = velocity.jacobian - model.jacobian
    for p in range(num_traps):
        for a in range(3):
            iX = 2 + 6 * p + 2 * a
            Γ = model.Γ[p, a]
            assert D[iX, iX] == pytest.approx(Γ / 2, rel=1e-6, abs=1e-9 * scale)
            assert D[iX + 1, iX + 1] == pytest.approx(-Γ / 2, rel=1e-6, abs=1e-9 * scale)
            D[iX, iX] = D[iX + 1, iX + 1] = 0
    assert np.abs(D).max() <= 1e-9 * scale
    # the noise: the symmetric form reproduces QuadratureSystem.from_multitrap's diffusion
    reference = QuadratureSystem.from_multitrap(mtrap).diffusion_matrix
    np.testing.assert_allclose(model.quadrature_system.diffusion_matrix[np.ix_(idx, idx)], reference, rtol=1e-12)


def test_off_node_equilibrium_and_linearisation(off_node):
    model = NonlinearCavityModel(off_node, damping='symmetric')
    # converged: the drift vanishes to round-off of its terms
    assert np.abs(model.residual).max() < 1e-12 * np.abs(model.jacobian).max() * np.abs(model.u_eq).max()
    # the coherent-scattering force pushes the bead tens of nm along z (Phase 1: ~33 nm estimate)
    assert 25 * nm < model.r_eq[0, 2] < 45 * nm
    assert np.abs(model.r_eq[0, :2]).max() < 10 * nm
    # the static field at the true equilibrium differs from MultiTrap's (the scattering phase ψ(z_eq))
    α_MT = (off_node.cavity.A0 + 1j * off_node.cavity.P0) / 2
    assert abs(model.α_eq - α_MT) > 0.05 * abs(α_MT)
    assert abs(abs(model.α_eq) / abs(α_MT) - 1) < 0.1
    # MultiTrap linearises at the trap centre: its drift matrix differs at the per-cent level
    idx = model.modelled_indices()
    A = quadrature_drift_matrix(off_node.drift_matrix)
    rel = np.abs(model.jacobian[np.ix_(idx, idx)] - A).max() / np.abs(A).max()
    assert 1e-3 < rel < 0.1
    # mechanical frequencies at the true equilibrium within 0.2 kHz of MultiTrap's
    freq = lambda M: np.sort(np.abs(np.linalg.eigvals(M).imag))[::2] / (2 * np.pi)
    assert np.abs(freq(model.jacobian[np.ix_(idx, idx)])[:3] - freq(A)[:3]).max() < 200
    # a strongly driven system (three beads with equal tweezer phases) needs the trust region
    strong = NonlinearCavityModel(physical(3, 2, Φ=[45 * degrees, 52 * degrees, 59 * degrees], φn_dict={2: 0.0, 3: 0.0}))
    assert np.abs(strong.r_eq[:, 2]).max() < 150 * nm


def test_scattering_force_equilibrium_matches_the_trap():
    """With a PhotonRecoil element (and the bead at the node) the equilibrium is Trap.z_eq_exact"""
    mtrap = physical(1, 3, trap_kwargs=[{'photon_recoil': True}])
    model = NonlinearCavityModel(mtrap)
    trap = mtrap.traps[0]
    assert model.r_eq[0, 2] == pytest.approx(trap.z_eq_exact, rel=1e-9)
    assert abs(model.r_eq[0, 0]) < 1e-15 and abs(model.r_eq[0, 1]) < 1e-15
    # the stiffness there is the tweezer's at z_eq (softer than at the focus)
    K_eq = model.stiffness_at_equilibrium()[0]
    K_fd = sum(element.stiffness(np.array([0, 0, trap.z_eq_exact]), trap.bead) for element in trap.fields)
    np.testing.assert_allclose(np.diag(K_eq), np.diag(K_fd), rtol=1e-6)


# --------------------------------------------------------------------------------------------------
# small amplitude: the linear model
# --------------------------------------------------------------------------------------------------
def test_everything_linearised_has_no_remainder():
    mtrap = cold(3)
    model = NonlinearCavityModel(mtrap, linearised_terms=ALL_TERMS)
    u = np.random.default_rng(1).normal(size=(5, model.n)) * 100
    assert np.array_equal(model.remainder(u), np.zeros_like(u))
    assert NonlinearCavityModel(mtrap).remainder(np.zeros((1, model.n))).max() == 0.0


def test_pathwise_difference_from_the_linear_model_scales_with_amplitude():
    differences = []
    for T in (3, 0.03):
        kw = {'model': 'nonlinear', 'damping': 'symmetric', 'num_samples': 2**12, 'Δt': 0.1 * μs, 'seed': 7,
              'num_trajectories': 2, 'detection': [Homodyne(), Position('x1')], 'backend': 'numpy'}
        mtrap = cold(T)
        a = CavitySimulation(mtrap, **kw)
        kw['detection'] = [Homodyne(), Position('x1')]
        b = CavitySimulation(mtrap, linearised_terms=ALL_TERMS, **kw)
        differences.append(np.std(a.records['homodyne'] - b.records['homodyne']) / np.std(b.records['homodyne']))
    # leading nonlinearity is quadratic: relative difference ∝ amplitude ∝ √T (factor 10 for 100× T)
    assert differences[0] < 0.05
    assert 5 < differences[0] / differences[1] < 20


@needs_numba
def test_small_amplitude_spectra_converge_to_the_linear_model():
    mtrap = cold(0.03)
    detection = lambda: [Homodyne(), Position('x1'), Position('y1')]
    nl = CavitySimulation(mtrap, model='nonlinear', damping='symmetric', num_samples=2**16, Δt=0.1 * μs, seed=21,
                          num_trajectories=16, detection=detection(), backend='numba')
    lin = CavitySimulation(mtrap, model='linear', num_samples=16, Δt=0.1 * μs, seed=1, detection=detection())
    for label in ('homodyne', 'x1', 'y1'):
        f, S = nl.averaged_psd(label, num_segments=16)
        theory = lin.record_spectrum(label, f[1:], kind='discrete')
        for lo, hi in ((1e3, 100e3), (100e3, 160e3), (160e3, 400e3), (1e6, 4.9e6)):
            band = (f[1:] > lo) & (f[1:] < hi)
            assert abs(np.median(S[1:][band] / theory[band]) - 1) < 0.06, (label, lo, hi)


# --------------------------------------------------------------------------------------------------
# convergence order
# --------------------------------------------------------------------------------------------------
def _split_path(model, u0, Φδ, ξ, level, δ):
    """Strang splitting with h = 2^level·2δ driven by fine exact linear increments ξ (shared path)"""
    half = 2**level
    h = 2 * half * δ
    steps = ξ.shape[0] // (2 * half)
    u, j = u0.copy(), 0

    def linear(u, count):
        nonlocal j
        for _ in range(count):
            u = u @ Φδ.T + ξ[j]
            j += 1
        return u

    R = model.remainder
    u = linear(u, half)
    for s in range(steps):
        k1 = R(u)
        k2 = R(u + h / 2 * k1)
        k3 = R(u + h / 2 * k2)
        k4 = R(u + h * k3)
        u = u + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        u = linear(u, 2 * half if s < steps - 1 else half)
    return u


def strong_errors(model, T, δ, K_paths, levels, amplitude, seed=0):
    system = model.quadrature_system
    Φδ, Qδ = van_loan_discretisation(system.A, system.diffusion_matrix, δ)
    rng = np.random.default_rng(seed)
    ξ = rng.standard_normal((round(T / δ), K_paths, model.n)) @ _psd_square_root(Qδ).T
    u0 = amplitude * rng.standard_normal((K_paths, model.n)) @ np.linalg.cholesky(model.steady_state_covariance()).T
    paths = {level: _split_path(model, u0, Φδ, ξ, level, δ) for level in levels}
    reference = paths[levels[0]]
    scale = np.sqrt(np.mean(reference**2, axis=0))
    h = np.array([2**level * 2 * δ for level in levels[1:]])
    error = np.array([np.sqrt(np.mean(np.sum(((paths[level] - reference) / scale)**2, axis=1))) for level in levels[1:]])
    return h, error


def test_split_step_convergence_order(off_node):
    """
    Measured, not assumed: errors against a fine reference on the SAME Brownian path (fine exact
    linear increments composed exactly). At 10× the thermal amplitude the strong order is ≈ 2 (the
    Strang splitting error); at the thermal amplitude it falls to ≈ 1 at small h.
    """
    model = NonlinearCavityModel(off_node)
    δ = 0.0078125 * μs / 2
    h, error = strong_errors(model, 4 * μs, δ, 32, [0, 2, 3, 4, 5], amplitude=10.0)
    order = np.polyfit(np.log(h), np.log(error), 1)[0]
    assert 1.6 < order < 2.4, order
    # at the default Δt = 0.1 μs and thermal amplitude the error is tiny relative to the rms
    h, error = strong_errors(model, 4 * μs, δ, 32, [0, 4], amplitude=1.0)
    assert h[0] == pytest.approx(0.125 * μs)
    assert error[0] < 1e-4


# --------------------------------------------------------------------------------------------------
# beyond the spectra
# --------------------------------------------------------------------------------------------------
def _energies(sim):
    """(X² + P²)/4 per mechanical axis, shape (K, T, 3N)"""
    s = sim.states
    return 0.25 * (s[..., 2::2]**2 + s[..., 3::2]**2)


@needs_numba
def test_unstable_system_grows_at_max_growth_rate_then_saturates():
    mtrap = physical(2, 2)          # built-in phases 0°, 90°: unstable
    assert not mtrap.is_stable
    sim = CavitySimulation(mtrap, model='nonlinear', num_samples=80_000, Δt=0.1 * μs, seed=2, num_trajectories=4,
                           initial_state='equilibrium', store_state=True, detection=[Homodyne()], backend='numba')
    assert sim.system.largest_growth_rate() == pytest.approx(mtrap.max_growth_rate, rel=1e-9)
    E = _energies(sim)[..., [0, 1, 3, 4]].sum(-1)       # x and y of both beads
    t = np.arange(E.shape[1]) * sim.Δt
    rates = []
    for k in range(E.shape[0]):
        i0, i1 = np.argmax(E[k] > 1e4), np.argmax(E[k] > 1e7)
        assert 0 < i0 < i1
        rates.append(np.polyfit(t[i0:i1], np.log(E[k, i0:i1]), 1)[0] / 2)
    assert np.mean(rates) == pytest.approx(mtrap.max_growth_rate, rel=0.03)
    # it saturates: every trajectory peaks within the run and has decayed below 10% of its peak by the
    # end (8 ms; the slowest of the 4 is at 6%, still decaying), the median below 1%; the motion stays
    # bounded and finite
    assert np.all(np.isfinite(sim.states))
    peak = E.max(axis=1)
    assert np.all(np.argmax(E, axis=1) < E.shape[1] - 1)
    assert np.all(E[:, -1] < 0.1 * peak)
    assert np.median(E[:, -1] / peak) < 1e-2
    y = sim.states[..., 2 + 2] * sim.nonlinear_model.zpf[0, 1]
    assert np.abs(y).max() < 1e-6


def test_tweezer_duffing_shift_matches_exact_period_and_free_space():
    gas = {'P': 1e-12 * mbar, 'T': 300 * K, 'vth': 500}
    mtrap = physical(1, 3, gas_kwargs=gas, photon_recoil=False, trap_kwargs=[{'Γ_fbx': 0, 'Γ_fby': 0, 'Γ_fbz': 0}])
    trap = mtrap.traps[0]
    a = 200 * nm
    model = NonlinearCavityModel(mtrap, terms=('elements',))
    u0 = np.zeros(model.n)
    u0[2] = a / model.zpf[0, 0]
    Δt = 0.1 * μs
    sim = CavitySimulation(mtrap, model='nonlinear', terms=('elements',), num_samples=10_000, Δt=Δt, seed=1,
                           initial_state=u0, store_state=True, detection=[Position('x1')], backend='numpy')
    x = sim.absolute_states[0, :, 2] * model.zpf[0, 0]

    def frequency(x, dt):
        sign = np.signbit(x)
        i = np.where(sign[1:] != sign[:-1])[0]
        tc = (i + x[i] / (x[i] - x[i + 1])) * dt
        return (len(tc) - 1) / 2 / (tc[-1] - tc[0])

    m, w = trap.bead.m, trap.tweezer.wx0
    U0 = np.real(trap.α_eff) / 4 * trap.tweezer.E0**2
    U = lambda x: -U0 * np.exp(-2 * x**2 / w**2)
    period = 4 * quad(lambda θ: a * np.cos(θ) / np.sqrt(2 * (U(a) - U(a * np.sin(θ))) / m), 0, np.pi / 2, limit=200)[0]
    f_sim = frequency(x, Δt)
    assert f_sim == pytest.approx(1 / period, rel=2e-5)
    assert trap.Ωj['x'] / (2 * np.pi) - f_sim > 5e3       # a 5.4 kHz softening at 200 nm
    # FreeSpaceSimulation of the same trap converges to the same frequency (its Kasdin RK4 needs Δt ≤ 0.02 μs)
    dt = 0.01 * μs
    free = FreeSpaceSimulation(trap=trap, num_samples=100_000, Δt=dt, X0=np.array([a, 0, 0, 0, 0, 0]), seed=1,
                               zero_mean=False, backend='auto')
    assert frequency(free.X[:, 0], dt) == pytest.approx(f_sim, rel=2e-5)


# --------------------------------------------------------------------------------------------------
# backends, integrators, interactions, records, arguments
# --------------------------------------------------------------------------------------------------
@pytest.mark.skipif(not numba_available(), reason='needs numba')
@pytest.mark.parametrize('stage', ['rk4', 'rk2'])
def test_numpy_and_numba_agree(off_node, stage):
    def kw():
        return {'model': 'nonlinear', 'num_samples': 1500, 'Δt': 0.1 * μs, 'seed': 3, 'store_state': True,
                'num_trajectories': 3, 'substeps': 2, 'nonlinear_stage': stage,
                'detection': [Homodyne(), PhotonNumber(), Position('z1')]}
    a = CavitySimulation(off_node, backend='numpy', **kw())
    b = CavitySimulation(off_node, backend='numba', **kw())
    assert (a.backend_used, b.backend_used) == ('numpy', 'numba')
    np.testing.assert_allclose(b.states, a.states, rtol=1e-9, atol=1e-11 * np.abs(a.states).max())
    for label in a.records:
        np.testing.assert_allclose(b.records[label], a.records[label], rtol=1e-9, atol=1e-11 * np.abs(a.records[label]).max())


def test_heun_cross_check_and_seeding():
    mtrap = cold(3)

    def kw():
        return {'model': 'nonlinear', 'num_samples': 2**12, 'Δt': 0.1 * μs, 'num_trajectories': 8, 'backend': 'numpy',
                'detection': [Position('x1')]}
    split = CavitySimulation(mtrap, seed=5, **kw())
    again = CavitySimulation(mtrap, seed=5, **kw())
    assert np.array_equal(split.records['x1'], again.records['x1'])
    heun = CavitySimulation(mtrap, seed=6, integrator='heun', substeps=8, **kw())
    assert 'Heun' in heun.RK_method
    v_split, v_heun = np.var(split.records['x1']), np.var(heun.records['x1'])
    assert v_heun == pytest.approx(v_split, rel=0.25)


def test_explicit_numba_without_a_compiled_kernel_raises():
    from levisim.simulation.backends import UnsupportedBackendError
    kw = {'model': 'nonlinear', 'num_samples': 64, 'Δt': 0.1 * μs, 'seed': 1, 'detection': [Position('x1')]}
    with pytest.raises(UnsupportedBackendError, match='heun'):
        CavitySimulation(cold(3), integrator='heun', backend='numba', **kw)
    # backend='auto' runs the same configuration with NumPy
    assert CavitySimulation(cold(3), integrator='heun', backend='auto', **kw).backend_used == 'numpy'


class SpringStub:
    """A stand-in Interaction (the round-5 interface): a linear spring between particles 1 and 2"""

    def __init__(self, k, rest):
        self.k, self.rest = k, rest

    def forces(self, positions, particles, t=0.0):
        d = positions[1] - positions[0]
        F = self.k * (np.linalg.norm(d) - self.rest) * d / np.linalg.norm(d)
        return np.array([F, -F])

    def stiffness_blocks(self, positions0, particles):
        raise NotImplementedError


def test_pair_interactions_enter_the_drift_and_the_jacobian():
    mtrap = physical(2, 2)
    trap_positions = np.array([[0, 0, 0], [5e-6, 0, 0]])
    k = 1e-7
    model = NonlinearCavityModel(mtrap, interactions=[SpringStub(k, rest=5e-6)], trap_positions=trap_positions,
                                 damping='symmetric')
    assert not model.compilable
    # at rest length no static force: the equilibrium is unchanged, and the spring adds k between x1 and x2
    plain = NonlinearCavityModel(mtrap, damping='symmetric')
    J = model.jacobian - plain.jacobian
    iP1, iX2 = 2 + 1, 2 + 6
    expected = k * model.zpf[1, 0] * model.parameters[0, 3]      # dP_x1/dX_x2 = k·zpf_x2/(m Ω zpf_x1)
    assert J[iP1, iX2] == pytest.approx(expected, rel=1e-6)
    # a spring shorter than the trap separation pulls the beads towards each other
    stretched = NonlinearCavityModel(mtrap, interactions=[SpringStub(k, rest=4.9e-6)], trap_positions=trap_positions)
    assert stretched.r_eq[0, 0] > 0 > stretched.r_eq[1, 0]
    sim = CavitySimulation(mtrap, model='nonlinear', interactions=[SpringStub(k, rest=5e-6)], trap_positions=trap_positions,
                           num_samples=50, seed=1, backend='auto', detection=[Homodyne()], initial_state='equilibrium')
    assert sim.backend_used == 'numpy'


def test_records_and_state_conventions(off_node):
    sim = CavitySimulation(off_node, model='nonlinear', num_samples=200, Δt=0.1 * μs, seed=1, store_state=True,
                           detection=[Homodyne(), PhotonNumber(), Position('z1')])
    model = sim.nonlinear_model
    assert sim.system.labels == ['cavity', 'x1', 'y1', 'z1']
    assert sim.static_field('cavity') == model.α_eq
    assert sim.detection[1].α_used == model.α_eq
    assert sim.zero_point_fluctuation('z1') == model.zpf[0, 2]
    np.testing.assert_array_equal(sim.absolute_states, sim.states + model.u_eq)
    # default detection: homodyne plus every axis of every particle (MultiTrap models only x, y here)
    default = physical(1, 2).simulate(model='nonlinear', num_samples=10, seed=1)
    assert list(default.records) == ['homodyne', 'x1', 'y1', 'z1']
    # theory spectra of the records are those of the linearised model
    f = np.array([10e3, 150e3])
    assert np.all(sim.record_spectrum('homodyne', f) > 0)


def test_argument_checks(off_node):
    toy = MultiTrap(num_traps=1, Cartesian_axes_per_trap=2, toy_params=shelf.PHASE_EQUAL_toy_params)
    with pytest.raises(ValueError, match='physical parameters'):
        toy.simulate(model='nonlinear', num_samples=2)
    with pytest.raises(TypeError, match='physical parameters'):
        CavitySimulation(QuadratureSystem.from_multitrap(off_node), model='nonlinear', num_samples=2)
    with pytest.raises(ValueError, match='damping'):
        off_node.simulate(model='nonlinear', num_samples=2, damping='viscous')
    with pytest.raises(ValueError, match='unknown terms'):
        off_node.simulate(model='nonlinear', num_samples=2, terms=('tweezer',))
    with pytest.raises(ValueError, match='integrator'):
        off_node.simulate(model='nonlinear', num_samples=2, integrator='euler')
    with pytest.raises(ValueError, match='nonlinear_stage'):
        off_node.simulate(model='nonlinear', num_samples=2, nonlinear_stage='rk3')
    with pytest.raises(ValueError, match='substeps'):
        off_node.simulate(model='nonlinear', num_samples=2, substeps=0)
    with pytest.raises(TypeError, match="only to model='nonlinear'"):
        off_node.simulate(num_samples=2, damping='velocity')
    with pytest.raises(ValueError, match='equilibrium'):
        off_node.simulate(model='nonlinear', num_samples=2, initial_state='centre')
    assert set(NONLINEAR_TERMS) == {'elements', 'coherent_scattering', 'cavity', 'interactions'}
    # unstable linearisation refuses 'steady'
    with pytest.raises(ValueError, match='unstable'):
        physical(2, 2).simulate(model='nonlinear', num_samples=2)
    assert physical(2, 2, φn_dict={2: 30 * degrees}).simulate(model='nonlinear', num_samples=2, seed=0).records

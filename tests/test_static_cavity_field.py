"""
The static cavity field of MultiTrap and its consistency with the nonlinear cavity model

- static_cavity_phases: the static field sums the drives with the tweezer phases e^{iφn}, and each
  bead sees it in its own tweezer's frame (bit-identical with equal phases and at the node)
- cavity_dispersive_coupling: the light-mechanical coupling of the cavity's own standing wave and the
  beads' pull of the cavity frequency
- cavity_envelope: the curvature of the cavity mode's transverse envelope
- cavity_effective_polarisability: Re α_eff in every cavity term
- override_Ω_dict: measured frequencies get neither the static cavity shift nor the interactions'
  self-stiffness

The reference for the first three is the Jacobian of the nonlinear drift (NonlinearCavityModel,
damping='symmetric') evaluated at MultiTrap's static field and the trap centres, which must equal
MultiTrap's drift matrix to round-off.
"""
import numpy as np
import pytest

from levisim.core.constants import ħ, π, ε0
from levisim.core.units import degrees, kHz, nm, μm
from levisim.multiphysics.covariance import quadrature_drift_matrix
from levisim.multiphysics.enums import QuantumLinearTheoryMethod
from levisim.multiphysics.multitrap import MultiTrap
from levisim.multiphysics.utils.shelf import DEFAULT_cavity_kwargs, DEFAULT_gas_kwargs
from levisim.physics import coherent_scattering as cs
from levisim.physics.bead import Bead
from levisim.physics.interactions import Coulomb
from levisim.simulation.backends import numba_available
from levisim.simulation.nonlinear_cavity import NonlinearCavityModel

LAM = QuantumLinearTheoryMethod.LINEAR_AMPLIFIER_MODEL
ω_RANGE = np.arange(100, 330, 1.0) * kHz
OPT_OUTS = ('static_cavity_phases', 'cavity_dispersive_coupling', 'cavity_envelope', 'cavity_effective_polarisability')


def build(Φs, φs=None, axes=3, **kwargs):
    beads = [Bead(r=(60.1 + 0.1*n)*nm, Φ=Φ) for n, Φ in enumerate(Φs)]
    if φs is not None:
        kwargs['φn_dict'] = {n + 1: φ for n, φ in enumerate(φs)}
    return MultiTrap(num_traps=len(Φs), Cartesian_axes_per_trap=axes, beads=beads, ω_range=ω_RANGE,
                     cavity_kwargs=DEFAULT_cavity_kwargs, gas_kwargs=DEFAULT_gas_kwargs, **kwargs)


def jacobian_at_the_trap_centres(mtrap, model):
    """The Jacobian of the nonlinear drift at MultiTrap's static field and r = 0, modelled block"""
    U0 = np.zeros(model.n)
    U0[0], U0[1] = mtrap.cavity.A0, mtrap.cavity.P0
    J = model.L0 + model.term_jacobian(U0, model.terms)
    idx = model.modelled_indices()
    return J[np.ix_(idx, idx)]


OFF_NODE = {
    '1×3 Φ=π/4': ([π/4], [0.0], 3),
    '2×3 unequal phases': ([π/4, 1.1], [0.0, 70*degrees], 3),
    '3×3 unequal phases': ([0.7, 1.1, 1.3], [0.0, 20*degrees, 95*degrees], 3),
    '2×2 unequal phases': ([π/4, 1.1], [0.0, 70*degrees], 2),
}


# --------------------------------------------------------------------------------------------------
# consistency with the nonlinear model
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('case', list(OFF_NODE))
def test_drift_matrix_is_the_nonlinear_jacobian_off_the_node(case):
    Φs, φs, axes = OFF_NODE[case]
    mtrap = build(Φs, φs, axes)
    model = NonlinearCavityModel(mtrap, damping='symmetric')
    A = quadrature_drift_matrix(mtrap.drift_matrix)
    J = jacobian_at_the_trap_centres(mtrap, model)
    # measured 1.5-2.5e-13 (fourth-order central differences of the closed-form forces)
    assert np.abs(J - A).max() <= 1e-11 * np.abs(A).max()
    # MultiTrap's static field is the nonlinear drive's fixed point at the trap centres
    U0 = np.zeros(model.n)
    U0[0], U0[1] = mtrap.cavity.A0, mtrap.cavity.P0
    dα = model.drift(U0)[:2]
    assert np.abs(dα).max() <= 1e-12 * mtrap.cavity.κ * np.abs(U0[:2]).max()


@pytest.mark.parametrize('opt_out', ['static_cavity_phases', 'cavity_dispersive_coupling'])
def test_each_opt_out_leaves_out_a_term_of_the_nonlinear_model(opt_out):
    """Without the phases (unequal phases) or the dispersive terms MultiTrap differs from the nonlinear model"""
    Φs, φs, axes = OFF_NODE['2×3 unequal phases']
    reference = NonlinearCavityModel(build(Φs, φs, axes), damping='symmetric')
    mtrap = build(Φs, φs, axes, **{opt_out: False})
    A = quadrature_drift_matrix(mtrap.drift_matrix)
    J = jacobian_at_the_trap_centres(build(Φs, φs, axes), reference)
    # measured: 2.9e-2 without the phases, 5.3e-3 without the dispersive terms
    assert np.abs(J - A).max() > 1e-3 * np.abs(A).max()


def test_equal_phases_are_bit_identical_in_the_bead_frame():
    # (without the dispersive coupling, which uses the cavity-frame field: with static_cavity_phases=False
    # that frame is not the one of the couplings G = g·e^{iφn} unless φ = 0)
    for φ in (0.0, 30*degrees):
        on = build([π/4, 1.1], [φ, φ], 3, cavity_dispersive_coupling=False)
        off = build([π/4, 1.1], [φ, φ], 3, static_cavity_phases=False, cavity_dispersive_coupling=False)
        assert on.static_bead_frame_amplitudes == off.static_bead_frame_amplitudes
        assert np.array_equal(on.drift_matrix, off.drift_matrix)
        assert np.array_equal(on.mechanical_coupling_matrix, off.mechanical_coupling_matrix)
        assert [d.Ω for d in on.dimensions.values()] == [d.Ω for d in off.dimensions.values()]
        assert np.array_equal(on.homodyne_spectrum(LAM), off.homodyne_spectrum(LAM))
        # the cavity-frame field is rotated by the common phase
        α_on = complex(on.cavity.αR, on.cavity.αI)
        α_off = complex(off.cavity.αR, off.cavity.αI)
        assert α_on == pytest.approx(α_off * np.exp(1j * φ), rel=1e-14)


@pytest.mark.parametrize('system', [(1, 2), (2, 2), (2, 3), (3, 2)])
def test_at_the_node_every_correction_leaves_the_spectra_bit_identical(system):
    num_traps, axes = system
    kwargs = dict(num_traps=num_traps, Cartesian_axes_per_trap=axes, ω_range=ω_RANGE, cavity_effective_polarisability=False)
    new = MultiTrap(**kwargs)
    old = MultiTrap(**kwargs, static_cavity_phases=False, cavity_dispersive_coupling=False, cavity_envelope=False)
    assert new.cavity.dispersive_shift == 0.0 or abs(new.cavity.dispersive_shift) < 1e-25 * new.cavity.κ
    for label in new.labels:
        assert new.dimensions[label].Ω == old.dimensions[label].Ω
        assert new.dimensions[label].G == old.dimensions[label].G
    assert np.array_equal(new.homodyne_spectrum(LAM), old.homodyne_spectrum(LAM))
    assert np.array_equal(new.all_position_power_spectra(LAM)[0], old.all_position_power_spectra(LAM)[0])
    # the drift matrices differ only in entries at the round-off of the node's static field (cos(π/2) ≈ 6e-17)
    assert np.abs(new.drift_matrix - old.drift_matrix).max() <= 1e-17 * np.abs(old.drift_matrix).max()


# --------------------------------------------------------------------------------------------------
# the dispersive terms
# --------------------------------------------------------------------------------------------------
def test_the_pull_of_the_cavity_frequency():
    mtrap = build([π/4, 1.1])
    δ = [cs.cavity_mode_polarisability(trap) * mtrap.cavity.E0**2 / ħ for trap in mtrap.traps]
    expected = sum(δn * np.cos(trap.bead.Φ)**2 for δn, trap in zip(δ, mtrap.traps))
    assert mtrap.cavity.dispersive_shift == pytest.approx(expected, rel=1e-14)
    assert mtrap.cavity.Δ_eff == mtrap.cavity.Δ + mtrap.cavity.dispersive_shift
    assert mtrap.drift_matrix[0, 0] == 1j * mtrap.cavity.Δ_eff - mtrap.cavity.κ / 2
    # δ/2π ≈ 3.4 kHz for a 60 nm bead in levisim's default cavity
    assert δ[0] / (2*π) == pytest.approx(3.37e3, rel=1e-2)
    ω = 120 * kHz
    assert mtrap.cavity.χC(ω) == 1 / (-1j * (ω + mtrap.cavity.Δ_eff) + mtrap.cavity.κ / 2)
    off = build([π/4, 1.1], cavity_dispersive_coupling=False)
    assert off.cavity.dispersive_shift == 0.0 and off.cavity.Δ_eff == off.cavity.Δ
    assert all(dim.dispersive_G == 0 for dim in off.dimensions.values())


def test_dispersive_coupling_is_along_the_cavity_axis_with_the_phase_of_the_static_field():
    mtrap = build([0.6], [0.0], 3)
    α = complex(mtrap.cavity.αR, mtrap.cavity.αI)
    x, y, z = (mtrap.dimensions[f'{axis}1'] for axis in 'xyz')
    θ = mtrap.traps[0].drive_tweezer.θ
    assert z.dispersive_G == 0
    assert abs((x.dispersive_G / α).imag) < 1e-12 * abs(x.dispersive_G / α)  # ∝ ᾱ
    assert (x.dispersive_G / x.zpf) / (y.dispersive_G / y.zpf) == pytest.approx(np.tan(θ), rel=1e-12)
    # ~1e-3 of the coherent-scattering coupling
    assert 1e-4 < abs(x.dispersive_G) / abs(x.G) < 1e-2


def test_librational_dispersive_coupling_against_finite_differences():
    """G = (E0²/ħ)·cos²Φ·∂c·θ_zpf·ᾱ from U_cav = -E0²·|α|²·cos²Φ·c(θ), c = e_c·α(θ)·e_c"""
    from scipy.linalg import expm

    from levisim.physics import Spheroid, Tweezer
    from levisim.physics.particle import body_axis_to_quaternion
    from levisim.physics.rotation import GENERATORS

    def rotate(tensor, θ):
        R = expm(np.einsum('j,jab->ab', θ, GENERATORS))
        return R @ tensor @ R.T
    particle = Spheroid(a=70*nm, b=50*nm, orientation=body_axis_to_quaternion((0, 1, 0)), Φ=0.6)
    mtrap = MultiTrap(num_traps=1, Cartesian_axes_per_trap=('x', 'y', 'z', 'α', 'γ'), beads=[particle],
                      tweezers=[Tweezer(θ=60*degrees)], ω_range=ω_RANGE)
    trap = mtrap.traps[0]
    e_c = cs.cavity_polarisation(trap.drive_tweezer.θ)
    α_lab = cs._lab_polarisability(trap)
    E0, Φ = mtrap.cavity.E0, particle.Φ
    ᾱ = complex(mtrap.cavity.αR, mtrap.cavity.αI)

    def U_cav(θ, α):
        return -E0**2 * abs(α)**2 * np.cos(Φ)**2 * (e_c @ rotate(α_lab, θ) @ e_c)
    h = 1e-6
    for axis, j in (('γ', 2),):
        dim = mtrap.dimensions[f'{axis}1']
        eθ = np.eye(3)[j] * h
        # ∂²U/∂θ∂A with α = ᾱ + δA/2 (central differences; U is quadratic in α)
        d2 = ((U_cav(eθ, ᾱ + 0.5) - U_cav(-eθ, ᾱ + 0.5)) - (U_cav(eθ, ᾱ - 0.5) - U_cav(-eθ, ᾱ - 0.5))) / (2 * h * 2)
        assert -d2 * dim.zpf / ħ == pytest.approx(np.real(dim.dispersive_G), rel=1e-6)


# --------------------------------------------------------------------------------------------------
# the cavity mode envelope
# --------------------------------------------------------------------------------------------------
def fd_hessian(f, h):
    H = np.zeros((3, 3))
    e = np.eye(3)
    for i in range(3):
        for j in range(3):
            a, b = e[i] * h[i], e[j] * h[j]
            H[i, j] = (f(a + b) - f(a - b) - f(-a + b) + f(-a - b)) / (4 * h[i] * h[j])
    return H


def test_envelope_hessian_force_and_drive_against_finite_differences():
    mtrap = build([0.6, 1.1], [0.0, 40*degrees])
    trap, E0, w = mtrap.traps[0], mtrap.cavity.E0, mtrap.cavity.waist
    α = mtrap.static_bead_frame_amplitudes[1]
    terms = ('coherent_scattering', 'cavity')
    k = trap.drive_tweezer.k
    h = np.array([3e-3 / k] * 3)
    # the envelope part of the static Hessian: the difference of the full Hessians with and without it
    H = lambda waist: fd_hessian(lambda r: cs.optical_potential(r, trap, E0, α, terms=terms, cavity_waist=waist), h)
    K_env = cs.static_cavity_hessian(trap, E0, α, cavity_waist=w) - cs.static_cavity_hessian(trap, E0, α)
    np.testing.assert_allclose(H(w) - H(None), K_env, rtol=0, atol=2e-3 * np.abs(K_env).max())
    assert np.abs(K_env).max() < 1e-4 * np.abs(cs.static_cavity_hessian(trap, E0, α)).max()
    d = cs.cavity_axis(trap.drive_tweezer.θ)
    assert abs(d @ K_env @ d) < 1e-12 * np.abs(K_env).max()  # transverse to the cavity axis
    # the force with the envelope, against central differences of the potential, away from the centre
    r0 = np.array([40*nm, -25*nm, 60*nm])
    e = np.eye(3)
    for waist in (w, 3*μm):
        U = lambda r: cs.optical_potential(r, trap, E0, α, terms=terms, cavity_waist=waist)
        F_fd = np.array([-(U(r0 + 1e-11 * e[i]) - U(r0 - 1e-11 * e[i])) / 2e-11 for i in range(3)])
        F = cs.optical_force(r0, trap, E0, α, terms=terms, cavity_waist=waist)
        np.testing.assert_allclose(F, F_fd, rtol=0, atol=1e-6 * np.abs(F).max())
        # the drive is -(i/ħ)·∂U/∂α* (∂/∂α* = (∂/∂αR + i∂/∂αI)/2), for the bead-frame amplitude (φ = 0)
        ε = 1e-3
        dU_dR = (cs.optical_potential(r0, trap, E0, α + ε, terms=terms, cavity_waist=waist)
                 - cs.optical_potential(r0, trap, E0, α - ε, terms=terms, cavity_waist=waist)) / (2 * ε)
        dU_dI = (cs.optical_potential(r0, trap, E0, α + 1j*ε, terms=terms, cavity_waist=waist)
                 - cs.optical_potential(r0, trap, E0, α - 1j*ε, terms=terms, cavity_waist=waist)) / (2 * ε)
        drive = cs.cavity_drive(r0, trap, E0, α, terms=terms, cavity_waist=waist)
        assert drive == pytest.approx(-1j / ħ * (dU_dR + 1j * dU_dI) / 2, rel=1e-6)


def test_nonlinear_drift_with_the_envelope_matches_the_closed_forms():
    mtrap = build([0.6, 1.1], [0.0, 40*degrees])
    model = NonlinearCavityModel(mtrap)
    assert model.cavity_waist == mtrap.cavity.waist
    assert NonlinearCavityModel(build([0.6, 1.1], [0.0, 40*degrees], cavity_envelope=False)).cavity_waist is None
    rng = np.random.default_rng(3)
    U = model.u_eq + rng.normal(scale=300.0, size=model.n)
    r = model.positions(U[None])[0]
    α = complex(U[0], U[1]) / 2
    out = model.term_drift(U[None], ('coherent_scattering', 'cavity'))[0]
    dα = 0
    for n, trap in enumerate(mtrap.traps):
        φ = trap.tweezer.φ
        F = cs.optical_force(r[n], trap, mtrap.cavity.E0, α * np.exp(-1j * φ), terms=('coherent_scattering', 'cavity'),
                             cavity_waist=mtrap.cavity.waist)
        np.testing.assert_allclose(out[3 + 6*n:3 + 6*n + 6:2], F * model.parameters[n, 3:6], rtol=1e-10)
        dα = dα + cs.cavity_drive(r[n], trap, mtrap.cavity.E0, α, φ=φ, cavity_waist=mtrap.cavity.waist)
    assert complex(out[0], out[1]) / 2 == pytest.approx(complex(dα), rel=1e-10)
    if numba_available():
        from levisim.simulation.backends.numba_nonlinear import _term_drift
        flags = np.array([False, True, True, False])
        compiled = np.zeros(model.n)
        _term_drift(np.ascontiguousarray(U), np.ascontiguousarray(model.parameters), flags,
                    model.compiled_extra_arguments(), compiled)
        np.testing.assert_allclose(compiled, out, rtol=1e-12, atol=1e-12 * np.abs(out).max())


# --------------------------------------------------------------------------------------------------
# the polarisability
# --------------------------------------------------------------------------------------------------
def test_cavity_terms_use_the_radiation_reaction_corrected_polarisability():
    mtrap = build([π/4])
    trap = mtrap.traps[0]
    α0, k = trap.bead.α, trap.drive_tweezer.k
    assert cs.drive_polarisability(trap) == np.real(trap.α_eff) == cs.cavity_mode_polarisability(trap)
    assert cs.drive_polarisability(trap, effective_polarisability=False) == α0
    # Re α_eff = α/(1 + (k³α/6πε0)²): 5e-5 smaller for a 60 nm bead at 1064 nm
    x = k**3 * α0 / (6 * π * ε0)
    assert np.real(trap.α_eff) / α0 - 1 == pytest.approx(-x**2 / (1 + x**2), rel=1e-6)
    assert 4e-5 < x**2 < 6e-5
    static = build([π/4], cavity_effective_polarisability=False)
    assert trap.Ed / static.traps[0].Ed == pytest.approx(np.real(trap.α_eff) / α0, rel=1e-14)


# --------------------------------------------------------------------------------------------------
# measured frequencies
# --------------------------------------------------------------------------------------------------
def test_measured_frequencies_get_neither_static_shift():
    Ω = {'x': 150*kHz, 'y': 130*kHz, 'z': 40*kHz}
    beads = [Bead(r=60.1*nm, Φ=0.7, charge=300), Bead(r=60.2*nm, Φ=1.1, charge=-200)]
    kwargs = dict(num_traps=2, Cartesian_axes_per_trap=3, beads=beads, ω_range=ω_RANGE, interactions=[Coulomb()],
                  trap_positions=[(0, 0, 0), (0, 8*μm, 0)], trap_kwargs=[{'override_Ω_dict': Ω}, {}])
    mtrap = MultiTrap(**kwargs)
    for axis in 'xyz':
        assert mtrap.dimensions[f'{axis}1'].Ω == Ω[axis]
    # the computed frequencies of trap 2 are shifted, and the pair couplings are added
    assert mtrap.dimensions['y2'].Ω != mtrap.traps[1].Ωj['y']
    assert mtrap.mechanical_coupling_matrix[mtrap.labels.index('y1'), mtrap.labels.index('y2')] != 0
    old = MultiTrap(**kwargs, interaction_frequency_shifts='all')
    K = old.interaction_stiffness_blocks[0, 0]
    for i, axis in enumerate('xyz'):
        assert old.dimensions[f'{axis}1'].Ω == pytest.approx(Ω[axis] * np.sqrt(1 + K[i, i] / (beads[0].m * Ω[axis]**2)), rel=1e-12)
    assert old.dimensions['y2'].Ω == mtrap.dimensions['y2'].Ω
    with pytest.raises(ValueError, match="'all'"):
        MultiTrap(**kwargs, interaction_frequency_shifts='measured')


# --------------------------------------------------------------------------------------------------
# dropped light-coupled axes
# --------------------------------------------------------------------------------------------------
def capture_warnings(make):
    from levisim.core.logger import logger
    captured = []
    handler_id = logger.add(lambda message: captured.append(message.record['message']), level='WARNING', format='{message}')
    try:
        result = make()
    finally:
        logger.remove(handler_id)
    return result, [message for message in captured if 'are left out' in message]


def test_dropping_a_light_coupled_axis_warns():
    two, warnings = capture_warnings(lambda: build([π/4], axes=2))
    assert two.dropped_light_coupled_axes == ('z1',) and len(warnings) == 1
    # the couplings it reports are those of the axis in the 3-axis model
    three, warnings3 = capture_warnings(lambda: build([π/4], axes=3))
    assert not warnings3 and three.dropped_light_coupled_axes == ()
    G, g = two.dropped_axis_couplings()['z1']
    z = three.dimensions['z1']
    assert G == pytest.approx(abs(z.G), rel=1e-12)
    assert g == pytest.approx(max(abs(three.dimensions['x1'].gxz), abs(three.dimensions['x1'].gyz)), rel=1e-12)
    # at the node z is decoupled (g_z ∝ cos Φ, g_xz ∝ P0): no warning
    node, warnings = capture_warnings(lambda: build([π/2], axes=2))
    assert not warnings and node.dropped_light_coupled_axes == ()
    # silenced on request
    _, warnings = capture_warnings(lambda: build([π/4], axes=2, dropped_axis_warning=False))
    assert not warnings


@pytest.mark.skipif(not numba_available(), reason='needs numba')
def test_compiled_drift_with_envelope_interactions_and_elements_matches_numpy():
    """The envelope terms in the compiled kernel together with its pair forces and force elements"""
    from levisim.simulation.backends.numba_nonlinear import _term_drift
    beads = [Bead(r=60.1*nm, Φ=0.7, charge=10), Bead(r=60.2*nm, Φ=1.1, charge=-10)]
    mtrap = MultiTrap(num_traps=2, Cartesian_axes_per_trap=3, beads=beads, ω_range=ω_RANGE, interactions=[Coulomb()],
                      trap_positions=[(0, 0, 0), (0, 4*μm, 0)], φn_dict={2: 40*degrees},
                      cavity_kwargs=DEFAULT_cavity_kwargs, gas_kwargs=DEFAULT_gas_kwargs)
    model = NonlinearCavityModel(mtrap, interactions=mtrap.interactions)
    assert model.cavity_waist == mtrap.cavity.waist and model.compilable
    rng = np.random.default_rng(5)
    U = model.u_eq + rng.normal(scale=500.0, size=model.n)
    terms = ('elements', 'coherent_scattering', 'cavity', 'interactions')
    reference = model.term_drift(U[None], terms)[0]
    compiled = np.zeros(model.n)
    _term_drift(np.ascontiguousarray(U), np.ascontiguousarray(model.parameters), np.array([True] * 4),
                model.compiled_extra_arguments(), compiled)
    np.testing.assert_allclose(compiled, reference, rtol=1e-11, atol=1e-11 * np.abs(reference).max())


def test_floquet_paul_frequencies_fold_in_the_corrected_static_hessian():
    """A hybrid Paul trap off the node: its exact (Floquet) frequencies include the static cavity Hessian
    with the phases, the envelope and Re α_eff; measured frequencies (decision 8) bypass both"""
    from levisim.core.units import Hz, mbar, mm
    from levisim.physics.fields import PaulTrap
    from levisim.physics.trap import Trap
    from levisim.physics.tweezer import Tweezer
    bead = Bead(charge=2000, Φ=π/4)
    tweezer = Tweezer(Power=4e-3, θ=45*degrees)   # (θ = 0 scatters nothing into the cavity)
    f_rf = 300e3
    V = 0.2 * bead.m * (1*mm)**2 * (f_rf*Hz)**2 / (4 * bead.q)
    paul = PaulTrap.ring(V, f_rf*Hz, r0=1*mm, axis='x')
    common = dict(tweezers=[tweezer], beads=[bead], Cartesian_axes_per_trap=3, gas_kwargs={'P': 1e-9*mbar},
                  cavity_kwargs=DEFAULT_cavity_kwargs)
    mt = MultiTrap(trap_kwargs=[{'extra_fields': [paul], 'Γ_fbx': 0.0, 'Γ_fby': 0.0}], **common)
    trap = mt.traps[0]
    assert mt.floquet_paul_traps == {1}
    K = cs.static_cavity_hessian(trap, mt.cavity.E0, mt.static_bead_frame_amplitudes[1], cavity_waist=mt.cavity.waist)
    np.testing.assert_array_equal(mt.static_cavity_hessians[1], K)
    assert np.abs(np.diag(K)).max() > 0
    K_tw = np.diag(Trap(tweezer=tweezer, bead=bead).stiffness_tensor)
    exact = paul.secular_frequencies(bead, 'exact', extra_stiffness=K_tw + np.diag(K))
    for j, axis in enumerate('xyz'):
        assert mt.dimensions[f'{axis}1'].Ω == pytest.approx(exact[j], rel=1e-10)
    # without the envelope the a_j and so the frequencies change (by the envelope's ~1e-5 of K)
    plane = MultiTrap(trap_kwargs=[{'extra_fields': [paul], 'Γ_fbx': 0.0, 'Γ_fby': 0.0}], cavity_envelope=False, **common)
    assert plane.dimensions['z1'].Ω != mt.dimensions['z1'].Ω
    # measured frequencies: neither the Floquet frequencies nor any static shift
    Ω = {'x': 2*π*20e3, 'y': 2*π*25e3, 'z': 2*π*8e3}
    measured = MultiTrap(trap_kwargs=[{'extra_fields': [paul], 'override_Ω_dict': Ω}], **common)
    assert measured.floquet_paul_traps == set()
    assert [measured.dimensions[f'{axis}1'].Ω for axis in 'xyz'] == [Ω[axis] for axis in 'xyz']
    # the rf force is time-dependent: no static equilibrium to linearise at
    with pytest.raises(NotImplementedError, match='PaulTrap'):
        MultiTrap.from_nonlinear_equilibrium(trap_kwargs=[{'extra_fields': [paul]}], **common)

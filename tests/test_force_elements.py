"""
Force elements (levisim.physics.fields) and the multi-field Trap

- the Tweezer element reproduces Trap.Ftw / Utw / F_scattering bit for bit
- stiffness tensors: closed form vs central differences, sums of elements, off-diagonal couplings
- optical-only quantities raise without an optical element
- a trap made of a HarmonicPotential only simulates with the expected Ω and equipartition
- gravity sag g/Ω², the gravity keyword vs a Gravity element
- the shorthand Trap(tweezer=, bead=) equals the explicit fields list, number for number
"""

import pickle

import numpy as np
import pytest
from scipy.signal import welch

from levisim.core.constants import g, kB, ħ
from levisim.core.units import Hz, kHz, mbar, ms, nm
from levisim.physics.bead import Bead
from levisim.physics.fields import (
    Feedback,
    ForceElement,
    Gravity,
    HarmonicPotential,
    PhotonRecoil,
)
from levisim.physics.gas import Gas
from levisim.physics.kick import Kick
from levisim.physics.trap import MissingOpticalFieldError, Trap
from levisim.physics.tweezer import Tweezer
from levisim.simulation import FreeSpaceSimulation

POINTS = [
    (0.0, 0.0, 0.0),
    (10*nm, -20*nm, 30*nm),
    (-50*nm, 40*nm, -100*nm),
    (300*nm, 250*nm, 900*nm),
]
TWEEZERS = [Tweezer(), Tweezer(λ=1550*nm, Power=0.3, wx0=1.1e-6, wy0=1.2e-6)]
BEADS = [Bead(), Bead(r=75*nm, ρ=2200.0), Bead(permittivity_model='Malitson')]


# --------------------------------------------------------------------------------------------------
# the tweezer element is Trap's tweezer arithmetic
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('tweezer', TWEEZERS)
@pytest.mark.parametrize('bead', BEADS)
def test_tweezer_force_is_bit_identical_to_trap_Ftw(tweezer, bead):
    trap = Trap(tweezer=tweezer, bead=bead)
    for point in POINTS:
        F = tweezer.force(point, 0.0, bead)
        assert np.array_equal(F, np.array(trap.Ftw(*point)))
        assert np.array_equal(F, np.array([trap.Ftwx(*point), trap.Ftwy(*point), trap.Ftwz(*point)]))
        assert tweezer.potential(point, bead) == trap.Utw(*point)
        assert PhotonRecoil(tweezer).force(point, 0.0, bead)[2] == trap.F_scattering(*point)
    assert tweezer.effective_polarisability(bead) == trap.α_eff


@pytest.mark.parametrize('tweezer', TWEEZERS)
def test_tweezer_stiffness_closed_form_matches_finite_differences(tweezer):
    bead = Bead()
    closed = tweezer.stiffness(None, bead)
    numerical = ForceElement.stiffness(tweezer, None, bead)  # central differences of force
    assert np.allclose(np.diag(numerical), np.diag(closed), rtol=1e-6, atol=0)
    assert np.all(np.abs(numerical - np.diag(np.diag(numerical))) <= 1e-9 * np.max(closed))
    trap = Trap(tweezer=tweezer, bead=bead)
    assert np.array_equal(trap.stiffness_tensor, closed)
    assert (trap.kx, trap.ky, trap.kz) == tuple(np.diag(closed))


def test_tweezer_stiffness_off_focus_matches_Ωj_at_z_eq():
    """Off the focus the element uses central differences, the same scheme as Trap.Ωj_at_z_eq"""
    trap = Trap(photon_recoil=True)
    r0 = (0.0, 0.0, trap.z_eq_exact)
    K = trap.tweezer.stiffness(r0, trap.bead) + PhotonRecoil(trap.tweezer).stiffness(r0, trap.bead)
    Ω = np.sqrt(np.diag(K) / trap.bead.m)
    expected = np.array([trap.Ωj_at_z_eq[axis] for axis in 'xyz'])
    assert np.allclose(Ω, expected, rtol=1e-9)
    shifted = Trap(photon_recoil=True, linearisation_point='equilibrium')
    assert np.allclose(shifted.linearisation_point, r0, rtol=1e-8, atol=1e-15)
    assert np.allclose([shifted.Ωj[axis] for axis in 'xyz'], expected, rtol=1e-6)


def test_tweezer_noise_is_the_photon_recoil_heating():
    trap = Trap()
    S = trap.tweezer.noise(trap.bead)
    assert np.count_nonzero(S - np.diag(np.diag(S))) == 0
    for index, axis in enumerate('xyz'):
        rate = S[index, index] / (2 * trap.bead.m * ħ * trap.Ωj[axis])
        assert rate == pytest.approx(trap.photon_recoil_heating_rates[axis], rel=1e-12)
    assert np.array_equal(trap.field_noise_psd, S)


# --------------------------------------------------------------------------------------------------
# stiffness tensors add; off-diagonal parts become couplings
# --------------------------------------------------------------------------------------------------
def test_tweezer_and_harmonic_stiffness_tensors_add():
    tweezer, bead = Tweezer(), Bead()
    K_h = np.array([2e-7, 3e-7, 5e-8])
    trap = Trap(fields=[tweezer, HarmonicPotential(K_h)], particle=bead)
    K_tw = tweezer.stiffness(None, bead)
    assert np.array_equal(trap.stiffness_tensor, K_tw + np.diag(K_h))
    for index, axis in enumerate('xyz'):
        assert trap.Ωj[axis] == np.sqrt((K_tw[index, index] + K_h[index]) / bead.m)
    assert all(value == 0 for value in trap.intra_trap_couplings.values())
    # no Feedback element: only gas damping
    assert trap.Γ_fbx == trap.Γ_fby == trap.Γ_fbz == 0.0
    assert trap.Γx == trap.Γth
    # the simulated field force is the sum of both forces
    point = POINTS[2]
    expected = np.array(trap.Ftw(*point)) + HarmonicPotential(K_h).force(point, 0.0, bead)
    assert np.array_equal(np.array(trap.field_force(*point)), expected)


def test_off_diagonal_stiffness_gives_intra_trap_couplings():
    bead = Bead()
    θ = 0.3
    R = np.array([[np.cos(θ), -np.sin(θ), 0], [np.sin(θ), np.cos(θ), 0], [0, 0, 1]])
    K = R @ np.diag([2e-6, 1e-6, 3e-7]) @ R.T
    trap = Trap(fields=[HarmonicPotential(K)], particle=bead)
    assert np.allclose(trap.stiffness_tensor, K, rtol=0, atol=1e-20)
    zpf = {'x': trap.x_zpf, 'y': trap.y_zpf, 'z': trap.z_zpf}
    gxy = -K[0, 1] * zpf['x'] * zpf['y'] / ħ
    assert trap.intra_trap_couplings['xy'] == pytest.approx(gxy, rel=1e-12)
    assert trap.intra_trap_couplings['yx'] == pytest.approx(gxy, rel=1e-12)
    assert trap.intra_trap_couplings['xz'] == 0 and trap.intra_trap_couplings['yz'] == 0
    g_matrix = trap.intra_trap_coupling_matrix
    assert np.allclose(g_matrix, g_matrix.T) and np.all(np.diag(g_matrix) == 0)
    # Ωj come from the diagonal only
    assert trap.Ωx == np.sqrt(K[0, 0] / bead.m)


@pytest.mark.parametrize('element', [
    HarmonicPotential([1e-6, 2e-6, 3e-7], centre=(5*nm, 0, -3*nm)),
    Gravity(),
    Feedback(1*kHz, 2*kHz, 3*kHz),
])
def test_closed_form_stiffness_matches_finite_differences(element):
    bead = Bead()
    for r0 in POINTS:
        numerical = ForceElement.stiffness(element, r0, bead)
        assert np.allclose(numerical, element.stiffness(r0, bead), rtol=1e-6, atol=1e-18)


def test_harmonic_potential_force_is_minus_gradient():
    element = HarmonicPotential(np.array([[2.0, 0.5, 0.0], [0.5, 1.0, 0.1], [0.0, 0.1, 3.0]]) * 1e-6, centre=(1*nm, 2*nm, 0))
    r, h = np.array([30*nm, -20*nm, 10*nm]), 1e-12
    gradient = [(element.potential(r + h*e, None) - element.potential(r - h*e, None)) / (2*h) for e in np.eye(3)]
    assert np.allclose(element.force(r, 0.0, None), -np.array(gradient), rtol=1e-6)
    with pytest.raises(ValueError, match='symmetric'):
        HarmonicPotential([[1, 2, 0], [0, 1, 0], [0, 0, 1]])


# --------------------------------------------------------------------------------------------------
# optical-only quantities
# --------------------------------------------------------------------------------------------------
OPTICAL_ONLY = [
    'tweezer', 'α_eff', 'U_depth', 'v_escape', 'photon_recoil_heating_rates', 'σ_scattering', 'z_eq',
    'z_eq_exact', 'Ωj_at_z_eq', 'σ_Rayleigh', 'max_Ftwx', 'drive_tweezer', 'cavity_drive_field',
]


@pytest.mark.parametrize('name', OPTICAL_ONLY)
def test_optical_only_quantities_raise_without_an_optical_field(name):
    trap = Trap(fields=[HarmonicPotential.from_frequencies([100*kHz, 120*kHz, 40*kHz], Bead().m)])
    with pytest.raises(MissingOpticalFieldError, match='no optical field'):
        getattr(trap, name)
    with pytest.raises(ValueError):
        trap.Ftw(0, 0, 0)


def test_non_optical_trap_has_mechanical_quantities():
    m = Bead().m
    trap = Trap(fields=[HarmonicPotential.from_frequencies([100*kHz, 120*kHz, 40*kHz], m)], Γ_fbx=1*kHz)
    assert not trap.has_optical_field
    assert trap.Ωx == pytest.approx(100*kHz, rel=1e-14)
    assert trap.Γx == trap.Γth + 1*kHz and trap.Γ_fby == 0.0
    for name in ('A', 'T_eff', 'Qx', 'x_zpf', 'equipartition_factor', 'Sth', 'theory'):
        getattr(trap, name)
    assert 'fields = [HarmonicPotential, Feedback]' in trap.info_str
    with pytest.raises(ValueError, match='no optical field'):
        Trap(fields=[HarmonicPotential(1e-6)], photon_recoil=True)
    # a cavity drive can still be given explicitly (design: a non-optical trap scatters nothing)
    drive = Tweezer()
    driven = Trap(fields=[HarmonicPotential(1e-6)], cavity_drive_tweezer=drive)
    assert driven.drive_tweezer is drive


# --------------------------------------------------------------------------------------------------
# construction rules, properties and pickles
# --------------------------------------------------------------------------------------------------
def test_shorthand_builds_tweezer_and_feedback():
    tweezer, kick = Tweezer(), Kick(start=0.1*ms)
    trap = Trap(tweezer=tweezer, photon_recoil=True, kicks=[kick])
    assert [type(element) for element in trap.fields] == [Tweezer, Feedback, PhotonRecoil, Kick]
    assert trap.tweezer is tweezer and trap.fields[2].tweezer is tweezer
    assert (trap.Γ_fbx, trap.Γ_fby, trap.Γ_fbz) == (2.1*kHz, 2.5*kHz, 0.0)
    assert trap.kicks == [kick] and trap.photon_recoil


def test_construction_errors():
    with pytest.raises(ValueError, match='inside fields'):
        Trap(fields=[Tweezer()], tweezer=Tweezer())
    with pytest.raises(TypeError, match='ForceElement'):
        Trap(fields=[Tweezer(), 'gravity'])
    with pytest.raises(ValueError, match='Feedback element'):
        Trap(fields=[Tweezer(), Feedback(1.0)], Γ_fbx=2.0)
    with pytest.raises(ValueError, match='linearisation_point'):
        Trap(linearisation_point='focus')


def test_property_setters_edit_the_fields():
    trap = Trap()
    trap.Γ_fbx = 7*kHz
    assert trap.fields[1].Γ_fbx == 7*kHz
    trap.photon_recoil = True
    assert isinstance(trap.fields[-1], PhotonRecoil)
    trap.photon_recoil = False
    assert not trap.fields_of_type(PhotonRecoil)
    kicks = [Kick(start=0.0), Kick(start=1*ms)]
    trap.kicks = kicks
    assert trap.kicks == kicks
    new_tweezer = Tweezer(Power=0.2)
    trap.photon_recoil = True
    trap.tweezer = new_tweezer
    assert trap.fields[0] is new_tweezer and trap.fields_of_type(PhotonRecoil)[0].tweezer is new_tweezer
    bare = Trap(fields=[HarmonicPotential(1e-6)])
    bare.Γ_fbz = 1*kHz
    assert bare.fields_of_type(Feedback)[0].Γ_fbz == 1*kHz


def test_explicit_fields_equal_the_shorthand():
    tweezer, bead = Tweezer(), Bead(r=70*nm)
    short = Trap(tweezer=tweezer, bead=bead, Γ_fbx=3*kHz, Γ_fby=1*kHz, Γ_fbz=0.5*kHz, photon_recoil=True)
    explicit = Trap(fields=[tweezer, Feedback(3*kHz, 1*kHz, 0.5*kHz), PhotonRecoil(tweezer)], particle=bead)
    for name in ('kx', 'ky', 'kz', 'Ωx', 'Ωy', 'Ωz', 'Γx', 'Γy', 'Γz', 'U_depth', 'z_eq'):
        assert getattr(short, name) == getattr(explicit, name), name
    traces = []
    for trap in (short, explicit):
        sim = FreeSpaceSimulation(trap=trap, num_samples=2000, seed=5, backend='numpy')
        sim.simulate_dynamics()
        traces.append(np.concatenate([sim.traces[axis].values for axis in 'xyz']))
    assert np.array_equal(*traces)


def test_legacy_pickle_state_is_converted():
    trap = Trap(photon_recoil=True, kicks=[Kick(start=0.0)], Γ_fbz=1*kHz)
    state = trap.__getstate__()
    fields = state.pop('_fields')
    state.pop('linearisation_point')
    state.update(tweezer=fields[0], Γ_fbx=2.1*kHz, Γ_fby=2.5*kHz, Γ_fbz=1*kHz, photon_recoil=True, kicks=fields[3:])
    restored = Trap.__new__(Trap)
    restored.__setstate__(state)
    assert restored.tweezer is fields[0] and restored.photon_recoil and len(restored.kicks) == 1
    assert restored.Γ_fbz == 1*kHz
    assert pickle.loads(pickle.dumps(trap)).Ωx == trap.Ωx


def test_element_plan_follows_in_place_edits_of_fields():
    trap = Trap()
    assert not trap.photon_recoil
    trap.fields.append(PhotonRecoil(trap.tweezer))
    assert trap.photon_recoil
    trap.fields[0] = Tweezer(Power=0.1)
    assert trap.tweezer is trap.fields[0]
    del trap.fields[0]
    assert not trap.has_optical_field
    trap.fields += [HarmonicPotential(1e-6)]
    assert trap.fields_in_group('field')[0] is trap.fields[-1]
    trap.fields = [Tweezer()]
    assert trap.has_optical_field and not trap.photon_recoil


def test_kick_element_force_matches_trap_kick_force():
    kick = Kick(start=0.1*ms, duration=1e-5, K=50, θ=0.4)
    trap = Trap(kicks=[kick])
    t = 0.105*ms
    Fx, Fy = trap.kick_force(t)
    F = kick.force((0, 0, 0), t, trap.bead, gas=trap.gas)
    assert np.allclose(F, [Fx, Fy, 0.0], rtol=1e-15)
    assert not np.any(kick.force((0, 0, 0), 0.0, trap.bead, gas=trap.gas))
    with pytest.raises(ValueError, match='gas'):
        kick.force((0, 0, 0), t, trap.bead)


# --------------------------------------------------------------------------------------------------
# gravity
# --------------------------------------------------------------------------------------------------
def test_gravity_sag_is_g_over_Ω_squared():
    bead = Bead()
    Ω = np.array([100, 150, 40]) * kHz
    for direction in [(-1, 0, 0), (0, 0, 1), (1, 1, 0)]:
        gravity = Gravity(direction=direction)
        trap = Trap(fields=[HarmonicPotential.from_frequencies(Ω, bead.m), gravity], particle=bead)
        r_eq = trap.find_equilibrium()
        assert np.allclose(r_eq, g * gravity.direction / Ω**2, rtol=1e-9, atol=1e-25)
    # gravity does not change the stiffness
    assert np.array_equal(trap.stiffness_tensor, np.diag(bead.m * Ω**2))


def test_gravity_keyword_equals_gravity_element():
    sims = [
        FreeSpaceSimulation(trap=Trap(), gravity=True, num_samples=1500, seed=2, backend='numpy'),
        FreeSpaceSimulation(trap=Trap(fields=[Tweezer(), Feedback(2.1*kHz, 2.5*kHz), Gravity()]),
                            num_samples=1500, seed=2, backend='numpy'),
    ]
    for sim in sims:
        sim.simulate_dynamics()
    assert np.array_equal(sims[0].F_gravity, sims[1].F_gravity)
    for axis in 'xyz':
        assert np.array_equal(sims[0].traces[axis].values, sims[1].traces[axis].values)
    with pytest.raises(ValueError, match='twice'):
        FreeSpaceSimulation(trap=Trap(fields=[Tweezer(), Gravity()]), gravity=True)


# --------------------------------------------------------------------------------------------------
# simulations of non-tweezer traps
# --------------------------------------------------------------------------------------------------
def test_harmonic_only_trap_simulates_with_expected_Ω_and_equipartition():
    bead = Bead()
    Ω = np.array([200, 160, 120]) * kHz
    trap = Trap(fields=[HarmonicPotential.from_frequencies(Ω, bead.m)], particle=bead,
                gas=Gas(P=1*mbar), β=0.0)
    Δt = 1e-7
    sim = FreeSpaceSimulation(trap=trap, num_samples=200_000, Δt=Δt, seed=21, backend='auto')
    sim.simulate_dynamics()
    assert sim.backend_used == 'numpy'
    for index, axis in enumerate('xyz'):
        x = sim.traces[axis].values
        variance = np.var(x)
        expected = kB * trap.gas.T / (bead.m * Ω[index]**2)
        assert variance == pytest.approx(expected, rel=0.15), axis
        f, S = welch(x, fs=1/Δt, nperseg=2**14)
        assert f[np.argmax(S)] == pytest.approx(Ω[index] / Hz, rel=0.02), axis


def test_numba_falls_back_to_numpy_for_other_elements():
    # skip when numba is missing OR disabled with LEVISIM_DISABLE_NUMBA (importorskip only covers the first)
    from levisim.simulation.backends import numba_available
    if not numba_available():
        pytest.skip('numba is not installed or is disabled')
    from levisim.simulation.backends import (
        AUTO_NUMBA_MIN_STEPS,
        UnsupportedBackendError,
    )
    trap = Trap(fields=[Tweezer(), HarmonicPotential(1e-7), Feedback(2*kHz, 2*kHz)])
    # an explicit backend='numba' never runs another backend: it raises, naming the element
    with pytest.raises(UnsupportedBackendError, match='HarmonicPotential'):
        FreeSpaceSimulation(trap=trap, num_samples=200, seed=1, backend='numba')
    # backend='auto' falls back to the NumPy loop
    sim = FreeSpaceSimulation(trap=trap, num_samples=AUTO_NUMBA_MIN_STEPS, seed=1, backend='auto')
    assert 'HarmonicPotential' in sim._compiled_kernel_unsupported_fields()
    assert sim.backend_used == 'numpy'
    # the standard set, spelled as explicit fields, still compiles and matches its NumPy run
    explicit = Trap(fields=[Tweezer(), Feedback(2.1*kHz, 2.5*kHz), Gravity()], photon_recoil=True)
    runs = {}
    for backend in ('numba', 'numpy'):
        sim = FreeSpaceSimulation(trap=explicit, num_samples=2000, seed=4, backend=backend)
        sim.simulate_dynamics()
        runs[backend] = sim
    assert runs['numba'].backend_used == 'numba'
    assert np.allclose(runs['numba'].x.values, runs['numpy'].x.values, rtol=1e-9, atol=1e-18)


def test_force_element_defaults():
    element = ForceElement()
    with pytest.raises(NotImplementedError):
        element.force((0, 0, 0), 0.0, None)
    with pytest.raises(NotImplementedError):
        element.potential((0, 0, 0), None)
    assert not np.any(element.torque((0, 0, 0), 0.0, None))
    assert not np.any(element.noise(None)) and not np.any(element.damping_rates())
    assert not element.is_optical and Tweezer.is_optical

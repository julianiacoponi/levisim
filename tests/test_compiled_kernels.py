"""
The targeted numba kernels of round 5's follow-up: compiled force elements (PaulTrap, MagneticTrap and
its field sources, HarmonicPotential, UniformElectricField), compiled optical binding, and pair
interactions in the multi-particle free-space and nonlinear cavity kernels

NumPy is the reference: every compiled force is compared with the NumPy `force` it replaces, and
seeded trajectories with backend='numba' with those of backend='numpy' (same random numbers). Every
test here needs numba and is skipped without it (or with LEVISIM_DISABLE_NUMBA=1).
"""

import numpy as np
import pytest

from levisim.core.units import Hz, mbar, mm, nm, μm, μs
from levisim.multiphysics.multitrap import MultiTrap
from levisim.multiphysics.utils import shelf
from levisim.physics import Bead, Gas, OpticalBinding, Trap, Tweezer
from levisim.physics.fields import (
    CallableField,
    CoilPair,
    CuboidMagnet,
    CurrentLoop,
    Gravity,
    HarmonicPotential,
    LocalFieldExpansion,
    MagneticTrap,
    MultipoleField,
    PaulTrap,
    UniformElectricField,
)
from levisim.physics.fields.magnetic_sources import _rotation_to_axis
from levisim.physics.interactions import Casimir, Coulomb, NewtonianGravity
from levisim.physics.materials import BISMUTH, FUSED_SILICA
from levisim.physics.particle import Sphere
from levisim.simulation import (
    CavitySimulation,
    FreeSpaceSimulation,
    Homodyne,
    MultiParticleFreeSpaceSimulation,
    Position,
)
from levisim.simulation.backends import numba_available

pytestmark = pytest.mark.skipif(not numba_available(), reason='numba is not installed (or is disabled)')

CHARGED = Sphere(r=100*nm, charge=500)
SILICA = Sphere(r=1e-6, material=FUSED_SILICA, charge=500)
UHV = Gas(P=1e-9*mbar)
TILT = _rotation_to_axis((0.3, -0.4, 1.0))


def ring(q, f_rf, particle=CHARGED, **kwargs):
    V = q * particle.m * (1*mm)**2 * (f_rf*Hz)**2 / (2 * particle.q)
    return PaulTrap.ring(V, f_rf*Hz, r0=1*mm, **kwargs)


def max_relative_difference(a, b):
    return np.max(np.abs(a - b)) / np.max(np.abs(a))


# --------------------------------------------------------------------------------------------------
# compiled forces against the elements' NumPy forces
# --------------------------------------------------------------------------------------------------
FIELD_CASES = {
    'tweezer and Paul trap': ([Tweezer(), PaulTrap.ring(50.0, 2*np.pi*20e3, r0=1*mm, U_dc=0.3, centre=(1e-7, 0, -2e-7),
                                                        φ_rf=0.3)], 3e-7, 1e-15),
    'local expansion': ([MagneticTrap(LocalFieldExpansion(B0=(0.1, 0, 1), gradient=1e3*np.diag([1.0, 0.5, -1.5])))], 2e-5, 1e-15),
    'tilted multipoles': ([MagneticTrap(MultipoleField({(1, 0): 0.2, (2, 1): 0.05, (3, -2): 0.01}, r0=1*mm, rotation=TILT))],
                          2e-5, 1e-14),
    'tilted current loop': ([MagneticTrap(CurrentLoop(1*mm, 300.0, centre=(0.1*mm, 0, 0.5*mm), axis=(0.2, 0.1, 1)))],
                            2e-5, 1e-12),
    # a central difference of B near the zero of an anti-Helmholtz pair: J is ~1e-12 relative
    'anti-Helmholtz pair': ([MagneticTrap(CoilPair.anti_helmholtz(1*mm, 300.0, axis=(1, 0, 0)))], 2e-5, 3e-11),
    'tilted cuboid': ([MagneticTrap(CuboidMagnet((1*mm, 2*mm, 1.5*mm), magnetisation=(2e5, -3e5, 8e5), centre=(0, 0, -2*mm),
                                                 rotation=TILT))], 2e-5, 1e-14),
    'sum, harmonic, uniform field': ([
        MagneticTrap(CurrentLoop(1*mm, 200.0, centre=(0, 0, 0.6*mm)) + CuboidMagnet((1*mm, 1*mm, 1*mm), remanence=1.2,
                                                                                     centre=(0, 0, -1.5*mm))),
        HarmonicPotential(np.array([[1e-6, 2e-7, 0], [2e-7, 2e-6, 0], [0, 0, 3e-6]]), centre=(1e-7, 2e-7, 0)),
        UniformElectricField((10.0, -5.0, 2.0)),
    ], 2e-5, 1e-14),
}


@pytest.mark.filterwarnings('ignore:invalid value encountered in sqrt:RuntimeWarning')
@pytest.mark.parametrize('case', list(FIELD_CASES))
def test_compiled_element_forces_match_numpy(case):
    from levisim.simulation.backends.numba_elements import encode_elements, field_force
    fields, scale, tolerance = FIELD_CASES[case]
    trap = Trap(fields=fields, particle=SILICA)
    program = encode_elements([trap], [trap.fields_in_group('field')])
    out = np.empty(3)
    rng = np.random.default_rng(1)
    for _ in range(20):
        r = rng.normal(size=3) * scale
        t = rng.uniform(0, 1e-4)
        reference = np.array(trap.field_force(*r, t))
        field_force(t, r[0], r[1], r[2], 0, len(program.rows), *program.arrays, out)
        assert np.max(np.abs(out - reference)) <= tolerance * np.max(np.abs(reference))


def test_elliptic_integrals_match_scipy():
    from scipy.special import ellipe, ellipk

    from levisim.simulation.backends.numba_elements import _ellipke
    m = np.array([0.0, 1e-9, 0.1, 0.5, 0.9, 0.99, 0.999999])
    K, E = np.array([_ellipke(value) for value in m]).T
    np.testing.assert_allclose(K, ellipk(m), rtol=1e-15)
    np.testing.assert_allclose(E, ellipe(m), rtol=1e-15)


@pytest.mark.parametrize('order', ['first', 'self_consistent'])
@pytest.mark.parametrize('field', ['gaussian', 'plane_wave'])
@pytest.mark.parametrize('polarisation', [None, 'elliptical'])
def test_compiled_binding_forces_match_numpy(order, field, polarisation):
    from levisim.simulation.backends.numba_binding import binding_forces, encode_binding
    kwargs = {} if polarisation is None else {'polarisation': Tweezer.elliptical_polarisation(0.3, 0.5)}
    traps = [Trap(bead=Bead(r=90*nm), tweezer=Tweezer(φ=0.3*n, **kwargs), position=(0.4*n*μm, 2.1*n*μm, 0.1*n*μm))
             for n in range(3)]
    binding = OpticalBinding(traps=traps, order=order, tweezer_field=field)
    particles = [trap.bead for trap in traps]
    arrays = encode_binding(binding, particles)
    rng = np.random.default_rng(2)
    for _ in range(10):
        positions = np.array([trap.position for trap in traps]) + rng.normal(size=(3, 3)) * 30e-9
        reference = binding.forces(positions, particles)
        F = np.zeros((3, 3))
        binding_forces(positions, *arrays, F)
        assert max_relative_difference(reference, F) < 1e-12


# --------------------------------------------------------------------------------------------------
# FreeSpaceSimulation: trajectories
# --------------------------------------------------------------------------------------------------
def run_both(make_trap, num_samples, Δt, **kwargs):
    sims = {}
    for backend in ('numpy', 'numba'):
        sims[backend] = FreeSpaceSimulation(trap=make_trap(), num_samples=num_samples, Δt=Δt, seed=3, backend=backend,
                                            **kwargs)
        assert sims[backend].backend_used == backend
    return sims['numpy'], sims['numba']


FREE_SPACE_CASES = {
    # linear forces: no amplification of round-off
    'ring Paul trap': (lambda: Trap(fields=[ring(0.3, 10e3)], particle=CHARGED, gas=UHV), 6000, 1/(40*10e3),
                       {'X0': np.array([100*nm, 60*nm, 80*nm, 0, 0, 0])}),
    'Paul trap with electrode noise': (lambda: Trap(fields=[ring(0.3, 10e3, voltage_noise_psd=(40e-9)**2, noise_distances=1*mm)],
                                                    particle=CHARGED), 6000, 1/(40*10e3), {}),
    'diamagnetic quadrupole with gravity': (lambda: Trap(fields=[MagneticTrap(LocalFieldExpansion(gradient=1e5*np.diag([1.0, 0.5, -1.5]))),
                                                                 Gravity(direction=(0, 0, -1))],
                                                         particle=Sphere(r=1e-6, material=BISMUTH), gas=Gas(P=5*mbar), β=0.0),
                                            4000, 10e-6, {}),
    # Gaussian tweezer and magnet, in the harmonic regime
    'tweezer, current loop and cuboid': (lambda: Trap(fields=[
        MagneticTrap(CurrentLoop(1*mm, 200.0, centre=(0, 0, 0.6*mm)) + CuboidMagnet((1*mm, 1*mm, 1*mm), remanence=1.2,
                                                                                     centre=(0, 0, -1.5*mm))), Tweezer()],
        particle=Sphere(r=1e-6, material=FUSED_SILICA)), 4000, 0.1e-6, {}),
    'hybrid Paul-optical, small amplitude': (lambda: Trap(fields=[Tweezer(), ring(0.1, 300e3, axis='x')], particle=CHARGED,
                                                          gas=UHV), 3000, 1/(40*300e3), {}),
}


@pytest.mark.filterwarnings('ignore:.*Heun, second order.*:UserWarning')
@pytest.mark.parametrize('RK_method', ['vector_4', 'vector_2'])
@pytest.mark.parametrize('case', list(FREE_SPACE_CASES))
def test_free_space_kernel_matches_numpy(case, RK_method):
    """Same random numbers, compiled forces: ≲ 1e-12 relative (observed ≲ 1e-14)"""
    make_trap, num_samples, Δt, kwargs = FREE_SPACE_CASES[case]
    a, b = run_both(make_trap, num_samples, Δt, RK_method=RK_method, **kwargs)
    assert max_relative_difference(a.X[:, :3], b.X[:, :3]) < 1e-12
    assert max_relative_difference(a.X[:, 3:], b.X[:, 3:]) < 1e-12
    # the random stream (with the electrode-noise columns) is consumed identically
    assert a.rng.standard_normal() == b.rng.standard_normal()


def test_free_space_kernel_falls_back_for_uncompiled_sources():
    from levisim.simulation.backends import (
        AUTO_NUMBA_MIN_STEPS,
        UnsupportedBackendError,
    )

    def trap():
        callable_field = MagneticTrap(CallableField(lambda r: np.array([0.0, 0.0, 1.0]) + 1e3 * np.asarray(r)))
        return Trap(fields=[callable_field], particle=Sphere(r=1e-6, material=BISMUTH))
    # an explicit backend='numba' raises, naming the source; 'auto' falls back to NumPy
    with pytest.raises(UnsupportedBackendError, match='CallableField'):
        FreeSpaceSimulation(trap=trap(), num_samples=50, Δt=1e-6, seed=1, backend='numba')
    sim = FreeSpaceSimulation(trap=trap(), num_samples=AUTO_NUMBA_MIN_STEPS, Δt=1e-6, seed=1, backend='auto')
    assert sim.backend_used == 'numpy'
    assert 'CallableField' in sim._compiled_kernel_unsupported_fields()


# --------------------------------------------------------------------------------------------------
# MultiParticleFreeSpaceSimulation
# --------------------------------------------------------------------------------------------------
def binding_traps(polarisation=None):
    kwargs = {} if polarisation is None else {'polarisation': polarisation}
    return [Trap(bead=Bead(r=105*nm), tweezer=Tweezer(**kwargs)),
            Trap(bead=Bead(r=105*nm), tweezer=Tweezer(φ=0.4, **kwargs), position=(0.3*μm, 18.4*μm, 0.0))]


MULTI_CASES = {
    'binding first, Gaussian': (binding_traps, [OpticalBinding()]),
    'binding first, plane wave, circular': (lambda: binding_traps(Tweezer.elliptical_polarisation(np.pi/4)),
                                            [OpticalBinding(tweezer_field='plane_wave')]),
    'binding self-consistent': (binding_traps, [OpticalBinding(order='self_consistent')]),
    'Paul traps, Coulomb, gravity, Casimir': (
        lambda: [Trap(fields=[ring(0.3, 10e3)], particle=CHARGED, gas=UHV),
                 Trap(fields=[ring(0.3, 10e3)], particle=Sphere(r=100*nm, charge=400), gas=UHV, position=(0, 30*μm, 0))],
        [Coulomb(), NewtonianGravity(), Casimir()]),
}


@pytest.mark.filterwarnings('ignore:.*Heun, second order.*:UserWarning')
@pytest.mark.parametrize('RK_method', ['vector_4', 'vector_2'])
@pytest.mark.parametrize('case', list(MULTI_CASES))
def test_multi_particle_kernel_matches_numpy(case, RK_method):
    make_traps, interactions = MULTI_CASES[case]
    Δt = 1/(40*10e3) if 'Paul' in case else 0.5*μs
    runs = {backend: MultiParticleFreeSpaceSimulation(traps=make_traps(), interactions=interactions, num_samples=1500, Δt=Δt,
                                                      seed=5, RK_method=RK_method, backend=backend)
            for backend in ('numpy', 'numba')}
    assert runs['numba'].backend_used == 'numba'
    scale = np.abs(runs['numpy'].X).max(axis=0)
    assert np.all(np.abs(runs['numba'].X - runs['numpy'].X) <= 1e-11 * scale)


def test_multi_particle_kernel_takes_one_binding():
    from levisim.simulation.backends import UnsupportedBackendError
    with pytest.raises(UnsupportedBackendError, match='one OpticalBinding'):
        MultiParticleFreeSpaceSimulation(traps=binding_traps(), interactions=[OpticalBinding(), OpticalBinding()],
                                         num_samples=20, seed=1, backend='numba')
    sim = MultiParticleFreeSpaceSimulation(traps=binding_traps(), interactions=[OpticalBinding(), OpticalBinding()],
                                           num_samples=20, seed=1, backend='auto')
    assert sim.backend_used == 'numpy'


def test_nonlinear_cavity_kernel_raises_for_a_paul_trap_with_explicit_numba():
    from levisim.simulation.backends import UnsupportedBackendError
    mtrap = cavity_mtrap(trap_kwargs=[{'extra_fields': [PaulTrap.ring(1.0, 2*np.pi*200e3)]}, {}], paul_exact_frequencies=False)
    with pytest.raises(UnsupportedBackendError, match='PaulTrap'):
        CavitySimulation(mtrap, model='nonlinear', trap_positions=TRAP_POSITIONS, num_samples=50, Δt=0.1*μs, seed=3,
                         backend='numba', initial_state='equilibrium', detection=[Homodyne()])


# --------------------------------------------------------------------------------------------------
# the nonlinear CavitySimulation
# --------------------------------------------------------------------------------------------------
def cavity_mtrap(**kwargs):
    beads = [Bead(r=60.1*nm, Φ=np.pi/4, charge=10), Bead(r=60.2*nm, Φ=np.pi/4, charge=8)]
    return MultiTrap(num_traps=2, Cartesian_axes_per_trap=3, cavity_kwargs=shelf.DEFAULT_cavity_kwargs,
                     gas_kwargs=shelf.DEFAULT_gas_kwargs, beads=beads, **kwargs)


TRAP_POSITIONS = np.array([[0.0, 0.0, 0.0], [0.3e-6, 4.2e-6, 0.0]])
QUADRUPOLE = LocalFieldExpansion(B0=(0.0, 0.0, 0.5), gradient=50.0*np.diag([1.0, 0.5, -1.5]))


@pytest.mark.parametrize('case', ['power laws', 'binding', 'binding self-consistent', 'magnetic element'])
def test_nonlinear_cavity_kernel_matches_numpy(case):
    kwargs, mtrap_kwargs = {}, {}
    if case == 'power laws':
        kwargs['interactions'] = [Coulomb(), Casimir()]
    elif case == 'binding':
        kwargs['interactions'] = [OpticalBinding()]
    elif case == 'binding self-consistent':
        kwargs['interactions'] = [Coulomb(), OpticalBinding(order='self_consistent')]
    else:
        mtrap_kwargs['trap_kwargs'] = [{'extra_fields': [MagneticTrap(QUADRUPOLE)]}, {}]
    mtrap = cavity_mtrap(**mtrap_kwargs)
    runs = {}
    for backend in ('numpy', 'numba'):
        runs[backend] = CavitySimulation(
            mtrap, model='nonlinear', trap_positions=TRAP_POSITIONS, num_samples=1200, Δt=0.1*μs, seed=3, store_state=True,
            num_trajectories=2, substeps=2, backend=backend, initial_state='equilibrium',
            detection=[Homodyne(), Position('z1')], **kwargs)
    a, b = runs['numpy'], runs['numba']
    assert (a.backend_used, b.backend_used) == ('numpy', 'numba')
    np.testing.assert_allclose(b.states, a.states, rtol=1e-9, atol=1e-11 * np.abs(a.states).max())


def test_nonlinear_cavity_kernel_excludes_time_dependent_paul_traps():
    from levisim.simulation.nonlinear_cavity import NonlinearCavityModel
    assert not NonlinearCavityModel._compilable_elements([PaulTrap.ring(50.0, 2*np.pi*20e3)])
    assert NonlinearCavityModel._compilable_elements([MagneticTrap(QUADRUPOLE), UniformElectricField((1.0, 0, 0))])

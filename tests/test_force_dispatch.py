"""
The per-run force dispatch of the NumPy free-space loop (FreeSpaceSimulation.force_dispatch): resolved
once per run by prepare_numpy_loop, bit-identical to the per-call expressions it replaces, blind to
edits of trap.fields made during the run, and dropped afterwards
"""

import numpy as np
import pytest

from levisim.core.units import Hz, kHz, mbar, mm, nm, ns
from levisim.physics import Gas, Trap, Tweezer
from levisim.physics.fields import Feedback, ForceElement, HarmonicPotential, PaulTrap
from levisim.physics.kick import Kick
from levisim.physics.particle import Sphere
from levisim.simulation import FreeSpaceSimulation

CHARGED = Sphere(r=100*nm, charge=500)


def per_call_dynamics(sim, t, X):
    """The dynamics_vector of levisim before the per-run dispatch, from the Trap's own methods"""
    x, y, z, px, py, pz = X
    trap = sim.trap
    m = trap.bead.m
    plan = trap._element_plan
    if trap.override_Ω_dict:
        F = [-m * (trap.override_Ω_dict[axis])**2 * value for axis, value in zip('xyz', (x, y, z))]
    elif plan['single_tweezer']:
        F = trap.Ftw(x, y, z)
    else:
        F = trap.field_force(x, y, z, t)
    F_kick_x, F_kick_y = trap.kick_force(t) if plan['kicks'] else (0, 0)
    dynamics = np.array([px / m, py / m, pz / m, F[0] - trap.Γx*px + F_kick_x, F[1] - trap.Γy*py + F_kick_y,
                         F[2] - trap.Γz*pz])
    if plan['recoils']:
        dynamics[5] += trap.F_photon_recoil(x, y, z)
    if sim.gravity_field is not None or 'gravity' in plan['groups']:
        dynamics[3:] += sim.F_gravity
    return dynamics


def ring(q, f_rf):
    return PaulTrap.ring(q * CHARGED.m * (1*mm)**2 * (f_rf*Hz)**2 / (2 * CHARGED.q), f_rf*Hz, r0=1*mm, axis='x')


TRAPS = {
    'default': (lambda: Trap(), {}),
    'recoil, kicks, gravity': (lambda: Trap(photon_recoil=True, kicks=[Kick(start=0.0, duration=200*ns, K=20, θ=0.3)]),
                               {'gravity': True, 'gravity_direction': (0, 1, -2)}),
    'override_Ω_dict': (lambda: Trap(override_Ω_dict={'x': 2*np.pi*150e3, 'y': 2*np.pi*170e3, 'z': 2*np.pi*40e3}), {}),
    'tweezer, harmonic, feedback': (lambda: Trap(fields=[Tweezer(), HarmonicPotential(1e-7), Feedback(2*kHz, 2*kHz)]), {}),
    'hybrid Paul-optical': (lambda: Trap(fields=[Tweezer(), ring(0.1, 300e3)], particle=CHARGED), {'Δt': 1/(40*300e3)}),
}


@pytest.mark.parametrize('case', list(TRAPS))
def test_prepared_dispatch_is_bit_identical_to_the_per_call_expressions(case):
    make_trap, kwargs = TRAPS[case]
    sim = FreeSpaceSimulation(trap=make_trap(), num_samples=10, seed=1, backend='numpy', **kwargs)
    rng = np.random.default_rng(3)
    sim.prepare_numpy_loop()
    try:
        for _ in range(50):
            X = np.r_[rng.normal(size=3) * 50*nm, rng.normal(size=3) * 1e-20]
            t = rng.uniform(0, 1e-7)
            assert np.array_equal(sim.dynamics_vector(t, X), per_call_dynamics(sim, t, X))
    finally:
        sim.release_numpy_loop()


class AddsAnElementWhenCalled(ForceElement):
    """A 'field' element that, on its first evaluation, appends a stiff HarmonicPotential to its trap"""

    def __init__(self):
        self.trap = None

    def force(self, r, t, particle):
        if self.trap is not None and not any(isinstance(e, HarmonicPotential) for e in self.trap.fields):
            self.trap.fields.append(HarmonicPotential(1e-3))
        return np.zeros(3)


def test_edits_to_the_fields_during_a_run_are_ignored_until_the_next_run():
    def run(edit):
        element = AddsAnElementWhenCalled()
        trap = Trap(fields=[Tweezer(), element], gas=Gas(P=1*mbar))
        element.trap = trap if edit else None
        sim = FreeSpaceSimulation(trap=trap, num_samples=500, seed=2, backend='numpy')
        return sim, trap
    edited, trap = run(edit=True)
    plain, _ = run(edit=False)
    # the HarmonicPotential appended during the run did not act on it ...
    assert np.array_equal(edited.X, plain.X)
    assert any(isinstance(element, HarmonicPotential) for element in trap.fields)
    # ... but a direct call afterwards sees the trap as it is now
    X = np.array([10*nm, 0, 0, 0, 0, 0])
    assert edited._numpy_loop_dispatch is None
    assert edited.dynamics_vector(0.0, X)[3] == pytest.approx(per_call_dynamics(edited, 0.0, X)[3], rel=0, abs=0)
    assert edited.dynamics_vector(0.0, X)[3] < plain.dynamics_vector(0.0, X)[3] - 0.5 * 1e-3 * 10*nm


def test_the_plan_is_dropped_when_the_loop_raises():
    class Fails(ForceElement):
        def force(self, r, t, particle):
            raise RuntimeError('boom')
    sim = FreeSpaceSimulation(num_samples=10, seed=1, backend='numpy')
    sim.trap.fields.append(Fails())
    with pytest.raises(RuntimeError, match='boom'):
        sim.simulate_dynamics()
    assert sim._numpy_loop_dispatch is None

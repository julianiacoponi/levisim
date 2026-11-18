"""
Tests of the opt-in free-space physics: photon recoil (scattering force) and the shifted z
equilibrium, gravity, and the Malitson SiO2 permittivity. Every feature defaults to off.
"""
import numpy as np
import pytest

from levisim.core.constants import g, π
from levisim.core.units import kHz, nm, μm
from levisim.data.simulation import Simulation
from levisim.physics.bead import Bead, SiO2_refractive_index_Malitson
from levisim.physics.trap import Trap
from levisim.physics.tweezer import Tweezer


Δt = 1e-7


def test_features_default_off():
    assert Bead().εr == 1.98
    assert Bead().permittivity_model is None
    trap = Trap()
    assert trap.photon_recoil is False
    assert trap.F_photon_recoil(0, 0, 0) == 0
    a = Simulation(num_samples=500, Δt=Δt, seed=1)
    b = Simulation(num_samples=500, Δt=Δt, seed=1, gravity=False, trap=Trap(photon_recoil=False))
    assert np.array_equal(a.X, b.X)


def test_malitson_sellmeier():
    # fused silica: n_D(589.3 nm) = 1.4585 and n(1064 nm) = 1.4496 (refractiveindex.info, Malitson page)
    assert SiO2_refractive_index_Malitson(589.3*nm) == pytest.approx(1.4585, abs=1e-4)
    assert SiO2_refractive_index_Malitson(1064*nm) == pytest.approx(1.4496, abs=1e-4)
    bead = Bead(permittivity_model='Malitson', λ=1064*nm)
    assert bead.εr == pytest.approx(1.4496**2, abs=3e-4)
    with pytest.raises(ValueError):
        Bead(permittivity_model='Malitson', εr=2.0)
    with pytest.raises(ValueError):
        SiO2_refractive_index_Malitson(5*μm)


def test_gouy_phase():
    tweezer = Tweezer()
    assert tweezer.Gouy(tweezer.zR) == pytest.approx(π/4)


def test_scattering_force_and_equilibrium():
    trap = Trap()
    k, zR = trap.tweezer.k, trap.tweezer.zR
    # on axis at the focus: σ_scat I0 / c × (1 - 1/(k zR))
    from levisim.core.constants import c
    expected = trap.σ_scattering * trap.tweezer.I0 / c * (1 - 1/(k*zR))
    assert trap.F_scattering(0, 0, 0) == pytest.approx(expected, rel=1e-12)
    # σ_Rayleigh is the small-particle limit of σ_scattering
    assert trap.σ_Rayleigh == pytest.approx(trap.σ_scattering, rel=1e-3)
    # the exact root balances the forces, and is close to the linearised z_eq
    z0 = trap.z_eq_exact
    assert abs(trap.Ftwz(0, 0, z0) + trap.F_scattering(0, 0, z0)) < 1e-12 * trap.F_scattering(0, 0, 0)
    assert z0 == pytest.approx(trap.z_eq, rel=0.03)
    # Eq. (A4) of arXiv:2012.15822 is z_eq without the Gouy factor (1 - 1/(k zR)), to leading order in size
    b = trap.bead
    z0_A4 = (b.εr - 1)/(b.εr + 2) * 2 * k**4 * zR**2 / 3 * b.r**3
    assert trap.z_eq / (1 - 1/(k*zR)) == pytest.approx(z0_A4, rel=1e-3)
    # the beam is wider at z0, so the transverse frequencies drop
    assert trap.Ωj_at_z_eq['x'] < trap.Ωx


def test_simulated_mean_z_with_recoil():
    """With recoil the bead settles at z_eq_exact (strong z feedback: σz ≈ 2 nm ≪ z_eq ≈ 355 nm)"""
    trap = Trap(photon_recoil=True, Γ_fbz=50*kHz)
    sim = Simulation(trap=trap, num_samples=20_000, Δt=Δt, seed=2, zero_mean=False)
    z = sim.z.values[10_000:]
    assert np.mean(z) == pytest.approx(trap.z_eq_exact, rel=0.01)


def test_gravity_shifts_x_equilibrium():
    """Weight along -x shifts 〈x〉 by -g/Ωx² ≈ -10 pm; the same seed isolates the shift"""
    trap = Trap(Γ_fbx=20*kHz, Γ_fby=20*kHz, Γ_fbz=5*kHz)
    kwargs = dict(trap=trap, num_samples=100_000, Δt=Δt, seed=9, zero_mean=False)
    shift = np.mean(Simulation(gravity=True, **kwargs).X[20_000:, 0]) - np.mean(Simulation(**kwargs).X[20_000:, 0])
    assert shift == pytest.approx(-g / trap.Ωx**2, rel=0.02)
    # a custom direction is normalised
    sim = Simulation(trap=trap, num_samples=10, Δt=Δt, seed=9, gravity=True, gravity_direction=(0, 0, -2))
    assert np.allclose(sim.F_gravity, [0, 0, -trap.bead.weight])

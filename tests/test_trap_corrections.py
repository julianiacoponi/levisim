"""
Corrections of `Trap` and its use in `MultiTrap` (levisim 1.0)

- the tweezer potential `Trap.Utw` (and `Tweezer.potential`, `U_depth`) is the cycle-averaged dipole
  energy, whose curvature is the stiffness and whose gradient is the force (bit-identical to before)
- the off-diagonal stiffness of a trap's combined fields couples the axes of its particle in
  `MultiTrap`: the normal modes of a misaligned hybrid trap against the eigenvalues of its Hessian
- a hybrid Paul trap in `MultiTrap` has the exact (Floquet) secular frequencies, checked against an
  independent Hill-determinant solution of Mathieu's equation
- directed force noise β: √2 for the tweezer shorthand, 0 for traps built with fields=
- the finite-size torque is on by default for non-spherical particles, off for spheres
"""
import numpy as np
import pytest
from scipy.optimize import brentq

from levisim.core.constants import ħ
from levisim.core.units import Hz, mbar, mm, nm
from levisim.multiphysics.multitrap import MultiTrap
from levisim.physics import Spheroid
from levisim.physics.bead import Bead
from levisim.physics.fields import HarmonicPotential, PaulTrap
from levisim.physics.particle import Sphere
from levisim.physics.shapes import Dumbbell, Ellipsoid
from levisim.physics.trap import Trap
from levisim.physics.tweezer import Tweezer


def rotation_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


# --------------------------------------------------------------------------------------------------
# decision 33: the tweezer potential is the cycle-averaged one
# --------------------------------------------------------------------------------------------------
def test_potential_curvature_is_the_stiffness_and_its_gradient_the_force():
    trap = Trap()
    h = 1e-3 * trap.tweezer.w0
    for j, axis in enumerate('xyz'):
        e = np.eye(3)[j] * h
        curvature = (trap.Utw(*e) - 2 * trap.Utw(0, 0, 0) + trap.Utw(*(-e))) / h**2
        assert curvature == pytest.approx(trap.kj[axis], rel=1e-5)
    r = np.array([0.13, -0.21, 0.4]) * trap.tweezer.w0
    gradient = np.array([(trap.Utw(*(r + h*e)) - trap.Utw(*(r - h*e))) / (2*h) for e in np.eye(3)])
    assert -gradient == pytest.approx(np.array(trap.Ftw(*r)), rel=1e-6)
    assert trap.U_depth == pytest.approx(np.real(trap.α_eff) * trap.tweezer.E0**2 / 4, rel=1e-14)
    assert trap.v_escape == pytest.approx(np.sqrt(2 * trap.U_depth / trap.bead.m), rel=1e-14)
    assert trap.tweezer.potential(r, trap.bead) == trap.Utw(*r)


def test_tweezer_force_is_bit_identical_to_the_peak_energy_form():
    """The levisim <= 0.1 force 0.5·(4x/wx²)·U with the peak energy U = -(Re α/2)|E|²"""
    trap = Trap()
    tweezer, α = trap.tweezer, np.real(trap.α_eff)
    rng = np.random.default_rng(3)
    for r in rng.normal(scale=0.3 * tweezer.w0, size=(50, 3)):
        x, y, z = r
        U_peak = -(α / 2) * tweezer.E_field_amplitude(x, y, z)**2
        legacy = (
            0.5 * (4 * x / tweezer.wx(z)**2) * U_peak,
            0.5 * (4 * y / tweezer.wy(z)**2) * U_peak,
            0.5 * (4 * z / tweezer.wz(z)**2) * U_peak * (1 - 2 * tweezer.ellipse(x, y, z)),
        )
        assert trap.Ftw(x, y, z) == legacy
        assert (trap.Ftwx(x, y, z), trap.Ftwy(x, y, z), trap.Ftwz(x, y, z)) == legacy
        assert tuple(tweezer.force(r, 0.0, trap.bead)) == legacy


# --------------------------------------------------------------------------------------------------
# decision 24: intra-trap couplings in MultiTrap
# --------------------------------------------------------------------------------------------------
def _misaligned_multitrap(**kwargs):
    bead = Bead(r=80*nm)
    # θ = 0: no coherent scattering into the cavity, so the mechanics are on their own
    tweezer = Tweezer(θ=0.0)
    K_tw = Trap(tweezer=tweezer, bead=bead).stiffness_tensor
    # a harmonic field as stiff as the tweezer's x axis, its principal axes rotated by 30° about z
    # and with its own x-z tilt
    K_h = np.diag([0.6, 0.2, 0.5]) * K_tw[0, 0]
    R = rotation_z(np.radians(30)) @ np.array([[np.cos(0.3), 0, -np.sin(0.3)], [0, 1, 0], [np.sin(0.3), 0, np.cos(0.3)]])
    harmonic = HarmonicPotential(R @ K_h @ R.T)
    mt = MultiTrap(
        tweezers=[tweezer], beads=[bead], Cartesian_axes_per_trap=3, gas_kwargs={'P': 1e-9*mbar},
        trap_kwargs=[{'extra_fields': [harmonic], 'Γ_fbx': 0.0, 'Γ_fby': 0.0}], photon_recoil=False, **kwargs,
    )
    return mt, bead


def _mechanical_frequencies(mt):
    λ = mt.drift_eigenvalues
    # the cavity mode decays at κ/2, the mechanics at the (tiny) gas damping
    mechanical = λ[np.abs(λ.real) < 1e-3 * np.abs(λ.imag).max()]
    return np.sort(np.abs(mechanical.imag))[::2]


def test_misaligned_hybrid_trap_normal_modes_are_the_hessian_eigenvalues():
    mt, bead = _misaligned_multitrap()
    trap = mt.traps[0]
    assert all(dim.G == 0 for dim in mt.dimensions.values())
    # the full Hessian, independently from central differences of the summed static force
    h = 1e-4 * trap.tweezer.w0
    hessian = -np.array([(trap.static_force(h * e) - trap.static_force(-h * e)) / (2*h) for e in np.eye(3)]).T
    assert hessian == pytest.approx(trap.stiffness_tensor, rel=1e-6, abs=1e-6 * np.abs(hessian).max())
    Ω_modes = np.sqrt(np.sort(np.linalg.eigvalsh((hessian + hessian.T) / 2)) / bead.m)
    assert _mechanical_frequencies(mt) == pytest.approx(Ω_modes, rel=1e-6)
    # the couplings are the off-diagonal stiffness at the zero-point fluctuations
    labels = mt.labels
    zpf = {label: mt.dimensions[label].zpf for label in labels}
    for j, a in enumerate('xyz'):
        for k, b in enumerate('xyz'):
            if j != k:
                g = mt.mechanical_coupling_matrix[mt.label_to_index[f'{a}1'], mt.label_to_index[f'{b}1']]
                K_jk = trap.intra_trap_coupling_stiffness[j, k]
                assert g == pytest.approx(-K_jk * zpf[f'{a}1'] * zpf[f'{b}1'] / ħ, rel=1e-8)
    # opt-out: the axes stay uncoupled, at the diagonal frequencies
    off, _ = _misaligned_multitrap(intra_trap_couplings=False)
    assert not np.any(off.mechanical_coupling_matrix)
    assert _mechanical_frequencies(off) == pytest.approx(np.sort([trap.Ωx, trap.Ωy, trap.Ωz]), rel=1e-6)
    assert not np.allclose(_mechanical_frequencies(off), Ω_modes, rtol=1e-3)


def test_plain_tweezer_multitrap_is_unchanged_by_intra_trap_couplings():
    common = {'num_traps': 2, 'Cartesian_axes_per_trap': 3}
    on = MultiTrap(**common)
    off = MultiTrap(intra_trap_couplings=False, **common)
    assert np.array_equal(on.mechanical_coupling_matrix, off.mechanical_coupling_matrix)
    assert np.array_equal(on.drift_matrix, off.drift_matrix)


# --------------------------------------------------------------------------------------------------
# decision 34: exact (Floquet) frequencies of a hybrid Paul trap
# --------------------------------------------------------------------------------------------------
def hill_beta(a, q, N=40):
    """
    Characteristic exponent β of y'' + (a + 2q cos 2τ) y = 0 from Hill's method, independent of the
    monodromy: for a Floquet solution e^{iβτ} Σ c_n e^{2inτ}, a is the lowest eigenvalue of the
    tridiagonal matrix (β + 2n)² δ_nm - q (δ_n,m+1 + δ_n,m-1), which brentq inverts for β ∈ (0, 1)
    """
    n = np.arange(-N, N + 1)

    def lowest_a(β):
        M = np.diag((β + 2*n)**2) - q * (np.eye(2*N + 1, k=1) + np.eye(2*N + 1, k=-1))
        return np.linalg.eigvalsh(M)[0]
    return brentq(lambda β: lowest_a(β) - a, 1e-9, 1 - 1e-9, xtol=1e-14)


def _hybrid_paul_multitrap(q_axis, **kwargs):
    bead = Bead(charge=2000)
    tweezer = Tweezer(Power=4e-3)
    f_rf = 300e3
    # the ring trap about x has rf curvature -2 on x: q_x = -4 Q V/(m r0² Ω_rf²)
    V = q_axis * bead.m * (1*mm)**2 * (f_rf*Hz)**2 / (4 * bead.q)
    paul = PaulTrap.ring(V, f_rf*Hz, r0=1*mm, axis='x')
    mt = MultiTrap(
        tweezers=[tweezer], beads=[bead], trap_kwargs=[{'extra_fields': [paul], 'Γ_fbx': 0.0, 'Γ_fby': 0.0}],
        Cartesian_axes_per_trap=3, static_cavity_shifts=False, gas_kwargs={'P': 1e-9*mbar}, **kwargs,
    )
    return mt, paul, bead, tweezer


@pytest.mark.parametrize('q_axis', [0.2, 0.45])
def test_hybrid_paul_multitrap_frequencies_are_the_floquet_values(q_axis):
    mt, paul, bead, tweezer = _hybrid_paul_multitrap(q_axis)
    K_tw = np.diag(Trap(tweezer=tweezer, bead=bead).stiffness_tensor)
    a, q = paul.mathieu_parameters(bead, extra_stiffness=K_tw)
    assert abs(q[0]) == pytest.approx(q_axis, rel=1e-12)
    trap = mt.traps[0]
    assert mt.floquet_paul_traps == {1}
    for j, axis in enumerate('xyz'):
        exact = hill_beta(a[j], q[j]) * paul.Ω_rf / 2
        Ω = mt.dimensions[f'{axis}1'].Ω
        assert Ω == pytest.approx(exact, rel=1e-8)
        assert Ω == trap.paul_trap_report()['Ω_exact'][j]
        assert mt.dimensions[f'{axis}1'].static_cavity_shift == 0.0
    # the lowest-order stiffness is below the Floquet value on x, by 0.9% at |q| = 0.2 and 4.6% at
    # |q| = 0.45 in this hybrid trap
    lowest = mt.dimensions['x1'].Ω / trap.Ωx - 1
    assert lowest == pytest.approx(0.0088 if q_axis < 0.3 else 0.046, rel=0.05)
    old, *_ = _hybrid_paul_multitrap(q_axis, paul_exact_frequencies=False)
    assert old.floquet_paul_traps == set()
    assert [old.dimensions[f'{axis}1'].Ω for axis in 'xyz'] == [trap.Ωx, trap.Ωy, trap.Ωz]


def test_hybrid_paul_multitrap_beyond_the_stability_region_raises():
    with pytest.raises(ValueError, match='first Mathieu stability region'):
        _hybrid_paul_multitrap(1.2)


# --------------------------------------------------------------------------------------------------
# decision 36: directed force noise β
# --------------------------------------------------------------------------------------------------
def test_directed_noise_default_depends_on_how_the_trap_is_built():
    paul = PaulTrap.ring(10.0, 2e4 * Hz)
    particle = Sphere(r=100*nm, charge=500)
    assert Trap().β == np.sqrt(2)
    assert Trap(tweezer=Tweezer(), extra_fields=[paul], bead=Bead(charge=500)).β == np.sqrt(2)
    fields_trap = Trap(fields=[paul], particle=particle)
    assert fields_trap.β == 0.0 and fields_trap.Sdir == 0.0
    assert Trap(fields=[Tweezer()]).β == 0.0
    assert Trap(fields=[paul], particle=particle, β=np.sqrt(2)).β == np.sqrt(2)


# --------------------------------------------------------------------------------------------------
# decision 46: finite-size torque on for non-spherical particles
# --------------------------------------------------------------------------------------------------
def test_finite_size_torque_default_is_on_for_non_spherical_particles():
    assert not Trap().finite_size_torque
    assert not Trap(particle=Sphere(r=100*nm)).finite_size_torque
    assert not Trap(particle=Ellipsoid(a=60*nm, b=60*nm, c=60*nm)).finite_size_torque
    assert Trap(particle=Spheroid(a=70*nm, b=50*nm)).finite_size_torque
    assert Trap(particle=Dumbbell(r1=50*nm)).finite_size_torque
    assert not Trap(particle=Spheroid(a=70*nm, b=50*nm), finite_size_torque=False).finite_size_torque
    assert MultiTrap(num_traps=1, beads=[Spheroid(a=70*nm, b=50*nm)], Cartesian_axes_per_trap=3).traps[0].finite_size_torque

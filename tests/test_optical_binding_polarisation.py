"""
Optical binding with each tweezer's own polarisation (Jones vector): the default y-polarised tweezer
is unchanged bit for bit, a linear polarisation at an angle Θ equals the y-polarised calculation
in a geometry rotated by -Θ about the beam axis, circular light is covariant under rotations, and the
first-order force for elliptical light agrees with an independent evaluation from Jackson's dipole
field and finite differences.
"""

import numpy as np
import pytest

from levisim.core.units import nm, μm
from levisim.physics import Bead, OpticalBinding, Trap, Tweezer
from levisim.physics.interactions.optical_binding import dipole_field, tweezer_field

W0 = 0.75 * μm            # round tweezers, so that rotations about z are symmetries of the beams
R_BEAD = 80 * nm


def rotation_z(θ):
    c, s = np.cos(θ), np.sin(θ)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def make_traps(foci, polarisation=None, φ=(0.0, 0.0)):
    traps = []
    for n, focus in enumerate(foci):
        kwargs = {} if polarisation is None else {'polarisation': polarisation}
        traps.append(Trap(bead=Bead(r=R_BEAD), tweezer=Tweezer(wx0=W0, wy0=W0, φ=φ[n], **kwargs), position=focus))
    return traps


def binding_forces(traps, displacements, **kwargs):
    binding = OpticalBinding(traps=traps, **kwargs)
    positions = np.array([trap.position for trap in traps]) + displacements
    return binding, positions, binding.forces(positions, [trap.bead for trap in traps])


FOCI = np.array([[0.0, 0.0, 0.0], [1.9 * μm, 1.2 * μm, 0.0]])
DISPLACEMENTS = np.array([[12 * nm, -7 * nm, 30 * nm], [-5 * nm, 9 * nm, -20 * nm]])


@pytest.mark.parametrize('order', ['first', 'self_consistent'])
def test_default_polarisation_is_bit_identical(order):
    _, _, default = binding_forces(make_traps(FOCI), DISPLACEMENTS, order=order)
    _, _, explicit_y = binding_forces(make_traps(FOCI, polarisation=(0, 1)), DISPLACEMENTS, order=order)
    _, _, assumed_y = binding_forces(make_traps(FOCI, polarisation=(1, 0)), DISPLACEMENTS, order=order, polarisation='y')
    assert np.array_equal(default, explicit_y)
    assert np.array_equal(default, assumed_y)


@pytest.mark.parametrize('order', ['first', 'self_consistent'])
@pytest.mark.parametrize('field', ['gaussian', 'plane_wave'])
@pytest.mark.parametrize('Θ', [0.4, np.pi / 2, 2.2])
def test_linear_polarisation_equals_the_rotated_geometry(order, field, Θ):
    """F_Θ(r) = R(Θ)·F_y(R(-Θ)·r): a polarisation turned by Θ from y is the y calculation, rotated"""
    R = rotation_z(Θ)
    turned = make_traps(FOCI, polarisation=Tweezer.elliptical_polarisation(0.0, ζ=Θ))
    binding, _, F_turned = binding_forces(turned, DISPLACEMENTS, order=order, tweezer_field=field)
    reference = make_traps(FOCI @ R)                       # rows: R(-Θ)·r
    ref_binding, _, F_reference = binding_forces(reference, DISPLACEMENTS @ R, order=order, tweezer_field=field)
    scale = np.abs(F_reference).max()
    assert scale > 0
    np.testing.assert_allclose(F_turned, F_reference @ R.T, rtol=0, atol=1e-12 * scale)
    if order == 'first':
        positions = FOCI + DISPLACEMENTS
        K_turned = binding.stiffness_blocks(positions, [t.bead for t in turned])
        K_reference = ref_binding.stiffness_blocks(positions @ R, [t.bead for t in reference])
        K_rotated = np.einsum('ia,nmab,jb->nmij', R, K_reference, R)
        np.testing.assert_allclose(K_turned, K_rotated, rtol=0, atol=1e-11 * np.abs(K_reference).max())


@pytest.mark.parametrize('order', ['first', 'self_consistent'])
def test_circular_polarisation_is_covariant_under_rotations(order):
    """Turning the whole geometry about z turns the forces: the phase e^{iθ} of the rotated Jones
    vector is common to both tweezers and cancels"""
    circular = Tweezer.elliptical_polarisation(np.pi / 4)
    _, _, F = binding_forces(make_traps(FOCI, circular), DISPLACEMENTS, order=order)
    θ = 1.1
    R = rotation_z(θ)
    _, _, F_rotated = binding_forces(make_traps(FOCI @ R.T, circular), DISPLACEMENTS @ R.T, order=order)
    np.testing.assert_allclose(F_rotated, F @ R.T, rtol=0, atol=1e-12 * np.abs(F).max())
    # and circular light is not its real part (y-polarised light): the old ComplexWarning path
    _, _, F_y = binding_forces(make_traps(FOCI), DISPLACEMENTS, order=order)
    assert np.abs(F - F_y).max() > 0.1 * np.abs(F_y).max()


def test_first_order_elliptical_force_matches_jackson_dipole_field():
    """
    F_a = ½Re[p_a*·∂E_sc(r_a) + (∂u_a)*·α_a·E_sc(r_a)], E_sc the field of p_b = α_b·u_b from
    Jackson's Eq. (9.18) and ∂ by central differences: independent of the Green-tensor code
    """
    jones = Tweezer.elliptical_polarisation(0.3, ζ=0.5)
    traps = make_traps(FOCI, jones, φ=(0.0, 0.7))
    binding, positions, F = binding_forces(traps, DISPLACEMENTS, order='first')
    beads = [t.bead for t in traps]
    u = [tweezer_field(t.tweezer, positions[n] - FOCI[n], order=1) for n, t in enumerate(traps)]
    α = [binding.polarisability(n, beads[n]) for n in range(2)]
    k = traps[0].tweezer.k
    h = 1e-3 / k
    for a, b in ((0, 1), (1, 0)):
        p_a = α[a] @ u[a][0]
        p_b = α[b] @ u[b][0]
        E_sc = dipole_field(positions[a], positions[b], p_b, k)
        F_a = np.zeros(3)
        for c in range(3):
            step = np.zeros(3)
            step[c] = h
            dE = (8 * (dipole_field(positions[a] + step, positions[b], p_b, k) - dipole_field(positions[a] - step, positions[b], p_b, k))
                  - (dipole_field(positions[a] + 2*step, positions[b], p_b, k) - dipole_field(positions[a] - 2*step, positions[b], p_b, k))) / (12 * h)
            F_a[c] = 0.5 * np.real(np.conj(p_a) @ dE + np.conj(u[a][1][c]) @ (α[a] @ E_sc))
        np.testing.assert_allclose(F[a], F_a, rtol=0, atol=1e-8 * np.abs(F_a).max())


def test_tweezer_field_uses_the_jones_vector():
    circular = Tweezer(polarisation=Tweezer.elliptical_polarisation(np.pi / 4))
    E = tweezer_field(circular, np.zeros(3), order=0)[0]
    np.testing.assert_allclose(E / circular.E0, circular.polarisation, rtol=0, atol=1e-15)


def test_polarisation_keyword_is_validated():
    with pytest.raises(ValueError, match='polarisation'):
        OpticalBinding(polarisation='x')
    assert 'polarisation' in repr(OpticalBinding(polarisation='y'))

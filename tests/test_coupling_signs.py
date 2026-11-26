"""
Couplings of either sign: beads anywhere along the standing wave, blue detuning, toy couplings

Dimension used to assert gx, gy, gxy, gxz, gyz ≤ 0 and gz ≥ 0, which only holds for red detuning
and 0 ≤ Φ ≤ π/2. Moving every bead to π - Φ, -Φ or Φ + π only flips the signs of some couplings,
which is a relabelling of axes (z → -z, (x, y) → -(x, y), or all three), so the spectra must agree.
"""
import numpy as np
import pytest

from levisim.core.constants import π
from levisim.core.units import kHz, nm
from levisim.multiphysics.enums import QuantumLinearTheoryMethod
from levisim.multiphysics.multitrap import MultiTrap
from levisim.physics.bead import Bead

LAM = QuantumLinearTheoryMethod.LINEAR_AMPLIFIER_MODEL
ω_RANGE = np.arange(-400, 400, 2.0) * kHz


def build(Φs, axes=3, **kwargs):
    beads = [Bead(r=(60.1 + 0.1*n)*nm, Φ=Φ) for n, Φ in enumerate(Φs)]
    return MultiTrap(num_traps=len(Φs), Cartesian_axes_per_trap=axes, beads=beads, ω_range=ω_RANGE, **kwargs)


def spectra(mtrap):
    return (
        mtrap.homodyne_spectrum(LAM),
        mtrap.heterodyne_spectrum(LAM),
        mtrap.all_position_power_spectra(LAM)[0],
    )


def assert_same_spectra(a, b, rtol=1e-12):
    for x, y in zip(spectra(a), spectra(b)):
        np.testing.assert_allclose(x, y, rtol=rtol, atol=0)


def test_full_period_of_bead_positions_builds_and_couplings_follow_the_standing_wave():
    # without the static cavity shifts, which change each dimension's Ω (and so its zero-point
    # fluctuation) with Φ, and without the dispersive coupling of the cavity's own standing wave (∝ the
    # static field, tests/test_static_cavity_field.py); tests/test_static_cavity_shifts.py checks the
    # couplings with them
    at_node = build([π/2], static_cavity_shifts=False, cavity_dispersive_coupling=False).dimensions
    at_antinode = build([0.0], static_cavity_shifts=False, cavity_dispersive_coupling=False).dimensions
    for Φ in np.linspace(0, 2*π, 24, endpoint=False):
        dimensions = build([Φ], static_cavity_shifts=False, cavity_dispersive_coupling=False).dimensions
        assert dimensions['x1'].g == pytest.approx(at_node['x1'].g * np.sin(Φ), rel=1e-12, abs=1e-12 * abs(at_node['x1'].g))
        assert dimensions['y1'].g == pytest.approx(at_node['y1'].g * np.sin(Φ), rel=1e-12, abs=1e-12 * abs(at_node['y1'].g))
        assert dimensions['z1'].g == pytest.approx(at_antinode['z1'].g * np.cos(Φ), rel=1e-12, abs=1e-12 * abs(at_antinode['z1'].g))


@pytest.mark.parametrize('Φ', [0.2*π, 0.37*π])
@pytest.mark.parametrize('mirror', [lambda Φ: π - Φ, lambda Φ: -Φ, lambda Φ: Φ + π], ids=['π-Φ', '-Φ', 'Φ+π'])
def test_mirrored_bead_position_gives_the_same_spectra(Φ, mirror):
    assert_same_spectra(build([mirror(Φ)]), build([Φ]))


def test_two_beads_mirrored_together_give_the_same_spectra():
    cavity_kwargs = dict(Δ=-300*kHz)
    assert_same_spectra(
        build([0.8*π, 0.7*π], cavity_kwargs=cavity_kwargs),
        build([0.2*π, 0.3*π], cavity_kwargs=cavity_kwargs),
    )


def test_blue_detuning_off_the_node_builds():
    mtrap = build([0.3*π], cavity_kwargs=dict(Δ=+300*kHz))
    assert mtrap.dimensions['x1'].gxy > 0


def test_toy_couplings_of_either_sign_build():
    toy_params = dict(Ω_model='custom', toy_Ωjn_dict={'x1': 100*kHz, 'x2': 110*kHz}, gjn_dict={'x1': 10*kHz, 'x2': -10*kHz})
    mtrap = MultiTrap(num_traps=2, Cartesian_axes_per_trap=1, toy_params=toy_params, ω_range=ω_RANGE)
    assert mtrap.dimensions['x1'].g > 0 > mtrap.dimensions['x2'].g

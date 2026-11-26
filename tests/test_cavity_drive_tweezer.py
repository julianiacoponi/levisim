"""
A bead can scatter into the cavity the light of another tweezer than the one trapping it

Pontin et al., PRL 135, 073602 (2025): a bead held by a 976 nm tweezer, which the cavity does not
resonate with, scatters the out-of-focus field ζ·E0·exp(iφ) of the 1064 nm tweezer that traps the
other bead. `Trap(cavity_drive_tweezer=..., cavity_drive_field_ratio=ζ)` (through MultiTrap's
`trap_kwargs`) models it; the phase φ stays the trapping tweezer's (`φn_dict`).
"""
from copy import copy

import numpy as np
import pytest

from levisim.core.logger import logger
from levisim.core.units import degrees, kHz, mW, nm, μm, mbar
from levisim.multiphysics.multitrap import MultiTrap
from levisim.physics import coherent_scattering as cs
from levisim.physics.bead import Bead
from levisim.physics.trap import Trap
from levisim.physics.tweezer import Tweezer


@pytest.fixture
def warnings_logged():
    captured = []
    handler_id = logger.add(lambda message: captured.append(message.record['message']), level='WARNING', format='{message}')
    yield captured
    logger.remove(handler_id)


TWEEZER_1064 = Tweezer(Power=250*mW, λ=1064*nm, wx0=0.8*μm, wy0=0.8*μm, θ=45*degrees)
TWEEZER_976 = Tweezer(Power=300*mW, λ=976*nm, wx0=0.6*μm, wy0=0.6*μm, θ=135*degrees)
BEADS = [Bead(r=62.5*nm, ρ=2200.0, Φ=1.225), Bead(r=62.5*nm, ρ=2200.0, Φ=1.92)]
Ω1 = {'x': 124*kHz, 'y': 117*kHz, 'z': 19*kHz}
Ω2 = {'x': 120*kHz, 'y': 138*kHz, 'z': 26*kHz}


def pair(ζ=0.35, φ2=0.0, axes=2, drive=True, **kwargs):
    trap_kwargs = [{'override_Ω_dict': Ω1}, {'override_Ω_dict': Ω2}]
    if drive:
        trap_kwargs[1].update(cavity_drive_tweezer=TWEEZER_1064, cavity_drive_field_ratio=ζ)
    return MultiTrap(
        tweezers=[TWEEZER_1064, TWEEZER_976], beads=BEADS, trap_kwargs=trap_kwargs, φn_dict={2: φ2},
        Cartesian_axes_per_trap=axes, gas_kwargs={'P': 5e-6*mbar},
        cavity_kwargs={'L': 48.8e-3, 'w': 31*μm, 'κ': 57*kHz, 'Δ': -425*kHz},
        static_cavity_shifts=False,  # the Pontin et al. system as published, without static shifts
        **kwargs,
    )


def test_default_drive_is_the_trapping_tweezer_exactly():
    trap = Trap(tweezer=TWEEZER_976, bead=BEADS[0])
    assert trap.drive_tweezer is TWEEZER_976
    assert trap.cavity_drive_field_ratio == 1.0
    assert trap.cavity_drive_field == TWEEZER_976.E0


def test_explicit_own_tweezer_with_unit_ratio_is_bit_identical():
    default = MultiTrap(num_traps=2, Cartesian_axes_per_trap=3, beads=BEADS)
    explicit = MultiTrap(
        num_traps=2, Cartesian_axes_per_trap=3, beads=BEADS,
        trap_kwargs=[{'cavity_drive_tweezer': copy(tweezer), 'cavity_drive_field_ratio': 1} for tweezer in default.tweezers],
    )
    assert np.array_equal(default.drift_matrix, explicit.drift_matrix)
    assert default.cavity.αR == explicit.cavity.αR and default.cavity.αI == explicit.cavity.αI


def test_coupling_ratio_follows_ζ_position_and_zero_point_motion():
    # (the coherent-scattering couplings alone: without the dispersive coupling of the cavity's own
    # standing wave, which does not scale with ζ)
    mtrap = pair(ζ=0.35, φ2=0.0, cavity_dispersive_coupling=False)
    d = mtrap.dimensions
    for axis in 'xy':
        expected = 0.35 * np.sin(BEADS[1].Φ) / np.sin(BEADS[0].Φ) * np.sqrt(Ω1[axis] / Ω2[axis])
        assert d[f'{axis}2'].g / d[f'{axis}1'].g == pytest.approx(expected, rel=1e-12)
    # both beads couple with the 1064 nm wavenumber and the 1064 nm tweezer's θ = 45°
    assert d['x1'].g / d['y1'].g == pytest.approx(np.sqrt(Ω1['y'] / Ω1['x']), rel=1e-12)
    assert d['x2'].g / d['y2'].g == pytest.approx(np.sqrt(Ω2['y'] / Ω2['x']), rel=1e-12)
    # Re α_eff at the drive's (1064 nm) wavenumber for both beads
    α_drive = [cs.drive_polarisability(trap) for trap in mtrap.traps]
    assert mtrap.traps[1].ωd == pytest.approx(0.35 * mtrap.traps[0].ωd * α_drive[1] / α_drive[0], rel=1e-12)


def test_the_phase_is_the_trapping_tweezers_φ():
    mtrap = pair(ζ=0.35, φ2=172*degrees, cavity_dispersive_coupling=False)
    G_x1, G_x2 = mtrap.dimensions['x1'].G, mtrap.dimensions['x2'].G
    assert np.angle(G_x2 / G_x1) == pytest.approx(172*degrees, abs=1e-12)


def test_cavity_follows_the_drive_wavelength_without_a_warning(warnings_logged):
    mtrap = pair()
    assert mtrap.cavity.ωtw == TWEEZER_1064.ω
    assert not [message for message in warnings_logged if 'wavelength' in message]


def test_a_976_nm_tweezer_without_a_drive_tweezer_warns(warnings_logged):
    mtrap = pair(drive=False)
    assert mtrap.cavity.ωtw == TWEEZER_1064.ω
    assert any('different wavelengths' in message for message in warnings_logged)
    # and couples with the 976 nm wavenumber and its own (wrong, for this experiment) field
    assert abs(mtrap.dimensions['x2'].g) > abs(pair().dimensions['x2'].g)


def test_zero_ratio_decouples_the_bead():
    mtrap = pair(ζ=0.0, axes=3)
    for label in ('x2', 'y2', 'z2'):
        # no coherent scattering; the bead still feels the cavity's own standing wave (the dispersive
        # coupling, along the cavity axis), unless it is left out
        dim = mtrap.dimensions[label]
        assert dim.G - dim.dispersive_G == 0
        assert (dim.dispersive_G != 0) == (label != 'z2')
    off = pair(ζ=0.0, axes=3, cavity_dispersive_coupling=False)
    for label in ('x2', 'y2', 'z2'):
        assert off.dimensions[label].G == 0


@pytest.mark.parametrize('ratio', [-0.1, np.nan, np.inf])
def test_invalid_ratio_raises(ratio):
    with pytest.raises(ValueError, match='cavity_drive_field_ratio'):
        Trap(tweezer=TWEEZER_976, cavity_drive_tweezer=TWEEZER_1064, cavity_drive_field_ratio=ratio)


@pytest.mark.parametrize('ζ', [0.2, 0.35, 0.6])
def test_dark_mode_at_the_bare_crossing_for_any_ζ(ζ):
    """
    Two x modes at the same bare frequency, couplings g and ζ·g with a real ratio (φ = 0): the mode
    ∝ (ζ·g, -g) does not couple to the cavity and keeps only the gas damping (Pontin et al.,
    supplement Eq. (34)), whatever ζ; the bright mode is optically damped
    """
    Ω = {'x': 120*kHz, 'y': 100*kHz, 'z': 20*kHz}
    beads = [Bead(r=62.5*nm, ρ=2200.0, Φ=1.225), Bead(r=62.5*nm, ρ=2200.0, Φ=1.225)]
    mtrap = MultiTrap(
        tweezers=[TWEEZER_1064, TWEEZER_976], beads=beads, Cartesian_axes_per_trap=1,
        trap_kwargs=[{'override_Ω_dict': Ω}, {'override_Ω_dict': Ω, 'cavity_drive_tweezer': TWEEZER_1064, 'cavity_drive_field_ratio': ζ}],
        φn_dict={2: 0.0}, gas_kwargs={'P': 5e-6*mbar}, photon_recoil=False,
        cavity_kwargs={'L': 48.8e-3, 'w': 31*μm, 'κ': 57*kHz, 'Δ': -425*kHz},
        static_cavity_shifts=False,  # the Pontin et al. system as published, without static shifts
    )
    Γgas = mtrap.dimensions['x1'].Γ
    damping = np.sort(-2 * mtrap.drift_eigenvalues.real)
    # damping rates: dark pair (Γgas), bright pair (≫ Γgas), cavity pair (κ)
    assert damping[0] == pytest.approx(Γgas, rel=1e-3) and damping[1] == pytest.approx(Γgas, rel=1e-3)
    assert damping[2] > 100 * Γgas
    assert mtrap.is_stable

"""
A physical MultiTrap drives its cavity at the tweezers' frequency, not at Cavity's 1064 nm default
"""
import numpy as np
import pytest

from levisim.core.logger import logger
from levisim.core.units import degrees, kHz, nm
from levisim.multiphysics.multitrap import MultiTrap
from levisim.physics.cavity import Cavity
from levisim.physics.tweezer import Tweezer


@pytest.fixture
def warnings_logged():
    captured = []
    handler_id = logger.add(lambda message: captured.append(message.record['message']), level='WARNING', format='{message}')
    yield captured
    logger.remove(handler_id)


def test_default_physical_system_still_uses_1064_nm_exactly():
    mtrap = MultiTrap(num_traps=2, Cartesian_axes_per_trap=2)
    assert mtrap.cavity.ωtw == Tweezer().ω == Cavity().ωtw
    assert mtrap.cavity.E0 == Cavity().E0


def test_1550_nm_tweezers_set_the_cavity_frequency(warnings_logged):
    tweezers = [Tweezer(λ=1550*nm, θ=45*degrees), Tweezer(λ=1550*nm, θ=45*degrees, φ=30*degrees)]
    mtrap = MultiTrap(num_traps=2, Cartesian_axes_per_trap=2, tweezers=tweezers)
    assert mtrap.cavity.ωtw == tweezers[0].ω
    # the old behaviour, the cavity at 1064 nm, is still available explicitly (with a warning)
    at_1064 = MultiTrap(num_traps=2, Cartesian_axes_per_trap=2, tweezers=tweezers, cavity_kwargs={'ωtw': Tweezer().ω})
    assert at_1064.cavity.ωtw == Tweezer().ω
    assert any('differs from every tweezer frequency' in message for message in warnings_logged)
    # the cavity field per photon goes as √ω, and so do the couplings g ∝ E0; the cavity's own
    # frequency is ωtw - Δ, which differs from the tweezers' by Δ/ωtw ~ 1e-9 only
    ratio = np.sqrt(mtrap.cavity.ω / at_1064.cavity.ω)
    assert ratio == pytest.approx(np.sqrt(1064 / 1550), rel=1e-8)
    assert mtrap.cavity.E0 / at_1064.cavity.E0 == pytest.approx(ratio, rel=1e-12)
    for label in mtrap.labels:
        assert mtrap.dimensions[label].g / at_1064.dimensions[label].g == pytest.approx(ratio, rel=1e-12)


def test_explicit_matching_ωtw_gives_no_warning(warnings_logged):
    tweezers = [Tweezer(λ=1550*nm, θ=45*degrees)]
    # (one axis leaves out y, which couples to the light: dropped_axis_warning=False silences that warning)
    mtrap = MultiTrap(num_traps=1, Cartesian_axes_per_trap=1, tweezers=tweezers, cavity_kwargs={'ωtw': tweezers[0].ω},
                      dropped_axis_warning=False)
    assert mtrap.cavity.ωtw == tweezers[0].ω
    assert not warnings_logged


def test_mixed_tweezer_wavelengths_warn(warnings_logged):
    mtrap = MultiTrap(num_traps=2, Cartesian_axes_per_trap=1, tweezers=[Tweezer(), Tweezer(λ=1550*nm)])
    assert mtrap.cavity.ωtw == Tweezer().ω
    assert any('different wavelengths' in message for message in warnings_logged)


def test_toy_parameters_keep_the_cavity_kwargs():
    toy_params = dict(Ω_model='custom', toy_Ωjn_dict={'x1': 100*kHz}, gjn_dict={'x1': -10*kHz})
    mtrap = MultiTrap(num_traps=1, Cartesian_axes_per_trap=1, toy_params=toy_params)
    assert mtrap.cavity.ωtw == Cavity().ωtw

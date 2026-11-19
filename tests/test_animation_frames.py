"""
The animation helpers vary the parameter they are given: consecutive frames differ

update_parameter_in_multitrap_kwargs used to set toy_params['φ2'] for PHASE_OF_2ND_TWEEZER, which
MultiTrap ignores (it reads φn_dict), so every frame of a tweezer-phase animation showed the same
system; HOMODYNE_ANGLE was not handled at all, and the frame calculation called MultiTrap methods
that no longer exist (calculate_heterodyne_spectrum, ...).
"""
import copy

import numpy as np
import pytest

from levisim.core.units import degrees, kHz
from levisim.multiphysics.enums import QuantumLinearTheoryMethod, SpectrumType
from levisim.multiphysics.enums.parameter import Parameter, ParameterType
from levisim.multiphysics.multitrap import MultiTrap
from levisim.multiphysics.utils.animations import (
    calculate_spectra_for_all_frames,
    update_parameter_in_multitrap_kwargs,
)

ω_RANGE = np.arange(80, 160, 0.5) * kHz


def parameter(_type, units, axis_range):
    return Parameter(_type=_type, symbol='', units=units, unit_symbol='', unit_string='',
                     default_value=axis_range[0] * units, axis_range=np.asarray(axis_range, dtype=float))


PHASE_FRAMES = parameter(ParameterType.PHASE_OF_2ND_TWEEZER, degrees, [0, 30, 60])
COUPLING_Y = parameter(ParameterType.LIGHT_MECH_COUPLING, kHz, [-10, -20])


def toy_kwargs():
    toy_params = dict(Ω_model='custom', toy_Ωjn_dict={'x1': 110*kHz, 'x2': 120*kHz, 'x3': 130*kHz}, φ_odd_n=10*degrees)
    return dict(num_traps=3, Cartesian_axes_per_trap=1, toy_params=toy_params,
                cavity_kwargs=dict(Δ=-300*kHz, κ=200*kHz), gas_kwargs={}, ω_range=ω_RANGE)


def test_phase_of_2nd_tweezer_reaches_the_toy_multitrap():
    kwargs = update_parameter_in_multitrap_kwargs(toy_kwargs(), PHASE_FRAMES, 40*degrees)
    mtrap = MultiTrap(**kwargs)
    # trap 2 takes the new phase, the odd traps keep φ_odd_n
    assert mtrap.φn_dict == {1: 10*degrees, 2: 40*degrees, 3: 10*degrees}


def test_phase_of_2nd_tweezer_reaches_a_physical_multitrap():
    kwargs = dict(num_traps=2, Cartesian_axes_per_trap=1, cavity_kwargs={}, gas_kwargs={}, ω_range=ω_RANGE)
    kwargs = update_parameter_in_multitrap_kwargs(kwargs, PHASE_FRAMES, 40*degrees)
    assert MultiTrap(**kwargs).φn_dict[2] == 40*degrees


@pytest.mark.parametrize('spectrum_type', [SpectrumType.HETERODYNE, SpectrumType.POSITION_PSD])
def test_consecutive_tweezer_phase_frames_differ(spectrum_type):
    all_spectra, limits, info = calculate_spectra_for_all_frames(
        toy_kwargs(), spectrum_type, COUPLING_Y, PHASE_FRAMES,
        method=QuantumLinearTheoryMethod.LINEAR_AMPLIFIER_MODEL, PSD_label='x2',
    )
    assert len(all_spectra) == 3 and all_spectra[0].shape == (2, len(ω_RANGE))
    for before, after in zip(all_spectra, all_spectra[1:]):
        assert np.all(np.isfinite(after))
        assert np.max(np.abs(after / before - 1)) > 1e-3
    if spectrum_type == SpectrumType.POSITION_PSD:
        assert all(np.isfinite(info_frame['Phonons']).all() for info_frame in info)


def test_homodyne_angle_frames_differ():
    angle_frames = parameter(ParameterType.HOMODYNE_ANGLE, degrees, [0, 45])
    all_spectra, _, _ = calculate_spectra_for_all_frames(
        toy_kwargs(), SpectrumType.HOMODYNE, COUPLING_Y, angle_frames,
        method=QuantumLinearTheoryMethod.LINEAR_AMPLIFIER_MODEL,
    )
    assert np.max(np.abs(all_spectra[1] / all_spectra[0] - 1)) > 1e-3


def test_unknown_parameter_type_raises():
    unknown = copy.copy(PHASE_FRAMES)
    unknown.type = 'not a parameter'
    with pytest.raises(ValueError, match='does not know how to set'):
        update_parameter_in_multitrap_kwargs(toy_kwargs(), unknown, 1.0)

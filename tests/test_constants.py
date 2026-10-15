"""
Checks that core/constants.py re-exports scipy's constants under the physics symbols

This module is a deliberate re-export layer - every symbol here looks like an unused import
to a linter, but removing any of them breaks the whole package.
"""
import math

import scipy.constants

from levisim.core import constants


def test_scipy_constants_are_aliased_correctly():
    assert constants.π == scipy.constants.pi
    assert constants.ħ == scipy.constants.hbar
    assert constants.c == scipy.constants.speed_of_light
    assert constants.kB == scipy.constants.Boltzmann
    assert constants.e == scipy.constants.elementary_charge
    assert constants.u == scipy.constants.atomic_mass
    assert constants.ε0 == scipy.constants.epsilon_0
    assert constants.μ0 == scipy.constants.mu_0
    assert constants.g == scipy.constants.g
    assert constants.Avogadro == scipy.constants.Avogadro


def test_absolute_zero_and_lab_temperature():
    assert constants.absolute_zero == -scipy.constants.zero_Celsius
    # 21.5 degrees Celsius expressed in kelvin
    assert constants.T_lab == 21.5 + scipy.constants.zero_Celsius


def test_standing_wave_phase_positions():
    """Phases are given as Phi = k*x0, so a node sits a quarter wavelength from an antinode"""
    assert constants.antinode == 0
    assert constants.halfnode == math.pi / 4
    assert constants.node == math.pi / 2

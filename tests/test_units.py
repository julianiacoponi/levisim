"""
Unit-conversion sanity checks

`levisim.core.units` stores conversion factors as plain floats, and the angular-frequency
family deliberately folds in a factor of 2*pi - these tests pin that convention down.
"""
import math

import pytest

from levisim.core import units


def test_si_prefixes():
    assert units.metre == 1
    assert units.cm == 1e-2
    assert units.mm == 1e-3
    assert units.nm == 1e-9
    assert units.pm == 1e-12


def test_time_prefixes():
    assert units.second == 1
    assert units.ms == 1e-3
    assert units.ns == 1e-9


def test_megasamples_is_the_name_trace_info_relies_on():
    """data/trace.py divides by `megasamples`; a miscapitalised use site was a live NameError"""
    assert units.megasamples == 1e6


def test_angular_frequency_family_carries_the_factor_of_two_pi():
    """Hz here converts radians/second to hertz, so it is 2*pi rather than 1"""
    assert units.Hz == pytest.approx(2 * math.pi)
    assert units.kHz == pytest.approx(1e3 * 2 * math.pi)
    assert units.MHz == pytest.approx(1e6 * 2 * math.pi)
    assert units.GHz == pytest.approx(1e9 * 2 * math.pi)
    assert units.THz == pytest.approx(1e12 * 2 * math.pi)


def test_plain_hertz_family_does_not():
    """The lowercase-h family is a straight hertz conversion with no 2*pi"""
    assert units.hertz == 1
    assert units.kiloHz == 1e3
    assert units.megaHz == 1e6
    assert units.gigaHz == 1e9
    assert units.teraHz == 1e12


def test_mass_prefixes_are_relative_to_grams():
    assert units.kg == 1
    assert units.grams == 1e-3
    assert units.ng == pytest.approx(1e-12)
    assert units.fg == pytest.approx(1e-18)


def test_pressure_conversions():
    assert units.bar == 100_000
    assert units.mbar == pytest.approx(100)
    assert units.torr == pytest.approx(133.3224)


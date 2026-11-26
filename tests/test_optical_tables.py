"""
Full n, κ tables of the material library (owner decision 13): SILICON, DIAMOND and GOLD carry the
complete tables of their sources as a RefractiveIndexTable, interpolated linearly in n and κ
"""
import glob
import os

import numpy as np
import pytest

from levisim.core.units import nm, μm
from levisim.physics import Sphere
from levisim.physics.material import Material, RefractiveIndexTable, optical_data_path
from levisim.physics.materials import (
    DIAMOND,
    DIAMOND_NK,
    GOLD,
    GOLD_NK,
    SILICON,
    SILICON_NK,
)


def _linear(λ, λ1, v1, λ2, v2):
    return v1 + (λ - λ1) * (v2 - v1) / (λ2 - λ1)


def test_values_at_1064_and_1550_nm_are_those_of_the_former_two_point_library_bit_for_bit():
    # the expressions the library used before the full tables (same source rows, same arithmetic)
    n, κ = _linear(1.064, 1.06, 3.553, 1.07, 3.550), _linear(1.064, 1.06, 9.3631e-5, 1.07, 6.8118e-5)
    assert SILICON.εr(1064*nm) == (n + 1j*κ)**2
    assert SILICON.εr(1550*nm) == 3.4757**2
    assert DIAMOND.εr(1064*nm) == _linear(1.064, 1.033201562, 2.390522955, 1.239841874, 2.388124787)**2
    assert DIAMOND.εr(1550*nm) == _linear(1.55, 1.549802343, 2.387783491, 2.066403124, 2.38324149)**2
    n, κ = _linear(1.064, 1.06, 0.1705, 1.07, 0.1731), _linear(1.064, 1.06, 7.075, 1.07, 7.152)
    assert GOLD.εr(1064*nm) == (n + 1j*κ)**2
    assert GOLD.εr(1550*nm) == (0.3226 + 10.62j)**2


def test_library_tables_cover_their_sources_ranges():
    assert np.allclose(SILICON.wavelength_range, (0.25*μm, 14*μm), rtol=1e-12)
    assert np.allclose(DIAMOND.wavelength_range, (0.035424054*μm, 10*μm), rtol=1e-12)
    assert np.allclose(GOLD.wavelength_range, (0.3*μm, 24.93*μm), rtol=1e-12)
    for material in (SILICON, DIAMOND, GOLD):
        assert material.permittivity_kind == 'nk table' and material.is_dispersive
        assert 'doi:' in material.permittivity.source


def test_any_wavelength_inside_the_table_is_interpolated_and_rows_are_exact():
    # a tabulated row comes back exactly; the midpoint of two rows is their mean
    assert GOLD.n(1.06*μm) == 0.1705 + 7.075j
    mid = GOLD.n(1.065*μm)
    assert np.isclose(mid, 0.5 * ((0.1705 + 7.075j) + (0.1731 + 7.152j)), rtol=1e-14)
    # silicon at a common laser wavelength in the visible absorbs strongly; beyond 1.45 μm κ = 0
    assert SILICON.is_absorbing(633*nm) and not SILICON.is_absorbing(2*μm)
    # a sphere evaluates the table at its own λ
    s = Sphere(r=50*nm, material=DIAMOND, λ=780*nm)
    assert s.εr == DIAMOND.εr(780*nm)


def test_outside_the_table_raises_without_extrapolation():
    with pytest.raises(ValueError, match='outside the table'):
        GOLD.εr(200*nm)
    with pytest.raises(ValueError, match='outside the table'):
        SILICON.εr(20*μm)


def test_silicon_joins_green_and_li_at_1450_nm():
    λ = SILICON_NK._λ_μm
    assert np.all(np.diff(λ) > 0)
    assert SILICON_NK.nk(1.45*μm) == (3.485, 1.3846e-13)  # last row of Green (2008)
    assert SILICON.n(1.55*μm) == 3.4757                  # Li (1980), 293 K
    assert SILICON_NK.nk(1.5*μm)[1] == 0.0               # Li (1980) rows: κ = 0
    assert abs(SILICON.n(1.45*μm).real - 3.4845) < 1e-3   # Li's value at the join


def test_shipped_tables_have_a_cited_header_and_increasing_wavelengths():
    files = sorted(glob.glob(os.path.join(os.path.dirname(optical_data_path('x')), '*.csv')))
    assert len(files) >= 4
    for path in files:
        with open(path, encoding='utf-8') as file:
            header = [line for line in file if line.startswith('#')]
        text = ' '.join(header)
        assert 'doi:' in text and 'refractiveindex.info' in text and 'CC0' in text, path
        table = RefractiveIndexTable.from_csv(path)
        assert len(table) > 10 and np.all(np.diff(table.λ) > 0)
        assert np.all(table.κ_values >= 0)


def test_refractive_index_table_validation_and_user_tables():
    with pytest.raises(ValueError, match='strictly increasing'):
        RefractiveIndexTable([2e-6, 1e-6], [1.5, 1.5])
    with pytest.raises(ValueError, match='κ must be'):
        RefractiveIndexTable([1e-6, 2e-6], [1.5, 1.5], [0.0, -1.0])
    table = RefractiveIndexTable([1e-6, 2e-6], [1.5, 1.4], name='glass')
    material = Material('glass', 2500.0, permittivity=table)
    assert material.εr(1.5e-6) == pytest.approx(1.45**2, rel=1e-14)
    assert isinstance(material.εr(1.5e-6), float)
    assert material.tabulated_wavelengths == ()
    assert DIAMOND_NK.refractive_index(1.55*μm) == DIAMOND.n(1.55*μm)
    assert 'nk table' in GOLD.info_str and GOLD_NK.name in repr(GOLD_NK)

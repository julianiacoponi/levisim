"""
Cited magnetic and electrical data of the material library (owner decision 16): NdFeB recoil
permeability and grades, conductivity of bismuth and (anisotropic) pyrolytic graphite
"""
import numpy as np
import pytest

from levisim.core.constants import μ0
from levisim.physics import EddyCurrentDamping, Material, Sphere
from levisim.physics.fields.magnetic_sources import LocalFieldExpansion
from levisim.physics.materials import (
    BISMUTH,
    NDFEB,
    NDFEB_GRADES,
    NDFEB_RECOIL_PERMEABILITY,
    PYROLYTIC_GRAPHITE,
    ndfeb_grade,
)


def test_ndfeb_recoil_permeability_and_grades():
    assert NDFEB_RECOIL_PERMEABILITY == 1.05
    np.testing.assert_allclose(NDFEB.χ_m, np.diag([0.0, 0.0, 0.05]), rtol=1e-12, atol=0)
    assert np.isclose(μ0 * NDFEB.magnetisation[2], 1.23)          # unchanged Sagawa (1984) specimen
    assert NDFEB_GRADES['N35'] == 1.210 and NDFEB_GRADES['N42'] == 1.315 and NDFEB_GRADES['N52'] == 1.450
    assert list(NDFEB_GRADES.values()) == sorted(NDFEB_GRADES.values())
    n42 = ndfeb_grade('n42')
    assert np.isclose(μ0 * n42.magnetisation[2], 1.315) and n42.ρ == 7600.0
    assert n42.χ_m[2, 2] == pytest.approx(0.05) and ndfeb_grade('N42', recoil_susceptibility=False).χ_m[2, 2] == 0
    np.testing.assert_allclose(np.diag(n42.conductivity_tensor), [1/1.3e-6, 1/1.3e-6, 1/1.5e-6])
    assert n42.is_conductor and n42.is_electrically_anisotropic and 'arnoldmagnetics' in n42.references['magnetisation']
    with pytest.raises(KeyError, match='Unknown NdFeB grade'):
        ndfeb_grade('N99')


def test_bismuth_and_graphite_conductivities():
    assert BISMUTH.conductivity == 8.7e5 and BISMUTH.is_conductor and not BISMUTH.is_electrically_anisotropic
    assert 'doi:10.1038/s42005-025-02010-7' in BISMUTH.references['conductivity']
    σ = PYROLYTIC_GRAPHITE.conductivity_tensor
    np.testing.assert_array_equal(σ, np.diag([2e5, 2e5, 200.0]))     # in plane (x, y), c axis (z)
    assert PYROLYTIC_GRAPHITE.is_conductor and PYROLYTIC_GRAPHITE.is_electrically_anisotropic
    assert 'arXiv:2208.10393' in PYROLYTIC_GRAPHITE.references['conductivity']


def test_conductivity_tensor_validation():
    with pytest.raises(ValueError, match='symmetric and positive'):
        Material('x', 1000.0, conductivity=np.diag([1.0, -1.0, 1.0]))
    with pytest.raises(ValueError, match='scalar or a 3×3'):
        Material('x', 1000.0, conductivity=[1.0, 2.0])
    assert Material('x', 1000.0, conductivity=3.0).conductivity_tensor[1, 1] == 3.0
    assert not Material('x', 1000.0).is_conductor


def _gradient_field():
    G = np.array([[1.0, 0.2, 0.0], [0.2, -0.4, 0.1], [0.0, 0.1, -0.6]])    # symmetric, traceless [T/m]
    return LocalFieldExpansion(B0=(0.0, 0.0, 0.5), gradient=G)


def test_eddy_damping_uses_the_library_bismuth_conductivity():
    source = _gradient_field()
    sphere = Sphere(r=10e-6, material=BISMUTH)
    J = source.gradB(np.zeros(3))
    Γ = EddyCurrentDamping(source).damping_tensor(sphere)
    np.testing.assert_allclose(Γ, 8.7e5 * sphere.r**2 * (J.T @ J) / (10 * BISMUTH.ρ), rtol=1e-12)


def test_eddy_damping_refuses_an_anisotropic_conductivity_unless_given_one():
    source = _gradient_field()
    sphere = Sphere(r=10e-6, material=PYROLYTIC_GRAPHITE)
    with pytest.raises(ValueError, match='anisotropic conductivity'):
        EddyCurrentDamping(source).damping_tensor(sphere)
    Γ_high = EddyCurrentDamping(source, conductivity=2e5).damping_rates(sphere)
    Γ_low = EddyCurrentDamping(source, conductivity=200.0).damping_rates(sphere)
    assert np.all(Γ_high / Γ_low == pytest.approx(1000.0))

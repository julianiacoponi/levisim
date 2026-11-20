"""
The optional Mie backend (levisim[mie]) against miepython's own efficiencies and the dipole model

Skipped when miepython is not installed.
"""
import numpy as np
import pytest

miepython = pytest.importorskip('miepython')

from levisim.core.constants import π
from levisim.core.units import nm, μm
from levisim.physics import materials as M
from levisim.physics.bead import Bead
from levisim.physics.mie import (
    compare_with_dipole,
    dipole_cross_sections,
    mie_available,
    mie_coefficients,
    mie_cross_sections,
    mie_dipole_polarisability,
)
from levisim.physics.particle import Sphere

λ = 1064*nm


def test_backend_available():
    assert mie_available()


@pytest.mark.parametrize('r', [20*nm, 60.1*nm, 300*nm, 1*μm])
@pytest.mark.parametrize('material', [M.LEVISIM_DEFAULT_SILICA, M.SILICON, M.GOLD])
def test_cross_sections_match_miepython_efficiencies(r, material):
    εr = material.εr(λ)
    σ = mie_cross_sections(r, εr, λ)
    n = np.sqrt(complex(εr))
    qext, qsca, _, _ = miepython.efficiencies_mx(np.conj(n), 2*π*r/λ)
    assert σ['σ_ext'] == pytest.approx(qext * π * r**2, rel=1e-12)
    assert σ['σ_sca'] == pytest.approx(qsca * π * r**2, rel=1e-12)


def test_small_sphere_limit_is_the_dipole_with_radiation_reaction():
    s = Sphere(r=5*nm)
    α_eff = s.polarisability_tensor(radiation_reaction=True)[0, 0]
    α_mie = mie_dipole_polarisability(s.r, s.εr, λ)
    assert α_mie.real == pytest.approx(α_eff.real, rel=1e-5)
    assert α_mie.imag == pytest.approx(α_eff.imag, rel=1e-4)


def test_default_bead_agrees_with_the_dipole_model():
    """
    k r ≈ 0.355 for the default bead. The O(x²) correction to a₁ is (3x²/5)(εr - 2)/(εr + 2), which
    almost vanishes for εr = 1.98, so the dipole model is good to ~1e-3, well inside x² ≈ 0.13
    """
    c = compare_with_dipole(Bead())
    assert c.x == pytest.approx(0.355, abs=1e-3)
    assert abs(c.polarisability_ratio - 1) < 3e-3
    assert abs(c.scattering_ratio - 1) < 5e-3
    assert c.multipole_fraction < 1e-3
    # lossless: extinction is scattering in both models
    assert c.mie['σ_abs'] == pytest.approx(0, abs=1e-12 * c.mie['σ_ext'])
    assert c.dipole['σ_abs'] == pytest.approx(0, abs=1e-12 * c.dipole['σ_ext'])


@pytest.mark.parametrize('r, minimum', [(300*nm, 0.05), (1*μm, 0.5)])
def test_dipole_model_fails_for_large_spheres(r, minimum):
    c = compare_with_dipole(Sphere(r=r))
    assert abs(c.polarisability_ratio - 1) > minimum
    assert c.multipole_fraction > 0.3


def test_absorbing_sphere_conventions():
    s = Sphere(r=40*nm, material=M.GOLD)
    a, b = mie_coefficients(s.r, s.εr, λ)
    assert a[0].real > 0 and b[0].real > 0
    α_mie = mie_dipole_polarisability(s.r, s.εr, λ)
    assert α_mie.imag > 0
    σ = dipole_cross_sections(α_mie, λ)
    assert σ['σ_abs'] > 0


def test_anisotropic_material_is_refused():
    with pytest.raises(ValueError, match='isotropic'):
        mie_cross_sections(100*nm, M.VATERITE.εr(λ), λ)

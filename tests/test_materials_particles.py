"""
Material, the material library, Particle and Sphere, and Bead as a Sphere

`Bead` became a `Sphere` made of a `Material` without any number changing; the bit-identity of
every Bead, Trap, Simulation and MultiTrap output against main was checked when the refactor was
made (see the pull request), and `test_bead_numbers_are_pinned` keeps a few of them pinned here.
"""
import re

import numpy as np
import pytest

from levisim.core.constants import e, kB, μ0, π
from levisim.core.units import nm, mbar
from levisim.physics import Bead, Material, Particle, Sphere
from levisim.physics.gas import Gas
from levisim.physics.material import SiO2_permittivity_Malitson, SiO2_refractive_index_Malitson
from levisim.physics.materials import (
    BISMUTH,
    DIAMOND,
    FUSED_SILICA,
    GOLD,
    LEVISIM_DEFAULT_SILICA,
    MATERIALS,
    NDFEB,
    PYROLYTIC_GRAPHITE,
    SILICON,
    VATERITE,
    get_material,
)
from levisim.physics.trap import Trap
from levisim.physics.tweezer import Tweezer


####################################################################################################
# Material
# --------------------------------------------------------------------------------------------------
def test_constant_real_permittivity():
    material = Material('glass', 2000.0, permittivity=2.25)
    assert material.εr() == 2.25 and type(material.εr()) is float
    assert material.εr(1550*nm) == 2.25
    assert material.n() == 1.5
    assert not material.is_absorbing() and not material.is_anisotropic() and not material.is_dispersive
    np.testing.assert_array_equal(material.εr_tensor(), 2.25 * np.eye(3))


def test_complex_permittivity_is_absorbing_with_positive_imaginary_index():
    material = Material('lossy', 1000.0, permittivity=(2 + 0.1j)**2)
    assert material.is_absorbing()
    n = material.n()
    assert np.isclose(n, 2 + 0.1j)
    assert n.real > 0 and n.imag > 0


def test_tensor_permittivity_is_anisotropic_and_gives_principal_indices():
    ε = np.diag([1.5**2, 1.5**2, 1.6**2])
    material = Material('uniaxial', 2500.0, permittivity=ε)
    assert material.permittivity_kind == 'tensor'
    assert material.is_anisotropic()
    np.testing.assert_allclose(material.n(), [1.5, 1.5, 1.6])
    # a rotated (non-diagonal) tensor has the same principal indices
    c, s = np.cos(0.3), np.sin(0.3)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    rotated = Material('rotated', 2500.0, permittivity=R @ ε @ R.T)
    np.testing.assert_allclose(np.sort(rotated.n()), [1.5, 1.5, 1.6])
    # an isotropic tensor is not anisotropic
    assert not Material('iso', 1.0, permittivity=2 * np.eye(3)).is_anisotropic()


def test_named_and_callable_dispersion_models():
    malitson = Material('fs', 2201.0, permittivity='malitson')
    assert malitson.is_dispersive
    for λ in (532*nm, 1064*nm, 1550*nm):
        assert malitson.εr(λ) == SiO2_refractive_index_Malitson(λ)**2 == SiO2_permittivity_Malitson(λ)
    with pytest.raises(ValueError, match='pass the vacuum wavelength'):
        malitson.εr()
    with pytest.raises(ValueError, match='0.21-3.71'):
        malitson.εr(5e-6)

    drude = Material('drude', 1.0, permittivity=lambda λ: 1 - (λ / (150*nm))**2 + 0.5j)
    assert drude.εr(150*nm) == 0.5j
    assert drude.is_absorbing(150*nm)
    with pytest.raises(ValueError, match='Unknown permittivity model'):
        Material('x', 1.0, permittivity='Sellmeier-unknown')


def test_tabulated_permittivity_is_exact_at_its_wavelengths_only():
    material = Material('table', 1.0, permittivity={1064*nm: 2.0, 1550*nm: 1.9 + 0.01j})
    assert material.permittivity_kind == 'tabulated'
    assert material.εr(1064*nm) == 2.0
    assert material.εr(1550*nm) == 1.9 + 0.01j
    assert material.tabulated_wavelengths == (1064*nm, 1550*nm)
    with pytest.raises(ValueError, match='tabulated permittivity only at 1064.0 nm, 1550.0 nm'):
        material.εr(980*nm)


def test_material_without_optical_data_raises_on_permittivity():
    material = Material('magnet', 7000.0)
    assert not material.has_permittivity
    with pytest.raises(ValueError, match='no permittivity'):
        material.εr(1064*nm)


@pytest.mark.parametrize('kwargs, message', [
    (dict(ρ=-1.0), 'positive, finite density'),
    (dict(ρ=1.0, permittivity=np.ones((2, 2))), 'scalar or a 3×3'),
    (dict(ρ=1.0, χ_m=np.ones(2)), 'χ_m must be'),
    (dict(ρ=1.0, magnetisation=np.ones(2)), 'magnetisation must be'),
])
def test_material_validates_its_inputs(kwargs, message):
    with pytest.raises(ValueError, match=message):
        Material('bad', **kwargs)


def test_material_is_immutable_and_replace_derives_a_copy():
    with pytest.raises(AttributeError, match='immutable'):
        FUSED_SILICA.ρ = 1000.0
    porous = FUSED_SILICA.replace(name='porous silica', ρ=1850.0)
    assert porous.ρ == 1850.0 and FUSED_SILICA.ρ == 2201.0
    assert porous.εr(1064*nm) == FUSED_SILICA.εr(1064*nm)
    with pytest.raises(TypeError, match='unknown arguments'):
        FUSED_SILICA.replace(density=1.0)


def test_magnetic_flags():
    assert not LEVISIM_DEFAULT_SILICA.is_magnetic and not LEVISIM_DEFAULT_SILICA.is_diamagnetic
    assert FUSED_SILICA.is_diamagnetic and not FUSED_SILICA.is_magnetically_anisotropic
    assert BISMUTH.is_magnetic and BISMUTH.is_diamagnetic and not BISMUTH.is_magnetically_anisotropic
    assert PYROLYTIC_GRAPHITE.is_diamagnetic and PYROLYTIC_GRAPHITE.is_magnetically_anisotropic
    assert NDFEB.is_magnetic and not NDFEB.is_diamagnetic and NDFEB.is_conductor
    np.testing.assert_array_equal(BISMUTH.χ_m_tensor, -170e-6 * np.eye(3))


####################################################################################################
# Library
# --------------------------------------------------------------------------------------------------
SOURCE = re.compile(r'doi:10\.\d{4,9}/\S+|https?://\S+|NIST Standard Reference Database')


@pytest.mark.parametrize('name', sorted(MATERIALS))
def test_every_library_number_has_a_source(name):
    material = MATERIALS[name]
    assert 'ρ' in material.references
    sourced = set(material.references)
    if material.has_permittivity:
        assert any(key.startswith('εr') for key in sourced), name
    if np.any(material.χ_m_tensor != 0):
        assert 'χ_m' in sourced
    if material.magnetisation is not None:
        assert 'magnetisation' in sourced
    if material.conductivity is not None:
        assert 'conductivity' in sourced
    for quantity, source in material.references.items():
        if name == 'levisim_default_silica' and quantity == 'εr':
            assert 'not recorded' in source  # stated honestly, see the open question
            continue
        assert SOURCE.search(source), (name, quantity, source)


def test_library_values():
    assert LEVISIM_DEFAULT_SILICA.ρ == 1850.0 and LEVISIM_DEFAULT_SILICA.εr() == 1.98
    assert 'not mutually consistent' in LEVISIM_DEFAULT_SILICA.notes

    assert FUSED_SILICA.ρ == 2201.0
    assert np.isclose(FUSED_SILICA.n(1064*nm), 1.44963, atol=5e-6)

    assert SILICON.ρ == 2330.0
    assert np.isclose(SILICON.n(1550*nm), 3.4757)
    assert np.isclose(SILICON.n(1064*nm), 3.5518 + 8.34258e-5j)
    assert SILICON.is_absorbing(1064*nm) and not SILICON.is_absorbing(1550*nm)

    assert DIAMOND.ρ == 3515.0
    assert np.isclose(DIAMOND.n(1064*nm), 2.39017, atol=1e-5)
    assert np.isclose(DIAMOND.n(1550*nm), 2.38778, atol=1e-5)

    assert VATERITE.ρ == 2540.0 and VATERITE.is_anisotropic()
    np.testing.assert_allclose(VATERITE.n(), [1.55, 1.55, 1.65])

    assert GOLD.ρ == 19320.0
    assert np.isclose(GOLD.n(1550*nm), 0.3226 + 10.62j)
    assert np.isclose(GOLD.n(1064*nm), 0.17154 + 7.1058j)
    assert GOLD.εr(1064*nm).real < 0 and GOLD.is_absorbing(1550*nm)

    np.testing.assert_array_equal(np.diag(PYROLYTIC_GRAPHITE.χ_m), [-85e-6, -85e-6, -450e-6])
    assert PYROLYTIC_GRAPHITE.ρ == 2200.0
    assert BISMUTH.χ_m == -170e-6 and BISMUTH.ρ == 9747.0

    assert NDFEB.ρ == 7400.0
    assert np.isclose(μ0 * np.linalg.norm(NDFEB.magnetisation), 1.23)
    assert np.isclose(NDFEB.conductivity, 1 / 1.4e-6)


def test_get_material_by_name():
    assert get_material('Fused silica') is FUSED_SILICA
    assert get_material('pyrolytic-graphite') is PYROLYTIC_GRAPHITE
    with pytest.raises(KeyError, match='Unknown material'):
        get_material('unobtainium')


####################################################################################################
# Particle and Sphere
# --------------------------------------------------------------------------------------------------
def test_particle_base_needs_a_shape():
    particle = Particle(FUSED_SILICA)
    assert particle.charge == 0 and particle.q == 0
    with pytest.raises(NotImplementedError):
        particle.V
    with pytest.raises(NotImplementedError):
        particle.translational_damping_tensor(Gas())
    with pytest.raises(NotImplementedError):
        particle.rotational_damping_tensor(Gas())
    with pytest.raises(TypeError, match='must be a Material'):
        Particle('silica')
    with pytest.raises(ValueError, match='unit quaternion'):
        Particle(FUSED_SILICA, orientation=(1, 1, 0, 0))


def test_sphere_geometry_mass_inertia_and_charge():
    sphere = Sphere(r=100*nm, material=DIAMOND, charge=-3)
    assert sphere.V == (4/3)*π * (100*nm)**3
    assert sphere.A == 4*π * (100*nm)**2
    assert sphere.m == 3515.0 * sphere.V
    np.testing.assert_array_equal(sphere.inertia_tensor, (2/5) * sphere.m * sphere.r**2 * np.eye(3))
    assert sphere.I == (2/5) * sphere.m * sphere.r**2
    assert sphere.charge == -3 and sphere.q == -3 * e
    assert Sphere().charge == 0
    np.testing.assert_array_equal(sphere.orientation, [1, 0, 0, 0])
    with pytest.raises(ValueError, match='positive, finite radius'):
        Sphere(r=0)


def test_sphere_polarisability_of_an_anisotropic_material_is_diagonal_clausius_mossotti():
    sphere = Sphere(r=1e-6, material=VATERITE)
    α = sphere.α
    assert α.shape == (3, 3)
    for i, n in enumerate([1.55, 1.55, 1.65]):
        εr = n**2
        assert np.isclose(α[i, i], 3 * sphere.V * 8.8541878128e-12 * (εr - 1) / (εr + 2), rtol=1e-9)
    assert np.allclose(α - np.diag(np.diag(α)), 0)


def test_sphere_polarisability_at_another_wavelength():
    sphere = Sphere(material=FUSED_SILICA, λ=1064*nm)
    at_1550 = sphere.static_polarisability_tensor(1550*nm)[0, 0]
    expected = Sphere(material=FUSED_SILICA, λ=1550*nm).α
    assert at_1550 == expected
    assert sphere.static_polarisability_tensor()[0, 0] == sphere.α


def test_radiation_reaction_polarisability_matches_trap_alpha_eff():
    for bead in (Bead(), Bead(r=75*nm, permittivity_model='Malitson')):
        trap = Trap(bead=bead, tweezer=Tweezer(λ=bead.λ))
        α_eff = bead.polarisability_tensor(radiation_reaction=True)
        np.testing.assert_allclose(α_eff, trap.α_eff * np.eye(3), rtol=1e-12, atol=0)
        assert np.imag(α_eff[0, 0]) > 0
    np.testing.assert_array_equal(Bead().polarisability_tensor(), Bead().α * np.eye(3))


def test_sphere_translational_damping_equals_trap_gamma_th_exactly():
    gas = Gas(P=1e-2*mbar)
    for bead in (Bead(), Bead(r=75*nm, ρ=2200.0)):
        trap = Trap(bead=bead, gas=gas)
        np.testing.assert_array_equal(bead.translational_damping_tensor(gas), trap.Γth * np.eye(3))


def test_sphere_rotational_damping_against_martinetz_table_one():
    gas = Gas(P=1e-2*mbar, T=300.0)
    sphere = Sphere(r=100*nm, material=FUSED_SILICA)
    Γ_rot = sphere.rotational_damping_tensor(gas)[0, 0]
    # Martinetz et al. 2018, Table I, written directly in terms of the gas number density
    n_g = gas.P / (kB * gas.T)
    expected = 10 * n_g * sphere.r**2 * np.sqrt(2*π * gas.m * kB * gas.T) / (3 * sphere.m)
    assert np.isclose(Γ_rot, expected, rtol=1e-12)
    Γ_cm = sphere.translational_damping_tensor(gas)[0, 0]
    assert np.isclose(Γ_rot / Γ_cm, (5/4) / (1 + π/8), rtol=1e-12)
    assert sphere.rotational_damping_tensor(gas, accommodation=0)[0, 0] == 0


####################################################################################################
# Bead as a Sphere
# --------------------------------------------------------------------------------------------------
BEAD_ATTRIBUTES = ['r', 'ρ', 'εr', 'λ', 'Φ', 'n', 'V', 'A', 'm', 'weight', 'α', 'charge']


def test_bead_is_a_sphere_made_of_the_named_default_silica():
    bead = Bead()
    assert isinstance(bead, Sphere) and isinstance(bead, Particle)
    assert bead.material is LEVISIM_DEFAULT_SILICA
    assert Bead(ρ=1850.0, εr=1.98).material is LEVISIM_DEFAULT_SILICA
    sphere = Sphere()
    for attribute in BEAD_ATTRIBUTES:
        b, s = getattr(bead, attribute), getattr(sphere, attribute)
        assert type(b) is type(s) and b == s, attribute


def test_bead_keywords_build_an_ad_hoc_material():
    bead = Bead(r=62.5*nm, ρ=2200.0, εr=2.1, Φ=1.2)
    assert bead.material.ρ == 2200.0 and bead.material.εr() == 2.1
    assert bead.material.name.startswith('ad hoc')
    malitson = Bead(ρ=2201.0, permittivity_model='Malitson', λ=1550*nm)
    sphere = Sphere(material=FUSED_SILICA, λ=1550*nm)
    for attribute in ['ρ', 'εr', 'n', 'V', 'm', 'α']:
        assert getattr(malitson, attribute) == getattr(sphere, attribute), attribute


def test_bead_accepts_a_material_but_not_together_with_rho_or_epsilon():
    bead = Bead(r=100*nm, material=SILICON, λ=1550*nm)
    assert bead.εr == SILICON.εr(1550*nm)
    assert bead.permittivity_model is None
    with pytest.raises(ValueError, match='either material or'):
        Bead(material=SILICON, ρ=2000.0)


def test_bead_rho_and_epsilon_setters_do_not_touch_shared_materials():
    bead = Bead()
    bead.ρ = 2000.0
    bead.εr = 2.0
    assert bead.ρ == 2000.0 and bead.εr == 2.0 and bead.m == 2000.0 * bead.V
    assert LEVISIM_DEFAULT_SILICA.ρ == 1850.0 and LEVISIM_DEFAULT_SILICA.εr() == 1.98
    assert Bead().ρ == 1850.0


def test_bead_numbers_are_pinned():
    """Values computed on main before the refactor (levisim 0.1.0); exact equality, except for α, which is
    proportional to ε0: scipy < 1.15 has the CODATA 2018 value and later versions the CODATA 2022 one,
    6.8 × 10⁻¹⁰ apart (the oldest supported scipy, 1.13, gives 5.947370762838848e-33)"""
    bead = Bead()
    assert bead.m == 1.6822237250824491e-18
    assert bead.α == pytest.approx(5.947370766869056e-33, rel=1e-9)
    assert Bead(permittivity_model='Malitson').εr == 2.101430006759768


def test_bead_info_str_is_unchanged_and_sphere_has_its_own():
    assert Bead().info_str.splitlines()[1:] == [
        '    r = 60.1 nm',
        '    ρ = 1850.0 kg/m³',
        '    εr = 1.98',
        '    permittivity_model = None',
        '',
        f'    volume = {Bead().V:.5e} m³',
        f'    area = {Bead().A:.5e} m²',
        f'    mass = {Bead().m / 1e-18:.5f} fg',
        f'    weight = {Bead().weight / 1e-18:.5f} aN',
        f'    refractive index, n = {Bead().n:.5f}',
        f'    α = {Bead().α:.5e} Fm²',
        '',
        '    trapping position phase, Φ/2π = x0/λtw = 0.250',
        ')',
    ]
    assert 'material = vaterite' in Sphere(material=VATERITE).info_str
    assert 'source of ρ' in GOLD.info_str


def test_trap_particle_alias():
    sphere = Sphere(r=80*nm)
    trap = Trap(particle=sphere)
    assert trap.particle is trap.bead is sphere
    assert isinstance(Trap().particle, Bead)
    with pytest.raises(ValueError, match='either bead or particle'):
        Trap(bead=Bead(), particle=sphere)

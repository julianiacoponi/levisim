"""
Pins the package's public export surface

`from levisim import *` is how nearly every notebook imports this package (38 of the 41 in
the uploaded set), so the set of names it provides is load-bearing API. These tests make an
accidental change to it fail the suite rather than surface later as a NameError in a notebook.
"""
import pytest

import levisim
from levisim.core import colours, constants, units


def exported_names(module):
    """The names `from <module> import *` would bind"""
    declared = getattr(module, '__all__', None)
    if declared is not None:
        return set(declared)
    return {n for n in vars(module) if not n.startswith('_')}


def test_every_subpackage_declares_its_export_surface():
    """Each __init__ should say what it exports rather than leaving it to wildcard side effects"""
    from levisim import analysis, core, data, physics
    from levisim.multiphysics import enums, utils
    import levisim.multiphysics as multiphysics
    for package in (core, physics, data, analysis, multiphysics, enums, utils):
        assert hasattr(package, '__all__'), f'{package.__name__} has no __all__'
        assert package.__all__, f'{package.__name__}.__all__ is empty'


def test_data_modules_declare_bounded_surfaces():
    """colours, constants and units are wildcard-imported, so their surface must be declared"""
    assert len(colours.__all__) == 46
    assert len(constants.__all__) == 20
    assert len(units.__all__) == 48


def test_constants_reexports_are_listed():
    """
    The scipy re-exports must appear in __all__

    A linter reports them as unused imports; they are the package's physics symbols and
    removing them breaks every module. Listing them makes the intent explicit.
    """
    for symbol in ['π', 'ħ', 'c', 'kB', 'e', 'u', 'ε0', 'μ0', 'g', 'Avogadro']:
        assert symbol in constants.__all__, symbol


@pytest.mark.parametrize('name', [
    # a representative slice of the surface notebooks rely on
    'MultiTrap', 'Trap', 'Cavity', 'Bead', 'Gas', 'Tweezer', 'FocussingOptics', 'Kick',
    'Dimension', 'Simulation', 'Experiment', 'Trace', 'Test', 'logger',
    'kHz', 'MHz', 'Hz', 'ms', 'ns', 'mbar', 'fg', 'ħ', 'kB', 'π',
    'QuantumLinearTheoryMethod', 'SpectrumType', 'NoiseVectorType',
    'DEFAULT_gas_kwargs', 'DEFAULT_cavity_kwargs', 'PHASE_DEPENDENT_toy_params',
])
def test_expected_name_is_importable_from_the_top_level(name):
    """`from levisim import *` must keep providing the names notebooks use"""
    assert hasattr(levisim, name), f'levisim.{name} is gone'


def test_submodule_objects_are_not_part_of_the_star_surface():
    """
    `from levisim import *` should not bind submodule objects

    These used to leak in as a side effect of wildcard imports without __all__. They are still
    reachable as levisim.core.units and so on; they are simply no longer dumped into the
    caller's namespace. No notebook in the uploaded set referenced them that way.
    """
    star_surface = exported_names(levisim)
    for submodule in ['colours', 'constants', 'units', 'comparison', 'shelf', 'animations']:
        assert submodule not in star_surface, f'{submodule} is back in the star surface'


def test_lazy_fixtures_stay_out_of_the_star_surface():
    """
    The heavy shelf fixtures must not be in __all__

    `from ... import *` resolves every name in __all__, which would build all 17 MultiTrap
    instances eagerly and defeat the lazy __getattr__. They remain reachable by attribute
    access and by explicit import.
    """
    from levisim.multiphysics.utils import shelf
    for name in shelf._LAZY_ATTRIBUTES:
        assert name not in shelf.__all__, name
        assert hasattr(shelf, name), name

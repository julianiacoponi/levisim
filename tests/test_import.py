"""
Import-level regression tests

These guard the two problems that made `import levisim` impossible on a fresh clone:
  - heavy `MultiTrap` fixtures being constructed at module import time
  - optional third-party packages being hard import-time requirements
"""
import pytest


def test_import_levisim_succeeds():
    """A bare `import levisim` must work with no optional packages and no pickle data"""
    import levisim
    assert levisim.__name__ == 'levisim'


def test_no_multitrap_fixtures_built_at_import():
    """
    Importing levisim must not construct any of the heavy shelf fixtures

    These used to be built eagerly - 17 MultiTrap instances - which both made every import
    slow and made it fail outright when the power-series pickles were absent.
    """
    import levisim  # noqa: F401
    from levisim.multiphysics.utils import shelf

    eagerly_built = [name for name in shelf._LAZY_ATTRIBUTES if name in vars(shelf)]
    assert eagerly_built == [], f'built at import time: {eagerly_built}'


def test_lazy_fixture_resolves_through_every_access_path():
    """All three historical ways of reaching FORTRAN_8x2 must give the same cached object"""
    import levisim
    from levisim.multiphysics.utils.shelf import FORTRAN_8x2 as from_shelf
    from levisim import FORTRAN_8x2 as from_package

    assert from_shelf is from_package
    assert levisim.FORTRAN_8x2 is from_shelf
    assert type(from_shelf).__name__ == 'MultiTrap'


def test_unknown_attribute_still_raises_attribute_error():
    """The lazy __getattr__ must not swallow genuine typos"""
    import levisim
    from levisim.multiphysics.utils import shelf

    with pytest.raises(AttributeError, match='NOT_A_REAL_FIXTURE'):
        levisim.NOT_A_REAL_FIXTURE
    with pytest.raises(AttributeError, match='NOT_A_REAL_FIXTURE'):
        shelf.NOT_A_REAL_FIXTURE


def test_lazy_fixtures_are_listed_in_dir():
    """dir() must still advertise the fixtures, which __getattr__ alone would hide"""
    from levisim.multiphysics.utils import shelf
    for name in shelf._LAZY_ATTRIBUTES:
        assert name in dir(shelf)


@pytest.mark.parametrize(
    'module_path, attribute',
    [
        ('levisim.core.logger', 'elastic_search_client'),
        ('levisim.core.testing', 'notification'),
        ('levisim.multiphysics.utils.comparison', 'cv2'),
    ],
)
def test_optional_dependencies_are_guarded(module_path, attribute):
    """
    Each optional package must resolve to either the real thing or None, never an ImportError

    The attribute is None when the package is not installed, which is the whole point - the
    module must still import.
    """
    import importlib
    module = importlib.import_module(module_path)
    assert hasattr(module, attribute)


def test_require_cv2_reports_a_helpful_error_when_opencv_is_absent():
    """The cv2 guard must raise a named, actionable ImportError rather than a NameError"""
    from levisim.multiphysics.utils import comparison

    if comparison.cv2 is None:
        with pytest.raises(ImportError, match='OpenCV'):
            comparison._require_cv2()
    else:
        comparison._require_cv2()  # installed, so this must not raise

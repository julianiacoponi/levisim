"""
Packaging: the version, the shipped power-series pickles and where MultiTrap looks for them

The three small power-series pickles ship inside the package (levisim/multiphysics/
power_series_data/) so that `MultiTrap(load_in_power_series=True)` works from a pip install, not
only from a clone; larger series can be generated into a folder given by `power_series_dir=` or
`$LEVISIM_POWER_SERIES_DIR`.
"""
import shutil
from pathlib import Path

import pytest

import levisim
from levisim.multiphysics.multitrap import POWER_SERIES_DATA_DIR, MultiTrap, find_power_series_file
from levisim.multiphysics.utils.shelf import OFF_NODE_toy_params_pairs

REPO_ROOT = Path(__file__).resolve().parent.parent
SHIPPED = ['2D_1x2_up_to_2.pickle', '4D_2x2_up_to_4.pickle', '6D_3x2_up_to_6.pickle']


def read_pyproject():
    try:
        import tomllib  # Python 3.11+
    except ImportError:  # Python 3.10
        tomllib = pytest.importorskip('tomli')
    with open(REPO_ROOT / 'pyproject.toml', 'rb') as file:
        return tomllib.load(file)


def test_version_matches_pyproject():
    declared = read_pyproject()['project']['version']
    assert levisim.__version__ == declared


def test_package_data_declares_the_shipped_pickles():
    package_data = read_pyproject()['tool']['setuptools']['package-data']['levisim.multiphysics']
    assert 'power_series_data/*.pickle' in package_data
    for name in SHIPPED:
        assert (POWER_SERIES_DATA_DIR / name).is_file()


def test_shipped_pickles_are_found_without_configuration(monkeypatch):
    monkeypatch.delenv('LEVISIM_POWER_SERIES_DIR', raising=False)
    for name in SHIPPED:
        assert find_power_series_file(name) == POWER_SERIES_DATA_DIR / name


def test_power_series_dir_and_environment_take_precedence(tmp_path, monkeypatch):
    name = SHIPPED[1]
    explicit, from_env = tmp_path / 'explicit', tmp_path / 'env'
    for folder in (explicit, from_env):
        folder.mkdir()
        shutil.copy(POWER_SERIES_DATA_DIR / name, folder / name)
    monkeypatch.setenv('LEVISIM_POWER_SERIES_DIR', str(from_env))
    assert find_power_series_file(name, explicit) == explicit / name
    assert find_power_series_file(name) == from_env / name


def test_missing_pickle_says_where_it_looked_and_how_to_make_it(tmp_path, monkeypatch):
    monkeypatch.delenv('LEVISIM_POWER_SERIES_DIR', raising=False)
    with pytest.raises(FileNotFoundError, match='generate_power_series_pickles'):
        find_power_series_file('8D_4x2_up_to_8.pickle', tmp_path)


def test_multitrap_loads_a_series_from_power_series_dir(tmp_path):
    name = SHIPPED[1]
    shutil.copy(POWER_SERIES_DATA_DIR / name, tmp_path / name)
    mtrap = MultiTrap(
        num_traps=2, Cartesian_axes_per_trap=2, toy_params=OFF_NODE_toy_params_pairs,
        load_in_power_series=True, power_series_dir=tmp_path,
    )
    assert len(mtrap.deranged_determinant_by_order) > 0


def test_citation_and_changelog_agree_with_the_version():
    """The release workflow refuses to upload unless these agree, so check them on every push"""
    import re
    version = read_pyproject()['project']['version']
    citation = (REPO_ROOT / 'CITATION.cff').read_text()
    assert f'\nversion: {version}\n' in citation
    assert re.search(r'^date-released: \d{4}-\d{2}-\d{2}$', citation, re.M)
    changelog = (REPO_ROOT / 'CHANGELOG.md').read_text()
    assert re.search(rf'^## \[{re.escape(version)}\] - \d{{4}}-\d{{2}}-\d{{2}}$', changelog, re.M)

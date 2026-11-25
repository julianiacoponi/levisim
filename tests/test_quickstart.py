"""
The README and getting-started quickstart runs, on a linearly stable system, without a WARNING

The quickstart's 2×2 physical system is unstable for a tweezer phase difference |φ2 - φ1| of
about 54-126° (modulo 180°), which includes the second built-in tweezer's 90°, so the quickstart
sets φ2 = 30°.
"""
import re
from pathlib import Path

import matplotlib
import numpy as np
import pytest

from levisim.core.logger import logger
from levisim.core.units import degrees
from levisim.multiphysics.multitrap import MultiTrap
from levisim.multiphysics.utils.shelf import DEFAULT_cavity_kwargs, DEFAULT_gas_kwargs

matplotlib.use('Agg')

REPO_ROOT = Path(__file__).resolve().parent.parent


def quickstart_code(path):
    text = (REPO_ROOT / path).read_text(encoding='utf-8')
    section = text[text.index('## Quickstart'):]
    return re.search(r'```python\n(.*?)```', section, flags=re.DOTALL).group(1)


@pytest.fixture
def warnings_logged():
    captured = []
    handler_id = logger.add(lambda message: captured.append(message.record['message']), level='WARNING', format='{message}')
    yield captured
    logger.remove(handler_id)


def max_growth_rate(φ1, φ2):
    mtrap = MultiTrap(
        num_traps=2, Cartesian_axes_per_trap=2,
        gas_kwargs=DEFAULT_gas_kwargs, cavity_kwargs=DEFAULT_cavity_kwargs,
        φn_dict={1: φ1, 2: φ2}, ω_range=np.array([0.0, 1.0]),
    )
    return np.linalg.eigvals(mtrap.drift_matrix).real.max()


@pytest.mark.parametrize('path', ['README.md', 'docs/getting_started.md'])
def test_quickstart_runs_on_a_stable_system_without_warnings(path, tmp_path, monkeypatch, warnings_logged):
    monkeypatch.chdir(tmp_path)
    namespace = {}
    exec(compile(quickstart_code(path), path, 'exec'), namespace)
    assert not warnings_logged
    assert np.linalg.eigvals(namespace['mtrap'].drift_matrix).real.max() < 0
    assert np.all(np.isfinite(namespace['S_hh'])) and namespace['S_hh'].min() > 1
    assert (tmp_path / 'quickstart_homodyne.png').exists()


@pytest.mark.parametrize('φ1, φ2, stable', [
    (0, 0, True), (0, 30, True), (0, 50, True), (0, 60, False), (0, 90, False),
    (20, 110, False), (0, 120, False), (0, 130, True), (0, 270, False),
])
def test_stability_depends_on_the_phase_difference(φ1, φ2, stable):
    assert (max_growth_rate(φ1*degrees, φ2*degrees) < 0) == stable


def test_builtin_phases_log_the_instability_warning(warnings_logged):
    MultiTrap(num_traps=2, Cartesian_axes_per_trap=2, gas_kwargs=DEFAULT_gas_kwargs, cavity_kwargs=DEFAULT_cavity_kwargs)
    assert any('positive real part' in message for message in warnings_logged)


def test_readme_phonon_occupancy_snippet_gives_the_quoted_numbers(tmp_path, monkeypatch):
    """The README's second quickstart snippet (steady-state occupancies) runs on the quickstart system"""
    monkeypatch.chdir(tmp_path)
    text = (REPO_ROOT / 'README.md').read_text(encoding='utf-8')
    section = text[text.index('## Quickstart'):text.index('## Three papers in levisim')]
    first, second = re.findall(r'```python\n(.*?)```', section, flags=re.DOTALL)[:2]
    namespace = {}
    exec(compile(first + second, 'README.md', 'exec'), namespace)
    assert namespace['labels'] == ['x1', 'y1', 'x2', 'y2']
    assert np.allclose(namespace['n'], [139, 161, 76, 486], atol=0.5)

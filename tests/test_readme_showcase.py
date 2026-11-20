"""
The README showcase: its code snippets run, and the scripts behind its figures give the numbers the
README and docs/showcase/ quote

The snippets are the ```python blocks between <!-- SHOWCASE-START --> and <!-- SHOWCASE-END --> in
README.md, each run on its own as a reader would paste it. The scripts are those in
docs/guide/scripts/showcase/, run with compute(quick=True) (coarser grids, shorter simulations).
"""
import importlib.util
import re
from pathlib import Path

import numpy as np
import pytest

from levisim.core.units import degrees, kHz

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / 'docs' / 'guide' / 'scripts' / 'showcase'


def readme_showcase_snippets():
    text = (REPO / 'README.md').read_text(encoding='utf-8')
    section = text.split('<!-- SHOWCASE-START -->')[1].split('<!-- SHOWCASE-END -->')[0]
    return re.findall(r'```python\n(.*?)\n```', section, re.S)


SNIPPETS = readme_showcase_snippets()


def test_the_readme_has_one_snippet_per_example():
    assert len(SNIPPETS) == 4


@pytest.mark.parametrize(
    'snippet',
    [pytest.param(s, marks=pytest.mark.slow) if 'Simulation(' in s else s for s in SNIPPETS],
    ids=['prr5', 'prr6', 'prr8', 'stability'],
)
def test_readme_showcase_snippet_runs(snippet):
    namespace = {}
    exec(compile(snippet, 'README.md', 'exec'), namespace)


def load(name):
    spec = importlib.util.spec_from_file_location(f'showcase_{name}', SCRIPTS / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def draw_both(module, data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.close(module.static_figure(data))
    if importlib.util.find_spec('plotly') is not None:
        figure = module.interactive_figure(data)
        assert len(figure.data) > 0


def test_prr5_cancellation_point():
    module = load('prr5_mode_orientations')
    data = module.compute(quick=True)
    Φc = module.cancellation_phase(data, -450 * kHz)
    assert 0.24 * np.pi < Φc < 0.26 * np.pi
    # the cancellation point moves towards the node as the detuning approaches the mechanical frequencies
    assert module.cancellation_phase(data, -150 * kHz) > Φc + 0.04 * np.pi
    # S_xy changes sign between the node and the antinode: at the node the higher-frequency (x) peak
    # is negative, at Φ = 0.05π it is positive
    ω = data['ω_range']
    high = ω > 145 * kHz
    assert data['S_xy'][0][high].min() < -abs(data['S_xy'][0][high].max())
    assert data['S_xy'][-1][high].max() > abs(data['S_xy'][-1][high].min())
    draw_both(module, data)


def test_prr6_cospectrum_gives_the_quadrant():
    module = load('prr6_directional_force')
    data = module.compute(quick=True)
    plus, minus = data['theory_45'], data['theory_-45']
    np.testing.assert_allclose(plus['xx'], minus['xx'])
    np.testing.assert_allclose(plus['xy'], -minus['xy'])
    assert np.max(abs(plus['xy'])) > 1e-23
    # the simulated co-spectra follow the theory's sign where it is largest
    for ψ in module.ANGLES:
        f, simulated = data[ψ]['f'], data[ψ]['xy']
        theory = np.interp(f, data['f_theory'], data[f'theory_{ψ}']['xy'])
        strong = abs(theory) > 0.5 * abs(theory).max()
        assert np.mean(np.sign(simulated[strong]) == np.sign(theory[strong])) > 0.7
    draw_both(module, data)


def test_prr8_bright_mode_leaves_the_comb():
    module = load('prr8_mode_comb')
    data = module.compute(quick=True)
    bright = {N: modes[-1] for N, modes in data['modes'].items()}
    Ns = sorted(bright)
    frequencies = [bright[N][0] for N in Ns]
    widths = [bright[N][1] for N in Ns]
    assert all(np.diff(frequencies) < 0) and all(np.diff(widths) > 0)
    for N in Ns:
        assert bright[N][0] < data['bare_spans'][N][0]  # below the comb
    assert abs(bright[1][0] - 286.7) < 0.1
    draw_both(module, data)


def test_stability_window_of_the_default_two_particle_system():
    module = load('stability_tweezer_phase')
    data = module.compute(quick=True)
    lo, hi = module.unstable_window(data)
    assert abs(lo - 54.40) < 0.01 and abs(hi - 125.60) < 0.01
    assert abs(data['max_growth_rate'].max() - 2677.5) < 1
    # the eigenvalues at φ2 and 180° - φ2 coincide
    λ = data['eigenvalues']
    np.testing.assert_allclose(np.sort_complex(λ[1]), np.sort_complex(λ[-2]), rtol=0, atol=1e-6)
    assert abs(data['φ2'][1] + data['φ2'][-2] - 180 * degrees) < 1e-9
    draw_both(module, data)

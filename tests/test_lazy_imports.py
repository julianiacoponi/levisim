"""
`import levisim` and building a MultiTrap do not import sympy or scipy.signal

sympy (about 1 s) is only needed to generate hybridisation power series with sympy permutations
(hybridisation.symmetric_group_with_min_fixed_points), and scipy.signal (about 0.5 s) only for
core.utils.apply_filter and SpectralDensityEstimate.make_window; both are imported inside them.
"""
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

CODE = """
import sys
import numpy as np
import levisim
from levisim.multiphysics.multitrap import MultiTrap
from levisim.multiphysics.enums import QuantumLinearTheoryMethod
mtrap = MultiTrap(num_traps=1, Cartesian_axes_per_trap=3, ω_range=np.linspace(1e5, 1e6, 5))
mtrap.homodyne_spectrum(QuantumLinearTheoryMethod.ANALYTIC_CAVITY_OUTPUT)
print(sorted(name for name in ('sympy', 'scipy.signal') if name in sys.modules))
"""


def test_import_and_multitrap_build_do_not_load_sympy_or_scipy_signal():
    env = {**os.environ, 'PYTHONPATH': str(REPO_ROOT), 'LEVISIM_LOG_FILE': '0'}
    result = subprocess.run([sys.executable, '-c', CODE], capture_output=True, text=True, env=env, cwd=REPO_ROOT, check=True)
    assert result.stdout.strip().splitlines()[-1] == '[]'


def test_lazily_imported_helpers_still_work():
    import numpy as np
    from levisim.analysis.spectral_density_estimate import SpectralDensityEstimate
    from levisim.core.utils import apply_filter
    from levisim.multiphysics.hybridisation import symmetric_group_with_min_fixed_points

    window = SpectralDensityEstimate.make_window('hann', 64)
    assert abs(np.mean(window**2) - 1) < 1e-12
    t = np.arange(4096) / 1e6
    filtered = apply_filter(np.sin(2*np.pi*150e3*t), 0.5e6, 150e3, 'butter', 20e3, plot=False)
    assert filtered.shape == t.shape
    assert len(list(symmetric_group_with_min_fixed_points(4, 0))) == 24

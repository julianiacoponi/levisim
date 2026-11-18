# Getting started

## Installation

`levisim` is installed from a clone of the repository:

```sh
git clone https://github.com/julianiacoponi/levisim.git
cd levisim
pip install numpy scipy matplotlib sympy loguru tqdm   # runtime dependencies
pip install -e .                                       # editable install
```

The runtime dependencies are listed explicitly because `pyproject.toml` does not yet declare
them, and only the **editable** install works for now: `pyproject.toml` names the top-level
package only, so a regular `pip install .` ships `levisim/__init__.py` without its subpackages
and `import levisim` fails with `ModuleNotFoundError: No module named 'levisim.core'`. Both are
fixed by the packaging pull requests (#3, then #6); once they are merged, `pip install .`
installs the package and its dependencies in one step.

Alternatively, skip installing altogether and work from the repo root: the tests are configured
to find the in-tree package (see [Testing](testing.md)), and scripts and notebooks can do the same
with `export PYTHONPATH=/path/to/levisim`.

### Note on importing from `levisim`

The repository and the Python package inside it are both called `levisim`, so there are two
layers of `levisim` directory. Importing `levisim` into a Python session whose current working
directory itself contains a directory called `levisim` - for example the parent folder of the
clone - picks up the wrong one and gives an empty import, by
[name shadowing](https://en.wikipedia.org/wiki/Variable_shadowing). This is one reason to keep
notebooks in a local `notebooks/` directory inside the clone (it is git-ignored).

## Quickstart

Two particles, each moving in $x$ and $y$, in one red-detuned cavity, using the default gas and
cavity parameters from {mod}`levisim.multiphysics.utils.shelf`:

```python
import numpy as np
import matplotlib.pyplot as plt

from levisim.core.plotting import set_plt_rcParams_to_xmgrace_style
from levisim.core.units import kHz
from levisim.multiphysics.enums import QuantumLinearTheoryMethod
from levisim.multiphysics.multitrap import MultiTrap
from levisim.multiphysics.utils.shelf import DEFAULT_cavity_kwargs, DEFAULT_gas_kwargs

set_plt_rcParams_to_xmgrace_style()

mtrap = MultiTrap(
    num_traps=2,
    Cartesian_axes_per_trap=2,
    gas_kwargs=DEFAULT_gas_kwargs,          # 1e-6 mbar, 300 K
    cavity_kwargs=DEFAULT_cavity_kwargs,    # Δ = -2π × 450 kHz, L = 0.96 cm, finesse 40 000
    ω_range=np.arange(125, 165, 0.02) * kHz,
)
mtrap.info  # prints a summary

S_hh = mtrap.homodyne_spectrum(QuantumLinearTheoryMethod.LINEAR_AMPLIFIER_MODEL)

fig, ax = plt.subplots()
ax.semilogy(mtrap.ω_range / kHz, S_hh)
ax.set_xlabel(r'$\omega / 2\pi$ (kHz)')
ax.set_ylabel(r'$S_{hh}(\omega)$ (shot-noise units)')
fig.savefig('quickstart_homodyne.png')
```

The full script, which also marks each motion's trap frequency, is
`docs/guide/scripts/quickstart_homodyne.py`; it runs in about a second after import.

```{figure} guide/figures/quickstart_homodyne.png
:width: 90%
:alt: Homodyne spectrum with three visible peaks between 130 and 160 kHz

Homodyne spectrum $S_{hh}(\omega)$ for the system above. Dotted lines: bare trap frequencies
$\Omega_{x_1}/2\pi = 156.3$, $\Omega_{x_2}/2\pi = 152.1$, $\Omega_{y_2}/2\pi = 137.6$ and
$\Omega_{y_1}/2\pi = 135.8$ kHz. The peaks sit below them (optical spring), and the two $y$
motions appear as a single peak because the cavity hybridises them.
```

Three things to know before going further:

- **Frequencies are angular.** `levisim.core.units.kHz` is $2\pi \times 10^3$ rad/s, so
  `np.arange(125, 165, 0.02) * kHz` is a grid in $\omega$, and dividing by `kHz` gives back
  ordinary kHz for plotting. See [Conventions](concepts/conventions.md).
- **The homodyne spectrum is in shot-noise units.** Far from the mechanical features it tends to
  1 (at $\omega/2\pi = 10$ MHz it is $1 + 2 \times 10^{-8}$).
- **The method is a choice.** `LINEAR_AMPLIFIER_MODEL` is the reference; the other six methods
  reach the same spectra by different routes, with different costs and limits. See
  [Calculation methods](concepts/calculation_methods.md).

Position spectra come from {meth}`~levisim.multiphysics.multitrap.MultiTrap.all_position_power_spectra`,
which returns an array of every motion's PSD and a dict of them keyed by label (`'x1'`, `'y1'`, ...):

```python
psd_array, psd_dict = mtrap.all_position_power_spectra(QuantumLinearTheoryMethod.LINEAR_AMPLIFIER_MODEL)
```

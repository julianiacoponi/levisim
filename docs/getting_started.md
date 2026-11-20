# Getting started

## Installation

The latest release is on [PyPI](https://pypi.org/project/levisim/):

```sh
pip install levisim
```

The notebooks, tests and scripts are not part of the package, so to run them, or to use the
development version, install from a clone of the repository instead:

```sh
git clone https://github.com/julianiacoponi/levisim.git
cd levisim
pip install .        # or `pip install -e .` to edit the code in place
```

Both need Python 3.10 or newer and install the runtime dependencies too. Optional extras: `elk`
(ship logs to Elasticsearch, see [Logging](guide/logging.md)), `notify` (desktop notifications
from long test runs), `video` (OpenCV, for the image-comparison helpers), `notebooks` (JupyterLab
and what the notebooks import), `docs` (Sphinx, to build these pages) and `dev` (pytest and ruff),
e.g. `pip install 'levisim[elk]'` or, from a clone, `pip install '.[dev]'`.

`levisim.__version__` says which version is installed; `CHANGELOG.md` in the repository lists what
changed in each.

Alternatively, skip installing altogether and work from the repo root: the tests are configured
to find the in-tree package (see [Testing](testing.md)), and scripts and notebooks can do the same
with `export PYTHONPATH=/path/to/levisim`.

### Note on importing from `levisim`

The repository and the Python package inside it are both called `levisim`. Importing `levisim` into a Python session whose current working
directory itself contains a directory called `levisim` - for example the parent folder of the
clone - picks up the wrong one and gives an empty import, by
[name shadowing](https://en.wikipedia.org/wiki/Variable_shadowing). This is one reason to keep
notebooks in the `notebooks/` directory inside the clone.

## Quickstart

Two particles, each moving in $x$ and $y$, in one red-detuned cavity, using the default gas and
cavity parameters from {mod}`levisim.multiphysics.utils.shelf`:

```python
import numpy as np
import matplotlib.pyplot as plt

from levisim.core.plotting import set_plt_rcParams_to_xmgrace_style
from levisim.core.units import degrees, kHz
from levisim.multiphysics.enums import QuantumLinearTheoryMethod
from levisim.multiphysics.multitrap import MultiTrap
from levisim.multiphysics.utils.shelf import DEFAULT_cavity_kwargs, DEFAULT_gas_kwargs

set_plt_rcParams_to_xmgrace_style()

mtrap = MultiTrap(
    num_traps=2,
    Cartesian_axes_per_trap=2,
    gas_kwargs=DEFAULT_gas_kwargs,          # 1e-6 mbar, 300 K
    cavity_kwargs=DEFAULT_cavity_kwargs,    # Δ = -2π × 450 kHz, L = 0.96 cm, finesse 40 000
    φn_dict={2: 30*degrees},                # second tweezer's phase (built-in 90°: unstable, see below)
    ω_range=np.arange(125, 165, 0.02) * kHz,
)
mtrap.info  # prints a summary

S_hh = mtrap.homodyne_spectrum(QuantumLinearTheoryMethod.LINEAR_AMPLIFIER_MODEL)

fig, ax = plt.subplots()
ax.semilogy(mtrap.ω_range / kHz, S_hh)
ax.margins(x=0)
ax.set_xlabel(r'$\omega / 2\pi$ (kHz)')
ax.set_ylabel(r'$S_{hh}(\omega)$ (shot-noise units)')
fig.savefig('quickstart_homodyne.png')
```

The full script, which also marks each motion's trap frequency, is
`docs/guide/scripts/quickstart_homodyne.py`; it runs in about a second after import.

`DEFAULT_cavity_kwargs` and `DEFAULT_gas_kwargs` are the standard parameter sets kept in
{mod}`~levisim.multiphysics.utils.shelf`. Leaving `cavity_kwargs` out does not give the same cavity:
`Cavity()`'s own defaults are $L$ = 1.223 cm, finesse 30 000 and $\Delta/2\pi$ = −180 kHz
($\kappa/2\pi$ ≈ 409 kHz), against 0.96 cm, 40 000 and −450 kHz ($\kappa/2\pi$ ≈ 390 kHz) here, with
the same 61 µm waist. Couplings, linewidths and stability differ between the two, so say which set
a number refers to.

```{figure} guide/figures/quickstart_homodyne.png
:width: 90%
:alt: Homodyne spectrum with four peaks between 130 and 160 kHz

Homodyne spectrum $S_{hh}(\omega)$ for the system above (tweezer phases 0° and 30°). Dotted lines:
bare trap frequencies $\Omega_{y_1}/2\pi = 135.8$, $\Omega_{y_2}/2\pi = 137.6$,
$\Omega_{x_2}/2\pi = 152.1$ and $\Omega_{x_1}/2\pi = 156.3$ kHz. The $x$ peaks at 150.2 and
155.1 kHz sit 1.9 and 1.3 kHz below their bare frequencies (optical spring). The cavity hybridises
the two $y$ motions into a broad, bright mode at 131.7 kHz (linewidth 2.8 kHz) and a narrow, nearly
dark mode at 136.5 kHz (linewidth 94 Hz).
```

```{note}
**Stability.** The tweezer phases matter. With unequal phases the cavity-mediated coupling between
the particles gains a non-reciprocal part $\propto \sin(\phi_2 - \phi_1)$, which can anti-damp a
hybridised mode of near-degenerate motions. For this system a phase difference $|\phi_2 - \phi_1|$
between about 54° and 126° (modulo 180°) makes it linearly unstable: with the second tweezer's
built-in phase of 90° one $y$ mode grows at $\mathrm{Re}\,\lambda = +2.7 \times 10^3$ s⁻¹, which is
why the quickstart sets 30°. `MultiTrap` logs a WARNING when its drift matrix has an eigenvalue with
positive real part. The stationary spectra of such a system are still finite numbers, but they are
formal: they describe no physical steady state.
```

Three things to know before going further:

- **Frequencies are angular.** `levisim.core.units.kHz` is $2\pi \times 10^3$ rad/s, so
  `np.arange(125, 165, 0.02) * kHz` is a grid in $\omega$, and dividing by `kHz` gives back
  ordinary kHz for plotting. See [Conventions](concepts/conventions.md).
- **The homodyne spectrum is in shot-noise units.** Far from the mechanical features it tends to
  1 (at $\omega/2\pi = 10$ MHz it is $1 + 6 \times 10^{-8}$).
- **The method is a choice.** `LINEAR_AMPLIFIER_MODEL` is the reference; the other six methods
  reach the same spectra by different routes, with different costs and limits. See
  [Calculation methods](concepts/calculation_methods.md).

Position spectra come from {meth}`~levisim.multiphysics.multitrap.MultiTrap.all_position_power_spectra`,
which returns an array of every motion's PSD and a dict of them keyed by label (`'x1'`, `'y1'`, ...):

```python
psd_array, psd_dict = mtrap.all_position_power_spectra(QuantumLinearTheoryMethod.LINEAR_AMPLIFIER_MODEL)
```

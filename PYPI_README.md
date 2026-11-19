# levisim

`levisim` simulates optically levitated nanoparticles.

- **Several particles, one cavity.** `MultiTrap` solves the linearised quantum Langevin equations
  of N levitated particles, each moving in up to three dimensions, coupled by coherent scattering
  to one shared cavity mode. It returns the homodyne and heterodyne spectra of the cavity output
  and the power and cross spectral densities of every motion. Seven independent calculation
  routes are provided (linear-amplifier models, numerical and power-series hybridisation, and an
  exact closed form that inverts no N×N matrix), which agree with each other to within 5e-13
  (relative) on every system in the test suite and notebooks.
- **One particle, free space.** `Simulation` integrates the stochastic (Langevin) dynamics of a bead
  in a Gaussian-beam tweezer with gas damping, feedback cold damping, a directed force, kicks and
  an optional detection floor, photon recoil and gravity; `SpectralDensityEstimate` and
  `SpectralDensityTheory` analyse the traces.
- **Entanglement.** Covariance-matrix dynamics, an exact multi-tone (Floquet) drive and Gaussian-state
  measures (log-negativity, discord, Duan and Hillery–Zubairy witnesses) for two-tone schemes.

```sh
pip install levisim
```

```python
import numpy as np
from levisim.core.units import kHz, degrees
from levisim.multiphysics.enums import QuantumLinearTheoryMethod
from levisim.multiphysics.multitrap import MultiTrap
from levisim.multiphysics.utils.shelf import DEFAULT_cavity_kwargs, DEFAULT_gas_kwargs

mtrap = MultiTrap(
    num_traps=2,
    Cartesian_axes_per_trap=2,
    gas_kwargs=DEFAULT_gas_kwargs,
    cavity_kwargs=DEFAULT_cavity_kwargs,
    φn_dict={2: 30*degrees},                 # second tweezer's phase
    ω_range=np.arange(125, 165, 0.02) * kHz,  # angular units: kHz is 2π × 10³ rad/s
)
S_hh = mtrap.homodyne_spectrum(QuantumLinearTheoryMethod.LINEAR_AMPLIFIER_MODEL)
```

**Documentation:** <https://julianiacoponi.github.io/levisim/> — user guide, the calculation methods
(with a flowchart), conventions, logging, an API reference, and a gallery of demo notebooks and
reproductions of published experiments.

Optional extras: `levisim[elk]` (logs to Elasticsearch/Kibana via the `levisim-logs` command),
`levisim[notebooks]`, `levisim[docs]`, `levisim[dev]`, `levisim[video]`, `levisim[notify]`.

`levisim` is licensed under the GNU GPL v3 or later. Parts of it were written with an LLM at the
author's request; `DISCLAIMER.md` in the source distribution records which.

"""
Quantum linear theory of several levitated particles coherently scattering into one cavity
(MultiTrap)

`MultiTrap` is assembled from a base class, three mixins, one per family of
`QuantumLinearTheoryMethod`, and a mixin for time-domain simulation, each in its own module of
this package:

- `base.MultiTrapBase` - builds the system (trap physics or toy parameters, dimensions, noise
  correlations), prints its information, owns the spectrum entry points, and dispatches each method to
  its mixin (the light and position noise vectors call each other across the mixins, so the dispatch
  cannot live in any one of them)
- `linear_amplifier_mixin.LinearAmplifierMixin` - methods 1-3: drift matrix, T(ω), c(ω), TNT arrays
- `hybridisation_mixin.HybridisationMixin` - methods 4-5: derangement and hybridisation matrices,
  deranged determinant and adjugate (numerical inversion or power series), x1D noise vector
- `analytic_mixin.AnalyticMixin` - methods 6-7: gbar, D(n), P, M_norm, the 𝔽 / 𝔽̄ transduction
  functions and the closed-form noise vectors
- `simulation_mixin.SimulationMixin` - `simulate`, the time-domain `CavitySimulation` of the system
  (imported lazily)
- `omega_model.ΩModel` - the toy-parameter mechanical frequency models

Import `MultiTrap` and `ΩModel` from here, as before the split.
"""

from levisim.multiphysics.multitrap.analytic_mixin import AnalyticMixin

# PHYSICAL_PER_TRAP_KWARGS and the TIMER / PROFILER / PROGRESS_BAR switches are re-exported so that
# the module attributes of the former single-module multitrap.py stay available
from levisim.multiphysics.multitrap.base import (  # noqa: F401
    PHYSICAL_PER_TRAP_KWARGS,
    PROFILER,
    PROGRESS_BAR,
    TIMER,
    MultiTrapBase,
)
from levisim.multiphysics.multitrap.hybridisation_mixin import (  # noqa: F401
    POWER_SERIES_DATA_DIR,
    POWER_SERIES_PICKLE_DIR,
    HybridisationMixin,
    find_power_series_file,
)
from levisim.multiphysics.multitrap.linear_amplifier_mixin import LinearAmplifierMixin
from levisim.multiphysics.multitrap.omega_model import ΩModel
from levisim.multiphysics.multitrap.simulation_mixin import SimulationMixin

__all__ = [
    'MultiTrap',
    'ΩModel',
]


class MultiTrap(LinearAmplifierMixin, HybridisationMixin, AnalyticMixin, SimulationMixin, MultiTrapBase):
    """
    Class to calculate Quantum Linear Theory for a multi-particle cavity trap

    The calculations of the seven `QuantumLinearTheoryMethod` values come from the three mixins
    (`LinearAmplifierMixin`, `HybridisationMixin`, `AnalyticMixin`); the construction, the spectrum
    entry points and the method dispatch from `MultiTrapBase`. See the package docstring of
    `levisim.multiphysics.multitrap` for what lives where.
    """

"""
Deprecated home of the free-space stochastic simulation

`Simulation` is now `levisim.simulation.FreeSpaceSimulation` (same keyword arguments, attributes
and numbers). The name `Simulation` and this module are kept so that existing code and pickles keep
working; constructing a `Simulation` emits a DeprecationWarning once per session. Both will be
removed in levisim 2.0, the next major release.
"""

import warnings

from levisim.simulation.free_space import FreeSpaceSimulation

__all__ = [
    'Simulation',
]


class Simulation(FreeSpaceSimulation):
    """
    Deprecated alias of `levisim.simulation.FreeSpaceSimulation`

    Identical keyword arguments, attributes and (seeded) numbers; only the name differs. Use
    `FreeSpaceSimulation` in new code (also available as `from levisim import FreeSpaceSimulation`).
    Deprecated since levisim 1.0; removed in levisim 2.0.
    """

    _deprecation_warned = False

    def __init__(self, **kwargs):
        if not Simulation._deprecation_warned:
            Simulation._deprecation_warned = True
            warnings.warn(
                'levisim.Simulation (levisim.data.simulation.Simulation) is deprecated and will be removed '
                'in levisim 2.0; use levisim.FreeSpaceSimulation (levisim.simulation.FreeSpaceSimulation), '
                'which takes the same keyword arguments and gives the same numbers.',
                DeprecationWarning,
                stacklevel=2,
            )
        super().__init__(**kwargs)

"""
Time-domain simulation of a `MultiTrap`: `MultiTrap.simulate`

A small mixin so that the frequency-domain model and its time traces share one entry point. The
simulation classes are imported only when `simulate` is called, so `import levisim` does not load
them through `MultiTrap`.
"""

__all__ = [
    'SimulationMixin',
]


class SimulationMixin:
    """
    `simulate(**kwargs)`: a `levisim.simulation.cavity.CavitySimulation` of this MultiTrap
    """

    def simulate(self, **kwargs):
        """
        Simulates time traces of this MultiTrap and returns the `CavitySimulation`

        Keyword arguments are those of `levisim.simulation.cavity.CavitySimulation`, e.g.
        ``mtrap.simulate(num_samples=2**20, Δt=0.1*μs, seed=1, detection=[Homodyne(θ=45*degrees)])``.
        The simulation holds this MultiTrap by composition; one MultiTrap serves many runs.
        """
        from levisim.simulation.cavity import CavitySimulation
        return CavitySimulation(self, **kwargs)

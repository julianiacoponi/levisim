"""
Step-like impulses (Kick) on the levitated bead
"""

import numpy as np

from levisim.core.constants import c
from levisim.core.units import (
    degrees,
    fN,
    keV,
    ms,
    ns,
)
from levisim.physics.fields.base import ForceElement


class Kick(ForceElement):
    """
    Step-like impulses on the levitated bead

    This class is used both to
    - simulate kicks via Simulation
    - keep track of kicks imported via Experiment

    As a `ForceElement`, a kick is a constant force F0 (cos θ, sin θ, 0) in the x-y plane during
    start ≤ t < end, with F0 = K·pth/duration and pth the gas's thermal momentum.
    """

    simulation_group = 'kick'
    def __init__(
        self,
        **kwargs
    ):
        self.start = kwargs.get('start', 0.0*ms)
        self.duration = kwargs.get('duration', 100*ns)
        self.end = self.start + self.duration

        self.K = kwargs.get('K', 10)  # multiple of the gas's thermal momentum pth = m.vth
        self.θ = kwargs.get('θ', 45*degrees)

        # set these in the simulation loop, once comparison to a gas molecule momentum can be made
        self.p = None
        self.F0 = None
        self.Fx = None
        self.Fy = None

    @property
    def info(self):
        """
        Prints information on the Kick instance
        """
        print(self.info_str)

    @property
    def info_str(self):
        """
        String describing the Kick instance

        NOTE: this was named `info`, which shadowed the `info` method above - so `Kick.info`
        returned the string instead of printing it, and `self.info_str` (which the method
        references) did not exist at all. Renamed to match the info()/info_str convention
        used by Gas, Bead, Trap, Tweezer, Cavity, FocussingOptics, Dimension and MultiTrap.
        """
        F0 = f'{self.F0 / fN:.1f} fN' if self.F0 is not None else 'n/a'
        Fx = f'{self.Fx / fN:.1f} fN' if self.Fx is not None else 'n/a'
        Fy = f'{self.Fy / fN:.1f} fN' if self.Fy is not None else 'n/a'

        p = f'{self.p:.5e} kg·m/s' if self.p is not None else 'n/a'
        p_keVc = f'{self.p / (keV/c):.1f} keV/c' if self.p is not None else 'n/a'
        return (
            f'Kick(id={id(self)}'
            f'\n    start = {self.start / ms:.6f} ms'
            f'\n    end = {self.end / ms:.6f} ms'
            f'\n    duration = {self.duration / ms:.6f} ms'
            f'\n    K = {self.K :.1f}'
            f'\n    θ = {self.θ / degrees:.1f}°'
            f'\n    F0 = {F0}'
            f'\n    Fx = {Fx}'
            f'\n    Fy = {Fy}'
            f'\n    p = {p}'
            f'\n    p = {p_keVc}'
            f'\n)'
        )
    def is_active(self, t):
        """Whether the kick acts at time t, start ≤ t < end"""
        return self.start <= t < self.end

    def force(self, r, t, particle, gas=None):
        """
        F0 (cos θ, sin θ, 0) while the kick is active, else zero, with F0 = K·pth/duration and pth
        the thermal momentum of `gas` (needed; a Trap passes its own)
        [N]
        """
        if not self.is_active(t):
            return np.zeros(3)
        if gas is None:
            raise ValueError('Kick.force needs the gas: its momentum is K times the gas thermal momentum pth')
        F0 = self.K * gas.pth / self.duration
        return np.array([F0 * np.cos(self.θ), F0 * np.sin(self.θ), 0.0])

    def stiffness(self, r0=None, particle=None):
        """
        Zero: a kick does not depend on position
        [N/m]
        """
        return np.zeros((3, 3))

    def __repr__(self):
        return f'Kick(start={self.start}, duration={self.duration}, K={self.K}, θ={self.θ})'

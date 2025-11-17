from levisim.core.constants import c
from levisim.core.units import (
    ms, ns,
    degrees,
    keV,
    fN,
)


class Kick:
    """
    Step-like impulses on the levitated bead

    This class is used both to
    - simulate kicks via Simulation
    - keep track of kicks imported via Experiment
    """
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
    def info(self):
        """
        String describing the Kick instance
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
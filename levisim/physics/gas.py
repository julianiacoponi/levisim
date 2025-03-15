import numpy as np
from functools import cached_property

from levisim.core.constants import π, c, kB, m_air
from levisim.core.units import (
    eV, keV,
    mbar,
)


class Gas:
    """
    The environmental gas that the tweezer levitates the bead in
    """
    def __init__(
        self,
        **kwargs,
    ):
        """
        Initialises a Gas instance
        """
        ############################################################################################
        # DEFINED PROPERTIES
        # ------------------------------------------------------------------------------------------
        # Mass of gas molecule
        # [kg]
        self.m = float(kwargs.get('m', m_air))

        # ------------------------------------------------------------------------------------------
        # Gas pressure
        # [Pa] = [N/m²]
        self.P = float(kwargs.get('P', 4e-3*mbar))

        # ------------------------------------------------------------------------------------------
        # Gas temperature
        # [K]
        self.T = float(kwargs.get('T', 300.0))

        # ------------------------------------------------------------------------------------------
        # Manually overriden value for the thermal gas velocity
        # [m/s]
        self.override_vth = kwargs.get('vth')

    ################################################################################################
    # INFORMATION PRINTING FUNCTIONS
    # ----------------------------------------------------------------------------------------------
    @property
    def info(self):
        """
        Prints information on the Gas instance
        """
        print(self.info_str)

    @property
    def info_str(self):
        """
        String describing the Gas instance
        """
        return (
            f'Gas(id={id(self)}'
            f'\n    m = {self.m :.5e} kg'
            f'\n    P = {self.P / mbar:.5e} mbar'
            f'\n    T = {self.T :.5f} K'
            f'\n'
            f'\n    〈v〉 = {self.vth :.5f} m/s'
            f'\n    〈p〉 = {self.pth :.5e} kg.m/s'
            f'\n    〈p〉 = {self.pth / (keV/c):.5f} keV/c'
            f'\n    〈KE〉 = {self.KE / eV:.5f} eV'
            f'\n    〈KE〉= {self.KE / kB:.5f} K'
            f'\n)'
        )

    ################################################################################################
    # DERIVED PROPERTIES
    # ----------------------------------------------------------------------------------------------
    @cached_property
    def vth(self):
        """
        Average speed of a gas molecule
        [m/s]
        """
        if self.override_vth:
            return float(self.override_vth)

        # NOTE: https://en.wikipedia.org/wiki/Thermal_velocity#In_three_dimensions
        # "Root-mean square" (rms) value has factor = 3
        # "Most probable" (peak of Maxwell-Boltzmann) has a factor = 2
        # Expected value i.e. 〈.〉 has a factor = 8/π
        # TODO: why do we use the expected value and not the other options?
        factor = 8 / π
        return np.sqrt(factor * kB * self.T / self.m)

    @cached_property
    def pth(self):
        """
        Average momentum of a gas molecule
        [kg.m/s]
        """
        return self.m * self.vth

    @cached_property
    def KE(self):
        """
        Average kinetic energy of a gas molecule
        [J]
        """
        return 0.5 * self.m * self.vth**2


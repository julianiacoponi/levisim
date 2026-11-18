"""
The background gas (Gas) that the bead is levitated in
"""

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
            f'\n    〈p〉 = {self.pth :.5e} kg·m/s'
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
        # NOTE: answer to "why the expected value and not the other options?": it is not a free choice.
        # The gas damping (Trap.Γth) is the Epstein (Phys. Rev. 23, 710 (1924)) free-molecular drag,
        # as used by Cavalleri et al. (Phys. Lett. A 374, 3365 (2010)): the momentum delivered to the
        # bead is set by the molecular flux onto its surface, n〈v〉/4 per unit area, and integrating
        # over the Maxwell-Boltzmann distribution produces the mean speed 〈v〉 = √(8kBT/πm) itself.
        # Indeed Γth = (1 + π/8) m_gas 〈v〉 P / (kB T ρ r) is exactly the Epstein drag force per unit
        # momentum, F/(M v) = (1 + π/8)(4π/3) r² n m_gas 〈v〉 / M, for diffuse reflection with full
        # accommodation. Using v_rms (factor 3) or v_p (factor 2) would mis-scale Γth by
        # √(3π/8) = 1.085 or √(π/4) = 0.886.
        factor = 8 / π
        return np.sqrt(factor * kB * self.T / self.m)

    @cached_property
    def pth(self):
        """
        Average momentum of a gas molecule
        [kg·m/s]
        """
        return self.m * self.vth

    @cached_property
    def KE(self):
        """
        Average kinetic energy of a gas molecule
        [J]
        """
        return 0.5 * self.m * self.vth**2


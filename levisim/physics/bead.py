from functools import cached_property

from levisim.core.constants import π, ε0, g
from levisim.core.units import (
    μm, nm,
    fg,
    aN,
)


class Bead:
    """
    Defines a spherical nanoparticle bead
    """
    def __init__(
        self,
        **kwargs,
    ):
        """
        Initialises a Bead instance
        """
        ############################################################################################
        # DEFINED PROPERTIES
        # ------------------------------------------------------------------------------------------
        # Bead radius
        # [m]
        # Markus said the batch they use was 'nominal 142nm diameter'
        # Tania uses r=60.1nm
        self.r = float(kwargs.get('r', 60.1*nm))

        # ------------------------------------------------------------------------------------------
        # Bead density
        # [kg/m³]
        # https://en.wikipedia.org/wiki/Silicon_dioxide
        # quotes ρ_SiO2_amorphous = 2.196 g/cm³ = 2196.0 kg/m³
        # https://www.microparticles-shop.de/Monodisperse-Particles-for-Research-Purposes/Silica-particles-SiO2-R-in-the-size-range-150-nm-25-m:::1_6.html
        # the spec-sheet for the particles used in the experiments online
        # quotes ρ_SiO2_R = 1.85 g/cm³ = 1850.0 kg/m³
        self.ρ = float(kwargs.get('ρ', 1850.0))

        # ------------------------------------------------------------------------------------------
        # Bead relative permittivity
        # [·]
        # NOTE: Refractive index depends on wavelength, so this should be Tweezer dependent...
        # TODO: make the above happen! Use the dispersion formula at refractiveindex.info?
        # n = √(εR*μR); μR = 1
        # εR = n²
        # for SiO2 @ λ = 1064nm
        # https://refractiveindex.info/?shelf=main&book=SiO2&page=Malitson
        # n = 1.4496
        # https://refractiveindex.info/?shelf=main&book=SiO2&page=Arosa
        # n = 1.4497
        # εR = n² = 2.1016...
        # Tania uses εR=1.98, implies n = 1.407...
        # TODO: ask why?
        self.εR = float(kwargs.get('εR', 1.98))

        ############################################################################################
        # MULTI-PHYSICS PROPERTIES
        # ------------------------------------------------------------------------------------------
        # Trapping position away from the antinode of a the cavity standing
        # [rad]
        antinode, halfnode, node  = 0, π/4, π/2
        self.Φ = float(kwargs.get('Φ', node))

    ################################################################################################
    # INFORMATION PRINTING FUNCTIONS
    # ----------------------------------------------------------------------------------------------
    @property
    def info(self):
        """
        Prints info on the Bead object
        """
        print(
            f'Bead(id={id(self)}'
            f'\n    r = {self.r / nm:.1f} nm'
            f'\n    ρ = {self.ρ:.1f} kg/m³'
            f'\n    εR = {self.εR:.2f}'
            f'\n'
            f'\n    Volume = {self.V:.5e} m³'
            f'\n    Area = {self.A:.5e} m²'
            f'\n    mass = {self.m / fg:.5f} fg'
            f'\n    weight = {self.weight / aN:.5f} aN'
            f'\n    α = {self.α:.5e} Fm²'
            f'\n'
            f'\n    Φ/2π = x0/λtw = {self.Φ /(2*π):.3f}'
            f'\n)'
        )
        return

    ################################################################################################
    # DERIVED PROPERTIES
    # ----------------------------------------------------------------------------------------------
    @cached_property
    def V(self):
        """
        Volume of the bead
        [m³]
        """
        return (4/3)*π * self.r**3

    @cached_property
    def A(self):
        """
        Surface area of the bead
        [m²]
        """
        return 4*π * self.r**2

    @cached_property
    def m(self):
        """
        Mass of the bead
        [kg]
        """
        return self.ρ * self.V

    @cached_property
    def weight(self):
        """
        Bead weight under Earth's gravity
        [N]
        """
        return self.m * g

    @cached_property
    def α(self):
        """
        Polarisability of the bead
        [Capacitance × Area] = [Fm²]
        [F] = [Farad] = Capacitance = Charge / Electric Potential = [C/V]
        NOTE: https://en.wikipedia.org/wiki/Clausius–Mossotti_relation#Lorentz–Lorenz_equation
        """
        return 3 * self.V * ε0 * (self.εR - 1) / (self.εR + 2)

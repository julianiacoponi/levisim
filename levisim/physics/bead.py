import numpy as np
from functools import cached_property

from levisim.core.constants import π, ε0, g
from levisim.core.units import (
    μm, nm,
    fg,
    aN,
)


def SiO2_refractive_index_Malitson(λ):
    """
    Refractive index of fused silica (SiO2) at vacuum wavelength λ, from the Sellmeier formula of
    I. H. Malitson, J. Opt. Soc. Am. 55, 1205 (1965), valid for 0.21-3.71 μm (20 °C):
    n² - 1 = Σi Bi λ² / (λ² - Ci²), λ in μm,
    B = (0.6961663, 0.4079426, 0.8974794), C = (0.0684043, 0.1162414, 9.896161) μm
    e.g. n(1064 nm) = 1.44963
    [·]
    """
    λ_μm = np.asarray(λ, dtype=float) / μm
    if np.any((λ_μm < 0.21) | (λ_μm > 3.71)):
        raise ValueError(f'Malitson formula is valid for 0.21-3.71 μm, got λ = {λ_μm} μm')
    B = (0.6961663, 0.4079426, 0.8974794)
    C = (0.0684043, 0.1162414, 9.896161)
    λ2 = λ_μm**2
    n2 = 1 + sum(Bi * λ2 / (λ2 - Ci**2) for Bi, Ci in zip(B, C))
    return float(np.sqrt(n2)) if np.ndim(n2) == 0 else np.sqrt(n2)


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
        # n = √(εr*μr); μr = 1
        # εr = n²
        # for SiO2 @ λ = 1064nm
        # https://refractiveindex.info/?shelf=main&book=SiO2&page=Malitson
        # n = 1.4496
        # https://refractiveindex.info/?shelf=main&book=SiO2&page=Arosa
        # n = 1.4497
        # εr = n² = 2.1016...
        # NOTE: the refractive index depends on wavelength. Opt in with permittivity_model='Malitson',
        # which sets εr = n(λ)² from the Malitson (1965) Sellmeier formula for fused silica at the
        # wavelength `λ` (default 1064 nm; pass the Tweezer's λ). The default stays εr = 1.98.
        # NOTE (owner): Tania uses εr = 1.98, i.e. n = 1.4071, below bulk fused silica (n = 1.4496 at
        # 1064 nm). A plausible reason is that the beads are not bulk silica: the manufacturer's density
        # (1.85 g/cm³, used for ρ above) is 84% of bulk (2.196 g/cm³). But as a Lorentz-Lorenz void
        # mixture, (εr-1)/(εr+2) ∝ density, εr = 1.98 corresponds to 92% of bulk density (2.01 g/cm³),
        # while ρ = 1.85 g/cm³ would give εr = 1.877 (n = 1.370) - so the two defaults are not
        # mutually consistent under that model. Where does 1.98 come from (a measurement, a supplier
        # value, a fit)?
        self.permittivity_model = kwargs.get('permittivity_model', None)
        self.λ = float(kwargs.get('λ', 1064*nm))
        if self.permittivity_model is None:
            self.εr = float(kwargs.get('εr', 1.98))
        elif str(self.permittivity_model).lower() == 'malitson':
            if 'εr' in kwargs:
                raise ValueError("Pass either εr or permittivity_model='Malitson', not both")
            self.εr = SiO2_refractive_index_Malitson(self.λ)**2
        else:
            raise ValueError(f"Unknown permittivity_model={self.permittivity_model!r}; use None or 'Malitson'")

        ############################################################################################
        # MULTI-TRAP PROPERTIES
        # ------------------------------------------------------------------------------------------
        # Trapping position phase away from the antinode of a the cavity standing
        # Φ = k·x0
        # [·]
        antinode, halfnode, node  = 0, π/4, π/2
        self.Φ = float(kwargs.get('Φ', node))

    ################################################################################################
    # INFORMATION PRINTING FUNCTIONS
    # ----------------------------------------------------------------------------------------------
    @property
    def info(self):
        """
        Prints information on the Bead instance
        """
        print(self.info_str)

    @property
    def info_str(self):
        """
        String describing the Gas instance
        """
        return (
            f'Bead(id={id(self)}'
            f'\n    r = {self.r / nm:.1f} nm'
            f'\n    ρ = {self.ρ:.1f} kg/m³'
            f'\n    εr = {self.εr:.2f}'
            f'\n    permittivity_model = {self.permittivity_model}'
            f'\n'
            f'\n    volume = {self.V:.5e} m³'
            f'\n    area = {self.A:.5e} m²'
            f'\n    mass = {self.m / fg:.5f} fg'
            f'\n    weight = {self.weight / aN:.5f} aN'
            f'\n    refractive index, n = {self.n:.5f}'
            f'\n    α = {self.α:.5e} Fm²'
            f'\n'
            f'\n    trapping position phase, Φ/2π = x0/λtw = {self.Φ /(2*π):.3f}'
            f'\n)'
        )

    ################################################################################################
    # DERIVED PROPERTIES
    # ----------------------------------------------------------------------------------------------
    @cached_property
    def n(self):
        """
        Refractive index of the (non-magnetic, μr=1) bead
        [·]
        """
        return np.sqrt(self.εr)

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
        Static polarisability of the bead
        [Capacitance × Area] = [Fm²]
        [F] = [Farad] = Capacitance = Charge / Electric Potential = [C/V]
        NOTE: https://en.wikipedia.org/wiki/Clausius–Mossotti_relation#Lorentz–Lorenz_equation
        """
        return 3 * self.V * ε0 * (self.εr - 1) / (self.εr + 2)

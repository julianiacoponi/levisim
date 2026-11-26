"""
The levitated nanoparticle (Bead): a `Sphere` with the original keyword arguments

`Bead` is kept for backward compatibility; it is a `levisim.physics.particle.Sphere` whose material
is built from the original keywords `ρ`, `εr` and `permittivity_model`. Every number it gives is the
same as before the `Material`/`Particle` refactor. The refractive index of fused silica now lives in
`levisim.physics.material` and is re-exported here.
"""

from levisim.core.constants import π
from levisim.core.units import (
    aN,
    fg,
    nm,
)
from levisim.physics.material import Material, SiO2_refractive_index_Malitson
from levisim.physics.particle import Sphere

__all__ = ['Bead', 'SiO2_refractive_index_Malitson']


class Bead(Sphere):
    """
    Defines a spherical nanoparticle bead

    ::

        Bead(r=60.1 nm, ρ=1850.0, εr=1.98, permittivity_model=None, λ=1064 nm, Φ=π/2, charge=0)

    - r: radius [m] (Tania uses 60.1 nm; Markus said the batch they use was 'nominal 142nm diameter')
    - ρ: density [kg/m³]
    - εr: relative permittivity [·]
    - permittivity_model: None (use εr), or 'Malitson' for εr = n(λ)² from the Malitson (1965)
      Sellmeier formula for fused silica at the wavelength λ (pass the Tweezer's λ)
    - λ: vacuum wavelength for the permittivity model [m]
    - Φ: trapping position phase along the cavity axis [·] (default π/2, a node)
    - charge: net charge in units of e [·] (new, default 0)
    - material: a `Material` instead of ρ/εr/permittivity_model (new)

    With none of ρ, εr, permittivity_model or material given (or with exactly ρ = 1850.0 and
    εr = 1.98) the material is the library's `LEVISIM_DEFAULT_SILICA`; otherwise an ad-hoc
    `Material` is built from the keywords. See that material for the sources of the defaults and the
    open question about their mutual consistency.
    """
    def __init__(
        self,
        **kwargs,
    ):
        """
        Initialises a Bead instance
        """
        from levisim.physics.materials import LEVISIM_DEFAULT_SILICA

        material = kwargs.get('material', None)
        permittivity_model = kwargs.get('permittivity_model', None)
        λ = float(kwargs.get('λ', 1064*nm))

        if material is not None:
            clashes = [key for key in ('ρ', 'εr', 'permittivity_model') if kwargs.get(key) is not None]
            if clashes:
                raise ValueError(f'Pass either material or {clashes}, not both')
        else:
            ρ = float(kwargs.get('ρ', 1850.0))
            if permittivity_model is None:
                εr = float(kwargs.get('εr', 1.98))
                if ρ == LEVISIM_DEFAULT_SILICA.ρ and εr == LEVISIM_DEFAULT_SILICA.εr():
                    material = LEVISIM_DEFAULT_SILICA
                else:
                    material = Material(name=f'ad hoc (ρ={ρ!r}, εr={εr!r})', ρ=ρ, permittivity=εr)
            elif str(permittivity_model).lower() == 'malitson':
                if 'εr' in kwargs:
                    raise ValueError("Pass either εr or permittivity_model='Malitson', not both")
                material = Material(
                    name=f'ad hoc (ρ={ρ!r}, Malitson fused-silica εr(λ))', ρ=ρ, permittivity='Malitson',
                    references={'εr': 'Malitson, J. Opt. Soc. Am. 55, 1205 (1965), doi:10.1364/JOSA.55.001205'},
                )
            else:
                raise ValueError(f"Unknown permittivity_model={permittivity_model!r}; use None or 'Malitson'")

        node = π/2
        super().__init__(
            r=float(kwargs.get('r', 60.1*nm)),
            material=material,
            charge=kwargs.get('charge', 0),
            λ=λ,
            Φ=float(kwargs.get('Φ', node)),
        )
        self.permittivity_model = permittivity_model

    ################################################################################################
    # INFORMATION PRINTING FUNCTIONS
    # ----------------------------------------------------------------------------------------------
    @property
    def info_str(self):
        """
        String describing the Bead instance
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

"""
levisim.physics.fields
======================
Force elements: the forces that act on one levitated particle

A `Trap` holds a list of them (``Trap(fields=[...], particle=...)``); the total stiffness at the
linearisation point is the sum of their stiffness tensors. See the concepts page "Traps as sums of
fields".

- `ForceElement`: the base class (force, torque, potential, stiffness, noise, damping_rates)
- `Tweezer`: the optical tweezer (a focussed Gaussian beam), defined in `levisim.physics.tweezer`
- `HarmonicPotential`: F = -K·(r - c)
- `Gravity`: F = m g d̂
- `Feedback`: cold damping -Γfb,j·pj
- `PhotonRecoil`: the scattering force of a tweezer along its axis
- `Kick`: a step-like impulse, defined in `levisim.physics.kick`
- `PaulTrap`: a quadrupole radio-frequency trap for a charged particle (full rf force, secular
  stiffness, Mathieu parameters and stability), with the presets `PaulTrap.ring` and `PaulTrap.linear`
- `UniformElectricField`: F = Q E, a stray or compensation field
- `MagneticTrap`: a static magnetic field acting on a magnetisable (diamagnetic) particle,
  U = -(V/2μ0) B·χ_eff·B, with the field from a source in `levisim.physics.fields.magnetic_sources`
  (`LocalFieldExpansion`, `MultipoleField`, `CurrentLoop`, `CoilPair`, `CuboidMagnet`, `FieldSum`,
  `CallableField`)
- `EddyCurrentDamping`: velocity damping (and its thermal noise) of a conducting sphere in a field
  gradient
"""

from levisim.physics.fields.base import SIMULATION_GROUPS, ForceElement
from levisim.physics.fields.electric import (
    PaulTrap,
    UniformElectricField,
    mathieu_characteristic_exponent,
    mathieu_first_region_stable,
    mathieu_floquet_multiplier,
    mathieu_secular_beta,
)
from levisim.physics.fields.elements import (
    Feedback,
    Gravity,
    HarmonicPotential,
    PhotonRecoil,
)
from levisim.physics.fields.magnetic import (
    EddyCurrentDamping,
    MagneticTrap,
    demagnetising_factors_of,
    effective_susceptibility,
)
from levisim.physics.fields.magnetic_sources import (
    CallableField,
    CoilPair,
    CuboidMagnet,
    CurrentLoop,
    FieldSum,
    LocalFieldExpansion,
    MagneticFieldSource,
    MultipoleField,
    real_solid_harmonic,
)

__all__ = [
    'SIMULATION_GROUPS',
    'CallableField',
    'CoilPair',
    'CuboidMagnet',
    'CurrentLoop',
    'EddyCurrentDamping',
    'Feedback',
    'FieldSum',
    'ForceElement',
    'Gravity',
    'HarmonicPotential',
    'Kick',
    'LocalFieldExpansion',
    'MagneticFieldSource',
    'MagneticTrap',
    'MultipoleField',
    'PaulTrap',
    'PhotonRecoil',
    'Tweezer',
    'UniformElectricField',
    'demagnetising_factors_of',
    'effective_susceptibility',
    'mathieu_characteristic_exponent',
    'mathieu_first_region_stable',
    'mathieu_floquet_multiplier',
    'mathieu_secular_beta',
    'real_solid_harmonic',
]


def __getattr__(name):
    # Tweezer and Kick subclass ForceElement in their own modules, which import this package; they
    # are re-exported lazily to keep the import order free of cycles
    if name == 'Tweezer':
        from levisim.physics.tweezer import Tweezer
        return Tweezer
    if name == 'Kick':
        from levisim.physics.kick import Kick
        return Kick
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')

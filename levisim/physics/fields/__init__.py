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
"""

from levisim.physics.fields.base import SIMULATION_GROUPS, ForceElement
from levisim.physics.fields.elements import (
    Feedback,
    Gravity,
    HarmonicPotential,
    PhotonRecoil,
)

__all__ = [
    'SIMULATION_GROUPS',
    'Feedback',
    'ForceElement',
    'Gravity',
    'HarmonicPotential',
    'Kick',
    'PhotonRecoil',
    'Tweezer',
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

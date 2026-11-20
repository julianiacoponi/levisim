"""
levisim.physics
===============
A module to handle all the physics involved in levitation
"""


__all__ = [
    'Bead',
    'Cavity',
    'Feedback',
    'FocalExpansion',
    'FocussingOptics',
    'ForceElement',
    'Gas',
    'Gravity',
    'HarmonicPotential',
    'Kick',
    'Material',
    'MissingOpticalFieldError',
    'Particle',
    'PaulTrap',
    'PhotonRecoil',
    'Sphere',
    'Trap',
    'Tweezer',
    'UniformElectricField',
    'polarisation_matrix',
]


from levisim.physics.bead import Bead
from levisim.physics.cavity import Cavity
from levisim.physics.fields import (
    Feedback,
    ForceElement,
    Gravity,
    HarmonicPotential,
    PaulTrap,
    PhotonRecoil,
    UniformElectricField,
)
from levisim.physics.focal_expansion import FocalExpansion, polarisation_matrix
from levisim.physics.focussing_optics import FocussingOptics
from levisim.physics.gas import Gas
from levisim.physics.kick import Kick
from levisim.physics.material import Material
from levisim.physics.particle import Particle, Sphere
from levisim.physics.trap import MissingOpticalFieldError, Trap
from levisim.physics.tweezer import Tweezer

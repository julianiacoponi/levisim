"""
levisim.physics
===============
A module to handle all the physics involved in levitation
"""


__all__ = [
    'Bead',
    'Cavity',
    'Cylinder',
    'Disk',
    'Dumbbell',
    'Ellipsoid',
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
    'PhotonRecoil',
    'Rod',
    'Sphere',
    'Spheroid',
    'TabulatedParticle',
    'Trap',
    'Tweezer',
    'polarisation_matrix',
]


from levisim.physics.bead import Bead
from levisim.physics.cavity import Cavity
from levisim.physics.fields import (
    Feedback,
    ForceElement,
    Gravity,
    HarmonicPotential,
    PhotonRecoil,
)
from levisim.physics.focal_expansion import FocalExpansion, polarisation_matrix
from levisim.physics.focussing_optics import FocussingOptics
from levisim.physics.gas import Gas
from levisim.physics.kick import Kick
from levisim.physics.material import Material
from levisim.physics.particle import Particle, Sphere
from levisim.physics.shapes import Cylinder, Disk, Dumbbell, Ellipsoid, Rod, Spheroid
from levisim.physics.tabulated import TabulatedParticle
from levisim.physics.trap import MissingOpticalFieldError, Trap
from levisim.physics.tweezer import Tweezer

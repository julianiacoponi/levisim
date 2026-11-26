"""
levisim.physics
===============
A module to handle all the physics involved in levitation
"""


__all__ = [
    'Bead',
    'Casimir',
    'Cavity',
    'Coulomb',
    'Cylinder',
    'Disk',
    'Dumbbell',
    'EddyCurrentDamping',
    'Ellipsoid',
    'Feedback',
    'FocalExpansion',
    'FocussingOptics',
    'ForceElement',
    'Gas',
    'Gravity',
    'HarmonicPotential',
    'Interaction',
    'Kick',
    'MagneticTrap',
    'Material',
    'MissingOpticalFieldError',
    'NewtonianGravity',
    'OpticalBinding',
    'Particle',
    'PaulTrap',
    'PhotonRecoil',
    'RefractiveIndexTable',
    'Rod',
    'Sphere',
    'Spheroid',
    'TabulatedParticle',
    'Trap',
    'Tweezer',
    'UniformElectricField',
    'polarisation_matrix',
]


from levisim.physics.bead import Bead
from levisim.physics.cavity import Cavity
from levisim.physics.fields import (
    EddyCurrentDamping,
    Feedback,
    ForceElement,
    Gravity,
    HarmonicPotential,
    MagneticTrap,
    PaulTrap,
    PhotonRecoil,
    UniformElectricField,
)
from levisim.physics.focal_expansion import FocalExpansion, polarisation_matrix
from levisim.physics.focussing_optics import FocussingOptics
from levisim.physics.gas import Gas
from levisim.physics.interactions import (
    Casimir,
    Coulomb,
    Interaction,
    NewtonianGravity,
    OpticalBinding,
)
from levisim.physics.kick import Kick
from levisim.physics.material import Material, RefractiveIndexTable
from levisim.physics.particle import Particle, Sphere
from levisim.physics.shapes import Cylinder, Disk, Dumbbell, Ellipsoid, Rod, Spheroid
from levisim.physics.tabulated import TabulatedParticle
from levisim.physics.trap import MissingOpticalFieldError, Trap
from levisim.physics.tweezer import Tweezer

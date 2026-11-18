"""
levisim.multiphysics
===============
A module to handle all the physics involved in multitrap cavity levitation
"""

from levisim.multiphysics.multitrap import *
from levisim.multiphysics.dimension import *
from levisim.multiphysics.hybridisation import *
from levisim.multiphysics.enums import *
from levisim.multiphysics.utils import *


def __getattr__(name):
    """
    Forwards the lazily-built `shelf` fixtures (PEP 562)

    `FORTRAN_8x2`, `PHASE_EQUAL_xy_dict` and `PHASE_DEPENDENT_xy_dict` are built on first
    access rather than at import time, so they are not re-exported by the `import *` above.
    Forwarding them here keeps `from levisim.multiphysics import FORTRAN_8x2` working as it always did.
    """
    from levisim.multiphysics.utils import shelf
    if name in shelf._LAZY_ATTRIBUTES:
        return getattr(shelf, name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')

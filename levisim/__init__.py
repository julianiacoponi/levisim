#!/opt/homebrew/bin/python3
"""
levisim
=======
A library to simulate the dynamics of levitated particles

"""


from levisim.core import *
from levisim.physics import *
from levisim.multiphysics import *
from levisim.data import *
from levisim.analysis import *


def __getattr__(name):
    """
    Forwards the lazily-built `shelf` fixtures (PEP 562)

    `FORTRAN_8x2`, `PHASE_EQUAL_xy_dict` and `PHASE_DEPENDENT_xy_dict` are built on first
    access rather than at import time, so they are not re-exported by the `import *` above.
    Forwarding them here keeps `from levisim import FORTRAN_8x2` working as it always did.
    """
    from levisim.multiphysics.utils import shelf
    if name in shelf._LAZY_ATTRIBUTES:
        return getattr(shelf, name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')

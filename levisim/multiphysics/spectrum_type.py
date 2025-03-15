from enum import Enum, auto

__all__ = [
    'SpectrumType',
]

class SpectrumType(Enum):
    """
    An enumeration class for the type of spectrum to be calculated
    """

    HETERODYNE = auto()
    HOMODYNE = auto()
    PSD = auto()

    def __repr__(self):
        """ Return an appropriate string when printed """

        if self == SpectrumType.HETERODYNE:
            return 'Heterodyne'

        if self == SpectrumType.HOMODYNE:
            return 'Homodyne'

        if self == SpectrumType.PSD:
            return 'PSD'

    @property
    def is_optical(self):
        """
        Boolean for whether the spectrum is optical (heterodyne or homodyne)
        """
        return self in [SpectrumType.HETERODYNE, SpectrumType.HOMODYNE]


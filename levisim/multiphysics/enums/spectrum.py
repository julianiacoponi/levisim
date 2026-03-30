from enum import Enum, auto


__all__ = [
    'SpectrumType',
]


class SpectrumType(Enum):
    """
    An enumeration class for the type of spectrum to be calculated
    """

    HOMODYNE = auto()
    HETERODYNE = auto()
    POSITION_PSD = auto()
    POSITION_CSD = auto()

    def __repr__(self):
        """
        Return an appropriate string when printed
        """

        if self == SpectrumType.HOMODYNE:
            return 'Homodyne'

        if self == SpectrumType.HETERODYNE:
            return 'Heterodyne'

        if self == SpectrumType.POSITION_PSD:
            return 'Position PSD'

        if self == SpectrumType.POSITION_CSD:
            return 'Position CSD'

    @property
    def is_optical(self):
        """
        Boolean for whether the spectrum is optical (heterodyne or homodyne)
        """
        return self in [SpectrumType.HOMODYNE, SpectrumType.HETERODYNE]

    @property
    def is_position(self):
        """
        Boolean for whether the spectrum is for position (power 'PSD' or cross 'CSD')
        """
        return self in [SpectrumType.POSITION_PSD, SpectrumType.POSITION_CSD]

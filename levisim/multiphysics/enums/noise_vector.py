from enum import Enum, auto


__all__ = [
    'NoiseVectorType',
]


class NoiseVectorType(Enum):
    """
    An enumeration class for the type of noise vector to be calculated
    """

    HYBRIDISED_POSITIONS = auto()
    LIGHT_INSIDE_CAVITY = auto()
    CAVITY_OUTPUT_LIGHT = auto()

    def __repr__(self):
        """
        Return an appropriate string when printed
        """

        if self == NoiseVectorType.LIGHT_INSIDE_CAVITY:
            return 'Light inside cavity'

        if self == NoiseVectorType.CAVITY_OUTPUT_LIGHT:
            return 'Cavity output light'

        if self == NoiseVectorType.HYBRIDISED_POSITIONS:
            return 'Hybridised positions'

    @property
    def is_light(self):
        """
        Boolean for whether the noise vector is for light (inside the cavity are its output)
        """
        return self in [NoiseVectorType.LIGHT_INSIDE_CAVITY, NoiseVectorType.CAVITY_OUTPUT_LIGHT]

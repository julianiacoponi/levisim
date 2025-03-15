import numpy as np
from enum import Enum, auto
from levisim.core.constants import π
from levisim.core.units import (
    degrees,
    kHz,
    mbar,
    K,
)

__all__ = [
    'ParameterType',
    'Parameter',

    'light_mech_coupling',
    'phase_of_2nd_tweezer',
    'cavity_detuning',
    'optical_damping',
    'air_pressure',
    'evaluation_order',

    'all_parameters',
]


class ParameterType(Enum):
    """
    An enumeration class for the parameter used in generating animations
    """

    LIGHT_MECH_COUPLING = auto()

    PHASE_OF_2ND_TWEEZER = auto()

    CAVITY_DETUNING = auto()
    OPTICAL_DAMPING = auto()

    AIR_PRESSURE = auto()
    AIR_TEMPERATURE = auto()

    HOMODYNE_ANGLE = auto()
    EVALUATION_ORDER = auto()

    def __repr__(self):
        """
        Return an appropriate string when printed
        """

        if self == ParameterType.LIGHT_MECH_COUPLING:
            return 'Light-Mech Coupling'

        # tweezer
        if self == ParameterType.PHASE_OF_2ND_TWEEZER:
            return 'Phase of 2nd Tweezer'

        # cavity
        if self == ParameterType.CAVITY_DETUNING:
            return 'Cavity Detuning'
        if self == ParameterType.OPTICAL_DAMPING:
            return 'Optical Damping'
        # gas
        if self == ParameterType.AIR_PRESSURE:
            return 'Air Pressure'
        if self == ParameterType.AIR_TEMPERATURE:
            return 'Air Temperature'

        # other
        if self == ParameterType.HOMODYNE_ANGLE:
            return 'Homodyne Angle'
        if self == ParameterType.EVALUATION_ORDER:
            return 'Evaluation Order'


class Parameter:
    def __init__(
        self,
        _type,
        symbol,

        units,
        unit_symbol,
        unit_string,

        default_value,
        axis_range,
        **kwargs,
    ):
        self.type = _type
        self.symbol = symbol

        self.units = units
        # this is added for latex
        self.unit_symbol = unit_symbol
        # this is added for written
        self.unit_string = unit_string

        self.default_value = default_value
        self.axis_range = axis_range
        self.range = axis_range * units

        self.format_specifier = kwargs.get('format_specifier', '.0f')

light_mech_coupling = Parameter(
    _type=ParameterType.LIGHT_MECH_COUPLING,
    symbol='$g$/2π',
    units=kHz,
    unit_symbol=' kHz',
    unit_string='(kHz)',
    default_value=-40*kHz,
    # ignore g = 0 kHz case
    axis_range=np.linspace(-1, -50, 50),
)
phase_of_2nd_tweezer = Parameter(
    _type=ParameterType.PHASE_OF_2ND_TWEEZER,
    symbol='$\phi_2$',
    units=degrees,
    unit_symbol='°',
    unit_string='(degrees)',
    default_value=0*degrees,
    axis_range=np.linspace(0, 360, 12 + 1),
)
cavity_detuning = Parameter(
    _type=ParameterType.CAVITY_DETUNING,
    symbol='$\Delta$/2π',
    units=kHz,
    unit_symbol=' kHz',
    unit_string='(kHz)',
    default_value=-450*kHz,
    axis_range=np.linspace(-100, -800, 8),
)
optical_damping = Parameter(
    _type=ParameterType.OPTICAL_DAMPING,
    symbol='$\kappa$/2π',
    units=kHz,
    unit_symbol=' kHz',
    unit_string='(kHz)',
    default_value=400*kHz,
    axis_range=np.linspace(0, 500, 10 + 1),
)
air_pressure = Parameter(
    _type=ParameterType.AIR_PRESSURE,
    symbol=r'$P_{air}$',
    units=mbar,
    unit_symbol=' mbar',
    unit_string='(mbar)',
    default_value=1e-6*mbar,
    axis_range=np.logspace(1, -12, 13 + 1),
    format_specifier='.0e',
)
air_temperature = Parameter(
    _type=ParameterType.AIR_TEMPERATURE,
    symbol=r'$T_{air}$',
    units=K,
    unit_symbol=' K',
    unit_string='(K)',
    default_value=300*K,
    axis_range=np.logspace(3e-1, 3e2, 3 + 1),
    format_specifier='.1f',
)
homodyne_angle = Parameter(
    _type=ParameterType.HOMODYNE_ANGLE,
    symbol='θ$_{hom}$',
    units=degrees,
    unit_symbol='°',
    unit_string='(degrees)',
    default_value=π/4,
    axis_range=np.linspace(0, 360, 8 + 1),
)
evaluation_order = Parameter(
    _type=ParameterType.EVALUATION_ORDER,
    symbol='E.O.',
    units=1,
    unit_symbol='',
    unit_string='',
    default_value=4,
    axis_range=np.array([1, 2, 3, 4]),
)
all_parameters = {
    ParameterType.LIGHT_MECH_COUPLING: light_mech_coupling,
    ParameterType.PHASE_OF_2ND_TWEEZER: phase_of_2nd_tweezer,
    ParameterType.CAVITY_DETUNING: cavity_detuning,
    ParameterType.OPTICAL_DAMPING: optical_damping,
    ParameterType.AIR_PRESSURE: air_pressure,
    ParameterType.AIR_TEMPERATURE: air_temperature,
    ParameterType.HOMODYNE_ANGLE: homodyne_angle,
    ParameterType.EVALUATION_ORDER: evaluation_order,
}
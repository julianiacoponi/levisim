import numpy as np
from copy import copy

from levisim.core.constants import π
from levisim.core.units import (
    degrees,
    metre, cm,
    second,
    kHz,
    mbar,
    K,
)
from levisim.multiphysics.multitrap import MultiTrap


__all__ = [
    # latex
    'latex_ω_units',

    # default
    'DEFAULT_ω_range',
    'DEFAULT_cavity_kwargs',
    'DEFAULT_gas_kwargs',
    'DEFAULT_Ω_model_pairs',

    # to match the fortran code
    'FORTRAN_ω_range',
    'FORTRAN_Ω_model_pairs',
    'FORTRAN_toy_params',
    'FORTRAN_8x2',

    # for off-node tests (including z-axis and mech-mech-couplings)
    'OFF_NODE_toy_params_pairs',
    'OFF_NODE_toy_params_equal_spacing_simple',

    # with phase dependence
    'PHASE_EQUAL_toy_params',
    'PHASE_DEPENDENT_toy_params',
    'PHASE_EQUAL_xy_dict',
    'PHASE_DEPENDENT_xy_dict',
]

latex_ω_units = {
    +2: r'[($2\pi\cdot$Hz)$^{\mathregular{2}}$]',
    +1: r'[$2\pi\cdot$Hz]',
    -1: r'[($2\pi\cdot$Hz)$^{\mathregular{-1}}$]',
    -2: r'[($2\pi\cdot$Hz)$^{\mathregular{-2}}$]',
}


####################################################################################################
# Default setup kwargs
# --------------------------------------------------------------------------------------------------
DEFAULT_gas_kwargs = dict(
    P=1e-6*mbar,
    T=300*K,
    vth=500*metre/second,
)
DEFAULT_cavity_kwargs = dict(
    Δ=-450*kHz,
    L=0.96*cm,
    Finesse=40_000,
    θhom=π/4,
)


# mimics fortran code
# TODO: make a class for Ω_model
FORTRAN_Ω_model_pairs = dict(
    Ω_model='pairs',
    Ω_first=312*kHz,
    Ω_intrapair_gap=1.5*kHz,
    Ω_interpair_gap=15*kHz,
)
FORTRAN_toy_params = dict(
    **FORTRAN_Ω_model_pairs,
    toy_gx=-40*kHz,
    toy_gy=-40*kHz,
)

FORTRAN_Ω_model_equal_spacing_simple = dict(
    Ω_model='equal spacing: simple',
    Ω_start=312*kHz,
    Ω_gap=7*kHz,
)


# for mimicking trapping away from the node
DEFAULT_Ω_model_pairs = dict(
    Ω_model='pairs',
    Ω_first=305*kHz,
    Ω_intrapair_gap=2.5*kHz,
    Ω_interpair_gap=7*kHz,
)
DEFAULT_Ω_model_equal_spacing = dict(
    Ω_model='equal spacing',
    Ω_mid=275*kHz,
    Ω_span=50*kHz,
)
DEFAULT_Ω_model_equal_spacing_simple = dict(
    Ω_model='equal spacing: simple',
    Ω_start=305*kHz,
    Ω_gap=7*kHz,
)


OFF_NODE_coupling_and_phase_params = dict(
    toy_gx=-40*kHz,
    toy_gy=-40*kHz,
    # non-zero z and mech-mech couplings
    toy_gz=+170*kHz,

    toy_gxy=-2*kHz,
    toy_gxz=-15*kHz,
    toy_gyz=-15*kHz,

    # on-off 45 degree phase difference
    φ_odd_n=0*degrees,
    φ_even_n=45*degrees,
)

OFF_NODE_toy_params_pairs = dict(
    **DEFAULT_Ω_model_pairs,
    **OFF_NODE_coupling_and_phase_params,
    # toy_gx=-40*kHz,
    # toy_gy=-40*kHz,
    # # non-zero z and mech-mech couplings
    # toy_gz=+170*kHz,

    # toy_gxy=-2*kHz,
    # toy_gxz=-15*kHz,
    # toy_gyz=-15*kHz,

    # # on-off 45 degree phase difference
    # φ_odd_n=0*degrees,
    # φ_even_n=45*degrees,
)
OFF_NODE_toy_params_equal_spacing = dict(
    **DEFAULT_Ω_model_equal_spacing,
    toy_gx=-40*kHz,
    toy_gy=-40*kHz,
    # non-zero z and mech-mech couplings
    toy_gz=+170*kHz,

    toy_gxy=-2*kHz,
    toy_gxz=-15*kHz,
    toy_gyz=-15*kHz,

    # on-off 45 degree phase difference
    φ_odd_n=0*degrees,
    φ_even_n=45*degrees,
)
OFF_NODE_toy_params_equal_spacing_simple = dict(
    **DEFAULT_Ω_model_equal_spacing_simple,
    toy_gx=-40*kHz,
    toy_gy=-40*kHz,
    # non-zero z and mech-mech couplings
    toy_gz=+170*kHz,

    toy_gxy=-2*kHz,
    toy_gxz=-15*kHz,
    toy_gyz=-15*kHz,

    # on-off 45 degree phase difference
    φ_odd_n=0*degrees,
    φ_even_n=45*degrees,
)

PHASE_EQUAL_toy_params = dict(
    **DEFAULT_Ω_model_pairs,
    gjn_dict=dict(
        x1=-40*kHz,
        y1=-39*kHz,

        x2=-38*kHz,
        y2=-37*kHz,

        x3=-36*kHz,
        y3=-35*kHz,

        x4=-34*kHz,
        y4=-33*kHz,
    ),
    φn_dict={
        1: 0*degrees,
        2: 0*degrees,
        3: 0*degrees,
        4: 0*degrees,
    },
)
PHASE_DEPENDENT_toy_params = copy(PHASE_EQUAL_toy_params)
PHASE_DEPENDENT_toy_params['φn_dict'] = {
    1: 0*degrees,
    2: 30*degrees,
    3: 60*degrees,
    4: 90*degrees,
}

####################################################################################################
# `MultiTrap` instances
# --------------------------------------------------------------------------------------------------
FORTRAN_ω_range = np.arange(0, 400, 0.5) * kHz
FORTRAN_8x2 = MultiTrap(
    num_traps=8,
    Cartesian_axes_per_trap=2,
    ω_range=FORTRAN_ω_range,
    gas_kwargs=DEFAULT_gas_kwargs,
    cavity_kwargs=DEFAULT_cavity_kwargs,
    toy_params=FORTRAN_toy_params,
)

DEFAULT_ω_range = np.arange(250, 320 + 0.05, 0.05) * kHz
PHASE_EQUAL_xy_dict = {}
PHASE_DEPENDENT_xy_dict = {}
for system in [
    '1×2',
    '2×2',
    '3×2',
    '4×2',
]:
    Nd = system.split('×')
    N, d = int(Nd[0]), int(Nd[1])
    PHASE_EQUAL_xy_dict[system] = {}
    PHASE_DEPENDENT_xy_dict[system] = {}
    for sign in [+1, -1]:
        PHASE_EQUAL_xy_dict[system][sign] = MultiTrap(
            num_traps=N,
            Cartesian_axes_per_trap=d,
            ω_range=DEFAULT_ω_range,
            gas_kwargs=DEFAULT_gas_kwargs,
            cavity_kwargs=DEFAULT_cavity_kwargs,
            toy_params=PHASE_EQUAL_toy_params,
            tweezer_phase_sign_convention=sign,
            load_in_power_series=N * d <= 6,
        )
        PHASE_DEPENDENT_xy_dict[system][sign] = MultiTrap(
            num_traps=N,
            Cartesian_axes_per_trap=d,
            ω_range=DEFAULT_ω_range,
            gas_kwargs=DEFAULT_gas_kwargs,
            cavity_kwargs=DEFAULT_cavity_kwargs,
            toy_params=PHASE_DEPENDENT_toy_params,
            tweezer_phase_sign_convention=sign,
            load_in_power_series=N * d <= 6,
        )

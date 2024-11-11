from scipy.constants import (
    pi as π,
    hbar as ħ,  # Planck's reduced constant [J.s]
    speed_of_light as c,   # [m/s]
    Boltzmann as kB,  # [J/K]
    elementary_charge as e,  # [C]
    atomic_mass as u,  # [kg]
    epsilon_0 as ε0,  # vacuum permittivity [F/m]
    g,  # Earth's gravitational acceleration [m/s²]
    zero_Celsius,  # [K]
)

absolute_zero = -zero_Celsius  # 0 K in °C

# physical constants of air as a gas
d_gas = 0.372e-9  # diameter of air molecule [m]
ξ_gas = 18.27e-6  # viscosity of air [kg/m/s] = [Pa.s]
m_gas = 4.8e-26  # mass of air molecule [kg]
T_gas = 21.5 - absolute_zero  # gas temperature (actual lab temperature) [K]

# number of photons contributing to noise / the dynamic part of a photon field
shot_noise = 0
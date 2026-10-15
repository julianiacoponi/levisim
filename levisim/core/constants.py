from scipy.constants import (
    pi as π,
    hbar as ħ,  # Planck's reduced constant [J·s]
    speed_of_light as c,   # [m/s]
    Boltzmann as kB,  # [J/K]
    elementary_charge as e,  # [C]
    atomic_mass as u,  # [kg]
    epsilon_0 as ε0,  # vacuum permittivity [F/m]
    mu_0 as μ0,  # vacuum permeability [H/m]
    g,  # Earth's gravitational acceleration [m/s²]
    zero_Celsius,  # [K]
    Avogadro,
)

absolute_zero = -zero_Celsius  # 0 K in °C
T_lab = 21.5 - absolute_zero  # gas temperature (actual lab temperature) [K]

# physical constants of air as a gas
d_air = 0.372e-9  # diameter of air molecule [m]
ξ_air = 18.27e-6  # viscosity of air [kg/m/s] = [Pa·s]
m_air = 4.8e-26  # mass of air molecule [kg]

# number of photons contributing to noise / the dynamic part of a photon field
shot_noise = 0

# positions of beads in a cavity standing wave
# as a phase Φ, where Φ/2π = x0/λ
antinode, halfnode, node = 0, π/4, π/2
# Φ = k·x0
# k = 2π/λ
# Φ = (antinode, halfnode, node)
# x0 = (0, 0.125, 0.25)λ


# ------------------------------------------------------------------
# Public names exported by this module.
# Physical constants, re-exported from scipy under the symbols used in the physics
# literature. NOTE: every scipy import at the top of this file is deliberately
# re-exported - a linter reports them as unused, and removing them breaks every
# module in the package.
# ------------------------------------------------------------------
__all__ = [
    'Avogadro',
    'T_lab',
    'absolute_zero',
    'antinode',
    'c',
    'd_air',
    'e',
    'g',
    'halfnode',
    'kB',
    'm_air',
    'node',
    'shot_noise',
    'u',
    'zero_Celsius',
    'ħ',
    'ε0',
    'μ0',
    'ξ_air',
    'π',
]

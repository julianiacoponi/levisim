"""
Decoherence budget of levitated particles: heating rates per mode, in phonons per second

The rates add to the noise of the linear model, as the photon recoil already does in MultiTrap (an
extra heating rate Γ_h is the same as raising the bath occupation n → n + Γ_h/Γ; see
`levisim.multiphysics.covariance.quadrature_matrices(..., extra_heating=...)`). They are:

- gas: the Brownian heating of levisim's gas bath, Γ_gas·n_th with n_th the Bose-Einstein occupation
  at the gas temperature (`Dimension.Γ`, `Dimension.noise_phonons_exact`)
- photon_recoil: the recoil of the tweezer photons the particle scatters
  (`Trap.photon_recoil_heating_rates`, rescaled to the mode's frequency as in MultiTrap)
- blackbody_scattering, blackbody_emission, blackbody_absorption: thermal photons, from the
  localisation parameters of Romero-Isart, Phys. Rev. A 84, 052121 (2011), Eqs. (28)-(29) ::

      Λ_sc     = 8!·8ζ(9)·c·R⁶/(9π) · (kB·T_e/ħc)⁹ · Re[(ε_bb - 1)/(ε_bb + 2)]²
      Λ_e(a)   = 16π⁵·c·R³/189 · (kB·T_i(e)/ħc)⁶ · Im[(ε_bb - 1)/(ε_bb + 2)]

  with T_e the environment and T_i the particle's internal temperature, and ε_bb the permittivity
  averaged over the thermal spectrum (default 2.1 + 0.57i, silica, Table III of that paper). A
  localisation term dρ/dt = -Λ[x, [x, ρ]] heats a mode at 2Λ·x_zpf² phonons per second
  (d〈p²〉/dt = 2ħ²Λ, so dE/dt = ħ²Λ/m).
- electric_field_noise: a charged particle in fluctuating electric fields (electrode voltage noise,
  patch potentials), Γ = q²·S_E(Ω)/(4mħΩ) with S_E the single-sided electric-field noise spectral
  density at the mode frequency [V²/m²/Hz] (Brownnutt et al., Rev. Mod. Phys. 87, 1419 (2015),
  Eq. (10) without RF micromotion). Electrode voltage noise S_V on electrodes a distance d away
  gives S_E ≈ S_V/d² (times a geometry factor of order one).

These rates are valid in the long-wavelength (localisation) regime and for Q = Ω/Γ ≫ 1, which is
where levisim's linear Gaussian model applies. The blackbody rates are those of a point dipole,
valid for spheres much smaller than the thermal wavelength, R·kB·T/(ħc) ≪ 1 (ħc/kB·T = 0.57 mm at
4 K, 7.6 μm at 300 K); above 0.1 `decoherence_budget` logs a WARNING and lists the modes in
`DecoherenceBudget.warnings`, since the R⁶ scattering term then grossly overestimates the rate.
"""

import math

import numpy as np
from scipy.special import zeta

from levisim.core.constants import c, kB, ħ, π
from levisim.core.logger import logger

__all__ = [
    'DEFAULT_BLACKBODY_PERMITTIVITY',
    'DecoherenceBudget',
    'blackbody_localisation_parameters',
    'decoherence_budget',
    'electric_field_noise_heating_rate',
    'localisation_heating_rate',
]

# permittivity of silica averaged over the thermal spectrum (Romero-Isart 2011, Table III)
DEFAULT_BLACKBODY_PERMITTIVITY = 2.1 + 0.57j

MECHANISMS = (
    'gas', 'photon_recoil', 'blackbody_scattering', 'blackbody_emission', 'blackbody_absorption',
    'electric_field_noise',
)


def blackbody_localisation_parameters(R, T_environment, T_internal=None, ε_bb=DEFAULT_BLACKBODY_PERMITTIVITY):
    """
    Localisation parameters (Λ_sc, Λ_e, Λ_a) of a dielectric sphere of radius R by blackbody
    radiation: scattering of and absorption from the environment at T_environment, and emission at
    the particle's internal temperature T_internal (default T_environment); Romero-Isart, Phys. Rev.
    A 84, 052121 (2011), Eqs. (28)-(29)
    [1/(m²·s) each]
    """
    T_internal = T_environment if T_internal is None else T_internal
    response = (ε_bb - 1) / (ε_bb + 2)
    Λ_sc = (math.factorial(8) * 8 * zeta(9) * c * R**6 / (9 * π)
            * (kB * T_environment / (ħ * c))**9 * np.real(response)**2)
    prefactor = 16 * π**5 * c * R**3 / 189 * np.imag(response)
    Λ_e = prefactor * (kB * T_internal / (ħ * c))**6
    Λ_a = prefactor * (kB * T_environment / (ħ * c))**6
    return float(Λ_sc), float(Λ_e), float(Λ_a)


def localisation_heating_rate(Λ, m, Ω):
    """
    Heating rate of a mode of mass m and frequency Ω under dρ/dt = -Λ[x, [x, ρ]]: 2Λ·x_zpf² = ħΛ/(mΩ)
    [phonons/s]
    """
    return ħ * Λ / (m * Ω)


def electric_field_noise_heating_rate(q, S_E, m, Ω):
    """
    Heating rate q²·S_E/(4mħΩ) of a charge q [C] in an electric field of single-sided noise spectral
    density S_E [V²/m²/Hz] at the mode frequency (Brownnutt et al., RMP 87, 1419 (2015), Eq. (10))
    [phonons/s]
    """
    return q**2 * S_E / (4 * m * ħ * Ω)


class DecoherenceBudget(dict):
    """
    {label: {mechanism: rate [phonons/s], ..., 'total': ...}} with helpers

    - `table()`: a plain-text table, one row per mode
    - `totals()`: {label: total rate}
    - `extra_heating(exclude=None)`: {label: rate} summed over the mechanisms that the MultiTrap's
      noise does not already contain, for `covariance.quadrature_matrices(..., extra_heating=...)`
    - `to_dataframe()`: a pandas DataFrame (needs pandas)
    - `included`: the mechanisms already in the MultiTrap's noise ('gas', and 'photon_recoil' with
      MultiTrap(photon_recoil=True))
    """

    included = ('gas',)
    warnings = ()

    def totals(self):
        """{label: total heating rate} [phonons/s]"""
        return {label: rates['total'] for label, rates in self.items()}

    def extra_heating(self, exclude=None):
        """{label: summed rate of the mechanisms not in `exclude` (default `included`)} [phonons/s]"""
        exclude = self.included if exclude is None else exclude
        return {
            label: sum(rate for name, rate in rates.items() if name in MECHANISMS and name not in exclude)
            for label, rates in self.items()
        }

    def table(self):
        """The budget as plain text, one row per mode [phonons/s]"""
        names = [*MECHANISMS, 'total']
        lines = [f'{"mode":>6s} ' + ' '.join(f'{name:>21s}' for name in names)]
        for label, rates in self.items():
            lines.append(f'{label:>6s} ' + ' '.join(f'{rates[name]:21.4g}' for name in names))
        return '\n'.join(lines)

    def to_dataframe(self):
        """The budget as a pandas DataFrame indexed by mode [phonons/s]"""
        import pandas as pd
        return pd.DataFrame.from_dict(self, orient='index')

    def __str__(self):
        return self.table()


def decoherence_budget(mtrap, T_environment=None, T_internal=None, ε_bb=DEFAULT_BLACKBODY_PERMITTIVITY,
                       electric_field_noise=0.0):
    """
    Heating rates of every mode of a MultiTrap (physical parameters), by mechanism

    - T_environment [the gas temperature]: temperature of the thermal radiation [K]
    - T_internal [T_environment]: internal (bulk) temperature of the particles [K], a number or a
      list with one per trap; laser absorption usually makes it larger than T_environment
    - ε_bb [2.1 + 0.57i]: permittivity averaged over the thermal spectrum
    - electric_field_noise [0]: single-sided electric-field noise S_E [V²/m²/Hz] at each particle, a
      number or {trap index: S_E}; it heats charged particles only (`Particle.charge`)

    Each rate uses the mode's frequency `Dimension.Ω` (with any static shifts) and its x_zpf. The
    photon recoil is reported whether or not the MultiTrap includes it in its noise
    (`DecoherenceBudget.included` lists those that are). Mechanisms, formulas and sources are in
    the module docstring.
    [DecoherenceBudget of phonons/s]
    """
    if mtrap.toy_params:
        raise ValueError('decoherence_budget needs physical parameters (particles, traps and gas), not toy_params')
    T_environment = mtrap.gas.T if T_environment is None else T_environment
    budget = DecoherenceBudget()
    budget.included = ('gas', 'photon_recoil') if mtrap.photon_recoil else ('gas',)
    warnings = []
    for label, dim in mtrap.dimensions.items():
        trap = dim.trap
        particle = trap.bead
        n = dim.trap_index
        T_i = T_internal[n - 1] if np.ndim(T_internal) == 1 else T_internal
        R = getattr(particle, 'r', None)
        rates = dict.fromkeys(MECHANISMS, 0.0)
        rates['gas'] = dim.Γ * dim.noise_phonons_exact
        try:
            rates['photon_recoil'] = trap.photon_recoil_heating_rates[dim.axis] * trap.Ωj[dim.axis] / dim.Ω
        except ValueError:  # a trap without an optical element scatters no tweezer light
            rates['photon_recoil'] = 0.0
        if R is not None:
            size_parameter = R * kB * max(T_environment, T_environment if T_i is None else T_i) / (ħ * c)
            if size_parameter > 0.1:
                warnings.append(
                    f'{label}: R·kB·T/(ħc) = {size_parameter:.3g} > 0.1, outside the dipole limit of the blackbody rates'
                )
            Λ_sc, Λ_e, Λ_a = blackbody_localisation_parameters(R, T_environment, T_i, ε_bb)
            rates['blackbody_scattering'] = localisation_heating_rate(Λ_sc, dim.m, dim.Ω)
            rates['blackbody_emission'] = localisation_heating_rate(Λ_e, dim.m, dim.Ω)
            rates['blackbody_absorption'] = localisation_heating_rate(Λ_a, dim.m, dim.Ω)
        S_E = electric_field_noise.get(n, 0.0) if isinstance(electric_field_noise, dict) else electric_field_noise
        if S_E and particle.q:
            rates['electric_field_noise'] = electric_field_noise_heating_rate(particle.q, S_E, dim.m, dim.Ω)
        rates['total'] = sum(rates[name] for name in MECHANISMS)
        budget[label] = rates
    budget.warnings = tuple(warnings)
    if warnings:
        logger.warning(
            'decoherence_budget: the blackbody rates are point-dipole results, and these modes are outside'
            ' their validity (they overestimate the rate): {}', '; '.join(warnings),
        )
    return budget

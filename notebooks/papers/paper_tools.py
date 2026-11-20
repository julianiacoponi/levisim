"""
Helpers shared by the paper-reproduction notebooks in this folder

Everything here is a thin layer over levisim's public API. The published systems themselves are
built with levisim directly: `MultiTrap(tweezers=[...], beads=[...], trap_kwargs=[...])` takes the
paper's `Tweezer` and `Bead` instances (and per-trap `Trap` keyword arguments such as measured
frequencies through `override_Ω_dict`), drives the cavity at the first tweezer's wavelength, and
adds photon-recoil heating (`Trap.photon_recoil_heating_rates`) to the gas bath of every mode
(`MultiTrap(photon_recoil=True)`, the default with physical parameters). What remains here:

- `toy_multitrap` and `add_heating`: a toy-parameter `MultiTrap` with given frequencies, couplings
  and *total* heating rates (a paper's fitted rates). Photon recoil needs physical
  parameters, so for toy systems the extra heating is still imposed here, by raising each mode's
  bath occupation, n → n + Γ_heat/Γ, which is exact for a heating term that adds no damping (the
  same way levisim includes photon recoil).
- `heterodyne`: the same output noise vector as `MultiTrap.heterodyne_spectrum` with the Linear
  Amplifier Model, evaluated for many frequencies at once (the levisim method loops in Python).
  Each notebook checks it against `MultiTrap.heterodyne_spectrum` before using it.
- `occupations`, `mechanical_eigenfrequencies`, `gas_heating_rate`, `load_digitised`: small
  read-outs of a built `MultiTrap`, and the loader for the digitised figure data in `data/`.

Conventions (see docs/concepts/conventions.md): angular frequencies (`kHz` = 2π × 10³ rad/s);
Δ = ω_tweezer − ω_cavity, so red detuning is Δ < 0 (most papers use the opposite sign); the
tweezer polarisation is along the tweezer's y axis and `Tweezer.θ` is its angle to the cavity
axis (x), so θ = 90° is polarisation perpendicular to the cavity axis.
"""

import json
from pathlib import Path

import numpy as np

from levisim.multiphysics.multitrap import MultiTrap
from levisim.multiphysics.covariance import steady_state_covariance, mode_index
from levisim.physics.tweezer import Tweezer


DATA_DIR = Path(__file__).resolve().parent / 'data'


def toy_multitrap(Ω, g, heating, Δ, κ, m, Γ, T=300.0, λ=1550e-9, num_traps=1, axes=2, ω_range=None, **toy_extra):
    """
    A toy-parameter `MultiTrap` with given frequencies, couplings and total heating rates

    Ω, g, heating: dicts {label: value} [2π·Hz]; heating is the total phonon heating rate of each
    mode (gas + recoil + ...), imposed with `add_heating` on top of the gas bath (damping Γ at
    temperature T). Δ uses levisim's sign (Δ < 0 is red detuning); couplings of either sign are
    accepted. A toy system has no tweezer, so the cavity is given the tweezer frequency of
    wavelength λ through `cavity_kwargs['ωtw']` (physical systems take it from their first tweezer).
    """
    toy_params = dict(Ω_model='custom', toy_Ωjn_dict=dict(Ω), gjn_dict=dict(g),
                      use_Γ_formula=False, toy_Γ=Γ, T=T, m=m, **toy_extra)
    kwargs = dict(num_traps=num_traps, Cartesian_axes_per_trap=axes, toy_params=toy_params,
                  cavity_kwargs=dict(κ=κ, Δ=Δ, ωtw=Tweezer(λ=λ).ω))
    if ω_range is not None:
        kwargs['ω_range'] = ω_range
    mtrap = MultiTrap(**kwargs)
    add_heating(mtrap, {label: heating[label] - gas_heating_rate(mtrap, label) for label in mtrap.labels})
    return mtrap


def gas_heating_rate(mtrap, label):
    """
    Heating rate of one mode from its bath alone, Γ·n_th [2π·Hz, phonons per second]
    """
    dimension = mtrap.dimensions[label]
    return dimension.Γ * dimension.noise_phonons_exact


def add_heating(mtrap, heating):
    """
    Add a pure heating rate {label: Γ_heat} [2π·Hz] to modes of a built toy-parameter `MultiTrap`

    levisim's bath for mode j has damping Γ_j and occupation n_j, i.e. d〈n〉/dt = −Γ_j (〈n〉 − n_j).
    A heating process without damping adds Γ_heat to the right-hand side, which is the same as
    raising n_j by Γ_heat/Γ_j. Both noise factors of the pair (n + 1, n) are raised, and the noise
    matrix is rebuilt, so spectra and `steady_state_covariance` see the new bath.

    Physical systems get photon-recoil heating from levisim itself (`MultiTrap(photon_recoil=True)`, the default),
    which raises the bath occupation in the same way. This helper is for toy systems, where a
    paper's fitted total heating rate is imposed (see `toy_multitrap`).
    """
    factors = list(mtrap.quantum_factor_per_noise_operator)
    for label, Γ_heat in heating.items():
        index = mtrap.label_to_noise_index[label]
        extra = Γ_heat / mtrap.dimensions[label].Γ
        factors[index] += extra
        factors[index + 1] += extra
    mtrap.quantum_factor_per_noise_operator = factors
    mtrap.noise_matrix = np.diag(mtrap.γ * np.asarray(factors))


def occupations(mtrap):
    """
    Steady-state phonon occupations {label: 〈b†b〉} from levisim's Lyapunov covariance
    """
    V = steady_state_covariance(mtrap)
    out = {}
    for label in mtrap.labels:
        k = mode_index(mtrap, label)
        out[label] = float((V[2*k, 2*k] + V[2*k + 1, 2*k + 1] - 2) / 4)
    return out


def heterodyne(mtrap, ω):
    """
    levisim's heterodyne spectrum S_het(ω) = 〈a_out(ω) a_out†(ω)〉 (shot noise = 1), vectorised over ω

    Same noise vector as `MultiTrap.heterodyne_spectrum(LINEAR_AMPLIFIER_MODEL)`:
    a_out = −√κ · [T(ω) c_in]_0 + a_in with T(ω) = (−iω − A)⁻¹. S_het − 1 is the normally ordered
    spectrum, so the ω > 0 side is anti-Stokes (∝ n) and the ω < 0 side Stokes (∝ n + 1).
    """
    ω = np.atleast_1d(np.asarray(ω, dtype=float))
    n = mtrap.num_operators
    M = -1j * ω[:, None, None] * np.eye(n)[None] - mtrap.drift_matrix[None]
    e0 = np.zeros((len(ω), n, 1))
    e0[:, 0, 0] = 1
    T_row0 = np.linalg.solve(np.transpose(M, (0, 2, 1)), e0)[..., 0]
    v = -np.sqrt(mtrap.cavity.κ) * T_row0 * np.sqrt(mtrap.γ)[None]
    v[:, 0] += 1
    return np.sum(np.asarray(mtrap.quantum_factor_per_noise_operator)[None] * np.abs(v)**2, axis=1)


def mechanical_eigenfrequencies(mtrap):
    """
    Normal-mode frequencies and linewidths of the mechanical-like eigenvalues of the drift matrix

    Returns (frequencies, linewidths) [2π·Hz], sorted by frequency, one entry per mechanical mode:
    the eigenvalues λ of A with c ∝ exp(λt) come in pairs ±iω − γ/2; the cavity-like pair (largest
    damping) is dropped, and of each mechanical pair the one with Im λ < 0 (an annihilation
    operator) is kept.
    """
    λ = np.linalg.eigvals(mtrap.drift_matrix)
    λ = λ[np.argsort(np.abs(λ.real))][:-2]
    λ = λ[λ.imag < 0]
    order = np.argsort(-λ.imag)
    return -λ.imag[order], -2 * λ.real[order]


def load_digitised(name):
    """
    Digitised figure data (JSON in ./data, with its provenance in the 'method' entry)
    """
    with open(DATA_DIR / name, encoding='utf-8') as f:
        return json.load(f)

"""
Helpers shared by the paper-reproduction notebooks in this folder

Everything here is a thin layer over levisim's public API. Each helper exists because levisim does
not (yet) offer the feature directly, and its docstring says which levisim limitation it works
around, so that the helper can be retired when levisim grows the feature:

- `PhysicalMultiTrap`: `MultiTrap.setup_levitated_physics` hard-codes its tweezers and beads
  (1064 nm, 60 nm beads, all at the node), so a published system cannot be built from its stated
  power, waist, wavelength and particle size. This subclass takes the `Tweezer` and `Bead`
  instances (and per-trap `Trap` keyword arguments, e.g. measured frequencies) as arguments.
- `add_heating`: levisim's quantum linear theory has only the gas bath (damping `Γ` = `Trap.Γth`,
  occupation from the gas temperature). Photon-recoil heating, which dominates at low pressure,
  is added here as an extra pure heating rate by raising each mode's bath occupation,
  n → n + Γ_heat/Γ, which is exact for a heating term that adds no damping.
- `heterodyne`: the same output noise vector as `MultiTrap.heterodyne_spectrum` with the Linear
  Amplifier Model, evaluated for many frequencies at once (the levisim method loops in Python).
  Each notebook checks it against `MultiTrap.heterodyne_spectrum` before using it.

Conventions (see docs/concepts/conventions.md): angular frequencies (`kHz` = 2π × 10³ rad/s);
Δ = ω_tweezer − ω_cavity, so red detuning is Δ < 0 (most papers use the opposite sign); the
tweezer polarisation is along the tweezer's y axis and `Tweezer.θ` is its angle to the cavity
axis (x), so θ = 90° is polarisation perpendicular to the cavity axis.
"""

import contextlib
import io
import json
from pathlib import Path

import numpy as np

from levisim.core.constants import π, ħ, ε0
from levisim.multiphysics.multitrap import MultiTrap
from levisim.multiphysics.covariance import steady_state_covariance, mode_index
from levisim.physics.trap import Trap
from levisim.physics.tweezer import Tweezer


DATA_DIR = Path(__file__).resolve().parent / 'data'


class PhysicalMultiTrap(MultiTrap):
    """
    A `MultiTrap` built from given `Tweezer` and `Bead` instances instead of the hard-coded ones

    `trap_kwargs` is a list with one dict per trap of extra `Trap` keyword arguments, e.g.
    `override_Ω_dict` to use measured mechanical frequencies. All other keyword arguments go to
    `MultiTrap` (`Cartesian_axes_per_trap`, `gas_kwargs`, `cavity_kwargs`, `ω_range`, ...).
    Pass the tweezer's angular frequency as `cavity_kwargs['ωtw']` when λ ≠ 1064 nm.
    """
    def __init__(self, tweezers, beads, trap_kwargs=None, **kwargs):
        self._paper_tweezers = list(tweezers)
        self._paper_beads = list(beads)
        self._paper_trap_kwargs = list(trap_kwargs) if trap_kwargs else [{} for _ in self._paper_tweezers]
        assert len(self._paper_tweezers) == len(self._paper_beads) == len(self._paper_trap_kwargs)
        kwargs.setdefault('num_traps', len(self._paper_tweezers))
        super().__init__(**kwargs)

    def setup_levitated_physics(self, **kwargs):
        self.tweezers = self._paper_tweezers
        self.beads = self._paper_beads
        self.φn_dict = {ii + 1: tweezer.φ for ii, tweezer in enumerate(self.tweezers)}
        self.traps = [
            Trap(tweezer=tweezer, bead=bead, gas=self.gas, trap_index=ii + 1, **extra)
            for ii, (tweezer, bead, extra) in enumerate(zip(self.tweezers, self.beads, self._paper_trap_kwargs))
        ]


def toy_multitrap(Ω, g, heating, Δ, κ, m, Γ, T=300.0, λ=1550e-9, num_traps=1, axes=2, ω_range=None, **toy_extra):
    """
    A toy-parameter `MultiTrap` with given frequencies, couplings and total heating rates

    Ω, g, heating: dicts {label: value} [2π·Hz]; heating is the total phonon heating rate of each
    mode (gas + recoil + ...), imposed with `add_heating` on top of the gas bath (damping Γ at
    temperature T). Δ uses levisim's sign (Δ < 0 is red detuning).
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


def recoil_heating_rates(trap):
    """
    Photon-recoil heating rates {axis: Γ_rec} [2π·Hz, phonons per second] of a bead in a tweezer

    Γ_rec,j = π/(15 ħ ε0) · (α E0 / 2π)² · k⁵ · c_j · x_zpf,j², with c_j = 1 along the polarisation,
    2 perpendicular to it in the focal plane and 7 along the beam (5 of them from the absorbed
    photon's momentum), as in Eq. (4) of Piotrowski et al., Nat. Phys. 19, 1009 (2023) and Table II of
    Gonzalez-Ballestero et al., Phys. Rev. A 100, 013805 (2019). In levisim's tweezer frame the
    polarisation is along y, so (c_x, c_y, c_z) = (2, 1, 7). The zero-point fluctuations use the
    trap's frequencies (`override_Ω_dict` if given).
    """
    tweezer = trap.tweezer
    prefactor = π / (15 * ħ * ε0) * (np.real(trap.α_eff) * tweezer.E0 / (2 * π))**2 * tweezer.k**5
    coefficient = {'x': 2, 'y': 1, 'z': 7}
    return {axis: prefactor * coefficient[axis] * ħ / (2 * trap.bead.m * trap.Ωj[axis]) for axis in 'xyz'}


def gas_heating_rate(mtrap, label):
    """
    Heating rate of one mode from its bath alone, Γ·n_th [2π·Hz, phonons per second]
    """
    dimension = mtrap.dimensions[label]
    return dimension.Γ * dimension.noise_phonons_exact


def add_heating(mtrap, heating):
    """
    Add a pure heating rate {label: Γ_heat} [2π·Hz] to modes of a built `MultiTrap`

    levisim's bath for mode j has damping Γ_j and occupation n_j, i.e. d〈n〉/dt = −Γ_j (〈n〉 − n_j).
    A heating process without damping (photon recoil) adds Γ_heat to the right-hand side, which is
    the same as raising n_j by Γ_heat/Γ_j. Both noise factors of the pair (n + 1, n) are raised, and
    the noise matrix is rebuilt, so spectra and `steady_state_covariance` see the new bath.
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


@contextlib.contextmanager
def quiet():
    """
    Silence levisim's progress prints (e.g. inside `MultiTrap.heterodyne_spectrum`)
    """
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield

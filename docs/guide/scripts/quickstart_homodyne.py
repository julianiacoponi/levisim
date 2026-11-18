"""Quickstart: two particles in one cavity, homodyne spectrum of the cavity output"""
import numpy as np
import matplotlib.pyplot as plt

from levisim.core.plotting import set_plt_rcParams_to_xmgrace_style
from levisim.core.units import kHz
from levisim.multiphysics.enums import QuantumLinearTheoryMethod
from levisim.multiphysics.multitrap import MultiTrap
from levisim.multiphysics.utils.shelf import DEFAULT_cavity_kwargs, DEFAULT_gas_kwargs

set_plt_rcParams_to_xmgrace_style()

# two particles, each moving in x and y, in one red-detuned cavity (Δ = -2π × 450 kHz)
mtrap = MultiTrap(
    num_traps=2,
    Cartesian_axes_per_trap=2,
    gas_kwargs=DEFAULT_gas_kwargs,
    cavity_kwargs=DEFAULT_cavity_kwargs,
    ω_range=np.arange(125, 165, 0.02) * kHz,  # kHz ≡ 2π × 10³ rad/s, so this grid is angular
)
mtrap.info

S_hh = mtrap.homodyne_spectrum(QuantumLinearTheoryMethod.LINEAR_AMPLIFIER_MODEL)

fig, ax = plt.subplots(figsize=(6, 4))
ax.semilogy(mtrap.ω_range / kHz, S_hh, color='k')
for label, dim in mtrap.dimensions.items():
    ax.axvline(dim.Ω / kHz, ls=':', lw=1, color='grey')
    ax.text(dim.Ω / kHz, 1.15 * S_hh.max(), f'$\\Omega_{{{label[0]}_{label[1:]}}}$', ha='center', va='bottom')
ax.set_xlabel(r'$\omega / 2\pi$ (kHz)')
ax.set_ylabel(r'$S_{hh}(\omega)$ (shot-noise units)')
ax.set_ylim(top=4 * S_hh.max())
fig.tight_layout()
fig.savefig('quickstart_homodyne.png', dpi=150)

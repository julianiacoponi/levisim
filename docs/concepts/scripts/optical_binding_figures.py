"""
Figure of the optical binding section of docs/concepts/interactions.md: the axial coupling rates of
two 105 nm silica spheres in 400 mW, 730 nm-waist tweezers (the parameters of Rieser et al., Science
377, 987 (2022)) against the separation and against the relative optical phase, and the far-field
zero along the polarisation

    python docs/concepts/scripts/optical_binding_figures.py

Writes docs/concepts/figures/optical_binding.png.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from levisim.core.constants import π
from levisim.core.logger import set_console_level
from levisim.core.plotting import set_plt_rcParams_to_xmgrace_style
from levisim.core.units import mW, nm, μm
from levisim.physics import Bead, OpticalBinding, Trap, Tweezer

FIGURES = Path(__file__).resolve().parents[1] / 'figures'
set_plt_rcParams_to_xmgrace_style()
set_console_level('ERROR')


def pair(d0, Δφ, axis=(1, 0, 0)):
    return [Trap(bead=Bead(r=105*nm), tweezer=Tweezer(Power=400*mW, wx0=730*nm, wy0=730*nm, φ=Δφ)),
            Trap(bead=Bead(r=105*nm), tweezer=Tweezer(Power=400*mW, wx0=730*nm, wy0=730*nm),
                 position=tuple(d0 * np.asarray(axis, dtype=float)))]


def axial_rates(d0, Δφ, axis=(1, 0, 0)):
    """(g12, g21)/Ωz: the rates at which z2 drives z1 and z1 drives z2, g_jk = -K_jk/(2 m Ωz)"""
    traps = pair(d0, Δφ, axis)
    K = OpticalBinding(traps=traps).stiffness_blocks(np.array([t.position for t in traps]), [t.bead for t in traps])
    m, Ω = traps[0].bead.m, traps[0].Ωj['z']
    return -K[0, 1, 2, 2] / (2 * m * Ω**2), -K[1, 0, 2, 2] / (2 * m * Ω**2)


separations = np.linspace(1.0, 6.0, 501) * μm
phases = np.linspace(0, 2 * π, 241)
fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))

ax = axes[0]
rates = np.array([axial_rates(d0, 0.5 * π) for d0 in separations])
ax.plot(separations / μm, rates[:, 0], 'C0', label=r'$g_{12}/\Omega$ (2 drives 1)')
ax.plot(separations / μm, rates[:, 1], 'C1', label=r'$g_{21}/\Omega$ (1 drives 2)')
ax.axhline(0, color='0.5', lw=0.8)
ax.set_xlabel(r'separation $d_0$ along $x$ (μm)')
ax.set_title(r'$\Delta\phi = \pi/2$: non-reciprocal', fontsize=10)
ax.legend(fontsize=8)

ax = axes[1]
rates = np.array([axial_rates(3.2 * μm, Δφ) for Δφ in phases])
ax.plot(phases / π, (rates[:, 0] + rates[:, 1]) / 2, 'C2', label=r'reciprocal $g_r/\Omega$')
ax.plot(phases / π, (rates[:, 0] - rates[:, 1]) / 2, 'C3', label=r'anti-reciprocal $g_a/\Omega$')
ax.axhline(0, color='0.5', lw=0.8)
ax.set_xlabel(r'$\Delta\phi = \phi_1 - \phi_2$ ($\pi$)')
ax.set_title(r'$d_0$ = 3.2 μm', fontsize=10)
ax.legend(fontsize=8)

ax = axes[2]
for axis, colour, name in (((1, 0, 0), 'C0', r'along $x$ ($\perp$ polarisation)'), ((0, 1, 0), 'C1', r'along $y$ ($\parallel$ polarisation)')):
    rates = np.array([axial_rates(d0, 0.0, axis)[0] for d0 in separations])
    ax.semilogy(separations / μm, np.abs(rates), colour, lw=0.8, label=name)
ax.semilogy(separations / μm, 0.2 * (1 * μm / separations), 'k:', lw=1, label=r'$\propto 1/d_0$')
ax.semilogy(separations / μm, 0.1 * (1 * μm / separations)**2, 'k--', lw=1, label=r'$\propto 1/d_0^2$')
ax.set_ylim(1e-5, 1)
ax.set_xlabel(r'separation $d_0$ (μm)')
ax.set_title(r'$|g_{12}|/\Omega$, $\Delta\phi = 0$', fontsize=10)
ax.legend(fontsize=7, loc='lower left')
fig.tight_layout()
fig.savefig(FIGURES / 'optical_binding.png', dpi=150)
print('wrote', FIGURES / 'optical_binding.png')

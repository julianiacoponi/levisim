"""
Figures of docs/concepts/stability.md: eigenvalues of the drift matrix as the second tweezer's phase
is swept for the default 2×2 system, and the two-mode model of avoided crossings, level attraction
and exceptional points

    python docs/concepts/scripts/stability_figures.py

Writes docs/concepts/figures/stability_eigenvalue_trajectories.png,
docs/concepts/figures/stability_phase_sweep.png and docs/concepts/figures/stability_two_mode_model.png.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from levisim.core.logger import set_console_level
from levisim.core.plotting import set_plt_rcParams_to_xmgrace_style
from levisim.core.units import degrees, kHz
from levisim.multiphysics.multitrap import MultiTrap
from levisim.multiphysics.utils.shelf import DEFAULT_cavity_kwargs, DEFAULT_gas_kwargs

FIGURES = Path(__file__).resolve().parents[1] / 'figures'
set_plt_rcParams_to_xmgrace_style()
# the unstable phases would each log the construction WARNING
set_console_level('ERROR')


def mechanical_eigenvalues(φ2, **kwargs):
    """the four eigenvalues with Im λ > 0 in the mechanical band (one per normal mode), by frequency"""
    mtrap = MultiTrap(num_traps=2, Cartesian_axes_per_trap=2, φn_dict={2: φ2}, ω_range=np.array([0.0]), **kwargs)
    λ = mtrap.drift_eigenvalues
    λ = λ[(λ.imag > 100 * kHz) & (λ.imag < 200 * kHz) & (λ.real > -1e5)]
    return λ[np.argsort(λ.imag)]


shelf_kwargs = dict(gas_kwargs=DEFAULT_gas_kwargs, cavity_kwargs=DEFAULT_cavity_kwargs)
φ2s = np.arange(0, 180.01, 0.5) * degrees
λs = np.array([mechanical_eigenvalues(φ2, **shelf_kwargs) for φ2 in φ2s])
λs_defaults = np.array([mechanical_eigenvalues(φ2) for φ2 in φ2s])
unstable = λs.real.max(axis=1) > 0
print(f'RESULT unstable for {φ2s[unstable].min() / degrees:.1f}° to {φ2s[unstable].max() / degrees:.1f}° on the 0.5° grid')

# ---- eigenvalue trajectories in the complex plane --------------------------------------------------
# λ(φ2) = λ(180° - φ2) (see the page), so the trajectories for 90° to 180° retrace those for 0° to 90°
quarter = φ2s <= 90 * degrees
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw=dict(width_ratios=[1, 1]))
norm = Normalize(0, 90)
for ax, (low, high), (left, right) in zip(axes, [(128, 158), (133.5, 138)], [(-10000, 4000), (-3500, 3500)]):
    ax.axvspan(0, right, color='0.9', zorder=0)
    ax.axvline(0, color='k', lw=1)
    for mode in range(λs.shape[1]):
        points = ax.scatter(λs[quarter, mode].real, λs[quarter, mode].imag / kHz, c=φ2s[quarter] / degrees,
                            cmap='viridis', norm=norm, s=6, zorder=2)
    for φ2_mark, marker in ((0, 'o'), (30, 's'), (90, '*')):
        k = np.argmin(abs(φ2s / degrees - φ2_mark))
        ax.plot(λs[k].real, λs[k].imag / kHz, marker, mfc='none', mec='C3', ms=9, mew=1.5, ls='none',
                label=rf'$\phi_2 = {φ2_mark}^\circ$', zorder=3)
    ax.set_xlim(left, right)
    ax.set_ylim(low, high)
    ax.set_xlabel(r'$\mathrm{Re}\,\lambda$ (s$^{-1}$)')
    ax.set_ylabel(r'$\mathrm{Im}\,\lambda / 2\pi$ (kHz)')
axes[0].text(400, 129.5, 'unstable', fontsize=10)
axes[0].text(-9500, 129.5, 'stable', fontsize=10)
axes[1].legend(loc='lower left', fontsize=9, frameon=False)
fig.colorbar(points, ax=axes, label=r'$\phi_2$ (degrees; 90° to 180° retrace 90° to 0°)', pad=0.02)
fig.savefig(FIGURES / 'stability_eigenvalue_trajectories.png', dpi=150, bbox_inches='tight')

# ---- frequencies and real parts against the phase --------------------------------------------------
fig, (ax_f, ax_r) = plt.subplots(2, 1, figsize=(6.5, 6), sharex=True)
for ax in (ax_f, ax_r):
    ax.axvspan(φ2s[unstable].min() / degrees, φ2s[unstable].max() / degrees, color='0.9', zorder=0)
for mode in range(λs.shape[1]):
    ax_f.plot(φ2s / degrees, λs[:, mode].imag / kHz, color='k')
    ax_r.plot(φ2s / degrees, λs[:, mode].real, color='k')
ax_r.plot(φ2s / degrees, λs_defaults.real.max(axis=1), color='C3', ls='--', label=r'max Re $\lambda$, Cavity() defaults')
ax_r.axhline(0, color='grey', lw=1)
ax_f.set_ylabel(r'$\mathrm{Im}\,\lambda / 2\pi$ (kHz)')
ax_r.set_ylabel(r'$\mathrm{Re}\,\lambda$ (s$^{-1}$)')
ax_r.set_xlabel(r'second tweezer phase $\phi_2$ (degrees), $\phi_1 = 0$')
ax_r.set_xlim(0, 180)
ax_r.set_ylim(-10000, 10000)
ax_r.legend(loc='upper left', fontsize=9, frameon=False)
fig.tight_layout()
fig.savefig(FIGURES / 'stability_phase_sweep.png', dpi=150)

# ---- two-mode model -------------------------------------------------------------------------------
# dα/dt = M α with M = [[-iΩ1 - γ/2, -i g12], [-i g21, -iΩ2 - γ/2]]; eigenvalues
# λ± = -iΩ̄ - γ/2 ± sqrt(-(δ/2)² - g12·g21), δ = Ω1 - Ω2 (all rates in units of g)
δ = np.linspace(-4, 4, 801)
γ = 0.5
cases = {
    r'reciprocal: $g_{12}g_{21} = +g^2$': +1.0,
    r'dissipative: $g_{12}g_{21} = -g^2$': -1.0,
}
fig, axes = plt.subplots(2, 2, figsize=(9, 5.5), sharex=True)
for column, (title, product) in enumerate(cases.items()):
    root = np.sqrt(-(δ / 2)**2 - product + 0j)
    for sign, style in ((+1, '-'), (-1, '--')):
        λ = -γ / 2 + sign * root  # in the frame rotating at Ω̄
        axes[0, column].plot(δ, λ.imag, 'k' + style)
        axes[1, column].plot(δ, λ.real, 'k' + style)
    axes[0, column].plot(δ, δ / 2, color='grey', lw=0.8, ls=':')
    axes[0, column].plot(δ, -δ / 2, color='grey', lw=0.8, ls=':')
    axes[1, column].axhline(0, color='C3', lw=1)
    axes[0, column].set_title(title, fontsize=10)
    axes[1, column].set_xlabel(r'bare detuning $\delta / g$')
    if product < 0:
        for edge in (-2, 2):
            for row in (0, 1):
                axes[row, column].axvline(edge, color='C0', lw=0.8, ls='-.')
        axes[0, column].text(2.1, 1.5, 'EP', color='C0')
axes[0, 0].set_ylabel(r'$(\mathrm{Im}\,\lambda - \bar\Omega) / g$')
axes[1, 0].set_ylabel(r'$\mathrm{Re}\,\lambda / g$')
for ax in axes.flat:
    ax.set_xlim(δ[0], δ[-1])
for ax in axes[1]:
    ax.set_ylim(-1.4, 0.9)
fig.tight_layout()
fig.savefig(FIGURES / 'stability_two_mode_model.png', dpi=150)

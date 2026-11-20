"""
Worked example for docs/guide/cavity_simulations.md

Simulates the shelf 1×2 system (one particle, x and y, shelf.DEFAULT_cavity_kwargs: κ/2π = 390.35 kHz,
Δ/2π = −450 kHz, θhom = 45°) with the exact linear CavitySimulation, records a homodyne, a
heterodyne (Ωhet/2π = 1 MHz) and both positions, and compares their averaged PSDs with the
symmetrised MultiTrap spectra; then lets the unstable 2×2 system (default tweezer phases, φ2 = 90°)
grow from zero and fits its growth rate. Seeded; runs in about a minute.

    python docs/guide/scripts/cavity_simulation_example.py

Writes docs/guide/figures/cavity_simulation_psd.png and prints the numbers quoted in the guide.
"""
import time
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from levisim.core.plotting import set_plt_rcParams_to_xmgrace_style
from levisim.core.units import Hz, kHz, ms, μs
from levisim.multiphysics.enums.quantum_linear_theory_method import (
    QuantumLinearTheoryMethod,
)
from levisim.multiphysics.multitrap import MultiTrap
from levisim.multiphysics.utils import shelf
from levisim.simulation import Heterodyne, Homodyne, Position

FIGURES = Path(__file__).resolve().parents[1] / 'figures'
LAM = QuantumLinearTheoryMethod.LINEAR_AMPLIFIER_MODEL


def at(mtrap, ω, spectrum):
    old = mtrap.ω_range
    mtrap.ω_range = np.asarray(ω)
    try:
        return spectrum()
    finally:
        mtrap.ω_range = old


def symmetrised(mtrap, f, spectrum):
    ω = 2 * np.pi * f
    S = at(mtrap, np.concatenate([ω, -ω]), spectrum)
    return S[:len(f)], S[len(f):], (S[:len(f)] + S[len(f):]) / 2


# --- the stable 1×2 system ---------------------------------------------------------------------
mtrap = MultiTrap(num_traps=1, Cartesian_axes_per_trap=2, cavity_kwargs=shelf.DEFAULT_cavity_kwargs,
                  gas_kwargs=shelf.DEFAULT_gas_kwargs)
Ωhet = 1e6 * Hz
start = time.time()
sim = mtrap.simulate(num_samples=2**20, Δt=0.1 * μs, seed=2026, num_trajectories=16,
                     detection=[Homodyne(), Heterodyne(Ωhet=Ωhet), Position('x1'), Position('y1')])
runtime = time.time() - start
print(f'{sim.num_trajectories} × {sim.num_samples} samples of Δt = {sim.Δt / μs:.2f} μs in {runtime:.1f} s '
      f'({1e6 * runtime / (sim.num_samples * sim.num_trajectories):.2f} μs per sample and trajectory, {sim.backend_used})')

segments = 16
f, S_hom = sim.averaged_psd('homodyne', num_segments=segments)
band = (f > 20e3) & (f < 390e3)
plus, minus, sym = symmetrised(mtrap, f[band], lambda: mtrap.homodyne_spectrum(LAM))
sinc2 = np.sinc(f[band] * sim.Δt)**2
r = S_hom[band] / (sym * sinc2)
print(f'homodyne 20-390 kHz: median ratio {np.median(r):.4f}, mean {np.mean(r):.4f}, 5-95% {np.percentile(r, 5):.3f}-'
      f'{np.percentile(r, 95):.3f} (expected scatter ±{1 / np.sqrt(segments * sim.num_trajectories):.3f})')
for label in ('x1', 'y1'):
    _, S_x = sim.averaged_psd(label, num_segments=segments)
    *_, Sq = symmetrised(mtrap, f[band], lambda label=label: mtrap.position_power_spectrum(label, LAM))
    theory = Sq * mtrap.dimensions[label].zpf**2 * sinc2
    rx = S_x[band] / theory
    print(f'{label} 20-390 kHz: median ratio {np.median(rx):.4f}, mean {np.mean(rx):.4f}')

f_het, S_het = sim.averaged_psd('heterodyne', num_segments=segments)
band_het = (f_het > 500e3) & (f_het < 1500e3)
het_theory = sim.record_spectrum('heterodyne', f_het[band_het])
rh = S_het[band_het] / het_theory
print(f'heterodyne 0.5-1.5 MHz: median ratio {np.median(rh):.4f}, mean {np.mean(rh):.4f}')
Ωx = mtrap.dimensions['x1'].Ω / Hz
lower, upper = sim.record_spectrum('heterodyne', [1e6 - Ωx, 1e6 + Ωx], kind='continuous')
print(f'heterodyne x1 sidebands at 1 MHz ∓ Ωx/2π: {lower:.4g}, {upper:.4g} (ratio {lower / upper:.3f})')

# --- the unstable 2×2 system -------------------------------------------------------------------
unstable = MultiTrap(num_traps=2, Cartesian_axes_per_trap=2, cavity_kwargs=shelf.DEFAULT_cavity_kwargs,
                     gas_kwargs=shelf.DEFAULT_gas_kwargs)
grow = unstable.simulate(num_samples=40_000, seed=1, num_trajectories=16, initial_state='zero', store_state=True,
                         detection=[Position('x1')])
eigenvalues, left = np.linalg.eig(grow.system.A.T)
mode = np.argmax(eigenvalues.real)
power = np.mean(np.abs(grow.states @ left[:, mode])**2, axis=0)
t = np.arange(power.size) * grow.Δt
late = t > 1.5e-3
rate = np.polyfit(t[late], np.log(power[late]), 1)[0] / 2
print(f'unstable 2×2: fitted growth rate {rate:.1f} s⁻¹, max_growth_rate {unstable.max_growth_rate:.1f} s⁻¹ '
      f'(ratio {rate / unstable.max_growth_rate:.5f})')

# --- figure ------------------------------------------------------------------------------------
set_plt_rcParams_to_xmgrace_style()
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
ax = axes[0, 0]
ax.semilogy(f[band] / 1e3, S_hom[band], color='0.6', lw=1, label='simulation')
ax.semilogy(f[band] / 1e3, sym * sinc2, 'k', lw=1.5, label=r'MultiTrap, $\frac{1}{2}[S(\omega) + S(-\omega)]\,\mathrm{sinc}^2(f\Delta t)$')
ax.set_xlabel('frequency (kHz)')
ax.set_ylabel(r'$S_{hh}$ (shot-noise units)')
ax.set_title(r'homodyne, $\theta = 45^\circ$', fontsize=12)
ax.legend(fontsize=9, frameon=False)

ax = axes[0, 1]
for label, colour in (('x1', 'C0'), ('y1', 'C3')):
    _, S_x = sim.averaged_psd(label, num_segments=segments)
    *_, Sq = symmetrised(mtrap, f[band], lambda label=label: mtrap.position_power_spectrum(label, LAM))
    ax.semilogy(f[band] / 1e3, S_x[band], color=colour, alpha=0.4, lw=1)
    ax.semilogy(f[band] / 1e3, Sq * mtrap.dimensions[label].zpf**2 * sinc2, color=colour, lw=1.5, label=label)
ax.set_xlabel('frequency (kHz)')
ax.set_ylabel(r'$S_{xx}$ (m$^2$/Hz)')
ax.set_title('positions (light: simulation)', fontsize=12)
ax.legend(fontsize=9, frameon=False)

ax = axes[1, 0]
ax.semilogy(f_het[band_het] / 1e3, S_het[band_het], color='0.6', lw=1, label='simulation')
ax.semilogy(f_het[band_het] / 1e3, het_theory, 'k', lw=1.5, label='exact sampled spectrum')
ax.axvline(1e3, color='0.8', lw=1, zorder=0)
ax.set_xlabel('frequency (kHz)')
ax.set_ylabel(r'$S_\mathrm{het}$ (shot-noise units)')
ax.set_title(r'heterodyne, $\Omega_\mathrm{het}/2\pi = 1$ MHz', fontsize=12)
ax.legend(fontsize=9, frameon=False)

ax = axes[1, 1]
ax.semilogy(t / ms, power, color='0.4', lw=1.5, label=r'$\langle|c(t)|^2\rangle$, 16 trajectories')
fit = np.exp(np.polyval(np.polyfit(t[late], np.log(power[late]), 1), t))
ax.semilogy(t / ms, fit, 'C3--', lw=1.2, label=fr'fit: {rate:.0f} s$^{{-1}}$ (theory {unstable.max_growth_rate:.0f})')
ax.set_xlabel('time (ms)')
ax.set_ylabel('least-stable mode power')
ax.set_title(r'unstable 2×2 ($\phi_2 = 90^\circ$) from zero', fontsize=12)
ax.legend(fontsize=9, frameon=False)
for label, ax in zip('abcd', axes.flat):
    ax.text(0.02, 0.95, f'({label})', transform=ax.transAxes, va='top', fontsize=12)
fig.tight_layout()
FIGURES.mkdir(exist_ok=True)
fig.savefig(FIGURES / 'cavity_simulation_psd.png', dpi=150)
print('figure written to', FIGURES / 'cavity_simulation_psd.png')
print(f'kHz check: Ωx/2π = {mtrap.dimensions["x1"].Ω / kHz:.2f} kHz, Ωy/2π = {mtrap.dimensions["y1"].Ω / kHz:.2f} kHz')

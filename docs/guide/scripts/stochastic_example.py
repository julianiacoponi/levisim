"""
Worked example for docs/guide/stochastic_simulations.md

Simulates the default free-space Trap (directed force β = √2 at ψ = 45°, cold damping in x, y)
with a detection floor, estimates Sxx, Syy and the co-spectrum Cxy, and compares them with
SpectralDensityTheory and with equipartition. Seeded; runs in well under a minute.

    python docs/guide/scripts/stochastic_example.py

Writes docs/guide/figures/stochastic_psd.png, docs/guide/figures/stochastic_broadband.png and prints
the numbers quoted in the guide.
"""
from pathlib import Path
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from levisim.analysis.spectral_density_estimate import SpectralDensityEstimate
from levisim.core.plotting import set_plt_rcParams_to_xmgrace_style
from levisim.core.units import Hz, kHz, nm, μs
from levisim.data.simulation import Simulation
from levisim.physics.trap import Trap

FIGURES = Path(__file__).resolve().parents[1] / 'figures'

# --- simulate -----------------------------------------------------------------------------------
trap = Trap()  # defaults: 485 mW, 1064 nm tweezer; 60.1 nm bead; 4e-3 mbar; Γfb,x/y = 2.1, 2.5 kHz
G_floor = 1e-27  # one-sided detection floor [m²/Hz]
start = time.time()
sim = Simulation(trap=trap, num_samples=400_000, Δt=0.1*μs, seed=2026, detection_floor=G_floor)
runtime = time.time() - start

Sxx = SpectralDensityEstimate(sim, 'xx', num_segments=40)
Syy = SpectralDensityEstimate(sim, 'yy', num_segments=40)
Sxy = SpectralDensityEstimate(sim, 'xy', num_segments=40)
# a Blackman-Harris window removes the rectangular window's leakage from the resonance into the tail
Sxx_bh = SpectralDensityEstimate(sim, 'xx', num_segments=40, window='blackmanharris')
theory = trap.theory  # one-sided by default, like the estimates

print(f'simulated {sim.num_samples} samples ({sim.time_length*1e3:.0f} ms) in {runtime:.1f} s')
print(f'Γth/2π = {trap.Γth/Hz:.3f} Hz, Γx/2π = {trap.Γx/kHz:.4f} kHz, Γy/2π = {trap.Γy/kHz:.4f} kHz')
print(f'Ωx/2π = {trap.Ωx/kHz:.2f} kHz, Ωy/2π = {trap.Ωy/kHz:.2f} kHz, Ωz/2π = {trap.Ωz/kHz:.2f} kHz')
for axis, est in (('x', Sxx), ('y', Syy)):
    floor_var = G_floor / (2 * sim.Δt)
    T_var = trap.equipartition_factor[axis] * (np.var(sim.traces[axis].values) - floor_var)
    print(
        f'{axis}: T_eff = {trap.T_eff[axis]:.3f} K | via Var (floor removed) = {T_var:.3f} K'
        f' | via fit = {est.temperature_via_fit:.3f} K'
        f' | Ωfit/Ω = {est.Ωfit/trap.Ωj[axis]:.4f} | Γfit/Γ = {est.Γfit/trap.Γj[axis]:.3f}'
    )
for lo, hi in ((0.3e6, 0.5e6), (1e6, 4.5e6)):
    band = (Sxx.freq_range > lo) & (Sxx.freq_range < hi)
    model = np.mean(theory.Sxx(2*np.pi*Sxx.freq_range[band]) + G_floor)
    print(
        f'{lo/1e6}-{hi/1e6} MHz: mean Sxx / (theory + floor) = {np.mean(np.real(Sxx.values[band]))/model:.3f} (rectangular),'
        f' {np.mean(np.real(Sxx_bh.values[band]))/model:.3f} (Blackman-Harris)'
    )
cov_th = np.trapezoid(theory.Cxy(2*np.pi*Sxy.freq_range), Sxy.freq_range)
print(f'Cov(x, y): simulated {np.cov(sim.x.values, sim.y.values)[0, 1]/nm**2:.4f} nm², ∫Cxy df = {cov_th/nm**2:.4f} nm²')

# --- figures ------------------------------------------------------------------------------------
set_plt_rcParams_to_xmgrace_style()

f = Sxx.freq_range
sel = (f > 80e3) & (f < 220e3)
ω = 2*np.pi*f[sel]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.4, 6.4), sharex=True)
ax1.semilogy(f[sel]/1e3, np.real(Sxx.values[sel]), '.', ms=3, color='tab:blue', label=r'$S_{xx}$ simulated')
ax1.semilogy(f[sel]/1e3, np.real(Syy.values[sel]), '.', ms=3, color='tab:red', label=r'$S_{yy}$ simulated')
ax1.semilogy(f[sel]/1e3, theory.Sxx(ω) + G_floor, '-', color='navy', label='theory + floor')
ax1.semilogy(f[sel]/1e3, theory.Syy(ω) + G_floor, '-', color='darkred')
ax1.axhline(G_floor, ls='--', color='grey', lw=1, label='detection floor')
ax1.set_ylabel(r'one-sided PSD (m$^2$/Hz)')
ax1.legend(fontsize=9, loc='upper right')

ax2.plot(f[sel]/1e3, np.real(Sxy.values[sel]), '.', ms=3, color='tab:green', label=r'$C_{xy}$ simulated')
ax2.plot(f[sel]/1e3, theory.Cxy(ω), '-', color='darkgreen', label=r'$C_{xy}$ theory')
ax2.axhline(0, color='k', lw=0.8)
ax2.set_xlabel('frequency (kHz)')
ax2.set_ylabel(r'co-spectrum (m$^2$/Hz)')
ax2.legend(fontsize=9, loc='upper right')
fig.tight_layout()
fig.savefig(FIGURES / 'stochastic_psd.png', dpi=150)

fig, ax = plt.subplots(figsize=(6.4, 4.2))
wide = (f > 0) & (f < sim.Nyquist_frequency)
ax.loglog(f[wide], np.real(Sxx.values[wide]), '.', ms=2, color='lightsteelblue', label='simulated, rectangular window')
ax.loglog(f[wide], np.real(Sxx_bh.values[wide]), '.', ms=2, color='tab:blue', label='simulated, Blackman-Harris')
ax.loglog(f[wide], theory.Sxx(2*np.pi*f[wide]) + G_floor, '-', color='navy', label='theory + floor')
ax.loglog(f[wide], theory.Sxx(2*np.pi*f[wide]), ':', color='navy', lw=1, label='theory (motion only)')
ax.axhline(G_floor, ls='--', color='grey', lw=1, label='detection floor')
ax.set_xlabel('frequency (Hz)')
ax.set_ylabel(r'one-sided PSD (m$^2$/Hz)')
ax.legend(fontsize=9, loc='lower left')
fig.tight_layout()
fig.savefig(FIGURES / 'stochastic_broadband.png', dpi=150)
print('figures written to', FIGURES)

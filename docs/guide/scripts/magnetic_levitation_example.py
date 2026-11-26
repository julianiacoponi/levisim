"""
Figures and numbers of the guide page "Magnetic levitation" (docs/guide/magnetic_levitation.md)

A pyrolytic-graphite and a bismuth microsphere between two NdFeB blocks with like poles facing (a
permanent-magnet quadrupole), with gravity along -z: the magnetic_trap_report of each, the field map
with the equilibria, and a time-domain simulation of the graphite sphere whose PSD peaks are compared
with the report's frequencies.

    python docs/guide/scripts/magnetic_levitation_example.py

writes docs/guide/figures/magnetic_levitation_field.png and magnetic_levitation_psd.png (about 2 min: the NumPy loop evaluates the block fields at every step).
"""
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..', '..')))

from levisim.core.constants import μ0
from levisim.core.plotting import set_plt_rcParams_to_xmgrace_style
from levisim.core.units import Hz, mbar, μm
from levisim.physics import Sphere, Trap
from levisim.physics.fields import CuboidMagnet, FieldSum, Gravity, MagneticTrap
from levisim.physics.gas import Gas
from levisim.physics.materials import BISMUTH, NDFEB, PYROLYTIC_GRAPHITE
from levisim.simulation import FreeSpaceSimulation

FIGURES = os.path.join(HERE, '..', 'figures')
set_plt_rcParams_to_xmgrace_style()

# two 500 μm NdFeB cubes (B_r = 1.23 T, the library's sintered NdFeB), magnetised towards each other,
# 300 μm apart: a field zero at the centre with gradients 1.4 and -2.7 × 10³ T/m
B_r = float(NDFEB.magnetisation[2] * μ0)
L, gap = 500 * μm, 300 * μm
source = FieldSum([
    CuboidMagnet((L, L, L), remanence=B_r, direction=(0, 0, 1), centre=(0, 0, -(gap + L) / 2)),
    CuboidMagnet((L, L, L), remanence=B_r, direction=(0, 0, -1), centre=(0, 0, (gap + L) / 2)),
])
print('gradient at the centre [T/m]:', np.round(np.diag(source.gradB(np.zeros(3))), 1))

reports = {}
for material in (PYROLYTIC_GRAPHITE, BISMUTH):
    trap = Trap(fields=[MagneticTrap(source), Gravity(direction=(0, 0, -1))], particle=Sphere(r=5 * μm, material=material))
    reports[material.name] = trap.magnetic_trap_report(r_guess=(0, 0, 0), scan_range=100 * μm)
    print(material.name)
    print(trap.magnetic_trap_report_str(r_guess=(0, 0, 0), scan_range=100 * μm))

# field map in the x-z plane
x = np.linspace(-140, 140, 281) * μm
z = np.linspace(-170, 170, 341) * μm
X, Z = np.meshgrid(x, z)
B = np.linalg.norm(source.B(np.stack([X, 0 * X, Z], -1)), axis=-1)
fig, ax = plt.subplots(figsize=(5.5, 4.5))
im = ax.pcolormesh(X / μm, Z / μm, B, shading='auto', cmap='viridis')
ax.contour(X / μm, Z / μm, B, levels=np.arange(0.1, 1.0, 0.1), colors='w', linewidths=0.5)
for (name, rep), marker in zip(reports.items(), ('r+', 'mx')):
    ax.plot(rep['equilibrium'][0] / μm, rep['equilibrium'][2] / μm, marker, ms=12, mew=2, label=name)
ax.axhspan(gap / 2 / μm, 170, color='0.6', alpha=0.6)
ax.axhspan(-170, -gap / 2 / μm, color='0.6', alpha=0.6)
ax.set_xlabel('x [μm]')
ax.set_ylabel('z [μm]')
ax.legend(loc='upper right', fontsize=9)
fig.colorbar(im, ax=ax, label='|B| [T]')
fig.savefig(os.path.join(FIGURES, 'magnetic_levitation_field.png'), dpi=130, bbox_inches='tight')

# time domain: the graphite sphere at 0.1 mbar, started at its equilibrium
rep = reports[PYROLYTIC_GRAPHITE.name]
particle = Sphere(r=5 * μm, material=PYROLYTIC_GRAPHITE)
trap = Trap(fields=[MagneticTrap(source), Gravity(direction=(0, 0, -1))], particle=particle,
            gas=Gas(P=0.1 * mbar), β=0.0, linearisation_point=rep['equilibrium'])
t0 = time.time()
sim = FreeSpaceSimulation(trap=trap, num_samples=40_000, Δt=0.2e-3, X0=np.r_[rep['equilibrium'], 0, 0, 0], seed=7)
sim.simulate_dynamics()
print(f'simulation: {time.time() - t0:.0f} s, Γ/2π = {trap.Γx / Hz:.2f} Hz')
fig, ax = plt.subplots(figsize=(6, 3.8))
peaks = []
for axis, colour, Ω in zip('xyz', ('C0', 'C1', 'C2'), rep['Ω_axes']):
    f, S = welch(sim.traces[axis].values, fs=1 / sim.Δt, nperseg=2**13)
    ax.semilogy(f, S, color=colour, label=axis)
    ax.axvline(Ω / Hz, color=colour, ls='--', lw=0.8)
    i = int(np.argmax(S[1:])) + 1
    a, b, c = np.log(S[i - 1:i + 2])
    peaks.append(f[i] + 0.5 * (a - c) / (a - 2 * b + c) * (f[1] - f[0]))
ax.set_xlim(0, 300)
ax.set_xlabel('frequency [Hz]')
ax.set_ylabel('PSD [m²/Hz]')
ax.legend()
fig.savefig(os.path.join(FIGURES, 'magnetic_levitation_psd.png'), dpi=130, bbox_inches='tight')
print('PSD peaks [Hz]:', np.round(peaks, 2), ' report Ω/2π [Hz]:', np.round(rep['Ω_axes'] / Hz, 2))

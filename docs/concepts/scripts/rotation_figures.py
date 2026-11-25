"""
Figures of docs/concepts/rotation.md: rigid-body checks of RigidBodySimulation, and the libration
spectrum of the time domain against the linear model of MultiTrap

    python docs/concepts/scripts/rotation_figures.py

Writes docs/concepts/figures/rotation_rigid_body.png and docs/concepts/figures/rotation_libration_psd.png,
and prints the numbers quoted on the page (lines starting RESULT).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

from levisim.core.constants import kB
from levisim.core.logger import logger
from levisim.core.plotting import set_plt_rcParams_to_xmgrace_style
from levisim.core.units import kHz, mbar, nm, μs
from levisim.multiphysics.enums.quantum_linear_theory_method import QuantumLinearTheoryMethod as Q
from levisim.multiphysics.multitrap import MultiTrap
from levisim.physics import Ellipsoid, Gas, HarmonicPotential, Spheroid, Trap, Tweezer
from levisim.physics.particle import body_axis_to_quaternion
from levisim.simulation import RigidBodySimulation

logger.remove()
FIGURES = Path(__file__).resolve().parents[1] / 'figures'
set_plt_rcParams_to_xmgrace_style()

# ---- torque-free rigid bodies --------------------------------------------------------------------
free = [HarmonicPotential(np.eye(3) * 1e-6)]
top = Spheroid(a=100*nm, b=50*nm)
I1, _, I3 = np.diag(top.inertia_tensor)
ω0 = np.array([1e5, 0, 3e5])
sim = RigidBodySimulation(trap=Trap(fields=free, particle=top), num_samples=20000, Δt=1e-8, ω0=ω0, gas_damping=False)
t = sim.time_array
ω = sim.angular_velocities_body()
Ωp = (I3 - I1) / I1 * ω0[2]
error = np.max(np.abs(ω[:, 0] - ω0[0] * np.cos(Ωp * t))) / ω0[0]
E = sim.rotational_kinetic_energies()
print(f'RESULT symmetric top: body precession Ωp/2π = {Ωp / kHz:.3f} kHz; max |ω1 - ω⊥cos Ωp t|/ω⊥ = {error:.2e}; '
      f'energy drift {np.ptp(E) / E[0]:.1e}; | |q| - 1 | ≤ {sim.quaternion_norm_error():.1e}')

body = Ellipsoid(90*nm, 60*nm, 40*nm)
order = np.argsort(np.diag(body.inertia_tensor))
runs = {}
for name, axis in zip(('minor axis', 'intermediate axis', 'major axis'), order):
    start = np.full(3, 1e3)
    start[axis] = 1e6
    s = RigidBodySimulation(trap=Trap(fields=free, particle=body), num_samples=20000, Δt=2e-8, ω0=start, gas_damping=False)
    runs[name] = (s.time_array, s.angular_velocities_body()[:, axis] / start[axis])
    E = s.rotational_kinetic_energies()
    print(f'RESULT spin about the {name}: min ω/ω0 = {runs[name][1].min():+.3f}, energy drift {np.ptp(E) / E[0]:.1e}')

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
ax = axes[0]
ax.plot(t / μs, ω[:, 0] / ω0[0], color='C0', label=r'$\omega_1/\omega_\perp$ (simulation)')
ax.plot(t / μs, ω[:, 1] / ω0[0], color='C1', label=r'$\omega_2/\omega_\perp$ (simulation)')
ax.plot(t[::400] / μs, np.cos(Ωp * t[::400]), 'o', color='C0', ms=4, label=r'$\cos\Omega_p t$')
ax.plot(t[::400] / μs, np.sin(Ωp * t[::400]), 's', color='C1', ms=4, label=r'$\sin\Omega_p t$')
ax.set_xlabel(r'time [$\mu$s]')
ax.set_ylabel('body-frame angular velocity')
ax.set_title('torque-free symmetric top', fontsize=12)
ax.set_ylim(-1.7, 1.15)
ax.legend(frameon=False, fontsize=8, loc='lower center', ncol=2)
ax = axes[1]
for (name, (ts, values)), colour in zip(runs.items(), ('C2', 'C3', 'C4')):
    ax.plot(ts / μs, values, color=colour, label=f'spin about the {name}')
ax.set_xlabel(r'time [$\mu$s]')
ax.set_ylabel(r'$\omega_\mathrm{axis}/\omega_0$')
ax.set_title('intermediate-axis instability (ellipsoid 90:60:40 nm)', fontsize=12)
ax.legend(frameon=False, fontsize=9, loc='lower left')
fig.tight_layout()
fig.savefig(FIGURES / 'rotation_rigid_body.png', dpi=150)

# ---- libration spectrum: time domain against MultiTrap ------------------------------------------
gas = Gas(P=10 * mbar, T=3.0)  # a cold gas keeps the libration small (〈γ²〉 ≈ 1e-3 rad²), i.e. linear
particle = Spheroid(a=70*nm, b=50*nm, orientation=body_axis_to_quaternion((0, 1, 0)))
trap = Trap(tweezer=Tweezer(), particle=particle, gas=gas)
Ωγ, Γγ, I = trap.Ωθ['γ'], trap.Γrot_j['γ'], trap.Ij['γ']
K = np.diag(trap.torque_stiffness_tensor)[2]
Δt = 5e-9
sim = RigidBodySimulation(trap=trap, num_samples=400_000, Δt=Δt, seed=2)
tilt = -sim.lab_directions((0, 0, 1))[20_000:, 0]  # tilt of the symmetry axis towards x: γ
f, G = welch(tilt, fs=1 / Δt, nperseg=2**15)
ω = 2 * np.pi * f
S_τ = 2 * kB * gas.T * Γγ * I  # two-sided torque noise of the gas
G_theory = 2 * S_τ / I**2 / ((Ωγ**2 - ω**2)**2 + (Γγ * ω)**2)
variance = np.var(tilt)
print(f'RESULT libration γ: Ωγ/2π = {Ωγ / kHz:.2f} kHz, Γγ/2π = {Γγ / (2 * np.pi * 1e3):.3f} kHz (10 mbar); '
      f'〈γ²〉 = {variance:.3e} rad² against kB T/K = {kB * gas.T / K:.3e} rad² (ratio {variance * K / (kB * gas.T):.3f})')
peak = f[np.argmax(G)]
print(f'RESULT libration γ: PSD peak at {peak / 1e3:.1f} kHz against Ωγ/2π = {Ωγ / kHz:.1f} kHz')

# MultiTrap at the node (no light coupling of γ), without recoil: the thermal mechanical spectrum
particle_node = Spheroid(a=70*nm, b=50*nm, orientation=body_axis_to_quaternion((0, 1, 0)))
ω_range = 2 * np.pi * np.linspace(600e3, 1100e3, 2001)
mt = MultiTrap(num_traps=1, Cartesian_axes_per_trap=('x', 'α', 'γ'), beads=[particle_node], gas_kwargs={'P': 10 * mbar, 'T': 3.0},
               ω_range=ω_range, photon_recoil=False)
S_mt = mt.position_power_spectrum('γ1', Q.LINEAR_AMPLIFIER_MODEL, zpf_scaled=False)
dim = mt.dimensions['γ1']
# MultiTrap's spectrum is two-sided per unit angular frequency (∫ S dω/2π over both signs = 〈q²〉)
area = np.trapezoid(S_mt, ω_range) / (2 * np.pi)
print(f'RESULT MultiTrap γ1: Ω/2π = {dim.Ω / kHz:.2f} kHz, area of the positive-frequency peak ∫S dω/2π = {area:.3e} rad²'
      f' (half the variance kB T/K = {0.5 * kB * gas.T / K:.3e} rad²)')

fig, ax = plt.subplots(figsize=(6.5, 4.5))
mask = (f > 500e3) & (f < 1200e3)
ax.semilogy(f[mask] / 1e3, G[mask], color='C0', label='RigidBodySimulation (Welch)')
ax.semilogy(f[mask] / 1e3, G_theory[mask], '--', color='k', label=r'$2S_\tau/I^2 / [(\Omega_\gamma^2-\omega^2)^2+\Gamma_\gamma^2\omega^2]$')
ax.semilogy(ω_range / (2 * np.pi * 1e3), 2 * S_mt, ':', color='C3', lw=2.5, label=r'MultiTrap $\gamma_1$ (one-sided, per Hz)')
ax.set_xlabel('frequency [kHz]')
ax.set_ylabel(r'$G_{\gamma\gamma}$ [rad$^2$/Hz]')
ax.set_title('libration of a 70:50 nm spheroid, 10 mbar, 3 K gas', fontsize=12)
ax.legend(frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig(FIGURES / 'rotation_libration_psd.png', dpi=150)

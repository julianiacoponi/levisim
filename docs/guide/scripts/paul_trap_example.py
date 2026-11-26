"""
Worked example for docs/guide/paul_traps.md

Draws the first stability region of Mathieu's equation with the operating points of a ring trap, the
accuracy of the lowest- and next-order secular frequencies against the exact (Floquet) value, and a
seeded simulation of a ring trap (q = 0.3) under gravity whose spectrum shows the secular frequencies,
the micromotion sidebands and the excess micromotion at Ω_rf. Runs in about 15 s.

    python docs/guide/scripts/paul_trap_example.py

Writes docs/guide/figures/paul_trap_stability.png and docs/guide/figures/paul_trap_spectrum.png and
prints the numbers quoted in the guide.
"""
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import mathieu_a, mathieu_b

from levisim.core.constants import g
from levisim.core.plotting import set_plt_rcParams_to_xmgrace_style
from levisim.core.units import Hz, mbar, mm, nm
from levisim.physics.fields import Gravity, PaulTrap, mathieu_secular_beta
from levisim.physics.gas import Gas
from levisim.physics.particle import Sphere
from levisim.physics.trap import Trap
from levisim.simulation import FreeSpaceSimulation

FIGURES = Path(__file__).resolve().parents[1] / 'figures'
set_plt_rcParams_to_xmgrace_style()

# --- the stability region and the accuracy of the secular frequency ------------------------------
q = np.linspace(0, 1.2, 241)
a0 = np.array([mathieu_a(0, qq) for qq in q])
b1 = np.array([mathieu_b(1, qq) for qq in q])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
ax1.fill_between(q, a0, b1, color='C0', alpha=0.2, label='first stability region')
ax1.plot(q, a0, 'C0-', lw=1.5, label=r'$a_0(q)$  ($\beta = 0$)')
ax1.plot(q, b1, 'C3-', lw=1.5, label=r'$b_1(q)$  ($\beta = 1$)')
# a ring trap: (a, q) radial -> (-2a, -2q) axial, drawn at |q|
for q_r, a_r, marker in ((0.3, 0.0, 'o'), (0.3, 0.02, 's')):
    ax1.plot([q_r, 2*q_r], [a_r, -2*a_r], 'k' + marker, ms=6, mfc='white')
    ax1.annotate('', xy=(2*q_r, -2*a_r), xytext=(q_r, a_r), arrowprops={'arrowstyle': '->', 'color': '0.4'})
ax1.set_xlim(0, 1.2)
ax1.set_ylim(-0.6, 0.6)
ax1.set_xlabel(r'$|q|$')
ax1.set_ylabel(r'$a$')
ax1.legend(fontsize=8, loc='lower left')
ax1.set_title('ring trap: radial → axial operating points', fontsize=10)

qs = np.linspace(0.02, 0.85, 60)
exact = np.array([mathieu_secular_beta(0, qq, 'exact') for qq in qs])
for order, style in (('lowest', 'C0-'), ('next', 'C3--')):
    approx = np.array([mathieu_secular_beta(0, qq, order) for qq in qs])
    ax2.semilogy(qs, np.abs(approx/exact - 1), style, label=order)
ax2.set_xlabel(r'$|q|$  ($a = 0$)')
ax2.set_ylabel(r'$|\beta/\beta_\mathrm{exact} - 1|$')
ax2.legend()
fig.tight_layout()
fig.savefig(FIGURES / 'paul_trap_stability.png', dpi=150)

for qq in (0.1, 0.2, 0.4, 0.7):
    e = mathieu_secular_beta(0, qq, 'exact')
    print(f'q = {qq}: lowest {mathieu_secular_beta(0, qq, "lowest")/e - 1:+.2e}, '
          f'next {mathieu_secular_beta(0, qq, "next")/e - 1:+.2e}')

# --- a simulated ring trap under gravity -----------------------------------------------------------
particle = Sphere(r=100*nm, charge=500)
f_rf = 10e3
V = 0.3 * particle.m * (1*mm)**2 * (f_rf*Hz)**2 / (2 * particle.q)
paul = PaulTrap.ring(V, f_rf*Hz, r0=1*mm)
print(paul.report_str(particle))
trap = Trap(fields=[paul, Gravity()], particle=particle, gas=Gas(P=1e-9*mbar))
sag = trap.find_equilibrium()
print(f'sag {sag[0]/nm:.2f} nm = -g/Ωx² ({-g/trap.Ωx**2/nm:.2f} nm); excess micromotion '
      f'{paul.excess_micromotion_amplitude(sag, particle)[0]/nm:.2f} nm')

Δt = 1 / (40 * f_rf)
sim = FreeSpaceSimulation(trap=trap, num_samples=200_000, Δt=Δt, seed=3, zero_mean=False,
                          X0=np.array([sag[0] + 80*nm, 60*nm, 40*nm, 0, 0, 0]))
exact_Ω = paul.secular_frequencies(particle, 'exact') / Hz


def periodogram(values):
    values = values - values.mean()
    f = np.fft.rfftfreq(len(values), Δt)
    return f, np.abs(np.fft.rfft(values * np.hanning(len(values))))**2


fig, ax = plt.subplots(figsize=(8, 4))
for axis, colour in (('x', 'C0'), ('z', 'C1')):
    f, S = periodogram(sim.traces[axis].values)
    ax.semilogy(f/1e3, S / S.max(), colour, lw=0.8, label=axis)
marks = [(exact_Ω[0], r'$\Omega_x$'), (exact_Ω[2], r'$\Omega_z$'), (f_rf, r'$\Omega_\mathrm{rf}$'),
         (f_rf - exact_Ω[0], ''), (f_rf + exact_Ω[0], ''), (f_rf - exact_Ω[2], ''), (f_rf + exact_Ω[2], '')]
for f0, text in marks:
    ax.axvline(f0/1e3, color='0.6', ls=':', lw=1)
    if text:
        ax.text(f0/1e3 + 0.1, 0.3, text, fontsize=9)
ax.set_xlim(0, 13)
ax.set_ylim(1e-12, 3)
ax.set_xlabel('frequency [kHz]')
ax.set_ylabel('periodogram (normalised)')
ax.legend(loc='upper right')
fig.tight_layout()
fig.savefig(FIGURES / 'paul_trap_spectrum.png', dpi=150)


def peak(values, lo, hi):
    f, S = periodogram(values)
    band = np.flatnonzero((f > lo) & (f < hi))
    i = band[np.argmax(S[band])]
    lo_, mid, hi_ = np.log(S[i - 1:i + 2])
    return f[i] + 0.5*(lo_ - hi_)/(lo_ - 2*mid + hi_)*(f[1] - f[0])


for j, axis in ((0, 'x'), (2, 'z')):
    print(f'{axis}: simulated {peak(sim.traces[axis].values, 300, 4000):.1f} Hz, Floquet {exact_Ω[j]:.1f} Hz, '
          f'pseudopotential {paul.secular_frequencies(particle, "lowest")[j]/Hz:.1f} Hz')
x = sim.x.values
t = np.arange(len(x)) * Δt
amplitude = 2*np.hypot(np.mean(x*np.cos(f_rf*Hz*t)), np.mean(x*np.sin(f_rf*Hz*t)))
print(f'mean x {x.mean()/nm:.2f} nm; line at Ω_rf {amplitude/nm:.2f} nm')

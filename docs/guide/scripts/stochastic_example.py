"""
Worked example for docs/guide/stochastic_simulations.md and the README's stochastic section

Simulates 200 ms of the default free-space Trap (directed force β = √2 at ψ = 45°, cold damping in
x and y, none in z) with a detection floor, estimates Sxx, Syy and the co-spectrum Cxy, and compares
them with SpectralDensityTheory and with equipartition. Seeded; runs in a few minutes.

    python docs/guide/scripts/stochastic_example.py

Prints the numbers quoted in the guide and writes three figures into docs/guide/figures/:

stochastic_traces.png
    (a) z over the whole run: not cooled, it is heated from rest by the gas towards 300 K
    (b) a 150 µs stretch of x and y, feedback-cooled to about 1 K
    (c) the distribution of the position in the x-y plane, where the correlation that the directed
        force puts between x and y is far too small to see

stochastic_psd.png
    (a) Sxx and Syy, simulated and from SpectralDensityTheory
    (b) the co-spectrum Cxy, in which the same correlation is plain

stochastic_broadband.png
    (a) Sxx from 1 kHz to the Nyquist frequency with a rectangular and a Blackman-Harris window
    (b) the ratio of each estimate to theory + floor
"""
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

sys.path.insert(0, str(Path(__file__).resolve().parent / 'showcase'))
from common import COLOURS, save, setup_matplotlib  # noqa: E402

from levisim.analysis.spectral_density_estimate import SpectralDensityEstimate  # noqa: E402
from levisim.core.units import Hz, kHz, nm, μs  # noqa: E402
from levisim.data.simulation import Simulation  # noqa: E402
from levisim.physics.trap import Trap  # noqa: E402

FIGURES = Path(__file__).resolve().parents[1] / 'figures'
G_FLOOR = 1e-27  # one-sided detection floor [m²/Hz]
NUM_SEGMENTS = 100  # 2 ms each, so Δf = 0.5 kHz
AXIS_COLOURS = {'x': COLOURS['blue'], 'y': COLOURS['red'], 'z': COLOURS['teal']}
IN_PHASE, IN_ANTIPHASE = COLOURS['orange'], COLOURS['teal']


def simulate():
    """200 ms of the default Trap: 485 mW, 1064 nm tweezer; 60.1 nm bead; 4e-3 mbar; Γfb,x/y = 2.1, 2.5 kHz"""
    return Simulation(trap=Trap(), num_samples=2_000_000, Δt=0.1*μs, seed=2026, detection_floor=G_FLOOR)


def estimate(sim):
    """The spectral density estimates of the example, as {name: SpectralDensityEstimate}"""
    spectra = {AB: SpectralDensityEstimate(sim, AB, num_segments=NUM_SEGMENTS) for AB in ('xx', 'yy', 'xy')}
    # a Blackman-Harris window removes the rectangular window's leakage from the resonance into the tail
    spectra['xx, Blackman-Harris'] = SpectralDensityEstimate(sim, 'xx', num_segments=NUM_SEGMENTS, window='blackmanharris')
    return spectra


def report(sim, spectra):
    """Prints the numbers quoted in the guide"""
    trap, theory = sim.trap, sim.trap.theory  # the theory is one-sided by default, like the estimates
    Sxx, Sxy, Sxx_bh = spectra['xx'], spectra['xy'], spectra['xx, Blackman-Harris']
    print(f'{sim.num_samples} samples ({sim.time_length*1e3:.0f} ms), Δf = {Sxx.Δf:.0f} Hz, {Sxx.num_segments} averages')
    print(f'Γth/2π = {trap.Γth/Hz:.3f} Hz, Γx/2π = {trap.Γx/kHz:.4f} kHz, Γy/2π = {trap.Γy/kHz:.4f} kHz')
    print(f'Ωx/2π = {trap.Ωx/kHz:.2f} kHz, Ωy/2π = {trap.Ωy/kHz:.2f} kHz, Ωz/2π = {trap.Ωz/kHz:.2f} kHz')
    for axis in 'xy':
        est = spectra[2 * axis]
        floor_var = G_FLOOR / (2 * sim.Δt)
        T_var = trap.equipartition_factor[axis] * (np.var(sim.traces[axis].values) - floor_var)
        print(
            f'{axis}: T_eff = {trap.T_eff[axis]:.3f} K | via Var (floor removed) = {T_var:.3f} K'
            f' | via fit = {est.temperature_via_fit:.3f} K'
            f' | Ωfit/Ω = {est.Ωfit/trap.Ωj[axis]:.4f} | Γfit/Γ = {est.Γfit/trap.Γj[axis]:.3f}'
            f' | correlation times in the run = {sim.time_length * trap.Γj[axis]:.0f}'
        )
    for lo, hi in ((0.3e6, 0.5e6), (1e6, 4.5e6)):
        band = (Sxx.freq_range > lo) & (Sxx.freq_range < hi)
        model = np.mean(theory.Sxx(2*np.pi*Sxx.freq_range[band]) + G_FLOOR)
        print(
            f'{lo/1e6}-{hi/1e6} MHz: mean Sxx / (theory + floor) = {np.mean(np.real(Sxx.values[band]))/model:.3f} (rectangular),'
            f' {np.mean(np.real(Sxx_bh.values[band]))/model:.3f} (Blackman-Harris)'
        )
    cov_th = np.trapezoid(theory.Cxy(2*np.pi*Sxy.freq_range), Sxy.freq_range)
    x, y, z = sim.x.values, sim.y.values, sim.z.values
    print(f'Cov(x, y): simulated {np.cov(x, y)[0, 1]/nm**2:.4f} nm², ∫Cxy df = {cov_th/nm**2:.4f} nm²')
    print(f'correlation coefficient of x and y: {np.corrcoef(x, y)[0, 1]:.4f}')
    print(f'rms x, y, z: {x.std()/nm:.2f}, {y.std()/nm:.2f}, {z.std()/nm:.1f} nm; z at 300 K: {z_rms(trap, np.inf)/nm:.1f} nm')


def z_rms(trap, t):
    """
    Expected rms z a time t after starting at rest, √(k_B T/mΩz² · (1 - e^(-Γz t))) (harmonic trap)

    The gas feeds energy into the motion at the rate it would damp it, Γz.
    """
    return np.sqrt(trap.gas.T / trap.equipartition_factor['z'] * (1 - np.exp(-trap.Γz * t)))


def panel_label(ax, text, **kwargs):
    ax.text(0.012, 0.95, text, transform=ax.transAxes, va='top', fontsize=14, **kwargs)


def traces_figure(sim):
    trap = sim.trap
    chunk = 1000  # samples, i.e. 0.1 ms; z has a period of 29 µs
    whole = slice(0, len(sim.x.values) // chunk * chunk)
    t = sim.x.time_array[whole]
    x, y, z = (sim.traces[axis].values[whole] / nm for axis in 'xyz')
    fig = plt.figure(figsize=(11, 5.6))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.62, 1], hspace=0.34, wspace=0.2, left=0.065, right=0.985, top=0.97, bottom=0.1)
    ax1, ax2, ax3 = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[:, 1])

    # (a) the envelopes over the whole run: the extremes of each chunk
    t_chunk = t.reshape(-1, chunk).mean(axis=1) * 1e3
    for values, axis in ((z, 'z'), (x, 'x')):
        chunks = values.reshape(-1, chunk)
        ax1.fill_between(t_chunk, chunks.min(axis=1), chunks.max(axis=1), color=AXIS_COLOURS[axis], lw=0, alpha=0.75 if axis == 'z' else 1)
    for sign in (1, -1):
        ax1.plot(t_chunk, sign * 2 * z_rms(trap, t_chunk * 1e-3) / nm, color='k', ls='--', lw=1.1)
    ax1.set_xlim(0, t_chunk[-1])
    ax1.set_ylim(-780, 980)
    ax1.text(0.07, 0.95, r'$z$ is not cooled: the gas heats it from rest towards 300 K', transform=ax1.transAxes,
             va='top', fontsize=10.5, color=AXIS_COLOURS['z'])
    ax1.text(0.07, 0.83, r'dashed: the expected $\pm 2\,z_\mathrm{rms}(t)$', transform=ax1.transAxes, va='top', fontsize=10.5)
    ax1.text(0.99, 0.05, r'$x$ on the same scale', transform=ax1.transAxes, ha='right', va='bottom', fontsize=10.5, color=AXIS_COLOURS['x'])
    ax1.set_xlabel('time (ms)')
    ax1.set_ylabel('$z$ (nm)')
    panel_label(ax1, '(a)')

    # (b) a short stretch of x and y
    start, length = len(t) // 2, 1500
    stretch = slice(start, start + length)
    t_stretch = (t[stretch] - t[start]) / μs
    ax2.plot(t_stretch, x[stretch], color=AXIS_COLOURS['x'], lw=1.5, label=f'$x$, {trap.Ωx/kHz:.0f} kHz')
    ax2.plot(t_stretch, y[stretch], color=AXIS_COLOURS['y'], lw=1.5, label=f'$y$, {trap.Ωy/kHz:.0f} kHz')
    ax2.set_xlim(0, t_stretch[-1])
    ax2.set_ylim(-11, 15)
    ax2.legend(ncol=2, loc='upper right', frameon=False, fontsize=11, handlelength=1.4, columnspacing=1.2)
    ax2.set_xlabel(f'time after {t[start]*1e3:.0f} ms (µs)')
    ax2.set_ylabel('$x$, $y$ (nm)')
    panel_label(ax2, '(b)')

    # (c) where the particle is in the x-y plane
    limit = 11.5
    ax3.hist2d(x, y, bins=150, range=[[-limit, limit]] * 2, cmap='magma', rasterized=True)
    for num_sigma in (1, 2):
        ax3.add_patch(Ellipse((0, 0), 2 * num_sigma * x.std(), 2 * num_sigma * y.std(), fill=False, color='w', ls='--', lw=1))
    reach = 0.72 * limit
    ψ = trap.ψ
    ax3.annotate('', xy=(reach * np.cos(ψ), reach * np.sin(ψ)), xytext=(-reach * np.cos(ψ), -reach * np.sin(ψ)),
                 arrowprops=dict(arrowstyle='<|-|>', color=COLOURS['orange'], lw=2))
    ax3.text(0.8 * reach * np.cos(ψ) - 0.6, 0.8 * reach * np.sin(ψ) + 0.6, 'directed\nforce', color=COLOURS['orange'],
             ha='right', va='bottom', fontsize=10.5)
    ax3.text(0.97, 0.03, f'correlation of $x$ and $y$: {np.corrcoef(x, y)[0, 1]:.3f}', transform=ax3.transAxes,
             ha='right', color='w', fontsize=10.5)
    ax3.set_aspect('equal')
    ax3.set_xlabel('$x$ (nm)')
    ax3.set_ylabel('$y$ (nm)', labelpad=-4)
    panel_label(ax3, '(c)', color='w')
    return fig


def spectra_figure(sim, spectra):
    trap, theory = sim.trap, sim.trap.theory
    Sxx, Syy, Sxy = spectra['xx'], spectra['yy'], spectra['xy']
    f = Sxx.freq_range
    sel = (f >= 100e3) & (f <= 190e3)
    f_fine = np.linspace(f[sel][0], f[sel][-1], 3000)  # the theory on a grid fine enough to draw a smooth curve
    ω_fine = 2 * np.pi * f_fine

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7.2), sharex=True, gridspec_kw=dict(height_ratios=[1.2, 1], hspace=0.05))
    for axis, est, S_theory in (('x', Sxx, theory.Sxx), ('y', Syy, theory.Syy)):
        colour = AXIS_COLOURS[axis]
        ax1.fill_between(f_fine/1e3, 1e-27, S_theory(ω_fine) + G_FLOOR, color=colour, alpha=0.12, lw=0)
        ax1.semilogy(f_fine/1e3, S_theory(ω_fine) + G_FLOOR, color=colour, lw=2)
        ax1.semilogy(f[sel]/1e3, np.real(est.values[sel]), 'o', ms=3.2, mfc='w', mec=colour, mew=0.9)
        Ω = trap.Ωj[axis] / kHz
        ax1.plot([Ω] * 2, [2.2e-20, 5.5e-20], color=colour, lw=3.5, solid_capstyle='butt')
        ax1.text(Ω + 1.2, 3.4e-20, f'$\\Omega_{axis}$', color=colour, va='center', fontsize=14)
        ax1.text(Ω + (6.5 if axis == 'x' else -7.5), 4.5e-22, f'$T_\\mathrm{{eff}}$ = {trap.T_eff[axis]:.2f} K\nfit: {est.temperature_via_fit:.2f} K',
                 color=colour, ha='left' if axis == 'x' else 'right', va='center', fontsize=10.5)
    ax1.plot([], [], 'o', ms=3.2, mfc='w', mec='k', mew=0.9, label=f'simulated: {Sxx.num_segments} segments of {Sxx.segment_time*1e3:.0f} ms')
    ax1.plot([], [], color='k', lw=2, label='theory')
    ax1.legend(loc='upper left', bbox_to_anchor=(0.045, 1.0), frameon=False, fontsize=10.5)
    ax1.set_ylim(8e-25, 8e-20)
    ax1.set_ylabel(r'$S_{xx}$, $S_{yy}$ (m$^2$/Hz, one-sided)')
    panel_label(ax1, '(a)')

    scale = 1e-23
    C_theory = theory.Cxy(ω_fine) / scale
    σ = theory.σCxy(ω_fine) / np.sqrt(Sxy.num_segments) / scale  # the theory object's nd is 1
    ax2.fill_between(f_fine/1e3, C_theory - σ, C_theory + σ, color=COLOURS['grey'], alpha=0.25, lw=0, label=r'theory $\pm 1\sigma$ of the estimate')
    ax2.fill_between(f_fine/1e3, 0, C_theory, where=C_theory > 0, color=IN_PHASE, alpha=0.45, lw=0)
    ax2.fill_between(f_fine/1e3, 0, C_theory, where=C_theory < 0, color=IN_ANTIPHASE, alpha=0.45, lw=0)
    ax2.plot(f_fine/1e3, C_theory, color=COLOURS['purple'], lw=2, label='theory')
    ax2.plot(f[sel]/1e3, np.real(Sxy.values[sel]) / scale, 'o', ms=3.2, mfc='w', mec=COLOURS['purple'], mew=0.9, label='simulated')
    ax2.axhline(0, color='k', lw=0.8)
    middle = 0.5 * (trap.Ωx + trap.Ωy) / kHz
    ax2.text(middle, -0.8, '$x$ and $y$\nin antiphase', color=IN_ANTIPHASE, ha='center', va='center', fontsize=11)
    ax2.text(trap.Ωy/kHz - 8, 2.2, '$x$ and $y$\nin phase', color=IN_PHASE, ha='center', va='center', fontsize=11)
    ax2.text(trap.Ωx/kHz + 8, 2.2, '$x$ and $y$\nin phase', color=IN_PHASE, ha='center', va='center', fontsize=11)
    ax2.legend(loc='lower right', frameon=False, fontsize=10.5)
    ax2.set_xlim(f[sel][0]/1e3, f[sel][-1]/1e3)
    ax2.set_xlabel('frequency (kHz)')
    ax2.set_ylabel(r'co-spectrum $C_{xy}$ ($10^{-23}$ m$^2$/Hz)')
    panel_label(ax2, '(b)')
    fig.subplots_adjust(left=0.1, right=0.985, top=0.985, bottom=0.085)
    return fig


def log_binned(f, values, bins_per_decade=10):
    """Geometric-mean frequency and mean of `values` in logarithmic frequency bins (empty bins dropped)"""
    edges = 10**np.arange(np.log10(f[0]), np.log10(f[-1]) + 1/bins_per_decade, 1/bins_per_decade)
    index = np.digitize(f, edges)
    filled = np.unique(index)
    return np.array([np.exp(np.mean(np.log(f[index == k]))) for k in filled]), np.array([values[index == k].mean() for k in filled])


def broadband_figure(sim, spectra):
    theory = sim.trap.theory
    Sxx, Sxx_bh = spectra['xx'], spectra['xx, Blackman-Harris']
    f = Sxx.freq_range
    wide = (f >= 1e3) & (f < sim.Nyquist_frequency)
    f = f[wide]
    motion = theory.Sxx(2 * np.pi * f)
    windows = (('rectangular window', Sxx, COLOURS['orange']), ('Blackman-Harris window', Sxx_bh, COLOURS['blue']))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), gridspec_kw=dict(width_ratios=[1.25, 1]))
    for name, est, colour in windows:
        ax1.loglog(f, np.real(est.values[wide]), color=colour, lw=0.7, alpha=0.9, label=name)
    ax1.loglog(f, motion + G_FLOOR, color='k', lw=1.6, label='theory + floor')
    ax1.loglog(f, motion, color='k', lw=1.2, ls=':', label='theory, motion only')
    ax1.axhline(G_FLOOR, color=COLOURS['grey'], ls='--', lw=1.1)
    ax1.text(1.3e3, 1.25 * G_FLOOR, 'detection floor', color=COLOURS['grey'], fontsize=10.5, va='bottom')
    ax1.annotate('leakage from\nthe resonance', xy=(5.5e5, 9e-27), xytext=(1.3e6, 2.5e-25), fontsize=10.5, color=COLOURS['orange'], ha='center',
                 arrowprops=dict(arrowstyle='-', color=COLOURS['orange'], lw=0.9, shrinkA=2, shrinkB=2))
    ax1.set_xlim(f[0], f[-1])
    ax1.set_ylim(3e-31, 2e-20)
    ax1.legend(loc='lower left', frameon=False, fontsize=10.5)
    ax1.set_xlabel('frequency (Hz)')
    ax1.set_ylabel(r'$S_{xx}$ (m$^2$/Hz, one-sided)')
    panel_label(ax1, '(a)')

    above = f >= 1e4  # below 10 kHz a logarithmic bin holds too few points of the estimate to average
    for name, est, colour in windows:
        ax2.semilogx(*log_binned(f[above], (np.real(est.values[wide]) / (motion + G_FLOOR))[above]), color=colour, lw=2.2, label=name)
    ax2.axhline(1, color='k', lw=0.9)
    ax2.axvline(sim.trap.Ωx / Hz, color=COLOURS['grey'], ls=':', lw=1.1)
    ax2.text(sim.trap.Ωx / Hz * 0.93, 0.74, r'$\Omega_x / 2\pi$', color=COLOURS['grey'], ha='right', fontsize=11)
    ax2.set_xlim(1e4, f[-1])
    ax2.set_ylim(0.7, 1.8)
    ax2.legend(loc='upper left', bbox_to_anchor=(0.09, 1), frameon=False, fontsize=10.5)
    ax2.text(0.97, 0.05, 'averaged in logarithmic bins', transform=ax2.transAxes, ha='right', fontsize=10.5)
    ax2.set_xlabel('frequency (Hz)')
    ax2.set_ylabel('estimate / (theory + floor)')
    panel_label(ax2, '(b)')
    fig.tight_layout()
    return fig


def main():
    setup_matplotlib()
    start = time.time()
    sim = simulate()
    print(f'simulated in {time.time() - start:.0f} s')
    spectra = estimate(sim)
    report(sim, spectra)
    print('written', save(traces_figure(sim), 'stochastic_traces', folder=FIGURES))
    print('written', save(spectra_figure(sim, spectra), 'stochastic_psd', folder=FIGURES))
    print('written', save(broadband_figure(sim, spectra), 'stochastic_broadband', folder=FIGURES))


if __name__ == '__main__':
    main()

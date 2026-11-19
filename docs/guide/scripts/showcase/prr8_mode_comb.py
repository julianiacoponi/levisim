"""
README showcase, PRR 8 (2026): a mechanical mode comb and its collective bright mode

J. H. Iacoponi, M. Rademacher and T. S. Monteiro, *Nanoparticle arrays levitated in a cavity for
quantum sensing*, Phys. Rev. Research 8, 033286 (2026), doi:10.1103/wrd3-t5cf

N nanoparticles, each moving in x and y, trapped at the node of one cavity mode, with their 2N
mechanical frequencies spaced evenly (6.67 kHz apart) about 300 kHz and the same light-mechanical
coupling g/2π = 40 kHz for every motion (toy parameters, as in the paper's Fig. 1); the cavity and
gas are levisim's defaults (Δ/2π = -450 kHz, κ/2π = 390 kHz, 1e-6 mbar). The cavity mixes the comb
into one collective bright mode, which takes almost all of the coupling to the light, is pulled
far below the comb and broadened, and 2N - 1 nearly dark modes left close to the bare frequencies.

    (a) heterodyne spectrum of the cavity output for N = 8, with the bare frequencies Ω_j marked
    (b) every mechanical eigenfrequency of the drift matrix against N, coloured by its linewidth
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import quiet_console, COLOURS, colormap_to_plotly, plotly_layout, rgb_hex, save, setup_matplotlib  # noqa: E402

from levisim.core.units import kHz  # noqa: E402
from levisim.multiphysics.enums import QuantumLinearTheoryMethod  # noqa: E402
from levisim.multiphysics.multitrap import MultiTrap  # noqa: E402
from levisim.multiphysics.utils.shelf import DEFAULT_cavity_kwargs, DEFAULT_gas_kwargs  # noqa: E402

NAME = 'prr8_mode_comb'
LAM = QuantumLinearTheoryMethod.LINEAR_AMPLIFIER_MODEL
SPACING = 100 / 15  # kHz: the paper's 16 frequencies over 250-350 kHz
N_VALUES = (1, 2, 3, 4, 5, 6, 7, 8)


def array(N, ω_range=np.arange(100, 400, 0.05) * kHz):
    """N particles (x and y each), 2N frequencies evenly spaced about 300 kHz, g/2π = 40 kHz"""
    return MultiTrap(
        num_traps=N,
        Cartesian_axes_per_trap=2,
        gas_kwargs=DEFAULT_gas_kwargs,
        cavity_kwargs=DEFAULT_cavity_kwargs,
        toy_params=dict(
            Ω_model='equal spacing', Ω_mid=300 * kHz, Ω_span=SPACING * (2 * N - 1) * kHz,
            toy_gx=-40 * kHz, toy_gy=-40 * kHz,
        ),
        ω_range=ω_range,
    )


def mechanical_modes(mtrap):
    """(frequency [kHz], linewidth Γ/2π [kHz]) of each mechanical normal mode, without the cavity-like one"""
    λ = mtrap.drift_eigenvalues
    modes = sorted(((abs(l.imag) / kHz, -2 * l.real / kHz) for l in λ if l.imag > 0), key=lambda m: m[1])
    return modes[:-1]  # the broadest is the cavity-like eigenvalue (Γ/2π ≈ κ/2π)


def compute(quick=False):
    """The N = 8 heterodyne spectrum, and the mechanical modes for each N"""
    quiet_console()
    mtrap = array(8, np.arange(100, 400, 0.25 if quick else 0.05) * kHz)
    assert mtrap.is_stable
    data = dict(ω_range=mtrap.ω_range, S_het=mtrap.heterodyne_spectrum(LAM),
                bare=np.array(sorted(d.Ω for d in mtrap.dimensions.values())))
    modes, bare_spans = {}, {}
    for N in ((1, 4, 8) if quick else N_VALUES):
        m = array(N, mtrap.ω_range[:1])
        assert m.is_stable, f'N = {N} is unstable'
        modes[N] = np.array(mechanical_modes(m))
        Ω = [d.Ω / kHz for d in m.dimensions.values()]
        bare_spans[N] = (min(Ω), max(Ω))
    data.update(modes=modes, bare_spans=bare_spans)
    return data


def static_figure(data):
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    setup_matplotlib()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), gridspec_kw=dict(width_ratios=[1.35, 1]))

    f = data['ω_range'] / kHz
    viridis = matplotlib.colormaps['viridis']
    bare = data['bare'] / kHz
    for k, Ω in enumerate(bare):
        ax1.axvline(Ω, ymin=0.86, ymax=0.97, color=viridis(0.05 + 0.85 * k / (len(bare) - 1)), lw=2.2, zorder=1)
    ax1.fill_between(f, 0.3, data['S_het'], color=COLOURS['blue'], alpha=0.25, lw=0, zorder=2)
    ax1.semilogy(f, data['S_het'], color=COLOURS['blue'], lw=1.3, zorder=3)
    cbm = data['modes'][8][-1][0]
    ax1.annotate('collective\nbright mode', (cbm, 4.0), (cbm - 40, 40), fontsize=11, ha='left',
                 arrowprops=dict(arrowstyle='-|>', color='k', lw=1.2))
    ax1.text(bare.mean(), 1.1e3, r'bare frequencies $\Omega_j$', ha='center', fontsize=11)
    ax1.text(bare.mean(), 0.55, 'nearly dark modes', ha='center', fontsize=11)
    ax1.set_xlim(f[0], f[-1])
    ax1.set_ylim(0.3, 1e4)
    ax1.set_xlabel(r'$\omega / 2\pi$ (kHz)')
    ax1.set_ylabel('heterodyne PSD (photons/Hz)')
    ax1.text(0.02, 0.86, r'$N = 8$ particles', transform=ax1.transAxes, fontsize=11)

    norm = LogNorm(vmin=0.3, vmax=100)
    for N, modes in data['modes'].items():
        lo, hi = data['bare_spans'][N]
        ax2.plot([N, N], [lo, hi], color='0.85', lw=9, solid_capstyle='butt', zorder=1)
        sc = ax2.scatter(np.full(len(modes), N), modes[:, 0], c=modes[:, 1], cmap='magma_r', norm=norm,
                         s=22 + 6 * np.sqrt(modes[:, 1]), edgecolors='k', linewidths=0.4, zorder=3)
    cbms = np.array([[N, modes[-1][0]] for N, modes in data['modes'].items()])
    ax2.plot(cbms[:, 0], cbms[:, 1], color=COLOURS['purple'], lw=1.4, zorder=2)
    ax2.set_xlim(0.4, max(data['modes']) + 0.6)
    ax2.set_xlabel('number of particles $N$')
    ax2.set_ylabel(r'mechanical eigenfrequency / $2\pi$ (kHz)')
    bar = fig.colorbar(sc, ax=ax2, pad=0.02)
    bar.set_label(r'linewidth $\Gamma / 2\pi$ (kHz)')
    for ax, letter in ((ax1, '(a)'), (ax2, '(b)')):
        ax.text(0.02, 0.97, letter, transform=ax.transAxes, ha='left', va='top', fontsize=13)
    fig.tight_layout()
    return fig


def interactive_figure(data):
    import matplotlib
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=1, cols=2, column_widths=[0.58, 0.42], horizontal_spacing=0.09)
    f = data['ω_range'] / kHz
    viridis = matplotlib.colormaps['viridis']
    bare = data['bare'] / kHz
    for k, Ω in enumerate(bare):
        fig.add_trace(go.Scatter(x=[Ω, Ω], y=[2e3, 8e3], mode='lines', line=dict(color=rgb_hex(viridis(0.05 + 0.85 * k / (len(bare) - 1))), width=3),
                                 showlegend=False, hovertemplate=f'bare Ω/2π = {Ω:.2f} kHz<extra></extra>'), row=1, col=1)
    fig.add_trace(go.Scatter(x=f, y=data['S_het'], name='heterodyne, N = 8', line=dict(color=COLOURS['blue'], width=1.5),
                             fill='tozeroy', fillcolor='rgba(31,79,180,0.2)',
                             hovertemplate='ω/2π = %{x:.2f} kHz<br>S = %{y:.3g} photons/Hz<extra></extra>'), row=1, col=1)
    for N, modes in data['modes'].items():
        fig.add_trace(go.Scatter(
            x=np.full(len(modes), N), y=modes[:, 0], mode='markers', showlegend=False,
            marker=dict(color=np.log10(modes[:, 1]), cmin=np.log10(0.3), cmax=2, colorscale=colormap_to_plotly('magma_r'),
                        size=7 + 1.5 * np.sqrt(modes[:, 1]), line=dict(color='black', width=0.5),
                        colorbar=dict(title='Γ/2π (kHz)', tickvals=[0, 1, 2], ticktext=['1', '10', '100']) if N == max(data['modes']) else None),
            customdata=modes[:, 1], hovertemplate=f'N = {N}<br>' + 'Ω/2π = %{y:.2f} kHz<br>Γ/2π = %{customdata:.3f} kHz<extra></extra>'), row=1, col=2)
    fig.update_yaxes(type='log', title_text='heterodyne PSD (photons/Hz)', range=[np.log10(0.3), 4], row=1, col=1)
    fig.update_xaxes(title_text='ω/2π (kHz)', range=[f[0], f[-1]], row=1, col=1)
    fig.update_xaxes(title_text='number of particles N', dtick=1, row=1, col=2)
    fig.update_yaxes(title_text='mechanical eigenfrequency/2π (kHz)', row=1, col=2)
    return plotly_layout(fig, height=480)


if __name__ == '__main__':
    import time
    start = time.time()
    data = compute()
    print(f'computed in {time.time() - start:.1f} s')
    for N, modes in data['modes'].items():
        print(f'N = {N}: bright mode {modes[-1][0]:.1f} kHz, Γ/2π = {modes[-1][1]:.2f} kHz;'
              f' narrowest {modes[0][1]*1e3:.0f} Hz; bare span {data["bare_spans"][N][0]:.1f}-{data["bare_spans"][N][1]:.1f} kHz')
    print('written', save(static_figure(data), NAME))

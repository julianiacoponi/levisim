"""
README showcase, PRR 6 (2024): sensing a directional noise bath

J. M. H. Gosling, A. Pontin, J. H. Iacoponi, P. F. Barker and T. S. Monteiro, *Sensing directional
noise baths in levitated optomechanics*, Phys. Rev. Research 6, 013129 (2024),
doi:10.1103/PhysRevResearch.6.013129

A nanosphere in a free-space tweezer, feedback-cooled in x and y, is kicked by a white-noise force
of strength β² S_th along a fixed direction ψ in the x-y plane, on top of the thermal force. The
stochastic Runge-Kutta `Simulation` applies the same random force to x (∝ cos ψ) and y (∝ sin ψ),
which correlates the two motions:

    (a) the power spectral densities S_xx and S_yy are the same for ψ = +45° and -45°; the shaded
        areas are the thermal parts alone (β = 0), so the PSDs reveal the extra force only after a
        calibrated subtraction, and never its quadrant
    (b) the co-spectrum C_xy = Re S_xy is zero without a directed force, and its sign gives the
        quadrant of ψ: no calibration is needed to see it

Points are simulated (40 averaged segments of one seeded run per angle), lines are
`SpectralDensityTheory`.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import quiet_console, COLOURS, plotly_layout, save, setup_matplotlib  # noqa: E402

from levisim.analysis.spectral_density_estimate import SpectralDensityEstimate  # noqa: E402
from levisim.core.units import degrees, kHz, μs  # noqa: E402
from levisim.data.simulation import Simulation  # noqa: E402
from levisim.physics.trap import Trap  # noqa: E402

NAME = 'prr6_directional_force'
ANGLES = (45, -45)
ANGLE_COLOURS = {45: COLOURS['red'], -45: COLOURS['teal']}
BAND = (95e3, 200e3)  # Hz, the plotted band around Ωy/2π = 135.8 kHz and Ωx/2π = 156.3 kHz


def compute(quick=False):
    """Seeded simulations at ψ = ±45° and their theory curves, one-sided, in m²/Hz"""
    quiet_console()
    num_samples, num_segments = (50_000, 10) if quick else (400_000, 40)
    data = {}
    for ψ in ANGLES:
        trap = Trap(ψ=ψ * degrees)   # β = √2: directed force twice the thermal force PSD
        sim = Simulation(trap=trap, num_samples=num_samples, Δt=0.1 * μs, seed=2024)
        estimates = {ij: SpectralDensityEstimate(sim, ij, num_segments=num_segments) for ij in ('xx', 'yy', 'xy')}
        f = estimates['xx'].freq_range
        band = (f > BAND[0]) & (f < BAND[1])
        data[ψ] = {ij: np.real(est.values[band]) for ij, est in estimates.items()}
        data[ψ]['f'] = f[band]
    f_theory = np.linspace(*BAND, 1500)
    ω = 2 * np.pi * f_theory
    data['f_theory'] = f_theory
    for ψ in ANGLES:
        theory = Trap(ψ=ψ * degrees).theory
        data[f'theory_{ψ}'] = dict(xx=theory.Sxx(ω), yy=theory.Syy(ω), xy=theory.Cxy(ω))
    thermal = Trap(β=0.0).theory   # same cooling, no directed force
    data['thermal'] = dict(xx=thermal.Sxx(ω), yy=thermal.Syy(ω))
    return data


def static_figure(data):
    import matplotlib.pyplot as plt
    setup_matplotlib()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 6.4), sharex=True, gridspec_kw=dict(height_ratios=[1.15, 1]))
    ft = data['f_theory'] / 1e3
    ax1.fill_between(ft, 1e-26, data['thermal']['xx'], color=COLOURS['blue'], alpha=0.18, lw=0, label=r'thermal only ($\beta = 0$)')
    ax1.fill_between(ft, 1e-26, data['thermal']['yy'], color=COLOURS['blue'], alpha=0.18, lw=0)
    for ψ in ANGLES:
        c = ANGLE_COLOURS[ψ]
        marker = 'o' if ψ > 0 else 's'
        f = data[ψ]['f'] / 1e3
        ax1.semilogy(f, data[ψ]['xx'] + data[ψ]['yy'], marker, ms=3.2, mfc='none', mec=c, mew=0.9,
                     label=rf'$S_{{xx}} + S_{{yy}}$, $\psi = {ψ:+d}^\circ$')
    ax1.semilogy(ft, data['theory_45']['xx'] + data['theory_45']['yy'], color='k', lw=1.4, label='theory, either sign')
    ax1.set_ylim(3e-24, 3e-20)
    ax1.set_ylabel(r'PSD (m$^2$/Hz)')
    ax1.legend(fontsize=9.5, loc='upper left', ncol=2, frameon=False)

    for ψ in ANGLES:
        c = ANGLE_COLOURS[ψ]
        f = data[ψ]['f'] / 1e3
        th = data[f'theory_{ψ}']['xy'] / 1e-23
        ax2.fill_between(ft, 0, th, color=c, alpha=0.2, lw=0)
        ax2.plot(ft, th, color=c, lw=1.8, label=rf'$\psi = {ψ:+d}^\circ$')
        ax2.plot(f, data[ψ]['xy'] / 1e-23, 'o' if ψ > 0 else 's', ms=3.2, mfc='none', mec=c, mew=0.9)
    ax2.axhline(0, color='k', lw=0.9)
    ax2.set_xlim(ft[0], ft[-1])
    ax2.set_xlabel('frequency (kHz)')
    ax2.set_ylabel(r'$C_{xy}$ (10$^{-23}$ m$^2$/Hz)')
    ax2.legend(fontsize=10, loc='upper left', frameon=False)
    for ax, letter in ((ax1, '(a)'), (ax2, '(b)')):
        ax.text(0.985, 0.96, letter, transform=ax.transAxes, ha='right', va='top', fontsize=13)
    fig.tight_layout()
    return fig


def interactive_figure(data):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.55, 0.45])
    ft = data['f_theory'] / 1e3
    for ij, label in (('xx', 'S<sub>xx</sub>'), ('yy', 'S<sub>yy</sub>')):
        fig.add_trace(go.Scatter(x=ft, y=data['thermal'][ij], name=f'{label}, thermal only', line=dict(color=COLOURS['blue'], width=1, dash='dot'),
                                 fill='tozeroy', fillcolor='rgba(31,79,180,0.15)', legendgroup='thermal'), row=1, col=1)
    for ψ in ANGLES:
        c = ANGLE_COLOURS[ψ]
        f = data[ψ]['f'] / 1e3
        for ij, label in (('xx', 'S<sub>xx</sub>'), ('yy', 'S<sub>yy</sub>')):
            fig.add_trace(go.Scatter(x=f, y=data[ψ][ij], mode='markers', name=f'{label} simulated, ψ = {ψ:+d}°', legendgroup=f'psd{ψ}',
                                     marker=dict(color=c, size=4, symbol='circle-open' if ψ > 0 else 'square-open')), row=1, col=1)
        fig.add_trace(go.Scatter(x=ft, y=data[f'theory_{ψ}']['xy'], name=f'C<sub>xy</sub> theory, ψ = {ψ:+d}°', legendgroup=f'csd{ψ}',
                                 line=dict(color=c, width=2.5)), row=2, col=1)
        fig.add_trace(go.Scatter(x=f, y=data[ψ]['xy'], mode='markers', name=f'C<sub>xy</sub> simulated, ψ = {ψ:+d}°', legendgroup=f'csd{ψ}',
                                 marker=dict(color=c, size=4, symbol='circle-open' if ψ > 0 else 'square-open')), row=2, col=1)
    for ij in ('xx', 'yy'):
        fig.add_trace(go.Scatter(x=ft, y=data['theory_45'][ij], name='PSD theory (either sign of ψ)', line=dict(color='black', width=1.5),
                                 legendgroup='theory', showlegend=ij == 'xx'), row=1, col=1)
    fig.update_yaxes(type='log', title_text='PSD (m²/Hz)', range=[np.log10(3e-25), np.log10(3e-20)], row=1, col=1)
    fig.update_yaxes(title_text='C<sub>xy</sub> (m²/Hz)', row=2, col=1)
    fig.update_xaxes(title_text='frequency (kHz)', range=[ft[0], ft[-1]], row=2, col=1)
    return plotly_layout(fig, height=650)


if __name__ == '__main__':
    import time
    start = time.time()
    data = compute()
    print(f'computed in {time.time() - start:.1f} s')
    trap = Trap()
    print(f'Ωx/2π = {trap.Ωx/kHz:.2f} kHz, Ωy/2π = {trap.Ωy/kHz:.2f} kHz, T_eff = {trap.T_eff}')
    for ψ in ANGLES:
        f, th = data['f_theory'], data[f'theory_{ψ}']['xy']
        print(f'ψ = {ψ:+d}°: C_xy theory {th.min():.2e} at {f[np.argmin(th)]/1e3:.1f} kHz, {th.max():.2e} at {f[np.argmax(th)]/1e3:.1f} kHz')
        df = data[ψ]['f'][1] - data[ψ]['f'][0]
        print(f'   ∫C_xy df over the band: theory {np.trapezoid(th, f):+.3e} m², simulated {np.sum(data[ψ]["xy"]) * df:+.3e} m²')
    print('written', save(static_figure(data), NAME))

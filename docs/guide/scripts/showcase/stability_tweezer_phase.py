"""
README showcase: linear stability of the default two-particle system as one tweezer phase is swept

Two particles (x and y each) in levisim's default physical setup and cavity, with the second
tweezer's phase φ2 swept over 0-180° (φ1 = 0). With unequal phases the cavity-mediated coupling of
the two particles gains a non-reciprocal part ∝ sin(φ2 - φ1), which can anti-damp a hybridised pair
of near-degenerate motions. The system is linearly stable only while every eigenvalue λ of the
drift matrix has Re λ < 0.

    (a) the mechanical eigenvalues λ in the complex plane for φ2 = 0-90°, coloured by φ2; the shaded
        half-plane Re λ > 0 is unstable
    (b) the largest growth rate max Re λ against φ2, with the unstable window filled

The eigenvalues at φ2 and 180° - φ2 are the same (to ~1e-8 s⁻¹), so (a) shows 0-90° only and both
panels colour φ2 folded into 0-90°.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import quiet_console, COLOURS, colormap_to_plotly, plotly_layout, save, setup_matplotlib  # noqa: E402

from levisim.core.units import degrees, kHz  # noqa: E402
from levisim.multiphysics.multitrap import MultiTrap  # noqa: E402
from levisim.multiphysics.utils.shelf import DEFAULT_cavity_kwargs, DEFAULT_gas_kwargs  # noqa: E402

NAME = 'stability_tweezer_phase'


def two_particles(φ2):
    """The README quickstart system with the second tweezer at phase φ2"""
    return MultiTrap(
        num_traps=2,
        Cartesian_axes_per_trap=2,
        gas_kwargs=DEFAULT_gas_kwargs,
        cavity_kwargs=DEFAULT_cavity_kwargs,
        φn_dict={2: φ2},
        ω_range=np.array([150 * kHz]),  # the drift matrix does not depend on the frequency grid
    )


def compute(quick=False):
    """Mechanical drift eigenvalues [s⁻¹, rad/s] and max Re λ for φ2 over 0-180°"""
    quiet_console()
    φ2 = np.arange(0, 180.0001, 2.0 if quick else 0.2) * degrees
    eigenvalues, growth = [], []
    for φ in φ2:
        mtrap = two_particles(φ)
        λ = mtrap.drift_eigenvalues
        mechanical = λ[(λ.imag > 0) & (λ.imag < 300 * kHz)]  # positive-frequency mechanical modes
        eigenvalues.append(np.sort_complex(mechanical))
        growth.append(mtrap.max_growth_rate)
    return dict(φ2=φ2, eigenvalues=np.array(eigenvalues), max_growth_rate=np.array(growth))


def unstable_window(data):
    """(first, last) φ2 [°] of the unstable window, each edge found by root-finding max Re λ = 0"""
    from scipy.optimize import brentq
    φ, g = data['φ2'], data['max_growth_rate']
    edges = np.where(np.diff(np.sign(g)) != 0)[0]
    roots = [brentq(lambda p: two_particles(p).max_growth_rate, φ[k], φ[k + 1], xtol=1e-6) / degrees for k in edges]
    return min(roots), max(roots)


def static_figure(data):
    import matplotlib
    import matplotlib.pyplot as plt
    setup_matplotlib()
    cmap = matplotlib.colormaps['viridis']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), gridspec_kw=dict(width_ratios=[1.2, 1]))
    φ = data['φ2'] / degrees
    folded = np.minimum(φ, 180 - φ)
    λ = data['eigenvalues'][φ <= 90]
    re_min, re_max = 1.05 * λ.real.min(), 1.8 * max(λ.real.max(), 1)
    ax1.axvspan(0, re_max, color=COLOURS['red'], alpha=0.10, lw=0)
    ax1.axvline(0, color='k', lw=1)
    for k in range(λ.shape[1]):
        sc = ax1.scatter(λ[:, k].real, λ[:, k].imag / kHz, c=φ[φ <= 90], cmap=cmap, vmin=0, vmax=90, s=7, lw=0)
    ax1.text(0.97, 0.04, 'unstable', transform=ax1.transAxes, ha='right', color=COLOURS['red'], fontsize=12)
    ax1.text(0.03, 0.04, 'stable', transform=ax1.transAxes, ha='left', color=COLOURS['grey'], fontsize=12)
    ax1.set_xlim(re_min, re_max)
    ax1.set_xlabel(r'Re $\lambda$ (s$^{-1}$)')
    ax1.set_ylabel(r'Im $\lambda / 2\pi$ (kHz)')
    bar = fig.colorbar(sc, ax=ax1, pad=0.02, ticks=[0, 15, 30, 45, 60, 75, 90])
    bar.set_label(r'$\phi_2$, or 180° $-$ $\phi_2$ (°)')

    g = data['max_growth_rate']
    lo, hi = unstable_window(data)
    ax2.fill_between(φ, 0, g, where=g > 0, color=COLOURS['red'], alpha=0.35, lw=0, interpolate=True)
    ax2.fill_between(φ, g, 0, where=g <= 0, color=COLOURS['blue'], alpha=0.18, lw=0, interpolate=True)
    ax2.scatter(φ, g, c=folded, cmap=cmap, vmin=0, vmax=90, s=6, lw=0, zorder=3)
    ax2.axhline(0, color='k', lw=1)
    ax2.text(147, 0.62 * g.max(), f'unstable for\n{lo:.1f}° < $\\phi_2$ < {hi:.1f}°', ha='center', fontsize=10.5, color=COLOURS['red'])
    ax2.set_xlim(0, 180)
    ax2.set_xticks([0, 30, 60, 90, 120, 150, 180])
    ax2.set_xlabel(r'$\phi_2$ (°)')
    ax2.set_ylabel(r'max Re $\lambda$ (s$^{-1}$)')
    for ax, letter in ((ax1, '(a)'), (ax2, '(b)')):
        ax.text(0.03, 0.97, letter, transform=ax.transAxes, ha='left', va='top', fontsize=13)
    fig.tight_layout()
    return fig


def interactive_figure(data):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    φ = data['φ2'] / degrees
    folded = np.minimum(φ, 180 - φ)
    λ = data['eigenvalues'][φ <= 90]
    φa = φ[φ <= 90]
    scale = colormap_to_plotly('viridis', 21)
    fig = make_subplots(rows=1, cols=2, column_widths=[0.55, 0.45], horizontal_spacing=0.12)
    re_min, re_max = 1.05 * λ.real.min(), 1.8 * max(λ.real.max(), 1)
    fig.add_vrect(x0=0, x1=re_max, fillcolor='rgba(192,57,43,0.10)', line_width=0, row=1, col=1)
    for k in range(λ.shape[1]):
        fig.add_trace(go.Scatter(
            x=λ[:, k].real, y=λ[:, k].imag / kHz, mode='markers', showlegend=False, customdata=φa,
            marker=dict(color=φa, cmin=0, cmax=90, colorscale=scale, size=5,
                        colorbar=dict(title='φ<sub>2</sub> or<br>180° − φ<sub>2</sub>', x=0.47, tickvals=[0, 30, 60, 90]) if k == 0 else None),
            hovertemplate='φ<sub>2</sub> = %{customdata:.1f}°<br>Re λ = %{x:.1f} s⁻¹<br>Im λ/2π = %{y:.3f} kHz<extra></extra>'), row=1, col=1)
    g = data['max_growth_rate']
    fig.add_trace(go.Scatter(x=φ, y=np.where(g > 0, g, 0), fill='tozeroy', fillcolor='rgba(192,57,43,0.35)', line=dict(width=0),
                             hoverinfo='skip', showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=φ, y=g, mode='markers', showlegend=False, marker=dict(color=folded, cmin=0, cmax=90, colorscale=scale, size=4),
                             hovertemplate='φ<sub>2</sub> = %{x:.1f}°<br>max Re λ = %{y:.1f} s⁻¹<extra></extra>'), row=1, col=2)
    fig.update_xaxes(title_text='Re λ (s⁻¹)', range=[re_min, re_max], row=1, col=1)
    fig.update_yaxes(title_text='Im λ/2π (kHz)', row=1, col=1)
    fig.update_xaxes(title_text='φ<sub>2</sub> (°)', range=[0, 180], dtick=30, row=1, col=2)
    fig.update_yaxes(title_text='max Re λ (s⁻¹)', row=1, col=2)
    return plotly_layout(fig, height=470)


if __name__ == '__main__':
    import time
    start = time.time()
    data = compute()
    print(f'computed in {time.time() - start:.1f} s')
    lo, hi = unstable_window(data)
    k = np.argmax(data['max_growth_rate'])
    print(f'unstable for φ2 = {lo:.2f}° to {hi:.2f}°; max Re λ = {data["max_growth_rate"][k]:+.1f} /s at {data["φ2"][k]/degrees:.1f}°')
    print(f'at φ2 = 30°: max Re λ = {data["max_growth_rate"][np.argmin(abs(data["φ2"] - 30*degrees))]:+.1f} /s')
    print('written', save(static_figure(data), NAME))

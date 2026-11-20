"""
README showcase, PRR 5 (2023): mode orientations and the cancellation point

A. Pontin, H. Fu, J. H. Iacoponi, P. F. Barker and T. S. Monteiro, *Controlling mode orientations
and frequencies in levitated cavity optomechanics*, Phys. Rev. Research 5, 013013 (2023),
doi:10.1103/PhysRevResearch.5.013013

One nanosphere moving in x and y, trapped at a phase Φ on the cavity standing wave (Φ = π/2 is the
node, Φ = 0 the antinode). Two couplings rotate its x-y normal modes in opposite directions: the
cavity dynamics, through g_x g_y ∝ sin²Φ, and the static coherent-scattering field, through the
direct coupling g_xy ∝ A0 cos Φ (levisim's `Dimension.mech_mech_coupling`). Where they cancel, the
modes keep their unperturbed orientation and the x-y cross spectrum S_xy vanishes.

    (a) Re S_xy(ω), each curve scaled to its own largest |S_xy|, as Φ moves from node to antinode
    (b) rotation of the x-like normal mode away from the tweezer x axis, over Φ and the detuning Δ,
        from the eigenvectors of the drift matrix; the black contour is the cancellation point

The figure is drawn with static_cavity_shifts=False, the model this showcase was first made with.
`MultiTrap` now also includes the static coherent-scattering shift of the mechanical frequencies (on
by default; see `MultiTrap.setup_static_cavity_shifts`). It moves the cancellation point by at most
0.001π, and with it the net frequency shift of both modes (static stiffening minus optical spring)
crosses zero at Φ = 0.257π (x) and 0.255π (y) for Δ/2π = -450 kHz, beside the rotation's 0.248π,
and at 0.315π and 0.308π for -150 kHz (rotation, with the shifts: 0.3135π; without: 0.3133π): the paper's coincidence of the two
cancellations.
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
from levisim.physics.bead import Bead  # noqa: E402

NAME = 'prr5_mode_orientations'
LAM = QuantumLinearTheoryMethod.LINEAR_AMPLIFIER_MODEL


def one_particle(Φ, Δ=DEFAULT_cavity_kwargs['Δ'], ω_range=np.arange(125, 165, 0.02) * kHz):
    """One 60.1 nm sphere in the default 1064 nm tweezer (θ = 45°), moving in x and y, at phase Φ"""
    return MultiTrap(
        num_traps=1,
        Cartesian_axes_per_trap=2,
        gas_kwargs=DEFAULT_gas_kwargs,                    # 1e-6 mbar, 300 K
        cavity_kwargs=dict(DEFAULT_cavity_kwargs, Δ=Δ),   # L = 0.96 cm, finesse 40 000
        beads=[Bead(Φ=Φ)],
        ω_range=ω_range,
        static_cavity_shifts=False,                       # reproduces the published figure without them
    )


def mode_rotations(mtrap):
    """
    (frequency [kHz], rotation [°]) of each mechanical normal mode, highest frequency first

    The rotation is the angle of the major axis of the mode's x-y displacement ellipse, measured
    from the tweezer x axis, read from the eigenvectors of the drift matrix in the basis
    (a, a†, b_x, b_x†, b_y, b_y†); x = x_zpf (b_x + b_x†), and likewise for y.
    """
    eigenvalues, eigenvectors = np.linalg.eig(mtrap.drift_matrix)
    trap = mtrap.traps[0]
    modes = []
    for λ, v in zip(eigenvalues, eigenvectors.T):
        if λ.imag >= 0:  # keep one of each conjugate pair
            continue
        u_x, u_y = trap.x_zpf * (v[2] + v[3]), trap.y_zpf * (v[4] + v[5])
        if abs(v[0])**2 + abs(v[1])**2 > abs(v[2])**2 + abs(v[3])**2 + abs(v[4])**2 + abs(v[5])**2:
            continue  # the cavity-like eigenvalue
        α = 0.5 * np.arctan2(2 * np.real(u_x * np.conj(u_y)), abs(u_x)**2 - abs(u_y)**2)
        modes.append((-λ.imag / kHz, np.degrees(α)))
    return sorted(modes, reverse=True)


def compute(quick=False):
    """S_xy at a set of trap positions, and the x-mode rotation over (Φ, Δ)"""
    quiet_console()
    Φ_curves = np.linspace(0.5, 0.05, 4 if quick else 10) * np.pi
    S_xy, ω_range = [], None
    for Φ in Φ_curves:
        mtrap = one_particle(Φ)
        assert mtrap.is_stable, f'unexpectedly unstable at Φ = {Φ/np.pi:.2f}π'
        S_xy.append(np.real(mtrap.position_cross_spectrum('x1', 'y1', LAM)))
        ω_range = mtrap.ω_range

    Φ_map = np.linspace(0.0, 0.5, 26 if quick else 101) * np.pi
    Δ_map = -np.linspace(100, 1000, 19 if quick else 91) * kHz
    rotation = np.array([[mode_rotations(one_particle(Φ, Δ, ω_range[:1]))[0][1] for Φ in Φ_map] for Δ in Δ_map])
    return dict(Φ_curves=Φ_curves, ω_range=ω_range, S_xy=np.array(S_xy), Φ_map=Φ_map, Δ_map=Δ_map, rotation=rotation)


def cancellation_phase(data, Δ):
    """Φ of the cancellation point at the detuning closest to Δ, by linear interpolation of the map"""
    row = data['rotation'][np.argmin(abs(data['Δ_map'] - Δ))]
    k = np.where(np.diff(np.sign(row)) != 0)[0][0]
    Φ0, Φ1 = data['Φ_map'][k], data['Φ_map'][k + 1]
    return Φ0 - row[k] * (Φ1 - Φ0) / (row[k + 1] - row[k])


def static_figure(data):
    import matplotlib
    import matplotlib.pyplot as plt
    setup_matplotlib()
    cmap = matplotlib.colormaps['plasma']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw=dict(width_ratios=[1, 1.15]))

    f = data['ω_range'] / kHz
    n = len(data['Φ_curves'])
    for k, (Φ, S) in enumerate(zip(data['Φ_curves'], data['S_xy'])):
        colour = cmap(0.02 + 0.76 * k / max(n - 1, 1))
        y = S / np.max(abs(S)) + 2.2 * k
        ax1.fill_between(f, 2.2 * k, y, color=colour, alpha=0.35, lw=0)
        ax1.plot(f, y, color=colour, lw=1.6)
        ax1.text(f[-1] - 0.5, 2.2 * k + 0.35, rf'$\Phi = {Φ/np.pi:.2f}\pi$', ha='right', va='bottom', fontsize=10, color=colour)
    ax1.set_xlim(f[0], f[-1])
    ax1.set_ylim(-1.3, 2.2 * (n - 1) + 1.3)
    ax1.set_yticks([])
    ax1.set_xlabel(r'$\omega / 2\pi$ (kHz)')
    ax1.set_ylabel(r'Re $S_{xy}(\omega)$, scaled, node at bottom')

    X, Y = data['Φ_map'] / np.pi, data['Δ_map'] / kHz
    lim = 8.0
    mesh = ax2.pcolormesh(X, Y, np.clip(data['rotation'], -lim, lim), cmap='RdBu_r', vmin=-lim, vmax=lim, shading='gouraud', rasterized=True)
    ax2.contour(X, Y, data['rotation'], levels=[0], colors='k', linewidths=2)
    ax2.axhline(DEFAULT_cavity_kwargs['Δ'] / kHz, color=COLOURS['grey'], ls='--', lw=1)
    Φc = cancellation_phase(data, DEFAULT_cavity_kwargs['Δ'])
    ax2.annotate('cancellation point', (Φc / np.pi, -720), (0.33, -880), fontsize=11,
                 arrowprops=dict(arrowstyle='-|>', color='k', lw=1.2), ha='left')
    ax2.set_xlabel(r'trap position $\Phi / \pi$ (antinode 0, node 0.5)')
    ax2.set_ylabel(r'$\Delta / 2\pi$ (kHz)')
    bar = fig.colorbar(mesh, ax=ax2, pad=0.02, extend='both')
    bar.set_label(r'rotation of the $x$ mode (°)')
    for ax, letter in ((ax1, '(a)'), (ax2, '(b)')):
        ax.text(0.015, 0.985, letter, transform=ax.transAxes, ha='left', va='top', fontsize=13,
                bbox=dict(fc='w', ec='none', alpha=0.8, pad=1))
    fig.tight_layout()
    return fig


def interactive_figure(data):
    import matplotlib
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    cmap = matplotlib.colormaps['plasma']
    fig = make_subplots(rows=1, cols=2, column_widths=[0.45, 0.55], horizontal_spacing=0.1,
                        subplot_titles=('S<sub>xy</sub>(ω), node to antinode', 'rotation of the x mode (°)'))
    f = data['ω_range'] / kHz
    n = len(data['Φ_curves'])
    for k, (Φ, S) in enumerate(zip(data['Φ_curves'], data['S_xy'])):
        colour = rgb_hex(cmap(0.02 + 0.76 * k / max(n - 1, 1)))
        fig.add_trace(go.Scatter(
            x=f, y=S / np.max(abs(S)), name=f'Φ = {Φ/np.pi:.2f}π', line=dict(color=colour, width=2),
            customdata=S, hovertemplate='ω/2π = %{x:.2f} kHz<br>Re S<sub>xy</sub> = %{customdata:.3e}<extra>%{fullData.name}</extra>',
            visible=True if k in (0, n // 2, n - 1) else 'legendonly'), row=1, col=1)
    fig.add_trace(go.Heatmap(
        x=data['Φ_map'] / np.pi, y=data['Δ_map'] / kHz, z=data['rotation'], zmin=-8, zmax=8, zmid=0,
        colorscale=colormap_to_plotly('RdBu_r'), colorbar=dict(title='°', x=1.01),
        hovertemplate='Φ = %{x:.3f}π<br>Δ/2π = %{y:.0f} kHz<br>rotation = %{z:.2f}°<extra></extra>', showlegend=False), row=1, col=2)
    fig.add_trace(go.Contour(
        x=data['Φ_map'] / np.pi, y=data['Δ_map'] / kHz, z=data['rotation'], contours=dict(start=0, end=0, size=1, coloring='none'),
        line=dict(color='black', width=3), showscale=False, hoverinfo='skip', name='cancellation point'), row=1, col=2)
    fig.update_xaxes(title_text='ω/2π (kHz)', row=1, col=1)
    fig.update_yaxes(title_text='Re S<sub>xy</sub>, scaled to its largest |S<sub>xy</sub>|', row=1, col=1)
    fig.update_xaxes(title_text='trap position Φ/π (antinode 0, node 0.5)', row=1, col=2)
    fig.update_yaxes(title_text='Δ/2π (kHz)', row=1, col=2)
    return plotly_layout(fig, height=500)


if __name__ == '__main__':
    import time
    start = time.time()
    data = compute()
    print(f'computed in {time.time() - start:.1f} s')
    for Δ in (-150, -300, -450, -1000):
        print(f'cancellation point at Δ/2π = {Δ} kHz: Φ = {cancellation_phase(data, Δ * kHz)/np.pi:.3f}π')
    for Φ, S in zip(data['Φ_curves'], data['S_xy']):
        print(f'Φ = {Φ/np.pi:.2f}π: S_xy min at {data["ω_range"][np.argmin(S)]/kHz:.2f} kHz, max at {data["ω_range"][np.argmax(S)]/kHz:.2f} kHz')
    print('written', save(static_figure(data), NAME))

"""
Quickstart: two particles in one cavity, and how to read the homodyne spectrum of the cavity output

    python docs/guide/scripts/quickstart_homodyne.py

Writes two figures into docs/guide/figures/ (about 10 s):

quickstart_homodyne.png (the README's first figure)
    (a) the homodyne spectrum, with the bare trap frequencies and the four normal modes marked;
        the colour under the curve mixes the colours of the four motions in proportion to what
        each contributes to the light, g² S_qq
    (b) the displacement spectrum of each motion, which shows who moves in each peak

quickstart_detuning.png
    (a) the homodyne spectrum as the cavity detuning is swept
    (b) the steady-state phonon occupancy of each motion over the same sweep
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, to_rgb

sys.path.insert(0, str(Path(__file__).resolve().parent / 'showcase'))
from common import COLOURS, quiet_console, save, setup_matplotlib  # noqa: E402

from levisim.analysis.gaussian_states import mean_phonon_numbers  # noqa: E402
from levisim.core.constants import kB, ħ  # noqa: E402
from levisim.core.units import degrees, kHz  # noqa: E402
from levisim.multiphysics.covariance import dimension_covariance, steady_state_covariance  # noqa: E402
from levisim.multiphysics.enums import QuantumLinearTheoryMethod  # noqa: E402
from levisim.multiphysics.multitrap import MultiTrap  # noqa: E402
from levisim.multiphysics.utils.shelf import DEFAULT_cavity_kwargs, DEFAULT_gas_kwargs  # noqa: E402

FIGURES = Path(__file__).resolve().parents[1] / 'figures'
LAM = QuantumLinearTheoryMethod.LINEAR_AMPLIFIER_MODEL
# one colour per motion, the same in every panel
MOTION_COLOURS = {'x1': COLOURS['blue'], 'y1': COLOURS['red'], 'x2': COLOURS['teal'], 'y2': COLOURS['orange']}
DETUNINGS = -np.arange(200, 1001, 5.0)  # Δ/2π in kHz, for the sweep


def quickstart_system(Δ=DEFAULT_cavity_kwargs['Δ'], ω_range=np.arange(125, 165, 0.02) * kHz):
    """
    Two particles, each moving in x and y, in one red-detuned cavity (Δ = -2π × 450 kHz by default)

    The second tweezer's phase is set to 30°, since with its built-in 90° (any phase difference of
    about 54-126°) this system is linearly unstable and MultiTrap logs a WARNING.
    """
    return MultiTrap(
        num_traps=2,
        Cartesian_axes_per_trap=2,
        gas_kwargs=DEFAULT_gas_kwargs,
        cavity_kwargs={**DEFAULT_cavity_kwargs, 'Δ': Δ},
        φn_dict={2: 30*degrees},
        ω_range=ω_range,  # kHz ≡ 2π × 10³ rad/s, so this grid is angular
    )


def mechanical_modes(mtrap):
    """Frequency and linewidth (FWHM), both /2π in kHz, of the mechanical normal modes, lowest frequency first"""
    λ = mtrap.drift_eigenvalues
    λ = λ[λ.imag > 0]
    λ = λ[np.argsort(-λ.real)][:len(mtrap.dimensions)]  # the cavity mode is the most damped one
    λ = λ[np.argsort(λ.imag)]
    return λ.imag / kHz, -2 * λ.real / kHz


def phonon_occupancies(mtrap):
    """Steady-state 〈b†b〉 of each motion, from the Lyapunov equation, as {label: n}"""
    labels = list(mtrap.dimensions)
    V = steady_state_covariance(mtrap)
    return dict(zip(labels, mean_phonon_numbers(dimension_covariance(mtrap, V, labels))))


def latex(label):
    """'x1' -> 'x_1'"""
    return f'{label[0]}_{label[1:]}'


def width_text(Γ):
    """A linewidth in kHz as text, in Hz below 0.2 kHz"""
    return f'{1e3 * Γ:.0f} Hz wide' if Γ < 0.2 else f'{Γ:.2g} kHz wide'


def spectrum_figure(mtrap):
    f = mtrap.ω_range / kHz
    S_hh = mtrap.homodyne_spectrum(LAM)
    _, S_qq = mtrap.all_position_power_spectra(LAM)  # in units of each motion's zero-point variance
    f_mode, Γ_mode = mechanical_modes(mtrap)
    Ω = {label: dim.Ω / kHz for label, dim in mtrap.dimensions.items()}

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9, 7.2), sharex=True, gridspec_kw=dict(height_ratios=[1.3, 1], hspace=0.05),
    )

    # (a) the colour under the curve: the motions' colours mixed by their share of g² S_qq
    floor, top = 0.6, 6e5
    share = np.array([mtrap.dimensions[label].g**2 * S_qq[label] for label in MOTION_COLOURS])
    rgb = np.einsum('jn,jc->nc', share / share.sum(axis=0), np.array([to_rgb(c) for c in MOTION_COLOURS.values()]))
    image = ax1.imshow(rgb[None], extent=(f[0], f[-1], floor, top), aspect='auto', alpha=0.55, zorder=1)
    under_the_curve = ax1.fill_between(f, floor, S_hh, color='none')
    image.set_clip_path(under_the_curve.get_paths()[0], transform=ax1.transData)
    ax1.semilogy(f, S_hh, color='k', lw=1.6, zorder=3)
    ax1.axhline(1, color=COLOURS['grey'], ls='--', lw=1, zorder=2)
    ax1.text(f[0] + 0.4, 1.1, 'shot noise', color=COLOURS['grey'], fontsize=10, va='bottom')

    # the bare trap frequencies, and the shift of each mode from them
    arrow_height = 2.6e4
    for label, colour in MOTION_COLOURS.items():
        ax1.plot([Ω[label]] * 2, [5e4, 1.6e5], color=colour, lw=3.5, solid_capstyle='butt', zorder=4)
        ax1.text(Ω[label], 2e5, f'$\\Omega_{{{latex(label)}}}$', color=colour, va='bottom', fontsize=14,
                 ha={'y1': 'right', 'y2': 'left'}.get(label, 'center'))
    for label, mode in (('y1', 0), ('y2', 1), ('x2', 2), ('x1', 3)):
        ax1.annotate('', xy=(f_mode[mode], arrow_height), xytext=(Ω[label], arrow_height), zorder=4,
                     arrowprops=dict(arrowstyle='-|>', color=MOTION_COLOURS[label], lw=1.6, shrinkA=0, shrinkB=0))
    ax1.text(0.5 * (Ω['x2'] + f_mode[3]), arrow_height, 'optical\nspring', ha='center', va='center', fontsize=10)
    ax1.text(f_mode[0] - 0.5, arrow_height, 'the cavity\nhybridises\n$y_1$ and $y_2$', ha='right', va='center', fontsize=10)

    # the four normal modes
    peak = np.interp(f_mode, f, S_hh)
    notes = (
        ('bright $y$ mode', (f[0] + 0.5, 3.5e2), 'left'),
        ('dark $y$ mode', (f_mode[1] + 1.2, 2.6e3), 'left'),
        ('$x_2$ mode', (f_mode[2] - 3.3, 6e2), 'center'),
        ('$x_1$ mode', (f_mode[3] + 1.1, 2.2e3), 'left'),
    )
    for mode, (name, position, align) in enumerate(notes):
        ax1.annotate(
            f'{name}\n{f_mode[mode]:.1f} kHz, {width_text(Γ_mode[mode])}', xy=(f_mode[mode], 0.75 * peak[mode]), xytext=position,
            ha=align, va='center', fontsize=10.5, arrowprops=dict(arrowstyle='-', color='k', lw=0.8, shrinkA=2, shrinkB=1),
        )
    ax1.set_ylim(floor, top)
    ax1.set_ylabel(r'homodyne $S_{hh}(\omega)$ (shot-noise units)')
    ax1.text(0.012, 0.95, '(a)', transform=ax1.transAxes, va='top', fontsize=14)

    # (b) who moves in each peak
    for label, colour in MOTION_COLOURS.items():
        ax2.semilogy(f, S_qq[label], color=colour, lw=1.9, label=f'${latex(label)}$')
    for axis in (ax1, ax2):
        for mode_frequency in f_mode:
            axis.axvline(mode_frequency, color='k', lw=0.6, ls=':', alpha=0.45, zorder=0)
    ax2.legend(ncol=2, loc='upper center', bbox_to_anchor=(0.46, 0.99), frameon=False, fontsize=13, columnspacing=1.5, handlelength=1.6)
    ax2.text(f_mode[0], 1.2e-2, 'both $y$ motions\nmove in both $y$ modes', ha='center', va='bottom', fontsize=10.5)
    ax2.text(f_mode[3] + 1.4, 1.2, 'each $x$ mode is\nmostly one particle', ha='left', va='center', fontsize=10.5)
    ax2.set_ylim(6e-6, 20)
    ax2.set_xlim(f[0], f[-1])
    ax2.set_xlabel(r'$\omega / 2\pi$ (kHz)')
    ax2.set_ylabel(r'displacement $S_{qq}(\omega)$ ($x_\mathrm{zpf}^2$ per rad/s)')
    ax2.text(0.012, 0.95, '(b)', transform=ax2.transAxes, va='top', fontsize=14)
    fig.subplots_adjust(left=0.1, right=0.985, top=0.985, bottom=0.085)
    return fig


def detuning_sweep(detunings=DETUNINGS, ω_range=np.arange(120, 165, 0.05) * kHz):
    """Homodyne spectra and phonon occupancies of the quickstart system for each detuning Δ/2π [kHz]"""
    spectra = np.empty((len(detunings), len(ω_range)))
    occupancies = {label: np.empty(len(detunings)) for label in MOTION_COLOURS}
    for row, Δ in enumerate(detunings):
        mtrap = quickstart_system(Δ=Δ * kHz, ω_range=ω_range)
        assert mtrap.is_stable, f'unstable at Δ/2π = {Δ} kHz'
        spectra[row] = mtrap.homodyne_spectrum(LAM)
        for label, n in phonon_occupancies(mtrap).items():
            occupancies[label][row] = n
    return dict(detunings=detunings, ω_range=ω_range, spectra=spectra, occupancies=occupancies)


def detuning_figure(sweep, mtrap):
    f = sweep['ω_range'] / kHz
    detuning = -sweep['detunings']
    here = -mtrap.cavity.Δ / kHz
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), gridspec_kw=dict(width_ratios=[1.25, 1]))

    mesh = ax1.pcolormesh(f, detuning, sweep['spectra'], norm=LogNorm(1, 1e4), cmap='magma', shading='nearest', rasterized=True)
    bar = fig.colorbar(mesh, ax=ax1, pad=0.02, extend='max')
    bar.set_label(r'homodyne $S_{hh}(\omega)$ (shot-noise units)')
    ax1.axhline(here, color='w', ls='--', lw=1.2)
    ax1.text(f[0] + 1, here + 12, 'the quickstart', color='w', fontsize=10.5, va='bottom')
    for label, colour in MOTION_COLOURS.items():
        Ω = mtrap.dimensions[label].Ω / kHz
        ax1.plot([Ω] * 2, [detuning.max() - 45, detuning.max()], color=colour, lw=3.5, solid_capstyle='butt')
    ax1.text(mtrap.dimensions['y1'].Ω / kHz - 1, detuning.max() - 22, 'bare frequencies', color='w', fontsize=10.5, va='center', ha='right')
    ax1.set_xlabel(r'$\omega / 2\pi$ (kHz)')
    ax1.set_ylabel(r'cavity detuning $-\Delta / 2\pi$ (kHz)')
    ax1.text(0.02, 0.96, '(a)', transform=ax1.transAxes, va='top', fontsize=14, color='w')

    height = 0  # of the previous label, so that the labels at the right-hand end do not overlap
    for label in sorted(MOTION_COLOURS, key=lambda label: sweep['occupancies'][label][-1]):
        n, colour = sweep['occupancies'][label], MOTION_COLOURS[label]
        ax2.semilogy(detuning, n, color=colour, lw=2.2)
        height = max(n[-1], 1.16 * height)
        ax2.text(detuning[-1] + 12, height, f'${latex(label)}$', color=colour, va='center', fontsize=13)
        ax2.plot(here, np.interp(here, detuning, n), 'o', color=colour, ms=6, mec='k', mew=0.6, zorder=3)
    ax2.axvline(here, color=COLOURS['grey'], ls='--', lw=1.2, zorder=0)
    n_y2 = sweep['occupancies']['y2']
    ax2.annotate(
        '$y_2$ holds most of the dark mode,\nwhich the cavity hardly damps', xy=(detuning[70], n_y2[70]), xytext=(0.3, 0.9),
        textcoords='axes fraction', ha='left', va='top', fontsize=10.5, arrowprops=dict(arrowstyle='-', color='k', lw=0.8, shrinkA=2, shrinkB=2),
    )
    n_bath = np.mean([mtrap.gas.T * kB / (ħ * dim.Ω) for dim in mtrap.dimensions.values()])
    exponent = int(np.floor(np.log10(n_bath)))
    ax2.text(0.97, 0.05, f'gas alone, at {mtrap.gas.T:.0f} K: $n \\approx {n_bath / 10**exponent:.0f} \\times 10^{exponent}$',
             transform=ax2.transAxes, ha='right', fontsize=10.5)
    ax2.set_xlim(detuning[0], detuning[-1] + 75)
    ax2.set_xlabel(r'cavity detuning $-\Delta / 2\pi$ (kHz)')
    ax2.set_ylabel('phonon occupancy of each motion')
    ax2.text(0.035, 0.95, '(b)', transform=ax2.transAxes, va='top', fontsize=14)
    fig.tight_layout()
    return fig


def main():
    quiet_console()
    setup_matplotlib()
    mtrap = quickstart_system()
    mtrap.info

    f_mode, Γ_mode = mechanical_modes(mtrap)
    print('normal modes: ' + ', '.join(f'{f:.2f} kHz ({1e3 * Γ:.0f} Hz wide)' for f, Γ in zip(f_mode, Γ_mode)))
    print('bare frequencies: ' + ', '.join(f'{label} {dim.Ω / kHz:.2f} kHz' for label, dim in mtrap.dimensions.items()))
    print('occupancies: ' + ', '.join(f'{label} {n:.0f}' for label, n in phonon_occupancies(mtrap).items()))
    print('written', save(spectrum_figure(mtrap), 'quickstart_homodyne', folder=FIGURES))

    sweep = detuning_sweep()
    for label, n in sweep['occupancies'].items():
        print(f'{label}: lowest occupancy {n.min():.0f} at Δ/2π = {sweep["detunings"][n.argmin()]:.0f} kHz')
    print('written', save(detuning_figure(sweep, mtrap), 'quickstart_detuning', folder=FIGURES))


if __name__ == '__main__':
    main()

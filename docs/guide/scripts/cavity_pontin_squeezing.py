"""
Docs figure for docs/guide/cavity_simulations.md: ponderomotive squeezing of the cavity output with
the parameters of Pontin et al.

System: one 100 nm silica sphere whose two transverse modes couple to a sideband-resolved cavity,
with the parameters fitted in Q. Deplano, A. Pontin, A. Ranfagni, F. Marino and F. Marin, Nat.
Commun. 16, 4215 (2025), doi:10.1038/s41467-025-59213-3, Table 1: Ω/2π = 122.17, 109.37 kHz,
g/2π = 14.13, 10.37 kHz, heating (decoherence) rates Γ/2π = 4.03, 3.05 kHz; κ/2π = 57 kHz,
Δ/2π = −111 kHz, detection efficiency η = 0.32 (all in the text and Table 1); 250 mW tweezer and
3 × 10⁻⁸ mbar (text). The same system is analysed for optical squeezing in A. Pontin, Q. Deplano,
F. Marino and F. Marin, Phys. Rev. A 114, 053704 (2026), doi:10.1103/hms3-yb2h, reproduced in
notebooks/papers/pontin2026_optical_squeezing.ipynb, whose toy MultiTrap (fitted total heating
imposed on top of the gas bath) is used here unchanged. The silica density is not stated; it only
sets x_zpf, which does not enter optical records in shot-noise units.

A CavitySimulation (exact linear model) records two homodynes at the angle θ* that minimises the
symmetrised MultiTrap homodyne spectrum, with η = 1 and η = 0.32. Their ensemble-averaged PSDs are
compared with the analytic ½[S(ω) + S(−ω)] of MultiTrap.homodyne_spectrum, mixed with vacuum as
η·S + 1 − η and multiplied by the sinc²(fΔt) response of the sample average. Seeded; about two minutes
with numba.

    python docs/guide/scripts/cavity_pontin_squeezing.py

Writes docs/guide/figures/cavity_pontin_squeezing.png and prints the numbers quoted in the guide.
"""
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'notebooks' / 'papers'))

from paper_tools import toy_multitrap  # the notebook's toy-system helper

from levisim.core.constants import π
from levisim.core.plotting import set_plt_rcParams_to_xmgrace_style
from levisim.core.units import kHz, mbar, nm, μs
from levisim.multiphysics.enums import QuantumLinearTheoryMethod
from levisim.multiphysics.multitrap import MultiTrap
from levisim.physics.bead import Bead
from levisim.physics.tweezer import Tweezer
from levisim.simulation import Homodyne

FIGURES = Path(__file__).resolve().parents[1] / 'figures'
LAM = QuantumLinearTheoryMethod.LINEAR_AMPLIFIER_MODEL

# Nat. Commun. 16, 4215 (2025), Table 1 and text
Ω_X, Ω_Y = 122.17 * kHz, 109.37 * kHz
G_X, G_Y = 14.13 * kHz, 10.37 * kHz
Γ_X, Γ_Y = 4.03 * kHz, 3.05 * kHz
κ_P, Δ_P = 57 * kHz, -111 * kHz
η_PAPER = 0.32
BEAD = Bead(r=50 * nm, Φ=π / 2)


def pontin_multitrap():
    """The toy MultiTrap of notebooks/papers/pontin2026_optical_squeezing.ipynb"""
    physical = MultiTrap(tweezers=[Tweezer(Power=0.25, λ=1064 * nm, θ=π / 4)], beads=[BEAD],
                         trap_kwargs=[{'override_Ω_dict': {'x': Ω_X, 'y': Ω_Y, 'z': 21.4 * kHz}}],
                         Cartesian_axes_per_trap=2, gas_kwargs={'P': 3e-8 * mbar, 'T': 300.0},
                         cavity_kwargs={'κ': κ_P, 'Δ': Δ_P}, photon_recoil=False, static_cavity_shifts=False)
    Γ_gas = physical.dimensions['x1'].Γ
    return toy_multitrap({'x1': Ω_X, 'y1': Ω_Y}, {'x1': G_X, 'y1': G_Y}, {'x1': Γ_X, 'y1': Γ_Y},
                         Δ=Δ_P, κ=κ_P, m=BEAD.m, Γ=Γ_gas, λ=1064 * nm)


def symmetrised_homodyne(mtrap, f, θ):
    """½[S(ω) + S(−ω)] of MultiTrap.homodyne_spectrum at cavity.θhom = θ, at f [Hz]"""
    ω = 2 * π * np.asarray(f)
    old = mtrap.ω_range, mtrap.cavity.θhom
    mtrap.ω_range, mtrap.cavity.θhom = np.concatenate([ω, -ω]), θ
    try:
        S = np.real(mtrap.homodyne_spectrum(LAM))
    finally:
        mtrap.ω_range, mtrap.cavity.θhom = old
    return (S[:len(ω)] + S[len(ω):]) / 2


def optimal_angle(mtrap, f):
    """The homodyne angle and frequency of the deepest symmetrised spectrum on a 0.25° grid"""
    θs = np.linspace(-π / 2, π / 2, 721)
    S = np.array([symmetrised_homodyne(mtrap, f, θ) for θ in θs])
    i, j = np.unravel_index(S.argmin(), S.shape)
    return θs[i], f[j], S[i, j]


def simulate(mtrap, θ, Δt, num_samples, num_trajectories, seeds, num_segments):
    """Ensemble-averaged two-sided PSDs of the two homodynes, averaged over independent seeded runs"""
    spectra = {'ideal': [], 'paper': []}
    for seed in seeds:
        sim = mtrap.simulate(num_samples=num_samples, Δt=Δt, seed=seed, num_trajectories=num_trajectories,
                             detection=[Homodyne(θ=θ, label='ideal'), Homodyne(θ=θ, η=η_PAPER, label='paper')])
        for label, collected in spectra.items():
            f, S = sim.averaged_psd(label, num_segments=num_segments)
            collected.append(S)
    return f, {label: np.mean(S, axis=0) for label, S in spectra.items()}, sim


if __name__ == '__main__':
    mtrap = pontin_multitrap()
    f_grid = np.arange(60e3, 170e3, 250.0)
    θ_opt, f_opt, S_opt = optimal_angle(mtrap, f_grid)
    print(f'optimal homodyne angle θ* = {θ_opt:.4f} rad, deepest symmetrised spectrum {S_opt:.4f} at {f_opt / 1e3:.2f} kHz '
          f'(η = 1); η = {η_PAPER}: {η_PAPER * S_opt + 1 - η_PAPER:.4f}')

    Δt = 0.1 * μs
    num_samples, num_trajectories, seeds, num_segments = 2**20, 8, range(2026, 2042), 128
    start = time.time()
    f, S, sim = simulate(mtrap, θ_opt, Δt, num_samples, num_trajectories, seeds, num_segments)
    runtime = time.time() - start
    num_periodograms = len(seeds) * num_trajectories * num_segments
    print(f'{len(seeds)} runs × {num_trajectories} trajectories × {num_samples} samples of Δt = {Δt / μs:.1f} μs in '
          f'{runtime:.1f} s ({sim.backend_used}); {num_periodograms} periodograms of {num_samples // num_segments} samples, '
          f'Δf = {f[1]:.1f} Hz')

    band = (f > 60e3) & (f < 170e3)
    fb = f[band]
    response = np.sinc(fb * Δt)**2
    S_mt = symmetrised_homodyne(mtrap, fb, θ_opt)
    theory = {'ideal': S_mt * response, 'paper': (η_PAPER * S_mt + 1 - η_PAPER) * response}
    exact = {label: sim.record_spectrum(label, fb) for label in theory}
    σ = 1 / np.sqrt(num_periodograms)          # relative scatter of one bin
    dip = (fb > f_opt - 5e3) & (fb < f_opt + 5e3)
    for label, η in (('ideal', 1.0), ('paper', η_PAPER)):
        r = S[label][band] / theory[label]
        print(f'η = {η:4.2f}: simulated minimum {S[label][band].min():.4f} at {fb[S[label][band].argmin()] / 1e3:.2f} kHz, '
              f'MultiTrap × sinc² minimum {theory[label].min():.4f}; mean ratio over 60-170 kHz {r.mean():.5f} '
              f'(±{σ / np.sqrt(band.sum()):.5f}), over ±5 kHz of the dip {r[dip].mean():.5f} (±{σ / np.sqrt(dip.sum()):.5f}); '
              f'max |MultiTrap × sinc² / exact discrete − 1| = {np.max(np.abs(theory[label] / exact[label] - 1)):.1e}')

    set_plt_rcParams_to_xmgrace_style()
    fig, (ax, ax_r) = plt.subplots(2, 1, figsize=(6.4, 5.2), sharex=True, gridspec_kw={'height_ratios': [3, 1.3]})
    width = 2                                   # bins averaged per plotted point
    nb = band.sum() // width * width
    fp = fb[:nb].reshape(-1, width).mean(axis=1) / 1e3
    colours = {'ideal': 'C3', 'paper': 'C0'}
    names = {'ideal': r'$\eta = 1$', 'paper': rf'$\eta = {η_PAPER}$'}
    for label in ('ideal', 'paper'):
        Sp = S[label][band][:nb].reshape(-1, width).mean(axis=1)
        ax.plot(fp, Sp, '.', ms=3, color=colours[label], label=f'CavitySimulation, {names[label]}')
        ax.plot(fb / 1e3, theory[label], '-', lw=1.2, color='k' if label == 'ideal' else '0.4',
                label=f'MultiTrap, {names[label]}')
        ax_r.plot(fp, Sp / theory[label][:nb].reshape(-1, width).mean(axis=1) - 1, '.', ms=3, color=colours[label])
    ax.axhline(1, color='0.6', lw=0.8, ls='--')
    ax.set_ylabel(r'$S_{\theta^*}/S_\mathrm{shot}$')
    ax.legend(fontsize=8, frameon=False, loc='upper right')
    ax.set_ylim(0.84, 1.25)
    band_σ = σ / np.sqrt(width)
    ax_r.axhspan(-2 * band_σ, 2 * band_σ, color='0.85', lw=0)
    ax_r.axhline(0, color='k', lw=0.8)
    ax_r.set_ylabel('sim./theory − 1')
    ax_r.set_xlabel('frequency [kHz]')
    ax_r.set_xlim(60, 170)
    fig.align_ylabels()
    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    out = FIGURES / 'cavity_pontin_squeezing.png'
    fig.savefig(out, dpi=150)
    print(f'wrote {out.relative_to(ROOT)}')

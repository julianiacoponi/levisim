"""
Reproduces every figure and number in docs/two_tone_entanglement/README.md

    PYTHONPATH=. python docs/two_tone_entanglement/make_figures.py [--plot-only]

Writes figures/*.png and results.json next to this file; about two minutes on a laptop.
--plot-only redraws the figures from an existing results.json.
Everything goes through levisim's public API, starting from the Deplano et al. (2025)
experimental parameters (levisim.multiphysics.two_tone.DEPLANO2025).
"""
import re
import sys
import json
import time
import pathlib

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

from levisim.core.units import kHz, μs
from levisim.analysis.gaussian_states import (
    common_phase_averaged,
    duan_criterion,
    gaussian_discord,
    hillery_zubairy_ratio,
    log_negativity,
    mutual_information,
    optimised_duan_criterion,
    partially_transposed_symplectic_eigenvalues,
    rotate_transverse_axes,
)
from levisim.multiphysics.covariance import dimension_covariance
from levisim.multiphysics.multitone import MultiToneDrive, Tone
from levisim.multiphysics.two_tone import (
    DEPLANO2025,
    analytic_log_negativity,
    analytic_optimum,
    analytic_threshold_cooperativity,
    bogoliubov_occupations,
    deplano2025_multitrap,
    quantum_cooperativity,
    two_tone_drive,
    two_tone_rwa_steady_state,
    two_tone_steady_state,
)


HERE = pathlib.Path(__file__).resolve().parent
FIGURES = HERE / 'figures'
RATIOS = np.round(np.arange(0.0, 0.951, 0.025), 4)

# one colour per entity, reused across all figures
COLOURS = {
    'floquet': '#1b1b3a',      # full model, all counter-rotating terms
    'rwa': '#3b75af',          # rotating-wave model
    'analytic': '#8c8c8c',     # ideal two-mode squeezed reservoir
    'Γ/3': '#2a9d8f',
    'Γ/10': '#e76f51',
    'experiment': '#b5179e',
}


def style():
    plt.rcParams.update({
        'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8,
        'legend.fontsize': 7, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5,
        'axes.spines.top': False, 'axes.spines.right': False,
        'xtick.direction': 'out', 'ytick.direction': 'out',
        'legend.frameon': False, 'savefig.dpi': 200, 'figure.dpi': 100,
        'axes.titlelocation': 'left',
    })


def panel_letter(ax, letter, x=-0.16):
    ax.text(x, 1.06, letter, transform=ax.transAxes, fontsize=10, fontweight='bold', va='bottom')


def log_ticks(ax, ticks, axis='x'):
    """Explicit major ticks on a log axis, and no labels on the minor ones"""
    target = ax.xaxis if axis == 'x' else ax.yaxis
    (ax.set_xticks if axis == 'x' else ax.set_yticks)(ticks)
    (ax.set_xticklabels if axis == 'x' else ax.set_yticklabels)([f'{t:g}' for t in ticks])
    target.set_minor_formatter(NullFormatter())


# ----------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------
def rwa_E_N(mtrap, ratio, **kwargs):
    stable, _, V = two_tone_rwa_steady_state(mtrap, ratio, **kwargs)
    return log_negativity(V) if stable else np.nan


def floquet_E_N(mtrap, ratio, **kwargs):
    result = two_tone_steady_state(mtrap, ratio, num_samples=8, **kwargs)
    return result.E_N if result.stable else np.nan


def best_over_ratios(mtrap, model, ratios=RATIOS, **kwargs):
    values = np.array([model(mtrap, ratio, **kwargs) for ratio in ratios])
    if np.all(np.isnan(values)):
        return 0.0, np.nan, values
    ii = int(np.nanargmax(values))
    return float(values[ii]), float(ratios[ii]), values


def threshold_Γ_scale(model, low=0.05, high=1.5, iterations=12, **mtrap_kwargs):
    """Largest heating scale (relative to the experiment) at which the best E_N is still > 0"""
    for _ in range(iterations):
        middle = np.sqrt(low * high)
        best, _, _ = best_over_ratios(deplano2025_multitrap(Γ_scale=middle, **mtrap_kwargs), model)
        low, high = (middle, high) if best > 0 else (low, middle)
    return float(np.sqrt(low * high))


def threshold_coupling_scale(model, low=1.0, high=3.0, iterations=12, **mtrap_kwargs):
    """Smallest coupling scale (relative to the experiment) at which the best E_N is > 0; None if not even at `high`"""
    if best_over_ratios(deplano2025_multitrap(coupling_scale=high, **mtrap_kwargs), model)[0] == 0:
        return None
    for _ in range(iterations):
        middle = np.sqrt(low * high)
        best, _, _ = best_over_ratios(deplano2025_multitrap(coupling_scale=middle, **mtrap_kwargs), model)
        low, high = (low, middle) if best > 0 else (middle, high)
    return float(np.sqrt(low * high))


def half_width(x, y):
    """(lower, upper) x at which y first falls below half its value at x = 0, by interpolation"""
    x, y = np.asarray(x, dtype=float), np.nan_to_num(np.asarray(y, dtype=float), nan=0.0)
    i0 = int(np.argmin(np.abs(x)))
    half = y[i0] / 2
    edges = []
    for step in (-1, 1):
        ii = i0
        while 0 <= ii + step < len(x) and y[ii + step] >= half:
            ii += step
        if not 0 <= ii + step < len(x):
            edges.append(None)
            continue
        jj = ii + step
        edges.append(float(x[ii] + (half - y[ii]) * (x[jj] - x[ii]) / (y[jj] - y[ii])))
    return edges


# ----------------------------------------------------------------------------------------------
# computations
# ----------------------------------------------------------------------------------------------
def compute():
    results = {'parameters': {k: (v / kHz if isinstance(v, float) and k != 'detection_efficiency' else v)
                              for k, v in DEPLANO2025.items()}}
    results['parameters_unit'] = 'frequencies in kHz (2π·kHz in code)'
    t0 = time.time()

    experiment = deplano2025_multitrap(balanced=True)
    results['C_experiment_balanced'] = quantum_cooperativity(experiment)
    results['C_experiment_published_couplings'] = quantum_cooperativity(deplano2025_multitrap())
    results['analytic_threshold_C'] = analytic_threshold_cooperativity()
    results['analytic_threshold_C_fixed_red_power'] = analytic_threshold_cooperativity(fixed_total_power=False)

    # theory levels against ratio at Γ/3 and Γ/10
    levels = {}
    for name, Γ_scale in [('Γ/3', 1/3), ('Γ/10', 1/10)]:
        mtrap = deplano2025_multitrap(Γ_scale=Γ_scale, balanced=True)
        C = quantum_cooperativity(mtrap)
        levels[name] = {
            'C': C,
            'analytic': [analytic_log_negativity(np.arctanh(ratio), C) for ratio in RATIOS],
            'rwa': [rwa_E_N(mtrap, ratio) for ratio in RATIOS],
            'floquet': [floquet_E_N(mtrap, ratio) for ratio in RATIOS],
        }
    results['levels'] = levels

    # best E_N against cooperativity, and the (ratio, Γ) map
    Γ_scales = np.logspace(np.log10(0.01), np.log10(1.5), 25)
    sweep = {'Γ_scale': list(Γ_scales), 'C': [], 'analytic': [], 'rwa': [], 'floquet': [], 'floquet_map': [], 'best_ratio': []}
    for Γ_scale in Γ_scales:
        mtrap = deplano2025_multitrap(Γ_scale=Γ_scale, balanced=True)
        C = quantum_cooperativity(mtrap)
        best_rwa, _, _ = best_over_ratios(mtrap, rwa_E_N)
        best_floquet, best_ratio, values = best_over_ratios(mtrap, floquet_E_N)
        sweep['C'].append(C)
        sweep['analytic'].append(analytic_optimum(C)[1])
        sweep['rwa'].append(best_rwa)
        sweep['floquet'].append(best_floquet)
        sweep['best_ratio'].append(best_ratio)
        sweep['floquet_map'].append(list(np.nan_to_num(values, nan=-1)))
    results['sweep'] = sweep

    # thresholds (as heating scale relative to experiment, and as cooperativity)
    thresholds = {}
    for name, model, kwargs in [
        ('rwa_balanced', rwa_E_N, dict(balanced=True)),
        ('floquet_balanced', floquet_E_N, dict(balanced=True)),
        ('floquet_published_couplings', floquet_E_N, dict()),
    ]:
        scale = threshold_Γ_scale(model, **kwargs)
        thresholds[name] = {'Γ_scale': scale, 'C': quantum_cooperativity(deplano2025_multitrap(Γ_scale=scale, **kwargs))}
    scale = threshold_coupling_scale(floquet_E_N, high=4.0, balanced=True)
    thresholds['floquet_balanced_coupling_scale_at_experimental_heating'] = {
        'coupling_scale': scale, 'searched_up_to': 4.0,
        'C': None if scale is None else quantum_cooperativity(deplano2025_multitrap(coupling_scale=scale, balanced=True))}

    # one tone only (G_+ = 0): the smallest partially transposed symplectic eigenvalue
    results['single_tone_ν_tilde'] = {
        f'{Γ_scale:g}': float(np.min(two_tone_steady_state(deplano2025_multitrap(Γ_scale=Γ_scale, balanced=True),
                                                           0.0, num_samples=8).ν_tildes))
        for Γ_scale in (1.0, 1/3, 1/10, 1e-3, 1e-6)
    }
    results['single_tone_ν_tilde_published_couplings'] = float(np.min(
        two_tone_steady_state(deplano2025_multitrap(), 0.0, num_samples=8).ν_tildes))
    results['published_couplings_best'] = {
        name: dict(zip(('E_N', 'ratio'), best_over_ratios(deplano2025_multitrap(Γ_scale=Γ_scale), floquet_E_N)[:2]))
        for name, Γ_scale in [('Γ/3', 1/3), ('Γ/10', 1/10)]
    }
    results['thresholds'] = thresholds
    print(f'  sweeps and thresholds done ({time.time() - t0:.0f} s)')

    # constraints at Γ/3 (and Γ/10 where useful)
    constraints = {}
    splittings = np.array([1, 2, 4, 6, 8, 10, 12.8, 16, 20, 25, 30, 40, 50, 60, 80])
    Ω_mean = (DEPLANO2025['Ωx'] + DEPLANO2025['Ωy']) / 2
    constraints['splitting_kHz'] = list(splittings)
    for name, model in [('rwa', rwa_E_N), ('floquet', floquet_E_N)]:
        constraints[f'splitting_{name}'] = [
            best_over_ratios(deplano2025_multitrap(Γ_scale=1/3, balanced=True,
                                                   Ωx=Ω_mean + s*kHz/2, Ωy=Ω_mean - s*kHz/2), model)[0]
            for s in splittings
        ]
    constraints['splitting_analytic'] = [
        analytic_optimum(quantum_cooperativity(deplano2025_multitrap(
            Γ_scale=1/3, balanced=True, Ωx=Ω_mean + s*kHz/2, Ωy=Ω_mean - s*kHz/2)))[1]
        for s in splittings
    ]

    κs = np.logspace(np.log10(10), np.log10(300), 15)
    constraints['κ_kHz'] = list(κs)
    for scale_name, Γ_scale in [('Γ/3', 1/3), ('Γ', 1.0)]:
        for name, model in [('rwa', rwa_E_N), ('floquet', floquet_E_N)]:
            constraints[f'κ_{name}_{scale_name}'] = [
                best_over_ratios(deplano2025_multitrap(Γ_scale=Γ_scale, balanced=True, κ=κ*kHz), model)[0] for κ in κs
            ]

    imbalances = np.logspace(np.log10(0.5), np.log10(2.0), 13)
    constraints['gx_over_gy'] = list(imbalances)
    for name, model in [('rwa', rwa_E_N), ('floquet', floquet_E_N)]:
        constraints[f'imbalance_{name}'] = [
            best_over_ratios(deplano2025_multitrap(Γ_scale=1/3, gx_over_gy=q), model)[0] for q in imbalances
        ]

    visibilities = np.logspace(-3, 0, 13)
    constraints['beat_visibility'] = list(visibilities)
    for scale_name, Γ_scale in [('Γ/3', 1/3), ('Γ/10', 1/10)]:
        mtrap = deplano2025_multitrap(Γ_scale=Γ_scale, balanced=True)
        _, best_ratio, _ = best_over_ratios(mtrap, floquet_E_N)
        constraints[f'beat_ratio_{scale_name}'] = best_ratio
        constraints[f'beat_{scale_name}'] = [floquet_E_N(mtrap, best_ratio, beat_visibility=v) for v in visibilities]
        stable = [two_tone_steady_state(mtrap, best_ratio, beat_visibility=v, num_samples=2).stable for v in visibilities]
        constraints[f'beat_stable_{scale_name}'] = stable

    # the tones' DIFFERENCE must equal Ω_x + Ω_y; moving both together only moves the sidebands
    mismatches = np.round(np.arange(-6.0, 6.001, 0.25), 3)
    shifts = np.arange(-40.0, 40.001, 4.0)
    constraints['difference_mismatch_kHz'] = list(mismatches)
    constraints['common_shift_kHz'] = list(shifts)
    for scale_name, Γ_scale in [('Γ/3', 1/3), ('Γ/10', 1/10)]:
        mtrap = deplano2025_multitrap(Γ_scale=Γ_scale, balanced=True)
        _, best_ratio, _ = best_over_ratios(mtrap, floquet_E_N)
        constraints[f'tone_ratio_{scale_name}'] = best_ratio
        # offset ε moves the tones apart symmetrically, so the difference is off by 2ε
        constraints[f'difference_{scale_name}'] = [floquet_E_N(mtrap, best_ratio, offset=e/2*kHz) for e in mismatches]
        constraints[f'common_{scale_name}'] = [floquet_E_N(mtrap, best_ratio, common_shift=s*kHz) for s in shifts]
        constraints[f'difference_half_width_kHz_{scale_name}'] = half_width(mismatches, constraints[f'difference_{scale_name}'])
        constraints[f'common_half_width_kHz_{scale_name}'] = half_width(shifts, constraints[f'common_{scale_name}'])
    results['constraints'] = constraints
    print(f'  constraints done ({time.time() - t0:.0f} s)')

    # transient after switching the blue tone on (fixed total power) from red-only cooling
    transient = {}
    for scale_name, Γ_scale in [('Γ/3', 1/3), ('Γ/10', 1/10)]:
        mtrap = deplano2025_multitrap(Γ_scale=Γ_scale, balanced=True)
        _, best_ratio, _ = best_over_ratios(mtrap, floquet_E_N)
        red, blue = two_tone_drive(mtrap, best_ratio)
        cooling = MultiToneDrive(mtrap, [Tone(Δ=red.Δ, amplitude=1.0, label='red')])
        V0 = cooling.periodic_steady_state(num_samples=1).covariances[0]
        drive = MultiToneDrive(mtrap, [red, blue])
        times = np.linspace(0, 250*μs, 1001)
        Vs = drive.evolve(V0, times)
        mechanical = [dimension_covariance(mtrap, V, ('x1', 'y1')) for V in Vs]
        E_N = np.array([log_negativity(V) for V in mechanical])
        transient[scale_name] = {
            'ratio': best_ratio,
            't_us': list(times / μs),
            'E_N': list(E_N),
            'n_x': [float((V[0, 0] + V[1, 1]) / 4 - 0.5) for V in mechanical],
            'steady_E_N': floquet_E_N(mtrap, best_ratio),
            'first_entangled_us': float(times[np.argmax(E_N > 0)] / μs) if np.any(E_N > 0) else None,
        }
    results['transient'] = transient
    print(f'  transient done ({time.time() - t0:.0f} s)')

    # the resolution limit: how fast the slowest mechanical combination relaxes (no heating)
    resolution = {'coupling_scale': list(np.logspace(np.log10(0.3), np.log10(4), 25)), 'splittings_kHz': [12.8, 25.0, 40.0]}
    for δ in resolution['splittings_kHz']:
        rates = []
        for coupling_scale in resolution['coupling_scale']:
            mtrap = deplano2025_multitrap(Γ_scale=1e-6, coupling_scale=coupling_scale, balanced=True,
                                          Ωx=Ω_mean + δ*kHz/2, Ωy=Ω_mean - δ*kHz/2)
            drive = MultiToneDrive(mtrap, two_tone_drive(mtrap, 0.5))
            μ = np.max(np.abs(drive.floquet_multipliers()))
            # covariances relax at twice the amplitude rate -ln|μ|/T
            rates.append(float(-2 * np.log(μ) / drive.period / kHz))
        resolution[f'rate_kHz_δ={δ}'] = rates
    results['resolution'] = resolution

    # at the published heating: best E_N against splitting and coupling
    splittings, scales = [10, 12.8, 16, 20, 25, 30, 40, 50, 60], [1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0]
    grid = [[best_over_ratios(deplano2025_multitrap(coupling_scale=s, balanced=True, Ωx=Ω_mean + δ*kHz/2,
                                                    Ωy=Ω_mean - δ*kHz/2), floquet_E_N)[0] for s in scales]
            for δ in splittings]
    results['published_heating_grid'] = {'splitting_kHz': splittings, 'coupling_scale': scales, 'E_N': grid}
    print(f'  resolution limit done ({time.time() - t0:.0f} s)')

    # measures and witnesses
    measures = {}
    for scale_name, Γ_scale in [('Γ/3', 1/3), ('Γ/10', 1/10)]:
        mtrap = deplano2025_multitrap(Γ_scale=Γ_scale, balanced=True)
        rows = {k: [] for k in ['ν_tilde', 'ν_tilde_period_mean', 'duan_fixed', 'duan_optimised', 'hz', 'discord_xy', 'discord_yx',
                                'mutual_information', 'ν_tilde_time_averaged', 'ν_tilde_phase_averaged',
                                'bogoliubov_prediction', 'rwa_E_N']}
        for ratio in RATIOS:
            result = two_tone_steady_state(mtrap, ratio, num_samples=16)
            if not result.stable:
                for k in rows:
                    rows[k].append(np.nan)
                continue
            V0 = result.mechanical_covariances[0]
            rows['ν_tilde'].append(float(result.ν_tildes[0]))
            rows['ν_tilde_period_mean'].append(float(np.mean(result.ν_tildes)))
            rows['duan_fixed'].append(duan_criterion(V0))
            rows['duan_optimised'].append(optimised_duan_criterion(V0, num_starts=6)[0])
            rows['hz'].append(hillery_zubairy_ratio(V0))
            rows['discord_xy'].append(gaussian_discord(V0, measured_mode=1))
            rows['discord_yx'].append(gaussian_discord(V0, measured_mode=0))
            rows['mutual_information'].append(mutual_information(V0))
            V_time_averaged = np.mean(result.mechanical_covariances, axis=0)
            rows['ν_tilde_time_averaged'].append(float(partially_transposed_symplectic_eigenvalues(V_time_averaged).min()))
            rows['ν_tilde_phase_averaged'].append(float(partially_transposed_symplectic_eigenvalues(common_phase_averaged(V0)).min()))
            # what treating the Bogoliubov modes as independent thermal modes would predict
            stable, _, V_rwa = two_tone_rwa_steady_state(mtrap, ratio)
            n_β = np.mean(bogoliubov_occupations(V_rwa, mtrap, ratio))
            r = np.arctanh(ratio)
            rows['bogoliubov_prediction'].append(float(max(0.0, 2*r - np.log(2*n_β + 1))))
            rows['rwa_E_N'].append(log_negativity(V_rwa))
        measures[scale_name] = rows

    # sufficient witnesses against PPT, with and without a residual trap beat (local squeezing)
    mtrap = deplano2025_multitrap(Γ_scale=1/10, balanced=True)
    witnesses = {}
    for case, visibility in [('no beat', 0.0), ('beat 0.1', 0.1)]:
        rows = {'ν_tilde': [], 'hz': [], 'duan_optimised': []}
        for ratio in RATIOS:
            result = two_tone_steady_state(mtrap, ratio, beat_visibility=visibility, num_samples=1)
            if not result.stable:
                for k in rows:
                    rows[k].append(np.nan)
                continue
            V0 = result.mechanical_covariances[0]
            rows['ν_tilde'].append(float(partially_transposed_symplectic_eigenvalues(V0).min()))
            rows['hz'].append(hillery_zubairy_ratio(V0))
            rows['duan_optimised'].append(optimised_duan_criterion(V0, num_starts=6)[0])
        witnesses[case] = rows
    measures['witnesses'] = witnesses

    # within one period, at Γ/10 and the best ratio
    mtrap = deplano2025_multitrap(Γ_scale=1/10, balanced=True)
    _, best_ratio, _ = best_over_ratios(mtrap, floquet_E_N)
    result = two_tone_steady_state(mtrap, best_ratio, num_samples=96)
    measures['period'] = {
        'ratio': best_ratio,
        't_over_T': list(result.times / (2*np.pi / result.ν)),
        'ν_tilde': list(result.ν_tildes),
        'duan_fixed': [duan_criterion(V) for V in result.mechanical_covariances],
        'hz': [hillery_zubairy_ratio(V) for V in result.mechanical_covariances],
    }
    # partition: the two-tone state and the trap ground state along rotated axes
    φs = np.linspace(0, np.pi/2, 91)
    V_state = result.mechanical_covariances[0]
    Ωx, Ωy = mtrap.dimensions['x1'].Ω, mtrap.dimensions['y1'].Ω
    measures['axes'] = {
        'φ_deg': list(np.degrees(φs)),
        'two_tone': [log_negativity(rotate_transverse_axes(V_state, Ωx, Ωy, φ)) for φ in φs],
        'ground_state': [log_negativity(rotate_transverse_axes(np.eye(4), Ωx, Ωy, φ)) for φ in φs],
    }
    results['measures'] = measures
    results['runtime_s'] = time.time() - t0
    print(f'  measures done ({time.time() - t0:.0f} s)')
    return results


# ----------------------------------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------------------------------
def figure_mechanism(results):
    fig = plt.figure(figsize=(7.2, 8.4))
    grid = fig.add_gridspec(3, 2, left=0.08, right=0.975, top=0.955, bottom=0.06, wspace=0.34, hspace=0.62)
    p = results['parameters']
    Ω_mean, δ_half, κ = (p['Ωx'] + p['Ωy']) / 2, (p['Ωx'] - p['Ωy']) / 2, p['κ']

    # (a) frequency bookkeeping on a broken frequency axis: tones far off resonance, sidebands on it
    segments = grid[0, 0].subgridspec(1, 3, width_ratios=[1, 2.4, 1], wspace=0.08)
    windows = [(-Ω_mean - 12, -Ω_mean + 12), (-15, 15), (Ω_mean - 12, Ω_mean + 12)]
    sub = [fig.add_subplot(segments[0, ii]) for ii in range(3)]
    for ii, (ax, (lo, hi)) in enumerate(zip(sub, windows)):
        ω = np.linspace(lo, hi, 400)
        ax.fill_between(ω, 0, 1 / (1 + (2*ω/κ)**2), color='0.88', lw=0)
        ax.set_xlim(lo, hi); ax.set_ylim(0, 1.3)
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)
        # break marks between segments
        for side in ([1] if ii == 0 else [0] if ii == 2 else [0, 1]):
            ax.plot([side - 0.04, side + 0.04], [-0.05, 0.05], transform=ax.transAxes, color='k', lw=0.8, clip_on=False)
    for ax, x, colour, name in [(sub[0], -Ω_mean, '#c1121f', 'red tone'), (sub[2], Ω_mean, '#0353a4', 'blue tone')]:
        ax.annotate('', xy=(x, 1.0), xytext=(x, 0), arrowprops=dict(arrowstyle='-|>', color=colour, lw=1.4))
        ax.text(x, 1.06, name, ha='center', fontsize=6.5, color=colour)
    sub[0].set_xticks([-Ω_mean]); sub[0].set_xticklabels([r'$-\bar{\Omega}$'])
    sub[2].set_xticks([Ω_mean]); sub[2].set_xticklabels([r'$+\bar{\Omega}$'])
    ax = sub[1]
    ax.set_xticks([-δ_half, 0, δ_half]); ax.set_xticklabels(['−δ/2', '0', '+δ/2'])
    for x, red_label, blue_label, mode in [(δ_half, r'$a^\dagger b_x$', r'$a^\dagger b_y^\dagger$', r'$\beta_1$'),
                                           (-δ_half, r'$a^\dagger b_y$', r'$a^\dagger b_x^\dagger$', r'$\beta_2$')]:
        ax.plot([x], [0.95], 'o', color='#c1121f', ms=4)
        ax.plot([x], [0.62], 's', color='#0353a4', ms=4, mfc='white', mew=1.1)
        ha, dx = ('left', 1.2) if x > 0 else ('right', -1.2)
        ax.text(x + dx, 0.95, red_label, color='#c1121f', fontsize=7, ha=ha, va='center')
        ax.text(x + dx, 0.62, blue_label, color='#0353a4', fontsize=7, ha=ha, va='center')
        ax.text(x, 0.3, mode, fontsize=7.5, ha='center', va='center')
    ax.text(0, 1.17, 'cavity line (κ/2π = 57 kHz)', ha='center', fontsize=6.5, color='0.35')
    ax.set_xlabel('(ω − ω$_c$)/2π, with δ = Ω$_x$ − Ω$_y$')
    sub[0].set_title('Red anti-Stokes and blue Stokes photons coincide', loc='left')
    panel_letter(sub[0], 'a', x=-0.55)

    # (b) three levels of theory against the tone ratio
    ax = fig.add_subplot(grid[0, 1])
    for scale_name in ['Γ/3', 'Γ/10']:
        level = results['levels'][scale_name]
        ax.plot(RATIOS, level['floquet'], color=COLOURS[scale_name], lw=1.6)
        ax.plot(RATIOS, level['rwa'], color=COLOURS[scale_name], lw=1.1, ls=':')
        ax.plot(RATIOS, level['analytic'], color=COLOURS[scale_name], lw=1.0, ls='--')
        x, y = (0.05, 0.78) if scale_name == 'Γ/10' else (0.45, 0.40)
        ax.text(x, y, f'heating {scale_name}, C = {level["C"]:.1f}', ha='left', va='bottom', fontsize=6.5,
                color=COLOURS[scale_name])
    ax.plot([], [], color='k', lw=1.6, label='full (Floquet)')
    ax.plot([], [], color='k', lw=1.1, ls=':', label='rotating-wave')
    ax.plot([], [], color='k', lw=1.0, ls='--', label='ideal reservoir')
    ax.legend(loc='upper left', handlelength=2.2)
    ax.set_xlabel('blue/red amplitude G$_+$/G$_-$ = tanh r'); ax.set_ylabel('steady-state $E_N$')
    ax.set_xlim(0, 0.95); ax.set_ylim(-0.02, 1.25)
    ax.set_title(results['figure_titles']['levels'])
    panel_letter(ax, 'b')

    # (c) best E_N against cooperativity
    ax = fig.add_subplot(grid[1, 0])
    sweep = results['sweep']
    C = np.array(sweep['C'])
    ax.semilogx(C, sweep['floquet'], color=COLOURS['floquet'], lw=1.6, label='full (Floquet)')
    ax.semilogx(C, sweep['rwa'], color=COLOURS['rwa'], lw=1.1, ls=':', label='rotating-wave')
    ax.semilogx(C, sweep['analytic'], color=COLOURS['analytic'], lw=1.0, ls='--', label='ideal reservoir')
    ax.axvline(results['C_experiment_balanced'], color=COLOURS['experiment'], lw=0.9)
    ax.text(results['C_experiment_balanced'] * 1.08, 1.45, 'Deplano 2025', fontsize=6.5, color=COLOURS['experiment'],
            va='top', ha='left')
    ax.legend(loc='upper left', bbox_to_anchor=(0.0, 0.88))
    ax.set_xlabel('quantum cooperativity C = 4G²L/(κΓn)'); ax.set_ylabel('best steady-state $E_N$')
    ax.set_ylim(-0.02, 1.5)
    ax.set_title(results['figure_titles']['sweep'])
    panel_letter(ax, 'c')

    # (d) map over ratio and heating
    ax = fig.add_subplot(grid[1, 1])
    E_map = np.array(sweep['floquet_map'])
    Γ_scales = np.array(sweep['Γ_scale'])
    category = np.where(E_map < 0, 0.0, 1.0)  # 0: unstable, 1: stable
    ax.pcolormesh(RATIOS, Γ_scales, category, shading='nearest', cmap=matplotlib.colors.ListedColormap(['white', '0.85']),
                  vmin=0, vmax=1)
    entangled = np.ma.masked_where(E_map <= 0, E_map)
    mesh = ax.pcolormesh(RATIOS, Γ_scales, entangled, shading='nearest', cmap='viridis', vmin=0, vmax=float(E_map.max()))
    ax.contour(RATIOS, Γ_scales, np.where(E_map > 0, 1.0, 0.0), levels=[0.5], colors='k', linewidths=0.8)
    ax.set_yscale('log')
    ax.axhline(1.0, color=COLOURS['experiment'], lw=0.9)
    ax.text(0.03, 1.1, 'experiment', fontsize=6.5, color=COLOURS['experiment'], va='bottom')
    ax.text(0.03, 0.6, 'separable', fontsize=6.5, color='0.3', va='top')
    ax.set_xlabel('blue/red amplitude G$_+$/G$_-$'); ax.set_ylabel('heating rate / experiment')
    colourbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    colourbar.set_label('steady-state $E_N$')
    ax.set_title(results['figure_titles']['map'])
    panel_letter(ax, 'd')

    # (e) the Bogoliubov modes must stay resolved: the slowest relaxation rate peaks, then falls
    ax = fig.add_subplot(grid[2, 0])
    resolution = results['resolution']
    for δ, colour in zip(resolution['splittings_kHz'], ['#1b1b3a', '#2a9d8f', '#e9c46a']):
        ax.semilogx(resolution['coupling_scale'], resolution[f'rate_kHz_δ={δ}'], color=colour, lw=1.5,
                    label=f'δ/2π = {δ:g} kHz' + (' (experiment)' if δ == 12.8 else ''))
    heating = (DEPLANO2025['Γx_decoherence'] + DEPLANO2025['Γy_decoherence']) / 2 / kHz
    ax.axhline(heating, color=COLOURS['experiment'], lw=0.9, ls='--')
    ax.text(0.32, heating * 0.92, r'heating $\Gamma\bar{n}$', fontsize=6.5, color=COLOURS['experiment'], va='top')
    ax.axvline(1.0, color='0.6', lw=0.7)
    ax.legend(loc='upper left', fontsize=6.5)
    log_ticks(ax, [0.3, 0.5, 1, 2, 4])
    ax.set_xlabel('coupling G / G$_{experiment}$'); ax.set_ylabel('slowest relaxation rate /2π (kHz)')
    ax.set_title(results['figure_titles']['resolution'])
    panel_letter(ax, 'e')

    # (f) at the published heating, only a larger splitting AND a larger coupling help
    ax = fig.add_subplot(grid[2, 1])
    published = results['published_heating_grid']
    E_grid = np.array(published['E_N'], dtype=float)
    x_edges = np.arange(len(published['coupling_scale']) + 1) - 0.5
    y_edges = np.arange(len(published['splitting_kHz']) + 1) - 0.5
    mesh = ax.pcolormesh(x_edges, y_edges, np.ma.masked_where(E_grid <= 0, E_grid), cmap='viridis', vmin=0,
                         vmax=max(float(np.nanmax(E_grid)), 1e-3))
    ax.pcolormesh(x_edges, y_edges, np.ma.masked_where(E_grid > 0, np.ones_like(E_grid)),
                  cmap=matplotlib.colors.ListedColormap(['0.85']), vmin=0, vmax=1)
    ax.set_xticks(range(len(published['coupling_scale'])))
    ax.set_xticklabels([f'{s:g}' for s in published['coupling_scale']])
    ax.set_yticks(range(len(published['splitting_kHz'])))
    ax.set_yticklabels([f'{s:g}' for s in published['splitting_kHz']])
    ii = published['splitting_kHz'].index(12.8)
    ax.plot([0], [ii], marker='*', color=COLOURS['experiment'], ms=9, mec='white', mew=0.6)
    ax.set_xlabel('coupling G / G$_{experiment}$'); ax.set_ylabel('splitting (Ω$_x$ − Ω$_y$)/2π (kHz)')
    colourbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    colourbar.set_label('best $E_N$ (grey: separable)')
    ax.set_title(results['figure_titles']['published_grid'])
    panel_letter(ax, 'f')
    fig.savefig(FIGURES / 'mechanism.png')
    plt.close(fig)


def figure_constraints(results):
    c = results['constraints']
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.9))
    plt.subplots_adjust(left=0.08, right=0.985, top=0.93, bottom=0.1, wspace=0.42, hspace=0.62)

    ax = axes[0, 0]
    ax.plot(c['splitting_kHz'], c['splitting_floquet'], color=COLOURS['floquet'], lw=1.6, marker='o', ms=2.5)
    ax.plot(c['splitting_kHz'], c['splitting_rwa'], color=COLOURS['rwa'], lw=1.1, ls=':')
    ax.plot(c['splitting_kHz'], c['splitting_analytic'], color=COLOURS['analytic'], lw=1.0, ls='--')
    ax.axvline(12.8, color=COLOURS['experiment'], lw=0.9)
    ax.set_xscale('log')
    ax.set_xlabel('(Ω$_x$ − Ω$_y$)/2π (kHz)'); ax.set_ylabel('best $E_N$ (heating Γ/3)')
    log_ticks(ax, [1, 3, 10, 30, 80])
    ax.set_title('Splitting: dark mode vs. filter')
    panel_letter(ax, 'a')

    ax = axes[0, 1]
    for scale_name, colour in [('Γ/3', COLOURS['Γ/3']), ('Γ', COLOURS['experiment'])]:
        ax.plot(c['κ_kHz'], c[f'κ_floquet_{scale_name}'], color=colour, lw=1.6)
        ax.plot(c['κ_kHz'], c[f'κ_rwa_{scale_name}'], color=colour, lw=1.1, ls=':')
        if scale_name == 'Γ':
            ax.text(11, 0.03, 'heating Γ: separable for every κ', color=colour, fontsize=6.5)
        else:
            ax.text(95, 0.25, 'heating ' + scale_name, color=colour, fontsize=6.5)
    ax.axvline(57, color=COLOURS['experiment'], lw=0.9)
    ax.set_xscale('log')
    log_ticks(ax, [10, 30, 100, 300])
    ax.set_xlabel('κ/2π (kHz), couplings fixed'); ax.set_ylabel('best $E_N$')
    ax.set_title(results['figure_titles']['κ'])
    panel_letter(ax, 'b')

    ax = axes[0, 2]
    ax.plot(c['gx_over_gy'], c['imbalance_floquet'], color=COLOURS['floquet'], lw=1.6, marker='o', ms=2.5)
    ax.plot(c['gx_over_gy'], c['imbalance_rwa'], color=COLOURS['rwa'], lw=1.1, ls=':')
    ax.axvline(DEPLANO2025['gx'] / DEPLANO2025['gy'], color=COLOURS['experiment'], lw=0.9)
    ax.set_xscale('log')
    log_ticks(ax, [0.5, 0.7, 1, 1.4, 2])
    ax.set_xlabel('g$_x$/g$_y$ at fixed RMS coupling'); ax.set_ylabel('best $E_N$ (heating Γ/3)')
    ax.set_title('Couplings must be near balance')
    panel_letter(ax, 'c')

    ax = axes[1, 0]
    for scale_name in ['Γ/3', 'Γ/10']:
        values = np.array(c[f'beat_{scale_name}'], dtype=float)
        ax.semilogx(c['beat_visibility'], np.nan_to_num(values, nan=0), color=COLOURS[scale_name], lw=1.6, marker='o', ms=2.5)
        unstable = ~np.array(c[f'beat_stable_{scale_name}'])
        if unstable.any():
            first = np.array(c['beat_visibility'])[unstable][0]
            ax.axvline(first, color=COLOURS[scale_name], lw=0.8, ls='--')
        ax.text(1.2e-3, values[0] + 0.03, 'heating ' + scale_name, color=COLOURS[scale_name], fontsize=6.5)
    ax.set_xlabel('fraction of the tones\u2019 intensity beat in the trap'); ax.set_ylabel('$E_N$ at best G$_+$/G$_-$')
    ax.set_title('The trap must not see the beat')
    panel_letter(ax, 'd')

    ax = axes[1, 1]
    for scale_name in ['Γ/3', 'Γ/10']:
        ax.plot(c['difference_mismatch_kHz'], c[f'difference_{scale_name}'], color=COLOURS[scale_name], lw=1.6)
        ax.plot(c['common_shift_kHz'], c[f'common_{scale_name}'], color=COLOURS[scale_name], lw=1.1, ls='--')
    ax.plot([], [], color='k', lw=1.6, label='difference − (Ω$_x$+Ω$_y$)')
    ax.plot([], [], color='k', lw=1.1, ls='--', label='both tones shifted')
    ax.legend(loc='upper center', fontsize=6.5, ncol=1)
    ax.set_xlim(-40, 40)
    ax.set_ylim(0, 1.45 * np.nanmax(np.array(c['difference_Γ/10'], dtype=float)))
    ax.set_xlabel('tone frequency error (kHz)'); ax.set_ylabel('$E_N$ at best G$_+$/G$_-$')
    ax.set_title(results['figure_titles']['tones'])
    panel_letter(ax, 'e')

    ax = axes[1, 2]
    for scale_name in ['Γ/3', 'Γ/10']:
        transient = results['transient'][scale_name]
        ax.plot(transient['t_us'], transient['E_N'], color=COLOURS[scale_name], lw=1.0)
        ax.axhline(transient['steady_E_N'], color=COLOURS[scale_name], lw=0.7, ls='--')
        if scale_name == 'Γ/10':
            ax.text(245, transient['steady_E_N'] - 0.06, 'heating ' + scale_name, color=COLOURS[scale_name], fontsize=6.5,
                    ha='right', va='top')
        else:
            ax.text(245, transient['steady_E_N'] + 0.06, 'heating ' + scale_name, color=COLOURS[scale_name], fontsize=6.5,
                    ha='right', va='bottom')
    ax.set_xlabel('time after blue tone on (μs)'); ax.set_ylabel('$E_N$')
    ax.set_xlim(0, 250)
    ax.set_title(results['figure_titles']['transient'])
    panel_letter(ax, 'f')

    for ax in axes.flat:
        ax.margins(x=0.03)
    axes[0, 0].plot([], [], color=COLOURS['floquet'], lw=1.6, label='full (Floquet)')
    axes[0, 0].plot([], [], color=COLOURS['rwa'], lw=1.1, ls=':', label='rotating-wave')
    axes[0, 0].plot([], [], color=COLOURS['analytic'], lw=1.0, ls='--', label='ideal reservoir')
    axes[0, 0].legend(loc='upper left', bbox_to_anchor=(0.0, 0.88), fontsize=6, handlelength=1.2)
    fig.savefig(FIGURES / 'constraints.png')
    plt.close(fig)


def figure_measures(results):
    m = results['measures']
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.9))
    plt.subplots_adjust(left=0.08, right=0.985, top=0.93, bottom=0.1, wspace=0.42, hspace=0.62)
    witness_colours = {'PPT (E_N)': COLOURS['floquet'], 'Duan, optimised': '#3b75af',
                       'Duan, lab quadratures': '#e9c46a', 'Hillery–Zubairy': '#9d4edd'}

    ax = axes[0, 0]
    case_colours = {'no beat': COLOURS['floquet'], 'beat 0.1': '#e76f51'}
    for case, rows in m['witnesses'].items():
        ppt = -np.log(np.array(rows['ν_tilde'], dtype=float))
        hz = np.log(np.maximum(np.array(rows['hz'], dtype=float), 1e-12))
        ax.plot(RATIOS, ppt, color=case_colours[case], lw=1.6)
        ax.plot(RATIOS, hz, color=case_colours[case], lw=1.1, ls='--')
        if case == 'beat 0.1':
            missed = (ppt > 0) & ~(hz > 0)
            for ratio in RATIOS[missed]:
                ax.axvspan(ratio - 0.0125, ratio + 0.0125, color=case_colours[case], alpha=0.12, lw=0)
    ax.axhline(0, color='0.4', lw=0.7)
    ax.plot([], [], color='k', lw=1.6, label='PPT: −ln ν̃$_-$')
    ax.plot([], [], color='k', lw=1.1, ls='--', label='Hillery–Zubairy ≡ optimised Duan')
    ax.plot([], [], color=case_colours['no beat'], lw=4, label='beat-free trap')
    ax.plot([], [], color=case_colours['beat 0.1'], lw=4, label='10% trap beat')
    ax.legend(loc='upper left', fontsize=6, ncol=1, bbox_to_anchor=(0, 1.03), handlelength=1.8)
    ax.set_xlabel('G$_+$/G$_-$ (heating Γ/10)'); ax.set_ylabel('evidence (> 0 certifies entanglement)')
    ax.set_ylim(-1.2, 2.7)
    ax.set_title(results['figure_titles']['witnesses'])
    panel_letter(ax, 'a')

    ax = axes[0, 1]
    period = m['period']
    ax.plot(period['t_over_T'], -np.log(period['ν_tilde']), color=witness_colours['PPT (E_N)'], lw=1.4, label='PPT')
    ax.plot(period['t_over_T'], -np.log(period['duan_fixed']), color=witness_colours['Duan, lab quadratures'], lw=1.4,
            label='Duan, lab quadratures')
    ax.plot(period['t_over_T'], np.log(period['hz']), color=witness_colours['Hillery–Zubairy'], lw=1.4, label='Hillery–Zubairy')
    ax.axhline(0, color='0.4', lw=0.7)
    ax.set_ylim(-1.9, 1.9)
    ax.legend(loc='upper center', fontsize=6)
    ax.set_xlabel('time within one drive period, t/T'); ax.set_ylabel('evidence: −ln(criterion ratio)')
    ax.set_title(results['figure_titles']['period'])
    panel_letter(ax, 'b')

    ax = axes[0, 2]
    for scale_name in ['Γ/3', 'Γ/10']:
        rows = m[scale_name]
        E_N = np.maximum(0, -np.log(np.array(rows['ν_tilde'], dtype=float)))
        ax.plot(E_N, rows['discord_xy'], 'o', ms=2.5, color=COLOURS[scale_name], label=f'heating {scale_name}')
    ax.axhspan(0.0423, 0.0471, color=COLOURS['experiment'], alpha=0.25, lw=0)
    ax.text(0.98, 0.13, 'Deplano 2025\n(separable)', fontsize=6, color=COLOURS['experiment'],
            transform=ax.transAxes, ha='right', va='bottom')
    ax.legend(loc='upper left', fontsize=6.5)
    ax.set_xlabel('$E_N$ (0: separable)'); ax.set_ylabel('Gaussian discord D(x|y) (nats)')
    ax.set_title(results['figure_titles']['discord'])
    panel_letter(ax, 'c')

    ax = axes[1, 0]
    rows = m['Γ/10']
    ax.plot(RATIOS, rows['ν_tilde'], color=COLOURS['floquet'], lw=1.6, label='instantaneous')
    ax.plot(RATIOS, rows['ν_tilde_time_averaged'], color='#e76f51', lw=1.2, label='time-averaged')
    ax.plot(RATIOS, rows['ν_tilde_phase_averaged'], color='#e9c46a', lw=1.2, ls='--', label='phase-averaged')
    ax.set_yscale('log')
    ax.axhspan(0.3, 1, color='0.93', lw=0, label='entangled (ν̃$_-$ < 1)')
    ax.set_ylim(0.3, 30)
    log_ticks(ax, [0.3, 1, 3, 10, 30], axis='y')
    ax.legend(loc='upper left', fontsize=6.5)
    ax.set_xlabel('G$_+$/G$_-$ (heating Γ/10)'); ax.set_ylabel('PT eigenvalue ν̃$_-$')
    ax.set_title('Needs a phase reference')
    panel_letter(ax, 'd')

    ax = axes[1, 1]
    axes_data = m['axes']
    ax.plot(axes_data['φ_deg'], axes_data['two_tone'], color=COLOURS['floquet'], lw=1.6, label='two-tone state')
    ax.plot(axes_data['φ_deg'], axes_data['ground_state'], color='#8c8c8c', lw=1.2, label='trap ground state')
    ax.set_ylim(-0.03, 1.1)
    ax.legend(loc='upper center', fontsize=6.5)
    ax.set_xlabel('measurement axes rotated by φ (deg)'); ax.set_ylabel('$E_N$')
    ax.set_title('Rotated axes change the verdict')
    panel_letter(ax, 'e')

    ax = axes[1, 2]
    for scale_name in ['Γ/3', 'Γ/10']:
        rows = m[scale_name]
        ax.plot(RATIOS, rows['rwa_E_N'], color=COLOURS[scale_name], lw=1.6)
        ax.plot(RATIOS, rows['bogoliubov_prediction'], color=COLOURS[scale_name], lw=1.1, ls='--')
        ax.text(0.05, np.nanmax(rows['bogoliubov_prediction']) + 0.02, 'heating ' + scale_name, color=COLOURS[scale_name], fontsize=6.5)
    ax.plot([], [], color='k', lw=1.6, label='actual $E_N$ (RWA)')
    ax.plot([], [], color='k', lw=1.1, ls='--', label=r'from $\langle\beta^\dagger\beta\rangle$ alone')
    ax.legend(loc='upper left', fontsize=6.5, bbox_to_anchor=(0, 0.88))
    ax.set_xlabel('G$_+$/G$_-$'); ax.set_ylabel('$E_N$')
    ax.set_title(r'$\langle\beta^\dagger\beta\rangle$ alone overstates $E_N$')
    panel_letter(ax, 'f')
    fig.savefig(FIGURES / 'measures.png')
    plt.close(fig)


def make_titles(results):
    """Panel titles that state a result, generated from the numbers so they cannot drift from them"""
    levels = results['levels']
    costs = [1 - np.nanmax(levels[name]['floquet']) / np.nanmax(levels[name]['rwa']) for name in ['Γ/3', 'Γ/10']]
    p = results['parameters']
    κ_over_Ω = p['κ'] / ((p['Ωx'] + p['Ωy']) / 2)
    thresholds = results['thresholds']
    period = results['measures']['period']
    certified = float(np.mean(np.array(period['duan_fixed']) < 1))
    published_best = float(np.nanmax(np.array(results['published_heating_grid']['E_N'], dtype=float)))
    c = results['constraints']
    κ_best = c['κ_kHz'][int(np.nanargmax(np.array(c['κ_floquet_Γ/3'], dtype=float)))]
    first = [results['transient'][name]['first_entangled_us'] for name in ['Γ/3', 'Γ/10']]
    max_separable_discord = max(
        d for name in ['Γ/3', 'Γ/10'] for ν, d in zip(results['measures'][name]['ν_tilde'], results['measures'][name]['discord_xy'])
        if ν is not None and np.isfinite(ν) and ν >= 1
    )
    return {
        'levels': f'Counter-rotating terms cost {100*min(costs):.0f}–{100*max(costs):.0f}% ' + rf'($\kappa/\bar{{\Omega}}$ = {κ_over_Ω:.2f})',
        'sweep': (f'Threshold C ≈ {thresholds["floquet_balanced"]["C"]:.1f} (RWA {thresholds["rwa_balanced"]["C"]:.1f});'
                  f' today {results["C_experiment_balanced"]:.1f}'),
        'map': 'Entangled region (inside black contour)',
        'period': f'Lab-frame Duan: {100*certified:.0f}% of the period',
        'tones': 'Lock the difference, not the mean',
        'κ': f'Best κ/2π ≈ {κ_best:.0f} kHz (heating Γ/3)',
        'transient': f'$E_N$ > 0 after {first[1]:.0f}–{first[0]:.0f} μs',
        'witnesses': 'Witnesses miss local squeezing',
        'resolution': 'Resolving the β modes caps their cooling',
        'published_grid': (f'Published heating: best $E_N$ {published_best:.2f} here'
                           if published_best > 0 else 'Published heating: separable everywhere'),
        'discord': f'Separable, yet discord {max_separable_discord:.2f}',
    }


def to_jsonable(value):
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (np.floating, float)):
        # 6 significant digits keeps the file small and is far beyond the model's accuracy
        return None if not np.isfinite(value) else float(f'{value:.6g}')
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    return value


def compact_json(value):
    """Indented JSON with every innermost list on one line, so the file diffs and reads well"""
    text = json.dumps(value, indent=1, ensure_ascii=False)
    return re.sub(r'\[\s+([^\[\]{}]*?)\s+\]', lambda m: '[' + re.sub(r'\s*\n\s*', ' ', m.group(1)) + ']', text)


def from_jsonable(value):
    if isinstance(value, dict):
        return {k: from_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [from_jsonable(v) for v in value]
    return np.nan if value is None else value


if __name__ == '__main__':
    FIGURES.mkdir(exist_ok=True)
    style()
    if '--plot-only' in sys.argv:
        # redraw from a previous results.json, e.g. after changing the styling
        results = from_jsonable(json.loads((HERE / 'results.json').read_text()))
    else:
        print('computing ...')
        results = compute()
    results['figure_titles'] = make_titles(results)
    (HERE / 'results.json').write_text(compact_json(to_jsonable(results)) + '\n')
    figure_mechanism(results)
    figure_constraints(results)
    figure_measures(results)
    print(f'done in {results["runtime_s"]:.0f} s; figures in {FIGURES}')

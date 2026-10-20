"""
Tone polarisation, mech-mech coupling, node position and a second particle: reproduces every
figure and number in docs/two_tone_entanglement/polarisation_and_two_particles.md

    PYTHONPATH=. python docs/two_tone_entanglement/polarisation_and_two_particles.py [--plot-only]

Writes figures/{trap_shaping,orthogonal_polarisation,mech_mech_and_node,two_particles}.png and
polarisation_results.json next to this file; about ten minutes on an 8-core laptop (it uses
all but one core). --plot-only redraws the figures from an existing polarisation_results.json.

Every trap is built by levisim.multiphysics.two_tone_traps from the tones' polarisations, the
cavity geometry and the node position, calibrated so that one tone reproduces Deplano et al.
(2025). Every steady state is the exact periodic (Floquet) solution of MultiToneDrive, with all
counter-rotating terms and all beats.
"""
import os
import sys
import json
import time
import pathlib
import functools
import multiprocessing

for _variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_variable, '1')      # tiny matrices; one thread per worker process

from dataclasses import dataclass, replace

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize, minimize_scalar

from levisim.core.units import kHz
from levisim.core.constants import π
from levisim.analysis.gaussian_states import partially_transposed_symplectic_eigenvalues
from levisim.multiphysics.covariance import dimension_covariance, mode_index
from levisim.multiphysics.multitone import QuadraticPotential, Tone
from levisim.multiphysics.two_tone import DEPLANO2025
from levisim.multiphysics.two_tone_traps import (
    CavityGeometry,
    TweezerOptics,
    TweezerTone,
    deplano2025_tweezer,
    multitone_drive,
    principal_axes_log_negativity,
    subsystem_log_negativity,
    tweezer_trap,
    two_tone_tones,
)
from levisim.physics.focal_expansion import FocalExpansion
from levisim.physics.focussing_optics import FocussingOptics

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from make_figures import compact_json, from_jsonable, log_ticks, panel_letter, style, to_jsonable  # noqa: E402

FIGURES = HERE / 'figures'
RESULTS = HERE / 'polarisation_results.json'
P = DEPLANO2025
θ45 = np.radians(45.0)
θ_DEPLANO = float(deplano2025_tweezer()[1].θ)       # 55.2°, from the published coupling ratio
# ASSUMPTION: the z decoherence rate is not published; 50 kHz at the published heating, scaled
# with the x and y rates (pure photon recoil would give 20-80 kHz)
Γz_PUBLISHED = 50 * kHz
WORKERS = 7
# Operating points must relax: the slowest Floquet mode must decay at >= 2π × MARGIN. Without this
# the optimiser parks the orthogonal scheme on its parametric threshold (|μ| → 1, relaxation
# times of seconds), where E_N is largest but any drift makes the trap unstable.
MARGIN = 1.0 * kHz

# search grids for the tones: difference offset (tone difference = 2(ν + offset)) and common shift
OFFSETS = (-3, -2, -1, 0, 1)                       # kHz
SHIFTS = (-30, -20, -10, 0, 10, 20, 30)            # kHz
RATIO_GRID = (0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85)

COLOURS = {
    'orthogonal': '#1b1b3a',               # the focal scheme
    'co-polarised, no beat': '#3b75af',    # PR #16's idealisation
    'orthogonal, no beat': '#8c8c8c',
    'toy': '#b07aa1',
    'f_pol=0.7': '#6b6b9a',
    'Deplano θ': '#2a9d8f',
    'particle 1': '#1b1b3a',
    'particle 2': '#e76f51',
    'inter': '#2a9d8f',
    'experiment': '#b5179e',
}


# ----------------------------------------------------------------------------------------------
# SCENARIOS
# ----------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Particle:
    """
    One tweezer with its tones. scheme is one of
        'orthogonal'                         red along y, blue along x (+η)
        'co-polarised'                       both along y, with the full intensity beat
        'co-polarised, no beat'              PR #16's idealisation (v = 0)
        'orthogonal, no beat'                orthogonal tones with the beat term deleted (toy)
        'orthogonal, beat sign flipped'      orthogonal tones with the beat's sign reversed (toy)
        'orthogonal, blue not scattering'    blue tone shapes the trap but does not scatter (toy)
        'single tone'                        red tone only, full power
        'single blue tone'                   blue tone only, full power
    """
    scheme: str = 'orthogonal'
    θ: float = θ45
    f_pol: float = 1.0
    Φ: float = π / 2
    include_z: bool = False
    η: float = 0.0                      # error in the angle between the tones' polarisations [rad]
    g_xy: float = 0.0                   # toy static coupling H/ħ ⊃ -g_xy·X_xX_y (tweezer axes) [2π·Hz]
    anisotropy_scale: float = 1.0       # Ω_⊥² - Ω_∥² relative to the calibrated tweezer
    frequency_scale: float = 1.0        # every trap frequency scaled by this
    coupling_scale: float = 1.0         # ω_d relative to the calibrated tweezer
    mass_scale: float = 1.0
    Γz_scale: float = 1.0               # z decoherence relative to the assumed Γz_PUBLISHED


@functools.lru_cache(maxsize=None)
def calibration(f_pol=1.0, include_z=False):
    return deplano2025_tweezer(f_pol=f_pol, include_z=include_z)


def build_trap(pt, ρ, offset, shift, index=1, ν=None):
    """
    TrapModel of particle `pt` with tones at ω_c ∓ (ν + offset) + shift; ν defaults to the mean
    of the trap's own static x, y frequencies. Returns (trap, ν_own).
    """
    optics, _, ω_d, m = calibration(pt.f_pol, pt.include_z)
    if pt.anisotropy_scale != 1.0 or pt.frequency_scale != 1.0:
        M = optics.mean_square * pt.frequency_scale**2
        D = optics.anisotropy * pt.frequency_scale**2 * pt.anisotropy_scale
        Ω_z = None if optics.Ω_z is None else optics.Ω_z * pt.frequency_scale
        optics = TweezerOptics(Ω_perp=np.sqrt(M + D/2), Ω_par=np.sqrt(M - D/2), Ω_z=Ω_z, f_pol=optics.f_pol)
    ω_d, m = ω_d * pt.coupling_scale, m * pt.mass_scale
    geometry = CavityGeometry(θ=pt.θ, Φ=pt.Φ)
    ψ_red = π / 2
    ψ_blue = (0.0 if pt.scheme.startswith('orthogonal') else π / 2) + pt.η
    p_red, p_blue = (np.array([np.cos(ψ), np.sin(ψ), 0.0]) @ geometry.polarisation for ψ in (ψ_red, ψ_blue))
    extra = None
    if pt.g_xy:
        n = 3 if pt.include_z else 2
        extra = np.zeros((n, n))
        # H/ħ = W·X_xX_y ⇔ stiffness K_xy = 2W·(Ω_xΩ_y)^½, with (Ω_xΩ_y)^½ ≈ Ω̄
        extra[0, 1] = extra[1, 0] = -2 * pt.g_xy * np.sqrt(optics.mean_square)

    def tones_for(ν_tones):
        if pt.scheme == 'single tone':
            return [TweezerTone(Δ=-(ν_tones + offset) + shift, amplitude=1.0, ψ=ψ_red, label=f'red{index}')]
        if pt.scheme == 'single blue tone':
            return [TweezerTone(Δ=+(ν_tones + offset) + shift, amplitude=1.0, ψ=ψ_red, label=f'blue{index}')]
        return two_tone_tones(ν_tones, ρ, ψ_red, ψ_blue, projections=(p_red, p_blue), offset=offset,
                              common_shift=shift, label=str(index))

    kwargs = dict(index=index, κ=P['κ'], extra_stiffness=extra)
    trap = tweezer_trap(optics, tones_for(np.sqrt(optics.mean_square)), geometry, ω_d, m, **kwargs)
    for _ in range(2 if np.isclose(np.cos(pt.Φ), 0) else 3):   # off a node the trap depends on the tones
        ν_own = float(np.mean([trap.frequencies[f'x{index}'], trap.frequencies[f'y{index}']]))
        trap = tweezer_trap(optics, tones_for(ν_own if ν is None else ν), geometry, ω_d, m, **kwargs)

    if pt.scheme in ('co-polarised, no beat', 'orthogonal, no beat'):
        trap.potentials = [q for q in trap.potentials if not q.label.startswith('beat')]
    elif pt.scheme == 'orthogonal, beat sign flipped':
        trap.potentials = [
            QuadraticPotential(q.frequency, {k: -v for k, v in q.coefficients.items()}, q.label)
            if q.label.startswith('beat') else q for q in trap.potentials
        ]
    elif pt.scheme == 'orthogonal, blue not scattering':
        trap.tones = [
            Tone(Δ=t.Δ, amplitude=t.amplitude, phase=t.phase, label=t.label, couplings={k: 0.0 for k in t.couplings})
            if t.label.startswith('blue') else t for t in trap.tones
        ]
    return trap, ν_own


def heating_rates(traps, Γ_scale, per_trap=None, z_scale=1.0):
    """Published decoherence rates × Γ_scale (× per_trap[index]); z uses the assumed Γz_PUBLISHED"""
    base = {'x': P['Γx_decoherence'], 'y': P['Γy_decoherence'], 'z': z_scale * Γz_PUBLISHED}
    rates = {}
    for trap in traps:
        scale = Γ_scale * (1.0 if per_trap is None else per_trap.get(trap.index, 1.0))
        for label in trap.labels:
            rates[label] = scale * base[label[0]]
    return rates


def steady_state(traps, Γ_scale, frame_offset=0.0, per_trap=None, samples=6, z_scale=1.0):
    rates = heating_rates(traps, Γ_scale, per_trap, z_scale)
    mtrap, drive = multitone_drive(traps, P['κ'], rates, frame_offset=frame_offset)
    return mtrap, drive, drive.periodic_steady_state(samples)


def pair_lnν(mtrap, covariances, labels_A, labels_B):
    """-ln ν̃_- of the partial transpose at each sample (positive ⇔ entangled, for 1 × 1 modes)"""
    out = []
    for V in covariances:
        V_AB = dimension_covariance(mtrap, V, list(labels_A) + list(labels_B))
        out.append(-np.log(partially_transposed_symplectic_eigenvalues(V_AB).min()))
    return np.array(out)


def slowest_relaxation(mtrap, drive, ss, labels, weight=0.1):
    """
    Slowest decay rate [1/s] of the Floquet modes with at least `weight` of their amplitude in the
    given dimensions. This leaves out z near a node, which decouples from the cavity and relaxes
    only through gas damping, but has nothing to do with the x-y state.
    """
    μ, vectors = np.linalg.eig(ss.monodromy)
    indices = np.ravel([[2 * mode_index(mtrap, label), 2 * mode_index(mtrap, label) + 1] for label in labels])
    share = np.sum(np.abs(vectors[indices, :])**2, axis=0) / np.sum(np.abs(vectors)**2, axis=0)
    return float(-np.log(np.max(np.abs(μ[share >= weight]))) / drive.period)


def evaluate(pt, Γ_scale, ρ, offset, shift, extra=False, samples=6, margin=MARGIN):
    """
    Single particle: period-averaged E_N(x:y) along the principal axes, the slowest relaxation
    rate, and a 'score' that equals E_N when entangled and max(-ln ν̃) (≤ 0) otherwise, so that it is
    continuous through threshold. Operating points that relax slower than `margin` score -10.
    """
    trap, ν = build_trap(pt, ρ, offset, shift)
    mtrap, drive, ss = steady_state([trap], Γ_scale, frame_offset=shift, samples=samples, z_scale=pt.Γz_scale)
    if not ss.stable:
        return dict(stable=False, μ=float(ss.largest_multiplier), E=np.nan, score=-10.0, relaxation_kHz=0.0)
    relaxation = slowest_relaxation(mtrap, drive, ss, ['x1', 'y1'])
    lnν = pair_lnν(mtrap, ss.covariances, ['x1'], ['y1'])
    E = float(np.mean(np.maximum(lnν, 0)))
    out = dict(stable=True, μ=float(ss.largest_multiplier), E=E, relaxation_kHz=float(relaxation / (2 * π * 1e3)),
               score=(E if E > 0 else float(lnν.max())) if relaxation >= margin else -10.0)
    if extra:
        out['E_tweezer_axes'] = float(np.mean([principal_axes_log_negativity(mtrap, V, trap) for V in ss.covariances]))
        out['axes_angle_deg'] = float(np.degrees(trap.axes_angle))
        out['frequencies_kHz'] = [float(Ω / kHz) for Ω in trap.frequencies.values()]
        out['ν_kHz'] = ν / kHz
    return out


def optimise_tones(pt, Γ_scale, ρ, margin=MARGIN):
    """Best (offset, shift) at fixed ρ: a grid, then Nelder-Mead from its best point"""
    grid = []
    for o in OFFSETS:
        for s in SHIFTS:
            r = evaluate(pt, Γ_scale, ρ, o * kHz, s * kHz, margin=margin)
            grid.append((r['score'], o, s, r['μ']))
    score0, o0, s0, _ = max(grid)
    best = (score0, o0, s0)
    if score0 > -10:
        result = minimize(
            lambda q: -evaluate(pt, Γ_scale, ρ, q[0] * kHz, q[1] * kHz, margin=margin)['score'], [o0, s0],
            method='Nelder-Mead',
            options=dict(xatol=0.05, fatol=1e-4, maxfev=80, initial_simplex=[[o0, s0], [o0 + 0.5, s0], [o0, s0 + 3]]),
        )
        if -result.fun > score0:
            best = (-result.fun, *result.x)
    score, o, s = best
    out = evaluate(pt, Γ_scale, ρ, o * kHz, s * kHz, extra=score > -10, margin=margin)
    if score <= -10:
        out['E'] = np.nan                      # no admissible operating point at this ρ
    out.update(ρ=ρ, offset_kHz=float(o), shift_kHz=float(s), score=float(score), margin_kHz=margin / kHz,
               smallest_grid_multiplier=float(min(g[3] for g in grid)))
    return out


def _optimise_task(args):
    pt, Γ_scale, ρ, *rest = args
    return optimise_tones(pt, Γ_scale, ρ, *rest)


def pmap(function, tasks):
    """Parallel map over worker processes (forked, so it also works where spawning is restricted)"""
    with multiprocessing.get_context('fork').Pool(WORKERS) as pool:
        return pool.map(function, tasks, chunksize=1)


def best_of(rows):
    """Row with the highest score (rows from optimise_tones at several ρ)"""
    return max(rows, key=lambda r: r['score'])


def admissible_E(row):
    """E_N of an optimise_tones row, 0 when no admissible operating point exists"""
    return row['E'] if row['score'] > -10 and np.isfinite(row['E']) else 0.0


def threshold_from_scores(Γ_scales, scores):
    """Γ_scale where the best score crosses zero, interpolating in ln Γ (None if not bracketed)"""
    Γ_scales, scores = np.asarray(Γ_scales), np.asarray(scores)
    order = np.argsort(Γ_scales)
    Γ_scales, scores = Γ_scales[order], scores[order]
    for i in range(len(Γ_scales) - 1):
        if scores[i] > 0 >= scores[i + 1]:
            x0, x1 = np.log(Γ_scales[i]), np.log(Γ_scales[i + 1])
            return float(np.exp(x0 + (x1 - x0) * scores[i] / (scores[i] - scores[i + 1])))
    return None


def optimise_over_ratios(configs, Γ_scales, ratios=RATIO_GRID, margin=MARGIN):
    """{(name, Γ_scale): [rows over ratios]} for every config, in parallel"""
    tasks = [(pt, Γ, ρ, margin) for name, pt in configs.items() for Γ in Γ_scales for ρ in ratios]
    rows = pmap(_optimise_task, tasks)
    out, i = {}, 0
    for name in configs:
        for Γ in Γ_scales:
            out[(name, Γ)] = rows[i:i + len(ratios)]
            i += len(ratios)
    return out


# ----------------------------------------------------------------------------------------------
# A. THE TRAP: ANALYTIC FOCAL FIELD AND TWO-TONE POTENTIALS
# ----------------------------------------------------------------------------------------------
def focussing_optics(NA, f0):
    """FocussingOptics at numerical aperture NA with filling factor f0 = w0/(f·sinθmax)"""
    reference = FocussingOptics(NA=NA)
    return FocussingOptics(NA=NA, w0=reference.w0 * f0 / reference.f0)


def compute_optics():
    out = {'NA': np.round(np.arange(0.30, 0.951, 0.025), 3)}
    for key, apodisation, f0 in (('levisim, f0=1', 'levisim', 1.0), ('Novotny, f0=1', 'novotny', 1.0),
                                 ('levisim, f0=2', 'levisim', 2.0)):
        rows = {'perp/par': [], 'perp/z': [], 'anisotropy/mean': []}
        for NA in out['NA']:
            fe = FocalExpansion(focussing_optics(NA, f0), apodisation=apodisation)
            κ = fe.curvatures
            rows['perp/par'].append(fe.frequency_ratios['perp/par'])
            rows['perp/z'].append(fe.frequency_ratios['perp/z'])
            rows['anisotropy/mean'].append(fe.anisotropy / ((κ['par'] + κ['perp']) / 2))
        out[key] = rows
    # the closed forms against finite differences of the full Richards-Wolf field
    checks = []
    for NA in (0.5, 0.77, 0.9):
        fe = FocalExpansion(focussing_optics(NA, 1.0))
        for ψ1, ψ2 in ((0.0, 0.0), (π/2, π/2), (0.0, π/2), (π/4, π/4), (π/6, 2*π/3)):
            analytic, numeric = fe.interference_hessian(ψ1, ψ2), fe.numerical_hessian(ψ1, ψ2)
            checks.append(float(np.abs(analytic - numeric).max() / np.abs(analytic).max()))
    out['max_relative_error_vs_full_field'] = max(checks)
    out['deplano_ratios'] = {'perp/par': P['Ωx'] / P['Ωy'], 'perp/z': P['Ωx'] / (21.4 * kHz)}
    return out


def compute_shaping():
    """Static splitting and beat strengths against G+/G-, θ = 45°, model and closed forms"""
    ratios = np.round(np.arange(0.0, 0.951, 0.025), 3)
    out = {'ratios': ratios}
    for f_pol in (1.0, 0.7):
        optics = calibration(f_pol)[0]
        Δp, Δs = f_pol * optics.anisotropy, (1 - f_pol) * optics.anisotropy
        for scheme in ('co-polarised', 'orthogonal'):
            key = f'{scheme}, f_pol={f_pol:g}'
            rows = {'δ_kHz': [], 'δ_closed_form_kHz': [], 'rates_kHz': [], 'rates_closed_form_kHz': [], 'cooling_kHz': []}
            for ρ in ratios:
                trap, ν = build_trap(Particle(scheme, f_pol=f_pol), ρ, 0.0, 0.0)
                Ωx, Ωy = trap.frequencies['x1'], trap.frequencies['y1']
                c_red, c_blue = trap.tones[0].amplitude, trap.tones[1].amplitude
                rows['δ_kHz'].append((Ωx - Ωy) / kHz)
                if scheme == 'orthogonal':
                    s2 = c_red**2 - c_blue**2
                    Kx, Ky = optics.mean_square + (Δs + Δp * s2) / 2, optics.mean_square - (Δs + Δp * s2) / 2
                    rows['δ_closed_form_kHz'].append((np.sqrt(Kx) - np.sqrt(Ky)) / kHz)
                    beat = [q for q in trap.potentials if q.label.startswith('beat')]
                    W = abs(beat[0].coefficients.get(('x1', 'y1'), 0.0)) if beat else 0.0
                    rows['rates_kHz'].append([W / 2 / kHz])                    # χ of χ(b_x b_y + h.c.)
                    rows['rates_closed_form_kHz'].append([c_red * c_blue * Δp / (4 * np.sqrt(Ωx * Ωy)) / kHz])
                else:
                    rows['δ_closed_form_kHz'].append((optics.Ω_perp - optics.Ω_par) / kHz)
                    beat = [q for q in trap.potentials if q.label.startswith('beat')]
                    W = [abs(beat[0].coefficients.get((l, l), 0.0)) if beat else 0.0 for l in ('x1', 'y1')]
                    rows['rates_kHz'].append([w / kHz for w in W])              # 2ε of ε(b² + h.c.)
                    rows['rates_closed_form_kHz'].append([c_red * c_blue * Ω / 2 / kHz for Ω in (Ωx, Ωy)])
                # cavity cooling of the Bogoliubov modes, 4(G-² - G+²)/κ with G the rms red coupling
                G_red = np.sqrt(np.mean([abs(trap.tones[0].couplings[l] * c_red)**2 for l in ('x1', 'y1')]))
                G_blue = np.sqrt(np.mean([abs(trap.tones[1].couplings[l] * c_blue)**2 for l in ('x1', 'y1')]))
                rows['cooling_kHz'].append(4 * (G_red**2 - G_blue**2) / P['κ'] / kHz)
            out[key] = rows
    # instantaneous trap over one beat period, G+/G- = 0.525, f_pol = 1
    ρ = 0.525
    snapshots = {}
    for scheme in ('co-polarised', 'orthogonal'):
        trap, ν = build_trap(Particle(scheme), ρ, 0.0, 0.0)
        Ω = np.array([trap.frequencies['x1'], trap.frequencies['y1']])
        K0 = np.diag(Ω**2)
        beat = [q for q in trap.potentials if q.label.startswith('beat')][0]
        phases, Ks = np.linspace(0, 2*π, 9)[:-1], []
        for φ in phases:
            K = K0.copy()
            for (a, b), W in beat.coefficients.items():
                i, j = ('x1', 'y1').index(a), ('x1', 'y1').index(b)
                if i == j:
                    K[i, i] += 4 * Ω[i] * np.real(W * np.exp(-1j * φ))
                else:
                    K[i, j] += 2 * np.sqrt(Ω[i] * Ω[j]) * np.real(W * np.exp(-1j * φ))
                    K[j, i] = K[i, j]
            Ks.append(K / (2 * π * kHz)**2)
        snapshots[scheme] = {'phases': phases, 'K_kHz2': Ks}
    out['snapshots'] = snapshots
    return out


def compute_off_node_terms():
    """What moving off the node adds to one red tone's trap (z included), Deplano geometry"""
    dΦs = np.round(np.linspace(0.0, 1.5, 31), 3)
    node, _ = build_trap(Particle('single tone', θ=θ_DEPLANO, include_z=True), 0.0, 0.0, 0.0)
    rows = {'dΦ': dΦs, 'photons': [], 'shift_kHz': [], 'g_xy_kHz': [], 'g_xz_kHz': [], 'axes_deg': [],
            'coupling_x_rel': [], 'coupling_z_kHz': []}
    for dΦ in dΦs:
        trap, _ = build_trap(Particle('single tone', θ=θ_DEPLANO, include_z=True, Φ=π/2 - dΦ), 0.0, 0.0, 0.0)
        rows['photons'].append(float(np.sum(np.abs(trap.coherent_amplitudes)**2)))
        rows['shift_kHz'].append([(trap.frequencies[l] - node.frequencies[l]) / kHz for l in ('x1', 'y1', 'z1')])
        E = trap.axes[:2, :2]
        K = E @ np.diag([trap.frequencies['x1']**2, trap.frequencies['y1']**2]) @ E.T     # tweezer axes
        rows['g_xy_kHz'].append(float(-K[0, 1] / (2 * np.sqrt(trap.frequencies['x1'] * trap.frequencies['y1'])) / kHz))
        static = [q for q in trap.potentials if q.label == 'static xz, yz']
        rows['g_xz_kHz'].append(float(max(abs(v) for v in static[0].coefficients.values()) / kHz) if static else 0.0)
        rows['axes_deg'].append(float(np.degrees(trap.axes_angle)))
        rows['coupling_x_rel'].append(float(abs(trap.tones[0].couplings['x1']) / abs(node.tones[0].couplings['x1'])))
        rows['coupling_z_kHz'].append(float(abs(trap.tones[0].couplings['z1']) / kHz))
    return rows


# ----------------------------------------------------------------------------------------------
# B. ORTHOGONAL POLARISATION
# ----------------------------------------------------------------------------------------------
B1_SCHEMES = ('orthogonal', 'orthogonal, no beat', 'co-polarised, no beat', 'co-polarised',
              'orthogonal, beat sign flipped', 'orthogonal, blue not scattering')
B1_RATIOS = tuple(np.round(np.arange(0.05, 0.851, 0.05), 3))
B2_HEATING = (1.0, 0.85, 0.7, 0.6, 0.5, 0.4, 1/3, 0.25, 0.2, 0.15, 0.1)
B2_CONFIGS = {
    'orthogonal': Particle('orthogonal'),
    'orthogonal, f_pol=0.7': Particle('orthogonal', f_pol=0.7),
    'orthogonal, Deplano θ': Particle('orthogonal', θ=θ_DEPLANO),
    'co-polarised, no beat': Particle('co-polarised, no beat'),
    'co-polarised, no beat, Deplano θ': Particle('co-polarised, no beat', θ=θ_DEPLANO),
}
B3_θ = tuple(np.arange(20, 71, 5))
B4_FPOL = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
B5_η = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
B6_SHIFTS = tuple(np.round(np.arange(-40, 40.1, 2.5), 2))
B6_OFFSETS = tuple(np.round(np.arange(-7, 3.01, 0.5), 2))
B7_MARGINS = (0.0, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0)     # kHz


def _evaluate_task(args):
    pt, Γ_scale, ρ, offset, shift = args
    return evaluate(pt, Γ_scale, ρ, offset, shift, margin=0.0)


def compute_orthogonal():
    out = {}
    b1 = optimise_over_ratios({name: Particle(name) for name in B1_SCHEMES}, (1/3, 0.1), B1_RATIOS)
    out['ratio_scan'] = {f'{name} | Γ×{Γ:.3g}': rows for (name, Γ), rows in b1.items()}
    out['ratio_scan_ratios'] = B1_RATIOS
    b2 = optimise_over_ratios(B2_CONFIGS, B2_HEATING)
    out['heating_scan'] = {}
    for name in B2_CONFIGS:
        best = [best_of(b2[(name, Γ)]) for Γ in B2_HEATING]
        out['heating_scan'][name] = {
            'Γ_scale': B2_HEATING, 'E_N': [admissible_E(b) for b in best],
            'score': [b['score'] for b in best], 'best': best,
            'threshold_Γ_scale': threshold_from_scores(B2_HEATING, [b['score'] for b in best]),
        }
    b3 = optimise_over_ratios({f'{scheme} | θ={θ}': Particle(scheme, θ=np.radians(θ))
                               for scheme in ('orthogonal', 'co-polarised, no beat') for θ in B3_θ}, (1/3,))
    out['θ_scan'] = {'θ_deg': B3_θ}
    for scheme in ('orthogonal', 'co-polarised, no beat'):
        out['θ_scan'][scheme] = [best_of(b3[(f'{scheme} | θ={θ}', 1/3)]) for θ in B3_θ]
    b4 = optimise_over_ratios({f'f_pol={f}': Particle('orthogonal', f_pol=f) for f in B4_FPOL}, (1/3, 0.1))
    out['f_pol_scan'] = {'f_pol': B4_FPOL}
    for Γ in (1/3, 0.1):
        out['f_pol_scan'][f'Γ×{Γ:.3g}'] = [best_of(b4[(f'f_pol={f}', Γ)]) for f in B4_FPOL]
    b5 = optimise_over_ratios({f'η={η}': Particle('orthogonal', η=np.radians(η)) for η in B5_η}, (1/3,))
    out['η_scan'] = {'η_deg': B5_η, 'rows': [best_of(b5[(f'η={η}', 1/3)]) for η in B5_η]}
    # windows around the optimum: common shift and tone-difference offset, one at a time
    out['windows'] = {'shift_kHz': B6_SHIFTS, 'offset_kHz': B6_OFFSETS}
    tasks, keys = [], []
    for scheme in ('orthogonal', 'co-polarised, no beat'):
        for Γ in (1/3, 0.1):
            best = best_of(out['ratio_scan'][f'{scheme} | Γ×{Γ:.3g}'])
            keys.append((f'{scheme} | Γ×{Γ:.3g}', best))
            tasks += [(Particle(scheme), Γ, best['ρ'], best['offset_kHz'] * kHz, sh * kHz) for sh in B6_SHIFTS]
            tasks += [(Particle(scheme), Γ, best['ρ'], of * kHz, best['shift_kHz'] * kHz) for of in B6_OFFSETS]
    rows = pmap(_evaluate_task, tasks)
    n1, n2 = len(B6_SHIFTS), len(B6_OFFSETS)
    for i, (key, best) in enumerate(keys):
        chunk = rows[i * (n1 + n2):(i + 1) * (n1 + n2)]
        out['windows'][key] = {'best': best, 'shift': [r['E'] for r in chunk[:n1]], 'offset': [r['E'] for r in chunk[n1:]],
                               'shift_relaxation_kHz': [r['relaxation_kHz'] for r in chunk[:n1]],
                               'offset_relaxation_kHz': [r['relaxation_kHz'] for r in chunk[n1:]]}
    # how much of each scheme's entanglement needs a slowly relaxing (near-threshold) operating point
    out['margin_scan'] = {'margin_kHz': B7_MARGINS}
    tasks = [(Particle(scheme), Γ, ρ, m * kHz) for scheme in ('orthogonal', 'co-polarised, no beat')
             for Γ in (1/3, 0.1) for m in B7_MARGINS for ρ in RATIO_GRID]
    rows = pmap(_optimise_task, tasks)
    i = 0
    for scheme in ('orthogonal', 'co-polarised, no beat'):
        for Γ in (1/3, 0.1):
            best = []
            for m in B7_MARGINS:
                best.append(best_of(rows[i:i + len(RATIO_GRID)]))
                i += len(RATIO_GRID)
            out['margin_scan'][f'{scheme} | Γ×{Γ:.3g}'] = best
    return out


# ----------------------------------------------------------------------------------------------
# C. MECH-MECH COUPLING AND THE NODE
# ----------------------------------------------------------------------------------------------
C1_G = tuple(range(-8, 9))
C3_dΦ = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1.2)
C_SCHEMES = {'orthogonal | Γ×0.333': ('orthogonal', 1/3), 'co-polarised, no beat | Γ×0.1': ('co-polarised, no beat', 0.1)}


def _single_tone_task(args):
    dΦ, Γ_scale = args
    pt = Particle('single tone', θ=θ_DEPLANO, include_z=True, Φ=π/2 - dΦ)
    result = minimize_scalar(lambda o: -evaluate(pt, Γ_scale, 0.0, o * kHz, 0.0, margin=0.0)['score'],
                             bounds=(-40, 40), method='bounded', options={'xatol': 0.2})
    best = evaluate(pt, Γ_scale, 0.0, result.x * kHz, 0.0, margin=0.0)
    best.update(offset_kHz=float(result.x), ν̃_min=float(np.exp(-best['score'])) if best['stable'] else np.nan)
    return best


def compute_mech_mech():
    out = {'g_xy_kHz': C1_G, 'dΦ': C3_dΦ}
    configs = {}
    for key, (scheme, Γ) in C_SCHEMES.items():
        for g in C1_G:
            configs[f'{key} | g={g}'] = (Particle(scheme, g_xy=g * kHz), Γ)
        for dΦ in C3_dΦ:
            configs[f'{key} | dΦ={dΦ}'] = (Particle(scheme, Φ=π/2 - dΦ, include_z=True), Γ)
            # the same x-y couplings at a node, without z or the carrier field: isolates what the
            # off-node terms themselves do
            configs[f'{key} | node, couplings × sinΦ, dΦ={dΦ}'] = (Particle(scheme, coupling_scale=float(np.cos(dΦ))), Γ)
    for key in ('orthogonal | Γ×0.333',):
        for dΦ in (0.3, 0.7):
            for zs in (0.4, 1.6):
                configs[f'{key} | dΦ={dΦ} | Γz×{zs}'] = (Particle('orthogonal', Φ=π/2 - dΦ, include_z=True, Γz_scale=zs), 1/3)
    tasks = [(pt, Γ, ρ) for pt, Γ in configs.values() for ρ in RATIO_GRID]
    rows = pmap(_optimise_task, tasks)
    best = {name: best_of(rows[i * len(RATIO_GRID):(i + 1) * len(RATIO_GRID)]) for i, name in enumerate(configs)}
    for key in C_SCHEMES:
        out[f'static coupling | {key}'] = [best[f'{key} | g={g}'] for g in C1_G]
        out[f'off node | {key}'] = [best[f'{key} | dΦ={dΦ}'] for dΦ in C3_dΦ]
        out[f'node, couplings × sinΦ | {key}'] = [best[f'{key} | node, couplings × sinΦ, dΦ={dΦ}'] for dΦ in C3_dΦ]
    out['Γz sensitivity'] = {name.split(' | ', 1)[1]: best[name] for name in configs if 'Γz×' in name}
    single_dΦ = tuple(np.round(np.arange(0.0, 1.51, 0.1), 2))
    out['single tone'] = {'dΦ': single_dΦ}
    for Γ in (1e-3, 1.0):
        out['single tone'][f'Γ×{Γ:g}'] = pmap(_single_tone_task, [(dΦ, Γ) for dΦ in single_dΦ])
    return out



# ----------------------------------------------------------------------------------------------
# D. A SECOND PARTICLE IN THE SAME CAVITY
# ----------------------------------------------------------------------------------------------
D_δ_RATIOS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0)
D_SHIFT_STEPS = tuple(range(0, 25)) + (28, 32)     # Δs = (ν + offset)·k/64
D_INTER_RATIOS = tuple(np.round(np.arange(0.1, 0.901, 0.05), 3))
D_INTER_HEATING = (1.0, 0.5, 1/3, 0.1)


def occupations(mtrap, V, labels):
    return [float(np.trace(dimension_covariance(mtrap, V, [label])) / 4 - 0.5) for label in labels]


def pair_report(traps, Γ_scale, frame_offset, per_trap=None, samples=6):
    """Intra- and inter-particle entanglement of two traps sharing the cavity"""
    mtrap, drive, ss = steady_state(traps, Γ_scale, frame_offset=frame_offset, per_trap=per_trap, samples=samples)
    out = dict(stable=bool(ss.stable), μ=float(ss.largest_multiplier), period_us=drive.period * 1e6)
    if not ss.stable:
        return out
    E = lambda A, B: float(np.mean([subsystem_log_negativity(mtrap, V, A, B) for V in ss.covariances]))
    out.update(E_11=E(['x1'], ['y1']), E_22=E(['x2'], ['y2']), E_1_2=E(['x1', 'y1'], ['x2', 'y2']),
               E_x1y2=E(['x1'], ['y2']), E_y1x2=E(['y1'], ['x2']), E_x1x2=E(['x1'], ['x2']),
               n=occupations(mtrap, ss.covariances[0], ['x1', 'y1', 'x2', 'y2']))
    out['score_1_2'] = out['E_1_2'] if out['E_1_2'] > 0 else float(max(
        pair_lnν(mtrap, ss.covariances, ['x1', 'y1'], ['x2', 'y2'])))
    return out


def alone(pt, ρ, offset, shift, Γ_scale, ν=None):
    trap, _ = build_trap(pt, ρ, offset, shift, ν=ν)
    mtrap, drive, ss = steady_state([trap], Γ_scale, frame_offset=shift)
    if not ss.stable:
        return np.nan
    return float(np.mean(np.maximum(pair_lnν(mtrap, ss.covariances, ['x1'], ['y1']), 0)))


def _δ_ratio_task(args):
    pt, ρ, offset, shift, Γ_scale, a = args
    trap1, ν1 = build_trap(pt, ρ, offset, shift, index=1)
    pt2 = replace(pt, anisotropy_scale=a)
    for _ in range(3):              # rescale particle 2's trap so that its mean frequency is ν1
        _, ν2 = build_trap(pt2, ρ, offset, shift, index=2)
        pt2 = replace(pt2, frequency_scale=pt2.frequency_scale * ν1 / ν2)
    trap2, ν2 = build_trap(pt2, ρ, offset, shift, index=2, ν=ν1)
    out = pair_report([trap1, trap2], Γ_scale, frame_offset=shift)
    out.update(a=a, δ1_kHz=(trap1.frequencies['x1'] - trap1.frequencies['y1']) / kHz,
               δ2_kHz=(trap2.frequencies['x2'] - trap2.frequencies['y2']) / kHz, ν2_minus_ν1_Hz=(ν2 - ν1) / (2 * π),
               E_2_alone=alone(pt2, ρ, offset, shift, Γ_scale, ν=ν1))
    return out


def _shift_task(args):
    """Particles at common shifts s1, s2 with s2 - s1 = ±Δs; Δs = (ν + offset)·k/64 keeps the drive periodic"""
    pt, ρ, offset, s0, Γ_scale, k, arrangement = args
    _, ν1 = build_trap(pt, ρ, offset, s0, index=1)
    Δs = (ν1 + offset) * k / 64
    s1, s2 = {'symmetric': (s0 - Δs / 2, s0 + Δs / 2), 'second below': (s0, s0 - Δs)}[arrangement]
    trap1, _ = build_trap(pt, ρ, offset, s1, index=1)
    trap2, _ = build_trap(pt, ρ, offset, s2, index=2)
    out = pair_report([trap1, trap2], Γ_scale, frame_offset=(s1 + s2) / 2)
    out.update(k=k, Δs_kHz=Δs / kHz, s1_kHz=s1 / kHz, s2_kHz=s2 / kHz, E_1_alone=alone(pt, ρ, offset, s1, Γ_scale),
               E_2_alone=alone(pt, ρ, offset, s2, Γ_scale))
    return out


def _inter_task(args):
    """One red tone on particle 1, one blue tone on a smaller particle 2 (G_+/G_- = ρ, same trap frequencies)"""
    ρ, Γ_scale, θ, heating_model = args
    per_trap = {2: ρ**2 if heating_model == 'recoil ∝ mass' else 1.0}

    def report(offset):
        trap1, ν = build_trap(Particle('single tone', θ=θ), 0.0, offset, 0.0, index=1)
        trap2, _ = build_trap(Particle('single blue tone', θ=θ, coupling_scale=ρ**2, mass_scale=ρ**2), 0.0, offset, 0.0,
                              index=2, ν=ν)
        return pair_report([trap1, trap2], Γ_scale, frame_offset=0.0, per_trap=per_trap)

    score = lambda o: report(o * kHz).get('score_1_2', -10.0)
    result = minimize_scalar(lambda o: -score(o), bounds=(-3, 3), method='bounded', options={'xatol': 0.05})
    best_offset = result.x if -result.fun > score(0.0) else 0.0
    out = report(best_offset * kHz)
    out.update(ρ=ρ, offset_kHz=float(best_offset))
    return out


def compute_two_particles(orthogonal_results):
    """Uses the single-particle optima found in compute_orthogonal()"""
    out = {}
    schemes = {}
    for key, scheme, Γ, ρ in (('co-polarised, no beat | Γ×0.1', 'co-polarised, no beat', 0.1, None),
                              ('orthogonal | Γ×0.333', 'orthogonal', 1/3, 0.35)):
        rows = orthogonal_results['ratio_scan'][f'{scheme} | Γ×{Γ:.3g}']
        # the orthogonal scheme is run at G+/G- = 0.35, away from its instability, rather than at its optimum
        best = best_of(rows) if ρ is None else min(rows, key=lambda r: abs(r['ρ'] - ρ))
        schemes[key] = (Particle(scheme), best['ρ'], best['offset_kHz'] * kHz, best['shift_kHz'] * kHz, Γ, best['E'])
    out['single-particle optima'] = {key: dict(ρ=v[1], offset_kHz=v[2] / kHz, shift_kHz=v[3] / kHz, E_N=v[5])
                                     for key, v in schemes.items()}
    out['identical'] = {}
    for key, (pt, ρ, o, s, Γ, _) in schemes.items():
        trap1, _ = build_trap(pt, ρ, o, s, index=1)
        for node, Φ2 in (('same node sign', π/2), ('adjacent node', 3*π/2)):
            trap2, _ = build_trap(replace(pt, Φ=Φ2), ρ, o, s, index=2)
            out['identical'][f'{key} | {node}'] = pair_report([trap1, trap2], Γ, frame_offset=s, samples=4)
    tasks = [(pt, ρ, o, s, Γ, a) for (pt, ρ, o, s, Γ, _) in schemes.values() for a in D_δ_RATIOS]
    rows = pmap(_δ_ratio_task, tasks)
    out['δ ratio'] = {key: rows[i * len(D_δ_RATIOS):(i + 1) * len(D_δ_RATIOS)] for i, key in enumerate(schemes)}
    arrangements = [(key, arrangement) for key in schemes for arrangement in ('symmetric', 'second below')]
    tasks = [(*schemes[key][:5], k, arrangement) for key, arrangement in arrangements for k in D_SHIFT_STEPS]
    rows = pmap(_shift_task, tasks)
    out['common shift'] = {f'{key} | {arrangement}': rows[i * len(D_SHIFT_STEPS):(i + 1) * len(D_SHIFT_STEPS)]
                           for i, (key, arrangement) in enumerate(arrangements)}
    out['inter-particle'] = {}
    keys = [(θ_name, θ, model, Γ) for θ_name, θ in (('Deplano θ', θ_DEPLANO), ('θ=45°', θ45))
            for model in ('recoil ∝ mass', 'equal') for Γ in D_INTER_HEATING]
    rows = pmap(_inter_task, [(ρ, Γ, θ, model) for (_, θ, model, Γ) in keys for ρ in D_INTER_RATIOS])
    for i, (θ_name, θ, model, Γ) in enumerate(keys):
        out['inter-particle'][f'{θ_name} | {model} | Γ×{Γ:.3g}'] = rows[i * len(D_INTER_RATIOS):(i + 1) * len(D_INTER_RATIOS)]
    out['inter-particle ratios'] = D_INTER_RATIOS
    return out


# ----------------------------------------------------------------------------------------------
def compute():
    t0 = time.time()
    results = {}
    for name, function in (('optics', compute_optics), ('shaping', compute_shaping),
                           ('off_node_terms', compute_off_node_terms), ('orthogonal', compute_orthogonal),
                           ('mech_mech', compute_mech_mech)):
        t = time.time()
        results[name] = function()
        print(f'  {name}: {time.time() - t:.0f} s', flush=True)
    t = time.time()
    results['two_particles'] = compute_two_particles(results['orthogonal'])
    print(f'  two_particles: {time.time() - t:.0f} s', flush=True)
    results['assumptions'] = {
        'Γz_published_kHz': Γz_PUBLISHED / kHz, 'θ_deplano_deg': float(np.degrees(θ_DEPLANO)),
        'mass_kg': float(calibration()[3]), 'ω_d_MHz': float(calibration()[2] / (2 * π * 1e6)),
    }
    results['runtime_s'] = time.time() - t0
    return results


# ----------------------------------------------------------------------------------------------
# FIGURES
# ----------------------------------------------------------------------------------------------
def _best_E(rows):
    return max((admissible_E(r) for r in rows), default=0.0)


def _top(values, factor):
    """factor × the largest finite value, or 1 if there is none (axis limits)"""
    values = np.asarray(values, dtype=float)
    return factor * np.nanmax(values) if np.isfinite(values).any() and np.nanmax(values) > 0 else 1.0


def _curve(rows, key='E'):
    """One value per row; NaN where unstable or (for optimised rows) no admissible operating point"""
    return np.array([r.get(key, np.nan) if r['stable'] and r.get('score', 0.0) > -10 else np.nan for r in rows],
                    dtype=float)


def _crossing(x, y, level):
    """First x where y crosses level (linear interpolation), or nan"""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float) - level
    for i in range(len(x) - 1):
        if y[i] == 0 or y[i] * y[i + 1] < 0:
            return float(x[i] - y[i] * (x[i + 1] - x[i]) / (y[i + 1] - y[i]))
    return np.nan


def figure_trap_shaping(results):
    optics, shaping, off = results['optics'], results['shaping'], results['off_node_terms']
    fig = plt.figure(figsize=(7.2, 5.0))
    grid = fig.add_gridspec(2, 3, left=0.075, right=0.975, top=0.92, bottom=0.09, wspace=0.42, hspace=0.62)
    NA = np.array(optics['NA'])
    styles = {'levisim, f0=1': dict(color='#1b1b3a', ls='-'), 'Novotny, f0=1': dict(color='#3b75af', ls='--'),
              'levisim, f0=2': dict(color='#8c8c8c', ls=':')}
    NA_par = _crossing(NA, optics['levisim, f0=1']['perp/par'], optics['deplano_ratios']['perp/par'])
    NA_z = _crossing(NA, optics['levisim, f0=1']['perp/z'], optics['deplano_ratios']['perp/z'])
    titles = ('The focal field alone makes the trap\nstiffer across the polarisation',
              f'Measured ratios imply NA ≈ {NA_par:.2f}\n(Ω$_⊥$/Ω$_∥$) but NA ≈ {NA_z:.2f} (Ω$_⊥$/Ω$_z$)')
    for col, (key, label) in enumerate((('perp/par', 'Ω$_⊥$/Ω$_∥$ (one tone)'), ('perp/z', 'Ω$_⊥$/Ω$_z$ (one tone)'))):
        ax = fig.add_subplot(grid[0, col])
        for name, st in styles.items():
            ax.plot(NA, optics[name][key], lw=1.4, label=name.replace('f0', 'f$_0$'), **st)
        level = optics['deplano_ratios'][key]
        ax.axhline(level, color=COLOURS['experiment'], lw=0.9)
        ax.plot([NA_par, NA_z][col], level, 'o', ms=3.5, color=COLOURS['experiment'])
        ax.text(0.03 if col == 0 else 0.97, level, 'Deplano et al.', color=COLOURS['experiment'], fontsize=6.5,
                va='bottom', ha='left' if col == 0 else 'right', transform=ax.get_yaxis_transform())
        ax.set_xlabel('numerical aperture')
        ax.set_ylabel(label)
        ax.set_title(titles[col])
        ax.margins(x=0.02)
        if col == 0:
            ax.legend(loc='upper left', handlelength=1.8)
        panel_letter(ax, 'ab'[col])
    # (c) static splitting
    ax = fig.add_subplot(grid[0, 2])
    ratios = np.array(shaping['ratios'])
    for key, colour, label in (('co-polarised, f_pol=1', COLOURS['co-polarised, no beat'], 'co-polarised'),
                               ('orthogonal, f_pol=1', COLOURS['orthogonal'], 'orthogonal, f$_{pol}$ = 1'),
                               ('orthogonal, f_pol=0.7', COLOURS['f_pol=0.7'], 'orthogonal, f$_{pol}$ = 0.7')):
        ax.plot(ratios, shaping[key]['δ_closed_form_kHz'], color=colour, lw=1.3, label=label)
        ax.plot(ratios[::4], np.array(shaping[key]['δ_kHz'])[::4], 'o', ms=3, mfc='white', mec=colour, mew=0.9)
    ax.set_xlabel('coupling ratio G$_+$/G$_-$')
    ax.set_ylabel('static splitting δ/2π (kHz)')
    ax.set_title('Orthogonal tones shrink the\nsplitting by c$_-^2$ − c$_+^2$')
    ax.set_ylim(0, 17.5)
    ax.legend(loc='lower left', handlelength=1.6)
    ax.text(0.97, 0.97, 'lines: closed form\nmarkers: model', transform=ax.transAxes, ha='right', va='top',
            fontsize=6.5, color='#555555')
    panel_letter(ax, 'c')
    # (d) beat strengths, labelled directly
    ax = fig.add_subplot(grid[1, 0])
    copol = np.array(shaping['co-polarised, f_pol=1']['rates_kHz'])
    orth1 = np.array(shaping['orthogonal, f_pol=1']['rates_kHz'])[:, 0]
    orth7 = np.array(shaping['orthogonal, f_pol=0.7']['rates_kHz'])[:, 0]
    cooling = np.array(shaping['orthogonal, f_pol=1']['cooling_kHz'])
    ax.plot(ratios, copol[:, 0], color=COLOURS['co-polarised, no beat'], lw=1.3, label='co-polarised: local squeezing of x')
    ax.plot(ratios, copol[:, 1], color=COLOURS['co-polarised, no beat'], lw=1.0, ls='--', label='… of y')
    ax.plot(ratios, orth1, color=COLOURS['orthogonal'], lw=1.3, label='orthogonal: two-mode squeezing, f$_{pol}$ = 1')
    ax.plot(ratios, orth7, color=COLOURS['f_pol=0.7'], lw=1.3, label='… f$_{pol}$ = 0.7')
    ax.plot(ratios, np.where(cooling > 0, cooling, np.nan), color='#8c8c8c', lw=1.0, ls=':',
            label='cavity cooling 4(G$_-^2$−G$_+^2$)/κ')
    i = int(np.argmin(np.abs(ratios - 0.55)))
    ax.legend(loc='upper left', fontsize=6, handlelength=1.6)
    ax.set_yscale('log')
    ax.set_ylim(0.3, 2000)
    log_ticks(ax, [1, 10, 100, 1000], axis='y')
    ax.set_xlabel('coupling ratio G$_+$/G$_-$')
    ax.set_ylabel('amplification rate /2π (kHz)')
    ax.set_title(f'The orthogonal beat squeezes two modes,\n'
                 f'{copol[i, 0] / orth1[i]:.0f}× more weakly than the local beat')
    panel_letter(ax, 'd')
    # (e) snapshots of the instantaneous trap over one beat period
    sub = grid[1, 1].subgridspec(1, 2, wspace=0.08)
    for i_s, (scheme, label) in enumerate((('co-polarised', 'co-polarised:\nbreathes'), ('orthogonal', 'orthogonal:\naxes rock'))):
        ax = fig.add_subplot(sub[0, i_s])
        snaps = shaping['snapshots'][scheme]
        t = np.linspace(0, 2 * π, 200)
        Ks = [np.array(K) for K in snaps['K_kHz2']]
        scale = np.sqrt(np.mean([np.trace(K) / 2 for K in Ks]))      # the static trap: same for every snapshot
        colours = plt.cm.viridis(np.linspace(0, 0.9, len(Ks)))
        for K, c in zip(Ks, colours):
            w, v = np.linalg.eigh(K)
            # iso-energy contour ½rᵀKr = const: semi-axes ∝ 1/√(eigenvalue)
            r = 1.2 * (v @ np.diag(scale / np.sqrt(w)) @ np.vstack([np.cos(t), np.sin(t)]))
            ax.plot(r[0], r[1], color=c, lw=0.9)
            soft = v[:, 0] * 1.2 * scale / np.sqrt(w[0]) * 1.15
            ax.plot([-soft[0], soft[0]], [-soft[1], soft[1]], color=c, lw=0.6)
        ax.set_aspect('equal')
        ax.set_xlim(-3.3, 3.3)
        ax.set_ylim(-3.3, 3.3)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('#bbbbbb')
        ax.set_xlabel(label, fontsize=6.5)
        if i_s == 0:
            ax.set_title('One beat period, G$_+$/G$_-$ = 0.525', loc='left')
            panel_letter(ax, 'e', x=-0.3)
            ax.text(0.04, 0.96, 'x →', transform=ax.transAxes, fontsize=6.5, va='top')
        else:
            ax.text(0.96, 0.96, 'colour:\nbeat phase', transform=ax.transAxes, fontsize=6.5, va='top', ha='right')
    # (f) off the node
    ax = fig.add_subplot(grid[1, 2])
    dΦ = np.array(off['dΦ'])
    shifts = np.array(off['shift_kHz'])
    for values, colour, ls, label in (
            (shifts[:, 2], COLOURS['particle 2'], '-', 'Ω$_z$ shift'),
            (shifts[:, 0], COLOURS['orthogonal'], '-', 'Ω$_x$ shift'),
            (np.abs(off['g_xy_kHz']), COLOURS['co-polarised, no beat'], '-', '|g$_{xy}$| (levisim\'s mech-mech term)'),
            (np.array(off['g_xz_kHz']), COLOURS['Deplano θ'], '--', 'largest x–z, y–z term')):
        ax.plot(dΦ, values, color=colour, ls=ls, lw=1.3, label=label)
    ax.legend(loc='upper left', fontsize=6, handlelength=1.6)
    ax.set_xlabel('distance from node, π/2 − Φ (rad)')
    ax.set_ylabel('static term /2π (kHz)')
    ax.set_title('Off the node the carrier adds\nstatic couplings and shifts')
    ax.set_ylim(0, 1.25 * shifts[:, 2].max())
    panel_letter(ax, 'f')
    fig.savefig(FIGURES / 'trap_shaping.png')
    plt.close(fig)


def figure_orthogonal(results):
    orth = results['orthogonal']
    fig = plt.figure(figsize=(7.2, 7.4))
    grid = fig.add_gridspec(3, 3, left=0.07, right=0.98, top=0.95, bottom=0.06, wspace=0.36, hspace=0.72)
    ratios = np.array(orth['ratio_scan_ratios'])
    series = (('orthogonal', COLOURS['orthogonal'], '-', 1.8, 'orthogonal tones'),
              ('co-polarised, no beat', COLOURS['co-polarised, no beat'], '-', 1.3, 'co-polarised, beat removed (PR #16)'),
              ('orthogonal, no beat', COLOURS['orthogonal, no beat'], '-', 1.1, 'orthogonal, beat removed (toy)'),
              ('orthogonal, blue not scattering', COLOURS['toy'], ':', 1.2, 'orthogonal, blue not scattering (toy)'),
              ('orthogonal, beat sign flipped', COLOURS['toy'], '--', 1.0, 'orthogonal, beat sign flipped (toy)'))
    for col, Γ in enumerate((1/3, 0.1)):
        ax = fig.add_subplot(grid[0, col])
        for name, colour, ls, lw, label in series:
            ax.plot(ratios, _curve(orth['ratio_scan'][f'{name} | Γ×{Γ:.3g}']), color=colour, ls=ls, lw=lw, label=label)
        copol = orth['ratio_scan'][f'co-polarised | Γ×{Γ:.3g}']
        first_unstable = min((r['ρ'] for r in copol if r['smallest_grid_multiplier'] > 1), default=None)
        if col == 0:
            ax.text(0.02, 0.97, f'co-polarised with its beat: separable\nbelow G$_+$/G$_-$ = {first_unstable:g}, '
                    'unstable above', transform=ax.transAxes, va='top', fontsize=6.5, color='#555555')
        ax.set_xlabel('coupling ratio G$_+$/G$_-$')
        ax.set_ylabel('log-negativity E$_N$')
        ax.set_ylim(0, _top(_curve(orth['ratio_scan'][f'orthogonal | Γ×{Γ:.3g}']), 1.45 if col == 0 else 1.85))
        best_o = _best_E(orth['ratio_scan'][f'orthogonal | Γ×{Γ:.3g}'])
        best_c = _best_E(orth['ratio_scan'][f'co-polarised, no beat | Γ×{Γ:.3g}'])
        ax.set_title(f'Γ/{1/Γ:.0f}: best E$_N$ {best_o:.2f} orthogonal\nvs {best_c:.2f} beat-free co-polarised')
        if col == 1:
            ax.legend(loc='upper left', fontsize=6, handlelength=1.8)
        panel_letter(ax, 'ab'[col])
    # (c) the stability margin
    ax = fig.add_subplot(grid[0, 2])
    margins = np.array(orth['margin_scan']['margin_kHz'])
    for scheme, colour in (('orthogonal', COLOURS['orthogonal']), ('co-polarised, no beat', COLOURS['co-polarised, no beat'])):
        for Γ, ls in ((1/3, '-'), (0.1, '--')):
            rows = orth['margin_scan'][f'{scheme} | Γ×{Γ:.3g}']
            values = [admissible_E(r) for r in rows]
            ax.plot(margins, values, color=colour, ls=ls, lw=1.3, marker='o', ms=2.5)
            ax.text(margins[-1] + 0.12, values[-1], f'{"orth." if scheme == "orthogonal" else "co-pol."}, Γ/{1/Γ:.0f}',
                    color=colour, fontsize=6.5, va='center')
    ax.axvline(MARGIN / kHz, color='#555555', lw=0.7, ls=':')
    ax.set_xlim(-0.15, margins[-1] + 1.35)
    ax.set_xlabel('required slowest relaxation (kHz)')
    ax.set_ylabel('best E$_N$')
    unconstrained = admissible_E(orth['margin_scan']['orthogonal | Γ×0.333'][0])
    at_margin = admissible_E(orth['margin_scan']['orthogonal | Γ×0.333'][list(margins).index(MARGIN / kHz)])
    ax.set_title(f'Optimum sits at threshold;\na 1 kHz margin costs {1 - at_margin / unconstrained:.0%}')
    panel_letter(ax, 'c')
    # (d) against the heating
    ax = fig.add_subplot(grid[1, 0])
    heat = orth['heating_scan']
    styles = {'orthogonal': (COLOURS['orthogonal'], '-'), 'orthogonal, f_pol=0.7': (COLOURS['f_pol=0.7'], '-'),
              'orthogonal, Deplano θ': (COLOURS['Deplano θ'], '-'),
              'co-polarised, no beat': (COLOURS['co-polarised, no beat'], '-'),
              'co-polarised, no beat, Deplano θ': (COLOURS['co-polarised, no beat'], '--')}
    for name, (colour, ls) in styles.items():
        Γs = np.array(heat[name]['Γ_scale'])
        ax.plot(1 / Γs, heat[name]['E_N'], color=colour, ls=ls, lw=1.3, marker='o', ms=2.5,
                label=name.replace('f_pol', 'f$_{pol}$').replace('co-polarised, no beat', 'co-pol., beat removed'))
    ax.set_xscale('log')
    log_ticks(ax, [1, 2, 3, 5, 10])
    ax.set_xlabel('heating reduction Γ$_{published}$/Γ')
    ax.set_ylabel('best E$_N$')
    t_o, t_c = heat['orthogonal']['threshold_Γ_scale'], heat['co-polarised, no beat']['threshold_Γ_scale']
    ax.set_title(f'Entangled from heating ÷{1/t_o:.1f}\n(orthogonal) vs ÷{1/t_c:.1f} (beat-free)'
                 if t_o and t_c else 'Best E$_N$ against the heating')
    ax.legend(loc='upper left', fontsize=6, handlelength=1.6, bbox_to_anchor=(0.0, 1.0))
    ax.set_ylim(0, 1.6 * max(max(heat[name]['E_N']) for name in styles))
    panel_letter(ax, 'd')
    # (e) cavity angle
    ax = fig.add_subplot(grid[1, 1])
    θs = np.array(orth['θ_scan']['θ_deg'])
    for name, colour in (('orthogonal', COLOURS['orthogonal']), ('co-polarised, no beat', COLOURS['co-polarised, no beat'])):
        ax.plot(θs, [admissible_E(r) for r in orth['θ_scan'][name]], color=colour, lw=1.4, marker='o', ms=2.5,
                label='orthogonal' if name == 'orthogonal' else 'co-polarised,\nbeat removed')
    ax.axvline(np.degrees(θ_DEPLANO), color=COLOURS['experiment'], lw=0.8)
    ax.set_ylim(0, 1.15 * max(admissible_E(r) for r in orth['θ_scan']['orthogonal']))
    ax.text(np.degrees(θ_DEPLANO) + 0.8, 0.97, 'Deplano\net al.', color=COLOURS['experiment'], fontsize=6.5, va='top',
            transform=ax.get_xaxis_transform())
    ax.set_xlabel('cavity axis from the trap y axis, θ (°)')
    ax.set_ylabel('best E$_N$ (heating Γ/3)')
    ax.set_title('Projections and coupling balance\nfavour θ near 45°')
    ax.legend(loc='center left', fontsize=6, handlelength=1.6, bbox_to_anchor=(0.0, 0.62))
    panel_letter(ax, 'e')
    # (f) f_pol
    ax = fig.add_subplot(grid[1, 2])
    f = np.array(orth['f_pol_scan']['f_pol'])
    for Γ, colour in ((1/3, COLOURS['orthogonal']), (0.1, COLOURS['particle 2'])):
        ax.plot(f, [admissible_E(r) for r in orth['f_pol_scan'][f'Γ×{Γ:.3g}']], color=colour, lw=1.4,
                marker='o', ms=2.5, label=f'orthogonal, Γ/{1/Γ:.0f}')
        ax.axhline(_best_E(orth['ratio_scan'][f'co-polarised, no beat | Γ×{Γ:.3g}']), color=colour, lw=0.9, ls=':')
    ax.text(0.5, 0.58, 'dotted: co-polarised, beat removed', transform=ax.transAxes, ha='center', va='top',
            fontsize=6.5, color='#555555')
    ax.set_xlabel('polarisation share f$_{pol}$')
    ax.set_ylabel('best E$_N$')
    ax.set_title('The gain needs the anisotropy\nto come from the polarisation')
    ax.legend(loc='upper left', fontsize=6, handlelength=1.6)
    panel_letter(ax, 'f')
    # (g) polarisation error
    ax = fig.add_subplot(grid[2, 0])
    η = np.array(orth['η_scan']['η_deg'])
    E_η = np.array([admissible_E(r) for r in orth['η_scan']['rows']])
    ax.plot(η, E_η, color=COLOURS['orthogonal'], lw=1.4, marker='o', ms=2.5)
    ax.set_xlabel('polarisation angle error (°)')
    ax.set_ylabel('best E$_N$ (heating Γ/3)')
    ax.set_ylim(0, _top(E_η, 1.25))
    ax.set_title(f'A {η[-1]:g}° polarisation error keeps\n{E_η[-1] / E_η[0]:.0%} of the entanglement' if E_η[0] > 0
                 else 'Tolerance to the polarisation angle')
    panel_letter(ax, 'g')
    # (h), (i) tolerance windows of the two tone parameters
    windows = orth['windows']
    for col, (param, xlabel, title) in enumerate((
            ('shift', 'common shift of both tones (kHz)', 'The cavity lock: a narrow, one-sided\nwindow for the orthogonal scheme'),
            ('offset', 'tone-difference offset (kHz)', 'Lock the tone difference to\n~1 kHz in both schemes'))):
        ax = fig.add_subplot(grid[2, 1 + col])
        x = np.array(windows[f'{param}_kHz'])
        for scheme, colour in (('orthogonal', COLOURS['orthogonal']), ('co-polarised, no beat', COLOURS['co-polarised, no beat'])):
            for Γ, ls in ((1/3, '-'), (0.1, '--')):
                w = windows[f'{scheme} | Γ×{Γ:.3g}']
                E = np.array(w[param], dtype=float)
                slow = np.array(w[f'{param}_relaxation_kHz']) < MARGIN / kHz
                ax.plot(x, np.where(slow, np.nan, E), color=colour, ls=ls, lw=1.3,
                        label=f'{"orth." if scheme == "orthogonal" else "co-pol."}, Γ/{1/Γ:.0f}')
                ax.plot(x, np.where(slow, E, np.nan), color=colour, ls=ls, lw=0.6, alpha=0.5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('E$_N$')
        ax.set_title(title)
        ax.set_ylim(0, 1.45 * max(np.nanmax(np.array(windows[f'{s_} | Γ×{g_:.3g}'][param], dtype=float))
                                  for s_ in ('orthogonal', 'co-polarised, no beat') for g_ in (1/3, 0.1)))
        if col == 0:
            ax.legend(loc='upper left', fontsize=6, handlelength=1.6, ncol=2, columnspacing=0.8)
        else:
            ax.text(0.98, 0.97, 'faint: relaxes slower\nthan 1 kHz', transform=ax.transAxes, ha='right',
                    va='top', fontsize=6.5, color='#555555')
        panel_letter(ax, 'hi'[col])
    fig.savefig(FIGURES / 'orthogonal_polarisation.png')
    plt.close(fig)


def figure_mech_mech(results):
    mm = results['mech_mech']
    fig = plt.figure(figsize=(7.2, 2.7))
    grid = fig.add_gridspec(1, 3, left=0.07, right=0.98, top=0.82, bottom=0.17, wspace=0.38)
    keys = (('orthogonal | Γ×0.333', COLOURS['orthogonal'], 'orthogonal, Γ/3'),
            ('co-polarised, no beat | Γ×0.1', COLOURS['co-polarised, no beat'], 'co-polarised, beat removed, Γ/10'))
    ax = fig.add_subplot(grid[0, 0])
    g = np.array(mm['g_xy_kHz'])
    for key, colour, label in keys:
        rows = mm[f'static coupling | {key}']
        ax.plot(g, _curve(rows), color=colour, lw=1.4, label=label)
        ax.plot(g, _curve(rows, 'E_tweezer_axes'), color=colour, lw=1.0, ls='--')
    ax.set_xlabel('toy static coupling g$_{xy}$/2π (kHz)')
    ax.set_ylabel('best E$_N$')
    ax.set_title('A static x–y coupling only rotates\nthe modes: at best +1%')
    ax.set_ylim(0, _top(np.concatenate([_curve(mm[f'static coupling | {k}']) for k, _, _ in keys]), 1.75))
    ax.plot([], [], color='#555555', lw=1.4, label='normal-mode partition')
    ax.plot([], [], color='#555555', lw=1.0, ls='--', label='tweezer-axis partition')
    ax.legend(loc='upper left', fontsize=6, handlelength=1.6)
    panel_letter(ax, 'a')
    ax = fig.add_subplot(grid[0, 1])
    single = mm['single tone']
    for Γ, colour, label in ((1e-3, COLOURS['orthogonal'], 'heating × 10$^{-3}$'), (1.0, COLOURS['particle 2'], 'published heating')):
        ax.plot(single['dΦ'], [r['ν̃_min'] for r in single[f'Γ×{Γ:g}']], color=colour, lw=1.4, marker='o', ms=2.5, label=label)
    ax.axhline(1, color='#555555', lw=0.8, ls=':')
    ax.text(0.98, 1.0, 'entangled below', fontsize=6.5, color='#555555', va='top', ha='right',
            transform=ax.get_yaxis_transform())
    ax.set_yscale('log')
    log_ticks(ax, [0.8, 1, 1.5, 2, 3, 5], axis='y')
    ax.set_ylim(0.75, 6)
    ax.set_xlabel('distance from node, π/2 − Φ (rad)')
    ax.set_ylabel('smallest ν̃$_-$ (best red detuning)')
    ax.set_title('One tone stays separable at every\nnode position, even without heating')
    ax.legend(loc='upper left', fontsize=6, handlelength=1.6)
    panel_letter(ax, 'b')
    ax = fig.add_subplot(grid[0, 2])
    dΦ = np.array(mm['dΦ'])
    for key, colour, label in keys:
        ax.plot(dΦ, _curve(mm[f'off node | {key}']), color=colour, lw=1.4, marker='o', ms=2.5, label=label)
        if f'node, couplings × sinΦ | {key}' in mm:
            ax.plot(dΦ, _curve(mm[f'node, couplings × sinΦ | {key}']), color=colour, lw=1.0, ls='--')
    ax.plot([], [], color='#555555', lw=1.0, ls='--', label='at a node, couplings × sinΦ')
    for name, r in mm['Γz sensitivity'].items():
        d = float(name.split('dΦ=')[1].split(' |')[0])
        ax.plot(d, admissible_E(r), marker='_', ms=8, color=COLOURS['orthogonal'], mew=1.2)
    ax.text(0.02, 0.03, 'ticks: z heating ×0.4 and ×1.6', transform=ax.transAxes, va='bottom', fontsize=6.5,
            color='#555555')
    ax.set_xlabel('distance from node, π/2 − Φ (rad)')
    ax.set_ylabel('best E$_N$ (z included)')
    ax.set_title('The node is optimal; moving off\nit costs entanglement slowly')
    ax.set_ylim(0, _top(np.concatenate([_curve(mm[f'off node | {k}']) for k, _, _ in keys]), 1.35))
    ax.legend(loc='upper right', fontsize=6, handlelength=1.6)
    panel_letter(ax, 'c')
    fig.savefig(FIGURES / 'mech_mech_and_node.png')
    plt.close(fig)


def figure_two_particles(results):
    tp = results['two_particles']
    fig = plt.figure(figsize=(7.2, 5.0))
    grid = fig.add_gridspec(2, 2, left=0.08, right=0.98, top=0.92, bottom=0.09, wspace=0.28, hspace=0.62)
    copol_key, orth_key = 'co-polarised, no beat | Γ×0.1', 'orthogonal | Γ×0.333'
    optima = tp['single-particle optima']
    ident = tp['identical'][f'{copol_key} | same node sign']
    # (a) splitting ratio
    ax = fig.add_subplot(grid[0, 0])
    rows = tp['δ ratio'][copol_key]
    a = np.array([r['δ2_kHz'] / r['δ1_kHz'] for r in rows])
    ax.plot(a, _curve(rows, 'E_11'), color=COLOURS['particle 1'], lw=1.4, marker='o', ms=2.5, label='particle 1 (x$_1$:y$_1$)')
    ax.plot(a, _curve(rows, 'E_22'), color=COLOURS['particle 2'], lw=1.4, marker='o', ms=2.5, label='particle 2 (x$_2$:y$_2$)')
    ax.plot(a, [r['E_2_alone'] for r in rows], color=COLOURS['particle 2'], lw=0.9, ls='--', label='particle 2 alone')
    ax.axhline(optima[copol_key]['E_N'], color=COLOURS['particle 1'], lw=0.9, ls=':', label='particle 1 alone')
    n_dark = np.mean(ident['n'])
    ax.annotate(f'identical traps: dark modes,\n{n_dark / 1e7:.1f}×10$^7$ phonons', xy=(1.0, 0.0), xytext=(0.36, 0.40),
                fontsize=6.5, arrowprops=dict(arrowstyle='-', color='#555555', lw=0.7))
    ax.set_xscale('log')
    log_ticks(ax, [0.2, 0.5, 1, 2, 3])
    ax.set_xlabel('splitting of particle 2 / particle 1, δ$_2$/δ$_1$')
    ax.set_ylabel('E$_N$')
    ax.set_title('Same tone frequencies: the particles need\nsplittings ≳ 2× apart (co-polarised, Γ/10)')
    ax.set_ylim(0, 0.95)
    ax.legend(loc='upper left', fontsize=6, handlelength=1.6, ncol=2, columnspacing=0.8)
    panel_letter(ax, 'a')
    # (b), (c) common shifts
    for col, (key, title) in enumerate(((f'{copol_key} | symmetric', 'Shifting the particles apart in the cavity\nline works once their sidebands separate'),
                                        (f'{orth_key} | second below', 'Orthogonal tones: a second particle fits\nonly in a narrow window (Γ/3)'))):
        ax = fig.add_subplot(grid[0, 1] if col == 0 else grid[1, 0])
        rows = tp['common shift'][key]
        Δs = np.array([r['Δs_kHz'] for r in rows])
        ax.plot(Δs, _curve(rows, 'E_11'), color=COLOURS['particle 1'], lw=1.4, marker='o', ms=2, label='particle 1')
        ax.plot(Δs, _curve(rows, 'E_22'), color=COLOURS['particle 2'], lw=1.4, marker='o', ms=2, label='particle 2')
        ax.plot(Δs, [r['E_1_alone'] for r in rows], color=COLOURS['particle 1'], lw=0.9, ls=':', label='particle 1 alone')
        ax.plot(Δs, [r['E_2_alone'] for r in rows], color=COLOURS['particle 2'], lw=0.9, ls='--', label='particle 2 alone')
        for d in [d for d, r in zip(Δs, rows) if not r['stable']]:
            ax.axvspan(d - 0.9, d + 0.9, color='#dddddd', lw=0, zorder=0)
        δ = tp['δ ratio'][copol_key if col == 0 else orth_key][0]['δ1_kHz']
        ax.axvline(δ, color='#555555', lw=0.7, ls=':')
        ax.text(δ, 0.995, ' Δs = δ', transform=ax.get_xaxis_transform(), fontsize=6.5, color='#555555', va='top')
        ax.set_xlabel('shift between the particles\' tones, Δs/2π (kHz)')
        ax.set_ylabel('E$_N$ (x$_n$:y$_n$)')
        ax.set_title(title)
        top = np.nanmax(np.concatenate([_curve(rows, 'E_11'), _curve(rows, 'E_22'),
                                        np.array([r['E_1_alone'] for r in rows], dtype=float)]))
        ax.set_ylim(0, 1.6 * top)
        ax.text(0.98, 0.97 if col == 0 else 0.70, 'grey: unstable', transform=ax.transAxes, ha='right', va='top',
                fontsize=6.5, color='#555555')
        ax.legend(loc='upper right' if col == 1 else 'lower right', fontsize=6, handlelength=1.6, ncol=2,
                  columnspacing=0.8)
        panel_letter(ax, 'bc'[col])
    # (d) one tone per tweezer
    ax = fig.add_subplot(grid[1, 1])
    ratios = np.array(tp['inter-particle ratios'])
    for Γ, colour in ((1/3, COLOURS['orthogonal']), (0.1, COLOURS['particle 2'])):
        for model, ls in (('recoil ∝ mass', '-'), ('equal', '--')):
            rows = tp['inter-particle'][f'Deplano θ | {model} | Γ×{Γ:.3g}']
            ax.plot(ratios, _curve(rows, 'E_1_2'), color=colour, ls=ls, lw=1.3,
                    label=f'Γ/{1/Γ:.0f}, particle-2 heating {"∝ its mass" if model == "recoil ∝ mass" else "unchanged"}')
    published = max(np.nan_to_num(_curve(tp['inter-particle'][f'{t} | {m} | Γ×1'], 'E_1_2')).max()
                    for t in ('Deplano θ', 'θ=45°') for m in ('recoil ∝ mass', 'equal'))
    ax.text(0.02, 0.60, f'published heating: E$_N$ = {published:.2f}\nfor every G$_+$/G$_-$', transform=ax.transAxes,
            ha='left', va='top', fontsize=6.5, color='#555555')
    ax.set_xlabel('coupling ratio G$_+$/G$_-$ (particle 2 / particle 1)')
    ax.set_ylabel('E$_N$ between the particles (1 | 2)')
    ax.set_title('One tone per tweezer entangles the two\nparticles, with no beat at all')
    ax.set_ylim(0, 1.45 * np.nanmax(_curve(tp['inter-particle']['Deplano θ | recoil ∝ mass | Γ×0.1'], 'E_1_2')))
    ax.legend(loc='upper left', fontsize=6, handlelength=1.6)
    panel_letter(ax, 'd')
    fig.savefig(FIGURES / 'two_particles.png')
    plt.close(fig)


if __name__ == '__main__':
    FIGURES.mkdir(exist_ok=True)
    style()
    if '--plot-only' in sys.argv:
        results = from_jsonable(json.loads(RESULTS.read_text()))
    elif '--recompute' in sys.argv:
        # redo some sections, e.g. --recompute mech_mech,two_particles, keeping the rest
        results = json.loads(RESULTS.read_text())
        sections = sys.argv[sys.argv.index('--recompute') + 1].split(',')
        t0 = time.time()
        for name in sections:
            if name == 'two_particles':
                results[name] = compute_two_particles(from_jsonable(results)['orthogonal'])
            else:
                results[name] = globals()[f'compute_{name}']()
            print(f'  {name}: recomputed', flush=True)
        results['runtime_s'] = results.get('runtime_s', 0)
        results['recomputed'] = {'sections': sections, 'runtime_s': time.time() - t0}
        RESULTS.write_text(compact_json(to_jsonable(results)) + '\n')
        results = from_jsonable(json.loads(RESULTS.read_text()))
    else:
        print('computing ...', flush=True)
        results = compute()
        RESULTS.write_text(compact_json(to_jsonable(results)) + '\n')
        results = from_jsonable(json.loads(RESULTS.read_text()))
    figure_trap_shaping(results)
    figure_orthogonal(results)
    figure_mech_mech(results)
    figure_two_particles(results)
    print(f'done in {results["runtime_s"]:.0f} s; figures in {FIGURES}')

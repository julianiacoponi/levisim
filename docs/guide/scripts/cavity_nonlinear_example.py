"""
Figures and numbers of the nonlinear model in docs/guide/cavity_simulations.md

    python docs/guide/scripts/cavity_nonlinear_example.py

1. the unstable 2×2 system (built-in phases 0°, 90°, shelf cavity): growth at max_growth_rate and
   saturation, with the nonlinear terms switched to their linearisation one at a time, and with the
   z motion started thermal (docs/guide/figures/cavity_nonlinear_growth.png)
2. validation (docs/guide/figures/cavity_nonlinear_validation.png): the Duffing softening of the
   tweezer against the exact period and FreeSpaceSimulation, the measured Δt convergence of the
   split step, and the convergence of the homodyne spectrum to the linear model as the bath cools
3. off-node systems: the true equilibrium against MultiTrap's linearisation at the trap centre
   (docs/guide/tables/cavity_nonlinear_off_node.csv)

Seeded; about a minute with numba.
"""
import csv
import time
from itertools import pairwise
from pathlib import Path

import matplotlib
import matplotlib.ticker
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.linalg import solve_continuous_lyapunov, solve_discrete_lyapunov

from levisim.core.plotting import set_plt_rcParams_to_xmgrace_style
from levisim.core.units import K, degrees, mbar, ms, nm, μs
from levisim.multiphysics.covariance import quadrature_drift_matrix
from levisim.multiphysics.multitrap import MultiTrap
from levisim.multiphysics.utils import shelf
from levisim.physics.bead import Bead
from levisim.simulation import CavitySimulation, FreeSpaceSimulation, Homodyne, Position
from levisim.simulation.cavity import _psd_square_root, van_loan_discretisation
from levisim.simulation.nonlinear_cavity import NonlinearCavityModel

HERE = Path(__file__).resolve().parents[1]
FIGURES, TABLES = HERE / 'figures', HERE / 'tables'
start_all = time.time()


def physical(num_traps, axes, Φ=None, **kwargs):
    if Φ is not None:
        kwargs['beads'] = [Bead(r=(60.1 + 0.1 * i) * nm, Φ=Φ[i]) for i in range(num_traps)]
    return MultiTrap(num_traps=num_traps, Cartesian_axes_per_trap=axes, cavity_kwargs=shelf.DEFAULT_cavity_kwargs,
                     gas_kwargs=kwargs.pop('gas_kwargs', shelf.DEFAULT_gas_kwargs), **kwargs)


def energies(sim):
    s = sim.states
    return 0.25 * (s[..., 2::2]**2 + s[..., 3::2]**2)


# ------------------------------------------------------------------------------------------------
# 1. the unstable 2×2 system
# ------------------------------------------------------------------------------------------------
mtrap22 = physical(2, 2)
λ = mtrap22.max_growth_rate
print(f'2×2 (φ2 = 90°): max_growth_rate = {λ:.1f} s⁻¹')
K_traj, T_steps = 4, 120_000


def unstable_run(z_thermal=False, **kwargs):
    model = NonlinearCavityModel(mtrap22, linearised_terms=kwargs.get('linearised_terms', ()))
    u0 = np.zeros((K_traj, model.n))
    if z_thermal:
        rng = np.random.default_rng(102)
        for p in range(model.num_particles):
            iz = 2 + 6 * p + 4
            variance = model.diffusion_per_axis[p, 2] / (2 * model.Γ[p, 2])
            u0[:, iz:iz + 2] = rng.standard_normal((K_traj, 2)) * np.sqrt(variance)
    return CavitySimulation(mtrap22, model='nonlinear', num_samples=T_steps, Δt=0.1 * μs, seed=2,
                            num_trajectories=K_traj, initial_state=u0, store_state=True, detection=[Homodyne()], **kwargs)


variants = [
    ('full nonlinear model', {}),
    ('harmonic tweezer', {'linearised_terms': ('elements',)}),
    ('linear coherent scattering', {'linearised_terms': ('coherent_scattering', 'cavity')}),
    ('full, z thermal at start', {'z_thermal': True}),
]
growth = {}
for name, kwargs in variants:
    with np.errstate(over='ignore', invalid='ignore'):
        sim = unstable_run(**kwargs)
        E = energies(sim)
    Exy = E[..., [0, 1, 3, 4]].sum(-1)
    t = np.arange(Exy.shape[1]) * sim.Δt
    growth[name] = (t, Exy, sim)
    with np.errstate(invalid='ignore', divide='ignore'):
        peak = np.nanmax(np.where(np.isfinite(Exy), Exy, np.nan), axis=1)
        left = int(np.sum(~np.isfinite(Exy[:, -1]) | (Exy[:, -1] > 1e14)))
        print(f'  {name:28s}: peak x-y energy 10^{np.log10(np.nanmedian(peak)):.2f} phonons (median of {K_traj});'
              f' {left} of {K_traj} trajectories left the trap')
t, Exy, full = growth['full nonlinear model']
rates = []
for k in range(K_traj):
    i0, i1 = np.argmax(Exy[k] > 1e4), np.argmax(Exy[k] > 1e7)
    rates.append(np.polyfit(t[i0:i1], np.log(Exy[k, i0:i1]), 1)[0] / 2)
print(f'  fitted growth rate {np.mean(rates):.1f} ± {np.std(rates) / np.sqrt(K_traj):.1f} s⁻¹ (max_growth_rate {λ:.1f})')
zpf = full.nonlinear_model.zpf.ravel()
r = full.states[..., 2::2] * zpf
imax = np.argmax(Exy, axis=1)
amplitudes = np.array([np.abs(r[k, max(i - 300, 0):i + 300]).max(axis=0) for k, i in enumerate(imax)])
k_d = mtrap22.traps[0].tweezer.k / np.sqrt(2)
print(f'  saturation at {t[imax].mean() / ms:.2f} ms: |y1|, |y2| ≈ {amplitudes[:, 1].mean() / nm:.0f}, {amplitudes[:, 4].mean() / nm:.0f} nm'
      f' (k·d·r ≈ {k_d * amplitudes[:, 1].mean():.2f}, {k_d * amplitudes[:, 4].mean():.2f} rad); |z1| ≈ {amplitudes[:, 2].mean() / nm:.0f} nm')

set_plt_rcParams_to_xmgrace_style()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
colours = ['black', 'tab:blue', 'tab:orange', 'tab:green']
for (name, _), colour in zip(variants, colours):
    tt, E_xy, _ = growth[name]
    with np.errstate(invalid='ignore', divide='ignore'):
        mean = np.exp(np.nanmean(np.log(np.where(np.isfinite(E_xy) & (E_xy > 0), E_xy, np.nan)), axis=0))
    ax1.semilogy(tt / ms, mean, color=colour, lw=1.4, label=name)
guide_t = np.linspace(0.6, 3.2, 10) * ms
ax1.semilogy(guide_t / ms, 3e3 * np.exp(2 * λ * (guide_t - guide_t[0])), 'k--', lw=1,
             label=r'$\propto e^{2\lambda t}$, $\lambda$ = max_growth_rate')
ax1.set_ylim(1, 1e11)
ax1.set_xlim(0, T_steps * 0.1 * μs / ms)
ax1.set_xlabel('time [ms]')
ax1.set_ylabel('x and y energy of both beads [phonons]')
ax1.legend(frameon=False, fontsize=9, loc='upper right')
ax1.set_title('(a) growth, then saturation', loc='left', fontsize=11)
window = slice(int(2.5 * ms / full.Δt), int(9 * ms / full.Δt))
ax2.plot(t[window] / ms, r[0, window, 1] / nm, color='tab:blue', lw=0.4, label='$y_1$')
ax2.plot(t[window] / ms, r[0, window, 2] / nm, color='tab:red', lw=0.6, label='$z_1$')
ax2.set_xlabel('time [ms]')
ax2.set_ylabel('displacement from equilibrium [nm]')
ax2.legend(frameon=False, fontsize=9, loc='upper right')
ax2.set_title('(b) energy moves from y into z', loc='left', fontsize=11)
fig.tight_layout()
fig.savefig(FIGURES / 'cavity_nonlinear_growth.png', dpi=150)
plt.close(fig)

# ------------------------------------------------------------------------------------------------
# 2. validation
# ------------------------------------------------------------------------------------------------
# (a) Duffing softening of the tweezer, cavity coupling off
gas = {'P': 1e-12 * mbar, 'T': 300 * K, 'vth': 500}
mtrapD = physical(1, 3, gas_kwargs=gas, photon_recoil=False, trap_kwargs=[{'Γ_fbx': 0, 'Γ_fby': 0, 'Γ_fbz': 0}])
trapD = mtrapD.traps[0]
m, w = trapD.bead.m, trapD.tweezer.wx0
U0 = np.real(trapD.α_eff) / 4 * trapD.tweezer.E0**2


def exact_frequency(a):
    U = lambda x: -U0 * np.exp(-2 * x**2 / w**2)
    return 1 / (4 * quad(lambda θ: a * np.cos(θ) / np.sqrt(2 * (U(a) - U(a * np.sin(θ))) / m), 0, np.pi / 2, limit=200)[0])


def crossing_frequency(x, dt):
    sign = np.signbit(x)
    i = np.where(sign[1:] != sign[:-1])[0]
    tc = (i + x[i] / (x[i] - x[i + 1])) * dt
    return (len(tc) - 1) / 2 / (tc[-1] - tc[0])


amplitudes_D = np.array([25, 50, 100, 150, 200, 300]) * nm
duffing = []
for a in amplitudes_D:
    model = NonlinearCavityModel(mtrapD, terms=('elements',))
    u0 = np.zeros(model.n)
    u0[2] = a / model.zpf[0, 0]
    sim = CavitySimulation(mtrapD, model='nonlinear', terms=('elements',), num_samples=20_000, Δt=0.1 * μs, seed=1,
                           initial_state=u0, store_state=True, detection=[Position('x1')])
    f_split = crossing_frequency(sim.absolute_states[0, :, 2] * model.zpf[0, 0], 0.1 * μs)
    f_free = {}
    for dt in (0.1 * μs, 0.01 * μs):
        free = FreeSpaceSimulation(trap=trapD, num_samples=round(2 * ms / dt), Δt=dt, X0=np.array([a, 0, 0, 0, 0, 0]),
                                   seed=1, zero_mean=False)
        f_free[dt] = crossing_frequency(free.X[:, 0], dt)
    duffing.append((a, exact_frequency(a), f_split, f_free[0.1 * μs], f_free[0.01 * μs]))
duffing = np.array(duffing)
f0 = trapD.Ωj['x'] / (2 * np.pi)
print(f'Duffing softening of x (cavity coupling off), Ωx/2π = {f0 / 1e3:.4f} kHz')
for a, fe, fs, f1, f2 in duffing:
    print(f'  a = {a / nm:5.0f} nm: exact {fe / 1e3:.4f}, split step (Δt 0.1 μs) {fs / 1e3:.4f}, FreeSpaceSimulation'
          f' {f1 / 1e3:.4f} (Δt 0.1 μs) / {f2 / 1e3:.4f} (Δt 0.01 μs) kHz')

# (b) measured Δt convergence on a shared Brownian path
mtrap_off = physical(1, 3, Φ=[np.pi / 4])
model_off = NonlinearCavityModel(mtrap_off)


def strong_errors(model, T, δ, K_paths, levels, amplitude, seed=0):
    system = model.quadrature_system
    Φδ, Qδ = van_loan_discretisation(system.A, system.diffusion_matrix, δ)
    rng = np.random.default_rng(seed)
    ξ = rng.standard_normal((round(T / δ), K_paths, model.n)) @ _psd_square_root(Qδ).T
    u0 = amplitude * rng.standard_normal((K_paths, model.n)) @ np.linalg.cholesky(model.steady_state_covariance()).T
    R = model.remainder
    paths = {}
    for level in levels:
        half = 2**level
        h = 2 * half * δ
        steps = ξ.shape[0] // (2 * half)
        u, j = u0.copy(), 0
        for _ in range(half):
            u = u @ Φδ.T + ξ[j]
            j += 1
        for s in range(steps):
            k1 = R(u)
            k2 = R(u + h / 2 * k1)
            k3 = R(u + h / 2 * k2)
            k4 = R(u + h * k3)
            u = u + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            for _ in range(2 * half if s < steps - 1 else half):
                u = u @ Φδ.T + ξ[j]
                j += 1
        paths[level] = u
    reference = paths[levels[0]]
    scale = np.sqrt(np.mean(reference**2, axis=0))
    h = np.array([2**level * 2 * δ for level in levels[1:]])
    error = np.array([np.sqrt(np.mean(np.sum(((paths[level] - reference) / scale)**2, axis=1))) for level in levels[1:]])
    return h, error


δ = 0.0078125 * μs / 2
order_data = {}
for amplitude in (1.0, 10.0):
    h, error = strong_errors(model_off, 8 * μs, δ, 128, [0, 1, 2, 3, 4, 5, 6], amplitude)
    small, large = np.polyfit(np.log(h[:3]), np.log(error[:3]), 1)[0], np.polyfit(np.log(h[-3:]), np.log(error[-3:]), 1)[0]
    order_data[amplitude] = (h, error, small, large)
    print(f'Δt convergence (1×3, Φ = π/4, initial amplitude {amplitude:g}× thermal): strong errors '
          + ', '.join(f'{e:.2e}' for e in error) + ' at h = ' + ', '.join(f'{x / μs:g}' for x in h)
          + f' μs; slopes {small:.2f} (small h), {large:.2f} (large h)')
# Heun's stationary covariance error at the same Δt (linear part alone)
A, D = model_off.quadrature_system.A, model_off.quadrature_system.diffusion_matrix
V = solve_continuous_lyapunov(A, -D)
for h in (0.1 * μs, 0.025 * μs):
    I = np.eye(len(A))
    M = I + h * A + (h * A) @ (h * A) / 2
    G = I + h * A / 2
    Vh = solve_discrete_lyapunov(M, G @ D @ G.T * h)
    print(f'  Heun at Δt = {h / μs:g} μs: stationary variance error up to {np.max(np.abs(np.diag(Vh) / np.diag(V) - 1)):.2e}'
          ' (split step: exact for the linear part)')

# (c) homodyne spectrum against the linear model as the bath cools
spectra = {}
bands = [(1e3, 100e3), (100e3, 160e3), (160e3, 400e3), (1e6, 4.9e6)]
for T in (300, 0.03):
    gas_T = dict(shelf.DEFAULT_gas_kwargs, T=T * K, P=shelf.DEFAULT_gas_kwargs['P'] * T / 300)
    mt = physical(1, 2, gas_kwargs=gas_T, photon_recoil=False)
    nl = CavitySimulation(mt, model='nonlinear', damping='symmetric', num_samples=2**17, Δt=0.1 * μs, seed=21,
                          num_trajectories=16, detection=[Homodyne()], store_state=True)
    lin = CavitySimulation(mt, model='linear', num_samples=16, Δt=0.1 * μs, seed=1, detection=[Homodyne()])
    f, S = nl.averaged_psd('homodyne', num_segments=16)
    theory = lin.record_spectrum('homodyne', f[1:], kind='discrete')
    z_rms = np.std(nl.states[..., 7]) * nl.nonlinear_model.zpf[0, 2]
    spectra[T] = (f[1:], S[1:] / theory, z_rms)
    medians = [np.median((S[1:] / theory)[(f[1:] > lo) & (f[1:] < hi)]) for lo, hi in bands]
    print(f'homodyne / linear theory at T = {T:g} K (z rms {z_rms / nm:.3g} nm): band medians '
          + ', '.join(f'{lo / 1e3:g}-{hi / 1e3:g} kHz: {v:.3f}' for (lo, hi), v in zip(bands, medians)))

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
ax = axes[0]
a_fine = np.linspace(5, 320, 80) * nm
ax.plot(a_fine / nm, [(exact_frequency(a) - f0) / 1e3 for a in a_fine], 'k-', lw=1.2, label='exact period (1D Gaussian well)')
ax.plot(duffing[:, 0] / nm, (duffing[:, 2] - f0) / 1e3, 'o', mfc='none', mec='tab:blue', ms=8,
        label='CavitySimulation, nonlinear (Δt = 0.1 μs)')
ax.plot(duffing[:, 0] / nm, (duffing[:, 3] - f0) / 1e3, 's', mfc='none', mec='tab:orange', ms=6, label='FreeSpaceSimulation, Δt = 0.1 μs')
ax.plot(duffing[:, 0] / nm, (duffing[:, 4] - f0) / 1e3, 'x', color='tab:green', ms=6, label='FreeSpaceSimulation, Δt = 0.01 μs')
ax.set_xlabel('x amplitude [nm]')
ax.set_ylabel(r'frequency shift $(\Omega_x(a) - \Omega_x)/2\pi$ [kHz]')
ax.legend(frameon=False, fontsize=8, loc='lower left')
ax.set_title('(a) tweezer softening, cavity coupling off', loc='left', fontsize=11)
ax = axes[1]
for amplitude, marker in ((1.0, 'o'), (10.0, 's')):
    h, error, _, _ = order_data[amplitude]
    ax.loglog(h / μs, error, marker + '-', mfc='none', ms=6, label=f'initial amplitude {amplitude:g}× thermal')
hh = np.array([0.03, 0.5])
ax.loglog(hh, 2e-4 * (hh / 0.5)**2, 'k--', lw=1, label='order 2')
ax.loglog(hh, 3e-6 * (hh / 0.03)**1, 'k:', lw=1, label='order 1')
ax.set_xticks([0.02, 0.05, 0.1, 0.2, 0.5])
ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%g'))
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_xlabel('split step h [μs]')
ax.set_ylabel('strong error (rms, relative to the state rms)')
ax.legend(frameon=False, fontsize=8, loc='upper left')
ax.set_title('(b) measured Δt convergence (1×3, Φ = π/4)', loc='left', fontsize=11)
ax = axes[2]
for T, colour in ((300, 'tab:red'), (0.03, 'tab:blue')):
    f, ratio, z_rms = spectra[T]
    edges = np.logspace(np.log10(5e3), np.log10(4.9e6), 60)
    centres, values = [], []
    for lo, hi in pairwise(edges):
        sel = (f > lo) & (f < hi)
        if sel.any():
            centres.append(np.sqrt(lo * hi))
            values.append(np.median(ratio[sel]))
    ax.semilogx(np.array(centres) / 1e3, values, '-', color=colour, lw=1.4, label=f'T = {T:g} K (z rms {z_rms / nm:.3g} nm)')
ax.axhline(1, color='k', lw=0.8)
ax.set_xlabel('frequency [kHz]')
ax.set_ylabel('nonlinear homodyne PSD / linear theory')
ax.legend(frameon=False, fontsize=8, loc='upper right')
ax.set_title('(c) small amplitude: the linear model', loc='left', fontsize=11)
fig.tight_layout()
fig.savefig(FIGURES / 'cavity_nonlinear_validation.png', dpi=150)
plt.close(fig)


# ------------------------------------------------------------------------------------------------
# 3. off-node: the true equilibrium against MultiTrap's linearisation at the trap centre
# ------------------------------------------------------------------------------------------------
def mode_frequencies(M):
    ev = np.linalg.eigvals(M)
    ev = ev[ev.imag > 0]
    order = np.argsort(ev.imag)
    return ev.imag[order] / (2 * np.pi), -ev.real[order]


cases = [
    ('1×2, Φ = 45°', 1, 2, [45], None),
    ('1×3, Φ = 45°', 1, 3, [45], None),
    ('1×3, Φ = 30°', 1, 3, [30], None),
    ('2×2, Φ = 45°, 52°, φ2 = 0°', 2, 2, [45, 52], {2: 0.0}),
    ('2×2, Φ = 45°, 52°, φ2 = 90°', 2, 2, [45, 52], None),
    ('3×2, Φ = 45°, 52°, 59°, φn = 0°', 3, 2, [45, 52, 59], {2: 0.0, 3: 0.0}),
]
rows = []
print("Off-node: true equilibrium vs MultiTrap (linearised at the trap centre), damping='symmetric'")
for name, N, axes_per_trap, Φ_deg, φn in cases:
    kwargs = {'φn_dict': φn} if φn else {}
    mt = physical(N, axes_per_trap, Φ=[p * degrees for p in Φ_deg], **kwargs)
    model = NonlinearCavityModel(mt, damping='symmetric')
    idx = model.modelled_indices()
    J = model.jacobian[np.ix_(idx, idx)]
    A = quadrature_drift_matrix(mt.drift_matrix)
    fA, γA = mode_frequencies(A)
    fJ, γJ = mode_frequencies(J)
    α_MT = (mt.cavity.A0 + 1j * mt.cavity.P0) / 2
    row = {
        'system': name,
        'r_eq [nm]': '; '.join('(' + ', '.join(f'{v:.1f}' for v in r) + ')' for r in model.r_eq / nm),
        '|α| MultiTrap': f'{abs(α_MT):.0f}', '|α| equilibrium': f'{abs(model.α_eq):.0f}',
        'arg change [°]': f'{np.degrees(np.angle(model.α_eq / α_MT)):.1f}',
        'drift-matrix difference': f'{np.abs(J - A).max() / np.abs(A).max():.3f}',
        'mode frequency shifts [kHz]': '; '.join(f'{v:+.3f}' for v in (fJ[:-1] - fA[:-1]) / 1e3),
        'mode damping MultiTrap [s⁻¹]': '; '.join(f'{v:.0f}' for v in γA[:-1]),
        'mode damping equilibrium [s⁻¹]': '; '.join(f'{v:.0f}' for v in γJ[:-1]),
        'max growth MultiTrap [s⁻¹]': f'{mt.max_growth_rate:.0f}',
        'max growth equilibrium, modelled axes [s⁻¹]': f'{np.max(np.linalg.eigvals(J).real):.0f}',
        'max growth equilibrium, all axes [s⁻¹]': f'{model.quadrature_system.largest_growth_rate():.1f}',
    }
    rows.append(row)
    print('  ' + ' | '.join(f'{k}: {v}' for k, v in row.items()))
with open(TABLES / 'cavity_nonlinear_off_node.csv', 'w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(f'done in {time.time() - start_all:.0f} s')

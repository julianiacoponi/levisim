"""
Local HTTP server and JSON API for the interactive ``levisim`` Three.js web labs.

Launch from the command line via::

    levisim-web                          # serves on http://127.0.0.1:8765 and opens browser
    levisim-web --port 8080 --no-open    # serve without opening a browser tab
    python -m levisim.web --lab tweezer  # open directly to the free-space tweezer lab

Only the Python standard library is used for HTTP serving; physics calculations
delegate directly to ``levisim.physics`` and ``levisim.analysis``.
"""

from __future__ import annotations

import argparse
import http.server
import json
import mimetypes
import os
import sys
import urllib.parse
import webbrowser
from pathlib import Path

import numpy as np

from levisim.core.logger import logger
from levisim.core.units import Hz, cm, degrees, kHz, mbar, mW, nm, μm, μs

__all__ = [
    'STATIC_DIR',
    'THREE_CDN_PREFIX',
    'compute_tweezer_spectra',
    'compute_cavity_spectra',
    'run_tweezer_simulation',
    'dispatch_request',
    'create_server',
    'serve',
    'main',
]

STATIC_DIR = Path(__file__).resolve().parent / 'static'
THREE_CDN_PREFIX = 'https://cdn.jsdelivr.net/npm/three@0.187.0/'

# Explicit MIME types so ES modules load reliably across platforms
MIME_OVERRIDES = {
    '.js': 'application/javascript; charset=utf-8',
    '.mjs': 'application/javascript; charset=utf-8',
    '.css': 'text/css; charset=utf-8',
    '.html': 'text/html; charset=utf-8',
    '.json': 'application/json; charset=utf-8',
    '.svg': 'image/svg+xml',
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
}


def _float_param(params: dict, key: str, default: float, *, min_value: float | None = None, max_value: float | None = None) -> float:
    """Extracts and validates a finite floating-point parameter from a request dict."""
    raw = params.get(key, default)
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        msg = f'parameter {key!r} must be a number, got {raw!r}'
        logger.error(msg)
        raise ValueError(msg) from exc
    if not np.isfinite(value):
        msg = f'parameter {key!r} must be finite, got {value}'
        logger.error(msg)
        raise ValueError(msg)
    if min_value is not None and value < min_value:
        msg = f'parameter {key!r} must be >= {min_value}, got {value}'
        logger.error(msg)
        raise ValueError(msg)
    if max_value is not None and value > max_value:
        msg = f'parameter {key!r} must be <= {max_value}, got {value}'
        logger.error(msg)
        raise ValueError(msg)
    return value


def _build_trap_from_params(params: dict):
    """Builds a ``levisim.physics.trap.Trap`` from user-facing web parameters."""
    from levisim.physics.bead import Bead
    from levisim.physics.gas import Gas
    from levisim.physics.trap import Trap
    from levisim.physics.tweezer import Tweezer

    power_mW = _float_param(params, 'power_mW', 485.0, min_value=1.0, max_value=5000.0)
    wavelength_nm = _float_param(params, 'wavelength_nm', 1064.0, min_value=300.0, max_value=3000.0)
    wx0_um = _float_param(params, 'wx0_um', 0.9276, min_value=0.25, max_value=5.0)
    wy0_um = _float_param(params, 'wy0_um', 1.0679, min_value=0.25, max_value=5.0)

    radius_nm = _float_param(params, 'radius_nm', 60.1, min_value=10.0, max_value=300.0)
    density = _float_param(params, 'density', 1850.0, min_value=200.0, max_value=20000.0)
    use_malitson = bool(params.get('use_malitson', False))

    pressure_mbar = _float_param(params, 'pressure_mbar', 4e-3, min_value=1e-9, max_value=100.0)
    temperature_K = _float_param(params, 'temperature_K', 300.0, min_value=1.0, max_value=1000.0)

    gamma_fbx_kHz = _float_param(params, 'gamma_fbx_kHz', 2.1, min_value=0.0, max_value=50.0)
    gamma_fby_kHz = _float_param(params, 'gamma_fby_kHz', 2.5, min_value=0.0, max_value=50.0)
    gamma_fbz_kHz = _float_param(params, 'gamma_fbz_kHz', 0.2, min_value=0.0, max_value=20.0)

    beta = _float_param(params, 'beta', float(np.sqrt(2.0)), min_value=0.0, max_value=20.0)
    psi_deg = _float_param(params, 'psi_deg', 45.0, min_value=-180.0, max_value=180.0)
    photon_recoil = bool(params.get('photon_recoil', False))

    tweezer = Tweezer(
        Power=power_mW * mW,
        λ=wavelength_nm * nm,
        wx0=wx0_um * μm,
        wy0=wy0_um * μm,
    )
    bead_kwargs = dict(r=radius_nm * nm, ρ=density, λ=wavelength_nm * nm)
    if use_malitson:
        bead_kwargs['permittivity_model'] = 'Malitson'
    else:
        bead_kwargs['εr'] = _float_param(params, 'eps_r', 1.98, min_value=1.05, max_value=12.0)
    bead = Bead(**bead_kwargs)
    gas = Gas(P=pressure_mbar * mbar, T=temperature_K)

    trap = Trap(
        tweezer=tweezer,
        bead=bead,
        gas=gas,
        Γ_fbx=gamma_fbx_kHz * kHz,
        Γ_fby=gamma_fby_kHz * kHz,
        Γ_fbz=gamma_fbz_kHz * kHz,
        β=beta,
        ψ=psi_deg * degrees,
        photon_recoil=photon_recoil,
    )
    return trap


def compute_tweezer_spectra(params: dict) -> dict:
    """
    Computes live free-space tweezer motional and optical spectra using ``Trap``.

    Parameters
    ----------
    params : dict
        Physical parameters (laser power, waists, wavelength, bead radius,
        gas pressure and temperature, cold feedback damping, directed force
        strength/angle, detection floor, photon recoil flag, and frequency grid).

    Returns
    -------
    dict
        JSON-serialisable dictionary with frequency grid (kHz), one-sided
        motional PSDs ``Sxx``, ``Syy``, ``Szz`` [m²/Hz], co-/quad-spectra
        ``Cxy``, ``Qxy`` [m²/Hz], thermal baselines, balanced optical detector
        spectra in shot-noise units, and derived trap telemetry.
    """
    from levisim.analysis.spectral_density_theory import SpectralDensityTheory

    trap = _build_trap_from_params(params)
    detection_floor = _float_param(params, 'detection_floor', 1e-27, min_value=1e-32, max_value=1e-18)
    f_min_kHz = _float_param(params, 'f_min_kHz', 10.0, min_value=0.1, max_value=1000.0)
    f_max_kHz = _float_param(params, 'f_max_kHz', 220.0, min_value=1.0, max_value=2000.0)
    if f_max_kHz <= f_min_kHz:
        raise ValueError(f'f_max_kHz ({f_max_kHz}) must exceed f_min_kHz ({f_min_kHz})')
    num_points = int(_float_param(params, 'num_points', 360, min_value=32, max_value=2000))

    # When photon recoil is active, shift harmonic frequencies to z_eq_exact
    if trap.photon_recoil:
        z_eq = float(trap.z_eq_exact)
        omega_eff = trap.Ωj_at_z_eq
        theory = SpectralDensityTheory(
            trap.bead.m,
            omega_eff['x'], omega_eff['y'], omega_eff['z'],
            trap.Γx, trap.Γy, trap.Γz,
            trap.fth, trap.β, trap.ψ,
        )
    else:
        z_eq = 0.0
        omega_eff = trap.Ωj
        theory = trap.theory

    f_kHz = np.linspace(f_min_kHz, f_max_kHz, num_points)
    ω = f_kHz * kHz

    Sxx = theory.Sxx(ω) + detection_floor
    Syy = theory.Syy(ω) + detection_floor
    Szz = theory.Szz(ω) + detection_floor
    Sxx_th = np.abs(theory.χx(ω)) ** 2 * theory.Sth
    Syy_th = np.abs(theory.χy(ω)) ** 2 * theory.Sth
    Cxy = theory.Cxy(ω)
    Qxy = theory.Qxy(ω)

    # Balanced forward-scattered optical detection spectra in shot-noise units (SNU)
    S_opt_x = 1.0 + theory.Sxx(ω) / detection_floor
    S_opt_y = 1.0 + theory.Syy(ω) / detection_floor
    S_opt_z = 1.0 + theory.Szz(ω) / detection_floor

    recoil_rates = trap.photon_recoil_heating_rates
    S_imp_sql_x = float(trap.x_zpf ** 2 / (4.0 * recoil_rates['x'])) if recoil_rates['x'] > 0 else 0.0

    return {
        'engine': 'python-levisim',
        'f_kHz': f_kHz.tolist(),
        'Sxx': Sxx.tolist(),
        'Syy': Syy.tolist(),
        'Szz': Szz.tolist(),
        'Sxx_th': Sxx_th.tolist(),
        'Syy_th': Syy_th.tolist(),
        'Cxy': Cxy.tolist(),
        'Qxy': Qxy.tolist(),
        'S_opt_x': S_opt_x.tolist(),
        'S_opt_y': S_opt_y.tolist(),
        'S_opt_z': S_opt_z.tolist(),
        'detection_floor': detection_floor,
        'telemetry': {
            'fx_kHz': float(omega_eff['x'] / kHz),
            'fy_kHz': float(omega_eff['y'] / kHz),
            'fz_kHz': float(omega_eff['z'] / kHz),
            'gamma_th_Hz': float(trap.Γth / Hz),
            'gamma_x_kHz': float(trap.Γx / kHz),
            'gamma_y_kHz': float(trap.Γy / kHz),
            'gamma_z_kHz': float(trap.Γz / kHz),
            'T_eff_x_K': float(trap.T_eff['x']),
            'T_eff_y_K': float(trap.T_eff['y']),
            'T_eff_z_K': float(trap.T_eff['z']),
            'coherence_xy': float(theory.γxy),
            'z_eq_nm': float(z_eq / nm),
            'z_eq_linear_nm': float(trap.z_eq / nm),
            'mass_fg': float(trap.bead.m * 1e18),
            'x_zpf_pm': float(trap.x_zpf * 1e12),
            'recoil_x_kHz': float(recoil_rates['x'] / kHz),
            'S_imp_sql_x': S_imp_sql_x,
        },
    }


def run_tweezer_simulation(params: dict) -> dict:
    """
    Runs a short stochastic Kasdin RK4 ``Simulation`` and returns Bartlett spectral estimates.

    Parameters
    ----------
    params : dict
        Free-space trap parameters plus optional ``num_samples``, ``num_segments`` and ``seed``.

    Returns
    -------
    dict
        Estimated frequency bins (kHz) and one-sided ``Sxx``, ``Syy`` and ``Cxy`` arrays.
    """
    from levisim.analysis.spectral_density_estimate import SpectralDensityEstimate
    from levisim.simulation import FreeSpaceSimulation

    trap = _build_trap_from_params(params)
    detection_floor = _float_param(params, 'detection_floor', 1e-27, min_value=1e-32, max_value=1e-18)
    num_samples = int(_float_param(params, 'num_samples', 60_000, min_value=4_000, max_value=400_000))
    num_segments = int(_float_param(params, 'num_segments', 12, min_value=2, max_value=80))
    seed = int(_float_param(params, 'seed', 2026, min_value=0, max_value=1_000_000))
    f_min_kHz = _float_param(params, 'f_min_kHz', 10.0, min_value=0.1, max_value=1000.0)
    f_max_kHz = _float_param(params, 'f_max_kHz', 220.0, min_value=1.0, max_value=2000.0)

    sim = FreeSpaceSimulation(
        trap=trap,
        num_samples=num_samples,
        Δt=0.2 * μs,
        RK_method='vector_2',
        seed=seed,
        detection_floor=detection_floor,
    )
    est_xx = SpectralDensityEstimate(sim, 'xx', num_segments=num_segments, window='hann')
    est_yy = SpectralDensityEstimate(sim, 'yy', num_segments=num_segments, window='hann')
    est_xy = SpectralDensityEstimate(sim, 'xy', num_segments=num_segments, window='hann')

    freq_kHz = np.asarray(est_xx.freq_range, dtype=float) / 1e3
    mask = (freq_kHz >= f_min_kHz) & (freq_kHz <= f_max_kHz)

    return {
        'engine': 'python-levisim-rk2',
        'f_kHz': freq_kHz[mask].tolist(),
        'Sxx': np.asarray(est_xx.S_estimate, dtype=float)[mask].tolist(),
        'Syy': np.asarray(est_yy.S_estimate, dtype=float)[mask].tolist(),
        'Cxy': np.real(np.asarray(est_xy.S_estimate, dtype=complex))[mask].tolist(),
    }


# The cavity lab's browser engine (cavity_physics.js) implements MultiTrap's linearised model as in
# levisim 0.1, without the static cavity field. The live spectra use the same model, so that the server
# and the browser agree (tests/test_web_cavity.py).
WEB_LAB_MULTITRAP_KWARGS = {'static_cavity_shifts': False}


def compute_cavity_spectra(params: dict) -> dict:
    """
    Computes live coherent-scattering cavity optical and motional spectra via ``MultiTrap``.

    Parameters
    ----------
    params : dict
        Physical or mode-comb cavity parameters: ``mode`` (``'physical'`` or ``'comb'``),
        ``num_traps``, ``delta_kHz``, ``finesse``, ``L_cm``, ``waist_um``, ``theta_hom_deg``,
        ``theta_tw_deg``, ``phi_sw_pi``, ``phi2_deg``, ``power_mW``, ``wavelength_nm``,
        ``radius_nm``, ``pressure_mbar``, ``temperature_K``, ``photon_recoil``, and grid.

    Returns
    -------
    dict
        JSON-serialisable dictionary containing frequency grid (kHz), homodyne spectrum
        ``S_hh`` and heterodyne spectrum ``S_het`` (shot-noise units), hybridised position
        PSDs ``S_pos``, cross-spectrum ``S_x1y1_re``, drift eigenvalues, stability flags,
        and cavity/coupling telemetry.
    """
    from levisim.multiphysics.enums import QuantumLinearTheoryMethod
    from levisim.multiphysics.multitrap import MultiTrap
    from levisim.physics.bead import Bead
    from levisim.physics.tweezer import Tweezer

    mode = str(params.get('mode', 'physical')).lower()
    if mode not in ('physical', 'comb'):
        raise ValueError(f'mode must be "physical" or "comb", got {mode!r}')

    num_traps = int(_float_param(params, 'num_traps', 2, min_value=1, max_value=8))
    delta_kHz = _float_param(params, 'delta_kHz', -450.0, min_value=-2000.0, max_value=1000.0)
    finesse = _float_param(params, 'finesse', 40_000.0, min_value=1_000.0, max_value=500_000.0)
    L_cm = _float_param(params, 'L_cm', 0.96, min_value=0.1, max_value=10.0)
    waist_um = _float_param(params, 'waist_um', 61.0, min_value=10.0, max_value=300.0)
    theta_hom_deg = _float_param(params, 'theta_hom_deg', 45.0, min_value=-180.0, max_value=180.0)
    phi2_deg = _float_param(params, 'phi2_deg', 30.0, min_value=-180.0, max_value=360.0)

    pressure_mbar = _float_param(params, 'pressure_mbar', 1e-6, min_value=1e-10, max_value=1.0)
    temperature_K = _float_param(params, 'temperature_K', 300.0, min_value=1.0, max_value=1000.0)

    f_min_kHz = _float_param(params, 'f_min_kHz', 122.0, min_value=1.0, max_value=1500.0)
    f_max_kHz = _float_param(params, 'f_max_kHz', 166.0, min_value=2.0, max_value=2000.0)
    if f_max_kHz <= f_min_kHz:
        raise ValueError(f'f_max_kHz ({f_max_kHz}) must exceed f_min_kHz ({f_min_kHz})')
    num_points = int(_float_param(params, 'num_points', 300, min_value=32, max_value=1500))

    f_kHz = np.linspace(f_min_kHz, f_max_kHz, num_points)
    ω_range = f_kHz * kHz

    gas_kwargs = dict(P=pressure_mbar * mbar, T=temperature_K, vth=500.0)
    cavity_kwargs = dict(
        Δ=delta_kHz * kHz,
        L=L_cm * cm,
        w=waist_um * μm,
        Finesse=finesse,
        θhom=theta_hom_deg * degrees,
    )

    if mode == 'comb':
        comb_mid_kHz = _float_param(params, 'comb_mid_kHz', 300.0, min_value=20.0, max_value=1000.0)
        comb_span_kHz = _float_param(params, 'comb_span_kHz', 100.0, min_value=1.0, max_value=500.0)
        comb_g_kHz = _float_param(params, 'comb_g_kHz', -40.0, min_value=-200.0, max_value=200.0)
        φn_dict = {n: (phi2_deg * degrees if n == 2 else 0.0) for n in range(1, num_traps + 1)}
        mtrap = MultiTrap(
            num_traps=num_traps,
            Cartesian_axes_per_trap=2,
            gas_kwargs=gas_kwargs,
            cavity_kwargs=cavity_kwargs,
            toy_params=dict(
                Ω_model='equal spacing',
                Ω_mid=comb_mid_kHz * kHz,
                Ω_span=comb_span_kHz * kHz,
                toy_gx=comb_g_kHz * kHz,
                toy_gy=comb_g_kHz * kHz,
                φn_dict=φn_dict,
            ),
            ω_range=ω_range,
        )
    else:
        theta_tw_deg = _float_param(params, 'theta_tw_deg', 45.0, min_value=0.0, max_value=90.0)
        phi_sw_pi = _float_param(params, 'phi_sw_pi', 0.5, min_value=0.0, max_value=1.0)
        power_mW = _float_param(params, 'power_mW', 485.0, min_value=10.0, max_value=3000.0)
        wavelength_nm = _float_param(params, 'wavelength_nm', 1064.0, min_value=400.0, max_value=2000.0)
        radius_nm = _float_param(params, 'radius_nm', 60.1, min_value=15.0, max_value=200.0)
        photon_recoil = bool(params.get('photon_recoil', True))

        builtin_waists = [
            (0.9276, 1.0679), (0.950, 1.050), (0.975, 1.025), (0.965, 1.015),
            (0.940, 1.020), (0.930, 1.010), (0.915, 1.055), (0.925, 1.045),
        ]
        builtin_phases = [0.0, phi2_deg, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]

        tweezers = [
            Tweezer(
                Power=power_mW * mW,
                λ=wavelength_nm * nm,
                wx0=builtin_waists[i][0] * μm,
                wy0=builtin_waists[i][1] * μm,
                θ=theta_tw_deg * degrees,
                φ=builtin_phases[i] * degrees,
            )
            for i in range(num_traps)
        ]
        beads = [
            Bead(r=(radius_nm + 0.1 * i) * nm, Φ=phi_sw_pi * np.pi)
            for i in range(num_traps)
        ]
        mtrap = MultiTrap(
            num_traps=num_traps,
            Cartesian_axes_per_trap=2,
            tweezers=tweezers,
            beads=beads,
            gas_kwargs=gas_kwargs,
            cavity_kwargs=cavity_kwargs,
            photon_recoil=photon_recoil,
            **WEB_LAB_MULTITRAP_KWARGS,
            ω_range=ω_range,
        )

    LAM = QuantumLinearTheoryMethod.LINEAR_AMPLIFIER_MODEL
    S_hh, S_het = mtrap.homodyne_and_heterodyne_spectra(LAM)
    _, psd_dict = mtrap.all_position_power_spectra(LAM, zpf_scaled=True)
    S_x1y1 = mtrap.position_cross_spectrum('x1', 'y1', LAM, zpf_scaled=True).real

    eigs = mtrap.drift_eigenvalues
    kappa = float(mtrap.cavity.κ)
    mech_eigs = [
        {'re': float(ev.real), 'im': float(ev.imag / kHz)}
        for ev in eigs
        if ev.imag >= -1e-6 and abs(ev.real) < 0.25 * kappa
    ]

    dim_x1 = mtrap.dimensions['x1']
    dim_y1 = mtrap.dimensions['y1']
    dOmega = float(dim_x1.Ω - dim_y1.Ω)
    Delta = float(mtrap.cavity.Δ)
    g_dyn = 2.0 * float(dim_x1.g * dim_y1.g) * Delta / (Delta ** 2 + 0.25 * kappa ** 2)
    g_cs = float(dim_x1.gxy)
    if abs(dOmega) > 1.0:
        dyn_angle_deg = float(0.5 * np.arctan2(2.0 * g_dyn, dOmega) / degrees)
        cs_angle_deg = float(0.5 * np.arctan2(2.0 * g_cs, dOmega) / degrees)
        mode_angle_deg = float(0.5 * np.arctan2(2.0 * (g_dyn + g_cs), dOmega) / degrees)
    else:
        dyn_angle_deg = cs_angle_deg = mode_angle_deg = 0.0

    n_cav = float(abs(mtrap.cavity.αR + 1j * mtrap.cavity.αI) ** 2)

    return {
        'engine': 'python-levisim',
        'f_kHz': f_kHz.tolist(),
        'S_hh': np.asarray(S_hh, dtype=float).tolist(),
        'S_het': np.asarray(S_het, dtype=float).tolist(),
        'S_pos': {label: np.asarray(arr, dtype=float).tolist() for label, arr in psd_dict.items()},
        'S_x1y1_re': np.asarray(S_x1y1, dtype=float).tolist(),
        'eigenvalues': [{'re': float(ev.real), 'im_kHz': float(ev.imag / kHz)} for ev in eigs],
        'mech_eigenvalues': mech_eigs,
        'telemetry': {
            'is_stable': bool(mtrap.is_stable),
            'max_growth_rate': float(mtrap.max_growth_rate),
            'kappa_kHz': float(kappa / kHz),
            'delta_kHz': float(Delta / kHz),
            'n_cav': n_cav,
            'gx1_kHz': float(dim_x1.g / kHz),
            'gy1_kHz': float(dim_y1.g / kHz),
            'gxy1_kHz': float(dim_x1.gxy / kHz),
            'mode_angle_deg': mode_angle_deg,
            'dyn_angle_deg': dyn_angle_deg,
            'cs_angle_deg': cs_angle_deg,
            'dimensions': [
                {
                    'label': dim.label,
                    'omega_kHz': float(dim.Ω / kHz),
                    'g_kHz': float(dim.g / kHz),
                    'gxy_kHz': float(dim.gxy / kHz),
                    'phi_deg': float(dim.φ / degrees),
                }
                for dim in mtrap.dimensions.values()
            ],
        },
    }


def dispatch_request(
    method: str,
    path: str,
    body: bytes = b'',
    *,
    static_dir: Path | None = None,
    three_dir: Path | None = None,
) -> tuple[int, str, bytes, str]:
    """
    Dispatches an HTTP request to static lab assets or JSON physics APIs.

    Returns ``(status_code, content_type, body_bytes, cache_control)``.
    """
    assets = Path(static_dir) if static_dir is not None else STATIC_DIR
    parsed = urllib.parse.urlparse(path)
    route = parsed.path

    def json_resp(status: int, obj: dict) -> tuple[int, str, bytes, str]:
        return status, 'application/json; charset=utf-8', json.dumps(obj).encode('utf-8'), 'no-cache'

    if method == 'GET':
        if route == '/api/health':
            return json_resp(200, {'ok': True, 'engine': 'python-levisim', 'labs': ['free_space_tweezer', 'coherent_scattering_cavity']})

        if three_dir is not None and route.startswith('/vendor/three/'):
            rel = route[len('/vendor/three/'):]
            target = (three_dir / rel).resolve()
            if three_dir.resolve() not in target.parents and target != three_dir.resolve():
                return json_resp(403, {'error': 'forbidden'})
            if not target.is_file():
                return json_resp(404, {'error': 'not found'})
            ctype = MIME_OVERRIDES.get(target.suffix.lower()) or mimetypes.guess_type(str(target))[0] or 'application/octet-stream'
            return 200, ctype, target.read_bytes(), 'public, max-age=3600'

        rel_path = route.lstrip('/')
        if not rel_path or rel_path.endswith('/'):
            rel_path = f'{rel_path}index.html'
        target = (assets / rel_path).resolve()
        if assets.resolve() not in target.parents and target != assets.resolve():
            return json_resp(403, {'error': 'forbidden'})
        if target.is_dir():
            target = target / 'index.html'
        if not target.is_file():
            return json_resp(404, {'error': f'not found: {route}'})

        raw = target.read_bytes()
        suffix = target.suffix.lower()
        ctype = MIME_OVERRIDES.get(suffix) or mimetypes.guess_type(str(target))[0] or 'application/octet-stream'
        if suffix == '.html' and three_dir is not None:
            raw = raw.decode('utf-8').replace(THREE_CDN_PREFIX, '/vendor/three/').encode('utf-8')
        cache = 'no-cache'
        return 200, ctype, raw, cache

    if method == 'POST':
        try:
            payload = json.loads(body.decode('utf-8')) if body else {}
            if not isinstance(payload, dict):
                raise ValueError('JSON body must be an object')
        except (ValueError, json.JSONDecodeError) as exc:
            return json_resp(400, {'error': str(exc)})

        try:
            if route == '/api/tweezer/spectra':
                return json_resp(200, compute_tweezer_spectra(payload))
            if route == '/api/tweezer/simulate':
                return json_resp(200, run_tweezer_simulation(payload))
            if route == '/api/cavity/spectra':
                return json_resp(200, compute_cavity_spectra(payload))
            return json_resp(404, {'error': f'unknown endpoint: {route}'})
        except ValueError as exc:
            return json_resp(400, {'error': str(exc)})
        except Exception as exc:
            logger.error('API error on {}: {}', route, exc)
            return json_resp(500, {'error': str(exc)})

    return json_resp(405, {'error': f'method not allowed: {method}'})


def _make_handler_class(static_dir: Path, three_dir: Path | None = None):
    """Creates a request handler class serving static lab files and JSON physics APIs."""

    class LevisimWebHandler(http.server.BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def _respond(self, method: str, body: bytes = b''):
            status, content_type, data, cache = dispatch_request(
                method, self.path, body, static_dir=static_dir, three_dir=three_dir,
            )
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', cache)
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self._respond('GET')

        def do_POST(self):
            length = int(self.headers.get('Content-Length', '0'))
            raw_body = self.rfile.read(length) if length > 0 else b'{}'
            self._respond('POST', raw_body)

        def log_message(self, format, *args):
            logger.debug('levisim-web {} - ' + format, self.address_string(), *args)

    return LevisimWebHandler


def create_server(
    host: str = '127.0.0.1',
    port: int = 8765,
    *,
    static_dir: Path | None = None,
    three_dir: Path | None = None,
) -> http.server.ThreadingHTTPServer:
    """
    Creates a threaded HTTP server for the ``levisim`` Three.js web interfaces.

    Parameters
    ----------
    host : str
        Interface to bind (default ``'127.0.0.1'``).
    port : int
        TCP port (``0`` selects an ephemeral port).
    static_dir : Path, optional
        Override directory containing static web assets.
    three_dir : Path, optional
        Optional local directory of the ``three`` npm package so HTML pages can
        load Three.js from ``/vendor/three/`` without internet access.

    Returns
    -------
    http.server.ThreadingHTTPServer
        Ready-to-serve threaded HTTP server instance.
    """
    assets = Path(static_dir) if static_dir is not None else STATIC_DIR
    env_three = os.environ.get('LEVISIM_THREE_DIR')
    resolved_three = Path(three_dir) if three_dir is not None else (Path(env_three) if env_three else None)
    handler_cls = _make_handler_class(assets, resolved_three)
    return http.server.ThreadingHTTPServer((host, port), handler_cls)


def serve(
    host: str = '127.0.0.1',
    port: int = 8765,
    *,
    lab: str = 'tweezer',
    open_browser: bool = True,
    three_dir: Path | None = None,
) -> None:
    """Starts the ``levisim`` Three.js web server and blocks until interrupted."""
    server = create_server(host, port, three_dir=three_dir)
    actual_host, actual_port = server.server_address
    suffix = {
        'tweezer': '/free_space_tweezer/',
        'cavity': '/coherent_scattering_cavity/',
        'home': '/',
    }.get(lab, '/')
    url = f'http://{actual_host}:{actual_port}{suffix}'
    logger.info('Serving levisim interactive web lab at {}', url)
    sys.stdout.write(f'levisim-web running at {url}\nPress Ctrl+C to stop.\n')
    sys.stdout.flush()
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info('Stopping levisim-web server')
    finally:
        server.server_close()


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point for ``levisim-web``."""
    parser = argparse.ArgumentParser(
        prog='levisim-web',
        description='Launch the interactive Three.js optical-bench labs backed by levisim.',
    )
    parser.add_argument('--host', default='127.0.0.1', help='Host interface to bind (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=8765, help='Port to bind (default: 8765)')
    parser.add_argument('--lab', choices=('tweezer', 'cavity', 'home'), default='home', help='Initial lab page to open')
    parser.add_argument('--no-open', action='store_true', help='Do not open a browser tab automatically')
    parser.add_argument('--three-dir', type=Path, default=None, help='Local path to the `three` package for offline serving')
    args = parser.parse_args(argv)
    serve(
        host=args.host,
        port=args.port,
        lab=args.lab,
        open_browser=not args.no_open,
        three_dir=args.three_dir,
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

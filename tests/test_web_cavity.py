"""
Tests for the coherent-scattering cavity Three.js web interface and backend API.

Verifies that:
  - the standalone JavaScript LAM + complex Hessenberg QR engine (`cavity_physics.js`)
    agrees with Python `MultiTrap(QuantumLinearTheoryMethod.LINEAR_AMPLIFIER_MODEL)`
    across 1x2 (PRR 5 node & cancellation point), 2x2 (Quickstart stable & unstable),
    and 8x2 (PRR 8 equal-spacing mechanical mode comb);
  - linear stability (`is_stable`, `max_growth_rate`) matches `MultiTrap.drift_eigenvalues`;
  - the `/api/cavity/spectra` HTTP dispatch route validates inputs and returns exact spectra.
"""

import json
import shutil
import subprocess

import numpy as np
import pytest

from levisim.web.server import STATIC_DIR, compute_cavity_spectra, dispatch_request

NODE = shutil.which('node')
CAVITY_JS = STATIC_DIR / 'coherent_scattering_cavity' / 'cavity_physics.js'


def _run_js_cavity(params: dict) -> dict:
    """Evaluates `computeCavitySpectraJS(params)` in Node.js and returns parsed JSON."""
    script = (
        f"import {{ computeCavitySpectraJS }} from {json.dumps(CAVITY_JS.as_uri())};\n"
        f"const out = computeCavitySpectraJS({json.dumps(params)});\n"
        "process.stdout.write(JSON.stringify(out));\n"
    )
    proc = subprocess.run(
        [NODE, '--input-type=module', '-e', script],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(proc.stdout)


@pytest.mark.parametrize(
    'case_name, params',
    [
        (
            'quickstart_2x2_stable',
            dict(mode='physical', num_traps=2, delta_kHz=-450.0, phi_sw_pi=0.5, phi2_deg=30.0, num_points=64),
        ),
        (
            'quickstart_2x2_unstable',
            dict(mode='physical', num_traps=2, delta_kHz=-450.0, phi_sw_pi=0.5, phi2_deg=90.0, num_points=64),
        ),
        (
            'prr5_node',
            dict(mode='physical', num_traps=1, delta_kHz=-450.0, phi_sw_pi=0.5, phi2_deg=0.0, num_points=64),
        ),
        (
            'prr5_cancellation',
            dict(mode='physical', num_traps=1, delta_kHz=-450.0, phi_sw_pi=0.248, phi2_deg=0.0, num_points=64),
        ),
        (
            'prr5_antinode_side',
            dict(mode='physical', num_traps=1, delta_kHz=-450.0, phi_sw_pi=0.15, phi2_deg=0.0, num_points=64),
        ),
        (
            'prr8_8bead_comb',
            dict(
                mode='comb', num_traps=8, delta_kHz=-450.0, phi2_deg=0.0,
                comb_mid_kHz=300.0, comb_span_kHz=100.0, comb_g_kHz=-40.0,
                f_min_kHz=120.0, f_max_kHz=380.0, num_points=48,
            ),
        ),
    ],
)
def test_js_cavity_physics_matches_python_multitrap(case_name, params):
    """Browser LAM + QR eigenvalue engine matches Python `MultiTrap` across all regimes."""
    if NODE is None:
        pytest.skip('Node.js not available in test environment')

    py = compute_cavity_spectra(params)
    js = _run_js_cavity(params)

    assert np.allclose(js['f_kHz'], py['f_kHz'], rtol=1e-12, atol=1e-12)
    assert np.allclose(js['S_hh'], py['S_hh'], rtol=1e-5, atol=1e-12), f'{case_name}: S_hh mismatch'
    assert np.allclose(js['S_het'], py['S_het'], rtol=1e-5, atol=1e-12), f'{case_name}: S_het mismatch'

    for label, py_psd in py['S_pos'].items():
        assert np.allclose(js['S_pos'][label], py_psd, rtol=1e-5, atol=1e-12), (
            f'{case_name}: position PSD mismatch for {label}'
        )

    csd_scale = max(1.0, float(np.max(np.abs(py['S_x1y1_re']))))
    assert np.allclose(js['S_x1y1_re'], py['S_x1y1_re'], rtol=1e-5, atol=1e-7 * csd_scale), (
        f'{case_name}: Re S_x1y1 mismatch'
    )

    assert js['telemetry']['is_stable'] is py['telemetry']['is_stable'], f'{case_name}: stability flag mismatch'
    assert np.isclose(
        js['telemetry']['max_growth_rate'],
        py['telemetry']['max_growth_rate'],
        rtol=1e-4,
        atol=1e-1,
    ), f'{case_name}: max_growth_rate mismatch'

    for tkey in ('kappa_kHz', 'n_cav', 'gx1_kHz', 'gy1_kHz', 'gxy1_kHz'):
        assert np.isclose(js['telemetry'][tkey], py['telemetry'][tkey], rtol=1e-5, atol=1e-8), (
            f'{case_name}: telemetry mismatch in {tkey}'
        )


def test_quickstart_phase_stability_transition():
    """Quickstart 2x2 is linearly stable at φ2=30° and unstable (+2678 /s) at φ2=90°."""
    stable = compute_cavity_spectra({'num_traps': 2, 'phi2_deg': 30.0, 'num_points': 40})
    unstable = compute_cavity_spectra({'num_traps': 2, 'phi2_deg': 90.0, 'num_points': 40})

    assert stable['telemetry']['is_stable'] is True
    assert stable['telemetry']['max_growth_rate'] < 0.0

    assert unstable['telemetry']['is_stable'] is False
    assert unstable['telemetry']['max_growth_rate'] == pytest.approx(2678.0, rel=0.05)


def test_prr5_cross_spectrum_flips_sign_across_cancellation_point():
    """Re S_x1y1 flips sign between node side (Φ=0.35π) and antinode side (Φ=0.15π) of Φc."""
    node_side = compute_cavity_spectra({'num_traps': 1, 'phi_sw_pi': 0.35, 'f_min_kHz': 130, 'f_max_kHz': 160, 'num_points': 120})
    anti_side = compute_cavity_spectra({'num_traps': 1, 'phi_sw_pi': 0.15, 'f_min_kHz': 130, 'f_max_kHz': 160, 'num_points': 120})

    # Dominant peak near y1 (~135 kHz) flips sign across the PRR 5 cancellation point Φc ≈ 0.248π
    idx_y = int(np.argmin(np.abs(np.asarray(node_side['f_kHz']) - 135.0)))
    band = slice(max(0, idx_y - 10), min(120, idx_y + 10))
    peak_node = node_side['S_x1y1_re'][band][int(np.argmax(np.abs(node_side['S_x1y1_re'][band])))]
    peak_anti = anti_side['S_x1y1_re'][band][int(np.argmax(np.abs(anti_side['S_x1y1_re'][band])))]
    assert peak_node * peak_anti < 0.0


def test_cavity_dispatch_routes_and_validation(tmp_path):
    """HTTP router exposes coherent_scattering_cavity lab and validates POST payloads."""
    fake_three = tmp_path / 'three'
    (fake_three / 'build').mkdir(parents=True)
    (fake_three / 'build' / 'three.module.js').write_text('// stub\n', encoding='utf-8')

    status, _, body, _ = dispatch_request('GET', '/api/health')
    assert status == 200
    assert 'coherent_scattering_cavity' in json.loads(body.decode('utf-8'))['labs']

    status, ctype, body, _ = dispatch_request('GET', '/coherent_scattering_cavity/', three_dir=fake_three)
    assert status == 200 and 'text/html' in ctype
    assert '/vendor/three/build/three.module.js' in body.decode('utf-8')

    status, _, body, _ = dispatch_request(
        'POST', '/api/cavity/spectra',
        json.dumps({'num_traps': 2, 'phi2_deg': 30.0, 'num_points': 40}).encode('utf-8'),
    )
    assert status == 200
    payload = json.loads(body.decode('utf-8'))
    assert payload['engine'] == 'python-levisim'
    assert payload['telemetry']['is_stable'] is True
    assert len(payload['S_hh']) == 40

    status, _, _, _ = dispatch_request(
        'POST', '/api/cavity/spectra',
        json.dumps({'mode': 'invalid'}).encode('utf-8'),
    )
    assert status == 400

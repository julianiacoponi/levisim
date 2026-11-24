"""
Tests for the free-space optical tweezer Three.js web interface and backend API.

Verifies that:
  - the Python JSON API computes exact `Trap` / `SpectralDensityTheory` spectra and
    handles invalid inputs cleanly;
  - the standalone JavaScript physics module (`tweezer_physics.js`) agrees with
    Python `levisim` across default, mirrored-quadrant, thermal, Malitson, and
    photon-recoil configurations;
  - the local HTTP server serves static assets, rewrites the Three.js importmap
    when ``three_dir`` is provided, and returns live spectra over HTTP.
"""

import json
import shutil
import subprocess
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest

from levisim.web.server import STATIC_DIR, compute_tweezer_spectra, create_server, dispatch_request

NODE = shutil.which('node')
TWEEZER_JS = STATIC_DIR / 'free_space_tweezer' / 'tweezer_physics.js'


def _run_js_tweezer(params: dict) -> dict:
    """Evaluates `computeTweezerSpectraJS(params)` in Node.js and returns parsed JSON."""
    script = (
        f"import {{ computeTweezerSpectraJS }} from {json.dumps(TWEEZER_JS.as_uri())};\n"
        f"const out = computeTweezerSpectraJS({json.dumps(params)});\n"
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
        ('prr6_default', {'psi_deg': 45.0, 'beta': float(np.sqrt(2.0)), 'num_points': 120}),
        ('prr6_mirrored', {'psi_deg': -45.0, 'beta': float(np.sqrt(2.0)), 'num_points': 120}),
        ('thermal_only', {'beta': 0.0, 'gamma_fbx_kHz': 0.5, 'gamma_fby_kHz': 0.8, 'num_points': 120}),
        ('malitson_1550nm', {'wavelength_nm': 1550.0, 'use_malitson': True, 'power_mW': 320.0, 'num_points': 120}),
        ('photon_recoil_shift', {'photon_recoil': True, 'radius_nm': 75.0, 'power_mW': 650.0, 'num_points': 120}),
    ],
)
def test_js_tweezer_physics_matches_python_levisim(case_name, params):
    """The 60 FPS browser JS physics engine matches Python `Trap` / `SpectralDensityTheory`."""
    if NODE is None:
        pytest.skip('Node.js not available in test environment')

    py = compute_tweezer_spectra(params)
    js = _run_js_tweezer(params)

    assert np.allclose(js['f_kHz'], py['f_kHz'], rtol=1e-12, atol=1e-12)
    for key in ('Sxx', 'Syy', 'Szz', 'Sxx_th', 'Syy_th', 'S_opt_x', 'S_opt_y', 'S_opt_z'):
        assert np.allclose(js[key], py[key], rtol=1e-6, atol=0.0), f'{case_name}: mismatch in {key}'
    for key in ('Cxy', 'Qxy'):
        scale = max(1e-35, float(np.max(np.abs(py[key]))))
        assert np.allclose(js[key], py[key], rtol=1e-6, atol=1e-10 * scale), f'{case_name}: mismatch in {key}'

    for tkey in ('fx_kHz', 'fy_kHz', 'fz_kHz', 'gamma_th_Hz', 'T_eff_x_K', 'T_eff_y_K', 'coherence_xy', 'z_eq_nm'):
        assert np.isclose(js['telemetry'][tkey], py['telemetry'][tkey], rtol=1e-6, atol=1e-9), (
            f'{case_name}: telemetry mismatch in {tkey}: js={js["telemetry"][tkey]} vs py={py["telemetry"][tkey]}'
        )


def test_cospectrum_flips_sign_for_mirrored_force_angle():
    """Flipping ψ from +45° to -45° mirrors Cxy while leaving Sxx and Syy identical (PRR 6)."""
    pos = compute_tweezer_spectra({'psi_deg': 45.0, 'num_points': 100})
    neg = compute_tweezer_spectra({'psi_deg': -45.0, 'num_points': 100})

    assert np.allclose(pos['Sxx'], neg['Sxx'], rtol=1e-12)
    assert np.allclose(pos['Syy'], neg['Syy'], rtol=1e-12)
    assert np.allclose(pos['Cxy'], -np.asarray(neg['Cxy']), rtol=1e-12)


def test_optical_spectrum_approaches_shot_noise_floor_off_resonance():
    """Balanced forward-scattering optical spectrum tends to 1 SNU far from mechanical peaks."""
    out = compute_tweezer_spectra({'f_min_kHz': 10.0, 'f_max_kHz': 900.0, 'num_points': 200, 'detection_floor': 1e-24})
    assert out['S_opt_x'][-1] == pytest.approx(1.0, rel=1e-2)
    assert max(out['S_opt_x']) > 100.0


def test_web_dispatch_static_routes_and_tweezer_api(tmp_path):
    """HTTP router serves static assets, rewrites Three.js paths, and computes live spectra."""
    fake_three = tmp_path / 'three'
    (fake_three / 'build').mkdir(parents=True)
    (fake_three / 'build' / 'three.module.js').write_text('// stub three\n', encoding='utf-8')

    status, _, body, _ = dispatch_request('GET', '/api/health', three_dir=fake_three)
    assert status == 200
    health = json.loads(body.decode('utf-8'))
    assert health['ok'] is True
    assert 'free_space_tweezer' in health['labs']

    status, ctype, body, _ = dispatch_request('GET', '/free_space_tweezer/', three_dir=fake_three)
    assert status == 200 and 'text/html' in ctype
    assert '/vendor/three/build/three.module.js' in body.decode('utf-8')

    status, _, body, _ = dispatch_request('GET', '/vendor/three/build/three.module.js', three_dir=fake_three)
    assert status == 200 and 'stub three' in body.decode('utf-8')

    status, _, body, _ = dispatch_request(
        'POST', '/api/tweezer/spectra',
        json.dumps({'psi_deg': -45.0, 'num_points': 64}).encode('utf-8'),
    )
    assert status == 200
    payload = json.loads(body.decode('utf-8'))
    assert payload['engine'] == 'python-levisim'
    assert len(payload['f_kHz']) == 64
    assert payload['telemetry']['fx_kHz'] == pytest.approx(156.34, rel=1e-2)

    status, _, _, _ = dispatch_request(
        'POST', '/api/tweezer/spectra',
        json.dumps({'power_mW': -10}).encode('utf-8'),
    )
    assert status == 400

    status, _, _, _ = dispatch_request('GET', '/../pyproject.toml')
    assert status == 403


def test_web_server_binds_and_accepts_http_when_loopback_permitted(tmp_path):
    """End-to-end TCP server test (skips cleanly if sandbox blocks loopback connect)."""
    server = create_server('127.0.0.1', 0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/api/health') as resp:
                assert resp.status == 200
                assert json.loads(resp.read().decode('utf-8'))['ok'] is True
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, PermissionError):
                pytest.skip('Loopback TCP connect blocked by sandbox')
            raise
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

/**
 * Application controller wiring the Coherent-Scattering Cavity Three.js bench,
 * hybrid JS (60 FPS LAM + Hessenberg QR eigenvalues) / Python levisim MultiTrap backend,
 * diegetic 3D optic controls, paper presets, and the in-game RF spectrum analyzer monitor.
 */
import {
  DEFAULT_CAVITY_PARAMS,
  computeCavitySpectraJS,
} from './cavity_physics.js';
import { CoherentScatteringCavityScene } from './cavity_scene.js';

const PRESETS = {
  quickstart_2x2: {
    label: 'Quickstart 2×2 (φ₂=30°)',
    params: {
      ...DEFAULT_CAVITY_PARAMS,
      mode: 'physical', num_traps: 2, delta_kHz: -450, phi_sw_pi: 0.5, phi2_deg: 30,
      theta_hom_deg: 45, theta_tw_deg: 45, f_min_kHz: 125, f_max_kHz: 165,
    },
  },
  unstable_90: {
    label: 'Unstable 2×2 (φ₂=90°)',
    params: {
      ...DEFAULT_CAVITY_PARAMS,
      mode: 'physical', num_traps: 2, delta_kHz: -450, phi_sw_pi: 0.5, phi2_deg: 90,
      theta_hom_deg: 45, theta_tw_deg: 45, f_min_kHz: 125, f_max_kHz: 165,
    },
  },
  prr5_node: {
    label: 'PRR 5 Node (Φ=0.5π)',
    params: {
      ...DEFAULT_CAVITY_PARAMS,
      mode: 'physical', num_traps: 1, delta_kHz: -450, phi_sw_pi: 0.5, phi2_deg: 0,
      theta_hom_deg: 45, theta_tw_deg: 45, f_min_kHz: 125, f_max_kHz: 165,
    },
  },
  prr5_cancel: {
    label: 'PRR 5 Cancel (Φ=0.248π)',
    params: {
      ...DEFAULT_CAVITY_PARAMS,
      mode: 'physical', num_traps: 1, delta_kHz: -450, phi_sw_pi: 0.248, phi2_deg: 0,
      theta_hom_deg: 45, theta_tw_deg: 45, f_min_kHz: 125, f_max_kHz: 165,
    },
  },
  prr5_anti: {
    label: 'PRR 5 Off-Node (Φ=0.15π)',
    params: {
      ...DEFAULT_CAVITY_PARAMS,
      mode: 'physical', num_traps: 1, delta_kHz: -450, phi_sw_pi: 0.15, phi2_deg: 0,
      theta_hom_deg: 45, theta_tw_deg: 45, f_min_kHz: 125, f_max_kHz: 165,
    },
  },
  prr8_4p: {
    label: 'PRR 8 · 4-Bead Comb',
    params: {
      ...DEFAULT_CAVITY_PARAMS,
      mode: 'comb', num_traps: 4, delta_kHz: -450, phi2_deg: 0,
      comb_mid_kHz: 300, comb_span_kHz: 100, comb_g_kHz: -40,
      f_min_kHz: 140, f_max_kHz: 380, num_points: 280,
    },
  },
  prr8_8p: {
    label: 'PRR 8 · 8-Bead Comb',
    params: {
      ...DEFAULT_CAVITY_PARAMS,
      mode: 'comb', num_traps: 8, delta_kHz: -450, phi2_deg: 0,
      comb_mid_kHz: 300, comb_span_kHz: 100, comb_g_kHz: -40,
      f_min_kHz: 100, f_max_kHz: 400, num_points: 260,
    },
  },
};

const state = {
  params: { ...PRESETS.quickstart_2x2.params },
  monitorView: 'combined', // 'optical' | 'motional' | 'combined' | 'cross'
  backendAvailable: false,
  usePythonBackend: false,
  activePreset: 'quickstart_2x2',
};

let pendingPythonTimer = null;
let sceneApp = null;

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function buildMonitorPayload(spectra) {
  const tel = spectra.telemetry;
  const p = state.params;
  const dims = tel.dimensions || [];
  const refs = dims.length <= 6
    ? dims.map(d => ({ x: d.omega_kHz, label: `Ω${d.label}` }))
    : dims.filter((_, i) => i % 2 === 0).map(d => ({ x: d.omega_kHz, label: '' }));

  const panels = [];

  if (state.monitorView === 'optical' || state.monitorView === 'combined') {
    const maxOpt = Math.max(20, ...spectra.S_hh, ...spectra.S_het);
    const minOpt = Math.min(0.5, ...spectra.S_hh, ...spectra.S_het);
    panels.push({
      title: `CAVITY OPTICAL OUTPUT · θ_hom=${p.theta_hom_deg.toFixed(0)}° (SNU)`,
      xLabel: state.monitorView === 'optical' ? 'ω / 2π (kHz)' : '',
      yLabel: 'S_opt (SNU)',
      logY: true,
      yMin: Math.min(-0.3, Math.floor(Math.log10(Math.max(0.05, minOpt)))),
      yMax: Math.max(2.5, Math.ceil(Math.log10(maxOpt)) + 0.3),
      xs: spectra.f_kHz,
      refs,
      floorValue: 1.0,
      series: [
        { ys: spectra.S_hh, label: 'S_hh (homodyne)', colorIndex: 0, fill: true },
        { ys: spectra.S_het, label: 'S_het (heterodyne)', colorIndex: 1, dashed: true, lineWidth: 2.2 },
      ],
    });
  }

  if (state.monitorView === 'motional' || state.monitorView === 'combined') {
    const posLabels = Object.keys(spectra.S_pos || {}).slice(0, 4);
    const allPosVals = posLabels.flatMap(k => spectra.S_pos[k]);
    const maxP = Math.max(1e2, ...allPosVals);
    const minP = Math.max(1e-2, Math.min(...allPosVals));
    panels.push({
      title: 'HYBRIDISED MOTIONAL PSDs (zpf-scaled)',
      xLabel: 'ω / 2π (kHz)',
      yLabel: 'S_qq (q²/Hz)',
      logY: true,
      yMin: Math.floor(Math.log10(minP)) - 0.3,
      yMax: Math.ceil(Math.log10(maxP)) + 0.3,
      xs: spectra.f_kHz,
      refs,
      series: posLabels.map((lbl, idx) => ({
        ys: spectra.S_pos[lbl],
        label: `S_${lbl}${lbl}`,
        colorIndex: idx,
        lineWidth: idx < 2 ? 2.6 : 2.0,
      })),
    });
  }

  if (state.monitorView === 'cross') {
    const maxC = Math.max(1e3, ...spectra.S_x1y1_re.map(Math.abs)) * 1.18;
    panels.push({
      title: `PRR 5 CROSS-SPECTRUM · Re S_x1y1(ω) at Φ = ${p.phi_sw_pi.toFixed(3)}π (cancels near 0.248π)`,
      xLabel: 'ω / 2π (kHz)',
      yLabel: 'Re S_x1y1',
      logY: false,
      zeroLine: true,
      yMin: -maxC,
      yMax: maxC,
      xs: spectra.f_kHz,
      refs,
      series: [
        { ys: spectra.S_x1y1_re, label: 'Re S_x1y1 (q-units)', colorIndex: 2 },
      ],
    });
  }

  const stableTxt = tel.is_stable
    ? (spectra.engine === 'python-levisim' ? 'STABLE · PYTHON' : 'STABLE')
    : `UNSTABLE (+${tel.max_growth_rate.toFixed(0)} /s)`;
  const modeTag = p.mode === 'comb' ? `${p.num_traps}×2 COMB` : `${p.num_traps}×2 MultiTrap`;

  return {
    title: `LEVISIM · COHERENT SCATTERING CAVITY (${modeTag})`,
    statusText: stableTxt,
    statusColor: tel.is_stable ? '#10b981' : '#ef4444',
    footer:
      `Δ/2π=${tel.delta_kHz.toFixed(0)} kHz  κ/2π=${tel.kappa_kHz.toFixed(0)} kHz  |  ` +
      `gx1=${tel.gx1_kHz.toFixed(1)} kHz  gxy=${tel.gxy1_kHz.toFixed(2)} kHz  |  ` +
      `Φ=${p.phi_sw_pi.toFixed(3)}π  φ₂=${p.phi2_deg.toFixed(0)}°  |α|²=${tel.n_cav.toExponential(1)}`,
    panels,
    eigenInset: {
      eigenvalues: spectra.mech_eigenvalues,
      isStable: tel.is_stable,
    },
  };
}

function updateTelemetryDOM(tel) {
  const stabEl = document.getElementById('ro-stable');
  stabEl.textContent = tel.is_stable ? 'STABLE' : `UNSTABLE (+${tel.max_growth_rate.toFixed(0)} /s)`;
  stabEl.className = 'readout-v ' + (tel.is_stable ? 'good' : 'bad');

  document.getElementById('ro-kappa').textContent = `${tel.kappa_kHz.toFixed(1)} kHz`;
  document.getElementById('ro-gx').textContent = `${tel.gx1_kHz.toFixed(2)} kHz`;
  document.getElementById('ro-gy').textContent = `${tel.gy1_kHz.toFixed(2)} kHz`;
  document.getElementById('ro-gxy').textContent = `${tel.gxy1_kHz.toFixed(2)} kHz`;
  document.getElementById('ro-ncav').textContent = tel.n_cav.toExponential(2);
  document.getElementById('ro-rot').textContent = `${tel.mode_angle_deg >= 0 ? '+' : ''}${tel.mode_angle_deg.toFixed(2)}°`;
  document.getElementById('ro-growth').textContent = `${tel.max_growth_rate >= 0 ? '+' : ''}${tel.max_growth_rate.toFixed(1)} /s`;

  const banner = document.getElementById('stability-pill');
  if (banner) {
    banner.textContent = tel.is_stable ? 'Stable Steady State' : `Unstable (Re λ = +${tel.max_growth_rate.toFixed(0)} /s)`;
    banner.className = 'pill-btn ' + (tel.is_stable ? 'active' : 'warn');
  }
}

function syncSlidersFromState() {
  const p = state.params;
  const set = (id, val, labelId, text) => {
    const el = document.getElementById(id);
    if (el) el.value = val;
    const lbl = document.getElementById(labelId);
    if (lbl) lbl.textContent = text;
  };
  set('sl-delta', p.delta_kHz, 'val-delta', `${p.delta_kHz >= 0 ? '+' : ''}${p.delta_kHz.toFixed(0)} kHz`);
  set('sl-finesse', p.finesse / 1000, 'val-finesse', `${(p.finesse / 1000).toFixed(0)}k`);
  set('sl-phi-sw', p.phi_sw_pi, 'val-phi-sw',
    `${p.phi_sw_pi.toFixed(3)}π` + (p.phi_sw_pi >= 0.495 ? ' (node)' : (Math.abs(p.phi_sw_pi - 0.248) < 0.01 ? ' (cancel!)' : ''))
  );
  set('sl-theta-tw', p.theta_tw_deg, 'val-theta-tw', `${p.theta_tw_deg.toFixed(1)}°`);
  set('sl-theta-hom', p.theta_hom_deg, 'val-theta-hom', `${p.theta_hom_deg.toFixed(1)}°`);
  set('sl-phi2', p.phi2_deg, 'val-phi2', `${p.phi2_deg.toFixed(1)}°`);
  set('sl-ntrap', p.num_traps, 'val-ntrap', `${p.num_traps} bead${p.num_traps > 1 ? 's' : ''}`);
  set('sl-power', p.power_mW, 'val-power', `${p.power_mW.toFixed(0)} mW`);
  set('sl-logp', Math.log10(p.pressure_mbar), 'val-logp', `${p.pressure_mbar.toExponential(1)} mbar`);
  document.getElementById('chk-recoil').checked = Boolean(p.photon_recoil);
  document.getElementById('chk-mode-vec').checked = Boolean(sceneApp ? sceneApp.showModeVectors : true);
}

function recomputeAndRender() {
  const spectra = computeCavitySpectraJS(state.params);
  sceneApp.syncVisuals(state.params, spectra.telemetry);
  sceneApp.monitor.render(buildMonitorPayload(spectra));
  updateTelemetryDOM(spectra.telemetry);

  if (state.backendAvailable && state.usePythonBackend) {
    clearTimeout(pendingPythonTimer);
    pendingPythonTimer = setTimeout(async () => {
      try {
        const res = await fetch('/api/cavity/spectra', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(state.params),
        });
        if (!res.ok) return;
        const pySpectra = await res.json();
        sceneApp.monitor.render(buildMonitorPayload(pySpectra));
        updateTelemetryDOM(pySpectra.telemetry);
      } catch (_) {
        // Stay gracefully on JS engine
      }
    }, 95);
  }
}

async function detectPythonBackend() {
  const badge = document.getElementById('engine-badge');
  try {
    const res = await fetch('/api/health');
    if (res.ok) {
      const info = await res.json();
      if (info.ok && Array.isArray(info.labs) && info.labs.includes('coherent_scattering_cavity')) {
        state.backendAvailable = true;
        state.usePythonBackend = true;
        badge.textContent = 'Python levisim + 60 FPS JS';
        badge.classList.add('python');
        recomputeAndRender();
      }
    }
  } catch (_) {
    // Pure static GitHub Pages mode
  }
}

function initUI() {
  const container = document.getElementById('viewport');
  const tooltip = document.getElementById('diegetic-tooltip');

  sceneApp = new CoherentScatteringCavityScene(container, tooltip, {
    onDeltaParam: (key, delta) => {
      if (key === 'delta_kHz') state.params.delta_kHz = clamp(state.params.delta_kHz + delta, -800, 300);
      if (key === 'phi_sw_pi') {
        state.params.mode = 'physical';
        state.params.phi_sw_pi = clamp(state.params.phi_sw_pi + delta, 0.04, 0.50);
      }
      if (key === 'theta_tw_deg') {
        state.params.mode = 'physical';
        state.params.theta_tw_deg = clamp(state.params.theta_tw_deg + delta, 5, 85);
      }
      if (key === 'theta_hom_deg') state.params.theta_hom_deg = clamp(state.params.theta_hom_deg + delta, 0, 180);
      if (key === 'phi2_deg') state.params.phi2_deg = clamp(state.params.phi2_deg + delta, 0, 180);
      syncSlidersFromState();
      recomputeAndRender();
    },
    onDeltaLogPressure: (dLog) => {
      const lp = clamp(Math.log10(state.params.pressure_mbar) + dLog, -9, -2);
      state.params.pressure_mbar = Math.pow(10, lp);
      syncSlidersFromState();
      recomputeAndRender();
    },
  });

  const bindSlider = (id, handler) => {
    document.getElementById(id).addEventListener('input', (ev) => {
      handler(parseFloat(ev.target.value));
      syncSlidersFromState();
      recomputeAndRender();
    });
  };
  bindSlider('sl-delta', v => { state.params.delta_kHz = v; });
  bindSlider('sl-finesse', v => { state.params.finesse = v * 1000; });
  bindSlider('sl-phi-sw', v => { state.params.mode = 'physical'; state.params.phi_sw_pi = v; });
  bindSlider('sl-theta-tw', v => { state.params.mode = 'physical'; state.params.theta_tw_deg = v; });
  bindSlider('sl-theta-hom', v => { state.params.theta_hom_deg = v; });
  bindSlider('sl-phi2', v => { state.params.phi2_deg = v; });
  bindSlider('sl-ntrap', v => {
    state.params.num_traps = Math.round(v);
    if (state.params.mode === 'physical') {
      state.params.f_min_kHz = 125;
      state.params.f_max_kHz = 165;
    }
  });
  bindSlider('sl-power', v => { state.params.power_mW = v; });
  bindSlider('sl-logp', v => { state.params.pressure_mbar = Math.pow(10, v); });

  document.getElementById('chk-recoil').addEventListener('change', (ev) => {
    state.params.photon_recoil = ev.target.checked;
    recomputeAndRender();
  });
  document.getElementById('chk-mode-vec').addEventListener('change', (ev) => {
    sceneApp.setShowModeVectors(ev.target.checked);
  });

  // Paper preset buttons
  const presetBox = document.getElementById('preset-grid');
  Object.entries(PRESETS).forEach(([key, preset]) => {
    const btn = document.createElement('button');
    btn.className = 'preset-btn' + (key === state.activePreset ? ' active' : '');
    btn.textContent = preset.label;
    btn.addEventListener('click', () => {
      state.activePreset = key;
      state.params = { ...preset.params };
      presetBox.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      if (key.startsWith('prr5_')) {
        state.monitorView = 'cross';
        document.getElementById('sel-view').value = 'cross';
      } else if (key.startsWith('prr8_')) {
        state.monitorView = 'optical';
        document.getElementById('sel-view').value = 'optical';
      } else {
        state.monitorView = 'combined';
        document.getElementById('sel-view').value = 'combined';
      }
      syncSlidersFromState();
      recomputeAndRender();
    });
    presetBox.appendChild(btn);
  });

  // Camera bookmarks
  document.querySelectorAll('[data-cam]').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('[data-cam]').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      sceneApp.flyToBookmark(btn.dataset.cam);
    });
  });

  document.getElementById('sel-chamber').addEventListener('change', (ev) => sceneApp.setChamberMode(ev.target.value));
  document.getElementById('sel-theme').addEventListener('change', (ev) => sceneApp.monitor.setTheme(ev.target.value));
  document.getElementById('sel-view').addEventListener('change', (ev) => {
    state.monitorView = ev.target.value;
    recomputeAndRender();
  });

  const btnMonPower = document.getElementById('btn-mon-power');
  btnMonPower.addEventListener('click', () => {
    const next = !sceneApp.monitor.powerOn;
    sceneApp.monitor.setPower(next);
    btnMonPower.classList.toggle('active', next);
    btnMonPower.textContent = next ? 'Monitor: ON' : 'Monitor: OFF';
  });

  // 2D Pop-Out Spectrum Analyzer
  const popout = document.getElementById('popout-overlay');
  const popoutCanvas = document.getElementById('popout-canvas');
  const btnPopout = document.getElementById('btn-popout');
  sceneApp.monitor.attachPopoutCanvas(popoutCanvas);

  const togglePopout = (force) => {
    const open = force !== undefined ? force : !popout.classList.contains('open');
    popout.classList.toggle('open', open);
    btnPopout.classList.toggle('active', open);
    if (open) recomputeAndRender();
  };
  btnPopout.addEventListener('click', () => togglePopout());
  document.getElementById('btn-popout-close').addEventListener('click', () => togglePopout(false));

  popoutCanvas.addEventListener('mousemove', (ev) => {
    const rect = popoutCanvas.getBoundingClientRect();
    const px = ((ev.clientX - rect.left) / rect.width) * popoutCanvas.width;
    const norm = (px - 94) / (popoutCanvas.width - 120);
    sceneApp.monitor.hoverNormX = norm >= 0 && norm <= 1 ? norm : null;
    if (sceneApp.monitor.lastPayload) sceneApp.monitor.render(sceneApp.monitor.lastPayload);
  });
  popoutCanvas.addEventListener('mouseleave', () => {
    sceneApp.monitor.hoverNormX = null;
    if (sceneApp.monitor.lastPayload) sceneApp.monitor.render(sceneApp.monitor.lastPayload);
  });

  const drawer = document.getElementById('drawer');
  const btnDrawer = document.getElementById('btn-drawer');
  btnDrawer.addEventListener('click', () => {
    const collapsed = drawer.classList.toggle('collapsed');
    btnDrawer.classList.toggle('active', !collapsed);
  });

  window.__LEVISIM_CAVITY__ = { state, sceneApp, recomputeAndRender };

  syncSlidersFromState();
  recomputeAndRender();
  detectPythonBackend();
}

initUI();

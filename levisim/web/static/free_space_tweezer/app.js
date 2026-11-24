/**
 * Application controller wiring the Free-Space Tweezer Three.js bench,
 * hybrid JS (60 FPS) / Python levisim backend physics, diegetic 3D controls,
 * precision drawer sliders, presets, and the in-game RF spectrum monitor.
 */
import {
  DEFAULT_TWEEZER_PARAMS,
  computeTweezerSpectraJS,
  generateSyntheticEstimate,
} from './tweezer_physics.js';
import { FreeSpaceTweezerScene } from './tweezer_scene.js';

const PRESETS = {
  prr6_pos: {
    label: 'PRR 6 (+45°)',
    params: { ...DEFAULT_TWEEZER_PARAMS, psi_deg: 45, beta: Math.SQRT2, photon_recoil: false },
  },
  prr6_neg: {
    label: 'PRR 6 (−45°)',
    params: { ...DEFAULT_TWEEZER_PARAMS, psi_deg: -45, beta: Math.SQRT2, photon_recoil: false },
  },
  thermal: {
    label: 'Thermal (β=0)',
    params: { ...DEFAULT_TWEEZER_PARAMS, beta: 0.0, gamma_fbx_kHz: 0.5, gamma_fby_kHz: 0.5, photon_recoil: false },
  },
  deep_cool: {
    label: 'Deep Cooling',
    params: { ...DEFAULT_TWEEZER_PARAMS, pressure_mbar: 1e-5, gamma_fbx_kHz: 6.5, gamma_fby_kHz: 7.2, beta: 0.4, detection_floor: 1e-29 },
  },
  recoil: {
    label: 'Photon Recoil',
    params: { ...DEFAULT_TWEEZER_PARAMS, radius_nm: 75.0, power_mW: 650, photon_recoil: true },
  },
};

const state = {
  params: { ...DEFAULT_TWEEZER_PARAMS },
  monitorView: 'motional', // 'motional' | 'optical' | 'both'
  showSimulated: true,
  backendAvailable: false,
  usePythonBackend: false,
  pythonSimOverlay: null,
  activePreset: 'prr6_pos',
};

let pendingPythonTimer = null;
let sceneApp = null;

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function buildMonitorPayload(spectra, simDots) {
  const tel = spectra.telemetry;
  const p = state.params;
  const refs = [
    { x: tel.fz_kHz, label: 'Ωz' },
    { x: tel.fy_kHz, label: 'Ωy' },
    { x: tel.fx_kHz, label: 'Ωx' },
  ];

  const panels = [];

  if (state.monitorView === 'motional' || state.monitorView === 'both') {
    const psdSeries = [
      { ys: spectra.Sxx, label: 'Sxx (m²/Hz)', colorIndex: 0 },
      { ys: spectra.Syy, label: 'Syy (m²/Hz)', colorIndex: 1 },
      { ys: spectra.Szz, label: 'Szz', colorIndex: 3, dashed: true, lineWidth: 2.0 },
    ];
    if (state.showSimulated && simDots) {
      psdSeries.push({ xs: simDots.f_kHz, ys: simDots.Sxx, label: 'sim', colorIndex: 0, dots: true });
      psdSeries.push({ xs: simDots.f_kHz, ys: simDots.Syy, label: '', colorIndex: 1, dots: true });
    }
    const maxVal = Math.max(...spectra.Sxx, ...spectra.Syy, 1e-24);
    const yMax = Math.ceil(Math.log10(maxVal)) + 0.5;
    const yMin = Math.floor(Math.log10(Math.min(p.detection_floor, maxVal * 1e-5))) - 0.5;
    panels.push({
      title: 'MOTIONAL PSD · Sxx, Syy, Szz (one-sided)',
      xLabel: state.monitorView === 'both' ? '' : '',
      yLabel: 'PSD (m²/Hz)',
      logY: true,
      yMin,
      yMax,
      xs: spectra.f_kHz,
      refs,
      floorValue: p.detection_floor,
      series: psdSeries,
    });
  }

  if (state.monitorView === 'motional') {
    const maxC = Math.max(1e-26, ...spectra.Cxy.map(Math.abs), ...spectra.Qxy.map(Math.abs)) * 1.18;
    const cSeries = [
      { ys: spectra.Cxy, label: 'Cxy (co-spectrum)', colorIndex: 2 },
      { ys: spectra.Qxy, label: 'Qxy (quad)', colorIndex: 4, dashed: true, lineWidth: 2.0 },
    ];
    if (state.showSimulated && simDots) {
      cSeries.push({ xs: simDots.f_kHz, ys: simDots.Cxy, label: 'sim Cxy', colorIndex: 2, dots: true });
    }
    panels.push({
      title: `DIRECTIONAL CROSS-SPECTRUM · ψ = ${p.psi_deg >= 0 ? '+' : ''}${p.psi_deg.toFixed(1)}°, β = ${p.beta.toFixed(2)}`,
      xLabel: 'ω / 2π (kHz)',
      yLabel: 'Cxy (m²/Hz)',
      logY: false,
      zeroLine: true,
      yMin: -maxC,
      yMax: maxC,
      xs: spectra.f_kHz,
      refs,
      series: cSeries,
    });
  }

  if (state.monitorView === 'optical' || state.monitorView === 'both') {
    const maxOpt = Math.max(10, ...spectra.S_opt_x, ...spectra.S_opt_y);
    panels.push({
      title: 'OPTICAL BALANCED DETECTOR · Photocurrent PSD (Shot-Noise Units)',
      xLabel: 'ω / 2π (kHz)',
      yLabel: 'S_opt (SNU)',
      logY: true,
      yMin: -0.2,
      yMax: Math.max(2.0, Math.ceil(Math.log10(maxOpt)) + 0.4),
      xs: spectra.f_kHz,
      refs,
      floorValue: 1.0,
      series: [
        { ys: spectra.S_opt_x, label: 'x-arm (SNU)', colorIndex: 0, fill: true },
        { ys: spectra.S_opt_y, label: 'y-arm (SNU)', colorIndex: 1 },
        { ys: spectra.S_opt_z, label: 'z-arm', colorIndex: 3, dashed: true, lineWidth: 2.0 },
      ],
    });
  }

  const engineTag = spectra.engine === 'python-levisim' ? 'PYTHON LEVISIM' : 'LIVE 60 FPS';
  return {
    title: 'LEVISIM · FREE-SPACE OPTICAL TWEEZER (Trap)',
    statusText: engineTag,
    statusColor: spectra.engine === 'python-levisim' ? '#fbbf24' : '#10b981',
    footer:
      `Ωx=${tel.fx_kHz.toFixed(1)} kHz  Ωy=${tel.fy_kHz.toFixed(1)} kHz  Ωz=${tel.fz_kHz.toFixed(1)} kHz  |  ` +
      `Teff,x=${tel.T_eff_x_K.toFixed(2)} K  Teff,y=${tel.T_eff_y_K.toFixed(2)} K  |  γxy=${tel.coherence_xy.toFixed(2)}  |  z_eq=${tel.z_eq_nm.toFixed(1)} nm`,
    panels,
  };
}

function updateTelemetryDOM(tel) {
  document.getElementById('ro-fx').textContent = `${tel.fx_kHz.toFixed(2)} kHz`;
  document.getElementById('ro-fy').textContent = `${tel.fy_kHz.toFixed(2)} kHz`;
  document.getElementById('ro-fz').textContent = `${tel.fz_kHz.toFixed(2)} kHz`;
  document.getElementById('ro-tx').textContent = tel.T_eff_x_K < 0.01 ? `${(tel.T_eff_x_K * 1e3).toFixed(2)} mK` : `${tel.T_eff_x_K.toFixed(2)} K`;
  document.getElementById('ro-ty').textContent = tel.T_eff_y_K < 0.01 ? `${(tel.T_eff_y_K * 1e3).toFixed(2)} mK` : `${tel.T_eff_y_K.toFixed(2)} K`;
  document.getElementById('ro-gth').textContent = tel.gamma_th_Hz < 0.1 ? `${(tel.gamma_th_Hz * 1e3).toFixed(2)} mHz` : `${tel.gamma_th_Hz.toFixed(2)} Hz`;
  document.getElementById('ro-coh').textContent = tel.coherence_xy.toFixed(3);
  document.getElementById('ro-zeq').textContent = `${tel.z_eq_nm.toFixed(1)} nm`;
}

function syncSlidersFromState() {
  const p = state.params;
  const set = (id, val, labelId, text) => {
    const el = document.getElementById(id);
    if (el) el.value = val;
    const lbl = document.getElementById(labelId);
    if (lbl) lbl.textContent = text;
  };
  set('sl-power', p.power_mW, 'val-power', `${p.power_mW.toFixed(0)} mW`);
  set('sl-lambda', p.wavelength_nm, 'val-lambda', `${p.wavelength_nm.toFixed(0)} nm`);
  set('sl-wx', p.wx0_um, 'val-wx', `${p.wx0_um.toFixed(3)} μm`);
  set('sl-wy', p.wy0_um, 'val-wy', `${p.wy0_um.toFixed(3)} μm`);
  set('sl-radius', p.radius_nm, 'val-radius', `${p.radius_nm.toFixed(1)} nm`);
  set('sl-logp', Math.log10(p.pressure_mbar), 'val-logp', `${p.pressure_mbar.toExponential(1)} mbar`);
  set('sl-gfbx', p.gamma_fbx_kHz, 'val-gfbx', `${p.gamma_fbx_kHz.toFixed(2)} kHz`);
  set('sl-gfby', p.gamma_fby_kHz, 'val-gfby', `${p.gamma_fby_kHz.toFixed(2)} kHz`);
  set('sl-beta', p.beta, 'val-beta', p.beta.toFixed(2));
  set('sl-psi', p.psi_deg, 'val-psi', `${p.psi_deg >= 0 ? '+' : ''}${p.psi_deg.toFixed(1)}°`);
  set('sl-floor', Math.log10(p.detection_floor), 'val-floor', `10^${Math.log10(p.detection_floor).toFixed(1)} m²/Hz`);

  document.getElementById('chk-recoil').checked = Boolean(p.photon_recoil);
  document.getElementById('chk-malitson').checked = Boolean(p.use_malitson);
  document.getElementById('chk-sim').checked = Boolean(state.showSimulated);
}

function recomputeAndRender() {
  // Always update instantaneously at 60 FPS with exact JS physics first
  const spectra = computeTweezerSpectraJS(state.params);
  const simDots = state.pythonSimOverlay || (state.showSimulated ? generateSyntheticEstimate(spectra, 40, 2026) : null);

  sceneApp.syncVisuals(state.params, spectra.telemetry);
  sceneApp.monitor.render(buildMonitorPayload(spectra, simDots));
  updateTelemetryDOM(spectra.telemetry);

  // If connected to local Python levisim-web server and enabled, verify/refresh via Python API
  if (state.backendAvailable && state.usePythonBackend) {
    clearTimeout(pendingPythonTimer);
    pendingPythonTimer = setTimeout(async () => {
      try {
        const res = await fetch('/api/tweezer/spectra', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(state.params),
        });
        if (!res.ok) return;
        const pySpectra = await res.json();
        const pyDots = state.pythonSimOverlay || (state.showSimulated ? generateSyntheticEstimate(pySpectra, 40, 2026) : null);
        sceneApp.monitor.render(buildMonitorPayload(pySpectra, pyDots));
        updateTelemetryDOM(pySpectra.telemetry);
      } catch (_) {
        // Stay gracefully on JS engine if server disconnects
      }
    }, 80);
  }
}

async function detectPythonBackend() {
  const badge = document.getElementById('engine-badge');
  const rkBtn = document.getElementById('btn-run-rk');
  try {
    const res = await fetch('/api/health');
    if (res.ok) {
      const info = await res.json();
      if (info.ok) {
        state.backendAvailable = true;
        state.usePythonBackend = true;
        badge.textContent = 'Python levisim + 60 FPS JS';
        badge.classList.add('python');
        if (rkBtn) rkBtn.style.display = 'inline-flex';
        recomputeAndRender();
      }
    }
  } catch (_) {
    // Pure static GitHub Pages mode — 60 FPS JS engine active
  }
}

function initUI() {
  const container = document.getElementById('viewport');
  const tooltip = document.getElementById('diegetic-tooltip');

  sceneApp = new FreeSpaceTweezerScene(container, tooltip, {
    onDeltaParam: (key, delta) => {
      state.pythonSimOverlay = null;
      if (key === 'power_mW') state.params.power_mW = clamp(state.params.power_mW + delta, 50, 1500);
      if (key === 'wx0_um') state.params.wx0_um = clamp(state.params.wx0_um + delta, 0.55, 1.45);
      if (key === 'psi_deg') state.params.psi_deg = clamp(state.params.psi_deg + delta, -90, 90);
      syncSlidersFromState();
      recomputeAndRender();
    },
    onDeltaLogPressure: (dLog) => {
      state.pythonSimOverlay = null;
      const lp = clamp(Math.log10(state.params.pressure_mbar) + dLog, -7, 0);
      state.params.pressure_mbar = Math.pow(10, lp);
      syncSlidersFromState();
      recomputeAndRender();
    },
    onDeltaFeedback: (dkHz) => {
      state.pythonSimOverlay = null;
      state.params.gamma_fbx_kHz = clamp(state.params.gamma_fbx_kHz + dkHz, 0, 15);
      state.params.gamma_fby_kHz = clamp(state.params.gamma_fby_kHz + dkHz * 1.15, 0, 15);
      syncSlidersFromState();
      recomputeAndRender();
    },
  });

  // Wire precision drawer sliders
  const bindSlider = (id, handler) => {
    document.getElementById(id).addEventListener('input', (ev) => {
      state.pythonSimOverlay = null;
      handler(parseFloat(ev.target.value));
      syncSlidersFromState();
      recomputeAndRender();
    });
  };
  bindSlider('sl-power', v => { state.params.power_mW = v; });
  bindSlider('sl-lambda', v => { state.params.wavelength_nm = v; });
  bindSlider('sl-wx', v => { state.params.wx0_um = v; });
  bindSlider('sl-wy', v => { state.params.wy0_um = v; });
  bindSlider('sl-radius', v => { state.params.radius_nm = v; });
  bindSlider('sl-logp', v => { state.params.pressure_mbar = Math.pow(10, v); });
  bindSlider('sl-gfbx', v => { state.params.gamma_fbx_kHz = v; });
  bindSlider('sl-gfby', v => { state.params.gamma_fby_kHz = v; });
  bindSlider('sl-beta', v => { state.params.beta = v; });
  bindSlider('sl-psi', v => { state.params.psi_deg = v; });
  bindSlider('sl-floor', v => { state.params.detection_floor = Math.pow(10, v); });

  document.getElementById('chk-recoil').addEventListener('change', (ev) => {
    state.pythonSimOverlay = null;
    state.params.photon_recoil = ev.target.checked;
    recomputeAndRender();
  });
  document.getElementById('chk-malitson').addEventListener('change', (ev) => {
    state.pythonSimOverlay = null;
    state.params.use_malitson = ev.target.checked;
    recomputeAndRender();
  });
  document.getElementById('chk-sim').addEventListener('change', (ev) => {
    state.showSimulated = ev.target.checked;
    recomputeAndRender();
  });

  // Paper preset buttons
  const presetBox = document.getElementById('preset-grid');
  Object.entries(PRESETS).forEach(([key, preset]) => {
    const btn = document.createElement('button');
    btn.className = 'preset-btn' + (key === state.activePreset ? ' active' : '');
    btn.textContent = preset.label;
    btn.addEventListener('click', () => {
      state.activePreset = key;
      state.pythonSimOverlay = null;
      state.params = { ...preset.params };
      presetBox.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
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

  // Chamber visibility mode (Full / Cutaway / Open)
  const chamberSel = document.getElementById('sel-chamber');
  chamberSel.addEventListener('change', () => sceneApp.setChamberMode(chamberSel.value));

  // Monitor power On/Off toggle
  const btnMonPower = document.getElementById('btn-mon-power');
  btnMonPower.addEventListener('click', () => {
    const next = !sceneApp.monitor.powerOn;
    sceneApp.monitor.setPower(next);
    btnMonPower.classList.toggle('active', next);
    btnMonPower.textContent = next ? 'Monitor: ON' : 'Monitor: OFF';
  });

  // Monitor theme switcher (RF Analyzer / levisim xmgrace / Quantum OLED)
  const selTheme = document.getElementById('sel-theme');
  selTheme.addEventListener('change', () => sceneApp.monitor.setTheme(selTheme.value));

  // Monitor spectrum tab switcher (Motional+Cxy / Optical / Combined)
  const selView = document.getElementById('sel-view');
  selView.addEventListener('change', () => {
    state.monitorView = selView.value;
    recomputeAndRender();
  });

  // 2D Pop-Out Spectrum Analyzer Overlay
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

  // Precision drawer collapse toggle
  const drawer = document.getElementById('drawer');
  const btnDrawer = document.getElementById('btn-drawer');
  btnDrawer.addEventListener('click', () => {
    const collapsed = drawer.classList.toggle('collapsed');
    btnDrawer.classList.toggle('active', !collapsed);
  });

  // Optional live Python stochastic Langevin simulation button (visible when levisim-web server is active)
  const rkBtn = document.getElementById('btn-run-rk');
  if (rkBtn) {
    rkBtn.addEventListener('click', async () => {
      rkBtn.disabled = true;
      rkBtn.textContent = 'Running Python Simulation...';
      try {
        const res = await fetch('/api/tweezer/simulate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(state.params),
        });
        if (res.ok) {
          state.pythonSimOverlay = await res.json();
          state.showSimulated = true;
          syncSlidersFromState();
          recomputeAndRender();
        }
      } finally {
        rkBtn.disabled = false;
        rkBtn.textContent = 'Run Python RK Simulation';
      }
    });
  }

  // Expose testing hook for automated visual/physics checks
  window.__LEVISIM_TWEEZER__ = { state, sceneApp, recomputeAndRender };

  syncSlidersFromState();
  recomputeAndRender();
  detectPythonBackend();
}

initUI();

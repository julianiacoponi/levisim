/**
 * Live In-Game RF Spectrum Analyzer Monitor (Three.js benchtop instrument rig +
 * 1024x640 CanvasTexture + synced 2D pop-out overlay) supporting:
 *   - 3 themes: 'rf' (Keysight/R&S default), 'xmgrace' (levisim publication), 'glass' (Quantum OLED)
 *   - Monitor Power On/Off toggle (physically powers down the 3D screen & LED)
 *   - Dual-panel log/linear plots, bare-Ω dotted reference lines, simulated FFT dots,
 *     detection floor lines, and complex drift-eigenvalue stability insets.
 */
import * as THREE from 'three';
import { RoundedBoxGeometry } from 'three/addons/geometries/RoundedBoxGeometry.js';

export const MONITOR_THEMES = {
  rf: {
    name: 'RF Analyzer',
    bg: '#091017',
    headerBg: '#101a26',
    grid: 'rgba(56, 189, 248, 0.16)',
    axis: 'rgba(148, 163, 184, 0.65)',
    text: '#94a3b8',
    label: '#e2e8f0',
    ref: 'rgba(250, 204, 21, 0.55)',
    refText: '#fde68a',
    series: ['#38bdf8', '#fb923c', '#34d399', '#c084fc', '#f43f5e'],
    fill: 'rgba(56, 189, 248, 0.16)',
    floor: 'rgba(148, 163, 184, 0.55)',
    font: 'ui-monospace, SFMono-Regular, Menlo, monospace',
    chassisColor: 0x161c26,
    emissiveIntensity: 0.90,
  },
  xmgrace: {
    name: 'levisim xmgrace',
    bg: '#ffffff',
    headerBg: '#edf2f7',
    grid: 'rgba(15, 23, 42, 0.10)',
    axis: '#0f172a',
    text: '#334155',
    label: '#0f172a',
    ref: 'rgba(71, 85, 105, 0.72)',
    refText: '#1e293b',
    series: ['#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#ff7f0e'],
    fill: 'rgba(31, 119, 180, 0.06)',
    floor: 'rgba(100, 116, 139, 0.75)',
    font: 'Georgia, "Times New Roman", serif',
    chassisColor: 0x262f3d,
    emissiveIntensity: 0.62,
  },
  glass: {
    name: 'Quantum OLED',
    bg: '#06080f',
    headerBg: '#0c101d',
    grid: 'rgba(148, 163, 184, 0.12)',
    axis: 'rgba(203, 213, 225, 0.42)',
    text: '#94a3b8',
    label: '#f8fafc',
    ref: 'rgba(248, 250, 252, 0.35)',
    refText: '#e2e8f0',
    series: ['#22d3ee', '#f472b6', '#a78bfa', '#34d399', '#fbbf24'],
    fill: 'rgba(34, 211, 238, 0.20)',
    floor: 'rgba(148, 163, 184, 0.45)',
    font: 'system-ui, -apple-system, BlinkMacSystemFont, sans-serif',
    chassisColor: 0x0e1118,
    emissiveIntensity: 0.95,
  },
};

export class SpectrumMonitor {
  constructor(mats, {
    position = new THREE.Vector3(3.1, 2.38, -2.85),
    rotationY = -0.28,
    width = 4.65,
    height = 3.0,
    theme = 'rf',
  } = {}) {
    this.width = 1024;
    this.height = 640;
    this.themeKey = theme;
    this.powerOn = true;
    this.lastPayload = null;
    this.popoutCanvas = null;
    this.hoverNormX = null;

    // Offscreen 2D canvas feeding the 3D monitor CanvasTexture
    this.canvas = document.createElement('canvas');
    this.canvas.width = this.width;
    this.canvas.height = this.height;
    this.ctx = this.canvas.getContext('2d');

    this.texture = new THREE.CanvasTexture(this.canvas);
    this.texture.colorSpace = THREE.SRGBColorSpace;
    this.texture.anisotropy = 16;

    // Build photorealistic Keysight/R&S-inspired benchtop instrument rig
    this.group = new THREE.Group();
    this.group.position.copy(position);
    this.group.rotation.y = rotationY;

    this.chassisMat = new THREE.MeshStandardMaterial({
      color: MONITOR_THEMES[theme].chassisColor,
      metalness: 0.58,
      roughness: 0.36,
    });
    const body = new THREE.Mesh(new RoundedBoxGeometry(width, height, 0.36, 4, 0.05), this.chassisMat);
    body.castShadow = true;
    body.receiveShadow = true;
    this.group.add(body);

    // Subtle recessed metallic inner bezel frame around screen
    const screenW = width - 0.56;
    const screenH = height - 0.34;
    const bezelFrame = new THREE.Mesh(
      new RoundedBoxGeometry(screenW + 0.08, screenH + 0.08, 0.06, 2, 0.02),
      mats.anodizedRing
    );
    bezelFrame.position.set(-0.14, 0.02, 0.17);
    this.group.add(bezelFrame);

    // Anti-glare monitor screen plane
    this.screenMat = new THREE.MeshStandardMaterial({
      map: this.texture,
      emissiveMap: this.texture,
      emissive: new THREE.Color(0xffffff),
      emissiveIntensity: MONITOR_THEMES[theme].emissiveIntensity,
      roughness: 0.72,
      metalness: 0.04,
    });
    this.screenMesh = new THREE.Mesh(new THREE.PlaneGeometry(screenW, screenH), this.screenMat);
    this.screenMesh.position.set(-0.14, 0.02, 0.204);
    this.group.add(this.screenMesh);

    // Right bezel tactile soft-key buttons + rotary encoder knob (authentic RF analyzer look!)
    for (let i = 0; i < 6; i++) {
      const btn = new THREE.Mesh(new RoundedBoxGeometry(0.18, 0.14, 0.06, 2, 0.015), mats.anodizedRing);
      btn.position.set(width / 2 - 0.19, height / 2 - 0.44 - i * 0.32, 0.19);
      this.group.add(btn);
    }
    const encKnob = new THREE.Mesh(new THREE.CylinderGeometry(0.11, 0.12, 0.12, 24), mats.stainless);
    encKnob.rotation.x = Math.PI / 2;
    encKnob.position.set(width / 2 - 0.19, -height / 2 + 0.62, 0.22);
    this.group.add(encKnob);

    // Front power LED + RF BNC input jack
    this.powerLedMat = new THREE.MeshBasicMaterial({ color: 0x22c55e });
    const powerLed = new THREE.Mesh(new THREE.SphereGeometry(0.036, 12, 12), this.powerLedMat);
    powerLed.position.set(width / 2 - 0.19, -height / 2 + 0.24, 0.20);
    this.group.add(powerLed);

    const bncPort = new THREE.Mesh(new THREE.CylinderGeometry(0.055, 0.055, 0.14, 16), mats.stainless);
    bncPort.rotation.x = Math.PI / 2;
    bncPort.position.set(width / 2 - 0.62, -height / 2 + 0.12, 0.20);
    this.group.add(bncPort);

    // Heavy machined stand resting on optical breadboard
    const standH = Math.max(0.3, position.y - height / 2);
    const neck = new THREE.Mesh(new THREE.CylinderGeometry(0.18, 0.24, standH + 0.4, 24), mats.anodized);
    neck.position.set(0, -height / 2 - standH / 2 + 0.15, -0.14);
    neck.castShadow = true;
    this.group.add(neck);

    const foot = new THREE.Mesh(new RoundedBoxGeometry(2.0, 0.12, 1.2, 3, 0.03), mats.anodizedRing);
    foot.position.set(0, -position.y + 0.06, 0.02);
    foot.castShadow = true;
    foot.receiveShadow = true;
    this.group.add(foot);
  }

  setTheme(themeKey) {
    if (!MONITOR_THEMES[themeKey]) return;
    this.themeKey = themeKey;
    const th = MONITOR_THEMES[themeKey];
    this.chassisMat.color.setHex(th.chassisColor);
    if (this.powerOn) {
      this.screenMat.emissiveIntensity = th.emissiveIntensity;
    }
    if (this.lastPayload) this.render(this.lastPayload);
  }

  setPower(on) {
    this.powerOn = Boolean(on);
    const th = MONITOR_THEMES[this.themeKey];
    this.powerLedMat.color.setHex(this.powerOn ? 0x22c55e : 0xf59e0b);
    this.screenMat.emissiveIntensity = this.powerOn ? th.emissiveIntensity : 0.08;
    if (this.lastPayload) {
      this.render(this.lastPayload);
    } else {
      this._drawStandby(this.ctx, this.width, this.height);
      this.texture.needsUpdate = true;
    }
  }

  attachPopoutCanvas(canvasEl) {
    this.popoutCanvas = canvasEl;
    if (this.lastPayload) this.render(this.lastPayload);
  }

  _drawStandby(ctx, W, H) {
    ctx.fillStyle = '#070a0f';
    ctx.fillRect(0, 0, W, H);
    ctx.strokeStyle = 'rgba(148,163,184,0.12)';
    ctx.lineWidth = 2;
    ctx.strokeRect(24, 24, W - 48, H - 48);
    ctx.fillStyle = 'rgba(148,163,184,0.45)';
    ctx.font = '600 24px ui-monospace, Menlo, monospace';
    ctx.textAlign = 'center';
    ctx.fillText('MONITOR STANDBY — POWER OFF', W / 2, H / 2 - 8);
    ctx.font = '16px ui-monospace, Menlo, monospace';
    ctx.fillStyle = 'rgba(148,163,184,0.32)';
    ctx.fillText('Toggle "Monitor" in top bar or open 2D Analyzer to resume display', W / 2, H / 2 + 26);
  }

  /**
   * Renders a structured spectrum payload onto the 3D in-game monitor CanvasTexture
   * (and mirrored onto the 2D pop-out canvas whenever visible).
   *
   * Payload shape:
   * {
   *   title: string,
   *   statusText: string,
   *   statusColor: '#10b981' | '#ef4444' | '#38bdf8',
   *   footer: string,
   *   panels: [
   *     {
   *       title, xLabel, yLabel, logY, yMin, yMax, zeroLine,
   *       xs: number[],
   *       refs: [{ x, label }],
   *       floorValue?: number,
   *       series: [{ ys, label, colorIndex, dashed?, fill?, dots? }],
   *     }, ...
   *   ],
   *   eigenInset?: { eigenvalues: [{ re, im }], isStable: boolean }
   * }
   */
  render(payload) {
    this.lastPayload = payload;
    if (!this.powerOn) {
      this._drawStandby(this.ctx, this.width, this.height);
      this.texture.needsUpdate = true;
    } else {
      this._drawToContext(this.ctx, this.width, this.height, payload);
      this.texture.needsUpdate = true;
    }

    if (this.popoutCanvas && this.popoutCanvas.offsetParent !== null) {
      const pCtx = this.popoutCanvas.getContext('2d');
      this._drawToContext(pCtx, this.popoutCanvas.width, this.popoutCanvas.height, payload, this.hoverNormX);
    }
  }

  _drawToContext(ctx, W, H, payload, hoverNormX = null) {
    const th = MONITOR_THEMES[this.themeKey];
    const isLight = this.themeKey === 'xmgrace';

    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = th.bg;
    ctx.fillRect(0, 0, W, H);

    // Header strip
    const headerH = 52;
    ctx.fillStyle = th.headerBg;
    ctx.fillRect(0, 0, W, headerH);
    ctx.fillStyle = th.label;
    ctx.font = `700 20px ${th.font}`;
    ctx.textAlign = 'left';
    ctx.fillText(payload.title || 'LEVISIM SPECTRUM ANALYZER', 24, 33);

    // Status pill (right)
    if (payload.statusText) {
      const col = payload.statusColor || '#10b981';
      ctx.font = `700 17px ui-monospace, Menlo, monospace`;
      const tw = ctx.measureText(payload.statusText).width;
      ctx.fillStyle = col;
      ctx.beginPath();
      ctx.arc(W - tw - 40, 27, 6.5, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = isLight && col === '#10b981' ? '#047857' : col;
      ctx.textAlign = 'right';
      ctx.fillText(payload.statusText, W - 22, 33);
    }

    // Layout 1 or 2 stacked or side-by-side panels
    const panels = payload.panels || [];
    const padL = 94, padR = 26, padT = headerH + 22, padB = 74;
    const availW = W - padL - padR;
    const availH = H - padT - padB;

    if (panels.length === 1) {
      this._drawPanel(ctx, th, padL, padT, availW, availH, panels[0], hoverNormX);
    } else if (panels.length >= 2) {
      const gap = 58;
      const ph = (availH - gap) / 2;
      this._drawPanel(ctx, th, padL, padT, availW, ph, panels[0], hoverNormX);
      this._drawPanel(ctx, th, padL, padT + ph + gap, availW, ph, panels[1], hoverNormX);
    }

    // Optional stability complex eigenvalue inset (top-right inside plot 1)
    if (payload.eigenInset) {
      this._drawEigenInset(ctx, th, W - 242, padT + 14, 206, 134, payload.eigenInset);
    }

    // Footer telemetry strip
    ctx.fillStyle = th.headerBg;
    ctx.fillRect(0, H - 36, W, 36);
    ctx.fillStyle = th.text;
    ctx.font = `16px ui-monospace, Menlo, monospace`;
    ctx.textAlign = 'left';
    ctx.fillText(payload.footer || '', 24, H - 12);
  }

  _drawPanel(ctx, th, x0, y0, w, h, panel, hoverNormX) {
    const xs = panel.xs;
    if (!xs || xs.length < 2) return;
    const xMin = xs[0];
    const xMax = xs[xs.length - 1];
    const yMin = panel.yMin;
    const yMax = panel.yMax;
    const isLight = this.themeKey === 'xmgrace';

    const mapX = v => x0 + ((v - xMin) / (xMax - xMin)) * w;
    const mapY = v => {
      const val = panel.logY ? Math.log10(Math.max(v, 1e-40)) : v;
      const clamped = Math.max(yMin, Math.min(yMax, val));
      return y0 + h - ((clamped - yMin) / (yMax - yMin)) * h;
    };

    // Grid lines
    ctx.strokeStyle = th.grid;
    ctx.lineWidth = 1.25;
    const yTicks = 4;
    for (let i = 0; i <= yTicks; i++) {
      const gy = y0 + (h * i) / yTicks;
      ctx.beginPath();
      ctx.moveTo(x0, gy);
      ctx.lineTo(x0 + w, gy);
      ctx.stroke();
    }
    const xTicks = 6;
    for (let i = 0; i <= xTicks; i++) {
      const gx = x0 + (w * i) / xTicks;
      ctx.beginPath();
      ctx.moveTo(gx, y0);
      ctx.lineTo(gx, y0 + h);
      ctx.stroke();
    }

    // Bare mechanical frequency dotted reference lines
    if (panel.refs && panel.refs.length) {
      ctx.save();
      ctx.strokeStyle = th.ref;
      ctx.setLineDash([5, 5]);
      ctx.lineWidth = 1.8;
      ctx.font = `600 15px ${th.font}`;
      ctx.fillStyle = th.refText;
      ctx.textAlign = 'center';
      panel.refs.forEach((r, idx) => {
        if (r.x >= xMin && r.x <= xMax) {
          const rx = mapX(r.x);
          ctx.beginPath();
          ctx.moveTo(rx, y0);
          ctx.lineTo(rx, y0 + h);
          ctx.stroke();
          if (r.label) {
            ctx.fillText(r.label, rx, y0 + 18 + (idx % 2) * 16);
          }
        }
      });
      ctx.restore();
    }

    // Optional zero line for bipolar cross-spectra
    if (panel.zeroLine && yMin < 0 && yMax > 0) {
      const zy = mapY(0);
      ctx.strokeStyle = th.axis;
      ctx.lineWidth = 1.4;
      ctx.beginPath();
      ctx.moveTo(x0, zy);
      ctx.lineTo(x0 + w, zy);
      ctx.stroke();
    }

    // Optional horizontal detection / shot-noise floor line
    if (panel.floorValue !== undefined && panel.floorValue !== null) {
      const fy = mapY(panel.floorValue);
      if (fy >= y0 && fy <= y0 + h) {
        ctx.save();
        ctx.strokeStyle = th.floor;
        ctx.setLineDash([8, 6]);
        ctx.lineWidth = 1.6;
        ctx.beginPath();
        ctx.moveTo(x0, fy);
        ctx.lineTo(x0 + w, fy);
        ctx.stroke();
        ctx.restore();
      }
    }

    // Clip curves cleanly inside frame
    ctx.save();
    ctx.beginPath();
    ctx.rect(x0, y0, w, h);
    ctx.clip();

    for (const s of panel.series) {
      const col = s.color || th.series[s.colorIndex % th.series.length];
      if (s.dots) {
        // Scatter dots (e.g., simulated Langevin periodogram points)
        ctx.fillStyle = col;
        const step = Math.max(1, Math.floor(s.ys.length / 180));
        for (let i = 0; i < s.ys.length; i += step) {
          const px = mapX(s.xs ? s.xs[i] : xs[i]);
          const py = mapY(s.ys[i]);
          ctx.beginPath();
          ctx.arc(px, py, 2.7, 0, Math.PI * 2);
          ctx.fill();
        }
        continue;
      }

      ctx.save();
      if (s.dashed) ctx.setLineDash([7, 5]);
      ctx.beginPath();
      const sxs = s.xs || xs;
      for (let i = 0; i < s.ys.length; i++) {
        const px = mapX(sxs[i]);
        const py = mapY(s.ys[i]);
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }
      ctx.strokeStyle = col;
      ctx.lineWidth = s.lineWidth || (isLight ? 2.4 : 3.0);
      ctx.stroke();

      if (s.fill && !isLight) {
        ctx.lineTo(mapX(sxs[sxs.length - 1]), y0 + h);
        ctx.lineTo(mapX(sxs[0]), y0 + h);
        ctx.closePath();
        const grad = ctx.createLinearGradient(0, y0, 0, y0 + h);
        grad.addColorStop(0, th.fill);
        grad.addColorStop(1, 'rgba(0,0,0,0)');
        ctx.fillStyle = grad;
        ctx.fill();
      }
      ctx.restore();
    }
    ctx.restore();

    // Frame border + inward ticks (xmgrace style!)
    ctx.strokeStyle = th.axis;
    ctx.lineWidth = isLight ? 2.4 : 1.8;
    ctx.strokeRect(x0, y0, w, h);

    // X ticks & labels
    ctx.fillStyle = th.text;
    ctx.font = `15px ${th.font}`;
    ctx.textAlign = 'center';
    for (let i = 0; i <= xTicks; i++) {
      const xv = xMin + ((xMax - xMin) * i) / xTicks;
      const gx = x0 + (w * i) / xTicks;
      ctx.beginPath();
      ctx.moveTo(gx, y0 + h);
      ctx.lineTo(gx, y0 + h - 7);
      ctx.moveTo(gx, y0);
      ctx.lineTo(gx, y0 + 7);
      ctx.stroke();
      ctx.fillText(xv.toFixed(0), gx, y0 + h + 19);
    }
    if (panel.xLabel) {
      ctx.fillStyle = th.label;
      ctx.font = `600 16px ${th.font}`;
      ctx.fillText(panel.xLabel, x0 + w / 2, y0 + h + 38);
    }

    // Y ticks & labels
    ctx.fillStyle = th.text;
    ctx.font = `15px ${th.font}`;
    ctx.textAlign = 'right';
    for (let i = 0; i <= yTicks; i++) {
      const yv = yMax - ((yMax - yMin) * i) / yTicks;
      const gy = y0 + (h * i) / yTicks;
      ctx.beginPath();
      ctx.moveTo(x0, gy);
      ctx.lineTo(x0 + 7, gy);
      ctx.moveTo(x0 + w, gy);
      ctx.lineTo(x0 + w - 7, gy);
      ctx.stroke();
      let txt;
      if (panel.logY) {
        txt = `10^${Math.round(yv)}`;
      } else if (Math.abs(yv) < 1e-35) {
        txt = '0';
      } else if (Math.abs(yv) >= 1e4 || Math.abs(yv) < 1e-2) {
        txt = yv.toExponential(1);
      } else {
        txt = yv.toFixed(1);
      }
      ctx.fillText(txt, x0 - 10, gy + 5);
    }

    // Rotated Y-axis title
    if (panel.yLabel) {
      ctx.save();
      ctx.translate(22, y0 + h / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.fillStyle = th.label;
      ctx.font = `600 16px ${th.font}`;
      ctx.textAlign = 'center';
      ctx.fillText(panel.yLabel, 0, 0);
      ctx.restore();
    }

    // Panel title (top-left) and legend (top-right)
    ctx.font = `700 16px ${th.font}`;
    ctx.textAlign = 'left';
    ctx.fillStyle = th.label;
    if (panel.title) ctx.fillText(panel.title, x0 + 10, y0 - 7);

    let lx = x0 + w;
    ctx.textAlign = 'right';
    ctx.font = `600 15px ${th.font}`;
    for (let i = panel.series.length - 1; i >= 0; i--) {
      const s = panel.series[i];
      if (!s.label) continue;
      const col = s.color || th.series[s.colorIndex % th.series.length];
      const tag = (s.dots ? '• ' : '— ') + s.label;
      ctx.fillStyle = col;
      ctx.fillText(tag, lx, y0 - 7);
      lx -= ctx.measureText(tag).width + 16;
    }

    // Interactive cursor crosshair on 2D pop-out
    if (hoverNormX !== null && hoverNormX >= 0 && hoverNormX <= 1) {
      const idx = Math.min(xs.length - 1, Math.max(0, Math.round(hoverNormX * (xs.length - 1))));
      const cx = mapX(xs[idx]);
      ctx.save();
      ctx.strokeStyle = isLight ? 'rgba(15,23,42,0.45)' : 'rgba(248,250,252,0.45)';
      ctx.setLineDash([4, 4]);
      ctx.lineWidth = 1.4;
      ctx.beginPath();
      ctx.moveTo(cx, y0);
      ctx.lineTo(cx, y0 + h);
      ctx.stroke();

      const primary = panel.series[0];
      if (primary && primary.ys[idx] !== undefined) {
        const val = primary.ys[idx];
        const txt = `${xs[idx].toFixed(2)} kHz · ${Math.abs(val) < 1e-2 || Math.abs(val) > 1e4 ? val.toExponential(2) : val.toFixed(2)}`;
        ctx.fillStyle = isLight ? 'rgba(15,23,42,0.88)' : 'rgba(15,23,42,0.92)';
        const bw = ctx.measureText(txt).width + 18;
        const bx = Math.min(x0 + w - bw - 4, Math.max(x0 + 4, cx + 8));
        ctx.fillRect(bx, y0 + 6, bw, 24);
        ctx.fillStyle = '#f8fafc';
        ctx.textAlign = 'left';
        ctx.font = '600 13px ui-monospace, Menlo, monospace';
        ctx.fillText(txt, bx + 9, y0 + 22);
      }
      ctx.restore();
    }
  }

  _drawEigenInset(ctx, th, x0, y0, w, h, inset) {
    const isLight = this.themeKey === 'xmgrace';
    ctx.save();
    ctx.fillStyle = isLight ? 'rgba(255,255,255,0.92)' : 'rgba(9,14,22,0.90)';
    ctx.fillRect(x0, y0, w, h);

    // Unstable right half-plane shading (Re λ > 0)
    const zeroX = x0 + w * 0.68;
    ctx.fillStyle = inset.isStable ? 'rgba(239,68,68,0.08)' : 'rgba(239,68,68,0.22)';
    ctx.fillRect(zeroX, y0, x0 + w - zeroX, h);

    // Stability boundary Re λ = 0
    ctx.strokeStyle = inset.isStable ? 'rgba(148,163,184,0.6)' : '#ef4444';
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    ctx.moveTo(zeroX, y0);
    ctx.lineTo(zeroX, y0 + h);
    ctx.stroke();

    // Plot mechanical eigenvalues (Im λ >= 0)
    const mechs = inset.eigenvalues || [];
    if (mechs.length) {
      const fVals = mechs.map(e => Math.abs(e.im));
      const fMin = Math.min(...fVals) - 8;
      const fMax = Math.max(...fVals) + 8;
      const reScale = Math.max(4000, ...mechs.map(e => Math.abs(e.re))) * 1.15;

      for (const ev of mechs) {
        const px = zeroX + (ev.re / reScale) * (w * 0.56);
        const py = y0 + h - 18 - ((Math.abs(ev.im) - fMin) / Math.max(1, fMax - fMin)) * (h - 42);
        const unstable = ev.re > 1e-4;
        ctx.fillStyle = unstable ? '#ef4444' : '#34d399';
        ctx.beginPath();
        ctx.arc(Math.max(x0 + 6, Math.min(x0 + w - 6, px)), py, unstable ? 5.0 : 4.0, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    ctx.strokeStyle = th.axis;
    ctx.lineWidth = 1.4;
    ctx.strokeRect(x0, y0, w, h);

    ctx.fillStyle = th.label;
    ctx.font = '700 12px ui-monospace, Menlo, monospace';
    ctx.textAlign = 'left';
    ctx.fillText('DRIFT EIGENVALUES λ', x0 + 8, y0 + 15);
    ctx.fillStyle = th.text;
    ctx.font = '11px ui-monospace, Menlo, monospace';
    ctx.fillText('Re λ < 0', x0 + 8, y0 + h - 6);
    ctx.textAlign = 'right';
    ctx.fillStyle = '#f87171';
    ctx.fillText('Re λ > 0', x0 + w - 6, y0 + h - 6);
    ctx.restore();
  }
}

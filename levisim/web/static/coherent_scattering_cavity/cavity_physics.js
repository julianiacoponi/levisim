/**
 * Exact ES module port of levisim's multi-particle coherent-scattering cavity QLT:
 *   - levisim/physics/cavity.py (Cavity)
 *   - levisim/multiphysics/dimension.py (Dimension)
 *   - levisim/multiphysics/multitrap/base.py (MultiTrapBase)
 *   - levisim/multiphysics/multitrap/linear_amplifier_mixin.py (LinearAmplifierMixin)
 *
 * Implements the full (2N+2) x (2N+2) Linear Amplifier Model (LAM) solution
 * T(ω) = (-iωI - A)^{-1}, exact homodyne S_hh(ω), heterodyne S_het(ω),
 * hybridised position PSDs S_jj(ω), cross-spectrum S_x1y1(ω), and exact complex
 * drift-matrix eigenvalues λ via Hessenberg QR iteration for live stability detection.
 *
 * Cross-verified against Python levisim.multiphysics.multitrap.MultiTrap in
 * tests/test_web_cavity.py.
 */

export const PI = Math.PI;
export const HBAR = 1.0545718176461565e-34;
export const C = 299792458.0;
export const KB = 1.380649e-23;
export const EPS0 = 8.8541878188e-12;
export const M_AIR = 4.8e-26;

export const HZ = 2.0 * PI;
export const KHZ = 1e3 * HZ;
export const DEG = PI / 180.0;
export const CM = 1e-2;
export const UM = 1e-6;
export const NM = 1e-9;
export const MW = 1e-3;
export const MBAR = 100.0;

// Built-in physical tweezers from MultiTrapBase.setup_levitated_physics
const BUILTIN_WAISTS_UM = [
  [0.9276, 1.0679],
  [0.950, 1.050],
  [0.975, 1.025],
  [0.965, 1.015],
  [0.940, 1.020],
  [0.930, 1.010],
  [0.915, 1.055],
  [0.925, 1.045],
];
const BUILTIN_PHASES_DEG = [0, 30, 20, 30, 40, 50, 60, 70];

export const DEFAULT_CAVITY_PARAMS = {
  mode: 'physical', // 'physical' | 'comb'
  num_traps: 2,
  delta_kHz: -450.0,
  finesse: 40000.0,
  L_cm: 0.96,
  waist_um: 61.0,
  theta_hom_deg: 45.0,
  theta_tw_deg: 45.0,
  phi_sw_pi: 0.5, // standing-wave phase Φ / π (0 = antinode, 0.5 = node)
  phi2_deg: 30.0, // relative phase of 2nd tweezer (54.4°..125.6° is unstable for 2×2!)
  power_mW: 485.0,
  wavelength_nm: 1064.0,
  radius_nm: 60.1,
  pressure_mbar: 1e-6,
  temperature_K: 300.0,
  photon_recoil: true,
  // PRR 8 mode-comb parameters (used when mode === 'comb')
  comb_mid_kHz: 300.0,
  comb_span_kHz: 100.0,
  comb_g_kHz: -40.0,
  f_min_kHz: 122.0,
  f_max_kHz: 166.0,
  num_points: 300,
};

// ---------------------------------------------------------------------------
// Complex linear algebra helpers (dense flat arrays: re[i*n+j], im[i*n+j])
// ---------------------------------------------------------------------------

/**
 * Inverts a complex n x n matrix (Ar + i Ai) using partial-pivoting Gaussian elimination.
 * Writes result into (Tr, Ti) flat Float64Arrays of length n*n.
 */
function invertComplexMatrix(n, Ar, Ai, Tr, Ti) {
  const cols = 2 * n;
  const Mr = new Float64Array(n * cols);
  const Mi = new Float64Array(n * cols);
  for (let i = 0; i < n; i++) {
    const rOff = i * cols;
    const aOff = i * n;
    for (let j = 0; j < n; j++) {
      Mr[rOff + j] = Ar[aOff + j];
      Mi[rOff + j] = Ai[aOff + j];
    }
    Mr[rOff + n + i] = 1.0;
  }

  for (let col = 0; col < n; col++) {
    let piv = col;
    let best = 0.0;
    for (let r = col; r < n; r++) {
      const idx = r * cols + col;
      const mag = Mr[idx] * Mr[idx] + Mi[idx] * Mi[idx];
      if (mag > best) {
        best = mag;
        piv = r;
      }
    }
    if (best < 1e-60) continue;

    if (piv !== col) {
      const cOff = col * cols;
      const pOff = piv * cols;
      for (let j = col; j < cols; j++) {
        let tmp = Mr[cOff + j]; Mr[cOff + j] = Mr[pOff + j]; Mr[pOff + j] = tmp;
        tmp = Mi[cOff + j]; Mi[cOff + j] = Mi[pOff + j]; Mi[pOff + j] = tmp;
      }
    }

    const cOff = col * cols;
    const pr = Mr[cOff + col];
    const pi = Mi[cOff + col];
    const invDen = 1.0 / (pr * pr + pi * pi);
    for (let j = col; j < cols; j++) {
      const a = Mr[cOff + j], b = Mi[cOff + j];
      Mr[cOff + j] = (a * pr + b * pi) * invDen;
      Mi[cOff + j] = (b * pr - a * pi) * invDen;
    }

    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const rOff = r * cols;
      const fr = Mr[rOff + col], fi = Mi[rOff + col];
      if (fr === 0.0 && fi === 0.0) continue;
      for (let j = col; j < cols; j++) {
        const cr = Mr[cOff + j], ci = Mi[cOff + j];
        Mr[rOff + j] -= fr * cr - fi * ci;
        Mi[rOff + j] -= fr * ci + fi * cr;
      }
    }
  }

  for (let i = 0; i < n; i++) {
    const rOff = i * cols;
    const tOff = i * n;
    for (let j = 0; j < n; j++) {
      Tr[tOff + j] = Mr[rOff + n + j];
      Ti[tOff + j] = Mi[rOff + n + j];
    }
  }
}

/**
 * Computes all complex eigenvalues of a general n x n complex matrix (Ar, Ai)
 * via similarity reduction to upper Hessenberg form (Householder reflections)
 * followed by Wilkinson-shifted complex QR iteration with deflation.
 *
 * Accurate to ~1e-11 relative error against numpy.linalg.eigvals for n <= 24.
 */
export function complexEigenvalues(n, ArIn, AiIn) {
  const Hr = new Float64Array(ArIn);
  const Hi = new Float64Array(AiIn);

  // 1. Scale matrix by max magnitude to keep QR well-conditioned across 10^6 rad/s frequencies
  let scale = 0.0;
  for (let i = 0; i < n * n; i++) {
    const m = Math.hypot(Hr[i], Hi[i]);
    if (m > scale) scale = m;
  }
  if (scale === 0.0) {
    return Array.from({ length: n }, () => ({ re: 0, im: 0 }));
  }
  const invScale = 1.0 / scale;
  for (let i = 0; i < n * n; i++) {
    Hr[i] *= invScale;
    Hi[i] *= invScale;
  }

  // 2. Reduction to upper Hessenberg form via complex Householder reflections
  const vr = new Float64Array(n);
  const vi = new Float64Array(n);
  for (let k = 0; k < n - 2; k++) {
    let norm2 = 0.0;
    for (let i = k + 1; i < n; i++) {
      const idx = i * n + k;
      norm2 += Hr[idx] * Hr[idx] + Hi[idx] * Hi[idx];
    }
    const norm = Math.sqrt(norm2);
    if (norm < 1e-15) continue;

    const x0r = Hr[(k + 1) * n + k];
    const x0i = Hi[(k + 1) * n + k];
    const x0abs = Math.hypot(x0r, x0i);
    // alpha = -exp(i arg(x0)) * norm
    const sr = x0abs > 0 ? x0r / x0abs : 1.0;
    const si = x0abs > 0 ? x0i / x0abs : 0.0;

    for (let i = k + 1; i < n; i++) {
      vr[i] = Hr[i * n + k];
      vi[i] = Hi[i * n + k];
    }
    vr[k + 1] += sr * norm;
    vi[k + 1] += si * norm;

    let vNorm2 = 0.0;
    for (let i = k + 1; i < n; i++) vNorm2 += vr[i] * vr[i] + vi[i] * vi[i];
    if (vNorm2 < 1e-30) continue;
    const beta = 2.0 / vNorm2;

    // Left multiply: H = (I - beta v v^H) H
    for (let j = k; j < n; j++) {
      let dr = 0.0, di = 0.0;
      for (let i = k + 1; i < n; i++) {
        // conj(v[i]) * H[i,j]
        const hr = Hr[i * n + j], hi = Hi[i * n + j];
        dr += vr[i] * hr + vi[i] * hi;
        di += vr[i] * hi - vi[i] * hr;
      }
      dr *= beta; di *= beta;
      for (let i = k + 1; i < n; i++) {
        Hr[i * n + j] -= vr[i] * dr - vi[i] * di;
        Hi[i * n + j] -= vr[i] * di + vi[i] * dr;
      }
    }

    // Right multiply: H = H (I - beta v v^H)
    for (let i = 0; i < n; i++) {
      let dr = 0.0, di = 0.0;
      for (let j = k + 1; j < n; j++) {
        // H[i,j] * v[j]
        const hr = Hr[i * n + j], hi = Hi[i * n + j];
        dr += hr * vr[j] - hi * vi[j];
        di += hr * vi[j] + hi * vr[j];
      }
      dr *= beta; di *= beta;
      for (let j = k + 1; j < n; j++) {
        // - d * conj(v[j])
        Hr[i * n + j] -= dr * vr[j] + di * vi[j];
        Hi[i * n + j] -= di * vr[j] - dr * vi[j];
      }
    }
    for (let i = k + 2; i < n; i++) {
      Hr[i * n + k] = 0.0;
      Hi[i * n + k] = 0.0;
    }
  }

  // 3. Shifted complex QR iteration with Givens rotations on upper Hessenberg H
  const eigs = [];
  const cs = new Float64Array(n);
  const snr = new Float64Array(n);
  const sni = new Float64Array(n);

  let m = n;
  let totalIter = 0;
  let subIter = 0;
  const maxTotal = 160 * n;

  while (m > 0 && totalIter < maxTotal) {
    totalIter++;
    if (m === 1) {
      eigs.push({ re: Hr[0] * scale, im: Hi[0] * scale });
      break;
    }

    // Check for deflation along subdiagonal
    let l = m - 1;
    while (l > 0) {
      const sub = Math.hypot(Hr[l * n + (l - 1)], Hi[l * n + (l - 1)]);
      const diagSum =
        Math.hypot(Hr[(l - 1) * n + (l - 1)], Hi[(l - 1) * n + (l - 1)]) +
        Math.hypot(Hr[l * n + l], Hi[l * n + l]);
      if (sub <= 1e-13 * Math.max(1e-12, diagSum)) {
        Hr[l * n + (l - 1)] = 0.0;
        Hi[l * n + (l - 1)] = 0.0;
        break;
      }
      l--;
    }

    if (l === m - 1) {
      eigs.push({ re: Hr[(m - 1) * n + (m - 1)] * scale, im: Hi[(m - 1) * n + (m - 1)] * scale });
      m--;
      subIter = 0;
      continue;
    }

    subIter++;
    // Wilkinson shift from trailing 2x2 block
    let shiftR, shiftI;
    if (subIter % 15 === 0) {
      // Exceptional shift to break any symmetry cycling
      shiftR = Hr[(m - 1) * n + (m - 1)] + 0.37;
      shiftI = Hi[(m - 1) * n + (m - 1)] + 0.61;
    } else {
      const aR = Hr[(m - 2) * n + (m - 2)], aI = Hi[(m - 2) * n + (m - 2)];
      const bR = Hr[(m - 2) * n + (m - 1)], bI = Hi[(m - 2) * n + (m - 1)];
      const cR = Hr[(m - 1) * n + (m - 2)], cI = Hi[(m - 1) * n + (m - 2)];
      const dR = Hr[(m - 1) * n + (m - 1)], dI = Hi[(m - 1) * n + (m - 1)];
      const trR = 0.5 * (aR + dR), trI = 0.5 * (aI + dI);
      const diffR = 0.5 * (aR - dR), diffI = 0.5 * (aI - dI);
      // disc = diff^2 + b*c
      const discR = (diffR * diffR - diffI * diffI) + (bR * cR - bI * cI);
      const discI = 2.0 * diffR * diffI + (bR * cI + bI * cR);
      const discMag = Math.hypot(discR, discI);
      const sqR = Math.sqrt(0.5 * (discMag + discR));
      const sqI = (discI >= 0 ? 1 : -1) * Math.sqrt(Math.max(0, 0.5 * (discMag - discR)));
      const e1R = trR + sqR, e1I = trI + sqI;
      const e2R = trR - sqR, e2I = trI - sqI;
      if (Math.hypot(e1R - dR, e1I - dI) <= Math.hypot(e2R - dR, e2I - dI)) {
        shiftR = e1R; shiftI = e1I;
      } else {
        shiftR = e2R; shiftI = e2I;
      }
    }

    // Shift H[l..m-1] by -shift * I
    for (let i = l; i < m; i++) {
      Hr[i * n + i] -= shiftR;
      Hi[i * n + i] -= shiftI;
    }

    // QR step via Givens rotations on rows i, i+1 (i = l..m-2)
    for (let i = l; i < m - 1; i++) {
      const ar = Hr[i * n + i], ai = Hi[i * n + i];
      const br = Hr[(i + 1) * n + i], bi = Hi[(i + 1) * n + i];
      const rMag = Math.hypot(ar, ai, br, bi);
      if (rMag < 1e-30) {
        cs[i] = 1.0; snr[i] = 0.0; sni[i] = 0.0;
        continue;
      }
      const aMag = Math.hypot(ar, ai);
      let cVal, srVal, siVal;
      if (aMag < 1e-30) {
        cVal = 0.0;
        const invB = 1.0 / Math.hypot(br, bi);
        srVal = br * invB;
        siVal = -bi * invB;
      } else {
        cVal = aMag / rMag;
        // s = (a/|a|) * conj(b) / rMag -> G = [[c, s], [-conj(s), c]] zeros b
        const ur = ar / aMag, ui = ai / aMag;
        const kr = (ur * br + ui * bi) / rMag;
        const ki = (ui * br - ur * bi) / rMag;
        srVal = kr;
        siVal = ki;
      }
      cs[i] = cVal;
      snr[i] = srVal;
      sni[i] = siVal;

      // Apply G on left to rows i, i+1 for columns j = i..m-1
      for (let j = i; j < m; j++) {
        const x1r = Hr[i * n + j], x1i = Hi[i * n + j];
        const x2r = Hr[(i + 1) * n + j], x2i = Hi[(i + 1) * n + j];
        // row i = c*x1 + s*x2
        Hr[i * n + j] = cVal * x1r + (srVal * x2r - siVal * x2i);
        Hi[i * n + j] = cVal * x1i + (srVal * x2i + siVal * x2r);
        // row i+1 = -conj(s)*x1 + c*x2
        Hr[(i + 1) * n + j] = -(srVal * x1r + siVal * x1i) + cVal * x2r;
        Hi[(i + 1) * n + j] = -(srVal * x1i - siVal * x1r) + cVal * x2i;
      }
      Hr[(i + 1) * n + i] = 0.0;
      Hi[(i + 1) * n + i] = 0.0;
    }

    // Apply G^H on right to columns i, i+1 for rows r = l..min(i+1, m-1)
    for (let i = l; i < m - 1; i++) {
      const cVal = cs[i], srVal = snr[i], siVal = sni[i];
      const rMax = Math.min(i + 1, m - 1);
      for (let r = l; r <= rMax; r++) {
        const y1r = Hr[r * n + i], y1i = Hi[r * n + i];
        const y2r = Hr[r * n + (i + 1)], y2i = Hi[r * n + (i + 1)];
        // col i = c*y1 + conj(s)*y2
        Hr[r * n + i] = cVal * y1r + (srVal * y2r + siVal * y2i);
        Hi[r * n + i] = cVal * y1i + (srVal * y2i - siVal * y2r);
        // col i+1 = -s*y1 + c*y2
        Hr[r * n + (i + 1)] = -(srVal * y1r - siVal * y1i) + cVal * y2r;
        Hi[r * n + (i + 1)] = -(srVal * y1i + siVal * y1r) + cVal * y2i;
      }
    }

    // Restore shift
    for (let i = l; i < m; i++) {
      Hr[i * n + i] += shiftR;
      Hi[i * n + i] += shiftI;
    }
  }

  // Fallback if any eigenvalue didn't deflate
  while (eigs.length < n) {
    const idx = m - 1;
    eigs.push({ re: Hr[idx * n + idx] * scale, im: Hi[idx * n + idx] * scale });
    m = Math.max(1, m - 1);
  }

  // Sort by decreasing real part (largest growth rate first, matching MultiTrap.drift_eigenvalues)
  eigs.sort((a, b) => b.re - a.re);
  return eigs;
}

/**
 * Builds the complete MultiTrap drift matrix A, noise weights, dimensions, and cavity state.
 */
export function buildCavitySystem(rawParams = {}) {
  const p = { ...DEFAULT_CAVITY_PARAMS, ...rawParams };
  const numTraps = Math.max(1, Math.min(8, Math.round(p.num_traps)));
  const axesPerTrap = 2; // x and y transverse motions
  const numDims = numTraps * axesPerTrap;
  const numOps = 2 * numDims + 2;

  const Delta = p.delta_kHz * KHZ;
  const L = p.L_cm * CM;
  const Finesse = p.finesse;
  const wc = p.waist_um * UM;
  const kappa = (PI * C) / (Finesse * L);
  const Vc = (L * PI * wc * wc) / 4.0;
  const thetaHom = p.theta_hom_deg * DEG;

  const Pgas = p.pressure_mbar * MBAR;
  const Tgas = p.temperature_K;
  const vth = 500.0; // DEFAULT_gas_kwargs vth=500 m/s

  const dims = [];
  let A0 = 0.0, P0 = 0.0, nCav = 0.0;
  const lambda = p.wavelength_nm * NM;
  const kTw = (2.0 * PI) / lambda;
  const omegaTw = C * kTw;
  const omegaCav = omegaTw - Delta;
  const E0Cav = Math.sqrt((HBAR * omegaCav) / (2.0 * EPS0 * Vc));

  if (p.mode === 'comb') {
    // Exact PRR 8 equal-spacing mode comb (matching ΩModel('equal spacing') + shelf defaults)
    const rho = 2200.0;
    const r = 75.0 * NM;
    const m = rho * (4.0 / 3.0) * PI * r * r * r;
    const Gamma = (16.0 * Pgas) / (PI * vth * rho * r);
    const omegaMid = p.comb_mid_kHz * KHZ;
    const omegaSpan = p.comb_span_kHz * KHZ;
    const gVal = p.comb_g_kHz * KHZ;

    for (let i = 0; i < numDims; i++) {
      const trapIdx = Math.floor(i / 2) + 1;
      const axis = i % 2 === 0 ? 'x' : 'y';
      const frac = numDims > 1 ? i / (numDims - 1) - 0.5 : 0.0;
      const Omega = omegaMid + frac * omegaSpan;
      const phi = trapIdx === 2 ? p.phi2_deg * DEG : 0.0;
      const Gr = gVal * Math.cos(phi);
      const Gi = gVal * Math.sin(phi);
      const nExact = 1.0 / (Math.exp((HBAR * Omega) / (KB * Tgas)) - 1.0);
      const zpf = Math.sqrt(HBAR / (2.0 * m * Omega));
      dims.push({
        label: `${axis}${trapIdx}`,
        axis,
        trapIndex: trapIdx,
        Omega,
        Gamma,
        g: gVal,
        Gr,
        Gi,
        gxy: 0.0,
        nBath: nExact,
        zpf,
        phi,
      });
    }
  } else {
    // Physical MultiTrap coherent-scattering system (matching MultiTrapBase.setup_levitated_physics)
    const Power = p.power_mW * MW;
    const thetaTw = p.theta_tw_deg * DEG;
    const Phi = p.phi_sw_pi * PI;
    const rho = 1850.0;
    const epsr = 1.98;

    const traps = [];
    let wdTotal = 0.0;
    for (let t = 0; t < numTraps; t++) {
      const [wxUm, wyUm] = BUILTIN_WAISTS_UM[t % BUILTIN_WAISTS_UM.length];
      const wx0 = wxUm * UM;
      const wy0 = wyUm * UM;
      const w0 = Math.sqrt(wx0 * wy0);
      const E0tw = Math.sqrt((4.0 * Power) / (PI * w0 * w0 * EPS0 * C));
      const r = (p.radius_nm + 0.1 * t) * NM;
      const Vb = (4.0 / 3.0) * PI * r * r * r;
      const m = rho * Vb;
      const alpha0 = 3.0 * Vb * EPS0 * ((epsr - 1.0) / (epsr + 2.0));
      const rad = (kTw * kTw * kTw / (6.0 * PI * EPS0)) * alpha0;
      const alphaRe = alpha0 / (1.0 + rad * rad);
      const Ox = Math.sqrt((alphaRe * E0tw * E0tw) / (wx0 * wx0 * m));
      const Oy = Math.sqrt((alphaRe * E0tw * E0tw) / (wy0 * wy0 * m));
      const Gth = (1.0 + PI / 8.0) * (1.0 / (rho * r)) * ((M_AIR * vth * Pgas) / (KB * Tgas));

      const Ed = 0.5 * alpha0 * E0tw * E0Cav;
      const wd = Ed / HBAR;
      wdTotal += wd * Math.sin(thetaTw) * Math.cos(Phi);

      const phiDeg = t === 0 ? 0.0 : (t === 1 ? p.phi2_deg : BUILTIN_PHASES_DEG[t % BUILTIN_PHASES_DEG.length]);
      traps.push({
        t: t + 1,
        m, r, alphaRe, E0tw, Ox, Oy, Gth, wd,
        thetaTw, Phi,
        phi: phiDeg * DEG,
      });
    }

    const alphaPref = wdTotal / (Delta * Delta + 0.25 * kappa * kappa);
    const alphaR = alphaPref * (-Delta);
    const alphaI = alphaPref * (0.5 * kappa);
    A0 = 2.0 * alphaR;
    P0 = 2.0 * alphaI;
    nCav = alphaR * alphaR + alphaI * alphaI;

    const sinTh = Math.sin(thetaTw);
    const cosTh = Math.cos(thetaTw);
    const sinPhi = Math.sin(Phi);
    const cosPhi = Math.cos(Phi);

    for (const tr of traps) {
      const xzpf = Math.sqrt(HBAR / (2.0 * tr.m * tr.Ox));
      const yzpf = Math.sqrt(HBAR / (2.0 * tr.m * tr.Oy));
      const wdSin = tr.wd * sinTh;
      const gx = -wdSin * kTw * xzpf * sinPhi * sinTh;
      const gy = -wdSin * kTw * yzpf * sinPhi * cosTh;
      const gxy = -A0 * wdSin * (kTw * xzpf) * (kTw * yzpf) * sinTh * cosTh * cosPhi;

      const recPref = (PI / (15.0 * HBAR * EPS0)) * ((tr.alphaRe * tr.E0tw) / (2.0 * PI)) ** 2 * (kTw ** 5);
      const GrecX = recPref * 2.0 * HBAR / (2.0 * tr.m * tr.Ox);
      const GrecY = recPref * 1.0 * HBAR / (2.0 * tr.m * tr.Oy);

      const nx0 = 1.0 / (Math.exp((HBAR * tr.Ox) / (KB * Tgas)) - 1.0);
      const ny0 = 1.0 / (Math.exp((HBAR * tr.Oy) / (KB * Tgas)) - 1.0);
      const nx = p.photon_recoil && tr.Gth > 0 ? nx0 + GrecX / tr.Gth : nx0;
      const ny = p.photon_recoil && tr.Gth > 0 ? ny0 + GrecY / tr.Gth : ny0;

      const cosP = Math.cos(tr.phi), sinP = Math.sin(tr.phi);
      dims.push({
        label: `x${tr.t}`,
        axis: 'x',
        trapIndex: tr.t,
        Omega: tr.Ox,
        Gamma: tr.Gth,
        g: gx,
        Gr: gx * cosP,
        Gi: gx * sinP,
        gxy,
        nBath: nx,
        zpf: xzpf,
        phi: tr.phi,
      });
      dims.push({
        label: `y${tr.t}`,
        axis: 'y',
        trapIndex: tr.t,
        Omega: tr.Oy,
        Gamma: tr.Gth,
        g: gy,
        Gr: gy * cosP,
        Gi: gy * sinP,
        gxy,
        nBath: ny,
        zpf: yzpf,
        phi: tr.phi,
      });
    }
  }

  // Assemble (2N+2) x (2N+2) complex drift matrix A matching LinearAmplifierMixin
  const Dr = new Float64Array(numOps * numOps);
  const Di = new Float64Array(numOps * numOps);
  const gam = new Float64Array(numOps);
  const qfac = new Float64Array(numOps);
  const cin = new Float64Array(numOps);

  gam[0] = kappa; gam[1] = kappa;
  qfac[0] = 1.0; qfac[1] = 0.0;
  Dr[0 * numOps + 0] = -0.5 * kappa; Di[0 * numOps + 0] = Delta;
  Dr[1 * numOps + 1] = -0.5 * kappa; Di[1 * numOps + 1] = -Delta;

  for (let d = 0; d < numDims; d++) {
    const dim = dims[d];
    const ai = 2 + 2 * d;
    const ci = ai + 1;
    gam[ai] = dim.Gamma; gam[ci] = dim.Gamma;
    qfac[ai] = dim.nBath + 1.0; qfac[ci] = dim.nBath;

    // i*G = (-Gi) + i*(Gr); -i*G* = (-Gi) + i*(-Gr); i*G* = (Gi) + i*(Gr)
    const iGr = -dim.Gi, iGi = dim.Gr;
    const miGsr = -dim.Gi, miGsi = -dim.Gr;
    const iGsr = dim.Gi, iGsi = dim.Gr;

    // Light rows (a, a†) couple to b, b†
    Dr[0 * numOps + ai] = iGr;  Di[0 * numOps + ai] = iGi;
    Dr[0 * numOps + ci] = iGr;  Di[0 * numOps + ci] = iGi;
    Dr[1 * numOps + ai] = miGsr; Di[1 * numOps + ai] = miGsi;
    Dr[1 * numOps + ci] = miGsr; Di[1 * numOps + ci] = miGsi;

    // Annihilation operator b_jn row
    Dr[ai * numOps + 0] = iGsr;  Di[ai * numOps + 0] = iGsi;
    Dr[ai * numOps + 1] = iGr;   Di[ai * numOps + 1] = iGi;
    Dr[ai * numOps + ai] = -0.5 * dim.Gamma; Di[ai * numOps + ai] = -dim.Omega;

    // Creation operator b_jn† row
    Dr[ci * numOps + 0] = -iGsr; Di[ci * numOps + 0] = -iGsi;
    Dr[ci * numOps + 1] = -iGr;  Di[ci * numOps + 1] = -iGi;
    Dr[ci * numOps + ci] = -0.5 * dim.Gamma; Di[ci * numOps + ci] = dim.Omega;

    // Intra-particle mech-mech coupling gxy (2D x-y pairs)
    if (dim.gxy !== 0.0) {
      if (dim.axis === 'x') {
        Di[ai * numOps + (ai + 2)] = dim.gxy;
        Di[ai * numOps + (ai + 3)] = dim.gxy;
        Di[ci * numOps + (ai + 2)] = -dim.gxy;
        Di[ci * numOps + (ai + 3)] = -dim.gxy;
      } else {
        Di[ai * numOps + (ai - 2)] = dim.gxy;
        Di[ai * numOps + (ai - 1)] = dim.gxy;
        Di[ci * numOps + (ai - 2)] = -dim.gxy;
        Di[ci * numOps + (ai - 1)] = -dim.gxy;
      }
    }
  }

  for (let i = 0; i < numOps; i++) cin[i] = Math.sqrt(Math.max(0.0, gam[i]));

  // Exact drift-matrix eigenvalues and linear stability metrics
  const eigs = complexEigenvalues(numOps, Dr, Di);
  let maxAbsA = 0.0;
  for (let i = 0; i < numOps * numOps; i++) {
    const m = Math.hypot(Dr[i], Di[i]);
    if (m > maxAbsA) maxAbsA = m;
  }
  const stabTol = 1e-8 * maxAbsA;
  const maxGrowthRate = eigs[0].re;
  const isStable = maxGrowthRate <= stabTol;

  // Estimate PRR 5 normal-mode rotation angle Θ_mode for particle 1 (x1-y1 coupling)
  // Effective dynamical coupling between x1 and y1 via cavity: g_dyn ~ g_x g_y * 2Δ / (Δ^2 + (κ/2)^2)
  // Static coherent-scattering coupling: g_xy
  // Net x-y coupling cancels when g_dyn + g_xy ≈ 0 (exact cancellation at Φc ≈ 0.248π!)
  let modeAngleDeg = 0.0;
  let dynAngleDeg = 0.0;
  let csAngleDeg = 0.0;
  if (dims.length >= 2) {
    const dx = dims[0], dy = dims[1];
    const dOmega = dx.Omega - dy.Omega;
    const gDyn = (2.0 * dx.g * dy.g * Delta) / (Delta * Delta + 0.25 * kappa * kappa);
    const gCs = dx.gxy;
    if (Math.abs(dOmega) > 1.0) {
      dynAngleDeg = 0.5 * Math.atan2(2.0 * gDyn, dOmega) / DEG;
      csAngleDeg = 0.5 * Math.atan2(2.0 * gCs, dOmega) / DEG;
      modeAngleDeg = 0.5 * Math.atan2(2.0 * (gDyn + gCs), dOmega) / DEG;
    }
  }

  return {
    params: p,
    numTraps,
    numDims,
    numOps,
    Delta,
    kappa,
    thetaHom,
    A0,
    P0,
    nCav,
    dims,
    Dr,
    Di,
    gam,
    qfac,
    cin,
    eigs,
    stabTol,
    maxGrowthRate,
    isStable,
    modeAngleDeg,
    dynAngleDeg,
    csAngleDeg,
  };
}

/**
 * Computes live coherent-scattering cavity optical (S_hh, S_het) and motional
 * (S_x1x1, S_y1y1, ..., Re S_x1y1) spectra plus drift eigenvalues via exact LAM.
 */
export function computeCavitySpectraJS(rawParams = {}) {
  const sys = buildCavitySystem(rawParams);
  const p = sys.params;
  const N = Math.max(32, Math.round(p.num_points));
  const fMin = p.f_min_kHz;
  const fMax = p.f_max_kHz;

  const f_kHz = new Array(N);
  const S_hh = new Array(N);
  const S_het = new Array(N);
  const S_x1y1_re = new Array(N);
  const S_pos = {};
  for (const dim of sys.dims) S_pos[dim.label] = new Array(N);

  const n = sys.numOps;
  const Ar = new Float64Array(n * n);
  const Ai = new Float64Array(n * n);
  const Tr = new Float64Array(n * n);
  const Ti = new Float64Array(n * n);

  const cosH = Math.cos(sys.thetaHom);
  const sinH = Math.sin(sys.thetaHom);
  const sk = Math.sqrt(sys.kappa);

  for (let idx = 0; idx < N; idx++) {
    const fk = fMin + ((fMax - fMin) * idx) / (N - 1);
    const w = fk * KHZ;
    f_kHz[idx] = fk;

    // (-iωI - A)
    for (let r = 0; r < n; r++) {
      const off = r * n;
      for (let c = 0; c < n; c++) {
        Ar[off + c] = -sys.Dr[off + c];
        Ai[off + c] = -sys.Di[off + c] - (r === c ? w : 0.0);
      }
    }
    invertComplexMatrix(n, Ar, Ai, Tr, Ti);

    // Optical output spectra S_hh & S_het
    let shh = 0.0;
    let shet = 0.0;
    for (let j = 0; j < n; j++) {
      const c = sys.cin[j];
      const q = sys.qfac[j];
      const ar = Tr[0 * n + j] * c;
      const ai = Ti[0 * n + j] * c;
      const adr = Tr[1 * n + j] * c;
      const adi = Ti[1 * n + j] * c;

      const aor = -sk * ar + (j === 0 ? 1.0 : 0.0);
      const aoi = -sk * ai;
      const aodr = -sk * adr + (j === 1 ? 1.0 : 0.0);
      const aodi = -sk * adi;

      shet += q * (aor * aor + aoi * aoi);

      // h = (a_out + a_out†) cos θ + i(a_out - a_out†) sin θ
      const Aor = aor + aodr;
      const Aoi = aoi + aodi;
      const Por = -(aoi - aodi);
      const Poi = aor - aodr;
      const hr = Aor * cosH + Por * sinH;
      const hi = Aoi * cosH + Poi * sinH;
      shh += q * (hr * hr + hi * hi);
    }
    S_hh[idx] = shh;
    S_het[idx] = shet;

    // Position PSDs for every dimension & cross-spectrum Re S_x1y1
    for (let d = 0; d < sys.numDims; d++) {
      const aiRow = (2 + 2 * d) * n;
      const ciRow = (3 + 2 * d) * n;
      let psd = 0.0;
      for (let j = 0; j < n; j++) {
        const c = sys.cin[j];
        const qr = (Tr[aiRow + j] + Tr[ciRow + j]) * c;
        const qi = (Ti[aiRow + j] + Ti[ciRow + j]) * c;
        psd += sys.qfac[j] * (qr * qr + qi * qi);
      }
      S_pos[sys.dims[d].label][idx] = psd;
    }

    if (sys.numDims >= 2) {
      let csdRe = 0.0;
      for (let j = 0; j < n; j++) {
        const c = sys.cin[j];
        const qxr = (Tr[2 * n + j] + Tr[3 * n + j]) * c;
        const qxi = (Ti[2 * n + j] + Ti[3 * n + j]) * c;
        const qyr = (Tr[4 * n + j] + Tr[5 * n + j]) * c;
        const qyi = (Ti[4 * n + j] + Ti[5 * n + j]) * c;
        csdRe += sys.qfac[j] * (qxr * qyr + qxi * qyi);
      }
      S_x1y1_re[idx] = csdRe;
    } else {
      S_x1y1_re[idx] = 0.0;
    }
  }

  // Filter positive-frequency mechanical eigenvalues for stability inset & display
  const mechEigs = [];
  for (const ev of sys.eigs) {
    if (ev.im >= -1e-6 && Math.abs(ev.re) < 0.25 * sys.kappa) {
      mechEigs.push({ re: ev.re, im: ev.im / KHZ });
    }
  }

  const dimSummary = sys.dims.map(d => ({
    label: d.label,
    omega_kHz: d.Omega / KHZ,
    g_kHz: d.g / KHZ,
    gxy_kHz: d.gxy / KHZ,
    phi_deg: d.phi / DEG,
  }));

  return {
    engine: 'js-60fps',
    f_kHz,
    S_hh,
    S_het,
    S_pos,
    S_x1y1_re,
    eigenvalues: sys.eigs.map(e => ({ re: e.re, im_kHz: e.im / KHZ })),
    mech_eigenvalues: mechEigs,
    telemetry: {
      is_stable: sys.isStable,
      max_growth_rate: sys.maxGrowthRate,
      kappa_kHz: sys.kappa / KHZ,
      delta_kHz: sys.Delta / KHZ,
      n_cav: sys.nCav,
      gx1_kHz: sys.dims[0] ? sys.dims[0].g / KHZ : 0.0,
      gy1_kHz: sys.dims[1] ? sys.dims[1].g / KHZ : 0.0,
      gxy1_kHz: sys.dims[0] ? sys.dims[0].gxy / KHZ : 0.0,
      mode_angle_deg: sys.modeAngleDeg,
      dyn_angle_deg: sys.dynAngleDeg,
      cs_angle_deg: sys.csAngleDeg,
      dimensions: dimSummary,
    },
  };
}

/**
 * Exact ES module port of levisim's free-space optical tweezer physics:
 *   - levisim/physics/tweezer.py (Tweezer)
 *   - levisim/physics/bead.py (Bead + SiO2_refractive_index_Malitson)
 *   - levisim/physics/gas.py (Gas)
 *   - levisim/physics/trap.py (Trap, including photon recoil z_eq_exact & Ωj_at_z_eq)
 *   - levisim/analysis/spectral_density_theory.py (SpectralDensityTheory)
 *
 * Cross-verified against Python levisim in tests/test_web_free_space.py.
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
export const UM = 1e-6;
export const NM = 1e-9;
export const MW = 1e-3;
export const MBAR = 100.0;

export const DEFAULT_TWEEZER_PARAMS = {
  power_mW: 485.0,
  wavelength_nm: 1064.0,
  wx0_um: 0.9276,
  wy0_um: 1.0679,
  radius_nm: 60.1,
  density: 1850.0,
  eps_r: 1.98,
  use_malitson: false,
  pressure_mbar: 4e-3,
  temperature_K: 300.0,
  gamma_fbx_kHz: 2.1,
  gamma_fby_kHz: 2.5,
  gamma_fbz_kHz: 0.2,
  beta: Math.SQRT2,
  psi_deg: 45.0,
  detection_floor: 1e-27,
  photon_recoil: false,
  f_min_kHz: 10.0,
  f_max_kHz: 220.0,
  num_points: 360,
};

/**
 * Malitson (1965) Sellmeier refractive index of fused silica (SiO2).
 * Valid for 0.21-3.71 μm.
 */
export function sio2RefractiveIndexMalitson(lambdaM) {
  const lamUm = lambdaM / UM;
  const B = [0.6961663, 0.4079426, 0.8974794];
  const Cs = [0.0684043, 0.1162414, 9.896161];
  const lam2 = lamUm * lamUm;
  let n2 = 1.0;
  for (let i = 0; i < 3; i++) {
    n2 += (B[i] * lam2) / (lam2 - Cs[i] * Cs[i]);
  }
  return Math.sqrt(n2);
}

/**
 * Computes exact Trap derived quantities matching levisim.physics.trap.Trap.
 */
export function buildTrapPhysics(rawParams = {}) {
  const p = { ...DEFAULT_TWEEZER_PARAMS, ...rawParams };

  const Power = p.power_mW * MW;
  const lambda = p.wavelength_nm * NM;
  const wx0 = p.wx0_um * UM;
  const wy0 = p.wy0_um * UM;
  const k = (2.0 * PI) / lambda;
  const w0 = Math.sqrt(wx0 * wy0);
  const zR = (PI * w0 * w0) / lambda;
  const wz0 = Math.SQRT2 * zR;
  const E0sq = (4.0 * Power) / (PI * w0 * w0 * EPS0 * C);
  const E0 = Math.sqrt(E0sq);

  // Bead
  const r = p.radius_nm * NM;
  const rho = p.density;
  const V = (4.0 / 3.0) * PI * r * r * r;
  const m = rho * V;
  const epsr = p.use_malitson ? sio2RefractiveIndexMalitson(lambda) ** 2 : p.eps_r;
  const alpha0 = 3.0 * V * EPS0 * ((epsr - 1.0) / (epsr + 2.0));

  // Radiation-reaction effective polarisability: α_eff = α / (1 - i (k^3 / 6πε0) α)
  const radTerm = (k * k * k / (6.0 * PI * EPS0)) * alpha0;
  const radDen = 1.0 + radTerm * radTerm;
  const alphaRe = alpha0 / radDen;
  const alphaIm = (alpha0 * radTerm) / radDen;

  // Bare Gaussian focal harmonic frequencies
  const kx0 = (alphaRe * E0sq) / (wx0 * wx0);
  const ky0 = (alphaRe * E0sq) / (wy0 * wy0);
  const kz0 = (alphaRe * E0sq) / (wz0 * wz0);
  const Ox0 = Math.sqrt(kx0 / m);
  const Oy0 = Math.sqrt(ky0 / m);
  const Oz0 = Math.sqrt(kz0 / m);

  // Gaussian beam force & scattering helpers (matching Trap.Ftwj & Trap.F_scattering)
  const wProf = z => Math.sqrt(1.0 + (z / zR) ** 2);
  const EampSq = (x, y, z) => {
    const wz = wProf(z);
    const ell = (x / (wx0 * wz)) ** 2 + (y / (wy0 * wz)) ** 2;
    return (E0sq / (wz * wz)) * Math.exp(-2.0 * ell);
  };
  const Utw = (x, y, z) => -0.5 * alphaRe * EampSq(x, y, z);
  const Ftwx = (x, y, z) => 0.5 * ((4.0 * x) / ((wx0 * wProf(z)) ** 2)) * Utw(x, y, z);
  const Ftwy = (x, y, z) => 0.5 * ((4.0 * y) / ((wy0 * wProf(z)) ** 2)) * Utw(x, y, z);
  const Ftwz = (x, y, z) => {
    const wz = wProf(z);
    const ell = (x / (wx0 * wz)) ** 2 + (y / (wy0 * wz)) ** 2;
    return 0.5 * ((4.0 * z) / ((wz0 * wz) ** 2)) * Utw(x, y, z) * (1.0 - 2.0 * ell);
  };
  const Fscat = (x, y, z) => {
    const phaseGrad = k - zR / (zR * zR + z * z);
    return 0.5 * alphaIm * EampSq(x, y, z) * phaseGrad;
  };

  // Linearised and exact on-axis photon-recoil equilibrium z_eq
  const zEqLinear = (alphaIm / alphaRe) * k * (zR * zR - zR / k);
  let zEqExact = 0.0;
  let Ox = Ox0, Oy = Oy0, Oz = Oz0;

  if (p.photon_recoil) {
    // Bisection root finder in [0, zR] matching scipy.optimize.brentq(Ftwz + Fscat)
    const totalFz = z => Ftwz(0, 0, z) + Fscat(0, 0, z);
    let lo = 0.0, hi = zR;
    let flo = totalFz(lo);
    for (let iter = 0; iter < 80; iter++) {
      const mid = 0.5 * (lo + hi);
      const fmid = totalFz(mid);
      if (Math.abs(fmid) < 1e-34 || (hi - lo) < 1e-17) {
        zEqExact = mid;
        break;
      }
      if ((flo > 0 && fmid > 0) || (flo < 0 && fmid < 0)) {
        lo = mid;
        flo = fmid;
      } else {
        hi = mid;
      }
      zEqExact = mid;
    }
    // Central-difference shifted harmonic frequencies at z_eq_exact (Trap.Ωj_at_z_eq)
    const h = 1e-4 * w0;
    const kxShift = -(Ftwx(h, 0, zEqExact) - Ftwx(-h, 0, zEqExact)) / (2.0 * h);
    const kyShift = -(Ftwy(0, h, zEqExact) - Ftwy(0, -h, zEqExact)) / (2.0 * h);
    const kzShift = -(totalFz(zEqExact + h) - totalFz(zEqExact - h)) / (2.0 * h);
    Ox = Math.sqrt(Math.max(1e-30, kxShift / m));
    Oy = Math.sqrt(Math.max(1e-30, kyShift / m));
    Oz = Math.sqrt(Math.max(1e-30, kzShift / m));
  }

  // Gas damping (Epstein drag, Cavalleri et al. 2010)
  const Pgas = p.pressure_mbar * MBAR;
  const Tgas = p.temperature_K;
  const vth = Math.sqrt((8.0 / PI) * KB * Tgas / M_AIR);
  const Gth = (1.0 + PI / 8.0) * (1.0 / (rho * r)) * ((M_AIR * vth * Pgas) / (KB * Tgas));

  const Gfbx = p.gamma_fbx_kHz * KHZ;
  const Gfby = p.gamma_fby_kHz * KHZ;
  const Gfbz = p.gamma_fbz_kHz * KHZ;
  const Gx = Gth + Gfbx;
  const Gy = Gth + Gfby;
  const Gz = Gth + Gfbz;

  // Thermal & directed force spectral densities
  const SthTwo = 2.0 * m * KB * Tgas * Gth; // two-sided [N^2/Hz]
  const fth = Math.sqrt(SthTwo);
  const beta = p.beta;
  const psi = p.psi_deg * DEG;
  const cosPsi = Math.cos(psi);
  const sinPsi = Math.sin(psi);

  // One-sided (matching SpectralDensityTheory(one_sided=True))
  const SthOne = 2.0 * SthTwo;
  const SdirOne = 2.0 * (beta * beta) * SthTwo;
  const Sffx = SthOne + SdirOne * cosPsi * cosPsi;
  const Sffy = SthOne + SdirOne * sinPsi * sinPsi;
  const Sffz = SthOne;
  const Sfpsi = SdirOne * cosPsi * sinPsi;

  // Effective motional temperatures from equipartition (Trap.T_eff)
  const TeffX = Tgas * Gth * (1.0 + beta * beta * cosPsi * cosPsi) / Gx;
  const TeffY = Tgas * Gth * (1.0 + beta * beta * sinPsi * sinPsi) / Gy;
  const TeffZ = Tgas * Gth / Gz;

  // Coherence γ_xy (SpectralDensityTheory.γxy)
  const Xcoh = (beta * beta * cosPsi * sinPsi) ** 2;
  const Bcoh = 1.0 + beta * beta;
  const gammaXY = Math.sqrt(Xcoh / (Xcoh + Bcoh));

  // Photon recoil heating rates (Trap.photon_recoil_heating_rates)
  const recPref = (PI / (15.0 * HBAR * EPS0)) * ((alphaRe * E0) / (2.0 * PI)) ** 2 * (k ** 5);
  const recX = recPref * 2.0 * HBAR / (2.0 * m * Ox0);
  const xZpf = Math.sqrt(HBAR / (2.0 * m * Ox0));
  const sImpSqlX = recX > 0 ? (xZpf * xZpf) / (4.0 * recX) : 0.0;

  return {
    params: p,
    m, r, V, rho, epsr, lambda, k, w0, wx0, wy0, wz0, zR, E0,
    alphaRe, alphaIm,
    Ox, Oy, Oz, Ox0, Oy0, Oz0,
    Gth, Gx, Gy, Gz,
    fth, SthTwo, SthOne, SdirOne, Sffx, Sffy, Sffz, Sfpsi,
    TeffX, TeffY, TeffZ, gammaXY,
    zEqLinear, zEqExact,
    xZpf, recX, sImpSqlX,
  };
}

/**
 * Evaluates full motional PSDs, co-/quad-spectra, and balanced optical detector spectra
 * on the requested frequency grid.
 */
export function computeTweezerSpectraJS(rawParams = {}) {
  const tr = buildTrapPhysics(rawParams);
  const p = tr.params;
  const N = Math.max(32, Math.round(p.num_points));
  const fMin = p.f_min_kHz;
  const fMax = p.f_max_kHz;
  const floor = p.detection_floor;

  const f_kHz = new Array(N);
  const Sxx = new Array(N);
  const Syy = new Array(N);
  const Szz = new Array(N);
  const Sxx_th = new Array(N);
  const Syy_th = new Array(N);
  const Cxy = new Array(N);
  const Qxy = new Array(N);
  const S_opt_x = new Array(N);
  const S_opt_y = new Array(N);
  const S_opt_z = new Array(N);

  const m2 = tr.m * tr.m;
  const Ox2 = tr.Ox * tr.Ox;
  const Oy2 = tr.Oy * tr.Oy;
  const Oz2 = tr.Oz * tr.Oz;

  for (let i = 0; i < N; i++) {
    const fk = fMin + ((fMax - fMin) * i) / (N - 1);
    const w = fk * KHZ;
    const w2 = w * w;

    const rx = Ox2 - w2, ix = tr.Gx * w;
    const ry = Oy2 - w2, iy = tr.Gy * w;
    const rz = Oz2 - w2, iz = tr.Gz * w;

    const dx = m2 * (rx * rx + ix * ix);
    const dy = m2 * (ry * ry + iy * iy);
    const dz = m2 * (rz * rz + iz * iz);

    const sxxPure = tr.Sffx / dx;
    const syyPure = tr.Sffy / dy;
    const szzPure = tr.Sffz / dz;

    f_kHz[i] = fk;
    Sxx[i] = sxxPure + floor;
    Syy[i] = syyPure + floor;
    Szz[i] = szzPure + floor;
    Sxx_th[i] = tr.SthOne / dx;
    Syy_th[i] = tr.SthOne / dy;

    // χxy(ω) = χx*(ω) χy(ω) = [(rx + i ix)(ry - i iy)] / (m^2 (rx^2+ix^2)(ry^2+iy^2))
    const denomXY = m2 * (rx * rx + ix * ix) * (ry * ry + iy * iy);
    const reChiXY = (rx * ry + ix * iy) / denomXY;
    const imChiXY = (ix * ry - rx * iy) / denomXY;
    Cxy[i] = reChiXY * tr.Sfpsi;
    Qxy[i] = -imChiXY * tr.Sfpsi;

    // Optical balanced forward-scattering detection spectra in shot-noise units (SNU)
    S_opt_x[i] = 1.0 + sxxPure / floor;
    S_opt_y[i] = 1.0 + syyPure / floor;
    S_opt_z[i] = 1.0 + szzPure / floor;
  }

  return {
    engine: 'js-60fps',
    f_kHz,
    Sxx, Syy, Szz,
    Sxx_th, Syy_th,
    Cxy, Qxy,
    S_opt_x, S_opt_y, S_opt_z,
    detection_floor: floor,
    telemetry: {
      fx_kHz: tr.Ox / KHZ,
      fy_kHz: tr.Oy / KHZ,
      fz_kHz: tr.Oz / KHZ,
      gamma_th_Hz: tr.Gth / HZ,
      gamma_x_kHz: tr.Gx / KHZ,
      gamma_y_kHz: tr.Gy / KHZ,
      gamma_z_kHz: tr.Gz / KHZ,
      T_eff_x_K: tr.TeffX,
      T_eff_y_K: tr.TeffY,
      T_eff_z_K: tr.TeffZ,
      coherence_xy: tr.gammaXY,
      z_eq_nm: tr.zEqExact / NM,
      z_eq_linear_nm: tr.zEqLinear / NM,
      mass_fg: tr.m * 1e18,
      x_zpf_pm: tr.xZpf * 1e12,
      recoil_x_kHz: tr.recX / KHZ,
      S_imp_sql_x: tr.sImpSqlX,
    },
  };
}

/**
 * Generates realistic Bartlett-averaged noisy spectral points around the exact theory
 * (using exact Bendat-Piersol chi-squared / Wishart complex Gaussian statistics with `nd`
 * segments, matching SpectralDensityEstimate variance formulas!) for instant 60 FPS
 * simulated-trace overlay in static browser mode.
 */
export function generateSyntheticEstimate(spectra, numSegments = 40, seed = 2026) {
  // Deterministic Mulberry32 PRNG + Box-Muller normals
  let s = seed >>> 0;
  const rand = () => {
    s = (s + 0x6D2B79F5) >>> 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const randn = () => {
    const u1 = Math.max(1e-12, rand());
    const u2 = rand();
    return Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * PI * u2);
  };

  const stride = Math.max(1, Math.floor(spectra.f_kHz.length / 95));
  const fOut = [], sxxOut = [], syyOut = [], cxyOut = [];

  for (let i = 0; i < spectra.f_kHz.length; i += stride) {
    const Sxx = spectra.Sxx[i];
    const Syy = spectra.Syy[i];
    const Cxy = spectra.Cxy[i];
    const Qxy = spectra.Qxy[i];
    const sigX = Math.sqrt(Sxx / 2);
    const sigY = Math.sqrt(Syy / 2);
    // Complex coherence ρ = (Cxy - i Qxy) / sqrt(Sxx Syy)
    const den = Math.sqrt(Math.max(1e-80, Sxx * Syy));
    const rhoR = Math.max(-0.999, Math.min(0.999, Cxy / den));
    const rhoI = Math.max(-0.999, Math.min(0.999, -Qxy / den));
    const rho2 = Math.min(0.998, rhoR * rhoR + rhoI * rhoI);
    const orth = Math.sqrt(1.0 - rho2);

    let accXX = 0, accYY = 0, accC = 0;
    for (let seg = 0; seg < numSegments; seg++) {
      const z1r = randn(), z1i = randn();
      const z2r = randn(), z2i = randn();
      const xr = sigX * z1r, xi = sigX * z1i;
      // y = sigY * (rho* z1 + orth z2) so <x* y> = sigX sigY rho
      const yr = sigY * (rhoR * z1r + rhoI * z1i + orth * z2r);
      const yi = sigY * (rhoR * z1i - rhoI * z1r + orth * z2i);
      accXX += xr * xr + xi * xi;
      accYY += yr * yr + yi * yi;
      accC += xr * yr + xi * yi;
    }
    fOut.push(spectra.f_kHz[i]);
    sxxOut.push(accXX / numSegments);
    syyOut.push(accYY / numSegments);
    cxyOut.push(accC / numSegments);
  }
  return { f_kHz: fOut, Sxx: sxxOut, Syy: syyOut, Cxy: cxyOut };
}

/**
 * Photorealistic Three.js Darkroom Quantum Optics Lab scene for the
 * Coherent-Scattering Cavity Levitated Optomechanics Trap
 * (levisim MultiTrap / PRR 5, 2023 & PRR 8, 2026 setups).
 */
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { RoundedBoxGeometry } from 'three/addons/geometries/RoundedBoxGeometry.js';

import {
  buildDarkroomBackdrop,
  buildDarkroomEnvironment,
  createOpticsMaterials,
  wavelengthToLaserColors,
} from '../shared/optics_materials.js';
import {
  createBeamSegment,
  createBeamSplitterCube,
  createCoaxCable,
  createKinematicMirror,
  createObjectiveBarrel,
  createOpticalTable,
  createPhotodetector,
  createPostAssembly,
  createUHVChamber,
  createWaveplateMount,
} from '../shared/optics_bench.js';
import { SpectrumMonitor } from '../shared/monitor_screen.js';
import { DiegeticController } from '../shared/diegetic_ux.js';

export const CAVITY_BOOKMARKS = {
  bench:    { pos: [0.35, 5.9, 13.0],   target: [0.15, 1.75, -0.45] },
  focus:    { pos: [-1.15, 2.55, 5.15], target: [-1.65, 1.88, 0.85] },
  homodyne: { pos: [5.95, 3.65, 5.65],  target: [3.45, 1.80, 0.75] },
  monitor:  { pos: [1.65, 3.62, 2.75],  target: [2.45, 2.98, -3.75] },
};

const CAV_X = -1.65;
const CAV_Y = 1.88;
const CAV_Z = 0.85;
const CAV_HL = 2.85;
const SW_NODES = 18.0; // visual standing-wave half-wavelength lobes between mirrors

export class CoherentScatteringCavityScene {
  constructor(container, tooltipEl, callbacks) {
    this.container = container;
    this.callbacks = callbacks;

    this.renderer = new THREE.WebGLRenderer({
      antialias: true,
      preserveDrawingBuffer: true,
      powerPreference: 'high-performance',
    });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setSize(container.clientWidth, container.clientHeight);
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.08;
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    container.appendChild(this.renderer.domElement);

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x06090f);
    this.scene.fog = new THREE.Fog(0x06090f, 18, 46);
    this.scene.environment = buildDarkroomEnvironment(this.renderer);
    this.scene.environmentIntensity = 0.84;

    this.camera = new THREE.PerspectiveCamera(
      34,
      container.clientWidth / container.clientHeight,
      0.1,
      100
    );
    this.camera.position.set(...CAVITY_BOOKMARKS.bench.pos);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.target.set(...CAVITY_BOOKMARKS.bench.target);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.06;
    this.controls.maxPolarAngle = Math.PI * 0.49;
    this.controls.minDistance = 2.2;
    this.controls.maxDistance = 25;
    this.controls.update();

    this.diegetic = new DiegeticController(this.camera, this.renderer.domElement, this.controls, tooltipEl);
    this.mats = createOpticsMaterials(3.8, 2.4);

    this.composer = new EffectComposer(this.renderer);
    this.composer.addPass(new RenderPass(this.scene, this.camera));
    this.bloomPass = new UnrealBloomPass(
      new THREE.Vector2(container.clientWidth, container.clientHeight),
      0.24, 0.42, 1.05
    );
    this.composer.addPass(this.bloomPass);
    this.composer.addPass(new OutputPass());

    this.beams = [];
    this.particleAssemblies = [];
    this.builtParticleCount = 0;
    this.currentParams = null;
    this.currentTelemetry = null;
    this.showModeVectors = true;
    this.clock = new THREE.Clock();

    this._buildLights();
    buildDarkroomBackdrop(this.scene);
    this._buildBenchEquipment();
    this._rebuildParticleArray(2);
    this._bindDiegeticControls();

    window.addEventListener('resize', () => this.resize());
    this._animate();
  }

  _buildLights() {
    const key = new THREE.DirectionalLight(0xfff8f2, 1.35);
    key.position.set(7, 13, 9);
    key.castShadow = true;
    key.shadow.mapSize.set(2048, 2048);
    key.shadow.camera.near = 2;
    key.shadow.camera.far = 36;
    const d = 11;
    key.shadow.camera.left = -d;
    key.shadow.camera.right = d;
    key.shadow.camera.top = d;
    key.shadow.camera.bottom = -d;
    key.shadow.bias = -0.0004;
    this.scene.add(key);

    const fill = new THREE.DirectionalLight(0xa5c8ff, 0.58);
    fill.position.set(-10, 7, 5);
    this.scene.add(fill);

    const rim = new THREE.DirectionalLight(0xdbeafe, 1.25);
    rim.position.set(-4, 8, -11);
    this.scene.add(rim);

    this.cavityGlow = new THREE.PointLight(0xff1a34, 2.8, 6.2, 2.0);
    this.cavityGlow.position.set(CAV_X, CAV_Y - 0.3, CAV_Z);
    this.scene.add(this.cavityGlow);
  }

  _buildCavityMirror(side) {
    const g = new THREE.Group();
    const xPos = CAV_X + side * CAV_HL;
    g.position.set(xPos, CAV_Y, CAV_Z);
    const dir = -side;

    const glassR = 0.72, glassT = 0.44;
    const glass = new THREE.Mesh(new THREE.CylinderGeometry(glassR, glassR, glassT, 48), this.mats.fusedSilica);
    glass.rotation.z = Math.PI / 2;
    glass.position.x = dir * 0.06;
    g.add(glass);

    const edgeRing = new THREE.Mesh(new THREE.TorusGeometry(glassR, 0.016, 12, 48), this.mats.stainless);
    edgeRing.rotation.y = Math.PI / 2;
    edgeRing.position.x = dir * (0.06 + glassT / 2);
    g.add(edgeRing);

    const hrFace = new THREE.Mesh(new THREE.CircleGeometry(glassR * 0.94, 48), this.mats.hrCoating);
    hrFace.rotation.y = dir > 0 ? Math.PI / 2 : -Math.PI / 2;
    hrFace.position.x = dir * (0.06 + glassT / 2 + 0.004);
    g.add(hrFace);

    const spotMat = new THREE.MeshBasicMaterial({
      color: 0xe11d48,
      transparent: true,
      opacity: 0.92,
      side: THREE.DoubleSide,
    });
    const spot = new THREE.Mesh(new THREE.CircleGeometry(0.14, 24), spotMat);
    spot.rotation.y = hrFace.rotation.y;
    spot.position.x = hrFace.position.x + dir * 0.005;
    g.add(spot);

    const housing = new THREE.Mesh(new THREE.CylinderGeometry(glassR + 0.14, glassR + 0.14, 0.44, 48), this.mats.anodized);
    housing.rotation.z = Math.PI / 2;
    housing.position.x = -dir * 0.34;
    housing.castShadow = true;
    g.add(housing);

    const piezoRing = new THREE.Mesh(new THREE.CylinderGeometry(glassR + 0.075, glassR + 0.075, 0.13, 40), this.mats.copper);
    piezoRing.rotation.z = Math.PI / 2;
    piezoRing.position.x = -dir * 0.10;
    g.add(piezoRing);

    const plate = new THREE.Mesh(new RoundedBoxGeometry(0.16, 1.66, 1.66, 2, 0.03), this.mats.anodizedRing);
    plate.position.x = -dir * 0.68;
    plate.castShadow = true;
    g.add(plate);

    [[0.58, 0.58], [0.58, -0.58], [-0.58, 0.58]].forEach(([my, mz]) => {
      const screw = new THREE.Mesh(new THREE.CylinderGeometry(0.082, 0.082, 0.42, 20), this.mats.brass);
      screw.rotation.z = Math.PI / 2;
      screw.position.set(-dir * 0.92, my, mz);
      g.add(screw);
    });

    g.userData.baseX = xPos;
    g.userData.spotMat = spotMat;
    this.scene.add(g);
    this.scene.add(createPostAssembly(this.mats, xPos - dir * 0.68, CAV_Z, CAV_Y - 0.83));
    return g;
  }

  _buildBenchEquipment() {
    // 1. Matte-black anodized optical breadboard table
    const table = createOpticalTable(this.mats, 17.4, 10.6, 0.56);
    this.scene.add(table);

    // 2. Left & Right Curved Fused-Silica Fabry-Perot Cavity Mirrors
    this.leftMirror = this._buildCavityMirror(-1);
    this.rightMirror = this._buildCavityMirror(+1);

    // 3. Volumetric GLSL Intracavity Standing-Wave Optical Mode (bright antinodes, dark nodes!)
    this.standingWaveMat = new THREE.ShaderMaterial({
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      side: THREE.DoubleSide,
      uniforms: {
        uOuter: { value: new THREE.Color(0xd90b24) },
        uCore:  { value: new THREE.Color(0xff6b7a) },
        uNodes: { value: SW_NODES },
        uAlpha: { value: 0.78 },
        uBoost: { value: 1.24 },
      },
      vertexShader: `
        varying vec2 vUv; varying vec3 vNormal; varying vec3 vViewDir;
        void main() {
          vUv = uv;
          vec4 mvPos = modelViewMatrix * vec4(position, 1.0);
          vNormal = normalize(normalMatrix * normal);
          vViewDir = normalize(-mvPos.xyz);
          gl_Position = projectionMatrix * mvPos;
        }
      `,
      fragmentShader: `
        uniform vec3 uOuter; uniform vec3 uCore; uniform float uNodes; uniform float uAlpha; uniform float uBoost;
        varying vec2 vUv; varying vec3 vNormal; varying vec3 vViewDir;
        void main() {
          float fresnel = pow(abs(dot(vNormal, vViewDir)), 1.45);
          float standing = 0.20 + 0.80 * pow(cos(vUv.y * 3.14159265 * uNodes), 2.0);
          float edgeFade = smoothstep(0.0, 0.04, vUv.y) * smoothstep(1.0, 0.96, vUv.y);
          vec3 col = mix(uOuter, uCore, pow(fresnel, 2.5) * standing * 0.65);
          float a = fresnel * standing * edgeFade * uAlpha;
          gl_FragColor = vec4(col * uBoost, clamp(a, 0.0, 0.92));
        }
      `,
    });

    this.cavitySpan = CAV_HL * 2 - 0.56;
    const swOuter = new THREE.Mesh(
      new THREE.CylinderGeometry(0.18, 0.18, this.cavitySpan, 36, 110, true),
      this.standingWaveMat
    );
    swOuter.rotation.z = Math.PI / 2;
    swOuter.position.set(CAV_X, CAV_Y, CAV_Z);
    this.scene.add(swOuter);

    const swCore = new THREE.Mesh(
      new THREE.CylinderGeometry(0.072, 0.072, this.cavitySpan, 24, 110, true),
      this.standingWaveMat
    );
    swCore.rotation.z = Math.PI / 2;
    swCore.position.set(CAV_X, CAV_Y, CAV_Z);
    this.scene.add(swCore);

    // Vertical tweezer cone shader (shared across all N particles)
    this.tweezerConeMat = new THREE.ShaderMaterial({
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      side: THREE.DoubleSide,
      uniforms: {
        uOuter: { value: new THREE.Color(0xe0112b) },
        uCore:  { value: new THREE.Color(0xff7a88) },
        uBoost: { value: 1.22 },
      },
      vertexShader: `
        varying vec2 vUv; varying vec3 vNormal; varying vec3 vViewDir;
        void main() {
          vUv = uv;
          vec4 mv = modelViewMatrix * vec4(position, 1.0);
          vNormal = normalize(normalMatrix * normal);
          vViewDir = normalize(-mv.xyz);
          gl_Position = projectionMatrix * mv;
        }
      `,
      fragmentShader: `
        uniform vec3 uOuter; uniform vec3 uCore; uniform float uBoost;
        varying vec2 vUv; varying vec3 vNormal; varying vec3 vViewDir;
        void main() {
          float f = pow(abs(dot(vNormal, vViewDir)), 1.3);
          float waist = 0.55 + 0.9 * exp(-pow((vUv.y - 0.5) * 4.0, 2.0));
          vec3 col = mix(uOuter, uCore, clamp(f * waist * 0.45, 0.0, 1.0));
          gl_FragColor = vec4(col * uBoost, clamp(f * waist * 0.56, 0.0, 0.88));
        }
      `,
    });

    // Dynamic container for N levitated particles & objective pairs
    this.arrayGroup = new THREE.Group();
    this.arrayGroup.position.set(CAV_X, CAV_Y, CAV_Z);
    this.scene.add(this.arrayGroup);

    // 4. UHV Cylindrical Stainless-Steel & Sapphire Glass Vacuum Chamber enclosing the cavity
    this.chamber = createUHVChamber(this.mats, new THREE.Vector3(CAV_X, CAV_Y, CAV_Z), {
      radius: 2.15,
      height: 2.88,
      sidePorts: true,
      frontBezel: true,
    });
    this.scene.add(this.chamber);

    // 5. Front-Left Tweezer Polarisation λ/2 Waveplate (diegetic drag -> θ)
    this.wpPol = createWaveplateMount(this.mats, new THREE.Vector3(-4.95, 1.55, 2.85), Math.PI / 6, this.mats.arLensViolet);
    this.scene.add(this.wpPol);
    this.scene.add(createPostAssembly(this.mats, -4.95, 2.85, 1.05));

    // 6. Front-Center Electro-Optic Phase Shifter (EOM) controlling relative tweezer phase φ2
    this.eomGroup = new THREE.Group();
    this.eomGroup.position.set(-1.65, 0.42, 3.65);
    const eomBase = new THREE.Mesh(new RoundedBoxGeometry(1.55, 0.62, 0.92, 3, 0.04), this.mats.ampChassis);
    eomBase.castShadow = true;
    eomBase.receiveShadow = true;
    this.eomGroup.add(eomBase);
    const eomCrystal = new THREE.Mesh(new RoundedBoxGeometry(0.65, 0.34, 0.34, 2, 0.02), this.mats.fusedSilica);
    eomCrystal.position.set(-0.22, 0.46, 0);
    this.eomGroup.add(eomCrystal);
    this.eomDial = new THREE.Mesh(new THREE.CylinderGeometry(0.16, 0.17, 0.15, 24), this.mats.brass);
    this.eomDial.rotation.x = Math.PI / 2;
    this.eomDial.position.set(0.45, 0.0, 0.50);
    this.eomGroup.add(this.eomDial);
    this.eomLedMat = new THREE.MeshBasicMaterial({ color: 0x10b981 });
    const eomLed = new THREE.Mesh(new THREE.SphereGeometry(0.052, 12, 12), this.eomLedMat);
    eomLed.position.set(-0.55, 0.14, 0.47);
    this.eomGroup.add(eomLed);
    this.scene.add(this.eomGroup);

    // 7. PRR 5 3D Mode-Orientation Eigenvector Overlay (follows particle 1 inside the cavity)
    this.modeVecGroup = new THREE.Group();
    const makeAxisArrow = (color, length, opacity = 0.88) => {
      const grp = new THREE.Group();
      const mat = new THREE.MeshBasicMaterial({
        color, transparent: true, opacity, blending: THREE.AdditiveBlending, depthWrite: false,
      });
      const rod = new THREE.Mesh(new THREE.CylinderGeometry(0.015, 0.015, length, 10), mat);
      rod.rotation.z = Math.PI / 2;
      const tip = new THREE.Mesh(new THREE.ConeGeometry(0.042, 0.10, 12), mat);
      tip.rotation.z = -Math.PI / 2;
      tip.position.x = length / 2 + 0.04;
      grp.add(rod, tip);
      return grp;
    };
    // Bare x/y axes in horizontal (x, z) plane perpendicular to vertical tweezer
    this.vecBareX = makeAxisArrow(0x94a3b8, 0.85, 0.45);
    this.vecBareY = makeAxisArrow(0x94a3b8, 0.85, 0.45);
    this.vecBareY.rotation.y = Math.PI / 2;
    this.vecDyn = makeAxisArrow(0x38bdf8, 0.92, 0.92); // cavity backaction Θ_dyn (blue)
    this.vecCs  = makeAxisArrow(0xc084fc, 0.92, 0.92); // static coherent scattering Θ_CS (purple)
    this.vecNet = makeAxisArrow(0x34d399, 1.04, 0.96); // net hybridised normal-mode axis (emerald)
    this.modeVecGroup.add(this.vecBareX, this.vecBareY, this.vecDyn, this.vecCs, this.vecNet);
    this.modeVecGroup.position.set(CAV_X, CAV_Y, CAV_Z);
    this.scene.add(this.modeVecGroup);

    // 8. Downstream Balanced Homodyne Detection Stage (Output beam -> λ/2 -> 50/50 BS -> LO Mirror -> Detectors)
    const BS_X = 3.55;
    const addBeam = (a, b, r, op) => {
      const seg = createBeamSegment(a, b, { radius: r, opacity: op });
      this.beams.push(seg);
      this.scene.add(seg);
    };

    const cavOutStart = new THREE.Vector3(CAV_X + CAV_HL + 0.72, CAV_Y, CAV_Z);
    const bsPos = new THREE.Vector3(BS_X, CAV_Y, CAV_Z);
    addBeam(cavOutStart, bsPos, 0.052, 0.72);

    // Output waveplate
    const wpOutPos = new THREE.Vector3(2.35, CAV_Y, CAV_Z);
    this.scene.add(createWaveplateMount(this.mats, wpOutPos, 0, this.mats.arLensEmerald));
    this.scene.add(createPostAssembly(this.mats, wpOutPos.x, wpOutPos.z, CAV_Y - 0.50));

    // Photorealistic N-BK7 50/50 Glass Beam Splitter Cube
    this.scene.add(createBeamSplitterCube(this.mats, bsPos, Math.PI / 4, 0.86));
    this.scene.add(createPostAssembly(this.mats, bsPos.x, bsPos.z, CAV_Y - 0.52));

    // Local Oscillator (LO) Piezo Steering Mirror (diegetic drag -> θ_hom!)
    const loPos = new THREE.Vector3(BS_X, CAV_Y, CAV_Z + 2.30);
    this.loMirror = createKinematicMirror(this.mats, loPos, -Math.PI * 0.75, 0.40);
    this.loMirror.userData.baseZ = loPos.z;
    this.scene.add(this.loMirror);
    this.scene.add(createPostAssembly(this.mats, loPos.x, loPos.z, CAV_Y - 0.45));
    addBeam(loPos, bsPos, 0.044, 0.65);

    // Balanced Homodyne Photodetectors
    const pd1Pos = new THREE.Vector3(BS_X + 2.25, CAV_Y, CAV_Z);
    const pd2Pos = new THREE.Vector3(BS_X, CAV_Y, CAV_Z - 1.95);
    addBeam(bsPos, pd1Pos, 0.046, 0.70);
    addBeam(bsPos, pd2Pos, 0.046, 0.70);

    this.scene.add(createPhotodetector(this.mats, pd1Pos, -Math.PI / 2));
    this.scene.add(createPostAssembly(this.mats, pd1Pos.x, pd1Pos.z, CAV_Y - 0.36));
    this.scene.add(createPhotodetector(this.mats, pd2Pos, Math.PI));
    this.scene.add(createPostAssembly(this.mats, pd2Pos.x, pd2Pos.z, CAV_Y - 0.36));

    // Braided Coaxial Cables to Monitor & EOM
    this.scene.add(createCoaxCable(this.mats, [
      [pd1Pos.x + 0.48, CAV_Y, pd1Pos.z], [pd1Pos.x + 0.8, 0.28, -0.4], [4.6, 0.06, -2.5], [3.3, 0.42, -3.5],
    ]));
    this.scene.add(createCoaxCable(this.mats, [
      [pd2Pos.x, CAV_Y, pd2Pos.z - 0.48], [3.75, 0.32, -2.2], [2.95, 0.14, -3.5],
    ]));
    this.scene.add(createCoaxCable(this.mats, [
      [-1.65, 0.25, 3.18], [-1.65, 0.08, 2.75], [-1.65, 0.25, 2.35],
    ]));

    // 9. In-Game Benchtop RF Spectrum Analyzer Monitor (elevated cleanly above detection optics)
    this.monitor = new SpectrumMonitor(this.mats, {
      position: new THREE.Vector3(2.45, 2.82, -3.75),
      rotationY: -0.18,
      width: 4.85,
      height: 3.12,
      theme: 'rf',
    });
    this.scene.add(this.monitor.group);
  }

  /**
   * Dynamically rebuilds the N vertical high-NA tweezer objective pairs,
   * Gaussian focus cones, and levitated silica nanospheres along the cavity axis
   * (supports N = 1..8, matching prr5_setup_a.png through prr8_array.png).
   */
  _rebuildParticleArray(numTraps) {
    if (numTraps === this.builtParticleCount) return;
    this.builtParticleCount = numTraps;

    while (this.arrayGroup.children.length > 0) {
      this.arrayGroup.remove(this.arrayGroup.children[0]);
    }
    this.particleAssemblies = [];

    const span = numTraps === 1 ? 0 : Math.min(3.15, 0.82 * (numTraps - 1));
    const scale = numTraps <= 2 ? 0.94 : (numTraps <= 4 ? 0.72 : 0.46);
    const objDist = 0.96 * Math.max(0.76, scale);
    const coneH = 0.54 * Math.max(0.76, scale);

    for (let i = 0; i < numTraps; i++) {
      const baseX = numTraps === 1 ? 0 : -span / 2 + (span * i) / (numTraps - 1);
      const unit = new THREE.Group();
      unit.position.set(baseX, 0, 0);

      // Bottom high-NA trapping objective & top condenser lens barrel
      unit.add(createObjectiveBarrel(this.mats, new THREE.Vector3(0, -objDist, 0), new THREE.Vector3(0, 1, 0), scale));
      unit.add(createObjectiveBarrel(this.mats, new THREE.Vector3(0, +objDist, 0), new THREE.Vector3(0, -1, 0), scale));

      // Vertical translucent crimson Gaussian focus cones
      const coneTop = new THREE.Mesh(
        new THREE.CylinderGeometry(0.22 * scale, 0.032 * scale, coneH, 28, 1, true),
        this.tweezerConeMat
      );
      coneTop.position.y = coneH / 2 + 0.01;
      const coneBot = new THREE.Mesh(
        new THREE.CylinderGeometry(0.032 * scale, 0.22 * scale, coneH, 28, 1, true),
        this.tweezerConeMat
      );
      coneBot.position.y = -coneH / 2 - 0.01;
      unit.add(coneTop, coneBot);

      // Glowing levitated silica nanosphere
      const beadMat = new THREE.MeshStandardMaterial({
        color: 0xffffff,
        emissive: 0xff4d64,
        emissiveIntensity: 3.0,
        roughness: 0.1,
      });
      const bead = new THREE.Mesh(new THREE.SphereGeometry(0.072 * Math.max(0.75, scale), 28, 28), beadMat);
      unit.add(bead);

      // Subtle instability warning halo ring (visible only when Re λ > 0!)
      const warnMat = new THREE.MeshBasicMaterial({
        color: 0xef4444,
        transparent: true,
        opacity: 0.0,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
        side: THREE.DoubleSide,
      });
      const warnRing = new THREE.Mesh(new THREE.TorusGeometry(0.18 * Math.max(0.8, scale), 0.018, 12, 32), warnMat);
      warnRing.rotation.x = Math.PI / 2;
      unit.add(warnRing);

      this.arrayGroup.add(unit);
      this.particleAssemblies.push({
        index: i,
        unit,
        baseX,
        bead,
        beadMat,
        warnRing,
        warnMat,
        phaseOffset: i * 1.37,
      });
    }
  }

  _bindDiegeticControls() {
    const addRing = (parent, radius, rot = [0, 0, 0], offset = [0, 0, 0]) => {
      const ring = new THREE.Mesh(new THREE.TorusGeometry(radius, 0.034, 12, 40), this.mats.hoverHighlight);
      ring.rotation.set(...rot);
      ring.position.set(...offset);
      parent.add(ring);
      return ring;
    };

    // 1. Right Cavity Mirror Piezo Mount -> Cavity Detuning Δ/2π
    this.diegetic.register({
      object: this.rightMirror,
      highlightRing: addRing(this.rightMirror, 0.88, [0, Math.PI / 2, 0], [0.10, 0, 0]),
      title: 'Cavity HR Mirror Piezo Stage',
      hint: 'Drag left/right to sweep cavity detuning Δ/2π (red ↔ blue)',
      getValue: () => this.currentParams
        ? `Δ/2π = ${this.currentParams.delta_kHz >= 0 ? '+' : ''}${this.currentParams.delta_kHz.toFixed(0)} kHz`
        : '',
      onDrag: (dx) => this.callbacks.onDeltaParam('delta_kHz', dx * 3.5),
    });

    // 2. Left Cavity Mirror Mount -> Standing-Wave Phase Φ
    this.diegetic.register({
      object: this.leftMirror,
      highlightRing: addRing(this.leftMirror, 0.88, [0, Math.PI / 2, 0], [-0.10, 0, 0]),
      title: 'Standing-Wave Position Stage (Φ)',
      hint: 'Drag left/right to move trap from antinode (0) → cancellation → node (0.5π)',
      getValue: () => this.currentParams
        ? `Φ = ${this.currentParams.phi_sw_pi.toFixed(3)}π`
        : '',
      onDrag: (dx) => this.callbacks.onDeltaParam('phi_sw_pi', dx * 0.0022),
    });

    // 3. Polarisation λ/2 Waveplate -> Tweezer polarisation angle θ
    this.diegetic.register({
      object: this.wpPol,
      highlightRing: addRing(this.wpPol, 0.56, [0, Math.PI / 2, 0]),
      title: 'λ/2 Tweezer Polarisation Waveplate (θ)',
      hint: 'Drag to rotate tweezer polarisation θ relative to cavity axis Yc',
      getValue: () => this.currentParams ? `θ = ${this.currentParams.theta_tw_deg.toFixed(1)}°` : '',
      onDrag: (dx) => this.callbacks.onDeltaParam('theta_tw_deg', dx * 0.45),
    });

    // 4. Electro-Optic Phase Modulator (EOM) -> 2nd Tweezer Relative Phase φ2
    this.diegetic.register({
      object: this.eomGroup,
      highlightRing: addRing(this.eomGroup, 0.28, [0, 0, 0], [0.45, 0.0, 0.55]),
      title: '2nd Tweezer Phase Modulator (φ₂)',
      hint: 'Drag to sweep relative phase φ₂ (54.4°–125.6° is non-reciprocally unstable!)',
      getValue: () => this.currentParams ? `φ₂ = ${this.currentParams.phi2_deg.toFixed(1)}°` : '',
      onDrag: (dx) => this.callbacks.onDeltaParam('phi2_deg', dx * 0.75),
    });

    // 5. LO Piezo Steering Mirror -> Homodyne Angle θ_hom
    this.diegetic.register({
      object: this.loMirror,
      highlightRing: addRing(this.loMirror, 0.52, [0, 0, 0], [0, 0, 0.14]),
      title: 'Local Oscillator Piezo Mirror (θ_hom)',
      hint: 'Drag to sweep homodyne quadrature angle θ_hom (0°..180°)',
      getValue: () => this.currentParams ? `θ_hom = ${this.currentParams.theta_hom_deg.toFixed(1)}°` : '',
      onDrag: (dx) => this.callbacks.onDeltaParam('theta_hom_deg', dx * 0.65),
    });

    // 6. Vacuum Pressure Gauge -> Gas pressure P_gas
    this.diegetic.register({
      object: this.chamber.userData.gaugeGroup,
      highlightRing: addRing(this.chamber.userData.gaugeGroup, 0.38, [0, 0, 0], [0, 0.14, 0.08]),
      title: 'UHV Chamber Pressure Gauge',
      hint: 'Drag to adjust cavity vacuum pressure (log mbar)',
      getValue: () => this.currentParams ? `P = ${this.currentParams.pressure_mbar.toExponential(1)} mbar` : '',
      onDrag: (dx) => this.callbacks.onDeltaLogPressure(dx * 0.025),
    });

    // 7. In-Game RF Spectrum Analyzer Monitor -> Click to focus
    this.diegetic.register({
      object: this.monitor.group,
      title: 'In-Game RF Spectrum Analyzer',
      hint: 'Click to zoom camera onto monitor screen',
      getValue: () => this.monitor.powerOn ? 'Monitor ON' : 'Monitor STANDBY',
      onClick: () => this.flyToBookmark('monitor'),
    });
  }

  flyToBookmark(key) {
    const bm = CAVITY_BOOKMARKS[key];
    if (bm) this.diegetic.flyTo(bm.pos, bm.target, 680);
  }

  setChamberMode(mode) {
    this.chamber.userData.setMode(mode);
  }

  setShowModeVectors(visible) {
    this.showModeVectors = Boolean(visible);
    this.modeVecGroup.visible = this.showModeVectors;
  }

  syncVisuals(params, telemetry) {
    this.currentParams = params;
    this.currentTelemetry = telemetry;

    this._rebuildParticleArray(params.num_traps);

    // Laser color & intracavity standing-wave brightness driven by |α|^2 and resonance
    const cols = wavelengthToLaserColors(params.wavelength_nm);
    const nCavLog = telemetry.n_cav > 1 ? Math.min(1.0, Math.log10(telemetry.n_cav) / 9.0) : 0.0;
    const swLevel = params.mode === 'comb' ? 0.82 : (0.34 + 0.66 * nCavLog);

    this.standingWaveMat.uniforms.uOuter.value.copy(cols.outer);
    this.standingWaveMat.uniforms.uCore.value.copy(cols.core);
    this.standingWaveMat.uniforms.uAlpha.value = 0.36 + 0.50 * swLevel;
    this.standingWaveMat.uniforms.uBoost.value = 0.85 + 0.55 * swLevel;

    this.tweezerConeMat.uniforms.uOuter.value.copy(cols.outer);
    this.tweezerConeMat.uniforms.uCore.value.copy(cols.core);
    const pwrNorm = Math.max(0.35, Math.min(1.5, params.power_mW / 485.0));
    this.tweezerConeMat.uniforms.uBoost.value = 0.75 + 0.50 * pwrNorm;

    for (const seg of this.beams) {
      seg.userData.outerMat.color.copy(cols.outer);
      seg.userData.coreMat.color.copy(cols.core);
      seg.userData.outerMat.opacity = Math.min(0.92, seg.userData.baseOpacity * (0.45 + 0.55 * swLevel));
    }
    this.cavityGlow.color.setHex(cols.point);
    this.cavityGlow.intensity = 1.5 + 2.0 * swLevel;

    // Physical piezo displacement of right cavity mirror with Δ/2π
    const mirrorShift = ((params.delta_kHz + 450) / 1000) * 0.16;
    this.rightMirror.position.x = this.rightMirror.userData.baseX + mirrorShift;

    // Standing-wave node translation: 1 half-period of visual standing wave = cavitySpan / SW_NODES
    const halfPeriod = this.cavitySpan / SW_NODES;
    // Φ = 0 -> antinode crest, Φ = 0.5π -> dark node waist (shift by 0.5 * halfPeriod)
    const swShiftX = params.mode === 'comb' ? 0 : (params.phi_sw_pi * halfPeriod);
    for (const item of this.particleAssemblies) {
      item.unit.position.x = item.baseX + swShiftX;
      const rScale = Math.max(0.65, Math.min(1.6, params.radius_nm / 60.1));
      item.bead.scale.setScalar(rScale);
    }

    // Polarisation λ/2 waveplate rotor (rotates by θ/2 physically)
    this.wpPol.userData.rotor.rotation.x = (params.theta_tw_deg * Math.PI) / 180;

    // EOM phase dial & stability status LED
    this.eomDial.rotation.y = -(params.phi2_deg / 180) * Math.PI * 1.6;
    this.eomLedMat.color.setHex(telemetry.is_stable ? 0x10b981 : 0xef4444);

    // LO piezo mirror translation with homodyne angle θ_hom
    this.loMirror.position.z = this.loMirror.userData.baseZ + ((params.theta_hom_deg - 45) / 180) * 0.16;

    // Vacuum gauge needle
    const logP = Math.max(-9, Math.min(-2, Math.log10(Math.max(1e-10, params.pressure_mbar))));
    this.chamber.userData.gaugeNeedle.rotation.z = -((logP + 9) / 7) * Math.PI * 1.35 + Math.PI * 0.65;

    // PRR 5 Mode-Orientation Eigenvectors positioned at Particle 1
    if (this.particleAssemblies.length > 0) {
      const p1 = this.particleAssemblies[0];
      this.modeVecGroup.position.set(CAV_X + p1.unit.position.x, CAV_Y, CAV_Z);
      const baseRot = (params.theta_tw_deg * Math.PI) / 180;
      this.vecBareX.rotation.y = baseRot;
      this.vecBareY.rotation.y = baseRot + Math.PI / 2;
      // Exaggerate small mode rotations slightly (x3) so cancellation vs tilt is unmistakably clear in 3D
      this.vecDyn.rotation.y = baseRot + ((telemetry.dyn_angle_deg * 3.0) * Math.PI) / 180;
      this.vecCs.rotation.y  = baseRot + ((telemetry.cs_angle_deg * 3.0) * Math.PI) / 180;
      this.vecNet.rotation.y = baseRot + ((telemetry.mode_angle_deg * 3.0) * Math.PI) / 180;
      this.modeVecGroup.visible = this.showModeVectors && params.mode === 'physical';
    }
  }

  _animate() {
    requestAnimationFrame(() => this._animate());
    const t = this.clock.getElapsedTime();
    this.controls.update();
    this.diegetic.update();

    if (this.currentTelemetry && this.particleAssemblies.length) {
      const stable = this.currentTelemetry.is_stable;
      const dims = this.currentTelemetry.dimensions || [];
      for (const item of this.particleAssemblies) {
        const dxDim = dims[2 * item.index];
        const dyDim = dims[2 * item.index + 1];
        const fx = dxDim ? dxDim.omega_kHz * 0.03 : 4.5;
        const fy = dyDim ? dyDim.omega_kHz * 0.03 : 4.0;
        const amp = stable ? 0.022 : 0.072;
        item.bead.position.set(
          amp * Math.sin(t * fx * Math.PI * 2 + item.phaseOffset),
          0.010 * Math.cos(t * 2.1 + item.phaseOffset),
          amp * Math.cos(t * fy * Math.PI * 2 + item.phaseOffset * 1.4)
        );
        // Pulse crimson warning halo when linearly unstable (Re λ > 0)
        if (!stable) {
          const pulse = 0.45 + 0.45 * Math.sin(t * 7.5 + item.index);
          item.warnMat.opacity = pulse;
          item.warnRing.scale.setScalar(1.0 + 0.18 * pulse);
          item.beadMat.emissive.setHex(0xff2238);
          item.beadMat.emissiveIntensity = 3.2 + 1.4 * pulse;
        } else {
          item.warnMat.opacity = 0.0;
          item.beadMat.emissive.setHex(0xff5a6e);
          item.beadMat.emissiveIntensity = 2.8;
        }
      }
    }

    this.composer.render();
  }

  resize() {
    const w = this.container.clientWidth;
    const h = this.container.clientHeight;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
    this.composer.setSize(w, h);
  }
}

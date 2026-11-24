/**
 * Photorealistic Three.js Darkroom Quantum Optics Lab scene for the
 * Free-Space Optical Tweezer Trap (levisim Trap / PRR 6, 2024 setup).
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
  createLaserHead,
  createObjectiveBarrel,
  createOpticalTable,
  createPhotodetector,
  createPostAssembly,
  createUHVChamber,
  createWaveplateMount,
} from '../shared/optics_bench.js';
import { SpectrumMonitor } from '../shared/monitor_screen.js';
import { DiegeticController } from '../shared/diegetic_ux.js';

export const TWEEZER_BOOKMARKS = {
  bench:   { pos: [0.20, 6.1, 13.0],  target: [0.25, 1.60, -0.60] },
  focus:   { pos: [0.95, 2.30, 4.15], target: [0.0, 1.68, 1.25] },
  monitor: { pos: [2.05, 3.15, 1.05], target: [2.42, 2.78, -3.72] },
  optics:  { pos: [-5.8, 3.6, 6.2],   target: [-3.2, 1.55, 1.15] },
};

export class FreeSpaceTweezerScene {
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
    this.camera.position.set(...TWEEZER_BOOKMARKS.bench.pos);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.target.set(...TWEEZER_BOOKMARKS.bench.target);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.06;
    this.controls.maxPolarAngle = Math.PI * 0.49;
    this.controls.minDistance = 2.2;
    this.controls.maxDistance = 24;
    this.controls.update();

    this.diegetic = new DiegeticController(this.camera, this.renderer.domElement, this.controls, tooltipEl);
    this.mats = createOpticsMaterials(3.8, 2.4);

    // Post-processing with controlled HDR bloom (keeps monitor & glass razor sharp)
    this.composer = new EffectComposer(this.renderer);
    this.composer.addPass(new RenderPass(this.scene, this.camera));
    this.bloomPass = new UnrealBloomPass(
      new THREE.Vector2(container.clientWidth, container.clientHeight),
      0.24, 0.42, 1.05
    );
    this.composer.addPass(this.bloomPass);
    this.composer.addPass(new OutputPass());

    this.beams = [];
    this.simState = { x: 0, y: 0, z: 0, vx: 0, vy: 0, vz: 0 };
    this.currentParams = null;
    this.currentTelemetry = null;
    this.clock = new THREE.Clock();

    this._buildLights();
    buildDarkroomBackdrop(this.scene);
    this._buildBenchEquipment();
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

    // Localized ruby bounce light inside the trapping chamber
    this.trapGlow = new THREE.PointLight(0xff1a34, 2.8, 5.4, 2.0);
    this.trapGlow.position.set(0.0, 1.55, 1.25);
    this.scene.add(this.trapGlow);
  }

  _buildBenchEquipment() {
    const Y = 1.68;
    const RAIL_Z = 1.25;
    const LASER_Z = -2.35;

    // 1. Matte-black anodized optical breadboard table
    const table = createOpticalTable(this.mats, 17.2, 10.6, 0.56);
    this.scene.add(table);

    // 2. 1064 nm Trapping Laser Head firing -X toward Mirror M1
    this.laserHead = createLaserHead(this.mats, new THREE.Vector3(-3.55, Y, LASER_Z), Math.PI);
    this.scene.add(this.laserHead);

    // 3. Kinematic Steering Mirrors M1 & M2
    const m1Pos = new THREE.Vector3(-6.85, Y, LASER_Z);
    const m2Pos = new THREE.Vector3(-6.85, Y, RAIL_Z);
    this.scene.add(createKinematicMirror(this.mats, m1Pos, Math.PI * 0.25, 0.42));
    this.scene.add(createPostAssembly(this.mats, m1Pos.x, m1Pos.z, Y - 0.48));
    this.scene.add(createKinematicMirror(this.mats, m2Pos, Math.PI * 0.75, 0.42));
    this.scene.add(createPostAssembly(this.mats, m2Pos.x, m2Pos.z, Y - 0.48));

    // 4. Polarization Optics Train along main front rail (z = +1.25)
    // λ/2 Waveplate 1 (Power attenuation before PBS)
    this.wpPower = createWaveplateMount(this.mats, new THREE.Vector3(-5.55, Y, RAIL_Z), 0, this.mats.arLensViolet);
    this.scene.add(this.wpPower);
    this.scene.add(createPostAssembly(this.mats, -5.55, RAIL_Z, Y - 0.50));

    // Polarizing Beam Splitter (PBS) glass cube
    const pbsPos = new THREE.Vector3(-4.35, Y, RAIL_Z);
    this.scene.add(createBeamSplitterCube(this.mats, pbsPos, Math.PI / 4, 0.82));
    this.scene.add(createPostAssembly(this.mats, pbsPos.x, pbsPos.z, Y - 0.49));

    // λ/2 Waveplate 2 (Ellipticity / transverse axis rotation)
    this.wpEllip = createWaveplateMount(this.mats, new THREE.Vector3(-3.12, Y, RAIL_Z), 0, this.mats.arLensViolet);
    this.scene.add(this.wpEllip);
    this.scene.add(createPostAssembly(this.mats, -3.12, RAIL_Z, Y - 0.50));

    // λ/4 Waveplate (Emerald AR coating)
    this.wpQuarter = createWaveplateMount(this.mats, new THREE.Vector3(-2.28, Y, RAIL_Z), 0, this.mats.arLensEmerald);
    this.scene.add(this.wpQuarter);
    this.scene.add(createPostAssembly(this.mats, -2.28, RAIL_Z, Y - 0.50));

    // 5. Central UHV Vacuum Chamber + Opposing High-NA Objectives + 4 Electrodes + Nanosphere
    const focusCenter = new THREE.Vector3(0.0, Y, RAIL_Z);
    this.chamber = createUHVChamber(this.mats, focusCenter, {
      radius: 1.62,
      height: 2.58,
      sidePorts: true,
      frontBezel: true,
    });
    this.scene.add(this.chamber);

    // Opposing NA=0.77 aspheric trapping objective & condenser inside the UHV chamber
    this.scene.add(createObjectiveBarrel(this.mats, new THREE.Vector3(-0.88, Y, RAIL_Z), new THREE.Vector3(1, 0, 0), 0.92));
    this.scene.add(createObjectiveBarrel(this.mats, new THREE.Vector3(+0.88, Y, RAIL_Z), new THREE.Vector3(-1, 0, 0), 0.92));
    this.scene.add(createPostAssembly(this.mats, -0.88, RAIL_Z, Y - 0.34));
    this.scene.add(createPostAssembly(this.mats, +0.88, RAIL_Z, Y - 0.34));

    // Volumetric Gaussian Focusing Cones inside chamber (left incoming cone + right diverging cone)
    this.focusConeMat = new THREE.ShaderMaterial({
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
          float waist = 0.55 + 1.15 * exp(-pow((vUv.y - 0.5) * 4.2, 2.0));
          vec3 col = mix(uOuter, uCore, clamp(f * waist * 0.45, 0.0, 1.0));
          gl_FragColor = vec4(col * uBoost, clamp(f * waist * 0.58, 0.0, 0.88));
        }
      `,
    });

    this.coneLeft = new THREE.Mesh(new THREE.CylinderGeometry(0.032, 0.21, 0.50, 32, 1, true), this.focusConeMat);
    this.coneLeft.rotation.z = -Math.PI / 2;
    this.coneLeft.position.set(-0.25, Y, RAIL_Z);
    this.scene.add(this.coneLeft);

    this.coneRight = new THREE.Mesh(new THREE.CylinderGeometry(0.21, 0.032, 0.50, 32, 1, true), this.focusConeMat);
    this.coneRight.rotation.z = -Math.PI / 2;
    this.coneRight.position.set(+0.25, Y, RAIL_Z);
    this.scene.add(this.coneRight);

    // 4 Cylindrical Stainless-Steel Electrodes (PRR 6 setup_b geometry in transverse x-y plane)
    this.electrodeGroup = new THREE.Group();
    this.electrodeGroup.position.copy(focusCenter);
    const elecAngles = [Math.PI * 0.25, Math.PI * 0.75, Math.PI * 1.25, Math.PI * 1.75];
    elecAngles.forEach(ang => {
      const rod = new THREE.Mesh(new THREE.CylinderGeometry(0.055, 0.055, 0.72, 20), this.mats.electrode);
      // Transverse plane is world (Y, Z) since beam propagates along +X
      const dirY = Math.cos(ang);
      const dirZ = Math.sin(ang);
      rod.position.set(0, dirY * 0.54, dirZ * 0.54);
      rod.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), new THREE.Vector3(0, dirY, dirZ));
      this.electrodeGroup.add(rod);
    });

    // Subtle glowing directed-force angle ψ vector + ring inside the 4 electrodes
    this.forceVecGroup = new THREE.Group();
    this.forceMat = new THREE.MeshBasicMaterial({
      color: 0x34d399,
      transparent: true,
      opacity: 0.85,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });
    const shaft = new THREE.Mesh(new THREE.CylinderGeometry(0.016, 0.016, 0.68, 12), this.forceMat);
    const tipPos = new THREE.Mesh(new THREE.ConeGeometry(0.048, 0.11, 14), this.forceMat);
    tipPos.position.y = 0.36;
    const tipNeg = new THREE.Mesh(new THREE.ConeGeometry(0.048, 0.11, 14), this.forceMat);
    tipNeg.rotation.z = Math.PI;
    tipNeg.position.y = -0.36;
    this.forceVecGroup.add(shaft, tipPos, tipNeg);
    this.electrodeGroup.add(this.forceVecGroup);

    // Translucent interaction ring around electrodes for easy diegetic dragging of ψ
    const elecHitRing = new THREE.Mesh(
      new THREE.TorusGeometry(0.68, 0.05, 12, 40),
      new THREE.MeshBasicMaterial({ color: 0x34d399, transparent: true, opacity: 0.0, depthWrite: false })
    );
    elecHitRing.rotation.y = Math.PI / 2;
    this.electrodeGroup.add(elecHitRing);
    this.scene.add(this.electrodeGroup);

    // Levitated Glowing Silica Nanosphere + subtle motion trail halo
    this.beadAnchor = focusCenter.clone();
    this.beadMesh = new THREE.Mesh(
      new THREE.SphereGeometry(0.068, 32, 32),
      new THREE.MeshStandardMaterial({
        color: 0xffffff,
        emissive: 0xff5268,
        emissiveIntensity: 3.0,
        roughness: 0.1,
      })
    );
    this.beadMesh.position.copy(this.beadAnchor);
    this.scene.add(this.beadMesh);

    // 6. Downstream 50/50 Glass Beam Splitter Cube + Balanced x & y Detection Arms
    const bs50Pos = new THREE.Vector3(3.35, Y, RAIL_Z);
    this.scene.add(createBeamSplitterCube(this.mats, bs50Pos, -Math.PI / 4, 0.84));
    this.scene.add(createPostAssembly(this.mats, bs50Pos.x, bs50Pos.z, Y - 0.50));

    // Steering mirror M3 at far right directing y-detection arm backward (-Z)
    const m3Pos = new THREE.Vector3(6.25, Y, RAIL_Z);
    this.scene.add(createKinematicMirror(this.mats, m3Pos, -Math.PI * 0.75, 0.42));
    this.scene.add(createPostAssembly(this.mats, m3Pos.x, m3Pos.z, Y - 0.48));

    // D-shaped split mirrors + balanced photodetectors for x-arm (x=3.35) and y-arm (x=6.25)
    const dMirrorX = new THREE.Vector3(3.35, Y, -0.55);
    const dMirrorY = new THREE.Vector3(6.25, Y, -0.55);
    [dMirrorX, dMirrorY].forEach(p => {
      const dm = createKinematicMirror(this.mats, p, -Math.PI * 0.25, 0.30);
      this.scene.add(dm);
      this.scene.add(createPostAssembly(this.mats, p.x, p.z, Y - 0.36));
    });

    const pdX1 = new THREE.Vector3(3.35, Y, -1.90);
    const pdX2 = new THREE.Vector3(2.05, Y, -0.55);
    const pdY1 = new THREE.Vector3(6.25, Y, -1.90);
    const pdY2 = new THREE.Vector3(4.95, Y, -0.55);

    this.scene.add(createPhotodetector(this.mats, pdX1, Math.PI));
    this.scene.add(createPostAssembly(this.mats, pdX1.x, pdX1.z, Y - 0.36));
    this.scene.add(createPhotodetector(this.mats, pdX2, Math.PI / 2));
    this.scene.add(createPostAssembly(this.mats, pdX2.x, pdX2.z, Y - 0.36));

    this.scene.add(createPhotodetector(this.mats, pdY1, Math.PI));
    this.scene.add(createPostAssembly(this.mats, pdY1.x, pdY1.z, Y - 0.36));
    this.scene.add(createPhotodetector(this.mats, pdY2, Math.PI / 2));
    this.scene.add(createPostAssembly(this.mats, pdY2.x, pdY2.z, Y - 0.36));

    // 7. Cold-Damping Feedback / DAQ Amplifier Box (interactive diegetic for Γ_fb!)
    this.daqBox = new THREE.Group();
    this.daqBox.position.set(-0.35, 0.36, -2.25);
    const daqBody = new THREE.Mesh(new RoundedBoxGeometry(1.65, 0.72, 1.15, 3, 0.04), this.mats.ampChassis);
    daqBody.castShadow = true;
    daqBody.receiveShadow = true;
    this.daqBox.add(daqBody);
    const daqFront = new THREE.Mesh(new THREE.BoxGeometry(1.52, 0.56, 0.04), this.mats.anodizedRing);
    daqFront.position.z = 0.57;
    this.daqBox.add(daqFront);
    this.daqKnob = new THREE.Mesh(new THREE.CylinderGeometry(0.14, 0.15, 0.14, 24), this.mats.brass);
    this.daqKnob.rotation.x = Math.PI / 2;
    this.daqKnob.position.set(0.42, 0.0, 0.64);
    this.daqBox.add(this.daqKnob);
    const daqLed = new THREE.Mesh(new THREE.SphereGeometry(0.045, 10, 10), new THREE.MeshBasicMaterial({ color: 0x38bdf8 }));
    daqLed.position.set(-0.52, 0.14, 0.60);
    this.daqBox.add(daqLed);
    this.scene.add(this.daqBox);

    // 8. Laser Beam Paths across the entire optical bench
    const addBeam = (a, b, r, op) => {
      const seg = createBeamSegment(a, b, { radius: r, opacity: op });
      this.beams.push(seg);
      this.scene.add(seg);
    };
    addBeam(new THREE.Vector3(-4.95, Y, LASER_Z), m1Pos, 0.056, 0.82);
    addBeam(m1Pos, m2Pos, 0.056, 0.80);
    addBeam(m2Pos, new THREE.Vector3(-0.98, Y, RAIL_Z), 0.056, 0.82);
    addBeam(new THREE.Vector3(+0.98, Y, RAIL_Z), bs50Pos, 0.052, 0.76);
    // x-detection split arm
    addBeam(bs50Pos, pdX1, 0.044, 0.64);
    addBeam(dMirrorX, pdX2, 0.042, 0.58);
    // y-detection split arm
    addBeam(bs50Pos, m3Pos, 0.046, 0.68);
    addBeam(m3Pos, pdY1, 0.044, 0.64);
    addBeam(dMirrorY, pdY2, 0.042, 0.58);

    // 9. Braided Coaxial Cables routing signals to DAQ box & RF Spectrum Analyzer
    const monBack = [2.4, 0.35, -3.5];
    this.scene.add(createCoaxCable(this.mats, [
      [pdX1.x, Y, pdX1.z - 0.48], [3.35, 0.25, -2.55], [2.7, 0.08, -3.1], monBack,
    ]));
    this.scene.add(createCoaxCable(this.mats, [
      [pdY1.x, Y, pdY1.z - 0.48], [6.2, 0.22, -2.65], [4.5, 0.06, -3.2], [3.1, 0.35, -3.55],
    ]));
    this.scene.add(createCoaxCable(this.mats, [
      [0.0, 0.42, -0.35], [-0.15, 0.12, -1.1], [-0.35, 0.32, -1.68],
    ]));
    this.scene.add(createCoaxCable(this.mats, [
      [0.48, 0.36, -2.25], [1.2, 0.08, -2.95], [1.9, 0.32, -3.45],
    ]));

    // 10. In-Game Benchtop RF Spectrum Analyzer Monitor (elevated cleanly above optics line)
    this.monitor = new SpectrumMonitor(this.mats, {
      position: new THREE.Vector3(2.45, 2.78, -3.72),
      rotationY: -0.18,
      width: 4.85,
      height: 3.12,
      theme: 'rf',
    });
    this.scene.add(this.monitor.group);
  }

  _bindDiegeticControls() {
    const addRing = (parent, radius, rot = [0, 0, 0], offset = [0, 0, 0]) => {
      const ring = new THREE.Mesh(new THREE.TorusGeometry(radius, 0.032, 12, 40), this.mats.hoverHighlight);
      ring.rotation.set(...rot);
      ring.position.set(...offset);
      parent.add(ring);
      return ring;
    };

    // 1. Laser Head -> Power P (mW)
    this.diegetic.register({
      object: this.laserHead,
      highlightRing: addRing(this.laserHead, 0.34, [Math.PI / 2, 0, 0], [0.88, 0.68, 0.34]),
      title: '1064 nm Trapping Laser',
      hint: 'Drag left/right to vary laser power P',
      getValue: () => this.currentParams ? `P = ${this.currentParams.power_mW.toFixed(0)} mW` : '',
      onDrag: (dx) => this.callbacks.onDeltaParam('power_mW', dx * 3.2),
    });

    // 2. λ/2 Waveplate 2 -> Beam Waist Ellipticity (wx0 vs wy0)
    this.diegetic.register({
      object: this.wpEllip,
      highlightRing: addRing(this.wpEllip, 0.56, [0, Math.PI / 2, 0]),
      title: 'λ/2 Ellipticity Waveplate',
      hint: 'Drag to tune transverse waist wx0 (splits Ωx & Ωy)',
      getValue: () => this.currentParams
        ? `wx0 = ${this.currentParams.wx0_um.toFixed(3)} μm, wy0 = ${this.currentParams.wy0_um.toFixed(3)} μm`
        : '',
      onDrag: (dx) => this.callbacks.onDeltaParam('wx0_um', dx * 0.0022),
    });

    // 3. 4-Electrode Ring inside UHV chamber -> Directed-force angle ψ
    this.diegetic.register({
      object: this.electrodeGroup,
      highlightRing: addRing(this.electrodeGroup, 0.72, [0, Math.PI / 2, 0]),
      title: '4-Electrode Force Stage (PRR 6)',
      hint: 'Drag to rotate directed noise angle ψ (flips Cxy sign!)',
      getValue: () => this.currentParams
        ? `ψ = ${this.currentParams.psi_deg >= 0 ? '+' : ''}${this.currentParams.psi_deg.toFixed(1)}° (β = ${this.currentParams.beta.toFixed(2)})`
        : '',
      onDrag: (dx) => this.callbacks.onDeltaParam('psi_deg', dx * 0.65),
    });

    // 4. Vacuum Pressure Gauge -> Gas pressure P_gas (log scale)
    this.diegetic.register({
      object: this.chamber.userData.gaugeGroup,
      highlightRing: addRing(this.chamber.userData.gaugeGroup, 0.38, [0, 0, 0], [0, 0.14, 0.08]),
      title: 'UHV Chamber Pressure Gauge',
      hint: 'Drag to adjust vacuum gas pressure (log mbar)',
      getValue: () => this.currentParams ? `P_gas = ${this.currentParams.pressure_mbar.toExponential(1)} mbar` : '',
      onDrag: (dx) => this.callbacks.onDeltaLogPressure(dx * 0.025),
    });

    // 5. DAQ / Cold-Damping Feedback Box -> Γ_fb,x & Γ_fb,y
    this.diegetic.register({
      object: this.daqBox,
      highlightRing: addRing(this.daqBox, 0.25, [0, 0, 0], [0.42, 0.0, 0.68]),
      title: 'Feedback Cold-Damping Controller',
      hint: 'Drag to adjust feedback damping Γ_fb (cools T_eff)',
      getValue: () => this.currentParams
        ? `Γfb,x = ${this.currentParams.gamma_fbx_kHz.toFixed(2)} kHz`
        : '',
      onDrag: (dx) => this.callbacks.onDeltaFeedback(dx * 0.045),
    });

    // 6. In-Game RF Spectrum Analyzer Monitor -> Click to zoom in
    this.diegetic.register({
      object: this.monitor.group,
      title: 'In-Game RF Spectrum Analyzer',
      hint: 'Click to focus camera onto monitor screen',
      getValue: () => this.monitor.powerOn ? 'Monitor ON' : 'Monitor STANDBY',
      onClick: () => this.flyToBookmark('monitor'),
    });
  }

  flyToBookmark(key) {
    const bm = TWEEZER_BOOKMARKS[key];
    if (bm) this.diegetic.flyTo(bm.pos, bm.target, 680);
  }

  setChamberMode(mode) {
    this.chamber.userData.setMode(mode);
  }

  /**
   * Updates all 3D visual elements (laser brightness/wavelength, waveplate rotation,
   * electrode force field arrow, vacuum gauge needle, photon-recoil z_eq shift,
   * bead size) smoothly from current physics state.
   */
  syncVisuals(params, telemetry) {
    this.currentParams = params;
    this.currentTelemetry = telemetry;

    // Laser color & power intensity
    const cols = wavelengthToLaserColors(params.wavelength_nm);
    const powerNorm = Math.max(0.25, Math.min(1.65, params.power_mW / 485.0));
    for (const seg of this.beams) {
      seg.userData.outerMat.color.copy(cols.outer);
      seg.userData.coreMat.color.copy(cols.core);
      seg.userData.outerMat.opacity = Math.min(0.92, seg.userData.baseOpacity * (0.45 + 0.55 * powerNorm));
      seg.userData.coreMat.opacity = Math.min(0.96, seg.userData.baseOpacity * (0.55 + 0.55 * powerNorm));
    }
    this.focusConeMat.uniforms.uOuter.value.copy(cols.outer);
    this.focusConeMat.uniforms.uCore.value.copy(cols.core);
    this.focusConeMat.uniforms.uBoost.value = 0.72 + 0.52 * powerNorm;
    this.trapGlow.color.setHex(cols.point);
    this.trapGlow.intensity = 1.4 + 1.6 * powerNorm;
    this.laserHead.userData.emitCore.material.color.copy(cols.core);
    this.laserHead.userData.powerKnob.rotation.y = -(params.power_mW / 1200) * Math.PI * 1.6;

    // Waveplate visual rotations
    this.wpPower.userData.rotor.rotation.x = (params.power_mW / 1000) * Math.PI;
    const ellipAngle = ((params.wx0_um - 0.9276) / 0.5) * Math.PI * 0.8;
    this.wpEllip.userData.rotor.rotation.x = ellipAngle;

    // Directed force vector orientation ψ in transverse (Y, Z) plane & opacity from β
    const psiRad = (params.psi_deg * Math.PI) / 180;
    this.forceVecGroup.rotation.x = psiRad;
    this.forceVecGroup.visible = params.beta > 0.02;
    this.forceMat.opacity = Math.min(0.95, 0.25 + 0.35 * params.beta);
    this.forceMat.color.setHex(params.psi_deg >= 0 ? 0x34d399 : 0xf472b6);

    // Vacuum gauge needle (log10 pressure from -7 to 0)
    const logP = Math.max(-7, Math.min(0, Math.log10(Math.max(1e-8, params.pressure_mbar))));
    const gaugeAngle = -((logP + 7) / 7) * Math.PI * 1.35 + Math.PI * 0.65;
    this.chamber.userData.gaugeNeedle.rotation.z = gaugeAngle;

    // Feedback knob rotation
    this.daqKnob.rotation.y = -(params.gamma_fbx_kHz / 10) * Math.PI * 1.5;

    // Bead radius subtle visual scaling
    const rScale = Math.max(0.55, Math.min(1.85, params.radius_nm / 60.1));
    this.beadMesh.scale.setScalar(rScale);

    // Beam waist cone subtle visual stretching when wx0/wy0 change
    const wScale = Math.max(0.7, Math.min(1.5, Math.sqrt(params.wx0_um * params.wy0_um)));
    this.coneLeft.scale.set(wScale, 1, wScale);
    this.coneRight.scale.set(wScale, 1, wScale);
  }

  _animate() {
    requestAnimationFrame(() => this._animate());
    const dt = Math.min(0.04, this.clock.getDelta());

    this.controls.update();
    this.diegetic.update();

    // Live exaggerated stochastic Langevin motion of the levitated silica nanosphere
    if (this.currentParams && this.currentTelemetry) {
      const tel = this.currentTelemetry;
      const p = this.currentParams;
      // Scale visual harmonic frequencies down into visible ~2-6 Hz slow motion preserving Ωx:Ωy:Ωz ratios
      const fScale = 0.032;
      const wx = tel.fx_kHz * fScale * Math.PI * 2;
      const wy = tel.fy_kHz * fScale * Math.PI * 2;
      const wz = tel.fz_kHz * fScale * Math.PI * 2;

      const dampX = Math.max(0.4, Math.min(6.0, tel.gamma_x_kHz * 0.55));
      const dampY = Math.max(0.4, Math.min(6.0, tel.gamma_y_kHz * 0.55));
      const dampZ = Math.max(0.25, Math.min(3.5, tel.gamma_z_kHz * 0.55 + 0.35));

      // Amplitude scales realistically with sqrt(T_eff)
      const ampX = Math.min(0.11, 0.012 + 0.022 * Math.sqrt(Math.max(0.01, tel.T_eff_x_K)));
      const ampY = Math.min(0.11, 0.012 + 0.022 * Math.sqrt(Math.max(0.01, tel.T_eff_y_K)));
      const ampZ = Math.min(0.09, 0.014 + 0.004 * Math.sqrt(Math.max(0.01, tel.T_eff_z_K)));

      const psiRad = (p.psi_deg * Math.PI) / 180;
      const randn = () => (Math.random() + Math.random() + Math.random() - 1.5) * 1.63;
      const dirKick = p.beta * 0.65 * randn();
      const s = this.simState;
      const sub = 4;
      const h = dt / sub;
      for (let i = 0; i < sub; i++) {
        const ax = -wx * wx * s.x - dampX * s.vx + ampX * wx * 2.4 * (randn() + dirKick * Math.cos(psiRad));
        const ay = -wy * wy * s.y - dampY * s.vy + ampY * wy * 2.4 * (randn() + dirKick * Math.sin(psiRad));
        const az = -wz * wz * s.z - dampZ * s.vz + ampZ * wz * 2.0 * randn();
        s.vx += ax * h; s.vy += ay * h; s.vz += az * h;
        s.x = Math.max(-0.14, Math.min(0.14, s.x + s.vx * h));
        s.y = Math.max(-0.14, Math.min(0.14, s.y + s.vy * h));
        s.z = Math.max(-0.14, Math.min(0.14, s.z + s.vz * h));
      }

      // Photon recoil downstream equilibrium shift (+X world = +z beam axis)
      const zShiftWorld = p.photon_recoil ? Math.min(0.16, (tel.z_eq_nm / 355.0) * 0.095) : 0.0;
      this.beadMesh.position.set(
        this.beadAnchor.x + zShiftWorld + s.z,
        this.beadAnchor.y + s.x,
        this.beadAnchor.z + s.y
      );
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

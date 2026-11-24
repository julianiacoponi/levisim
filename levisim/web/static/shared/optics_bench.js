/**
 * Reusable Thorlabs-inspired precision optomechanical equipment builders
 * for levisim Three.js optical bench scenes.
 */
import * as THREE from 'three';
import { RoundedBoxGeometry } from 'three/addons/geometries/RoundedBoxGeometry.js';

/**
 * Creates a beveled optical breadboard table with M6 hole grid on top.
 */
export function createOpticalTable(mats, width = 16.4, depth = 10.2, thickness = 0.56) {
  const geo = new RoundedBoxGeometry(width, thickness, depth, 3, 0.04);
  const mesh = new THREE.Mesh(geo, [
    mats.tableSide, mats.tableSide,
    mats.breadboard, mats.tableSide,
    mats.tableSide, mats.tableSide,
  ]);
  mesh.position.set(0, -thickness / 2, 0);
  mesh.receiveShadow = true;

  // Brushed stainless edge trim accent along front bevel
  const trim = new THREE.Mesh(
    new THREE.BoxGeometry(width - 0.1, 0.045, 0.04),
    mats.stainless
  );
  trim.position.set(0, -0.05, depth / 2 + 0.005);
  mesh.add(trim);
  return mesh;
}

/**
 * Creates a Thorlabs-style pedestal base, post holder with brass locking thumbscrew,
 * and brushed stainless-steel optical post reaching up to `topY`.
 */
export function createPostAssembly(mats, x, z, topY) {
  const g = new THREE.Group();
  g.position.set(x, 0, z);

  const base = new THREE.Mesh(new THREE.CylinderGeometry(0.32, 0.37, 0.13, 32), mats.anodized);
  base.position.y = 0.065;
  base.castShadow = true;
  base.receiveShadow = true;
  g.add(base);

  const holderH = Math.max(0.32, Math.min(0.82, topY * 0.52));
  const holder = new THREE.Mesh(new THREE.CylinderGeometry(0.17, 0.17, holderH, 24), mats.anodized);
  holder.position.y = 0.13 + holderH / 2;
  holder.castShadow = true;
  g.add(holder);

  // Brass knurled locking thumbscrew
  const thumb = new THREE.Mesh(new THREE.CylinderGeometry(0.068, 0.068, 0.22, 20), mats.brass);
  thumb.rotation.z = Math.PI / 2;
  thumb.position.set(0.21, 0.13 + holderH * 0.72, 0);
  g.add(thumb);

  const postH = Math.max(0.22, topY - 0.13);
  const post = new THREE.Mesh(new THREE.CylinderGeometry(0.10, 0.10, postH, 24), mats.stainless);
  post.position.y = 0.13 + postH / 2;
  post.castShadow = true;
  g.add(post);

  return g;
}

/**
 * Creates a matte-black anodized 1064 nm trapping laser head with heatsink fins,
 * brushed front plate, power emission LED, and rotatable top power dial.
 */
export function createLaserHead(mats, pos, rotY = 0) {
  const g = new THREE.Group();
  g.position.copy(pos);
  g.rotation.y = rotY;

  // Base mounting plate bolted to breadboard
  const basePlate = new THREE.Mesh(new RoundedBoxGeometry(2.75, 0.16, 1.75, 2, 0.02), mats.anodizedRing);
  basePlate.position.y = -pos.y + 0.08;
  basePlate.castShadow = true;
  basePlate.receiveShadow = true;
  g.add(basePlate);

  // Support riser
  const riserH = Math.max(0.1, pos.y - 0.72);
  const riser = new THREE.Mesh(new THREE.BoxGeometry(2.2, riserH, 1.25), mats.anodized);
  riser.position.y = -pos.y + 0.16 + riserH / 2;
  riser.castShadow = true;
  g.add(riser);

  // Main laser housing body
  const body = new THREE.Mesh(new RoundedBoxGeometry(2.55, 1.15, 1.42, 4, 0.06), mats.anodized);
  body.castShadow = true;
  g.add(body);

  // Top cooling fins
  for (let i = -4; i <= 4; i++) {
    const fin = new THREE.Mesh(new THREE.BoxGeometry(2.15, 0.05, 0.055), mats.anodizedRing);
    fin.position.set(-0.08, 0.59, i * 0.14);
    g.add(fin);
  }

  // Front stainless bezel & recessed laser output aperture (+X direction)
  const frontBezel = new THREE.Mesh(new RoundedBoxGeometry(0.10, 1.06, 1.32, 3, 0.03), mats.stainless);
  frontBezel.position.x = 1.26;
  g.add(frontBezel);

  const barrel = new THREE.Mesh(new THREE.CylinderGeometry(0.24, 0.28, 0.22, 32), mats.anodizedRing);
  barrel.rotation.z = Math.PI / 2;
  barrel.position.x = 1.36;
  g.add(barrel);

  const emitCore = new THREE.Mesh(
    new THREE.CircleGeometry(0.14, 24),
    new THREE.MeshBasicMaterial({ color: 0xff3b52 })
  );
  emitCore.rotation.y = Math.PI / 2;
  emitCore.position.x = 1.475;
  g.add(emitCore);

  // Green emission status LED
  const led = new THREE.Mesh(
    new THREE.SphereGeometry(0.048, 12, 12),
    new THREE.MeshBasicMaterial({ color: 0x22c55e })
  );
  led.position.set(1.32, -0.32, 0.44);
  g.add(led);

  // Interactive rotary power knob on top-front
  const knobGroup = new THREE.Group();
  knobGroup.position.set(0.88, 0.66, 0.34);
  const knobBody = new THREE.Mesh(new THREE.CylinderGeometry(0.16, 0.18, 0.16, 28), mats.brass);
  knobGroup.add(knobBody);
  const notch = new THREE.Mesh(new THREE.BoxGeometry(0.12, 0.04, 0.035), mats.anodized);
  notch.position.set(0.06, 0.085, 0);
  knobGroup.add(notch);
  g.add(knobGroup);

  g.userData.emitCore = emitCore;
  g.userData.powerKnob = knobGroup;
  return g;
}

/**
 * Creates a precision kinematic steering mirror mount with 3 brass micrometer adjusters
 * and a thick fused-silica dielectric mirror substrate.
 * The mirror reflective face points along local +Z.
 */
export function createKinematicMirror(mats, pos, rotY = 0, radius = 0.42) {
  const g = new THREE.Group();
  g.position.copy(pos);
  g.rotation.y = rotY;

  const backPlate = new THREE.Mesh(new RoundedBoxGeometry(radius * 2.25, radius * 2.25, 0.14, 2, 0.02), mats.anodizedRing);
  backPlate.position.z = -0.22;
  backPlate.castShadow = true;
  g.add(backPlate);

  const frontPlate = new THREE.Mesh(new RoundedBoxGeometry(radius * 2.15, radius * 2.15, 0.14, 2, 0.02), mats.anodized);
  frontPlate.position.z = 0.0;
  frontPlate.castShadow = true;
  g.add(frontPlate);

  // Fused silica mirror substrate + HR coated front face
  const glass = new THREE.Mesh(new THREE.CylinderGeometry(radius, radius, 0.18, 40), mats.fusedSilica);
  glass.rotation.x = Math.PI / 2;
  glass.position.z = 0.12;
  g.add(glass);

  const hrFace = new THREE.Mesh(new THREE.CircleGeometry(radius * 0.94, 40), mats.hrCoating);
  hrFace.position.z = 0.214;
  g.add(hrFace);

  // 3 fine-pitch brass micrometer screws protruding behind backPlate
  const off = radius * 0.78;
  [[off, off], [off, -off], [-off, off]].forEach(([mx, my]) => {
    const screw = new THREE.Mesh(new THREE.CylinderGeometry(0.068, 0.068, 0.34, 18), mats.brass);
    screw.rotation.x = Math.PI / 2;
    screw.position.set(mx, my, -0.38);
    g.add(screw);
  });

  return g;
}

/**
 * Creates a precision rotating waveplate mount (λ/2 or λ/4) whose inner ring
 * and radial indicator handle rotate around the beam axis (+X local).
 */
export function createWaveplateMount(mats, pos, rotY = 0, lensMat = null) {
  const g = new THREE.Group();
  g.position.copy(pos);
  g.rotation.y = rotY;

  // Outer stationary anodized frame
  const outerRing = new THREE.Mesh(new THREE.TorusGeometry(0.46, 0.075, 20, 48), mats.anodized);
  outerRing.rotation.y = Math.PI / 2;
  outerRing.castShadow = true;
  g.add(outerRing);

  // Rotatable inner degree-scale collar + knurled thumb lever
  const rotor = new THREE.Group();
  const innerRing = new THREE.Mesh(new THREE.TorusGeometry(0.39, 0.048, 16, 48), mats.anodizedRed);
  innerRing.rotation.y = Math.PI / 2;
  rotor.add(innerRing);

  const optic = new THREE.Mesh(
    new THREE.CylinderGeometry(0.36, 0.36, 0.06, 36),
    lensMat || mats.arLensViolet
  );
  optic.rotation.z = Math.PI / 2;
  rotor.add(optic);

  // Brass radial rotation lever tab (clearly shows waveplate orientation angle!)
  const lever = new THREE.Mesh(new THREE.CylinderGeometry(0.036, 0.044, 0.28, 14), mats.brass);
  lever.position.set(0, 0.52, 0);
  rotor.add(lever);

  g.add(rotor);
  g.userData.rotor = rotor;
  return g;
}

/**
 * Creates a crystal-clear N-BK7 Polarizing or 50:50 Beam Splitter Cube
 * with visible internal iridescent diagonal dielectric plane and clamping cap.
 */
export function createBeamSplitterCube(mats, pos, diagRotY = Math.PI / 4, size = 0.84) {
  const g = new THREE.Group();
  g.position.copy(pos);

  const cube = new THREE.Mesh(new RoundedBoxGeometry(size, size, size, 3, 0.02), mats.pbsGlass);
  g.add(cube);

  const diag = new THREE.Mesh(new THREE.PlaneGeometry(size * Math.SQRT2 * 0.96, size * 0.95), mats.pbsDiagonal);
  diag.rotation.y = diagRotY;
  g.add(diag);

  // Subtle metallic frame edge highlights & pedestal platform
  const platform = new THREE.Mesh(new RoundedBoxGeometry(size + 0.14, 0.14, size + 0.14, 2, 0.02), mats.anodized);
  platform.position.y = -size / 2 - 0.07;
  platform.castShadow = true;
  g.add(platform);

  // Top precision nylon/steel retaining cap thumbscrew
  const clampBar = new THREE.Mesh(new THREE.BoxGeometry(size * 0.82, 0.055, 0.16), mats.anodizedRing);
  clampBar.position.y = size / 2 + 0.03;
  g.add(clampBar);
  const topScrew = new THREE.Mesh(new THREE.CylinderGeometry(0.085, 0.085, 0.18, 20), mats.stainless);
  topScrew.position.y = size / 2 + 0.14;
  g.add(topScrew);

  return g;
}

/**
 * Creates a lathe-turned high-NA (0.77) aspheric objective/condenser lens barrel.
 * By default its optical axis points along +Y; pass axisDir to orient along any unit vector.
 */
export function createObjectiveBarrel(mats, pos, axisDir = new THREE.Vector3(0, 1, 0), scale = 1.0) {
  const g = new THREE.Group();
  g.position.copy(pos);
  g.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), axisDir.clone().normalize());
  g.scale.setScalar(scale);

  const body = new THREE.Mesh(new THREE.CylinderGeometry(0.32, 0.36, 0.72, 36), mats.anodized);
  body.castShadow = true;
  g.add(body);

  const grip = new THREE.Mesh(new THREE.CylinderGeometry(0.38, 0.38, 0.14, 36), mats.anodizedRing);
  grip.position.y = -0.14;
  g.add(grip);

  const redStripe = new THREE.Mesh(new THREE.CylinderGeometry(0.345, 0.345, 0.04, 36), mats.anodizedRed);
  redStripe.position.y = 0.22;
  g.add(redStripe);

  const nose = new THREE.Mesh(new THREE.CylinderGeometry(0.23, 0.32, 0.17, 36), mats.anodized);
  nose.position.y = 0.44;
  g.add(nose);

  const lens = new THREE.Mesh(
    new THREE.SphereGeometry(0.21, 32, 16, 0, Math.PI * 2, 0, Math.PI * 0.42),
    mats.arLensViolet
  );
  lens.position.y = 0.39;
  g.add(lens);

  return g;
}

/**
 * Creates a balanced photodetector module with recessed stainless aperture and rear BNC port.
 * Aperture faces local -Z.
 */
export function createPhotodetector(mats, pos, rotY = 0) {
  const g = new THREE.Group();
  g.position.copy(pos);
  g.rotation.y = rotY;

  const box = new THREE.Mesh(new RoundedBoxGeometry(0.64, 0.68, 0.80, 3, 0.03), mats.anodized);
  box.castShadow = true;
  g.add(box);

  const collar = new THREE.Mesh(new THREE.CylinderGeometry(0.19, 0.21, 0.09, 28), mats.stainless);
  collar.rotation.x = Math.PI / 2;
  collar.position.z = -0.42;
  g.add(collar);

  const photodiode = new THREE.Mesh(new THREE.CircleGeometry(0.13, 20), mats.arLensEmerald);
  photodiode.rotation.y = Math.PI;
  photodiode.position.z = -0.466;
  g.add(photodiode);

  const bnc = new THREE.Mesh(new THREE.CylinderGeometry(0.055, 0.055, 0.20, 16), mats.stainless);
  bnc.rotation.x = Math.PI / 2;
  bnc.position.z = 0.48;
  g.add(bnc);

  const led = new THREE.Mesh(new THREE.SphereGeometry(0.035, 10, 10), new THREE.MeshBasicMaterial({ color: 0x38bdf8 }));
  led.position.set(0.22, 0.22, 0.41);
  g.add(led);

  return g;
}

/**
 * Creates a smooth Catmull-Rom coaxial cable connecting electronics and detectors.
 */
export function createCoaxCable(mats, points, radius = 0.036) {
  const curve = new THREE.CatmullRomCurve3(points.map(p => (p.isVector3 ? p : new THREE.Vector3(...p))));
  const mesh = new THREE.Mesh(new THREE.TubeGeometry(curve, 44, radius, 10, false), mats.cable);
  mesh.castShadow = true;
  return mesh;
}

/**
 * Creates a volumetric translucent laser beam segment (outer halo + bright inner core)
 * whose color and opacity can be updated live as laser power/wavelength change.
 */
export function createBeamSegment(p1, p2, {
  radius = 0.052,
  outerColor = 0xd90b24,
  coreColor = 0xff7585,
  opacity = 0.72,
} = {}) {
  const g = new THREE.Group();
  const dir = new THREE.Vector3().subVectors(p2, p1);
  const len = Math.max(1e-4, dir.length());

  const outerMat = new THREE.MeshBasicMaterial({
    color: outerColor,
    transparent: true,
    opacity,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
  });
  const coreMat = new THREE.MeshBasicMaterial({
    color: coreColor,
    transparent: true,
    opacity: Math.min(0.95, opacity * 1.15),
    blending: THREE.AdditiveBlending,
    depthWrite: false,
  });

  const outer = new THREE.Mesh(new THREE.CylinderGeometry(radius, radius, len, 18, 1, true), outerMat);
  const core = new THREE.Mesh(new THREE.CylinderGeometry(radius * 0.40, radius * 0.40, len, 12, 1, true), coreMat);
  g.add(outer, core);

  g.position.copy(p1).add(p2).multiplyScalar(0.5);
  g.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.normalize());
  g.userData.outerMat = outerMat;
  g.userData.coreMat = coreMat;
  g.userData.baseOpacity = opacity;
  return g;
}

/**
 * Creates a machined cylindrical stainless-steel + crystal-sapphire UHV Vacuum Chamber
 * with ConFlat bolted flanges, optical port collars, and a clean `.setMode('full'|'cutaway'|'open')` API.
 */
export function createUHVChamber(mats, center, {
  radius = 1.65,
  height = 2.65,
  sidePorts = true,
  frontBezel = true,
} = {}) {
  const ch = new THREE.Group();
  ch.position.copy(center);

  // Bottom ConFlat flange bolted to optical breadboard
  const baseY = -center.y;
  const flangeBot = new THREE.Mesh(new THREE.CylinderGeometry(radius + 0.28, radius + 0.38, 0.28, 48), mats.stainless);
  flangeBot.position.y = baseY + 0.14;
  flangeBot.castShadow = true;
  flangeBot.receiveShadow = true;
  ch.add(flangeBot);

  // Removable upper shell group (top flange, bolts, struts, sapphire wall, viewports)
  const upperGroup = new THREE.Group();
  const topY = baseY + height;

  const flangeTop = new THREE.Mesh(new THREE.CylinderGeometry(radius + 0.26, radius + 0.26, 0.24, 48), mats.stainless);
  flangeTop.position.y = topY;
  flangeTop.castShadow = true;
  upperGroup.add(flangeTop);

  // Top & bottom ConFlat hex bolts
  const numBolts = 16;
  for (let i = 0; i < numBolts; i++) {
    const a = (i / numBolts) * Math.PI * 2;
    const bx = Math.cos(a) * (radius + 0.12);
    const bz = Math.sin(a) * (radius + 0.12);
    const topBolt = new THREE.Mesh(new THREE.CylinderGeometry(0.055, 0.055, 0.08, 6), mats.titanium);
    topBolt.position.set(bx, topY + 0.15, bz);
    upperGroup.add(topBolt);

    const botBolt = new THREE.Mesh(new THREE.CylinderGeometry(0.055, 0.055, 0.08, 6), mats.titanium);
    botBolt.position.set(bx, baseY + 0.30, bz);
    ch.add(botBolt);
  }

  // 4 machined stainless tie-rods connecting flanges
  const strutH = height - 0.18;
  [Math.PI * 0.24, Math.PI * 0.76, Math.PI * 1.24, Math.PI * 1.76].forEach(ang => {
    const strut = new THREE.Mesh(new THREE.CylinderGeometry(0.095, 0.095, strutH, 20), mats.stainless);
    strut.position.set(Math.cos(ang) * (radius + 0.06), baseY + 0.14 + strutH / 2, Math.sin(ang) * (radius + 0.06));
    strut.castShadow = true;
    upperGroup.add(strut);
  });

  // Crystal-clear sapphire glass cylindrical sleeve (renderOrder keeps internal beams visible)
  const glassWall = new THREE.Mesh(
    new THREE.CylinderGeometry(radius - 0.02, radius - 0.02, strutH, 72, 1, true),
    mats.sapphireViewport
  );
  glassWall.position.y = baseY + 0.14 + strutH / 2;
  glassWall.renderOrder = 6;
  upperGroup.add(glassWall);

  // Side ConFlat optical viewport flanges along ±X
  if (sidePorts) {
    [-1, 1].forEach(side => {
      const portCollar = new THREE.Mesh(new THREE.TorusGeometry(0.68, 0.10, 18, 48), mats.stainless);
      portCollar.rotation.y = Math.PI / 2;
      portCollar.position.set(side * radius, 0, 0);
      upperGroup.add(portCollar);
    });
  }

  // Front circular sapphire viewport bezel ring with micro-bolts
  let frontBezelGroup = null;
  if (frontBezel) {
    frontBezelGroup = new THREE.Group();
    const rBez = Math.min(1.08, radius * 0.62);
    const ring = new THREE.Mesh(new THREE.TorusGeometry(rBez, 0.088, 18, 52), mats.stainless);
    ring.position.set(0, 0, radius - 0.01);
    frontBezelGroup.add(ring);
    for (let i = 0; i < 12; i++) {
      const a = (i / 12) * Math.PI * 2;
      const b = new THREE.Mesh(new THREE.CylinderGeometry(0.036, 0.036, 0.07, 6), mats.titanium);
      b.rotation.x = Math.PI / 2;
      b.position.set(Math.cos(a) * rBez, Math.sin(a) * rBez, radius + 0.07);
      frontBezelGroup.add(b);
    }
    upperGroup.add(frontBezelGroup);
  }

  // Side vacuum pressure gauge dial on back-left (also serves as diegetic gas pressure handle!)
  const gaugeGroup = new THREE.Group();
  gaugeGroup.position.set(-radius * 0.72, topY + 0.36, -radius * 0.55);
  const stem = new THREE.Mesh(new THREE.CylinderGeometry(0.07, 0.07, 0.45, 16), mats.stainless);
  stem.position.y = -0.18;
  gaugeGroup.add(stem);
  const dialBody = new THREE.Mesh(new THREE.CylinderGeometry(0.32, 0.32, 0.16, 32), mats.stainless);
  dialBody.rotation.x = Math.PI / 2;
  dialBody.position.y = 0.14;
  gaugeGroup.add(dialBody);
  const dialFace = new THREE.Mesh(
    new THREE.CircleGeometry(0.26, 32),
    new THREE.MeshBasicMaterial({ color: 0x0f172a })
  );
  dialFace.position.set(0, 0.14, 0.084);
  gaugeGroup.add(dialFace);
  const needle = new THREE.Mesh(
    new THREE.BoxGeometry(0.025, 0.20, 0.01),
    new THREE.MeshBasicMaterial({ color: 0x38bdf8 })
  );
  needle.position.set(0, 0.18, 0.09);
  gaugeGroup.add(needle);
  ch.add(gaugeGroup);

  ch.add(upperGroup);

  ch.userData.gaugeGroup = gaugeGroup;
  ch.userData.gaugeNeedle = needle;
  ch.userData.setMode = (mode) => {
    if (mode === 'open') {
      upperGroup.visible = false;
      glassWall.visible = false;
    } else if (mode === 'cutaway') {
      upperGroup.visible = true;
      glassWall.visible = false;
      if (frontBezelGroup) frontBezelGroup.visible = false;
    } else {
      // 'full'
      upperGroup.visible = true;
      glassWall.visible = true;
      if (frontBezelGroup) frontBezelGroup.visible = true;
    }
  };

  return ch;
}

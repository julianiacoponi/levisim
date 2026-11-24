/**
 * Photorealistic PBR materials, procedural M6 optical breadboard textures,
 * and darkroom quantum optics lab HDR environment maps for levisim web labs.
 */
import * as THREE from 'three';

/**
 * Generates procedural color, roughness, and bump CanvasTextures for a
 * black-anodized (or brushed stainless) optical breadboard with M6 tapped holes.
 */
export function makeBreadboardTextures(variant = 'anodized', repeatX = 3.6, repeatY = 2.2) {
  const size = 1024;
  const step = 64;
  const cColor = document.createElement('canvas');
  const cRough = document.createElement('canvas');
  const cBump = document.createElement('canvas');
  cColor.width = cColor.height = size;
  cRough.width = cRough.height = size;
  cBump.width = cBump.height = size;

  const gC = cColor.getContext('2d');
  const gR = cRough.getContext('2d');
  const gB = cBump.getContext('2d');

  const isSilver = variant === 'silver';
  const baseHex = isSilver ? '#b8c1cc' : '#141820';
  const baseRough = isSilver ? 84 : 92;

  gC.fillStyle = baseHex;
  gC.fillRect(0, 0, size, size);
  gR.fillStyle = `rgb(${baseRough},${baseRough},${baseRough})`;
  gR.fillRect(0, 0, size, size);
  gB.fillStyle = '#dadada';
  gB.fillRect(0, 0, size, size);

  // Subtle anisotropic brushed-metal grain along the optical table length
  for (let y = 0; y < size; y++) {
    const grain = (Math.random() - 0.5) * (isSilver ? 14 : 7.5);
    gC.fillStyle = grain > 0
      ? `rgba(255,255,255,${grain / 255})`
      : `rgba(0,0,0,${-grain / 255})`;
    gC.fillRect(0, y, size, 1);
    const rv = Math.max(48, Math.min(165, Math.round(baseRough + grain * 1.5)));
    gR.fillStyle = `rgb(${rv},${rv},${rv})`;
    gR.fillRect(0, y, size, 1);
  }

  // Countersunk M6 tapped holes on regular 25 mm grid
  for (let x = step / 2; x < size; x += step) {
    for (let y = step / 2; y < size; y += step) {
      const bevel = gC.createRadialGradient(x, y, 4.4, x, y, 9.6);
      bevel.addColorStop(0, '#05070a');
      bevel.addColorStop(0.56, isSilver ? '#6d7785' : '#323a47');
      bevel.addColorStop(1, 'rgba(0,0,0,0)');
      gC.fillStyle = bevel;
      gC.beginPath();
      gC.arc(x, y, 9.6, 0, Math.PI * 2);
      gC.fill();

      // Inner dark threaded hole
      gC.fillStyle = '#040507';
      gC.beginPath();
      gC.arc(x, y, 5.1, 0, Math.PI * 2);
      gC.fill();

      // High roughness inside hole
      gR.fillStyle = '#ececec';
      gR.beginPath();
      gR.arc(x, y, 6.2, 0, Math.PI * 2);
      gR.fill();

      // Deep chamfer in bump map
      const bGrad = gB.createRadialGradient(x, y, 3.0, x, y, 9.2);
      bGrad.addColorStop(0, '#080808');
      bGrad.addColorStop(0.65, '#686868');
      bGrad.addColorStop(1, '#dadada');
      gB.fillStyle = bGrad;
      gB.beginPath();
      gB.arc(x, y, 9.2, 0, Math.PI * 2);
      gB.fill();
    }
  }

  const makeTex = (canvas, isSRGB) => {
    const tex = new THREE.CanvasTexture(canvas);
    tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
    tex.repeat.set(repeatX, repeatY);
    tex.anisotropy = 16;
    if (isSRGB) tex.colorSpace = THREE.SRGBColorSpace;
    return tex;
  };

  return {
    map: makeTex(cColor, true),
    roughnessMap: makeTex(cRough, false),
    bumpMap: makeTex(cBump, false),
  };
}

/**
 * Generates a fine vertical knurling bump texture for brass micrometers and lens rings.
 */
export function makeKnurlBumpTexture() {
  const w = 256, h = 64;
  const c = document.createElement('canvas');
  c.width = w;
  c.height = h;
  const ctx = c.getContext('2d');
  for (let x = 0; x < w; x++) {
    const v = Math.round(128 + 110 * Math.sin((x / w) * Math.PI * 2 * 36));
    ctx.fillStyle = `rgb(${v},${v},${v})`;
    ctx.fillRect(x, 0, 1, h);
  }
  const tex = new THREE.CanvasTexture(c);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  return tex;
}

/**
 * Builds a procedural HDR softbox environment map for the Darkroom Quantum Optics Lab
 * so glass prisms, fused silica mirrors, stainless steel, and brass show rich reflections.
 */
export function buildDarkroomEnvironment(renderer) {
  const envScene = new THREE.Scene();
  envScene.background = new THREE.Color(0x080c14);

  const addPanel = (w, h, pos, rot, color, intensity) => {
    const mat = new THREE.MeshBasicMaterial({
      color: new THREE.Color(color).multiplyScalar(intensity),
      side: THREE.DoubleSide,
    });
    const mesh = new THREE.Mesh(new THREE.PlaneGeometry(w, h), mat);
    mesh.position.set(...pos);
    mesh.rotation.set(...rot);
    envScene.add(mesh);
  };

  // Overhead lab ceiling softboxes + cool cyan and subtle ruby fill accents
  addPanel(18, 4.0, [-2, 13, 2], [Math.PI / 2, 0, 0], 0xdceaff, 2.0);
  addPanel(14, 2.5, [6, 12, -4], [Math.PI / 2, 0, 0], 0xbfe3ff, 1.4);
  addPanel(3.5, 11, [-14, 6, 6], [0, Math.PI / 3, 0], 0x7dd3fc, 1.85);
  addPanel(3.5, 10, [14, 5, 5], [0, -Math.PI / 3, 0], 0xfecdd3, 1.35);
  addPanel(9.0, 2.5, [1, 3.5, -14], [0, 0, 0], 0x38bdf8, 1.35);
  addPanel(8.0, 1.6, [0, 2.2, 11], [0, Math.PI, 0], 0xff2a44, 0.85);

  const pmrem = new THREE.PMREMGenerator(renderer);
  const envTex = pmrem.fromScene(envScene, 0.03).texture;
  pmrem.dispose();
  return envTex;
}

/**
 * Creates the full shared library of physically-based optical & lab materials.
 */
export function createOpticsMaterials(repeatX = 3.6, repeatY = 2.2) {
  const bbTex = makeBreadboardTextures('anodized', repeatX, repeatY);
  const knurlBump = makeKnurlBumpTexture();

  return {
    breadboard: new THREE.MeshStandardMaterial({
      map: bbTex.map,
      roughnessMap: bbTex.roughnessMap,
      bumpMap: bbTex.bumpMap,
      bumpScale: 0.025,
      metalness: 0.74,
      roughness: 0.44,
    }),
    tableSide: new THREE.MeshStandardMaterial({
      color: 0x161b23,
      metalness: 0.84,
      roughness: 0.34,
    }),
    anodized: new THREE.MeshStandardMaterial({
      color: 0x191d25,
      metalness: 0.68,
      roughness: 0.34,
    }),
    anodizedRing: new THREE.MeshStandardMaterial({
      color: 0x28303d,
      metalness: 0.78,
      roughness: 0.26,
      bumpMap: knurlBump,
      bumpScale: 0.008,
    }),
    anodizedRed: new THREE.MeshStandardMaterial({
      color: 0x7f1d28,
      metalness: 0.76,
      roughness: 0.28,
    }),
    stainless: new THREE.MeshStandardMaterial({
      color: 0xd9e2ec,
      metalness: 0.94,
      roughness: 0.16,
    }),
    titanium: new THREE.MeshStandardMaterial({
      color: 0x8e98a6,
      metalness: 0.88,
      roughness: 0.24,
    }),
    brass: new THREE.MeshStandardMaterial({
      color: 0xd7a852,
      metalness: 0.93,
      roughness: 0.22,
      bumpMap: knurlBump,
      bumpScale: 0.012,
    }),
    copper: new THREE.MeshStandardMaterial({
      color: 0xc97b53,
      metalness: 0.91,
      roughness: 0.24,
    }),
    cable: new THREE.MeshStandardMaterial({
      color: 0x12151a,
      metalness: 0.12,
      roughness: 0.68,
    }),
    ampChassis: new THREE.MeshStandardMaterial({
      color: 0x202733,
      metalness: 0.55,
      roughness: 0.36,
    }),
    electrode: new THREE.MeshStandardMaterial({
      color: 0xe2e8f0,
      metalness: 0.96,
      roughness: 0.12,
    }),

    // Photorealistic Fused Silica (n = 1.45 at 1064 nm)
    fusedSilica: new THREE.MeshPhysicalMaterial({
      color: 0xf4f9ff,
      metalness: 0.0,
      roughness: 0.022,
      transmission: 0.96,
      ior: 1.45,
      thickness: 1.25,
      attenuationColor: new THREE.Color(0xcce6ff),
      attenuationDistance: 4.0,
      clearcoat: 1.0,
      clearcoatRoughness: 0.02,
      iridescence: 0.65,
      iridescenceIOR: 1.38,
      iridescenceThicknessRange: [180, 440],
    }),

    // Curved cavity HR dielectric mirror coating
    hrCoating: new THREE.MeshPhysicalMaterial({
      color: 0x9ec0ff,
      metalness: 0.86,
      roughness: 0.04,
      clearcoat: 1.0,
      iridescence: 1.0,
      iridescenceIOR: 1.45,
      iridescenceThicknessRange: [260, 520],
    }),

    // Photorealistic N-BK7 Beam Splitter Cube glass (n = 1.517)
    pbsGlass: new THREE.MeshPhysicalMaterial({
      color: 0xffffff,
      metalness: 0.0,
      roughness: 0.02,
      transmission: 0.96,
      ior: 1.517,
      thickness: 1.15,
      attenuationColor: new THREE.Color(0xdcf3ff),
      attenuationDistance: 5.5,
      clearcoat: 1.0,
      clearcoatRoughness: 0.015,
      iridescence: 0.52,
      iridescenceIOR: 1.35,
      iridescenceThicknessRange: [200, 390],
    }),

    // Internal diagonal dielectric coating inside PBS / 50:50 cube
    pbsDiagonal: new THREE.MeshPhysicalMaterial({
      color: 0x67e8f9,
      metalness: 0.25,
      roughness: 0.04,
      transparent: true,
      opacity: 0.28,
      iridescence: 1.0,
      iridescenceIOR: 1.4,
      side: THREE.DoubleSide,
    }),

    // AR-coated waveplate / aspheric lens glass
    arLensViolet: new THREE.MeshPhysicalMaterial({
      color: 0xa78bfa,
      metalness: 0.05,
      roughness: 0.03,
      transmission: 0.86,
      ior: 1.52,
      thickness: 0.35,
      iridescence: 1.0,
      iridescenceIOR: 1.38,
      iridescenceThicknessRange: [280, 480],
    }),
    arLensEmerald: new THREE.MeshPhysicalMaterial({
      color: 0x34d399,
      metalness: 0.05,
      roughness: 0.03,
      transmission: 0.86,
      ior: 1.52,
      thickness: 0.35,
      iridescence: 0.95,
      iridescenceIOR: 1.38,
      iridescenceThicknessRange: [220, 400],
    }),

    // Crystal sapphire vacuum viewport glass (Fresnel clearcoat so internal beams stay vivid)
    sapphireViewport: new THREE.MeshPhysicalMaterial({
      color: 0xdbeafe,
      roughness: 0.035,
      metalness: 0.08,
      transparent: true,
      opacity: 0.15,
      depthWrite: false,
      clearcoat: 1.0,
      clearcoatRoughness: 0.02,
      iridescence: 0.85,
      iridescenceIOR: 1.76,
      iridescenceThicknessRange: [220, 460],
      side: THREE.DoubleSide,
    }),

    // Subtle glowing hover highlight material for diegetic optics
    hoverHighlight: new THREE.MeshBasicMaterial({
      color: 0x38bdf8,
      transparent: true,
      opacity: 0.38,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      side: THREE.DoubleSide,
    }),
  };
}

/**
 * Adds a moody, dimly lit university quantum optics darkroom laboratory backdrop
 * (dark acoustic panels, equipment racks, faint status LEDs) behind the optical table.
 */
export function buildDarkroomBackdrop(scene) {
  const group = new THREE.Group();
  const wallMat = new THREE.MeshStandardMaterial({ color: 0x090d14, roughness: 0.85, metalness: 0.25 });
  const panelMat = new THREE.MeshStandardMaterial({ color: 0x0f141e, roughness: 0.88, metalness: 0.2 });
  const rackMat = new THREE.MeshStandardMaterial({ color: 0x111620, roughness: 0.65, metalness: 0.5 });

  // Back wall + side acoustic panels
  const backWall = new THREE.Mesh(new THREE.PlaneGeometry(54, 26), wallMat);
  backWall.position.set(0, 7, -12.5);
  group.add(backWall);

  for (let i = -4; i <= 4; i++) {
    const panel = new THREE.Mesh(new THREE.BoxGeometry(2.7, 5.8, 0.25), panelMat);
    panel.position.set(i * 3.1, 5.2, -12.2);
    group.add(panel);
  }

  // Background instrument rack silhouette (far left-center)
  const rack = new THREE.Mesh(new THREE.BoxGeometry(2.6, 8.2, 1.4), rackMat);
  rack.position.set(-4.2, 3.6, -10.8);
  group.add(rack);

  // Subtle instrument rack status LEDs
  const leds = [
    [-4.9, 6.2, -10.05, 0x22c55e],
    [-4.6, 6.2, -10.05, 0x38bdf8],
    [-4.9, 4.9, -10.05, 0x22c55e],
    [-3.7, 4.9, -10.05, 0xf59e0b],
    [-4.9, 3.8, -10.05, 0x38bdf8],
    [6.6, 5.1, -11.9, 0x22c55e],
  ];
  for (const [x, y, z, col] of leds) {
    const led = new THREE.Mesh(
      new THREE.SphereGeometry(0.055, 8, 8),
      new THREE.MeshBasicMaterial({ color: col })
    );
    led.position.set(x, y, z);
    group.add(led);
  }

  scene.add(group);
  return group;
}

/**
 * Converts an optical wavelength in nm (e.g. 532..1550 nm) to a vivid display RGB color.
 * Near-IR trapping lasers (780-1550 nm, default 1064 nm) are rendered in luminous lab ruby-crimson.
 */
export function wavelengthToLaserColors(lambdaNm) {
  if (lambdaNm < 560) {
    return { outer: new THREE.Color(0x0ea84b), core: new THREE.Color(0x86efac), point: 0x22c55e };
  }
  if (lambdaNm < 610) {
    return { outer: new THREE.Color(0xd97706), core: new THREE.Color(0xfde68a), point: 0xf59e0b };
  }
  if (lambdaNm < 780) {
    return { outer: new THREE.Color(0xdc2626), core: new THREE.Color(0xff7a88), point: 0xef4444 };
  }
  if (lambdaNm < 1250) {
    // 1064 nm Nd:YAG — canonical crimson ruby visualization
    return { outer: new THREE.Color(0xd90b24), core: new THREE.Color(0xff7082), point: 0xff1a34 };
  }
  // 1550 nm telecom IR — deep magenta-crimson visualization
  return { outer: new THREE.Color(0xb80d48), core: new THREE.Color(0xff6b9d), point: 0xe11d48 };
}

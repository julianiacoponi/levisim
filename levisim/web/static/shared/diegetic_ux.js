/**
 * Hybrid Diegetic 3D Optics Interaction Controller + Smooth Camera Bookmarks.
 *
 * Lets the user hover any registered bench optic (waveplates, laser power knob,
 * electrode ring, cavity mirror piezo, standing-wave stage, LO mirror, monitor)
 * to see a subtle floating physics tooltip and drag directly in 3D to tune
 * parameters — seamlessly pausing OrbitControls during optic manipulation.
 */
import * as THREE from 'three';

export class DiegeticController {
  constructor(camera, domElement, orbitControls, tooltipEl) {
    this.camera = camera;
    this.dom = domElement;
    this.controls = orbitControls;
    this.tooltip = tooltipEl;
    this.raycaster = new THREE.Raycaster();
    this.pointer = new THREE.Vector2();
    this.items = [];
    this.hovered = null;
    this.dragging = null;
    this.dragStartX = 0;
    this.dragStartY = 0;
    this.dragMoved = false;

    // Smooth camera tween state
    this.camAnim = null;

    this._onPointerMove = this._onPointerMove.bind(this);
    this._onPointerDown = this._onPointerDown.bind(this);
    this._onPointerUp = this._onPointerUp.bind(this);

    domElement.addEventListener('pointermove', this._onPointerMove);
    domElement.addEventListener('pointerdown', this._onPointerDown);
    window.addEventListener('pointerup', this._onPointerUp);
  }

  /**
   * Registers a 3D object (or Group) as an interactive diegetic optical control.
   * config: { object, title, hint, getValue, onDrag(dx, dy), onClick() }
   */
  register(config) {
    const meshes = [];
    config.object.traverse(child => {
      if (child.isMesh) {
        child.userData.__diegetic = config;
        meshes.push(child);
      }
    });
    // Optional glowing selection ring attached to object center
    if (config.highlightRing) {
      config.highlightRing.visible = false;
    }
    this.items.push({ ...config, meshes });
  }

  /**
   * Smoothly animates camera & OrbitControls target to a preset bookmark.
   */
  flyTo(pos, target, durationMs = 700) {
    this.camAnim = {
      startPos: this.camera.position.clone(),
      endPos: new THREE.Vector3(...pos),
      startTarget: this.controls.target.clone(),
      endTarget: new THREE.Vector3(...target),
      t0: performance.now(),
      duration: durationMs,
    };
  }

  update() {
    if (!this.camAnim) return;
    const elapsed = performance.now() - this.camAnim.t0;
    const u = Math.min(1, elapsed / this.camAnim.duration);
    // Cubic ease-out
    const k = 1 - Math.pow(1 - u, 3);
    this.camera.position.lerpVectors(this.camAnim.startPos, this.camAnim.endPos, k);
    this.controls.target.lerpVectors(this.camAnim.startTarget, this.camAnim.endTarget, k);
    this.controls.update();
    if (u >= 1) this.camAnim = null;
  }

  _updatePointer(ev) {
    const rect = this.dom.getBoundingClientRect();
    this.pointer.x = ((ev.clientX - rect.left) / rect.width) * 2 - 1;
    this.pointer.y = -((ev.clientY - rect.top) / rect.height) * 2 + 1;
  }

  _hitTest() {
    this.raycaster.setFromCamera(this.pointer, this.camera);
    const allMeshes = [];
    for (const item of this.items) {
      if (item.object.visible) allMeshes.push(...item.meshes);
    }
    const hits = this.raycaster.intersectObjects(allMeshes, false);
    if (!hits.length) return null;
    return hits[0].object.userData.__diegetic || null;
  }

  _showTooltip(item, clientX, clientY) {
    if (!this.tooltip) return;
    const valStr = item.getValue ? item.getValue() : '';
    this.tooltip.innerHTML =
      `<div class="dt-title">${item.title}</div>` +
      (valStr ? `<div class="dt-val">${valStr}</div>` : '') +
      `<div class="dt-hint">${item.hint}</div>`;
    this.tooltip.style.left = `${Math.min(window.innerWidth - 260, clientX + 16)}px`;
    this.tooltip.style.top = `${Math.max(12, clientY - 14)}px`;
    this.tooltip.classList.add('visible');
  }

  _hideTooltip() {
    if (this.tooltip) this.tooltip.classList.remove('visible');
  }

  _setHighlight(activeItem) {
    for (const item of this.items) {
      if (item.highlightRing) {
        item.highlightRing.visible = (item === activeItem);
      }
    }
  }

  _onPointerMove(ev) {
    if (this.dragging) {
      const dx = ev.clientX - this.dragStartX;
      const dy = ev.clientY - this.dragStartY;
      if (Math.hypot(dx, dy) > 2) this.dragMoved = true;
      this.dragStartX = ev.clientX;
      this.dragStartY = ev.clientY;
      if (this.dragging.onDrag) {
        this.dragging.onDrag(dx, dy);
      }
      this._showTooltip(this.dragging, ev.clientX, ev.clientY);
      return;
    }

    this._updatePointer(ev);
    const hit = this._hitTest();
    this.hovered = hit;
    this._setHighlight(hit);
    if (hit) {
      this.dom.style.cursor = hit.onDrag ? 'ew-resize' : 'pointer';
      this._showTooltip(hit, ev.clientX, ev.clientY);
    } else {
      this.dom.style.cursor = 'grab';
      this._hideTooltip();
    }
  }

  _onPointerDown(ev) {
    if (ev.button !== 0) return;
    this.camAnim = null;
    this._updatePointer(ev);
    const hit = this._hitTest();
    if (!hit) return;

    this.dragging = hit;
    this.dragStartX = ev.clientX;
    this.dragStartY = ev.clientY;
    this.dragMoved = false;
    this.controls.enabled = false;
    this.dom.style.cursor = hit.onDrag ? 'grabbing' : 'pointer';
  }

  _onPointerUp() {
    if (!this.dragging) return;
    const item = this.dragging;
    const wasClick = !this.dragMoved;
    this.dragging = null;
    this.controls.enabled = true;
    this.dom.style.cursor = this.hovered ? (this.hovered.onDrag ? 'ew-resize' : 'pointer') : 'grab';
    if (wasClick && item.onClick) {
      item.onClick();
    }
  }
}

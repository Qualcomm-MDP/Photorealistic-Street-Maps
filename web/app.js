import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

const $ = (id) => document.getElementById(id);
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const esc = (s) =>
  String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;

const toast = (() => {
  const el = $("toast");
  let t;
  const canPopover = typeof el.showPopover === "function";
  return (msg) => {
    el.textContent = msg;
    if (canPopover) {
      // Re-show so it's on top of any dialog opened since last time.
      if (el.matches(":popover-open")) el.hidePopover();
      el.showPopover();
    }
    requestAnimationFrame(() => el.classList.add("show"));
    clearTimeout(t);
    t = setTimeout(() => {
      el.classList.remove("show");
      if (canPopover) setTimeout(() => el.matches(":popover-open") && el.hidePopover(), 250);
    }, 3000);
  };
})();

// ---------- API ----------
async function api(path, options = {}) {
  const res = await fetch(path, { headers: { "Content-Type": "application/json" }, ...options });
  let body = null;
  try {
    body = await res.json();
  } catch {
    /* empty or non-JSON body */
  }
  if (!res.ok) throw new Error(errorMessage(body) || `Request failed (HTTP ${res.status}).`);
  return body;
}

function errorMessage(body) {
  const d = body?.detail;
  if (!d) return null;
  if (typeof d === "string") return d;
  if (Array.isArray(d)) return d.map((e) => String(e.msg || "").replace(/^Value error, /, "")).join(" ");
  return null;
}

const state = {
  stages: [],
  system: { cuda: false, max_side_m: 2500 },
  jobs: [],
  current: null, // job shown in the progress timeline
  shown: null, // job whose model and photos are loaded
  modelJobId: null, // job whose model has finished loading into the scene
  imagesJobId: null, // job whose photos are in state.images
  modelWanted: null, // job whose model was last requested
  imagesWanted: null, // job whose photos were last requested
  settings: null, // from /api/settings: token status and defaults for new jobs
  modalJobId: null, // job open in the job details dialog
  placeName: null,
  images: [],
  showCleaned: true,
  socket: null,
  reconnectDelay: 1000,
};

// ---------- Bounding box ----------
const DEFAULT_BOX = { west: -83.71978, south: 42.29025, east: -83.71205, north: 42.29422 };
let box = { ...DEFAULT_BOX };
const ids = ["west", "south", "east", "north"];
const inputs = Object.fromEntries(ids.map((k) => [k, $(k)]));

const M_PER_DEG_LAT = 110574;
const mPerDegLon = (lat) => 111320 * Math.cos((lat * Math.PI) / 180);
function boxSize(b) {
  const lat = (b.south + b.north) / 2;
  return { w: (b.east - b.west) * mPerDegLon(lat), h: (b.north - b.south) * M_PER_DEG_LAT };
}
function boxAround(lat, lon, wM = 640, hM = 440) {
  const dLon = wM / 2 / mPerDegLon(lat);
  const dLat = hM / 2 / M_PER_DEG_LAT;
  return { west: lon - dLon, east: lon + dLon, south: lat - dLat, north: lat + dLat };
}
const fmtBox = (b) => [b.west, b.south, b.east, b.north].map((v) => v.toFixed(4)).join(", ");

function sizeHtml() {
  const { w, h } = boxSize(box);
  const tooBig = Math.max(w, h) > state.system.max_side_m;
  return (
    `<span><b>${Math.round(w)} × ${Math.round(h)} m</b></span><span>${((w * h) / 1e6).toFixed(2)} km²</span>` +
    (tooBig ? `<span style="color:var(--bad)">Max ${state.system.max_side_m} m per side</span>` : "")
  );
}
function renderStats() {
  $("stats").innerHTML = sizeHtml();
  $("modalStats").innerHTML = sizeHtml();
  renderNaipNote();
}

// NAIP covers the continental US. This rough box can't tell a border town
// from its neighbour across the border; it's there to catch obvious misses.
const NAIP_COVERAGE = { west: -124.9, south: 24.4, east: -66.9, north: 49.5 };
function renderNaipNote() {
  const lat = (box.south + box.north) / 2;
  const lon = (box.west + box.east) / 2;
  const c = NAIP_COVERAGE;
  const inside = lat >= c.south && lat <= c.north && lon >= c.west && lon <= c.east;
  $("naipNote").hidden = !$("naip").checked || inside;
}
function syncInputs() {
  ids.forEach((k) => (inputs[k].value = box[k].toFixed(5)));
}

// ---------- Map (Leaflet + OpenStreetMap tiles) ----------
const map = L.map("map", { zoomControl: false });
L.control.zoom({ position: "bottomright" }).addTo(map);
L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
  maxZoom: 19,
  attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
}).addTo(map);

const toBounds = (b) => L.latLngBounds([b.south, b.west], [b.north, b.east]);
const fromBounds = (lb) => ({ west: lb.getWest(), south: lb.getSouth(), east: lb.getEast(), north: lb.getNorth() });
const rectStyle = (dashed = false) => ({
  color: css("--accent"),
  weight: 2.5,
  fillColor: css("--accent"),
  fillOpacity: 0.3,
  dashArray: dashed ? "6 5" : null,
  interactive: false,
});
const rect = L.rectangle(toBounds(box), rectStyle()).addTo(map);
map.fitBounds(toBounds(box), { padding: [28, 28] });

function setBox(b, { fit = false } = {}) {
  box = { ...b };
  rect.setBounds(toBounds(box));
  rect.setStyle(rectStyle());
  syncInputs();
  renderStats();
  if (fit) map.fitBounds(toBounds(box), { padding: [28, 28] });
}

// Draw mode: map panning is paused so a drag draws the selection instead.
const mapEl = $("map");
const drawBtn = $("drawBtn");
let drawing = false;
let dragStart = null;
function setDrawMode(on) {
  drawing = on;
  drawBtn.setAttribute("aria-pressed", String(on));
  drawBtn.textContent = on ? "Drag to draw…" : "Draw area";
  $("mapBox").classList.toggle("drawing", on);
  on ? map.dragging.disable() : map.dragging.enable();
  $("mapHint").textContent = on ? "Drag a rectangle over the buildings you want" : "Press Draw area, then drag on the map";
}
drawBtn.addEventListener("click", () => setDrawMode(!drawing));
document.addEventListener("keydown", (e) => {
  if (e.key !== "Escape" || !drawing) return;
  dragStart = null;
  setBox(box);
  setDrawMode(false);
});
mapEl.addEventListener("pointercancel", () => {
  if (!dragStart) return;
  dragStart = null;
  setBox(box);
});
mapEl.addEventListener("pointerdown", (e) => {
  if (!drawing || e.button !== 0 || e.target.closest(".leaflet-control")) return;
  mapEl.setPointerCapture(e.pointerId);
  dragStart = map.mouseEventToLatLng(e);
  e.preventDefault();
  e.stopPropagation();
});
mapEl.addEventListener("pointermove", (e) => {
  if (!dragStart) return;
  rect.setBounds(L.latLngBounds(dragStart, map.mouseEventToLatLng(e)));
  rect.setStyle(rectStyle(true));
});
mapEl.addEventListener("pointerup", (e) => {
  if (!dragStart) return;
  const a = map.latLngToContainerPoint(dragStart);
  const b = map.latLngToContainerPoint(map.mouseEventToLatLng(e));
  const drawn = L.latLngBounds(dragStart, map.mouseEventToLatLng(e));
  dragStart = null;
  if (Math.abs(a.x - b.x) < 10 || Math.abs(a.y - b.y) < 10) {
    setBox(box); // too small: keep the previous selection
    return;
  }
  state.placeName = null;
  setBox(fromBounds(drawn));
  setDrawMode(false);
});

ids.forEach((k) =>
  inputs[k].addEventListener("change", () => {
    const b = { ...box };
    const v = parseFloat(inputs[k].value);
    if (!Number.isNaN(v)) b[k] = v;
    if (b.west > b.east) [b.west, b.east] = [b.east, b.west];
    if (b.south > b.north) [b.south, b.north] = [b.north, b.south];
    state.placeName = null;
    setBox(b, { fit: true });
  }),
);

// ---------- Address search ----------
// Typing shows suggestions from /api/suggest (Photon, built for autocomplete).
// The Search button runs a full search through /api/geocode (Nominatim).
const addr = $("address");
const suggest = $("suggest");
let results = [];
let active = 0;
let suggestTimer = null;
let suggestCtrl = null;

function cancelSuggestions() {
  clearTimeout(suggestTimer);
  suggestCtrl?.abort();
  suggestCtrl = null;
}

function closeSuggest() {
  suggest.hidden = true;
  addr.setAttribute("aria-expanded", "false");
  addr.removeAttribute("aria-activedescendant");
}
function suggestMessage(msg) {
  results = [];
  suggest.innerHTML = `<li class="empty">${esc(msg)}</li>`;
  suggest.hidden = false;
  addr.setAttribute("aria-expanded", "true");
  addr.removeAttribute("aria-activedescendant");
}
function renderResults(list, query, emptyMessage = `No places found for "${query}".`) {
  results = list;
  active = 0;
  if (!list.length) return suggestMessage(emptyMessage);
  suggest.innerHTML = list
    .map(
      (p, i) =>
        `<li role="option" id="suggest-${i}" aria-selected="false" data-i="${i}">${esc(p.name)}<span>${esc(p.address)}</span></li>`,
    )
    .join("");
  openSuggest();
}
function openSuggest() {
  suggest.hidden = false;
  addr.setAttribute("aria-expanded", "true");
  highlight();
}
// Focus stays in the input, so aria-activedescendant tells screen readers
// which suggestion Enter will pick.
function highlight() {
  suggest.querySelectorAll("li[data-i]").forEach((li, i) => li.setAttribute("aria-selected", String(i === active)));
  if (results.length) addr.setAttribute("aria-activedescendant", `suggest-${active}`);
}
function choosePlace(p) {
  addr.value = p.address;
  state.placeName = p.name;
  closeSuggest();
  // Use the place's own extent when it's a sensible size (a block, a campus),
  // otherwise a block-sized box around its centre (an address, a whole city).
  const { w, h } = boxSize(p.bbox);
  const fits = w >= 150 && h >= 150 && Math.max(w, h) <= state.system.max_side_m;
  setBox(fits ? p.bbox : boxAround(p.lat, p.lon), { fit: true });
}

async function fetchSuggestions(q) {
  suggestCtrl = new AbortController();
  const c = map.getCenter();
  const url = `/api/suggest?q=${encodeURIComponent(q)}&lat=${c.lat.toFixed(4)}&lon=${c.lng.toFixed(4)}`;
  try {
    const data = await api(url, { signal: suggestCtrl.signal });
    if (addr.value.trim() !== q) return; // the user kept typing
    renderResults(data.results, q, "No suggestions. Press Search for a full search.");
  } catch (err) {
    if (err.name === "AbortError") return;
    suggestMessage("Suggestions are unavailable. Press Search to look it up.");
  }
}

$("searchForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  cancelSuggestions();
  const q = addr.value.trim();
  if (q.length < 3) return suggestMessage("Type at least 3 characters.");
  const btn = $("searchBtn");
  btn.disabled = true;
  btn.textContent = "Searching…";
  try {
    const data = await api(`/api/geocode?q=${encodeURIComponent(q)}`);
    renderResults(data.results, q);
  } catch (err) {
    suggestMessage(err.message);
  } finally {
    btn.disabled = false;
    btn.textContent = "Search";
  }
});
addr.addEventListener("input", () => {
  cancelSuggestions();
  const q = addr.value.trim();
  if (!q) return closeSuggest();
  if (q.length < 3) return suggestMessage("Keep typing to see suggestions…");
  // Show feedback right away; keep earlier suggestions visible until new ones arrive.
  if (!results.length) suggestMessage("Searching…");
  suggestTimer = setTimeout(() => fetchSuggestions(q), 200);
});
addr.addEventListener("focus", () => {
  if (addr.value.trim().length >= 3 && results.length) openSuggest();
});
addr.addEventListener("keydown", (e) => {
  if (suggest.hidden || !results.length) return;
  if (e.key === "Enter") {
    e.preventDefault(); // pick the highlighted suggestion instead of running a full search
    cancelSuggestions();
    choosePlace(results[active]);
  } else if (e.key === "ArrowDown") (active = (active + 1) % results.length), highlight(), e.preventDefault();
  else if (e.key === "ArrowUp") (active = (active - 1 + results.length) % results.length), highlight(), e.preventDefault();
  else if (e.key === "Escape") {
    e.preventDefault(); // close only the dropdown, not the larger-map dialog around it
    closeSuggest();
  }
});
suggest.addEventListener("mousedown", (e) => {
  const li = e.target.closest("li[data-i]");
  if (!li) return;
  e.preventDefault();
  cancelSuggestions();
  choosePlace(results[+li.dataset.i]);
});
suggest.addEventListener("mousemove", (e) => {
  const li = e.target.closest("li[data-i]");
  if (li && +li.dataset.i !== active) (active = +li.dataset.i), highlight();
});
document.addEventListener("click", (e) => {
  if (!e.target.closest("#searchForm")) closeSuggest();
});

// ---------- Modals: move the live element into the dialog and back ----------
function modal(dialogId, bodyId, elId, placeholderId, onToggle) {
  const dlg = $(dialogId);
  const body = $(bodyId);
  const el = $(elId);
  const ph = $(placeholderId);
  const home = el.parentNode;
  const next = el.nextSibling;
  dlg.addEventListener("click", (e) => {
    if (e.target === dlg || e.target.closest("[data-close]")) dlg.close();
  });
  dlg.addEventListener("close", () => {
    home.insertBefore(el, next);
    ph.hidden = true;
    ph.style.height = "";
    onToggle(false);
  });
  return () => {
    // The placeholder takes the element's exact height, so the page behind
    // the dialog doesn't shift while it's away.
    ph.style.height = `${el.offsetHeight}px`;
    body.appendChild(el);
    ph.hidden = false;
    dlg.showModal();
    onToggle(true);
  };
}
$("openMap").addEventListener(
  "click",
  modal("mapModal", "mapModalBody", "mapTool", "mapPlaceholder", () =>
    requestAnimationFrame(() => {
      map.invalidateSize();
      map.fitBounds(toBounds(box), { padding: [40, 40] });
    }),
  ),
);
$("openScene").addEventListener(
  "click",
  modal("sceneModal", "sceneModalBody", "scene", "stagePlaceholder", (open) => {
    $("stageTools").hidden = open;
    $("stageHint").hidden = open;
  }),
);

// ---------- Options ----------
function segmented(id, onChange) {
  const root = $(id);
  root.addEventListener("click", (e) => {
    const btn = e.target.closest("button");
    if (!btn || btn.disabled) return;
    root.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", String(b === btn)));
    onChange(btn.dataset.v);
  });
}
let device = "auto";
segmented("device", (v) => (device = v));

// Aerial photos are draped on the terrain, so turning them on turns Terrain on
// and locks it; turning them off restores whatever Terrain was set to before.
let terrainWanted = false;
$("terrain").addEventListener("change", () => (terrainWanted = $("terrain").checked));
$("naip").addEventListener("change", () => {
  const naip = $("naip").checked;
  $("terrain").checked = naip || terrainWanted;
  $("terrain").disabled = naip;
  renderNaipNote();
});
for (const id of ["beforeAfter", "jobBeforeAfter"]) {
  segmented(id, (v) => {
    state.showCleaned = v === "after";
    renderGallery();
  });
}

// ---------- 3D preview ----------
const sceneEl = $("scene");
// three.js lights a matte surface to colour × intensity / π, so these are
// multiplied by π: together they light a facing wall to about its own colour,
// which keeps the facade photos at their real brightness.
const AMBIENT_LIGHT = 0.7 * Math.PI;
const CORNER_LIGHT = 0.25 * Math.PI;
// Without WebGL (turned off, blocked, or no GPU driver) there's no 3D preview,
// but everything else (generating, logs, photos, downloads) still works.
let renderer = null;
try {
  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(2, devicePixelRatio || 1));
  sceneEl.appendChild(renderer.domElement);
} catch (err) {
  console.warn("3D preview unavailable:", err);
  renderer = null;
  const note = document.createElement("div");
  note.className = "stage-fallback";
  note.textContent = "The 3D preview needs WebGL, which this browser has turned off. You can still download the .glb.";
  sceneEl.appendChild(note);
}
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(40, 1, 0.5, 20000);
// Even light from all sides: a soft overall light plus four gentle lights from
// the diagonal corners above the model, so every wall gets about the same
// light and the facade photos show close to their real brightness, while
// edges between faces still read.
scene.add(new THREE.AmbientLight(0xffffff, AMBIENT_LIGHT));
for (const angle of [45, 135, 225, 315]) {
  const light = new THREE.DirectionalLight(0xffffff, CORNER_LIGHT);
  const a = THREE.MathUtils.degToRad(angle);
  light.position.set(Math.cos(a), 0.7, Math.sin(a));
  scene.add(light);
}
const controls = new OrbitControls(camera, renderer?.domElement ?? sceneEl);
controls.enableDamping = true;
controls.autoRotate = !reduceMotion;
controls.autoRotateSpeed = 0.6;
controls.addEventListener("start", () => (controls.autoRotate = false));
camera.position.set(400, 300, 400);
const loader = new GLTFLoader();
let model = null;
let homeView = null;

function applySceneTheme() {
  scene.background = new THREE.Color(css("--stage"));
}
function resizeScene() {
  const r = sceneEl.getBoundingClientRect();
  if (!renderer || !r.width || !r.height) return;
  renderer.setSize(r.width, r.height, false);
  camera.aspect = r.width / r.height;
  camera.updateProjectionMatrix();
}
new ResizeObserver(resizeScene).observe(sceneEl);
renderer?.setAnimationLoop(() => {
  controls.update();
  renderer.render(scene, camera);
});

function disposeModel(obj) {
  obj.traverse((o) => {
    o.geometry?.dispose();
    const mats = Array.isArray(o.material) ? o.material : o.material ? [o.material] : [];
    for (const m of mats) {
      for (const v of Object.values(m)) if (v?.isTexture) v.dispose();
      m.dispose();
    }
  });
}
function frameModel() {
  const bounds = new THREE.Box3().setFromObject(model);
  const center = bounds.getCenter(new THREE.Vector3());
  const radius = bounds.getSize(new THREE.Vector3()).length() / 2 || 100;
  const dist = radius / Math.sin(THREE.MathUtils.degToRad(camera.fov / 2));
  camera.position.copy(center).add(new THREE.Vector3(1, 0.75, 1).normalize().multiplyScalar(dist));
  camera.near = dist / 200;
  camera.far = dist * 20;
  camera.updateProjectionMatrix();
  controls.target.copy(center);
  controls.update();
  homeView = { position: camera.position.clone(), target: center.clone() };
}
function resetView() {
  if (!homeView) return;
  camera.position.copy(homeView.position);
  controls.target.copy(homeView.target);
  controls.autoRotate = !reduceMotion;
}
$("resetView").addEventListener("click", resetView);
$("resetView2").addEventListener("click", resetView);

// ---------- Image capture ----------
// Saves the current view as a PNG at a chosen resolution and focal length.
// The focal length is 35 mm-equivalent: the image is 36 mm wide, so in pixels
// fx = fy = focal / 36 × width, with the principal point at the image centre.
const SENSOR_WIDTH_MM = 36;
const fovDegrees = (pixels, focalPx) => THREE.MathUtils.radToDeg(2 * Math.atan(pixels / (2 * focalPx)));
const focalPixels = (width, focalMm) => (focalMm / SENSOR_WIDTH_MM) * width;

function captureLimit() {
  const gl = renderer.getContext();
  return Math.min(gl.getParameter(gl.MAX_RENDERBUFFER_SIZE), ...gl.getParameter(gl.MAX_VIEWPORT_DIMS));
}

// This borrows the preview's renderer for one frame: its textures can't be
// shared with another, and a render target would skip the sRGB output step.
// Everything is put back before the browser next paints, so the preview doesn't
// flicker. toBlob takes its copy of the canvas straight away; only encoding is async.
function renderCapture(width, height, focalMm) {
  const size = renderer.getSize(new THREE.Vector2());
  const pixelRatio = renderer.getPixelRatio();
  const { fov, aspect } = camera;
  const gl = renderer.getContext();
  try {
    renderer.setDrawingBufferSize(width, height, 1);
    // Browsers shrink an oversized canvas silently instead of failing.
    if (gl.drawingBufferWidth !== width || gl.drawingBufferHeight !== height) {
      throw new Error(`This browser can't render ${width} × ${height}. Try a smaller size.`);
    }
    camera.fov = fovDegrees(height, focalPixels(width, focalMm));
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.render(scene, camera);
    return new Promise((resolve, reject) =>
      renderer.domElement.toBlob(
        (blob) => (blob ? resolve(blob) : reject(new Error("The browser couldn't encode the image."))),
        "image/png",
      ),
    );
  } finally {
    camera.fov = fov;
    camera.aspect = aspect;
    camera.updateProjectionMatrix();
    renderer.setDrawingBufferSize(size.x, size.y, pixelRatio);
  }
}

// The button lives inside #scene so it follows the preview into both dialogs.
const captureBtn = document.createElement("button");
captureBtn.className = "btn small";
captureBtn.type = "button";
captureBtn.hidden = true;
captureBtn.title = "Save this view as an image";
captureBtn.style.cssText = "position: absolute; z-index: 1; right: 12px; bottom: 12px;";
captureBtn.innerHTML = `
  <svg viewBox="0 0 16 16" fill="none" aria-hidden="true"><path d="M2 5.5h2.5L6 3.5h4l1.5 2H14v7H2z" stroke="currentColor" stroke-width="1.4" stroke-linejoin="round"/><circle cx="8" cy="8.75" r="2" stroke="currentColor" stroke-width="1.4"/></svg>
  Capture`;
sceneEl.appendChild(captureBtn);

const captureDlg = document.createElement("dialog");
captureDlg.className = "modal";
captureDlg.setAttribute("aria-labelledby", "captureTitle");
captureDlg.style.cssText = "width: min(360px, calc(100vw - 32px)); height: fit-content;";
captureDlg.innerHTML = `
  <form id="captureForm">
    <div class="modal-head">
      <h2 id="captureTitle" style="flex: 1">Capture image</h2>
      <button class="btn icon" type="button" data-close aria-label="Close">
        <svg viewBox="0 0 16 16" fill="none" aria-hidden="true"><path d="m4 4 8 8M12 4l-8 8" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg>
      </button>
    </div>
    <div class="modal-body">
      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px">
        <div class="field">
          <label for="captureWidth">Width (px)</label>
          <input id="captureWidth" type="number" min="1" step="1" value="1920" required />
        </div>
        <div class="field">
          <label for="captureHeight">Height (px)</label>
          <input id="captureHeight" type="number" min="1" step="1" value="1080" required />
        </div>
      </div>
      <div class="field">
        <label for="captureFocal">Focal length (mm, 35 mm equiv.)</label>
        <input id="captureFocal" type="number" min="1" step="any" value="28" required />
      </div>
      <span class="stats" id="captureInfo"></span>
    </div>
    <div class="modal-foot">
      <button class="btn primary" type="submit">Save PNG</button>
    </div>
  </form>`;
document.body.appendChild(captureDlg);

function renderCaptureInfo() {
  const width = $("captureWidth").valueAsNumber;
  const height = $("captureHeight").valueAsNumber;
  const focal = focalPixels(width, $("captureFocal").valueAsNumber);
  $("captureInfo").textContent =
    width > 0 && height > 0 && focal > 0
      ? `${fovDegrees(width, focal).toFixed(1)}° × ${fovDegrees(height, focal).toFixed(1)}° · f ${Math.round(focal)} px`
      : "";
}

captureBtn.addEventListener("click", () => {
  const limit = captureLimit();
  $("captureWidth").max = limit;
  $("captureHeight").max = limit;
  renderCaptureInfo();
  captureDlg.showModal();
});
captureDlg.addEventListener("input", renderCaptureInfo);
captureDlg.addEventListener("click", (e) => {
  if (e.target === captureDlg || e.target.closest("[data-close]")) captureDlg.close();
});
$("captureForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  captureDlg.close();
  const width = $("captureWidth").valueAsNumber;
  const height = $("captureHeight").valueAsNumber;
  const focal = $("captureFocal").valueAsNumber;
  try {
    const blob = await renderCapture(width, height, focal);
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `capture-${state.modelJobId}-${width}x${height}-${focal}mm.png`;
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    toast(`Saved ${width} × ${height} image`);
  } catch (err) {
    toast(err.message || String(err));
  }
});

async function loadModel(job) {
  // Only the latest request may change the scene: switching jobs quickly can
  // make an older model (or its error) arrive after the newer one.
  state.modelWanted = job.id;
  if (overlays.jobId !== job.id) disposeOverlays();
  const empty = $("stageEmpty");
  if (!renderer) {
    state.modelJobId = job.id; // nothing to load; the WebGL note shows instead
    empty.hidden = true;
    jobStageNote(job, "");
    showModelInfo(job);
    return;
  }
  empty.textContent = "Loading model…";
  empty.hidden = false;
  jobStageNote(job, "Loading model…");
  try {
    const gltf = await loader.loadAsync(`/api/jobs/${job.id}/model.glb`);
    if (state.modelWanted !== job.id) {
      disposeModel(gltf.scene);
      return;
    }
    if (model) {
      scene.remove(model);
      disposeModel(model);
    }
    model = gltf.scene;
    // The pipeline's materials don't set metalness, which glTF treats as fully
    // metallic. Metal ignores ordinary lights (it only reflects), so lighting
    // from all sides only works on a matte surface. Facades are matte anyway.
    model.traverse((o) => {
      const mats = Array.isArray(o.material) ? o.material : o.material ? [o.material] : [];
      for (const m of mats) {
        if ("metalness" in m && !m.metalnessMap) m.metalness = 0;
      }
    });
    scene.add(model);
    frameModel();
    captureBtn.hidden = false;
    state.modelJobId = job.id;
    empty.hidden = true;
    jobStageNote(job, "");
    loadOverlays(job);
  } catch (err) {
    if (state.modelWanted !== job.id) return;
    empty.textContent = `Couldn't load the model: ${err.message || err}`;
    jobStageNote(job, empty.textContent);
    return;
  }
  showModelInfo(job);
}
// ---------- Overlays: the photos' cameras and the alignment's point cloud ----------
// /api/jobs/<id>/overlays.json (common/overlays.py), already in the model's frame.
const CAMERA_COLOURS = { colmap: 0x3fb950, computed: 0xf0a020, raw: 0xe5534b };
const overlays = { cameras: null, points: null, jobId: null };

function disposeOverlays() {
  for (const key of ["cameras", "points"]) {
    if (overlays[key]) {
      scene.remove(overlays[key]);
      disposeModel(overlays[key]);
      overlays[key] = null;
    }
  }
  overlays.jobId = null;
  for (const id of ["showCameras", "showPoints"]) $(id).hidden = true;
}

function cameraFrusta(cameras) {
  // A small pyramid per camera: apex at the camera, 2 m deep, the photo's field of view.
  const pos = [];
  const col = [];
  const depth = 2;
  for (const c of cameras) {
    const p = new THREE.Vector3(...c.p);
    const f = new THREE.Vector3(...c.f).normalize();
    const r = new THREE.Vector3().crossVectors(f, new THREE.Vector3(...c.u)).normalize();
    const u = new THREE.Vector3().crossVectors(r, f);
    const hh = depth * Math.tan(((c.fov || 60) * Math.PI) / 360);
    const hw = hh * (c.aspect || 1.333);
    const corner = (sx, sy) =>
      p.clone().addScaledVector(f, depth).addScaledVector(r, sx * hw).addScaledVector(u, sy * hh);
    const k = [corner(-1, 1), corner(1, 1), corner(1, -1), corner(-1, -1)];
    const segs = [
      [p, k[0]], [p, k[1]], [p, k[2]], [p, k[3]],
      [k[0], k[1]], [k[1], k[2]], [k[2], k[3]], [k[3], k[0]],
    ];
    const colour = new THREE.Color(CAMERA_COLOURS[c.source] ?? CAMERA_COLOURS.raw);
    for (const [a, b] of segs) {
      pos.push(a.x, a.y, a.z, b.x, b.y, b.z);
      col.push(colour.r, colour.g, colour.b, colour.r, colour.g, colour.b);
    }
  }
  const geom = new THREE.BufferGeometry();
  geom.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  geom.setAttribute("color", new THREE.Float32BufferAttribute(col, 3));
  return new THREE.LineSegments(geom, new THREE.LineBasicMaterial({ vertexColors: true }));
}

function pointCloud(points) {
  const geom = new THREE.BufferGeometry();
  geom.setAttribute("position", new THREE.Float32BufferAttribute(points.xyz, 3));
  geom.setAttribute("color", new THREE.Uint8BufferAttribute(points.rgb, 3, true));
  return new THREE.Points(
    geom,
    new THREE.PointsMaterial({ size: 0.35, sizeAttenuation: true, vertexColors: true })
  );
}

async function loadOverlays(job) {
  disposeOverlays();
  let data;
  try {
    data = await api(`/api/jobs/${job.id}/overlays.json`);
  } catch {
    return; // jobs from before overlays existed have none
  }
  if (state.modelJobId !== job.id) return;
  overlays.jobId = job.id;
  if (data.cameras?.length) {
    overlays.cameras = cameraFrusta(data.cameras);
    $("showCameras").hidden = false;
  }
  if (data.points?.xyz?.length) {
    overlays.points = pointCloud(data.points);
    $("showPoints").hidden = false;
  }
  for (const [id, key] of [["showCameras", "cameras"], ["showPoints", "points"]]) {
    const obj = overlays[key];
    if (obj && $(id).getAttribute("aria-pressed") === "true") scene.add(obj);
  }
}

for (const [id, key] of [["showCameras", "cameras"], ["showPoints", "points"]]) {
  $(id).addEventListener("click", () => {
    const on = $(id).getAttribute("aria-pressed") !== "true";
    $(id).setAttribute("aria-pressed", String(on));
    const obj = overlays[key];
    if (!obj) return;
    if (on) scene.add(obj);
    else scene.remove(obj);
  });
}

function showModelInfo(job) {
  const dl = $("download");
  dl.href = `/api/jobs/${job.id}/model.glb?download=true`;
  dl.setAttribute("aria-disabled", "false");
  const stats = modelStatsHtml(job);
  $("meshStats").innerHTML = stats;
  $("sceneModalStats").innerHTML = stats;
}
function modelStatsHtml(job) {
  const r = job.result || {};
  return (
    `<span><b>${r.buildings ?? "—"}</b> buildings</span>` +
    (r.faces != null ? `<span>${r.faces.toLocaleString()} faces</span>` : "") +
    (r.model_bytes != null ? `<span>${(r.model_bytes / 1e6).toFixed(1)} MB</span>` : "")
  );
}

// ---------- Street imagery ----------
async function loadImages(job) {
  // As with models, ignore a response for a job that's no longer selected.
  state.imagesWanted = job.id;
  let images = [];
  try {
    images = (await api(`/api/jobs/${job.id}/images`)).images;
  } catch {
    /* no photos for this job */
  }
  if (state.imagesWanted !== job.id) return;
  state.images = images;
  state.imagesJobId = job.id;
  renderGallery();
}
function renderGallery() {
  const hasCleaned = state.images.some((i) => i.cleaned);
  const useCleaned = state.showCleaned && hasCleaned;
  for (const id of ["beforeAfter", "jobBeforeAfter"]) {
    const root = $(id);
    root.querySelector('[data-v="after"]').disabled = !hasCleaned;
    root
      .querySelectorAll("button")
      .forEach((b) => b.setAttribute("aria-pressed", String((b.dataset.v === "after") === useCleaned)));
  }
  $("galleryEmpty").hidden = state.images.length > 0;
  const html = state.images
    .map((img) => {
      const src = useCleaned && img.cleaned ? img.cleaned : img.original;
      const chip = useCleaned && img.cleaned ? `<span class="chip ok">Cleaned</span>` : `<span class="chip">Original</span>`;
      const heading = img.heading != null ? ` · ${img.heading}°` : "";
      return `<figure class="shot"><img src="${esc(src)}" alt="Street photo ${esc(img.id)}" loading="lazy">
        <figcaption><span>${esc(img.id)}${heading}</span>${chip}</figcaption></figure>`;
    })
    .join("");
  $("gallery").innerHTML = html;

  // The job details dialog shows these photos only if they belong to its job.
  const modalJob = state.jobs.find((j) => j.id === state.modalJobId);
  if (!modalJob) return;
  const own = state.imagesJobId === modalJob.id;
  $("jobGallery").innerHTML = own ? html : "";
  const empty = $("jobGalleryEmpty");
  empty.hidden = own && state.images.length > 0;
  $("jobBeforeAfter").hidden = !empty.hidden;
  empty.textContent =
    modalJob.status === "failed"
      ? "No photos: this job failed."
      : modalJob.status !== "done"
        ? "Photos appear here when the job finishes."
        : "Loading photos…";
}

// ---------- Progress timeline (built from the pipeline's own stages) ----------
function buildStepper() {
  $("stepper").style.setProperty("--n", state.stages.length);
  $("steps").innerHTML = state.stages
    .map(
      (s) => `<li class="step" data-stage="${esc(s.name)}" data-state="pending">
        <span class="dot"><svg viewBox="0 0 12 12" aria-hidden="true"><path d="M2.5 6.2 5 8.5l4.5-5"/></svg></span>
        <div><strong>${esc(s.label)}</strong><small>${esc(s.detail)}</small></div></li>`,
    )
    .join("");
}
// ---------- Pipeline log ----------
// Shows the log of the job in the timeline: loads the saved file over HTTP,
// then appends lines pushed over the WebSocket. Lines are numbered so nothing
// is shown twice when the two overlap.
const logState = { jobId: null, count: 0, loading: false, pending: [] };
const logBody = $("logBody");
logBody.dataset.empty = "Logs appear here when a job runs.";

function appendLogLine(line) {
  const nearBottom = logBody.scrollHeight - logBody.scrollTop - logBody.clientHeight < 40;
  const [ts, ...rest] = line.split("  ");
  const text = rest.join("  ");
  const row = document.createElement("div");
  const time = document.createElement("span");
  time.className = "ts";
  time.textContent = ts + "  ";
  const body = document.createElement("span");
  if (/^(Warning|Error):|^Traceback|Error: /.test(text)) body.className = "warn";
  body.textContent = text;
  row.append(time, body);
  logBody.append(row);
  if (nearBottom) logBody.scrollTop = logBody.scrollHeight;
}

async function loadLog(job) {
  logState.jobId = job.id;
  logState.count = 0;
  logState.pending = [];
  logState.loading = true;
  logBody.replaceChildren();
  $("logProgress").hidden = true;
  $("logMeta").textContent = job.name;
  const dl = $("logDownload");
  dl.href = `/api/jobs/${job.id}/log?download=true`;
  dl.setAttribute("aria-disabled", "false");
  let text = "";
  try {
    text = await (await fetch(`/api/jobs/${job.id}/log`)).text();
  } catch {
    /* the live stream still fills in new lines */
  }
  if (logState.jobId !== job.id) return; // another job was selected meanwhile
  for (const line of text.split("\n")) {
    if (!line) continue;
    logState.count++;
    appendLogLine(line);
  }
  logState.loading = false;
  for (const msg of logState.pending) onLogMessage(msg);
  logState.pending = [];
  logBody.scrollTop = logBody.scrollHeight;
}

function syncLog(job) {
  if (!job) return;
  if (job.id !== logState.jobId) loadLog(job);
}

function onLogMessage(msg) {
  if (msg.job_id !== logState.jobId) return;
  if (logState.loading) return void logState.pending.push(msg);
  if (msg.type === "progress") {
    $("logProgress").textContent = msg.text;
    $("logProgress").hidden = !msg.text;
    return;
  }
  if (msg.n <= logState.count) return; // already loaded from the file
  logState.count = msg.n;
  appendLogLine(msg.line);
}

$("logToggle").addEventListener("click", () => {
  const open = $("logToggle").getAttribute("aria-expanded") !== "true";
  $("logToggle").setAttribute("aria-expanded", String(open));
  $("logWrap").hidden = !open;
});

function renderProgress(job) {
  syncLog(job);
  const n = state.stages.length;
  const steps = $("steps").children;
  let done = 0;
  let skipped = 0;
  let reach = null;
  let finished = -1; // last stage of the unbroken run of done/skipped ones
  state.stages.forEach((s, i) => {
    let v = job?.stages?.[s.name];
    // Jobs from before a stage existed never ran it.
    if (!v) v = job && !isActive(job) ? "skipped" : "pending";
    steps[i].dataset.state = v;
    steps[i].querySelector("small").textContent = v === "skipped" ? "Off for this job" : s.detail;
    if (v === "done") done++;
    if (v === "skipped") skipped++;
    if ((v === "done" || v === "skipped") && finished === i - 1) finished = i;
    if (v === "running" || v === "error") reach = i;
  });
  if (reach == null) reach = Math.max(0, finished);
  $("fill").style.width = n > 1 && job ? `calc(${reach / (n - 1)} * (100% - 100% / ${n}))` : "0";
  const counted = n - skipped; // skipped stages don't count toward the percentage
  $("progPct").textContent = `${counted ? Math.round((done / counted) * 100) : 0}%`;

  let label = "Ready to generate";
  if (job?.status === "queued") label = "Waiting in queue";
  else if (job?.status === "running") {
    const running = state.stages.find((s) => job.stages[s.name] === "running");
    label = running ? `${running.label}…` : "Starting…";
  } else if (job?.status === "done") label = "Model ready";
  else if (job?.status === "failed") label = `Failed: ${job.error || "unknown error"}`;
  $("progLabel").textContent = label;
  $("status").classList.toggle("failed", job?.status === "failed");
}

// ---------- Jobs ----------
const isActive = (j) => j.status === "queued" || j.status === "running";
function duration(j) {
  if (!j.started_at || !j.finished_at) return isActive(j) ? "In progress" : "—";
  const s = Math.max(0, Math.round((new Date(j.finished_at) - new Date(j.started_at)) / 1000));
  return s >= 60 ? `${Math.floor(s / 60)}m ${String(s % 60).padStart(2, "0")}s` : `${s}s`;
}
function upsertJob(job) {
  const i = state.jobs.findIndex((j) => j.id === job.id);
  if (i >= 0) state.jobs[i] = job;
  else state.jobs.unshift(job);
}
const statusChip = (status) =>
  ({
    done: `<span class="chip ok">Done</span>`,
    failed: `<span class="chip bad">Failed</span>`,
    running: `<span class="chip run">Running</span>`,
    queued: `<span class="chip">Queued</span>`,
  })[status] || "";
function renderJobs() {
  $("jobsEmpty").hidden = state.jobs.length > 0;
  $("jobRows").innerHTML = state.jobs
    .map((j) => {
      const chip = statusChip(j.status);
      const thumb = j.result?.thumbnail
        ? `<img class="thumb" src="/api/jobs/${esc(j.id)}/images/${esc(j.result.thumbnail)}" alt="">`
        : `<span class="thumb"></span>`;
      const current = state.shown?.id === j.id || state.current?.id === j.id;
      return `<tr data-id="${esc(j.id)}" aria-current="${current}">
        <td><div class="name-cell">${thumb}<div><button class="job-link" type="button" data-id="${esc(j.id)}">${esc(j.name)}</button>
          ${j.error ? `<span class="sub">${esc(j.error)}</span>` : ""}</div></div></td>
        <td class="mono">${fmtBox(j.bbox)}</td><td>${chip}</td>
        <td class="mono">${j.result?.buildings ?? "—"}</td><td class="mono">${duration(j)}</td></tr>`;
    })
    .join("");
}
// The whole row is clickable; the name button keeps it reachable by keyboard.
$("jobRows").addEventListener("click", (e) => {
  const row = e.target.closest("tr[data-id]");
  if (!row || window.getSelection().toString()) return; // let people copy a bbox
  const job = state.jobs.find((j) => j.id === row.dataset.id);
  if (!job) return;
  showJob(job);
  openJobModal(job);
});

function showJob(job) {
  state.current = job;
  renderProgress(job);
  setBox(job.bbox, { fit: true });
  if (job.status === "done") {
    state.shown = job;
    loadModel(job);
    loadImages(job);
  }
  renderJobs();
}

// ---------- Job details dialog ----------
// Like the larger views, the live 3D preview and pipeline log move into the
// dialog (so they keep updating) and go back where they were on close.
const jobDlg = $("jobModal");
const homes = new Map(); // element -> the spot it left, while it's in the dialog
// holdSpace keeps the element's height in the page, so nothing behind the
// dialog shifts while it's away.
function moveInto(el, parent, { holdSpace = false } = {}) {
  if (!homes.has(el)) {
    const spot = document.createElement("div");
    if (holdSpace) spot.style.height = `${el.offsetHeight}px`;
    el.replaceWith(spot);
    homes.set(el, spot);
  }
  parent.appendChild(el);
}
function moveHome(el) {
  const spot = homes.get(el);
  if (!spot) return;
  spot.replaceWith(el);
  homes.delete(el);
}

// The page behind can still change height (e.g. the gallery fills with the
// job's photos), so on close the jobs list is put back where it was.
let jobsTopBefore = null;
const jobsTop = () => $("jobs").getBoundingClientRect().top;

function openJobModal(job) {
  state.modalJobId = job.id;
  if (!jobDlg.open) jobsTopBefore = jobsTop();
  moveInto($("logSection"), $("jobSide"), { holdSpace: true });
  renderJobModal(job);
  if (!jobDlg.open) jobDlg.showModal();
}

function renderJobModal(job) {
  if (state.modalJobId !== job.id) return;
  $("jobModalTitle").textContent = job.name;
  $("jobModalChip").innerHTML = statusChip(job.status);
  const done = job.status === "done";
  $("jobModalStats").innerHTML = done ? modelStatsHtml(job) : "";
  const dl = $("jobModalDownload");
  if (done) dl.href = `/api/jobs/${job.id}/model.glb?download=true`;
  else dl.removeAttribute("href");
  dl.setAttribute("aria-disabled", String(!done));
  $("jobFacts").innerHTML = jobFactsHtml(job);

  // Only a finished job has a model to show; otherwise say why there isn't one.
  if (done) {
    moveInto(sceneEl, $("jobStage"));
    jobStageNote(job, state.modelJobId === job.id ? "" : "Loading model…");
  } else {
    moveHome(sceneEl);
    jobStageNote(
      job,
      job.status === "failed" ? "No model: this job failed." : "The model appears here when the job finishes.",
    );
  }
  $("stagePlaceholder").hidden = !homes.has(sceneEl);
  renderGallery();
}

function jobStageNote(job, text) {
  if (state.modalJobId !== job.id) return;
  $("jobStageEmpty").textContent = text;
  $("jobStageEmpty").hidden = !text;
}

// Jobs from before the option existed used Mapillary's raw GPS.
const POSE_LABELS = {
  colmap: "Aligned to OpenStreetMap",
  computed: "Mapillary computed",
  raw: "Mapillary raw GPS",
};

function jobFactsHtml(job) {
  const { w, h } = boxSize(job.bbox);
  const o = job.options || {};
  const when = (t) => (t ? new Date(t).toLocaleString() : "—");
  const rows = [
    ["Bounding box", `<span class="mono">${fmtBox(job.bbox)}</span>`],
    ["Area", `<span class="mono">${Math.round(w)} × ${Math.round(h)} m · ${((w * h) / 1e6).toFixed(2)} km²</span>`],
    ["Created", esc(when(job.created_at))],
    ["Duration", esc(duration(job))],
    ["Camera poses", esc(POSE_LABELS[o.poses] || POSE_LABELS.raw) + (o.aligned_only ? " · aligned photos only" : "")],
    [
      "Obstructions",
      o.remove_obstructions
        ? `${o.fill_obstructions === false ? "Masked, not filled" : "Removed"} · ${esc(o.seg_model)} model`
        : "Kept",
    ],
    ["Device", esc(o.device || "auto")],
  ];
  // Jobs from before these options existed don't have them.
  if ("terrain" in o) rows.push(["Terrain", o.terrain ? "On" : "Off"]);
  if ("naip" in o) rows.push(["Aerial photos", o.naip ? "NAIP ground and roofs" : "Off"]);
  if (job.result?.photos != null) rows.push(["Photos", `<span class="mono">${job.result.photos}</span>`]);
  if (job.result?.aligned_photos != null)
    rows.push(["Aligned photos", `<span class="mono">${job.result.aligned_photos}</span>`]);
  if (job.error) rows.push(["Error", `<span class="bad">${esc(job.error)}</span>`]);
  return rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("");
}

jobDlg.addEventListener("click", (e) => {
  if (e.target === jobDlg || e.target.closest("[data-close]")) jobDlg.close();
});
jobDlg.addEventListener("close", () => {
  state.modalJobId = null;
  moveHome(sceneEl);
  moveHome($("logSection"));
  $("stagePlaceholder").hidden = true;
  if (jobsTopBefore != null) scrollBy({ top: jobsTop() - jobsTopBefore, behavior: "instant" });
  jobsTopBefore = null;
});

// ---------- Live updates over WebSocket ----------
// The server pushes {"type":"snapshot","jobs":[...]} on connect and
// {"type":"job","job":{...}} whenever a job is created or a stage changes.

function stageProgress(job) {
  const i = state.stages.findIndex((s) => job.stages?.[s.name] === "running");
  return i >= 0 ? `${state.stages[i].label} · step ${i + 1} of ${state.stages.length}` : "Waiting to start";
}

// Grey out the preview and disable Generate while any job is queued or running.
function renderBusy() {
  const active = state.jobs.find(isActive);
  const busy = Boolean(active);
  $("stage").classList.toggle("busy", busy);
  sceneEl.classList.toggle("busy", busy);
  $("stageBusy").hidden = !busy;
  $("stageEmpty").classList.toggle("under-busy", busy);
  if (busy) {
    $("busyLabel").textContent = active.status === "queued" ? "Job queued" : "Generating model";
    $("busyDetail").textContent = active.status === "queued" ? "Starts when the current job finishes" : stageProgress(active);
  }
  $("go").disabled = busy || !state.stages.length;
  $("go").querySelector("span").textContent = busy ? "Generating…" : "Generate model";
}

function finishJob(job) {
  if (job.status === "done") {
    toast(`Model ready: ${job.result?.buildings ?? 0} buildings`);
    if (state.current?.id === job.id) {
      state.shown = job;
      loadModel(job);
      loadImages(job);
    }
  } else if (job.status === "failed") {
    toast("The job failed. See the details under the preview.");
  }
}

function onJobUpdate(job) {
  const prev = state.jobs.find((j) => j.id === job.id);
  upsertJob(job);
  // Follow a job that starts while nothing else is being watched (e.g. from
  // another tab), but not while the details of another job are open.
  const watching = state.current && (isActive(state.current) || state.modalJobId === state.current.id);
  if (isActive(job) && !(watching && state.current.id !== job.id)) {
    state.current = job;
  }
  if (state.current?.id === job.id) {
    state.current = job;
    renderProgress(job);
  }
  if (prev && isActive(prev) && !isActive(job)) finishJob(job);
  renderJobModal(job);
  renderJobs();
  renderBusy();
}

function onSnapshot(jobs) {
  const before = new Map(state.jobs.map((j) => [j.id, j]));
  const firstLoad = state.jobs.length === 0 && !state.current;
  state.jobs = jobs;
  const active = jobs.find(isActive);
  const lastDone = jobs.find((j) => j.status === "done");

  if (firstLoad) {
    if (lastDone) showJob(lastDone);
    if (active) state.current = active;
    renderProgress(state.current);
  } else {
    // Reconnected: anything that finished while we were offline.
    for (const job of jobs) {
      const prev = before.get(job.id);
      if (prev && isActive(prev) && !isActive(job)) {
        if (state.current?.id === job.id) state.current = job;
        finishJob(job);
      }
    }
    // Follow the running job, unless another job's details are open (the
    // dialog shares the log and 3D preview with the page).
    if (active && !state.modalJobId) state.current = active;
    if (state.current) state.current = jobs.find((j) => j.id === state.current.id) || state.current;
    // Lines logged while we were offline never arrive live, so reload the
    // saved log; live lines that overlap it are skipped by their number.
    logState.jobId = null;
    renderProgress(state.current);
    const modalJob = jobs.find((j) => j.id === state.modalJobId);
    if (modalJob) renderJobModal(modalJob);
  }
  renderJobs();
  renderBusy();
}

function setConnection(text) {
  $("liveStatus").textContent = text;
  $("liveStatus").hidden = !text;
}

function connectLive() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/api/ws`);
  state.socket = ws;
  ws.addEventListener("open", () => {
    state.reconnectDelay = 1000;
    setConnection("");
  });
  ws.addEventListener("message", (e) => {
    let msg;
    try {
      msg = JSON.parse(e.data);
    } catch {
      return;
    }
    if (msg.type === "snapshot") onSnapshot(msg.jobs);
    else if (msg.type === "job") onJobUpdate(msg.job);
    else if (msg.type === "log" || msg.type === "progress") onLogMessage(msg);
  });
  ws.addEventListener("close", () => {
    if (state.socket !== ws) return;
    setConnection("Reconnecting to the server…");
    setTimeout(connectLive, state.reconnectDelay);
    state.reconnectDelay = Math.min(state.reconnectDelay * 2, 10000);
  });
}

// ---------- Settings ----------
// /api/settings says whether a token is set, where it came from (saved here,
// or the APP_ENV's env file) and its last characters. The token itself comes
// from its own endpoint, only while this dialog is open, and shows as dots
// until "Show". Changing it takes Change, Save token, then a confirmation.
const settingsDlg = $("settingsModal");
const tokenInput = $("tokenInput");
let defDevice = "auto";
segmented("defDevice", (v) => (defDevice = v));

const tokenUi = { current: "", editing: false, confirm: null };
const hasToken = () => Boolean(state.settings?.mapillary.set);
const envFile = () => state.settings?.mapillary.env_file || ".env";

function renderTokenStatus() {
  const m = state.settings?.mapillary;
  const status = !m?.set
    ? `<span class="chip bad">Not set</span>`
    : `<span class="chip ok">${m.source === "saved" ? "Saved here" : `From ${esc(m.env_file)}`}${m.hint ? ` · ends ${esc(m.hint)}` : ""}</span>`;
  $("tokenStatus").innerHTML = status;
  $("tokenNotice").hidden = !state.settings || m.set;
}

// View mode: the current token, read-only. Edit mode: an empty field to paste
// a new one. There's nothing to view without a token, so that's always edit.
function renderTokenControls() {
  const m = state.settings?.mapillary;
  const editing = tokenUi.editing || !m?.set;
  const confirming = Boolean(tokenUi.confirm);
  tokenInput.readOnly = !editing || confirming;
  $("tokenInputLabel").textContent = editing ? (m?.set ? "New token" : "Paste a token") : "Token";
  $("tokenChange").hidden = editing || confirming;
  $("tokenSave").hidden = !editing || confirming;
  $("tokenCancel").hidden = !editing || !m?.set || confirming;
  $("tokenTest").hidden = confirming;
  $("tokenClear").hidden = editing || confirming || m?.source !== "saved";
  $("tokenClear").textContent = m?.env_available ? `Use the ${envFile()} token instead` : "Remove saved token";
  $("tokenConfirm").hidden = !confirming;
  if (confirming) {
    $("tokenConfirmText").textContent = tokenUi.confirm.text;
    $("tokenConfirmYes").textContent = tokenUi.confirm.yes;
    $("tokenConfirmNo").textContent = tokenUi.confirm.no;
  }
}

function showTokenValue(show) {
  tokenInput.type = show ? "text" : "password";
  $("tokenShow").setAttribute("aria-pressed", String(show));
  $("tokenShow").setAttribute("aria-label", show ? "Hide token" : "Show token");
  $("tokenShow").textContent = show ? "Hide" : "Show";
}

function setDeviceButtons(id, value) {
  $(id)
    .querySelectorAll("button")
    .forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.v === value)));
}

// Put the saved defaults into the Options panel (on load, and after saving).
function applyDefaults(d) {
  $("poses").value = d.poses || "colmap";
  $("fillObs").checked = d.fill_obstructions !== false;
  $("seg").checked = d.remove_obstructions;
  $("model").value = d.seg_model;
  // "cuda" was the GPU button's value before it meant any GPU (NVIDIA or Apple)
  const wanted = d.device === "cuda" ? "gpu" : d.device;
  device = wanted === "gpu" && !state.system.gpu ? "auto" : wanted;
  setDeviceButtons("device", device);
  syncOptionRows();
}

// "Aligned photos only" applies to aligned poses; "Fill obstructions" to removing them:
// each is shown only when it applies (and keeps its setting while hidden).
function syncOptionRows() {
  $("alignedOnlyRow").hidden = $("poses").value !== "colmap";
  $("fillObsRow").hidden = !$("seg").checked;
}
$("poses").addEventListener("change", syncOptionRows);
$("seg").addEventListener("change", syncOptionRows);

function tokenMessage(text, kind = "") {
  const msg = $("tokenMsg");
  msg.textContent = text;
  msg.className = `set-msg ${kind}`;
  msg.hidden = !text;
}

// Back to view mode with the current token (fetched fresh from the server).
async function showCurrentToken() {
  tokenUi.editing = false;
  tokenUi.confirm = null;
  tokenUi.current = "";
  if (hasToken()) {
    try {
      tokenUi.current = (await api("/api/settings/mapillary/token")).token || "";
    } catch {
      /* the status chip still says it's set */
    }
  }
  tokenInput.value = tokenUi.current;
  renderTokenStatus();
  renderTokenControls();
}

async function openSettings(intro = "") {
  const d = state.settings?.defaults;
  showTokenValue(false);
  tokenMessage("");
  $("settingsError").hidden = true;
  $("settingsIntro").textContent = intro;
  $("settingsIntro").hidden = !intro;
  if (d) {
    $("defPoses").value = d.poses || "colmap";
    $("defFill").checked = d.fill_obstructions !== false;
    $("defSeg").checked = d.remove_obstructions;
    $("defModel").value = d.seg_model;
    defDevice = d.device === "cuda" ? "gpu" : d.device;
    setDeviceButtons("defDevice", defDevice);
  }
  $("defDevice").querySelector('[data-v="gpu"]').disabled = !state.system.gpu;
  tokenInput.value = "";
  renderTokenStatus();
  renderTokenControls();
  settingsDlg.showModal();
  await showCurrentToken();
  if (!hasToken()) tokenInput.focus();
}

async function loadSettings() {
  state.settings = await api("/api/settings");
  renderTokenStatus();
}

async function saveToken(body) {
  state.settings = await api("/api/settings", { method: "PUT", body: JSON.stringify(body) });
}

$("openSettings").addEventListener("click", () => openSettings());
$("tokenNoticeBtn").addEventListener("click", () => openSettings());
settingsDlg.addEventListener("click", (e) => {
  if (e.target === settingsDlg || e.target.closest("[data-close]")) settingsDlg.close();
});
// Don't keep the token in the page once the dialog is closed.
settingsDlg.addEventListener("close", () => {
  tokenInput.value = "";
  tokenUi.current = "";
  tokenUi.editing = false;
  tokenUi.confirm = null;
});

$("tokenShow").addEventListener("click", () => showTokenValue(tokenInput.type === "password"));
// Enter in the token field means "Save token", not the footer's Save.
tokenInput.addEventListener("keydown", (e) => {
  if (e.key !== "Enter") return;
  e.preventDefault();
  if (!$("tokenSave").hidden) $("tokenSave").click();
});

$("tokenChange").addEventListener("click", () => {
  tokenUi.editing = true;
  tokenInput.value = "";
  tokenMessage("");
  renderTokenControls();
  tokenInput.focus();
});

$("tokenCancel").addEventListener("click", () => {
  tokenMessage("");
  showCurrentToken();
});

$("tokenSave").addEventListener("click", () => {
  const token = tokenInput.value.trim();
  if (!token) {
    tokenMessage("Paste the new token first.", "bad");
    tokenInput.focus();
    return;
  }
  if (token === tokenUi.current) {
    tokenMessage("That's the token you already have.", "bad");
    return;
  }
  const m = state.settings.mapillary;
  // Without an env file yet, saving creates it (e.g. .env) with this token.
  const where = m.env_file_exists ? "" : ` PSM will create ${envFile()} with it.`;
  tokenMessage("");
  tokenUi.confirm = {
    text: m.set
      ? `Replace the current token (ends ${m.hint || "…"}) with this one? Jobs started after this use the new token.${where}`
      : `Save this token? PSM uses it to download street photos for every job.${where}`,
    yes: m.set ? "Replace token" : "Save token",
    no: m.set ? "Keep current" : "Not yet",
    run: async () => {
      const res = await api("/api/settings", { method: "PUT", body: JSON.stringify({ mapillary_token: token }) });
      state.settings = res;
      await showCurrentToken();
      tokenMessage(
        res.token_saved_to === "env_file"
          ? `Token saved in a new ${envFile()} file. It's used from the next job.`
          : "Token saved. It's used from the next job.",
        "ok",
      );
    },
  };
  renderTokenControls();
  $("tokenConfirmYes").focus();
});

$("tokenClear").addEventListener("click", () => {
  const m = state.settings.mapillary;
  tokenMessage("");
  tokenUi.confirm = {
    text: m.env_available
      ? `Remove the token saved here? PSM goes back to the token in ${envFile()}.`
      : "Remove the saved token? You won't be able to generate models until you add one again.",
    yes: "Remove token",
    no: "Keep it",
    run: async () => {
      await saveToken({ clear_mapillary_token: true });
      await showCurrentToken();
      tokenMessage(hasToken() ? `Removed. Using the token from ${envFile()}.` : "Removed. No token is set now.");
    },
  };
  renderTokenControls();
  $("tokenConfirmNo").focus();
});

$("tokenConfirmYes").addEventListener("click", async () => {
  const { run } = tokenUi.confirm || {};
  if (!run) return;
  $("tokenConfirmYes").disabled = true;
  try {
    await run();
  } catch (e) {
    tokenUi.confirm = null;
    renderTokenControls();
    tokenMessage(e.message, "bad");
  } finally {
    $("tokenConfirmYes").disabled = false;
  }
});

$("tokenConfirmNo").addEventListener("click", () => {
  tokenUi.confirm = null;
  renderTokenControls();
  (tokenUi.editing || !hasToken() ? tokenInput : $("tokenChange")).focus();
});

$("tokenTest").addEventListener("click", async () => {
  const btn = $("tokenTest");
  const typed = tokenUi.editing || !hasToken() ? tokenInput.value.trim() : "";
  btn.disabled = true;
  tokenMessage(typed ? "Validating the new token…" : "Validating the current token…");
  try {
    const res = await api("/api/settings/mapillary/test", {
      method: "POST",
      body: JSON.stringify({ token: typed || null }),
    });
    tokenMessage(res.message, res.ok ? "ok" : "bad");
  } catch (e) {
    tokenMessage(e.message, "bad");
  } finally {
    btn.disabled = false;
  }
});

// The footer's Save is for the defaults; the token has its own Save + confirm.
$("settingsForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const save = $("settingsSave");
  const err = $("settingsError");
  err.hidden = true;
  if ((tokenUi.editing && tokenInput.value.trim()) || tokenUi.confirm) {
    err.textContent = "Save or cancel the token change first.";
    err.hidden = false;
    return;
  }
  save.disabled = true;
  try {
    state.settings = await api("/api/settings", {
      method: "PUT",
      body: JSON.stringify({
        defaults: {
          poses: $("defPoses").value,
          remove_obstructions: $("defSeg").checked,
          fill_obstructions: $("defFill").checked,
          seg_model: $("defModel").value,
          device: defDevice,
        },
      }),
    });
    applyDefaults(state.settings.defaults);
    renderTokenStatus();
    settingsDlg.close();
    toast("Settings saved");
  } catch (e) {
    err.textContent = e.message;
    err.hidden = false;
  } finally {
    save.disabled = false;
  }
});

// ---------- Generate ----------
$("go").addEventListener("click", async () => {
  const err = $("formError");
  err.hidden = true;
  if (state.settings && !hasToken()) {
    openSettings("Add your Mapillary token to generate a model. PSM uses it to download the street photos.");
    return;
  }
  const go = $("go");
  go.disabled = true;
  try {
    const job = await api("/api/jobs", {
      method: "POST",
      body: JSON.stringify({
        bbox: box,
        name: state.placeName || "Custom area",
        poses: $("poses").value,
        aligned_only: $("poses").value === "colmap" && $("alignedOnly").checked,
        remove_obstructions: $("seg").checked,
        fill_obstructions: $("fillObs").checked,
        seg_model: $("model").value,
        device,
        terrain: $("terrain").checked,
        naip: $("naip").checked,
      }),
    });
    // The socket often delivers newer updates for this job (running, first
    // stage) before this response arrives, so don't overwrite them.
    const known = state.jobs.find((j) => j.id === job.id);
    state.current = known || job;
    if (known) {
      renderProgress(known);
      renderJobs();
      renderBusy();
    } else {
      onJobUpdate(job);
    }
  } catch (e) {
    err.textContent = e.message;
    err.hidden = false;
    go.disabled = false;
  }
});

// ---------- Theme ----------
const root = document.documentElement;
const systemDark = matchMedia("(prefers-color-scheme: dark)");
const themeBtn = $("themeToggle");

function effectiveTheme() {
  return root.dataset.theme || (systemDark.matches ? "dark" : "light");
}
function syncThemeButton() {
  const theme = effectiveTheme();
  root.dataset.effectiveTheme = theme;
  themeBtn.setAttribute("aria-label", theme === "dark" ? "Switch to light mode" : "Switch to dark mode");
  themeBtn.title = themeBtn.getAttribute("aria-label");
}
themeBtn.addEventListener("click", () => {
  const next = effectiveTheme() === "dark" ? "light" : "dark";
  if (!reduceMotion) {
    root.classList.add("theme-switching");
    setTimeout(() => root.classList.remove("theme-switching"), 300);
  }
  root.dataset.theme = next;
  try {
    localStorage.setItem("psm-theme", next);
  } catch {
    /* private mode: the choice just isn't remembered */
  }
});

function onTheme() {
  syncThemeButton();
  rect.setStyle(rectStyle());
  applySceneTheme();
}
systemDark.addEventListener("change", onTheme);
syncThemeButton();
new MutationObserver(onTheme).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });

// ---------- Start ----------
syncInputs();
renderStats();
applySceneTheme();
renderGallery();

(async () => {
  try {
    const [pipeline, system] = await Promise.all([api("/api/pipeline"), api("/api/system")]);
    state.stages = pipeline.stages;
    state.system = system;
  } catch (e) {
    $("formError").textContent = `Can't reach the PSM server: ${e.message}`;
    $("formError").hidden = false;
    $("go").disabled = true;
    return;
  }
  const gpu = $("device").querySelector('[data-v="gpu"]');
  gpu.disabled = !state.system.gpu;
  gpu.title = state.system.gpu
    ? state.system.gpu === "mps"
      ? "Apple GPU"
      : "NVIDIA GPU (CUDA)"
    : "No GPU on this server";
  try {
    await loadSettings();
    applyDefaults(state.settings.defaults);
  } catch {
    /* an older server without settings; the Options panel keeps its defaults */
  }
  buildStepper();
  renderStats();
  renderProgress(null);
  connectLive();
})();

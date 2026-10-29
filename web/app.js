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
}
function suggestMessage(msg) {
  results = [];
  suggest.innerHTML = `<li class="empty">${esc(msg)}</li>`;
  suggest.hidden = false;
  addr.setAttribute("aria-expanded", "true");
}
function renderResults(list, query, emptyMessage = `No places found for "${query}".`) {
  results = list;
  active = 0;
  if (!list.length) return suggestMessage(emptyMessage);
  suggest.innerHTML = list
    .map((p, i) => `<li role="option" aria-selected="${i === 0}" data-i="${i}">${esc(p.name)}<span>${esc(p.address)}</span></li>`)
    .join("");
  suggest.hidden = false;
  addr.setAttribute("aria-expanded", "true");
}
function highlight() {
  suggest.querySelectorAll("li[data-i]").forEach((li, i) => li.setAttribute("aria-selected", String(i === active)));
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
  if (addr.value.trim().length >= 3 && results.length) {
    suggest.hidden = false;
    addr.setAttribute("aria-expanded", "true");
  }
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
    onToggle(false);
  });
  return () => {
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
segmented("beforeAfter", (v) => {
  state.showCleaned = v === "after";
  renderGallery();
});

// ---------- 3D preview ----------
const sceneEl = $("scene");
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(2, devicePixelRatio || 1));
sceneEl.appendChild(renderer.domElement);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(40, 1, 0.5, 20000);
scene.add(new THREE.HemisphereLight(0xffffff, 0x8a8a8a, 1.8));
const sun = new THREE.DirectionalLight(0xffffff, 1.2);
sun.position.set(300, 600, 200);
scene.add(sun);
const controls = new OrbitControls(camera, renderer.domElement);
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
  if (!r.width || !r.height) return;
  renderer.setSize(r.width, r.height, false);
  camera.aspect = r.width / r.height;
  camera.updateProjectionMatrix();
}
new ResizeObserver(resizeScene).observe(sceneEl);
renderer.setAnimationLoop(() => {
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

async function loadModel(job) {
  const empty = $("stageEmpty");
  empty.textContent = "Loading model…";
  empty.hidden = false;
  try {
    const gltf = await loader.loadAsync(`/api/jobs/${job.id}/model.glb`);
    if (model) {
      scene.remove(model);
      disposeModel(model);
    }
    model = gltf.scene;
    scene.add(model);
    frameModel();
    empty.hidden = true;
  } catch (err) {
    empty.textContent = `Couldn't load the model: ${err.message || err}`;
    return;
  }
  const dl = $("download");
  dl.href = `/api/jobs/${job.id}/model.glb?download=true`;
  dl.setAttribute("aria-disabled", "false");
  const r = job.result || {};
  const stats =
    `<span><b>${r.buildings ?? "—"}</b> buildings</span>` +
    (r.faces != null ? `<span>${r.faces.toLocaleString()} faces</span>` : "") +
    (r.model_bytes != null ? `<span>${(r.model_bytes / 1e6).toFixed(1)} MB</span>` : "");
  $("meshStats").innerHTML = stats;
  $("sceneModalStats").innerHTML = stats;
}

// ---------- Street imagery ----------
async function loadImages(job) {
  try {
    state.images = (await api(`/api/jobs/${job.id}/images`)).images;
  } catch {
    state.images = [];
  }
  renderGallery();
}
function renderGallery() {
  const hasCleaned = state.images.some((i) => i.cleaned);
  const cleanedBtn = $("beforeAfter").querySelector('[data-v="after"]');
  cleanedBtn.disabled = !hasCleaned;
  const useCleaned = state.showCleaned && hasCleaned;
  $("beforeAfter")
    .querySelectorAll("button")
    .forEach((b) => b.setAttribute("aria-pressed", String((b.dataset.v === "after") === useCleaned)));
  $("galleryEmpty").hidden = state.images.length > 0;
  $("gallery").innerHTML = state.images
    .map((img) => {
      const src = useCleaned && img.cleaned ? img.cleaned : img.original;
      const chip = useCleaned && img.cleaned ? `<span class="chip ok">Cleaned</span>` : `<span class="chip">Original</span>`;
      const heading = img.heading != null ? ` · ${img.heading}°` : "";
      return `<figure class="shot"><img src="${esc(src)}" alt="Street photo ${esc(img.id)}" loading="lazy">
        <figcaption><span>${esc(img.id)}${heading}</span>${chip}</figcaption></figure>`;
    })
    .join("");
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
  let reach = null;
  state.stages.forEach((s, i) => {
    const v = job?.stages?.[s.name] || "pending";
    steps[i].dataset.state = v;
    if (v === "done") done++;
    if (v === "running" || v === "error") reach = i;
  });
  if (reach == null) reach = Math.max(0, done - 1);
  $("fill").style.width = n > 1 && job ? `calc(${reach / (n - 1)} * (100% - 100% / ${n}))` : "0";
  $("progPct").textContent = `${n ? Math.round((done / n) * 100) : 0}%`;

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
function renderJobs() {
  $("jobsEmpty").hidden = state.jobs.length > 0;
  $("jobRows").innerHTML = state.jobs
    .map((j) => {
      const chip = {
        done: `<span class="chip ok">Done</span>`,
        failed: `<span class="chip bad">Failed</span>`,
        running: `<span class="chip run">Running</span>`,
        queued: `<span class="chip">Queued</span>`,
      }[j.status];
      const thumb = j.result?.thumbnail
        ? `<img class="thumb" src="/api/jobs/${esc(j.id)}/images/${esc(j.result.thumbnail)}" alt="">`
        : `<span class="thumb"></span>`;
      const current = state.shown?.id === j.id || state.current?.id === j.id;
      return `<tr aria-current="${current}">
        <td><div class="name-cell">${thumb}<div><button class="job-link" type="button" data-id="${esc(j.id)}">${esc(j.name)}</button>
          ${j.error ? `<span class="sub">${esc(j.error)}</span>` : ""}</div></div></td>
        <td class="mono">${fmtBox(j.bbox)}</td><td>${chip}</td>
        <td class="mono">${j.result?.buildings ?? "—"}</td><td class="mono">${duration(j)}</td></tr>`;
    })
    .join("");
}
$("jobRows").addEventListener("click", (e) => {
  const btn = e.target.closest(".job-link");
  if (!btn) return;
  const job = state.jobs.find((j) => j.id === btn.dataset.id);
  if (job) showJob(job);
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
  // Follow a job that starts while nothing else is being watched (e.g. from another tab).
  if (isActive(job) && !(state.current && isActive(state.current) && state.current.id !== job.id)) {
    state.current = job;
  }
  if (state.current?.id === job.id) {
    state.current = job;
    renderProgress(job);
  }
  if (prev && isActive(prev) && !isActive(job)) finishJob(job);
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
    if (active) state.current = active;
    if (state.current) state.current = jobs.find((j) => j.id === state.current.id) || state.current;
    renderProgress(state.current);
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

// ---------- Generate ----------
$("go").addEventListener("click", async () => {
  const err = $("formError");
  err.hidden = true;
  const go = $("go");
  go.disabled = true;
  try {
    const job = await api("/api/jobs", {
      method: "POST",
      body: JSON.stringify({
        bbox: box,
        name: state.placeName || "Custom area",
        remove_obstructions: $("seg").checked,
        seg_model: $("model").value,
        device,
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
    localStorage.setItem("posm-theme", next);
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
    $("formError").textContent = `Can't reach the POSM server: ${e.message}`;
    $("formError").hidden = false;
    $("go").disabled = true;
    return;
  }
  const gpu = $("device").querySelector('[data-v="cuda"]');
  gpu.disabled = !state.system.cuda;
  gpu.title = state.system.cuda ? "" : "No CUDA GPU on this server";
  buildStepper();
  renderStats();
  renderProgress(null);
  connectLive();
})();

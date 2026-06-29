"use strict";
// Manuscript cleanup editor: paint/erase the ink mask over the photo, export.

const qs = new URLSearchParams(location.search);
const ID = qs.get("id");
const DPI = parseInt(qs.get("dpi") || "0", 10) || 0;

const view = document.getElementById("view");
const vctx = view.getContext("2d");
const $ = (s) => document.getElementById(s);

// Offscreen full-resolution layers.
const mask = document.createElement("canvas");   // editable ink (black + alpha)
const mctx = mask.getContext("2d", { willReadFrequently: true });
let photo = null;                                  // reference image (cropped)

const state = { scale: 1, tx: 0, ty: 0, tool: "brush", size: 14,
                showPhoto: true, opac: 0.55, drawing: false, panning: false,
                lastX: 0, lastY: 0, spaceDown: false };
const undoStack = [];

// --------------------------------------------------------------------------- //
init();
async function init() {
  if (!ID) { setStatus("No image id. Open this from the workspace 'Edit' button."); return; }
  const [m, p] = await Promise.all([loadImg("/file/facsimile/" + ID),
                                    loadImg("/file/crop/" + ID).catch(() => null)]);
  mask.width = m.naturalWidth; mask.height = m.naturalHeight;
  mctx.drawImage(m, 0, 0);
  photo = p;
  resizeView(); fit(); bind();
  window.__ed = { mask, mctx, state };   // test/debug hook
}

function loadImg(src) {
  return new Promise((res, rej) => {
    const im = new Image(); im.onload = () => res(im); im.onerror = rej;
    im.src = src + "?t=" + Date.now();
  });
}

// --------------------------------------------------------------------------- //
// View transform + render
// --------------------------------------------------------------------------- //
function resizeView() {
  const r = view.parentElement.getBoundingClientRect();
  view.width = r.width; view.height = r.height;
  render();
}
function fit() {
  const r = view.getBoundingClientRect();
  const s = Math.min(r.width / mask.width, r.height / mask.height) * 0.95;
  state.scale = s;
  state.tx = (r.width - mask.width * s) / 2;
  state.ty = (r.height - mask.height * s) / 2;
  render();
}
function render() {
  vctx.setTransform(1, 0, 0, 1, 0, 0);
  vctx.clearRect(0, 0, view.width, view.height);
  vctx.imageSmoothingEnabled = state.scale < 2;
  vctx.setTransform(state.scale, 0, 0, state.scale, state.tx, state.ty);
  if (state.showPhoto && photo) {
    vctx.globalAlpha = state.opac;
    vctx.drawImage(photo, 0, 0);
  }
  vctx.globalAlpha = 1;
  vctx.drawImage(mask, 0, 0);
}
function toImg(ev) {
  const r = view.getBoundingClientRect();
  return { x: (ev.clientX - r.left - state.tx) / state.scale,
           y: (ev.clientY - r.top - state.ty) / state.scale };
}

// --------------------------------------------------------------------------- //
// Painting
// --------------------------------------------------------------------------- //
function stamp(a, b) {
  mctx.save();
  mctx.lineCap = "round"; mctx.lineJoin = "round";
  mctx.lineWidth = state.size;
  if (state.tool === "eraser") {
    mctx.globalCompositeOperation = "destination-out";
    mctx.strokeStyle = "rgba(0,0,0,1)";
  } else {
    mctx.globalCompositeOperation = "source-over";
    mctx.strokeStyle = "#000000";
  }
  mctx.beginPath(); mctx.moveTo(a.x, a.y); mctx.lineTo(b.x, b.y); mctx.stroke();
  mctx.restore();
}
function pushUndo() {
  try { undoStack.push(mctx.getImageData(0, 0, mask.width, mask.height)); }
  catch (e) { /* ignore */ }
  if (undoStack.length > 6) undoStack.shift();
}
function undo() {
  const img = undoStack.pop();
  if (img) { mctx.putImageData(img, 0, 0); render(); }
}

// --------------------------------------------------------------------------- //
// Events
// --------------------------------------------------------------------------- //
function bind() {
  window.addEventListener("resize", resizeView);
  $("fit").onclick = fit;
  $("undo").onclick = undo;

  document.querySelectorAll(".toolbtn[data-tool]").forEach((b) =>
    b.onclick = () => setTool(b.dataset.tool));
  $("size").oninput = (e) => { state.size = +e.target.value; $("size-out").textContent = e.target.value; };
  $("show-photo").onchange = (e) => { state.showPhoto = e.target.checked; render(); };
  $("opac").oninput = (e) => { state.opac = +e.target.value; $("opac-out").textContent = e.target.value; render(); };
  $("export-png").onclick = exportPng;
  $("export-svg").onclick = exportSvg;

  view.addEventListener("pointerdown", onDown);
  view.addEventListener("pointermove", onMove);
  window.addEventListener("pointerup", onUp);
  view.addEventListener("wheel", onWheel, { passive: false });
  window.addEventListener("keydown", onKey);
  window.addEventListener("keyup", (e) => { if (e.code === "Space") state.spaceDown = false; });
}

function setTool(t) {
  state.tool = t;
  document.querySelectorAll(".toolbtn[data-tool]").forEach((b) =>
    b.classList.toggle("active", b.dataset.tool === t));
  view.style.cursor = t === "pan" ? "grab" : "crosshair";
}

function onDown(ev) {
  view.setPointerCapture(ev.pointerId);
  const panMode = state.tool === "pan" || state.spaceDown || ev.button === 1;
  if (panMode) { state.panning = true; state.lastX = ev.clientX; state.lastY = ev.clientY; return; }
  state.drawing = true; pushUndo();
  const p = toImg(ev); stamp(p, p); render();
}
function onMove(ev) {
  if (state.panning) {
    state.tx += ev.clientX - state.lastX; state.ty += ev.clientY - state.lastY;
    state.lastX = ev.clientX; state.lastY = ev.clientY; render(); return;
  }
  if (!state.drawing) return;
  const p = toImg(ev); stamp({ x: state.px ?? p.x, y: state.py ?? p.y }, p);
  state.px = p.x; state.py = p.y; render();
}
function onUp() { state.drawing = false; state.panning = false; state.px = state.py = null; }

function onWheel(ev) {
  ev.preventDefault();
  const r = view.getBoundingClientRect();
  const cx = ev.clientX - r.left, cy = ev.clientY - r.top;
  const f = ev.deltaY < 0 ? 1.15 : 1 / 1.15;
  const ns = Math.max(0.05, Math.min(40, state.scale * f));
  const k = ns / state.scale;
  state.tx = cx - (cx - state.tx) * k;
  state.ty = cy - (cy - state.ty) * k;
  state.scale = ns; render();
}
function onKey(e) {
  if (e.code === "Space") { state.spaceDown = true; return; }
  if (e.key === "b") setTool("brush");
  else if (e.key === "e") setTool("eraser");
  else if (e.key === "z" && (e.ctrlKey || e.metaKey || true)) undo();
  else if (e.key === "[") { state.size = Math.max(1, state.size - 2); $("size").value = state.size; $("size-out").textContent = state.size; }
  else if (e.key === "]") { state.size = Math.min(120, state.size + 2); $("size").value = state.size; $("size-out").textContent = state.size; }
}

// --------------------------------------------------------------------------- //
// Export
// --------------------------------------------------------------------------- //
function exportPng() {
  mask.toBlob((blob) => {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `text_${ID}.png`; a.click();
    setStatus("Saved Text PNG (filled, transparent background).");
  }, "image/png");
}
async function exportSvg() {
  setStatus("Tracing outline… (a few seconds)");
  const png = mask.toDataURL("image/png");
  const r = await fetch("/api/vectorize", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ id: ID, png, outline: true, dpi: DPI }),
  });
  const j = await r.json();
  if (j.error) { setStatus("SVG error: " + j.error); return; }
  const a = document.createElement("a");
  a.href = j.image; a.download = `outline_${ID}.svg`; a.click();
  setStatus("Saved Outline SVG (contours only, no fill).");
}

function setStatus(t) { $("status").textContent = t; }

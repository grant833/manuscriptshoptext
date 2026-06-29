"use strict";

// Per-side state. Each side: {id, natW, natH, crop, autocrop, facsimileReady}
const sides = { front: null, back: null };
let active = "front";
let activeTab = "original";

const $ = (s) => document.querySelector(s);
const $$ = (s) => Array.from(document.querySelectorAll(s));

const SLIDERS = ["k_weak", "edge_trim", "min_blob", "cm"];

window.addEventListener("DOMContentLoaded", () => {
  bindSlots();
  bindControls();
  bindTabs();
  bindCrop();
  $("#auto-crop").addEventListener("click", applyAutoCrop);
  $("#reset-crop").addEventListener("click", () => { cur() && (cur().crop = null); renderCrop(); maybeLive(); });
  $("#process").addEventListener("click", () => renderTab(activeTab, true));
  $("#run-both").addEventListener("click", runBoth);
  $("#ocr").addEventListener("click", runOcr);
  $("#ink_color").addEventListener("change", maybeLive);
  $("#edit-clean").addEventListener("click", openEditor);
  checkHealth();
});

function cur() { return sides[active]; }

// --------------------------------------------------------------------------- //
// Health
// --------------------------------------------------------------------------- //
async function checkHealth() {
  try {
    const j = await (await fetch("/api/health")).json();
    const pill = $("#ocr-status");
    if (j.ocr_available) {
      pill.textContent = "OCR ready (grc)"; pill.className = "pill pill-ok";
      $("#ocr").disabled = false;
    } else {
      pill.textContent = "OCR off"; pill.className = "pill pill-off";
    }
  } catch { $("#ocr-status").textContent = "OCR unknown"; }
}

// --------------------------------------------------------------------------- //
// Controls
// --------------------------------------------------------------------------- //
function bindControls() {
  SLIDERS.forEach((id) => {
    const el = $("#" + id), out = $("#" + id + "-out");
    const sync = () => { if (out) out.textContent = el.value; };
    el.addEventListener("input", () => { sync(); if (id === "cm") updateDpi(); maybeLive(); });
    sync();
  });
  updateDpi();
}

function widthPx() {
  const s = cur(); if (!s) return 0;
  return s.crop ? s.crop.w : s.natW;
}
function computeDpi() {
  const cm = +$("#cm").value;
  if (!cm || !widthPx()) return 0;
  return Math.round(widthPx() * 2.54 / cm);
}
function updateDpi() {
  const cm = +$("#cm").value;
  $("#cm-out").textContent = cm ? cm.toFixed(1) : "—";
  const dpi = computeDpi();
  $("#dpi-readout").textContent = dpi ? dpi + " DPI" : "unset";
}

function collectParams() {
  const s = cur();
  const p = {
    k_weak: +$("#k_weak").value,
    edge_trim: +$("#edge_trim").value,
    min_blob: +$("#min_blob").value,
    ink_color: $("#ink_color").value,
    min_blob: +$("#min_blob").value,
    dpi: computeDpi(),
  };
  if (s && s.crop) {
    p.crop_x = s.crop.x; p.crop_y = s.crop.y; p.crop_w = s.crop.w; p.crop_h = s.crop.h;
  }
  return p;
}

let liveTimer = null;
function maybeLive() {
  updateDpi();
  if (!cur() || !$("#live").checked) return;
  if (activeTab === "original") return;     // nothing to recompute
  clearTimeout(liveTimer);
  liveTimer = setTimeout(() => renderTab(activeTab, true), 250);
}

// --------------------------------------------------------------------------- //
// Side slots + upload
// --------------------------------------------------------------------------- //
function bindSlots() {
  $$(".side-slot input").forEach((inp) => {
    inp.addEventListener("change", (e) => onUpload(e, inp.closest(".side-slot").dataset.side));
  });
  $$(".side-tab").forEach((t) =>
    t.addEventListener("click", () => { if (!t.disabled) setActive(t.dataset.side); }));
}

async function onUpload(e, side) {
  const file = e.target.files[0]; if (!file) return;
  const fd = new FormData(); fd.append("image", file);
  const slot = document.querySelector(`.side-slot[data-side="${side}"]`);
  slot.querySelector(".slot-status").textContent = "uploading…";
  const j = await (await fetch("/api/upload", { method: "POST", body: fd })).json();
  if (j.error) { alert("Upload failed: " + j.error); return; }

  sides[side] = { id: j.id, natW: j.w, natH: j.h, crop: j.autocrop ? { ...j.autocrop } : null,
                  autocrop: j.autocrop, facsimileReady: false };
  slot.classList.add("loaded");
  slot.querySelector(".slot-status").textContent = `${j.w}×${j.h}`;
  document.querySelector(`.side-tab[data-side="${side}"]`).disabled = false;
  $("#run-both").disabled = false;
  setActive(side);
}

function setActive(side) {
  if (!sides[side]) return;
  active = side;
  $$(".side-tab").forEach((t) => t.classList.toggle("active", t.dataset.side === side));
  const img = $("#img-original");
  img.onload = () => { renderCrop(); };
  img.src = "/file/original/" + sides[side].id + "?t=" + Date.now();
  $("#placeholder").hidden = true;
  updateDpi();
  renderTab(activeTab, false);
}

// --------------------------------------------------------------------------- //
// Tabs / views
// --------------------------------------------------------------------------- //
function bindTabs() {
  $$(".tab").forEach((t) => t.addEventListener("click", () => { activeTab = t.dataset.tab; renderTab(activeTab, false); }));
}

function showStage(name) {
  $("#stage-original").hidden = name !== "original";
  $("#stage-facsimile").hidden = name !== "facsimile";
  $("#stage-svg").hidden = name !== "svg";
  $("#stage-preview").hidden = name !== "preview";
  $$(".tab").forEach((t) => t.classList.toggle("active", t.dataset.tab === name));
}

async function renderTab(name, force) {
  activeTab = name;
  showStage(name);
  const s = cur();
  const dl = $("#download");
  // "Edit & clean" applies to the facsimile result.
  $("#edit-clean").hidden = !(s && name === "facsimile");
  if (name === "original") { dl.hidden = true; renderCrop(); return; }
  if (!s) return;
  if (name === "facsimile") {
    await runFacsimile(force);
    dl.hidden = false; dl.href = "/api/download/facsimile/" + s.id;
    dl.textContent = "⬇ Download PNG";
  } else if (name === "svg") {
    await runSvg(force);
    dl.hidden = false; dl.href = "/api/download/svg/" + s.id;
    dl.textContent = "⬇ Download SVG";
  } else if (name === "preview") {
    dl.hidden = true;
    await runPreview();
  }
}

async function openEditor() {
  const s = cur(); if (!s) return;
  // Make sure the facsimile + cropped reference are generated/cached first.
  if (!s.facsimileReady) await runFacsimile(true);
  const dpi = computeDpi();
  window.open(`/editor?id=${s.id}${dpi ? "&dpi=" + dpi : ""}`, "_blank");
}

async function runSvg(force) {
  const s = cur(); if (!s) return;
  const btn = $("#process"); btn.disabled = true; btn.textContent = "Tracing…";
  try {
    const j = await post("/api/svg", { id: s.id, params: collectParams() });
    if (j.error) { alert("SVG: " + j.error); return; }
    $("#img-svg").src = j.image;
    $("#dims").textContent = "vector (SVG)";
  } finally { btn.disabled = false; btn.textContent = "Process side"; }
}

async function runFacsimile(force) {
  const s = cur(); if (!s) return;
  const btn = $("#process"); btn.disabled = true; btn.textContent = "Processing…";
  try {
    const j = await post("/api/facsimile", { id: s.id, params: collectParams() });
    if (j.error) { alert(j.error); return; }
    $("#img-facsimile").src = j.image;
    s.facsimileReady = true;
    $("#dims").textContent = `${j.w}×${j.h}px` + (computeDpi() ? ` @ ${computeDpi()} DPI` : "");
  } finally { btn.disabled = false; btn.textContent = "Process side"; }
}

async function runPreview() {
  const s = cur(); if (!s) return;
  const j = await post("/api/preview", { id: s.id, params: collectParams() });
  if (j.error) { alert(j.error); return; }
  $("#img-preview").src = j.image;
}

async function runBoth() {
  const order = ["front", "back"].filter((k) => sides[k]);
  for (const side of order) {
    setActive(side);
    activeTab = "facsimile"; showStage("facsimile");
    await runFacsimile(true);
    $("#download").hidden = false;
    $("#download").href = "/api/download/facsimile/" + sides[side].id;
  }
  alert("Done. Facsimile ready for: " + order.join(" + ") +
        ".\nSwitch sides above and use Download PNG for each.");
}

// --------------------------------------------------------------------------- //
// Crop (drag on original)
// --------------------------------------------------------------------------- //
function bindCrop() {
  const wrap = $("#image-wrap");
  let dragging = false, sx = 0, sy = 0;
  wrap.addEventListener("mousedown", (e) => {
    if (!cur() || activeTab !== "original") return;
    dragging = true;
    const r = wrap.getBoundingClientRect(); sx = e.clientX - r.left; sy = e.clientY - r.top;
  });
  window.addEventListener("mousemove", (e) => {
    if (!dragging) return;
    const r = wrap.getBoundingClientRect();
    const cx = Math.max(0, Math.min(e.clientX - r.left, r.width));
    const cy = Math.max(0, Math.min(e.clientY - r.top, r.height));
    const scale = cur().natW / r.width;
    cur().crop = { x: Math.round(Math.min(sx, cx) * scale), y: Math.round(Math.min(sy, cy) * scale),
                   w: Math.round(Math.abs(cx - sx) * scale), h: Math.round(Math.abs(cy - sy) * scale) };
    renderCrop();
  });
  window.addEventListener("mouseup", () => {
    if (!dragging) return; dragging = false;
    if (cur().crop && (cur().crop.w < 10 || cur().crop.h < 10)) cur().crop = null;
    renderCrop(); updateDpi(); maybeLive();
  });
  window.addEventListener("resize", renderCrop);
}

function renderCrop() {
  const box = $("#crop-box"), wrap = $("#image-wrap"), s = cur();
  if (!s || !s.crop) { box.hidden = true; $("#crop-readout").textContent = "no crop"; return; }
  const r = wrap.getBoundingClientRect(), scale = r.width / s.natW;
  box.hidden = false;
  box.style.left = s.crop.x * scale + "px"; box.style.top = s.crop.y * scale + "px";
  box.style.width = s.crop.w * scale + "px"; box.style.height = s.crop.h * scale + "px";
  $("#crop-readout").textContent = `crop ${s.crop.w}×${s.crop.h} @ (${s.crop.x}, ${s.crop.y})`;
}

function applyAutoCrop() {
  const s = cur(); if (!s) return;
  if (s.autocrop) { s.crop = { ...s.autocrop }; renderCrop(); updateDpi(); maybeLive(); }
  else alert("No crop auto-detected — draw one by hand on the Original view.");
}

// --------------------------------------------------------------------------- //
// OCR (optional)
// --------------------------------------------------------------------------- //
async function runOcr() {
  const s = cur(); if (!s) { alert("Upload an image first."); return; }
  const btn = $("#ocr"); btn.disabled = true; btn.textContent = "Reading…";
  try {
    const j = await post("/api/ocr", { id: s.id, params: collectParams() });
    if (j.error) { alert("OCR: " + j.error); return; }
    $("#text").value = "--- OCR draft (UNVERIFIED) ---\n" + j.text;
  } finally { btn.disabled = false; btn.textContent = "Get OCR draft"; }
}

// --------------------------------------------------------------------------- //
async function post(url, body) {
  const r = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  return r.json();
}

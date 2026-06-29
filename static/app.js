"use strict";

const state = {
  id: null,
  natW: 0,        // natural (original) image width in px
  natH: 0,
  crop: null,     // {x, y, w, h} in natural px
};

const $ = (sel) => document.querySelector(sel);

// Slider/checkbox/select ids whose values are sent to the backend.
const PARAM_IDS = ["flatten", "bg_kernel", "clahe", "clahe_clip",
                   "threshold", "block_size", "C", "manual_thresh", "min_blob"];

// --------------------------------------------------------------------------- //
// Init
// --------------------------------------------------------------------------- //

window.addEventListener("DOMContentLoaded", () => {
  bindControls();
  bindTabs();
  bindCrop();
  bindMarkers();
  bindActions();
  checkHealth();
});

async function checkHealth() {
  try {
    const r = await fetch("/api/health");
    const j = await r.json();
    const pill = $("#ocr-status");
    if (j.ocr_available) {
      pill.textContent = "OCR ready (grc)";
      pill.className = "pill pill-ok";
      $("#ocr").disabled = false;
      $("#ocr-note").hidden = false;
    } else {
      pill.textContent = "OCR off (manual only)";
      pill.className = "pill pill-off";
    }
  } catch (e) {
    $("#ocr-status").textContent = "OCR unknown";
  }
}

// --------------------------------------------------------------------------- //
// Controls -> params
// --------------------------------------------------------------------------- //

function bindControls() {
  // live numeric readouts
  PARAM_IDS.forEach((id) => {
    const el = $("#" + id);
    if (!el) return;
    const out = $("#" + id + "-out");
    const sync = () => { if (out) out.textContent = el.value; };
    el.addEventListener("input", () => {
      sync();
      toggleThresholdRows();
      maybeLiveProcess();
    });
    sync();
  });
  toggleThresholdRows();
}

function toggleThresholdRows() {
  const method = $("#threshold").value;
  document.querySelectorAll("[data-when]").forEach((d) => {
    d.hidden = d.getAttribute("data-when") !== method;
  });
}

function collectParams() {
  const p = {
    flatten: $("#flatten").checked,
    bg_kernel: +$("#bg_kernel").value,
    clahe: $("#clahe").checked,
    clahe_clip: +$("#clahe_clip").value,
    threshold: $("#threshold").value,
    block_size: +$("#block_size").value,
    C: +$("#C").value,
    manual_thresh: +$("#manual_thresh").value,
    min_blob: +$("#min_blob").value,
  };
  if (state.crop) {
    p.crop_x = state.crop.x; p.crop_y = state.crop.y;
    p.crop_w = state.crop.w; p.crop_h = state.crop.h;
  }
  return p;
}

let liveTimer = null;
function maybeLiveProcess() {
  if (!state.id || !$("#live").checked) return;
  clearTimeout(liveTimer);
  liveTimer = setTimeout(process, 250);
}

// --------------------------------------------------------------------------- //
// Tabs
// --------------------------------------------------------------------------- //

function bindTabs() {
  document.querySelectorAll(".tab").forEach((t) => {
    t.addEventListener("click", () => showTab(t.dataset.tab));
  });
}
function showTab(name) {
  document.querySelectorAll(".tab").forEach((t) =>
    t.classList.toggle("active", t.dataset.tab === name));
  $("#stage-original").hidden = name !== "original";
  $("#stage-cleaned").hidden = name !== "cleaned";
}

// --------------------------------------------------------------------------- //
// Upload
// --------------------------------------------------------------------------- //

function bindActions() {
  $("#file-input").addEventListener("change", onUpload);
  $("#process").addEventListener("click", process);
  $("#auto-crop").addEventListener("click", applyAutoCrop);
  $("#reset-crop").addEventListener("click", () => { state.crop = null; renderCrop(); maybeLiveProcess(); });
  $("#ocr").addEventListener("click", runOcr);
  $("#save").addEventListener("click", save);
  // re-run on checkbox/select change too
  ["flatten", "clahe", "threshold"].forEach((id) =>
    $("#" + id).addEventListener("change", maybeLiveProcess));
}

async function onUpload(e) {
  const file = e.target.files[0];
  if (!file) return;
  const fd = new FormData();
  fd.append("image", file);
  const r = await fetch("/api/upload", { method: "POST", body: fd });
  const j = await r.json();
  if (j.error) { alert("Upload failed: " + j.error); return; }

  state.id = j.id; state.natW = j.w; state.natH = j.h;
  state.crop = null;
  state.autocrop = j.autocrop || null;

  const img = $("#img-original");
  img.onload = () => { renderCrop(); process(); };
  img.src = "/file/original/" + j.id + "?t=" + Date.now();
  $("#placeholder").hidden = true;
  showTab("original");
}

// --------------------------------------------------------------------------- //
// Crop selection (drag a box on the original image)
// --------------------------------------------------------------------------- //

function bindCrop() {
  const wrap = $("#image-wrap");
  let dragging = false, sx = 0, sy = 0;

  wrap.addEventListener("mousedown", (e) => {
    if (!state.id) return;
    dragging = true;
    const r = wrap.getBoundingClientRect();
    sx = e.clientX - r.left; sy = e.clientY - r.top;
  });
  window.addEventListener("mousemove", (e) => {
    if (!dragging) return;
    const r = wrap.getBoundingClientRect();
    const cx = Math.max(0, Math.min(e.clientX - r.left, r.width));
    const cy = Math.max(0, Math.min(e.clientY - r.top, r.height));
    const x = Math.min(sx, cx), y = Math.min(sy, cy);
    const w = Math.abs(cx - sx), h = Math.abs(cy - sy);
    const scale = state.natW / r.width;   // displayed -> natural
    state.crop = { x: Math.round(x * scale), y: Math.round(y * scale),
                   w: Math.round(w * scale), h: Math.round(h * scale) };
    renderCrop();
  });
  window.addEventListener("mouseup", () => {
    if (!dragging) return;
    dragging = false;
    if (state.crop && (state.crop.w < 10 || state.crop.h < 10)) state.crop = null;
    renderCrop();
    maybeLiveProcess();
  });
}

function renderCrop() {
  const box = $("#crop-box");
  const wrap = $("#image-wrap");
  const r = wrap.getBoundingClientRect();
  if (!state.crop || !state.natW) {
    box.hidden = true;
    $("#crop-readout").textContent = "no crop";
    return;
  }
  const scale = r.width / state.natW;   // natural -> displayed
  box.hidden = false;
  box.style.left = state.crop.x * scale + "px";
  box.style.top = state.crop.y * scale + "px";
  box.style.width = state.crop.w * scale + "px";
  box.style.height = state.crop.h * scale + "px";
  const c = state.crop;
  $("#crop-readout").textContent = `crop ${c.w}×${c.h} @ (${c.x}, ${c.y})`;
}

function applyAutoCrop() {
  if (state.autocrop) { state.crop = { ...state.autocrop }; renderCrop(); maybeLiveProcess(); }
  else alert("No crop could be auto-detected — draw one by hand.");
}

window.addEventListener("resize", renderCrop);

// --------------------------------------------------------------------------- //
// Process / OCR / Save
// --------------------------------------------------------------------------- //

async function process() {
  if (!state.id) return;
  const btn = $("#process");
  btn.disabled = true; btn.textContent = "Processing…";
  try {
    const r = await fetch("/api/process", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id: state.id, params: collectParams() }),
    });
    const j = await r.json();
    if (j.error) { alert(j.error); return; }
    $("#img-cleaned").src = j.image;
    showTab("cleaned");
  } finally {
    btn.disabled = false; btn.textContent = "Process";
  }
}

async function runOcr() {
  if (!state.id) return;
  if ($("#text").value.trim() &&
      !confirm("Append an OCR draft below your current text?")) return;
  const btn = $("#ocr");
  btn.disabled = true; btn.textContent = "Reading…";
  try {
    const r = await fetch("/api/ocr", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id: state.id, params: collectParams() }),
    });
    const j = await r.json();
    if (j.error) { alert("OCR: " + j.error); return; }
    const ta = $("#text");
    const banner = "\n\n--- OCR draft (UNVERIFIED — correct against image) ---\n";
    ta.value = ta.value.trim() ? ta.value + banner + j.text : j.text;
  } finally {
    btn.disabled = false; btn.textContent = "OCR draft";
  }
}

async function save() {
  if (!state.id) { alert("Upload an image first."); return; }
  const r = await fetch("/api/save", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ id: state.id, text: $("#text").value }),
  });
  const j = await r.json();
  if (j.error) { alert(j.error); return; }
  $("#save-status").textContent = "saved ✓";
  const dl = $("#download");
  dl.hidden = false;
  dl.href = "/api/download/" + state.id;
  setTimeout(() => ($("#save-status").textContent = ""), 2500);
}

// --------------------------------------------------------------------------- //
// Leiden marker toolbar
// --------------------------------------------------------------------------- //

function bindMarkers() {
  document.querySelectorAll(".marker-bar button").forEach((b) => {
    b.addEventListener("click", () => {
      const ta = $("#text");
      const s = ta.selectionStart, e = ta.selectionEnd;
      const sel = ta.value.slice(s, e);
      let insert, caret;
      if (b.dataset.wrap) {
        const [pre, post] = b.dataset.wrap.split("|");
        insert = pre + sel + post;
        caret = s + pre.length + sel.length;
      } else {
        insert = b.dataset.ins;
        caret = s + insert.length;
      }
      ta.setRangeText(insert, s, e, "end");
      ta.focus();
      ta.selectionStart = ta.selectionEnd = caret;
    });
  });
}

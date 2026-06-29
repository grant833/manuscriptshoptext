"""
Manuscript facsimile workspace -- local web app.

Run:  python app.py   ->  http://127.0.0.1:5000

The deliverable is a faithful FACSIMILE of the surviving ink on a transparent
background (print-ready, 1:1), produced by deterministic image processing only.
No text recognition, no generation, nothing invented.

Endpoints
  GET  /                       the single-page workspace
  GET  /api/health             reports whether optional OCR is available
  POST /api/upload             upload an image -> {id, w, h, autocrop}
  POST /api/facsimile          {id, params} -> transparent PNG (base64) preview
  POST /api/preview            {id, params} -> flat black/white PNG (tuning aid)
  POST /api/ocr                {id, params} -> rough OCR draft (UNVERIFIED)
  GET  /api/download/facsimile/<id>   download the saved transparent PNG
  GET  /file/<kind>/<id>       serve original / facsimile / preview images
"""

from __future__ import annotations

import base64
import io
import os
import uuid

import cv2
import numpy as np
from PIL import Image
from flask import (Flask, jsonify, request, send_file, send_from_directory,
                   abort)

from manuscript import ocr
from manuscript.pipeline import (Params, process, extract_ink_rgba,
                                 extract_ink_svg, svg_available,
                                 detect_manuscript_bbox, encode_png)

BASE = os.path.dirname(os.path.abspath(__file__))
UPLOADS = os.path.join(BASE, "uploads")
OUTPUTS = os.path.join(BASE, "outputs")
STATIC = os.path.join(BASE, "static")
os.makedirs(UPLOADS, exist_ok=True)
os.makedirs(OUTPUTS, exist_ok=True)

ALLOWED = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
MAX_BYTES = 60 * 1024 * 1024

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = MAX_BYTES


# --------------------------------------------------------------------------- #
# Storage helpers
# --------------------------------------------------------------------------- #

def _upload_path(file_id: str) -> str | None:
    for ext in ALLOWED:
        p = os.path.join(UPLOADS, file_id + ext)
        if os.path.exists(p):
            return p
    return None


def _read_bgr(path: str) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    return img


def _rgba_to_png_bytes(rgba: np.ndarray, dpi: int = 0) -> bytes:
    """Encode an RGBA numpy image to PNG bytes, embedding DPI when given."""
    im = Image.fromarray(rgba, "RGBA")
    buf = io.BytesIO()
    if dpi and dpi > 0:
        im.save(buf, format="PNG", dpi=(dpi, dpi))
    else:
        im.save(buf, format="PNG")
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #

@app.get("/")
def index():
    return send_from_directory(STATIC, "index.html")


@app.get("/static/<path:fn>")
def static_files(fn):
    return send_from_directory(STATIC, fn)


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #

@app.get("/api/health")
def health():
    return jsonify({"ocr_available": ocr.available(),
                    "svg_available": svg_available()})


@app.post("/api/upload")
def upload():
    if "image" not in request.files:
        return jsonify({"error": "no image field"}), 400
    f = request.files["image"]
    ext = os.path.splitext(f.filename or "")[1].lower()
    if ext not in ALLOWED:
        return jsonify({"error": f"unsupported type {ext!r}"}), 400

    file_id = uuid.uuid4().hex
    path = os.path.join(UPLOADS, file_id + ext)
    f.save(path)
    try:
        bgr = _read_bgr(path)
    except ValueError:
        os.remove(path)
        return jsonify({"error": "could not read image"}), 400

    h, w = bgr.shape[:2]
    return jsonify({"id": file_id, "w": int(w), "h": int(h),
                    "autocrop": detect_manuscript_bbox(bgr)})


def _load(payload: dict):
    file_id = (payload or {}).get("id", "")
    path = _upload_path(file_id)
    if not path:
        abort(404, "unknown image id")
    return file_id, _read_bgr(path), Params.from_dict(payload.get("params", {}))


@app.post("/api/facsimile")
def api_facsimile():
    file_id, bgr, params = _load(request.get_json(force=True))
    rgba = extract_ink_rgba(bgr, params)
    png = _rgba_to_png_bytes(rgba, params.dpi)
    # cache for download
    with open(os.path.join(OUTPUTS, file_id + "_facsimile.png"), "wb") as fh:
        fh.write(png)
    b64 = base64.b64encode(png).decode("ascii")
    return jsonify({"image": "data:image/png;base64," + b64,
                    "w": int(rgba.shape[1]), "h": int(rgba.shape[0])})


@app.post("/api/svg")
def api_svg():
    if not svg_available():
        return jsonify({"error": "SVG tracer (potrace) not installed"}), 503
    file_id, bgr, params = _load(request.get_json(force=True))
    try:
        svg = extract_ink_svg(bgr, params)
    except Exception as e:  # pragma: no cover
        return jsonify({"error": str(e)}), 500
    with open(os.path.join(OUTPUTS, file_id + "_facsimile.svg"), "w",
              encoding="utf-8") as fh:
        fh.write(svg)
    b64 = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return jsonify({"svg": svg, "image": "data:image/svg+xml;base64," + b64})


@app.post("/api/preview")
def api_preview():
    file_id, bgr, params = _load(request.get_json(force=True))
    flat = process(bgr, params)
    b64 = base64.b64encode(encode_png(flat)).decode("ascii")
    return jsonify({"image": "data:image/png;base64," + b64})


@app.post("/api/ocr")
def api_ocr():
    if not ocr.available():
        return jsonify({"error": "OCR engine not installed"}), 503
    _, bgr, params = _load(request.get_json(force=True))
    try:
        text = ocr.draft(process(bgr, params))
    except Exception as e:  # pragma: no cover
        return jsonify({"error": str(e)}), 500
    return jsonify({"text": text})


@app.get("/api/download/facsimile/<file_id>")
def download_facsimile(file_id):
    p = os.path.join(OUTPUTS, file_id + "_facsimile.png")
    if not os.path.exists(p):
        abort(404)
    return send_file(p, as_attachment=True,
                     download_name=f"facsimile_{file_id}.png",
                     mimetype="image/png")


@app.get("/api/download/svg/<file_id>")
def download_svg(file_id):
    p = os.path.join(OUTPUTS, file_id + "_facsimile.svg")
    if not os.path.exists(p):
        abort(404)
    return send_file(p, as_attachment=True,
                     download_name=f"facsimile_{file_id}.svg",
                     mimetype="image/svg+xml")


@app.get("/file/<kind>/<file_id>")
def serve_file(kind, file_id):
    if kind == "original":
        p = _upload_path(file_id)
    elif kind == "facsimile":
        p = os.path.join(OUTPUTS, file_id + "_facsimile.png")
        p = p if os.path.exists(p) else None
    else:
        abort(404)
    if not p:
        abort(404)
    return send_file(p)


if __name__ == "__main__":
    print("Manuscript facsimile workspace -> http://127.0.0.1:5000")
    print("OCR available:", ocr.available())
    app.run(host="127.0.0.1", port=5000, debug=True)

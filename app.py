"""
Manuscript text-extraction workspace -- local web app.

Run:  python app.py   ->  http://127.0.0.1:5000

Endpoints
  GET  /                     the single-page workspace
  GET  /api/health           reports whether OCR is available
  POST /api/upload           upload an image, returns {id, w, h, autocrop}
  POST /api/process          {id, params} -> cleaned PNG (base64) for preview
  POST /api/ocr              {id, params} -> rough OCR draft (unverified)
  POST /api/save             {id, text}   -> writes outputs/<id>.txt
  GET  /api/download/<id>    download the saved transcription
  GET  /file/<kind>/<id>     serve original / processed images
"""

from __future__ import annotations

import base64
import io
import os
import uuid

import cv2
import numpy as np
from flask import (Flask, jsonify, request, send_file, send_from_directory,
                   abort)

from manuscript import ocr
from manuscript.pipeline import Params, process, detect_manuscript_bbox, encode_png

BASE = os.path.dirname(os.path.abspath(__file__))
UPLOADS = os.path.join(BASE, "uploads")
OUTPUTS = os.path.join(BASE, "outputs")
STATIC = os.path.join(BASE, "static")
os.makedirs(UPLOADS, exist_ok=True)
os.makedirs(OUTPUTS, exist_ok=True)

ALLOWED = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
MAX_BYTES = 40 * 1024 * 1024  # 40 MB

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
    return jsonify({"ocr_available": ocr.available()})


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
    return jsonify({
        "id": file_id,
        "w": int(w),
        "h": int(h),
        "autocrop": detect_manuscript_bbox(bgr),
    })


def _load_and_process(payload: dict) -> np.ndarray:
    file_id = (payload or {}).get("id", "")
    path = _upload_path(file_id)
    if not path:
        abort(404, "unknown image id")
    bgr = _read_bgr(path)
    params = Params.from_dict(payload.get("params", {}))
    cleaned = process(bgr, params)
    # cache the processed result so download/ocr reuse it
    cv2.imwrite(os.path.join(OUTPUTS, file_id + "_processed.png"), cleaned)
    return cleaned


@app.post("/api/process")
def api_process():
    cleaned = _load_and_process(request.get_json(force=True))
    b64 = base64.b64encode(encode_png(cleaned)).decode("ascii")
    return jsonify({"image": "data:image/png;base64," + b64})


@app.post("/api/ocr")
def api_ocr():
    if not ocr.available():
        return jsonify({"error": "OCR engine not installed"}), 503
    cleaned = _load_and_process(request.get_json(force=True))
    try:
        text = ocr.draft(cleaned)
    except Exception as e:  # pragma: no cover
        return jsonify({"error": str(e)}), 500
    return jsonify({"text": text})


@app.post("/api/save")
def api_save():
    data = request.get_json(force=True)
    file_id = data.get("id", "")
    if not _upload_path(file_id):
        return jsonify({"error": "unknown image id"}), 404
    text = data.get("text", "")
    out = os.path.join(OUTPUTS, file_id + ".txt")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(text)
    return jsonify({"ok": True, "path": os.path.relpath(out, BASE)})


@app.get("/api/download/<file_id>")
def api_download(file_id):
    out = os.path.join(OUTPUTS, file_id + ".txt")
    if not os.path.exists(out):
        abort(404)
    return send_file(out, as_attachment=True,
                     download_name=f"transcription_{file_id}.txt")


@app.get("/file/<kind>/<file_id>")
def serve_file(kind, file_id):
    if kind == "original":
        p = _upload_path(file_id)
    elif kind == "processed":
        p = os.path.join(OUTPUTS, file_id + "_processed.png")
        p = p if os.path.exists(p) else None
    else:
        abort(404)
    if not p:
        abort(404)
    return send_file(p)


if __name__ == "__main__":
    print("Manuscript workspace -> http://127.0.0.1:5000")
    print("OCR available:", ocr.available())
    app.run(host="127.0.0.1", port=5000, debug=True)

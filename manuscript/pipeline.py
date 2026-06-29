"""
Deterministic manuscript image pipeline.

NO machine learning, NO text recognition, NO text generation. Every step is a
classic, reproducible image-processing operation. We never invent, reconstruct,
or "improve" anything -- we only isolate the ink that is physically present on
the papyrus.

Primary output (`extract_ink_rgba`): a faithful FACSIMILE of the surviving ink,
floating on a fully transparent background, ready to be printed 1:1 onto blank
papyrus. Fading, broken strokes, smudges and gaps are preserved because ink is
carried as a *soft* alpha channel -- faint ink stays faintly transparent rather
than being forced to solid black. Where there is no ink, the output is empty.

Secondary output (`process`): a flat black-on-white preview, useful for eyeing
contrast while tuning.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
# Parameters
# --------------------------------------------------------------------------- #

@dataclass
class Params:
    """All tunable knobs. Defaults aim at brown-papyrus majuscule hands."""

    # Crop to the manuscript region (pixels in the ORIGINAL image). None => none.
    crop_x: Optional[int] = None
    crop_y: Optional[int] = None
    crop_w: Optional[int] = None
    crop_h: Optional[int] = None

    # --- Ink extraction (transparent facsimile) ---------------------------- #
    # Size of the morphological background estimate (removes ink to reveal the
    # bare papyrus tone). Larger = copes with bigger lighting gradients.
    bg_kernel: int = 61
    # Adaptive-threshold window + sensitivity for separating ink from the
    # background-normalised papyrus. Larger C = only clearer ink kept. These are
    # given at a 1800px-wide reference and auto-scaled to the real image size.
    block_size: int = 51
    C: int = 22
    # Brown-rejection (OPTIONAL, off by default = 0): carbon ink soaked into
    # brown papyrus often reads as brown as the papyrus, so a chroma gate tends
    # to cut real ink. Set > 0 to fade out pixels browner than this LAB chroma.
    chroma_gate: float = 0.0
    chroma_soft: float = 6.0
    # Colour of the rendered ink: "black" (default), "original" (sampled from the
    # photo), or "sepia" (a dark brown). Fading is carried by the alpha channel.
    ink_color: str = "black"
    # Remove isolated ink specks smaller than this many pixels (at the 1800px
    # reference, auto-scaled). Clears papyrus-fibre flecks; real strokes survive.
    min_blob: int = 50

    # --- Output scale ------------------------------------------------------ #
    # DPI to embed so the PNG prints at the correct physical size. 0 = unset.
    dpi: int = 0

    # --- Binary preview only ('process') ----------------------------------- #
    clahe: bool = True
    clahe_clip: float = 2.0
    threshold: str = "adaptive"
    manual_thresh: int = 127

    @staticmethod
    def from_dict(d: dict) -> "Params":
        p = Params()
        for k, v in (d or {}).items():
            if hasattr(p, k) and v is not None:
                setattr(p, k, v)
        return p

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _odd(n: int, lo: int = 3) -> int:
    n = int(n)
    if n < lo:
        n = lo
    return n if n % 2 else n + 1


def _apply_crop(bgr: np.ndarray, p: Params) -> np.ndarray:
    if None in (p.crop_x, p.crop_y, p.crop_w, p.crop_h):
        return bgr
    H, W = bgr.shape[:2]
    x = max(0, min(int(p.crop_x), W - 1))
    y = max(0, min(int(p.crop_y), H - 1))
    w = max(1, min(int(p.crop_w), W - x))
    h = max(1, min(int(p.crop_h), H - y))
    return bgr[y:y + h, x:x + w]


def detect_manuscript_bbox(bgr: np.ndarray) -> Optional[dict]:
    """
    Best-effort auto-crop: find the papyrus leaf and exclude the bright mounting
    mat and the saturated Kodak colour-reference card. Returns {x,y,w,h} in
    original-image pixels, or None. Always confirmable/adjustable in the UI.
    """
    h, w = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sat, val = hsv[:, :, 1], hsv[:, :, 2]
    mask = ((val < 200) & (sat < 90)).astype(np.uint8) * 255   # not-mat & not-card
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35)))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    biggest = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(biggest) < 0.04 * w * h:
        return None
    x, y, bw, bh = cv2.boundingRect(biggest)
    pad = int(0.01 * max(w, h))
    x, y = max(0, x - pad), max(0, y - pad)
    bw, bh = min(w - x, bw + 2 * pad), min(h - y, bh + 2 * pad)
    return {"x": int(x), "y": int(y), "w": int(bw), "h": int(bh)}


# --------------------------------------------------------------------------- #
# Primary: ink -> transparent facsimile
# --------------------------------------------------------------------------- #

def extract_ink_rgba(bgr: np.ndarray, p: Params) -> np.ndarray:
    """
    Isolate the ink and return an RGBA image (uint8, channels R,G,B,A) on a
    fully transparent background. Nothing is recognised, reconstructed or
    invented -- only ink that is physically present is kept.

    Method (all deterministic), tuned against real papyrus photos:
      1. Background-normalise the CIELAB lightness so uneven lighting, the dark
         frayed edges and the papyrus fibre tone are flattened out.
      2. Local adaptive threshold -> a clean ink mask that ignores gradual
         darkening (edges / fold) and keeps only locally-dark strokes.
      3. Brown-rejection chroma gate: carbon ink is near-neutral, papyrus fibre
         is brown, so fade out the browner pixels. Faded/brown ink survives as
         partial transparency rather than being dropped or forced solid.
      4. Despeckle stray fibre flecks (real strokes are far larger).
    """
    img = _apply_crop(bgr, p)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0]

    # Auto-scale pixel-sized parameters by the cropped leaf width (a proxy for
    # letter size) against a ~1000px reference, so the same defaults work at any
    # resolution -- a bigger photo of a similar leaf gets proportionally bigger
    # kernels.
    scale = max(0.25, img.shape[1] / 1000.0)

    # 1. Background-normalise lightness (close ink away, then divide).
    k = _odd(round(p.bg_kernel * scale), lo=9)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    bg = cv2.morphologyEx(L.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    bg = bg.astype(np.float32) + 1e-3
    norm = np.clip(L / bg * 255.0, 0, 255).astype(np.uint8)

    # 2. Local adaptive threshold -> ink mask (1.0 where ink).
    mask = cv2.adaptiveThreshold(norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV,
                                 _odd(round(p.block_size * scale), 3),
                                 int(p.C)).astype(np.float32) / 255.0

    # 3. Brown-rejection (optional): fade out pixels browner than the gate.
    if p.chroma_gate and p.chroma_gate > 0:
        chroma = np.hypot(lab[:, :, 1] - 128.0, lab[:, :, 2] - 128.0)
        c_lo = float(p.chroma_gate)
        c_hi = c_lo + max(0.1, float(p.chroma_soft))
        mask = mask * np.clip((c_hi - chroma) / (c_hi - c_lo), 0.0, 1.0)
    alpha = mask

    # 4. Despeckle fibre flecks (area scales with resolution).
    if p.min_blob and p.min_blob > 0:
        binar = (alpha > 0.25).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(binar, 8)
        keep = np.ones(n, dtype=bool)
        keep[0] = False
        min_area = int(p.min_blob * scale * scale)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < min_area:
                keep[i] = False
        alpha = alpha * keep[labels]

    # Light feather so stroke edges are not jagged, then to 0..255.
    alpha8 = cv2.GaussianBlur((alpha * 255.0).astype(np.uint8), (0, 0),
                              max(0.4, 0.6 * scale))

    # Ink colour (fading carried by alpha).
    mode = (p.ink_color or "black").lower()
    if mode == "original":
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif mode == "sepia":
        rgb = np.zeros_like(img)
        rgb[:] = (60, 40, 25)            # R,G,B dark brown
    else:                                # black
        rgb = np.zeros_like(img)

    return np.dstack([rgb, alpha8]).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Secondary: flat black-on-white preview
# --------------------------------------------------------------------------- #

def process(bgr: np.ndarray, p: Params) -> np.ndarray:
    """Flat black-on-white binary preview (single channel)."""
    img = _apply_crop(bgr, p)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    k = _odd(p.bg_kernel, lo=3)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    gray = cv2.divide(gray, bg, scale=255)

    if p.clahe:
        gray = cv2.createCLAHE(clipLimit=max(0.1, float(p.clahe_clip)),
                               tileGridSize=(8, 8)).apply(gray)

    method = (p.threshold or "adaptive").lower()
    if method == "otsu":
        _, out = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == "manual":
        _, out = cv2.threshold(gray, int(p.manual_thresh), 255, cv2.THRESH_BINARY)
    else:
        out = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, _odd(p.block_size, 3), int(p.C))
    return out


def auto_params(bgr: np.ndarray) -> Params:
    """
    Hands-off defaults for unattended processing: best-effort auto-crop to the
    leaf plus the standard ink-extraction settings. Used by the drop-folder
    watcher so front/back images can be processed with no interaction.
    """
    p = Params(ink_color="black")
    bbox = detect_manuscript_bbox(bgr)
    if bbox:
        p.crop_x, p.crop_y = bbox["x"], bbox["y"]
        p.crop_w, p.crop_h = bbox["w"], bbox["h"]
    return p


def encode_png(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return buf.tobytes()

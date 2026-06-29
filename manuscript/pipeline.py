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
    bg_kernel: int = 51
    # Ink that is darker than the local background by less than this fraction is
    # treated as papyrus texture and dropped. Keep LOW to preserve faint ink.
    ink_floor: float = 0.10
    # Contrast of the alpha ramp above the floor. Higher = ink becomes opaque
    # sooner (less translucency); lower = gentler fading preserved.
    ink_gain: float = 4.0
    # Colour of the rendered ink: "original" (sampled from the photo, maximally
    # faithful), "black", or "sepia" (a dark brown).
    ink_color: str = "original"
    # Remove isolated ink specks smaller than this many pixels (0 = keep all).
    # OFF by default so faint / broken strokes are never silently erased.
    min_blob: int = 0

    # --- Output scale ------------------------------------------------------ #
    # DPI to embed so the PNG prints at the correct physical size. 0 = unset.
    dpi: int = 0

    # --- Binary preview only ('process') ----------------------------------- #
    clahe: bool = True
    clahe_clip: float = 2.0
    threshold: str = "adaptive"
    block_size: int = 31
    C: int = 15
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


def _despeckle_alpha(alpha8: np.ndarray, min_blob: int) -> np.ndarray:
    """Drop connected ink regions smaller than min_blob pixels."""
    mask = (alpha8 > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    keep = np.ones(n, dtype=bool)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < int(min_blob):
            keep[i] = False
    return np.where(keep[labels], alpha8, 0).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Primary: ink -> transparent facsimile
# --------------------------------------------------------------------------- #

def extract_ink_rgba(bgr: np.ndarray, p: Params) -> np.ndarray:
    """
    Isolate the ink and return an RGBA image (uint8, channels R,G,B,A) with a
    fully transparent background. Ink darkness becomes alpha, so fading and
    ambiguous marks are preserved as partial transparency -- nothing is forced,
    nothing is invented.
    """
    img = _apply_crop(bgr, p)

    # Work in CIELAB lightness: ink is darker than papyrus regardless of hue.
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0].astype(np.float32)

    # Background = papyrus lightness with the ink "closed" away.
    k = _odd(p.bg_kernel, lo=9)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    bg = cv2.morphologyEx(lab[:, :, 0], cv2.MORPH_CLOSE, kernel)
    bg = cv2.GaussianBlur(bg, (0, 0), max(1.0, k / 3.0)).astype(np.float32) + 1e-3

    # How much darker than the local papyrus each pixel is (0..1).
    ink_frac = np.clip((bg - L) / bg, 0.0, 1.0)

    # Soft ramp: below the floor -> transparent (papyrus texture); above it,
    # ramp up gently so faint ink stays translucent.
    floor = float(p.ink_floor)
    gain = max(0.1, float(p.ink_gain))
    alpha = np.clip((ink_frac - floor) * gain, 0.0, 1.0)
    alpha8 = (alpha * 255.0).astype(np.uint8)

    if p.min_blob and p.min_blob > 0:
        alpha8 = _despeckle_alpha(alpha8, p.min_blob)

    # Ink colour.
    mode = (p.ink_color or "original").lower()
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


def encode_png(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return buf.tobytes()

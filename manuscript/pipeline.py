"""
Deterministic manuscript image-cleaning pipeline.

NO machine learning, NO text generation. Every step here is a classic,
reproducible image-processing operation. Where ink is lost or broken on the
papyrus, the output simply stays blank in that spot -- damage is shown exactly
as it is, never "filled in".

The pipeline turns a photograph of a manuscript leaf into a faithful
black-on-white rendering of the surviving ink, which can then be transcribed by
a human in the workspace UI.
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
    """All tunable knobs. Sensible defaults aim at brown-papyrus majuscules."""

    # Crop to the manuscript region (pixels in the ORIGINAL image). None => no crop.
    crop_x: Optional[int] = None
    crop_y: Optional[int] = None
    crop_w: Optional[int] = None
    crop_h: Optional[int] = None

    # Background flattening (removes uneven lighting / papyrus fibre tone).
    flatten: bool = True
    bg_kernel: int = 41          # size of the morphological background estimate

    # Local contrast boost (CLAHE).
    clahe: bool = True
    clahe_clip: float = 2.0

    # Thresholding: "adaptive" | "otsu" | "manual"
    threshold: str = "adaptive"
    block_size: int = 31         # adaptive: odd window size
    C: int = 15                  # adaptive: subtracted constant
    manual_thresh: int = 127     # manual: 0-255 cutoff

    # Despeckle: drop ink blobs smaller than this many pixels. 0 = keep everything.
    # Kept OFF by default so faint / broken strokes are never silently erased.
    min_blob: int = 0

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
    """Force an odd value >= lo (required by several OpenCV kernels)."""
    n = int(n)
    if n < lo:
        n = lo
    if n % 2 == 0:
        n += 1
    return n


def detect_manuscript_bbox(bgr: np.ndarray) -> Optional[dict]:
    """
    Best-effort auto-crop suggestion: find the papyrus leaf and exclude the
    white mounting mat and the saturated colour-reference card.

    This is only a *suggestion* shown in the UI -- the user always confirms or
    adjusts it. Returns {x, y, w, h} in original-image pixels, or None.
    """
    h, w = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    # Papyrus: clearly darker/tanner than the bright mat, but not the vivid,
    # highly-saturated patches of the Kodak colour card.
    not_mat = val < 200            # exclude bright white mat
    not_card = sat < 90            # exclude vivid colour patches
    mask = (not_mat & not_card).astype(np.uint8) * 255

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35)))

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    biggest = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(biggest) < 0.04 * w * h:   # too small to be the leaf
        return None
    x, y, bw, bh = cv2.boundingRect(biggest)

    # Pad slightly so we don't shave edge strokes.
    pad = int(0.01 * max(w, h))
    x = max(0, x - pad)
    y = max(0, y - pad)
    bw = min(w - x, bw + 2 * pad)
    bh = min(h - y, bh + 2 * pad)
    return {"x": int(x), "y": int(y), "w": int(bw), "h": int(bh)}


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #

def process(bgr: np.ndarray, p: Params) -> np.ndarray:
    """
    Run the cleaning pipeline. Input: BGR image (as read by cv2).
    Output: single-channel uint8 black-on-white image (0 = ink, 255 = blank).
    """
    img = bgr

    # 1. Crop ----------------------------------------------------------------
    if None not in (p.crop_x, p.crop_y, p.crop_w, p.crop_h):
        H, W = img.shape[:2]
        x = max(0, min(int(p.crop_x), W - 1))
        y = max(0, min(int(p.crop_y), H - 1))
        wd = max(1, min(int(p.crop_w), W - x))
        ht = max(1, min(int(p.crop_h), H - y))
        img = img[y:y + ht, x:x + wd]

    # 2. Grayscale -----------------------------------------------------------
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. Background flattening (even out lighting & fibre tone) ---------------
    if p.flatten:
        k = _odd(p.bg_kernel, lo=3)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        # Closing estimates the background (text removed); divide to flatten.
        bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        gray = cv2.divide(gray, bg, scale=255)

    # 4. Local contrast ------------------------------------------------------
    if p.clahe:
        clahe = cv2.createCLAHE(clipLimit=max(0.1, float(p.clahe_clip)),
                                tileGridSize=(8, 8))
        gray = clahe.apply(gray)

    # 5. Threshold to black-on-white ----------------------------------------
    method = (p.threshold or "adaptive").lower()
    if method == "otsu":
        _, binimg = cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == "manual":
        _, binimg = cv2.threshold(gray, int(p.manual_thresh), 255,
                                  cv2.THRESH_BINARY)
    else:  # adaptive (default)
        binimg = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
            _odd(p.block_size, lo=3), int(p.C))

    # 6. Optional despeckle --------------------------------------------------
    # Removes isolated specks ONLY if the user opts in (min_blob > 0). Off by
    # default so genuine but faint ink is never discarded.
    if p.min_blob and p.min_blob > 0:
        ink = (binimg == 0).astype(np.uint8)  # ink pixels = 1
        n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
        keep = np.ones(n, dtype=bool)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < int(p.min_blob):
                keep[i] = False
        cleaned_ink = keep[labels]
        binimg = np.where(cleaned_ink, 0, 255).astype(np.uint8)

    return binimg


def encode_png(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return buf.tobytes()

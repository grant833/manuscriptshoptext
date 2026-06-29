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
import os
import re
import shutil
import subprocess
import tempfile
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
    # Isolate the writing area: detect the papyrus leaf and erode inward by this
    # fraction of the width to drop the ragged, fibrous edge.
    edge_trim: float = 0.012
    # Two-level (hysteresis) Sauvola thresholding. The window auto-scales with
    # the leaf width. `k_weak` is the permissive pass that captures complete
    # faded strokes; `k_strong` is the strict pass that seeds them. Only weak
    # regions containing a strong seed survive -> complete strokes, less noise.
    k_weak: float = 0.06
    k_strong: float = 0.30
    # Colour of the rendered ink: "black" (default), "original", or "sepia".
    ink_color: str = "black"
    # Remove isolated ink specks smaller than this many pixels (at a 2150px-wide
    # leaf reference, auto-scaled). Clears papyrus-fibre flecks.
    min_blob: int = 150

    # --- Output scale ------------------------------------------------------ #
    # DPI to embed so the PNG/SVG prints at the correct physical size. 0 = unset.
    dpi: int = 0

    # --- SVG vector export ------------------------------------------------- #
    # Alpha cutoff (0-255) for which ink becomes a solid vector shape. Higher =
    # only firmer ink is vectorised.
    svg_threshold: int = 90

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

def leaf_interior_mask(bgr: np.ndarray, edge_trim: float = 0.012) -> np.ndarray:
    """
    Mask of the papyrus leaf's interior: the brown leaf with the bright mat and
    saturated colour card excluded, holes filled, then eroded inward by
    `edge_trim` of the width to drop the ragged, fibrous edge. uint8 0/255.
    """
    H, W = bgr.shape[:2]
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0].astype(int)
    b = lab[:, :, 2].astype(int) - 128
    S = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 1].astype(int)
    leaf = ((L < 205) & (b > 6) & (S < 150)).astype(np.uint8) * 255
    leaf = cv2.morphologyEx(leaf, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
    leaf = cv2.morphologyEx(leaf, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (75, 75)))
    n, lbl, st, _ = cv2.connectedComponentsWithStats(leaf, 8)
    if n <= 1:
        return np.full((H, W), 255, np.uint8)
    big = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    mask = (lbl == big).astype(np.uint8) * 255
    ff = mask.copy()
    cv2.floodFill(ff, np.zeros((H + 2, W + 2), np.uint8), (0, 0), 255)
    mask = mask | cv2.bitwise_not(ff)            # fill interior holes
    r = max(1, int(edge_trim * W))
    return cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2))


def _sauvola(g: np.ndarray, win: int, k: float) -> np.ndarray:
    """Sauvola local threshold: ink where g < mean*(1 + k*(std/128 - 1))."""
    win = _odd(win)
    m = cv2.boxFilter(g, -1, (win, win), normalize=True)
    s = cv2.sqrt(np.clip(cv2.boxFilter(g * g, -1, (win, win), normalize=True) - m * m, 0, None))
    return (g < m * (1 + k * (s / 128.0 - 1))).astype(np.uint8)


def _ink_alpha(bgr: np.ndarray, p: Params):
    """
    Shared core: isolate the ink and return (cropped_bgr, alpha8), alpha8 being
    ink coverage (0 = blank, 255 = full ink). All deterministic:
      1. Restrict to the leaf interior (drops mat, card and the frayed edge).
      2. Hysteresis Sauvola: a permissive pass captures complete faded strokes,
         a strict pass seeds them; keep only permissive regions with a seed.
         -> complete strokes, isolated noise dropped.
      3. Despeckle stray fibre flecks.
    This isolates the writing well; remaining fibre noise is meant to be wiped
    in the manual cleanup editor.
    """
    img = _apply_crop(bgr, p)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)

    interior = leaf_interior_mask(img, p.edge_trim)
    # Window auto-scales with leaf width (~61px at a 2150px leaf).
    win = _odd(max(15, round(img.shape[1] / 35.0)))
    scale = max(0.25, img.shape[1] / 2150.0)

    weak = _sauvola(g, win, p.k_weak)
    strong = _sauvola(g, win, p.k_strong)
    weak[interior == 0] = 0
    strong[interior == 0] = 0

    # Keep weak components that contain at least one strong (seed) pixel.
    n, lbl, st, _ = cv2.connectedComponentsWithStats(weak, 8)
    seeded = np.zeros(n, dtype=bool)
    seeded[np.unique(lbl[strong > 0])] = True
    seeded[0] = False
    mask = np.where(seeded[lbl], 255, 0).astype(np.uint8)

    # Despeckle fibre flecks (area scales with resolution).
    if p.min_blob and p.min_blob > 0:
        n, lbl, st, _ = cv2.connectedComponentsWithStats(mask, 8)
        keep = np.ones(n, dtype=bool)
        keep[0] = False
        min_area = int(p.min_blob * scale * scale)
        for i in range(1, n):
            if st[i, cv2.CC_STAT_AREA] < min_area:
                keep[i] = False
        mask = np.where(keep[lbl], 255, 0).astype(np.uint8)

    alpha8 = cv2.GaussianBlur(mask, (0, 0), max(0.4, 0.6 * scale))
    return img, alpha8


def _ink_rgb(img: np.ndarray, mode: str) -> np.ndarray:
    mode = (mode or "black").lower()
    if mode == "original":
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if mode == "sepia":
        rgb = np.zeros_like(img)
        rgb[:] = (60, 40, 25)            # R,G,B dark brown
        return rgb
    return np.zeros_like(img)            # black


def extract_ink_rgba(bgr: np.ndarray, p: Params) -> np.ndarray:
    """
    Isolate the ink and return an RGBA image (channels R,G,B,A) on a fully
    transparent background. Nothing is recognised, reconstructed or invented --
    only ink physically present is kept; fading is carried by the alpha channel.
    """
    img, alpha8 = _ink_alpha(bgr, p)
    return np.dstack([_ink_rgb(img, p.ink_color), alpha8]).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Vector (SVG) facsimile
# --------------------------------------------------------------------------- #

_INK_HEX = {"black": "#000000", "sepia": "#3c2819"}


def svg_available() -> bool:
    """True if the `potrace` tracer binary is installed."""
    return shutil.which("potrace") is not None


def extract_ink_svg(bgr: np.ndarray, p: Params) -> str:
    """
    Trace the isolated ink into a scalable SVG of vector paths on a transparent
    background (the same kind of output as a hand Illustrator image-trace). The
    ink mask comes from the exact same deterministic extraction as the PNG, so
    nothing is invented; potrace only smooths the existing shapes into curves.

    Requires the `potrace` binary. Raises RuntimeError if it is missing.
    """
    if not svg_available():
        raise RuntimeError("SVG tracer 'potrace' is not installed")

    img, alpha8 = _ink_alpha(bgr, p)
    h, w = alpha8.shape[:2]
    # Bitmap for potrace: ink = black (0), background = white (255).
    mask = np.where(alpha8 >= int(p.svg_threshold), 0, 255).astype(np.uint8)

    tmp = tempfile.mkdtemp(prefix="msvg_")
    try:
        pbm = os.path.join(tmp, "mask.pbm")
        out = os.path.join(tmp, "trace.svg")
        # PBM P4 (1-bit). 0 -> ink (black foreground for potrace).
        from PIL import Image
        Image.fromarray(mask, "L").convert("1").save(pbm)
        subprocess.run(
            ["potrace", pbm, "-s", "-t", "4", "-a", "1.2", "-O", "0.2", "-o", out],
            check=True, capture_output=True)
        svg = open(out, "r", encoding="utf-8").read()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    fill = _INK_HEX.get((p.ink_color or "black").lower(), "#000000")
    svg = svg.replace('fill="#000000"', f'fill="{fill}"')

    # Size for true 1:1 printing when a DPI is known; else leave potrace's units.
    if p.dpi and p.dpi > 0:
        win = w / float(p.dpi)
        hin = h / float(p.dpi)
        svg = re.sub(r'width="[^"]*"\s+height="[^"]*"',
                     f'width="{win:.4f}in" height="{hin:.4f}in"', svg, count=1)
    return svg


# --------------------------------------------------------------------------- #
# Secondary: flat black-on-white preview
# --------------------------------------------------------------------------- #

def process(bgr: np.ndarray, p: Params) -> np.ndarray:
    """Flat black-on-white binary preview (single channel) of the isolated ink."""
    _, alpha8 = _ink_alpha(bgr, p)
    return np.where(alpha8 >= 90, 0, 255).astype(np.uint8)


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

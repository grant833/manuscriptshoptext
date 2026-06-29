"""
Small U-Net ink/fibre denoiser (deep learning, but still NOT generative).

Like manuscript/denoiser.py it is a per-pixel ink/not-ink classifier -- it can
only keep or drop pixels that exist, never invent a letter. The difference is
that a CNN learns its own features, so it handles oriented papyrus fibres
(including the vertical fibres on versos) far better than the hand-crafted
RandomForest features.

Input is a lighting-invariant "ink darkness" map (see `preprocess`), so the same
representation is used for synthetic training patches and real photos.

Requires PyTorch (optional dependency).
"""

from __future__ import annotations

import os

import cv2
import numpy as np

try:
    import torch
    import torch.nn as nn
    _TORCH = True
except Exception:  # pragma: no cover
    _TORCH = False

from .pipeline import _odd, leaf_interior_mask

REF_W = 1400


def available() -> bool:
    return _TORCH


# --------------------------------------------------------------------------- #
# Preprocess: lighting-invariant ink-darkness map in [0,1]
# --------------------------------------------------------------------------- #

def preprocess(L: np.ndarray) -> np.ndarray:
    # FIXED kernel (independent of array size): training patches are crops of
    # REF_W-scale images, so the same absolute background scale must be used for
    # patches and full images, or the model sees a different representation.
    L = L.astype(np.float32)
    k = _odd(max(15, round(REF_W / 30)))     # ~47 px at REF_W
    bg = cv2.morphologyEx(L.astype(np.uint8), cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    bg = cv2.GaussianBlur(bg.astype(np.float32), (0, 0), k / 3.0) + 1e-3
    return np.clip((bg - L) / bg, 0.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #

if _TORCH:
    class _DoubleConv(nn.Module):
        def __init__(self, ci, co):
            super().__init__()
            # GroupNorm (per-instance, no running stats) so a model trained on
            # small synthetic patches behaves identically on a full real photo.
            g = min(8, co)
            self.net = nn.Sequential(
                nn.Conv2d(ci, co, 3, padding=1), nn.GroupNorm(g, co), nn.ReLU(True),
                nn.Conv2d(co, co, 3, padding=1), nn.GroupNorm(g, co), nn.ReLU(True))

        def forward(self, x):
            return self.net(x)

    class UNet(nn.Module):
        def __init__(self, ch=(16, 32, 64, 128)):
            super().__init__()
            self.d1 = _DoubleConv(1, ch[0])
            self.d2 = _DoubleConv(ch[0], ch[1])
            self.d3 = _DoubleConv(ch[1], ch[2])
            self.bott = _DoubleConv(ch[2], ch[3])
            self.pool = nn.MaxPool2d(2)
            self.up3 = nn.ConvTranspose2d(ch[3], ch[2], 2, stride=2)
            self.u3 = _DoubleConv(ch[3], ch[2])
            self.up2 = nn.ConvTranspose2d(ch[2], ch[1], 2, stride=2)
            self.u2 = _DoubleConv(ch[2], ch[1])
            self.up1 = nn.ConvTranspose2d(ch[1], ch[0], 2, stride=2)
            self.u1 = _DoubleConv(ch[1], ch[0])
            self.out = nn.Conv2d(ch[0], 1, 1)

        def forward(self, x):
            c1 = self.d1(x)
            c2 = self.d2(self.pool(c1))
            c3 = self.d3(self.pool(c2))
            b = self.bott(self.pool(c3))
            x = self.u3(torch.cat([self.up3(b), c3], 1))
            x = self.u2(torch.cat([self.up2(x), c2], 1))
            x = self.u1(torch.cat([self.up1(x), c1], 1))
            return self.out(x)            # logits


def load(path: str):
    if not _TORCH:
        raise RuntimeError("PyTorch not installed")
    model = UNet()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "models")
DEFAULT_MODEL = os.path.join(_MODELS_DIR, "unet.pt")


def load_default():
    if _TORCH and os.path.exists(DEFAULT_MODEL):
        try:
            return load(DEFAULT_MODEL)
        except Exception:
            return None
    return None


# --------------------------------------------------------------------------- #
# Apply
# --------------------------------------------------------------------------- #

def apply(model, bgr_leaf: np.ndarray, thresh: float = 0.5,
          min_blob: int = 60, min_thick: float = 5.0) -> np.ndarray:
    """Cleaned ink mask (uint8 0/255) at the leaf's original size."""
    H0, W0 = bgr_leaf.shape[:2]
    s = REF_W / W0
    img = cv2.resize(bgr_leaf, (REF_W, int(H0 * s)))
    L = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:, :, 0]
    inp = preprocess(L)
    # pad to multiple of 8 for the U-Net
    h, w = inp.shape
    ph, pw = (-h) % 8, (-w) % 8
    padded = np.pad(inp, ((0, ph), (0, pw)))
    with torch.no_grad():
        t = torch.from_numpy(padded)[None, None]
        prob = torch.sigmoid(model(t))[0, 0].numpy()[:h, :w]
    interior = leaf_interior_mask(img, 0.012)
    mask = ((prob >= thresh) & (interior > 0)).astype(np.uint8) * 255
    if min_blob or min_thick:
        n, lbl, st, _ = cv2.connectedComponentsWithStats(mask, 8)
        keep = np.ones(n, bool); keep[0] = False
        if min_thick:
            dt = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
            md = np.zeros(n, np.float32); np.maximum.at(md, lbl.ravel(), dt.ravel())
        for i in range(1, n):
            if (min_blob and st[i, cv2.CC_STAT_AREA] < min_blob) or \
               (min_thick and 2 * md[i] < min_thick):
                keep[i] = False
        mask = np.where(keep[lbl], 255, 0).astype(np.uint8)
    return cv2.resize(mask, (W0, H0), interpolation=cv2.INTER_NEAREST)

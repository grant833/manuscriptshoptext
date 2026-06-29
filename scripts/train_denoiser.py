#!/usr/bin/env python3
"""
Train the ink/fibre denoiser and save it to models/denoiser.joblib.

The model is a per-pixel ink/not-ink classifier (NOT generative — it can never
invent a letter). Training data is built synthetically: clean ink shapes are
composited onto real papyrus fibre texture so labels are exact and there is no
need to pixel-align hand-traced facsimiles to photos.

Edit PAIRS below to point at your data:
  - a RAW leaf photo (+ a crop box to the leaf) supplies the papyrus TEXTURE
  - a clean hand-traced facsimile PNG (transparent, ink in the alpha) supplies
    INK SHAPES. Use shapes from a DIFFERENT leaf than the texture.

More/varied pairs -> a model that generalises across leaves and lighting.

Usage:  python scripts/train_denoiser.py
"""

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from manuscript import denoiser as D

# (raw_photo_path, (crop_x, crop_y, crop_w, crop_h), clean_facsimile_png_path)
PAIRS = [
    # ("data/raw/leafA.webp", (437, 374, 2163, 3785), "data/clean/leafB_clean.png"),
    # ("data/raw/leafB.webp", (650, 350, 2450, 3950), "data/clean/leafA_clean.png"),
]


def _leaf(path, crop):
    img = cv2.imread(path)
    if img is None:
        raise SystemExit(f"cannot read {path}")
    x, y, w, h = crop
    return img[y:y + h, x:x + w]


def _shape(path):
    m = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if m is None or m.ndim != 3 or m.shape[2] != 4:
        raise SystemExit(f"{path} must be an RGBA PNG (ink in alpha)")
    return (m[:, :, 3] >= 128).astype(np.uint8) * 255


def main():
    if not D.available():
        raise SystemExit("scikit-learn not installed (pip install scikit-learn)")
    if not PAIRS:
        raise SystemExit("Edit PAIRS in this script to point at your data first.")
    pairs = [(_leaf(r, c), _shape(s)) for r, c, s in PAIRS]
    print(f"building training set from {len(pairs)} pairs…")
    X, Y = D.build_training_set(pairs, per_leaf=35000)
    print(f"training on {X.shape[0]} samples…")
    clf = D.train(X, Y, n_estimators=140)
    os.makedirs(os.path.dirname(D.DEFAULT_MODEL), exist_ok=True)
    D.save(clf, D.DEFAULT_MODEL)
    print(f"saved {D.DEFAULT_MODEL} "
          f"({os.path.getsize(D.DEFAULT_MODEL) / 1048576:.1f} MB)")


if __name__ == "__main__":
    main()

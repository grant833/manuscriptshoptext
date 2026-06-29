"""
Discriminative ink/fibre denoiser (no text generation).

The deterministic pipeline gets the LETTERS right but leaves papyrus fibre-noise
that can't be removed by thresholds without harming the writing (fibres and the
thin parts of letters look identical to classical filters). This module learns
to tell them apart with a per-pixel classifier.

It is NOT generative: it only ever classifies existing pixels as ink or
not-ink, so it can never invent or reconstruct a letter -- it just cleans.

Training data is built SYNTHETICALLY, which sidesteps the fact that hand-traced
facsimiles don't pixel-align to the photos:
  * background = real papyrus fibre texture (a photo with its ink inpainted out)
  * foreground = clean ink shapes (from hand-traced facsimiles of OTHER leaves),
    darkened to realistic ink values sampled from the real photo
  * label = exactly where the clean ink was placed (perfect by construction)

A classifier trained on this learns to reject fibre texture and keep ink, then
runs on the real photo. Features capture what separates them: Gabor responses
(fibres are thin and oriented; ink is not), local statistics and darkness.

Requires scikit-learn (optional dependency). `available()` reports if usable.
"""

from __future__ import annotations

import os

import numpy as np
import cv2

from .pipeline import _odd, _sauvola, leaf_interior_mask

try:
    from sklearn.ensemble import RandomForestClassifier
    import joblib
    _SK = True
except Exception:  # pragma: no cover
    _SK = False

REF_W = 1400               # work width the features are tuned at
GABOR_ANGLES = (0, 45, 90, 135)
GABOR_LAMBDAS = (5, 10)


def available() -> bool:
    return _SK


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #

def features(L: np.ndarray) -> np.ndarray:
    """Per-pixel feature stack (H, W, C) from a lightness channel L."""
    L = L.astype(np.float32)
    win = _odd(max(11, round(L.shape[1] / 35)))
    dark = 255.0 - L
    out = [dark]
    for th in GABOR_ANGLES:
        for lam in GABOR_LAMBDAS:
            k = cv2.getGaborKernel((15, 15), 2.5, np.deg2rad(th), lam, 0.5, 0,
                                   ktype=cv2.CV_32F)
            out.append(np.abs(cv2.filter2D(dark, cv2.CV_32F, k)))
    m = cv2.boxFilter(L, -1, (win, win))
    out.append(L - m)
    out.append(cv2.boxFilter((L - m) ** 2, -1, (win, win)))
    return np.stack(out, -1).astype(np.float32)


def _scaled_L(bgr_leaf: np.ndarray):
    s = REF_W / bgr_leaf.shape[1]
    img = cv2.resize(bgr_leaf, (REF_W, int(bgr_leaf.shape[0] * s)))
    return img, cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:, :, 0]


# --------------------------------------------------------------------------- #
# Synthetic training data
# --------------------------------------------------------------------------- #

def _blank_background(L: np.ndarray, interior: np.ndarray):
    """Inpaint the ink out of a real leaf -> pure papyrus fibre texture, plus
    the distribution of real ink darkness (local background minus L)."""
    weak = _sauvola(L.astype(np.float32), _odd(round(L.shape[1] / 35)), 0.06)
    weak[interior == 0] = 0
    blank = cv2.inpaint(L, cv2.dilate(weak, np.ones((5, 5), np.uint8)), 5,
                        cv2.INPAINT_TELEA).astype(np.float32)
    dt = cv2.distanceTransform(weak, cv2.DIST_L2, 5)
    dark = (blank - L)[(dt >= 4) & (weak > 0)]
    return blank, dark[dark > 0]


def _make_synthetic(blank, dark_samples, ink_shape, rng):
    """Darken the blank papyrus where ink_shape is set, using realistic ink
    darkness. Returns (synthetic L, label mask)."""
    shp = cv2.resize(ink_shape, (blank.shape[1], blank.shape[0])) > 128
    field = np.zeros_like(blank)
    field[shp] = rng.choice(dark_samples, size=int(shp.sum()))
    field = cv2.GaussianBlur(field, (0, 0), 0.8)
    return np.clip(blank - field, 0, 255), shp.astype(np.uint8)


def build_training_set(pairs, per_leaf=50000, seed=0):
    """
    pairs: list of (bgr_leaf_for_texture, ink_shape_mask_uint8). The ink shapes
    should come from a DIFFERENT leaf than the texture so labels are honest.
    Returns (X, y) ready for a classifier.
    """
    rng = np.random.RandomState(seed)
    Xs, ys = [], []
    for bgr_leaf, ink_shape in pairs:
        img, L = _scaled_L(bgr_leaf)
        interior = leaf_interior_mask(img, 0.012)
        blank, dark = _blank_background(L, interior)
        if len(dark) == 0:
            continue
        synthL, label = _make_synthetic(blank, dark, ink_shape, rng)
        F = features(synthL)
        pos = np.argwhere(label > 0)
        neg = np.argwhere((label == 0) & (interior > 0))
        for arr, lab in ((pos, 1), (neg, 0)):
            if len(arr) == 0:
                continue
            idx = rng.choice(len(arr), min(per_leaf, len(arr)), replace=False)
            Xs.append(F[arr[idx, 0], arr[idx, 1]])
            ys.append(np.full(len(idx), lab))
    return np.vstack(Xs), np.concatenate(ys)


# --------------------------------------------------------------------------- #
# Train / apply / persist
# --------------------------------------------------------------------------- #

def train(X, y, n_estimators=120, max_depth=18, jobs=4):
    if not _SK:
        raise RuntimeError("scikit-learn not installed")
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                 n_jobs=jobs, random_state=0)
    clf.fit(X, y)
    return clf


def apply(clf, bgr_leaf: np.ndarray, min_blob: int = 90,
          min_thick: float = 7.0) -> np.ndarray:
    """Return a cleaned ink mask (uint8 0/255) at the leaf's original size."""
    H0, W0 = bgr_leaf.shape[:2]
    img, L = _scaled_L(bgr_leaf)
    interior = leaf_interior_mask(img, 0.012)
    F = features(L)
    prob = clf.predict_proba(F.reshape(-1, F.shape[-1]))[:, 1].reshape(L.shape)
    mask = ((prob >= 0.5) & (interior > 0)).astype(np.uint8) * 255

    # Post-filter: drop specks (area) and thin fibres (stroke thickness).
    if min_blob or min_thick:
        n, lbl, st, _ = cv2.connectedComponentsWithStats(mask, 8)
        keep = np.ones(n, bool); keep[0] = False
        if min_thick:
            dt = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
            maxd = np.zeros(n, np.float32)
            np.maximum.at(maxd, lbl.ravel(), dt.ravel())
        for i in range(1, n):
            if (min_blob and st[i, cv2.CC_STAT_AREA] < min_blob) or \
               (min_thick and 2 * maxd[i] < min_thick):
                keep[i] = False
        mask = np.where(keep[lbl], 255, 0).astype(np.uint8)
    return cv2.resize(mask, (W0, H0), interpolation=cv2.INTER_NEAREST)


def save(clf, path: str):
    joblib.dump(clf, path, compress=3)


def load(path: str):
    return joblib.load(path)


_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "models")
DEFAULT_MODEL = os.path.join(_MODELS_DIR, "denoiser.joblib")


def load_default():
    """Load the bundled model if scikit-learn is available and the file exists."""
    if _SK and os.path.exists(DEFAULT_MODEL):
        try:
            return joblib.load(DEFAULT_MODEL)
        except Exception:
            return None
    return None

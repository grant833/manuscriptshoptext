#!/usr/bin/env python3
"""
Train the U-Net ink/fibre denoiser on REAL aligned pairs and save models/unet.pt.

This is the path that can finally clean versos. It uses the (photo, hand-cleaned
mask) pairs collected by the editor's "Save to training set" button -- they
pixel-align (the mask was drawn over the photo), unlike re-traced facsimiles, so
the model learns from real faint-verso ink instead of synthetic guesses.

It is still NOT generative -- a per-pixel ink/not-ink classifier that can only
keep or drop existing pixels, never invent a letter.

Use a GPU. Renting a cloud GPU box for an hour is enough:
    pip install torch opencv-python-headless numpy
    python scripts/train_unet.py            # auto-uses CUDA if present

Aim for ~15-30 cleaned pairs in training/ before training. Rotation augmentation
makes it orientation-invariant, so it handles both recto and verso fibres.
"""

import os
import sys
import glob
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
from manuscript.unet import UNet, preprocess, REF_W

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(BASE, "training")
OUT = os.path.join(BASE, "models", "unet.pt")
PS, STEPS, BATCH = 192, 4000, 16
DEV = "cuda" if torch.cuda.is_available() else "cpu"
rng = np.random.RandomState(0)
torch.manual_seed(0)


def load_pairs():
    pairs = []
    for mp in sorted(glob.glob(os.path.join(TRAIN, "*_mask.png"))):
        pp = mp.replace("_mask.png", "_photo.png")
        if not os.path.exists(pp):
            continue
        photo = cv2.imread(pp)
        mask = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        # normalise to REF_W scale so patch sizes are consistent
        s = REF_W / photo.shape[1]
        photo = cv2.resize(photo, (REF_W, int(photo.shape[0] * s)))
        mask = cv2.resize(mask, (photo.shape[1], photo.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
        L = cv2.cvtColor(photo, cv2.COLOR_BGR2LAB)[:, :, 0]
        pairs.append((L, (mask >= 128).astype(np.float32)))
    return pairs


def patch(pairs):
    L, m = pairs[rng.randint(len(pairs))]
    H, W = L.shape
    big = int(PS * 1.5)
    if H <= big or W <= big:
        return None
    y, x = rng.randint(0, H - big), rng.randint(0, W - big)
    Lp, mp = L[y:y+big, x:x+big].astype(np.float32), m[y:y+big, x:x+big]
    ang = rng.uniform(0, 360)
    M = cv2.getRotationMatrix2D((big/2, big/2), ang, 1.0)
    Lp = cv2.warpAffine(Lp, M, (big, big), borderMode=cv2.BORDER_REFLECT)
    mp = cv2.warpAffine(mp, M, (big, big), flags=cv2.INTER_NEAREST,
                        borderMode=cv2.BORDER_REFLECT)
    o = (big - PS) // 2
    Lp, mp = Lp[o:o+PS, o:o+PS], mp[o:o+PS, o:o+PS]
    if rng.rand() < 0.5: Lp, mp = Lp[:, ::-1], mp[:, ::-1]
    Lp = np.clip(Lp * rng.uniform(0.9, 1.1) + rng.normal(0, rng.uniform(0, 3), Lp.shape), 0, 255)
    return preprocess(Lp).astype(np.float32), mp.astype(np.float32)


def main():
    pairs = load_pairs()
    if len(pairs) < 4:
        raise SystemExit(f"Need >=4 aligned pairs in {TRAIN} (have {len(pairs)}). "
                         "Clean leaves in the editor and click 'Save to training set'.")
    print(f"{len(pairs)} pairs | device {DEV}", flush=True)
    model = UNet().to(DEV)
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    lossf = nn.BCEWithLogitsLoss()
    t0 = time.time()
    for step in range(1, STEPS + 1):
        xs, ys = [], []
        while len(xs) < BATCH:
            p = patch(pairs)
            if p: xs.append(p[0]); ys.append(p[1])
        x = torch.from_numpy(np.stack(xs))[:, None].to(DEV)
        y = torch.from_numpy(np.stack(ys))[:, None].to(DEV)
        opt.zero_grad(); loss = lossf(model(x), y); loss.backward(); opt.step()
        if step % 100 == 0:
            print(f"step {step}/{STEPS} loss {loss.item():.4f} {int(time.time()-t0)}s", flush=True)
        if step % 500 == 0 or step == STEPS:
            os.makedirs(os.path.dirname(OUT), exist_ok=True)
            torch.save(model.cpu().state_dict(), OUT); model.to(DEV)
    print(f"saved {OUT}", flush=True)


if __name__ == "__main__":
    main()

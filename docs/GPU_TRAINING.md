# Training the verso model on a GPU

This is the step that makes **versos** (and other hard leaves) clean up
automatically. It trains the U-Net (`manuscript/unet.py`) on the **aligned
(photo, mask) pairs** the editor collects. It is still not generative — a
per-pixel ink/not-ink classifier that can only keep or drop real pixels.

You only need this occasionally, on a rented GPU. ~1 hour of a cheap cloud GPU
is plenty.

## Before you start
- Collect **~15–30 cleaned leaves** in `training/` (editor → **Save to training
  set**). More and more varied (especially versos, faded, heavy-fibre) is
  better. Each pair is `<id>_photo.png` + `<id>_mask.png`.

## 1. Rent a cloud GPU
Any provider with a basic NVIDIA GPU works (e.g., Lambda, RunPod, Vast.ai, a
GPU VM on a cloud you already use). A single mid-range GPU (e.g., T4/A10/3090)
is more than enough. Pick an image that already has Python + CUDA.

## 2. Get the code and your data onto it
```bash
git clone https://github.com/grant833/manuscriptshoptext.git
cd manuscriptshoptext
pip install torch opencv-python-headless numpy
```
Copy your `training/` folder (the `*_photo.png` / `*_mask.png` pairs) into the
repo's `training/` directory — e.g. with `scp`, or zip it and upload.

## 3. Train
```bash
python scripts/train_unet.py
```
It auto-detects the GPU (`device cuda`), trains with rotation augmentation (so it
learns fibres at every orientation — the key to versos), and writes
`models/unet.pt`. On a GPU this is minutes, not the ~25 min it takes on CPU.

Tune inside `scripts/train_unet.py` if needed: `STEPS` (more data → more steps),
`PS` (patch size).

## 4. Use the trained model
Copy `models/unet.pt` back into your local `models/` folder. The app
**automatically prefers `unet.pt`** over the bundled RandomForest when it's
present (`/api/health` shows `"ml_backend": "unet"`). Nothing else to change —
"ML clean" now uses it.

## 5. Keep improving
As you clean more leaves, re-run training periodically with the larger dataset.
Quality climbs as the aligned set grows — that's the whole point of the
**Save to training set** button.

## Why this works when the bundled model doesn't on versos
The bundled RandomForest (and the CPU-trained U-Net) learned from **synthetic**
data, which can't reproduce real faint-verso ink. Your editor pairs are **real
and pixel-aligned**, so the model finally sees what verso ink actually looks
like — the missing ingredient.

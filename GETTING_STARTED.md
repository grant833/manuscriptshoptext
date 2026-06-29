# Getting started (on your new computer)

A plain-language setup. One-time install, then it's one command each time.

## 1. Install the two free programs

- **Python** — download from [python.org/downloads](https://www.python.org/downloads/).
  During install, **check the box "Add Python to PATH."**
- **Git** (only needed to download the code; or use the ZIP option below) —
  [git-scm.com/downloads](https://git-scm.com/downloads).

## 2. Get the code

Easiest, no Git: open this link, which downloads a ZIP, then right-click →
**Extract All**:

`https://github.com/grant833/manuscriptshoptext/archive/refs/heads/claude/manuscript-text-extraction-lzpa97.zip`

## 3. Start it

Open the extracted folder, click the address bar, type `cmd` (Windows) and press
Enter, then:

```
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe app.py
```

(Mac/Linux: `./run.sh`.)

When you see `Running on http://127.0.0.1:5000`, open that address in your
browser.

### Optional extras (better, not required)
- **Clean SVG export + OCR:** install `potrace` and `tesseract`.
- **ML cleanup ("ML clean" toggle):** `pip install scikit-learn` (the model is
  bundled). For the deep-learning model later, see `docs/GPU_TRAINING.md`.

## 4. How to use it

1. **Upload** the front and back photos.
2. **Crop** to the leaf (drag a box on *Original*, or *Auto-detect*).
3. **ML clean** is on by default — it removes most papyrus fibre-noise.
4. For an easy/clean leaf, the **Facsimile / Vector (SVG)** tabs are your result —
   **Download PNG / SVG.**
5. For a hard leaf (faint versos, heavy fibres), click **Edit & clean** to open
   the editor: zoom in, **brush** to complete letters, **eraser** to wipe noise,
   then export **Text PNG** + **Outline SVG.**

## 5. Build the dataset that automates versos (important, zero extra effort)

Every time you clean a leaf in the editor, click **★ Save to training set.**
That stores the photo + your cleaned mask as an *aligned pair* in the
`training/` folder. Once you've collected ~15–30 of these, a GPU can train a
model that cleans versos automatically — see `docs/GPU_TRAINING.md`. The manual
cleanup you do today is what teaches the automatic tool tomorrow.

## Two-minute recap
- Easy leaves → automatic.
- Hard leaves → editor (fast), and click **Save to training set** each time.
- Collect pairs → train on a GPU later → hard leaves become automatic too.

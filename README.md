# Manuscript Facsimile Workspace

A local, browser-based tool that extracts the surviving ink from photographs of
manuscripts (e.g. Greek majuscule papyri) and outputs it as a **faithful
facsimile on a transparent background** — print-ready, 1:1, suitable for
printing onto blank papyrus.

## Design principle: nothing is invented

The output is a **historical facsimile, not a restoration**. We never recognise,
reconstruct, "improve", or generate any character — we only isolate the ink that
is physically present:

- **100% deterministic image processing** (classic OpenCV). No machine learning,
  no text recognition, no generation, so hallucination is not even possible.
- **Ink is carried as a soft alpha channel**, so fading, broken strokes, smudges
  and gaps are preserved as partial transparency rather than forced to solid
  black. Faint ink stays faint; ambiguous marks stay ambiguous.
- **Where there is no ink, the output is empty** (transparent). Damage shows
  exactly as it is.
- Despeckle is **off by default** so faint / broken strokes are never erased.

An optional Tesseract OCR draft exists purely as a typing aid; it is clearly
labelled **UNVERIFIED**, never trusted, and disabled if the engine is absent.

## Quick start

```bash
./run.sh
# then open http://127.0.0.1:5000
```

Or manually:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# optional OCR engine (Debian/Ubuntu):
apt-get install -y tesseract-ocr tesseract-ocr-grc
python app.py
```

## Workflow

1. **Upload** the **Front** and **Back** of the leaf into their slots.
2. **Crop** to the leaf — drag a box on the *Original* view (or *Auto-detect*)
   to drop the mounting mat and colour-reference card.
3. **Tune** the extraction (all deterministic):
   - *Background scale* — size of the lighting/papyrus-tone estimate.
   - *Floor* — drops faint papyrus texture; keep low to preserve faint ink.
   - *Gain* — how gently fading ramps into opacity.
   - *Ink colour* — Original (most faithful), Black, or Sepia.
   - *Despeckle* — off by default.
4. **Print scale** — enter the leaf's real width (read it off the photo's ruler)
   so the PNG embeds the correct DPI and prints at true 1:1 size.
5. **Run both (automatic)** processes front + back, or *Process side* for one.
6. **Download PNG** — a transparent-background facsimile per side.

## Output

- RGBA PNG, fully transparent background, ink as soft alpha.
- Original orientation, size, proportions and layout preserved (no deskew /
  dewarp — "exactly as seen").
- Embedded DPI for 1:1 printing.

## Layout

```
app.py                 Flask server + API
manuscript/
  pipeline.py          deterministic ink extraction + background removal (OpenCV)
  ocr.py               optional, clearly-flagged OCR draft
static/                single-page workspace (HTML/CSS/JS)
uploads/ outputs/      runtime data (git-ignored)
```

## Roadmap toward full automation

The goal is hands-off: upload front + back, get two facsimiles automatically.
Already automatic: background removal, ink isolation, transparency, soft fading,
best-effort auto-crop. Still to come:

- Robust auto-crop / leaf detection tuned on real manuscript photos.
- Automatic ruler detection for true 1:1 scale without manual entry.
- Optional SVG (vector) export.
- Batch / folder processing and an API for unattended runs.

This is an iterative, trial-and-error project; defaults will be tuned per real
manuscript image as samples are provided.

# Manuscript Text Workspace

A local, browser-based tool for faithfully extracting the text from photographs
of manuscripts (e.g. Greek majuscule papyri).

## Design principle: no invented text

The hard truth about ancient manuscripts: no off-the-shelf OCR can read papyrus
majuscule script accurately, and AI models that *can* sometimes **hallucinate**
plausible-but-wrong text. This tool is built so that **nothing is ever
fabricated**:

1. **Image cleaning is 100% deterministic** — classic OpenCV operations only
   (crop, lighting flatten, contrast, threshold). Where ink is lost or broken,
   the output stays blank, so **damage is shown exactly as it is** and never
   "filled in".
2. **The text is transcribed by you**, the human, in a side-by-side workspace,
   using the standard papyrological (Leiden) markers for damaged / uncertain
   readings.
3. **OCR is optional and clearly labelled "UNVERIFIED"** — a rough draft to
   correct against the image, never trusted output. It is fully disabled if the
   `tesseract` engine isn't installed.

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

1. **Upload** a manuscript photo.
2. **Crop** to the leaf — drag a box on the *Original* image (or *Auto-detect*)
   to drop the mounting mat and colour-reference card.
3. **Tune** the cleaning sliders until the *Cleaned* image shows the surviving
   ink clearly. Every control is a deterministic image operation:
   - *Flatten lighting* — evens out the papyrus tone.
   - *CLAHE* — local contrast.
   - *Threshold* — adaptive (default), Otsu, or a manual cutoff to get
     black-on-white.
   - *Despeckle* — off by default so faint strokes are never erased.
4. **Transcribe** in the right-hand editor. Marker buttons insert Leiden
   conventions: `[ ]` restored, ◌̣ uncertain, `( )` expansion, ⟦ ⟧ deletion,
   lacuna, ◌̅ overline (*nomen sacrum*).
5. **Save / Download** the transcription as a `.txt` file.

## Layout

```
app.py                 Flask server + API
manuscript/
  pipeline.py          deterministic image cleaning (OpenCV)
  ocr.py               optional, clearly-flagged OCR draft
static/                single-page workspace (HTML/CSS/JS)
uploads/ outputs/      runtime data (git-ignored)
```

## Status

Early scaffold — this is an iterative, trial-and-error project. The cleaning
pipeline and workspace are in place; expect to tune defaults per manuscript and
extend the pipeline (deskew, dewarp, per-line cropping) over time.

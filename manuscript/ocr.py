"""
Optional OCR draft.

IMPORTANT: this is NOT trusted output. Off-the-shelf OCR cannot read papyrus
majuscule script reliably. Whatever this returns is a rough, error-filled draft
meant only to give the human transcriber a starting point. The UI labels it as
an unverified draft and never treats it as the final text.

If the `tesseract` binary (with the `grc` ancient-Greek model) is not installed,
OCR is simply unavailable and the rest of the app works fine without it.
"""

from __future__ import annotations

import numpy as np

try:
    import pytesseract
    from PIL import Image
    _IMPORT_OK = True
except Exception:  # pragma: no cover
    _IMPORT_OK = False


def available() -> bool:
    """True only if pytesseract is importable AND the grc model is present."""
    if not _IMPORT_OK:
        return False
    try:
        langs = pytesseract.get_languages(config="")
        return "grc" in langs
    except Exception:
        return False


def draft(binimg: np.ndarray, lang: str = "grc") -> str:
    """Return a rough OCR draft for the given black-on-white image."""
    if not available():
        raise RuntimeError("OCR engine (tesseract + grc) is not installed")
    pil = Image.fromarray(binimg)
    # PSM 6 = assume a single uniform block of text.
    return pytesseract.image_to_string(pil, lang=lang, config="--psm 6").strip()

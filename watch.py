"""
Drop-folder watcher: fully automatic facsimile extraction.

Drop manuscript photos (front and/or back) into ./inbox and this process picks
them up and writes a transparent-background ink facsimile PNG to ./outbox --
no clicking, no interaction. Pure deterministic image processing; nothing is
recognised, reconstructed or invented.

Run:
    python watch.py                # watches ./inbox -> ./outbox
    python watch.py IN OUT         # custom folders
    python watch.py --once         # process whatever is there, then exit

Naming: drop e.g. "BP-II-86_front.jpg" and "BP-II-86_back.jpg"; you get
"BP-II-86_front_facsimile.png" and "BP-II-86_back_facsimile.png".
"""

from __future__ import annotations

import os
import sys
import time

import cv2
import numpy as np

from manuscript.pipeline import (auto_params, extract_ink_rgba,
                                 extract_ink_svg, svg_available)

BASE = os.path.dirname(os.path.abspath(__file__))
ALLOWED = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
POLL_SECONDS = 2.0


def _log(msg: str) -> None:
    print(time.strftime("[%H:%M:%S] ") + msg, flush=True)


def _read_bgr(path: str) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("could not decode image")
    return img


def process_file(path: str, outbox: str) -> str:
    """Extract the facsimile for one image -> transparent PNG (+ SVG if able)."""
    bgr = _read_bgr(path)
    params = auto_params(bgr)
    rgba = extract_ink_rgba(bgr, params)                     # R,G,B,A
    bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)           # cv2 wants BGRA
    stem = os.path.splitext(os.path.basename(path))[0]
    out = os.path.join(outbox, stem + "_facsimile.png")
    cv2.imwrite(out, bgra)
    if svg_available():
        with open(os.path.join(outbox, stem + "_facsimile.svg"), "w",
                  encoding="utf-8") as fh:
            fh.write(extract_ink_svg(bgr, params))
    return out


def _candidates(inbox: str):
    for name in sorted(os.listdir(inbox)):
        if os.path.splitext(name)[1].lower() in ALLOWED:
            yield os.path.join(inbox, name)


def run(inbox: str, outbox: str, once: bool = False) -> None:
    os.makedirs(inbox, exist_ok=True)
    os.makedirs(outbox, exist_ok=True)
    _log(f"watching {inbox}  ->  {outbox}")

    # Remember files already handled (by path + mtime + size) and pending sizes
    # so we never process a file that is still being copied in.
    done: dict[str, tuple] = {}
    sizes: dict[str, int] = {}

    # In one-shot mode, files already sitting in the inbox are finished copies,
    # so seed their sizes as "stable" and process them on the first pass.
    if once:
        for path in _candidates(inbox):
            try:
                sizes[path] = os.stat(path).st_size
            except FileNotFoundError:
                pass

    while True:
        for path in _candidates(inbox):
            try:
                st = os.stat(path)
            except FileNotFoundError:
                continue
            key = (path, st.st_mtime, st.st_size)
            if done.get(path) == key:
                continue
            # Wait until the file size is stable across two polls (copy finished).
            if sizes.get(path) != st.st_size:
                sizes[path] = st.st_size
                continue
            try:
                out = process_file(path, outbox)
                done[path] = key
                _log(f"✓ {os.path.basename(path)} -> {os.path.basename(out)}")
            except Exception as e:
                _log(f"✗ {os.path.basename(path)}: {e}")
                done[path] = key  # don't loop forever on a bad file
        if once:
            return
        time.sleep(POLL_SECONDS)


def main(argv: list[str]) -> None:
    once = "--once" in argv
    args = [a for a in argv if not a.startswith("--")]
    inbox = args[0] if len(args) > 0 else os.path.join(BASE, "inbox")
    outbox = args[1] if len(args) > 1 else os.path.join(BASE, "outbox")
    try:
        run(inbox, outbox, once=once)
    except KeyboardInterrupt:
        _log("stopped")


if __name__ == "__main__":
    main(sys.argv[1:])

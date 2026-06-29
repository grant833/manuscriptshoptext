#!/usr/bin/env bash
# Launch the manuscript workspace on http://127.0.0.1:5000
set -e
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  echo "Creating virtual environment…"
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --quiet -r requirements.txt

if ! command -v tesseract >/dev/null 2>&1; then
  echo "Note: 'tesseract' not found — optional OCR draft will be disabled."
  echo "      To enable: apt-get install -y tesseract-ocr tesseract-ocr-grc"
fi

python app.py

"""Shared data-directory paths."""

from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
RAW = DATA_DIR / "raw"
RAW_LARGE = DATA_DIR / "raw_large"
PROCESSED = DATA_DIR / "processed"

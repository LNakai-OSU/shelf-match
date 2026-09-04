"""
Downloads the raw UCSD Goodreads Book Graph files (Wan & McAuley,
RecSys'18: https://cseweb.ucsd.edu/~jmcauley/datasets/goodreads.html) into
data/raw_large/ - 2.36M books (with real description text, unlike
goodbooks-10k), their authors, and genre labels, plus a partial slice of
the interaction/ratings data.

goodreads_books.json.gz is ~2GB and goodreads_interactions.csv is ~4.3GB
uncompressed - not something to pull in full here. This downloads the book
metadata in full (needed for every book, can't subsample and still build
a real catalog) but only the first INTERACTIONS_BYTES of the interactions
file via an HTTP Range request - the file is sorted by user_id, so this is
every interaction from the first N real users rather than a uniform random
sample of all ~876K users in the full dataset. That trade-off (a real
partial slice, not a claimed-random sample) is documented in the README,
not hidden.
"""

import pathlib
import sys
import urllib.request

RAW_LARGE = pathlib.Path(__file__).parent.parent / "data" / "raw_large"
BASE_URL = "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/"

FULL_FILES = [
    "goodreads_books.json.gz",  # ~2.0 GB - every book's metadata + description
    "goodreads_book_authors.json.gz",  # ~18 MB
    "goodreads_book_genres_initial.json.gz",  # ~24 MB
    "book_id_map.csv",  # ~38 MB - anonymized interaction book_id -> real goodreads book_id
]

INTERACTIONS_BYTES = 600 * 1024 * 1024  # ~600MB prefix of the 4.3GB interactions.csv


def _download(name, dest, range_bytes=None):
    if dest.exists():
        print(f"skip (already have) {name}")
        return
    url = BASE_URL + name
    req = urllib.request.Request(url)
    if range_bytes:
        req.add_header("Range", f"bytes=0-{range_bytes}")
    print(f"downloading {url}{' (partial)' if range_bytes else ''} -> {dest}")
    with urllib.request.urlopen(req) as resp, open(dest, "wb") as out:
        while chunk := resp.read(1024 * 1024):
            out.write(chunk)
    print(f"  {dest.stat().st_size / 1e6:.1f} MB")


def main():
    RAW_LARGE.mkdir(parents=True, exist_ok=True)
    for name in FULL_FILES:
        _download(name, RAW_LARGE / name)
    _download("goodreads_interactions.csv", RAW_LARGE / "interactions_prefix.csv", range_bytes=INTERACTIONS_BYTES)


if __name__ == "__main__":
    sys.exit(main())

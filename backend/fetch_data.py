"""
Downloads the goodbooks-10k dataset (Zajac, https://github.com/zygmuntz/goodbooks-10k):
10,000 books, ~6 million user ratings, and Goodreads folksonomy tags.

Goodreads shut down its public API in 2020, so this project uses this
pre-collected dataset rather than scraping goodreads.com directly (which
would also violate its Terms of Service). goodbooks-10k is hosted as plain
CSVs on GitHub, so no Kaggle account/API key is needed.
"""

import pathlib
import sys
import urllib.request

RAW_DIR = pathlib.Path(__file__).parent.parent / "data" / "raw"
BASE_URL = "https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/"

FILES = ["books.csv", "ratings.csv", "book_tags.csv", "tags.csv", "to_read.csv"]


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        dest = RAW_DIR / name
        if dest.exists():
            print(f"skip (already have) {name}")
            continue
        url = BASE_URL + name
        print(f"downloading {url} -> {dest}")
        urllib.request.urlretrieve(url, dest)
        print(f"  {dest.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    sys.exit(main())

"""Loads the raw goodbooks-10k CSVs and derives a per-book genre matrix
from the (very noisy) Goodreads folksonomy tags."""

from pathlib import Path

import pandas as pd

from .genre_tags import GENRE_TAG_MAP

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
RAW = DATA_DIR / "raw"
PROCESSED = DATA_DIR / "processed"


def load_books():
    return pd.read_csv(RAW / "books.csv")


def load_ratings():
    return pd.read_csv(RAW / "ratings.csv")


def load_tags():
    return pd.read_csv(RAW / "tags.csv")


def load_book_tags():
    return pd.read_csv(RAW / "book_tags.csv")


def build_genre_matrix(books=None, tags=None, book_tags=None, min_count=50):
    """Aggregates raw Goodreads shelf tags into canonical genre weights per
    book. Returns a DataFrame indexed by book_id, one column per canonical
    genre, values are summed tag-usage counts (unnormalized - callers scale
    as needed)."""
    books = books if books is not None else load_books()
    tags = tags if tags is not None else load_tags()
    book_tags = book_tags if book_tags is not None else load_book_tags()

    tags = tags.copy()
    tags["genre"] = tags["tag_name"].map(GENRE_TAG_MAP)
    mapped_tags = tags.dropna(subset=["genre"])[["tag_id", "genre"]]

    bt = book_tags[book_tags["count"] >= min_count].merge(mapped_tags, on="tag_id")

    # book_tags.csv keys on goodreads_book_id, not our sequential book_id
    gid_to_bid = books.set_index("goodreads_book_id")["book_id"]
    bt = bt.assign(book_id=bt["goodreads_book_id"].map(gid_to_bid)).dropna(subset=["book_id"])
    bt["book_id"] = bt["book_id"].astype(int)

    genre_matrix = bt.pivot_table(index="book_id", columns="genre", values="count", aggfunc="sum", fill_value=0)
    genre_matrix = genre_matrix.reindex(books["book_id"], fill_value=0)
    return genre_matrix


def primary_genre(genre_matrix):
    """Series: book_id -> single highest-weight genre, or None if no signal."""

    def top(row):
        row = row[row > 0]
        return row.idxmax() if len(row) else None

    return genre_matrix.apply(top, axis=1)


def top_genres(genre_matrix, n=3):
    """Series: book_id -> list of up to n highest-weight genres."""

    def top(row):
        row = row[row > 0].sort_values(ascending=False)
        return list(row.index[:n])

    return genre_matrix.apply(top, axis=1)

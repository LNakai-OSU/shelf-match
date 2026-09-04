"""
Parses a real Goodreads library export (Settings -> Export Library on
goodreads.com, downloads `goodreads_library_export.csv`) and matches its
rows against the goodbooks-10k catalog.

Goodreads' public API has been gone since 2020, so a personal export is the
only first-party way to bring a real reading history into an app like this
- there's no live lookup to fall back on.
"""

import io

import pandas as pd

from .matching import BookIndex


def parse_goodreads_export(file_bytes, books_df):
    """books_df: the goodbooks-10k books table (needs isbn, isbn13, title, authors).
    Returns a DataFrame with columns [book_id, rating, title, matched_on],
    one row per export row that could be matched and had a usable rating."""
    raw = pd.read_csv(io.BytesIO(file_bytes))
    raw.columns = [c.strip() for c in raw.columns]

    rating_col = "My Rating" if "My Rating" in raw.columns else None
    shelf_col = "Exclusive Shelf" if "Exclusive Shelf" in raw.columns else None
    if rating_col is None:
        raise ValueError("This doesn't look like a Goodreads export (no 'My Rating' column).")

    raw = raw[raw[rating_col].fillna(0) > 0]
    if shelf_col:
        raw = raw[raw[shelf_col].isin(["read", "currently-reading"]) | raw[shelf_col].isna()]

    author_col = "Author" if "Author" in raw.columns else "Author l-f"
    index = BookIndex(books_df)

    matches = []
    for _, row in raw.iterrows():
        book_row, matched_on = index.match(
            title=row.get("Title"),
            author=row.get(author_col),
            isbn=row.get("ISBN"),
            isbn13=row.get("ISBN13"),
        )
        if book_row is not None:
            matches.append(
                {
                    "book_id": int(book_row["book_id"]),
                    "rating": float(row[rating_col]),
                    "title": book_row["title"],
                    "matched_on": matched_on,
                }
            )

    result = pd.DataFrame(matches).drop_duplicates("book_id")
    return result, len(raw)

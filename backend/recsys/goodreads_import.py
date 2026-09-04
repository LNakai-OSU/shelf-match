"""
Parses a real Goodreads library export (Settings -> Export Library on
goodreads.com, downloads `goodreads_library_export.csv`) and matches its
rows against the goodbooks-10k catalog.

Goodreads' public API has been gone since 2020, so a personal export is the
only first-party way to bring a real reading history into an app like this
- there's no live lookup to fall back on. Matching goes ISBN13 -> ISBN ->
normalized title+author, since a meaningful fraction of export rows have a
blank ISBN (self-published/older editions Goodreads never filled in), and
title formatting differs just enough between Goodreads' own export and this
dataset's snapshot (series suffixes, subtitle punctuation) that exact
string equality misses real matches.
"""

import io
import re

import pandas as pd


def _clean_isbn(value):
    if pd.isna(value):
        return None
    s = str(value).strip().strip('="').strip('"')
    s = re.sub(r"[^0-9Xx]", "", s)
    return s or None


def _normalize_title(title):
    if pd.isna(title):
        return ""
    t = str(title).lower()
    t = re.sub(r"\(.*?\)", "", t)  # drop "(Series Name, #1)" suffixes
    t = re.sub(r"[^a-z0-9 ]", "", t)
    return re.sub(r"\s+", " ", t).strip()


def _normalize_author(author):
    if pd.isna(author):
        return ""
    a = str(author).lower().split(",")[0]  # first listed author only
    a = re.sub(r"[^a-z ]", "", a)
    return re.sub(r"\s+", " ", a).strip()


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

    raw["_isbn13"] = raw.get("ISBN13", pd.Series(dtype=object)).map(_clean_isbn)
    raw["_isbn"] = raw.get("ISBN", pd.Series(dtype=object)).map(_clean_isbn)
    raw["_norm_title"] = raw["Title"].map(_normalize_title)
    author_col = "Author" if "Author" in raw.columns else "Author l-f"
    raw["_norm_author"] = raw[author_col].map(_normalize_author) if author_col in raw.columns else ""

    books = books_df.copy()
    books["_isbn13"] = books["isbn13"].map(lambda x: _clean_isbn(str(int(x))) if pd.notna(x) else None)
    books["_isbn"] = books["isbn"].map(_clean_isbn)
    books["_norm_title"] = books["title"].map(_normalize_title)
    books["_norm_author"] = books["authors"].map(_normalize_author)

    by_isbn13 = books.dropna(subset=["_isbn13"]).drop_duplicates("_isbn13").set_index("_isbn13")
    by_isbn = books.dropna(subset=["_isbn"]).drop_duplicates("_isbn").set_index("_isbn")
    by_title_author = books.drop_duplicates(["_norm_title", "_norm_author"]).set_index(["_norm_title", "_norm_author"])

    matches = []
    for _, row in raw.iterrows():
        book_row, matched_on = None, None
        if row["_isbn13"] and row["_isbn13"] in by_isbn13.index:
            book_row, matched_on = by_isbn13.loc[row["_isbn13"]], "isbn13"
        elif row["_isbn"] and row["_isbn"] in by_isbn.index:
            book_row, matched_on = by_isbn.loc[row["_isbn"]], "isbn"
        elif (row["_norm_title"], row["_norm_author"]) in by_title_author.index:
            book_row, matched_on = by_title_author.loc[(row["_norm_title"], row["_norm_author"])], "title_author"

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

"""
Parses a store's uploaded book list into a custom inventory.

Real stores don't export data in one consistent shape, so this accepts
whatever's easiest for them to produce:
- a CSV with a "title" column (any case) and optional author/isbn/isbn13/
  quantity columns, in any order, under any of a few common header names, or
- a plain text list, one book per line, optionally "Title - Author" or
  "Title, Author" (a title alone still works - it just falls back to
  title-only matching, see `matching.BookIndex`).

Every row gets matched against the goodbooks-10k catalog the same way a
Goodreads export row does (ISBN13 -> ISBN -> title+author -> title alone) -
a store's list is just another external source with its own formatting
quirks, not a fundamentally different problem.
"""

import csv
import io
import re

import pandas as pd

from .matching import BookIndex

TITLE_KEYS = {"title", "book", "book title", "name"}
AUTHOR_KEYS = {"author", "authors", "writer"}
ISBN_KEYS = {"isbn"}
ISBN13_KEYS = {"isbn13", "isbn-13"}
QUANTITY_KEYS = {"quantity", "stock", "copies", "count", "qty"}


def _find_column(fieldnames, candidates):
    lookup = {f.strip().lower(): f for f in fieldnames}
    for key in candidates:
        if key in lookup:
            return lookup[key]
    return None


def _parse_csv_rows(text):
    reader = csv.DictReader(io.StringIO(text))
    fieldnames = reader.fieldnames or []
    title_col = _find_column(fieldnames, TITLE_KEYS)
    if title_col is None:
        return None  # not a recognizable CSV - caller falls back to plain text

    author_col = _find_column(fieldnames, AUTHOR_KEYS)
    isbn_col = _find_column(fieldnames, ISBN_KEYS)
    isbn13_col = _find_column(fieldnames, ISBN13_KEYS)
    qty_col = _find_column(fieldnames, QUANTITY_KEYS)

    rows = []
    for r in reader:
        title = (r.get(title_col) or "").strip()
        if not title:
            continue
        rows.append(
            {
                "title": title,
                "author": (r.get(author_col) or "").strip() if author_col else "",
                "isbn": (r.get(isbn_col) or "").strip() if isbn_col else "",
                "isbn13": (r.get(isbn13_col) or "").strip() if isbn13_col else "",
                "quantity": (r.get(qty_col) or "").strip() if qty_col else "",
            }
        )
    return rows


def _parse_plain_text_rows(text):
    rows = []
    for line in text.splitlines():
        line = line.strip().lstrip("-*•").strip()
        if not line or line.lower() in {"title", "book", "books"}:
            continue
        parts = re.split(r"\s+[-–—]\s+|,\s*", line, maxsplit=1)
        title = parts[0].strip()
        author = parts[1].strip() if len(parts) > 1 else ""
        rows.append({"title": title, "author": author, "isbn": "", "isbn13": "", "quantity": ""})
    return rows


def parse_store_upload(file_bytes, books_df, default_quantity=3):
    """Returns (matches_df[book_id, title, quantity, matched_on], total_rows)."""
    text = file_bytes.decode("utf-8-sig", errors="ignore")

    lines = text.splitlines()
    rows = _parse_csv_rows(text) if lines and "," in lines[0] else None
    if not rows:
        rows = _parse_plain_text_rows(text)

    index = BookIndex(books_df)
    matches = []
    for row in rows:
        book_row, matched_on = index.match(
            title=row["title"], author=row["author"], isbn=row["isbn"], isbn13=row["isbn13"]
        )
        if book_row is None:
            continue
        try:
            quantity = max(1, int(float(row["quantity"]))) if row["quantity"] else default_quantity
        except ValueError:
            quantity = default_quantity
        matches.append(
            {
                "book_id": int(book_row["book_id"]),
                "title": book_row["title"],
                "quantity": quantity,
                "matched_on": matched_on,
            }
        )

    result = pd.DataFrame(matches).drop_duplicates("book_id")
    return result, len(rows)

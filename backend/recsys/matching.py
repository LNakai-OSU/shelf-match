"""
Shared fuzzy book-matching: ISBN13 -> ISBN -> normalized title+author ->
normalized title alone, used by both the Goodreads-export importer and the
store-inventory importer.

Exact string equality misses real matches because title/author formatting
differs just enough between any outside source and this dataset's
snapshot: series suffixes ("(Harry Potter, #1)"), subtitle punctuation,
"Lastname, First" vs "First Lastname," Excel-escaped ISBNs (`="0439..."`).
"""

import re

import pandas as pd


def clean_isbn(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    s = str(value).strip().strip('="').strip('"')
    s = re.sub(r"[^0-9Xx]", "", s)
    return s or None


def normalize_title(title):
    if title is None or (isinstance(title, float) and pd.isna(title)):
        return ""
    t = str(title).lower()
    t = re.sub(r"\(.*?\)", "", t)  # drop "(Series Name, #1)" suffixes
    t = re.sub(r"[^a-z0-9 ]", "", t)
    return re.sub(r"\s+", " ", t).strip()


def _isbn_column_value(x):
    """Handles both goodbooks-10k's numeric (float, NaN-for-missing) isbn
    columns and the large catalog's plain-string ("" for missing) ones -
    str(9780312853129.0) is "9780312853129.0", and blindly stripping
    non-digit characters would keep the stray trailing 0 from ".0" and
    corrupt the ISBN, so a float needs int() first; a string doesn't."""
    if x is None or (isinstance(x, float) and pd.isna(x)) or x == "":
        return None
    if isinstance(x, float):
        return clean_isbn(str(int(x)))
    return clean_isbn(x)


def normalize_author(author):
    if author is None or (isinstance(author, float) and pd.isna(author)):
        return ""
    a = str(author).lower().split(",")[0]  # first listed author only
    a = re.sub(r"[^a-z ]", "", a)
    return re.sub(r"\s+", " ", a).strip()


class BookIndex:
    """Precomputed lookup indexes over the catalog for fast repeated matching."""

    def __init__(self, books_df):
        books = books_df.copy()
        books["_isbn13"] = books["isbn13"].map(_isbn_column_value)
        books["_isbn"] = books["isbn"].map(_isbn_column_value)
        books["_norm_title"] = books["title"].map(normalize_title)
        books["_norm_author"] = books["authors"].map(normalize_author)

        self.by_isbn13 = books.dropna(subset=["_isbn13"]).drop_duplicates("_isbn13").set_index("_isbn13")
        self.by_isbn = books.dropna(subset=["_isbn"]).drop_duplicates("_isbn").set_index("_isbn")
        self.by_title_author = books.drop_duplicates(["_norm_title", "_norm_author"]).set_index(
            ["_norm_title", "_norm_author"]
        )
        # title-only fallback (first catalog match wins on a duplicate title -
        # a known, accepted approximation when a source gives no author/ISBN)
        self.by_title_only = books.drop_duplicates(["_norm_title"]).set_index("_norm_title")

    def match(self, title=None, author=None, isbn=None, isbn13=None):
        """Returns (book_row, matched_on) or (None, None)."""
        isbn13c = clean_isbn(isbn13) if isbn13 else None
        isbnc = clean_isbn(isbn) if isbn else None
        norm_title = normalize_title(title) if title else ""
        norm_author = normalize_author(author) if author else ""

        if isbn13c and isbn13c in self.by_isbn13.index:
            return self.by_isbn13.loc[isbn13c], "isbn13"
        if isbnc and isbnc in self.by_isbn.index:
            return self.by_isbn.loc[isbnc], "isbn"
        if norm_title and norm_author and (norm_title, norm_author) in self.by_title_author.index:
            return self.by_title_author.loc[(norm_title, norm_author)], "title_author"
        if norm_title and norm_title in self.by_title_only.index:
            return self.by_title_only.loc[norm_title], "title_only"
        return None, None

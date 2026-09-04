"""
Regression test for a real bug: BookIndex's ISBN cleaning used to assume
ISBN13 was always a numeric column (goodbooks-10k's schema) and crashed
with ValueError on the large catalog's plain-string schema (empty string
for a missing ISBN instead of NaN). See README "Real bugs this surfaced."

Run directly: python test_matching.py
"""

import pandas as pd

from recsys.matching import BookIndex


def test_string_isbn_schema():
    """The large catalog (books_large.csv): isbn13 is a string, "" for missing."""
    df = pd.DataFrame(
        {
            "book_id": [1, 2],
            "title": ["Foo", "Bar"],
            "authors": ["A. Author", "B. Author"],
            "isbn": ["", "1234567890"],
            "isbn13": ["", "9781234567897"],
        }
    )
    idx = BookIndex(df)  # must not raise
    assert list(idx.by_isbn13.index) == ["9781234567897"]


def test_float_isbn_schema():
    """goodbooks-10k's original schema: isbn13 is a float, NaN for missing."""
    df = pd.DataFrame(
        {
            "book_id": [1],
            "title": ["Foo"],
            "authors": ["A. Author"],
            "isbn": ["0312853122"],
            "isbn13": [9780312853129.0],
        }
    )
    idx = BookIndex(df)  # must not raise, and must not corrupt the ISBN via the ".0" suffix
    assert list(idx.by_isbn13.index) == ["9780312853129"]


if __name__ == "__main__":
    test_string_isbn_schema()
    test_float_isbn_schema()
    print("OK - both ISBN column schemas handled correctly")

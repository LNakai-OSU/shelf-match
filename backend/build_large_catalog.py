"""
Builds a much larger book catalog than goodbooks-10k (10,000 books, no
description text) from the UCSD Goodreads Book Graph (Wan & McAuley,
RecSys'18) - 2.36M books, downloaded in full by fetch_large_dataset.sh
into data/raw_large/.

goodreads_books.json.gz is ~2GB compressed / decompresses to many times
that - not something to hold in memory as 2.36M Python dicts. This streams
it as newline-delimited JSON straight out of gzip, and keeps only a
bounded streaming top-K (by ratings_count) via a min-heap, so peak memory
is O(MAX_BOOKS), never O(total records). Only English-language books with
a non-empty description are eligible, since the description is exactly
what powers semantic search - a book with no description text is useless
to embed no matter how popular it is.

Run after downloading the raw files (see README):
    python build_large_catalog.py
"""

import gzip
import heapq
import itertools
import json
import time
from pathlib import Path

import pandas as pd

RAW = Path(__file__).parent.parent / "data" / "raw_large"
PROCESSED = Path(__file__).parent.parent / "data" / "processed"

MAX_BOOKS = 75_000
PROGRESS_EVERY = 200_000


def load_authors():
    authors = {}
    with gzip.open(RAW / "goodreads_book_authors.json.gz", "rt") as f:
        for line in f:
            rec = json.loads(line)
            authors[rec["author_id"]] = rec["name"]
    print(f"loaded {len(authors):,} authors")
    return authors


def load_genres():
    genres = {}
    with gzip.open(RAW / "goodreads_book_genres_initial.json.gz", "rt") as f:
        for line in f:
            rec = json.loads(line)
            g = rec.get("genres") or {}
            if g:
                genres[rec["book_id"]] = g
    print(f"loaded genres for {len(genres):,} books")
    return genres


def _to_int(v, default=0):
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _to_float(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def stream_top_books():
    heap = []  # (ratings_count, tiebreak, record)
    counter = itertools.count()
    seen = 0
    kept_candidates = 0
    t0 = time.time()

    with gzip.open(RAW / "books_full.json.gz", "rt") as f:
        for line in f:
            seen += 1
            if seen % PROGRESS_EVERY == 0:
                print(f"  scanned {seen:,}  heap={len(heap):,}  ({time.time() - t0:.0f}s)")
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            description = (rec.get("description") or "").strip()
            if len(description) < 20:
                continue
            lang = rec.get("language_code") or ""
            if lang and not lang.startswith("en"):
                continue
            title = (rec.get("title_without_series") or rec.get("title") or "").strip()
            if not title:
                continue

            ratings_count = _to_int(rec.get("ratings_count"))
            kept_candidates += 1
            key = (ratings_count, next(counter))
            if len(heap) < MAX_BOOKS:
                heapq.heappush(heap, (*key, rec))
            elif ratings_count > heap[0][0]:
                heapq.heapreplace(heap, (*key, rec))

    print(f"scanned {seen:,} total records, {kept_candidates:,} eligible (has description, English), "
          f"kept top {len(heap):,} by ratings_count in {time.time() - t0:.0f}s")
    return [r for *_ , r in heap]


def main():
    authors = load_authors()
    genres = load_genres()
    top_books = stream_top_books()

    rows = []
    for rec in top_books:
        author_names = [authors.get(a["author_id"], "") for a in rec.get("authors", [])]
        author_names = [a for a in author_names if a]
        g = genres.get(rec["book_id"], {})
        top_genre = max(g, key=g.get) if g else ""
        rows.append(
            {
                "goodreads_book_id": rec["book_id"],
                "title": rec.get("title_without_series") or rec.get("title"),
                "authors": ", ".join(author_names) if author_names else "Unknown",
                "description": (rec.get("description") or "").strip(),
                "average_rating": _to_float(rec.get("average_rating")),
                "ratings_count": _to_int(rec.get("ratings_count")),
                "text_reviews_count": _to_int(rec.get("text_reviews_count")),
                "publication_year": _to_int(rec.get("publication_year")) or None,
                "isbn": rec.get("isbn") or "",
                "isbn13": rec.get("isbn13") or "",
                "image_url": rec.get("image_url") or "",
                "genre_weights": json.dumps(g),
                "primary_genre": top_genre,
            }
        )

    df = pd.DataFrame(rows).sort_values("ratings_count", ascending=False).reset_index(drop=True)
    df.insert(0, "book_id", range(1, len(df) + 1))

    PROCESSED.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED / "books_large.csv"
    df.to_csv(out_path, index=False)
    print(f"\nwrote {len(df):,} books -> {out_path}")
    print("\nprimary genre distribution:")
    print(df["primary_genre"].value_counts())


if __name__ == "__main__":
    main()

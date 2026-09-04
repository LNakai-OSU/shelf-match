"""
Filters the UCSD Goodreads interactions data down to explicit ratings
(rating > 0) for books that made it into books_large.csv, remapped from
Goodreads' own anonymized ids to our catalog's sequential book_id.

Only the first ~600MB of the full ~4.3GB goodreads_interactions.csv was
downloaded (an HTTP Range request, since the whole file is well beyond
what's practical here) - the file is sorted by user_id, so this is every
interaction from the first N users, not a uniform random sample across
all ~876K users in the full dataset. That's a real, accepted trade-off
for this project's scale, documented here rather than dressed up as a
proper random sample: a partial-but-real slice of real user behavior.

Run after build_large_catalog.py:
    python build_large_interactions.py
"""

from pathlib import Path

import pandas as pd

RAW = Path(__file__).parent.parent / "data" / "raw_large"
PROCESSED = Path(__file__).parent.parent / "data" / "processed"

MIN_RATINGS_PER_USER = 5


def main():
    books = pd.read_csv(PROCESSED / "books_large.csv")
    goodreads_to_our_id = dict(zip(books["goodreads_book_id"].astype(str), books["book_id"]))

    id_map = pd.read_csv(RAW / "book_id_map.csv", dtype=str)
    csv_to_goodreads = dict(zip(id_map["book_id_csv"], id_map["book_id"]))

    print("streaming interactions_prefix.csv...")
    kept_chunks = []
    total = 0
    for chunk in pd.read_csv(RAW / "interactions_prefix.csv", dtype=str, chunksize=1_000_000):
        total += len(chunk)
        chunk = chunk[chunk["rating"].fillna("0").astype(int) > 0].copy()
        chunk["goodreads_book_id"] = chunk["book_id"].map(csv_to_goodreads)
        chunk["our_book_id"] = chunk["goodreads_book_id"].map(goodreads_to_our_id)
        chunk = chunk.dropna(subset=["our_book_id"])
        if len(chunk):
            kept_chunks.append(chunk[["user_id", "our_book_id", "rating"]])
        print(f"  scanned {total:,} rows, kept so far {sum(len(c) for c in kept_chunks):,}")

    result = pd.concat(kept_chunks, ignore_index=True)
    result.columns = ["user_id", "book_id", "rating"]
    result = result.astype({"user_id": int, "book_id": int, "rating": int})

    user_counts = result["user_id"].value_counts()
    keep_users = user_counts[user_counts >= MIN_RATINGS_PER_USER].index
    before = len(result)
    result = result[result["user_id"].isin(keep_users)]

    out_path = PROCESSED / "ratings_large.csv"
    result.to_csv(out_path, index=False)
    print(f"\ntotal interaction rows scanned: {total:,}")
    print(f"rated + in-catalog rows: {before:,}")
    print(f"after requiring >= {MIN_RATINGS_PER_USER} ratings/user: {len(result):,} rows, "
          f"{result.user_id.nunique():,} users, {result.book_id.nunique():,} books")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

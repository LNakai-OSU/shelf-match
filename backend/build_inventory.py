"""
Builds a simulated indie-bookstore inventory: a curated subset of the
goodbooks-10k catalog, plus per-book genre labels used by both the
inventory-construction logic here and the content-based recommender.

There's no real store's real inventory behind this - getting one would mean
scraping a specific shop's website, which is fragile and out of scope for a
portfolio piece. Instead this constructs a *plausible* one, transparently:

1. Every book gets a primary genre from `data_loader.build_genre_matrix`
   (aggregated Goodreads shelf tags -> a curated genre map).
2. Books are grouped into broader "sections" (`genre_tags.GENRE_SECTION`) -
   what a real store's shelves would be labeled.
3. Each section's shelf-space quota is proportional to that section's share
   of the *full* 10k-book catalog (so the inventory's genre mix reflects
   real reader demand, not an invented ratio).
4. Within a section, books are sampled without replacement, weighted by
   sqrt(ratings_count) - popular books are more likely to be carried, but
   it's not just a top-N cut. At INVENTORY_SIZE ~= 18% of the full catalog,
   most books - including plenty of well-known ones - simply aren't
   selected. Those "misses" are exactly the case the recommender has to
   handle: the ideal recommendation often isn't on the shelf.

Re-run any time after `fetch_data.py`; it's deterministic given SEED.
"""

import numpy as np
import pandas as pd

from recsys.data_loader import (
    PROCESSED,
    build_genre_matrix,
    load_book_tags,
    load_books,
    load_tags,
    primary_genre,
    top_genres,
)
from recsys.genre_tags import GENRE_SECTION

SEED = 42
INVENTORY_SIZE = 1800
MIN_SECTION_QUOTA = 15


def main():
    rng = np.random.default_rng(SEED)

    books = load_books()
    tags = load_tags()
    book_tags = load_book_tags()
    genre_matrix = build_genre_matrix(books, tags, book_tags)

    pg = primary_genre(genre_matrix)
    tg = top_genres(genre_matrix, n=4)
    books = books.set_index("book_id")
    books["primary_genre"] = pg
    books["genres"] = tg.apply(lambda g: "|".join(g) if g else "")
    books["section"] = books["primary_genre"].map(GENRE_SECTION).fillna("General")

    section_counts = books["section"].value_counts()
    total = section_counts.sum()
    quotas = (section_counts / total * INVENTORY_SIZE).round().astype(int)
    quotas = quotas.clip(lower=MIN_SECTION_QUOTA)

    chosen_ids = []
    for section, quota in quotas.items():
        pool = books[books["section"] == section]
        quota = min(quota, len(pool))
        weights = np.sqrt(pool["ratings_count"].to_numpy(dtype=float))
        weights = weights / weights.sum()
        picked = rng.choice(pool.index.to_numpy(), size=quota, replace=False, p=weights)
        chosen_ids.extend(picked.tolist())

    books["in_inventory"] = books.index.isin(chosen_ids)

    # a little shelf-quantity flavor for the UI - more copies of popular books
    stock = np.where(
        books["in_inventory"],
        1 + rng.poisson(lam=np.clip(books["ratings_count"] / books["ratings_count"].max(), 0.05, 1) * 4),
        0,
    )
    books["stock_quantity"] = stock

    PROCESSED.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED / "books_enriched.csv"
    books.reset_index().to_csv(out_path, index=False)

    n_in = int(books["in_inventory"].sum())
    print(f"catalog: {len(books)} books -> inventory: {n_in} ({n_in / len(books):.1%})")
    print("\nsection quotas (target -> actual):")
    actual = books[books["in_inventory"]]["section"].value_counts()
    for section in quotas.index:
        print(f"  {section:28s} target={quotas[section]:>4d}  actual={actual.get(section, 0):>4d}")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()

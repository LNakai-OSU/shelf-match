"""
Builds a simulated indie-bookstore inventory: a curated subset of the
large catalog (books_large.csv, ~75k books from the UCSD Goodreads Book
Graph).

There's no real store's real inventory behind this - getting one would
mean scraping a specific shop's website, which is fragile and out of
scope for a portfolio piece. Instead this constructs a *plausible* one,
transparently:

1. Every book already has a clean primary genre (see recsys/genres.py -
   the UCSD dataset ships pre-bucketed into ~10 top-level genres, unlike
   goodbooks-10k's noisy folksonomy tags).
2. Each section's shelf-space quota is proportional to that section's
   share of the *full* catalog (so the inventory's genre mix reflects
   real reader demand, not an invented ratio).
3. Within a section, books are sampled without replacement, weighted by
   sqrt(ratings_count) - popular books are more likely to be carried, but
   it's not just a top-N cut. At INVENTORY_SIZE ~= 8% of the full catalog,
   the vast majority of books - including plenty of well-known ones -
   simply aren't selected. Those "misses" are exactly the case the
   recommender has to handle: the ideal recommendation often isn't on
   the shelf.

Re-run any time after build_large_catalog.py; it's deterministic given SEED.
"""

import numpy as np
import pandas as pd

from recsys.data_loader import PROCESSED
from recsys.genres import section_for

SEED = 42
INVENTORY_SIZE = 6000
MIN_SECTION_QUOTA = 40


def main():
    rng = np.random.default_rng(SEED)

    books = pd.read_csv(PROCESSED / "books_large.csv").set_index("book_id")
    books["primary_genre"] = books["primary_genre"].fillna("")
    books["section"] = books["primary_genre"].map(section_for)

    section_counts = books["section"].value_counts()
    total = section_counts.sum()
    quotas = (section_counts / total * INVENTORY_SIZE).round().astype(int)
    quotas = quotas.clip(lower=MIN_SECTION_QUOTA)

    chosen_ids = []
    for section, quota in quotas.items():
        pool = books[books["section"] == section]
        quota = min(quota, len(pool))
        weights = np.sqrt(pool["ratings_count"].clip(lower=1).to_numpy(dtype=float))
        weights = weights / weights.sum()
        picked = rng.choice(pool.index.to_numpy(), size=quota, replace=False, p=weights)
        chosen_ids.extend(picked.tolist())

    books["in_inventory"] = books.index.isin(chosen_ids)

    stock = np.where(
        books["in_inventory"],
        1 + rng.poisson(lam=np.clip(books["ratings_count"] / books["ratings_count"].max(), 0.05, 1) * 4),
        0,
    )
    books["stock_quantity"] = stock

    out_path = PROCESSED / "books_large.csv"
    books.reset_index().to_csv(out_path, index=False)

    n_in = int(books["in_inventory"].sum())
    print(f"catalog: {len(books):,} books -> inventory: {n_in:,} ({n_in / len(books):.1%})")
    print("\nsection quotas (target -> actual):")
    actual = books[books["in_inventory"]]["section"].value_counts()
    for section in quotas.index:
        print(f"  {section:35s} target={quotas[section]:>5d}  actual={actual.get(section, 0):>5d}")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()

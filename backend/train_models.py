"""
Fits the content-based and collaborative-filtering models and saves their
artifacts to data/processed/ for the API to load at startup.

Run after fetch_data.py + build_inventory.py:
    python train_models.py
"""

import pickle
import time

import numpy as np
import pandas as pd

from recsys.collaborative_model import MatrixFactorization
from recsys.content_model import ContentModel, build_documents
from recsys.data_loader import PROCESSED, build_genre_matrix, load_book_tags, load_ratings, load_tags

SEED = 42


def fit_content_model(books):
    print("building genre matrix / content documents...")
    genre_matrix = build_genre_matrix(books, load_tags(), load_book_tags())
    docs = build_documents(books.set_index("book_id"), genre_matrix)
    model = ContentModel(books["book_id"].to_numpy(), docs)
    print(f"  content model: {model.matrix.shape[0]} books x {model.matrix.shape[1]} tf-idf terms")
    return model


def fit_collaborative_model(n_items):
    print("loading ratings...")
    ratings = load_ratings()
    rng = np.random.default_rng(SEED)

    user_ids = ratings["user_id"].unique()
    user_id_to_idx = {u: i for i, u in enumerate(user_ids)}
    # book_id in ratings.csv is already the 1..10000 sequential id used elsewhere;
    # models still index items 0..n_items-1 internally.
    item_idx_all = ratings["book_id"].to_numpy() - 1
    user_idx_all = ratings["user_id"].map(user_id_to_idx).to_numpy()
    rating_vals = ratings["rating"].to_numpy(dtype=float)

    n = len(ratings)
    perm = rng.permutation(n)
    val_size = int(n * 0.05)
    val_idx, train_idx = perm[:val_size], perm[val_size:]

    print(f"training matrix factorization on {len(train_idx):,} ratings "
          f"({len(user_ids):,} users x {n_items:,} books), validating on {len(val_idx):,}...")

    model = MatrixFactorization(n_users=len(user_ids), n_items=n_items, seed=SEED)
    t0 = time.time()
    model.fit(
        user_idx_all[train_idx], item_idx_all[train_idx], rating_vals[train_idx],
        val_user_idx=user_idx_all[val_idx], val_item_idx=item_idx_all[val_idx], val_ratings=rating_vals[val_idx],
    )
    print(f"  trained in {time.time() - t0:.1f}s")
    return model, user_id_to_idx


def main():
    books = pd.read_csv(PROCESSED / "books_enriched.csv")
    n_items = books["book_id"].max()  # item indices are 0-based book_id-1, sized to the full id range

    content_model = fit_content_model(books)
    cf_model, user_id_to_idx = fit_collaborative_model(n_items)

    PROCESSED.mkdir(parents=True, exist_ok=True)

    with open(PROCESSED / "content_model.pkl", "wb") as f:
        pickle.dump(content_model, f)

    with open(PROCESSED / "cf_model.pkl", "wb") as f:
        pickle.dump({"model": cf_model, "user_id_to_idx": user_id_to_idx}, f)

    print("\nsaved content_model.pkl and cf_model.pkl to", PROCESSED)


if __name__ == "__main__":
    main()

"""
Trains the semantic embedding model and the collaborative-filtering model
on the large catalog (books_large.csv / ratings_large.csv), saving
artifacts to data/processed/ for the API to load at startup.

Run after build_large_catalog.py and build_large_interactions.py:
    python train_large_models.py
"""

import pickle
import time

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from recsys.collaborative_model import MatrixFactorization
from recsys.data_loader import PROCESSED
from recsys.semantic_model import SemanticModel

SEED = 42


def fit_semantic_model(books):
    print(f"encoding {len(books):,} book descriptions with {SemanticModel.MODEL_NAME}...")
    t0 = time.time()
    model = SentenceTransformer(SemanticModel.MODEL_NAME)
    semantic_model, _ = SemanticModel.build(books.set_index("book_id"), model=model)
    print(f"  encoded in {time.time() - t0:.0f}s -> {semantic_model.embeddings.shape}")
    return semantic_model


def fit_collaborative_model(n_items):
    print("loading ratings_large.csv...")
    ratings = pd.read_csv(PROCESSED / "ratings_large.csv")
    rng = np.random.default_rng(SEED)

    user_ids = ratings["user_id"].unique()
    user_id_to_idx = {u: i for i, u in enumerate(user_ids)}
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
    return model


def main():
    books = pd.read_csv(PROCESSED / "books_large.csv")
    n_items = int(books["book_id"].max())

    semantic_model = fit_semantic_model(books)
    cf_model = fit_collaborative_model(n_items)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    with open(PROCESSED / "semantic_model.pkl", "wb") as f:
        pickle.dump(semantic_model, f)
    with open(PROCESSED / "cf_model_large.pkl", "wb") as f:
        pickle.dump({"model": cf_model}, f)

    print("\nsaved semantic_model.pkl and cf_model_large.pkl to", PROCESSED)


if __name__ == "__main__":
    main()

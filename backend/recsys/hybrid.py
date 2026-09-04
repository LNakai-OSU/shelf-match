"""
Combines the content-based and collaborative scores and - the actual point
of this project - ranks only books that are in the simulated store's
inventory.

A generic recommender ranks the whole catalog and hopes the top picks are
purchasable somewhere. This one only ever surfaces books the store can put
in the customer's hands today: both single-model rankings and the hybrid
are computed by scoring the *inventory* subset only, so a book with a
perfect collaborative-filtering match but zero shelf copies simply never
appears - there's no separate "filter out of stock" pass bolted on
afterward, because a top-N-then-filter approach could return fewer than N
results whenever the ideal picks aren't stocked.
"""

import numpy as np


def _minmax(scores):
    lo, hi = scores.min(), scores.max()
    if hi - lo < 1e-9:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


def content_scores_for_inventory(content_model, inventory_book_ids, liked_book_ids, ratings=None):
    profile = content_model.profile_vector(liked_book_ids, ratings)
    if profile is None:
        return np.zeros(len(inventory_book_ids))
    all_scores = content_model.score_all_items(profile)
    row_of = content_model.book_id_to_row
    return np.array([all_scores[row_of[b]] if b in row_of else 0.0 for b in inventory_book_ids])


def collaborative_scores_for_inventory(cf_model, inventory_book_ids, liked_item_idx, liked_ratings):
    user_bias, user_vector = cf_model.fold_in_user(liked_item_idx, liked_ratings)
    all_scores = cf_model.score_all_items(user_bias, user_vector)
    item_idx = np.array(inventory_book_ids) - 1  # book_id is 1-based, item index is 0-based
    return all_scores[item_idx]


def rank_by_profile(content_model, inventory_book_ids, profile_vector, top_n=12, exclude_book_ids=None):
    """Ranks the inventory by content-model similarity to an arbitrary
    profile vector (e.g. from a free-text genre query rather than a set of
    liked books) - the customer "describe what you want" path, which has
    no ratings to fold into the collaborative model at all."""
    exclude = set(exclude_book_ids or [])
    candidate_ids = np.array([b for b in inventory_book_ids if b not in exclude])
    all_scores = content_model.score_all_items(profile_vector)
    row_of = content_model.book_id_to_row
    scores = np.array([all_scores[row_of[b]] if b in row_of else 0.0 for b in candidate_ids])
    order = np.argsort(-scores)[:top_n]
    return [(int(candidate_ids[i]), float(scores[i])) for i in order]


def hybrid_rank(
    content_model,
    cf_model,
    inventory_book_ids,
    liked_book_ids,
    liked_ratings,
    alpha=0.5,
    top_n=10,
    exclude_book_ids=None,
):
    """Returns three ranked lists (content-only, collaborative-only, hybrid),
    each a list of (book_id, score) over the inventory only."""
    exclude = set(exclude_book_ids or [])
    candidate_ids = np.array([b for b in inventory_book_ids if b not in exclude])

    content_scores = content_scores_for_inventory(content_model, candidate_ids, liked_book_ids, liked_ratings)
    liked_item_idx = np.array(liked_book_ids) - 1
    cf_scores = collaborative_scores_for_inventory(cf_model, candidate_ids, liked_item_idx, liked_ratings)

    content_norm = _minmax(content_scores)
    cf_norm = _minmax(cf_scores)
    hybrid_scores = alpha * cf_norm + (1 - alpha) * content_norm

    def top(scores):
        order = np.argsort(-scores)[:top_n]
        return [(int(candidate_ids[i]), float(scores[i])) for i in order]

    return {
        "content": top(content_scores),
        "collaborative": top(cf_scores),
        "hybrid": top(hybrid_scores),
    }

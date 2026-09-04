"""
Content-based recommender: TF-IDF over each book's genre-tag profile and
author, cosine similarity for ranking.

Unlike the collaborative model, this needs no ratings history at all to
score a book - only its own metadata - so it's what keeps this system
useful for a genuinely cold-start visitor (someone who names a few
favorite books the collaborative model may have very little signal on) and
is the fallback when a user's collaborative-filtering candidates aren't in
this store's inventory.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

AUTHOR_WEIGHT = 3  # how many times an author token is repeated relative to a genre token


def _author_tokens(authors_field):
    return [a.strip().lower().replace(" ", "_").replace(".", "") for a in str(authors_field).split(",") if a.strip()]


def build_documents(books_df, genre_matrix):
    """books_df indexed by book_id, must have an 'authors' column.
    genre_matrix indexed by book_id, one column per canonical genre."""
    docs = []
    for book_id in books_df.index:
        row = genre_matrix.loc[book_id]
        max_w = row.max()
        weights = (row / max_w * 5).round().astype(int) if max_w > 0 else row.astype(int)
        genre_tokens = []
        for genre, w in weights.items():
            if w > 0:
                genre_tokens.extend([genre.replace("-", "_")] * int(w))
        author_tokens = _author_tokens(books_df.loc[book_id, "authors"]) * AUTHOR_WEIGHT
        docs.append(" ".join(genre_tokens + author_tokens))
    return docs


class ContentModel:
    def __init__(self, book_ids, documents):
        self.book_ids = np.asarray(book_ids)
        self.book_id_to_row = {bid: i for i, bid in enumerate(self.book_ids)}
        self.vectorizer = TfidfVectorizer(token_pattern=r"[^\s]+")
        self.matrix = self.vectorizer.fit_transform(documents)

    def profile_vector(self, liked_book_ids, ratings=None):
        rows = [self.book_id_to_row[b] for b in liked_book_ids if b in self.book_id_to_row]
        if not rows:
            return None
        if ratings is None:
            weights = np.ones(len(rows))
        else:
            weights = np.clip(np.asarray(ratings, dtype=float) - 1.5, 0.2, None)
        sub = self.matrix[rows]
        weighted = sub.multiply(weights[:, None])
        return np.asarray(weighted.sum(axis=0))

    def profile_from_document(self, document):
        """Same vector space as profile_vector, but from an arbitrary token
        string (e.g. genre keywords detected in a free-text query) instead
        of an existing book's own document."""
        return self.vectorizer.transform([document])

    def score_all_items(self, profile_vector):
        """Returns a book_id -> cosine-similarity score array (aligned with self.book_ids)."""
        sims = cosine_similarity(profile_vector, self.matrix)[0]
        return sims

    def similar_to(self, book_id, top_n=10):
        row = self.book_id_to_row.get(book_id)
        if row is None:
            return []
        sims = cosine_similarity(self.matrix[row], self.matrix)[0]
        order = np.argsort(-sims)
        return [(self.book_ids[i], sims[i]) for i in order if self.book_ids[i] != book_id][:top_n]

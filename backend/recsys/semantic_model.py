"""
Real semantic search over book descriptions, via sentence embeddings
(`sentence-transformers/all-MiniLM-L6-v2`, 384-dim) - not keyword/genre-tag
matching.

goodbooks-10k (the original dataset this project started with) has no
description field at all, only Goodreads' folksonomy shelf tags - so the
first version's free-text search worked by whole-word-matching a query
against ~300 known tag phrases and could only ever recognize a query that
happened to name a genre outright. A phrase like "stylistically
groundbreaking with an intriguing plot" has no genre tag and matches
nothing in that scheme, by construction - it isn't a genre, it's a
judgment about prose style and narrative craft that only shows up in the
text of an actual description.

The UCSD Goodreads Book Graph (books_large.csv, see build_large_catalog.py)
does have real description text for every book, so this embeds each
book's title + description + genre labels into the same 384-dim space a
query gets embedded into, and ranks by cosine similarity. This is a
meaning-based comparison, not a word-overlap one: two descriptions that
both talk about "an unconventional narrative structure" and "defying
genre expectations" land close together in that space even if they share
almost no vocabulary with a query about being "stylistically
groundbreaking" - see the model card / README for a worked example.

all-MiniLM-L6-v2 was picked over a larger model specifically for
CPU-friendly encode speed (a few hundred docs/sec on CPU) at 75k books -
a bigger model would likely score marginally better but cost meaningfully
more to encode the whole catalog and every live query.
"""

import numpy as np


class SemanticModel:
    MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self, book_ids, embeddings):
        """embeddings: float32 array (n_books, dim), L2-normalized, aligned with book_ids."""
        self.book_ids = np.asarray(book_ids)
        self.book_id_to_row = {bid: i for i, bid in enumerate(self.book_ids)}
        self.embeddings = embeddings

    @staticmethod
    def _document(title, description, genre_label):
        parts = [title or ""]
        if genre_label:
            parts.append(f"Genre: {genre_label}.")
        if description:
            parts.append(description)
        return " ".join(parts)

    @classmethod
    def build(cls, books_df, model=None, batch_size=128, progress=True):
        """books_df indexed by book_id, needs title/description/primary_genre columns."""
        from sentence_transformers import SentenceTransformer

        model = model or SentenceTransformer(cls.MODEL_NAME)
        docs = [
            cls._document(row.title, row.description, row.primary_genre)
            for row in books_df.itertuples()
        ]
        embeddings = model.encode(
            docs,
            batch_size=batch_size,
            show_progress_bar=progress,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)
        return cls(books_df.index.to_numpy(), embeddings), model

    def encode_query(self, text, model):
        emb = model.encode([text], normalize_embeddings=True, convert_to_numpy=True)
        return emb[0].astype(np.float32)

    def profile_vector(self, liked_book_ids, ratings=None):
        """Same name/shape as ContentModel.profile_vector so hybrid.py can
        treat either model interchangeably as the "content" scorer."""
        rows = [self.book_id_to_row[b] for b in liked_book_ids if b in self.book_id_to_row]
        if not rows:
            return None
        if ratings is None:
            weights = np.ones(len(rows))
        else:
            weights = np.clip(np.asarray(ratings, dtype=float) - 1.5, 0.2, None)
        vecs = self.embeddings[rows]
        profile = (vecs * weights[:, None]).sum(axis=0)
        norm = np.linalg.norm(profile)
        return profile / norm if norm > 0 else profile

    def score_all_items(self, vector):
        """vector must already be L2-normalized (query/profile embeddings from this
        class always are) - cosine similarity reduces to a single dot product."""
        return self.embeddings @ vector

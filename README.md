# Shelf Match

A book recommender constrained to what a small store actually has on the
shelf, not the whole internet's catalog. Trained on real Goodreads rating
data, but every recommendation is filtered down to a simulated indie
bookstore's curated inventory - so the interesting engineering problem
isn't "recommend the best book," it's "recommend the best book we can
actually hand the customer today."

- `backend/` - Python. `recsys/` holds the content-based model, the
  collaborative-filtering model, and the hybrid ranker; `api/` is a thin
  FastAPI layer over them.
- `frontend/` - React + Vite. Search-and-rate or import a real Goodreads
  export to build a profile, then compare content-based vs. collaborative
  vs. hybrid recommendations side by side.

## Why this dataset, not a live Goodreads pull

Goodreads shut down its public API in 2020, and scraping goodreads.com
directly would violate its Terms of Service - a bad foundation for a
portfolio piece. Instead this uses
[goodbooks-10k](https://github.com/zygmuntz/goodbooks-10k) (Zajac): 10,000
books and about 6 million ratings from ~53,000 real Goodreads users,
already collected and hosted as plain CSVs. `fetch_data.py` downloads it
directly - no Kaggle account/API key needed.

A real user's history still gets to participate: `Import Goodreads export`
in the app accepts an actual `goodreads_library_export.csv` (Settings ->
Export Library on goodreads.com) and matches its rows against the catalog
by ISBN13, ISBN, then normalized title+author.

## Running it

```bash
# Terminal 1 - backend
cd backend
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python fetch_data.py        # downloads goodbooks-10k (~100MB)
python build_inventory.py   # builds the simulated store inventory
python train_models.py      # fits the content + collaborative models (~2 min)
uvicorn api.main:app --reload --port 8020

# Terminal 2 - frontend
cd frontend
npm install
npm run dev   # http://localhost:5173
```

## The constrained-ranking problem

A generic recommender ranks the whole catalog and hopes the top pick is
purchasable somewhere. That's not what a bookstore's website needs - it
needs to rank *only what's on the shelf*. This project treats that as the
actual problem statement, not an afterthought:

- `build_inventory.py` builds a simulated ~1,900-book inventory (about 19%
  of the full catalog) by grouping books into sections (genre tags rolled
  up via `recsys/genre_tags.py`), giving each section a shelf-space quota
  proportional to its share of the full catalog, and sampling within a
  section weighted by `sqrt(ratings_count)` - popular books are more
  likely to be carried, but it's not simply "stock the top N." Even among
  the 50 most-rated books on Goodreads, **22% aren't in this store's
  inventory** (including *To Kill a Mockingbird* and *The Catcher in the
  Rye* in the sampled run) - realistic misses a recommender has to route
  around, not an edge case.
- `recsys/hybrid.py` scores *only* inventory books for content, collaborative,
  and hybrid ranking alike. There's no "rank everything, then filter to
  in-stock" pass - that approach can silently return fewer than N results
  whenever the ideal picks aren't stocked. Scoring the constrained set
  directly means the recommender is always solving the actual problem: best
  available, not best hypothetical.

## Two models, compared, plus a hybrid

**Content-based** (`recsys/content_model.py`): TF-IDF over each book's
genre-tag profile (aggregated from Goodreads' folksonomy tags - see
`genre_tags.py` for how ~300 raw shelf tags like `ya-fantasy`,
`sci-fi-fantasy`, `mystery-suspense` collapse into ~40 canonical genres)
plus its author, cosine similarity for ranking. Needs no ratings history
at all - just the book's own metadata - so it's what keeps the system
useful for a genuinely cold visitor.

**Collaborative filtering** (`recsys/collaborative_model.py`): a
latent-factor model (`rating ≈ global_mean + user_bias + item_bias + U·V`,
40 factors) trained with full-batch gradient descent on the ~6M observed
ratings only - not TruncatedSVD on the raw sparse matrix, which would
treat every unrated (user, book) pair as a rating of zero and bias the
whole model toward predicting low ratings everywhere. Final validation
RMSE: **0.857** (on the 1-5 scale, ~5% held out).

Since a real visitor isn't one of the 53k training users, `fold_in_user`
solves a small ridge-regression problem - given a few stated (book,
rating) pairs, find the latent vector that best explains them against the
already-trained item factors. This "folding-in" is standard for
latent-factor models and is what makes the collaborative model usable for
a new user's stated favorites at all.

**Collaborative filtering's honest limit, in this app specifically:** even
after fixing the bias-shrinkage bug below, folding in a handful of stated
ratings (tested with both 6 and 12) isn't enough signal to meaningfully
beat a "solve for 40 latent dimensions from a dozen data points"
regression - the raw collaborative-only column tends to surface broadly
beloved books (Calvin & Hobbes compilations genuinely average 4.8+ on
Goodreads) rather than anything specific to a stated Harry Potter/A Song
of Ice and Fire profile. This is a well-known, real characteristic of
latent-factor cold start, not a leftover bug - it's exactly why the hybrid
column exists: blending in the content model, which needs no ratings
history at all, is what keeps the top recommendations genre-relevant when
collaborative filtering alone can't yet say much about a brand-new user.

**Hybrid** (`recsys/hybrid.py`): min-max normalizes both scores and blends
them with a user-adjustable weight (the frontend's slider).

## Two real bugs this surfaced (and what fixed them)

Both were caught the same way as everything else in this project's family:
by checking whether a *specific, known* case (a Harry Potter fan's
recommendations) produced a sane answer, not just whether the code ran.

1. **Bias terms weren't actually shrinking by sample size.** The first
   training loop regularized item bias with `bi += lr*(grad/count -
   reg*bi)`. Its fixed point is `bi = (grad/count)/reg` - shrinkage that's
   *independent of how many ratings support that estimate*. A niche book
   rated by 80 enthusiasts got exactly as little regularization as one
   rated by 30,000 people, so a handful of five-star outliers could push
   an obscure title's bias above every well-known book's. Once personalization
   is weak (a new user with only a few stated ratings), those inflated
   biases dominated the ranking outright - recommending Calvin & Hobbes
   compilations and a random poetry collection to a stated Harry Potter
   fan. Fixed with the standard ridge/ALS closed form instead,
   `bi = sum_residuals / (count + bias_reg)`, which shrinks low-count
   items toward 0 and barely touches well-supported ones.
2. **`np.add.at` is not a fast way to scatter-accumulate 6M gradients.**
   The obvious numpy translation of "sum each rating's gradient into its
   user's/item's row" is `np.add.at(grad, idx, values)`. It's correct and
   ~15x slower than expressing the same accumulation as a sparse-matrix
   multiply (a precomputed sparse "which user does rating r belong to"
   selector times the dense per-rating contribution) - roughly 35s/epoch
   vs. ~2.5s/epoch on this dataset, because `add.at` forgoes the
   vectorization a real BLAS matmul gets.

## API

| Endpoint | What it does |
|---|---|
| `GET /api/books/search?q=` | Search the full 10k-book catalog by title/author (not just inventory - you can name a favorite the store doesn't carry) |
| `GET /api/inventory/sections` | Section names + counts in the simulated inventory |
| `GET /api/inventory?section=&offset=&limit=` | Browse the inventory, optionally filtered by section |
| `POST /api/recommendations` | Body: `{"liked": [{"book_id", "rating"}, ...], "alpha": 0.5, "top_n": 12}` - returns content/collaborative/hybrid rankings, inventory-only |
| `POST /api/goodreads-import` | Multipart CSV upload - matches a real Goodreads export against the catalog |

## What's real vs. simulated

The ratings, books, and genre tags are real Goodreads data. The
**inventory is not a real store's** - getting one would mean scraping a
specific shop's website, which is fragile, likely against that site's
terms, and out of scope for a portfolio piece. It's constructed
transparently (see `build_inventory.py`) to have a plausible, genre-diverse
mix with real misses, rather than invented to make the recommender look
good.

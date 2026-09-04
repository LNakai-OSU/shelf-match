"""
Latent-factor collaborative filtering, trained with full-batch gradient
descent on observed ratings only.

The common shortcut for a from-scratch SVD recommender is to run
TruncatedSVD directly on the sparse user-item matrix - but that treats
every *unrated* (user, book) pair as if the user had rated it 0, which
biases the factors toward "predicting low ratings for everything" and is
not what real matrix-factorization recommenders (Funk SVD / the Netflix
Prize approach) do. Here, only observed (user, item, rating) triples
contribute to the loss - gradients are scatter-accumulated via sparse-matrix multiplication (a
precomputed sparse "which user/item does rating r belong to" selector,
multiplied against the dense per-rating gradient contributions) so a full
epoch over ~6M ratings is a couple of BLAS-backed sparse-dense matmuls
instead of a 6M-iteration Python loop. An earlier version used
`np.add.at` for the same scatter-accumulation, which is the obvious numpy
way to write it - and was ~15x slower in practice (roughly 35s/epoch vs.
~2.5s/epoch here), because `add.at` forgoes the buffering/vectorization
that a real BLAS matmul gets.

Model: predicted_rating = global_mean + user_bias + item_bias + U[u] . V[i]

A real bug found while sanity-checking this against known books: the first
version regularized bias terms with `bu += lr*(grad/count - reg*bu)`, whose
fixed point is `bu = (grad/count)/reg` - a shrinkage that's *independent of
count*. An item rated by only 80 people gets exactly as little shrinkage as
one rated by 30,000, so a handful of enthusiastic raters on an obscure book
can push its bias arbitrarily high, and once a folded-in user's personal
signal is small (few stated ratings), those inflated biases dominated every
ranking - "The Divan" and Calvin & Hobbes compilations outranking anything
actually similar to a stated Harry Potter/fantasy profile. The fix is the
standard ridge/ALS closed form for a bias term, `bi = sum_residuals /
(count + bias_reg)`, which shrinks low-count items toward 0 and barely
touches high-count ones - implemented in `fit()` below instead of the
naive gradient step. `U`/`V` keep ordinary regularized gradient descent,
which doesn't have this specific pathology (folding a few ratings across a
40-dimensional shared factor space).

Cold start / fold-in: a real visitor to this app isn't one of the 53k users
this model was trained on. `fold_in_user` solves a small ridge-regression
problem - given a few (item, rating) pairs, find the latent vector and bias
that best explain them against the *already-trained* item factors. This is
the standard "folding-in" technique for latent-factor models and is what
makes the collaborative model usable for a brand-new user's stated
favorites instead of only the users it was trained on.
"""

import numpy as np
import scipy.sparse as sp


class MatrixFactorization:
    def __init__(self, n_users, n_items, n_factors=40, reg=0.06, bias_reg=25.0, lr=0.06, n_epochs=80, seed=42):
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.reg = reg
        self.bias_reg = bias_reg
        self.lr = lr
        self.n_epochs = n_epochs
        rng = np.random.default_rng(seed)
        self.U = rng.normal(0, 0.05, size=(n_users, n_factors)).astype(np.float32)
        self.V = rng.normal(0, 0.05, size=(n_items, n_factors)).astype(np.float32)
        self.bu = np.zeros(n_users, dtype=np.float32)
        self.bi = np.zeros(n_items, dtype=np.float32)
        self.global_mean = 0.0

    def fit(self, user_idx, item_idx, ratings, val_user_idx=None, val_item_idx=None, val_ratings=None, verbose=True):
        user_idx = np.asarray(user_idx)
        item_idx = np.asarray(item_idx)
        ratings = np.asarray(ratings, dtype=np.float32)
        self.global_mean = float(ratings.mean())
        n = len(ratings)

        ones = np.ones(n, dtype=np.float32)
        cols = np.arange(n)
        user_sel = sp.csr_matrix((ones, (user_idx, cols)), shape=(self.n_users, n))
        item_sel = sp.csr_matrix((ones, (item_idx, cols)), shape=(self.n_items, n))
        user_counts = np.asarray(user_sel.sum(axis=1)).ravel().clip(min=1)
        item_counts = np.asarray(item_sel.sum(axis=1)).ravel().clip(min=1)

        for epoch in range(self.n_epochs):
            pred = self.global_mean + self.bu[user_idx] + self.bi[item_idx] + np.sum(self.U[user_idx] * self.V[item_idx], axis=1)
            err = (ratings - pred).astype(np.float32)

            grad_bu = user_sel @ err
            grad_bi = item_sel @ err
            grad_U = user_sel @ (err[:, None] * self.V[item_idx])
            grad_V = item_sel @ (err[:, None] * self.U[user_idx])

            # closed-form ridge shrinkage for the bias scalars (see module docstring)
            self.bu = (grad_bu + user_counts * self.bu) / (user_counts + self.bias_reg)
            self.bi = (grad_bi + item_counts * self.bi) / (item_counts + self.bias_reg)
            # ordinary regularized gradient descent for the factor vectors
            self.U += self.lr * (grad_U / user_counts[:, None] - self.reg * self.U)
            self.V += self.lr * (grad_V / item_counts[:, None] - self.reg * self.V)

            if verbose and (epoch % 10 == 0 or epoch == self.n_epochs - 1):
                train_rmse = np.sqrt(np.mean(err ** 2))
                msg = f"epoch {epoch:>3d}  train_rmse={train_rmse:.4f}"
                if val_ratings is not None:
                    msg += f"  val_rmse={self.rmse(val_user_idx, val_item_idx, val_ratings):.4f}"
                print(msg)

    def predict(self, user_idx, item_idx):
        return self.global_mean + self.bu[user_idx] + self.bi[item_idx] + np.sum(self.U[user_idx] * self.V[item_idx], axis=1)

    def rmse(self, user_idx, item_idx, ratings):
        pred = np.clip(self.predict(user_idx, item_idx), 1, 5)
        return float(np.sqrt(np.mean((np.asarray(ratings) - pred) ** 2)))

    def fold_in_user(self, item_idx, ratings, reg=3.0):
        """Solve for a new user's (bias, latent vector) via ridge regression
        against this model's trained item factors, without retraining.

        reg=3.0 was picked empirically: solving for 41 unknowns (1 bias +
        40 factors) from a handful of stated ratings is a badly
        underdetermined regression, and a small reg (e.g. the original
        0.3) lets the fitted user vector's norm blow up to 10-40x a
        typical item vector's norm - fitting the few stated ratings
        essentially perfectly while extrapolating as noise everywhere
        else. reg=3.0 keeps the fitted vector in the same scale as real
        item vectors. Even so, a handful of ratings won't fully overcome
        collaborative filtering's cold-start limits - see README."""
        item_idx = np.asarray(item_idx)
        ratings = np.asarray(ratings, dtype=float)
        V_sub = self.V[item_idx]
        design = np.hstack([np.ones((len(item_idx), 1)), V_sub])
        target = ratings - self.global_mean - self.bi[item_idx]

        k = design.shape[1]
        A = design.T @ design + reg * np.eye(k)
        b = design.T @ target
        solution = np.linalg.solve(A, b)
        user_bias, user_vector = solution[0], solution[1:]
        return user_bias, user_vector

    def score_all_items(self, user_bias, user_vector):
        return self.global_mean + user_bias + self.bi + self.V @ user_vector

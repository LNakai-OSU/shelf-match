import { useState } from "react";
import BookSearch from "./components/BookSearch";
import FavoritesList from "./components/FavoritesList";
import GoodreadsUpload from "./components/GoodreadsUpload";
import RecommendationResults from "./components/RecommendationResults";
import InventoryBrowser from "./components/InventoryBrowser";
import { getRecommendations } from "./api";
import "./App.css";

const TABS = [
  { id: "recommend", label: "Get recommendations" },
  { id: "browse", label: "Browse the shelves" },
];

export default function App() {
  const [tab, setTab] = useState("recommend");
  const [inputMode, setInputMode] = useState("search");
  const [liked, setLiked] = useState([]);
  const [alpha, setAlpha] = useState(0.5);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const addBook = (book) => {
    setLiked((prev) => (prev.some((l) => l.book.book_id === book.book_id) ? prev : [...prev, { book, rating: 4 }]));
  };

  const rateBook = (bookId, rating) => {
    setLiked((prev) => prev.map((l) => (l.book.book_id === bookId ? { ...l, rating } : l)));
  };

  const removeBook = (bookId) => {
    setLiked((prev) => prev.filter((l) => l.book.book_id !== bookId));
  };

  const importGoodreads = (matches) => {
    setLiked(matches.map((m) => ({ book: m, rating: m.rating })));
  };

  const runRecommend = () => {
    if (liked.length === 0) return;
    setLoading(true);
    setError(null);
    getRecommendations(
      liked.map((l) => ({ book_id: l.book.book_id, rating: l.rating })),
      alpha
    )
      .then(setResults)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  };

  return (
    <div className="app">
      <header className="app-header">
        <div>
          <h1>Shelf Match</h1>
          <p className="muted">
            A book recommender constrained to what a small store actually has on the shelf.
            Trained on ~6M real Goodreads ratings across 10,000 books - but every
            recommendation below is filtered down to a simulated indie bookstore's ~1,900-title
            inventory, so the ideal match isn't always available. Compare how content-based,
            collaborative, and hybrid ranking each handle that.
          </p>
        </div>
        <nav className="tabs">
          {TABS.map((t) => (
            <button
              key={t.id}
              className={`tab-btn ${tab === t.id ? "tab-btn-active" : ""}`}
              onClick={() => setTab(t.id)}
            >
              {t.label}
            </button>
          ))}
        </nav>
      </header>

      {tab === "browse" && <InventoryBrowser />}

      {tab === "recommend" && (
        <>
          <section className="panel">
            <div className="input-mode-toggle">
              <button
                className={`chip-btn ${inputMode === "search" ? "chip-btn-active" : ""}`}
                onClick={() => setInputMode("search")}
              >
                Search &amp; rate
              </button>
              <button
                className={`chip-btn ${inputMode === "goodreads" ? "chip-btn-active" : ""}`}
                onClick={() => setInputMode("goodreads")}
              >
                Import Goodreads export
              </button>
            </div>

            {inputMode === "search" ? (
              <BookSearch onAdd={addBook} />
            ) : (
              <GoodreadsUpload onImport={importGoodreads} />
            )}

            <h3 className="favorites-heading">Your favorites ({liked.length})</h3>
            <FavoritesList liked={liked} onRate={rateBook} onRemove={removeBook} />

            <div className="alpha-row">
              <label className="field">
                <span>
                  Hybrid weight - content <span className="mono">{(1 - alpha).toFixed(2)}</span> / collaborative{" "}
                  <span className="mono">{alpha.toFixed(2)}</span>
                </span>
                <input type="range" min={0} max={1} step={0.05} value={alpha} onChange={(e) => setAlpha(Number(e.target.value))} />
              </label>
              <button className="btn btn-primary" onClick={runRecommend} disabled={liked.length === 0 || loading}>
                {loading ? "Ranking the shelves..." : "Get recommendations"}
              </button>
            </div>
            {error && <div className="error-banner">{error}</div>}
          </section>

          <RecommendationResults results={results} />
        </>
      )}

      <footer className="footer mono">
        <span>Data: goodbooks-10k (Zajac) - FastAPI + React</span>
        <span>Content model, matrix-factorization CF, and hybrid re-ranking - see README</span>
      </footer>
    </div>
  );
}

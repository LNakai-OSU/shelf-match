import { useEffect, useState } from "react";
import BookSearch from "./components/BookSearch";
import FavoritesList from "./components/FavoritesList";
import GoodreadsUpload from "./components/GoodreadsUpload";
import RecommendationResults from "./components/RecommendationResults";
import InventoryBrowser from "./components/InventoryBrowser";
import StoreSelector from "./components/StoreSelector";
import DescribeQuery from "./components/DescribeQuery";
import StoreOwnerView from "./components/StoreOwnerView";
import { getRecommendations } from "./api";
import "./App.css";

const TABS = [
  { id: "recommend", label: "Get recommendations" },
  { id: "browse", label: "Browse the shelves" },
];

const INPUT_MODES = [
  { id: "describe", label: "Describe what you want" },
  { id: "search", label: "Search & rate favorites" },
  { id: "goodreads", label: "Import Goodreads export" },
];

function CustomerView() {
  const [tab, setTab] = useState("recommend");
  const [inputMode, setInputMode] = useState("describe");
  const [storeId, setStoreId] = useState("default");
  const [liked, setLiked] = useState([]);
  const [alpha, setAlpha] = useState(0.5);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    setResults(null);
  }, [storeId]);

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
      alpha,
      storeId
    )
      .then(setResults)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  };

  return (
    <>
      <div className="store-context-row">
        <StoreSelector storeId={storeId} onChange={setStoreId} />
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
      </div>

      {tab === "browse" && <InventoryBrowser storeId={storeId} />}

      {tab === "recommend" && (
        <>
          <section className="panel">
            <div className="input-mode-toggle">
              {INPUT_MODES.map((m) => (
                <button
                  key={m.id}
                  className={`chip-btn ${inputMode === m.id ? "chip-btn-active" : ""}`}
                  onClick={() => setInputMode(m.id)}
                >
                  {m.label}
                </button>
              ))}
            </div>

            {inputMode === "describe" && <DescribeQuery storeId={storeId} />}

            {inputMode !== "describe" && (
              <>
                {inputMode === "search" ? (
                  <BookSearch onAdd={addBook} storeId={storeId} />
                ) : (
                  <GoodreadsUpload onImport={importGoodreads} storeId={storeId} />
                )}

                <h3 className="favorites-heading">Your favorites ({liked.length})</h3>
                <FavoritesList liked={liked} onRate={rateBook} onRemove={removeBook} />

                <div className="alpha-row">
                  <label className="field">
                    <span>
                      Hybrid weight - content <span className="mono">{(1 - alpha).toFixed(2)}</span> / collaborative{" "}
                      <span className="mono">{alpha.toFixed(2)}</span>
                    </span>
                    <input
                      type="range"
                      min={0}
                      max={1}
                      step={0.05}
                      value={alpha}
                      onChange={(e) => setAlpha(Number(e.target.value))}
                    />
                  </label>
                  <button className="btn btn-primary" onClick={runRecommend} disabled={liked.length === 0 || loading}>
                    {loading ? "Ranking the shelves..." : "Get recommendations"}
                  </button>
                </div>
                {error && <div className="error-banner">{error}</div>}
              </>
            )}
          </section>

          {inputMode !== "describe" && <RecommendationResults results={results} />}
        </>
      )}
    </>
  );
}

export default function App() {
  const [role, setRole] = useState("customer");

  return (
    <div className="app">
      <header className="app-header">
        <div>
          <h1>Shelf Match</h1>
          <p className="muted">
            A book recommender constrained to what a small store actually has on the shelf.
            Trained on 11.5M real Goodreads ratings across 75,000 books - with real semantic
            search over actual book descriptions, not just genre keywords - but every
            recommendation is filtered down to one store's actual inventory - pick a store as
            a customer, or list your own shelf as a store.
          </p>
        </div>
        <div className="role-toggle">
          <button className={`role-btn ${role === "customer" ? "role-btn-active" : ""}`} onClick={() => setRole("customer")}>
            I'm a customer
          </button>
          <button className={`role-btn ${role === "store" ? "role-btn-active" : ""}`} onClick={() => setRole("store")}>
            I'm a store
          </button>
        </div>
      </header>

      {role === "customer" ? <CustomerView /> : <StoreOwnerView onSwitchToCustomer={() => setRole("customer")} />}

      <footer className="footer mono">
        <span>Data: UCSD Goodreads Book Graph (Wan &amp; McAuley) - FastAPI + React</span>
        <span>Sentence-embedding search, matrix-factorization CF, and hybrid re-ranking - see README</span>
      </footer>
    </div>
  );
}

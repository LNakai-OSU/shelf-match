import { useState } from "react";
import { getRecommendationsByDescription } from "../api";
import BookCard from "./BookCard";

const EXAMPLES = [
  "a cozy mystery with a strong female lead",
  "epic fantasy with dragons",
  "historical romance set in England",
  "a fast-paced sci-fi thriller",
];

export default function DescribeQuery({ storeId }) {
  const [query, setQuery] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const run = (q) => {
    const text = (q ?? query).trim();
    if (!text) return;
    setQuery(text);
    setLoading(true);
    setError(null);
    getRecommendationsByDescription(text, storeId)
      .then(setResult)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  };

  return (
    <div className="describe-query">
      <p className="muted describe-hint">
        Say what you're in the mood for - this matches genre/vibe words against the store's
        shelf directly, no ratings needed. It's keyword matching against a curated genre
        vocabulary, not a language model, so naming a genre or two works better than a full
        sentence about your day.
      </p>
      <div className="describe-input-row">
        <input
          className="search-input"
          type="text"
          placeholder='e.g. "a cozy mystery with a strong female lead"'
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && run()}
        />
        <button className="btn btn-primary" onClick={() => run()} disabled={loading || !query.trim()}>
          {loading ? "Searching..." : "Find books"}
        </button>
      </div>
      <div className="describe-examples">
        {EXAMPLES.map((ex) => (
          <button key={ex} className="chip-btn" onClick={() => run(ex)}>
            {ex}
          </button>
        ))}
      </div>

      {error && <div className="error-banner">{error}</div>}

      {result && result.message && <p className="muted">{result.message}</p>}

      {result && result.genres_detected.length > 0 && (
        <p className="describe-detected muted">
          Detected: {result.sections_detected.join(", ")}
        </p>
      )}

      {result && result.results.length > 0 && (
        <div className="describe-results">
          {result.results.map((book, i) => (
            <BookCard key={book.book_id} book={book} score={book.score} scoreType="percent" rank={i + 1} />
          ))}
        </div>
      )}
    </div>
  );
}

import { useState } from "react";
import { getRecommendationsByDescription } from "../api";
import BookCard from "./BookCard";

const EXAMPLES = [
  "stylistically groundbreaking with an intriguing plot",
  "a cozy mystery with a strong female lead",
  "unreliable narrator, slow-burn dread",
  "a sweeping multigenerational family saga",
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
        Describe the book you want in your own words - real sentence embeddings match the
        meaning of what you write against every book's actual description, not just keywords
        against a genre list. Abstract, specific, or a full sentence all work.
      </p>
      <div className="describe-input-row">
        <input
          className="search-input"
          type="text"
          placeholder='e.g. "stylistically groundbreaking with an intriguing plot"'
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

      {result && result.sections_detected.length > 0 && (
        <p className="describe-detected muted">
          Top matches lean toward: {result.sections_detected.join(", ")}
        </p>
      )}

      {result && result.results.length > 0 && (
        <div className="describe-results">
          {result.results.map((book, i) => (
            <BookCard key={book.book_id} book={book} score={book.score} scoreType="percent" rank={i + 1} showDescription />
          ))}
        </div>
      )}
    </div>
  );
}

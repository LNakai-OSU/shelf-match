import { useEffect, useRef, useState } from "react";
import { searchBooks } from "../api";

export default function BookSearch({ onAdd, storeId = "default" }) {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const debounceRef = useRef(null);

  useEffect(() => {
    clearTimeout(debounceRef.current);
    if (query.trim().length < 2) {
      setResults([]);
      return;
    }
    debounceRef.current = setTimeout(() => {
      setLoading(true);
      searchBooks(query.trim(), storeId)
        .then(setResults)
        .catch(() => setResults([]))
        .finally(() => setLoading(false));
    }, 250);
    return () => clearTimeout(debounceRef.current);
  }, [query, storeId]);

  return (
    <div className="book-search">
      <input
        className="search-input"
        type="text"
        placeholder="Search a favorite book or author..."
        value={query}
        onChange={(e) => setQuery(e.target.value)}
      />
      {loading && <p className="muted search-status">Searching...</p>}
      {results.length > 0 && (
        <ul className="search-results">
          {results.map((b) => (
            <li key={b.book_id} className="search-result-row">
              <img src={b.image_url} alt="" className="search-result-thumb" />
              <div className="search-result-text">
                <span className="search-result-title">{b.title}</span>
                <span className="search-result-author muted">{b.authors}</span>
              </div>
              {b.in_inventory && <span className="chip chip-stock search-result-stock">in stock here</span>}
              <button
                className="btn btn-primary btn-small"
                onClick={() => {
                  onAdd(b);
                  setQuery("");
                  setResults([]);
                }}
              >
                Add
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

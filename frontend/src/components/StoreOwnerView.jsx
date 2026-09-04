import { useRef, useState } from "react";
import { uploadStore } from "../api";

export default function StoreOwnerView({ onSwitchToCustomer }) {
  const [name, setName] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);
  const inputRef = useRef(null);

  const handleFile = (file) => {
    if (!file) return;
    if (!name.trim()) {
      setError("Give your store a name first.");
      return;
    }
    setBusy(true);
    setError(null);
    uploadStore(name.trim(), file)
      .then(setResult)
      .catch((e) => setError(e.message))
      .finally(() => setBusy(false));
  };

  return (
    <section className="panel store-owner-view">
      <h2>List your store's inventory</h2>
      <p className="muted">
        Upload the books you carry and customers can get recommendations from just your
        shelf - not the whole internet's catalog. Two formats work:
      </p>
      <ul className="store-format-list muted">
        <li>
          <strong>A plain text list</strong> - one book per line: <code className="mono">Title</code>,{" "}
          <code className="mono">Title - Author</code>, or <code className="mono">Title, Author</code>.
        </li>
        <li>
          <strong>A CSV</strong> with a <code className="mono">title</code> column and optional{" "}
          <code className="mono">author</code>, <code className="mono">isbn</code>/<code className="mono">isbn13</code>,
          and <code className="mono">quantity</code> columns.
        </li>
      </ul>
      <p className="muted store-catalog-note">
        Only books in the goodbooks-10k catalog (~10,000 well-known titles) can be matched -
        this is a demo built on a fixed public dataset, not a live lookup against every book in
        print.
      </p>

      <label className="field store-name-field">
        <span>Store name</span>
        <input
          type="text"
          className="search-input"
          placeholder="e.g. Corner Books"
          value={name}
          onChange={(e) => setName(e.target.value)}
        />
      </label>

      <input
        ref={inputRef}
        type="file"
        accept=".csv,.txt"
        style={{ display: "none" }}
        onChange={(e) => handleFile(e.target.files[0])}
      />
      <button className="btn btn-primary" onClick={() => inputRef.current.click()} disabled={busy}>
        {busy ? "Matching against catalog..." : "Upload book list"}
      </button>

      {error && <div className="error-banner">{error}</div>}

      {result && (
        <div className="store-result">
          <p>
            Matched <strong>{result.matched_count}</strong> of {result.total_rows} rows against
            the catalog.
          </p>
          <div className="store-code-card">
            <span className="muted">Your store code</span>
            <span className="store-code mono">{result.store_id}</span>
            <span className="muted">"{result.name}" - {result.matched_count} titles</span>
          </div>
          <button className="btn btn-ghost" onClick={onSwitchToCustomer}>
            Switch to customer view to try it
          </button>
          {result.matched_count < result.total_rows && (
            <details className="store-unmatched">
              <summary className="muted">
                {result.total_rows - result.matched_count} row(s) couldn't be matched
              </summary>
              <p className="muted">
                Usually means the title isn't one of the ~10,000 books in this demo's catalog,
                or the title/author spelling didn't line up closely enough to match.
              </p>
            </details>
          )}
        </div>
      )}
    </section>
  );
}

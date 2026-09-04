import { useRef, useState } from "react";
import { uploadGoodreadsExport } from "../api";

export default function GoodreadsUpload({ onImport }) {
  const [status, setStatus] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);
  const inputRef = useRef(null);

  const handleFile = (file) => {
    if (!file) return;
    setBusy(true);
    setError(null);
    uploadGoodreadsExport(file)
      .then((res) => {
        setStatus(res);
        onImport(res.matches);
      })
      .catch((e) => setError(e.message))
      .finally(() => setBusy(false));
  };

  return (
    <div className="goodreads-upload">
      <p className="muted upload-hint">
        Export your library from Goodreads (Settings &rarr; Export Library) and drop the CSV
        here - rated books that match our catalog become your profile automatically.
      </p>
      <input
        ref={inputRef}
        type="file"
        accept=".csv"
        style={{ display: "none" }}
        onChange={(e) => handleFile(e.target.files[0])}
      />
      <button className="btn btn-ghost" onClick={() => inputRef.current.click()} disabled={busy}>
        {busy ? "Matching against catalog..." : "Upload goodreads_library_export.csv"}
      </button>
      {error && <div className="error-banner">{error}</div>}
      {status && (
        <p className="upload-result muted">
          Matched {status.matched_count} of {status.total_rated_rows} rated books against this
          catalog (goodbooks-10k only covers ~10,000 popular titles, so some of your history
          won't be found here).
        </p>
      )}
    </div>
  );
}

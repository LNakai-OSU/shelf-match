const STARS = [1, 2, 3, 4, 5];

export default function BookCard({ book, score, scoreType = "percent", rank, onAdd, rating, onRate, onRemove }) {
  return (
    <div className="book-card">
      {typeof rank === "number" && <span className="book-rank mono">{rank}</span>}
      <img className="book-cover" src={book.image_url} alt={`Cover of ${book.title}`} loading="lazy" />
      <div className="book-info">
        <p className="book-title">{book.title}</p>
        <p className="book-authors">{book.authors}</p>
        <div className="book-meta">
          <span className="chip chip-section">{book.section}</span>
          {book.in_inventory ? (
            <span className="chip chip-stock">{book.stock_quantity} in stock</span>
          ) : (
            <span className="chip chip-out">not carried</span>
          )}
        </div>
        {score != null && scoreType === "rating" && (
          <p className="book-score mono">predicted rating {score.toFixed(2)} / 5</p>
        )}
        {score != null && scoreType === "percent" && (
          <p className="book-score mono">match {(score * 100).toFixed(0)}%</p>
        )}
        {onRate && (
          <div className="star-picker">
            {STARS.map((s) => (
              <button
                key={s}
                className={`star ${rating >= s ? "star-filled" : ""}`}
                onClick={() => onRate(s)}
                aria-label={`Rate ${s} stars`}
              >
                {"★"}
              </button>
            ))}
          </div>
        )}
        {onAdd && (
          <button className="btn btn-ghost btn-small" onClick={onAdd}>
            Add to favorites
          </button>
        )}
        {onRemove && (
          <button className="btn btn-ghost btn-small" onClick={onRemove}>
            Remove
          </button>
        )}
      </div>
    </div>
  );
}

import BookCard from "./BookCard";

export default function FavoritesList({ liked, onRate, onRemove }) {
  if (liked.length === 0) {
    return <p className="muted favorites-empty">No favorites yet - search above or import your Goodreads export.</p>;
  }
  return (
    <div className="favorites-list">
      {liked.map((item) => (
        <BookCard
          key={item.book.book_id}
          book={item.book}
          rating={item.rating}
          onRate={(r) => onRate(item.book.book_id, r)}
          onRemove={() => onRemove(item.book.book_id)}
        />
      ))}
    </div>
  );
}

import BookCard from "./BookCard";

const COLUMNS = [
  {
    key: "content",
    title: "Content-based",
    blurb: "Semantic similarity of each book's actual description to your favorites (sentence embeddings, not genre tags). Works even with no ratings history.",
    scoreType: "percent",
  },
  {
    key: "collaborative",
    title: "Collaborative filtering",
    blurb: "\"Readers like you\" - a latent-factor model trained on 11.5M Goodreads ratings, folded in for your profile. With only a handful of favorites, this tends toward broadly-loved books rather than fine-grained taste - a real cold-start limit, not a bug (see README).",
    scoreType: "rating",
  },
  {
    key: "hybrid",
    title: "Hybrid",
    blurb: "Blends both scores, weighted by the slider above.",
    scoreType: "percent",
  },
];

export default function RecommendationResults({ results }) {
  if (!results) return null;
  return (
    <div className="results-grid">
      {COLUMNS.map((col) => (
        <div className="results-column" key={col.key}>
          <h3>{col.title}</h3>
          <p className="muted results-blurb">{col.blurb}</p>
          <div className="results-list">
            {results[col.key].map((book, i) => (
              <BookCard key={book.book_id} book={book} score={book.score} scoreType={col.scoreType} rank={i + 1} />
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

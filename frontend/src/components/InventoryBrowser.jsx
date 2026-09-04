import { useEffect, useState } from "react";
import { getInventory, getInventorySections } from "../api";
import BookCard from "./BookCard";

export default function InventoryBrowser({ storeId = "default" }) {
  const [sections, setSections] = useState([]);
  const [section, setSection] = useState(null);
  const [data, setData] = useState({ total: 0, items: [] });

  useEffect(() => {
    setSection(null);
    getInventorySections(storeId).then(setSections);
  }, [storeId]);

  useEffect(() => {
    getInventory(storeId, section, 0, 24).then(setData);
  }, [storeId, section]);

  return (
    <div className="inventory-browser">
      <div className="inventory-filters">
        <button
          className={`chip-btn ${!section ? "chip-btn-active" : ""}`}
          onClick={() => setSection(null)}
        >
          All ({sections.reduce((s, x) => s + x.count, 0)})
        </button>
        {sections.map((s) => (
          <button
            key={s.section}
            className={`chip-btn ${section === s.section ? "chip-btn-active" : ""}`}
            onClick={() => setSection(s.section)}
          >
            {s.section} ({s.count})
          </button>
        ))}
      </div>
      <p className="muted inventory-count">Showing {data.items.length} of {data.total} in-stock titles</p>
      <div className="inventory-grid">
        {data.items.map((book) => (
          <BookCard key={book.book_id} book={book} />
        ))}
      </div>
    </div>
  );
}

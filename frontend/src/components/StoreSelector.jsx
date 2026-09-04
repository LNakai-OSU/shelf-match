import { useEffect, useState } from "react";
import { getStores } from "../api";

export default function StoreSelector({ storeId, onChange }) {
  const [stores, setStores] = useState([]);

  const refresh = () => getStores().then(setStores).catch(() => {});

  useEffect(() => {
    refresh();
  }, []);

  return (
    <label className="field store-selector">
      <span>Shopping at</span>
      <select value={storeId} onChange={(e) => onChange(e.target.value)} onFocus={refresh}>
        {stores.map((s) => (
          <option key={s.id} value={s.id}>
            {s.name} ({s.size} titles)
          </option>
        ))}
      </select>
    </label>
  );
}

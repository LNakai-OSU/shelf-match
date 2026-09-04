const BASE = "http://localhost:8020";

async function req(path, options) {
  const res = await fetch(`${BASE}${path}`, options);
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`${path} failed: ${res.status} ${body}`);
  }
  return res.json();
}

export function searchBooks(q, storeId = "default") {
  return req(`/api/books/search?q=${encodeURIComponent(q)}&store_id=${storeId}`);
}

export function getStores() {
  return req("/api/stores");
}

export async function uploadStore(name, file) {
  const form = new FormData();
  form.append("name", name);
  form.append("file", file);
  const res = await fetch(`${BASE}/api/stores`, { method: "POST", body: form });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`upload store failed: ${res.status} ${body}`);
  }
  return res.json();
}

export function getInventorySections(storeId = "default") {
  return req(`/api/stores/${storeId}/sections`);
}

export function getInventory(storeId = "default", section, offset = 0, limit = 40) {
  const params = new URLSearchParams({ offset, limit });
  if (section) params.set("section", section);
  return req(`/api/stores/${storeId}/inventory?${params.toString()}`);
}

export function getRecommendations(liked, alpha, storeId = "default", topN = 12) {
  return req("/api/recommendations", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ liked, alpha, store_id: storeId, top_n: topN }),
  });
}

export function getRecommendationsByDescription(query, storeId = "default", topN = 12) {
  return req("/api/recommendations/by-description", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, store_id: storeId, top_n: topN }),
  });
}

export async function uploadGoodreadsExport(file, storeId = "default") {
  const form = new FormData();
  form.append("file", file);
  form.append("store_id", storeId);
  const res = await fetch(`${BASE}/api/goodreads-import`, { method: "POST", body: form });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`goodreads-import failed: ${res.status} ${body}`);
  }
  return res.json();
}

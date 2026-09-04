const BASE = "http://localhost:8020";

async function req(path, options) {
  const res = await fetch(`${BASE}${path}`, options);
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`${path} failed: ${res.status} ${body}`);
  }
  return res.json();
}

export function searchBooks(q) {
  return req(`/api/books/search?q=${encodeURIComponent(q)}`);
}

export function getInventorySections() {
  return req("/api/inventory/sections");
}

export function getInventory(section, offset = 0, limit = 40) {
  const params = new URLSearchParams({ offset, limit });
  if (section) params.set("section", section);
  return req(`/api/inventory?${params.toString()}`);
}

export function getRecommendations(liked, alpha, topN = 12) {
  return req("/api/recommendations", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ liked, alpha, top_n: topN }),
  });
}

export async function uploadGoodreadsExport(file) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${BASE}/api/goodreads-import`, { method: "POST", body: form });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`goodreads-import failed: ${res.status} ${body}`);
  }
  return res.json();
}

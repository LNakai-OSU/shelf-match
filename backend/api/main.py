import pickle
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from recsys.data_loader import PROCESSED
from recsys.goodreads_import import parse_goodreads_export
from recsys.hybrid import hybrid_rank, rank_by_profile
from recsys.semantic_model import SemanticModel
from recsys.store_import import parse_store_upload
from recsys.stores import create_store, get_store, list_stores, register_store

app = FastAPI(title="Shelf Match - inventory-constrained book recommender")

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_methods=["*"],
    allow_headers=["*"],
)

BOOKS = pd.read_csv(PROCESSED / "books_large.csv").set_index("book_id")
BOOKS["authors"] = BOOKS["authors"].fillna("Unknown")
BOOKS["description"] = BOOKS["description"].fillna("")

with open(PROCESSED / "semantic_model.pkl", "rb") as f:
    SEMANTIC_MODEL: SemanticModel = pickle.load(f)

QUERY_ENCODER = SentenceTransformer(SemanticModel.MODEL_NAME)

with open(PROCESSED / "cf_model_large.pkl", "rb") as f:
    _cf_blob = pickle.load(f)
    CF_MODEL = _cf_blob["model"]

_default_inventory = BOOKS[BOOKS["in_inventory"]]
register_store(
    "default",
    "Simulated Indie Bookstore (demo)",
    _default_inventory.index.to_numpy(),
    stock={int(bid): int(row.stock_quantity) for bid, row in _default_inventory.iterrows()},
)


def _get_store_or_404(store_id):
    store = get_store(store_id)
    if store is None:
        raise HTTPException(404, f"no store with id '{store_id}' - check /api/stores")
    return store


def _book_payload(book_id, store=None, score=None):
    row = BOOKS.loc[book_id]
    in_stock = store is not None and book_id in store["book_ids"]
    payload = {
        "book_id": int(book_id),
        "title": row["title"],
        "authors": row["authors"],
        "description": row["description"],
        "average_rating": float(row["average_rating"]),
        "ratings_count": int(row["ratings_count"]),
        "publication_year": int(row["publication_year"]) if pd.notna(row["publication_year"]) else None,
        "image_url": row["image_url"],
        "section": row["section"] if pd.notna(row["section"]) else "General",
        "primary_genre": row["primary_genre"] if pd.notna(row["primary_genre"]) else "",
        "in_inventory": in_stock,
        "stock_quantity": store["stock"].get(int(book_id), 0) if in_stock else 0,
    }
    if score is not None:
        payload["score"] = round(float(score), 4)
    return payload


@app.get("/api/books/search")
def search_books(q: str, store_id: str = "default", limit: int = 15):
    if len(q) < 2:
        return []
    store = _get_store_or_404(store_id)
    ql = q.lower()
    mask = BOOKS["title"].str.lower().str.contains(ql, na=False) | BOOKS["authors"].str.lower().str.contains(ql, na=False)
    hits = BOOKS[mask].sort_values("ratings_count", ascending=False).head(limit)
    return [_book_payload(bid, store) for bid in hits.index]


@app.get("/api/stores")
def stores():
    return list_stores()


@app.post("/api/stores")
async def upload_store(name: str = Form(...), file: UploadFile = File(...)):
    content = await file.read()
    matches, total_rows = parse_store_upload(content, BOOKS.reset_index())
    if len(matches) == 0:
        raise HTTPException(400, "couldn't match any books from that file against the catalog")
    stock = {int(r.book_id): int(r.quantity) for r in matches.itertuples()}
    store = create_store(name.strip() or "Unnamed store", stock.keys(), stock)
    return {
        "store_id": store["id"],
        "name": store["name"],
        "total_rows": total_rows,
        "matched_count": len(matches),
        "matches": [
            {"book_id": int(r.book_id), "title": r.title, "quantity": r.quantity, "matched_on": r.matched_on}
            for r in matches.itertuples()
        ],
    }


@app.get("/api/stores/{store_id}/sections")
def store_sections(store_id: str):
    store = _get_store_or_404(store_id)
    df = BOOKS.loc[list(store["book_ids"])]
    counts = df["section"].value_counts()
    return [{"section": s, "count": int(c)} for s, c in counts.items()]


@app.get("/api/stores/{store_id}/inventory")
def store_inventory(store_id: str, section: Optional[str] = None, limit: int = 40, offset: int = 0):
    store = _get_store_or_404(store_id)
    df = BOOKS.loc[list(store["book_ids"])]
    if section:
        df = df[df["section"] == section]
    df = df.sort_values("ratings_count", ascending=False)
    page = df.iloc[offset : offset + limit]
    return {"total": len(df), "items": [_book_payload(bid, store) for bid in page.index]}


class LikedBook(BaseModel):
    book_id: int
    rating: float


class RecommendRequest(BaseModel):
    liked: List[LikedBook]
    store_id: str = "default"
    alpha: float = 0.5
    top_n: int = 12


@app.post("/api/recommendations")
def recommendations(req: RecommendRequest):
    if len(req.liked) < 1:
        raise HTTPException(400, "need at least one liked book")
    store = _get_store_or_404(req.store_id)
    liked_ids = [b.book_id for b in req.liked]
    liked_ratings = [b.rating for b in req.liked]
    unknown = [b for b in liked_ids if b not in BOOKS.index]
    if unknown:
        raise HTTPException(400, f"unknown book_id(s): {unknown}")

    ranked = hybrid_rank(
        SEMANTIC_MODEL,
        CF_MODEL,
        list(store["book_ids"]),
        liked_ids,
        liked_ratings,
        alpha=req.alpha,
        top_n=req.top_n,
        exclude_book_ids=liked_ids,
    )
    return {method: [_book_payload(bid, store, score) for bid, score in items] for method, items in ranked.items()}


class DescribeRequest(BaseModel):
    query: str
    store_id: str = "default"
    top_n: int = 12


@app.post("/api/recommendations/by-description")
def recommendations_by_description(req: DescribeRequest):
    store = _get_store_or_404(req.store_id)
    query = req.query.strip()
    if len(query) < 3:
        raise HTTPException(400, "query is too short")

    query_vector = SEMANTIC_MODEL.encode_query(query, QUERY_ENCODER)
    ranked = rank_by_profile(SEMANTIC_MODEL, list(store["book_ids"]), query_vector, top_n=req.top_n)
    results = [_book_payload(bid, store, score) for bid, score in ranked]

    section_counts = {}
    for r in results[:8]:
        section_counts[r["section"]] = section_counts.get(r["section"], 0) + 1
    top_sections = sorted(section_counts, key=section_counts.get, reverse=True)[:3]

    return {
        "sections_detected": top_sections,
        "results": results,
        "message": None,
    }


@app.post("/api/goodreads-import")
async def goodreads_import(file: UploadFile = File(...), store_id: str = Form("default")):
    store = _get_store_or_404(store_id)
    content = await file.read()
    try:
        matches, total_rows = parse_goodreads_export(content, BOOKS.reset_index())
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {
        "total_rated_rows": total_rows,
        "matched_count": len(matches),
        "matches": [
            {**_book_payload(int(r.book_id), store), "rating": r.rating, "matched_on": r.matched_on}
            for r in matches.itertuples()
        ],
    }


@app.get("/api/health")
def health():
    return {"status": "ok", "catalog_size": len(BOOKS), "stores": list_stores()}

import pickle
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from recsys.data_loader import PROCESSED
from recsys.goodreads_import import parse_goodreads_export
from recsys.hybrid import hybrid_rank

app = FastAPI(title="Shelf Match - inventory-constrained book recommender")

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_methods=["*"],
    allow_headers=["*"],
)

BOOKS = pd.read_csv(PROCESSED / "books_enriched.csv").set_index("book_id")
BOOKS["authors"] = BOOKS["authors"].fillna("Unknown")

with open(PROCESSED / "content_model.pkl", "rb") as f:
    CONTENT_MODEL = pickle.load(f)

with open(PROCESSED / "cf_model.pkl", "rb") as f:
    _cf_blob = pickle.load(f)
    CF_MODEL = _cf_blob["model"]

INVENTORY_IDS = BOOKS[BOOKS["in_inventory"]].index.to_numpy()


def _book_payload(book_id, score=None):
    row = BOOKS.loc[book_id]
    payload = {
        "book_id": int(book_id),
        "title": row["title"],
        "authors": row["authors"],
        "average_rating": float(row["average_rating"]),
        "ratings_count": int(row["ratings_count"]),
        "image_url": row["image_url"],
        "section": row["section"] if pd.notna(row["section"]) else "General",
        "genres": row["genres"].split("|") if isinstance(row["genres"], str) and row["genres"] else [],
        "in_inventory": bool(row["in_inventory"]),
        "stock_quantity": int(row["stock_quantity"]),
    }
    if score is not None:
        payload["score"] = round(float(score), 4)
    return payload


@app.get("/api/books/search")
def search_books(q: str, limit: int = 15):
    if len(q) < 2:
        return []
    ql = q.lower()
    mask = BOOKS["title"].str.lower().str.contains(ql, na=False) | BOOKS["authors"].str.lower().str.contains(ql, na=False)
    hits = BOOKS[mask].sort_values("ratings_count", ascending=False).head(limit)
    return [_book_payload(bid) for bid in hits.index]


@app.get("/api/inventory/sections")
def inventory_sections():
    counts = BOOKS[BOOKS["in_inventory"]]["section"].value_counts()
    return [{"section": s, "count": int(c)} for s, c in counts.items()]


@app.get("/api/inventory")
def inventory(section: Optional[str] = None, limit: int = 40, offset: int = 0):
    df = BOOKS[BOOKS["in_inventory"]]
    if section:
        df = df[df["section"] == section]
    df = df.sort_values("ratings_count", ascending=False)
    page = df.iloc[offset : offset + limit]
    return {"total": len(df), "items": [_book_payload(bid) for bid in page.index]}


class LikedBook(BaseModel):
    book_id: int
    rating: float


class RecommendRequest(BaseModel):
    liked: List[LikedBook]
    alpha: float = 0.5
    top_n: int = 12


@app.post("/api/recommendations")
def recommendations(req: RecommendRequest):
    if len(req.liked) < 1:
        raise HTTPException(400, "need at least one liked book")
    liked_ids = [b.book_id for b in req.liked]
    liked_ratings = [b.rating for b in req.liked]
    unknown = [b for b in liked_ids if b not in BOOKS.index]
    if unknown:
        raise HTTPException(400, f"unknown book_id(s): {unknown}")

    ranked = hybrid_rank(
        CONTENT_MODEL,
        CF_MODEL,
        INVENTORY_IDS,
        liked_ids,
        liked_ratings,
        alpha=req.alpha,
        top_n=req.top_n,
        exclude_book_ids=liked_ids,
    )
    return {
        method: [_book_payload(bid, score) for bid, score in items] for method, items in ranked.items()
    }


@app.post("/api/goodreads-import")
async def goodreads_import(file: UploadFile = File(...)):
    content = await file.read()
    try:
        matches, total_rows = parse_goodreads_export(content, BOOKS.reset_index())
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {
        "total_rated_rows": total_rows,
        "matched_count": len(matches),
        "matches": [
            {**_book_payload(int(r.book_id)), "rating": r.rating, "matched_on": r.matched_on}
            for r in matches.itertuples()
        ],
    }


@app.get("/api/health")
def health():
    return {"status": "ok", "catalog_size": len(BOOKS), "inventory_size": len(INVENTORY_IDS)}

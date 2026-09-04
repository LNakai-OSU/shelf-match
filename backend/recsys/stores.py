"""
In-memory store registry. Each "store" is just a named subset of the
goodbooks-10k catalog (a set of book_ids + per-book stock quantities).

The pre-built simulated inventory from `build_inventory.py` is registered
as the "default" store at API startup; any store owner can create their
own by uploading a book list (matched against the catalog via
`store_import.py`), and customers pick which store's shelf to search
against.

This is intentionally in-memory and ephemeral - no database, no auth, no
persistence across a restart. That's the right scope for a portfolio demo
of the recommendation logic; a real multi-tenant version would need actual
accounts and a database table here instead of a module-level dict.
"""

import secrets

_STORES = {}


def register_store(store_id, name, book_ids, stock=None):
    _STORES[store_id] = {
        "id": store_id,
        "name": name,
        "book_ids": {int(b) for b in book_ids},
        "stock": {int(k): int(v) for k, v in (stock or {}).items()},
    }
    return _STORES[store_id]


def create_store(name, book_ids, stock=None):
    store_id = secrets.token_hex(3).upper()
    while store_id in _STORES:
        store_id = secrets.token_hex(3).upper()
    return register_store(store_id, name, book_ids, stock)


def get_store(store_id):
    return _STORES.get(store_id)


def list_stores():
    return [{"id": s["id"], "name": s["name"], "size": len(s["book_ids"])} for s in _STORES.values()]

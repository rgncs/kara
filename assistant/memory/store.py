"""embed(), save_memory(), recall() — the durable-memory store.

Two scopes, each a persistent Chroma DB with an active collection plus a
recently-deleted stockpile:
  - global: cross-project facts (this install's MEMORY_DB_PATH).
  - local:  facts tied to the launch directory (<workspace>/.kara_memory),
            created lazily the first time something is saved locally.
recall() searches both; writes/deletes target or span scopes as needed.
All vectors must come from the same EMBED_MODEL (cosine space).
"""
import logging
import os
import time
import uuid

import chromadb
from openai import OpenAI

import config

log = logging.getLogger("assistant.memory")

_client = OpenAI(base_url=config.OLLAMA_BASE_URL, api_key="ollama")
_chroma = chromadb.PersistentClient(path=config.MEMORY_DB_PATH)
# Global scope (kept as module-level _col/_trash so tests can patch them).
_col = _chroma.get_or_create_collection("memories", metadata={"hnsw:space": "cosine"})
_trash = _chroma.get_or_create_collection("deleted_memories", metadata={"hnsw:space": "cosine"})
_local = None  # lazily-created (col, trash) for the launch-directory scope


def embed(text: str, query: bool = False) -> list[float]:
    """Return an embedding for `text` from EMBED_MODEL via Ollama.

    nomic-embed-text is asymmetric: stored facts use the "search_document:" prefix
    and lookups use "search_query:", which tightens relevant matches.
    """
    prefix = "search_query: " if query else "search_document: "
    return _client.embeddings.create(model=config.EMBED_MODEL, input=prefix + text).data[0].embedding


def _local_cols(create: bool):
    """Return the local scope (col, trash), or None if it doesn't exist and create=False."""
    global _local
    if _local is None:
        path = config.LOCAL_MEMORY_DB_PATH
        if not create and not os.path.isdir(path):
            return None
        client = chromadb.PersistentClient(path=path)
        _local = (client.get_or_create_collection("memories", metadata={"hnsw:space": "cosine"}),
                  client.get_or_create_collection("deleted_memories", metadata={"hnsw:space": "cosine"}))
    return _local


def _scope_cols(scope: str, create: bool = True):
    """(active, trash) collections for a scope ('global' or 'local')."""
    if scope == "local":
        return _local_cols(create=create)
    return (_col, _trash)


def _active_scopes():
    """[(active_col, scope_name)] for every existing scope (global always, local if present)."""
    scopes = [(_col, "global")]
    loc = _local_cols(create=False)
    if loc:
        scopes.append((loc[0], "local"))
    return scopes


def _trash_scopes():
    scopes = [(_trash, "global")]
    loc = _local_cols(create=False)
    if loc:
        scopes.append((loc[1], "local"))
    return scopes


def save_memory(text: str, kind: str = "fact", scope: str = "global") -> str:
    """Embed and store `text` in the given scope. Near-duplicates refresh their
    timestamp instead of adding a copy. Returns "saved" | "duplicate" | "empty"."""
    text = (text or "").strip()
    if not text:
        return "empty"
    col, _ = _scope_cols(scope, create=True)
    emb = embed(text)
    if col.count() > 0:
        res = col.query(query_embeddings=[emb], n_results=1)
        ids, dists = res["ids"][0], res["distances"][0]
        if ids and dists[0] <= config.MEMORY_DUP_DIST:
            col.update(ids=[ids[0]], metadatas=[{"kind": kind, "ts": time.time()}])
            return "duplicate"
    col.add(ids=[str(uuid.uuid4())], embeddings=[emb], documents=[text],
            metadatas=[{"kind": kind, "ts": time.time()}])
    log.debug("save_memory[%s]: stored %r", scope, text)
    return "saved"


def recall(query: str, k: int = config.RECALL_K, max_distance: float = config.RECALL_MAX_DIST) -> list[dict]:
    """Nearest memories to `query` across BOTH scopes, dropping matches weaker than
    `max_distance`, most-recent-first. Each item: {text, ts, distance, scope}."""
    qemb = embed(query, query=True)
    hits = []
    for col, scope in _active_scopes():
        if col.count() == 0:
            continue
        res = col.query(query_embeddings=[qemb], n_results=min(k, col.count()))
        for doc, dist, meta in zip(res["documents"][0], res["distances"][0], res["metadatas"][0]):
            if dist <= max_distance:
                hits.append({"text": doc, "ts": (meta or {}).get("ts", 0.0),
                             "distance": dist, "scope": scope})
    hits.sort(key=lambda h: h["ts"], reverse=True)
    return hits[:k]


def _closest(col, qemb, max_distance: float):
    """(id, document, distance) of the nearest entry within distance, else (None, None, None)."""
    if col.count() == 0:
        return None, None, None
    res = col.query(query_embeddings=[qemb], n_results=1)
    docs, dists, ids = res["documents"][0], res["distances"][0], res["ids"][0]
    if docs and dists[0] <= max_distance:
        return ids[0], docs[0], dists[0]
    return None, None, None


def soft_delete(query: str, max_distance: float = 0.5):
    """Move the active memory closest to `query` (in either scope) to that scope's
    stockpile. Returns the deleted text, or None."""
    qemb = embed(query, query=True)
    best = None  # (dist, scope, id, doc)
    for col, scope in _active_scopes():
        id_, doc, dist = _closest(col, qemb, max_distance)
        if id_ is not None and (best is None or dist < best[0]):
            best = (dist, scope, id_, doc)
    if best is None:
        return None
    _, scope, id_, doc = best
    col, trash = _scope_cols(scope)
    trash.add(ids=[id_], embeddings=[embed(doc)], documents=[doc],
              metadatas=[{"kind": "fact", "deleted_ts": time.time()}])
    col.delete(ids=[id_])
    log.debug("soft-deleted[%s]: %r", scope, doc)
    return doc


def hard_delete(query: str, max_distance: float = 0.5):
    """Permanently remove the single closest active memory (across scopes) for `query`,
    and purge any matching entry from the stockpiles too. Returns the deleted text."""
    qemb = embed(query, query=True)
    best = None  # (dist, col, id, doc) — globally-closest active match
    for col, _scope in _active_scopes():
        id_, doc, dist = _closest(col, qemb, max_distance)
        if id_ is not None and (best is None or dist < best[0]):
            best = (dist, col, id_, doc)
    deleted = None
    if best is not None:
        _, col, id_, doc = best
        col.delete(ids=[id_])
        deleted = doc
    for trash, _scope in _trash_scopes():           # also purge soft-deleted copies
        id_, doc, _ = _closest(trash, qemb, max_distance)
        if id_ is not None:
            trash.delete(ids=[id_])
            deleted = deleted or doc
    if deleted:
        log.debug("hard-deleted: %r", deleted)
    return deleted


def delete_texts(texts: list[str], hard: bool = False) -> int:
    """Soft- (or hard-) delete active memories whose document matches one of `texts`, any scope."""
    n = 0
    for col, scope in _active_scopes():
        got = col.get()
        pairs = [(i, d) for i, d in zip(got["ids"], got["documents"]) if d in texts]
        if not pairs:
            continue
        if not hard:
            trash = _scope_cols(scope)[1]
            for i, d in pairs:
                trash.add(ids=[i], embeddings=[embed(d)], documents=[d],
                          metadatas=[{"kind": "fact", "deleted_ts": time.time()}])
        col.delete(ids=[i for i, _ in pairs])
        n += len(pairs)
    return n


def recall_deleted(query: str, k: int = 3, max_distance: float = 0.5) -> list[dict]:
    """Search the recently-deleted stockpiles (both scopes) to offer a restore."""
    qemb = embed(query, query=True)
    hits = []
    for trash, _scope in _trash_scopes():
        if trash.count() == 0:
            continue
        res = trash.query(query_embeddings=[qemb], n_results=min(k, trash.count()))
        for doc, dist in zip(res["documents"][0], res["distances"][0]):
            if dist <= max_distance:
                hits.append({"text": doc, "distance": dist})
    hits.sort(key=lambda h: h["distance"])
    return hits[:k]


def restore(text: str):
    """Move an exact-text entry from a stockpile back into that scope's active memory."""
    for trash, scope in _trash_scopes():
        g = trash.get()
        ids = [i for i, d in zip(g["ids"], g["documents"]) if d == text]
        if not ids:
            continue
        col = _scope_cols(scope)[0]
        col.add(ids=ids, embeddings=[embed(text)] * len(ids),
                documents=[text] * len(ids), metadatas=[{"kind": "fact", "ts": time.time()}] * len(ids))
        trash.delete(ids=ids)
        log.debug("restored[%s]: %r", scope, text)
        return text
    return None


def deleted_count() -> int:
    return sum(t.count() for t, _ in _trash_scopes())


def purge_old_deleted(max_age_days: float = config.TRASH_TTL_DAYS) -> int:
    """Permanently drop stockpile entries older than `max_age_days`, all scopes."""
    cutoff = time.time() - max_age_days * 86400
    n = 0
    for trash, _scope in _trash_scopes():
        if trash.count() == 0:
            continue
        g = trash.get()
        old = [i for i, m in zip(g["ids"], g["metadatas"]) if (m or {}).get("deleted_ts", 0) < cutoff]
        if old:
            trash.delete(ids=old)
            n += len(old)
    if n:
        log.debug("purged %d expired deleted memories", n)
    return n


def count() -> int:
    """Total active memories across scopes."""
    return sum(c.count() for c, _ in _active_scopes())

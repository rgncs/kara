"""Measure cosine distances for related vs. unrelated memory queries, so the
RECALL_MAX_DIST / MEMORY_DUP_DIST thresholds in config are set from data, not guessed.

Run:  .venv/bin/python scripts/calibrate_memory.py
Uses a throwaway in-memory Chroma collection — does not touch the real store.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "assistant"))

import chromadb  # noqa: E402
from memory.store import embed  # noqa: E402

col = chromadb.Client().get_or_create_collection("calib", metadata={"hnsw:space": "cosine"})

facts = [
    "The user's name is Wontaek",
    "The user prefers dark mode",
    "The user lives in Seattle",
    "The user's favorite language is Python",
]
for i, f in enumerate(facts):
    col.add(ids=[str(i)], embeddings=[embed(f)], documents=[f])  # document side


def nearest(q):
    r = col.query(query_embeddings=[embed(q, query=True)], n_results=1)  # query side
    return r["documents"][0][0], r["distances"][0][0]


print("RELATED queries (should be CLOSE — want RECALL_MAX_DIST above these):")
for q in ["what is my name", "do I like light or dark theme", "where do I live", "what language do I code in"]:
    doc, d = nearest(q)
    print(f"  {d:.3f}  {q!r:42} -> {doc!r}")

print("\nUNRELATED queries (should be FAR — want RECALL_MAX_DIST below these):")
for q in ["what's the weather tomorrow", "how do I center a div", "explain quicksort"]:
    doc, d = nearest(q)
    print(f"  {d:.3f}  {q!r:42} -> {doc!r}")

print("\nDUPLICATE check (should be ~0 — want MEMORY_DUP_DIST above this):")
for q in ["The user's name is Wontaek", "my name is Wontaek", "The user lives in Austin"]:
    doc, d = nearest(q)
    tag = "(identical)" if d < 0.02 else "(paraphrase/diff)"
    print(f"  {d:.3f}  {q!r:42} -> {doc!r} {tag}")

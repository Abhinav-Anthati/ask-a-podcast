"""Hybrid search: BM25 + embedding-based retrieval."""

from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


# Rebuilt periodically since BM25Okapi has no incremental update API
# (see rebuild_index_job in main.py). chunk_ids tracks IDs in the same
# order as the index, since get_scores() returns bare scores only.
bm25_index = None
chunk_ids = []


def tokenize(text):
    """Tokenizes text for BM25, lowercasing and removing stopwords."""
    return [w for w in text.lower().split() if w not in ENGLISH_STOP_WORDS]


def rebuild_bm25_index(cur):
    """Rebuilds the BM25 index from the `chunks` table. Must use the
    same tokenize() as search_bm25, or matching silently breaks.
    """
    global bm25_index, chunk_ids
    cur.execute("SELECT id, text FROM chunks")
    rows = cur.fetchall()
    chunk_ids = [row[0] for row in rows]
    tokenized_docs = [tokenize(row[1]) for row in rows]
    bm25_index = BM25Okapi(tokenized_docs)

def search_bm25(query, top_k=10):
    """Searches the BM25 index for the top_k most relevant chunk IDs to the query."""
    tokenized_query = tokenize(query)
    scores = bm25_index.get_scores(tokenized_query)
    ranked = sorted(zip(chunk_ids, scores), key=lambda x: x[1], reverse=True)
    return [chunk_id for chunk_id, score in ranked[:top_k]]


def reciprocal_rank_fusion(list1, list2, k=60):
    """Merges two ranked ID lists via RRF: each rank contributes
    1/(k+rank), summed across lists. k=60 is RRF's standard default.
    """
    scores = {}
    for pos, chunk_id in enumerate(list1):
        scores[chunk_id] = scores.get(chunk_id, 0) + (1 / (k + pos))
    for pos, chunk_id in enumerate(list2):
        scores[chunk_id] = scores.get(chunk_id, 0) + (1 / (k + pos))

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [chunk_id for chunk_id, score in ranked]
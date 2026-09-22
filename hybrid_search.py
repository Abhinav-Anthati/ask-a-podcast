from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

bm25_index = None
chunk_ids = []


def tokenize(text):
    return [w for w in text.lower().split() if w not in ENGLISH_STOP_WORDS]


def rebuild_bm25_index(cur):
    global bm25_index, chunk_ids
    cur.execute("SELECT id, text FROM chunks")
    rows = cur.fetchall()
    chunk_ids = [row[0] for row in rows]
    tokenized_docs = [tokenize(row[1]) for row in rows]
    bm25_index = BM25Okapi(tokenized_docs)

def search_bm25(query, top_k=10):
    tokenized_query = tokenize(query)
    scores = bm25_index.get_scores(tokenized_query)
    ranked = sorted(zip(chunk_ids, scores), key=lambda x: x[1], reverse=True)
    return [chunk_id for chunk_id, score in ranked[:top_k]]


def reciprocal_rank_fusion(list1, list2, k=60):
    scores = {}
    for pos, chunk_id in enumerate(list1):
        scores[chunk_id] = scores.get(chunk_id, 0) + (1 / (k + pos))
    for pos, chunk_id in enumerate(list2):
        scores[chunk_id] = scores.get(chunk_id, 0) + (1 / (k + pos))

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [chunk_id for chunk_id, score in ranked]
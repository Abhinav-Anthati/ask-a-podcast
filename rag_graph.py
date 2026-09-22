import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import json
from typing import TypedDict
import anthropic
import psycopg2
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from hybrid_search import search_bm25, reciprocal_rank_fusion
from langgraph.graph import StateGraph, END

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def get_connection():
    return psycopg2.connect(
        dbname=os.getenv("DB_NAME"), user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"), host=os.getenv("DB_HOST"), port=os.getenv("DB_PORT")
    )

class RAGState(TypedDict):
    question: str
    podcast_url: str | None
    sub_queries: list
    rows: list
    attempt: int

def decompose(state: RAGState) -> RAGState:
    prompt = f"""Does this question need multiple separate search queries to answer completely? Respond with ONLY a JSON list of strings which are the search queries needed. If one query suffices, return a list with just the original question.

    Question: {state['question']}
    """
    resp = client.messages.create(model="claude-haiku-4-5-20251001", max_tokens=200,
        messages=[{"role": "user", "content": prompt}])
    try:
        state["sub_queries"] = json.loads(resp.content[0].text)
    except json.JSONDecodeError:
        state["sub_queries"] = [state["question"]]
    return state

def retrieve(state: RAGState) -> RAGState:
    conn = get_connection()
    cur = conn.cursor()
    ranked_ids = []
    seen = set()

    for q in state["sub_queries"]:
        embedding = embed_model.encode(q)
        query = """
            SELECT chunks.id FROM chunks
            JOIN episodes ON chunks.episode_id = episodes.id
            JOIN podcasts ON episodes.podcast_url = podcasts.url
            WHERE podcasts.subscribed = TRUE
        """
        params = []
        if state.get("podcast_url"):
            query += " AND episodes.podcast_url = %s"
            params.append(state["podcast_url"])
        query += " ORDER BY embedding <=> %s::vector LIMIT 10"
        params.append(str(embedding.tolist()))
        cur.execute(query, tuple(params))
        vector_ids = [r[0] for r in cur.fetchall()]
        bm25_ids = search_bm25(q, top_k=10)

        for chunk_id in reciprocal_rank_fusion(vector_ids, bm25_ids)[:5]:
            if chunk_id not in seen:
                ranked_ids.append(chunk_id)
                seen.add(chunk_id)

    if not ranked_ids:
        state["rows"] = []
        cur.close(); conn.close()
        return state

    cur.execute("""
        SELECT chunks.id, chunks.text, chunks.start_time, chunks.episode_id, episodes.title, episodes.url
        FROM chunks JOIN episodes ON chunks.episode_id = episodes.id
        WHERE chunks.id IN %s
    """, (tuple(ranked_ids),))
    rows_by_id = {row[0]: row for row in cur.fetchall()}
    state["rows"] = [rows_by_id[cid] for cid in ranked_ids if cid in rows_by_id]
    cur.close(); conn.close()
    return state

def grade(state: RAGState) -> str:
    if not state["rows"] or state["attempt"] >= 2:
        return "done"
    context = "\n".join(r[1] for r in state["rows"])
    prompt = f"""Question: {state['question']}

    Context:
    {context}

    Does this context contain enough information to answer the question? Respond with exactly one word: "yes" or "no".
    """
    resp = client.messages.create(model="claude-haiku-4-5-20251001", max_tokens=10,
        messages=[{"role": "user", "content": prompt}])
    return "done" if "yes" in resp.content[0].text.strip().lower() else "rewrite"

def rewrite(state: RAGState) -> RAGState:
    prompt = f"""Rewrite this question to be clearer/more specific for a search engine. Respond with ONLY the rewritten question.

    Original: {state['question']}
    """
    resp = client.messages.create(model="claude-haiku-4-5-20251001", max_tokens=100,
        messages=[{"role": "user", "content": prompt}])
    state["question"] = resp.content[0].text.strip()
    state["attempt"] += 1
    return state

graph = StateGraph(RAGState)
graph.add_node("decompose", decompose)
graph.add_node("retrieve", retrieve)
graph.add_node("rewrite", rewrite)
graph.set_entry_point("decompose")
graph.add_edge("decompose", "retrieve")
graph.add_conditional_edges("retrieve", grade, {"done": END, "rewrite": "rewrite"})
graph.add_edge("rewrite", "decompose")

app_graph = graph.compile()
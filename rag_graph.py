"""RAG graph for Ask-a-Podcast."""

import os

# Force single-threaded CPU math to avoid a segfault in torch's OpenMP
# thread pool during embedding calls. See numpy<2 pin in requirements.txt.
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
from llm_utils import safe_claude_call

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def get_connection():
    return psycopg2.connect(
        dbname=os.getenv("DB_NAME"), user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"), host=os.getenv("DB_HOST"), port=os.getenv("DB_PORT")
    )

class RAGState(TypedDict):
    """Shared state threaded through the graph.

    question: current question being searched (may be rewritten by
        rewrite(), which re-enters the graph from decompose()).
    podcast_url: optional filter to scope search to one podcast.
    sub_queries: set by decompose(); one or more search queries.
    rows: set by retrieve(); (id, text, start_time, episode_id, title, url)
        tuples for the top fused chunks.
    attempt: retry counter; grade() forces "done" once it hits 2, so a
        stubborn retrieve-grade-rewrite loop can't run forever.
    """
    question: str
    original_question: str
    podcast_url: str | None
    sub_queries: list
    rows: list
    attempt: int

def decompose(state: RAGState) -> RAGState:
    """Asks Claude whether the question needs splitting into multiple
    search queries. Falls back to the original question on a bad
    response (the model doesn't always return valid JSON).

    On a retry (after rewrite), also includes the original question
    among the sub-queries as a hedge against rewrite() shifting meaning.
    """
    if state["attempt"] == 0:
        state["original_question"] = state["question"]

    prompt = f"""Does this question need multiple separate search queries to answer completely? Respond with ONLY a JSON list of strings which are the search queries needed. If one query suffices, return a list with just the original question.

    Question: {state['question']}
    """
    resp = safe_claude_call(client, model="claude-haiku-4-5-20251001", max_tokens=200,
        messages=[{"role": "user", "content": prompt}])
    if resp is None:
        state["sub_queries"] = [state["question"]]
        return state
    try:
        state["sub_queries"] = json.loads(resp.content[0].text)
    except json.JSONDecodeError:
        state["sub_queries"] = [state["question"]]
    return state

def retrieve(state: RAGState) -> RAGState:
    """Runs hybrid search (vector + BM25, fused via RRF) for each
    sub-query, merging results across sub-queries while preserving
    fused rank order.
    """
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
    """Asks Claude whether the retrieved context can answer the question.
    Returns "done" directly (no LLM call) if nothing was retrieved or
    two rewrite attempts have already happened.
    """
    if not state["rows"] or state["attempt"] >= 2:
        return "done"
    context = "\n".join(r[1] for r in state["rows"])
    prompt = f"""Question: {state['question']}

    Context:
    {context}

    Does this context contain enough information to answer the question? Respond with exactly one word: "yes" or "no".
    """
    resp = safe_claude_call(client, model="claude-haiku-4-5-20251001", max_tokens=10,
        messages=[{"role": "user", "content": prompt}])
    if resp is None:
        return "done"
    return "done" if "yes" in resp.content[0].text.strip().lower() else "rewrite"

def rewrite(state: RAGState) -> RAGState:
    """Asks Claude to rewrite the question to be clearer/more specific.

    Known limitation: rewriting can shift meaning (e.g. "about to turn
    60" became "birth year" in testing). decompose() adds the original
    question as a hedge subquery on retry to partially compensate.
    """
    prompt = f"""Rewrite this question to be clearer/more specific for a search engine. Respond with ONLY the rewritten question.

    Original: {state['question']}
    """
    resp = safe_claude_call(client, model="claude-haiku-4-5-20251001", max_tokens=100,
        messages=[{"role": "user", "content": prompt}])
    if resp is None:
        state["attempt"] += 1
        return state
    state["question"] = resp.content[0].text.strip()
    state["attempt"] += 1
    return state


# Graph shape: decompose -> retrieve -> grade (conditional).
# grade routes to END if the context looks sufficient, or to rewrite,
# which loops back to decompose so a reformulated question gets a
# fresh chance at splitting and retrieval.
graph = StateGraph(RAGState)
graph.add_node("decompose", decompose)
graph.add_node("retrieve", retrieve)
graph.add_node("rewrite", rewrite)
graph.set_entry_point("decompose")
graph.add_edge("decompose", "retrieve")
graph.add_conditional_edges("retrieve", grade, {"done": END, "rewrite": "rewrite"})
graph.add_edge("rewrite", "decompose")

app_graph = graph.compile()
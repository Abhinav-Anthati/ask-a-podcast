"""FastAPI app for Ask-a-Podcast, a question-answering system over podcast transcripts."""

import os

# Force single-threaded CPU math to avoid a segfault in torch's OpenMP
# thread pool during embedding calls. See numpy<2 pin in requirements.txt.
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from fastapi import FastAPI, BackgroundTasks, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import psycopg2
import json
from pipeline import sync_and_ingest, backfill_podcast
from apscheduler.schedulers.background import BackgroundScheduler
from dotenv import load_dotenv
from hybrid_search import rebuild_bm25_index
import anthropic
from rag_graph import app_graph
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from starlette.background import BackgroundTask

load_dotenv()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://ask-a-podcast-ui.vercel.app"],
    allow_methods=["*"],
    allow_headers=["*"],
)

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


# --- Request models ---

class Question(BaseModel):
    question: str
    podcast_url: str | None = None
    
class FeedURL(BaseModel):
    feed_url: str


# --- Helpers ---

def get_connection():
    return psycopg2.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT")
    )
    

def stream_answer(prompt):
    """Streams Claude's response to `prompt`, yielding plain text pieces."""
    try:
        with client.messages.stream(
            model="claude-haiku-4-5-20251001", max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        ) as stream:
            for text in stream.text_stream:
                yield text
    except anthropic.APIError as e:
        yield f"\n\n[Error: couldn't generate a response - {e}]"
            
def stream_response(prompt, citations_list):
    """Wraps stream_answer into NDJSON lines: one "citations" event first
    (already known before generation starts), then "token" events per piece.
    """
    yield json.dumps({"type": "citations", "data": citations_list}) + "\n"
    for piece in stream_answer(prompt):
        yield json.dumps({"type": "token", "text": piece}) + "\n"


# --- Scheduled jobs ---

def check_all_podcasts():
    """Daily scheduled job: syncs every subscribed podcast for new episodes."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT url 
        FROM podcasts
        WHERE subscribed = TRUE
        """
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    
    for (url,) in rows:
        for _ in sync_and_ingest(url):
            pass
    
    
def rebuild_index_job():
    """Hourly scheduled job: rebuilds the in-memory BM25 index from
    current chunk text, so newly ingested content becomes searchable
    via keyword search.
    """
    conn = get_connection()
    cur = conn.cursor()
    rebuild_bm25_index(cur)
    cur.close()
    conn.close()
    

# --- Routes ---

@app.post("/ask")
@limiter.limit("10/minute")
def ask(request: Request, payload: Question):
    """Answers a question using the transcripts of all subscribed podcasts, or a specific podcast if provided.

    Returns a streaming response with JSON lines: first a "citations" event with the relevant
    context, followed by "token" events with the answer.
    """
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    result = app_graph.invoke({
        "question": payload.question, "podcast_url": payload.podcast_url,
        "sub_queries": [], "rows": [], "attempt": 0
    })
    rows = result["rows"]
        
    context_string = ""
    citations_list = []
    for chunk_id, text, start_time, episode_id, title, episode_url in rows:
        context_string += f"[ID: {episode_id}, Title: {title}, starts at {start_time:.0f}s]\n{text}\n\n"
        citations_list.append({
            "episode_id": episode_id,
            "title": title,
            "start_time": start_time,
            "text": text,
            "url": episode_url
        })
    
    prompt = f"""Answer the question using only the context below. Be concise.

    Context:
    {context_string}

    Question: {payload.question}

    Answer:"""

    return StreamingResponse(stream_response(prompt, citations_list), media_type="text/plain")
    

@app.post("/podcasts")
@limiter.limit("5/hour")
def subscribe(request: Request, payload: FeedURL):
    """Subscribes to a podcast feed and starts syncing/ingesting new episodes.
    If this is a brand new podcast (no existing episodes), automatically
    backfills its full history in the background after the initial sync,
    rather than leaving that to the slow daily scheduler.
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM episodes WHERE podcast_url = %s", (payload.feed_url,))
    is_new = cur.fetchone()[0] == 0
    cur.close()
    conn.close()

    background = BackgroundTask(backfill_podcast, payload.feed_url) if is_new else None
    return StreamingResponse(sync_and_ingest(payload.feed_url), media_type="text/plain", background=background)


@app.get("/podcasts")
def get_podcast():
    """Returns a list of subscribed podcasts with their title, URL, and episode count."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT podcasts.url, podcasts.title, COUNT(episodes.id) AS episode_count
        FROM podcasts
        LEFT JOIN episodes ON podcasts.url = episodes.podcast_url
        WHERE podcasts.subscribed = TRUE
        GROUP BY podcasts.url, podcasts.title
        """
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    
    podcast_list = []
    for url, title, episode_count in rows:
        podcast_list.append({
            "url": url,
            "title": title,
            "episode_count": episode_count,
        })
    
    return podcast_list


@app.post("/podcasts/backfill")
def backfill(payload: FeedURL, background_tasks: BackgroundTasks):
    """Backfills a podcast feed by repeatedly syncing and ingesting until no new episodes are found."""
    background_tasks.add_task(backfill_podcast, payload.feed_url)
    return {"status": "backfilling"}


@app.delete("/podcasts")
def unsubscribe(payload: FeedURL):
    """Unsubscribes from a podcast feed and stops syncing new episodes."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "UPDATE podcasts SET subscribed = FALSE WHERE url = %s",
        (payload.feed_url,)
    )
    conn.commit()
    cur.close()
    conn.close()
    return {"status": "unsubscribed"}


# --- Scheduler startup (runs at import time) ---

rebuild_index_job()

scheduler = BackgroundScheduler()
scheduler.add_job(rebuild_index_job, "interval", hours=1)
scheduler.add_job(check_all_podcasts, "interval", hours=24)
scheduler.start()
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import psycopg2
import requests
import json
from pipeline import sync_and_ingest, backfill_podcast
from apscheduler.schedulers.background import BackgroundScheduler
from dotenv import load_dotenv
import os
from requests.exceptions import ConnectionError
from hybrid_search import rebuild_bm25_index, search_bm25, reciprocal_rank_fusion

load_dotenv()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = SentenceTransformer("all-MiniLM-L6-v2")


class Question(BaseModel):
    question: str
    podcast_url: str | None = None
    
class FeedURL(BaseModel):
    feed_url: str


def get_connection():
    return psycopg2.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT")
    )
    

def stream_answer(prompt):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama3.2:1b", "prompt": prompt, "stream": True},
            stream=True
        )
        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                yield data["response"]
    except ConnectionError:
        yield "Ollama is down"
            
def stream_response(prompt, citations_list):
    yield json.dumps({"type": "citations", "data": citations_list}) + "\n"
    for piece in stream_answer(prompt):
        yield json.dumps({"type": "token", "text": piece}) + "\n"


def check_all_podcasts():
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
    conn = get_connection()
    cur = conn.cursor()
    rebuild_bm25_index(cur)
    cur.close()
    conn.close()
    

@app.post("/ask")
def ask(payload: Question):
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    question_embedding = model.encode(payload.question)
    conn = get_connection()
    cur = conn.cursor()
    
    query = """
        SELECT chunks.id
        FROM chunks
        JOIN episodes ON chunks.episode_id = episodes.id
        JOIN podcasts ON episodes.podcast_url = podcasts.url
        WHERE podcasts.subscribed = TRUE
    """
    params = []

    if payload.podcast_url:
        query += " AND episodes.podcast_url = %s"
        params.append(payload.podcast_url)

    query += " ORDER BY embedding <=> %s::vector LIMIT 10"
    params.append(str(question_embedding.tolist()))

    cur.execute(query, tuple(params))
    
    vector_ids = [row[0] for row in cur.fetchall()]
    bm25_ids = search_bm25(payload.question, top_k=10)
    
    fused_ids = reciprocal_rank_fusion(vector_ids, bm25_ids)[:5]
    
    if not fused_ids:
        return StreamingResponse(
            stream_response("No relevant context found.", []),
            media_type="text/plain"
        )
    
    cur.execute(
        """
        SELECT chunks.id, chunks.text, chunks.start_time, chunks.episode_id, episodes.title, episodes.url
        FROM chunks
        JOIN episodes ON chunks.episode_id = episodes.id
        WHERE chunks.id IN %s
        """,
        (tuple(fused_ids),)
    )
    rows_by_id = {row[0]: row for row in cur.fetchall()}
    rows = [rows_by_id[chunk_id] for chunk_id in fused_ids if chunk_id in rows_by_id]
    
    cur.close()
    conn.close()
    
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
def subscribe(payload: FeedURL):
    return StreamingResponse(sync_and_ingest(payload.feed_url),media_type="text/plain")


@app.get("/podcasts")
def get_podcast():
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
    background_tasks.add_task(backfill_podcast, payload.feed_url)
    return {"status": "backfilling"}


@app.delete("/podcasts")
def unsubscribe(payload: FeedURL):
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


rebuild_index_job()

scheduler = BackgroundScheduler()
scheduler.add_job(rebuild_index_job, "interval", hours=1)
scheduler.add_job(check_all_podcasts, "interval", hours=24)
scheduler.start()
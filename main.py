from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import psycopg2
import json
from pipeline import sync_and_ingest, backfill_podcast
from apscheduler.schedulers.background import BackgroundScheduler
from dotenv import load_dotenv
import os
from hybrid_search import rebuild_bm25_index, search_bm25, reciprocal_rank_fusion
import anthropic
from rag_graph import app_graph

load_dotenv()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


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
    with client.messages.stream(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    ) as stream:
        for text in stream.text_stream:
            yield text
            
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
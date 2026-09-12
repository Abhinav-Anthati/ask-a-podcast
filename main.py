from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import psycopg2
import requests
import json

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


def get_connection():
    return psycopg2.connect(
        dbname="podcasts", user="postgres", password="postgres",
        host="localhost", port=5432
    )
    

def stream_answer(prompt):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "llama3.2:1b", "prompt": prompt, "stream": True},
        stream=True
    )
    for line in response.iter_lines():
        if line:
            data = json.loads(line)
            yield data["response"]
            
            
def stream_response(prompt, citations_list):
    yield json.dumps({"type": "citations", "data": citations_list}) + "\n"
    for piece in stream_answer(prompt):
        yield json.dumps({"type": "token", "text": piece}) + "\n"

@app.post("/ask")
def ask(payload: Question):
    question_embedding = model.encode(payload.question)
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT chunks.text, chunks.start_time, chunks.episode_id, episodes.title
        FROM chunks JOIN episodes
        ON chunks.episode_id = episodes.episode_id
        ORDER BY embedding <=> %s::vector
        LIMIT 5
        """,
        (str(question_embedding.tolist()),)
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    
    context_string = ""
    citations_list = []
    for text, start_time, episode_id, title in rows:
        context_string += f"[ID: {episode_id}, Title: {title}, starts at {start_time:.0f}s]\n{text}\n\n"
        citations_list.append({
            "episode_id": episode_id,
            "title": title,
            "start_time": start_time,
            "text": text,
            "url": f"https://www.youtube.com/watch?v={episode_id}&t={int(start_time)}s"
        })
    
    prompt = f"""Answer the question using only the context below. Be concise.

    Context:
    {context_string}

    Question: {payload.question}

    Answer:"""

    return StreamingResponse(stream_response(prompt, citations_list), media_type="text/plain")
    
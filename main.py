from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import psycopg2
import requests

app = FastAPI()
model = SentenceTransformer("all-MiniLM-L6-v2")


class Question(BaseModel):
    question: str


def get_connection():
    return psycopg2.connect(
        dbname="podcasts", user="postgres", password="postgres",
        host="localhost", port=5432
    )
    

@app.post("/ask")
def ask(payload: Question):
    question_embedding = model.encode(payload.question)
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT text, start_time, episode_id
        FROM chunks
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
    for text, start_time, episode_id in rows:
        context_string += f"[{episode_id}, starts at {start_time:.0f}s]\n{text}\n\n"
        citations_list.append({
            "episode_id": episode_id,
            "start_time": start_time,
            "text": text
        })
    
    prompt = f"""Answer the question using only the context below. Be concise.

    Context:
    {context_string}

    Question: {payload.question}

    Answer:"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3.2:1b",
            "prompt": prompt,
            "stream": False
        }
    )
    answer = response.json()["response"]

    return {"answer": answer, "citations": citations_list}
    
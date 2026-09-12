import os
import json
import psycopg2
from sentence_transformers import SentenceTransformer
import requests


def get_episode_title(episode_id):
    url = f"https://www.youtube.com/watch?v={episode_id}"
    response = requests.get(
        "https://www.youtube.com/oembed",
        params={"url": url, "format": "json"}
    )
    return response.json().get("title", episode_id)


def count_tokens(text, tokenizer):
    return len(tokenizer.encode(text))


def chunk_transcript(transcript, episode_id, tokenizer, target_tokens=200, overlap_tokens=40):
    chunks = []
    current_segments = []
    current_token_count = 0
    segments_since_flush = 0

    for segment in transcript["segments"]:
        current_segments.append(segment)
        current_token_count += count_tokens(segment["text"], tokenizer)
        segments_since_flush += 1

        if current_token_count >= target_tokens:
            chunks.append({
                "episode_id": episode_id,
                "text": " ".join(s["text"].strip() for s in current_segments),
                "start_time": current_segments[0]["start"],
                "end_time": current_segments[-1]["end"],
            })

            overlap_segments = []
            overlap_token_count = 0
            for seg in reversed(current_segments):
                overlap_segments.insert(0, seg)
                overlap_token_count += count_tokens(seg["text"], tokenizer)
                if overlap_token_count >= overlap_tokens:
                    break

            current_segments = overlap_segments
            current_token_count = overlap_token_count
            segments_since_flush = 0

    if segments_since_flush > 0:
        chunks.append({
            "episode_id": episode_id,
            "text": " ".join(s["text"].strip() for s in current_segments),
            "start_time": current_segments[0]["start"],
            "end_time": current_segments[-1]["end"],
        })

    return chunks


def ingest_all():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    tokenizer = model.tokenizer

    conn = psycopg2.connect(
        dbname="podcasts", user="postgres", password="postgres",
        host="localhost", port=5432
    )
    cur = conn.cursor()
    
    cur.execute("SELECT episode_id FROM episodes")
    existing_ids = {row[0] for row in cur.fetchall()}

    for filename in os.listdir("transcripts"):
        if not filename.endswith(".json"):
            continue
        
        episode_id = filename.replace(".json", "")
        
        if episode_id in existing_ids:
            continue

        with open(f"transcripts/{filename}") as f:
            transcript = json.load(f)
        
        cur.execute("""
            INSERT INTO episodes (episode_id, title) 
            VALUES (%s, %s) ON CONFLICT (episode_id) DO NOTHING
        """, (episode_id, get_episode_title(episode_id)))
        
        chunks = chunk_transcript(transcript, episode_id, tokenizer)

        texts = [c["text"] for c in chunks]
        embeddings = model.encode(texts)

        for chunk, embedding in zip(chunks, embeddings):
            cur.execute(
                """
                INSERT INTO chunks (embedding, text, start_time, end_time, episode_id)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (str(embedding.tolist()), chunk["text"], chunk["start_time"], chunk["end_time"], chunk["episode_id"])
            )

        print(f"{episode_id}: inserted {len(chunks)} chunks")

    conn.commit()
    cur.close()
    conn.close()
    print("Ingestion complete.")


if __name__ == "__main__":
    ingest_all()
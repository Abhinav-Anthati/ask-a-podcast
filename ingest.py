import os
import json
import psycopg2
from sentence_transformers import SentenceTransformer


def chunk_transcript(transcript, episode_id, target_words=300, overlap_words=50):
    chunks = []
    current_segments = []
    current_word_count = 0
    segments_since_flush = 0
    
    for segment in transcript["segments"]:
        current_segments.append(segment)
        current_word_count += len(segment["words"])
        segments_since_flush += 1
        
        if current_word_count >= target_words:
            chunks.append({
                "episode_id": episode_id,
                "text": " ".join(s["text"].strip() for s in current_segments),
                "start_time": current_segments[0]["start"],
                "end_time": current_segments[-1]["end"],
            })
        
            overlap_segments = []
            overlap_word_count = 0
            for seg in reversed(current_segments):
                overlap_segments.insert(0, seg)
                overlap_word_count += len(seg["words"])
                if overlap_word_count >= overlap_words:
                    break
                
            current_segments = overlap_segments
            current_word_count = overlap_word_count
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
    
    conn = psycopg2.connect(
        dbname="podcasts", user="postgres", password="postgres",
        host="localhost", port=5432
    )
    cur = conn.cursor()
    
    for filename in os.listdir("transcripts"):
        if not filename.endswith(".json"):
            continue
        
        with open(f"transcripts/{filename}") as f:
            transcript = json.load(f)
        
        episode_id = filename.replace(".json", "")
        chunks = chunk_transcript(transcript, episode_id)

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
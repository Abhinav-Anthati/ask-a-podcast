import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from download import sync_podcast
from transcribe import transcribe_episode
from ingest import ingest_episode
from faster_whisper import WhisperModel
from sentence_transformers import SentenceTransformer
import psycopg2
import json
from tracing import tracer


def sync_and_ingest(feed_url):
    whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    conn = psycopg2.connect(
        dbname="podcasts", user="postgres", password="postgres",
        host="localhost", port=5432
    )
    cur = conn.cursor()

    episodes = None
    for update in sync_podcast(feed_url, cur):
        if update["status"] == "episodes_ready":
            episodes = update["episodes"]
        else:
            yield json.dumps(update) + "\n"
    
    conn.commit()

    yield json.dumps({"status": "downloaded", "count": len(episodes)}) + "\n"

    for episode in episodes:
        try:
            with tracer.start_as_current_span("transcribe_episode"):
                result = transcribe_episode(episode, whisper_model)

            out_name = f"transcripts/{episode['guid']}.json"
            with open(out_name, "w") as f:
                json.dump(result, f, indent=2)
            yield json.dumps({"status": "transcribed", "episode": episode["guid"]}) + "\n"
            
            with tracer.start_as_current_span("ingest_episode"):
                ingest_episode(episode, embed_model, cur)
            conn.commit()
            yield json.dumps({"status": "ingested", "episode": episode["guid"]}) + "\n"

        except Exception as e:
            yield json.dumps({"status": "failed", "episode": episode["guid"], "error": str(e)}) + "\n"
            conn.rollback()
            continue

    cur.close()
    conn.close()

    yield json.dumps({"status": "done"}) + "\n"
    
    
def backfill_podcast(feed_url):
    count = -1
    while count != 0:
        for update in sync_and_ingest(feed_url):
            data = json.loads(update)
            if data["status"] == "downloaded":
                count = data["count"]
        print(f"Backfill round complete for {feed_url}: {count} new episodes")
            

if __name__ == "__main__":
    for update in sync_and_ingest("https://feeds.npr.org/500005/podcast.xml"):
        print(update)
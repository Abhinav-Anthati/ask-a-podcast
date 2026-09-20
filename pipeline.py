from download import sync_podcast
from transcribe import transcribe_in_subprocess
from ingest import ingest_episode
from sentence_transformers import SentenceTransformer
import psycopg2
import json
from tracing import tracer
from concurrent.futures import ProcessPoolExecutor
from dotenv import load_dotenv
import os

load_dotenv()


def sync_and_ingest(feed_url):
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    conn = psycopg2.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT")
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

    with ProcessPoolExecutor() as executor:
        for episode in episodes:
            try:
                with tracer.start_as_current_span("transcribe_episode"):
                    future = executor.submit(transcribe_in_subprocess, episode)
                    result = future.result()

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
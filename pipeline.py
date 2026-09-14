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


def sync_and_ingest(feed_url):
    whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    conn = psycopg2.connect(
        dbname="podcasts", user="postgres", password="postgres",
        host="localhost", port=5432
    )
    cur = conn.cursor()

    episodes = sync_podcast(feed_url, cur)
    conn.commit()

    for episode in episodes:
        try:
            result = transcribe_episode(episode, whisper_model)

            out_name = f"transcripts/{episode['guid']}.json"
            with open(out_name, "w") as f:
                json.dump(result, f, indent=2)
            print(f"Transcribed {out_name} - {len(result['segments'])} segments, {result['duration']:.1f}s")
            
            ingest_episode(episode, embed_model, cur)
            conn.commit()

        except Exception as e:
            print(f"Failed to process {episode['guid']}: {e}")
            conn.rollback()
            continue

    cur.close()
    conn.close()

if __name__ == "__main__":
    sync_and_ingest("https://feeds.npr.org/500005/podcast.xml")
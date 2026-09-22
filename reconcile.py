import os
import glob
from dotenv import load_dotenv
import psycopg2

load_dotenv()

def get_connection():
    return psycopg2.connect(
        dbname=os.getenv("DB_NAME"), user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"), host=os.getenv("DB_HOST"), port=os.getenv("DB_PORT")
    )

conn = get_connection()
cur = conn.cursor()
cur.execute("SELECT id FROM episodes")
db_ids = {row[0] for row in cur.fetchall()}
cur.close()
conn.close()

transcript_files = glob.glob("transcripts/*.json")
orphaned = []

for path in transcript_files:
    guid = os.path.basename(path).replace(".json", "")
    if guid not in db_ids:
        orphaned.append((guid, path))

print(f"{len(transcript_files)} transcripts on disk, {len(db_ids)} episodes in DB, {len(orphaned)} orphaned.")

for guid, path in orphaned:
    os.remove(path)
    mp3_path = f"episodes/{guid}.mp3"
    if os.path.exists(mp3_path):
        os.remove(mp3_path)
    print(f"Deleted orphan: {guid}")
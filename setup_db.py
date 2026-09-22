"""Sets up the database tables for Ask-a-Podcast.
Run this once before starting the app.
"""

import psycopg2
from dotenv import load_dotenv
import os

load_dotenv()

conn = psycopg2.connect(
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT")
)

cur = conn.cursor()

cur.execute("""
    CREATE TABLE IF NOT EXISTS podcasts (
        url TEXT PRIMARY KEY, 
        title TEXT
    )
""")

cur.execute("""
    ALTER TABLE podcasts ADD COLUMN IF NOT EXISTS subscribed BOOLEAN DEFAULT TRUE
""")

cur.execute("""
    CREATE TABLE IF NOT EXISTS episodes (
        id TEXT PRIMARY KEY,
        title TEXT,
        url TEXT,
        podcast_url TEXT REFERENCES podcasts(url)
    )
""")

cur.execute("""
    CREATE TABLE IF NOT EXISTS chunks (
        id SERIAL PRIMARY KEY,
        embedding VECTOR(384),
        text TEXT,
        start_time FLOAT,
        end_time FLOAT,
        episode_id TEXT REFERENCES episodes(id) 
    )        
""")

conn.commit()
cur.close()
conn.close()
print("Tables ready.")
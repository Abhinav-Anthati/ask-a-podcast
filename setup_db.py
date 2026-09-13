import psycopg2

conn = psycopg2.connect(
    dbname="podcasts", user="postgres", password="postgres",
    host="localhost", port=5432
)

cur = conn.cursor()

cur.execute("""
    CREATE TABLE IF NOT EXISTS podcasts (
        url TEXT PRIMARY KEY, 
        title TEXT
    )
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
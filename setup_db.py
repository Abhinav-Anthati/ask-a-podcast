import psycopg2

conn = psycopg2.connect(
    dbname="podcasts", user="postgres", password="postgres",
    host="localhost", port=5432
)

cur = conn.cursor()

cur.execute("""
    CREATE TABLE IF NOT EXISTS chunks (
        id SERIAL PRIMARY KEY,
        embedding VECTOR(384),
        text TEXT,
        start_time FLOAT,
        end_time FLOAT,
        episode_id TEXT
    )        
""")

conn.commit()
cur.close()
conn.close()
print("Table ready.")
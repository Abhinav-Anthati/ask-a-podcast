import feedparser
import requests
from requests import RequestException
import psycopg2
from tracing import tracer

def get_episodes(feed):
    episodes = []
    for episode in feed.entries:
        episodes.append({
            "guid": episode.id,
            "title": episode.title,
            "published": episode.published,
            "href": episode.enclosures[0]["href"],
        })
    return episodes


def download_episode(audio_url: str, output_path: str):
    print(f"Downloading {output_path}")
    response = requests.get(audio_url, stream=True)
    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            

def sync_podcast(feed_url: str, cur):
    feed = feedparser.parse(feed_url)
    
    cur.execute(
        """
        INSERT INTO podcasts (url, title) 
        VALUES (%s, %s)
        ON CONFLICT (url) DO NOTHING
        """,
        (feed_url, feed.feed.title)
    )
    
    episodes = get_episodes(feed)
    print(f"Found {len(episodes)} episodes")
    
    cur.execute("SELECT id FROM episodes")
    existing_episodes = {row[0] for row in cur.fetchall()}
    
    MAX_PER_SYNC = 3

    new_episodes = []
    for episode in episodes:
        if len(new_episodes) >= MAX_PER_SYNC:
            break

        if episode["guid"] in existing_episodes:
            continue
        
        local_path = f"episodes/{episode['guid']}.mp3"
        
        yield {"status": "downloading", "episode": episode["guid"]}
        try:
            with tracer.start_as_current_span("download_episode"):
                download_episode(episode["href"], local_path)
        except RequestException as e:
            print(f"Skipping {episode['guid']} — download failed: {e}")
            continue
        
        cur.execute(
            """
            INSERT INTO episodes (id, title, url, podcast_url)
            VALUES (%s, %s, %s, %s)
            """,
            (episode["guid"], episode["title"], episode["href"], feed_url)
        )
        
        episode["path"] = local_path
        new_episodes.append(episode)

    print(f"Downloaded {len(new_episodes)} new episodes.")
    yield {"status": "episodes_ready", "episodes": new_episodes}
     
if __name__ == "__main__":
    sync_podcast("https://feeds.npr.org/500005/podcast.xml")
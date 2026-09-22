"""Download episodes from podcast that are not already in database."""

import feedparser
import requests
from requests import RequestException
from tracing import tracer

def get_episodes(feed):
    """Extracts episode metadata from a parsed feedparser feed object.

    Returns:
        list[dict]: each dict has keys 'guid', 'title', 'published', 'href'
            (the audio URL), in feed order (newest first).
    """
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
    """Downloads single episode at audio_url to output_path."""
    response = requests.get(audio_url, stream=True)
    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            

def sync_podcast(feed_url: str, cur):
    """Downloads new episodes from a podcast feed, capped at
    MAX_PER_SYNC per call. Ensures the podcast's row exists in
    `podcasts`. Does not insert into `episodes`; that happens in
    pipeline.py alongside each episode's chunks.

    Yields:
        dict: progress events with a "status" key ("downloading",
            "failed", or "episodes_ready" as the final event,
            carrying the downloaded episode dicts).
    """
    feed = feedparser.parse(feed_url)
    if not feed.entries:
        yield {"status": "failed", "error": "Couldn't find any episodes in this feed"}
        return
    
    cur.execute(
        """
        INSERT INTO podcasts (url, title) 
        VALUES (%s, %s)
        ON CONFLICT (url) DO UPDATE SET subscribed = TRUE
        """,
        (feed_url, feed.feed.title)
    )
    
    episodes = get_episodes(feed)
    print(f"Found {len(episodes)} episodes in feed {feed_url}.")
    
    cur.execute("SELECT id FROM episodes")
    existing_episodes = {row[0] for row in cur.fetchall()}
    
    # Caps new downloads per sync call so a large first time backfill
    # doesn't overwhelm a single sync. Remaining episodes get picked up
    # gradually by the daily scheduled sync.
    MAX_PER_SYNC = 3

    new_episodes = []
    for episode in episodes:
        if len(new_episodes) >= MAX_PER_SYNC:
            break
        if episode["guid"] in existing_episodes:
            continue

        local_path = f"episodes/{episode['guid']}.mp3"
        yield {"status": "downloading", "episode": episode["title"]}
        try:
            with tracer.start_as_current_span("download_episode"):
                download_episode(episode["href"], local_path)
            print(f"Downloaded {episode['title']} to {local_path}.")
        except RequestException as e:
            print(f"Skipping {episode['title']} - download failed: {e}")
            continue

        episode["path"] = local_path
        new_episodes.append(episode)

    yield {"status": "episodes_ready", "episodes": new_episodes}
     
if __name__ == "__main__":
    sync_podcast("https://feeds.npr.org/500005/podcast.xml")
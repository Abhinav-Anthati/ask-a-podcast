import feedparser, requests

FEED_URL = "https://lexfridman.com/feed/podcast/"

def download_podcasts(feed_url: str):
    """Download three latest podcasts."""
    feed = feedparser.parse(feed_url)

    print(feed.channel.title)
    print(feed.channel.description)

    for entry in feed.entries[:3]:
        url = entry["links"][-1]["href"]
        filename = url.split("/")[-1]
        with open(filename, "wb") as f:
            f.write(requests.get(url).content)
        print(f"Downloaded: {filename}")


"""Step 2: discover places that OSM misses, from YouTube food vlogs and web articles.

Collects videos (with view counts) and articles about Blok M food, then asks an LLM
to extract the specific places each one mentions, with a supporting quote."""
import os
from concurrent.futures import ThreadPoolExecutor

from common import RAW, cached, http, llm_json, write_json

YT_QUERIES = [
    ("kuliner blok m", "relevance"),
    ("kuliner blok m", "viewCount"),
    ("makanan viral blok m", "relevance"),
    ("cafe blok m", "relevance"),
    ("tempat nongkrong blok m", "relevance"),
    ("tempat nugas blok m", "relevance"),
    ("little tokyo blok m", "relevance"),
    ("kuliner melawai", "relevance"),
    ("m bloc space kuliner", "relevance"),
    ("blok m square basement kuliner", "relevance"),
    ("jajanan blok m", "relevance"),
    ("ramen blok m", "relevance"),
    ("kopi blok m", "relevance"),
]
TAVILY_QUERIES = [
    "tempat makan viral di Blok M",
    "rekomendasi kuliner Blok M terbaru 2026",
    "cafe aesthetic di Blok M",
    "cafe buat nugas di Blok M wifi colokan",
    "restoran halal di Blok M",
    "kuliner Little Tokyo Melawai",
    "tenant kuliner M Bloc Space",
    "kuliner basement Blok M Square",
    "makanan viral TikTok Blok M",
]
AREA_KEYWORDS = ["blok m", "blokm", "melawai", "little tokyo", "m bloc", "mbloc",
                 "panglima polim", "bulungan", "kebayoran baru"]
PUBLISHED_AFTER = "2024-01-01T00:00:00Z"

EXTRACT_SYSTEM = """You extract food and drink businesses from Indonesian or English text about Blok M, Jakarta.
Return JSON: {"places": [{"name": str, "area_hint": str, "kind": str, "quote": str}]}
Rules:
- Only specific, named businesses (restaurants, cafes, stalls, bakeries, dessert shops, bars).
- Do NOT include malls, buildings, streets or areas as places (e.g. "Blok M Square", "M Bloc Space", "Little Tokyo").
  Put those in area_hint instead.
- area_hint: the most specific location mentioned for that place (building, floor, food court, street), or "" if none.
- kind: one of cafe, restaurant, street_food, dessert, bakery, bar, other.
- quote: a short verbatim snippet (max 25 words) from the text that mentions the place.
- Skip places the text says are outside the Blok M area.
- If no places are mentioned, return {"places": []}."""


def yt_get(endpoint: str, **params) -> dict:
    params["key"] = os.environ["YOUTUBE_API_KEY"]
    return http("GET", f"https://www.googleapis.com/youtube/v3/{endpoint}", params=params).json()


def fetch_videos() -> list[dict]:
    ids = {}
    for q, order in YT_QUERIES:
        res = cached("youtube_search", f"{q}|{order}|{PUBLISHED_AFTER}", lambda: yt_get(
            "search", part="snippet", q=q, type="video", maxResults=50, order=order,
            regionCode="ID", relevanceLanguage="id", publishedAfter=PUBLISHED_AFTER))
        for item in res.get("items", []):
            ids.setdefault(item["id"]["videoId"], q)

    videos = []
    id_list = list(ids)
    for i in range(0, len(id_list), 50):
        batch = id_list[i:i + 50]
        res = cached("youtube_videos", ",".join(batch), lambda: yt_get(
            "videos", part="snippet,statistics", id=",".join(batch), maxResults=50))
        for v in res.get("items", []):
            s, st = v["snippet"], v.get("statistics", {})
            videos.append({
                "video_id": v["id"],
                "url": f"https://www.youtube.com/watch?v={v['id']}",
                "title": s["title"],
                "description": s.get("description", ""),
                "channel": s.get("channelTitle", ""),
                "published_at": s["publishedAt"],
                "views": int(st.get("viewCount", 0)),
                "likes": int(st.get("likeCount", 0)),
                "comments": int(st.get("commentCount", 0)),
                "thumbnail": s["thumbnails"].get("high", s["thumbnails"].get("default", {})).get("url"),
                "query": ids[v["id"]],
            })
    relevant = [v for v in videos
                if any(k in (v["title"] + " " + v["description"]).lower() for k in AREA_KEYWORDS)]
    print(f"YouTube: {len(id_list)} unique videos, {len(relevant)} mention the Blok M area")
    return relevant


def fetch_articles() -> list[dict]:
    seen = {}
    for q in TAVILY_QUERIES:
        res = cached("tavily", q, lambda: http(
            "POST", "https://api.tavily.com/search",
            headers={"Authorization": f"Bearer {os.environ['TAVILY_API_KEY']}"},
            json={"query": q, "search_depth": "basic", "max_results": 8,
                  "include_raw_content": True, "country": "indonesia"},
        ).json())
        for r in res.get("results", []):
            text = r.get("raw_content") or r.get("content") or ""
            if r["url"] not in seen and any(k in text.lower() for k in AREA_KEYWORDS):
                seen[r["url"]] = {"url": r["url"], "title": r.get("title", ""), "content": text, "query": q}
    print(f"Tavily: {len(seen)} relevant articles")
    return list(seen.values())


def extract(doc_text: str, prefer: str) -> list[dict]:
    try:
        return llm_json(EXTRACT_SYSTEM, doc_text, cache_ns="extract", prefer=prefer).get("places", [])
    except Exception as e:
        print(f"  extraction failed: {str(e)[:100]}")
        return []


def main():
    videos = fetch_videos()
    articles = fetch_articles()
    write_json(RAW / "youtube_videos.json", videos)
    write_json(RAW / "articles.json", articles)

    jobs = []
    for v in videos:
        if len(v["description"].strip()) < 40:
            continue  # empty descriptions (mostly Shorts) rarely name a place; kept for viral matching later
        text = f"VIDEO TITLE: {v['title']}\nDESCRIPTION:\n{v['description'][:2500]}"
        jobs.append((text, {"source_type": "youtube", "source_url": v["url"], "source_title": v["title"],
                            "published_at": v["published_at"], "views": v["views"]}))
    for a in articles:
        text = f"ARTICLE TITLE: {a['title']}\nCONTENT:\n{a['content'][:7000]}"
        jobs.append((text, {"source_type": "article", "source_url": a["url"], "source_title": a["title"]}))

    print(f"Extracting places from {len(jobs)} documents (Groq + Gemini Flash Lite in parallel)...")

    def run(i_job):
        i, (text, meta) = i_job
        places = extract(text, prefer="groq" if i % 2 == 0 else "flash")
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(jobs)} documents done")
        return [{**p, **meta} for p in places if p.get("name")]

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(run, enumerate(jobs)))
    candidates = [c for r in results for c in r]
    write_json(RAW / "candidates.json", candidates)
    print(f"Candidates: {len(candidates)} mentions saved to data/raw/candidates.json")


if __name__ == "__main__":
    main()

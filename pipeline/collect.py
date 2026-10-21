"""Step 4: collect raw evidence for every place.

Per place: the videos that mention it (incl. Shorts whose title names it), relevant viewer
comments, article passages around its name, and thumbnail URLs. Output goes to
data/raw/evidence.json (gitignored: it contains third-party text)."""
import os
from collections import defaultdict

from common import DATA, RAW, cached, http, read_json, write_json
from resolve import GENERIC, norm

COMMENTS_PER_VIDEO = 20
PASSAGE_RADIUS = 500
MAX_PASSAGES = 3


def name_in(text_norm: str, name_norm: str) -> bool:
    if len(name_norm) < 5 or name_norm in GENERIC:
        return False
    if f" {name_norm} " in f" {text_norm} ":
        return True
    flat = name_norm.replace(" ", "")
    return len(flat) >= 7 and flat in text_norm.replace(" ", "")


def video_comments(video_id: str) -> list[str]:
    def call():
        try:
            res = http("GET", "https://www.googleapis.com/youtube/v3/commentThreads", params={
                "part": "snippet", "videoId": video_id, "maxResults": COMMENTS_PER_VIDEO,
                "order": "relevance", "textFormat": "plainText", "key": os.environ["YOUTUBE_API_KEY"],
            }, retries=2).json()
        except Exception:
            return []  # comments disabled or video unavailable
        return [i["snippet"]["topLevelComment"]["snippet"]["textDisplay"] for i in res.get("items", [])]
    return cached("youtube_comments", video_id, call)


def passages(content: str, name: str) -> list[str]:
    low, key, out, start = content.lower(), name.lower(), [], 0
    while len(out) < MAX_PASSAGES:
        i = low.find(key, start)
        if i < 0:
            break
        out.append(content[max(0, i - PASSAGE_RADIUS): i + len(name) + PASSAGE_RADIUS].strip())
        start = i + PASSAGE_RADIUS
    return out


def main():
    places = read_json(DATA / "places_raw.json")
    videos = {v["url"]: v for v in read_json(RAW / "youtube_videos.json")}
    articles = {a["url"]: a for a in read_json(RAW / "articles.json")}

    # How many places each video covers decides whether its comments/thumbnail are place-specific
    video_places = defaultdict(set)
    title_norms = {url: norm(v["title"]) for url, v in videos.items()}
    for p in places:
        key = norm(p["name"])
        urls = {m["source_url"] for m in p["mentions"] if m["source_type"] == "youtube"}
        urls |= {url for url, t in title_norms.items() if name_in(t, key)}
        p["_video_urls"] = urls
        for u in urls:
            video_places[u].add(p["id"])

    evidence = {}
    n_comment_calls = 0
    for i, p in enumerate(places):
        key = norm(p["name"])
        ev = {"videos": [], "comments": [], "passages": [], "thumbnails": []}
        for url in sorted(p["_video_urls"]):
            v = videos.get(url)
            if not v:
                continue
            solo = len(video_places[url]) == 1
            ev["videos"].append({"url": url, "title": v["title"], "views": v["views"],
                                 "published_at": v["published_at"], "solo": solo,
                                 "description": v["description"][:1500]})
            if v["comments"]:
                n_comment_calls += 1
                for c in video_comments(v["video_id"]):
                    if solo or name_in(norm(c), key):
                        ev["comments"].append({"url": url, "text": c[:400]})
            if solo and v.get("thumbnail"):
                ev["thumbnails"].append({"url": v["thumbnail"], "video_url": url})
        for m in p["mentions"]:
            a = articles.get(m["source_url"])
            if a:
                for text in passages(a["content"], p["name"]):
                    ev["passages"].append({"url": a["url"], "title": a["title"], "text": text})
        evidence[p["id"]] = ev
        del p["_video_urls"]
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(places)} places collected")

    write_json(RAW / "evidence.json", evidence)
    with_any = sum(1 for e in evidence.values() if e["videos"] or e["passages"])
    print(f"Evidence for {with_any}/{len(places)} places | comment requests: {n_comment_calls} | "
          f"comments kept: {sum(len(e['comments']) for e in evidence.values())} | "
          f"passages: {sum(len(e['passages']) for e in evidence.values())} | "
          f"thumbnails: {sum(len(e['thumbnails']) for e in evidence.values())}")


if __name__ == "__main__":
    main()

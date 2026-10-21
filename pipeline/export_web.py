"""Step 8: export compact data files for the static web app (web/, served by GitHub Pages).

web/data/places.json   slim place records (only what the UI shows)
web/data/vectors.json  MiniLM text vectors (same order as places) and CLIP photo vectors,
                       as base64 float32, so the browser can search without a server."""
import base64
from datetime import datetime, timezone

import numpy as np

from common import DATA, ROOT, read_json, write_json

WEB_DATA = ROOT / "web" / "data"


def b64(vectors) -> str:
    return base64.b64encode(np.asarray(vectors, dtype=np.float32).tobytes()).decode()


def slim(p: dict) -> dict:
    def sig(s):
        return {"value": s["value"], "confidence": s["confidence"],
                "evidence": [{k: e.get(k) for k in ("source_type", "url", "quote")} for e in s["evidence"][:3]]}
    b = p["buzz"]
    return {
        "id": p["id"], "name": p["name"], "lat": round(p["lat"], 6), "lon": round(p["lon"], 6),
        "precision": p["location_precision"], "where": p.get("address") or p.get("area_hint") or "Blok M",
        "category": p["category"], "cuisine": p["cuisine"], "price": p["price_idr_per_person"],
        "vibe": p["vibe_tags"], "summary": p["summary"],
        "signals": {k: sig(v) for k, v in p["signals"].items()},
        "wfc": p["wfc_score"], "wfc_known": p["wfc_evidence"],
        "buzz": {k: b[k] for k in ("trending_score", "favorite_score", "views_60d", "n_recent_videos",
                                   "n_recent_articles", "n_videos", "n_articles", "first_seen", "last_seen")},
        "new": ({"kind": b["new"]["kind"], "evidence": b["new"]["evidence"][:2]} if b["new"] else None),
        "videos": sorted(({k: v[k] for k in ("url", "title", "views", "published_at")} for v in p["videos"]),
                         key=lambda v: v["published_at"], reverse=True)[:3],
        "thumb": p["thumbnails"][0]["url"] if p.get("thumbnails") else None,
    }


def main():
    places = read_json(DATA / "places.json")
    vectors = read_json(DATA / "vectors.json")
    order = {pid: i for i, pid in enumerate(vectors["places_text"]["ids"])}
    places.sort(key=lambda p: order[p["id"]])  # text vectors and places share one order
    photo = vectors["places_photo"]
    write_json(WEB_DATA / "places.json", {
        "built": datetime.now(timezone.utc).date().isoformat(),
        "places": [slim(p) for p in places],
    })
    write_json(WEB_DATA / "vectors.json", {
        "text": {"dim": 384, "data": b64(vectors["places_text"]["embeddings"])},
        "photo": {"dim": 512, "data": b64(photo["embeddings"]),
                  "items": [{"place_id": m["place_id"], "thumb": m["thumbnail"], "video": m["video_url"]}
                            for m in photo["metadatas"]]},
    })
    sizes = {f.name: f"{f.stat().st_size / 1e6:.2f} MB" for f in WEB_DATA.glob("*.json")}
    print(f"web export: {len(places)} places, {len(photo['ids'])} photos -> {sizes}")


if __name__ == "__main__":
    main()

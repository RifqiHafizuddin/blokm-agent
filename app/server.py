"""Food Scout MCP server: Blok M places exposed as tools (capstone Module 4 pattern).

Tools: search_places, get_place, vibe_match_photo, trending, work_spots, group_pick.
All tools are deterministic (no LLM calls), so the server itself never touches LLM quotas."""
import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
from fastmcp import FastMCP

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CLIP_MODEL = "openai/clip-vit-base-patch32"

mcp = FastMCP("Food-Scout-Blok-M")


# ---------- data and models (loaded lazily, once) ----------

@lru_cache(maxsize=1)
def places() -> dict:
    return {p["id"]: p for p in json.loads((DATA / "places.json").read_text(encoding="utf-8"))}


@lru_cache(maxsize=1)
def chroma():
    import chromadb
    client = chromadb.PersistentClient(path=str(DATA / "chroma"))
    return client.get_collection("places_text"), client.get_collection("places_photo")


@lru_cache(maxsize=1)
def text_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(TEXT_MODEL)


@lru_cache(maxsize=1)
def clip():
    from transformers import CLIPModel, CLIPProcessor
    return CLIPModel.from_pretrained(CLIP_MODEL).eval(), CLIPProcessor.from_pretrained(CLIP_MODEL)


def embed_text(texts: list[str]) -> np.ndarray:
    return text_model().encode(texts, normalize_embeddings=True)


def embed_image(path: str) -> np.ndarray:
    import torch
    from PIL import Image
    model, processor = clip()
    with torch.no_grad():
        pixels = processor(images=[Image.open(path).convert("RGB")], return_tensors="pt")["pixel_values"]
        f = model.visual_projection(model.vision_model(pixel_values=pixels).pooler_output)
    return (f / f.norm(dim=-1, keepdim=True)).numpy()[0]


# ---------- formatting ----------

def card(p: dict, **extra) -> dict:
    """Compact view of a place for tool results."""
    s = p["signals"]
    return {
        "id": p["id"], "name": p["name"], "category": p["category"], "cuisine": p["cuisine"],
        "price_idr_per_person": p["price_idr_per_person"],
        "halal": ("not confirmed" if s["halal"]["value"] == "unknown"
                  else f'{s["halal"]["value"]} ({s["halal"]["confidence"]} confidence)'),
        "vegetarian_friendly": s["vegetarian_friendly"]["value"],
        "work_from_cafe_score": p["wfc_score"], "viral_score": p["viral"]["score"],
        "vibe": p["vibe_tags"], "summary": p["summary"],
        "location": p.get("address") or p.get("area_hint") or "Blok M",
        "location_precision": p["location_precision"],
        "map_is_approximate": p["location_precision"] in ("street", "landmark", "area"),
        "map": f"https://www.openstreetmap.org/?mlat={p['lat']}&mlon={p['lon']}#map=19/{p['lat']}/{p['lon']}",
        **extra,
    }


def unique_by_name(ids: list[str]) -> list[str]:
    """Keep the first (best-ranked) place per name, e.g. one Kopi Kenangan branch, not three."""
    seen, out = set(), []
    for pid in ids:
        key = places()[pid]["name"].lower().strip()
        if key not in seen:
            seen.add(key)
            out.append(pid)
    return out


def build_where(category, halal_only, vegetarian_only, max_price_idr) -> Optional[dict]:
    clauses = []
    if category:
        clauses.append({"category": category})
    if halal_only:
        clauses.append({"halal": "yes"})
    if vegetarian_only:
        clauses.append({"vegetarian": "yes"})
    if max_price_idr:
        clauses += [{"price_idr": {"$gt": 0}}, {"price_idr": {"$lte": max_price_idr}}]
    return None if not clauses else clauses[0] if len(clauses) == 1 else {"$and": clauses}


def sim(squared_l2: float) -> float:
    return round(1 - squared_l2 / 2, 3)  # unit vectors: cosine = 1 - d^2/2


# ---------- tools ----------

@mcp.tool()
def search_places(query: str, category: Optional[str] = None, halal_only: bool = False,
                  vegetarian_only: bool = False, max_price_idr: Optional[int] = None, k: int = 5) -> str:
    """Semantic search over Blok M food places.
    query: what the user wants, e.g. "cozy ramen spot" or "cheap padang food".
    category: optional, one of cafe, restaurant, street_food, dessert, bakery, bar, food_court.
    halal_only / vegetarian_only: only places with that signal = yes.
    max_price_idr: only places with a known price per person at or below this (IDR)."""
    text_col, _ = chroma()
    where = build_where(category, halal_only, vegetarian_only, max_price_idr)
    res = text_col.query(query_embeddings=embed_text([query]).tolist(), n_results=min(3 * k, 30), where=where)
    dist = dict(zip(res["ids"][0], res["distances"][0]))
    out = [card(places()[pid], similarity=sim(dist[pid])) for pid in unique_by_name(res["ids"][0])[:k]]
    return json.dumps({"query": query, "filters": where, "results": out}, ensure_ascii=False, indent=2)


@mcp.tool()
def get_place(name_or_id: str) -> str:
    """Full record for one place, including the evidence (quotes and links) behind every
    halal, vegetarian, wifi, outlet, quiet and long-stay signal. Accepts an id or a (partial) name."""
    q = name_or_id.lower().strip()
    p = places().get(name_or_id) or next((p for p in places().values() if q in p["name"].lower()), None)
    if not p:
        return json.dumps({"status": "not_found", "message": f"No place matching '{name_or_id}'."})
    signals = {k: {"value": v["value"], "confidence": v["confidence"], "evidence": v["evidence"][:4]}
               for k, v in p["signals"].items()}
    full = card(p, signals=signals, price_evidence=p["price_evidence"][:2],
                videos=sorted(p["videos"], key=lambda v: -v["views"])[:3], viral=p["viral"])
    return json.dumps({"status": "found", "place": full}, ensure_ascii=False, indent=2)


@mcp.tool()
def vibe_match_photo(image_path: str, text_hint: Optional[str] = None, w_photo: float = 0.7, k: int = 5) -> str:
    """Find Blok M places whose photos look like the given image (CLIP image-to-image search).
    image_path: local path of the uploaded photo. text_hint: optional words to blend in
    (e.g. "coffee", "ramen"), weighted by 1 - w_photo, as in late fusion."""
    text_col, photo_col = chroma()
    res = photo_col.query(query_embeddings=[embed_image(image_path).tolist()], n_results=30)
    best = {}
    for meta, d in zip(res["metadatas"][0], res["distances"][0]):
        pid = meta["place_id"]
        if pid not in best or sim(d) > best[pid]["photo_similarity"]:
            best[pid] = {"photo_similarity": sim(d), "thumbnail": meta["thumbnail"], "video": meta["video_url"]}
    if text_hint and best:
        t = text_col.get(ids=list(best), include=["embeddings"])
        hint = embed_text([text_hint])[0]
        for pid, emb in zip(t["ids"], t["embeddings"]):
            best[pid]["text_similarity"] = round(float(np.dot(hint, emb)), 3)
    for b in best.values():
        b["score"] = round(w_photo * b["photo_similarity"] + (1 - w_photo) * b.get("text_similarity", 0)
                           if text_hint else b["photo_similarity"], 3)
    ranked = sorted(best.items(), key=lambda kv: -kv[1]["score"])[:k]
    return json.dumps({"results": [card(places()[pid], **b) for pid, b in ranked]}, ensure_ascii=False, indent=2)


@mcp.tool()
def trending(k: int = 10, category: Optional[str] = None) -> str:
    """Viral Radar: Blok M places ranked by recent buzz (YouTube views weighted by recency,
    plus food-media mentions). Optionally filter by category."""
    ps = [p for p in places().values() if not category or p["category"] == category]
    ps.sort(key=lambda p: -p["viral"]["score"])
    ps = [places()[pid] for pid in unique_by_name([p["id"] for p in ps])]
    out = [card(p, buzz={k2: p["viral"][k2] for k2 in ("views_90d", "n_videos", "n_articles", "last_seen")},
                top_video=max(p["videos"], key=lambda v: v["views"])["url"] if p["videos"] else None)
           for p in ps[:k]]
    return json.dumps({"results": out}, ensure_ascii=False, indent=2)


@mcp.tool()
def work_spots(k: int = 5, halal_only: bool = False) -> str:
    """Best places to work or study (WFC / nugas): ranked by the work-from-cafe score (0-10)
    from wifi, power outlets, quiet and long-stay evidence, with each signal shown."""
    ps = [p for p in places().values() if p["wfc_score"] > 0
          and (not halal_only or p["signals"]["halal"]["value"] == "yes")]
    ps.sort(key=lambda p: (-p["wfc_score"], -p["wfc_evidence"]))
    ps = [places()[pid] for pid in unique_by_name([p["id"] for p in ps])]
    def fmt(sig):
        return "not confirmed" if sig["value"] == "unknown" else f'{sig["value"]} ({sig["confidence"]})'
    out = [card(p, wfc_signals={s: fmt(p["signals"][s]) for s in ("wifi", "outlets", "quiet", "long_stay")})
           for p in ps[:k]]
    return json.dumps({"results": out}, ensure_ascii=False, indent=2)


@mcp.tool()
def group_pick(members: list[dict], k: int = 3) -> str:
    """Pick places that suit a whole group. members: list of
    {"name": str, "wants": str, "halal": bool, "vegetarian": bool, "max_price_idr": int|null}.
    Hard rules (halal, vegetarian, budget) apply to everyone; each place is then scored per member
    against what they want, and places are ranked by the LEAST happy member's score (maximin),
    so nobody is left out. Returns each member's fit so the trade-offs can be explained."""
    text_col, _ = chroma()
    pool = list(places().values())
    for m in members:
        if m.get("halal"):
            pool = [p for p in pool if p["signals"]["halal"]["value"] == "yes"]
        if m.get("vegetarian"):
            pool = [p for p in pool if p["signals"]["vegetarian_friendly"]["value"] == "yes"]
        if m.get("max_price_idr"):
            pool = [p for p in pool if not p["price_idr_per_person"] or p["price_idr_per_person"] <= m["max_price_idr"]]
    if not pool:
        return json.dumps({"status": "no_match", "message": "No place satisfies everyone's hard rules. Try relaxing one."})

    emb = text_col.get(ids=[p["id"] for p in pool], include=["embeddings"])
    vecs = dict(zip(emb["ids"], np.array(emb["embeddings"])))
    wants = embed_text([m.get("wants") or "good food" for m in members])
    scores = {}
    for i, m in enumerate(members):
        raw = {pid: float(np.dot(wants[i], v)) for pid, v in vecs.items()}
        lo, hi = min(raw.values()), max(raw.values())
        scores[m.get("name", f"member{i + 1}")] = {pid: (s - lo) / (hi - lo + 1e-9) for pid, s in raw.items()}
    ranked = sorted(vecs, key=lambda pid: (-min(s[pid] for s in scores.values()),
                                           -np.mean([s[pid] for s in scores.values()])))
    out = [card(places()[pid], fit_per_member={n: round(s[pid], 2) for n, s in scores.items()},
                group_fit=round(min(s[pid] for s in scores.values()), 2)) for pid in unique_by_name(ranked)[:k]]
    return json.dumps({"status": "ok", "pool_size": len(pool), "results": out}, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    print("Food Scout MCP server starting", file=sys.stderr)
    mcp.run()

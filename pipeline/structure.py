"""Step 5: turn each place's raw evidence into a validated Place record.

The LLM only reads the evidence and cites it ([S1], [S2], ... plus a verbatim quote).
Code then (a) keeps only quotes that really appear in the cited source and (b) assigns
confidence by rule, so the model cannot inflate its own certainty:
  high    OSM tag, or 2+ independent sources agree
  medium  1 source
  low     the model's inference without a verifiable quote (e.g. from the name or cuisine)
Places with only a vague location get geocoded again from any address found in the evidence."""
import json
import re
from typing import Literal, Optional

from pydantic import BaseModel, Field, ValidationError
from rapidfuzz import fuzz

from common import DATA, RAW, in_bbox, llm_json, read_json, write_json
from resolve import FOOD_TYPES, landmark_query, nominatim

BATCH_WITH_EVIDENCE = 5
BATCH_OSM_ONLY = 10
MAX_CHARS_PER_PLACE = 4500
SIGNALS = ["halal", "vegetarian_friendly", "wifi", "outlets", "quiet", "long_stay"]
CATEGORIES = ["cafe", "restaurant", "street_food", "dessert", "bakery", "bar", "food_court", "other"]


class Cite(BaseModel):
    source: str = ""  # missing source -> the quote is dropped by verified()
    quote: str = ""


class SignalOut(BaseModel):
    value: Literal["yes", "no", "unknown"]
    evidence: list[Cite] = Field(default_factory=list)


class Signals(BaseModel):
    halal: SignalOut
    vegetarian_friendly: SignalOut
    wifi: SignalOut
    outlets: SignalOut
    quiet: SignalOut
    long_stay: SignalOut


class PlaceOut(BaseModel):
    id: str
    clean_name: str
    category: Literal["cafe", "restaurant", "street_food", "dessert", "bakery", "bar", "food_court", "other"]
    cuisine: list[str]
    price_idr_per_person: Optional[int] = None
    price_evidence: list[Cite] = Field(default_factory=list)
    vibe_tags: list[str]
    summary: str
    address: Optional[str] = None
    signals: Signals


SYSTEM = f"""You are a careful data analyst structuring evidence about food places in Blok M, Jakarta.
For EACH place in the input, return one object. Output JSON: {{"places": [ ... ]}} with fields:
- id: copy exactly from the input
- clean_name: the business name, properly capitalised, without location words like "Blok M"
- category: one of {CATEGORIES}
- cuisine: 1-3 lowercase English cuisine tags (e.g. "padang", "japanese", "coffee", "bakery", "noodles")
- price_idr_per_person: typical spend per person in IDR as an integer ONLY if the evidence states prices
  (e.g. "25rb" = 25000, "Rp 45.000" = 45000), else null; cite it in price_evidence
- vibe_tags: 2-5 lowercase English words describing the ATMOSPHERE, look or setting
  (e.g. "cozy", "minimalist", "industrial", "street-side", "crowded", "spacious", "outdoor", "retro").
  Never use popularity words such as "viral", "popular", "hits", "trendy", "famous", and never food words.
- summary: 1-2 English sentences, max 40 words, using only the evidence: what it is, what to order
  (named dishes or drinks if mentioned), and anything notable (price, queue, seating). Avoid filler like "culinary spot"
- address: a street address or building + floor if the evidence gives one, else null
- signals: an object with keys {SIGNALS}; each is {{"value": "yes"|"no"|"unknown", "evidence": [{{"source": "S<n>", "quote": "..."}}]}}
  halal = halal food (pork, "babi", "non-halal" or lard means "no"); vegetarian_friendly = good meat-free options;
  wifi = wifi available; outlets = power outlets / "colokan"; quiet = calm enough to work or study;
  long_stay = OK to sit for hours (e.g. "nugas", "WFC", "betah").
Rules:
- Use ONLY the given evidence. Never invent facts.
- Every quote must be copied verbatim from the cited source (max 20 words).
- If there is no evidence for a signal, use "unknown" with an empty evidence list.
- You may set a signal without a quote only when it follows directly from the name or cuisine
  (e.g. a pork restaurant is not halal); leave evidence empty in that case."""

REPAIR_SYSTEM = """You fix JSON for a food place record so it validates against the required schema.
Keep all correct content, fix only what the validation error points at, and return {"places": [<fixed object>]}."""


def build_doc(p: dict, ev: dict) -> tuple[str, dict]:
    """Evidence text with [S<n>] source tags, plus the map from tag to source info."""
    sources, lines = {}, []
    tags = ", ".join(f"{k}={v}" for k, v in p.get("osm_tags", {}).items())
    lines.append(f"PLACE id={p['id']} | name={p['name']} | category={p.get('category')} | "
                 f"cuisine={','.join(p.get('cuisine', []))} | osm_tags: {tags or 'none'}")
    if p.get("area_hint"):
        lines.append(f"location hint: {p['area_hint']}")

    def add(kind, url, text, title=""):
        sid = f"S{len(sources) + 1}"
        sources[sid] = {"type": kind, "url": url, "text": text, "title": title}
        lines.append(f"[{sid}] {kind} {('“' + title[:80] + '”') if title else ''}: {text}")

    # Most descriptive evidence first, so the character budget is spent on substance, not on
    # multi-place vlog lists (which mostly just name the place).
    name = p["name"].lower()
    for a in ev.get("passages", [])[:6]:
        add("web" if a.get("web") else "article", a["url"], a["text"][:700].replace("\n", " "), a["title"])
    for c in ev.get("comments", [])[:10]:
        add("comment", c["url"], c["text"].replace("\n", " "))
    for v in ev.get("videos", []):
        d = v["description"]
        i = d.lower().find(name)
        text = d[:500] if v["solo"] else (d[max(0, i - 150): i + 200] if i >= 0 else "")
        add("youtube", v["url"], f"{v['title']} | {text}".replace("\n", " "), "")
    for m in p.get("mentions", []):
        if m.get("quote"):
            add(m["source_type"], m["source_url"], m["quote"], m.get("source_title", ""))

    doc = "\n".join(lines)
    return doc[:MAX_CHARS_PER_PLACE], sources


def clean_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


NEGATION = re.compile(r"(non|tidak|bukan|belum|not|no|tanpa|without)[\s-]*$")
NEGATIVE_TERMS = ("non halal", "non-halal", "nonhalal", "tidak halal", "bukan halal", "belum halal",
                  "babi", "pork", "lard", "b2", "bpk")
DIET_SIGNALS = ("halal", "vegetarian_friendly")


def verified(cites: list[Cite], sources: dict) -> list[dict]:
    """Keep only quotes that really appear in the cited source. Exact match, or a near match for long
    quotes only (a short word like "halal" must match exactly). Records the 20 characters before
    the quote so negations ("non halal", "tidak halal") can be detected."""
    out = []
    for c in cites:
        src = sources.get(c.source.strip("[]"))
        q = clean_spaces(c.quote)
        if not src or len(q) < 3:
            continue
        text = clean_spaces(src["text"] + " " + src.get("title", ""))
        i = text.find(q)
        if i < 0 and not (len(q) >= 20 and fuzz.partial_ratio(q, text) >= 95):
            continue
        out.append({"source_type": src["type"], "url": src["url"], "quote": c.quote.strip()[:300],
                    "_before": text[max(0, i - 20):i] if i >= 0 else ""})
    return out


def supports(name: str, value: str, e: dict) -> bool:
    """Does this quote support the claimed value? Guards against "halal" matched inside "non halal"."""
    if name != "halal" or value != "yes":
        return True
    q = clean_spaces(e["quote"])
    return not NEGATION.search(e["_before"]) and not any(t in q for t in NEGATIVE_TERMS)


def osm_signal(tags: dict, name: str):
    key = {"halal": "diet:halal", "vegetarian_friendly": "diet:vegetarian", "wifi": "internet_access"}.get(name)
    if not key or key not in tags:
        return None
    val = tags[key].lower()
    if val in ("yes", "only", "wlan", "wifi"):
        return "yes"
    if val == "no":
        return "no"
    return None


def finalize_signal(name: str, out: SignalOut, sources: dict, tags: dict) -> dict:
    ev = [e for e in verified(out.evidence, sources) if supports(name, out.value, e)]
    for e in ev:
        e.pop("_before", None)
    osm_val = osm_signal(tags, name)
    if osm_val:
        ev.insert(0, {"source_type": "osm", "url": "https://www.openstreetmap.org", "quote": f"OSM tag: {name}={osm_val}"})
        return {"value": osm_val, "confidence": "high", "evidence": ev}
    if out.value == "unknown" or (name in DIET_SIGNALS and not ev):
        return {"value": "unknown", "confidence": "none", "evidence": []}  # diet claims are never inferred
    independent = {e["url"] + ("#comments" if e["source_type"] == "comment" else "") for e in ev}
    conf = "high" if len(independent) >= 2 else "medium" if independent else "low"
    if name in DIET_SIGNALS and all(e["source_type"] == "comment" for e in ev):
        conf = "low"  # anonymous viewer comments alone never make a diet claim more than low
    return {"value": out.value, "confidence": conf, "evidence": ev}


URLISH = re.compile(r"(https?://|www\.)\S*|[<>]", re.I)


def tidy(text, limit: int) -> str:
    """Free LLM text is published: strip links and markup, collapse whitespace, cap the length."""
    return re.sub(r"\s+", " ", URLISH.sub("", str(text or ""))).strip()[:limit]


def run_batch(docs: list[str], prefer: str) -> dict:
    res = llm_json(SYSTEM, "\n\n---\n\n".join(docs), cache_ns="structure", prefer=prefer)
    return {item.get("id"): item for item in res.get("places", []) if isinstance(item, dict)}


def validate(raw: dict, place_id: str, doc: str) -> PlaceOut:
    """Pydantic validation with an auto-repair loop (the M1L1 pattern)."""
    candidate = raw
    if candidate is None:  # the model skipped this place in its batch: ask about it alone
        candidate = run_batch([doc], prefer="groq").get(place_id) or next(iter(run_batch([doc], prefer="flash").values()), {})
    for attempt in range(3):
        try:
            obj = PlaceOut.model_validate(candidate)
            obj.id = place_id
            return obj
        except ValidationError as e:
            if attempt == 2:
                raise
            fixed = llm_json(REPAIR_SYSTEM, f"Evidence:\n{doc}\n\nCandidate:\n{json.dumps(candidate)}\n\n"
                             f"Validation error:\n{e}", cache_ns="structure_repair", prefer="flash")
            candidate = (fixed.get("places") or [None])[0]


def relocate(p: dict, address: Optional[str]) -> None:
    """Step 2.7: use an address found in the evidence to improve a vague location."""
    if p["location_precision"] != "area" or not address:
        return
    query = landmark_query(address)
    for q, precision in ((address, "exact"), (query, "landmark")):
        if not q:
            continue
        for hit in nominatim(q):
            lat, lon = float(hit["lat"]), float(hit["lon"])
            if in_bbox(lat, lon) and (precision == "landmark" or hit.get("type") in FOOD_TYPES or hit.get("category") == "highway"):
                p.update(lat=lat, lon=lon, location_precision=precision if precision == "landmark" else "street")
                return


def main():
    places = read_json(DATA / "places_raw.json")
    evidence = read_json(RAW / "evidence.json")
    docs = {p["id"]: build_doc(p, evidence.get(p["id"], {})) for p in places}

    rich = [p for p in places if evidence.get(p["id"], {}).get("videos") or evidence.get(p["id"], {}).get("passages")]
    rich_ids = {p["id"] for p in rich}
    lean = [p for p in places if p["id"] not in rich_ids]
    batches = [rich[i:i + BATCH_WITH_EVIDENCE] for i in range(0, len(rich), BATCH_WITH_EVIDENCE)]
    batches += [lean[i:i + BATCH_OSM_ONLY] for i in range(0, len(lean), BATCH_OSM_ONLY)]
    print(f"Structuring {len(places)} places in {len(batches)} LLM calls...")

    raw_out = {}
    for i, batch in enumerate(batches):
        try:
            raw_out.update(run_batch([docs[p["id"]][0] for p in batch], prefer="flash"))
        except Exception as e:
            print(f"  batch {i} failed: {str(e)[:100]}")
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(batches)} batches done")

    final, failed = [], []
    for p in places:
        doc, sources = docs[p["id"]]
        try:
            out = validate(raw_out.get(p["id"]), p["id"], doc)
        except Exception as e:
            failed.append(p["id"])
            print(f"  {p['name']}: could not validate ({str(e)[:80]})")
            continue
        tags = p.get("osm_tags", {})
        relocate(p, out.address)
        price_ev = [{k: v for k, v in e.items() if k != "_before"} for e in verified(out.price_evidence, sources)]
        if out.price_idr_per_person and not 1000 <= out.price_idr_per_person <= 2_000_000:
            price_ev = []  # implausible per-person price: drop it
        final.append({
            "id": p["id"], "name": tidy(out.clean_name, 80) or tidy(p["name"], 80), "lat": p["lat"], "lon": p["lon"],
            "location_precision": p["location_precision"], "area_hint": p.get("area_hint", ""),
            "address": tidy(out.address, 200) or None, "origin": p["origin"], "category": out.category,
            "cuisine": [tidy(c, 30) for c in (out.cuisine or p.get("cuisine", []))][:3],
            "price_idr_per_person": out.price_idr_per_person if price_ev else None,
            "price_evidence": price_ev,
            "vibe_tags": [tidy(t, 30).lower() for t in out.vibe_tags if tidy(t, 30)][:5],
            "summary": tidy(out.summary, 400),
            "signals": {s: finalize_signal(s, getattr(out.signals, s), sources, tags) for s in SIGNALS},
            "osm_tags": tags, "mentions": p["mentions"], "n_sources": p["n_sources"],
            "single_source": p.get("single_source"),
            "total_views": p["total_views"],
            "videos": [{k: v[k] for k in ("url", "title", "views", "published_at", "solo")}
                       for v in evidence.get(p["id"], {}).get("videos", [])],
            "thumbnails": evidence.get(p["id"], {}).get("thumbnails", []),
        })

    write_json(DATA / "places.json", final)
    conf = {s: {c: sum(1 for f in final if f["signals"][s]["confidence"] == c) for c in ("high", "medium", "low")}
            for s in SIGNALS}
    print(f"Structured {len(final)}/{len(places)} places ({len(failed)} failed) -> data/places.json")
    print(f"With price: {sum(1 for f in final if f['price_idr_per_person'])}")
    for s, c in conf.items():
        print(f"  {s:20} high {c['high']:3} | medium {c['medium']:3} | low {c['low']:3}")


if __name__ == "__main__":
    main()

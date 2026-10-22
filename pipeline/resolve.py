"""Step 3: merge discovered mentions into places, match them to OSM, and locate new ones.

Location precision, from best to worst:
  osm       matched an OSM venue (exact point)
  exact     found by name in Nominatim inside the Blok M box
  landmark  placed at the building/street it was mentioned with (e.g. "Pujasera Blok M Square")
  area      no location found, but mentioned by 2+ independent sources about Blok M
Places that are not in OSM need 2+ independent sources, so a single (possibly injected) video or article
can't invent a place. Middle ground: one source is enough only if it is a YouTube video dedicated to that
place (the only place extracted from it, and named in its title). Anything weaker goes to unresolved.json."""
import re
import time
import unicodedata
from collections import Counter

from rapidfuzz import fuzz

from common import BBOX, DATA, RAW, cached, http, in_bbox, read_json, write_json

MATCH_THRESHOLD = 90
STOPWORDS = {"blok", "m", "blokm", "jakarta", "selatan", "melawai", "cabang", "by", "the", "di"}
GENERIC = {"kopi", "cafe", "coffee", "bakso", "sate", "mie", "bakmi", "ramen", "resto", "warung", "kedai", "rm"}
FOOD_TYPES = {"restaurant", "cafe", "fast_food", "food_court", "ice_cream", "bar", "pub",
              "bakery", "coffee", "pastry", "confectionery", "deli", "tea"}
BLOK_M_CENTER = ((BBOX["south"] + BBOX["north"]) / 2, (BBOX["west"] + BBOX["east"]) / 2)

# Keyword in a location hint -> the landmark to geocode. Order matters: most specific first.
LANDMARKS = [
    ("blok m square", "Blok M Square"),
    ("blok m plaza", "Blok M Plaza"),
    ("m bloc", "M Bloc Space"),
    ("mbloc", "M Bloc Space"),
    ("pasaraya", "Pasaraya Blok M"),
    ("blok m hub", "Blok M Hub"),
    ("terminal", "Terminal Blok M"),
    ("taman literasi", "Taman Literasi Martha Christina Tiahahu"),
    ("little tokyo", "Jalan Melawai"),
    ("melawai", "Jalan Melawai"),
    ("bulungan", "Jalan Bulungan"),
    ("barito", "Taman Barito"),
    ("panglima polim", "Jalan Panglima Polim"),
]
# Hints that only name the neighborhood, so they carry no extra location information
VAGUE_HINTS = {"blok m", "blokm", "blok m jakarta", "jakarta", "jakarta selatan", "kebayoran baru", "jaksel"}


def landmark_query(hint: str):
    h = hint.lower().strip()
    for keyword, query in LANDMARKS:
        if keyword in h:
            return query
    return None if h in VAGUE_HINTS else hint


def norm(name: str) -> str:
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    name = re.sub(r"[^a-z0-9 ]+", " ", name)
    return " ".join(t for t in name.split() if t not in STOPWORDS)


def same_place(a: str, b: str) -> bool:
    if not a or not b:
        return False
    if a == b:
        return True
    shorter = min(a, b, key=len)
    if len(shorter) < 5 or shorter in GENERIC:
        return False
    a_flat, b_flat = a.replace(" ", ""), b.replace(" ", "")
    short_flat, long_flat = sorted((a_flat, b_flat), key=len)
    if len(short_flat) >= 6 and long_flat.startswith(short_flat):  # "holy cow" vs "holycow steakhouse"
        return True
    return fuzz.token_set_ratio(a, b) >= MATCH_THRESHOLD


_last_call = [0.0]


def nominatim(query: str) -> list[dict]:
    """Bounded Nominatim search; cached, max 1 request/second (OSMF usage policy)."""
    def call():
        wait = 1.1 - (time.time() - _last_call[0])
        if wait > 0:
            time.sleep(wait)
        _last_call[0] = time.time()
        return http("GET", "https://nominatim.openstreetmap.org/search", params={
            "q": f"{query}, Jakarta", "format": "jsonv2", "limit": 3, "countrycodes": "id",
            "viewbox": f'{BBOX["west"]},{BBOX["north"]},{BBOX["east"]},{BBOX["south"]}', "bounded": 1,
        }).json()
    return cached("nominatim", query.lower().strip(), call)


def cluster_candidates(candidates: list[dict]) -> list[dict]:
    clusters = []
    for c in candidates:
        key = norm(c["name"])
        if not key:
            continue
        for cl in clusters:
            if same_place(key, cl["key"]):
                cl["mentions"].append(c)
                break
        else:
            clusters.append({"key": key, "mentions": [c]})
    for cl in clusters:
        m = cl["mentions"]
        cl["name"] = Counter(x["name"].strip() for x in m).most_common(1)[0][0]
        hints = [x.get("area_hint", "").strip() for x in m if x.get("area_hint", "").strip()]
        cl["area_hint"] = Counter(hints).most_common(1)[0][0] if hints else ""
        cl["kind"] = Counter(x.get("kind", "other") for x in m).most_common(1)[0][0]
        cl["n_sources"] = len({x["source_url"] for x in m})
    return clusters


def mention_record(c: dict) -> dict:
    keep = ("source_type", "source_url", "source_title", "published_at", "views", "quote", "area_hint", "recent")
    return {k: c[k] for k in keep if c.get(k) not in (None, "")}


def dedicated_video(cl: dict, places_per_url: dict) -> bool:
    """Middle ground for single-source places: allowed only from a YouTube video about this one place,
    i.e. it is the only place extracted from that video AND its name is in the video title.
    One line in a "23 spots" list doesn't qualify."""
    if cl["n_sources"] != 1:
        return False
    m = cl["mentions"][0]
    if m.get("source_type") != "youtube" or len(places_per_url.get(m["source_url"], ())) != 1:
        return False
    title = norm(m.get("source_title", ""))
    return bool(title) and fuzz.partial_ratio(cl["key"], title) >= 90


def locate(cl: dict, dedicated: bool = False):
    for hit in nominatim(cl["name"]):
        lat, lon = float(hit["lat"]), float(hit["lon"])
        if in_bbox(lat, lon) and (hit.get("type") in FOOD_TYPES or hit.get("category") in ("amenity", "shop")):
            return lat, lon, "exact"
    # Not in OSM: a single source isn't enough to publish a place (one injected video could invent one)
    if cl["n_sources"] < 2 and not dedicated:
        return None
    query = landmark_query(cl["area_hint"]) if cl["area_hint"] else None
    if query:
        for hit in nominatim(query):
            lat, lon = float(hit["lat"]), float(hit["lon"])
            if in_bbox(lat, lon):
                return lat, lon, "landmark"
    return BLOK_M_CENTER[0], BLOK_M_CENTER[1], "area"


def main():
    osm = read_json(RAW / "osm_places.json")
    candidates = read_json(RAW / "candidates.json")
    clusters = cluster_candidates(candidates)
    print(f"{len(candidates)} mentions -> {len(clusters)} distinct names")
    places_per_url = {}
    for cl in clusters:
        for m in cl["mentions"]:
            places_per_url.setdefault(m["source_url"], set()).add(cl["key"])

    places = {p["id"]: {**p, "origin": "osm", "location_precision": "osm", "mentions": []} for p in osm}
    osm_keys = [(p["id"], norm(p["name"])) for p in osm]

    discovered, unresolved = [], []
    for i, cl in enumerate(clusters):
        mentions = [mention_record(m) for m in cl["mentions"]]
        match = next((pid for pid, key in osm_keys if same_place(cl["key"], key)), None)
        if match:
            places[match]["mentions"].extend(mentions)
            continue
        dedicated = dedicated_video(cl, places_per_url)
        loc = locate(cl, dedicated)
        if (i + 1) % 25 == 0:
            print(f"  located {i + 1}/{len(clusters)} names")
        if not loc:
            unresolved.append({"name": cl["name"], "area_hint": cl["area_hint"], "mentions": mentions})
            continue
        lat, lon, precision = loc
        discovered.append({
            "id": f"disc-{re.sub(r'[^a-z0-9]+', '-', cl['key']).strip('-')}",
            "name": cl["name"], "lat": lat, "lon": lon, "category": cl["kind"], "cuisine": [],
            "osm_tags": {}, "origin": "discovered", "location_precision": precision,
            "area_hint": cl["area_hint"], "mentions": mentions,
            "single_source": "dedicated video" if dedicated and precision != "exact" else None,
        })

    all_places = list(places.values()) + discovered
    for p in all_places:
        p["n_mentions"] = len(p["mentions"])
        p["n_sources"] = len({m["source_url"] for m in p["mentions"]})
        p["total_views"] = sum(m.get("views", 0) for m in p["mentions"])
    write_json(DATA / "places_raw.json", all_places)
    write_json(RAW / "unresolved.json", unresolved)
    write_report(all_places, unresolved, len(candidates), len(clusters))


def write_report(places, unresolved, n_mentions, n_names):
    prec = Counter(p["location_precision"] for p in places)
    osm_mentioned = sum(1 for p in places if p["origin"] == "osm" and p["n_mentions"])
    top = sorted(places, key=lambda p: (-p["n_sources"], -p["total_views"]))[:15]
    lines = [
        "# Coverage report: data skeleton", "",
        "| Metric | Value |", "|---|---|",
        f"| Mentions extracted from videos and articles | {n_mentions} |",
        f"| Distinct place names | {n_names} |",
        f"| **Places in dataset** | **{len(places)}** |",
        f"| From OSM | {sum(1 for p in places if p['origin'] == 'osm')} (with social evidence: {osm_mentioned}) |",
        f"| Newly discovered | {sum(1 for p in places if p['origin'] == 'discovered')} |",
        f"| Location: osm / exact / landmark / area | {prec['osm']} / {prec['exact']} / {prec['landmark']} / {prec['area']} |",
        f"| Unresolved (1 mention, no location) | {len(unresolved)} |",
        "", "## Most-mentioned places", "",
        "| # | Place | Origin | Location | Sources | Video views |", "|---|---|---|---|---|---|",
    ]
    for i, p in enumerate(top, 1):
        lines.append(f"| {i} | {p['name']} | {p['origin']} | {p['location_precision']} | "
                     f"{p['n_sources']} | {p['total_views']:,} |")
    (DATA / "coverage_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:12]))


if __name__ == "__main__":
    main()

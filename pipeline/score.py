"""Step 6: work-from-cafe (WFC) and viral scores.

WFC (0-10): wifi 3 + outlets 3 + quiet 2 + long_stay 2. A "yes" earns its weight times a
confidence factor (high 1.0, medium 0.8, low 0.5); "no" and "unknown" earn 0, so places
with no evidence are never inflated. `wfc_evidence` says how many of the 4 signals are known.

Viral (0-100): sum over linked videos of log10(1 + views) x recency decay (half-life ~4 months)
x 1.0 for single-place videos / 0.5 for multi-place vlogs, plus 0.5 per article source;
normalised so the top place is 100."""
import math
from datetime import datetime, timezone

from common import DATA, read_json, write_json

WFC_WEIGHTS = {"wifi": 3, "outlets": 3, "quiet": 2, "long_stay": 2}
CONF_FACTOR = {"high": 1.0, "medium": 0.8, "low": 0.5}
HALF_LIFE_DAYS = 120


def wfc(signals: dict) -> tuple[float, int]:
    score, known = 0.0, 0
    for name, weight in WFC_WEIGHTS.items():
        s = signals[name]
        if s["value"] != "unknown":
            known += 1
        if s["value"] == "yes":
            score += weight * CONF_FACTOR.get(s["confidence"], 0)
    return round(score, 1), known


def viral_raw(p: dict, now: datetime) -> tuple[float, dict]:
    raw, views_90d, last_seen = 0.0, 0, None
    for v in p.get("videos", []):
        published = datetime.fromisoformat(v["published_at"].replace("Z", "+00:00"))
        age = (now - published).days
        raw += math.log10(1 + v["views"]) * 0.5 ** (age / HALF_LIFE_DAYS) * (1.0 if v["solo"] else 0.5)
        if age <= 90:
            views_90d += v["views"]
        last_seen = max(last_seen or published, published)
    articles = {m["source_url"] for m in p.get("mentions", []) if m.get("source_type") == "article"}
    raw += 0.5 * len(articles)
    return raw, {"views_90d": views_90d, "n_videos": len(p.get("videos", [])), "n_articles": len(articles),
                 "last_seen": last_seen.date().isoformat() if last_seen else None}


def main():
    places = read_json(DATA / "places.json")
    now = datetime.now(timezone.utc)
    raws = {}
    for p in places:
        p["wfc_score"], p["wfc_evidence"] = wfc(p["signals"])
        raws[p["id"]], p["viral"] = viral_raw(p, now)
    top = max(raws.values()) or 1
    for p in places:
        p["viral"]["score"] = round(100 * raws[p["id"]] / top)
    write_json(DATA / "places.json", places)

    print("Top 10 viral:")
    for p in sorted(places, key=lambda p: -p["viral"]["score"])[:10]:
        print(f"  {p['viral']['score']:3}  {p['name']:30} videos {p['viral']['n_videos']:2} | "
              f"90-day views {p['viral']['views_90d']:>9,} | last seen {p['viral']['last_seen']}")
    print("Top 10 work-from-cafe:")
    for p in sorted(places, key=lambda p: (-p["wfc_score"], -p["wfc_evidence"]))[:10]:
        print(f"  {p['wfc_score']:4}  {p['name']:30} ({p['category']}, {p['wfc_evidence']}/4 signals known)")


if __name__ == "__main__":
    main()

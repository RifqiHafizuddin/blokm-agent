"""Step 6: work-from-cafe (WFC) score and three separate "buzz" views.

WFC (0-10): wifi 3 + outlets 3 + quiet 2 + long_stay 2. A "yes" earns its weight times a confidence
factor (high 1.0, medium 0.8, low 0.5); "no"/"unknown" earn 0, so places without evidence aren't inflated.

Buzz, kept separate on purpose:
  trending_score  MOMENTUM: videos from the last 60 days, scored by views per day (log), plus
                  articles from past-month searches. Old hits score 0 here.
  favorite_score  ALL-TIME popularity: total views (log) and number of sources, no time decay.
  new             "opening" if a source says it just opened (quote kept, near the place's name),
                  or "spotted" if its first mention in any source is within the last 120 days.
A multi-place vlog counts half for each place it covers. Scores are normalised to 0-100."""
import math
import re
from datetime import datetime, timezone

from common import DATA, RAW, read_json, write_json

WFC_WEIGHTS = {"wifi": 3, "outlets": 3, "quiet": 2, "long_stay": 2}
CONF_FACTOR = {"high": 1.0, "medium": 0.8, "low": 0.5}
NOW_WINDOW_DAYS = 60
NEW_WINDOW_DAYS = 120
OPENING_WINDOW_DAYS = 180
NEAR_CHARS = 150
OPENING = re.compile(r"baru (?:aja |saja )?(?:buka|dibuka|launching|launch)|grand opening|soft opening|"
                     r"new opening|just opened|newly opened|opening soon|cabang baru|outlet baru|new branch|"
                     r"buka cabang|tempat baru|spot baru", re.I)


def wfc(signals: dict) -> tuple[float, int]:
    score, known = 0.0, 0
    for name, weight in WFC_WEIGHTS.items():
        s = signals[name]
        if s["value"] != "unknown":
            known += 1
        if s["value"] == "yes":
            score += weight * CONF_FACTOR.get(s["confidence"], 0)
    return round(score, 1), known


def parse_date(s):
    if not s:
        return None
    try:
        d = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except ValueError:
        try:
            d = datetime.strptime(str(s)[:25], "%a, %d %b %Y %H:%M:%S")
        except ValueError:
            return None
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def opening_evidence(p: dict, ev: dict, dated: dict) -> list[dict]:
    """Opening keywords near the place's name, from quotes, passages, video text and comments."""
    name = p["name"].lower()
    texts = [(m.get("quote", ""), m["source_url"]) for m in p.get("mentions", [])]
    texts += [(a["text"], a["url"]) for a in ev.get("passages", [])]
    texts += [(f"{v['title']} {v.get('description', '')}", v["url"]) for v in ev.get("videos", [])]
    texts += [(c["text"], c["url"]) for c in ev.get("comments", [])]
    solo = {v["url"] for v in ev.get("videos", []) if v.get("solo")}
    found = []
    for text, url in texts:
        low = text.lower()
        for m in OPENING.finditer(text):
            i = low.find(name)
            near = url in solo or (i >= 0 and abs(i - m.start()) <= NEAR_CHARS)
            if not near:
                continue
            when = dated.get(url)
            age = (datetime.now(timezone.utc) - when).days if when else None
            if age is not None and age > OPENING_WINDOW_DAYS:
                continue  # an old "baru buka" is not new any more
            s = max(0, m.start() - 60)
            found.append({"url": url, "quote": text[s:m.end() + 60].replace("\n", " ").strip(),
                          "date": when.date().isoformat() if when else None})
            break
    uniq = {f["url"]: f for f in found}
    return list(uniq.values())[:3]


def buzz(p: dict, ev: dict, now: datetime) -> dict:
    trend, fav, views_now, n_recent_videos = 0.0, 0.0, 0, 0
    dated, seen_dates = {}, []
    for v in p.get("videos", []):
        published = parse_date(v["published_at"])
        dated[v["url"]] = published
        seen_dates.append(published)
        w = 1.0 if v["solo"] else 0.5
        age = max((now - published).days, 1)
        fav += math.log10(1 + v["views"]) * w
        if age <= NOW_WINDOW_DAYS:
            n_recent_videos += 1
            views_now += v["views"]
            trend += math.log10(1 + 30 * v["views"] / age) * w  # ~ views per month at the current pace
    articles = {m["source_url"]: m for m in p.get("mentions", []) if m.get("source_type") == "article"}
    recent_articles = [m for m in articles.values() if m.get("recent")]
    for m in articles.values():
        d = parse_date(m.get("published_at"))
        if d:
            dated[m["source_url"]] = d
            seen_dates.append(d)
    trend += 1.0 * len(recent_articles)
    fav += 0.5 * len(articles)

    first_seen = min(seen_dates) if seen_dates else None
    opening = opening_evidence(p, ev, dated)
    new = None
    if opening:
        new = {"kind": "opening", "evidence": opening}
    elif first_seen and (now - first_seen).days <= NEW_WINDOW_DAYS:
        new = {"kind": "spotted", "evidence": []}
    return {"trending_raw": trend, "favorite_raw": fav, "views_60d": views_now,
            "n_recent_videos": n_recent_videos, "n_recent_articles": len(recent_articles),
            "n_videos": len(p.get("videos", [])), "n_articles": len(articles),
            "first_seen": first_seen.date().isoformat() if first_seen else None,
            "last_seen": max(seen_dates).date().isoformat() if seen_dates else None, "new": new}


def main():
    places = read_json(DATA / "places.json")
    evidence = read_json(RAW / "evidence.json", {})
    now = datetime.now(timezone.utc)
    for p in places:
        p["wfc_score"], p["wfc_evidence"] = wfc(p["signals"])
        p["buzz"] = buzz(p, evidence.get(p["id"], {}), now)
        p.pop("viral", None)
    for key in ("trending", "favorite"):
        top = max(p["buzz"][f"{key}_raw"] for p in places) or 1
        for p in places:
            p["buzz"][f"{key}_score"] = round(100 * p["buzz"].pop(f"{key}_raw") / top)
    write_json(DATA / "places.json", places)

    def show(title, ps, fmt):
        print(title)
        for p in ps:
            print("  " + fmt(p))
    b = lambda p: p["buzz"]  # noqa: E731
    show("🔥 Trending now (last 60 days):", sorted(places, key=lambda p: -b(p)["trending_score"])[:8],
         lambda p: f"{b(p)['trending_score']:3}  {p['name']:28} recent videos {b(p)['n_recent_videos']} | "
                   f"60-day views {b(p)['views_60d']:>9,} | recent articles {b(p)['n_recent_articles']}")
    news = [p for p in places if b(p)["new"]]
    show(f"🆕 New spots ({len(news)}):", sorted(news, key=lambda p: (b(p)["new"]["kind"] != "opening",
                                                                    -b(p)["trending_score"]))[:8],
         lambda p: f"{b(p)['new']['kind']:8} {p['name']:28} first seen {b(p)['first_seen']} | "
                   f"{(b(p)['new']['evidence'] or [{'quote': ''}])[0]['quote'][:70]}")
    show("⭐ Favorites (all-time):", sorted(places, key=lambda p: -b(p)["favorite_score"])[:8],
         lambda p: f"{b(p)['favorite_score']:3}  {p['name']:28} videos {b(p)['n_videos']} | articles {b(p)['n_articles']}")
    show("💻 Work-from-cafe:", sorted(places, key=lambda p: (-p["wfc_score"], -p["wfc_evidence"]))[:5],
         lambda p: f"{p['wfc_score']:4}  {p['name']:28} ({p['wfc_evidence']}/4 signals known)")


if __name__ == "__main__":
    main()

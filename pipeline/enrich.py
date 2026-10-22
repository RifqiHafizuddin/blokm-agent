"""Step 4b: targeted web search per place to deepen thin evidence.

One Tavily search "<name> Blok M" per place that has any social mention or is a cafe,
plus "<name> Blok M wifi colokan nugas" for cafes (work-from-cafe signals).
Passages around the place name are appended to data/raw/evidence.json as `web` sources.
Budget: ~1 credit per search on Tavily's free 1,000/month."""
from collect import passages
from common import DATA, RAW, BudgetExceeded, read_json, tavily_search, write_json

MAX_PASSAGES_PER_RESULT = 2
RESULTS_PER_SEARCH = 5


def tavily(query: str) -> dict:
    return tavily_search(query, {"query": query, "search_depth": "basic", "max_results": RESULTS_PER_SEARCH,
                                 "include_raw_content": True, "country": "indonesia"})


def main():
    places = read_json(DATA / "places_raw.json")
    evidence = read_json(RAW / "evidence.json")
    queries, capped = 0, False
    for i, p in enumerate(places):
        targets = []
        if p["n_mentions"] or p["category"] in ("cafe", "coffee"):
            targets.append(f'"{p["name"]}" Blok M')
        if p["category"] in ("cafe", "coffee"):
            targets.append(f'"{p["name"]}" Blok M wifi colokan nugas')
        ev = evidence.setdefault(p["id"], {"videos": [], "comments": [], "passages": [], "thumbnails": []})
        seen = {x["url"] for x in ev["passages"]}
        for q in targets:
            queries += 1
            try:
                res = tavily(q)
            except BudgetExceeded as e:
                if not capped:
                    print(f"  {e}; only cached searches from here on")
                capped = True
                continue
            except Exception as e:
                print(f"  {q}: failed ({str(e)[:60]})")
                continue
            for r in res.get("results", []):
                text = r.get("raw_content") or r.get("content") or ""
                if r["url"] in seen:
                    continue
                found = passages(text, p["name"])[:MAX_PASSAGES_PER_RESULT]
                if found:
                    seen.add(r["url"])
                    ev["passages"] += [{"url": r["url"], "title": r.get("title", ""), "text": t, "web": True}
                                       for t in found]
        if (i + 1) % 40 == 0:
            print(f"  {i + 1}/{len(places)} places enriched ({queries} searches)")
    write_json(RAW / "evidence.json", evidence)
    print(f"Enrichment: {queries} searches | places with passages now: "
          f"{sum(1 for e in evidence.values() if e['passages'])}/{len(places)}")


if __name__ == "__main__":
    main()

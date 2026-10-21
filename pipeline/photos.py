"""Step 5b: photo evidence from video thumbnails (Gemini Flash Lite vision).

Only thumbnails from single-place videos are used, so the photo is really of that place.
Adds visual vibe tags, and halal/pork/price clues as `photo` evidence on the signals."""
from common import DATA, http, read_json, vision_json, write_json

PROMPT = """This is a thumbnail of a food video about one restaurant or cafe in Jakarta.
Return JSON: {"food": str, "vibe_tags": [str], "halal_logo": bool, "pork_visible": bool, "price_text": str|null}
- food: what food or drink is shown, max 8 words, English
- vibe_tags: 0-3 lowercase English words about the place's look if a place interior/exterior is visible, else []
- halal_logo: true only if a halal logo or the word "halal" is clearly visible
- pork_visible: true only if pork/"babi"/"pork" text or clearly pork dishes are visible
- price_text: any price written on the image exactly as shown (e.g. "15rb", "Rp 25.000"), else null"""


def recompute(signal: dict) -> None:
    if signal["confidence"] == "high" and any(e["source_type"] == "osm" for e in signal["evidence"]):
        return
    if signal["value"] == "unknown":
        return
    if all(e["source_type"] == "photo" for e in signal["evidence"]):
        signal["confidence"] = "low"  # a vision guess alone is never more than low confidence
        return
    independent = {e["url"] + ("#comments" if e["source_type"] == "comment" else "") for e in signal["evidence"]}
    signal["confidence"] = "high" if len(independent) >= 2 else "medium" if independent else "low"


def main():
    places = read_json(DATA / "places.json")
    n_images = n_halal = n_pork = 0
    for p in places:
        visual_tags = []
        for t in p.get("thumbnails", [])[:2]:
            try:
                img = http("GET", t["url"], retries=2).content
                res = vision_json(PROMPT, img, cache_key=t["url"])
            except Exception as e:
                print(f"  {p['name']}: vision failed ({str(e)[:80]})")
                continue
            n_images += 1
            visual_tags += [v.lower() for v in res.get("vibe_tags", [])]
            ev = {"source_type": "photo", "url": t["video_url"]}
            halal = p["signals"]["halal"]
            if res.get("pork_visible"):
                n_pork += 1
                halal["evidence"].append({**ev, "quote": "Thumbnail may show pork (vision model; unverified)"})
                halal["value"] = "no"  # visible pork outweighs any halal claim
            elif res.get("halal_logo"):
                n_halal += 1
                halal["evidence"].append({**ev, "quote": "Halal logo visible in thumbnail (vision model)"})
                if halal["value"] == "unknown":
                    halal["value"] = "yes"
            recompute(halal)
            p.setdefault("photo_notes", []).append({"url": t["url"], "food": res.get("food"),
                                                    "price_text": res.get("price_text")})
        if visual_tags:
            p["vibe_tags"] = list(dict.fromkeys(p["vibe_tags"] + visual_tags))[:6]
    write_json(DATA / "places.json", places)
    print(f"Analysed {n_images} thumbnails | halal logos: {n_halal} | pork flags: {n_pork}")


if __name__ == "__main__":
    main()

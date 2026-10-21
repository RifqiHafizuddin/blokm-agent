"""Step 1: seed the place list with OpenStreetMap food & drink venues inside the Blok M box."""
from common import BBOX, RAW, cached, http, write_json

OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]
AMENITIES = "cafe|restaurant|fast_food|food_court|ice_cream|bar|pub"
SHOPS = "bakery|coffee|pastry|confectionery|deli|tea"
KEEP_TAGS = [
    "amenity", "shop", "cuisine", "opening_hours", "diet:halal", "diet:vegetarian",
    "internet_access", "website", "phone", "addr:street", "addr:housenumber", "brand", "level",
]


def query() -> str:
    b = f'{BBOX["south"]},{BBOX["west"]},{BBOX["north"]},{BBOX["east"]}'
    return (
        "[out:json][timeout:120];("
        f'nwr["amenity"~"^({AMENITIES})$"]({b});'
        f'nwr["shop"~"^({SHOPS})$"]({b});'
        ");out tags center;"
    )


def fetch() -> dict:
    last = None
    for url in OVERPASS_URLS:
        try:
            return http("POST", url, data={"data": query()}, headers={"Accept": "application/json"}).json()
        except Exception as e:
            last = e
            print(f"  {url} failed: {e}")
    raise RuntimeError(f"all Overpass servers failed: {last}")


def main():
    raw = cached("overpass", query(), fetch)
    places = []
    for el in raw["elements"]:
        tags = el.get("tags", {})
        if not tags.get("name"):
            continue
        lat = el.get("lat") or el["center"]["lat"]
        lon = el.get("lon") or el["center"]["lon"]
        places.append({
            "id": f"osm-{el['type']}-{el['id']}",
            "name": tags["name"],
            "lat": lat,
            "lon": lon,
            "category": tags.get("amenity") or tags.get("shop"),
            "cuisine": [c for c in tags.get("cuisine", "").split(";") if c],
            "osm_tags": {k: tags[k] for k in KEEP_TAGS if k in tags},
        })
    write_json(RAW / "osm_places.json", places)
    print(f"OSM: {len(raw['elements'])} elements -> {len(places)} named places saved to data/raw/osm_places.json")


if __name__ == "__main__":
    main()

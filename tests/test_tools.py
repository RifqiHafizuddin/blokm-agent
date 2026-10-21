"""MCP tool tests. They call tools through the MCP protocol (in-memory client), against the
real data/places.json and data/chroma built by the pipeline. No LLM calls."""
import asyncio
import json
import sys
from pathlib import Path

import pytest
from fastmcp import Client

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from server import mcp, places  # noqa: E402


def call(name: str, args: dict) -> dict:
    async def run():
        async with Client(mcp) as c:
            res = await c.call_tool(name, args)
            return json.loads(res.content[0].text)
    return asyncio.run(run())


def test_tools_are_discoverable():
    async def run():
        async with Client(mcp) as c:
            return {t.name for t in await c.list_tools()}
    assert asyncio.run(run()) == {"search_places", "get_place", "vibe_match_photo", "trending", "work_spots", "group_pick"}


def test_search_returns_ranked_results():
    data = call("search_places", {"query": "ramen", "k": 5})
    assert len(data["results"]) == 5
    sims = [r["similarity"] for r in data["results"]]
    assert sims == sorted(sims, reverse=True)
    assert any("ramen" in (r["name"] + r["summary"] + " ".join(r["cuisine"])).lower() for r in data["results"][:3])


def test_halal_filter_only_returns_halal():
    data = call("search_places", {"query": "noodles", "halal_only": True, "k": 5})
    assert data["results"], "expected at least one halal noodle place"
    assert all(r["halal"].startswith("yes") for r in data["results"])


def test_budget_filter():
    data = call("search_places", {"query": "food", "max_price_idr": 50000, "k": 5})
    assert all(r["price_idr_per_person"] and r["price_idr_per_person"] <= 50000 for r in data["results"])


def test_get_place_includes_evidence():
    data = call("get_place", {"name_or_id": "Kyuka"})
    assert data["status"] == "found"
    halal = data["place"]["signals"]["halal"]
    assert halal["value"] in ("yes", "no", "unknown")
    if halal["value"] != "unknown" and halal["confidence"] != "low":
        assert halal["evidence"], "non-low-confidence signals must carry evidence"


def test_get_place_not_found():
    assert call("get_place", {"name_or_id": "zzz-no-such-place"})["status"] == "not_found"


def test_every_high_confidence_signal_has_evidence():
    for p in places().values():
        for name, s in p["signals"].items():
            if s["confidence"] in ("high", "medium"):
                assert s["evidence"], f"{p['name']}.{name} is {s['confidence']} without evidence"


def test_trending_now_is_recent_momentum():
    data = call("trending", {"mode": "now", "k": 10})
    scores = [r["trending_score"] for r in data["results"]]
    assert scores and scores == sorted(scores, reverse=True) and scores[0] == 100
    assert all(r["buzz"]["n_recent_videos"] or r["buzz"]["n_recent_articles"] for r in data["results"])


def test_new_spots_have_a_reason():
    for r in call("trending", {"mode": "new", "k": 20})["results"]:
        assert r["new"] in ("opening", "spotted")
        if r["new"] == "opening":
            assert r["new_evidence"], "an 'opening' badge must carry its quote"


def test_favorites_sorted_all_time():
    scores = [r["favorite_score"] for r in call("trending", {"mode": "favorites", "k": 10})["results"]]
    assert len(scores) == 10 and scores == sorted(scores, reverse=True) and scores[0] == 100


def test_work_spots_have_positive_scores():
    data = call("work_spots", {"k": 5})
    assert all(r["work_from_cafe_score"] > 0 for r in data["results"])


def test_group_pick_maximin_and_hard_rules():
    members = [{"name": "A", "wants": "ramen", "halal": True},
               {"name": "B", "wants": "coffee and dessert", "max_price_idr": 100000}]
    data = call("group_pick", {"members": members, "k": 3})
    assert data["status"] == "ok"
    fits = [r["group_fit"] for r in data["results"]]
    assert fits == sorted(fits, reverse=True)
    for r in data["results"]:
        assert r["halal"].startswith("yes")
        assert r["group_fit"] == min(r["fit_per_member"].values())


def test_vibe_match_photo_finds_its_own_place(tmp_path):
    p = next(p for p in places().values() if p.get("thumbnails"))
    import requests
    img = tmp_path / "q.jpg"
    img.write_bytes(requests.get(p["thumbnails"][0]["url"], timeout=30).content)
    data = call("vibe_match_photo", {"image_path": str(img), "k": 3})
    assert data["results"][0]["id"] == p["id"], "a place's own thumbnail should be its best match"
    assert data["results"][0]["photo_similarity"] > 0.95

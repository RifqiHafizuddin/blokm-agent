---
title: Food Scout Blok M
emoji: 🍜
colorFrom: yellow
colorTo: red
sdk: gradio
sdk_version: 6.9.0
python_version: "3.12"
app_file: app/ui.py
pinned: false
short_description: Agentic multimodal food guide for Blok M, Jakarta
---

# 🍜 Food Scout: Blok M

An **agentic, multimodal food guide** for Blok M, Jakarta, built entirely on **free, real data**.
Ask in plain language ("ramen halal yang enak", "cozy cafe for nugas under 50k"), upload a photo to find
places with the same vibe, let a group settle on one spot, and see what's trending right now.

Every claim about a place (halal, wifi, power outlets, price) links to the **quote and source** it came from,
with a confidence level, so the app never presents a guess as a fact.

## What it does

| Feature | How |
|---|---|
| 💬 **Chat** | A ReAct agent (Gemini 3.5 Flash Lite, Groq Qwen 3.8 as fallback) that discovers and calls MCP tools |
| 📸 **Vibe Match** | CLIP image-to-image search over real place photos, optionally blended with a text hint |
| 👥 **Group Pick** | Hard rules (halal, vegetarian, budget) for everyone, then the place where the *least-happy* member is happiest (maximin) |
| 🔥 **Buzz** | Three separate views: **Trending now** (momentum: views per day on videos from the last 60 days + past-month articles), **New spots** (a source says it just opened, with the quote, or first mentioned in the last 120 days), and all-time **Favorites** |
| 💻 **Work Spots** | Work-from-cafe score from wifi, outlets, quiet and long-stay evidence |
| 🗺️ **Map** | Every located place on OpenStreetMap, sized by buzz |

## Data (refreshed 2026-10-21)

| | |
|---|---|
| Places | **233** (95 from OpenStreetMap + 138 discovered in vlogs and articles; OSM alone had 83 named venues in the area) |
| Sources | 611 YouTube videos · 86 articles + 240 targeted web searches · 456 viewer comments · 116 thumbnails |
| Evidence-backed signals | halal 39 · long-stay 19 · quiet 17 · vegetarian 14 · wifi 11 · price 18 places |
| Buzz | 81 places with momentum in the last 60 days · 17 confirmed new openings (with quotes) · 39 newly spotted |

## Architecture

```
OFFLINE PIPELINE (pipeline/)                                   ONLINE APP (app/)
 1 seed_osm     OpenStreetMap venues in the Blok M box
 2 discover     YouTube + Tavily → LLM extracts place names     MCP server (server.py)
 3 resolve      fuzzy-merge, match OSM, geocode (Nominatim)       search_places · get_place
 4 collect      videos, comments, article passages, thumbnails    vibe_match_photo · trending
   enrich       one targeted web search per place                 work_spots · group_pick
 5 structure    LLM → Pydantic Place + cited evidence           ReAct agent (agent.py)
   photos       vision LLM on thumbnails (low confidence only)  Gradio UI (ui.py)
 6 score        work-from-cafe score; trending / new / favorites
 7 index        MiniLM text + CLIP photo vectors → JSON → in-memory Chroma
```

**Trust rules.** The LLM cites sources, but *code* decides confidence: quotes are checked against the source
text, and a signal is **high** only with an OSM tag or 2+ independent sources, **medium** with one source, and
**low** when inferred or based on a vision guess alone. Location precision is explicit too: osm / exact /
street / landmark ("inside Blok M Square") / area ("somewhere in Blok M").

**Freshness.** Discovery runs both relevance-sorted searches (the classics) and newest-first searches over
the last 120 days, so "trending now" reflects current momentum rather than last year's mega-hit.

**Staying free.** Nearly all LLM work happens once, offline, and every API response is cached. The MCP tools
are deterministic, so a chat turn costs only 2-4 LLM calls. When every LLM's quota runs out, the chat falls
back to a plain search instead of failing.

## Run it locally

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt   # Windows; use .venv/bin on macOS/Linux
cp .env.example .env    # add GEMINI_API_KEY and GROQ_API_KEY (YouTube/Tavily keys are only needed to rebuild data)
python app/ui.py
```

Rebuild the data: `cd pipeline && python seed_osm.py && python discover.py && python resolve.py && python collect.py && python enrich.py && python structure.py && python photos.py && python score.py && python index.py`

Tests: `python -m pytest tests` (13 MCP tool tests, no LLM calls).

The MCP server also runs on its own (`python app/server.py`), for use from any MCP client such as Claude Desktop.

## Limitations

- Signals come from public mentions and can be wrong or outdated. **Check halal status at the venue.**
- Many small places have little or no evidence; they show "not confirmed" rather than a guess.
- Prices are only shown when a source states them.
- Coverage is Blok M only (Melawai / Little Tokyo, Blok M Square, M Bloc, Bulungan, Panglima Polim).

## Attribution

Map data © [OpenStreetMap](https://www.openstreetmap.org/copyright) contributors (ODbL) · video data via the
YouTube Data API · web search via Tavily · LLMs: Google Gemini, Groq · embeddings: MiniLM, OpenAI CLIP.

Built as a portfolio follow-up to the IBM RAG and Agentic AI Professional Certificate.

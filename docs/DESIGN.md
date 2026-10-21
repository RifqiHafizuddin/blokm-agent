# Food Scout: Blok M

> An agentic, multimodal food guide for Blok M, Jakarta, built only on free, real data.
> Ask in plain language ("cozy cafe for nugas, halal, under 50k"), upload a photo to find a place with a similar vibe, settle on a spot as a group, and see what's trending right now.

Status: design draft · Last updated: 2026-10-20

---

## 1. Goals and non-goals

**Goals**
- Rebuild the capstone's *Connoisseur Companion* on **real data** for one neighborhood (Blok M) with **zero manual data entry**.
- Four hero features plus a trending view:
  - **A. Vibe match from photo**: upload a photo, get real Blok M places with a similar look.
  - **B. Work-from-cafe score**: wifi, power outlets (colokan), noise, and whether you can stay long.
  - **C. Halal and diet confidence**: a label, a confidence level, and the **evidence** behind it.
  - **D. Group decision agent**: several people's preferences in, one place out that best suits everyone, with the trade-offs explained.
  - **Viral Radar**: what's trending in Blok M, based on YouTube, food media and search signals.
- Free to run, with a live public demo on **Hugging Face Spaces**.

**Non-goals**
- No scraping of TikTok, Instagram, GoFood, GrabFood or Google Maps. It breaks their terms and gets blocked.
- No coverage beyond Blok M in v1.
- No user accounts and no stored personal data.

---

## 2. Scope: the Blok M area

Bounding box (lat/lon): **-6.2520, 106.7930 → -6.2330, 106.8080**. It covers the Blok M terminal, Blok M Square, Melawai / Little Tokyo, Panglima Polim, Bulungan and M Bloc.

**Baseline measured on 2026-10-18 (OpenStreetMap via Overpass):**

| Metric | Value |
|---|---|
| Food places | **83** (43 restaurants · 28 cafes · 12 fast food) |
| With name | 100% |
| With cuisine | 43% |
| With opening hours | 6% |
| With halal tag | 4% |
| With wifi tag | 5% |
| With photo | ~0% |

OpenStreetMap gives a clean, named and located **skeleton**, but almost none of the information the hero features need. The discovery and enrichment pipeline (§5) fills those gaps automatically.

---

## 3. Data sources: free tiers, checked 2026-10-20

| Source | Use | Free allowance | Notes |
|---|---|---|---|
| **OpenStreetMap / Overpass** | Base list of places | Free (public server; it gets busy, so retry with backoff) | ODbL license: attribution required |
| **Nominatim** | Find coordinates for newly discovered places | Max **1 request/sec**, custom User-Agent, cache results; batch jobs run single-threaded | Not for autocomplete or systematic area downloads ([policy](https://operations.osmfoundation.org/policies/nominatim/)) |
| **YouTube Data API v3** | Vlogs: titles, descriptions, view counts, dates, top comments, thumbnails | 10,000 units/day + **100 `search.list` calls/day**; most reads cost 1 unit ([docs](https://developers.google.com/youtube/v3/getting-started)) | Thumbnails are shown as links back to the video (YouTube API terms) |
| **Tavily Search API** | Food media lists ("kuliner viral Blok M"), TikTok mentions in search snippets | **1,000 credits/month**, no card needed; a basic search costs 1 credit ([docs](https://docs.tavily.com/documentation/api-credits.md)) | Preferred over Brave, which now needs a card and gives about $5 of credit a month |
| **Reddit API** *(optional)* | r/jakarta opinions | Free for non-commercial use with limits | Verify before using; low priority |
| **Gemini API (free tier)** | LLM + vision: structuring, extracting evidence, agents | ⚠️ **Unclear.** Third-party reports range from ~20 to ~1,500 requests/day per model ([example](https://klymentiev.com/blog/gemini-api-free-tier)) | **Check the real limits in AI Studio.** Design for a low quota (see §8). Tested 2026-10-20 on our key: `gemini-3.5-flash`, `gemini-3.5-flash-lite` and `gemini-flash-latest` work; the 2.5 models are closed to new users. Use **3.5-flash-lite** for bulk structuring and **3.5-flash** for agents. |
| **Groq API (free tier)** | Bulk text: name extraction, Place structuring, evidence quotes; backup for the agent | **1,000 requests/day, 8,000 tokens/min** for `qwen/qwen3.8-27b` (read from the response headers, 2026-10-20) | Text only. JSON mode tested OK. The low token/min limit is fine for offline batches but too tight for the main live agent. |
| **CLIP ViT-B/32** (local) | Photo embeddings | Free, runs on CPU | The same model as in the capstone |
| **MiniLM-L6-v2** (local) | Text embeddings | Free, runs on CPU | The same model as in the capstone |

---

## 4. Architecture

Two halves, the same split as in the capstone (index offline, retrieve online):

```
┌──────────────────────── OFFLINE PIPELINE (run locally / weekly) ────────────────────────┐
│                                                                                          │
│  1 seed        OSM Overpass ──► 83 places (name, lat/lon, cuisine, tags)                 │
│  2 discover    YouTube search + Tavily search ──► candidate place names (LLM extraction)  │
│  3 resolve     fuzzy-match to OSM; else Nominatim geocode (1 req/s, cached) ──► new places│
│  4 collect     per place: videos, comments, thumbnails, article snippets ──► raw evidence │
│  5 structure   Gemini: raw evidence ──► Place JSON (Pydantic-validated, auto-repair loop) │
│  6 score       halal / WFC / vibe / viral scores + evidence lists                         │
│  7 index       MiniLM text index + CLIP image index ──► Chroma (persisted)               │
│                                                                                          │
│  outputs: data/places.json · data/chroma/ · data/thumbs/ (embeddings only, URLs kept)     │
└──────────────────────────────────────────────────────────────────────────────────────────┘
                                         │ (committed to the Space)
                                         ▼
┌──────────────────────────────── ONLINE APP (Hugging Face Space) ────────────────────────┐
│                                                                                          │
│  MCP server "food-scout"  ── tools ──►  search_places · vibe_match_photo ·               │
│                                         get_place · trending · group_pick                │
│                                                                                          │
│  Host: Gemini ReAct agent (tool discovery at runtime, like capstone M4L3)                │
│  UI:   Gradio — chat · photo upload · group tab · trending tab · map (Folium/Leaflet)    │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

**Why the split matters:** almost all LLM calls happen **offline, once**. At runtime the agent mostly calls deterministic tools over data that's already processed, so a small Gemini quota still works for a public demo.

---

## 5. Pipeline details

### 5.1 Discover (filling OSM's gaps)
- Run about 10–15 fixed queries, for example: `kuliner Blok M`, `cafe Blok M`, `tempat nugas Blok M`, `makanan viral Blok M`, `Little Tokyo Melawai`, `M Bloc food`, `ramen Blok M`, `halal Blok M`, `kopi Blok M`.
- YouTube `search.list` (sorted by views and by date) gives about 300 videos, and Tavily gives about 100 articles. That's well within the free limits.
- Gemini pulls out **place names plus area hints** from titles, descriptions and article text into a list of `Candidate {name, area_hint, source_url}` records.

### 5.2 Resolve
- Fuzzy-match each candidate against the OSM names (`rapidfuzz`, threshold ~85) to merge duplicates.
- Geocode unmatched candidates with Nominatim (cached, 1 request/sec). **Drop anything outside the Blok M box.**
- Expected result: ~83 → **~150–200 places** (an estimate; we'll measure it).

### 5.3 Structure (the capstone M1L1 pattern)
Each place's raw evidence goes to Gemini, which returns a `Place` validated by Pydantic. If validation fails, the auto-repair loop runs (max 3 tries).

```python
class Evidence(BaseModel):
    source: Literal["osm", "youtube", "comment", "article", "photo"]
    url: str
    quote: str              # the exact snippet supporting the claim

class Signal(BaseModel):
    value: Literal["yes", "no", "unknown"]
    confidence: Literal["high", "medium", "low"]
    evidence: list[Evidence] = []

class Place(BaseModel):
    id: str
    name: str
    lat: float
    lon: float
    category: Literal["cafe", "restaurant", "street_food", "dessert", "bar"]
    cuisine: list[str]
    price_idr_per_person: Optional[int]       # LLM estimate from mentions, else None
    vibe_tags: list[str]                       # "cozy", "industrial", "aesthetic", ...
    summary: str
    halal: Signal                              # feature C
    vegetarian_friendly: Signal                # feature C
    wfc: dict[str, Signal]                     # feature B: wifi, outlets, quiet, long_stay
    photos: list[str]                          # thumbnail URLs (link back to source)
    viral: Optional[dict]                      # views_30d, mentions, last_seen, score
    sources: list[str]
```

**Evidence rule:** a signal can be `high` confidence only if it has an OSM tag **or** at least 2 independent sources. One mention gives `medium`, and an inference from the name or cuisine alone gives `low`. If there's no evidence at all, the value is `unknown`; the system never guesses silently.

### 5.4 Scores
- **Work-from-cafe score (0–10)**: a weighted sum over wifi (3), outlets (3), quiet (2) and long_stay (2). `yes` counts fully and `unknown` counts half, and the confidence level is shown next to the score.
- **Viral score**: `log10(1 + views_in_last_90d) × recency_decay + 0.5 × distinct_sources`, normalized to 0–100.
- **Vibe**: CLIP embeddings of the thumbnails, plus MiniLM embeddings of `vibe_tags + summary`.

---

## 6. Online features → MCP tools

| Feature | MCP tool | Logic |
|---|---|---|
| Natural-language search | `search_places(query, filters)` | MiniLM search with filters (distance, price, halal, open-now when known), as in capstone M2L2 |
| **A** Vibe match from photo | `vibe_match_photo(image)` | CLIP image→image over thumbnails; the top K are combined with text similarity using adjustable weights (capstone M2L3) |
| **B** Work-from-cafe | (filter + sort) | `search_places(..., sort="wfc")` with a breakdown of each signal |
| **C** Halal and diet | `get_place(id)` | Returns the signal **with its evidence quotes and links** |
| **D** Group decision | `group_pick(members[])` | One "advocate" LLM step per member scores the candidates → a mediator picks the place with the best **minimum** satisfaction and explains the trade-offs |
| Viral Radar | `trending(window="30d")` | Ranked by viral score; shows the vlog thumbnail and the date it was mentioned |

The host is a Gemini ReAct loop that discovers tools at runtime (the capstone M4L3 pattern). The same MCP server also works from Claude Desktop, which makes a nice demo extra.

---

## 7. Repository layout

```
food-scout/
├── docs/DESIGN.md
├── pipeline/
│   ├── seed_osm.py          # 1 seed
│   ├── discover.py          # 2 YouTube + Tavily → candidates
│   ├── resolve.py           # 3 fuzzy match + Nominatim
│   ├── collect.py           # 4 per-place evidence
│   ├── structure.py         # 5 Gemini → Place (Pydantic + auto-repair)
│   ├── score.py             # 6 halal / WFC / viral
│   └── index.py             # 7 Chroma (MiniLM + CLIP)
├── app/
│   ├── server.py            # MCP server (FastMCP)
│   ├── agent.py             # ReAct host (Gemini)
│   └── ui.py                # Gradio
├── data/                    # places.json, chroma/ (generated)
├── tests/
├── .env.example             # GEMINI_API_KEY, GROQ_API_KEY, YOUTUBE_API_KEY, TAVILY_API_KEY
├── requirements.txt
└── README.md                # demo GIF, architecture, sources & attribution
```

---

## 8. Quota budget (staying free)

| Stage | Calls | Fits the free tier? |
|---|---|---|
| Discover: YouTube searches | ~15 `search.list` | ✅ (100/day) |
| Discover: YouTube details and comments | ~600 units | ✅ (10,000/day) |
| Discover: Tavily | ~30 searches per run | ✅ (1,000/month) |
| Resolve: Nominatim | ~100 lookups at 1/sec, cached | ✅ (~2 min) |
| Structure: Gemini | ~200 places × ~1.3 calls ≈ 260 | ⚠️ Fits if the quota is ~1,500/day; **at ~20/day, spread it over days or batch several places per call** |
| Runtime: Gemini per user question | ~2–4 calls | ⚠️ Add caching, a per-session cap and a **text-only fallback** (search without the LLM) when the quota runs out |

**Which model does what:**

Free-tier limits for our project, taken from AI Studio on 2026-10-20:

| Model | Requests/min | Tokens/min | Requests/day |
|---|---|---|---|
| Gemini 3.6 / 3.8 Flash | 5 | 250K | **20** |
| **Gemini 3.5 Flash Lite** | 15 | 250K | **500** |
| Gemma 4 31B / 26B | 30 | 16K | **14,400** |
| Groq Qwen 3.8 27B | — | 8K | **1,000** |

| Job | Model | Why |
|---|---|---|
| Offline: extract names, structure Places, pull evidence (text) | **Groq `qwen/qwen3.8-27b`** | 1,000 requests/day; fast; JSON mode |
| Offline: photo evidence (halal signs, menus, vibe check) | **Gemini `gemini-3.5-flash-lite`** | Reads images (tested OK); 500 requests/day |
| Online: ReAct agent, group decisions | **Gemini `gemini-3.5-flash-lite`** | 500 requests/day is enough for a demo; full Flash is only 20/day |
| Fallback for text tasks | **Gemma 4 31B** (14,400 requests/day) → Groq | Huge daily quota; 16K tokens/min. Image input returned a server error when tested. |

**Mitigations:**
- **Batch the work:** structure 5–10 places per Gemini call.
- **Cache results:** cache every LLM output to disk, so the pipeline can resume without repeating calls.
- **Back off:** if a call returns HTTP 429 (rate limited), wait and retry.
- **Plan B:** route offline batch work to a second free provider (such as Groq) if Gemini's limits are too low.

---

## 9. Build plan (about 4 weeks, part-time)

| Week | Deliverable | Done when |
|---|---|---|
| **1: Data skeleton** | Steps 1–3: OSM seed, discovery, resolve | `data/candidates.json` and `data/places_raw.json` exist, with a coverage report (how many places, how many new) |
| **2: Structure & score** | Steps 4–6: collect evidence, Gemini → Place, scores | `data/places.json` validates for 100% of places; spot-check 10 by hand |
| **3: Retrieval & agents** | Step 7 + MCP server + features A, B, C + Viral Radar | Each MCP tool has a test; vibe match returns sensible results on 5 test photos |
| **4: Group agent, UI, deploy** | Feature D, Gradio UI with a map, HF Space, README + demo GIF | A public URL works end to end; README explains sources and limits |

Stretch goals: Telegram bot front-end, weekly scheduled refresh (GitHub Actions), Bahasa Indonesia / English toggle.

---

## 10. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Gemini free quota is very low | Pipeline or demo stalls | Batching, caching, fallback provider, a no-LLM fallback search |
| Few vlogs exist for small places | Photos and evidence are sparse | Show `unknown` honestly; rank places with more evidence higher |
| Place names in vlogs are ambiguous | Wrong matches | Fuzzy matching + the Blok M box + an LLM confirmation step |
| YouTube terms on thumbnails | Compliance | Keep only URLs + embeddings; always link back to the video |
| Overpass / Nominatim downtime | Pipeline errors | Retries with backoff; cache all responses |
| Trending data goes stale | Viral Radar looks outdated | Show "last updated"; optional weekly refresh |

---

## 11. Open questions

1. What are the actual Gemini free-tier limits for our project (check in AI Studio)?
2. Should Viral Radar look back 30 days or 90 days? It depends on how many recent vlogs exist.
3. Should the UI be Bahasa Indonesia first, English first, or offer both?

---

*Attribution: map data © OpenStreetMap contributors (ODbL). Video data via the YouTube Data API.*

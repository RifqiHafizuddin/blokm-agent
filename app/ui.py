"""Food Scout: Blok M — Gradio UI.

Tabs: Chat (ReAct agent) · Vibe Match (photo) · Group Pick · Trending · Work Spots · Map · About."""
import html
import json

import gradio as gr

import agent
from server import places

MAX_CHAT_TURNS = 20  # per browser session, protects the free LLM quota
CATEGORY_COLORS = {"cafe": "#8B5E3C", "restaurant": "#D9480F", "street_food": "#E8590C", "dessert": "#D6336C",
                   "bakery": "#F08C00", "bar": "#5F3DC4", "food_court": "#1971C2", "other": "#495057"}


def place_md(p: dict, extra: str = "") -> str:
    price = f" · ~Rp{p['price_idr_per_person']:,}".replace(",", ".") if p.get("price_idr_per_person") else ""
    return (f"**{p['name']}** · {p['category']}{price}{extra}  \n"
            f"{p['summary']}  \n"
            f"Halal: {p['halal']} · Vibe: {', '.join(p['vibe'][:3]) or '—'} · "
            f"[map]({p['map']}) · {p['location']} ({p['location_precision']})")


# ---------- chat ----------

async def chat(message, history, turns):
    if not message or not message.strip():
        yield history, turns, ""
        return
    if turns >= MAX_CHAT_TURNS:
        history = history + [{"role": "user", "content": message},
                             {"role": "assistant", "content": "This demo session hit its question limit. Refresh the page to start again."}]
        yield history, turns, ""
        return
    history = history + [{"role": "user", "content": message}, {"role": "assistant", "content": "Scouting Blok M…"}]
    yield history, turns, ""
    answer, tools = await agent.ask(message, history[:-2])
    if tools:
        answer += f"\n\n<sub>tools used: {', '.join(dict.fromkeys(tools))}</sub>"
    history[-1] = {"role": "assistant", "content": answer}
    yield history, turns + 1, ""


# ---------- vibe match ----------

async def vibe_match(image_path, hint, w_photo):
    if not image_path:
        return [], "Upload a photo first."
    data = await agent.call_tool("vibe_match_photo", {"image_path": image_path, "text_hint": hint or None,
                                                      "w_photo": w_photo, "k": 6})
    gallery = [(r["thumbnail"], f"{r['name']} · {r['score']:.2f}") for r in data["results"]]
    md = "\n\n".join(place_md(r, f" · match {r['score']:.2f}") + f" · [video]({r['video']})" for r in data["results"])
    return gallery, md or "No photo matches yet."


# ---------- group pick ----------

def _truthy(v) -> bool:
    return str(v).strip().lower() in ("true", "yes", "y", "1", "ya")


async def group(table):
    rows = table.values.tolist() if hasattr(table, "values") else table
    members = []
    for r in rows:
        if not r or not str(r[0]).strip():
            continue
        budget = str(r[4]).strip().replace(".", "")
        members.append({"name": str(r[0]).strip(), "wants": str(r[1] or ""), "halal": _truthy(r[2]),
                        "vegetarian": _truthy(r[3]), "max_price_idr": int(budget) if budget.isdigit() else None})
    if len(members) < 2:
        return "Add at least two people."
    data = await agent.call_tool("group_pick", {"members": members, "k": 3})
    if data.get("status") != "ok":
        return data.get("message", "No match.")
    picks = "\n\n".join(
        place_md(r, f" · group fit {r['group_fit']}") + "  \n" +
        " · ".join(f"{n}: {s}" for n, s in r["fit_per_member"].items()) for r in data["results"])
    return f"### Top picks ({data['pool_size']} places pass everyone's rules)\n\n{picks}\n\n### Why\n\n" + \
        agent.explain_group_pick(members, data)


# ---------- trending / work spots ----------

async def trending(category):
    data = await agent.call_tool("trending", {"k": 10, "category": None if category == "all" else category})
    rows = []
    for i, r in enumerate(data["results"], 1):
        b = r["buzz"]
        top = f" · [top video]({r['top_video']})" if r.get("top_video") else ""
        rows.append(f"{i}. " + place_md(r, f" · 🔥 {r['viral_score']}") +
                    f"  \n{b['n_videos']} videos, {b['n_articles']} articles, last seen {b['last_seen']}{top}")
    return "\n\n".join(rows)


async def work(halal_only):
    data = await agent.call_tool("work_spots", {"k": 8, "halal_only": halal_only})
    if not data["results"]:
        return "No place has work-friendly evidence yet."
    return "\n\n".join(place_md(r, f" · 💻 {r['work_from_cafe_score']}/10") + "  \n" +
                       " · ".join(f"{k}: {v}" for k, v in r["wfc_signals"].items()) for r in data["results"])


# ---------- map ----------

def map_html() -> str:
    pts = [{"lat": p["lat"], "lon": p["lon"], "name": p["name"], "cat": p["category"],
            "color": CATEGORY_COLORS.get(p["category"], "#495057"), "summary": p["summary"],
            "precision": p["location_precision"], "viral": p["viral"]["score"]}
           for p in places().values() if p["location_precision"] != "area"]
    doc = f"""<!doctype html><html><head><meta charset="utf-8">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>html,body,#m{{height:100%;margin:0}}</style></head><body><div id="m"></div><script>
const pts = {json.dumps(pts, ensure_ascii=False)};
const m = L.map('m').setView([-6.2430, 106.8005], 16);
L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png',
  {{maxZoom: 19, attribution: '&copy; OpenStreetMap contributors'}}).addTo(m);
const esc = s => String(s).replace(/[&<>"]/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}}[c]));
pts.forEach(p => L.circleMarker([p.lat, p.lon], {{radius: 5 + p.viral / 20, color: p.color, fillOpacity: .7}})
  .bindPopup(`<b>${{esc(p.name)}}</b> · ${{esc(p.cat)}}<br>${{esc(p.summary)}}<br><i>location: ${{esc(p.precision)}}</i>`)
  .addTo(m));
</script></body></html>"""
    return f'<iframe style="width:100%;height:560px;border:0" srcdoc="{html.escape(doc)}"></iframe>'


ABOUT = """## About Food Scout: Blok M
An agentic, multimodal food guide for Blok M, Jakarta, built only on free, real data.

**How it works:** OpenStreetMap venues + places discovered in YouTube food vlogs and web articles →
LLM structuring with **verifiable evidence** (every halal / wifi / price claim links to its source) →
MiniLM + CLIP vector search → an MCP server whose tools a ReAct agent discovers and calls.

**Confidence:** *high* = OSM tag or 2+ independent sources · *medium* = 1 source · *low* = inferred.
Location precision: osm / exact / street / landmark (e.g. inside Blok M Square) / area (somewhere in Blok M).

**Data:** © OpenStreetMap contributors (ODbL) · video data via the YouTube Data API · web search via Tavily.
Signals come from public mentions and can be wrong or outdated; check before you go, especially halal status."""


with gr.Blocks(title="Food Scout: Blok M") as demo:
    gr.Markdown("# 🍜 Food Scout: Blok M\nWhere to eat, work, and hang out in Blok M, grounded in real vlogs, articles and maps.")
    with gr.Tabs():
        with gr.Tab("💬 Chat"):
            turns = gr.State(0)
            bot = gr.Chatbot(height=480)
            box = gr.Textbox(placeholder='e.g. "ramen halal yang enak" or "cozy cafe for nugas under 50k"', label="Ask Food Scout")
            with gr.Row():
                for q in ["What's viral in Blok M right now?", "Cafe buat nugas yang ada colokan",
                          "Halal ramen near Melawai", "Cheap eats under Rp30k"]:
                    gr.Button(q, size="sm").click(chat, [gr.State(q), bot, turns], [bot, turns, box])
            box.submit(chat, [box, bot, turns], [bot, turns, box])
        with gr.Tab("📸 Vibe Match"):
            gr.Markdown("Upload any food or cafe photo and find Blok M places that **look** like it (CLIP image search).")
            with gr.Row():
                with gr.Column(scale=1):
                    img = gr.Image(type="filepath", label="Your photo")
                    hint = gr.Textbox(label="Optional hint (e.g. 'coffee', 'ramen')")
                    w = gr.Slider(0, 1, value=0.7, step=0.1, label="Photo weight vs. hint")
                    go = gr.Button("Find similar places", variant="primary")
                with gr.Column(scale=2):
                    gal = gr.Gallery(columns=3, height=300, label="Matches")
                    vm = gr.Markdown()
            go.click(vibe_match, [img, hint, w], [gal, vm])
        with gr.Tab("👥 Group Pick"):
            gr.Markdown("Everyone's wishes and rules in, one place that suits the **whole** group out.")
            table = gr.Dataframe(headers=["name", "wants", "halal (yes/no)", "vegetarian (yes/no)", "max price (IDR)"],
                                 value=[["Rina", "spicy indonesian food", "yes", "no", "50000"],
                                        ["Dimas", "japanese, ramen", "yes", "no", ""],
                                        ["Sarah", "cozy place to chat, good coffee", "yes", "no", "75000"]],
                                 column_count=5, interactive=True, wrap=True)
            gr.Markdown("<sub>halal = only places with confirmed halal evidence; budget = only places with a known price at or under it, or no price info.</sub>")
            gp = gr.Button("Pick for the group", variant="primary")
            gout = gr.Markdown()
            gp.click(group, [table], [gout])
        with gr.Tab("🔥 Trending"):
            cat = gr.Dropdown(["all", "cafe", "restaurant", "street_food", "dessert", "bakery"], value="all", label="Category")
            tout = gr.Markdown()
            cat.change(trending, [cat], [tout])
            demo.load(trending, [cat], [tout])
        with gr.Tab("💻 Work Spots"):
            halal = gr.Checkbox(label="Halal only")
            wout = gr.Markdown()
            halal.change(work, [halal], [wout])
            demo.load(work, [halal], [wout])
        with gr.Tab("🗺️ Map"):
            gr.HTML(map_html())
        with gr.Tab("ℹ️ About"):
            gr.Markdown(ABOUT)

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())

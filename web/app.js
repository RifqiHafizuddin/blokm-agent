// Food Scout: Blok M — static web app. Search (MiniLM) and Vibe Match (CLIP) run in the browser
// via transformers.js; the optional chat agent calls the same tool functions through Gemini.
import {
  pipeline, env, AutoProcessor, CLIPVisionModelWithProjection, RawImage,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.0";

env.allowLocalModels = false;

const TEXT_MODEL = "Xenova/all-MiniLM-L6-v2";
const CLIP_MODEL = "Xenova/clip-vit-base-patch32";
const CHAT_MODEL = "gemini-3.5-flash-lite";
const $ = (sel) => document.querySelector(sel);

const S = { places: [], byId: new Map(), text: null, photo: null, photoItems: [], built: "" };

// ---------- utils ----------
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const safeUrl = (u) => (/^https?:\/\//i.test(u || "") ? u : "#");
const fmtRp = (n) => "Rp" + Number(n).toLocaleString("id-ID");
const fmtNum = (n) => Number(n || 0).toLocaleString("en-US");

function decodeVectors(b64, dim) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  const data = new Float32Array(bytes.buffer);
  return { dim, n: data.length / dim, row: (i) => data.subarray(i * dim, (i + 1) * dim) };
}
function dot(a, b) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }
function uniqueByName(items) {
  const seen = new Set();
  return items.filter((x) => { const k = x.place.name.toLowerCase().trim(); if (seen.has(k)) return false; seen.add(k); return true; });
}
function store(key, value) { try { value == null ? localStorage.removeItem(key) : localStorage.setItem(key, value); } catch { /* storage unavailable */ } }
function recall(key) { try { return localStorage.getItem(key); } catch { return null; } }

// ---------- models (lazy, cached by the browser) ----------
let textExtractor = null;
let clip = null;
function progress(statusEl, label) {
  return (p) => { if (p.status === "progress" && p.total) statusEl.textContent = `${label}… ${Math.round(p.progress)}%`; };
}
async function embedText(texts, statusEl) {
  if (!textExtractor) {
    statusEl && (statusEl.textContent = "Loading the search model (first time only, ~90 MB)…");
    textExtractor = await pipeline("feature-extraction", TEXT_MODEL, {
      dtype: "fp32", progress_callback: statusEl ? progress(statusEl, "Loading the search model") : undefined,
    });
  }
  const out = await textExtractor(texts, { pooling: "mean", normalize: true });
  return out.tolist().map((v) => Float32Array.from(v));
}
async function embedImage(url, statusEl) {
  if (!clip) {
    statusEl.textContent = "Loading CLIP (first time only, ~90 MB)…";
    const cb = progress(statusEl, "Loading CLIP");
    clip = {
      processor: await AutoProcessor.from_pretrained(CLIP_MODEL, { progress_callback: cb }),
      model: await CLIPVisionModelWithProjection.from_pretrained(CLIP_MODEL, { dtype: "fp32", progress_callback: cb }),
    };
  }
  const image = await RawImage.read(url);
  const { image_embeds } = await clip.model(await clip.processor(image));
  const v = Float32Array.from(image_embeds.data);
  const norm = Math.hypot(...v);
  return v.map((x) => x / norm);
}

// ---------- tools (used by the UI and by the chat agent) ----------
function passes(p, f = {}) {
  if (f.category && p.category !== f.category) return false;
  if (f.halal_only && p.signals.halal.value !== "yes") return false;
  if (f.vegetarian_only && p.signals.vegetarian_friendly.value !== "yes") return false;
  if (f.max_price && !(p.price && p.price <= f.max_price)) return false;
  return true;
}
async function searchPlaces({ query, k = 6, ...filters }, statusEl) {
  const [q] = await embedText([query], statusEl);
  const scored = S.places.map((p, i) => ({ place: p, score: dot(q, S.text.row(i)) })).filter((x) => passes(x.place, filters));
  scored.sort((a, b) => b.score - a.score);
  return uniqueByName(scored).slice(0, k);
}
function getPlace(nameOrId) {
  const q = String(nameOrId || "").toLowerCase().trim();
  return S.byId.get(nameOrId) || S.places.find((p) => p.name.toLowerCase().includes(q)) || null;
}
function trending({ mode = "now", k = 10, category } = {}) {
  let ps = S.places.filter((p) => !category || p.category === category);
  if (mode === "new") {
    ps = ps.filter((p) => p.new);
    ps.sort((a, b) => (a.new.kind !== "opening") - (b.new.kind !== "opening") || b.buzz.trending_score - a.buzz.trending_score);
  } else if (mode === "favorites") {
    ps.sort((a, b) => b.buzz.favorite_score - a.buzz.favorite_score);
  } else {
    ps = ps.filter((p) => p.buzz.trending_score > 0).sort((a, b) => b.buzz.trending_score - a.buzz.trending_score);
  }
  return uniqueByName(ps.map((p) => ({ place: p }))).slice(0, k);
}
function workSpots({ k = 8, halal_only = false } = {}) {
  const ps = S.places.filter((p) => p.wfc > 0 && (!halal_only || p.signals.halal.value === "yes"))
    .sort((a, b) => b.wfc - a.wfc || b.wfc_known - a.wfc_known);
  return uniqueByName(ps.map((p) => ({ place: p }))).slice(0, k);
}
async function groupPick(members, k = 3, statusEl) {
  let pool = S.places.map((p, i) => ({ p, i }));
  for (const m of members) {
    if (m.halal) pool = pool.filter(({ p }) => p.signals.halal.value === "yes");
    if (m.vegetarian) pool = pool.filter(({ p }) => p.signals.vegetarian_friendly.value === "yes");
    if (m.max_price) pool = pool.filter(({ p }) => !p.price || p.price <= m.max_price);
  }
  if (!pool.length) return { pool: 0, results: [] };
  const wants = await embedText(members.map((m) => m.wants || "good food"), statusEl);
  const fits = members.map((m, mi) => {
    const raw = pool.map(({ i }) => dot(wants[mi], S.text.row(i)));
    const lo = Math.min(...raw), hi = Math.max(...raw);
    return raw.map((s) => (s - lo) / (hi - lo + 1e-9));
  });
  const ranked = pool.map(({ p }, j) => {
    const per = fits.map((f) => f[j]);
    return { place: p, group_fit: Math.min(...per), mean: per.reduce((a, b) => a + b, 0) / per.length,
             per: Object.fromEntries(members.map((m, mi) => [m.name || `Person ${mi + 1}`, per[mi]])) };
  }).sort((a, b) => b.group_fit - a.group_fit || b.mean - a.mean);
  return { pool: pool.length, results: uniqueByName(ranked).slice(0, k) };
}
async function vibeMatch(url, hint, wPhoto, statusEl, k = 6) {
  const v = await embedImage(url, statusEl);
  const best = new Map();
  S.photoItems.forEach((item, i) => {
    const s = dot(v, S.photo.row(i));
    const cur = best.get(item.place_id);
    if (!cur || s > cur.photo) best.set(item.place_id, { photo: s, thumb: item.thumb, video: item.video });
  });
  let hintVec = null;
  if (hint) [hintVec] = await embedText([hint], statusEl);
  const out = [...best.entries()].map(([pid, b]) => {
    const idx = S.places.findIndex((p) => p.id === pid);
    const text = hintVec ? dot(hintVec, S.text.row(idx)) : 0;
    return { place: S.byId.get(pid), score: hintVec ? wPhoto * b.photo + (1 - wPhoto) * text : b.photo, ...b };
  });
  out.sort((a, b) => b.score - a.score);
  return uniqueByName(out).slice(0, k);
}

// ---------- rendering ----------
const SIGNAL_LABELS = { halal: "Halal", vegetarian_friendly: "Vegetarian", wifi: "Wifi", outlets: "Outlets", quiet: "Quiet", long_stay: "Long stay" };
const PRECISION = { osm: "exact", exact: "exact", street: "street", landmark: "approx.", area: "somewhere in Blok M" };

function halalBadge(s) {
  if (s.value === "yes") return `<span class="badge good">✓ halal · ${esc(s.confidence)}</span>`;
  if (s.value === "no") return `<span class="badge bad">✗ not halal · ${esc(s.confidence)}</span>`;
  return `<span class="badge">halal not confirmed</span>`;
}
function card(p, extra = "", thumb = p.thumb) {
  const badges = [halalBadge(p.signals.halal)];
  if (p.new) badges.push(`<span class="badge hot">${p.new.kind === "opening" ? "🆕 new opening" : "👀 newly spotted"}</span>`);
  if (p.buzz.trending_score > 0) badges.push(`<span class="badge hot">🔥 ${p.buzz.trending_score}</span>`);
  if (p.wfc > 0) badges.push(`<span class="badge">💻 ${p.wfc}/10</span>`);
  if (p.price) badges.push(`<span class="badge">~${fmtRp(p.price)}</span>`);
  const evidence = Object.entries(p.signals).filter(([, s]) => s.value !== "unknown" && s.evidence.length);
  const ev = evidence.length ? `<details><summary>Evidence (${evidence.length})</summary><ul>${evidence.map(([k, s]) =>
    `<li><b>${SIGNAL_LABELS[k]}: ${esc(s.value)}</b> (${esc(s.confidence)}) — ${s.evidence.map((e) =>
      `<a href="${esc(safeUrl(e.url))}" target="_blank" rel="noopener" class="quote">“${esc(e.quote)}”</a>`).join("; ")}</li>`).join("")}</ul></details>` : "";
  const video = p.videos[0] ? `<a href="${esc(safeUrl(p.videos[0].url))}" target="_blank" rel="noopener">▶ latest video</a>` : "";
  const map = `<a href="https://www.openstreetmap.org/?mlat=${p.lat}&mlon=${p.lon}#map=19/${p.lat}/${p.lon}" target="_blank" rel="noopener">📍 ${esc(p.where)} (${PRECISION[p.precision] || esc(p.precision)})</a>`;
  return `<article class="card ${thumb ? "" : "no-thumb"}">
    ${thumb ? `<img class="thumb" src="${esc(safeUrl(thumb))}" alt="" loading="lazy" referrerpolicy="no-referrer">` : ""}
    <div><h3>${esc(p.name)}</h3>
      <div class="meta">${esc(p.category.replace("_", " "))}${p.cuisine.length ? " · " + esc(p.cuisine.join(", ")) : ""}${p.vibe.length ? " · " + esc(p.vibe.slice(0, 3).join(", ")) : ""}</div>
      <div class="badges">${badges.join("")}</div>
      <p>${esc(p.summary)}</p>
      ${extra ? `<div class="extra">${extra}</div>` : ""}
      <div class="links">${map}${video}</div>${ev}</div></article>`;
}
const renderList = (el, html, empty = "Nothing found.") => { el.innerHTML = html || `<p class="muted">${empty}</p>`; };

// ---------- search tab ----------
const EXAMPLES = ["cozy ramen", "bakmi enak", "matcha dessert", "salt bread", "padang murah", "coffee and pastry"];
async function runSearch(query) {
  const status = $("#search-status");
  if (!query.trim()) return;
  status.textContent = "Searching…";
  const res = await searchPlaces({
    query, k: 8, category: $("#f-cat").value || undefined, halal_only: $("#f-halal").checked,
    vegetarian_only: $("#f-veg").checked, max_price: Number($("#f-price").value) || undefined,
  }, status);
  status.textContent = res.length ? `Best matches for “${query}”` : "";
  renderList($("#search-results"), res.map((r) => card(r.place)).join(""), "No place passes those filters. Try loosening one.");
}

// ---------- buzz tab ----------
const BUZZ_INTRO = {
  now: "Momentum from the last 60 days: views per day on recent videos plus past-month food articles. Old hits don't count here.",
  new: "🆕 new opening = a source says it just opened (quote shown) · 👀 newly spotted = first mentioned anywhere in the last 120 days.",
  favorites: "All-time most-covered places, with no time decay: the Blok M classics.",
};
function renderBuzz(mode) {
  document.querySelectorAll(".subtabs button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.mode === mode)));
  $("#buzz-intro").textContent = BUZZ_INTRO[mode];
  const html = trending({ mode, k: 12 }).map(({ place: p }, i) => {
    const b = p.buzz;
    let extra;
    if (mode === "now") extra = `#${i + 1} · ${b.n_recent_videos} videos in 60 days (${fmtNum(b.views_60d)} views) · ${b.n_recent_articles} recent articles`;
    else if (mode === "new") {
      const q = p.new.evidence[0];
      extra = `first seen ${esc(b.first_seen || "recently")}` + (q ? ` · <a class="quote" href="${esc(safeUrl(q.url))}" target="_blank" rel="noopener">“${esc(q.quote.slice(0, 160))}”</a>` : "");
    } else extra = `#${i + 1} · ⭐ ${b.favorite_score} · ${b.n_videos} videos, ${b.n_articles} articles overall`;
    return card(p, extra);
  }).join("");
  renderList($("#buzz-results"), html);
}

// ---------- work tab ----------
function renderWork() {
  const html = workSpots({ k: 10, halal_only: $("#work-halal").checked }).map(({ place: p }) => {
    const parts = ["wifi", "outlets", "quiet", "long_stay"].map((k) => {
      const s = p.signals[k];
      return `${SIGNAL_LABELS[k]}: ${s.value === "unknown" ? "not confirmed" : `${esc(s.value)} (${esc(s.confidence)})`}`;
    });
    return card(p, `💻 ${p.wfc}/10 · ${parts.join(" · ")}`);
  }).join("");
  renderList($("#work-results"), html, "No place has work-friendly evidence yet.");
}

// ---------- photo tab ----------
let photoUrl = null;
$("#photo-input").addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (!file) return;
  if (photoUrl) URL.revokeObjectURL(photoUrl);
  photoUrl = URL.createObjectURL(file);
  $("#photo-preview").src = photoUrl;
  $("#photo-preview").hidden = false;
  $("#photo-placeholder").hidden = true;
  $("#photo-go").disabled = false;
});
$("#photo-weight").addEventListener("input", (e) => { $("#photo-weight-out").textContent = e.target.value; });
$("#photo-go").addEventListener("click", async () => {
  const status = $("#photo-status");
  $("#photo-go").disabled = true;
  try {
    const res = await vibeMatch(photoUrl, $("#photo-hint").value.trim(), Number($("#photo-weight").value), status);
    status.textContent = "Places whose photos look most like yours:";
    renderList($("#photo-results"), res.map((r) =>
      card(r.place, `match ${(r.score * 100).toFixed(0)}% · <a href="${esc(safeUrl(r.video))}" target="_blank" rel="noopener">source video</a>`, r.thumb)).join(""));
  } catch (err) {
    status.textContent = "Couldn't analyse that photo: " + err.message;
  } finally { $("#photo-go").disabled = false; }
});

// ---------- group tab ----------
const DEFAULT_GROUP = [
  { name: "Rina", wants: "spicy indonesian food", halal: true, max_price: 50000 },
  { name: "Dimas", wants: "japanese, ramen", halal: true },
  { name: "Sarah", wants: "cozy place to chat, good coffee", halal: true, max_price: 75000 },
];
function addGroupRow(m = {}) {
  const tr = document.createElement("tr");
  tr.innerHTML = `<td><input type="text" class="g-name" value="${esc(m.name || "")}" aria-label="Name"></td>
    <td><input type="text" class="g-wants" value="${esc(m.wants || "")}" placeholder="what they want" aria-label="Wants"></td>
    <td><input type="checkbox" class="g-halal" ${m.halal ? "checked" : ""} aria-label="Halal"></td>
    <td><input type="checkbox" class="g-veg" ${m.vegetarian ? "checked" : ""} aria-label="Vegetarian"></td>
    <td><input type="number" class="g-price" min="0" step="5000" value="${m.max_price || ""}" aria-label="Max price"></td>
    <td><button type="button" class="g-del" aria-label="Remove">✕</button></td>`;
  tr.querySelector(".g-del").addEventListener("click", () => tr.remove());
  $("#group-rows").appendChild(tr);
}
function fitWord(x) { return x >= 0.7 ? "strong match" : x >= 0.4 ? "decent match" : "weak match"; }
$("#group-add").addEventListener("click", () => addGroupRow());
$("#group-go").addEventListener("click", async () => {
  const members = [...document.querySelectorAll("#group-rows tr")].map((tr) => ({
    name: tr.querySelector(".g-name").value.trim(), wants: tr.querySelector(".g-wants").value.trim(),
    halal: tr.querySelector(".g-halal").checked, vegetarian: tr.querySelector(".g-veg").checked,
    max_price: Number(tr.querySelector(".g-price").value) || undefined,
  })).filter((m) => m.name || m.wants);
  const status = $("#group-status");
  if (members.length < 2) { status.textContent = "Add at least two people."; return; }
  status.textContent = "Thinking about everyone…";
  const { pool, results } = await groupPick(members, 3, status);
  if (!results.length) { status.textContent = ""; renderList($("#group-results"), "", "No place satisfies everyone's hard rules. Try relaxing one."); return; }
  status.textContent = `${pool} places pass everyone's rules.`;
  const top = results[0];
  const why = `<div class="explain"><b>Why ${esc(top.place.name)}?</b> It's the place where the least-happy person is happiest.<ul>${
    Object.entries(top.per).map(([n, f]) => `<li>${esc(n)}: ${fitWord(f)} (${f.toFixed(2)})</li>`).join("")}</ul></div>`;
  renderList($("#group-results"), why + results.map((r, i) => card(r.place,
    `#${i + 1} · group fit ${r.group_fit.toFixed(2)} · ${Object.entries(r.per).map(([n, f]) => `${esc(n)} ${f.toFixed(2)}`).join(" · ")}`)).join(""));
});

// ---------- map tab ----------
let map = null;
const CAT_COLORS = { cafe: "#8B5E3C", restaurant: "#D9480F", street_food: "#E8590C", dessert: "#D6336C", bakery: "#F08C00", bar: "#5F3DC4", food_court: "#1971C2", other: "#495057" };
function initMap() {
  if (map) { map.invalidateSize(); return; }
  map = L.map("map").setView([-6.2430, 106.8005], 16);
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", { maxZoom: 19, attribution: "&copy; OpenStreetMap contributors" }).addTo(map);
  S.places.filter((p) => p.precision !== "area").forEach((p) => {
    L.circleMarker([p.lat, p.lon], { radius: 5 + p.buzz.trending_score / 15, color: CAT_COLORS[p.category] || "#495057", fillOpacity: 0.7, weight: 2 })
      .bindPopup(`<b>${esc(p.name)}</b> · ${esc(p.category)}<br>${esc(p.summary)}<br><i>${esc(p.where)} (${PRECISION[p.precision] || ""})</i>`)
      .addTo(map);
  });
}

// ---------- chat tab (bring your own Gemini key) ----------
const CHAT_SYSTEM = `You are Food Scout, a friendly local guide to food and cafes in Blok M, Jakarta.
Reply in the same language as the user's latest message. Always use the tools; never invent places, prices or facts.
Recommend 2-4 places with one line each on why they fit. Be honest about evidence: say "halal (high confidence)" or
"halal not confirmed". trending mode "now" = hot right now, "new" = newly opened or spotted, "favorites" = all-time.
Don't call an old favorite trending. Keep answers short and skimmable.`;
const TOOL_DECLS = [
  { name: "search_places", description: "Semantic search over Blok M food places with optional filters.",
    parameters: { type: "OBJECT", properties: {
      query: { type: "STRING", description: "what the user wants, e.g. 'cozy ramen'" },
      category: { type: "STRING", description: "cafe, restaurant, street_food, dessert, bakery, bar, food_court" },
      halal_only: { type: "BOOLEAN" }, vegetarian_only: { type: "BOOLEAN" },
      max_price: { type: "NUMBER", description: "max IDR per person" } }, required: ["query"] } },
  { name: "get_place", description: "Full details and evidence for one place, by (partial) name.",
    parameters: { type: "OBJECT", properties: { name: { type: "STRING" } }, required: ["name"] } },
  { name: "trending", description: "Buzz views: mode 'now' (last 60 days momentum), 'new' (just opened / newly spotted), 'favorites' (all-time).",
    parameters: { type: "OBJECT", properties: { mode: { type: "STRING", enum: ["now", "new", "favorites"] }, category: { type: "STRING" } } } },
  { name: "work_spots", description: "Best places to work or study (wifi, outlets, quiet, long stay).",
    parameters: { type: "OBJECT", properties: { halal_only: { type: "BOOLEAN" } } } },
  { name: "group_pick", description: "Pick a place for a group; each member has wishes and hard rules.",
    parameters: { type: "OBJECT", properties: { members: { type: "ARRAY", items: { type: "OBJECT", properties: {
      name: { type: "STRING" }, wants: { type: "STRING" }, halal: { type: "BOOLEAN" }, vegetarian: { type: "BOOLEAN" },
      max_price: { type: "NUMBER" } } } } }, required: ["members"] } },
];
function brief(p, extra = {}) {
  return { name: p.name, category: p.category, cuisine: p.cuisine, summary: p.summary, where: p.where,
    location_precision: p.precision, price_idr: p.price,
    halal: p.signals.halal.value === "unknown" ? "not confirmed" : `${p.signals.halal.value} (${p.signals.halal.confidence})`,
    trending_score: p.buzz.trending_score, new: p.new?.kind || null, work_score: p.wfc,
    map: `https://www.openstreetmap.org/?mlat=${p.lat}&mlon=${p.lon}`, ...extra };
}
async function runTool(name, args) {
  if (name === "search_places") return (await searchPlaces({ ...args, k: 5 })).map((r) => brief(r.place));
  if (name === "get_place") {
    const p = getPlace(args.name);
    return p ? brief(p, { signals: p.signals, latest_videos: p.videos, new_evidence: p.new?.evidence || [] }) : { status: "not_found" };
  }
  if (name === "trending") return trending({ mode: args.mode, k: 6, category: args.category }).map((r) => brief(r.place, { buzz: r.place.buzz }));
  if (name === "work_spots") return workSpots({ k: 5, halal_only: args.halal_only }).map((r) => brief(r.place,
    { wifi: r.place.signals.wifi.value, outlets: r.place.signals.outlets.value }));
  if (name === "group_pick") {
    const { pool, results } = await groupPick(args.members || [], 3);
    return { pool_size: pool, results: results.map((r) => brief(r.place, { group_fit: r.group_fit, fit_per_member: r.per })) };
  }
  return { error: "unknown tool" };
}
const chatContents = [];
function mdToHtml(text) {
  return esc(text).replace(/\*\*(.+?)\*\*/g, "<b>$1</b>")
    .replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
}
function addMsg(role, html) {
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  div.innerHTML = html;
  $("#chat-log").appendChild(div);
  div.scrollIntoView({ block: "nearest" });
  return div;
}
async function gemini(key) {
  const r = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/${CHAT_MODEL}:generateContent`, {
    method: "POST", headers: { "Content-Type": "application/json", "x-goog-api-key": key },
    body: JSON.stringify({ systemInstruction: { parts: [{ text: CHAT_SYSTEM }] }, contents: chatContents,
      tools: [{ functionDeclarations: TOOL_DECLS }], generationConfig: { temperature: 0.4 } }),
  });
  const data = await r.json();
  if (!r.ok) throw new Error(data.error?.message || `HTTP ${r.status}`);
  return data.candidates?.[0]?.content || { role: "model", parts: [{ text: "(no answer)" }] };
}
$("#chat-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const key = $("#api-key").value.trim();
  const text = $("#chat-input").value.trim();
  if (!text) return;
  if (!key) { addMsg("bot", "Paste a Gemini API key above first. It's free from Google AI Studio."); return; }
  store("fs-gemini-key", $("#remember-key").checked ? key : null);
  $("#chat-input").value = "";
  addMsg("user", esc(text));
  const pending = addMsg("bot", "Scouting Blok M…");
  chatContents.push({ role: "user", parts: [{ text }] });
  const used = [];
  try {
    for (let step = 0; step < 6; step++) {
      const content = await gemini(key);
      chatContents.push(content);
      const calls = (content.parts || []).filter((p) => p.functionCall);
      if (!calls.length) {
        const answer = (content.parts || []).map((p) => p.text || "").join("").trim();
        pending.innerHTML = mdToHtml(answer) + (used.length ? `<span class="tools">tools: ${esc([...new Set(used)].join(", "))}</span>` : "");
        return;
      }
      const responses = [];
      for (const c of calls) {
        used.push(c.functionCall.name);
        let result;
        try { result = await runTool(c.functionCall.name, c.functionCall.args || {}); } catch (err) { result = { error: err.message }; }
        responses.push({ functionResponse: { name: c.functionCall.name, response: { result } } });
      }
      chatContents.push({ role: "user", parts: responses });
    }
    pending.textContent = "I couldn't finish that one. Could you rephrase it?";
  } catch (err) {
    chatContents.pop();
    pending.textContent = "Gemini error: " + err.message;
  }
});

// ---------- tabs & boot ----------
function showTab(name) {
  document.querySelectorAll(".tabs button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.tab === name)));
  document.querySelectorAll(".panel").forEach((p) => { p.hidden = p.id !== `tab-${name}`; });
  if (name === "map") initMap();
  if (location.hash !== `#${name}`) history.replaceState(null, "", `#${name}`);
}
document.querySelectorAll(".tabs button").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));
document.querySelectorAll(".subtabs button").forEach((b) => b.addEventListener("click", () => renderBuzz(b.dataset.mode)));
$("#search-form").addEventListener("submit", (e) => { e.preventDefault(); runSearch($("#q").value); });
$("#work-halal").addEventListener("change", renderWork);
$("#search-examples").innerHTML = EXAMPLES.map((q) => `<button type="button">${esc(q)}</button>`).join("");
$("#search-examples").addEventListener("click", (e) => { if (e.target.tagName === "BUTTON") { $("#q").value = e.target.textContent; runSearch(e.target.textContent); } });

async function boot() {
  const [places, vectors] = await Promise.all([fetch("data/places.json").then((r) => r.json()), fetch("data/vectors.json").then((r) => r.json())]);
  S.places = places.places;
  S.built = places.built;
  S.places.forEach((p) => S.byId.set(p.id, p));
  S.text = decodeVectors(vectors.text.data, vectors.text.dim);
  S.photo = decodeVectors(vectors.photo.data, vectors.photo.dim);
  S.photoItems = vectors.photo.items;
  $("#built-on").textContent = `Data built ${S.built} · ${S.places.length} places · ${S.photoItems.length} photos.`;
  renderBuzz("now");
  renderWork();
  DEFAULT_GROUP.forEach(addGroupRow);
  const saved = recall("fs-gemini-key");
  if (saved) { $("#api-key").value = saved; $("#remember-key").checked = true; }
  $("#search-status").textContent = `${S.places.length} places in Blok M. Search runs in your browser.`;
  const tab = location.hash.slice(1);
  showTab(document.querySelector(`.tabs button[data-tab="${tab}"]`) ? tab : "search");
}
boot().catch((err) => { $("#search-status").textContent = "Couldn't load the data: " + err.message; });

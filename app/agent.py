"""Food Scout agent: a ReAct loop over the MCP tools (capstone M4L3 pattern).

The MCP server runs in-process (FastMCP in-memory transport), so models load once,
but tools are still discovered and called through the MCP protocol.
LLM: Gemini 3.5 Flash Lite, falling back to Groq Qwen; if both fail, a plain search answer."""
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from fastmcp import Client
from openai import OpenAI

from server import mcp

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

LLMS = [
    ("gemini", "https://generativelanguage.googleapis.com/v1beta/openai/", "GEMINI_API_KEY", "gemini-3.5-flash-lite"),
    ("groq", "https://api.groq.com/openai/v1", "GROQ_API_KEY", "qwen/qwen3.8-27b"),
]
MAX_STEPS = 6

SYSTEM_PROMPT = """You are Food Scout, a friendly local guide to food and cafes in Blok M, Jakarta.
Reply in the same language as the user's LATEST message (English question -> English answer,
Indonesian question -> Indonesian answer), even if the data is in another language.
Always use the tools; never invent places, prices, or facts:
- search_places: general requests ("cozy ramen", "cheap padang"), with optional halal/vegetarian/budget/category filters
- get_place: details and the evidence for one place (use it when asked "is X halal?", "does X have wifi?")
- work_spots: places to work or study (WFC, nugas, wifi, colokan)
- trending: mode="now" for what is viral/hot right now, mode="new" for newly opened or newly spotted places,
  mode="favorites" for all-time favorites. Don't call an old favorite "trending"; say which view you used.
- group_pick: choosing one place for a group with different wishes or restrictions
- vibe_match_photo: only when the user has uploaded a photo (you will be given its path)
How to answer:
- Recommend 2-4 places, each with one line on why it fits, plus its map link.
- Be honest about evidence: say "halal (high confidence, OSM tag)" or "halal not confirmed".
  If a signal is "unknown", say it is not confirmed rather than guessing.
- If map_is_approximate is true, say where it roughly is (e.g. "inside Blok M Square", "on Jl. Melawai")
  and call the link an approximate map.
- Keep it short and skimmable."""


def _clients():
    for name, base, key_env, model in LLMS:
        if os.environ.get(key_env):
            yield name, OpenAI(base_url=base, api_key=os.environ[key_env]), model


def _complete(messages, tools):
    last = None
    for name, client, model in _clients():
        try:
            return client.chat.completions.create(model=model, messages=messages, tools=tools, temperature=0.4), name
        except Exception as e:  # quota, outage, unsupported feature
            last = e
    raise RuntimeError(f"All LLMs failed: {last}")


async def _fallback(client: Client, question: str) -> str:
    """No LLM available: answer with a plain semantic search so the demo never dead-ends."""
    res = await client.call_tool("search_places", {"query": question, "k": 4})
    data = json.loads(res.content[0].text)
    lines = ["_(The AI guide is resting, so here are the closest search matches.)_", ""]
    for p in data["results"]:
        lines.append(f"- **{p['name']}** ({p['category']}): {p['summary']} [map]({p['map']})")
    return "\n".join(lines)


async def ask(question: str, history: list[dict] | None = None) -> tuple[str, list[str]]:
    """Run the ReAct loop. Returns (answer, names of tools called)."""
    async with Client(mcp) as client:
        tools = [{"type": "function", "function": {"name": t.name, "description": t.description or "",
                                                   "parameters": t.inputSchema}}
                 for t in await client.list_tools()]
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for m in history or []:
            if m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str) and m["content"]:
                messages.append({"role": m["role"], "content": m["content"]})
        messages.append({"role": "user", "content": question})

        called = []
        for _ in range(MAX_STEPS):
            try:
                resp, _provider = _complete(messages, tools)
            except RuntimeError:
                return await _fallback(client, question), called
            msg = resp.choices[0].message
            if not msg.tool_calls:
                return msg.content or "", called
            messages.append({"role": "assistant", "content": msg.content or "",
                             "tool_calls": [tc.model_dump() for tc in msg.tool_calls]})
            for tc in msg.tool_calls:
                called.append(tc.function.name)
                try:
                    args = json.loads(tc.function.arguments or "{}")
                    result = await client.call_tool(tc.function.name, args)
                    output = " ".join(getattr(c, "text", str(c)) for c in result.content)
                except Exception as e:
                    output = json.dumps({"error": str(e)[:300]})
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": output[:12000]})
        return "I couldn't finish that one. Could you rephrase it?", called


async def call_tool(name: str, args: dict) -> dict:
    """Direct tool call (used by the UI's photo / group / trending tabs)."""
    async with Client(mcp) as client:
        res = await client.call_tool(name, args)
        return json.loads(res.content[0].text)


def explain_group_pick(members: list[dict], result: dict) -> str:
    """One short LLM call to explain the group decision's trade-offs; plain text if no LLM."""
    prompt = ("Explain in 3-5 short bullet points, in the same language as the members' wishes, why the top place "
              "is the best compromise for this group: what it offers each person, and where someone's wish is only "
              "partly met (fit_per_member near 0 = weak match, near 1 = strong). Be warm and concrete. "
              "Use only this data.\n\n"
              f"Members: {json.dumps(members, ensure_ascii=False)}\n\nResult: {json.dumps(result, ensure_ascii=False)[:6000]}")
    for _, client, model in _clients():
        try:
            r = client.chat.completions.create(model=model, temperature=0.4,
                                               messages=[{"role": "user", "content": prompt}])
            return r.choices[0].message.content or ""
        except Exception:
            continue
    top = result["results"][0]
    return f"**{top['name']}** has the best fit for the least-happy member ({top['group_fit']})."

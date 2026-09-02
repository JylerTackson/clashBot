"""Shared helpers for the Clash Royale knowledge-base scraper.

All network access goes through the Fandom MediaWiki API (`api.php`); the
rendered wiki pages themselves sit behind a Cloudflare challenge, but the API
and the static image CDN do not.
"""
from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

WIKI = "https://clashroyale.fandom.com"
API = f"{WIKI}/api.php"

REPO_ROOT = Path(__file__).resolve().parents[2]
KB = Path(os.environ.get("KB_ROOT", REPO_ROOT / "knowledge_base"))  # tests redirect this
CARDS_DIR = KB / "cards"
CARD_IMG_DIR = CARDS_DIR / "images"
EVO_DIR = KB / "evolutions"
HERO_DIR = KB / "heroes"
HERO_IMG_DIR = HERO_DIR / "images"
META_DIR = KB / "meta"
MANIFEST = META_DIR / "scrape_manifest.json"
INDEX_MD = META_DIR / "card_index.md"
INDEX_JSON = META_DIR / "card_index.json"
HERO_INDEX_JSON = META_DIR / "hero_index.json"

# Page cache (wikitext + rendered html per page). Lives outside the repo by
# default because it is ~20MB of HTML; override with KB_CACHE_DIR.
CACHE_DIR = Path(os.environ.get("KB_CACHE_DIR", REPO_ROOT / ".kb_cache"))

SESSION = requests.Session()
SESSION.headers["User-Agent"] = (
    "Mozilla/5.0 (X11; Linux x86_64) clashbot-kb/0.1 (knowledge-base scraper)"
)


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def api(retries: int = 4, **params):
    """Call api.php with retries/backoff. Returns parsed JSON."""
    params["format"] = "json"
    delay = 2.0
    last = None
    for _ in range(retries):
        try:
            r = SESSION.get(API, params=params, timeout=90)
            if r.status_code == 200:
                return r.json()
            last = RuntimeError(f"HTTP {r.status_code} for {params}")
        except requests.RequestException as e:  # network hiccup
            last = e
        time.sleep(delay)
        delay *= 2
    raise last  # type: ignore[misc]


def slugify(title: str) -> str:
    """'Mega Knight' -> 'mega-knight'; 'P.E.K.K.A.' -> 'p-e-k-k-a';
    'Knight/Hero' -> 'knight-hero'."""
    s = title.lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-")


def page_url(title: str) -> str:
    return f"{WIKI}/wiki/{title.replace(' ', '_')}"


def page_exists(titles: list[str]) -> dict[str, bool]:
    out: dict[str, bool] = {}
    for i in range(0, len(titles), 50):
        chunk = titles[i : i + 50]
        r = api(action="query", titles="|".join(chunk), prop="info")
        norm = {n["from"]: n["to"] for n in r["query"].get("normalized", [])}
        found = {
            p["title"]: ("missing" not in p) for p in r["query"]["pages"].values()
        }
        for t in chunk:
            out[t] = found.get(norm.get(t, t), False)
    return out


def load_json(path: Path, default):
    if path.exists():
        return json.loads(path.read_text())
    return default


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def load_manifest() -> dict:
    return load_json(MANIFEST, {"generated_at": None, "items": {}})


def save_manifest(m: dict) -> None:
    m["generated_at"] = now_iso()
    save_json(MANIFEST, m)


# --- wikitext -> plain-ish markdown -----------------------------------------

_ICON_NAMES = {
    "Elixir": "Elixir",
    "Hero Shard": "Hero Shards",
    "Gold": "Gold",
    "Gem": "Gems",
    "XP": "XP",
}


def _strip_templates(text: str) -> str:
    """Remove/replace {{...}} templates (handles nesting) with useful text."""
    out = []
    i = 0
    n = len(text)
    while i < n:
        if text.startswith("{{", i):
            depth = 0
            j = i
            while j < n:
                if text.startswith("{{", j):
                    depth += 1
                    j += 2
                elif text.startswith("}}", j):
                    depth -= 1
                    j += 2
                    if depth == 0:
                        break
                else:
                    j += 1
            inner = text[i + 2 : j - 2]
            out.append(_render_template(inner))
            i = j
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def _render_template(inner: str) -> str:
    inner = _strip_templates(inner)  # nested
    parts = [p.strip() for p in inner.split("|")]
    name = parts[0].lower()
    args = parts[1:]
    kw = {}
    pos = []
    for a in args:
        if "=" in a:
            k, v = a.split("=", 1)
            kw[k.strip().lower()] = v.strip()
        else:
            pos.append(a)
    if name == "rarity" and pos:
        return pos[0]
    if name == "icon":
        v = kw.get("i", pos[0] if pos else "")
        return _ICON_NAMES.get(v, v)
    if name == "quote" and pos:
        return f'> "{pos[0]}"'
    if name in ("hero ability", "battle deck with card", "protection", "magicwords",
                "clashofclanslink", "similar", "head image", "subpagenavbox",
                "mergetacticslink", "cardoverviewnav", "statistics",
                "statisticssubheader", "statuseffects", "navigation",
                "displaytitle", "card infobox", "evolved card infobox",
                "heroic card infobox", "statisticsheader"):
        return ""
    if name.startswith("displaytitle"):
        return ""
    if name == "balance":
        return ""
    if name == "dps" and len(pos) >= 2:
        return f"{pos[0]}/{pos[1]}s"
    # unknown: keep positional text if any, else drop
    return pos[0] if pos else ""


def wikitext_to_md(text: str) -> str:
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    text = re.sub(r"<ref[^>]*/>", "", text)
    text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.S)
    text = _strip_templates(text)
    # files / categories
    text = re.sub(r"\[\[(?:File|Image|Category):[^\]]*\]\]", "", text, flags=re.I)
    # links
    text = re.sub(r"\[\[:?Category:[^|\]]+\|([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"\[\[[^|\]]+\|([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"\[https?://\S+ ([^\]]+)\]", r"\1", text)
    # html
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
    text = re.sub(r"</?(?:center|div|span|small|big|sup|sub|u|s|table-progress-tracking)[^>]*>", "", text, flags=re.I)
    text = re.sub(r"<[^>]+>", "", text)
    # bullets: *, **, *** -> nested "-"; numbered lists # -> "1." (before headings,
    # because converted headings start with '#')
    def _bul(m):
        depth = len(m.group(1))
        return "  " * (depth - 1) + "- "
    text = re.sub(r"^(\*+)\s*", _bul, text, flags=re.M)
    text = re.sub(r"^(#+)\s*", lambda m: "  " * (len(m.group(1)) - 1) + "1. ", text, flags=re.M)
    # headings ==X== -> ### X (we place these inside md, so demote)
    text = re.sub(r"^\s*====\s*(.*?)\s*====\s*$", r"##### \1", text, flags=re.M)
    text = re.sub(r"^\s*===\s*(.*?)\s*===\s*$", r"#### \1", text, flags=re.M)
    text = re.sub(r"^\s*==\s*(.*?)\s*==\s*$", r"### \1", text, flags=re.M)
    # bold/italic (after bullets, so a line-leading ''' is not read as a bullet)
    text = text.replace("'''", "**").replace("''", "*")
    text = text.replace("&nbsp;", " ").replace("&zwnj;", "")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_sections(wikitext: str) -> list[tuple[str, str]]:
    """Split wikitext into [(heading, body)], heading '' for the lead."""
    parts = re.split(r"^(==+)\s*(.*?)\s*\1\s*$", wikitext, flags=re.M)
    sections = [("", parts[0])]
    i = 1
    while i < len(parts):
        _eq, heading, body = parts[i], parts[i + 1], parts[i + 2]
        sections.append((heading.strip(), body))
        i += 3
    return sections

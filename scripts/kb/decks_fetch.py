"""Phase 2, steps 0-1: fetch the RoyaleAPI popular-decks page and enumerate decks.

Fetch strategy (in order):
  1. raw HTTP GET; if the response is an app shell / challenge page, then
  2. render with headless Chromium (Playwright, Node) and read the DOM; if that
     still yields a Cloudflare challenge, then
  3. stop and record the blocker in the manifest + qa_report (never fabricate).

`--html FILE` skips fetching and parses a page saved from a normal browser
session (File > Save Page As, "Webpage, Complete" or the DOM from devtools),
which is the intended fallback when the live site blocks automation.

Output: knowledge_base/meta/deck_index.json  (raw enumeration, no
classification yet) + Step 0 findings in scrape_manifest.json["phase2"].
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from common import INDEX_JSON, META_DIR, SESSION, load_json, load_manifest, now_iso, save_json, save_manifest, slugify

POPULAR_URL = "https://royaleapi.com/decks/popular?lang=en"
ROBOTS_URL = "https://royaleapi.com/robots.txt"
TERMS_URL = "https://royaleapi.com/terms"
DECK_INDEX_JSON = META_DIR / "deck_index.json"
SCRATCH = Path(os.environ.get("KB_SCRATCH", Path(__file__).resolve().parent / ".phase2_scratch"))
RENDER_JS = Path(__file__).resolve().parent / "render_page.js"

BROWSER_UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/124.0.0.0 Safari/537.36")


# --------------------------------------------------------------------------
# step 0: policy check + fetch
# --------------------------------------------------------------------------

def is_challenge(html: str) -> bool:
    return bool(re.search(r"<title>\s*Just a moment|cf_chl_opt|challenge-platform|Verify you are human",
                          html, re.I))


def check_policy() -> dict:
    """Record what robots.txt and the terms page say. Never blocks the run."""
    out = {"checked_at": now_iso(), "robots_url": ROBOTS_URL, "terms_url": TERMS_URL}
    try:
        r = SESSION.get(ROBOTS_URL, headers={"User-Agent": BROWSER_UA}, timeout=60)
        out["robots_status"] = r.status_code
        txt = r.text if r.status_code == 200 else ""
        out["robots_excerpt"] = "\n".join(
            l for l in txt.splitlines()
            if re.match(r"^(User-agent|Disallow|Allow|Crawl-delay|Content-Signal)", l, re.I)
            and re.search(r"\*|Claude|Content-Signal|decks", l, re.I)
        )[:2000]
        # summarise the rules that apply to us
        blocks = re.split(r"\n\s*\n", txt)
        applies = []
        for b in blocks:
            agents = re.findall(r"^User-agent:\s*(.+)$", b, re.M | re.I)
            if any(a.strip() in ("*", "ClaudeBot") for a in agents):
                applies.append(b.strip())
        out["robots_rules_for_star_and_claudebot"] = applies
        path = "/decks/popular"
        disallowed = any(re.search(r"^Disallow:\s*/\s*$", b, re.M | re.I) for b in applies
                         if re.search(r"User-agent:\s*ClaudeBot", b, re.I))
        out["decks_popular_disallowed_for_claudebot"] = disallowed
        out["decks_popular_disallowed_for_star"] = any(
            re.search(rf"^Disallow:\s*{re.escape(path)}", b, re.M | re.I) or re.search(r"^Disallow:\s*/\s*$", b, re.M | re.I)
            for b in applies if re.search(r"User-agent:\s*\*", b))
        m = re.search(r"Content-Signal:\s*(.+)", txt)
        out["content_signal"] = m.group(1).strip() if m else None
        cd = [re.search(r"Crawl-delay:\s*(\d+)", b) for b in applies]
        out["crawl_delay_seconds"] = max([int(x.group(1)) for x in cd if x] or [0])
    except Exception as e:  # noqa: BLE001
        out["robots_error"] = str(e)
    try:
        r = SESSION.get(TERMS_URL, headers={"User-Agent": BROWSER_UA}, timeout=60)
        out["terms_status"] = r.status_code
        out["terms_readable"] = r.status_code == 200 and not is_challenge(r.text)
        if out["terms_readable"]:
            soup = BeautifulSoup(r.text, "lxml")
            text = soup.get_text(" ", strip=True)
            hits = [m.group(0) for m in re.finditer(r".{0,120}(scrap|crawl|automat|bot)[^.]{0,160}\.", text, re.I)]
            out["terms_scraping_excerpts"] = hits[:8]
        else:
            out["terms_note"] = "terms page returned a Cloudflare challenge; could not be read"
    except Exception as e:  # noqa: BLE001
        out["terms_error"] = str(e)
    return out


def fetch_raw(url: str) -> tuple[int, str]:
    r = SESSION.get(url, headers={"User-Agent": BROWSER_UA}, timeout=90)
    return r.status_code, r.text


def render_with_browser(url: str, out_html: Path) -> tuple[bool, str]:
    """Headless Chromium via the globally installed Node Playwright. Returns
    (ok, note). The TLS 1.2 cap and full-chromium channel are needed for the
    session's egress proxy (its TLS terminator drops Chromium's default hello)."""
    out_html.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, NODE_EXTRA_CA_CERTS="/root/.ccr/ca-bundle.crt")
    try:
        p = subprocess.run(["node", str(RENDER_JS), url, str(out_html)], env=env,
                           capture_output=True, text=True, timeout=240)
    except FileNotFoundError:
        return False, "node/playwright not available"
    except subprocess.TimeoutExpired:
        return False, "browser render timed out"
    log = (p.stdout + p.stderr)[-1500:]
    if p.returncode != 0 or not out_html.exists():
        return False, f"render failed: {log}"
    return True, log


# --------------------------------------------------------------------------
# step 1: parse decks out of the (rendered) DOM
# --------------------------------------------------------------------------

STAT_WORDS = ("rating", "usage", "wins", "draws", "losses", "win rate", "win%", "avg elixir", "elixir")


def _card_lookup() -> tuple[dict[str, str], dict[str, str]]:
    idx = load_json(INDEX_JSON, {"cards": []})
    by_norm: dict[str, str] = {}
    by_slug: dict[str, str] = {}
    for c in idx["cards"]:
        by_norm[re.sub(r"[^a-z0-9]", "", c["name"].lower())] = c["slug"]
        by_slug[c["slug"]] = c["slug"]
        by_slug[c["slug"].replace("-", "")] = c["slug"]
    # RoyaleAPI spellings that differ from the wiki
    aliases = {"pekka": "p-e-k-k-a", "minipekka": "mini-p-e-k-k-a", "log": "the-log",
               "xbow": "x-bow", "skarmy": "skeleton-army", "ebarbs": "elite-barbarians",
               "egiant": "electro-giant", "ewiz": "electro-wizard", "edragon": "electro-dragon",
               "megaknight": "mega-knight", "goblingang": "goblin-gang"}
    for k, v in aliases.items():
        by_norm.setdefault(k, v)
    return by_norm, by_slug


def resolve_card(name_or_href: str, by_norm: dict, by_slug: dict) -> tuple[str | None, bool]:
    """Return (phase1_slug, is_evolution)."""
    s = name_or_href.strip()
    evo = bool(re.search(r"evolv|evolution|-ev\d|\bevo\b", s, re.I))
    s = re.sub(r"(?i)\bevolved\b|\bevolution\b|-ev\d+|\bevo\b", "", s)
    m = re.search(r"/card/([^/?#]+)", s)
    if m:
        key = m.group(1).lower()
        key = re.sub(r"-ev\d+$", "", key)
        if key in by_slug:
            return by_slug[key], evo
        n = re.sub(r"[^a-z0-9]", "", key)
        if n in by_norm:
            return by_norm[n], evo
    n = re.sub(r"[^a-z0-9]", "", s.lower())
    return by_norm.get(n), evo


def parse_popular_decks(html: str) -> tuple[list[dict], dict]:
    """Best-effort DOM parser. RoyaleAPI's markup could not be inspected when
    this was written (Cloudflare challenge), so this looks for the generic
    shape: a container holding 8 card images/links, a nearby heading, and
    label/value stat pairs. It reports diagnostics when it finds nothing."""
    soup = BeautifulSoup(html, "lxml")
    by_norm, by_slug = _card_lookup()
    diag = {"html_bytes": len(html), "challenge_page": is_challenge(html)}
    decks: list[dict] = []
    seen_keys: set[str] = set()

    def card_refs(el) -> list[tuple[str, bool]]:
        refs = []
        for a in el.select("a[href*='/card/']"):
            slug, evo = resolve_card(a.get("href", ""), by_norm, by_slug)
            if not slug:
                img = a.find("img")
                if img is not None:
                    slug, evo2 = resolve_card(img.get("alt") or img.get("title") or "", by_norm, by_slug)
                    evo = evo or evo2
            if slug:
                if not evo:
                    img = a.find("img")
                    if img is not None and re.search(r"evo", " ".join(img.get("class", [])) + (img.get("src") or ""), re.I):
                        evo = True
                refs.append((slug, evo))
        if len(refs) < 8:
            refs = []
            for img in el.select("img[alt]"):
                slug, evo = resolve_card(img.get("alt", ""), by_norm, by_slug)
                if slug:
                    refs.append((slug, evo))
        # de-dup keeping order
        out, seen = [], set()
        for s, e in refs:
            if s not in seen:
                seen.add(s)
                out.append((s, e))
        return out

    # candidate containers: smallest elements that contain exactly 8 distinct cards
    candidates = []
    for el in soup.find_all(True):
        if el.name in ("html", "body", "head", "script", "style"):
            continue
        cls = " ".join(el.get("class", []))
        if not re.search(r"deck", cls + " " + (el.get("id") or ""), re.I):
            continue
        refs = card_refs(el)
        if len(refs) == 8:
            candidates.append((el, refs))
    if not candidates:  # class names unknown: fall back to any element with exactly 8 cards
        for el in soup.find_all(["div", "section", "article", "li", "tr"]):
            refs = card_refs(el)
            if len(refs) == 8:
                candidates.append((el, refs))
    diag["candidate_containers"] = len(candidates)

    for el, refs in candidates:
        slugs = sorted(s for s, _ in refs)
        key = "-".join(slugs)
        if key in seen_keys:
            continue
        # prefer the OUTERMOST container with exactly these 8 cards: it holds the
        # deck's name/stats, while an inner wrapper holds only the card images
        outer = [c for c, r in candidates if c is not el and c in el.parents and sorted(s for s, _ in r) == slugs]
        if outer:
            continue
        seen_keys.add(key)
        text = el.get_text(" ", strip=True)
        name = None
        for h in el.select("h1,h2,h3,h4,h5,.header,.deck_name,.deck-name,[class*='name'],[class*='title']"):
            t = h.get_text(" ", strip=True)
            if t and not re.search(r"^\d|rating|usage|wins|losses", t, re.I):
                name = t
                break
        if not name:
            a = el.select_one("a[href*='/decks/stats/'], a[href*='/deck/']")
            if a is not None:
                name = a.get_text(" ", strip=True) or None
        stats = {}
        for lab, val in re.findall(r"(Rating|Usage|Wins|Draws|Losses|Win\s*Rate|Avg\.?\s*Elixir|Elixir)\s*:?\s*([\d][\d.,]*\s*%?)",
                                   text, re.I):
            k = lab.lower().replace(".", "").replace(" ", "_")
            stats.setdefault(k, val.strip())
        # data attributes some sites use
        for attr, v in el.attrs.items():
            if attr.startswith("data-") and re.search(r"rating|usage|win|loss|draw|elixir|name|archetype", attr):
                stats.setdefault(attr[5:].replace("-", "_"), v)
        label = None
        for t in el.select("a[href*='archetype'], a[href*='/decks/'][href*='type'], [class*='archetype'], [class*='tag'], .label"):
            tt = t.get_text(" ", strip=True)
            if tt and len(tt) < 40 and not re.search(r"^\d", tt):
                label = tt
                break
        link = el.select_one("a[href*='/decks/stats/'], a[href*='/deck/']")
        decks.append({
            "deck_key": key,
            "display_name": name or "",
            "cards": [s for s, _ in refs],
            "evolutions": [s for s, e in refs if e],
            "site_stats_raw": stats,
            "site_label": label,
            "site_deck_url": ("https://royaleapi.com" + link["href"]) if link is not None and link.get("href", "").startswith("/") else (link.get("href") if link is not None else None),
        })
    diag["decks_found"] = len(decks)
    if not decks:
        diag["sample_classes"] = sorted({" ".join(e.get("class", [])) for e in soup.find_all(True) if e.get("class")})[:60]
        diag["card_links"] = len(soup.select("a[href*='/card/']"))
    # infinite scroll / pagination hints
    diag["more_available_hint"] = bool(re.search(r"load more|show more|infinite|next page|page=2", html, re.I))
    return decks, diag


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=POPULAR_URL)
    ap.add_argument("--html", help="parse a locally saved copy of the page instead of fetching")
    ap.add_argument("--skip-policy-check", action="store_true")
    args = ap.parse_args()

    manifest = load_manifest()
    p2 = manifest.setdefault("phase2", {})
    p2["source_url"] = args.url

    if not args.skip_policy_check:
        p2["policy_check"] = check_policy()
        print("policy check:", json.dumps({k: v for k, v in p2["policy_check"].items()
                                           if k in ("robots_status", "content_signal", "crawl_delay_seconds",
                                                    "decks_popular_disallowed_for_star",
                                                    "decks_popular_disallowed_for_claudebot", "terms_readable")}))

    html = None
    fetch_log = []
    if args.html:
        html = Path(args.html).read_text(errors="replace")
        fetch_log.append({"method": "local_html", "path": args.html, "bytes": len(html)})
    else:
        status, body = fetch_raw(args.url)
        rec = {"method": "raw_http", "status": status, "bytes": len(body), "challenge": is_challenge(body)}
        fetch_log.append(rec)
        print("raw fetch:", rec)
        if status == 200 and not is_challenge(body) and "/card/" in body:
            html = body
        else:
            out = SCRATCH / "popular_rendered.html"
            ok, note = render_with_browser(args.url, out)
            body = out.read_text(errors="replace") if ok and out.exists() else ""
            rec = {"method": "headless_chromium", "ok": ok, "bytes": len(body),
                   "challenge": is_challenge(body) if body else None, "note": note[-400:]}
            fetch_log.append(rec)
            print("browser render:", {k: v for k, v in rec.items() if k != "note"})
            if ok and body and not is_challenge(body):
                html = body
    p2["fetch_log"] = fetch_log
    p2["fetched_at"] = now_iso()

    if html is None:
        p2["status"] = "blocked"
        p2["blocker"] = ("Both the raw HTTP fetch and a headless Chromium render of the popular-decks page "
                         "returned a Cloudflare managed challenge (interactive 'Verify you are human' Turnstile). "
                         "Not bypassed. Re-run with --html <saved page> from a normal browser session.")
        save_manifest(manifest)
        print("BLOCKED:", p2["blocker"])
        return 3

    decks, diag = parse_popular_decks(html)
    p2["parse_diagnostics"] = diag
    if not decks:
        p2["status"] = "blocked"
        p2["blocker"] = "Page fetched but no deck containers with 8 recognised cards were found; see parse_diagnostics."
        save_manifest(manifest)
        print("NO DECKS PARSED:", json.dumps(diag, indent=1)[:2000])
        return 4

    p2["status"] = "enumerated"
    p2["deck_count"] = len(decks)
    p2["more_available_hint"] = diag.get("more_available_hint")
    items = manifest["items"]
    for d in decks:
        items.setdefault("deck:" + d["deck_key"], {"kind": "deck", "title": d["display_name"], "url": args.url,
                                                   "status": "pending", "stage": "enumerated", "reason": None,
                                                   "updated_at": now_iso()})
    save_manifest(manifest)
    save_json(DECK_INDEX_JSON, {"generated_at": now_iso(), "source_url": args.url,
                                "deck_count": len(decks), "diagnostics": diag, "decks": decks})
    print(f"enumerated {len(decks)} decks -> {DECK_INDEX_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

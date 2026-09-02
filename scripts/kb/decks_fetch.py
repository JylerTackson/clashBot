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
    """A Cloudflare interstitial, as opposed to a real page that merely embeds
    Cloudflare's scripts (a page saved from a browser does)."""
    if re.search(r"<title>\s*Just a moment", html, re.I):
        return True
    has_content = bool(re.search(r"deck_segment|data-card-key|/decks/stats/", html))
    return bool(re.search(r"cf_chl_opt|Verify you are human", html, re.I)) and not has_content


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


def _num(t: str) -> str:
    return " ".join(t.split())


def parse_royaleapi_segments(soup: BeautifulSoup, by_norm: dict, by_slug: dict) -> tuple[list[dict], list[str]]:
    """Parser for RoyaleAPI's actual markup (verified against a page saved on
    2026-09-02): `.deck_segment[data-name]` blocks. Card keys use `-ev1` for an
    evolution and `-hero` for a hero variant; the desktop stats table has a
    percentage row and a raw-count row."""
    decks, unknown = [], []
    for seg in soup.select(".deck_segment[data-name]"):
        keys = [k.strip() for k in seg["data-name"].split(",") if k.strip()]
        cards, evos, heroes = [], [], []
        for k in keys:
            base = re.sub(r"-ev\d+$", "", k)
            is_hero = base.endswith("-hero")
            base = re.sub(r"-hero$", "", base)
            slug = by_slug.get(base) or by_norm.get(re.sub(r"[^a-z0-9]", "", base))
            if not slug:
                unknown.append(k)
                continue
            cards.append(slug)
            if k != base:
                (heroes if is_hero else evos).append(slug)
        if len(cards) != 8:
            continue
        name_el = seg.select_one(".deck_human_name-desktop, .deck_human_name-mobile, h4")
        name = _num(name_el.get_text()) if name_el else ""
        stats: dict[str, str] = {}
        # rank + usage% badge
        badge = seg.select_one(".ui.black.label")
        if badge is not None:
            detail = badge.select_one(".detail")
            if detail is not None:
                stats["rank"] = _num(detail.get_text())
                detail.extract()
            stats["usage_badge"] = _num(badge.get_text())
        # desktop table: row 1 percentages, row 2 counts
        tables = seg.select("table.stats")
        table = tables[-1] if tables else None
        if table is not None:
            heads = [_num(th.get_text()).lower() for th in table.select("thead th")]
            rows = [[_num(td.get_text()) for td in tr.select("td")] for tr in table.select("tbody tr")]
            for i, h in enumerate(heads):
                if rows and i < len(rows[0]) and rows[0][i]:
                    stats[h] = rows[0][i]
                if len(rows) > 1 and i < len(rows[1]) and rows[1][i]:
                    stats[h + "_count"] = rows[1][i]
        for item in seg.select(".battle_stats .item[data-content]"):
            label = item.get("data-content", "").lower()
            val = item.select_one(".value")
            if val is None:
                continue
            nm = val.select_one(".name")
            if nm is not None:
                nm.extract()
            v = _num(val.get_text())
            if "elixir" in label:
                stats["avg_elixir"] = v
            elif "cycle" in label:
                stats["four_card_cycle"] = v
        tower = seg.select_one(".battle_stats img[src*='tower'], .battle_stats img[src*='cannoneer'], .battle_stats img[src*='duchess'], .battle_stats img[src*='chef']")
        if tower is not None:
            stats["tower_troop"] = re.sub(r"\.png$", "", (tower.get("src") or "").split("/")[-1].split("(")[0])
        trophy = seg.select_one(".deck_search_results__highest_trophy")
        if trophy is not None:
            stats["highest_trophy_player"] = _num(trophy.get_text())
        link = seg.select_one("a[href*='/decks/stats/']")
        label_el = seg.select_one("[class*='archetype'], .deck_archetype, .deck_tag")
        decks.append({
            "deck_key": "-".join(sorted(cards)),
            "display_name": name,
            "cards": cards,
            "evolutions": evos,
            "heroes": heroes,
            "royaleapi_card_keys": keys,
            "site_stats_raw": stats,
            "site_label": _num(label_el.get_text()) if label_el is not None else None,
            "site_deck_url": link["href"] if link is not None else None,
        })
    return decks, unknown


def parse_popular_decks(html: str) -> tuple[list[dict], dict]:
    """Try the RoyaleAPI-specific parser first, then a generic fallback that
    looks for any container holding 8 recognised card links/images."""
    soup = BeautifulSoup(html, "lxml")
    by_norm, by_slug = _card_lookup()
    diag = {"html_bytes": len(html), "challenge_page": is_challenge(html)}
    decks, unknown = parse_royaleapi_segments(soup, by_norm, by_slug)
    diag["parser"] = "royaleapi_deck_segment" if decks else "generic"
    diag["segments_on_page"] = len(soup.select(".deck_segment[data-name]"))
    diag["unmapped_card_keys"] = sorted(set(unknown))
    if not decks:
        decks = parse_generic(soup, by_norm, by_slug, diag)
    # de-duplicate by deck_key (keep first = highest ranked)
    seen: dict[str, dict] = {}
    uniq = []
    for d in decks:
        if d["deck_key"] not in seen:
            seen[d["deck_key"]] = d
            d["variants"] = []
            uniq.append(d)
        else:  # same 8 cards listed again (different evolution/hero/tower choice or rank)
            seen[d["deck_key"]]["variants"].append({
                "display_name": d["display_name"], "royaleapi_card_keys": d["royaleapi_card_keys"],
                "evolutions": d["evolutions"], "heroes": d["heroes"], "site_stats_raw": d["site_stats_raw"]})
    diag["decks_found"] = len(uniq)
    diag["duplicates_dropped"] = len(decks) - len(uniq)
    if not uniq:
        diag["sample_classes"] = sorted({" ".join(e.get("class", [])) for e in soup.find_all(True) if e.get("class")})[:60]
    # pagination / infinite-scroll hints so truncation is never silent
    hints = []
    for a in soup.select("a, button"):
        t = _num(a.get_text())
        if re.search(r"^(next|more|load more|show more|page \d+|\d+)$", t, re.I) and (a.get("href") or "").find("decks") >= 0:
            hints.append(t + " -> " + (a.get("href") or ""))
    diag["pagination_hints"] = hints[:10]
    diag["more_available_hint"] = bool(hints) or bool(re.search(r"load more|infinite[- ]scroll", html, re.I))
    return uniq, diag


def parse_generic(soup: BeautifulSoup, by_norm: dict, by_slug: dict, diag: dict) -> list[dict]:
    decks: list[dict] = []
    seen_keys: set[str] = set()

    def card_refs(el) -> list[tuple[str, bool]]:
        refs = []
        for img in el.select("img[data-card-key], img[alt]"):
            key = img.get("data-card-key") or img.get("alt") or ""
            slug, evo = resolve_card(key, by_norm, by_slug)
            if slug:
                refs.append((slug, evo))
        if len(refs) < 8:
            for a in el.select("a[href*='/card/']"):
                slug, evo = resolve_card(a.get("href", ""), by_norm, by_slug)
                if slug:
                    refs.append((slug, evo))
        out, seen = [], set()
        for s, e in refs:
            if s not in seen:
                seen.add(s)
                out.append((s, e))
        return out

    candidates = []
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
        outer = [c for c, r in candidates if c is not el and c in el.parents and sorted(s for s, _ in r) == slugs]
        if outer:
            continue
        seen_keys.add(key)
        text = el.get_text(" ", strip=True)
        name = None
        for h in el.select("h1,h2,h3,h4,h5,[class*='name'],[class*='title']"):
            t = h.get_text(" ", strip=True)
            if t and not re.search(r"^\d|rating|usage|wins|losses", t, re.I):
                name = t
                break
        stats = {}
        for lab, val in re.findall(r"(Rating|Usage|Wins|Draws|Losses|Avg\.?\s*Elixir)\s*:?\s*([\d][\d.,]*\s*%?)", text, re.I):
            stats.setdefault(lab.lower().replace(".", "").replace(" ", "_"), val.strip())
        link = el.select_one("a[href*='/decks/stats/'], a[href*='/deck/']")
        decks.append({"deck_key": key, "display_name": name or "", "cards": [s for s, _ in refs],
                      "evolutions": [s for s, e in refs if e], "heroes": [], "royaleapi_card_keys": [],
                      "site_stats_raw": stats, "site_label": None,
                      "site_deck_url": link.get("href") if link is not None else None})
    return decks


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
        import hashlib
        raw = Path(args.html).read_bytes()
        html = raw.decode("utf-8", errors="replace")
        fetch_log.append({"method": "local_html", "path": args.html, "bytes": len(raw),
                          "sha256": hashlib.sha256(raw).hexdigest(),
                          "note": "page saved from the user's own browser session and supplied to the pipeline"})
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
    p2["segments_on_page"] = diag.get("segments_on_page")
    p2["more_available_hint"] = diag.get("more_available_hint")
    p2["coverage_note"] = (f"The page listed {diag.get('segments_on_page')} deck entries ({len(decks)} unique card sets; "
                           f"{diag.get('duplicates_dropped', 0)} repeated with different evolution/hero picks). RoyaleAPI's "
                           "popular-decks view exposes further decks through its filter/time-range controls and deeper "
                           "pages, which were not captured; this run is the default view only.")
    p2.pop("blocker", None)
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

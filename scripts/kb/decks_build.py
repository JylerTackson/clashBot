"""Phase 2, steps 2-6: classify decks, write deck + archetype files, cross-link
cards, verify, and append to meta/qa_report.md.

  python3 decks_build.py build      # deck_index.json -> decks/<key>.md (heuristic
                                    # classification, agent placeholders)
  python3 decks_build.py finalize   # decks/*.md -> deck_index.md, archetypes/*.md,
                                    # card cross-links, manifest, QA report

`build` preserves agent-written content in existing deck files (the
"Why this deck works" section, and archetype fields when
classification_source is `agent`), so re-runs update rather than clobber.
`finalize` is idempotent: card cross-link sections are replaced in place
between markers, and the Phase 2 block of qa_report.md is replaced, never
duplicated. Every step works with zero decks (the archetype files are still
written with deck_count: 0).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from common import CARDS_DIR, INDEX_JSON, KB, META_DIR, load_json, load_manifest, now_iso, save_json, save_manifest

DECKS_DIR = KB / "decks"
ARCH_DIR = KB / "archetypes"
DECK_INDEX_JSON = META_DIR / "deck_index.json"
DECK_INDEX_MD = META_DIR / "deck_index.md"
QA_MD = META_DIR / "qa_report.md"
PLACEHOLDER = "<!-- AGENT:FILL -->"
POPULAR_URL = "https://royaleapi.com/decks/popular?lang=en"

ARCHETYPES = ["beatdown", "control", "cycle", "bait", "bridge-spam", "siege"]
ARCH_NAME = {"beatdown": "Beatdown", "control": "Control", "cycle": "Cycle", "bait": "Bait",
             "bridge-spam": "Bridge Spam", "siege": "Siege"}

# Fixed taxonomy text (from the task definitions, expanded into a gameplan).
ARCH_TEXT = {
    "beatdown": {
        "definition": "A heavy tank absorbs damage while support troops clear the defenders; the goal is an unstoppable push in double elixir.",
        "gameplan": [
            "Play the tank (Golem, Giant, Lava Hound, Electro Giant) at the back of the King Tower so elixir regenerates while it walks.",
            "Stack support behind it as the opponent commits: splash (Baby Dragon, Wizard), anti-air and tank-killer answers, and a big spell (Lightning, Fireball) to remove the opponent's key defender.",
            "Accept chip damage on your own towers in single elixir; the deck is built to win the double-elixir and overtime phases when a full push cannot be stopped.",
            "Defend with the cheapest possible cards and never over-invest on defense while a push is building.",
        ],
        "why": [
            "The tank's hitpoints buy time: every second the opponent spends hitting it is a second the support troops deal free damage.",
            "Spending elixir in bulk behind a tank converts elixir into tower damage more efficiently than trading card-for-card.",
            "Double elixir removes the deck's main weakness (being punished on the other lane while building a push).",
        ],
        "common": ["golem", "giant", "lava-hound", "electro-giant", "baby-dragon", "lightning", "night-witch", "lumberjack"],
    },
    "control": {
        "definition": "Defense-first: make positive elixir trades, then convert the surviving defenders into a counter-push.",
        "gameplan": [
            "Answer every push for less elixir than it cost, using buildings, splash and high-value defenders.",
            "Counter-push with whatever survived, usually with a cheap win condition that punishes the opponent's low elixir (Miner, Graveyard, Hog Rider).",
            "Use spells (Poison, Fireball, Rocket) both to defend and to close out games with chip damage.",
            "Keep elixir in reserve; the deck wins by never being out-of-position rather than by out-damaging the opponent.",
        ],
        "why": [
            "Consistent positive trades compound: a few elixir of advantage per exchange becomes an unanswerable counter-push.",
            "Win conditions that bypass defenses (Miner, Graveyard) do not need a full push to deal damage.",
            "The opponent is forced to attack into prepared defenses instead of dictating the tempo.",
        ],
        "common": ["graveyard", "miner", "poison", "bowler", "ice-wizard", "tornado", "valkyrie", "inferno-tower", "electro-wizard", "executioner"],
    },
    "cycle": {
        "definition": "Low average elixir (typically under 3.0) lets you replay the same win condition faster than the opponent can cycle their counter.",
        "gameplan": [
            "Play the win condition (Hog Rider, Royal Hogs, Wall Breakers) every time the opponent's best counter is out of rotation.",
            "Defend with 1-2 elixir cards (Ice Spirit, Skeletons, The Log) and a cheap building (Cannon, Tesla) to keep the cycle short.",
            "Track the opponent's rotation; the deck wins by tempo and by making the opponent's expensive cards awkward.",
            "Prefer chip damage every cycle over one big push.",
        ],
        "why": [
            "A four-card cycle of 1-3 elixir cards means the win condition is back in hand before the opponent's counter is.",
            "Cheap cards make every defensive trade close to neutral, so the opponent can never build an elixir lead.",
            "Fast, repeated pressure punishes any opponent who plays a heavy card at the wrong time.",
        ],
        "common": ["hog-rider", "ice-spirit", "the-log", "skeletons", "ice-golem", "musketeer", "cannon", "fireball", "earthquake"],
    },
    "bait": {
        "definition": "Several cheap threats each demand the same small spell, so whichever one the opponent does not have an answer for connects.",
        "gameplan": [
            "Play the threat that the opponent's current hand cannot answer: Goblin Barrel, Princess, Goblin Gang, Dart Goblin each punish a missing Log/Arrows/Zap.",
            "Defend with cheap swarms and a building (Inferno Tower, Tesla) that also work as bait for the opponent's spells.",
            "Keep a big spell (Rocket, Fireball) to punish the opponent's support and to finish low towers.",
            "Track which spell the opponent has used and cycle to the threat it would have countered.",
        ],
        "why": [
            "Small spells are the natural answer to swarms; when multiple cards demand the same spell, at least one gets through every rotation.",
            "Cheap cards make positive trades easy on defense and let the deck cycle back to its threats quickly.",
            "Opponents who mis-spend a spell take heavy tower damage from a 3-elixir card.",
        ],
        "common": ["goblin-barrel", "princess", "goblin-gang", "dart-goblin", "rocket", "inferno-tower", "knight", "skeleton-army", "guards"],
    },
    "bridge-spam": {
        "definition": "High-threat units dropped at the bridge force immediate, often awkward, reactions and punish slow or expensive plays.",
        "gameplan": [
            "Wait for the opponent to commit elixir (a tank at the back, a heavy spell), then drop fast threats at the bridge: Battle Ram, Bandit, Royal Ghost, Ram Rider.",
            "Chain threats so the opponent has to answer several at once; a defended Battle Ram still leaves Barbarians on the tower.",
            "Use P.E.K.K.A or Mega Knight as the defensive anchor and counter-push with it.",
            "Never build a slow push; the deck wants short, sharp exchanges.",
        ],
        "why": [
            "Deploying at the bridge removes the opponent's time to react; many counters need a placement window that no longer exists.",
            "Each threat demands a different answer (charge, dash, invisibility), so a single defensive card rarely handles the wave.",
            "Punishing back-deploys means heavy decks never get to build their ideal push.",
        ],
        "common": ["p-e-k-k-a", "battle-ram", "bandit", "royal-ghost", "ram-rider", "dark-prince", "electro-wizard", "magic-archer", "royal-recruits"],
    },
    "siege": {
        "definition": "A building placed on your own side (X-Bow, Mortar) hits the enemy tower from range while the rest of the deck protects it.",
        "gameplan": [
            "Place the siege building at the river when the opponent is low on elixir or has no building-targeting answer in hand.",
            "Protect it with cheap troops and a second building (Tesla, Cannon) that pulls attackers away.",
            "When the opponent commits a heavy push, defend with the siege building itself as a distraction and counter-push with cycle cards.",
            "Keep spells to clear the troops the opponent drops on the building.",
        ],
        "why": [
            "Damage from your own side of the arena forces the opponent to cross the river into your defenses to stop it.",
            "Cheap support makes every siege placement a positive trade if the opponent over-commits to killing the building.",
            "Buildings tank for the deck on defense, so siege decks are strong against beatdown and slow win conditions.",
        ],
        "common": ["x-bow", "mortar", "tesla", "cannon", "knight", "archers", "ice-spirit", "the-log", "fireball", "rocket"],
    },
}

# Heuristic card weights per archetype (card slug -> weight). Tanks and
# defining win conditions weigh more than generic support.
WEIGHTS = {
    "beatdown": {"golem": 4, "lava-hound": 4, "giant": 3, "electro-giant": 4, "goblin-giant": 3, "elixir-golem": 3,
                 "royal-giant": 1, "giant-skeleton": 2, "baby-dragon": 1, "lightning": 2, "night-witch": 2,
                 "lumberjack": 1, "balloon": 1, "mega-minion": 1, "witch": 1, "sparky": 2, "rune-giant": 3},
    "control": {"graveyard": 3, "miner": 2, "poison": 2, "bowler": 2, "ice-wizard": 2, "tornado": 1, "valkyrie": 1,
                "inferno-tower": 1, "bomb-tower": 1, "executioner": 1, "electro-wizard": 1, "mother-witch": 1,
                "rocket": 1, "royal-delivery": 1, "phoenix": 1, "mighty-miner": 2, "archer-queen": 1},
    "cycle": {"hog-rider": 3, "royal-hogs": 2, "wall-breakers": 1, "ice-spirit": 1, "skeletons": 1, "the-log": 1,
              "ice-golem": 1, "cannon": 1, "musketeer": 1, "earthquake": 1, "fireball": 0.5, "firecracker": 0.5,
              "mortar": 0, "electro-spirit": 1, "heal-spirit": 1, "fire-spirit": 1, "bats": 0.5},
    "bait": {"goblin-barrel": 4, "princess": 2, "goblin-gang": 2, "dart-goblin": 1.5, "rocket": 1, "inferno-tower": 1,
             "skeleton-army": 1, "guards": 1, "goblin-drill": 2, "wall-breakers": 1, "knight": 0.5, "rascals": 1,
             "spear-goblins": 0.5, "minion-horde": 0.5, "skeleton-barrel": 1.5, "suspicious-bush": 1},
    "bridge-spam": {"battle-ram": 4, "bandit": 3, "royal-ghost": 3, "ram-rider": 3, "p-e-k-k-a": 2, "dark-prince": 2,
                    "prince": 1, "elite-barbarians": 2, "royal-recruits": 1, "magic-archer": 1, "electro-wizard": 0.5,
                    "mega-knight": 1, "golden-knight": 1, "boss-bandit": 2, "goblin-machine": 1},
    "siege": {"x-bow": 6, "mortar": 6, "tesla": 1, "cannon": 0.5, "archers": 0.5, "rocket": 0.5, "knight": 0.25},
}

LABEL_KEYWORDS = [  # (regex on site label / display name, archetype)
    (r"bridge\s*spam|\bspam\b", "bridge-spam"), (r"\bsiege\b|x-?bow|mortar", "siege"),
    (r"\bbait\b", "bait"), (r"beat\s*down|\bgolem\b|lava\s*hound|lavaloon|\bgiant\b|e-?giant", "beatdown"),
    (r"\bcycle\b|2\.\d\b|\bhog\b|royal\s*hogs|wall\s*breakers", "cycle"),
    (r"\bcontrol\b|graveyard|miner|poison", "control"),
]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def parse_frontmatter(text: str) -> dict:
    m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    out = {}
    if not m:
        return out
    for line in m.group(1).split("\n"):
        if ":" in line:
            k, v = line.split(":", 1)
            v = v.strip()
            if len(v) >= 2 and v[0] == v[-1] == '"':
                v = v[1:-1].replace('\\"', '"')
            out[k.strip()] = v
    return out


def fm_line(k: str, v) -> str:
    if isinstance(v, bool):
        return f"{k}: {'true' if v else 'false'}"
    v = "" if v is None else str(v)
    if v == "":
        return f'{k}: ""'
    if re.search(r"[:#\[\]{}&*!|>'\"%@`]|^\s|\s$", v) or v.lower() in ("yes", "no", "null", "true", "false", "n/a", "none"):
        v = '"' + v.replace('"', '\\"') + '"'
    return f"{k}: {v}"


def card_data() -> dict[str, dict]:
    idx = load_json(INDEX_JSON, {"cards": []})
    return {c["slug"]: c for c in idx["cards"]}


def avg_elixir(cards: list[str], cd: dict) -> tuple[float | None, list[str]]:
    costs, missing = [], []
    for s in cards:
        c = cd.get(s)
        try:
            costs.append(float(c["elixir_cost"]))
        except (TypeError, KeyError, ValueError):
            missing.append(s)
    if not costs:
        return None, missing
    return round(sum(costs) / len(costs), 2), missing


def section_body(text: str, heading: str) -> str | None:
    m = re.search(rf"^## {re.escape(heading)}\n(.*?)(?=^## |\Z)", text, re.M | re.S)
    return m.group(1).strip() if m else None


# --------------------------------------------------------------------------
# classification
# --------------------------------------------------------------------------

def map_label(label: str) -> str | None:
    for rx, arch in LABEL_KEYWORDS:
        if re.search(rx, label, re.I):
            return arch
    return None


def classify(deck: dict, cd: dict) -> dict:
    cards = deck["cards"]
    avg, missing = avg_elixir(cards, cd)
    scores = {a: 0.0 for a in ARCHETYPES}
    evidence = {a: [] for a in ARCHETYPES}
    for a in ARCHETYPES:
        for s in cards:
            w = WEIGHTS[a].get(s, 0)
            if w:
                scores[a] += w
                evidence[a].append(s)
    if avg is not None and avg < 3.0:
        scores["cycle"] += 3
        evidence["cycle"].append(f"avg elixir {avg} < 3.0")
    elif avg is not None and avg >= 4.0:
        scores["beatdown"] += 1.5
        evidence["beatdown"].append(f"avg elixir {avg} >= 4.0")

    source, rationale = "heuristic", ""
    primary = secondary = None
    site_label = deck.get("site_label")
    if site_label and map_label(site_label):
        primary = map_label(site_label)
        source = "site_label"
        rationale = f"RoyaleAPI labels this deck '{site_label}', mapped to {ARCH_NAME[primary]}."
    else:
        # hard rules first
        if any(s in ("x-bow", "mortar") for s in cards) and scores["siege"] >= 6:
            primary = "siege"
            rationale = "A siege building (" + ", ".join(s for s in cards if s in ("x-bow", "mortar")) + ") is the win condition."
        elif avg is not None and avg < 3.0 and scores["cycle"] >= 5:
            primary = "cycle"
            rationale = f"Average elixir {avg} is under 3.0 with a fast win condition ({', '.join(e for e in evidence['cycle'] if not e.startswith('avg'))})."
        else:
            ranked = sorted(ARCHETYPES, key=lambda a: -scores[a])
            primary = ranked[0]
            rationale = (f"Card composition scores highest for {ARCH_NAME[primary]} "
                         f"({', '.join(evidence[primary]) or 'no defining cards'}; avg elixir {avg}).")
        # a name hint that disagrees is worth recording
        hint = map_label(deck.get("display_name") or "")
        if hint and hint != primary:
            rationale += f" Display name suggests {ARCH_NAME[hint]}."
            if scores[hint] >= 0.5 * max(scores.values() or [1]):
                secondary = hint
    ranked = sorted(ARCHETYPES, key=lambda a: -scores[a])
    if secondary is None:
        for a in ranked:
            if a != primary and scores[a] >= max(4.0, 0.7 * scores[primary]):
                secondary = a
                rationale += f" It also carries {ARCH_NAME[a]} elements ({', '.join(evidence[a])})."
                break
    return {"archetype_primary": primary, "archetype_secondary": secondary or "none",
            "classification_source": source, "classification_rationale": rationale,
            "avg_elixir": avg, "missing_cost_cards": missing,
            "scores": {a: round(scores[a], 2) for a in ARCHETYPES}}


# --------------------------------------------------------------------------
# build: deck files
# --------------------------------------------------------------------------

def stat_unit(raw: dict) -> str:
    vals = " ".join(str(v) for v in raw.values())
    if "%" in vals:
        return "percentage"
    if re.search(r"#\s*\d|\brank\b", vals, re.I):
        return "rank"
    return "count" if raw else "unknown"


def build(args) -> int:
    idx = load_json(DECK_INDEX_JSON, None)
    if not idx:
        print("no deck_index.json; run decks_fetch.py first (nothing to build)")
        return 0
    cd = card_data()
    manifest = load_manifest()
    items = manifest["items"]
    DECKS_DIR.mkdir(parents=True, exist_ok=True)
    n = 0
    for d in idx["decks"]:
        key = d["deck_key"]
        path = DECKS_DIR / f"{key}.md"
        existing = path.read_text() if path.exists() else ""
        ex_fm = parse_frontmatter(existing)
        cls = classify(d, cd)
        if ex_fm.get("classification_source") == "agent":  # keep agent judgement
            for k in ("archetype_primary", "archetype_secondary", "classification_source", "classification_rationale"):
                cls[k] = ex_fm.get(k, cls[k])
        d.update({k: cls[k] for k in ("archetype_primary", "archetype_secondary", "classification_source",
                                       "classification_rationale", "avg_elixir")})
        d["heuristic_scores"] = cls["scores"]
        raw = d.get("site_stats_raw", {})
        why = section_body(existing, "Why this deck works")
        if not why or PLACEHOLDER in why:
            why = PLACEHOLDER
        fm = [
            fm_line("deck_key", key), fm_line("display_name", d.get("display_name", "")),
            fm_line("archetype_primary", cls["archetype_primary"]),
            fm_line("archetype_secondary", cls["archetype_secondary"]),
            fm_line("classification_source", cls["classification_source"]),
            fm_line("classification_rationale", cls["classification_rationale"]),
            fm_line("avg_elixir", cls["avg_elixir"] if cls["avg_elixir"] is not None else "n/a"),
            fm_line("rating", raw.get("rating", "n/a")), fm_line("usage", raw.get("usage", "n/a")),
            fm_line("wins", raw.get("wins", "n/a")), fm_line("draws", raw.get("draws", "n/a")),
            fm_line("losses", raw.get("losses", "n/a")), fm_line("stat_unit", stat_unit(raw)),
            fm_line("site_avg_elixir", raw.get("avg_elixir", raw.get("elixir", "n/a"))),
            fm_line("evolutions", ", ".join(d.get("evolutions", [])) or "none"),
            fm_line("source_url", idx.get("source_url", POPULAR_URL)),
            fm_line("site_deck_url", d.get("site_deck_url") or ""),
            fm_line("scraped_at", idx.get("generated_at", now_iso())),
        ]
        body = ["---", *fm, "---", "", f"# {d.get('display_name') or key}", ""]
        body += ["## Cards", ""]
        for s in d["cards"]:
            c = cd.get(s, {})
            evo = " (evolution)" if s in d.get("evolutions", []) else ""
            body.append(f"- [{c.get('name', s)}](../cards/{s}.md){evo} — {c.get('elixir_cost', '?')} elixir, "
                        f"{c.get('rarity', '?')} {c.get('card_type', '?')}, targets {c.get('targets', '?')}")
        if cls["missing_cost_cards"]:
            body.append(f"\nCards without Phase 1 cost data: {', '.join(cls['missing_cost_cards'])}")
        body += ["", "## Classification", "",
                 f"- Primary: **{ARCH_NAME[cls['archetype_primary']]}** ([archetype file](../archetypes/{cls['archetype_primary']}.md))",
                 f"- Secondary: {ARCH_NAME.get(cls['archetype_secondary'], 'none')}"
                 + (f" ([archetype file](../archetypes/{cls['archetype_secondary']}.md))" if cls['archetype_secondary'] != 'none' else ""),
                 f"- Source: `{cls['classification_source']}` — {cls['classification_rationale']}",
                 f"- Heuristic scores: " + ", ".join(f"{ARCH_NAME[a]} {cls['scores'][a]}" for a in ARCHETYPES), "",
                 "## Site statistics (as displayed)", ""]
        if raw:
            body += [f"- {k}: {v}" for k, v in raw.items()]
        else:
            body.append("Not captured from source page")
        body += ["", "## Why this deck works", "", why, "",
                 "## Source", "", f"- {idx.get('source_url', POPULAR_URL)} (scraped {idx.get('generated_at', '')})", ""]
        path.write_text("\n".join(body))
        e = items.setdefault("deck:" + key, {"kind": "deck", "title": d.get("display_name", key), "url": idx.get("source_url")})
        e.update({"stage": "built", "status": "pending" if why == PLACEHOLDER else e.get("status", "pending"),
                  "updated_at": now_iso()})
        n += 1
    save_json(DECK_INDEX_JSON, idx)
    save_manifest(manifest)
    print(f"built {n} deck files")
    return 0


# --------------------------------------------------------------------------
# finalize: index, archetypes, cross-links, verify, QA
# --------------------------------------------------------------------------

def load_decks() -> list[dict]:
    decks = []
    for p in sorted(DECKS_DIR.glob("*.md")):
        text = p.read_text()
        fm = parse_frontmatter(text)
        cards = re.findall(r"^- \[[^\]]+\]\(\.\./cards/([^)]+)\.md\)", text, re.M)
        decks.append({"path": p, "fm": fm, "cards": cards, "text": text})
    return decks


def write_deck_index_md(decks: list[dict], manifest: dict, p2: dict) -> None:
    lines = ["# Deck index", "", f"Generated {now_iso()} from {p2.get('source_url', POPULAR_URL)}.", ""]
    st = p2.get("status", "not run")
    lines.append(f"**Decks: {len(decks)}** · scrape status: `{st}`")
    if p2.get("blocker"):
        lines += ["", f"> **Blocker:** {p2['blocker']}"]
    if p2.get("more_available_hint") is not None:
        lines += ["", f"More decks beyond this page: {'hint of pagination/infinite scroll detected' if p2['more_available_hint'] else 'no pagination hint detected'} (not automated)."]
    units = sorted({d["fm"].get("stat_unit", "unknown") for d in decks}) or ["n/a"]
    lines += ["", f"Stat units (rating/usage/W/D/L) as displayed on the page: {', '.join(units)}. "
              "`avg_elixir` is computed from Phase 1 card costs; the site's own value is kept in each deck file as `site_avg_elixir`.", "",
              "| deck_key | display_name | archetype (primary / secondary) | source | rating | usage | W | D | L | avg_elixir | status |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for d in decks:
        f = d["fm"]
        key = f.get("deck_key", d["path"].stem)
        stt = manifest["items"].get("deck:" + key, {}).get("status", "pending")
        lines.append(f"| [`{key}`](../decks/{key}.md) | {f.get('display_name', '')} | "
                     f"{ARCH_NAME.get(f.get('archetype_primary', ''), f.get('archetype_primary', ''))} / {ARCH_NAME.get(f.get('archetype_secondary', ''), f.get('archetype_secondary', 'none'))} | "
                     f"{f.get('classification_source', '')} | {f.get('rating', '')} | {f.get('usage', '')} | {f.get('wins', '')} | "
                     f"{f.get('draws', '')} | {f.get('losses', '')} | {f.get('avg_elixir', '')} | {stt} |")
    if not decks:
        lines.append("| (no decks scraped this run) | | | | | | | | | | |")
    DECK_INDEX_MD.write_text("\n".join(lines) + "\n")


def write_archetypes(decks: list[dict], cd: dict, scraped_at: str) -> None:
    ARCH_DIR.mkdir(parents=True, exist_ok=True)
    for a in ARCHETYPES:
        mine = [d for d in decks if d["fm"].get("archetype_primary") == a]
        sec = [d for d in decks if d["fm"].get("archetype_secondary") == a]
        t = ARCH_TEXT[a]
        counter = Counter(s for d in mine for s in d["cards"])
        recurring = [(s, n) for s, n in counter.most_common() if n >= 2]
        body = ["---", fm_line("archetype", a), fm_line("archetype_name", ARCH_NAME[a]),
                fm_line("deck_count", len(mine)), fm_line("secondary_deck_count", len(sec)),
                fm_line("scraped_at", scraped_at), "---", "",
                f"# {ARCH_NAME[a]}", "", f"> {t['definition']}", "",
                f"Decks classified as {ARCH_NAME[a]} in this run: **{len(mine)}** primary"
                + (f", {len(sec)} secondary" if sec else "") + ".", ""]
        if not mine:
            body += ["No popular decks were classified into this archetype in this run. That is a signal about the current meta "
                     "(or about the scrape being blocked; see meta/deck_index.md), not a gap in the taxonomy. The gameplan below is the general one.", ""]
        body += ["## Gameplan", ""] + [f"- {x}" for x in t["gameplan"]] + [""]
        body += ["## Why it works", ""] + [f"- {x}" for x in t["why"]]
        if mine:
            evos = Counter(e.strip() for d in mine for e in d["fm"].get("evolutions", "none").split(",") if e.strip() and e.strip() != "none")
            avgs = [float(d["fm"]["avg_elixir"]) for d in mine if re.match(r"^[\d.]+$", d["fm"].get("avg_elixir", ""))]
            if avgs:
                body.append(f"- Across this run's {len(mine)} deck(s) the average elixir cost is {round(sum(avgs)/len(avgs), 2)} "
                            f"(min {min(avgs)}, max {max(avgs)}).")
            if evos:
                body.append("- Evolutions used: " + ", ".join(f"{cd.get(s, {}).get('name', s)} ({n})" for s, n in evos.most_common()))
        body += ["", "## Cards that recur across these decks", ""]
        if recurring:
            body += [f"- [{cd.get(s, {}).get('name', s)}](../cards/{s}.md) — {n} of {len(mine)} decks" for s, n in recurring]
        elif mine:
            body.append("No card appears in more than one deck of this archetype in this run.")
        else:
            body += ["No decks this run. Cards that typically define the archetype: "
                     + ", ".join(f"[{cd.get(s, {}).get('name', s)}](../cards/{s}.md)" for s in t["common"] if s in cd) + "."]
        body += ["", "## Example decks", ""]
        if mine:
            body += [f"- [{d['fm'].get('display_name') or d['fm'].get('deck_key')}](../decks/{d['fm'].get('deck_key', d['path'].stem)}.md)"
                     f" — avg elixir {d['fm'].get('avg_elixir', '?')}, classification `{d['fm'].get('classification_source', '')}`" for d in mine]
        else:
            body.append("None this run.")
        if sec:
            body += ["", "### Decks with this as a secondary archetype", ""]
            body += [f"- [{d['fm'].get('display_name') or d['fm'].get('deck_key')}](../decks/{d['fm'].get('deck_key', d['path'].stem)}.md)"
                     f" (primary: {ARCH_NAME.get(d['fm'].get('archetype_primary', ''), '?')})" for d in sec]
        body.append("")
        (ARCH_DIR / f"{a}.md").write_text("\n".join(body))


START = "<!-- deck-archetypes:start -->"
END = "<!-- deck-archetypes:end -->"


def crosslink_cards(decks: list[dict], cd: dict, scraped_at: str, p2: dict) -> tuple[int, int]:
    """Replace/insert the `## Deck archetypes` section in every card that
    appears in a deck (and refresh it in cards that had one before)."""
    by_card: dict[str, list[dict]] = defaultdict(list)
    for d in decks:
        for s in d["cards"]:
            by_card[s].append(d)
    touched = updated_empty = 0
    for path in sorted(CARDS_DIR.glob("*.md")):
        slug = path.stem
        text = path.read_text()
        has_section = "## Deck archetypes" in text
        ds = by_card.get(slug, [])
        if not ds and not has_section:
            continue
        lines = ["## Deck archetypes", "", START,
                 f"Generated {now_deck(scraped_at)} from {len(decks)} popular deck(s) scraped from RoyaleAPI "
                 f"({p2.get('source_url', POPULAR_URL)}). Re-running Phase 2 replaces this block.", ""]
        if ds:
            groups: dict[str, list[dict]] = defaultdict(list)
            for d in ds:
                groups[d["fm"].get("archetype_primary", "unknown")].append(d)
            lines.append(f"This card appears in {len(ds)} of {len(decks)} scraped decks:")
            lines.append("")
            for a, group in sorted(groups.items(), key=lambda kv: -len(kv[1])):
                ex = ", ".join(f"[{g['fm'].get('display_name') or g['fm'].get('deck_key')}](../decks/{g['fm'].get('deck_key', g['path'].stem)}.md)" for g in group[:5])
                more = f" (+{len(group) - 5} more)" if len(group) > 5 else ""
                lines.append(f"- **[{ARCH_NAME.get(a, a)}](../archetypes/{a}.md)** — {len(group)} deck(s): {ex}{more}")
        else:
            lines.append("This card does not appear in any popular deck scraped in this run.")
            updated_empty += 1
        lines += [END, ""]
        block = "\n".join(lines)
        if has_section:
            new = re.sub(r"## Deck archetypes\n.*?(?=^## |\Z)", lambda m: block + "\n", text, count=1, flags=re.S | re.M)
        elif "\n## Source\n" in text:
            new = text.replace("\n## Source\n", "\n" + block + "\n## Source\n", 1)
        else:
            new = text.rstrip("\n") + "\n\n" + block
        if new != text:
            path.write_text(new)
        touched += 1
    return touched, updated_empty


def now_deck(s: str) -> str:
    return s or now_iso()


def verify(decks: list[dict], p2: dict) -> tuple[list[str], list[str]]:
    passed, failed = [], []
    def rec(ok, msg):
        (passed if ok else failed).append(msg)
    n_files = len(list(DECKS_DIR.glob("*.md")))
    n_rows = len(re.findall(r"^\| \[`", DECK_INDEX_MD.read_text(), re.M)) if DECK_INDEX_MD.exists() else 0
    rec(n_rows == n_files, f"deck_index.md rows ({n_rows}) match deck files in decks/ ({n_files})")
    bad = [d["path"].name for d in decks if d["fm"].get("archetype_primary", "") in ("", "none")]
    rec(not bad, f"every deck file has a non-empty archetype_primary ({len(decks) - len(bad)}/{len(decks)})" + (f"; BAD: {bad}" if bad else ""))
    missing_arch = [a for a in ARCHETYPES if not (ARCH_DIR / f"{a}.md").exists()]
    rec(not missing_arch, "all six archetype files exist" + (f"; MISSING: {missing_arch}" if missing_arch else ""))
    in_decks = sorted({s for d in decks for s in d["cards"]})
    unlinked = []
    for s in in_decks:
        p = CARDS_DIR / f"{s}.md"
        t = p.read_text() if p.exists() else ""
        if "## Deck archetypes" not in t or "does not appear in any popular deck" in t:
            unlinked.append(s)
    rec(not unlinked, f"every card in a scraped deck has a populated '## Deck archetypes' section ({len(in_decks) - len(unlinked)}/{len(in_decks)})"
        + (f"; MISSING: {unlinked}" if unlinked else ""))
    ph = [d["path"].name for d in decks if PLACEHOLDER in d["text"]]
    rec(not ph, f"'Why this deck works' written for every deck ({len(decks) - len(ph)}/{len(decks)})" + (f"; UNFILLED: {ph}" if ph else ""))
    # broken links inside deck + archetype files
    broken = []
    for p in list(DECKS_DIR.glob("*.md")) + list(ARCH_DIR.glob("*.md")):
        for link in re.findall(r"\]\((\.\./[^)]+\.md)\)", p.read_text()):
            if not (p.parent / link).resolve().exists():
                broken.append(f"{p.name} -> {link}")
    rec(not broken, "all relative links in decks/ and archetypes/ resolve" + (f"; BROKEN: {broken[:10]}" if broken else ""))
    heur = sum(1 for d in decks if d["fm"].get("classification_source") == "heuristic")
    rec(True, f"classification sources: {len(decks) - heur} site_label/agent, {heur} heuristic")
    return passed, failed


def qa_block(decks: list[dict], passed: list[str], failed: list[str], p2: dict) -> str:
    pc = p2.get("policy_check", {})
    lines = ["<!-- phase2-qa:start -->", "# Phase 2 QA report (decks and archetypes)", "",
             f"Generated {now_iso()}", "", "## Step 0 — source access and policy check", "",
             f"- Source: {p2.get('source_url', POPULAR_URL)}", f"- Scrape status: `{p2.get('status', 'not run')}`"]
    for f in p2.get("fetch_log", []):
        lines.append("- Fetch attempt: " + ", ".join(f"{k}={v}" for k, v in f.items() if k != "note"))
    if p2.get("blocker"):
        lines.append(f"- **Blocker:** {p2['blocker']}")
    if pc:
        lines += ["", f"- robots.txt ({pc.get('robots_url')}): HTTP {pc.get('robots_status')}; "
                  f"Content-Signal `{pc.get('content_signal')}`; crawl-delay for `*`/ClaudeBot: {pc.get('crawl_delay_seconds')}s; "
                  f"/decks/popular disallowed for `*`: {pc.get('decks_popular_disallowed_for_star')}; "
                  f"ClaudeBot disallowed from `/`: {pc.get('decks_popular_disallowed_for_claudebot')}"]
        for b in pc.get("robots_rules_for_star_and_claudebot", []):
            lines.append("  ```")
            lines += ["  " + l for l in b.splitlines()]
            lines.append("  ```")
        lines.append(f"- Terms of service ({pc.get('terms_url')}): HTTP {pc.get('terms_status')}; readable: {pc.get('terms_readable')}"
                     + (f"; {pc.get('terms_note')}" if pc.get('terms_note') else ""))
        for ex in pc.get("terms_scraping_excerpts", []):
            lines.append(f"  - \"{ex}\"")
    by_arch = Counter(d["fm"].get("archetype_primary") for d in decks)
    lines += ["", "## Counts", "", f"- Decks scraped: {len(decks)}"]
    lines += [f"- {ARCH_NAME[a]}: {by_arch.get(a, 0)}" for a in ARCHETYPES]
    lines += ["", "## Checks", ""] + [f"- PASS: {m}" for m in passed] + [f"- FAIL: {m}" for m in failed]
    lines += ["<!-- phase2-qa:end -->"]
    return "\n".join(lines)


def finalize(args) -> int:
    cd = card_data()
    manifest = load_manifest()
    p2 = manifest.setdefault("phase2", {})
    DECKS_DIR.mkdir(parents=True, exist_ok=True)
    decks = load_decks()
    scraped_at = p2.get("fetched_at") or now_iso()
    write_archetypes(decks, cd, scraped_at)
    touched, emptied = crosslink_cards(decks, cd, scraped_at, p2)
    # manifest statuses
    for d in decks:
        key = d["fm"].get("deck_key", d["path"].stem)
        e = manifest["items"].setdefault("deck:" + key, {"kind": "deck", "title": d["fm"].get("display_name", key), "url": p2.get("source_url")})
        ok = PLACEHOLDER not in d["text"] and d["fm"].get("archetype_primary") not in ("", "none")
        e.update({"status": "done" if ok else "pending", "stage": "done" if ok else "built",
                  "reason": None if ok else "why-this-deck-works unfilled or archetype missing", "updated_at": now_iso()})
    write_deck_index_md(decks, manifest, p2)
    passed, failed = verify(decks, p2)
    p2["verified_at"] = now_iso()
    p2["counts"] = {"decks": len(decks), **{a: sum(1 for d in decks if d["fm"].get("archetype_primary") == a) for a in ARCHETYPES}}
    save_manifest(manifest)
    block = qa_block(decks, passed, failed, p2)
    old = QA_MD.read_text() if QA_MD.exists() else ""
    if "<!-- phase2-qa:start -->" in old:
        new = re.sub(r"<!-- phase2-qa:start -->.*?<!-- phase2-qa:end -->", lambda m: block, old, flags=re.S)
    else:
        new = old.rstrip("\n") + "\n\n" + block + "\n"
    QA_MD.write_text(new)
    print(block)
    print(f"\ncards cross-linked: {touched} (of which {emptied} now reference no deck)")
    return 0 if not failed else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "finalize", "all"])
    args = ap.parse_args()
    if args.cmd in ("build", "all"):
        build(args)
    if args.cmd in ("finalize", "all"):
        return finalize(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

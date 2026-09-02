"""End-to-end smoke test for the Phase 2 pipeline on a SYNTHETIC page.

The fixture below is not real RoyaleAPI data; it only exercises parsing,
classification, file generation, cross-linking and idempotency inside a
temporary copy of the knowledge base (KB_ROOT), never the real one.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
KB_SRC = HERE.parents[2] / "knowledge_base"

FIXTURE_DECKS = [
    ("WB Log Bait 2.8 Cycle", "Log Bait", ["goblin-barrel", "princess", "goblin-gang", "dart-goblin", "rocket", "inferno-tower", "knight", "the-log"]),
    ("Golem Night Witch Beatdown", None, ["golem", "night-witch", "baby-dragon", "lightning", "tornado", "lumberjack", "mega-minion", "barbarian-barrel"]),
    ("Hog 2.6 Cycle", None, ["hog-rider", "ice-spirit", "skeletons", "the-log", "ice-golem", "musketeer", "cannon", "fireball"]),
    ("Pekka Bridge Spam", None, ["p-e-k-k-a", "battle-ram", "bandit", "royal-ghost", "electro-wizard", "magic-archer", "zap", "poison"]),
    ("X-Bow 3.0", None, ["x-bow", "tesla", "archers", "ice-spirit", "skeletons", "the-log", "fireball", "knight"]),
    ("Miner Poison Control", None, ["miner", "poison", "bowler", "ice-wizard", "tornado", "valkyrie", "inferno-tower", "the-log"]),
]


def fixture_html(cards_by_slug: dict) -> str:
    parts = ["<html><body><div class='deck_list'>"]
    for i, (name, label, cards) in enumerate(FIXTURE_DECKS):
        parts.append(f"<div class='deck_segment'><h4 class='deck_name'><a href='/decks/stats/{i}'>{name}</a></h4>")
        if label:
            parts.append(f"<a class='archetype' href='/decks/archetype/x'>{label}</a>")
        parts.append("<div class='deck_cards'>")
        for j, s in enumerate(cards):
            nm = cards_by_slug[s]["name"]
            evo = "-ev1" if (i == 0 and j == 0) else ""
            parts.append(f"<a href='/card/{s}{evo}'><img class='deck_card' alt='{nm}'></a>")
        parts.append("</div><div class='stats'>Rating 8{0} Usage 12,3{0}0 Wins 6,{0}00 Draws 1{0} Losses 5,{0}00 Avg Elixir 3.{0}</div></div>".format(i))
    parts.append("</div></body></html>")
    return "".join(parts)


def run(cmd, env):
    p = subprocess.run(cmd, cwd=HERE.parent, env=env, capture_output=True, text=True)
    if p.returncode not in (0,):
        print(p.stdout[-3000:], p.stderr[-3000:])
    return p


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="kb_phase2_test_"))
    kb = tmp / "knowledge_base"
    (kb / "meta").mkdir(parents=True)
    shutil.copy(KB_SRC / "meta" / "card_index.json", kb / "meta" / "card_index.json")
    (kb / "cards").mkdir()
    idx = json.loads((kb / "meta" / "card_index.json").read_text())
    by_slug = {c["slug"]: c for c in idx["cards"]}
    needed = sorted({s for _, _, cs in FIXTURE_DECKS for s in cs})
    for s in needed:
        assert s in by_slug, f"fixture uses unknown slug {s}"
        shutil.copy(KB_SRC / "cards" / f"{s}.md", kb / "cards" / f"{s}.md")
    html = tmp / "popular.html"
    html.write_text(fixture_html(by_slug))
    env = dict(os.environ, KB_ROOT=str(kb), KB_SCRATCH=str(tmp / "scratch"))

    p = run([sys.executable, "decks_fetch.py", "--html", str(html), "--skip-policy-check"], env)
    assert p.returncode == 0, "fetch/parse failed"
    dj = json.loads((kb / "meta" / "deck_index.json").read_text())
    assert dj["deck_count"] == len(FIXTURE_DECKS), dj["deck_count"]
    d0 = dj["decks"][0]
    assert d0["display_name"] == "WB Log Bait 2.8 Cycle", d0["display_name"]
    assert d0["site_label"] == "Log Bait", d0["site_label"]
    assert d0["evolutions"] == ["goblin-barrel"], d0["evolutions"]
    assert d0["site_stats_raw"]["rating"] == "80", d0["site_stats_raw"]
    assert d0["deck_key"] == "-".join(sorted(FIXTURE_DECKS[0][2]))

    p = run([sys.executable, "decks_build.py", "all"], env)
    assert p.returncode in (0, 1), "build/finalize crashed"  # 1 = checks failed (agent section unfilled, expected here)
    decks = {d["deck_key"]: d for d in json.loads((kb / "meta" / "deck_index.json").read_text())["decks"]}
    expect = {0: ("bait", "site_label"), 1: ("beatdown", "heuristic"), 2: ("cycle", "heuristic"),
              3: ("bridge-spam", "heuristic"), 4: ("siege", "heuristic"), 5: ("control", "heuristic")}
    for i, (arch, src) in expect.items():
        key = "-".join(sorted(FIXTURE_DECKS[i][2]))
        d = decks[key]
        assert d["archetype_primary"] == arch, (FIXTURE_DECKS[i][0], d["archetype_primary"], d["heuristic_scores"])
        assert d["classification_source"] == src, (FIXTURE_DECKS[i][0], d["classification_source"])
        assert (kb / "decks" / f"{key}.md").exists()
    hog = decks["-".join(sorted(FIXTURE_DECKS[2][2]))]
    assert hog["avg_elixir"] < 3.0, hog["avg_elixir"]
    for a in ["beatdown", "control", "cycle", "bait", "bridge-spam", "siege"]:
        t = (kb / "archetypes" / f"{a}.md").read_text()
        assert "deck_count: 1" in t, a
    log = (kb / "cards" / "the-log.md").read_text()
    assert log.count("## Deck archetypes") == 1 and "appears in 4 of 6" in log, log[-1200:]
    qa = (kb / "meta" / "qa_report.md").read_text()
    checks = qa.split("## Checks")[1].split("<!--")[0]
    fails = [l for l in checks.splitlines() if l.startswith("- FAIL")]
    # the only expected failure at this stage is the unfilled agent section
    assert all("Why this deck works" in l for l in fails), fails
    # idempotency: finalize again -> identical card files, single QA block
    before = {p.name: p.read_text() for p in (kb / "cards").glob("*.md")}
    p = run([sys.executable, "decks_build.py", "finalize"], env)
    after = {p.name: p.read_text() for p in (kb / "cards").glob("*.md")}
    changed = [n for n in before if before[n] != after[n]]
    # only the timestamp line inside the marker block may differ
    for n in changed:
        b = [l for l in before[n].splitlines() if not l.startswith("Generated ")]
        a = [l for l in after[n].splitlines() if not l.startswith("Generated ")]
        assert a == b, f"non-idempotent cross-link in {n}"
    qa = (kb / "meta" / "qa_report.md").read_text()
    assert qa.count("phase2-qa:start") == 1
    assert (kb / "meta" / "deck_index.md").read_text().count("| [`") == len(FIXTURE_DECKS)
    print("phase2 smoke test OK in", tmp)
    shutil.rmtree(tmp)
    return 0


if __name__ == "__main__":
    sys.exit(main())

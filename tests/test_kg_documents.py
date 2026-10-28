"""WP2: knowledge-graph and document builders."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clashbot.game_rules import load_cards  # noqa: E402
from tools.data.kb_markdown import (  # noqa: E402
    build_aliases, bullets, creator_blocks, first_table, mentions, parse_frontmatter, section,
)

DATASET = ROOT / "dataset" / "v1"


def test_parse_frontmatter():
    text = '---\nname: "Archers"\ntags: [a, "b", c]\nelixir_cost: 3\n  nested: ignored\n---\n# Body\n'
    fm, body = parse_frontmatter(text)
    assert fm == {"name": "Archers", "tags": ["a", "b", "c"], "elixir_cost": "3"}
    assert body == "# Body\n"
    assert parse_frontmatter("no frontmatter") == ({}, "no frontmatter")


def test_section_and_bullets():
    body = "# T\n\n## One\n\n- first bullet\n  continued here\n- second\nplain line\n- third\n\n## Two\n- other\n"
    sec = section(body, "One")
    assert "other" not in sec and "## One" not in sec
    assert bullets(sec) == ["first bullet continued here", "second", "third"]
    assert section(body, "Missing") is None
    assert bullets(section(body, "Two")) == ["other"]


def test_first_table():
    text = "intro\n| A | B | C |\n|---|---|---|\n| 1 | 2 | 3 |\n| 4 | 5 | 6 |\n\n| X |\n|---|\n| 9 |\n"
    assert first_table(text) == [{"A": "1", "B": "2", "C": "3"}, {"A": "4", "B": "5", "C": "6"}]
    assert first_table("no table") == []


def test_mentions():
    aliases = build_aliases(load_cards())
    text = "Skeleton Army and Skeletons beat Mini P.E.K.K.A., not Zappies or Zap"
    assert mentions(text, aliases) == ["skeleton-army", "skeletons", "mini-p-e-k-k-a", "zappies", "zap"]
    assert mentions("Spear Goblins", aliases) == ["spear-goblins"]


def test_creator_blocks():
    text = (
        "x\n<!-- creator-insights:ryleycr1:gameA-m0.0:start -->\n first body \n"
        "<!-- creator-insights:ryleycr1:gameA-m0.0:end -->\nmid\n"
        "<!-- creator-insights:ryleycr1:gameB:start -->second<!-- creator-insights:ryleycr1:gameB:end -->\n"
    )
    assert creator_blocks(text) == [("gameA-m0.0", "first body"), ("gameB", "second")]


@pytest.fixture(scope="module")
def built():
    if not (DATASET / "games.jsonl").exists():
        pytest.skip("dataset/v1/games.jsonl missing")
    for script in ("build_kg.py", "build_documents.py"):
        subprocess.run([sys.executable, str(ROOT / "tools" / "data" / script)], check=True, cwd=ROOT)
    return DATASET


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_kg_integration(built):
    nodes = _jsonl(built / "kg" / "nodes.jsonl")
    edges = _jsonl(built / "kg" / "edges.jsonl")
    ids = {n["id"] for n in nodes}
    for e in edges:
        assert e["src"] in ids, e
        assert e["dst"] in ids, e
    counters = [e for e in edges if e["rel"] == "COUNTERS"]
    assert {"wiki_strong_against", "wiki_weak_against"} <= {e["props"]["source"] for e in counters}
    assert any(e["src"] == "card:archers" and e["dst"] == "card:minions" for e in counters)
    for e in edges:
        if e["rel"] == "ANSWERED_BY":
            assert e["props"]["support"] >= 3


def test_documents_integration(built):
    docs = _jsonl(built / "documents.jsonl")
    assert docs
    assert len({d["doc_id"] for d in docs}) == len(docs)
    for d in docs:
        assert d["text"], d["doc_id"]
        assert "http" not in d["text"], d["doc_id"]
        assert "t=" not in d["text"], d["doc_id"]

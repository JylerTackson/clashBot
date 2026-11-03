"""MAC-R5 tests: memory stores (episodic / semantic / procedural), patch-versioned card facts, conflict weighting and the
knowledge-editing evaluation."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clashbot.memory import (BASE_PATCH, MemoryStore, conflict_weight, reweight_hits,  # noqa: E402
                             stale_cards)
from clashbot.retrieval import Hit  # noqa: E402
from tools.eval import edit_eval  # noqa: E402

NOW = "2026-11-02T12:00:00+00:00"
BASE_CARDS = {
    "hog-rider": {"slug": "hog-rider", "name": "Hog Rider", "card_type": "Troop", "rarity": "Rare",
                  "targets": "buildings", "elixir_cost": 4.0},
    "knight": {"slug": "knight", "name": "Knight", "card_type": "Troop", "rarity": "Common",
               "targets": "ground", "elixir_cost": 3.0},
}
REAL_DB = ROOT / "db" / "clashbot.sqlite"


def make_store(path=":memory:") -> MemoryStore:
    return MemoryStore(path, now=lambda: NOW)


def edit(store, card="hog-rider", attribute="elixir_cost", value=5, patch="P1"):
    return store.apply_edit(card, attribute, value, patch, "test", "unit test")


# ---- 1. patches -------------------------------------------------------------------------------------------

def test_patches():
    s = make_store()
    assert s.patches() == [BASE_PATCH] and s.patch_seq(BASE_PATCH) == 0
    assert s.add_patch("P1", "first") == 1
    assert s.add_patch("P2") == 2
    assert s.add_patch("P1") == 1                      # idempotent
    assert s.patches() == [BASE_PATCH, "P1", "P2"]
    with pytest.raises(KeyError):
        s.patch_seq("nope")


# ---- 2. apply_edit / card_fact ----------------------------------------------------------------------------

def test_apply_edit_and_card_fact():
    s = make_store()
    assert s.card_fact("hog-rider", "elixir_cost", base_cards=BASE_CARDS) == 4.0
    f1 = edit(s, value=5, patch="P1")
    assert s.card_fact("hog-rider", "elixir_cost", base_cards=BASE_CARDS) == 5
    assert s.card_fact("hog-rider", "elixir_cost", patch=BASE_PATCH, base_cards=BASE_CARDS) == 4.0
    assert s.card_fact("hog-rider", "elixir_cost", patch="P1", base_cards=BASE_CARDS) == 5
    assert s.card_fact("knight", "elixir_cost", base_cards=BASE_CARDS) == 3.0

    # a second edit in a later patch closes the first
    f2 = edit(s, value=6, patch="P2")
    assert f2 != f1
    assert s.card_fact("hog-rider", "elixir_cost", base_cards=BASE_CARDS) == 6
    assert s.card_fact("hog-rider", "elixir_cost", patch="P1", base_cards=BASE_CARDS) == 5
    assert s.card_fact("hog-rider", "elixir_cost", patch=BASE_PATCH, base_cards=BASE_CARDS) == 4.0
    row = s.conn.execute("SELECT valid_from, valid_to, source, provenance FROM card_facts WHERE fact_id = ?",
                         (f1,)).fetchone()
    assert (row["valid_from"], row["valid_to"], row["source"], row["provenance"]) == (1, 2, "test", "unit test")

    # re-edit inside the same patch replaces it; the first value was never valid
    f3 = edit(s, value=7, patch="P2")
    assert s.card_fact("hog-rider", "elixir_cost", base_cards=BASE_CARDS) == 7
    row = s.conn.execute("SELECT valid_from, valid_to FROM card_facts WHERE fact_id = ?", (f2,)).fetchone()
    assert row["valid_to"] == row["valid_from"] == 2
    assert s.card_fact("hog-rider", "elixir_cost", patch="P1", base_cards=BASE_CARDS) == 5
    assert f3 > f2

    with pytest.raises(ValueError):
        s.apply_edit("hog-rider", "name", "Pig", "P3", "test", "unit test")
    assert [r["op"] for r in s.log()] == ["ADD", "UPDATE", "UPDATE"]


# ---- 3. retract / edits_since -----------------------------------------------------------------------------

def test_retract_and_edits_since():
    s = make_store()
    f1 = edit(s, value=5, patch="P1")
    f2 = edit(s, value=6, patch="P2")
    assert [f["fact_id"] for f in s.edits_since(BASE_PATCH)] == [f1, f2]
    assert s.edits_since("P1")[0]["value"] == 6 and len(s.edits_since("P1")) == 1
    s.retract(f2)
    assert s.card_fact("hog-rider", "elixir_cost", base_cards=BASE_CARDS) == 5      # f1 reopened
    assert [f["fact_id"] for f in s.edits_since(BASE_PATCH)] == [f1]
    s.retract(f1)
    assert s.card_fact("hog-rider", "elixir_cost", base_cards=BASE_CARDS) == 4.0
    assert s.edits_since(BASE_PATCH) == []
    s.retract(f1)                                                                    # idempotent
    with pytest.raises(KeyError):
        s.retract(999)
    assert [r["op"] for r in s.log()] == ["ADD", "UPDATE", "DELETE", "DELETE"]


# ---- 4. edited_cards --------------------------------------------------------------------------------------

def test_edited_cards_deep_copy():
    s = make_store()
    edit(s, value=5, patch="P1")
    edit(s, card="ghost", value=2, patch="P1")                    # card absent from base_cards: ignored
    snapshot = json.loads(json.dumps(BASE_CARDS))
    out = s.edited_cards(BASE_CARDS)
    assert BASE_CARDS == snapshot and out is not BASE_CARDS
    assert out["hog-rider"]["elixir_cost"] == 5 and BASE_CARDS["hog-rider"]["elixir_cost"] == 4.0
    assert out["knight"] == BASE_CARDS["knight"] and out["knight"] is not BASE_CARDS["knight"]
    assert "ghost" not in out
    assert {k: v for k, v in out["hog-rider"].items() if k != "elixir_cost"} == \
           {k: v for k, v in BASE_CARDS["hog-rider"].items() if k != "elixir_cost"}
    assert s.edited_cards(BASE_CARDS, patch=BASE_PATCH) == BASE_CARDS
    out["knight"]["rarity"] = "Epic"                              # mutating the result leaves the input alone
    assert BASE_CARDS["knight"]["rarity"] == "Common"


# ---- 5. lessons -------------------------------------------------------------------------------------------

def test_lessons():
    s = make_store()
    a = s.write_lesson("ADD", "defence", "  Zap the skeleton army.  ")
    assert s.lessons()[0]["text"] == "Zap the skeleton army."
    assert s.write_lesson("ADD", "defence", "Zap the skeleton army.") == a
    assert s.lessons("defence")[0]["support"] == 2
    b = s.write_lesson("ADD", "offence", "Pump behind the king tower.", source="human")
    assert b != a and s.write_lesson("ADD", "offence", "Zap the skeleton army.") not in (a, b)
    assert s.write_lesson("UPDATE", "defence", "Zap the skeleton army, then counter-push.", lesson_id=a) == a
    got = {l["lesson_id"]: l for l in s.lessons()}
    assert got[a]["text"].endswith("counter-push.") and got[a]["support"] == 3
    assert s.lessons()[0]["lesson_id"] == a                       # ordered by support desc
    assert s.write_lesson("NOOP", "defence", lesson_id=a) == a
    assert s.write_lesson("DELETE", "offence", lesson_id=b) == b
    assert b not in [l["lesson_id"] for l in s.lessons()]
    assert b not in [l["lesson_id"] for l in s.lessons("offence")]
    with pytest.raises(ValueError):
        s.write_lesson("MERGE", "defence", "x")
    with pytest.raises(KeyError):
        s.write_lesson("UPDATE", "offence", "revive", lesson_id=b)       # deleted
    with pytest.raises(KeyError):
        s.write_lesson("UPDATE", "offence", "revive", lesson_id=12345)   # unknown
    with pytest.raises(KeyError):
        s.write_lesson("DELETE", "offence", lesson_id=b)
    ops = [r["op"] for r in s.log()]
    assert ops == ["ADD", "NOOP", "ADD", "ADD", "UPDATE", "NOOP", "DELETE"]
    assert all(r["store"] == "procedural" for r in s.log())
    assert s.log()[1]["detail"]["reason"] == "duplicate"


# ---- 6. episodes ------------------------------------------------------------------------------------------

def test_episodes_round_trip_and_filters():
    s = make_store()
    s.add_patch("P1")
    state = {"own": {"elixir": 5.5, "hand": ["knight", None]}, "units": [{"tile": [3, 4]}]}
    outcome = {"elixir_trade": -1.0, "towers_taken": []}
    e1 = s.add_episode("g1", 12.5, state, "knight|left|bridge", outcome)
    e2 = s.add_episode("g1", 20.0, state, None, None, source="pro", patch=BASE_PATCH)
    e3 = s.add_episode("g2", 3.0, {}, "hold", {})
    eps = s.episodes()
    assert [e["episode_id"] for e in eps] == [e1, e2, e3]
    assert eps[0]["state"] == state and eps[0]["outcome"] == outcome and eps[0]["t"] == 12.5
    assert eps[0]["patch"] == "P1" and eps[0]["source"] == "own"          # default patch = latest
    assert eps[1]["outcome"] is None and eps[1]["patch"] == BASE_PATCH and eps[1]["action_key"] is None
    assert [e["episode_id"] for e in s.episodes(game_id="g1")] == [e1, e2]
    assert [e["episode_id"] for e in s.episodes(source="pro")] == [e2]
    assert [e["episode_id"] for e in s.episodes(game_id="g1", source="own")] == [e1]
    assert s.episodes(game_id="nope") == []
    assert [r["op"] for r in s.log()] == ["ADD", "ADD", "ADD"] and s.log()[0]["store"] == "episodic"


# ---- 7. stale cards / conflict weighting ------------------------------------------------------------------

def _hit(row, key, score):
    return Hit(row=row, id=f"s{row}", game_id="g", kind="play", action_key=key, verdict=None, elixir_trade=None,
               score=score, ranks={"knn": row})


def test_stale_cards_and_reweighting():
    s = make_store()
    assert stale_cards(s) == set()
    f1 = edit(s, value=5, patch="P1")
    edit(s, card="knight", attribute="rarity", value="Rare", patch="P2")
    assert stale_cards(s) == {"hog-rider", "knight"}
    assert stale_cards(s, current_patch="P1") == {"hog-rider"}
    assert stale_cards(s, sample_patch="P1") == {"knight"}
    s.retract(f1)
    assert stale_cards(s) == {"knight"}

    stale = {"hog-rider"}
    assert conflict_weight("hog-rider|left|bridge", stale) == 0.5
    assert conflict_weight("hog-rider|left|bridge", stale, weight=0.25) == 0.25
    assert conflict_weight("knight|left|bridge", stale) == 1.0
    assert conflict_weight("hold", stale) == 1.0 and conflict_weight(None, stale) == 1.0

    hits = [_hit(0, "hog-rider|left|bridge", 0.030), _hit(1, "knight|left|bridge", 0.030),
            _hit(2, None, 0.020), _hit(3, "hog-rider|right|mid", 0.010)]
    out = reweight_hits(hits, stale)
    assert [h.row for h in out] == [1, 2, 0, 3]                  # the equal-score non-stale hit now ranks first
    assert out[2].score == pytest.approx(0.015) and out[0].score == 0.030
    assert hits[0].score == 0.030                                # inputs untouched
    assert isinstance(out[0], Hit) and out[0] is not hits[1]
    assert [h.row for h in reweight_hits(hits, set())] == [0, 1, 2, 3]


# ---- 8. persistence ---------------------------------------------------------------------------------------

def test_file_persistence(tmp_path):
    path = tmp_path / "sub" / "mem.sqlite"
    s = make_store(path)
    fid = edit(s, value=5, patch="P1")
    lid = s.write_lesson("ADD", "defence", "Keep a cheap cycle card.")
    eid = s.add_episode("g1", 1.0, {"a": 1}, "hold", {"b": 2})
    s.close()
    r = make_store(path)
    assert r.patches() == [BASE_PATCH, "P1"]
    assert r.card_fact("hog-rider", "elixir_cost", base_cards=BASE_CARDS) == 5
    assert [f["fact_id"] for f in r.edits_since(BASE_PATCH)] == [fid]
    assert [l["lesson_id"] for l in r.lessons()] == [lid]
    assert r.episodes()[0]["episode_id"] == eid and r.episodes()[0]["state"] == {"a": 1}
    assert len(r.log()) == 3
    r.close()


# ---- 9. edit_eval on a synthetic dataset ------------------------------------------------------------------

def _record(i, split, hand, elixir, deck=None, action_card=None, eligible=True, action_type="play"):
    state = {"phase": "single_elixir", "match_seconds": 30,
             "own": {"elixir": elixir, "hand": hand, "next_card": None,
                     "deck": deck if deck is not None else [h for h in hand if h]},
             "opponent": {"elixir_estimate": 5.0, "towers": {"king": None, "left": 4858, "right": 4858},
                          "deck_known": [], "recent_plays": []},
             "units": [], "threats": []}
    action = None if action_card is None and action_type == "play" else {"type": action_type, "card": action_card}
    return {"id": f"g{i}#{i}.0", "game_id": f"g{i}", "split": split, "kind": "play", "t": float(i), "state": state,
            "context_text": "own elixir hand", "action": action, "train_eligible": eligible}


def _write_records(folder: Path, recs: list[dict]) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "states.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    return folder


def _write_dataset(folder: Path) -> Path:
    return _write_records(folder, [
        _record(1, "val", ["hog-rider", "knight", "zap", "archers"], 4.0),        # legality flips 4 -> 5
        _record(2, "test", ["hog-rider", "knight", "zap", "archers"], 3.6),       # flips
        _record(3, "val", ["hog-rider", "knight", "zap", "archers"], 8.0),        # hog legal either way
        _record(4, "test", ["knight", "zap", "archers", "giant"], 6.0),           # does not involve hog-rider
        _record(5, "val", ["knight", "zap", "archers", None], 3.0, deck=["knight", "zap", "archers", "giant"]),
        _record(6, "train", ["hog-rider", "knight", "zap", "archers"], 4.0),      # train: not a probe
        _record(7, "test", ["knight", "zap", "archers", None], 4.2,               # hog only *possible* (unread slot)
                deck=["knight", "zap", "archers", "hog-rider"]),
    ])


def test_edit_eval_synthetic(tmp_path):
    ds = _write_dataset(tmp_path / "ds")
    out = tmp_path / "res" / "edit_eval.json"
    edit_eval.main(["--dataset", str(ds), "--db", str(tmp_path / "missing.sqlite"), "--card", "hog-rider",
                    "--n-probe", "10", "--out", str(out)])
    res = json.loads(out.read_text())
    assert res["edit"]["synthetic"] is True
    assert (res["edit"]["card"], res["edit"]["old"], res["edit"]["new"], res["edit"]["patch"]) == \
           ("hog-rider", 4.0, 5, "P1")
    c = res["criteria"]
    assert c["reliability"]["edited"] == 1.0 and c["reliability"]["no_edit"] == 0.0
    # records 1, 2 (in hand) and 7 (possible only) sit in the elixir band where legality flips
    g = c["generalization"]
    assert g["n"] == 3 and g["n_in_hand"] == 2 and g["n_possible"] == 3
    assert g["edited"] == 1.0 and g["no_edit"] == 0.0
    assert c["locality_cards"]["edited"] == 1.0 and c["locality_cards"]["n"] > 100
    assert c["locality_states"]["n"] == 2 and c["locality_states"]["edited"] == 1.0
    assert "skipped" in c["portability"] and "skipped" in res["conflict"]
    assert res["versioning"] is True and res["rollback"] is True
    md = out.with_suffix(".md").read_text()
    assert "| criterion | n | edited | no edit |" in md and "SYNTHETIC" in md and "hand-or-possible 3" in md


def test_edit_eval_other_attribute(tmp_path):
    ds = _write_dataset(tmp_path / "ds")
    out = tmp_path / "e.json"
    res = edit_eval.main(["--dataset", str(ds), "--db", str(tmp_path / "missing.sqlite"), "--card", "hog-rider",
                          "--attribute", "rarity", "--value", "Champion", "--out", str(out)])
    assert res["criteria"]["generalization"]["skipped"] == "n/a for this attribute"
    assert res["criteria"]["portability"]["skipped"] == "n/a for this attribute"
    assert res["criteria"]["reliability"]["edited"] == 1.0 and res["criteria"]["reliability"]["no_edit"] == 0.0
    assert res["criteria"]["locality_cards"]["edited"] == 1.0 and res["rollback"] is True
    with pytest.raises(ValueError):                         # no-op edit
        edit_eval.main(["--dataset", str(ds), "--db", str(tmp_path / "x"), "--card", "hog-rider", "--value", "4",
                        "--out", str(out)])


# ---- 9b. --cards auto / pooled criteria (Amendment 1) -----------------------------------------------------

def _auto_dataset(folder: Path) -> Path:
    cost = {"knight": 3.0, "zap": 2.0, "archers": 3.0, "giant": 5.0, "hog-rider": 4.0}
    recs = []
    i = 100
    # train plays: knight 5, zap 4, archers 3, giant 3 (tie with archers -> slug order), hog-rider 2, musketeer 1
    for card, n in {"knight": 5, "zap": 4, "archers": 3, "giant": 3, "hog-rider": 2, "musketeer": 1}.items():
        for _ in range(n):
            i += 1
            recs.append(_record(i, "train", ["knight", "zap", "archers", "giant"], 5.0, action_card=card))
    for _ in range(10):                                     # excluded: not train-eligible / val split / hold / no cost
        i += 1
        recs.append(_record(i, "train", ["knight", "zap", "archers", "giant"], 5.0, action_card="fireball",
                            eligible=False))
        recs.append(_record(i + 1000, "val", ["knight", "zap", "archers", "giant"], 5.0, action_card="fireball"))
        recs.append(_record(i + 2000, "train", ["knight", "zap", "archers", "giant"], 5.0, action_card=None,
                            action_type="hold"))
        recs.append(_record(i + 3000, "train", ["knight", "zap", "archers", "giant"], 5.0, action_card="mirror"))
    # probes: each card in hand at its own cost (inside the flip band), plus knight only possible
    for k, (card, c) in enumerate(cost.items()):
        recs.append(_record(10 + k, "val", [card, "x1", "x2", "x3"], c, deck=[card, "x1", "x2", "x3"]))
        recs.append(_record(20 + k, "test", [card, None, None, None], c, deck=[card, "x1", "x2", "x3"]))
    recs.append(_record(30, "test", ["zap", "archers", "giant", None], 3.0,
                        deck=["zap", "archers", "giant", "knight"]))                      # knight possible only
    recs.append(_record(31, "val", ["musketeer", "fireball", "minions", "x1"], 4.0))      # unrelated state
    return _write_records(folder, recs)


def test_auto_cards_selection():
    counts = edit_eval.Counter({"b": 3, "a": 3, "c": 5, "d": 1, "e": 2, "f": 2, "g": 1})
    assert edit_eval.auto_cards(counts) == ["c", "a", "b", "e", "f"]            # by count, ties by slug
    assert edit_eval.auto_cards(counts, k=2) == ["c", "a"]


def test_edit_eval_auto_pooled(tmp_path):
    ds = _auto_dataset(tmp_path / "ds")
    counts = edit_eval.train_play_counts(ds, edit_eval.load_cards())
    assert counts == {"knight": 5, "zap": 4, "archers": 3, "giant": 3, "hog-rider": 2, "musketeer": 1}
    out = tmp_path / "res" / "auto.json"
    res = edit_eval.main(["--dataset", str(ds), "--db", str(tmp_path / "missing.sqlite"), "--n-probe", "10",
                          "--out", str(out)])                       # no --card -> --cards auto
    assert json.loads(out.read_text())["pooled"] == res["pooled"]
    cards = [e["edit"]["card"] for e in res["edits"]]
    assert cards == ["knight", "zap", "archers", "giant", "hog-rider"] and len(set(cards)) == 5
    assert res["selection"]["mode"] == "auto" and res["selection"]["cards"] == cards
    for e in res["edits"]:
        ed = e["edit"]
        assert ed["synthetic"] is True and ed["attribute"] == "elixir_cost" and ed["patch"] == "P1"
        assert ed["new"] == ed["old"] + 1
        assert e["criteria"]["reliability"]["edited"] == 1.0 and e["criteria"]["reliability"]["no_edit"] == 0.0
        assert e["versioning"] is True and e["rollback"] is True

    p = res["pooled"]
    for name in ("reliability", "generalization", "locality_cards", "locality_states"):
        assert p[name]["n"] == sum(e["criteria"][name]["n"] for e in res["edits"]) > 0
    assert p["generalization"]["n_in_hand"] == sum(e["criteria"]["generalization"]["n_in_hand"] for e in res["edits"])
    assert p["generalization"]["n_possible"] == p["generalization"]["n"] > p["generalization"]["n_in_hand"]
    assert p["generalization"]["edited"] == 1.0 and p["generalization"]["no_edit"] == 0.0
    assert p["reliability"]["edited"] == 1.0 and p["reliability"]["no_edit"] == 0.0
    assert p["locality_cards"]["edited"] == 1.0 and p["locality_states"]["edited"] == 1.0
    assert "skipped" in p["portability"] and "skipped" in res["conflict_pooled"]
    assert res["versioning"] is True and res["rollback"] is True
    knight = res["edits"][0]["criteria"]["generalization"]
    assert knight["n_in_hand"] == 2 and knight["n_possible"] == 3        # 2 in hand, 1 only possible (knight at 3.0)
    md = out.with_suffix(".md").read_text()
    assert "Pooled" in md and "Per card" in md and "| card | edit |" in md and "SYNTHETIC" in md


def test_edit_eval_explicit_cards_and_errors(tmp_path):
    ds = _auto_dataset(tmp_path / "ds")
    db = str(tmp_path / "missing.sqlite")
    res = edit_eval.main(["--dataset", str(ds), "--db", db, "--cards", "zap, knight,zap", "--out",
                          str(tmp_path / "c.json")])
    assert [e["edit"]["card"] for e in res["edits"]] == ["zap", "knight"] and res["selection"]["mode"] == "explicit"
    with pytest.raises(SystemExit):                           # --card and --cards together
        edit_eval.main(["--dataset", str(ds), "--db", db, "--card", "knight", "--cards", "zap",
                        "--out", str(tmp_path / "x.json")])
    with pytest.raises(SystemExit):                           # --cards is +1 elixir only
        edit_eval.main(["--dataset", str(ds), "--db", db, "--attribute", "rarity", "--out", str(tmp_path / "x.json")])
    with pytest.raises(ValueError):
        edit_eval.main(["--dataset", str(ds), "--db", db, "--cards", "mirror", "--out", str(tmp_path / "x.json")])
    empty = _write_records(tmp_path / "empty", [_record(1, "val", ["knight", None, None, None], 3.0)])
    with pytest.raises(ValueError):                           # nothing to pick from
        edit_eval.main(["--dataset", str(empty), "--db", db, "--out", str(tmp_path / "x.json")])


# ---- 10. real data ----------------------------------------------------------------------------------------

@pytest.mark.skipif(not REAL_DB.exists() or not (ROOT / "dataset" / "v1" / "states.jsonl").exists(),
                    reason="db/clashbot.sqlite or dataset/v1/states.jsonl missing")
def test_edit_eval_real_data(tmp_path):
    res = edit_eval.main(["--n-probe", "50", "--out", str(tmp_path / "e.json")])
    p = res["pooled"]
    assert len(res["edits"]) == 5 and len({e["edit"]["card"] for e in res["edits"]}) == 5
    assert p["reliability"]["edited"] == 1.0
    assert p["locality_cards"]["edited"] == 1.0 and p["locality_states"]["edited"] == 1.0
    assert p["generalization"]["n"] == sum(e["criteria"]["generalization"]["n"] for e in res["edits"])
    assert p["generalization"]["n_possible"] >= p["generalization"]["n_in_hand"]
    assert res["versioning"] is True and res["rollback"] is True
    assert (tmp_path / "e.json").exists() and (tmp_path / "e.md").exists()

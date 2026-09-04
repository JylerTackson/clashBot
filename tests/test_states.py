"""Phase 2 game-state samples: schema validator, extractor, enrichment merge.

Everything here runs on a synthetic context (no video, no perception, no
torch): the extractor is fed a hand-built `context.json` plus a hand-built
match file and its output is checked against the real schema.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ex = _load("extract_states")
vs = _load("validate_states")
me = _load("merge_state_enrichment")
sv = _load("states_view")
SCHEMA = vs.load_schema()


# --------------------------------------------------------------------------- fixtures

def _clock(t: float) -> str:
    left = max(0, int(round(180 - t)))
    return f"{left // 60}:{left % 60:02d}"


def _timeline() -> list[dict]:
    rows = []
    for i in range(31):  # 0 .. 60 s, one row every 2 s
        t = i * 2.0
        enemy_left = 4000 if t < 40 else 3500  # a 500 HP swing between 38 and 40
        rows.append({
            "t": t, "clock": _clock(t),
            "phase": "single_elixir" if t < 40 else "double_elixir",
            "own_elixir": min(10, 4 + i % 6), "opp_elixir_est": 5.0,
            # from t=44 the HUD misreads a slot and the next card (neither is in his deck)
            "hand": (["hog-rider", "fireball", "musketeer", "skeletons"] if t < 44
                     else ["hog-rider", "mega-minion", "musketeer", "skeletons"]),
            "next": "the-log" if t < 44 else "goblinstein",
            "towers_own": {"king": 6000, "left": 4200, "right": 4200},
            "towers_enemy": {"king": 6000, "left": enemy_left, "right": 4200},
            "units": ([{"class": "battle-ram", "side": "enemy", "tile": [3, 12],
                        "heading": "advancing left lane", "speed": 1.8, "pred_2s": [3, 9],
                        "pos_std": 0.3, "eta_tower": {"tower": "own_left", "s": 3.5}}]
                      if 8 <= t <= 16 else []),
            "threats": (["battle-ram(e) advancing left lane, tower in 3.5s"] if 8 <= t <= 16 else []),
        })
    return rows


def _context() -> dict:
    ev = [
        {"timestamp": 10.0, "match_clock": "2:50", "player": "own", "card": "hog-rider",
         "tile": [9, 14], "elixir_before": 8.0, "elixir_after": 4.0,
         "detect_source": "hud", "confidence": "high", "type": "play_event"},
        {"timestamp": 30.0, "match_clock": "2:30", "player": "own", "card": "fireball",
         "tile": [10, 20], "elixir_before": 7.0, "elixir_after": 3.0,
         "detect_source": "deploy_label", "confidence": "medium", "type": "play_event"},
        {"timestamp": 32.0, "match_clock": "2:28", "player": "opponent", "card": "musketeer",
         "tile": [8, 22], "elixir_before": None, "elixir_after": None,
         "detect_source": "arena", "confidence": "low", "type": "play_event"},
        # inside the second key moment's span (20-25): must still get its own sample
        {"timestamp": 23.0, "match_clock": "2:37", "player": "own", "card": "musketeer",
         "tile": [7, 10], "elixir_before": 6.0, "elixir_after": 2.0,
         "detect_source": "deploy_label", "confidence": "medium", "type": "play_event"},
        # a HUD read of a card he does not play: dropped from the play lists
        {"timestamp": 47.0, "match_clock": "2:13", "player": "own", "card": "mega-minion",
         "tile": None, "elixir_before": 7.0, "elixir_after": 4.0,
         "detect_source": "hud", "confidence": "high", "type": "play_event"},
        # same, but read from an in-game deploy label: trusted, kept as it is
        {"timestamp": 50.0, "match_clock": "2:10", "player": "own", "card": "goblinstein",
         "tile": [6, 12], "elixir_before": None, "elixir_after": None,
         "detect_source": "deploy_label", "confidence": "medium", "type": "play_event"},
    ]
    return {
        "video_id": "TESTVID", "title": "Synthetic test match", "match_index": "0",
        "url": "https://www.youtube.com/watch?v=TESTVID", "calibration_method": "towers",
        "period": [0.0, 60.0], "start_t": 0.0, "end_t": 60.0,
        "own_deck_observed": ["hog-rider", "fireball", "musketeer", "skeletons",
                              "the-log", "ice-spirit", "cannon", "ice-golem"],
        "own_deck_key": "cannon-fireball-hog-rider-ice-golem-ice-spirit-musketeer-skeletons-the-log",
        "opponent": {"deck_known": ["musketeer"], "deck_complete": False, "kb_matches": []},
        "events": ev, "timeline": _timeline(),
        "transcript": [{"t": 9.0, "end": 12.0, "text": "hog rider at the bridge here"},
                       {"t": 37.0, "end": 39.0, "text": "eight seconds before the sample"},
                       {"t": 40.0, "end": 42.0, "text": "five seconds before the sample"},
                       {"t": 50.0, "end": 52.0, "text": "nothing to do right now"}],
        "quality": {"hand_conf_mean": 0.62, "readable_seconds": 60.0},
        "hero_note": None,
    }


MATCH_MD = """---
video_id: TESTVID
video_title: Synthetic test match
match_index: 0
own_deck: [hog-rider, fireball, musketeer, skeletons, the-log, ice-spirit, cannon, ice-golem]
own_deck_key: cannon-fireball-hog-rider-ice-golem-ice-spirit-musketeer-skeletons-the-log
result: win
---

# Match: synthetic

## Key moments

- t=10.0 (clock 2:50) — [Hog Rider](../cards/hog-rider.md) at [9, 14] into the
  Battle Ram push. "hog rider at the bridge here" (t=9-12).
- t=20.0-25.0 (clock 2:40-2:35) — a long key moment with a play inside it.

## How Ryley uses his cards

- nothing here.
"""


@pytest.fixture()
def game_dir(tmp_path, monkeypatch):
    """A synthetic run + match file, with the extractor pointed at them."""
    ctx_dir = tmp_path / "runs" / "videos" / "TESTVID" / "match_0"
    ctx_dir.mkdir(parents=True)
    (ctx_dir / "context.json").write_text(json.dumps(_context()))
    matches = tmp_path / "matches"
    matches.mkdir()
    (matches / "TESTVID-m0.md").write_text(MATCH_MD)
    states = tmp_path / "states"
    monkeypatch.setattr(ex, "MATCHES", matches)
    monkeypatch.setattr(ex, "STATES_DIR", states)
    monkeypatch.setattr(ex, "ready", lambda: [{"key": "TESTVID-m0",
                                               "context_json": str(ctx_dir / "context.json")}])
    monkeypatch.setattr(me, "STATES_DIR", states)
    monkeypatch.setattr(sv, "STATES_DIR", states)
    return {"states": states, "ctx": ctx_dir, "matches": matches}


def run_extractor(states_dir: Path, argv: list[str] | None = None) -> list[dict]:
    args = ["extract_states.py", "--out-dir", str(states_dir)] + (argv or [])
    old = sys.argv
    sys.argv = args
    try:
        assert ex.main() == 0
    finally:
        sys.argv = old
    return [json.loads(ln) for ln in (states_dir / "TESTVID-m0.jsonl").read_text().splitlines()]


# --------------------------------------------------------------------------- validator

def _valid_sample() -> dict:
    return {
        "id": "TESTVID-m0#9.0", "schema_version": 1,
        "source": {"video_id": "TESTVID", "match_index": "0",
                   "match_file": "knowledge_base/matches/TESTVID-m0.md",
                   "video_title": "Synthetic test match", "creator": "ryleycr1"},
        "kind": "key", "t": 9.0,
        "state": {"clock": "2:51", "phase": "single_elixir", "match_seconds": 9,
                  "own": {"elixir": 8, "hand": ["hog-rider", "fireball", None, "skeletons"],
                          "next_card": "the-log", "deck": ["hog-rider"],
                          "towers": {"king": 6000, "left": 4200, "right": 4200}},
                  "opponent": {"elixir_estimate": 5.0, "towers": {"king": None, "left": None, "right": None},
                               "deck_known": [], "recent_plays": []},
                  "units": [], "threats": []},
        "action": {"type": "hold"},
        "outcome": {"horizon_s": 15},
        "state_text": "[single_elixir|2:51] elixir 8 (opp ~5). hand: hog-rider, fireball, ?, "
                      "skeletons (next the-log). deck: hog-rider\nfield: none\nthreats: none\n"
                      "recent: none\ntowers own 4200/4200/6000, enemy ?/?/?\naction: hold",
        "key_moment": {"t_start": 10.0, "t_end": None, "text": "bullet", "clock": "2:50"},
        "context_refs": {"match_file": "knowledge_base/matches/TESTVID-m0.md", "cards": []},
        "quality": {"clock_read": True, "hand_confidence": 0.62, "calibration": "towers",
                    "state_source": "context.timeline"},
    }


def test_validator_accepts_a_valid_sample():
    assert vs.check_sample(_valid_sample(), SCHEMA) == []


@pytest.mark.parametrize("mutate, needle", [
    (lambda s: s.pop("outcome"), "missing required property 'outcome'"),
    (lambda s: s.update(id="TESTVID#9.0"), "pattern"),
    (lambda s: s.update(kind="highlight"), "enum"),
    (lambda s: s.update(schema_version=2), "const"),
    (lambda s: s.update(nonsense=1), "additional property 'nonsense'"),
    (lambda s: s["state"]["own"].update(elixir=12), "maximum"),
    (lambda s: s["state"]["own"].update(hand=["a", "b", "c"]), "minItems"),
    (lambda s: s["state"].update(phase="double_elixir_overtime"), "enum"),
    (lambda s: s["action"].update(type="cast"), "enum"),
    (lambda s: s["state"]["units"].append({"unit": "bats", "side": "them", "tile": [1, 2]}), "enum"),
    (lambda s: s.update(t="9.0"), "expected type number"),
    (lambda s: s.update(state_text="elixir 8 blah"), "template"),
])
def test_validator_rejects_bad_samples(mutate, needle):
    sample = _valid_sample()
    mutate(sample)
    errors = vs.check_sample(sample, SCHEMA)
    assert errors, f"expected an error mentioning {needle}"
    assert any(needle in e for e in errors), errors


def test_validator_flags_duplicate_ids(tmp_path):
    path = tmp_path / "dup.jsonl"
    line = json.dumps(_valid_sample())
    path.write_text(line + "\n" + line + "\n")
    errors, samples = vs.validate_file(path, SCHEMA, {})
    assert len(samples) == 2
    assert any("duplicate id" in e for e in errors)


# --------------------------------------------------------------------------- extractor

def test_extractor_kind_counts_and_spacing(game_dir):
    samples = run_extractor(game_dir["states"])
    by_kind = {}
    for s in samples:
        by_kind.setdefault(s["kind"], []).append(s)
    # one key sample per bullet, at t_start - 1
    assert [s["t"] for s in by_kind["key"]] == [9.0, 19.0]
    # plays: t=10 collides with the key sample at 9.0; t=23 sits inside the
    # 20-25 key span but more than 2 s from its key sample, so it survives
    assert [s["t"] for s in by_kind["play"]] == [22.0, 29.0, 46.0, 49.0]
    assert [s["action"]["card"] for s in by_kind["play"]] == ["musketeer", "fireball",
                                                              None, "goblinstein"]
    assert all(min(abs(s["t"] - k) for k in (9.0, 19.0)) > ex.PLAY_NEAR_KEY_S
               for s in by_kind["play"])
    # periodic every 10 s, skipping anything within +-4 s of an existing sample
    assert [s["t"] for s in by_kind["periodic"]] == [0.0, 40.0, 60.0]
    anchors = [9.0, 19.0, 22.0, 29.0, 46.0, 49.0]
    assert all(abs(s["t"] - a) > ex.NEAR_S for s in by_kind["periodic"] for a in anchors)
    assert len({s["id"] for s in samples}) == len(samples)
    assert [s["t"] for s in samples] == sorted(s["t"] for s in samples)


def test_extractor_key_sample_content(game_dir):
    key = [s for s in run_extractor(game_dir["states"]) if s["kind"] == "key"][0]
    assert key["id"] == "TESTVID-m0#9.0"
    assert key["key_moment"]["t_start"] == 10.0
    assert "Hog Rider" in key["key_moment"]["text"]
    assert key["source"]["creator"] == "ryleycr1"
    # the deck comes from the match-file frontmatter, not the pipeline's guess
    assert key["state"]["own"]["deck"][0] == "hog-rider"
    assert key["state"]["threats"] == ["battle-ram(e) advancing left lane, tower in 3.5s"]
    assert key["action"] == {**key["action"], "type": "play", "card": "hog-rider",
                             "tile": [9, 14], "lane": "middle", "zone": "bridge", "delay_s": 1.0}
    assert key["quality"]["state_source"] == "context.timeline"
    assert key["quality"]["clock_read"] is True
    assert [c["text"] for c in key["commentary"]] == ["hog rider at the bridge here"]
    assert key["context_refs"]["match_file"] == "knowledge_base/matches/TESTVID-m0.md"
    assert "knowledge_base/cards/hog-rider.md" in key["context_refs"]["cards"]


def test_extractor_outcome_deltas_and_verdict(game_dir):
    samples = {s["t"]: s for s in run_extractor(game_dir["states"])}
    # 29 -> 44 covers the 4000 -> 3500 drop on the enemy left tower
    out = samples[29.0]["outcome"]
    assert out["horizon_s"] == 15.0
    assert out["enemy_tower_hp_delta"]["left"] == -500.0
    assert out["own_tower_hp_delta"] == {"king": 0.0, "left": 0.0, "right": 0.0}
    assert out["verdict"] == "positive"
    assert out["game_result"] == "win"
    assert [p["card"] for p in out["opponent_plays"]] == ["musketeer"]
    # fireball (4) played by Ryley, musketeer (4) by the opponent -> trade 0
    assert out["elixir_trade"] == 0.0
    quiet = samples[40.0]["outcome"]
    assert quiet["verdict"] == "neutral" and quiet["towers_taken"] == []
    assert samples[40.0]["action"]["type"] == "hold"  # next own play is 6 s away


def test_state_text_template(game_dir):
    key = [s for s in run_extractor(game_dir["states"]) if s["kind"] == "key"][0]
    lines = key["state_text"].splitlines()
    assert len(lines) == 6
    assert vs.STATE_TEXT_HEAD.match(lines[0])
    assert lines[0].startswith("[single_elixir|2:52] elixir ")  # timeline row at or before t=9
    assert "hand: hog-rider, fireball, musketeer, skeletons (next the-log)" in lines[0]
    assert lines[1].startswith("field: battle-ram(e)@[3,12] advancing left lane eta own_left 3.5s")
    assert lines[2] == "threats: battle-ram(e) advancing left lane, tower in 3.5s"
    assert lines[3] == "recent: none"
    assert lines[4] == "towers own 4200/4200/6000, enemy 4000/4200/6000"
    assert lines[5] == "action: play hog-rider at [9,14] (bridge middle) after 1s"


def test_extracted_file_validates(game_dir):
    run_extractor(game_dir["states"])
    errors, samples = vs.validate_file(game_dir["states"] / "TESTVID-m0.jsonl", SCHEMA, {})
    assert errors == []
    assert len(samples) == 9


def test_extractor_is_deterministic_and_idempotent(game_dir):
    first = (game_dir["states"] / "TESTVID-m0.jsonl")
    run_extractor(game_dir["states"])
    text = first.read_text()
    run_extractor(game_dir["states"])
    assert first.read_text() == text


def test_periodic_interval_and_horizon_flags(game_dir):
    samples = run_extractor(game_dir["states"], ["--periodic", "12", "--horizon", "10"])
    periodic = [s["t"] for s in samples if s["kind"] == "periodic"]
    assert periodic == [0.0, 36.0, 60.0]  # 12/24/48 fall within 4 s of a key or play sample
    assert all(s["outcome"]["horizon_s"] == 10.0 for s in samples)


# --------------------------------------------------------------------------- enrichment merge

def test_merge_enrichment_survives_reextraction(game_dir):
    states = game_dir["states"]
    samples = run_extractor(states)
    target = samples[0]["id"]
    enrichment = {"situation_read": "Battle Ram is 3.5 s from the left tower.",
                  "reaction": "Answer with Cannon, keep The Log for the follow-up.",
                  "pro_action_rationale": "He counter-pushes instead of only defending.",
                  "principle": "Trade defence into pressure.",
                  "alternatives": ["Skeletons only, cheaper but no counter-push."],
                  "confidence": "high", "tags": ["defend-bridge-push"]}
    (states / "TESTVID-m0.enrich.json").write_text(json.dumps({target: enrichment}))

    merged, errors = me.merge_key("TESTVID-m0", SCHEMA, keep_sidecar=False)
    assert (merged, errors) == (1, [])
    assert not (states / "TESTVID-m0.enrich.json").exists()
    after = [json.loads(ln) for ln in (states / "TESTVID-m0.jsonl").read_text().splitlines()]
    assert after[0]["enrichment"] == enrichment
    assert vs.validate_file(states / "TESTVID-m0.jsonl", SCHEMA, {})[0] == []

    # re-extracting must not drop what the enrichment agent wrote
    again = run_extractor(states)
    assert again[0]["enrichment"] == enrichment
    assert all("enrichment" not in s for s in again[1:])


def test_merge_rejects_unknown_ids_and_bad_fields(game_dir):
    states = game_dir["states"]
    run_extractor(states)
    sidecar = states / "TESTVID-m0.enrich.json"
    sidecar.write_text(json.dumps({"TESTVID-m0#999.0": {"situation_read": "no such sample"}}))
    merged, errors = me.merge_key("TESTVID-m0", SCHEMA, keep_sidecar=False)
    assert merged == 0 and any("is not in" in e for e in errors)
    assert sidecar.exists(), "the sidecar must survive a failed merge"

    good_id = json.loads((states / "TESTVID-m0.jsonl").read_text().splitlines()[0])["id"]
    sidecar.write_text(json.dumps({good_id: {"confidence": "very-high"}}))
    merged, errors = me.merge_key("TESTVID-m0", SCHEMA, keep_sidecar=False)
    assert merged == 0 and any("enum" in e for e in errors)


# --------------------------------------------------------------------------- deck sanitisation

def test_hand_and_next_card_outside_the_deck_are_nulled(game_dir):
    samples = {s["t"]: s for s in run_extractor(game_dir["states"])}
    late = samples[46.0]  # built from the timeline row at t=46
    assert late["state"]["own"]["hand"] == ["hog-rider", None, "musketeer", "skeletons"]
    assert late["state"]["own"]["next_card"] is None
    assert "hand slot 1 read as mega-minion (not in deck)" in late["quality"]["notes"]
    assert any("next card read as goblinstein" in n for n in late["quality"]["notes"])
    assert "?" in late["state_text"].splitlines()[0]
    early = samples[29.0]
    assert early["state"]["own"]["hand"] == ["hog-rider", "fireball", "musketeer", "skeletons"]
    assert early["quality"]["notes"] == []


def test_hud_play_outside_the_deck_becomes_an_unknown_action(game_dir):
    samples = {s["t"]: s for s in run_extractor(game_dir["states"])}
    unknown = samples[46.0]  # anchored on the t=47 HUD read of mega-minion
    assert unknown["kind"] == "play"
    assert unknown["action"]["type"] == "unknown" and unknown["action"]["card"] is None
    assert unknown["action"]["detect_source"] == "hud"
    assert any("type unknown" in n for n in unknown["quality"]["notes"])
    assert unknown["state_text"].splitlines()[-1] == "action: unknown"
    # the same play is dropped from every play list, in this sample and later ones
    assert [p["card"] for p in samples[49.0]["state"]["own"]["recent_plays"]] == []
    assert [p["card"] for p in unknown["action"]["following_plays"]] == ["goblinstein"]


def test_misread_hud_play_is_not_priced_into_the_elixir_trade(game_dir):
    costs = ex.load_card_index()[0]
    samples = {s["t"]: s for s in run_extractor(game_dir["states"])}
    # horizon 46 -> 61 holds the dropped mega-minion HUD read (t=47) and the
    # trusted goblinstein label (t=50); only the label is priced
    assert samples[46.0]["outcome"]["elixir_trade"] == -costs["goblinstein"]
    assert costs["mega-minion"] > 0


def test_label_read_outside_the_deck_is_trusted(game_dir):
    samples = {s["t"]: s for s in run_extractor(game_dir["states"])}
    labelled = samples[49.0]  # anchored on the t=50 deploy label of goblinstein
    assert labelled["action"]["type"] == "play" and labelled["action"]["card"] == "goblinstein"
    assert labelled["action"]["detect_source"] == "deploy_label"


def test_sanitisation_is_skipped_when_the_deck_is_unknown(game_dir, monkeypatch):
    short = MATCH_MD.replace("own_deck: [hog-rider, fireball, musketeer, skeletons, "
                             "the-log, ice-spirit, cannon, ice-golem]",
                             "own_deck: [hog-rider, fireball, musketeer]")
    (game_dir["matches"] / "TESTVID-m0.md").write_text(short)
    samples = {s["t"]: s for s in run_extractor(game_dir["states"])}
    assert samples[46.0]["state"]["own"]["hand"][1] == "mega-minion"
    assert samples[46.0]["action"]["card"] == "mega-minion"
    assert samples[46.0]["quality"]["notes"] == []


def test_commentary_window(game_dir):
    samples = {s["t"]: s for s in run_extractor(game_dir["states"])}
    texts = [c["text"] for c in samples[46.0]["commentary"]]
    assert "five seconds before the sample" in texts       # |40 - 46| = 6
    assert "eight seconds before the sample" not in texts  # |37 - 46| = 9
    assert all(abs(c["t"] - s["t"]) <= ex.COMMENTARY_S
               for s in samples.values() for c in s["commentary"])


# --------------------------------------------------------------------------- digest view

def test_view_lists_every_id(game_dir):
    samples = run_extractor(game_dir["states"])
    text = sv.render("TESTVID-m0", sv.load("TESTVID-m0"), sv.KINDS)
    for s in samples:
        assert f"-- {s['id']} | {s['kind']} | t={s['t']}" in text, s["id"]
    assert text.count("\n-- ") == len(samples)
    # every block carries the state_text and an outcome line
    assert text.count("\nout: ") == len(samples)
    for s in samples:
        assert s["state_text"] in text
    head = text.splitlines()[:5]
    assert head[0].startswith("# TESTVID-m0 — Synthetic test match")
    assert "hog-rider" in head[1] and "result: win" in head[1]
    assert "samples: 9 (key 2, play 4, periodic 3)" in head[3]


def test_view_key_blocks_carry_bullet_commentary_and_outcome(game_dir):
    run_extractor(game_dir["states"])
    text = sv.render("TESTVID-m0", sv.load("TESTVID-m0"), ("key",))
    assert text.count("\n-- ") == 2
    first = text.split("\n-- ")[1]
    assert "key: t=10.0 (clock 2:50) — [Hog Rider]" in first
    assert "say: hog rider at the bridge here" in first
    assert "out: own k/l/r 0/0/0 | enemy 0/0/0 | neutral | trade -8 | game win" in first


def test_cli_accepts_keys_that_start_with_a_dash(game_dir, capsys):
    run_extractor(game_dir["states"])
    dashed = game_dir["states"] / "-DASHVID-m0.jsonl"
    dashed.write_text((game_dir["states"] / "TESTVID-m0.jsonl").read_text())
    capsys.readouterr()  # drop the extractor's output
    old = sys.argv
    sys.argv = ["states_view.py", "-DASHVID-m0", "--kind", "key"]
    try:
        assert sv.main() == 0
    finally:
        sys.argv = old
    out = capsys.readouterr().out
    assert out.startswith("# -DASHVID-m0 ") and out.count("\n-- ") == 2
    assert ex.pull_only(["--only", "-V4H_YeMGGk-m0.0", "x-m1", "--periodic", "5"]) == (
        ["--periodic", "5"], ["-V4H_YeMGGk-m0.0", "x-m1"])
    assert me.cli_argv(["-V4H_YeMGGk-m0.0", "--keep-sidecar"]) == [
        "--keep-sidecar", "--", "-V4H_YeMGGk-m0.0"]


def test_view_is_much_smaller_than_the_jsonl(game_dir):
    run_extractor(game_dir["states"])
    raw = (game_dir["states"] / "TESTVID-m0.jsonl").stat().st_size
    samples = sv.load("TESTVID-m0")
    view = len(sv.render("TESTVID-m0", samples, sv.KINDS).encode())
    assert view < raw / 2
    assert view / len(samples) <= 1200  # bytes per sample


# --------------------------------------------------------------------------- helpers

def test_bullet_parsing_variants():
    cases = {
        "t=738.6 (clock 2:50) — opens by splitting": (738.6, None, "2:50"),
        "**t=24-31 (2:32 → 2:25)** — first defensive package": (24.0, 31.0, "2:32"),
        "t≈1249 (clock 0:03 -> OT 1:59) — the clock resets": (1249.0, None, "0:03"),
        "t=~262-270 (clock 1:32-1:24): with a Lava Hound": (262.0, 270.0, "1:32"),
        "t=176 — result stated": (176.0, None, None),
    }
    for text, (t0, t1, clock) in cases.items():
        b = ex.parse_bullet(text)
        assert (b["t_start"], b["t_end"], b["clock"]) == (t0, t1, clock), text
    plain = ex.parse_bullet("314-321s (clock ~2:05, commentary only): the matchup thesis")
    assert plain["t_start"] is None and plain["seconds_alt"][0] == (314.0, 321.0)


def test_phase_zone_lane_helpers():
    assert ex.norm_phase("single_elixir") == ("single_elixir", None)
    phase, note = ex.norm_phase("double_elixir_overtime")
    assert phase == "overtime" and "overtime" in note
    assert ex.norm_phase(None) == (None, None)
    assert [ex.zone_of([9, r]) for r in (0, 8, 14, 20)] == ["back", "mid", "bridge", "enemy_half"]
    assert [ex.lane_of([c, 8]) for c in (2, 9, 15)] == ["left", "middle", "right"]
    assert ex.clock_to_seconds("2:50", "single_elixir") == 10
    assert ex.clock_to_seconds("1:00", "overtime") == 240
    assert ex.clock_to_seconds(None, None) is None

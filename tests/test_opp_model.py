"""Opponent-deck model (MAC-AK K3): prediction on 4 fixture decks, the deck-co-occurrence builder, the evaluation helpers."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clashbot.config import DATASET_DIR, KB  # noqa: E402
from clashbot.opp_model import OppDeckModel, cap_normalise  # noqa: E402
from tools.data import build_deck_cooc as bdc  # noqa: E402
from tools.eval import eval_opp_model as ev  # noqa: E402

DECK_A = ["giant", "balloon", "lightning", "tornado", "night-witch", "baby-dragon", "mini-p-e-k-k-a", "zap"]
DECK_B = ["hog-rider", "musketeer", "cannon", "ice-spirit", "skeletons", "the-log", "fireball", "ice-golem"]
DECK_C = ["x-bow", "tesla", "archers", "knight", "fireball", "the-log", "skeletons", "electro-spirit"]
DECK_D = ["miner", "graveyard", "poison", "valkyrie", "bats", "zap", "the-log", "knight"]
DECKS = [DECK_A, DECK_B, DECK_C, DECK_D]
ARCHETYPES = ["beatdown", "cycle", "siege", "control"]
REAL_GAMES = DATASET_DIR / "games.jsonl"
REAL_STATES = DATASET_DIR / "states.jsonl"


def fixture_model(**kw) -> OppDeckModel:
    return OppDeckModel(DECKS, archetypes=ARCHETYPES, **kw)


# ---------------------------------------------------------------- model
def test_seen_subset_of_deck_ranks_the_rest_of_that_deck_first():
    model = fixture_model()
    seen = ["giant", "balloon"]
    preds = model.predict(seen, top=6)
    assert {c for c, _ in preds} == set(DECK_A) - set(seen)
    assert all(p > 0.5 for _, p in preds)
    # also with a different deck and a single card that is unique to it
    assert {c for c, _ in fixture_model().predict(["x-bow"], top=7)} == set(DECK_C) - {"x-bow"}


def test_seen_cards_are_never_predicted():
    model = fixture_model()
    for seen in (["giant"], ["giant", "balloon", "zap"], ["the-log", "knight"], ["not-a-card", "giant"], list(DECK_A)):
        got = {c for c, _ in model.predict(seen, top=len(model.cards))}
        assert not got & set(seen)


def test_probabilities_sum_to_hidden_slots_and_stay_in_unit_interval():
    model = fixture_model()
    for seen in ([], ["giant"], ["giant", "balloon"], ["zap", "the-log", "knight"], ["giant", "not-a-card"],
                 DECK_A[:5], DECK_A[:7]):
        preds = model.predict(seen, top=len(model.cards))
        assert sum(p for _, p in preds) == pytest.approx(8 - len(set(seen)), abs=1e-3)
        assert all(0.0 <= p <= 1.0 for _, p in preds)
        assert [p for _, p in preds] == sorted((p for _, p in preds), reverse=True)


def test_predict_top_limit_and_full_deck():
    model = fixture_model()
    assert len(model.predict(["giant"], top=3)) == 3
    assert len(model.predict(["giant"])) == 8
    assert model.predict(DECK_A) == []                       # nothing left to predict
    assert model.predict(["giant"], top=0) == []
    assert len(model.predict([])) == 8                       # no evidence: popularity


def test_cap_normalise_caps_at_one_and_keeps_the_sum():
    out = cap_normalise({"a": 10.0, "b": 1.0, "c": 1.0, "d": 0.0}, 2.5)
    assert out["a"] == 1.0
    assert sum(out.values()) == pytest.approx(2.5)
    assert out["b"] == pytest.approx(out["c"]) and out["d"] == 0.0
    assert sum(cap_normalise({"a": 0.0, "b": 0.0}, 1.0).values()) == pytest.approx(1.0)      # no signal: even split
    assert cap_normalise({"a": 1.0, "b": 2.0}, 5.0) == {"a": 1.0, "b": 1.0}                    # more slots than cards


def test_pair_counts_and_dedup_counts():
    model = OppDeckModel([DECK_A, DECK_B, DECK_C], counts=[3, 1, 1], count_power=1.0)
    assert model.card_counts["giant"] == 3 and model.card_counts["fireball"] == 2
    assert model.pair_counts["giant"]["balloon"] == 3 and model.pair_counts["balloon"]["giant"] == 3
    assert model.pair_counts["fireball"]["the-log"] == 2 and "fireball" not in model.pair_counts["fireball"]
    unit = OppDeckModel([DECK_A, DECK_B, DECK_C], counts=[3, 1, 1])      # default count_power 0: each distinct deck once
    assert unit.card_counts["giant"] == 1 and unit.counts == [3, 1, 1]


def test_save_load_roundtrip(tmp_path):
    model = fixture_model(temperature=0.08)
    path = tmp_path / "kg2" / "deck_cooc.json"
    model.save(path, meta={"note": "fixture"})
    data = json.loads(path.read_text())
    assert {"cards", "pair_counts", "card_counts", "decks"} <= set(data)
    assert data["decks"][0] and all(len(d) == 8 for d in data["decks"]) and data["meta"] == {"note": "fixture"}
    again = OppDeckModel.load(path)
    assert again.temperature == 0.08 and again.cards == model.cards
    assert again.predict(["giant", "zap"]) == model.predict(["giant", "zap"])
    assert again.pair_counts == model.pair_counts


def test_likely_archetype():
    model = fixture_model()
    arch, prob = model.likely_archetype(["giant", "balloon"])
    assert arch == "beatdown" and prob > 0.8
    assert model.likely_archetype(["x-bow"])[0] == "siege"
    assert OppDeckModel(DECKS).likely_archetype(["giant"]) is None       # no labels


# ---------------------------------------------------------------- builder
DECK_MD = """---
deck_key: {key}
display_name: Fixture {key}
archetype_primary: {arch}
archetype_secondary: none
avg_elixir: 3.0
---

# Fixture {key}

## Cards

{cards}

## Classification

Primary archetype: [{arch}](../archetypes/{arch}.md).
"""


def write_kb(kb: Path, decks: dict[str, tuple[str, list[str]]]) -> None:
    (kb / "decks").mkdir(parents=True)
    for key, (arch, cards) in decks.items():
        bullets = "\n".join(f"- [{c.title()}](../cards/{c}.md) — 3 elixir, Common Troop, targets ground" for c in cards)
        (kb / "decks" / f"{key}.md").write_text(DECK_MD.format(key=key, arch=arch, cards=bullets), encoding="utf-8")


def game(gid: str, own: list[str], opp: list[str], status: str = "valid", split: str = "train") -> dict:
    return {"game_id": gid, "split": split, "own_deck": own, "deck_status": status, "own_archetype": "beatdown",
            "opponent_deck": opp, "opponent_archetype_guess": "control"}


def write_games(ds: Path, games: list[dict]) -> None:
    ds.mkdir(parents=True, exist_ok=True)
    (ds / "games.jsonl").write_text("\n".join(json.dumps(g) for g in games) + "\n", encoding="utf-8")


def test_builder_parses_a_fixture_kb_deck_file(tmp_path):
    write_kb(tmp_path / "kb", {"deck-a": ("beatdown", DECK_A), "deck-b": ("cycle", DECK_B)})
    d = bdc.parse_deck_file(tmp_path / "kb" / "decks" / "deck-a.md")
    assert d == {"deck_key": "deck-a", "cards": DECK_A, "archetype": "beatdown"}
    assert [x["deck_key"] for x in bdc.load_kb_decks(tmp_path / "kb")] == ["deck-a", "deck-b"]


def test_builder_collects_dedups_and_counts(tmp_path):
    kb, ds = tmp_path / "kb", tmp_path / "ds"
    write_kb(kb, {"deck-a": ("beatdown", DECK_A), "deck-c": ("siege", DECK_C)})
    write_games(ds, [
        game("g0", DECK_A, DECK_B),                                        # own A (also in KB), opp B
        game("g1", DECK_A, DECK_B[:7]),                                    # own A again, opp B minus one card
        game("g2", DECK_D, DECK_B),                                        # own D, opp B again
        game("g3", DECK_A[:6], DECK_C[:5], status="invalid"),              # own invalid, opp too small
        game("g4", DECK_C, DECK_A + ["extra-card"]),                       # opp with 9 cards: dropped
    ])
    stats = Counter()
    entries = bdc.collect_entries(ds, kb, stats=stats)
    assert [e["source"] for e in entries].count("kb") == 2
    assert [e["source"] for e in entries].count("own") == 4 and [e["source"] for e in entries].count("opp") == 3
    assert stats["own_not_valid"] == 1 and stats["opp_too_small"] == 2
    decks, counts, archetypes = bdc.aggregate(entries)
    by_deck = {d: (n, a) for d, n, a in zip(decks, counts, archetypes)}
    assert len(decks) == 5 and sum(counts) == len(entries) == 9
    assert by_deck[tuple(sorted(DECK_A))] == (3, "beatdown")       # kb + 2 own games, KB label wins
    assert by_deck[tuple(sorted(DECK_B))] == (2, "control")        # opponent guess is the only label
    assert by_deck[tuple(sorted(DECK_D))][1] == "beatdown"         # own_archetype


def test_builder_drops_decks_with_unknown_cards(tmp_path):
    ds = tmp_path / "ds"
    write_games(ds, [game("g0", DECK_A, DECK_B), game("g1", DECK_A, DECK_C)])
    (tmp_path / "kb" / "decks").mkdir(parents=True)
    stats = Counter()
    entries = bdc.collect_entries(ds, tmp_path / "kb", card_slugs=set(DECK_A) | set(DECK_B), stats=stats)
    assert stats["unknown_card"] == 1 and len(entries) == 3


def test_builder_main_writes_a_loadable_file(tmp_path):
    kb, ds, out = tmp_path / "kb", tmp_path / "ds", tmp_path / "out" / "deck_cooc.json"
    write_kb(kb, {"deck-c": ("siege", DECK_C)})
    write_games(ds, [game("g0", DECK_A, DECK_B), game("g1", DECK_A, DECK_D)])
    info = bdc.main(["--dataset", str(ds), "--kb", str(kb), "--out", str(out)])
    assert info["total_decks"] == 5 and info["unique_decks"] == 4
    data = json.loads(out.read_text())
    assert len(data["decks"]) == 4 and sum(data["deck_counts"]) == 5 and data["meta"]["unique_decks"] == 4
    model = OppDeckModel.load(out)
    assert {c for c, _ in model.predict(["giant", "balloon"], top=6)} == set(DECK_A) - {"giant", "balloon"}


# ---------------------------------------------------------------- live Training Camp decks (K4a, FIX #42 d)
def write_run(runs: Path, name: str, contexts: list[str | None], frames: list[dict] | None = None) -> Path:
    run = runs / name
    run.mkdir(parents=True)
    lines = [json.dumps({"t": float(i), "idx": i, "key": "hold", "context_text": c}) for i, c in enumerate(contexts)]
    (run / "decisions.jsonl").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    if frames is not None:
        (run / "frames.jsonl").write_text("\n".join(json.dumps(f) for f in frames) + "\n", encoding="utf-8")
    return run


def ctx(opp: str) -> str:
    return f"[single_elixir|2:00] elixir 5 (opp ~4). hand: a, b, c, d (next e). deck: x. opp deck: {opp}\nfield: nothing"


def test_deck_weights_and_tags_and_training_camp_share():
    model = OppDeckModel(DECKS, deck_weights=[3, 1, 1, 1], deck_tags=[["training_camp"], [], [], []])
    assert model.total_weight() == pytest.approx(6.0)             # count_power 0: each deck once, deck A thrice
    assert model.training_camp_share() == pytest.approx(0.5) and model.training_camp_decks() == 1
    assert model.card_counts["giant"] == 3 and model.card_counts["fireball"] == 2     # weighted counts
    assert model.pair_counts["giant"]["balloon"] == 3
    assert fixture_model().training_camp_share() == 0.0 and fixture_model().total_weight() == 4.0
    # the heavier deck pulls the prediction: with one card both A and C-like decks share, A wins at weight 3
    heavy = OppDeckModel([DECK_A, DECK_D], deck_weights=[3, 1])
    light = OppDeckModel([DECK_A, DECK_D], deck_weights=[1, 1])
    assert dict(heavy.predict(["zap"], top=20))["giant"] > dict(light.predict(["zap"], top=20))["giant"]
    with pytest.raises(ValueError):
        OppDeckModel(DECKS, deck_weights=[1, 1])


def test_weights_and_tags_survive_save_load(tmp_path):
    model = OppDeckModel(DECKS, deck_weights=[3, 1, 1, 1], deck_tags=[["training_camp"], [], [], []])
    path = tmp_path / "deck_cooc.json"
    model.save(path)
    data = json.loads(path.read_text())
    assert data["deck_weights"] == [3, 1, 1, 1] and data["total_weight"] == 6 and data["deck_tags"][0] == ["training_camp"]
    again = OppDeckModel.load(path)
    assert again.training_camp_share() == pytest.approx(0.5) and again.predict(["zap"]) == model.predict(["zap"])
    # a file written before K4a has no weights / tags
    for k in ("deck_weights", "deck_tags", "total_weight"):
        data.pop(k)
    old = OppDeckModel.from_dict(data)
    assert old.training_camp_share() == 0.0 and old.deck_weights == [1.0] * 4


def test_live_run_deck_reads_the_last_context_with_an_opp_deck_line(tmp_path):
    assert bdc.deck_from_context(ctx("giant, balloon, night-witch")) == ["giant", "balloon", "night-witch"]
    assert bdc.deck_from_context(ctx("unknown")) == [] and bdc.deck_from_context(None) == []
    assert bdc.deck_from_context("no such line") == []
    r1 = write_run(tmp_path, "r1", [ctx("giant"), ctx("giant, balloon"), ctx("giant, balloon, bats, zap")])
    r2 = write_run(tmp_path, "r2", [ctx("giant, zap"), ctx("unknown"), None])           # the last context has none: use the previous
    r3 = write_run(tmp_path, "r3", [], frames=[{"idx": 1, "live_state": 1.0},
                                                {"idx": 2, "perception": {"opponent": {"deck_known": ["miner", "bats"]}}}])
    assert bdc.live_run_deck(r1) == ["giant", "balloon", "bats", "zap"]
    assert bdc.live_run_deck(r2) == ["giant", "zap"]
    assert bdc.live_run_deck(r3) == ["miner", "bats"]                                    # frames' perception when no decision
    assert bdc.live_run_deck(tmp_path / "missing") == []


def test_collect_live_entries_weights_tags_and_drops(tmp_path):
    runs = tmp_path / "runs"
    write_run(runs, "full", [ctx(", ".join(DECK_A))])
    write_run(runs, "four", [ctx("giant, balloon, bats, zap")])
    write_run(runs, "small", [ctx("giant, balloon")])
    write_run(runs, "empty", [ctx("unknown")])
    write_run(runs, "odd", [ctx("giant, balloon, bats, zap, not-a-card")])
    vid = write_run(runs, "vid", [ctx(", ".join(DECK_B))])                               # a replayed video: a human, not a bot
    (vid / "config.json").write_text(json.dumps({"source": "video", "video": "data/videos/x.mp4"}), encoding="utf-8")
    mir = write_run(runs, "mirror-dry", [ctx(", ".join(DECK_C))])                         # a dry run on the live mirror: kept
    (mir / "config.json").write_text(json.dumps({"source": "mirroring", "video": None, "dry_run": True}), encoding="utf-8")
    (runs / "notes").mkdir()                                                             # not a run directory
    stats = Counter()
    got = bdc.collect_live_entries(runs, card_slugs=set(DECK_A) | set(DECK_C) | {"bats"}, stats=stats)
    assert [e["game_id"] for e in got] == ["four", "full", "mirror-dry", "odd"]
    assert all(e["source"] == "live" and e["weight"] == 3.0 and e["tags"] == ("training_camp",) for e in got)
    assert got[3]["cards"] == ("balloon", "bats", "giant", "zap")                        # the unknown card is dropped
    assert stats["live_runs"] == 7 and stats["live_too_small"] == 1 and stats["live_no_deck"] == 1
    assert stats["live_unknown_card"] == 1 and stats["live_video"] == 1
    assert bdc.is_video_run(vid) and not bdc.is_video_run(mir) and not bdc.is_video_run(runs / "full")
    assert len(bdc.collect_live_entries(runs, None, min_cards=2)) == 5                   # 'small' now counts, 'empty' never
    assert bdc.collect_live_entries(runs, None, weight=5.0)[0]["weight"] == 5.0


def test_builder_with_live_runs_adds_weighted_tagged_decks(tmp_path):
    kb, ds, runs, out = tmp_path / "kb", tmp_path / "ds", tmp_path / "runs", tmp_path / "out" / "deck_cooc.json"
    write_kb(kb, {"deck-c": ("siege", DECK_C)})
    write_games(ds, [game("g0", DECK_A, DECK_B)])
    write_run(runs, "r1", [ctx(", ".join(DECK_D))])
    write_run(runs, "r2", [ctx(", ".join(DECK_D))])                                       # the same deck twice: merged
    base = bdc.main(["--dataset", str(ds), "--kb", str(kb), "--out", str(out)])
    assert base["training_camp_share"] == 0.0 and base["live_decks"] == 0
    info = bdc.main(["--dataset", str(ds), "--kb", str(kb), "--out", str(out), "--live-runs", str(runs)])
    assert info["total_decks"] == base["total_decks"] + 2 and info["unique_decks"] == base["unique_decks"] + 1
    assert info["live_decks"] == 1 and info["training_camp_decks"] == 1
    assert info["total_weight"] == pytest.approx(base["total_weight"] + 3.0)              # count_power 0: one deck, weight 3
    assert info["training_camp_share"] == pytest.approx(3.0 / (base["total_weight"] + 3.0), abs=1e-3)
    model = OppDeckModel.load(out)
    assert model.training_camp_share() == pytest.approx(info["training_camp_share"], abs=1e-3)
    assert info["dropped"].get("live_runs") == 2
    # a deck that is also a pro deck keeps its KB archetype and gains the tag and the weight
    write_run(runs, "r3", [ctx(", ".join(DECK_C))])
    entries = bdc.collect_entries(ds, kb, live_runs=runs)
    decks, counts, archetypes, weights, tags = bdc.aggregate_full(entries)
    i = decks.index(tuple(sorted(DECK_C)))
    assert counts[i] == 2 and archetypes[i] == "siege" and weights[i] == 3.0 and tags[i] == ("training_camp",)
    assert bdc.aggregate(entries) == (decks, counts, archetypes)                          # the old 3-tuple still works


@pytest.mark.skipif(not (ROOT / "runs" / "poc").exists(), reason="runs/poc missing")
def test_real_live_runs_give_a_few_decks():
    got = bdc.collect_live_entries(ROOT / "runs" / "poc", bdc.known_card_slugs())
    assert got and all(4 <= len(e["cards"]) <= 8 and e["weight"] == 3.0 for e in got)


# ---------------------------------------------------------------- evaluation
def test_reveal_orders_sorts_by_play_time_and_drops_cards_outside_the_deck(tmp_path):
    def rec(gid, plays):
        return {"game_id": gid, "state": {"opponent": {"recent_plays": plays}}}
    lines = [
        rec("g0", [{"t": 30.0, "card": "giant"}, {"t": 5.0, "card": "zap"}, {"t": 7.0, "card": "prince"}]),
        rec("g0", [{"t": 30.0, "card": "giant"}, {"t": 20.0, "card": "balloon"}, {"t": None, "card": "tornado"},
                   {"t": 40.0, "card": None}, {"t": 4.0, "card": "zap"}]),
        rec("other", [{"t": 1.0, "card": "giant"}]),
    ]
    path = tmp_path / "states.jsonl"
    path.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    orders = ev.reveal_orders(path, {"g0": set(DECK_A)})
    assert orders == {"g0": ["zap", "balloon", "giant"]}


def test_score_and_random_expectation():
    assert ev.score(["a", "b", "c", "d"], {"a", "d", "z"}, 4) == (pytest.approx(2 / 3), pytest.approx(1 / 3))
    assert ev.score(["x", "y", "z"], {"a"}, 3) == (0.0, 0.0)
    rec, prec = ev.random_score(candidates=20, hidden_in_cand=5, hidden=5, n=6)
    assert rec == pytest.approx(6 / 20) and prec == pytest.approx(5 / 20)
    assert ev.random_score(0, 0, 3, 5) == (0.0, 0.0)


def test_evaluate_game_on_fixture_model():
    model = fixture_model()
    res = ev.evaluate_game(model, set(DECK_A), ["giant", "balloon", "lightning"])
    assert sorted(res) == [1, 2, 3]
    assert res[2]["model"] == (1.0, 1.0)                         # A's other 6 cards fill the top 6, its top 3 are all hits
    assert res[2]["model"][0] > res[2]["popularity"][0] > 0.0
    vocab = len(model.cards)
    assert res[2]["random"][0] == pytest.approx(6 / (vocab - 2))
    assert ev.evaluate_game(model, set(DECK_A), ["giant"], ks=(1, 2, 3)).keys() == {1}      # fewer reveals than ks
    assert ev.evaluate_game(model, set(DECK_A[:2]), ["giant", "balloon"]).keys() == {1}      # k=2 leaves nothing hidden


def test_run_protocol_holds_the_game_out(tmp_path, monkeypatch):
    entries = [
        {"cards": tuple(sorted(DECK_A)), "source": "kb", "game_id": "kb-a", "archetype": None},
        {"cards": tuple(sorted(DECK_A)), "source": "opp", "game_id": "g0", "archetype": None},
        {"cards": tuple(sorted(DECK_A)), "source": "opp", "game_id": "g1", "archetype": None},
        {"cards": tuple(sorted(DECK_B)), "source": "own", "game_id": "g0", "archetype": None},
    ]
    games = {"g0": {"split": "train", "opponent_deck": DECK_A}, "g1": {"split": "test", "opponent_deck": DECK_A},
             "g2": {"split": "train", "opponent_deck": DECK_B[:3]}}
    orders = {"g0": ["giant", "balloon"], "g1": ["giant"], "g2": ["hog-rider"]}
    seen_train: dict[str, list[dict]] = {}
    real_build = ev.build_model

    def spy(train, temperature, count_power):
        seen_train[f"{len(seen_train)}"] = train
        return real_build(train, temperature, count_power)

    monkeypatch.setattr(ev, "build_model", spy)
    rows = ev.run_protocol(entries, games, orders, "logo", 0.05, 0.0)
    assert [r["game_id"] for r in rows] == ["g0", "g1"]                  # g2: fewer than 7 cards
    assert len(seen_train["0"]) == 3
    assert all(not (e["source"] == "opp" and e["game_id"] == "g0") for e in seen_train["0"])
    assert sum(e["cards"] == tuple(sorted(DECK_A)) for e in seen_train["0"]) == 2     # kb copy and g1 stay

    seen_train.clear()
    rows = ev.run_protocol(entries, games, orders, "lodo", 0.05, 0.0)
    assert len(seen_train) == 1                                          # identical decks share one model
    assert all(e["cards"] != tuple(sorted(DECK_A)) for e in seen_train["0"])
    only_test = ev.run_protocol(entries, games, orders, "logo", 0.05, 0.0, only_split="test")
    assert [r["game_id"] for r in only_test] == ["g1"]
    agg = ev.aggregate_rows(rows)
    assert agg["1"]["n"] == 2 and agg["2"]["n"] == 1 and agg["3"]["n"] == 0 and agg["3"]["model"] is None


# ---------------------------------------------------------------- real data
@pytest.mark.skipif(not REAL_GAMES.exists() or not (KB / "decks").exists(), reason="dataset/v1/games.jsonl missing")
def test_real_data_deck_pool_and_prediction():
    from tools.data.kb_markdown import load_deck_registry
    registry = load_deck_registry()
    kb_decks = {d["deck_key"]: d["cards"] for d in bdc.load_kb_decks()}
    assert kb_decks == {k: v["cards"] for k, v in registry.items()}
    entries = bdc.collect_entries(DATASET_DIR, KB, bdc.known_card_slugs())
    model = bdc.build_model(entries)
    assert len(kb_decks) == 68 and len(model.decks) > 150 and len(entries) > len(model.decks)
    assert all(len(d) in (7, 8) for d in model.decks)
    preds = model.predict(["giant", "balloon"])
    assert len(preds) == 8 and not {c for c, _ in preds} & {"giant", "balloon"}
    assert sum(p for _, p in model.predict(["giant", "balloon"], top=len(model.cards))) == pytest.approx(6.0, abs=1e-3)


@pytest.mark.skipif(not REAL_GAMES.exists() or not REAL_STATES.exists(), reason="dataset/v1 missing")
def test_real_data_reveal_orders_cover_most_games():
    games = {}
    for line in REAL_GAMES.read_text().splitlines():
        g = json.loads(line)
        games[g["game_id"]] = set(g["opponent_deck"])
    orders = ev.reveal_orders(REAL_STATES, games)
    assert len(orders) > 150
    assert all(set(o) <= games[gid] and len(set(o)) == len(o) for gid, o in orders.items())

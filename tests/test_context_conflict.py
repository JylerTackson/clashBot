"""MAC-AR R2: context-conflict test of the d1 ranker - perturbations, follow-rate, statistics (fake ranker, toy lines)."""
from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clashbot.d1_format import JOINT_NAME, NOTE_SEP, joint_request, option_description  # noqa: E402
from clashbot.ranker import D1Ranker  # noqa: E402
from tools.eval import context_conflict as C  # noqa: E402

FILLER = "filler: no pro or counter support"
GOOD = "tactics suggests this card (defend: tanks, trade +1)"
KEYS = ["berserker|left|mid", "berserker|right|mid", "archers|left|mid", "tornado|left|mid"]
BODY = ("[single_elixir|2:48] elixir 10 (opp ~6.31). hand: tornado, archers, berserker, ?\nfield: none\nthreats: none\n"
        "recent: none\ntowers own 4858/4858/?, enemy 4858/4858/?")


def make_line(i=0, gold=2, notes=None, best="archers", game="g1", block_extra=("KG: royal-ghost L15 answers archers",)):
    notes = notes if notes is not None else {KEYS[gold]: GOOD}
    block = [f"tactics: best {best} @[5,2.5] leak: elixir 10 near full", *block_extra] if best else list(block_extra)
    state, questions = joint_request(BODY + ("\n" + "\n".join(block) if block else ""), KEYS,
                                     notes={k: notes.get(k, FILLER) for k in KEYS})
    return {"id": f"{game}#{i}", "game_id": game, "split": "val", "kind": "play", "variant": "joint", "state": state,
            "questions": questions, "answers": {JOINT_NAME: KEYS[gold]}}


def crit_notes(line) -> list:
    return [C.split_option(t)[1] for t in line["questions"][JOINT_NAME]["criteria"].values()]


# ---------------------------------------------------------------------------------------------- parsing
def test_option_and_state_round_trip():
    line = make_line()
    p = C.parse_line(line)
    assert p["keys"] == KEYS and p["gold"] == 2
    assert p["descs"] == [option_description(k) for k in KEYS]
    assert p["notes"][2] == GOOD and p["notes"][0] == FILLER
    assert C.join_option(*C.split_option(f"abc{NOTE_SEP}x")) == f"abc{NOTE_SEP}x"
    assert C.split_option("abc") == ("abc", None)
    assert C.join_state(p["body"], p["block"]) == line["state"]
    assert p["block"][0].startswith("tactics: best archers") and p["block"][1].startswith("KG:")
    assert C.best_card(p["block"]) == "archers" and C.best_card(["tactics: hold: nothing"]) is None


def test_context_status_labels():
    assert C.note_status(C.parse_line(make_line(gold=2))) == "right"
    assert C.note_status(C.parse_line(make_line(gold=2, notes={KEYS[0]: GOOD}))) == "wrong"
    assert C.note_status(C.parse_line(make_line(gold=2, notes={KEYS[2]: FILLER}))) == "none"
    assert C.tactics_status(C.parse_line(make_line(best="archers"))) == "names the gold card"
    assert C.tactics_status(C.parse_line(make_line(best="tornado"))) == "names another card"
    assert C.tactics_status(C.parse_line(make_line(best=None))) == "no best line"
    assert C.informative(GOOD) and not C.informative(FILLER) and not C.informative(None)


# ---------------------------------------------------------------------------------------------- perturbations
def test_c0_is_the_export_and_c1_strips_everything():
    line = make_line()
    v0 = C.perturb(line, "C0")
    assert v0["state"] == line["state"] and v0["criteria"] == line["questions"][JOINT_NAME]["criteria"]
    assert v0["changed"] is False and v0["fav"] == [2]
    v1 = C.perturb(line, "C1")
    assert v1["criteria"] == {k: option_description(k) for k in KEYS}          # no notes at all
    assert v1["state"] == BODY and "tactics:" not in v1["state"] and "KG:" not in v1["state"]
    assert v1["changed"] is True and v1["fav"] == []


def test_c2_shuffles_notes_across_options_deterministically():
    line = make_line(notes={KEYS[2]: GOOD, KEYS[0]: "tactics: defend, x"})
    v = C.perturb(line, "C2", seed=3)
    assert sorted(map(str, crit_notes({"questions": {JOINT_NAME: {"criteria": v["criteria"]}}}))) == \
        sorted(map(str, crit_notes(line)))                                  # the same notes, different places
    assert v["changed"] and v["state"] == line["state"]                      # state lines stay
    assert C.perturb(line, "C2", seed=3) == v
    assert any(C.perturb(line, "C2", seed=s)["criteria"] != v["criteria"] for s in range(4, 12))
    assert v["fav"] == C.fav_of_notes([C.split_option(t)[1] for t in v["criteria"].values()])
    same = make_line(notes={})                                                # four identical fillers: nothing to shuffle
    assert C.perturb(same, "C2")["changed"] is False


def test_c3_moves_the_gold_note_to_a_distractor_and_names_a_wrong_card():
    line = make_line(gold=2, best="archers")
    v = C.perturb(line, "C3", seed=1)
    d = v["info"]["distractor"]
    assert d != 2 and C.option_card(KEYS[d]) != "archers"                    # a distractor with another card
    notes = [C.split_option(t)[1] for t in v["criteria"].values()]
    assert notes[2] == FILLER and notes[d] == GOOD and v["info"]["swapped"]
    assert f"tactics: best {C.option_card(KEYS[d])} @[5,2.5]" in v["state"] and "best archers" not in v["state"]
    assert v["fav"] == [d] and v["changed"]
    assert [l for l in v["state"].split("\n") if l.startswith("KG:")] == [l for l in line["state"].split("\n")
                                                                          if l.startswith("KG:")]
    assert C.perturb(line, "C3", seed=1) == v


def test_c3_when_the_gold_note_is_already_filler_only_the_tactics_card_changes():
    line = make_line(gold=2, notes={KEYS[3]: GOOD}, best="archers")
    v = C.perturb(line, "C3", seed=0)
    assert v["fav"] == [3] and "swapped" not in v["info"]                    # the notes already point at the tornado
    assert v["criteria"] == line["questions"][JOINT_NAME]["criteria"]
    assert "tactics: best tornado @[5,2.5]" in v["state"] and v["changed"]
    nothing = C.perturb(make_line(gold=2, notes={}, best=None), "C3")         # no best line, nothing to swap: untouched
    assert nothing["changed"] is False


def test_c4_transplants_a_block_from_another_game_only():
    line = make_line(game="g1", best="archers")
    donor = make_line(i=5, game="g2", best="tornado", block_extra=("KG: donor line",))
    own = make_line(i=6, game="g1", best="berserker", block_extra=("KG: own game",))
    pool = C.make_pool([line, donor, own])
    v = C.perturb(line, "C4", seed=0, pool=pool)
    assert v["state"].split("\n")[:5] == BODY.split("\n")                      # body unchanged
    assert [l for l in v["state"].split("\n") if l.startswith(C.BLOCK_PREFIXES)] == donor["state"].split("\n")[-2:]
    assert "own game" not in v["state"] and v["criteria"] == line["questions"][JOINT_NAME]["criteria"]
    assert v["fav"] == [3] and v["info"]["card"] == "tornado"
    assert C.perturb(line, "C4", seed=0, pool=pool) == v


def test_gold_label_never_changes_and_inputs_are_not_mutated():
    lines = [make_line(i=i, gold=i % 4, notes={KEYS[(i + 1) % 4]: GOOD} if i % 2 else {KEYS[i % 4]: GOOD},
                       best=("archers", "tornado", None)[i % 3], game=f"g{i % 3}") for i in range(24)]
    pool = C.make_pool(lines)
    for line in lines:
        before = copy.deepcopy(line)
        for cond in C.CONDITIONS:
            v = C.perturb(line, cond, seed=7, pool=pool)
            assert list(v["criteria"]) == KEYS                                  # same option keys, same order
            assert line == before                                               # the answers and the line are untouched
            assert KEYS[C.parse_line(line)["gold"]] == line["answers"][JOINT_NAME]
            for k in KEYS:                                                       # descriptions never edited
                assert C.split_option(v["criteria"][k])[0] == option_description(k)
    with pytest.raises(ValueError):
        C.perturb(lines[0], "C9")


def test_seed_changes_random_choices():
    line = make_line(gold=2)
    picks = {C.perturb(line, "C3", seed=s)["info"]["distractor"] for s in range(30)}
    assert picks <= {0, 1, 3} and len(picks) > 1


# ---------------------------------------------------------------------------------------------- following the context
def fake_predict(state, questions):
    """A model that follows the context: +2 logit for an informative note, +1 for the card of the `tactics: best` line."""
    crit = questions[JOINT_NAME]["criteria"]
    best = C.best_card(C.split_state(state)[1])
    z = [2.0 * C.informative(C.split_option(t)[1]) + 1.0 * (best is not None and C.option_card(k) == best)
         for k, t in crit.items()]
    e = np.exp(np.array(z) - max(z))
    return {"answers": {JOINT_NAME: {"probabilities": list(e / e.sum())}}}


def ranker():
    return D1Ranker(model_path="fake", backend="transformers", predict_fn=fake_predict)


def test_run_condition_with_the_ranker_class_and_follow_rate():
    lines = [make_line(i=i, gold=2, game=f"g{i % 3}") for i in range(9)]
    pool = C.make_pool(lines)
    rows = {c: C.run_condition(ranker(), lines, c, 0, pool) for c in C.CONDITIONS}
    assert all(len(r) == 9 and abs(sum(x["probs"]) - 1) < 1e-9 for r in rows.values() for x in r)
    gold = np.array([2] * 9)
    pick = {c: np.array([int(np.argmax(x["probs"])) for x in rows[c]]) for c in rows}
    assert (pick["C0"] == gold).all()                                           # notes + tactics agree on the gold option
    assert (pick["C1"] == 0).all()                                              # no context: ties go to the first option
    assert (pick["C3"] != gold).all()                                           # the poisoned context moves the pick
    f = C.follow_stats(pick["C0"], pick["C3"], [x["fav"] for x in rows["C3"]], [x["changed"] for x in rows["C3"]],
                       [x["fav"] for x in rows["C0"]])
    assert f["active_lines"] == f["movable_lines"] == f["moved"] == 9 and f["follow_rate"] == 1.0
    assert f["pick_on_favoured"] == 1.0 and f["c0_pick_on_favoured"] == 0.0


def test_follow_stats_counts_toy_cases():
    pick0 = np.array([0, 1, 2, 3, 0])
    pickx = np.array([1, 1, 3, 0, 0])
    fav = [[1], [1], [3], [2], []]                       # line 1: C0 already on the favoured option; line 4: no favoured
    changed = [True, True, True, False, True]            # line 3 unchanged
    fav0 = [[0], [0], [0], [2], []]
    f = C.follow_stats(pick0, pickx, fav, changed, fav0)
    assert f["active_lines"] == 3 and f["movable_lines"] == 2 and f["moved"] == 2      # lines 0, 2 move; line 1 not movable
    assert f["follow_rate"] == 1.0 and f["pick_on_favoured"] == 1.0
    assert C.follow_stats(pick0, pick0, fav, changed, fav0)["follow_rate"] == 0.0
    assert C.follow_stats(pick0, pickx, [[]] * 5, changed, fav0)["follow_rate"] is None


# ---------------------------------------------------------------------------------------------- statistics
def test_mcnemar_exact():
    a = np.array([1] * 5 + [0] * 5, bool)
    b = np.array([0] * 5 + [0] * 5, bool)                # a right and b wrong five times, never the reverse
    r = C.mcnemar_exact(a, b)
    assert (r["a_only"], r["b_only"]) == (5, 0) and r["p"] == pytest.approx(2 / 32)
    assert C.mcnemar_exact(a, ~a)["p"] == 1.0            # 5 vs 5
    assert C.mcnemar_exact(a, a)["p"] == 1.0


def test_ece_and_jsd_and_cad():
    probs = np.array([[0.9, 0.1]] * 10)
    correct = np.array([1.0] * 5 + [0.0] * 5)
    assert C.ece(probs, correct) == pytest.approx(0.4)
    assert C.ece(np.array([[1.0, 0.0]] * 4), np.ones(4)) == pytest.approx(0.0)
    p = np.array([[0.7, 0.3], [1.0, 0.0]])
    assert np.allclose(C.jsd_bits(p, p), 0.0, atol=1e-6)
    assert C.jsd_bits(np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]]))[0] == pytest.approx(1.0, abs=1e-6)
    ctx, free = np.array([[0.45, 0.55]]), np.array([[0.2, 0.8]])
    assert C.cad_scores(ctx, free, 0.0).argmax(1)[0] == 1                       # plain: the wrong option
    assert C.cad_scores(ctx, free, 1.0).argmax(1)[0] == 0                       # a = 1 contrasts the context's bias away
    assert np.allclose(C.cad_scores(ctx, free, 0.0), np.log(ctx))
    assert np.allclose(C.cad_scores(ctx, free, 1.0), 2 * np.log(ctx) - np.log(free))
    per_row = C.cad_scores(np.vstack([ctx, ctx]), np.vstack([free, free]), np.array([0.0, 1.0]))
    assert per_row.argmax(1).tolist() == [1, 0]                                 # one alpha per line


# ---------------------------------------------------------------------------------------------- whole report on toy files
def toy_dataset(tmp_path):
    ds = tmp_path / "ds"
    for split in C.SPLITS:
        lines = []
        for i in range(12):
            gold = i % 4
            notes = {KEYS[gold]: GOOD} if i % 3 == 0 else {KEYS[(gold + 1) % 4]: GOOD} if i % 3 == 1 else {}
            best = C.option_card(KEYS[gold]) if i % 2 == 0 else "tornado"
            line = make_line(i=i, gold=gold, notes=notes, best=best, game=f"{split}-g{i % 3}")
            line["split"] = split
            lines.append(line)
        out = ds / "d1" / "joint_tac" / f"{split}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("".join(json.dumps(l) + "\n" for l in lines))
    return ds


def test_report_on_cached_probabilities(tmp_path):
    ds, cache = toy_dataset(tmp_path), tmp_path / "cache"
    for split in C.SPLITS:
        lines = C.load_lines(ds, "joint_tac", split)
        pool = C.make_pool(lines)
        for cond in C.CONDITIONS:
            rows = C.run_condition(ranker(), lines, cond, 0, pool)
            path = C.cache_path(cache, "joint_tac", split, cond)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"rows": rows}))
    rep = C.build_report(cache, "joint_tac", ds, boot_n=200, seed=0)
    assert rep["runs"]["val"]["lines"] == 12 and rep["runs"]["test"]["games"] == 3
    val = rep["splits"]["val"]
    c0 = val["conditions"]["C0"]["acc"]
    assert c0["n"] == 12 and 0.0 <= c0["lo"] <= c0["mean"] <= c0["hi"] <= 1.0
    assert val["conditions"]["C1"]["vs_c0"]["pick_changed"] >= 0.0 and "p" in val["conditions"]["C1"]["vs_c0"]
    assert set(val["strata"]["note"]) == {"right", "wrong", "none"} and sum(r["n"] for r in val["strata"]["note"].values()) == 12
    # a model that follows its notes is right when the gold note is right and wrong when the notes point elsewhere
    assert val["strata"]["note"]["wrong"]["C0"]["mean"] == 0.0
    assert val["strata"]["note"]["right"]["C0"]["mean"] == 1.0
    assert val["follow"]["C3"]["follow_rate"] is not None and "val+test" in rep["splits"]
    assert set(rep["cad"]["C0"]) >= {"tuned_on_val", "val", "test"} and "adacad" in rep["cad"]["C0"]["test"]
    md = C.render_markdown(rep)
    assert "Accuracy by condition - val" in md and "Follow-rate" in md and "Contrastive decoding" in md


def test_cache_ids_must_match_the_split(tmp_path):
    ds, cache = toy_dataset(tmp_path), tmp_path / "cache"
    lines = C.load_lines(ds, "joint_tac", "val")
    rows = C.run_condition(ranker(), lines, "C0", 0, C.make_pool(lines))
    rows[0]["id"] = "something-else"
    path = C.cache_path(cache, "joint_tac", "val", "C0")
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"rows": rows}))
    with pytest.raises(ValueError):
        C.load_cells(cache, "joint_tac", "val", lines, ["C0"])


@pytest.mark.skipif(not (ROOT / "dataset" / "v1" / "d1" / "joint_tac" / "val.jsonl").exists(), reason="joint_tac not built")
def test_real_export_parses_and_perturbs_without_touching_gold():
    lines = C.load_lines(C.DATASET_DIR, "joint_tac", "val", limit=120)
    pool = C.make_pool(lines)
    for line in lines:
        p = C.parse_line(line)
        assert p["descs"] == [option_description(k) for k in p["keys"]]
        for cond in C.CONDITIONS:
            v = C.perturb(line, cond, 0, pool)
            assert list(v["criteria"]) == p["keys"] and "<|" not in v["state"]

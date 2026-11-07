"""MAC-AQ: the bot-vs-pro evaluation on hand-made toys with known answers (no models, no video decoding beyond a tiny synthetic
file); the dataset checks are skipped when `dataset/` is missing."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clashbot.config import DATASET_DIR  # noqa: E402
from tools.eval import eval_vs_pro as E  # noqa: E402

CARDS = {"knight": {"elixir_cost": 3.0}, "zap": {"elixir_cost": 2.0}, "archers": {"elixir_cost": 3.0}, "giant": {"elixir_cost": 5.0}}
PRIOR = {"by_key": {"knight|left|mid": {"count": 5}, "knight|middle|back": {"count": 2}, "zap|right|enemy_half": {"count": 3},
                    "archers|middle|back": {"count": 4}, "giant|middle|back": {"count": 1}}}
needs_dataset = pytest.mark.skipif(not (DATASET_DIR / "games.jsonl").exists(), reason="dataset/v1 missing")


def kg_scores(threat):
    return {"archers": 1.0, "zap": 0.0} if threat == "bomber" else {}


def baselines():
    return E.Baselines(PRIOR, kg_scores)


# ------------------------------------------------------------------------------------------------ the toy game
def toy_meta(game_id="g1"):
    return {"game_id": game_id, "video_id": "vid", "minutes": 5.0, "start": 0.0, "end": 300.0, "result": "win"}


def toy_pro():
    def p(t, card, tile, conf, hand, threats, ok=True):
        lane, zone = (E.lane_of(tile), E.zone_of(tile)) if tile else (None, None)
        key = f"{card}|{lane}|{zone}" if card and tile else None
        return {"t": t, "card": card, "tile": tile, "lane": lane, "zone": zone, "key": key, "confidence": conf, "source": "row",
                "card_ok": ok, "hand": hand, "threats": threats}
    return [p(10.0, "knight", [3, 9], "high", ["knight", "zap", "archers", "giant"], []),
            p(30.0, "zap", [14, 20], "medium", ["zap", "archers", None, None], ["bomber"]),
            p(50.0, None, None, "high", ["knight", None, None, None], [], ok=False),
            p(100.0, "archers", [9, 2], "low", ["archers", "giant", None, None], []),
            p(200.0, "giant", [8, 4], "medium", ["giant", None, None, None], [])]


def toy_decisions():
    def d(t, key, tile, mode, status=None, idx=0, probs=None, cands=None):
        ex = {"t": t, "key": key, "tile": tile}
        if status:
            ex["status"] = status
        return {"t": t, "idx": idx, "key": key, "tile": tile, "mode": mode, "reason": f"{mode} reason", "executor": ex,
                "candidates": [{"key": k} for k in (cands or [key])], "probs": probs or [1.0],
                "hand": ["archers", "giant", None, None] if idx == 5 else ["knight", "zap", None, None]}
    return [d(10.5, "knight|left|mid", [4, 9], "defend", idx=1),                                  # pairs with the pro's knight
            d(31.5, "archers|right|enemy_half", [14, 20], "punish", idx=2, probs=[0.4, 0.6],
              cands=["zap|right|enemy_half", "archers|right|enemy_half"]),                        # another card, same lane / zone
            d(51.0, "zap|left|mid", [3, 9], "defend", idx=3),                                     # pairs with a play of unknown card
            d(70.0, "knight|left|mid", [3, 9], "defend", idx=4),                                  # no pro play near
            d(99.0, "archers|middle|back", [9, 2], "counter", "planned", idx=5),                  # pairs with the pro's archers
            d(100.5, "hold", None, "punish", "hold_policy", idx=6),                               # not a tap
            d(101.0, "ability:musketeer|left|mid", None, "defend", "planned_ability", idx=7)]    # the hero ability: counted apart


def toy_holds_frames():
    holds = [{"t": 9.0, "mode": "none", "act": False, "reason": "no affordable card", "hand": ["knight", "zap", "archers", "giant"]},
             {"t": 29.0, "mode": "none", "act": False, "reason": "no affordable card", "hand": ["archers", "giant", None, None]},
             {"t": 198.5, "mode": "save", "act": False, "reason": "no adequate answer in hand for baby-dragon left",
              "hand": ["giant", None, None, None]}]
    frames = [{"t": float(t), "readiness": "match"} for t in range(0, 301)]
    return holds, frames


def toy_record(game_id="g1", shots=True):
    holds, frames = toy_holds_frames()
    index = {i: [f"d{i:04d}_a.jpg"] for i in range(1, 8)} if shots else None
    return E.score_game(toy_meta(game_id), toy_pro(), toy_decisions(), holds, frames, baselines(), index, CARDS)


def rate(rec, name):
    return E.pooled([rec], name)


# ------------------------------------------------------------------------------------------------ matching
def test_pair_up_is_one_to_one_nearest_first():
    # bot 10.0 and 10.4 both want the pro at 10.2; the nearer (10.0, |0.2|) wins, 10.4 pairs with nothing else within 2 s
    assert E.pair_up([10.0, 10.4], [10.2], 2.0) == [(0, 0)]
    assert E.pair_up([10.0, 10.4], [10.2, 12.3], 2.0) == [(0, 0), (1, 1)]
    assert E.pair_up([0.0], [3.0], 2.0) == []
    assert E.pair_up([0.0], [2.0], 2.0) == [(0, 0)]                      # the window is inclusive


def test_covered_windows_and_shift():
    assert E.covered([10.0, 20.0], [11.0, 25.0], 1.0) == [True, False]
    assert E.covered([10.0], [12.4], 2.0) == [False]
    assert E.covered([10.0], [12.4], 2.0, shift=-0.5) == [True]           # y shifted by -0.5 s -> 11.9


def test_bot_acts_keeps_taps_only():
    acts, n_ability = E.bot_acts(toy_decisions())
    assert [a["key"] for a in acts] == ["knight|left|mid", "archers|right|enemy_half", "zap|left|mid", "knight|left|mid", "archers|middle|back"]
    assert n_ability == 1
    assert acts[1]["cands"] == [("zap|right|enemy_half", 0.4), ("archers|right|enemy_half", 0.6)]
    gated = [{"t": 1.0, "key": "zap|left|mid", "tile": None, "mode": "defend", "executor": {"status": "gated"}}]
    assert E.bot_acts(gated)[0] == []


def test_bucket_reason_drops_cards_and_numbers():
    assert E.bucket_reason("no adequate answer in hand for baby-dragon left") == "no adequate answer in hand"
    assert E.bucket_reason("fast wait p=0.32 h=2s") == "act model: wait (p below threshold)"
    assert E.bucket_reason("tower handles skeletons right") == "the tower handles it"
    assert E.bucket_reason("push right answered by musketeer") == "push already answered"
    assert E.bucket_reason("something 12 new 3.5 odd words") == "something # new #"


# ------------------------------------------------------------------------------------------------ the toy game's metrics
def test_timing_metrics_have_known_answers():
    rec = toy_record()
    c = rec["counts"]
    assert (c["n_pro"], c["n_bot"], c["n_ability"]) == (5, 5, 1)
    assert rate(rec, "recall_1s") == pytest.approx(3 / 5)               # pro plays at 10 (0.5 s), 50 (1.0 s), 100 (1.0 s)
    assert rate(rec, "recall_2s") == pytest.approx(4 / 5)               # + the play at 30 (1.5 s); the one at 200 is missed
    assert rate(rec, "recall_3s") == pytest.approx(4 / 5)
    assert rate(rec, "precision_2s") == pytest.approx(4 / 5)            # the tap at 70 has no pro play near
    assert rate(rec, "bot_acts_per_min") == pytest.approx(1.0) and rate(rec, "pro_plays_per_min") == pytest.approx(1.0)
    assert rate(rec, "bot_earlier") == pytest.approx(1 / 4)              # dt = bot - pro: +0.5, +1.5, +1.0, -1.0
    assert E.median_abs_dt([rec]) == pytest.approx(1.0) and E.median_dt([rec]) == pytest.approx(0.75)
    assert rate(rec, "bot_elixir_per_min") == pytest.approx(14 / 5)      # knight 3 + archers 3 + zap 2 + knight 3 + archers 3
    assert rate(rec, "pro_elixir_per_min") == pytest.approx(13 / 5)      # only plays whose card is in the deck
    assert rate(rec, "recall_2s_pro_shift") is not None and 0.0 <= rate(rec, "null_recall_2s") <= 1.0
    assert 0.0 <= rate(rec, "null_precision_2s") <= 1.0


def test_choice_metrics_have_known_answers():
    rec = toy_record()
    assert len(rec["pairs"]) == 4
    assert rate(rec, "same_card") == pytest.approx(2 / 3)               # the pair with an unknown pro card is not scored
    assert rate(rec, "same_lane") == pytest.approx(3 / 3)
    assert rate(rec, "same_zone") == pytest.approx(3 / 3)
    assert rate(rec, "same_key") == pytest.approx(2 / 3)
    assert rate(rec, "same_key_given_card") == pytest.approx(2 / 2)
    assert rate(rec, "tile_dist_given_card") == pytest.approx(0.5)      # 1.0 and 0.0 tiles
    assert rate(rec, "within_2_tiles_given_card") == pytest.approx(1.0)
    by_game = {(p["bot_key"], p["pro_key"]): p for p in rec["pairs"]}
    assert by_game[("knight|left|mid", "knight|left|mid")]["tile_dist"] == pytest.approx(1.0)
    assert by_game[("archers|right|enemy_half", "zap|right|enemy_half")]["same_card"] is False


def test_breakdowns_by_mode_and_confidence():
    c = toy_record()["counts"]
    assert (c["mode.defend.acts"], c["mode.defend.precise"], c["mode.defend.pairs"]) == (3, 2, 2)
    assert (c["mode.punish.acts"], c["mode.counter.acts"]) == (1, 1)
    assert (c["mode.defend.same_card.num"], c["mode.defend.same_card.den"]) == (1, 1)     # the unknown-card pair has no den
    assert (c["conf.high.pro"], c["conf.high.recalled"]) == (2, 2)
    assert (c["conf.medium.pro"], c["conf.medium.recalled"]) == (2, 1)
    assert c["conf.low.pairs"] == 1


def test_baselines_at_the_pro_plays():
    c = toy_record()["counts"]

    def b(scope, name):
        return c[f"base_{scope}.{name}.num"], c[f"base_{scope}.{name}.den"]
    # the bot's perceived hand at the pro plays: 10.0 [knight zap archers giant] (row at 9.0), 30.0 [archers giant] (29.0), 100.0 [archers giant]
    # (the tap at 99.0), 200.0 [giant] (198.5); 1 / known hand size when the card is in it, else 0
    assert b("all", "rand_hand") == pytest.approx((0.25 + 0.0 + 0.5 + 1.0, 4))
    assert b("pair", "rand_hand") == pytest.approx((0.25 + 0.0 + 0.5, 3))
    # the dataset's hand (the pro rows' own): [knight zap archers giant], [zap archers], [archers giant], [giant]
    assert b("all", "rand_hand_ds") == pytest.approx((0.25 + 0.5 + 0.5 + 1.0, 4))
    assert b("pair", "rand_hand_ds") == pytest.approx((0.25 + 0.5 + 0.5, 3))
    assert b("all", "kg_first") == (0.0, 1) and b("pair", "kg_first") == (0.0, 1)    # bomber -> archers first, the pro played zap
    assert b("all", "kg_first_ds") == (0.0, 1)
    assert (c["pro_in_bot_hand.num"], c["pro_in_bot_hand.den"]) == (3, 4)            # the bot's hand at 30.0 lacks the pro's zap
    assert (c["pro_in_dataset_hand.num"], c["pro_in_dataset_hand.den"]) == (4, 4)
    assert c["bot_on_kg_subset.den"] == 1 and c["bot_on_kg_subset.num"] == 0
    assert b("all", "prior_global_key") == (1.0, 4) and b("pair", "prior_global_key") == (1.0, 3)
    assert b("all", "prior_key") == (4.0, 4)                                          # each card's most common key is the pro's key here
    assert c["base_pair_given_card.prior_lane.den"] == 2                              # the two same-card pairs


def test_baselines_for_unscorable_plays():
    bl = baselines()
    out = bl.for_play({"card": None, "card_ok": False, "hand": ["zap"], "threats": ["bomber"], "key": None})
    assert all(v is None for v in out.values())
    out = bl.for_play({"card": "zap", "card_ok": True, "hand": [None, None, None, None], "threats": [], "key": "zap|left|mid",
                       "lane": "left", "zone": "mid"})
    assert out["rand_hand"] is None and out["kg_first"] is None and out["prior_lane"] == 0.0 and out["prior_key"] == 0.0


def test_bot_hand_at_is_the_latest_row_not_older_than_three_seconds():
    rows = [{"t": 5.0, "hand": ["zap", None, "knight", None]}, {"t": 8.0, "hand": [None, None, None, None]}, {"t": 20.0, "hand": ["giant"]}]
    assert E.bot_hand_at(5.0, rows) == ["zap", "knight"]          # at or before t
    assert E.bot_hand_at(7.9, rows) == ["zap", "knight"]
    assert E.bot_hand_at(8.5, rows) == []                         # a row with nothing read is a perceived (empty) hand
    assert E.bot_hand_at(4.9, rows) is None                       # nothing yet
    assert E.bot_hand_at(12.0, rows) is None                      # the latest row is 4 s old
    assert E.bot_hand_at(22.9, rows) == ["giant"] and E.bot_hand_at(23.1, rows) is None


def test_miss_reasons_say_what_the_bot_was_doing():
    rec = toy_record()
    assert rec["miss_reasons"] == {"save: no adequate answer in hand": 1}
    holds, frames = toy_holds_frames()
    decs = toy_decisions()
    assert E.miss_reason(200.0, holds, decs, [{"t": 199.0, "readiness": "menu"}]) == "not in a match frame (readiness menu)"
    assert E.miss_reason(100.8, holds, decs, frames).startswith("decision held: hold_policy")
    assert E.miss_reason(101.0, [], [], frames).startswith("no hold line")
    assert E.miss_reason(5.0, [], [], []) == "no frame yet"
    act_line = [{"t": 199.0, "mode": "defend", "act": True, "reason": "defend x"}]
    assert E.miss_reason(200.0, act_line, [], frames) == "policy said act (defend) but no tap followed"


# ------------------------------------------------------------------------------------------------ pooling, CIs, examples, report
def test_aggregate_pools_games_and_bootstraps_over_games():
    recs = [toy_record("g1"), toy_record("g2"), toy_record("g3")]
    agg = E.aggregate(recs, n_boot=200)
    m = agg["metrics"]
    assert agg["n_games"] == 3
    assert m["recall_2s"]["value"] == pytest.approx(4 / 5) and m["recall_2s"]["n"] == 15
    assert m["same_card"]["value"] == pytest.approx(2 / 3)
    lo, hi = m["recall_2s"]["ci95"]
    assert lo is not None and lo <= m["recall_2s"]["value"] <= hi          # identical games: the interval collapses onto the value
    assert (lo, hi) == (pytest.approx(0.8), pytest.approx(0.8))
    assert agg["baselines"]["pair"]["rand_hand"]["value"] == pytest.approx((0.25 + 0.0 + 0.5) / 3)
    assert agg["baselines"]["pair"]["rand_hand_ds"]["value"] == pytest.approx((0.25 + 0.5 + 0.5) / 3)
    assert agg["metrics"]["pro_in_bot_hand"]["value"] == pytest.approx(3 / 4)
    assert agg["by_mode"]["defend"]["acts"] == 9 and agg["by_mode"]["defend"]["precision_2s"] == pytest.approx(2 / 3)
    assert agg["by_pro_confidence"]["high"]["recall_2s"] == pytest.approx(1.0)
    assert agg["miss_reasons"][0] == {"reason": "save: no adequate answer in hand", "n": 3, "share": 1.0}
    assert [g["game_id"] for g in agg["per_game"]] == ["g1", "g2", "g3"]
    assert E.bootstrap_ci(recs[:1], lambda s: 1.0) == (None, None)        # one game has no interval


def test_bootstrap_interval_is_wide_when_games_differ():
    good, bad = toy_record("g1"), toy_record("g2")
    bad["counts"] = {k: (0.0 if k.startswith("recall_2s.num") else v) for k, v in bad["counts"].items()}
    lo, hi = E.bootstrap_ci([good, bad], lambda s: E.pooled(s, "recall_2s"), n=500)
    assert lo == pytest.approx(0.0) and hi == pytest.approx(0.8)
    assert E.bootstrap_ci([good, bad], lambda s: E.pooled(s, "recall_2s"), n=500) == (lo, hi)     # seeded: repeatable


def test_pick_examples_agree_and_disagree_from_different_games():
    recs = [toy_record("g1"), toy_record("g2"), toy_record("g3")]
    ex = E.pick_examples(recs, n_agree=2, n_disagree=1)
    assert [e["verdict"] for e in ex] == ["agree", "agree", "disagree"]
    assert len({e["game_id"] for e in ex}) == 3
    assert all(e["same_card"] and e["tile_dist"] <= 2.0 for e in ex[:2])
    assert not ex[2]["same_card"] and ex[2]["shots"]
    assert E.pick_examples([toy_record("g1", shots=False)]) == []          # an example needs its screenshot


def test_report_renders_every_section():
    recs = [toy_record("g1"), toy_record("g2")]
    report = {"meta": {"generated_at": "now", "head": "abc1234567", "decision_code_changed": [], "pro_minutes": 10.0, "runs_dir": "runs/eval/vs_pro",
                       "games": [{"game_id": "g1", "minutes": 5.0, "wall_s": 400.0, "frames": 1, "decisions": 2, "calib": "c.json",
                                  "hero_cards": ["knight"], "evo_cards": []}],
                       "skipped": [{"game_id": "x", "skip_reason": "deck invalid"}]},
              "aggregate": E.aggregate(recs, n_boot=50), "examples": E.pick_examples(recs + [toy_record("g3")])}
    md = E.render_md(report)
    for needle in ("## Headline", "## Timing", "## Baselines at the pro plays", "## By the bot's mode", "## By game", "## Examples for the slides",
                   "## How to read this", "same taps shifted", "Not run: x (deck invalid)", "runs/eval/vs_pro/g1/shots/d0001_a.jpg"):
        assert needle in md, needle
    assert json.loads(json.dumps(report, default=str))["aggregate"]["n_games"] == 2


# ------------------------------------------------------------------------------------------------ pro plays and the time mapping
def row(rid, t, kind, action, hand=None, threats=None):
    return {"id": rid, "kind": kind, "t": t, "action": action,
            "state": {"own": {"hand": hand or ["knight", "zap", None, None], "hero_cards": []}, "threats": threats or []}}


def test_pro_plays_time_is_state_time_plus_delay_and_dedups_following_plays():
    rows = [row("a#6.0", 6.0, "play", {"type": "play", "card": "knight", "tile": [3, 9], "lane": "left", "zone": "mid",
                                        "key": "knight|left|mid", "delay_s": 1.01, "confidence": "high",
                                        "following_plays": [{"t": 7.007, "player": "own", "card": "knight", "tile": [3, 9], "confidence": "high"},
                                                            {"t": 11.845, "player": "own", "card": "zap", "tile": [14, 20], "confidence": "medium"},
                                                            {"t": 12.0, "player": "opponent", "card": "giant", "tile": [3, 20]}]}),
            row("a#10.8", 10.8, "play", {"type": "unknown", "card": None, "tile": None, "delay_s": 3.2, "confidence": "high", "following_plays": []}),
            row("a#20.0", 20.0, "play", {"type": "play", "card": "goblins", "tile": [8, 4], "lane": "middle", "zone": "back",
                                         "key": "goblins|middle|back", "delay_s": 1.0, "confidence": "medium", "following_plays": []}),
            row("a#30.0", 30.0, "periodic", {"type": "hold", "card": None, "delay_s": None, "following_plays": []}),
            row("a#500.0", 500.0, "play", {"type": "play", "card": "zap", "tile": [1, 1], "lane": "left", "zone": "back",
                                           "key": "zap|left|back", "delay_s": 1.0, "confidence": "high", "following_plays": []})]
    plays = E.pro_plays(rows, ["knight", "zap"], start=0.0, end=100.0)
    assert [(p["t"], p["card"]) for p in plays] == [(7.01, "knight"), (11.845, "zap"), (14.0, None), (21.0, "goblins")]
    assert plays[0]["source"] == "row" and plays[0]["key"] == "knight|left|mid" and plays[0]["hand"] == ["knight", "zap", None, None]
    assert plays[1]["source"] == "following" and plays[1]["key"] == "zap|right|enemy_half" and plays[1]["hand"] is None
    assert [p["card_ok"] for p in plays] == [True, True, False, False]       # unknown card and a card outside the deck: timing only


def test_refresh_pro_plays_rewrites_from_the_dataset(tmp_path):
    ds = tmp_path / "ds"
    ds.mkdir()
    r = row("g1#6.0", 6.0, "play", {"type": "play", "card": "knight", "tile": [3, 9], "lane": "left", "zone": "mid", "key": "knight|left|mid",
                                    "delay_s": 1.0, "confidence": "high", "following_plays": []})
    r.update({"game_id": "g1", "split": "test"})
    (ds / "states.jsonl").write_text(json.dumps(r) + "\n")
    run = tmp_path / "runs" / "g1"
    run.mkdir(parents=True)
    (run / "vs_pro_meta.json").write_text(json.dumps({"own_deck": ["knight"], "start": 0.0, "end": 50.0}))
    (run / "pro_plays.jsonl").write_text("stale\n")
    (tmp_path / "runs" / "not_a_run").mkdir()
    assert E.refresh_pro_plays(tmp_path / "runs", ds) == 1
    assert [json.loads(x)["t"] for x in (run / "pro_plays.jsonl").read_text().splitlines()] == [7.0]


def test_threat_cards_parse():
    assert E.threat_cards({"threats": ["bomber(e) advancing left lane, tower in 7.8s", "goblins(e) x", "bomber(e) again"]}) == ["bomber", "goblins"]
    assert E.threat_cards({}) == []


def test_video_source_time_is_video_seconds(tmp_path):
    """The run's `t` is the dataset's `t`: frames are stamped idx / fps from the start of the mp4, `--start` seeks there."""
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    path = tmp_path / "v.mp4"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48))
    if not w.isOpened():
        pytest.skip("no mp4 writer")
    for i in range(40):
        w.write(np.full((48, 64, 3), i * 6, np.uint8))
    w.release()
    from cr_perception.sources import VideoFrameSource
    src = VideoFrameSource(path, start=1.0, end=2.5, stride=3)
    ts = [round(t, 2) for _f, t, _i in src.frames()]
    src.close()
    assert ts[0] == pytest.approx(1.0, abs=0.11) and ts[1] - ts[0] == pytest.approx(0.3, abs=0.01) and ts[-1] <= 2.5


def test_game_argv_is_the_live_configuration_over_the_video_window(tmp_path):
    game = {"game_id": "v-m0.0", "video_id": "v", "own_deck": list("abcdefgh"), "start": 12.5, "end": 100.0}
    argv = E.game_argv(game, tmp_path / "out", stride=3, calib="runs/poc/mirror_calib.json", heroes=["a"])
    from clashbot import live
    args = live.build_parser().parse_args(argv)
    assert (args.source, args.ranker, args.d1_backend, args.policy, args.record_shots) == ("video", "d1", "transformers", "fast", True)
    assert (args.d1_model, args.act_model, args.fps, args.detect_every) == ("db/models/d1-ft-joint-tac", "db/models/act-gbdt", 10.0, 3)
    assert (args.start, args.end, args.stride, args.hero_cards, args.live) == (12.5, 100.0, 3, "a", False)
    assert args.video == "data/videos/v/v.mp4" and args.deck == "a,b,c,d,e,f,g,h" and args.seconds == pytest.approx(97.5)
    assert not args.no_tactics and not args.no_kg2 and args.arbiter == "none" and args.planner == "rules"


def test_resolve_calib_prefers_the_videos_own(tmp_path):
    own = tmp_path / "runs" / "videos" / "vid" / "match_0"
    own.mkdir(parents=True)
    (own / "calib.json").write_text("{}")
    assert E.resolve_calib("vid", "auto", tmp_path) == "runs/videos/vid/match_0/calib.json"
    assert E.resolve_calib("other", "auto", tmp_path) == E.MIRROR_CALIB
    assert E.resolve_calib("vid", "runs/poc/mirror_calib.json", tmp_path) == "runs/poc/mirror_calib.json"


def test_deck_variants_from_states_and_the_deck_page(tmp_path):
    deck = ["archers", "royal-hogs", "valkyrie", "zap", "a", "b", "c", "d"]
    cards = {"archers": {"has_evolution": True}, "royal-hogs": {"has_evolution": True}, "valkyrie": {"has_hero_variant": True, "has_evolution": True},
             "zap": {}, "a": {}, "b": {}, "c": {}, "d": {}}
    (tmp_path / "k.md").write_text('---\nevolutions: archers, royal-hogs, zap, valkyrie\nheroes: "none"\n---\nbody\n')
    game = {"own_deck": deck, "own_deck_key": "k"}
    rows = [{"state": {"own": {"hero_cards": ["valkyrie"]}}}, {"state": {"own": {"hero_cards": []}}}]
    evo, hero = E.deck_variants(game, rows, tmp_path, cards)
    assert hero == ["valkyrie"] and evo == ["archers", "royal-hogs"]          # zap has no evolution; a hero is never also an evo
    assert E.deck_variants({"own_deck": deck, "own_deck_key": None}, [], tmp_path, cards) == ([], [])


def test_select_games_skips_incomplete_decks_and_round_robins_over_videos():
    def g(gid, vid, start, deck=8, status="valid", result="win"):
        return {"game_id": gid, "video_id": vid, "own_deck": list("abcdefgh")[:deck], "deck_status": status, "result": result, "start": start,
                "end": start + 100, "split": "test"}
    games = [g("a0", "A", 0), g("a1", "A", 200), g("b0", "B", 0), g("b1", "B", 200), g("c0", "C", 0, deck=7, status="incomplete"),
             g("a2", "A", 400, result="unknown")]
    sel, skipped = E.select_games(games)
    assert [x["game_id"] for x in sel] == ["a0", "b0", "a1", "b1", "a2"]      # the k-th game of each video first, unknown results last
    assert [s["game_id"] for s in skipped] == ["c0"] and "8-card" in skipped[0]["skip_reason"]
    sel, skipped = E.select_games(games, n=2)
    assert [x["game_id"] for x in sel] == ["a0", "b0"] and {s["game_id"] for s in skipped} == {"c0", "a1", "b1", "a2"}
    assert [x["game_id"] for x in E.select_games(games, only=["b1"])[0]] == ["b1"]


# ------------------------------------------------------------------------------------------------ resumability and the heavy lock
def toy_games():
    return [{"game_id": gid, "split": "test", "start": 0.0, "end": 60.0} for gid in ("A", "B")]


def fake_runner(out, rcs, log):
    def run(cmd, cwd=None):
        log.append(cmd)
        gid = cmd[cmd.index("--run-game") + 1]
        rc = rcs[gid].pop(0)
        if rc == 0:
            (out / gid).mkdir(exist_ok=True)
            (out / gid / "summary.json").write_text("{}")
        return SimpleNamespace(returncode=rc)
    return run


def test_orchestrate_skips_finished_games_and_wraps_each_run_in_lockf(tmp_path):
    out = tmp_path / "out"
    (out / "A").mkdir(parents=True)
    (out / "A" / "summary.json").write_text("{}")                    # finished: skipped
    log = []
    res = E.orchestrate(toy_games(), out, stride=3, calib="auto", lock=tmp_path / "lock", hold_file=tmp_path / "HOLD", poll_s=0,
                        runner=fake_runner(out, {"B": [0]}, log), sleep=lambda s: None)
    assert res["skipped"] == ["A"] and res["done"] == ["B"] and res["failed"] == []
    assert len(log) == 1 and log[0][:2] == [E.LOCKF, str(tmp_path / "lock")] and log[0][log[0].index("--run-game") + 1] == "B"
    log.clear()                                                         # a second pass has nothing left to run
    res = E.orchestrate(toy_games(), out, stride=3, calib="auto", lock=tmp_path / "lock", hold_file=tmp_path / "HOLD", poll_s=0,
                        runner=fake_runner(out, {}, log), sleep=lambda s: None)
    assert res["skipped"] == ["A", "B"] and log == []


def test_orchestrate_waits_for_hold_heavy_and_retries_when_it_appears_after_the_lock(tmp_path):
    out = tmp_path / "out"
    hold = tmp_path / "HOLD"
    hold.write_text("")
    sleeps = []

    def sleep(s):
        sleeps.append(s)
        if len(sleeps) == 2:
            hold.unlink()                                               # the user stopped playing
    log = []
    games = toy_games()[:1]
    res = E.orchestrate(games, out, stride=3, calib="auto", lock=tmp_path / "lock", hold_file=hold, poll_s=60,
                        runner=fake_runner(out, {"A": [E.EX_HELD, 0]}, log), sleep=sleep)
    assert sleeps == [60, 60] and res["done"] == ["A"] and len(log) == 2            # waited, ran, got EX_HELD, ran again


def test_orchestrate_retries_a_failed_game_once_then_gives_up(tmp_path):
    out = tmp_path / "out"
    log = []
    res = E.orchestrate(toy_games()[:1], out, stride=3, calib="auto", lock=tmp_path / "lock", hold_file=tmp_path / "HOLD", poll_s=0,
                        retries=1, runner=fake_runner(out, {"A": [1, 1]}, log), sleep=lambda s: None)
    assert res["failed"] == ["A"] and len(log) == 2
    log.clear()
    res = E.orchestrate(toy_games()[:1], out, stride=3, calib="auto", lock=tmp_path / "lock", hold_file=tmp_path / "HOLD", poll_s=0,
                        retries=1, runner=fake_runner(out, {"A": [1, 0]}, log), sleep=lambda s: None)
    assert res["done"] == ["A"]


def test_run_game_stands_down_when_hold_heavy_exists(tmp_path):
    hold = tmp_path / "HOLD"
    hold.write_text("")
    assert E.run_game({"game_id": "x"}, tmp_path, stride=3, calib="auto", hold_file=hold) == E.EX_HELD
    assert not (tmp_path / "x").exists()                                # nothing was started


def test_run_finished_needs_summary_json(tmp_path):
    assert not E.run_finished(tmp_path)
    (tmp_path / "decisions.jsonl").write_text("")
    assert not E.run_finished(tmp_path)
    (tmp_path / "summary.json").write_text("{}")
    assert E.run_finished(tmp_path)


def test_score_runs_reads_finished_run_dirs_only(tmp_path):
    for gid, finished in (("g1", True), ("g2", False)):
        d = tmp_path / gid
        d.mkdir()
        meta = {**toy_meta(gid), "head": "abc", "decision_code_changed": [], "wall_s": 1.0, "frames": 1, "decisions": 7}
        (d / "vs_pro_meta.json").write_text(json.dumps(meta))
        (d / "pro_plays.jsonl").write_text("".join(json.dumps(p) + "\n" for p in toy_pro()))
        (d / "decisions.jsonl").write_text("".join(json.dumps(x) + "\n" for x in toy_decisions()))
        holds, frames = toy_holds_frames()
        (d / "holds.jsonl").write_text("".join(json.dumps(x) + "\n" for x in holds))
        (d / "frames.jsonl").write_text("".join(json.dumps(x) + "\n" for x in frames))
        if finished:
            (d / "summary.json").write_text("{}")
    report = E.score_runs(tmp_path, baselines(), n_boot=20)
    assert report["aggregate"]["n_games"] == 1 and report["meta"]["head"] == "abc"
    assert report["aggregate"]["metrics"]["recall_2s"]["value"] == pytest.approx(0.8)


# ------------------------------------------------------------------------------------------------ the real dataset
@needs_dataset
def test_test_split_selection_and_pro_plays_on_the_real_dataset():
    games = E.load_games("test")
    assert len(games) == 28
    sel, skipped = E.select_games(games)
    assert len(sel) + len(skipped) == 28 and len(sel) >= 8
    assert all(len(g["own_deck"]) == 8 and g["end"] > g["start"] for g in sel)
    g = sel[0]
    rows = E.read_states({g["game_id"]})[g["game_id"]]
    plays = E.pro_plays(rows, g["own_deck"], g["start"], g["end"])
    ts = [p["t"] for p in plays]
    assert len(plays) >= 20 and ts == sorted(ts) and all(g["start"] <= t <= g["end"] for t in ts)
    assert sum(p["card_ok"] for p in plays) > 0.6 * len(plays)
    assert any(p["hand"] for p in plays)

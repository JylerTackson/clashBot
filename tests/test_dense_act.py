"""MAC-V: dense wait/act dataset builder (D-040).

Synthetic dataset (games/provenance/states) and a fake runs/ tree of perception rows; no model, no real data.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clashbot import d1_format  # noqa: E402
from clashbot.game_rules import load_cards  # noqa: E402
from cr_perception.state import GameState, PlayEvent, UnitObs  # noqa: E402
from tools.data import build_dense_act as bda  # noqa: E402

CARDS = load_cards()
DECK = ["archers", "berserker", "bomb-tower", "goblinstein", "magic-archer", "royal-delivery",
        "tornado", "wall-breakers"]
HAND = ["tornado", "berserker", "archers", "bomb-tower"]

TRAIN_WINDOW = (10.0, 20.0)         # g_train, video "vid_a" (video_time given as strings)
VAL_WINDOW = (30.0, 40.0)           # g_val, video "vid_b" (video_time given as numbers)
TRAIN_DEPLOYS = [(12.5, "tornado"), (16.5, "archers")]
VAL_DEPLOYS = [(33.5, "archers")]
OPP_PLAY_T = 14.5                   # one opponent miner in vid_a
THREAT_FROM, THREAT_TO = 15.5, 17.5  # an enemy knight 3 s from the tower on rows in [from, to)


# --------------------------------------------------------------------------- fixtures

def play(t, card, player="own"):
    return {"t": t, "player": player, "card": card, "tile": [8, 11], "elixir_cost": 3.0,
            "detect_source": "arena", "confidence": "high"}


def state_row(t: float, units=(), match_id=1, opp_deck=(), opp_elixir=5.5) -> dict:
    gs = GameState(t=t, frame_index=int(t * 10), readiness="match", match_id=match_id, match_clock="1:20",
                   match_seconds=100.0, phase="double_elixir",
                   own={"elixir": 5.0, "hand": list(HAND), "hand_conf": [0.9] * 4, "next_card": "wall-breakers",
                        "towers": {"king": 4000, "left": 3000, "right": 2500}},
                   opponent={"elixir_est": opp_elixir, "towers": {"king": 4800, "left": 4000, "right": None},
                             "deck_known": list(opp_deck), "deck_complete": False},
                   units=list(units))
    return gs.to_json()


def knight() -> UnitObs:
    motion = {"heading": "advancing left lane", "speed": 1.2, "pred_2s": [3, 10],
              "eta_tower": {"tower": "own_left", "s": 3.0}, "tile": [3, 12]}
    return UnitObs(cls="knight", side="enemy", tile=(3, 12), conf=0.9, bbox=(0, 0, 10, 10), track_id=5,
                   category="unit", motion=motion)


def opp_event(ts: float, card: str = "miner") -> dict:
    return PlayEvent(timestamp=ts, match_clock="1:20", player="opponent", card=card, tile=(3, 20),
                     elixir_before=None, elixir_after=None, detect_source="arena", confidence="high").to_json()


def perception_rows(window, *, threat=None, events=(), gap=None) -> list[dict]:
    """A session_start row, state rows every 0.2 s over [start - 1, end + 1] and the given play events."""
    rows = [{"type": "session_start", "wall_time": 0.0}]
    lo, hi = window[0] - 1.0, window[1] + 1.0
    for k in range(int(round((hi - lo) / 0.2)) + 1):
        t = round(lo + 0.2 * k, 3)
        if gap and gap[0] < t < gap[1]:
            continue
        units = [knight()] if threat and threat[0] <= t < threat[1] else []
        rows.append(state_row(t, units))
    rows.extend(events)
    return rows


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def sparse_records(game_id: str, split: str, deploys: list[tuple[float, str]]) -> list[dict]:
    """Records shaped like dataset/v1/states.jsonl, carrying the deploys three different ways."""
    first_t, first_card = deploys[0]
    recs = [{"id": f"{game_id}#a", "game_id": game_id, "split": split, "t": first_t - 0.5,
             "state": {"own": {"recent_plays": []}},
             "action": {"type": "play", "card": first_card, "delay_s": 0.5,
                        "following_plays": [play(t, c) for t, c in deploys[1:]]}}]
    recs.append({"id": f"{game_id}#b", "game_id": game_id, "split": split, "t": deploys[-1][0] + 2.0,
                 "state": {"own": {"recent_plays": [play(t, c) for t, c in deploys]}},
                 "action": {"type": "hold", "card": None, "delay_s": None, "following_plays": []}})
    return recs


def neighbour_rows() -> list[dict]:
    """The tail of the previous match (same match_id value, other file) that spills into g_train's window."""
    rows = [{"type": "session_start", "wall_time": 0.0}]
    rows += [state_row(round(8.0 + 0.2 * k, 3), opp_deck=["hog-rider"], opp_elixir=9.0) for k in range(21)]
    rows.append(opp_event(11.0, "hog-rider"))
    return rows


def make_world(tmp_path: Path, *, gap=None, extra_games: bool = False, neighbour: bool = False) -> tuple[Path, Path]:
    ds, runs = tmp_path / "ds", tmp_path / "runs" / "videos"
    ds.mkdir()
    games = [{"game_id": "g_train", "split": "train", "own_deck": DECK},
             {"game_id": "g_val", "split": "val", "own_deck": DECK}]
    prov = [{"game_id": "g_train", "video_id": "vid_a", "video_time": [str(TRAIN_WINDOW[0]), str(TRAIN_WINDOW[1])]},
            {"game_id": "g_val", "video_id": "vid_b", "video_time": list(VAL_WINDOW)}]
    if extra_games:
        games += [{"game_id": "g_notime", "split": "test", "own_deck": DECK},
                  {"game_id": "g_badtime", "split": "test", "own_deck": DECK},
                  {"game_id": "g_noperc", "split": "test", "own_deck": DECK}]
        prov += [{"game_id": "g_badtime", "video_id": "vid_c", "video_time": ["n/a", "5.0"]},
                 {"game_id": "g_noperc", "video_id": "vid_missing", "video_time": ["5.0", "15.0"]}]
    write_jsonl(ds / "games.jsonl", games)
    write_jsonl(ds / "provenance.jsonl", prov)
    write_jsonl(ds / "states.jsonl", sparse_records("g_train", "train", TRAIN_DEPLOYS)
                + sparse_records("g_val", "val", VAL_DEPLOYS))
    write_jsonl(runs / "vid_a" / "match_0" / "states.jsonl",
                perception_rows(TRAIN_WINDOW, threat=(THREAT_FROM, THREAT_TO), events=[opp_event(OPP_PLAY_T)],
                                gap=gap))
    write_jsonl(runs / "vid_b" / "match_0" / "game_1" / "states.jsonl", perception_rows(VAL_WINDOW))
    if neighbour:       # sorts before the dominant file, so a merge of the files would prefer its rows
        write_jsonl(runs / "vid_a" / "match_0" / "game_0" / "states.jsonl", neighbour_rows())
    return ds, runs


def run(tmp_path: Path, *extra: str, world: tuple[Path, Path] | None = None) -> tuple[Path, list[str]]:
    ds, runs = world or make_world(tmp_path)
    out = tmp_path / "out"
    rc = bda.main(["--dataset", str(ds), "--runs", str(runs), "--out", str(out), *extra])
    assert rc == 0
    return out, [str(ds), str(runs)]


def read_split(out: Path, split: str) -> list[dict]:
    return [json.loads(l) for l in (out / f"{split}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]


def by_t(lines: list[dict]) -> dict[float, dict]:
    return {float(l["id"].rsplit("#", 1)[1]): l for l in lines}


# --------------------------------------------------------------------------- 1: pro_deploy_times

def test_pro_deploy_times_dedupes_windows_and_collapses_rereads():
    records = [
        {"t": 12.0, "state": {"own": {"recent_plays": []}},
         "action": {"type": "play", "card": "tornado", "delay_s": 0.5,
                    "following_plays": [play(16.5, "archers")]}},
        # overlapping window repeats tornado@12.5 and archers@16.5; tornado is re-read 0.4 s later; unplayable /
        # unidentified entries are dropped; the 16.52 re-read of archers falls on the same (round(t, 1), card)
        {"t": 14.0, "state": {"own": {"recent_plays": [play(12.5, "tornado"), play(12.9, "tornado"),
                                                       play(13.0, None), play(13.5, "tower-princess"),
                                                       play(13.6, "not-a-card")]}},
         "action": {"type": "hold", "card": None, "delay_s": None,
                    "following_plays": [play(16.5, "archers"), play(16.52, "archers"), play(19.0, "tornado")]}},
        {"t": 18.0, "state": {"own": {"recent_plays": [play(16.5, "archers")]}},
         "action": {"type": "unknown", "card": None, "delay_s": 1.0, "following_plays": []}},
        {"t": 20.0, "state": {"own": {}}, "action": {"type": "play", "card": "berserker", "delay_s": None}},
    ]
    got = bda.pro_deploy_times(records, CARDS)
    assert got == [(12.5, "tornado"), (16.5, "archers"), (19.0, "tornado")]   # 19.0 is 6.5 s after: kept
    assert bda.pro_deploy_times([], CARDS) == []


def test_pro_deploy_times_action_time_is_record_t_plus_delay():
    rec = {"t": 30.0, "state": {"own": {"recent_plays": []}},
           "action": {"type": "play", "card": "archers", "delay_s": 1.01, "following_plays": []}}
    # the same event as read by a later record's recent_plays (timestamp 31.009): one deploy
    later = {"t": 35.0, "state": {"own": {"recent_plays": [play(31.009, "archers")]}},
             "action": {"type": "hold", "card": None, "delay_s": None, "following_plays": []}}
    assert bda.pro_deploy_times([rec, later], CARDS) == [(31.009, "archers")]


# --------------------------------------------------------------------------- 2: labels

def test_labels_follow_the_horizon(tmp_path):
    out, _ = run(tmp_path)
    lines = by_t(read_split(out, "train"))
    # deploys at 12.5 and 16.5
    assert lines[12.0]["answers"] == {"act": "true"}       # deploy at t + 0.5
    assert lines[11.0]["answers"] == {"act": "false"}      # deploy at t + 1.5 (horizon 1.0)
    assert lines[16.0]["answers"] == {"act": "true"}
    assert lines[15.0]["answers"] == {"act": "false"}
    assert lines[13.0]["answers"] == {"act": "false"}      # the deploy is behind us
    assert [t for t, l in lines.items() if l["answers"]["act"] == "true"] == [12.0, 16.0]
    assert lines[11.0]["meta"]["next_deploy_in"] == 1.5
    assert lines[12.0]["meta"]["next_deploy_in"] == 0.5
    assert lines[17.0]["meta"]["next_deploy_in"] is None


def test_horizon_option(tmp_path):
    out, _ = run(tmp_path, "--horizon", "2.0")
    lines = by_t(read_split(out, "train"))
    assert [t for t, l in lines.items() if l["answers"]["act"] == "true"] == [11.0, 12.0, 15.0, 16.0]


# --------------------------------------------------------------------------- 3: sampling

def test_step_one_over_a_ten_second_game(tmp_path):
    out, _ = run(tmp_path)
    train, val = read_split(out, "train"), read_split(out, "val")
    assert len(train) == 11 and len(val) == 11                     # t = start .. end inclusive
    assert list(by_t(train)) == [float(t) for t in range(10, 21)]
    assert train[0]["id"] == "g_train#10.0" and val[-1]["id"] == "g_val#40.0"
    stats = json.loads((out / "stats.json").read_text())
    assert stats["splits"]["train"]["skipped"]["no_state"] == 0


def test_step_option(tmp_path):
    out, _ = run(tmp_path, "--step", "2.5")
    assert list(by_t(read_split(out, "train"))) == [10.0, 12.5, 15.0, 17.5, 20.0]


def test_gap_longer_than_tol_counts_no_state(tmp_path):
    world = make_world(tmp_path, gap=(14.0, 17.0))                 # rows at 14.0 and 17.0, nothing between
    out, _ = run(tmp_path, world=world)
    lines = by_t(read_split(out, "train"))
    assert 15.0 not in lines and 16.0 not in lines                 # 1.0 s from the nearest row > tol 0.6
    assert 14.0 in lines and 17.0 in lines
    assert len(lines) == 9
    stats = json.loads((out / "stats.json").read_text())
    assert stats["splits"]["train"]["skipped"]["no_state"] == 2
    assert stats["splits"]["val"]["skipped"]["no_state"] == 0
    assert stats["total"]["skipped"]["no_state"] == 2


def test_tol_option_bridges_the_gap(tmp_path):
    world = make_world(tmp_path, gap=(14.0, 17.0))
    out, _ = run(tmp_path, "--tol", "1.0", world=world)
    assert len(read_split(out, "train")) == 11


# --------------------------------------------------------------------------- 4: incremental events

def test_opponent_play_visible_only_after_its_timestamp(tmp_path):
    out, _ = run(tmp_path)
    lines = by_t(read_split(out, "train"))
    for t in (10.0, 12.0, 14.0):                                   # before 14.5
        assert "opp miner" not in lines[t]["state"], t
        assert lines[t]["meta"]["since_opp_play"] is None
    for t in (15.0, 17.0, 20.0):
        assert "opp miner@[3,20]" in lines[t]["state"], t
        assert lines[t]["meta"]["since_opp_play"] == pytest.approx(t - OPP_PLAY_T)
    assert "opp miner@[3,20] 0.5s ago" in lines[15.0]["state"]    # the sample time, not the row time, is the clock
    assert "opp deck: unknown" in lines[14.0]["state"] and "opp deck: miner" in lines[15.0]["state"]
    assert all(l["meta"]["since_opp_play"] is None for l in read_split(out, "val"))


def test_each_game_gets_a_fresh_builder(tmp_path):
    out, _ = run(tmp_path)
    assert all("miner" not in l["state"] for l in read_split(out, "val"))


# --------------------------------------------------------------------------- 5: format and splits

def test_every_line_validates_and_splits_follow_games(tmp_path):
    out, _ = run(tmp_path)
    assert read_split(out, "test") == []
    for split, game in (("train", "g_train"), ("val", "g_val")):
        lines = read_split(out, split)
        assert lines and all(l["game_id"] == game and l["split"] == split for l in lines)
        ts = [float(l["id"].rsplit("#", 1)[1]) for l in lines]
        assert ts == sorted(ts)
        for l in lines:
            d1_format.validate(l["state"], l["questions"], l["answers"])
            s, q = d1_format.act_request(l["state"])
            assert l["questions"] == q and l["state"] == s
            assert l["kind"] == "dense" and l["variant"] == "act"
            assert list(l) == ["id", "game_id", "split", "kind", "variant", "state", "questions", "answers", "meta"]
            assert set(l["meta"]) == {"since_opp_play", "threat", "own_elixir", "opp_elixir", "next_deploy_in"}
            assert l["meta"]["own_elixir"] == 5.0 and l["meta"]["opp_elixir"] == 5.5
            assert "hand: tornado, berserker, archers, bomb-tower" in l["state"]


def test_threat_meta(tmp_path):
    out, _ = run(tmp_path)
    lines = by_t(read_split(out, "train"))
    assert [t for t, l in lines.items() if l["meta"]["threat"]] == [16.0, 17.0]
    assert "threats: knight(e) advancing left lane, tower in 3.0s" in lines[16.0]["state"]
    assert "threats: none" in lines[15.0]["state"]


# --------------------------------------------------------------------------- 6: main outputs

def test_main_writes_files_stats_and_readme(tmp_path, capsys):
    out, _ = run(tmp_path)
    for name in ("train.jsonl", "val.jsonl", "test.jsonl", "stats.json", "README.md"):
        assert (out / name).exists(), name
    stats = json.loads((out / "stats.json").read_text())
    tr, va, te = (stats["splits"][s] for s in ("train", "val", "test"))
    assert tr["lines"] == 11 and tr["games"] == 1 and tr["act_rate"] == pytest.approx(2 / 11, abs=1e-4)
    assert va["lines"] == 11 and va["games"] == 1 and va["act_rate"] == pytest.approx(1 / 11, abs=1e-4)
    assert te["lines"] == 0 and te["games"] == 0 and te["act_rate"] is None
    assert tr["act_rate_when_threat"] == 0.5 and tr["n_threat"] == 2           # samples 16 (act) and 17
    assert va["act_rate_when_threat"] is None
    assert tr["n_within_3s_of_opp_play"] == 3                                   # samples 15, 16, 17
    assert tr["act_rate_within_3s_of_opp_play"] == pytest.approx(1 / 3, abs=1e-4)
    assert stats["total"]["lines"] == 22 and stats["total"]["games"] == 2
    assert stats["total"]["act_rate"] == pytest.approx(3 / 22, abs=1e-4)
    assert set(tr["skipped"]) == {"skipped_no_time", "skipped_no_perception", "no_state", "no_build"}
    assert tr["multi_match_groups"] == va["multi_match_groups"] == stats["total"]["multi_match_groups"] == 0
    shown = capsys.readouterr().out
    assert "train" in shown and "act_rate" in shown and "total" in shown and "multi_match_groups" in shown
    readme = (out / "README.md").read_text()
    assert "D-040" in readme and "LiveStateBuilder" in readme and "(t, t + 1" in readme


def test_states_sidecar_matches_the_act_files(tmp_path):
    from clashbot.state_text import render_context

    out, _ = run(tmp_path)
    for split in ("train", "val", "test"):
        side = read_split(out, f"states_{split}")
        acts = read_split(out, split)
        assert [r["id"] for r in side] == [l["id"] for l in acts]         # one row per act line, same order
    side, acts = read_split(out, "states_train"), read_split(out, "train")
    assert side and all(list(r) == ["id", "t", "state"] for r in side)
    for r, line in zip(side, acts):
        assert r["t"] == float(line["id"].rsplit("#", 1)[1])
        assert render_context(r["state"], r["t"]) == line["state"]        # the clean state the line was rendered from
    assert side[0]["state"]["own"]["hand"] == HAND and side[0]["state"]["opponent"]["recent_plays"] == []
    assert [r["state"]["threats"] != [] for r in side].count(True) == 2    # the knight at t = 16 and 17


def test_no_states_flag_skips_the_sidecars(tmp_path):
    out, _ = run(tmp_path, "--no-states")
    assert not list(out.glob("states_*.jsonl"))
    assert len(read_split(out, "train")) == 11 and (out / "stats.json").exists()
    other = tmp_path / "other"
    other.mkdir()
    with_states, _ = run(other)
    assert read_split(out, "train") == read_split(with_states, "train")   # the act files do not depend on the flag


def test_default_out_is_dataset_act_dense(tmp_path):
    ds, runs = make_world(tmp_path)
    assert bda.main(["--dataset", str(ds), "--runs", str(runs)]) == 0
    assert (ds / "act_dense" / "train.jsonl").exists() and (ds / "act_dense" / "stats.json").exists()


def test_skipped_games_are_counted(tmp_path):
    out, _ = run(tmp_path, world=make_world(tmp_path, extra_games=True))
    stats = json.loads((out / "stats.json").read_text())
    te = stats["splits"]["test"]
    assert te["games_listed"] == 3 and te["games"] == 0 and te["lines"] == 0
    assert te["skipped"]["skipped_no_time"] == 2                   # no provenance row, unparsable video_time
    assert te["skipped"]["skipped_no_perception"] == 1             # no runs/vid_missing
    assert stats["splits"]["train"]["lines"] == 11 and stats["total"]["skipped"]["skipped_no_time"] == 2


def test_games_option_selects_by_game_or_video_id(tmp_path):
    out, _ = run(tmp_path, "--games", "g_val")
    assert read_split(out, "train") == [] and len(read_split(out, "val")) == 11
    out2 = tmp_path / "out2"
    ds, runs = tmp_path / "ds", tmp_path / "runs" / "videos"
    assert bda.main(["--dataset", str(ds), "--runs", str(runs), "--out", str(out2), "--games", "vid_a"]) == 0
    assert len(read_split(out2, "train")) == 11 and read_split(out2, "val") == []


def test_step_below_minimum_is_rejected(tmp_path):
    ds, runs = make_world(tmp_path)
    with pytest.raises(SystemExit):
        bda.main(["--dataset", str(ds), "--runs", str(runs), "--out", str(tmp_path / "o"), "--step", "0.01"])


# --------------------------------------------------------------------------- 7: one perception match per game (FIX_LIST #24)

def test_neighbouring_match_rows_and_events_are_ignored(tmp_path):
    base_dir, multi_dir = tmp_path / "base", tmp_path / "multi"
    base_dir.mkdir()
    multi_dir.mkdir()
    base, _ = run(base_dir, world=make_world(base_dir))
    out, _ = run(multi_dir, world=make_world(multi_dir, neighbour=True))
    train = read_split(out, "train")
    assert len(train) == 11
    # identical to the run without the neighbouring match: its rows (same match_id value, other file) and its
    # opponent play never reach the builder, so the opponent deck of the other match never appears
    assert train == read_split(base, "train") and read_split(out, "val") == read_split(base, "val")
    assert all("hog-rider" not in l["state"] for l in train)
    assert all(l["meta"]["opp_elixir"] == 5.5 for l in train)
    assert "opp deck: unknown" in by_t(train)[10.0]["state"] and "opp miner@[3,20]" in by_t(train)[15.0]["state"]
    stats = json.loads((out / "stats.json").read_text())
    assert stats["splits"]["train"]["multi_match_groups"] == 1
    assert stats["splits"]["val"]["multi_match_groups"] == 0 and stats["splits"]["test"]["multi_match_groups"] == 0
    assert stats["total"]["multi_match_groups"] == 1
    assert json.loads((base / "stats.json").read_text())["total"]["multi_match_groups"] == 0
    assert "multi_match_groups" in bda.stats_table(stats)


def test_window_picks_the_group_with_most_rows_inside_the_game(tmp_path):
    runs = tmp_path / "runs"
    # same match_id value in both files; "z" has the most rows inside [10, 12] although "a" starts earlier
    write_jsonl(runs / "v" / "a" / "states.jsonl",
                [state_row(9.5, match_id=3, opp_elixir=1.0), state_row(10.5, match_id=3, opp_elixir=1.0),
                 opp_event(10.6, "hog-rider")])
    write_jsonl(runs / "v" / "z" / "states.jsonl",
                [state_row(t, match_id=3, opp_elixir=2.0) for t in (10.0, 10.5, 11.0, 11.5, 12.0)]
                + [opp_event(11.2, "miner")])
    states, events, n_groups = bda.Perception(runs).window("v", 10.0, 12.0)
    assert n_groups == 2
    assert [r["t"] for r in states] == [10.0, 10.5, 11.0, 11.5, 12.0]
    assert {r["opponent"]["elixir_est"] for r in states} == {2.0}
    assert [e["card"] for e in events] == ["miner"]


def test_window_tie_goes_to_the_group_starting_closest_to_start(tmp_path):
    runs = tmp_path / "runs"
    # two rows inside [10, 12] each; "a" first shows up 0.8 s before start, "z" right at start
    write_jsonl(runs / "v" / "a" / "states.jsonl",
                [state_row(t, match_id=1, opp_elixir=1.0) for t in (9.2, 10.2, 11.2)] + [opp_event(10.3, "hog-rider")])
    write_jsonl(runs / "v" / "z" / "states.jsonl",
                [state_row(t, match_id=1, opp_elixir=2.0) for t in (10.0, 11.0)] + [opp_event(10.1, "miner")])
    states, events, n_groups = bda.Perception(runs).window("v", 10.0, 12.0)
    assert n_groups == 2 and [r["t"] for r in states] == [10.0, 11.0]
    assert [e["card"] for e in events] == ["miner"]


def test_window_with_one_group_is_unchanged(tmp_path):
    runs = tmp_path / "runs"
    rows = [state_row(t) for t in (9.0, 10.0, 10.0, 11.0, 13.5)] + [opp_event(10.5), opp_event(15.0)]
    write_jsonl(runs / "v" / "states.jsonl", rows)
    states, events, n_groups = bda.Perception(runs).window("v", 10.0, 12.0)
    assert n_groups == 1 and [r["t"] for r in states] == [9.0, 10.0, 11.0]       # duplicate time dropped, 13.5 outside
    assert [e["timestamp"] for e in events] == [10.5]
    assert bda.Perception(runs).window("v", 100.0, 110.0) == ([], [], 0)

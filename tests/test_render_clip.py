"""MAC-AT: the annotated demo clip on hand-made toys (a calibration fixture, a toy run, a 2-second synthetic video); no model."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.eval import render_clip as R  # noqa: E402

# a rectangle on a 200x400 frame: the arena corners bl, br, tr, tl as content fractions (axis-aligned, so the homography is affine)
CALIB = {"content_rect_frac": [0.0, 0.0, 1.0, 1.0],
         "arena_corners_frac": [[0.1, 0.8], [0.9, 0.8], [0.9, 0.2], [0.1, 0.2]]}
FW, FH = 200, 400


def acts_and_pro():
    """3 bot taps vs 3 pro plays: a same-card pair, a different-card pair, an unanswered pro play (and a lone bot tap)."""
    acts = [{"t": 10.0, "card": "knight", "tile": [3.5, 10.0]}, {"t": 21.5, "card": "zap", "tile": [8.0, 12.0]},
            {"t": 90.0, "card": "giant", "tile": None}]
    pro = [{"t": 10.4, "card": "knight", "lane": "left", "zone": "mid", "tile": [3, 10]},
           {"t": 20.0, "card": "archers", "lane": "right", "zone": "back", "tile": [14, 2]},
           {"t": 50.0, "card": "giant", "lane": None, "zone": None, "tile": None}]
    return acts, pro


# ------------------------------------------------------------------------------------------------ tile -> pixel

def test_tile_mapping_round_trip():
    m = R.TileMapper(CALIB, FW, FH, view_h=800)
    assert (m.view_w, m.view_h, m.scale) == (400, 800, 2.0)
    for col, row in [(0, 0), (17, 31), (8, 15), (3.5, 10.0)]:
        x, y = m.to_view([col, row])
        # the tap target is the tile centre: (col + 0.5, row + 0.5) of 18 x 32 tiles between the corners
        assert abs(x - 2 * FW * (0.1 + 0.8 * (col + 0.5) / 18)) < 1e-6
        assert abs(y - 2 * FH * (0.8 - 0.6 * (row + 0.5) / 32)) < 1e-6
        # and back through the inverse homography (view pixel -> frame pixel -> tile)
        p = np.linalg.inv(m.ex.H) @ np.array([x / m.scale + m.x0, y / m.scale + m.y0, 1.0])
        assert abs(p[0] / p[2] - (col + 0.5)) < 1e-6 and abs(p[1] / p[2] - (row + 0.5)) < 1e-6


def test_tile_mapping_follows_the_content_rect():
    calib = {**CALIB, "content_rect_frac": [0.0, 0.25, 1.0, 0.5], "arena_corners_frac": [[0.1, 0.9], [0.9, 0.9], [0.9, 0.1], [0.1, 0.1]]}
    m = R.TileMapper(calib, FW, FH, view_h=500)
    assert (m.y0, m.h, m.view_w) == (100, 200, 500)          # 200 x 200 crop scaled to 500 x 500
    x, y = m.to_view([0, 31])
    assert 0 <= x <= m.view_w and 0 <= y <= 500
    assert m.view(np.zeros((FH, FW, 3), np.uint8)).shape == (500, 500, 3)


# ------------------------------------------------------------------------------------------------ pairs and windows

def test_pair_events_badges():
    acts, pro = acts_and_pro()
    events = {e["t"]: e for e in R.pair_events(acts, pro)}
    same, diff, miss = events[10.4], events[20.0], events[50.0]
    assert same["same_card"] is True and abs(same["dt"] - (-0.4)) < 1e-9 and same["resolved"] == 10.4
    assert diff["same_card"] is False and diff["act"]["card"] == "zap" and diff["resolved"] == 21.5
    assert miss["act"] is None and miss["resolved"] == 50.0 + R.MATCH_S
    assert [R.strip_text(e)[0] for e in (same, diff, miss)] == ["same", "diff", "miss"]
    assert "pro played knight (left, mid) at t=10.4 s" == R.strip_text(same)[1]
    assert "dt = -0.4 s (bot earlier)" in R.strip_text(same)[2]
    assert R.strip_text(miss)[1] == "pro played giant at t=50.0 s"       # no lane / zone read


def toy_events(starts_with_pairs):
    """Matched pairs (all same card) beginning at the given seconds."""
    acts = [{"t": s + 0.1, "card": "knight"} for s in starts_with_pairs]
    pro = [{"t": float(s), "card": "knight"} for s in starts_with_pairs]
    return R.pair_events(acts, pro)


def test_pick_window_most_matched_then_example():
    ev = toy_events([10, 12, 14, 60, 62, 64, 66])                 # 3 pairs near 10-14, 4 near 60-66
    s = R.pick_window(ev, 0, 100, 20)
    assert 46 <= s <= 60 and R.window_score(ev, s, s + 20, [])[0] == 4
    ev = toy_events([10, 12, 14, 60, 62, 64])                     # a tie of 3: the window around the example moment wins
    assert R.pick_window(ev, 0, 100, 20, examples=[62.0]) >= 44
    assert R.pick_window(ev, 0, 100, 20, examples=[12.0]) <= 12
    assert R.pick_window(ev, 0, 100, 20) <= 10                    # no example: the earliest
    assert R.pick_window(ev, 0, 10, 90) == 0                      # a run shorter than the clip: the whole run


def test_parse_examples():
    text = ("**1. AGREE** game g-m0.3, video t = 1251.2 s (bot tap at 1251.4 s, difference +0.1 s)\n- pro: x\n"
            "**4. DISAGREE** game h-m0.4, video t = 738.6 s (bot tap at 737.1 s, difference -1.5 s)")
    assert R.parse_examples(text) == [{"kind": "agree", "game": "g-m0.3", "t": 1251.2}, {"kind": "disagree", "game": "h-m0.4", "t": 738.6}]


# ------------------------------------------------------------------------------------------------ the panel text

DECISION = {"t": 12.0, "idx": 5, "key": "knight|left|mid", "tile": [3.5, 10.0], "mode": "defend",
            "reason": "defend monk left: knight (high)",
            "candidates": [{"key": "zap|left|mid", "sources": {"knn": 0.4, "kg": 0.0}, "note": "pros played this"},
                           {"key": "knight|left|mid", "sources": {"tac": 0.6, "knn": 0.9}, "note": "tactics: tank in front"},
                           {"key": "archers|left|back", "sources": {"knn": 0.1}, "note": ""},
                           {"key": "giant|left|back", "sources": {}, "note": ""},
                           {"key": "bats|left|back", "sources": {"knn": 0.05}, "note": ""}],
            "probs": [0.30, 0.50, 0.12, 0.06, 0.02],
            "tactics_lines": ["tactics: best knight @[3.5,10] tank", "tactics: second line"], "kg_lines": ["KG: knight answers monk"],
            "retrieval": [{"action_key": f"card{i}|left|mid", "score": 0.02 - i * 0.001, "verdict": "positive"} for i in range(5)],
            "context_text": "[double_elixir|2:42] elixir 8"}


def test_decision_panel_from_a_fixture_row():
    p = R.decision_panel(DECISION)
    assert p["title"] == "knight / left / mid" and p["tile"] == "tile (3.5, 10)" and p["reason"].startswith("defend monk left")
    assert [c["key"] for c in p["cands"]] == ["knight|left|mid", "zap|left|mid", "archers|left|back", "giant|left|back"]   # by d1 probability, 4
    assert [c["chosen"] for c in p["cands"]] == [True, False, False, False]
    assert p["cands"][0]["chips"] == ["tac", "knn"] and p["cands"][1]["chips"] == ["knn"] and p["cands"][3]["chips"] == []   # kg 0.0: no chip
    assert p["note"] == "tactics: tank in front" and p["tactic"] == "tactics: best knight @[3.5,10] tank" and p["kg"] == "KG: knight answers monk"
    assert [r["play"] for r in p["retrieved"]] == ["card0|left|mid", "card1|left|mid", "card2|left|mid"]
    # a chosen candidate that is not among the 4 most probable is still shown
    low = {**DECISION, "key": "bats|left|back"}
    assert "bats|left|back" in [c["key"] for c in R.decision_panel(low)["cands"]]
    held = R.decision_panel({"key": "hold", "reason": "wait", "candidates": [], "probs": []})
    assert held["title"] == "HOLD" and held["held"] and held["tile"] == "no tile" and held["cands"] == []


def toy_run(tmp_path, decisions=(DECISION,), holds=(), pro=()):
    acts, _ = R.bot_acts(list(decisions))
    return R.Run(tmp_path, {"game_id": "toy-m0.0", "result": "win", "own_deck": ["knight", "zap"]}, {"video": "x.mp4", "calib": "c.json", "start": 0.0},
                 list(decisions), list(holds), list(pro), acts, R.pair_events(acts, list(pro)), R.clock_anchors(list(decisions)))


def test_panel_model_mode_holds_clock_and_strip(tmp_path):
    pro = [{"t": 12.2, "card": "knight", "lane": "left", "zone": "mid", "tile": [3, 10]}]
    holds = [{"t": 14.0, "mode": "save", "act": False, "reason": "tower handles bats right"}]
    run = toy_run(tmp_path, holds=holds, pro=pro)
    m = R.panel_model(run, 12.5)
    assert m["mode"] == "defend" and m["decision"]["title"] == "knight / left / mid" and m["hold_line"] is None
    assert m["clock"] == "2:41" and m["strip"][0] == "same"       # 2:42 at t=12.0, counted down; resolved when both happened (12.2)
    m = R.panel_model(run, 14.5)
    assert m["mode"] == "hold" and m["hold_line"] == "save: tower handles bats right" and m["strip"][0] == "same"
    m = R.panel_model(run, 30.0)                                  # rows older than STATE_AGE_S describe nothing; the strip expired
    assert m["mode"] is None and m["strip"] is None and m["hold_line"] is None
    assert R.panel_model(run, 5.0)["decision"] is None            # before the first decision
    assert R.clock_at([(10.0, 5, "overtime")], 13.0) == "OT 0:02" and R.clock_at([], 1.0) is None


def test_draw_panel_runs_on_extreme_rows(tmp_path):
    run = toy_run(tmp_path, pro=[{"t": 12.2, "card": "knight", "lane": None, "zone": None, "tile": None}])
    long_row = {**DECISION, "reason": "x " * 300, "tactics_lines": ["t " * 300], "kg_lines": ["k " * 300]}
    for r in (run, toy_run(tmp_path, decisions=(long_row,))):
        img = R.draw_panel(R.panel_model(r, 12.5), 1400)
        assert img.shape == (1080, 1400, 3) and img.any()
    assert R.draw_panel(R.panel_model(run, 1.0), 1400).any()      # nothing yet: still draws


# ------------------------------------------------------------------------------------------------ a 2-second render

def test_render_two_seconds_of_a_synthetic_video(tmp_path):
    fps = 10.0
    vw = cv2.VideoWriter(str(tmp_path / "src.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (FW, FH))
    for i in range(40):                                           # 4 s, the colour moves with the frame
        vw.write(np.full((FH, FW, 3), (i * 6, 80, 200 - i * 4), np.uint8))
    vw.release()
    (tmp_path / "calib.json").write_text(json.dumps(CALIB))
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "config.json").write_text(json.dumps({"video": str(tmp_path / "src.mp4"), "calib": str(tmp_path / "calib.json"), "start": 0.0}))
    (run_dir / "vs_pro_meta.json").write_text(json.dumps({"game_id": "toy-m0.0", "result": "win", "own_deck": ["knight", "zap"]}))
    decisions = [{**DECISION, "t": 1.2}, {**DECISION, "t": 2.0, "key": "zap|left|mid", "context_text": "[double_elixir|2:40] x"}]
    holds = [{"t": 1.0, "mode": "save", "act": False, "reason": "tower handles bats right"}]
    pro = [{"t": 1.3, "card": "knight", "lane": "left", "zone": "mid", "tile": [3, 10]},
           {"t": 2.2, "card": "archers", "lane": "right", "zone": "back", "tile": [14, 2]}]
    for name, rows in (("decisions", decisions), ("holds", holds), ("pro_plays", pro)):
        (run_dir / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    run = R.load_run(run_dir)
    assert len(run.acts) == 2 and len(run.events) == 2

    out = tmp_path / "out" / "clip.mp4"
    info = R.render_clip(run, 1.0, 2.0, out, title_s=0.5, outro_s=0.5, stills=tmp_path / "stills")
    assert 19 <= info["clip_frames"] <= 21 and info["frames"] == info["clip_frames"] + 10
    v = R.verify(out)
    assert v["frames"] == info["frames"] and v["size"] == (1920, 1080) and abs(v["fps"] - fps) < 0.1
    assert info["codec"] in ("avc1", "mp4v") and info["pro_plays"] == 2 and info["matched"] == 2 and info["same_card"] == 1
    assert (tmp_path / "stills" / "clip_still_title.png").exists()

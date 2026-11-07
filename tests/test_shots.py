"""Decision screenshots (MAC-AF): ShotRecorder, the --record-shots wiring and tools/eval/review_shots.py - synthetic frames,
fake executor, no models."""
import json
import sys
import threading
import time
import types
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clashbot.candidates import Candidate  # noqa: E402
from clashbot.executor import ScreenMap, card_slot_points  # noqa: E402
from clashbot.game_rules import load_cards  # noqa: E402
from clashbot.live import LiveLoop, build_parser  # noqa: E402
from clashbot.policy import ActDecision  # noqa: E402
from clashbot.recorder_shots import TAP_STATUSES, ShotRecorder, annotate_decision, readings  # noqa: E402
from tools.eval.review_shots import classify, landing_verdict, main as review_main, read_index, review  # noqa: E402
from tools.eval.review_shots import is_tap, tap_cost  # noqa: E402

HAND = ["zap", "archers", "knight", "giant"]
PRIOR = {"by_key": {"giant|left|back": {"median_tile": [3.0, 2.0]}}, "by_lane_zone": {}}
W, H = 400, 800                                    # frame pixels; the window is 200 x 400 points at (100, 50): 2x
SMAP = ScreenMap(rect=[100.0, 50.0, 200.0, 400.0], content_rect_frac=[0.0, 0.0, 1.0, 1.0],
                 arena_corners_frac=[[0.02, 0.75], [0.98, 0.75], [0.97, 0.11], [0.03, 0.11]],
                 hand_rois=[[0.05 + 0.24 * i, 0.85, 0.2, 0.1] for i in range(4)])
FIELD_PT = [200.0, 250.0]                          # screen points -> image pixels (200, 400)


class Source:
    def __init__(self, n=40, fps=10.0):
        self.n, self.fps = n, fps

    def frames(self):
        for i in range(self.n):
            yield np.full((H, W, 3), 100, np.uint8), round(i / self.fps, 3), i

    def close(self):
        pass


class Perception:
    """Raw readings: elixir 8.0 until the tap, then `after` (3.2 = the tap landed, 8.0 = it did not)."""

    def __init__(self, after=3.2, tap_t=0.3):
        self.after, self.tap_t = after, tap_t

    def process(self, frame, t, idx):
        elixir = 8.0 if t < self.tap_t + 0.3 else self.after
        hand = list(HAND)
        if t >= self.tap_t + 0.3 and self.after < 8.0:
            hand[3] = "musketeer"                  # the tapped card left its slot
        return types.SimpleNamespace(readiness="match", own={"elixir": elixir, "hand": hand})

    def drain_events(self):
        return []


class Builder:
    """The tracked state: elixir 9.9 (differs from the raw reading) and an always-full hand."""

    def observe_events(self, events):
        pass

    def build(self, gs):
        return {"phase": "single_elixir", "match_seconds": 150,
                "own": {"elixir": 9.9, "hand": ["zap", "archers", "knight", "giant"], "hand_confidence": 0.9,
                        "deck": [], "towers": {}, "recent_plays": []},
                "opponent": {"elixir_estimate": 4.0, "towers": {}, "deck_known": [], "recent_plays": []},
                "units": [], "threats": []}, "CTX"


class Generator:
    def generate(self, state, t, ctx, n=4, **filters):
        return [Candidate("giant|left|back", 1.0, {"knn": 1.0})]


class Ranker:
    name = "none"


class OncePolicy:
    """Acts on the first frame at or after `at`, then holds."""

    def __init__(self, at=0.3):
        self.at, self.done = at, False

    def decide(self, state, t, context_text=None):
        go = (not self.done) and t >= self.at - 1e-9
        self.done = self.done or go
        return ActDecision(go, "leak", None, "test reason")

    def filter_candidates(self, candidates, decision, state):
        return candidates


class Executor:
    """MirroringExecutor-shaped: planned entries with screen-point clicks, `sent` as given."""

    def __init__(self, sent=True):
        self.screen_map = SMAP
        self.sent = sent
        self.calls = []

    def execute(self, key, tile, t, hand=None):
        self.calls.append(key)
        slot = list(hand).index(key.split("|")[0])
        return {"t": t, "key": key, "tile": tile, "status": "planned", "slot": slot,
                "clicks": [card_slot_points(SMAP)[slot], FIELD_PT], "sent": self.sent}


def make_loop(tmp_path, shots, after=3.2, sent=True, n=40, policy=None):
    return LiveLoop(Source(n), Perception(after), Builder(), Generator(), Ranker(), Executor(sent), load_cards(), PRIOR,
                    out_dir=tmp_path, min_gap_s=1.0, policy=policy or OncePolicy(), shots=shots)


def read_rows(shots_dir: Path) -> list[dict]:
    return [json.loads(line) for line in (shots_dir / "index.jsonl").read_text().splitlines()]


# 1 ---------------------------------------------------------------------------------------------------------------------
def test_recording_is_off_by_default(tmp_path):
    args = build_parser().parse_args(["--calib", "c.json", "--deck", "a"])
    assert args.record_shots is False and args.shots_max == 400
    args = build_parser().parse_args(["--calib", "c.json", "--deck", "a", "--record-shots", "--shots-max", "5"])
    assert args.record_shots is True and args.shots_max == 5
    summary = make_loop(tmp_path, None).run()
    assert summary["decisions"] == 1 and "shots" not in summary
    assert not (tmp_path / "shots").exists()
    assert "shots" not in json.loads((tmp_path / "decisions.jsonl").read_text())["timings_ms"]


# 2 ---------------------------------------------------------------------------------------------------------------------
def test_record_shots_writes_annotated_abc_and_the_index(tmp_path):
    shots = ShotRecorder(tmp_path / "shots")
    summary = make_loop(tmp_path, shots).run()
    sd = tmp_path / "shots"
    assert summary["decisions"] == 1
    assert summary["shots"]["decisions"] == 1 and summary["shots"]["taps"] == 1 and summary["shots"]["shots_dropped"] == 0
    assert sorted(p.name for p in sd.glob("*.jpg")) == ["d0003_t0.3_a.jpg", "d0003_t0.3_b.jpg", "d0003_t0.3_c.jpg"]

    ref = cv2.resize(np.full((H, W, 3), 100, np.uint8), None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    a, b, c = (cv2.imread(str(sd / f"d0003_t0.3_{k}.jpg")).astype(int) for k in "abc")
    assert a.shape == b.shape == c.shape == ref.shape == (400, 200, 3)         # 50 % scale
    assert np.abs(a - ref).max() > 100                                          # annotated: boxes, points, text
    assert np.abs(a[330:390, 100:200] - ref[330:390, 100:200]).sum() > 0        # the hand slot ROIs
    for img in (b, c):
        assert np.abs(img[:20] - ref[:20]).max() > 100                          # the small header
        assert np.abs(img[20:] - ref[20:]).max() <= 6                           # and nothing else

    row = read_rows(sd)[0]
    assert row["idx"] == 3 and row["t"] == 0.3 and row["key"] == "giant|left|back" and row["mode"] == "leak"
    assert row["reason"] == "test reason" and row["status"] == "planned" and row["sent"] is True and row["slot"] == 3
    assert row["tile"] == [3.0, 2.0]
    assert row["clicks_screen"] == [[274.0, 410.0], FIELD_PT]
    assert row["clicks_image"] == [[348.0, 720.0], [200.0, 400.0]]
    assert row["elixir_before"] == 8.0 and row["elixir_state_before"] == 9.9 and row["hand_conf"] == 0.9
    assert row["hand_before"] == HAND and row["hand_raw_before"] == HAND
    assert row["files"] == ["d0003_t0.3_a.jpg", "d0003_t0.3_b.jpg", "d0003_t0.3_c.jpg"]
    assert row["elixir_after_0.5"] == 3.2 and row["elixir_after_1.5"] == 3.2
    assert row["hand_after_1.5"] == HAND and row["hand_raw_after_1.5"][3] == "musketeer"
    # the recorder's time in the loop's thread is part of the decision's total
    timings = json.loads((tmp_path / "decisions.jsonl").read_text())["timings_ms"]
    assert timings["shots"] > 0 and timings["total"] >= timings["shots"]


def test_annotation_draws_the_slot_and_field_points_and_the_tapped_roi():
    img = np.full((400, 200, 3), 100, np.uint8)
    spec = {"header": "h", "hand": HAND, "rois": [[20 + 96 * i, 680, 100 + 96 * i, 760] for i in range(4)],
            "tap": {"slot": 3, "slot_pt": [348.0, 720.0], "field_pt": [200.0, 400.0], "tile": [3.0, 2.0]}}
    annotate_decision(img, spec, 0.5)
    assert tuple(img[360, 174]) == (255, 255, 0)                 # slot point: cyan
    assert tuple(img[200, 100]) == (255, 0, 255)                 # field point: magenta (the cross)
    assert tuple(img[360, 154]) == (255, 255, 0)                 # the tapped ROI's edge is cyan
    assert tuple(img[360, 10 + 48 * 0]) == (0, 220, 0)           # the other slots' green


def test_a_tap_that_did_not_land_keeps_its_elixir_and_the_card(tmp_path):
    shots = ShotRecorder(tmp_path / "shots")
    make_loop(tmp_path, shots, after=8.0).run()
    row = read_rows(tmp_path / "shots")[0]
    assert row["elixir_after_0.5"] == 8.0 and row["hand_raw_after_1.5"] == HAND


def test_a_hold_record_has_only_the_a_image(tmp_path):
    shots = ShotRecorder(tmp_path / "shots")
    loop = make_loop(tmp_path, shots)
    loop.generator = types.SimpleNamespace(generate=lambda *a, **k: [Candidate("fireball|left|mid", 1.0, {"knn": 1.0})])
    loop.run()
    row = read_rows(tmp_path / "shots")[0]
    assert row["key"] == "hold" and row["status"] == "hold_hand_unknown" and row["clicks_image"] == []
    assert row["files"] == ["d0003_t0.3_a.jpg", None, None]
    assert [p.name for p in (tmp_path / "shots").glob("*.jpg")] == ["d0003_t0.3_a.jpg"]


def test_a_dry_run_without_a_screen_map_plans_the_tap_points_from_the_calib(tmp_path):
    calib = {"content_rect_frac": [0.0, 0.0, 1.0, 1.0], "arena_corners_frac": SMAP.arena_corners_frac,
             "rois": {f"hand_{i}": SMAP.hand_rois[i] for i in range(4)}}
    shots = ShotRecorder(tmp_path / "shots", calib=calib)
    loop = make_loop(tmp_path, shots, sent=False)
    loop.executor = types.SimpleNamespace(execute=lambda key, tile, t, hand=None: {"t": t, "key": key, "tile": tile})
    loop.run()
    row = read_rows(tmp_path / "shots")[0]
    assert row["sent"] is False and row["status"] == "dry_run" and row["clicks_screen"] == []
    assert row["slot"] == 3 and row["clicks_image"][0] == [348.0, 720.0] and len(row["clicks_image"]) == 2
    assert row["files"][1] and row["files"][2]                                  # a dry-run tap is still followed


# 3 ---------------------------------------------------------------------------------------------------------------------
def test_a_full_queue_drops_images_without_blocking_the_loop(tmp_path):
    gate = threading.Event()
    shots = ShotRecorder(tmp_path / "shots", queue_size=2, writer=lambda job: gate.wait(10))
    loop = make_loop(tmp_path, shots, n=60, policy=types.SimpleNamespace(
        decide=lambda state, t, context_text=None: ActDecision(True, "leak", None, "r"),
        filter_candidates=lambda c, v, s: c))
    loop.min_gap_s = 0.0                                  # a tap on every frame: far more images than the queue holds
    worst = 0.0
    it = iter(loop.source.frames())
    for frame, t, idx in it:
        t0 = time.perf_counter()
        loop.step(frame, t, idx, 1.0)
        worst = max(worst, time.perf_counter() - t0)
    assert worst < 0.25                                   # no step waited for the blocked writer
    assert shots.shots_dropped > 50 and shots.summary()["shots_dropped"] == shots.shots_dropped
    gate.set()
    shots.close()
    rows = read_rows(tmp_path / "shots")
    assert len(rows) == 60                                # every decision keeps its index row
    assert sum(1 for r in rows if r["files"][0] is None) > 0 and any(r["files"][0] for r in rows)


def test_shots_max_stops_recording(tmp_path):
    shots = ShotRecorder(tmp_path / "shots", max_decisions=2)
    loop = make_loop(tmp_path, shots, n=60, policy=types.SimpleNamespace(
        decide=lambda state, t, context_text=None: ActDecision(True, "leak", None, "r"),
        filter_candidates=lambda c, v, s: c))
    summary = loop.run()
    assert summary["decisions"] > 2 and summary["shots"]["decisions"] == 2 and summary["shots"]["capped"] is True
    assert len(read_rows(tmp_path / "shots")) == 2


def test_pending_taps_are_written_when_the_run_ends(tmp_path):
    shots = ShotRecorder(tmp_path / "shots")
    make_loop(tmp_path, shots, n=8).run()                 # the run ends 0.4 s after the tap: no b / c frame
    row = read_rows(tmp_path / "shots")[0]
    assert row["files"] == ["d0003_t0.3_a.jpg", None, None] and row["elixir_after_0.5"] is None


def test_readings_prefer_the_raw_perception_state():
    gs = types.SimpleNamespace(own={"elixir": 4.04, "hand": ["zap", None, None, None]})
    state = {"own": {"elixir": 6.0, "hand": ["zap", "knight", "giant", "archers"], "hand_confidence": 0.5}}
    rd = readings(gs, state)
    assert rd["elixir"] == 4.04 and rd["elixir_state"] == 6.0 and rd["hand_raw"][1] is None and rd["hand"][1] == "knight"
    assert readings(None, state)["elixir"] == 6.0


# 4 ---------------------------------------------------------------------------------------------------------------------
def _row(idx, key, slot, before, a05, a15, raw15, sent=True, mode="leak", tile=(3.0, 2.0), status="planned"):
    return {"idx": idx, "t": idx / 10, "key": key, "mode": mode, "reason": "r", "status": status, "sent": sent,
            "slot": slot, "clicks_screen": [], "clicks_image": [], "tile": list(tile) if tile else None,
            "elixir_before": before, "elixir_state_before": before, "hand_before": HAND, "hand_raw_before": HAND,
            "hand_conf": 0.9, "files": [f"d{idx:04d}_a.jpg", f"d{idx:04d}_b.jpg", f"d{idx:04d}_c.jpg"],
            "elixir_after_0.5": a05, "elixir_after_1.5": a15, "hand_after_1.5": raw15, "hand_raw_after_0.5": None,
            "hand_raw_after_1.5": raw15}


def _make_run(tmp_path):
    shots = tmp_path / "run" / "shots"
    shots.mkdir(parents=True)
    rows = [
        _row(1, "giant|left|back", 3, 8.0, 4.0, 3.1, HAND),                                   # drop 4.9 >= 5 - 0.6: landed
        _row(2, "knight|left|mid", 2, 7.0, 7.3, 7.5, HAND, mode="defend", tile=(9, 20)),     # no drop, still there: failed
        _row(3, "zap|left|mid", 0, None, None, None, None),                                  # no readings: unknown
        _row(4, "archers|left|mid", 1, 5.0, 5.1, 4.8, ["zap", "musketeer", "knight", "giant"], tile=(9, 26)),  # left: landed
        _row(5, "hold", None, 5.0, None, None, None, status="hold_policy", sent=False, tile=None),
        _row(6, "knight|left|mid", 2, 7.0, 7.3, 7.5, HAND, sent=False, status="dry_run"),     # dry: judged only with the flag
        _row(7, "giant|left|back", 3, 8.0, 7.9, 7.9, HAND, mode="defend", tile=(3, 2)),      # failed
        _row(8, "fireball|left|mid", None, 5.0, None, None, None, sent=False, status="skipped_card_not_in_hand"),
    ]
    (shots / "index.jsonl").write_text("".join(json.dumps(r) + "\n" for r in reversed(rows)))   # unsorted on purpose
    for r in rows:
        for k, f in enumerate(r["files"]):
            if r["idx"] != 7 or k == 0:                    # decision 7 has no b / c image on disk
                cv2.imwrite(str(shots / f), np.full((120, 60, 3), 30 * (k + 1), np.uint8))
    return tmp_path / "run"


def test_landing_verdicts(tmp_path):
    cards = load_cards()
    run = _make_run(tmp_path)
    rows = classify(read_index(run), cards)
    verdict = {r["idx"]: r.get("verdict") for r in rows}
    assert verdict == {1: "landed", 2: "failed", 3: "unknown", 4: "landed", 5: None, 6: None, 7: "failed", 8: None}
    assert {r["idx"]: r["kind"] for r in rows} == {1: "tap", 2: "tap", 3: "tap", 4: "tap", 5: "hold", 6: "dry",
                                                   7: "tap", 8: "skipped"}
    dry = {r["idx"]: r.get("verdict") for r in classify(read_index(run), cards, True)}
    assert dry[6] == "failed"
    assert landing_verdict({"key": "zap|a|b", "slot": 0, "elixir_before": 5.0, "elixir_after_0.5": 5.0,
                            "elixir_after_1.5": None}, 2.0) == "unknown"        # the 1.5 s window is missing
    assert landing_verdict({"key": "zap|a|b", "slot": 0, "elixir_before": 5.0, "elixir_after_0.5": 3.0,
                            "elixir_after_1.5": 3.2}, 2.0) == "landed"          # the drop counts at either reading


def test_review_writes_sheets_and_review_md(tmp_path):
    run = _make_run(tmp_path)
    res = review(run, "all", include_dry=False, per_sheet=3, cell_w=60)
    out = run / "shots" / "sheets"
    assert [Path(p).name for p in res["sheets"]] == ["sheet_001.png", "sheet_002.png", "sheet_003.png"]   # 7 decisions
    sheet = cv2.imread(str(out / "sheet_001.png"))
    assert sheet.shape[1] == 3 * 60 + 4 * 4 and sheet.shape[0] == 3 * (40 + 120 + 4) + 4      # 3 rows of a / b / c
    text = (out / "review.md").read_text()
    assert "sent taps 5" in text and "holds 1" in text and "dry-run taps 1" in text
    assert "landed 2, failed 2, unknown 1" in text and "landing rate 50%" in text
    for needle in ("### Landing rate by card", "### Landing rate by slot", "### Landing rate by target row band",
                   "### Landing rate by mode", "still in its slot at +1.5 s"):
        assert needle in text
    assert "| giant | 2 | 1 | 1 | 0 | 50% |" in text                    # by card
    assert "| 3 | 2 | 1 | 1 | 0 | 50% |" in text                         # by slot
    assert "| defend | 2 | 0 | 2 | 0 | 0% |" in text                     # by mode
    assert "| rows 24-31 (enemy back) | 1 | 1 | 0 | 0 | 100% |" in text   # by target row band (archers at row 26)
    assert "- all: 3/4 (75%)" in text                                     # knight, giant, giant still there; archers left


def test_review_only_failed_and_dry(tmp_path):
    run = _make_run(tmp_path)
    res = review(run, "failed", include_dry=False, out_dir=tmp_path / "failed_sheets", cell_w=60)
    assert [Path(p).name for p in res["sheets"]] == ["sheet_001.png"]                 # 2 failed taps on one sheet
    assert cv2.imread(res["sheets"][0]).shape[0] == 2 * (40 + 120 + 4) + 4
    assert "landed 2, failed 2, unknown 1" in res["text"]                              # the tables still cover every tap
    res = review(run, "all", include_dry=True, out_dir=tmp_path / "dry_sheets")
    assert "landed 2, failed 3, unknown 1" in res["text"]                              # the dry-run tap is judged too


def test_review_cli_needs_an_index(tmp_path, capsys):
    assert review_main([str(tmp_path)]) == 2
    run = _make_run(tmp_path / "x")
    assert review_main([str(run), "--only", "failed"]) == 0
    assert (run / "shots" / "sheets" / "review.md").is_file()
    assert "Landing rate by card" in capsys.readouterr().out


# 5 (FIX #40: hero ability taps) ------------------------------------------------------------------------------------------
def _ability_decision(idx=7, t=1.0, sent=True):
    entry = {"t": t, "key": "ability:musketeer", "tile": None, "side": "right", "status": "planned_ability", "slot": None,
             "clicks": [[250.0, 420.0]], "sent": sent}
    return types.SimpleNamespace(idx=idx, t=t, key="ability:musketeer", tile=None, mode="defend", reason="turret",
                                 executor=entry)


def _ability_state(elixir):
    return {"own": {"elixir": elixir, "hand": HAND}}


def test_ability_taps_are_followed_like_card_taps(tmp_path):
    assert "planned_ability" in TAP_STATUSES and {"planned", "dry_run"} <= set(TAP_STATUSES)
    shots = ShotRecorder(tmp_path / "shots", writer=lambda job: True)
    frame = np.full((H, W, 3), 100, np.uint8)
    shots.on_decision(frame, _ability_decision(), _ability_state(6.0), executor=types.SimpleNamespace(screen_map=SMAP))
    assert shots.taps == 1                                           # a tap: it waits for its +0.5 s / +1.5 s frames
    shots.on_frame(frame, 1.2, 10, _ability_state(5.5))              # too early for the first frame
    shots.on_frame(frame, 1.6, 11, _ability_state(3.4))
    shots.on_frame(frame, 2.6, 12, _ability_state(3.9))
    shots.close()
    row = read_rows(tmp_path / "shots")[0]
    assert row["key"] == "ability:musketeer" and row["status"] == "planned_ability" and row["sent"] is True
    assert row["files"] == ["d0007_t1.0_a.jpg", "d0007_t1.0_b.jpg", "d0007_t1.0_c.jpg"]
    assert row["elixir_before"] == 6.0 and row["elixir_after_0.5"] == 3.4 and row["elixir_after_1.5"] == 3.9
    assert row["slot"] is None and len(row["clicks_image"]) == 1     # the button tap point


def test_an_ability_entry_without_a_planned_tap_is_not_followed(tmp_path):
    shots = ShotRecorder(tmp_path / "shots", writer=lambda job: True)
    dec = _ability_decision()
    dec.executor["status"] = "no_ability_tap"
    shots.on_decision(np.full((H, W, 3), 100, np.uint8), dec, _ability_state(6.0),
                      executor=types.SimpleNamespace(screen_map=SMAP))
    shots.close()
    assert shots.taps == 0 and read_rows(tmp_path / "shots")[0]["files"][1:] == [None, None]


def _ability_row(idx, before, a05, a15, sent=True, status="planned_ability"):
    return {"idx": idx, "t": idx / 10, "key": "ability:musketeer", "mode": "defend", "reason": "turret", "status": status,
            "sent": sent, "slot": None, "clicks_screen": [], "clicks_image": [], "tile": None, "elixir_before": before,
            "elixir_state_before": before, "hand_before": HAND, "hand_raw_before": HAND, "hand_conf": 0.9,
            "files": [None, None, None], "elixir_after_0.5": a05, "elixir_after_1.5": a15, "hand_after_1.5": HAND,
            "hand_raw_after_0.5": None, "hand_raw_after_1.5": HAND}


def _verdict(row):
    return classify([row], load_cards())[0]["verdict"]


def test_ability_taps_are_judged_by_an_elixir_drop_of_at_least_2_4():
    cards = load_cards()
    assert is_tap(_ability_row(1, 6.0, 3.0, 3.0)) and tap_cost(_ability_row(1, 6.0, 3.0, 3.0), cards) == 3.0
    assert _verdict(_ability_row(1, 6.0, 3.5, 3.6)) == "landed"      # fell by 2.5 >= 2.4
    assert _verdict(_ability_row(2, 6.0, 4.0, 3.5)) == "landed"      # the lowest of the two readings counts
    assert _verdict(_ability_row(3, 6.0, 4.0, 3.9)) == "failed"      # 2.1 only, with a reading at +1.5 s
    assert _verdict(_ability_row(4, 6.0, 5.8, 5.9)) == "failed"      # nothing spent
    assert _verdict(_ability_row(5, 6.0, None, None)) == "unknown"   # no readings at all
    assert _verdict(_ability_row(6, 6.0, 3.0, None)) == "landed"     # the +0.5 s reading alone shows the drop
    assert _verdict(_ability_row(7, None, 3.0, 3.0)) == "unknown"    # no elixir before
    # a planned_ability that was not sent is a dry-run tap: judged only with include_dry
    dry = classify([_ability_row(8, 6.0, 3.5, 3.5, sent=False)], cards)[0]
    assert dry["kind"] == "dry" and "verdict" not in dry
    assert classify([_ability_row(8, 6.0, 3.5, 3.5, sent=False)], cards, include_dry=True)[0]["verdict"] == "landed"
    # a card tap is still judged by its own cost (zap 2: fell by 2.0 >= 2 - 0.6)
    assert _verdict(_row(9, "zap|left|mid", 0, 5.0, 3.0, 3.0, HAND)) == "landed"


def test_review_covers_ability_taps_in_its_tables(tmp_path):
    run = tmp_path / "run"
    (run / "shots").mkdir(parents=True)
    rows = [_ability_row(1, 6.0, 3.5, 3.6), _ability_row(2, 6.0, 5.8, 5.9)]
    (run / "shots" / "index.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    res = review(run, "all")
    assert "sent taps 2" in res["text"] and "landed 1, failed 1, unknown 0" in res["text"]
    assert "| ability:musketeer | 2 | 1 | 1 | 0 | 50% |" in res["text"]

"""Live loop (MAC-G): rank_keys, LiveLoop.step/run, the CLI and the FastLoop edits - all with fakes, no models."""
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import clashbot.loop as loop_mod  # noqa: E402
from clashbot.actions import option_text, placement_tile  # noqa: E402
from clashbot.candidates import Candidate  # noqa: E402
from clashbot.config import HOLD_KEY  # noqa: E402
from clashbot.game_rules import load_cards  # noqa: E402
from clashbot.live import STAGES, TOS_NOTE, LiveLoop, build_parser, main, rank_keys  # noqa: E402
from clashbot.loop import FastLoop  # noqa: E402
from clashbot.ranker import RankResult, ScoreRanker  # noqa: E402

DECK = ["zap", "archers", "knight", "giant", "musketeer", "fireball", "valkyrie", "minions"]
PRIOR = {"by_key": {"giant|left|back": {"median_tile": [3.0, 2.0]}},
         "by_lane_zone": {"left|mid": {"median_tile": [4.0, 8.0]}}}
KEYS = ["zap|left|mid", "giant|left|back"]


def _state(elixir=5.0, hand=None):
    return {
        "phase": "double_elixir", "match_seconds": 150,
        "own": {"elixir": elixir, "hand": hand or ["zap", "archers", "knight", "giant"],
                "deck": ["zap", "archers", "knight", "giant"],
                "towers": {"king": None, "left": 4000, "right": 5000}, "recent_plays": []},
        "opponent": {"elixir_estimate": 4.0, "towers": {"king": None, "left": 3000, "right": 4858},
                     "deck_known": [], "recent_plays": []},
        "units": [], "threats": [],
    }


class FakeSource:
    def __init__(self, n=10):
        self.n = n
        self.last_grab_ms = 2.0
        self.closed = 0

    def frames(self):
        for i in range(self.n):
            yield f"frame{i}", round(i * 0.1, 3), i

    def stats(self):
        return {"frames": self.n}

    def close(self):
        self.closed += 1


class FakePerception:
    def process(self, frame, t, idx):
        return types.SimpleNamespace(readiness="match")

    def drain_events(self):
        return []

    def timing_report(self):
        return {}


class FakeBuilder:
    """build() returns None for the first `warmup` frames, then a fixed (state, ctx)."""

    def __init__(self, state=None, warmup=2):
        self.state = state or _state()
        self.warmup = warmup
        self.calls = 0
        self.observed = []

    def observe_events(self, events):
        self.observed.append(events)

    def build(self, gs):
        self.calls += 1
        return None if self.calls <= self.warmup else (self.state, "CTX")


class FakeGenerator:
    def __init__(self, keys=KEYS):
        self.keys = keys
        self.calls = []

    def generate(self, state, t, ctx, n=4, **filters):
        self.calls.append((t, n, filters))
        return [Candidate(k, 1.0 - 0.1 * i, {"knn": 1.0}) for i, k in enumerate(self.keys)]


class FakeKeyRanker:
    name = "fake-keys"

    def __init__(self):
        self.calls = []

    def choose_keys(self, context, keys, plan=None):
        self.calls.append((context, list(keys), plan))
        return RankResult(index=1, probs=[0.2, 0.8] + [0.0] * (len(keys) - 2), model="fake")


class FakeTextRanker:
    name = "fake-text"

    def __init__(self):
        self.calls = []

    def choose(self, context, options, plan=None):
        self.calls.append((context, list(options), plan))
        return RankResult(index=0, probs=[1.0] + [0.0] * (len(options) - 1), model="fake")


class FakeExecutor:
    def __init__(self):
        self.calls = []

    def execute(self, key, tile, t, hand=None):
        self.calls.append((key, tile, t, hand))
        return {"t": t, "key": key, "tile": tile}


class FakeClock:
    """Advances 1 ms per call."""

    def __init__(self):
        self.now = 0.0

    def __call__(self):
        self.now += 0.001
        return self.now


def _loop(tmp_path=None, source=None, builder=None, generator=None, ranker=None, executor=None, plan_text="PLAN"):
    return LiveLoop(source or FakeSource(), FakePerception(), builder or FakeBuilder(),
                    generator or FakeGenerator(), ranker or FakeKeyRanker(), executor or FakeExecutor(),
                    load_cards(), PRIOR, plan_text=plan_text, out_dir=tmp_path, clock=FakeClock())


# 1 -----------------------------------------------------------------------------------------------------------------
def test_rank_keys_single_key_skips_ranker():
    class Boom:
        def choose_keys(self, *a, **k):
            raise AssertionError("must not be called")

    assert rank_keys(Boom(), "ctx", ["zap|left|mid"], None) == (0, [1.0])


def test_rank_keys_uses_choose_keys_when_available():
    r = FakeKeyRanker()
    assert rank_keys(r, "ctx", KEYS, "plan") == (1, [0.2, 0.8])
    assert r.calls == [("ctx", KEYS, "plan")]


def test_rank_keys_text_ranker_gets_option_text():
    r = FakeTextRanker()
    assert rank_keys(r, "ctx", KEYS, None) == (0, [1.0, 0.0])
    assert r.calls == [("ctx", [option_text(k) for k in KEYS], None)]


def test_rank_keys_empty_raises():
    with pytest.raises(ValueError):
        rank_keys(FakeKeyRanker(), "ctx", [], None)


# 2 -----------------------------------------------------------------------------------------------------------------
def test_step_none_while_builder_returns_none():
    lp = _loop(builder=FakeBuilder(warmup=2))
    assert lp.step("f", 0.0, 0, 2.0) is None
    assert lp.step("f", 0.1, 1, 2.0) is None
    assert len(lp.frames) == 2 and not lp.decisions
    assert lp.frames[0]["readiness"] == "match" and lp.frames[0]["capture"] == 2.0
    assert set(lp.frames[0]) == {"t", "idx", "readiness", "capture", "perception", "live_state", "rect_changed"}
    assert lp.frames[0]["rect_changed"] is False


def test_step_none_when_should_decide_false():
    gen = FakeGenerator()
    lp = _loop(builder=FakeBuilder(_state(elixir=0.0), warmup=0), generator=gen)
    assert lp.step("f", 10.0, 0, 2.0) is None
    assert gen.calls == [] and lp.last_decision_t is None and len(lp.frames) == 1


# 3 -----------------------------------------------------------------------------------------------------------------
def test_step_decision_executes_with_hand_and_tile_and_timings():
    ex = FakeExecutor()
    rk = FakeKeyRanker()
    gen = FakeGenerator()
    lp = _loop(builder=FakeBuilder(warmup=0), generator=gen, ranker=rk, executor=ex)
    d = lp.step("f", 1.0, 7, 2.0)
    assert d is not None and d.key == "giant|left|back" and d.idx == 7 and d.t == 1.0
    state = lp.builder.state
    assert d.tile == placement_tile("giant|left|back", PRIOR) == [3.0, 2.0]
    assert ex.calls == [("giant|left|back", [3.0, 2.0], 1.0, state["own"]["hand"])]
    assert rk.calls == [("CTX", KEYS, "PLAN")]
    assert gen.calls == [(1.0, 4, {})]
    assert d.probs == [0.2, 0.8]
    assert [c["key"] for c in d.candidates] == KEYS and set(d.candidates[0]) == {"key", "score", "sources"}
    assert d.executor == {"t": 1.0, "key": "giant|left|back", "tile": [3.0, 2.0]}
    assert set(d.timings_ms) == set(STAGES)
    assert d.timings_ms["capture"] == 2.0
    parts = [d.timings_ms[s] for s in STAGES if s != "total"]
    assert d.timings_ms["total"] == pytest.approx(sum(parts))
    for s in ("perception", "live_state", "trigger", "candidates", "rank", "execute"):
        assert d.timings_ms[s] == pytest.approx(1.0)      # FakeClock: 1 ms per call
    assert lp.last_decision_t == 1.0 and lp.decisions == [d]


def test_step_passes_filters_to_generator():
    gen = FakeGenerator()
    lp = LiveLoop(FakeSource(), FakePerception(), FakeBuilder(warmup=0), gen, FakeKeyRanker(), FakeExecutor(),
                  load_cards(), PRIOR, filters={"exclude_game": "g1"}, clock=FakeClock())
    lp.step("f", 1.0, 0, 2.0)
    assert gen.calls == [(1.0, 4, {"exclude_game": "g1"})]


# 4 -----------------------------------------------------------------------------------------------------------------
def test_run_writes_outputs(tmp_path):
    src = FakeSource(10)
    lp = _loop(tmp_path, source=src, builder=FakeBuilder(_state(elixir=9.5), warmup=2))
    summary = lp.run(max_frames=10)
    for name in ("decisions.jsonl", "frames.jsonl", "summary.json"):
        assert (tmp_path / name).exists()
    dec_lines = [json.loads(x) for x in (tmp_path / "decisions.jsonl").read_text().splitlines() if x.strip()]
    frame_lines = [json.loads(x) for x in (tmp_path / "frames.jsonl").read_text().splitlines() if x.strip()]
    assert summary["decisions"] == len(dec_lines) == 8
    assert summary["frames"] == len(frame_lines) == 10
    assert src.closed == 1
    assert isinstance(summary["target_met"], bool)
    assert summary == json.loads((tmp_path / "summary.json").read_text())
    d0 = dec_lines[0]
    assert d0["elixir"] == 9.5 and d0["hand"] == ["zap", "archers", "knight", "giant"] and d0["context_text"] == "CTX"
    assert set(d0["timings_ms"]) == set(STAGES) and d0["key"] == "giant|left|back"
    assert summary["readiness_counts"] == {"match": 10}
    assert summary["decisions_by_status"] == {"dry_run": 8}
    assert summary["source_stats"] == {"frames": 10} and summary["perception_timing"] == {}
    assert summary["target_p95_ms"] == 300
    assert set(summary["decision_ms"]) == set(STAGES) and set(summary["frame_ms"]) == {"capture", "perception", "live_state"}
    assert summary["frame_ms"]["capture"] == {"p50": 2.0, "p95": 2.0}
    assert summary["decision_ms"]["total"]["p95"] == pytest.approx(8.0)


def test_summary_without_decisions():
    lp = _loop(builder=FakeBuilder(warmup=99))
    summary = lp.run(max_frames=3)
    assert summary["frames"] == 3 and summary["decisions"] == 0
    assert summary["target_met"] is None
    assert summary["decision_ms"]["total"] == {"p50": None, "p95": None}
    assert summary["decisions_by_status"] == {}


def test_run_measures_capture_when_source_has_no_grab_ms():
    class Bare:
        def frames(self):
            yield "f", 0.0, 0

        def close(self):
            pass

    lp = _loop(source=Bare(), builder=FakeBuilder(warmup=0))
    lp.run()
    assert lp.frames[0]["capture"] == pytest.approx(1.0)      # two FakeClock calls around next()
    assert lp.summary()["source_stats"] is None


def test_run_stops_on_max_seconds_relative_to_first_frame():
    class Offset(FakeSource):
        def frames(self):
            for i in range(50):
                yield "f", 100.0 + i * 0.1, i

    lp = _loop(source=Offset(), builder=FakeBuilder(warmup=0))
    summary = lp.run(max_seconds=1.0)
    assert summary["frames"] == 10


def test_run_keyboard_interrupt_stops_cleanly():
    class Interrupting(FakeSource):
        def frames(self):
            yield "f", 0.0, 0
            raise KeyboardInterrupt

    src = Interrupting()
    lp = _loop(source=src, builder=FakeBuilder(warmup=0))
    summary = lp.run()
    assert summary["frames"] == 1 and src.closed == 1


# Amendment 1: the executor follows a moved window ---------------------------------------------------------------------
class MovingSource(FakeSource):
    """rect_changes goes 0 -> 1 when frame 3 is produced (the real source bumps it before yielding the frame)."""

    def __init__(self, n=10, change_at=3):
        super().__init__(n)
        self.change_at = change_at
        self.rect_changes = 0
        self.rect = "rect0"

    def frames(self):
        for i in range(self.n):
            if i == self.change_at:
                self.rect = "rect1"
                self.rect_changes += 1
            yield f"frame{i}", round(i * 0.1, 3), i


class RectExecutor(FakeExecutor):
    def __init__(self):
        super().__init__()
        self.rects = []
        self.order = []                                   # ("rect", rect) / ("exec", t) in call order

    def update_rect(self, rect):
        self.rects.append(rect)
        self.order.append(("rect", rect))

    def execute(self, key, tile, t, hand=None):
        self.order.append(("exec", t))
        return super().execute(key, tile, t, hand=hand)


def test_rect_change_calls_executor_update_rect_once(tmp_path):
    ex = RectExecutor()
    lp = _loop(tmp_path, source=MovingSource(), builder=FakeBuilder(_state(elixir=9.5), warmup=0), executor=ex)
    summary = lp.run(max_frames=10)
    assert ex.rects == ["rect1"]
    assert [r["rect_changed"] for r in lp.frames] == [i == 3 for i in range(10)]
    assert summary["rect_updates"] == 1
    frame_lines = [json.loads(x) for x in (tmp_path / "frames.jsonl").read_text().splitlines() if x.strip()]
    assert [r["idx"] for r in frame_lines if r["rect_changed"]] == [3]
    # update_rect ran before the frame-3 decision (t = 0.3) was executed, and not again afterwards
    assert ex.order.index(("rect", "rect1")) < ex.order.index(("exec", 0.3))
    assert [o for o in ex.order if o[0] == "rect"] == [("rect", "rect1")]


def test_rect_change_two_bumps_between_frames_is_one_update():
    class Jumpy(MovingSource):
        def frames(self):
            for i in range(self.n):
                if i == self.change_at:
                    self.rect = "rect2"
                    self.rect_changes += 2
                yield f"frame{i}", round(i * 0.1, 3), i

    ex = RectExecutor()
    lp = _loop(source=Jumpy(), builder=FakeBuilder(warmup=0), executor=ex)
    assert lp.run(max_frames=10)["rect_updates"] == 1
    assert ex.rects == ["rect2"]


def test_rect_change_executor_without_update_rect_is_left_alone():
    ex = FakeExecutor()                                   # no update_rect attribute
    lp = _loop(source=MovingSource(), builder=FakeBuilder(warmup=0), executor=ex)
    summary = lp.run(max_frames=10)
    assert not hasattr(ex, "update_rect")
    assert summary["decisions"] >= 1 and summary["rect_updates"] == 1      # no AttributeError, still counted


def test_no_rect_changes_attribute_means_no_updates():
    ex = RectExecutor()
    lp = _loop(source=FakeSource(5), builder=FakeBuilder(warmup=0), executor=ex)
    summary = lp.run(max_frames=5)
    assert ex.rects == [] and summary["rect_updates"] == 0


# 5 -----------------------------------------------------------------------------------------------------------------
def test_no_candidates_holds_with_zero_rank_time():
    ex = FakeExecutor()
    rk = FakeKeyRanker()
    lp = _loop(builder=FakeBuilder(warmup=0), generator=FakeGenerator(keys=[]), ranker=rk, executor=ex)
    d = lp.step("f", 1.0, 0, 2.0)
    assert d.key == HOLD_KEY and d.tile is None and d.probs == [] and d.candidates == []
    assert d.timings_ms["rank"] == 0.0 and rk.calls == []
    assert ex.calls[0][:3] == (HOLD_KEY, None, 1.0)
    assert d.timings_ms["total"] == pytest.approx(sum(d.timings_ms[s] for s in STAGES if s != "total"))


def test_single_candidate_skips_ranker():
    rk = FakeKeyRanker()
    lp = _loop(builder=FakeBuilder(warmup=0), generator=FakeGenerator(keys=["zap|left|mid"]), ranker=rk)
    d = lp.step("f", 1.0, 0, 2.0)
    assert d.key == "zap|left|mid" and d.probs == [1.0] and rk.calls == []
    assert d.tile == [4.0, 8.0]                      # by_lane_zone fallback


# 6 -----------------------------------------------------------------------------------------------------------------
def test_max_decisions_stops_after_first_decision():
    src = FakeSource(10)
    lp = _loop(source=src, builder=FakeBuilder(_state(elixir=9.5), warmup=2))
    summary = lp.run(max_decisions=1)
    assert summary["decisions"] == 1 and summary["frames"] == 3 and src.closed == 1


# 7 -----------------------------------------------------------------------------------------------------------------
def _argv(*extra, deck=DECK):
    return ["--calib", "calib.json", "--deck", ",".join(deck), *extra]


def test_cli_live_and_dry_run_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        main(["--live", "--dry-run", *_argv()])


def test_cli_bad_deck_returns_2(capsys):
    assert main(_argv(deck=DECK[:7])) == 2
    out = capsys.readouterr()
    assert out.out.startswith(TOS_NOTE)
    assert "8" in out.err


def test_cli_usage_errors_return_2(capsys):
    assert main(_argv("--source", "video")) == 2                       # video needs --video
    assert main(_argv("--live", "--source", "video", "--video", "x.mp4")) == 2
    assert main(_argv("--rect", "1,2,3")) == 2
    capsys.readouterr()


def test_cli_defaults():
    args = build_parser().parse_args(_argv())
    assert args.dry_run is True and args.live is False
    assert args.fps == 10 and args.ranker == "score" and args.source == "mirroring"
    assert args.detector == "katacr" and args.detect_every == 3 and args.imgsz == 640
    assert args.planner == "rules" and args.d1_backend == "mlx" and args.seconds == 60
    assert args.max_decisions is None and args.out is None and args.hero_cards == ""
    assert build_parser().parse_args(_argv("--live")).live is True


# 8 -----------------------------------------------------------------------------------------------------------------
def test_import_isolation():
    code = ("import sys, clashbot.live\n"
            "names = ('mlx', 'mlx_vlm', 'mlx_lm', 'transformers', 'Quartz', 'cr_perception.detect')\n"
            "print(','.join(m for m in names if m in sys.modules))")
    proc = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == ""
    assert "CGEvent" not in (ROOT / "clashbot" / "live.py").read_text()


# 9 -----------------------------------------------------------------------------------------------------------------
class _Cands:
    def __init__(self, keys):
        self.keys = keys

    def generate(self, *a, **k):
        return [Candidate(key, 1.0, {}) for key in self.keys]


def test_fastloop_rank_keys_integration():
    ex = FakeExecutor()
    d = FastLoop(_Cands(KEYS), FakeKeyRanker(), ex, PRIOR).decide(_state(), 1.0, "CTX")
    assert d.choice == KEYS[1] and d.tile == [3.0, 2.0]
    assert ex.calls == [(KEYS[1], [3.0, 2.0], 1.0, None)]                 # FastLoop passes no hand
    d = FastLoop(_Cands(KEYS), ScoreRanker(), FakeExecutor(), PRIOR).decide(_state(), 1.0, "CTX")
    assert d.choice == KEYS[0]


def test_fastloop_single_candidate_skips_ranker():
    class Boom:
        def choose(self, *a, **k):
            raise AssertionError("ranker must not be called")

        choose_keys = choose

    d = FastLoop(_Cands(["zap|left|mid"]), Boom(), FakeExecutor(), PRIOR).decide(_state(), 1.0, "CTX")
    assert d.choice == "zap|left|mid"


def test_loop_cli_accepts_ranker_flags(monkeypatch):
    seen = []
    monkeypatch.setattr(loop_mod, "replay", lambda *a: seen.append(a) or 0)
    assert loop_mod.main(["--replay", "-abc", "--ranker", "d1", "--d1-backend", "transformers"]) == 0
    assert loop_mod.main(["--replay", "xyz", "--limit", "3"]) == 0
    assert seen == [("-abc", None, "d1", "transformers"), ("xyz", 3, "score", "mlx")]
    with pytest.raises(SystemExit):
        loop_mod.main(["--replay", "xyz", "--ranker", "bogus"])

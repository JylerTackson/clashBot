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


def _loop(tmp_path=None, source=None, builder=None, generator=None, ranker=None, executor=None, plan_text="PLAN",
          perception=None, min_gap_s=0.0, policy="trigger"):
    # min_gap_s=0 keeps the pre-amendment-2 behaviour for the older tests; the min-gap tests pass it explicitly.
    # policy="trigger" keeps the should_decide path the older tests were written for; the MAC-W tests pass "reactive".
    return LiveLoop(source or FakeSource(), perception or FakePerception(), builder or FakeBuilder(),
                    generator or FakeGenerator(), ranker or FakeKeyRanker(), executor or FakeExecutor(),
                    load_cards(), PRIOR, plan_text=plan_text, out_dir=tmp_path, clock=FakeClock(), min_gap_s=min_gap_s,
                    policy=policy)


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
                  load_cards(), PRIOR, filters={"exclude_game": "g1"}, clock=FakeClock(), policy="trigger")
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
    assert args.d1_model is None
    assert args.max_decisions is None and args.out is None and args.hero_cards == ""
    assert args.min_gap == 1.0 and args.sync_ocr is False
    assert build_parser().parse_args(_argv("--live")).live is True
    args = build_parser().parse_args(_argv("--min-gap", "0", "--sync-ocr"))
    assert args.min_gap == 0.0 and args.sync_ocr is True


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
    assert loop_mod.main(["--replay", "xyz", "--ranker", "d1", "--d1-model", "/m/ft"]) == 0
    assert seen == [("-abc", None, "d1", "transformers", None), ("xyz", 3, "score", "mlx", None),
                    ("xyz", None, "d1", "mlx", "/m/ft")]
    with pytest.raises(SystemExit):
        loop_mod.main(["--replay", "xyz", "--ranker", "bogus"])


# Amendment 4: --d1-model reaches D1Ranker from both CLIs (D1Ranker is monkeypatched: no model is ever loaded) ------
class RecordingD1:
    """Stand-in for clashbot.ranker.D1Ranker: records its constructor kwargs, scores like candidate 0."""
    inits: list = []

    def __init__(self, **kwargs):
        RecordingD1.inits.append(kwargs)

    def choose_keys(self, context, keys, plan=None):
        return RankResult(0, [1.0] + [0.0] * (len(keys) - 1), "recording")


def test_d1_model_defaults_to_none_in_both_parsers():
    assert build_parser().parse_args(_argv()).d1_model is None
    assert build_parser().parse_args(_argv("--ranker", "d1", "--d1-model", "/m/ft")).d1_model == "/m/ft"


def test_live_build_ranker_passes_d1_model(monkeypatch):
    import clashbot.live as live_mod
    import clashbot.ranker as ranker_mod

    RecordingD1.inits = []
    monkeypatch.setattr(ranker_mod, "D1Ranker", RecordingD1)
    args = build_parser().parse_args(_argv("--ranker", "d1", "--d1-model", "/m/ft", "--d1-backend", "transformers"))
    assert isinstance(live_mod._build_ranker(args), RecordingD1)
    live_mod._build_ranker(build_parser().parse_args(_argv("--ranker", "d1")))
    assert RecordingD1.inits == [{"model_path": "/m/ft", "backend": "transformers"},
                                 {"model_path": None, "backend": "mlx"}]
    assert isinstance(live_mod._build_ranker(build_parser().parse_args(_argv())), ScoreRanker)   # score ignores it


def test_live_main_records_d1_model_in_config(monkeypatch, tmp_path):
    import clashbot.live as live_mod
    import clashbot.ranker as ranker_mod

    RecordingD1.inits = []

    class StubLoop:
        def __init__(self, *a, **k):
            self.summary = _loop().summary()

        def run(self, **k):
            return self.summary

    monkeypatch.setattr(ranker_mod, "D1Ranker", RecordingD1)
    monkeypatch.setattr(live_mod, "_build_source", lambda args: FakeSource())
    monkeypatch.setattr(live_mod, "_build_perception", lambda args, source: FakePerception())
    monkeypatch.setattr(live_mod, "_build_plan_text", lambda args, deck: None)
    monkeypatch.setattr(live_mod, "load_cards", lambda: {})
    monkeypatch.setattr(live_mod, "load_placement_prior", lambda: PRIOR)
    monkeypatch.setattr(live_mod, "StateIndex", lambda: None)
    monkeypatch.setattr(live_mod, "KnowledgeGraph", lambda: None)
    monkeypatch.setattr(live_mod, "CandidateGenerator", lambda *a: FakeGenerator())
    monkeypatch.setattr(live_mod, "LiveLoop", StubLoop)
    out = tmp_path / "run"
    rc = main(_argv("--source", "video", "--video", "x.mp4", "--device", "cpu", "--ranker", "d1",
                    "--d1-model", "/m/ft", "--out", str(out)))
    assert rc == 0
    assert RecordingD1.inits == [{"model_path": "/m/ft", "backend": "mlx"}]
    assert json.loads((out / "config.json").read_text())["d1_model"] == "/m/ft"


def test_replay_passes_d1_model_to_d1ranker(monkeypatch):
    import clashbot.ranker as ranker_mod

    RecordingD1.inits = []
    monkeypatch.setattr(ranker_mod, "D1Ranker", RecordingD1)
    monkeypatch.setattr(loop_mod, "_load_game", lambda gid: ({"own_deck": DECK, "own_deck_key": None}, []))
    monkeypatch.setattr(loop_mod, "load_cards", lambda: {})
    monkeypatch.setattr(loop_mod, "StateIndex", lambda: None)
    monkeypatch.setattr(loop_mod, "KnowledgeGraph", lambda: None)
    monkeypatch.setattr(loop_mod, "CandidateGenerator", lambda *a: None)
    monkeypatch.setattr(loop_mod, "RuleBasedPlanner",
                        lambda: types.SimpleNamespace(pregame=lambda deck, key: types.SimpleNamespace(text="PLAN")))
    monkeypatch.setattr(loop_mod, "load_placement_prior", lambda: PRIOR)
    assert loop_mod.main(["--replay", "g1", "--ranker", "d1", "--d1-model", "/m/ft", "--d1-backend",
                          "transformers"]) == 0
    assert loop_mod.main(["--replay", "g1", "--ranker", "d1"]) == 0
    assert loop_mod.main(["--replay", "g1", "--d1-model", "/m/ft"]) == 0          # score ranker: no D1Ranker built
    assert RecordingD1.inits == [{"model_path": "/m/ft", "backend": "transformers"},
                                 {"model_path": None, "backend": "mlx"}]


# Amendment 2: min gap between decisions, each opponent play triggers once, perception.close, async_ocr ---------------
class TimedBuilder:
    """build() always returns a state; the opponent plays in `plays` are visible once t has reached their own t."""

    def __init__(self, elixir=5.0, plays=()):
        self.elixir = elixir
        self.plays = list(plays)
        self.t = 0.0

    def observe_events(self, events):
        pass

    def build(self, gs):
        state = _state(elixir=self.elixir)
        state["opponent"]["recent_plays"] = [{"t": pt, "card": "knight"} for pt in self.plays if pt <= gs.t]
        return state, "CTX"


class TimedPerception(FakePerception):
    def __init__(self):
        self.closed = 0

    def process(self, frame, t, idx):
        return types.SimpleNamespace(readiness="match", t=t)

    def close(self):
        self.closed += 1


class RangeSource(FakeSource):
    """Frames every 0.1 s over [t0, t1)."""

    def __init__(self, t0, t1):
        super().__init__(0)
        self.t0, self.t1 = t0, t1

    def frames(self):
        i = 0
        while round(self.t0 + i * 0.1, 3) < self.t1:
            yield f"frame{i}", round(self.t0 + i * 0.1, 3), i
            i += 1


def test_min_gap_default_is_one_second():
    lp = LiveLoop(FakeSource(), FakePerception(), FakeBuilder(), FakeGenerator(), FakeKeyRanker(), FakeExecutor(),
                  load_cards(), PRIOR)
    assert lp.min_gap_s == 1.0


def test_min_gap_spaces_decisions_at_least_one_second_apart():
    # elixir 9.5 keeps should_decide true on every frame; frames are 0.1 s apart
    lp = _loop(source=RangeSource(0.0, 3.0), builder=TimedBuilder(elixir=9.5), perception=TimedPerception(),
               min_gap_s=1.0)
    summary = lp.run()
    times = [d.t for d in lp.decisions]
    assert times == [0.0, 1.0, 2.0]
    assert all(b - a >= 1.0 for a, b in zip(times, times[1:]))
    assert summary["frames"] == 30 and summary["decisions"] == 3 and summary["skipped_min_gap"] == 27


def test_min_gap_zero_restores_decision_on_every_frame():
    lp = _loop(source=RangeSource(0.0, 3.0), builder=TimedBuilder(elixir=9.5), perception=TimedPerception(),
               min_gap_s=0.0)
    summary = lp.run()
    assert summary["decisions"] == summary["frames"] == 30 and summary["skipped_min_gap"] == 0


def test_min_gap_skip_returns_none_and_logs_frame():
    lp = _loop(builder=TimedBuilder(elixir=9.5), perception=TimedPerception(), min_gap_s=1.0)
    assert lp.step("f", 0.0, 0, 2.0) is not None
    assert lp.step("f", 0.5, 1, 2.0) is None
    assert lp.step("f", 1.0, 2, 2.0) is not None
    assert len(lp.frames) == 3 and lp.skipped_min_gap == 1


def test_opponent_play_triggers_one_decision_not_one_per_frame():
    state = _state(elixir=5.0)
    state["opponent"]["recent_plays"] = [{"t": 5.0}]
    cards = load_cards()
    assert loop_mod.should_decide(state, 5.5, 4.0, cards) is True        # the raw trigger would stay true for 2 s
    # a prior decision at t=4.0 (so the 3 s rule is quiet until 7.0); the play at t=5.0 is then visible for 2 s
    lp = _loop(source=RangeSource(4.1, 6.9), builder=TimedBuilder(plays=[5.0]), perception=TimedPerception(),
               min_gap_s=0.0)
    lp.last_decision_t = 4.0
    summary = lp.run()
    assert [d.t for d in lp.decisions] == [5.0] and summary["frames"] == 28


def test_opponent_play_blocked_by_min_gap_still_triggers_once_after_it():
    lp = _loop(source=RangeSource(4.1, 6.9), builder=TimedBuilder(plays=[5.0]), perception=TimedPerception(),
               min_gap_s=1.5)
    lp.last_decision_t = 4.0
    lp.run()
    assert [d.t for d in lp.decisions] == [5.5]                         # first frame after the gap, play still fresh


def test_second_opponent_play_triggers_a_second_decision():
    lp = _loop(source=RangeSource(4.1, 8.0), builder=TimedBuilder(plays=[5.0, 6.5]), perception=TimedPerception(),
               min_gap_s=1.0)
    lp.last_decision_t = 4.0
    lp.run()
    assert [d.t for d in lp.decisions] == [5.0, 6.5]
    assert lp._handled_opp_t == 6.5


def test_candidates_and_ranker_still_get_the_full_state():
    class Spy(FakeGenerator):
        def generate(self, state, t, ctx, n=4, **filters):
            self.seen = [p["t"] for p in state["opponent"]["recent_plays"]]
            return super().generate(state, t, ctx, n=n, **filters)

    gen = Spy()
    lp = _loop(generator=gen, builder=TimedBuilder(plays=[5.0]), perception=TimedPerception())
    lp.last_decision_t = 4.0
    assert lp.step("f", 5.0, 0, 2.0) is not None and gen.seen == [5.0]
    assert lp.step("f", 5.5, 1, 2.0) is None and gen.seen == [5.0]          # handled: the filtered copy hides it


def test_run_closes_perception_once():
    perc = TimedPerception()
    lp = _loop(source=FakeSource(5), builder=FakeBuilder(warmup=0), perception=perc)
    lp.run()
    assert perc.closed == 1


def test_run_closes_perception_after_keyboard_interrupt():
    class Interrupting(FakeSource):
        def frames(self):
            yield "f", 0.0, 0
            raise KeyboardInterrupt

    perc = TimedPerception()
    lp = _loop(source=Interrupting(), builder=FakeBuilder(warmup=0), perception=perc)
    lp.run()
    assert perc.closed == 1


def test_run_without_perception_close_is_fine():
    lp = _loop(source=FakeSource(3), builder=FakeBuilder(warmup=0))       # FakePerception has no close()
    assert lp.run()["frames"] == 3


@pytest.mark.parametrize("extra,expected", [([], True), (["--sync-ocr"], False)])
def test_build_perception_passes_async_ocr(monkeypatch, extra, expected):
    import cr_perception
    import clashbot.live as live_mod

    seen = {}

    class FakePerceptionClass:
        def __init__(self, **kwargs):
            seen.update(kwargs)

    monkeypatch.setattr(cr_perception, "Perception", FakePerceptionClass)
    args = build_parser().parse_args(_argv("--detector", "none", "--detect-every", "5", *extra))
    live_mod._build_perception(args, "SRC")
    assert seen == {"config": "calib.json", "source": "SRC", "detector": None, "detect_every": 5,
                    "async_ocr": expected}


# Amendment 3: hold instead of a low-elixir fallback candidate; --device defaults to mps when available ---------------
class FallbackGenerator(FakeGenerator):
    """Every candidate is a filler (`sources` has "fallback"), as CandidateGenerator pads with when knn/kg run dry."""

    def generate(self, state, t, ctx, n=4, **filters):
        return [Candidate(k, 0.0, {"fallback": 1}) for k in self.keys]


class FirstKeyRanker:
    def choose_keys(self, context, keys, plan=None):
        return RankResult(index=0, probs=[1.0] + [0.0] * (len(keys) - 1), model="fake")


def _fallback_loop(elixir, generator=None, tmp_path=None):
    return _loop(tmp_path=tmp_path, builder=FakeBuilder(state=_state(elixir=elixir), warmup=0),
                 generator=generator or FallbackGenerator(), ranker=FirstKeyRanker(), executor=FakeExecutor())


def test_fallback_candidate_below_9_5_elixir_holds_without_executing(tmp_path):
    lp = _fallback_loop(5.0, tmp_path=tmp_path)
    d = lp.step("f", 1.0, 0, 2.0)
    assert d is not None and d.key == HOLD_KEY and d.tile is None
    assert d.executor == {"t": 1.0, "key": "hold", "tile": None, "status": "hold_fallback", "clicks": [],
                          "sent": False}
    assert lp.executor.calls == []                               # the executor is never asked to tap
    assert lp.last_decision_t == 1.0 and lp.decisions == [d]     # still a recorded decision
    assert d.candidates[0]["sources"] == {"fallback": 1} and d.probs == [1.0, 0.0]
    assert set(d.timings_ms) == set(STAGES)
    line = json.loads((tmp_path / "decisions.jsonl").read_text().splitlines()[0])
    assert line["key"] == "hold" and line["executor"]["status"] == "hold_fallback"


def test_fallback_candidate_with_unknown_elixir_holds(monkeypatch):
    import clashbot.live as live_mod

    monkeypatch.setattr(live_mod, "should_decide", lambda *a, **k: True)     # the real trigger needs a legal card
    lp = _fallback_loop(None)
    d = lp.step("f", 1.0, 0, 2.0)
    assert d.key == HOLD_KEY and d.executor["status"] == "hold_fallback" and lp.executor.calls == []


def test_fallback_candidate_at_10_elixir_is_executed():
    lp = _fallback_loop(10.0)
    d = lp.step("f", 1.0, 0, 2.0)
    assert d.key == KEYS[0] and d.executor == {"t": 1.0, "key": KEYS[0], "tile": d.tile}
    assert [c[0] for c in lp.executor.calls] == [KEYS[0]]


def test_fallback_threshold_is_inclusive_at_9_5():
    assert _fallback_loop(9.5).step("f", 1.0, 0, 2.0).key == KEYS[0]
    assert _fallback_loop(9.4).step("f", 1.0, 0, 2.0).key == HOLD_KEY


def test_non_fallback_candidate_is_executed_at_low_elixir():
    lp = _loop(builder=FakeBuilder(state=_state(elixir=5.0), warmup=0), ranker=FirstKeyRanker())
    d = lp.step("f", 1.0, 0, 2.0)
    assert d.key == KEYS[0] and [c[0] for c in lp.executor.calls] == [KEYS[0]]


def test_only_the_chosen_candidate_matters_for_the_fallback_hold():
    class Mixed(FakeGenerator):
        def generate(self, state, t, ctx, n=4, **filters):
            return [Candidate(KEYS[0], 1.0, {"knn": 1.0}), Candidate(KEYS[1], 0.0, {"fallback": 1})]

    low = _loop(builder=FakeBuilder(state=_state(elixir=5.0), warmup=0), generator=Mixed(), ranker=FakeKeyRanker())
    assert low.step("f", 1.0, 0, 2.0).key == HOLD_KEY             # FakeKeyRanker picks index 1 = the fallback
    ok = _loop(builder=FakeBuilder(state=_state(elixir=5.0), warmup=0), generator=Mixed(), ranker=FirstKeyRanker())
    assert ok.step("f", 1.0, 0, 2.0).key == KEYS[0]


def test_summary_counts_hold_fallback_status():
    lp = _fallback_loop(5.0)
    lp.source = RangeSource(0.0, 9.0)                      # frames 0.1 s apart; should_decide re-fires every 3 s
    summary = lp.run(max_frames=61)                        # -> decisions at t = 0, 3.0, 6.0
    assert summary["decisions"] == 3 and summary["decisions_by_status"] == {"hold_fallback": 3}
    assert lp.executor.calls == []


@pytest.mark.parametrize("available,expected", [(True, "mps"), (False, "cpu")])
def test_resolve_device_follows_torch_mps_availability(monkeypatch, available, expected):
    import torch
    import clashbot.live as live_mod

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: available)
    assert live_mod.resolve_device(None) == expected


def test_resolve_device_respects_explicit_choice(monkeypatch):
    import torch
    import clashbot.live as live_mod

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert live_mod.resolve_device("cpu") == "cpu" and live_mod.resolve_device("mps") == "mps"


def test_cli_device_default_is_none_until_main_resolves_it():
    assert build_parser().parse_args(_argv()).device is None
    assert build_parser().parse_args(_argv("--device", "cpu")).device == "cpu"


@pytest.mark.parametrize("extra,mps,expected",
                         [([], True, "mps"), ([], False, "cpu"), (["--device", "cpu"], True, "cpu")])
def test_main_resolves_device_and_records_it_in_config(monkeypatch, tmp_path, extra, mps, expected):
    import torch
    import clashbot.live as live_mod

    seen = {}

    class StubLoop:
        def __init__(self, *a, **k):
            self.summary = _loop().summary()

        def run(self, **k):
            return self.summary

    def build_perception(args, source):
        seen["device"] = args.device
        return FakePerception()

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
    monkeypatch.setattr(live_mod, "_build_source", lambda args: FakeSource())
    monkeypatch.setattr(live_mod, "_build_perception", build_perception)
    monkeypatch.setattr(live_mod, "_build_ranker", lambda args: FakeKeyRanker())
    monkeypatch.setattr(live_mod, "_build_plan_text", lambda args, deck: None)
    monkeypatch.setattr(live_mod, "load_cards", lambda: {})
    monkeypatch.setattr(live_mod, "load_placement_prior", lambda: PRIOR)
    monkeypatch.setattr(live_mod, "StateIndex", lambda: None)
    monkeypatch.setattr(live_mod, "KnowledgeGraph", lambda: None)
    monkeypatch.setattr(live_mod, "CandidateGenerator", lambda *a: FakeGenerator())
    monkeypatch.setattr(live_mod, "LiveLoop", StubLoop)
    out = tmp_path / "run"
    rc = main(_argv("--source", "video", "--video", "x.mp4", "--out", str(out), *extra))
    assert rc == 0
    assert seen["device"] == expected
    assert json.loads((out / "config.json").read_text())["device"] == expected


# MAC-W: reactive act/hold policy (default) vs the old should_decide trigger -------------------------------------------
def _enemy_unit(tile, eta=None, tower="own_left"):
    return {"unit_class": "knight", "card": "knight", "side": "enemy", "tile": list(tile),
            "eta_tower": None if eta is None else {"tower": tower, "s": eta}}


def _threat_state(elixir=5.0, units=None):
    return {**_state(elixir=elixir), "units": units if units is not None else [_enemy_unit((5, 9), eta=2.0)]}


def _reactive_loop(state, generator=None, tmp_path=None, ranker=None, executor=None, **kw):
    return _loop(tmp_path=tmp_path, builder=FakeBuilder(state, warmup=0), generator=generator, ranker=ranker,
                 executor=executor, policy="reactive", **kw)


def test_policy_argument_selects_reactive_trigger_or_object():
    from clashbot.policy import ReactivePolicy

    def make(policy):
        return LiveLoop(FakeSource(), FakePerception(), FakeBuilder(), FakeGenerator(), FakeKeyRanker(),
                        FakeExecutor(), load_cards(), PRIOR, policy=policy)

    assert isinstance(make(None).policy, ReactivePolicy)
    assert isinstance(make("reactive").policy, ReactivePolicy)
    assert make("trigger").policy is None
    custom = ReactivePolicy(load_cards(), leak_elixir=7.0)
    assert make(custom).policy is custom


def test_reactive_hold_returns_none_and_is_counted():
    gen, ex = FakeGenerator(), FakeExecutor()
    lp = _reactive_loop(_state(elixir=5.0), generator=gen, executor=ex)
    assert lp.step("f", 1.0, 0, 2.0) is None
    assert lp.step("f", 1.1, 1, 2.0) is None
    assert gen.calls == [] and ex.calls == [] and not lp.decisions and lp.last_decision_t is None
    assert lp.policy_holds == {"hold": 2}
    assert lp.summary()["policy_holds"] == {"hold": 2} and lp.summary()["decisions_by_mode"] == {}


def test_reactive_no_affordable_card_counts_as_none():
    lp = _reactive_loop(_state(elixir=0.0))
    assert lp.step("f", 1.0, 0, 2.0) is None
    assert lp.policy_holds == {"none": 1}


def test_min_gap_check_runs_before_the_policy():
    lp = _reactive_loop(_state(elixir=10.0), min_gap_s=1.0)
    assert lp.step("f", 0.0, 0, 2.0) is not None
    assert lp.step("f", 0.5, 1, 2.0) is None
    assert lp.skipped_min_gap == 1 and lp.policy_holds == {}


def test_reactive_leak_plays_and_logs_mode_and_reason(tmp_path):
    ex = FakeExecutor()
    lp = _reactive_loop(_state(elixir=10.0), generator=FakeGenerator(keys=["knight|left|mid", "giant|left|back"]),
                        tmp_path=tmp_path, ranker=FirstKeyRanker(), executor=ex)
    d = lp.step("f", 1.0, 3, 2.0)
    assert d.mode == "leak" and "near full" in d.reason
    assert d.key == "giant|left|back" and [c["key"] for c in d.candidates] == ["giant|left|back"]    # leak: back/bridge
    assert [c[0] for c in ex.calls] == ["giant|left|back"]
    line = json.loads((tmp_path / "decisions.jsonl").read_text().splitlines()[0])
    assert line["mode"] == "leak" and line["reason"] == d.reason
    assert lp.summary()["decisions_by_mode"] == {"leak": 1} and lp.summary()["policy_holds"] == {}


def test_reactive_defend_filters_to_the_threat_lane_and_executes():
    ex = FakeExecutor()
    gen = FakeGenerator(keys=["knight|right|mid", "knight|left|mid", "fireball|right|mid"])
    lp = _reactive_loop(_threat_state(), generator=gen, ranker=FirstKeyRanker(), executor=ex)
    d = lp.step("f", 1.0, 0, 2.0)
    assert d.mode == "defend" and "knight" in d.reason and "left" in d.reason
    assert [c["key"] for c in d.candidates] == ["knight|left|mid"]               # wrong lane + blind spell dropped
    assert d.key == "knight|left|mid" and d.tile == [4.0, 8.0] and ex.calls[0][0] == "knight|left|mid"


def test_reactive_defend_with_every_candidate_filtered_records_hold_policy(tmp_path):
    ex, rk = FakeExecutor(), FakeKeyRanker()
    gen = FakeGenerator(keys=["fireball|right|mid", "zap|right|back"])           # blind spells only
    lp = _reactive_loop(_threat_state(), generator=gen, tmp_path=tmp_path, ranker=rk, executor=ex)
    d = lp.step("f", 1.0, 0, 2.0)
    assert d is not None and d.key == HOLD_KEY and d.tile is None and d.probs == []
    assert d.executor == {"t": 1.0, "key": "hold", "tile": None, "status": "hold_policy", "clicks": [], "sent": False}
    assert ex.calls == [] and rk.calls == []                                     # no tap, no ranker call
    assert d.mode == "defend" and d.reason
    assert lp.last_decision_t == 1.0 and lp.decisions == [d]
    assert [c["key"] for c in d.candidates] == ["fireball|right|mid", "zap|right|back"]    # the unfiltered ones
    assert d.timings_ms["rank"] == 0.0 and set(d.timings_ms) == set(STAGES)
    line = json.loads((tmp_path / "decisions.jsonl").read_text().splitlines()[0])
    assert line["key"] == "hold" and line["mode"] == "defend" and line["executor"]["status"] == "hold_policy"
    assert lp.summary()["decisions_by_status"] == {"hold_policy": 1}


def test_reactive_hold_policy_still_respects_the_min_gap():
    lp = _reactive_loop(_threat_state(), generator=FakeGenerator(keys=["fireball|right|mid"]), min_gap_s=1.0)
    assert lp.step("f", 1.0, 0, 2.0).key == HOLD_KEY
    assert lp.step("f", 1.5, 1, 2.0) is None and lp.skipped_min_gap == 1


def test_reactive_policy_ignores_the_opponent_play_trigger():
    # the old trigger fires on every fresh opponent play; the reactive policy waits for it to become a threat
    lp = _loop(source=RangeSource(0.0, 3.0), builder=TimedBuilder(elixir=5.0, plays=[1.0]),
               perception=TimedPerception(), policy="reactive")
    summary = lp.run()
    assert summary["decisions"] == 0 and summary["policy_holds"] == {"hold": 30}


def test_trigger_policy_keeps_the_old_behaviour():
    gen = FakeGenerator()
    lp = _loop(builder=FakeBuilder(_state(elixir=0.0), warmup=0), generator=gen, policy="trigger")
    assert lp.step("f", 10.0, 0, 2.0) is None and lp.policy_holds == {} and gen.calls == []
    ex = FakeExecutor()
    lp = _loop(builder=FakeBuilder(_state(elixir=5.0), warmup=0), executor=ex, policy="trigger")
    d = lp.step("f", 1.0, 0, 2.0)                       # quiet state, but the 3 s rule fires on the first frame
    assert d.key == "giant|left|back" and d.mode == "trigger" and d.reason == "should_decide"
    assert [c[0] for c in ex.calls] == ["giant|left|back"]
    assert lp.summary()["decisions_by_mode"] == {"trigger": 1} and lp.summary()["policy_holds"] == {}


def test_cli_policy_flag():
    assert build_parser().parse_args(_argv()).policy == "reactive"
    assert build_parser().parse_args(_argv("--policy", "trigger")).policy == "trigger"
    with pytest.raises(SystemExit):
        build_parser().parse_args(_argv("--policy", "bogus"))


@pytest.mark.parametrize("extra,expected", [([], "reactive"), (["--policy", "trigger"], "trigger")])
def test_main_passes_the_policy_to_the_loop(monkeypatch, tmp_path, extra, expected):
    import clashbot.live as live_mod

    seen = {}

    class StubLoop:
        def __init__(self, *a, **k):
            seen.update(k)
            self.summary = _loop().summary()

        def run(self, **k):
            return self.summary

    monkeypatch.setattr(live_mod, "_build_source", lambda args: FakeSource())
    monkeypatch.setattr(live_mod, "_build_perception", lambda args, source: FakePerception())
    monkeypatch.setattr(live_mod, "_build_ranker", lambda args: FakeKeyRanker())
    monkeypatch.setattr(live_mod, "_build_plan_text", lambda args, deck: None)
    monkeypatch.setattr(live_mod, "load_cards", lambda: {})
    monkeypatch.setattr(live_mod, "load_placement_prior", lambda: PRIOR)
    monkeypatch.setattr(live_mod, "StateIndex", lambda: None)
    monkeypatch.setattr(live_mod, "KnowledgeGraph", lambda: None)
    monkeypatch.setattr(live_mod, "CandidateGenerator", lambda *a: FakeGenerator())
    monkeypatch.setattr(live_mod, "LiveLoop", StubLoop)
    out = tmp_path / "run"
    assert main(_argv("--source", "video", "--video", "x.mp4", "--device", "cpu", "--out", str(out), *extra)) == 0
    assert seen["policy"] == expected
    assert json.loads((out / "config.json").read_text())["policy"] == expected

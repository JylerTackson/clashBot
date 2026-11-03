# MAC-G — live loop CLI `python3 -m clashbot.live` (spec for a Sonnet implementer)

Author: Opus · 2026-11-02 · Implements D-007, D-031. Read `docs/plans/MAC-00_common.md` first.
Default is **dry run**. Never run with `--live`; tests never construct a live executor.

## Owns (only these files)
```
clashbot/live.py            (new)
clashbot/loop.py            (small edits listed in §4)
tests/test_live_loop.py     (new)
```

## Contracts you consume (written in parallel by other packages — import them lazily inside `main()`/builders, never at module top, and use fakes in tests)
| from | object | use |
|---|---|---|
| B `cr_perception.sources` | `MirroringSource(target_fps=..., rect=None)` → `.frames()` yields `(frame, t, idx)`; `.rect` (WindowRect, points); `.last_grab_ms`; `.stats()` | capture |
| `cr_perception` (exists) | `Perception(config=calib_path, source=src, detector=det, detect_every=n)`; `.process(frame, t, idx) -> GameState`; `.drain_events() -> list[PlayEvent]`; `.timing_report()` | perception |
| A `cr_perception.detect` | `default_katacr_detector(imgsz=..., device=...)` | detector |
| C `clashbot.live_state` | `LiveConfig(own_deck, hero_cards)`, `LiveStateBuilder(cfg)`; `.observe_events(events)`; `.build(gs) -> (state, context_text) | None` | live state |
| D `clashbot.ranker` | `D1Ranker(backend=...)` with `.choose_keys(context, keys, plan) -> RankResult`; `ScoreRanker` (exists, `.choose(context, options, plan)`) | rank |
| F `clashbot.executor` | `MirroringExecutor.from_calib(calib, rect, live=False)`; `DryRunExecutor()`; both `.execute(key, tile, t, hand=...) -> dict` | execute |
| E `clashbot.planner` | `RuleBasedPlanner().pregame(deck, deck_key) -> Plan`; `LLMPlanner().pregame(deck, deck_key, ctx) -> Plan` | plan text |
| exists | `should_decide`, `CandidateGenerator(index, kg, cards).generate(state, t, ctx, n=4)`, `StateIndex()`, `KnowledgeGraph()`, `load_placement_prior()`, `placement_tile`, `option_text`, `load_cards` | |

## 1. `clashbot/live.py`
```python
STAGES = ("capture", "perception", "live_state", "trigger", "candidates", "rank", "execute", "total")
TOS_NOTE = "Live runs only in Training Camp or friendly battles (Supercell terms of service prohibit bots in normal play)."

@dataclass
class LiveDecision:
    t: float
    idx: int
    key: str
    tile: list[float] | None
    candidates: list[dict]        # [{"key", "score", "sources"}]
    probs: list[float]
    executor: dict                # the executor's returned entry
    timings_ms: dict              # every STAGES name

def rank_keys(ranker, context: str, keys: list[str], plan: str | None) -> tuple[int, list[float]]
class LiveLoop:
    def __init__(self, source, perception, builder, generator, ranker, executor, cards: dict, prior: dict,
                 plan_text: str | None = None, out_dir: Path | None = None, filters: dict | None = None,
                 clock=time.perf_counter): ...
    def step(self, frame, t: float, idx: int, capture_ms: float) -> LiveDecision | None
    def run(self, max_seconds: float | None = None, max_frames: int | None = None,
            max_decisions: int | None = None) -> dict
    def summary(self) -> dict
def build_parser() -> argparse.ArgumentParser
def main(argv: list[str] | None = None) -> int
```
- `rank_keys`: 0 keys → `ValueError`; 1 key → `(0, [1.0])` (d1 needs ≥ 2 options); ranker has `choose_keys` → `res = ranker.choose_keys(context, keys, plan)`; else `res = ranker.choose(context, [option_text(k) for k in keys], plan)`; return `(res.index, list(res.probs))`.
- `step` (all times from the injected `clock`, in ms):
  1. `gs = perception.process(frame, t, idx)`; `events = perception.drain_events()` → `perception` ms.
  2. `builder.observe_events(events)`; `built = builder.build(gs)` → `live_state` ms. Append a frame row `{"t", "idx", "readiness": getattr(gs, "readiness", None), "capture", "perception", "live_state"}` to the in-memory frame list (and to `frames.jsonl` if `out_dir`). `built is None` → return `None`.
  3. `state, ctx = built`; `go = should_decide(state, t, self.last_decision_t, cards)` → `trigger` ms; `not go` → return `None`.
  4. `cands = generator.generate(state, t, ctx, n=4, **filters)` → `candidates` ms.
  5. no candidates → key `"hold"`, probs `[]`, `rank` 0.0; else `i, probs = rank_keys(ranker, ctx, [c.key for c in cands], plan_text)`; key `cands[i].key` → `rank` ms.
  6. `tile = placement_tile(key, prior)`; `entry = executor.execute(key, tile, t, hand=state["own"]["hand"])` → `execute` ms.
  7. `total = capture + perception + live_state + trigger + candidates + rank + execute`. `self.last_decision_t = t`. Build `LiveDecision`; append; if `out_dir` write one JSON line to `decisions.jsonl`: `asdict(decision)` plus `"elixir": state["own"]["elixir"], "hand": state["own"]["hand"], "context_text": ctx`.
- `run`: iterate `source.frames()`; `capture_ms = getattr(source, "last_grab_ms", None)`; if `None` use the measured duration of the `next()` call. Stop when `t >= max_seconds`, frames ≥ `max_frames`, or decisions ≥ `max_decisions`; `KeyboardInterrupt` stops cleanly. Always `source.close()`; write `summary.json` if `out_dir`; return `summary()`.
- `summary()`: `{"frames", "decisions", "readiness_counts", "frame_ms": {stage: {"p50","p95"}} for capture/perception/live_state, "decision_ms": {stage: {"p50","p95"}} for every STAGES name, "target_p95_ms": 300, "target_met": decision total p95 < 300 (None if no decisions), "decisions_by_status": Counter of executor entry "status" (or "dry_run" when missing), "source_stats": source.stats() if available, "perception_timing": perception.timing_report() if available}`. Percentiles with `numpy.percentile`, rounded to 2; empty lists → `None`.

## 2. CLI
```
python3 -m clashbot.live [--dry-run | --live] --calib PATH --deck c1,...,c8 [--hero-cards a,b] [--deck-key KEY]
      [--source mirroring|video] [--video PATH] [--start S] [--end E] [--stride N] [--rect l,t,w,h] [--fps 10]
      [--detector katacr|none] [--detect-every 3] [--imgsz 640] [--device cpu|mps]
      [--ranker score|d1] [--d1-backend mlx|transformers] [--planner rules|llm|none]
      [--seconds 60] [--max-decisions N] [--out DIR]
```
- `--dry-run` / `--live` mutually exclusive; neither given → dry run. Print `TOS_NOTE` first, always.
- `--deck` must have 8 comma-separated slugs (else exit 2 with a message).
- Source: `mirroring` → `MirroringSource(rect=WindowRect(l,t,w,h,"manual") if --rect else None, target_fps=--fps)`; `video` → `VideoFrameSource(--video, --start, --end, --stride)` (requires `--video`).
- Detector: `katacr` → `default_katacr_detector(imgsz=--imgsz, device=--device)`; `none` → `None`. `Perception(config=--calib, source=src, detector=det, detect_every=--detect-every)`.
- Builder: `LiveStateBuilder(LiveConfig(own_deck=deck, hero_cards=heroes))`. `cards = load_cards()`; `StateIndex()`, `KnowledgeGraph()`, `CandidateGenerator(index, kg, cards)`; `prior = load_placement_prior()`.
- Ranker: `score` → `ScoreRanker()`; `d1` → `D1Ranker(backend=--d1-backend)`.
- Planner (once, at start): `rules` → `RuleBasedPlanner().pregame(deck, deck_key).text`; `llm` → `LLMPlanner().pregame(deck, deck_key).text`; `none` → `None`.
- Executor: `--live` → `MirroringExecutor.from_calib(calib, src.rect, live=True)` (raises `PermissionError` unless `CLASHBOT_ALLOW_LIVE=1`; catch → print it, return 5); dry run with a mirroring source → `MirroringExecutor.from_calib(calib, src.rect, live=False)` (logs the planned clicks); dry run with a video source → `DryRunExecutor()`.
- `--out` default `runs/poc/<YYYYmmdd-HHMMSS>`; write `config.json` = `vars(args)`; run `LiveLoop(...).run(max_seconds=--seconds, max_decisions=--max-decisions)`; print the decision-latency table `| stage | p50 ms | p95 ms |` and `target_met`. Return 0.

## 3. Tests — `tests/test_live_loop.py` (fakes only)
Fakes in the test file: `FakeSource` (yields N synthetic frames at t = 0, 0.1, …; `last_grab_ms = 2.0`; `stats()`; `close()` records the call), `FakePerception` (`process` returns `types.SimpleNamespace(readiness="match")`, `drain_events` returns `[]`, `timing_report` returns `{}`), `FakeBuilder` (`build` returns a fixed `(state, ctx)` built from `tests/test_runtime.py::_state` style dicts, or `None` for the first 2 frames), `FakeGenerator` (returns `Candidate`s), `FakeKeyRanker` (`choose_keys` → index 1), `FakeTextRanker` (`choose` only), `FakeExecutor` (records `(key, tile, t, hand)`), and a `FakeClock` advancing 1 ms per call.
1. `rank_keys`: 1 key → `(0, [1.0])`; key ranker used when it has `choose_keys`; text ranker gets `option_text` strings; 0 keys → `ValueError`.
2. `step` returns `None` while the builder returns `None`, and when `should_decide` is false (elixir 0 → no legal cards).
3. A decision: executor receives `hand=state["own"]["hand"]` and `tile = placement_tile(key, prior)` for a prior dict given in the test; `timings_ms` has every `STAGES` key and `total` equals the sum of the parts.
4. `run(max_frames=10)` with `out_dir=tmp_path`: `decisions.jsonl`, `frames.jsonl`, `summary.json` exist; the number of decision lines equals `summary["decisions"]`; `source.close()` was called; `target_met` is a bool.
5. No candidates → key `"hold"`, rank time 0.0.
6. `max_decisions=1` stops after the first decision.
7. CLI: `main(["--live", "--dry-run", ...])` → argparse `SystemExit`; `--deck` with 7 cards → return 2; `build_parser().parse_args([...])` defaults: dry run, `fps == 10`, `ranker == "score"`.
8. Isolation: `import clashbot.live` does not import `mlx`, `mlx_vlm`, `transformers`, `Quartz`, `cv2`-heavy detector modules (`cr_perception.detect`) — check `sys.modules` after popping them; `clashbot/live.py` source contains no `CGEvent`.
9. `loop.py` edits (§4): `FastLoop` with `FakeKeyRanker` picks candidate 1; with `ScoreRanker` picks 0; a single candidate skips the ranker (the ranker raises if called).

## 4. `clashbot/loop.py` edits (only these)
- `FastLoop.decide`: replace the ranking lines with `index, _probs = rank_keys(self.ranker, context_text, [c.key for c in candidates], self.plan.text if self.plan else None)` imported from `clashbot.live` **inside the method** (avoid an import cycle), and `choice = candidates[index].key`. The executor call stays `self.executor.execute(choice, tile, t)` (no `hand` here).
- `replay(game_id, limit, ranker_name="score", d1_backend="mlx")`: `ranker = ScoreRanker()` or `D1Ranker(backend=d1_backend)` (import inside). CLI: add `--ranker {score,d1}` (default score) and `--d1-backend {mlx,transformers}` (default mlx). Keep the `--replay` dash-id handling.
- Do not change `should_decide`, `Decision`, `_load_game` or the printed summary format.

## 5. Smoke (allowed once, after the symlinks; no models, no screen)
`$PY -m pytest -q tests/test_live_loop.py tests/test_runtime.py` and `$PY -m clashbot.live --help`.
Expected in your worktree only: `--source mirroring` and the detector/builder imports cannot run yet (packages A/B/C/F are merged later) — do not stub them in `live.py`.

## 6. Heavy run (Opus, at integration and Wave 4)
`python3 -m clashbot.live --dry-run --source video --video data/videos/<id>/<id>.mp4 --start S --end E --calib runs/parity/calib.json --deck ... --ranker d1` (recorded footage, no phone), then `--source mirroring` with the user in Wave 4. Target: decision total p95 < 300 ms.

## 7. Report
Per MAC-00 §5.

# MAC-AE — bigger LLMs: d1-2B decision LLM and a Qwen3-14B coach (specs for two Sonnet implementers)

Author: Opus · 2026-11-05 · Implements D-046. Read `docs/plans/MAC-00_common.md` first.
- D-045 holds: the decision LLM (d1) picks every move. The coach only writes plan text into d1's context.
- Two packages run in parallel: **E1** (coach) and **E2** (d1-2B support).
- MAC-AD runs at the same time (`policy.py`, `tactics.py`, `live_state.py`, one line in `live.py`).
- A GPU job runs until about 23:00 (`train_d1.py`). Before it finishes, use **CPU only** and never load a model on MPS. The model downloads go into the HF cache: `LiquidAI/d1-2B`, then `mlx-community/Qwen3-14B-4bit` (progress in `runs/overnight/downloads.log`).

---
## E1 — the coach: Qwen3-14B in the slow loop

### Owns
```
clashbot/planner.py   clashbot/live.py (planner wiring only: _build_plan_text, a coach worker, --planner choices/flags)
tools/data/build_coach_plans.py (new)   tools/data/build_d1_tac.py (only a --plans option)
tools/eval/bench_planner.py (coach timing)   tests/test_planner*.py   tests/test_coach.py (new)
```

### `clashbot/planner.py`
- `COACH_MODEL_ID = "mlx-community/Qwen3-14B-4bit"`.
- `class CoachPlanner`. Reuse `MLXChat`, and add an `enable_thinking=False` path. Qwen3 is a hybrid thinking model, and the coach must not think: pass `enable_thinking=False` to `apply_chat_template` when the tokenizer supports it, else append `/no_think` to the user turn. Strip any `<think>…</think>` from the output.
  - `pregame(own_deck, deck_key)`: the same as `LLMPlanner.pregame` (tools allowed), with the coach model.
  - `update(plan, state, t, tactics_lines=None, lookup=None) -> Plan`: **no tool calls**, for speed. The prompt (a constant `COACH_UPDATE_TEMPLATE`) holds:
    - our deck;
    - the opponent cards seen;
    - `render_context(state, t)`;
    - the tactics lines;
    - the lookup summary (up to 3: "pros in similar states played …");
    - the current plan.

    It asks for **at most 3 plain lines of ≤ 100 characters**: what to save for which threat, where to attack next, and when to spend. Mention only cards in our deck or seen opponent cards. Output passes through `clip_plan`.
- `lookup_summary(hits) -> list[str]`: from `StateIndex.search` hits, `"{card} {lane} {zone}"` of the top distinct action keys.

### Live: a background coach worker (`clashbot/live.py`)
- `--planner rules|llm|coach|none` (default `rules`, unchanged), `--coach-model ID` (default `COACH_MODEL_ID`) and `--coach-period 10.0` (game seconds).
- `CoachWorker` (a thread; one job at a time):
  - `submit(state, ctx, tactics_lines, lookup, t)` is non-blocking. If a job is running or not yet due, it does nothing.
  - "Due" means the first time, after `coach_period` game seconds since the last plan, or when a new opponent card appeared (reuse `should_update`).
  - On completion it publishes `latest_plan` (text plus game time plus latency) and appends `{t, plan, latency_s}` to `plans.jsonl` in the run dir.
  - Any exception is logged and the old plan is kept.
- `LiveLoop`:
  - when a coach worker is set, call `submit(...)` on decision steps (after the policy says act) and on hold steps, at most once per second;
  - use `latest_plan.text` (else the pregame plan) as `plan_text` for `rank_keys`;
  - log `plan_t` and `plan_age_s` per decision.
- **Never block `step`.** Measure the decision-latency impact in the smoke: MLX and d1 share the GPU.

### Offline coach plans for training (heavy run by Opus later)
- `tools/data/build_coach_plans.py [--dataset DIR] [--period 10] [--limit-games N] [--out <dataset>/coach_plans.jsonl]`. For each game of `<dataset>/games.jsonl`, walk the game's `states.jsonl` records in time. At every `period` of game time, plus at a new opponent card, call `CoachPlanner.update` with:
  - the record's state;
  - `TacticalEngine().analyse(state, t).lines`;
  - the lookup summary from `StateIndex.search(..., splits=("train",), exclude_game=game_id)`.

  Write `{game_id, t, plan, latency_s}`. The run must be resumable: skip `(game_id, t)` pairs that are already written.
- `build_d1_tac.py --plans PATH`: add the latest plan with `t_plan ≤ t` as `[plan] …`, via `d1_format.state_string(context, plan)`, to each line. Default: no plans (unchanged output).

### Tests (fake chat, no model)
- The coach prompt contains the tactics lines, the lookup summary and the deck.
- Thinking is disabled (`enable_thinking=False` is passed, or `/no_think` is appended) and `<think>` is stripped.
- At most 3 lines.
- `CoachWorker`: non-blocking submit; one job at a time; due logic (period, new card); latest plan published; exception keeps the old plan.
- `LiveLoop` uses the latest plan in `rank_keys`.
- `build_coach_plans` on a tiny synthetic dataset with a fake planner: cadence, resumability.
- `build_d1_tac --plans` adds `[plan]` and passes `validate`.

### Smoke
1. CPU unit tests.
2. **After** `runs/overnight/next2.log` shows `DONE`, and only if `pgrep -f "train_d1|clashbot.live"` finds nothing:
   - `bench_planner.py` with the coach model: load time, pregame and update latency, tokens/s, peak memory;
   - a 60 s **video** dry run, `--source video --video data/videos/Z8-4VhLjrGU/Z8-4VhLjrGU.mp4 --planner coach --ranker d1 --d1-model db/models/d1-ft-joint`: plans per minute, plan latency, and decision p50 / p95 with vs without the coach.
   - Never use `--live`; never touch the screen.

---
## E2 — d1-2B as the decision LLM

### Owns
```
clashbot/ranker.py   tools/train/train_d1.py   tools/eval/eval_d1_zero_shot.py (only a --dtype pass-through)
tests/test_train_d1.py   tests/test_ranker*.py   tests/test_d1_2b.py (new)
```

### Read the model code first
- The cached snapshot is `~/.cache/huggingface/hub/models--LiquidAI--d1-2B/snapshots/*/`: `modeling_d1.py`, `prompt.py`, `modeling_lfm3*.py`, `api.py`, `calibration.py`, `config.json`.
- d1-2B is LFM3 (24 layers: 13 gated short-conv, 5 sliding-window attention, 6 global attention), built on LFM3-VL-2B, 1.92B parameters. It exposes `system_one(state, questions, media=None)` with the same Decision Index question format as d1-omni-600M. The card recommends bf16 on MPS.
- Document the trainable text path, equivalent to MAC-T §0 for the 600M model, in the docstring of a new adapter: how ids/markers are built, where the option logits come from, and where the temperatures/calibration live (`calibration.json`?).

### `clashbot/ranker.py`
- `D1Ranker(model_path, backend, dtype="auto")`. `auto` = bf16 for LFM3 / d1-2B models (detected from `config.json` `model_type` / `architectures`), else today's behaviour (fp16 on MPS).
- Check that `probabilities()` parses d1-2B's `system_one` answers into the same `{name: [probs in criteria order]}`, and the noul as `[p_false, p_true]`.
- Live / eval CLI: `--d1-dtype auto|bf16|fp16|fp32`. Add the flag to `clashbot/live.py` only through the existing ranker builder. If that function is not yours to edit, report it, and Opus adds a one-line pass-through.

### `tools/train/train_d1.py` — an architecture adapter
- `--init LiquidAI/d1-2B` must work. Detect the architecture and use an adapter with the same interface the trainer uses today: `build_rows`, `option_logits`, trainable-parameter selection, temperature refit, save, reload check.
- `--train-layers N` counts from the top of the 24 LFM3 layers. The head is always trainable.
- **Precision:** frozen weights may be bf16; trainable parameters are fp32 (cast up), or all fp32 if `--fp32`.
- Gradient checkpointing if the model supports it (`--grad-ckpt`).
- Print a memory estimate before training. Abort with a clear message if it exceeds `--max-gb` (default 26).
- **Save:** `save_pretrained` plus copies of every `*.py`, `*.json`, `*.jinja` and tokenizer file of the snapshot, so the checkpoint reloads with `trust_remote_code`. Refit the temperatures the way d1-2B stores them; read its calibration code first.

### Tests (no real model, fast)
- A fake LFM3-like module set (same attribute names as the real adapter uses), covering:
  - architecture detection;
  - top-N layer freezing (the bottom layers are bit-identical after a step);
  - option logits with gradients;
  - save and reload of the remote-code files;
  - the memory-estimate abort.
- `@pytest.mark.slow` + `CLASHBOT_SLOW_TESTS=1`: the real d1-2B on CPU — 4 joint lines, 1 epoch, `--train-layers 1`.

### Smoke
- CPU: once `runs/overnight/downloads.log` shows `done LiquidAI/d1-2B`, run the slow test once on CPU, and `system_one` on 3 `joint_tac` val lines (CPU, fp32).
- **After** `next2.log` shows `DONE`, and if no `train_d1` / `clashbot.live` process runs:
  - `eval_d1_zero_shot.py --model LiquidAI/d1-2B --backend transformers --variant joint --limit 200`, on both `dataset/v1` (joint) and the shadow dataset `runs/eval_tac` (joint_tac);
  - report accuracy and ms p50 / p95 per decision on the M5;
  - also the same `--limit 200` for `d1-ft-joint`, for a same-subset comparison.

---
## Heavy runs (Opus, GPU serialised, after both packages)
1. Full zero-shot eval of d1-2B on `joint` and `joint_tac`, and the decision latency on the M5.
2. Fine-tune `d1-2b-ft-joint-tac`. Memory and the train-layers count depend on E2's estimate. Expect about 5–8 h, so an overnight run.
3. `build_coach_plans.py` over the dataset (Qwen3-14B, hours), then `build_d1_tac --plans`, then a fine-tune with plans.
4. A live match with `--d1-model <best> --planner coach`.

## Report
Per MAC-00 §5, per package.

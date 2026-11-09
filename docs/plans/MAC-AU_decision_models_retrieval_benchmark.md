# MAC-AU — more decision models and every retrieval method, on 5 held-out games: accuracy against the pro and latency (spec for three Sonnet implementers working in parallel)

Author: Opus · 2026-11-08 · Read `docs/plans/MAC-00_common.md` first. Code freeze D-051 holds: evaluation tooling only, no decision-code changes.

User: *"I want a benchmark for latency and accuracy for how many correct moves they pick compared to the pro. I also want a benchmark comparison for all the different retrieval methods we have for the KG and KB. We don't need to do all the videos for the benchmark, just compare them against like 5 games."*

Decision models (the user's picks):
1. Qwen3-4B zero-shot.
2. Qwen3-14B zero-shot.
3. Retrieval-augmented few-shot Qwen.
4. d1-2B fine-tuned.
5. Claude Haiku 5.5 and Claude Sonnet 5.5.

## Common ground (all three parts)

**The 5 games** (test split, the largest 4-way sets; 4 different videos; 232 lines):
`6qYQNJ1Uaeg-m0.4` (56), `5nOaTBlYLlg-m0.3` (53), `Zt6Onn89EVk-m0.0` (43), `SsOmbv1PfUg-m0.5` (41), `5nOaTBlYLlg-m0.0` (39).

- **Lines:** the rows of `dataset/v1/d1/joint_tac_v2/test.jsonl` for those games: the state text, 4 options with notes, and the gold option, which is the pro's play.
- **Line file:** write them once to `runs/bench/au/lines.jsonl` with a stable `line_id`, `game_id`, `state_id` (the dataset state id behind the line), `gold` (0–3), and the 4 option keys.
- **Who writes the line file:** Part B writes it first. Parts A and C read it. If they start before it exists, they build it with the same filter and check the result is identical.

**Accuracy** = the share of lines where the model picks the pro's play (the gold option). **Card accuracy** = the picked option has the pro's card. Report both with 95% CIs from a bootstrap over lines (10,000 resamples, seed 0). Five games are too few to bootstrap over games, so state that. Also report an exact McNemar test against d1-ft-joint-tac on the same lines.

**Latency** = wall time per decision, p50 / p95 / mean. Local models include tokenisation, the forward pass and parsing. API models include the network round trip. Report the cost per decision too, from usage times price.

**Predictions contract** (one file per model / variant): `runs/bench/au/preds/<model_id>.jsonl`, one row per line:
```
{line_id, game_id, gold, pred (0-3 or null), probs (list or null), latency_ms, in_tokens, out_tokens, cost_usd, raw (first 200 chars of the reply), error}
```
`pred` null = unparseable reply or error, which counts as wrong. Report the parse-failure rate.

**GPU discipline.** Accuracy and latency are measured in the same pass, so every local-model run must hold the global heavy lock:
`/usr/bin/lockf /private/tmp/claude-501/-Users-tylerjackson-Projects-clashbot/b32b0bc7-ec70-4c37-81b7-3ec6b49dd77b/scratchpad/heavy.lock <cmd>`. This covers d1, Qwen via MLX, KataCR and the full pytest suite. Hold it for one model or variant per acquisition. API calls and CPU work run in parallel without the lock, but the timed retrieval queries in Part C take the lock briefly.

**Secrets.** The Anthropic key is in `~/.clashbot/anthropic_api_key` (mode 600). Load it into `ANTHROPIC_API_KEY` for the process only, e.g. `ANTHROPIC_API_KEY="$(cat ~/.clashbot/anthropic_api_key)" <cmd>`. Never print, log, echo or write it anywhere, never put it in a file in the repo or in a run dir, and never include it in a report.

## Part A — LLM rankers: Qwen and Claude (one implementer)
Owns:
- `tools/eval/decision_prompt.py` (new): the one prompt builder all LLMs share;
- `tools/eval/bench_llm_rankers.py` (new): Qwen runner (MLX);
- `tools/eval/bench_claude_rankers.py` (new): Claude runner;
- `tests/test_decision_prompt.py` (new).

- **Prompt.** Render each line as d1 sees it: the state text including its `tactics:` / `KG:` lines, then the four options lettered A–D with their notes. End with: "Which option did the professional player choose? Answer with one letter: A, B, C or D." Give a short system prompt that explains the game and the option format. Parse the first standalone A–D letter of the reply.
- **Qwen3-4B zero-shot:** `mlx-community/Qwen3-4B-Instruct-2507-4bit` (in the HF cache; `HF_HUB_OFFLINE=1`) through `mlx_lm`. Greedy, max 8 new tokens, chat template.
- **Qwen3-14B zero-shot:** `mlx-community/Qwen3-14B-4bit`.
  - (a) Thinking off (`enable_thinking=False` in the chat template, or `/no_think`) on all 232 lines.
  - (b) Thinking on, on the first 60 lines (12 per game), with max 1,024 new tokens. Parse the letter after the thinking block.
- **Retrieval-augmented few-shot (Qwen3-4B):**
  - Retrieve the k = 3 most similar **train-split** pro states for each line. Use `clashbot.retrieval.StateIndex.search` with `splits=["train"]`, the line's state dict from `dataset/v1/states.jsonl` by `state_id`, and its `context_text`.
  - Give each retrieved state as an example: its context text, truncated to 600 characters, then "the pro played <action_key>".
  - Then the line's own prompt. Record k and the prompt length in tokens.
- **Claude Haiku 5.5 (`claude-haiku-5-5`) and Claude Sonnet 5.5 (`claude-sonnet-5-5`):**
  - Use the official `anthropic` Python SDK from the isolated venv `/private/tmp/claude-501/-Users-tylerjackson-Projects-clashbot/b32b0bc7-ec70-4c37-81b7-3ec6b49dd77b/scratchpad/claudeenv/bin/python` (anthropic 2.6.0 is installed there). Do not install anything into the project venv.
  - The script must not import torch, mlx or clashbot modules that pull them in; it reads `lines.jsonl` and the prompts written by `decision_prompt.py`. Have Part A's main script pre-render the prompts to `runs/bench/au/prompts.jsonl`.
  - Read the skill reference before writing it: `/private/tmp/claude-501/bundled-skills/2.1.312/4385f1b67b5ecc04b408f20a952c7238/claude-api/python/claude-api/README.md`.
  - **Thinking off:** Haiku takes `thinking={"type": "disabled"}`; Sonnet 5.5 rejects `disabled` with a 400, so use `thinking={"type": "between_tools"}`. Both use max_tokens 16. No temperature, top_p or top_k (they 400 on these models), no assistant prefill, and no `fallbacks` (it would swap the model being measured).
  - **Thinking on, same 60 lines as Qwen (b):** Haiku `thinking={"type": "adaptive"}` with `output_config={"effort": "medium"}`; Sonnet `thinking={"type": "adaptive"}` with `output_config={"effort": "medium"}`; max_tokens 4,000.
  - Check `stop_reason`: `refusal` means `pred` null with error "refusal". Catch the SDK's typed errors, most specific first (`RateLimitError`, `APIStatusError`, `APIConnectionError`). The SDK retries 429 and 5xx itself; set `max_retries=4`.
  - Run sequentially, so the latency is clean and there is no rate-limit noise.
  - Record `usage.input_tokens` / `output_tokens`. Cost per million tokens: Haiku 5.5 $0.10 in / $0.50 out; Sonnet 5.5 $2.00 in / $10.00 out.
  - **Budget:** stop and report if the projected total exceeds $15.
- **Model ids for the preds files:** `qwen3-4b`, `qwen3-14b`, `qwen3-14b-think60`, `qwen3-4b-rag3`, `claude-haiku-5-5`, `claude-haiku-5-5-think60`, `claude-sonnet-5-5`, `claude-sonnet-5-5-think60`.

## Part B — the d1 family, baselines and the scorer (one implementer)
Owns:
- `tools/eval/bench_d1_models.py` (new);
- `tools/eval/score_decision_models.py` (new);
- the loader fix for `db/models/d1-2b-ft-joint-tac`, inside `bench_d1_models.py` or a small helper under `tools/`;
- `tests/test_score_decision_models.py` (new);
- `docs/data/decision_models_v1.{json,md}` (new).

No change to `clashbot/ranker.py` unless the loader fix needs it there. In that case keep the change minimal and backward compatible, and say so.

- Write `runs/bench/au/lines.jsonl` first (Common ground).
- **d1-ft-joint-tac (600M, the live model):** the reference model, run through `clashbot.ranker.D1Ranker` (transformers, MPS).
- **d1 zero-shot (600M)**, for reference.
- **d1-2B fine-tuned (`db/models/d1-2b-ft-joint-tac`):**
  - Fix `KeyError: 'lfm3'`. The checkpoint's config `model_type` is not registered with transformers' `CONFIG_MAPPING`.
  - Register the custom config and model classes the checkpoint ships, or load them with `trust_remote_code` from its own code files, the way the 600M checkpoint loads.
  - Never modify the model directory. If a file must change, copy the directory to `runs/bench/au/d1-2b-loadable/` and change the copy.
  - Report what the fix was.
- **Baselines on the same lines:** random (expected 0.25 plus one seeded draw), popularity, placement prior, and LightGBM (from `docs/data/feature_ranker_preds_v2.json`, seed 0, filtered to the 5 games). Latency for these is the cost of the scoring call.
- **Scorer:** reads every `runs/bench/au/preds/*.jsonl` file, from Part A as well. It writes `docs/data/decision_models_v1.{json,md}`.
  - One table: model, size/type, accuracy [CI], card accuracy, McNemar p vs d1-ft-joint-tac, latency p50/p95, cost per decision, parse failures.
  - A second table for the 60-line thinking subset, with d1 and the non-thinking variants on the same 60 lines.
  - Rerunnable: Opus runs it again when Part A finishes.
- **CPU and memory:** local models run one at a time under the heavy lock (Common ground).

## Part C — every retrieval method for the KB and the KG (one implementer)
Owns:
- `tools/eval/bench_retrieval_methods.py` (new);
- `tests/test_bench_retrieval_methods.py` (new);
- `docs/data/retrieval_methods_v1.{json,md}` (new).

Read-only use of `clashbot/retrieval.py`, `clashbot/features.py`, `clashbot/text_embed.py`, `clashbot/kg.py`, `clashbot/kg2.py`, `tools/eval/eval_retrieval.py`, `tools/eval/kge_link_prediction.py` and `tools/eval/eval_answer_rules.py`. Reuse their functions; do not duplicate logic.

**Leakage:** the index holds every split, so always filter retrieval to `splits=["train"]` (or train+val, and say which) for these test games.

**KB, the pro-state database.** For each line's state, each method returns the top 20 states:
1. Dense exact: the 644-d structured vector, dot product.
2. BM25 (SQLite FTS5) over `context_text`.
3. RRF of dense + BM25 (k = 60). This is what the live bot uses.
4. bge-small text embeddings (`db/state_text_emb.npy`, from `text_embed.py`).
5. RRF of all three.
6. HNSW at ef 10, 40 and 160, over the dense vector.
7. Deck pre-filter, then dense.

Metrics per method:
- (a) hit@10 / card_hit@10 / MRR: does a retrieved state's pro play equal the line's gold play key, or the gold card?
- (b) **4-way accuracy as a decision model:** score each of the line's 4 options by the summed retrieval scores of the hits whose `action_key` equals the option key. Score ties in expectation, uniformly among the tied options (report coverage = the share of lines where any option has support).
- (c) Query latency p50 / p95. Time it under the heavy lock, after a warm-up.

**KG.** Methods:
- KG v1 `counter_scores` (wiki COUNTERS + mined ANSWERED_BY);
- KG v1 wiki COUNTERS only;
- KG v2 `answers_for`, the full version and each source alone (pro, role transfer, wiki, sim);
- KGE RotatE and TransE: retrain with the best config from `docs/data/kge_link_prediction_v1.json` and score (card, counters, threat);
- train popularity of the answer card;
- random.

Metrics per method:
- (a) On the threat → answer events of the 5 games (`dataset/v1/kg2/answer_events.jsonl`, rules from train only, as in `eval_answer_rules.py`): hit@1, hit@3 and MRR over the pro's deck, plus coverage.
- (b) **4-way accuracy as a decision model:** score each option by the method's score for (option card vs the line's recent enemy threats, i.e. the units heading to our towers in the state). Lines with no threat are tied (expected 0.25); report coverage.
- (c) Query latency p50 / p95.

**Output:** `docs/data/retrieval_methods_v1.{json,md}` with two tables (KB, KG), each with every method × the metrics. Also one combined table, "retrieval as a decision model" (4-way accuracy, coverage, latency), next to d1-ft-joint-tac on the same lines: the d1 number comes from Part B's preds when present, otherwise leave it for the scorer.

## Tests and finish (each part)
- Unit tests for the parsing, metrics and leakage filter on toy fixtures.
- Full suite green under the heavy lock: `../.venv/bin/python -m pytest -q --deselect tests/test_perception_latency.py -p no:cacheprovider`.
- Commit your own files on your worktree branch. Commit messages end with the two attribution lines. Do not push.
- Report: branch, commit, the headline table, wall time and (Part A) the total API cost.

## Amendment 1 (user, 2026-11-08 23:45): all 766 test decisions, thinking off only
User: *"can we just leave thinking off and do all 766 decisions"*
- **Lines:** every row of `dataset/v1/d1/joint_tac_v2/test.jsonl`: 766 lines, 27 test games, the same set as the 50.7% headline. This replaces the 5-game filter. `runs/bench/au/lines.jsonl` is rewritten with all 766 lines, in the file order, with the same fields.
- **No thinking variants:** drop `qwen3-14b-think60`, `claude-haiku-5-5-think60`, `claude-sonnet-5-5-think60` and the 60-line subset table. Qwen3-14B runs with thinking off. Claude runs with Haiku `{"type": "disabled"}` and Sonnet 5.5 `{"type": "between_tools"}`.
- **CIs:** cluster bootstrap over the 27 games (10,000 resamples, seed 0), as in `docs/data/stats_audit_v1`. Also report the exact McNemar test against d1-ft-joint-tac.
- **Part C:** all 766 lines, and the threat → answer events of all 27 test games.
- **Claude budget:** about $3–4 projected; the $15 stop stays.

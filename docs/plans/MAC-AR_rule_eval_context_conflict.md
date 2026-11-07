# MAC-AR — two course experiments: held-out test of the mined answer rules (E6) and the context-conflict test of the d1 ranker (E2) (spec for one Sonnet implementer)

Author: Opus · 2026-11-07 · Showcase Monday 2026-11-09. Code freeze D-051: no change to decision code. Read `docs/plans/MAC-00_common.md` first. The experiment designs are in `docs/research/course_gap_analysis.md` §7.3 ("E6" and "E2"); follow them, with the scope cuts below.

## Owns
```
tools/eval/eval_answer_rules.py (new)    tools/eval/context_conflict.py (new)
tests/test_eval_answer_rules.py (new)    tests/test_context_conflict.py (new)
docs/data/answer_rules_v1.{json,md} (new)    docs/data/context_conflict_v1.{json,md} (new)
```
Read-only use of `clashbot/kg2.py`, `clashbot/kg.py`, `clashbot/ranker.py`, `clashbot/d1_format.py`, `tools/data/mine_pro_answers.py` (import its functions; if the window sweep needs re-mining, write the re-mined rules to a temp dir, never over `dataset/v1/kg2` or `db/kg2.sqlite`).

## R1. E6 — are the mined "opponent played X → pro answered Y" rules predictive on held-out games?
- Rules from train events only; evaluate on val and test events of `dataset/v1/kg2/answer_events.jsonl` (check the split field and that no val/test event built a rule).
- For each event rank the pro's own 8-card deck (games.jsonl) with: (a) `KG2.answers_for(threat, deck)` (full), (b) each source alone and each leave-one-source-out (pro, role transfer, wiki, simulation — see `KG2.source_weights`), (c) wiki COUNTERS only (`KnowledgeGraph.counter_scores`), (d) train popularity of the answer card, (e) random in deck (expected value).
- Sweeps: support threshold n ≥ 1, 3, 5, 10; answer window 2, 4, 8 s (re-mine on train only).
- Metrics: hit@1, hit@3, MRR over the deck, coverage; game-cluster bootstrap CIs; the share of events where the answer "held" (if the field exists) by rank of the answer.

## R2. E2 — does d1 follow the notes even when they are wrong?
- Model `db/models/d1-ft-joint-tac` (transformers, MPS) on `dataset/v1/d1/joint_tac/{val,test}.jsonl` (and on `joint_tac_v2` if MAC-AP has built it by the time you run; say which).
- Conditions C0–C4 of the plan (as exported; notes and tactics/KG lines removed; notes shuffled across options; poisoned: the gold option's note swapped with a distractor's; transplanted tactics/KG block from another line). C5 (edit line) and CAD/AdaCAD are optional — only if C0–C4 are done and time remains.
- Metrics: accuracy per condition with game-cluster bootstrap CIs and paired McNemar vs C0; follow-rate (share of lines whose pick moves to the option the corrupted note favours); accuracy when the gold option's note is right vs wrong.
- Heavy: every d1 run under `/usr/bin/lockf /private/tmp/claude-501/-Users-tylerjackson-Projects-clashbot/b32b0bc7-ec70-4c37-81b7-3ec6b49dd77b/scratchpad/heavy.lock …`, and wait while `/private/tmp/claude-501/-Users-tylerjackson-Projects-clashbot/b32b0bc7-ec70-4c37-81b7-3ec6b49dd77b/scratchpad/HOLD_HEAVY` exists (the user is playing live). Run R1 (CPU) first.

## Tests
Toy fixtures with known answers for: ranking + hit@k/MRR, source ablation wiring, every perturbation of R2 (deterministic with a seed; the gold label is never changed), follow-rate. Full suite green under the lock: `../.venv/bin/python -m pytest -q --deselect tests/test_perception_latency.py -p no:cacheprovider`.

## Report
Branch, commit, files; R1 headline table (full vs each source vs wiki vs popularity vs random, val/test, CIs) and the sweep; R2 table (C0–C4 accuracy, follow-rate, McNemar); one paragraph each on what it means for the report (course links in the plan); deviations.

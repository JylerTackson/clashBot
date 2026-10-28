# Phase 3 — clean data, knowledge graph, retrieval and runtime skeleton

Status as of 2026-10-28. Decisions: `docs/decisions/DECISIONS.md` (D-001…D-020). Specs given to the implementers: `docs/plans/WP1…WP5`. Design: `docs/design/pipeline_review.md`. Course mapping and report outline: `docs/research/course_mapping.md`.

## Rebuild everything (on the Mac or here)

```bash
pip install numpy hnswlib pytest            # networkx later for graph algorithms
python3 tools/data/build_clean_dataset.py   # knowledge_base/ -> dataset/v1/{states,games,provenance}.jsonl, splits.json
python3 tools/data/build_kg.py              # -> dataset/v1/kg/{nodes,edges}.jsonl, card_stats.jsonl
python3 tools/data/build_documents.py       # -> dataset/v1/documents.jsonl (pro advice for the planner)
python3 tools/db/seed.py --hnsw             # -> db/clashbot.sqlite + state_features.npy + .hnsw
python3 tools/data/build_ranker_dataset.py  # -> dataset/v1/ranker/*.jsonl, act/*.jsonl, placement_prior.json
python3 tools/eval/eval_ranker_baselines.py # -> dataset/v1/ranker/baselines.json
python3 tools/data/build_d1_dataset.py --tokenizer db/hf/d1_tokenizer.json  # -> dataset/v1/d1/{joint,factorized,act}/*.jsonl (run after pytest; tokenizer: curl from huggingface.co/LiquidAI/d1-omni-600M)
python3 -m clashbot.loop --replay LGb0mz8Sb8w-m0.0 --limit 15   # fast loop on a recorded game, no models needed
```
`dataset/` and `db/` are build outputs (gitignored, D-013); `knowledge_base/` is the untouched source layer.

## What exists

| layer | contents | course link |
|---|---|---|
| clean states | 10,923 samples, repaired + flagged (D-011, D-016); 6,658 train-eligible; video-level splits 142/27/28 games | RETRO contamination → video splits |
| knowledge graph | 590 nodes (cards, evolutions, heroes, units, 68 decks, archetypes, games), 12,030 edges: COUNTERS (wiki, 4,562), SYNERGY, SUBSTITUTE, IN_DECK, SPAWNED_BY, mined ANSWERED_BY (129, train split only) | KG lectures; AMIE/TLogic-style mined rules |
| documents | 3,654 pro-advice docs (card/deck insights, match lessons, card usage, opponent reads, archetype guides) | RAG context for the slow-loop planner |
| database | SQLite (states, games, cards, KG, documents, FTS5/BM25) + 644-d structured feature vectors + HNSW index | Retrieval lectures (BM25, dense, RRF, HNSW, filtering) |
| ranker data | multiple-choice files (1 pro play + 3 hard negatives) for Laya/d1; act/hold export; placement prior | Two-stage RecSys, hard negatives, RAFT distractors |
| d1 training data | Decision Index format for d1-omni-600M (D-023): joint 4-way choice (3,865 / 807 / 797), factorized card+lane+zone (3,864 / 807 / 797), provisional act yes/no; states 90–733 tokens, no option over d1's per-option budget | TALLRec/P5-style LM-as-ranker; setwise ranking |
| runtime | `clashbot/`: candidates (kNN + KG + rules), ranker interface (+ Laya/d1 stubs), read-only tools (6), rule planner (+ Qwen stub), executor (dry run + Mac stub with tile→screen homography), fast loop with replay | Agents lecture: typed tools, read-only agent, executor separation |

## First results (val / test, no models trained yet)

- Retrieval search latency ≈ 40–50 ms; full fast-loop decision ≈ 41 ms p50 / 60 ms p95 (CPU, no models).
- Candidate recall@10 of the pro's exact action (card × lane × zone): kNN vote 0.36 / 0.31, popularity 0.49 / 0.40; at card level kNN 0.71 / 0.65 vs popularity 0.65 / 0.61. A tuned simple baseline is competitive at key level — the RecSys lecture's main lesson, now measured on our data.
- Multiple-choice baselines (4 options, random 0.25): placement prior 0.37 / 0.36, popularity 0.31 / 0.26. These are the numbers the fine-tuned Laya/d1 ranker has to beat.
- Act/hold: 92% of eligible samples are plays, so the act head needs denser negatives (D-020).

## Known limitations (for the report)

- One creator, 79% wins; perception noise (side attribution, unread hand slots, elixir misreads); 447 samples with invalid own decks (multi-deck videos) excluded from training.
- Wiki relation extraction is mention-based; evidence text is stored on every edge for auditing.
- Act/hold negatives are sparse until own games or dense perception timelines are added.

## Next steps (proposed, not started)

1. Mac setup: install deps, rebuild, calibrate the iPhone Mirroring window (4-point tile→pixel calibration) and capture → `cr_perception` → clean state (same code path as training, D-015).
2. Fine-tune d1-omni-600M (user's go-ahead pending, D-022) on `dataset/v1/d1/joint` and/or `factorized`; compare with the baselines above.
3. Slow-loop Qwen planner with the six read-only tools (ReAct), pre-game plan from documents + KG.
4. Memory and knowledge editing: episodic writes of own games, patch-versioned card edits, edit evaluation (reliability / locality / portability).
5. Report experiments: retrieval channels (BM25 vs features vs RRF; exact vs HNSW; pre- vs post-filtering), ranker vs baselines, KG embeddings for missing COUNTERS links, end-to-end latency and live trials.

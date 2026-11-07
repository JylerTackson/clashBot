# Course mapping audit: what is built, evaluated and live (2026-11-07)

Audited against git HEAD `015368f` (branch `claude/clash-royale-kb-phase-1-4rtfm6`), read-only: code, `docs/data/*`, `dataset/v1/eval/*`, `runs/poc/*`, `docs/mac/STATUS.md`, `docs/decisions/DECISIONS.md`, `docs/mac/FIX_LIST.md`. Nothing was re-run: every number below was copied from an existing output file or from STATUS (named in each cell). The pytest suite was not run (STATUS: 1284 passed, 7 skipped at MAC-AM merge).
Replaces the "status then" column of the table written by the other chat (`/private/tmp/.../scratchpad/paper_table.md`). The older plan `docs/research/course_mapping.md` (Opus, 2026-10-28) is unchanged and stale on the points listed in section D.

**Status legend.** `built+evaluated` = code located and an output file with numbers exists. `built (no eval)` = code located, no output file with numbers. `partial` = part of the row exists. `not built` = no code. `cite-only` = design justification, no experiment.
**"Used live?"** is traced for: `python -m clashbot.live --live --source mirroring ... --ranker d1 --d1-model db/models/d1-ft-joint-tac --policy fast --act-model db/models/act-gbdt`, tactics and kg2 on (defaults), `--planner rules` (default), `--arbiter none` (default).

## 0. Live-path trace (`clashbot/live.py`, `LiveLoop.step`, `main`)

| stage | code | on this command? |
|---|---|---|
| capture, perception | `cr_perception.sources.MirroringSource` (Quartz window id), `Perception` (KataCR YOLOv8 d1/d2 on MPS, OCR, Kalman tracker) | yes |
| clean live state | `clashbot/live_state.py` `LiveStateBuilder` (hand tracker, strict units) | yes |
| when to act | `--policy fast` = `FastActPolicy(ActGBDT h=2 s, threshold 0.47)` wrapping `ReactivePolicy(cards, kg=kg2)`; rules override the model in defend, counter, spell, leak; `save` and `none` always hold | yes |
| engine | `build_engine(cards, kg, kg2)` then `TacticalEngine.analyse` (`clashbot/tactics.py`, `unit_stats.simulate_lane`) | yes (off only with `--no-tactics`) |
| graph facts | `kg_fact_lines(kg2, state)`: up to 3 `KG:` lines appended to the ranker context via `rank_context` | yes (off only with `--no-kg2`) |
| candidates | `CandidateGenerator.generate(n=4, tac_n=2)`: up to 2 engine moves first, then lookup = exact kNN over the 644-d vector + BM25 (FTS5), fused by RRF k=60, plus KG v1 `counter_scores` (weight 0.5), then fallback legal keys; optional hero-ability option | yes |
| filters | `policy.filter_candidates`, hand check (`require_hand`), fallback hold below 9.5 elixir | yes |
| ranker | `D1Ranker(d1-ft-joint-tac)`; called only when 2 or more candidates survive (119 of 279 executed plays in the 3 latest runs; the other 160 had a single candidate) | yes |
| `[plan]` line | `--planner rules` without `--deck-key` gives `Deck: <8 slugs>`; it is appended to d1's state as `[plan] ...` although no training line has a `[plan]` (0 of `joint_tac/train.jsonl`) | yes (small train/serve mismatch, effect unmeasured) |
| tile | spell aim, engine tile, candidate tile, pro-median prior | yes |
| executor | `MirroringExecutor` Quartz clicks, needs `--live` and `CLASHBOT_ALLOW_LIVE=1` (D-031) | yes |
| NOT imported on this path | `clashbot/memory.py`, `clashbot/tools.py` (only `ToolContext` inside `--planner coach`), `LLMPlanner`, `CoachWorker`, `text_embed.py`, HNSW (`StateIndex(use_hnsw=True)` is never set), `RuleArbiter`, `LearnedActPolicy`, all `tools/eval/*` | offline or flag |

Real runs used `--d1-backend transformers` (config.json of every run after the fine-tune); the parser default is `mlx`, and no MLX conversion of the fine-tuned checkpoints exists in `db/models/`.

---

## A. Updated table per lecture group

### A1. Retrieval systems (Lectures 8-10)

| Element | What we have now | Status | Evidence (paths + numbers) | Used live? |
|---|---|---|---|---|
| BM25 [32] | SQLite FTS5 table `states_fts` over each state's `context_text`; `StateIndex.bm25` | built+evaluated | Spec `docs/plans/WP3_features_db_retrieval.md`, `MAC-R1_retrieval_ablation.md`. Eval `docs/data/retrieval_ablation_v1.{json,md}` (val n=916 / test n=902, depth 50): BM25 hit@10 0.142 / 0.159, card_hit@10 0.396 / 0.483, ndcg@10 0.039 / 0.064, 24.4 / 25.8 ms p50. On test BM25 beats the structured channel (0.159 vs 0.136; card 0.483 vs 0.412) | yes (`retrieval.py`, via `CandidateGenerator._channels`) |
| Dense vectors, bi-encoder idea [14] | Hand-built structured state vector, **644-d (verified: `db/state_features.meta.json`, shape 10923x644)**: scalars 3, phase 5, hand/deck/opp_deck/opp_recent/own_recent 120 each, towers 6, field 24, threat 6. It is a feature vector with dot-product kNN, not a learned bi-encoder | built+evaluated | `clashbot/features.py` (`FEATURE_VERSION=1`); ablation: struct hit@10 0.166 / 0.136, card_hit@10 0.434 / 0.412, ndcg@10 0.048 / 0.053, 1.3 / 2.1 ms p50 | yes |
| Reciprocal Rank Fusion [3] | `StateIndex.rrf`, k_rrf=60, fuses `knn` and `bm25` | built+evaluated | ablation: struct+bm25 (the runtime list) hit@10 0.138 / 0.160, card_hit@10 0.434 / 0.487, 25.8 / 28.0 ms; all3 (adds bge text) 0.156 / 0.170, card 0.461 / 0.514, 47.0 / 49.6 ms. Channels are within noise of each other | yes |
| HNSW [20]; exact search below ~1M | `db/state_features.hnsw` (M=16, efC=200, 29.8 MB); `StateIndex(use_hnsw=True)` | built+evaluated | `retrieval_ablation_v1.md`: recall@10 / ms p50 = ef10 0.8861 / 0.127; ef20 0.9493 / 0.192; ef40 0.9779 / 0.310; ef80 0.9889 / 0.530; ef160 0.9930 / 0.914; ef320 0.9943 / 1.498; exact 1.0000 / 1.506 | offline only (live uses exact; no flag) |
| Pre- vs post-filter (§2.2) | `StateIndex.mask(deck_key, phase, splits, exclude_game)` = pre-filter; post-filter simulated on HNSW top-K | built+evaluated | `retrieval_ablation_v1.md` (struct channel): deck filter: pool empty for 0.889 of queries; pre hit@10 0.332; post@10 hit 0.103, empty 0.500; post@50 0.223, empty 0.234; post@200 0.321, empty 0.005. Phase filter: pre 0.134; post@10 0.001, empty 0.873; post@50 0.054, empty 0.185 | offline only (live passes `filters={}`; only `require_action=True` is applied) |
| Recall@k, "BM25 is a strong baseline" [40] | Two measures. (1) candidate recall over all possible action keys, kNN-vote vs popularity. (2) hit@k of the retrieved list (above) | built+evaluated | `docs/data/ranker_baselines_v1.json` `candidate_recall` (val n=761 / test n=766): kNN-vote recall@10 **0.3614 / 0.3172**, card_recall@10 0.7201 / 0.6514; popularity recall@10 **0.4993 / 0.4021**, card 0.6728 / 0.6070. kNN recall@1 0.0696 / 0.0718. Popularity beats kNN at exact-key level, kNN wins at card level | yes (same generator; eval is offline) |
| Setwise LLM ranking [49] | d1-omni-600M fine-tuned to pick 1 of 4 options (setwise choice, listwise softmax loss) | built+evaluated | See C4. Zero-shot 0.2576 / 0.2206 -> fine-tuned `d1-ft-joint-tac` **0.5401 / 0.4974** (4-way, random 0.25, placement prior 0.4928 / 0.3838). Specs `MAC-D`, `MAC-T`, `MAC-H`; D-023, D-037, D-039 | yes (`D1Ranker`) |
| Learned text-embedding channel | `clashbot/text_embed.py`, BAAI/bge-small-en-v1.5, 384-d, `db/state_text_emb.npy` (10923x384) | built+evaluated; not integrated | ablation: text hit@10 0.132 / 0.143, card 0.397 / 0.436, 20.6 / 22.2 ms. +~20 ms for no clear gain, so not in the runtime (D-033, STATUS 2026-11-03) | offline only |
| Quantization, Matryoshka [11, 15] | No quantization or truncation experiment. 4-bit weights are used for the Qwen models (Qwen3-14B-4bit coach, Qwen3-4B-4bit planner); d1 MLX fp16 vs transformers agree on 99.7% of top answers (STATUS 2026-11-03) | cite-only | no output file | no |

### A2. RAG and recommendation (Lectures 11-14)

| Element | What we have now | Status | Evidence (paths + numbers) | Used live? |
|---|---|---|---|---|
| Two-stage serving (§4.4-4.5) | Stage 1 = engine moves + kNN/BM25 + KG + fallback (4 options); stage 2 = d1 | built+evaluated (per stage offline; end to end only as recorded live runs) | Recall: above. Ranker: C4/C5. End to end: 13 live runs, 931 decisions (C6); one controlled comparison only: STATUS 2026-11-05 match A (`--no-tactics`, 1 min 30 s, one tower -830) vs match B (tactics, 2 min 10 s, both towers -830). Executed-play sources, run 20261107-111554: engine-only 47, engine+kNN+KG 25, kNN+KG only 16, fallback 7 | yes |
| Hard negatives; pairwise/listwise losses [50, 51] | **Changed.** Each example = 1 pro play + 3 negatives in a 2x2 grid (card x placement): same placement other card (`G_card`), same card other placement (`G_place`), other card other placement (`G_diag`), 3,641 each in train. Loss = softmax cross-entropy over the 4 options (listwise). LightGBM LambdaRank in the feature ranker. No pairwise BPR | built+evaluated | `docs/data/ranker_stats_v1.json` (train 3,641 / val 761 / test 766; option_set_shortcut 0.25; lane-only rule 0.2764 train, 0.2658 val, 0.2722 test). Old N1/N2/N3 negatives leaked: option-key rule scored 0.848 / 0.853 / 0.852 (D-037). `tools/train/train_d1.py` `batch_loss` | training only |
| RAFT gold + distractor context [15] | Training format is 1 correct + 3 distractor *options*. There are no distractor *documents* in the context | partial | `dataset/v1/d1/joint*`; context lines are `tactics:` and `KG:` facts, not distractor passages | training only |
| TALLRec / P5, LM as recommender [69, 70] | Full fine-tune (not LoRA) of d1-omni-600M: all 16 trunk layers + head, 313,623,297 trainable params, fp32 on MPS | built+evaluated | `docs/data/d1_finetune_joint_{train,eval}_v1.json`, `..._joint_tac_...`, `..._joint_kg_...`, `..._act_train_v1.json`; C4. D-039 | yes (the shipped checkpoint is `d1-ft-joint-tac`) |
| "Tuned simple baselines often win" [5, 6, 7] | Placement prior, popularity, LightGBM/logreg/DCN-V2 rankers, copy-frequency (next card), LightGBM (act) all beat or match the LLM variants | built+evaluated | 4-way val / test: placement prior 0.4928 / 0.3838; popularity 0.4678 / 0.3251; logreg 0.5458 / 0.4761; **LightGBM LambdaRank 0.5484 / 0.4913**; DCN-V2 0.5261 / 0.4121; d1 fine-tuned 0.5243 / 0.4830. Next card: copy 0.207 / 0.243 vs SASRec 0.129 / 0.222. Act: LightGBM bacc 0.598 vs d1 head 0.558 (test, h=1 s). Old "0.37 vs random 0.25" is pre-D-037 | offline except the act LightGBM (live) |
| RETRO test contamination [12] | Split by video: `sha1(video_id)` order, 70/15/15 | built+evaluated (protocol) | `dataset/v1/splits.json`: 23 / 5 / 5 videos; `games.jsonl`: 142 / 27 / 28 games; `states.jsonl`: 7,693 / 1,574 / 1,656 states. Known open leak: KG facts on train lines are built from the same pro games (FIX_LIST #49) | n/a (data protocol) |
| Adaptive retrieval triggers (FLARE, Adaptive-RAG) [27, 31] | (a) Learned "act now?" trigger: LightGBM, 29 features, horizons 1/2/3 s, threshold 0.47 at 2 s, wrapped by rules. (b) d1 act head `d1-ft-act`. (c) `RuleArbiter` narrows d1's options when d1's top probability is low. (d) Coach re-plans every 20 s or on a new opponent card, in quiet moments only | partial: (a)(b)(c) built+evaluated, (d) built (no plan-quality eval); the "consult Qwen only when sources disagree" idea is not built | (a) `docs/data/act_gbdt_v1.md`: test AUC 0.659, balanced acc 0.607 vs rules 0.533 vs fast (rules+model) 0.577 at h=2 s; 0.045 ms. (b) `act_head_v1.md`: test bacc 0.558 vs rules 0.535. (c) `arbiter_v1.md`: d1 alone 52.4% / 48.3%, best arbiter 52.6% / 48.3% (no gain); d1 calibration: top prob >= 0.55 gives 65.2% on 290 val / 207 test lines. Specs `MAC-W`, `MAC-X`, `MAC-AB` | (a) yes; (b) flag `--policy learned`; (c) flag `--arbiter rules`; (d) flag `--planner coach` |
| Parametric memory bottleneck [8, 10] | Why retrieval and tactics feed d1. Measured: d1-omni-600M zero-shot is at chance, d1-2B zero-shot too on `joint` | cite-only (with supporting numbers) | `docs/data/d1_zero_shot_v1.json`: joint 0.2576 / 0.2206, factorized card 0.134 / 0.1619, lane 0.3693 / 0.3551, zone 0.3535 / 0.3316. d1-2B (200 lines each, `runs/smoke/E2/*.json`): joint 0.250 / 0.235; joint_tac 0.345 / 0.285 | n/a |
| GraphRAG / HippoRAG PPR [39, 41] | No personalized PageRank. Graph used as (i) 1-hop KG candidate channel (`counter_scores`, weight 0.5) and (ii) up to 3 `KG:` fact lines in the d1 context (GraphRAG-style structured context) | partial | `d1_finetune_joint_kg_{train,eval}_v1.json`: `d1-ft-joint-kg` 0.5309 / 0.4752 vs `d1-ft-joint-tac` 0.5361 / 0.5065 on the same KG-fact data (`d1_finetune_joint_tac_on_kg_eval_v1.json`): no gain; epoch 2 overfit (val loss 1.192 -> 1.671). No PPR code (grep: no pagerank/networkx) | yes ((i) and (ii)) |
| FM / DLRM / DCN-V2 [56-58] | `tools/eval/feature_ranker.py`: logreg, LightGBM LambdaRank, DCN-V2-style MLP on 783 features. FM and DLRM not implemented | built+evaluated | `docs/data/feature_ranker_v1.{json,md}`: val / test logreg 0.5458 +- 0.0034 / 0.4761 +- 0.0043; LightGBM 0.5484 +- 0.0140 / 0.4913 +- 0.0126; DCN-V2 0.5261 +- 0.0134 / 0.4121 +- 0.0132. Top LightGBM gain: `opt:log_pop_key` 0.1782, `opt:p_zone_given_card` 0.1240. Ablation test acc: full 0.4817, no_train_stats 0.5209, no_popularity 0.5039. Spec `MAC-R3` | offline only |
| Long context hurts [16, 17] | No experiment. Contexts are kept small by design (d1 state 90-733 tokens; 3-line coach plans; at most 3 `KG:` lines) | cite-only | `docs/data/d1_stats_v1.json` token counts | n/a |
| SASRec [61] | `tools/eval/next_card.py`: opponent next-card prediction, SASRec vs copy-frequency vs cycle vs popularity | built+evaluated | `docs/data/next_card_v1.md`: acc@1 val / test: copy_frequency **0.207 / 0.243**, SASRec 0.129 +- 0.014 / 0.222 +- 0.012, popularity 0.000 / 0.002; cycle rule best w = 0 (no gain). acc@3 copy 0.438 / 0.493; MRR 0.365 / 0.398. Oracle-seen ceiling 0.678 / 0.644. Spec `MAC-R4`, `WP7` | offline only (live opponent model is the KG2 deck model, see B) |

### A3. Knowledge graphs and neuro-symbolic reasoning (Lectures 5-7)

| Element | What we have now | Status | Evidence (paths + numbers) | Used live? |
|---|---|---|---|---|
| KG of entities and typed relations | Two graphs. **v1** (`dataset/v1/kg`, `db/clashbot.sqlite` `kg_edges`): 590 nodes, 12,030 edges, 12 relations. **v2** (`db/kg2.sqlite`, MAC-AK, D-048): 840 nodes, 27,112 edges, 19 relations, 1,734 per-level stat rows for 126 cards, 234 situation nodes, 16 ability nodes | built+evaluated (through KGE and the opponent-deck eval) | Counts: C2. Spec `WP2_kg_documents.md`, `MAC-AK_knowledge_graph_v2.md`. Evals: `kge_link_prediction_v1.json` (v1), `dataset/v1/eval/opp_model.md` (v2 component) | yes (v1 `KnowledgeGraph()` for candidates; v2 `KG2()` for engine prior, pro offsets, air reserve, `KG:` lines) |
| Symmetric vs asymmetric relations (§1.2) | **Not enforced in the stored graph**: SYNERGY has 3,136 directed edges, only 1,060 have a reverse edge; COUNTERS 3,804 distinct pairs, 940 have a reverse. Measured through the KGE asymmetry score | partial | `kge_link_prediction_v1.json`: asymmetry TransE 0.584, DistMult **0.0** (symmetric by construction), ComplEx 0.541, RotatE 0.628 | offline only |
| Rule mining, support/confidence (AMIE 3, TLogic) [23, 28] | Frequency-and-confidence mining of "opponent plays X, pro answers Y within a window" from train games only. v1: 129 ANSWERED_BY edges (window 4 s). v2 (`tools/data/mine_pro_answers.py`): 3,620 answer events, 287 (threat, answer) pairs, 234 situations, 450 sequences, with placement offsets and timing. No AMIE-style rule search or TLogic walks | built (no eval) | `dataset/v1/kg2/*.jsonl`; `FIX_LIST #42` (quality caveats: 41% of answers at the bridge, 34% unit-derived threats). Indirect: dropping the near-inverse ANSWERED_BY edges does not change KGE (RotatE 0.1604 vs 0.1588) | yes (engine prior via `KG2.answers_for`) |
| Neuro-symbolic hard constraints [41] | `game_rules.legal_action_keys` (card in hand, elixir, placement), `policy.filter_candidates` (no blind spells, real threats only), hand check, fallback hold. Perceived legality is noisy: the pro's own action passes the legality check only 0.542 (train) / 0.557 (val) / 0.597 (test) of the time | built (no eval beyond unit tests and the legality rate) | `clashbot/game_rules.py`, `clashbot/policy.py`; `ranker_stats_v1.json` `pos_in_legal_rate`; D-019 | yes |
| TransE / RotatE link prediction [2, 36] | Held-out wiki COUNTERS triples (3,043 train, 380 val, 381 test; 126 candidate cards), filtered ranks, 3 seeds, dim in {32, 64} x lr in {0.01, 0.001} chosen on val MRR | built+evaluated | `docs/data/kge_link_prediction_v1.{json,md}` (test, n=762 queries): RotatE **MRR 0.1588 +- 0.0020, H@1 0.0713, H@3 0.1549, H@10 0.3259**; ComplEx 0.1162; DistMult 0.0907; TransE 0.0819 (H@10 0.2380); popularity 0.1103 / H@10 0.2385; random 0.0223 / 0.1129. no_inverse setting: RotatE 0.1604. Spec `MAC-R2` | offline only |
| Evaluation audit: equal tuning, random tie-breaking [34, 37] | Same grid and budget for all four models, 3 seeds, filtered setting, realistic (tie-aware) rank; random baseline fixed for ties (FIX_LIST #5) | built+evaluated (protocol) | same file; `FIX_LIST #5` | offline only |
| Temporal KG, CyGNet copy mechanism [46] | Next-card experiment with a copy-frequency baseline (the CyGNet intuition) and SASRec; KG v2 holds 450 PLAYED_AFTER edges and an opponent-deck model. No temporal-KG embedding model, no CyGNet network | partial | `next_card_v1.md` (A2); opponent-deck model: B | opponent-deck model yes; next-card experiment offline |
| Semantic loss [41] | Not implemented. (The legality filter is applied at inference, not as a training penalty) | not built | grep: no violation/semantic-loss code | no |
| Scene graphs, ESCA [20, 25] | Cite. Project evidence that perception dominates failures: hand read `?` made 15 of 27 decisions skip (run 20261105-172506, FIX_LIST #26/#27); live-state parity vs the dataset on one game (100 records): phase 0.62, towers 0.77, elixir 0.55, hand slots 0.57, units Jaccard 0.3385, median feature cosine 0.6714 | cite-only | `docs/data/live_state_parity_v1.json`; FIX_LIST #12, #26, #30, #36, #57 | n/a |
| ChatRule [29] | Not built | not built | no code | no |

### A4. Agents, memory and knowledge editing (Lectures 15-17)

| Element | What we have now | Status | Evidence (paths + numbers) | Used live? |
|---|---|---|---|---|
| Typed JSON-schema tools; BFCL [1-3] | 6 read-only tools with argument validation (`clashbot/tools.py`: get_game_state, search_similar_states, kg_query, card_info, search_documents, get_current_plan). The Qwen planner makes **0 tool calls** | built (no eval of tool use) | `docs/data/planner_bench_v1.json` (Qwen3-4B-4bit, 5 decks): mean_tool_calls 0.0, tool_error_rate 0.0, pregame 3.31 s p50, update 2.63 s p50. Coach prompt says "call no tool". Spec `MAC-E` | no (only `--planner llm`) |
| Separate execution and checks (§7.3) | **Not a stub any more.** `MirroringExecutor` sends Quartz clicks to iPhone Mirroring; double guard `--live` + `CLASHBOT_ALLOW_LIVE=1` checked at construction and before every tap (D-031); D-050 spec: taps only in a battle whose banner reads Trainer (launcher not built yet); shot recorder verifies taps | built+evaluated (live) | `clashbot/executor.py`; STATUS 2026-11-03: taps register from t ~ 32 s; MAC-AF run: 86% of taps landed (vs ~60%); run 20261106-233907: 95/100 taps landed, 6/6 ability taps fired. 746 of 931 logged decisions have `sent: true` | yes |
| ReAct [4] | ReAct loop exists (`LLMPlanner.run`, up to 6 steps, 2 per update) and is benchmarked, but the model never calls a tool | partial | `clashbot/planner.py`; `planner_bench_v1.json` (0 tool calls, 0 fallbacks) | no (flag `--planner llm`) |
| Reflexion [5] | No post-game reflection generator. Only the store operation `write_lesson(source="reflection")` exists and nothing calls it | not built | `clashbot/memory.py` | no |
| Lost in the Middle, Context Rot, ACE playbook, MemGPT [12, 13, 22, 23] | Context discipline only (small prompts); no playbook, no paging | cite-only | none | n/a |
| Episodic / semantic / procedural memory (§4.1) | Episodic: pro-state DB (10,923 states) for retrieval is live; own-game episodes are never written (MemoryStore used only in `tools/eval/edit_eval.py`). Semantic: KG v1/v2 (live) plus patch-versioned card facts in `MemoryStore` (offline). Procedural: lessons table exists, empty | partial | `clashbot/memory.py`, `tests/test_memory.py`; no `db/memory.sqlite` exists | KG and state DB yes; MemoryStore no |
| Mem0 ADD / UPDATE / DELETE [25] | `MemoryStore.write_lesson(op in ADD, UPDATE, DELETE, NOOP)` with `memory_log` | built (no eval) | `clashbot/memory.py`, `tests/test_memory.py`; no output file | no |
| Knowledge-editing criteria [46] | Synthetic +1-elixir edit on 5 cards, one fresh store per edit, pooled | built+evaluated (synthetic, deterministic store) | `docs/data/edit_eval_v1.{json,md}`: reliability n=15: 1.000 vs no-edit 0.000; generalization n=465: 1.000 vs 0.000; locality cards n=2500 and states n=1500: 1.000 / 1.000; portability n=465: 1.000 vs 0.067; versioning and rollback True on 5/5. Spec `MAC-R5`, D-034 | offline only |
| Non-parametric editing vs ROME/MEMIT/SERAC [49, 51, 53] | Only the non-parametric route (versioned external fact store). No weight-editing or LoRA-edit comparison | partial | `edit_eval_v1.md`; `clashbot/memory.py` | offline only |
| Knowledge conflicts, ClashEval [64, 67] | `conflict_weight` / `reweight_hits` down-weight pre-patch hits on edited cards | built+evaluated (synthetic) | `edit_eval_v1.md`: share of top-10 hits using the edited card 0.124 (before) -> 0.002 (after re-weighting), n=1245 | offline only (not wired into live retrieval) |
| Trajectory-level evaluation (§7.2) | Per-run `decisions.jsonl`, `frames.jsonl`, `summary.json`, `holds.jsonl`, screenshots at each decision and +0.5 s / +1.5 s. No win rate with variance: 1 documented win | partial | C6. Only recorded result: run 20261107-013441, 3 games, game 3 won in overtime (STATUS 2026-11-07 night) | n/a |
| Multi-agent failure modes [44] | One decision LLM (d1) + deterministic engine + optional coach LLM that never picks a move (D-045); no debate | cite-only | D-045, D-046 | n/a |

---

## B. Built but not in the old table

| Component | Closest course topic | Status | Evidence (paths + numbers) | Used live? |
|---|---|---|---|---|
| Tactical engine `clashbot/tactics.py` (1,750 lines), `card_roles.py` + `resources/card_roles.json` (122 cards), `tactics_weights.json` | Neuro-symbolic reasoning / agent tool the LLM reasons over (D-042, D-045) | built+evaluated | `docs/data/tactics_eval_v1.md`: top-1 card 10.4% val / 9.0% test (push situations 19.1% / 18.0%); 4-way tactics ranker 27.5% / 25.5% (random 25%): agreement with the pro is at chance level. Its value is shown downstream: d1 trained with its notes +1.6 / +1.4 points (C4). Tuning `tactics_tuning_v1.md`: val objective 0.5129 -> 0.5321, test 0.4606 -> 0.4758 | yes |
| Lane simulation / threat model `unit_stats.simulate_lane`, per-level stats `resources/unit_stats.json` (MAC-AH) | Model-based planning, world model as a tool (LATS-style rollout without search) | built (no eval against ground truth) | Spec `MAC-AH_unit_stats_threat_model.md`; STATUS: lone L15 balloon 3,162 tower HP unanswered, 351 with a musketeer behind the tower, 0 with a tesla pull. Feeds 8,951 STOPS/KILLS/PULLS edges in KG v2 | yes |
| Tracker ETA to tower (attack range, card speed; MAC-AL T) | Perception quality (ESCA lecture) | built+evaluated | `runs/smoke/AL/eta.md` (video Z8-4VhLjrGU, 240 s, 70 tracks, 11 arrivals, 384 frames): median abs error all 1.49 s -> 0.30 s; melee 1.30 -> 0.30; ranged 2.19 -> 0.40; flyer 2.30 -> 0.20; share within 1 s 34% -> 80%. Not copied to `docs/data/` | yes |
| Reactive policy `clashbot/policy.py` (1,576 lines): modes defend, leak, punish, counter, spell, save, none | Symbolic gating of a neural ranker (neuro-symbolic); trigger | built+evaluated | `docs/data/act_policy_v1.json` (early, test n=1369); on dense data `act_gbdt_v1.md` test bacc rules 0.533 (h=2 s). Specs `MAC-W`, `MAC-AC`, `MAC-AD`, `MAC-AG`, `MAC-AL`, `MAC-AM` | yes (inside `FastActPolicy`) |
| LightGBM act model `clashbot/act_model.py`, `tools/train/train_act_gbdt.py` | Learned trigger (FLARE / Adaptive-RAG "when to retrieve"); tuned simple baseline beats the LLM head | built+evaluated | `docs/data/act_gbdt_v1.{json,md}`: 29 features; h=1 s test AUC 0.632 bacc 0.594; h=2 s 0.659 / 0.607; h=3 s 0.672 / 0.613; rules 0.529 / 0.533 / 0.530; fast (rules + model) 0.562 / 0.577 / 0.583. Apples to apples with the d1 head (test n=3,221, h=1 s): gbdt 0.598 vs d1 learned 0.558. Predict 0.045 ms p50. Top gain h=2 s: own_elixir 0.380, phase_idx 0.225. Model dir `db/models/act-gbdt` (3 small boosters). Spec `MAC-AB` | yes (`--policy fast`; offline "fast" is the live logic, 0.577 not 0.607) |
| d1 act head `db/models/d1-ft-act` (top 8 layers, 168,883,713 trainable, 2 epochs on 8,405 balanced lines) | LM as a trigger | built+evaluated | `act_head_v1.md`, `d1_finetune_act_train_v1.json`: test bacc 0.558 (val 0.573), act rate 0.429 vs pro 0.206; epoch 1 -> 2 val bacc 0.5088 -> 0.5725. Spec `MAC-X` | flag `--policy learned` |
| Dense wait/act dataset `tools/data/build_dense_act.py` (D-040) | Negative sampling for the trigger; RETRO-safe splits | built+evaluated | `dataset/v1/act_dense`: 39,875 lines (train 29,036 / val 5,560 / test 5,279), act rate 0.204; 33/33 videos; STATUS 2026-11-05. Spec `MAC-V` | n/a (training data) |
| KG v2 facts in the LLM context (`KG2.facts_lines`, `kg_fact_lines`) | GraphRAG-style structured context | built+evaluated (no gain) | A2 GraphRAG row. Present in 102 of 125 decisions of run 20261107-111554 (`kg_lines`). Example line: `KG: archers L15 x2 445hp 164/hit; answers mini-p-e-k-k-a (role of elite-barbarians, n=4), tesla (role of furnace, n=6)` | yes |
| Opponent-deck model `clashbot/opp_model.py`, `KG2.predict_opp_deck` (K3) | Next-entity prediction on a temporal KG; popularity vs model | built+evaluated | `dataset/v1/eval/opp_model.md` (not in `docs/data/`), leave-one-game-out, 155 games, recall over hidden cards: k=1 0.234 (popularity 0.130, random 0.059); k=2 0.291 (0.124, 0.051); k=3 0.327 (0.118, 0.043); test split only k=1 0.318 (n=20). Precision@3 k=1 0.267 vs 0.170 | yes (`KG: opp likely also has ...` line, air reserve in policy) |
| d1 fine-tune family: `d1-ft-joint`, `-tac`, `-kg`, `-act`, `d1-2b-ft-joint-tac` | TALLRec / P5; RAFT-like distractor options; model scale | built+evaluated; the 2B has no test number | C4. 2B: top 8 layers, 486,913,024 trainable, 3 epochs, 21,634 s, val 4-way 0.4192 (best epoch 3); the eval job crashed (`KeyError: 'lfm3'` loading the checkpoint with transformers, `runs/overnight/next3.log`), so no test accuracy. STATUS never reports it | only `d1-ft-joint-tac` |
| Qwen3-14B coach `CoachWorker`, `CoachPlanner` (D-046) | Slow-loop planner; agents; context management | built+evaluated (latency only; no plan-quality eval) | Run 20261106-133529 (900 s): 21 plans, 0 errors, 5.98 s p50 / 6.86 s p95, pregame 20.3 s, period 20 s (`summary.json` `coach`). Peak 9.1 GB, 3-line plans (STATUS 2026-11-06). Plans are not in any training line: `coach_plans.jsonl` was never built | flag `--planner coach` (used once) |
| Rule arbiter `clashbot/arbiter.py` (R7) | Adaptive-RAG router / Self-RAG evaluator | built+evaluated (no gain) | A2 trigger row; `arbiter_v1.md`: tactics-first (non-LLM) 51.2% / 46.9%; knn-first 46.8% / 45.6%; default arbiter 50.2% / 47.6% | flag `--arbiter rules` |
| Pro-advice document corpus: 3,654 documents (card_insight 2,807, deck_insight 182, deck_why 68, match_card_usage 197, match_lessons 197, match_opponent 197, archetype_guide 6), `documents_fts` | RAG corpus | built (no retrieval eval) | `db/clashbot.sqlite` `documents`; used by `RuleBasedPlanner` only when `--deck-key` is given | no (the command passes no `--deck-key`, so the plan is `Deck: ...`) |
| Hero ability option (MAC-AI, D-049) | Action-space extension gated by simulation | built+evaluated (live check) | Button detected 26/26 recorded frames, 0 false positives (STATUS 2026-11-06); 6/6 taps fired in run 20261106-233907; last run: 1,399 button frames, 0 offered | yes (offered only when the engine proposes it) |
| Shot recorder and review `clashbot/recorder_shots.py`, `tools/eval/review_shots.py` (MAC-AF) | Trajectory-level evaluation tooling | built+evaluated | `shots` blocks in `summary.json` (e.g. 125 decisions, 95 taps, 315 images, 0 errors) | flag `--record-shots` (on in the last 4 runs) |
| Perception stack `cr_perception/` (KataCR YOLOv8 d1 + d2, 87.8 MB each; OCR; Kalman tracker; hand tracker) | Scene graphs / ESCA | built+evaluated | Parity numbers in A3; latency in C7 | yes |

Already covered as rows in A (no separate entry): retrieval ablation, KGE link prediction, next-card prediction, knowledge-editing eval, feature ranker.
Specified but not built (spec only): demo launcher `clashbot.demo` (MAC-AN, D-050) and decision dashboard (MAC-AO).

---

## C. Headline facts with sources

### C1. Dataset

| item | value | source |
|---|---|---|
| States | **10,923** (play 6,901, key 3,539, periodic 483); train-eligible 6,658 (4,689 / 989 / 980) | `dataset/v1/MANIFEST.json`; `docs/data/CLEANING_REPORT_v1.md` |
| Games / videos | **197 games** from **33 videos**, one creator | `provenance.jsonl`; `splits.json` |
| Split (by video, sha1 order, 70/15/15) | videos 23 / 5 / 5; games 142 / 27 / 28; states 7,693 / 1,574 / 1,656 | `splits.json`, `games.jsonl`, `states.jsonl` |
| Pro results | 162 win, 16 loss, 19 unknown (82.2% of 197). `PHASE3_OVERVIEW.md` says 79% | `games.jsonl` |
| Match time | about 12 match-hours | STATUS 2026-11-04, D-040 |
| 4-way ranker set | 3,641 / 761 / 766 (skipped no grid 123 / 10 / 14; position impossible 431 / 145 / 122) | `docs/data/ranker_stats_v1.json` |
| Dense act set | 39,875 lines: 29,036 / 5,560 / 5,279 (act rate 0.196 / 0.233 / 0.217) | STATUS 2026-11-05 |
| Documents | 3,654 | `db/clashbot.sqlite` |
| Retrieval store | 10,923 x 644 float32 vectors; 10,923 x 384 bge-small vectors | `db/state_features.meta.json`, `db/state_text_emb.meta.json` |
| Flags | deck_invalid 447, action_unknown 1,668, placement_unknown 2,165, units_side_repaired 1,916 | `CLEANING_REPORT_v1.md` |

### C2. Knowledge graph, v1 vs v2

| | v1 (`dataset/v1/kg`, `db/clashbot.sqlite`) | v2 (`db/kg2.sqlite`, `meta` table) |
|---|---|---|
| Nodes | **590**: game 197, unit 135, card 126, deck 68, evolution 42, hero 16, archetype 6 | **840**: v1 types + situation 234 + ability 16 |
| Edges | **12,030** | **27,112** |
| Relations | COUNTERS 4,562; SYNERGY 3,136; GAME_OWN_CARD 1,613; GAME_OPP_CARD 1,401; IN_DECK 544; GAME_DECK 181; SPAWNED_BY 154; SUBSTITUTE 141; ANSWERED_BY 129; HAS_ARCHETYPE 111; EVOLUTION_OF 42; HERO_OF 16 | all v1 relations except ANSWERED_BY now 287, plus STOPS 8,464; CO_OCCURS 5,142; PLAYED_AFTER 450; SIMILAR_ROLE 365; KILLS 283; PULLS 204; HAS_ABILITY 16 |
| Edge sources | wiki text, KB, mined | wiki 7,839; sim 8,951; deck_cooc 5,142; kb_table 4,078; pro_mined 737; role_transfer 365 |
| Extras | none | 1,734 stat rows (126 cards), `answer_events.jsonl` 3,620, `answers.jsonl` 287, `situations.jsonl` 234, `sequences.jsonl` 450 |
| Live | `KnowledgeGraph()` (counter channel) | `KG2()` (engine, policy, `KG:` lines) |

### C3. Model sizes

| model | size | role |
|---|---|---|
| LiquidAI d1-omni-600M | about 600M params; fine-tuned checkpoints 2.35 GB fp32 each; 313,623,297 trainable (16 trunk layers + head) | decision LLM (live: `d1-ft-joint-tac`) |
| d1 act head | 168,883,713 trainable (top 8 layers) | optional trigger |
| LiquidAI d1-2B (LFM3) | 1.92B; checkpoint 3.84 GB bf16; 486,913,024 trainable (top 8 layers) | trained, not evaluated, not live |
| Qwen3-14B-4bit | about 9 GB, peak 9.1 GB | optional coach |
| Qwen3-4B-Instruct-2507-4bit | | optional planner (`--planner llm`) |
| KataCR YOLOv8 d1 + d2 | 87.8 MB each | detector |
| BAAI/bge-small-en-v1.5 | 384-d | offline text channel |
| LightGBM act model | 29 features, 3 boosters (rounds 4 / 21 / 75), json 47 KB / 237 KB / 842 KB | live trigger |
| Feature ranker | 783 features | offline |
| Hardware | Apple M5, 32 GB (D-025) | |

### C4. d1 fine-tunes: zero-shot vs fine-tuned (4-way, random 0.25; n 761 val / 766 test)

| model | init, data | epochs, trainable | val loss by epoch | val / test acc | ms p50 / p95 | source |
|---|---|---|---|---|---|---|
| d1-omni-600M zero-shot (MLX fp16) | - | - | - | **0.2576 / 0.2206** | 44.3 / 78.8 (val) | `d1_zero_shot_v1.json` |
| `d1-ft-joint` | d1-omni-600M, `joint` | 3 epochs, 313.6M; 13,241 s | 1.1587, 1.0908, 1.1987 (acc 0.4599, 0.5138, 0.5256, best = 3) | **0.5243 / 0.4830** | 67.0 / 95.6 | `d1_finetune_joint_{train,eval}_v1.json` |
| `d1-ft-joint` on `joint_tac` (notes unseen in training) | | | | 0.4796 / 0.4700 | 63.2 | `dataset/v1/eval/d1_eval_transformers_d1-ft-joint_on_tac.json`; STATUS |
| **`d1-ft-joint-tac`** (shipped) | `d1-ft-joint`, `joint_tac` | 2 epochs, 313.6M; 4,554 s, peak 22.75 GB | 1.1203, 1.2938 (acc 0.5361, 0.5401, best = 2); T = 2.45 | **0.5401 / 0.4974** (rebuilt MAC-AD data 0.5375 / 0.4935; KG-fact data 0.5361 / 0.5065) | 64.5 / 97.0 | `d1_finetune_joint_tac_{train,eval}_v1.json`, `d1_finetune_joint_tac_on_kg_eval_v1.json`, `dataset/v1/eval/d1_eval_*_rebuilt.json` |
| `d1-ft-joint-kg` | `d1-ft-joint-tac`, `joint_tac` + KG facts | 2 epochs; 4,984 s | 1.1916, 1.6708 (acc 0.5309, 0.5138, best = 1) | 0.5309 / 0.4752 | 75.5 / 105.2 | `d1_finetune_joint_kg_{train,eval}_v1.json` |
| `d1-ft-act` | `d1-ft-joint`, `act_dense_bal` | 2 epochs, 168.9M; 6,286 s | 0.6459, 0.6788 (bacc 0.5088, 0.5725) | test bacc 0.558 (not 4-way) | ~70 | `d1_finetune_act_train_v1.json`, `act_head_v1.md` |
| d1-2B zero-shot (200 lines per split) | | | | joint 0.250 / 0.235; joint_tac 0.345 / 0.285 | 247-266 / 305-354 | `runs/smoke/E2/*.json` |
| `d1-2b-ft-joint-tac` | d1-2B, `joint_tac` | 3 epochs, 486.9M; 21,634 s, peak 26.14 GiB | 1.3385, 1.3030, 1.2335 (acc 0.3482, 0.3876, 0.4192) | val 0.4192 only; test never computed | | `db/models/d1-2b-ft-joint-tac/train_report.json`, `runs/overnight/next3.log` |

Other 4-way references (val / test): random 0.25; popularity 0.4678 / 0.3251; placement prior 0.4928 / 0.3838; LightGBM LambdaRank 0.5484 / 0.4913; tactics ranker 27.5% / 25.5%; option-set shortcut 0.25 / 0.25 (was 0.848 / 0.853 / 0.852 before D-037).

### C5. Other evaluated results, one line each

| experiment | headline | file |
|---|---|---|
| Retrieval channels | all within noise; best test hit@10 all3 0.170 (47-50 ms), struct+bm25 0.160 (26-28 ms) | `retrieval_ablation_v1.md` |
| HNSW | ef40 recall@10 0.9779 at 0.310 ms vs exact 1.506 ms | same |
| Filtering | deck pool empty for 0.889 of queries; post@10 empty 0.500 (deck), 0.873 (phase) | same |
| KGE | RotatE test MRR 0.1588 +- 0.0020, H@10 0.3259 | `kge_link_prediction_v1.md` |
| Next card | copy-frequency 0.207 / 0.243 > SASRec 0.129 / 0.222 | `next_card_v1.md` |
| Knowledge edit (synthetic) | reliability 1.000 vs 0.000; portability 1.000 vs 0.067; conflict share 0.124 -> 0.002 | `edit_eval_v1.md` |
| Act model | LightGBM test AUC 0.659, bacc 0.607 (h=2 s) vs rules 0.533 | `act_gbdt_v1.md` |
| Opponent deck model | recall 0.234-0.327 vs popularity 0.118-0.130 | `dataset/v1/eval/opp_model.md` |
| ETA | median abs error 1.49 s -> 0.30 s | `runs/smoke/AL/eta.md` |
| Live-state parity | feature cosine 0.6714 (one game) | `live_state_parity_v1.json` |

### C6. Live-run history (`runs/poc/2026*`, 13 runs, first 2026-11-03, latest 2026-11-07 11:15)

Totals: **13 runs (12 live, 1 mirrored dry run), 931 logged decisions, 746 with `sent: true`**, 10 runs have `summary.json`. Latency here = whole decision pipeline (capture + perception + live state + trigger + candidates + rank + execute), ms, from `decisions.jsonl` `timings_ms.total`.

| run | what | frames | decisions | taps sent | p50 / p95 ms | summary.json |
|---|---|---|---|---|---|---|
| 20261103-192225 | first live, score ranker, user deck v1 (macOS blocked clicks for ~30 s) | 221 | 28 | 26 | 527 / 850 | yes |
| 20261103-192709 | live match 2 (MPS detector) | 1,817 | 25 | 13 | 441 / 624 | yes |
| 20261105-172506 | d1-ft-joint + reactive | 874 | 27 | 8 | 236 / 710 | no |
| 20261105-193555 | D-044 deck, `--no-tactics` (match A) | 520 | 56 | 42 | 422 / 673 | no |
| 20261105-194101 | D-044 deck, tactics (match B) | 1,409 | 80 | 80 | 393 / 577 | yes |
| 20261105-205720 | dry run 15 s | 150 | 0 | 0 | - | yes |
| 20261105-205934 | live 300 s | 2,533 | 61 | 53 | 331 / 492 | yes |
| 20261106-133529 | d1-ft-joint-tac, `--policy fast`, **coach** | 6,885 | 181 | 161 | 371 / 791 | yes |
| 20261106-135724 | D-047 deck, fast policy | 8,341 | 61 | 41 | 263 / 572 | yes |
| 20261106-170722 | + screenshots | 8,376 | 56 | 43 | 303 / 486 | yes |
| 20261106-233907 | + kg2; 4 full games + part of a fifth | 7,151 | 116 | 100 | 336 / 506 | yes |
| 20261107-013441 | MAC-AL code; **3 games, first win (game 3, overtime)**; crashed when the window closed (#56) | 6,698 | 115 | 84 | 341 / 552 | no |
| 20261107-111554 | MAC-AM code | 7,169 | 125 | 95 | 339 / 572 | yes |

Overall decision latency p50 / p95: 347 / 613 ms (all 931 decisions). Three latest runs, executed plays (n=279): 364.0 / 551.1 ms, of which the two-tap click sequence is 184.2 / 188.3 ms and d1 (when called, n=119) 92.7 / 174.7 ms. The 300 ms p95 target is not met in any run since the click time counts (`target_met: false`).

### C7. Latency by stage, run 20261107-111554 (`summary.json` `decision_ms`, p50 / p95 ms)

capture 21.2 / 32.1; perception 102.4 / 230.0; live_state 0.13 / 0.33; trigger 1.5 / 8.7; candidates 26.1 / 56.9; rank 0.0 / 136.7 (0.0 median because most decisions have one candidate); execute 183.3 / 188.1; total 338.9 / 571.8. Source fps 7.97 of 10 target (1,831 frames dropped); detector 100.5 ms mean per call, hand reader 33.1 ms mean.

---

## D. Claims in the old table that are wrong or outdated

| # | Old row | Old claim | Current truth |
|---|---|---|---|
| 1 | Setwise LLM ranking; TALLRec/P5 | "Data built; training pending" | Trained (D-039): zero-shot 0.2576 / 0.2206 -> 0.5401 / 0.4974 (`d1-ft-joint-tac`). Full fine-tune, not LoRA |
| 2 | Hard negatives | 3 negatives = neighbours' plays, same card other spot, other card same spot | Those options leaked the answer (rule on option keys scored 0.848 / 0.853 / 0.852, D-037). Replaced by a 2x2 card x placement grid (`G_card`, `G_place`, `G_diag`), option-set rule now 0.25 |
| 3 | Tuned simple baselines | "best simple baseline 0.37 vs random 0.25" | That was the pre-grid number. Now placement prior 0.4928 / 0.3838, tuned LightGBM 0.5484 / 0.4913 |
| 4 | KG | "590 nodes and 12,030 edges"; "129 mined edges" | True for v1 only. The live bot also loads KG v2: 840 nodes, 27,112 edges, 19 relations; ANSWERED_BY 287 pairs from 3,620 answer events |
| 5 | Symmetric vs asymmetric | "SYNERGY treated as symmetric" | Not enforced: 1,060 of 3,136 SYNERGY edges have a reverse edge. Symmetry only shows up in the KGE comparison (DistMult asymmetry 0.0) |
| 6 | Recall@k | "recall@10 0.36 val / 0.31 test" | 0.3614 / 0.3172 (test rounds to 0.32). Popularity is higher at exact-key level (0.4993 / 0.4021); kNN wins only at card level (0.7201 / 0.6514 vs 0.6728 / 0.6070) |
| 7 | HNSW; pre/post filter; text channel; KGE; next card; feature ranker; knowledge-editing | "comparison planned", "experiment planned", "Planned" | All run with output files (A1-A4). Text channel built but not integrated (no gain) |
| 8 | Adaptive triggers | "Rule trigger built; adaptive part planned" | Learned trigger live (LightGBM, `--policy fast`); arbiter built, no gain; coach optional |
| 9 | Separate execution | "Mac tapping is a stub" | Real Quartz executor with double guard; 86% (MAC-AF run) and 95/100 (run 233907) of taps landed |
| 10 | ReAct | "Planned (stub exists)" | Loop built and benchmarked, but the model makes 0 tool calls |
| 11 | Typed tools | "Six read-only tools for Qwen" | Built; Qwen does not use them, and the live default uses no LLM planner |
| 12 | Trajectory-level evaluation | "Planned" | 13 recorded runs with per-decision logs; still no win rate with variance (1 documented win) |
| 13 | Two-stage serving | "ranker training pending" | d1 trained and live; candidate stage now leads with engine moves (up to 2 of 4 options) |
| 14 | Dense vectors | "644-number vector" | Correct (644-d verified). It is a hand-built feature vector, not a learned bi-encoder |
| 15 | RAFT | "The 1-correct + 3-distractor format" counts as RAFT | Options are distractors, not distractor context documents; mark partial |
| 16 | Multi-agent | "one planner + one ranker" | Now deterministic engine + d1 decision LLM + optional Qwen3-14B coach; still no debating agents |

## E. Not verified or to treat carefully

- Outcomes of live runs other than run 20261107-013441 (the only result in STATUS); run 20261107-111554 has no STATUS entry, and no win rate was computed.
- Pytest not run; 1284 passed / 7 skipped is STATUS at commit `9234be4`.
- d1-2B: no test accuracy exists (eval crash); whether it was ever re-evaluated is not recorded.
- Effect of the `[plan] Deck: ...` line on d1 in live is not measured.
- Metric definitions differ: "kNN-vote recall@10" (over possible keys, `ranker_baselines_v1.json`) is not the same as "hit@10" of the retrieved list (`retrieval_ablation_v1.json`); cite them separately.
- Numbers present only outside `docs/data/`: `dataset/v1/eval/opp_model.md`, `runs/smoke/AL/eta.md`, `runs/smoke/E2/*.json`, d1 eval files for `_on_tac`, `_rebuilt`. `dataset/` and `runs/` are gitignored, so copy them before submission if they are cited.
- `PHASE3_OVERVIEW.md` (79% pro win rate; "next steps not started") and `README.md` (Phase 1 only) are stale.
- The edit eval is synthetic and the store is deterministic, so 1.000 scores say the mechanism works, not that it generalises to a real balance patch.

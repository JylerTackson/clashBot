# Course gap analysis: what the CAP 6938 segment write-ups cover that the project table does not

Written 2026-11-07. Inputs: the five course segment write-ups (Retrieval L8-10, Dense Representation L2-4, Knowledge Graph L5-7, RAG and Recommendation L11-14, Agents L15-18), `paper_table.md` (the earlier course-to-project mapping), the project in `/Users/tylerjackson/Projects/clashbot/clashBot` and its offline data. Bibliography keys refer to `docs/report/refs.bib` (304 merged entries).

## How to read this

| Item | Meaning |
|---|---|
| Class a | Already in `paper_table.md`. The project link names the row: T1.x Retrieval, T2.x RAG and Recommendation, T3.x Knowledge Graph, T4.x Agents (numbered in order of appearance in that file). Statuses are not judged here; where an artifact already exists in `docs/data/` it is named. |
| Class b | Not in the table, and doable in 1-3 days with existing data on the M5 (32 GB). The idea id (E1...E18) points to the catalogue in section 7. |
| Class c | Cite only: one line on why the technique is relevant. |
| Class d | Not applicable. |
| Effort | S = at most half a day, M = 1 to 1.5 days, L = 2 to 3 days (GPU wait included). Shown for class b only. |
| Granularity | One row per technique or per tight group of techniques that share a class. Row counts are therefore not technique counts. A (b) row can share its idea id with other rows. |
| Section numbers | Follow the compiled numbering of each write-up (Lecture, then section.subsection). The table's own section references for the Agents block (for example 'section 4.1', 'section 7.3') point to an older version of that document and do not match the current one. |

## 1. Summary

| Block | Rows | a | b | c | d |
|---|---|---|---|---|---|
| Retrieval Systems (Lectures 8-10) | 31 | 10 | 8 | 12 | 1 |
| Dense Representation (Lectures 2-4) | 19 | 1 | 4 | 7 | 7 |
| Knowledge Graph (Lectures 5-7) | 32 | 15 | 4 | 10 | 3 |
| RAG and Recommendation (Lectures 11-14) | 48 | 16 | 10 | 17 | 5 |
| Agents and Knowledge Updating (Lectures 15-18) | 52 | 19 | 6 | 20 | 7 |
| **Total** | **182** | **61** | **32** | **66** | **23** |

The 32 (b) rows collapse into 17 distinct experiments (E1-E18 except E11). E11 is listed in section 7 but only upgrades class-a rows (T1.1, T1.3, T1.9), so it is not counted as a (b) idea. Findings about the table itself are in section 8.

## 2. Retrieval Systems (Lectures 8-10)

Rows of `paper_table.md` that belong to this block: T1.1 to T1.9.

| Technique | Section in course doc | Bib key | Class | Project link / experiment idea | Effort |
|---|---|---|---|---|---|
| BM25 scoring (IDF, k1 saturation, b length norm) | L8 §1.1 | `robertson2009bm25` | a | T1.1: FTS5 BM25 over `state_text` (`clashbot/retrieval.py::StateIndex.bm25`). FTS5 fixes k1/b, so the knobs and the zero-hit failure are untested (upgrade: E11). | - |
| SPLADE learned sparse expansion, FLOPS regulariser | L8 §1.2 | `formal2021splade`, `paria2020flops` | c | Learned sparse expansion for the vocabulary-mismatch failure of BM25. Our lightweight analogue is KG/LLM query expansion (E12, row 'HyDE, Query2doc' in the RAG block). | - |
| Compression: int8/binary, PQ, OPQ, ScaNN anisotropic loss | L8 §1.3 | `jegou2011pq`, `ge2013opq`, `guo2020scann` | a | T1.9 (cite only). Upgrade E11: int8, sign-binary + rescoring and PQ (sklearn KMeans) recall@10 vs exact on the 644-d vectors (`db/state_features.npy`, 28 MB). | - |
| Codes trained to retrieve: JPQ, RepCONC, MoPQ, Distill-VQ | L8 §1.4 | `zhan2021jpq`, `zhan2022repconc`, `xiao2021mopq`, `xiao2022distillvq` | c | Quantisers trained on a ranking loss; relevant only if the index outgrew RAM (it is 28 MB). | - |
| Matryoshka representations (MRL) nested prefixes | L8 §1.5 | `kusupati2022mrl` | a | T1.9 (cite only). Upgrade: nested-prefix loss on the learned encoder inside E4 (8/16/32/64-d). | - |
| CSR sparse codes, CompresSAE | L8 §1.5 | `wen2025csr`, `kasalicky2025compressae` | c | Sparse autoencoder codes keep recsys gains at equal memory; same 'memory is not our bottleneck' argument. | - |
| HNSW graph index, recall-vs-QPS dial (efSearch); exact search below ~1M | L9 §2.1 | `malkov2020hnsw` | a | T1.4. Artifact `docs/data/retrieval_ablation_v1.md`: ef 40 gives recall@10 0.978 at 0.31 ms vs exact 1.51 ms. | - |
| IVF partitions (nprobe) | L9 §2.1 | `jegou2011pq` | c | Cell-boundary misses are the IVF failure; with 10.9k vectors exact search already costs 1.5 ms. | - |
| DiskANN / Vamana on SSD | L9 §2.1 | `subramanya2019diskann` | d | Billion-scale, SSD-resident; our corpus is 28 MB. | - |
| LSH / SimHash (collision bounds, deduplication) | L9 §2.1 | `indyk1998lsh`, `charikar2002simhash` | b | E10 near-duplicate audit: SimHash buckets + exact cosine between val/test queries and the train pool (RETRO-style leakage check) and false-negative audit of the 3 hard negatives. Pre-check run for this report: 0 of 591 val / 547 test play states have a train neighbour with cosine >= 0.95 (mean best cosine 0.65), so expect a clean bill. | S |
| Filtering: pre- vs post-filter pitfalls | L9 §2.2 | `johnson2019faiss` | a | T1.5. Artifact: post-filtering at K=10 returns nothing for 50% (deck) and 87% (phase) of queries. | - |
| Incremental updates (FreshDiskANN) | L9 §2.2 | `singh2021freshdiskann` | c | Maps to appending own live games to the index (hnswlib `add_items` / `mark_deleted`); no experiment needed at this size. | - |
| Bi-encoder single-vector scoring and its capacity limit | L9 §2.3 | `khattab2020colbert` | a | T1.2: the 644-d structured state vector is a pooled bi-encoder (`clashbot/features.py::featurize`). | - |
| ColBERT late interaction (MaxSim) | L9 §2.3 | `khattab2020colbert` | b | E5: a state is a SET of units (88% of states have >= 1 unit). Token vectors per unit (card role, side, lane, zone, ETA); Chamfer/MaxSim rerank of the struct-kNN top-100; hit@10 and card_hit@10 vs pooled vector via `tools/eval/eval_retrieval.py`. | M |
| ColBERTv2 residual compression + distillation, PLAID pruning | L9 §2.3 | `santhanam2022colbertv2`, `santhanam2022plaid` | c | E5's 'cheap first stage, MaxSim only on survivors' is the PLAID pattern; compression is irrelevant at this scale. | - |
| MUVERA fixed-dimensional encodings | L9 §2.4 | `dhulipala2024muvera` | b | E5 stretch: SimHash-partition FDE of the unit tokens, single-vector MIPS, compare recall@100 and time against exact Chamfer. | M |
| ColPali (VLM patch embeddings of page images, no OCR) | L9 §2.4 | `faysse2025colpali` | c | Perception-free retrieval from screenshots. Project keeps the YOLO/OCR pipeline; would need frame-level labels to benchmark. | - |
| Cross-encoders: monoBERT, monoT5 | L10 §3.1 | `nogueira2019monobert`, `nogueira2020monot5` | c | d1 reads state + option jointly (cross-encoder style) but ranks 4 options, not 100 (T1.7). | - |
| Distilling a cross-encoder: TAS-B, RocketQAv2 | L10 §3.1 | `hofstatter2021tasb`, `ren2021rocketqav2` | c | Fast scorer = the LightGBM models (the act model runs at 0.05 ms), slow = d1 (about 65 ms). Soft labels from d1 on train would be overfit (kg run: train loss 0.43, val 1.67), so not run. | - |
| Setwise LLM ranking; FIRST single-pass logit read-out | L10 §3.2 | `zhuang2024setwise`, `reddy2024first` | a | T1.7: d1 reads option logits in one forward pass (FIRST-like), 4-way. Test acc 0.497 (joint_tac) vs placement prior 0.384. | - |
| Pairwise ranking (PRP) and input-order sensitivity | L10 §3.2 | `qin2024prp`, `sun2023rankgpt` | b | E3: fine-tuned d1 under 4 rotations + reversal (position bias, pass^k-style consistency, permutation vote) and all-pairs PRP (6 two-option calls) vs the 4-option setwise call; latency x calls. | M |
| Listwise generation and its distillation: RankGPT, RankVicuna, RankZephyr | L10 §3.2 | `sun2023rankgpt`, `pradeep2023rankvicuna`, `pradeep2023rankzephyr` | c | Generated permutations need >= 7B models and can be malformed; d1 scores options directly so malformed output cannot occur. | - |
| Reasoning rerankers: Rank1, Rank-R1, ReasonRank | L10 §3.3 | `weller2025rank1`, `zhuang2025rankr1`, `liu2025reasonrank` | b | E3 extension: zero-shot Qwen3-14B-4bit (`clashbot/planner.py::MLXChat`, thinking on/off) as a setwise reasoner on a 200-line test subset vs fine-tuned 600M d1. | M |
| Generative retrieval: DSI, NCI, SEAL | L10 §3.4 | `tay2022dsi`, `wang2022nci`, `bevilacqua2022seal` | b | E17: the action key is already a 3-part identifier. Fine-tune the factorized d1 variant (card, lane, zone; `dataset/v1/d1/factorized`) vs joint; count invalid compositions (card not in hand). | L |
| Learned identifiers: TIGER residual quantisation | L10 §3.5 | `rajput2023tiger` | b | E17 (same experiment; multi-codebook analogy: card x lane x zone). | L |
| RIPOR, scaling collapse of generative retrieval | L10 §3.5 | `zeng2024ripor`, `pradeep2023grscale` | c | Scale issues at 8.8M passages; irrelevant at 630 action keys. | - |
| BEIR: BM25 strong baseline, dense models lose out-of-domain | L10 §3.6 | `thakur2021beir` | a | T1.6 (recall@k, BM25 baseline). Channels are within noise of each other on our data. | - |
| BEIR-style zero-shot / domain-shift test | L10 §3.6 | `thakur2021beir` | b | E13: run the 3 channels on live-run queries (`runs/poc/*/decisions.jsonl` `context_text`) vs pro val queries; compare top-1 similarity, empty-pool rate, BM25-vs-kNN overlap. No gold labels, so coverage metrics only. | S |
| MTEB (embedder selection) | L10 §3.6 | `muennighoff2023mteb` | a | T1.8: bge-small-en-v1.5 text channel (D-033, `clashbot/text_embed.py`). | - |
| Reciprocal Rank Fusion | L10 §3.6 | `cormack2009rrf` | a | T1.3 (`StateIndex.rrf`, k=60). Upgrade E11: k sweep and weighted fusion. | - |
| BRIGHT: rerankers can hurt on reasoning-heavy queries | L10 §3.6 | `su2025bright` | c | Parallels 'untrained notes hurt d1 (-4.4 / -1.3 pts)' and the arbiter's null result (`docs/data/arbiter_v1.md`). | - |

## 3. Dense Representation (Lectures 2-4)

Rows of `paper_table.md` that belong to this block: none (`paper_table.md` has no Dense block).

| Technique | Section in course doc | Bib key | Class | Project link / experiment idea | Effort |
|---|---|---|---|---|---|
| ColBERT query/document encoders and MaxSim | L02 §1.1 | `khattab2020colbert` | b | E5 (see Retrieval L9 §2.3). | M |
| InfoNCE / contrastive predictive coding | L02 §1.2 | `oord2018cpc` | b | E4: train a state encoder + action tower with in-batch InfoNCE on (state, pro action) pairs from 6,658 train-eligible states; evaluate as (A) embedding for kNN (R1 metrics) and (B) direct action retrieval (candidate recall@k). | L |
| CLIP symmetric dual-encoder loss | L02 §1.3 | `radford2021clip` | b | E4 ablation: symmetric (state-to-action + action-to-state) vs one-directional InfoNCE. | L |
| SigLIP sigmoid loss (no batch-wide denominator) | L02 §1.4 | `zhai2023siglip`, `tschannen2025siglip2` | b | E4 ablation: sigmoid pairwise loss vs softmax. Popular action keys repeat inside a batch, so in-batch softmax has false negatives; the sigmoid loss and logQ are the two fixes. | L |
| SigLIP 2 add-ons: LocCa captioning, SILC self-distillation, TIPS masked prediction | L02 §1.4 | `tschannen2025siglip2` | d | Image-specific auxiliary losses. | - |
| Contrastive loss with mined hard negatives; false negatives | L02 §1.5 | `xiong2021ance` | a | T2.2: every ranker example has 3 hard negatives (`tools/data/build_ranker_dataset.py`). False-negative risk audited in E10; used as training negatives in E4. | - |
| VILA-U unified tokenizer (equal-weight loss sum) | L02 §1.6 | `wu2025vilau` | d | Image tokenizer; no counterpart. | - |
| UniTok multi-codebook quantisation (codes multiply) | L02 §1.7 | `ma2025unitok` | c | card x lane x zone is a three-codebook identifier; E17 tests joint vs factorized decoding of it. | - |
| VQ-VAE, BSQ, MAGVIT-v2, straight-through estimator | L02 §1.7-1.8 (bib) | `oord2017vqvae`, `zhao2025bsq`, `yu2024magvit2`, `bengio2013ste` | c | Quantiser background for TIGER's RQ-VAE (E17); cite only. | - |
| QLIP two-stage training, loss weights 1000x apart | L02 §1.8 | `zhao2025qlip` | d | Visual tokenizer; the weight-balancing lesson has no counterpart. | - |
| GME, VLM2Vec, UniME (MLLM-based universal embedders) | L03 §2.1, 2.3, 2.4 | `zhang2024gme`, `jiang2024vlm2vec`, `unime2025` | c | A VLM embedder of the screenshot would be a perception-free state representation; not tried (structured vector used). | - |
| Qwen3-VL-Embedding (instruction-aware, MRL) | L03 §2.2 | `li2026qwen3vl` | c | MRL support motivates the prefix sweep in E4; would be the encoder for any frame-embedding channel. | - |
| MetaEmbed (learned meta tokens, multi-vector Matryoshka) | L03 §2.5 | `xiao2026metaembed` | c | Learned few-vector summaries; E5 stretch would learn token vectors instead of fixing them. | - |
| DINO, iBOT, DINOv2, DINOv3 (self-distillation vision backbones) | L03 §2.6-2.9 | `caron2021dino`, `zhou2022ibot`, `oquab2024dinov2`, `simeoni2025dinov3` | d | Perception uses KataCR YOLOv8 + template/OCR readers; no label-free backbone needed. | - |
| Think-Then-Embed (reasoner trace, then embed) | L04 §3.1 | `cui2025tte` | c | The `tactics:` lines are an engine-written trace prepended to d1's input (TTE-t analogue). Untrained use hurt (-4.4 / -1.3), trained use helped (+1.6 / +1.4), matching TTE's 'must train with the trace'. | - |
| UME-R1 (dual embeddings, GRPO) | L04 §3.2 | `umer1` | c | RL-trained embeddings; no RL stage planned. | - |
| PLUME latent reasoning | L04 §3.3 | `plume` | d | Latent reasoning for multimodal embedders. | - |
| MoEmb mixture-of-experts embedding | L04 §3.4 | `lim2026moemb` | d | Unpublished lab work; scale far beyond this project. | - |
| Bib-only: CoCa, BLIP-2, GRIT, OpenVision 2, Janus, unified-model survey, DREAM, Embed-RL | L02-L04 (bib only) | `yu2022coca`, `li2023blip2`, `muennighoff2024grit`, `openvision2`, `wu2024janus`, `unified2025survey`, `li2026dream`, `embedrl` | d | Multimodal pretraining/generation background with no section in the write-up. | - |

## 4. Knowledge Graph (Lectures 5-7)

Rows of `paper_table.md` that belong to this block: T3.1 to T3.10.

| Technique | Section in course doc | Bib key | Class | Project link / experiment idea | Effort |
|---|---|---|---|---|---|
| Knowledge graph as typed triples; symmetric vs asymmetric relations | L5 §1.1-1.2 | `cap6938L05` | a | T3.1, T3.2: 590 nodes / 12,030 edges (`tools/data/build_kg.py`, `build_kg_v2.py`); SYNERGY symmetric, COUNTERS directed. | - |
| TransE (translation) | L5 §1.1 | `bordes2013transe` | a | T3.5 (`tools/eval/kge_link_prediction.py`): MRR 0.082. | - |
| DistMult, ComplEx (multiplicative scoring) | L5 §1.2 | `yang2015distmult`, `trouillon2016complex` | a | T3.5/T3.2: MRR 0.091 / 0.116; DistMult asymmetry score 0 by construction. | - |
| RotatE (rotation) incl. self-adversarial negative sampling | L5 §1.2 | `sun2019rotate` | a | T3.5: best model, MRR 0.159 (full) / 0.160 (no_inverse). The self-adversarial loss factor itself is not tested (E15). | - |
| ConvE and inverse-relation leakage (WN18RR idea) | L5 §1.2 | `dettmers2018conve` | b | E15: add ConvE and a loss/negative-sampling factor (PyKEEN `NSSALoss`) to `kge_link_prediction.py`; tests 'the gap is training, not architecture' (TransE 0.332 vs RotatE 0.338 in the lecture). The `no_inverse` setting already reproduces ConvE's leakage point. | S |
| Equal-tuning audit of KGE comparisons | L5 §1.3 | `ruffinelli2020olddog` | a | T3.6: equal grid per model, 3 seeds. | - |
| PyKEEN large-scale reproduction | L5 §1.3 | `ali2021pykeen` | a | T3.6: PyKEEN 1.11 is the library used. | - |
| Random tie-breaking in the ranking protocol | L5 §1.3 | `sun2020reevaluation` | a | T3.6: realistic rank / expected Hits@k under random tie-breaking in `kge_link_prediction.py`. | - |
| Temporal interpolation: HyTE, TComplEx, DE-SimplE (+ TTransE, TA-DistMult) | L6 §2.1 | `dasgupta2018hyte`, `lacroix2020tcomplex`, `goel2020desimple`, `leblay2018validitytime`, `garciaduran2018tadistmult` | c | Timestamp completion; our temporal task is forecasting (extrapolation). | - |
| Temporal extrapolation: RE-NET, RE-GCN (+ TiTer in bib) | L6 §2.2 | `jin2020renet`, `li2021regcn`, `sun2021titer` | c | SASRec (`tools/eval/next_card.py`) plays the autoregressive-history role (T2.12). | - |
| xERTE query-conditioned subgraph expansion | L6 §2.3 | `han2021xerte` | c | Explainable attention subgraph; no equivalent. | - |
| CyGNet copy mechanism (repeat past triples) | L6 §2.3 | `zhu2021cygnet` | a | T3.7: copy_frequency acc@1 0.207 val / 0.243 test vs SASRec 0.129 / 0.222 (`docs/data/next_card_v1.md`). | - |
| TLogic temporal rules with timestamp constraints | L6 §2.3 | `liu2022tlogic` | a | T3.3 (`tools/data/mine_pro_answers.py`, 8 s answer window; 3,620 answer events). | - |
| Scene graphs from pixels: SGTR, OvSGTR | L6 §2.4 | `li2022sgtr`, `chen2024ovsgtr` | a | T3.9 (cite): perception as scene graph. | - |
| Video scene graphs: STTran (spatial then temporal attention) | L6 §2.5 | `cong2021sttran` | b | E16: render the perceived field as relation triples (unit-in-lane, unit-attacks-tower, unit-near-unit) in `state_text`; compare BM25 hit@10 and card_hit@10 against the flat unit list. | S |
| One-stage video scene graphs: OED, UNO | L6 §2.5 | `wang2024oed`, `le2026uno` | c | Detector-free scene-graph models; no counterpart. | - |
| SceneSayer anticipation of future graphs | L6 §2.6 | `peddi2024scenesayer` | c | Tracker `predicted_tile_2s` and `tools/eval/eval_eta.py` (median ETA error 1.49 to 0.30 s) are our anticipation metric. | - |
| ESCA: perception errors dominate agent failures | L6 §2.6 | `huang2025esca` | a | T3.9 (cite). | - |
| Knowledge compilation map, SDDs, weighted model counting | L7 §3.1 | `darwiche2002map`, `darwiche2011sdd`, `chavira2008wmc` | d | No probabilistic logic program or large Boolean KB to compile. | - |
| Differentiable fuzzy logic (t-norms) | L7 §3.2 | `vankrieken2022fuzzy` | c | `clashbot/tactics.py` combines need flags and scores with hand-set products/min; t-norm choice is the lecture's point, not tuned. | - |
| AMIE 3 Horn-rule mining | L7 §3.3 | `lajus2020amie3` | a | T3.3: 129 ANSWERED_BY edges (v1) and 3,620 answer events (K2), train split only. | - |
| Rule scoring: support, head coverage, standard vs PCA confidence, window sweep | L7 §3.3 | `lajus2020amie3`, `liu2022tlogic` | b | E6: held-out evaluation of the mined answers (`KG2.answers_for`) on val/test events, with source ablation, support thresholds and the answer-window sweep. | M |
| Differentiable rule learning: Neural LP, DRUM, RNNLogic, NCRL | L7 §3.4 | `yang2017neurallp`, `sadeghian2019drum`, `qu2021rnnlogic`, `cheng2023ncrl` | c | A discrete miner suffices at 126 cards; NCRL's length generalisation matters only if counter-of-counter chains were mined. | - |
| Neural theorem proving, soft unification | L7 §3.5 | `rocktaschel2017ntp`, `cui2025softunification` | d | Proof trees over a large KB; not applicable. | - |
| Hard symbolic constraints as a filter (legal plays) | L7 §3.6 | `xu2018semanticloss` | a | T3.4: `clashbot/game_rules.py` filters candidates (card in hand, elixir, placement). | - |
| Semantic loss (differentiable constraint penalty) | L7 §3.6 | `xu2018semanticloss` | a | T3.8 (planned). Cheap variant: illegal-option penalty on the logreg ranker in `tools/eval/feature_ranker.py`. | - |
| DeepProbLog, Scallop (neural predicates + logic programs) | L7 §3.6 | `manhaeve2018deepproblog`, `huang2021scallop`, `li2023scallop` | c | Closest analogue: perception confidences feeding the rule engine; not tried. | - |
| ChatRule (LLM verbalises mined rules) | L7 §3.7 | `luo2025chatrule` | a | T3.10 (stretch). | - |
| GRiD rule generation by denoising | L7 §3.7 | `cheng2026grid` | c | Equal-footing rule-learning benchmark; cite. | - |
| LeanDojo, miniF2F, Seed-Prover | L7 §3.8 | `yang2023leandojo`, `zheng2022minif2f`, `bytedance2025seedprover` | d | Formal theorem proving. | - |
| Inductive link prediction (new entities): GraIL, NBFNet, ULTRA | L6 (bib only) | `teru2020grail`, `zhu2021nbfnet`, `galkin2024ultra` | b | E14: new cards appear each patch. Hold out all COUNTERS edges of ~10 cards; compare transductive KGE (no embedding for an unseen card) with role-similarity transfer (`clashbot/card_roles.py`, `KG2` SIMILAR_ROLE) and a feature logreg; reuse the filtered evaluator in `kge_link_prediction.py`. | M |
| GNN-based KG reasoning: R-GCN, CompGCN, RED-GNN, A*Net, AdaProp, MOTIF | L6 (bib only) | `schlichtkrull2018rgcn`, `vashishth2020compgcn`, `zhang2022redgnn`, `zhu2023astarnet`, `zhang2023adaprop`, `huang2025motif` | c | In the bib but no write-up section; cite as the learned-reasoner family behind E14. | - |

## 5. RAG and Recommendation (Lectures 11-14)

Rows of `paper_table.md` that belong to this block: T2.1 to T2.12.

| Technique | Section in course doc | Bib key | Class | Project link / experiment idea | Effort |
|---|---|---|---|---|---|
| Parametric memory bottleneck (PopQA: accuracy tracks popularity) | L11 §2.1 | `mallen2023popqa` | a | T2.8 (cite). Popularity-stratified accuracy of d1/LightGBM is part of E1. | - |
| REALM (joint retriever-reader pretraining) | L11 §2.2 | `guu2020realm` | c | Our retriever is fixed kNN/BM25; only E4 makes it learned. | - |
| RAG (retrieve then generate, marginalisation) | L11 §2.2 | `lewis2020rag` | a | T2.8. | - |
| Fusion-in-Decoder | L11 §2.3 | `izacard2021fid` | c | Many passages fused linearly; d1's 160-char per-option note budget is the analogue. | - |
| RETRO and test-set contamination | L11 §2.3 | `borgeaud2022retro` | a | T2.6: split by video. E10 adds a near-duplicate check (pre-check clean). | - |
| Atlas | L11 §2.4 | `izacard2023atlas` | c | Joint retriever-reader training at few-shot scale; cite. | - |
| RA-DIT (tune the LM to use retrieval, tune the retriever to the LM) | L11 §2.4 | `lin2024radit` | c | d1-ft-joint-tac is the LM-side step; the retriever-side step is E4. | - |
| RAFT (gold + distractor documents) | L11 §2.4 | `zhang2024raft` | a | T2.3. | - |
| Long context vs retrieval (LongICLBench, LongRAG inverted U) | L11 §2.5 | `xu2024longcontext`, `jin2024longrag` | a | T2.11 (planned): how many lookup lines to feed the coach/d1. | - |
| HyDE (hypothetical document), Query2doc | L11 §2.6 | `gao2023hyde`, `wang2023query2doc` | b | E12: expand the BM25 query with KG-derived tokens (counters/answers of threat cards, archetype tags; zero latency) and with a Qwen3-4B 'what would a pro play' pseudo-document on a 300-query subset; hit@10/card_hit@10/ndcg@10 via `eval_retrieval.py`. | M |
| Rewrite-Retrieve-Read with RL rewriter, PPO | L11 §2.6 | `ma2023rrr`, `schulman2017ppo` | c | RL-trained rewriter; out of scope. | - |
| KILT: credit only with correct answer AND correct provenance | L11 §2.7 | `petroni2021kilt` | b | E1: 'supported accuracy' = d1 picks the gold option AND that option carries non-filler support (pro lookup / tactics / KG note) vs raw accuracy. | S |
| RAGAS reference-free faithfulness | L11 §2.7 | `es2024ragas` | d | Scores free-text answers; ours are 4-way choices. | - |
| RAGChecker noise sensitivity (relevant-but-misleading vs irrelevant chunks) | L11 §2.7 | `ru2024ragchecker` | b | E2: poisoned (misleading but plausible) notes vs random distractor notes; follow-rate and accuracy drop per condition. | M |
| CRAG benchmark scoring (+1 correct, 0 abstain, -1 wrong) | L11 §2.7 | `yang2024cragbench` | b | E1: risk-coverage curve of d1 with 'hold' as abstention; utility under +1/0/-1 and the best confidence threshold (per_example probs already saved). | S |
| IRCoT interleaved retrieval | L12 §3.1 | `trivedi2023ircot` | c | Fixed-schedule retrieval; our trigger is play/wait. | - |
| FLARE confidence-triggered retrieval | L12 §3.1 | `jiang2023flare` | a | T2.7 (rule trigger; LightGBM fast act model is the learned trigger: balanced acc 0.607 vs rules 0.533). | - |
| Search-o1 | L12 §3.1 | `li2025searcho1` | c | Reasoning model issuing its own searches; slow loop makes 0 tool calls. | - |
| Self-RAG reflection tokens; CRAG corrective evaluator | L12 §3.2 | `asai2024selfrag`, `yan2024crag` | c | The arbiter and d1 confidence play the critique role (`docs/data/arbiter_v1.md`: no gain). | - |
| Adaptive-RAG router | L12 §3.2 | `jeong2024adaptiverag` | a | T2.7: `clashbot/arbiter.py` narrows d1's options only when d1 is unsure. | - |
| GRPO, Dr. GRPO, DAPO | L12 §3.3 | `shao2024deepseekmath`, `liu2025drgrpo`, `yu2025dapo` | c | RL from outcome reward; states carry `verdict` / `elixir_trade` but there is no simulator to roll out. | - |
| RL search agents: Search-R1, R1-Searcher, DeepResearcher, ZeroSearch | L12 §3.4 | `jin2025searchr1`, `song2025r1searcher`, `zheng2025deepresearcher`, `sun2025zerosearch` | d | Web-QA search agents; our tools are 6 local read-only functions. | - |
| GraphRAG community summaries; Han et al. RAG vs GraphRAG | L12 §3.5 | `edge2024graphrag`, `han2025ragvsgraphrag` | a | T2.9: 'start with dense or hybrid RAG', which is what the system does. | - |
| HippoRAG / HippoRAG 2 (personalised PageRank) | L12 §3.5 | `gutierrez2024hipporag`, `gutierrez2025hipporag2` | a | T2.9 (planned): `networkx.pagerank` personalised on current threat nodes over `dataset/v1/kg/edges.jsonl` (networkx 3.7 installed). | - |
| LazyGraphRAG, RAPTOR, Graph-R1 | L12 §3.5 | `edge2024lazygraphrag`, `sarthi2024raptor`, `luo2025graphr1` | c | Index-cost and long-document trees; no LLM-built index here. | - |
| Matrix factorisation with biases, implicit and time terms | L13 §4.1 | `koren2009mf` | c | Netflix-era MF; the 'user x item' matrix here is deck x card (E7). | - |
| iALS implicit-feedback ALS | L13 §4.1 | `hu2008ials` | b | E7: deck completion (opponent deck given k seen cards) with iALS (numpy), EASE (closed form) and LightGCN vs `OppDeckModel`, using the leave-one-game-out protocol of `tools/eval/eval_opp_model.py`. | M |
| Tuned iALS still beats neural models; tuned baselines win | L13 §4.1 | `rendle2022ials`, `rendle2020ncfvsmf`, `klenitskiy2023dross` | a | T2.5: popularity / placement prior / kNN next to d1 (LightGBM 0.491, d1 0.497 vs prior 0.384 test). | - |
| NCF / NeuMF | L13 §4.2 | `he2017ncf` | c | Replaced by the tuned dot-product lesson (rendle2020ncfvsmf). | - |
| LightGCN (graph CF without weights or nonlinearities) | L13 §4.2 | `he2020lightgcn` | b | E7: LightGCN on the deck-card bipartite graph (IN_DECK edges, 68 KB decks + 197 game decks). | M |
| BPR pairwise loss | L13 §4.3 | `rendle2009bpr` | a | T2.2 (pairwise losses); loss swap at fixed architecture in E18. | - |
| Sampled softmax with in-batch negatives | L13 §4.3 | `wu2024ssm` | a | T2.2: d1 trains with softmax cross-entropy over the 4 options (`tools/train/train_d1.py::batch_loss`). | - |
| Two-stage serving: candidate retrieval then ranking | L13 §4.4-4.5 | `yi2019twotower`, `naumov2019dlrm` | a | T2.1 (`clashbot/candidates.py` then d1). Funnel metric = P(gold in candidates) x P(pick \| in candidates). | - |
| Two-tower training with in-batch negatives | L13 §4.4 | `yi2019twotower` | b | E4 (learned state tower + factorised action tower; softmax vs BPR vs BCE vs sigmoid). | L |
| logQ popularity correction (and its revision) | L13 §4.4 | `khrylchenko2025logq` | b | E4: popular keys ('hold', 'goblin-barrel\|...') are over-penalised as in-batch negatives; compare plain, logQ and the revised weighting by Recall@10. | L |
| Mixture of Logits (breaking the rank cap) | L13 §4.4 | `zhai2023mol`, `ding2025mol` | c | Dot-product rank cap d+1; irrelevant for 64-d towers vs 630 targets. | - |
| FM / DLRM / DCN-V2 feature-interaction rankers | L13 §4.5 | `rendle2010fm`, `naumov2019dlrm`, `wang2021dcnv2` | a | T2.10: `tools/eval/feature_ranker.py` (DCN-V2-style 0.412 test vs LightGBM 0.491). | - |
| Wukong, RankMixer (scaling ranking models) | L13 §4.6 | `zhang2024wukong`, `zhu2025rankmixer` | d | Industrial scaling laws; 6.6k examples here. | - |
| SASRec (causal self-attention over history) | L14 §5.1 | `kang2018sasrec` | a | T2.12: `tools/eval/next_card.py`; opponent next card, loses to the copy baseline. | - |
| BERT4Rec (bidirectional Cloze) | L14 §5.1 | `sun2019bert4rec` | c | Next-card prediction is causal; cite. | - |
| Loss, not architecture: SASRec+ (full cross-entropy) | L14 §5.2 | `klenitskiy2023dross` | b | E18: swap pointwise BCE / pairwise BPR / listwise softmax / LambdaRank at fixed features on the 4-way ranker (`feature_ranker.py`). `next_card.py` already uses full cross-entropy. | S |
| gSASRec (calibration under negative sampling) | L14 §5.2 | `petrov2023gsasrec` | c | ECE of d1/LightGBM is measured in E1. | - |
| HSTU, OneRec generative recommenders | L14 §5.3 | `zhai2024hstu`, `deng2025onerec`, `onerec2025report` | d | Billion-parameter industrial recommenders. | - |
| Semantic IDs: TIGER | L14 §5.4 | `rajput2023tiger` | b | E17 (see Retrieval L10 §3.5). | L |
| GRID (swap encoder/quantiser/generator one at a time) | L14 §5.4 | `ju2025grid` | c | E17 is a one-factor swap of the same kind. | - |
| P5 and TALLRec (LM as recommender, LoRA on few samples) | L14 §5.5 | `geng2022p5`, `bao2023tallrec` | a | T2.4: d1 fine-tuned on pro decisions. | - |
| R2ec (reasoning head + rec head, GRPO) | L14 §5.5 | `you2025r2ec` | c | Relates to the coach (reasoning) vs d1 (picking) split, D-045. | - |
| DiffRec diffusion recommender | L14 §5.6 | `wang2023diffrec` | d | No counterpart. | - |

## 6. Agents and Knowledge Updating (Lectures 15-18)

Rows of `paper_table.md` that belong to this block: T4.1 to T4.12.

| Technique | Section in course doc | Bib key | Class | Project link / experiment idea | Effort |
|---|---|---|---|---|---|
| Toolformer self-supervised API calls; ToolLLM DFS tool search | L15 §2.1 | `schick2023toolformer`, `qin2024toolllm` | c | Tool-use training at scale; our 6 tools are prompted, not trained. | - |
| MCP / typed JSON-schema tools | L15 §2.1 | `mcp2026spec` | a | T4.1: `clashbot/tools.py` (6 tools, `_validate`). | - |
| BFCL-style AST-match evaluation (simple, multiple, irrelevance, missing parameter, multi-turn) | L15 §2.1 | `patil2025bfcl` | b | E9: ~140 synthetic prompts over the 6 tools run through `LLMPlanner` with Qwen3-4B and Qwen3-14B; report exact-match, irrelevance abstention, missing-argument handling. Explains the observed '0 tool calls'. | M |
| ReAct | L15 §2.2 | `yao2023react` | a | T4.3 (planner loop; stub). | - |
| Reflexion | L15 §2.2 | `shinn2023reflexion` | a | T4.4 (post-game lessons into `MemoryStore` procedural memory). | - |
| CodeAct (Python as action) | L15 §2.2 | `wang2024codeact` | d | Actions are a 4-way choice run by the executor. | - |
| Tree of Thoughts | L15 §2.3 | `yao2023tot` | c | LM proposer/evaluator BFS at 5-100x tokens; the lane simulation (`clashbot/tactics.py`) is the cheap simulator-based evaluator. | - |
| LATS (MCTS over ReAct) | L15 §2.3 | `zhou2024lats` | c | Search per tick is unaffordable (decision 65 ms vs seconds for a generative LLM, D-046). | - |
| SWE-bench | L15 §2.4 | `jimenez2024swebench` | d | Software-engineering benchmark. | - |
| SWE-agent: the interface matters | L15 §2.4 | `yang2024sweagent` | c | Candidate list + per-option notes is d1's interface: trained +1.6/+1.4, untrained -4.4/-1.3 pts. | - |
| tau-bench pass^k reliability | L15 §2.4 | `yao2025taubench` | a | T4.11 (live trials with variance). Permutation pass^k is part of E3. | - |
| Lost in the Middle; Context Rot | L15 §2.5 | `liu2024lostmiddle`, `hong2025contextrot` | a | T4.5: keep the coach/d1 context short. | - |
| Length alone hurts; H2O KV eviction | L15 §2.5 | `du2025contextlength`, `zhang2023h2o` | c | Coach prefix KV-cache reuse (-34% prefill) is the cheap relative. | - |
| Observation masking, ReSum, SUPO, CompactionRL | L15 §2.6 | `lindenbauer2025complexitytrap`, `wu2025resum`, `lu2025supo`, `li2026compactionrl` | c | Coach prompts are short plans, no growing history. | - |
| MEM1, MemAgent, FoldAgent, Recursive LMs | L15 §2.7 | `zhou2026mem1`, `yu2026memagent`, `sun2026foldagent`, `zhang2025rlm` | c | Long-input memory tricks; not needed at 3-minute games. | - |
| ACE evolving playbook | L15 §2.7 | `zhang2026ace` | a | T4.5: per-deck playbook. | - |
| Memory types: episodic / semantic / procedural | L16 §3.1 | `lim2026l16` | a | T4.6: `clashbot/memory.py` (episodes, patch-versioned card_facts, lessons). | - |
| MemGPT paging | L16 §3.1 | `packer2023memgpt` | a | T4.5. | - |
| Mem0 ADD / UPDATE / DELETE / NOOP | L16 §3.1 | `chhikara2025mem0` | a | T4.7: `MemoryStore.write_lesson` ops. | - |
| ReasoningBank strategies from successes and failures | L16 §3.1 | `ouyang2026reasoningbank` | a | T4.6 (procedural memory). | - |
| Generative Agents (recency x importance x relevance), A-MEM | L16 §3.1 | `park2023genagents`, `xu2025amem` | c | Retrieval scoring for episodic memory; no episodic retrieval task to evaluate yet. | - |
| LoCoMo, LongMemEval | L16 §3.2 | `maharana2024locomo`, `wu2025longmemeval` | c | Memory benchmarks; knowledge-update + abstention abilities mirror edit_eval versioning/rollback. | - |
| METR time horizons; Lost in Conversation | L16 §3.3 | `kwa2025metr`, `laban2026lostconv` | d | Human-time horizons and sharded instructions; games are 3 minutes. | - |
| Vending-Bench long-term coherence | L16 §3.3 | `backlund2025vendingbench` | c | Meltdowns not tied to memory fill: compare our 'panic plays' (FIX_LIST #44). | - |
| OSWorld: final-state scoring, misclicks dominate failures | L16 §3.3 | `xie2024osworld` | a | T4.11: taps landed 95/100 in run 233907; per-run failure taxonomy from `runs/poc/*/decisions.jsonl`. | - |
| Agent Lightning, SWE-RL, Self-Play SWE-RL, Tongyi DeepResearch | L16 §3.4 | `luo2025agentlightning`, `wei2025swerl`, `wei2026ssr`, `tongyi2025deepresearch` | d | RL infrastructure for software/search agents. | - |
| MetaGPT roles; Kimi K2.5 agent swarm | L16 §3.5 | `hong2024metagpt`, `kimi2026k25` | d | Role-play and RL swarms; the design is one planner + one ranker. | - |
| Multiagent debate (and 'a simple vote matches debate') | L16 §3.5 | `du2024debate` | b | E1: vote / probability-average ensemble of d1 variants + LightGBM + placement prior vs best single model, paired bootstrap on the same 766 test lines. | S |
| MAST multi-agent failure taxonomy | L16 §3.5 | `cemri2025mast` | a | T4.12 (cite). | - |
| Editing criteria: reliability, generalisation, locality, portability | L17 §4.1 | `yao2023editingsurvey` | a | T4.8: `tools/eval/edit_eval.py` (reliability, generalisation, portability 1.000 vs 0.000 / 0.000 / 0.067 without the edit; locality 1.000 both). | - |
| FFN key-value memories | L17 §4.1 | `geva2021ffnkv` | c | Mechanism behind locate-then-edit; cite. | - |
| ROME / MEMIT locate-then-edit | L17 §4.2 | `meng2022rome`, `meng2023memit` | a | T4.9: weight editing only as a comparison. | - |
| MEND, WISE (learned or side-memory weight editors) | L17 §4.3 | `mitchell2022mend`, `wang2024wise` | c | d1 uses a non-GPT architecture; weight editors not portable. | - |
| SERAC (external counterfactual memory) | L17 §4.3 | `mitchell2022serac` | a | T4.9: patch-versioned external fact store (`MemoryStore.apply_edit`, `retract`). | - |
| IKE in-context editing | L17 §4.3 | `zheng2023ike` | b | E2 part (ii): state an edit as a context line ('goblin-barrel now costs 4') and measure whether d1 obeys it; shows why legality/elixir stays external. | M |
| Edits at scale: collapse, AlphaEdit, UltraEdit | L17 §4.4 | `gupta2024editscale`, `yang2024butterfly`, `fang2025alphaedit`, `gu2026ultraedit` | c | Sequential weight-edit collapse cannot occur in an external store. | - |
| MQuAKE multi-hop consequences of an edit | L17 §4.5 | `zhong2023mquake` | a | T4.8 portability: edit_eval 1.000 edited vs 0.067 no edit. | - |
| Mirage of model editing (teacher-forced scoring leaks) | L17 §4.5 | `yang2025mirage` | c | Caveat: store-lookup reliability is 1.0 by construction; portability and E2 are the informative metrics. | - |
| Conflict taxonomy (inter-context / context-memory / intra-memory); entity-swap conflicts | L17 §4.6 | `xu2024conflictsurvey`, `longpre2021entityconflict` | a | T4.10 (taxonomy of stale vs misinformation). | - |
| Pre-patch evidence down-weighting (ClashEval-style retrieval weighting) | L17 §4.6 | `wu2024clasheval` | a | T4.10: `memory.py::reweight_hits` cuts edited-card hits 0.124 to 0.002. | - |
| Context-vs-prior conflict measurement (ClashEval protocol; chameleon vs sloth) | L17 §4.6 | `wu2024clasheval`, `xie2024chameleon` | b | E2: corrupt / shuffle / swap the `tactics:`, `KG:` and per-option note text in `dataset/v1/d1/joint_tac` and measure d1's follow-rate and accuracy drop. | M |
| CAD and AdaCAD (contrast logits with vs without context) | L17 §4.7 | `shi2024cad`, `wang2025adacad` | b | E2: d1-ft-joint-tac option log-probs with notes vs notes stripped; combine (1+a) log p_ctx - a log p_noctx; Jensen-Shannon-adaptive a; tune on val, report test. | M |
| JuICE attention-head steering | L17 §4.7 | `li2025juice` | c | Needs white-box head interventions on d1; skipped. | - |
| Continual-learning scenarios, EWC, tiny replay, PackNet | L18 §5.1 | `vandeven2019threescenarios`, `kirkpatrick2017ewc`, `chaudhry2019tiny`, `mallya2018packnet` | c | d1 is fine-tuned in sequence (joint, tac, kg) but never class-incremental. | - |
| GEM: average accuracy and backward transfer | L18 §5.1 | `lopezpaz2017gem` | b | E8: ACC/BWT matrix of d1-ft-joint, -tac, -kg checkpoints on joint / joint_tac / joint_tac.prev (3 missing evals of ~2 min each). | S |
| Continued-pretraining recipe (re-warm, 5-25% replay) | L18 §5.2 | `ibrahim2024cpt` | c | If E8 shows forgetting, replay of old `joint` lines is the fix to cite. | - |
| LoRA learns less and forgets less | L18 §5.2 | `biderman2024lora` | c | Trainer updates top-N layers full-rank (`train_d1.py`); no LoRA path. Overfit seen in the kg run. | - |
| RL's Razor, Retaining by Doing, SDFT, sparse memory finetuning | L18 §5.2 | `shenfeld2026rlrazor`, `chen2026retaining`, `shenfeld2026sdft`, `lin2025sparsememft` | d | On-policy RL / self-distillation training regimes; not used. | - |
| Unknown facts hurt fine-tuning (MaybeKnown examples) | L18 §5.3 | `gekhman2024unknown` | c | Fits the kg retrain overfit (train loss 0.82 to 0.43, val 1.19 to 1.67). | - |
| Fine-tuning vs RAG for new knowledge | L18 §5.3 | `ovadia2024ftvsrag` | a | T4.9: edit graph and database, not weights (RAG 0.875 vs FT 0.504 in the lecture). | - |
| EntiGraph, Active Reading, SEAL (synthetic data into weights) | L18 §5.4 | `yang2025entigraph`, `lin2026activereading`, `zweiger2025seal` | c | The KG-derived per-option notes in `joint_tac` are a small version of KG-to-training-text. | - |
| Cartridges, Doc-to-LoRA, TTT-E2E | L18 §5.5 | `eyuboglu2026cartridges`, `charakorn2026doc2lora`, `tandon2025ttte2e` | d | Context-to-weights compilation; retrieval stays the default when facts change. | - |

## 7. The (b) experiments

### 7.1 Catalogue

Value (V) is 1-5: strength of the course link, how much it strengthens the report, and the chance of an informative result (a null result counts if it is reportable). V/E uses E = 0.5 / 1 / 2.5 days for S / M / L.

| Id | Experiment | Existing code to reuse | Data | Metric | Effort | V | V/E |
|---|---|---|---|---|---|---|---|
| E1 | Evaluation audit and ensembling on saved predictions: cluster-bootstrap CIs, McNemar, ECE, risk-coverage with +1/0/-1 utility, KILT-style supported accuracy, popularity strata, d1+LightGBM vote/average | `tools/eval/eval_d1_zero_shot.py` outputs, `tools/eval/feature_ranker.py`, `tools/eval/eval_arbiter.py` (d1 confidence) | `dataset/v1/eval/d1_eval_transformers_*.json` (`per_example`: 761 val + 766 test lines per checkpoint) | accuracy with 95% CI; paired delta; ECE; utility; ensemble delta | S | 5 | 10.0 |
| E2 | Context-conflict and context-reliance test on d1 (poisoned / shuffled / transplanted notes, edit lines) plus CAD and AdaCAD recombination | `clashbot/ranker.py::D1Ranker.probabilities`, `clashbot/d1_format.py` (NOTE_SEP), `tools/eval/eval_d1_zero_shot.py --model` | `dataset/v1/d1/joint_tac/{val,test}.jsonl`, `db/models/d1-ft-joint-tac` | accuracy per condition; follow-rate; ECE shift; CAD accuracy | M | 5 | 5.0 |
| E3 | Ranking-mode study: setwise vs pairwise (PRP) vs permutation vote; position bias; zero-shot Qwen3-14B reasoning ranker on a subset | `D1Ranker.choose_keys`, `d1_format.joint_request`, `clashbot/planner.py::MLXChat` | `dataset/v1/d1/joint`, `joint_tac`; 200-line test subset | 4-way accuracy; consistency across rotations; calls x ms | M | 4 | 4.0 |
| E6 | Held-out evaluation of the mined answer rules with source ablation, support and window sweeps (AMIE/TLogic scoring) | `clashbot/kg2.py::KG2.answers_for`, `clashbot/kg.py::KnowledgeGraph.counter_scores`, `tools/data/mine_pro_answers.py` | `dataset/v1/kg2/answer_events.jsonl` (3,620 events, split-tagged), `dataset/v1/games.jsonl` | hit@1/3, MRR, coverage, calibration vs `held` | M | 4 | 4.0 |
| E5 | Late interaction over units: MaxSim/Chamfer rerank of the struct-kNN top-100 vs the pooled vector; MUVERA-style FDE stretch | `clashbot/retrieval.py::StateIndex.knn`, `tools/eval/eval_retrieval.py::{load_queries,query_metrics}`, `clashbot/card_roles.py` | `dataset/v1/states.jsonl` units; 916 / 902 val / test queries | hit@10, card_hit@10, ndcg@10 by unit-count stratum; latency | M | 4 | 2.7 |
| E4 | Learned two-tower / InfoNCE state-action encoder: softmax vs logQ vs sigmoid vs BPR vs BCE; hard negatives; Matryoshka prefixes | `clashbot/features.py::featurize`, `db/state_features.npy`, `tools/data/build_ranker_dataset.py` (negatives), `tools/eval/eval_ranker_baselines.py` (candidate recall) | 6,658 train-eligible states (7,090 with an action), video splits | candidate recall@1/3/5/10 (exact key, card); R1 hit@10 when used as kNN space | L | 5 | 2.0 |
| E18 | Loss swap at fixed architecture on the 4-way ranker (BCE / BPR / softmax / LambdaRank) | `tools/eval/feature_ranker.py` (logreg and DCN-V2 scorers) | `dataset/v1/ranker/*.jsonl` | test accuracy, 3 seeds; ECE | S | 3 | 6.0 |
| E8 | Forgetting matrix across the d1 fine-tune chain (ACC and backward transfer) | `tools/eval/eval_d1_zero_shot.py --model --dataset` | `db/models/d1-ft-joint`, `-tac`, `-kg` x `dataset/v1/d1/joint`, `joint_tac`, `joint_tac.prev` | ACC matrix; BWT | S | 3 | 6.0 |
| E11 | UPGRADE of table rows T1.1 / T1.3 / T1.9 (class a), not a (b) idea. Retrieval knobs: int8 / sign-binary + rescoring / PQ recall vs exact; feature-block truncation; RRF k sweep and weighted fusion; BM25 zero-hit audit with tunable k1, b | `tools/eval/eval_retrieval.py::{run_hnsw,query_metrics}`, `clashbot/features.py::block_slices`, `StateIndex.rrf` | struct vectors, 916 / 902 queries | recall@10 vs exact; hit@10 vs bytes | S | 3 | 6.0 |
| E13 | BEIR-style domain-shift check: retrieval channels on live Training Camp queries vs pro val queries | `StateIndex.search`, `tools/eval/eval_retrieval.py::run_channels` | `runs/poc/*/decisions.jsonl` (`context_text`), val states | top-1 similarity, empty-pool rate, BM25-vs-kNN overlap | S | 2.5 | 5.0 |
| E15 | KGE controlled-comparison extension: ConvE plus loss/negative-sampling factor (NSSALoss) under equal tuning | `tools/eval/kge_link_prediction.py` (PyKEEN, filtered evaluator) | `dataset/v1/kg/edges.jsonl` COUNTERS split 80/10/10 | MRR, Hits@k, 3 seeds | S | 2.5 | 5.0 |
| E7 | Deck completion with RecSys models (iALS, EASE, LightGCN) vs `OppDeckModel` | `clashbot/opp_model.py`, `tools/eval/eval_opp_model.py` (logo / lodo), `tools/data/build_deck_cooc.py` | KB decks + 197 game decks | recall@(8-k), precision@3 | M | 3 | 3.0 |
| E9 | Mini-BFCL on the six tools (simple, multiple, irrelevance, missing parameter, multi-turn) with Qwen3-4B and 14B | `clashbot/tools.py`, `clashbot/planner.py::{LLMPlanner,parse_tool_calls}`, `tools/eval/bench_planner.py` | ~140 synthetic prompts | exact match; abstention; schema-reject rate | M | 3 | 3.0 |
| E12 | Query expansion for the BM25 channel: KG-derived tokens (zero latency) and Qwen3-4B pseudo-document (HyDE / Query2doc style) | `StateIndex.bm25`, `KnowledgeGraph.counter_scores`, `KG2.answers_for`, `MLXChat` | 916 / 902 queries (300-query subset for the LLM variant) | hit@10, card_hit@10, ndcg@10; added ms | M | 3 | 3.0 |
| E10 | Near-duplicate / contamination audit (SimHash + cosine) and false-negative audit of hard negatives | `db/state_features.npy`, `clashbot.sqlite` states, `tools/data/build_ranker_dataset.py` | val/test vs train pool | share with a close train neighbour; same-key rate | S | 2 | 4.0 |
| E16 | Scene-graph (relation triple) rendering of the perceived field as BM25/context text | `clashbot/state_text.py`, `tools/eval/eval_retrieval.py::run_channels` | states with units | hit@10, card_hit@10 | S | 2 | 4.0 |
| E14 | Inductive / cold-start COUNTERS prediction: held-out cards, KGE vs role transfer vs feature logreg | `tools/eval/kge_link_prediction.py`, `clashbot/card_roles.py`, `KG2` SIMILAR_ROLE | `dataset/v1/kg/edges.jsonl`, `clashbot/resources/card_roles.json` | filtered MRR / Hits@10 on cold cards | M | 3 | 2.0 |
| E17 | Factorized (card, lane, zone) vs joint d1 fine-tune: semantic-ID analogy, invalid-composition rate | `tools/train/train_d1.py --variant factorized`, `eval_d1_zero_shot.py --variant factorized` | `dataset/v1/d1/factorized` (3,641 / 761 / 766 lines) | composed exact-key accuracy; invalid rate | L | 3 | 1.2 |

### 7.2 Ranking and top 6

Top 6 = every experiment with V of 4 or more, ordered by V/E. The cheaper ideas (V of 3 or less) are listed in 7.4 as spare-afternoon fillers.

| Rank | Id | Experiment | Effort | Blocks touched | Why it earns its place |
|---|---|---|---|---|---|
| 1 | E1 | Evaluation audit and ensembling | S | KG L5 audit, RAG L11/L13, Agents L16 | Every headline 4-way number rests on n = 766; the binomial standard error is 1.8 pts, so 0.497 vs 0.491 vs 0.483 are not yet separable claims. No GPU. Also tests the 'obvious next experiment' already noted in STATUS (d1 + LightGBM). |
| 2 | E2 | Context-conflict and CAD | M | RAG L11/L12, Agents L17 | Directly explains why the KG-trained d1 did not improve and what a wrong perception does to the notes. Covers ClashEval, Chameleon, CAD, AdaCAD, IKE, RAFT distractors in one run. |
| 3 | E3 | Ranking modes and order sensitivity | M | Retrieval L10, Agents L15 | Turns T1.7 (setwise) into a comparison of ranking modes with a real consistency metric and a latency cost per decision. |
| 4 | E6 | Held-out rule evaluation | M | KG L6/L7 | The KG's main data-driven contribution (mined answers, 3,620 events) currently has no held-out number. |
| 5 | E5 | Late interaction over units | M | Retrieval L9, Dense L2 | Tests the lecture's capacity-limit argument on the one structure here that is genuinely a set (units on the board). |
| 6 | E4 | Two-tower / InfoNCE encoder | L | Dense L2, RAG L13, Retrieval L8 | Only experiment that exercises the Dense block's core (InfoNCE, hard and false negatives, SigLIP, Matryoshka) and the two-tower/logQ/loss material together; popularity is a strong baseline, so a null result is still a finding. |

### 7.3 Experiment plans for the top 6

#### E1: Evaluation audit and ensembling (S, about 0.5 day, no GPU)

| Field | Plan |
|---|---|
| Question | Are the 4-way gaps real (d1-ft-joint-tac 0.497, LightGBM 0.491, d1-ft-joint 0.483, placement prior 0.384 on the 766 test lines)? Is d1 calibrated enough to hold when unsure? Does it only learn popular keys? Does an ensemble beat its members? |
| Build | New `tools/eval/eval_stats.py` (numpy only, about 200 lines). Add `--dump-preds` to `tools/eval/feature_ranker.py` so logreg / LightGBM / DCN-V2 write per-line option scores. |
| Data | `dataset/v1/eval/d1_eval_transformers_d1-ft-joint{,-tac,-kg,_on_tac}.json` `per_example` (id, split, gold, pred, 4 probabilities); option order from `dataset/v1/d1/joint_tac/<split>.jsonl`; `dataset/v1/placement_prior.json`; `game_id` for clustering. |
| Protocol | Cluster bootstrap by `game_id` (10k resamples) for accuracy CIs and paired differences; exact McNemar; ECE (15 bins) and Brier; risk-coverage with +1 correct / 0 hold / -1 wrong utility (CRAG benchmark scoring), threshold fixed on val and applied to test; supported accuracy (gold option also carries a non-filler lookup/tactics/KG note, KILT style); accuracy by tercile of log train-count of the gold key (PopQA style); ensembles = mean log-probability of d1-tac and LightGBM (+ prior) with the weight fitted on val only, and majority vote (the 'debate vs vote' question). |
| Metrics | Accuracy with 95% CI; paired delta with CI and p; ECE; best utility and its coverage; accuracy by popularity tercile; ensemble delta. |
| Expected / risks | At n = 766 the 95% interval is about +/-3.5 pts, so gaps under about 3 pts will probably not separate. An ensemble gain is plausible because LightGBM's top features are popularity priors that d1 never sees (hypothesis, not a result). LightGBM configuration was chosen on val, so only test numbers are clean. |
| Course links | `ruffinelli2020olddog`, `sun2020reevaluation` (evaluation audit); `petroni2021kilt`, `yang2024cragbench` (provenance-aware scoring, abstention); `mallen2023popqa`; `du2024debate`; `rendle2020ncfvsmf` (tuned baselines). |

#### E2: Context-conflict test and CAD/AdaCAD (M, about 1 day, about 12 min GPU)

| Field | Plan |
|---|---|
| Question | Does d1-ft-joint-tac follow the `tactics:`, `KG:` and per-option notes even when they are wrong, and can contrasting with/without-context logits (CAD) recover accuracy? |
| Build | New `tools/eval/context_conflict.py`. `D1Ranker(model_path='db/models/d1-ft-joint-tac', backend='transformers').probabilities(state, questions)`; split notes on `clashbot/d1_format.py::NOTE_SEP`; drop state lines starting `tactics:` / `KG:`. |
| Data | `dataset/v1/d1/joint_tac/{val,test}.jsonl` (761 / 766 lines; current version carries tactics and KG lines). |
| Conditions | C0 as exported. C1 notes and tactics/KG lines removed (the `joint` format). C2 notes shuffled across the 4 options. C3 poisoned: the gold option's note swapped with a distractor's, `tactics:` line naming a wrong counter. C4 transplant: tactics/KG block from a random other line (irrelevant distractor). C5 edit line, for example 'goblin-barrel costs 4 elixir', inserted where it changes affordability (IKE-style). C6 CAD: score = (1+a) log p(C0) - a log p(C1) with a in {0, 0.25, 0.5, 1}; AdaCAD with a = Jensen-Shannon divergence of the two distributions; a tuned on val. |
| Metrics | Accuracy per condition; follow-rate (share of lines whose pick moves to the option the corrupted note favours); accuracy when the note is right vs wrong; ECE shift; CAD/AdaCAD test accuracy with the E1 paired bootstrap. |
| Expected / risks | C1 should drop (the model was trained with notes). Poisoned notes (relevant but misleading) should hurt more than random transplants, as RAGChecker's noise sensitivity found (26-37% vs 3.5-6%). CAD may add 0-2 pts or nothing; AdaCAD assumes the context is right, which perception errors violate. About 1,527 lines x 7 conditions x 65 ms is about 12 minutes. |
| Course links | `wu2024clasheval`, `xie2024chameleon`, `shi2024cad`, `wang2025adacad`, `zheng2023ike`, `ru2024ragchecker`, `zhang2024raft`, `cui2025tte` (traces must be trained on). |

#### E3: Ranking-mode study (M, about 1 day, about 35 min compute)

| Field | Plan |
|---|---|
| Question | Does d1 depend on option position? Is an all-pairs pairwise tournament (PRP) or a permutation vote better than one setwise call? Can a zero-shot reasoning LLM rank as well as the fine-tuned 600M? |
| Build | New `tools/eval/ranking_modes.py`; call `D1Ranker.probabilities(state, questions)` (`clashbot/ranker.py`) with each line's own `state` string and a re-ordered, reversed or two-option subset of its `questions.play.criteria` (d1 accepts any 2+ options; `d1_format.joint_request` shows the format). |
| Data | `dataset/v1/d1/joint/{val,test}.jsonl` with `db/models/d1-ft-joint`; the same on `joint_tac` with `d1-ft-joint-tac`; a 200-line test subset for Qwen3-14B-4bit (`COACH_MODEL_ID`, thinking off and on). |
| Modes | (1) setwise baseline; (2) 4 cyclic rotations + reversal: accuracy per slot arrangement, agreement with the original order, 'all arrangements correct' (a pass^k analogue), mean-probability vote; (3) PRP: 6 two-option calls per line, win count with probability tie-break; (4) setwise top-2 then one pairwise call; (5) optional pointwise through d1's native `score` question (zero-shot only); (6) Qwen3-14B setwise prompt. |
| Metrics | 4-way accuracy; position bias (saved `pred_position_counts` [190, 199, 196, 181] already look flat); consistency; calls x ms per decision (d1 is about 65 ms per call). |
| Compute | 1,527 lines x 11 calls x 65 ms is about 18 min on MPS; Qwen subset 200 x about 4 s is about 15 min without thinking. |
| Expected / risks | Position bias is probably small (options were shuffled in training). PRP may lose because d1 never saw 2-option questions; a permutation vote may add 0-1 pt (hypotheses). Zero-shot d1 is at chance, so a zero-shot 14B model may be too (the lecture's RankGPT result needed GPT-4). |
| Course links | `sun2023rankgpt` (65.80 to 32.77 when the input is reversed), `qin2024prp` (order insensitive), `zhuang2024setwise`, `reddy2024first`, `weller2025rank1`, `zhuang2025rankr1`, `yao2025taubench` (pass^k). |

#### E6: Held-out evaluation of the mined answer rules (M, about 1 day, CPU)

| Field | Plan |
|---|---|
| Question | Are the mined 'opponent played X, pro answered Y' rules predictive on held-out games, and which evidence source (pro, role transfer, wiki, simulation) carries the score? |
| Build | New `tools/eval/eval_answer_rules.py`; reuse `clashbot/kg2.py::KG2.answers_for` and `KG2.source_weights`; for the window sweep, re-run `tools/data/mine_pro_answers.py` with a different `WINDOW_S`. |
| Data | `dataset/v1/kg2/answer_events.jsonl` (3,620 events with `split`, `threat`, `answer`, `held`, `situation`); `dataset/v1/games.jsonl` for the player's 8-card deck; rules from train events only (as `answers.jsonl`). |
| Protocol | For each val/test event rank the game's own deck by `answers_for(threat, own_deck)`. Baselines: wiki COUNTERS only (`KnowledgeGraph.counter_scores`), train popularity of the answer card, random in deck. Ablate each of the 4 sources; sweep support threshold n >= 1, 3, 5, 10 (AMIE support / head coverage) and answer window 2, 4, 8 s (TLogic timestamp constraint); report standard vs PCA-style confidence (denominator = threats the pro answered with some in-deck card). |
| Metrics | hit@1, hit@3, MRR over the deck; coverage (share of events with any evidence); calibration of score against `held`. |
| Expected / risks | Pro evidence should dominate where support is 5 or more; role transfer should cover cards with no pro data; wiki-only should sit near random (hypotheses). K-series leakage note (FIX_LIST #49) concerns train lines, not this protocol, as long as only train events build the rules. |
| Course links | `lajus2020amie3`, `liu2022tlogic`, `ruffinelli2020olddog` (equal-footing comparison), `luo2025chatrule` (the rules ChatRule would verbalise). |

#### E5: Late interaction over units (M, about 1.5 days, CPU)

| Field | Plan |
|---|---|
| Question | Does set-based MaxSim over the units on the board retrieve better pro states than the pooled 644-d vector, whose single slot averages away rare decisive units? |
| Build | New `tools/eval/eval_multivector.py`. First stage `StateIndex.knn(depth=100)`. Token vector per unit: card role vector (`clashbot/card_roles.py`), side, lane and zone one-hots, ETA; hand cards as extra tokens. Chamfer = sum_i max_j cos(q_i, d_j), fused with the struct score by RRF. Stretch: MUVERA-style fixed-dimensional encoding (SimHash 4 bits, 5 repetitions) compared with exact Chamfer. |
| Data | `dataset/v1/states.jsonl` (88% of states have at least one unit, 2-4 typical); the 916 / 902 val / test queries of `tools/eval/eval_retrieval.py::load_queries`; train pool only. |
| Metrics | hit@10, card_hit@10, vote_recall@10, ndcg@10 (`query_metrics`), split by unit count (0, 1-2, 3+); rerank latency; FDE recall@100 of exact Chamfer. |
| Expected / risks | Raw struct baseline is hit@10 0.166 / 0.136 (val / test), so even +2 pts shows; gain should concentrate in states with 3+ units. A null result is reportable. 88% of states are non-empty, so the 12% empty ones fall back to the pooled score. |
| Course links | `khattab2020colbert`, `santhanam2022plaid` (prune then score), `dhulipala2024muvera`, `xiao2026metaembed` (learned tokens as a follow-up). |

#### E4: Learned two-tower / InfoNCE encoder (L, about 2 days, minutes of compute per run)

| Field | Plan |
|---|---|
| Question | Can a learned metric beat the hand-built vector and popularity at candidate recall, and which loss matters? |
| Build | New `tools/train/train_two_tower.py` and `tools/eval/eval_two_tower.py` (torch). State tower: MLP 644 to 256 to 64 on `featurize` vectors (`db/state_features.npy`, row order from `StateIndex`). Action tower: sum of card + lane + zone embeddings (a three-codebook identifier) to 64. Losses: L1 in-batch softmax; L2 plus logQ correction (and the revised weighting); L3 plus hard negatives (same card other spot, other card same spot, as in `build_ranker_dataset.py`); L4 SigLIP sigmoid; L5 BPR; L6 BCE with sampled negatives; L7 full softmax over the 630 keys (reference). Matryoshka loss at prefixes 8 / 16 / 32 / 64. Optional symmetric CLIP-style loss. |
| Data | 6,658 train-eligible states, 7,090 with an action, split by video; 3 seeds. |
| Evaluation | (A) the embedding as kNN space, scored with the R1 metrics against raw struct hit@10 0.166 / 0.136; (B) direct action retrieval, candidate recall@1/3/5/10 at exact-key and card level, against kNN vote 0.361 / 0.317 and popularity 0.499 / 0.402 at exact-key recall@10 (val / test, n 761 / 766; card level 0.720 / 0.651 vs 0.673 / 0.607), all from `docs/data/ranker_baselines_v1.json`. |
| Expected / risks | Popularity is a strong exact-key baseline (0.499 vs 0.361 for kNN on val), so expect to match it rather than beat it clearly; the loss ablation and the false-negative effect of repeated popular keys inside a batch are the learning. Keep 'hold' out of the candidate set. |
| Course links | `oord2018cpc`, `radford2021clip`, `zhai2023siglip`, `xiong2021ance`, `yi2019twotower`, `khrylchenko2025logq`, `rendle2009bpr`, `wu2024ssm`, `kusupati2022mrl`, `kang2018sasrec`/`klenitskiy2023dross` (full softmax as reference). |

### 7.4 The other 12 experiments in one line each

| Id | Plan | Effort |
|---|---|---|
| E18 | Add `--loss {softmax,bpr,bce,lambdarank}` to the logreg/MLP scorers in `tools/eval/feature_ranker.py`; same features, 3 seeds, report test accuracy and ECE. 'Loss, not architecture' on our data. | S |
| E8 | Run `eval_d1_zero_shot.py --model db/models/d1-ft-joint-kg --dataset ...joint` and the other 2 missing cells (tac on `joint`, kg on `joint_tac.prev`); table of ACC and BWT. If forgetting shows, cite replay (`ibrahim2024cpt`). | S |
| E11 | Quantise `db/state_features.npy` (int8; sign-binary + float rescoring; PQ with sklearn KMeans), plot recall@10 vs bytes next to the existing HNSW curve; drop feature blocks (`block_slices`) as a Matryoshka-style truncation; RRF k in {1, 10, 60, 200} and weighted RRF; count queries where FTS5 BM25 returns fewer than 10 hits. | S |
| E13 | Replay every `runs/poc/*/decisions.jsonl` `context_text` through `StateIndex.search` and the channels; compare top-1 similarity, empty-pool rate (STATUS: 105 of 197 pro games share none of the user's cards) and BM25/kNN overlap against val queries. | S |
| E15 | Add ConvE and `NSSALoss` to `kge_link_prediction.py` under the existing equal grid; test whether TransE with the better loss closes the gap to RotatE (0.082 vs 0.159 MRR now). | S |
| E7 | Implement iALS (numpy), EASE (closed form, 126 x 126) and LightGCN (torch) for hidden-card prediction; reuse the leave-one-game-out and leave-one-deck-out protocols and baselines of `eval_opp_model.py`. | M |
| E9 | Write about 140 prompts over the 6 tools (30 simple, 30 multiple, 20 irrelevance, 20 missing parameter, 20 enum/range violation, 20 two-step); score exact name + argument match; run Qwen3-4B-Instruct-2507-4bit and Qwen3-14B-4bit. Explains the observed 0 tool calls. | M |
| E12 | Prepend KG tokens (counters / answers of the threat cards, archetype tags) to the FTS5 query; on 300 queries also a Qwen3-4B pseudo-document; compare hit@10 and latency. | M |
| E10 | Cosine/SimHash audit of val/test against train (pre-check run for this report found none at cosine 0.95 or above) plus a false-negative audit: share of the 3 hard negatives that a close train neighbour actually played. | S |
| E16 | Render the field as relation triples in `state_text.py` for the BM25 channel only; compare hit@10 against the flat unit list. | S |
| E14 | Hold out all COUNTERS edges of about 10 cards; compare KGE (cannot embed an unseen node), role-similarity transfer and a feature logreg with the filtered evaluator of `kge_link_prediction.py`. | M |
| E17 | `python tools/train/train_d1.py --variant factorized` (about 10 h GPU per the STATUS estimate), then compose card, lane, zone into a key and compare exact-key accuracy and invalid-composition rate with joint. | L |

## 8. Observations on the table and the inputs

| Observation | Detail | Suggested action |
|---|---|---|
| The Dense Representation block has no rows in the table | None of Lectures 2-4 (ColBERT, InfoNCE, CLIP, SigLIP, hard negatives, multimodal embedders, DINO, Think-Then-Embed) is mapped. Only E4 and E5 give it substance; the rest is cite or n/a. | Add a short Dense section to the report that points at E4/E5 and cites the rest. |
| T3.3 window | The table says 'pro answered with Y within 4 s'; `tools/data/mine_pro_answers.py` uses `WINDOW_S = 8.0` (the answer is the first defensive play within 8 s). | Fix one of them before it reaches the report; E6 sweeps the window. |
| Table section references for Agents | T4.2 ('section 7.3'), T4.6 ('section 4.1'), T4.11 ('section 7.2') cite an older document; the current write-up has memory in section 3.1 and no execution-safety section. T4.2 (executor separation) has no technique row in the current write-up. | Cite by lecture and technique, not by section number. |
| Reference numbers | Bracketed numbers in the table match the current compiled numbering for Retrieval (BM25 [32], RRF [3], HNSW [20]) and RAG (BPR [50], SASRec [61], FLARE [27]). For KG they match partly (TransE [2], RotatE [36], semantic loss [41] match; AMIE 3 / TLogic [23, 28] are [19, 24] and ChatRule [29] is [25] in the current order). For Agents they do not (ReAct is [9] in the current first-citation order, [4] in the table). | Cite by bib key in the report. |
| Artifacts exist for several rows | `docs/data/` already holds `retrieval_ablation_v1` (HNSW, pre/post filter), `kge_link_prediction_v1`, `feature_ranker_v1`, `next_card_v1`, `edit_eval_v1`, `arbiter_v1`, `act_gbdt_v1`, `planner_bench_v1`, `tactics_eval_v1`. The table's statuses predate them. | Left to the status audit. |
| KG bib entries without a section | 13 entries (R-GCN, CompGCN, GraIL, NBFNet, RED-GNN, A*Net, AdaProp, ULTRA, MOTIF, TiTer, TTransE, TA-DistMult, lecture 6 deck) are in the KG `refs.bib` but never cited. They are the GNN-based reasoning and temporal material of Lecture 6. | Cited here as class c/b (E14); decide whether to keep them in the bibliography. |
| Course number | The folders say CAP 6000; every write-up and every lecture entry says CAP 6938 AI Knowledge Systems. | Use whichever the report template requires. |
| Pre-check run for E10 | On `db/state_features.npy` (val 591, test 547 train-eligible play states vs the train pool of train-eligible states with an action): mean best cosine 0.65 and 0.65, none at 0.95 or above, top-1 neighbour shares the exact key 2.4% / 1.8% of the time. Not a full audit. | Run E10 only if the report wants the full leakage statement. |

## 9. Lecture-citation keys in `docs/report/refs.bib`

| Key | Write-up | What it is |
|---|---|---|
| `cap6938L05` | KG | Lecture 5, Knowledge Graphs (author given as CAP 6938 Course Staff) |
| `cap6938L06` | KG | Lecture 6, KG Learning and Reasoning (GNN deck; never cited in the KG write-up) |
| `cap6938L06C` | KG | Lecture 6 (Topic 02C), Temporal and Visual Knowledge Graphs (the Lecture-6 key the KG write-up cites) |
| `cap6938L07` | KG | Lecture 7, Symbolic and Neuro-Symbolic Reasoning |
| `cap6938L08 / L09 / L10` | Retrieval | Lectures 8 to 10: Representation (Sep 29), Indexing and Search (Oct 1), Ranking and Generative Retrieval (Oct 6) |
| `lim2026l02 / l03 / l04` | Dense | Lectures 2 to 4: Dense Knowledge Representation I to III (Sep 1, 3, 8) |
| `lim2026moemb` | Dense | MoEmb, unpublished lab work in progress presented in Lecture 4 |
| `lim2026l11 / l12` | RAG and Rec | Lectures 11 and 12: RAG I and II (Oct 1, Oct 6) |
| `lim2026l13 / l14` | RAG and Rec | Lectures 13 and 14: Recommendation I and II (both dated Oct 8) |
| `lim2026l15 / l16` | Agents | Lectures 15 and 16: AI Agents I and II (Oct 20, 22) |
| `lim2026l17 / l18` | Agents | Lectures 17 and 18: Knowledge Updating I and II (Oct 27, 29) |

Two naming schemes coexist (`cap6938Lnn` in the KG and Retrieval files, `lim2026lnn` in the others) and the RAG file's dates for L11 to L14 overlap those of L9 and L10 in the Retrieval file. Both are kept as found; verify the dates before citing.

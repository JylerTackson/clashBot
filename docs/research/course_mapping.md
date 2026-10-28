# Course material → project mapping (Opus, 2026-10-28)

Sources: the four CAP 6938 segment write-ups uploaded on 2026-10-28.
Reference numbers below are the numbers used inside each PDF, prefixed by segment:
**RAG** = "RAG and Recommendation System Segments" (L11–L14), **RET** = "Retrieval Systems Segment Breakdown" (L8–L10),
**KG** = "Knowledge Graph Segment Breakdown" (L5–L7), **AGT** = "LLM Agents, Memory, Long-Horizon Reasoning, and Knowledge Editing" (L15–L17).

Priority: **core** = built and evaluated in the project; **ablation** = an experiment/comparison in the report; **cite** = discussed as related work or design justification only.
The aim is a few techniques done well with measured comparisons, not a long list of name-drops.

## 1. Retrieval systems (RET) → the fast loop's candidate generator

| idea | where it lands | priority |
|---|---|---|
| BM25 sparse scoring [RET 32] over `state_text` + tags | sparse channel of hybrid retrieval | core |
| Dense bi-encoder representation; bi-encoder capacity limit [RET 14] | text-embedding channel over `state_text`; plus a hand-built structured state vector (elixir, hand, deck, unit occupancy per lane/zone/side, tower HP, threat ETA) | core |
| Hybrid BM25 + dense fused by **Reciprocal Rank Fusion** [RET 3] (the post-BEIR default) | fuse sparse, text-dense and structured-dense ranked lists | core |
| Exact search is affordable below ~1M vectors; HNSW [RET 20] recall/QPS dial (`efSearch`) | 10.9K vectors → exact search is the honest baseline; HNSW measured as the corpus grows with own games | ablation (latency vs recall curve) |
| Predicate filtering: pre-filter degrades, post-filter can return nothing [RET §2.2] | filter by own deck / archetype / phase; measure empty-result rate of post-filtering vs pre-filtering | ablation |
| Quantization (int8/PQ) [RET 11], Matryoshka truncation [RET 15] | memory/latency on the Mac | cite (ablation if time) |
| FreshDiskANN-style incremental updates [RET 35] | appending own games to the live index | cite + implemented as incremental insert |
| Cross-encoder as reranker / teacher [RET 22,23]; LLM rankers pointwise/pairwise/listwise/setwise [RET 28,38,49] | the Laya/d1 ranker is a setwise multiple-choice ranker over ≤4 candidates | core |
| Evaluation: recall@k, nDCG, BEIR lesson that BM25 is a strong baseline [RET 40] | retrieval evaluated by whether neighbours' actions contain the pro's action | core |

## 2. RAG and recommendation (RAG) → system framing and the ranker

| idea | where it lands | priority |
|---|---|---|
| Parametric memory bottleneck [RAG 8]; RAG [RAG 10] | why the agent retrieves pro states instead of trusting a small model's weights | cite |
| Long context ≠ more passages; inverted-U with hard negatives [RAG 16,17] | cap pre-game context; measure plan quality vs number of retrieved docs | ablation (small) |
| RAFT: train the reader on golden + distractor context [RAG 15] | ranker training examples include distractor candidates and noisy neighbours | core (data format) |
| Adaptive retrieval triggers: FLARE confidence trigger [RAG 27], Adaptive-RAG router [RAG 31], Self-RAG/CRAG evaluators [RAG 29,30] | decision trigger + "consult LLM only when candidate generators disagree and time allows" | ablation |
| GraphRAG / HippoRAG (PPR over a triple graph) [RAG 39,41]; "start with dense or hybrid RAG" [RAG 45] | KG as a second candidate generator; PPR over the card graph from current threats | ablation |
| **Two-stage serving: candidate retrieval → ranking** [RAG §4.4–4.5] | the core fast-loop architecture | core |
| Ranking losses: pointwise BCE, pairwise BPR [RAG 50], listwise sampled softmax [RAG 51]; hard negatives | ranker loss choice; negatives from neighbours' different actions | core |
| Feature-interaction rankers FM/DLRM/DCN-V2 [RAG 56–58] | feature-ranker baseline on the structured vector | ablation (baseline) |
| "Tuned simple baselines win" [RAG 5,6,7] | popularity and kNN baselines reported next to the fine-tuned ranker | core (evaluation) |
| TALLRec: LoRA-tune an LM as a recommender on few samples [RAG 70]; P5 text-to-text [RAG 69] | fine-tuning Laya/d1 on multiple-choice decisions | core |
| Sequential recommenders SASRec/BERT4Rec [RAG 61,62] | next-action prediction from the recent-play sequence (own + opponent) as a feature or baseline | ablation (stretch) |
| RETRO's test-in-retrieval-source contamination [RAG 12] | split by video; exclude same-game neighbours at evaluation | core (protocol) |

## 3. Knowledge graphs, temporal KGs, neuro-symbolic (KG)

| idea | where it lands | priority |
|---|---|---|
| KG of cards, units, decks, archetypes, evolutions, heroes; relations COUNTERS, WEAK_TO, SYNERGY, IN_DECK, ARCHETYPE_OF, SPAWNS_UNIT, EVOLVES_TO | built from wiki sections + RoyaleAPI decks | core |
| KG embeddings TransE/DistMult/ComplEx/RotatE [KG 2,42,38,36]; symmetric vs asymmetric relations (SYNERGY is symmetric, COUNTERS is not) | link prediction of missing COUNTERS edges; compare models under the same training budget | ablation |
| Evaluation audit: equal tuning [KG 34], random tie-breaking [KG 37] | protocol for the KGE experiment | core (protocol) |
| Temporal KG: match events as timestamped triples (player, plays, card, t, tile) [KG §2] | game states as a temporal KG; CyGNet copy-mechanism intuition: opponents repeat cards [KG 46] | ablation (opponent next-card prediction) |
| Rule mining AMIE 3 [KG 23]; TLogic temporal rules [KG 28] | mine "opp plays X at bridge → pro answers Y within 3 s (positive outcome)" rules from states; rules become KG edges with support/confidence | core |
| Scene graphs from video [KG 25, 11], ESCA (perception errors dominate agent failures) [KG 20] | per-frame units + relations (attacking, near tower, same lane) as the perception layer's output; justification for perception QA | cite + light implementation |
| Neuro-symbolic: semantic loss / constraints [KG 41], DeepProbLog/Scallop [KG 30,19] | hard legality rules filter candidates; optional constraint-violation penalty in ranker training | core (filter); ablation (penalty) |
| LLMs as rule generators, ChatRule [KG 29] | slow-loop LLM verbalises mined rules; rules are scored against the graph before acceptance | stretch |

## 4. Agents, memory, knowledge editing (AGT)

| idea | where it lands | priority |
|---|---|---|
| Function calling with typed JSON-schema tools; MCP [AGT 1–3]; BFCL lesson (know when not to call) | read-only tools for the slow-loop LLM: `get_state`, `search_similar_states`, `kg_query`, `card_info`, `match_memory`, `get_plan` | core |
| ReAct [AGT 4]; Reflexion verbal feedback between trials [AGT 5] | slow-loop plan updates; post-game reflection written to procedural memory | core |
| Tree search / LATS [AGT 7,8] — only where branching repays latency | not in the fast loop; cite as why we do not search per tick | cite |
| Context management: Lost in the Middle [AGT 12], Context Rot [AGT 13], ACE playbook [AGT 22], MemGPT paging [AGT 23] | bounded pre-game context; a living "playbook" per deck/archetype | core |
| Three memory types: episodic / semantic / procedural [AGT §4.1]; Mem0 ADD/UPDATE/DELETE [AGT 25]; ReasoningBank [AGT 27]; Generative Agents recency·importance·relevance [AGT 24] | episodic = state DB (pro + own games); semantic = KG facts and card stats; procedural = playbook lessons. Memory write operations follow Mem0's add/update/delete/ignore | core |
| Knowledge editing criteria: reliability, generalization, locality, portability [AGT 46] | evaluating a KB edit (e.g., a balance change to a card's cost/HP) | core |
| Edit routes: RAG store vs in-context (IKE) [AGT 56] vs weight edit ROME/MEMIT [AGT 49,51] vs external edit memory SERAC/GRACE [AGT 53,54] | primary route = non-parametric edit of KG/state DB with provenance and validity windows; comparison: in-context edit for the LLM, optional LoRA re-tune of the ranker | core (non-parametric) / ablation |
| Knowledge conflicts, provenance, ClashEval [AGT 64,67] | patch-versioned facts: when a new balance change conflicts with old pro data, retrieval down-weights pre-patch samples involving that card | core |
| Trajectory-level evaluation, variance over repeated trials [AGT §7.2] | live evaluation over N friendly battles, reporting invalid actions, latency, win rate with variance | core |
| Multi-agent pitfalls [AGT 44] | justification for one planner + one ranker instead of debating agents | cite |

## 5. Proposed 8-page report outline

1. Introduction and problem (real-time game agent on commodity hardware).
2. Background, one paragraph per segment (RET, RAG/RecSys, KG, AGT).
3. Data: collection, GameStateSample schema, cleaning and repair, splits.
4. System: perception → state; hybrid retrieval; KG candidate generation and legality rules; ranker; slow-loop LLM with read-only tools; memory and knowledge editing.
5. Experiments: retrieval (recall@k/nDCG, exact vs HNSW, filter strategy); ranker vs baselines; KG link prediction + mined rules; knowledge-edit evaluation (reliability/locality/portability); end-to-end latency and live trials.
6. Limitations and ethics (single creator, win bias, perception noise, ToS).
7. Conclusion.

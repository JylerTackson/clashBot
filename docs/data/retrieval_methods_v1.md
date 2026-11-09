# Retrieval methods for the KB and the KG, benchmarked as retrievers and as decision models (MAC-AU part C)

Run at HEAD `5bb8048ccb`. 766 decision lines of `dataset/v1/d1/joint_tac_v2/test.jsonl` from 27 test games (5 videos); CIs are a cluster bootstrap over games (10000 resamples, seed 0). Chance on a 4-way line is 0.25.

## Headline

- Best KB method as a decision model: BM25 (FTS5 over context_text), accuracy 0.318 [0.283, 0.351] (coverage 0.53, query ms p50 / p95 22.14 / 31.97); d1-ft-joint-tac on the same lines 0.507 [0.473, 0.539].
- Best KG method as a decision model: KG v2 simulation (STOPS) only, accuracy 0.258 [0.250, 0.267], card accuracy 0.517 [0.501, 0.534] (random 0.250 / 0.500; the KG cannot choose a placement).
- As a threat-to-answer ranker the best KG method is train popularity of the answer card (MRR 0.384 [0.349, 0.427]; random 0.345).

## Setup and leakage

- KB retrieval is restricted to the **train** split (`splits=['train']`; 4997 of 10923 indexed states are train states with an action). Every retrieved row was checked: split in train and game not one of the 27 evaluated games (125599 hits checked, none violated). Splits are by video, so no other match of an evaluated video is in the pool.
- KG rules are the train-split answer rules: rules over all events equal rules over train events only (True) and equal the shipped answers.jsonl (True); popularity counts train events only. The v1 ANSWERED_BY edges are mined from train states. KGE: the graph of `kge_link_prediction_v1` (setting `full`) with the IN_DECK / HAS_ARCHETYPE edges of decks played only in val / test games removed, trained on every COUNTERS edge (wiki facts), best config per model from that file (RotatE dim 64 lr 0.001; TransE dim 64 lr 0.01), seed 0.
- **Decision rule.** KB: an option's score is the sum of the retrieval scores of the top-20 hits whose action key equals the option key. KG: an option's score is the sum, over the line's recent enemy threats (opponent plays of the last 5 s and enemy units reaching our towers within 8 s), of the method's score for the option's card against the threat, taken over the own deck. Ties split uniformly (expected credit); a line where no option has a positive score is unsupported (all four tied, 0.25). McNemar uses one seeded tie-break draw per line.
- **The KG cannot place.** The four options are a 2x2 grid (the pro's play, the same card elsewhere, another card at the pro's placement, another card elsewhere), so two options share each card. A card-level KG score can at best separate the two cards and then splits 50/50: expected accuracy is capped at 0.5 and random card accuracy is 0.5.
- Latency is wall time per query after a 10-query warm-up, measured with the heavy lock held (so no local model runs alongside); other CPU jobs may still have run, so read the figures as upper bounds. KB queries include featurising and, for bge, one single-query encode on MPS.

## KB: the pro-state database (top-20 hits per query)

| method | hit@10 | card_hit@10 | MRR | 4-way acc | card acc | coverage | acc if supported | empty | recall@20 vs exact | ms p50 / p95 |
|---|---|---|---|---|---|---|---|---|---|---|
| dense exact (644-d structured, dot) | 0.144 [0.093, 0.200] | 0.423 [0.347, 0.513] | 0.067 [0.038, 0.102] | 0.284 [0.243, 0.331] | 0.476 [0.442, 0.513] | 0.53 [0.46, 0.60] | 0.314 [0.235, 0.391] | 0.000 | - | 1.16 / 1.22 |
| BM25 (FTS5 over context_text) | 0.166 [0.126, 0.205] | 0.491 [0.399, 0.580] | 0.060 [0.045, 0.075] | 0.318 [0.283, 0.351] | 0.526 [0.495, 0.556] | 0.53 [0.46, 0.59] | 0.379 [0.313, 0.440] | 0.000 | - | 22.14 / 31.97 |
| RRF dense + BM25 (live bot) | 0.175 [0.125, 0.229] | 0.501 [0.425, 0.586] | 0.074 [0.053, 0.097] | 0.305 [0.268, 0.347] | 0.495 [0.462, 0.529] | 0.58 [0.51, 0.64] | 0.346 [0.282, 0.412] | 0.000 | - | 24.19 / 33.84 |
| bge-small text embeddings (cosine) | 0.155 [0.103, 0.214] | 0.450 [0.382, 0.521] | 0.067 [0.041, 0.097] | 0.307 [0.265, 0.351] | 0.504 [0.464, 0.544] | 0.46 [0.39, 0.53] | 0.373 [0.285, 0.455] | 0.000 | - | 13.64 / 16.34 |
| RRF dense + BM25 + bge | 0.184 [0.137, 0.234] | 0.530 [0.455, 0.609] | 0.082 [0.056, 0.112] | 0.314 [0.279, 0.351] | 0.505 [0.473, 0.537] | 0.54 [0.47, 0.61] | 0.367 [0.306, 0.427] | 0.000 | - | 45.16 / 63.27 |
| HNSW ef 10 (dense, train-filtered) | 0.141 [0.089, 0.198] | 0.418 [0.341, 0.510] | 0.067 [0.038, 0.102] | 0.287 [0.245, 0.333] | 0.479 [0.446, 0.516] | 0.52 [0.46, 0.59] | 0.321 [0.241, 0.398] | 0.000 | 0.977 | 0.84 / 1.31 |
| HNSW ef 40 (dense, train-filtered) | 0.141 [0.089, 0.198] | 0.419 [0.342, 0.511] | 0.067 [0.038, 0.102] | 0.286 [0.244, 0.332] | 0.476 [0.443, 0.513] | 0.53 [0.46, 0.60] | 0.317 [0.239, 0.393] | 0.000 | 0.990 | 1.11 / 1.65 |
| HNSW ef 160 (dense, train-filtered) | 0.142 [0.091, 0.199] | 0.422 [0.346, 0.512] | 0.067 [0.038, 0.102] | 0.286 [0.244, 0.332] | 0.477 [0.444, 0.514] | 0.53 [0.46, 0.60] | 0.317 [0.237, 0.394] | 0.000 | 0.998 | 2.24 / 3.07 |
| deck pre-filter, then dense | 0.076 [0.016, 0.144] | 0.157 [0.035, 0.300] | 0.041 [0.007, 0.080] | 0.286 [0.253, 0.326] | 0.516 [0.494, 0.542] | 0.16 [0.04, 0.29] | 0.475 [0.312, 0.587] | 0.783 | - | 0.82 / 0.87 |

hit@10 / card_hit@10 / MRR: a retrieved state's pro play equals the line's gold play key (or has the gold card); empty = share of lines with no hit at all; recall@20 = overlap of the HNSW top-20 with the exact dense top-20 over the same train pool. HNSW applies the train filter inside the graph search; hnswlib raises ef to k = 20, so ef 10 equals ef 20.

## KG: threat to answer (events: the pro's answer to a threat, ranked over the pro's deck)

Events: 433 threat to answer events of the 27 test games ({'answer_not_in_deck': 3} skipped because the answer is not in the recorded deck). The decision columns use all 766 lines (374 have at least one threat).

| method | hit@1 | hit@3 | MRR | event coverage | 4-way acc | card acc | line coverage | query ms p50 / p95 | decision ms p50 / p95 |
|---|---|---|---|---|---|---|---|---|---|
| KG v1 counter_scores (wiki COUNTERS + mined ANSWERED_BY) | 0.143 [0.122, 0.167] | 0.409 [0.372, 0.447] | 0.360 [0.339, 0.381] | 0.97 [0.94, 0.99] | 0.249 [0.241, 0.257] | 0.499 [0.482, 0.515] | 0.31 [0.26, 0.37] | 0.01 / 0.02 | 0.02 / 0.05 |
| KG v1 wiki COUNTERS only | 0.141 [0.121, 0.163] | 0.406 [0.369, 0.444] | 0.359 [0.339, 0.379] | 0.97 [0.93, 0.99] | 0.250 [0.242, 0.258] | 0.501 [0.485, 0.516] | 0.29 [0.24, 0.34] | 0.01 / 0.01 | 0.02 / 0.04 |
| KG v2 answers_for (full) | 0.154 [0.117, 0.196] | 0.422 [0.372, 0.480] | 0.374 [0.343, 0.410] | 0.99 [0.97, 1.00] | 0.239 [0.229, 0.249] | 0.477 [0.457, 0.498] | 0.45 [0.39, 0.51] | 0.05 / 0.07 | 0.02 / 0.13 |
| KG v2 pro source only | 0.129 [0.096, 0.168] | 0.399 [0.367, 0.438] | 0.350 [0.325, 0.381] | 0.63 [0.50, 0.75] | 0.236 [0.228, 0.244] | 0.472 [0.456, 0.487] | 0.20 [0.15, 0.26] | 0.04 / 0.06 | 0.02 / 0.12 |
| KG v2 role transfer only | 0.145 [0.118, 0.172] | 0.398 [0.366, 0.430] | 0.356 [0.335, 0.376] | 0.67 [0.55, 0.77] | 0.241 [0.231, 0.252] | 0.482 [0.462, 0.505] | 0.24 [0.18, 0.28] | 0.04 / 0.06 | 0.02 / 0.12 |
| KG v2 wiki only | 0.147 [0.123, 0.173] | 0.407 [0.371, 0.442] | 0.362 [0.341, 0.383] | 0.97 [0.93, 0.99] | 0.249 [0.241, 0.256] | 0.498 [0.482, 0.512] | 0.29 [0.24, 0.34] | 0.04 / 0.06 | 0.02 / 0.12 |
| KG v2 simulation (STOPS) only | 0.157 [0.140, 0.176] | 0.440 [0.410, 0.473] | 0.376 [0.359, 0.395] | 0.77 [0.66, 0.86] | 0.258 [0.250, 0.267] | 0.517 [0.501, 0.534] | 0.31 [0.25, 0.37] | 0.04 / 0.06 | 0.02 / 0.12 |
| KGE RotatE (card COUNTERS threat) | 0.125 [0.095, 0.158] | 0.400 [0.352, 0.448] | 0.340 [0.315, 0.367] | 1.00 [1.00, 1.00] | 0.253 [0.239, 0.265] | 0.505 [0.477, 0.530] | 0.49 [0.43, 0.54] | 0.18 / 0.18 | 0.02 / 0.39 |
| KGE TransE (card COUNTERS threat) | 0.118 [0.081, 0.152] | 0.353 [0.318, 0.389] | 0.335 [0.309, 0.360] | 1.00 [1.00, 1.00] | 0.245 [0.231, 0.257] | 0.491 [0.462, 0.515] | 0.49 [0.43, 0.54] | 0.13 / 0.14 | 0.02 / 0.30 |
| train popularity of the answer card | 0.169 [0.124, 0.223] | 0.433 [0.374, 0.492] | 0.384 [0.349, 0.427] | 1.00 [1.00, 1.00] | 0.217 [0.205, 0.231] | 0.435 [0.410, 0.462] | 0.49 [0.43, 0.54] | 0.00 / 0.00 | 0.02 / 0.03 |
| random (expected) | 0.128 [0.125, 0.132] | 0.384 [0.375, 0.395] | 0.345 [0.340, 0.351] | - | 0.250 [0.250, 0.250] | 0.500 [0.500, 0.500] | - | - | - |

Ties are expectations, so a method without evidence scores exactly like random (hit@1 about 1/8 on a deck of 8). Query = one threat scored over the deck; decision = threat extraction plus all threats for one line (392 of 766 lines have no threat, so the decision p50 is mostly the cost of finding none; the p95 is a line with threats).

## Retrieval as a decision model, next to the d1 reference

d1-ft-joint-tac numbers from runs/bench/au/preds/d1-ft-joint-tac.jsonl (Part B).

| method | 4-way acc [95% CI] | card acc | coverage | decision ms p50 / p95 | McNemar p vs d1 | delta vs d1 [CI] |
|---|---|---|---|---|---|---|
| **d1-ft-joint-tac (reference)** | 0.507 [0.473, 0.539] | 0.692 [0.649, 0.730] | 1.00 | 74.86 / 116.92 | - | - |
| KB: dense exact (644-d structured, dot) | 0.284 [0.243, 0.331] | 0.476 [0.442, 0.513] | 0.53 | 1.17 / 1.22 | 4.63e-17 | -0.222 [-0.279, -0.163] |
| KB: BM25 (FTS5 over context_text) | 0.318 [0.283, 0.351] | 0.526 [0.495, 0.556] | 0.53 | 22.15 / 31.98 | 1.07e-16 | -0.189 [-0.235, -0.145] |
| KB: RRF dense + BM25 (live bot) | 0.305 [0.268, 0.347] | 0.495 [0.462, 0.529] | 0.58 | 24.20 / 33.85 | 1.24e-16 | -0.201 [-0.249, -0.152] |
| KB: bge-small text embeddings (cosine) | 0.307 [0.265, 0.351] | 0.504 [0.464, 0.544] | 0.46 | 13.65 / 16.35 | 1.4e-18 | -0.200 [-0.257, -0.142] |
| KB: RRF dense + BM25 + bge | 0.314 [0.279, 0.351] | 0.505 [0.473, 0.537] | 0.54 | 45.17 / 63.29 | 2.28e-13 | -0.193 [-0.241, -0.144] |
| KB: HNSW ef 10 (dense, train-filtered) | 0.287 [0.245, 0.333] | 0.479 [0.446, 0.516] | 0.52 | 0.84 / 1.32 | 3.03e-19 | -0.219 [-0.275, -0.161] |
| KB: HNSW ef 40 (dense, train-filtered) | 0.286 [0.244, 0.332] | 0.476 [0.443, 0.513] | 0.53 | 1.12 / 1.66 | 3.69e-23 | -0.221 [-0.276, -0.163] |
| KB: HNSW ef 160 (dense, train-filtered) | 0.286 [0.244, 0.332] | 0.477 [0.444, 0.514] | 0.53 | 2.25 / 3.08 | 6.34e-20 | -0.221 [-0.278, -0.162] |
| KB: deck pre-filter, then dense | 0.286 [0.253, 0.326] | 0.516 [0.494, 0.542] | 0.16 | 0.82 / 0.87 | 1.03e-17 | -0.221 [-0.266, -0.174] |
| KG: KG v1 counter_scores (wiki COUNTERS + mined ANSWERED_BY) | 0.249 [0.241, 0.257] | 0.499 [0.482, 0.515] | 0.31 | 0.02 / 0.05 | 6.16e-20 | -0.257 [-0.292, -0.222] |
| KG: KG v1 wiki COUNTERS only | 0.250 [0.242, 0.258] | 0.501 [0.485, 0.516] | 0.29 | 0.02 / 0.04 | 4.18e-26 | -0.256 [-0.291, -0.221] |
| KG: KG v2 answers_for (full) | 0.239 [0.229, 0.249] | 0.477 [0.457, 0.498] | 0.45 | 0.02 / 0.13 | 1.79e-21 | -0.268 [-0.301, -0.233] |
| KG: KG v2 pro source only | 0.236 [0.228, 0.244] | 0.472 [0.456, 0.487] | 0.20 | 0.02 / 0.12 | 2.05e-26 | -0.271 [-0.303, -0.236] |
| KG: KG v2 role transfer only | 0.241 [0.231, 0.252] | 0.482 [0.462, 0.505] | 0.24 | 0.02 / 0.12 | 1.19e-30 | -0.265 [-0.299, -0.230] |
| KG: KG v2 wiki only | 0.249 [0.241, 0.256] | 0.498 [0.482, 0.512] | 0.29 | 0.02 / 0.12 | 4.32e-25 | -0.258 [-0.292, -0.223] |
| KG: KG v2 simulation (STOPS) only | 0.258 [0.250, 0.267] | 0.517 [0.501, 0.534] | 0.31 | 0.02 / 0.12 | 1.52e-21 | -0.248 [-0.280, -0.213] |
| KG: KGE RotatE (card COUNTERS threat) | 0.253 [0.239, 0.265] | 0.505 [0.477, 0.530] | 0.49 | 0.02 / 0.39 | 2.47e-26 | -0.254 [-0.292, -0.216] |
| KG: KGE TransE (card COUNTERS threat) | 0.245 [0.231, 0.257] | 0.491 [0.462, 0.515] | 0.49 | 0.02 / 0.30 | 4.77e-26 | -0.261 [-0.295, -0.228] |
| KG: train popularity of the answer card | 0.217 [0.205, 0.231] | 0.435 [0.410, 0.462] | 0.49 | 0.02 / 0.03 | 3.03e-34 | -0.289 [-0.327, -0.251] |
| KG: random (expected) | 0.250 [0.250, 0.250] | 0.500 [0.500, 0.500] | - | - | 5.42e-25 | -0.257 [-0.289, -0.223] |

## Notes

- Only 5 distinct videos stand behind the 27 games, so game-level resampling still understates the uncertainty between videos; read the CIs as optimistic. McNemar is an exact test on lines and ignores the clustering.
- The KB 4-way rule needs the exact (card, lane, zone) key of an option among the 20 hits, so many lines have no supported option (coverage column) and count 0.25: exact hit@10 is far below card_hit@10.
- The deck pre-filter only returns hits when the line's own deck was played in a train game (its 'empty' share is the lines whose deck never appears in train); where it does return, the hits are plays from train games with that deck.
- The d1 options' notes are produced by the same train-only retrieval and KG, so d1 sees these signals in its prompt; the methods here use them without a language model.
- KGE scores are shifted to positives per threat (order preserving) so the tie logic and the event metrics treat them like the other methods; a deck card the graph does not know ranks last.

Runtime of the accuracy pass 65.4 s, latency pass 96.6 s.

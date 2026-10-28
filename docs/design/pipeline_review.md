# Pipeline review and recommended architecture (Opus, 2026-10-28)

## 1. The proposed pipeline (as stated by the user)

1. iPhone Mirroring casts the phone to the Mac; the Mac captures the window.
2. YOLOv8 detector + Kalman tracker produce units, sides, positions, velocities and projections (existing `cr_perception`).
3. A local LLM (Qwen family, 16 GB Mac) is primed at game start with professional context.
4. In the game loop the LLM receives game state, detections and projections and retrieves the most likely next game state from the database.
5. A fine-tuned small decision model (Laya ≈ 421M, or LiquidAI d1-omni-600M) receives 2–3 options proposed by the LLM and picks one.
6. Our code executes the chosen action; the agent itself only has read tools.

## 2. What is right about it

- **Read-only agent + separate executor** (step 6) is the pattern the agents survey recommends: typed tools, validated arguments, independent checks before high-impact actions. Keep it.
- **Retrieve similar states, then decide** is exactly a two-stage recommender (candidate retrieval → ranking). This is the strongest link to the course and should be the spine of the report.
- **A small decision model as the final ranker** is a good fit: both Laya and d1 output probabilities over a fixed answer set, so "choose option A/B/C/D" is a native task for them, and they are fast enough for a real-time loop.
- **Pre-game priming with pro context** maps to RAG/in-context knowledge and to memory (semantic + procedural).

## 3. Problems to fix before building

### 3.1 Latency: a generative LLM in the per-tick loop is the bottleneck

Clash Royale decisions are made every 1–3 s, and defensive answers often have to land within a second or two of a unit crossing the bridge. Elixir regenerates at roughly one per 2.8 s (single), 1.4 s (double) and faster in triple elixir, so a late decision is a wasted decision.

Per-decision cost of the proposed loop (estimates for a 16 GB Apple-silicon Mac; to be benchmarked):

| stage | estimate |
|---|---|
| capture + YOLO + tracking | 30–80 ms |
| LLM prefill of a state prompt (~1–2K tokens; most of it cacheable) | 0.2–1.5 s |
| LLM generating 2–3 options (~60–100 tokens at ~25–60 tok/s for a 3–4B 4-bit model) | 1–4 s |
| decision model (≈0.4–0.6B, one forward pass) | 30–150 ms |
| executor (tap) | < 50 ms |

The generation step alone can exceed the decision window. Two sequential models doubles the risk. **The LLM should not be on the critical path of every decision.**

### 3.2 The LLM should not be the retriever

"The LLM retrieves the most likely next game state" mixes two jobs. Retrieval should be a deterministic index lookup (state → vector/keys → nearest neighbours with their actions and outcomes): it is faster, reproducible, measurable (recall@k, nDCG) and is the subject of three lectures. Letting the LLM choose what to retrieve is the adaptive-RAG idea, which is worth using only as a trigger (when to consult memory/KG), not as the per-tick search engine.

### 3.3 Where the 2–3 options come from

If the LLM writes the options, option quality is bounded by a 3–4B model's game knowledge and by its latency. In the recommender framing the candidates should come from **candidate generators** — the actions pros took in the k most similar states, and counter-cards for the current threats from the knowledge graph — filtered by **hard game rules** (card in hand, elixir ≥ cost, legal placement). The ranker then picks among them. This makes the options grounded, legal by construction, and fast.

### 3.4 Training data for the decision model

- 9,255 labelled decisions (8,555 plays + 700 holds) from one creator, 79% wins. Enough for a LoRA/head fine-tune of a ≤0.6B model on a multiple-choice task; not enough to learn exact tiles.
- The action space must be coarse: card × lane × zone (plus hold), with the exact tile chosen afterwards from the pro's placement distribution for that card. Exact-tile labels are noisy (perception) and too sparse.
- Negatives for the multiple-choice format: other legal cards in hand, the same card in a different zone, and actions taken in retrieved neighbours that differ from the pro's choice (hard negatives, as in the ranking lectures).
- Splits by video to avoid leakage.

### 3.5 Train/serve skew

Training states come from YouTube footage (overlays, creator layout); live states come from iPhone Mirroring. The live state must be produced by the **same code path** that produced the training samples (same schema, same `state_text` template, same feature vector) or retrieval distances and ranker inputs will drift. Perception calibration for the mirrored window is a separate, required task.

### 3.6 Terms of service

Supercell's terms prohibit bots in normal play. For a course project, run the agent in friendly battles or Training Camp and state this in the report's limitations/ethics paragraph.

## 4. Recommended architecture: two-speed retrieve-and-rank

```
                         SLOW LOOP (LLM, Qwen ~3-4B 4-bit, async)
   game start ──► read own deck ─► KG + match memory ─► GAME PLAN (short text + flags)
   every ~10 s / new enemy card revealed ─► update plan  (memory: playbook)
   game end ──► reflection ─► write lessons / new episodes / KB edits (with provenance)
                               │ plan, flags
                               ▼
 FAST LOOP (deterministic + small model, every perception tick)
 iPhone Mirroring ─► capture ─► YOLO + Kalman ─► GameState (same schema as KB)
        ─► decision trigger? (elixir ≥ cheapest card AND (threat ETA < x OR elixir near cap OR periodic))
        ─► candidate generation:
              (a) hybrid kNN over state DB (structured vector + text embedding + BM25, RRF fusion,
                  metadata pre-filter: own deck/archetype, phase)
              (b) KG: counters(threat units) ∩ hand
              (c) "hold"
        ─► symbolic legality filter (hand, elixir, placement rules)
        ─► ranker: Laya / d1 fine-tuned as multiple-choice over ≤4 options (+ plan text)
        ─► executor: (card, lane, zone) → tile (pro placement prior) → screen point (inverse homography) → tap
```

- The user's components all survive: the LLM is primed with pro context, retrieval finds similar states, the small model chooses among 2–3 options, code executes, the agent is read-only.
- The change is **where** the LLM sits: it plans and reflects asynchronously instead of generating options every tick, and the options come from retrieval + KG + rules.
- Optional: if the candidate generators disagree strongly and there is time (no threat within ~3 s), the LLM may be asked to arbitrate via read-only tools. This is the adaptive-retrieval/adaptive-reasoning idea and is a good ablation, not a dependency.

## 5. Models (16 GB Mac)

| role | recommendation | memory (4-bit unless noted) |
|---|---|---|
| slow-loop LLM | Qwen 3–4B instruct (fallback ~1.7B) via MLX or llama.cpp | ≈ 2.5–3 GB |
| ranker | Laya (≈421M, Apache-2.0, ONNX export) or d1-omni-600M (LFM Open License) | ≈ 0.5–1.2 GB |
| text embedder | small sentence embedder (≈ 30–110M params, 384-d) | ≈ 0.1–0.4 GB |
| perception | YOLOv8 (existing) | ≈ 0.1–0.3 GB |

All four fit alongside iPhone Mirroring with headroom. A 7–8B LLM fits in memory but is not recommended while the game runs.

## 6. Baselines the report should include

The recommendation lectures' main lesson is that tuned simple baselines often match complex models. The ranker should be compared against: (1) most-frequent action per phase/hand (popularity), (2) kNN vote over retrieved neighbours, (3) a feature ranker (gradient-boosted trees or a small MLP on the structured vector), (4) the fine-tuned Laya/d1 ranker, with and without the LLM plan text.

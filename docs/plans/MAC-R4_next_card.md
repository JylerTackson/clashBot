# MAC-R4 — opponent next-card prediction: copy/frequency/cycle baselines vs SASRec (spec for a Sonnet implementer)

Author: Opus · 2026-11-02 · Course: KG CyGNet copy mechanism [46], RE-NET [21]; RAG SASRec [61]. Read `docs/plans/MAC-00_common.md` first.

## Owns (only these files)
```
tools/eval/next_card.py     (new)
tests/test_next_card.py     (new)
```

## 1. Sequences (one per game, from the clean dataset)
From `<dataset>/states.jsonl` group records by `game_id` (split from the record). Opponent plays per game: every entry of `state.opponent.recent_plays` and of `outcome.opponent_plays` (when `outcome` is a dict), keep entries whose `card` is playable (`is_playable(card, cards)`) and `t` is not None; dedupe on `(round(t, 1), card)`; sort by `t`; then **collapse perception double-reads**: drop a play if the same card was already kept less than 1.0 s earlier. Result `seq_g = [(t, card), ...]`. Put this in `opponent_sequences(records, cards) -> dict[game_id, {"split", "plays"}]`.

Prediction targets: for every game and every position `i >= 1`, predict `card_i` from `plays[:i]`. Candidate set: `vocab = clashbot.features.vocab()`.

## 2. Methods (each returns a ranked list of all vocab cards)
1. `popularity`: train-split frequency of opponent cards (counts over train sequences), ties by slug.
2. `copy_frequency` (CyGNet intuition): cards seen in `plays[:i]` ranked by count desc, then most recent last play first; then unseen cards by `popularity`.
3. `cycle@w`: Clash Royale cycle rule — a card played in the last 4 plays is back in the queue and cannot be played (8-card deck: 4 in hand, 4 queued). Take the `copy_frequency` order of seen cards, move the cards among the last `w` plays to the end of the seen block (keep their relative order), then unseen by popularity. Evaluate `w ∈ {0, 1, 2, 3, 4}`; choose `w` on val (perception misses plays, so the best `w` may be < 4); report all.
4. `sasrec`: causal transformer over card ids (pad id 0, cards 1..V). Item embedding `d = 64` + learned positional embedding (`max_len = 50`, keep the last 50 plays), 2 blocks, 2 heads, feed-forward 128, dropout 0.2, LayerNorm (pre-norm), causal mask; output = item-embedding-tied logits over the V cards at every position; loss = cross-entropy of the next card at every non-pad position of the train sequences. Adam lr 1e-3, batch 32 sequences, up to 60 epochs, early stopping on val acc@1 (patience 6), seeds `{0, 1, 2}` (report mean ± std). CPU. Prediction for target `i` = logits at position `i-1` of `plays[:i]`.
5. `sasrec+cycle`: SASRec logits with the cards of the last `w*` plays (the `w` chosen for `cycle`) set to `-inf` before ranking.

## 3. Metrics (val and test, every target position)
`acc@1`, `acc@3`, `acc@5`, `MRR` (rank of the true card in the method's list), `n`; and the same split into two buckets: `early` (fewer than 8 distinct opponent cards seen in `plays[:i]`) and `late` (8 or more). Also `oracle_seen` = fraction of targets whose card already appears in `plays[:i]` (upper bound for copy methods).

## 4. CLI and output
```
$PY tools/eval/next_card.py [--dataset DIR] [--methods popularity,copy_frequency,cycle,sasrec,sasrec+cycle]
        [--seeds 0,1,2] [--epochs 60] [--quick] [--out PATH]
```
`--quick`: 3 epochs, seed 0. Output JSON (`--out`, default `<dataset>/eval/next_card.json`): `{"n_games": {split}, "n_targets": {split}, "oracle_seen": {split}, "cycle_w_val": {w: acc@1}, "chosen_w", "methods": {name: {split: metrics (+ buckets) [, "std"]}}}`; `<out>.md` with `| method | val acc@1 | val acc@3 | val MRR | test acc@1 | test acc@3 | test MRR |`; print it.

## 5. Tests — `tests/test_next_card.py`
1. `opponent_sequences`: overlapping `recent_plays` windows of three records produce each play once; a same-card read 0.4 s later is collapsed; a non-playable card is dropped; order by `t`.
2. `copy_frequency` on `[a, b, a, c]` ranks `a` first, then `c` (more recent) before `b`.
3. `cycle@4` on `[a, b, c, d, e]` (all seen once) puts `a` first (the only seen card not in the last 4); `cycle@0` equals `copy_frequency`.
4. SASRec: on a synthetic perfectly periodic sequence set (`[a,b,c,d,e,f,g,h]` repeated) a 30-epoch tiny model (d=16, 1 block) reaches val acc@1 ≥ 0.9; the causal mask means changing a future token does not change earlier logits.
5. Metrics: `MRR` of a list with the truth at rank 3 is 1/3; acc@k boundaries.
6. `--quick` end-to-end on a synthetic `states.jsonl` in `tmp_path` (4 games across train/val/test) writes the JSON with every method.
7. Real data (skip if `dataset/v1/states.jsonl` missing): number of train games equals the train games in `games.jsonl` that have ≥ 2 opponent plays.

## 6. Smoke (allowed once, after the symlinks)
`$PY tools/eval/next_card.py --quick --out runs/smoke/R4/n.json` — paste the table.

## 7. Heavy run (Opus, at integration)
Full run → `docs/data/next_card_v1.{json,md}`.

## 8. Report
Per MAC-00 §5.

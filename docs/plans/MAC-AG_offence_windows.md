# MAC-AG — offence: counter-push and split-lane windows during pressure; never block the win condition as "filler" (spec for one Sonnet implementer)

Author: Opus · 2026-11-06 · From `runs/poc/20261106-135724` (deck D-047, 2 games lost; game 1 took the enemy right tower to 2,088). Read `docs/plans/MAC-00_common.md` first. D-045 holds: the LLM picks every move.

## Evidence
- **Ram Rider was never played.** It sat in the hand for most of both games.
- **Offence windows were suppressed.** The engine proposes offence only when no push has ETA ≤ 8 s (`URGENT_ETA_S`). The Training Camp bot pushes almost nonstop: 29 of 39 decisions were `defend`.
- **The fallback rule blocked the win condition.** At t = 144.8 / 145.9 the fast policy said act (mode `learned`) and the candidates were ram-rider keys, but the decision was `hold_fallback`. Ram Rider has 0 pro plays in the knowledge base, so its candidates are always `fallback`. The live rule (FIX_LIST #18) blocks fallback candidates below 9.5 elixir.
- **Leaking played defence.** At 10 elixir with ram-rider in hand, the bot leaked musketeer at the back. A push existed, so the engine computed defend moves only.

## Owns
```
clashbot/tactics.py   clashbot/policy.py   clashbot/live.py (only the hold_fallback condition)
tests/test_tactics.py   tests/test_policy.py   tests/test_live_loop.py
```
- Another agent (MAC-AF) is adding recording hooks to `clashbot/live.py`. Touch only the `hold_fallback` condition there.
- Pin engine weights in tests (`DEFAULT_WEIGHTS` or the tuned file explicitly).

## 1. Engine (`tactics.py`): offence moves during pressure
- Compute `_offence_moves` also when pushes exist, in these windows:
  1. **Counter-push.** Our defenders are winning the lane: ally units (any role except building) at row ≥ 9 in lane L, and the push in L is answered or gone. The existing `counter_push` intent applies; drop the "no push ≤ 8 s" gate for it.
  2. **Split lane** (new intent `split`). The opponent committed ≥ 5 elixir in a push in lane P that is `answered`. If our elixir ≥ win-condition cost + 1, put the win condition at the bridge of the other lane. Score `w.split` (default 0.7). Reason `split: they committed {X} elixir {P}`.
  3. **Leak with win condition first.** At elixir ≥ 9.5, propose the win condition at the bridge (`leak`, `w.leak_wc`), even when pushes exist, as long as no push has ETA ≤ 3 s. The defend moves for urgent pushes still score higher when their window is open.
- Win-condition tiles are as today: bridge of the lane; spell win conditions on the princess tower.
- `split` and the new windows go into `report.moves`, sorted by score as usual. Their lines read `tactics: best ram-rider @[x,y] split: …` when they are best.

## 2. Policy (`policy.py`)
- New mode **`counter`** (an act). Consider it after the per-push defend rules, in this order:
  - if every push is `answered` or has ETA > 5 s (no defend act needed now);
  - and the engine's best offence move (intent in `counter_push`, `split`, `punish`, `leak`) has score ≥ `counter_min_score` (0.5);
  - and own elixir ≥ that card's cost + `reserve`, where `reserve` = the cost of the cheapest adequate answer in hand to the most urgent unanswered push, or 0 if there is none;
  - then act with mode `counter`, `allowed_cards = {that card}`, reason `counter: {intent} {card} {lane}`.
- **Leak:** when the engine's best move is a win-condition `leak`, the leak decision uses `allowed_cards = {win condition}`.
- `LearnedActPolicy` and `FastActPolicy` pass `counter` through like `defend` / `leak`. Check `clashbot/act_model.py` delegation: no edit should be needed. If one is, report it.

## 3. Live (`live.py`, only the hold_fallback condition)
A `fallback`-only candidate is **not** held if:
- the decision mode is `counter`, `leak`, `punish`, `learned` or `defend`;
- **and** the candidate's card has role `win_condition`, or its card is in `decision.allowed_cards`.

The fallback hold keeps protecting against random filler plays only.

## Tests
1. A giant push in the right lane answered by our tesla, ally knight at row 10 in the left lane, elixir 7, ram-rider in hand → `counter` with ram-rider at the left bridge (`counter_push` or `split`).
2. The same with elixir 5, and an unanswered balloon in the left lane with ETA 4 → defend wins (the reserve rule).
3. Elixir 10, a push with ETA 6 s, ram-rider in hand → leak with ram-rider at the bridge. Elixir 10 with a push at ETA 2 s → defend first.
4. Live: a `learned` / `counter` decision with ram-rider fallback candidates → the executor is called (no `hold_fallback`). A non-win-condition fallback in `learned` mode is still held.
5. Existing MAC-AD situations 1–7 are unchanged.
6. Offline: run `tools/eval/eval_act_policy.py --out runs/smoke/AG/act_policy.json` and report the `counter` count.

Run the full suite (`--deselect tests/test_perception_latency.py`).

## Report
Per MAC-00 §5.

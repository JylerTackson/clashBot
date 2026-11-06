# MAC-AB — fast play/wait model, engine offence, phantom opponent cards, lookup-vs-tactics arbitration (specs for four Sonnet implementers)

Author: Opus · 2026-11-05 · User picked all four (2026-11-05 evening). Read `docs/plans/MAC-00_common.md` first.

- The four packages run in parallel: **A1**, **A2**, **A3**, **A4**.
- **None of them may use the GPU.** The user is playing live matches; use CPU only and never load d1. LightGBM needs `n_jobs=1` (libomp, see `requirements-mac.txt`).
- Never touch the screen or send input.
- Two packages touch `clashbot/live.py`, in disjoint places:
  - **A1:** only `_build_policy` and the `--policy` argument (add the choice `fast`).
  - **A4:** only the ranking call site in `LiveLoop.step` and a new `--arbiter` argument, added at the **end** of `build_parser`.

---
## A1 — fast play/wait model on structured features

Context: `docs/data/act_head_v1.md`. The d1 act head reaches 0.558 balanced accuracy on test vs the rules' 0.535; its training loss stayed at chance.

### Owns
```
clashbot/act_model.py          (new)   tools/train/train_act_gbdt.py (new)   tests/test_act_model.py (new)
clashbot/live.py               (only _build_policy and the --policy choices)
```

### Features — `act_features(state, t) -> list[float]` and `FEATURES: list[str]` (fixed order)
Use the clean state dict (the live builder output, also stored in `dataset/v1/act_dense/states_<split>.jsonl` as `state` with `t`) and nothing else, so the live loop and training compute the same thing.

| group | features |
|---|---|
| elixir | own elixir; opponent estimate (−1 if None); own − opponent (0 if None); full flag (own ≥ 9.5) |
| time | phase index (single 0, double 1, triple 2, overtime 3, unknown −1); `match_seconds` (−1 if None) |
| enemy units | count; count on our half (row ≤ 14); min ETA to an own tower (cap 20, 20 if none); elixir on the field (distinct known cards); elixir on our half |
| ally units | count; count on the enemy half (row ≥ 17); elixir on the field |
| plays | seconds since the opponent's newest play (cap 20); its cost (0 if none); opponent elixir spent in the last 5 s; seconds since our newest play (cap 20); our spend in the last 5 s |
| hand | known slots; min / mean cost of known cards (−1 if none); affordable known cards; next-card cost (−1) |
| towers | own princess min HP / 5000 (−1 unknown); enemy princess min HP / 5000; own towers down; enemy towers down |
| threats | `len(state["threats"])` |

- Costs come from `load_cards()[card]["elixir_cost"]`.
- Play times come from `recent_plays[i]["t"]`.

### Labels
- For the lines of `dataset/v1/act_dense/{train,val,test}.jsonl` (the **full** dense data, about 40 K lines), compute `act_h` for horizons `h ∈ {1.0, 2.0, 3.0}` s.
- Use `meta.next_deploy_in`: `act_h = next_deploy_in is not None and next_deploy_in <= h`.
  - First check the semantics in `tools/data/build_dense_act.py`.
  - `act_1` must equal the line's `answers.act`; assert this on the data.
- Join each line with its sidecar line by `id` to get the state.

### `tools/train/train_act_gbdt.py`
```
$PY tools/train/train_act_gbdt.py [--data dataset/v1/act_dense] [--out db/models/act-gbdt] [--horizons 1,2,3] [--seed 0]
```
- **Model:** one LightGBM binary model per horizon.
  - Parameters: `n_jobs=1`, `is_unbalance=True`, `num_leaves=31`, `learning_rate=0.05`, up to 2,000 rounds with early stopping (50) on val logloss, and `min_data_in_leaf=50`.
- **Thresholds:** on val, pick the threshold with the best balanced accuracy, sweeping 0.05–0.95 in steps of 0.01 (ties → closest to 0.5).
- **Metrics,** per horizon and split, on the full dense val/test: AUC, balanced accuracy, act rate vs the pro act rate, act / wait precision and recall.
  - Report the same for the **rules**: `ReactivePolicy(cards).decide(state, t).act` on the sidecar states.
  - Report the **reaction slice** (`meta.since_opp_play <= 3`): act rate and act recall.
- **Apples-to-apples with the d1 head (h = 1):** also report the metrics on the ids of `dataset/v1/act_dense_bal/{val,test}.jsonl`, next to the d1-head numbers in `docs/data/act_head_v1.json`.
- **Feature importance:** gain, top 15 per horizon.
- **Outputs:**
  - `db/models/act-gbdt/model_h{h}.txt`;
  - `meta.json`: features, horizons, per-horizon threshold, metrics, and `default_horizon` = 2.0;
  - `dataset/v1/eval/act_gbdt.{json,md}`.
- Print the tables.
- **Speed:** time `predict` on one feature row; it must be < 1 ms.

### `clashbot/act_model.py`
- `class ActGBDT`: `load(dir, horizon=None)` and `p_act(state, t) -> float`.
- `class FastActPolicy`: same `decide` / `filter_candidates` / `target_tile` contract and safety rules as `LearnedActPolicy` (see `clashbot/policy.py`).
  - The rules decide `none` / `defend` / `leak`.
  - Otherwise act iff `p_act ≥ threshold`, with mode `learned` (or the rule's mode and lane when the rules want to punish).
  - The reason is `fast act p=0.xx h=2s` / `fast wait p=...`.
  - No caching is needed.
- **live:** `--policy fast --act-model db/models/act-gbdt [--act-threshold X]`. The threshold defaults to `meta.json` for the default horizon. `--act-model` is required, as for `learned`.

### Tests
- Features:
  - the order and length match `FEATURES`;
  - a hand-made state gives the hand-computed values (ETA cap, distinct-card elixir, since-play caps);
  - None values map to −1.
- Labels:
  - `act_1 == answers.act` on a synthetic dense dir;
  - the horizons are nested (`act_1 ⇒ act_2 ⇒ act_3`).
- The trainer on a tiny synthetic dir (a separable toy feature) writes the models and `meta.json`, with a threshold chosen on val.
- `FastActPolicy`:
  - defend and leak pass through;
  - p ≥ threshold → act;
  - p < threshold → hold.
- Live CLI:
  - `--policy fast` without `--act-model` → exit 2;
  - the threshold is read from `meta.json` (with a monkeypatched model).

### Heavy run (CPU, allowed — minutes)
Run the trainer on the real dense data. Paste the tables (all horizons, rules, d1-head comparison, feature importance).

---
## A2 — tactical engine: offence for the bait deck, and weights tuned on pro data

Context: `docs/data/tactics_eval_v1.md`. With no push (57 % of pro plays) the engine proposes a move in only 11–16 % of moments. The user's deck (D-044) is berserker, bomb-tower, fire-spirit, goblin-barrel, ice-spirit, magic-archer, royal-delivery, wall-breakers.

### Owns
```
clashbot/tactics.py   clashbot/resources/tactics_weights.json (new)   tools/train/tune_tactics.py (new)
tools/eval/eval_tactics.py   tests/test_tactics.py   tests/test_tactics_integration.py (only if needed)
```
Keep the public API: the dataclasses, `analyse`, `score_key`, and `TacticalEngine(cards, roles, kg)`. Add `weights=None` as a new keyword.

### 1. Weights
- Move **every** hand-set number of the scoring (base, need bonuses and penalties, KG factor, trade terms, win-condition penalty, offence intent scores) into `DEFAULT_WEIGHTS: dict[str, float]` with descriptive keys.
- `TacticalEngine(weights=None)` uses the defaults, updated from `clashbot/resources/tactics_weights.json` if that file exists. An explicit dict overrides both.
- With the defaults, the behaviour must be identical to today; existing tests pass.

### 2. New offence intents (no push with ETA ≤ 8 s)
Use the opponent's known deck (`state.opponent.deck_known`), their `recent_plays` and the roles. Each move gets a reason ≤ 60 characters.
- **`bait`:** a spell win condition (goblin-barrel; also graveyard, miner and goblin-drill if present) on the enemy princess tower of lane L. Conditions:
  - own elixir ≥ cost + 1;
  - the opponent's known small spells that answer it are counted: role `spell_small`, or a spell whose `kills` include `swarm`, e.g. the-log, zap, arrows, barbarian-barrel, giant-snowball;
  - if one of those small spells was played in the last 8 s, score `w.bait_counter_used` (default 0.6), with reason "their {spell} is cycling";
  - if none is known, score `w.bait_unknown` (0.45);
  - otherwise score `w.bait_counter_ready` (0.25).
  - L = the lane with the lower enemy princess HP, else the lane opposite the opponent's newest play, else left.
- **`pressure`:** a troop win condition (wall-breakers, hog-rider, ram-rider, battle-ram, ...) at the bridge of lane L, with own elixir ≥ cost + 3 (a defence reserve) and opponent estimate ≤ 5 or unknown. Score `w.pressure` (0.45).
- **`support`:** when an ally unit with role `win_condition` or `tank` is on the enemy half (row ≥ 17) or at the bridge in lane L, and own elixir ≥ cost + 1:
  - a ranged troop 2 tiles behind it: `(ally.x, ally.y − 2)`, clamped to our half if needed;
  - or a small damage spell whose `spell_target` hits the units defending against it.
  - Score `w.support` (0.5).
- **`cycle`:** own elixir ≥ 7, no other move scored ≥ 0.4, and the cheapest non-spell, non-win-condition card in hand costs ≤ 2. Score `w.cycle` (0.35).
  - kamikaze → the bridge of the lane with the lower enemy princess HP (chip damage);
  - other → back, `(tower.x + 1.5 × toward_centre, 1.5)` of the lane away from enemy units.
- Keep `leak`, `punish` and `counter_push`. Every intent name appears in `TacticalMove.intent`.

### 3. `tools/train/tune_tactics.py` (CPU)
```
$PY tools/train/tune_tactics.py [--dataset DIR] [--passes 3] [--out clashbot/resources/tactics_weights.json]
```
- **Objective,** on `dataset/v1/states.jsonl` train play records whose pro card is in the affordable known pool (else skip):
  - mean of `[best.card == pro] + 0.5 × [pro card among moves[:3] cards] + 0.25 × [lane_of(best.tile) == pro lane, when the card matches]`;
  - computed with one `analyse` per record.
- **Search:** coordinate search. For each weight in turn, try ×{0.5, 0.75, 1.25, 1.5, 2.0} (and sign-preserving ±0.1 for weights near 0), keep the best, and repeat for `--passes` passes.
- **Report** train / val / test before vs after: objective, top-1, top-3, coverage, by push / no push, by intent. Use the same random-pool baseline as `docs/data/tactics_eval_v1.md` (random top-1 ≈ 0.17).
- Write the weights JSON (committed) and `dataset/v1/eval/tactics_tuning.{json,md}`.
- **Guard against overfitting:** keep the tuned weights only if val improves; otherwise write the defaults and say so.

### 4. `eval_tactics.py`
Add a per-intent breakdown, and `--weights PATH|default`.

### Tests
- One test per new intent:
  - bait after their log was played 3 s ago;
  - bait not preferred when their zap is ready (score 0.25);
  - pressure needs the reserve;
  - support behind an ally hog on the enemy half;
  - cycle with spirits.
- The weights file is loaded; an explicit weights dict overrides it.
- The tuner on a tiny synthetic record set improves its objective, and keeps the defaults when val does not improve.

### Heavy run (CPU, allowed)
Run the tuner on the real data, then `eval_tactics.py`. Paste the tables before and after.

---
## A3 — phantom opponent cards (FIX_LIST #31)

Context: MAC-PF report §8 (`runs/smoke/PF`). After MAC-PF, two arena plays (0.5 + 0.5) still admit a phantom card:
- `skeletons`, with no spawner;
- `barbarian-barrel` at [3,13], which was our own goblin barrel misread;
- `tornado` on our half.

The opponent's elixir is still over-spent: 48 spent vs 37 available.

### Owns
```
cr_perception/events.py   cr_perception/decktracker.py   tests/test_perception_fixes.py (append a section)
```

### Fixes (`PlayDetector`)
1. **Our half needs a label.** An opponent arena play whose first tile is on our half (row ≤ 14) is held as *pending*. It is emitted only if a deploy label of the same card (score ≥ 0.85) arrives within ±2 s; a label that arrives first also confirms it. Otherwise it is dropped and counted in `stats["dropped_own_half_unconfirmed"]`. With confirmation, emit one event (not two).
2. **Coincidence with our own plays.** Drop an opponent play (arena or label) that falls within ±1.0 s and 3 tiles of one of our own plays (HUD or label), and count it.
   - **Spells:** additionally drop an opponent **spell** play that falls within 2.0 s after one of our own spell plays, anywhere on our half or the river. This covers a goblin barrel in flight being read as a barbarian barrel.
3. **Unexplained spawn-type units.** `skeletons`, `goblins` and `bats` seen by the arena path with no spawner explanation need a label (as in 1) when they first appear on the enemy half within 4 tiles of an enemy tower (tower troops and spawners hide there). Otherwise they keep the current rule.
4. Expose the counters in the perception stats, so the smoke can show them.

### Tests
One synthetic test per rule:
- an own-half arena play without a label → no event;
- an own-half arena play with a label at +1 s → exactly one event;
- an opponent barbarian-barrel 1 s after our goblin-barrel → dropped;
- an opponent play 0.5 s after ours at 2 tiles → dropped;
- the same at 5 tiles → kept;
- skeletons near the enemy king without a label → dropped.

Existing MAC-PF tests must still pass.

### Smoke (CPU only)
The same clip and commands as MAC-PF §5 (video `B7SW-i94v8s`, m0.3, 752–842 s, `--device cpu`), before vs after. Report:
- opponent plays/min;
- spent vs available elixir;
- the final `deck_known` vs the true opponent deck (archers, electro-spirit, elixir-collector, fisherman, knight, skeleton-dragons, the-log, x-bow);
- the phantom cards that remain;
- the new counters.

---
## A4 — lookup-vs-tactics arbitration (brief R7)

### Owns
```
clashbot/arbiter.py (new)   tools/eval/eval_arbiter.py (new)   tests/test_arbiter.py (new)
clashbot/live.py (only the ranking call site in LiveLoop.step and a new --arbiter argument at the end of build_parser)
```

### `clashbot/arbiter.py`
```python
@dataclass
class ArbiterConfig:
    conf: float = 0.55        # d1 top probability at which d1 is trusted outright
    urgent_eta_s: float = 4.0 # a push this close -> prefer the tactical answer when d1 is unsure
    knn_min: float = 0.5      # kNN support at which "pros played this" is preferred when d1 is unsure

class RuleArbiter:
    def __init__(self, cfg: ArbiterConfig | None = None): ...
    def choose(self, cands: list, probs: list[float], urgent: bool) -> tuple[int, str]:
        """cands: Candidate-like (key, sources dict). Returns (index, reason)."""
```
Rules, in order:
1. `max(probs) ≥ conf` → argmax, reason "d1 confident".
2. `urgent` and a candidate has `tac` → the highest `tac`, reason "urgent push: tactics".
3. A candidate has `knn ≥ knn_min` → the highest `knn`, reason "pros played this".
4. Otherwise → argmax, reason "d1".

Ties go to the first. With empty probs, d1 counts as unsure and the argmax is index 0.
- `urgent_from_report(report, urgent_eta_s)` → True if any push has `eta_s ≤ urgent_eta_s`.

### Live (`clashbot/live.py`)
- `--arbiter none|rules`, default `none`; plus `--arbiter-conf`, `--arbiter-knn` and `--arbiter-eta`.
- With `rules`:
  - after `rank_keys` returns probs, `i, why = arbiter.choose(cands, probs, urgent)`;
  - `urgent` comes from the tactics report (False without one);
  - log `arbiter_reason` per decision.
- With `--ranker tactics`, the arbiter is not used.

### `tools/eval/eval_arbiter.py` (offline, CPU, no model)
```
$PY tools/eval/eval_arbiter.py [--dataset DIR] [--d1-eval dataset/v1/eval/d1_eval_transformers_d1-ft-joint.json]
        [--lines dataset/v1/d1/joint_tac] [--out dataset/v1/eval/arbiter.json]
```
- **Inputs.** For each val/test line of `joint_tac`:
  - **d1 probs:** from the eval JSON's `per_example` (match on `id` and `split`; the probs are in the line's criteria order, so check the key order against the `joint` line);
  - **sources, from the criteria notes:**
    - `" — tactics: "` → `tac` = 1.0;
    - `" — pros played this in similar states (support X)"` → `knn` = X;
    - `"known counter"` → `kg` = 1;
    - `"; pros played this too"` → also `knn` = 0.5, unless a value was given.
  - **urgent:** from the state's `tactics: push … tower in Xs` lines (X ≤ eta).
- **Report** accuracy on val and test for:
  - d1 alone;
  - tactics-first (the highest `tac`, else d1);
  - knn-first;
  - the arbiter, with a grid over `conf ∈ {0.3, 0.35, …, 0.8}`, `knn_min ∈ {0.2, 0.4, 0.6, 0.8, 1.0}` and `urgent_eta_s ∈ {2, 4, 6}`, **chosen on val** and then applied to test.
- **Report** how often each rule fired, and its accuracy when it fired.
- `--d1-eval` lets the same script score the tactics-trained model after the overnight run (`d1_eval_transformers_d1-ft-joint-tac.json`).
- Write JSON and `.md`; print the tables.

### Tests
- Each rule fires in its situation, and the order is respected.
- Ties go to the first.
- Empty probs are handled.
- Note parsing extracts `tac`, `knn` and `kg`.
- The eval script on a tiny synthetic line set and eval JSON: the grid is chosen on val, and test is reported.
- Live: `--arbiter rules` uses the arbiter's pick and logs the reason; `--arbiter none` changes nothing.

### Heavy run (CPU)
Run `eval_arbiter.py` on the real files. Paste the tables.

---
## Report
Per MAC-00 §5, per package: files, tests (counts), heavy-run tables, every changed existing test (with the reason), ambiguities, deviations.

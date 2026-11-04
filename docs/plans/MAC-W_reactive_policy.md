# MAC-W — reactive play/hold policy: save elixir, defend on time, punish overspends (spec for a Sonnet implementer)

Author: Opus · 2026-11-03 · Implements D-041 (user: "it doesn't always have to place a card … save elixir and react to what the opponent is doing"). Read `docs/plans/MAC-00_common.md` first.

## Why
The live loop calls `clashbot.loop.should_decide`, which fires every 3 s regardless of the situation, on elixir ≥ 9 and after any opponent play; every trigger then plays a card. In two live matches the bot always played and lost quickly.

## Owns (only these files)
```
clashbot/policy.py              (new)
tools/eval/eval_act_policy.py   (new)
tests/test_policy.py            (new)
clashbot/live.py                (wire the policy in; keep everything else)
tests/test_live_loop.py         (update)
```
Do not change `clashbot/loop.py` (`should_decide` stays for replay and as the `--policy trigger` fallback).

## 1. `clashbot/policy.py`
```python
DEFEND_ETA_S = 4.0          # an enemy unit this close (seconds) to one of our towers must be answered now
LEAK_ELIXIR = 9.5           # at (near) full elixir, waiting wastes elixir
PUNISH_OPP_ELIXIR = 3.0     # opponent estimated this low after spending -> push the other lane
PUNISH_OWN_ELIXIR = 6.0
OWN_HALF_MAX_ROW = 15       # rows 0..15 are our side of the river

@dataclass
class ActDecision:
    act: bool
    mode: str                 # "defend" | "leak" | "punish" | "hold" | "none"
    lane: str | None          # lane to play in (None = any)
    reason: str               # short human-readable reason, logged
    threat: dict | None = None   # the enemy unit being answered (defend)

class ReactivePolicy:
    def __init__(self, cards: dict, defend_eta_s=DEFEND_ETA_S, leak_elixir=LEAK_ELIXIR,
                 punish_opp_elixir=PUNISH_OPP_ELIXIR, punish_own_elixir=PUNISH_OWN_ELIXIR): ...
    def decide(self, state: dict, t: float) -> ActDecision
    def filter_candidates(self, candidates: list, decision: ActDecision, state: dict) -> list
```
`decide` — first rule that applies wins:
0. `legal_cards(state, cards)` empty → `ActDecision(False, "none", None, "no affordable card")`.
1. **defend**: threats = enemy units (`side == "enemy"`) with `eta_tower.tower` starting `"own"` and `eta_tower.s <= defend_eta_s`, **or** with a tile on our half (`tile[1] <= OWN_HALF_MAX_ROW`). Most urgent = smallest ETA (units without ETA but on our half count as ETA 0 if `tile[1] <= 8`, else 2.0). → `act=True, mode="defend", lane=lane_of(threat tile), reason=f"defend {card or unit_class} {lane} eta {s}s", threat=unit`. A fresh opponent play whose unit is still on their side does **not** trigger: we wait until it becomes a threat (that is the "react at the right moment" behaviour).
2. **leak**: own elixir ≥ `leak_elixir` → `act=True, mode="leak"`; `lane` = the lane with no enemy units if exactly one lane has enemy units (play away from them), else `None`.
3. **punish**: opponent `elixir_estimate` is not None and ≤ `punish_opp_elixir`, own elixir ≥ `punish_own_elixir`, and no threat → `act=True, mode="punish"`, `lane` = the lane opposite (left↔right; middle → None) of the opponent's most recent play with a tile, else `None`.
4. otherwise → `ActDecision(False, "hold", None, "save elixir: no threat, not full, opponent not overspent")`.

`filter_candidates` (keys are `card|lane|zone`; card types from `cards`):
- Always drop **spell** candidates unless an enemy unit lies in the same lane **and** zone (`lane_of/zone_of` of the unit tile) — no blind spells.
- **defend**: keep candidates in `decision.lane` with zone in `{back, mid, bridge}` (spells: any zone, but the rule above applies). If none survive: keep candidates whose `sources.get("kg", 0) > 0` (KG counters of the threat) regardless of lane; if still none, return the spell-filtered list.
- **leak**: keep zone `back` or `bridge` (and `decision.lane` if set); if none survive, return the spell-filtered list.
- **punish**: keep non-spell candidates in `decision.lane` (any lane if None) with zone `bridge` or `enemy_half`; if none survive return `[]` (the loop then holds).
- Never reorder: keep the generator's order among survivors.

## 2. `clashbot/live.py`
- `LiveLoop.__init__(..., policy=None)`: `None` → `ReactivePolicy(cards)`; also accept the string `"trigger"` → keep today's `should_decide` path (old behaviour).
- In `step`, replace the `should_decide(...)` call (after the min-gap check) with `decision = self.policy.decide(state, t)` (reactive) — the opponent-play filtering of `_fresh_opponent_state` is not needed for the reactive policy; keep it for `"trigger"`. If `not decision.act`: count `self.policy_holds[decision.reason_key]` where `reason_key = decision.mode` and return `None`.
- After `generator.generate(...)`: `cands = self.policy.filter_candidates(cands, decision, state)`; if empty → record a decision with key `"hold"`, executor entry `{"t", "key": "hold", "tile": None, "status": "hold_policy", "clicks": [], "sent": False}` (do not call the executor), set `last_decision_t`.
- `LiveDecision` and the `decisions.jsonl` line gain `"mode"` and `"reason"`; `summary()` gains `"policy_holds"` (Counter by mode) and `"decisions_by_mode"`.
- CLI `--policy reactive|trigger` (default `reactive`).
- Keep min gap, fallback hold (#18), `rank_keys`, `update_rect`, hand reuse, everything else.

## 3. `tools/eval/eval_act_policy.py` (offline comparison)
```
$PY tools/eval/eval_act_policy.py [--dataset DIR] [--splits val,test] [--out PATH]
```
For every record of the split (all kinds): label = 1 if `action.type == "play"` else 0 (`unknown` skipped). Predictions: `reactive` = `ReactivePolicy(cards).decide(rec["state"], rec["t"]).act`; `trigger` = `should_decide(rec["state"], rec["t"], None, cards)`; `always` = 1. Per policy and split: accuracy, balanced accuracy, act rate vs pro play rate, confusion matrix; reactive also counts by mode. Report the same on `kind == "periodic"` records only (closest to random moments). JSON + `.md` to `<dataset>/eval/act_policy.json` (default); print the table. (The labels are biased toward plays — D-020 — so this is a sanity check; the dense data of MAC-V is the real test.)

## 4. Tests
`tests/test_policy.py` (hand-made states like `tests/test_runtime.py::_state`, real `load_cards()`):
1. No legal cards → act False, mode none.
2. Enemy unit with ETA 3 s to `own_left` → defend, lane left; ETA 6 s and on the enemy half → hold.
3. Enemy unit at tile row 10 without ETA → defend.
4. Elixir 10, no threats → leak; enemy units only in the right lane → leak lane left.
5. Opponent estimate 2.5, own elixir 7, last opponent play at a right-lane tile → punish lane left; own elixir 4 → hold.
6. Fresh opponent play, unit still at row 25 → hold (wait for it).
7. `filter_candidates`: spells without an enemy unit in their lane+zone are dropped; defend keeps the threat lane; punish keeps only bridge/enemy_half troops and returns `[]` when none; order preserved.
`tests/test_live_loop.py`: with fakes, a hold decision returns `None` and is counted in `policy_holds`; a defend decision with all candidates filtered out yields a `hold_policy` record without calling the executor; `--policy trigger` keeps the old behaviour (existing trigger tests pass with `policy="trigger"`); decisions carry `mode`/`reason`.

## 5. Smoke (allowed once; offline only — never touch the screen)
`$PY tools/eval/eval_act_policy.py --out runs/smoke/W/act_policy.json` — paste the table.

## 6. Report
Per MAC-00 §5.

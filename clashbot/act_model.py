"""Fast play/wait model on structured features (MAC-AB A1): a LightGBM head instead of the d1 act-head.

  `act_features(state, t)` turns the clean live state dict (the builder output, also stored in
  `dataset/v1/act_dense/states_<split>.jsonl`) into the fixed-order `FEATURES` vector, the same way for training and for
  the live loop. `ActGBDT.load(dir)` reads `meta.json` + `model_h<h>.txt` written by `tools/train/train_act_gbdt.py`
  and answers `p_act(state, t)` = P(the pro deploys a card within `horizon` seconds), in well under a millisecond.
  `FastActPolicy` is `LearnedActPolicy` with that model: the rules keep the safety modes (none / defend / leak), the
  model decides every other moment (`p >= threshold` plays, else hold). LightGBM runs with one thread (libomp next to
  torch) and is imported only when a model is loaded.
"""
from __future__ import annotations

import json
from pathlib import Path

from clashbot.game_rules import load_cards
from clashbot.policy import ActDecision, ReactivePolicy

OWN_HALF_ROW = 14           # enemy units on rows <= this stand on our half
ENEMY_HALF_ROW = 17         # ally units on rows >= this stand on the enemy half
ETA_CAP_S = 20.0            # eta to an own tower / seconds since a play are capped here (also the "none" value)
SPEND_WINDOW_S = 5.0
FULL_ELIXIR = 9.5
TOWER_HP_SCALE = 5000.0
PHASES = {"single_elixir": 0, "double_elixir": 1, "triple_elixir": 2, "overtime": 3}
DEFAULT_HORIZON = 2.0
MODEL_FILE = "model_h{h:g}.txt"
META_FILE = "meta.json"

FEATURES: list[str] = [
    # elixir
    "own_elixir", "opp_elixir_est", "elixir_diff", "own_full",
    # time
    "phase_idx", "match_seconds",
    # enemy units
    "enemy_count", "enemy_count_own_half", "enemy_min_eta_own", "enemy_elixir_field", "enemy_elixir_own_half",
    # ally units
    "ally_count", "ally_count_enemy_half", "ally_elixir_field",
    # plays
    "since_opp_play", "opp_last_cost", "opp_spent_5s", "since_own_play", "own_spent_5s",
    # hand
    "hand_known", "hand_cost_min", "hand_cost_mean", "hand_affordable", "next_card_cost",
    # towers
    "own_princess_hp_min", "opp_princess_hp_min", "own_towers_down", "opp_towers_down",
    # threats
    "n_threats",
]

_CARDS: dict | None = None


def _cards() -> dict:
    global _CARDS
    if _CARDS is None:
        _CARDS = load_cards()
    return _CARDS


def _num(x) -> float | None:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    return float(x)


def _cost(card, cards: dict) -> float:
    """Elixir cost of `card` (0.0 when unknown or variable)."""
    c = _num((cards.get(card) or {}).get("elixir_cost")) if isinstance(card, str) else None
    return c if c is not None else 0.0


def _row(unit: dict) -> float | None:
    tile = unit.get("tile")
    if isinstance(tile, (list, tuple)) and len(tile) >= 2:
        return _num(tile[1])
    return None


def _own_eta(unit: dict) -> float | None:
    eta = unit.get("eta_tower")
    if isinstance(eta, dict) and str(eta.get("tower", "")).startswith("own"):
        return _num(eta.get("s"))
    return None


def _field_elixir(units: list[dict], cards: dict) -> float:
    """Elixir of the distinct known cards among `units` (a card seen twice counts once, as in `spell_target`)."""
    known = {u.get("card") for u in units if _cost(u.get("card"), cards) > 0.0}
    return float(sum(_cost(c, cards) for c in known))


def _plays(plays, now: float, cards: dict) -> tuple[float, float, float]:
    """(seconds since the newest play, its cost, elixir spent in the last SPEND_WINDOW_S) of one side's plays."""
    newest: dict | None = None
    spent = 0.0
    for p in plays or []:
        tp = _num(p.get("t"))
        if tp is None:
            continue
        if newest is None or tp >= newest["t"]:
            newest = p
        if 0.0 <= now - tp <= SPEND_WINDOW_S:
            spent += _cost(p.get("card"), cards)
    if newest is None:
        return ETA_CAP_S, 0.0, spent
    since = min(max(now - float(newest["t"]), 0.0), ETA_CAP_S)
    return since, _cost(newest.get("card"), cards), spent


def _princess(towers: dict | None, hud_up: bool) -> tuple[float, float]:
    """(min princess HP / TOWER_HP_SCALE or -1 when unread, princess towers down). A princess tower that the HUD no
    longer shows (HP None while the clock is up) or at 0 HP counts as down."""
    towers = towers or {}
    hps = [_num(towers.get(side)) for side in ("left", "right")]
    known = [h for h in hps if h is not None]
    down = sum(1 for h in hps if (h is None and hud_up) or (h is not None and h <= 0.0))
    return (min(known) / TOWER_HP_SCALE if known else -1.0), float(down)


def act_features(state: dict, t: float, cards: dict | None = None) -> list[float]:
    """The `FEATURES` vector of one clean state at video time `t` (unknown values are -1, see the groups below)."""
    cards = cards if cards is not None else _cards()
    own, opp = state.get("own") or {}, state.get("opponent") or {}
    units = state.get("units") or []
    enemy = [u for u in units if u.get("side") == "enemy"]
    ally = [u for u in units if u.get("side") == "ally"]

    # elixir
    own_e, opp_e = _num(own.get("elixir")), _num(opp.get("elixir_estimate"))
    diff = own_e - opp_e if own_e is not None and opp_e is not None else 0.0
    f: list[float] = [-1.0 if own_e is None else own_e, -1.0 if opp_e is None else opp_e, diff,
                      float(own_e is not None and own_e >= FULL_ELIXIR)]

    # time
    ms = _num(state.get("match_seconds"))
    f += [float(PHASES.get(state.get("phase"), -1)), -1.0 if ms is None else ms]

    # enemy units
    own_half = [u for u in enemy if (r := _row(u)) is not None and r <= OWN_HALF_ROW]
    etas = [e for e in (_own_eta(u) for u in enemy) if e is not None]
    f += [float(len(enemy)), float(len(own_half)), min(min(etas), ETA_CAP_S) if etas else ETA_CAP_S,
          _field_elixir(enemy, cards), _field_elixir(own_half, cards)]

    # ally units
    ally_far = [u for u in ally if (r := _row(u)) is not None and r >= ENEMY_HALF_ROW]
    f += [float(len(ally)), float(len(ally_far)), _field_elixir(ally, cards)]

    # plays
    s_opp, c_opp, spent_opp = _plays(opp.get("recent_plays"), t, cards)
    s_own, _, spent_own = _plays(own.get("recent_plays"), t, cards)
    f += [s_opp, c_opp, spent_opp, s_own, spent_own]

    # hand
    hand = [c for c in (own.get("hand") or []) if c]
    costs = [_cost(c, cards) for c in hand if _cost(c, cards) > 0.0]
    nxt = own.get("next_card")
    f += [float(len(hand)), min(costs) if costs else -1.0, sum(costs) / len(costs) if costs else -1.0,
          float(sum(1 for c in costs if own_e is not None and c <= own_e)),
          _cost(nxt, cards) if _cost(nxt, cards) > 0.0 else -1.0]

    # towers
    hud_up = ms is not None
    own_hp, own_down = _princess(own.get("towers"), hud_up)
    opp_hp, opp_down = _princess(opp.get("towers"), hud_up)
    f += [own_hp, opp_hp, own_down, opp_down]

    # threats
    f.append(float(len(state.get("threats") or [])))
    return f


# --------------------------------------------------------------------------- model

class ActGBDT:
    """One LightGBM binary model (one horizon) of a directory written by `tools/train/train_act_gbdt.py`."""

    def __init__(self, booster, horizon: float, threshold: float, meta: dict | None = None,
                 cards: dict | None = None):
        self.booster = booster
        self.horizon = float(horizon)
        self.threshold = float(threshold)
        self.meta = meta or {}
        self.cards = cards

    @classmethod
    def load(cls, directory: str | Path, horizon: float | None = None) -> "ActGBDT":
        """Model of `horizon` (default `default_horizon` of meta.json) with the threshold stored for it. Raises
        OSError / ValueError for a missing or inconsistent directory."""
        import lightgbm as lgb
        d = Path(directory)
        meta = json.loads((d / META_FILE).read_text(encoding="utf-8"))
        if meta.get("features") != FEATURES:
            raise ValueError(f"{d / META_FILE}: features differ from clashbot.act_model.FEATURES; retrain the model")
        h = float(meta["default_horizon"] if horizon is None else horizon)
        key = f"{h:g}"
        thresholds = meta.get("thresholds") or {}
        if key not in thresholds:
            raise ValueError(f"{d / META_FILE}: no model for horizon {key} s (has {sorted(thresholds)})")
        booster = lgb.Booster(model_file=str(d / MODEL_FILE.format(h=h)))
        return cls(booster, h, float(thresholds[key]), meta)

    def p_act(self, state: dict, t: float) -> float:
        """P(the pro deploys a card within `horizon` s) for one state."""
        row = act_features(state, t, self.cards)
        return float(self.booster.predict([row], num_threads=1)[0])


class FastActPolicy:
    """Learned play/wait decision on structured features (MAC-AB A1); same contract and safety rules as
    `LearnedActPolicy`.

    No affordable card -> none; a threat to answer (defend) or a full bar (leak) -> act, whatever the model says; a
    save (the rules wait for a better answer, MAC-AC) -> hold, whatever the model says. In every other moment
    p(act) >= threshold plays (keeping the rule's mode and lane when the rules also wanted a punish, else mode
    "learned"), else hold. `model.p_act(state, t)` is an `ActGBDT`; the model is cheap enough to be asked on every
    frame, so nothing is cached."""

    def __init__(self, model, cards: dict | None = None, threshold: float | None = None,
                 rules: ReactivePolicy | None = None):
        self.model = model
        self.threshold = float(model.threshold if threshold is None else threshold)
        self.rules = rules if rules is not None else ReactivePolicy(cards)
        self.cards = self.rules.cards
        self.horizon = float(getattr(model, "horizon", DEFAULT_HORIZON))

    def decide(self, state: dict, t: float, context_text: str | None = None) -> ActDecision:
        """`context_text` is accepted for the policy interface of the live loop and ignored."""
        r = self.rules.decide(state, t)
        if r.mode in ("none", "save"):         # no card, or the rules wait for a better answer (MAC-AC): never overridden
            return r
        if r.act and r.mode in ("defend", "leak"):
            return r
        p = float(self.model.p_act(state, t))
        if p >= self.threshold:
            return ActDecision(True, r.mode if r.act else "learned", r.lane if r.act else None,
                               f"fast act p={p:.2f} h={self.horizon:g}s")
        return ActDecision(False, "hold", None, f"fast wait p={p:.2f} h={self.horizon:g}s")

    def target_tile(self, key: str, state: dict) -> list[float] | None:
        return self.rules.target_tile(key, state)

    def filter_candidates(self, candidates: list, decision: ActDecision, state: dict) -> list:
        if decision.mode == "learned":
            return self.rules.no_blind_spells(candidates, state)
        return self.rules.filter_candidates(candidates, decision, state)

"""Fast play/wait model on structured features (MAC-AB A1): a LightGBM head instead of the d1 act-head.

  `act_features(state, t)` turns the clean live state dict (the builder output, also stored in
  `dataset/v1/act_dense/states_<split>.jsonl`) into the fixed-order `FEATURES` vector, the same way for training and for
  the live loop. `ActGBDT.load(dir)` reads `meta.json` + `model_h<h>.json` (the trees of `Booster.dump_model()`,
  written next to `model_h<h>.txt` by `tools/train/train_act_gbdt.py`) and answers `p_act(state, t)` = P(the pro deploys
  a card within `horizon` seconds), in well under a millisecond. `FastActPolicy` is `LearnedActPolicy` with that model:
  the rules keep the safety modes (none / defend / leak), the model decides every other moment (`p >= threshold` plays,
  else hold).

  The runtime never imports lightgbm: the live loop imports torch first, and torch + lightgbm in one process crash
  (two OpenMP runtimes; `Booster.predict` segfaults). `TreeEnsemble` walks the dumped trees in plain Python / numpy
  with LightGBM's own decision rule instead. A missing .json next to an existing .txt is exported once by the trainer
  script in a subprocess (which imports lightgbm, never torch).
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

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
MODEL_FILE = "model_h{h:g}.txt"           # LightGBM text model (trainer output, read only by the exporter)
JSON_FILE = "model_h{h:g}.json"           # its trees as `Booster.dump_model()` JSON (what the runtime reads)
META_FILE = "meta.json"
TRAINER = Path(__file__).resolve().parents[1] / "tools" / "train" / "train_act_gbdt.py"
EXPORT_TIMEOUT_S = 120

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

_ZERO_THRESHOLD = 1.0000000180025095e-35        # LightGBM kZeroThreshold (1e-35f as a double)
_MISSING = {"None": 0, "Zero": 1, "NaN": 2}     # `missing_type` of a split node in the dump
_NAN, _ZERO = _MISSING["NaN"], _MISSING["Zero"]


def _compile_tree(root: dict) -> tuple[list, list, list, list, list, list, list]:
    """(feature, threshold, left, right, default_left, missing_type, leaf_values) of one dumped tree, indexed by the
    dump's own `split_index` / `leaf_index`. A child is an internal node index (>= 0) or `~leaf_index` (< 0)."""
    if "split_index" not in root:                                 # a tree that never split: one constant leaf
        return [], [], [], [], [], [], [float(root["leaf_value"])]
    nodes: dict[int, dict] = {}
    leaves: dict[int, float] = {}
    stack = [root]
    while stack:
        node = stack.pop()
        if "split_index" not in node:
            leaves[node["leaf_index"]] = float(node["leaf_value"])
            continue
        if node.get("decision_type", "<=") != "<=":
            raise ValueError(f"unsupported split {node.get('decision_type')!r}: only numerical '<=' splits")
        nodes[node["split_index"]] = node
        stack += [node["left_child"], node["right_child"]]

    def ref(child: dict) -> int:
        return child["split_index"] if "split_index" in child else ~child["leaf_index"]

    order = [nodes[i] for i in range(len(nodes))]
    return ([n["split_feature"] for n in order], [float(n["threshold"]) for n in order],
            [ref(n["left_child"]) for n in order], [ref(n["right_child"]) for n in order],
            [bool(n["default_left"]) for n in order], [_MISSING[n.get("missing_type", "None")] for n in order],
            [leaves[i] for i in range(len(leaves))])


class TreeEnsemble:
    """A LightGBM binary model read from the JSON of `Booster.dump_model()`, evaluated without lightgbm.

    Per tree the walk is LightGBM's `NumericalDecision`: a NaN counts as 0 unless the node's `missing_type` is NaN; a
    NaN (missing_type NaN) or a zero (|v| <= 1e-35, missing_type Zero) takes the node's default direction; every other
    value goes left iff `v <= threshold`. The raw score is `init_score` (if the dump has one) plus the sum of the leaf
    values (divided by the tree count when `average_output`), the probability is `1 / (1 + exp(-sigmoid * raw))`.
    Categorical splits, multiclass and non-binary objectives raise ValueError."""

    def __init__(self, dump: dict):
        objective = str(dump.get("objective", ""))
        if (not objective.startswith("binary") or int(dump.get("num_class", 1)) != 1
                or int(dump.get("num_tree_per_iteration", 1)) != 1):
            raise ValueError(f"unsupported LightGBM objective {objective!r}: only binary, single-output models")
        m = re.search(r"sigmoid:([-+0-9.eE]+)", objective)
        self.sigmoid = float(m.group(1)) if m else 1.0
        self.average = bool(dump.get("average_output", False))
        self.init_score = float(dump.get("init_score") or 0.0)
        self.n_features = int(dump.get("max_feature_idx", -1)) + 1
        self.trees = [_compile_tree(t["tree_structure"]) for t in dump["tree_info"]]
        self._arrays: list | None = None

    def raw(self, row) -> float:
        """Raw score of one feature row (a sequence of floats)."""
        total = self.init_score
        for feat, thr, left, right, dleft, miss, leaves in self.trees:
            if not feat:                                            # a tree that never split
                total += leaves[0]
                continue
            node = 0
            while node >= 0:
                v = row[feat[node]]
                m = miss[node]
                if v != v:                                          # NaN
                    if m == _NAN:
                        node = left[node] if dleft[node] else right[node]
                        continue
                    v = 0.0
                if m == _ZERO and -_ZERO_THRESHOLD <= v <= _ZERO_THRESHOLD:
                    node = left[node] if dleft[node] else right[node]
                else:
                    node = left[node] if v <= thr[node] else right[node]
            total += leaves[~node]
        return total / len(self.trees) if self.average and self.trees else total

    def _sigmoid(self, raw):
        return 1.0 / (1.0 + np.exp(-self.sigmoid * raw))

    def predict_row(self, row) -> float:
        """P(act) of one feature row, in plain Python (about 0.1 ms for the shipped models)."""
        return float(self._sigmoid(self.raw(row)))

    def predict(self, X) -> np.ndarray:
        """P(act) of every row of a 2-D array (the same walk, vectorised per tree)."""
        X = np.asarray(X, dtype=np.float64)
        if self._arrays is None:
            self._arrays = [(np.array(f, dtype=np.int64), np.array(t), np.array(l, dtype=np.int64),
                             np.array(r, dtype=np.int64), np.array(d, dtype=bool), np.array(m, dtype=np.int64),
                             np.array(v)) for f, t, l, r, d, m, v in self.trees]
        total = np.full(len(X), self.init_score)
        for feat, thr, left, right, dleft, miss, leaves in self._arrays:
            if not len(feat):
                total += leaves[0]
                continue
            node = np.zeros(len(X), dtype=np.int64)
            active = np.arange(len(X))
            while active.size:
                nd = node[active]
                v = X[active, feat[nd]]
                nan = np.isnan(v)
                m = miss[nd]
                v = np.where(nan & (m != _NAN), 0.0, v)
                default = ((m == _ZERO) & (np.abs(v) <= _ZERO_THRESHOLD)) | ((m == _NAN) & nan)
                nxt = np.where(np.where(default, dleft[nd], v <= thr[nd]), left[nd], right[nd])
                node[active] = nxt
                active = active[nxt >= 0]
            total += leaves[~node]
        if self.average and self.trees:
            total = total / len(self.trees)
        return self._sigmoid(total)


def _export_json(directory: Path) -> None:
    """Write the missing model_h*.json of `directory` from its model_h*.txt. Imports lightgbm, so it runs in a
    subprocess (the trainer script, which never imports torch): lightgbm must stay out of this process."""
    try:
        proc = subprocess.run([sys.executable, str(TRAINER), "--export-json", str(directory)], capture_output=True,
                              text=True, timeout=EXPORT_TIMEOUT_S)
    except (OSError, subprocess.SubprocessError) as exc:
        raise OSError(f"cannot export {directory}/model_h*.json with {TRAINER}: {exc}") from exc
    if proc.returncode != 0:
        raise OSError(f"exporting {directory}/model_h*.json failed (exit {proc.returncode}): "
                      f"{proc.stderr.strip()[-300:]}")


class ActGBDT:
    """One LightGBM binary model (one horizon) of a directory written by `tools/train/train_act_gbdt.py`, evaluated
    by a `TreeEnsemble` (no lightgbm in this process)."""

    def __init__(self, trees: TreeEnsemble, horizon: float, threshold: float, meta: dict | None = None,
                 cards: dict | None = None):
        self.trees = trees
        self.horizon = float(horizon)
        self.threshold = float(threshold)
        self.meta = meta or {}
        self.cards = cards

    @classmethod
    def load(cls, directory: str | Path, horizon: float | None = None) -> "ActGBDT":
        """Model of `horizon` (default `default_horizon` of meta.json) with the threshold stored for it. Raises
        OSError / ValueError for a missing or inconsistent directory. If model_h<h>.json is missing but
        model_h<h>.txt exists, the directory is exported once in a subprocess first."""
        d = Path(directory)
        meta = json.loads((d / META_FILE).read_text(encoding="utf-8"))
        if meta.get("features") != FEATURES:
            raise ValueError(f"{d / META_FILE}: features differ from clashbot.act_model.FEATURES; retrain the model")
        h = float(meta["default_horizon"] if horizon is None else horizon)
        key = f"{h:g}"
        thresholds = meta.get("thresholds") or {}
        if key not in thresholds:
            raise ValueError(f"{d / META_FILE}: no model for horizon {key} s (has {sorted(thresholds)})")
        path = d / JSON_FILE.format(h=h)
        if not path.is_file() and (d / MODEL_FILE.format(h=h)).is_file():
            _export_json(d)
        trees = TreeEnsemble(json.loads(path.read_text(encoding="utf-8")))
        return cls(trees, h, float(thresholds[key]), meta)

    def p_act(self, state: dict, t: float) -> float:
        """P(the pro deploys a card within `horizon` s) for one state."""
        return self.trees.predict_row(act_features(state, t, self.cards))


class FastActPolicy:
    """Learned play/wait decision on structured features (MAC-AB A1); same contract and safety rules as
    `LearnedActPolicy`.

    No affordable card -> none; a threat to answer (defend), an offence window (counter, MAC-AG), a great spell (spell,
    MAC-AL) or a full bar (leak) -> act, whatever the model says (the rules' decision is returned unchanged, with its allowed cards and tiles); a
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
        if r.act and r.mode in ("defend", "counter", "spell", "leak"):
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

"""Structured state vector for kNN retrieval (FEATURE_VERSION = 1)."""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from clashbot.config import LANES, ZONES
from clashbot.game_rules import is_playable, lane_of, load_cards, zone_of

FEATURE_VERSION = 1
BLOCK_WEIGHTS = {"scalars": 1.0, "phase": 1.0, "hand": 1.0, "deck": 1.0, "opp_deck": 1.0,
                 "opp_recent": 1.0, "own_recent": 1.0, "towers": 1.0, "field": 1.0, "threat": 1.0}
RECENT_S = 5.0
TOWER_HP_SCALE = 5000.0

PHASE_NAMES = ("single_elixir", "double_elixir", "triple_elixir", "overtime", "unknown")
SIDES = ("ally", "enemy")
OWN_TOWERS = ("own_left", "own_right", "own_king")
ENEMY_TOWERS = ("enemy_left", "enemy_right", "enemy_king")
TOWER_KEYS = ("left", "right", "king")
BLOCK_ORDER = ("scalars", "phase", "hand", "deck", "opp_deck", "opp_recent", "own_recent",
               "towers", "field", "threat")


@lru_cache(maxsize=1)
def vocab() -> list[str]:
    cards = load_cards()
    return sorted(s for s in cards if is_playable(s, cards))


@lru_cache(maxsize=1)
def _vocab_index() -> dict[str, int]:
    return {s: i for i, s in enumerate(vocab())}


def _block_dims() -> dict[str, int]:
    v = len(vocab())
    return {"scalars": 3, "phase": 5, "hand": v, "deck": v, "opp_deck": v, "opp_recent": v,
            "own_recent": v, "towers": 6, "field": 24, "threat": 6}


def dim() -> int:
    return sum(_block_dims().values())


def block_slices() -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    pos = 0
    dims = _block_dims()
    for name in BLOCK_ORDER:
        out[name] = [pos, pos + dims[name]]
        pos += dims[name]
    return out


def feature_names() -> list[str]:
    names = ["scalars:own_elixir", "scalars:opp_elixir", "scalars:match_progress"]
    names += [f"phase:{p}" for p in PHASE_NAMES]
    for block in ("hand", "deck", "opp_deck", "opp_recent", "own_recent"):
        names += [f"{block}:{s}" for s in vocab()]
    names += [f"towers:{t}" for t in OWN_TOWERS]
    names += [f"towers:{t}" for t in ENEMY_TOWERS]
    for side in SIDES:
        for lane in LANES:
            for zone in ZONES:
                names.append(f"field:{side}:{lane}:{zone}")
    names += [f"threat:{t}" for t in OWN_TOWERS + ENEMY_TOWERS]
    return names


def _multi_hot(cards) -> np.ndarray:
    idx = _vocab_index()
    v = np.zeros(len(idx), dtype=np.float64)
    for c in cards or []:
        if c is not None and c in idx:
            v[idx[c]] = 1.0
    return v


def _recent(plays, t: float) -> np.ndarray:
    cards = []
    for p in plays or []:
        pt = p.get("t")
        if pt is None:
            continue
        if 0 <= t - pt <= RECENT_S:
            cards.append(p.get("card"))
    return _multi_hot(cards)


def _tower_hp(towers: dict | None, key: str) -> float:
    hp = (towers or {}).get(key)
    if hp is None:
        return 1.0
    return min(max(hp / TOWER_HP_SCALE, 0.0), 1.0)


def _blocks(state: dict, t: float) -> dict[str, np.ndarray]:
    own = state.get("own") or {}
    opp = state.get("opponent") or {}

    own_el = own.get("elixir")
    opp_el = opp.get("elixir_estimate")
    ms = state.get("match_seconds")
    scalars = np.array([
        0.0 if own_el is None else own_el / 10.0,
        0.5 if opp_el is None else opp_el / 10.0,
        0.5 if ms is None else min(max(ms / 300.0, 0.0), 1.5),
    ], dtype=np.float64)

    phase = np.zeros(len(PHASE_NAMES), dtype=np.float64)
    ph = state.get("phase")
    phase[PHASE_NAMES.index(ph) if ph in PHASE_NAMES[:4] else 4] = 1.0

    towers = np.array(
        [_tower_hp(own.get("towers"), k) for k in TOWER_KEYS]
        + [_tower_hp(opp.get("towers"), k) for k in TOWER_KEYS], dtype=np.float64)

    counts = np.zeros(24, dtype=np.float64)
    best: dict[tuple[str, str], float] = {}
    for u in state.get("units") or []:
        side = u.get("side")
        tile = u.get("tile")
        lane, zone = lane_of(tile), zone_of(tile)
        if side in SIDES and lane is not None and zone is not None:
            i = SIDES.index(side) * 12 + LANES.index(lane) * 4 + ZONES.index(zone)
            counts[i] += 1
        eta = u.get("eta_tower")
        if eta and eta.get("tower") is not None and eta.get("s") is not None:
            tw = eta["tower"]
            if (side == "enemy" and tw in OWN_TOWERS) or (side == "ally" and tw in ENEMY_TOWERS):
                key = (side, tw)
                if key not in best or eta["s"] < best[key]:
                    best[key] = eta["s"]
    field = np.log1p(counts)

    threat = np.zeros(6, dtype=np.float64)
    for i, tw in enumerate(OWN_TOWERS):
        if ("enemy", tw) in best:
            threat[i] = 1.0 / (1.0 + best[("enemy", tw)])
    for i, tw in enumerate(ENEMY_TOWERS):
        if ("ally", tw) in best:
            threat[3 + i] = 1.0 / (1.0 + best[("ally", tw)])

    return {
        "scalars": scalars,
        "phase": phase,
        "hand": _multi_hot(own.get("hand")),
        "deck": _multi_hot(own.get("deck")),
        "opp_deck": _multi_hot(opp.get("deck_known")),
        "opp_recent": _recent(opp.get("recent_plays"), t),
        "own_recent": _recent(own.get("recent_plays"), t),
        "towers": towers,
        "field": field,
        "threat": threat,
    }


def featurize(state: dict, t: float) -> np.ndarray:
    blocks = _blocks(state, t)
    parts = []
    for name in BLOCK_ORDER:
        b = blocks[name]
        n = float(np.linalg.norm(b))
        if n > 0:
            b = b / n
        parts.append(b * BLOCK_WEIGHTS[name])
    vec = np.concatenate(parts)
    n = float(np.linalg.norm(vec))
    if n > 0:
        vec = vec / n
    return vec.astype(np.float32)

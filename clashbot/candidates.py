"""Candidate generation: kNN over similar pro states fused with KG counters (D-007), plus tactical moves (MAC-Z3).

With a `TacticalReport` (clashbot.tactics) the generator puts the engine's concrete moves first, then the lookup
candidates, and gives every candidate a one-line note for d1. Without a report nothing changes: no notes, no tiles.
A key that is not an exact tactical move but plays the card of a report move also gets a card-level tactics note, so
d1 sees that the engine agreed on the card even when its placement differs. `generate` (live) and `option_notes`
(training export) build every note through the one `_key_note` helper.
"""
from __future__ import annotations

from dataclasses import dataclass

from clashbot.config import HOLD_KEY
from clashbot.game_rules import (action_key, lane_of, legal_action_keys, legal_cards, load_cards,
                                 parse_action_key, zone_of)
from clashbot.kg import KnowledgeGraph
from clashbot.retrieval import StateIndex

PROS_ALSO = "; pros played this too"
FILLER_NOTE = "filler: no pro or counter support"
TAC_CARD_NOTE = "tactics suggests this card"      # prefix of the card-level tactics note


@dataclass
class Candidate:
    key: str
    score: float
    sources: dict          # e.g. {"knn": 0.031, "kg": 0.5}; {"tac": 0.9} for a tactical move
    tile: list[float] | None = None     # concrete deploy tile (tactical candidates only)
    note: str | None = None             # one line for the d1 option description


def _normalise(channel: dict[str, float]) -> dict[str, float]:
    if not channel:
        return {}
    top = max(channel.values())
    if top <= 0:
        return {k: 0.0 for k in channel}
    return {k: v / top for k, v in channel.items()}


def _lookup_note(key: str, knn_n: dict[str, float], kg_n: dict[str, float], kg_threat: dict[str, str],
                 filler: bool = False) -> str | None:
    """The note of a lookup (non-tactical) key: pros played it, else it is a known counter, else (`filler`, for
    fallback candidates) a filler line. `knn_n` / `kg_n` are the normalised channels, `kg_threat` the threat card
    that produced each KG key."""
    if knn_n.get(key, 0.0) > 0:
        return f"pros played this in similar states (support {knn_n[key]:.2f})"
    if kg_n.get(key, 0.0) > 0:
        return f"known counter of {kg_threat.get(key, 'the threat')}"
    return FILLER_NOTE if filler else None


def _tac_note(move, knn_value: float) -> str:
    note = f"tactics: {move.intent}, {move.reason}"
    return note + PROS_ALSO if knn_value > 0 else note


def _tac_card_note(move) -> str:
    """The note of a key that is not the engine's placement but plays the card of `move`."""
    parts = parse_action_key(move.key)
    return f"{TAC_CARD_NOTE} ({move.intent}: {move.reason}, at {parts['lane']} {parts['zone']})"


def _card_moves(report) -> dict[str, object]:
    """The first move of each card in `report.moves` (any move, legal or not): the source of card-level notes."""
    out: dict[str, object] = {}
    for move in (report.moves if report is not None else []):
        out.setdefault(move.card, move)
    return out


def _key_note(key: str, exact: dict[str, object], card_moves: dict[str, object], knn_n: dict[str, float],
              kg_n: dict[str, float], kg_threat: dict[str, str], filler: bool = False) -> str | None:
    """The one note builder behind `generate` and `option_notes`. `exact` maps the keys that are tactical moves to
    their move (exact tactics note); any other key gets its lookup note, joined with the card-level tactics note when
    its card is the card of a report move (`card_moves`). The filler line (`filler=True`, no pro or counter support)
    is only used when there is nothing else to say: a card note replaces it instead of being joined to it."""
    if key in exact:
        return _tac_note(exact[key], knn_n.get(key, 0.0))
    note = _lookup_note(key, knn_n, kg_n, kg_threat)               # real support only
    move = card_moves.get(key.split("|", 1)[0])
    if move is None:
        return note or (FILLER_NOTE if filler else None)
    card_note = _tac_card_note(move)
    return f"{note}; {card_note}" if note else card_note


class CandidateGenerator:
    def __init__(self, index: StateIndex, kg: KnowledgeGraph, cards: dict | None = None,
                 kg_weight: float = 0.5, hold_prior: float = 0.01):
        self.index = index
        self.kg = kg
        self.cards = cards if cards is not None else load_cards()
        self.kg_weight = kg_weight
        self.hold_prior = hold_prior

    def _threats(self, state: dict, t: float) -> list[tuple[str, list | None]]:
        threats: list[tuple[str, list | None]] = []
        for p in state["opponent"].get("recent_plays", []):
            if p.get("card") and p.get("t") is not None and 0 <= t - p["t"] <= 5.0:
                threats.append((p["card"], p.get("tile")))
        for u in state.get("units", []):
            eta = u.get("eta_tower")
            if (u.get("side") == "enemy" and u.get("card") and isinstance(eta, dict)
                    and str(eta.get("tower", "")).startswith("own")
                    and eta.get("s") is not None and eta["s"] <= 8.0):
                threats.append((u["card"], u.get("tile")))
        return threats

    def _channels(self, state: dict, t: float, context_text: str, legal_set: set[str], **filters):
        """(knn_n, kg_n, kg_threat): the normalised kNN and KG channels over the legal keys, and for each KG key the
        threat card that contributed most to it."""
        cards = self.cards
        knn: dict[str, float] = {}
        hits = self.index.search(state, t, context_text, k=20, depth=50, **filters)
        for hit in hits:
            if hit.action_key != HOLD_KEY and hit.action_key in legal_set:
                knn[hit.action_key] = knn.get(hit.action_key, 0.0) + hit.score

        kg: dict[str, float] = {}
        kg_threat: dict[str, str] = {}
        kg_best: dict[str, float] = {}
        own_cards = set(legal_cards(state, cards))
        for threat, tile in self._threats(state, t):
            for c, s in self.kg.counter_scores(threat).items():
                if c not in own_cards:
                    continue
                if cards[c]["card_type"] == "Spell":
                    if tile is None:
                        continue
                    key = action_key(c, lane_of(tile), zone_of(tile))
                else:
                    key = action_key(c, lane_of(tile) or "middle", "mid")
                if key in legal_set:
                    kg[key] = kg.get(key, 0.0) + s
                    if s > kg_best.get(key, float("-inf")):
                        kg_best[key], kg_threat[key] = s, threat
        return _normalise(knn), _normalise(kg), kg_threat

    def generate(self, state: dict, t: float, context_text: str, n: int = 4, report=None, tac_n: int = 2,
                 **filters) -> list[Candidate]:
        """`n` candidates. With a tactics `report` the first (up to `tac_n`) are its legal moves, each with its tile
        and a note; the rest come from the lookup ranking (skipping the tactical keys, whose knn / kg support is
        merged into the tactical candidate) and then from the legal keys. Without a report: lookup only, no notes."""
        cards = self.cards
        legal = legal_action_keys(state, cards)
        legal_set = set(legal)
        knn_n, kg_n, kg_threat = self._channels(state, t, context_text, legal_set, **filters)

        score: dict[str, float] = {}
        for key in set(knn_n) | set(kg_n):
            score[key] = knn_n.get(key, 0.0) + self.kg_weight * kg_n.get(key, 0.0)

        out: list[Candidate] = []
        taken: set[str] = set()
        exact: dict[str, object] = {}                       # tactical candidate key -> its move
        card_moves = _card_moves(report)
        if report is not None:
            for move in report.moves:
                if len(out) >= min(max(tac_n, 0), max(n, 0)):
                    break
                if move.key in taken or move.key not in legal_set:
                    continue
                sources = {"tac": round(move.score, 4)}
                if move.key in score:
                    sources.update(knn=round(knn_n.get(move.key, 0.0), 4), kg=round(kg_n.get(move.key, 0.0), 4))
                exact[move.key] = move
                out.append(Candidate(move.key, move.score, sources, list(move.tile),
                                     _key_note(move.key, exact, card_moves, knn_n, kg_n, kg_threat)))
                taken.add(move.key)

        ranked = [k for k in sorted(score, key=lambda k: (-score[k], k)) if k not in taken]
        for k in ranked[: max(n - len(out), 0)]:
            note = _key_note(k, exact, card_moves, knn_n, kg_n, kg_threat) if report is not None else None
            out.append(Candidate(k, score[k], {"knn": round(knn_n.get(k, 0.0), 4), "kg": round(kg_n.get(k, 0.0), 4)},
                                 note=note))
            taken.add(k)
        if len(out) < n:
            for k in legal:
                if len(out) >= n:
                    break
                if k == HOLD_KEY or k in taken:
                    continue
                note = (_key_note(k, exact, card_moves, knn_n, kg_n, kg_threat, filler=True)
                        if report is not None else None)
                out.append(Candidate(k, 0.0, {"fallback": 1}, note=note))
                taken.add(k)
        return out

    def option_notes(self, state: dict, t: float, context_text: str, keys: list[str], report=None,
                     tac_n: int | None = None, **filters) -> dict[str, str]:
        """The notes `generate(..., report=report)` would attach to `keys` (any keys, e.g. the options of a training
        example): a key that is one of the report's moves gets its tactics note, every other key its lookup note
        (pros played it / known counter / filler) joined with the card-level tactics note when its card is the card
        of a report move; all from the same kNN / KG normalisation and the same `_key_note` as `generate`.
        `tac_n=None` (training export) lets every report move count; `tac_n=N` restricts them to the first N legal
        moves, exactly the ones `generate(..., tac_n=N)` turns into tactical candidates."""
        legal_set = set(legal_action_keys(state, self.cards))
        knn_n, kg_n, kg_threat = self._channels(state, t, context_text, legal_set, **filters)
        moves: dict[str, object] = {}
        for move in (report.moves if report is not None else []):
            if tac_n is not None and (len(moves) >= tac_n or move.key not in legal_set):
                continue
            moves.setdefault(move.key, move)
        card_moves = _card_moves(report)
        return {key: _key_note(key, moves, card_moves, knn_n, kg_n, kg_threat, filler=True) for key in keys}

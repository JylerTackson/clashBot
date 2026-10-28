"""Candidate generation: kNN over similar pro states fused with KG counters (D-007)."""
from __future__ import annotations

from dataclasses import dataclass

from clashbot.config import HOLD_KEY
from clashbot.game_rules import (action_key, lane_of, legal_action_keys, legal_cards, load_cards,
                                 zone_of)
from clashbot.kg import KnowledgeGraph
from clashbot.retrieval import StateIndex


@dataclass
class Candidate:
    key: str
    score: float
    sources: dict          # e.g. {"knn": 0.031, "kg": 0.5}


def _normalise(channel: dict[str, float]) -> dict[str, float]:
    if not channel:
        return {}
    top = max(channel.values())
    if top <= 0:
        return {k: 0.0 for k in channel}
    return {k: v / top for k, v in channel.items()}


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

    def generate(self, state: dict, t: float, context_text: str, n: int = 4, **filters) -> list[Candidate]:
        cards = self.cards
        legal = legal_action_keys(state, cards)
        legal_set = set(legal)

        knn: dict[str, float] = {}
        hits = self.index.search(state, t, context_text, k=20, depth=50, **filters)
        for hit in hits:
            if hit.action_key != HOLD_KEY and hit.action_key in legal_set:
                knn[hit.action_key] = knn.get(hit.action_key, 0.0) + hit.score

        kg: dict[str, float] = {}
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

        knn_n = _normalise(knn)
        kg_n = _normalise(kg)
        score: dict[str, float] = {}
        for key in set(knn_n) | set(kg_n):
            score[key] = knn_n.get(key, 0.0) + self.kg_weight * kg_n.get(key, 0.0)

        ranked = sorted(score, key=lambda k: (-score[k], k))
        plays = ranked[: max(n, 0)]
        out = [Candidate(k, score[k], {"knn": round(knn_n.get(k, 0.0), 4), "kg": round(kg_n.get(k, 0.0), 4)})
               for k in plays]
        if len(out) < n:
            chosen = set(plays)
            for k in legal:
                if len(out) >= n:
                    break
                if k == HOLD_KEY or k in chosen:
                    continue
                out.append(Candidate(k, 0.0, {"fallback": 1}))
                chosen.add(k)
        return out

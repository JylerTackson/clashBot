"""Opponent-deck model (MAC-AK K3): given the opponent cards seen so far, which cards is the rest of the deck?

The model is built from a list of known decks (pro decks from the KB and the game decks, deduplicated and counted) and
scores every unseen card by mixing two views of the same evidence:

  (a) nearest decks - Jaccard(seen, deck) over the known decks, softmax with a temperature, then the weighted share of
      decks that contain the card;
  (b) naive co-occurrence - sum over the seen cards s of log P(card | s) with add-one smoothing, then softmax.

Both are normalised to sum to 1 over the unseen cards, mixed 0.6 (a) + 0.4 (b), and scaled so the probabilities sum to
(8 - |seen|), the number of hidden slots (each capped at 1). `dataset/v1/kg2/deck_cooc.json` is written by
`tools/data/build_deck_cooc.py` through `OppDeckModel.save` and read back by `OppDeckModel.load`.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

DECK_SIZE = 8
W_NEAREST, W_COOC = 0.6, 0.4
DEFAULT_TEMPERATURE = 0.05       # softmax temperature on Jaccard similarity (a)
DEFAULT_COUNT_POWER = 0.0        # a deck seen n times weighs n ** count_power (0 = each distinct deck once)
NO_ARCHETYPE = {None, "", "none", "null"}


def _num(x: float) -> int | float:
    return int(x) if float(x).is_integer() else round(float(x), 4)


def cap_normalise(scores: dict[str, float], total: float) -> dict[str, float]:
    """Scale non-negative scores to sum to `total` with every value <= 1 (water-filling: capped cards keep 1)."""
    out: dict[str, float] = {}
    free = {c: max(0.0, v) for c, v in scores.items()}
    remaining = float(total)
    while free:
        if remaining <= 1e-12:
            out.update({c: 0.0 for c in free})
            break
        s = sum(free.values())
        if s <= 0:                                   # no signal left: spread the rest evenly
            share = min(1.0, remaining / len(free))
            out.update({c: share for c in free})
            break
        scale = remaining / s
        over = [c for c, v in free.items() if v * scale > 1.0]
        if not over:
            out.update({c: v * scale for c, v in free.items()})
            break
        for c in over:
            out[c] = 1.0
            remaining -= 1.0
            del free[c]
    return out


class OppDeckModel:
    """Known decks -> `predict(seen)`: the likely other cards of the opponent's deck."""

    def __init__(self, decks: Sequence[Sequence[str]], counts: Sequence[float] | None = None,
                 archetypes: Sequence[str | None] | None = None, cards: Sequence[str] | None = None,
                 temperature: float = DEFAULT_TEMPERATURE, count_power: float = DEFAULT_COUNT_POWER) -> None:
        self.decks: list[tuple[str, ...]] = [tuple(sorted(set(d))) for d in decks]
        self.counts: list[float] = list(counts) if counts is not None else [1] * len(self.decks)
        self.archetypes: list[str | None] = (
            [None if a in NO_ARCHETYPE else a for a in archetypes] if archetypes is not None else [None] * len(self.decks))
        if not (len(self.decks) == len(self.counts) == len(self.archetypes)):
            raise ValueError("decks, counts and archetypes must have the same length")
        self.temperature = float(temperature)
        self.count_power = float(count_power)
        self._sets = [frozenset(d) for d in self.decks]
        self._weights = [float(c) ** self.count_power for c in self.counts]       # prior weight of each known deck
        raw_cards = {c for d in self.decks for c in d}
        self.cards: list[str] = sorted(set(cards) | raw_cards) if cards is not None else sorted(raw_cards)
        card_counts: Counter = Counter()
        pairs: dict[str, Counter] = {c: Counter() for c in self.cards}
        for deck, w in zip(self.decks, self._weights):
            for a in deck:
                card_counts[a] += w
                for b in deck:
                    if a != b:
                        pairs[a][b] += w
        self.card_counts: dict[str, int | float] = {c: _num(card_counts[c]) for c in self.cards}
        self.pair_counts: dict[str, dict[str, int | float]] = {
            a: {b: _num(n) for b, n in sorted(pairs[a].items())} for a in self.cards}
        self._cooc_n = {c: float(card_counts[c]) for c in self.cards}
        self._pair = {a: {b: float(n) for b, n in row.items()} for a, row in pairs.items()}

    # ------------------------------------------------------------------ io
    def to_dict(self, meta: dict | None = None) -> dict:
        data = {"cards": self.cards, "pair_counts": self.pair_counts, "card_counts": self.card_counts,
                "decks": [list(d) for d in self.decks], "deck_counts": [_num(c) for c in self.counts],
                "deck_archetypes": self.archetypes,
                "params": {"temperature": self.temperature, "count_power": self.count_power,
                           "w_nearest": W_NEAREST, "w_cooc": W_COOC}}
        if meta:
            data["meta"] = meta
        return data

    def save(self, path: str | Path, meta: dict | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(meta), sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")

    @classmethod
    def from_dict(cls, data: dict, **overrides) -> "OppDeckModel":
        params = dict(data.get("params") or {})
        kw = {"temperature": params.get("temperature", DEFAULT_TEMPERATURE),
              "count_power": params.get("count_power", DEFAULT_COUNT_POWER)}
        kw.update(overrides)
        return cls(data["decks"], data.get("deck_counts"), data.get("deck_archetypes"), data.get("cards"), **kw)

    @classmethod
    def load(cls, path: str | Path, **overrides) -> "OppDeckModel":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")), **overrides)

    # ------------------------------------------------------------------ scoring
    def _deck_posterior(self, seen: frozenset[str]) -> list[float]:
        """Normalised weight of every known deck given the seen cards: prior * exp(Jaccard / temperature)."""
        if not self.decks:
            return []
        logs = []
        for d, w in zip(self._sets, self._weights):
            union = len(seen | d)
            jac = len(seen & d) / union if union else 0.0
            logs.append(math.log(w) + jac / self.temperature if w > 0 else -math.inf)
        top = max(logs)
        if top == -math.inf:
            return [0.0] * len(logs)
        ex = [math.exp(x - top) for x in logs]
        z = sum(ex)
        return [e / z for e in ex]

    def _nearest(self, seen: frozenset[str], cand: list[str]) -> dict[str, float]:
        post = self._deck_posterior(seen)
        mass = {c: 0.0 for c in cand}
        for p, d in zip(post, self._sets):
            if p > 0:
                for c in d:
                    if c in mass:
                        mass[c] += p
        return self._distribution(mass, cand)

    def _cooc(self, seen: frozenset[str], cand: list[str]) -> dict[str, float]:
        known = [s for s in seen if s in self._cooc_n]
        if not known:                                         # nothing to condition on: plain popularity
            return self._distribution({c: self._cooc_n.get(c, 0.0) for c in cand}, cand)
        scores = {}
        for c in cand:
            scores[c] = sum(math.log((self._pair[s].get(c, 0.0) + 1.0) / (self._cooc_n[s] + 2.0)) for s in known)
        top = max(scores.values())
        return self._distribution({c: math.exp(v - top) for c, v in scores.items()}, cand)

    @staticmethod
    def _distribution(mass: dict[str, float], cand: list[str]) -> dict[str, float]:
        z = sum(mass.values())
        if z <= 0:
            return {c: 1.0 / len(cand) for c in cand}
        return {c: mass[c] / z for c in cand}

    def predict(self, seen: Iterable[str], top: int = 8) -> list[tuple[str, float]]:
        """The `top` most likely unseen cards as (card, probability), best first; probabilities sum to 8 - |seen|."""
        seen_set = frozenset(seen)
        slots = DECK_SIZE - len(seen_set)
        cand = [c for c in self.cards if c not in seen_set]
        if slots <= 0 or not cand or top <= 0:
            return []
        pa = self._nearest(seen_set, cand)
        pb = self._cooc(seen_set, cand)
        mixed = {c: W_NEAREST * pa[c] + W_COOC * pb[c] for c in cand}
        probs = cap_normalise(mixed, slots)
        ranked = sorted(probs.items(), key=lambda kv: (-kv[1], kv[0]))
        return [(c, round(p, 6)) for c, p in ranked[:top]]

    def popularity(self, seen: Iterable[str] = (), top: int = 8) -> list[tuple[str, float]]:
        """Baseline: the most common cards over the known decks (weighted), seen cards excluded."""
        seen_set = frozenset(seen)
        total = sum(self._weights) or 1.0
        ranked = sorted(((c, self._cooc_n[c] / total) for c in self.cards if c not in seen_set),
                        key=lambda kv: (-kv[1], kv[0]))
        return ranked[:top]

    def archetype_probs(self, seen: Iterable[str]) -> dict[str, float]:
        """Archetype -> probability, from the posterior over the known decks that carry an archetype label."""
        post = self._deck_posterior(frozenset(seen))
        mass: Counter = Counter()
        for p, a in zip(post, self.archetypes):
            if a is not None:
                mass[a] += p
        z = sum(mass.values())
        return {a: m / z for a, m in sorted(mass.items(), key=lambda kv: (-kv[1], kv[0]))} if z > 0 else {}

    def likely_archetype(self, seen: Iterable[str]) -> tuple[str, float] | None:
        """(archetype, probability) of the most likely archetype, or None when no known deck has a label."""
        probs = self.archetype_probs(seen)
        if not probs:
            return None
        arch = next(iter(probs))
        return arch, round(probs[arch], 4)

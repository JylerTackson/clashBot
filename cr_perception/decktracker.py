"""Opponent deck and cycle tracking, with Phase 2 knowledge-base inference.

Clash Royale rules used here:
  * a deck is exactly 8 cards;
  * the hand holds 4, plus one "next" card;
  * after a card is played it goes to the back of the queue, so once the deck
    is known the play order is a fixed cycle.

Before all 8 cards are seen, the partial observation is matched against the
Phase 2 deck files (knowledge_base/decks/*.md) to rank likely full decks and
surface probable remaining cards. Those are INFERENCES and are labelled so.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

DECK_SIZE = 8
HAND_SIZE = 4


def load_kb_decks(kb_root: Path) -> list[dict]:
    """Phase 2 deck files -> [{deck_key, name, cards[8], archetype, usage}]"""
    decks = []
    for p in sorted((kb_root / "decks").glob("*.md")):
        text = p.read_text()
        cards = re.findall(r"^- \[[^\]]+\]\(\.\./cards/([^)]+)\.md\)", text, re.M)
        if len(cards) != DECK_SIZE:
            continue
        fm = dict(re.findall(r"^([a-z_]+): (.*)$", text.split("\n---\n", 1)[0].strip("-\n"), re.M))
        usage = fm.get("usage", "").strip('"').rstrip("%")
        try:
            usage_f = float(usage)
        except ValueError:
            usage_f = 1.0
        decks.append({"deck_key": fm.get("deck_key", p.stem), "name": fm.get("display_name", p.stem).strip('"'),
                      "cards": cards, "archetype": fm.get("archetype_primary", ""), "usage": usage_f})
    return decks


def load_kb_archetype_priors(kb_root: Path) -> dict[str, set[str]]:
    """archetype -> defining cards (from the archetype files' recurring-card lists)."""
    out: dict[str, set[str]] = {}
    for p in sorted((kb_root / "archetypes").glob("*.md")):
        text = p.read_text()
        sec = text.split("## Cards that recur across these decks", 1)[-1].split("## Example decks", 1)[0]
        out[p.stem] = set(re.findall(r"\(\.\./cards/([^)]+)\.md\)", sec))
    return out


@dataclass
class DeckPrediction:
    card: str
    p: float
    source: str = "kb_inference"


@dataclass
class OpponentDeckTracker:
    kb_decks: list[dict] = field(default_factory=list)
    known_cards: list[str] = field(default_factory=list)  # in order of first observation
    play_log: list[str] = field(default_factory=list)     # every observed play, in order
    cycle_confirmed: int = 0
    cycle_violations: int = 0

    # --- observation -----------------------------------------------------
    def observe_play(self, card: str | None) -> None:
        if card is None:
            self.play_log.append("?")
            return
        self.play_log.append(card)
        if card not in self.known_cards:
            if len(self.known_cards) >= DECK_SIZE:
                # 9th distinct card: something earlier was misread. Keep the
                # count honest by flagging rather than silently growing.
                self.cycle_violations += 1
                return
            self.known_cards.append(card)

    def reset(self) -> None:
        self.known_cards.clear()
        self.play_log.clear()
        self.cycle_confirmed = self.cycle_violations = 0

    @property
    def deck_complete(self) -> bool:
        return len(self.known_cards) == DECK_SIZE

    # --- cycle model -----------------------------------------------------
    def queue(self) -> list[str]:
        """Best estimate of the opponent's card queue (front = next to draw).
        Each played card goes to the back; cards never seen played sit at the
        front in unknown order."""
        q: list[str] = []
        for c in self.play_log:
            if c == "?":
                continue
            if c in q:
                q.remove(c)
            q.append(c)
        unseen = [c for c in self.known_cards if c not in q]
        return unseen + q

    def predicted_hand(self) -> list[str] | None:
        """Cards currently in the opponent's hand (the 4 played longest ago),
        only once the deck is complete. Before that the order of unseen cards
        is unknown."""
        if not self.deck_complete:
            return None
        return self.queue()[:HAND_SIZE]

    def predicted_next(self) -> str | None:
        if not self.deck_complete:
            return None
        return self.queue()[HAND_SIZE]

    def cards_until(self, card: str) -> int | None:
        """How many opponent plays until `card` is back in hand (0 = in hand
        now). None if unknown."""
        if not self.deck_complete or card not in self.known_cards:
            return None
        pos = self.queue().index(card)
        return max(0, pos - (HAND_SIZE - 1))

    def check_cycle(self, card: str) -> bool | None:
        """Call BEFORE observe_play: was `card` predicted to be in hand?"""
        hand = self.predicted_hand()
        if hand is None or card not in self.known_cards:
            return None
        ok = card in hand
        if ok:
            self.cycle_confirmed += 1
        else:
            self.cycle_violations += 1
        return ok

    # --- KB inference ----------------------------------------------------
    def rank_kb_decks(self, top: int = 5) -> list[dict]:
        """Score each KB deck by how many of the observed cards it contains,
        with a mild prior on usage. Decks contradicting an observation
        (missing an observed card) get a heavy penalty rather than zero, so
        near-variants still rank."""
        if not self.known_cards or not self.kb_decks:
            return []
        obs = set(self.known_cards)
        scored = []
        for d in self.kb_decks:
            s = set(d["cards"])
            hit = len(obs & s)
            miss = len(obs - s)
            score = hit - 2.0 * miss + 0.05 * math.log1p(d["usage"])
            scored.append((score, hit, miss, d))
        scored.sort(key=lambda x: -x[0])
        out = []
        for score, hit, miss, d in scored[:top]:
            out.append({"deck_key": d["deck_key"], "name": d["name"], "archetype": d["archetype"],
                        "matched": hit, "contradicted": miss, "score": round(score, 2),
                        "remaining": [c for c in d["cards"] if c not in obs]})
        return out

    def deck_predictions(self, top: int = 5) -> list[DeckPrediction]:
        """Probable remaining cards, as a soft-max over the top KB decks."""
        if self.deck_complete:
            return []
        ranked = self.rank_kb_decks(top=top)
        if not ranked:
            return []
        weights = [math.exp(r["score"]) for r in ranked]
        z = sum(weights)
        acc: Counter = Counter()
        for r, w in zip(ranked, weights):
            for c in r["remaining"]:
                acc[c] += w / z
        return [DeckPrediction(c, round(min(p, 0.99), 2)) for c, p in acc.most_common(DECK_SIZE - len(self.known_cards) + 4)]

    def summary(self) -> dict:
        return {"deck_known": list(self.known_cards), "deck_complete": self.deck_complete,
                "predicted_hand": self.predicted_hand(), "predicted_next": self.predicted_next(),
                "deck_predictions": [{"card": p.card, "p": p.p, "source": p.source} for p in self.deck_predictions()],
                "kb_matches": self.rank_kb_decks(top=3),
                "cycle_confirmed": self.cycle_confirmed, "cycle_violations": self.cycle_violations}

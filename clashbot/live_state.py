"""Live state builder (D-015): perception GameState + PlayEvents -> the clean `state` dict of dataset/v1/states.jsonl.

Re-implements, for one live instant, the dataset path `tools/extract_states.py::Game.build_state`
(states.jsonl branch) followed by `tools/data/build_clean_dataset.py::clean_sample`, so retrieval,
features, candidates and d1 requests see live states the way they saw training states. The small
helpers are private copies (the runtime package must not import build scripts);
tests/test_live_state.py proves they match the originals.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass, field

from clashbot.game_rules import is_playable, load_cards
from clashbot.state_text import render_context, render_threats
from clashbot.units import load_unit_map, resolve

try:  # pure class vocabulary; fall back to the literal set when cv2 is absent
    from cr_perception.detect import TOWER_CLASSES
except Exception:  # pragma: no cover
    TOWER_CLASSES = {"king-tower", "queen-tower", "cannoneer-tower", "dagger-duchess-tower",
                     "royal-chef-tower"}

RECENT_S = 12.0                     # same as tools/extract_states.py
SCHEMA_PHASES = ("single_elixir", "double_elixir", "triple_elixir", "overtime")
MAX_OPP_DECK = 8
PENDING_S = 1.5                     # a tap's result is not trusted on screen until the card has left the slot this long
SECONDS_PER_ELIXIR = {"single_elixir": 2.8, "double_elixir": 1.4, "triple_elixir": 0.93, "overtime": 0.93}
TAP_TOLERANCE = 0.5                 # elixir: a drop of >= cost - this confirms a tap; a drop of <= this means it failed
EXECUTOR_MERGE_S = 3.0              # a perception play of the same card this close to an executor play is the same play
_UNSET = object()                   # "no match seen yet" (distinct from a match_id of None)


@dataclass
class LiveConfig:
    own_deck: list[str]                       # the 8 card slugs the user plays
    hero_cards: list[str] = field(default_factory=list)
    recent_s: float = RECENT_S
    hand_hold_s: float = 2.0                  # reuse the last confident hand this long when all 4 slots are unread; 0 = off
    hand_tracker: bool = False                # positional hand memory (HandTracker) instead of the timed hold; live loop only
    opp_evidence: bool = False                # an opponent card is "seen" on summed evidence (label 1.0 / 0.6, arena 0.5), not on one play; live loop only
    strict_units: bool = False                # drop allies outside our deck (misread towers, wrong side) when the deck is valid; live loop only
    confirm_s: float = 2.5                    # tracker mode: a sent tap is followed this long (own-elixir accounting, failed-tap check)
    stale_s: float = 1.2                      # tracker mode: the HUD elixir lags our tap this long; until then min(read, expected)
    cooldown_s: float = 2.0                   # tracker mode: a card whose tap failed is hidden from the hand this long


# --------------------------------------------------------------------------- private helpers
# Copies of tools/extract_states.py and tools/data/build_clean_dataset.py (same output, same key order).

def _pad4(hand) -> list:
    hand = list(hand or [])[:4]
    return hand + [None] * (4 - len(hand))


def _towers_of(d) -> dict:
    d = d or {}
    return {"king": d.get("king"), "left": d.get("left"), "right": d.get("right")}


def _norm_phase(phase) -> str | None:
    if not phase:
        return None
    if phase in SCHEMA_PHASES:
        return phase
    if phase.endswith("_overtime") or phase == "overtime":
        return "overtime"
    return None


def _play_record(ev: dict, cards: dict) -> dict:
    card = ev.get("card")
    return {"t": round(float(ev["timestamp"]), 3), "player": ev.get("player"),
            "card": card, "tile": ev.get("tile"),
            "elixir_cost": cards[card]["elixir_cost"] if card in cards else None,
            "detect_source": ev.get("detect_source"), "confidence": ev.get("confidence")}


def _clean_plays(plays, cards: dict) -> list[dict]:
    out = []
    for p in plays or []:
        q = dict(p)
        card = q.get("card")
        if not (card and is_playable(card, cards)):
            q["card"] = None
        out.append(q)
    return out


def _unit_from_state(u: dict) -> dict:
    m = u.get("motion") or {}
    return {"unit": u.get("class") or "unknown_unit", "side": u.get("side") or "enemy",
            "tile": list(u.get("tile") or m.get("tile") or [0, 0])[:2],
            "heading": m.get("heading"), "speed_tiles_s": m.get("speed"),
            "predicted_tile_2s": m.get("pred_2s"), "eta_tower": m.get("eta_tower")}


def _clean_units(raw_units: list[dict], deck_status: str, own_deck: list[str], opp_deck: list[str],
                 cards: dict, unit_map: dict, strict: bool = False) -> list[dict]:
    """strict (live loop, valid deck only): an ally that is still not one of our cards after the side repair is a
    misread (our own tower shows up as a 'stationary' skeleton-dragons) and is dropped, `unknown_unit` allies
    included. Enemy `unknown_unit`s stay: an unknown enemy can still be a threat."""
    units = []
    own_set, opp_set = set(own_deck), set(opp_deck)
    for u in raw_units or []:
        r = resolve(u["unit"], cards, unit_map)
        side, repaired, eta = u["side"], False, u.get("eta_tower")
        if deck_status != "invalid":
            cand = set(r["candidates"])
            own_ok, opp_ok = bool(cand & own_set), bool(cand & opp_set)
            if side == "ally" and not own_ok and opp_ok:
                side, repaired = "enemy", True
            elif side == "enemy" and not opp_ok and own_ok:
                side, repaired = "ally", True
            if repaired:
                eta = None
            if strict and deck_status == "valid" and side == "ally" and not own_ok:
                continue
        units.append({"unit_class": u["unit"], "card": r["card"], "is_evo": r["is_evo"],
                      "side": side, "side_repaired": repaired, "tile": u.get("tile"),
                      "heading": u.get("heading"), "speed_tiles_s": u.get("speed_tiles_s"),
                      "predicted_tile_2s": u.get("predicted_tile_2s"), "eta_tower": eta})
    return units


def _as_json(obj) -> dict:
    return obj.to_json() if hasattr(obj, "to_json") else obj


# --------------------------------------------------------------------------- positional hand memory (FIX_LIST #26)

def _drop_last(seq: list, card) -> None:
    """Remove the last occurrence of `card` from `seq` (no-op when absent)."""
    for k in range(len(seq) - 1, -1, -1):
        if seq[k] == card:
            del seq[k]
            return


class HandTracker:
    """Positional hand memory for the live loop. A slot keeps its last read card until we play from it
    (executor-confirmed taps); greyed-out cards are therefore remembered instead of lost. With the 8-card deck
    known, the cycle (last 4 own plays = the queue, front = next card) fills a single unknown slot."""

    def __init__(self, deck: list[str] | None, pending_s: float = PENDING_S):
        self.deck = list(deck) if deck is not None and len(set(deck)) == 8 else None
        self.pending_s = pending_s
        self.reset()

    def reset(self) -> None:
        self.slots: list[str | None] = [None] * 4
        self.next: str | None = None
        self.log: list[str] = []                       # own plays we sent, oldest first
        self.pending: dict[int, tuple[str, float]] = {}  # slot -> (card played from it, t of the tap)
        self.filled: list[bool] = [False] * 4          # per slot: value NOT read in this frame (memory / inference)
        self._next_from_log = False                    # `next` came from the play log, not from a read

    def _next_from_queue(self) -> None:
        self.next = self.log[-4] if len(self.log) >= 4 else None
        self._next_from_log = True

    def note_play(self, card: str, slot: int, t: float) -> None:
        """A tap on `slot` was sent: the next card moves into it, `card` goes to the back of the queue."""
        if self.next and self.next != card:
            incoming = self.next
        else:
            incoming = self.log[-4] if len(self.log) >= 4 else None   # before the play, the queue is log[-4:]
        self.log.append(card)
        self.slots[slot] = incoming
        self.pending[slot] = (card, t)
        self._next_from_queue()

    def mark_failed(self, slot: int, card: str) -> None:
        """The elixir shows the tap on `slot` never landed: `card` is still there. Undoes `note_play`."""
        restored = False
        pend = self.pending.get(slot)
        if pend is not None and pend[0] == card:
            del self.pending[slot]
            incoming = self.slots[slot]
            if incoming is not None and incoming != card:
                self.next, restored = incoming, True   # the card note_play moved in was the next card; it still is
        self.slots[slot] = card
        for j in range(4):
            if j != slot and self.slots[j] == card:
                self.slots[j] = None                   # a card is in at most one slot
        _drop_last(self.log, card)
        if self._next_from_log and not restored:
            self._next_from_queue()

    def update(self, read: list, read_next: str | None, t: float) -> list:
        """Fold one frame's hand read (4 slots, None = unread) into the memory; returns the 4 tracked slots."""
        if read_next is not None:
            self.next = read_next
            self._next_from_log = False
        for i in range(4):
            c = read[i] if i < len(read) else None
            pend = self.pending.get(i)
            if pend is not None:
                pc, pt = pend
                age = t - pt
                if c is None:
                    if age >= self.pending_s:
                        del self.pending[i]            # the play landed; the slot stays as note_play set it
                    continue
                if c == pc:
                    if age < self.pending_s:
                        continue                       # the screen lags behind the tap
                    self.slots[i] = pc                 # the play did not land
                    _drop_last(self.log, pc)
                    del self.pending[i]
                    if self._next_from_log:
                        self._next_from_queue()
                    continue
                del self.pending[i]                    # a different card: the slot changed, read it normally
            if c is None:
                continue
            self.slots[i] = c
            for j in range(4):
                if j != i and self.slots[j] == c:
                    self.slots[j] = None               # a card is in at most one slot
            if c in self.log[-4:]:
                _drop_last(self.log, c)                # it is in the hand, so that play did not really happen
        if self.deck is not None and len(self.log) >= 4:
            queue = set(self.log[-4:])
            for i in range(4):
                if self.slots[i] in queue:
                    self.slots[i] = None               # stale: that card is in the queue
            rest = [c for c in self.deck if c not in queue and c not in self.slots]
            unknown = [i for i in range(4) if self.slots[i] is None]
            if len(unknown) == 1 and len(rest) == 1:
                self.slots[unknown[0]] = rest[0]
            if read_next is None:
                self._next_from_queue()
        self.filled = [(read[i] if i < len(read) else None) is None and self.slots[i] is not None
                       for i in range(4)]
        return list(self.slots)


# --------------------------------------------------------------------------- builder

@dataclass
class _OwnPlay:
    """A tap we sent, followed until its effect on the HUD elixir is known (tracker mode)."""
    t: float
    card: str
    slot: int
    cost: float
    elixir_at_play: float               # the own elixir the previous build() returned
    checked: bool = False               # the failed-tap check has run (once per play)
    event: dict | None = None           # the synthetic play event of this tap (dropped when the tap failed)
    ability: bool = False               # the hero ability button, not a card: no hand slot, no play event (MAC-AI)


class LiveStateBuilder:
    """Stateful: remembers the play events and when each opponent card was first seen, per match."""

    def __init__(self, config: LiveConfig, cards: dict | None = None, unit_map: dict | None = None):
        self.config = config
        self.cards = load_cards() if cards is None else cards
        self.unit_map = load_unit_map() if unit_map is None else unit_map
        self.deck = sorted({c for c in config.own_deck if is_playable(c, self.cards)})
        self.deck_status = ("valid" if len(self.deck) == 8 else
                            "incomplete" if len(self.deck) == 7 else "invalid")
        # HUD hand reads are noisy: with the 8-card deck known, anything outside it is a misread.
        self.deck_set = set(config.own_deck) if len(set(config.own_deck)) >= 8 else None
        deck8 = sorted(set(config.own_deck))
        self.tracker = (HandTracker(deck8 if len(deck8) == 8 else None) if config.hand_tracker else None)
        self._opp_deck_cls = None
        if config.opp_evidence:             # lazy: the evidence model lives with the perception deck tracker
            from cr_perception.decktracker import OpponentDeckTracker, evidence_weight
            self._opp_deck_cls, self._evidence_weight = OpponentDeckTracker, evidence_weight
        self.reset()

    def reset(self) -> None:
        """Forget the current match: events, opponent first-seen times, last match_id."""
        self._events: list[dict] = []          # normalised play-event dicts, sorted by timestamp
        self._event_keys: set[tuple] = set()
        self._opp_first: dict[str, float] = {}  # playable opponent card -> first-seen time
        self._opp_deck = self._opp_deck_cls() if self._opp_deck_cls is not None else None   # evidence per card (opp_evidence)
        self._match_id = _UNSET
        self._last_hand: tuple[float, list] | None = None  # (t, hand) of the last read with >= 2 known slots
        self.last_hand_reused = False          # the hand of the latest build() came from _last_hand / the tracker
        self._own_plays: list[_OwnPlay] = []   # taps not yet confirmed or expired (tracker mode)
        self._cooldown: dict[str, float] = {}  # card whose tap failed -> t until which it is hidden from the hand
        self._last_elixir: float | None = None  # own elixir the latest build() returned (tracker mode)
        self.failed_taps = 0                   # taps the elixir showed never landed (tracker mode)
        self.confirmed_taps = 0                # taps the elixir / the slot confirmed (tracker mode)
        if self.tracker is not None:
            self.tracker.reset()

    # -- inputs ------------------------------------------------------------
    def observe_events(self, events: list) -> None:
        for e in events or []:
            ev = dict(_as_json(e))
            ts = ev.get("timestamp")
            if ts is None:
                continue
            ev["timestamp"] = ts = float(ts)
            # the tile keeps two unidentified plays (card None) at the same instant apart
            tile = ev.get("tile")
            key = (round(ts, 1), ev.get("player"), ev.get("card"), tuple(tile) if tile else None)
            if key in self._event_keys:
                continue
            self._event_keys.add(key)
            if ev.get("tile") is not None:
                ev["tile"] = list(ev["tile"])
            if self._merge_into_executor_play(ev):
                continue
            bisect.insort(self._events, ev, key=lambda x: x["timestamp"])
            card = ev.get("card")
            if ev.get("player") == "opponent" and card and is_playable(card, self.cards):
                if self._opp_deck is None:
                    self._opp_first[card] = min(self._opp_first.get(card, ts), ts)
                else:
                    self._observe_opp(card, ev, ts)

    def _merge_into_executor_play(self, ev: dict) -> bool:
        """Tracker mode: a perception play of ours that is the tap we sent (same card, within EXECUTOR_MERGE_S of the
        nearest executor play) is that play, not a second one: it fills the executor play's missing tile and the
        executor timestamp stays. True when merged."""
        card = ev.get("card")
        if ev.get("player") != "own" or not card or ev.get("detect_source") == "executor":
            return False
        near = [e for e in self._events
                if e.get("detect_source") == "executor" and e.get("player") == "own" and e.get("card") == card
                and abs(e["timestamp"] - ev["timestamp"]) <= EXECUTOR_MERGE_S]
        if not near:
            return False
        target = min(near, key=lambda e: abs(e["timestamp"] - ev["timestamp"]))
        if target.get("tile") is None and ev.get("tile") is not None:
            target["tile"] = list(ev["tile"])
        target["_perceived"] = True                     # perception saw this play too: a failed-tap verdict cannot undo it
        return True

    def _observe_opp(self, card: str, ev: dict, ts: float) -> None:
        """Evidence mode: one weak play does not make a card "seen" (the phantom plays of a misread unit would
        fill the deck in seconds). A card is first seen when its summed evidence makes it known; a stronger
        newcomer can push the weakest known card out of the 8."""
        score = ev.get("score", ev.get("conf"))
        weight = self._evidence_weight(ev.get("detect_source"), ev.get("confidence"),
                                       float(score) if isinstance(score, (int, float)) else None)
        self._opp_deck.observe_play(card, weight)
        known = set(self._opp_deck.known_cards)
        for c in known - self._opp_first.keys():
            self._opp_first[c] = ts
        for c in self._opp_first.keys() - known:
            del self._opp_first[c]

    def note_own_play(self, card: str, slot: int | None, t: float, tile=None, cost: float | None = None) -> None:
        """The executor sent a tap that plays `card` from hand `slot` at time `t`, onto `tile` (no-op without the hand
        tracker). The play is known at once: it enters `own.recent_plays` as an "executor" event, so the tactical
        engine sees our new building before perception does. The perception event of the same play, when it comes,
        is merged into it (`observe_events`); a tap the elixir shows failed takes it out again.
        `slot=None` is the hero ability button (MAC-AI, `card` "ability:musketeer", `cost` its elixir): it only enters
        the elixir accounting (min(read, expected) while the HUD lags, the failed-tap check); it has no hand slot, no
        place in the card cycle and no play event."""
        if self.tracker is None:
            return
        ability = slot is None
        ev = None
        if not ability:
            self.tracker.note_play(card, slot, t)
            ev = {"player": "own", "card": card, "tile": list(tile)[:2] if tile is not None else None,
                  "timestamp": float(t), "detect_source": "executor"}
            bisect.insort(self._events, ev, key=lambda x: x["timestamp"])
        if cost is None and card in self.cards:
            cost = self.cards[card]["elixir_cost"]
        if cost is not None and self._last_elixir is not None:
            self._own_plays.append(_OwnPlay(float(t), card, -1 if ability else slot, float(cost), self._last_elixir,
                                            event=ev, ability=ability))

    def _settle_plays(self, elixir: float | None, hand: list, t: float, phase: str | None) -> float | None:
        """Own-elixir accounting for the taps we sent (tracker mode). The HUD reading lags our tap, so while a tap is
        younger than `stale_s` the elixir is min(read, expected), expected = at_play - cost + regen * age. Once the
        reading is trusted, a tap whose elixir never dropped failed: the tracker puts the card back and the card is
        hidden from the hand for `cooldown_s`. A tap is confirmed when the reading dropped by >= cost - 0.5 or the
        slot reads a different card. Returns the corrected elixir."""
        cfg = self.config
        regen = 1.0 / SECONDS_PER_ELIXIR.get(phase, SECONDS_PER_ELIXIR["single_elixir"])
        cap = 10.0
        kept: list[_OwnPlay] = []
        for p in self._own_plays:                                  # oldest first
            age = max(0.0, t - p.t)
            slot_card = hand[p.slot] if 0 <= p.slot < len(hand) else None
            if (elixir is not None and p.elixir_at_play - elixir >= p.cost - TAP_TOLERANCE) or \
                    (slot_card is not None and slot_card != p.card):
                self.confirmed_taps += 1
                continue
            if age >= cfg.confirm_s:
                self.confirmed_taps += int(p.checked)              # it passed the failed-tap check and nothing contradicted it
                continue
            if age >= cfg.stale_s and not p.checked and elixir is not None:
                p.checked = True
                if elixir >= p.elixir_at_play - TAP_TOLERANCE:     # no drop: the tap did not land
                    if not p.ability:                              # (the ability has no hand slot to restore)
                        self.tracker.mark_failed(p.slot, p.card)
                        self._drop_event(p.event)                  # it never happened: not in recent_plays either
                        self._cooldown[p.card] = t + cfg.cooldown_s
                    self.failed_taps += 1
                    for q in self._own_plays:                      # later taps started from an elixir that already paid for this one
                        if q is not p and 0.0 <= q.t - p.t < cfg.stale_s:
                            q.elixir_at_play = min(10.0, q.elixir_at_play + p.cost)
                    continue
            if age < cfg.stale_s:
                cap = min(cap, max(0.0, min(10.0, p.elixir_at_play - p.cost + regen * age)))
            kept.append(p)
        self._own_plays = kept
        return elixir if elixir is None else min(elixir, cap)

    def _drop_event(self, ev: dict | None) -> None:
        """Remove the executor play event `ev` (by identity) from the event list, unless perception has confirmed it."""
        if ev is None or ev.get("_perceived"):
            return
        for k, e in enumerate(self._events):
            if e is ev:
                del self._events[k]
                return

    def _foreign(self, card) -> bool:
        """The card is not one of the 8 he is playing (only judged when the deck is known)."""
        return self.deck_set is not None and card is not None and card not in self.deck_set

    def _plays(self, player: str, t: float, drop_foreign_hud: bool) -> list[dict]:
        lo = t - self.config.recent_s - 1e-9
        out = []
        for e in self._events:
            if e.get("player") != player or not (lo <= e["timestamp"] <= t):
                continue
            if drop_foreign_hud and e.get("detect_source") == "hud" and self._foreign(e.get("card")):
                continue
            out.append(_play_record(e, self.cards))
        return _clean_plays(out, self.cards)

    def _hold_hand(self, hand: list, t: float) -> list:
        """Greyed (unaffordable) cards are missed by the matcher: when all four slots read None, reuse the
        last confident hand for at most `hand_hold_s`, minus cards played since (they left the hand)."""
        known = sum(c is not None for c in hand)
        if known >= 2:
            self._last_hand = (t, list(hand))
            return hand
        hold = self.config.hand_hold_s
        if known or hold <= 0 or self._last_hand is None:
            return hand
        t_last, last = self._last_hand
        if not 0.0 <= t - t_last <= hold:
            return hand
        played = {e.get("card") for e in self._events
                  if e.get("player") == "own" and e["timestamp"] > t_last}
        reused = [None if c in played else c for c in last]
        self.last_hand_reused = any(c is not None for c in reused)
        return reused if self.last_hand_reused else hand

    # -- output ------------------------------------------------------------
    def build(self, gs) -> tuple[dict, str] | None:
        g = _as_json(gs)
        self.last_hand_reused = False
        if g.get("readiness") != "match" or not g.get("own"):
            return None
        mid = g.get("match_id")
        if self._match_id is not _UNSET and mid != self._match_id:
            self.reset()                       # new match: forget the old one's events
        self._match_id = mid
        cards = self.cards
        t = float(g["t"])
        own, opp = g.get("own") or {}, g.get("opponent") or {}

        confs = [c for c in (own.get("hand_conf") or []) if isinstance(c, (int, float))]
        elixir = own.get("elixir")
        if elixir is not None:
            elixir = max(0.0, min(10.0, float(elixir)))
        hand = [None if self._foreign(c) else c for c in _pad4(own.get("hand"))]
        hand = [c if (c and is_playable(c, cards)) else None for c in hand]
        nxt = own.get("next_card")
        nxt = None if self._foreign(nxt) else nxt
        nxt = nxt if (nxt and is_playable(nxt, cards)) else None
        phase = _norm_phase(g.get("phase"))
        if self.tracker is not None:
            elixir = self._settle_plays(elixir, hand, t, phase)
            self._last_elixir = elixir
            hand = self.tracker.update(hand, nxt, t)
            self._cooldown = {c: u for c, u in self._cooldown.items() if u > t}
            hand = [None if c in self._cooldown else c for c in hand]      # a failed tap's card is not re-tapped at once
            nxt = self.tracker.next
            self.last_hand_reused = any(self.tracker.filled)
        else:
            hand = self._hold_hand(hand, t)
        confidence = None if self.last_hand_reused else (round(sum(confs) / len(confs), 3) if confs else None)

        for c in ([] if self._opp_deck is not None else opp.get("deck_known") or []):  # the tracker's own list counts from the first t it shows
            if is_playable(c, cards):
                self._opp_first[c] = min(self._opp_first.get(c, t), t)
        seen = sorted((c for c, ts in self._opp_first.items() if ts <= t),
                      key=lambda c: (self._opp_first[c], c))[:MAX_OPP_DECK]

        raw_units = [_unit_from_state(u) for u in (g.get("units") or [])
                     if u.get("category") != "tower" and u.get("class") not in TOWER_CLASSES]
        units = _clean_units(raw_units, self.deck_status, self.deck, seen, cards, self.unit_map,
                             strict=self.config.strict_units)

        state = {
            "clock": g.get("match_clock"), "phase": phase,
            "match_seconds": g.get("match_seconds"),
            "own": {
                "elixir": elixir,
                "hand_confidence": confidence,
                "hero_cards": list(self.config.hero_cards), "towers": _towers_of(own.get("towers")),
                "hand": hand, "next_card": nxt, "deck": list(self.deck),
                "deck_status": self.deck_status,
                "recent_plays": self._plays("own", t, drop_foreign_hud=True),
            },
            "opponent": {
                "elixir_estimate": opp.get("elixir_est"), "towers": _towers_of(opp.get("towers")),
                "deck_known": seen, "deck_complete": len(seen) == MAX_OPP_DECK,
                "recent_plays": self._plays("opponent", t, drop_foreign_hud=False),
            },
            "units": units,
            "threats": render_threats(units),
        }
        ability = own.get("hero_ability")
        if isinstance(ability, dict) and ability.get("visible"):      # the button is on screen (MAC-AI); absent otherwise
            state["own"]["hero_ability"] = dict(ability)
        return state, render_context(state, t)

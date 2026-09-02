"""Play-event detection and attribution.

Own plays  : HUD-based. A hand slot changes identity AND elixir drops by that
             card's Phase 1 cost within a short window -> exact card, exact
             time, detect_source="hud".
Opponent   : arena-based. A unit track appears that no own play explains.
             Spawn side is a weak cross-check only (Miner, Goblin Barrel,
             Graveyard, spells appear on the wrong half).
Spells     : detector spell classes when available; otherwise sudden own tower
             HP drops with no enemy unit in range -> PlayEvent(card=None,
             confidence="low") so the deck tracker knows there is a hole.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import geometry as g
from .state import PlayEvent, UnitObs

OWN_MATCH_WINDOW = 1.5      # seconds between hand change and elixir drop
TRACK_DIST_TILES = 2.5      # same unit if within this many tiles between frames
TRACK_TTL = 1.0             # seconds a track survives without a detection
NEW_UNIT_GRACE = 0.8        # seconds a new enemy track must persist before it counts as a play
GROUP_WINDOW = 0.6          # seconds within which several new units of one class are one play (swarms)


@dataclass
class Track:
    id: int
    cls: str
    side: str
    tile: tuple[int, int] | None
    first_t: float
    last_t: float
    hits: int = 1
    reported: bool = False


@dataclass
class HandDiff:
    t: float
    slot: int
    before: str | None
    after: str | None


@dataclass
class PlayDetector:
    card_costs: dict[str, float]
    tracks: dict[int, Track] = field(default_factory=dict)
    next_id: int = 1
    last_hand: list[str | None] = field(default_factory=lambda: [None] * 4)
    pending_hand: list[HandDiff] = field(default_factory=list)
    last_elixir: int | None = None
    last_elixir_t: float | None = None
    own_play_cards_recent: list[tuple[float, str]] = field(default_factory=list)
    last_enemy_hp: dict[str, int] = field(default_factory=dict)
    recent_group: dict[str, float] = field(default_factory=dict)

    # ------------------------------------------------------------------ HUD
    def update_hand(self, t: float, hand: list[str | None], hand_conf: list[float], min_conf: float = 0.5) -> None:
        for i, (before, after) in enumerate(zip(self.last_hand, hand)):
            if after is not None and hand_conf[i] >= min_conf and after != before and before is not None:
                self.pending_hand.append(HandDiff(t, i, before, after))
            if after is not None and hand_conf[i] >= min_conf:
                self.last_hand[i] = after
        self.pending_hand = [p for p in self.pending_hand if t - p.t <= OWN_MATCH_WINDOW]

    DEBOUNCE = 3   # consecutive frames a new elixir value must persist

    def update_elixir(self, t: float, elixir: int | None, clock: str | None) -> list[PlayEvent]:
        """Call after update_hand. An elixir drop matched to a pending hand
        change (by cost) yields an own PlayEvent. Readings are debounced: a
        value must repeat DEBOUNCE frames before it counts, so bar-edge
        flicker never becomes a play."""
        events: list[PlayEvent] = []
        if elixir is None:
            return events
        if elixir != getattr(self, "_cand", None):
            self._cand, self._cand_n = elixir, 1
            return events
        self._cand_n += 1
        if self._cand_n < self.DEBOUNCE:
            return events
        if self.last_elixir is not None and elixir < self.last_elixir:
            drop = self.last_elixir - elixir
            # match against pending hand changes: the card that LEFT the slot
            best = None
            for p in self.pending_hand:
                cost = self.card_costs.get(p.before)
                if cost is None:
                    continue
                if abs(cost - drop) <= 0.5 or (cost <= drop and best is None):
                    best = p
                    if abs(cost - drop) <= 0.5:
                        break
            if best is not None:
                self.pending_hand.remove(best)
                cost = self.card_costs.get(best.before, drop)
                events.append(PlayEvent(t, clock, "own", best.before, None, float(self.last_elixir), float(elixir),
                                        "hud", "high" if abs(cost - drop) <= 0.5 else "medium",
                                        f"slot {best.slot}: {best.before} -> {best.after}, drop {drop}"))
                self.own_play_cards_recent.append((t, best.before))
            elif drop >= 2:
                events.append(PlayEvent(t, clock, "own", None, None, float(self.last_elixir), float(elixir),
                                        "inferred", "low", f"elixir dropped by {drop} with no readable hand change"))
            # a lone 1-elixir drop without a hand change is treated as read noise
        self.last_elixir, self.last_elixir_t = elixir, t
        self.own_play_cards_recent = [(tt, c) for tt, c in self.own_play_cards_recent if t - tt <= 3.0]
        return events

    # ---------------------------------------------------------------- arena
    def update_units(self, t: float, units: list[UnitObs], clock: str | None) -> list[PlayEvent]:
        """Associate detections to tracks; a new ENEMY track that survives the
        grace period is an opponent play."""
        events: list[PlayEvent] = []
        unmatched = []
        for u in units:
            if u.category in ("tower", "ui") or u.tile is None:
                continue
            best, best_d = None, 1e9
            for tr in self.tracks.values():
                if tr.cls != u.cls or tr.side != u.side or tr.tile is None:
                    continue
                d = abs(tr.tile[0] - u.tile[0]) + abs(tr.tile[1] - u.tile[1])
                if d < best_d:
                    best, best_d = tr, d
            if best is not None and best_d <= TRACK_DIST_TILES:
                best.tile, best.last_t, best.hits = u.tile, t, best.hits + 1
                u.track_id = best.id
            else:
                unmatched.append(u)
        for u in unmatched:
            tr = Track(self.next_id, u.cls, u.side, u.tile, t, t)
            self.tracks[tr.id] = tr
            u.track_id = tr.id
            self.next_id += 1
        # expire
        for tid in [k for k, tr in self.tracks.items() if t - tr.last_t > TRACK_TTL]:
            del self.tracks[tid]
        # report new tracks
        for tr in self.tracks.values():
            if tr.reported or t - tr.first_t < NEW_UNIT_GRACE or tr.hits < 2:
                continue
            tr.reported = True
            if tr.side == "ally":
                continue  # own plays come from the HUD
            if tr.cls == "unknown_unit":
                events.append(PlayEvent(tr.first_t, clock, "opponent", None, tr.tile, None, None, "arena", "low",
                                        "unrecognised unit appeared (weights predate this card?)"))
                continue
            from .detect import to_card_slug, categorize
            card = to_card_slug(tr.cls)
            # spawned sub-units of a card that was already reported are not plays
            key = f"{card}"
            if key in self.recent_group and tr.first_t - self.recent_group[key] <= GROUP_WINDOW:
                continue
            self.recent_group[key] = tr.first_t
            conf = "high" if categorize(tr.cls) == "spell" or tr.side == "enemy" else "medium"
            own_half = tr.tile is not None and tr.tile[1] < 15
            detail = "enemy-side track" + (" but spawned on own half (miner/barrel/spell?)" if own_half else "")
            events.append(PlayEvent(tr.first_t, clock, "opponent", card, tr.tile, None, None, "arena",
                                    "medium" if own_half else conf, detail))
        return events

    def update_tower_hp(self, t: float, own_hp: dict[str, int | None], units: list[UnitObs], clock: str | None) -> list[PlayEvent]:
        """Sudden own-tower HP drop with no enemy unit near that tower -> an
        unidentified opponent spell."""
        events: list[PlayEvent] = []
        for name, hp in own_hp.items():
            if hp is None:
                continue
            prev = self.last_enemy_hp.get(name)
            if prev is not None and hp > 0 and 150 <= prev - hp <= 1500:
                near = any(u.side == "enemy" and u.tile is not None and u.tile[1] <= 8 for u in units)
                if not near:
                    events.append(PlayEvent(t, clock, "opponent", None, None, None, None, "inferred", "low",
                                            f"{name} lost {prev - hp} HP with no enemy unit in range: unidentified spell"))
            self.last_enemy_hp[name] = hp
        return events

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
    DEBOUNCE: int = 3       # consecutive frames a new elixir value must persist
    tracks: dict[int, Track] = field(default_factory=dict)
    next_id: int = 1
    last_hand: list[str | None] = field(default_factory=lambda: [None] * 4)
    pending_hand: list[HandDiff] = field(default_factory=list)
    last_elixir: int | None = None
    last_elixir_t: float | None = None
    own_play_cards_recent: list[tuple[float, str]] = field(default_factory=list)
    last_enemy_hp: dict[str, int] = field(default_factory=dict)
    recent_group: dict[str, float] = field(default_factory=dict)
    own_hold: list[PlayEvent] = field(default_factory=list)       # own plays waiting for a deploy label (tile)
    label_tiles: dict[str, tuple] = field(default_factory=dict)  # card -> (t, tile, score, text) seen before the HUD event
    recent_labels: list[tuple[float, str, tuple[int, int]]] = field(default_factory=list)
    HOLD_SECONDS: float = 2.5
    LABEL_DEDUP_SECONDS: float = 3.0
    LABEL_DEDUP_TILES: int = 4

    # ------------------------------------------------------------------ HUD
    EMPTY_WINDOW = 12.0     # a lifted card can be held for many seconds before placement
    DROP_WINDOW = 0.45      # seconds over which the bar's drain animation is summed

    def update_hand(self, t: float, hand: list[str | None], hand_conf: list[float], min_conf: float = 0.5) -> None:
        """In this HUD a played card's slot goes BLANK for ~1.5 s, then the next
        card slides in. So the play signal is card -> None (or, for a
        direct swap, card -> other card); None -> card is the refill."""
        for i, (before, after) in enumerate(zip(self.last_hand, hand)):
            confident = after is not None and hand_conf[i] >= min_conf
            if before is not None and (after is None or (confident and after != before)):
                self.pending_hand.append(HandDiff(t, i, before, after))
            if confident:
                self.last_hand[i] = after
            elif after is None:
                self.last_hand[i] = None
        self.pending_hand = [p for p in self.pending_hand if t - p.t <= self.EMPTY_WINDOW]

    def _finalize_drop(self, t: float, clock: str | None) -> list[PlayEvent]:
        pd = self._pending_drop
        self._pending_drop = None
        total, e_before, e_after, t0 = pd["total"], pd["before"], pd["after"], pd["t0"]
        best, best_err = None, 9.0
        for p in self.pending_hand:
            cost = self.card_costs.get(p.before)
            if cost is None:
                continue
            err = abs(cost - total)
            if err < best_err:
                best, best_err = p, err
        def _tile_for(card: str) -> tuple:
            lt = self.label_tiles.pop(card, None)
            if lt and t0 - lt[0] <= 6.0:
                return lt[1], f"; deploy label '{lt[3]}' at {lt[1]} (score {lt[2]})"
            return None, ""
        if best is not None and best_err <= 0.5:
            self.pending_hand.remove(best)
            self.own_play_cards_recent.append((t0, best.before))
            tile, note = _tile_for(best.before)
            ev = PlayEvent(t0, clock, "own", best.before, tile, e_before, e_after, "hud", "high",
                           f"slot {best.slot} emptied ({best.before}), elixir -{total}{note}")
            ev.slot = best.slot
            return [ev]
        if best is not None and len(self.pending_hand) == 1:
            self.pending_hand.remove(best)
            self.own_play_cards_recent.append((t0, best.before))
            tile, note = _tile_for(best.before)
            ev = PlayEvent(t0, clock, "own", best.before, tile, e_before, e_after, "hud", "medium",
                           f"slot {best.slot} emptied ({best.before}) but elixir -{total} != cost {self.card_costs.get(best.before)}{note}")
            ev.slot = best.slot
            return [ev]
        # no slot change explains it: a deploy label seen on Ryley's side with a
        # matching cost identifies the card (the hand read was wrong or unreadable)
        for card, (lt, tile, score, text) in list(self.label_tiles.items()):
            if t0 - lt <= 6.0 and abs(self.card_costs.get(card, -9) - total) <= 0.5:
                del self.label_tiles[card]
                self.own_play_cards_recent.append((t0, card))
                ev = PlayEvent(t0, clock, "own", card, tile, e_before, e_after, "deploy_label",
                               "medium" if score >= 0.85 else "low",
                               f"deploy label '{text}' -> {card} (score {score}) at {tile} explains elixir -{total} (hand read disagreed)")
                recent_empty = [p for p in self.pending_hand if t0 - p.t <= 6.0 and p.after is None]
                if len(recent_empty) == 1:
                    ev.slot = recent_empty[0].slot
                    self.pending_hand.remove(recent_empty[0])
                return [ev]
        if total >= 2:
            return [PlayEvent(t0, clock, "own", None, None, e_before, e_after, "inferred", "low",
                              f"elixir dropped by {total} with no readable hand change")]
        return []   # a lone 1-elixir drop with no slot change is read noise

    def update_elixir(self, t: float, elixir: int | None, clock: str | None) -> list[PlayEvent]:
        """Call after update_hand. Debounced elixir drops are summed over the
        drain animation (DROP_WINDOW) and then matched to an emptied slot."""
        events: list[PlayEvent] = []
        if not hasattr(self, "_pending_drop"):
            self._pending_drop = None
        if self._pending_drop is not None and t - self._pending_drop["t_last"] > self.DROP_WINDOW:
            events += self._finalize_drop(t, clock)
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
            if self._pending_drop is None:
                self._pending_drop = {"t0": t, "t_last": t, "total": drop, "before": float(self.last_elixir), "after": float(elixir)}
            else:
                self._pending_drop.update(t_last=t, total=self._pending_drop["total"] + drop, after=float(elixir))
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
            if key in self.recent_group and tr.first_t - self.recent_group[key] <= 3.0:
                continue   # already reported (deploy label, or a swarm's first unit)
            self.recent_group[key] = tr.first_t
            conf = "high" if categorize(tr.cls) == "spell" or tr.side == "enemy" else "medium"
            own_half = tr.tile is not None and tr.tile[1] < 15
            # a troop track that STARTS deep in Ryley's half is his own unit with the
            # detector's side channel wrong (the opponent cannot deploy troops there;
            # spells/miner/drill/barrel/graveyard are the exceptions and are exempt)
            from .detect import SPELL_CLASSES
            if tr.tile is not None and tr.tile[1] < 13 and card not in SPELL_CLASSES and card not in ("miner", "goblin-drill", "graveyard", "goblin-barrel"):
                continue
            detail = "enemy-side track" + (" but spawned on own half (miner/barrel/spell?)" if own_half else "")
            events.append(PlayEvent(tr.first_t, clock, "opponent", card, tr.tile, None, None, "arena",
                                    "medium" if own_half else conf, detail))
        return events

    # ---------------------------------------------------------- deploy labels
    def hold_own(self, ev: PlayEvent) -> None:
        """Keep an own HUD play for a short time so a deploy label can attach
        its tile; release() returns the ones that timed out."""
        self.own_hold.append(ev)

    def release_holds(self, t: float) -> list[PlayEvent]:
        out = [e for e in self.own_hold if t - e.timestamp > self.HOLD_SECONDS]
        self.own_hold = [e for e in self.own_hold if t - e.timestamp <= self.HOLD_SECONDS]
        return out

    def update_labels(self, t: float, labels, tiles: list[tuple[int, int] | None], clock: str | None) -> list[PlayEvent]:
        """labels: DeployLabel list; tiles: the ground tile per label. A label
        that matches a held own play attaches the tile to it; otherwise it is
        an opponent play (the opponent's hand is invisible, so this is the
        most reliable opponent signal we have)."""
        events: list[PlayEvent] = []
        self.recent_labels = [r for r in self.recent_labels if t - r[0] <= self.LABEL_DEDUP_SECONDS]
        for lbl, tile in zip(labels, tiles):
            if tile is None:
                continue
            dup = any(c == lbl.card and abs(tt[0] - tile[0]) + abs(tt[1] - tile[1]) <= self.LABEL_DEDUP_TILES
                      for _, c, tt in self.recent_labels)
            if dup:
                continue
            own = next((e for e in self.own_hold if e.card == lbl.card), None)
            # Labels also appear while a card is being DRAGGED (placement
            # preview), and the HUD event for an own play is finalised only
            # after the elixir drain animation, so the label can arrive first.
            # Treat as Ryley's when: a slot emptied of this card is pending,
            # the card is in his hand, or the label is on his half for a
            # non-spell card (the opponent cannot deploy troops there).
            from .detect import categorize, SPELL_CLASSES
            pending_own = any(p.before == lbl.card for p in self.pending_hand)
            in_hand = lbl.card in self.last_hand
            # rows 12-14 are excluded: an opponent bridge deployment's label
            # sits above the unit and its ground point can land there
            own_half_troop = tile[1] <= 11 and lbl.card not in SPELL_CLASSES and lbl.card not in ("miner", "goblin-drill", "graveyard", "goblin-barrel", "wall-breakers")
            if own is None and (pending_own or in_hand or own_half_troop):
                prev = self.label_tiles.get(lbl.card)
                if prev is None or t - prev[0] > 6.0 or lbl.match_score >= prev[2]:
                    self.label_tiles[lbl.card] = (t, tile, lbl.match_score, lbl.text)
                self.recent_labels.append((t, lbl.card, tile))
                continue
            # a weak fuzzy read only counts if a HUD play corroborates it
            if lbl.match_score < 0.85 and own is None:
                continue
            self.recent_labels.append((t, lbl.card, tile))
            if own is not None:
                own.tile = tile
                own.detail += f"; deploy label at {tile} (score {lbl.match_score})"
                self.own_hold.remove(own)
                events.append(own)
                continue
            # own play whose HUD event already left the hold window (long drain etc.)
            if any(c == lbl.card and t - tt <= 4.0 for tt, c in self.own_play_cards_recent):
                continue
            self.recent_group[lbl.card] = t
            side_note = " on own half" if tile[1] < 15 else ""
            events.append(PlayEvent(t, clock, "opponent", lbl.card, tile, None, None, "deploy_label",
                                    "high" if lbl.match_score >= 0.9 else "medium",
                                    f"deploy label '{lbl.text}'{side_note}" + (f", lvl {lbl.level}" if lbl.level else "")))
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

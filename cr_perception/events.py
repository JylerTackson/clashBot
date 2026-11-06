"""Play-event detection and attribution.

Own plays  : HUD-based. A hand slot changes identity AND elixir drops by that
             card's Phase 1 cost within a short window -> exact card, exact
             time, detect_source="hud".
Opponent   : arena-based. A unit track appears that no own play explains.
             Spawn side is a weak cross-check only (Miner, Goblin Barrel,
             Graveyard, spells appear on the wrong half).
             An arena play first seen on our half, or a spawn-type unit next to an enemy tower, counts only
             with a deploy label of the same card; one that coincides with an own play is dropped (A3).
Spells     : detector spell classes when available; otherwise sudden own tower
             HP drops with no enemy unit in range -> PlayEvent(card=None,
             confidence="low") so the deck tracker knows there is a hole.
"""
from __future__ import annotations

from collections import Counter

from dataclasses import dataclass, field

from . import geometry as g
from .state import PlayEvent, UnitObs

OWN_MATCH_WINDOW = 1.5      # seconds between hand change and elixir drop
TRACK_DIST_TILES = 2.5      # same unit if within this many tiles between frames
TRACK_TTL = 1.0             # seconds a track survives without a detection
NEW_UNIT_GRACE = 0.8        # seconds a new enemy track must persist before it counts as a play
GROUP_WINDOW = 0.6          # seconds within which several new units of one class are one play (swarms)

# MAC-PF (D-043): phantom opponent plays
MEASURED_SINCE_SEEN = 0.05  # a unit counts as a detection (not a Kalman prediction) when last measured this recently (s)
OPP_DEDUP_SECONDS = 3.0     # an opponent play of the same card within this window and OPP_DEDUP_TILES is the same play
OPP_DEDUP_TILES = 4.0
SPAWN_NEAR_TILES = 5.0      # a spawned unit starts within this many tiles of its spawner
SPAWN_ENDED_SECONDS = 2.0   # a spawner track that ended this recently still explains new spawned units
SPAWN_PLAY_SECONDS = 4.0    # a spawner PLAY reported this recently explains new spawned units (spells)
CONT_NEAR_TILES = 1.5       # a new track this close to another enemy track is that unit under a new class
CONT_ENDED_SECONDS = 1.0
CONT_MIN_HITS = 2           # ... when that other track is a real one (measured at least this often), not a one-frame ghost

# MAC-AB A3 (FIX_LIST #31): phantom opponent cards that survive MAC-PF
PENDING_SECONDS = 2.0       # an arena play that needs a deploy label waits for one stamped within this of its first frame
PENDING_GRACE = 1.5         # ... and is kept this much longer (OCR latency: the label is applied after it is stamped)
OWN_COINCIDE_SECONDS = 1.0  # an opponent play this close in time ...
OWN_COINCIDE_TILES = 3.0    # ... and space to one of our own plays is our own unit / label misread
OWN_SPELL_SECONDS = 2.0     # an opponent spell this soon after one of ours (in flight, read as another spell) is ours
OWN_SPELL_LEAD = 0.5        # ... the HUD stamps our spell slightly after the unit-track of its flight may start
OWN_SPELL_MAX_ROW = 16      # ... anywhere on our half (row <= 14) or the river (15, 16)
OWN_PLAY_KEEP_SECONDS = 6.0
OWN_HALF_MAX_ROW = 14       # an opponent arena play whose first tile is this far back needs a deploy label
SPAWN_TYPES = {"skeletons", "goblins", "bats"}   # what tower troops and spawners next to a tower leave behind
TOWER_NEAR_TILES = 4.0
STAT_KEYS = ("dropped_own_half_unconfirmed", "dropped_spawn_near_tower_unconfirmed",
             "dropped_own_play_coincidence", "dropped_own_spell_in_flight", "confirmed_by_label")

# spawned card/unit class -> cards whose units or spells create it (slugs of knowledge_base / unit_map.json).
# A "play" of the key near a live spawner is a spawn, not an opponent deployment.
SPAWNED_BY: dict[str, set[str]] = {
    "bats": {"night-witch"},
    "skeletons": {"graveyard", "witch", "tombstone", "skeleton-barrel", "skeleton-king"},
    "goblins": {"goblin-hut", "goblin-barrel", "goblin-drill", "goblin-machine"},
    "spear-goblins": {"goblin-hut", "goblin-giant"},
    "goblin-brawler": {"goblin-cage"},
    "fire-spirit": {"furnace"},
    "barbarians": {"battle-ram", "barbarian-barrel", "barbarian-hut"},
    "royal-recruits": {"royal-delivery"},
    "golemite": {"golem"},
    "lava-pup": {"lava-hound"},
    "elixir-golem-mid": {"elixir-golem"},
    "elixir-golem-small": {"elixir-golem"},
}


def _near(a: tuple[int, int] | None, b: tuple[int, int] | None, tiles: float) -> bool:
    """Within `tiles` (euclidean); True when either tile is unknown (time-only comparison)."""
    if a is None or b is None:
        return True
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5 <= tiles


def _measured(u: UnitObs) -> bool:
    """True for a detection of this frame; False for a Kalman track that is only predicted (motion.since_seen > 0).
    Units without motion (hand-built, older callers) count as measured."""
    return u.motion is None or float(u.motion.get("since_seen", 0.0)) <= MEASURED_SINCE_SEEN


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
    first_tile: tuple[int, int] | None = None   # where it first appeared (spawn point)
    preds: list = field(default_factory=list)   # tracks it started on top of (alive / just ended): same unit, new class?

    def __post_init__(self) -> None:
        if self.first_tile is None:
            self.first_tile = self.tile


@dataclass
class PendingPlay:
    """An opponent arena play held back until a deploy label of the same card confirms it (or it expires)."""
    first_t: float
    card: str
    tile: tuple[int, int] | None
    reason: str                  # "own_half" | "spawn_near_tower"


@dataclass
class OwnPlay:
    """One of our own plays (HUD event or deploy label). A HUD event gets its tile later, from its label."""
    t: float
    card: str | None
    tile: tuple[int, int] | None = None
    ev: PlayEvent | None = None

    @property
    def where(self) -> tuple[int, int] | None:
        return self.ev.tile if self.ev is not None and self.ev.tile is not None else self.tile


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
    own_deck_seen: Counter = field(default_factory=Counter)   # HUD-confirmed own cards this match
    own_levels: Counter = field(default_factory=Counter)   # deploy-label levels of confirmed own plays
    opp_levels: Counter = field(default_factory=Counter)   # deploy-label levels of confident opponent plays
    recent_labels: list[tuple[float, str, tuple[int, int]]] = field(default_factory=list)
    ended_tracks: list[tuple[Track, float]] = field(default_factory=list)    # (expired track, time it expired), last ~2 s
    opp_plays: list[tuple[float, str, tuple[int, int] | None]] = field(default_factory=list)   # reported opponent plays (t, card, tile)
    pending: list[PendingPlay] = field(default_factory=list)      # arena plays waiting for a deploy label (A3)
    own_plays: list[OwnPlay] = field(default_factory=list)        # our recent plays, to spot our own units read as the opponent's
    stats: dict[str, int] = field(default_factory=lambda: dict.fromkeys(STAT_KEYS, 0))   # A3 counters, shown by Perception.timing_report
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
            self.own_deck_seen[best.before] += 1
            self.note_own_play(t0, best.before, ev.tile, ev)
            return [ev]
        if best is not None and len(self.pending_hand) == 1:
            self.pending_hand.remove(best)
            self.own_play_cards_recent.append((t0, best.before))
            tile, note = _tile_for(best.before)
            ev = PlayEvent(t0, clock, "own", best.before, tile, e_before, e_after, "hud", "medium",
                           f"slot {best.slot} emptied ({best.before}) but elixir -{total} != cost {self.card_costs.get(best.before)}{note}")
            ev.slot = best.slot
            self.note_own_play(t0, best.before, ev.tile, ev)
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
                self.note_own_play(t0, card, tile, ev)
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
    def _predecessors(self, tr: Track, t: float) -> list[Track]:
        """`tr` has just started: the tracks of its side it may be the same unit as under another class
        (unknown_unit -> giant, giant <-> royal-giant): within CONT_NEAR_TILES of its first tile and alive but NOT
        detected in this very frame (two boxes in one frame are two units), or ended < CONT_ENDED_SECONDS ago.
        It is a continuation (not a play) once one of them proves real, see `_continues`."""
        near = [o for o in self.tracks.values() if o is not tr and o.side == tr.side and o.last_t < t]
        near += [o for o, te in self.ended_tracks if o.side == tr.side and t - te < CONT_ENDED_SECONDS]
        return [o for o in near if o.tile is not None and _near(o.tile, tr.first_tile, CONT_NEAR_TILES)]

    @staticmethod
    def _continues(preds: list[Track]) -> bool:
        """At report time: a predecessor was measured at least CONT_MIN_HITS times. A one-frame ghost (a building
        still being placed, a misread) must not swallow the real unit that follows it."""
        return any(o.hits >= CONT_MIN_HITS for o in preds)

    def _spawned(self, tr: Track, card: str, t: float) -> bool:
        """`tr` (card `card`) is a unit created by a spawner: a track of a spawner of its side is alive or ended
        < SPAWN_ENDED_SECONDS ago within SPAWN_NEAR_TILES, or a spawner play was reported < SPAWN_PLAY_SECONDS ago
        within SPAWN_NEAR_TILES (spell spawners: graveyard, goblin barrel...)."""
        from .detect import to_card_slug
        spawners = SPAWNED_BY.get(card)
        if not spawners:
            return False
        for o in self.tracks.values():
            if (o is not tr and o.side == tr.side and to_card_slug(o.cls) in spawners
                    and _near(o.tile, tr.first_tile, SPAWN_NEAR_TILES)):
                return True
        for o, te in self.ended_tracks:
            if (o.side == tr.side and t - te < SPAWN_ENDED_SECONDS and to_card_slug(o.cls) in spawners
                    and _near(o.tile, tr.first_tile, SPAWN_NEAR_TILES)):
                return True
        return any(c in spawners and -1.5 <= tr.first_t - tp <= SPAWN_PLAY_SECONDS
                   and _near(tl, tr.first_tile, SPAWN_NEAR_TILES) for tp, c, tl in self.opp_plays)

    def _opp_duplicate(self, t: float, card: str, tile: tuple[int, int] | None) -> bool:
        """An opponent play of `card` was already reported within OPP_DEDUP_SECONDS (either way) and OPP_DEDUP_TILES."""
        return any(c == card and abs(t - tp) <= OPP_DEDUP_SECONDS and _near(tl, tile, OPP_DEDUP_TILES)
                   for tp, c, tl in self.opp_plays)

    def _log_opp_play(self, t: float, card: str | None, tile: tuple[int, int] | None) -> None:
        if card:
            self.opp_plays.append((t, card, tile))
        self.opp_plays = [p for p in self.opp_plays if t - p[0] <= 10.0]

    # ------------------------------------------------ A3: phantom cards that survive the spawner / dedup rules
    def note_own_play(self, t: float, card: str | None, tile: tuple[int, int] | None = None,
                      ev: PlayEvent | None = None) -> None:
        """Remember one of our own plays (HUD event or deploy label) for OWN_PLAY_KEEP_SECONDS."""
        self.own_plays.append(OwnPlay(t, card, tile, ev))
        self.own_plays = [o for o in self.own_plays if t - o.t <= OWN_PLAY_KEEP_SECONDS]

    def _own_coincidence(self, t: float, tile: tuple[int, int] | None, card: str | None) -> str | None:
        """Why an opponent play (card `card`, first seen at `t` on `tile`) is really one of ours.
        "near": within OWN_COINCIDE_SECONDS and OWN_COINCIDE_TILES of one of our plays (its tile known: a HUD play
        whose label has not arrived yet cannot be placed, so it never matches).
        "spell": a spell on our half or the river OWN_SPELL_SECONDS after one of our spells (a goblin barrel in
        flight, read as a barbarian barrel)."""
        from .detect import SPELL_CLASSES
        for o in self.own_plays:
            where = o.where
            if (where is not None and tile is not None and abs(t - o.t) <= OWN_COINCIDE_SECONDS
                    and _near(where, tile, OWN_COINCIDE_TILES)):
                return "near"
        if card in SPELL_CLASSES and tile is not None and tile[1] <= OWN_SPELL_MAX_ROW:
            if any(o.card in SPELL_CLASSES and -OWN_SPELL_LEAD <= t - o.t <= OWN_SPELL_SECONDS for o in self.own_plays):
                return "spell"
        return None

    def _count_own_coincidence(self, why: str) -> None:
        self.stats["dropped_own_play_coincidence" if why == "near" else "dropped_own_spell_in_flight"] += 1

    @staticmethod
    def _needs_label(tile: tuple[int, int] | None, card: str) -> str | None:
        """An arena play that only counts with a deploy label: first seen on our half, or a spawn-type unit
        (skeletons, goblins, bats) that no spawner explains first seen on the enemy half next to an enemy tower."""
        if tile is None:
            return None
        if tile[1] <= OWN_HALF_MAX_ROW:
            return "own_half"
        if card in SPAWN_TYPES:
            from .tracking import TOWER_TILES
            if any(_near(tile, TOWER_TILES[k], TOWER_NEAR_TILES) for k in ("enemy_left", "enemy_right", "enemy_king")):
                return "spawn_near_tower"
        return None

    def _expire_pending(self, now: float) -> None:
        """Drop the arena plays whose label never came (PENDING_SECONDS + OCR latency after their first frame)."""
        keep = []
        for p in self.pending:
            if now - p.first_t <= PENDING_SECONDS + PENDING_GRACE:
                keep.append(p)
                continue
            self.stats["dropped_own_half_unconfirmed" if p.reason == "own_half"
                       else "dropped_spawn_near_tower_unconfirmed"] += 1
        self.pending = keep

    def _take_pending(self, card: str, t: float) -> list[PendingPlay]:
        """Remove (and return) the waiting arena plays of `card` that a label stamped `t` confirms."""
        hit = [p for p in self.pending if p.card == card and abs(t - p.first_t) <= PENDING_SECONDS]
        self.pending = [p for p in self.pending if p not in hit]
        return hit

    def update_units(self, t: float, units: list[UnitObs], clock: str | None) -> list[PlayEvent]:
        """Associate detections to tracks; a new ENEMY track that survives the
        grace period is an opponent play. Only frames where the unit was
        measured count (a Kalman prediction is not a detection); a track that
        starts on top of another enemy track is that unit under a new class,
        and the units a spawner (night witch, graveyard...) creates are not plays."""
        events: list[PlayEvent] = []
        self._expire_pending(t)
        unmatched = []
        for u in units:
            if u.category in ("tower", "ui") or u.tile is None:
                continue
            measured = _measured(u)
            best, best_d = None, 1e9
            for tr in self.tracks.values():
                if tr.cls != u.cls or tr.side != u.side or tr.tile is None:
                    continue
                d = abs(tr.tile[0] - u.tile[0]) + abs(tr.tile[1] - u.tile[1])
                if d < best_d:
                    best, best_d = tr, d
            if best is not None and best_d <= TRACK_DIST_TILES:
                if measured:
                    best.tile, best.last_t, best.hits = u.tile, t, best.hits + 1
                u.track_id = best.id
            elif measured:
                unmatched.append(u)
        fresh = []
        for u in unmatched:
            tr = Track(self.next_id, u.cls, u.side, u.tile, t, t)
            fresh.append(tr)
            u.track_id = tr.id
            self.next_id += 1
        for tr in fresh:        # judged against the tracks of earlier frames only
            tr.preds = self._predecessors(tr, t)
        for tr in fresh:
            self.tracks[tr.id] = tr
        # expire
        for tid in [k for k, tr in self.tracks.items() if t - tr.last_t > TRACK_TTL]:
            self.ended_tracks.append((self.tracks.pop(tid), t))
        self.ended_tracks = [e for e in self.ended_tracks if t - e[1] < max(SPAWN_ENDED_SECONDS, CONT_ENDED_SECONDS)]
        # report new tracks
        for tr in self.tracks.values():
            if tr.reported or t - tr.first_t < NEW_UNIT_GRACE or tr.hits < 2:
                continue
            tr.reported = True
            preds, tr.preds = tr.preds, []          # judged once; drop the references so old tracks can go
            if tr.side == "ally":
                continue  # own plays come from the HUD
            if self._continues(preds):
                continue  # the same unit read under another class (flicker), not a new deployment
            if tr.cls == "unknown_unit":
                events.append(PlayEvent(tr.first_t, clock, "opponent", None, tr.tile, None, None, "arena", "low",
                                        "unrecognised unit appeared (weights predate this card?)"))
                continue
            from .detect import to_card_slug, categorize
            card = to_card_slug(tr.cls)
            if self._spawned(tr, card, t):
                continue  # created by a spawner (bats of a night witch, skeletons of a graveyard...)
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
            # one of our own plays read as the opponent's (our unit with the side channel wrong, our spell in flight)
            why = self._own_coincidence(tr.first_t, tr.first_tile, card)
            if why is not None:
                self._count_own_coincidence(why)
                continue
            # our half / a spawn-type unit next to an enemy tower: only with a deploy label of the same card
            need = self._needs_label(tr.first_tile, card)
            if need is not None:
                self.pending.append(PendingPlay(tr.first_t, card, tr.first_tile, need))
                continue
            detail = "enemy-side track" + (" but spawned on own half (miner/barrel/spell?)" if own_half else "")
            self._log_opp_play(tr.first_t, card, tr.tile)
            events.append(PlayEvent(tr.first_t, clock, "opponent", card, tr.tile, None, None, "arena",
                                    "medium" if own_half else conf, detail))
        return events

    # ---------------------------------------------------------- deploy labels
    def hold_own(self, ev: PlayEvent) -> None:
        """Keep an own HUD play for a short time so a deploy label can attach
        its tile; release() returns the ones that timed out."""
        self.own_hold.append(ev)

    def release_holds(self, t: float) -> list[PlayEvent]:
        self._expire_pending(t)
        out = [e for e in self.own_hold if t - e.timestamp > self.HOLD_SECONDS]
        self.own_hold = [e for e in self.own_hold if t - e.timestamp <= self.HOLD_SECONDS]
        # a label credited to Ryley (his half / his deck / his level) that no HUD
        # event claimed within the window is still his play: the hand reader
        # missed the slot change. Emit it rather than lose the play.
        for card, (lt, tile, score, text) in list(self.label_tiles.items()):
            if t - lt > 6.0:
                del self.label_tiles[card]
                if score >= 0.85 and not any(c == card and abs(tt - lt) <= 6.0 for tt, c in self.own_play_cards_recent):
                    self.own_play_cards_recent.append((lt, card))
                    out.append(PlayEvent(lt, None, "own", card, tile, None, None, "deploy_label", "medium",
                                         f"deploy label '{text}' at {tile} (score {score}); no HUD slot change read (hand read missed it)"))
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
            own_lvl, opp_lvl = self.known_levels()
            # card levels: Ryley's cards share one level (16 at the top of
            # ladder); once both players' levels are known and differ, the
            # "lvl N" line under the label attributes the play by itself
            lvl_own = lbl.level is not None and own_lvl is not None and opp_lvl is not None and lbl.level == own_lvl and lbl.level != opp_lvl
            lvl_opp = lbl.level is not None and own_lvl is not None and opp_lvl is not None and lbl.level == opp_lvl and lbl.level != own_lvl
            if (pending_own or in_hand) and lbl.level is not None:
                self.own_levels[lbl.level] += 1
            # rows 12-14 are excluded: an opponent bridge deployment's label
            # sits above the unit and its ground point can land there
            own_half_troop = tile[1] <= 11 and lbl.card not in SPELL_CLASSES and lbl.card not in ("miner", "goblin-drill", "graveyard", "goblin-barrel", "wall-breakers")
            deck_card = self.own_deck_seen.get(lbl.card, 0) >= 2 and not lvl_opp
            if own is None and (pending_own or in_hand or (own_half_troop and not lvl_opp) or lvl_own or deck_card):
                self.note_own_play(t, lbl.card, tile)
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
            # the arena path may have reported this deployment already (the label is read later than the unit track)
            if self._opp_duplicate(t, lbl.card, tile):
                continue
            why = self._own_coincidence(t, tile, lbl.card)
            if why is not None:
                self._take_pending(lbl.card, t)      # the arena track it would have confirmed is ours too
                self._count_own_coincidence(why)
                continue
            confirmed = self._take_pending(lbl.card, t)   # one event: the label's, with the arena track folded in
            if confirmed:
                self.stats["confirmed_by_label"] += 1
            self.recent_group[lbl.card] = t
            self._log_opp_play(t, lbl.card, tile)
            if lbl.level is not None and lbl.match_score >= 0.9 and tile[1] >= 17 and not in_hand:
                self.opp_levels[lbl.level] += 1
            side_note = " on own half" if tile[1] < 15 else ""
            events.append(PlayEvent(t, clock, "opponent", lbl.card, tile, None, None, "deploy_label",
                                    "high" if lbl.match_score >= 0.9 else "medium",
                                    f"deploy label '{lbl.text}'{side_note}" + (f", lvl {lbl.level}" if lbl.level else "")
                                    + (f"; confirms the arena track at {confirmed[0].tile}" if confirmed else "")))
        return events

    def known_levels(self) -> tuple[int | None, int | None]:
        """(own card level, opponent card level) once each is seen >= 3 times
        with a clear majority; None otherwise."""
        def mode(c: Counter) -> int | None:
            if sum(c.values()) < 3:
                return None
            lvl, n = c.most_common(1)[0]
            return lvl if n >= 0.7 * sum(c.values()) else None
        return mode(self.own_levels), mode(self.opp_levels)

    def update_tower_hp(self, t: float, own_hp: dict[str, int | None], units: list[UnitObs], clock: str | None) -> list[PlayEvent]:
        """Sudden own-tower HP drop with no enemy unit near that tower -> an
        unidentified opponent spell."""
        events: list[PlayEvent] = []
        if not hasattr(self, "_hp_pending"):
            self._hp_pending = {}
        for name, hp in own_hp.items():
            if hp is None:
                continue
            prev = self.last_enemy_hp.get(name)
            pend = self._hp_pending.pop(name, None)
            if pend is not None:
                # a drop counts only if the next reading stays down (the OCR
                # can oscillate between two values and fake a spell each time)
                p_t, p_prev, p_hp, p_clock = pend
                if abs(hp - p_hp) <= 100 and hp < p_prev - 100:
                    near = any(u.side == "enemy" and u.tile is not None and u.tile[1] <= 8 for u in units)
                    if not near:
                        events.append(PlayEvent(p_t, p_clock, "opponent", None, None, None, None, "inferred", "low",
                                                f"{name} lost {p_prev - p_hp} HP with no enemy unit in range: unidentified spell"))
                    self.last_enemy_hp[name] = hp
                    continue
                if abs(hp - p_prev) <= 100:      # bounced back: OCR flicker
                    self.last_enemy_hp[name] = hp
                    continue
            if prev is not None and hp > 0 and 150 <= prev - hp <= 1500:
                self._hp_pending[name] = (t, prev, hp, clock)
                continue                          # do not move last_enemy_hp until confirmed
            self.last_enemy_hp[name] = hp
        return events

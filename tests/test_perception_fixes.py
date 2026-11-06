"""MAC-PF (D-043): perception fixes for the live loop.

1. opponent plays counted several times (label after arena, spawned units, class flicker, predicted positions,
   a match joined late), 2. the opponent deck admitted on one weak observation, 3. our towers read as ally units,
4. evolution art and the own-deck restriction of the hand reader.
Synthetic UnitObs / label events / frames only; no model is loaded and nothing touches data/templates.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clashbot.live_state import LiveConfig, LiveStateBuilder, _clean_units  # noqa: E402
from clashbot.units import load_unit_map  # noqa: E402
from cr_perception import Perception  # noqa: E402
from cr_perception.config import Calibration  # noqa: E402
from cr_perception.decktracker import (EVIDENCE_KNOWN, EVIDENCE_REPLACE, OpponentDeckTracker,  # noqa: E402
                                       evidence_weight)
from cr_perception.detect import Detection, KataCRDetector  # noqa: E402
from cr_perception.elixir_sim import late_join_start  # noqa: E402
from cr_perception.events import (PENDING_GRACE, PENDING_SECONDS, SPAWNED_BY, STAT_KEYS,  # noqa: E402
                                  PlayDetector)
from cr_perception.hud import DEFAULT_ROIS, CardMatcher, HandReader  # noqa: E402
from cr_perception.labels import DeployLabel  # noqa: E402
from cr_perception.screen import Readiness  # noqa: E402
from cr_perception.sources import FrameSource  # noqa: E402
from cr_perception.state import PlayEvent, UnitObs  # noqa: E402
from cr_perception.tracking import TOWER_TILES  # noqa: E402
from tests.test_hud_synthetic import IMAGES, synthetic_hud  # noqa: E402
from tests.test_live_state import CARDS, DECK, OPP_CARDS, builder, make_gs  # noqa: E402
from tests.test_pipeline_synthetic import arena_frame  # noqa: E402

KB = ROOT / "knowledge_base"
EVO = KB / "evolutions" / "images"
DT = 0.3                                                   # seconds between two detector passes


# ============================================================================ 1. opponent plays
def U(cls, side="enemy", tile=(9, 20), since=None, category="unit") -> UnitObs:
    """A unit as the arena path sees it; `since` = seconds since the track was last MEASURED (None = no motion)."""
    motion = None if since is None else {"since_seen": since}
    return UnitObs(cls, side, tile, 0.9, (0, 0, 10, 10), None, category, motion)


def run(pd: PlayDetector, script, t_end: float, t0: float = 0.0) -> list[PlayEvent]:
    """Feed `script(t) -> list[UnitObs]` to update_units every DT seconds; returns all events."""
    events, k = [], 0
    while t0 + k * DT <= t_end + 1e-9:
        t = round(t0 + k * DT, 3)
        events += pd.update_units(t, script(t), "2:00")
        k += 1
    return events


def cards(events) -> list[str | None]:
    return [e.card for e in events if e.player == "opponent"]


def test_spawned_by_covers_the_named_spawners():
    assert SPAWNED_BY["bats"] == {"night-witch"}
    assert {"graveyard", "witch", "tombstone", "skeleton-barrel", "skeleton-king"} <= SPAWNED_BY["skeletons"]
    assert {"goblin-hut", "goblin-barrel", "goblin-drill", "goblin-machine"} <= SPAWNED_BY["goblins"]
    assert {"goblin-hut", "goblin-giant"} <= SPAWNED_BY["spear-goblins"]
    assert SPAWNED_BY["fire-spirit"] == {"furnace"}
    assert {"battle-ram", "barbarian-barrel"} <= SPAWNED_BY["barbarians"]
    assert SPAWNED_BY["royal-recruits"] == {"royal-delivery"}


def test_bats_of_a_night_witch_are_not_plays():
    def script(t):
        units = [U("night-witch", tile=(13, 24))]
        if 5.0 <= t <= 7.0 or 10.0 <= t <= 12.0:                      # bats leave the witch at t+5 and t+10
            units.append(U("bat", tile=(13, 22)))
        return units
    events = run(PlayDetector({}), script, 16.0)
    assert cards(events) == ["night-witch"]


def test_bats_far_from_every_night_witch_are_a_play():
    def script(t):
        units = [U("night-witch", tile=(2, 26))]
        if 5.0 <= t <= 7.0:
            units.append(U("bat", tile=(9, 19)))             # (A3: not within 4 tiles of an enemy tower, which would need a label)
        return units
    assert cards(run(PlayDetector({}), script, 9.0)) == ["night-witch", "bats"]


def test_units_of_an_ended_spawner_track_are_still_spawned():
    """The witch is gone (killed) 1 s before the last bats appear: the track ended < 2 s ago."""
    def script(t):
        units = [U("night-witch", tile=(13, 24))] if t <= 4.0 else []
        if 5.0 <= t <= 7.0:
            units.append(U("bat", tile=(13, 21)))
        return units
    assert cards(run(PlayDetector({}), script, 9.0)) == ["night-witch"]


def test_skeletons_after_a_graveyard_play_are_not_plays():
    pd = PlayDetector({})
    pd.opp_plays.append((0.0, "graveyard", (9, 22)))                # a graveyard play reported at t = 0
    near = run(pd, lambda t: [U("skeleton", tile=(9, 20))] if 3.0 <= t <= 5.0 else [], 7.0)
    assert cards(near) == []
    late = PlayDetector({})
    late.opp_plays.append((-2.0, "graveyard", (9, 22)))             # 5 s before the skeletons: too old to explain them
    assert cards(run(late, lambda t: [U("skeleton", tile=(9, 20))] if 3.0 <= t <= 5.0 else [], 7.0)) == ["skeletons"]
    far = PlayDetector({})
    far.opp_plays.append((0.0, "graveyard", (2, 22)))               # graveyard on the other side of the arena
    assert cards(run(far, lambda t: [U("skeleton", tile=(14, 18))] if 3.0 <= t <= 5.0 else [], 7.0)) == ["skeletons"]


def test_graveyard_track_then_skeletons_is_one_play():
    def script(t):
        units = [U("graveyard", tile=(9, 22), category="spell")] if t <= 1.5 else []
        if 2.0 <= t <= 4.0:
            units.append(U("skeleton", tile=(9, 20)))
        return units
    assert cards(run(PlayDetector({}), script, 7.0)) == ["graveyard"]


def test_a_giant_label_after_the_arena_track_is_the_same_play():
    pd = PlayDetector({})
    arena = run(pd, lambda t: [U("giant")], 1.5)
    assert cards(arena) == ["giant"]
    lbl = DeployLabel("giant", "Giant", 0.95, 0.9, (0, 0, 1, 1), level=15)
    assert pd.update_labels(2.5, [lbl], [(9, 20)], "1:57") == []     # 1 s after the track started, same tile


def test_a_label_far_away_or_much_later_is_another_play():
    lbl = DeployLabel("giant", "Giant", 0.95, 0.9, (0, 0, 1, 1), level=15)
    pd = PlayDetector({})
    run(pd, lambda t: [U("giant")], 1.5)
    assert [e.card for e in pd.update_labels(2.5, [lbl], [(2, 22)], "1:57")] == ["giant"]      # 7 tiles away
    pd = PlayDetector({})
    run(pd, lambda t: [U("giant", tile=(9, 20))] if t <= 1.5 else [], 1.5)
    assert [e.card for e in pd.update_labels(8.0, [lbl], [(9, 20)], "1:51")] == ["giant"]      # 7 s later


def test_label_before_arena_and_label_twice_stay_one_play():
    lbl = DeployLabel("giant", "Giant", 0.95, 0.9, (0, 0, 1, 1), level=15)
    pd = PlayDetector({})
    assert [e.card for e in pd.update_labels(1.0, [lbl], [(9, 20)], "1:57")] == ["giant"]
    assert cards(run(pd, lambda t: [U("giant")] if t >= 1.2 else [], 4.0)) == []      # the arena track is the same play


def test_class_flicker_unknown_unit_then_giant_is_one_play():
    def script(t):
        return [U("unknown_unit")] if t <= 1.5 else [U("giant")]
    assert len(cards(run(PlayDetector({}), script, 4.0))) == 1


def test_class_flicker_giant_and_royal_giant_alternating_is_one_play():
    def script(t):
        return [U("giant" if round(t / DT) % 2 == 0 else "royal-giant", tile=(9, 20))]
    events = run(PlayDetector({}), script, 6.0)
    assert len(cards(events)) == 1


def test_a_one_frame_ghost_does_not_swallow_the_real_unit_next_to_it():
    """Smoke run: a building being placed was first read as a magic-archer for one detection, then as the
    elixir-collector it is. The ghost never became a play, so the building must."""
    def script(t):
        units = []
        if t <= 2.0:
            units.append(U("magic-archer", tile=(5, 26), since=0.0 if t == 0.0 else round(t, 2)))
        if t >= 0.6:
            units.append(U("elixir-collector", tile=(5, 27)))
        return units
    assert cards(run(PlayDetector({}), script, 4.0)) == ["elixir-collector"]


def test_a_new_card_next_to_a_live_unit_in_the_same_frame_is_still_a_play():
    """Two boxes in one frame are two units: a musketeer dropped behind the giant is a play."""
    def script(t):
        units = [U("giant", tile=(9, 20))]
        if t >= 3.0:
            units.append(U("musketeer", tile=(9, 19)))
        return units
    assert cards(run(PlayDetector({}), script, 6.0)) == ["giant", "musketeer"]


def test_a_unit_seen_once_and_then_only_predicted_is_not_a_play():
    def script(t):
        return [U("giant", since=0.0 if t == 0.0 else round(t, 2))]     # measured at t = 0, Kalman-predicted afterwards
    pd = PlayDetector({})
    assert run(pd, script, 2.4) == []
    assert all(tr.hits == 1 for tr in pd.tracks.values())


def test_a_unit_measured_on_every_pass_is_a_play_and_predictions_do_not_add_hits():
    pd = PlayDetector({})
    events = run(pd, lambda t: [U("giant", since=0.0)], 2.0)
    assert cards(events) == ["giant"]
    mixed = PlayDetector({})
    mixed.update_units(0.0, [U("giant", since=0.0)], "2:00")
    for k in range(1, 4):
        mixed.update_units(k * DT, [U("giant", since=0.3 * k)], "2:00")
    assert mixed.tracks[1].hits == 1                                 # three predicted passes added nothing
    mixed.update_units(4 * DT, [U("giant", since=0.0)], "2:00")
    assert mixed.tracks[1].hits == 2


def test_units_without_motion_count_as_measured():
    assert cards(run(PlayDetector({}), lambda t: [U("giant")], 2.0)) == ["giant"]


# ---------------------------------------------------------------------------- Perception-level (late join, towers)
class NullSource(FrameSource):
    fps = 10.0

    def frames(self):
        return iter(())


class FakeOcr:
    def __init__(self, clock="2:43"):
        self.clock = clock

    def read(self, img, allow="0123456789:"):
        return self.clock, 0.9


class ScriptDetector:
    """detect() returns Detection objects built from (cls, side, (col, row)) positions through the perception's homography."""

    def __init__(self, items):
        self.items, self.p = items, None

    def detect(self, content, arena_crop):
        out = []
        for cls, side, (c, r) in self.items:
            px, py = self.p.H.tile_to_pixel(c, r, centre=False)
            out.append(Detection(cls, 0.9, (px - 8, py - 26, px + 8, py), side, 1.0, cls, "unit"))
        return out


FRAME = arena_frame(synthetic_hud(["knight", "fireball", "musketeer", "ice-spirit"], "hog-rider", 7))


@pytest.fixture
def make_perception(tmp_path, monkeypatch):
    calib = Calibration(source={"type": "synthetic", "frame_size": [576, 1024]}, rois=dict(DEFAULT_ROIS))
    calib.arena_corners_frac = [[0.072, 0.769], [0.922, 0.769], [0.905, 0.079], [0.089, 0.079]]
    cfg = tmp_path / "calib.json"
    calib.save(cfg)
    monkeypatch.setattr("cr_perception.perception.assess", lambda *a, **k: Readiness("match", 1.0, {}))
    monkeypatch.setattr("cr_perception.perception.REPO", tmp_path)   # learned_dir = tmp/data/templates: the repo's are neither read nor written

    def make(detector=None, clock="2:43", **kw) -> Perception:
        p = Perception(cfg, NullSource(), detector=detector, use_ocr=False, ENTER_FRAMES=1, **kw)
        p.ocr = FakeOcr(clock)
        assert p.learned_dir == tmp_path / "data" / "templates"
        if isinstance(detector, ScriptDetector):
            detector.p = p
        return p
    return make


def test_late_join_start_formula():
    assert late_join_start(163) == 10.0                            # 2:43 left: 17 s in -> 5 + 6.07 -> capped
    assert late_join_start(180) == 5.0
    assert late_join_start(178) == pytest.approx(5.0 + 2 / 2.8)
    assert late_join_start(120) == 10.0
    assert late_join_start(0) == 10.0


def test_first_clock_read_of_2_43_starts_the_opponent_simulation_at_10(make_perception, monkeypatch):
    p = make_perception(clock="2:43")
    s = p.process(FRAME, 0.0, 0)
    assert s.match_clock == "2:43"
    assert p.opp_sim.elixir == 10.0
    assert s.opponent["elixir_est"] == 10.0
    assert s.opponent["elixir_conf"] <= 0.2                         # a guess until the first opponent play corrects it
    s = p.process(FRAME, 0.1, 1)
    assert s.opponent["elixir_conf"] <= 0.2
    # the first counted opponent play lowers the estimate and restores the confidence
    played = []

    def one_play(self, t, units, clock):
        if not played:
            played.append(t)
            return [PlayEvent(t, clock, "opponent", "giant", (9, 20), None, None, "arena", "high", "")]
        return []
    monkeypatch.setattr(PlayDetector, "update_units", one_play)
    p.detector = ScriptDetector([])
    p.detector.p = p
    for k in range(2, 8):
        s = p.process(FRAME, k * 0.1, k)
    assert s.opponent["elixir_est"] < 6.0                           # 10 - 5 (giant) + a little regeneration
    assert s.opponent["elixir_conf"] > 0.2


def test_a_match_read_from_the_start_is_not_inflated(make_perception):
    p = make_perception(clock="2:59")
    p.process(FRAME, 0.0, 0)
    assert p.opp_sim.elixir == pytest.approx(5.0 + 1 / 2.8, abs=0.01)


def test_the_late_join_guess_is_made_once_per_match(make_perception):
    p = make_perception(clock="2:43")
    p.process(FRAME, 0.0, 0)
    p.opp_sim.spend(4)
    p.process(FRAME, 0.1, 1)
    assert p.opp_sim.elixir < 7.0                                   # not reset to 10 by the second clock read
    p.reset_match(1.0)
    assert not p._clock_seen and not p._opp_conf_low


# ============================================================================ 2. opponent deck on evidence
def test_evidence_weights():
    assert evidence_weight("deploy_label", score=0.92) == 1.0
    assert evidence_weight("deploy_label", score=0.9) == 1.0
    assert evidence_weight("deploy_label", score=0.87) == 0.6
    assert evidence_weight("deploy_label", confidence="high") == 1.0
    assert evidence_weight("deploy_label", confidence="medium") == 0.6
    assert evidence_weight("arena", confidence="high") == 0.5
    assert evidence_weight("arena") == 0.5


def test_one_arena_play_does_not_make_a_card_known():
    t = OpponentDeckTracker()
    t.observe_play("giant", 0.5)
    assert t.known_cards == [] and t.evidence["giant"] == 0.5


def test_two_arena_plays_make_it_known():
    t = OpponentDeckTracker()
    t.observe_play("giant", 0.5)
    t.observe_play("giant", 0.5)
    assert t.known_cards == ["giant"]
    assert t.summary()["deck_known"] == ["giant"]


def test_one_strong_label_makes_it_known_and_a_medium_label_does_not():
    t = OpponentDeckTracker()
    t.observe_play("giant", evidence_weight("deploy_label", score=0.92))
    t.observe_play("miner", evidence_weight("deploy_label", score=0.87))
    assert t.known_cards == ["giant"]
    t.observe_play("miner", evidence_weight("deploy_label", score=0.87))
    assert t.known_cards == ["giant", "miner"]                      # 0.6 + 0.6


def test_known_cards_keep_first_known_order():
    t = OpponentDeckTracker()
    t.observe_play("b", 0.5)
    t.observe_play("a", 1.0)
    t.observe_play("b", 0.5)
    assert t.known_cards == ["a", "b"]


EIGHT = ["c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8"]


def _full(strong: dict[str, float] | None = None) -> OpponentDeckTracker:
    t = OpponentDeckTracker()
    for c in EIGHT:
        t.observe_play(c, 1.0)
    for c, w in (strong or {}).items():
        t.observe_play(c, w)
    return t


def test_a_ninth_card_with_strong_evidence_replaces_the_weakest():
    t = _full({"c1": 1.0, "c2": 1.0, "c3": 1.0, "c4": 1.0, "c5": 1.0, "c6": 1.0})   # c7, c8 stay at 1.0
    assert len(t.known_cards) == 8
    for _ in range(3):
        t.observe_play("new", 0.5)                                  # 0.5 and 1.0 are below 1.5: not yet
        if len(t.evidence) and t.evidence["new"] < EVIDENCE_REPLACE:
            assert "new" not in t.known_cards
    assert t.evidence["new"] == pytest.approx(1.5)
    assert "new" in t.known_cards and len(t.known_cards) == 8
    assert "c8" not in t.known_cards                                # weakest, latest-known on a tie
    assert t.known_cards[:-1] == ["c1", "c2", "c3", "c4", "c5", "c6", "c7"] and t.known_cards[-1] == "new"


def test_a_ninth_card_does_not_replace_a_card_with_more_evidence():
    t = _full({c: 2.0 for c in EIGHT})                              # every known card has 3.0
    for _ in range(4):
        t.observe_play("new", 0.5)                                  # 2.0 < 3.0
    assert "new" not in t.known_cards and t.known_cards == EIGHT


def test_a_ninth_card_below_the_replace_threshold_is_ignored():
    t = _full()
    t.observe_play("new", 1.0)
    assert t.known_cards == EIGHT and EVIDENCE_KNOWN <= 1.0 < EVIDENCE_REPLACE


def test_replaced_card_leaves_the_cycle_queue():
    t = _full()
    for c in EIGHT:
        t.observe_play(c, 0.0)                                      # a full cycle in the log
    for _ in range(3):
        t.observe_play("new", 0.5)
    assert "new" in t.known_cards and len(t.queue()) == 8 and set(t.queue()) == set(t.known_cards)


def test_without_a_weight_the_old_behaviour_is_unchanged():
    t = OpponentDeckTracker()
    for c in EIGHT + ["c9"]:
        t.observe_play(c)
    assert t.known_cards == EIGHT and t.cycle_violations == 1 and t.deck_complete and not t.weighted


# ---------------------------------------------------------------------------- the live state builder
def opp_ev(ts, card, source="arena", conf="high", tile=(9, 20)) -> PlayEvent:
    return PlayEvent(ts, "1:50", "opponent", card, tile, None, None, source, conf, "")


def known(b: LiveStateBuilder, t: float, **kw) -> list[str]:
    return b.build(make_gs(t, **kw))[0]["opponent"]["deck_known"]


def test_live_config_flags_default_off():
    c = LiveConfig(own_deck=list(DECK))
    assert c.opp_evidence is False and c.strict_units is False


def test_default_builder_admits_a_card_on_one_play():
    b = builder()
    b.observe_events([opp_ev(10.0, "miner")])
    assert known(b, 20.0) == ["miner"]


def test_builder_one_arena_play_is_not_known_two_are():
    b = builder(opp_evidence=True)
    b.observe_events([opp_ev(10.0, "miner")])
    assert known(b, 20.0) == []
    b.observe_events([opp_ev(15.0, "miner", tile=(3, 22))])
    assert known(b, 20.0) == ["miner"]
    assert known(b, 12.0) == []                                      # first seen when it became known (t = 15)


def test_builder_one_strong_label_is_known_at_once():
    b = builder(opp_evidence=True)
    b.observe_events([opp_ev(10.0, "miner", "deploy_label", "high")])
    assert known(b, 10.0) == ["miner"]
    b.observe_events([opp_ev(12.0, "ronin", "deploy_label", "medium")])
    assert known(b, 20.0) == ["miner"]                               # 0.6 only


def test_builder_ignores_the_trackers_deck_known_in_evidence_mode():
    legacy, ev = builder(), builder(opp_evidence=True)
    assert known(legacy, 20.0, opp_known=["miner"]) == ["miner"]
    assert known(ev, 20.0, opp_known=["miner"]) == []


def test_builder_ninth_card_replaces_the_weakest_known_card():
    b = builder(opp_evidence=True)
    cards8 = OPP_CARDS[:8]
    b.observe_events([opp_ev(1.0 + k, c, "deploy_label", "high") for k, c in enumerate(cards8)])
    assert known(b, 20.0) == sorted(cards8, key=cards8.index)
    nine = OPP_CARDS[8]
    b.observe_events([opp_ev(30.0, nine, tile=(2, 20)), opp_ev(34.0, nine, tile=(3, 21)), opp_ev(38.0, nine, tile=(4, 20))])
    deck = known(b, 40.0)
    assert len(deck) == 8 and deck[-1] == nine and cards8[-1] not in deck


def test_builder_reset_forgets_the_evidence():
    b = builder(opp_evidence=True)
    b.observe_events([opp_ev(10.0, "miner", "deploy_label", "high")])
    assert known(b, 20.0) == ["miner"]
    assert b.build(make_gs(30.0, match_id=2))[0]["opponent"]["deck_known"] == []


# ============================================================================ 3. towers read as ally units
UNIT_MAP = load_unit_map()


def raw(unit, side="ally", tile=(14, 4)) -> dict:
    return {"unit": unit, "side": side, "tile": list(tile), "heading": None, "speed_tiles_s": None,
            "predicted_tile_2s": None, "eta_tower": None}


def test_clean_units_strict_drops_allies_outside_the_deck():
    rows = [raw("skeleton-dragon"), raw("axe", tile=(3, 5)), raw("unknown_unit", tile=(9, 25)),
            raw("archer", tile=(8, 9)), raw("unknown_unit", "enemy", (9, 25)), raw("giant", "enemy", (9, 20))]
    deck = ["archers", "berserker", "bomb-tower", "goblin-barrel", "magic-archer", "royal-delivery", "tornado", "wall-breakers"]
    strict = _clean_units(rows, "valid", deck, [], CARDS, UNIT_MAP, strict=True)
    assert [(u["unit_class"], u["side"]) for u in strict] == [("archer", "ally"), ("unknown_unit", "enemy"), ("giant", "enemy")]
    lax = _clean_units(rows, "valid", deck, [], CARDS, UNIT_MAP)
    assert [u["unit_class"] for u in lax] == [r["unit"] for r in rows]


def test_clean_units_strict_keeps_side_repaired_units_and_needs_a_valid_deck():
    deck = ["archers", "berserker", "bomb-tower", "goblin-barrel", "magic-archer", "royal-delivery", "tornado", "wall-breakers"]
    # an ally skeleton-dragon whose card the opponent has played is an enemy with the wrong side: repaired, kept
    out = _clean_units([raw("skeleton-dragon")], "valid", deck, ["skeleton-dragons"], CARDS, UNIT_MAP, strict=True)
    assert [(u["side"], u["side_repaired"]) for u in out] == [("enemy", True)]
    for status in ("incomplete", "invalid"):                         # the deck is not trusted: nothing is dropped
        assert len(_clean_units([raw("skeleton-dragon")], status, deck[:7], [], CARDS, UNIT_MAP, strict=True)) == 1


def test_builder_strict_units_flag():
    dragon = UnitObs("skeleton-dragon", "ally", (14, 4), 0.9, (0, 0, 1, 1), 1, "unit", {"tile": [14, 4]})
    archer = UnitObs("archer", "ally", (8, 9), 0.9, (0, 0, 1, 1), 2, "unit", {"tile": [8, 9]})
    gs = lambda: make_gs(100.0, units=[dragon, archer])
    assert [u["unit_class"] for u in builder().build(gs())[0]["units"]] == ["skeleton-dragon", "archer"]
    assert [u["unit_class"] for u in builder(strict_units=True).build(gs())[0]["units"]] == ["archer"]


def test_weak_tower_detection_stays_a_tower():
    import torch

    class Net:
        names = {0: "queen-tower", 1: "giant"}

        def __call__(self, x):
            pred = torch.zeros(1, 4 + 2 + 1, 2)                     # 2 anchors: 4 box (xywh) + 2 class scores + belonging
            pred[0, :4, 0] = torch.tensor([100.0, 100.0, 40.0, 40.0])
            pred[0, 4, 0] = 0.3                                     # a weak queen-tower, ally
            pred[0, :4, 1] = torch.tensor([200.0, 200.0, 40.0, 40.0])
            pred[0, 5, 1] = 0.3                                     # a weak giant, ally
            return pred

    d = KataCRDetector.__new__(KataCRDetector)
    d.torch, d.models, d.names = torch, [Net()], [Net.names]
    d.imgsz, d.conf, d.iou, d.unknown_conf, d.device, d.last_ms = 320, 0.5, 0.6, 0.25, "cpu", 0.0
    out = {x.raw_cls: x for x in d.detect(np.zeros((320, 320, 3), np.uint8))}
    assert out["queen-tower"].cls == "queen-tower" and out["queen-tower"].category == "tower"
    assert out["giant"].cls == "unknown_unit" and out["giant"].category == "unit"


def _tower_run(make_perception, items, **kw):
    det = ScriptDetector(items)
    p = make_perception(det, clock="2:30", **kw)
    states = [p.process(FRAME, k * 0.1, k) for k in range(12)]
    return p, states[-1]


DECK8 = ["knight", "fireball", "zap", "ram-rider", "musketeer", "electro-dragon", "princess", "tesla"]
OWN_LEFT = TOWER_TILES["own_left"]


def test_unknown_unit_on_a_tower_centre_is_not_tracked_but_a_hog_rider_beside_it_is(make_perception):
    items = [("unknown_unit", "ally", OWN_LEFT), ("hog-rider", "enemy", (OWN_LEFT[0] + 1.8, OWN_LEFT[1]))]
    for kw in ({}, {"own_deck": DECK8}):                            # unknown_unit goes with or without a deck
        _, s = _tower_run(make_perception, items, **kw)
        assert [u.cls for u in s.units] == ["hog-rider"], kw


def test_non_deck_class_on_a_tower_is_dropped_only_with_a_deck(make_perception):
    items = [("skeleton-dragon", "ally", (OWN_LEFT[0] + 0.3, OWN_LEFT[1] + 0.2))]
    _, s = _tower_run(make_perception, items)
    assert [u.cls for u in s.units] == ["skeleton-dragon"]          # no deck: only unknown_unit is judged
    _, s = _tower_run(make_perception, items, own_deck=DECK8)
    assert s.units == []


def test_deck_class_on_a_tower_is_kept(make_perception):
    items = [("princess", "ally", (OWN_LEFT[0] + 0.3, OWN_LEFT[1] + 0.2))]
    _, s = _tower_run(make_perception, items, own_deck=DECK8)
    assert [u.cls for u in s.units] == ["princess"]


def test_a_destroyed_tower_no_longer_filters_its_centre(make_perception):
    det = ScriptDetector([("unknown_unit", "enemy", TOWER_TILES["enemy_left"])])
    p = make_perception(det, clock="2:30")
    p.towers.enemy_left = False
    assert not p._tower_phantom(type("D", (), {"category": "unit", "cls": "unknown_unit", "pos_f": TOWER_TILES["enemy_left"]})())
    assert p._tower_phantom(type("D", (), {"category": "unit", "cls": "unknown_unit", "pos_f": TOWER_TILES["enemy_right"]})())


# ============================================================================ 4. hand art
def paste(img: np.ndarray, key: str, art: np.ndarray) -> np.ndarray:
    H, W = img.shape[:2]
    x, y, w, h = DEFAULT_ROIS[key]
    x0, y0, x1, y1 = int(x * W), int(y * H), int((x + w) * W), int((y + h) * H)
    img[y0:y1, x0:x1] = cv2.resize(art, (x1 - x0, y1 - y0))
    return img


def art(slug: str, folder: Path = IMAGES, suffix: str = "") -> np.ndarray:
    return cv2.imread(str(folder / f"{slug}{suffix}.png"))


BASE_HAND = ["fireball", "zap", "musketeer", "tesla"]


def test_evolution_art_is_a_template_of_the_base_card():
    plain, with_evo = CardMatcher(IMAGES), CardMatcher(IMAGES, extra_dirs=[EVO])
    evo = art("knight", EVO, "-evolution")
    assert with_evo.slugs.count("knight") > plain.slugs.count("knight")
    assert plain.match(evo)[0][0] != "knight"                        # the wiki art alone does not know it
    best, score = with_evo.match(evo)[0]
    assert best == "knight" and score > 0.95
    assert not any(s.endswith("-evolution") for s in with_evo.slugs)


def test_knight_evolution_in_a_hand_slot_reads_knight():
    img = synthetic_hud(["knight"] + BASE_HAND[1:], "hog-rider", 7)
    paste(img, "hand_0", art("knight", EVO, "-evolution"))
    out = HandReader(CardMatcher(IMAGES, extra_dirs=[EVO]), DEFAULT_ROIS).read(img)
    assert out["hand"][0] == "knight" and out["hand_conf"][0] > 0.45
    plain = HandReader(CardMatcher(IMAGES), DEFAULT_ROIS).read(img)
    assert plain["hand"][0] != "knight"


def test_perception_loads_the_evolution_art_by_default(make_perception):
    p = make_perception()
    assert p.matcher.slugs.count("knight") >= 2 and "knight-evolution" not in p.matcher.slugs


def _blur(a: np.ndarray, sigma: float, seed: int = 1) -> np.ndarray:
    a = cv2.GaussianBlur(cv2.resize(a, (48, 56)), (0, 0), sigma)
    a = cv2.convertScaleAbs(a, alpha=0.8, beta=20)
    return np.clip(a + np.random.default_rng(seed).normal(0, 18, a.shape), 0, 255).astype(np.uint8)


def test_deck_restriction_picks_the_best_card_of_the_deck():
    m = CardMatcher(IMAGES)
    degraded = _blur(art("ram-rider"), 5.0)                          # in-game art that matches the wiki art badly
    assert m.match(degraded)[0][0] != "ram-rider"
    assert m.match(degraded, allowed=set(DECK8))[0][0] == "ram-rider"
    assert all(s in DECK8 for s, _ in m.match(degraded, top=5, allowed=set(DECK8)))


def test_ram_rider_slot_with_the_deck_restriction_reads_ram_rider():
    img = synthetic_hud(["knight", "ram-rider", "musketeer", "tesla"], "princess", 8)
    paste(img, "hand_1", _blur(art("ram-rider"), 3.0))
    out = HandReader(CardMatcher(IMAGES), DEFAULT_ROIS, allowed=set(DECK8)).read(img)
    assert out["hand"][1] == "ram-rider"
    assert out["hand"][0] == "knight" and out["next_card"] == "princess"


def test_a_card_outside_the_deck_is_not_returned_with_the_restriction():
    img = synthetic_hud(["knight", "golem", "musketeer", "tesla"], "princess", 8)
    free = HandReader(CardMatcher(IMAGES), DEFAULT_ROIS).read(img)
    assert free["hand"][1] == "golem"
    out = HandReader(CardMatcher(IMAGES), DEFAULT_ROIS, allowed=set(DECK8)).read(img)
    assert out["hand"][1] != "golem"
    assert all(c is None or c in DECK8 for c in out["hand"] + [out["next_card"]])


def test_no_deck_leaves_the_matcher_unrestricted():
    m = CardMatcher(IMAGES)
    a = art("golem")
    assert m.match(a, allowed=None) == m.match(a) == m.match(a, allowed=set())
    assert m.match(a, allowed={"not-a-card"}) == m.match(a)          # a set no template belongs to is ignored


def test_perception_passes_the_deck_to_the_hand_reader(make_perception):
    assert make_perception().hand_reader.allowed is None
    assert make_perception(own_deck=DECK8).hand_reader.allowed == set(DECK8)


# ---------------------------------------------------------------------------- learned templates
def _touch(folder: Path, names: list[str]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for n in names:
        cv2.imwrite(str(folder / n), np.full((40, 30, 3), 90, np.uint8))


def test_learned_files_take_the_last_12_of_each_kind(tmp_path):
    d = tmp_path / "knight"
    _touch(d, [f"vid_{k:03d}.0.png" for k in range(15)] + [f"live_{k:03d}.0.png" for k in range(14)])
    files = [p.name for p in CardMatcher.learned_files(d)]
    assert len(files) == 24
    assert files[:12] == [f"vid_{k:03d}.0.png" for k in range(3, 15)]
    assert files[12:] == [f"live_{k:03d}.0.png" for k in range(2, 14)]
    assert CardMatcher.learned_files(tmp_path / "missing") == []


def test_matcher_loads_live_crops_next_to_a_full_set_of_video_crops(tmp_path):
    d = tmp_path / "templates" / "knight"
    _touch(d, [f"vid_{k:03d}.0.png" for k in range(12)] + ["live_100.0.png", "live_101.0.png"])
    m = CardMatcher(IMAGES, learned_dir=tmp_path / "templates")
    assert m.slugs.count("knight") == CardMatcher(IMAGES).slugs.count("knight") + 14


def test_live_crops_are_written_when_the_video_cap_is_full(make_perception, tmp_path):
    p = make_perception()
    card_dir = p.learned_dir / "knight"
    _touch(card_dir, [f"vid_{k:03d}.0.png" for k in range(12)])
    crop = np.random.default_rng(0).integers(0, 255, (40, 30, 3), np.uint8)
    p._slot_crops = [(9.0, [crop] * 4)]
    ev = PlayEvent(10.0, "2:00", "own", "knight", (8, 8), None, None, "hud", "high", "")
    ev.slot = 0
    before = len(p.matcher.templates)
    p._learn_template(ev)
    assert (card_dir / "live_10.0.png").exists() and len(p.matcher.templates) == before + 1
    for k in range(11):                                              # up to the live cap of its own
        ev.timestamp = 20.0 + k
        p._slot_crops = [(19.0 + k, [crop] * 4)]
        p._learn_template(ev)
    assert len(list(card_dir.glob("live_*.png"))) == 12
    ev.timestamp = 99.0
    p._slot_crops = [(98.0, [crop] * 4)]
    p._learn_template(ev)
    assert len(list(card_dir.glob("live_*.png"))) == 12 and len(list(card_dir.glob("vid_*.png"))) == 12


# ============================================================================ A3 (MAC-AB, FIX_LIST #31): phantom cards that remain
# 1. our half needs a deploy label, 2. coincidence with our own plays, 3. spawn-type units next to an enemy tower
def L(card, score=0.95, level=None) -> DeployLabel:
    return DeployLabel(card, card, score, 0.9, (0, 0, 1, 1), level=level)


def opp_events(events) -> list[PlayEvent]:
    return [e for e in events if e.player == "opponent"]


def own_half_tornado(t):
    return [U("tornado", tile=(13, 5), category="spell")]


def test_stats_start_at_zero_and_are_in_the_perception_timing_report(make_perception):
    assert PlayDetector({}).stats == dict.fromkeys(STAT_KEYS, 0)
    assert make_perception().timing_report()["play_stats"] == dict.fromkeys(STAT_KEYS, 0)


# ---- 1. our half needs a label
def test_own_half_arena_play_without_a_label_is_dropped_and_counted():
    pd = PlayDetector({})
    events = run(pd, own_half_tornado, 1.5)
    assert events == [] and len(pd.pending) == 1                       # held, not emitted
    pd.release_holds(1.5 + PENDING_SECONDS + PENDING_GRACE)             # nothing confirmed it in time
    assert pd.pending == [] and pd.stats["dropped_own_half_unconfirmed"] == 1
    assert pd.opp_plays == []


def test_own_half_arena_play_with_a_label_one_second_later_is_exactly_one_event():
    pd = PlayDetector({})
    events = run(pd, own_half_tornado, 1.2)                             # first seen at t = 0
    events += pd.update_labels(1.0, [L("tornado")], [(13, 5)], "2:00")
    assert [(e.card, e.detect_source) for e in opp_events(events)] == [("tornado", "deploy_label")]
    assert pd.pending == [] and pd.stats["confirmed_by_label"] == 1 and pd.stats["dropped_own_half_unconfirmed"] == 0
    assert "confirms the arena track" in events[0].detail
    assert run(pd, own_half_tornado, 6.0, t0=1.5) == []                 # the same unit, tracked on: no second play


def test_label_first_then_the_own_half_arena_track_is_still_one_event():
    pd = PlayDetector({})
    events = pd.update_labels(0.2, [L("tornado")], [(13, 5)], "2:00")
    events += run(pd, lambda t: own_half_tornado(t) if t >= 0.4 else [], 4.0)
    assert [e.card for e in opp_events(events)] == ["tornado"]
    assert pd.stats["dropped_own_half_unconfirmed"] == 0


@pytest.mark.parametrize("label_t, card, score, label_emits", [
    (1.0, "tornado", 0.80, 0),          # a weak fuzzy read does not count
    (1.0, "the-log", 0.95, 1),          # another card (an event of its own, but not this one's confirmation)
    (3.5, "tornado", 0.95, 1),          # 3.5 s after the track appeared: another play
])
def test_a_weak_wrong_card_or_late_label_does_not_confirm(label_t, card, score, label_emits):
    pd = PlayDetector({})
    assert run(pd, own_half_tornado, 1.2) == [] and len(pd.pending) == 1
    got = pd.update_labels(label_t, [L(card, score)], [(13, 5)], "2:00")
    assert len(got) == label_emits and pd.stats["confirmed_by_label"] == 0 and len(pd.pending) == 1
    pd.release_holds(8.0)
    assert pd.pending == [] and pd.stats["dropped_own_half_unconfirmed"] == 1


def test_enemy_half_arena_play_needs_no_label():
    assert cards(run(PlayDetector({}), lambda t: [U("knight", tile=(9, 22))], 1.5)) == ["knight"]


# ---- 2. coincidence with our own plays
def test_barbarian_barrel_one_second_after_our_goblin_barrel_is_dropped():
    pd = PlayDetector({})
    pd.note_own_play(10.0, "goblin-barrel", (3, 20))                    # our barrel, thrown at their left princess
    assert run(pd, lambda t: [U("barbarian-barrel", tile=(3, 13), category="spell")], 12.5, t0=11.0) == []
    assert pd.stats["dropped_own_spell_in_flight"] == 1                 # first seen 1 s after ours, over our half
    assert pd.pending == [] and pd.stats["dropped_own_half_unconfirmed"] == 0     # dropped for what it is, not left waiting


def test_an_opponent_spell_far_from_our_spells_in_time_or_place_is_not_the_rule():
    old = PlayDetector({})
    old.note_own_play(5.0, "goblin-barrel", (3, 20))                    # 6 s before: long landed
    run(old, lambda t: [U("barbarian-barrel", tile=(3, 13), category="spell")], 12.5, t0=11.0)
    assert old.stats["dropped_own_spell_in_flight"] == 0 and len(old.pending) == 1
    deep = PlayDetector({})
    deep.note_own_play(10.0, "goblin-barrel", (3, 20))
    assert cards(run(deep, lambda t: [U("fireball", tile=(9, 24), category="spell")], 12.5, t0=11.0)) == ["fireball"]
    assert deep.stats["dropped_own_spell_in_flight"] == 0               # on their half: a real play
    troop = PlayDetector({})
    troop.note_own_play(10.0, "goblin-barrel", (3, 20))
    assert cards(run(troop, lambda t: [U("knight", tile=(9, 22))], 12.5, t0=11.0)) == ["knight"]     # not a spell


def test_a_labelled_opponent_spell_after_our_spell_is_dropped_too():
    pd = PlayDetector({})
    pd.note_own_play(10.0, "goblin-barrel", (3, 20))
    assert pd.update_labels(11.0, [L("barbarian-barrel")], [(3, 13)], "2:00") == []
    assert pd.stats["dropped_own_spell_in_flight"] == 1
    assert [e.card for e in pd.update_labels(14.5, [L("barbarian-barrel")], [(3, 13)], "2:00")] == ["barbarian-barrel"]


def test_an_opponent_play_half_a_second_after_ours_at_two_tiles_is_dropped():
    pd = PlayDetector({})
    pd.note_own_play(10.0, "golem", (9, 20))
    assert run(pd, lambda t: [U("golem", tile=(10, 21))], 12.0, t0=10.5) == []     # sqrt(2) tiles away
    assert pd.stats["dropped_own_play_coincidence"] == 1


def test_the_same_play_at_five_tiles_or_a_second_and_a_half_later_is_kept():
    far = PlayDetector({})
    far.note_own_play(10.0, "golem", (9, 20))
    assert cards(run(far, lambda t: [U("golem", tile=(14, 20))], 12.0, t0=10.5)) == ["golem"]
    late = PlayDetector({})
    late.note_own_play(10.0, "golem", (9, 20))
    assert cards(run(late, lambda t: [U("golem", tile=(10, 21))], 13.0, t0=11.5)) == ["golem"]
    assert far.stats["dropped_own_play_coincidence"] == late.stats["dropped_own_play_coincidence"] == 0


def test_an_own_play_without_a_known_tile_cannot_match():
    pd = PlayDetector({})
    pd.note_own_play(10.0, "golem")                                     # a HUD play whose label has not come yet
    assert cards(run(pd, lambda t: [U("golem", tile=(10, 21))], 12.0, t0=10.5)) == ["golem"]


def test_a_labelled_opponent_play_next_to_ours_is_dropped_and_a_distant_one_is_kept():
    pd = PlayDetector({})
    pd.note_own_play(10.0, "giant", (9, 20))
    assert pd.update_labels(10.5, [L("golem")], [(10, 21)], "2:00") == []
    assert pd.stats["dropped_own_play_coincidence"] == 1
    assert [e.card for e in pd.update_labels(10.5, [L("golem")], [(15, 24)], "2:00")] == ["golem"]


def test_an_own_deploy_label_is_remembered_as_one_of_our_plays():
    pd = PlayDetector({})
    pd.last_hand = ["knight", None, None, None]                         # the card is in our hand: the label is ours
    assert pd.update_labels(5.0, [L("knight")], [(9, 14)], "2:00") == []
    assert [(o.t, o.card, o.where) for o in pd.own_plays] == [(5.0, "knight", (9, 14))]
    assert run(pd, lambda t: [U("knight", tile=(10, 14))], 7.0, t0=5.5) == []      # our unit, side channel wrong
    assert pd.stats["dropped_own_play_coincidence"] == 1


def hud_play(pd: PlayDetector, card: str, cost: int, t: float) -> PlayEvent:
    """Our HUD play the way Perception feeds it: the slot empties, the elixir bar drops by the card's cost."""
    pd.update_hand(t, [card, None, None, None], [0.9] * 4)
    pd.update_hand(t + 0.1, [None, None, None, None], [0.9] * 4)
    evs: list[PlayEvent] = []
    for k in range(3):
        evs += pd.update_elixir(t + 0.1 * k, 8, "2:00")
    for k in range(3, 6):
        evs += pd.update_elixir(t + 0.1 * k, 8 - cost, "2:00")
    evs += pd.update_elixir(t + 1.5, 8 - cost, "2:00")
    return [e for e in evs if e.player == "own"][0]


def test_a_hud_play_is_remembered_and_takes_its_tile_from_the_label():
    pd = PlayDetector({"goblin-barrel": 3.0})
    ev = hud_play(pd, "goblin-barrel", 3, 10.0)                         # stamped 10.5 (the debounced drop)
    assert ev.card == "goblin-barrel" and [o.card for o in pd.own_plays] == ["goblin-barrel"]
    assert pd.own_plays[0].where is None
    pd.hold_own(ev)
    pd.update_labels(11.6, [L("goblin-barrel")], [(3, 20)], "2:00")      # the label attaches the tile to the held event
    assert pd.own_plays[0].where == (3, 20)
    # ... and an opponent barbarian-barrel over our half right after is that barrel in flight
    assert run(pd, lambda t: [U("barbarian-barrel", tile=(3, 13), category="spell")], 13.5, t0=11.9) == []
    assert pd.stats["dropped_own_spell_in_flight"] == 1


# ---- 3. spawn-type units next to an enemy tower
def test_skeletons_next_to_the_enemy_king_without_a_label_are_dropped():
    pd = PlayDetector({})
    assert run(pd, lambda t: [U("skeleton", tile=(9, 26))], 1.5) == [] and len(pd.pending) == 1
    pd.release_holds(1.5 + PENDING_SECONDS + PENDING_GRACE)
    assert pd.pending == [] and pd.stats["dropped_spawn_near_tower_unconfirmed"] == 1
    assert pd.stats["dropped_own_half_unconfirmed"] == 0


def test_skeletons_next_to_the_enemy_king_with_a_label_are_one_play():
    pd = PlayDetector({})
    events = run(pd, lambda t: [U("skeleton", tile=(9, 26))], 1.5)
    events += pd.update_labels(1.2, [L("skeletons")], [(9, 26)], "2:00")
    assert [e.card for e in opp_events(events)] == ["skeletons"] and pd.stats["confirmed_by_label"] == 1


@pytest.mark.parametrize("cls, card", [("goblin", "goblins"), ("bat", "bats")])
def test_goblins_and_bats_near_an_enemy_princess_tower_need_a_label_too(cls, card):
    pd = PlayDetector({})
    assert run(pd, lambda t: [U(cls, tile=(14, 21))], 1.5) == [] and [p.card for p in pd.pending] == [card]


def test_skeletons_away_from_the_towers_and_other_cards_near_them_keep_the_old_rule():
    assert cards(run(PlayDetector({}), lambda t: [U("skeleton", tile=(9, 19))], 1.5)) == ["skeletons"]
    assert cards(run(PlayDetector({}), lambda t: [U("knight", tile=(9, 26))], 1.5)) == ["knight"]


def test_skeletons_of_a_spawner_near_the_tower_are_still_spawned_not_pending():
    def script(t):
        units = [U("tombstone", tile=(9, 25), category="building")]
        if t >= 3.0:
            units.append(U("skeleton", tile=(9, 26)))
        return units
    pd = PlayDetector({})
    assert cards(run(pd, script, 6.0)) == ["tombstone"] and pd.pending == []

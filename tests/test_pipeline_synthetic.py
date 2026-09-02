"""End-to-end plumbing test on a synthetic match: a HUD composited from the
Phase 1 card art over a drawn arena, with a scripted own play (hand slot
changes + elixir drops by the card's cost) and a menu frame. No detector."""
import json
from pathlib import Path

import cv2
import numpy as np

from cr_perception import Perception
from cr_perception.config import Calibration
from cr_perception.hud import DEFAULT_ROIS
from cr_perception.sources import FrameSource
from tests.test_hud_synthetic import synthetic_hud

KB = Path(__file__).resolve().parents[1] / "knowledge_base"


def arena_frame(hud: np.ndarray) -> np.ndarray:
    """Paint a green field into the arena ROI so readiness says 'match'."""
    H, W = hud.shape[:2]
    x, y, w, h = DEFAULT_ROIS["arena"]
    hud[int(y * H):int((y + h) * H), int(x * W):int((x + w) * W)] = (60, 150, 70)
    return hud


class ScriptedSource(FrameSource):
    fps = 10.0

    def frames(self):
        hand = ["knight", "fireball", "musketeer", "ice-spirit"]
        i = 0
        # 0-1.9 s: menu (no purple bar, no field)
        for _ in range(20):
            img = np.full((1024, 576, 3), (90, 90, 90), np.uint8)
            cv2.putText(img, "MENU", (150, 500), cv2.FONT_HERSHEY_DUPLEX, 3, (255, 255, 255), 5)
            yield img, i / self.fps, i
            i += 1
        # 2-4.9 s: match, elixir 7, static hand
        for _ in range(30):
            yield arena_frame(synthetic_hud(hand, "hog-rider", 7)), i / self.fps, i
            i += 1
        # 5 s: knight (3) played: slot 0 becomes hog-rider, elixir 7 -> 4, next = cannon
        hand = ["hog-rider", "fireball", "musketeer", "ice-spirit"]
        for _ in range(30):
            yield arena_frame(synthetic_hud(hand, "cannon", 4)), i / self.fps, i
            i += 1


def test_pipeline_on_synthetic_match(tmp_path):
    calib = Calibration(source={"type": "synthetic", "frame_size": [576, 1024]}, rois=dict(DEFAULT_ROIS))
    calib.arena_corners_frac = [[0.072, 0.769], [0.922, 0.769], [0.905, 0.079], [0.089, 0.079]]
    cfg = tmp_path / "calib.json"
    calib.save(cfg)
    p = Perception(cfg, ScriptedSource(), detector=None, use_ocr=False)
    states = list(p.states())
    assert [s.readiness for s in states[:20]] == ["menu"] * 20
    match_states = [s for s in states if s.readiness == "match"]
    assert len(match_states) == 60
    assert match_states[5].own["elixir"] == 7 and match_states[5].own["hand"] == ["knight", "fireball", "musketeer", "ice-spirit"]
    assert match_states[-1].own["elixir"] == 4 and match_states[-1].own["hand"][0] == "hog-rider"
    assert match_states[-1].own["next_card"] == "cannon"
    events = p.drain_events()
    own = [e for e in events if e.player == "own"]
    assert len(own) == 1, [e.to_json() for e in events]
    ev = own[0]
    assert ev.card == "knight" and ev.detect_source == "hud" and ev.confidence == "high"
    assert ev.elixir_before == 7 and ev.elixir_after == 4
    # own simulator was charged the play and the drift stays measured
    assert p.own_sim.plays == 1 and p.own_sim.drift_stats()["n"] > 0
    # legal mask hash present and state serialises
    js = json.loads(match_states[-1].dumps())
    assert js["legal_mask_hash"] and js["opponent"]["deck_complete"] is False

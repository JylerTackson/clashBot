"""MAC-AI part 1: the hero ability button (Hero Musketeer's Trusty Turret) - detection (synthetic frames and the recorded
shots), the executor tap and its live guard, the option key / text / candidate, the live-loop routing and the elixir
accounting. Fakes only: no model, no screen, no input."""
from __future__ import annotations

import sys
import types
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clashbot import d1_format  # noqa: E402
from clashbot import executor as ex_mod  # noqa: E402
from clashbot.actions import (ABILITY_COST, ABILITY_KEY, ability_key, is_ability_key, option_text,  # noqa: E402
                              placement_tile)
from clashbot.candidates import (ABILITY_NOTE, AbilityMove, Candidate, CandidateGenerator,  # noqa: E402
                                 hero_ability_move)
from clashbot.config import HOLD_KEY  # noqa: E402
from clashbot.executor import (LIVE_ENV, DryRunExecutor, MirroringExecutor, ability_button_point,  # noqa: E402
                               content_to_screen, screen_map_from_calib)
from clashbot.game_rules import load_cards  # noqa: E402
from clashbot.live import top_source  # noqa: E402
from clashbot.ranker import RankResult  # noqa: E402
from clashbot.state_text import render_context  # noqa: E402
from cr_perception import Perception  # noqa: E402
from cr_perception.config import Calibration  # noqa: E402
from cr_perception.hud import ABILITY_DIR, DEFAULT_ROIS, AbilityButtonReader  # noqa: E402
from cr_perception.sources import FrameSource  # noqa: E402
from tests.test_live_loop import (PRIOR, FakeBuilder, FakeClock, FakeKeyRanker, FakePerception, FakeSource,  # noqa: E402
                                  _loop, _state)
from tests.test_live_state import FULL_HAND, acct, build_at, make_gs, unit_obs  # noqa: E402
from tests.test_pipeline_synthetic import arena_frame  # noqa: E402
from tests.test_hud_synthetic import synthetic_hud  # noqa: E402
from tests.test_tactics_integration import FakeIndex, FakeKG  # noqa: E402

CARDS = load_cards()
SHOT = ROOT / "runs" / "poc" / "20261106-170722" / "shots"
MIRROR_CONTENT = [0.02531645569620253, 0.05459770114942529, 0.9493670886075949, 0.9339080459770115]   # runs/poc/mirror_calib.json

RIGHT, LEFT = (0.900, 0.776), (0.100, 0.776)     # button centres, content fractions


# -- synthetic frames -------------------------------------------------------------------------------------------------
def make_content(w: int = 600, h: int = 1300, seed: int = 0) -> np.ndarray:
    """A grass-like textured content image (no button)."""
    rng = np.random.default_rng(seed)
    img = np.clip(np.full((h, w, 3), (70, 150, 95), np.int16) + rng.integers(-16, 16, (h, w, 1)), 0, 255).astype(np.uint8)
    for gy in range(0, h, 36):
        cv2.line(img, (0, gy), (w, gy), (55, 120, 70), 2)
    for gx in range(0, w, 36):
        cv2.line(img, (gx, 0), (gx, h), (55, 120, 70), 2)
    return img


def paste_button(frame: np.ndarray, center, name: str = "musketeer", flip: bool = False, scale: float = 1.0,
                 dx: float = 0.0, dy: float = 0.0, blur: float = 0.8) -> np.ndarray:
    """Alpha-blend an asset template (BGRA, drawn for a 600 px wide content) so that its disc centre lands on `center`
    (content fractions) plus (dx, dy) px; `scale` is on top of the content-width scale."""
    H, W = frame.shape[:2]
    tpl = cv2.imread(str(ABILITY_DIR / f"{name}.png"), cv2.IMREAD_UNCHANGED)
    if flip:
        tpl = np.ascontiguousarray(tpl[:, ::-1])
    k = W / 600.0 * scale
    tpl = cv2.resize(tpl, None, fx=k, fy=k, interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_LINEAR)
    if blur:
        tpl = cv2.GaussianBlur(tpl, (0, 0), blur)
    disc_x = (tpl.shape[1] / k - 60.0 if flip else 60.0) * k           # the disc centre inside the 112 x 104 template
    x0 = int(round(center[0] * W + dx - disc_x))
    y0 = int(round(center[1] * H + dy - 52.0 * k))
    h, w = tpl.shape[:2]
    out = frame.copy()
    roi = out[y0:y0 + h, x0:x0 + w].astype(np.float32)
    a = (tpl[..., 3:4] / 255.0)
    out[y0:y0 + h, x0:x0 + w] = (tpl[..., :3] * a + roi * (1 - a)).astype(np.uint8)
    return out


def reader(**kw) -> AbilityButtonReader:
    return AbilityButtonReader(DEFAULT_ROIS, **kw)


# -- 1. perception: detection -----------------------------------------------------------------------------------------
def test_default_rois_are_centred_on_the_button_and_mirrored():
    r, l = DEFAULT_ROIS["hero_ability_right"], DEFAULT_ROIS["hero_ability_left"]
    assert (r[0] + r[2] / 2, r[1] + r[3] / 2) == pytest.approx(RIGHT, abs=1e-6)
    assert (l[0] + l[2] / 2, l[1] + l[3] / 2) == pytest.approx(LEFT, abs=1e-6)
    assert r[2:] == l[2:] and r[2] >= 0.16 and r[3] >= 0.09
    assert reader().center("right") == pytest.approx(RIGHT)


def test_templates_are_saved_with_a_greyed_one():
    names = {p.name for p in ABILITY_DIR.glob("musketeer*.png")}
    assert {"musketeer.png", "musketeer_grey.png"} <= names
    rd = reader()
    assert {t.state for t in rd.templates} >= {"ready", "greyed"}
    for p in ABILITY_DIR.glob("musketeer*.png"):
        assert cv2.imread(str(p), cv2.IMREAD_UNCHANGED).shape[2] == 4         # BGRA: the alpha channel is the mask


@pytest.mark.parametrize("side,center,flip", [("right", RIGHT, False), ("left", LEFT, False), ("left", LEFT, True)])
def test_reader_detects_the_button_on_each_side(side, center, flip):
    frame = paste_button(make_content(), center, flip=flip)
    out = reader().read_raw(frame, elixir=5)
    assert out["visible"] and out["side"] == side and out["affordable"] is True
    assert out["hero"] == "musketeer" and out["cost"] == 3 and out["conf"] > 0.7


def test_reader_rejects_frames_without_the_button():
    rd = reader()
    for seed in range(6):
        out = rd.read_raw(make_content(seed=seed), elixir=5)
        assert out["visible"] is False and out["side"] is None and out["affordable"] is False
    # a purple blob where the badge would be is not a button
    frame = make_content()
    cv2.circle(frame, (int(0.86 * 600), int(0.757 * 1300)), 14, (200, 60, 170), -1)
    assert rd.read_raw(frame, elixir=5)["visible"] is False
    # an empty / tiny content does not crash
    assert rd.read_raw(np.zeros((0, 0, 3), np.uint8))["visible"] is False
    assert rd.read_raw(make_content(60, 130))["visible"] is False


def test_reader_rejects_another_heros_button():
    """Every hero has the same disc and cost badge; only the icon differs (the correlation over the inner disc)."""
    frame = paste_button(make_content(), RIGHT, name="musketeer_grey")
    assert reader().read_raw(frame)["visible"] is True                       # control: the Musketeer's
    H, W = frame.shape[:2]
    cx, cy = int(RIGHT[0] * W), int(RIGHT[1] * H)
    other = frame.copy()
    cv2.circle(other, (cx, cy), 28, (150, 150, 150), -1)                     # wipe the turret icon ...
    cv2.fillPoly(other, [np.array([[cx - 22, cy + 18], [cx - 6, cy - 24], [cx + 4, cy + 18]]),
                         np.array([[cx + 2, cy + 20], [cx + 14, cy - 22], [cx + 24, cy + 10]])], (245, 245, 245))   # ... for a claw
    out = reader().detect(other)
    assert out["visible"] is False and out["icon"] < 0.7 and out["score"] > 0.4


def test_reader_affordable_from_saturation_else_elixir():
    rd = reader()
    ready = rd.read_raw(paste_button(make_content(), RIGHT, name="musketeer"), elixir=0)
    assert ready["visible"] and ready["affordable"] is True                  # coloured: saturation wins over a stale elixir
    grey = rd.read_raw(paste_button(make_content(), RIGHT, name="musketeer_grey"), elixir=9)
    assert grey["visible"] and grey["affordable"] is False                   # greyed: saturation wins over the elixir
    flash = paste_button(make_content(), RIGHT, name="musketeer_flash")
    assert rd.read_raw(flash, elixir=3)["affordable"] is True                # the white flash: the elixir decides
    assert rd.read_raw(flash, elixir=1)["affordable"] is False
    assert rd.read_raw(flash)["affordable"] is False


@pytest.mark.parametrize("w,scale,dx,dy", [(600, 1.0, 0, 0), (600, 1.07, 4, -5), (600, 0.93, -4, 6), (400, 1.0, 0, 0),
                                           (900, 1.0, 3, 3)])
def test_reader_tolerates_scale_offset_and_content_size(w, scale, dx, dy):
    frame = paste_button(make_content(w, int(w * 2.1667)), RIGHT, scale=scale, dx=dx, dy=dy)
    out = reader().read_raw(frame, elixir=5)
    assert out["visible"] and out["side"] == "right"


def test_reader_hysteresis_two_of_the_last_three():
    rd = reader()
    on = paste_button(make_content(), RIGHT)
    off = make_content()
    assert rd.read(on, 5)["visible"] is False          # one look is not enough
    seen = rd.read(on, 5)
    assert seen["visible"] and seen["side"] == "right" and seen["affordable"] is True and seen["conf"] > 0.7
    assert rd.read(off, 5)["visible"] is True          # a one-frame miss is bridged (2 of the last 3)
    gone = rd.read(off, 5)
    assert gone["visible"] is False and gone["side"] is None and gone["affordable"] is False
    assert rd.read(on, 5)["visible"] is False and rd.read(on, 5)["visible"] is True
    rd.reset()
    assert rd.read(on, 5)["visible"] is False          # the history is per match


def test_reader_roi_is_calibratable_and_falls_back_to_the_default():
    moved = [0.70, 0.60, 0.20, 0.12]                    # centre (0.80, 0.66)
    frame = paste_button(make_content(), (0.80, 0.66))
    assert AbilityButtonReader({"hero_ability_right": moved}).read_raw(frame)["visible"] is True
    assert reader().read_raw(frame)["visible"] is False
    old = {k: v for k, v in DEFAULT_ROIS.items() if not k.startswith("hero_ability")}     # a calib.json from before MAC-AI
    assert AbilityButtonReader(old).read_raw(paste_button(make_content(), RIGHT))["visible"] is True


def _recorded(name: str) -> np.ndarray:
    """A recorded live shot (50 % scale) as a content image: the frame's content rect of the mirror calibration at 2x."""
    img = cv2.imread(str(SHOT / name))
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    h, w = img.shape[:2]
    x, y, cw, ch = (int(round(v)) for v in (MIRROR_CONTENT[0] * w, MIRROR_CONTENT[1] * h, MIRROR_CONTENT[2] * w,
                                              MIRROR_CONTENT[3] * h))
    return img[y:y + ch, x:x + cw]


@pytest.mark.skipif(not (SHOT / "d1518_t191.4_a.jpg").exists(), reason="recorded shots (runs/) not available")
def test_reader_finds_the_button_on_the_recorded_shots():
    rd = reader()
    ready = rd.read_raw(_recorded("d1518_t191.4_a.jpg"), elixir=3.0)          # the frame the spec names: coloured
    assert ready["visible"] and ready["side"] == "right" and ready["affordable"] is True
    grey = rd.read_raw(_recorded("d1518_t191.4_b.jpg"), elixir=0.5)           # the same button 0.5 s after the tap's elixir drop
    assert grey["visible"] and grey["side"] == "right" and grey["affordable"] is False
    for name in ("d0224_t23.3_a.jpg", "d0366_t39.7_a.jpg", "d1127_t136.7_a.jpg"):     # no button on screen yet
        assert rd.read_raw(_recorded(name))["visible"] is False


@pytest.mark.skipif(not SHOT.exists(), reason="recorded shots (runs/) not available")
def test_reader_precision_recall_on_labelled_shots():
    """Frames labelled by eye (the 26 with the button: coloured, greyed and the white flash; the other 116 without)."""
    pos = {"d1175_t144.7_a", "d1175_t144.7_b", "d1175_t144.7_c", "d1243_t153.3_a", "d1518_t191.4_a", "d1518_t191.4_b",
           "d1518_t191.4_c", "d1535_t193.9_a", "d1535_t193.9_b", "d1856_t232.0_c", "d1926_t240.7_a", "d1926_t240.7_b",
           "d1926_t240.7_c", "d1999_t249.9_a", "d1999_t249.9_b", "d1999_t249.9_c", "d2013_t251.8_a", "d2013_t251.8_b",
           "d2013_t251.8_c", "d2020_t253.0_a", "d2020_t253.0_b", "d2020_t253.0_c", "d2040_t255.8_a", "d2040_t255.8_b",
           "d2040_t255.8_c", "d2416_t300.6_c"}
    files = sorted(SHOT.glob("*.jpg"))
    assert len(files) >= 140 and pos <= {p.stem for p in files}
    rd = reader()
    tp = fp = fn = 0
    for p in files:
        vis = rd.read_raw(_recorded(p.name), elixir=5.0)["visible"]
        tp += vis and p.stem in pos
        fp += vis and p.stem not in pos
        fn += (not vis) and p.stem in pos
    assert (tp, fp, fn) == (len(pos), 0, 0)


# -- 1b. perception: state.own["hero_ability"] ----------------------------------------------------------------------------
class ButtonSource(FrameSource):
    """20 menu frames, 20 match frames without the button, 15 with the coloured one on the right, 15 without again."""
    fps = 10.0

    def frames(self):
        hand = ["knight", "fireball", "musketeer", "ice-spirit"]
        rng = np.random.default_rng(0)
        i = 0
        for _ in range(20):
            img = np.full((1024, 576, 3), (120, 90, 60), np.uint8)
            img[900:1000, 40:540] = rng.integers(0, 255, (100, 500, 3), np.uint8)
            yield img, i / self.fps, i
            i += 1
        for n_on in (0, 1, 0):
            for _ in range(20 if n_on == 0 else 15):
                frame = arena_frame(synthetic_hud(hand, "hog-rider", 7))
                if n_on:
                    frame = paste_button(frame, RIGHT)
                yield frame, i / self.fps, i
                i += 1


def test_perception_adds_hero_ability_only_while_the_button_is_seen(tmp_path):
    calib = Calibration(source={"type": "synthetic", "frame_size": [576, 1024]}, rois=dict(DEFAULT_ROIS))
    calib.arena_corners_frac = [[0.072, 0.769], [0.922, 0.769], [0.905, 0.079], [0.089, 0.079]]
    cfg = tmp_path / "calib.json"
    calib.save(cfg)
    p = Perception(cfg, ButtonSource(), detector=None, use_ocr=False)
    states = [s for s in p.states() if s.readiness == "match"]
    keyed = [i for i, s in enumerate(states) if "hero_ability" in s.own]
    assert keyed and keyed == list(range(keyed[0], keyed[-1] + 1))             # one contiguous stretch
    assert 0 < keyed[0] and keyed[-1] < len(states) - 1                        # absent before the button and after it
    ab = states[keyed[0]].own["hero_ability"]
    assert ab == {"hero": "musketeer", "visible": True, "side": "right", "affordable": True, "cost": 3,
                  "conf": pytest.approx(ab["conf"])} and ab["conf"] > 0.7
    assert "hero_ability" in states[keyed[0]].to_json()["own"]
    assert all("hero_ability" not in s.own for s in states[:keyed[0]])
    assert p.timing_report()["ability"]["n"] == len(states)


# -- 2. executor ----------------------------------------------------------------------------------------------------------
RECT = [100, 50, 300, 600]
CALIB = {"content_rect_frac": [0.0, 0.1, 1.0, 0.8],
         "arena_corners_frac": [[0.1, 0.75], [0.9, 0.75], [0.88, 0.1], [0.12, 0.1]],
         "rois": {f"hand_{i}": DEFAULT_ROIS[f"hand_{i}"] for i in range(4)}}


class Poster:
    def __init__(self):
        self.calls = []

    def __call__(self, x, y):
        self.calls.append((x, y))


def _live(monkeypatch, poster, calib=CALIB, **kw):
    monkeypatch.setenv(LIVE_ENV, "1")
    monkeypatch.setattr(sys, "platform", "darwin")
    return MirroringExecutor.from_calib(calib, RECT, live=True, poster=poster, sleep=lambda s: None, **kw)


def test_ability_button_point_is_the_roi_centre_on_screen():
    smap = screen_map_from_calib(CALIB, RECT)
    assert smap.ability_rois["right"] == DEFAULT_ROIS["hero_ability_right"]      # missing from calib.json: the defaults
    for side, (fx, fy) in (("right", RIGHT), ("left", LEFT)):
        assert ability_button_point(smap, side) == pytest.approx(list(content_to_screen(fx, fy, smap)))
    # content rect [0, .1, 1, .8] inside the window (100, 50, 300, 600): x = 100 + .9 * 300, y = 50 + (.1 + .776 * .8) * 600
    assert ability_button_point(smap, "right") == pytest.approx([370.0, 482.48])
    custom = {**CALIB, "rois": {**CALIB["rois"], "hero_ability_left": [0.0, 0.5, 0.2, 0.1]}}
    assert ability_button_point(screen_map_from_calib(custom, RECT), "left") == pytest.approx(
        list(content_to_screen(0.1, 0.55, screen_map_from_calib(custom, RECT))))


def test_tap_ability_dry_run_logs_planned_ability_and_never_posts():
    poster = Poster()
    ex = MirroringExecutor.from_calib(CALIB, RECT, poster=poster)
    e = ex.tap_ability("right", 12.5)
    assert e["status"] == "planned_ability" and e["sent"] is False and e["key"] == ABILITY_KEY and e["side"] == "right"
    assert e["clicks"] == [pytest.approx([370.0, 482.48])] and e["slot"] is None and e["t"] == 12.5
    assert poster.calls == [] and ex.log[-1] is e
    d = DryRunExecutor().tap_ability("left", 3.0)
    assert d["status"] == "planned_ability" and d["sent"] is False and d["side"] == "left" and d["key"] == ABILITY_KEY


def test_tap_ability_live_sends_one_click_at_the_button(monkeypatch):
    poster = Poster()
    ex = _live(monkeypatch, poster)
    e = ex.tap_ability("left", 1.0)
    assert e["sent"] is True and e["status"] == "planned_ability"
    assert poster.calls == [pytest.approx((130.0, 482.48))]                       # 100 + 0.1 * 300, 50 + (0.1 + 0.776 * 0.8) * 600


def test_tap_ability_has_the_same_double_live_guard(monkeypatch):
    poster = Poster()
    ex = _live(monkeypatch, poster)
    monkeypatch.delenv(LIVE_ENV)                                    # the env switch removed after construction
    with pytest.raises(PermissionError):
        ex.tap_ability("right", 1.0)
    assert poster.calls == []
    monkeypatch.delenv(LIVE_ENV, raising=False)
    with pytest.raises(PermissionError):                            # and no live executor without it
        MirroringExecutor.from_calib(CALIB, RECT, live=True, poster=poster)
    monkeypatch.setenv(LIVE_ENV, "1")
    monkeypatch.setattr(sys, "platform", "linux")
    with pytest.raises(PermissionError):
        MirroringExecutor.from_calib(CALIB, RECT, live=True, poster=poster)


def test_tap_ability_is_bounds_checked_and_rejects_a_bad_side(monkeypatch):
    poster = Poster()
    ex = _live(monkeypatch, poster)
    ex.bounds = [100, 50, 200, 600]                                 # the button (x = 370) is outside this window
    e = ex.tap_ability("right", 1.0)
    assert e["status"] == "skipped_out_of_window" and e["sent"] is False and poster.calls == []
    assert ex.tap_ability("middle", 1.0)["status"] == "skipped_bad_side"
    assert MirroringExecutor(ex_mod.calibration_from_screen_map(screen_map_from_calib(CALIB, RECT)),
                             [[0, 0]] * 4).tap_ability("right", 1.0)["status"] == "skipped_no_screen_map"


def test_tap_ability_follows_a_moved_window(monkeypatch):
    ex = _live(monkeypatch, Poster())
    ex.update_rect([200, 80, 300, 600])
    assert ex.tap_ability("right", 1.0)["clicks"] == [pytest.approx([470.0, 512.48])]


def test_the_only_place_that_sends_input_is_unchanged():
    src = (ROOT / "clashbot" / "executor.py").read_text()
    assert src.count("CGEventPost") == 1                            # still only quartz_click


# -- 3. the option key, text and candidate --------------------------------------------------------------------------------
KEY = "ability:musketeer|middle|mid"


def test_ability_key_helpers():
    assert is_ability_key(KEY) and not is_ability_key("zap|left|mid") and not is_ability_key(HOLD_KEY) and not is_ability_key(None)
    assert ability_key("left", "bridge") == "ability:musketeer|left|bridge" and ABILITY_KEY == "ability:musketeer"
    assert placement_tile(KEY, PRIOR) is None                        # no pro-median tile: the executor taps a button


def test_option_description_and_validate_accept_the_ability_key():
    want = ("activate Hero Musketeer's Trusty Turret (3 elixir): a turret 3 tiles in front of her, range 4, "
            "pulls win conditions, 10 s")
    assert d1_format.option_description(KEY) == want == option_text(KEY)
    assert d1_format.option_with_note(KEY, ABILITY_NOTE) == f"{want}{d1_format.NOTE_SEP}{ABILITY_NOTE}"
    state, questions = d1_format.joint_request("ctx", ["zap|left|mid", KEY], notes={KEY: ABILITY_NOTE})
    d1_format.validate(state, questions, {d1_format.JOINT_NAME: KEY})
    assert questions[d1_format.JOINT_NAME]["criteria"][KEY].startswith("activate Hero Musketeer's Trusty Turret")
    assert d1_format.option_description("zap|left|mid") == "zap left lane, mid, on our side"     # cards unchanged


def _ability_state(elixir=6.0, tile=(8, 9), side="ally", visible=True, affordable=True, units=None, ability=True):
    st = _state(elixir=elixir, hand=["zap", "archers", "knight", "giant"])
    st["units"] = units if units is not None else [{"unit_class": "musketeer", "card": "musketeer", "side": side,
                                                    "tile": list(tile), "eta_tower": None}]
    if ability:
        st["own"]["hero_ability"] = {"hero": "musketeer", "visible": visible, "side": "right", "affordable": affordable,
                                     "cost": 3, "conf": 0.9}
    return st


def test_hero_ability_move_needs_a_visible_affordable_button_and_her_musketeer():
    m = hero_ability_move(_ability_state(tile=(8, 9)))
    assert isinstance(m, AbilityMove) and m.key == "ability:musketeer|middle|bridge" and m.tile == [8.0, 12.0]
    assert m.note == ABILITY_NOTE == "hero ability: turret in front of the Musketeer"
    assert hero_ability_move(_ability_state(tile=(3, 5))).key == "ability:musketeer|left|mid" and \
        hero_ability_move(_ability_state(tile=(3, 5))).tile == [3.0, 8.0]
    assert hero_ability_move(_ability_state(tile=(3, 1))).key == "ability:musketeer|left|back"       # row 4 is still our back
    assert hero_ability_move(_ability_state(tile=(15, 29))).tile == [15.0, 31.0]      # row + 3 clamped to the arena
    assert hero_ability_move(_ability_state(tile=(15, 29))).key == "ability:musketeer|right|enemy_half"
    assert hero_ability_move(_ability_state(visible=False)) is None
    assert hero_ability_move(_ability_state(affordable=False)) is None
    assert hero_ability_move(_ability_state(elixir=2.0)) is None                      # the (settled) elixir does not cover 3
    assert hero_ability_move(_ability_state(elixir=None)) is not None                 # unread elixir: the button's own word
    assert hero_ability_move(_ability_state(side="enemy")) is None                    # an enemy musketeer is not hers
    assert hero_ability_move(_ability_state(units=[])) is None                        # none on the field
    assert hero_ability_move(_ability_state(ability=False)) is None                   # no button seen (every recorded state)


def _generator():
    hits = [("knight|left|mid", 0.03), ("zap|left|mid", 0.02), ("archers|left|back", 0.01)]
    return CandidateGenerator(FakeIndex(hits), FakeKG({}), CARDS)


def test_generate_appends_ability_candidates_with_source_and_note():
    st = _ability_state()
    base = _generator().generate(st, 10.0, "ctx", n=3)
    assert not any(is_ability_key(c.key) for c in base)
    move = AbilityMove(KEY, [8.0, 12.0])
    out = _generator().generate(st, 10.0, "ctx", n=3, ability_moves=[move])
    assert [c.key for c in out[:-1]] == [c.key for c in base] and out[-1].key == KEY
    c = out[-1]
    assert c.sources == {"ability": 1.0} and c.note == "hero ability: turret in front of the Musketeer" and c.tile == [8.0, 12.0]
    assert top_source(c.sources) == "ability" and top_source({"tac": 0.9}) == "tac"
    assert _generator().generate(st, 10.0, "ctx", n=3, ability_moves=[]) == base
    dup = _generator().generate(st, 10.0, "ctx", n=3, ability_moves=[move, move])
    assert [c.key for c in dup].count(KEY) == 1


# -- 3b. context text -------------------------------------------------------------------------------------------------------
def _clean(hero=None):
    st = _state(elixir=5.0)
    st["own"].update(deck=["zap"], next_card="giant", hero_cards=[], deck_status="valid")
    st["own"]["hand"] = ["zap", "archers", "knight", "giant"]
    st["clock"] = "1:20"
    st["opponent"].update(elixir_estimate=4.0, deck_known=[], recent_plays=[])
    if hero is not None:
        st["own"]["hero_ability"] = hero
    return st


def test_context_has_the_hero_token_only_when_the_button_is_visible():
    plain = render_context(_clean())
    assert "hero:" not in plain
    ready = render_context(_clean({"hero": "musketeer", "visible": True, "side": "right", "affordable": True, "cost": 3,
                                   "conf": 0.9}))
    assert "hand: zap, archers, knight, giant (next giant); hero: musketeer turret ready (3). deck:" in ready
    assert ready.replace("; hero: musketeer turret ready (3)", "") == plain
    waiting = render_context(_clean({"visible": True, "affordable": False, "cost": 3}))
    assert "hero: musketeer turret not ready (3)" in waiting
    assert "hero:" not in render_context(_clean({"visible": False, "affordable": True}))


# -- 4. live state: pass-through and the elixir accounting ------------------------------------------------------------------
ABILITY_OWN = {"elixir": 6.0, "hand": list(FULL_HAND), "hand_conf": [0.9] * 4, "next_card": "wall-breakers",
               "towers": {"king": 4000, "left": 3000, "right": 2500},
               "hero_ability": {"hero": "musketeer", "visible": True, "side": "right", "affordable": True, "cost": 3,
                                "conf": 0.88}}


def test_builder_passes_hero_ability_into_the_clean_state_only_when_visible():
    from tests.test_live_state import builder
    b = builder(hand_tracker=True)
    state, ctx = b.build(make_gs(100.0, own=dict(ABILITY_OWN)))
    assert state["own"]["hero_ability"] == ABILITY_OWN["hero_ability"] and "hero: musketeer turret ready (3)" in ctx
    hidden = dict(ABILITY_OWN, hero_ability={**ABILITY_OWN["hero_ability"], "visible": False})
    state, ctx = builder(hand_tracker=True).build(make_gs(100.0, own=hidden))
    assert "hero_ability" not in state["own"] and "hero:" not in ctx
    state, _ = builder(hand_tracker=True).build(make_gs(100.0))
    assert "hero_ability" not in state["own"]                                     # recorded states: unchanged


def test_ability_tap_enters_the_elixir_accounting_without_a_hand_slot():
    b = acct()                                                                    # 6 elixir at t=100, full hand
    hand_before = list(b.tracker.slots)
    b.note_own_play(ABILITY_KEY, None, 100.0, cost=ABILITY_COST)
    assert b.tracker.slots == hand_before and b.tracker.log == [] and b.tracker.pending == {}   # not a card play
    assert not [e for e in b._events if e["player"] == "own"]                       # and no play event
    assert build_at(b, 100.3, 6.0)[0] == pytest.approx(3.0 + 0.3 / 2.8)           # the stale 6 is replaced by 6 - 3 + regen
    assert build_at(b, 100.6, 3.0)[0] == 3.0 and b.confirmed_taps == 1            # the reading dropped by the cost: confirmed
    assert b.failed_taps == 0


def test_ability_tap_that_never_lands_is_counted_and_leaves_the_hand_alone():
    b = acct()
    b.note_own_play(ABILITY_KEY, None, 100.0, cost=3)
    assert build_at(b, 100.3, 6.0)[0] < 3.2
    elixir, hand = build_at(b, 101.3, 6.0)                                         # 1.3 s on and the bar never dropped
    assert elixir == 6.0 and b.failed_taps == 1 and hand == list(FULL_HAND) and b._cooldown == {}
    b2 = acct()
    b2.note_own_play(ABILITY_KEY, None, 100.0)                                     # cost unknown to the cards table: no accounting
    assert build_at(b2, 100.3, 6.0)[0] == 6.0


def test_ability_tap_cap_stacks_with_a_card_tap():
    b = acct()
    b.note_own_play(ABILITY_KEY, None, 100.0, cost=3)
    assert build_at(b, 100.1, 6.0)[0] == pytest.approx(3.0 + 0.1 / 2.8)
    b.note_own_play("bomb-tower", 3, 100.1)                                        # cost 4 from the corrected ~3.04
    assert build_at(b, 100.3, 6.0)[0] < 0.2


# -- 5. the live loop -------------------------------------------------------------------------------------------------------
class AbilityExecutor:
    """Records card taps (`execute`) and ability taps; `sent` as a live executor would report it."""

    def __init__(self, sent=True, with_ability=True):
        self.calls, self.taps, self.sent = [], [], sent
        if not with_ability:
            self.tap_ability = None

    def execute(self, key, tile, t, hand=None):
        self.calls.append((key, tile, t, hand))
        return {"t": t, "key": key, "tile": tile, "status": "planned", "slot": 0, "clicks": [], "sent": self.sent}

    def tap_ability(self, side, t):
        self.taps.append((side, t))
        return {"t": t, "key": ABILITY_KEY, "tile": None, "side": side, "status": "planned_ability", "slot": None,
                "clicks": [[370.0, 482.0]], "sent": self.sent}


class RecordingBuilder(FakeBuilder):
    def __init__(self, state=None, warmup=0):
        super().__init__(state, warmup)
        self.plays = []

    def note_own_play(self, card, slot, t, tile=None, cost=None):
        self.plays.append((card, slot, t, tile, cost))


class AbilityGenerator:
    """Card candidates, plus one ability candidate per move the loop passes (as the real generator does)."""

    def __init__(self, keys=("zap|left|mid", "giant|left|back")):
        self.keys, self.calls = list(keys), []

    def generate(self, state, t, ctx, n=4, **filters):
        self.calls.append(dict(filters))
        out = [Candidate(k, 1.0 - 0.1 * i, {"knn": 1.0}) for i, k in enumerate(self.keys)]
        for m in filters.get("ability_moves") or []:
            out.append(Candidate(m.key, m.score, {"ability": 1.0}, list(m.tile), m.note))
        return out


class PickKey:
    """A key ranker that picks the option whose key starts with `prefix` (else the last one)."""
    name = "pick"

    def __init__(self, prefix="ability:"):
        self.prefix, self.calls = prefix, []

    def choose_keys(self, context, keys, plan=None, notes=None):
        self.calls.append((list(keys), dict(notes or {})))
        i = next((j for j, k in enumerate(keys) if k.startswith(self.prefix)), len(keys) - 1)
        return RankResult(index=i, probs=[1.0 if j == i else 0.0 for j in range(len(keys))], model="pick")


def _ability_loop(state=None, executor=None, generator=None, ranker=None, builder=None, **kw):
    state = state or _ability_state()
    return _loop(builder=builder or RecordingBuilder(state), generator=generator or AbilityGenerator(),
                 ranker=ranker or PickKey(), executor=executor or AbilityExecutor(), **kw)


def test_live_loop_offers_the_ability_only_when_the_button_is_usable():
    for state in (_ability_state(visible=False), _ability_state(affordable=False), _ability_state(units=[]),
                  _ability_state(ability=False), _ability_state(elixir=2.5)):
        gen = AbilityGenerator()
        d = _ability_loop(state, generator=gen, ranker=FakeKeyRanker()).step("f", 1.0, 0, 1.0)
        assert gen.calls == [{}] or not gen.calls, state["own"]
        assert d is None or not any(is_ability_key(c["key"]) for c in d.candidates)
    gen = AbilityGenerator()
    lp = _ability_loop(generator=gen, ranker=FakeKeyRanker())
    lp.step("f", 1.0, 0, 1.0)
    assert [m.key for m in gen.calls[0]["ability_moves"]] == ["ability:musketeer|middle|bridge"]


def test_live_loop_routes_an_ability_key_to_tap_ability():
    ex, bld, rk = AbilityExecutor(), RecordingBuilder(_ability_state()), PickKey()
    lp = _ability_loop(executor=ex, builder=bld, ranker=rk)
    d = lp.step("f", 1.0, 0, 1.0)
    assert d.key == "ability:musketeer|middle|bridge" and ex.calls == [] and ex.taps == [("right", 1.0)]
    assert d.tile_source == "ability" and d.tile == [8.0, 12.0] and d.executor["status"] == "planned_ability"
    assert d.mode == "trigger" and d.candidates[-1]["sources"] == {"ability": 1.0}
    assert d.candidates[-1]["note"] == "hero ability: turret in front of the Musketeer"
    assert rk.calls[0][0] == ["zap|left|mid", "giant|left|back", "ability:musketeer|middle|bridge"]
    assert rk.calls[0][1]["ability:musketeer|middle|bridge"] == "hero ability: turret in front of the Musketeer"


def test_live_loop_ability_skips_the_hand_check_and_accounts_for_3_elixir():
    bld = RecordingBuilder(_ability_state())
    lp = _ability_loop(builder=bld)
    assert lp.require_hand
    d = lp.step("f", 1.0, 0, 1.0)
    assert d.key.startswith("ability:") and d.executor["status"] == "planned_ability"
    assert bld.plays == [("ability:musketeer", None, 1.0, None, 3)]                # the 3 elixir enter C2's accounting
    # a hand that holds none of the card options: the ability key alone is not enough to act (see below), but it never
    # needs a card in the hand
    st = _ability_state()
    st["own"]["hand"] = ["zap", None, None, None]
    d = _ability_loop(st, builder=RecordingBuilder(st)).step("f", 1.0, 0, 1.0)
    assert d.key.startswith("ability:")


def test_live_loop_does_not_account_a_tap_that_was_not_sent():
    bld = RecordingBuilder(_ability_state())
    d = _ability_loop(builder=bld, executor=AbilityExecutor(sent=False)).step("f", 1.0, 0, 1.0)
    assert d.key.startswith("ability:") and bld.plays == []                       # dry run: nothing was tapped


def test_live_loop_never_plays_the_ability_as_the_only_option():
    ex = AbilityExecutor()
    gen = AbilityGenerator(keys=("fireball|left|mid",))                           # not in the hand: nothing else to play
    d = _ability_loop(executor=ex, generator=gen).step("f", 1.0, 0, 1.0)
    assert d.key == HOLD_KEY and d.executor["status"] == "hold_hand_unknown" and ex.taps == [] and ex.calls == []


def test_live_loop_ability_retry_guard_and_a_card_choice_is_unchanged():
    ex = AbilityExecutor()
    lp = _ability_loop(executor=ex, ranker=PickKey(prefix="zap"))
    d = lp.step("f", 1.0, 0, 1.0)
    assert d.key == "zap|left|mid" and ex.taps == [] and ex.calls[0][0] == "zap|left|mid"      # the model chose a card
    lp2 = _ability_loop(executor=ex)
    assert lp2.step("f", 1.0, 0, 1.0).key.startswith("ability:")
    gen = lp2.generator
    lp2.step("f", 5.0, 1, 1.0)
    assert "ability_moves" not in gen.calls[-1]                                   # 4 s after a tap: not offered again
    lp2.step("f", 12.0, 2, 1.0)
    assert "ability_moves" in gen.calls[-1]


def test_live_loop_with_an_executor_without_tap_ability_logs_it_instead_of_crashing():
    d = _ability_loop(executor=AbilityExecutor(with_ability=False)).step("f", 1.0, 0, 1.0)
    assert d.key.startswith("ability:") and d.executor["status"] == "no_ability_tap" and d.executor["sent"] is False


def test_live_loop_with_the_real_generator_and_dry_run_executor():
    """End to end on the real candidate generator (fake index / KG) and the DryRunExecutor: the ranker may pick it."""
    ex = DryRunExecutor()
    lp = _loop(builder=RecordingBuilder(_ability_state()), generator=_generator(), ranker=PickKey(), executor=ex)
    d = lp.step("f", 1.0, 0, 1.0)
    assert d.key == "ability:musketeer|middle|bridge" and ex.log[-1]["status"] == "planned_ability"
    assert d.candidates[-1]["key"] == d.key and d.candidates[-1]["sources"] == {"ability": 1.0}
    from clashbot.live import decision_source
    assert decision_source(d) == "ability"
    assert lp.summary()["decisions_by_source"] == {"ability": 1}
    assert lp.summary()["decisions_by_status"] == {"planned_ability": 1}
    assert lp.summary()["hero_ability"] == {"button_frames": 1, "offered": 1, "taps": 1}


def test_live_loop_without_the_button_is_what_it_was():
    """No hero_ability in the state: the generator is called exactly as before and the card path is untouched."""
    ex = AbilityExecutor()
    gen = AbilityGenerator()
    lp = _loop(builder=RecordingBuilder(_state()), generator=gen, ranker=FakeKeyRanker(), executor=ex)
    d = lp.step("f", 1.0, 0, 1.0)
    assert gen.calls == [{}] and d.key == "giant|left|back" and d.tile_source == "prior" and ex.taps == []
    assert d.tile == placement_tile("giant|left|back", PRIOR)
    assert "hero_ability" not in lp.summary()                                      # no button seen: no new summary key


# -- part 2 (MAC-AK K4b): the engine proposes the ability, the policy lists it, the loop may offer it alone ------------------
from clashbot.policy import ActDecision  # noqa: E402
from clashbot.tactics import TacticalMove, TacticalReport  # noqa: E402

ENGINE_NOTE = "turret pulls giant right, saves ~1,100 tower HP, 3 elixir"


def _ability_move(key="ability:musketeer|middle|bridge", tile=(8.0, 12.0), reason=ENGINE_NOTE):
    return TacticalMove(ABILITY_KEY, key, list(tile), "ability", reason, 0.8, None, stop="turret", adequate=True,
                        prevented=1110.0, tile_source="ability", push_lane="right")


class AbilityTactics:
    """An engine stand-in whose report proposes the ability (or not) and records the states it was asked about."""

    roles = {}

    def __init__(self, propose=True):
        self.propose, self.states = propose, []

    def analyse(self, state, t=None):
        self.states.append(state)
        moves = [_ability_move()] if self.propose else []
        return TacticalReport([], moves, moves[0] if moves else None, None, ["tactics: best ability:musketeer @[8,12] x"],
                              abilities=moves)


class VerdictPolicy:
    """Hands out one decision; candidates pass the filter unchanged. Records the states it decides on."""

    def __init__(self, verdict):
        self.verdict, self.states = verdict, []

    def decide(self, state, t, context_text=None):
        self.states.append(state)
        return self.verdict

    def filter_candidates(self, candidates, decision, state):
        return candidates


ALLOW_ABILITY = ActDecision(True, "defend", "middle", "defend giant right: ability:musketeer (turret)",
                            allowed_cards=frozenset({ABILITY_KEY}))
NOT_IN_HAND = ("fireball|left|mid", "valkyrie|left|mid")           # card candidates the hand cannot play


def _part2_loop(propose=True, verdict=ALLOW_ABILITY, keys=NOT_IN_HAND, executor=None, ranker=None, state=None):
    ex = executor or AbilityExecutor()
    tac = AbilityTactics(propose)
    pol = VerdictPolicy(verdict)
    lp = _loop(builder=RecordingBuilder(state or _ability_state()), generator=AbilityGenerator(keys=keys),
               ranker=ranker or PickKey(), executor=ex, policy=pol, tactics=tac)
    return lp, ex, tac, pol


def test_part2_an_engine_proposed_ability_the_policy_allowed_is_offered_even_when_no_card_survives():
    lp, ex, tac, pol = _part2_loop()
    d = lp.step("f", 1.0, 0, 1.0)
    assert d.key == "ability:musketeer|middle|bridge" and ex.taps == [("right", 1.0)] and ex.calls == []
    assert d.executor["status"] == "planned_ability" and d.tile_source == "ability" and d.mode == "defend"
    cand = d.candidates[-1]
    assert cand["key"] == d.key and cand["sources"] == {"ability": 1.0}
    assert cand["note"] == f"hero ability: {ENGINE_NOTE}" and cand["score"] == 0.8     # the engine's reason and score
    assert lp.summary()["hero_ability"] == {"button_frames": 1, "offered": 1, "taps": 1}
    assert lp.summary()["decisions_by_source"] == {"ability": 1}


def test_part2_the_part_one_rule_stays_unless_the_engine_proposed_and_the_policy_allowed_it():
    # the engine proposes it but the policy does not list it, and no card can be played: held, as in part 1
    other = ActDecision(True, "defend", "middle", "x", allowed_cards=frozenset({"fireball"}))
    for lp, ex, *_ in (_part2_loop(verdict=other), _part2_loop(propose=False),
                       _part2_loop(propose=False, verdict=ALLOW_ABILITY)):
        d = lp.step("f", 1.0, 0, 1.0)
        assert d.key == HOLD_KEY and d.executor["status"] == "hold_hand_unknown" and ex.taps == []
    # allowed without allowed_cards (an old-style decision): not the only option either
    plain = ActDecision(True, "defend", "middle", "x")
    lp, ex, *_ = _part2_loop(verdict=plain)
    assert lp.step("f", 1.0, 0, 1.0).key == HOLD_KEY and ex.taps == []


def test_part2_with_card_options_the_ability_is_one_more_and_the_model_decides():
    ex = AbilityExecutor()
    lp, ex, *_ = _part2_loop(keys=("zap|left|mid", "giant|left|back"), executor=ex, ranker=PickKey(prefix="zap"))
    d = lp.step("f", 1.0, 0, 1.0)
    assert d.key == "zap|left|mid" and ex.taps == [] and [c["key"] for c in d.candidates][-1].startswith("ability:")
    assert d.candidates[-1]["note"] == f"hero ability: {ENGINE_NOTE}"                 # the model reads the simulation's reason
    lp, ex, *_ = _part2_loop(keys=("zap|left|mid", "giant|left|back"))
    assert lp.step("f", 1.0, 0, 1.0).key.startswith("ability:")


def test_part2_the_ability_falls_back_to_the_plain_option_without_a_proposal():
    lp, ex, *_ = _part2_loop(propose=False, keys=("zap|left|mid", "giant|left|back"))
    d = lp.step("f", 1.0, 0, 1.0)
    assert d.key.startswith("ability:") and d.candidates[-1]["note"] == ABILITY_NOTE       # part 1's candidate


def test_part2_after_a_tap_neither_the_engine_nor_the_policy_is_offered_the_button_for_ten_seconds():
    lp, ex, tac, pol = _part2_loop(keys=("zap|left|mid", "giant|left|back"))
    lp.step("f", 1.0, 0, 1.0)
    assert ex.taps == [("right", 1.0)]
    lp.step("f", 4.0, 1, 1.0)
    assert pol.states[1]["own"]["hero_ability"]["affordable"] is False and tac.states[1]["own"]["hero_ability"]["affordable"] is False
    assert pol.states[0]["own"]["hero_ability"]["affordable"] is True               # (the first look was before the tap)
    d = lp.step("f", 12.0, 2, 1.0)
    assert pol.states[2]["own"]["hero_ability"]["affordable"] is True and d.key.startswith("ability:")

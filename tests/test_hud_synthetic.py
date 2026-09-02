"""HUD readers on a synthetic HUD composited from the Phase 1 card art: the
readers must identify the cards placed in the slots and the elixir from a
drawn bar. This validates the logic, not the real-frame ROIs (those are
measured in the video evaluation)."""
from pathlib import Path

import cv2
import numpy as np
import pytest

from cr_perception.hud import DEFAULT_ROIS, CardMatcher, ElixirBarReader, HandReader, DigitTemplates, DigitReader, crop_roi

KB = Path(__file__).resolve().parents[1] / "knowledge_base"
IMAGES = KB / "cards" / "images"


def synthetic_hud(cards: list[str], next_card: str, elixir: int, W=576, H=1024) -> np.ndarray:
    img = np.full((H, W, 3), (40, 60, 40), np.uint8)
    for i, slug in enumerate(cards):
        x, y, w, h = DEFAULT_ROIS[f"hand_{i}"]
        art = cv2.imread(str(IMAGES / f"{slug}.png"))
        x0, y0 = int(x * W), int(y * H)
        x1, y1 = int((x + w) * W), int((y + h) * H)
        img[y0:y1, x0:x1] = cv2.resize(art, (x1 - x0, y1 - y0))
    x, y, w, h = DEFAULT_ROIS["next_card"]
    art = cv2.imread(str(IMAGES / f"{next_card}.png"))
    x0, y0, x1, y1 = int(x * W), int(y * H), int((x + w) * W), int((y + h) * H)
    img[y0:y1, x0:x1] = cv2.resize(art, (x1 - x0, y1 - y0))
    # elixir bar: dark track with a purple fill of elixir/10
    x, y, w, h = DEFAULT_ROIS["elixir_bar"]
    x0, y0, x1, y1 = int(x * W), int(y * H), int((x + w) * W), int((y + h) * H)
    img[y0:y1, x0:x1] = (30, 20, 40)
    fill = x0 + int((x1 - x0) * elixir / 10)
    img[y0:y1, x0:fill] = (220, 60, 200)  # BGR purple/magenta
    # elixir numeral
    x, y, w, h = DEFAULT_ROIS["elixir_num"]
    x0, y0 = int(x * W), int(y * H)
    cv2.putText(img, str(elixir), (x0 + 4, y0 + int(h * H) - 6), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 3)
    return img


@pytest.mark.parametrize("cards,next_card,elixir", [
    (["knight", "fireball", "musketeer", "ice-spirit"], "hog-rider", 7),
    (["golem", "baby-dragon", "lightning", "tornado"], "night-witch", 3),
    (["x-bow", "tesla", "archers", "the-log"], "skeletons", 10),
])
def test_hand_and_elixir_readers(cards, next_card, elixir):
    img = synthetic_hud(cards, next_card, elixir)
    # mild lighting change + downscale to mimic a video frame
    img = cv2.convertScaleAbs(img, alpha=0.9, beta=8)
    img = cv2.resize(img, None, fx=0.75, fy=0.75)
    m = CardMatcher(IMAGES)
    hr = HandReader(m, DEFAULT_ROIS)
    out = hr.read(img)
    assert out["hand"] == cards, (out["hand"], out["candidates"])
    assert out["next_card"] == next_card
    assert min(out["hand_conf"]) > 0.45
    e, c = ElixirBarReader(DEFAULT_ROIS["elixir_bar"]).read(img)
    assert e == elixir and c > 0.5
    s, dc = DigitReader(DEFAULT_ROIS["elixir_num"], DigitTemplates()).read_string(img)
    assert s == str(elixir), (s, dc)


def test_blank_slot_is_none_not_a_guess():
    img = synthetic_hud(["knight", "fireball", "musketeer", "ice-spirit"], "hog-rider", 5)
    x, y, w, h = DEFAULT_ROIS["hand_1"]
    H, W = img.shape[:2]
    img[int(y * H):int((y + h) * H), int(x * W):int((x + w) * W)] = (70, 70, 70)  # greyed-out slot
    out = HandReader(CardMatcher(IMAGES), DEFAULT_ROIS).read(img)
    assert out["hand"][1] is None and out["hand_conf"][1] < 0.45
    assert out["hand"][0] == "knight"


def test_card_matcher_separates_all_phase1_cards():
    """Every card's own art must be its nearest neighbour (self-retrieval),
    after a resize/lighting perturbation. Reports the confusable pairs."""
    m = CardMatcher(IMAGES)
    wrong = []
    for p in sorted(IMAGES.glob("*.png")):
        art = cv2.imread(str(p))
        art = cv2.convertScaleAbs(cv2.resize(art, (60, 72)), alpha=1.1, beta=-10)
        best = m.match(art, top=2)
        if best[0][0] != p.stem:
            wrong.append((p.stem, best[0][0]))
    assert not wrong, wrong

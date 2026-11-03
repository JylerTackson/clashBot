"""Draw what the executor WOULD tap on a captured mirroring frame. Never sends input.

  python tools/executor_overlay.py --calib calib.json --image frame.png [--rect l,t,w,h]
         [--keys k1,k2,...] [--hand c1,c2,c3,c4] [--prior dataset/v1/placement_prior.json] [--out overlay.png]

`--rect` is the captured window in global screen points (default 0,0,image_w,image_h, i.e. points == pixels);
on a Retina capture pass the real rect and the drawing maps back to image pixels. The executor is always
built with live=False. Prints one JSON entry per key: key, tile, status, clicks_screen, clicks_image.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from clashbot.actions import load_placement_prior, placement_tile  # noqa: E402
from clashbot.executor import (GRID_COLS, GRID_ROWS, MirroringExecutor, screen_to_image,  # noqa: E402
                               tile_to_screen)

DEFAULT_HAND = "knight,archers,fireball,hog-rider"
LANE_COL = {"left": 3, "middle": 9, "right": 14}
ZONE_ROW = {"back": 2, "mid": 8, "bridge": 14, "enemy_half": 20}
COLOURS = [(0, 0, 255), (0, 160, 0), (255, 0, 0), (0, 140, 255), (200, 0, 200), (200, 200, 0)]  # BGR


def default_keys(hand: list[str]) -> list[str]:
    spots = ["left|bridge", "right|mid", "middle|back", "right|enemy_half"]
    return [f"{card}|{spot}" for card, spot in zip(hand, spots)]


def fallback_tile(key: str) -> list[float] | None:
    """Lane/zone centre for keys the placement prior does not know."""
    parts = key.split("|")
    if len(parts) != 3 or parts[1] not in LANE_COL or parts[2] not in ZONE_ROW:
        return None
    return [float(LANE_COL[parts[1]]), float(ZONE_ROW[parts[2]])]


def key_tile(key: str, prior: dict | None) -> list[float] | None:
    tile = placement_tile(key, prior) if prior else None
    return tile if tile is not None else fallback_tile(key)


def _px(pt: tuple[float, float]) -> tuple[int, int]:
    return int(round(pt[0])), int(round(pt[1]))


def main(argv: list[str] | None = None) -> list[dict]:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--rect", help="window rect in screen points: left,top,width,height (default 0,0,image_w,image_h)")
    ap.add_argument("--keys", help="comma-separated action keys card|lane|zone")
    ap.add_argument("--hand", default=DEFAULT_HAND, help="four card names, slot 0..3")
    ap.add_argument("--prior", default="dataset/v1/placement_prior.json")
    ap.add_argument("--out")
    a = ap.parse_args(argv)

    img = cv2.imread(a.image)
    if img is None:
        sys.exit(f"cannot read {a.image}")
    fh, fw = img.shape[:2]
    rect = [float(v) for v in a.rect.split(",")] if a.rect else [0.0, 0.0, float(fw), float(fh)]
    hand = [c.strip() for c in a.hand.split(",")]
    keys = [k.strip() for k in a.keys.split(",")] if a.keys else default_keys(hand)
    prior = load_placement_prior(a.prior) if Path(a.prior).exists() else None

    ex = MirroringExecutor.from_calib(a.calib, rect, live=False)   # never live
    smap = ex.screen_map

    def to_img(pt) -> tuple[float, float]:
        return screen_to_image(pt[0], pt[1], smap, fw, fh)

    # executor-homography grid: 19 x 33 lines of tile -> screen -> image
    for c in range(GRID_COLS + 1):
        p0, p1 = (to_img(tile_to_screen([c, r], ex.H)) for r in (0, GRID_ROWS))
        cv2.line(img, _px(p0), _px(p1), (255, 255, 0), 1)
    for r in range(GRID_ROWS + 1):
        p0, p1 = (to_img(tile_to_screen([c, r], ex.H)) for c in (0, GRID_COLS))
        cv2.line(img, _px(p0), _px(p1), (255, 255, 0), 1)

    # the four card slots
    for i, pt in enumerate(ex.card_slot_pixels):
        p = _px(to_img(pt))
        cv2.circle(img, p, 9, (255, 255, 255), 2)
        cv2.putText(img, str(i), (p[0] - 4, p[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    out = []
    for i, key in enumerate(keys):
        tile = key_tile(key, prior)
        plan = ex.plan_clicks(key, tile, hand)
        clicks_img = [list(to_img(p)) for p in plan["clicks"]]
        if len(clicks_img) == 2:
            colour = COLOURS[i % len(COLOURS)]
            slot_p, tgt_p = _px(clicks_img[0]), _px(clicks_img[1])
            cv2.circle(img, slot_p, 12, colour, -1)
            cv2.line(img, slot_p, tgt_p, colour, 2)
            cv2.drawMarker(img, tgt_p, colour, cv2.MARKER_CROSS, 22, 3)
            cv2.putText(img, key, (tgt_p[0] + 8, tgt_p[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)
        out.append({"key": key, "tile": tile, "status": plan["status"],
                    "clicks_screen": plan["clicks"], "clicks_image": clicks_img})

    out_path = a.out or str(Path(a.image).with_suffix("")) + ".executor_overlay.png"
    cv2.imwrite(out_path, img)
    print(json.dumps(out, indent=1))
    print(f"wrote {out_path}", file=sys.stderr)
    return out


if __name__ == "__main__":
    main()

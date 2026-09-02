"""One-time calibration from a clean in-match frame -> calib.json.

  python3 tools/calibrate.py --video data/videos/<id>/<id>.mp4 --time 90 --out calib.json
  python3 tools/calibrate.py --image frame.png --corners "x,y x,y x,y x,y" --out calib.json
  python3 tools/calibrate.py ... --detector katacr    # auto arena from tower detections

Steps:
  1. content rect: strip black bars (fractions of the raw frame);
  2. arena corners (bl, br, tr, tl of the 18x32 grid, content pixels):
       a. --corners given manually, or
       b. --detector: detect the six towers, fit a homography to their known
          grid positions (least squares), derive the corners, or
       c. fallback: default portrait-layout fractions (flagged low-confidence);
  3. homography content-pixel -> tile via cv2 (perspective, not linear);
  4. verification image with the grid drawn back over the frame;
  5. HUD ROIs stored as fractions (defaults; edit calib.json or use --roi).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cr_perception import geometry as g  # noqa: E402
from cr_perception.config import Calibration  # noqa: E402
from cr_perception.hud import DEFAULT_ROIS  # noqa: E402
from cr_perception.overlay import draw_grid, draw_rois  # noqa: E402
from cr_perception.screen import assess, detect_content_rect  # noqa: E402

# Grid positions of the tower bases (bottom-centre of the sprite), our
# convention (row 0 = own bottom). Values derived from KataCR's generator
# config (towers_bottom_center_grid_position, top-down rows) via row' = 31 - y.
TOWER_TILES = {
    "own_king": (9.0, 0.5), "own_left": (3.5, 4.3), "own_right": (14.5, 4.3),
    "enemy_king": (9.0, 26.3), "enemy_left": (3.5, 23.3), "enemy_right": (14.5, 23.3),
}
# Fallback arena corners as fractions of the content rect (portrait layout),
# tuned to the BuildABot 720x1280 constants (TILE_INIT_X 52, TILE_INIT_Y 296,
# 18x34 wide, 32x27.6 tall) -> bl (0.072, 0.769), br (0.922, 0.769), top at
# 0.769 - 32*27.6/1280 = 0.079. Treated as a linear guess; --detector refines.
FALLBACK_CORNERS = [[0.072, 0.769], [0.922, 0.769], [0.905, 0.079], [0.089, 0.079]]


def load_frame(args) -> np.ndarray:
    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            sys.exit(f"cannot read {args.image}")
        return img
    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_MSEC, args.time * 1000)
    ok, img = cap.read()
    if not ok:
        sys.exit("cannot read frame")
    return img


def towers_from_detector(content: np.ndarray, det, arena_guess) -> dict[str, tuple[float, float]]:
    """Return tower name -> bottom-centre pixel, using detector classes
    king-tower / queen-tower (any tower-troop variant) split by position."""
    dets = det.detect(content, arena_guess)
    kings = [d for d in dets if d.raw_cls == "king-tower"]
    queens = [d for d in dets if d.raw_cls in ("queen-tower", "cannoneer-tower", "dagger-duchess-tower", "royal-chef-tower")]
    out = {}
    H = content.shape[0]
    for d in kings:
        bx, by = g.bbox_bottom_centre(d.bbox)
        name = "own_king" if by > H / 2 else "enemy_king"
        if name not in out or d.conf > out[name][2]:
            out[name] = (bx, by, d.conf)
    for d in queens:
        bx, by = g.bbox_bottom_centre(d.bbox)
        side = "own" if by > H / 2 else "enemy"
        lane = "left" if bx < content.shape[1] / 2 else "right"
        name = f"{side}_{lane}"
        if name not in out or d.conf > out[name][2]:
            out[name] = (bx, by, d.conf)
    return {k: (v[0], v[1]) for k, v in out.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video")
    ap.add_argument("--image")
    ap.add_argument("--time", type=float, default=60.0)
    ap.add_argument("--corners", help="'x,y x,y x,y x,y' bl br tr tl in CONTENT pixels")
    ap.add_argument("--detector", choices=["katacr", "buildabot"], help="auto arena from towers")
    ap.add_argument("--katacr-root", default="/home/user/wty-yy/katacr")
    ap.add_argument("--katacr-weights", nargs="*", default=[])
    ap.add_argument("--buildabot-root", default="/home/user/pbatch/clashroyalebuildabot")
    ap.add_argument("--out", default="calib.json")
    ap.add_argument("--verify", default=None, help="verification image path (default <out>.verify.png)")
    a = ap.parse_args()

    frame = load_frame(a)
    fh, fw = frame.shape[:2]
    rect = detect_content_rect(frame)
    content = rect.crop(frame)
    ch, cw = content.shape[:2]
    ready = assess(frame, rect)
    print(f"frame {fw}x{fh}, content {rect.to_json()} (aspect {rect.aspect:.3f}), readiness={ready.state} {ready.scores}")
    if ready.state != "match":
        print("WARNING: this frame does not look like an in-match frame; pick another --time")

    calib = Calibration(source={"type": "video" if a.video else "image", "path": a.video or a.image,
                                "frame_size": [fw, fh], "time": a.time},
                        content_rect_frac=[rect.x / fw, rect.y / fh, rect.w / fw, rect.h / fh],
                        rois=dict(DEFAULT_ROIS))
    corners_px = None
    method = "fallback"
    if a.corners:
        corners_px = [tuple(float(v) for v in p.split(",")) for p in a.corners.split()]
        method = "manual"
    elif a.detector:
        if a.detector == "katacr":
            from cr_perception.detect import KataCRDetector
            det = KataCRDetector(a.katacr_weights, a.katacr_root, conf=0.3)
        else:
            from cr_perception.detect import BuildABotDetector
            det = BuildABotDetector(a.buildabot_root, conf=0.3)
        guess = calib.arena_crop(cw, ch)
        towers = towers_from_detector(content, det, guess)
        print("towers found:", {k: (round(x), round(y)) for k, (x, y) in towers.items()})
        if len(towers) >= 4:
            px = [towers[k] for k in towers]
            tiles = [TOWER_TILES[k] for k in towers]
            h = g.Homography.from_correspondences(px, tiles)
            corners_px = [h.tile_to_pixel(c, r, centre=False) for c, r in ((0, 0), (g.COLS, 0), (g.COLS, g.ROWS), (0, g.ROWS))]
            method = f"towers({len(towers)})"
            calib.tower_anchors = [{"name": k, "px_frac": [x / cw, y / ch], "tile": list(TOWER_TILES[k])} for k, (x, y) in towers.items()]
            # residual: how far each tower base is from its grid position after the fit
            res = [np.hypot(*(np.array(h.pixel_to_tile_f(*towers[k])) - np.array(TOWER_TILES[k]))) for k in towers]
            calib.notes["tower_fit_residual_tiles"] = [round(float(r), 3) for r in res]
            print("fit residual (tiles):", calib.notes["tower_fit_residual_tiles"])
        else:
            print("fewer than 4 towers detected; falling back to default corners")
    if corners_px is None:
        corners_px = [(x * cw, y * ch) for x, y in FALLBACK_CORNERS]
    calib.arena_corners_frac = [[x / cw, y / ch] for x, y in corners_px]
    h = g.Homography.from_corners(corners_px)
    calib.set_homography_from_pixels(h.H, cw, ch)
    calib.notes["arena_method"] = method
    calib.notes["tile_px_bottom"] = [round(v, 2) for v in np.subtract(h.tile_to_pixel(1, 0), h.tile_to_pixel(0, 0))]
    calib.notes["tile_px_top"] = [round(v, 2) for v in np.subtract(h.tile_to_pixel(1, 31), h.tile_to_pixel(0, 31))]
    rb = h.tile_to_pixel(9, 0, centre=False)[1] - h.tile_to_pixel(9, 1, centre=False)[1]
    rt = h.tile_to_pixel(9, 30, centre=False)[1] - h.tile_to_pixel(9, 31, centre=False)[1]
    calib.notes["row_height_px_bottom_vs_top"] = [round(rb, 2), round(rt, 2)]
    calib.save(a.out)

    ver = content.copy()
    draw_grid(ver, h)
    draw_rois(ver, calib.rois)
    for k, (x, y) in (calib.tower_anchors and {t["name"]: (t["px_frac"][0] * cw, t["px_frac"][1] * ch) for t in calib.tower_anchors} or {}).items():
        cv2.circle(ver, (int(x), int(y)), 5, (0, 0, 255), -1)
        cv2.putText(ver, k, (int(x) + 6, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    vpath = a.verify or (str(Path(a.out).with_suffix("")) + ".verify.png")
    cv2.imwrite(vpath, ver)
    print(f"wrote {a.out} (arena via {method}) and {vpath}; tile px bottom={calib.notes['tile_px_bottom']} top={calib.notes['tile_px_top']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

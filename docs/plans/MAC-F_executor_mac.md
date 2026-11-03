# MAC-F — Mac executor: tile → screen point, Quartz taps behind a double guard, overlay mode (spec for a Sonnet implementer)

Author: Opus · 2026-11-02 · Implements D-003, D-031, D-035. Read `docs/plans/MAC-00_common.md` first.
**You must never send a real click.** Tests use a recording `poster`; never set `CLASHBOT_ALLOW_LIVE`.

## Owns (only these files)
```
clashbot/executor.py         (extend; keep Calibration, homography, tile_to_screen, DryRunExecutor behaviour)
tools/executor_overlay.py    (new)
tests/test_executor_mac.py   (new)
```

## Contract (package G codes against this — keep it exact)
```python
# clashbot/executor.py
LIVE_ENV = "CLASHBOT_ALLOW_LIVE"
GRID_COLS, GRID_ROWS = 18, 32
ARENA_TILE_CORNERS = [[0.0, 0.0], [18.0, 0.0], [18.0, 32.0], [0.0, 32.0]]   # bl, br, tr, tl (row 0 = own side)

@dataclass
class ScreenMap:
    rect: list[float]                     # [left, top, width, height] of the captured window, global screen POINTS
    content_rect_frac: list[float]        # [x, y, w, h] as fractions of the raw frame (calib.json)
    arena_corners_frac: list[list[float]] # 4 x [x, y] content fractions: bl, br, tr, tl (calib.json)
    hand_rois: list[list[float]]          # rois hand_0..hand_3, [x, y, w, h] content fractions (calib.json)

def screen_map_from_calib(calib: dict | str | Path, rect) -> ScreenMap
def content_to_screen(fx: float, fy: float, smap: ScreenMap) -> tuple[float, float]
def screen_to_image(x: float, y: float, smap: ScreenMap, frame_w: int, frame_h: int) -> tuple[float, float]
def calibration_from_screen_map(smap: ScreenMap) -> Calibration
def card_slot_points(smap: ScreenMap) -> list[list[float]]
def tile_target(tile: list[float]) -> list[float]
def check_live_allowed() -> None                 # raises PermissionError unless darwin and env LIVE_ENV == "1"
def quartz_click(x: float, y: float) -> None     # the ONLY function that posts input

class DryRunExecutor:
    def execute(self, key: str, tile: list[float] | None, t: float, hand: list | None = None) -> dict

class MirroringExecutor:
    def __init__(self, calibration: Calibration, card_slot_pixels: list[list[float]], live: bool = False,
                 poster=None, bounds: list[float] | None = None, click_gap_s: float = 0.12, sleep=time.sleep): ...
    @classmethod
    def from_calib(cls, calib: dict | str | Path, rect, live: bool = False, **kw) -> "MirroringExecutor"
    def plan_clicks(self, key: str, tile: list[float] | None, hand: list | None) -> dict
    def execute(self, key: str, tile: list[float] | None, t: float, hand: list | None = None) -> dict
    log: list[dict]
```
`rect` arguments accept an object with `left, top, width, height` attributes (e.g. `cr_perception.sources.WindowRect`) or a 4-sequence.

## 1. Geometry (all exact)
- `screen_map_from_calib`: `calib` may be a dict, or a path to `calib.json` (`json.loads(Path(p).read_text())`). Fields: `content_rect_frac` (default `[0,0,1,1]`), `arena_corners_frac` (must have 4 entries, else `ValueError("calib.json has no arena_corners_frac; run tools/calibrate.py")`), `rois["hand_0".."hand_3"]` (fall back to `cr_perception.hud.DEFAULT_ROIS` per missing slot — import lazily inside the function).
- `content_to_screen(fx, fy)`: `cx, cy, cw, ch = content_rect_frac`; `rx, ry = cx + fx*cw, cy + fy*ch`; `L, T, W, H = rect`; return `(L + rx*W, T + ry*H)`.
- `screen_to_image(x, y, smap, fw, fh)`: `((x - L) / W * fw, (y - T) / H * fh)` (screen points back into captured-image pixels; Retina frames are 2× the points).
- `calibration_from_screen_map`: `Calibration(src_tiles=ARENA_TILE_CORNERS, dst_pixels=[list(content_to_screen(x, y, smap)) for x, y in smap.arena_corners_frac])`. (The arena corners in calib.json are the images of exactly these tile corners, see `tools/calibrate.py`, so a 4-point homography reproduces the perception mapping.)
- `card_slot_points`: centre of each hand ROI: `content_to_screen(x + w/2, y + h/2)` as `[x, y]` lists, slots 0..3.
- `tile_target(tile)`: tap the centre of the tile cell: `[min(max(tile[0] + 0.5, 0.5), 17.5), min(max(tile[1] + 0.5, 0.5), 31.5)]`.

## 2. Live guard (D-031) and Quartz
- `check_live_allowed()`: `if sys.platform != "darwin": raise PermissionError("live taps need macOS + iPhone Mirroring")`; `if os.environ.get(LIVE_ENV) != "1": raise PermissionError(f"live taps are disabled; the user enables them with --live and {LIVE_ENV}=1 (Training Camp / friendly battles only)")`.
- `quartz_click(x, y)`: `import Quartz` inside; post `kCGEventMouseMoved`, `kCGEventLeftMouseDown`, `kCGEventLeftMouseUp` at `(float(x), float(y))` with `kCGMouseButtonLeft` via `CGEventCreateMouseEvent` + `CGEventPost(kCGHIDEventTap, ev)`, `time.sleep(0.02)` between events. This is the only place in the repo allowed to contain `CGEventPost`.

## 3. Executors
- `DryRunExecutor.execute(key, tile, t, hand=None)`: unchanged entry `{"t", "key", "tile"}` (ignore `hand`); old 3-argument calls keep working.
- `MirroringExecutor.__init__`: `self.H = homography(calibration)`; `self.card_slot_pixels`; `self.live = live`; if `live`: `check_live_allowed()` (raises at construction). `self.poster = poster or quartz_click`; `self.bounds = bounds`; `self.log = []`. Remove the old `NotImplementedError` stubs.
- `from_calib(calib, rect, live=False, **kw)`: `smap = screen_map_from_calib(calib, rect)`; `cls(calibration_from_screen_map(smap), card_slot_points(smap), live=live, bounds=list(smap.rect), **kw)`; also store `self.screen_map = smap` on the instance (attribute `None` when built directly).
- `plan_clicks(key, tile, hand)` → dict:
  - `key == "hold"` or `tile is None` → `{"status": "hold", "slot": None, "clicks": []}`
  - card = `key.split("|")[0]`; `hand` is `None` or card not in `hand` → `{"status": "skipped_card_not_in_hand", "slot": None, "clicks": []}`
  - `slot = hand.index(card)`; `target = tile_to_screen(tile_target(tile), self.H)`; `clicks = [list(self.card_slot_pixels[slot]), [target[0], target[1]]]`
  - if `bounds` and any click is outside `[L, L+W] x [T, T+H]` → `{"status": "skipped_out_of_window", "slot": slot, "clicks": clicks}`
  - else `{"status": "planned", "slot": slot, "clicks": clicks}`.
- `execute(key, tile, t, hand=None)`: `plan = self.plan_clicks(...)`; `entry = {"t": t, "key": key, "tile": tile, **plan, "sent": False}`; if `self.live and plan["status"] == "planned"`: `check_live_allowed()` again (env may have been removed), then for each click `self.poster(x, y)` and `self.sleep(self.click_gap_s)`; `entry["sent"] = True`. Append to `self.log`; return entry.

## 4. `tools/executor_overlay.py` (never live)
```
$PY tools/executor_overlay.py --calib CALIB.json --image FRAME.png [--rect l,t,w,h] [--keys k1,k2,...]
                              [--hand c1,c2,c3,c4] [--prior dataset/v1/placement_prior.json] [--out overlay.png]
```
- `rect` default `0,0,image_w,image_h` (screen points == image pixels). With a real Retina capture pass the window rect in points; drawing maps back with `screen_to_image`.
- `ex = MirroringExecutor.from_calib(calib, rect, live=False)` — never `live=True`.
- `--keys` default: four keys spread over the arena, `"<c0>|left|bridge,<c1>|right|mid,<c2>|middle|back,<c3>|right|enemy_half"` with `c0..c3` from `--hand` (default hand `knight,archers,fireball,hog-rider`). Tile per key: `clashbot.actions.placement_tile(key, prior)` if `--prior` exists, else the lane/zone centre (`left`→col 3, `middle`→9, `right`→14; `back`→row 2, `mid`→8, `bridge`→14, `enemy_half`→20).
- Draw on a copy of the image (cv2): the 19×33 grid lines of the executor homography (tile → screen → image), the four card slot points (white circles, slot index text), and per key a coloured circle at the slot, a cross at the target, a line between and the key text. Write `--out` (default `<image stem>.executor_overlay.png`).
- Print JSON: `[{"key", "tile", "status", "clicks_screen", "clicks_image"}]`.

## 5. Tests — `tests/test_executor_mac.py`
Synthetic calib dict: `content_rect_frac [0.0, 0.1, 1.0, 0.8]`, arena corners `[[0.1,0.75],[0.9,0.75],[0.88,0.1],[0.12,0.1]]`, hand ROIs as in `DEFAULT_ROIS`; rect `[100, 50, 300, 600]`.
1. `content_to_screen(0.5, 0.5)` == `(250.0, 350.0)`; `screen_to_image` inverts it for a 600×1200 frame.
2. Executor homography maps tile corners `(0,0)` → the bl screen point and `(18,32)` → the tr screen point (1e-6).
3. Perception parity: build `cr_perception.geometry.Homography.from_corners` on the corners in **content pixels** for a 300×480 content size; for tiles `[(0,0), (8,14), (17,31), (3,4)]` the perception centre pixel `tile_to_pixel(c, r, centre=True)` → content fraction → `content_to_screen` equals `tile_to_screen(tile_target([c, r]), H_exec)` within 1e-6.
4. `card_slot_points` are the ROI centres mapped to the screen.
5. `plan_clicks`: hold → no clicks; card not in hand → `skipped_card_not_in_hand`; hand `["a","b","knight","c"]` + `knight|left|bridge` → slot 2, first click = slot-2 point, status `planned`; a target outside `bounds` → `skipped_out_of_window`.
6. `live=False`: `execute` never calls the recording poster; `sent is False`.
7. `live=True` without env → `PermissionError` at construction. With `monkeypatch.setenv(LIVE_ENV, "1")` and `monkeypatch.setattr(sys, "platform", "darwin")` and a recording poster + no-op sleep: `execute` calls the poster exactly twice with the planned points, `sent is True`; then `monkeypatch.delenv(LIVE_ENV)` → next `execute` raises `PermissionError` and the poster is not called again.
8. `DryRunExecutor().execute("hold", None, 1.0)` and `execute(k, tile, 1.0, hand=[...])` both work; entries have exactly keys `t, key, tile`.
9. Old API: `Calibration`, `homography`, `tile_to_screen` unchanged (the axis-aligned case from `tests/test_runtime.py` passes).
10. Overlay tool on a synthetic 600×1200 image + the synthetic calib written to `tmp_path`: `main([...])` writes the PNG and returns/prints 4 entries.
11. Static: in `clashbot/executor.py`, `CGEventPost` appears only inside `quartz_click`; no file under `clashbot/`, `cr_perception/` or `tools/` other than `clashbot/executor.py` contains `CGEventPost`.

## 6. Smoke (allowed once)
`$PY -m pytest -q tests/test_executor_mac.py tests/test_runtime.py`. No real calibration exists yet; the overlay on a real mirrored frame is Wave 4.

## 7. Heavy run (Opus with the user, Wave 4)
Overlay on a captured iPhone Mirroring frame after calibration; visual check only, no click.

## 8. Report
Per MAC-00 §5.

---
## Amendment 1 (Opus review, 2026-11-02) — FIX_LIST #3 (executor half; package G does the loop half in parallel)
The capture source follows a moved iPhone Mirroring window, but the executor keeps the rect it was built with, so live taps would land off by the move.
Add `MirroringExecutor.update_rect(self, rect) -> None`: requires `self.screen_map` (built via `from_calib`; else `ValueError("update_rect needs an executor built with from_calib")`); `smap = dataclasses.replace(self.screen_map, rect=_rect_list(rect))`; recompute `self.calibration = calibration_from_screen_map(smap)`, `self.H = homography(self.calibration)`, `self.card_slot_pixels = card_slot_points(smap)`, `self.bounds = list(smap.rect)`, `self.screen_map = smap`; leave `live` and `poster` unchanged; append `{"event": "rect_update", "rect": list(smap.rect)}` to `self.log`.
Tests: after `update_rect` shifted by (+40, +25) every planned click shifts by exactly (+40, +25) (1e-6); a directly-constructed executor raises `ValueError`; `live` flag unchanged.

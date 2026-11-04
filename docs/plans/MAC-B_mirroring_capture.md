# MAC-B — iPhone Mirroring capture source (spec for a Sonnet implementer)

Author: Opus · 2026-11-02 · Implements D-001. Read `docs/plans/MAC-00_common.md` first.

## Owns (only these files)
```
cr_perception/sources.py         (ADD the functions/class below; do not change existing classes)
tools/benchmark_capture.py       (rewrite)
tests/test_mirroring_source.py   (new)
```

## Contract (other packages code against this — keep it exact)
```python
# cr_perception/sources.py
MIRRORING_OWNERS = ("iPhone Mirroring",)

def list_windows() -> list[dict]
def find_mirroring_window(owner_hint: str | None = None, windows: list[dict] | None = None) -> WindowRect | None

class MirroringSource(FrameSource):
    def __init__(self, rect: WindowRect | None = None, target_fps: float = 10.0, owner_hint: str | None = None,
                 grab_fn=None, clock=time.perf_counter, sleep=time.sleep, check_permission: bool = True,
                 refind_every_s: float = 2.0): ...
    rect: WindowRect                 # current window rect in global screen POINTS
    last_grab_ms: float              # duration of the most recent grab (0.0 before the first)
    def frames(self) -> Iterator[tuple[np.ndarray, float, int]]   # (BGR frame, t seconds since start, index)
    def grab(self) -> np.ndarray     # one BGR frame now
    def stats(self) -> dict
    def close(self) -> None
```
Package G constructs `MirroringSource(target_fps=...)` and iterates `frames()`; package F reads `source.rect` to map screen points.

## 1. `list_windows()`
macOS only; returns `[]` when `sys.platform != "darwin"` or `import Quartz` fails. Same Quartz call as `find_emulator_window` (`kCGWindowListOptionOnScreenOnly | kCGWindowListExcludeDesktopElements`). One dict per window:
`{"owner": str(kCGWindowOwnerName), "title": str(kCGWindowName or ""), "window_id": int(kCGWindowNumber), "layer": int(kCGWindowLayer), "rect": WindowRect(X, Y, Width, Height, owner, title)}` (ints).

## 2. `find_mirroring_window(owner_hint=None, windows=None)`
`windows = list_windows() if windows is None else windows`. Hints = `(owner_hint,)` if given else `MIRRORING_OWNERS`. Keep windows whose `owner` contains any hint (case-insensitive), whose `layer == 0`, and whose rect has `width >= 100 and height >= 100`. Return the rect with the largest area, else `None`.

## 3. `MirroringSource`
- `__init__`: if `rect is None`: `rect = find_mirroring_window(owner_hint)`; if still `None` raise `RuntimeError("iPhone Mirroring window not found: open iPhone Mirroring, keep the window visible (not minimised), or pass a manual rect")`. Store `self._auto = rect_was_none`. `self.mon = {"left","top","width","height"}` from the rect. `grab_fn`: if `None`, lazily `import mss` and use `self._sct = mss.mss()`, grab = `np.asarray(self._sct.grab(mon))[:, :, :3].copy()` (BGRA→BGR, same as `ScreenSource.grab`). Store `clock`, `sleep`, `self.fps = target_fps`. Counters: `frames_out = 0`, `dropped = 0`, `rect_changes = 0`, `_grab_ms: list[float]`, `_t0 = None`, `_last_shape = None`, `looks_like_game = None`.
- Permission check (when `check_permission`): `frame = self.grab()`; `frame.size == 0` → `RuntimeError("empty capture")`; `float(frame.std()) < 2.0` → `PermissionError` with the exact message used by `ScreenSource._check_permission` (copy the text). Then `from .screen import looks_like_game` and set `self.looks_like_game = bool(looks_like_game(frame))` — **informational only, never raise** (the phone may still be on its home screen).
- `grab()`: calls the grab function with `self.mon`, records `_last_shape = frame.shape`.
- `frames()` — steady-rate deadline scheduler using only the injected `clock`/`sleep`:
  ```
  period = 1.0 / fps if fps and fps > 0 else 0.0
  t0 = clock(); self._t0 = t0; next_due = t0; last_refind = t0; i = 0
  loop forever:
      now = clock()
      if period and now < next_due: sleep(next_due - now); now = clock()
      if period:
          late = int((now - next_due) // period)        # whole periods we are behind
          if late > 0: self.dropped += late; next_due += late * period
      if self._auto and now - last_refind >= refind_every_s:
          r = find_mirroring_window(); last_refind = now
          if r is not None and (r.left, r.top, r.width, r.height) != (self.rect.left, self.rect.top, self.rect.width, self.rect.height):
              self.rect = r; self.mon = {...from r...}; self.rect_changes += 1
      g0 = clock(); frame = self.grab(); g1 = clock()
      self._grab_ms.append((g1 - g0) * 1000.0); self.last_grab_ms = self._grab_ms[-1]
      yield frame, g0 - t0, i
      i += 1; self.frames_out += 1
      if period: next_due += period
  ```
- `stats()` → `{"frames": frames_out, "dropped": dropped, "elapsed_s": round(clock() - _t0, 3) (0.0 before start), "fps": round(frames_out / elapsed, 2) or 0.0, "target_fps": fps, "grab_ms_p50", "grab_ms_p95" (numpy percentile, rounded 2; None if no grabs), "frame_shape": list(_last_shape[:2]) or None, "scale": round(frame_w / rect.width, 3) or None, "rect": dict(rect.__dict__), "rect_changes", "looks_like_game"}`.
- `close()`: close `_sct` if it exists.
- Update the module docstring list with one line: `MirroringSource - live capture of the macOS "iPhone Mirroring" window at a steady rate (D-001).`
- **Read-only**: the file must not contain `CGEventCreate`, `CGEventPost` or any input call.

## 4. `tools/benchmark_capture.py` (rewrite; read-only)
```
$PY tools/benchmark_capture.py [--source mirroring|emulator] [--owner NAME] [--rect l,t,w,h]
                               [--fps 0] [--sweep 10,20,30,60] [--seconds 5] [--save-frame PATH] [--list-windows]
```
- `--list-windows`: print one line per `list_windows()` entry (`owner | title | rect | layer`) and exit 0.
- Source `mirroring` (default): `MirroringSource(rect, target_fps=fps, owner_hint=a.owner)`; `emulator`: old behaviour (`ScreenSource`). Manual `--rect` → `WindowRect(..., "manual")`.
- Capture errors (`RuntimeError`, `PermissionError`) → print `CAPTURE CHECK FAILED: <msg>` and return 3 (no traceback).
- Without `--sweep`: run `frames()` for `--seconds`, print one JSON line `{"rect", **source.stats()}`.
- With `--sweep`: for each target fps a new source, same run, print one JSON line per target, then a markdown table `| target fps | achieved fps | dropped | grab p50 ms | grab p95 ms |`.
- `--save-frame PATH`: write the first captured frame with `cv2.imwrite` (used for calibration in Wave 4).
- Keep `sys.path.insert(0, ...)` pattern of the old file.

## 5. Tests — `tests/test_mirroring_source.py` (no real capture)
Helpers in the test file:
```python
class FakeClock:
    def __init__(self): self.now = 100.0
    def clock(self): return self.now
    def sleep(self, dt): self.now += max(0.0, dt)
def make_grab(clock, cost_s, shape=(40, 30, 3)):
    rng = np.random.default_rng(0)
    def grab(mon):
        clock.now += cost_s
        return rng.integers(0, 255, size=shape, dtype=np.uint8)
    return grab
WIN = lambda owner, w, h, layer=0: {"owner": owner, "title": "", "window_id": 1, "layer": layer,
                                   "rect": WindowRect(0, 0, w, h, owner, "")}
```
1. `find_mirroring_window(windows=[WIN("Finder", 800, 600), WIN("iPhone Mirroring", 300, 650), WIN("iPhone Mirroring", 50, 50)])` returns the 300×650 rect; layer-1 window ignored; no match → `None`; `owner_hint="finder"` finds Finder.
2. Rect given + `check_permission=True` with a grab returning `np.zeros((40,30,3), np.uint8)` → `PermissionError`; returning an empty array → `RuntimeError`.
3. Steady rate: `target_fps=10`, grab cost 0.01 s, iterate `frames()` until `t >= 1.0` → between 10 and 11 frames, consecutive `t` deltas ≈ 0.1 (abs tol 0.02), `dropped == 0`, `stats()["fps"]` within 9.5..11.
4. Too slow: `target_fps=10`, grab cost 0.25 s, 8 frames → `dropped >= 10`, `t` strictly increasing.
5. `target_fps=0` → no sleeps (FakeClock advances only by grab cost); 5 frames.
6. Refind: `_auto` source (rect=None) with `find_mirroring_window` monkeypatched to return a first rect, then a moved rect → after `refind_every_s` of fake time `rect_changes == 1` and `mon["left"]` updated.
7. `stats()` keys exactly as listed; `scale == 30 / rect.width` for a 40×30 frame.
8. Static: `Path("cr_perception/sources.py").read_text()` contains none of `CGEventCreate`, `CGEventPost`.
9. `subprocess.run([sys.executable, "tools/benchmark_capture.py", "--help"])` returns 0.
Use `rect=WindowRect(0, 0, 300, 650)` and `check_permission=False` unless the test is about the permission check.

## 6. Smoke (allowed once)
`$PY tools/benchmark_capture.py --list-windows` (prints window owners; harmless). Do **not** run a capture (needs the user's Screen Recording permission).

## 7. Heavy run (Opus with the user, Wave 4)
`$PY tools/benchmark_capture.py --sweep 10,20,30,60 --seconds 5 --save-frame runs/poc/mirror_frame.png` with iPhone Mirroring open.

## 8. Report
Per MAC-00 §5.

---
## Amendment 1 (Opus, 2026-11-03) — FIX_LIST #17: capture the window itself (Quartz), at Retina resolution
On the real Mac the mss region grab is 1× (316×696 px for a 316×696-point window) and contains other windows showing through the mirroring window's transparent margins. Capturing the window by id gives only its own pixels at 2× (verified by Opus: 632×1392, ~22 ms/grab, black margins that `detect_content_rect` strips).
1. `find_mirroring_window_info(owner_hint=None, windows=None) -> dict | None`: same selection as `find_mirroring_window` but returns the whole window dict (incl. `window_id`); `find_mirroring_window` returns `info["rect"]` from it.
2. `quartz_window_grab(window_id: int) -> np.ndarray`: `import Quartz` inside; `img = Quartz.CGWindowListCreateImage(Quartz.CGRectNull, Quartz.kCGWindowListOptionIncludingWindow, window_id, Quartz.kCGWindowImageBoundsIgnoreFraming | Quartz.kCGWindowImageBestResolution)`; `None` → `RuntimeError("window capture failed (window closed or no Screen Recording permission)")`; width/height/bytes-per-row via `CGImageGetWidth/Height/BytesPerRow`; bytes via `CGDataProviderCopyData(CGImageGetDataProvider(img))`; `np.frombuffer(..., np.uint8).reshape(h, bpr // 4, 4)[:, :w, :3].copy()` (BGRA → BGR).
3. `MirroringSource(..., backend: str = "quartz", window_id: int | None = None)`: when `grab_fn` is None: `backend == "quartz"` and a window id is known (given, or from `find_mirroring_window_info` when `rect` is None) → `grab_fn = lambda mon: quartz_window_grab(self.window_id)`; otherwise the existing mss grab. A manual `rect` without `window_id` uses mss. Store `self.backend` (the one actually used) and `self.window_id`. On refind (`_auto`), use `find_mirroring_window_info` and update `window_id` too. `stats()` adds `"backend"`, `"window_id"`. Unknown backend → `ValueError`.
4. `tools/benchmark_capture.py`: `--backend quartz|mss` (default quartz), passed through; `--list-windows` prints `window_id` too.
5. Tests (fakes, no real capture): backend selection (quartz with a window id, mss fallback with a manual rect, bad backend raises); `quartz_window_grab` with a monkeypatched fake `Quartz` module in `sys.modules` (fake CGImage functions returning a 4×3 BGRA buffer with padding in bytes-per-row) → returns the right (4, 3, 3) BGR array; `None` image → `RuntimeError`; refind updates `window_id`; still no `CGEventCreate`/`CGEventPost` in the file.

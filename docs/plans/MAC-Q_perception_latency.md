# MAC-Q — perception latency for the live loop: hand-read cache + background OCR (spec for a Sonnet implementer)

Author: Opus · 2026-11-02 · FIX_LIST #15, D-038. Read `docs/plans/MAC-00_common.md` first.

## Owns (only these files)
```
cr_perception/perception.py      (hand cache + async OCR; default behaviour unchanged)
tests/test_perception_latency.py (new)
```

## Measurements (live dry run on the recorded game, M5, KataCR on MPS)
Per frame: hand reader 77 ms **every frame**; clock OCR ≈ 290 ms every 10th frame; deploy-label OCR ≈ 250 ms every 15th; detector 91 ms every 6th. Decision latency 171 ms p50 / **471 ms p95** (target < 300 ms p95). The p95 comes from OCR frames.

## 1. Hand-read cache (always on; parity-safe)
In `process`, around `hand = self.hand_reader.read(content)`:
- Signature per slot `i` (ROIs `hand_0..hand_3` via `crop_roi`): `cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (16, 20), interpolation=cv2.INTER_AREA).astype(np.float32)`; also the `next_card` ROI.
- Reuse the previous **raw** read (`self._last_hand_raw`, a dict exactly as returned by `hand_reader.read`) when all 5 signatures differ from the previous signatures by mean-abs < `HAND_CACHE_DIFF` (= 4.0) **and** `t - self._last_hand_read_t < HAND_CACHE_MAX_AGE_S` (= 1.0); else call `hand_reader.read` and store raw result, signatures and `t`. `_smooth_hand` and everything after it is unchanged. Record `self.hand_cache_hits` / `self.hand_cache_misses`. The `hand` timing tick still measures the whole block.
- `reset_match` clears the cache.

## 2. Background OCR (opt-in: `async_ocr: bool = False` dataclass field)
With `async_ocr=False` the code path is exactly today's (offline video runs and the dataset parity stay identical). With `async_ocr=True`:
- One worker: `concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="cr-ocr")`, created lazily; the OCR engine is used **only** on that worker thread (clock and labels share it sequentially).
- **Clock**: when the clock gate fires (`n % ocr_every == 0`) and no clock job is pending: submit a pure function on copies of the two crops (`crop_roi(content, rois["clock"]).copy()` and the taller `[0.72, 0.0, 0.28, 0.12]` crop) that returns `(s, c)` with the same fallback logic as today (main crop; if `parse_clock` fails, the alt crop + `\d{1,2}:\d\d` regex). Each frame: if the pending job is done, take `(s, c)` and run the same acceptance as today (`parse_clock(s) is not None and c >= 0.6 and self.clock.valid(...)`) to set `clock_s, clock_conf, remaining`; otherwise fall through to the template-digit path exactly as today. Factor today's synchronous logic into a helper used by both modes.
- **Labels** (+ tower numbers): when the label gate fires and no label job is pending: submit `self.labels.read(content.copy(), self.calib.arena_crop(cw, ch))` and remember `t_submit`. When done: process `labels, numbers` exactly as today, but pass `t_submit` (not the current `t`) to `self.play.update_labels` and as the `last_good_t` time of tower numbers.
- Tower OCR (`tower_ocr=True`) stays synchronous (off by default).
- `close()` (new public method, also fine to call with `async_ocr=False`): shut the pool down (`wait=False, cancel_futures=True`).
- `timing_report()` gains `"ocr_async": {"submitted", "applied"}` counts when async.

## 3. Tests — `tests/test_perception_latency.py`
Build a synthetic `Perception` like `tests/test_perception_rates.py` / `tests/test_pipeline_synthetic.py` (monkeypatched `assess` → match, `ENTER_FRAMES=1`).
1. Hand cache: 30 identical frames → `hand_reader.read` (wrapped counter) called ≤ 2 times in the first 1.0 s of `t` and the returned hands are identical to the uncached run; change one slot's pixels → the next frame re-reads.
2. `async_ocr=False`: with a fake OCR that sleeps 0.2 s, identical `match_clock` sequence to the current implementation (compare against `n_processed`-based expectations) and OCR is called on the main thread (`threading.current_thread().name` recorded by the fake).
3. `async_ocr=True`: same fake (sleep 0.2 s, returns `"1:59"`, conf 0.9): no `process` call takes ≥ 0.15 s of wall time; after enough frames `state.match_clock == "1:59"`; the fake recorded only `cr-ocr` thread names; `close()` works twice.
4. Async labels: a fake `labels.read` returning one label → `play.update_labels` (wrapped) receives the submit time, not a later frame's time.

## 4. Acceptance
`$PY -m pytest -q tests/test_perception_latency.py tests/test_perception_rates.py tests/test_pipeline_synthetic.py tests/test_hud_synthetic.py` and the whole suite.

## 5. Heavy run (Opus)
Live dry run on the recorded game with `async_ocr` (G's default) → decision total p95 vs the 300 ms target; hand cache hit rate.

## 6. Report
Per MAC-00 §5.

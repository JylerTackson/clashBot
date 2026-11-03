# MAC-P — perception rate gating counts processed frames, not source frame indices (spec for a Sonnet implementer)

Author: Opus · 2026-11-02 · FIX_LIST #12. Read `docs/plans/MAC-00_common.md` first.

## Owns (only these files)
```
cr_perception/perception.py      (rate gating only)
tests/test_perception_rates.py   (new)
```

## Bug (found in the live-state parity run)
`Perception.process(frame, t, idx)` gates the slow readers on `idx % N`, where `idx` is the **source's absolute frame index**. With a strided video source (`VideoFrameSource(..., stride=5)`) that starts at an arbitrary frame (e.g. 23,676), `idx` only takes values ≡ 1 or 6 (mod 10), so `idx % 10 == 0` (clock OCR, `ocr_every=10`) and `idx % 15 == 0` (deploy labels + tower-HP numbers, `label_every=15`) are **never true**: on the parity game the clock was never read (phase/clock agreement 0.0) and no deploy label was ever read. Detection (`detect_every`) and the hand-crop ring buffer (`idx % 3`) fire at the wrong rates for the same reason. Live capture (`MirroringSource`, idx += 1) is unaffected.

## Fix
- Add a processed-frame counter: in `__post_init__` set `self.n_processed = 0`; at the very start of `process`, `n = self.n_processed; self.n_processed += 1`. `reset_match` does **not** reset it.
- Replace every `idx % …` gate in `process` with `n % …` — the six lines found by `grep -n "idx %" cr_perception/perception.py` (hand crops `% 3`, clock OCR `% self.ocr_every`, tower OCR `% self.ocr_every`, labels `% self.label_every`, detection `% self.detect_every`, unit events `% self.detect_every`). Keep `self.frame_i = idx` and `state.frame_index = idx` (the source index stays the frame identity in the output).
- Docstring of the module: one sentence that rates count processed frames (`n_processed`), so they are independent of the source's stride and start.
- No other behaviour change.

## Tests — `tests/test_perception_rates.py`
Build a `Perception` like `tests/test_pipeline_synthetic.py` does (synthetic calib + frames; read that file first and reuse its helpers by import or by copying the minimal frame/calib construction). Replace the slow readers with counters:
1. Feed 60 frames with source indices `23676 + 5*k` (a stride-5 source starting off-grid): monkeypatch `p.ocr.read` (or set `p.ocr` to a fake with `read` and `ocr` attributes) to count calls, set `p.detector` to a fake whose `detect` counts calls and returns `[]`, and force `readiness == "match"` (monkeypatch `cr_perception.perception.assess` to return a `Readiness`-like object with `state="match"`, `conf=1.0` — check `cr_perception/screen.py` for the exact type). Expect clock-OCR passes = 6 (every 10th processed frame, `ocr_every=10`), label passes = 4 (`label_every=15`), detector calls = 20 (`detect_every=3`).
2. Same with consecutive indices `0..59` → identical counts (the gating no longer depends on the index values).
3. `state.frame_index` equals the source index that was passed in.
4. `n_processed` keeps counting across `reset_match`.

## Acceptance
`$PY -m pytest -q tests/test_perception_rates.py tests/test_pipeline_synthetic.py tests/test_hud_synthetic.py`; whole suite (with the symlinks) green.

## Heavy run (Opus, at integration)
Re-run the parity perception (`tools/run_video.py ... --stride 5`) and `live_state_parity.py`; the clock/phase/towers agreement should no longer be 0.

## Report
Per MAC-00 §5.

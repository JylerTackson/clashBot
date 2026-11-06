"""Decision screenshots (MAC-AF, `--record-shots`): what the bot saw and where it tapped.

For every decision that reaches the executor or becomes a hold record, `ShotRecorder` saves
`shots/d{idx:04d}_t{t:.1f}_a.jpg` (the frame the decision used at half size, with the 4 hand slot ROIs and the tracked
hand labels, the card-slot tap point in cyan, the field tap point in magenta with its tile, and the elixir reading). A
tap is also followed: the first frames >= 0.5 s and >= 1.5 s later are saved as `_b` / `_c`, unannotated but for a small
header. `shots/index.jsonl` has one line per decision (fields in `ShotRecorder._row`), written once the `_after` fields
are filled (or at `close`). Elixir and hand readings come from the raw perception state when the loop passes it (the
builder's elixir is capped by its own tap prediction and its hand memory refills a played slot at once), else from the
builder's state.

The loop only queues work. Drawing and JPEG encoding run on one background thread behind a bounded queue; a full queue
drops the image (the index keeps its row with a null file) and counts `shots_dropped`. It frees the frames' references
as the thread finishes. It only observes: it reads frames and executor entries and never sends input.
"""
from __future__ import annotations

import json
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from clashbot.config import HOLD_KEY
from clashbot.executor import MirroringExecutor, screen_to_image

log = logging.getLogger("clashbot.shots")

SCALE = 0.5                    # saved images are this fraction of the captured frame
JPEG_QUALITY = 80
AFTER_S = (0.5, 1.5)           # a tap is followed by the first frame >= each delay (the _b and _c images)
TAGS = ("a", "b", "c")
QUEUE_SIZE = 8                 # images waiting for the thread; a full queue drops
SHOTS_MAX = 400                # decisions recorded per run (about 60 KB per image)
TAP_STATUSES = ("planned", "dry_run")      # executor statuses that mean "a tap was (or would have been) made"

CYAN, MAGENTA, GREEN, WHITE, BLACK = (255, 255, 0), (255, 0, 255), (0, 220, 0), (255, 255, 255), (0, 0, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX


# ----------------------------------------------------------------------------- readings

def _num(v) -> float | None:
    return round(float(v), 2) if isinstance(v, (int, float, np.number)) and not isinstance(v, bool) else None


def readings(gs, state: dict | None) -> dict:
    """The elixir and hand of one frame: `elixir` / `hand_raw` from perception (`gs.own`, unread = None), `elixir_state` /
    `hand` from the builder's state. Without a perception `own` the raw elixir falls back to the builder's."""
    own_raw = gs.get("own") if isinstance(gs, dict) else getattr(gs, "own", None)
    own_raw = own_raw if isinstance(own_raw, dict) else None
    own = (state or {}).get("own") or {}
    elixir_state = _num(own.get("elixir"))
    elixir = _num(own_raw.get("elixir")) if own_raw is not None else elixir_state
    hand_raw = own_raw.get("hand") if own_raw is not None else None
    return {"elixir": elixir, "elixir_state": elixir_state, "hand": list(own["hand"]) if own.get("hand") else None,
            "hand_raw": list(hand_raw) if hand_raw else None, "hand_conf": _num(own.get("hand_confidence"))}


def roi_boxes(smap, width: int, height: int) -> list[list[float]]:
    """The 4 hand ROIs of a ScreenMap as [x0, y0, x1, y1] pixels of a width x height frame."""
    cx, cy, cw, ch = smap.content_rect_frac
    return [[(cx + x * cw) * width, (cy + y * ch) * height, (cx + (x + w) * cw) * width, (cy + (y + h) * ch) * height]
            for x, y, w, h in smap.hand_rois]


# ----------------------------------------------------------------------------- drawing (thread)

def _text(img, s: str, org: tuple[int, int], color=WHITE, scale: float = 0.4) -> None:
    cv2.putText(img, s, org, FONT, scale, BLACK, 3, cv2.LINE_AA)
    cv2.putText(img, s, org, FONT, scale, color, 1, cv2.LINE_AA)


def _to_bgr(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame


def annotate_decision(img: np.ndarray, spec: dict, scale: float) -> None:
    """Draw the `a` annotations on the (already scaled) image: the header, the hand ROIs with their labels, the tap points."""
    h, w = img.shape[:2]
    hand = spec.get("hand") or []
    tap = spec.get("tap") or {}
    for i, box in enumerate(spec.get("rois") or []):
        x0, y0, x1, y1 = (int(round(v * scale)) for v in box)
        tapped = tap.get("slot") == i
        cv2.rectangle(img, (x0, y0), (x1, y1), CYAN if tapped else GREEN, 2 if tapped else 1)
        card = hand[i] if i < len(hand) and hand[i] else "?"
        _text(img, f"{i}:{card}"[:11], (x0 + 2, max(y0 - 4 - 11 * (i % 2), 10)), CYAN if tapped else WHITE, 0.33)
    slot_pt, field_pt = tap.get("slot_pt"), tap.get("field_pt")
    if slot_pt is not None:
        c = (int(round(slot_pt[0] * scale)), int(round(slot_pt[1] * scale)))
        cv2.circle(img, c, 6, CYAN, -1)
        cv2.circle(img, c, 7, BLACK, 1)
    if field_pt is not None:
        c = (int(round(field_pt[0] * scale)), int(round(field_pt[1] * scale)))
        cv2.circle(img, c, 8, MAGENTA, 2)
        cv2.line(img, (c[0] - 12, c[1]), (c[0] + 12, c[1]), MAGENTA, 1)
        cv2.line(img, (c[0], c[1] - 12), (c[0], c[1] + 12), MAGENTA, 1)
        if tap.get("tile") is not None:
            label = f"tile {tap['tile'][0]:g},{tap['tile'][1]:g}"
            _text(img, label, (min(max(c[0] + 10, 2), max(w - 90, 2)), min(max(c[1] - 10, 12), h - 4)), MAGENTA)
    _text(img, spec.get("header", ""), (4, 12))
    if spec.get("header2"):
        _text(img, spec["header2"], (4, 26))


def annotate_after(img: np.ndarray, spec: dict) -> None:
    """The `b` / `c` header only."""
    _text(img, spec.get("header", ""), (4, 12))


def write_image(job: dict, scale: float = SCALE, quality: int = JPEG_QUALITY) -> bool:
    """Scale, annotate and encode one queued image; True when the file was written."""
    img = _to_bgr(np.asarray(job["frame"]))
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    if job["kind"] == "a":
        annotate_decision(img, job["spec"], scale)
    else:
        annotate_after(img, job["spec"])
    return bool(cv2.imwrite(str(job["path"]), img, [cv2.IMWRITE_JPEG_QUALITY, quality]))


# ----------------------------------------------------------------------------- recorder

@dataclass
class _Pending:
    row: dict
    t: float
    todo: list = field(default_factory=list)       # [(delay_s, tag)] still to capture, in order


class ShotRecorder:
    """Records decision screenshots and the index. `on_frame` / `on_decision` run on the loop's thread and only queue work;
    `close` finishes the index and joins the writer thread. `calib` (a calib.json path or dict) lets a dry run without a
    screen map (video source) still show where the tap would land: the clicks are planned on a virtual window the size
    of the frame (a MirroringExecutor that is never live and never executes). `writer` replaces the image writer (tests)."""

    def __init__(self, out_dir: Path | str, calib=None, max_decisions: int = SHOTS_MAX, queue_size: int = QUEUE_SIZE,
                 scale: float = SCALE, quality: int = JPEG_QUALITY, writer=None):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.calib = calib
        self.max_decisions = max_decisions
        self.scale = scale
        self.quality = quality
        self._writer = writer or (lambda job: write_image(job, self.scale, self.quality))
        self.decisions = 0            # decisions recorded
        self.taps = 0                 # of which followed for b / c
        self.written = 0              # images the thread wrote
        self.shots_dropped = 0        # images dropped because the queue was full
        self.errors = 0               # images the thread failed to write
        self.capped = False           # `max_decisions` reached: later decisions are not recorded
        self._pending: list[_Pending] = []
        self._closed = False
        self._virtual: MirroringExecutor | None = None
        self._virtual_wh: tuple[int, int] | None = None
        self._index = open(self.dir / "index.jsonl", "w")
        self._q: queue.Queue = queue.Queue(maxsize=queue_size)
        self._thread = threading.Thread(target=self._run, name="shots", daemon=True)
        self._thread.start()

    # -- the writer thread ----------------------------------------------------------------------------------------
    def _run(self) -> None:
        while True:
            job = self._q.get()
            if job is None:
                return
            try:
                if self._writer(job) is False:
                    self.errors += 1
                else:
                    self.written += 1
            except Exception as exc:                  # a bad image must not stop the thread
                self.errors += 1
                log.warning("shot %s failed: %r", job.get("path"), exc)

    def _put(self, job: dict) -> bool:
        try:
            self._q.put_nowait(job)
            return True
        except queue.Full:
            self.shots_dropped += 1
            return False

    # -- geometry -------------------------------------------------------------------------------------------------
    def _virtual_executor(self, width: int, height: int) -> MirroringExecutor | None:
        """A never-live executor on a window equal to the frame ([0, 0, w, h] points): its clicks are image pixels."""
        if self.calib is None:
            return None
        if self._virtual is None or self._virtual_wh != (width, height):
            try:
                self._virtual = MirroringExecutor.from_calib(self.calib, [0.0, 0.0, float(width), float(height)],
                                                             live=False)
            except (OSError, ValueError, KeyError, ImportError) as exc:
                log.warning("shots: no calibration for tap points (%r)", exc)
                self.calib = None
                return None
            self._virtual_wh = (width, height)
        return self._virtual

    # -- the loop's hooks -----------------------------------------------------------------------------------------
    def on_decision(self, frame, decision, state: dict | None, executor=None, gs=None) -> None:
        """A decision was made on `frame`: queue its `a` image and, for a tap, wait for the frames that follow."""
        if self._closed:
            return
        if self.decisions >= self.max_decisions:
            self.capped = True
            return
        self.decisions += 1
        entry = decision.executor or {}
        key = decision.key
        status = entry.get("status", "dry_run")
        rd = readings(gs, state)
        height, width = frame.shape[:2]
        hand = rd["hand"] or []
        slot = entry.get("slot")
        clicks_screen = [[round(float(x), 2), round(float(y), 2)] for x, y in entry.get("clicks") or []]
        clicks_image: list[list[float]] = []
        smap = getattr(executor, "screen_map", None)
        tap = key != HOLD_KEY and status in TAP_STATUSES
        if clicks_screen and smap is not None:
            clicks_image = [[round(v, 1) for v in screen_to_image(x, y, smap, width, height)] for x, y in clicks_screen]
        elif not clicks_screen and tap and decision.tile is not None:
            vex = self._virtual_executor(width, height)
            if vex is not None:
                plan = vex.plan_clicks(key, list(decision.tile), hand or None)
                clicks_image = [[round(float(v), 1) for v in c] for c in plan["clicks"]]
                slot = plan["slot"] if slot is None else slot
        if smap is None and self.calib is not None:
            vex = self._virtual_executor(width, height)
            smap = vex.screen_map if vex is not None else None
        rois = roi_boxes(smap, width, height) if smap is not None else None

        name = f"d{decision.idx:04d}_t{decision.t:.1f}"
        files = [f"{name}_a.jpg", None, None]     # b / c are set when their frames arrive and are queued
        tile = [round(float(v), 2) for v in decision.tile] if decision.tile is not None else None
        spec = {"hand": hand, "rois": rois,
                "header": f"d{decision.idx:04d} t={decision.t:.1f}  elixir {_fmt(rd['elixir'])}",
                "header2": f"{key}  {decision.mode}  {status}" + (f"  tile {tile[0]:g},{tile[1]:g}" if tile else ""),
                "tap": {"slot": slot, "slot_pt": clicks_image[0] if len(clicks_image) > 0 else None,
                        "field_pt": clicks_image[1] if len(clicks_image) > 1 else None, "tile": tile}}
        if not self._put({"kind": "a", "path": self.dir / files[0], "frame": frame, "spec": spec}):
            files[0] = None
        row = {"idx": decision.idx, "t": round(float(decision.t), 3), "key": key, "mode": decision.mode,
               "reason": decision.reason, "status": status, "sent": bool(entry.get("sent")), "slot": slot,
               "clicks_screen": clicks_screen, "clicks_image": clicks_image, "tile": tile,
               "elixir_before": rd["elixir"], "elixir_state_before": rd["elixir_state"], "hand_before": rd["hand"],
               "hand_raw_before": rd["hand_raw"], "hand_conf": rd["hand_conf"],
               "files": files, "elixir_after_0.5": None, "elixir_after_1.5": None, "hand_after_1.5": None,
               "hand_raw_after_0.5": None, "hand_raw_after_1.5": None}
        if tap:
            self.taps += 1
            self._pending.append(_Pending(row, float(decision.t), list(zip(AFTER_S, TAGS[1:]))))
        else:
            self._finish(row)

    def on_frame(self, frame, t: float, idx: int, state: dict | None = None, gs=None) -> None:
        """A frame arrived (any frame, decision or not): give it to the taps waiting for their b / c image."""
        if not self._pending or self._closed:
            return
        rd = readings(gs, state)
        for p in self._pending[:]:
            delay, tag = p.todo[0]
            dt = t - p.t
            if dt + 1e-6 < delay:
                continue
            p.todo.pop(0)
            row = p.row
            row[f"elixir_after_{delay}"] = rd["elixir"]
            row[f"hand_raw_after_{delay}"] = rd["hand_raw"]
            if delay == AFTER_S[-1]:
                row["hand_after_1.5"] = rd["hand"]
            name = f"d{row['idx']:04d}_t{p.t:.1f}_{tag}.jpg"
            spec = {"header": f"+{delay:g}s  dt {dt:.1f}s  elixir {_fmt(rd['elixir'])}"}
            if self._put({"kind": tag, "path": self.dir / name, "frame": frame, "spec": spec}):
                row["files"][TAGS.index(tag)] = name
            if not p.todo:
                self._pending.remove(p)
                self._finish(row)

    # -- index ----------------------------------------------------------------------------------------------------
    def _finish(self, row: dict) -> None:
        self._index.write(json.dumps(row) + "\n")
        self._index.flush()

    def summary(self) -> dict:
        return {"decisions": self.decisions, "taps": self.taps, "images_written": self.written,
                "shots_dropped": self.shots_dropped, "errors": self.errors, "capped": self.capped,
                "pending": len(self._pending)}

    def close(self, timeout: float = 5.0) -> None:
        """Write the rows still waiting for a frame (their missing images stay null), let the thread finish what is queued."""
        if self._closed:
            return
        self._closed = True
        for p in self._pending:
            self._finish(p.row)
        self._pending = []
        try:
            self._q.put(None, timeout=timeout)
            self._thread.join(timeout)
        except queue.Full:
            log.warning("shots: writer thread did not drain in %.0f s", timeout)
        self._index.close()


def _fmt(v) -> str:
    return "?" if v is None else f"{v:.1f}"

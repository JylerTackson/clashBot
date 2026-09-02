"""Perception orchestrator: frames in, GameState / PlayEvent out. Read-only.

    p = Perception(config="calib.json", source=VideoFrameSource("match.mp4"))
    for state in p.states(): ...
    for ev in p.events(): ...

Rates: HUD readers run every frame; the unit detector and OCR run at a
reduced rate (detect_every / ocr_every frames) so the loop keeps its rate.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np

from . import geometry as g
from .config import Calibration
from .decktracker import OpponentDeckTracker, load_kb_decks
from .detect import to_card_slug
from .elixir_sim import ElixirSimulator
from .events import PlayDetector
from .hud import (CardMatcher, DigitReader, DigitTemplates, ElixirBarReader, HandReader, OcrReader, TowerHpReader,
                  crop_roi, parse_clock)
from .screen import ContentRect, assess, detect_content_rect
from .sources import FrameSource
from .state import GameState, PlayEvent, UnitObs

REPO = Path(__file__).resolve().parents[1]
KB = REPO / "knowledge_base"

CONF_THRESH = {"elixir": 0.5, "hand": 0.45, "clock": 0.6, "units": 0.0}
PHASE_BY_CLOCK = [(120, "single_elixir"), (60, "double_elixir"), (0, "triple_elixir")]  # remaining seconds thresholds


def load_card_costs(kb: Path = KB) -> dict[str, float]:
    import json
    idx = json.loads((kb / "meta" / "card_index.json").read_text())
    out = {}
    for c in idx["cards"]:
        try:
            out[c["slug"]] = float(c["elixir_cost"])
        except (TypeError, ValueError):
            pass
    return out


@dataclass
class Perception:
    config: str | Path
    source: FrameSource
    detector: object | None = None            # KataCRDetector / BuildABotDetector
    detect_every: int = 3
    ocr_every: int = 15
    use_ocr: bool = True
    kb: Path = KB
    calib: Calibration = field(init=False)
    _events: list[PlayEvent] = field(default_factory=list, init=False)

    def __post_init__(self):
        self.calib = Calibration.load(self.config)
        self.costs = load_card_costs(self.kb)
        self.matcher = CardMatcher(self.kb / "cards" / "images")
        self.hand_reader = HandReader(self.matcher, self.calib.rois)
        self.elixir_bar = ElixirBarReader(self.calib.rois["elixir_bar"])
        self.digits = DigitTemplates()
        self.elixir_digit = DigitReader(self.calib.rois["elixir_num"], self.digits)
        self.clock_digit = DigitReader(self.calib.rois["clock"], self.digits)
        self.ocr = OcrReader() if self.use_ocr else None
        self.tower_reader = TowerHpReader(self.calib.rois, self.ocr)
        self.play = PlayDetector(self.costs)
        self.own_sim = ElixirSimulator()
        self.opp_sim = ElixirSimulator()
        self.deck = OpponentDeckTracker(load_kb_decks(self.kb))
        self.towers = g.TowerState()
        self.last_good: dict = {}
        self.last_good_t: dict = {}
        self.match_started = False
        self.frame_i = 0
        self.H = None
        self.timing: dict[str, list[float]] = {}
        self.match_t0: float | None = None

    # ------------------------------------------------------------------
    def _tick(self, name: str, t0: float) -> None:
        self.timing.setdefault(name, []).append((time.perf_counter() - t0) * 1000)

    def _phase_from_clock(self, remaining: int | None, overtime: bool) -> str | None:
        if remaining is None:
            return None
        if overtime:
            return "double_elixir_overtime" if remaining > 60 else "triple_elixir_overtime"
        for thr, name in PHASE_BY_CLOCK:
            if remaining > thr:
                return name
        return "triple_elixir"

    def _commit(self, field_name: str, value, conf: float, t: float) -> tuple[object, float, bool]:
        """Apply the confidence rule: below threshold keep the previous value,
        report it stale."""
        if value is not None and conf >= CONF_THRESH.get(field_name, 0.5):
            self.last_good[field_name] = value
            self.last_good_t[field_name] = t
            return value, conf, False
        return self.last_good.get(field_name), conf, True

    def reset_match(self, t: float) -> None:
        self.play = PlayDetector(self.costs)
        self.own_sim.reset(t)
        self.opp_sim.reset(t)
        self.deck.reset()
        self.towers = g.TowerState()
        self.last_good.clear()
        self.match_t0 = t

    # ------------------------------------------------------------------
    def process(self, frame: np.ndarray, t: float, idx: int) -> GameState:
        self.frame_i = idx
        t_all = time.perf_counter()
        fw, fh = frame.shape[1], frame.shape[0]
        cx, cy, cw, ch = self.calib.content_rect(fw, fh)
        content = frame[cy:cy + ch, cx:cx + cw]
        if self.H is None or getattr(self, "_hsize", None) != (cw, ch):
            self.H = self.calib.homography_for(cw, ch)
            self._hsize = (cw, ch)
        ready = assess(frame, ContentRect(cx, cy, cw, ch), {"elixir_bar": tuple(self.calib.rois["elixir_bar"]),
                                                            "arena": tuple(self.calib.rois["arena"])})
        state = GameState(t=t, frame_index=idx, readiness=ready.state)
        state.field_confidence["readiness"] = ready.conf
        if ready.state != "match":
            self.match_started = False
            self._tick("total", t_all)
            return state
        if not self.match_started:
            self.match_started = True
            self.reset_match(t)

        # ---- HUD: elixir (bar + digit cross-check) ----
        t0 = time.perf_counter()
        eb, eb_conf = self.elixir_bar.read(content)
        ed_s, ed_conf = self.elixir_digit.read_string(content)
        ed = int(ed_s) if ed_s.isdigit() and 0 <= int(ed_s) <= 10 else None
        if ed is not None and eb is not None and ed == eb:
            elixir, econf = eb, min(1.0, max(eb_conf, ed_conf) + 0.2)
        elif ed is not None and ed_conf >= 0.8:
            elixir, econf = ed, ed_conf * 0.9
        else:
            elixir, econf = eb, eb_conf
        elixir, econf, stale_e = self._commit("elixir", elixir, econf, t)
        self._tick("elixir", t0)

        # ---- HUD: hand ----
        t0 = time.perf_counter()
        hand = self.hand_reader.read(content)
        self._tick("hand", t0)

        # ---- clock / phase (template digits every frame, OCR fallback at low rate) ----
        t0 = time.perf_counter()
        clock_s, clock_conf = self.clock_digit.read_string(content)
        remaining = parse_clock(clock_s)
        if remaining is None and self.ocr is not None and idx % self.ocr_every == 0:
            s, c = self.ocr.read(crop_roi(content, self.calib.rois["clock"]))
            if parse_clock(s) is not None:
                clock_s, clock_conf, remaining = s, c, parse_clock(s)
        clock_v, clock_conf, stale_c = self._commit("clock", clock_s if remaining is not None else None, clock_conf, t)
        remaining = parse_clock(clock_v) if clock_v else None
        overtime = bool(self.last_good.get("overtime", False))
        phase = self._phase_from_clock(remaining, overtime)
        self._tick("clock", t0)

        # ---- towers (OCR, low rate) ----
        towers_own = {"king": None, "left": None, "right": None}
        towers_enemy = {"king": None, "left": None, "right": None}
        if self.ocr is not None and idx % self.ocr_every == 0:
            t0 = time.perf_counter()
            hp = self.tower_reader.read(content)
            for k in ("king", "left", "right"):
                v, c = hp[f"own_{k}"]
                if v is not None and c >= 0.5:
                    self.last_good[f"own_{k}"] = v
                v, c = hp[f"enemy_{k}"]
                if v is not None and c >= 0.5:
                    self.last_good[f"enemy_{k}"] = v
            self._tick("towers", t0)
        for k in ("king", "left", "right"):
            towers_own[k] = self.last_good.get(f"own_{k}")
            towers_enemy[k] = self.last_good.get(f"enemy_{k}")
        self.towers.enemy_left = towers_enemy["left"] != 0
        self.towers.enemy_right = towers_enemy["right"] != 0
        self.towers.own_left = towers_own["left"] != 0
        self.towers.own_right = towers_own["right"] != 0

        # ---- units ----
        units: list[UnitObs] = []
        units_conf = None
        if self.detector is not None and self.H is not None and idx % self.detect_every == 0:
            t0 = time.perf_counter()
            dets = self.detector.detect(content, self.calib.arena_crop(cw, ch))
            for d in dets:
                tile = None
                if d.category in ("unit", "spell") and self.H.in_arena(*g.bbox_bottom_centre(d.bbox), margin=1.0):
                    tile = g.clamp_tile(*g.bbox_to_tile(self.H, d.bbox))
                units.append(UnitObs(d.cls, d.side, tile, d.conf, tuple(int(v) for v in d.bbox), None, d.category))
            units_conf = round(float(np.mean([u.conf for u in units])), 3) if units else 1.0
            self.last_good["units"] = units
            self.last_good_t["units"] = t
            self._tick("detect", t0)
        else:
            units = self.last_good.get("units", [])

        # ---- events ----
        phase_key = "single" if not phase else "double" if phase.startswith("double") else "triple" if phase.startswith("triple") else "single"
        self.own_sim.advance(t, phase_key)
        self.opp_sim.advance(t, phase_key)
        new_events: list[PlayEvent] = []
        self.play.update_hand(t, hand["hand"], hand["hand_conf"])
        if not stale_e:
            new_events += self.play.update_elixir(t, elixir, clock_v)
        if idx % self.detect_every == 0 and self.detector is not None:
            new_events += self.play.update_units(t, units, clock_v)
        new_events += self.play.update_tower_hp(t, {"own_king": towers_own["king"], "own_left": towers_own["left"],
                                                    "own_right": towers_own["right"]}, units, clock_v)
        for ev in new_events:
            if ev.player == "own" and ev.card and ev.card in self.costs:
                self.own_sim.spend(self.costs[ev.card])
            if ev.player == "opponent":
                if ev.card and ev.card in self.costs:
                    self.deck.check_cycle(ev.card)
                    self.opp_sim.spend(self.costs[ev.card])
                    ev.elixir_before = round(self.opp_sim.elixir + self.costs[ev.card], 2)
                    ev.elixir_after = round(self.opp_sim.elixir, 2)
                self.deck.observe_play(ev.card)
        self._events.extend(new_events)
        drift = self.own_sim.observe(t, elixir if not stale_e else None)
        # resync own sim when the HUD is confident, so the drift stays a measurement not a runaway
        if drift is not None and abs(drift) > 2.5:
            self.own_sim.resync(elixir)
        d_stats = self.own_sim.drift_stats(window=300)
        est, econf_opp, (lo, hi) = self.opp_sim.estimate(d_stats["abs_mean"])

        # ---- assemble ----
        mask = g.legal_placement_mask(self.towers)
        state.match_clock = clock_v
        state.match_seconds = (180 - remaining) if remaining is not None and not overtime else None
        state.phase = phase
        state.own = {"elixir": elixir, "elixir_sim": round(self.own_sim.elixir, 2), "hand": hand["hand"],
                     "hand_conf": hand["hand_conf"], "next_card": hand["next_card"], "next_conf": hand["next_conf"],
                     "towers": towers_own}
        state.opponent = {"elixir_est": est, "elixir_conf": econf_opp, "elixir_range": [lo, hi],
                          "towers": towers_enemy, **self.deck.summary()}
        state.units = units
        state.legal_mask_hash = g.mask_hash(mask)
        state.field_confidence = {"readiness": ready.conf, "elixir": round(econf, 2),
                                  "hand": round(float(np.mean([c for c in hand["hand_conf"]])), 2),
                                  "clock": round(clock_conf, 2), "units": units_conf,
                                  "own_elixir_drift": d_stats}
        state.stale = {"elixir": round(t - self.last_good_t.get("elixir", t), 2) if stale_e else 0,
                       "clock": round(t - self.last_good_t.get("clock", t), 2) if stale_c else 0,
                       "units": round(t - self.last_good_t.get("units", t), 2) if idx % self.detect_every else 0}
        self._mask = mask
        self._tick("total", t_all)
        return state

    # ------------------------------------------------------------------
    def states(self) -> Iterator[GameState]:
        for frame, t, idx in self.source.frames():
            yield self.process(frame, t, idx)

    def events(self) -> Iterator[PlayEvent]:
        for _ in self.states():
            while self._events:
                yield self._events.pop(0)

    def drain_events(self) -> list[PlayEvent]:
        ev, self._events = self._events, []
        return ev

    def timing_report(self) -> dict:
        out = {}
        for k, v in self.timing.items():
            if v:
                out[k] = {"n": len(v), "mean_ms": round(float(np.mean(v)), 1), "p95_ms": round(float(np.percentile(v, 95)), 1)}
        if "total" in out:
            out["loop_fps"] = round(1000.0 / out["total"]["mean_ms"], 1)
        return out

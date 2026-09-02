"""Perception orchestrator: frames in, GameState / PlayEvent out. Read-only.

    p = Perception(config="calib.json", source=VideoFrameSource("match.mp4"))
    for state in p.states(): ...
    for ev in p.events(): ...

Rates: HUD readers run every frame; the unit detector and OCR run at a
reduced rate (detect_every / ocr_every frames) so the loop keeps its rate.
"""
from __future__ import annotations

import re

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
import cv2

from .hud import (CardMatcher, DigitReader, DigitTemplates, ElixirBarReader, HandReader, OcrReader, TowerHpReader,
                  crop_roi, parse_clock)
from .labels import DeployLabelReader, label_ground_point
from .tracking import UnitTracker, load_speed_priors
from .clock import ClockTracker
from .screen import ContentRect, MatchGate, assess, detect_content_rect
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


# in-game cards without a Phase 1 page (wiki stubs); slug -> display name
EXTRA_CARD_NAMES = {"minion-giant": "Minion Giant"}


@dataclass
class Perception:
    config: str | Path
    source: FrameSource
    detector: object | None = None            # KataCRDetector / BuildABotDetector
    ENTER_FRAMES: int = 3
    EXIT_SECONDS: float = 2.0
    label_every: int = 15                     # frames between deploy-label OCR passes
    label_scale: float = 0.7
    tower_ocr: bool = False                   # per-tower OCR is slow; the arena label pass reads HP numbers
    detect_every: int = 3
    ocr_every: int = 10
    use_ocr: bool = True
    kb: Path = KB
    calib: Calibration = field(init=False)
    _events: list[PlayEvent] = field(default_factory=list, init=False)

    def __post_init__(self):
        self.calib = Calibration.load(self.config)
        self.costs = load_card_costs(self.kb)
        # wiki art only: BuildABot's in-game thumbnails were measured to lower
        # hand accuracy on real frames (style mismatch shrinks the margin)
        self.learned_dir = REPO / "data" / "templates"
        self.matcher = CardMatcher(self.kb / "cards" / "images", learned_dir=self.learned_dir,
                                   hero_images_dir=self.kb / "heroes" / "images", costs=self.costs)
        self._slot_crops: list[tuple[float, list]] = []   # (t, [4 slot crops]) ring buffer, ~3 s
        self.digits = DigitTemplates()
        self.hand_reader = HandReader(self.matcher, self.calib.rois, digits=self.digits)
        self.elixir_bar = ElixirBarReader(self.calib.rois["elixir_bar"], self.calib.rois.get("elixir_bar_full"))
        self.elixir_digit = DigitReader(self.calib.rois["elixir_num"], self.digits)
        self.clock_digit = DigitReader(self.calib.rois["clock"], self.digits)
        self.ocr = OcrReader() if self.use_ocr else None
        self.tower_reader = TowerHpReader(self.calib.rois, self.ocr)
        import json as _json
        _idx = _json.loads((self.kb / "meta" / "card_index.json").read_text())
        self.card_names = {c["slug"]: c["name"] for c in _idx["cards"]}
        # cards that exist in the game but have no knowledge-base page yet: the
        # deploy label still reads them, so give the label matcher their names
        # (otherwise "Minion Giant" fuzzy-matches "Giant")
        label_names = {**self.card_names, **EXTRA_CARD_NAMES}
        self.labels = DeployLabelReader(label_names, self.ocr.ocr, scale=self.label_scale) if self.ocr else None
        self.play = PlayDetector(self.costs)
        self.own_sim = ElixirSimulator()
        self.opp_sim = ElixirSimulator()
        self.deck = OpponentDeckTracker(load_kb_decks(self.kb))
        cats = {c["slug"]: ("building" if c["card_type"] == "Building" else "spell" if c["card_type"] == "Spell" else "unit")
                for c in _idx["cards"]}
        self.tracker = UnitTracker(speed_priors=load_speed_priors(self.kb), categories=cats)
        self.towers = g.TowerState()
        self.last_good: dict = {}
        self.last_good_t: dict = {}
        self.match_started = False
        self.match_id = 0
        self.clock = ClockTracker()
        self.gate = MatchGate(self.ENTER_FRAMES, self.EXIT_SECONDS)
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

    def _smooth_hand(self, hand: dict, window: int = 5) -> dict:
        """Majority vote per slot over the last `window` reads; a slot's
        value only changes when the new card wins the vote, which removes
        single-frame flicker (e.g. a raised/greyed card)."""
        hist = self.__dict__.setdefault("_hand_hist", [])
        hist.append((list(hand["hand"]), list(hand["hand_conf"])))
        del hist[:-window]
        out = dict(hand)
        sm, smc = [], []
        for i in range(4):
            votes = {}
            for h, c in hist:
                if h[i] is not None:
                    votes[h[i]] = votes.get(h[i], 0) + c[i]
            if votes:
                best = max(votes, key=votes.get)
                sm.append(best)
                smc.append(round(votes[best] / len(hist), 3))
            else:
                sm.append(None)
                smc.append(0.0)
        out["hand"], out["hand_conf"] = sm, smc
        return out

    def reset_match(self, t: float) -> None:
        self.match_id += 1
        self.clock = ClockTracker()
        self.play = PlayDetector(self.costs)
        self.own_sim.reset(t)
        self.opp_sim.reset(t)
        self.deck.reset()
        self.tracker.reset()
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
        readiness, entered = self.gate.update(ready.state, t)
        if entered:
            self.reset_match(t)
        self.match_started = self.gate.in_match
        state = GameState(t=t, frame_index=idx, readiness=readiness)
        state.match_id = self.match_id
        state.field_confidence["readiness"] = ready.conf
        state.field_confidence["readiness_raw"] = ready.state
        if not self.match_started:
            self._tick("total", t_all)
            return state

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
        if idx % 3 == 0:
            # enlarged boxes so a raised (selected) card is still fully inside
            self._slot_crops.append((t, [crop_roi(content, self._wide_slot(i)).copy() for i in range(4)]))
            self._slot_crops = [c for c in self._slot_crops if t - c[0] <= 9.0]
        hand = self._smooth_hand(hand)
        self._tick("hand", t0)

        # ---- clock / phase (template digits every frame, OCR fallback at low rate) ----
        t0 = time.perf_counter()
        clock_s, clock_conf, remaining = "", 0.0, None
        if self.ocr is not None and idx % self.ocr_every == 0:
            s, c = self.ocr.read(crop_roi(content, self.calib.rois["clock"]))
            if parse_clock(s) is None:
                # other HUD layouts print "OVERTIME 1:59" / "Time left 2:31" in one box
                # or place the timer higher: retry on a taller top-right region
                s_alt, c_alt = self.ocr.read(crop_roi(content, [0.72, 0.0, 0.28, 0.12]))
                m_alt = re.search(r"\d{1,2}:\d\d", s_alt or "")
                if m_alt:
                    s, c = m_alt.group(0), c_alt
            if parse_clock(s) is not None and c >= 0.6 and self.clock.valid(parse_clock(s)):
                clock_s, clock_conf, remaining = s, c, parse_clock(s)
        if remaining is None:
            s2, c2 = self.clock_digit.read_string(content)
            prev = parse_clock(self.last_good.get("clock") or "")
            # template digits are trusted only when they continue the previous read
            if parse_clock(s2) is not None and prev is not None and 0 <= prev - parse_clock(s2) <= 3:
                clock_s, clock_conf, remaining = s2, min(c2, 0.7), parse_clock(s2)
        clock_v, clock_conf, stale_c = self._commit("clock", clock_s if remaining is not None else None, clock_conf, t)
        remaining = parse_clock(clock_v) if clock_v else None
        # clock tracker: confirmed jump to a fresh 3:00 = new game (reset
        # per-match state); confirmed jump to ~2:00 from 0:00 = overtime
        if remaining is not None and not stale_c:
            ev = self.clock.update(t, remaining)
            if ev == "new_match":
                self.reset_match(t)
                self.clock.update(t, remaining)
        overtime = self.clock.overtime
        phase = self.clock.phase() if self.clock.remaining is not None else None
        self._tick("clock", t0)

        # ---- towers (OCR, low rate) ----
        towers_own = {"king": None, "left": None, "right": None}
        towers_enemy = {"king": None, "left": None, "right": None}
        if self.tower_ocr and self.ocr is not None and idx % self.ocr_every == 0:
            t0 = time.perf_counter()
            hp = self.tower_reader.read(content)
            for k in ("king", "left", "right"):
                v, c = hp[f"own_{k}"]
                if v is not None and c >= 0.8:
                    self.last_good[f"own_{k}"] = v
                v, c = hp[f"enemy_{k}"]
                if v is not None and c >= 0.8:
                    self.last_good[f"enemy_{k}"] = v
            self._tick("towers", t0)
        for k in ("king", "left", "right"):
            towers_own[k] = self.last_good.get(f"own_{k}")
            towers_enemy[k] = self.last_good.get(f"enemy_{k}")
        self.towers.enemy_left = towers_enemy["left"] != 0
        self.towers.enemy_right = towers_enemy["right"] != 0
        self.towers.own_left = towers_own["left"] != 0
        self.towers.own_right = towers_own["right"] != 0

        # ---- deploy labels + tower HP numbers (arena OCR, low rate) ----
        label_events_input = ([], [])
        if self.labels is not None and self.H is not None and idx % self.label_every == 0:
            t0 = time.perf_counter()
            labels, numbers = self.labels.read(content, self.calib.arena_crop(cw, ch))
            tiles = []
            for lbl in labels:
                gx, gy = label_ground_point(lbl)
                tiles.append(g.clamp_tile(*self.H.pixel_to_tile(gx, gy)) if self.H.in_arena(gx, gy, margin=1.0) else None)
            label_events_input = (labels, tiles)
            for num in numbers:
                if not 100 <= num.value <= 9999:
                    continue
                c, r = self.H.pixel_to_tile((num.bbox[0] + num.bbox[2]) / 2, num.bbox[3])
                name = None
                if 24 <= r <= 29 and c < 7:
                    name = "enemy_left"
                elif 24 <= r <= 29 and c > 10:
                    name = "enemy_right"
                elif r >= 27 and 6 <= c <= 11:
                    name = "enemy_king"
                elif 2 <= r <= 8 and c < 7:
                    name = "own_left"
                elif 2 <= r <= 8 and c > 10:
                    name = "own_right"
                elif r <= 3 and 6 <= c <= 11:
                    name = "own_king"
                if name:
                    self.last_good[name] = num.value
                    self.last_good_t[name] = t
            for k in ("king", "left", "right"):
                towers_own[k] = self.last_good.get(f"own_{k}")
                towers_enemy[k] = self.last_good.get(f"enemy_{k}")
            self._tick("labels", t0)

        # ---- units: detector at a low rate, Kalman tracks every frame ----
        units: list[UnitObs] = []
        units_conf = None
        if self.detector is not None and self.H is not None and idx % self.detect_every == 0:
            t0 = time.perf_counter()
            dets = self.detector.detect(content, self.calib.arena_crop(cw, ch))
            towers_seen: list[UnitObs] = []
            tracked_in = []
            for d in dets:
                bx, by = g.bbox_bottom_centre(d.bbox)
                if d.category in ("unit", "spell") and self.H.in_arena(bx, by, margin=0.5):
                    d.pos_f = self.H.pixel_to_tile_f(bx, by)
                    if -0.5 <= d.pos_f[1] <= 31.2:
                        tracked_in.append(d)
                elif d.category == "tower":
                    towers_seen.append(UnitObs(d.cls, d.side, None, d.conf, tuple(int(v) for v in d.bbox), None, d.category))
            self.tracker.update(t, tracked_in)
            self.last_good["towers_seen"] = towers_seen
            self.last_good_t["units"] = t
            units_conf = round(float(np.mean([d.conf for d in tracked_in])), 3) if tracked_in else 1.0
            self._tick("detect", t0)
        elif self.H is not None:
            self.tracker.predict(t)
        for tr in self.tracker.confirmed():
            m = tr.summary()
            units.append(UnitObs(tr.cls, tr.side, tuple(m["tile"]), tr.conf, tuple(int(v) for v in (tr.bbox or (0, 0, 0, 0))),
                                 tr.id, tr.category, m))
        units += self.last_good.get("towers_seen", [])

        # ---- events ----
        phase_key = self.clock.regen_key()
        self.own_sim.advance(t, phase_key)
        self.opp_sim.advance(t, phase_key)
        new_events: list[PlayEvent] = []
        self.play.update_hand(t, hand["hand"], hand["hand_conf"])
        if not stale_e:
            for ev in self.play.update_elixir(t, elixir, clock_v):
                if ev.player == "own" and ev.card:
                    if ev.card in self.costs:
                        self.own_sim.spend(self.costs[ev.card])
                        ev._charged = True
                    self.play.hold_own(ev)       # wait briefly for a deploy label to attach the tile
                else:
                    new_events.append(ev)
        if label_events_input[0]:
            new_events += self.play.update_labels(t, label_events_input[0], label_events_input[1], clock_v)
        new_events += self.play.release_holds(t)
        if idx % self.detect_every == 0 and self.detector is not None:
            new_events += self.play.update_units(t, units, clock_v)
        new_events += self.play.update_tower_hp(t, {"own_king": towers_own["king"], "own_left": towers_own["left"],
                                                    "own_right": towers_own["right"]}, units, clock_v)
        for ev in new_events:
            if (ev.player == "own" and ev.card and "deploy label" in ev.detail
                    and (ev.confidence == "high" or ev.detect_source == "deploy_label")):
                self._learn_template(ev)
            if ev.player == "own" and ev.card and ev.card in self.costs and not getattr(ev, "_charged", False):
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
                       "units": round(t - self.last_good_t.get("units", t), 2)}
        self._mask = mask
        self._tick("total", t_all)
        return state

    def _wide_slot(self, i: int) -> list:
        x, y, w, h = self.calib.rois[f"hand_{i}"]
        return [x - 0.01, y - 0.035, w * 1.14, h * 1.14 + 0.035]

    def _slot_from_change(self, t: float) -> int | None:
        """Which hand slot changed around t (a card left it)? The slot with the
        largest before/after difference, provided it held a coloured card."""
        before = [c for c in self._slot_crops if 0.6 <= t - c[0] <= 1.8]
        after = [c for c in self._slot_crops if 0.8 <= c[0] - t <= 2.5]
        if not before or not after:
            return None
        def th(img):
            return cv2.cvtColor(cv2.resize(img, (24, 28), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32)
        best, best_d = None, 0.0
        for i in range(4):
            b, a = before[-1][1][i], after[0][1][i]
            if b.size == 0 or a.size == 0 or th(b).std() < 15 or self.matcher.is_greyed(b):
                continue
            d = float(np.abs(th(b) - th(a)).mean())
            if d > best_d:
                best, best_d = i, d
        return best if best_d >= 18.0 else None

    def _learn_template(self, ev) -> None:
        """The slot crop from ~0.5-1.5 s before the play is the in-game art of
        the confirmed card: store it as a template (self-labelling). For a
        play known only from its deploy label the slot is inferred from the
        hand change."""
        slot = getattr(ev, "slot", None)
        if slot is None:
            slot = self._slot_from_change(ev.timestamp)
            if slot is None:
                return
        cands = [c for c in self._slot_crops if 0.6 <= ev.timestamp - c[0] <= 1.8]
        if not cands:
            return
        crop = cands[-1][1][slot]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.size else None
        if gray is None or gray.std() < 15:
            return
        existing = list((self.learned_dir / ev.card).glob("*.png")) if (self.learned_dir / ev.card).exists() else []
        if len(existing) >= 12:
            return
        vid = Path(getattr(self.source, "path", "live")).stem
        self.matcher.add_learned(ev.card, crop, self.learned_dir / ev.card / f"{vid}_{ev.timestamp:.1f}.png")

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

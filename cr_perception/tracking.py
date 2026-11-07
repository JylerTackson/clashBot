"""Unit tracking with per-track Kalman filters in tile space.

Why: the unit detector runs at ~1 Hz while the HUD runs at 10-30 Hz. A
constant-velocity Kalman filter per unit gives a position estimate at every
frame, a velocity (speed + heading), an uncertainty radius, and short-horizon
predictions (position in 1-2 s, time to the river / a tower). The agent then
gets "where is it going" rather than "where was it last seen".

Domain priors (Clash Royale):
  * movement speeds are fixed per card: Slow 45, Medium 60, Fast 90,
    Very Fast 120 (game units ~ tiles per minute, so Medium = 1.0 tile/s);
    the Phase 1 attribute tables carry them, so a new track is initialised
    with that speed pointing toward the enemy side (allies move up the
    screen, enemies down) instead of zero velocity;
  * buildings and spells do not move (static model, tiny process noise);
  * units follow lanes toward towers, so heading is reported relative to
    the field (toward enemy left/right tower, king tower, or retreating);
  * a unit attacks a tower from its attack range, so the tower ETA (MAC-AL T1)
    is the time to get within range of the tower edge, at the card's speed
    (the filter lags behind a unit that walks at a fixed speed).

State x = [col, row, v_col, v_row] in tiles and tiles/s; measurement z =
[col, row] of the bounding-box bottom-centre mapped through the homography.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import geometry as g

SPEED_TILES_PER_S = {"very slow": 30 / 60, "slow": 45 / 60, "medium": 60 / 60, "fast": 90 / 60, "very fast": 120 / 60}
DEFAULT_SPEED = 1.0          # tiles/s when the card's speed is unknown
STATIC_TYPES = {"building", "tower", "spell"}

TOWER_TILES = {"enemy_left": (3.5, 23.3), "enemy_right": (14.5, 23.3), "enemy_king": (9.0, 26.3),
               "own_left": (3.5, 4.3), "own_right": (14.5, 4.3), "own_king": (9.0, 0.5)}
TOWER_EDGE = 1.5             # tiles from a tower's centre to its edge (a melee unit stops here)
TOWARD_COS = 0.5             # moving toward a tower: cosine of the velocity to the tower direction at least this
PRIOR_SPEED_FRAC = 0.9       # a track walking toward the tower is at least this share of its card's speed
UNIT_STATS_PATH = Path(__file__).resolve().parents[1] / "clashbot" / "resources" / "unit_stats.json"


def load_speed_priors(kb: Path) -> dict[str, float]:
    """slug -> tiles/s from the Phase 1 card attribute tables ('Medium (60)')."""
    out: dict[str, float] = {}
    for p in (kb / "cards").glob("*.md"):
        text = p.read_text()
        m = re.search(r"\|\s*Speed\s*\|", text)
        if not m:
            continue
        # find the header row containing 'Speed' and the first data row
        for block in re.findall(r"(\|[^\n]*\bSpeed\b[^\n]*\|)\n\|[-| ]+\|\n(\|[^\n]*\|)", text):
            head = [h.strip().lower() for h in block[0].strip("|").split("|")]
            row = [c.strip() for c in block[1].strip("|").split("|")]
            if "speed" in head and len(row) == len(head):
                val = row[head.index("speed")]
                mm = re.search(r"\((\d+)\)", val)
                if mm:
                    out[p.stem] = int(mm.group(1)) / 60.0
                else:
                    key = val.lower().strip()
                    if key in SPEED_TILES_PER_S:
                        out[p.stem] = SPEED_TILES_PER_S[key]
                break
    return out


def load_range_priors(path: Path | None = None) -> dict[str, float]:
    """slug -> attack range in tiles of the card's first unit (`units[0].range_tiles`, 0.0 when missing), from
    `clashbot/resources/unit_stats.json`. Keyed by card slug like `load_speed_priors`, so a track finds both with
    `to_card_slug(cls)`. A missing or unreadable file gives {} (no range: the old tower-edge rule)."""
    try:
        cards = json.loads(Path(path or UNIT_STATS_PATH).read_text(encoding="utf-8")).get("cards") or {}
    except (OSError, ValueError, AttributeError):
        return {}
    out: dict[str, float] = {}
    for slug, card in cards.items():
        units = (card or {}).get("units") or []
        r = units[0].get("range_tiles") if units and isinstance(units[0], dict) else None
        out[slug] = float(r) if isinstance(r, (int, float)) and not isinstance(r, bool) else 0.0
    return out


def eta_to_tower(pos, vel, side: str, speed_prior: float | None = None,
                 range_prior: float = 0.0) -> tuple[str, float] | None:
    """(tower name, seconds) until a unit at `pos` (tiles) moving with `vel` (tiles/s) is within attack range of the
    closest tower on the opposing side (allies go for the enemy's towers, enemies for ours). Pure; no priors gives the
    old rule (straight-line distance to the tower edge at the filter's speed).
      * d = distance to the tower centre - TOWER_EDGE - range, floored at 0; d == 0 is ETA 0.0 whatever the speed
        (a unit in reach of a tower is at it, even when it reads stationary);
      * the filter lags behind a unit that walks at its card's speed: when the velocity points at the tower
        (cosine >= TOWARD_COS) and the card's speed is known, the speed is max(filter speed, 0.9 x card speed);
      * filter speed < 0.1 and d > 0 is None (not going anywhere)."""
    names = ("enemy_left", "enemy_right", "enemy_king") if side == "ally" else ("own_left", "own_right", "own_king")
    best = None
    for nm in names:
        tc, tr = TOWER_TILES[nm]
        dist = math.hypot(tc - pos[0], tr - pos[1])
        d = max(0.0, dist - TOWER_EDGE - range_prior)
        if best is None or (d, dist) < (best[1], best[2]):
            best = (nm, d, dist, tc, tr)
    nm, d, dist, tc, tr = best
    if d <= 0.0:
        return nm, 0.0
    sp = math.hypot(vel[0], vel[1])
    if sp < 0.1:
        return None
    if speed_prior and dist > 0 and (vel[0] * (tc - pos[0]) + vel[1] * (tr - pos[1])) / (sp * dist) >= TOWARD_COS:
        sp = max(sp, PRIOR_SPEED_FRAC * speed_prior)
    return nm, round(d / sp, 1)


@dataclass
class KalmanTrack:
    id: int
    cls: str
    side: str
    category: str
    x: np.ndarray                 # [c, r, vc, vr]
    P: np.ndarray                 # 4x4 covariance
    t: float
    hits: int = 1
    misses: int = 0
    first_t: float = 0.0
    last_meas_t: float = 0.0
    bbox: tuple[int, int, int, int] | None = None
    conf: float = 0.0
    history: list[tuple[float, float, float]] = field(default_factory=list)   # (t, c, r) measurements
    speed_prior: float | None = None   # the card's tiles/s (None: unknown card)
    range_prior: float = 0.0           # the card's attack range in tiles (0: unknown card)

    # --- model -----------------------------------------------------------
    @property
    def static(self) -> bool:
        return self.category in STATIC_TYPES

    def _F(self, dt: float) -> np.ndarray:
        F = np.eye(4)
        if not self.static:
            F[0, 2] = F[1, 3] = dt
        return F

    def _Q(self, dt: float, q: float) -> np.ndarray:
        if self.static:
            return np.diag([1e-4, 1e-4, 1e-6, 1e-6])
        # white-noise acceleration model
        dt2, dt3, dt4 = dt * dt, dt ** 3, dt ** 4
        Q = np.array([[dt4 / 4, 0, dt3 / 2, 0], [0, dt4 / 4, 0, dt3 / 2], [dt3 / 2, 0, dt2, 0], [0, dt3 / 2, 0, dt2]])
        return q * Q

    def predict_to(self, t: float, q: float = 0.6) -> tuple[np.ndarray, np.ndarray]:
        """State and covariance propagated to time t (does not mutate)."""
        dt = max(0.0, t - self.t)
        F = self._F(dt)
        return F @ self.x, F @ self.P @ F.T + self._Q(dt, q)

    def advance(self, t: float, q: float = 0.6) -> None:
        self.x, self.P = self.predict_to(t, q)
        self.t = t

    def update(self, z: np.ndarray, t: float, r: float = 0.35, q: float = 0.6, bbox=None, conf: float = 0.0) -> float:
        """Kalman update with measurement z=[c, r]; returns the Mahalanobis
        distance of the innovation (for gating diagnostics)."""
        self.advance(t, q)
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        R = np.eye(2) * r * r
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P
        if self.static:
            self.x[2:] = 0.0
        self.hits += 1
        self.misses = 0
        self.last_meas_t = t
        self.bbox, self.conf = bbox, conf
        self.history.append((t, float(z[0]), float(z[1])))
        if len(self.history) > 30:
            self.history.pop(0)
        return float(math.sqrt(y @ np.linalg.inv(S) @ y))

    def mahalanobis(self, z: np.ndarray, t: float, r: float = 0.35) -> float:
        x, P = self.predict_to(t)
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        S = H @ P @ H.T + np.eye(2) * r * r
        y = z - H @ x
        return float(math.sqrt(y @ np.linalg.inv(S) @ y))

    # --- derived quantities ---------------------------------------------
    def position(self) -> tuple[float, float]:
        return float(self.x[0]), float(self.x[1])

    def tile(self) -> tuple[int, int]:
        return g.clamp_tile(int(math.floor(self.x[0])), int(math.floor(self.x[1])))

    def velocity(self) -> tuple[float, float]:
        return float(self.x[2]), float(self.x[3])

    def speed(self) -> float:
        return float(math.hypot(self.x[2], self.x[3]))

    def pos_std(self) -> float:
        return float(math.sqrt(max(self.P[0, 0], 0) + max(self.P[1, 1], 0)) / math.sqrt(2))

    def predict_position(self, dt: float) -> tuple[float, float]:
        x, _ = self.predict_to(self.t + dt)
        return float(x[0]), float(x[1])

    def heading(self) -> str:
        """Field-relative heading label."""
        vc, vr = self.velocity()
        sp = math.hypot(vc, vr)
        if self.static or sp < 0.15:
            return "stationary"
        toward_enemy = vr > 0.15 * sp if self.side == "ally" else vr < -0.15 * sp
        retreat = vr < -0.15 * sp if self.side == "ally" else vr > 0.15 * sp
        lane = "left" if self.x[0] < 9 else "right"
        if abs(vc) > 0.7 * sp:
            return "moving " + ("right" if vc > 0 else "left") + " (crossing lanes)"
        if toward_enemy:
            return f"advancing {lane} lane"
        if retreat:
            return "retreating"
        return "drifting"

    def eta_to_row(self, row: float) -> float | None:
        """Seconds until the track crosses `row` at its current velocity."""
        vr = self.x[3]
        if abs(vr) < 0.05:
            return None
        dt = (row - self.x[1]) / vr
        return round(dt, 1) if dt > 0 else None

    def eta_to_nearest_enemy_tower(self) -> tuple[str, float] | None:
        """(tower name, seconds) until the track is within attack range of the closest tower on the opposing side
        (from this track's point of view); see `eta_to_tower`. Buildings and spells never arrive: None."""
        if self.static:
            return None
        return eta_to_tower((float(self.x[0]), float(self.x[1])), self.velocity(), self.side,
                            self.speed_prior, self.range_prior)

    def summary(self) -> dict:
        c, r = self.position()
        vc, vr = self.velocity()
        p1 = self.predict_position(1.0)
        p2 = self.predict_position(2.0)
        eta = self.eta_to_nearest_enemy_tower()
        return {"track_id": self.id, "class": self.cls, "side": self.side, "category": self.category,
                "pos": [round(c, 2), round(r, 2)], "tile": list(self.tile()),
                "vel": [round(vc, 2), round(vr, 2)], "speed": round(self.speed(), 2), "heading": self.heading(),
                "pos_std": round(self.pos_std(), 2), "age": round(self.t - self.first_t, 1),
                "since_seen": round(self.t - self.last_meas_t, 2), "hits": self.hits,
                "pred_1s": [round(p1[0], 1), round(p1[1], 1)], "pred_2s": [round(p2[0], 1), round(p2[1], 1)],
                "eta_river": self.eta_to_row(15.5 if self.side == "ally" else 15.5),
                "eta_tower": {"tower": eta[0], "s": eta[1], "range": round(self.range_prior, 2)} if eta else None,
                "conf": round(self.conf, 2)}


@dataclass
class UnitTracker:
    speed_priors: dict[str, float] = field(default_factory=dict)
    categories: dict[str, str] = field(default_factory=dict)     # slug -> unit|building|spell
    gate: float = 3.0               # Mahalanobis gate for association
    max_misses_s: float = 2.5       # drop a track unseen for this long
    confirm_hits: int = 2
    meas_noise: float = 0.35        # tiles (bottom-centre jitter)
    process_noise: float = 0.6      # tiles/s^2 (turns, kiting, knockback)
    tracks: dict[int, KalmanTrack] = field(default_factory=dict)
    next_id: int = 1
    t: float = 0.0
    range_priors: dict[str, float] = field(default_factory=dict)   # slug -> attack range in tiles (`load_range_priors`)

    def _priors(self, cls: str) -> tuple[float | None, float]:
        """(speed prior or None, range prior or 0.0) of a track's class."""
        from .detect import to_card_slug
        slug = to_card_slug(cls)
        return self.speed_priors.get(slug), float(self.range_priors.get(slug, 0.0))

    def _prior_velocity(self, cls: str, side: str, category: str) -> tuple[float, float]:
        if category in STATIC_TYPES:
            return 0.0, 0.0
        from .detect import to_card_slug
        sp = self.speed_priors.get(to_card_slug(cls), DEFAULT_SPEED)
        return 0.0, sp if side == "ally" else -sp

    def _category(self, cls: str, det_category: str) -> str:
        if det_category in ("tower", "spell"):
            return det_category
        from .detect import to_card_slug
        return self.categories.get(to_card_slug(cls), det_category or "unit")

    def predict(self, t: float) -> None:
        """Propagate every track to time t (call every frame)."""
        self.t = t
        for tr in self.tracks.values():
            tr.advance(t, self.process_noise)

    def update(self, t: float, detections: list) -> list[KalmanTrack]:
        """detections: objects with .cls .side .category .conf .bbox and a
        continuous tile position .pos_f = (c, r). Greedy gated nearest
        association (Mahalanobis), class-compatible (same class, or either
        side is unknown_unit)."""
        self.predict(t)
        unmatched = []
        used: set[int] = set()
        cands = []
        for i, d in enumerate(detections):
            z = np.array(d.pos_f, dtype=float)
            for tid, tr in self.tracks.items():
                if tr.side != d.side and d.side != "unknown":
                    continue
                if tr.cls != d.cls and "unknown_unit" not in (tr.cls, d.cls):
                    continue
                m = tr.mahalanobis(z, t, self.meas_noise)
                if m <= self.gate:
                    cands.append((m, i, tid))
        cands.sort()
        assigned: dict[int, int] = {}
        for m, i, tid in cands:
            if i in assigned or tid in used:
                continue
            assigned[i] = tid
            used.add(tid)
        for i, d in enumerate(detections):
            z = np.array(d.pos_f, dtype=float)
            if i in assigned:
                tr = self.tracks[assigned[i]]
                if tr.cls == "unknown_unit" and d.cls != "unknown_unit":
                    tr.cls = d.cls          # a later, better read of the same unit
                    tr.speed_prior, tr.range_prior = self._priors(d.cls)
                tr.update(z, t, self.meas_noise, self.process_noise, d.bbox, d.conf)
            else:
                unmatched.append(d)
        for d in unmatched:
            cat = self._category(d.cls, d.category)
            v = self._prior_velocity(d.cls, d.side, cat)
            x = np.array([d.pos_f[0], d.pos_f[1], v[0], v[1]], dtype=float)
            P = np.diag([self.meas_noise ** 2, self.meas_noise ** 2, 0.5 ** 2, 0.5 ** 2])
            tr = KalmanTrack(self.next_id, d.cls, d.side, cat, x, P, t, 1, 0, t, t, d.bbox, d.conf)
            tr.speed_prior, tr.range_prior = self._priors(d.cls)
            tr.history.append((t, float(d.pos_f[0]), float(d.pos_f[1])))
            self.tracks[self.next_id] = tr
            self.next_id += 1
        # expire
        for tid in [k for k, tr in self.tracks.items() if t - tr.last_meas_t > self.max_misses_s]:
            del self.tracks[tid]
        return self.confirmed()

    def confirmed(self) -> list[KalmanTrack]:
        return [tr for tr in self.tracks.values() if tr.hits >= self.confirm_hits]

    def reset(self) -> None:
        self.tracks.clear()
        self.next_id = 1

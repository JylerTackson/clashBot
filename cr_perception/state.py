"""The perception -> agent contract. Keep this stable."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class UnitObs:
    cls: str            # unit class name (KataCR label), or "unknown_unit"
    side: str           # "ally" | "enemy" | "unknown"
    tile: tuple[int, int] | None
    conf: float
    bbox: tuple[int, int, int, int]
    track_id: int | None = None
    category: str = "unit"  # "unit" | "spell" | "tower" | "ui"

    def to_json(self) -> dict:
        return {"class": self.cls, "side": self.side, "tile": list(self.tile) if self.tile else None,
                "conf": round(self.conf, 3), "bbox": list(self.bbox), "track_id": self.track_id,
                "category": self.category}


@dataclass
class PlayEvent:
    timestamp: float
    match_clock: str | None
    player: str                    # "own" | "opponent"
    card: str | None               # Phase 1 slug, or None when unidentified
    tile: tuple[int, int] | None
    elixir_before: float | None
    elixir_after: float | None
    detect_source: str             # "hud" | "arena" | "inferred"
    confidence: str                # "high" | "medium" | "low"
    detail: str = ""

    def to_json(self) -> dict:
        d = asdict(self)
        d["tile"] = list(self.tile) if self.tile else None
        d["type"] = "play_event"
        return d


@dataclass
class GameState:
    t: float
    frame_index: int
    readiness: str                 # "match" | "menu" | "unreadable" | "loading"
    match_clock: str | None = None
    match_seconds: float | None = None
    phase: str | None = None       # "single_elixir" | "double_elixir" | "triple_elixir" (+ "_overtime")
    own: dict = field(default_factory=dict)
    opponent: dict = field(default_factory=dict)
    units: list[UnitObs] = field(default_factory=list)
    legal_mask_hash: str | None = None
    field_confidence: dict = field(default_factory=dict)
    stale: dict = field(default_factory=dict)   # field -> seconds since last confident read

    def to_json(self) -> dict:
        return {"type": "state", "t": round(self.t, 3), "frame": self.frame_index, "readiness": self.readiness,
                "match_clock": self.match_clock, "match_seconds": self.match_seconds, "phase": self.phase,
                "own": self.own, "opponent": self.opponent, "units": [u.to_json() for u in self.units],
                "legal_mask_hash": self.legal_mask_hash, "field_confidence": self.field_confidence,
                "stale": self.stale}

    def dumps(self) -> str:
        return json.dumps(self.to_json(), separators=(",", ":"))


def dumps_any(obj: Any) -> str:
    if hasattr(obj, "to_json"):
        return json.dumps(obj.to_json(), separators=(",", ":"))
    return json.dumps(obj, separators=(",", ":"))

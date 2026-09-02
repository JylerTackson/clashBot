"""Match clock tracking shared by the live pipeline and the context builder.

Clash Royale's clock counts down from 3:00; at 0:00 a tied game goes to
overtime and the clock restarts from 2:00 (then 1:00 for 2v2/other modes is
not handled). A creator playing back-to-back games shows a fresh 3:00 (first
readable value ~2:5x) after the loading screen. OCR misreads produce
impossible values (9:59) and one-frame glitches, so every upward jump must be
confirmed by two consecutive consistent reads before it changes state.
"""
from __future__ import annotations

from dataclasses import dataclass, field

MAX_REMAINING = 180
NEW_MATCH_MIN = 150       # a confirmed jump to >= 2:30 is a new game
OVERTIME_RANGE = (95, 125)  # a confirmed jump to ~2:00 (or 1:xx) from <= 0:15 is overtime
JUMP_MIN = 20


@dataclass
class ClockTracker:
    remaining: int | None = None
    overtime: bool = False
    match_index: int = 0
    _cand: tuple[int, int] | None = None   # (value, consecutive count)
    events: list[tuple[float, str, int]] = field(default_factory=list)  # (t, "new_match"|"overtime", value)

    def valid(self, remaining: int | None) -> bool:
        return remaining is not None and 0 <= remaining <= MAX_REMAINING

    def update(self, t: float, remaining: int | None) -> str | None:
        """Feed one clock read (seconds remaining, or None). Returns
        "new_match", "overtime" or None. Impossible values are ignored."""
        if not self.valid(remaining):
            return None
        if self.remaining is None:
            self.remaining = remaining
            return None
        if remaining - self.remaining > JUMP_MIN:
            # upward jump: needs confirmation (a second read that continues the new clock)
            if self._cand and abs(self._cand[0] - remaining) <= 3:
                self._cand = (remaining, self._cand[1] + 1)
            else:
                self._cand = (remaining, 1)
                return None
            if self._cand[1] < 2:
                return None
            prev = self.remaining
            self._cand = None
            if remaining >= NEW_MATCH_MIN:
                self.match_index += 1
                self.overtime = False
                self.remaining = remaining
                self.events.append((t, "new_match", remaining))
                return "new_match"
            if OVERTIME_RANGE[0] <= remaining <= OVERTIME_RANGE[1] and prev <= 15 and not self.overtime:
                self.overtime = True
                self.remaining = remaining
                self.events.append((t, "overtime", remaining))
                return "overtime"
            # confirmed but unexplained (e.g. a replay/menu clock): adopt silently
            self.remaining = remaining
            return None
        self._cand = None
        if remaining <= self.remaining or self.remaining - remaining <= 3:
            self.remaining = remaining
        return None

    def phase(self) -> str | None:
        r = self.remaining
        if r is None:
            return None
        if self.overtime:
            return "double_elixir_overtime" if r > 60 else "triple_elixir_overtime"
        if r > 120:
            return "single_elixir"
        if r > 60:
            return "double_elixir"
        return "triple_elixir"

    def regen_key(self) -> str:
        p = self.phase() or "single_elixir"
        return "double" if p.startswith("double") else "triple" if p.startswith("triple") else "single"


def segment_by_clock(samples: list[tuple[float, int | None]], min_seconds: float = 60.0) -> list[tuple[int, float, float]]:
    """Offline version: (game_index, t0, t1) segments from a (t, remaining)
    series using the same confirmation rules."""
    ct = ClockTracker()
    segs: list[list[float]] = []
    cur_idx = -1
    for t, r in samples:
        ev = ct.update(t, r)
        if ev == "new_match" or cur_idx < 0:
            if cur_idx >= 0 and ev == "new_match":
                pass
            cur_idx += 1
            segs.append([t, t])
        else:
            segs[-1][1] = t
    return [(i, a, b) for i, (a, b) in enumerate(segs) if b - a >= min_seconds]

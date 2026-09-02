"""Elixir simulation with a self-validating drift measurement.

elixir(t) = clamp(start + regen(t, phase) - sum(costs of observed plays), 0, 10)

Regeneration (seconds per elixir): single 2.8, double 1.4, triple 0.93. The
same simulator is run against the OWN player, whose elixir is observable in
the HUD; the difference between simulated-own and observed-own is the error
bar we attach to the opponent estimate.
"""
from __future__ import annotations

from dataclasses import dataclass, field

PHASE_SECONDS_PER_ELIXIR = {"single": 2.8, "double": 1.4, "triple": 2.8 / 3.0}
MAX_ELIXIR = 10.0
START_ELIXIR = 5.0  # both players start a ladder match with 5 (set at first observation)


@dataclass
class ElixirSimulator:
    elixir: float = START_ELIXIR
    phase: str = "single"
    t_last: float | None = None
    spent: float = 0.0
    plays: int = 0
    # drift bookkeeping (only meaningful for the own-side instance)
    drift_samples: list[tuple[float, float]] = field(default_factory=list)  # (t, sim - observed)
    resyncs: int = 0

    def advance(self, t: float, phase: str | None = None) -> float:
        if phase:
            self.phase = phase
        if self.t_last is None:
            self.t_last = t
            return self.elixir
        dt = max(0.0, t - self.t_last)
        self.t_last = t
        self.elixir = min(MAX_ELIXIR, self.elixir + dt / PHASE_SECONDS_PER_ELIXIR[self.phase])
        return self.elixir

    def spend(self, cost: float) -> None:
        self.elixir = max(0.0, self.elixir - cost)
        self.spent += cost
        self.plays += 1

    def observe(self, t: float, observed: int | None) -> float | None:
        """Own-side only: record sim - observed drift. Returns the drift."""
        if observed is None:
            return None
        d = self.elixir - observed
        self.drift_samples.append((t, d))
        return d

    def resync(self, observed: float) -> None:
        self.elixir = float(observed)
        self.resyncs += 1

    def reset(self, t: float | None = None, start: float = START_ELIXIR, phase: str = "single") -> None:
        self.elixir, self.phase, self.t_last = start, phase, t
        self.spent, self.plays = 0.0, 0
        self.drift_samples.clear()
        self.resyncs = 0

    # --- reporting --------------------------------------------------------
    def drift_stats(self, window: int | None = None) -> dict:
        s = self.drift_samples[-window:] if window else self.drift_samples
        if not s:
            return {"n": 0, "mean": None, "abs_mean": None, "max_abs": None, "last": None}
        ds = [d for _, d in s]
        return {"n": len(ds), "mean": round(sum(ds) / len(ds), 3),
                "abs_mean": round(sum(abs(d) for d in ds) / len(ds), 3),
                "max_abs": round(max(abs(d) for d in ds), 3), "last": round(ds[-1], 3)}

    def estimate(self, drift_abs_mean: float | None) -> tuple[float, float, tuple[float, float]]:
        """(point estimate, confidence 0-1, (low, high)) using the own-side
        drift as the error bar; +/-1 elixir when no drift data yet."""
        err = 1.0 if drift_abs_mean is None else max(0.5, min(5.0, drift_abs_mean + 0.5))
        lo, hi = max(0.0, self.elixir - err), min(MAX_ELIXIR, self.elixir + err)
        conf = max(0.0, 1.0 - err / 5.0)
        return round(self.elixir, 2), round(conf, 2), (round(lo, 2), round(hi, 2))

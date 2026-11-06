"""Lookup-vs-tactics arbitration (MAC-AB A4, brief R7): the LLM makes every play decision; the arbiter only narrows its
options using tool evidence (D-045).

Every live candidate says where it came from (`sources`: `tac` = the tactics engine, `knn` = pros played it in similar
states, `kg` = a known counter). When d1 is unsure, `RuleArbiter` picks a *subset* of the candidates (the tactical ones,
the pro-supported ones) and d1 chooses among them: the pick is the argmax of d1's own probabilities restricted to the
subset (renormalising over the subset does not change the argmax, so no second d1 call is needed). The arbiter never
picks a candidate itself. Pure functions, no I/O, no models.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ArbiterConfig:
    conf: float = 0.55         # d1 top probability at which d1 is trusted over all candidates
    urgent_eta_s: float = 4.0  # a push this close -> d1 chooses among the tactical answers (+ best lookup) when unsure
    knn_min: float = 0.5       # kNN support at which a candidate counts as "pros played this"


@dataclass
class ArbiterDecision:
    index: int                 # index into the candidates: d1's argmax within `subset`
    reason: str
    subset: list[int] = field(default_factory=list)       # candidate indices d1 chose among (ascending)
    probs: list[float] = field(default_factory=list)      # d1's probabilities over `subset`, renormalised (sum 1)


def argmax(vals: list[float]) -> int:
    best = 0
    for i, v in enumerate(vals):
        if v > vals[best]:
            best = i                      # strict '>' keeps the first of equal values
    return best


def source_value(cand, name: str) -> float | None:
    value = (getattr(cand, "sources", None) or {}).get(name)
    return None if value is None else float(value)


def tac_indices(cands: list) -> list[int]:
    """Candidates that came from the tactics engine."""
    return [i for i, c in enumerate(cands) if source_value(c, "tac") is not None]


def knn_indices(cands: list, minimum: float) -> list[int]:
    """Candidates whose kNN support is >= `minimum` (and positive)."""
    return [i for i, c in enumerate(cands) if (source_value(c, "knn") or 0.0) > 0.0
            and source_value(c, "knn") >= minimum]


def best_lookup(cands: list) -> int | None:
    """The non-tactical candidate with the highest positive kNN support (ties -> the first), else None."""
    best: int | None = None
    best_v = 0.0
    for i, c in enumerate(cands):
        v = source_value(c, "knn") or 0.0
        if source_value(c, "tac") is None and v > best_v:
            best, best_v = i, v
    return best


def pick_within(probs: list[float], subset: list[int]) -> tuple[int, list[float]]:
    """d1's choice among `subset` (candidate indices): (the index with the highest probability, first on ties; the
    probabilities of the subset renormalised to sum 1 - uniform when they are all 0). `probs` shorter than the
    candidate list counts the missing entries as 0."""
    vals = [float(probs[i]) if i < len(probs) else 0.0 for i in subset]
    total = sum(vals)
    renorm = [v / total for v in vals] if total > 0 else [1.0 / len(vals)] * len(vals)
    return subset[argmax(vals)], renorm


class RuleArbiter:
    def __init__(self, cfg: ArbiterConfig | None = None):
        self.cfg = cfg if cfg is not None else ArbiterConfig()

    def decide(self, cands: list, probs: list[float], urgent: bool) -> ArbiterDecision:
        """cands: Candidate-like (`key`, `sources` dict); probs: d1's probability per candidate (empty for a single
        candidate); urgent: an enemy push is close (see `urgent_from_report`). The subset d1 chooses among, by rule:
        1. d1 confident (max prob >= conf) -> all candidates;
        2. unsure, urgent and tactical candidates exist -> the tactical ones plus the best lookup candidate;
        3. unsure and candidates with knn >= knn_min exist -> those plus the tactical ones;
        4. otherwise -> all candidates.
        The pick is always d1's argmax within the subset (first on ties); without probs it is index 0."""
        if not cands:
            raise ValueError("decide needs at least one candidate")
        everyone = list(range(len(cands)))
        if not probs:
            return ArbiterDecision(0, "no d1 probs", [0], [1.0])
        subset, reason = everyone, "d1"
        if max(probs[:len(cands)], default=0.0) >= self.cfg.conf:
            reason = "d1 confident"
        else:
            tac = tac_indices(cands)
            if urgent and tac:
                lookup = best_lookup(cands)
                subset = sorted(set(tac) | ({lookup} if lookup is not None else set()))
                reason = "urgent push: d1 among tactics+lookup"
            else:
                pros = knn_indices(cands, self.cfg.knn_min)
                if pros:
                    subset = sorted(set(pros) | set(tac))
                    reason = "d1 among pro-supported + tactics"
        index, renorm = pick_within(probs, subset)
        return ArbiterDecision(index, reason, subset, renorm)

    def choose(self, cands: list, probs: list[float], urgent: bool) -> tuple[int, str]:
        """(index, reason) of `decide`: the index is d1's argmax within the rule's subset."""
        d = self.decide(cands, probs, urgent)
        return d.index, d.reason


def urgent_from_report(report, urgent_eta_s: float) -> bool:
    """True if the tactics report has a push whose ETA to an own tower is <= `urgent_eta_s` (False without a report)."""
    if report is None:
        return False
    return any(p.eta_s <= urgent_eta_s for p in (getattr(report, "pushes", None) or []))

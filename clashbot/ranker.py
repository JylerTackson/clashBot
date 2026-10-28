"""Fast-loop ranker interface and stubs (D-007). Model weights are not available in this environment."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

LETTERS = "ABCDEFGH"


@dataclass
class RankResult:
    index: int
    probs: list[float]
    model: str


def format_multiple_choice(context: str, options: list[str], plan: str | None = None) -> str:
    if not options or len(options) > len(LETTERS):
        raise ValueError(f"need 1..{len(LETTERS)} options, got {len(options)}")
    lines = [context]
    if isinstance(plan, str) and plan:
        lines.append(f"[plan] {plan}")
    lines.append("Options:")
    for letter, opt in zip(LETTERS, options):
        lines.append(f"{letter}) {opt}")
    lines.append("Answer:")
    return "\n".join(lines)


class Ranker(Protocol):
    name: str

    def choose(self, context: str, options: list[str], plan: str | None = None) -> RankResult: ...


class ScoreRanker:
    """Baseline: trusts the candidate generator order."""
    name = "score"

    def choose(self, context: str, options: list[str], plan: str | None = None) -> RankResult:
        return RankResult(index=0, probs=[1.0] + [0.0] * (len(options) - 1), model=self.name)


class _StubDecisionModel:
    """Both models return probabilities over a fixed answer set; the answer set is the option letters A-H.
    Fine-tuning data: dataset/v1/ranker/*.jsonl rendered with format_multiple_choice."""
    name = "stub"
    model_id = ""

    def __init__(self, model_path: str | None = None):
        self.model_path = model_path

    def choose(self, context: str, options: list[str], plan: str | None = None) -> RankResult:
        raise NotImplementedError(
            f"{self.name}: load {self.model_id} on the Mac (see docs/design/pipeline_review.md §5); "
            "input is format_multiple_choice(...), output is a probability per option letter")


class LayaRanker(_StubDecisionModel):
    name = "laya"
    model_id = "convaiinnovations/laya (≈421M, Apache-2.0)"


class D1Ranker(_StubDecisionModel):
    name = "d1"
    model_id = "LiquidAI/d1-omni-600M (LFM Open License)"

"""Fast-loop rankers (D-007): ScoreRanker baseline, Laya stub, and D1Ranker running LiquidAI d1-omni-600M through mlx-vlm (default) or transformers (D-026)."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

from clashbot import d1_format

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


D1_MLX_ID = "LiquidAI/d1-omni-600M-MLX-fp16"
D1_TORCH_ID = "LiquidAI/d1-omni-600M"
D1_BACKENDS = ("mlx", "transformers")


def argmax_first(scores: list[float]) -> int:
    """Index of the first maximum (ties go to the earliest option)."""
    best = 0
    for i, s in enumerate(scores):
        if s > scores[best]:
            best = i
    return best


class D1Ranker:
    """Zero-shot LiquidAI d1-omni-600M scoring of the joint play question (D-022, D-023, D-026).

    backend "mlx" (default): mlx_vlm.predict on the MLX fp16 weights; "transformers": AutoModel.system_one
    (reference). The model is loaded lazily on the first `probabilities` call; `predict_fn(state, questions)`
    replaces the backend call (tests)."""
    name = "d1"

    def __init__(self, model_path: str | None = None, backend: str = "mlx", device: str | None = None,
                 predict_fn=None):
        if backend not in D1_BACKENDS:
            raise ValueError(f"unknown d1 backend {backend!r}; expected one of {D1_BACKENDS}")
        self.model_path = model_path
        self.backend = backend
        self.device = device
        self.predict_fn = predict_fn
        self.model_id = model_path or (D1_MLX_ID if backend == "mlx" else D1_TORCH_ID)
        self.last_ms = 0.0
        self._model = None
        self._processor = None

    def build_request(self, context: str, candidate_keys: list[str], plan: str | None = None) -> tuple[str, dict]:
        return d1_format.joint_request(context, candidate_keys, plan)

    # --- backends -------------------------------------------------------------------------------
    def _load(self) -> None:
        if self._model is not None:
            return
        if self.backend == "mlx":
            from mlx_vlm import load
            self._model, self._processor = load(self.model_id)
        else:
            import torch
            from transformers import AutoModel
            dev = self.device or ("mps" if torch.backends.mps.is_available() else "cpu")
            dtype = torch.float32 if dev == "cpu" else torch.float16
            self._model = AutoModel.from_pretrained(self.model_id, trust_remote_code=True,
                                                    dtype=dtype).to(dev).eval()

    def _predict(self, state: str, questions: dict) -> dict:
        if self.predict_fn is not None:
            return self.predict_fn(state, questions)
        self._load()
        if self.backend == "mlx":
            from mlx_vlm import predict
            return predict(self._model, self._processor, state, questions)
        import torch
        with torch.no_grad():
            return self._model.system_one(state, questions)

    # --- scoring --------------------------------------------------------------------------------
    def probabilities(self, state: str, questions: dict) -> dict[str, list[float]]:
        """{question name: probabilities in criteria order}; a noul is [P(false), P(true)]."""
        d1_format.validate(state, questions)
        t0 = time.perf_counter()
        raw = self._predict(state, questions)
        self.last_ms = (time.perf_counter() - t0) * 1000.0
        out: dict[str, list[float]] = {}
        for name, q in questions.items():
            answers = raw.get("answers", {})
            if name not in answers:
                raise ValueError(f"d1 answer for question {name!r} is missing")
            out[name] = _normalise(name, q, answers[name])
        return out

    def choose_keys(self, context: str, candidate_keys: list[str], plan: str | None = None) -> RankResult:
        state, questions = self.build_request(context, candidate_keys, plan)
        probs = self.probabilities(state, questions)[d1_format.JOINT_NAME]
        return RankResult(index=argmax_first(probs), probs=probs, model=f"d1-{self.backend}")

    def choose(self, context: str, options: list[str], plan: str | None = None) -> RankResult:
        if all(len(o.split("|")) == 3 for o in options):
            return self.choose_keys(context, options, plan)
        raise ValueError("D1Ranker.choose needs action keys (card|lane|zone); use choose_keys")


def _normalise(name: str, question: dict, ans: dict) -> list[float]:
    """One backend answer (mlx or transformers style) -> list of probabilities."""
    p = ans.get("probabilities")
    qtype = question["type"]
    if qtype in ("noul", "bool"):
        y = ans.get("probability", ans.get("noul"))
        if y is None and isinstance(p, dict):
            y = p.get("true", p.get("yes"))
        if y is None:
            raise ValueError(f"question {name!r}: noul answer has no probability")
        return [1.0 - float(y), float(y)]
    if p is None:
        raise ValueError(f"question {name!r}: answer has no probabilities")
    if qtype == "score":
        if isinstance(p, dict):
            return [float(p[k]) for k in sorted(p, key=int)]
        return [float(x) for x in p]
    keys = list(question["criteria"])
    if isinstance(p, dict):
        return [float(p[k]) for k in keys]
    probs = [float(x) for x in p]
    if len(probs) != len(keys):
        raise ValueError(f"question {name!r}: got {len(probs)} probabilities for {len(keys)} options")
    return probs

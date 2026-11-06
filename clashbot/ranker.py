"""Fast-loop rankers (D-007): ScoreRanker baseline, TacticsRanker (the tactics engine's own pick, MAC-Z3), Laya stub, and D1Ranker running LiquidAI d1-omni-600M or d1-2B through mlx-vlm (default) or transformers (D-026, D-046)."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
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


class TacticsRanker:
    """Picks the candidate the tactics engine ranked highest: the largest `sources["tac"]` (ties: the first), else the
    first candidate. Used through `choose_candidates` by the live loop; no model, no context."""
    name = "tactics"

    def choose_candidates(self, cands: list) -> RankResult:
        tac = [((c.get("sources") if isinstance(c, dict) else c.sources) or {}).get("tac") for c in cands]
        scored = [i for i, v in enumerate(tac) if v is not None]
        index = max(scored, key=lambda i: (tac[i], -i)) if scored else 0
        return RankResult(index=index, probs=[1.0 if i == index else 0.0 for i in range(len(cands))],
                          model=self.name)

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
D1_DTYPES = ("auto", "bf16", "fp16", "fp32")


def read_model_config(model_id: str) -> dict | None:
    """config.json of a local checkpoint dir or a Hugging Face repo (cache first), else None."""
    try:
        path = Path(model_id) / "config.json"
        if path.exists():
            return json.loads(path.read_text())
        from huggingface_hub import hf_hub_download
        return json.loads(Path(hf_hub_download(model_id, "config.json")).read_text())
    except Exception:
        return None


def is_lfm3_config(config: dict | None) -> bool:
    """True for the LFM3-based d1 models (d1-2B): `model_type` lfm3*, an lfm3 text model or an Lfm3 architecture."""
    if not config:
        return False
    text = config.get("text_config") or {}
    kinds = [str(config.get("model_type", "")), str(text.get("model_type", "") if isinstance(text, dict) else "")]
    names = " ".join(str(a) for a in config.get("architectures") or [])
    return any(k.startswith("lfm3") for k in kinds) or "lfm3" in names.lower()


def resolve_dtype(name: str, lfm3: bool, device: str):
    """torch dtype for `--d1-dtype`: auto = bf16 for LFM3 models (the d1-2B card's choice on MPS), else fp16 off the
    cpu; the cpu always gets fp32 under auto (its bf16 / fp16 kernels are slow)."""
    import torch
    if name == "auto":
        return torch.float32 if device == "cpu" else torch.bfloat16 if lfm3 else torch.float16
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def argmax_first(scores: list[float]) -> int:
    """Index of the first maximum (ties go to the earliest option)."""
    best = 0
    for i, s in enumerate(scores):
        if s > scores[best]:
            best = i
    return best


class D1Ranker:
    """LiquidAI d1 scoring of the joint play question (D-022, D-023, D-026): d1-omni-600M, or d1-2B (`model_path`
    "LiquidAI/d1-2B" or a fine-tuned LFM3 checkpoint, D-046; transformers backend).

    backend "mlx" (default): mlx_vlm.predict on the MLX fp16 weights; "transformers": AutoModel.system_one
    (reference). `dtype` (transformers only): auto | bf16 | fp16 | fp32; auto = bf16 for LFM3 models (detected from
    config.json), fp16 otherwise, fp32 on the cpu (`resolve_dtype`). The model is loaded lazily on the first
    `probabilities` call; `predict_fn(state, questions)` replaces the backend call (tests)."""
    name = "d1"

    def __init__(self, model_path: str | None = None, backend: str = "mlx", device: str | None = None,
                 predict_fn=None, dtype: str = "auto"):
        if backend not in D1_BACKENDS:
            raise ValueError(f"unknown d1 backend {backend!r}; expected one of {D1_BACKENDS}")
        if dtype not in D1_DTYPES:
            raise ValueError(f"unknown d1 dtype {dtype!r}; expected one of {D1_DTYPES}")
        self.model_path = model_path
        self.backend = backend
        self.device = device
        self.dtype = dtype
        self.predict_fn = predict_fn
        self.model_id = model_path or (D1_MLX_ID if backend == "mlx" else D1_TORCH_ID)
        self.last_ms = 0.0
        self._model = None
        self._processor = None

    def build_request(self, context: str, candidate_keys: list[str], plan: str | None = None,
                      notes: dict[str, str] | None = None) -> tuple[str, dict]:
        return d1_format.joint_request(context, candidate_keys, plan, notes)

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
            dtype = resolve_dtype(self.dtype, is_lfm3_config(read_model_config(self.model_id)), dev)
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

    def choose_keys(self, context: str, candidate_keys: list[str], plan: str | None = None,
                    notes: dict[str, str] | None = None) -> RankResult:
        state, questions = self.build_request(context, candidate_keys, plan, notes)
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

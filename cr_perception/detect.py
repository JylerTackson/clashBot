"""Arena unit detection with pretrained weights (nothing is trained here).

Backends:
  KataCRDetector    - wty-yy/KataCR YOLOv8-l x2 (85 classes each, 896 px),
                      outputs xyxy, conf, cls, bel (0 = ally/blue, 1 = enemy/red).
  BuildABotDetector - Pbatch/ClashRoyaleBuildABot ONNX (97 unit classes,
                      480x352, fp16) + a 16x16 side classifier.

Both return Detection objects in CONTENT-rect pixel coordinates. Class names
are mapped to a common vocabulary (Phase 1 card slugs where a unit maps to a
card; otherwise the detector's own label). Detections with objectness but a
weak class score are emitted as `unknown_unit` rather than dropped: the
weights are frozen at mid-2024 and newer cards are not in them.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

SPELL_CLASSES = {"zap", "giant-snowball", "arrows", "the-log", "fireball", "rocket", "lightning", "poison",
                 "earthquake", "tornado", "freeze", "rage", "clone", "graveyard", "barbarian-barrel",
                 "royal-delivery", "goblin-barrel", "goblin-curse", "vines", "void", "mirror", "skeleton-king-skill"}
TOWER_CLASSES = {"king-tower", "queen-tower", "cannoneer-tower", "dagger-duchess-tower", "royal-chef-tower"}
UI_CLASSES = {"tower-bar", "king-tower-bar", "bar", "bar-level", "clock", "emote", "text", "elixir", "selected",
              "skeleton-king-bar", "dagger-duchess-tower-bar", "evolution-symbol", "ice-spirit-evolution-symbol",
              "dirt", "blood", "butterfly", "flower", "skull", "cup", "snow", "grave", "ruin", "circle",
              "king-tower-ruin", "big-text", "small-text", "scoreboard", "crown-icon", "king-tower-level"}


@dataclass
class Detection:
    cls: str
    conf: float
    bbox: tuple[float, float, float, float]   # x1, y1, x2, y2 in content pixels
    side: str                                 # "ally" | "enemy" | "unknown"
    side_conf: float = 0.0
    raw_cls: str = ""
    category: str = "unit"                    # unit | spell | tower | ui


def categorize(name: str) -> str:
    if name in UI_CLASSES:
        return "ui"
    if name in TOWER_CLASSES:
        return "tower"
    if name in SPELL_CLASSES:
        return "spell"
    return "unit"


# KataCR label -> Phase 1 card slug (only where a 1:1 card exists). Unit
# labels that are sub-units (golemite, lava-pup, skeleton from tombstone...)
# keep their own label.
KATACR_TO_CARD = {
    "pekka": "p-e-k-k-a", "mini-pekka": "mini-p-e-k-k-a", "x-bow": "x-bow", "the-log": "the-log",
    "archer": "archers", "barbarian": "barbarians", "bat": "bats", "goblin": "goblins", "spear-goblin": "spear-goblins",
    "skeleton": "skeletons", "minion": "minions", "elite-barbarian": "elite-barbarians", "royal-hog": "royal-hogs",
    "royal-recruit": "royal-recruits", "wall-breaker": "wall-breakers", "guard": "guards", "zappy": "zappies",
    "rascal-boy": "rascals", "rascal-girl": "rascals", "skeleton-dragon": "skeleton-dragons", "hog": "hog-rider",
    "elixir-golem-big": "elixir-golem", "phoenix-big": "phoenix", "phoenix-small": "phoenix",
}


def to_card_slug(label: str) -> str:
    base = label.replace("-evolution", "")
    return KATACR_TO_CARD.get(base, base)


class KataCRDetector:
    """Direct inference on the KataCR YOLOv8 checkpoints (no dependency on the
    KataCR training/prediction code, which needs jax and dataset paths).
    Output layout per anchor: 4 box (xywh) + (nc-1) class scores + 1 belonging
    score (>0.5 = enemy)."""

    def __init__(self, weights: list[str | Path], katacr_root: str | Path, imgsz: int = 896,
                 conf: float = 0.5, iou: float = 0.6, unknown_conf: float = 0.25, device: str = "cpu"):
        import torch
        sys.path.insert(0, str(katacr_root))  # katacr.yolov8.custom_model is needed to unpickle
        self.torch = torch
        self.models, self.names = [], []
        for w in weights:
            ck = torch.load(str(w), map_location="cpu", weights_only=False)
            m = (ck.get("ema") or ck["model"]).float().eval()
            for p in m.parameters():
                p.requires_grad_(False)
            self.models.append(m.to(device))
            self.names.append(dict(m.names))
        self.imgsz, self.conf, self.iou, self.unknown_conf, self.device = imgsz, conf, iou, unknown_conf, device
        self.last_ms = 0.0

    def _letterbox(self, img: np.ndarray):
        h, w = img.shape[:2]
        r = min(self.imgsz / h, self.imgsz / w)
        nh, nw = int(round(h * r)), int(round(w * r))
        nh, nw = max(32, nh - nh % 32), max(32, nw - nw % 32)   # stride-32 multiple, no padding waste
        res = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        return res, nw / w, nh / h

    def detect(self, content_bgr: np.ndarray, arena_crop: tuple[int, int, int, int] | None = None) -> list[Detection]:
        import torchvision
        torch = self.torch
        x0 = y0 = 0
        img = content_bgr
        if arena_crop:
            x0, y0, w, h = arena_crop
            img = content_bgr[y0:y0 + h, x0:x0 + w]
        t = time.perf_counter()
        lb, rx, ry = self._letterbox(img)
        x = torch.from_numpy(lb[..., ::-1].copy()).permute(2, 0, 1)[None].float().div_(255.0).to(self.device)
        rows = []
        with torch.no_grad():
            for m, names in zip(self.models, self.names):
                y = m(x)
                pred = (y[0] if isinstance(y, (list, tuple)) else y)[0].T   # (anchors, 89)
                nc = pred.shape[1] - 5
                box, cls, bel = pred[:, :4], pred[:, 4:4 + nc], pred[:, 4 + nc]
                conf, j = cls.max(1)
                keep = conf > self.unknown_conf
                if not keep.any():
                    continue
                box, conf, j, bel = box[keep], conf[keep], j[keep], bel[keep]
                xyxy = torch.stack([box[:, 0] - box[:, 2] / 2, box[:, 1] - box[:, 3] / 2,
                                    box[:, 0] + box[:, 2] / 2, box[:, 1] + box[:, 3] / 2], 1)
                for b, c, k, be in zip(xyxy.tolist(), conf.tolist(), j.tolist(), bel.tolist()):
                    rows.append((b[0] / rx, b[1] / ry, b[2] / rx, b[3] / ry, c, names.get(int(k), str(k)), int(be > 0.5)))
        if rows:
            arr = torch.tensor([[r[0], r[1], r[2], r[3]] for r in rows])
            sc = torch.tensor([r[4] for r in rows])
            keep = torchvision.ops.nms(arr, sc, self.iou).tolist()
            rows = [rows[i] for i in keep]
        self.last_ms = (time.perf_counter() - t) * 1000
        out = []
        for x1, y1, x2, y2, c, name, bel in rows:
            cat = categorize(name)
            if cat == "ui":
                continue
            side = "enemy" if bel == 1 else "ally"
            label = name if c >= self.conf else "unknown_unit"
            out.append(Detection(label, float(c), (x1 + x0, y1 + y0, x2 + x0, y2 + y0), side, 1.0, name,
                                 cat if label != "unknown_unit" else "unit"))
        return out


class BuildABotDetector:
    def __init__(self, repo_root: str | Path, conf: float = 0.3, unknown_conf: float = 0.15):
        import onnxruntime as ort
        root = Path(repo_root) / "clashroyalebuildabot"
        self.sess = ort.InferenceSession(str(root / "models" / "units_M_480x352.onnx"), providers=["CPUExecutionProvider"])
        self.side = ort.InferenceSession(str(root / "models" / "side.onnx"), providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        self.H, self.W = self.sess.get_inputs()[0].shape[2:]
        # Read the class list from constants.py textually: importing the
        # package would pull in its bot/keyboard (input) code, which this
        # read-only pipeline must not depend on.
        import re
        const = (root / "constants.py").read_text()
        block = const[const.index("DETECTOR_UNITS = ["):]
        block = block[:block.index("]")]
        self.names = [n.lower().replace("_", "-") for n in re.findall(r"Units\.([A-Z_0-9]+)", block)]
        self.conf, self.unknown_conf = conf, unknown_conf
        self.last_ms = 0.0

    def _letterbox(self, img: np.ndarray):
        h, w = img.shape[:2]
        r = min(self.H / h, self.W / w)
        nh, nw = int(h * r), int(w * r)
        res = cv2.resize(img, (nw, nh))
        top, left = (self.H - nh) // 2, (self.W - nw) // 2
        canvas = np.full((self.H, self.W, 3), 114, np.uint8)
        canvas[top:top + nh, left:left + nw] = res
        return canvas, r, left, top

    def detect(self, content_bgr: np.ndarray, arena_crop: tuple[int, int, int, int] | None = None) -> list[Detection]:
        x0 = y0 = 0
        img = content_bgr
        if arena_crop:
            x0, y0, w, h = arena_crop
            img = content_bgr[y0:y0 + h, x0:x0 + w]
        t = time.perf_counter()
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        lb, r, left, top = self._letterbox(rgb)
        x = lb.transpose(2, 0, 1)[None].astype(np.float16) / 255.0
        pred = self.sess.run(None, {self.inp: x})[0][0]
        out = []
        for x1, y1, x2, y2, c, k in pred:
            if c < self.unknown_conf:
                continue
            bx = ((x1 - left) / r, (y1 - top) / r, (x2 - left) / r, (y2 - top) / r)
            crop = rgb[max(0, int(bx[1])):int(bx[3]), max(0, int(bx[0])):int(bx[2])]
            side, sc = "unknown", 0.0
            if crop.size:
                s = cv2.resize(crop, (16, 16), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255.0
                p = self.side.run(None, {self.side.get_inputs()[0].name: s[None]})[0][0]
                side = ("ally", "enemy")[int(np.argmax(p))]
                sc = float(np.max(p) / (np.sum(np.abs(p)) + 1e-6))
            name = self.names[int(k)]
            label = name if c >= self.conf else "unknown_unit"
            out.append(Detection(label, float(c), (bx[0] + x0, bx[1] + y0, bx[2] + x0, bx[3] + y0), side, sc, name,
                                 categorize(name)))
        self.last_ms = (time.perf_counter() - t) * 1000
        return out

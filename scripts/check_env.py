#!/usr/bin/env python3
"""Environment check (docs/QUICKSTART.md step 1): what this machine has and what each part of the project can run.

    python scripts/check_env.py

Prints one line per requirement (ok / MISSING / warn) and a summary of what is runnable. Read-only: imports packages,
looks at files, never loads a model, never touches the network, never sends input. Exit code 0 when the packages the
tests need are installed, 1 otherwise (the data, the weights and the optional packages only change the summary).
"""
from __future__ import annotations

import importlib
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CALIB = "runs/poc/mirror_calib.json"                 # where the demo launcher looks
CALIB_EXAMPLE = "configs/mirror_calib.example.json"   # tracked: clashbot.live takes it with --calib
CLIP = "data/videos/Z8-4VhLjrGU/Z8-4VhLjrGU.mp4"
D1_DIR = "db/models/d1-ft-joint-tac"
ACT_DIR = "db/models/act-gbdt"
# (import name, pip name, needed for)
PACKAGES = [("numpy", "numpy", "everything"), ("scipy", "scipy", "everything"), ("cv2", "opencv-python-headless", "perception"),
            ("pytest", "pytest", "tests"), ("lightgbm", "lightgbm", "act model, feature ranker (needs brew libomp)"),
            ("torch", "torch", "KataCR, d1"), ("ultralytics", "ultralytics==8.1.24", "KataCR checkpoints"),
            ("transformers", "transformers", "d1 backend"), ("hnswlib", "hnswlib", "seed.py --hnsw, retrieval eval (optional)"),
            ("sentence_transformers", "sentence-transformers", "bge text channel (optional)"),
            ("pykeen", "pykeen", "KG embedding eval (optional)"), ("rapidocr_onnxruntime", "rapidocr-onnxruntime", "HUD OCR"),
            ("mlx", "mlx", "MLX backends (optional)"), ("Quartz", "pyobjc-framework-Quartz", "iPhone Mirroring capture, live taps"),
            ("huggingface_hub", "huggingface_hub", "d1 loading")]


def version_of(module) -> str:
    return str(getattr(module, "__version__", "") or "")


def main() -> int:
    ok_all = True
    notes: list[str] = []

    def line(status: str, what: str, detail: str = "") -> None:
        print(f"  [{status:^7}] {what}" + (f"  {detail}" if detail else ""))

    print(f"clashbot environment check ({ROOT})")
    print("system")
    mac = sys.platform == "darwin"
    arm = platform.machine() in ("arm64", "aarch64")
    line("ok" if mac and arm else "warn", "platform", f"{platform.system()} {platform.machine()} {platform.mac_ver()[0] or ''}".strip())
    if not (mac and arm):
        notes.append("not an Apple-silicon Mac: no MPS, no iPhone Mirroring, no live play (everything else runs on CPU)")
    py_ok = sys.version_info[:2] == (3, 12)
    line("ok" if py_ok else "warn", "python", f"{platform.python_version()} ({sys.executable})" + ("" if py_ok else "  the pins were verified on 3.12"))

    print("packages")
    have: set[str] = set()
    for mod, pip_name, why in PACKAGES:
        try:
            m = importlib.import_module(mod)
            have.add(mod)
            line("ok", pip_name, version_of(m))
        except Exception as exc:                       # noqa: BLE001  a broken install is a finding, not a crash
            required = why in ("everything", "tests") or mod in ("cv2",)
            if required:
                ok_all = False
            line("MISSING" if required else "warn", pip_name, f"needed for {why}  ({type(exc).__name__})")
    if "torch" in have:
        import torch
        mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
        line("ok" if mps else "warn", "torch MPS", "available" if mps else "not available: --device cpu (about 6x slower per detection)")

    print("data (built or copied, never in the zip)")
    from clashbot.config import DATASET_DIR, DB_PATH, KG2_PATH
    data_ok = (DATASET_DIR / "states.jsonl").exists()
    line("ok" if data_ok else "warn", "dataset/v1", "built" if data_ok else "not built: python tools/data/build_clean_dataset.py (README: Rebuild the data)")
    db_ok = DB_PATH.exists()
    line("ok" if db_ok else "warn", "db/clashbot.sqlite", "" if db_ok else "not built: python tools/db/seed.py --hnsw")
    kg2_ok = KG2_PATH.exists()
    line("ok" if kg2_ok else "warn", "db/kg2.sqlite", "" if kg2_ok else "missing: knowledge graph v2 (prebuilt, not rebuildable from the zip)")

    print("perception")
    try:
        from cr_perception.detect import check_katacr_paths
        problems = check_katacr_paths()
    except Exception as exc:                           # noqa: BLE001  e.g. OpenCV missing: reported above
        problems = [f"cannot check ({type(exc).__name__})"]
    kata_ok = not problems
    line("ok" if kata_ok else "warn", "KataCR clone + weights", "" if kata_ok else "; ".join(problems))
    example_ok = (ROOT / CALIB_EXAMPLE).exists()
    line("ok" if example_ok else "warn", CALIB_EXAMPLE, "window geometry: pass it to clashbot.live with --calib" if example_ok else "missing")
    calib_ok = (ROOT / CALIB).exists()
    line("ok" if calib_ok else "warn", CALIB, "" if calib_ok else "the demo launcher reads this path: mkdir -p runs/poc && cp " + CALIB_EXAMPLE + " " + CALIB)
    clip_ok = (ROOT / CLIP).exists()
    line("ok" if clip_ok else "warn", CLIP, "" if clip_ok else "missing: any recorded Clash Royale video (the author's are not redistributed)")

    print("models")
    d1_ok = (ROOT / D1_DIR / "model.safetensors").exists()
    line("ok" if d1_ok else "warn", D1_DIR, "" if d1_ok else "missing (2.35 GB, stays on the author's Mac): use --ranker tactics")
    act_ok = (ROOT / ACT_DIR / "meta.json").exists()
    line("ok" if act_ok else "warn", ACT_DIR, "" if act_ok else "missing: use --policy reactive")

    print("what can run here")
    can = {
        "tests (python -m pytest -q --deselect tests/test_perception_latency.py)": ("pytest" in have and "numpy" in have, ""),
        "CPU evals (tools/eval/eval_tactics.py ...)": (data_ok and db_ok and "numpy" in have, "needs the built dataset/db"),
        "video dry run, engine only (--ranker tactics)": (kata_ok and example_ok and clip_ok and "torch" in have and db_ok,
                                                          "needs KataCR, the calib JSON and a clip"),
        "video dry run, full stack (--ranker d1 --policy fast)": (kata_ok and example_ok and clip_ok and d1_ok and act_ok and kg2_ok and db_ok,
                                                                  "needs the d1 weights, act-gbdt and kg2 too"),
        "demo / live (iPhone Mirroring)": (mac and "Quartz" in have and kata_ok and d1_ok and act_ok and kg2_ok and db_ok and calib_ok,
                                           "macOS + Quartz + the full stack"),
    }
    for name, (ok, why) in can.items():
        line("yes" if ok else "no", name, "" if ok else why)
    for note in notes:
        print("note:", note)
    print("result:", "required packages are installed" if ok_all else "required packages are missing (see MISSING above)")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())

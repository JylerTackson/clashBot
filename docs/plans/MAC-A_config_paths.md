# MAC-A — KataCR paths from env/config (spec for a Sonnet implementer)

Author: Opus · 2026-11-02 · Implements D-030. Read `docs/plans/MAC-00_common.md` first.

## Owns (only these files)
```
cr_perception/detect.py        (path handling only: add the helpers below; do not touch inference code)
tools/calibrate.py
tools/batch_process.py
tools/run_video.py
tools/demo_clip.py
tools/worker_setup.sh
tests/test_config_paths.py     (new)
```

## Problem
`/home/user/wty-yy/katacr` and a cloud scratch path are hard-coded in six places (`grep -rn "wty-yy\|CR_SCRATCH" tools cr_perception`). `tools/calibrate.py --detector katacr` even defaults to an empty weights list.

## 1. `cr_perception/detect.py` — add near the top, after the imports
```python
import os

DEFAULT_SCRATCH = Path.home() / "cr_scratch"
WEIGHT_FILES = ("katacr_d1.pt", "katacr_d2.pt")


def scratch_dir() -> Path:
    """$CR_SCRATCH, else ~/cr_scratch."""
    return Path(os.environ.get("CR_SCRATCH") or DEFAULT_SCRATCH).expanduser()


def katacr_root() -> Path:
    """$CLASHBOT_KATACR_ROOT, else <scratch>/katacr (the KataCR git clone, needed to unpickle the checkpoints)."""
    env = os.environ.get("CLASHBOT_KATACR_ROOT")
    return Path(env).expanduser() if env else scratch_dir() / "katacr"


def katacr_weights() -> list[Path]:
    """$CLASHBOT_KATACR_WEIGHTS (paths joined by os.pathsep), else <scratch>/katacr_d1.pt and katacr_d2.pt."""
    env = os.environ.get("CLASHBOT_KATACR_WEIGHTS")
    if env:
        return [Path(p).expanduser() for p in env.split(os.pathsep) if p]
    return [scratch_dir() / f for f in WEIGHT_FILES]


def torch_device() -> str:
    """$CLASHBOT_TORCH_DEVICE ("cpu" | "mps"), default "cpu"."""
    return os.environ.get("CLASHBOT_TORCH_DEVICE") or "cpu"


def check_katacr_paths(root: Path | None = None, weights: list[Path] | None = None) -> list[str]:
    """Human-readable problems (empty list = all present). Never raises."""
    root = katacr_root() if root is None else Path(root)
    weights = katacr_weights() if weights is None else [Path(w) for w in weights]
    problems = []
    if not (root / "katacr").is_dir():
        problems.append(f"KataCR source not found at {root} (expected {root}/katacr/); set CLASHBOT_KATACR_ROOT")
    for w in weights:
        if not w.is_file():
            problems.append(f"KataCR weights not found: {w}; set CLASHBOT_KATACR_WEIGHTS or CR_SCRATCH")
    return problems


def default_katacr_detector(imgsz: int = 896, conf: float = 0.5, device: str | None = None) -> "KataCRDetector":
    """KataCRDetector built from the env/config paths above; raises FileNotFoundError listing every missing path."""
    problems = check_katacr_paths()
    if problems:
        raise FileNotFoundError("; ".join(problems))
    return KataCRDetector([str(w) for w in katacr_weights()], str(katacr_root()), imgsz=imgsz, conf=conf,
                          device=device or torch_device())
```
`default_katacr_detector` must be defined **after** the `KataCRDetector` class (module bottom). The other helpers go after the imports. `detect.py` already imports `sys, time, Path, cv2, numpy`; add `import os` with the other stdlib imports. Do not change `KataCRDetector`, `BuildABotDetector`, `Detection`, `categorize`, `to_card_slug` or the class tables.

## 2. Tools — replace hard-coded paths
In every file below: delete the module-level `SCRATCH = os.environ.get("CR_SCRATCH", "/tmp/claude-0/...")` line; import the helpers lazily or at top **only where cv2/torch are already imported at top** (all of these tools import cv2 at top, so a top-level `from cr_perception.detect import katacr_root, katacr_weights` is fine).

| file | change |
|---|---|
| `tools/calibrate.py` | `--katacr-root` default `str(katacr_root())`; `--katacr-weights` default `[str(p) for p in katacr_weights()]` (keep `nargs="*"`); keep `--buildabot-root` default but make it `os.environ.get("CLASHBOT_BUILDABOT_ROOT", str(scratch_dir() / "clashroyalebuildabot"))` (add `import os`). Before building the KataCR detector call `check_katacr_paths(Path(a.katacr_root), [Path(w) for w in a.katacr_weights])`; if non-empty, print each problem and `return 4`. |
| `tools/batch_process.py` | `WEIGHTS = [str(p) for p in katacr_weights()]`; detector construction: `KataCRDetector(WEIGHTS, str(katacr_root()), imgsz=a.imgsz, device=torch_device())`. Add `--device` arg (default `torch_device()`) and pass it instead when given. |
| `tools/run_video.py` | `--katacr-root` default `str(katacr_root())`; `build_detector` uses `a.katacr_weights or [str(p) for p in katacr_weights()]`; add `--device` (default `torch_device()`) passed to `KataCRDetector(..., device=a.device)`; `--buildabot-root` default as in calibrate.py. |
| `tools/demo_clip.py` | same as run_video.py: `--katacr-root` default, weights from `katacr_weights()` in both the calibrate subprocess call (`"--katacr-weights", *[str(p) for p in katacr_weights()]`) and the detector, `--buildabot-root` default. |
| `tools/worker_setup.sh` | Mac-aware: no `/home/user` paths. `KATACR_ROOT="${CLASHBOT_KATACR_ROOT:-$CR_SCRATCH/katacr}"`; clone into `$KATACR_ROOT` if missing; weights into `$CR_SCRATCH` as now. Python: `PY="${PY:-python3}"` and use `$PY` everywhere. Torch install line: on `uname` = Darwin run `$PY -m pip install -q torch torchvision` (default wheel has MPS); otherwise keep the CPU index URL line. hnswlib needs Apple clang on macOS: export `CC=/usr/bin/clang CXX=/usr/bin/clang++` before pip on Darwin. Add `hnswlib tokenizers mss pyobjc-framework-Quartz` to the pip line on Darwin (`hnswlib tokenizers` everywhere). The python check block uses `sys.path.insert(0, os.environ["KATACR_ROOT"])` (export `KATACR_ROOT` before it). End message: print the three env vars to export. **Do not run this script** (the Mac is already set up); only `bash -n tools/worker_setup.sh`. |

Grep afterwards: `grep -rn "wty-yy\|/home/user\|claude-0" tools cr_perception` must print nothing.

## 3. Tests — `tests/test_config_paths.py`
Use `monkeypatch` for env vars and `tmp_path`.
1. `katacr_root()` with `CLASHBOT_KATACR_ROOT` set → that path; unset + `CR_SCRATCH=tmp` → `tmp/katacr`; both unset → `Path.home()/"cr_scratch"/"katacr"`.
2. `katacr_weights()` with `CLASHBOT_KATACR_WEIGHTS="a.pt" + os.pathsep + "b.pt"` → `[Path("a.pt"), Path("b.pt")]`; unset with `CR_SCRATCH=tmp` → `[tmp/katacr_d1.pt, tmp/katacr_d2.pt]`.
3. `torch_device()` default `"cpu"`, env `mps` → `"mps"`.
4. `check_katacr_paths(tmp_root, [tmp/missing.pt])` returns 2 problems on an empty tmp dir; after `mkdir tmp_root/katacr` and touching the weight file → `[]`.
5. `default_katacr_detector()` with env pointing at an empty tmp dir raises `FileNotFoundError` whose message names both missing pieces.
6. Static check: for each owned tool file, its text contains neither `"wty-yy"` nor `"/home/user"` nor `"claude-0"`.
7. `subprocess.run(["bash", "-n", "tools/worker_setup.sh"])` returns 0.
8. Real paths (skip unless `(Path.home()/"cr_scratch"/"katacr_d1.pt").exists()`): `check_katacr_paths()` returns `[]` with env vars cleared.

The module imports `cv2` at top; the test file may import `cr_perception.detect` directly (OpenCV is installed).

## 4. Smoke (allowed once)
```bash
$PY tools/calibrate.py --help | head -20
$PY -c "from cr_perception.detect import check_katacr_paths, katacr_root, katacr_weights; print(katacr_root(), katacr_weights(), check_katacr_paths())"
```

## 5. Heavy run (Opus, at integration — not you)
Full rebuild per `docs/PHASE3_OVERVIEW.md`, then `$PY -m pytest -q tests` including the OpenCV perception tests; `default_katacr_detector()` load + one `detect()` on a frame on cpu and mps.

## 6. Report
Per MAC-00 §5.

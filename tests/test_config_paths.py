"""KataCR path/device helpers read from env/config (D-030): no hard-coded cloud paths."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cr_perception import detect  # noqa: E402
from cr_perception.detect import (check_katacr_paths, default_katacr_detector, katacr_root,  # noqa: E402
                                  katacr_weights, torch_device)

ENV_VARS = ("CR_SCRATCH", "CLASHBOT_KATACR_ROOT", "CLASHBOT_KATACR_WEIGHTS", "CLASHBOT_TORCH_DEVICE")
OWNED_TOOLS = ("calibrate.py", "batch_process.py", "run_video.py", "demo_clip.py", "worker_setup.sh")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for k in ENV_VARS:
        monkeypatch.delenv(k, raising=False)


def test_katacr_root(monkeypatch, tmp_path):
    monkeypatch.setenv("CLASHBOT_KATACR_ROOT", str(tmp_path / "kroot"))
    assert katacr_root() == tmp_path / "kroot"
    monkeypatch.delenv("CLASHBOT_KATACR_ROOT")
    monkeypatch.setenv("CR_SCRATCH", str(tmp_path))
    assert katacr_root() == tmp_path / "katacr"
    monkeypatch.delenv("CR_SCRATCH")
    assert katacr_root() == Path.home() / "cr_scratch" / "katacr"


def test_katacr_weights(monkeypatch, tmp_path):
    monkeypatch.setenv("CLASHBOT_KATACR_WEIGHTS", "a.pt" + os.pathsep + "b.pt")
    assert katacr_weights() == [Path("a.pt"), Path("b.pt")]
    monkeypatch.delenv("CLASHBOT_KATACR_WEIGHTS")
    monkeypatch.setenv("CR_SCRATCH", str(tmp_path))
    assert katacr_weights() == [tmp_path / "katacr_d1.pt", tmp_path / "katacr_d2.pt"]


def test_torch_device(monkeypatch):
    assert torch_device() == "cpu"
    monkeypatch.setenv("CLASHBOT_TORCH_DEVICE", "mps")
    assert torch_device() == "mps"


def test_check_katacr_paths(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    weight = tmp_path / "missing.pt"
    problems = check_katacr_paths(root, [weight])
    assert len(problems) == 2
    assert "KataCR source not found" in problems[0] and "KataCR weights not found" in problems[1]
    (root / "katacr").mkdir()
    weight.touch()
    assert check_katacr_paths(root, [weight]) == []


def test_default_detector_missing_paths(monkeypatch, tmp_path):
    monkeypatch.setenv("CLASHBOT_KATACR_ROOT", str(tmp_path / "no_root"))
    monkeypatch.setenv("CLASHBOT_KATACR_WEIGHTS", str(tmp_path / "no_weights.pt"))
    with pytest.raises(FileNotFoundError) as exc:
        default_katacr_detector()
    msg = str(exc.value)
    assert "KataCR source not found" in msg and str(tmp_path / "no_root") in msg
    assert "KataCR weights not found" in msg and "no_weights.pt" in msg


def test_default_detector_defined_after_class():
    src = Path(detect.__file__).read_text()
    assert src.index("class KataCRDetector") < src.index("def default_katacr_detector")


@pytest.mark.parametrize("path", [ROOT / "tools" / n for n in OWNED_TOOLS] + [Path(detect.__file__)], ids=lambda p: p.name)
def test_no_hardcoded_paths(path):
    text = path.read_text()
    for bad in ("wty-yy", "/home/user", "claude-0"):
        assert bad not in text, f"{path.name} still contains {bad!r}"


def test_worker_setup_syntax():
    r = subprocess.run(["bash", "-n", str(ROOT / "tools" / "worker_setup.sh")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


@pytest.mark.skipif(not (Path.home() / "cr_scratch" / "katacr_d1.pt").exists(), reason="KataCR scratch not set up")
def test_real_paths_present():
    assert check_katacr_paths() == []

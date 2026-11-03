"""Shared fixtures: build the ranker, act, baselines, d1, KG and document files into a temp dir (never into dataset/v1)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REAL_DATASET = ROOT / "dataset" / "v1"
TOKENIZER = ROOT / "db" / "hf" / "d1_tokenizer.json"


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: real-model tests, run only with CLASHBOT_SLOW_TESTS=1")


def _temp_dataset(tmp_path_factory, prefix: str) -> Path:
    """Temp dir holding symlinks to the real states.jsonl / games.jsonl (the only inputs the builders read)."""
    if not (REAL_DATASET / "states.jsonl").exists():
        pytest.skip("dataset/v1/states.jsonl missing")
    d = tmp_path_factory.mktemp(prefix)
    for name in ("states.jsonl", "games.jsonl"):
        (d / name).symlink_to(REAL_DATASET / name)
    return d


@pytest.fixture(scope="session")
def built_dataset(tmp_path_factory):
    """Symlink the real states/games into a temp dir and build ranker, act, baselines and d1 there."""
    d = _temp_dataset(tmp_path_factory, "ds")
    d1_cmd = [sys.executable, str(ROOT / "tools/data/build_d1_dataset.py"), "--dataset", str(d)]
    if TOKENIZER.exists():
        d1_cmd += ["--tokenizer", str(TOKENIZER)]
    for cmd in ([sys.executable, str(ROOT / "tools/data/build_ranker_dataset.py"), "--dataset", str(d)],
                [sys.executable, str(ROOT / "tools/eval/eval_ranker_baselines.py"), "--dataset", str(d)],
                d1_cmd):
        subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=ROOT)
    return d


@pytest.fixture(scope="session")
def built_kg(tmp_path_factory):
    """Symlink the real states/games into a temp dir and run build_kg.py and build_documents.py there."""
    d = _temp_dataset(tmp_path_factory, "kg")
    for script in ("build_kg.py", "build_documents.py"):
        subprocess.run([sys.executable, str(ROOT / "tools" / "data" / script), "--dataset", str(d)],
                       check=True, capture_output=True, text=True, cwd=ROOT)
    return d

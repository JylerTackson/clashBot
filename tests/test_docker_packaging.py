"""Docker packaging (MAC-AV): pins match requirements-mac.txt, the entrypoint can never tap, the video path and the dashboard import
without Quartz / mss / mlx / pyobjc, `--live` and `--source mirroring` refuse on Linux, and the artefact builders keep what the
runtime reads. Nothing here builds an image, loads a model or sends input; the packaging files are skipped when this file runs
inside the image (it carries the code, not docker/ or scripts/)."""
from __future__ import annotations

import importlib.util
import re
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DOCKER = ROOT / "docker"
needs_docker_files = pytest.mark.skipif(not (DOCKER / "Dockerfile").exists(), reason="docker/ files are not part of this tree")
D1_REPO = "Jyler/clashbot-d1-ft-joint-tac"
D1_REV = "5b984f0cce7c6db6e8a768cc0df2d6836022a3e2"
MAC_ONLY = ("mlx", "mlx-lm", "mlx-vlm", "pyobjc-framework-quartz", "mss")


def no_secrets(text: str) -> bool:
    """The packaging files never mention the folder that holds the API key (~/.clashbot/) or its environment variable."""
    return ".clashbot/" not in text and "ANTHROPIC" not in text


def pins(path: Path) -> dict[str, str]:
    """{normalised package name: pinned version} of the `name==version` lines of a requirements file."""
    out = {}
    for line in path.read_text().splitlines():
        m = re.match(r"^([A-Za-z0-9_.\-]+)==([^\s#;]+)", line.strip())
        if m:
            out[m.group(1).lower().replace("_", "-")] = m.group(2)
    return out


# ---------------------------------------------------------------------------------------- the image's files
@needs_docker_files
def test_docker_requirements_follow_the_mac_pins():
    mac, docker = pins(ROOT / "requirements-mac.txt"), pins(DOCKER / "requirements-docker.txt")
    for name, version in docker.items():
        if name in mac:
            assert version == mac[name], f"{name}: docker {version} vs mac {mac[name]}"
    for name in MAC_ONLY:
        assert name not in docker, f"{name} is macOS-only"
    assert "opencv-python-headless" in docker and "opencv-python" not in docker
    assert "torch" not in docker and "torchvision" not in docker       # CPU wheels, installed by the Dockerfile
    # everything that runs on the video path is still there
    for name in ("numpy", "scipy", "onnxruntime", "transformers", "tokenizers", "huggingface-hub", "soundfile"):
        assert name in docker


@needs_docker_files
def test_dockerfile_installs_the_pinned_cpu_torch_and_the_no_deps_packages():
    text = (DOCKER / "Dockerfile").read_text()
    mac = pins(ROOT / "requirements-mac.txt")
    assert "FROM python:3.12-slim" in text
    assert f"torch=={mac['torch']}+cpu" in text and "https://download.pytorch.org/whl/cpu" in text
    assert re.search(r"torchvision==0\.29\.\d+\+cpu", text)             # the CPU index lacks 0.29.2 (see requirements-docker.txt)
    assert f"ultralytics=={mac['ultralytics']}" in text and f"rapidocr-onnxruntime=={mac['rapidocr-onnxruntime']}" in text
    assert "--no-deps" in text                                           # both declare the GUI build of OpenCV
    for env in ("HF_HUB_OFFLINE=1", "CLASHBOT_TORCH_DEVICE=cpu", "CLASHBOT_KATACR_ROOT=", "CLASHBOT_KATACR_WEIGHTS="):
        assert env in text
    for pkg in ("build-essential", "libgl1", "libglib2.0-0"):
        assert pkg in text
    assert "entrypoint.sh" in text and "8765" in text
    # d1 comes from Hugging Face at build time, pinned, in its own layer: after the packages, before the artefacts and the code,
    # and HF_HUB_OFFLINE=1 only after it
    assert f"snapshot_download('{D1_REPO}', revision='{D1_REV}', local_dir='/app/artifacts/d1-ft-joint-tac')" in text
    # cache-friendly order: Python packages, d1, the artefacts, the code
    order = [text.index(k) for k in ("requirements-docker.txt", "snapshot_download", "ENV HF_HUB_OFFLINE=1", "COPY artifacts/",
                                     "COPY clashbot/")]
    assert order == sorted(order)


@needs_docker_files
def test_entrypoint_is_a_dry_run_only_and_syntactically_valid():
    path = DOCKER / "entrypoint.sh"
    text = path.read_text()
    code = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    assert "--live" not in code and "CLASHBOT_ALLOW_LIVE" not in code
    assert "--dry-run" in text and "--source video" in text and "--host 0.0.0.0" in text
    assert "--follow" in text and "--record-shots" in text and "--policy fast" in text
    assert "d1 weights not present - the tactics engine is choosing moves" in text
    assert "D1_URL" not in text and "d1-ft-joint-tac" in text
    assert no_secrets(text)                       # no secrets
    sh = shutil.which("sh")
    if sh:
        assert subprocess.run([sh, "-n", str(path)], capture_output=True).returncode == 0


@needs_docker_files
def test_entrypoint_test_subset_exists_and_is_in_the_zip_script():
    text = (DOCKER / "entrypoint.sh").read_text()
    files = re.findall(r"tests/(test_\w+\.py)", text)
    assert "test_docker_packaging.py" in files
    for f in files:
        assert (ROOT / "tests" / f).is_file(), f
    maker = (ROOT / "scripts" / "make_docker_submission.sh").read_text()
    for f in files:
        assert f"tests/{f}" in maker, f"{f} would not be in the zip"


@needs_docker_files
def test_compose_file_shape():
    yaml = pytest.importorskip("yaml")
    doc = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    svc = doc["services"]["clashbot"]
    assert svc["build"]["context"] == "." and svc["build"]["dockerfile"] == "docker/Dockerfile"
    assert "8765:8765" in svc["ports"]
    assert any(v.startswith("clashbot-runs:/data/runs") for v in svc["volumes"])
    assert "clashbot-runs" in doc["volumes"]
    assert "mem_limit" not in svc and "deploy" not in svc
    assert svc["command"] == ["demo"]
    assert "CLASHBOT_ALLOW_LIVE" not in str(svc.get("environment", {}))


@needs_docker_files
def test_make_docker_submission_script():
    path = ROOT / "scripts" / "make_docker_submission.sh"
    text = path.read_text()
    assert "git archive" in text and "slim-db" in text and "scripts/setup_mac.sh" in text
    assert "--with-d1" not in text and "d1-fp16" not in text and "model.safetensors" not in text   # the zip never holds d1
    assert no_secrets(text)
    assert "knowledge_base/states" in text                                           # the pro-state markdown stays out
    sh = shutil.which("sh")
    if sh:
        assert subprocess.run([sh, "-n", str(path)], capture_output=True).returncode == 0


@needs_docker_files
def test_setup_mac_script_is_safe_and_pinned_to_the_same_d1_revision():
    path = ROOT / "scripts" / "setup_mac.sh"
    text = path.read_text()
    assert D1_REPO in text and D1_REV in text
    assert D1_REV in (DOCKER / "Dockerfile").read_text()
    assert "requirements-mac.txt" in text and "check_env.py" in text and "mirror_calib" in text
    assert no_secrets(text)
    # the script never switches live taps on: the live command only appears in the text it prints, after `cat <<EOF`
    head, _, printed = text.partition("cat <<EOF")
    code = "\n".join(line for line in head.splitlines() if not line.lstrip().startswith("#"))
    assert "CLASHBOT_ALLOW_LIVE" not in code and "--live" not in code
    assert "CLASHBOT_ALLOW_LIVE=1 scripts/demo.sh --live --dashboard" in printed
    assert "Screen Recording" in printed and "Accessibility" in printed
    sh = shutil.which("sh")
    if sh:
        assert subprocess.run([sh, "-n", str(path)], capture_output=True).returncode == 0
        bad = subprocess.run([sh, str(path), "--bogus"], capture_output=True, text=True)
        assert bad.returncode == 2 and "unknown option" in bad.stderr


# ---------------------------------------------------------------------------------------- portability on Linux
BLOCKED = ("mlx", "mlx.core", "mlx_lm", "mlx_vlm", "Quartz", "mss", "objc", "AppKit", "Foundation")


def run_py(code: str, tmp_path: Path) -> subprocess.CompletedProcess:
    block = "import sys\nfor _n in %r:\n    sys.modules[_n] = None\n" % (BLOCKED,)       # None = import raises ImportError
    return subprocess.run([sys.executable, "-c", block + code], cwd=tmp_path, capture_output=True, text=True, timeout=120,
                          env={"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin", "HF_HUB_OFFLINE": "1"})


def test_video_path_and_dashboard_import_without_mac_packages(tmp_path):
    proc = run_py("import clashbot.live, clashbot.viz.__main__, clashbot.viz.server, clashbot.executor, clashbot.ranker, "
                  "clashbot.act_model, cr_perception.sources, cr_perception.detect\n"
                  "from cr_perception.sources import VideoFrameSource, find_mirroring_window, list_windows\n"
                  "assert list_windows.__module__\n"
                  "print('ok')", tmp_path)
    assert proc.returncode == 0, proc.stderr[-800:]
    assert proc.stdout.strip().endswith("ok")


def test_mirroring_and_live_refuse_on_linux_with_a_clear_message(tmp_path):
    code = ("import sys\n"
            "from clashbot import live\n"
            "sys.platform = 'linux'\n"
            "deck = 'a,b,c,d,e,f,g,h'\n"
            "for extra in (['--source', 'mirroring'], ['--live', '--source', 'mirroring']):\n"
            "    rc = live.main(['--calib', 'x.json', '--deck', deck] + extra)\n"
            "    print('RC', rc)\n"
            "from cr_perception.sources import MirroringSource\n"
            "try:\n"
            "    MirroringSource(check_permission=False)\n"
            "except RuntimeError as exc:\n"
            "    print('SRC', exc)\n")
    proc = run_py(code, tmp_path)
    assert proc.returncode == 0, proc.stderr[-800:]
    assert proc.stdout.count("RC 2") == 2
    assert "need macOS" in proc.stderr and "--source video" in proc.stderr
    assert "SRC iPhone Mirroring window not found" in proc.stdout           # no window to find off macOS


def test_viz_host_flag_defaults_to_loopback():
    from clashbot.viz.__main__ import build_parser
    assert build_parser().parse_args([]).host == "127.0.0.1"
    args = build_parser().parse_args(["--follow", "/data/runs", "--host", "0.0.0.0", "--port", "8765"])
    assert (args.host, args.port, args.follow) == ("0.0.0.0", 8765, "/data/runs")


def test_viz_server_binds_the_requested_host(tmp_path):
    from clashbot.viz.data import Reference, Store
    from clashbot.viz.server import make_server
    server = make_server(Store(follow=tmp_path), port=0, host="127.0.0.1", reference=Reference(db_path=tmp_path / "no"), cards={})
    try:
        assert server.server_address[0] == "127.0.0.1"
    finally:
        server.server_close()


# ---------------------------------------------------------------------------------------- artefact builders
@pytest.fixture(scope="module")
def artifacts():
    """scripts/docker_artifacts.py as a module (scripts/ is not a package, and is not in the image)."""
    path = ROOT / "scripts" / "docker_artifacts.py"
    if not path.exists():
        pytest.skip("scripts/ is not part of this tree")
    spec = importlib.util.spec_from_file_location("docker_artifacts", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_states_db(path: Path) -> None:
    conn = sqlite3.connect(str(path))
    conn.executescript("""
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
        CREATE TABLE games (game_id TEXT PRIMARY KEY, own_deck TEXT);
        CREATE TABLE provenance (game_id TEXT PRIMARY KEY, video_id TEXT);
        CREATE TABLE states (vec_row INTEGER PRIMARY KEY, id TEXT UNIQUE NOT NULL, game_id TEXT, state_text TEXT,
                             context_text TEXT, record TEXT);
        CREATE VIRTUAL TABLE states_fts USING fts5(text);
        CREATE TABLE kg_edges (edge_id INTEGER PRIMARY KEY, src TEXT, rel TEXT, dst TEXT, props TEXT);
        CREATE TABLE documents (doc_id TEXT PRIMARY KEY, text TEXT);
        CREATE VIRTUAL TABLE documents_fts USING fts5(text);
    """)
    for i in range(20):
        conn.execute("INSERT INTO states VALUES (?, ?, 'g', 'state', 'context', ?)", (i, f"s{i}", "x" * 5000))
        conn.execute("INSERT INTO states_fts(rowid, text) VALUES (?, 'knight bridge')", (i,))
        conn.execute("INSERT INTO documents VALUES (?, 'doc')", (f"d{i}",))
        conn.execute("INSERT INTO documents_fts(rowid, text) VALUES (?, 'doc text')", (i,))
    conn.execute("INSERT INTO kg_edges VALUES (1, 'a', 'COUNTERS', 'b', '{}')")
    conn.execute("INSERT INTO games VALUES ('g', '[]')")
    conn.execute("INSERT INTO provenance VALUES ('g', 'v')")
    conn.commit()
    conn.close()


def test_slim_db_keeps_what_the_runtime_reads(tmp_path, artifacts):
    slim_db = artifacts.slim_db
    src, dst = tmp_path / "full.sqlite", tmp_path / "slim.sqlite"
    make_states_db(src)
    before = src.read_bytes()
    info = slim_db(src, dst)
    assert src.read_bytes() == before                                              # the source is untouched
    assert info["bytes"] < src.stat().st_size and info["rows"]["states"] == 20
    ro = sqlite3.connect(f"file:{dst}?mode=ro", uri=True)
    names = {r[0] for r in ro.execute("SELECT name FROM sqlite_master")}
    assert {"states", "states_fts", "kg_edges", "games", "provenance", "meta"} <= names
    assert "documents" not in names and "documents_fts" not in names
    assert ro.execute("SELECT COUNT(*) FROM states WHERE record IS NOT NULL").fetchone()[0] == 0
    assert ro.execute("SELECT context_text FROM states WHERE id = 's3'").fetchone()[0] == "context"
    assert ro.execute("SELECT COUNT(*) FROM states_fts WHERE states_fts MATCH 'knight'").fetchone()[0] == 20
    ro.close()


def test_cut_clip_keeps_the_frame_size(tmp_path, artifacts):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    cut_clip = artifacts.cut_clip
    src = tmp_path / "src.mp4"
    writer = cv2.VideoWriter(str(src), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 96))
    if not writer.isOpened():
        pytest.skip("OpenCV cannot write mp4 here")
    for i in range(40):
        writer.write(np.full((96, 64, 3), i * 5, np.uint8))
    writer.release()
    info = cut_clip(src, tmp_path / "clip.mp4", start=1.0, seconds=2.0)
    assert info["size"] == [64, 96] and 15 <= info["frames"] <= 20
    cap = cv2.VideoCapture(str(tmp_path / "clip.mp4"))
    assert cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == 64
    cap.release()

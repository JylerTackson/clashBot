"""MAC-AW part P: the prior games in the grader build. scripts/pack_prior_runs.py picks the runs and keeps what the dashboard reads
(screenshots re-encoded smaller, decision data untouched); the dashboard lists runs from several parents (`--parents`,
CLASHBOT_VIZ_PARENTS) with a readable label; the Dockerfile, make_docker_submission.sh and setup_mac.sh carry the pack. Fixture
runs only (no models, no network); the real-data checks are skipped when runs/ is missing."""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clashbot.viz import __main__ as viz_main  # noqa: E402
from clashbot.viz.data import Reference, Store  # noqa: E402
from clashbot.viz.server import Api  # noqa: E402

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

PACKER = ROOT / "scripts" / "pack_prior_runs.py"
needs_packer = pytest.mark.skipif(not PACKER.exists(), reason="scripts/ is not part of this tree")
needs_docker_files = pytest.mark.skipif(not (ROOT / "docker" / "Dockerfile").exists(), reason="docker/ files are not part of this tree")
REAL_RUNS = ROOT / "runs"
GROUPS = ("poc", "demo", "vs_pro", "smoke")


@pytest.fixture(scope="module")
def packer():
    spec = importlib.util.spec_from_file_location("pack_prior_runs", PACKER)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pack_prior_runs"] = mod            # dataclasses look their module up
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------------------- fixture runs
def _jpeg(path: Path, w: int = 100, h: int = 220, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    img[: h // 2] = (img[: h // 2] // 4 + 90)         # a flat-ish half, so the JPEG is not pure noise
    assert cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])


def _row(i: int, t: float) -> dict:
    return {"t": t, "idx": 100 + i, "key": "zap|left|mid", "tile": [3.5, 9.0], "executor": {"status": "dry_run", "sent": False},
            "timings_ms": {"total": 100.0}, "mode": "defend", "reason": "x", "context_text": "[double_elixir|1:40] hand: zap",
            "candidates": [{"key": "zap|left|mid", "score": 1.0, "sources": {"knn": 1.0}}], "probs": [1.0],
            "retrieval": [{"id": "v-m0.0#1.0", "game_id": "v-m0.0", "score": 0.1}]}


def make_run(runs: Path, rel: str, name: str, decisions: int = 3, shots: int = 3, live: bool = True, extra: dict | None = None,
             battles: int = 0, pro: bool = False) -> Path:
    """A run folder the way the live loop writes it, with the files the pack drops (frames, a clip, sheets, battle shots)."""
    d = runs / rel / name
    (d / "shots" / "sheets").mkdir(parents=True)
    (d / "battles").mkdir()
    (d / "config.json").write_text(json.dumps({"live": live, "dry_run": not live, "deck": "zap,knight,giant,bats", "hero_cards": "knight",
                                               "calib": "runs/poc/mirror_calib.json", **(extra or {})}))
    rows = [_row(i, 10.0 + 2.5 * i) for i in range(decisions)]
    (d / "decisions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (d / "holds.jsonl").write_text(json.dumps({"t": 9.0, "mode": "hold", "act": False, "reason": "fast wait p=0.3 h=2s"}) + "\n")
    (d / "frames.jsonl").write_text('{"t": 0.0}\n' * 50)
    (d / "summary.json").write_text(json.dumps({"frames": 50, "decisions": decisions}))
    (d / "clip.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"0" * 4000)
    index = []
    for i in range(shots):
        tag = "abc"[i % 3]
        f = f"d{100 + i // 3:04d}_t{10.0 + 2.5 * (i // 3):.1f}_{tag}.jpg"
        _jpeg(d / "shots" / f, seed=i)
        index.append({"idx": 100 + i // 3, "files": [f]})
    (d / "shots" / "index.jsonl").write_text("".join(json.dumps(r) + "\n" for r in index))
    (d / "shots" / "sheet_001.png").write_bytes(b"\x89PNG" + b"0" * 3000)
    (d / "shots" / "review.md").write_text("# review\n")
    (d / "shots" / "sheets" / "sheet_002.png").write_bytes(b"\x89PNG" + b"0" * 3000)
    _jpeg(d / "battles" / "1_end_0.jpg")
    if battles:
        (d / "battles.jsonl").write_text("".join(json.dumps({"n": k + 1, "verdict": "TRAINING_CAMP"}) + "\n" for k in range(battles)))
    if pro:
        (d / "pro_plays.jsonl").write_text(json.dumps({"t": 10.5, "card": "zap", "tile": [3, 9], "lane": "left", "zone": "mid"}) + "\n")
        (d / "vs_pro_meta.json").write_text(json.dumps({"game_id": name, "result": "win"}))
    return d


@pytest.fixture
def runs(tmp_path):
    r = tmp_path / "runs"
    make_run(r, "poc", "20261107-013441", decisions=4, shots=6, battles=0)          # a live game (the first win)
    make_run(r, "poc", "20261105-205720", decisions=0, shots=0)                       # empty session
    make_run(r, "poc", "mirror-dryrun-1", decisions=5, shots=0, live=False)            # not a dated live session
    (r / "poc" / "mirror_calib.json").write_text("{}")                                  # a plain file in the parent
    make_run(r, "demo", "20261107-213709", decisions=5, shots=9, battles=11)           # the trial
    make_run(r, "demo", "20261108-214546", decisions=0, shots=0)                       # empty
    (r / "demo" / "20261108-222319").mkdir()                                            # a folder with nothing in it
    make_run(r, "eval/vs_pro", "6qYQNJ1Uaeg-m0.3", decisions=3, shots=6, live=False, pro=True)
    make_run(r, "eval/vs_pro", "Z8-4VhLjrGU-m1.0", decisions=2, shots=3, live=False, pro=True)
    make_run(r, "eval/vs_pro", "Zt6Onn89EVk-m0.1", decisions=0, shots=0, live=False, pro=True)    # no decisions
    make_run(r, "smoke", "AO_demo", decisions=3, shots=3, live=False)
    make_run(r, "smoke", "AM_merged", decisions=3, shots=3, live=False)
    make_run(r, "smoke", "AL_dry", decisions=3, shots=3, live=False)                    # a smoke run that is not kept
    (r / "smoke" / "E1").mkdir()
    (r / "smoke" / "E1" / "bench.txt").write_text("benchmark")
    return r


def tree(path: Path) -> dict[str, tuple[int, int]]:
    out = {}
    for dp, _dn, fn in os.walk(path):
        for f in fn:
            p = Path(dp) / f
            st = p.stat()
            out[str(p.relative_to(path))] = (st.st_size, st.st_mtime_ns)
    return out


def listing(path: Path) -> set[str]:
    return {str(p.relative_to(path)) for p in path.rglob("*") if p.is_file()}


# ---------------------------------------------------------------------------------------- which runs
@needs_packer
def test_select_runs_picks_live_games_vs_pro_and_two_smoke_runs(packer, runs):
    picks, left = packer.select_runs(runs)
    assert [p.key for p in picks] == ["poc/20261107-013441", "demo/20261107-213709", "vs_pro/6qYQNJ1Uaeg-m0.3",
                                      "vs_pro/Z8-4VhLjrGU-m1.0", "smoke/AM_merged", "smoke/AO_demo"]
    why = dict(left)
    assert "no decisions" in why["poc/20261105-205720"]                    # empty session
    assert "not a dated live session" in why["poc/mirror-dryrun-1"]
    assert "no decisions.jsonl" in why["demo/20261108-222319"]
    assert "no decisions" in why["demo/20261108-214546"]
    assert "no decisions" in why["eval/vs_pro/Zt6Onn89EVk-m0.1"]
    assert "only AO_demo and AM_merged" in why["smoke/AL_dry"] and "smoke/E1" in why
    assert "poc/mirror_calib.json" not in why                               # plain files are not runs
    assert [p.decisions for p in picks][:2] == [4, 5]


@needs_packer
def test_labels_are_readable(packer, runs):
    assert packer.label_for("poc", "20261107-013441") == "Live 2026-11-07 01:34 (first win)"
    assert packer.label_for("demo", "20261107-213709") == "Live trial 2026-11-07 (10 battles)"
    assert packer.label_for("vs_pro", "Z8-4VhLjrGU-m1.0") == "Bot vs pro: Z8-4VhLjrGU m1.0"        # ids can hold a dash
    assert packer.label_for("vs_pro", "6qYQNJ1Uaeg-m0.3") == "Bot vs pro: 6qYQNJ1Uaeg m0.3"
    assert packer.label_for("smoke", "AO_demo") == "Dry run on the test clip (AO demo)"
    assert packer.label_for("poc", "20261106-133529", runs / "poc" / "20261107-013441") == "Live 2026-11-06 13:35"
    assert packer.label_for("poc", "20261106-133529", runs / "poc" / "mirror-dryrun-1") == "Mirroring dry run 2026-11-06 13:35"
    other = make_run(runs, "demo", "20261109-135341", battles=4)
    assert packer.label_for("demo", "20261109-135341", other) == "Live session 2026-11-09 13:53 (4 battles)"


# ---------------------------------------------------------------------------------------- what each run keeps
@needs_packer
def test_file_filter_keeps_the_dashboards_files_and_drops_the_rest(packer, runs, tmp_path):
    out = tmp_path / "pack"
    manifest = packer.pack_all(runs, out, packer.Plan())
    run = out / "poc" / "20261107-013441"
    assert listing(run) == {"config.json", "summary.json", "decisions.jsonl", "holds.jsonl", "label.txt", "shots/index.jsonl",
                            *(f"shots/{p.name}" for p in (runs / "poc" / "20261107-013441" / "shots").glob("d*.jpg"))}
    vs = out / "vs_pro" / "6qYQNJ1Uaeg-m0.3"
    assert {"pro_plays.jsonl", "vs_pro_meta.json"} <= listing(vs)
    names = {p.name for p in out.rglob("*") if p.is_file()}
    assert not names & {"frames.jsonl", "clip.mp4", "sheet_001.png", "sheet_002.png", "review.md", "1_end_0.jpg", "bench.txt"}
    assert not any(p.suffix in (".mp4", ".png") for p in out.rglob("*"))
    assert sorted(g for g in os.listdir(out) if g != "MANIFEST.json") == ["demo", "poc", "smoke", "vs_pro"]
    assert (run / "label.txt").read_text().strip() == "Live 2026-11-07 01:34 (first win)"
    assert (run / "decisions.jsonl").stat().st_mode & 0o222 == 0                      # written read-only
    # the data files are copied byte for byte (decision data is never rewritten, config.json included)
    for e in manifest["runs"]:
        src = runs / ("eval/vs_pro" if e["group"] == "vs_pro" else e["group"]) / e["name"]
        for f in ("decisions.jsonl", "holds.jsonl", "config.json", "summary.json"):
            assert (src / f).read_bytes() == (out / e["group"] / e["name"] / f).read_bytes()
    entry = next(e for e in manifest["runs"] if e["name"] == "20261107-013441")
    assert entry["before_bytes"] > entry["after_bytes"] > 0 and "frames.jsonl" in entry["dropped_bytes"]
    assert manifest["groups"]["poc"]["runs"] == 1 and manifest["groups"]["vs_pro"]["runs"] == 2


@needs_packer
def test_jpeg_reencode_keeps_the_count_the_names_and_shrinks(packer, runs, tmp_path):
    out = tmp_path / "pack"
    manifest = packer.pack_all(runs, out, packer.Plan(scale=0.6, quality=70))
    for e in manifest["runs"]:
        src = runs / ("eval/vs_pro" if e["group"] == "vs_pro" else e["group"]) / e["name"] / "shots"
        dst = out / e["group"] / e["name"] / "shots"
        a, b = sorted(p.name for p in src.glob("d*.jpg")), sorted(p.name for p in dst.glob("d*.jpg")) if dst.exists() else []
        assert a == b and e["shots"]["in"] == e["shots"]["out"] == len(a)
        for name in a:
            small, big = cv2.imread(str(dst / name)), cv2.imread(str(src / name))
            assert small is not None and small.shape[:2] == (round(big.shape[0] * 0.6), round(big.shape[1] * 0.6))
            assert (dst / name).stat().st_size < (src / name).stat().st_size
            assert (dst / name).stat().st_mtime_ns == (src / name).stat().st_mtime_ns        # the file time stays
    # an unreadable image is copied as it is: the number of shots never changes
    bad = runs / "poc" / "20261107-013441" / "shots" / "d0200_t99.0_a.jpg"
    bad.write_bytes(b"not a jpeg")
    out2 = tmp_path / "pack2"
    m2 = packer.pack_all(runs, out2, packer.Plan())
    entry = next(e for e in m2["runs"] if e["name"] == "20261107-013441")
    assert entry["shots"]["copied_as_is"] == 1 and entry["shots"]["out"] == entry["shots"]["in"]
    assert (out2 / "poc" / "20261107-013441" / "shots" / bad.name).read_bytes() == b"not a jpeg"


@needs_packer
def test_budget_lowers_the_vs_pro_shots_then_keeps_only_the_example_games(packer, runs, tmp_path):
    out = tmp_path / "pack"
    def quiet(*_):
        return None

    m = packer.pack_within_budget(runs, out, budget_mb=1000, log=quiet)
    assert m["plan"] == "default" and m["settings"]["vs_pro_scale"] is None
    full = m["bytes"]
    # a budget the default pass misses but the smaller vs_pro shots meet
    low = packer.pack_within_budget(runs, out, budget_mb=(full - 1) / 1e6, log=quiet)
    assert low["plan"].startswith("vs_pro shots at") and low["bytes"] < full
    assert low["settings"]["vs_pro_scale"] == packer.LOW_SCALE and low["settings"]["vs_pro_quality"] == packer.LOW_QUALITY
    # a budget nothing meets: the last resort keeps shots for the example games only, and every decisions.jsonl stays
    last = packer.pack_within_budget(runs, out, budget_mb=0.0, log=quiet)
    assert last["plan"] == "shots for the example games only"
    assert (out / "poc" / "20261107-013441" / "shots").is_dir()                 # an example game keeps its shots
    assert (out / "smoke" / "AO_demo" / "shots").is_dir() and (out / "smoke" / "AM_merged" / "shots").is_dir()
    assert not (out / "vs_pro" / "Z8-4VhLjrGU-m1.0" / "shots").exists()
    assert (out / "demo" / "20261107-213709" / "decisions.jsonl").exists() and (out / "vs_pro" / "Z8-4VhLjrGU-m1.0" / "decisions.jsonl").exists()
    entry = next(e for e in last["runs"] if e["name"] == "Z8-4VhLjrGU-m1.0")
    assert entry["shots"]["in"] == 3 and entry["shots"]["out"] == 0 and entry["shots"]["dropped"] == 3


@needs_packer
def test_the_source_is_never_modified_and_the_output_is_guarded(packer, runs, tmp_path):
    before = tree(runs)
    packer.pack_all(runs, tmp_path / "pack", packer.Plan())
    packer.pack_all(runs, tmp_path / "pack", packer.Plan())                 # a second pass replaces the first
    assert tree(runs) == before
    with pytest.raises(ValueError):
        packer.pack_all(runs, runs / "prior_runs", packer.Plan())            # inside the runs folder
    with pytest.raises(ValueError):
        packer.pack_all(runs / "poc", runs, packer.Plan())                   # contains the runs folder
    keep = tmp_path / "precious"
    keep.mkdir()
    (keep / "notes.txt").write_text("mine")
    with pytest.raises(ValueError):
        packer.pack_all(runs, keep, packer.Plan())                           # not a pack: never removed
    assert (keep / "notes.txt").read_text() == "mine"
    assert packer.main(["--runs", str(runs), "--out", str(tmp_path / "pack"), "--list"]) == 0
    assert packer.main(["--runs", str(tmp_path / "nope")]) == 2


# ---------------------------------------------------------------------------------------- the dashboard
def test_viz_parents_option_and_environment(monkeypatch, tmp_path):
    monkeypatch.delenv("CLASHBOT_VIZ_PARENTS", raising=False)
    assert viz_main.parent_paths(None) == []
    a, b, c, d = (tmp_path / n for n in "abcd")
    got = viz_main.parent_paths([f"{a},{b}", str(c)], env=f"{d}, {a},")
    assert got == [a, b, c, d]                                               # CLI first, then the environment; no duplicates
    monkeypatch.setenv("CLASHBOT_VIZ_PARENTS", f"{a},{b}")
    assert viz_main.parent_paths(None) == [a, b]
    args = viz_main.build_parser().parse_args(["--follow", "/data/runs", "--parents", f"{a},{b}", "--parents", str(c)])
    assert args.parents == [f"{a},{b}", str(c)] and viz_main.build_parser().parse_args([]).parents is None


def test_main_hands_the_extra_parents_to_the_store(monkeypatch, tmp_path):
    seen = {}

    def fake_server(store, port, host="127.0.0.1"):
        seen["store"] = store
        raise OSError("stop here")

    monkeypatch.setattr(viz_main, "make_server", fake_server)
    monkeypatch.delenv("CLASHBOT_VIZ_PARENTS", raising=False)
    live, a, b = tmp_path / "live", tmp_path / "pack" / "poc", tmp_path / "pack" / "vs_pro"
    live.mkdir()
    assert viz_main.main(["--follow", str(live), "--parents", f"{a},{b}"]) == 2
    assert seen["store"].parents == [a, b] and seen["store"].follow == live.resolve()
    monkeypatch.setenv("CLASHBOT_VIZ_PARENTS", str(b))
    assert viz_main.main(["--follow", str(live)]) == 2 and seen["store"].parents == [b]


def test_the_store_lists_runs_from_several_parents_once_each(tmp_path):
    def run(parent, name, decisions=1, label=None):
        d = tmp_path / parent / name
        d.mkdir(parents=True)
        (d / "decisions.jsonl").write_text('{"t": 1.0, "idx": 1}\n' * decisions)
        if label:
            (d / "label.txt").write_text(label + "\n")
        return d

    run("data_runs", "docker_demo_20261109-010101")
    run("pack/poc", "20261107-013441", label="Live 2026-11-07 01:34 (first win)")
    run("pack/vs_pro", "6qYQNJ1Uaeg-m0.3", label="Bot vs pro: 6qYQNJ1Uaeg m0.3")
    run("pack/poc", "20261106-233907")
    run("local_runs/poc", "20261107-013441", decisions=3)                    # the same id in the Mac's own runs/: it wins
    st = Store(follow=tmp_path / "data_runs", runs_dir=tmp_path / "local_runs", parents=[tmp_path / "pack" / "poc", tmp_path / "pack" / "vs_pro", tmp_path / "no" / "smoke"])
    rows = st.list_runs()
    ids = [r["id"] for r in rows]
    assert sorted(ids) == sorted(["data_runs:docker_demo_20261109-010101", "poc:20261107-013441", "poc:20261106-233907", "vs_pro:6qYQNJ1Uaeg-m0.3"])
    assert len(ids) == len(set(ids))
    by = {r["id"]: r for r in rows}
    assert by["poc:20261107-013441"]["decisions"] == 3 and by["poc:20261107-013441"]["label"] is None      # the first root's folder
    assert by["vs_pro:6qYQNJ1Uaeg-m0.3"]["label"] == "Bot vs pro: 6qYQNJ1Uaeg m0.3" and by["poc:20261106-233907"]["label"] is None
    assert st.resolve("vs_pro:6qYQNJ1Uaeg-m0.3") == tmp_path / "pack" / "vs_pro" / "6qYQNJ1Uaeg-m0.3"
    assert st.resolve("poc:20261106-233907") == tmp_path / "pack" / "poc" / "20261106-233907"
    assert st.resolve("smoke:nothing") is None
    assert st.current()["id"] == "data_runs:docker_demo_20261109-010101"      # the followed demo stays the selected run
    only = Store(follow=tmp_path / "data_runs", runs_dir=tmp_path / "nope")
    assert [r["id"] for r in only.list_runs()] == ["data_runs:docker_demo_20261109-010101"]      # no parents: as before


@needs_packer
def test_a_fixture_pack_loads_in_the_dashboard_api(packer, runs, tmp_path):
    pack = tmp_path / "prior_runs"
    manifest = packer.pack_all(runs, pack, packer.Plan())
    demo = tmp_path / "data_runs" / "docker_demo_20261109-120000"           # the run the container is making now
    demo.mkdir(parents=True)
    (demo / "decisions.jsonl").write_text(json.dumps(_row(0, 1.0)) + "\n")
    st = Store(follow=tmp_path / "data_runs", runs_dir=tmp_path / "none", parents=[pack / g for g in GROUPS])
    api = Api(st, Reference(db_path=tmp_path / "no.sqlite"), cards={"zap": {"name": "Zap"}})

    def get(path):
        status, ctype, body = api.handle(path)
        return status, ctype, (json.loads(body) if ctype == "application/json" else body)

    status, _, listing_ = get("/api/runs")
    rows = {r["id"]: r for r in listing_["runs"]}
    assert status == 200 and len(rows) == len(manifest["runs"]) + 1
    assert set(rows) >= {"poc:20261107-013441", "demo:20261107-213709", "vs_pro:6qYQNJ1Uaeg-m0.3", "smoke:AO_demo", "smoke:AM_merged",
                         "data_runs:docker_demo_20261109-120000"}
    assert rows["poc:20261107-013441"]["label"] == "Live 2026-11-07 01:34 (first win)"
    assert rows["demo:20261107-213709"]["label"] == "Live trial 2026-11-07 (10 battles)"
    assert rows["vs_pro:6qYQNJ1Uaeg-m0.3"]["label"] == "Bot vs pro: 6qYQNJ1Uaeg m0.3" and rows["vs_pro:6qYQNJ1Uaeg-m0.3"]["has_pro"]
    assert rows["poc:20261107-013441"]["has_shots"] and rows["poc:20261107-013441"]["decisions"] == 4 and not rows["poc:20261107-013441"]["live"]
    assert rows["poc:20261107-013441"]["has_retrieval"]
    assert not any(r["parent"] == "poc" and r["name"] == "mirror-dryrun-1" for r in rows.values())
    assert get("/api/current")[2]["id"] == "data_runs:docker_demo_20261109-120000"
    # a packed run answers every route a run does: decisions with their shots, holds, meta, the pro's plays, a screenshot
    status, _, dec = get("/api/run/poc:20261107-013441/decisions?since=0")
    assert status == 200 and dec["total"] == 4 and len(dec["rows"]) == 4
    first = dec["rows"][0]
    assert first["key"] == "zap|left|mid" and first["parsed"]["hand"] == ["zap"] and first["shots"]["a"]
    status, ctype, body = get(f"/api/shot/poc:20261107-013441/{first['shots']['a']}")
    assert status == 200 and ctype == "image/jpeg"
    img = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
    assert img is not None and img.shape[:2] == (132, 60)                    # 220 x 100 at 60%
    assert get("/api/run/poc:20261107-013441/holds?since=0")[2]["total"] == 1
    meta = get("/api/run/poc:20261107-013441/meta")[2]
    assert [c["slug"] for c in meta["deck"]] == ["zap", "knight", "giant", "bats"] and meta["has_shots"] and meta["summary"]["decisions"] == 4
    pro = get("/api/run/vs_pro:6qYQNJ1Uaeg-m0.3/pro")[2]
    assert pro["has_pro"] and pro["plays"][0]["card"] == "zap"
    assert get("/api/run/vs_pro:6qYQNJ1Uaeg-m0.3/decisions?since=0")[2]["rows"][0]["pro"]["badge"] == "same card"
    assert get("/api/run/poc:20261105-205720/meta")[0] == 404                # an empty session is not in the pack
    # decisions are identical to what the bot wrote
    assert (pack / "demo" / "20261107-213709" / "decisions.jsonl").read_bytes() == (runs / "demo" / "20261107-213709" / "decisions.jsonl").read_bytes()


# ---------------------------------------------------------------------------------------- packaging
@needs_docker_files
def test_dockerfile_copies_the_pack_read_only_before_the_code_and_points_the_dashboard_at_it():
    text = (ROOT / "docker" / "Dockerfile").read_text()
    code = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    assert "COPY prior_runs/ /app/prior_runs/" in code
    assert code.index("COPY artifacts/") < code.index("COPY prior_runs/") < code.index("COPY clashbot/")      # cache-friendly order
    env = next(line for line in code.splitlines() if line.startswith("ENV CLASHBOT_VIZ_PARENTS="))
    assert env.split("=", 1)[1].split(",") == [f"/app/prior_runs/{g}" for g in GROUPS]
    assert "CLASHBOT_ALLOW_LIVE" not in code and "--live" not in code
    entry = (ROOT / "docker" / "entrypoint.sh").read_text()
    assert "--follow" in entry and "--live" not in "\n".join(l for l in entry.splitlines() if not l.lstrip().startswith("#"))


@needs_docker_files
def test_make_docker_submission_builds_and_carries_the_pack():
    path = ROOT / "scripts" / "make_docker_submission.sh"
    text = path.read_text()
    assert "scripts/pack_prior_runs.py --runs runs --out" in text and '"$ROOT_DIR/prior_runs"' in text
    assert 'PRIOR_RUNS="${PRIOR_RUNS:-1}"' in text and "cp -Rp" in text          # on by default; file times and modes are kept
    assert "model.safetensors" not in text and "--with-d1" not in text
    sh = shutil.which("sh")
    if sh:
        assert subprocess.run([sh, "-n", str(path)], capture_output=True).returncode == 0
        helped = subprocess.run([sh, str(path), "--help"], capture_output=True, text=True)
        assert helped.returncode == 0 and "PRIOR_RUNS" in helped.stdout


@needs_docker_files
def test_setup_mac_copies_the_pack_into_runs_without_overwriting(tmp_path):
    text = (ROOT / "scripts" / "setup_mac.sh").read_text()
    start, end = text.index("PRIOR_COPIED=0"), text.index("if [ -d prior_runs ]")
    block = text[start:end]
    for pair in ("poc runs/poc", "demo runs/demo", "vs_pro runs/eval/vs_pro", "smoke runs/smoke"):
        assert f"copy_prior {pair}" in block
    sh = shutil.which("sh")
    if not sh:
        pytest.skip("no sh")
    work = tmp_path / "unzipped"
    for rel, name in (("poc", "20261107-013441"), ("demo", "20261107-213709"), ("vs_pro", "6qYQNJ1Uaeg-m0.3"), ("smoke", "AO_demo")):
        d = work / "prior_runs" / rel / name
        (d / "shots").mkdir(parents=True)
        (d / "decisions.jsonl").write_text("packed\n")
        (d / "shots" / "d0100_t10.0_a.jpg").write_bytes(b"jpg")
        os.chmod(d / "decisions.jsonl", 0o444)
    mine = work / "runs" / "poc" / "20261107-013441"                         # a full checkout's own run: left alone
    mine.mkdir(parents=True)
    (mine / "decisions.jsonl").write_text("original\n")
    script = "set -eu\n" + block + 'echo "copied $PRIOR_COPIED"\n'
    out = subprocess.run([sh, "-c", script], cwd=work, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "copied 3" in out.stdout
    assert (mine / "decisions.jsonl").read_text() == "original\n"
    assert (work / "runs" / "demo" / "20261107-213709" / "decisions.jsonl").read_text() == "packed\n"
    assert (work / "runs" / "eval" / "vs_pro" / "6qYQNJ1Uaeg-m0.3" / "shots" / "d0100_t10.0_a.jpg").read_bytes() == b"jpg"
    assert (work / "runs" / "smoke" / "AO_demo" / "decisions.jsonl").stat().st_mode & 0o200            # writable again on the Mac
    again = subprocess.run([sh, "-c", script], cwd=work, capture_output=True, text=True)
    assert "copied 0" in again.stdout                                                                 # idempotent
    # without a prior_runs/ folder (a full checkout) the block does nothing
    bare = tmp_path / "checkout"
    bare.mkdir()
    assert "copied 0" in subprocess.run([sh, "-c", script], cwd=bare, capture_output=True, text=True).stdout
    assert not (bare / "runs").exists()


@needs_docker_files
def test_docker_readme_has_the_browse_section():
    text = (ROOT / "docs" / "DOCKER.md").read_text()
    assert "## Browse every prior game" in text and "prior_runs/" in text and "python -m clashbot.viz --open" in text


# ---------------------------------------------------------------------------------------- the real runs (when present)
@needs_packer
@pytest.mark.skipif(not (REAL_RUNS / "eval" / "vs_pro").is_dir() or not (REAL_RUNS / "poc").is_dir(), reason="runs/ missing")
def test_the_real_selection_and_one_real_run(packer, tmp_path):
    picks, left = packer.select_runs(REAL_RUNS)
    keys = {p.key for p in picks}
    assert sum(k.startswith("vs_pro/") for k in keys) == 24
    assert {"poc/20261107-013441", "demo/20261107-213709", "smoke/AO_demo", "smoke/AM_merged"} <= keys
    assert not any(k.startswith("smoke/") and k not in ("smoke/AO_demo", "smoke/AM_merged") for k in keys)
    assert all(p.decisions > 0 for p in picks) and "poc/20261105-205720" not in keys
    small = next(p for p in picks if p.key == "smoke/AM_merged")
    out = tmp_path / "one"
    info = packer.pack_run(small, out / "smoke" / "AM_merged", packer.Plan())
    assert info["shots"]["in"] == info["shots"]["out"] > 0 and info["after_bytes"] < info["before_bytes"] / 2
    assert (out / "smoke" / "AM_merged" / "decisions.jsonl").read_bytes() == (small.src / "decisions.jsonl").read_bytes()

"""MAC-AO: the decision dashboard's API on a fixture run dir (no models), plus real-data checks when db/ exists."""
import http.client
import json
import os
import sqlite3
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clashbot.config import DB_PATH  # noqa: E402
from clashbot.viz import __main__ as viz_main  # noqa: E402
from clashbot.viz.data import Reference, Run, Store, Tail, bucket_reason, parse_context, percentile  # noqa: E402
from clashbot.viz.server import Api, make_server  # noqa: E402

needs_db = pytest.mark.skipif(not DB_PATH.exists(), reason="db/clashbot.sqlite missing")

CTX = ("[double_elixir|1:40] elixir 6 (opp ~4.0). hand: zap, archers, ?, giant (next knight). deck: a, b. opp deck: unknown\n"
       "field: knight(e)@[4,12] advancing left lane eta own_left 3.0s; zap(a)@[3,9] idle; bats(e)@[11,16] advancing right lane eta own_right 3.8s\n"
       "threats: knight(e) advancing left lane, tower in 3.0s; bats(e) advancing right lane, tower in 3.8s\n"
       "recent: none\ntowers own 4424/4424/?, enemy 4424/4424/?")


def _row(i, t, retrieval=True, key="zap|left|mid", status="dry_run", sent=False):
    row = {"t": t, "idx": 100 + i, "key": key, "tile": [3.5, 9.0],
           "candidates": [{"key": "zap|left|mid", "score": 1.0, "sources": {"knn": 1.0, "kg": 0.5}, "tile": None, "note": "n"},
                          {"key": "archers|left|mid", "score": 0.4, "sources": {"kg": 1.0}, "tile": None, "note": None}],
           "probs": [0.7, 0.3], "executor": {"t": t, "key": key, "status": status, "sent": sent, "clicks": []},
           "timings_ms": {"total": 100.0 + 10 * i}, "mode": "defend", "reason": "defend knight left", "tactics_lines": ["tactics: x"],
           "kg_lines": ["KG: knight"], "tile_source": "engine", "elixir": 6.0, "hand": ["zap", "archers", None, "giant"],
           "context_text": CTX}
    if retrieval:
        row["retrieval"] = [{"id": "vid-m0.0#10.0", "game_id": "vid-m0.0", "kind": "play", "action_key": "zap|left|mid",
                             "verdict": "positive", "elixir_trade": 2.0, "score": 0.0328, "ranks": {"knn": 1, "bm25": 4}}]
        row["kg_used"] = [{"threat": "knight", "card": "archers", "key": "archers|left|mid", "score": 1.0}]
    return row


def _write(path, rows, tail=""):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows) + tail)


@pytest.fixture
def run_dir(tmp_path):
    d = tmp_path / "runs" / "smoke" / "fix"
    (d / "shots").mkdir(parents=True)
    (d / "config.json").write_text(json.dumps({"deck": "zap,archers,knight,giant", "hero_cards": "knight", "evo_cards": ["zap"],
                                               "ranker": "d1", "policy": "fast", "dry_run": True}))
    _write(d / "decisions.jsonl", [_row(0, 10.0, retrieval=False), _row(1, 12.5, status="planned", sent=True), _row(2, 15.0)])
    _write(d / "holds.jsonl", [{"t": 9.0 + k, "mode": "hold", "act": False, "reason": f"fast wait p=0.{30 + k} h=2s", "elixir": 5.0,
                                "hand": ["zap"], "pushes": []} for k in range(4)])
    _write(d / "battles.jsonl", [{"n": 1, "t_start": 8.0, "t_end": 20.0, "banner": "Trainer Cheddar", "verdict": "training_camp"}])
    for name in ("d0100_t10.0_a.jpg", "d0101_t12.5_a.jpg", "d0101_t12.5_b.jpg", "d0101_t12.5_c.jpg"):
        (d / "shots" / name).write_bytes(b"\xff\xd8\xff fake jpeg")
    (d / "shots" / "index.jsonl").write_text("{}\n")
    (d / "secret.txt").write_text("nope")
    return d


@pytest.fixture
def api(run_dir):
    # a Reference over a missing db: the run endpoints never touch it
    return Api(Store(run=run_dir), Reference(db_path=run_dir / "no.sqlite"), cards={"zap": {"name": "Zap"}})


def get(api, path):
    status, ctype, body = api.handle(path)
    return status, (json.loads(body) if ctype == "application/json" else body)


def rid(run_dir):
    return Store.run_id(run_dir)


# -- helpers --------------------------------------------------------------------------------------------------------
def test_bucket_reason_replaces_digits():
    assert bucket_reason("fast wait p=0.32 h=2s") == "fast wait p=# h=#s"
    assert bucket_reason("waiting for the tesla window vs giant right: eta 20.7s") == "waiting for the tesla window vs giant right: eta #s"
    assert bucket_reason("no affordable card") == "no affordable card" and bucket_reason(None) == ""


def test_percentile():
    assert percentile([], 50) is None and percentile([5.0], 95) == 5.0
    assert percentile([1, 2, 3, 4], 50) == 2.5 and percentile([1, 2, 3, 4, 5], 95) == pytest.approx(4.8)


def test_parse_context_units_threats_and_hand():
    p = parse_context(CTX)
    assert p["enemy"] == ["knight", "bats"] and p["ally"] == ["zap"] and p["threats"] == ["knight", "bats"]
    assert p["hand"] == ["zap", "archers", "giant"]
    assert p["units"][0] == {"card": "knight", "side": "enemy", "tile": [4.0, 12.0]}
    assert parse_context(None)["enemy"] == [] and parse_context("field: none\nthreats: none")["units"] == []


# -- incremental reads ----------------------------------------------------------------------------------------------
def test_tail_waits_for_the_newline_of_a_partial_last_line(tmp_path):
    path = tmp_path / "d.jsonl"
    full = json.dumps({"a": 3})
    path.write_text(json.dumps({"a": 1}) + "\n" + json.dumps({"a": 2}) + "\n" + full[:5])
    tail = Tail(path)
    rows, total = tail.since(0)
    assert [r["a"] for r in rows] == [1, 2] and total == 2 and tail.bad == 0
    with open(path, "a") as fh:
        fh.write(full[5:] + "\n")
    rows, total = tail.since(2)
    assert [r["a"] for r in rows] == [3] and total == 3
    assert tail.since(3) == ([], 3)


def test_tail_accepts_a_complete_last_line_without_a_newline_and_skips_garbage(tmp_path):
    path = tmp_path / "d.jsonl"
    path.write_text('{"a": 1}\nnot json\n[1, 2]\n{"a": 2}')
    tail = Tail(path)
    assert [r["a"] for r in tail.poll()] == [1, 2] and tail.bad == 2
    assert Tail(tmp_path / "missing.jsonl").poll() == []


def test_tail_rereads_a_file_that_shrank(tmp_path):
    path = tmp_path / "d.jsonl"
    _write(path, [{"a": 1}, {"a": 2}])
    tail = Tail(path)
    assert len(tail.poll()) == 2
    _write(path, [{"a": 9}])
    assert [r["a"] for r in tail.poll()] == [9]


def test_decisions_endpoint_since_with_a_partial_last_line(api, run_dir):
    rid_ = rid(run_dir)
    half = json.dumps(_row(3, 18.0))
    with open(run_dir / "decisions.jsonl", "a") as fh:
        fh.write(half[:40])
    status, out = get(api, f"/api/run/{rid_}/decisions?since=0")
    assert status == 200 and out["total"] == 3 and out["next"] == 3 and [r["i"] for r in out["rows"]] == [0, 1, 2]
    with open(run_dir / "decisions.jsonl", "a") as fh:
        fh.write(half[40:] + "\n")
    status, out = get(api, f"/api/run/{rid_}/decisions?since=3")
    assert [r["i"] for r in out["rows"]] == [3] and out["total"] == 4 and out["next"] == 4
    assert get(api, f"/api/run/{rid_}/decisions?since=4")[1]["rows"] == []
    assert get(api, f"/api/run/{rid_}/decisions?since=junk")[1]["next"] == 4          # a bad `since` reads from 0
    assert [r["i"] for r in get(api, f"/api/run/{rid_}/decisions?since=1&limit=2")[1]["rows"]] == [1, 2]


def test_decision_rows_are_enriched_with_shots_and_parsed_context(api, run_dir):
    rows = get(api, f"/api/run/{rid(run_dir)}/decisions?since=0")[1]["rows"]
    assert rows[0]["shots"] == {"a": "d0100_t10.0_a.jpg", "b": None, "c": None}
    assert rows[1]["shots"] == {"a": "d0101_t12.5_a.jpg", "b": "d0101_t12.5_b.jpg", "c": "d0101_t12.5_c.jpg"}
    assert rows[2]["shots"] == {"a": None, "b": None, "c": None}
    assert rows[0]["parsed"]["enemy"] == ["knight", "bats"]
    assert "retrieval" not in rows[0] and rows[2]["retrieval"][0]["ranks"] == {"knn": 1, "bm25": 4}   # old and new format side by side
    assert rows[2]["kg_used"][0]["card"] == "archers"


def test_stats_count_taps_and_latency(api, run_dir):
    stats = get(api, f"/api/run/{rid(run_dir)}/decisions?since=0")[1]["stats"]
    assert stats == {"decisions": 3, "taps": 1, "p50_ms": 110.0, "p95_ms": 119.0}


def test_holds_endpoint_buckets_reasons(api, run_dir):
    status, out = get(api, f"/api/run/{rid(run_dir)}/holds?since=1")
    assert status == 200 and out["total"] == 4 and [r["i"] for r in out["rows"]] == [1, 2, 3]
    assert out["rows"][0]["bucket"] == "fast wait p=# h=#s" and out["buckets"] == {"fast wait p=# h=#s": 4}


def test_meta_has_config_deck_marks_battles(api, run_dir):
    status, meta = get(api, f"/api/run/{rid(run_dir)}/meta")
    assert status == 200 and meta["name"] == "fix" and meta["config"]["ranker"] == "d1" and meta["summary"] is None
    marks = {c["slug"]: (c["hero"], c["evo"]) for c in meta["deck"]}
    assert marks == {"zap": (False, True), "archers": (False, False), "knight": (True, False), "giant": (False, False)}
    assert meta["deck"][0]["name"] == "Zap" and meta["battles"][0]["banner"] == "Trainer Cheddar" and meta["has_shots"]
    (run_dir / "summary.json").write_text(json.dumps({"decisions": 3}))
    assert get(api, f"/api/run/{rid(run_dir)}/meta")[1]["summary"] == {"decisions": 3}


# -- shots ----------------------------------------------------------------------------------------------------------
def test_shot_endpoint_serves_only_jpgs_of_that_runs_shots(api, run_dir):
    rid_ = rid(run_dir)
    status, ctype, body = api.handle(f"/api/shot/{rid_}/d0100_t10.0_a.jpg")
    assert (status, ctype) == (200, "image/jpeg") and body.startswith(b"\xff\xd8")
    for bad in ("..%2Fsecret.txt", "..%2F..%2Fconfig.json", "%2Fetc%2Fpasswd", "index.jsonl", "%2e%2e%2f%2e%2e%2fsecret.jpg",
                "d0100_t10.0_a.png", ".hidden.jpg", "a%5Cb.jpg"):
        assert api.handle(f"/api/shot/{rid_}/{bad}")[0] in (400, 404), bad
    assert api.handle(f"/api/shot/{rid_}/nope.jpg")[0] == 404 and api.handle("/api/shot/smoke:zzz/x.jpg")[0] == 404
    assert api.handle("/api/shot/..:../x.jpg")[0] in (400, 404) and api.handle("/api/run/..%2F..:x/meta")[0] == 404


def test_shot_symlink_out_of_the_folder_is_refused(run_dir, tmp_path):
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(b"\xff\xd8 secret")
    os.symlink(outside, run_dir / "shots" / "link.jpg")
    assert Run("x:y", run_dir).shot_path("link.jpg") is None


# -- runs / follow --------------------------------------------------------------------------------------------------
def test_runs_listing_and_current(tmp_path, run_dir):
    runs_dir = run_dir.parent.parent
    old = runs_dir / "poc" / "20260101-000000"
    old.mkdir(parents=True)
    _write(old / "decisions.jsonl", [_row(0, 1.0, retrieval=False)])
    st = Store(run=run_dir, runs_dir=runs_dir)
    listing = Api(st, Reference(db_path=tmp_path / "no")).handle("/api/runs")
    runs = json.loads(listing[2])["runs"]
    by_id = {r["id"]: r for r in runs}
    assert set(by_id) == {"smoke:fix", "poc:20260101-000000"}
    assert by_id["smoke:fix"]["decisions"] == 3 and by_id["smoke:fix"]["has_retrieval"] is False      # first row is old format
    assert by_id["poc:20260101-000000"]["start"] == "2026-01-01T00:00:00" and by_id["smoke:fix"]["has_shots"]
    cur = st.current()
    assert cur["follow"] is False and cur["parent"] is None and cur["id"] == "smoke:fix"
    assert Store(follow=runs_dir / "poc").resolve("poc:20260101-000000").resolve() == old.resolve()


def test_follow_switches_to_the_newest_run(tmp_path):
    parent = tmp_path / "demo"
    parent.mkdir()
    st = Store(follow=parent)
    assert st.current()["id"] is None
    a = parent / "20261107-100000"
    a.mkdir()
    _write(a / "decisions.jsonl", [_row(0, 1.0)])
    assert st.current() == {"follow": True, "parent": str(parent.resolve()), "id": "demo:20261107-100000", "live": True}
    time.sleep(0.02)
    b = parent / "20261107-110000"
    b.mkdir()
    cur = st.current()
    assert cur["id"] == "demo:20261107-110000"
    api = Api(st, Reference(db_path=tmp_path / "no"))
    assert get(api, "/api/run/demo:20261107-110000/decisions?since=0")[1]["total"] == 0           # a run with no rows yet
    assert get(api, "/api/run/demo:20261107-100000/decisions?since=0")[1]["total"] == 1


def test_live_means_no_summary_and_recent_files(run_dir):
    assert Run("a:b", run_dir).is_live()
    (run_dir / "summary.json").write_text("{}")
    assert not Run("a:b", run_dir).is_live()
    (run_dir / "summary.json").unlink()
    old = time.time() - 600
    for f in run_dir.glob("*.jsonl"):
        os.utime(f, (old, old))
    assert not Run("a:b", run_dir).is_live()


def test_store_needs_exactly_one_of_run_and_follow(tmp_path):
    with pytest.raises(ValueError):
        Store()
    with pytest.raises(ValueError):
        Store(run=tmp_path, follow=tmp_path)


def test_unknown_routes_and_runs(api):
    assert api.handle("/api/nope")[0] == 404 and api.handle("/api/run/smoke:zzz/meta")[0] == 404
    assert api.handle("/api/run/bad id/meta")[0] == 404 and api.handle("/api/kg")[0] == 400
    assert api.handle("/api/state/whatever")[0] == 404                  # no db behind this Reference
    assert api.handle("/api/kg?cards=zap")[0] == 503


def test_static_files_are_served(api):
    for path, marker in (("/", b"<title>"), ("/app.js", b"use strict"), ("/style.css", b"--bg")):
        status, ctype, body = api.handle(path)
        assert status == 200 and marker in body, path
    assert "javascript" in api.handle("/app.js")[1]


def test_page_has_no_external_resources():
    static = ROOT / "clashbot" / "viz" / "static"
    for f in static.iterdir():
        text = f.read_text()
        assert "http://" not in text.replace("http://127.0.0.1", "") and "https://" not in text, f.name
        assert "@import" not in text and "cdn" not in text.lower(), f.name


# -- the real HTTP server -------------------------------------------------------------------------------------------
def test_http_server_end_to_end(run_dir, tmp_path):
    server = make_server(Store(run=run_dir), port=0, reference=Reference(db_path=tmp_path / "no"), cards={})
    assert server.server_address[0] == "127.0.0.1"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]

        def fetch(path):
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            conn.request("GET", path)                                    # raw path: no client-side normalisation
            resp = conn.getresponse()
            data = resp.read()
            conn.close()
            return resp.status, resp.getheader("Content-Type"), data

        assert fetch("/")[0] == 200 and fetch("/api/current")[0] == 200
        status, ctype, body = fetch(f"/api/run/{rid(run_dir)}/decisions?since=0")
        assert status == 200 and ctype == "application/json" and json.loads(body)["total"] == 3
        assert fetch(f"/api/shot/{rid(run_dir)}/d0101_t12.5_a.jpg")[:2] == (200, "image/jpeg")
        for bad in (f"/api/shot/{rid(run_dir)}/../secret.txt", f"/api/shot/{rid(run_dir)}/..%2Fsecret.txt",
                    f"/api/shot/{rid(run_dir)}/%2e%2e/secret.jpg", "/api/shot/smoke:fix//etc/passwd", "/static/../../etc/passwd"):
            assert fetch(bad)[0] in (400, 404), bad
        assert fetch("/nope")[0] == 404
    finally:
        server.shutdown()
        server.server_close()


def test_main_rejects_a_missing_run_and_both_modes(tmp_path, capsys):
    assert viz_main.main(["--run", str(tmp_path / "missing")]) == 2
    with pytest.raises(SystemExit):
        viz_main.main(["--run", str(tmp_path), "--follow", str(tmp_path)])


# -- real data ------------------------------------------------------------------------------------------------------
@needs_db
def test_state_endpoint_on_a_real_id():
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    sid, text = conn.execute("SELECT id, context_text FROM states WHERE action_key IS NOT NULL ORDER BY vec_row LIMIT 1").fetchone()
    conn.close()
    api = Api(Store(follow=ROOT / "runs" / "nope"))
    from urllib.parse import quote
    status, out = get(api, f"/api/state/{quote(sid, safe='')}")                # '#' must travel as %23
    assert status == 200 and out["id"] == sid and out["context_text"] == text
    assert out["action_key"] and out["video_id"] and out["video_link"].startswith("http") and out["game_id"]
    assert get(api, "/api/state/no-such-state%230.0")[0] == 404


@needs_db
def test_kg_endpoint_returns_nodes_and_edges_for_real_cards():
    api = Api(Store(follow=ROOT / "runs" / "nope"))
    status, kg = get(api, "/api/kg?cards=bats,knight,zap,tesla&own=zap,tesla,knight&enemy=bats&focus=zap&deck=zap,tesla,knight,fireball")
    assert status == 200
    ids = {n["id"]: n for n in kg["nodes"]}
    assert {"bats", "knight", "zap", "tesla"} <= set(ids)
    assert ids["bats"]["side"] == "enemy" and ids["zap"]["side"] == "own" and ids["bats"]["label"] == "Bats"
    assert kg["edges"] and all({"src", "dst", "rel", "weight"} <= set(e) for e in kg["edges"])
    assert len(kg["top_counters"]["bats"]) == 3
    extra = [n for n in kg["nodes"] if n["side"] == "other"]
    assert extra and all(n["id"] not in ("bats", "knight", "zap", "tesla") for n in extra)    # the top-3 counters beyond the cards
    # the graph's own direction: COUNTERS is stored counter -> countered, ANSWERED_BY threat -> answer
    for e in kg["edges"]:
        if e["rel"] == "COUNTERS":
            assert e["counter"] == e["src"] and e["threat"] == e["dst"]
        elif e["rel"] == "ANSWERED_BY":
            assert e["counter"] == e["dst"] and e["threat"] == e["src"]
    from clashbot.kg import KnowledgeGraph
    kgraph = KnowledgeGraph()
    ranked = sorted(kgraph.counter_scores("bats").items(), key=lambda kv: (-kv[1], kv[0]))[:3]
    assert [c["card"] for c in kg["top_counters"]["bats"]] == [c for c, _ in ranked]          # the KG's own top 3
    if kg["kg2"] is not None:
        assert "bats" in kg["kg2"] and all({"card", "score"} <= set(a) for a in kg["kg2"]["bats"])


@needs_db
def test_kg_without_enemy_param_counts_every_card_as_a_threat_and_empty_enemy_means_none():
    api = Api(Store(follow=ROOT / "runs" / "nope"))
    assert set(get(api, "/api/kg?cards=bats,zap")[1]["top_counters"]) == {"bats", "zap"}
    assert get(api, "/api/kg?cards=bats,zap&own=zap&enemy=")[1]["top_counters"] == {}


# -- MAC-AQ: a run made on a pro video shows the pro's plays ----------------------------------------------------------
@pytest.fixture
def pro_run_dir(tmp_path):
    """A run dir with pro_plays.jsonl (t in run time): decisions at 10 (zap|left|mid), 12.5, 15, 30 and 60 (a hold)."""
    d = tmp_path / "runs" / "eval" / "vs_pro" / "game-m0.0"
    (d / "shots").mkdir(parents=True)
    (d / "config.json").write_text(json.dumps({"deck": "zap,archers,knight,giant", "ranker": "d1", "policy": "fast", "source": "video"}))
    _write(d / "decisions.jsonl", [_row(0, 10.0), _row(1, 12.5, key="archers|right|mid"), _row(2, 15.0, key="knight|middle|back"),
                                   _row(3, 30.0), _row(4, 60.0, key="hold", status="hold_policy")])
    _write(d / "pro_plays.jsonl", [
        {"t": 10.8, "card": "zap", "tile": [3, 9], "lane": "left", "zone": "mid", "key": "zap|left|mid", "confidence": "high"},
        {"t": 13.0, "card": "giant", "tile": [4, 12], "lane": "left", "zone": "bridge", "key": "giant|left|bridge", "confidence": "medium"},
        {"t": 17.2, "card": "bats", "tile": [14, 4], "lane": "right", "zone": "back", "key": "bats|right|back", "confidence": "low"},
        {"t": 59.0, "card": None, "tile": None, "lane": None, "zone": None, "key": None, "confidence": "high"}])
    return d


def test_decisions_of_a_pro_run_carry_the_nearest_pro_play_and_a_badge(pro_run_dir):
    api = Api(Store(run=pro_run_dir), Reference(db_path=pro_run_dir / "no.sqlite"), cards={})
    rows = get(api, f"/api/run/{rid(pro_run_dir)}/decisions?since=0")[1]["rows"]
    p0, p1, p2, p3, p4 = (r["pro"] for r in rows)
    assert (p0["card"], p0["t"], p0["dt"], p0["badge"], p0["same_card"], p0["same_lane"]) == ("zap", 10.8, -0.8, "same card", True, True)
    assert p0["tile_dist"] == pytest.approx(0.5)                          # bot tile [3.5, 9.0] vs pro [3, 9]
    assert (p1["card"], p1["dt"], p1["badge"]) == ("giant", -0.5, "different")      # archers|right vs giant|left
    assert (p2["card"], p2["dt"], p2["badge"]) == ("giant", 2.0, "different")       # 13.0 is nearer (2.0 s) than 17.2 (2.2 s)
    assert p3 is None                                                    # nothing within 3 s of t = 30
    assert p4["badge"] == "bot held" and p4["card"] is None              # a hold decision is not judged


def test_nearest_pro_badges_and_window():
    from clashbot.viz.data import nearest_pro
    row = {"t": 20.0, "key": "zap|left|mid", "tile": [1, 1], "executor": {}}
    assert nearest_pro(row, [{"t": 23.5, "card": "zap"}]) is None
    best = nearest_pro(row, [{"t": 18.0, "card": "a"}, {"t": 20.9, "card": "zap", "lane": "left"}, {"t": 22.5, "card": "c"}])
    assert best["card"] == "zap" and best["dt"] == -0.9 and best["badge"] == "same card"
    lane = nearest_pro({"t": 5.0, "key": "knight|left|back", "tile": None, "executor": {}}, [{"t": 5.5, "card": "zap", "lane": "left", "tile": None}])
    assert lane["badge"] == "same lane" and lane["tile_dist"] is None
    unknown = nearest_pro({"t": 5.0, "key": "knight|left|back", "tile": None, "executor": {}}, [{"t": 5.5, "card": None, "lane": None}])
    assert unknown["badge"] == "pro card unknown" and not unknown["same_card"]


def test_pro_endpoint_meta_listing_and_timeline_data(pro_run_dir):
    api = Api(Store(run=pro_run_dir, runs_dir=pro_run_dir.parents[2]), Reference(db_path=pro_run_dir / "no.sqlite"), cards={})
    status, out = get(api, f"/api/run/{rid(pro_run_dir)}/pro")
    assert status == 200 and out["has_pro"] is True and [p["t"] for p in out["plays"]] == [10.8, 13.0, 17.2, 59.0]
    assert get(api, f"/api/run/{rid(pro_run_dir)}/meta")[1]["has_pro"] is True
    runs = json.loads(api.handle("/api/runs")[2])["runs"]                 # runs/eval/vs_pro is a listed parent
    assert [r["id"] for r in runs] == ["vs_pro:game-m0.0"] and runs[0]["has_pro"] is True


def test_runs_without_pro_plays_are_unchanged(api, run_dir):
    rows = get(api, f"/api/run/{rid(run_dir)}/decisions?since=0")[1]["rows"]
    assert all("pro" not in r for r in rows)
    assert get(api, f"/api/run/{rid(run_dir)}/meta")[1]["has_pro"] is False
    assert get(api, f"/api/run/{rid(run_dir)}/pro")[1] == {"has_pro": False, "plays": []}


def test_pro_plays_are_reread_when_the_file_changes(pro_run_dir):
    run = Run("vs_pro:game-m0.0", pro_run_dir)
    assert len(run.pro_plays()) == 4
    with open(pro_run_dir / "pro_plays.jsonl", "a") as fh:
        fh.write(json.dumps({"t": 5.0, "card": "zap", "confidence": "high"}) + "\n")
    plays = run.pro_plays()
    assert len(plays) == 5 and plays[0]["t"] == 5.0                       # sorted by time


def test_page_draws_the_pro_row_and_the_match_badge():
    js = (ROOT / "clashbot" / "viz" / "static" / "app.js").read_text()
    css = (ROOT / "clashbot" / "viz" / "static" / "style.css").read_text()
    assert "/pro`" in js and "data-pro" in js and "proBox" in js and "bot minus pro" in js
    assert ".match.same-card" in css and ".match.same-lane" in css and ".match.different" in css


# -- MAC-AW: the Recording panel's endpoints ------------------------------------------------------------------------
from clashbot.viz import livemeta  # noqa: E402
from clashbot.viz.server import parse_range  # noqa: E402

CLIP_BYTES = bytes(range(256)) * 40                                          # 10,240 bytes: every byte tells its own offset


@pytest.fixture
def video_run(run_dir, tmp_path):
    """The fixture run with `video` in its config.json naming a stand-in .mp4 (and a secret file next to it)."""
    media = tmp_path / "media"
    media.mkdir()
    clip = media / "clip.mp4"
    clip.write_bytes(CLIP_BYTES)
    (media / "secret.txt").write_text("not for the browser")
    cfg = json.loads((run_dir / "config.json").read_text())
    cfg.update(video=str(clip), start=0.0, stride=3, source="video", realtime=True)
    (run_dir / "config.json").write_text(json.dumps(cfg))
    return run_dir, clip


@pytest.fixture
def vapi(video_run):
    return Api(Store(run=video_run[0]), Reference(db_path=video_run[0] / "no.sqlite"), cards={"zap": {"name": "Zap"}})


def vget(api, run_dir, rng=None):
    status, ctype, body = api.handle(f"/api/run/{rid(run_dir)}/video", {"Range": rng} if rng else {})
    return status, ctype, body


def test_parse_range():
    assert parse_range(None, 100) is None and parse_range("", 100) is None
    assert parse_range("bytes=0-9", 100) == (0, 9) and parse_range("bytes=90-", 100) == (90, 99) and parse_range("bytes=-10", 100) == (90, 99)
    assert parse_range("bytes=50-5000", 100) == (50, 99) and parse_range("bytes=-500", 100) == (0, 99)
    assert parse_range("BYTES = 5 - 6", 100) == (5, 6)
    assert parse_range("bytes=100-", 100) == "unsatisfiable" and parse_range("bytes=500-600", 100) == "unsatisfiable"
    assert parse_range("bytes=-0", 100) == "unsatisfiable" and parse_range("bytes=0-0", 0) == "unsatisfiable"
    for ignored in ("bytes=9-3", "bytes=0-1,5-6", "items=0-5", "bytes=-", "bytes=x-y", "garbage"):
        assert parse_range(ignored, 100) is None, ignored                       # not a usable single range: the whole file is sent


def test_video_without_a_range_is_the_whole_file(vapi, video_run):
    status, ctype, body = vget(vapi, video_run[0])
    assert (status, ctype) == (200, "video/mp4") and body.read() == CLIP_BYTES
    assert body.headers == {"Accept-Ranges": "bytes"} and body.length == len(CLIP_BYTES)


def test_video_range_gives_206_with_the_right_bytes_and_headers(vapi, video_run):
    n = len(CLIP_BYTES)
    for rng, lo, hi in (("bytes=0-99", 0, 99), ("bytes=1000-1999", 1000, 1999), ("bytes=10000-", 10000, n - 1), ("bytes=-100", n - 100, n - 1),
                        ("bytes=10000-99999999", 10000, n - 1), ("bytes=0-0", 0, 0)):
        status, ctype, body = vget(vapi, video_run[0], rng)
        assert (status, ctype) == (206, "video/mp4"), rng
        assert body.read() == CLIP_BYTES[lo:hi + 1], rng
        assert body.headers["Content-Range"] == f"bytes {lo}-{hi}/{n}" and body.headers["Accept-Ranges"] == "bytes" and body.length == hi - lo + 1


def test_video_range_outside_the_file_is_416(vapi, video_run):
    n = len(CLIP_BYTES)
    for rng in (f"bytes={n}-", f"bytes={n + 5}-{n + 50}", "bytes=-0"):
        status, ctype, body = vget(vapi, video_run[0], rng)
        assert status == 416 and json.loads(body)["size"] == n, rng


def test_video_serves_only_the_mp4_named_in_that_runs_config(vapi, video_run, tmp_path):
    run, clip = video_run
    assert vget(vapi, run)[0] == 200
    cfg_path = run / "config.json"
    cfg = json.loads(cfg_path.read_text())
    for bad in (str(tmp_path / "media" / "secret.txt"),              # an existing file that is not an mp4
                str(tmp_path / "media" / "missing.mp4"),             # an mp4 that is not there
                "", None, 5):
        cfg["video"] = bad
        cfg_path.write_text(json.dumps(cfg))
        assert vget(vapi, run)[0] == 404, bad
    cfg.pop("video")
    cfg_path.write_text(json.dumps(cfg))
    assert vget(vapi, run)[0] == 404                                  # no video in the config: nothing to serve
    # the URL names no file: anything appended is an unknown route, a query string cannot choose the file
    assert vapi.handle(f"/api/run/{rid(run)}/video/../x", {})[0] == 404 and vapi.handle("/api/run/smoke:nope/video", {})[0] == 404
    assert vapi.handle(f"/api/run/{rid(run)}/video/clip.mp4", {})[0] == 404
    assert vapi.handle(f"/api/run/{rid(run)}/video?file=/etc/passwd", {})[0] == 404          # (still no video in the config)


def test_video_over_http_has_range_semantics(video_run, tmp_path):
    run, clip = video_run
    server = make_server(Store(run=run), port=0, reference=Reference(db_path=tmp_path / "no"), cards={})
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]

        def fetch(method, path, headers=None):
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            conn.request(method, path, headers=headers or {})
            resp = conn.getresponse()
            data = resp.read()
            heads = {k.lower(): v for k, v in resp.getheaders()}
            conn.close()
            return resp.status, heads, data

        url = f"/api/run/{rid(run)}/video"
        status, heads, data = fetch("GET", url, {"Range": "bytes=256-511"})
        assert status == 206 and data == CLIP_BYTES[256:512] and heads["content-range"] == f"bytes 256-511/{len(CLIP_BYTES)}"
        assert heads["content-length"] == "256" and heads["accept-ranges"] == "bytes" and heads["content-type"] == "video/mp4"
        status, heads, data = fetch("GET", url, {"Range": "bytes=-16"})                  # Safari asks for the end of the file first
        assert status == 206 and data == CLIP_BYTES[-16:]
        status, heads, data = fetch("GET", url)
        assert status == 200 and data == CLIP_BYTES and heads["accept-ranges"] == "bytes"
        status, heads, data = fetch("GET", url, {"Range": f"bytes={len(CLIP_BYTES)}-"})
        assert status == 416 and heads["content-range"] == f"bytes */{len(CLIP_BYTES)}"
        status, heads, data = fetch("HEAD", url)
        assert status == 200 and data == b"" and heads["content-length"] == str(len(CLIP_BYTES)) and heads["accept-ranges"] == "bytes"
        assert fetch("GET", url + "/..%2F..%2Fsecret.txt")[0] == 404
    finally:
        server.shutdown()
        server.server_close()


def lget(api, run_dir):
    status, out = get(api, f"/api/run/{rid(run_dir)}/livemeta")
    assert status == 200
    return out


def test_livemeta_passes_live_meta_json_through_with_the_servers_clock(vapi, video_run):
    run, clip = video_run
    geo = {"grid": {"cols": 18, "rows": 32}, "rect": [0.1, 0.1, 0.9, 0.8], "corners": [], "homography": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "slots": []}
    wall0 = time.time() - 10.0
    livemeta.write(run, {**livemeta.build(clip, 0.0, wall0, 30.0, 3, True, None, frame_size=(596, 1280), first_idx=0), "arena": geo})
    (run / "frames.jsonl").write_text("".join(json.dumps({"t": t, "idx": int(t * 30)}) + "\n" for t in (5.0, 6.5, 7.5)))
    out = lget(vapi, run)
    assert out["has_video"] is True and out["video_name"] == "clip.mp4" and out["derived"] is False
    assert out["wall_start"] == wall0 and out["video"] == str(clip) and out["video_t0"] == 0.0 and out["fps"] == 30.0 and out["stride"] == 3
    assert out["realtime"] is True and out["arena"] == geo and out["frame_size"] == [596, 1280]
    assert out["live"] is True and out["finished"] is False and abs(out["now"] - time.time()) < 5
    assert out["progress"]["latest_t"] == 7.5 and out["progress"]["lag_s"] == pytest.approx(2.5, abs=0.5)      # playhead ~10 s, agent at 7.5 s
    (run / "summary.json").write_text("{}")
    out = lget(vapi, run)
    assert out["finished"] is True and out["live"] is False and out["progress"]["lag_s"] is None


def test_livemeta_is_derived_from_the_config_for_a_run_without_the_file(vapi, video_run):
    run, clip = video_run
    out = lget(vapi, run)
    assert out["derived"] is True and out["has_video"] is True and out["wall_start"] is None and out["video"] == str(clip)
    assert out["stride"] == 3 and out["video_t0"] == 0.0 and out["progress"]["latest_t"] is None


def test_livemeta_of_a_run_without_a_video_says_so(api, run_dir):
    out = get(api, f"/api/run/{rid(run_dir)}/livemeta")[1]
    assert out["has_video"] is False and out["video_name"] is None and out["derived"] is False and "wall_start" not in out
    assert get(api, "/api/run/smoke:zzz/livemeta")[0] == 404


def test_livemeta_tolerates_a_broken_file(vapi, video_run):
    run, _clip = video_run
    (run / "live_meta.json").write_text("{not json")
    assert lget(vapi, run)["derived"] is True                       # an unreadable file counts as absent
    (run / "live_meta.json").write_text("[1, 2]")
    assert lget(vapi, run)["derived"] is True


def test_screenshot_player_frames_are_ordered_by_time(tmp_path):
    d = tmp_path / "runs" / "poc" / "20261101-120000"
    (d / "shots").mkdir(parents=True)
    (d / "config.json").write_text(json.dumps({"deck": "zap,archers,knight,giant", "source": "mirroring"}))
    # the rows are not in time order, and two share a time: the player must still step through game time
    rows = [_row(0, 30.0, retrieval=False), _row(1, 12.5, retrieval=False), _row(2, 12.5, retrieval=False, key="hold", status="hold_policy"),
            _row(3, 50.0, retrieval=False), _row(4, 41.0, retrieval=False)]
    for k, r in enumerate(rows):
        r["idx"] = 10 + k
    _write(d / "decisions.jsonl", rows)
    for name in ("d0010_t30.0_a.jpg", "d0011_t12.5_a.jpg", "d0011_t12.5_b.jpg", "d0012_t12.5_a.jpg", "d0014_t41.0_a.jpg"):       # row 3 (t=50) has no shot
        (d / "shots" / name).write_bytes(b"\xff\xd8\xff")
    api = Api(Store(run=d), Reference(db_path=d / "no.sqlite"), cards={})
    status, out = get(api, f"/api/run/{rid(d)}/player")
    assert status == 200 and out["total"] == 5
    fr = out["frames"]
    assert [(f["t"], f["i"]) for f in fr] == [(12.5, 1), (12.5, 2), (30.0, 0), (41.0, 4)]          # by time, then by decision index
    assert [f["file"] for f in fr] == ["d0011_t12.5_a.jpg", "d0012_t12.5_a.jpg", "d0010_t30.0_a.jpg", "d0014_t41.0_a.jpg"]
    assert [f["until"] for f in fr] == [12.5, 30.0, 41.0, None]                                      # the player holds a frame until the next one
    assert fr[1]["key"] == "hold" and fr[1]["status"] == "hold_policy" and fr[0]["mode"] == "defend" and fr[0]["tile"] == [3.5, 9.0]
    # the files the player asks for are served by the shot endpoint
    assert api.handle(f"/api/shot/{rid(d)}/{fr[0]['file']}")[0] == 200


def test_screenshot_player_of_a_run_without_shots_is_empty(api, run_dir):
    (run_dir / "shots" / "d0100_t10.0_a.jpg").unlink()
    (run_dir / "shots" / "d0101_t12.5_a.jpg").unlink()
    out = get(api, f"/api/run/{rid(run_dir)}/player")[1]
    assert out["frames"] == [] and out["total"] == 3


def test_page_has_the_recording_panel(run_dir):
    static = ROOT / "clashbot" / "viz" / "static"
    html, js, css, app = ((static / n).read_text() for n in ("index.html", "recording.js", "style.css", "app.js"))
    assert 'id="rec-panel"' in html and "recording.js" in html and "<video" in html and "<canvas" in html and 'id="grid"' in html
    assert "LIVE — the agent is watching this recording now" in js and "agent lag" in js
    assert "/livemeta`" in js and "/video`" in js and "/player`" in js and "data-speed" in js and "[1, 2, 3, 4]" in js
    assert "homography" in js and "tile[0] + 0.5" in js and "currentTime" in js and "pro: " in js
    assert "Rec.poll" in app and "Rec.render" in app and "Rec.onSelect" in app
    assert ".rec-banner" in css and ".rec-stage" in css and ".rec-feed" in css
    assert "http://" not in js and "https://" not in js and "eval(" not in js
    status, ctype, body = Api(Store(run=run_dir)).handle("/recording.js")
    assert status == 200 and "javascript" in ctype and b"Rec" in body

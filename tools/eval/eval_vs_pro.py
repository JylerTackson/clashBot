"""MAC-AQ: the bot against the pros on their own videos, open loop (D-052).

  python -m tools.eval.eval_vs_pro --split test [--games N] [--stride 3] [--out runs/eval/vs_pro]
      run the full live stack (`clashbot.live.build_stack` + `LiveLoop`, VideoFrameSource, DryRunExecutor, the exact live
      flags) over the video window of every selected game of the split, one game at a time, each under the heavy lock and
      paused while the HOLD_HEAVY flag exists. Resumable: a game whose run dir has `summary.json` is skipped.
  python -m tools.eval.eval_vs_pro --score [--out runs/eval/vs_pro] [--docs docs/data/vs_pro_v1]
      match the bot's taps with the pro's plays by time and write `<docs>.json` + `<docs>.md` (no model needed).

Time mapping: `dataset/v1/states.jsonl` `t` is VIDEO seconds from the start of the mp4 (`provenance.jsonl` `video_time` is
the game's window in the same clock). A `play` / `key` / `periodic` row is the state at (event - 1.0 s); the pro's play is
at `t + action.delay_s` (the same instants appear as `following_plays[].t`). `VideoFrameSource` stamps frames with
`idx / fps` (video seconds), so the run's `t` equals the dataset's `t` and no offset is applied.

The bot's taps never change the video (open loop): this measures what the bot would do in the pro's situation. Nothing in
this file sends input: the executor is the `DryRunExecutor` and `--live` is never passed.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import shutil
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from clashbot.actions import is_ability_key  # noqa: E402
from clashbot.config import DATASET_DIR, DECKS_DIR, HOLD_KEY  # noqa: E402
from clashbot.game_rules import lane_of, load_cards, parse_action_key, zone_of  # noqa: E402

SCRATCH = Path("/private/tmp/claude-501/-Users-tylerjackson-Projects-clashbot/b32b0bc7-ec70-4c37-81b7-3ec6b49dd77b/scratchpad")
LOCK_FILE = SCRATCH / "heavy.lock"
HOLD_FILE = SCRATCH / "HOLD_HEAVY"
LOCKF = "/usr/bin/lockf"
EX_HELD = 75                              # the child found HOLD_HEAVY after it got the lock: the parent waits and retries
POLL_S = 60.0

FROZEN_COMMIT = "66917e7"                 # D-051: decision code frozen here
DECISION_PATHS = ("clashbot", "cr_perception")
NOT_DECISION = ("clashbot/viz/",)
MIRROR_CALIB = "runs/poc/mirror_calib.json"
RETRIEVAL_SPLITS = ("train", "val")       # the live index holds every split: on test games it would hold the very state the pro played
OUT_DEFAULT = "runs/eval/vs_pro"
DOCS_DEFAULT = "docs/data/vs_pro_v1"
# the live configuration (docs/plans/MAC-AN: `live` fixed defaults); the per-game flags are added by `game_argv`
LIVE_FLAGS = ["--source", "video", "--ranker", "d1", "--d1-backend", "transformers", "--d1-model", "db/models/d1-ft-joint-tac",
              "--policy", "fast", "--act-model", "db/models/act-gbdt", "--record-shots", "--fps", "10", "--detect-every", "3"]

PLAY_DEDUP_S = 0.1                        # two entries of the same play (row action / following_plays / rounding)
WINDOWS = (1.0, 2.0, 3.0)                 # recall / precision tolerances, seconds
MATCH_S = 2.0                             # the window of the headline numbers and of the pairs
PRO_SHIFT_S = -0.5                        # sensitivity: the dataset stamps a play when the card lands, the tap is ~0.5 s earlier
NULL_SHIFTS_S = (-60.0, -40.0, -20.0, 20.0, 40.0, 60.0)   # null baseline: the bot's taps moved by these seconds (wrapping in the game)
HAND_SIZE = 4
HOLD_LOOKBACK_S = 3.0
BOOTSTRAP_N = 2000
CONFS = ("high", "medium", "low")
STAT_MODES = ("defend", "punish", "counter", "leak", "spell", "learned")


# ---------------------------------------------------------------------------------------------------- dataset

def read_jsonl(path: Path) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:         # a partial last line of a live file
                    pass
    return out


def load_games(split: str, dataset: Path = DATASET_DIR) -> list[dict]:
    """games.jsonl of `split` joined with provenance.jsonl: game_id, video_id, own_deck, deck_status, result, start, end."""
    prov = {r["game_id"]: r for r in read_jsonl(dataset / "provenance.jsonl")}
    out = []
    for g in read_jsonl(dataset / "games.jsonl"):
        if g.get("split") != split or g["game_id"] not in prov:
            continue
        p = prov[g["game_id"]]
        t0, t1 = (float(x) for x in p["video_time"])
        out.append({"game_id": g["game_id"], "video_id": p["video_id"], "split": split, "own_deck": list(g.get("own_deck") or []),
                    "own_deck_key": g.get("own_deck_key"), "deck_status": g.get("deck_status"), "result": g.get("result"),
                    "start": t0, "end": t1, "flags": list(g.get("flags") or []), "creator": p.get("creator")})
    return out


def select_games(games: list[dict], n: int | None = None, only: list[str] | None = None) -> tuple[list[dict], list[dict]]:
    """(selected, skipped). The stack needs a complete 8-card deck, so a game whose deck is not `valid` is skipped; games
    with a known result come before `unknown` ones and the order round-robins over videos (the k-th game of every video
    before the (k+1)-th), so a run cut short still covers every video. `n` keeps the first n; `only` keeps those ids."""
    selected, skipped = [], []
    for g in games:
        if only and g["game_id"] not in only:
            continue
        if g["deck_status"] != "valid" or len(g["own_deck"]) != 8:
            skipped.append({**g, "skip_reason": f"deck {g['deck_status']} ({len(g['own_deck'])} cards): the stack needs the 8-card deck"})
        else:
            selected.append(g)
    nth: dict[str, int] = defaultdict(int)
    keyed = []
    for g in sorted(selected, key=lambda g: (g["video_id"], g["start"])):
        keyed.append((0 if g["result"] in ("win", "loss") else 1, nth[g["video_id"]], g["video_id"], g))
        nth[g["video_id"]] += 1
    selected = [k[3] for k in sorted(keyed, key=lambda k: k[:3])]
    if n is not None:
        for g in selected[n:]:
            skipped.append({**g, "skip_reason": f"beyond --games {n}"})
        selected = selected[:n]
    return selected, skipped


def read_states(game_ids: set[str], dataset: Path = DATASET_DIR) -> dict[str, list[dict]]:
    """states.jsonl rows of the given games (cheap prefilter on the text before parsing)."""
    out: dict[str, list[dict]] = defaultdict(list)
    with open(dataset / "states.jsonl") as f:
        for line in f:
            head = line[:120]
            if not any(gid in head for gid in game_ids):
                continue
            row = json.loads(line)
            if row.get("game_id") in game_ids:
                out[row["game_id"]].append(row)
    return out


def threat_cards(state: dict) -> list[str]:
    """Enemy cards of the state's `threats` lines ("bomber(e) advancing left lane, tower in 3.0s")."""
    out = []
    for line in (state.get("threats") or []):
        m = re.match(r"\s*([a-z0-9.\-]+)\(e\)", str(line))
        if m and m.group(1) not in out:
            out.append(m.group(1))
    return out


def pro_plays(rows: list[dict], own_deck: list[str], start: float | None = None, end: float | None = None) -> list[dict]:
    """The pro's plays of one game from its states.jsonl rows, in video seconds: the action of every `play` / `unknown`
    row at `t + delay_s` and every own entry of `following_plays`, de-duplicated (same instant within PLAY_DEDUP_S and the
    same card or one unknown). A row's own action wins (it has lane / zone / key and the row's state); a `following_plays`
    entry has no state of its own, so its `hand` / `threats` are None. `card_ok` = the card is in the pro's deck (a card
    outside it is a labelling error: kept for timing, excluded from the choice metrics). Sorted by t."""
    deck = set(own_deck)
    cands: list[dict] = []
    for r in rows:
        a = r.get("action") or {}
        if a.get("type") in ("play", "unknown") and a.get("delay_s") is not None:
            st = r.get("state") or {}
            own = st.get("own") or {}
            cands.append({"t": float(r["t"]) + float(a["delay_s"]), "card": a.get("card"), "tile": a.get("tile"),
                          "lane": a.get("lane"), "zone": a.get("zone"), "key": a.get("key"),
                          "confidence": a.get("confidence"), "source": "row", "state_id": r.get("id"),
                          "hand": list(own.get("hand") or []), "threats": threat_cards(st)})
        for f in a.get("following_plays") or []:
            if f.get("player") == "own" and f.get("t") is not None:
                tile = f.get("tile")
                cands.append({"t": float(f["t"]), "card": f.get("card"), "tile": tile, "lane": lane_of(tile), "zone": zone_of(tile),
                              "key": None, "confidence": f.get("confidence"), "source": "following", "state_id": r.get("id"),
                              "hand": None, "threats": None})
    cands.sort(key=lambda c: (c["source"] != "row", c["t"]))          # a row's own action first, then the following_plays
    out: list[dict] = []
    for c in cands:
        if any(abs(o["t"] - c["t"]) <= PLAY_DEDUP_S and (o["card"] == c["card"] or o["card"] is None or c["card"] is None)
               for o in out):
            continue
        out.append(c)
    out.sort(key=lambda c: c["t"])
    keep = []
    for c in out:
        if (start is not None and c["t"] < start) or (end is not None and c["t"] > end):
            continue
        key = c["key"] or (f"{c['card']}|{c['lane']}|{c['zone']}" if c["card"] and c["lane"] and c["zone"] else None)
        keep.append({**c, "t": round(c["t"], 3), "key": key, "card_ok": bool(c["card"] and c["card"] in deck)})
    return keep


def deck_variants(game: dict, rows: list[dict], decks_dir: Path = DECKS_DIR, cards: dict | None = None) -> tuple[list[str], list[str]]:
    """(evo_cards, hero_cards) of the pro's deck, as far as the dataset says: heroes from the `hero_cards` of the game's
    states, evolutions from the front matter of the KB deck page (`evolutions:` / `heroes:`); both limited to cards of the
    deck whose card index has the variant, a card never both. Missing information gives none."""
    deck = list(game["own_deck"])
    cards = cards if cards is not None else load_cards()
    heroes: list[str] = []
    evos: list[str] = []
    seen = Counter(h for r in rows for h in ((r.get("state") or {}).get("own") or {}).get("hero_cards") or [])
    heroes += [h for h, _n in seen.most_common()]
    key = game.get("own_deck_key")
    page = Path(decks_dir) / f"{key}.md" if key else None
    if page is not None and page.exists():
        text = page.read_text()
        head = text.split("---")[1] if text.startswith("---") else ""
        for field, into in (("evolutions", evos), ("heroes", heroes)):
            m = re.search(rf"^{field}:\s*(.*)$", head, re.M)
            if m:
                into += [x.strip().strip('"') for x in m.group(1).split(",") if x.strip().strip('"') not in ("", "none")]
    heroes = [h for h in dict.fromkeys(heroes) if h in deck and (cards.get(h) or {}).get("has_hero_variant")]
    evos = [e for e in dict.fromkeys(evos) if e in deck and e not in heroes and (cards.get(e) or {}).get("has_evolution")]
    return evos, heroes


# ---------------------------------------------------------------------------------------------------- runner

def head_info() -> dict:
    """HEAD hash and the decision-code files that differ from the freeze (D-051), recorded with every run."""
    def git(*a: str) -> str:
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    changed = [p for p in git("diff", "--name-only", FROZEN_COMMIT, "--", *DECISION_PATHS).splitlines()
               if p and not p.startswith(NOT_DECISION)]
    return {"head": git("rev-parse", "HEAD"), "frozen_commit": FROZEN_COMMIT, "decision_code_changed": changed}


def resolve_calib(video_id: str, calib: str, root: Path = ROOT) -> str:
    """`auto` = the video's own calibration (`runs/videos/<video>/match_0/calib.json`, the one its dataset states were
    perceived with: a full-frame content rect and its own arena corners), else the mirroring calibration."""
    if calib != "auto":
        return calib
    own = root / "runs" / "videos" / video_id / "match_0" / "calib.json"
    return str(own.relative_to(root)) if own.exists() else MIRROR_CALIB


def game_argv(game: dict, out_dir: Path, *, stride: int, calib: str, heroes: list[str]) -> list[str]:
    """`clashbot.live` command line of one game (the live flags + the video window)."""
    vid = game["video_id"]
    span = game["end"] - game["start"]
    return [*LIVE_FLAGS, "--video", f"data/videos/{vid}/{vid}.mp4", "--start", f"{game['start']}", "--end", f"{game['end']}",
            "--stride", str(stride), "--calib", resolve_calib(vid, calib), "--deck", ",".join(game["own_deck"]),
            "--hero-cards", ",".join(heroes), "--seconds", f"{span + 10.0:.1f}", "--out", str(out_dir)]


def run_finished(run_dir: Path) -> bool:
    return (Path(run_dir) / "summary.json").is_file()


def run_game(game: dict, out_root: Path, *, stride: int, calib: str, hold_file: Path | None = HOLD_FILE) -> int:
    """Run one game (the child process, under the heavy lock). Returns 0, or EX_HELD when HOLD_HEAVY appeared while the
    lock was awaited. A previous partial run of the game is removed first."""
    if hold_file is not None and Path(hold_file).exists():
        return EX_HELD
    from clashbot import live
    from clashbot.actions import load_placement_prior
    from clashbot.recorder_shots import ShotRecorder

    out_dir = Path(out_root) / game["game_id"]
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    rows = read_states({game["game_id"]}).get(game["game_id"], [])
    plays = pro_plays(rows, game["own_deck"], game["start"], game["end"])
    (out_dir / "pro_plays.jsonl").write_text("".join(json.dumps(p) + "\n" for p in plays))
    evos, heroes = deck_variants(game, rows)
    args = live.build_parser().parse_args(game_argv(game, out_dir, stride=stride, calib=calib, heroes=heroes))
    args.dry_run = True
    args.device = live.resolve_device(args.device)
    args.evo_cards = ",".join(evos)
    t_start = time.time()
    stack = live.build_stack(args)
    (out_dir / "config.json").write_text(json.dumps({**vars(args), "retrieval_splits": list(RETRIEVAL_SPLITS)}, indent=2, default=str))
    shots = ShotRecorder(out_dir / "shots", calib=args.calib, max_decisions=args.shots_max)
    loop = live.LiveLoop(stack.source, stack.perception, stack.builder, stack.generator, stack.ranker, stack.executor, stack.cards,
                         load_placement_prior(), plan_text=stack.plan_text, out_dir=out_dir, min_gap_s=args.min_gap,
                         policy=stack.policy, tactics=stack.tactics, coach=stack.coach, shots=shots, kg2=stack.kg2,
                         filters={"splits": list(RETRIEVAL_SPLITS)})
    t_run = time.time()
    summary = loop.run(max_seconds=args.seconds)
    wall = time.time() - t_run
    meta = {"game_id": game["game_id"], "video_id": game["video_id"], "split": game["split"], "result": game["result"],
            "start": game["start"], "end": game["end"], "minutes": round((game["end"] - game["start"]) / 60.0, 3),
            "own_deck": game["own_deck"], "hero_cards": heroes, "evo_cards": evos, "calib": args.calib, "stride": stride,
            "n_pro_plays": len(plays), "load_s": stack.load_s, "wall_s": round(wall, 1),
            "total_s": round(time.time() - t_start, 1), "frames": summary["frames"], "decisions": summary["decisions"],
            "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **head_info()}
    (out_dir / "vs_pro_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"{game['game_id']}: frames={summary['frames']} decisions={summary['decisions']} wall={wall:.0f}s "
          f"({wall / max(1.0, game['end'] - game['start']):.2f}x real time)", flush=True)
    return 0


def wait_while_held(hold_file: Path, poll_s: float = POLL_S, sleep=time.sleep) -> float:
    """Block while the HOLD_HEAVY flag exists (the user is playing live); returns the seconds waited."""
    waited = 0.0
    while Path(hold_file).exists():
        sleep(poll_s)
        waited += poll_s
    return waited


def orchestrate(games: list[dict], out_root: Path, *, stride: int, calib: str, lock: Path = LOCK_FILE, hold_file: Path = HOLD_FILE,
                poll_s: float = POLL_S, retries: int = 1, runner=subprocess.run, sleep=time.sleep) -> dict:
    """Run every unfinished game, one `lockf` acquisition per game. The parent holds no model and no lock between games."""
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    done, failed, skipped = [], [], []
    t0 = time.time()
    for g in games:
        gid = g["game_id"]
        if run_finished(out_root / gid):
            skipped.append(gid)
            print(f"[skip] {gid}: finished", flush=True)
            continue
        attempts = 0
        while True:
            wait_while_held(hold_file, poll_s, sleep)
            cmd = [LOCKF, str(lock), sys.executable, "-m", "tools.eval.eval_vs_pro", "--run-game", gid, "--out", str(out_root),
                   "--stride", str(stride), "--calib", calib, "--hold-file", str(hold_file), "--split", g["split"]]
            print(f"[run] {gid} ({g['end'] - g['start']:.0f} s of video) elapsed {(time.time() - t0) / 60:.1f} min", flush=True)
            rc = runner(cmd, cwd=ROOT).returncode
            if rc == EX_HELD:
                continue                          # HOLD_HEAVY appeared while the lock was awaited: wait for it, retry
            if rc == 0 and run_finished(out_root / gid):
                done.append(gid)
                break
            attempts += 1
            print(f"[fail] {gid}: exit {rc} (attempt {attempts})", flush=True)
            if attempts > retries:
                failed.append(gid)
                break
    return {"done": done, "failed": failed, "skipped": skipped, "wall_s": round(time.time() - t0, 1)}


# ---------------------------------------------------------------------------------------------------- matching

def bot_acts(decisions: list[dict]) -> tuple[list[dict], int]:
    """(acts, n_ability): the decisions that would tap a card (executor status planned / dry_run, or none = the dry-run
    executor's plain entry); holds, gated taps and hero-ability taps are not acts (the ability is counted apart)."""
    acts, abilities = [], 0
    for d in decisions:
        ex = d.get("executor") or {}
        status = ex.get("status", "dry_run")
        key = d.get("key")
        if key == HOLD_KEY or key is None:
            continue
        if is_ability_key(key) or status == "planned_ability":
            abilities += 1
            continue
        if status not in ("planned", "dry_run"):
            continue
        try:
            p = parse_action_key(key)
        except ValueError:
            continue
        probs = d.get("probs") or []
        acts.append({"t": float(d["t"]), "idx": d.get("idx"), "key": key, "card": p["card"], "lane": p["lane"], "zone": p["zone"],
                     "tile": d.get("tile"), "mode": d.get("mode"), "reason": d.get("reason"),
                     "cands": [(c["key"], probs[i] if i < len(probs) else None) for i, c in enumerate(d.get("candidates") or [])][:5]})
    return acts, abilities


def pair_up(bot_ts: list[float], pro_ts: list[float], window: float) -> list[tuple[int, int]]:
    """One-to-one pairs (bot index, pro index) within `window` seconds: nearest first, greedy."""
    cand = sorted((abs(b - p), i, j) for i, b in enumerate(bot_ts) for j, p in enumerate(pro_ts) if abs(b - p) <= window)
    used_b: set[int] = set()
    used_p: set[int] = set()
    pairs = []
    for _d, i, j in cand:
        if i in used_b or j in used_p:
            continue
        used_b.add(i)
        used_p.add(j)
        pairs.append((i, j))
    return sorted(pairs)


def covered(xs: list[float], ys: list[float], window: float, shift: float = 0.0) -> list[bool]:
    """For each x: some y (shifted by `shift`) within `window` seconds."""
    return [any(abs(x - (y + shift)) <= window for y in ys) for x in xs]


def tile_distance(a, b) -> float | None:
    if not a or not b:
        return None
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


_REASON_RULES = [
    (r"^no affordable card", "no affordable card"),
    (r"^fast wait", "act model: wait (p below threshold)"),
    (r"^no adequate answer", "no adequate answer in hand"),
    (r"^push .* answered", "push already answered"),
    (r"^tower handles", "the tower handles it"),
    (r"^waiting for the .* window", "waiting for a card's window"),
    (r"^save for", "saving for a card"),
    (r"^leak", "leak (nothing to do)"),
]


def bucket_reason(reason: str | None) -> str:
    """A hold reason without card names and numbers: the table's row."""
    text = str(reason or "").strip()
    for pat, name in _REASON_RULES:
        if re.search(pat, text):
            return name
    return " ".join(re.sub(r"\d+(\.\d+)?", "#", text).split()[:4]) or "(empty)"


HAND_MAX_AGE_S = 3.0                      # the bot's perceived hand at a time: its latest hold / decision row no older than this


def bot_hand_at(t: float, rows: list[dict], max_age: float = HAND_MAX_AGE_S) -> list[str] | None:
    """The cards of the hand the bot perceived at video time `t`: the hand of the latest hold line or decision row at or
    before `t` (rows sorted by t, each with `hand`), at most `max_age` s old; None without one. Unread slots are dropped."""
    best = None
    for r in rows:
        if r["t"] <= t:
            best = r
        else:
            break
    if best is None or t - best["t"] > max_age:
        return None
    return [c for c in (best.get("hand") or []) if c]


def _last_before(rows: list[dict], t: float, key: str = "t") -> dict | None:
    best = None
    for r in rows:
        if r[key] <= t:
            best = r
        else:
            break
    return best


def miss_reason(t: float, holds: list[dict], decisions: list[dict], frames: list[dict]) -> str:
    """Why the bot did not tap near the pro play at video time `t`: not in a match frame; a decision that became a hold
    (policy / hand / fallback); else the hold line in force (mode and bucketed reason)."""
    fr = _last_before(frames, t)
    if fr is not None and fr.get("readiness") != "match":
        return f"not in a match frame (readiness {fr.get('readiness')})"
    if fr is None:
        return "no frame yet"
    for d in decisions:
        if abs(d["t"] - t) <= MATCH_S:
            status = (d.get("executor") or {}).get("status", "dry_run")
            if status.startswith("hold") or status == "gated" or d.get("key") == HOLD_KEY:
                return f"decision held: {status}"
            if is_ability_key(d.get("key")) or status == "planned_ability":
                return "hero ability tap instead"
    h = _last_before(holds, t)
    if h is None or t - h["t"] > HOLD_LOOKBACK_S:
        return "no hold line within 3 s (min-gap or no state)"
    if h.get("act"):
        return f"policy said act ({h.get('mode')}) but no tap followed"
    return f"{h.get('mode')}: {bucket_reason(h.get('reason'))}"


# ---------------------------------------------------------------------------------------------------- baselines

class Baselines:
    """What a play-by-the-numbers predictor would say at a pro play: a random card of the perceived hand, the placement
    prior (its most common key overall, and for the pro's card its most common lane / zone / key) and the card KG v1's
    `counter_scores` ranks first among the hand when a threat exists. `kg_scores(threat) -> {card: score}` and `prior`
    (placement_prior.json) are injected."""

    def __init__(self, prior: dict | None, kg_scores=None):
        by_key = (prior or {}).get("by_key") or {}
        self.top_key = max(by_key, key=lambda k: by_key[k]["count"]) if by_key else None
        self.by_card: dict[str, str] = {}
        best: dict[str, int] = {}
        for k, v in by_key.items():
            card = k.split("|")[0]
            if v["count"] > best.get(card, -1) or (v["count"] == best.get(card) and k < self.by_card[card]):
                best[card], self.by_card[card] = v["count"], k
        self.kg_scores = kg_scores

    def for_play(self, p: dict) -> dict:
        """{rand_hand, kg_first, prior_global_key, prior_lane, prior_zone, prior_key}; None = not applicable."""
        out = {"rand_hand": None, "kg_first": None, "prior_global_key": None, "prior_lane": None, "prior_zone": None, "prior_key": None}
        card = p.get("card")
        if not p.get("card_ok"):
            return out
        known = [c for c in (p.get("hand") or []) if c]
        if known:
            out["rand_hand"] = (1.0 / len(known)) if card in known else 0.0
        if known and p.get("threats") and self.kg_scores is not None:
            score = {c: 0.0 for c in known}
            for th in p["threats"]:
                for c, s in (self.kg_scores(th) or {}).items():
                    if c in score:
                        score[c] += s
            top = max(score.values())
            if top > 0:
                best = [c for c, s in score.items() if s == top]
                out["kg_first"] = (1.0 / len(best)) if card in best else 0.0
        if p.get("key") and self.top_key:
            out["prior_global_key"] = float(p["key"] == self.top_key)
        pk = self.by_card.get(card)
        if pk:
            _c, lane, zone = pk.split("|")
            if p.get("lane"):
                out["prior_lane"] = float(lane == p["lane"])
            if p.get("zone"):
                out["prior_zone"] = float(zone == p["zone"])
            if p.get("key"):
                out["prior_key"] = float(pk == p["key"])
        return out


# ---------------------------------------------------------------------------------------------------- scoring

def _add(counts: Counter, name: str, num: float, den: float) -> None:
    counts[f"{name}.num"] += num
    counts[f"{name}.den"] += den


def score_game(meta: dict, pro: list[dict], decisions: list[dict], holds: list[dict], frames: list[dict],
               baselines: Baselines, shots_index: dict[int, list[str]] | None = None, cards: dict | None = None) -> dict:
    """One game's record: `counts` (flat sums that pool across games), the per-pair rows, the missed plays' reasons."""
    cards = cards if cards is not None else load_cards()
    acts, n_ability = bot_acts(decisions)
    minutes = float(meta["minutes"])
    pro = sorted(pro, key=lambda p: p["t"])
    bt = [a["t"] for a in acts]
    pt = [p["t"] for p in pro]
    c: Counter = Counter()
    c["games"] = 1
    c["minutes"] = minutes
    c["n_pro"] = len(pro)
    c["n_bot"] = len(acts)
    c["n_ability"] = n_ability
    c["n_pro_card_ok"] = sum(p["card_ok"] for p in pro)
    c["n_pro_unknown_card"] = sum(p["card"] is None for p in pro)
    c["n_pro_card_not_in_deck"] = sum(p["card"] is not None and not p["card_ok"] for p in pro)
    for w in WINDOWS:
        _add(c, f"recall_{w:g}s", sum(covered(pt, bt, w)), len(pro))
        _add(c, f"precision_{w:g}s", sum(covered(bt, pt, w)), len(acts))
    _add(c, "recall_2s_pro_shift", sum(covered(pt, bt, MATCH_S, -PRO_SHIFT_S)), len(pro))     # pro tap = stamp + PRO_SHIFT_S
    _add(c, "precision_2s_pro_shift", sum(covered(bt, pt, MATCH_S, PRO_SHIFT_S)), len(acts))
    span = float(meta["end"]) - float(meta["start"])
    if span > 2 * max(NULL_SHIFTS_S, key=abs) and acts and pro:     # null: the bot's own taps at the wrong moment of the game
        rec = prec = 0.0
        for s in NULL_SHIFTS_S:
            moved = [float(meta["start"]) + ((t - float(meta["start"]) + s) % span) for t in bt]
            rec += sum(covered(pt, moved, MATCH_S)) / len(NULL_SHIFTS_S)
            prec += sum(covered(moved, pt, MATCH_S)) / len(NULL_SHIFTS_S)
        _add(c, "null_recall_2s", rec, len(pro))
        _add(c, "null_precision_2s", prec, len(acts))
    _add(c, "bot_acts_per_min", len(acts), minutes)
    _add(c, "pro_plays_per_min", len(pro), minutes)
    _add(c, "bot_elixir_per_min", sum((cards.get(a["card"]) or {}).get("elixir_cost") or 0.0 for a in acts), minutes)
    _add(c, "pro_elixir_per_min", sum((cards.get(p["card"]) or {}).get("elixir_cost") or 0.0 for p in pro if p["card_ok"]), minutes)
    pairs = pair_up(bt, pt, MATCH_S)
    matched_pro = {j for _i, j in pairs}
    covered_pro = covered(pt, bt, MATCH_S)
    cov_bot = covered(bt, pt, MATCH_S)
    hand_rows = sorted(({"t": float(r["t"]), "hand": r.get("hand")} for r in [*holds, *decisions] if r.get("hand") is not None),
                       key=lambda r: r["t"])
    base = []
    for p in pro:
        b = baselines.for_play({**p, "hand": bot_hand_at(p["t"], hand_rows)})      # the hand the bot perceived (the spec's baseline)
        ds = baselines.for_play(p)                                                  # the hand the dataset's pipeline read
        b["rand_hand_ds"], b["kg_first_ds"] = ds["rand_hand"], ds["kg_first"]
        base.append(b)
        if p["card_ok"]:
            bh = bot_hand_at(p["t"], hand_rows)
            if bh:
                _add(c, "pro_in_bot_hand", float(p["card"] in bh), 1)
            if any(p.get("hand") or []):
                _add(c, "pro_in_dataset_hand", float(p["card"] in p["hand"]), 1)
    for p, b, ok in zip(pro, base, covered_pro):                         # baselines at the pro plays (all, and bot-covered)
        for name, v in b.items():
            if v is not None:
                _add(c, f"base_all.{name}", v, 1)
                if ok:
                    _add(c, f"base_cov.{name}", v, 1)
    rows = []
    for i, j in pairs:
        a, p = acts[i], pro[j]
        dt = a["t"] - p["t"]
        known = p["card_ok"]
        same_card = known and a["card"] == p["card"]
        dist = tile_distance(a["tile"], p["tile"])
        row = {"game_id": meta["game_id"], "bot_t": round(a["t"], 2), "pro_t": p["t"], "dt": round(dt, 2), "bot_key": a["key"],
               "pro_key": p["key"], "bot_card": a["card"], "pro_card": p["card"], "bot_tile": a["tile"], "pro_tile": p["tile"],
               "tile_dist": None if dist is None else round(dist, 2), "same_card": bool(same_card), "card_known": bool(known),
               "mode": a["mode"], "reason": a["reason"], "pro_conf": p["confidence"], "cands": a["cands"], "bot_idx": a["idx"],
               "shots": (shots_index or {}).get(a["idx"])}
        rows.append(row)
        c["pair_dt_sum_abs"] += abs(dt)
        _add(c, "bot_earlier", float(dt < 0), 1)
        if known:
            _add(c, "same_card", float(same_card), 1)
            for fld in ("lane", "zone"):
                if p[fld] and a[fld]:
                    _add(c, f"same_{fld}", float(p[fld] == a[fld]), 1)
                    if same_card:
                        _add(c, f"same_{fld}_given_card", float(p[fld] == a[fld]), 1)
            if p["key"]:
                _add(c, "same_key", float(a["key"] == p["key"]), 1)
                if same_card:
                    _add(c, "same_key_given_card", float(a["key"] == p["key"]), 1)
            if dist is not None:
                _add(c, "tile_dist_all_pairs", dist, 1)
                if same_card:
                    _add(c, "tile_dist_given_card", dist, 1)
                    _add(c, "within_2_tiles_given_card", float(dist <= 2.0), 1)
            for name, v in base[j].items():                              # baselines on the matched pro plays
                if v is None:
                    continue
                _add(c, f"base_pair.{name}", v, 1)
                if name == "kg_first":                                    # the bot on the same subset of pairs
                    _add(c, "bot_on_kg_subset", float(same_card), 1)
                if same_card and name.startswith("prior_") and name != "prior_global_key":
                    _add(c, f"base_pair_given_card.{name}", v, 1)
        m = a["mode"] or "other"
        _add(c, f"mode.{m}.same_card", float(same_card), 1 if known else 0)
        c[f"mode.{m}.pairs"] += 1
        conf = p["confidence"] or "none"
        _add(c, f"conf.{conf}.same_card", float(same_card), 1 if known else 0)
        c[f"conf.{conf}.pairs"] += 1
    for a, ok in zip(acts, cov_bot):
        m = a["mode"] or "other"
        c[f"mode.{m}.acts"] += 1
        c[f"mode.{m}.precise"] += int(ok)
    for p, ok in zip(pro, covered_pro):
        conf = p["confidence"] or "none"
        c[f"conf.{conf}.pro"] += 1
        c[f"conf.{conf}.recalled"] += int(ok)
    misses = Counter()
    for j, (p, ok) in enumerate(zip(pro, covered_pro)):
        if not ok:
            misses[miss_reason(p["t"], holds, decisions, frames)] += 1
    return {"game_id": meta["game_id"], "video_id": meta.get("video_id"), "minutes": minutes, "result": meta.get("result"),
            "counts": dict(c), "pairs": rows, "miss_reasons": dict(misses), "pair_abs_dt": [abs(r["dt"]) for r in rows],
            "pair_dt": [r["dt"] for r in rows]}


def pooled(records: list[dict], name: str) -> float | None:
    """sum(num) / sum(den) of the counts `<name>.num` / `<name>.den` over the records."""
    num = sum(r["counts"].get(f"{name}.num", 0.0) for r in records)
    den = sum(r["counts"].get(f"{name}.den", 0.0) for r in records)
    return num / den if den else None


def bootstrap_ci(records: list[dict], stat, n: int = BOOTSTRAP_N, seed: int = 0, alpha: float = 0.05) -> tuple[float | None, float | None]:
    """Percentile CI of `stat(sampled records)`, resampling whole games with replacement."""
    if len(records) < 2:
        return None, None
    rng = random.Random(seed)
    vals = []
    for _ in range(n):
        sample = [records[rng.randrange(len(records))] for _ in records]
        v = stat(sample)
        if v is not None:
            vals.append(v)
    if not vals:
        return None, None
    vals.sort()
    return vals[int(alpha / 2 * len(vals))], vals[min(len(vals) - 1, int((1 - alpha / 2) * len(vals)))]


def median_abs_dt(records: list[dict]) -> float | None:
    xs = [x for r in records for x in r["pair_abs_dt"]]
    return statistics.median(xs) if xs else None


def median_dt(records: list[dict]) -> float | None:
    xs = [x for r in records for x in r["pair_dt"]]
    return statistics.median(xs) if xs else None


HEADLINE = [   # (label, ratio name, kind)
    ("recall +-1 s", "recall_1s", "rate"), ("recall +-2 s", "recall_2s", "rate"), ("recall +-3 s", "recall_3s", "rate"),
    ("precision +-1 s", "precision_1s", "rate"), ("precision +-2 s", "precision_2s", "rate"), ("precision +-3 s", "precision_3s", "rate"),
    ("same card (matched pairs)", "same_card", "rate"), ("same lane (matched pairs)", "same_lane", "rate"),
    ("same zone (matched pairs)", "same_zone", "rate"), ("same key (matched pairs)", "same_key", "rate"),
    ("same lane given same card", "same_lane_given_card", "rate"), ("same zone given same card", "same_zone_given_card", "rate"),
    ("same key given same card", "same_key_given_card", "rate"),
    ("mean tile distance, same card", "tile_dist_given_card", "mean"), ("within 2 tiles, same card", "within_2_tiles_given_card", "rate"),
    ("bot acts earlier than the pro", "bot_earlier", "rate"),
    ("same card, on the pairs where a KG threat exists", "bot_on_kg_subset", "rate"),
    ("the pro's card is in the hand the bot perceived", "pro_in_bot_hand", "rate"),
    ("the pro's card is in the hand the dataset read", "pro_in_dataset_hand", "rate"),
    ("bot taps per minute", "bot_acts_per_min", "mean"), ("pro plays per minute", "pro_plays_per_min", "mean"),
    ("bot elixir per minute", "bot_elixir_per_min", "mean"), ("pro elixir per minute", "pro_elixir_per_min", "mean"),
    ("null recall +-2 s (the bot's taps shifted 20-60 s)", "null_recall_2s", "rate"),
    ("null precision +-2 s (the bot's taps shifted 20-60 s)", "null_precision_2s", "rate"),
    ("recall +-2 s, pro stamp shifted -0.5 s", "recall_2s_pro_shift", "rate"),
    ("precision +-2 s, pro stamp shifted -0.5 s", "precision_2s_pro_shift", "rate"),
]
BASELINE_NAMES = ("rand_hand", "kg_first", "rand_hand_ds", "kg_first_ds", "prior_global_key", "prior_lane", "prior_zone", "prior_key")


def aggregate(records: list[dict], n_boot: int = BOOTSTRAP_N) -> dict:
    """Pooled numbers with bootstrap CIs (over games), baselines, breakdowns and the top miss reasons."""
    out: dict = {"n_games": len(records)}
    c = Counter()
    for r in records:
        c.update(r["counts"])
    out["totals"] = {k: round(v, 3) for k, v in c.items() if "." not in k}
    metrics = {}
    for label, name, kind in HEADLINE:
        v = pooled(records, name)
        lo, hi = bootstrap_ci(records, lambda s, nm=name: pooled(s, nm), n_boot)
        metrics[name] = {"label": label, "value": v, "ci95": [lo, hi], "n": sum(r["counts"].get(f"{name}.den", 0.0) for r in records),
                         "kind": kind}
    out["metrics"] = metrics
    mad = median_abs_dt(records)
    out["median_abs_dt_s"] = {"value": mad, "ci95": list(bootstrap_ci(records, median_abs_dt, n_boot)),
                              "n": sum(len(r["pair_abs_dt"]) for r in records)}
    out["median_dt_s"] = {"value": median_dt(records), "ci95": list(bootstrap_ci(records, median_dt, n_boot))}
    base = {}
    for scope in ("pair", "cov", "all"):
        base[scope] = {}
        for name in BASELINE_NAMES:
            key = f"base_{scope}.{name}"
            v = pooled(records, key)
            lo, hi = bootstrap_ci(records, lambda s, k=key: pooled(s, k), n_boot) if v is not None else (None, None)
            base[scope][name] = {"value": v, "ci95": [lo, hi], "n": sum(r["counts"].get(f"{key}.den", 0.0) for r in records)}
    base["pair_given_card"] = {name: {"value": pooled(records, f"base_pair_given_card.{name}"),
                                      "n": sum(r["counts"].get(f"base_pair_given_card.{name}.den", 0.0) for r in records)}
                               for name in ("prior_lane", "prior_zone", "prior_key")}
    base["bot_on_kg_subset_same_card"] = pooled(records, "bot_on_kg_subset")
    out["baselines"] = base
    by_mode = {}
    for m in sorted({k.split(".")[1] for k in c if k.startswith("mode.")}, key=lambda m: (m not in STAT_MODES, m)):
        acts, prec, pairs = c[f"mode.{m}.acts"], c[f"mode.{m}.precise"], c[f"mode.{m}.pairs"]
        sc_den = c[f"mode.{m}.same_card.den"]
        by_mode[m] = {"acts": acts, "precision_2s": prec / acts if acts else None, "pairs": pairs,
                      "same_card": c[f"mode.{m}.same_card.num"] / sc_den if sc_den else None, "same_card_n": sc_den}
    out["by_mode"] = by_mode
    by_conf = {}
    for k in sorted({k.split(".")[1] for k in c if k.startswith("conf.")}, key=lambda k: (k not in CONFS, k)):
        n_pro, rec, sc_den = c[f"conf.{k}.pro"], c[f"conf.{k}.recalled"], c[f"conf.{k}.same_card.den"]
        by_conf[k] = {"pro_plays": n_pro, "recall_2s": rec / n_pro if n_pro else None, "pairs": c[f"conf.{k}.pairs"],
                      "same_card": c[f"conf.{k}.same_card.num"] / sc_den if sc_den else None, "same_card_n": sc_den}
    out["by_pro_confidence"] = by_conf
    miss = Counter()
    for r in records:
        miss.update(r["miss_reasons"])
    total_miss = sum(miss.values())
    out["miss_reasons"] = [{"reason": k, "n": v, "share": v / total_miss} for k, v in miss.most_common(15)]
    out["n_missed_pro_plays"] = total_miss
    per_game = []
    for r in records:
        rc = r["counts"]
        per_game.append({"game_id": r["game_id"], "result": r["result"], "minutes": round(r["minutes"], 2), "pro_plays": rc.get("n_pro", 0),
                         "bot_acts": rc.get("n_bot", 0), "recall_2s": pooled([r], "recall_2s"), "precision_2s": pooled([r], "precision_2s"),
                         "same_card": pooled([r], "same_card"), "pairs": len(r["pairs"])})
    out["per_game"] = per_game
    return out


def pick_examples(records: list[dict], n_agree: int = 3, n_disagree: int = 3) -> list[dict]:
    """Q3: `n_agree` pairs where the bot played the pro's card within 2 tiles (different games, pro confidence high first),
    then `n_disagree` where it did not: a different card that was still among the bot's candidates (near miss), a
    different card the bot never considered, and the same card in another lane / zone, in that order of preference."""
    pairs = [p for r in records for p in r["pairs"] if p["card_known"] and p["shots"]]

    def conf_rank(p):
        return {"high": 0, "medium": 1}.get(p["pro_conf"], 2)

    agree = sorted((p for p in pairs if p["same_card"] and p["tile_dist"] is not None and p["tile_dist"] <= 2.0),
                   key=lambda p: (conf_rank(p), abs(p["dt"]), p["game_id"], p["bot_t"]))
    near = [p for p in pairs if not p["same_card"] and any(k.split("|")[0] == p["pro_card"] for k, _pr in p["cands"])]
    unseen = [p for p in pairs if not p["same_card"] and p not in near]
    other_place = [p for p in pairs if p["same_card"] and p["bot_key"] != p["pro_key"] and p["pro_key"]]

    def first_distinct(cands: list[dict], used_games: set[str]) -> dict | None:
        for p in sorted(cands, key=lambda p: (conf_rank(p), abs(p["dt"]), p["game_id"], p["bot_t"])):
            if p["game_id"] not in used_games:
                return p
        return None

    out: list[dict] = []
    used: set[str] = set()
    for p in agree:
        if len(out) >= n_agree:
            break
        if p["game_id"] not in used:
            out.append({**p, "verdict": "agree"})
            used.add(p["game_id"])
    for pool in (near, unseen, other_place, near, unseen, other_place):
        if sum(e["verdict"] == "disagree" for e in out) >= n_disagree:
            break
        p = first_distinct([q for q in pool if q not in out], used)
        if p is not None:
            out.append({**p, "verdict": "disagree"})
            used.add(p["game_id"])
    return out


# ---------------------------------------------------------------------------------------------------- report

def pct(v, digits: int = 1) -> str:
    return "n/a" if v is None else f"{100 * v:.{digits}f}%"


def num(v, digits: int = 2) -> str:
    return "n/a" if v is None else f"{v:.{digits}f}"


def ci_text(m: dict, rate: bool = True) -> str:
    lo, hi = m["ci95"]
    if lo is None:
        return "n/a"
    return f"{pct(lo)} to {pct(hi)}" if rate else f"{num(lo)} to {num(hi)}"


def shot_name(ex: dict, runs: str) -> str:
    files = ex.get("shots") or []
    return f"{runs}/{ex['game_id']}/shots/{files[0]}" if files else "no screenshot"


def render_md(report: dict) -> str:
    agg = report["aggregate"]
    m = agg["metrics"]
    b = agg["baselines"]
    meta = report["meta"]
    L: list[str] = []
    L.append("# The bot against the pros on their own videos (open loop)\n")
    L.append(f"Generated {meta['generated_at']} by `tools/eval/eval_vs_pro.py --score`. Code: HEAD at the runs `{'`, `'.join(h[:10] for h in meta['head'].split(','))}` (only evaluation, dashboard and test files differ between them), decision code frozen at "
             f"`{FROZEN_COMMIT}` (changed decision files: {', '.join(meta['decision_code_changed']) or 'none'}). "
             f"{agg['n_games']} games, {meta['pro_minutes']:.0f} min of pro video, {int(agg['totals'].get('n_pro', 0))} pro plays, "
             f"{int(agg['totals'].get('n_bot', 0))} bot taps. Confidence intervals: 95%, bootstrap over games ({BOOTSTRAP_N} resamples).\n")
    L.append("## Headline\n")
    L.append("| metric | bot | 95% CI | n | baseline | baseline value |")
    L.append("|---|---|---|---|---|---|")

    def row(label, key, baseline_label="", baseline_value=""):
        x = m[key]
        L.append(f"| {label} | {pct(x['value'])} | {ci_text(x)} | {int(x['n'])} | {baseline_label} | {baseline_value} |")

    nr, npc = m["null_recall_2s"], m["null_precision_2s"]
    row("pro-play recall (a bot tap within 2 s of the pro play)", "recall_2s", "same taps shifted 20-60 s (null)", f"{pct(nr['value'])} ({ci_text(nr)})")
    row("bot-tap precision (a pro play within 2 s of the tap)", "precision_2s", "same taps shifted 20-60 s (null)", f"{pct(npc['value'])} ({ci_text(npc)})")
    bt, pt = m["bot_acts_per_min"], m["pro_plays_per_min"]
    L.append(f"| taps per minute | {num(bt['value'], 1)} | {ci_text(bt, False)} | {int(bt['n'])} min | the pro's plays per minute | {num(pt['value'], 1)} ({ci_text(pt, False)}) |")
    rh, kg = b["pair"]["rand_hand"], b["pair"]["kg_first"]
    row("same card, on matched pairs", "same_card", "random card of the bot's perceived hand", f"{pct(rh['value'])} ({ci_text(rh)})")
    ks = m["bot_on_kg_subset"]
    L.append(f"| same card, on the pairs where a KG threat exists | {pct(ks['value'])} | {ci_text(ks)} | {int(kg['n'])} | "
             f"KG v1 `counter_scores` first among the bot's perceived hand | {pct(kg['value'])} ({ci_text(kg)}) |")
    pb, pd_ = m["pro_in_bot_hand"], m["pro_in_dataset_hand"]
    L.append(f"| ceiling: the pro's card is in the hand the bot perceived | {pct(pb['value'])} | {ci_text(pb)} | {int(pb['n'])} | "
             f"in the hand the dataset read | {pct(pd_['value'])} ({ci_text(pd_)}) |")
    pg = b["pair_given_card"]
    row("same lane, given the same card", "same_lane_given_card", "placement prior's lane for the pro's card", pct(pg["prior_lane"]["value"]))
    row("same zone, given the same card", "same_zone_given_card", "placement prior's zone for the pro's card", pct(pg["prior_zone"]["value"]))
    row("same key (card, lane, zone), given the same card", "same_key_given_card", "placement prior's key for the pro's card", pct(pg["prior_key"]["value"]))
    row("same key, on matched pairs", "same_key", "the prior's most common key overall", pct(b["pair"]["prior_global_key"]["value"]))
    L.append("")
    L.append("Baselines are evaluated on the same pro plays as the bot (the matched pairs). The hand is the one the bot perceived at that moment "
             "(its latest hold or decision row, at most 3 s old); the rows marked dataset use the hand the dataset's offline pipeline read, and "
             "the threats always come from the dataset's state. The ceiling row shows how often the pro's card was in the perceived hand at all: "
             "the bot cannot agree on a card it does not see. The placement-prior rows give the predictor the pro's card, which the bot does not get.\n")
    L.append("## Timing\n")
    L.append("| metric | value | 95% CI |")
    L.append("|---|---|---|")
    for key in ("recall_1s", "recall_2s", "recall_3s", "precision_1s", "precision_2s", "precision_3s",
                "recall_2s_pro_shift", "precision_2s_pro_shift", "bot_earlier"):
        x = m[key]
        L.append(f"| {x['label']} | {pct(x['value'])} | {ci_text(x)} |")
    mad = agg["median_abs_dt_s"]
    L.append(f"| median abs time difference of matched pairs | {num(mad['value'])} s | {num(mad['ci95'][0])} to {num(mad['ci95'][1])} s |")
    md = agg["median_dt_s"]
    L.append(f"| median signed difference (bot minus pro; negative = the bot is earlier) | {num(md['value'])} s | {num(md['ci95'][0])} to {num(md['ci95'][1])} s |")
    for key in ("bot_acts_per_min", "pro_plays_per_min", "bot_elixir_per_min", "pro_elixir_per_min"):
        x = m[key]
        L.append(f"| {x['label']} | {num(x['value'])} | {ci_text(x, False)} |")
    L.append("")
    L.append("## Choice on matched pairs\n")
    L.append("| metric | bot | 95% CI | n |")
    L.append("|---|---|---|---|")
    for key in ("same_card", "same_lane", "same_zone", "same_key", "same_lane_given_card", "same_zone_given_card", "same_key_given_card",
                "within_2_tiles_given_card"):
        x = m[key]
        L.append(f"| {x['label']} | {pct(x['value'])} | {ci_text(x)} | {int(x['n'])} |")
    x = m["tile_dist_given_card"]
    L.append(f"| {x['label']} | {num(x['value'])} tiles | {ci_text(x, False)} | {int(x['n'])} |")
    L.append("")
    L.append("## Baselines at the pro plays\n")
    L.append("| baseline | on matched pairs | on pro plays the bot covered (+-2 s) | on all pro plays |")
    L.append("|---|---|---|---|")
    names = {"rand_hand": "random card of the bot's perceived hand (expected same card)",
             "kg_first": "KG v1 counter_scores first in the bot's perceived hand (if a threat exists)",
             "rand_hand_ds": "random card of the dataset's hand (the pipeline's offline read)",
             "kg_first_ds": "KG v1 counter_scores first in the dataset's hand (if a threat exists)",
             "prior_global_key": "placement prior's most common key (same key)", "prior_lane": "placement prior, pro's card: same lane",
             "prior_zone": "placement prior, pro's card: same zone", "prior_key": "placement prior, pro's card: same key"}
    for name in BASELINE_NAMES:
        cells = [f"{pct(b[s][name]['value'])} (n={int(b[s][name]['n'])})" for s in ("pair", "cov", "all")]
        L.append(f"| {names[name]} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("## By the bot's mode\n")
    L.append("| mode | bot taps | precision +-2 s | matched pairs | same card | n |")
    L.append("|---|---|---|---|---|---|")
    for mode, v in agg["by_mode"].items():
        L.append(f"| {mode} | {int(v['acts'])} | {pct(v['precision_2s'])} | {int(v['pairs'])} | {pct(v['same_card'])} | {int(v['same_card_n'])} |")
    L.append("")
    L.append("## By the pro play's confidence (dataset detection)\n")
    L.append("| confidence | pro plays | recall +-2 s | matched pairs | same card | n |")
    L.append("|---|---|---|---|---|---|")
    for k, v in agg["by_pro_confidence"].items():
        L.append(f"| {k} | {int(v['pro_plays'])} | {pct(v['recall_2s'])} | {int(v['pairs'])} | {pct(v['same_card'])} | {int(v['same_card_n'])} |")
    L.append("")
    L.append("## By game\n")
    L.append("| game | result | min | pro plays | bot taps | recall +-2 s | precision +-2 s | same card |")
    L.append("|---|---|---|---|---|---|---|---|")
    for g in agg["per_game"]:
        L.append(f"| {g['game_id']} | {g['result']} | {g['minutes']} | {int(g['pro_plays'])} | {int(g['bot_acts'])} | {pct(g['recall_2s'])} | "
                 f"{pct(g['precision_2s'])} | {pct(g['same_card'])} |")
    L.append("")
    L.append(f"## When the bot held while the pro played ({agg['n_missed_pro_plays']} pro plays with no bot tap within 2 s)\n")
    L.append("| what the bot was doing (mode: reason) | pro plays | share |")
    L.append("|---|---|---|")
    for r in agg["miss_reasons"]:
        L.append(f"| {r['reason']} | {r['n']} | {pct(r['share'])} |")
    L.append("")
    L.append("## Examples for the slides\n")
    for i, ex in enumerate(report["examples"], 1):
        L.append(f"**{i}. {ex['verdict'].upper()}** game {ex['game_id']}, video t = {ex['pro_t']:.1f} s (bot tap at {ex['bot_t']:.1f} s, "
                 f"difference {ex['dt']:+.1f} s)")
        L.append(f"- pro: {ex['pro_key'] or ex['pro_card']} at tile {ex['pro_tile']} (detection confidence {ex['pro_conf']})")
        L.append(f"- bot: {ex['bot_key']} at tile {ex['bot_tile']}, mode {ex['mode']}: {ex['reason']}")
        cands = ", ".join(f"{k} ({'n/a' if p is None else f'{p:.2f}'})" for k, p in ex["cands"])
        L.append(f"- d1 over the bot's candidates: {cands}")
        L.append(f"- screenshot: `{shot_name(ex, meta['runs_dir'])}`\n")
    L.append("## Games\n")
    L.append("| game | video min | wall s | x real time | frames | decisions | calib | hero | evo |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for g in meta["games"]:
        L.append(f"| {g['game_id']} | {g['minutes']:.1f} | {g.get('wall_s')} | {num(g.get('wall_s', 0) / (60 * g['minutes']) if g['minutes'] else None)} | "
                 f"{g.get('frames')} | {g.get('decisions')} | {g.get('calib')} | {', '.join(g.get('hero_cards') or []) or '-'} | "
                 f"{', '.join(g.get('evo_cards') or []) or '-'} |")
    if meta.get("skipped"):
        L.append("\nNot run: " + "; ".join(f"{s['game_id']} ({s['skip_reason']})" for s in meta["skipped"]) + ".")
    L.append("\n## How to read this (caveats)\n")
    L.append(
        "Open loop: the bot's taps do not change the video, so each tap is what the bot would do in the pro's situation, not a game it "
        "played; after the pro's own play the board already shows the pro's units, which the bot also sees. Tap density: the bot taps more "
        "often than the pro (see taps per minute), so timing recall is read against the null row (the same taps moved 20-60 s), not against zero. "
        "Perception ceiling: the pro's card is in the hand the bot perceived in only the share the ceiling row gives, so same-card agreement "
        "cannot be high; in open loop the bot's hand memory is never told about a play (taps are not sent), which may explain a stale hand "
        "(not verified here). Calibration: each game uses its own video's calibration (`runs/videos/<video>/match_0/calib.json`), because "
        "`runs/poc/mirror_calib.json` puts the arena up to 2-4.5 tiles off on these videos. Perception noise: the bot "
        "perceives the screen with the same detector it uses live, and the pro's plays come from the dataset's offline pipeline "
        "(card labels on 'medium' / 'low' confidence plays and 'unknown'-card plays are noisy; a card outside the pro's deck is kept for timing "
        "and dropped from the choice metrics). Time stamps: the dataset stamps a play when the card lands on the arena, about 0.5 s after the pro's "
        "tap, so the bot looks slightly early; the shifted rows show the effect. Hidden information: the pro knows his own next cards, the "
        "opponent's real elixir and his plan, and plays for the match result; the bot sees only the screen and acts only when its act model "
        "says so, so low recall is not an error by itself and a different card is not always a worse one. Retrieval was limited to the "
        "train and val splits so the bot never retrieves the very state the pro played; the KG v2 answers and the placement prior are mined from "
        "train games only.")
    return "\n".join(L) + "\n"


def shots_index_of(run_dir: Path) -> dict[int, list[str]]:
    path = Path(run_dir) / "shots" / "index.jsonl"
    if not path.is_file():
        return {}
    return {r["idx"]: [f for f in (r.get("files") or []) if f] for r in read_jsonl(path) if "idx" in r}


def load_run(run_dir: Path) -> tuple[dict, list[dict], list[dict], list[dict], list[dict]]:
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "vs_pro_meta.json").read_text())
    return (meta, read_jsonl(run_dir / "pro_plays.jsonl"), read_jsonl(run_dir / "decisions.jsonl"),
            read_jsonl(run_dir / "holds.jsonl") if (run_dir / "holds.jsonl").exists() else [],
            read_jsonl(run_dir / "frames.jsonl") if (run_dir / "frames.jsonl").exists() else [])


def refresh_pro_plays(out_root: Path, dataset: Path = DATASET_DIR) -> int:
    """Rewrite `pro_plays.jsonl` of every run dir from the dataset with the current extractor (it depends only on the
    dataset, the game's deck and its window, never on the bot run). Returns the number of files written."""
    metas = {}
    for run_dir in sorted(Path(out_root).iterdir()):
        if run_dir.is_dir() and (run_dir / "vs_pro_meta.json").is_file():
            metas[run_dir.name] = json.loads((run_dir / "vs_pro_meta.json").read_text())
    if not metas:
        return 0
    states = read_states(set(metas), dataset)
    for gid, meta in metas.items():
        plays = pro_plays(states.get(gid, []), meta["own_deck"], meta["start"], meta["end"])
        (Path(out_root) / gid / "pro_plays.jsonl").write_text("".join(json.dumps(p) + "\n" for p in plays))
    return len(metas)


def score_runs(out_root: Path, baselines: Baselines, n_boot: int = BOOTSTRAP_N) -> dict:
    """Score every finished run dir under `out_root`."""
    records, metas = [], []
    for run_dir in sorted(Path(out_root).iterdir()):
        if not run_dir.is_dir() or not run_finished(run_dir) or not (run_dir / "vs_pro_meta.json").is_file():
            continue
        meta, pro, decisions, holds, frames = load_run(run_dir)
        records.append(score_game(meta, pro, decisions, holds, frames, baselines, shots_index_of(run_dir)))
        metas.append(meta)
    agg = aggregate(records, n_boot)
    heads = {m.get("head") for m in metas}
    changed = sorted({p for m in metas for p in m.get("decision_code_changed") or []})
    meta = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "head": ",".join(sorted(h for h in heads if h)) or "unknown",
            "frozen_commit": FROZEN_COMMIT, "decision_code_changed": changed, "pro_minutes": sum(m["minutes"] for m in metas),
            "runs_dir": str(out_root), "games": [{k: m.get(k) for k in ("game_id", "minutes", "wall_s", "frames", "decisions", "calib", "hero_cards",
                                                                          "evo_cards", "load_s")} for m in metas],
            "window_s": MATCH_S, "windows_s": list(WINDOWS), "pro_shift_s": PRO_SHIFT_S}
    return {"meta": meta, "aggregate": agg, "examples": pick_examples(records), "records": [
        {k: r[k] for k in ("game_id", "video_id", "minutes", "result", "counts", "miss_reasons")} for r in records]}


def make_baselines() -> Baselines:
    from clashbot.actions import load_placement_prior
    prior = load_placement_prior()
    try:
        from clashbot.kg import KnowledgeGraph
        kg = KnowledgeGraph()
        scores = kg.counter_scores
    except Exception as exc:                               # noqa: BLE001  no db: the KG baseline is simply not applicable
        print(f"note: no knowledge graph ({type(exc).__name__}: {exc}); the KG baseline is skipped")
        scores = None
    return Baselines(prior, scores)


# ---------------------------------------------------------------------------------------------------- CLI

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--split", default="test")
    ap.add_argument("--games", type=int, default=None, help="run only the first N selected games")
    ap.add_argument("--only", default=None, help="comma-separated game ids")
    ap.add_argument("--stride", type=int, default=3, help="video frame stride (3 = 10 fps of a 30 fps video, the live rate)")
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--calib", default="auto", help="`auto` = each video's own calibration, else a calibration path for all games")
    ap.add_argument("--score", action="store_true", help="score the finished run dirs instead of running games")
    ap.add_argument("--docs", default=DOCS_DEFAULT, help="--score output prefix (.json and .md)")
    ap.add_argument("--list", action="store_true", help="print the plan (selected / skipped games) and exit")
    ap.add_argument("--run-game", default=None, help="(internal) run one game in this process")
    ap.add_argument("--lock", default=str(LOCK_FILE))
    ap.add_argument("--hold-file", default=str(HOLD_FILE))
    ap.add_argument("--poll", type=float, default=POLL_S)
    ap.add_argument("--retries", type=int, default=1)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_root = Path(args.out)
    if args.score:
        print(f"pro_plays.jsonl refreshed in {refresh_pro_plays(out_root)} run dirs")
        report = score_runs(out_root, make_baselines())
        docs = Path(args.docs)
        docs.parent.mkdir(parents=True, exist_ok=True)
        plan_games, skipped = select_games(load_games(args.split))
        report["meta"]["skipped"] = [{"game_id": s["game_id"], "skip_reason": s["skip_reason"]} for s in skipped]
        ran = {g["game_id"] for g in report["meta"]["games"]}
        report["meta"]["skipped"] += [{"game_id": g["game_id"], "skip_reason": "not run yet: queued behind the shared heavy lock; "
                                      "`--split test` resumes, `--score` rewrites this report"} for g in plan_games
                                      if g["game_id"] not in ran]
        docs.with_suffix(".json").write_text(json.dumps(report, indent=1, default=str) + "\n")
        docs.with_suffix(".md").write_text(render_md(report))
        print(f"wrote {docs.with_suffix('.json')} and {docs.with_suffix('.md')}")
        return 0
    games = load_games(args.split)
    only = [x for x in (args.only or "").split(",") if x] or None
    selected, skipped = select_games(games, args.games, only)
    if args.run_game:
        game = next(g for g in games if g["game_id"] == args.run_game)
        return run_game(game, out_root, stride=args.stride, calib=args.calib, hold_file=Path(args.hold_file))
    minutes = sum(g["end"] - g["start"] for g in selected) / 60.0
    print(f"{len(selected)} games, {minutes:.0f} min of video; skipped {len(skipped)}: "
          + "; ".join(f"{s['game_id']} ({s['skip_reason']})" for s in skipped))
    if args.list:
        for g in selected:
            print(f"  {g['game_id']:22s} {g['result']:8s} {(g['end'] - g['start']) / 60:5.1f} min"
                  f"{'  [done]' if run_finished(out_root / g['game_id']) else ''}")
        return 0
    res = orchestrate(selected, out_root, stride=args.stride, calib=args.calib, lock=Path(args.lock), hold_file=Path(args.hold_file),
                      poll_s=args.poll, retries=args.retries)
    print(json.dumps(res))
    return 0 if not res["failed"] else 1


if __name__ == "__main__":
    sys.exit(main())

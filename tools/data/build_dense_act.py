"""Dense wait/act dataset from re-run perception output (MAC-V, D-040).

  python3 tools/data/build_dense_act.py [--dataset DIR] [--runs runs/videos] [--step 1.0] [--horizon 1.0]
         [--tol 0.6] [--games ID,...] [--out DIR]

The sparse act dataset only has "play" moments (positives) and a few hand-picked "key"/"periodic" waits.
Here every game is sampled once per `--step` seconds: the state comes from the re-run perception rows
(`<runs>/<video_id>/**/states.jsonl`) through the live builder `clashbot.live_state.LiveStateBuilder` (events fed
incrementally, so first-seen times and the hand hold behave exactly like the live loop) and the label is
"did the pro deploy a card within `--horizon` seconds". A game uses the rows of ONE perception match only (the
(source file, match_id) group with the most state rows inside the game window; `multi_match_groups` counts games
whose window held more than one). Writes <out>/{train,val,test}.jsonl in d1 act format
(<dataset>/act_dense by default), <out>/stats.json and <out>/README.md. Perception output is read only.
"""
from __future__ import annotations

import argparse
import bisect
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from clashbot import d1_format  # noqa: E402
from clashbot.config import DATASET_DIR  # noqa: E402
from clashbot.game_rules import is_playable, load_cards  # noqa: E402
from clashbot.live_state import LiveConfig, LiveStateBuilder  # noqa: E402
from clashbot.units import load_unit_map  # noqa: E402
from cr_perception.recorder import read_jsonl  # noqa: E402

SPLITS = ("train", "val", "test")
COLLAPSE_S = 1.0        # the same card read twice within this many seconds is one deploy
SOON_S = 3.0            # "shortly after an opponent play" for the stats
MARGIN_S = 1.0          # perception rows are kept this far outside the game's [start, end]
MIN_STEP_S = 0.1        # ids carry t with one decimal


# --------------------------------------------------------------------------- pro deploy times (labels)

def pro_deploy_times(records: list[dict], cards: dict) -> list[tuple[float, str]]:
    """Sorted (time, card) of the pro's own deploys in one game, from the sparse dataset records.

    Sources: `state.own.recent_plays`, `action.following_plays` and the record's own `play` action
    (`t + delay_s`). Entries without a playable card are dropped, repeats of one event across overlapping
    windows are deduped on (round(t, 1), card) and the same card re-read within COLLAPSE_S is collapsed.
    """
    found: dict[tuple[float, str], float] = {}

    def add(t, card) -> None:
        if isinstance(t, bool) or not isinstance(t, (int, float)):
            return
        if not (isinstance(card, str) and card and is_playable(card, cards)):
            return
        key = (round(float(t), 1), card)
        found[key] = min(found.get(key, float(t)), float(t))

    for rec in records:
        state = rec.get("state") or {}
        action = rec.get("action") or {}
        for p in (state.get("own") or {}).get("recent_plays") or []:
            add(p.get("t"), p.get("card"))
        for p in action.get("following_plays") or []:
            add(p.get("t"), p.get("card"))
        delay = action.get("delay_s")
        if (action.get("type") == "play" and isinstance(delay, (int, float)) and not isinstance(delay, bool)
                and isinstance(rec.get("t"), (int, float))):
            add(float(rec["t"]) + float(delay), action.get("card"))

    out: list[tuple[float, str]] = []
    last: dict[str, float] = {}
    for (_, card), t in sorted(found.items(), key=lambda kv: (kv[1], kv[0][1])):
        if card in last and t - last[card] <= COLLAPSE_S:
            continue
        last[card] = t
        out.append((round(t, 3), card))
    return out


# --------------------------------------------------------------------------- inputs

def _num(x) -> float | None:
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        try:
            return float(x)
        except ValueError:
            return None
    return None


def read_list(path: Path) -> list[dict]:
    return list(read_jsonl(path)) if path.exists() else []


def video_window(prov: dict | None) -> tuple[float, float] | None:
    vt = (prov or {}).get("video_time")
    if not isinstance(vt, (list, tuple)) or len(vt) != 2:
        return None
    start, end = _num(vt[0]), _num(vt[1])
    if start is None or end is None or end < start:
        return None
    return start, end


def _slim(rec: dict) -> dict:
    """Keep only what pro_deploy_times reads (the full file is ~70 MB)."""
    action = rec.get("action") or {}
    return {"t": rec.get("t"),
            "state": {"own": {"recent_plays": ((rec.get("state") or {}).get("own") or {}).get("recent_plays")}},
            "action": {"type": action.get("type"), "card": action.get("card"), "delay_s": action.get("delay_s"),
                       "following_plays": action.get("following_plays")}}


def load_records(path: Path, wanted: set[str]) -> dict[str, list[dict]]:
    by_game: dict[str, list[dict]] = {}
    if not path.exists():
        return by_game
    for rec in read_jsonl(path):
        gid = rec.get("game_id")
        if gid in wanted:
            by_game.setdefault(gid, []).append(_slim(rec))
    return by_game


Group = tuple[str, object]      # (source states.jsonl path, match_id): one perception match


class Perception:
    """Match-readiness state rows and play events of one video, read once (games of a video are contiguous).

    Every row is tagged with its group = (source file, match_id): `match_id` restarts in every file and the
    live builder resets on a change, so a game must be built from one group only (FIX_LIST #24). Play events
    carry no match_id; each takes the group of the nearest state row in its own file.
    """

    def __init__(self, runs: Path):
        self.runs = runs
        self._video: str | None = None
        self._states: list[dict] = []
        self._s_ts: list[float] = []
        self._s_grp: list[Group] = []
        self._events: list[dict] = []
        self._e_ts: list[float] = []
        self._e_grp: list[Group] = []

    def _load(self, video_id: str) -> None:
        states: list[tuple[float, Group, dict]] = []
        events: list[tuple[float, Group, dict]] = []
        for path in sorted((self.runs / video_id).glob("**/states.jsonl")):
            f_states: list[tuple[float, Group, dict]] = []
            f_events: list[dict] = []
            for row in read_jsonl(path):
                kind = row.get("type")
                if kind == "state" and row.get("readiness") == "match" and _num(row.get("t")) is not None:
                    f_states.append((row["t"], (str(path), row.get("match_id")), row))
                elif kind == "play_event" and _num(row.get("timestamp")) is not None:
                    f_events.append(row)
            f_states.sort(key=lambda x: x[0])
            f_ts = [x[0] for x in f_states]
            for ev in f_events:
                i = nearest(f_ts, ev["timestamp"], float("inf"))
                if i is not None:                         # a file without match rows owns no group
                    events.append((ev["timestamp"], f_states[i][1], ev))
            states.extend(f_states)
        states.sort(key=lambda x: x[0])
        events.sort(key=lambda x: x[0])
        self._video = video_id
        self._s_ts, self._s_grp, self._states = ([x[i] for x in states] for i in range(3))
        self._e_ts, self._e_grp, self._events = ([x[i] for x in events] for i in range(3))

    def window(self, video_id: str, start: float, end: float) -> tuple[list[dict], list[dict], int]:
        """(state rows, play events, number of groups present) of the dominant group around [start, end].

        Rows are looked up in [start - MARGIN_S, end + MARGIN_S]; the dominant group is the one with the most
        state rows inside [start, end] (ties: the group whose first row is closest to `start`). Only its rows
        and events are returned; exact-duplicate times (overlapping runs of one match) are dropped.
        """
        if self._video != video_id:
            self._load(video_id)
        lo, hi = start - MARGIN_S, end + MARGIN_S
        a, b = bisect.bisect_left(self._s_ts, lo), bisect.bisect_right(self._s_ts, hi)
        groups: dict[Group, list[int]] = {}
        for i in range(a, b):
            groups.setdefault(self._s_grp[i], []).append(i)
        if not groups:
            return [], [], 0

        def rank(g: Group) -> tuple:
            idx = groups[g]
            inside = sum(1 for i in idx if start <= self._s_ts[i] <= end)
            return (-inside, abs(self._s_ts[idx[0]] - start), g[0], repr(g[1]))

        best = min(groups, key=rank)
        uniq: list[dict] = []
        for i in groups[best]:
            if not uniq or self._s_ts[i] != uniq[-1]["t"]:
                uniq.append(self._states[i])
        ea, eb = bisect.bisect_left(self._e_ts, lo), bisect.bisect_right(self._e_ts, hi)
        events = [self._events[i] for i in range(ea, eb) if self._e_grp[i] == best]
        return uniq, events, len(groups)


def nearest(ts: list[float], t: float, tol: float) -> int | None:
    i = bisect.bisect_left(ts, t)
    best = None
    for j in (i - 1, i):
        if 0 <= j < len(ts) and abs(ts[j] - t) <= tol and (best is None or abs(ts[j] - t) < abs(ts[best] - t)):
            best = j
    return best


# --------------------------------------------------------------------------- one game

def build_game(game: dict, window: tuple[float, float], states: list[dict], events: list[dict],
               deploys: list[tuple[float, str]], args: argparse.Namespace, cards: dict,
               unit_map: dict) -> tuple[list[dict], Counter]:
    """Dense act lines of one game (in time order) and the counters of samples that were skipped."""
    start, end = window
    skips: Counter = Counter()
    lines: list[dict] = []
    s_ts = [r["t"] for r in states]
    d_ts = [d for d, _ in deploys]
    builder = LiveStateBuilder(LiveConfig(own_deck=list(game["own_deck"])), cards=cards, unit_map=unit_map)
    fed = 0
    n = int((end - start) / args.step + 1e-9) + 1
    for k in range(n):
        t = round(start + k * args.step, 3)
        j = fed
        while j < len(events) and events[j]["timestamp"] <= t:
            j += 1
        if j > fed:                                       # events become visible once their time has come
            builder.observe_events(events[fed:j])
            fed = j
        i = nearest(s_ts, t, args.tol)
        if i is None:
            skips["no_state"] += 1
            continue
        row = dict(states[i])
        row["t"] = t
        built = builder.build(row)
        if built is None:
            skips["no_build"] += 1
            continue
        state, context = built
        nxt = bisect.bisect_right(d_ts, t)
        next_in = round(d_ts[nxt] - t, 2) if nxt < len(d_ts) else None
        act = nxt < len(d_ts) and d_ts[nxt] <= t + args.horizon
        opp_plays = state["opponent"]["recent_plays"]
        since = round(t - max(p["t"] for p in opp_plays), 2) if opp_plays else None
        s, q = d1_format.act_request(context)
        answers = {"act": "true" if act else "false"}
        d1_format.validate(s, q, answers)
        lines.append({"id": f"{game['game_id']}#{t:.1f}", "game_id": game["game_id"], "split": game["split"],
                      "kind": "dense", "variant": "act", "state": s, "questions": q, "answers": answers,
                      "meta": {"since_opp_play": since, "threat": bool(state["threats"]),
                               "own_elixir": state["own"]["elixir"],
                               "opp_elixir": state["opponent"]["elixir_estimate"],
                               "next_deploy_in": next_in}})
    return lines, skips


# --------------------------------------------------------------------------- stats

def _rate(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


class SplitStats:
    def __init__(self) -> None:
        self.lines = self.acts = self.games = self.games_listed = self.multi_match_groups = 0
        self.threat_n = self.threat_acts = self.soon_n = self.soon_acts = 0
        self.skipped: Counter = Counter()

    def add_game(self, lines: list[dict], skips: Counter) -> None:
        self.games += bool(lines)
        self.skipped.update(skips)
        for line in lines:
            act = line["answers"]["act"] == "true"
            meta = line["meta"]
            self.lines += 1
            self.acts += act
            if meta["threat"]:
                self.threat_n += 1
                self.threat_acts += act
            if meta["since_opp_play"] is not None and meta["since_opp_play"] <= SOON_S:
                self.soon_n += 1
                self.soon_acts += act

    def to_json(self) -> dict:
        return {"lines": self.lines, "act_rate": _rate(self.acts, self.lines), "games": self.games,
                "games_listed": self.games_listed, "multi_match_groups": self.multi_match_groups,
                "skipped": {k: self.skipped.get(k, 0)
                            for k in ("skipped_no_time", "skipped_no_perception", "no_state", "no_build")},
                "n_threat": self.threat_n, "act_rate_when_threat": _rate(self.threat_acts, self.threat_n),
                "n_within_3s_of_opp_play": self.soon_n,
                "act_rate_within_3s_of_opp_play": _rate(self.soon_acts, self.soon_n)}


def combine(parts: list[SplitStats]) -> SplitStats:
    tot = SplitStats()
    for p in parts:
        tot.lines += p.lines
        tot.acts += p.acts
        tot.games += p.games
        tot.games_listed += p.games_listed
        tot.multi_match_groups += p.multi_match_groups
        tot.threat_n += p.threat_n
        tot.threat_acts += p.threat_acts
        tot.soon_n += p.soon_n
        tot.soon_acts += p.soon_acts
        tot.skipped.update(p.skipped)
    return tot


def _fmt(x) -> str:
    return "-" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))


def stats_table(stats: dict) -> str:
    cols = ("games", "lines", "act_rate", "act_rate_when_threat", "act_rate_within_3s_of_opp_play",
            "multi_match_groups")
    skip = ("skipped_no_time", "skipped_no_perception", "no_state", "no_build")
    head = ["split", *cols, *skip]
    rows = [head]
    for name in (*SPLITS, "total"):
        s = stats["splits"][name] if name != "total" else stats["total"]
        rows.append([name, *(_fmt(s[c]) for c in cols), *(_fmt(s["skipped"][k]) for k in skip)])
    widths = [max(len(r[i]) for r in rows) for i in range(len(head))]
    return "\n".join("  ".join(c.rjust(w) if i else c.ljust(w) for i, (c, w) in enumerate(zip(r, widths)))
                     for r in rows)


def readme_text(args: argparse.Namespace, stats: dict) -> str:
    t = stats["total"]
    return f"""# Dense wait/act dataset (D-040)

Generated by `tools/data/build_dense_act.py` (MAC-V). Do not edit by hand.

One sample per `{args.step:g}` s of every pro game, in d1's act format (`clashbot/d1_format.py::act_request`,
question `act`, answers `"true"` = deploy a card now, `"false"` = wait and keep elixir). Files:
`train.jsonl`, `val.jsonl`, `test.jsonl` (split = the game's split in `games.jsonl`; file order = game order,
then time), `stats.json`.

## Line
`{{"id": "<game_id>#<t>", "game_id", "split", "kind": "dense", "variant": "act", "state", "questions",
"answers": {{"act": "true"|"false"}}, "meta": {{...}}}}` with `t` in video seconds.

## States
Built from the re-run perception output (`<runs>/<video_id>/**/states.jsonl`, match-readiness rows) with the live
builder `clashbot/live_state.py::LiveStateBuilder`, a fresh builder per game, the play events fed incrementally
(each only once its timestamp has passed), so first-seen opponent cards, recent plays and the hand hold are
exactly what the live loop would have seen (D-040). Only one perception match is used per game: the
(source file, match_id) group with the most state rows inside the game window; rows and play events of any other
group in the window (a neighbouring match) are ignored, and stats `multi_match_groups` counts the games where
there was more than one. A sample takes the perception row nearest to `t` within `{args.tol:g}` s; otherwise it
is skipped (`no_state`).

## Labels
`act = true` iff the pro deployed a card in `(t, t + {args.horizon:g}]` s. Deploy times come from the sparse
dataset (`states.jsonl`: own `recent_plays`, `following_plays` and `t + delay_s` of play actions), deduped on
(round(t, 1), card), the same card within {COLLAPSE_S:g} s collapsed.

## meta
`since_opp_play` (s since the latest opponent play in the 12 s window, else null), `threat` (state has threats),
`own_elixir`, `opp_elixir` (estimate), `next_deploy_in` (s to the next pro deploy, else null).

## Numbers
{t['lines']} lines from {t['games']} of {t['games_listed']} games, act rate {_fmt(t['act_rate'])}
(when a threat is present {_fmt(t['act_rate_when_threat'])}, within {SOON_S:g} s of an opponent play
{_fmt(t['act_rate_within_3s_of_opp_play'])}). Per split: see `stats.json`.
"""


# --------------------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", type=Path, default=DATASET_DIR)
    ap.add_argument("--runs", type=Path, default=ROOT / "runs" / "videos", help="perception output root")
    ap.add_argument("--step", type=float, default=1.0, help="seconds between samples")
    ap.add_argument("--horizon", type=float, default=1.0, help="label = a pro deploy within this many seconds")
    ap.add_argument("--tol", type=float, default=0.6, help="max |t_sample - t_state| in seconds")
    ap.add_argument("--games", default=None,
                    help="comma-separated game_ids or video_ids to build (use --games=ID if one starts with '-')")
    ap.add_argument("--out", type=Path, default=None, help="default <dataset>/act_dense")
    a = ap.parse_args(argv)
    if a.step < MIN_STEP_S:
        ap.error(f"--step must be >= {MIN_STEP_S}")
    out_dir = a.out or a.dataset / "act_dense"

    games = read_list(a.dataset / "games.jsonl")
    if not games:
        raise SystemExit(f"no games in {a.dataset / 'games.jsonl'}")
    prov = {p["game_id"]: p for p in read_list(a.dataset / "provenance.jsonl") if "game_id" in p}
    if a.games:
        keep = {g.strip() for g in a.games.split(",") if g.strip()}
        games = [g for g in games if g["game_id"] in keep or (prov.get(g["game_id"]) or {}).get("video_id") in keep]
    records = load_records(a.dataset / "states.jsonl", {g["game_id"] for g in games})
    cards, unit_map = load_cards(), load_unit_map()
    perception = Perception(a.runs)

    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {s: SplitStats() for s in SPLITS}
    files = {s: (out_dir / f"{s}.jsonl").open("w", encoding="utf-8") for s in SPLITS}
    try:
        for game in games:
            split = game.get("split")
            if split not in stats:
                continue
            st = stats[split]
            st.games_listed += 1
            p = prov.get(game["game_id"])
            win = video_window(p)
            if win is None or not game.get("own_deck"):
                st.skipped["skipped_no_time"] += 1
                continue
            states, events, n_groups = perception.window(p["video_id"], *win)
            st.multi_match_groups += n_groups > 1
            if not states:
                st.skipped["skipped_no_perception"] += 1
                continue
            deploys = pro_deploy_times(records.get(game["game_id"], []), cards)
            lines, skips = build_game(game, win, states, events, deploys, a, cards, unit_map)
            for line in lines:
                files[split].write(json.dumps(line, ensure_ascii=False) + "\n")
            st.add_game(lines, skips)
    finally:
        for f in files.values():
            f.close()

    summary = {"args": {"step": a.step, "horizon": a.horizon, "tol": a.tol, "dataset": str(a.dataset),
                        "runs": str(a.runs)},
               "splits": {s: stats[s].to_json() for s in SPLITS},
               "total": combine(list(stats.values())).to_json()}
    (out_dir / "stats.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "README.md").write_text(readme_text(a, summary), encoding="utf-8")
    print(stats_table(summary))
    print(f"wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

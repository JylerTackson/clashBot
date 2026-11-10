#!/usr/bin/env python3
"""Pack every prior game for the Docker submission (MAC-AW part P): called by scripts/make_docker_submission.sh.

    python scripts/pack_prior_runs.py [--runs runs] [--out dist/docker_build/prior_runs] [--scale 0.6] [--quality 70]
                                      [--budget-mb 250] [--list]

Reads the run folders under `runs/` (read-only: nothing under --runs is ever modified or deleted) and writes a small copy of
each run the dashboard can browse (`python -m clashbot.viz --parents ...`):

    <out>/<group>/<run>/   config.json summary.json decisions.jsonl holds.jsonl battles.jsonl pro_plays.jsonl vs_pro_meta.json
                           shots/*.jpg (re-encoded: --scale of the size, JPEG --quality)  shots/index.jsonl  label.txt
    <out>/MANIFEST.json    what was packed, what was left out and why, sizes before and after

Groups (the dashboard's run parents): `poc` (runs/poc/2026*) and `demo` (runs/demo/2026*), the live games, those with a
non-empty decisions.jsonl; `vs_pro` (runs/eval/vs_pro/*), the bot on pro videos; `smoke` (runs/smoke/AO_demo and AM_merged only).
Dropped: frames.jsonl (the dashboard does not read it), every video, the battle end screenshots, the review sheets (PNG / md),
logs. decisions.jsonl and the other data files are copied byte for byte; only the screenshots are re-encoded. Files are written
read-only (0444). If the pack is over --budget-mb the vs_pro screenshots are re-encoded smaller first, then only the
screenshots of the EXAMPLE_RUNS are kept (decisions always stay).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import stat
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (group = the run parent the dashboard shows, folder under runs/)
GROUPS = (("poc", "poc"), ("demo", "demo"), ("vs_pro", "eval/vs_pro"), ("smoke", "smoke"))
SMOKE_KEEP = ("AO_demo", "AM_merged")                  # dry runs on the test clip; the other smoke folders are benchmarks / scratch
STAMP_RE = re.compile(r"^2026\d{4}-\d{6}$")             # a dated live session folder
SHOT_RE = re.compile(r"^d(\d+)_t([0-9.]+)_([abc])\.jpg$")        # the names the dashboard lists (clashbot.viz.data.SHOT_NAME_RE)
KEEP_FILES = ("config.json", "summary.json", "decisions.jsonl", "holds.jsonl", "battles.jsonl", "pro_plays.jsonl", "vs_pro_meta.json")
SHOT_INDEX = "index.jsonl"
LABEL_FILE = "label.txt"
DEFAULT_SCALE, DEFAULT_QUALITY, DEFAULT_BUDGET_MB = 0.6, 70, 250.0
LOW_SCALE, LOW_QUALITY = 0.5, 55                        # the vs_pro shots when the pack is over budget
# the games whose screenshots stay in the last resort of the budget: the first win, the 10-battle trial, the latest live session,
# the two smoke dry runs and one bot-vs-pro game
EXAMPLE_RUNS = ("poc/20261107-013441", "demo/20261107-213709", "demo/20261109-135341", "smoke/AO_demo", "smoke/AM_merged",
                "vs_pro/6qYQNJ1Uaeg-m0.3")
LABEL_OVERRIDES = {                                     # facts the run files do not hold (docs/data/BENCHMARKS.md, docs/mac/STATUS.md)
    "poc/20261107-013441": "Live 2026-11-07 01:34 (first win)",
    "demo/20261107-213709": "Live trial 2026-11-07 (10 battles)",
    "smoke/AO_demo": "Dry run on the test clip (AO demo)",
    "smoke/AM_merged": "Dry run on the test clip (AM merged)",
}


@dataclass
class Pick:
    group: str
    name: str
    src: Path
    label: str
    decisions: int = 0

    @property
    def key(self) -> str:
        return f"{self.group}/{self.name}"


@dataclass
class Plan:
    """How one pass packs: the JPEG scale / quality (per group override for vs_pro), and the runs that keep shots (None = all)."""
    scale: float = DEFAULT_SCALE
    quality: int = DEFAULT_QUALITY
    vs_scale: float | None = None
    vs_quality: int | None = None
    shots_only: tuple[str, ...] | None = None
    name: str = "default"

    def jpeg(self, group: str) -> tuple[float, int]:
        if group == "vs_pro":
            return (self.vs_scale or self.scale, self.vs_quality or self.quality)
        return (self.scale, self.quality)

    def keeps_shots(self, key: str) -> bool:
        return self.shots_only is None or key in self.shots_only


# -------------------------------------------------------------------------------------- which runs
def decision_rows(run: Path) -> int:
    """Non-blank lines of decisions.jsonl (0 for a missing or empty file)."""
    try:
        with open(run / "decisions.jsonl", "rb") as fh:
            return sum(1 for line in fh if line.strip())
    except OSError:
        return 0


def _json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _battle_rows(run: Path) -> int:
    try:
        with open(run / "battles.jsonl", "rb") as fh:
            return sum(1 for line in fh if line.strip())
    except OSError:
        return 0


def label_for(group: str, name: str, run: Path | None = None) -> str:
    """A readable name for the run picker: "Live 2026-11-07 01:34", "Live trial 2026-11-07 (10 battles)", "Bot vs pro:
    6qYQNJ1Uaeg m0.3". `run` (the folder) is only read for the config and the number of battles."""
    fixed = LABEL_OVERRIDES.get(f"{group}/{name}")
    if fixed:
        return fixed
    cfg = _json(run / "config.json") if run is not None else {}
    if group == "vs_pro":
        video, sep, match = name.rpartition("-m")
        return f"Bot vs pro: {video} m{match}" if sep else f"Bot vs pro: {name}"
    if group == "smoke":
        return f"Dry run on the test clip ({name.replace('_', ' ')})"
    m = re.match(r"^(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})\d{2}$", name)
    when = f"{m.group(1)}-{m.group(2)}-{m.group(3)} {m.group(4)}:{m.group(5)}" if m else name
    live = bool(cfg.get("live")) or not cfg.get("dry_run", True)
    battles = _battle_rows(run) if run is not None else 0
    if group == "demo":
        return f"Live session {when} ({battles} battles)" if battles > 1 else f"Live session {when}"
    return f"Live {when}" if live else f"Mirroring dry run {when}"


def select_runs(runs: Path) -> tuple[list[Pick], list[tuple[str, str]]]:
    """(the runs to pack, [(runs-relative path, why it is left out)]). Live games: every runs/poc/2026* and runs/demo/2026* with a
    non-empty decisions.jsonl; all of runs/eval/vs_pro with decisions; runs/smoke/AO_demo and AM_merged. Plain files are ignored."""
    picks: list[Pick] = []
    left: list[tuple[str, str]] = []
    for group, rel in GROUPS:
        root = runs / rel
        if not root.is_dir():
            left.append((rel, "folder not found"))
            continue
        for p in sorted(root.iterdir()):
            if not p.is_dir():
                continue
            path = f"{rel}/{p.name}"
            if group == "smoke":
                if p.name not in SMOKE_KEEP:
                    left.append((path, "smoke benchmark / scratch folder (only AO_demo and AM_merged are kept)"))
                    continue
            elif group in ("poc", "demo") and not STAMP_RE.match(p.name):
                left.append((path, "not a dated live session (dry-run scratch)"))
                continue
            n = decision_rows(p)
            if n == 0:
                left.append((path, "no decisions (empty or test session)" if (p / "decisions.jsonl").exists() else "no decisions.jsonl"))
                continue
            picks.append(Pick(group, p.name, p, label_for(group, p.name, p), n))
    return picks, left


# -------------------------------------------------------------------------------------- copying
def dir_bytes(path: Path) -> int:
    total = 0
    for dp, _dn, fn in os.walk(path):
        for f in fn:
            try:
                total += os.lstat(os.path.join(dp, f)).st_size
            except OSError:
                pass
    return total


def _read_only(path: Path) -> None:
    os.chmod(path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def reencode_jpeg(src: Path, dst: Path, scale: float, quality: int) -> bool:
    """Write `src` as a JPEG of `scale` times the size at `quality` (the file name stays). True when it was re-encoded; an
    unreadable image is copied as it is (False), so the number of shots never changes."""
    import cv2
    img = cv2.imread(str(src), cv2.IMREAD_COLOR)
    ok = False
    if img is not None:
        if abs(scale - 1.0) > 1e-9:
            h, w = img.shape[:2]
            img = cv2.resize(img, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
        ok = bool(cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, int(quality), cv2.IMWRITE_JPEG_OPTIMIZE, 1]))
    if not ok:
        shutil.copyfile(src, dst)
    st = src.stat()
    os.utime(dst, ns=(st.st_atime_ns, st.st_mtime_ns))
    return ok


def pack_run(pick: Pick, dst: Path, plan: Plan) -> dict:
    """Copy one run to `dst` (which must not exist yet). Returns the numbers for the manifest."""
    dst.mkdir(parents=True)
    kept, dropped = [], {}
    for name in KEEP_FILES:
        f = pick.src / name
        if f.is_file():
            shutil.copy2(f, dst / name)
            _read_only(dst / name)
            kept.append(name)
    shots = {"in": 0, "out": 0, "reencoded": 0, "copied_as_is": 0, "dropped": 0}
    folder = pick.src / "shots"
    if folder.is_dir():
        scale, quality = plan.jpeg(pick.group)
        want = plan.keeps_shots(pick.key)
        for f in sorted(folder.iterdir()):
            if not f.is_file():
                dropped[f"shots/{f.name}/"] = dir_bytes(f)
                continue
            if SHOT_RE.match(f.name):
                shots["in"] += 1
                if want:
                    (dst / "shots").mkdir(exist_ok=True)
                    shots["reencoded" if reencode_jpeg(f, dst / "shots" / f.name, scale, quality) else "copied_as_is"] += 1
                    shots["out"] += 1
                    _read_only(dst / "shots" / f.name)
                else:
                    shots["dropped"] += 1
            elif f.name == SHOT_INDEX and want:
                (dst / "shots").mkdir(exist_ok=True)
                shutil.copy2(f, dst / "shots" / f.name)
                _read_only(dst / "shots" / f.name)
            else:
                ext = f.suffix.lower() or f.name
                dropped[ext] = dropped.get(ext, 0) + f.stat().st_size
    (dst / LABEL_FILE).write_text(pick.label + "\n")
    _read_only(dst / LABEL_FILE)
    for f in sorted(pick.src.iterdir()):                  # everything else is left out; the manifest says how much
        if f.name in KEEP_FILES or f.name == "shots":
            continue
        dropped[f.name + "/" if f.is_dir() else f.name] = dir_bytes(f) if f.is_dir() else f.stat().st_size
    return {"kept": kept, "shots": shots, "dropped_bytes": dropped, "before_bytes": dir_bytes(pick.src),
            "after_bytes": dir_bytes(dst)}


def _rmtree(path: Path) -> None:
    def onerror(func, p, _exc):                          # a read-only file on Windows: make it writable and retry
        os.chmod(p, stat.S_IWRITE)
        func(p)
    shutil.rmtree(path, onerror=onerror)


def _inside(a: Path, b: Path) -> bool:
    a, b = a.resolve(), b.resolve()
    return a == b or b in a.parents


def pack_all(runs: Path, out: Path, plan: Plan, picks: list[Pick] | None = None,
             left: list[tuple[str, str]] | None = None) -> dict:
    """One packing pass into `out` (replaced). Refuses an `out` that overlaps `runs` or that is a non-empty folder this script
    did not make (it holds no MANIFEST.json)."""
    if _inside(out, runs) or _inside(runs, out):
        raise ValueError(f"--out {out} and --runs {runs} must not contain each other")
    if out.exists():
        if any(out.iterdir()) and not (out / "MANIFEST.json").exists():
            raise ValueError(f"{out} exists, is not empty and holds no MANIFEST.json: not removing it")
        _rmtree(out)
    if picks is None:
        picks, left = select_runs(runs)
    entries = []
    for pick in picks:
        info = pack_run(pick, out / pick.group / pick.name, plan)
        entries.append({"group": pick.group, "name": pick.name, "label": pick.label, "decisions": pick.decisions, **info})
    totals = {g: {"runs": 0, "before_bytes": 0, "after_bytes": 0, "shots_in": 0, "shots_out": 0} for g, _ in GROUPS}
    for e in entries:
        t = totals[e["group"]]
        t["runs"] += 1
        t["before_bytes"] += e["before_bytes"]
        t["after_bytes"] += e["after_bytes"]
        t["shots_in"] += e["shots"]["in"]
        t["shots_out"] += e["shots"]["out"]
    manifest = {"made": time.strftime("%Y-%m-%dT%H:%M:%S"), "source": "runs/", "plan": plan.name,
                "settings": {"scale": plan.scale, "quality": plan.quality, "vs_pro_scale": plan.vs_scale,
                             "vs_pro_quality": plan.vs_quality, "shots_only": list(plan.shots_only) if plan.shots_only else None},
                "groups": totals, "bytes": sum(dir_bytes(out / g) for g, _ in GROUPS if (out / g).is_dir()),
                "runs": entries, "left_out": [{"path": p, "why": why} for p, why in left or []]}
    out.mkdir(parents=True, exist_ok=True)
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


def plans(scale: float, quality: int) -> list[Plan]:
    """The packing passes in order; the next one runs only if the pack is still over budget."""
    return [Plan(scale, quality, name="default"),
            Plan(scale, quality, LOW_SCALE, LOW_QUALITY, name=f"vs_pro shots at {LOW_SCALE:g} / q{LOW_QUALITY}"),
            Plan(scale, quality, LOW_SCALE, LOW_QUALITY, shots_only=EXAMPLE_RUNS, name="shots for the example games only")]


def pack_within_budget(runs: Path, out: Path, scale: float = DEFAULT_SCALE, quality: int = DEFAULT_QUALITY,
                       budget_mb: float = DEFAULT_BUDGET_MB, log=print) -> dict:
    picks, left = select_runs(runs)
    manifest: dict = {}
    for plan in plans(scale, quality):
        manifest = pack_all(runs, out, plan, picks, left)
        mb = manifest["bytes"] / 1e6
        log(f"pass '{plan.name}': {mb:.1f} MB (budget {budget_mb:g} MB)")
        if mb <= budget_mb:
            break
    return manifest


def report(manifest: dict, log=print) -> None:
    log(f"{'group':8s} {'runs':>4s} {'before MB':>10s} {'after MB':>9s} {'shots in':>9s} {'shots out':>10s}")
    tb = ta = 0
    for g, t in manifest["groups"].items():
        log(f"{g:8s} {t['runs']:4d} {t['before_bytes'] / 1e6:10.1f} {t['after_bytes'] / 1e6:9.1f} {t['shots_in']:9d} {t['shots_out']:10d}")
        tb += t["before_bytes"]
        ta += t["after_bytes"]
    log(f"{'total':8s} {len(manifest['runs']):4d} {tb / 1e6:10.1f} {ta / 1e6:9.1f}   (plan: {manifest['plan']})")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", type=Path, default=ROOT / "runs", help="the runs/ folder to read (never modified)")
    ap.add_argument("--out", type=Path, default=ROOT / "dist" / "docker_build" / "prior_runs", help="where the pack goes (replaced)")
    ap.add_argument("--scale", type=float, default=DEFAULT_SCALE, help="screenshot size factor (default 0.6)")
    ap.add_argument("--quality", type=int, default=DEFAULT_QUALITY, help="screenshot JPEG quality (default 70)")
    ap.add_argument("--budget-mb", type=float, default=DEFAULT_BUDGET_MB, help="pack more tightly when over this many MB (default 250)")
    ap.add_argument("--list", action="store_true", help="only list which runs would be packed and which are left out")
    args = ap.parse_args(argv)
    if not args.runs.is_dir():
        print(f"error: {args.runs} is not a folder", file=sys.stderr)
        return 2
    if args.list:
        picks, left = select_runs(args.runs)
        for p in picks:
            print(f"pack  {p.key:34s} {p.decisions:5d} decisions  {p.label}")
        for path, why in left:
            print(f"leave {path:34s} {why}")
        return 0
    try:
        manifest = pack_within_budget(args.runs, args.out, args.scale, args.quality, args.budget_mb)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    report(manifest)
    print(f"pack: {args.out}  ({manifest['bytes'] / 1e6:.1f} MB, {len(manifest['runs'])} runs; {len(manifest['left_out'])} folders left out, see MANIFEST.json)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
